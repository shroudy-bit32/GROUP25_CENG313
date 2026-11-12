"""Baseline models and their training / evaluation.

Every model follows a tiny common interface:

    model.fit(df, y)            # df: DataFrame with prompt/response_a/response_b
    model.predict_proba(df)     # -> P(winner == model_b), shape (n,)
    model.predict(df)           # -> 0 (model_a) / 1 (model_b)
"""

from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import HandcraftedFeatureExtractor, hashed_char_counts


class BaseModel:
    name = "base"

    def fit(self, df: pd.DataFrame, y: np.ndarray):
        raise NotImplementedError

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(df) >= 0.5).astype(int)


class MajorityClassBaseline(BaseModel):
    """Always predicts the most frequent class of the training set."""

    name = "Majority class"

    def fit(self, df, y):
        self.p_ = float(np.mean(y))
        return self

    def predict_proba(self, df):
        return np.full(len(df), self.p_)


class LengthHeuristicBaseline(BaseModel):
    """Rule: the longer response wins (exploits verbosity bias). No learning."""

    name = "Longer response wins"

    def fit(self, df, y):
        return self

    def predict_proba(self, df):
        la = df["response_a"].str.len().to_numpy()
        lb = df["response_b"].str.len().to_numpy()
        return np.where(lb > la, 1.0, np.where(lb < la, 0.0, 0.5))


class FeatureModelBaseline(BaseModel):
    """Handcrafted features + a tabular model (``logreg`` or ``lgbm``).

    ``precomputed``: optional feature DataFrame (from :class:`HandcraftedFeatureExtractor`)
    indexed by the competition ``id``; avoids recomputing features in every CV fold.
    """

    def __init__(
        self,
        model: str = "lgbm",
        params: Optional[dict] = None,
        random_state: int = 42,
        precomputed: Optional[pd.DataFrame] = None,
    ):
        self.model = model
        self.params = params or {}
        self.random_state = random_state
        self.precomputed = precomputed
        self.extractor = HandcraftedFeatureExtractor()
        self.name = {"lgbm": "Handcrafted + LightGBM", "logreg": "Handcrafted + LogReg"}[model]

    def _build(self):
        if self.model == "logreg":
            return make_pipeline(
                StandardScaler(),
                LogisticRegression(C=self.params.get("C", 1.0), max_iter=2000),
            )
        import lightgbm as lgb

        default = dict(
            n_estimators=600,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=50,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=self.random_state,
            verbose=-1,
        )
        default.update(self.params)
        return lgb.LGBMClassifier(**default)

    def features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Reuse features computed once for the whole training set (saves time in CV).
        # ``precomputed`` is indexed by the competition ``id`` (never by row position,
        # otherwise test rows 0,1,2 would silently get train rows' features).
        if self.precomputed is not None:
            ids = pd.Index(df["id"].astype(str))
            if ids.isin(self.precomputed.index).all():
                return self.precomputed.loc[ids]
        return self.extractor.transform(df)

    def fit(self, df, y):
        X = self.features(df)
        self.clf_ = self._build()
        self.clf_.fit(X, y)
        self.feature_names_ = list(X.columns)
        return self

    def predict_proba(self, df):
        return self.clf_.predict_proba(self.features(df))[:, 1]

    def feature_importance(self) -> pd.Series:
        if self.model == "lgbm":
            imp = self.clf_.booster_.feature_importance(importance_type="gain")
        else:
            imp = np.abs(self.clf_[-1].coef_[0])
        return pd.Series(imp, index=self.feature_names_).sort_values(ascending=False)


class TfidfDiffBaseline(BaseModel):
    """Character n-gram TF-IDF (language agnostic) + Logistic Regression.

    Responses are turned into hashed char n-gram counts, re-weighted with TF-IDF
    (IDF fitted on the training fold only) and the model sees ``tfidf(A) - tfidf(B)``,
    i.e. which content makes a response *more* preferable than the other one.
    Training uses A/B swap augmentation ((A-B, y) and (B-A, 1-y)) so the model is symmetric.

    ``precomputed`` can be ``(counts_a, counts_b, ids)`` from :func:`hashed_char_counts`
    for the full training set; rows are then looked up by the ``id`` column (fast CV).
    """

    name = "Char TF-IDF diff + LogReg"

    def __init__(self, C: float = 1.0, ngram_range=(2, 4), max_chars: int = 2000, precomputed=None):
        self.C = C
        self.ngram_range = ngram_range
        self.max_chars = max_chars
        self.precomputed = precomputed

    def _counts(self, df):
        if self.precomputed is not None:
            ca, cb, ids = self.precomputed
            pos = pd.Index(pd.Series(ids).astype(str)).get_indexer(df["id"].astype(str))
            if (pos >= 0).all():
                return ca[pos], cb[pos]
        kw = dict(ngram_range=self.ngram_range, max_chars=self.max_chars)
        return hashed_char_counts(df["response_a"], **kw), hashed_char_counts(df["response_b"], **kw)

    def _matrix(self, df):
        ca, cb = self._counts(df)
        return (self.tfidf_.transform(ca) - self.tfidf_.transform(cb)).tocsr()

    def fit(self, df, y):
        ca, cb = self._counts(df)
        self.tfidf_ = TfidfTransformer(sublinear_tf=True).fit(sp.vstack([ca, cb]))
        X = (self.tfidf_.transform(ca) - self.tfidf_.transform(cb)).tocsr()
        X_aug = sp.vstack([X, -X]).tocsr()
        y_aug = np.concatenate([y, 1 - y])
        self.clf_ = LogisticRegression(C=self.C, max_iter=1000, solver="lbfgs")
        self.clf_.fit(X_aug, y_aug)
        return self

    def predict_proba(self, df):
        return self.clf_.predict_proba(self._matrix(df))[:, 1]


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------

def holdout_evaluate(model, train_df: pd.DataFrame, y_train: np.ndarray, valid_df: pd.DataFrame, y_valid: np.ndarray) -> Dict:
    """Fit on the training part, report accuracy (competition metric) on the validation part."""
    t0 = time.time()
    model.fit(train_df, y_train)
    p = model.predict_proba(valid_df)
    return {
        "model": getattr(model, "name", type(model).__name__),
        "valid_accuracy": float(accuracy_score(y_valid, (p >= 0.5).astype(int))),
        "proba": p,
        "fitted": model,
        "time_s": time.time() - t0,
    }


def cross_validate(
    model_factory: Callable[[], object],
    df: pd.DataFrame,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
    verbose: bool = True,
) -> Dict:
    """Stratified K-fold CV. Returns fold scores and out-of-fold probabilities.

    ``model_factory`` must return a *fresh* model each call, e.g. ``lambda: TfidfDiffBaseline()``.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof = np.zeros(len(df), dtype=np.float64)
    accs: List[float] = []
    name = None
    t0 = time.time()
    for fold, (tr_idx, va_idx) in enumerate(skf.split(df, y)):
        model = model_factory()
        name = getattr(model, "name", type(model).__name__)
        tr, va = df.iloc[tr_idx], df.iloc[va_idx]
        model.fit(tr, y[tr_idx])
        p = model.predict_proba(va)
        oof[va_idx] = p
        acc = accuracy_score(y[va_idx], (p >= 0.5).astype(int))
        accs.append(acc)
        if verbose:
            print(f"[{name}] fold {fold + 1}/{n_splits}: acc={acc:.4f}")
    res = {
        "model": name,
        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs)),
        "fold_acc": accs,
        "oof": oof,
        "time_s": time.time() - t0,
    }
    pc = np.clip(oof, 1e-6, 1 - 1e-6)
    res["logloss"] = float(log_loss(y, pc)) if len(np.unique(oof)) > 1 else float("nan")
    res["auc"] = float(roc_auc_score(y, oof)) if len(np.unique(oof)) > 1 else 0.5
    if verbose:
        print(f"[{name}] CV acc = {res['acc_mean']:.4f} ± {res['acc_std']:.4f}  ({res['time_s']:.0f}s)")
    return res


def summarize_results(results: List[Dict]) -> pd.DataFrame:
    rows = [
        {
            "model": r["model"],
            "cv_accuracy": r["acc_mean"],
            "cv_std": r["acc_std"],
            "auc": r["auc"],
            "logloss": r["logloss"],
            "time_s": r["time_s"],
        }
        for r in results
    ]
    return pd.DataFrame(rows).sort_values("cv_accuracy", ascending=False).reset_index(drop=True)


def accuracy_by_group(y: np.ndarray, proba: np.ndarray, groups: pd.Series, min_count: int = 200) -> pd.DataFrame:
    """Accuracy per group (e.g. per language) for error analysis."""
    d = pd.DataFrame({"y": y, "pred": (proba >= 0.5).astype(int), "g": groups.values})
    out = d.groupby("g").apply(lambda x: pd.Series({"n": len(x), "accuracy": (x.y == x.pred).mean()}))
    return out[out.n >= min_count].sort_values("n", ascending=False)
