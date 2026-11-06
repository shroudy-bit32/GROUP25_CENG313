"""Baseline models. Every model follows a tiny common interface:

    model.fit(df, y)            # df: DataFrame with prompt/response_a/response_b
    model.predict_proba(df)     # -> P(winner == model_b), shape (n,)
    model.predict(df)           # -> 0 (model_a) / 1 (model_b)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import HandcraftedFeatureExtractor


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


def hashed_char_counts(
    texts: pd.Series,
    ngram_range=(2, 4),
    n_features: int = 2**20,
    max_chars: int = 2000,
    n_jobs: int = -1,
    chunk: int = 2000,
) -> sp.csr_matrix:
    """Character n-gram counts via the hashing trick (stateless -> parallel & cacheable).

    ``char_wb`` n-grams work for every script (Latin, Cyrillic, CJK, ...), which matters
    because the data covers 100+ languages.
    """
    from joblib import Parallel, delayed

    hv = HashingVectorizer(
        analyzer="char_wb",
        ngram_range=ngram_range,
        n_features=n_features,
        alternate_sign=False,
        norm=None,
        dtype=np.float32,
    )
    t = texts.str.slice(0, max_chars).tolist()
    parts = Parallel(n_jobs=n_jobs)(delayed(hv.transform)(t[i : i + chunk]) for i in range(0, len(t), chunk))
    return sp.vstack(parts).tocsr()


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
