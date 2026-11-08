"""Cross-validation utilities."""

from __future__ import annotations

import time
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold


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
