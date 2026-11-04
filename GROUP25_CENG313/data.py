"""Data loading helpers that work locally, on Google Colab and on Kaggle."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd

COMPETITION_SLUG = "wsdm-cup-multilingual-chatbot-arena"
TARGET_CLASSES = ("model_a", "model_b")
TEXT_COLUMNS = ("prompt", "response_a", "response_b")

# Locations searched (in order) when no explicit data directory is given.
_CANDIDATE_DIRS: Tuple[str, ...] = (
    "data",
    "../data",
    "overview",
    "../overview",
    f"/kaggle/input/{COMPETITION_SLUG}",
    "/content/data",
    f"/content/drive/MyDrive/{COMPETITION_SLUG}",
)


def find_data_dir(candidates: Optional[Iterable[str]] = None) -> Path:
    """Return the first directory that contains ``train.parquet``."""
    for c in candidates or _CANDIDATE_DIRS:
        p = Path(c)
        if (p / "train.parquet").exists():
            return p
    raise FileNotFoundError(
        "train.parquet not found. Put the competition files into ./data "
        f"(e.g. `kaggle competitions download -c {COMPETITION_SLUG}`) or pass data_dir."
    )


def load_competition_data(
    data_dir: Optional[str | Path] = None,
    sample_frac: Optional[float] = None,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load ``train.parquet`` and ``test.parquet``.

    Parameters
    ----------
    data_dir : directory containing the parquet files (auto-detected if None).
    sample_frac : optionally subsample the training set (useful for quick experiments).
    """
    data_dir = Path(data_dir) if data_dir is not None else find_data_dir()
    train = pd.read_parquet(data_dir / "train.parquet")
    test = pd.read_parquet(data_dir / "test.parquet")
    for df in (train, test):
        for col in TEXT_COLUMNS:
            if col in df.columns:
                df[col] = df[col].fillna("").astype(str)
    if sample_frac is not None and sample_frac < 1.0:
        train = train.sample(frac=sample_frac, random_state=random_state).reset_index(drop=True)
    return train, test


def encode_target(winner: pd.Series) -> np.ndarray:
    """``model_a`` -> 0, ``model_b`` -> 1."""
    return (winner.astype(str) == "model_b").astype(int).to_numpy()


def decode_target(pred: np.ndarray) -> np.ndarray:
    """0/1 (or probability of model_b) -> class labels."""
    pred = np.asarray(pred)
    if pred.dtype.kind == "f":
        pred = (pred >= 0.5).astype(int)
    return np.where(pred == 1, TARGET_CLASSES[1], TARGET_CLASSES[0])


def swap_ab(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with responses A and B swapped (and the label flipped).

    Used for test-time / train-time augmentation against position bias.
    """
    out = df.copy()
    out["response_a"], out["response_b"] = df["response_b"].values, df["response_a"].values
    if {"model_a", "model_b"} <= set(df.columns):
        out["model_a"], out["model_b"] = df["model_b"].values, df["model_a"].values
    if "winner" in df.columns:
        out["winner"] = np.where(df["winner"] == "model_a", "model_b", "model_a")
    return out


def make_submission(ids, pred, path: str | Path = "submission.csv") -> pd.DataFrame:
    """Write ``submission.csv`` in the competition format (id,winner)."""
    sub = pd.DataFrame({"id": ids, "winner": decode_target(pred)})
    sub.to_csv(path, index=False)
    return sub
