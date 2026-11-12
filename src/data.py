"""Data loading and preprocessing.

* loading the competition parquet files (locally, on Colab or on Kaggle)
* target encoding (model_a -> 0, model_b -> 1) and A/B swapping
* language-agnostic handcrafted features for a (prompt, response_a, response_b) triple
* hashed character n-gram counts (input of the TF-IDF model)
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer

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


def train_valid_split(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Stratified train / validation split (80 / 20 by default)."""
    from sklearn.model_selection import train_test_split

    return train_test_split(df, test_size=test_size, random_state=random_state, stratify=df["winner"])


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


# ----------------------------------------------------------------------------
# Handcrafted features
# ----------------------------------------------------------------------------
# For each response-level statistic ``f`` we produce ``f_a``, ``f_b``, ``f_diff`` (a - b)
# and ``f_ratio`` (a / (a + b)) so that models can learn *relative* preferences.
# The statistics are script-independent because the data covers 100+ languages.

_RE_WORD = re.compile(r"\w+", re.UNICODE)
_RE_HEADER = re.compile(r"^\s{0,3}#{1,6}\s", re.MULTILINE)
_RE_BULLET = re.compile(r"^\s*[-*+•]\s+", re.MULTILINE)
_RE_NUMBERED = re.compile(r"^\s*\d+[.)]\s+", re.MULTILINE)
_RE_BOLD = re.compile(r"\*\*[^*]+\*\*|__[^_]+__")
_RE_CODE_BLOCK = re.compile(r"```")
_RE_INLINE_CODE = re.compile(r"`[^`\n]+`")
_RE_URL = re.compile(r"https?://\S+")
_RE_DIGIT = re.compile(r"\d")
_RE_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$", re.MULTILINE)
_RE_LATEX = re.compile(r"\$[^$]+\$|\\\(|\\\[")
_RE_EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿]")

# Common refusal / apology phrases in a few frequent languages of the dataset.
_REFUSAL_PATTERNS = [
    r"\bI(?:'m| am) sorry\b", r"\bI can(?:'t|not)\b", r"\bas an AI\b", r"\bI(?:'m| am) unable\b",
    r"\blo siento\b", r"\bno puedo\b",                      # es
    r"\bdésolé\b", r"\bje ne peux pas\b",                    # fr
    r"\bes tut mir leid\b", r"\bich kann nicht\b",           # de
    r"извините", r"не могу", r"к сожалению",                 # ru
    r"抱歉", r"对不起", r"无法", r"申し訳",                    # zh / ja
    r"üzgünüm", r"yapamam",                                  # tr
    r"죄송",                                                 # ko
]
_RE_REFUSAL = re.compile("|".join(_REFUSAL_PATTERNS), re.IGNORECASE)


def _text_stats(text: str) -> Dict[str, float]:
    words = _RE_WORD.findall(text)
    n_chars = len(text)
    n_words = len(words)
    lines = text.split("\n")
    return {
        "n_chars": n_chars,
        "n_words": n_words,
        "n_lines": len(lines),
        "n_empty_lines": sum(1 for l in lines if not l.strip()),
        "n_unique_words": len(set(w.lower() for w in words)),
        "avg_word_len": (sum(len(w) for w in words) / n_words) if n_words else 0.0,
        "n_headers": len(_RE_HEADER.findall(text)),
        "n_bullets": len(_RE_BULLET.findall(text)),
        "n_numbered": len(_RE_NUMBERED.findall(text)),
        "n_bold": len(_RE_BOLD.findall(text)),
        "n_code_blocks": len(_RE_CODE_BLOCK.findall(text)) // 2,
        "n_inline_code": len(_RE_INLINE_CODE.findall(text)),
        "n_table_rows": len(_RE_TABLE_ROW.findall(text)),
        "n_urls": len(_RE_URL.findall(text)),
        "n_latex": len(_RE_LATEX.findall(text)),
        "n_emoji": len(_RE_EMOJI.findall(text)),
        "digit_ratio": (len(_RE_DIGIT.findall(text)) / n_chars) if n_chars else 0.0,
        "n_question": text.count("?") + text.count("？"),
        "n_exclaim": text.count("!") + text.count("！"),
        "is_refusal": float(bool(_RE_REFUSAL.search(text[:300]))),
        "is_empty": float(n_chars == 0),
    }


def _word_set(text: str) -> set:
    return set(w.lower() for w in _RE_WORD.findall(text))


def _char_ngrams(text: str, n: int = 3) -> set:
    t = text.lower()
    return {t[i : i + n] for i in range(max(len(t) - n + 1, 0))}


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


class HandcraftedFeatureExtractor:
    """sklearn-style transformer producing a numeric feature DataFrame.

    Example
    -------
    >>> fe = HandcraftedFeatureExtractor()
    >>> X = fe.fit_transform(train_df)
    """

    def __init__(self, include_prompt: bool = True, include_overlap: bool = True):
        self.include_prompt = include_prompt
        self.include_overlap = include_overlap
        self.feature_names_: List[str] = []

    def fit(self, df: pd.DataFrame, y=None):
        return self

    def _row_features(self, prompt: str, ra: str, rb: str) -> Dict[str, float]:
        sa, sb = _text_stats(ra), _text_stats(rb)
        feats: Dict[str, float] = {}
        for k in sa:
            a, b = sa[k], sb[k]
            feats[f"{k}_a"] = a
            feats[f"{k}_b"] = b
            feats[f"{k}_diff"] = a - b
            feats[f"{k}_ratio"] = a / (a + b) if (a + b) else 0.5

        if self.include_prompt:
            feats["prompt_n_chars"] = len(prompt)
            feats["prompt_n_words"] = len(_RE_WORD.findall(prompt))
            feats["prompt_n_lines"] = prompt.count("\n") + 1

        if self.include_overlap:
            pw, aw, bw = _word_set(prompt), _word_set(ra), _word_set(rb)
            pc, ac, bc = _char_ngrams(prompt[:2000]), _char_ngrams(ra[:4000]), _char_ngrams(rb[:4000])
            feats["prompt_overlap_word_a"] = _jaccard(pw, aw)
            feats["prompt_overlap_word_b"] = _jaccard(pw, bw)
            feats["prompt_overlap_word_diff"] = feats["prompt_overlap_word_a"] - feats["prompt_overlap_word_b"]
            feats["prompt_overlap_char_a"] = _jaccard(pc, ac)
            feats["prompt_overlap_char_b"] = _jaccard(pc, bc)
            feats["prompt_overlap_char_diff"] = feats["prompt_overlap_char_a"] - feats["prompt_overlap_char_b"]
            feats["ab_similarity_char"] = _jaccard(ac, bc)
            feats["ab_identical"] = float(ra.strip() == rb.strip())
        return feats

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        rows = [
            self._row_features(p, a, b)
            for p, a, b in zip(df["prompt"].values, df["response_a"].values, df["response_b"].values)
        ]
        out = pd.DataFrame(rows, index=df.index).astype(np.float32)
        self.feature_names_ = list(out.columns)
        return out

    def fit_transform(self, df: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(df, y).transform(df)


# ----------------------------------------------------------------------------
# Character n-grams
# ----------------------------------------------------------------------------

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
