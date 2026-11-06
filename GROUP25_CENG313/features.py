"""Language-agnostic handcrafted features for a (prompt, response_a, response_b) triple.

The features are deliberately simple and script-independent (character counts,
markdown structure, overlap with the prompt, ...) because the data is multilingual.
For each response-level feature ``f`` we produce ``f_a``, ``f_b``, ``f_diff`` (a - b)
and ``f_ratio`` (a / (a + b)) so that models can learn relative preferences.
"""

from __future__ import annotations

import re
from typing import Dict, List

import numpy as np
import pandas as pd

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
