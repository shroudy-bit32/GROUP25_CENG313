"""Plotting / summary helpers used by the EDA notebook."""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {"model_a": "#2a78c8", "model_b": "#e0703a"}


def basic_overview(df: pd.DataFrame) -> pd.DataFrame:
    """Column dtype, missing values and number of unique values."""
    return pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "n_missing": df.isna().sum(),
            "n_unique": df.nunique(),
        }
    )


def plot_target_distribution(df: pd.DataFrame, ax: Optional[plt.Axes] = None):
    ax = ax or plt.gca()
    vc = df["winner"].value_counts()
    ax.bar(vc.index, vc.values, color=[COLORS.get(k, "gray") for k in vc.index])
    for i, v in enumerate(vc.values):
        ax.text(i, v, f"{v:,}\n({v / vc.sum():.1%})", ha="center", va="bottom")
    ax.set_title("Target distribution (winner)")
    ax.set_ylabel("count")
    ax.margins(y=0.15)
    return ax


def plot_top_languages(df: pd.DataFrame, top: int = 20, ax: Optional[plt.Axes] = None):
    ax = ax or plt.gca()
    vc = df["language"].value_counts().head(top)[::-1]
    ax.barh(vc.index, vc.values, color="#5b8c5a")
    ax.set_title(f"Top {top} languages")
    ax.set_xlabel("count")
    return ax


def plot_length_distributions(df: pd.DataFrame, ax: Optional[plt.Axes] = None):
    ax = ax or plt.gca()
    bins = np.logspace(0, 5, 60)
    for col, c in [("prompt", "#777777"), ("response_a", COLORS["model_a"]), ("response_b", COLORS["model_b"])]:
        ax.hist(df[col].str.len().clip(lower=1), bins=bins, alpha=0.5, label=col, color=c)
    ax.set_xscale("log")
    ax.set_title("Character length distribution (log scale)")
    ax.set_xlabel("characters")
    ax.legend()
    return ax


def verbosity_bias_table(df: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """Win-rate of response A as a function of the length ratio len(a)/(len(a)+len(b))."""
    la = df["response_a"].str.len()
    lb = df["response_b"].str.len()
    ratio = la / (la + lb).replace(0, np.nan)
    d = pd.DataFrame({"ratio_bin": pd.qcut(ratio, n_bins, duplicates="drop"), "a_wins": df["winner"] == "model_a"})
    return d.groupby("ratio_bin", observed=True)["a_wins"].agg(["mean", "size"]).rename(
        columns={"mean": "P(A wins)", "size": "n"}
    )


def plot_verbosity_bias(df: pd.DataFrame, n_bins: int = 10, ax: Optional[plt.Axes] = None):
    ax = ax or plt.gca()
    t = verbosity_bias_table(df, n_bins)
    ax.plot(range(len(t)), t["P(A wins)"].values, marker="o", color=COLORS["model_a"])
    ax.axhline(0.5, ls="--", color="gray")
    ax.set_xticks(range(len(t)))
    ax.set_xticklabels([f"{iv.mid:.2f}" for iv in t.index], rotation=45)
    ax.set_xlabel("len(A) / (len(A)+len(B))  (decile midpoint)")
    ax.set_ylabel("P(A wins)")
    ax.set_title("Verbosity bias: longer answer tends to win")
    return ax


def model_win_rates(df: pd.DataFrame, min_battles: int = 300) -> pd.DataFrame:
    """Per-LLM win rate computed over both A and B positions."""
    a = pd.DataFrame({"model": df["model_a"], "win": df["winner"] == "model_a"})
    b = pd.DataFrame({"model": df["model_b"], "win": df["winner"] == "model_b"})
    m = pd.concat([a, b]).groupby("model")["win"].agg(["mean", "size"])
    m.columns = ["win_rate", "battles"]
    return m[m.battles >= min_battles].sort_values("win_rate", ascending=False)


def feature_win_rate(df: pd.DataFrame, feature_a: pd.Series, feature_b: pd.Series) -> pd.DataFrame:
    """P(A wins) when A has more / equal / less of a feature than B."""
    diff = np.sign(feature_a.values - feature_b.values)
    lab = pd.Series(diff).map({1: "A > B", 0: "A = B", -1: "A < B"})
    return (
        pd.DataFrame({"cmp": lab, "a_wins": (df["winner"] == "model_a").values})
        .groupby("cmp")["a_wins"].agg(["mean", "size"])
        .rename(columns={"mean": "P(A wins)", "size": "n"})
    )
