"""Title length (Section 8, Table A29): does a longer title make attribution harder?

Barely, and mostly not independently of citation count. Two controls are
computed because they fail differently: partial correlation removes the linear
rank component shared with citations, while within-bin Fisher-z pooling
correlates inside each of the 13 bins and combines them, surviving a non-linear
citation relationship that the partial correlation does not.

Usage (from the repository root):
    python 07_additional_analyses/analyze_title_length.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from scipy import stats

from shared import (  # noqa: E402
    BIN_LABELS,
    MODEL_LABELS,
    load_responses,
    save_table,
)

SOURCE = "07_additional_analyses/analyze_title_length.py"


def partial_spearman(x, y, control) -> tuple[float, float]:
    """Spearman correlation of x and y with ``control`` partialled out.

    Implemented as the Pearson correlation of the residuals of the rank
    transforms, which is the standard rank-based partial correlation.
    """
    rx, ry, rc = (stats.rankdata(v) for v in (x, y, control))
    resid = []
    for r in (rx, ry):
        slope, intercept = np.polyfit(rc, r, 1)
        resid.append(r - (slope * rc + intercept))
    r, p = stats.pearsonr(resid[0], resid[1])
    return float(r), float(p)


def fisher_pooled_within_bin(df, metric: str = "hr2") -> dict:
    """Per-bin Spearman rho, pooled with a Fisher z transform weighted by n-3."""
    zs, weights, per_bin = [], [], []
    for label in BIN_LABELS:
        sub = df[df["citation_bin"] == label][["title_length", metric]].dropna()
        if len(sub) < 10:
            continue
        rho, p = stats.spearmanr(sub["title_length"], sub[metric])
        if np.isnan(rho) or abs(rho) >= 1:
            continue
        zs.append(np.arctanh(rho))
        weights.append(len(sub) - 3)
        per_bin.append({"citation_bin": label, "rho": float(rho), "p": float(p),
                        "n": int(len(sub))})
    if not zs:
        return {"pooled_rho": float("nan"), "per_bin": per_bin}
    z = float(np.average(zs, weights=weights))
    se = float(1 / np.sqrt(sum(weights)))
    p = float(2 * (1 - stats.norm.cdf(abs(z / se))))
    return {"pooled_rho": float(np.tanh(z)), "pooled_p": p, "per_bin": per_bin}


def table_title_length(metric: str = "hr2") -> dict:
    values, rows = {}, []
    # The paper orders this table GPT-4o, Claude, DeepSeek.
    for model in ("gpt-4o", "claude-sonnet-4.5", "deepseek-r1"):
        df = load_responses(model)
        sub = df[["title_length", metric, "citations"]].dropna()
        overall_rho, overall_p = stats.spearmanr(sub["title_length"], sub[metric])
        adj_rho, adj_p = partial_spearman(sub["title_length"], sub[metric], sub["citations"])
        pooled = fisher_pooled_within_bin(df, metric)
        values[model] = {
            "overall": {"rho": float(overall_rho), "p": float(overall_p), "n": int(len(sub))},
            "citation_adjusted_partial": {"rho": adj_rho, "p": adj_p},
            "within_bin_fisher": pooled,
        }
        rows.append([MODEL_LABELS[model], f"{overall_rho:+.3f}", f"{adj_rho:+.3f}"])

    save_table(
        "title_length",
        [r"\textbf{Model}", r"\textbf{Overall}", r"\textbf{Citation-Adjusted}"],
        rows, align="lcc", source=SOURCE,
        values={"metric": metric, "by_model": values},
    )
    return values


def run_all() -> dict:
    return table_title_length()


if __name__ == "__main__":
    run_all()
