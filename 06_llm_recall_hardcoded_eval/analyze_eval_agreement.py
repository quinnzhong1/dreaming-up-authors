"""Appendix A4.4: hard-coded matcher versus LLM-as-judge.

Both evaluators scored the same 9,108 GPT-4o answers, so the comparison is
paired per paper. One paper is excluded: the judge returned
``"None, None, None"`` after five retries. That single row is the difference
between N = 9,108 and the N = 9,107 quoted in the regression appendix.

The judge counts are the ``llm_eval_m/h/u`` columns of
``release/responses/gpt-4o.csv.gz``, produced by module 04
(``04_gpt4o_recall_llm_selfeval/generate_and_self_evaluate.py``).

Produces ``eval_agreement.tex`` (Table A22) and ``variance_comparison.tex``
(Table A23), plus ``fig_eval_agreement.pdf``.

Usage (from the repository root):
    python 06_llm_recall_hardcoded_eval/analyze_eval_agreement.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from scipy import stats

from common import (  # noqa: E402
    apply_style,
    fmt,
    load_responses,
    save_figure,
    save_table,
)


SOURCE = "06_llm_recall_hardcoded_eval/analyze_eval_agreement.py"
METRICS = ("hr1", "hr2", "hr3")
AGREEMENT_THRESHOLD = 0.1


def agreement_stats() -> dict:
    """Paired agreement between the two evaluators, per metric."""
    df = load_responses("gpt-4o")
    out = {"n_total": int(len(df)), "metrics": {}}
    for metric in METRICS:
        paired = df[[metric, f"judge_{metric}"]].dropna()
        code = paired[metric].to_numpy()
        judge = paired[f"judge_{metric}"].to_numpy()
        diff = judge - code
        r, p = stats.pearsonr(code, judge)
        out["metrics"][metric] = {
            "n": int(len(paired)),
            "pearson_r": float(r),
            "pearson_p": float(p),
            "rmse": float(np.sqrt(np.mean(diff**2))),
            "mae": float(np.mean(np.abs(diff))),
            "mean_signed_diff": float(np.mean(diff)),
            "agreement_within_0.1": float(np.mean(np.abs(diff) < AGREEMENT_THRESHOLD)),
            "var_judge": float(np.var(judge, ddof=1)),
            "var_code": float(np.var(code, ddof=1)),
        }
    out["n_judge_failures"] = out["n_total"] - out["metrics"]["hr2"]["n"]
    return out


def table_eval_agreement() -> dict:
    stats_ = agreement_stats()
    rows = []
    for metric in METRICS:
        s = stats_["metrics"][metric]
        rows.append([
            rf"HR$_{metric[-1]}$",
            fmt(s["pearson_r"]),
            fmt(s["rmse"]),
            fmt(s["mae"]),
            f"{100 * s['agreement_within_0.1']:.1f}\\%",
        ])
    save_table(
        "eval_agreement",
        [r"\textbf{Metric}", r"\textbf{Pearson} $r$", r"\textbf{RMSE}", r"\textbf{MAE}",
         r"\textbf{Agreement $<$ 0.1}"],
        rows, align="lcccc", source=SOURCE, values=stats_,
    )
    return stats_


def table_variance_comparison(stats_: dict | None = None) -> None:
    stats_ = stats_ or agreement_stats()
    rows = []
    for metric in METRICS:
        s = stats_["metrics"][metric]
        ratio = s["var_judge"] / s["var_code"] if s["var_code"] else float("nan")
        rows.append([rf"HR$_{metric[-1]}$", fmt(s["var_judge"], 4),
                     fmt(s["var_code"], 4), fmt(ratio, 2)])
    save_table(
        "variance_comparison",
        [r"\textbf{Metric}", r"\textbf{Var(LLM)}", r"\textbf{Var(Code)}", r"\textbf{Ratio}"],
        rows, align="lrrr", source=SOURCE,
    )


def figure_eval_agreement() -> None:
    """Judge HR2 against matcher HR2, one point per paper."""
    apply_style()
    import matplotlib.pyplot as plt

    df = load_responses("gpt-4o")[["hr2", "judge_hr2"]].dropna()
    fig, ax = plt.subplots(figsize=(4.0, 4.0))
    ax.scatter(df["hr2"], df["judge_hr2"], s=3, alpha=0.12, edgecolors="none", color="#1f77b4")
    ax.plot([0, 1], [0, 1], color="0.4", linestyle="--", linewidth=1)
    r, _ = stats.pearsonr(df["hr2"], df["judge_hr2"])
    ax.set_xlabel(r"Hard-coded HR$_2$")
    ax.set_ylabel(r"LLM self-evaluation HR$_2$")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"$r = {r:.3f}$, $n = {len(df):,}$", fontsize=10)
    fig.tight_layout()
    save_figure(fig, "fig_eval_agreement",
                {"pearson_r": float(r), "n": int(len(df))}, source=SOURCE)


def run_all() -> None:
    stats_ = table_eval_agreement()
    table_variance_comparison(stats_)
    figure_eval_agreement()


if __name__ == "__main__":
    run_all()
