"""Prompt sensitivity (Section 8, Tables A31-A33): is the trend an artifact of wording?

Three variants -- baseline, emotional, authoritative -- through DeepSeek-R1 on a
proportional stratified subsample of 1,004 papers. Reads the CSVs in
``release/prompt_sensitivity/``, which ``prompt_sensitivity_significance.py``
produces.

The design is paired, so the bootstrap resamples *papers*, not observations.
Treating the variants as independent samples would understate the shared
variance and make a null look significant.

Usage (from the repository root):
    python 07_additional_analyses/analyze_prompt_sensitivity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from shared import (  # noqa: E402
    fmt,
    save_table,
)
import config  # noqa: E402  (on sys.path via shared)

SOURCE = "07_additional_analyses/analyze_prompt_sensitivity.py"
METRICS = ("hr1", "hr2", "hr3", "hr4", "hr5")
VARIANT_LABELS = {"baseline": "Baseline", "emotional": "Emotional",
                  "authoritative": "Authoritative"}
VARIANT_ORDER = ("baseline", "emotional", "authoritative")


def artifacts_dir() -> Path:
    return config.PROMPT_SENSITIVITY_RELEASE_DIR


def _read(name: str) -> pd.DataFrame:
    path = artifacts_dir() / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. It is produced by "
            "07_additional_analyses/prompt_sensitivity_significance.py."
        )
    return pd.read_csv(path)


def table_spearman_prompt() -> dict:
    """Table A31: rho per variant per metric, plus the maximum spread."""
    df = _read("prompt_sensitivity_spearman.csv")
    pivot = df.pivot(index="prompt", columns="metric", values="spearman_rho")

    rows = []
    for variant in VARIANT_ORDER:
        if variant not in pivot.index:
            continue
        rows.append([VARIANT_LABELS[variant]]
                    + [f"${pivot.loc[variant, m]:.3f}$" for m in METRICS])
    spread = [float(pivot[m].max() - pivot[m].min()) for m in METRICS]
    rows.append([r"\textbf{Max Diff}"] + [rf"$\mathbf{{{s:.3f}}}$" for s in spread])

    save_table(
        "spearman_prompt",
        [r"\textbf{Prompt Variant}"] + [rf"\textbf{{HR$_{m[-1]}$}}" for m in METRICS],
        rows, align="lccccc", source=SOURCE, midrule_before={len(rows) - 1},
        values={"spearman": pivot.to_dict(), "max_spread": dict(zip(METRICS, spread, strict=True)),
                "n": int(df["n"].iloc[0])},
    )
    return {"spearman": pivot.to_dict(), "max_spread": dict(zip(METRICS, spread, strict=True))}


def table_prompt_spearman_diffs(metric: str = "hr2") -> dict:
    """Table A32: paired bootstrap tests on the correlation differences."""
    df = _read("prompt_sensitivity_spearman_diffs.csv")
    sub = df[df["metric"] == metric]
    rows, values = [], []
    for _, r in sub.iterrows():
        label = str(r["comparison"]).replace("_minus_", " - ").replace("_", " ").title()
        rows.append([label, fmt(r["rho_diff"], 4),
                     f"$[{r['rho_diff_ci_low']:.4f}, {r['rho_diff_ci_high']:.4f}]$",
                     fmt(r["rho_diff_p_two_sided"], 3)])
        values.append({k: (float(v) if isinstance(v, int | float) else v)
                       for k, v in r.items()})
    save_table(
        "prompt_spearman_diffs",
        [r"\textbf{Comparison}", rf"\textbf{{$\Delta\rho$ for HR$_{metric[-1]}$}}",
         r"\textbf{95\% bootstrap CI}", r"\textbf{$p$}"],
        rows, align="lrrr", source=SOURCE,
        values={"metric": metric, "comparisons": values},
    )
    return {"comparisons": values}


def table_mean_prompt() -> dict:
    """Table A33: mean metric values and their shift from baseline."""
    means = _read("prompt_sensitivity_mean_metrics.csv").set_index("prompt")
    baseline = means.loc["baseline"]

    # A \multicolumn spanning the whole row is ONE cell, not one cell plus
    # padding: padding it would declare 11 columns in a 6-column tabular and
    # fail to compile.
    rows = [[r"\multicolumn{6}{@{}l}{\textit{Mean Values}}"]]
    for variant in sorted(means.index):
        rows.append([VARIANT_LABELS.get(variant, variant)]
                    + [f"${means.loc[variant, f'{m}_mean']:.4f}$" for m in METRICS])
    rows.append([r"\multicolumn{6}{@{}l}{\textit{Absolute Difference from Baseline}}"])
    shifts = {}
    for variant in sorted(v for v in means.index if v != "baseline"):
        deltas = [float(means.loc[variant, f"{m}_mean"] - baseline[f"{m}_mean"]) for m in METRICS]
        shifts[variant] = dict(zip(METRICS, deltas, strict=True))
        rows.append([VARIANT_LABELS.get(variant, variant)]
                    + [f"${d:+.4f}$" for d in deltas])

    max_shift = max(abs(d) for v in shifts.values() for d in v.values())
    save_table(
        "mean_prompt",
        [r"\textbf{Prompt Variant}"] + [rf"\textbf{{HR$_{m[-1]}$}}" for m in METRICS],
        rows, align="lccccc", source=SOURCE, midrule_before={len(means.index) + 1},
        values={"means": means.to_dict(), "shifts_from_baseline": shifts,
                "max_abs_shift": max_shift},
    )
    return {"shifts_from_baseline": shifts, "max_abs_shift": max_shift}


def run_all() -> dict:
    return {
        "spearman": table_spearman_prompt(),
        "diffs": table_prompt_spearman_diffs(),
        "means": table_mean_prompt(),
    }


if __name__ == "__main__":
    run_all()
