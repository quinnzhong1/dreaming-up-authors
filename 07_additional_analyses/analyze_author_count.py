"""Author count (Section 8, Tables A26-A27): why more authors means a *lower* HR2.

The correlation between author count and HR2 is negative, which looks
backwards until you look at output length. Models emit roughly the same
number of names regardless of the truth, so a single-author paper collects
about 2.3 fabricated names against 1 real one, while a five-author paper
collects about 4.1 against 5. The denominator grows faster than the error.

Produces ``author_output.tex`` (Table A26) and ``field_author.tex`` (A27).

Usage (from the repository root):
    python 07_additional_analyses/analyze_author_count.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from scipy import stats

from shared import (  # noqa: E402
    FIELD_KEYS,
    OUT_DIR,
    RUNAWAY_GENERATION_THRESHOLD,
    field_label,
    fmt,
    load_all_responses,
    save_table,
    write_json,
)

SOURCE = "07_additional_analyses/analyze_author_count.py"

# Ground-truth author-count groups used by Table A26.
GROUPS = [("1", 1, 1), ("2--3", 2, 3), ("4--5", 4, 5), ("6--10", 6, 10), ("11+", 11, None)]


def _without_runaway_generations(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop answers that are runaway generations, not attribution attempts.

    The longest answer in the released data lists 4,096 names. The corpus caps
    ground truth at 20 authors, so nothing above the configured threshold is
    a serious attempt, and one such row moves a group mean by several names.
    Rate-based analyses keep these rows; only the raw-count analyses in this
    module drop them. Returns the frame and the number dropped.
    """
    threshold = RUNAWAY_GENERATION_THRESHOLD
    keep = df["n_generated"] <= threshold
    return df.loc[keep], int((~keep).sum())


def _group_of(n: int) -> str:
    for label, lo, hi in GROUPS:
        if n >= lo and (hi is None or n <= hi):
            return label
    return GROUPS[-1][0]


def table_author_output() -> dict:
    """Output length and fabrication count by ground-truth author count."""
    df, dropped = _without_runaway_generations(load_all_responses())
    df = df.copy()
    df["group"] = df["n_authors"].map(_group_of)

    records, rows = [], []
    for label, _, _ in GROUPS:
        sub = df[df["group"] == label]
        if sub.empty:
            continue
        output = float(sub["n_generated"].mean())
        halluc = float(sub["h"].mean())
        per_author = float((sub["h"] / sub["n_authors"].clip(lower=1)).mean())
        records.append({"gt_authors": label, "n_papers": int(len(sub)), "output": output,
                        "hallucinated": halluc, "hallucinated_per_author": per_author})
        rows.append([label, fmt(output, 1), fmt(halluc, 1), fmt(per_author, 1)])

    save_table(
        "author_output",
        [r"\textbf{GT Authors}", r"\textbf{Output}", r"\textbf{Halluc.}",
         r"\textbf{Halluc/Author}"],
        rows, align="lccc", source=SOURCE,
        values={"by_author_count": records, "runaway_rows_dropped": dropped},
    )
    return {"by_author_count": records, "runaway_rows_dropped": dropped}


def table_field_author() -> dict:
    """Output behaviour on single-author papers, by field."""
    df, dropped = _without_runaway_generations(load_all_responses())
    records = []
    for field in FIELD_KEYS:
        sub = df[df["field"] == field]
        singles = sub[sub["n_authors"] == 1]
        valid = sub[["n_authors", "hr2"]].dropna()
        rho, p = stats.spearmanr(valid["n_authors"], valid["hr2"])
        records.append({
            "field": field,
            "typical_authors": float(sub["n_authors"].mean()),
            "output_gt1": float(singles["n_generated"].mean()) if len(singles) else float("nan"),
            "hallucinated_gt1": float(singles["h"].mean()) if len(singles) else float("nan"),
            "n_single_author_papers": int(len(singles)),
            "rho_authors_hr2": float(rho), "p": float(p),
        })
    records.sort(key=lambda r: -r["typical_authors"])

    rows = [[field_label(r["field"]), fmt(r["typical_authors"], 2), fmt(r["output_gt1"], 1),
             fmt(r["hallucinated_gt1"], 1), fmt(r["rho_authors_hr2"], 2)] for r in records]
    save_table(
        "field_author",
        [r"\textbf{Field}", r"\textbf{Typical Authors}", r"\textbf{Output (GT=1)}",
         r"\textbf{Halluc. (GT=1)}", r"\textbf{$\rho$}"],
        rows, align="lcccc", source=SOURCE,
        values={"by_field": records, "runaway_rows_dropped": dropped},
    )
    return {"by_field": records}


def output_scaling() -> dict:
    """The rho ~ 0.55 claim: output length tracks the true author count."""
    df, _ = _without_runaway_generations(load_all_responses())
    valid = df[["n_authors", "n_generated", "hr2"]].dropna()
    rho_scale, p_scale = stats.spearmanr(valid["n_authors"], valid["n_generated"])
    rho_hr, p_hr = stats.spearmanr(valid["n_authors"], valid["hr2"])
    return {
        "spearman_authors_vs_output": {"rho": float(rho_scale), "p": float(p_scale)},
        "spearman_authors_vs_hr2": {"rho": float(rho_hr), "p": float(p_hr)},
        "n": int(len(valid)),
    }


def run_all() -> dict:
    return {**table_author_output(), **table_field_author(), "scaling": output_scaling()}


if __name__ == "__main__":
    out = run_all()
    write_json(OUT_DIR / "author_count_scaling.json", out["scaling"], source=SOURCE)
