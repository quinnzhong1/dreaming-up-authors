"""Author position (Section 8, Figure A6): does accuracy depend on list position?

Two curves -- miss rate by ground-truth position, hallucination rate by
generated position -- both restricted to answers that got at least one author
right. That restriction is the point: it conditions on the model knowing
something about the paper, so the curve measures degradation along the list
rather than total ignorance.

Re-runs the matcher of module 06, unlike every other analysis here, because
positions are not stored in the released CSVs. Takes about a minute.

Usage (from the repository root):
    python 07_additional_analyses/analyze_position.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import json

import numpy as np
import pandas as pd

from shared import (  # noqa: E402
    MODELS,
    MODEL_COLORS,
    MODEL_LABELS,
    OUT_DIR,
    apply_style,
    load_responses,
    save_figure,
    write_json,
)
from score_answers import Evaluator  # noqa: E402

SOURCE = "07_additional_analyses/analyze_position.py"
MAX_POSITION = 10


def positional_counts(model: str, max_position: int = MAX_POSITION) -> dict:
    """Per-position hit/miss tallies for one model."""
    evaluator = Evaluator()
    df = load_responses(model)

    gt_total = np.zeros(max_position, dtype=int)
    gt_missed = np.zeros(max_position, dtype=int)
    llm_total = np.zeros(max_position, dtype=int)
    llm_hallucinated = np.zeros(max_position, dtype=int)
    n_used = 0

    for gt_json, answer in zip(df["ground_truth_authors"], df["llm_answer"], strict=True):
        gt = json.loads(gt_json)
        result = evaluator.evaluate(gt, "" if pd.isna(answer) else str(answer))
        if not result.matched:
            # Figure A6 restricts to answers with at least one correct
            # author, so the curve is not dominated by total failures.
            continue
        n_used += 1

        for i in range(min(len(gt), max_position)):
            gt_total[i] += 1
        for i in result.unretrieved_gt_positions:
            if i < max_position:
                gt_missed[i] += 1

        for i in range(min(result.n_llm_names, max_position)):
            llm_total[i] += 1
        for i in result.hallucinated_llm_positions:
            if i < max_position:
                llm_hallucinated[i] += 1

    with np.errstate(invalid="ignore", divide="ignore"):
        miss_rate = np.where(gt_total > 0, gt_missed / gt_total, np.nan)
        halluc_rate = np.where(llm_total > 0, llm_hallucinated / llm_total, np.nan)

    return {
        "model": model,
        "n_papers_used": n_used,
        "n_papers_total": int(len(df)),
        "positions": list(range(1, max_position + 1)),
        "miss_rate": [None if np.isnan(v) else float(v) for v in miss_rate],
        "hallucination_rate": [None if np.isnan(v) else float(v) for v in halluc_rate],
        "gt_at_position": gt_total.tolist(),
        "generated_at_position": llm_total.tolist(),
    }


def _plot(all_counts: dict, key: str, ylabel: str, name: str) -> None:
    apply_style()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    for model in MODELS:
        c = all_counts[model]
        y = [np.nan if v is None else v for v in c[key]]
        ax.plot(c["positions"], y, marker="o", markersize=3.5, linewidth=1.5,
                color=MODEL_COLORS[model], label=MODEL_LABELS[model])
    ax.set_xlabel("Author position")
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(1, MAX_POSITION + 1))
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=7)
    fig.tight_layout()
    save_figure(fig, name, {"key": key, "models": all_counts}, source=SOURCE)


def run_all() -> dict:
    all_counts = {model: positional_counts(model) for model in MODELS}
    _plot(all_counts, "miss_rate", "Miss rate", "fig_positional_miss_rate")
    _plot(all_counts, "hallucination_rate", "Hallucination rate",
          "fig_positional_hallucination_rate")

    # The specific values quoted in the Section 8 prose.
    quoted = {}
    for model in MODELS:
        c = all_counts[model]
        quoted[model] = {
            "miss_rate_first": c["miss_rate"][0],
            "miss_rate_tenth": c["miss_rate"][MAX_POSITION - 1],
            "hallucination_rate_first": c["hallucination_rate"][0],
            "hallucination_rate_tenth": c["hallucination_rate"][MAX_POSITION - 1],
        }
    return {"per_model": all_counts, "quoted": quoted}


if __name__ == "__main__":
    write_json(OUT_DIR / "positional_quoted.json", run_all()["quoted"], source=SOURCE)
