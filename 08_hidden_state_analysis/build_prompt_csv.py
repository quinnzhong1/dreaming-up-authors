"""Step 1: rebuild the prompt CSV every GPU stage reads as ``--data_csv``.

The stages read ``paperId, citation, citation_bin, field, question, answer`` --
a view of the released evaluation sample, rebuilt here rather than shipped
twice. The release uses different column names and holds authors as JSON.

The question is a fixed template, so prompts reproduce exactly. The ``answer``
column differs for 68 of the 9,108 papers: the original stripped apostrophes
from surnames and the release keeps them. Scoring is unaffected -- the matcher
normalises them away on both sides.

Input:
    config.EVAL_SAMPLE_RELEASE_CSV   (06_llm_recall_hardcoded_eval/release/)
Output:
    config.PROBING_PROMPTS_CSV       data/probing/probing_prompts_9108.csv

Usage:
    python 08_hidden_state_analysis/build_prompt_csv.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

QUESTION_TEMPLATE = (
    "Who are the authors of the paper titled '{title}' which was published in {year}?"
)

# The column order the stages' ``load_ground_truth`` expects to find.
PROMPT_COLUMNS = ["paperId", "citation", "citation_bin", "field", "question", "answer"]


def _authors_to_answer(raw: str) -> str:
    """The release stores authors as a JSON list; the stages want one string."""
    try:
        names = json.loads(raw)
    except (TypeError, ValueError):
        return str(raw)
    return ", ".join(str(n) for n in names)


def build_prompt_frame(sample: pd.DataFrame) -> pd.DataFrame:
    missing = {"paper_id", "title", "year", "authors", "citations",
               "citation_bin", "field"} - set(sample.columns)
    if missing:
        raise ValueError(f"evaluation sample is missing columns: {sorted(missing)}")
    years = sample["year"].astype("Int64").astype(str)
    return pd.DataFrame({
        "paperId": sample["paper_id"].astype(str),
        "citation": sample["citations"].astype("Int64"),
        "citation_bin": sample["citation_bin"].astype(str),
        "field": sample["field"].astype(str),
        "question": [QUESTION_TEMPLATE.format(title=t, year=y)
                     for t, y in zip(sample["title"].astype(str), years, strict=True)],
        "answer": sample["authors"].map(_authors_to_answer),
    })[PROMPT_COLUMNS]


def build_prompt_csv(sample_path: Path, out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame = build_prompt_frame(pd.read_csv(sample_path))
    frame.to_csv(out_path, index=False)
    return len(frame)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", type=Path, default=config.EVAL_SAMPLE_RELEASE_CSV)
    ap.add_argument("--output", type=Path, default=config.PROBING_PROMPTS_CSV)
    args = ap.parse_args()
    n = build_prompt_csv(args.sample, args.output)
    print(f"Wrote {n} prompts to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
