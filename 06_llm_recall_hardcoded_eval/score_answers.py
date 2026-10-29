"""Score raw LLM answers with the hard-coded evaluation (Section 4.5).

One record in, one ``(M, H, U)`` triple out. ``name_matcher.NameMatcher``
reports a match level per (ground truth, LLM name) pair, and this script decides
which levels count: L1-L4 are matches, L5 (same surname, no matching initial)
is not.

A pair at L5 is demoted on both sides: the ground-truth author goes to U and the
LLM name goes to H, so a single weak pair costs the model twice. That is the
conservative reading, and it is what produced the published numbers.

Usage (from the repository root):
    python 06_llm_recall_hardcoded_eval/score_answers.py INPUT OUTPUT

INPUT is the NDJSON written by ``query_llms.py`` or any CSV with a ground-truth
column (``ground_truth_authors`` or ``answer``) and an answer column
(``llm_answer``, ``llm_authors`` or ``gpt_answer``). OUTPUT is a CSV with the
input columns plus ``hard_code_evaluation`` ("M, H, U") and the matched,
hallucinated, and unretrieved names.

Re-scoring the released answers reproduces the stored counts for every
DeepSeek-R1 and Claude Sonnet 4.5 paper and for 9,095 of 9,108 GPT-4o papers:
    python 06_llm_recall_hardcoded_eval/score_answers.py --check
"""

from __future__ import annotations

import argparse
import csv
import gzip
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import MATCH_LEVEL_THRESHOLD, MODELS, read_ndjson  # noqa: E402
from ground_truth import ground_truth_list, llm_answer  # noqa: E402
from name_matcher import NameMatcher  # noqa: E402

import config  # noqa: E402  (on sys.path via common)

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


def match_level_value(label: str) -> float:
    """``"L3.5_full_component_swapped"`` -> ``3.5``."""
    return float(label.split("_")[0][1:])


@dataclass
class EvaluationResult:
    """Per-paper evaluation output."""

    matched: list[str] = field(default_factory=list)        # M
    hallucinated: list[str] = field(default_factory=list)   # H
    unretrieved: list[str] = field(default_factory=list)    # U
    match_levels: list[str] = field(default_factory=list)

    # Positions, for the author-position analysis in module 07.
    matched_gt_positions: list[int] = field(default_factory=list)
    unretrieved_gt_positions: list[int] = field(default_factory=list)
    hallucinated_llm_positions: list[int] = field(default_factory=list)
    n_llm_names: int = 0

    @property
    def counts(self) -> tuple[int, int, int]:
        return len(self.matched), len(self.hallucinated), len(self.unretrieved)

    @property
    def counts_str(self) -> str:
        """The ``"M, H, U"`` form stored in ``hard_code_evaluation``."""
        m, h, u = self.counts
        return f"{m}, {h}, {u}"

    def as_columns(self) -> dict[str, str]:
        """The four evaluation columns appended to every scored CSV."""
        return {
            "hard_code_evaluation": self.counts_str,
            "matched_gt_authors(X)": "; ".join(self.matched),
            "hallucinated_llm_authors(Y)": "; ".join(self.hallucinated),
            "unmatched_gt_authors(Z)": "; ".join(self.unretrieved),
        }


class Evaluator:
    """Stateless scorer. Construct once, reuse across a whole file."""

    def __init__(self, match_level_threshold: float = MATCH_LEVEL_THRESHOLD):
        self.threshold = float(match_level_threshold)
        self.matcher = NameMatcher()

    def evaluate(self, gt_authors: list[str], llm_string: str) -> EvaluationResult:
        raw = self.matcher.find_matches(gt_authors, llm_string)

        strong, weak = [], []
        for pair in raw["matches"]:
            (strong if match_level_value(pair["match_level"]) <= self.threshold else weak).append(pair)

        return EvaluationResult(
            matched=[p["gt_author"] for p in strong],
            # A weak pair is demoted on both sides.
            hallucinated=list(raw["hallucinated_llm"]) + [p["llm_author"] for p in weak],
            unretrieved=[u["gt_author"] for u in raw["unmatched_gt"]] + [p["gt_author"] for p in weak],
            match_levels=[p["match_level"] for p in strong],
            matched_gt_positions=[p["gt_index"] for p in strong],
            unretrieved_gt_positions=(
                [u["gt_index"] for u in raw["unmatched_gt"]] + [p["gt_index"] for p in weak]
            ),
            hallucinated_llm_positions=(
                [d["llm_index"] for d in raw["hallucinated_llm_detail"]]
                + [p["llm_index"] for p in weak]
            ),
            n_llm_names=raw["n_llm_names"],
        )

    def evaluate_record(self, record: dict[str, Any]) -> EvaluationResult:
        """Evaluate a record in any of the shipped schemas."""
        return self.evaluate(
            ground_truth_list(record, self.matcher.splitter),
            llm_answer(record),
        )


def _read_rows(path: Path):
    if path.suffix in (".ndjson", ".jsonl") or path.name.endswith((".ndjson.gz", ".jsonl.gz")):
        yield from read_ndjson(path)
        return
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as f:
        yield from csv.DictReader(f)


def score_file(source: Path, output: Path, threshold: float) -> int:
    """Score every record of ``source`` and write ``output`` atomically."""
    evaluator = Evaluator(threshold)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    n = 0
    writer = None
    opener = gzip.open if output.suffix == ".gz" else open
    with opener(tmp, "wt", encoding="utf-8", newline="") as f:
        for record in _read_rows(source):
            record.update(evaluator.evaluate_record(record).as_columns())
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(record), extrasaction="ignore")
                writer.writeheader()
            writer.writerow(record)
            n += 1
    os.replace(tmp, output)
    return n


def check_release(max_examples: int = 3) -> int:
    """Re-score ``release/responses/`` and compare with the stored (m, h, u)."""
    evaluator = Evaluator()
    for model in MODELS:
        path = config.RESPONSES_RELEASE_DIR / f"{model}.csv.gz"
        rows = list(_read_rows(path))
        mismatches = []
        for row in rows:
            got = evaluator.evaluate_record(row).counts_str
            want = f"{row['m']}, {row['h']}, {row['u']}"
            if got != want:
                mismatches.append((row["paper_id"], want, got))
        identical = len(rows) - len(mismatches)
        print(f"{model:20s} {identical:,}/{len(rows):,} identical")
        for paper_id, want, got in mismatches[:max_examples]:
            print(f"    {paper_id}: stored {want!r}, recomputed {got!r}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", nargs="?", type=Path)
    ap.add_argument("output", nargs="?", type=Path)
    ap.add_argument("--match-level-threshold", type=float, default=MATCH_LEVEL_THRESHOLD,
                    help="highest matcher level that counts as a match (default: 4)")
    ap.add_argument("--check", action="store_true",
                    help="re-score release/responses/ and compare with the stored counts")
    args = ap.parse_args()

    if args.check:
        return check_release()
    if args.input is None or args.output is None:
        ap.error("INPUT and OUTPUT are required unless --check is given")
    n = score_file(args.input, args.output, args.match_level_threshold)
    print(f"Scored {n:,} records -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
