"""score_open_model: score generated answers against ground truth.

Reads generated text from ``embeddings.h5``, joins the source CSV for
ground-truth authors, and writes per-paper HR2 to ``scored.tsv`` plus per-name
hallucination labels for the P9 analysis.

Usage
─────
  python 08_hidden_state_analysis/score_open_model.py \\
      --h5_path   data/probing/runs/qwen3_32b/embeddings.h5 \\
      --data_csv  data/probing/probing_prompts_9108.csv \\
      --out_dir   data/probing/runs/qwen3_32b
"""

import os
import re
import csv
import json
import argparse

import numpy as np
import h5py

import sys
sys.path.insert(0, os.path.dirname(__file__))
from scholar_utils import NameMatcher, score_authors, per_name_is_hallucinated

# Single shared instance — NameMatcher is stateless after construction
_MATCHER = NameMatcher()


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

_QUESTION_RE = re.compile(
    r"titled '(.+?)' which was published in (\d{4})", re.IGNORECASE
)

def _parse_title_year(question: str) -> tuple[str, str]:
    m = _QUESTION_RE.search(question)
    return (m.group(1), m.group(2)) if m else ("", "")


def load_ground_truth(csv_path: str) -> dict[str, dict]:
    """
    Load ground-truth metadata keyed by paper_id.
    Falls back to row index as paper_id if the column is absent.

    Expected columns: paperId, question, answer, citation, citation_bin, field
    """
    import pandas as pd
    df = pd.read_csv(csv_path)
    has_pid = "paperId" in df.columns
    gt: dict[str, dict] = {}
    for i, row in df.iterrows():
        pid = str(row["paperId"]) if has_pid else str(i)
        title, year = _parse_title_year(str(row.get("question", "")))
        gt[pid] = {
            "gt_authors":     str(row.get("answer", "")),
            "field":          str(row.get("field", "")),
            "citation_count": int(row.get("citation", 0)),
            "citation_bin":   str(row.get("citation_bin", "")),
            "title":          title,
            "year":           year,
        }
    return gt


def load_generated_texts(h5_path: str) -> dict[str, str]:
    """
    Read paper_ids and generated_text arrays from the HDF5 written by extract_hidden_states.
    Returns {paper_id: generated_text}.
    """
    generated: dict[str, str] = {}
    with h5py.File(h5_path, "r") as h5:
        paper_ids = h5["paper_ids"][:]
        texts     = h5["generated_text"][:]
        mask      = h5["completed_mask"][:]
        for i, (pid_bytes, text, done) in enumerate(zip(paper_ids, texts, mask)):
            if not done:
                continue
            pid  = pid_bytes.decode("utf-8") if isinstance(pid_bytes, bytes) else str(pid_bytes)
            text = text.decode("utf-8") if isinstance(text, bytes) else str(text)
            generated[pid] = text
    return generated


def load_per_name_predictions(h5_path: str) -> list[dict]:
    """
    Read per_name/ arrays from HDF5.
    Returns list of {paper_id, name_str, author_pos} for each stored name.
    """
    rows = []
    with h5py.File(h5_path, "r") as h5:
        if "per_name/paper_id" not in h5:
            return rows
        paper_ids  = h5["per_name/paper_id"][:]
        name_strs  = h5["per_name/name_str"][:]
        author_pos = h5["per_name/author_pos"][:]
        for pid_b, ns_b, pos in zip(paper_ids, name_strs, author_pos):
            pid = pid_b.decode("utf-8") if isinstance(pid_b, bytes) else str(pid_b)
            ns  = ns_b.decode("utf-8")  if isinstance(ns_b,  bytes) else str(ns_b)
            rows.append({"paper_id": pid, "name_str": ns, "author_pos": int(pos)})
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────────

def clean_generated_text(text: str) -> str:
    """
    Strip Qwen3 artifacts from generated text before scoring.
      - Remove any <think>...</think> block if thinking was not fully disabled
      - Strip leading/trailing whitespace and quotation marks
    """
    # Remove thinking block if present
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Remove common preamble patterns ("The authors are:", "Authors:", etc.)
    text = re.sub(r"(?i)^(the\s+)?authors?\s*(of[^:]*)?:\s*", "", text.strip())
    # Strip leading delimiter the model occasionally emits before the first name
    text = text.strip().strip('"').strip("'").lstrip(";,，；").strip()
    return text


def score_paper(generated_text: str, gt_authors_str: str) -> dict:
    """Compute all per-paper metrics using hierarchical NameMatcher."""
    cleaned = clean_generated_text(generated_text)
    return score_authors(gt_authors_str, cleaned, _MATCHER)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

SCORED_FIELDS = [
    "paper_id", "field", "citation_count", "citation_bin",
    "title", "year",
    "generated_text", "predicted_authors",
    "hr2", "hr4", "hr5",
    "n_match", "n_hal", "n_unretr",
    "is_hallucinated", "n_gt_authors", "n_pred_authors",
]

PER_NAME_FIELDS = [
    "paper_id", "author_pos", "name_str",
    "is_hallucinated",   # 1 = not in GT, 0 = matched
    "field", "citation_count", "citation_bin",
]


def main():
    parser = argparse.ArgumentParser(description="score_open_model: Score generated answers")
    parser.add_argument("--h5_path",  required=True,
                        help="Path to embeddings.h5 from extract_hidden_states")
    parser.add_argument("--data_csv", required=True,
                        help="Source evaluation CSV with ground-truth authors")
    parser.add_argument("--out_dir",  required=True,
                        help="Output directory (same as extract_hidden_states --out_dir is fine)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"[score_open_model] Loading ground truth: {args.data_csv}")
    gt_dict = load_ground_truth(args.data_csv)
    print(f"[score_open_model] {len(gt_dict)} ground-truth records loaded")

    print(f"[score_open_model] Loading generated texts: {args.h5_path}")
    gen_dict = load_generated_texts(args.h5_path)
    print(f"[score_open_model] {len(gen_dict)} generated answers found")

    print(f"[score_open_model] Loading per-name predictions from HDF5 ...")
    per_name_rows = load_per_name_predictions(args.h5_path)
    print(f"[score_open_model] {len(per_name_rows)} per-name entries found")

    # ── Score per paper ───────────────────────────────────────────────────────
    scored_path = os.path.join(args.out_dir, "scored.tsv")
    n_missing = 0
    hr2_values = []

    with open(scored_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SCORED_FIELDS, delimiter="\t")
        writer.writeheader()

        for pid in sorted(gt_dict.keys()):
            if pid not in gen_dict:
                n_missing += 1
                continue
            gt  = gt_dict[pid]
            metrics = score_paper(gen_dict[pid], gt["gt_authors"])
            hr2_values.append(metrics["hr2"])

            writer.writerow({
                "paper_id":          pid,
                "field":             gt["field"],
                "citation_count":    gt["citation_count"],
                "citation_bin":      gt["citation_bin"],
                "title":             gt["title"],
                "year":              gt["year"],
                "generated_text":    gen_dict[pid],
                **metrics,
            })

    n_scored = len(hr2_values)
    mean_hr2 = sum(hr2_values) / n_scored if n_scored else float("nan")
    hal_pct  = 100 * sum(v > 0.5 for v in hr2_values) / n_scored if n_scored else 0.0
    print(f"[score_open_model] Scored {n_scored} papers  (missing={n_missing})")
    print(f"         mean HR2={mean_hr2:.4f}  hallucinated(HR2>0.5)={hal_pct:.1f}%")
    print(f"         Saved → {scored_path}")

    # ── Score per name (P9 labels) ────────────────────────────────────────────
    if per_name_rows:
        # Build predicted-name sets from scoring step for label lookup
        # We re-derive the set from the generated text to keep things consistent
        per_name_path = os.path.join(args.out_dir, "per_name_labels.tsv")
        with open(per_name_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=PER_NAME_FIELDS, delimiter="\t")
            writer.writeheader()

            for entry in per_name_rows:
                pid = entry["paper_id"]
                if pid not in gt_dict:
                    continue
                is_hal  = int(per_name_is_hallucinated(
                    entry["name_str"], gt_dict[pid]["gt_authors"], _MATCHER
                ))
                gt_meta = gt_dict[pid]
                writer.writerow({
                    "paper_id":       pid,
                    "author_pos":     entry["author_pos"],
                    "name_str":       entry["name_str"],
                    "is_hallucinated":is_hal,
                    "field":          gt_meta["field"],
                    "citation_count": gt_meta["citation_count"],
                    "citation_bin":   gt_meta["citation_bin"],
                })

        n_hal_names = sum(
            1 for e in per_name_rows
            if e["paper_id"] in gt_dict and
            per_name_is_hallucinated(e["name_str"], gt_dict[e["paper_id"]]["gt_authors"], _MATCHER)
        )
        print(f"[score_open_model] Per-name: {len(per_name_rows)} names  "
              f"({100*n_hal_names/len(per_name_rows):.1f}% hallucinated)")
        print(f"         Saved → {per_name_path}")


if __name__ == "__main__":
    main()
