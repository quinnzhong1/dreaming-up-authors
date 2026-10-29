"""Decompose the steering effect into recall and precision channels.

HR2 mixes both, so a fix that recovers one correct author can be masked if it
also adds hallucinated names. Re-scores the stored generations to compare
n_match, HR5 (recall) and HR4 (precision) per paper, baseline against steered,
with paired effect sizes and Wilcoxon tests. The induce arm doubles as a
sensitivity check: if the fine metric does not move where the effect is known
to be strong, it is not sensitive enough to trust on the subtle fix case.

Usage
─────
  python 08_hidden_state_analysis/steering_name_recall.py \
      --data_csv data/probing/probing_prompts_9108.csv \
      --result_dirs results/hidden_state_analysis/steering/qwen3_32b results/hidden_state_analysis/steering/mistral_small_3_2_24b
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from score_open_model import load_ground_truth, score_paper

# Prediction files to re-score, and the baseline condition each pairs against
# (baselines live in the same file so pairing is within-run).
PRED_FILES = [
    ("steering_predictions.tsv", "baseline"),
]

METRICS = ["n_match", "hr5", "hr4", "hr2"]


def rescore_file(path: Path, gt: dict) -> pd.DataFrame:
    """Re-score every row's generated_text; return long per-row metrics."""
    df = pd.read_csv(path, sep="\t")
    rows = []
    n_missing = 0
    for _, r in df.iterrows():
        pid = str(r["paper_id"])
        g = gt.get(pid)
        if g is None:
            n_missing += 1
            continue
        s = score_paper(str(r["generated_text"]), g["gt_authors"])
        rows.append({
            "paper_id": pid,
            "pool": r["pool"],
            "condition": r["condition"],
            "direction": r["direction"],
            "alpha": float(r["alpha"]),
            "citation_count": r.get("citation_count", np.nan),
            "n_gt_authors": s["n_gt_authors"],
            "n_match": s["n_match"],
            "hr5": s["hr5"],
            "hr4": s["hr4"],
            "hr2": s["hr2"],
            "stored_hr2": float(r["hr2"]),
        })
    if n_missing:
        print(f"  [warn] {n_missing} rows in {path.name} had no ground-truth match")
    return pd.DataFrame(rows)


def paired_stats(base: pd.Series, steer: pd.Series) -> dict:
    """Paired baseline-vs-steered stats for one metric (aligned by index)."""
    d = (steer - base).dropna()
    n = len(d)
    out = {
        "n_pairs": n,
        "base_mean": round(float(base.mean()), 4),
        "steer_mean": round(float(steer.mean()), 4),
        "delta_mean": round(float(d.mean()), 4),
        "frac_up": round(float((d > 0).mean()), 4),
        "frac_down": round(float((d < 0).mean()), 4),
        "cohens_d": np.nan,
        "wilcoxon_p": np.nan,
    }
    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    if sd > 0:
        out["cohens_d"] = round(float(d.mean()) / sd, 4)
    if n > 0 and (d != 0).any():
        try:
            out["wilcoxon_p"] = round(float(stats.wilcoxon(d, zero_method="wilcox").pvalue), 6)
        except ValueError:
            pass
    return out


def analyze_dir(result_dir: Path, gt: dict) -> pd.DataFrame:
    print(f"\n=== {result_dir.name} ===")
    per_row_frames = []
    summary_rows = []

    for fname, base_cond in PRED_FILES:
        path = result_dir / fname
        if not path.exists():
            print(f"  [skip] {fname} not found")
            continue
        rescored = rescore_file(path, gt)
        rescored["source_file"] = fname
        per_row_frames.append(rescored)

        # Sanity: recomputed hr2 must match stored hr2.
        mad = float((rescored["hr2"] - rescored["stored_hr2"]).abs().mean())
        print(f"  [{fname}] rows={len(rescored)}  mean|hr2_recomp - hr2_stored|={mad:.2e}")

        for pool in sorted(rescored["pool"].unique()):
            pool_df = rescored[rescored["pool"] == pool]
            base = pool_df[pool_df["condition"] == base_cond].set_index("paper_id")
            if base.empty:
                continue
            steered = pool_df[pool_df["condition"] != base_cond]
            for (cond, direction, alpha), grp in steered.groupby(["condition", "direction", "alpha"]):
                grp = grp.set_index("paper_id")
                common = base.index.intersection(grp.index)
                if len(common) == 0:
                    continue
                for metric in METRICS:
                    st = paired_stats(base.loc[common, metric], grp.loc[common, metric])
                    summary_rows.append({
                        "model": result_dir.name, "source_file": fname, "pool": pool,
                        "condition": cond, "direction": direction, "alpha": alpha,
                        "metric": metric, **st,
                    })

    summary = pd.DataFrame(summary_rows)
    if per_row_frames:
        per_row = pd.concat(per_row_frames, ignore_index=True)
        out_row = result_dir / "name_recall_per_row.tsv"
        per_row.to_csv(out_row, sep="\t", index=False)
        print(f"  wrote {out_row}")
    if not summary.empty:
        out_sum = result_dir / "name_recall_summary.csv"
        summary.to_csv(out_sum, index=False)
        print(f"  wrote {out_sum}")
    return summary


def print_report(summary: pd.DataFrame) -> None:
    """Human-readable focus on the two questions that matter."""
    if summary.empty:
        return
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)

    def show(title, mask):
        sub = summary[mask & (summary["metric"] == "n_match")]
        if sub.empty:
            return
        print(f"\n----- {title} (metric = n_match, correct names retrieved) -----")
        cols = ["model", "condition", "alpha", "n_pairs", "base_mean",
                "steer_mean", "delta_mean", "frac_up", "frac_down", "cohens_d", "wilcoxon_p"]
        print(sub[cols].to_string(index=False))

    print("\n" + "=" * 90)
    print("Q: does the FIX direction recover any correct names? (want delta_mean > 0)")
    show("eval_low  toward_cited  (real fix + controls)",
         (summary["pool"] == "eval_low") & (summary["direction"].isin(["toward_cited", "random"])))

    print("\n" + "=" * 90)
    print("Q: SYMMETRIC sensitivity — INDUCE direction should LOSE correct names (want delta_mean < 0)")
    show("eval_high  toward_hallucinated (+ random induce)",
         (summary["pool"] == "eval_high") & (summary["direction"].isin(["toward_hallucinated", "random"])))

    # Recall channel (hr5) headline for the fix direction, real vector only.
    fix = summary[(summary["pool"] == "eval_low") & (summary["condition"] == "steering")
                  & (summary["metric"] == "hr5")]
    if not fix.empty:
        print("\n----- recall channel: eval_low real steering, metric = hr5 (1 - recall; want delta < 0) -----")
        print(fix[["model", "alpha", "n_pairs", "base_mean", "steer_mean",
                   "delta_mean", "cohens_d", "wilcoxon_p"]].to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_csv", required=True)
    ap.add_argument("--result_dirs", nargs="+", required=True)
    args = ap.parse_args()

    gt = load_ground_truth(args.data_csv)
    print(f"[analyze] loaded ground truth for {len(gt)} papers")

    all_summaries = []
    for rd in args.result_dirs:
        rd = Path(rd)
        if not rd.exists():
            print(f"[skip] {rd} does not exist")
            continue
        all_summaries.append(analyze_dir(rd, gt))

    combined = pd.concat([s for s in all_summaries if not s.empty], ignore_index=True) \
        if any(not s.empty for s in all_summaries) else pd.DataFrame()
    print_report(combined)


if __name__ == "__main__":
    main()
