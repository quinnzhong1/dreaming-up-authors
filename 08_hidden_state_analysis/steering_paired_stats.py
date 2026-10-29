"""Paired uncertainty for the steering tables -- the CIs Appendix A7.7 quotes.

The steering stages report rates at n of 30-50 with no CI or test, and the data
is paired, so exact tests are nearly free. Per condition against its in-file
baseline: McNemar exact on the hallucinated indicator with discordant-pair
counts, a paired bootstrap CI for the change in rate and in mean HR2, and
Wilcoxon with ``zero_method="pratt"`` plus the effective n -- the default
silently drops zero differences, so a near-inert arm reports a p computed on a
handful of pairs while n looks full.

Also reports covariate balance across the vector pools, and the paired change
in emitted name count, since a vector that changes how many names the model
emits moves HR2 without touching attribution. CPU-only.

Usage
─────
  python 08_hidden_state_analysis/steering_paired_stats.py \
      --data_csv data/probing/probing_prompts_9108.csv \
      --result_dirs results/hidden_state_analysis/steering/qwen3_32b results/hidden_state_analysis/steering/mistral_small_3_2_24b
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from scholar_utils import NameMatcher
from score_open_model import load_ground_truth, score_paper

_MATCHER = NameMatcher()
BOOT = 5000
SEED = 42

# (prediction file, in-file baseline condition name). Same set the name-recall
# analysis uses; any missing file is skipped.
PRED_FILES = [
    ("steering_predictions.tsv", "baseline"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Paired statistics (E6)
# ─────────────────────────────────────────────────────────────────────────────

def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p on discordant pair counts b, c (binomial)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * tail)


def paired_bootstrap_ci(base: np.ndarray, steer: np.ndarray, rng, reducer) -> tuple:
    """Percentile 95% CI of reducer(steer) - reducer(base) over paired resamples."""
    n = len(base)
    if n == 0:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, n, size=(BOOT, n))
    diffs = reducer(steer[idx], axis=1) - reducer(base[idx], axis=1)
    return (round(float(np.percentile(diffs, 2.5)), 4), round(float(np.percentile(diffs, 97.5)), 4))


def paired_row(base_df: pd.DataFrame, steer_df: pd.DataFrame, rng) -> dict:
    """All E6 stats for one paired baseline-vs-steered comparison, aligned by paper_id."""
    common = base_df.index.intersection(steer_df.index)
    b_hr2 = base_df.loc[common, "hr2"].to_numpy(dtype=float)
    s_hr2 = steer_df.loc[common, "hr2"].to_numpy(dtype=float)
    b_hal = (b_hr2 > 0.5).astype(int)
    s_hal = (s_hr2 > 0.5).astype(int)

    b_disc = int(((b_hal == 0) & (s_hal == 1)).sum())   # baseline ok -> steered hallucinates
    c_disc = int(((b_hal == 1) & (s_hal == 0)).sum())   # baseline hallucinates -> steered ok
    mcp = mcnemar_exact(b_disc, c_disc)

    hr_ci = paired_bootstrap_ci(b_hal.astype(float) * 100, s_hal.astype(float) * 100,
                                rng, np.mean)  # in pp
    hr2_ci = paired_bootstrap_ci(b_hr2, s_hr2, rng, np.mean)

    d = s_hr2 - b_hr2
    nz = int((d != 0).sum())
    wilcox_p = np.nan
    if nz > 0:
        try:
            wilcox_p = round(float(stats.wilcoxon(d, zero_method="pratt").pvalue), 6)
        except ValueError:
            pass
    return {
        "n_pairs": len(common),
        "base_hr_pct": round(100.0 * float(b_hal.mean()), 2) if len(common) else float("nan"),
        "steer_hr_pct": round(100.0 * float(s_hal.mean()), 2) if len(common) else float("nan"),
        "delta_hr_pct": round(100.0 * float(s_hal.mean() - b_hal.mean()), 2) if len(common) else float("nan"),
        "delta_hr_ci_lo": hr_ci[0], "delta_hr_ci_hi": hr_ci[1],
        "mcnemar_b": b_disc, "mcnemar_c": c_disc, "mcnemar_exact_p": round(mcp, 6),
        "base_mean_hr2": round(float(b_hr2.mean()), 4) if len(common) else float("nan"),
        "steer_mean_hr2": round(float(s_hr2.mean()), 4) if len(common) else float("nan"),
        "delta_mean_hr2": round(float((s_hr2 - b_hr2).mean()), 4) if len(common) else float("nan"),
        "delta_hr2_ci_lo": hr2_ci[0], "delta_hr2_ci_hi": hr2_ci[1],
        "wilcoxon_pratt_p": wilcox_p, "n_effective_nonzero": nz,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Emitted-name-count re-scoring (E7b)
# ─────────────────────────────────────────────────────────────────────────────

def rescore_name_count(df: pd.DataFrame, gt: dict) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        pid = str(r["paper_id"])
        g = gt.get(pid)
        if g is None:
            continue
        s = score_paper(str(r["generated_text"]), g["gt_authors"])
        rows.append({"paper_id": pid, "pool": r["pool"], "condition": r["condition"],
                     "direction": r.get("direction", ""), "alpha": float(r.get("alpha", 0)),
                     "n_pred_authors": s["n_pred_authors"], "n_match": s["n_match"],
                     "hr2": s["hr2"]})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────

def analyze_dir(result_dir: Path, gt: dict, rng) -> tuple:
    print(f"\n=== {result_dir.name} ===")
    e6_rows, e7b_rows = [], []

    for fname, base_cond in PRED_FILES:
        path = result_dir / fname
        if not path.exists():
            print(f"  [skip] {fname} not found")
            continue
        df = pd.read_csv(path, sep="\t")
        if "pool" not in df.columns or "condition" not in df.columns:
            print(f"  [skip] {fname}: no pool/condition columns")
            continue
        namecnt = rescore_name_count(df, gt)
        print(f"  [{fname}] rows={len(df)}")

        for pool in sorted(df["pool"].astype(str).unique()):
            pool_df = df[df["pool"].astype(str) == pool]
            base = pool_df[pool_df["condition"] == base_cond].drop_duplicates("paper_id").set_index("paper_id")
            if base.empty:
                continue
            steered = pool_df[pool_df["condition"] != base_cond]
            for (cond, direction, alpha), grp in steered.groupby(
                    ["condition", "direction", "alpha"], dropna=False):
                grp = grp.drop_duplicates("paper_id").set_index("paper_id")
                if base.index.intersection(grp.index).empty:
                    continue
                stat = paired_row(base, grp, rng)
                e6_rows.append({"model": result_dir.name, "source_file": fname, "pool": pool,
                                "condition": cond, "direction": direction, "alpha": alpha, **stat})

                # E7b: emitted-name-count shift, same pairing.
                nc_pool = namecnt[namecnt["pool"].astype(str) == pool]
                nb = nc_pool[nc_pool["condition"] == base_cond].drop_duplicates("paper_id").set_index("paper_id")
                ng = nc_pool[(nc_pool["condition"] == cond) & (nc_pool["direction"].astype(str) == str(direction))
                             & (nc_pool["alpha"] == alpha)].drop_duplicates("paper_id").set_index("paper_id")
                common = nb.index.intersection(ng.index)
                if len(common):
                    dcnt = ng.loc[common, "n_pred_authors"].to_numpy(float) - \
                        nb.loc[common, "n_pred_authors"].to_numpy(float)
                    e7b_rows.append({"model": result_dir.name, "source_file": fname, "pool": pool,
                                     "condition": cond, "direction": direction, "alpha": alpha,
                                     "n_pairs": len(common),
                                     "base_mean_names": round(float(nb.loc[common, "n_pred_authors"].mean()), 3),
                                     "steer_mean_names": round(float(ng.loc[common, "n_pred_authors"].mean()), 3),
                                     "delta_mean_names": round(float(dcnt.mean()), 3)})

    e6 = pd.DataFrame(e6_rows)
    e7b = pd.DataFrame(e7b_rows)
    if not e6.empty:
        e6.to_csv(result_dir / "paired_hr_stats.csv", index=False)
        print(f"  wrote {result_dir / 'paired_hr_stats.csv'}")
    if not e7b.empty:
        e7b.to_csv(result_dir / "emitted_name_count_shift.csv", index=False)
        print(f"  wrote {result_dir / 'emitted_name_count_shift.csv'}")
    return e6, e7b


def covariate_balance(result_dir: Path, gt: dict) -> pd.DataFrame:
    """E7a: mean covariates for the two vector-building pools (from meta IDs)."""
    meta_path = result_dir / "steering_meta.json"
    if not meta_path.exists():
        print(f"  [E7a skip] no steering_meta.json in {result_dir.name}")
        return pd.DataFrame()
    meta = json.load(open(meta_path))
    rows = []
    for pool_name in ("vector_correct", "vector_hallucinated", "eval_low", "eval_high"):
        ids = [str(x) for x in meta.get("pool_paper_ids", {}).get(pool_name, [])]
        recs = [gt[i] for i in ids if i in gt]
        if not recs:
            continue
        n_authors = [len([n for n in _MATCHER.splitter.split(r["gt_authors"]) if n.strip()]) for r in recs]
        title_chars = [len(r["title"]) for r in recs]
        years = [float(r["year"]) for r in recs if str(r["year"]).strip() and str(r["year"]).isdigit()]
        cites = [r["citation_count"] for r in recs]
        rows.append({"model": result_dir.name, "pool": pool_name, "n": len(recs),
                     "mean_n_gt_authors": round(float(np.mean(n_authors)), 3),
                     "mean_title_chars": round(float(np.mean(title_chars)), 1),
                     "mean_year": round(float(np.mean(years)), 1) if years else float("nan"),
                     "mean_citation": round(float(np.mean(cites)), 1),
                     "median_citation": round(float(np.median(cites)), 1)})
    cb = pd.DataFrame(rows)
    if not cb.empty:
        cb.to_csv(result_dir / "vector_pool_covariate_balance.csv", index=False)
        print(f"  wrote {result_dir / 'vector_pool_covariate_balance.csv'}")
    return cb


def print_report(e6: pd.DataFrame, e7a: pd.DataFrame, e7b: pd.DataFrame) -> None:
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_rows", 200)
    if not e7a.empty:
        print("\n" + "=" * 100)
        print("E7a — vector-pool covariate balance (author count is the dangerous one for HR2)")
        print(e7a.to_string(index=False))
    if not e6.empty:
        print("\n" + "=" * 100)
        print("E6 — paired HR% change with McNemar exact p + bootstrap CI (headline tables, now with uncertainty)")
        cols = ["model", "source_file", "pool", "condition", "direction", "alpha", "n_pairs",
                "base_hr_pct", "steer_hr_pct", "delta_hr_pct", "delta_hr_ci_lo", "delta_hr_ci_hi",
                "mcnemar_b", "mcnemar_c", "mcnemar_exact_p", "wilcoxon_pratt_p", "n_effective_nonzero"]
        print(e6[cols].to_string(index=False))
    if not e7b.empty:
        print("\n" + "=" * 100)
        print("E7b — emitted-name-count shift (delta_mean_names co-moving with HR2 => compositional, not epistemic)")
        print(e7b.to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_csv", required=True)
    ap.add_argument("--result_dirs", nargs="+", required=True)
    args = ap.parse_args()

    gt = load_ground_truth(args.data_csv)
    print(f"[analyze] loaded ground truth for {len(gt)} papers")
    rng = np.random.default_rng(SEED)

    e6_all, e7a_all, e7b_all = [], [], []
    for rd in args.result_dirs:
        rd = Path(rd)
        if not rd.exists():
            print(f"[skip] {rd} does not exist")
            continue
        e6, e7b = analyze_dir(rd, gt, rng)
        e7a = covariate_balance(rd, gt)
        e6_all.append(e6); e7a_all.append(e7a); e7b_all.append(e7b)

    def cat(frames):
        frames = [f for f in frames if not f.empty]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    print_report(cat(e6_all), cat(e7a_all), cat(e7b_all))


if __name__ == "__main__":
    main()
