"""Paired significance tests for prompt sensitivity (Section 8, Tables A31-A33).

Aligns the evaluated prompt-variant CSVs by ``record_id`` and tests whether the
citation-bin Spearman correlations differ across variants, using a paired
nonparametric bootstrap over papers. Writes the five CSVs shipped in
``release/prompt_sensitivity/``, which ``analyze_prompt_sensitivity.py`` turns
into tables.

Each input is a scored CSV from ``06_llm_recall_hardcoded_eval/score_answers.py``
with ``record_id``, ``citation_bin`` and ``hard_code_evaluation`` columns. Pure
Python and CPU only; 10,000 resamples take several minutes.

Usage (from the repository root):
    python 07_additional_analyses/prompt_sensitivity_significance.py \
        --inputs baseline=/path/to/baseline_scored.csv \
                 emotional=/path/to/emotional_scored.csv \
                 authoritative=/path/to/authoritative_scored.csv \
        --out_dir results/additional_analyses/prompt_sensitivity \
        --n_boot 10000
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path
from typing import Callable


BIN_MIDPOINTS = {
    "0-2": 1.0,
    "3-4": 3.5,
    "5-8": 6.5,
    "9-16": 12.5,
    "17-32": 24.5,
    "33-64": 48.5,
    "65-128": 96.5,
    "129-256": 192.5,
    "257-512": 384.5,
    "513-1024": 768.5,
    "1025-2048": 1536.5,
    "2049-4096": 3072.5,
    "4097+": 5000.0,
}
METRICS = ("hr1", "hr2", "hr3", "hr4", "hr5")
DEFAULT_BASELINE = "baseline"
DEFAULT_SEED = 42


def parse_inputs(items: list[str]) -> dict[str, Path]:
    inputs: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--inputs entries must be name=/path, got {item!r}")
        name, path = item.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise ValueError(f"Invalid --inputs entry: {item!r}")
        inputs[name] = Path(path).expanduser().resolve()
    if len(inputs) < 2:
        raise ValueError("Need at least two prompt CSVs")
    return inputs


def parse_evaluation(value: str) -> tuple[int, int, int] | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "none":
        return None
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 3:
        return None
    try:
        return int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError:
        return None


def hr1(matched: int, hallucinated: int, unmatched: int) -> float:
    denom = max(hallucinated + matched, unmatched + matched)
    return math.nan if denom == 0 else 1.0 - matched / denom


def hr2(matched: int, hallucinated: int, unmatched: int) -> float:
    denom = hallucinated + matched + unmatched
    return math.nan if denom == 0 else (hallucinated + unmatched) / denom


def hr3(matched: int, hallucinated: int, unmatched: int) -> float:
    denom = (hallucinated + matched) * unmatched + (matched + unmatched) * hallucinated
    return 0.0 if denom == 0 else (2.0 * hallucinated * unmatched) / denom


def hr4(matched: int, hallucinated: int, unmatched: int) -> float:
    denom = matched + hallucinated
    return math.nan if denom == 0 else hallucinated / denom


def hr5(matched: int, hallucinated: int, unmatched: int) -> float:
    denom = matched + unmatched
    return math.nan if denom == 0 else unmatched / denom


METRIC_FNS: dict[str, Callable[[int, int, int], float]] = {
    "hr1": hr1,
    "hr2": hr2,
    "hr3": hr3,
    "hr4": hr4,
    "hr5": hr5,
}


def rank_average(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg_rank
        i = j
    return ranks


def pearson(x: list[float], y: list[float]) -> float:
    n = len(x)
    if n < 3:
        return math.nan
    mx = sum(x) / n
    my = sum(y) / n
    dx = [v - mx for v in x]
    dy = [v - my for v in y]
    sx = math.sqrt(sum(v * v for v in dx))
    sy = math.sqrt(sum(v * v for v in dy))
    if sx == 0 or sy == 0:
        return math.nan
    return sum(a * b for a, b in zip(dx, dy)) / (sx * sy)


def spearman(x: list[float], y: list[float]) -> float:
    pairs = [(a, b) for a, b in zip(x, y) if not (math.isnan(a) or math.isnan(b))]
    if len(pairs) < 3:
        return math.nan
    rx = rank_average([p[0] for p in pairs])
    ry = rank_average([p[1] for p in pairs])
    return pearson(rx, ry)


def quantile(values: list[float], q: float) -> float:
    clean = sorted(v for v in values if not math.isnan(v))
    if not clean:
        return math.nan
    pos = (len(clean) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return clean[lo]
    frac = pos - lo
    return clean[lo] * (1.0 - frac) + clean[hi] * frac


def bootstrap_pvalue(samples: list[float]) -> float:
    clean = [v for v in samples if not math.isnan(v)]
    if not clean:
        return math.nan
    le_zero = sum(1 for v in clean if v <= 0)
    ge_zero = sum(1 for v in clean if v >= 0)
    return min(1.0, 2.0 * (min(le_zero, ge_zero) + 1) / (len(clean) + 1))


def read_prompt_csv(path: Path, prompt_name: str) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"record_id", "citation_bin", "hard_code_evaluation"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing required columns: {sorted(missing)}")

        for row in reader:
            record_id = str(row.get("record_id", "")).strip()
            if not record_id:
                continue
            citation_bin = str(row.get("citation_bin", "")).strip()
            if citation_bin not in BIN_MIDPOINTS:
                continue
            counts = parse_evaluation(row.get("hard_code_evaluation", ""))
            if counts is None:
                continue
            matched, hallucinated, unmatched = counts
            out = {
                "record_id": record_id,
                "prompt": prompt_name,
                "citation_bin": citation_bin,
                "citation_bin_midpoint": BIN_MIDPOINTS[citation_bin],
                "matched": matched,
                "hallucinated": hallucinated,
                "unmatched": unmatched,
            }
            for metric, fn in METRIC_FNS.items():
                out[metric] = fn(matched, hallucinated, unmatched)
            rows[record_id] = out
    return rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aligned_records(prompt_rows: dict[str, dict[str, dict]]) -> list[str]:
    common: set[str] | None = None
    for rows in prompt_rows.values():
        ids = set(rows)
        common = ids if common is None else common & ids
    return sorted(common or set())


def build_long_rows(prompt_rows: dict[str, dict[str, dict]], ids: list[str]) -> list[dict]:
    rows = []
    for record_id in ids:
        for prompt, per_prompt in prompt_rows.items():
            rows.append(per_prompt[record_id])
    return rows


def summarize_spearman(prompt_rows: dict[str, dict[str, dict]], ids: list[str]) -> list[dict]:
    rows: list[dict] = []
    for prompt, per_prompt in prompt_rows.items():
        x = [per_prompt[rid]["citation_bin_midpoint"] for rid in ids]
        for metric in METRICS:
            y = [per_prompt[rid][metric] for rid in ids]
            rows.append({
                "prompt": prompt,
                "metric": metric,
                "n": len(ids),
                "spearman_rho": round(spearman(x, y), 6),
            })
    return rows


def summarize_means(prompt_rows: dict[str, dict[str, dict]], ids: list[str]) -> list[dict]:
    rows: list[dict] = []
    for prompt, per_prompt in prompt_rows.items():
        row = {"prompt": prompt, "n": len(ids)}
        for metric in METRICS:
            vals = [per_prompt[rid][metric] for rid in ids if not math.isnan(per_prompt[rid][metric])]
            row[f"{metric}_mean"] = round(sum(vals) / len(vals), 6)
        rows.append(row)
    return rows


def paired_bootstrap_diffs(
    prompt_rows: dict[str, dict[str, dict]],
    ids: list[str],
    baseline: str,
    n_boot: int,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    prompts = [p for p in prompt_rows if p != baseline]
    diff_rows: list[dict] = []
    mean_diff_rows: list[dict] = []

    for prompt in prompts:
        for metric in METRICS:
            x = [prompt_rows[baseline][rid]["citation_bin_midpoint"] for rid in ids]
            base_y = [prompt_rows[baseline][rid][metric] for rid in ids]
            test_y = [prompt_rows[prompt][rid][metric] for rid in ids]
            base_rho = spearman(x, base_y)
            test_rho = spearman(x, test_y)
            observed_diff = test_rho - base_rho

            base_mean = sum(v for v in base_y if not math.isnan(v)) / sum(1 for v in base_y if not math.isnan(v))
            test_mean = sum(v for v in test_y if not math.isnan(v)) / sum(1 for v in test_y if not math.isnan(v))
            observed_mean_diff = test_mean - base_mean

            boot_diffs: list[float] = []
            boot_mean_diffs: list[float] = []
            for _ in range(n_boot):
                sample_ids = [ids[rng.randrange(len(ids))] for _ in ids]
                bx = [prompt_rows[baseline][rid]["citation_bin_midpoint"] for rid in sample_ids]
                by = [prompt_rows[baseline][rid][metric] for rid in sample_ids]
                ty = [prompt_rows[prompt][rid][metric] for rid in sample_ids]
                boot_diffs.append(spearman(bx, ty) - spearman(bx, by))
                bvals = [v for v in by if not math.isnan(v)]
                tvals = [v for v in ty if not math.isnan(v)]
                boot_mean_diffs.append((sum(tvals) / len(tvals)) - (sum(bvals) / len(bvals)))

            diff_rows.append({
                "comparison": f"{prompt}_minus_{baseline}",
                "prompt": prompt,
                "baseline": baseline,
                "metric": metric,
                "n": len(ids),
                "baseline_rho": round(base_rho, 6),
                "prompt_rho": round(test_rho, 6),
                "rho_diff": round(observed_diff, 6),
                "abs_rho_diff": round(abs(observed_diff), 6),
                "rho_diff_ci_low": round(quantile(boot_diffs, 0.025), 6),
                "rho_diff_ci_high": round(quantile(boot_diffs, 0.975), 6),
                "rho_diff_p_two_sided": round(bootstrap_pvalue(boot_diffs), 6),
                "n_boot": n_boot,
            })
            mean_diff_rows.append({
                "comparison": f"{prompt}_minus_{baseline}",
                "prompt": prompt,
                "baseline": baseline,
                "metric": metric,
                "n": len(ids),
                "baseline_mean": round(base_mean, 6),
                "prompt_mean": round(test_mean, 6),
                "mean_diff": round(observed_mean_diff, 6),
                "abs_mean_diff": round(abs(observed_mean_diff), 6),
                "mean_diff_ci_low": round(quantile(boot_mean_diffs, 0.025), 6),
                "mean_diff_ci_high": round(quantile(boot_mean_diffs, 0.975), 6),
                "mean_diff_p_two_sided": round(bootstrap_pvalue(boot_mean_diffs), 6),
                "n_boot": n_boot,
            })

    return diff_rows, mean_diff_rows


def write_notes(path: Path, inputs: dict[str, Path], ids: list[str], n_boot: int, baseline: str) -> None:
    lines = [
        "# Prompt Sensitivity Significance Tests",
        "",
        f"Common paired records: {len(ids)}.",
        f"Baseline prompt: `{baseline}`.",
        f"Bootstrap resamples: {n_boot}.",
        "",
        "Inputs:",
    ]
    for name, input_path in inputs.items():
        lines.append(f"- `{name}`: `{input_path}`")
    lines += [
        "",
        "Method: align prompt variants by `record_id`, compute HR1-HR5 from `hard_code_evaluation`, map citation bins to fixed midpoints, and estimate Spearman rho between citation-bin midpoint and each hallucination metric for each prompt.",
        "",
        "Significance test: paired nonparametric bootstrap over record IDs. Each bootstrap sample preserves the same papers across prompt variants, recomputes the Spearman difference for prompt minus baseline, and reports percentile CIs plus a two-sided sign bootstrap p-value.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired significance tests for prompt sensitivity"
    )
    parser.add_argument("--inputs", nargs="+", required=True, help="Prompt CSVs as name=/path")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--baseline", default=DEFAULT_BASELINE, help="Baseline prompt name")
    parser.add_argument("--n_boot", type=int, default=10000, help="Paired bootstrap resamples")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    inputs = parse_inputs(args.inputs)
    if args.baseline not in inputs:
        raise ValueError(f"Baseline {args.baseline!r} is not in --inputs")

    out_dir = Path(args.out_dir).expanduser().resolve()
    prompt_rows = {name: read_prompt_csv(path, name) for name, path in inputs.items()}
    ids = aligned_records(prompt_rows)
    if len(ids) < 10:
        raise ValueError(f"Only {len(ids)} common record_id values across prompts")

    print(f"[prompt_sensitivity_significance] Common paired records: {len(ids)}")
    for name, rows in prompt_rows.items():
        print(f"[prompt_sensitivity_significance] {name}: {len(rows)} valid rows before pairing")

    long_rows = build_long_rows(prompt_rows, ids)
    spearman_rows = summarize_spearman(prompt_rows, ids)
    mean_rows = summarize_means(prompt_rows, ids)
    diff_rows, mean_diff_rows = paired_bootstrap_diffs(
        prompt_rows=prompt_rows,
        ids=ids,
        baseline=args.baseline,
        n_boot=args.n_boot,
        seed=args.seed,
    )

    write_csv(
        out_dir / "prompt_sensitivity_paired_rows.csv",
        long_rows,
        [
            "record_id", "prompt", "citation_bin", "citation_bin_midpoint",
            "matched", "hallucinated", "unmatched", *METRICS,
        ],
    )
    write_csv(
        out_dir / "prompt_sensitivity_spearman.csv",
        spearman_rows,
        ["prompt", "metric", "n", "spearman_rho"],
    )
    write_csv(
        out_dir / "prompt_sensitivity_spearman_diffs.csv",
        diff_rows,
        [
            "comparison", "prompt", "baseline", "metric", "n",
            "baseline_rho", "prompt_rho", "rho_diff", "abs_rho_diff",
            "rho_diff_ci_low", "rho_diff_ci_high", "rho_diff_p_two_sided",
            "n_boot",
        ],
    )
    write_csv(
        out_dir / "prompt_sensitivity_mean_metrics.csv",
        mean_rows,
        ["prompt", "n", *[f"{metric}_mean" for metric in METRICS]],
    )
    write_csv(
        out_dir / "prompt_sensitivity_mean_diffs.csv",
        mean_diff_rows,
        [
            "comparison", "prompt", "baseline", "metric", "n",
            "baseline_mean", "prompt_mean", "mean_diff", "abs_mean_diff",
            "mean_diff_ci_low", "mean_diff_ci_high", "mean_diff_p_two_sided",
            "n_boot",
        ],
    )
    write_notes(out_dir / "prompt_sensitivity_significance_notes.md", inputs, ids, args.n_boot, args.baseline)

    print(f"[prompt_sensitivity_significance] Wrote outputs to {out_dir}")
    for row in diff_rows:
        if row["metric"] == "hr2":
            print(
                "[prompt_sensitivity_significance] "
                f"{row['comparison']} HR2 rho diff={row['rho_diff']:.4f} "
                f"CI=[{row['rho_diff_ci_low']:.4f}, {row['rho_diff_ci_high']:.4f}] "
                f"p={row['rho_diff_p_two_sided']:.4g}"
            )


if __name__ == "__main__":
    main()
