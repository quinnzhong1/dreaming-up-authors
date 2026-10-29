"""Ethnicity of hallucinated names (Section 8, Table A28, Figures A7-A8).

The most methodologically delicate analysis in the paper. ``ethnicseer`` maps a
surname to one of twelve coarse categories from character n-grams -- a noisy
signal that captures neither self-identification nor names falling between
categories. Nothing here labels a person: every output is an aggregate over
hundreds of thousands of names, and no per-name label is written to disk.

Reports, per category, the ratio of its share among hallucinated names to its
share in the dataset. Read Cramer's V, not the p-value; with 700,000 names
almost anything reaches significance.

Both counting modes ship because either alone misleads in the opposite
direction: ``population`` shows almost no effect (V ~ 0.03), ``diversity`` a
larger one (V ~ 0.10). **Each needs its own baseline.** Diversity must compare
against *distinct* dataset surnames -- using occurrence-weighted counts instead
inverts the result.

Needs ``pip install ethnicseer==0.1.2 "setuptools<81"`` (ethnicseer's pickled
model expects scikit-learn 1.5.2). Classifying ~730,000 surnames takes about
six minutes.

Usage (from the repository root):
    python 07_additional_analyses/analyze_ethnicity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from scipy import stats

from shared import (  # noqa: E402
    MODELS,
    MODEL_COLORS,
    MODEL_LABELS,
    apply_style,
    fmt,
    save_figure,
    save_table,
)
from name_frequency import read_counts  # noqa: E402

import config  # noqa: E402  (on sys.path via shared)

SOURCE = "07_additional_analyses/analyze_ethnicity.py"

ETHNICITY_LABELS = {
    "eng": "English/Western", "chi": "Chinese", "ger": "German", "jap": "Japanese",
    "ind": "Indian", "ita": "Italian", "frn": "French", "spa": "Spanish",
    "rus": "Russian", "mea": "Middle Eastern", "kor": "Korean", "vie": "Vietnamese",
}
ETHNICITY_ORDER = tuple(ETHNICITY_LABELS)
MODES = ("population", "diversity")
N_COMPARISONS = len(MODELS)
ALPHA = 0.05


def bonferroni_alpha() -> float:
    return ALPHA / N_COMPARISONS


def classify(names: list[str]) -> dict[str, str]:
    """Surname -> ethnicity code, via ``ethnicseer``.

    Raises with an actionable message if the optional dependency is absent,
    rather than silently skipping the analysis.
    """
    try:
        from ethnicseer import EthnicClassifier
    except ImportError as exc:
        raise ImportError(
            "ethnicseer is required for Table A28. Install it with "
            "`pip install ethnicseer==0.1.2 'setuptools<81'`. It is optional because it "
            "is the only dependency in this project that infers a protected attribute."
        ) from exc

    classifier = EthnicClassifier.load_pretrained_model()
    return dict(zip(names, classifier.classify_names(names), strict=True))


def aggregate(counts: dict[str, int], labels: dict[str, str],
              mode: str = "population") -> dict[str, int]:
    """Collapse surname counts into per-ethnicity totals.

    ``population`` sums the counts. ``diversity`` counts each distinct surname
    once, whatever its count -- so the same tally can serve as either the
    hallucination side or the baseline side of a diversity comparison.
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
    totals = dict.fromkeys(ETHNICITY_ORDER, 0)
    for surname, count in counts.items():
        code = labels.get(surname)
        if code in totals and count > 0:
            totals[code] += count if mode == "population" else 1
    return totals


def compare(dataset: dict[str, int], hallucinated: dict[str, int]) -> dict:
    """Chi-square goodness of fit plus Cramer's V and per-category ratios.

    Categories absent from the dataset baseline are dropped rather than given
    an expected count of zero, which would make the statistic undefined.
    """
    codes = [c for c in ETHNICITY_ORDER if dataset.get(c, 0) > 0]
    observed = np.array([hallucinated.get(c, 0) for c in codes], dtype=float)
    baseline = np.array([dataset[c] for c in codes], dtype=float)
    expected = baseline / baseline.sum() * observed.sum()

    chi2, p = stats.chisquare(observed, expected)
    n = observed.sum()
    # For a goodness-of-fit test the Cramer's V denominator is n * (k - 1).
    cramers_v = float(np.sqrt(chi2 / (n * (len(codes) - 1)))) if n else float("nan")

    obs_share = observed / observed.sum() if observed.sum() else observed
    exp_share = baseline / baseline.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.where(exp_share > 0, obs_share / exp_share, np.nan)

    return {
        "codes": codes,
        "chi2": float(chi2), "p": float(p), "cramers_v": cramers_v,
        "n_hallucinated": int(n),
        "significant_bonferroni": bool(p < bonferroni_alpha()),
        "per_ethnicity": {
            code: {"dataset_share": float(exp_share[i]),
                   "hallucinated_share": float(obs_share[i]),
                   "ratio": float(ratios[i]),
                   "dataset_count": int(baseline[i]),
                   "hallucinated_count": int(observed[i])}
            for i, code in enumerate(codes)
        },
    }


def _artifacts() -> Path:
    return config.NAME_FREQUENCY_RELEASE_DIR


def load_tallies() -> tuple[dict[str, int], dict[str, dict[str, dict[str, int]]]]:
    """Read the shipped surname tallies."""
    base = _artifacts()
    dataset = read_counts(base / "dataset_surnames.json.gz")
    per_model = {}
    for model in MODELS:
        per_model[model] = {
            mode: read_counts(base / f"{model}_hallucinated_{mode}.json.gz")
            for mode in MODES
        }
    return dataset, per_model


def run_all() -> dict:
    dataset, per_model = load_tallies()

    surnames = sorted(
        set(dataset)
        | {s for m in per_model.values() for c in m.values() for s in c}
    )
    labels = classify(surnames)
    # One baseline per mode: occurrence-weighted for population, distinct
    # surnames for diversity. See the module docstring.
    baselines = {mode: aggregate(dataset, labels, mode) for mode in MODES}

    results: dict = {"n_surnames_classified": len(surnames),
                     "bonferroni_alpha": bonferroni_alpha(),
                     "dataset_totals": baselines, "models": {}}
    for model, tallies in per_model.items():
        results["models"][model] = {
            mode: compare(baselines[mode], aggregate(tallies[mode], labels, mode))
            for mode in MODES
        }

    rows = [[MODEL_LABELS[m],
             fmt(results["models"][m]["population"]["cramers_v"]),
             fmt(results["models"][m]["diversity"]["cramers_v"])]
            for m in MODELS]
    save_table(
        "ethnicity_bias",
        [r"\textbf{Model}", r"\textbf{Population}", r"\textbf{Diversity}"],
        rows, align="lcc", source=SOURCE, values=results,
    )

    for mode in MODES:
        _figure_ratios(results, mode)
    return results


def _figure_ratios(results: dict, mode: str) -> None:
    apply_style()
    import matplotlib.pyplot as plt

    codes = results["models"][MODELS[0]][mode]["codes"]
    x = np.arange(len(codes))
    width = 0.8 / len(MODELS)

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    for i, model in enumerate(MODELS):
        per = results["models"][model][mode]["per_ethnicity"]
        heights = [per[c]["ratio"] for c in codes]
        ax.bar(x + (i - (len(MODELS) - 1) / 2) * width, heights, width,
               color=MODEL_COLORS[model], label=MODEL_LABELS[model])
    ax.axhline(1.0, color="0.3", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([ETHNICITY_LABELS[c] for c in codes], rotation=45,
                       ha="right", fontsize=7)
    ax.set_ylabel("Hallucinated share / dataset share")
    ax.set_title(f"{mode.capitalize()} analysis", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_figure(fig, f"fig_ethnicity_ratio_{mode}",
                {"mode": mode, "models": {m: results["models"][m][mode] for m in MODELS}},
                source=SOURCE)


if __name__ == "__main__":
    run_all()
