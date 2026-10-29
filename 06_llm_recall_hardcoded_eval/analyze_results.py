"""Section 5 results and their appendix tables, from ``release/`` alone.

Figures 3-5 and A4-A5; Tables 2, 3, A18, A24 and A25. Outputs go to
``results/llm_recall_hardcoded_eval/``: one ``.tex`` per table, one ``.pdf`` per
figure, and a ``.json`` with the values behind each.

Usage (from the repository root):
    python 06_llm_recall_hardcoded_eval/analyze_results.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy import stats

from common import (  # noqa: E402
    BIN_LABELS,
    FIELD_KEYS,
    MODELS,
    MODEL_COLORS,
    MODEL_LABELS,
    OUT_DIR,
    apply_style,
    field_label,
    fmt,
    fmt_int,
    load_all_responses,
    load_responses,
    load_sample,
    save_figure,
    save_table,
    write_json,
)
from hallucination_metrics import mean_ci  # noqa: E402

SOURCE = "06_llm_recall_hardcoded_eval/analyze_results.py"


# ---------------------------------------------------------------------------
# per-bin aggregation
# ---------------------------------------------------------------------------

def rate_by_bin(df: pd.DataFrame, metric: str = "hr2") -> pd.DataFrame:
    """Mean rate with a 95% CI per citation bin, in canonical bin order."""
    rows = []
    for label, group in df.groupby("citation_bin", observed=True, sort=False):
        mean, half, n = mean_ci(group[metric])
        rows.append({"citation_bin": str(label), "mean": mean, "ci95": half, "n": n})
    out = pd.DataFrame(rows)
    order = BIN_LABELS
    out["_o"] = out["citation_bin"].map(order.index)
    return out.sort_values("_o").drop(columns="_o").reset_index(drop=True)


def _plot_rate_by_bin(metric: str, ylabel: str, name: str, title: str | None = None):
    apply_style()
    import matplotlib.pyplot as plt

    labels = BIN_LABELS
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    values: dict[str, object] = {"metric": metric, "citation_bins": labels, "models": {}}

    for model in MODELS:
        agg = rate_by_bin(load_responses(model), metric)
        x = np.arange(len(agg))
        ax.errorbar(x, agg["mean"], yerr=agg["ci95"], label=MODEL_LABELS[model],
                    color=MODEL_COLORS[model], marker="o", markersize=4,
                    capsize=3, linewidth=1.6, elinewidth=1.0)
        values["models"][model] = {
            "mean": [float(v) for v in agg["mean"]],
            "ci95": [float(v) for v in agg["ci95"]],
            "n": [int(v) for v in agg["n"]],
        }

    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlabel("Citation count")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return save_figure(fig, name, values, source=SOURCE)


def figure_hr_by_citation() -> None:
    """Figure 3 (HR2) plus its HR1 and HR3 counterparts, Figures A4 and A5."""
    _plot_rate_by_bin("hr2", r"Hallucination rate (HR$_2$)", "fig_hr2_by_citation")
    _plot_rate_by_bin("hr1", r"Hallucination rate (HR$_1$)", "fig_hr1_by_citation")
    _plot_rate_by_bin("hr3", r"Hallucination rate (HR$_3$)", "fig_hr3_by_citation")


# ---------------------------------------------------------------------------
# Table 3: Spearman rho between citation count and HR2, by field and model
# ---------------------------------------------------------------------------

def spearman_by_field(metric: str = "hr2") -> pd.DataFrame:
    """One row per field, one column per model, plus the p-values.

    The correlation is taken against the raw citation count, not the bin
    index, so it is not sensitive to the binning choice.
    """
    rows = []
    for field in FIELD_KEYS:
        row: dict[str, object] = {"field": field}
        for model in MODELS:
            df = load_responses(model)
            sub = df[df["field"] == field][["citations", metric]].dropna()
            rho, p = stats.spearmanr(sub["citations"], sub[metric])
            row[model] = float(rho)
            row[f"{model}_p"] = float(p)
            row[f"{model}_n"] = int(len(sub))
        rows.append(row)
    return pd.DataFrame(rows)


def table_spearman_all_models() -> None:
    table = spearman_by_field("hr2")
    # The paper orders this table alphabetically by display label.
    table = table.assign(_label=table["field"].map(field_label)).sort_values("_label")
    rows = [[r["_label"]] + [fmt(r[m]) for m in MODELS] for _, r in table.iterrows()]
    save_table(
        "spearman_all_models",
        [r"\textbf{Field}", r"\textbf{GPT-4o}", r"\textbf{DeepSeek}", r"\textbf{Claude}"],
        rows, align="lccc", source=SOURCE,
        values={"spearman_hr2_by_field": table.drop(columns="_label").to_dict(orient="records")},
    )


# ---------------------------------------------------------------------------
# Figure 4: field variance radar
# ---------------------------------------------------------------------------

def figure_field_variance_radar() -> None:
    apply_style()
    import matplotlib.pyplot as plt

    fields = FIELD_KEYS
    angles = np.linspace(0, 2 * np.pi, len(fields), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(4.6, 4.6), subplot_kw={"polar": True})
    values: dict[str, object] = {"fields": fields, "models": {}}

    for model in MODELS:
        df = load_responses(model)
        means = [float(df.loc[df["field"] == f, "hr2"].mean()) for f in fields]
        values["models"][model] = means
        closed = means + means[:1]
        ax.plot(angles, closed, color=MODEL_COLORS[model], linewidth=1.6,
                label=MODEL_LABELS[model])
        ax.fill(angles, closed, color=MODEL_COLORS[model], alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([field_label(f) for f in fields], fontsize=8)
    ax.set_ylim(0.75, 1.0)
    ax.set_yticks([0.80, 0.85, 0.90, 0.95, 1.00])
    ax.tick_params(axis="y", labelsize=7)
    ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.12), fontsize=8)
    fig.tight_layout()
    save_figure(fig, "fig_field_variance_radar", values, source=SOURCE)


# ---------------------------------------------------------------------------
# Figure 5: GPT-4o, HR2 by citation bin, one line per field
# ---------------------------------------------------------------------------

def figure_gpt4o_field_by_citation() -> None:
    apply_style()
    import matplotlib.pyplot as plt
    df = load_responses("gpt-4o")
    labels = BIN_LABELS
    cmap = plt.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    values: dict[str, object] = {"model": "gpt-4o", "citation_bins": labels, "fields": {}}

    for i, field in enumerate(FIELD_KEYS):
        agg = rate_by_bin(df[df["field"] == field], "hr2")
        ax.plot(np.arange(len(agg)), agg["mean"], color=cmap(i), linewidth=1.4,
                label=field_label(field))
        values["fields"][field] = {"mean": [float(v) for v in agg["mean"]],
                                   "n": [int(v) for v in agg["n"]]}

    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlabel("Citation count")
    ax.set_ylabel(r"Hallucination rate (HR$_2$)")
    ax.set_ylim(0, 1.02)
    ax.legend(ncol=2, fontsize=8, loc="lower left")
    fig.tight_layout()
    save_figure(fig, "fig_gpt4o_field_by_citation", values, source=SOURCE)


# ---------------------------------------------------------------------------
# Appendix tables
# ---------------------------------------------------------------------------

def table_multi_llm_summary() -> None:
    """Table A24: overall mean +/- SD under HR1, HR2, HR3."""
    rows, values = [], {}
    for model in MODELS:
        df = load_responses(model)
        cells, record = [], {}
        for metric in ("hr1", "hr2", "hr3"):
            mean, sd = float(df[metric].mean()), float(df[metric].std(ddof=1))
            cells.append(rf"{mean:.2f}{{\small$\pm${sd:.2f}}}")
            record[metric] = {"mean": mean, "sd": sd}
        rows.append([MODEL_LABELS[model], *cells])
        values[model] = record
    save_table(
        "multi_llm_summary",
        [r"\textbf{Model}", r"\textbf{HR$_1$}", r"\textbf{HR$_2$}", r"\textbf{HR$_3$}"],
        rows, align="lccc", source=SOURCE, values={"overall_rates": values},
    )


def table_detailed_gpt4o() -> None:
    """Table A25: GPT-4o by field, sorted by mean HR2 descending."""
    df = load_responses("gpt-4o")
    records = []
    for field in FIELD_KEYS:
        sub = df[df["field"] == field]
        rho, p = stats.spearmanr(sub["citations"], sub["hr2"])
        records.append({
            "field": field, "papers": int(len(sub)),
            "mean_authors": float(sub["n_authors"].mean()),
            "hr2": float(sub["hr2"].mean()), "sd": float(sub["hr2"].std(ddof=1)),
            "rho": float(rho), "p": float(p),
        })
    records.sort(key=lambda r: -r["hr2"])

    rho_all, p_all = stats.spearmanr(df["citations"], df["hr2"])
    overall = {
        "field": "Overall", "papers": int(len(df)),
        "mean_authors": float(df["n_authors"].mean()),
        "hr2": float(df["hr2"].mean()), "sd": float(df["hr2"].std(ddof=1)),
        "rho": float(rho_all), "p": float(p_all),
    }

    rows = [[field_label(r["field"]), fmt_int(r["papers"]), fmt(r["mean_authors"], 1),
             fmt(r["hr2"]), fmt(r["sd"]), fmt(r["rho"])] for r in records]
    rows.append([r"\textbf{Overall}", rf"\textbf{{{overall['papers']:,}}}",
                 rf"\textbf{{{overall['mean_authors']:.1f}}}",
                 rf"\textbf{{{overall['hr2']:.3f}}}", rf"\textbf{{{overall['sd']:.3f}}}",
                 rf"\textbf{{{overall['rho']:.3f}}}"])
    save_table(
        "detailed_gpt_4o",
        [r"\textbf{Field}", r"\textbf{Pap.}", r"\textbf{Aut.}", r"\textbf{HR$_2$}",
         r"$\sigma$", r"$\rho$"],
        rows, align="lrrrrr", source=SOURCE, midrule_before={len(rows) - 1},
        values={"gpt4o_by_field": records, "overall": overall},
    )


def table_field_sample_sizes() -> None:
    """Table 2: the stratified sample, per field.

    Reported over the released sample. The published Total row (3.8 authors,
    457 citations) does not follow from the per-field rows; see
    the paper.
    """
    sample = load_sample()
    records = []
    for field in FIELD_KEYS:
        sub = sample[sample["field"] == field]
        records.append({"field": field, "papers": int(len(sub)),
                        "mean_authors": float(sub["n_authors"].mean()),
                        "mean_citations": float(sub["citations"].mean())})
    records.sort(key=lambda r: -r["papers"])
    total = {"papers": int(len(sample)),
             "mean_authors": float(sample["n_authors"].mean()),
             "mean_citations": float(sample["citations"].mean())}

    rows = [[field_label(r["field"]), fmt_int(r["papers"]), fmt(r["mean_authors"], 1),
             fmt_int(r["mean_citations"])] for r in records]
    rows.append([r"\textbf{Total}", rf"\textbf{{{total['papers']:,}}}",
                 rf"\textbf{{{total['mean_authors']:.1f}}}",
                 rf"\textbf{{{round(total['mean_citations']):,}}}"])
    save_table(
        "field_sample_sizes",
        [r"\textbf{Field}", r"\textbf{Papers}", r"\textbf{Avg Authors}",
         r"\textbf{Avg Citations}"],
        rows, align="lrrr", source=SOURCE, midrule_before={len(rows) - 1},
        values={"per_field": records, "total": total},
    )


def table_metric_correlations() -> None:
    """Table A18: Pearson correlations among HR1, HR2, HR3 for GPT-4o."""
    df = load_responses("gpt-4o")[["hr1", "hr2", "hr3"]].dropna()
    names = ["hr1", "hr2", "hr3"]
    matrix, pvals = {}, {}
    for a in names:
        for b in names:
            r, p = stats.pearsonr(df[a], df[b])
            matrix[f"{a}_{b}"] = float(r)
            pvals[f"{a}_{b}"] = float(p)

    rows = []
    for i, a in enumerate(names):
        cells = []
        for j, b in enumerate(names):
            cells.append(fmt(matrix[f"{a}_{b}"], 3) if j <= i else "---")
        rows.append([rf"HR$_{a[-1]}$", *cells])
    save_table(
        "metric_correlations",
        ["", r"\textbf{HR}$_1$", r"\textbf{HR}$_2$", r"\textbf{HR}$_3$"],
        rows, align="lrrr", source=SOURCE,
        values={"pearson": matrix, "p": pvals, "n": int(len(df))},
    )


def headline_numbers() -> dict:
    """The Section 5 numbers quoted in the abstract, intro, and conclusion."""
    first, last = BIN_LABELS[0], BIN_LABELS[-1]
    out: dict[str, object] = {"lowest_bin": first, "highest_bin": last, "models": {}}
    for model in MODELS:
        agg = rate_by_bin(load_responses(model), "hr2").set_index("citation_bin")
        out["models"][model] = {
            "hr2_lowest_bin": float(agg.loc[first, "mean"]),
            "hr2_highest_bin": float(agg.loc[last, "mean"]),
            "hr2_overall": float(load_responses(model)["hr2"].mean()),
        }
    all_df = load_all_responses()
    rho, p = stats.spearmanr(all_df["citations"], all_df["hr2"])
    out["pooled_spearman_citations_hr2"] = {"rho": float(rho), "p": float(p),
                                            "n": int(len(all_df))}
    return out


def run_all() -> None:
    figure_hr_by_citation()
    figure_field_variance_radar()
    figure_gpt4o_field_by_citation()
    table_spearman_all_models()
    table_multi_llm_summary()
    table_detailed_gpt4o()
    table_field_sample_sizes()
    table_metric_correlations()
    write_json(OUT_DIR / "headline_numbers.json", headline_numbers(), source=SOURCE)


if __name__ == "__main__":
    run_all()
