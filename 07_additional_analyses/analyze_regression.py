"""Multivariate models (Section 8, Table A30): does citation count survive controls?

Three specifications on the GPT-4o responses: a fractional logit on HR2 (the
outcome is a proportion, so this is a quasi-likelihood fit with model-based
standard errors, matching the published z-values), the same on HR4, and a
linear mixed-effects model with a random intercept per field to absorb
unobserved field-level confounders.

Covariates throughout: log2(citations+1), author count, title length, year, and
field dummies with Biology as baseline.

Usage (from the repository root):
    python 07_additional_analyses/analyze_regression.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from shared import (  # noqa: E402
    fmt,
    load_responses,
    save_table,
)

SOURCE = "07_additional_analyses/analyze_regression.py"

BASELINE_FIELD = "biology"
CITATION_TERM = "log2_citations"


def design_matrix(model: str = "gpt-4o", metric: str = "hr2") -> pd.DataFrame:
    """Assemble the modelling frame, dropping rows with an undefined outcome.

    HR4 is undefined when the model produced no names at all, so the HR4 fit
    runs on slightly fewer rows than the HR2 fit. The count is reported rather
    than hidden.
    """
    df = load_responses(model)
    out = pd.DataFrame({
        "y": df[metric],
        CITATION_TERM: np.log2(df["citations"] + 1),
        "n_authors": df["n_authors"],
        "title_length": df["title_length"],
        "year": pd.to_numeric(df["year"], errors="coerce"),
        "field": df["field"],
    }).dropna()
    # Bound the outcome strictly inside (0, 1) is NOT done: statsmodels'
    # binomial GLM accepts the endpoints, and clipping would bias the
    # coefficient. Fractional logit is well defined at 0 and 1.
    return out


def _formula(metric_terms: list[str]) -> str:
    return "y ~ " + " + ".join(metric_terms) + f" + C(field, Treatment(reference='{BASELINE_FIELD}'))"


def fit_fractional_logit(model: str = "gpt-4o", metric: str = "hr2") -> dict:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    data = design_matrix(model, metric)
    fit = smf.glm(
        _formula([CITATION_TERM, "n_authors", "title_length", "year"]),
        data=data,
        family=sm.families.Binomial(),
    ).fit()

    return {
        "spec": "GLM, binomial logit",
        "metric": metric,
        "n": int(fit.nobs),
        "coefficients": {
            name: {"beta": float(fit.params[name]), "se": float(fit.bse[name]),
                   "z": float(fit.tvalues[name]), "p": float(fit.pvalues[name])}
            for name in fit.params.index
        },
        "citation": {
            "beta": float(fit.params[CITATION_TERM]),
            "z": float(fit.tvalues[CITATION_TERM]),
            "p": float(fit.pvalues[CITATION_TERM]),
            # exp(beta) - 1 is the change in odds per doubling of citations.
            "odds_ratio_per_doubling": float(np.exp(fit.params[CITATION_TERM])),
        },
        "summary_text": str(fit.summary()),
    }


def fit_mixed_effects(model: str = "gpt-4o", metric: str = "hr2") -> dict:
    """Random intercept per field, on the probability scale.

    The coefficient is therefore small (a change in HR2 per doubling of
    citations, not in log-odds) and is marked with a dagger in Table A30.
    """
    import statsmodels.formula.api as smf

    data = design_matrix(model, metric)
    fit = smf.mixedlm(
        f"y ~ {CITATION_TERM} + n_authors + title_length + year",
        data=data, groups=data["field"],
    ).fit(method="lbfgs")

    var_field = float(fit.cov_re.iloc[0, 0])
    var_resid = float(fit.scale)
    icc = var_field / (var_field + var_resid) if (var_field + var_resid) else float("nan")

    return {
        "spec": "Mixed-effects LMM",
        "metric": metric,
        "n": int(fit.nobs),
        "n_groups": int(data["field"].nunique()),
        "citation": {
            "beta": float(fit.params[CITATION_TERM]),
            "z": float(fit.tvalues[CITATION_TERM]),
            "p": float(fit.pvalues[CITATION_TERM]),
        },
        "var_field": var_field,
        "var_residual": var_resid,
        "icc": float(icc),
        "summary_text": str(fit.summary()),
    }


def table_regression_summary(model: str = "gpt-4o") -> dict:
    fits = [
        fit_fractional_logit(model, "hr2"),
        fit_fractional_logit(model, "hr4"),
        fit_mixed_effects(model, "hr2"),
    ]
    rows = []
    for f in fits:
        c = f["citation"]
        dagger = r"$^{\dagger}$" if f["spec"].startswith("Mixed") else ""
        p = r"$<0.001$" if c["p"] < 1e-3 else f"${c['p']:.3f}$"
        rows.append([
            f["spec"], rf"HR$_{f['metric'][-1]}$",
            f"${c['beta']:.3f}${dagger}", f"${c['z']:.1f}$", p,
        ])
    save_table(
        "regression_summary",
        [r"\textbf{Model}", r"\textbf{Metric}", r"$\boldsymbol{\beta}$",
         r"$\boldsymbol{z}$", r"$\boldsymbol{p}$"],
        rows, align="llccc", source=SOURCE,
        values={"model": model, "fits": [{k: v for k, v in f.items() if k != "summary_text"}
                                         for f in fits]},
    )
    return {"fits": fits}


def table_full_coefficients(model: str = "gpt-4o", metric: str = "hr2") -> None:
    """Every coefficient of the HR2 fractional logit, for the record."""
    fit = fit_fractional_logit(model, metric)
    rows = []
    for name, c in fit["coefficients"].items():
        pretty = (name.replace("C(field, Treatment(reference='biology'))[T.", "field: ")
                      .replace("]", "").replace("_", r"\_"))
        rows.append([pretty, fmt(c["beta"]), fmt(c["se"]), fmt(c["z"], 1),
                     r"$<0.001$" if c["p"] < 1e-3 else fmt(c["p"], 3)])
    save_table(
        "regression_full_coefficients",
        [r"\textbf{Term}", r"$\beta$", r"SE", r"$z$", r"$p$"],
        rows, align="lrrrr", source=SOURCE,
    )


def run_all(model: str = "gpt-4o") -> dict:
    out = table_regression_summary(model)
    table_full_coefficients(model)
    return out


if __name__ == "__main__":
    run_all()
