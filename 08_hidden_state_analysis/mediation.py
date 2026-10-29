"""mediation: does P1 prevalence mediate the citation-hallucination link?

Table A42, the 67.3% / 51.9% proportion-mediated claim.

Usage
-----
  python 08_hidden_state_analysis/mediation.py \
    --predictions_csv 08_hidden_state_analysis/release/p1_prevalence_predictions.csv \
    --out_dir results/hidden_state_analysis/probing \
    --sims 2000 \
    --boot
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import tempfile
from pathlib import Path


PRIMARY_MEDIATOR = "p1_prevalence_pred_oof"
CONTROL_MEDIATORS = [
    "metadata_prevalence_pred_oof",
    "combined_prevalence_pred_oof",
]
OUTCOME = "hr2"
TREATMENT = "log_citation"


R_SCRIPT = r"""
args <- commandArgs(trailingOnly = TRUE)
predictions_csv <- args[[1]]
summary_csv <- args[[2]]
notes_md <- args[[3]]
sims <- as.integer(args[[4]])
boot_flag <- args[[5]] == "TRUE"
manual_only <- args[[6]] == "TRUE"

required_cols <- c(
  "model",
  "paper_id",
  "log_citation",
  "hr2",
  "p1_prevalence_pred_oof",
  "metadata_prevalence_pred_oof",
  "combined_prevalence_pred_oof"
)

df <- read.csv(predictions_csv, stringsAsFactors = FALSE)
missing <- setdiff(required_cols, names(df))
if (length(missing) > 0) {
  stop(paste("predictions CSV missing required columns:", paste(missing, collapse = ", ")))
}

has_mediation <- requireNamespace("mediation", quietly = TRUE)
if (!has_mediation && !manual_only) {
  stop(paste(
    "R package 'mediation' is not installed.",
    "Install it with install.packages('mediation') or rerun with --manual_only."
  ))
}

mediators <- c(
  "p1_prevalence_pred_oof",
  "metadata_prevalence_pred_oof",
  "combined_prevalence_pred_oof"
)

clip01 <- function(x) {
  pmin(pmax(x, 1e-5), 1 - 1e-5)
}

empty_row <- function() {
  data.frame(
    model = character(),
    method = character(),
    outcome_model = character(),
    mediator = character(),
    control_value = numeric(),
    treat_value = numeric(),
    acme = numeric(),
    acme_lo = numeric(),
    acme_hi = numeric(),
    ade = numeric(),
    ade_lo = numeric(),
    ade_hi = numeric(),
    total_effect = numeric(),
    total_lo = numeric(),
    total_hi = numeric(),
    prop_mediated = numeric(),
    prop_lo = numeric(),
    prop_hi = numeric(),
    sims = integer(),
    boot = logical(),
    controls = character(),
    status = character(),
    stringsAsFactors = FALSE
  )
}

row_from_values <- function(model, method, outcome_model, mediator, x0, x1,
                            acme, acme_ci, ade, ade_ci, total, total_ci,
                            prop, prop_ci, status) {
  data.frame(
    model = model,
    method = method,
    outcome_model = outcome_model,
    mediator = mediator,
    control_value = x0,
    treat_value = x1,
    acme = acme,
    acme_lo = acme_ci[[1]],
    acme_hi = acme_ci[[2]],
    ade = ade,
    ade_lo = ade_ci[[1]],
    ade_hi = ade_ci[[2]],
    total_effect = total,
    total_lo = total_ci[[1]],
    total_hi = total_ci[[2]],
    prop_mediated = prop,
    prop_lo = prop_ci[[1]],
    prop_hi = prop_ci[[2]],
    sims = sims,
    boot = boot_flag,
    controls = "none",
    status = status,
    stringsAsFactors = FALSE
  )
}

manual_linear <- function(dat, mediator, model_name, x0, x1) {
  med_formula <- as.formula(paste(mediator, "~ log_citation"))
  out_formula <- as.formula(paste("hr2 ~ log_citation +", mediator))
  total_formula <- hr2 ~ log_citation

  total_fit <- lm(total_formula, data = dat)
  med_fit <- lm(med_formula, data = dat)
  out_fit <- lm(out_formula, data = dat)

  total_slope <- coef(total_fit)[["log_citation"]]
  a <- coef(med_fit)[["log_citation"]]
  b <- coef(out_fit)[[mediator]]
  direct_slope <- coef(out_fit)[["log_citation"]]

  delta_x <- x1 - x0
  acme <- a * b * delta_x
  ade <- direct_slope * delta_x
  total <- total_slope * delta_x
  prop <- acme / total

  n <- nrow(dat)
  boot_mat <- replicate(sims, {
    idx <- sample.int(n, n, replace = TRUE)
    bdat <- dat[idx, ]
    b_total <- coef(lm(total_formula, data = bdat))[["log_citation"]]
    b_a <- coef(lm(med_formula, data = bdat))[["log_citation"]]
    b_out <- lm(out_formula, data = bdat)
    b_b <- coef(b_out)[[mediator]]
    b_direct <- coef(b_out)[["log_citation"]]
    b_acme <- b_a * b_b * delta_x
    b_ade <- b_direct * delta_x
    b_total_effect <- b_total * delta_x
    c(
      acme = b_acme,
      ade = b_ade,
      total = b_total_effect,
      prop = b_acme / b_total_effect
    )
  })

  row_from_values(
    model_name,
    "manual_linear_decomposition",
    "lm",
    mediator,
    x0,
    x1,
    acme,
    as.numeric(quantile(boot_mat["acme", ], c(0.025, 0.975), na.rm = TRUE)),
    ade,
    as.numeric(quantile(boot_mat["ade", ], c(0.025, 0.975), na.rm = TRUE)),
    total,
    as.numeric(quantile(boot_mat["total", ], c(0.025, 0.975), na.rm = TRUE)),
    prop,
    as.numeric(quantile(boot_mat["prop", ], c(0.025, 0.975), na.rm = TRUE)),
    "ok"
  )
}

manual_fractional_logit <- function(dat, mediator, model_name, x0, x1) {
  dat$hr2_clipped <- clip01(dat$hr2)
  med_fit <- lm(as.formula(paste(mediator, "~ log_citation")), data = dat)
  out_fit <- suppressWarnings(glm(
    as.formula(paste("hr2_clipped ~ log_citation +", mediator)),
    data = dat,
    family = binomial("logit")
  ))

  beta_m <- coef(med_fit)[["log_citation"]]
  m0 <- dat[[mediator]] + beta_m * (x0 - dat$log_citation)
  m1 <- dat[[mediator]] + beta_m * (x1 - dat$log_citation)

  pred_mean <- function(x, m) {
    nd <- dat
    nd$log_citation <- x
    nd[[mediator]] <- m
    mean(predict(out_fit, newdata = nd, type = "response"))
  }

  y_x0_m0 <- pred_mean(x0, m0)
  y_x1_m0 <- pred_mean(x1, m0)
  y_x1_m1 <- pred_mean(x1, m1)

  acme <- y_x1_m1 - y_x1_m0
  ade <- y_x1_m0 - y_x0_m0
  total <- y_x1_m1 - y_x0_m0
  prop <- acme / total

  n <- nrow(dat)
  boot_mat <- replicate(sims, {
    idx <- sample.int(n, n, replace = TRUE)
    bdat <- dat[idx, ]
    bdat$hr2_clipped <- clip01(bdat$hr2)
    b_med_fit <- lm(as.formula(paste(mediator, "~ log_citation")), data = bdat)
    b_out_fit <- suppressWarnings(glm(
      as.formula(paste("hr2_clipped ~ log_citation +", mediator)),
      data = bdat,
      family = binomial("logit")
    ))
    b_beta_m <- coef(b_med_fit)[["log_citation"]]
    b_m0 <- bdat[[mediator]] + b_beta_m * (x0 - bdat$log_citation)
    b_m1 <- bdat[[mediator]] + b_beta_m * (x1 - bdat$log_citation)
    b_pred_mean <- function(x, m) {
      nd <- bdat
      nd$log_citation <- x
      nd[[mediator]] <- m
      mean(predict(b_out_fit, newdata = nd, type = "response"))
    }
    b_y_x0_m0 <- b_pred_mean(x0, b_m0)
    b_y_x1_m0 <- b_pred_mean(x1, b_m0)
    b_y_x1_m1 <- b_pred_mean(x1, b_m1)
    b_acme <- b_y_x1_m1 - b_y_x1_m0
    b_ade <- b_y_x1_m0 - b_y_x0_m0
    b_total <- b_y_x1_m1 - b_y_x0_m0
    c(acme = b_acme, ade = b_ade, total = b_total, prop = b_acme / b_total)
  })

  row_from_values(
    model_name,
    "manual_counterfactual_fractional_logit",
    "glm_binomial_logit",
    mediator,
    x0,
    x1,
    acme,
    as.numeric(quantile(boot_mat["acme", ], c(0.025, 0.975), na.rm = TRUE)),
    ade,
    as.numeric(quantile(boot_mat["ade", ], c(0.025, 0.975), na.rm = TRUE)),
    total,
    as.numeric(quantile(boot_mat["total", ], c(0.025, 0.975), na.rm = TRUE)),
    prop,
    as.numeric(quantile(boot_mat["prop", ], c(0.025, 0.975), na.rm = TRUE)),
    "ok"
  )
}

mediate_linear <- function(dat, mediator, model_name, x0, x1) {
  med_fit <- lm(as.formula(paste(mediator, "~ log_citation")), data = dat)
  out_fit <- lm(as.formula(paste("hr2 ~ log_citation +", mediator)), data = dat)
  med <- mediation::mediate(
    med_fit,
    out_fit,
    treat = "log_citation",
    mediator = mediator,
    control.value = x0,
    treat.value = x1,
    boot = boot_flag,
    sims = sims
  )

  row_from_values(
    model_name,
    "mediation_package",
    "lm",
    mediator,
    x0,
    x1,
    med$d.avg,
    med$d.avg.ci,
    med$z.avg,
    med$z.avg.ci,
    med$tau.coef,
    med$tau.ci,
    med$n.avg,
    med$n.avg.ci,
    "ok"
  )
}

set.seed(42)
rows <- empty_row()
for (model_name in sort(unique(df$model))) {
  dat <- df[df$model == model_name, ]
  dat <- dat[complete.cases(dat[, required_cols]), ]
  x0 <- as.numeric(quantile(dat$log_citation, 0.25, names = FALSE))
  x1 <- as.numeric(quantile(dat$log_citation, 0.75, names = FALSE))

  for (mediator in mediators) {
    if (has_mediation && !manual_only) {
      rows <- rbind(rows, mediate_linear(dat, mediator, model_name, x0, x1))
    }
    rows <- rbind(rows, manual_linear(dat, mediator, model_name, x0, x1))
  }

  rows <- rbind(rows, manual_fractional_logit(dat, "p1_prevalence_pred_oof", model_name, x0, x1))
  rows <- rbind(rows, manual_fractional_logit(dat, "combined_prevalence_pred_oof", model_name, x0, x1))
}

write.csv(rows, summary_csv, row.names = FALSE)

notes <- c(
  "# Modern Mediation Analysis",
  "",
  paste0("Input: `", predictions_csv, "`."),
  paste0("Simulations/bootstrap resamples: ", sims, "."),
  paste0("R mediation package available: ", has_mediation, "."),
  paste0("Manual-only mode: ", manual_only, "."),
  "",
  "Primary estimand: mediation::mediate ACME/ADE/total/proportion mediated for a continuous treatment contrast from the empirical Q25 to Q75 of log(citations + 1), computed separately by model.",
  "",
  "Primary mediator: `p1_prevalence_pred_oof`, the out-of-fold P1-decoded prevalence score from prevalence_decoding.",
  "",
  "Control checks: metadata-only decoded prevalence and combined metadata+P1 decoded prevalence.",
  "",
  "Interpretation: these are mediation-style decompositions using the Imai/Keele/Tingley framework, but the design is observational. The estimates should not be described as identified causal mediation effects unless the paper also defends the required sequential ignorability assumptions.",
  "",
  "The manual linear rows are included to compare against earlier path-decomposition numbers. The manual fractional-logit rows are bounded-outcome robustness checks and are not the primary reported mediate() result."
)
writeLines(notes, notes_md)
"""


def run_r_analysis(
    predictions_csv: Path,
    summary_csv: Path,
    notes_md: Path,
    sims: int,
    boot: bool,
    manual_only: bool,
) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".R", delete=False, encoding="utf-8") as handle:
        handle.write(R_SCRIPT)
        script_path = Path(handle.name)
    try:
        cmd = [
            "Rscript",
            str(script_path),
            str(predictions_csv),
            str(summary_csv),
            str(notes_md),
            str(sims),
            "TRUE" if boot else "FALSE",
            "TRUE" if manual_only else "FALSE",
        ]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            raise SystemExit(
                "[mediation] R mediation analysis failed. If the error says the "
                "'mediation' package is missing, install it in that R "
                "environment with install.packages('mediation') or rerun with "
                "--manual_only for diagnostic decompositions."
            ) from exc
    finally:
        try:
            script_path.unlink()
        except FileNotFoundError:
            pass


def summarize_primary(summary_csv: Path) -> None:
    with summary_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    primary = [
        row
        for row in rows
        if row["mediator"] == PRIMARY_MEDIATOR
        and row["method"] in {"mediation_package", "manual_linear_decomposition"}
        and row["outcome_model"] == "lm"
    ]
    print(f"[mediation] Wrote {summary_csv}")
    for row in primary:
        prop = 100.0 * float(row["prop_mediated"])
        lo = 100.0 * float(row["prop_lo"])
        hi = 100.0 * float(row["prop_hi"])
        print(
            "[mediation] "
            f"{row['model']} {row['method']} P1 proportion mediated: "
            f"{prop:.1f}% [{lo:.1f}, {hi:.1f}]"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="mediation: modern mediation analysis from prevalence_decoding P1 predictions"
    )
    parser.add_argument(
        "--predictions_csv",
        default="08_hidden_state_analysis/release/p1_prevalence_predictions.csv",
        help="prevalence_decoding per-paper P1 prevalence predictions CSV",
    )
    parser.add_argument(
        "--out_dir",
        default="results/hidden_state_analysis/probing",
        help="Output directory for modern mediation files",
    )
    parser.add_argument("--sims", type=int, default=2000, help="Simulation/bootstrap count")
    parser.add_argument("--boot", action="store_true", help="Use nonparametric bootstrap in mediation::mediate")
    parser.add_argument(
        "--manual_only",
        action="store_true",
        help="Skip mediation::mediate and write only base-R diagnostic decompositions",
    )
    args = parser.parse_args()

    predictions_csv = Path(args.predictions_csv).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    summary_csv = out_dir / "modern_mediation_summary.csv"
    notes_md = out_dir / "modern_mediation_notes.md"

    if not predictions_csv.exists():
        raise FileNotFoundError(f"Missing predictions CSV: {predictions_csv}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[mediation] Reading {predictions_csv}")
    run_r_analysis(
        predictions_csv=predictions_csv,
        summary_csv=summary_csv,
        notes_md=notes_md,
        sims=args.sims,
        boot=args.boot,
        manual_only=args.manual_only,
    )
    summarize_primary(summary_csv)
    print(f"[mediation] Wrote {notes_md}")


if __name__ == "__main__":
    main()
