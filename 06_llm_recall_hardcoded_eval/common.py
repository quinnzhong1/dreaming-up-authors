"""Shared constants, loaders, and output helpers for the hard-coded evaluation.

Module 07 imports this file too (via ``sys.path``), so "the data" means
one thing everywhere:

* ``load_sample()`` reads the released 9,108-paper evaluation sample.
* ``load_responses(model)`` reads one model's released answers and recomputes
  HR1-HR5 from the stored (M, H, U) counts on every load, so a metric
  definition cannot go stale in a cached column.

Every table is written as an ``\\input``-able LaTeX tabular with a provenance
header, and every figure writes a PDF plus the values it plots as JSON.
"""

import gzip
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from hallucination_metrics import ALL_METRICS, perfect_recall  # noqa: E402

RANDOM_SEED = 42

MODELS: tuple[str, ...] = ("gpt-4o", "deepseek-r1", "claude-sonnet-4.5")
MODEL_LABELS: dict[str, str] = {
    "gpt-4o": "GPT-4o",
    "deepseek-r1": "DeepSeek-R1",
    "claude-sonnet-4.5": "Claude Sonnet 4.5",
}
# Colour per model, identical across every figure in the paper.
MODEL_COLORS: dict[str, str] = {
    "gpt-4o": "#1f77b4",
    "deepseek-r1": "#d62728",
    "claude-sonnet-4.5": "#2ca02c",
}

# The eight fields in the order used by every table and figure of modules 06-08,
# with their display labels. The order differs from config.FIELDS on purpose:
# it fixes line colours and radar-axis order in the published figures.
FIELD_LABELS: dict[str, str] = {
    "medicine": "Medicine",
    "biology": "Biology",
    "chemistry": "Chemistry",
    "computer science": "Computer Sci.",
    "materials science": "Materials Sci.",
    "physics": "Physics",
    "psychology": "Psychology",
    "mathematics": "Mathematics",
}
FIELD_KEYS: list[str] = list(FIELD_LABELS)
BIN_LABELS: list[str] = config.CITATION_BIN_LABELS

# A ground-truth author counts as retrieved only if both the last name and the
# first initial match (Section 4.5): accept matcher levels L1-L4, reject L5.
MATCH_LEVEL_THRESHOLD = 4

OUT_DIR: Path = config.HARDCODED_EVAL_DIR


# ---------------------------------------------------------------------------
# bins and fields
# ---------------------------------------------------------------------------

def bin_label(citations: int) -> str:
    """Map a citation count to its bin label (right-inclusive edges in config.py)."""
    if citations < 0:
        raise ValueError(f"negative citation count: {citations}")
    for label, hi in zip(BIN_LABELS, config.CITATION_BIN_EDGES[1:], strict=True):
        if citations <= hi:
            return label
    raise ValueError(f"no bin covers {citations} citations")


def bin_index(label: str) -> int:
    """Position of a bin label in the canonical ordering, for sorting."""
    return BIN_LABELS.index(label)


def field_label(key: str) -> str:
    """Field key -> display label used in figures and LaTeX tables."""
    return FIELD_LABELS[key]


# ---------------------------------------------------------------------------
# loading the released artifacts
# ---------------------------------------------------------------------------

def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. The released artifacts are tracked in git under "
            f"{config.HARDCODED_EVAL_MODULE.name}/release/."
        )
    return path


def _add_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Attach HR1..HR5 and the binary perfect-recall flag from (m, h, u)."""
    for name, fn in ALL_METRICS.items():
        df[name] = [fn(m, h, u) for m, h, u in zip(df["m"], df["h"], df["u"], strict=True)]
    df["perfect_recall"] = [
        perfect_recall(m, h, u) for m, h, u in zip(df["m"], df["h"], df["u"], strict=True)
    ]
    return df


def _add_judge_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Attach ``judge_hr*`` from the LLM-as-judge counts, where present.

    Only GPT-4o was judged (module 04). For the other two models the columns
    are empty and every rate is NaN, which is what the agreement analysis expects.
    """
    cols = ["llm_eval_m", "llm_eval_h", "llm_eval_u"]
    if not all(c in df.columns for c in cols):
        for name in ALL_METRICS:
            df[f"judge_{name}"] = float("nan")
        return df
    triples = df[cols].apply(pd.to_numeric, errors="coerce")
    ok = triples.notna().all(axis=1)
    for name, fn in ALL_METRICS.items():
        out = pd.Series(float("nan"), index=df.index, dtype=float)
        out.loc[ok] = [
            fn(m, h, u)
            for m, h, u in zip(triples.loc[ok, cols[0]], triples.loc[ok, cols[1]],
                               triples.loc[ok, cols[2]], strict=True)
        ]
        df[f"judge_{name}"] = out
    df["has_judge"] = ok
    return df


def _order_bins(df: pd.DataFrame) -> pd.DataFrame:
    df["bin_order"] = df["citation_bin"].map(bin_index)
    df["citation_bin"] = pd.Categorical(df["citation_bin"], categories=BIN_LABELS, ordered=True)
    return df


@lru_cache(maxsize=1)
def load_sample() -> pd.DataFrame:
    """The 9,108-paper stratified evaluation sample (Section 4.1)."""
    df = pd.read_csv(_require(config.EVAL_SAMPLE_RELEASE_CSV))
    df["authors"] = df["authors"].map(json.loads)
    df["title_length"] = df["title"].fillna("").str.len()
    return _order_bins(df)


@lru_cache(maxsize=len(MODELS))
def load_responses(model: str) -> pd.DataFrame:
    """One model's 9,108 scored answers, with all five rates attached."""
    if model not in MODELS:
        raise KeyError(f"unknown model {model!r}; expected one of {MODELS}")
    df = pd.read_csv(_require(config.RESPONSES_RELEASE_DIR / f"{model}.csv.gz"))
    df["model_label"] = MODEL_LABELS[model]
    df["n_authors"] = df["m"] + df["u"]        # |ground truth| = M + U
    df["n_generated"] = df["m"] + df["h"]      # |LLM answer|  = M + H
    df["title_length"] = df["title"].fillna("").str.len()
    df = _order_bins(_add_rates(df))
    return _add_judge_rates(df)


@lru_cache(maxsize=1)
def load_all_responses() -> pd.DataFrame:
    """All three models stacked, 27,324 rows, one row per (paper, model)."""
    return pd.concat([load_responses(m) for m in MODELS], ignore_index=True)


# ---------------------------------------------------------------------------
# writing tables, figures, and JSON with provenance
# ---------------------------------------------------------------------------

def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True, timeout=5, cwd=config.REPO_ROOT)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def provenance(**extra: Any) -> dict[str, Any]:
    """The header block embedded in every generated artifact."""
    meta = {
        "generated_by": Path(sys.argv[0]).name if sys.argv and sys.argv[0] else "python",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "seed": RANDOM_SEED,
    }
    meta.update(extra)
    return meta


def _atomic_write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
    return path


def write_json(path: Path, payload: Any, **extra: Any) -> Path:
    """Write a JSON artifact with a ``_meta`` provenance header."""
    body: dict[str, Any] = {"_meta": provenance(**extra)}
    if isinstance(payload, dict):
        body.update(payload)
    else:
        body["data"] = payload
    return _atomic_write_text(Path(path), json.dumps(body, indent=2, ensure_ascii=False) + "\n")


def read_ndjson(path: Path):
    """Stream an NDJSON or NDJSON.GZ file. Malformed lines are skipped."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


_STYLE_APPLIED = False


def apply_style() -> None:
    """Set the shared matplotlib defaults. Idempotent."""
    global _STYLE_APPLIED
    if _STYLE_APPLIED:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": ":",
        "legend.frameon": False,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "pdf.fonttype": 42,   # embed TrueType, not Type 3; ACL requires it
        "ps.fonttype": 42,
    })
    _STYLE_APPLIED = True


def save_figure(fig, name: str, values: dict[str, Any], *, source: str,
                out_dir: Path = OUT_DIR) -> Path:
    """Save ``<name>.pdf`` and the numbers it plots as ``<name>.json``."""
    import matplotlib.pyplot as plt

    pdf = out_dir / f"{name}.pdf"
    pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf)
    write_json(out_dir / f"{name}.json", values, source=source)
    plt.close(fig)
    return pdf


def fmt(value: float | None, places: int = 3, dash: str = "--") -> str:
    """Format a number for LaTeX, rendering NaN and None as a dash."""
    import math

    if value is None or (isinstance(value, float) and math.isnan(value)):
        return dash
    return f"{value:.{places}f}"


def fmt_int(value: float | None, dash: str = "--") -> str:
    import math

    if value is None or (isinstance(value, float) and math.isnan(value)):
        return dash
    return f"{int(round(value)):,}"


def latex_tabular(header: list[str], rows: list[list[str]], *, align: str,
                  midrule_before: set[int] | None = None) -> str:
    """Build a booktabs tabular body. Caption and label stay in the paper."""
    midrule_before = midrule_before or set()
    lines = [f"\\begin{{tabular}}{{@{{}}{align}@{{}}}}", "\\toprule",
             " & ".join(header) + r" \\", "\\midrule"]
    for i, row in enumerate(rows):
        if i in midrule_before:
            lines.append("\\midrule")
        lines.append(" & ".join(row) + r" \\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def write_latex_table(path: Path, body: str, *, source: str) -> Path:
    """Write a generated LaTeX table body with a provenance header."""
    header = "\n".join(f"% {k}: {v}" for k, v in provenance(source=source).items())
    return _atomic_write_text(
        Path(path), f"% GENERATED FILE -- do not edit by hand.\n{header}\n{body.rstrip()}\n"
    )


def save_table(name: str, header: list[str], rows: list[list[str]], *, align: str,
               source: str, midrule_before: set[int] | None = None,
               values: dict[str, Any] | None = None, out_dir: Path = OUT_DIR) -> Path:
    """Write ``<out_dir>/<name>.tex`` and, if given, its values as ``<name>.json``."""
    body = latex_tabular(header, rows, align=align, midrule_before=midrule_before)
    path = write_latex_table(out_dir / f"{name}.tex", body, source=source)
    if values is not None:
        write_json(out_dir / f"{name}.json", values, source=source)
    return path
