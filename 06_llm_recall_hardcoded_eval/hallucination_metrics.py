"""Hallucination rate metrics (Section 4.4).

Each is a function of one triple per paper -- M matched, H hallucinated, U
unretrieved -- and runs from 0 to 1. Zero-denominator conventions came from the
scripts that produced the published numbers and are not uniform: HR3 returns
0.0 where the others return NaN.
"""

from __future__ import annotations

import re
from typing import NamedTuple

import numpy as np

NAN = float("nan")


class Counts(NamedTuple):
    """The (M, H, U) triple for one paper.

    The original CSVs called these X, Y, Z, in that order.
    """

    m: int
    h: int
    u: int


def hr1(m: float, h: float, u: float) -> float:
    """Max-normalised error rate. Robust to runaway generation length."""
    denom = max(h + m, u + m)
    return NAN if denom == 0 else 1 - m / denom


def hr2(m: float, h: float, u: float) -> float:
    """Jaccard hallucination rate: 1 minus Jaccard similarity."""
    denom = m + h + u
    return NAN if denom == 0 else (h + u) / denom


def hr3(m: float, h: float, u: float) -> float:
    """Harmonic hallucination rate. Peaks when the model both invents and omits."""
    denom = (h + m) * u + (m + u) * h
    return 0.0 if denom == 0 else (2 * h * u) / denom


def hr4(m: float, h: float, u: float) -> float:
    """Hallucination-only rate: the fraction of predicted names that are fabricated."""
    denom = m + h
    return NAN if denom == 0 else h / denom


def hr5(m: float, h: float, u: float) -> float:
    """Omission-only rate: the fraction of true authors the model failed to retrieve."""
    denom = m + u
    return NAN if denom == 0 else u / denom


ALL_METRICS = {"hr1": hr1, "hr2": hr2, "hr3": hr3, "hr4": hr4, "hr5": hr5}


def perfect_recall(m: float, h: float, u: float) -> float:
    """The binary criterion of the multiple-choice comparison (Section 6): ``H == 0 and U == 0``.

    Comparable to multiple-choice accuracy, unlike ``1 - HR2``, which gives
    partial credit.
    """
    return 1.0 if (h == 0 and u == 0) else 0.0


def all_rates(m: float, h: float, u: float) -> dict[str, float]:
    return {name: fn(m, h, u) for name, fn in ALL_METRICS.items()}


_TRIPLE = re.compile(r"^\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*$")


def parse_counts(value: str | None) -> Counts | None:
    """Parse the ``"M, H, U"`` string; ``None`` if it is not three integers.

    The LLM judge can emit ``"None, None, None"`` after failed retries.
    """
    if value is None:
        return None
    match = _TRIPLE.match(str(value))
    if not match:
        return None
    return Counts(*(int(g) for g in match.groups()))


def add_rates(df, counts_column: str = "hard_code_evaluation", prefix: str = ""):
    """Add ``hr1``..``hr5`` and ``m``/``h``/``u`` columns to a DataFrame.

    Returns the frame and the number of unparseable rows dropped, so a caller
    can report them rather than silently losing papers.
    """

    parsed = df[counts_column].map(parse_counts)
    keep = parsed.notna()
    dropped = int((~keep).sum())
    out = df.loc[keep].copy()
    triples = parsed.loc[keep]
    out[f"{prefix}m"] = [t.m for t in triples]
    out[f"{prefix}h"] = [t.h for t in triples]
    out[f"{prefix}u"] = [t.u for t in triples]
    for name, fn in ALL_METRICS.items():
        out[f"{prefix}{name}"] = [fn(t.m, t.h, t.u) for t in triples]
    out[f"{prefix}perfect_recall"] = [perfect_recall(t.m, t.h, t.u) for t in triples]
    return out, dropped


def mean_ci(values, confidence: float = 0.95) -> tuple[float, float, float]:
    """Mean, CI half-width, and the ``n`` actually used, ignoring NaN.

    Student's t, not the normal quantile: the smallest populated stratum holds
    22 papers, where the two differ by about 5%.
    """
    from scipy import stats

    arr = np.asarray(list(values), dtype=float)
    arr = arr[~np.isnan(arr)]
    n = arr.size
    if n == 0:
        return NAN, NAN, 0
    mean = float(arr.mean())
    if n == 1:
        return mean, 0.0, 1
    sem = float(arr.std(ddof=1) / np.sqrt(n))
    half = float(stats.t.ppf(0.5 + confidence / 2, n - 1) * sem)
    return mean, half, n
