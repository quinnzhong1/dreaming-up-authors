"""Module 07 settings on top of ``06_llm_recall_hardcoded_eval/common.py``.

Every analysis here reads the same released answers as module 06, so the
loaders, constants and LaTeX / figure writers are imported from there. This
file only changes where the outputs go (``results/additional_analyses/``) and
adds the constants module 07 needs.

Named ``shared.py`` rather than ``common.py`` so that it does not shadow module
06's ``common`` on ``sys.path``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

sys.path.insert(0, str(config.HARDCODED_EVAL_MODULE))
import common as _common  # noqa: E402
from common import (  # noqa: E402, F401  (re-exported)
    BIN_LABELS,
    FIELD_KEYS,
    MODEL_COLORS,
    MODEL_LABELS,
    MODELS,
    apply_style,
    field_label,
    fmt,
    fmt_int,
    load_all_responses,
    load_responses,
    load_sample,
    write_json,
)

OUT_DIR: Path = config.ADDITIONAL_ANALYSES_DIR

# A handful of answers are runaway generations rather than attribution attempts:
# the longest lists 4,096 names. Ground truth is capped at 20 authors, so any
# answer longer than this is excluded from the output-length analyses of
# analyze_author_count.py, where a single such row would dominate a mean.
# 106 of 27,324 rows (0.4%) are affected. Every rate-based analysis keeps them.
RUNAWAY_GENERATION_THRESHOLD = 50


def save_table(*args, **kwargs) -> Path:
    """``common.save_table`` writing to ``results/additional_analyses/``."""
    kwargs.setdefault("out_dir", OUT_DIR)
    return _common.save_table(*args, **kwargs)


def save_figure(*args, **kwargs) -> Path:
    """``common.save_figure`` writing to ``results/additional_analyses/``."""
    kwargs.setdefault("out_dir", OUT_DIR)
    return _common.save_figure(*args, **kwargs)
