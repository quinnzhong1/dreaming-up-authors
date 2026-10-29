"""Step 2b: the five-stage quality filter (Section 2.1, Figure A1).

Runs on the sampled OAG papers after they have been joined with their Semantic
Scholar record. Stages, in order (each removal rate is relative to what survived
the previous stage, as reported in the paper):

    1. document type: journal or conference only         (removes 56.9%)
    2. author count: 1-20 authors                         (0.3%)
    3. language: FastText English, confidence >= 0.8      (47.9%)
    4. author-name patterns: no single names, "Last, First", or groups  (0.8%)
    5. OAG and Semantic Scholar title + first-author match (13.5%)

The language filter reads the Semantic Scholar title plus abstract. Without the
FastText model (config.FASTTEXT_LID) it passes every paper and the script warns,
because retention will then not match the paper.

The Semantic Scholar querying that builds the input (one title search per OAG
paper) is not part of this repository.

Input:
    config.MATCHED_UNFILTERED_NDJSON   one record per paper, OAG fields under
                                       "original", Semantic Scholar under
                                       "semantic_scholar"
Output:
    config.MATCHED_NDJSON              the papers that pass all five filters
    config.FILTER_FUNNEL_JSON          per-stage counts (Figure A1)

Usage:
    python 01_data_collection/02b_filter_matched_records.py [--limit N]
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

VALID_DOC_TYPES = {"journal", "conference"}
MIN_AUTHORS = 1
MAX_AUTHORS = 20
MIN_LANGUAGE_CONFIDENCE = 0.8

VALID_NAME_PATTERNS = {
    "Initials, Lastname",
    "Firstname Lastname",
    "Mixed Name with Initials",
    "Hyphenated Name",
    "Multi-part Name (No Initials)",
}
INVALID_NAME_PATTERNS = {"Single Name", "Lastname, Firstname", "Institutional/Group"}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def normalize_text(text: Any) -> str:
    """Lowercase, strip punctuation, collapse whitespace -- filter 5's title form."""
    if not text:
        return ""
    out = re.sub(r"\s+", " ", str(text).lower().strip())
    out = re.sub(r"[^\w\s]", " ", out)
    return re.sub(r"\s+", " ", out).strip()


def enriched_of(record: dict) -> dict:
    """The Semantic Scholar block, whatever key it was written under."""
    source = record.get("source")
    key = "semantic_scholar" if source == "semantic" else source
    value = record.get(key) if key else None
    return value if isinstance(value, dict) else {}


def first_author_lastname(authors: Any) -> str:
    """Final whitespace-separated token of the first author, lowercased.

    Handles both representations: OAG's semicolon-separated string and
    Semantic Scholar's list of ``{authorId, name}``.
    """
    if not authors:
        return ""
    if isinstance(authors, list):
        if not authors:
            return ""
        first = authors[0]
        name = first.get("name", "") if isinstance(first, dict) else str(first)
    elif isinstance(authors, str):
        text = authors.strip()
        name = (text.split(";")[0] if ";" in text else text).strip()
    else:
        return ""
    parts = name.strip().split()
    return parts[-1].lower() if parts else ""


def classify_name(name: str) -> str:
    """Bucket an author name into one of the structural patterns of filter 4."""
    if not name or not isinstance(name, str):
        return "Empty/Invalid"

    name = re.sub(r"\s+", " ", name.strip())
    if "," in name:
        return "Lastname, Firstname"

    lower = name.lower()
    if any(token in lower for token in ("et al", "group", " and ", "institut")):
        return "Institutional/Group"

    parts = name.split()
    if len(parts) == 1:
        return "Single Name"
    if any("-" in part for part in parts):
        return "Hyphenated Name"

    def is_initial(part: str) -> bool:
        return len(part) == 1 or (len(part) == 2 and part.endswith("."))

    n_initials = sum(1 for p in parts if is_initial(p))
    if n_initials:
        return "Initials, Lastname" if len(parts) == n_initials + 1 else "Mixed Name with Initials"
    if len(parts) == 2:
        return "Firstname Lastname"
    return "Multi-part Name (No Initials)"


# --------------------------------------------------------------------------
# the filters
# --------------------------------------------------------------------------

Verdict = tuple[bool, str | None]


def filter_doc_type(record: dict, ctx: FilterContext) -> Verdict:
    """1. Journal and conference papers only; drop books, chapters, patents."""
    original = record.get("original")
    if isinstance(original, dict):
        value = str(original.get("doc_type", "")).lower()
        if value in VALID_DOC_TYPES:
            return True, None
    value = str(record.get("doc_type", "")).lower()
    if value in VALID_DOC_TYPES:
        return True, None
    return False, f"doc_type not journal/conference: {value!r}"


def filter_author_count(record: dict, ctx: FilterContext) -> Verdict:
    """2. Between 1 and 20 authors inclusive."""
    authors = enriched_of(record).get("authors")
    if not isinstance(authors, list):
        return False, "no author data"
    n = len(authors)
    if n == 0:
        return False, "no authors"
    if n > MAX_AUTHORS:
        return False, f"too many authors ({n})"
    return True, None


def filter_language(record: dict, ctx: FilterContext) -> Verdict:
    """3. FastText English identification at confidence >= 0.8.

    Without a model this passes everything and says so once, rather than
    silently changing the retention rate. A detector error keeps the paper.
    """
    if ctx.language_model is None:
        return True, None
    enriched = enriched_of(record)
    text = f"{enriched.get('title') or ''} {enriched.get('abstract') or ''}".strip()
    if not text or text == "<abstract_text>":
        return False, "no text for language detection"
    text = text.replace("\n", " ").replace("\r", " ")
    try:
        labels, probs = ctx.language_model.predict(text, k=1)
    except Exception:  # noqa: BLE001 -- a detector failure should not drop a paper
        return True, None
    code = labels[0].replace("__label__", "")
    confidence = float(probs[0]) if len(probs) else 0.0
    if code == "en" and confidence >= MIN_LANGUAGE_CONFIDENCE:
        return True, None
    return False, f"language {code!r} at confidence {confidence:.2f}"


def filter_author_name_patterns(record: dict, ctx: FilterContext) -> Verdict:
    """4. Every author name must have a usable structure; one bad name drops the paper.

    Strict, but a paper whose author list contains "Institute of Physics" has
    no usable ground truth, and keeping it would score a correct refusal as a
    hallucination.
    """
    authors = enriched_of(record).get("authors")
    if not isinstance(authors, list) or not authors:
        return False, "no author data for pattern check"
    offenders = [
        f"{classify_name(a['name'])}: {a['name']!r}"
        for a in authors
        if isinstance(a, dict) and a.get("name")
        and classify_name(a["name"]) in INVALID_NAME_PATTERNS
    ]
    if offenders:
        return False, "invalid name pattern(s): " + "; ".join(offenders[:2])
    return True, None


def filter_title_match(record: dict, ctx: FilterContext) -> Verdict:
    """5. The OAG and Semantic Scholar records must be the same paper.

    Semantic Scholar is queried by title, so a near miss can return a different
    paper; the exact normalised title plus first-author surname is what makes
    the ground truth trustworthy.
    """
    original = record.get("original")
    if not isinstance(original, dict):
        return False, "missing 'original' block"
    orig_title = normalize_text(original.get("title"))
    if not orig_title:
        return False, "missing title in 'original' block"

    enriched = enriched_of(record)
    if not enriched:
        return False, "missing enriched block"
    if orig_title != normalize_text(enriched.get("title")):
        return False, "title mismatch"

    orig_author = first_author_lastname(original.get("authors"))
    enrich_author = first_author_lastname(enriched.get("authors"))
    if orig_author and enrich_author and orig_author != enrich_author:
        return False, "first author mismatch"
    return True, None


PIPELINE: list[tuple[str, Callable[[dict, FilterContext], Verdict]]] = [
    ("document_type", filter_doc_type),
    ("author_count", filter_author_count),
    ("language", filter_language),
    ("author_name_patterns", filter_author_name_patterns),
    ("title_author_match", filter_title_match),
]


# --------------------------------------------------------------------------
# running the pipeline
# --------------------------------------------------------------------------

@dataclass
class FilterContext:
    """Shared state: the language model and the sampled rejections."""

    language_model: Any = None
    max_samples_per_reason: int = 3
    rejections: dict[str, list[dict]] = field(default_factory=lambda: defaultdict(list))

    @classmethod
    def from_config(cls, **kwargs) -> FilterContext:
        """Load the FastText model at ``config.FASTTEXT_LID`` if it is present."""
        path = config.FASTTEXT_LID
        model = None
        if path.exists():
            import fasttext

            model = fasttext.load_model(str(path))
        return cls(language_model=model, **kwargs)


class Funnel:
    """Per-stage counts as machine-readable JSON.

    Figure A1 and the Section 2.1 rates are the same numbers, written once so
    they cannot disagree.
    """

    def __init__(self, stages: list[str]):
        self.stages = stages
        self.n_in = 0
        self.removed: Counter[str] = Counter()
        self.reasons: dict[str, Counter[str]] = {s: Counter() for s in stages}

    def record(self, stage: str, reason: str) -> None:
        self.removed[stage] += 1
        self.reasons[stage][reason.split(":")[0]] += 1

    def as_dict(self) -> dict:
        rows, remaining = [], self.n_in
        for stage in self.stages:
            removed = self.removed[stage]
            rows.append({
                "stage": stage,
                "in": remaining,
                "removed": removed,
                "out": remaining - removed,
                "pct_removed": round(100 * removed / remaining, 1) if remaining else 0.0,
                "top_reasons": dict(self.reasons[stage].most_common(5)),
            })
            remaining -= removed
        return {
            "initial": self.n_in,
            "final": remaining,
            "retention_pct": round(100 * remaining / self.n_in, 1) if self.n_in else 0.0,
            "stages": rows,
        }


def apply_filters(records: Iterable[dict], ctx: FilterContext | None = None,
                  funnel: Funnel | None = None) -> Iterator[dict]:
    """Stream records through all five filters, yielding survivors."""
    ctx = ctx or FilterContext()
    funnel = funnel if funnel is not None else Funnel([name for name, _ in PIPELINE])
    for record in records:
        funnel.n_in += 1
        for stage, fn in PIPELINE:
            keep, reason = fn(record, ctx)
            if not keep:
                funnel.record(stage, reason or "unspecified")
                if len(ctx.rejections[stage]) < ctx.max_samples_per_reason:
                    ctx.rejections[stage].append(
                        {"reason": reason, "record_id": record.get("row_id") or record.get("id")}
                    )
                break
        else:
            yield record


def _open(path: Path, mode: str):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, mode, encoding="utf-8")
    return open(path, mode, encoding="utf-8")


def read_ndjson(path: Path) -> Iterator[dict]:
    """Stream an NDJSON or NDJSON.GZ file. Malformed lines are skipped."""
    with _open(path, "rt") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def write_ndjson(path: Path, records: Iterable[dict]) -> int:
    """Write records atomically; a killed run leaves no truncated file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    n = 0
    with _open(tmp, "wt") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1
    os.replace(tmp, path)
    return n


def filter_corpus(source: Path, destination: Path, funnel_path: Path | None = None,
                  limit: int | None = None) -> dict:
    """Filter an NDJSON corpus end to end and write the funnel counts."""
    from itertools import islice

    ctx = FilterContext.from_config()
    funnel = Funnel([name for name, _ in PIPELINE])
    stream = read_ndjson(source)
    if limit is not None:
        stream = islice(stream, limit)

    written = write_ndjson(destination, apply_filters(stream, ctx, funnel))
    summary = funnel.as_dict()
    summary["written"] = written
    summary["language_model_loaded"] = ctx.language_model is not None
    if funnel_path:
        funnel_path.parent.mkdir(parents=True, exist_ok=True)
        funnel_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=config.MATCHED_UNFILTERED_NDJSON)
    ap.add_argument("--output", type=Path, default=config.MATCHED_NDJSON)
    ap.add_argument("--funnel", type=Path, default=config.FILTER_FUNNEL_JSON)
    ap.add_argument("--limit", type=int, default=None, help="only read the first N records")
    args = ap.parse_args()

    summary = filter_corpus(args.input, args.output, args.funnel, limit=args.limit)
    if not summary["language_model_loaded"]:
        print("WARNING: no FastText model found, so the language filter passed "
              "everything through. Retention will not match the paper.", file=sys.stderr)
    for stage in summary["stages"]:
        print(f"  {stage['stage']:24s} {stage['in']:>10,} -> {stage['out']:>10,} "
              f"({stage['pct_removed']:>5.1f}% removed)")
    print(f"  {'TOTAL':24s} {summary['initial']:>10,} -> {summary['final']:>10,} "
          f"({summary['retention_pct']}% retained)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
