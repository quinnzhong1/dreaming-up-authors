"""Counting surnames, in the corpus and in what the models hallucinate.

Helper for ``analyze_ethnicity.py`` (Section 8, Table A28). Feeds dataset prevalence over the 1.92M papers (715,837
distinct surnames), and per-model hallucination tallies counted two ways.

``population`` counts a surname once per paper it was hallucinated into, which
is what a user encounters. ``diversity`` counts each distinct surname once,
which is where the effect shows up -- a model can lean on a handful of Russian
surnames while producing thousands of distinct English ones.
"""

from __future__ import annotations

import gzip
import importlib
import json
import sys
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

sys.path.insert(0, str(config.HARDCODED_EVAL_MODULE))
from name_matcher import NameMatcher  # noqa: E402


def surname_of(name: str, matcher: NameMatcher | None = None) -> str:
    """Particle-stripped surname, title-cased. Empty when unparseable."""
    matcher = matcher or NameMatcher()
    parsed = matcher._parse_single_name(name)["parsed_alpha"]
    surname = parsed["last_stripped"] or parsed["last"]
    return surname.title() if surname else ""


def count_corpus_surnames(records: Iterable[dict]) -> Counter[str]:
    """Dataset prevalence: one count per author appearance across the corpus."""
    # Record accessor from 01/02b (the module name starts with a digit).
    sys.path.insert(0, str(config.REPO_ROOT / "01_data_collection"))
    enriched_of = importlib.import_module("02b_filter_matched_records").enriched_of

    matcher = NameMatcher()
    counts: Counter[str] = Counter()
    for record in records:
        for author in enriched_of(record).get("authors", []) or []:
            name = author.get("name") if isinstance(author, dict) else author
            if not name:
                continue
            surname = surname_of(str(name), matcher)
            if surname:
                counts[surname] += 1
    return counts


def count_hallucinated_surnames(responses) -> dict[str, Counter[str]]:
    """Hallucination tallies for one model, in both counting modes.

    ``responses`` is a frame with a ``hallucinated`` column holding the
    semicolon-separated fabricated names for each paper.
    """
    matcher = NameMatcher()
    population: Counter[str] = Counter()
    occurrences: Counter[str] = Counter()

    for cell in responses["hallucinated"].fillna(""):
        seen_this_paper = set()
        for name in (n.strip() for n in str(cell).split(";")):
            if not name:
                continue
            surname = surname_of(name, matcher)
            if not surname:
                continue
            occurrences[surname] += 1
            seen_this_paper.add(surname)
        # One increment per paper, however often the surname appeared in it.
        for surname in seen_this_paper:
            population[surname] += 1

    return {
        # papers a surname was hallucinated into
        "population": population,
        # each distinct surname once, however often it appeared
        "diversity": Counter(dict.fromkeys(population, 1)),
        # raw occurrence count, kept for reference
        "occurrences": occurrences,
    }


def write_counts(path: str | Path, counts: Counter[str] | dict[str, int]) -> Path:
    """Write a surname tally as gzipped JSON, sorted by descending count."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(ordered, f, ensure_ascii=False)
    return path


def read_counts(path: str | Path) -> dict[str, int]:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        return json.load(f)
