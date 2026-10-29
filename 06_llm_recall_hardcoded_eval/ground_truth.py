"""Extracting a ground-truth author list from an archived record.

The ground truth reached the archives in four shapes: a plain string, a clean
list, Semantic Scholar's list of dicts, and that list *shredded by a CSV
round-trip* that treated its internal commas as field separators.

The fourth is the one that matters. It is baked into the released CSVs, and a
naive reader silently sees one author where there are nine, deflating every
count. Recovery tries ``ast.literal_eval`` on the rejoined text, then falls
back to a regex over ``'name'`` values.
"""

from __future__ import annotations

import ast
import json
import re

_NAME_PATTERNS = (
    re.compile(r"'name':\s*'([^']+)'"),
    re.compile(r'"name":\s*"([^"]+)"'),
    re.compile(r"name['\"]?\s*:\s*['\"]([^'\"]+)['\"]"),
)


def _regex_names(text: str) -> list[str]:
    for pattern in _NAME_PATTERNS:
        found = pattern.findall(text)
        if found:
            return found
    return []


def extract_author_names(raw) -> list[str] | str:
    """Normalise ground-truth author data to a list of names.

    Returns a list when the names could be separated reliably, or the original
    string when the caller should hand it to :class:`name_splitter.NameSplitter`
    instead. Returns ``[]`` for empty input.
    """
    if not raw:
        return []

    if isinstance(raw, str):
        stripped = raw.strip()
        if stripped.startswith(("[", "{")):
            try:
                return extract_author_names(ast.literal_eval(stripped))
            except (ValueError, SyntaxError):
                pass
            names = _regex_names(stripped)
            if names:
                return names
        # A plain comma-separated author string; the splitter handles it.
        return raw

    if isinstance(raw, list):
        if not raw:
            return []
        first = raw[0]

        if isinstance(first, dict):
            return [item["name"] for item in raw if isinstance(item, dict) and "name" in item]

        # Shape 4: fragments of a stringified list of dicts.
        if isinstance(first, str) and ("[{" in first or "'name'" in first or '"name"' in first):
            joined = "".join(raw).strip()
            try:
                parsed = ast.literal_eval(joined)
            except (ValueError, SyntaxError):
                try:
                    parsed = json.loads(re.sub(r'"\s*:\s*"', '":"', joined.replace("'", '"')))
                except json.JSONDecodeError:
                    return _regex_names(joined)
            if isinstance(parsed, list):
                return [i["name"] for i in parsed if isinstance(i, dict) and "name" in i]
            return _regex_names(joined)

        # Shape 2: a clean list of plain names.
        if all(isinstance(i, str) and not any(m in i for m in ("{'", '{"', "'name'", '"name"'))
               for i in raw):
            return list(raw)

        return list(raw)

    raise TypeError(f"unsupported ground-truth type: {type(raw).__name__}")


def ground_truth_list(record: dict, splitter) -> list[str]:
    """Pull the ground-truth author list out of a record of any shipped schema.

    ``ground_truth_authors`` is the NDJSON/OpenRouter field name; ``answer`` is
    the GPT-4o CSV field name.
    """
    raw = record.get("ground_truth_authors") or record.get("answer") or []
    extracted = extract_author_names(raw)
    if isinstance(extracted, str):
        return splitter.split(extracted)
    return list(extracted)


def llm_answer(record: dict) -> str:
    """Pull the raw LLM answer string out of a record of any shipped schema.

    ``llm_answer`` is the column written by ``query_llms.py`` and stored in
    ``release/responses/``; ``llm_authors`` and ``gpt_answer`` are the names in
    the original archives.
    """
    return (record.get("llm_answer") or record.get("llm_authors")
            or record.get("gpt_answer") or "")
