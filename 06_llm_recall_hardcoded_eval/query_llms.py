"""Ask GPT-4o, DeepSeek-R1 and Claude Sonnet 4.5 for the authors of each paper.

Open-ended authorship attribution over the 9,108-paper evaluation sample
(Sections 4.2-4.3): one few-shot prompt (``prompts/authorship_fewshot_paper.txt``),
temperature 0. GPT-4o is called through the OpenAI API; the other two through
OpenRouter.

Output is resumable NDJSON: re-running skips papers already in the output file,
and every API response is also cached on disk under ``data/cache/``. Each record
stores the exact prompt string sent (``prompt``) and the call envelope: HTTP
status, latency, timestamp, and, for OpenRouter, the upstream provider that
served it. OpenRouter routed the DeepSeek-R1 run across many providers, so
"temperature 0" does not by itself explain a given answer.

Usage (from the repository root):
    python 06_llm_recall_hardcoded_eval/query_llms.py --model gpt-4o
    python 06_llm_recall_hardcoded_eval/query_llms.py --model deepseek-r1 --limit 20
    python 06_llm_recall_hardcoded_eval/query_llms.py --model deepseek-r1-emotional \\
        --ids 07_additional_analyses/release/prompt_sensitivity/prompt_sensitivity_paired_rows.csv
Then score the answers:
    python 06_llm_recall_hardcoded_eval/score_answers.py \\
        results/llm_recall_hardcoded_eval/responses/gpt-4o.ndjson \\
        results/llm_recall_hardcoded_eval/responses/gpt-4o_scored.csv

Needs ``OPENAI_API_KEY`` (gpt-4o) or ``OPENROUTER_API_KEY`` (the other two) in
``.env``. Temperature 0 is not a determinism guarantee for hosted APIs, and two of
the three models were reached through aliases that float, so a new run will not
reproduce the released answers in ``release/responses/`` token for token.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_sample  # noqa: E402

import config  # noqa: E402  (on sys.path via common)

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
PERSONA_PROMPTS_DIR = config.ADDITIONAL_ANALYSES_MODULE / "prompts"
QUESTION = "Who are the authors of the paper titled '{title}' which was published in {year}?"

# Exact model identifiers and decoding parameters behind release/responses/.
MODEL_SPECS: dict[str, dict[str, Any]] = {
    "gpt-4o": {
        "access": "openai",
        "model_id": "chatgpt-4o-latest",   # floating alias, not a dated snapshot
        "temperature": 0,
        "max_tokens": None,                 # provider default
        "prompt": PROMPTS_DIR / "authorship_fewshot_paper.txt",
        # Collected 2025-10 with a separate script; the archived records carry no
        # per-call timestamp, raw response envelope, or resolved model string.
    },
    "deepseek-r1": {
        "access": "openrouter",
        "model_id": "deepseek/deepseek-r1-0528",
        "temperature": 0,
        "max_tokens": 500,
        "prompt": PROMPTS_DIR / "authorship_fewshot_paper.txt",
        # Collected 2025-10-22/23; the serving provider is stored per record.
    },
    "claude-sonnet-4.5": {
        "access": "openrouter",
        "model_id": "anthropic/claude-sonnet-4.5",   # floating alias
        "temperature": 0,
        "max_tokens": 500,
        "prompt": PROMPTS_DIR / "authorship_fewshot_paper.txt",
        # Collected 2025-10-22/23; routed across Google Vertex and Amazon Bedrock.
    },
    # Prompt sensitivity (Section 8, module 07): DeepSeek-R1 on 1,004 papers with
    # the baseline template's opening sentence replaced by a persona. Collected
    # 2026-02-17. Run with --ids to restrict to the 1,004 papers.
    "deepseek-r1-emotional": {
        "access": "openrouter",
        "model_id": "deepseek/deepseek-r1-0528",
        "temperature": 0,
        "max_tokens": 500,
        "prompt": PERSONA_PROMPTS_DIR / "authorship_fewshot_emotional.txt",
    },
    "deepseek-r1-authoritative": {
        "access": "openrouter",
        "model_id": "deepseek/deepseek-r1-0528",
        "temperature": 0,
        "max_tokens": 500,
        "prompt": PERSONA_PROMPTS_DIR / "authorship_fewshot_authoritative.txt",
    },
}

DEFAULT_TIMEOUT = 60
DEFAULT_MAX_RETRIES = 5


# ---------------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------------

class MissingAPIKeyError(RuntimeError):
    def __init__(self, env_var: str):
        super().__init__(
            f"{env_var} is not set. Copy .env.example to .env and fill it in, "
            f"or export {env_var} in your shell."
        )


@dataclass
class CallResult:
    """One API call, with everything needed to audit it later."""

    success: bool
    content: str | None
    error: str | None
    status_code: int | None
    latency_ms: int
    timestamp: str
    model_requested: str
    model_served: str | None = None
    provider: str | None = None
    raw_response: dict[str, Any] | None = None
    request_body: dict[str, Any] | None = field(default=None, repr=False)

    def as_record(self) -> dict[str, Any]:
        return {
            "api_success": self.success,
            "llm_answer": self.content,
            "api_error": self.error,
            "api_status_code": self.status_code,
            "api_response_time_ms": self.latency_ms,
            "timestamp": self.timestamp,
            "model_requested": self.model_requested,
            "model_served": self.model_served,
            "api_provider": self.provider,
            "api_response_full": json.dumps(self.raw_response, ensure_ascii=False)
            if self.raw_response else None,
        }


class ResponseCache:
    """Disk cache keyed by a hash of (endpoint, model, request body).

    Re-running a killed job costs nothing for the calls that already landed.
    The cache is under ``data/cache/`` and is git-ignored.
    """

    def __init__(self, root: Path, namespace: str):
        self.root = Path(root) / namespace
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(endpoint: str, model: str, body: dict) -> str:
        payload = json.dumps({"endpoint": endpoint, "model": model, "body": body},
                             sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        # Two-level fan-out; a flat directory of 9,108 files is slow to list.
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict | None:
        path = self._path(key)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

    def put(self, key: str, value: dict) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)


class ChatClient:
    """Minimal OpenAI-compatible chat client with retries and caching."""

    ENDPOINTS = {
        "openai": ("https://api.openai.com/v1/chat/completions", "OPENAI_API_KEY"),
        "openrouter": ("https://openrouter.ai/api/v1/chat/completions", "OPENROUTER_API_KEY"),
    }

    def __init__(self, access: str, *, timeout: int = DEFAULT_TIMEOUT,
                 max_retries: int = DEFAULT_MAX_RETRIES,
                 cache: ResponseCache | None = None):
        if access not in self.ENDPOINTS:
            raise KeyError(f"unknown access path {access!r}; expected one of "
                           f"{sorted(self.ENDPOINTS)}")
        self.access = access
        self.endpoint, self.env_var = self.ENDPOINTS[access]
        self.timeout = timeout
        self.max_retries = max_retries
        self.cache = cache
        self._key: str | None = None

    @property
    def api_key(self) -> str:
        if self._key is None:
            key = os.environ.get(self.env_var)
            if not key:
                raise MissingAPIKeyError(self.env_var)
            self._key = key
        return self._key

    def _headers(self) -> dict[str, str]:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        if self.access == "openrouter":
            # Optional attribution headers; harmless if the referer is unset.
            headers["HTTP-Referer"] = os.environ.get("OPENROUTER_REFERER", "")
            headers["X-Title"] = "dreaming-up-authors"
        return headers

    def call(self, prompt: str, model: str, *, temperature: float = 0.0,
             max_tokens: int | None = None, use_cache: bool = True) -> CallResult:
        import requests

        body: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        if max_tokens is not None:
            body["max_tokens"] = max_tokens

        cache_key = ResponseCache.key(self.endpoint, model, body)
        if use_cache and self.cache:
            hit = self.cache.get(cache_key)
            if hit is not None:
                return CallResult(**hit)

        last: CallResult | None = None
        for attempt in range(self.max_retries):
            started = time.time()
            now = datetime.now(timezone.utc).isoformat(timespec="seconds")
            try:
                response = requests.post(self.endpoint, headers=self._headers(),
                                         json=body, timeout=self.timeout)
                latency = int((time.time() - started) * 1000)
                if response.status_code == 200:
                    data = response.json()
                    content = (data.get("choices", [{}])[0]
                               .get("message", {}).get("content") or "")
                    result = CallResult(
                        success=True, content=content.strip(), error=None,
                        status_code=200, latency_ms=latency, timestamp=now,
                        model_requested=model, model_served=data.get("model"),
                        provider=data.get("provider"), raw_response=data,
                        request_body=body,
                    )
                    if self.cache:
                        self.cache.put(cache_key, dict(result.__dict__))
                    return result
                last = CallResult(
                    success=False, content=None,
                    error=f"HTTP {response.status_code}: {response.text[:300]}",
                    status_code=response.status_code, latency_ms=latency,
                    timestamp=now, model_requested=model, request_body=body,
                )
                # 4xx other than 429 will not fix themselves.
                if 400 <= response.status_code < 500 and response.status_code != 429:
                    return last
            except Exception as exc:  # noqa: BLE001 - record and retry
                last = CallResult(
                    success=False, content=None, error=f"{type(exc).__name__}: {exc}",
                    status_code=None, latency_ms=int((time.time() - started) * 1000),
                    timestamp=now, model_requested=model, request_body=body,
                )
            time.sleep(min(2**attempt, 30))

        return last  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# collection loop
# ---------------------------------------------------------------------------

def build_question(title: str, year) -> str:
    return QUESTION.format(title=title, year=year)


def already_done(output: Path) -> set[str]:
    """Paper ids already present in the output file, for resumption."""
    if not output.exists():
        return set()
    done = set()
    with open(output, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                done.add(json.loads(line)["paper_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def collect(papers: Iterable[dict], model_key: str, output: Path, *,
            limit: int | None = None, resume: bool = True,
            use_cache: bool = True) -> Iterator[dict]:
    """Query one model over ``papers`` and stream records to ``output``.

    ``papers`` yields dicts with ``paper_id``, ``title``, ``year``, and the
    metadata to carry through (``field``, ``citations``, ``citation_bin``, ``authors``).
    """
    spec = MODEL_SPECS[model_key]
    template = Path(spec["prompt"]).read_text(encoding="utf-8")

    output.parent.mkdir(parents=True, exist_ok=True)
    done = already_done(output) if resume else set()

    client = ChatClient(
        spec["access"],
        cache=ResponseCache(config.API_CACHE_DIR, f"authorship/{model_key}") if use_cache else None,
    )

    written = 0
    with open(output, "a", encoding="utf-8") as sink:
        for paper in papers:
            if limit is not None and written >= limit:
                break
            if paper["paper_id"] in done:
                continue

            question = build_question(paper["title"], paper["year"])
            prompt = template.format(question=question)
            result = client.call(
                prompt,
                spec["model_id"],
                temperature=spec.get("temperature", 0),
                max_tokens=spec.get("max_tokens"),
                use_cache=use_cache,
            )
            record = {
                "paper_id": paper["paper_id"],
                # The join key of 07_additional_analyses/prompt_sensitivity_significance.py.
                "record_id": paper.get("record_id"),
                "title": paper["title"],
                "year": paper["year"],
                "field": paper.get("field"),
                "citations": paper.get("citations"),
                "citation_bin": paper.get("citation_bin"),
                "ground_truth_authors": paper.get("authors"),
                "question": question,
                "model": model_key,
                "prompt_file": Path(spec["prompt"]).name,
                "prompt": prompt,
                **result.as_record(),
            }
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
            sink.flush()
            written += 1
            yield record


def main() -> int:
    from dotenv import load_dotenv

    load_dotenv(config.REPO_ROOT / ".env")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(MODEL_SPECS))
    ap.add_argument("--output", type=Path, default=None,
                    help="default: results/llm_recall_hardcoded_eval/responses/<model>.ndjson")
    ap.add_argument("--limit", type=int, default=None, help="stop after this many new calls")
    ap.add_argument("--ids", type=Path, default=None,
                    help="only query the papers listed in this CSV (a record_id or paper_id column)")
    args = ap.parse_args()

    # Check the key before doing any work.
    try:
        _ = ChatClient(MODEL_SPECS[args.model]["access"]).api_key
    except MissingAPIKeyError as exc:
        print(exc, file=sys.stderr)
        return 1

    output = args.output or config.LLM_RESPONSES_DIR / f"{args.model}.ndjson"
    sample = load_sample()
    if args.ids is not None:
        import pandas as pd

        wanted = pd.read_csv(args.ids, dtype=str)
        key = "record_id" if "record_id" in wanted.columns else "paper_id"
        sample = sample[sample[key].astype(str).isin(set(wanted[key]))]
        print(f"Restricted to {len(sample):,} papers listed in {args.ids}")
    papers = sample.to_dict("records")
    n = 0
    for _ in collect(papers, args.model, output, limit=args.limit):
        n += 1
        if n % 100 == 0:
            print(f"  {n:,} collected", flush=True)
    print(f"Collected {n:,} responses -> {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
