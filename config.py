"""Central configuration: paths, citation bins, and research fields.

Every script in this repository imports this module, so all file locations are
defined in one place. Paths can be overridden with environment variables (or a
``.env`` file in the repository root; see ``.env.example``). This is useful when
the large raw data lives on an external drive.
"""

import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

REPO_ROOT: Path = Path(__file__).resolve().parent
load_dotenv(REPO_ROOT / ".env")


def _path_from_env(var_name: str, default: Path) -> Path:
    """Return the path stored in ``var_name`` or ``default`` if it is unset.

    Args:
        var_name: Name of the environment variable.
        default: Fallback path.

    Returns:
        The resolved path.
    """
    value = os.environ.get(var_name, "").strip()
    return Path(value).expanduser().resolve() if value else default


# ---------------------------------------------------------------------------
# Root directories
# ---------------------------------------------------------------------------
# DATA_DIR holds the (large) intermediate datasets; RESULTS_DIR holds API
# outputs, analysis tables, and figures. Both are git-ignored.
DATA_DIR: Path = _path_from_env("DATA_DIR", REPO_ROOT / "data")
RESULTS_DIR: Path = _path_from_env("RESULTS_DIR", REPO_ROOT / "results")

# Raw Open Academic Graph v2 MAG paper dumps (mag_papers_0.txt ... mag_papers_10.txt).
OAG_RAW_DIR: Path = _path_from_env("OAG_RAW_DIR", DATA_DIR / "raw" / "oag_v2")
OAG_FILE_PREFIX: str = "mag_papers_"
OAG_FILE_COUNT: int = 11

# Sampled OAG papers joined with their Semantic Scholar record, before the quality
# filters (NDJSON, one record per paper with "original.*" OAG fields and
# "semantic_scholar.*" fields). Input of 01/02b_filter_matched_records.py.
MATCHED_UNFILTERED_NDJSON: Path = _path_from_env(
    "MATCHED_UNFILTERED_NDJSON", DATA_DIR / "semantic_scholar_matched" / "matched_unfiltered.ndjson"
)
# Output of the Semantic Scholar matching + quality-filtering step (same format;
# the 1,921,209 papers that pass all five filters).
MATCHED_NDJSON: Path = _path_from_env(
    "MATCHED_NDJSON", DATA_DIR / "semantic_scholar_matched" / "matched_filtered.ndjson"
)
# FastText language-identification model used by the language filter
# (https://fasttext.cc/docs/en/language-identification.html).
FASTTEXT_LID: Path = _path_from_env("FASTTEXT_LID", DATA_DIR / "raw" / "lid.176.bin")

# ---------------------------------------------------------------------------
# Module 01: data collection
# ---------------------------------------------------------------------------
OAG_SAMPLE_DIR: Path = DATA_DIR / "oag_sample"                # 10 x 1M sampled papers
MATCHED_CHUNKS_DIR: Path = DATA_DIR / "matched_chunks"        # flattened NDJSON chunks
QUERY_FIELDS_DIR: Path = DATA_DIR / "query_fields"            # author lists + search queries
EVAL_SET_DIR: Path = DATA_DIR / "eval_set"
EVAL_SET_RAW_CSV: Path = EVAL_SET_DIR / "eval_set_9108_all_columns.csv"
EVAL_SET_CSV: Path = EVAL_SET_DIR / "eval_set_9108.csv"       # used by modules 04 and 05
PROXY_SET_DIR: Path = DATA_DIR / "proxy_set"
PROXY_SET_CSV: Path = PROXY_SET_DIR / "proxy_set_6048.csv"    # used by modules 02 and 03
PROXY_RESAMPLE_CSV: Path = PROXY_SET_DIR / "proxy_set_6048_resample.csv"
# Released paper lists (paperId + citation count / bin / field as of the September 2025 snapshot).
PAPER_IDS_EVAL_CSV: Path = REPO_ROOT / "01_data_collection" / "paper_ids_9108.csv"
PAPER_IDS_PROXY_CSV: Path = REPO_ROOT / "01_data_collection" / "paper_ids_6048.csv"
PAPER_IDS_PROXY_RESAMPLE_CSV: Path = REPO_ROOT / "01_data_collection" / "paper_ids_6048_resample.csv"

# ---------------------------------------------------------------------------
# Module 02-05: results
# ---------------------------------------------------------------------------
GOOGLE_HITS_DIR: Path = RESULTS_DIR / "google_search_hits"
INFINI_GRAM_DIR: Path = RESULTS_DIR / "llm_corpus_occurrence"
GPT4O_RECALL_DIR: Path = RESULTS_DIR / "gpt4o_recall_llm_selfeval"
GPT4O_GENERATED_CSV: Path = GPT4O_RECALL_DIR / "generated_answers.csv"
GPT4O_SELF_EVAL_CSV: Path = GPT4O_RECALL_DIR / "self_evaluated_answers.csv"
MC_DIR: Path = RESULTS_DIR / "mc_recognition"
FIGURES_DIR: Path = RESULTS_DIR / "figures"
FILTER_FUNNEL_JSON: Path = RESULTS_DIR / "data_collection" / "filter_funnel.json"   # 01/02b

# ---------------------------------------------------------------------------
# Module 06: open-ended recall of three LLMs with the hard-coded evaluation
# ---------------------------------------------------------------------------
HARDCODED_EVAL_MODULE: Path = REPO_ROOT / "06_llm_recall_hardcoded_eval"
# Released artifacts (tracked in git): the evaluation sample with full metadata
# and every model's scored answers.
EVAL_SAMPLE_RELEASE_CSV: Path = HARDCODED_EVAL_MODULE / "release" / "eval_sample_9108.csv.gz"
RESPONSES_RELEASE_DIR: Path = HARDCODED_EVAL_MODULE / "release" / "responses"
HARDCODED_EVAL_DIR: Path = RESULTS_DIR / "llm_recall_hardcoded_eval"
LLM_RESPONSES_DIR: Path = HARDCODED_EVAL_DIR / "responses"    # new query_llms.py runs
API_CACHE_DIR: Path = DATA_DIR / "cache"

# ---------------------------------------------------------------------------
# Module 07: additional analyses (author position, author count, ethnicity,
# title length, regression, prompt sensitivity)
# ---------------------------------------------------------------------------
ADDITIONAL_ANALYSES_MODULE: Path = REPO_ROOT / "07_additional_analyses"
# Released artifacts (tracked in git): surname tallies and prompt-sensitivity CSVs.
NAME_FREQUENCY_RELEASE_DIR: Path = ADDITIONAL_ANALYSES_MODULE / "release" / "name_frequency"
PROMPT_SENSITIVITY_RELEASE_DIR: Path = ADDITIONAL_ANALYSES_MODULE / "release" / "prompt_sensitivity"
ADDITIONAL_ANALYSES_DIR: Path = RESULTS_DIR / "additional_analyses"

# ---------------------------------------------------------------------------
# Module 08: hidden-state analysis of two open models
# ---------------------------------------------------------------------------
# The prompt CSV every GPU stage reads (--data_csv), rebuilt from the evaluation sample.
PROBING_PROMPTS_CSV: Path = DATA_DIR / "probing" / "probing_prompts_9108.csv"

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
# 13 exponentially spaced citation bins (right-inclusive edges for pd.cut).
CITATION_BIN_EDGES: list[float] = [-1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, np.inf]
CITATION_BIN_LABELS: list[str] = [
    "0-2", "3-4", "5-8", "9-16", "17-32", "33-64", "65-128", "129-256",
    "257-512", "513-1024", "1025-2048", "2049-4096", "4097+",
]

# The eight research fields (lower-cased Semantic Scholar fieldsOfStudy labels).
FIELDS: list[str] = [
    "medicine", "biology", "chemistry", "computer science",
    "psychology", "physics", "materials science", "mathematics",
]
