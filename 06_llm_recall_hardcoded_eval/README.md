# Open-ended recall of three LLMs, scored by the hard-coded evaluation

**This folder produces the paper's main hallucination-rate results** (Sections 4-5): GPT-4o, DeepSeek-R1 and Claude Sonnet 4.5 are asked for the authors of the 9,108 evaluation papers, and each answer is scored against the ground truth with a deterministic name matcher.

| File | Purpose |
|---|---|
| `query_llms.py` | Asks one model for the authors of every paper with the few-shot prompt in `prompts/` (temperature 0). Resumable NDJSON output plus an on-disk response cache. Also runs the DeepSeek-R1 persona-prompt variants of module 07 (`--model deepseek-r1-emotional`, `deepseek-r1-authoritative`); `--ids` restricts a run to a list of papers. |
| `score_answers.py` | Scores raw answers into (M, H, U) = matched, hallucinated, unretrieved authors. `--check` re-scores the released answers and compares them with the stored counts. |
| `name_splitter.py`, `name_matcher.py`, `ground_truth.py` | The name matcher (Section 4.5): splits an answer into names, matches each ground-truth author at levels L1-L5, and reads ground truth in every archived format. |
| `hallucination_metrics.py` | HR1-HR5 from (M, H, U) and the mean with a 95% t-interval (Section 4.4). |
| `common.py` | Constants, loaders for `release/`, and the LaTeX / figure writers. Module 07 imports it too. |
| `analyze_results.py` | Figures 3-5 and A4-A5; Tables 2, 3, A18, A24, A25 (Section 5, Appendix A5.1). |
| `analyze_eval_agreement.py` | Hard-coded matcher vs. the LLM self-evaluation of module 04; Tables A22-A23 (Appendix A4.4). |
| `release/` | The released data: `eval_sample_9108.csv.gz` (the evaluation sample with full metadata) and `responses/<model>.csv.gz` (each model's answers and stored M, H, U). |

## Run

All analyses read only `release/`, so they need no API key and no network:

```bash
python 06_llm_recall_hardcoded_eval/analyze_results.py
python 06_llm_recall_hardcoded_eval/analyze_eval_agreement.py
python 06_llm_recall_hardcoded_eval/score_answers.py --check    # ~20 s
```

Outputs go to `results/llm_recall_hardcoded_eval/`: one `.tex` per table (the `tabular` only, so the caption stays in the paper), one `.pdf` per figure, and a `.json` with the values behind each. Every file starts with a provenance header (script, git commit, seed, UTC time).

To query the models again (needs `OPENAI_API_KEY` for GPT-4o, `OPENROUTER_API_KEY` for the other two):

```bash
python 06_llm_recall_hardcoded_eval/query_llms.py --model gpt-4o        # also deepseek-r1, claude-sonnet-4.5
python 06_llm_recall_hardcoded_eval/score_answers.py \
    results/llm_recall_hardcoded_eval/responses/gpt-4o.ndjson \
    results/llm_recall_hardcoded_eval/responses/gpt-4o_scored.csv
```

## Scoring rule

Each ground-truth author keeps its strongest match to an LLM name, matched greedily in ground-truth order:

| Level | Match | Counts as |
|---|---|---|
| L1 | exact | matched |
| L2 | Unicode-folded | matched |
| L3 | alpha-only | matched |
| L3.5 | surname and full first name | matched |
| L4 | surname and first initial | matched |
| L5 | surname only | the LLM name is hallucinated **and** the true author is unretrieved |

The threshold is `MATCH_LEVEL_THRESHOLD = 4` in `common.py`. HR2 = (H + U) / (M + H + U) is the primary rate.

The matcher's output depends on the name-parsing libraries, so `requirements.txt` pins the versions used for the paper: `nameparser==1.1.3` and `Unidecode==1.3.8`. With `nameparser` 2.7.0, `score_answers.py --check` drops to 9,090 / 9,098 / 9,097 identical rows (GPT-4o / DeepSeek-R1 / Claude).

## Released data

`release/responses/<model>.csv.gz` has one row per paper with the raw answer (`llm_answer`), the ground truth (`ground_truth_authors`, a JSON list in publication order), and the counts `m`, `h`, `u`. HR1-HR5 are recomputed from the counts on load. For GPT-4o, `llm_eval_m`, `llm_eval_h`, `llm_eval_u` are the self-evaluation counts from module 04. `paper_id` is the Semantic Scholar `paperId` and is the join key everywhere.

`score_answers.py --check` reproduces the stored counts for all 9,108 DeepSeek-R1 and Claude Sonnet 4.5 answers and for 9,095 of 9,108 GPT-4o answers. The 13 GPT-4o rows differ because their stored ground truth is not the list they were originally scored against (e.g. surnames only, or two authors in one string).

## Reproducibility notes

* **Model versions.** GPT-4o was called as `chatgpt-4o-latest` and Claude as `anthropic/claude-sonnet-4.5`, both floating aliases; DeepSeek-R1 as `deepseek/deepseek-r1-0528`. `MODEL_SPECS` in `query_llms.py` records exactly what was called. A new run will give similar but not identical answers.
* **Routing.** OpenRouter served DeepSeek-R1 through several upstream providers; the provider is recorded per row in `api_provider`.
