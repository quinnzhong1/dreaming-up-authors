# Dreaming Up Authors: Factual Prevalence and LLM Hallucinations in Academic Authorship Attribution

Code for the EMNLP 2026 paper *Dreaming Up Authors: Factual Prevalence and LLM Hallucinations in Academic Authorship Attribution*.

The paper asks whether LLMs hallucinate more when they attribute authors to papers that are rarely cited (i.e. less prevalent in their training data). This repository covers:

| Folder | What it does | Paper |
|---|---|---|
| [`01_data_collection/`](01_data_collection) | Samples papers from the Open Academic Graph, builds the 9,108-paper evaluation set and the 6,048-paper prevalence-proxy set | Section 2, Section 4.1, Appendix "Dataset Details" and "Data Sampling Strategy" |
| [`02_google_search_hits/`](02_google_search_hits) | Google search hit counts vs. citation count | Section 3.1, Figure 2 (left), Appendix "Google Search Hits Analysis" |
| [`03_llm_corpus_occurrence/`](03_llm_corpus_occurrence) | Infini-gram n-gram counts in open pretraining corpora (OLMo2-32B, RedPajama) vs. citation count | Section 3.2, Figure 2 (right), Appendix "LLM Corpus Occurrence Analysis" |
| [`04_gpt4o_recall_llm_selfeval/`](04_gpt4o_recall_llm_selfeval) | Open-ended authorship attribution with GPT-4o, scored by LLM self-evaluation | Appendix "Evaluation Method Comparison"; input to module 05 |
| [`05_mc_recognition/`](05_mc_recognition) | Multiple-choice recognition experiment (recall vs. recognition) with three distractor constructions | Section 6, Appendix "More Details on Multiple-choice Recognition Analysis" |
| [`06_llm_recall_hardcoded_eval/`](06_llm_recall_hardcoded_eval) | **Main results**: open-ended attribution by GPT-4o, DeepSeek-R1 and Claude Sonnet 4.5, scored by the hard-coded name matcher; HR1-HR5 vs. citation count | Sections 4-5, Figures 3-5, Tables 2-3, A18, A22-A25 |
| [`07_additional_analyses/`](07_additional_analyses) | Robustness checks: author position, author count, ethnicity of hallucinated names, title length, multivariate models, prompt wording | Section 8, Figures A6-A8, Tables A26-A33 |
| [`08_hidden_state_analysis/`](08_hidden_state_analysis) | Hidden states of Qwen3-32B and Mistral-Small-3.2-24B: prevalence decoding, mediation, activation steering, P9 entity probe | Section 7, Figures 7, A11-A12, Tables A37-A44 |

> **Main results.** Module 06 holds the main hallucination-rate results and ships every model's answers, so all Section 5 tables and figures regenerate without an API key. Module 04 is the GPT-4o run with LLM self-evaluation; it is used for the evaluation-method comparison (module 06, `analyze_eval_agreement.py`) and as the recall baseline of the MC experiment. Module 08 holds the hidden-state analyses (Section 7) and ships the outputs of the paper's runs.

---

## Contents

- [Dreaming Up Authors: Factual Prevalence and LLM Hallucinations in Academic Authorship Attribution](#dreaming-up-authors-factual-prevalence-and-llm-hallucinations-in-academic-authorship-attribution)
  - [Contents](#contents)
  - [1. Environment setup](#1-environment-setup)
    - [Credentials (`.env`)](#credentials-env)
    - [Data locations](#data-locations)
    - [How to run](#how-to-run)
  - [2. Pipeline overview](#2-pipeline-overview)
  - [3. Module 01: data collection](#3-module-01-data-collection)
    - [Step 0: download the raw data](#step-0-download-the-raw-data)
    - [Steps](#steps)
  - [4. Module 02: Google search hits](#4-module-02-google-search-hits)
  - [5. Module 03: LLM corpus occurrence (Infini-gram)](#5-module-03-llm-corpus-occurrence-infini-gram)
  - [6. Module 04: GPT-4o open-ended recall (LLM self-evaluation)](#6-module-04-gpt-4o-open-ended-recall-llm-self-evaluation)
  - [7. Module 05: multiple-choice recognition](#7-module-05-multiple-choice-recognition)
  - [8. Module 06: three-LLM recall, hard-coded evaluation](#8-module-06-three-llm-recall-hard-coded-evaluation)
  - [9. Module 07: additional analyses](#9-module-07-additional-analyses)
  - [10. Module 08: hidden-state analysis](#10-module-08-hidden-state-analysis)
  - [11. Reproducibility notes](#11-reproducibility-notes)
  - [12. Citation](#12-citation)

---

## 1. Environment setup

Requirements: Python 3.10 or newer.

```bash
git clone <this repository>
cd dreaming-up-authors

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env               # then fill in the keys you need
```

### Credentials (`.env`)

Only the keys for the modules you run are needed. `.env` is git-ignored; never commit it.

| Variable | Needed for | Where to get it |
|---|---|---|
| `OPENAI_API_KEY` | modules 04, 05, and GPT-4o in 06 | <https://platform.openai.com/api-keys> |
| `OPENROUTER_API_KEY` | DeepSeek-R1 and Claude Sonnet 4.5 in module 06, prompt variants for module 07 | <https://openrouter.ai/keys> |
| `BRIGHTDATA_API_KEY`, `BRIGHTDATA_ZONE` | module 02 | a BrightData account with a SERP API zone |
| `S2_API_KEY` (optional) | `01/07_fetch_datasets_from_ids.py`, `05/fetch_collaborators.py` | <https://www.semanticscholar.org/product/api>; without a key the shared rate limit (~1 request/s) is used |

The Infini-gram API (module 03) needs no key.

### Data locations

All paths are defined in [`config.py`](config.py). By default, intermediate data is written to `./data/` and results to `./results/` (both git-ignored). The raw data needs about 100 GB, so you can move it to another disk (e.g. an external drive) by setting these variables in `.env`:

| Variable | Default | Content |
|---|---|---|
| `DATA_DIR` | `./data` | all intermediate datasets |
| `RESULTS_DIR` | `./results` | API outputs, tables, figures |
| `OAG_RAW_DIR` | `$DATA_DIR/raw/oag_v2` | extracted OAG v2 files `mag_papers_0.txt` ... `mag_papers_10.txt` |
| `MATCHED_UNFILTERED_NDJSON` | `$DATA_DIR/semantic_scholar_matched/matched_unfiltered.ndjson` | sampled OAG papers joined with Semantic Scholar, input of the quality filter (step 2b) |
| `MATCHED_NDJSON` | `$DATA_DIR/semantic_scholar_matched/matched_filtered.ndjson` | output of the Semantic Scholar matching + filtering step |
| `FASTTEXT_LID` | `$DATA_DIR/raw/lid.176.bin` | FastText language-ID model for the language filter (step 2b) |

### How to run

* **Scripts** (`.py`) are run from the repository root, e.g. `python 01_data_collection/05_sample_eval_set.py`. Every script prints its usage with `--help` (or in its docstring).
* **Notebooks** (`.ipynb`) are run from their own folder, e.g. `cd 02_google_search_hits && jupyter notebook analysis_google_hits.ipynb`.

---

## 2. Pipeline overview

```
OAG v2 (MAG papers, ~100 GB)
  └─ 01/01_sample_oag_papers.py ........... 10M random papers
       └─ [Semantic Scholar title search: not in this repository]
            └─ 01/02b_filter_matched_records.py ... 5 quality filters -> matched NDJSON (1.92M papers)
                 └─ 01/03_flatten_matched_records.py ... CSV chunks
                      ├─ 01/05_sample_eval_set.py ....... 9,108-paper evaluation set ─┬─ 04 GPT-4o recall (self-evaluation)
                      │                                                                  ├─ 05 MC recognition (uses 04)
                      │                                                                  ├─ 06 three-LLM recall, hard-coded evaluation (main results; uses 04 for A22-A23)
                      │                                                                  │    └─ 07 additional analyses (Section 8; reads 06's released answers)
                      │                                                                  └─ 08 hidden states of two open models (GPU; Section 7)
                      └─ 01/04_build_query_fields.py
                           └─ 01/06_sample_proxy_set.py . 6,048-paper proxy set ─┬─ 02 Google search hits
                                                                                  └─ 03 Infini-gram counts ── Figure 2

Shortcut: 01/07_fetch_datasets_from_ids.py rebuilds both sets from the released ID files.
```

**Shortcut: skip the raw data.** You do not need the raw OAG data to rerun modules 02-05 on exactly the same papers as the paper. Three small ID files in `01_data_collection/` list the papers with their citation count and citation bin from the September 2025 snapshot; the evaluation-set file also stores the field and the ground-truth author list (`authors-info`: author names and Semantic Scholar author IDs):

| File | Papers | Used by |
|---|---|---|
| `paper_ids_9108.csv` | evaluation set (`paperId, citationCount, citation_bin, field, authors-info`) | modules 04, 05 |
| `paper_ids_6048.csv` | prevalence-proxy set (`paperId, citationCount, citation_bin`) | modules 02, 03 |
| `paper_ids_6048_resample.csv` | independent proxy resample (6,047 papers) | module 03 robustness check |

```bash
python 01_data_collection/07_fetch_datasets_from_ids.py --set all
```

This fetches the remaining metadata (titles and years; author lists for the proxy sets) from the Semantic Scholar API and writes `eval_set_9108.csv`, `proxy_set_6048.csv` and `proxy_set_6048_resample.csv` in the same format as steps 5 and 6 (see [Section 11](#11-reproducibility-notes) for caveats).

---

## 3. Module 01: data collection

Folder: [`01_data_collection/`](01_data_collection)

### Step 0: download the raw data

Download the MAG paper files of **Open Academic Graph v2** (`mag_papers_0.zip`, `mag_papers_1.zip`, `mag_papers_2.zip`) from the OAG website (<https://www.aminer.cn/oag-2-1>, or the Microsoft Research Open Academic Graph page), and extract them into `OAG_RAW_DIR`. You should get `mag_papers_0.txt` ... `mag_papers_10.txt` (JSON lines, one paper per line).

### Steps

| Step | Command | Output | Notes |
|---|---|---|---|
| 1 | `python 01_data_collection/01_sample_oag_papers.py` | `data/oag_sample/sampled_mag_papers_all_fields_chunk_{0..9}.csv` | 10 rounds of reservoir sampling (1M papers each, disjoint, seed 1234). Reads the full dump 10 times, so it takes many hours. |
| 2 | `python 01_data_collection/02_check_duplicates.py` | (printout) | Confirms that the 10M sampled papers are unique. |
| - | *Semantic Scholar matching* | `MATCHED_UNFILTERED_NDJSON` | **Not part of this repository.** Each sampled OAG paper is searched on Semantic Scholar by title (up-to-date citation counts, author names, fields of study). Each output line holds the OAG fields under `original.*` and the Semantic Scholar record under `semantic_scholar.*`. |
| 2b | `python 01_data_collection/02b_filter_matched_records.py` | `MATCHED_NDJSON`, `results/data_collection/filter_funnel.json` | The five quality filters, in order (Appendix, "Details of Data Filtering", Figure A1): journal/conference papers only, 1-20 authors, English (fastText on the Semantic Scholar title + abstract, confidence >= 0.8), no malformed author names, and an exact title / first-author match between OAG and Semantic Scholar. Without `FASTTEXT_LID` the language filter passes everything and the script warns. |
| 3 | `python 01_data_collection/03_flatten_matched_records.py` | `data/matched_chunks/chunk_{i}.csv` | Flattens the NDJSON into CSV chunks of 10,000 papers. |
| 4 | `python 01_data_collection/04_build_query_fields.py` | `data/query_fields/samples_google_search_{i}.csv` | Author-name lists, last names, and the Google query `"title" "author 1" "author 2" ...`. |
| 5 | `python 01_data_collection/05_sample_eval_set.py` | `data/eval_set/eval_set_9108.csv` (+ `_all_columns.csv`) | Stratified evaluation set: 8 fields x 13 citation bins, up to 100 papers per stratum; 9,108 papers. |
| 6 | `python 01_data_collection/06_sample_proxy_set.py` | `data/proxy_set/proxy_set_6048.csv` | Up to 500 papers per citation bin (6,048 papers) for modules 02 and 03. |
| 6b | `python 01_data_collection/06_sample_proxy_set.py --resample` | `data/proxy_set/proxy_set_6048_resample.csv` | Independent second sample (chunks scanned in reverse order) for the Infini-gram robustness check. |
| 7 | `python 01_data_collection/07_fetch_datasets_from_ids.py --set all` | `data/eval_set/eval_set_9108.csv`, `data/proxy_set/proxy_set_6048.csv`, `data/proxy_set/proxy_set_6048_resample.csv` | **Alternative to steps 1-6**: rebuilds the exact paper sets of the paper from the released ID files `paper_ids_*.csv`. |

**Citation bins** (13, exponentially spaced): 0-2, 3-4, 5-8, 9-16, 17-32, 33-64, 65-128, 129-256, 257-512, 513-1024, 1025-2048, 2049-4096, 4097+.
**Fields** (8): medicine, biology, chemistry, computer science, psychology, physics, materials science, mathematics. A paper's field is the first of its Semantic Scholar `fieldsOfStudy` that is one of these eight.

Columns of `eval_set_9108.csv`: `paperId, title, year, authors-info, citationCount, citation_bin, field, full_name_list`.

---

## 4. Module 02: Google search hits

Folder: [`02_google_search_hits/`](02_google_search_hits). Input: `data/proxy_set/proxy_set_6048.csv`.

1. **Crawl** (paid BrightData SERP API, one request per paper):
   ```bash
   python 02_google_search_hits/crawl_google_hits.py
   ```
   Results are saved in batches of 500 papers under `results/google_search_hits/crawl_batches/`, so an interrupted crawl resumes where it stopped; all batches are then merged into `results/google_search_hits/combined_google_results.csv`. Failed queries are skipped (5,179 of 6,048 papers returned a hit count in the paper).
2. **Analyse**: run `02_google_search_hits/analysis_google_hits.ipynb`.

**Results** (in `results/google_search_hits/`):

| File | Paper |
|---|---|
| `citation_bucket_stats_full.csv` | hit-count statistics per bucket on all crawled papers (N = 5,179) and the average query length per bucket |
| `citation_bucket_stats_80.csv` | after removing hits above the 80th percentile per bucket (N = 4,185); Figure 2 (left) |
| `citation_bucket_stats_{75,90,95}.csv` | other cutoffs (sensitivity check) |
| `citation_vs_google_hits.pdf` | log2-scale scatter plot with a linear fit |

Hit counts come from a live search engine, so a new crawl will give different absolute numbers, but the increasing trend with citation count should hold.

---

## 5. Module 03: LLM corpus occurrence (Infini-gram)

Folder: [`03_llm_corpus_occurrence/`](03_llm_corpus_occurrence). Input: `data/proxy_set/proxy_set_6048.csv` (and the resample).

For each paper the query `<title> AND <first author's last name>` is counted with the public [Infini-gram API](https://infini-gram.io/). Run all five configurations used in the paper (each takes about 2-3 hours because of rate limiting; runs are checkpointed and can be resumed):

```bash
python 03_llm_corpus_occurrence/query_infini_gram.py --corpus olmo2     --query-format capitalize             # main result
python 03_llm_corpus_occurrence/query_infini_gram.py --corpus redpajama --query-format capitalize
python 03_llm_corpus_occurrence/query_infini_gram.py --corpus olmo2     --query-format capitalize --resample  # robustness
python 03_llm_corpus_occurrence/query_infini_gram.py --corpus olmo2     --query-format camel                  # query-format sensitivity
python 03_llm_corpus_occurrence/query_infini_gram.py --corpus redpajama --query-format camel
```

`capitalize` formats the title as "Attention is all you need"; `camel` as "Attention Is All You Need".

Then run `03_llm_corpus_occurrence/analysis_infini_gram.ipynb`, which writes per-bucket tables `citation_bucket_stats_<corpus>_<format>.csv`, correlations and scatter plots to `results/llm_corpus_occurrence/`. Finally, draw **Figure 2** (needs the Google 80th-percentile table and the OLMo2 capitalized table):

```bash
python 03_llm_corpus_occurrence/plot_prevalence_proxy_figure.py   # -> results/figures/proxy_means_by_citation_bin_log.pdf
```

---

## 6. Module 04: GPT-4o open-ended recall (LLM self-evaluation)

Folder: [`04_gpt4o_recall_llm_selfeval/`](04_gpt4o_recall_llm_selfeval). Input: `data/eval_set/eval_set_9108.csv`.

> This is **not** the pipeline behind the paper's main hallucination-rate results; those use the hard-coded evaluation and three LLMs and are in [module 06](#8-module-06-three-llm-recall-hard-coded-evaluation). See the folder's [README](04_gpt4o_recall_llm_selfeval/README.md).

1. **Generate and self-evaluate** (2 x 9,108 OpenAI calls to `chatgpt-4o-latest`, temperature 0; about 6 hours; saved every 100 papers and resumable):
   ```bash
   python 04_gpt4o_recall_llm_selfeval/generate_and_self_evaluate.py --step all
   ```
   Outputs `results/gpt4o_recall_llm_selfeval/generated_answers.csv` and `self_evaluated_answers.csv`. The `evaluation` column holds `X, Y, Z` = matched, hallucinated, and missed authors.
2. **Analyse**: run `04_gpt4o_recall_llm_selfeval/compute_hallucination_rates.ipynb`. It computes HR1-HR5 per paper, tables per citation bin, per field x bin (with 95% confidence intervals) and per field, question-length statistics, and the OLS / mixed-effects / fractional-logit regressions.

---

## 7. Module 05: multiple-choice recognition

Folder: [`05_mc_recognition/`](05_mc_recognition). Inputs: `data/eval_set/eval_set_9108.csv` and `results/gpt4o_recall_llm_selfeval/self_evaluated_answers.csv` (module 04).

GPT-4o (temperature 0) sees the title and year of each paper with four author lists (A-D) and must pick the complete true list. Three distractor constructions are supported (`--variant`):

| Variant | Distractors | Paper |
|---|---|---|
| `original` | three random same-field author teams of the same size | main result (Section 6) |
| `same_field_swap` | the true list with one author replaced by a random same-field author | "Harder Distractor Controls" (S-F) |
| `collaborator_swap` | the true list with one non-first author replaced by a real collaborator of the first author | "Harder Distractor Controls" (Col.) |

```bash
# original (main result)
python 05_mc_recognition/build_options.py --variant original
python 05_mc_recognition/query_gpt4o.py   --variant original --dry-run   # optional: inspect prompts, no API call
python 05_mc_recognition/query_gpt4o.py   --variant original             # submit; re-run until it reports completion
python 05_mc_recognition/evaluate.py      --variant original

# same-field one-name swap (needs the original options)
python 05_mc_recognition/build_options.py --variant same_field_swap
python 05_mc_recognition/query_gpt4o.py   --variant same_field_swap
python 05_mc_recognition/evaluate.py      --variant same_field_swap

# collaborator one-name swap (needs the original options)
python 05_mc_recognition/fetch_collaborators.py          # Semantic Scholar, ~3 h without an API key
python 05_mc_recognition/build_options.py --variant collaborator_swap
python 05_mc_recognition/query_gpt4o.py   --variant collaborator_swap
python 05_mc_recognition/evaluate.py      --variant collaborator_swap

# tables and statistics reported in the paper
python 05_mc_recognition/compare_variants.py
```

`query_gpt4o.py` uses the OpenAI **Batch API** (one batch of 9,108 requests per variant, usually finished within 24 hours). It is a resumable state machine: run it once to submit, then run it again to poll and download; if some requests fail, it submits a retry batch automatically.

**Results** (in `results/mc_recognition/<variant>/`):

| File / command | Content |
|---|---|
| `options.csv` | the four options and the correct position for every paper (seed 42) |
| `responses.csv` | GPT-4o's chosen letters |
| `results.csv` | per paper: correct / incorrect, HR1-HR5, perfect recall |
| `bin_summary.csv` | per citation bin (overall, per field, single-/multi-author): raw and chance-corrected accuracy with Wilson 95% CI, recall accuracy (1 - HR), recall-recognition gap, residual recognition rate |
| `compare_variants.py` | the full-results table, the residual-recognition table, the harder-distractor table, the one-sample t-tests and Spearman correlations; saved as `results/mc_recognition/variant_comparison.csv` |
| `analysis_mc.ipynb` (set `VARIANT`) | figures in `figures/`: `fig2d_recall_recognition_gap` (main MC figure), `fig2a_overall_gap` (gap under HR1-HR5), `fig2e_perfect_recall_gap` (binary perfect recall), and further diagnostics |

---

## 8. Module 06: three-LLM recall, hard-coded evaluation

Folder: [`06_llm_recall_hardcoded_eval/`](06_llm_recall_hardcoded_eval). Input: the released `06_llm_recall_hardcoded_eval/release/` (tracked in git).

The released answers and their (M, H, U) counts are shipped, so the analyses run offline:

```bash
python 06_llm_recall_hardcoded_eval/analyze_results.py              # Figures 3-5, A4-A5; Tables 2, 3, A18, A24, A25
python 06_llm_recall_hardcoded_eval/analyze_eval_agreement.py       # matcher vs. self-evaluation, Tables A22-A23
python 06_llm_recall_hardcoded_eval/score_answers.py --check        # re-score the released answers
```

To collect new answers (temperature 0; GPT-4o via OpenAI, the other two via OpenRouter; resumable):

```bash
python 06_llm_recall_hardcoded_eval/query_llms.py --model gpt-4o   # or deepseek-r1, claude-sonnet-4.5
python 06_llm_recall_hardcoded_eval/score_answers.py <answers.ndjson> <scored.csv>
```

The matcher accepts a ground-truth author as retrieved at levels L1-L4 (exact, Unicode-folded, alpha-only, surname + full first name, surname + first initial); a surname-only match (L5) counts as one hallucinated and one unretrieved author. See the folder's [README](06_llm_recall_hardcoded_eval/README.md).

**Results** (in `results/llm_recall_hardcoded_eval/`): one `.tex` per table (`tabular` only), one `.pdf` per figure, and a `.json` with the values behind each; `headline_numbers.json` holds the numbers quoted in the abstract and conclusion.

---

## 9. Module 07: additional analyses

Folder: [`07_additional_analyses/`](07_additional_analyses). Inputs: module 06's `release/` and the small artifacts in `07_additional_analyses/release/` (surname tallies, prompt-sensitivity CSVs; tracked in git).

```bash
python 07_additional_analyses/analyze_position.py              # author position, Figure A6
python 07_additional_analyses/analyze_author_count.py          # author count, Tables A26-A27
python 07_additional_analyses/analyze_title_length.py          # title length, Table A29
python 07_additional_analyses/analyze_regression.py            # multivariate and mixed-effects models, Table A30
python 07_additional_analyses/analyze_prompt_sensitivity.py    # prompt wording, Tables A31-A33
python 07_additional_analyses/analyze_ethnicity.py             # Table A28, Figures A7-A8; needs ethnicseer, ~6 min
```

The prompt-sensitivity run queries DeepSeek-R1 on a proportional stratified subsample of 1,004 papers with the baseline, an emotional and an authoritative prompt (`query_llms.py --model deepseek-r1-emotional --ids ...`), and `prompt_sensitivity_significance.py` compares the variants with a paired bootstrap over papers. See the folder's [README](07_additional_analyses/README.md) for the full commands and caveats.

**Results** (in `results/additional_analyses/`): same format as module 06.

---

## 10. Module 08: hidden-state analysis

Folder: [`08_hidden_state_analysis/`](08_hidden_state_analysis). Input: the released `08_hidden_state_analysis/release/` (tracked in git) and, for the GPU stages, the prompt CSV built from module 06's sample.

The stages run Qwen3-32B (peak layer 48) and Mistral-Small-3.2-24B (peak layer 22) over the 9,108 papers: 6-12 GPU-hours and 40-90 GB of scratch disk per model. Install `requirements-probing.txt` (torch, transformers, h5py, scikit-learn, sentence-transformers); `mediation.py` also needs R with the `mediation` package. The run order is in the folder's [README](08_hidden_state_analysis/README.md). The outputs of the paper's runs (probing CSVs, per-paper scores, steering vectors and results) are shipped in `08_hidden_state_analysis/release/`.

---

## 11. Reproducibility notes

* **Random seeds.** OAG sampling uses seed 1234; MC option construction uses seed 42. The evaluation set and the proxy set are filled greedily in file order from the (randomly sampled) matched chunks, so they are deterministic given the same matched data.
* **Exact paper sets.** `01_data_collection/paper_ids_*.csv` list the papers of the evaluation set and the two proxy sets in their original row order (the order matters for the seeded MC option builder), together with the citation count and citation bin from the September 2025 Semantic Scholar snapshot used in the paper; the evaluation-set file also stores the field and the ground-truth author list. `07_fetch_datasets_from_ids.py` keeps these stored values, so every evaluation paper keeps its original bin, field and authors (Semantic Scholar revises author names over time, e.g. "S. Bree" became "S. V. van Bree"), and the seeded MC options are reproduced exactly. Author names are stored exactly as they appeared in Semantic Scholar, so a few of them (22 names in 8 papers) contain non-Latin characters. It also reports how many papers would fall into a different bin with today's counts. Titles and years, and the author lists of the proxy sets, are fetched live.
* **Semantic Scholar-dependent steps.** The collaborator pools (`fetch_collaborators.py`) are fetched live, so the `collaborator_swap` distractors will not be identical to the paper's.
* **Module 06 data.** `06_llm_recall_hardcoded_eval/release/eval_sample_9108.csv.gz` holds the same 9,108 papers as `paper_ids_9108.csv` with titles, years, DOIs and venues, so module 06 needs no Semantic Scholar call. Re-scoring the released answers reproduces the stored (M, H, U) for every DeepSeek-R1 and Claude Sonnet 4.5 answer and for 9,095 of 9,108 GPT-4o answers (see the module README).
* **LLM versions.** Module 04 uses the `chatgpt-4o-latest` alias and module 05 uses `gpt-4o`. Both point to model snapshots that OpenAI updates or retires over time, so newly generated answers can differ slightly from the paper even at temperature 0.
* **MC prompt.** The demonstration items are kept verbatim as used in the paper. The `original` run used a three-author second example; the two harder variants were run later with a seven-author second example (see `FEW_SHOT_BY_VARIANT` in `query_gpt4o.py`).
* **Web search.** Google hit counts depend on the live index and on BrightData's rendering, so they vary between crawls.
* **Costs.** Module 02 needs about 6,000 paid SERP requests; module 04 about 18,000 chat-completion calls; module 05 one Batch API job of 9,108 short requests per variant; module 06 9,108 chat-completion calls per model; the module 07 prompt variants 1,004 DeepSeek-R1 calls per variant.

---

## 12. Citation

```bibtex
@inproceedings{dreaming-up-authors-2026,
  title     = {Dreaming Up Authors: Factual Prevalence and LLM Hallucinations in Academic Authorship Attribution},
  author    = {Zhong, Sijia and Yirga, Tamir and Ma, Yijia and Le, Thai and Hu, Yifan},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year      = {2026}
}
```
