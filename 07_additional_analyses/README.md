# Additional analyses: position, author count, ethnicity, title length, regression, prompt wording

**This folder produces the Section 8 robustness checks.** Each asks whether something other than prevalence explains the citation trend of module 06. All of them read module 06's released answers (`../06_llm_recall_hardcoded_eval/release/`) plus the small artifacts in `release/` here.

| File | Purpose |
|---|---|
| `analyze_position.py` | Miss rate by ground-truth position and hallucination rate by generated position, over answers with at least one correct author; Figure A6. Re-runs module 06's matcher, because positions are not stored. |
| `analyze_author_count.py` | Output length and fabrication count by true author count, and per field for single-author papers; Tables A26-A27. |
| `analyze_ethnicity.py` | Share of each surname-origin category among hallucinated names vs. the dataset, counted by population and by diversity; Table A28, Figures A7-A8. Needs `ethnicseer`. |
| `analyze_title_length.py` | Spearman rho between title length and HR2, overall and controlled for citations (partial rank correlation, within-bin Fisher-z pooling); Table A29. |
| `analyze_regression.py` | Fractional logit on HR2 and HR4 and a mixed-effects model with a random intercept per field, GPT-4o; Table A30. |
| `analyze_prompt_sensitivity.py` | Tables A31-A33 from the CSVs in `release/prompt_sensitivity/`. |
| `prompt_sensitivity_significance.py` | Builds those CSVs from three scored prompt-variant runs: Spearman rho per variant and a paired bootstrap over papers. |
| `name_frequency.py` | Surname counting and the reader for `release/name_frequency/`. |
| `shared.py` | Imports module 06's `common.py` and sends outputs to `results/additional_analyses/`. |
| `prompts/` | The emotional and authoritative persona prompts: the baseline template with its first sentence replaced. |
| `release/name_frequency/` | Surname tallies: every author appearance in the 1.92M-paper corpus, and each model's hallucinated surnames (population and diversity). |
| `release/prompt_sensitivity/` | The five prompt-sensitivity CSVs, including the per-paper (M, H, U) of all three variants (`prompt_sensitivity_paired_rows.csv`). |

## Run

Everything except the ethnicity analysis runs in under a minute, with no API key and no network:

```bash
python 07_additional_analyses/analyze_position.py
python 07_additional_analyses/analyze_author_count.py
python 07_additional_analyses/analyze_title_length.py
python 07_additional_analyses/analyze_regression.py
python 07_additional_analyses/analyze_prompt_sensitivity.py

pip install ethnicseer==0.1.2 "setuptools<81"        # optional, see below
python 07_additional_analyses/analyze_ethnicity.py    # ~6 minutes
```

Outputs go to `results/additional_analyses/` in the same format as module 06: a `.tex` per table, a `.pdf` per figure, and a `.json` with the values behind each. `positional_quoted.json` and `author_count_scaling.json` hold the numbers quoted in the Section 8 prose.

To collect the prompt-variant answers again (needs `OPENROUTER_API_KEY`), query DeepSeek-R1 on the 1,004 papers with each prompt, score the answers, and run the significance tests:

```bash
IDS=07_additional_analyses/release/prompt_sensitivity/prompt_sensitivity_paired_rows.csv
OUT=results/llm_recall_hardcoded_eval/responses
for v in deepseek-r1 deepseek-r1-emotional deepseek-r1-authoritative; do
  python 06_llm_recall_hardcoded_eval/query_llms.py --model $v --ids $IDS --output $OUT/ps_$v.ndjson
  python 06_llm_recall_hardcoded_eval/score_answers.py $OUT/ps_$v.ndjson $OUT/ps_$v.csv
done
python 07_additional_analyses/prompt_sensitivity_significance.py \
  --inputs baseline=$OUT/ps_deepseek-r1.csv emotional=$OUT/ps_deepseek-r1-emotional.csv \
           authoritative=$OUT/ps_deepseek-r1-authoritative.csv \
  --out_dir results/additional_analyses/prompt_sensitivity --n_boot 10000
```

## Reproducibility notes

* **The 1,004-paper subset.** It is a proportional stratified draw from the evaluation sample: each of the 104 (field, citation bin) strata contributes its size x 1,004 / 9,108, rounded to the nearest integer. Which papers were picked within a stratum cannot be regenerated, because the code that drew them is not in this repository and no seed was recorded. The drawn papers are the `record_id` values of `release/prompt_sensitivity/prompt_sensitivity_paired_rows.csv`; `query_llms.py --ids` restricts a run to them.
* **Significance tests.** Feeding the per-variant (M, H, U) of `prompt_sensitivity_paired_rows.csv` back into `prompt_sensitivity_significance.py` (seed 42, 10,000 resamples) reproduces all five shipped CSVs byte for byte.
* **Runaway generations.** `analyze_author_count.py` drops answers with more than 50 names (`RUNAWAY_GENERATION_THRESHOLD` in `shared.py`; 106 of 27,324 rows) from the output-length means. The rate-based analyses keep them.
* **Surname tallies.** The shipped `release/name_frequency/` files were converted from the original counting runs. The counting functions in `name_frequency.py` normalise surnames differently (Unicode-folded, particles stripped, title-cased: `McDonald` becomes `Mcdonald`, `da Silva` becomes `Silva`), so recounting with them does not reproduce the shipped tallies exactly. Table A28 is computed from the shipped tallies.
* **Ethnicity labels.** `ethnicseer` infers a coarse surname-origin category, not a person's identity. Labels are computed at analysis time and only aggregates are written; no per-name label is saved. Its pickled model was trained with an older scikit-learn and warns when loaded; the published numbers were computed with scikit-learn 1.5.2.
* **Data use.** These tallies are a measurement instrument for model behaviour. Do not use them to evaluate researchers or to make per-author inferences.
