# Hidden-state analysis of two open models

**This folder produces the mechanistic evidence of Section 7** (Appendix A7, Tables A37-A44, Figures 7, A11 and A12). Qwen3-32B and Mistral-Small-3.2-24B answer the same 9,108 questions as in module 06. Their hidden states encode the paper's citation count before generation starts, that internal signal mediates much of the citation-hallucination link, and steering against it causes hallucinations.

The stages below run the two models over the 9,108 papers. Expect 6-12 GPU-hours and 40-90 GB of scratch disk per model. Install `requirements-probing.txt` first. The outputs of the paper's runs are shipped in `release/`.

| File | Purpose |
|---|---|
| `build_prompt_csv.py` | Step 1: rebuilds `data/probing/probing_prompts_9108.csv`, the `--data_csv` every stage reads, from module 06's released sample. |
| `extract_hidden_states.py` | One greedy generation per paper; saves hidden states at P1-P5 and P9 for the swept layers to `embeddings.h5`. GPU. |
| `score_open_model.py` | Scores the generated answers (HR2, HR4, HR5 per paper) and labels every generated name for P9. |
| `prevalence_decoding.py` | Ridge decoder of log(citations + 1), metadata and within-field controls, negative controls, citation-bucket probe, deciles (A7.1-A7.5). |
| `title_only_baselines.py` | Title-only TF-IDF and SBERT baselines, and P1 refit on the title residual (A7.2). |
| `mediation.py` | Mediation of citations -> HR2 through the P1 score (A7.6). Needs R with the `mediation` package. |
| `steering.py` | Difference-of-means steering vector; induce and fix arms (A7.7). GPU. |
| `steering_vector_analysis.py` | Probe direction in raw hidden space, and its cosine with the steering vector (A7.7). |
| `steering_controls.py` | Five norm-matched random directions and the probe-direction arm (A7.7, Table A43). GPU. |
| `steering_paired_stats.py` | McNemar tests and paired bootstrap CIs for the steering arms (A7.7). |
| `steering_name_recall.py` | Recall / precision split of the steering effect (A7.7). |
| `p9_entity_probe.py` | Entity-level probe on the P9 name states (A7.8, Table A44). |
| `scholar_utils.py` | The name matcher used by these stages (a copy of module 06's L1-L5 matcher, packaged for the GPU jobs). |
| `release/` | The shipped outputs: probing CSVs, `runs/<model>/scored.tsv.gz`, and `steering/<model>/` (vectors, results, meta). |

## Run

In run order (per model; Qwen3-32B shown, use `mistral_small_3_2_24b` and layer 22 for Mistral):

```bash
python 08_hidden_state_analysis/build_prompt_csv.py
python 08_hidden_state_analysis/extract_hidden_states.py --model_path Qwen/Qwen3-32B \
    --data_csv data/probing/probing_prompts_9108.csv --out_dir data/probing/runs/qwen3_32b
python 08_hidden_state_analysis/score_open_model.py --h5_path data/probing/runs/qwen3_32b/embeddings.h5 \
    --data_csv data/probing/probing_prompts_9108.csv --out_dir data/probing/runs/qwen3_32b
python 08_hidden_state_analysis/prevalence_decoding.py \
    --runs qwen3_32b=data/probing/runs/qwen3_32b mistral_small_3_2_24b=data/probing/runs/mistral_small_3_2_24b \
    --optimal_layers qwen3_32b:48 mistral_small_3_2_24b:22 --out_dir results/hidden_state_analysis/probing
python 08_hidden_state_analysis/mediation.py --sims 2000 --boot
```

The other stages print their arguments with `--help`; each docstring has a usage example.

## Hidden-state positions

Hidden states are saved at six positions. **P1**, the last prompt token before any answer is generated, is the one the paper reports. P2 and P3 are the first and last generated tokens, P4 and P5 are mean-pooled prompt and answer states, and P9 is the last token of each generated author name.

`extract_hidden_states.py` saves a fixed subset of layers per model family (`MODEL_REGISTRY`), not every layer: every second layer from 24 to 48 plus 63 for Qwen3, and every second layer from 10 to 38 plus 39 for Mistral-Small. Pass `--target_layers` to change it.

## Reproducibility notes

* **Determinism.** The open models are deterministic given the same weights, dtype and hardware. Record the model revision you pulled; the paper's runs used `main` in bfloat16.
* **Stages not runnable from `release/`.** `steering_paired_stats.py` and `steering_name_recall.py` re-score the steered generations (`steering_predictions.tsv`), which are not shipped. Their outputs (`paired_hr_stats.csv`) are.
