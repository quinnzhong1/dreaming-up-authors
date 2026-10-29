# GPT-4o open-ended recall with LLM self-evaluation

**This folder does not produce the paper's main results.** The main hallucination-rate results (Section 5: hard-coded fuzzy-matching evaluation of GPT-4o, DeepSeek-R1 and Claude Sonnet 4.5) are in [`../06_llm_recall_hardcoded_eval/`](../06_llm_recall_hardcoded_eval).

This folder contains a GPT-4o run in which the same model scores its own answers ("LLM self-evaluation"). It is used for:

* the comparison between LLM self-evaluation and hard-coded evaluation (Appendix, "Evaluation Method Comparison");
* the open-ended **recall baseline** of the multiple-choice recognition experiment in [`../05_mc_recognition/`](../05_mc_recognition): `evaluate.py` there reads `self_evaluated_answers.csv` from this module.

| File | Purpose |
|---|---|
| `generate_and_self_evaluate.py` | Asks GPT-4o (`chatgpt-4o-latest`, temperature 0) for the authors of each of the 9,108 papers with the paper's few-shot prompt, then asks it to count matched (X), hallucinated (Y) and missed (Z) authors against the ground truth. |
| `compute_hallucination_rates.ipynb` | HR1-HR5 from X, Y, Z; statistics per citation bin, per field x bin (with 95% CIs) and per field; question length; OLS, mixed-effects and fractional-logit regressions. |

See the main [README](../README.md#6-module-04-gpt-4o-open-ended-recall-llm-self-evaluation) for how to run it.
