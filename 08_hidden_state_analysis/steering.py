"""steering: RepE-style activation steering (Appendix A7.7).

Tests whether editing the decoded prevalence direction *causally* changes
hallucination, since prevalence_decoding's decodability is only correlational.

The correct pole is drawn from the best-attributed papers at any citation
level, not from low-citation correct ones: on the scored data only 63/9108
Mistral papers have HR2 == 0 and essentially none are low-citation, so
"low-citation and correct" is nearly empty. The hallucinated pole stays
low-citation with high HR2. Both vector pools are disjoint from every
evaluation pool by construction.

Controls: a shuffled-label vector, a bidirectional test, and output length plus
a degeneracy flag, so an HR2 gain bought with garbage output is visible.

Usage
─────
  python 08_hidden_state_analysis/steering.py \
      --model_path   Qwen/Qwen3-32B \
      --run_dir      data/probing/runs/qwen3_32b \
      --data_csv     data/probing/probing_prompts_9108.csv \
      --layer        48 \
      --out_dir      results/hidden_state_analysis/steering/qwen3_32b

  python 08_hidden_state_analysis/steering.py \
      --model_path   /scratch/$USER/scholar_qwen/models/Mistral-Small-3.2-24B \
      --run_dir      data/probing/runs/mistral_small_3_2_24b \
      --data_csv     data/probing/probing_prompts_9108.csv \
      --layer        22 \
      --out_dir      /scratch/$USER/scholar_qwen/steering_mistral_small_3_2_24b
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))
from scholar_utils import NameMatcher, score_authors
from extract_hidden_states import (
    MAX_PROMPT_TOKENS,
    build_prompt,
    detect_model_config,
    load_model_and_tokenizer,
)
from score_open_model import clean_generated_text, load_ground_truth

RANDOM_SEED = 42
MAX_NEW_TOKENS = 96
_MATCHER = NameMatcher()


# ─────────────────────────────────────────────────────────────────────────────
# Model plumbing shared with extract_hidden_states
# ─────────────────────────────────────────────────────────────────────────────

def get_decoder_layers(model) -> torch.nn.ModuleList:
    """
    Locate the decoder block list across the model shapes we use:
      * plain causal LMs (Qwen3ForCausalLM, MistralForCausalLM, ...)
        -> model.model.layers
      * Mistral3's ForConditionalGeneration class (vision-language wrapper registered
        in extract_hidden_states) -> model.model is Mistral3Model, which holds
        language_model / vision_tower / multi_modal_projector, not layers
        directly. transformers <=4.57 exposed a `model.language_model`
        convenience property forwarding to `model.model.language_model`;
        that property was removed in transformers 5.x, so we check both the
        property and the underlying submodule directly.
    """
    base = getattr(model, "model", model)
    if hasattr(base, "layers"):
        return base.layers
    if hasattr(base, "language_model") and hasattr(base.language_model, "layers"):
        return base.language_model.layers
    if hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        return model.language_model.layers
    raise AttributeError(f"Could not locate decoder layer list on {type(model)}")


def build_input(tokenizer, title: str, year: str, chat_style: str, device) -> tuple:
    prompt = build_prompt(tokenizer, title, year, chat_style)
    if isinstance(prompt, list):
        prompt = prompt[:MAX_PROMPT_TOKENS]
        input_ids = torch.tensor([prompt], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
    else:
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_PROMPT_TOKENS)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
    return input_ids, attention_mask


@torch.no_grad()
def extract_p1(model, tokenizer, title: str, year: str, chat_style: str, device, layer: int) -> np.ndarray:
    input_ids, attention_mask = build_input(tokenizer, title, year, chat_style, device)
    out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    h = out.hidden_states[layer + 1][0, -1, :]
    return h.float().cpu().numpy()


def make_steering_hook(vector: torch.Tensor, alpha: float):
    delta = None  # resolved lazily to the right device/dtype on first call

    def hook(module, inputs, output):
        nonlocal delta
        hidden = output[0] if isinstance(output, tuple) else output
        if delta is None or delta.device != hidden.device or delta.dtype != hidden.dtype:
            delta = (alpha * vector).to(device=hidden.device, dtype=hidden.dtype)
        hidden = hidden + delta
        if isinstance(output, tuple):
            return (hidden,) + tuple(output[1:])
        return hidden

    return hook


@torch.no_grad()
def generate_text(model, tokenizer, title: str, year: str, chat_style: str, device,
                   max_new_tokens: int) -> str:
    input_ids, attention_mask = build_input(tokenizer, title, year, chat_style, device)
    gen_out = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
    )
    prompt_len = input_ids.shape[1]
    return tokenizer.decode(gen_out[0, prompt_len:], skip_special_tokens=True)


def is_degenerate(text: str) -> bool:
    """Cheap collateral-damage proxy: empty, or one character repeated dominates.

    Only catches literal character spam (e.g. "aaaaaa..."). It does NOT catch
    word/phrase-level repetition (e.g. the same author name repeated several
    times in an otherwise varied list) — that showed up in practice at high
    alpha and is caught separately by repetition_metrics() below. Keep both:
    this one is cheap and catches the worst failures; repetition_metrics()
    catches the milder, more common ones a qualitative read would flag.
    """
    cleaned = text.strip()
    if len(cleaned) < 2:
        return True
    most_common_frac = max(cleaned.count(c) for c in set(cleaned)) / len(cleaned)
    return most_common_frac > 0.6


def repetition_metrics(text: str) -> dict:
    """Word/phrase-level repetition signal, complementing is_degenerate().

    dup_name_in_list: True if the semicolon-delimited author list contains
    the same name string twice — a well-formed list shouldn't repeat names,
    so any repeat is a sign the model looped rather than recalled distinct
    (even if wrong) authors.
    type_token_ratio: unique words / total words. Lower means more
    repetitive; this is a continuous signal, not a hard degenerate/not
    threshold, so it's reported as-is rather than collapsed into a boolean.
    """
    text = str(text)
    names = [n.strip() for n in text.split(";") if n.strip()]
    dup_name_in_list = len(names) != len(set(names))
    words = text.split()
    ttr = len(set(words)) / len(words) if words else 1.0
    return {"dup_name_in_list": dup_name_in_list, "type_token_ratio": round(ttr, 4)}


# ─────────────────────────────────────────────────────────────────────────────
# Paper-pool selection
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Pools:
    vector_correct: pd.DataFrame
    vector_hallucinated: pd.DataFrame
    eval_low: pd.DataFrame        # held-out low-citation hallucinated papers
    eval_high: pd.DataFrame       # held-out high-citation well-attributed papers


def select_pools(scored: pd.DataFrame, args) -> Pools:
    rng_seed = args.seed

    hallucinated_all = scored[
        (scored["citation_count"] <= args.low_citation_max)
        & (scored["hr2"] >= args.hallucinated_hr2_min)
    ]
    correct_all = scored[
        (scored["citation_count"] >= args.correct_citation_min)
        & (scored["hr2"] <= args.correct_hr2_max)
    ].sort_values("hr2", ascending=True)

    print(f"[steering] pool sizes before split: hallucinated_low_citation="
          f"{len(hallucinated_all)}  well_attributed={len(correct_all)}")
    if len(correct_all) < 2 * args.n_vector_papers:
        print(f"[steering] WARNING: only {len(correct_all)} well-attributed papers "
              f"(HR2<={args.correct_hr2_max}, citation>={args.correct_citation_min}) available; "
              f"requested {args.n_vector_papers} for the vector pool + "
              f"{args.n_eval_papers_high} for the held-out high-citation eval pool. "
              "Both will shrink below the requested size — see counts below.")

    vector_correct = correct_all.head(args.n_vector_papers)
    remaining_correct = correct_all.iloc[len(vector_correct):]
    eval_high = remaining_correct.head(args.n_eval_papers_high)

    vector_hallucinated = hallucinated_all.sample(
        n=min(args.n_vector_papers, len(hallucinated_all)), random_state=rng_seed
    )
    remaining_hallucinated = hallucinated_all.drop(vector_hallucinated.index)
    eval_low = remaining_hallucinated.sample(
        n=min(args.n_eval_papers, len(remaining_hallucinated)), random_state=rng_seed + 1
    )

    pools = Pools(vector_correct, vector_hallucinated, eval_low, eval_high)
    for name, df in [
        ("vector_correct", pools.vector_correct),
        ("vector_hallucinated", pools.vector_hallucinated),
        ("eval_low (held out, disjoint from vector pools)", pools.eval_low),
        ("eval_high (held out, disjoint from vector pools)", pools.eval_high),
    ]:
        print(f"[steering]   {name}: n={len(df)}")

    all_ids = [set(df["paper_id"]) for df in [
        pools.vector_correct, pools.vector_hallucinated, pools.eval_low, pools.eval_high
    ]]
    for i in range(len(all_ids)):
        for j in range(i + 1, len(all_ids)):
            overlap = all_ids[i] & all_ids[j]
            if overlap:
                raise AssertionError(f"Pool {i} and pool {j} overlap on {len(overlap)} paper_ids — bug")
    print("[steering] confirmed all four pools are pairwise disjoint")

    if len(pools.vector_correct) < 10 or len(pools.vector_hallucinated) < 10:
        raise RuntimeError(
            "Fewer than 10 papers available for one of the vector-building pools — "
            "cannot build a meaningful steering vector. Loosen --correct_hr2_max / "
            "--hallucinated_hr2_min / --low_citation_max / --correct_citation_min."
        )
    return pools


# ─────────────────────────────────────────────────────────────────────────────
# Vector construction
# ─────────────────────────────────────────────────────────────────────────────

def build_vector(p1_correct: np.ndarray, p1_hallucinated: np.ndarray) -> np.ndarray:
    return p1_correct.mean(axis=0) - p1_hallucinated.mean(axis=0)


def build_shuffled_control_vector(
    p1_correct: np.ndarray, p1_hallucinated: np.ndarray, seed: int
) -> np.ndarray:
    """Same construction as build_vector, but with pool labels shuffled —
    mirrors this repo's prevalence_decoding shuffled-citation negative controls.

    Caveat found empirically: a shuffled-label diff-of-means
    has much smaller norm than the real steering vector (~1/7-1/8x in our
    runs), because if the true groups really differ along some direction,
    random relabeling averages that difference away. That makes this a
    weaker perturbation than the real vector, not a same-magnitude control —
    fine as a "is there signal in a random split" check, but not a fair
    "same magnitude, wrong direction" test. Use
    build_random_norm_matched_vector() for that.
    """
    pooled = np.concatenate([p1_correct, p1_hallucinated], axis=0)
    n_correct = len(p1_correct)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(pooled))
    shuffled_correct = pooled[perm[:n_correct]]
    shuffled_hallucinated = pooled[perm[n_correct:]]
    return shuffled_correct.mean(axis=0) - shuffled_hallucinated.mean(axis=0)


def build_random_norm_matched_vector(dim: int, target_norm: float, seed: int) -> np.ndarray:
    """A genuinely random direction, scaled to match a reference vector's
    norm — the standard "is it magnitude or direction" control from the
    steering literature (e.g. Rimsky et al. compare a steering vector's
    effect against a random vector of the same norm)."""
    rng = np.random.RandomState(seed)
    v = rng.normal(size=dim)
    v = v / np.linalg.norm(v)
    return (v * target_norm).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Running conditions
# ─────────────────────────────────────────────────────────────────────────────

def run_condition(
    model, tokenizer, decoder_layers, layer_idx: int, device, chat_style: str,
    papers: pd.DataFrame, gt_dict: dict, max_new_tokens: int,
    vector: np.ndarray | None, alpha: float,
) -> pd.DataFrame:
    handle = None
    if vector is not None and alpha != 0.0:
        vec_t = torch.from_numpy(vector.astype(np.float32))
        hook = make_steering_hook(vec_t, alpha)
        handle = decoder_layers[layer_idx].register_forward_hook(hook)

    rows = []
    try:
        for _, row in papers.iterrows():
            pid = row["paper_id"]
            gt = gt_dict.get(pid)
            if gt is None:
                continue
            raw_text = generate_text(
                model, tokenizer, row["title"], row["year"], chat_style, device, max_new_tokens
            )
            cleaned = clean_generated_text(raw_text)
            metrics = score_authors(gt["gt_authors"], cleaned, _MATCHER)
            rows.append({
                "paper_id": pid,
                "citation_count": row["citation_count"],
                "generated_text": raw_text,
                "hr2": metrics["hr2"],
                "n_output_tokens": len(tokenizer.encode(raw_text, add_special_tokens=False)),
                "degenerate": is_degenerate(raw_text),
                **repetition_metrics(raw_text),
            })
    finally:
        if handle is not None:
            handle.remove()

    return pd.DataFrame(rows)


def summarize_condition(df: pd.DataFrame, **tags) -> dict:
    if df.empty:
        return {**tags, "n": 0, "mean_hr2": float("nan"), "hallucination_rate_pct": float("nan"),
                "mean_output_tokens": float("nan"), "pct_degenerate": float("nan"),
                "pct_dup_names": float("nan"), "mean_type_token_ratio": float("nan")}
    return {
        **tags,
        "n": len(df),
        "mean_hr2": round(float(df["hr2"].mean()), 6),
        "hallucination_rate_pct": round(100.0 * float((df["hr2"] > 0.5).mean()), 2),
        "mean_output_tokens": round(float(df["n_output_tokens"].mean()), 2),
        "pct_degenerate": round(100.0 * float(df["degenerate"].mean()), 2),
        "pct_dup_names": round(100.0 * float(df["dup_name_in_list"].mean()), 2),
        "mean_type_token_ratio": round(float(df["type_token_ratio"].mean()), 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="steering: causal activation-steering intervention")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--run_dir", required=True,
                        help="Directory containing scored.tsv from extract_hidden_states+score_open_model for this model")
    parser.add_argument("--data_csv", required=True,
                        help="Source evaluation CSV (same one passed to extract_hidden_states/score_open_model) — "
                             "needed to look up ground-truth authors for re-scoring steered generations")
    parser.add_argument("--layer", type=int, required=True,
                        help="Decoder block index to steer (Qwen3-32B: 48, Mistral-Small-3.2-24B: 22)")
    parser.add_argument("--alphas", type=float, nargs="+", default=[1.0, 2.0])
    parser.add_argument("--n_vector_papers", type=int, default=100)
    parser.add_argument("--n_eval_papers", type=int, default=100,
                        help="Held-out low-citation hallucinated papers for the main causal test")
    parser.add_argument("--n_eval_papers_high", type=int, default=50,
                        help="Held-out high-citation well-attributed papers for the bidirectional test "
                             "(kept smaller by default — this pool is scarce, see module docstring)")
    parser.add_argument("--n_control_shuffles", type=int, default=1)
    parser.add_argument("--low_citation_max", type=float, default=8,
                        help="citation_count <= this defines the 'hallucinated pole' pool")
    parser.add_argument("--correct_citation_min", type=float, default=0,
                        help="citation_count >= this defines the 'well-attributed pole' pool "
                             "(0 = no restriction; see module docstring for why low-citation-AND-correct "
                             "is not a viable pool in this data)")
    parser.add_argument("--hallucinated_hr2_min", type=float, default=1.0)
    parser.add_argument("--correct_hr2_max", type=float, default=0.5,
                        help="Matches this repo's own is_hallucinated = hr2 > 0.5 threshold")
    parser.add_argument("--direction", choices=["low_to_high", "both"], default="both")
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--quantize", default="none", choices=["none", "8bit", "4bit"])
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--chat_style", default=None)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_dir = Path(args.out_dir)

    scored_path = Path(args.run_dir) / "scored.tsv"
    print(f"[steering] loading {scored_path}")
    scored = pd.read_csv(scored_path, sep="\t")
    scored["citation_count"] = pd.to_numeric(scored["citation_count"], errors="coerce").fillna(0)

    print(f"[steering] loading ground truth: {args.data_csv}")
    gt_dict = load_ground_truth(args.data_csv)

    pools = select_pools(scored, args)

    registry_cfg = detect_model_config(args.model_path)
    chat_style = args.chat_style or (registry_cfg["chat_style"] if registry_cfg else "chatml")
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    print(f"[steering] loading model {args.model_path} (chat_style={chat_style!r})")
    model, tokenizer = load_model_and_tokenizer(args.model_path, dtype, args.quantize)
    device = next(model.parameters()).device
    decoder_layers = get_decoder_layers(model)
    n_layers = len(decoder_layers)
    if args.layer >= n_layers:
        raise ValueError(f"--layer {args.layer} out of range for a {n_layers}-layer model")

    # ── Extract P1 for both vector-building pools ───────────────────────────
    print(f"[steering] extracting P1 at layer {args.layer} for "
          f"{len(pools.vector_correct)} correct + {len(pools.vector_hallucinated)} hallucinated papers")
    p1_correct = np.stack([
        extract_p1(model, tokenizer, r["title"], r["year"], chat_style, device, args.layer)
        for _, r in pools.vector_correct.iterrows()
    ])
    p1_hallucinated = np.stack([
        extract_p1(model, tokenizer, r["title"], r["year"], chat_style, device, args.layer)
        for _, r in pools.vector_hallucinated.iterrows()
    ])

    steering_vector = build_vector(p1_correct, p1_hallucinated)
    control_vectors = [
        build_shuffled_control_vector(p1_correct, p1_hallucinated, seed=args.seed + 100 + i)
        for i in range(args.n_control_shuffles)
    ]
    print(f"[steering] steering vector norm={np.linalg.norm(steering_vector):.4f}  "
          f"control vector norm(s)={[round(float(np.linalg.norm(v)), 4) for v in control_vectors]}")

    np.save(out_dir / "steering_vector.npy", steering_vector)
    for i, cv in enumerate(control_vectors):
        np.save(out_dir / f"control_vector_{i}.npy", cv)

    # ── Run conditions ────────────────────────────────────────────────────────
    results_summary: list[dict] = []
    predictions: list[pd.DataFrame] = []

    def record(df: pd.DataFrame, **tags) -> None:
        if not df.empty:
            tagged = df.copy()
            for k, v in tags.items():
                tagged[k] = v
            predictions.append(tagged)
        results_summary.append(summarize_condition(df, **tags))

    print("[steering] === low-citation eval pool (main causal test) ===")
    print("[steering] baseline (no steering)")
    df = run_condition(model, tokenizer, decoder_layers, args.layer, device, chat_style,
                        pools.eval_low, gt_dict, args.max_new_tokens, vector=None, alpha=0.0)
    record(df, pool="eval_low", condition="baseline", direction="none", alpha=0.0)

    for alpha in args.alphas:
        print(f"[steering] steering toward 'well-attributed' direction, alpha={alpha}")
        df = run_condition(model, tokenizer, decoder_layers, args.layer, device, chat_style,
                            pools.eval_low, gt_dict, args.max_new_tokens,
                            vector=steering_vector, alpha=alpha)
        record(df, pool="eval_low", condition="steering", direction="toward_cited", alpha=alpha)

        for ci, cv in enumerate(control_vectors):
            print(f"[steering] control vector {ci}, alpha={alpha}")
            df = run_condition(model, tokenizer, decoder_layers, args.layer, device, chat_style,
                                pools.eval_low, gt_dict, args.max_new_tokens,
                                vector=cv, alpha=alpha)
            record(df, pool="eval_low", condition=f"control_{ci}", direction="toward_cited", alpha=alpha)

    if args.direction == "both" and len(pools.eval_high) > 0:
        print("[steering] === high-citation eval pool (bidirectional test) ===")
        print("[steering] baseline (no steering)")
        df = run_condition(model, tokenizer, decoder_layers, args.layer, device, chat_style,
                            pools.eval_high, gt_dict, args.max_new_tokens, vector=None, alpha=0.0)
        record(df, pool="eval_high", condition="baseline", direction="none", alpha=0.0)

        for alpha in args.alphas:
            print(f"[steering] steering toward 'hallucinated' direction, alpha={alpha}")
            df = run_condition(model, tokenizer, decoder_layers, args.layer, device, chat_style,
                                pools.eval_high, gt_dict, args.max_new_tokens,
                                vector=-steering_vector, alpha=alpha)
            record(df, pool="eval_high", condition="steering", direction="toward_hallucinated", alpha=alpha)
    elif args.direction == "both":
        print("[steering] --direction both requested but eval_high pool is empty — skipping bidirectional test")

    # ── Write outputs ────────────────────────────────────────────────────────
    summary_df = pd.DataFrame(results_summary)
    summary_df.to_csv(out_dir / "steering_results.csv", index=False)
    print(f"[steering] wrote {out_dir / 'steering_results.csv'}")

    if predictions:
        pd.concat(predictions, ignore_index=True).to_csv(
            out_dir / "steering_predictions.tsv", sep="\t", index=False
        )
        print(f"[steering] wrote {out_dir / 'steering_predictions.tsv'}")

    meta = {
        "model_path": args.model_path,
        "layer": args.layer,
        "alphas": args.alphas,
        "n_control_shuffles": args.n_control_shuffles,
        "steering_vector_norm": float(np.linalg.norm(steering_vector)),
        "control_vector_norms": [float(np.linalg.norm(v)) for v in control_vectors],
        "pool_sizes": {
            "vector_correct": len(pools.vector_correct),
            "vector_hallucinated": len(pools.vector_hallucinated),
            "eval_low": len(pools.eval_low),
            "eval_high": len(pools.eval_high),
        },
        "pool_paper_ids": {
            "vector_correct": pools.vector_correct["paper_id"].tolist(),
            "vector_hallucinated": pools.vector_hallucinated["paper_id"].tolist(),
            "eval_low": pools.eval_low["paper_id"].tolist(),
            "eval_high": pools.eval_high["paper_id"].tolist(),
        },
        "selection_args": {
            "low_citation_max": args.low_citation_max,
            "correct_citation_min": args.correct_citation_min,
            "hallucinated_hr2_min": args.hallucinated_hr2_min,
            "correct_hr2_max": args.correct_hr2_max,
        },
    }
    with open(out_dir / "steering_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[steering] wrote {out_dir / 'steering_meta.json'}")

    print("\n[steering] === summary ===")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
