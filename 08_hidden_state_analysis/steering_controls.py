"""steering_controls: position, probe-direction, and control-spread ablations (Table A43).

Three ablations sharing one generation pass, so they stay comparable.

*Positions.* The vector is read at the P1 boundary but the hook writes it at
every position, including title tokens, whose perturbation degrades the answer
on its own. Three restricted variants separate boundary movement from generic
title corruption.

*Vectors.* steering_vector_analysis found cos(steering, probe) around 0.27-0.41, so most of the
steering vector is orthogonal to the probe. This steers along the probe
direction itself, norm-matched so the comparison is direction-only.

*Control spread.* steering.py's random control uses one seed, leaving the specificity
margin without a variance estimate. This runs the control arm at
``--n_random_seeds`` seeds.

All arms are INDUCE (-vector). Table A43 quotes this stage.

Usage
─────
  python 08_hidden_state_analysis/steering_controls.py \
      --model_path   /scratch/$USER/scholar_qwen/models/Qwen3-32B \
      --run_dir      data/probing/runs/qwen3_32b \
      --data_csv     data/probing/probing_prompts_9108.csv \
      --meta_json    results/hidden_state_analysis/steering/qwen3_32b/steering_meta.json \
      --layer 48 --alpha 2.0 \
      --vectors diff_of_means=results/hidden_state_analysis/steering/qwen3_32b/steering_vector.npy \
                probe=results/hidden_state_analysis/steering/qwen3_32b/probe_direction.npy \
      --positions all p1_and_gen prompt_only \
      --pool_mode band --hr2_lo 0.65 --hr2_hi 0.95 \
      --n_random_seeds 5 \
      --out_dir results/hidden_state_analysis/steering/qwen3_32b/position_probe
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))
from scholar_utils import score_authors
from extract_hidden_states import detect_model_config, load_model_and_tokenizer
from score_open_model import clean_generated_text, load_ground_truth
from steering import (
    MAX_NEW_TOKENS,
    _MATCHER,
    build_input,
    build_random_norm_matched_vector,
    get_decoder_layers,
    is_degenerate,
    repetition_metrics,
    summarize_condition,
)


def load_pool(scored: pd.DataFrame, paper_ids: list) -> pd.DataFrame:
    ids = {str(x) for x in paper_ids}
    return scored[scored["paper_id"].astype(str).isin(ids)].reset_index(drop=True)


def select_band_pool(scored: pd.DataFrame, meta: dict, hr2_lo: float, hr2_hi: float,
                     n_eval: int, seed: int) -> pd.DataFrame:
    """Sample papers in an HR2 band, disjoint from every steering pool."""
    exclude = {str(x) for ids in meta["pool_paper_ids"].values() for x in ids}
    cand = scored[(scored["hr2"] > hr2_lo) & (scored["hr2"] <= hr2_hi)].copy()
    cand = cand[~cand["paper_id"].astype(str).isin(exclude)]
    n = min(n_eval, len(cand))
    if n < n_eval:
        print(f"[steering_controls] WARNING: only {n} papers available in this band (requested {n_eval})")
    pool = cand.sample(n=n, random_state=seed) if n > 0 else cand
    overlap = set(pool["paper_id"].astype(str)) & exclude
    if overlap:
        raise AssertionError(f"band pool overlaps steering pools on {len(overlap)} ids")
    return pool.reset_index(drop=True)


RANDOM_SEED = 42

SUMMARY_FIELDS = [
    "condition", "vector", "positions", "alpha", "seed",
    "n", "mean_hr2", "hallucination_rate_pct", "mean_output_tokens",
    "pct_degenerate", "pct_dup_names", "mean_type_token_ratio",
]


# ─────────────────────────────────────────────────────────────────────────────
# Position-masked additive steering hook
# ─────────────────────────────────────────────────────────────────────────────

def make_position_masked_hook(vector: torch.Tensor, alpha: float, mode: str, prompt_len: int):
    """Additive steering (alpha * vector) applied only at selected positions.

    Inferred phase from the layer-output sequence length: during `generate`
    with the KV cache, prefill sees the whole prompt (seq == prompt_len) and
    each decode step sees one new token (seq == 1). P1 is the last prompt
    position (prompt_len - 1).

      all         : every position, both phases (== steering.make_steering_hook)
      p1_and_gen  : the P1 position at prefill + every generated token
      prompt_only : the pre-P1 prompt positions at prefill; nothing at decode
    """
    if mode not in ("all", "p1_and_gen", "prompt_only"):
        raise ValueError(f"unknown positions mode {mode!r}")
    delta = None

    def hook(module, inputs, output):
        nonlocal delta
        hidden = output[0] if isinstance(output, tuple) else output
        if delta is None or delta.device != hidden.device or delta.dtype != hidden.dtype:
            delta = (alpha * vector).to(device=hidden.device, dtype=hidden.dtype)
        seq = hidden.shape[1]
        if seq > 1:  # prefill: positions 0 .. seq-1, with P1 = prompt_len - 1
            p1 = prompt_len - 1
            if mode == "all":
                hidden = hidden + delta
            elif mode == "p1_and_gen":
                hidden[:, p1, :] = hidden[:, p1, :] + delta
            else:  # prompt_only — exclude P1 (and there are no generated tokens yet)
                if p1 > 0:
                    hidden[:, :p1, :] = hidden[:, :p1, :] + delta
        else:  # decode: a single freshly generated token
            if mode in ("all", "p1_and_gen"):
                hidden = hidden + delta
            # prompt_only: leave generated tokens untouched
        if isinstance(output, tuple):
            return (hidden,) + tuple(output[1:])
        return hidden

    return hook


@torch.no_grad()
def generate_steered(model, tokenizer, decoder_layers, layer_idx, device, chat_style,
                     title, year, max_new_tokens, vector, alpha, positions):
    """Greedy generation for one paper with a position-masked hook (None vector
    = plain baseline). The hook needs this paper's prompt_len, so it is built
    and registered per paper."""
    input_ids, attention_mask = build_input(tokenizer, title, year, chat_style, device)
    prompt_len = input_ids.shape[1]
    handle = None
    if vector is not None and alpha != 0.0:
        vec_t = torch.from_numpy(vector.astype(np.float32))
        hook = make_position_masked_hook(vec_t, alpha, positions, prompt_len)
        handle = decoder_layers[layer_idx].register_forward_hook(hook)
    try:
        gen_out = model.generate(
            input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens,
            do_sample=False, temperature=None, top_p=None, use_cache=True,
        )
    finally:
        if handle is not None:
            handle.remove()
    return tokenizer.decode(gen_out[0, prompt_len:], skip_special_tokens=True)


def run_condition(model, tokenizer, decoder_layers, layer_idx, device, chat_style,
                  papers, gt_dict, max_new_tokens, vector, alpha, positions) -> pd.DataFrame:
    """Same per-row schema as steering.run_condition, using the position-masked
    hook. `positions` is ignored when vector is None (baseline)."""
    rows = []
    for _, row in papers.iterrows():
        pid = row["paper_id"]
        gt = gt_dict.get(pid)
        if gt is None:
            continue
        raw_text = generate_steered(model, tokenizer, decoder_layers, layer_idx, device,
                                    chat_style, row["title"], row["year"], max_new_tokens,
                                    vector, alpha, positions)
        cleaned = clean_generated_text(raw_text)
        metrics = score_authors(gt["gt_authors"], cleaned, _MATCHER)
        rows.append({
            "paper_id": pid, "citation_count": row["citation_count"], "generated_text": raw_text,
            "hr2": metrics["hr2"],
            "n_output_tokens": len(tokenizer.encode(raw_text, add_special_tokens=False)),
            "degenerate": is_degenerate(raw_text), **repetition_metrics(raw_text),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────

def parse_vectors(pairs: list[str]) -> list[tuple[str, str]]:
    out = []
    for p in pairs:
        if "=" not in p:
            raise ValueError(f"--vectors entry {p!r} must be label=path")
        label, path = p.split("=", 1)
        out.append((label, path))
    return out


def norm_match(vec: np.ndarray, target_norm: float) -> np.ndarray:
    n = float(np.linalg.norm(vec))
    return (vec * (target_norm / n)).astype(np.float32) if n > 0 else vec.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="steering_controls: position-restricted + probe steering + control spread")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--data_csv", required=True)
    parser.add_argument("--meta_json", required=True)
    parser.add_argument("--layer", type=int, required=True, help="peak layer (48 Qwen / 22 Mistral)")
    parser.add_argument("--alpha", type=float, default=2.0, help="additive magnitude (default 2.0)")
    parser.add_argument("--vectors", nargs="+", required=True,
                        help="label=path pairs, e.g. diff_of_means=steering_vector.npy probe=probe_direction.npy")
    parser.add_argument("--norm_match_to", default=None,
                        help="label whose norm all vectors are scaled to (default: first --vectors label)")
    parser.add_argument("--positions", nargs="+", default=["all", "p1_and_gen", "prompt_only"],
                        choices=["all", "p1_and_gen", "prompt_only"])
    parser.add_argument("--pool_mode", choices=["band", "eval_high"], default="band")
    parser.add_argument("--hr2_lo", type=float, default=0.65)
    parser.add_argument("--hr2_hi", type=float, default=0.95)
    parser.add_argument("--n_eval", type=int, default=50)
    parser.add_argument("--n_random_seeds", type=int, default=5,
                        help="E4: number of random norm-matched control directions (induce, positions=all)")
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--quantize", default="none", choices=["none", "8bit", "4bit"])
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--chat_style", default=None)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_dir = Path(args.out_dir)
    vectors = parse_vectors(args.vectors)
    ref_label = args.norm_match_to or vectors[0][0]

    scored = pd.read_csv(Path(args.run_dir) / "scored.tsv", sep="\t")
    scored["citation_count"] = pd.to_numeric(scored["citation_count"], errors="coerce").fillna(0)
    scored["hr2"] = pd.to_numeric(scored["hr2"], errors="coerce")
    gt_dict = load_ground_truth(args.data_csv)
    meta = json.load(open(args.meta_json))

    # ── Load + norm-match vectors; align sign to the reference (induce = −vector) ─
    raw = {label: np.load(path).astype(np.float32) for label, path in vectors}
    if ref_label not in raw:
        raise ValueError(f"--norm_match_to {ref_label!r} not among vector labels {list(raw)}")
    ref_norm = float(np.linalg.norm(raw[ref_label]))
    steer_vecs = {}
    for label, v in raw.items():
        cos = float(np.dot(v, raw[ref_label]) / (np.linalg.norm(v) * ref_norm)) if label != ref_label else 1.0
        vv = v.copy()
        if cos < 0:  # keep every vector pointing the same way as the reference
            vv = -vv
            print(f"[steering_controls] vector {label!r} had cos={cos:.3f} < 0 vs {ref_label!r} — flipping sign")
        steer_vecs[label] = norm_match(vv, ref_norm)
        print(f"[steering_controls] vector {label!r}: raw norm={np.linalg.norm(v):.4f} -> matched {ref_norm:.4f} "
              f"(cos vs {ref_label}={cos:.3f})")

    # ── Pool ─────────────────────────────────────────────────────────────────
    if args.pool_mode == "band":
        pool = select_band_pool(scored, meta, args.hr2_lo, args.hr2_hi, args.n_eval, args.seed)
        pool_tag = f"band_hr2_{args.hr2_lo}_{args.hr2_hi}"
    else:
        eval_all = load_pool(scored, meta["pool_paper_ids"]["eval_high"])
        n = min(args.n_eval, len(eval_all))
        pool = (eval_all.sample(n=n, random_state=args.seed) if n > 0 else eval_all).reset_index(drop=True)
        pool_tag = "eval_high"
    print(f"[steering_controls] pool={pool_tag} n={len(pool)}")
    if pool.empty:
        print("[steering_controls] pool empty — nothing to run; exiting cleanly.")
        return

    registry_cfg = detect_model_config(args.model_path)
    chat_style = args.chat_style or (registry_cfg["chat_style"] if registry_cfg else "chatml")
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    print(f"[steering_controls] loading model {args.model_path} (chat_style={chat_style!r})")
    model, tokenizer = load_model_and_tokenizer(args.model_path, dtype, args.quantize)
    device = next(model.parameters()).device
    decoder_layers = get_decoder_layers(model)
    if args.layer >= len(decoder_layers):
        raise ValueError(f"--layer {args.layer} out of range for a {len(decoder_layers)}-layer model")
    if args.layer == len(decoder_layers) - 1:
        print(f"[steering_controls] WARNING: --layer {args.layer} is the final layer — steering degenerates there.")

    # ── Streamed outputs ───────────────────────────────────────────────────────
    res_path = out_dir / f"position_probe_results_{pool_tag}.csv"
    pred_path = out_dir / f"position_probe_predictions_{pool_tag}.tsv"
    res_f = open(res_path, "w", newline="")
    writer = csv.DictWriter(res_f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
    writer.writeheader()
    res_f.flush()
    pred_state = {"header": False}

    def emit(df: pd.DataFrame, **tags) -> dict:
        summ = summarize_condition(df, **tags)
        writer.writerow({k: summ.get(k, "") for k in SUMMARY_FIELDS})
        res_f.flush()
        if not df.empty:
            tagged = df.copy()
            for k, v in tags.items():
                tagged[k] = v
            tagged.to_csv(pred_path, sep="\t", index=False, mode="a", header=not pred_state["header"])
            pred_state["header"] = True
        return summ

    try:
        # Baseline (regenerated in-run; bf16 nondeterminism).
        print("[steering_controls] baseline (no steering)")
        base_df = run_condition(model, tokenizer, decoder_layers, args.layer, device, chat_style,
                                pool, gt_dict, args.max_new_tokens, None, 0.0, "all")
        base = emit(base_df, condition="baseline", vector="none", positions="none",
                    alpha=0.0, seed=-1)
        print(f"[steering_controls] baseline HR%={base['hallucination_rate_pct']}  mean_hr2={base['mean_hr2']}")

        # E3 × E5: each vector × each position mode, INDUCE (−vector).
        for label, _ in vectors:
            for pos in args.positions:
                print(f"[steering_controls] induce vector={label} positions={pos} alpha={args.alpha}")
                df = run_condition(model, tokenizer, decoder_layers, args.layer, device, chat_style,
                                   pool, gt_dict, args.max_new_tokens,
                                   -steer_vecs[label], args.alpha, pos)
                s = emit(df, condition="induce", vector=label, positions=pos, alpha=args.alpha, seed=-1)
                print(f"[steering_controls]   -> HR%={s['hallucination_rate_pct']}  mean_hr2={s['mean_hr2']}")

        # E4: random norm-matched control induce, K seeds, positions=all.
        dim = raw[ref_label].shape[0]
        for k in range(args.n_random_seeds):
            rseed = args.seed + 500 + k
            rvec = build_random_norm_matched_vector(dim, ref_norm, seed=rseed)
            print(f"[steering_controls] random-control induce seed={rseed} (positions=all)")
            df = run_condition(model, tokenizer, decoder_layers, args.layer, device, chat_style,
                               pool, gt_dict, args.max_new_tokens, -rvec, args.alpha, "all")
            s = emit(df, condition="random_control", vector="random", positions="all",
                     alpha=args.alpha, seed=rseed)
            print(f"[steering_controls]   -> HR%={s['hallucination_rate_pct']}  mean_hr2={s['mean_hr2']}")
    finally:
        res_f.close()

    meta_out = {
        "model_path": args.model_path, "layer": args.layer, "alpha": args.alpha,
        "pool": pool_tag, "n_eval": len(pool), "positions": args.positions,
        "vectors": {label: path for label, path in vectors},
        "norm_match_to": ref_label, "reference_norm": ref_norm,
        "n_random_seeds": args.n_random_seeds,
        "eval_pool_paper_ids": list(pool["paper_id"].astype(str)),
        "note": ("All arms are INDUCE (−vector). E3=positions ablation, E5=probe-vs-diff-of-means "
                 "(norm-matched), E4=random-control spread over seeds. Compare each induce HR% "
                 "against baseline and against the random-control seed spread."),
    }
    with open(out_dir / f"position_probe_meta_{pool_tag}.json", "w") as f:
        json.dump(meta_out, f, indent=2)
    print(f"[steering_controls] wrote {res_path}, {pred_path}, and meta.")


if __name__ == "__main__":
    main()
