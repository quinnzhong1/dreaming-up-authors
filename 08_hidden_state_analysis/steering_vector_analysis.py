"""steering_vector_analysis: is the diff-of-means steering vector the prevalence direction?

CPU-only; loads no model, working from the cached P1 states and steering's
saved vectors. Compares by pairwise cosine: steering's vector, the same
construction recomputed here, the Fisher/LDA direction with Ledoit-Wolf
shrinkage (the raw covariance is singular at D >> n), and the prevalence_decoding probe
direction mapped back to raw hidden space.

Usage
─────
  python 08_hidden_state_analysis/steering_vector_analysis.py \
      --run_dir      data/probing/runs/qwen3_32b \
      --layer        48 \
      --meta_json    results/hidden_state_analysis/steering/qwen3_32b/steering_meta.json \
      --steering_vector_path results/hidden_state_analysis/steering/qwen3_32b/steering_vector.npy \
      --out_dir      results/hidden_state_analysis/steering/qwen3_32b
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
# Reuse prevalence_decoding's exact embeddings.h5 alignment logic rather than
# reimplementing it. prevalence_decoding lazily binds np/pd/h5py inside
# ensure_dependencies() (which also imports matplotlib); we only need the
# numeric parts, so bind those globals directly and skip the plotting import.
import prevalence_decoding as s8  # noqa: E402

s8.np = np
s8.pd = pd
s8.h5py = h5py
from prevalence_decoding import (  # noqa: E402
    N_SVD_COMPONENTS,
    RANDOM_SEED,
    RIDGE_ALPHAS,
    align_rows,
    find_h5_path,
    find_scored_path,
    layer_position,
    load_layer_indices,
)


# ─────────────────────────────────────────────────────────────────────────────
# Direction construction
# ─────────────────────────────────────────────────────────────────────────────

def diff_of_means(p1_correct: np.ndarray, p1_hallucinated: np.ndarray) -> np.ndarray:
    return p1_correct.mean(axis=0) - p1_hallucinated.mean(axis=0)


def lda_fisher_direction(p1_correct: np.ndarray, p1_hallucinated: np.ndarray) -> np.ndarray:
    """Fisher/LDA direction Σ_w⁻¹(μ_c − μ_h) with Ledoit-Wolf shrinkage.

    D ≫ n, so the empirical within-class covariance is singular; Ledoit-Wolf
    shrinks it toward a scaled identity, giving a well-conditioned precision
    matrix. Each class is centered by its own mean before pooling so the
    estimate is the *within*-class covariance, not the total covariance.
    """
    from sklearn.covariance import LedoitWolf

    mu_c = p1_correct.mean(axis=0)
    mu_h = p1_hallucinated.mean(axis=0)
    pooled_centered = np.vstack([p1_correct - mu_c, p1_hallucinated - mu_h])
    lw = LedoitWolf().fit(pooled_centered)
    w = lw.precision_ @ (mu_c - mu_h)
    return w.astype(np.float32)


def probe_ridge_direction(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """The prevalence_decoding probe as a single raw-hidden-space direction.

    Fits one PCA(100)→StandardScaler→RidgeCV pipeline on the full set (not
    fold-safe — we want the direction, not an unbiased R²) and maps the
    Ridge coefficients back through the scaler and PCA to raw hidden space.
    PCA/StandardScaler centering only shifts the intercept, so it does not
    affect the direction: w = components_.T @ (coef_ / scaler.scale_).
    """
    from sklearn.decomposition import PCA
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    k = min(N_SVD_COMPONENTS, X.shape[0] - 1, X.shape[1])
    pca = PCA(n_components=k, svd_solver="randomized", random_state=RANDOM_SEED)
    Z = pca.fit_transform(X)
    scaler = StandardScaler()
    Zs = scaler.fit_transform(Z)
    ridge = RidgeCV(alphas=RIDGE_ALPHAS)
    ridge.fit(Zs, y)
    w = pca.components_.T @ (ridge.coef_ / scaler.scale_)
    return w.astype(np.float32), float(ridge.alpha_)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


# ─────────────────────────────────────────────────────────────────────────────

def rows_for_ids(aligned: pd.DataFrame, paper_ids: list) -> list[int]:
    pid_to_row = {str(pid): i for i, pid in enumerate(aligned["paper_id"].astype(str))}
    rows = [pid_to_row[str(p)] for p in paper_ids if str(p) in pid_to_row]
    missing = len(paper_ids) - len(rows)
    if missing:
        print(f"[steering_vector_analysis] WARNING: {missing}/{len(paper_ids)} pool paper_ids not found "
              f"in embeddings.h5 alignment — using the {len(rows)} that were found")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="steering_vector_analysis: steering-vector direction analysis (cosine + LDA)")
    parser.add_argument("--run_dir", required=True,
                        help="Directory with scored.tsv and embeddings.h5 for this model")
    parser.add_argument("--layer", type=int, required=True,
                        help="Optimal P1 layer (Qwen3-32B: 48, Mistral-Small-3.2-24B: 22)")
    parser.add_argument("--meta_json", required=True,
                        help="steering_meta.json from the completed steering run (pool paper IDs)")
    parser.add_argument("--steering_vector_path", required=True,
                        help="steering_vector.npy from the completed steering run")
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_dir = Path(args.out_dir)

    # ── Load cached P1 at the optimal layer for all aligned papers ──────────
    scored = pd.read_csv(find_scored_path(Path(args.run_dir)), sep="\t")
    scored["citation_count"] = pd.to_numeric(scored["citation_count"], errors="coerce").fillna(0)
    h5_path = find_h5_path(Path(args.run_dir))
    print(f"[steering_vector_analysis] loading P1 at layer {args.layer} from {h5_path}")
    with h5py.File(h5_path, "r") as h5:
        if "per_query/p1" not in h5:
            raise KeyError(f"{h5_path} has no per_query/p1 dataset")
        layers = load_layer_indices(h5)
        pos = layer_position(layers, args.layer)
        valid_idx, aligned = align_rows(h5, scored)
        X = h5["per_query/p1"][:, pos, :][valid_idx].astype(np.float32)
    print(f"[steering_vector_analysis] aligned P1 matrix: {X.shape[0]} papers x {X.shape[1]} dims")

    y = np.log1p(pd.to_numeric(aligned["citation_count"], errors="coerce").fillna(0).to_numpy(np.float32))

    # ── Reconstruct steering's vector-building pools ────────────────────────
    meta = json.load(open(args.meta_json))
    correct_rows = rows_for_ids(aligned, meta["pool_paper_ids"]["vector_correct"])
    hallu_rows = rows_for_ids(aligned, meta["pool_paper_ids"]["vector_hallucinated"])
    p1_correct = X[correct_rows]
    p1_hallucinated = X[hallu_rows]
    print(f"[steering_vector_analysis] pools from meta_json: correct={len(p1_correct)}  hallucinated={len(p1_hallucinated)}")

    # ── Build the directions ────────────────────────────────────────────────
    directions: dict[str, np.ndarray] = {}

    # The key keeps its original name: it is the row label in the shipped direction_cosines.csv.
    directions["stage14_diff_of_means"] = np.load(args.steering_vector_path).astype(np.float32)
    directions["h5_diff_of_means"] = diff_of_means(p1_correct, p1_hallucinated)
    directions["lda_fisher"] = lda_fisher_direction(p1_correct, p1_hallucinated)
    probe_dir, probe_alpha = probe_ridge_direction(X, y)
    directions["probe_ridge"] = probe_dir

    dim = X.shape[1]
    for name, v in directions.items():
        if v.shape[0] != dim:
            raise ValueError(f"Direction {name!r} has dim {v.shape[0]} != P1 dim {dim}")

    np.save(out_dir / "lda_vector.npy", directions["lda_fisher"])
    np.save(out_dir / "probe_direction.npy", directions["probe_ridge"])

    # ── Pairwise cosine matrix (long format) ────────────────────────────────
    names = list(directions.keys())
    rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            rows.append({
                "direction_a": a,
                "direction_b": b,
                "cosine": round(cosine(directions[a], directions[b]), 6),
                "is_headline": (a == "stage14_diff_of_means" and b == "probe_ridge"),
            })
    cos_df = pd.DataFrame(rows)
    cos_df.to_csv(out_dir / "direction_cosines.csv", index=False)
    print(f"[steering_vector_analysis] wrote {out_dir / 'direction_cosines.csv'}")

    meta_out = {
        "run_dir": args.run_dir,
        "layer": args.layer,
        "n_papers_probe_fit": int(X.shape[0]),
        "hidden_dim": int(dim),
        "pool_sizes": {"vector_correct": len(p1_correct), "vector_hallucinated": len(p1_hallucinated)},
        "probe_ridge_alpha": probe_alpha,
        "vector_norms": {name: float(np.linalg.norm(v)) for name, v in directions.items()},
        "directions_present": names,
        "steering_vector_path": args.steering_vector_path,
    }
    with open(out_dir / "vector_analysis_meta.json", "w") as f:
        json.dump(meta_out, f, indent=2)
    print(f"[steering_vector_analysis] wrote {out_dir / 'vector_analysis_meta.json'}")

    print("\n[steering_vector_analysis] === pairwise cosine similarities ===")
    print(cos_df.to_string(index=False))
    headline = cos_df[cos_df["is_headline"]]
    if not headline.empty:
        print(f"\n[steering_vector_analysis] HEADLINE — steering vector vs probe direction: "
              f"cosine = {headline.iloc[0]['cosine']}")


if __name__ == "__main__":
    main()
