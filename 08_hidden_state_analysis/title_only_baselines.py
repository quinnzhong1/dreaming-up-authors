"""title_only_baselines: title-only lexical baselines, and the P1 residual (R2 = 0.13).

Answers whether P1 decoding just picks up surface features of the title.
Predicts log(citations+1) from the title alone via fold-safe TF-IDF+SVD ridge
and frozen SBERT, then asks the stronger question: after removing everything
the title text explains, does P1 still carry signal? If it does, that residual
cannot be lexical.

Usage
─────
  # Title-only baselines alone (no embeddings.h5 required):
  python 08_hidden_state_analysis/title_only_baselines.py \
    --runs mistral_small_3_2_24b=data/probing/runs/mistral_small_3_2_24b \
    --out_dir results/hidden_state_analysis/probing

  # Full comparison against P1 hidden states:
  python 08_hidden_state_analysis/title_only_baselines.py \
    --runs qwen3_32b=data/probing/runs/qwen3_32b mistral_small_3_2_24b=data/probing/runs/mistral_small_3_2_24b \
    --optimal_layers qwen3_32b:48 mistral_small_3_2_24b:22 \
    --out_dir results/hidden_state_analysis/probing
"""

from __future__ import annotations

import argparse
from pathlib import Path

import prevalence_decoding as s8

RANDOM_SEED = s8.RANDOM_SEED
RIDGE_ALPHAS = s8.RIDGE_ALPHAS
TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_SVD_COMPONENTS = 200
DEFAULT_SBERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

np = None
pd = None


def ensure_dependencies() -> None:
    global np, pd
    if np is not None:
        return
    import numpy as _np
    import pandas as _pd

    s8.ensure_dependencies()
    np = _np
    pd = _pd


def load_titles_and_target(run_dir: Path) -> "pd.DataFrame":
    """Load scored.tsv and return rows with non-empty title + citation_count."""
    scored = pd.read_csv(s8.find_scored_path(run_dir), sep="\t")
    scored = scored.copy()
    scored["title"] = scored.get("title", "").fillna("").astype(str)
    scored = scored[scored["title"].str.len() > 0].reset_index(drop=True)
    scored["log_citation"] = np.log1p(
        pd.to_numeric(scored["citation_count"], errors="coerce").fillna(0)
    )
    return scored


def fold_safe_tfidf_ridge_oof(
    titles: "np.ndarray",
    y: "np.ndarray",
    cv_splits: int = 5,
) -> tuple:
    """TF-IDF -> TruncatedSVD -> Ridge, refit per fold (no leakage)."""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import Ridge, RidgeCV
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler

    n_splits = min(cv_splits, len(y))
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    pred = np.empty(len(y), dtype=np.float64)
    alphas: list[float] = []
    n_features_used: list[int] = []

    for train_idx, test_idx in cv.split(titles):
        vec = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            ngram_range=TFIDF_NGRAM_RANGE,
            lowercase=True,
            stop_words="english",
            sublinear_tf=True,
        )
        X_train = vec.fit_transform(titles[train_idx])
        X_test = vec.transform(titles[test_idx])

        k = min(TFIDF_SVD_COMPONENTS, len(train_idx) - 1, X_train.shape[1] - 1)
        k = max(k, 1)
        svd = TruncatedSVD(n_components=k, random_state=RANDOM_SEED)
        Z_train = svd.fit_transform(X_train)
        Z_test = svd.transform(X_test)

        scaler = StandardScaler()
        Z_train = scaler.fit_transform(Z_train)
        Z_test = scaler.transform(Z_test)

        inner_cv = min(5, len(train_idx))
        model = RidgeCV(alphas=RIDGE_ALPHAS, cv=inner_cv)
        model.fit(Z_train, y[train_idx])
        alphas.append(float(model.alpha_))
        n_features_used.append(k)
        pred[test_idx] = model.predict(Z_test)

    return pred, float(np.median(alphas)), int(np.median(n_features_used))


def encode_sbert(titles: "np.ndarray", model_name: str) -> "np.ndarray":
    from sentence_transformers import SentenceTransformer

    print(f"[title_only_baselines] loading Sentence-BERT encoder: {model_name}")
    encoder = SentenceTransformer(model_name)
    embeddings = encoder.encode(
        list(titles), batch_size=64, show_progress_bar=True, convert_to_numpy=True
    )
    return embeddings.astype(np.float32)


def fold_safe_ridge_oof_precomputed(
    X: "np.ndarray",
    y: "np.ndarray",
    cv_splits: int = 5,
) -> tuple:
    """Ridge over precomputed (already-frozen) features. StandardScaler is
    still refit per fold since it depends on the training fold's stats."""
    from sklearn.linear_model import Ridge, RidgeCV
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler

    n_splits = min(cv_splits, len(y))
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    pred = np.empty(len(y), dtype=np.float64)
    alphas: list[float] = []

    for train_idx, test_idx in cv.split(X):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        inner_cv = min(5, len(train_idx))
        model = RidgeCV(alphas=RIDGE_ALPHAS, cv=inner_cv)
        model.fit(X_train, y[train_idx])
        alphas.append(float(model.alpha_))
        pred[test_idx] = model.predict(X_test)

    return pred, float(np.median(alphas))


def fold_safe_combined_ridge_oof(
    titles: "np.ndarray",
    sbert_X: "np.ndarray",
    y: "np.ndarray",
    cv_splits: int = 5,
) -> tuple:
    """TF-IDF-SVD (refit per fold) concatenated with precomputed SBERT
    features, then Ridge. This is the strongest title-only model and the
    one used to compute the OOF residual for the P1 comparison."""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import Ridge, RidgeCV
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler

    n_splits = min(cv_splits, len(y))
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    pred = np.empty(len(y), dtype=np.float64)
    alphas: list[float] = []

    for train_idx, test_idx in cv.split(titles):
        vec = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            ngram_range=TFIDF_NGRAM_RANGE,
            lowercase=True,
            stop_words="english",
            sublinear_tf=True,
        )
        Xt_train = vec.fit_transform(titles[train_idx])
        Xt_test = vec.transform(titles[test_idx])
        k = min(TFIDF_SVD_COMPONENTS, len(train_idx) - 1, Xt_train.shape[1] - 1)
        k = max(k, 1)
        svd = TruncatedSVD(n_components=k, random_state=RANDOM_SEED)
        Zt_train = svd.fit_transform(Xt_train)
        Zt_test = svd.transform(Xt_test)

        Zs_train, Zs_test = sbert_X[train_idx], sbert_X[test_idx]

        scaler = StandardScaler()
        combined_train = scaler.fit_transform(np.hstack([Zt_train, Zs_train]))
        combined_test = scaler.transform(np.hstack([Zt_test, Zs_test]))

        inner_cv = min(5, len(train_idx))
        model = RidgeCV(alphas=RIDGE_ALPHAS, cv=inner_cv)
        model.fit(combined_train, y[train_idx])
        alphas.append(float(model.alpha_))
        pred[test_idx] = model.predict(combined_test)

    return pred, float(np.median(alphas))


def run_for_model(spec_model: str, run_dir: Path, optimal_layer: int | None,
                   sbert_model_name: str, limit: int | None) -> tuple:
    """Returns (baseline_rows, comparison_row_or_None)."""
    scored = load_titles_and_target(run_dir)
    if limit:
        scored = scored.iloc[:limit].reset_index(drop=True)
    titles = scored["title"].to_numpy(dtype=object)
    y = scored["log_citation"].to_numpy(dtype=np.float64)
    n = len(y)
    print(f"[title_only_baselines] {spec_model}: {n} papers with non-empty titles")

    baseline_rows = []

    print(f"[title_only_baselines] {spec_model}: fold-safe TF-IDF ridge ...")
    tfidf_pred, tfidf_alpha, tfidf_nfeat = fold_safe_tfidf_ridge_oof(titles, y)
    baseline_rows.append({
        "model": spec_model, "feature_set": "tfidf_bow", "n": n,
        "n_features": tfidf_nfeat,
        "r2_cv": round(s8.r2_from_predictions(y, tfidf_pred), 6),
        "spearman_rho_cv": round(s8.spearman(y, tfidf_pred), 6),
        "best_alpha": tfidf_alpha,
        "protocol": "fold_safe_tfidf_svd_scaler_ridge",
    })

    print(f"[title_only_baselines] {spec_model}: Sentence-BERT ridge ...")
    sbert_X = encode_sbert(titles, sbert_model_name)
    sbert_pred, sbert_alpha = fold_safe_ridge_oof_precomputed(sbert_X, y)
    baseline_rows.append({
        "model": spec_model, "feature_set": "sbert_frozen", "n": n,
        "n_features": sbert_X.shape[1],
        "r2_cv": round(s8.r2_from_predictions(y, sbert_pred), 6),
        "spearman_rho_cv": round(s8.spearman(y, sbert_pred), 6),
        "best_alpha": sbert_alpha,
        "protocol": "frozen_sbert_scaler_ridge",
    })

    print(f"[title_only_baselines] {spec_model}: combined TF-IDF+SBERT ridge (title-only OOF) ...")
    combined_pred, combined_alpha = fold_safe_combined_ridge_oof(titles, sbert_X, y)
    baseline_rows.append({
        "model": spec_model, "feature_set": "tfidf_bow+sbert_frozen", "n": n,
        "n_features": tfidf_nfeat + sbert_X.shape[1],
        "r2_cv": round(s8.r2_from_predictions(y, combined_pred), 6),
        "spearman_rho_cv": round(s8.spearman(y, combined_pred), 6),
        "best_alpha": combined_alpha,
        "protocol": "fold_safe_tfidf_svd_plus_sbert_scaler_ridge",
    })

    comparison_row = None
    if optimal_layer is not None:
        try:
            spec = s8.RunSpec(model=spec_model, run_dir=run_dir, optimal_layer=optimal_layer)
            X_p1, aligned, _ = s8.load_p1_embeddings(spec)
        except (FileNotFoundError, KeyError) as e:
            print(f"[title_only_baselines] {spec_model}: no embeddings.h5 available, skipping P1 "
                  f"comparison ({e})")
        else:
            aligned = aligned.copy()
            aligned["title"] = aligned.get("title", "").fillna("").astype(str)
            has_title = aligned["title"].str.len() > 0
            if has_title.sum() < 10:
                print(f"[title_only_baselines] {spec_model}: too few titled rows in embeddings.h5 "
                      "alignment, skipping P1 comparison")
            else:
                X_p1 = X_p1[has_title.to_numpy()]
                aligned = aligned[has_title].reset_index(drop=True)
                y_p1 = np.log1p(
                    pd.to_numeric(aligned["citation_count"], errors="coerce").fillna(0)
                    .to_numpy(dtype=np.float64)
                )

                titles_p1 = aligned["title"].to_numpy(dtype=object)
                sbert_X_p1 = encode_sbert(titles_p1, sbert_model_name)
                title_oof_pred, _ = fold_safe_combined_ridge_oof(titles_p1, sbert_X_p1, y_p1)
                title_r2 = s8.r2_from_predictions(y_p1, title_oof_pred)
                residual = y_p1 - title_oof_pred

                p1_pred_raw, _ = s8.fold_safe_pca_ridge_oof(X_p1, y_p1)
                p1_r2_raw = s8.r2_from_predictions(y_p1, p1_pred_raw)

                p1_pred_resid, _ = s8.fold_safe_pca_ridge_oof(X_p1, residual)
                p1_r2_resid = s8.r2_from_predictions(residual, p1_pred_resid)

                comparison_row = {
                    "model": spec_model,
                    "layer_idx": optimal_layer,
                    "n": len(y_p1),
                    "title_only_r2_cv": round(title_r2, 6),
                    "p1_on_raw_log_citation_r2_cv": round(p1_r2_raw, 6),
                    "p1_on_title_residual_r2_cv": round(p1_r2_resid, 6),
                    "delta_p1_minus_title": round(p1_r2_raw - title_r2, 6),
                    "note": (
                        "p1_on_title_residual_r2_cv is P1's R^2 predicting what's left "
                        "of log-citation after the combined TF-IDF+SBERT title model's "
                        "OOF prediction is subtracted out; a value well above 0 means P1 "
                        "carries prevalence signal that title lexical features cannot explain."
                    ),
                }

    return baseline_rows, comparison_row


def main() -> None:
    parser = argparse.ArgumentParser(description="title_only_baselines: title-only lexical-confound baselines")
    parser.add_argument("--runs", nargs="+", required=True,
                        help="model=run_dir pairs, e.g. qwen3_32b=results")
    parser.add_argument("--optimal_layers", nargs="*", default=[],
                        help="model:layer pairs for the P1-vs-title comparison "
                             "(optional; omit to only compute title-only baselines)")
    parser.add_argument("--sbert_model", default=DEFAULT_SBERT_MODEL,
                        help="Sentence-BERT model name/path (default: %(default)s)")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N titled rows per run (smoke test)")
    args = parser.parse_args()

    ensure_dependencies()

    runs = s8.parse_runs(args.runs)
    layers = s8.parse_layers(args.optimal_layers) if args.optimal_layers else {}
    out_dir = Path(args.out_dir)

    all_baseline_rows: list[dict] = []
    all_comparison_rows: list[dict] = []

    for model, run_dir in runs.items():
        optimal_layer = layers.get(model)
        baseline_rows, comparison_row = run_for_model(
            model, run_dir, optimal_layer, args.sbert_model, args.limit
        )
        all_baseline_rows.extend(baseline_rows)
        if comparison_row is not None:
            all_comparison_rows.append(comparison_row)

    s8.write_df(out_dir / "title_only_baselines.csv", pd.DataFrame(all_baseline_rows))
    if all_comparison_rows:
        s8.write_df(
            out_dir / "title_vs_p1_comparison.csv", pd.DataFrame(all_comparison_rows)
        )
    else:
        print("[title_only_baselines] no --optimal_layers given (or no embeddings.h5 found) — "
              "title_vs_p1_comparison.csv not written")


if __name__ == "__main__":
    main()
