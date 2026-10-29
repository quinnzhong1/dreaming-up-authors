"""prevalence_decoding: the Appendix A7.1-A7.5 prevalence-decoding battery.

Produces the validated peak strategy, the per-layer curve, metadata-adjusted
and within-field decoding, shuffled-label negative controls, the citation
bucket probe, and the internal-prevalence deciles. Tables A37-A41, Figures 7, A11 and A12.

Usage
─────
  python 08_hidden_state_analysis/prevalence_decoding.py \
    --runs qwen3_32b=data/probing/runs/qwen3_32b mistral_small_3_2_24b=data/probing/runs/mistral_small_3_2_24b \
    --optimal_layers qwen3_32b:48 mistral_small_3_2_24b:22 \
    --out_dir results/hidden_state_analysis/probing
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


RANDOM_SEED = 42
N_SVD_COMPONENTS = 100
STRATEGIES = ["p1", "p2", "p3", "p4", "p5"]
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
MODEL_LABELS = {
    "qwen3_32b": "Qwen3-32B",
    "mistral_small_3_2_24b": "Mistral-Small-3.2-24B",
}
MODEL_COLORS = {
    "qwen3_32b": "#1B7837",
    "mistral_small_3_2_24b": "#B35806",
}
CITATION_BUCKET_BINS = [-0.5, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, float("inf")]
CITATION_BUCKET_LABELS = [
    "0-2",
    "3-4",
    "5-8",
    "9-16",
    "17-32",
    "33-64",
    "65-128",
    "129-256",
    "257-512",
    "513-1024",
    "1025-2048",
    "2049-4096",
    "4097+",
]

h5py = None
np = None
pd = None
plt = None


def ensure_dependencies() -> None:
    global h5py, np, pd, plt
    if h5py is not None:
        return
    import h5py as _h5py
    import matplotlib
    import numpy as _np
    import pandas as _pd

    matplotlib.use("Agg")
    import matplotlib.pyplot as _plt

    h5py = _h5py
    np = _np
    pd = _pd
    plt = _plt


@dataclass(frozen=True)
class RunSpec:
    model: str
    run_dir: Path
    optimal_layer: int


@dataclass
class AnalysisBundle:
    citation_probe: pd.DataFrame
    adjusted_decoding: pd.DataFrame
    within_field_decoding: pd.DataFrame
    deciles: pd.DataFrame
    negative_controls: pd.DataFrame
    predictions: pd.DataFrame


def model_label(model: str) -> str:
    return MODEL_LABELS.get(model, model.replace("_", " "))


def parse_runs(items: list[str]) -> dict[str, Path]:
    runs: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--runs entry must be model=/path, got {item!r}")
        name, path = item.split("=", 1)
        if not name.strip() or not path.strip():
            raise ValueError(f"Invalid --runs entry: {item!r}")
        runs[name.strip()] = Path(path.strip()).expanduser().resolve()
    return runs


def parse_layers(items: list[str]) -> dict[str, int]:
    layers: dict[str, int] = {}
    for item in items:
        if ":" not in item:
            raise ValueError(f"--optimal_layers entry must be model:layer, got {item!r}")
        name, layer = item.split(":", 1)
        layers[name.strip()] = int(layer)
    return layers


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def decode_array(values) -> list[str]:
    return [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in values]


def write_df(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[prevalence_decoding] wrote {path}")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"[prevalence_decoding] wrote {path}")


def find_scored_path(run_dir: Path) -> Path:
    candidates = [
        run_dir / "scored.tsv",
        run_dir.parent / "scored.tsv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing scored.tsv for run directory {run_dir}")


def find_h5_path(run_dir: Path) -> Path:
    candidates = [
        run_dir / "embeddings.h5",
        run_dir.parent / "embeddings.h5",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing embeddings.h5 for run directory {run_dir}")


def load_layer_indices(h5: h5py.File) -> list[int]:
    raw = h5.attrs["layer_indices"]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if isinstance(raw, str):
        return [int(x) for x in json.loads(raw)]
    return [int(x) for x in raw]


def layer_position(layers: list[int], layer_idx: int) -> int:
    if layer_idx not in layers:
        raise ValueError(f"Layer {layer_idx} is not in HDF5 layer_indices={layers}")
    return layers.index(layer_idx)


def align_rows(h5: h5py.File, scored: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    scored = scored.copy()
    scored["paper_id"] = scored["paper_id"].astype(str)
    by_pid = scored.set_index("paper_id", drop=False)
    valid_idx: list[int] = []
    rows: list[pd.Series] = []
    for i, pid in enumerate(decode_array(h5["paper_ids"][:])):
        if pid in by_pid.index:
            valid_idx.append(i)
            rows.append(by_pid.loc[pid])
    if not valid_idx:
        raise ValueError("No overlapping paper IDs between embeddings.h5 and scored.tsv")
    return np.array(valid_idx, dtype=np.int64), pd.DataFrame(rows).reset_index(drop=True)


def r2_from_predictions(y: np.ndarray, pred: np.ndarray) -> float:
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def spearman(y: np.ndarray, pred: np.ndarray) -> float:
    from scipy.stats import spearmanr

    return float(spearmanr(y, pred).statistic)


def add_citation_buckets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["citation_bucket"] = pd.cut(
        pd.to_numeric(out["citation_count"], errors="coerce"),
        bins=CITATION_BUCKET_BINS,
        labels=CITATION_BUCKET_LABELS,
        include_lowest=True,
        right=True,
    )
    return out


def ridge_oof(X: np.ndarray, y: np.ndarray, cv_splits: int = 5) -> tuple[np.ndarray, float]:
    from sklearn.linear_model import Ridge, RidgeCV
    from sklearn.model_selection import KFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    n_splits = min(cv_splits, len(y))
    if n_splits < 2:
        raise ValueError("Need at least two rows for cross-validation")
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    selector = make_pipeline(StandardScaler(), RidgeCV(alphas=RIDGE_ALPHAS, cv=cv))
    selector.fit(X, y)
    alpha = float(selector.named_steps["ridgecv"].alpha_)
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    pred = cross_val_predict(model, X, y, cv=cv)
    return pred, alpha


def fold_safe_pca_ridge_oof(
    X_raw: np.ndarray,
    y: np.ndarray,
    X_meta: np.ndarray | None = None,
    cv_splits: int = 5,
) -> tuple[np.ndarray, float]:
    """
    Leakage-safe prevalence decoder. PCA, scaling, and RidgeCV are fit only on
    each training fold before predicting the held-out fold.
    """
    from sklearn.decomposition import PCA
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler

    n_splits = min(cv_splits, len(y))
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    pred = np.empty(len(y), dtype=np.float64)
    alphas: list[float] = []

    for train_idx, test_idx in cv.split(X_raw):
        k = min(N_SVD_COMPONENTS, len(train_idx) - 1, X_raw.shape[1])
        pca = PCA(n_components=k, svd_solver="randomized", random_state=RANDOM_SEED)
        Z_train = pca.fit_transform(X_raw[train_idx])
        Z_test = pca.transform(X_raw[test_idx])

        z_scaler = StandardScaler()
        Z_train = z_scaler.fit_transform(Z_train)
        Z_test = z_scaler.transform(Z_test)

        if X_meta is not None:
            m_scaler = StandardScaler()
            M_train = m_scaler.fit_transform(X_meta[train_idx])
            M_test = m_scaler.transform(X_meta[test_idx])
            Z_train = np.hstack([M_train, Z_train])
            Z_test = np.hstack([M_test, Z_test])

        inner_cv = min(5, len(train_idx))
        model = RidgeCV(alphas=RIDGE_ALPHAS, cv=inner_cv)
        model.fit(Z_train, y[train_idx])
        alphas.append(float(model.alpha_))
        pred[test_idx] = model.predict(Z_test)

    return pred, float(np.median(alphas))


def metadata_matrix(scored: pd.DataFrame, include_field: bool = True) -> np.ndarray:
    meta = pd.DataFrame(index=scored.index)
    meta["year"] = pd.to_numeric(scored.get("year"), errors="coerce")
    meta["author_count"] = pd.to_numeric(
        scored.get("n_gt_authors", scored.get("n_pred_authors")), errors="coerce"
    )
    title = scored.get("title", pd.Series([""] * len(scored), index=scored.index)).fillna("")
    meta["title_length"] = title.astype(str).str.len()
    meta["title_tokens"] = title.astype(str).str.split().str.len()
    parts = [meta]
    if include_field:
        field = scored.get("field", pd.Series(["unknown"] * len(scored), index=scored.index))
        field_dummies = pd.get_dummies(field.fillna("unknown").astype(str), prefix="field")
        parts.append(field_dummies)
    out = pd.concat(parts, axis=1).fillna(meta.median(numeric_only=True))
    out = out.fillna(0.0)
    return out.to_numpy(dtype=np.float32)


def load_p1_embeddings(spec: RunSpec) -> tuple[np.ndarray, pd.DataFrame, int]:
    scored = pd.read_csv(find_scored_path(spec.run_dir), sep="\t")
    h5_path = find_h5_path(spec.run_dir)
    with h5py.File(h5_path, "r") as h5:
        if "per_query/p1" not in h5:
            raise KeyError(f"{h5_path} does not contain per_query/p1")
        layers = load_layer_indices(h5)
        pos = layer_position(layers, spec.optimal_layer)
        valid_idx, aligned = align_rows(h5, scored)
        X = h5["per_query/p1"][:, pos, :][valid_idx].astype(np.float32)
        n_model_layers = int(h5.attrs.get("n_model_layers", max(layers) + 1))
    return X, aligned, n_model_layers


def run_validated_strategy_layer_sweep(spec: RunSpec) -> pd.DataFrame:
    scored = pd.read_csv(find_scored_path(spec.run_dir), sep="\t")
    h5_path = find_h5_path(spec.run_dir)
    rows: list[dict] = []
    with h5py.File(h5_path, "r") as h5:
        layers = load_layer_indices(h5)
        valid_idx, aligned = align_rows(h5, scored)
        n_model_layers = int(h5.attrs.get("n_model_layers", max(layers) + 1))
        y = np.log1p(
            pd.to_numeric(aligned["citation_count"], errors="coerce")
            .fillna(0)
            .to_numpy(dtype=np.float32)
        )

        for strategy in STRATEGIES:
            key = f"per_query/{strategy}"
            if key not in h5:
                print(f"[prevalence_decoding] {spec.model}: skipping missing {key}")
                continue
            print(f"[prevalence_decoding] {spec.model}: validated prevalence sweep for {strategy}")
            for pos, layer_idx in enumerate(layers):
                X = h5[key][:, pos, :][valid_idx].astype(np.float32)
                pred, alpha = fold_safe_pca_ridge_oof(X, y)
                rows.append({
                    "model": spec.model,
                    "strategy": strategy,
                    "layer_idx": int(layer_idx),
                    "layer_depth_pct": round(100.0 * int(layer_idx) / (n_model_layers - 1), 1),
                    "r2_cv": round(r2_from_predictions(y, pred), 6),
                    "spearman_rho_cv": round(spearman(y, pred), 6),
                    "best_alpha": alpha,
                    "n": len(y),
                    "protocol": "fold_safe_pca_scaler_ridge",
                })
    return pd.DataFrame(rows)


def strategy_peak_summary(layer_df: pd.DataFrame) -> pd.DataFrame:
    if layer_df.empty:
        return pd.DataFrame()
    df = layer_df.copy()
    df["r2_cv"] = pd.to_numeric(df["r2_cv"], errors="coerce")
    rows: list[pd.Series] = []
    for _, grp in df.dropna(subset=["r2_cv"]).groupby(["model", "strategy"], sort=True):
        rows.append(grp.loc[grp["r2_cv"].idxmax()])
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).reset_index(drop=True)
    return out.rename(columns={
        "layer_idx": "peak_layer_idx",
        "layer_depth_pct": "peak_layer_depth_pct",
        "r2_cv": "peak_r2_cv",
        "spearman_rho_cv": "peak_spearman_rho_cv",
    })


def classify_with_cv(X_raw: np.ndarray, y: np.ndarray) -> dict[str, float]:
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        roc_auc_score,
    )
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        raise ValueError("Classification target has only one class")
    n_splits = min(5, int(counts.min()))
    if n_splits < 2:
        raise ValueError("Classification target has fewer than two examples in one class")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    proba = np.empty(len(y), dtype=np.float64)
    for train_idx, test_idx in cv.split(X_raw, y):
        k = min(N_SVD_COMPONENTS, len(train_idx) - 1, X_raw.shape[1])
        pca = PCA(n_components=k, svd_solver="randomized", random_state=RANDOM_SEED)
        Z_train = pca.fit_transform(X_raw[train_idx])
        Z_test = pca.transform(X_raw[test_idx])
        scaler = StandardScaler()
        Z_train = scaler.fit_transform(Z_train)
        Z_test = scaler.transform(Z_test)
        clf = LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            solver="lbfgs",
            random_state=RANDOM_SEED,
        )
        clf.fit(Z_train, y[train_idx])
        proba[test_idx] = clf.predict_proba(Z_test)[:, 1]
    pred = (proba >= 0.5).astype(np.int32)
    return {
        "auroc": float(roc_auc_score(y, proba)),
        "auprc": float(average_precision_score(y, proba)),
        "accuracy_at_0_5": float(accuracy_score(y, pred)),
        "balanced_accuracy_at_0_5": float(balanced_accuracy_score(y, pred)),
    }


def multiclass_bucket_probe(X_raw: np.ndarray, y: np.ndarray) -> dict[str, float]:
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        raise ValueError("Bucket target has only one class")
    n_splits = min(5, int(counts.min()))
    if n_splits < 2:
        raise ValueError("Bucket target has fewer than two examples in one class")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    pred = np.empty(len(y), dtype=np.int32)
    for train_idx, test_idx in cv.split(X_raw, y):
        k = min(N_SVD_COMPONENTS, len(train_idx) - 1, X_raw.shape[1])
        pca = PCA(n_components=k, svd_solver="randomized", random_state=RANDOM_SEED)
        Z_train = pca.fit_transform(X_raw[train_idx])
        Z_test = pca.transform(X_raw[test_idx])
        scaler = StandardScaler()
        Z_train = scaler.fit_transform(Z_train)
        Z_test = scaler.transform(Z_test)
        clf = LogisticRegression(
            class_weight="balanced",
            max_iter=3000,
            solver="lbfgs",
            random_state=RANDOM_SEED,
        )
        clf.fit(Z_train, y[train_idx])
        pred[test_idx] = clf.predict(Z_test)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
    }


def run_citation_bucket_probe(
    spec: RunSpec,
    X_raw: np.ndarray,
    scored: pd.DataFrame,
) -> pd.DataFrame:
    citations = pd.to_numeric(scored["citation_count"], errors="coerce").fillna(0).to_numpy()
    rows: list[dict] = []

    definitions: list[tuple[str, np.ndarray, np.ndarray]] = []
    definitions.append(("zero_vs_cited", np.ones(len(citations), dtype=bool), (citations > 0).astype(int)))
    median = float(np.median(citations))
    definitions.append((f"above_median_gt_{median:g}", np.ones(len(citations), dtype=bool), (citations > median).astype(int)))

    q25, q75 = np.quantile(citations, [0.25, 0.75])
    mask = (citations <= q25) | (citations >= q75)
    definitions.append((f"head_tail_q25_{q25:g}_q75_{q75:g}", mask, (citations[mask] >= q75).astype(int)))

    q90 = float(np.quantile(citations, 0.90))
    definitions.append((f"top_decile_ge_{q90:g}", np.ones(len(citations), dtype=bool), (citations >= q90).astype(int)))

    for label, mask, y in definitions:
        try:
            metrics = classify_with_cv(X_raw[mask], y)
            rows.append({
                "model": spec.model,
                "strategy": "p1",
                "layer_idx": spec.optimal_layer,
                "label_definition": label,
                "n": int(mask.sum()),
                "positive_rate": round(float(y.mean()), 6),
                **{k: round(v, 6) for k, v in metrics.items()},
            })
        except ValueError as exc:
            rows.append({
                "model": spec.model,
                "strategy": "p1",
                "layer_idx": spec.optimal_layer,
                "label_definition": label,
                "n": int(mask.sum()),
                "positive_rate": "",
                "auroc": "",
                "auprc": "",
                "accuracy_at_0_5": "",
                "balanced_accuracy_at_0_5": "",
                "skip_reason": str(exc),
            })

    try:
        bucket_y = pd.qcut(citations, q=4, labels=False, duplicates="drop")
        bucket_y_arr = np.asarray(bucket_y)
        keep = ~pd.isna(bucket_y_arr)
        metrics = multiclass_bucket_probe(X_raw[keep], bucket_y_arr[keep].astype(int))
        rows.append({
            "model": spec.model,
            "strategy": "p1",
            "layer_idx": spec.optimal_layer,
            "label_definition": "citation_quartile_multiclass",
            "n": int(keep.sum()),
            "positive_rate": "",
            "auroc": "",
            "auprc": "",
            "accuracy_at_0_5": round(metrics["accuracy"], 6),
            "balanced_accuracy_at_0_5": round(metrics["balanced_accuracy"], 6),
        })
    except ValueError as exc:
        rows.append({
            "model": spec.model,
            "strategy": "p1",
            "layer_idx": spec.optimal_layer,
            "label_definition": "citation_quartile_multiclass",
            "n": len(citations),
            "skip_reason": str(exc),
        })

    return pd.DataFrame(rows)


def run_adjusted_decoding(
    spec: RunSpec,
    X_raw: np.ndarray,
    scored: pd.DataFrame,
    n_model_layers: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = np.log1p(pd.to_numeric(scored["citation_count"], errors="coerce").fillna(0).to_numpy(dtype=np.float32))
    X_meta = metadata_matrix(scored)

    rows: list[dict] = []
    predictions: dict[str, np.ndarray] = {}

    pred, alpha = ridge_oof(X_meta, y)
    predictions["metadata_controls"] = pred
    rows.append({
        "model": spec.model,
        "strategy": "p1",
        "layer_idx": spec.optimal_layer,
        "layer_depth_pct": round(100.0 * spec.optimal_layer / (n_model_layers - 1), 1),
        "feature_set": "metadata_controls",
        "n": len(y),
        "n_features": X_meta.shape[1],
        "r2_cv": round(r2_from_predictions(y, pred), 6),
        "spearman_rho_cv": round(spearman(y, pred), 6),
        "best_alpha": alpha,
        "protocol": "metadata_scaled_ridge",
    })

    for feature_set, meta in [
        ("p1_embedding_fold_safe_pca", None),
        ("metadata_plus_p1_embedding_fold_safe_pca", X_meta),
    ]:
        pred, alpha = fold_safe_pca_ridge_oof(X_raw, y, X_meta=meta)
        predictions[feature_set] = pred
        rows.append({
            "model": spec.model,
            "strategy": "p1",
            "layer_idx": spec.optimal_layer,
            "layer_depth_pct": round(100.0 * spec.optimal_layer / (n_model_layers - 1), 1),
            "feature_set": feature_set,
            "n": len(y),
            "n_features": (N_SVD_COMPONENTS + (X_meta.shape[1] if meta is not None else 0)),
            "r2_cv": round(r2_from_predictions(y, pred), 6),
            "spearman_rho_cv": round(spearman(y, pred), 6),
            "best_alpha": alpha,
            "protocol": "fold_safe_pca_scaler_ridge",
        })

    meta_r2 = float([r for r in rows if r["feature_set"] == "metadata_controls"][0]["r2_cv"])
    emb_r2 = float([r for r in rows if r["feature_set"] == "p1_embedding_fold_safe_pca"][0]["r2_cv"])
    comb_r2 = float([r for r in rows if r["feature_set"] == "metadata_plus_p1_embedding_fold_safe_pca"][0]["r2_cv"])
    rows.append({
        "model": spec.model,
        "strategy": "p1",
        "layer_idx": spec.optimal_layer,
        "layer_depth_pct": round(100.0 * spec.optimal_layer / (n_model_layers - 1), 1),
        "feature_set": "incremental_embedding_over_metadata",
        "n": len(y),
        "n_features": N_SVD_COMPONENTS + X_meta.shape[1],
        "r2_cv": round(comb_r2 - meta_r2, 6),
        "spearman_rho_cv": "",
        "best_alpha": "",
        "metadata_r2_cv": meta_r2,
        "embedding_only_r2_cv": emb_r2,
        "combined_r2_cv": comb_r2,
        "protocol": "fold_safe_pca_scaler_ridge",
    })

    pred_df = pd.DataFrame({
        "model": spec.model,
        "paper_id": scored["paper_id"].astype(str),
        "citation_count": pd.to_numeric(scored["citation_count"], errors="coerce").fillna(0),
        "log_citation": y,
        "hr2": pd.to_numeric(scored["hr2"], errors="coerce"),
        "p1_prevalence_pred_oof": predictions["p1_embedding_fold_safe_pca"],
        "metadata_prevalence_pred_oof": predictions["metadata_controls"],
        "combined_prevalence_pred_oof": predictions["metadata_plus_p1_embedding_fold_safe_pca"],
    })
    return pd.DataFrame(rows), pred_df


def run_within_field_decoding(
    spec: RunSpec,
    X_raw: np.ndarray,
    scored: pd.DataFrame,
    n_model_layers: int,
    min_n: int = 50,
) -> pd.DataFrame:
    """
    Test whether P1 decodes prevalence after holding field fixed.

    This addresses the topic/field confound: paper titles carry field signals,
    and field can correlate with citation count and hallucination rate. Each row
    fits and evaluates a decoder only within one field.
    """
    if "field" not in scored.columns:
        return pd.DataFrame([{
            "model": spec.model,
            "strategy": "p1",
            "layer_idx": spec.optimal_layer,
            "layer_depth_pct": round(100.0 * spec.optimal_layer / (n_model_layers - 1), 1),
            "field": "",
            "feature_set": "",
            "n": len(scored),
            "skip_reason": "missing field column",
        }])

    rows: list[dict] = []
    fields = scored["field"].fillna("unknown").astype(str)
    for field, field_idx in fields.groupby(fields).groups.items():
        idx = np.asarray(list(field_idx), dtype=np.int64)
        field_scored = scored.iloc[idx].reset_index(drop=True)
        y = np.log1p(
            pd.to_numeric(field_scored["citation_count"], errors="coerce")
            .fillna(0)
            .to_numpy(dtype=np.float32)
        )
        base = {
            "model": spec.model,
            "strategy": "p1",
            "layer_idx": spec.optimal_layer,
            "layer_depth_pct": round(100.0 * spec.optimal_layer / (n_model_layers - 1), 1),
            "field": field,
            "n": len(y),
            "log_citation_std": round(float(np.std(y)), 6),
            "protocol": "within_field_fold_safe_pca_scaler_ridge",
        }
        if len(y) < min_n:
            rows.append({**base, "feature_set": "all", "skip_reason": f"n < {min_n}"})
            continue
        if float(np.std(y)) == 0.0:
            rows.append({**base, "feature_set": "all", "skip_reason": "constant citation target"})
            continue

        X_field = X_raw[idx]
        X_meta = metadata_matrix(field_scored, include_field=False)
        feature_specs = [
            ("metadata_controls_no_field", "metadata_scaled_ridge", lambda: ridge_oof(X_meta, y)),
            ("p1_embedding_fold_safe_pca", "within_field_fold_safe_pca_scaler_ridge", lambda: fold_safe_pca_ridge_oof(X_field, y)),
            (
                "metadata_plus_p1_embedding_fold_safe_pca",
                "within_field_fold_safe_pca_scaler_ridge",
                lambda: fold_safe_pca_ridge_oof(X_field, y, X_meta=X_meta),
            ),
        ]
        metric_rows: dict[str, float] = {}
        for feature_set, protocol, fn in feature_specs:
            try:
                pred, alpha = fn()
                r2 = r2_from_predictions(y, pred)
                metric_rows[feature_set] = r2
                rows.append({
                    **base,
                    "feature_set": feature_set,
                    "n_features": X_meta.shape[1] if feature_set == "metadata_controls_no_field" else (
                        N_SVD_COMPONENTS + (X_meta.shape[1] if feature_set.startswith("metadata_plus") else 0)
                    ),
                    "r2_cv": round(r2, 6),
                    "spearman_rho_cv": round(spearman(y, pred), 6),
                    "best_alpha": alpha,
                    "protocol": protocol,
                    "skip_reason": "",
                })
            except ValueError as exc:
                rows.append({
                    **base,
                    "feature_set": feature_set,
                    "n_features": "",
                    "r2_cv": "",
                    "spearman_rho_cv": "",
                    "best_alpha": "",
                    "protocol": protocol,
                    "skip_reason": str(exc),
                })

        if {
            "metadata_controls_no_field",
            "metadata_plus_p1_embedding_fold_safe_pca",
        }.issubset(metric_rows):
            rows.append({
                **base,
                "feature_set": "incremental_p1_over_within_field_metadata",
                "n_features": N_SVD_COMPONENTS + X_meta.shape[1],
                "r2_cv": round(
                    metric_rows["metadata_plus_p1_embedding_fold_safe_pca"]
                    - metric_rows["metadata_controls_no_field"],
                    6,
                ),
                "spearman_rho_cv": "",
                "best_alpha": "",
                "protocol": "within_field_fold_safe_pca_scaler_ridge",
                "skip_reason": "",
            })

    return pd.DataFrame(rows)


def run_internal_deciles(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for model, grp in predictions.groupby("model", sort=True):
        grp = grp.copy()
        grp["internal_prevalence_decile"] = pd.qcut(
            grp["p1_prevalence_pred_oof"],
            q=10,
            labels=False,
            duplicates="drop",
        )
        sub = grp.dropna(subset=["internal_prevalence_decile"])
        if sub.empty:
            continue
        out = sub.groupby("internal_prevalence_decile", as_index=False).agg(
            n=("paper_id", "size"),
            mean_predicted_log_citation=("p1_prevalence_pred_oof", "mean"),
            mean_log_citation=("log_citation", "mean"),
            mean_citation_count=("citation_count", "mean"),
            mean_hr2=("hr2", "mean"),
        )
        out["internal_prevalence_decile"] = out["internal_prevalence_decile"].astype(int) + 1
        out.insert(0, "model", model)
        rows.append(out)
    if not rows:
        return pd.DataFrame()
    df = pd.concat(rows, ignore_index=True)
    for col in ["mean_predicted_log_citation", "mean_log_citation", "mean_citation_count", "mean_hr2"]:
        df[col] = df[col].round(6)
    return df


def run_negative_controls(
    spec: RunSpec,
    X_raw: np.ndarray,
    scored: pd.DataFrame,
    real_r2: float,
    n_shuffles: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    y = np.log1p(pd.to_numeric(scored["citation_count"], errors="coerce").fillna(0).to_numpy(dtype=np.float32))
    rows: list[dict] = []
    for i in range(n_shuffles):
        y_perm = rng.permutation(y)
        pred, alpha = fold_safe_pca_ridge_oof(X_raw, y_perm)
        rows.append({
            "model": spec.model,
            "strategy": "p1",
            "layer_idx": spec.optimal_layer,
            "control": "shuffled_log_citation",
            "shuffle_idx": i + 1,
            "n": len(y),
            "r2_cv": round(r2_from_predictions(y_perm, pred), 6),
            "spearman_rho_cv": round(spearman(y_perm, pred), 6),
            "best_alpha": alpha,
            "real_embedding_r2_cv": round(real_r2, 6),
            "protocol": "fold_safe_pca_scaler_ridge",
        })
    return pd.DataFrame(rows)


def run_model_analyses(spec: RunSpec, n_shuffles: int) -> AnalysisBundle:
    X, scored, n_model_layers = load_p1_embeddings(spec)
    citation_probe = run_citation_bucket_probe(spec, X, scored)
    adjusted, predictions = run_adjusted_decoding(spec, X, scored, n_model_layers)
    within_field = run_within_field_decoding(spec, X, scored, n_model_layers)
    deciles = run_internal_deciles(predictions)
    real_row = adjusted[adjusted["feature_set"] == "p1_embedding_fold_safe_pca"].iloc[0]
    negative = run_negative_controls(spec, X, scored, float(real_row["r2_cv"]), n_shuffles)
    return AnalysisBundle(citation_probe, adjusted, within_field, deciles, negative, predictions)


def plot_prevalence_depth_trajectory(df: pd.DataFrame, fig_dir: Path) -> None:
    from matplotlib.lines import Line2D

    if df.empty:
        return
    df = df[df["strategy"].isin(["p1", "p4"])].copy()
    if df.empty:
        return
    df["layer_depth_pct"] = pd.to_numeric(df["layer_depth_pct"], errors="coerce")
    df["r2_cv"] = pd.to_numeric(df["r2_cv"], errors="coerce")
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    models = sorted(df["model"].unique())
    fallback_colors = ["#1B7837", "#B35806", "#4D4D4D", "#542788"]
    color_by_model = {
        model: MODEL_COLORS.get(model, fallback_colors[i % len(fallback_colors)])
        for i, model in enumerate(models)
    }
    dash_by_strategy = {"p1": "-", "p4": (0, (6, 3))}
    for (model, strategy), grp in df.groupby(["model", "strategy"]):
        grp = grp.sort_values("layer_depth_pct")
        ax.plot(
            grp["layer_depth_pct"],
            grp["r2_cv"],
            marker="o",
            lw=2,
            ls=dash_by_strategy.get(strategy, "-"),
            color=color_by_model[model],
            label=f"{model_label(model)} {strategy.upper()}",
        )
    ax.set_xlabel("Normalized model depth (%)")
    ax.set_ylabel("Prevalence decoding CV R2")
    ax.set_title("Prevalence signal across depth")
    ax.grid(alpha=0.25)
    model_handles = [
        Line2D([0], [0], color=color_by_model[model], lw=2, marker="o", label=model_label(model))
        for model in models
    ]
    strategy_handles = [
        Line2D([0], [0], color="#333333", lw=2.4, ls="-", label="P1: prompt-end state"),
        Line2D([0], [0], color="#333333", lw=2.4, ls=(0, (3, 1.5, 3, 1.5, 3)), label="P4: prompt-mean state"),
    ]
    handles = model_handles + strategy_handles
    ax.legend(handles=handles, frameon=False, fontsize=8, handlelength=3.2, ncol=2)
    fig.tight_layout()
    out = fig_dir / "figure_prevalence_depth_trajectory.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"[prevalence_decoding] wrote {out}")


def plot_strategy_comparison(df: pd.DataFrame, fig_dir: Path) -> None:
    if df.empty:
        return
    df["peak_r2_cv"] = pd.to_numeric(df["peak_r2_cv"], errors="coerce")
    pivot = df.pivot(index="model", columns="strategy", values="peak_r2_cv")
    strategies = [s for s in ["p1", "p2", "p3", "p4", "p5"] if s in pivot.columns]
    pivot = pivot.reindex(sorted(pivot.index))[strategies]
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    x = np.arange(len(strategies))
    marker_by_model = {
        "mistral_small_3_2_24b": "s",
        "qwen3_32b": "o",
    }
    label_offsets = {
        "mistral_small_3_2_24b": {
            "p1": (0, -12),
            "p2": (-4, -12),
            "p3": (0, 15),
            "p4": (4, -12),
            "p5": (0, 10),
        },
        "qwen3_32b": {
            "p1": (0, 8),
            "p2": (0, 8),
            "p3": (0, -10),
            "p4": (0, 8),
            "p5": (0, -10),
        },
    }
    fallback_colors = ["#1B7837", "#B35806", "#4D4D4D", "#542788"]
    for i, model in enumerate(pivot.index):
        y = pivot.loc[model].to_numpy(dtype=float)
        color = MODEL_COLORS.get(model, fallback_colors[i % len(fallback_colors)])
        ax.plot(
            x,
            y,
            lw=2.4,
            marker=marker_by_model.get(model, "o"),
            markersize=6.5,
            color=color,
            label=model_label(model),
        )
        for xi, yi, strategy in zip(x, y, strategies):
            if not np.isfinite(yi):
                continue
            dx, dy = label_offsets.get(model, {}).get(strategy, (0, 10))
            va = "bottom" if dy >= 0 else "top"
            ax.annotate(
                f"{yi:.3f}",
                xy=(xi, yi),
                xytext=(dx, dy),
                textcoords="offset points",
                ha="center",
                va=va,
                fontsize=8,
                color=color,
            )

    ax.axhline(0, color="#555555", lw=1, ls="--", alpha=0.75)
    ax.set_xticks(x)
    ax.set_xticklabels([s.upper() for s in strategies])
    ax.set_ylabel("Peak prevalence decoding CV R2")
    ax.set_title("Peak prevalence decoding by prompt strategy")
    ymin = min(-0.18, float(np.nanmin(pivot.to_numpy())) - 0.05)
    ymax = max(0.70, float(np.nanmax(pivot.to_numpy())) + 0.07)
    ax.set_ylim(ymin, ymax)
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    out = fig_dir / "figure_strategy_comparison.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"[prevalence_decoding] wrote {out}")


def plot_bucket_probe(df: pd.DataFrame, fig_dir: Path) -> None:
    if df.empty or "auroc" not in df.columns:
        return
    sub = df[df["label_definition"].astype(str).str.startswith("head_tail")].copy()
    if sub.empty:
        sub = df[pd.to_numeric(df["auroc"], errors="coerce").notna()].copy()
    if sub.empty:
        return
    sub["auroc"] = pd.to_numeric(sub["auroc"], errors="coerce")
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    colors = [MODEL_COLORS.get(m, "#4D4D4D") for m in sub["model"]]
    ax.bar([model_label(m) for m in sub["model"]], sub["auroc"], color=colors)
    ax.axhline(0.5, color="#555555", lw=1, ls="--")
    ax.set_ylim(0.45, max(1.0, float(sub["auroc"].max()) + 0.05))
    ax.set_ylabel("AUROC")
    ax.set_title("P1 head-tail citation probe")
    fig.tight_layout()
    out = fig_dir / "figure_prevalence_bucket_probe.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"[prevalence_decoding] wrote {out}")


def plot_internal_deciles(df: pd.DataFrame, fig_dir: Path) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for model, grp in df.groupby("model"):
        grp = grp.sort_values("internal_prevalence_decile")
        ax.plot(
            grp["internal_prevalence_decile"],
            grp["mean_hr2"],
            marker="o",
            lw=2,
            label=model_label(model),
        )
    ax.set_xlabel("P1-decoded prevalence-score group")
    ax.set_ylabel("Mean HR2")
    ax.set_title("Hallucination rate by internal prevalence state")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    out = fig_dir / "figure_internal_prevalence_deciles.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"[prevalence_decoding] wrote {out}")


def plot_negative_controls(df: pd.DataFrame, fig_dir: Path) -> None:
    if df.empty:
        return
    df = df.copy()
    df["r2_cv"] = pd.to_numeric(df["r2_cv"], errors="coerce")
    df["real_embedding_r2_cv"] = pd.to_numeric(df["real_embedding_r2_cv"], errors="coerce")
    models = sorted(df["model"].unique())
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    data = [df[df["model"] == m]["r2_cv"].dropna().to_numpy() for m in models]
    ax.boxplot(data, labels=[model_label(m) for m in models], showfliers=True)
    for i, model in enumerate(models, start=1):
        real = float(df[df["model"] == model]["real_embedding_r2_cv"].iloc[0])
        ax.scatter(i, real, color="#d62728", zorder=3, label="real labels" if i == 1 else None)
    ax.axhline(0, color="#555555", lw=1, ls="--")
    ax.set_ylabel("CV R2")
    ax.set_title("Shuffled citation-label negative controls")
    ax.legend(frameon=False)
    fig.tight_layout()
    out = fig_dir / "figure_negative_controls.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"[prevalence_decoding] wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="prevalence_decoding: prevalence appendix tables and figures")
    parser.add_argument("--runs", nargs="+", required=True, help="Entries like model_name=/path/to/run_dir")
    parser.add_argument("--optimal_layers", nargs="+", required=True, help="Entries like model_name:48")
    parser.add_argument("--out_dir", default="results/hidden_state_analysis/probing")
    parser.add_argument("--negative_control_shuffles", type=int, default=10)
    args = parser.parse_args()

    ensure_dependencies()

    runs = parse_runs(args.runs)
    layers = parse_layers(args.optimal_layers)
    missing = sorted(set(runs) - set(layers))
    if missing:
        raise ValueError(f"Missing --optimal_layers entries for: {missing}")
    specs = [RunSpec(name, path, layers[name]) for name, path in runs.items()]
    out_dir = Path(args.out_dir).expanduser().resolve()
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    citation_probe_parts: list[pd.DataFrame] = []
    adjusted_parts: list[pd.DataFrame] = []
    within_field_parts: list[pd.DataFrame] = []
    decile_parts: list[pd.DataFrame] = []
    negative_parts: list[pd.DataFrame] = []
    prediction_parts: list[pd.DataFrame] = []
    validated_layer_parts: list[pd.DataFrame] = []

    for spec in specs:
        print(f"[prevalence_decoding] {spec.model}: running P1 prevalence appendix analyses")
        bundle = run_model_analyses(spec, args.negative_control_shuffles)
        citation_probe_parts.append(bundle.citation_probe)
        adjusted_parts.append(bundle.adjusted_decoding)
        within_field_parts.append(bundle.within_field_decoding)
        decile_parts.append(bundle.deciles)
        negative_parts.append(bundle.negative_controls)
        prediction_parts.append(bundle.predictions)
        print(f"[prevalence_decoding] {spec.model}: running validated strategy/layer prevalence sweep")
        validated_layer_parts.append(run_validated_strategy_layer_sweep(spec))

    citation_probe = pd.concat(citation_probe_parts, ignore_index=True) if citation_probe_parts else pd.DataFrame()
    adjusted = pd.concat(adjusted_parts, ignore_index=True) if adjusted_parts else pd.DataFrame()
    within_field = pd.concat(within_field_parts, ignore_index=True) if within_field_parts else pd.DataFrame()
    deciles = pd.concat(decile_parts, ignore_index=True) if decile_parts else pd.DataFrame()
    negative = pd.concat(negative_parts, ignore_index=True) if negative_parts else pd.DataFrame()
    predictions = pd.concat(prediction_parts, ignore_index=True) if prediction_parts else pd.DataFrame()
    validated_layers = pd.concat(validated_layer_parts, ignore_index=True) if validated_layer_parts else pd.DataFrame()
    validated_peaks = strategy_peak_summary(validated_layers)

    write_df(out_dir / "citation_bucket_probe.csv", citation_probe)
    write_df(out_dir / "adjusted_prevalence_decoding.csv", adjusted)
    write_df(out_dir / "within_field_prevalence_decoding.csv", within_field)
    write_df(out_dir / "p1_prevalence_predictions.csv", predictions)
    write_df(out_dir / "internal_prevalence_deciles.csv", deciles)
    write_df(out_dir / "negative_controls.csv", negative)
    write_df(out_dir / "validated_strategy_layer_prevalence.csv", validated_layers)
    write_df(out_dir / "validated_strategy_peak_r2.csv", validated_peaks)

    plot_prevalence_depth_trajectory(validated_layers, fig_dir)
    plot_strategy_comparison(validated_peaks, fig_dir)
    plot_bucket_probe(citation_probe, fig_dir)
    plot_internal_deciles(deciles, fig_dir)
    plot_negative_controls(negative, fig_dir)

    print(f"[prevalence_decoding] Wrote prevalence appendix outputs to {out_dir}")


if __name__ == "__main__":
    main()
