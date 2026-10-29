"""p9_entity_probe: imbalance-aware geometry for the P9 entity probe (Table A44).

The unit is a generated author name and the positive class is ~97% of them, so
AUROC alone flatters the probe. Adds AUPRC lift over the base rate, centroid
distance in units of pooled within-class scatter, and nearest-centroid
balanced accuracy.

Usage
-----
  python 08_hidden_state_analysis/p9_entity_probe.py \
    --runs qwen3_32b=data/probing/runs/qwen3_32b \
           mistral_small_3_2_24b=data/probing/runs/mistral_small_3_2_24b \
    --optimal_layers qwen3_32b:48 mistral_small_3_2_24b:20 \
    --out_dir results/hidden_state_analysis/probing
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path


RANDOM_SEED = 42
MODEL_LABELS = {
    "qwen3_32b": "Qwen3-32B",
    "mistral_small_3_2_24b": "Mistral-Small-3.2-24B",
}


@dataclass(frozen=True)
class RunSpec:
    model: str
    run_dir: Path
    optimal_layer: int


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


def load_tsv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def decode_array(values) -> list[str]:
    return [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in values]


def load_layer_indices(h5) -> list[int]:
    raw = h5.attrs.get("layer_indices")
    if raw is None:
        raise ValueError("embeddings.h5 missing layer_indices attr")
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return [int(x) for x in json.loads(raw)]


def layer_position(target_layers: list[int], layer: int) -> int:
    if layer not in target_layers:
        raise ValueError(f"Layer {layer} not in saved target layers: {target_layers}")
    return target_layers.index(layer)


def standardize(X):
    import numpy as np

    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    return (X - mu) / sd


def pooled_rms_effect_size(Xs, y, c0, c1) -> tuple[float, float, float]:
    import numpy as np

    X0 = Xs[y == 0]
    X1 = Xs[y == 1]
    centroid_l2 = float(np.linalg.norm(c1 - c0))
    rms0 = float(np.sqrt(np.mean(np.sum((X0 - c0) ** 2, axis=1))))
    rms1 = float(np.sqrt(np.mean(np.sum((X1 - c1) ** 2, axis=1))))
    pooled = math.sqrt((rms0 ** 2 + rms1 ** 2) / 2.0)
    effect = centroid_l2 / pooled if pooled > 0 else float("nan")
    return centroid_l2, pooled, effect


def nearest_centroid_balanced_accuracy(Xs, y, c0, c1) -> float:
    import numpy as np

    d0 = np.linalg.norm(Xs - c0, axis=1)
    d1 = np.linalg.norm(Xs - c1, axis=1)
    pred = (d1 < d0).astype(int)
    recalls = []
    for cls in (0, 1):
        mask = y == cls
        if mask.sum() == 0:
            continue
        recalls.append(float((pred[mask] == cls).mean()))
    return float(np.mean(recalls)) if recalls else float("nan")


def run_probe(X, y) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    import numpy as np

    min_class = int(min((y == 0).sum(), (y == 1).sum()))
    n_splits = min(5, min_class)
    if n_splits < 2:
        return {
            "probe_auroc": float("nan"),
            "probe_auprc": float("nan"),
            "probe_balanced_accuracy": float("nan"),
        }

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            solver="lbfgs",
            random_state=RANDOM_SEED,
        ),
    )
    proba = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "probe_auroc": float(roc_auc_score(y, proba)),
        "probe_auprc": float(average_precision_score(y, proba)),
        "probe_balanced_accuracy": float(balanced_accuracy_score(y, pred)),
    }


def analyze_run(spec: RunSpec) -> dict:
    import h5py
    import numpy as np

    h5_path = spec.run_dir / "embeddings.h5"
    labels_path = spec.run_dir / "per_name_labels.tsv"
    if not h5_path.exists():
        raise FileNotFoundError(h5_path)
    if not labels_path.exists():
        raise FileNotFoundError(labels_path)

    label_rows = load_tsv(labels_path)
    with h5py.File(h5_path, "r") as h5:
        if "per_name/hidden" not in h5 or h5["per_name/hidden"].shape[0] == 0:
            raise ValueError(f"{h5_path} has no per_name/hidden data")
        target_layers = load_layer_indices(h5)
        l_pos = layer_position(target_layers, spec.optimal_layer)
        pids = decode_array(h5["per_name/paper_id"][:])

        n = min(len(label_rows), len(pids))
        keep = [
            i
            for i in range(n)
            if label_rows[i].get("paper_id") == pids[i]
            and label_rows[i].get("is_hallucinated", "") != ""
        ]
        if not keep:
            raise ValueError(f"No aligned per-name labels for {spec.model}")

        keep_idx = np.array(keep, dtype=np.int64)
        y = np.array([int(label_rows[i]["is_hallucinated"]) for i in keep], dtype=np.int32)
        X = h5["per_name/hidden"][:, l_pos, :][keep_idx].astype(np.float32)
        n_model_layers = int(h5.attrs.get("n_model_layers", max(target_layers) + 1))

    Xs = standardize(X)
    X0 = Xs[y == 0]
    X1 = Xs[y == 1]
    c0 = X0.mean(axis=0)
    c1 = X1.mean(axis=0)
    centroid_l2, pooled_rms, centroid_effect = pooled_rms_effect_size(Xs, y, c0, c1)
    nc_bal_acc = nearest_centroid_balanced_accuracy(Xs, y, c0, c1)
    probe = run_probe(X, y)

    n_names = int(len(y))
    n_matched = int((y == 0).sum())
    n_hallucinated = int((y == 1).sum())
    positive_rate = n_hallucinated / n_names
    auprc = probe["probe_auprc"]
    auprc_lift = auprc / positive_rate if positive_rate > 0 else float("nan")

    return {
        "model": spec.model,
        "model_label": MODEL_LABELS.get(spec.model, spec.model),
        "layer_idx": spec.optimal_layer,
        "layer_depth_pct": round(100.0 * spec.optimal_layer / (n_model_layers - 1), 1),
        "n_names": n_names,
        "n_matched": n_matched,
        "n_hallucinated": n_hallucinated,
        "hallucinated_positive_rate": round(positive_rate, 6),
        "probe_auroc": round(probe["probe_auroc"], 6),
        "probe_auprc": round(probe["probe_auprc"], 6),
        "probe_auprc_lift_over_baseline": round(auprc_lift, 6),
        "probe_balanced_accuracy": round(probe["probe_balanced_accuracy"], 6),
        "centroid_l2_standardized": round(centroid_l2, 6),
        "pooled_within_class_rms": round(pooled_rms, 6),
        "centroid_l2_over_pooled_rms": round(centroid_effect, 6),
        "nearest_centroid_balanced_accuracy": round(nc_bal_acc, 6),
    }


def write_notes(path: Path, rows: list[dict]) -> None:
    lines = [
        "# P9 Entity-Level Robustness",
        "",
        "Positive class is `is_hallucinated = 1`. Because hallucinated generated names are the majority class, raw AUPRC should be interpreted together with the positive-class baseline rate and AUPRC lift over baseline.",
        "",
    ]
    for row in rows:
        lines.append(
            f"- {row['model_label']}: AUROC={row['probe_auroc']:.3f}, "
            f"AUPRC={row['probe_auprc']:.3f}, positive baseline={row['hallucinated_positive_rate']:.3f}, "
            f"AUPRC lift={row['probe_auprc_lift_over_baseline']:.3f}, "
            f"centroid L2 / pooled RMS={row['centroid_l2_over_pooled_rms']:.3f}."
        )
    lines += [
        "",
        "Recommended interpretation: P9 contains separable information about matched versus hallucinated generated names, but the class imbalance means AUROC and normalized effect sizes are more informative than raw AUPRC alone. Treat P9 as entity-level supporting evidence, not the main mechanism.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="p9_entity_probe: P9 entity-level robustness metrics")
    parser.add_argument("--runs", nargs="+", required=True, help="Run dirs as model=/path")
    parser.add_argument("--optimal_layers", nargs="+", required=True, help="Optimal layers as model:layer")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    args = parser.parse_args()

    run_dirs = parse_runs(args.runs)
    layers = parse_layers(args.optimal_layers)
    specs = [
        RunSpec(model=model, run_dir=run_dir, optimal_layer=layers[model])
        for model, run_dir in run_dirs.items()
        if model in layers
    ]
    if len(specs) != len(run_dirs):
        missing = sorted(set(run_dirs) - set(layers))
        raise ValueError(f"Missing --optimal_layers entries for: {missing}")

    rows = []
    for spec in specs:
        print(f"[p9_entity_probe] Analyzing {spec.model} from {spec.run_dir}")
        row = analyze_run(spec)
        rows.append(row)
        print(
            f"[p9_entity_probe] {row['model_label']}: AUROC={row['probe_auroc']:.3f}, "
            f"AUPRC={row['probe_auprc']:.3f}, lift={row['probe_auprc_lift_over_baseline']:.3f}, "
            f"effect={row['centroid_l2_over_pooled_rms']:.3f}"
        )

    out_dir = Path(args.out_dir).expanduser().resolve()
    fields = [
        "model",
        "model_label",
        "layer_idx",
        "layer_depth_pct",
        "n_names",
        "n_matched",
        "n_hallucinated",
        "hallucinated_positive_rate",
        "probe_auroc",
        "probe_auprc",
        "probe_auprc_lift_over_baseline",
        "probe_balanced_accuracy",
        "centroid_l2_standardized",
        "pooled_within_class_rms",
        "centroid_l2_over_pooled_rms",
        "nearest_centroid_balanced_accuracy",
    ]
    write_csv(out_dir / "p9_robustness_summary.csv", rows, fields)
    write_notes(out_dir / "p9_robustness_notes.md", rows)
    print(f"[p9_entity_probe] Wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
