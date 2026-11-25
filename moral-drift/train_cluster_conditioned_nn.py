"""Train ONE cluster-conditioned moral-choice network for all three clusters.

Put this file beside generate.py in moral-drift/ and run:
    python train_cluster_conditioned_nn.py

Required inputs (produced by the existing project):
    moral-drift/data/preferences.json
    moral-classification/data/dilemmas.jsonl

The network input is [9 dilemma features | 3 one-hot cluster indicators].
It outputs two raw logits: [stay, swerve]. Training uses *observed*
cluster-majority labels, not simulated future choices. Each dilemma is
held out as a unit to avoid leakage between its three cluster examples.

Outputs:
    models/cluster_conditioned_moral_nn.pt
    models/cluster_conditioned_moral_nn.onnx
    reports/cluster_conditioned_metrics.json
    reports/cluster_conditioned_test_predictions.csv
    reports/cluster_conditioned_training_curves.png
    reports/cluster_conditioned_confusion_<cluster>.png
    data/frozen_nn_predictions.json   (one fixed decision per dilemma/cluster)
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    ConfusionMatrixDisplay,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

CLUSTERS = ("western", "eastern", "southern")
CLASS_NAMES = ("stay", "swerve")
LABEL_TO_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}


def load_dataset(project_root: Path, seed: int):
    preference_path = project_root / "data" / "preferences.json"
    dilemma_path = project_root.parent / "moral-classification" / "data" / "dilemmas.jsonl"
    if not preference_path.exists() or not dilemma_path.exists():
        raise FileNotFoundError(
            "Missing generated data. Expected:\n"
            f"  {preference_path}\n  {dilemma_path}\n"
            "Run generate.py and check the sibling moral-classification directory."
        )

    with preference_path.open(encoding="utf-8") as fh:
        prefs = json.load(fh)
    with dilemma_path.open(encoding="utf-8") as fh:
        dilemmas = [json.loads(line) for line in fh if line.strip()]

    dilemma_ids = list(prefs["dilemma_ids"])
    phi = np.asarray(prefs["phi"], dtype=np.float32)
    if phi.shape != (len(dilemma_ids), 9):
        raise ValueError(f"Expected one 9-feature row per dilemma, got {phi.shape}")
    if len(set(dilemma_ids)) != len(dilemma_ids):
        raise ValueError("Duplicate dilemma IDs in preferences.json")
    if not np.allclose(phi[:, 0], 1.0):
        raise ValueError("Feature 0 must be the constant action-bias feature 1")
    if not np.isfinite(phi).all():
        raise ValueError("Feature matrix contains non-finite values")

    feature_by_id = dict(zip(dilemma_ids, phi))
    seen_ids = [d["id"] for d in dilemmas]
    if len(set(seen_ids)) != len(seen_ids):
        raise ValueError("Duplicate dilemma IDs in dilemmas.jsonl")
    if set(seen_ids) != set(dilemma_ids):
        raise ValueError("Dilemma IDs differ between preferences.json and dilemmas.jsonl")
    for row in dilemmas:
        if row["split"] not in {"train", "test"}:
            raise ValueError(f"Unexpected split {row['split']} for dilemma {row['id']}")

    # Choose validation IDs only from the pre-existing training split.
    # Splitting unique dilemma IDs, NOT rows of (dilemma, cluster), is crucial.
    original_train_ids = sorted(d["id"] for d in dilemmas if d["split"] == "train")
    if len(original_train_ids) < 5:
        raise ValueError("Not enough training dilemmas to make a validation set")
    rng = np.random.default_rng(seed)
    rng.shuffle(original_train_ids)
    n_val = max(1, int(round(0.2 * len(original_train_ids))))
    validation_ids = set(original_train_ids[:n_val])

    X, y, c_idx, splits, ids, observed_shares = [], [], [], [], [], []
    for row in dilemmas:
        split = ("validation" if row["id"] in validation_ids
                 else row["split"])
        feature = feature_by_id[row["id"]]
        for i, cluster in enumerate(CLUSTERS):
            choice = row[cluster].get("choice")
            if choice is None:  # tied votes: no observed majority label
                continue
            if choice not in LABEL_TO_INDEX:
                raise ValueError(f"Unknown label: {choice!r}")
            one_hot = np.eye(len(CLUSTERS), dtype=np.float32)[i]
            X.append(np.concatenate((feature, one_hot)))
            y.append(LABEL_TO_INDEX[choice])
            c_idx.append(i)
            splits.append(split)
            ids.append(row["id"])
            observed_shares.append(float(row[cluster]["swerve_share"]))

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    c_idx = np.asarray(c_idx, dtype=np.int64)
    splits = np.asarray(splits)
    ids = np.asarray(ids)
    observed_shares = np.asarray(observed_shares, dtype=np.float64)

    if X.shape[1] != 12:
        raise ValueError(f"Expected 12 inputs (9 dilemma + 3 cluster), got {X.shape[1]}")
    for split in ("train", "validation", "test"):
        for ci in range(3):
            if not np.any((splits == split) & (c_idx == ci)):
                raise ValueError(f"No {split} examples for {CLUSTERS[ci]}")
    return (X, y, c_idx, splits, ids, observed_shares, dilemma_ids, phi,
            prefs["dimensions"])


class ClusterConditionedNN(nn.Module):
    """One model; the three cluster indicators are part of its input."""
    def __init__(self, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(12, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),  # logits: [stay, swerve]
        )

    def forward(self, x):
        return self.net(x)


def forward_numpy(model, x):
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(x)).detach().cpu().numpy()
    # stable softmax for two outputs
    shift = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(shift)
    probabilities = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    return logits, probabilities, logits.argmax(axis=1)


def evaluate(model, x, y, loss_fn):
    logits, probabilities, predictions = forward_numpy(model, x)
    with torch.no_grad():
        loss = float(loss_fn(torch.from_numpy(logits), torch.from_numpy(y)))
    return loss, accuracy_score(y, predictions), predictions, probabilities


def plot_curves(history, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric in zip(axes, ("loss", "accuracy")):
        ax.plot(history[f"train_{metric}"], label="Train")
        ax.plot(history[f"validation_{metric}"], label="Validation")
        ax.set(xlabel="Epoch", ylabel=metric.title(), title=f"{metric.title()} over training")
        ax.legend()
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def onnx_export_and_check(model, path: Path, x_example: np.ndarray, seed: int):
    """Export a single 12-feature example; test ONNX vs PyTorch on real rows."""
    import onnx
    import onnxruntime as ort

    model.eval()
    example = torch.from_numpy(x_example[:1].copy())
    torch.onnx.export(
        model, (example,), str(path),
        input_names=["dilemma_and_cluster"],
        output_names=["logits"],
        opset_version=18,
        dynamo=True,
    )
    onnx.checker.check_model(str(path))
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(seed)
    sample = rng.choice(len(x_example), size=min(12, len(x_example)), replace=False)
    differences = []
    for ix in sample:
        candidate = x_example[ix:ix + 1].astype(np.float32)
        actual = session.run(["logits"], {"dilemma_and_cluster": candidate})[0]
        with torch.no_grad():
            expected = model(torch.from_numpy(candidate)).cpu().numpy()
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-4)
        differences.append(float(np.abs(actual - expected).max()))
    return max(differences)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-onnx", action="store_true", help="Only for debugging without ONNX dependencies")
    args = parser.parse_args()
    if min(args.epochs, args.hidden, args.batch_size, args.patience) <= 0:
        parser.error("epochs, hidden, batch-size and patience must be positive")

    root = args.project_root.resolve()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)

    X, y, c_idx, splits, ids, observed_shares, dilemma_ids, phi, dimensions = load_dataset(root, args.seed)
    group = {k: np.flatnonzero(splits == k) for k in ("train", "validation", "test")}
    for k, idx in group.items():
        n_dilemmas = len(set(ids[idx].tolist()))
        print(f"{k:>10}: {len(idx):5} labelled cluster-dilemma examples "
              f"from {n_dilemmas:4} distinct dilemmas")
    assert not (set(ids[group["train"]]) & set(ids[group["validation"]]))
    assert not (set(ids[group["train"]]) & set(ids[group["test"]]))
    assert not (set(ids[group["validation"]]) & set(ids[group["test"]]))

    model = ClusterConditionedNN(args.hidden)
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    tr = group["train"]
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X[tr]), torch.from_numpy(y[tr])),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )

    history = {f"{side}_{metric}": [] for side in ("train", "validation")
               for metric in ("loss", "accuracy")}
    best_loss, best_weights, best_epoch, stale = float("inf"), None, -1, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
        for part in ("train", "validation"):
            idx = group[part]
            val_loss, val_acc, _, _ = evaluate(model, X[idx], y[idx], criterion)
            history[f"{part}_loss"].append(val_loss)
            history[f"{part}_accuracy"].append(float(val_acc))
        current_val_loss = history["validation_loss"][-1]
        if current_val_loss < best_loss - 1e-5:
            best_loss = current_val_loss
            best_weights = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 20 == 0:
            print(f"epoch {epoch:3d}: train loss={history['train_loss'][-1]:.4f}, "
                  f"validation loss={current_val_loss:.4f}, "
                  f"validation accuracy={history['validation_accuracy'][-1]:.3f}")
        if stale >= args.patience:
            print(f"Early stopping after {epoch} epochs")
            break

    model.load_state_dict(best_weights)
    reports = root / "reports"
    models = root / "models"
    data = root / "data"
    for directory in (reports, models, data):
        directory.mkdir(parents=True, exist_ok=True)

    # Test results are generated ONLY after training and model selection.
    test = group["test"]
    test_loss, test_acc, test_prediction, test_prob = evaluate(model, X[test], y[test], criterion)
    metrics = {
        "description": "One neural network with a 9-feature dilemma and 3-bit cluster one-hot input",
        "cluster_order": list(CLUSTERS),
        "label_order": list(CLASS_NAMES),
        "n_train": int(len(tr)),
        "n_validation": int(len(group["validation"])),
        "n_test": int(len(test)),
        "selected_epoch": int(best_epoch),
        "overall_test_accuracy": float(test_acc),
        "overall_test_macro_f1": float(f1_score(y[test], test_prediction, average="macro", zero_division=0)),
        "overall_test_loss": float(test_loss),
        "by_cluster": {},
    }
    print(f"\nSelected epoch: {best_epoch}")
    print(f"Overall held-out accuracy: {test_acc:.3f}")
    for i, cluster in enumerate(CLUSTERS):
        positions = np.flatnonzero(c_idx[test] == i)
        true = y[test][positions]
        pred = test_prediction[positions]
        training_cluster = y[tr][c_idx[tr] == i]
        majority = int(np.bincount(training_cluster, minlength=2).argmax())
        cm = confusion_matrix(true, pred, labels=[0, 1])
        accuracy = float(accuracy_score(true, pred))
        f1 = float(f1_score(true, pred, average="macro", zero_division=0))
        baseline = float(np.mean(true == majority))
        metrics["by_cluster"][cluster] = {
            "n_test": int(len(true)), "accuracy": accuracy, "macro_f1": f1,
            "always_majority_baseline_accuracy": baseline,
            "majority_label_in_training": CLASS_NAMES[majority],
            "confusion_matrix_true_rows_predicted_columns": cm.tolist(),
        }
        print(f"{cluster:>8}: test accuracy={accuracy:.3f}, macro F1={f1:.3f}, "
              f"always-{CLASS_NAMES[majority]} baseline={baseline:.3f}")
        print(classification_report(true, pred, labels=[0, 1], target_names=CLASS_NAMES, zero_division=0))
        disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
        disp.plot(values_format="d", cmap="Blues", colorbar=False)
        plt.title(f"{cluster.title()} — held-out test dilemmas")
        plt.tight_layout()
        plt.savefig(reports / f"cluster_conditioned_confusion_{cluster}.png", dpi=160)
        plt.close()

    with (reports / "cluster_conditioned_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    plot_curves(history, reports / "cluster_conditioned_training_curves.png")
    with (reports / "cluster_conditioned_test_predictions.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(("dilemma_id", "cluster", "observed_label", "predicted_label",
                         "predicted_p_swerve", "observed_swerve_share"))
        for p, idx in enumerate(test):
            writer.writerow((ids[idx], CLUSTERS[c_idx[idx]], CLASS_NAMES[y[idx]],
                             CLASS_NAMES[test_prediction[p]], float(test_prob[p, 1]),
                             float(observed_shares[idx])))

    checkpoint = {
        "state_dict": model.state_dict(),
        "hidden": int(args.hidden),
        "cluster_order": CLUSTERS,
        "feature_names": dimensions,
        "input_order": list(dimensions) + [f"cluster_{c}" for c in CLUSTERS],
        "label_order": CLASS_NAMES,
        "seed": args.seed,
    }
    torch.save(checkpoint, models / "cluster_conditioned_moral_nn.pt")

    # Freeze decisions for ALL dilemmas, not just the test set, for later
    # reachability comparisons. This is a deployment prediction catalogue,
    # NOT a held-out accuracy estimate on all dilemmas.
    labels, margins = [], []
    for i in range(3):
        indicator = np.tile(np.eye(3, dtype=np.float32)[i], (len(phi), 1))
        x_cluster = np.concatenate((phi, indicator), axis=1)
        logits, _, pred = forward_numpy(model, x_cluster)
        labels.append((2 * pred - 1).astype(int).tolist())  # -1 stay, +1 swerve
        margins.append((logits[:, 1] - logits[:, 0]).astype(float).tolist())
    frozen = {
        "description": "Frozen classifier predictions for each dilemma, conditioned on cluster",
        "clusters": list(CLUSTERS),
        "dilemma_ids": dilemma_ids,
        "labels": labels,                 # [cluster, dilemma] -> -1 or +1
        "logit_swerve_minus_stay": margins,
        "label_convention": {"-1": "stay", "1": "swerve"},
    }
    with (data / "frozen_nn_predictions.json").open("w", encoding="utf-8") as fh:
        json.dump(frozen, fh)

    if not args.skip_onnx:
        onnx_path = models / "cluster_conditioned_moral_nn.onnx"
        diff = onnx_export_and_check(model, onnx_path, X[test], args.seed)
        print(f"ONNX export checked: max absolute logit difference={diff:.2e}")
    else:
        print("ONNX export skipped (--skip-onnx).")

    print("\nSaved:")
    print(f"  {models / 'cluster_conditioned_moral_nn.pt'}")
    print(f"  {data / 'frozen_nn_predictions.json'}")
    print(f"  {reports / 'cluster_conditioned_metrics.json'}")
    print(f"  {reports / 'cluster_conditioned_test_predictions.csv'}")
    if not args.skip_onnx:
        print(f"  {models / 'cluster_conditioned_moral_nn.onnx'}")


if __name__ == "__main__":
    main()
