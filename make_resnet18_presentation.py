#!/usr/bin/env python3
"""Create presentation figures for the saved ResNet18 pneumonia classifier."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from resnet18_hirescam_report import evaluate, hirescam, load_metadata, load_model, make_dataset


CLASS_NAMES = ("NORMAL", "PNEUMONIA")
CATEGORIES = (
    ("TP", "True positive"),
    ("TN", "True negative"),
    ("FP", "False positive"),
    ("FN", "False negative"),
)
DISPLAY_METRICS = (
    ("accuracy", "Accuracy"),
    ("sensitivity", "Sensitivity"),
    ("specificity", "Specificity"),
    ("precision", "Precision"),
    ("f1", "F1 score"),
    ("balanced_accuracy", "Balanced accuracy"),
    ("roc_auc", "ROC-AUC"),
    ("average_precision", "Average precision"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--checkpoint", default="resnet18_pneumonia_model.pt")
    parser.add_argument("--metadata", default="resnet18_pneumonia_model.json")
    parser.add_argument("--data-dir", default="data/chest_xray_224/test")
    parser.add_argument("--output-dir", default="resnet18_presentation")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else float("nan")


def compute_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    predictions = (probabilities > threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "sensitivity": divide(int(tp), int(tp + fn)),
        "specificity": divide(int(tn), int(tn + fp)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
    }


def bootstrap_intervals(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float, draws: int, seed: int
) -> dict[str, tuple[float, float]]:
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {key: [] for key, _ in DISPLAY_METRICS}
    for _ in range(draws):
        indices = rng.integers(0, len(labels), len(labels))
        sampled_labels = labels[indices]
        if np.unique(sampled_labels).size < 2:
            continue
        current = compute_metrics(sampled_labels, probabilities[indices], threshold)
        for key in samples:
            samples[key].append(current[key])
    if any(not values for values in samples.values()):
        raise RuntimeError("Bootstrap produced no valid two-class resamples")
    return {
        key: (float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5)))
        for key, values in samples.items()
    }


def style_axis(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(labelsize=9)


def plot_confusion(axis: plt.Axes, matrix: np.ndarray, threshold: float = 0.5) -> None:
    axis.imshow(matrix, cmap="Blues", vmin=0, vmax=max(1, int(matrix.max())))
    row_totals = matrix.sum(axis=1, keepdims=True)
    row_percent = np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(matrix, dtype=np.float64),
        where=row_totals != 0,
    ) * 100
    cutoff = matrix.max() / 2
    for row in range(2):
        for column in range(2):
            axis.text(
                column,
                row,
                f"{matrix[row, column]}\n{row_percent[row, column]:.1f}%",
                ha="center",
                va="center",
                fontsize=12,
                color="white" if matrix[row, column] > cutoff else "#172033",
            )
    axis.set_xticks([0, 1], CLASS_NAMES)
    axis.set_yticks([0, 1], CLASS_NAMES)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_title(f"Confusion matrix (threshold = {threshold:g})", fontsize=12)


def plot_roc(
    axis: plt.Axes,
    labels: np.ndarray,
    probabilities: np.ndarray,
    auc: float,
    model_label: str = "ResNet18",
) -> None:
    false_positive, true_positive, _ = roc_curve(labels, probabilities)
    axis.plot(false_positive, true_positive, color="#1769aa", linewidth=2.5, label=f"{model_label} (AUC = {auc:.3f})")
    axis.plot([0, 1], [0, 1], color="#777777", linestyle="--", linewidth=1.3, label="Random classifier")
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="False-positive rate", ylabel="True-positive rate", title="ROC curve")
    axis.legend(loc="lower right", frameon=False, fontsize=9)
    axis.grid(alpha=0.2)
    style_axis(axis)


def plot_precision_recall(
    axis: plt.Axes,
    labels: np.ndarray,
    probabilities: np.ndarray,
    average_precision: float,
    model_label: str = "ResNet18",
) -> None:
    precision, recall, _ = precision_recall_curve(labels, probabilities)
    prevalence = float(labels.mean())
    axis.plot(recall, precision, color="#c64e52", linewidth=2.5, label=f"{model_label} (AP = {average_precision:.3f})")
    axis.axhline(prevalence, color="#777777", linestyle="--", linewidth=1.3, label=f"Prevalence = {prevalence:.3f}")
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Recall (sensitivity)", ylabel="Precision", title="Precision–recall curve")
    axis.legend(loc="lower left", frameon=False, fontsize=9)
    axis.grid(alpha=0.2)
    style_axis(axis)


def category(true_label: int, predicted_label: int) -> str:
    return {(1, 1): "TP", (0, 0): "TN", (0, 1): "FP", (1, 0): "FN"}[(true_label, predicted_label)]


def choose_examples(
    paths: list[Path], labels: np.ndarray, probabilities: np.ndarray, predictions: np.ndarray
) -> dict[str, int | None]:
    selected: dict[str, int | None] = {}
    for short_name, _ in CATEGORIES:
        candidates = [
            index
            for index, (truth, prediction) in enumerate(zip(labels, predictions))
            if category(int(truth), int(prediction)) == short_name
        ]
        if not candidates:
            selected[short_name] = None
            continue
        confidences = np.array(
            [probabilities[index] if predictions[index] else 1 - probabilities[index] for index in candidates]
        )
        median = float(np.median(confidences))
        selected[short_name] = min(
            candidates,
            key=lambda index: (
                abs((probabilities[index] if predictions[index] else 1 - probabilities[index]) - median),
                paths[index].name,
            ),
        )
    return selected


def original_and_overlay(path: Path, heatmap: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    original = Image.open(path).convert("L").resize((heatmap.shape[1], heatmap.shape[0]))
    grayscale = np.asarray(original, dtype=np.float32) / 255.0
    rgb = np.repeat(grayscale[..., None], 3, axis=-1)
    color = plt.get_cmap("inferno")(heatmap)[..., :3]
    alpha = np.clip(heatmap[..., None] * 0.72, 0, 0.72)
    return rgb, np.clip((1 - alpha) * rgb + alpha * color, 0, 1)


def plot_examples(
    figure: plt.Figure,
    grid,
    examples: dict[str, dict | None],
    panel_title: bool = True,
) -> None:
    for column, (short_name, long_name) in enumerate(CATEGORIES):
        item = examples[short_name]
        for row in range(2):
            axis = figure.add_subplot(grid[row, column])
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
            if item is None:
                axis.text(0.5, 0.5, "No cases", ha="center", va="center", transform=axis.transAxes)
            else:
                axis.imshow(item["original"] if row == 0 else item["overlay"], cmap=None)
                if row == 0:
                    axis.set_title(f"{short_name}: {long_name}", fontsize=8.5, pad=3)
                else:
                    axis.set_xlabel(
                        f"P(pneumonia) = {item['probability']:.3f}\n{item['filename']}",
                        fontsize=7,
                        labelpad=2,
                    )
            if column == 0:
                axis.set_ylabel("Original" if row == 0 else "HiResCAM", fontsize=8)
    if panel_title:
        title_axis = figure.add_subplot(grid[:, :], frameon=False)
        title_axis.set_xticks([])
        title_axis.set_yticks([])
        title_axis.set_title("Median-confidence examples by outcome", fontsize=12, pad=22)
        title_axis.set_zorder(-1)


def save_figure(figure: plt.Figure, output_dir: Path, stem: str) -> None:
    figure.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(
        output_dir / f"{stem}.pdf",
        bbox_inches="tight",
        facecolor="white",
        metadata={
            "Creator": "reproducible pneumonia presentation generator",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if not 0 < args.threshold < 1:
        raise ValueError("--threshold must lie strictly between zero and one")
    if args.bootstrap < 1:
        raise ValueError("--bootstrap must be positive")

    root = args.root.resolve()
    checkpoint_path = root / args.checkpoint
    metadata_path = root / args.metadata
    data_dir = root / args.data_dir
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_hash_before = sha256(checkpoint_path)
    metadata = load_metadata(metadata_path)
    model, checkpoint_info = load_model(checkpoint_path)
    dataset = make_dataset(data_dir)
    records = evaluate(model, dataset)

    paths = [record.path for record in records]
    labels = np.array([record.true_class for record in records], dtype=np.int64)
    probabilities = np.array([record.probability for record in records], dtype=np.float64)
    if len(labels) != 624:
        raise RuntimeError(f"Expected 624 test images, found {len(labels)}")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise RuntimeError("Model produced invalid probabilities")
    predictions = (probabilities > args.threshold).astype(np.int64)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    if int(matrix.sum()) != len(labels):
        raise RuntimeError("Confusion matrix does not contain every prediction")

    metrics = compute_metrics(labels, probabilities, args.threshold)
    stored_test_accuracy = checkpoint_info.get(
        "test_accuracy", metadata.get("metrics", {}).get("test_accuracy")
    )
    if stored_test_accuracy is None or not np.isclose(metrics["accuracy"], stored_test_accuracy, atol=1e-8):
        raise RuntimeError(
            f"Recomputed test accuracy {metrics['accuracy']:.12f} does not match saved value "
            f"{stored_test_accuracy}"
        )
    intervals = bootstrap_intervals(labels, probabilities, args.threshold, args.bootstrap, args.seed)

    chosen = choose_examples(paths, labels, probabilities, predictions)
    examples: dict[str, dict | None] = {}
    selected_metadata: dict[str, dict | None] = {}
    for short_name, _ in CATEGORIES:
        index = chosen[short_name]
        if index is None:
            examples[short_name] = None
            selected_metadata[short_name] = None
            continue
        image_tensor, _ = dataset[index]
        attribution = hirescam(model, image_tensor, int(predictions[index]))
        original, overlay = original_and_overlay(paths[index], attribution.resized)
        examples[short_name] = {
            "original": original,
            "overlay": overlay,
            "filename": paths[index].name,
            "probability": float(probabilities[index]),
        }
        selected_metadata[short_name] = {
            "file": str(paths[index].relative_to(root)),
            "true_label": CLASS_NAMES[int(labels[index])],
            "predicted_label": CLASS_NAMES[int(predictions[index])],
            "pneumonia_probability": float(probabilities[index]),
            "selection": "closest to the median prediction confidence within this outcome",
            "hirescam_target": "predicted-class pre-sigmoid score",
            "hirescam_layer": "layer4[-1]",
        }

    # Individual quantitative panels.
    figure, axis = plt.subplots(figsize=(5.6, 5.0), constrained_layout=True)
    plot_confusion(axis, matrix)
    save_figure(figure, output_dir, "panel_a_confusion_matrix")

    figure, axis = plt.subplots(figsize=(5.6, 5.0), constrained_layout=True)
    plot_roc(axis, labels, probabilities, metrics["roc_auc"])
    save_figure(figure, output_dir, "panel_b_roc_curve")

    figure, axis = plt.subplots(figsize=(5.6, 5.0), constrained_layout=True)
    plot_precision_recall(axis, labels, probabilities, metrics["average_precision"])
    save_figure(figure, output_dir, "panel_c_precision_recall_curve")

    figure = plt.figure(figsize=(12, 6), constrained_layout=True)
    grid = figure.add_gridspec(2, 4)
    plot_examples(figure, grid, examples)
    save_figure(figure, output_dir, "panel_d_hirescam_examples")

    # Combined four-panel figure.
    figure = plt.figure(figsize=(15, 11), constrained_layout=True)
    outer = figure.add_gridspec(2, 2, height_ratios=[1, 1.05])
    axis_a = figure.add_subplot(outer[0, 0])
    plot_confusion(axis_a, matrix)
    axis_a.text(-0.17, 1.08, "A", transform=axis_a.transAxes, fontsize=18, fontweight="bold")
    axis_b = figure.add_subplot(outer[0, 1])
    plot_roc(axis_b, labels, probabilities, metrics["roc_auc"])
    axis_b.text(-0.12, 1.08, "B", transform=axis_b.transAxes, fontsize=18, fontweight="bold")
    axis_c = figure.add_subplot(outer[1, 0])
    plot_precision_recall(axis_c, labels, probabilities, metrics["average_precision"])
    axis_c.text(-0.12, 1.08, "C", transform=axis_c.transAxes, fontsize=18, fontweight="bold")
    example_grid = outer[1, 1].subgridspec(2, 4, wspace=0.04, hspace=0.05)
    plot_examples(figure, example_grid, examples)
    panel_d_label = figure.add_subplot(outer[1, 1], frameon=False)
    panel_d_label.set_xticks([])
    panel_d_label.set_yticks([])
    panel_d_label.text(-0.12, 1.08, "D", transform=panel_d_label.transAxes, fontsize=18, fontweight="bold")
    panel_d_label.set_zorder(-1)
    figure.suptitle("ResNet18 pneumonia classifier — internal test set (n = 624)", fontsize=16)
    save_figure(figure, output_dir, "resnet18_presentation_four_panel")

    # Standalone presentation table.
    table_rows = [
        [label, f"{metrics[key]:.3f}", f"{intervals[key][0]:.3f}–{intervals[key][1]:.3f}"]
        for key, label in DISPLAY_METRICS
    ]
    figure, axis = plt.subplots(figsize=(8.2, 4.4))
    axis.axis("off")
    table = axis.table(
        cellText=table_rows,
        colLabels=["Metric", "Estimate", "95% CI (bootstrap)"],
        colLoc="left",
        cellLoc="left",
        loc="center",
        colWidths=[0.39, 0.19, 0.39],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.55)
    for column in range(3):
        table[(0, column)].set_text_props(weight="bold", color="white")
        table[(0, column)].set_facecolor("#244a73")
    for row in range(1, len(table_rows) + 1):
        if row % 2 == 0:
            for column in range(3):
                table[(row, column)].set_facecolor("#edf3f8")
    axis.set_title("ResNet18 test-set performance", fontsize=14, pad=12)
    axis.text(
        0.5,
        0.015,
        "Internal test set; previously used during model development. Threshold = 0.5. "
        "Image-level intervals; patient independence unverified.",
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=8,
        color="#444444",
    )
    save_figure(figure, output_dir, "resnet18_metrics_table")

    with (output_dir / "resnet18_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["file", "true_label", "predicted_label", "pneumonia_probability", "outcome"])
        for path, truth, prediction, probability in zip(paths, labels, predictions, probabilities):
            writer.writerow(
                [
                    str(path.relative_to(root)),
                    CLASS_NAMES[int(truth)],
                    CLASS_NAMES[int(prediction)],
                    f"{probability:.10f}",
                    category(int(truth), int(prediction)),
                ]
            )

    with (output_dir / "resnet18_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["metric", "estimate", "ci_2.5_percent", "ci_97.5_percent"])
        for key, label in DISPLAY_METRICS:
            writer.writerow([label, metrics[key], intervals[key][0], intervals[key][1]])

    results = {
        "model": str(checkpoint_path.relative_to(root)),
        "checkpoint_sha256": checkpoint_hash_before,
        "data": str(data_dir.relative_to(root)),
        "n_images": int(len(labels)),
        "class_counts": {CLASS_NAMES[value]: int((labels == value).sum()) for value in (0, 1)},
        "decision_rule": f"P(PNEUMONIA) > {args.threshold}",
        "confusion_matrix": {
            "rows": "true label [NORMAL, PNEUMONIA]",
            "columns": "predicted label [NORMAL, PNEUMONIA]",
            "values": matrix.tolist(),
        },
        "metrics": {
            key: {
                "estimate": metrics[key],
                "ci_95_percentile": list(intervals[key]),
            }
            for key, _ in DISPLAY_METRICS
        },
        "bootstrap": {
            "draws": args.bootstrap,
            "seed": args.seed,
            "unit": "image",
            "limitation": "Intervals condition on the fixed model and do not account for possible patient dependence.",
        },
        "selected_hirescam_examples": selected_metadata,
        "evaluation_note": "Internal test set; previously used during model development.",
        "known_limitations": [
            "Patient independence is unverified.",
            "The training/validation split contains exact duplicate images across those two subsets.",
        ],
    }
    (output_dir / "resnet18_results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    checkpoint_hash_after = sha256(checkpoint_path)
    if checkpoint_hash_after != checkpoint_hash_before:
        raise RuntimeError("Checkpoint changed while figures were generated")

    print(f"Evaluated {len(labels)} images")
    print(f"Confusion matrix [[TN, FP], [FN, TP]]: {matrix.tolist()}")
    for key, label in DISPLAY_METRICS:
        low, high = intervals[key]
        print(f"{label}: {metrics[key]:.4f} (95% bootstrap interval {low:.4f}–{high:.4f})")
    print(f"Wrote presentation outputs to {output_dir}")


if __name__ == "__main__":
    main()
