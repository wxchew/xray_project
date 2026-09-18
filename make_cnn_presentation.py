#!/usr/bin/env python3
"""Create reproducible presentation figures for the original pneumonia CNN."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from gradcam_report import evaluate, hirescam, load_model, make_dataset
from make_resnet18_presentation import (
    CATEGORIES,
    CLASS_NAMES,
    DISPLAY_METRICS,
    bootstrap_intervals,
    category,
    choose_examples,
    compute_metrics,
    original_and_overlay,
    plot_confusion,
    plot_examples,
    plot_precision_recall,
    plot_roc,
    save_figure,
    sha256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--checkpoint", default="cnn_pneumonia_model.pt")
    parser.add_argument("--data-dir", default="data/chest_xray_224/test")
    parser.add_argument("--output-dir", default="cnn_presentation")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 < args.threshold < 1:
        raise ValueError("--threshold must lie strictly between zero and one")
    if args.bootstrap < 1:
        raise ValueError("--bootstrap must be positive")

    root = args.root.resolve()
    checkpoint_path = root / args.checkpoint
    data_dir = root / args.data_dir
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_hash_before = sha256(checkpoint_path)
    model, checkpoint = load_model(checkpoint_path)
    if model.classifier[0].in_features != 484:
        raise RuntimeError(
            "This presentation is for the original 224x224 CNN; expected 484 flattened features, "
            f"found {model.classifier[0].in_features}"
        )
    dataset = make_dataset(data_dir)
    records = evaluate(model, dataset)

    paths = [record.path for record in records]
    labels = np.array([record.true_class for record in records], dtype=np.int64)
    probabilities = np.array([record.probability for record in records], dtype=np.float64)
    if len(labels) != 624:
        raise RuntimeError(f"Expected 624 validation images, found {len(labels)}")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise RuntimeError("Model produced invalid probabilities")

    predictions = (probabilities > args.threshold).astype(np.int64)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    if int(matrix.sum()) != len(labels):
        raise RuntimeError("Confusion matrix does not contain every prediction")

    metrics = compute_metrics(labels, probabilities, args.threshold)
    stored_accuracy = float(checkpoint["valid_accuracy"])
    if not np.isclose(metrics["accuracy"], stored_accuracy, atol=1e-8):
        raise RuntimeError(
            f"Recomputed validation accuracy {metrics['accuracy']:.12f} does not match "
            f"the saved value {stored_accuracy:.12f}"
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
        heatmap = hirescam(model, image_tensor, int(predictions[index]))
        if heatmap.shape != tuple(image_tensor.shape[-2:]) or not np.isfinite(heatmap).all():
            raise RuntimeError(f"Invalid HiResCAM map for {paths[index]}")
        original, overlay = original_and_overlay(paths[index], heatmap)
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
            "hirescam_layer": "conv[3]",
            "native_attribution_shape": [22, 22],
        }

    figure, axis = plt.subplots(figsize=(5.6, 5.0), constrained_layout=True)
    plot_confusion(axis, matrix, args.threshold)
    save_figure(figure, output_dir, "panel_a_confusion_matrix")

    figure, axis = plt.subplots(figsize=(5.6, 5.0), constrained_layout=True)
    plot_roc(axis, labels, probabilities, metrics["roc_auc"], model_label="Original CNN")
    save_figure(figure, output_dir, "panel_b_roc_curve")

    figure, axis = plt.subplots(figsize=(5.6, 5.0), constrained_layout=True)
    plot_precision_recall(
        axis,
        labels,
        probabilities,
        metrics["average_precision"],
        model_label="Original CNN",
    )
    save_figure(figure, output_dir, "panel_c_precision_recall_curve")

    figure = plt.figure(figsize=(12, 6), constrained_layout=True)
    grid = figure.add_gridspec(2, 4)
    plot_examples(figure, grid, examples)
    figure.text(
        0.99,
        -0.035,
        "HiResCAM from conv[3] (native 22×22); overlays are explanations, not lesion masks.",
        ha="right",
        fontsize=8,
        color="#444444",
    )
    save_figure(figure, output_dir, "panel_d_hirescam_examples")

    figure = plt.figure(figsize=(15, 11), constrained_layout=True)
    outer = figure.add_gridspec(2, 2, height_ratios=[1, 1.05])
    axis_a = figure.add_subplot(outer[0, 0])
    plot_confusion(axis_a, matrix, args.threshold)
    axis_a.text(-0.17, 1.08, "A", transform=axis_a.transAxes, fontsize=18, fontweight="bold")
    axis_b = figure.add_subplot(outer[0, 1])
    plot_roc(axis_b, labels, probabilities, metrics["roc_auc"], model_label="Original CNN")
    axis_b.text(-0.12, 1.08, "B", transform=axis_b.transAxes, fontsize=18, fontweight="bold")
    axis_c = figure.add_subplot(outer[1, 0])
    plot_precision_recall(
        axis_c,
        labels,
        probabilities,
        metrics["average_precision"],
        model_label="Original CNN",
    )
    axis_c.text(-0.12, 1.08, "C", transform=axis_c.transAxes, fontsize=18, fontweight="bold")
    example_grid = outer[1, 1].subgridspec(2, 4, wspace=0.04, hspace=0.05)
    plot_examples(figure, example_grid, examples)
    panel_d_label = figure.add_subplot(outer[1, 1], frameon=False)
    panel_d_label.set_xticks([])
    panel_d_label.set_yticks([])
    panel_d_label.text(-0.12, 1.08, "D", transform=panel_d_label.transAxes, fontsize=18, fontweight="bold")
    panel_d_label.set_zorder(-1)
    figure.suptitle("Original CNN pneumonia classifier — notebook validation set (n = 624)", fontsize=16)
    figure.text(
        0.99,
        -0.025,
        "HiResCAM native resolution: 22×22. ROI overlays are not lesion segmentations.",
        ha="right",
        fontsize=8,
        color="#444444",
    )
    save_figure(figure, output_dir, "cnn_presentation_four_panel")

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
    axis.set_title("Original CNN validation-set performance", fontsize=14, pad=12)
    axis.text(
        0.5,
        0.015,
        "Notebook validation set; used for early stopping/model selection. Threshold = 0.5. "
        "Image-level intervals; patient independence unverified.",
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=8,
        color="#444444",
    )
    save_figure(figure, output_dir, "cnn_metrics_table")

    with (output_dir / "cnn_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
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

    with (output_dir / "cnn_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["metric", "estimate", "ci_2.5_percent", "ci_97.5_percent"])
        for key, label in DISPLAY_METRICS:
            writer.writerow([label, metrics[key], intervals[key][0], intervals[key][1]])

    results = {
        "model": str(checkpoint_path.relative_to(root)),
        "checkpoint_sha256": checkpoint_hash_before,
        "architecture": {
            "source": "04_chest_Xray_CNN.ipynb",
            "input": "1x224x224 grayscale",
            "flattened_features": 484,
            "classifier": "Linear(484,16) -> Linear(16,8) -> Linear(8,1) -> Sigmoid",
        },
        "preprocessing": "read_image followed by ToDtype(torch.float32); pixel scale unchanged",
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
            key: {"estimate": metrics[key], "ci_95_percentile": list(intervals[key])}
            for key, _ in DISPLAY_METRICS
        },
        "checkpoint_metrics": {
            "train_accuracy": float(checkpoint["train_accuracy"]),
            "valid_accuracy": stored_accuracy,
        },
        "bootstrap": {
            "draws": args.bootstrap,
            "seed": args.seed,
            "unit": "image",
            "limitation": "Intervals condition on the fixed model and do not account for possible patient dependence.",
        },
        "selected_hirescam_examples": selected_metadata,
        "evaluation_note": "The notebook used data/chest_xray_224/test as its validation set during training.",
        "known_limitations": [
            "The evaluation data were used for early stopping/model selection.",
            "Patient independence is unverified.",
            "HiResCAM has 22x22 native resolution and is not a lesion segmentation.",
        ],
    }
    (output_dir / "cnn_results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

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
