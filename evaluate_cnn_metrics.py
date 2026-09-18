#!/usr/bin/env python3
"""Evaluate a saved pneumonia CNN checkpoint and explain each binary metric.

Example:
    python evaluate_cnn_metrics.py \
        --checkpoint cnn_pneumonia_model_224_lr_5_10_5_mom_09_patience60.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from gradcam_report import load_model, make_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default="data/chest_xray_224/test")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--json-output", type=Path, default=None)
    return parser.parse_args()


def divide(numerator: int, denominator: int) -> float:
    """Divide safely so an absent class produces NaN instead of a crash."""
    return numerator / denominator if denominator else float("nan")


def main() -> None:
    args = parse_args()
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("--threshold must lie between 0 and 1")

    root = args.root.resolve()
    checkpoint_path = root / args.checkpoint
    data_dir = root / args.data_dir
    model, checkpoint = load_model(checkpoint_path)
    dataset = make_dataset(data_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    probability_batches: list[torch.Tensor] = []
    label_batches: list[torch.Tensor] = []
    with torch.inference_mode():
        for images, labels in loader:
            probability_batches.append(model(images).squeeze(1).cpu())
            label_batches.append(labels.reshape(-1).cpu())

    probabilities = torch.cat(probability_batches).numpy()
    labels = torch.cat(label_batches).numpy().astype(np.int64)
    predictions = (probabilities >= args.threshold).astype(np.int64)

    # Compare every prediction with its known label.
    tn = int(np.sum((labels == 0) & (predictions == 0)))
    fp = int(np.sum((labels == 0) & (predictions == 1)))
    fn = int(np.sum((labels == 1) & (predictions == 0)))
    tp = int(np.sum((labels == 1) & (predictions == 1)))

    total = tn + fp + fn + tp
    sensitivity = divide(tp, tp + fn)
    specificity = divide(tn, tn + fp)
    precision = divide(tp, tp + fp)
    accuracy = divide(tp + tn, total)

    # Unlike threshold-based metrics, ROC-AUC uses every raw probability. It is
    # the probability that a random pneumonia image receives a higher score
    # than a random normal image (with half credit for tied scores).
    roc_auc = float(roc_auc_score(labels, probabilities))

    saved_accuracy = float(checkpoint["valid_accuracy"])
    if not np.isclose(accuracy, saved_accuracy, atol=1e-8):
        raise RuntimeError(
            f"Recomputed accuracy {accuracy:.12f} differs from the checkpoint's "
            f"valid_accuracy {saved_accuracy:.12f}"
        )

    results = {
        "checkpoint": checkpoint_path.name,
        "data_dir": str(data_dir),
        "threshold": args.threshold,
        "n_images": total,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "saved_valid_accuracy": saved_accuracy,
    }

    print(f"Evaluated {total} images at threshold {args.threshold:g}")
    print(f"Confusion matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"Sensitivity = TP / (TP + FN) = {tp} / {tp + fn} = {sensitivity:.1%}")
    print(f"Specificity = TN / (TN + FP) = {tn} / {tn + fp} = {specificity:.1%}")
    print(f"Precision   = TP / (TP + FP) = {tp} / {tp + fp} = {precision:.1%}")
    print(f"Accuracy    = (TP + TN) / N  = {tp + tn} / {total} = {accuracy:.1%}")
    print(f"ROC-AUC     = rank quality from all probabilities = {roc_auc:.3f}")
    print(f"Checkpoint valid_accuracy = {saved_accuracy:.1%} (match: yes)")

    if args.json_output is not None:
        output_path = args.json_output
        if not output_path.is_absolute():
            output_path = root / output_path
        output_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
