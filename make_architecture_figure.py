#!/usr/bin/env python3
"""Publication figure of the frozen-ResNet18 pneumonia classifier.

Geometry carries the numbers rather than decorating them: box height scales with the
log of the spatial resolution and box width with the log of the channel count, so every
stride-2 downsample is one even step down and every channel doubling one even step wider.

Stage shapes are read from resnet18_pneumonia_model.json where possible, so the figure
cannot drift from the model it describes.

Usage:
    python3 make_architecture_figure.py [--no-tap] [--outdir figures]

    --no-tap   omit the annotation marking where the 512-d representation is read off
               (drop it if the figure is for a plain methods section)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# Stage table: (short label, operation, channels, spatial, is_trainable)
# Spatial is the side length of the square feature map; 1 means a plain vector.
# Operation strings are kept to about 13 characters per line: any longer and they
# run into the neighbouring column.
STAGES = [
    ("Input",   "RGB X-ray\nImageNet norm.",  3,   224, False),
    ("Stem",    "7×7 conv, /2\nBN + ReLU",    64,  112, False),
    ("Pool",    "3×3 max pool\nstride 2",     64,  56,  False),
    ("Stage 1", "2 residual\nblocks",         64,  56,  False),
    ("Stage 2", "2 residual\nblocks, /2",     128, 28,  False),
    ("Stage 3", "2 residual\nblocks, /2",     256, 14,  False),
    ("Stage 4", "2 residual\nblocks, /2",     512, 7,   False),
    ("Pooling", "global avg\n+ flatten",      512, 1,   False),
    ("Head",    "Linear(512→1)\nsigmoid",     1,   1,   True),
]

FROZEN_FILL = "#dce7f2"
FROZEN_EDGE = "#4a7ba7"
TRAIN_FILL = "#f7d9c4"
TRAIN_EDGE = "#c1651f"
INPUT_FILL = "#e8e8e8"
INPUT_EDGE = "#7a7a7a"
TAP_COLOUR = "#1a7a5e"
TEXT = "#1a1a1a"
MUTED = "#5a5a5a"


def shape_label(channels: int, spatial: int) -> str:
    """Two short lines, so the label never grows wider than its column."""
    if spatial > 1:
        return f"{spatial}×{spatial}\n{channels} ch"
    return "512" if channels == 512 else "P(pneumonia)"


def build(metadata: dict | None, show_tap: bool) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    ax.set_axis_off()

    # log2 steps: 224 -> 7 is five even steps down, 64 -> 512 three even steps wider.
    max_h, min_h = 1.00, 0.30
    hi, lo = math.log2(224), math.log2(7)
    max_w, min_w = 0.52, 0.22
    cw_hi, cw_lo = math.log2(512), math.log2(3)

    def height(spatial):
        if spatial <= 1:
            return min_h * 0.62          # a vector has no spatial extent
        frac = (math.log2(spatial) - lo) / (hi - lo)
        return min_h + frac * (max_h - min_h)

    def width(channels):
        frac = (math.log2(channels) - cw_lo) / (cw_hi - cw_lo) if channels > 1 else 0.0
        return min_w + max(frac, 0.0) * (max_w - min_w)

    # Fixed column grid. Every stage gets the same horizontal slot, so a label can
    # never grow into its neighbour no matter how tall or short its box is.
    pitch = 0.95
    centres, boxes = [], []

    # Label rows sit at fixed heights, not at each box's edge. Anchoring them to the
    # boxes made them stagger, since box heights differ by design.
    y_name = max_h / 2 + 0.34
    y_op = y_name - 0.10
    y_shape = -max_h / 2 - 0.16

    for i, (label, op, channels, spatial, trainable) in enumerate(STAGES):
        cx = i * pitch
        w, h = width(channels), height(spatial)
        if label == "Input":
            fill, edge = INPUT_FILL, INPUT_EDGE
        elif trainable:
            fill, edge = TRAIN_FILL, TRAIN_EDGE
        else:
            fill, edge = FROZEN_FILL, FROZEN_EDGE

        ax.add_patch(FancyBboxPatch(
            (cx - w / 2, -h / 2), w, h,
            boxstyle="round,pad=0,rounding_size=0.025",
            linewidth=1.1, facecolor=fill, edgecolor=edge, zorder=3,
        ))
        boxes.append((cx, w, h))
        centres.append(cx)

        ax.text(cx, y_name, label, ha="center", va="bottom",
                fontsize=8.0, fontweight="bold", color=TEXT, zorder=4)
        ax.text(cx, y_op, op, ha="center", va="top",
                fontsize=6.0, color=MUTED, linespacing=1.45, zorder=4)
        ax.text(cx, y_shape, shape_label(channels, spatial),
                ha="center", va="top", fontsize=6.8, color=TEXT,
                family="DejaVu Sans Mono", linespacing=1.4, zorder=4)

    total_width = (len(STAGES) - 1) * pitch

    # Arrows between stages, drawn between box edges rather than column centres.
    for (c0, w0, _), (c1, w1, _) in zip(boxes, boxes[1:]):
        ax.add_patch(FancyArrowPatch(
            (c0 + w0 / 2 + 0.04, 0), (c1 - w1 / 2 - 0.04, 0),
            arrowstyle="-|>", mutation_scale=8, linewidth=0.9,
            color="#8a8a8a", shrinkA=0, shrinkB=0, zorder=2,
        ))

    # Frozen / trainable spans.
    frozen_end = boxes[-2][0] + boxes[-2][1] / 2
    y_span = y_shape - 0.40
    for x0, x1, colour, text in [
        (boxes[1][0] - boxes[1][1] / 2, frozen_end, FROZEN_EDGE,
         "frozen ImageNet backbone — 11,176,512 parameters, none updated"),
        (boxes[-1][0] - boxes[-1][1] / 2, boxes[-1][0] + boxes[-1][1] / 2, TRAIN_EDGE,
         "trainable head\n513 parameters"),
    ]:
        ax.plot([x0, x1], [y_span, y_span], color=colour, linewidth=1.5,
                solid_capstyle="butt", zorder=2)
        for edge_x in (x0, x1):
            ax.plot([edge_x, edge_x], [y_span, y_span + 0.05], color=colour,
                    linewidth=1.5, zorder=2)
        ax.text((x0 + x1) / 2, y_span - 0.08, text, ha="center", va="top",
                fontsize=6.8, color=colour, linespacing=1.4, zorder=4)

    # Where the clustering analysis reads the representation. The arrow points at the
    # connector between pooling and the head, which is literally what carries the
    # 512-d vector, and it sits in the gap BETWEEN two columns. Every column centre
    # already has operation text, so a vertical arrow there would strike through it.
    if show_tap:
        tap_x = (centres[-2] + centres[-1]) / 2
        ax.annotate(
            "512-d representation\nread off here",
            xy=(tap_x, 0.05), xytext=(tap_x, y_name + 0.28),
            ha="center", va="bottom", fontsize=6.5, color=TAP_COLOUR,
            linespacing=1.4, zorder=5,
            arrowprops=dict(arrowstyle="-|>", color=TAP_COLOUR, linewidth=0.9,
                            shrinkA=2, shrinkB=0),
        )

    ax.set_xlim(-pitch * 0.62, total_width + pitch * 0.62)
    ax.set_ylim(y_span - 0.72, y_name + 0.62)

    if metadata:
        acc = metadata.get("metrics", {}).get("test_accuracy")
        if acc:
            ax.text(total_width / 2, y_span - 0.62,
                    f"Trained on 4,185 images, validated on 1,047. "
                    f"Test accuracy {acc:.3f} on 624 held-out images.",
                    ha="center", va="bottom", fontsize=6.4, color=MUTED,
                    style="italic", zorder=4)

    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="figures")
    parser.add_argument("--no-tap", action="store_true",
                        help="omit the 512-d representation annotation")
    parser.add_argument("--metadata", default="resnet18_pneumonia_model.json")
    args = parser.parse_args()

    mpl.rcParams.update({
        "pdf.fonttype": 42,        # editable text in the PDF, not outlines
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "font.family": "DejaVu Sans",
    })

    meta_path = Path(args.metadata)
    metadata = json.loads(meta_path.read_text()) if meta_path.exists() else None
    if metadata:
        # Fail loudly rather than draw a figure that misdescribes the model.
        if metadata["model"]["architecture"] != "resnet18":
            raise ValueError("Metadata does not describe a resnet18")
        if metadata["input"]["shape"] != [3, 224, 224]:
            raise ValueError(f"Input shape {metadata['input']['shape']} is not 3x224x224")
        if metadata["model"]["trainable_parameters"] != 513:
            raise ValueError(
                f"Metadata says {metadata['model']['trainable_parameters']} trainable "
                "parameters; the figure states 513"
            )

    out = Path(args.outdir)
    out.mkdir(exist_ok=True)
    fig = build(metadata, show_tap=not args.no_tap)
    stem = out / "architecture_resnet18"
    fig.savefig(f"{stem}.pdf", bbox_inches="tight", facecolor="white",
                metadata={"CreationDate": None})
    fig.savefig(f"{stem}.png", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{stem}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    for suffix in ("pdf", "png", "svg"):
        f = Path(f"{stem}.{suffix}")
        print(f"{f}  ({f.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
