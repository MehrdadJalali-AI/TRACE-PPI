#!/usr/bin/env python3
"""Create a step-by-step TRACE-PPI pipeline figure for the revision."""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from revision_utils import REV_FIGURES, ensure_revision_dirs


STEPS = [
    ("1. Input", "Weighted PPI network\nand optional GO/text annotations"),
    ("2. Preprocess", "Clean identifiers\nand normalise edge weights"),
    ("3. Evidence profiles", "Topology, GO features,\nand semantic embeddings"),
    ("4. Nucleus ranking", "BH score + non-maximum\nsuppression (candidates only)"),
    ("5. Candidates", "MCL, ego, greedy, semantic\nk-NN, and hybrid unions"),
    ("6. Scoring", "Candidate scores and\nprotein–module scores"),
    ("7. Refinement", "Overlap reassignment and\nrecall-safe supplementation"),
    ("8-10. Output", "Confidence labels, evidence\nbundles, gold-standard evaluation"),
]


def main() -> None:
    ensure_revision_dirs()
    fig, ax = plt.subplots(figsize=(11.4, 6.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8.2)
    ax.axis("off")
    ax.set_title("TRACE-PPI algorithmic pipeline", fontsize=14, pad=8)

    coords = [
        (0.4, 5.8), (3.2, 5.8), (6.0, 5.8), (8.8, 5.8),
        (0.4, 2.6), (3.2, 2.6), (6.0, 2.6), (8.8, 2.6),
    ]
    boxes = []
    for (x, y), (title, body) in zip(coords, STEPS):
        box = FancyBboxPatch(
            (x, y), 2.7, 1.9,
            boxstyle="round,pad=0.04,rounding_size=0.12",
            linewidth=1.2,
            edgecolor="#1f4e79",
            facecolor="#e8f1fa",
        )
        ax.add_patch(box)
        ax.text(x + 1.35, y + 1.45, title, ha="center", va="center", fontsize=10, fontweight="bold")
        ax.text(x + 1.35, y + 0.7, body, ha="center", va="center", fontsize=8)
        boxes.append((x, y))

    def arrow(a, b):
        ax.annotate(
            "",
            xy=(b[0], b[1] + 0.95),
            xytext=(a[0] + 2.7, a[1] + 0.95),
            arrowprops=dict(arrowstyle="->", color="#1f4e79", lw=1.4),
        )

    for i in range(3):
        arrow(coords[i], coords[i + 1])
    ax.annotate(
        "",
        xy=(coords[4][0] + 1.35, coords[4][1] + 1.9),
        xytext=(coords[3][0] + 1.35, coords[3][1]),
        arrowprops=dict(arrowstyle="->", color="#1f4e79", lw=1.4),
    )
    for i in range(4, 7):
        arrow(coords[i], coords[i + 1])

    ax.text(
        6.0,
        0.55,
        "Final modules are seeded from MCL with overlap reassignment; nucleus/candidate sources support scoring, ablations, and audit context.",
        ha="center",
        va="center",
        fontsize=8,
        style="italic",
    )
    fig.tight_layout()
    fig.savefig(REV_FIGURES / "figure_pipeline.pdf", bbox_inches="tight")
    fig.savefig(REV_FIGURES / "figure_pipeline.png", dpi=220, bbox_inches="tight")
    fig.savefig(REV_FIGURES.parent.parent / "Figure_1_pipeline.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Wrote pipeline figure")


if __name__ == "__main__":
    main()
