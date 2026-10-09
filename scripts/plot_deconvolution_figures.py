#!/usr/bin/env python3
"""Generate data-backed figures for the deconvolution revision."""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "results" / "matplotlib_cache"))
RESULTS = ROOT / "results"; FIGURES = ROOT / "figures" / "revision"


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(RESULTS / "membership_metrics.csv")
    fig, ax = plt.subplots(figsize=(7, 3.8))
    methods = metrics["method"].tolist(); x = range(len(methods))
    ax.bar([i - .18 for i in x], metrics["membership_precision"], .35, label="Membership precision", color="#35618d")
    ax.bar([i + .18 for i in x], metrics["membership_recall"], .35, label="Membership recall", color="#d1843c")
    ax.set_xticks(list(x)); ax.set_xticklabels(methods, rotation=18, ha="right"); ax.set_ylim(0, 1); ax.set_ylabel("Score")
    ax.legend(frameon=False); fig.tight_layout(); fig.savefig(FIGURES / "fig_deconvolution_membership_evaluation.pdf"); plt.close(fig)

    coverage = pd.read_csv(RESULTS / "candidate_coverage.csv").set_index("construction")
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    names = coverage.index.tolist(); positions = list(range(len(names)))
    for offset, column, color in ((-.22, "candidate_recall", "#35618d"), (0, "candidate_coverage", "#d1843c"), (.22, "candidate_purity", "#5b9166")):
        ax.bar([x + offset for x in positions], coverage[column], .22, label=column.replace("candidate_", "").replace("_", " "), color=color)
    ax.set_xticks(positions); ax.set_xticklabels(names); ax.set_ylim(0, 1); ax.set_ylabel("Score"); ax.set_xlabel(""); ax.legend(frameon=False, fontsize=8); fig.tight_layout()
    fig.savefig(FIGURES / "fig_deconvolution_candidate_coverage.pdf"); plt.close(fig)

    # Conceptual figure uses the exact method stages and deliberately contains
    # no synthetic performance claim.
    fig, ax = plt.subplots(figsize=(10, 2.8)); ax.axis("off")
    labels = ["PPI graph", "MCL coarse\nlocalization", "Heterogeneity\ntest", "Local split into\nsubcomplexes", "Overlap-aware\nassignment", "Bounded\nsupplementation", "Evidence bundle"]
    for i, label in enumerate(labels):
        ax.text(i, .5, label, ha="center", va="center", bbox=dict(boxstyle="round,pad=.4", fc="#e8eff6", ec="#35618d"), fontsize=9)
        if i:
            ax.annotate("", xy=(i - .24, .5), xytext=(i - .72, .5), arrowprops=dict(arrowstyle="->", color="#555555"))
    ax.set_xlim(-.7, len(labels) - .3); ax.set_ylim(0, 1); fig.tight_layout()
    fig.savefig(FIGURES / "fig_revised_trace_ppi_pipeline.pdf"); plt.close(fig)


if __name__ == "__main__":
    main()
