#!/usr/bin/env python3
"""Global multi-membership statistics for TRACE-PPI and overlapping baselines."""
from __future__ import annotations

import pandas as pd

from revision_utils import (
    REV_FIGURES,
    REV_RESULTS,
    REV_TABLES,
    clusters_from_membership,
    ensure_revision_dirs,
    load_clusterone,
    load_dataset,
    load_trace_assignments,
    multi_membership_stats,
)


def _counts_from_clusters(clusters, method, dataset):
    stats = multi_membership_stats(clusters, method, dataset)
    counts = stats.pop("assignment_counts")
    return stats, counts


def main() -> None:
    ensure_revision_dirs()
    from trace_ppi import baselines

    rows = []
    hist_rows = []
    for dataset in ("gavin", "krogan"):
        graph, go = load_dataset(dataset)

        trace_df = load_trace_assignments(dataset)
        trace_clusters = clusters_from_membership(trace_df)
        stats, counts = _counts_from_clusters(trace_clusters, "TRACE-PPI", dataset)
        rows.append(stats)
        hist_rows.append(pd.DataFrame({"dataset": dataset, "method": "TRACE-PPI", "n_modules_per_protein": counts.values}))

        mcl = baselines.mcl_only(graph, inflation=2.0)
        stats, counts = _counts_from_clusters(mcl, "MCL", dataset)
        rows.append(stats)
        hist_rows.append(pd.DataFrame({"dataset": dataset, "method": "MCL", "n_modules_per_protein": counts.values}))

        overlap = baselines.mcl_overlap_heuristic(graph, go, inflation=2.0)
        stats, counts = _counts_from_clusters(overlap, "MCL+overlap", dataset)
        rows.append(stats)
        hist_rows.append(pd.DataFrame({"dataset": dataset, "method": "MCL+overlap", "n_modules_per_protein": counts.values}))

        clusterone = load_clusterone(dataset)
        if clusterone is not None:
            stats, counts = _counts_from_clusters(clusterone, "ClusterONE", dataset)
            rows.append(stats)
            hist_rows.append(pd.DataFrame({"dataset": dataset, "method": "ClusterONE", "n_modules_per_protein": counts.values}))

        slpa = baselines.slpa(graph, threshold=0.05, iterations=100, seed=42)
        stats, counts = _counts_from_clusters(slpa, "SLPA (t=0.05)", dataset)
        rows.append(stats)
        hist_rows.append(pd.DataFrame({"dataset": dataset, "method": "SLPA (t=0.05)", "n_modules_per_protein": counts.values}))

    df = pd.DataFrame(rows)
    df.to_csv(REV_TABLES / "table_multimembership.csv", index=False)
    df.to_csv(REV_RESULTS / "multimembership.csv", index=False)
    hist = pd.concat(hist_rows, ignore_index=True)
    hist.to_csv(REV_RESULTS / "multimembership_assignment_counts.csv", index=False)
    _plot(hist)
    print(df.to_string(index=False))


def _plot(hist: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    methods = ["MCL", "MCL+overlap", "ClusterONE", "SLPA (t=0.05)", "TRACE-PPI"]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1), sharey=True)
    for ax, dataset in zip(axes, ("gavin", "krogan")):
        sub = hist[hist["dataset"] == dataset]
        present = [m for m in methods if m in set(sub["method"])]
        data = [sub.loc[sub["method"] == m, "n_modules_per_protein"].clip(upper=8).tolist() for m in present]
        ax.boxplot(data, showfliers=False)
        ax.set_xticklabels(present)
        ax.set_title(dataset.capitalize())
        ax.set_xlabel("Method")
        ax.tick_params(axis="x", rotation=25)
        ax.set_ylabel("Modules per assigned protein")
    fig.suptitle("Distribution of module assignments per protein")
    fig.tight_layout()
    fig.savefig(REV_FIGURES / "figure_multimembership.pdf", bbox_inches="tight")
    fig.savefig(REV_FIGURES / "figure_multimembership.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
