#!/usr/bin/env python3
"""Cross-dataset figures for Gavin and Krogan benchmark transfer."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "results" / "matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = ROOT / "figures"
RESULTS = ROOT / "results"
TABLES = ROOT / "tables"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 300,
    }
)

LABELS = {
    "MCL": "MCL",
    "MCL_overlap_heuristic": "MCL+overlap",
    "ClusterONE_exact": "ClusterONE",
    "SLPA_best_sensitivity": "SLPA",
    "TRACE-PPI_final": "TRACE-PPI",
    "score_select_only_ablation": "Score-select",
    "naive_expansion_negative_control": "Naive expansion",
    "core_only_ablation": "Core-only",
}

COLORS = {
    "MCL": "#4D4D4D",
    "MCL+overlap": "#009E73",
    "ClusterONE": "#D55E00",
    "SLPA": "#CC79A7",
    "TRACE-PPI": "#0072B2",
    "Score-select": "#E69F00",
    "Naive expansion": "#999999",
    "Core-only": "#56B4E9",
}


def _load_benchmark():
    rows = []
    gavin = TABLES / "table1_trace_ppi_final_benchmark.csv"
    if gavin.exists():
        df = pd.read_csv(gavin)
        df["dataset"] = "Gavin"
        rows.append(df)
    krogan = RESULTS / "krogan" / "benchmark_summary.csv"
    if krogan.exists():
        df = pd.read_csv(krogan)
        df = df.rename(columns={"f1": "f1_mean", "precision": "precision_mean", "recall": "recall_mean"})
        df["dataset"] = "Krogan"
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True, sort=False)
    out["label"] = out["method"].map(LABELS).fillna(out["method"])
    return out


def _save(fig, name):
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)


def benchmark_comparison():
    df = _load_benchmark()
    keep = ["MCL", "MCL+overlap", "ClusterONE", "SLPA", "TRACE-PPI", "Score-select"]
    df = df[df["label"].isin(keep)].copy()
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    datasets = ["Gavin", "Krogan"]
    x = np.arange(len(datasets))
    width = 0.12
    offsets = np.linspace(-0.30, 0.30, len(keep))
    for off, method in zip(offsets, keep):
        vals = []
        for ds in datasets:
            sub = df[(df["dataset"] == ds) & (df["label"] == method)]
            vals.append(float(sub["f1_mean"].iloc[0]) if not sub.empty else np.nan)
        ax.bar(x + off, vals, width=width, label=method, color=COLORS.get(method, "#777"))
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("F1")
    ax.set_title("Cross-dataset predictive benchmark")
    ax.legend(frameon=False, ncol=3, fontsize=7.5)
    ax.grid(axis="y", color="#E8ECEF", lw=0.8)
    _save(fig, "trace_ppi_fig13_cross_dataset_benchmark.pdf")


def auditability_transfer():
    audit_path = RESULTS / "cross_dataset_auditability.csv"
    if not audit_path.exists():
        return
    df = pd.read_csv(audit_path)
    df = df[df["confidence_label"].isin(["core", "inner", "outer", "uncertain"])].copy()
    if df.empty:
        return
    df["dataset"] = df["dataset"].str.title()
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0), sharey=True)
    for ax, metric, title in [
        (axes[0], "nonzero_evidence_fraction", "Non-zero evidence"),
        (axes[1], "multi_channel_fraction", "Multi-channel evidence"),
    ]:
        pivot = df.pivot_table(index="confidence_label", columns="dataset", values=metric, aggfunc="mean")
        pivot = pivot.reindex(["core", "inner", "outer", "uncertain"])
        x = np.arange(len(pivot.index))
        width = 0.34
        for i, ds in enumerate(["Gavin", "Krogan"]):
            vals = pivot[ds].to_numpy() if ds in pivot else np.full(len(x), np.nan)
            ax.bar(x + (i - 0.5) * width, vals, width=width, label=ds, color=["#2F8F6B", "#C9823C"][i])
        ax.set_xticks(x)
        ax.set_xticklabels([str(v).title() for v in pivot.index], rotation=15, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        ax.grid(axis="y", color="#E8ECEF", lw=0.8)
    axes[0].set_ylabel("Fraction of assignments")
    axes[1].legend(frameon=False, fontsize=8)
    _save(fig, "trace_ppi_fig14_auditability_transfer.pdf")


def label_distribution():
    audit_path = RESULTS / "cross_dataset_auditability.csv"
    if not audit_path.exists():
        return
    df = pd.read_csv(audit_path)
    df = df[df["confidence_label"].isin(["core", "inner", "outer", "uncertain"])].copy()
    if df.empty:
        return
    df["dataset"] = df["dataset"].str.title()
    total = df.groupby("dataset")["assignments"].transform("sum")
    df["assignment_fraction"] = df["assignments"] / total
    pivot = df.pivot_table(index="dataset", columns="confidence_label", values="assignment_fraction", aggfunc="sum").fillna(0)
    pivot = pivot.reindex(columns=["core", "inner", "outer", "uncertain"])
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    bottom = np.zeros(len(pivot.index))
    colors = ["#2F8F6B", "#8FC7A6", "#F3C677", "#D9DEE3"]
    for col, color in zip(pivot.columns, colors):
        vals = pivot[col].to_numpy()
        ax.bar(pivot.index, vals, bottom=bottom, label=col.title(), color=color)
        bottom += vals
    ax.set_ylim(0, 1)
    ax.set_ylabel("Assignment fraction")
    ax.set_title("Confidence-label distribution")
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    _save(fig, "trace_ppi_fig15_label_distribution_transfer.pdf")


def main():
    benchmark_comparison()
    auditability_transfer()
    label_distribution()
    print("Cross-dataset figures written to", FIG)


if __name__ == "__main__":
    main()
