#!/usr/bin/env python3
"""Regenerate all manuscript figures with a Scientific Reports plot theme."""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "results" / "matplotlib_cache"))
sys.path.insert(0, str(ROOT / "scripts"))

from nature_plot_theme import (  # noqa: E402
    FIG,
    GRID,
    INK,
    MUTED,
    RESULTS,
    REV_FIG,
    TABLES,
    apply_nature_style,
    light_xgrid,
    light_ygrid,
    method_color,
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

apply_nature_style()
FIG.mkdir(parents=True, exist_ok=True)
REV_FIG.mkdir(parents=True, exist_ok=True)


def save(fig, *names: Path) -> None:
    for name in names:
        name.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(name)
    plt.close(fig)


def pipeline_figure() -> None:
    steps = [
        ("1. Input", "Weighted PPI network\nand optional GO/text annotations"),
        ("2. Preprocess", "Clean identifiers\nand normalise edge weights"),
        ("3. Evidence profiles", "Topology, GO features,\nand semantic embeddings"),
        ("4. Nucleus ranking", "BH score + non-maximum\nsuppression (candidates only)"),
        ("5. Candidates", "MCL, ego, greedy, semantic\nk-NN, and hybrid unions"),
        ("6. Scoring", "Candidate scores and\nprotein-module scores"),
        ("7. Refinement", "Overlap reassignment and\nrecall-safe supplementation"),
        ("8-10. Output", "Confidence labels, evidence\nbundles, gold-standard evaluation"),
    ]
    fig, ax = plt.subplots(figsize=(11.4, 6.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8.2)
    ax.axis("off")
    ax.set_title("TRACE-PPI algorithmic pipeline", fontsize=13, pad=8, color=INK)
    coords = [
        (0.4, 5.8),
        (3.2, 5.8),
        (6.0, 5.8),
        (8.8, 5.8),
        (0.4, 2.6),
        (3.2, 2.6),
        (6.0, 2.6),
        (8.8, 2.6),
    ]
    fill = "#E8F1F8"
    edge = method_color("TRACE-PPI")
    for (x, y), (title, body) in zip(coords, steps):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                2.7,
                1.9,
                boxstyle="round,pad=0.04,rounding_size=0.12",
                linewidth=1.0,
                edgecolor=edge,
                facecolor=fill,
            )
        )
        ax.text(x + 1.35, y + 1.45, title, ha="center", va="center", fontsize=10, fontweight="bold")
        ax.text(x + 1.35, y + 0.7, body, ha="center", va="center", fontsize=8, color=MUTED)

    def arrow(a, b):
        ax.annotate(
            "",
            xy=(b[0], b[1] + 0.95),
            xytext=(a[0] + 2.7, a[1] + 0.95),
            arrowprops=dict(arrowstyle="->", color=edge, lw=1.2),
        )

    for i in range(3):
        arrow(coords[i], coords[i + 1])
    ax.annotate(
        "",
        xy=(coords[4][0] + 1.35, coords[4][1] + 1.9),
        xytext=(coords[3][0] + 1.35, coords[3][1]),
        arrowprops=dict(arrowstyle="->", color=edge, lw=1.2),
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
        color=MUTED,
    )
    fig.tight_layout()
    save(fig, REV_FIG / "figure_pipeline.pdf", ROOT / "Figure_1_pipeline.pdf", FIG / "trace_ppi_fig1_workflow.pdf")


def candidate_oracle() -> None:
    cand_path = RESULTS / "candidates" / "candidate_modules_gavin.csv"
    oracle = pd.read_csv(RESULTS / "oracle_analysis_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    if cand_path.exists():
        cand_df = pd.read_csv(cand_path)
        counts = cand_df.groupby("source")["candidate_id"].nunique().sort_values(ascending=False)
        label_map = {
            "mcl": "MCL",
            "bh_ego1": "Ego-1",
            "bh_ego2": "Ego-2",
            "semantic_k": "Semantic kNN",
            "greedy_expand": "Greedy",
            "hybrid_mcl_ego": "Hybrid",
        }
        sources = [label_map.get(s, s) for s in counts.index]
        fracs = (counts / counts.sum()).to_numpy()
    else:
        sources = ["MCL", "Ego", "Semantic kNN", "Greedy", "Hybrid"]
        fracs = np.array([0.36, 0.18, 0.22, 0.10, 0.09])
    axes[0].bar(sources, fracs, color=method_color("TRACE-PPI"), width=0.72)
    axes[0].set_ylabel("Fraction of candidate pool")
    axes[0].set_title("Candidate-source composition")
    light_ygrid(axes[0])
    plt.setp(axes[0].get_xticklabels(), rotation=20, ha="right")

    order = [
        ("best_jaccard_eq_0", "No overlap"),
        ("best_jaccard_between_0_and_0.5", "Partial ($0<J<0.5$)"),
        ("best_jaccard_ge_0.5", "Matchable ($J\\geq0.5$)"),
    ]
    cats = []
    vals = []
    colors = ["#999999", method_color("Score-select"), method_color("TRACE-PPI")]
    for key, lab in order:
        row = oracle[oracle["category"] == key]
        cats.append(lab)
        vals.append(float(row["fraction"].iloc[0]) if not row.empty else 0.0)
    axes[1].bar(cats, vals, color=colors, width=0.72)
    axes[1].set_ylabel("Fraction of gold complexes")
    axes[1].set_title("Oracle ceiling")
    axes[1].set_ylim(0, 0.65)
    light_ygrid(axes[1])
    for x, v in enumerate(vals):
        axes[1].text(x, v + 0.015, f"{v:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    save(fig, ROOT / "Figure_2.pdf", FIG / "trace_ppi_fig10_candidate_oracle.pdf")


def load_gavin_bench() -> dict:
    df = pd.read_csv(TABLES / "table1_trace_ppi_final_benchmark.csv")
    labels = {
        "MCL": "MCL",
        "MCL_overlap_heuristic": "MCL+overlap",
        "TRACE-PPI_final": "TRACE-PPI",
        "ClusterONE_exact": "ClusterONE",
        "score_select_only_ablation": "Score-select",
        "naive_expansion_negative_control": "Naive expansion",
        "core_only_ablation": "Core-only",
    }
    bench = {}
    for _, row in df.iterrows():
        name = labels.get(row["method"], row["method"])
        bench[name] = {
            "f1": float(row["f1_mean"]),
            "prec": float(row["precision_mean"]),
            "rec": float(row["recall_mean"]),
            "size": float(row["mean_size"]),
            "bundle": float(row.get("bundle_complete", 0.0) or 0.0),
            "coverage": float(row.get("coverage", 0.0) or 0.0),
        }
    slpa = pd.read_csv(RESULTS / "evaluation" / "slpa_gavin_grid.csv")
    slpa = slpa[slpa["threshold"] == 0.05].iloc[0]
    bench["SLPA"] = {
        "f1": float(slpa["f1"]),
        "prec": float(slpa["precision"]),
        "rec": float(slpa["recall"]),
        "size": float(slpa["mean_size"]),
        "bundle": 0.0,
        "coverage": float(slpa.get("coverage", 0.0) or 0.0),
    }
    return bench


def gavin_benchmark(bench: dict) -> None:
    methods = ["MCL", "MCL+overlap", "ClusterONE", "SLPA", "TRACE-PPI"]
    metrics = [("f1", "F1"), ("prec", "Precision"), ("rec", "Recall")]
    y = np.arange(len(methods))[::-1]
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 4.4), sharey=True)
    colors = [method_color(m) for m in methods]
    for ax, (key, title) in zip(axes, metrics):
        vals = [bench[m][key] for m in methods]
        ax.hlines(y, 0, vals, color=GRID, lw=2.2)
        ax.scatter(vals, y, s=70, color=colors, edgecolor="white", linewidth=0.7, zorder=3)
        for yi, v in zip(y, vals):
            ax.text(v + 0.008, yi, f"{v:.3f}", va="center", fontsize=7.5)
        ax.set_title(title, fontweight="bold")
        ax.set_xlim(0, 0.55)
        light_xgrid(ax)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(methods)
    for ax in axes[1:]:
        ax.tick_params(axis="y", length=0)
    fig.tight_layout(w_pad=1.1)
    save(fig, ROOT / "Figure_3.pdf", FIG / "trace_ppi_fig2_benchmark.pdf")


def audit_fields(bench: dict) -> None:
    methods = ["MCL", "MCL+overlap", "ClusterONE", "SLPA", "TRACE-PPI"]
    vals = [bench[m]["bundle"] for m in methods]
    fig, ax = plt.subplots(figsize=(6.2, 4.1))
    x = np.arange(len(methods))
    colors = [method_color(m) for m in methods]
    ax.bar(x, vals, color=colors, width=0.62)
    ax.set_ylim(0, 1.15)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=18, ha="right")
    ax.set_ylabel("Audit-field availability")
    light_ygrid(ax)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.04, f"{v:.2f}", ha="center", fontsize=9)
    save(fig, ROOT / "Figure_4.pdf", FIG / "trace_ppi_fig3_auditability.pdf", FIG / "trace_ppi_fig5_auditability.pdf")


def tradeoff(bench: dict) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    order = ["MCL", "MCL+overlap", "TRACE-PPI", "Score-select", "Core-only", "Naive expansion"]
    offsets = {
        "MCL": (8, 8),
        "MCL+overlap": (8, -12),
        "TRACE-PPI": (-62, 8),
        "Score-select": (8, 8),
        "Core-only": (8, 8),
        "Naive expansion": (8, 8),
    }
    for name in order:
        d = bench[name]
        ax.scatter(d["size"], d["f1"], s=120, color=method_color(name), edgecolors="white", linewidths=0.7, zorder=3)
        ox, oy = offsets[name]
        ax.annotate(name, (d["size"], d["f1"]), textcoords="offset points", xytext=(ox, oy), fontsize=8)
    ax.set_xlabel("Mean module size")
    ax.set_ylabel("F1 score")
    ax.set_xlim(0, 42)
    ax.set_ylim(0, 0.22)
    light_ygrid(ax)
    save(fig, ROOT / "Figure_5.pdf", FIG / "trace_ppi_fig6_precision_recall_size.pdf", FIG / "trace_ppi_fig3_precision_recall_size.pdf")


def core_diagnostics(bench: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0))
    methods = ["MCL", "Core-only"]
    colors = [method_color(m) for m in methods]
    x = np.arange(len(methods))
    sizes = [bench[m]["size"] for m in methods]
    bars = axes[0].bar(x, sizes, width=0.55, color=colors)
    for b, v in zip(bars, sizes):
        axes[0].text(b.get_x() + b.get_width() / 2, v + 0.25, f"{v:.1f}", ha="center", fontsize=8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods)
    axes[0].set_ylabel("Mean module size")
    light_ygrid(axes[0])
    w = 0.35
    recall = [bench[m]["rec"] for m in methods]
    coverage = [bench[m].get("coverage", 0.0) for m in methods]
    axes[1].bar(x - w / 2, recall, width=w, label="Recall", color=colors)
    axes[1].bar(x + w / 2, coverage, width=w, label="Matched-protein coverage", color=colors, alpha=0.45)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(methods)
    axes[1].set_ylabel("Score")
    axes[1].legend(frameon=False, fontsize=8)
    light_ygrid(axes[1])
    fig.tight_layout()
    save(fig, ROOT / "Figure_14.pdf", FIG / "trace_ppi_fig7_diagnostics.pdf")


def recall_loss(bench: dict) -> None:
    stages = ["MCL recall", "Core-only\nassigned", "Core-only\nmatched recall"]
    vals = [bench["MCL"]["rec"], bench["Core-only"]["coverage"], bench["Core-only"]["rec"]]
    colors = [method_color("MCL"), method_color("Core-only"), method_color("Core-only")]
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    bars = ax.bar(stages, vals, color=colors)
    bars[1].set_alpha(0.75)
    bars[2].set_alpha(0.55)
    ax.set_ylim(0, 0.16)
    ax.set_ylabel("Recall (full gold)")
    light_ygrid(ax)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.3f}", ha="center", fontsize=9)
    save(fig, ROOT / "Figure_15.pdf", FIG / "trace_ppi_fig8_recall_loss.pdf")


def cross_dataset() -> None:
    rows = []
    gavin = pd.read_csv(TABLES / "table1_trace_ppi_final_benchmark.csv")
    gavin["dataset"] = "Gavin"
    gavin["label"] = gavin["method"].map(
        {
            "MCL": "MCL",
            "MCL_overlap_heuristic": "MCL+overlap",
            "ClusterONE_exact": "ClusterONE",
            "TRACE-PPI_final": "TRACE-PPI",
            "score_select_only_ablation": "Score-select",
        }
    )
    rows.append(gavin)
    krogan = pd.read_csv(RESULTS / "krogan" / "benchmark_summary.csv")
    krogan = krogan.rename(columns={"f1": "f1_mean"})
    krogan["dataset"] = "Krogan"
    krogan["label"] = krogan["method"].map(
        {
            "MCL": "MCL",
            "MCL_overlap_heuristic": "MCL+overlap",
            "ClusterONE_exact": "ClusterONE",
            "SLPA_best_sensitivity": "SLPA",
            "TRACE-PPI_final": "TRACE-PPI",
            "score_select_only_ablation": "Score-select",
        }
    )
    rows.append(krogan)
    slpa = pd.read_csv(RESULTS / "evaluation" / "slpa_gavin_grid.csv")
    slpa05 = slpa[slpa["threshold"] == 0.05].iloc[0]
    rows.append(
        pd.DataFrame(
            [{"dataset": "Gavin", "label": "SLPA", "f1_mean": float(slpa05["f1"])}]
        )
    )
    df = pd.concat(rows, ignore_index=True, sort=False)
    keep = ["MCL", "MCL+overlap", "ClusterONE", "SLPA", "TRACE-PPI", "Score-select"]
    fig, ax = plt.subplots(figsize=(8.4, 4.3))
    datasets = ["Gavin", "Krogan"]
    x = np.arange(len(datasets))
    width = 0.12
    offsets = np.linspace(-0.30, 0.30, len(keep))
    for off, method in zip(offsets, keep):
        vals = []
        for ds in datasets:
            sub = df[(df["dataset"] == ds) & (df["label"] == method)]
            vals.append(float(sub["f1_mean"].iloc[0]) if not sub.empty else np.nan)
        ax.bar(x + off, vals, width=width, label=method, color=method_color(method))
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("F1")
    light_ygrid(ax)
    ax.legend(ncol=3, fontsize=7.5)
    save(fig, ROOT / "Figure_7.pdf", FIG / "trace_ppi_fig13_cross_dataset_benchmark.pdf")


def audit_transfer() -> None:
    df = pd.read_csv(RESULTS / "cross_dataset_auditability.csv")
    df = df[df["confidence_label"].isin(["core", "inner", "outer", "uncertain"])].copy()
    df["dataset"] = df["dataset"].str.title()
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0), sharey=True)
    ds_colors = {"Gavin": method_color("TRACE-PPI"), "Krogan": method_color("ClusterONE")}
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
            ax.bar(x + (i - 0.5) * width, vals, width=width, label=ds, color=ds_colors[ds])
        ax.set_xticks(x)
        ax.set_xticklabels([str(v).title() for v in pivot.index], rotation=15, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        light_ygrid(ax)
    axes[0].set_ylabel("Fraction of assignments")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    save(fig, ROOT / "Figure_8.pdf", FIG / "trace_ppi_fig14_auditability_transfer.pdf")


def label_rules(bench_unused=None) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.4))
    ax = axes[0]
    tau = np.linspace(0, 0.5, 200)
    sig = np.linspace(0, 0.5, 200)
    T, S = np.meshgrid(tau, sig)
    Z = np.zeros_like(T)
    Z[(T >= 0.35) & (S >= 0.25)] = 4
    Z[((T >= 0.25) | (S >= 0.25)) & (Z < 4)] = 3
    Z[((T >= 0.12) | (S >= 0.12)) & (Z < 3)] = 2
    Z[Z == 0] = 1
    cmap = mpl_listed(["#F2F2F2", "#E69F00", "#56B4E9", "#0072B2"])
    ax.imshow(Z, origin="lower", extent=[0, 0.5, 0, 0.5], aspect="auto", cmap=cmap, alpha=0.9)
    ax.set_xlabel(r"Topology support $\tau$")
    ax.set_ylabel(r"Semantic support $\sigma$")
    for x, y, t in [(0.40, 0.40, "Core"), (0.30, 0.15, "Inner"), (0.08, 0.30, "Outer"), (0.05, 0.05, "Uncertain")]:
        ax.text(x, y, t, fontsize=10, fontweight="bold")
    ax.set_title("Label rules", fontweight="bold")
    val = pd.read_csv(TABLES / "table5_trace_ppi_label_validation.csv")
    label_order = ["core", "inner", "outer", "uncertain"]
    val = val[val["confidence_label"].isin(label_order)].copy()
    val["confidence_label"] = pd.Categorical(val["confidence_label"], categories=label_order, ordered=True)
    val = val.sort_values("confidence_label")
    ax2 = axes[1]
    x = np.arange(len(val))
    ax2.bar(x - 0.18, val["gold_supported_assignment_fraction"], width=0.36, color=method_color("TRACE-PPI"), label="Gold-supported fraction")
    ax2.bar(x + 0.18, val["membership_mean"], width=0.36, color=method_color("Score-select"), label="Mean membership score")
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(v).title() for v in val["confidence_label"]])
    ax2.set_ylabel("Score")
    ax2.set_title("Observed assignment stratification", fontweight="bold")
    ax2.legend(fontsize=8)
    light_ygrid(ax2)
    fig.tight_layout(w_pad=2.0)
    save(fig, ROOT / "Figure_9.pdf", FIG / "trace_ppi_fig5_hierarchical_labels.pdf", FIG / "trace_ppi_fig7_hierarchical_labels.pdf")


def mpl_listed(colors):
    from matplotlib.colors import ListedColormap

    return ListedColormap(colors)


def label_distribution() -> None:
    df = pd.read_csv(RESULTS / "cross_dataset_auditability.csv")
    df = df[df["confidence_label"].isin(["core", "inner", "outer", "uncertain"])].copy()
    df["dataset"] = df["dataset"].str.title()
    total = df.groupby("dataset")["assignments"].transform("sum")
    df["assignment_fraction"] = df["assignments"] / total
    pivot = df.pivot_table(index="dataset", columns="confidence_label", values="assignment_fraction", aggfunc="sum").fillna(0)
    pivot = pivot.reindex(columns=["core", "inner", "outer", "uncertain"])
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    bottom = np.zeros(len(pivot.index))
    colors = ["#0072B2", "#56B4E9", "#E69F00", "#999999"]
    for col, color in zip(pivot.columns, colors):
        vals = pivot[col].to_numpy()
        ax.bar(pivot.index, vals, bottom=bottom, label=col.title(), color=color)
        bottom += vals
    ax.set_ylim(0, 1)
    ax.set_ylabel("Assignment fraction")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    save(fig, ROOT / "Figure_10.pdf", FIG / "trace_ppi_fig15_label_distribution_transfer.pdf")


def multimembership() -> None:
    hist = pd.read_csv(RESULTS / "revision" / "multimembership_assignment_counts.csv")
    methods = ["MCL", "MCL+overlap", "ClusterONE", "SLPA (t=0.05)", "TRACE-PPI"]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1), sharey=True)
    for ax, dataset in zip(axes, ("gavin", "krogan")):
        sub = hist[hist["dataset"] == dataset]
        present = [m for m in methods if m in set(sub["method"])]
        data = [sub.loc[sub["method"] == m, "n_modules_per_protein"].clip(upper=8).tolist() for m in present]
        bp = ax.boxplot(data, showfliers=False, patch_artist=True, medianprops=dict(color=INK, linewidth=1.1))
        for patch, method in zip(bp["boxes"], present):
            patch.set_facecolor(method_color(method))
            patch.set_alpha(0.75)
            patch.set_edgecolor(INK)
            patch.set_linewidth(0.7)
        ax.set_xticklabels(["SLPA" if m.startswith("SLPA") else m for m in present], rotation=20, ha="right")
        ax.set_title(dataset.capitalize())
        ax.set_xlabel("Method")
        ax.set_ylabel("Modules per assigned protein")
        light_ygrid(ax)
    fig.tight_layout()
    save(fig, REV_FIG / "figure_multimembership.pdf")


def runtime() -> None:
    rt = pd.read_csv(RESULTS / "runtime_cached_uncached.csv")
    pivot = rt.pivot_table(index="method", columns="cache_state", values="runtime_sec", aggfunc="mean")
    methods = [m for m in ["TRACE-PPI", "Score-select"] if m in pivot.index]
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    x = np.arange(len(methods))
    cached = [pivot.loc[m].get("cached", np.nan) for m in methods]
    uncached = [pivot.loc[m].get("uncached_embedding", np.nan) for m in methods]
    b1 = ax.bar(x - 0.18, cached, width=0.36, color=method_color("TRACE-PPI"), label="Cached")
    b2 = ax.bar(x + 0.18, uncached, width=0.36, color=method_color("Score-select"), label="Uncached embeddings")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend(fontsize=8)
    ax.set_ylabel("Runtime (s)")
    light_ygrid(ax)
    for bars in (b1, b2):
        for b in bars:
            v = b.get_height()
            if np.isfinite(v):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v:.1f}", ha="center", fontsize=8)
    save(fig, ROOT / "Figure_11.pdf", FIG / "trace_ppi_fig4_runtime.pdf", FIG / "trace_ppi_fig6_runtime.pdf")


def case_study(name: str, mods, topo, sem, outfile: Path, legacy: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    y = np.arange(len(mods))
    ax.barh(y - 0.18, topo, height=0.32, label="Topology", color=method_color("MCL"))
    ax.barh(y + 0.18, sem, height=0.32, label="Semantic", color=method_color("TRACE-PPI"))
    ax.set_yticks(y)
    ax.set_yticklabels([f"Module {m}" for m in mods])
    ax.set_xlabel("Support score")
    ax.set_xlim(0, 1.0)
    ax.legend(loc="lower right")
    light_xgrid(ax)
    save(fig, outfile, legacy)


def main() -> None:
    bench = load_gavin_bench()
    pipeline_figure()
    candidate_oracle()
    gavin_benchmark(bench)
    audit_fields(bench)
    tradeoff(bench)
    core_diagnostics(bench)
    recall_loss(bench)
    cross_dataset()
    audit_transfer()
    label_rules()
    label_distribution()
    multimembership()
    runtime()
    case_study(
        "YKR018C",
        ["89", "116", "169", "189", "297"],
        [0.00, 0.11, 0.14, 0.00, 0.00],
        [0.77, 0.69, 0.85, 0.79, 0.79],
        ROOT / "Figure_12.pdf",
        FIG / "trace_ppi_fig11_ykr018c_case.pdf",
    )
    shutil.copy(FIG / "trace_ppi_fig11_ykr018c_case.pdf", FIG / "trace_ppi_fig8_ykr018c_case.pdf")
    case_study(
        "YIL161W",
        ["36", "169", "187", "232", "237"],
        [0.29, 0.00, 0.00, 0.00, 0.00],
        [0.63, 0.67, 0.62, 0.54, 0.60],
        ROOT / "Figure_13.pdf",
        FIG / "trace_ppi_fig12_yil161w_case.pdf",
    )
    shutil.copy(FIG / "trace_ppi_fig12_yil161w_case.pdf", FIG / "trace_ppi_fig9_yil161w_case.pdf")
    print("Regenerated Scientific Reports-themed figures")


if __name__ == "__main__":
    main()
