#!/usr/bin/env python3
"""Final Scientific Reports figures for the TRACE-PPI revision."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "results" / "matplotlib_cache"))
sys.path.insert(0, str(ROOT / "scripts"))

from nature_plot_theme import (  # noqa: E402
    CHANNEL_COLORS,
    FIG,
    GRID,
    INK,
    LABEL_COLORS,
    MUTED,
    RESULTS,
    REV_FIG,
    TABLES,
    apply_nature_style,
    method_color,
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

apply_nature_style()
FIG.mkdir(parents=True, exist_ok=True)
REV_FIG.mkdir(parents=True, exist_ok=True)
SI_FIG = FIG / "supplementary"
SI_FIG.mkdir(parents=True, exist_ok=True)


def save(fig, *stems: Path) -> None:
    for stem in stems:
        stem = Path(stem)
        stem.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(stem.with_suffix(".pdf"))
        fig.savefig(stem.with_suffix(".svg"))
        fig.savefig(stem.with_suffix(".png"), dpi=300)
    plt.close(fig)


def rounded(ax, x, y, w, h, fc, ec, lw=1.0, rs=0.08):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.01,rounding_size={rs}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2,
    )
    ax.add_patch(p)
    return p


def arrow(ax, x1, y1, x2, y2, color=INK):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle="-|>", mutation_scale=11,
            linewidth=1.15, color=color, zorder=3,
            shrinkA=0, shrinkB=0,
        )
    )


def figure1_workflow() -> None:
    """Original four-layer concept, redrawn for Scientific Reports."""
    fig, ax = plt.subplots(figsize=(12.2, 7.15))
    ax.set_xlim(0, 12.2)
    ax.set_ylim(0, 7.15)
    ax.axis("off")

    cols = [
        (0.18, method_color("TRACE-PPI"), "1. Input", "Weighted PPI network"),
        (3.22, method_color("ClusterONE"), "2. Evidence", "Topology, semantic, GO"),
        (6.26, method_color("MCL+overlap"), "3. Modules", "MCL seed + candidate pool"),
        (9.30, method_color("TRACE-PPI"), "4. Audit output", "Assignment-level records"),
    ]
    for x, color, title, subtitle in cols:
        rounded(ax, x, 6.28, 2.72, 0.72, "#F7FBFE", color, lw=1.15, rs=0.06)
        ax.text(x + 1.36, 6.78, title, ha="center", va="center", fontsize=11, fontweight="bold", color=INK)
        ax.text(x + 1.36, 6.46, subtitle, ha="center", va="center", fontsize=8, color=MUTED)

    for x1, x2 in [(2.90, 3.22), (5.94, 6.26), (8.98, 9.30)]:
        arrow(ax, x1, 6.64, x2, 6.64, method_color("TRACE-PPI"))

    # Panel 1: network
    rounded(ax, 0.18, 0.55, 2.72, 5.55, "white", "#D0D5DA", lw=0.8, rs=0.04)
    rng = np.random.default_rng(7)
    nodes = {
        "YKR018C": (1.54, 3.55),
        "A": (0.72, 4.55),
        "B": (2.28, 4.70),
        "C": (0.62, 2.55),
        "D": (2.38, 2.40),
        "E": (1.10, 1.55),
        "F": (2.00, 1.45),
        "G": (1.54, 5.15),
    }
    edges = [("YKR018C", "A"), ("YKR018C", "B"), ("YKR018C", "C"), ("YKR018C", "D"),
             ("A", "B"), ("A", "G"), ("B", "G"), ("C", "D"), ("C", "E"), ("D", "F"), ("E", "F")]
    for u, v in edges:
        ax.plot([nodes[u][0], nodes[v][0]], [nodes[u][1], nodes[v][1]], color="#B7C4CE", lw=1.0, zorder=3)
    for name, (x, y) in nodes.items():
        if name == "YKR018C":
            ax.add_patch(Circle((x, y), 0.18, facecolor=method_color("TRACE-PPI"), edgecolor="white", lw=0.8, zorder=4))
            ax.text(x, y - 0.32, "YKR018C", ha="center", fontsize=7, color=INK, zorder=5)
        else:
            ax.add_patch(Circle((x, y), 0.11, facecolor="#8AA4B5", edgecolor="white", lw=0.6, zorder=4))
    ax.text(1.54, 0.78, "Edges are weighted; MCL ignores weights", ha="center", fontsize=7.5, color=MUTED)

    # Panel 2: evidence
    rounded(ax, 3.22, 0.55, 2.72, 5.55, "white", "#D0D5DA", lw=0.8, rs=0.04)
    ev = [
        (4.70, method_color("MCL"), "Topology", "Neighbourhood, $k$-core,\nlocal cohesion"),
        (3.25, method_color("TRACE-PPI"), "Semantic", "TF-IDF+SVD profile $e_p$\n(not Sentence-BERT)"),
        (1.80, method_color("MCL+overlap"), "GO support", "Shared non-generic\nGO terms"),
    ]
    for y, color, title, body in ev:
        rounded(ax, 3.40, y, 2.36, 1.18, "#F7FBFE", color, lw=0.95, rs=0.05)
        ax.text(4.58, y + 0.86, title, ha="center", va="center", fontsize=9, fontweight="bold")
        ax.text(4.58, y + 0.42, body, ha="center", va="center", fontsize=7.3, color=MUTED)
    ax.text(4.58, 0.78, "Channels remain separate, not fused", ha="center", fontsize=7.5, color=MUTED)

    # Panel 3: modules
    rounded(ax, 6.26, 0.55, 2.72, 5.55, "white", "#D0D5DA", lw=0.8, rs=0.04)
    rounded(ax, 6.42, 4.18, 2.40, 1.55, "#F4F8F6", method_color("MCL+overlap"), lw=1.05, rs=0.05)
    ax.text(7.62, 5.42, "Final solution seed", ha="center", fontsize=8, fontweight="bold", color=INK)
    ax.text(7.62, 5.08, "Unweighted MCL partition", ha="center", fontsize=8.5, fontweight="bold", color=method_color("MCL"))
    ax.text(7.62, 4.58, "Overlap reassignment\n+ recall-safe supplementation", ha="center", fontsize=7.3, color=MUTED)
    rounded(ax, 6.42, 1.55, 2.40, 2.40, "#FBF8F2", method_color("Score-select"), lw=0.9, rs=0.05)
    ax.text(7.62, 3.65, "Candidate pool", ha="center", fontsize=8, fontweight="bold")
    ax.text(7.62, 3.32, "(oracle / score-select only)", ha="center", fontsize=7, color=MUTED, style="italic")
    for i, lab in enumerate(["Ego neighbourhoods", "Greedy expansion", "Semantic $k$-NN", "Hybrid unions"]):
        ax.text(7.62, 2.92 - 0.28 * i, lab, ha="center", fontsize=7.4, color=INK)
    ax.text(7.62, 0.78, "Candidates do not replace the MCL seed", ha="center", fontsize=7.5, color=MUTED)

    # Panel 4: audit
    rounded(ax, 9.30, 0.55, 2.72, 5.55, "white", "#D0D5DA", lw=0.8, rs=0.04)
    fields = [
        "Protein  YKR018C",
        "Module  169",
        r"Topology  $\tau = 0.14$",
        r"Semantic  $\sigma = 0.85$",
        "Label  Inner",
        "Provenance  exported",
    ]
    ax.text(10.66, 5.55, "Exported evidence bundle", ha="center", fontsize=8.5, fontweight="bold")
    ax.text(10.66, 5.25, "Measured Gavin assignment", ha="center", fontsize=7.4, color=MUTED)
    y0 = 4.78
    for i, lab in enumerate(fields):
        y = y0 - i * 0.55
        rounded(ax, 9.48, y - 0.16, 2.36, 0.44, "#F7FBFE", method_color("TRACE-PPI"), lw=0.7, rs=0.04)
        ax.add_patch(Circle((9.72, y + 0.06), 0.07, facecolor=method_color("TRACE-PPI"), edgecolor=method_color("TRACE-PPI"), zorder=5))
        ax.text(9.90, y + 0.06, lab, ha="left", va="center", fontsize=7.5)
    ax.text(10.66, 0.78, r"Inner: not Core, and $\sigma \geq 0.25$", ha="center", fontsize=7.5, color=MUTED)

    save(
        fig,
        REV_FIG / "figure1_workflow",
        FIG / "trace_ppi_fig1_workflow",
        ROOT / "Figure_1_pipeline",
    )


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
            "coverage": float(row.get("coverage", 0.0) or 0.0),
        }
    slpa = pd.read_csv(RESULTS / "evaluation" / "slpa_gavin_grid.csv")
    slpa = slpa[slpa["threshold"] == 0.05].iloc[0]
    bench["SLPA"] = {
        "f1": float(slpa["f1"]),
        "prec": float(slpa["precision"]),
        "rec": float(slpa["recall"]),
        "size": float(slpa["mean_size"]),
        "coverage": float(slpa.get("coverage", 0.0) or 0.0),
    }
    return bench


def figure2_oracle() -> None:
    cand_path = RESULTS / "candidates" / "candidate_modules_gavin.csv"
    oracle = pd.read_csv(RESULTS / "oracle_analysis_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.15))
    cand_df = pd.read_csv(cand_path)
    counts = cand_df.groupby("source")["candidate_id"].nunique().sort_values(ascending=False)
    label_map = {
        "mcl": "MCL", "bh_ego1": "Ego-1", "bh_ego2": "Ego-2",
        "semantic_k": "Semantic kNN", "greedy_expand": "Greedy", "hybrid_mcl_ego": "Hybrid",
    }
    sources = [label_map.get(s, s) for s in counts.index]
    fracs = (counts / counts.sum()).to_numpy()
    src_colors = [method_color("MCL") if s == "MCL" else method_color("TRACE-PPI") for s in sources]
    axes[0].bar(sources, fracs, color=src_colors, width=0.72, edgecolor="white", linewidth=0.4)
    axes[0].set_ylabel("Fraction of candidate pool")
    axes[0].set_title("a  Candidate-source composition")
    plt.setp(axes[0].get_xticklabels(), rotation=18, ha="right")
    order = [
        ("best_jaccard_eq_0", "No overlap", "#999999"),
        ("best_jaccard_between_0_and_0.5", r"Partial ($0<J<0.5$)", method_color("Score-select")),
        ("best_jaccard_ge_0.5", r"Matchable ($J\geq0.5$)", method_color("TRACE-PPI")),
    ]
    cats, vals, colors = [], [], []
    for key, lab, col in order:
        row = oracle[oracle["category"] == key]
        cats.append(lab)
        vals.append(float(row["fraction"].iloc[0]))
        colors.append(col)
    axes[1].bar(cats, vals, color=colors, width=0.72, edgecolor="white", linewidth=0.4)
    axes[1].set_ylabel("Fraction of gold complexes")
    axes[1].set_title("b  Oracle ceiling")
    axes[1].set_ylim(0, 0.65)
    for x, v in enumerate(vals):
        axes[1].text(x, v + 0.015, f"{v:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    save(fig, REV_FIG / "figure2_oracle", ROOT / "Figure_2")


def figure3_gavin(bench: dict) -> None:
    methods = ["MCL", "MCL+overlap", "ClusterONE", "SLPA", "TRACE-PPI"]
    metrics = [("f1", "F1"), ("prec", "Precision"), ("rec", "Recall")]
    y = np.arange(len(methods))[::-1]
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 4.25), sharey=True)
    colors = [method_color(m) for m in methods]
    for ax, (key, title) in zip(axes, metrics):
        vals = [bench[m][key] for m in methods]
        ax.hlines(y, 0, vals, color="#D9D9D9", lw=2.0, zorder=1)
        ax.scatter(vals, y, s=72, color=colors, edgecolor="white", linewidth=0.7, zorder=3)
        for yi, v in zip(y, vals):
            ax.text(v + 0.01, yi, f"{v:.3f}", va="center", fontsize=7.5)
        ax.set_title(title, fontweight="bold")
        ax.set_xlim(0, 0.56)
        ax.set_xlabel(title)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(methods)
    for ax in axes[1:]:
        ax.tick_params(axis="y", length=0)
    fig.tight_layout(w_pad=1.0)
    save(fig, REV_FIG / "figure3_gavin", ROOT / "Figure_3")


def figure4_audit_matrix() -> None:
    methods = ["MCL", "MCL+overlap", "ClusterONE", "SLPA", "TRACE-PPI"]
    fields = [
        "Overlap output",
        "Assignment ID",
        "Topology score",
        "Semantic score",
        "GO score",
        "Membership score",
        "Confidence label",
        "Provenance",
    ]
    # 1 full, 0.5 limited, 0 absent
    grid = np.array([
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0.5, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1],
    ], dtype=float)
    wrapped = [
        "Overlap\noutput",
        "Assignment\nID",
        "Topology\nscore",
        "Semantic\nscore",
        "GO\nscore",
        "Membership\nscore",
        "Confidence\nlabel",
        "Provenance",
    ]
    fig, ax = plt.subplots(figsize=(9.6, 3.55))
    ax.set_xlim(-0.55, len(fields) - 0.45)
    ax.set_ylim(-0.85, len(methods) - 0.25)
    ax.invert_yaxis()
    for i, method in enumerate(methods):
        for j, val in enumerate(grid[i]):
            x, y = j, i
            if val >= 1:
                ax.add_patch(Circle((x, y), 0.20, facecolor=method_color(method), edgecolor=method_color(method), lw=0.8, zorder=3))
            elif val == 0.5:
                ax.add_patch(Circle((x, y), 0.20, facecolor="white", edgecolor=method_color(method), lw=1.2, zorder=3))
                ax.add_patch(Circle((x, y), 0.08, facecolor=method_color(method), edgecolor=method_color(method), zorder=4))
            else:
                ax.add_patch(Circle((x, y), 0.20, facecolor="white", edgecolor="#B0B0B0", lw=1.0, zorder=3))
    ax.set_xticks(range(len(fields)))
    ax.set_xticklabels(wrapped, fontsize=8)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods)
    ax.tick_params(length=0, pad=4)
    for spine in ax.spines.values():
        spine.set_visible(False)
    legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=method_color("TRACE-PPI"),
               markeredgecolor=method_color("TRACE-PPI"), markersize=8, label="Field present"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
               markeredgecolor=method_color("MCL+overlap"), markersize=8, label="Limited / partial"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
               markeredgecolor="#B0B0B0", markersize=8, label="Not in standard output"),
    ]
    ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=3, frameon=False)
    save(fig, REV_FIG / "figure4_audit_matrix", ROOT / "Figure_4")


def figure5_tradeoff(bench: dict) -> None:
    fig, ax = plt.subplots(figsize=(7.1, 4.9))
    order = ["MCL", "MCL+overlap", "TRACE-PPI", "Score-select", "Core-only", "Naive expansion"]
    offsets = {
        "MCL": (8, 8), "MCL+overlap": (8, -12), "TRACE-PPI": (-70, 8),
        "Score-select": (8, 8), "Core-only": (8, 8), "Naive expansion": (8, 8),
    }
    for name in order:
        d = bench[name]
        ax.scatter(d["size"], d["f1"], s=118, color=method_color(name), edgecolors="white", linewidths=0.7, zorder=3)
        ax.annotate(name, (d["size"], d["f1"]), textcoords="offset points", xytext=offsets[name], fontsize=8)
    ax.set_xlabel("Mean module size")
    ax.set_ylabel("F1 score")
    ax.set_xlim(0, 42)
    ax.set_ylim(0, 0.22)
    save(fig, REV_FIG / "figure5_tradeoff", ROOT / "Figure_5")


def figure6_cross_dataset() -> None:
    gavin = pd.read_csv(TABLES / "table1_trace_ppi_final_benchmark.csv")
    gavin["dataset"] = "Gavin"
    gavin["label"] = gavin["method"].map({
        "MCL": "MCL", "MCL_overlap_heuristic": "MCL+overlap", "ClusterONE_exact": "ClusterONE",
        "TRACE-PPI_final": "TRACE-PPI", "score_select_only_ablation": "Score-select",
    })
    krogan = pd.read_csv(RESULTS / "krogan" / "benchmark_summary.csv").rename(columns={"f1": "f1_mean"})
    krogan["dataset"] = "Krogan"
    krogan["label"] = krogan["method"].map({
        "MCL": "MCL", "MCL_overlap_heuristic": "MCL+overlap", "ClusterONE_exact": "ClusterONE",
        "SLPA_best_sensitivity": "SLPA", "TRACE-PPI_final": "TRACE-PPI",
        "score_select_only_ablation": "Score-select",
    })
    slpa = pd.read_csv(RESULTS / "evaluation" / "slpa_gavin_grid.csv")
    slpa05 = slpa[slpa["threshold"] == 0.05].iloc[0]
    extra = pd.DataFrame([{"dataset": "Gavin", "label": "SLPA", "f1_mean": float(slpa05["f1"])}])
    df = pd.concat([gavin, krogan, extra], ignore_index=True, sort=False)
    keep = ["MCL", "MCL+overlap", "ClusterONE", "SLPA", "TRACE-PPI", "Score-select"]
    fig, ax = plt.subplots(figsize=(8.6, 4.25))
    datasets = ["Gavin", "Krogan"]
    x = np.arange(len(datasets))
    width = 0.12
    offsets = np.linspace(-0.30, 0.30, len(keep))
    for off, method in zip(offsets, keep):
        vals = []
        for ds in datasets:
            sub = df[(df["dataset"] == ds) & (df["label"] == method)]
            vals.append(float(sub["f1_mean"].iloc[0]) if not sub.empty else np.nan)
        ax.bar(x + off, vals, width=width, label=method, color=method_color(method), edgecolor="white", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("F1")
    ax.legend(ncol=3, fontsize=8)
    save(fig, REV_FIG / "figure6_cross_dataset", ROOT / "Figure_7")


def figure7_labels() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.9, 4.25))
    ax = axes[0]
    tau = np.linspace(0, 0.5, 220)
    sig = np.linspace(0, 0.5, 220)
    T, S = np.meshgrid(tau, sig)
    Z = np.zeros_like(T)
    Z[(T >= 0.35) & (S >= 0.25)] = 4
    Z[((T >= 0.25) | (S >= 0.25)) & (Z < 4)] = 3
    Z[((T >= 0.12) | (S >= 0.12)) & (Z < 3)] = 2
    Z[Z == 0] = 1
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap([LABEL_COLORS["Uncertain"], LABEL_COLORS["Outer"], LABEL_COLORS["Inner"], LABEL_COLORS["Core"]])
    ax.imshow(Z, origin="lower", extent=[0, 0.5, 0, 0.5], aspect="auto", cmap=cmap, alpha=0.92, interpolation="nearest")
    for boundary in (0.12, 0.25, 0.35):
        ax.axvline(boundary, color="white", lw=0.7, ls="--")
        ax.axhline(boundary if boundary < 0.35 else 0.25, color="white", lw=0.7, ls="--")
    ax.axhline(0.12, color="white", lw=0.7, ls="--")
    ax.set_xticks([0.0, 0.12, 0.25, 0.35, 0.50])
    ax.set_yticks([0.0, 0.12, 0.25, 0.50])
    ax.set_xlabel(r"Topology support $\tau$")
    ax.set_ylabel(r"Semantic support $\sigma$")
    for x, y, t, col in [
        (0.415, 0.40, "Core", "white"),
        (0.08, 0.40, "Inner", INK),
        (0.28, 0.06, "Outer", INK),
        (0.02, 0.02, "Uncertain", INK),
    ]:
        ax.text(x, y, t, fontsize=8, fontweight="bold", color=col, ha="left", va="bottom")
    ax.set_title("a  Operational label rules")
    val = pd.read_csv(TABLES / "table5_trace_ppi_label_validation.csv")
    order = ["core", "inner", "outer", "uncertain"]
    val = val[val["confidence_label"].isin(order)].copy()
    val["confidence_label"] = pd.Categorical(val["confidence_label"], categories=order, ordered=True)
    val = val.sort_values("confidence_label")
    ax2 = axes[1]
    x = np.arange(len(val))
    colors = [LABEL_COLORS[c] for c in val["confidence_label"]]
    ax2.bar(x, val["gold_supported_assignment_fraction"], color=colors, width=0.62, edgecolor="white", linewidth=0.4)
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(v).title() for v in val["confidence_label"]])
    ax2.set_ylabel("Gold-supported fraction")
    ax2.set_title("b  Gavin assignment stratification")
    for xi, v, n in zip(x, val["gold_supported_assignment_fraction"], val["n_assignments"]):
        ax2.text(xi, v + 0.012, f"n={int(n)}", ha="center", fontsize=7.2, color=MUTED)
    fig.tight_layout(w_pad=1.8)
    save(fig, REV_FIG / "figure7_labels", ROOT / "Figure_9")


def si_core_diagnostics(bench: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.9))
    methods = ["MCL", "Core-only"]
    colors = [method_color(m) for m in methods]
    x = np.arange(2)
    sizes = [bench[m]["size"] for m in methods]
    bars = axes[0].bar(x, sizes, width=0.55, color=colors)
    for b, v in zip(bars, sizes):
        axes[0].text(b.get_x() + b.get_width() / 2, v + 0.2, f"{v:.1f}", ha="center", fontsize=8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods)
    axes[0].set_ylabel("Mean module size")
    axes[0].set_title("a")
    w = 0.35
    recall = [bench[m]["rec"] for m in methods]
    coverage = [bench[m].get("coverage", 0.0) for m in methods]
    axes[1].bar(x - w / 2, recall, width=w, label="Recall", color=colors)
    axes[1].bar(x + w / 2, coverage, width=w, label="Matched-protein coverage", color=colors, alpha=0.45)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(methods)
    axes[1].set_ylabel("Score")
    axes[1].legend(fontsize=8)
    axes[1].set_title("b")
    fig.tight_layout()
    save(fig, SI_FIG / "figureS1_core_diagnostics")


def si_channel_support() -> None:
    df = pd.read_csv(RESULTS / "cross_dataset_auditability.csv")
    df = df[df["confidence_label"].isin(["core", "inner", "outer", "uncertain"])].copy()
    df["dataset"] = df["dataset"].str.title()
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.95), sharey=True)
    ds_colors = {"Gavin": method_color("TRACE-PPI"), "Krogan": method_color("ClusterONE")}
    for ax, metric, title in [
        (axes[0], "nonzero_evidence_fraction", "a  Non-zero evidence"),
        (axes[1], "multi_channel_fraction", "b  Multi-channel evidence"),
    ]:
        pivot = df.pivot_table(index="confidence_label", columns="dataset", values=metric, aggfunc="mean")
        pivot = pivot.reindex(["core", "inner", "outer", "uncertain"])
        x = np.arange(len(pivot.index))
        width = 0.34
        for i, ds in enumerate(["Gavin", "Krogan"]):
            vals = pivot[ds].to_numpy() if ds in pivot else np.full(len(x), np.nan)
            ax.bar(x + (i - 0.5) * width, vals, width=width, label=ds, color=ds_colors[ds])
        ax.set_xticks(x)
        ax.set_xticklabels([str(v).title() for v in pivot.index])
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
    axes[0].set_ylabel("Fraction of assignments")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    save(fig, SI_FIG / "figureS2_channel_support")


def si_label_distribution() -> None:
    df = pd.read_csv(RESULTS / "cross_dataset_auditability.csv")
    df = df[df["confidence_label"].isin(["core", "inner", "outer", "uncertain"])].copy()
    df["dataset"] = df["dataset"].str.title()
    total = df.groupby("dataset")["assignments"].transform("sum")
    df["assignment_fraction"] = df["assignments"] / total
    pivot = df.pivot_table(index="dataset", columns="confidence_label", values="assignment_fraction", aggfunc="sum").fillna(0)
    counts = df.pivot_table(index="dataset", columns="confidence_label", values="assignments", aggfunc="sum").fillna(0)
    pivot = pivot.reindex(columns=["core", "inner", "outer", "uncertain"])
    fig, ax = plt.subplots(figsize=(6.8, 4.1))
    bottom = np.zeros(len(pivot.index))
    colors = [LABEL_COLORS[c] for c in pivot.columns]
    x = np.arange(len(pivot.index))
    for col, color in zip(pivot.columns, colors):
        vals = pivot[col].to_numpy()
        ax.bar(x, vals, bottom=bottom, label=str(col).title(), color=color, width=0.55)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Assignment fraction")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.14))
    for i, ds in enumerate(pivot.index):
        n_out = int(counts.loc[ds, "outer"]) if "outer" in counts else 0
        n_unc = int(counts.loc[ds, "uncertain"]) if "uncertain" in counts else 0
        ax.text(i, 1.02, f"Outer n={n_out}; Uncertain n={n_unc}", ha="center", fontsize=7, color=MUTED)
    save(fig, SI_FIG / "figureS3_label_distribution")


def si_multimembership() -> None:
    df = pd.read_csv(TABLES / "revision" / "table_multimembership.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.05), sharex=False)
    for ax, dataset, title in zip(axes, ("gavin", "krogan"), ("a  Gavin", "b  Krogan")):
        sub = df[df["dataset"] == dataset].copy()
        order = ["MCL", "MCL+overlap", "TRACE-PPI", "ClusterONE", "SLPA (t=0.05)"]
        sub["method"] = pd.Categorical(sub["method"], categories=order, ordered=True)
        sub = sub.dropna(subset=["method"]).sort_values("method")
        y = np.arange(len(sub))
        ax.scatter(sub["pct_with_gt1_module"], y, s=70, color=[method_color(m) for m in sub["method"]], zorder=3)
        for yi, mean, pct in zip(y, sub["mean_assignments_per_protein"], sub["pct_with_gt1_module"]):
            ax.text(pct + 1.2, yi, f"mean {mean:.2f}", va="center", fontsize=7.2, color=MUTED)
        ax.set_yticks(y)
        ax.set_yticklabels(["SLPA" if str(m).startswith("SLPA") else m for m in sub["method"]])
        ax.set_xlabel("Proteins with >1 module (%)")
        ax.set_title(title)
        ax.set_xlim(-2, 50)
    fig.tight_layout()
    save(fig, SI_FIG / "figureS4_multimembership")


def si_runtime() -> None:
    rt = pd.read_csv(RESULTS / "runtime_cached_uncached.csv")
    pivot = rt.pivot_table(index="method", columns="cache_state", values="runtime_sec", aggfunc="mean")
    methods = [m for m in ["TRACE-PPI", "Score-select"] if m in pivot.index]
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    x = np.arange(len(methods))
    cached = [pivot.loc[m].get("cached", np.nan) for m in methods]
    uncached = [pivot.loc[m].get("uncached_embedding", np.nan) for m in methods]
    ax.bar(x - 0.18, cached, width=0.36, color=method_color("TRACE-PPI"), label="Cached")
    ax.bar(x + 0.18, uncached, width=0.36, color=method_color("Score-select"), label="Uncached embeddings")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Runtime (s)")
    ax.legend(fontsize=8)
    save(fig, SI_FIG / "figureS5_runtime")


def si_case_studies() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.15), sharex=True)
    panels = [
        (axes[0], "a  YKR018C", ["89", "116", "169", "189", "297"],
         [0.00, 0.11, 0.14, 0.00, 0.00], [0.77, 0.69, 0.85, 0.79, 0.79]),
        (axes[1], "b  YIL161W", ["36", "169", "187", "232", "237"],
         [0.29, 0.00, 0.00, 0.00, 0.00], [0.63, 0.67, 0.62, 0.54, 0.60]),
    ]
    for ax, title, mods, topo, sem in panels:
        y = np.arange(len(mods))
        ax.barh(y - 0.16, topo, height=0.30, label="Topology", color=CHANNEL_COLORS["topology"])
        ax.barh(y + 0.16, sem, height=0.30, label="Semantic", color=CHANNEL_COLORS["semantic"])
        ax.set_yticks(y)
        ax.set_yticklabels([f"Module {m}" for m in mods])
        ax.set_xlabel("Support score")
        ax.set_xlim(0, 1.0)
        ax.set_title(title)
    axes[0].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    save(fig, SI_FIG / "figureS6_case_studies")


def main() -> None:
    bench = load_gavin_bench()
    figure1_workflow()
    figure2_oracle()
    figure3_gavin(bench)
    figure4_audit_matrix()
    figure5_tradeoff(bench)
    figure6_cross_dataset()
    figure7_labels()
    si_core_diagnostics(bench)
    si_channel_support()
    si_label_distribution()
    si_multimembership()
    si_runtime()
    si_case_studies()
    print("Wrote main and supplementary figures")


if __name__ == "__main__":
    main()
