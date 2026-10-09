#!/usr/bin/env python3
"""Journal-style figures for TRACE-PPI manuscript."""
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
from matplotlib.patches import FancyBboxPatch

FIG = ROOT / "figures"
TABLES = ROOT / "tables"
RESULTS = ROOT / "results"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "savefig.dpi": 300,
    }
)

DEFAULT_BENCH = {
    "MCL": dict(f1=0.161, prec=0.237, rec=0.121, size=5.90, runtime=0.69, bundle=0.0, coverage=0.262),
    "MCL+overlap": dict(f1=0.165, prec=0.266, rec=0.120, size=6.80, runtime=3.73, bundle=0.0, coverage=0.265),
    "TRACE-PPI": dict(f1=0.165, prec=0.266, rec=0.120, size=6.81, runtime=5.17, bundle=1.0, coverage=0.265),
    "Score-select": dict(f1=0.154, prec=0.195, rec=0.128, size=8.39, runtime=1.34, bundle=1.0, coverage=0.263),
    "Naive expansion": dict(f1=0.043, prec=0.069, rec=0.031, size=39.06, runtime=22.66, bundle=1.0, coverage=0.087),
    "Core-only": dict(f1=0.055, prec=0.238, rec=0.031, size=12.80, runtime=0.0, bundle=0.0, coverage=0.122),
}

METHOD_LABELS = {
    "MCL": "MCL",
    "MCL_overlap_heuristic": "MCL+overlap",
    "TRACE-PPI_final": "TRACE-PPI",
    "ClusterONE_exact": "ClusterONE",
    "score_select_only_ablation": "Score-select",
    "naive_expansion_negative_control": "Naive expansion",
    "core_only_ablation": "Core-only",
}

C = {
    "mcl": "#4D4D4D",
    "overlap": "#009E73",
    "trace": "#0072B2",
    "score": "#E69F00",
    "clusterone": "#D55E00",
    "naive": "#999999",
    "core": "#56B4E9",
    "slpa": "#CC79A7",
    "ink": "#222222",
    "muted": "#555555",
    "grid": "#E6E6E6",
    "paper": "#FFFFFF",
}


def load_bench():
    path = TABLES / "table1_trace_ppi_final_benchmark.csv"
    if not path.exists():
        return DEFAULT_BENCH
    df = pd.read_csv(path)
    bench = {}
    for _, row in df.iterrows():
        label = METHOD_LABELS.get(row["method"], row["method"])
        bench[label] = {
            "f1": float(row["f1_mean"]),
            "prec": float(row["precision_mean"]),
            "rec": float(row["recall_mean"]),
            "size": float(row["mean_size"]),
            "runtime": float(row.get("runtime_sec", np.nan) or 0.0),
            "bundle": float(row.get("bundle_complete", np.nan) or 0.0),
            "coverage": float(row.get("coverage", np.nan) or 0.0),
            "num_modules": float(row.get("num_modules", np.nan) or 0.0),
            "n_proteins": float(row.get("n_proteins_assigned", np.nan) or 0.0),
        }
    for name, vals in DEFAULT_BENCH.items():
        bench.setdefault(name, vals)
    return bench


BENCH = load_bench()


def save(fig, name):
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)


def fig1_workflow():
    fig, ax = plt.subplots(figsize=(11.2, 3.9))
    ax.axis("off")
    groups = [
        ("1  Inputs", ["weighted PPI graph", "GO/text profiles"]),
        ("2  Evidence layer", ["topology features", "semantic embeddings", "GO TF-IDF", "evidence nuclei"]),
        ("3  Module layer", ["MCL seed", "candidate generation", "overlap reassignment", "recall-safe supplement"]),
        ("4  Audit layer", ["confidence labels", "evidence bundles", "curator review"]),
    ]
    x0s = [0.035, 0.285, 0.545, 0.805]
    widths = [0.18, 0.205, 0.205, 0.17]
    edge_cols = ["#526A8A", "#2F8F6B", "#C9823C", "#8E5EA2"]
    for i, ((title, items), x0, w, col) in enumerate(zip(groups, x0s, widths, edge_cols)):
        ax.add_patch(
            FancyBboxPatch(
                (x0, 0.14),
                w,
                0.68,
                boxstyle="round,pad=0.012,rounding_size=0.012",
                fc="white",
                ec=col,
                lw=1.35,
            )
        )
        ax.text(x0 + w / 2, 0.72, title, ha="center", va="center", fontsize=10.5, fontweight="bold", color=col)
        for j, item in enumerate(items):
            y = 0.58 - j * 0.115
            ax.add_patch(plt.Circle((x0 + 0.030, y), 0.010, color=col, alpha=0.85))
            ax.text(x0 + 0.048, y, item, ha="left", va="center", fontsize=8.4, color=C["ink"])
        if i < len(groups) - 1:
            ax.annotate(
                "",
                xy=(x0s[i + 1] - 0.018, 0.48),
                xytext=(x0 + w + 0.016, 0.48),
                arrowprops=dict(arrowstyle="-|>", color="#6B7280", lw=1.4, shrinkA=0, shrinkB=0),
            )
    ax.text(0.035, 0.93, "TRACE-PPI workflow", ha="left", va="center", fontsize=13.5, fontweight="bold", color=C["ink"])
    ax.text(0.230, 0.93, "from weighted interactions to auditable overlapping assignments", ha="left", va="center", fontsize=10.2, color=C["muted"])
    ax.plot([0.035, 0.965], [0.875, 0.875], color="#D9DEE3", lw=0.8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.savefig(FIG / "trace_ppi_fig1_workflow.pdf", bbox_inches="tight")
    fig.savefig(FIG / "trace_ppi_fig1_workflow.png", bbox_inches="tight", dpi=600)
    fig.savefig(FIG / "trace_ppi_fig1_workflow.svg", bbox_inches="tight")
    plt.close(fig)


def fig2_benchmark():
    methods = ["MCL", "MCL+overlap", "ClusterONE", "TRACE-PPI", "Score-select", "Naive expansion", "Core-only"]
    methods = [m for m in methods if m in BENCH]
    labels = methods
    metrics = [("f1", "F1"), ("prec", "Precision"), ("rec", "Recall")]
    y = np.arange(len(methods))[::-1]
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 4.8), sharey=True)
    color_map = {
        "MCL": C["mcl"],
        "MCL+overlap": C["overlap"],
        "ClusterONE": C["clusterone"],
        "TRACE-PPI": C["trace"],
        "Score-select": C["score"],
        "Naive expansion": C["naive"],
        "Core-only": C["core"],
    }
    colors = [color_map[m] for m in methods]
    for ax, (key, title) in zip(axes, metrics):
        vals = [BENCH[m][key] for m in methods]
        ax.hlines(y, 0, vals, color=C["grid"], lw=2.5)
        ax.scatter(vals, y, s=68, color=colors, edgecolor="white", linewidth=0.8, zorder=3)
        for yi, v in zip(y, vals):
            ax.text(v + 0.008, yi, f"{v:.3f}", va="center", fontsize=7.5, color=C["ink"])
        ax.set_title(title, fontsize=10.5, fontweight="bold")
        ax.set_xlim(0, 0.31)
        ax.grid(axis="x", color=C["grid"], lw=0.8)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=8.5)
    for ax in axes[1:]:
        ax.tick_params(axis="y", length=0)
    fig.tight_layout(w_pad=1.2)
    save(fig, "trace_ppi_fig2_benchmark.pdf")


def fig3_auditability():
    methods = ["MCL", "MCL+overlap", "ClusterONE", "TRACE-PPI"]
    methods = [m for m in methods if m in BENCH]
    vals = [BENCH[m]["bundle"] for m in methods]
    fig, ax = plt.subplots(figsize=(5.1, 4.2))
    x = np.arange(len(methods))
    audit_colors = {"MCL": C["mcl"], "MCL+overlap": C["overlap"], "ClusterONE": C["clusterone"], "TRACE-PPI": C["trace"]}
    colors = [audit_colors[m] for m in methods]
    ax.vlines(x, 0, vals, color=colors, lw=5, alpha=0.22)
    ax.scatter(x, vals, color=colors, s=170, edgecolor="white", linewidth=1.2, zorder=3)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Audit-field availability")
    ax.grid(axis="y", color=C["grid"], lw=0.8)
    for xi, v in zip(x, vals):
        ax.text(xi, max(v, 0.02) + 0.05, f"{v:.2f}", ha="center", fontsize=9, color=C["ink"])
    save(fig, "trace_ppi_fig3_auditability.pdf")


def fig4_runtime():
    rt_path = RESULTS / "runtime_cached_uncached.csv"
    if rt_path.exists():
        rt = pd.read_csv(rt_path)
        pivot = rt.pivot_table(index="method", columns="cache_state", values="runtime_sec", aggfunc="mean")
        methods = [m for m in ["TRACE-PPI", "Score-select"] if m in pivot.index]
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        x = np.arange(len(methods))
        cached = [pivot.loc[m].get("cached", np.nan) for m in methods]
        uncached = [pivot.loc[m].get("uncached_embedding", np.nan) for m in methods]
        b1 = ax.bar(x - 0.18, cached, width=0.36, color=C["trace"], label="Cached")
        b2 = ax.bar(x + 0.18, uncached, width=0.36, color=C["score"], label="Uncached embeddings")
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.legend(frameon=False, fontsize=8)
        ax.set_ylabel("Runtime (s)")
        for bars in (b1, b2):
            for b in bars:
                v = b.get_height()
                if np.isfinite(v):
                    ax.text(b.get_x() + b.get_width() / 2, v + max(0.05, 0.03 * v), f"{v:.1f}", ha="center", fontsize=7.5)
    else:
        methods = ["MCL", "MCL+overlap", "TRACE-PPI", "Score-select", "Naive expansion"]
        methods = [m for m in methods if m in BENCH]
        vals = [BENCH[m]["runtime"] for m in methods]
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        bars = ax.bar(methods, vals, color=[C["mcl"], C["overlap"], C["trace"], C["score"], C["naive"]])
        ax.set_ylabel("Runtime (s)")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v:.1f}", ha="center", fontsize=8)
        plt.setp(ax.get_xticklabels(), rotation=18, ha="right")
    save(fig, "trace_ppi_fig4_runtime.pdf")


def fig5_hierarchical_labels():
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.4))
    ax = axes[0]
    # 2D schematic: topology vs semantic
    tau = np.linspace(0, 0.5, 200)
    sig = np.linspace(0, 0.5, 200)
    T, S = np.meshgrid(tau, sig)
    Z = np.zeros_like(T)
    Z[(T >= 0.35) & (S >= 0.25)] = 4  # core
    Z[((T >= 0.25) | (S >= 0.25)) & (Z < 4)] = 3
    Z[((T >= 0.12) | (S >= 0.12)) & (Z < 3)] = 2
    Z[Z == 0] = 1
    cmap = plt.matplotlib.colors.ListedColormap(["#F0F2F4", "#F3C677", "#8FC7A6", "#2F8F6B"])
    ax.imshow(Z, origin="lower", extent=[0, 0.5, 0, 0.5], aspect="auto", cmap=cmap, alpha=0.9)
    ax.set_xlabel(r"Topology support $\tau$")
    ax.set_ylabel(r"Semantic support $\sigma$")
    labels = [
        (0.40, 0.40, "Core"),
        (0.30, 0.15, "Inner"),
        (0.08, 0.30, "Outer"),
        (0.05, 0.05, "Uncertain"),
    ]
    for x, y, t in labels:
        ax.text(x, y, t, fontsize=10, fontweight="bold")
    ax.set_xlim(0, 0.5)
    ax.set_ylim(0, 0.5)
    ax.set_title("Label rules", fontsize=10.5, fontweight="bold")
    val_path = TABLES / "table5_trace_ppi_label_validation.csv"
    ax2 = axes[1]
    if val_path.exists():
        val = pd.read_csv(val_path)
        label_order = ["core", "inner", "outer", "uncertain"]
        val = val[val["confidence_label"].isin(label_order)].copy()
        val["confidence_label"] = pd.Categorical(val["confidence_label"], categories=label_order, ordered=True)
        val = val.sort_values("confidence_label")
        x = np.arange(len(val))
        ax2.bar(
            x - 0.18,
            val["gold_supported_assignment_fraction"],
            width=0.36,
            color=C["trace"],
            label="Gold-supported fraction",
        )
        ax2.bar(
            x + 0.18,
            val["membership_mean"],
            width=0.36,
            color=C["overlap"],
            alpha=0.85,
            label="Mean membership score",
        )
        ax2.set_xticks(x)
        ax2.set_xticklabels([str(v).title() for v in val["confidence_label"]])
        ax2.set_ylim(0, max(0.65, float(max(val["gold_supported_assignment_fraction"].max(), val["membership_mean"].max())) + 0.08))
        ax2.set_ylabel("Score")
        ax2.set_title("Observed assignment stratification", fontsize=10.5, fontweight="bold")
        ax2.legend(frameon=False, fontsize=8)
        ax2.grid(axis="y", color=C["grid"], lw=0.8)
    else:
        ax2.axis("off")
        ax2.text(0.5, 0.5, "Label validation table not found", ha="center", va="center")
    fig.tight_layout(w_pad=2.4)
    save(fig, "trace_ppi_fig5_hierarchical_labels.pdf")


def fig6_tradeoff():
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    order = ["MCL", "MCL+overlap", "TRACE-PPI", "Score-select", "Core-only", "Naive expansion"]
    order = [m for m in order if m in BENCH]
    colors = [C["mcl"], C["overlap"], C["trace"], C["score"], C["core"], C["naive"]]
    offsets = {
        "MCL": (8, 6),
        "MCL+overlap": (8, -12),
        "TRACE-PPI": (-55, 8),
        "Score-select": (8, 8),
        "Core-only": (8, 8),
        "Naive expansion": (8, 8),
    }
    for name, col in zip(order, colors):
        d = BENCH[name]
        ax.scatter(d["size"], d["f1"], s=120, color=col, edgecolors="white", linewidths=0.8, zorder=3)
        ox, oy = offsets[name]
        ax.annotate(
            name,
            (d["size"], d["f1"]),
            textcoords="offset points",
            xytext=(ox, oy),
            fontsize=8,
            arrowprops=dict(arrowstyle="-", color="#666", lw=0.6) if name == "Naive expansion" else None,
        )
    ax.set_xlabel("Mean module size")
    ax.set_ylabel("F1 score")
    ax.set_xlim(0, 42)
    ax.set_ylim(0, 0.22)
    ax.axvspan(15, 42, color="#fde0e0", alpha=0.25)
    save(fig, "trace_ppi_fig6_precision_recall_size.pdf")


def fig7_diagnostics():
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    methods = ["MCL", "Core-only"]
    colors = [C["mcl"], C["core"]]
    sizes = [BENCH[m]["size"] for m in methods]
    x = np.arange(len(methods))
    w = 0.35
    bars = axes[0].bar(x, sizes, width=0.55, color=colors, alpha=0.9)
    for b, v in zip(bars, sizes):
        axes[0].text(b.get_x() + b.get_width() / 2, v + 0.35, f"{v:.1f}", ha="center", fontsize=8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods)
    axes[0].set_ylabel("Mean module size")

    recall = [BENCH[m]["rec"] for m in methods]
    coverage = [BENCH[m].get("coverage", 0.0) for m in methods]
    axes[1].bar(x - w / 2, recall, width=w, label="Recall", color=colors, alpha=0.9)
    axes[1].bar(x + w / 2, coverage, width=w, label="Matched-protein coverage", color=colors, alpha=0.45)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(methods)
    axes[1].set_ylabel("Score")
    axes[1].legend(frameon=False, fontsize=8)
    plt.tight_layout()
    save(fig, "trace_ppi_fig7_diagnostics.pdf")


def fig8_recall_loss():
    stages = ["MCL recall", "Core-only\nassigned", "Core-only\nmatched recall"]
    vals = [BENCH["MCL"]["rec"], BENCH["Core-only"]["coverage"], BENCH["Core-only"]["rec"]]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(stages, vals, color=[C["mcl"], C["core"], C["core"]])
    for bar, a in zip(bars, [1.0, 0.75, 0.55]):
        bar.set_alpha(a)
    ax.set_ylim(0, 0.15)
    ax.set_ylabel("Recall (full gold)")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.3f}", ha="center", fontsize=9)
    save(fig, "trace_ppi_fig8_recall_loss.pdf")


def fig9_failure_modes():
    labels = ["TRACE-PPI", "Score-select", "Naive expansion", "Core-only"]
    f1 = [BENCH[m]["f1"] for m in labels]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    bars = ax.bar(labels, f1, color=[C["trace"], C["score"], C["naive"], C["core"]])
    ax.set_ylim(0, 0.30)
    ax.set_ylabel("F1 score")
    for b, v in zip(bars, f1):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.006, f"{v:.3f}", ha="center", fontsize=9)
    plt.setp(ax.get_xticklabels(), rotation=12, ha="right")
    save(fig, "trace_ppi_fig9_failure_modes.pdf")


def fig10_candidate_oracle():
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    cand_path = RESULTS / "candidates" / "candidate_modules_gavin.csv"
    gold_path = ROOT / "data" / "gold_standards" / "cyc2008_yeast.csv"
    oracle_recall = 0.17
    if cand_path.exists():
        cand_df = pd.read_csv(cand_path)
        counts = cand_df.groupby("source")["candidate_id"].nunique().sort_values(ascending=False)
        label_map = {
            "mcl": "MCL",
            "bh_ego1": "Ego-1",
            "bh_ego2": "Ego-2",
            "semantic_k": "Semantic\nkNN",
            "greedy_expand": "Greedy",
            "hybrid_mcl_ego": "Hybrid",
        }
        sources = [label_map.get(s, s) for s in counts.index]
        fracs = (counts / counts.sum()).to_numpy()
        if gold_path.exists():
            gold_df = pd.read_csv(gold_path)
            gold = {cid: set(g["protein_id"].astype(str)) for cid, g in gold_df.groupby("cluster_id")}
            candidates = {
                cid: set(g["protein_id"].astype(str))
                for cid, g in cand_df.groupby("candidate_id")
            }
            matched = 0
            gold_sets = [g for g in gold.values() if len(g) >= 2]
            cand_sets = [c for c in candidates.values() if len(c) >= 2]
            for g in gold_sets:
                best = max(
                    (len(g & c) / len(g | c) if (g | c) else 0.0)
                    for c in cand_sets
                )
                matched += int(best >= 0.5)
            oracle_recall = matched / len(gold_sets) if gold_sets else oracle_recall
    else:
        sources = ["MCL", "Ego", "Semantic\nkNN", "Greedy", "Hybrid"]
        fracs = [0.36, 0.18, 0.22, 0.10, 0.09]
    axes[0].bar(sources, fracs, color=C["mcl"], alpha=0.85)
    axes[0].axhline(oracle_recall, color=C["naive"], ls="--", lw=2)
    axes[0].text(
        len(sources) - 1.2,
        oracle_recall + 0.006,
        f"Oracle recall {oracle_recall:.2f}",
        fontsize=8,
        color=C["naive"],
    )
    axes[0].set_ylabel("Fraction of candidate pool")
    axes[0].set_ylim(0, max(0.45, max(fracs) + 0.08))
    plt.setp(axes[0].get_xticklabels(), rotation=15, ha="right")

    heldout_path = TABLES / "table4_trace_ppi_heldout_benchmark.csv"
    if heldout_path.exists():
        held = pd.read_csv(heldout_path)
        held["label"] = held["method"].map(METHOD_LABELS).fillna(held["method"])
        keep = held[held["label"].isin(["MCL", "MCL+overlap", "TRACE-PPI", "Score-select", "Naive expansion"])]
        x = np.arange(len(keep))
        axes[1].bar(x, keep["f1_mean"], yerr=keep["f1_sd"].fillna(0.0), color=C["trace"], alpha=0.85, capsize=3)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(keep["label"], rotation=18, ha="right")
        axes[1].set_ylabel("Held-out F1")
        axes[1].set_ylim(0, max(0.12, float((keep["f1_mean"] + keep["f1_sd"].fillna(0.0)).max()) + 0.02))
    else:
        theta = np.array([0.30, 0.34, 0.38, 0.42, 0.46])
        f1_val = np.array([0.094, 0.096, 0.097, 0.096, 0.093])
        axes[1].plot(theta, f1_val, "o-", color=C["trace"], lw=2)
        axes[1].axvline(0.38, color="#888", ls=":", lw=1.5)
        axes[1].set_xlabel("Minimum evidence score")
        axes[1].set_ylabel("Validation F1")
        axes[1].set_ylim(0.09, 0.10)
    plt.tight_layout()
    save(fig, "trace_ppi_fig10_candidate_oracle.pdf")


def fig11_ykr018c():
    mods = ["89", "116", "169", "189", "297"]
    topo = [0.00, 0.11, 0.14, 0.00, 0.00]
    sem = [0.77, 0.69, 0.85, 0.79, 0.79]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    y = np.arange(len(mods))
    ax.barh(y - 0.18, topo, height=0.32, label="Topology", color=C["mcl"])
    ax.barh(y + 0.18, sem, height=0.32, label="Semantic", color=C["overlap"])
    for yi, lab in enumerate(["inner"] * 5):
        ax.text(0.92, yi, lab, va="center", fontsize=8, color="#333")
    ax.set_yticks(y)
    ax.set_yticklabels([f"Module {m}" for m in mods])
    ax.set_xlabel("Support score")
    ax.set_xlim(0, 1.0)
    ax.legend(frameon=False, loc="lower right")
    save(fig, "trace_ppi_fig11_ykr018c_case.pdf")


def fig12_yil161w():
    mods = ["36", "169", "187", "232", "237"]
    topo = [0.29, 0.00, 0.00, 0.00, 0.00]
    sem = [0.63, 0.67, 0.62, 0.54, 0.60]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    y = np.arange(len(mods))
    ax.barh(y - 0.18, topo, height=0.32, label="Topology", color=C["mcl"])
    ax.barh(y + 0.18, sem, height=0.32, label="Semantic", color=C["overlap"])
    ax.set_yticks(y)
    ax.set_yticklabels([f"Module {m}" for m in mods])
    ax.set_xlabel("Support score")
    ax.set_xlim(0, 1.0)
    ax.legend(frameon=False)
    save(fig, "trace_ppi_fig12_yil161w_case.pdf")


def main():
    fig1_workflow()
    fig2_benchmark()
    fig3_auditability()
    fig4_runtime()
    fig5_hierarchical_labels()
    fig6_tradeoff()
    fig7_diagnostics()
    fig8_recall_loss()
    fig9_failure_modes()
    fig10_candidate_oracle()
    fig11_ykr018c()
    fig12_yil161w()
    # Legacy names for backward compatibility
    import shutil
    for a, b in [
        ("trace_ppi_fig3_auditability.pdf", "trace_ppi_fig5_auditability.pdf"),
        ("trace_ppi_fig4_runtime.pdf", "trace_ppi_fig6_runtime.pdf"),
        ("trace_ppi_fig5_hierarchical_labels.pdf", "trace_ppi_fig7_hierarchical_labels.pdf"),
        ("trace_ppi_fig6_precision_recall_size.pdf", "trace_ppi_fig3_precision_recall_size.pdf"),
        ("trace_ppi_fig11_ykr018c_case.pdf", "trace_ppi_fig8_ykr018c_case.pdf"),
        ("trace_ppi_fig12_yil161w_case.pdf", "trace_ppi_fig9_yil161w_case.pdf"),
        ("trace_ppi_fig9_failure_modes.pdf", "trace_ppi_fig10_failure_modes.pdf"),
    ]:
        shutil.copy(FIG / a, FIG / b)
    print("Figures written to", FIG)


if __name__ == "__main__":
    main()
