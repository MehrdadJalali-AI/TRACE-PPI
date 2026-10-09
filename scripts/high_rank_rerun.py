#!/usr/bin/env python3
"""High-rank readiness rerun for TRACE-PPI.

This script is intentionally conservative: it runs only datasets and baselines
that are available in the local workspace, and writes explicit feasibility
tables for everything requested but not runnable.
"""
from __future__ import annotations

import gzip
import json
import os
import platform
import random
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "results" / "matplotlib_cache"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, str(ROOT))

from trace_ppi import baselines
from trace_ppi.cosmos_v2_runner import run_v2
from trace_ppi.evidence_metrics import compute_all_evidence_metrics
from trace_ppi.graph_io import filter_go, load_gavin
from trace_ppi.reuse import GOLoader, load_gold_standard_csv, precision_recall_f1_mmr
from trace_ppi.v2_membership import modules_to_membership

RESULTS = ROOT / "results" / "high_rank_rerun"
FIGURES = ROOT / "figures" / "high_rank_rerun"
TABLES = ROOT / "tables" / "high_rank_rerun"
MANUSCRIPT = ROOT / "manuscript" / "high_rank_version"
REPORTS = ROOT / "reports"
DATA = ROOT / "data"
SEEDS = [42, 43, 44, 45, 46]
ORF_RE = re.compile(r"^Y[A-P][LR][0-9]{3}[CW](?:-[A-Z])?$|^Q[0-9]{4}$")


def ensure_dirs() -> None:
    for p in (RESULTS, FIGURES, TABLES, MANUSCRIPT, REPORTS):
        p.mkdir(parents=True, exist_ok=True)


def load_gold() -> Dict[int, Set[str]]:
    return load_gold_standard_csv(str(DATA / "gold_standards" / "cyc2008_yeast.csv"))


def load_go_for_nodes(nodes: Iterable[str]) -> Dict[str, Set[str]]:
    go = GOLoader().load_from_gaf(str(DATA / "GO.txt"), taxid=559292, use_symbol=True)
    nodes = {str(n) for n in nodes}
    return filter_go({p: terms for p, terms in go.items() if p in nodes})


def load_alias_map(path: Path) -> Dict[str, str]:
    alias: Dict[str, str] = {}
    if not path.exists():
        return alias
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            sid, a = parts[0], parts[1]
            if ORF_RE.match(a):
                alias.setdefault(sid, a)
    return alias


def load_string_yeast(threshold: int = 700) -> Tuple[nx.Graph, Dict[str, Set[str]], dict]:
    path = DATA / "4932.protein.links.detailed.v11.5.txt"
    alias = load_alias_map(DATA / "cache" / "4932.protein.aliases.v11.5.txt.gz")
    stats = {"source_path": str(path), "input_rows": 0, "mapped_rows": 0, "dropped_unmapped_rows": 0}
    if not path.exists():
        return nx.Graph(), {}, {**stats, "available": False, "reason": "STRING file missing"}
    df = pd.read_csv(path, sep=r"\s+", usecols=["protein1", "protein2", "combined_score"])
    stats["input_rows"] = int(len(df))
    df = df[df["combined_score"] >= threshold].copy()
    g = nx.Graph()
    for r in df.itertuples(index=False):
        p1 = alias.get(r.protein1, str(r.protein1).replace("4932.", ""))
        p2 = alias.get(r.protein2, str(r.protein2).replace("4932.", ""))
        if not (ORF_RE.match(p1) and ORF_RE.match(p2)) or p1 == p2:
            stats["dropped_unmapped_rows"] += 1
            continue
        g.add_edge(p1, p2, weight=float(r.combined_score) / 1000.0)
        stats["mapped_rows"] += 1
    go = load_go_for_nodes(g.nodes())
    return g, go, {**stats, "available": True, "threshold": threshold}


def load_biogrid_yeast() -> Tuple[nx.Graph, Dict[str, Set[str]], dict]:
    path = DATA / "biogrid_scerivise.tab3.txt"
    stats = {"source_path": str(path), "input_rows": 0, "mapped_rows": 0, "dropped_unmapped_rows": 0}
    if not path.exists():
        return nx.Graph(), {}, {**stats, "available": False, "reason": "BioGRID TAB3 file missing"}
    usecols = [
        "Systematic Name Interactor A",
        "Systematic Name Interactor B",
        "Experimental System Type",
        "Organism ID Interactor A",
        "Organism ID Interactor B",
    ]
    df = pd.read_csv(path, sep="\t", usecols=usecols, dtype=str)
    stats["input_rows"] = int(len(df))
    df = df[
        (df["Experimental System Type"].str.lower() == "physical")
        & (df["Organism ID Interactor A"] == "559292")
        & (df["Organism ID Interactor B"] == "559292")
    ]
    g = nx.Graph()
    for r in df.itertuples(index=False):
        p1, p2 = str(r[0]), str(r[1])
        if not (ORF_RE.match(p1) and ORF_RE.match(p2)) or p1 == p2:
            stats["dropped_unmapped_rows"] += 1
            continue
        if g.has_edge(p1, p2):
            g[p1][p2]["weight"] += 1.0
        else:
            g.add_edge(p1, p2, weight=1.0)
        stats["mapped_rows"] += 1
    max_w = max((d["weight"] for _, _, d in g.edges(data=True)), default=1.0)
    for _, _, d in g.edges(data=True):
        d["weight"] = d["weight"] / max_w
    go = load_go_for_nodes(g.nodes())
    return g, go, {**stats, "available": True}


def dataset_inventory(gold: Dict[int, Set[str]]) -> Tuple[dict, pd.DataFrame]:
    datasets = {}
    g, go = load_gavin()
    datasets["gavin_yeast"] = (g, filter_go(go), {"available": True, "source_path": str(DATA / "gavin2006_socioaffinities_rescaled.txt")})
    sg, sgo, sstats = load_string_yeast()
    if sstats.get("available") and sg.number_of_edges() > 0:
        datasets["string_yeast"] = (sg, sgo, sstats)
    bg, bgo, bstats = load_biogrid_yeast()
    if bstats.get("available") and bg.number_of_edges() > 0:
        datasets["biogrid_yeast"] = (bg, bgo, bstats)

    gold_proteins = set().union(*gold.values())
    rows = []
    for name, (graph, go, stats) in datasets.items():
        nodes = set(graph.nodes())
        rows.append(
            {
                "dataset": name,
                "available": True,
                "source_path": stats.get("source_path", ""),
                "input_rows": stats.get("input_rows", graph.number_of_edges()),
                "mapped_edges": graph.number_of_edges(),
                "dropped_unmapped_rows": stats.get("dropped_unmapped_rows", 0),
                "nodes": graph.number_of_nodes(),
                "nodes_with_go": len(go),
                "gold_protein_coverage": len(nodes & gold_proteins) / len(gold_proteins),
                "notes": "Runnable local yeast network with CYC2008 gold mapping.",
            }
        )
    missing = [
        ("krogan_yeast", "No separate Krogan edge list found in this TRACE-PPI workspace or linked local data."),
        ("human_bioplex", "No local BioPlex/CORUM-compatible human PPI and no human gold mapping available."),
        ("human_corum", "No local human CORUM-compatible PPI/gold mapping available."),
    ]
    for name, note in missing:
        rows.append(
            {
                "dataset": name,
                "available": False,
                "source_path": "",
                "input_rows": 0,
                "mapped_edges": 0,
                "dropped_unmapped_rows": 0,
                "nodes": 0,
                "nodes_with_go": 0,
                "gold_protein_coverage": 0.0,
                "notes": note,
            }
        )
    return datasets, pd.DataFrame(rows)


def relabel_communities(comms: Iterable[Set[str]]) -> Dict[int, Set[str]]:
    seen = set()
    out = {}
    for c in comms:
        c = frozenset(str(x) for x in c if str(x))
        if len(c) < 2 or c in seen:
            continue
        seen.add(c)
        out[len(out)] = set(c)
    return out


def run_slpa(graph: nx.Graph, seed: int, iterations: int = 50, threshold: float = 0.10) -> Dict[int, Set[str]]:
    rng = random.Random(seed)
    memory = {n: Counter({n: 1}) for n in graph.nodes()}
    nodes = list(graph.nodes())
    for _ in range(iterations):
        rng.shuffle(nodes)
        for listener in nodes:
            labels = []
            for speaker in graph.neighbors(listener):
                labels.extend(memory[speaker].elements())
            if labels:
                memory[listener][rng.choice(labels)] += 1
    groups: Dict[str, Set[str]] = defaultdict(set)
    for node, mem in memory.items():
        total = sum(mem.values())
        for label, count in mem.items():
            if count / total >= threshold:
                groups[str(label)].add(str(node))
    return relabel_communities(groups.values())


def run_clique_percolation(graph: nx.Graph, k: int = 3, max_edges: int = 60000) -> Dict[int, Set[str]]:
    if graph.number_of_edges() > max_edges:
        raise RuntimeError(f"Skipped: k-clique percolation over {graph.number_of_edges()} edges exceeds local cap {max_edges}.")
    comms = list(nx.algorithms.community.k_clique_communities(graph, k))
    return relabel_communities(comms)


def run_link_communities(graph: nx.Graph, max_edges: int = 12000) -> Dict[int, Set[str]]:
    if graph.number_of_edges() > max_edges:
        raise RuntimeError(f"Skipped: line-graph clustering over {graph.number_of_edges()} edges exceeds local cap {max_edges}.")
    line = nx.line_graph(graph)
    edge_comms = nx.algorithms.community.greedy_modularity_communities(line)
    node_comms = []
    for ec in edge_comms:
        nodes = set()
        for edge in ec:
            nodes.update(edge)
        node_comms.append(nodes)
    return relabel_communities(node_comms)


def run_clusterone_like(graph: nx.Graph, max_seeds: int = 250) -> Dict[int, Set[str]]:
    seeds = sorted(graph.degree(weight="weight"), key=lambda x: (-x[1], str(x[0])))[:max_seeds]
    comms = []
    for seed, _ in seeds:
        c = {seed}
        improved = True
        while improved and len(c) < 60:
            improved = False
            boundary = sorted(set().union(*(set(graph.neighbors(n)) for n in c)) - c)
            best_node, best_gain = None, 0.0
            current = _cohesiveness(graph, c)
            for b in boundary:
                gain = _cohesiveness(graph, c | {b}) - current
                if gain > best_gain:
                    best_node, best_gain = b, gain
            if best_node is not None and best_gain > 0.015:
                c.add(best_node)
                improved = True
        if len(c) >= 3:
            comms.append(c)
    return relabel_communities(comms)


def _cohesiveness(graph: nx.Graph, comm: Set[str]) -> float:
    win = 0.0
    wout = 0.0
    for n in comm:
        for nb, ed in graph[n].items():
            if nb in comm:
                win += float(ed.get("weight", 1.0))
            else:
                wout += float(ed.get("weight", 1.0))
    win /= 2.0
    return win / (win + wout + len(comm)) if (win + wout + len(comm)) else 0.0


def membership_df(clusters: Dict[int, Set[str]], method: str, dataset: str, seed: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"community_id": cid, "protein_id": p, "method": method, "dataset": dataset, "seed": seed}
            for cid, mem in clusters.items()
            for p in sorted(mem)
        ]
    )


def evaluate_method(dataset: str, method: str, graph: nx.Graph, go: Dict[str, Set[str]], gold: dict, seed: int = 42) -> Tuple[dict, pd.DataFrame | None, str]:
    t0 = time.time()
    note = "ok"
    membership = None
    try:
        if method == "MCL":
            clusters = baselines.mcl_only(graph)
        elif method == "MCL+overlap":
            clusters = baselines.mcl_overlap_heuristic(graph, go)
        elif method == "TRACE-PPI":
            if dataset != "gavin_yeast":
                raise RuntimeError("Current TRACE-PPI runner is validated only for Gavin in this repository.")
            out = run_v2("gavin", variant="recall_safe_orbits", write_outputs=False)
            clusters = out["clusters"]
            membership = out["refined"]
        elif method == "SLPA":
            clusters = run_slpa(graph, seed=seed)
        elif method == "CFinder-like":
            clusters = run_clique_percolation(graph, k=3)
        elif method == "Link communities":
            clusters = run_link_communities(graph)
        elif method == "ClusterONE-like":
            clusters = run_clusterone_like(graph)
        else:
            raise RuntimeError(f"Unknown method {method}")
        runtime = time.time() - t0
        if membership is None:
            membership = membership_df(clusters, method, dataset, seed)
        prf = precision_recall_f1_mmr(clusters, gold)
        ev = compute_all_evidence_metrics(clusters, go, membership if "membership_type" in membership.columns else None, runtime)
        multi = membership.groupby("protein_id")["community_id"].nunique() if not membership.empty else pd.Series(dtype=float)
        row = {
            "dataset": dataset,
            "method": method,
            "seed": seed,
            "status": "ok",
            "note": note,
            "jaccard_threshold": 0.5,
            **prf,
            **ev,
            "median_module_size": ev["median_size"],
            "multi_membership_mean": float(multi.mean()) if len(multi) else 0.0,
            "multi_membership_max": int(multi.max()) if len(multi) else 0,
        }
        return row, membership, note
    except Exception as exc:
        return {
            "dataset": dataset,
            "method": method,
            "seed": seed,
            "status": "failed_or_skipped",
            "note": str(exc),
            "jaccard_threshold": 0.5,
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
            "coverage": np.nan,
            "mean_size": np.nan,
            "median_module_size": np.nan,
            "num_modules": np.nan,
            "runtime_sec": time.time() - t0,
        }, None, str(exc)


def sensitivity(gavin_graph: nx.Graph, gavin_go: Dict[str, Set[str]], gold: dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    for infl in [1.4, 1.8, 2.0, 2.4, 3.0]:
        t0 = time.time()
        clusters = baselines.mcl_only(gavin_graph, inflation=infl)
        m = precision_recall_f1_mmr(clusters, gold)
        ev = compute_all_evidence_metrics(clusters, gavin_go, None, time.time() - t0)
        rows.append({"parameter": "mcl_inflation", "value": infl, **m, **ev})
    for thr in [0.05, 0.10, 0.20]:
        clusters = run_slpa(gavin_graph, seed=42, threshold=thr)
        m = precision_recall_f1_mmr(clusters, gold)
        ev = compute_all_evidence_metrics(clusters, gavin_go)
        rows.append({"parameter": "slpa_threshold", "value": thr, **m, **ev})
    for score in [0.30, 0.38, 0.46]:
        out = run_v2(
            "gavin",
            variant="recall_safe_orbits",
            params={"min_evidence_score": score},
            write_outputs=False,
        )
        ev = compute_all_evidence_metrics(out["clusters"], gavin_go, out["refined"])
        rows.append({"parameter": "supplement_min_evidence", "value": score, **out["metrics"], **ev})
    for variant in ["topology_only", "semantic_only", "recall_safe_orbits"]:
        out = run_v2("gavin", variant=variant, write_outputs=False)
        ev = compute_all_evidence_metrics(out["clusters"], gavin_go, out["refined"])
        rows.append({"parameter": "trace_weight_variant", "value": variant, **out["metrics"], **ev})
    sens = pd.DataFrame(rows)

    refined = run_v2("gavin", variant="recall_safe_orbits", write_outputs=False)["refined"]
    clusters = {
        int(cid): set(grp["protein_id"].astype(str))
        for cid, grp in refined.groupby("community_id")
    }
    gold_sets = [set(v) for v in gold.values() if len(v) >= 2]
    assign_rows = []
    for _, r in refined.iterrows():
        cid, p = int(r["community_id"]), str(r["protein_id"])
        pred = clusters[cid]
        supported = any(p in gs and len(pred & gs) / len(pred | gs) >= 0.5 for gs in gold_sets)
        assign_rows.append({**r.to_dict(), "gold_supported": float(supported)})
    assign = pd.DataFrame(assign_rows)
    cal_rows = []
    for inner in [0.18, 0.25, 0.32]:
        labels = np.where(
            (assign["topology_score"] >= 0.35) & (assign["semantic_score"] >= 0.25),
            "core",
            np.where((assign["topology_score"] >= inner) | (assign["semantic_score"] >= inner), "inner_or_above", "outer_or_uncertain"),
        )
        tmp = assign.assign(threshold=inner, recalibrated_label=labels)
        cal = tmp.groupby("recalibrated_label").agg(
            n=("protein_id", "size"),
            gold_supported_fraction=("gold_supported", "mean"),
            membership_mean=("membership_score", "mean"),
        ).reset_index()
        cal["inner_threshold"] = inner
        cal_rows.extend(cal.to_dict("records"))
    return sens, pd.DataFrame(cal_rows), assign


def write_baseline_feasibility() -> pd.DataFrame:
    rows = [
        {"baseline": "MCL", "status": "run", "implementation": "local Python/markov_clustering fallback", "note": "Core hard-partition baseline."},
        {"baseline": "MCL+overlap", "status": "run", "implementation": "local permanence + GO TF-IDF heuristic", "note": "Core overlap heuristic."},
        {"baseline": "SLPA", "status": "run", "implementation": "local deterministic-seeded implementation", "note": "Overlapping label-propagation baseline; should be replaced by a vetted package for final submission."},
        {"baseline": "CFinder/clique percolation", "status": "run_or_size-skipped", "implementation": "NetworkX k_clique_communities", "note": "k=3 feasible on available local networks under edge cap."},
        {"baseline": "Link communities", "status": "run_or_size-skipped", "implementation": "line graph + greedy modularity", "note": "Skipped for graphs exceeding line-graph cap."},
        {"baseline": "ClusterONE", "status": "proxy_only", "implementation": "ClusterONE-like greedy local growth", "note": "Exact ClusterONE binary not installed; proxy is not a publishable replacement."},
        {"baseline": "OSLOM", "status": "not_run", "implementation": "none", "note": "OSLOM binary not found and network install is restricted."},
        {"baseline": "BIGCLAM", "status": "not_run", "implementation": "none", "note": "Snap/BIGCLAM implementation not installed; requires external dependency acquisition."},
    ]
    return pd.DataFrame(rows)


def make_figures(bench: pd.DataFrame, sens: pd.DataFrame, label_cal: pd.DataFrame, case: pd.DataFrame) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.dpi": 320,
    })
    palette = {"MCL": "#3B6EA8", "MCL+overlap": "#C9823C", "TRACE-PPI": "#2F8F6B", "SLPA": "#7B6BB2", "CFinder-like": "#A55A5A", "Link communities": "#6E7781", "ClusterONE-like": "#B8A03A"}

    fig, ax = plt.subplots(figsize=(10.5, 3.2))
    ax.axis("off")
    xs = [0.03, 0.27, 0.51, 0.75]
    blocks = [
        ("Inputs", ["PPI network", "GO/text profiles"]),
        ("Evidence", ["Topology", "Semantic", "GO support"]),
        ("Overlap modules", ["MCL seed", "candidate scoring", "supplementation"]),
        ("Audit output", ["confidence labels", "evidence bundle", "curator ranking"]),
    ]
    for i, (x, (title, items)) in enumerate(zip(xs, blocks)):
        ax.add_patch(FancyBboxPatch((x, 0.18), 0.19, 0.64, boxstyle="round,pad=0.015,rounding_size=0.02", fc="white", ec="#2F8F6B", lw=1.4))
        ax.text(x + 0.095, 0.68, title, ha="center", va="center", fontweight="bold", color="#2F8F6B")
        for j, item in enumerate(items):
            ax.text(x + 0.025, 0.54 - j * 0.13, item, ha="left", va="center")
        if i < 3:
            ax.annotate("", xy=(xs[i + 1] - 0.02, 0.50), xytext=(x + 0.21, 0.50), arrowprops=dict(arrowstyle="-|>", color="#59636E", lw=1.2))
    ax.text(0.03, 0.93, "Figure 1. TRACE-PPI auditable overlap workflow", fontsize=12, fontweight="bold")
    fig.savefig(FIGURES / "fig1_workflow_high_rank.pdf", bbox_inches="tight")
    plt.close(fig)

    ok = bench[bench["status"] == "ok"].copy()
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    pivot = ok.pivot_table(index="dataset", columns="method", values="f1", aggfunc="mean")
    pivot = pivot[[c for c in ["MCL", "MCL+overlap", "TRACE-PPI", "SLPA", "CFinder-like", "ClusterONE-like", "Link communities"] if c in pivot.columns]]
    pivot.plot(kind="bar", ax=ax, color=[palette.get(c, "#777777") for c in pivot.columns], width=0.78)
    ax.set_ylabel("F1 at Jaccard >= 0.5")
    ax.set_xlabel("")
    ax.legend(frameon=False, ncol=3, fontsize=7)
    ax.grid(axis="y", color="#E8ECEF")
    fig.savefig(FIGURES / "fig2_multi_dataset_benchmark.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True)
    for ax, metric in zip(axes, ["precision", "recall", "coverage"]):
        p = ok.pivot_table(index="method", columns="dataset", values=metric, aggfunc="mean").fillna(0)
        p.plot(kind="barh", ax=ax, legend=False)
        ax.set_title(metric.title())
        ax.grid(axis="x", color="#E8ECEF")
    axes[-1].legend(frameon=False, fontsize=7, loc="lower right")
    fig.savefig(FIGURES / "fig3_baseline_comparison.pdf", bbox_inches="tight")
    plt.close(fig)

    trace = ok[ok["method"] == "TRACE-PPI"]
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8))
    axes[0].bar(["bundle fields", "overlap rate"], [float(trace["evidence_bundle_completeness"].mean()) if not trace.empty else 0, float(trace["overlap_protein_fraction"].mean()) if not trace.empty else 0], color=["#2F8F6B", "#C9823C"])
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Auditability outputs")
    ev = case[["topology_score", "semantic_score", "go_score"]].mean() if not case.empty else pd.Series({"topology_score": 0, "semantic_score": 0, "go_score": 0})
    axes[1].bar(["topology", "semantic", "GO"], [ev["topology_score"], ev["semantic_score"], ev["go_score"]], color=["#3B6EA8", "#2F8F6B", "#B8A03A"])
    axes[1].set_title("Case-study evidence channels")
    fig.savefig(FIGURES / "fig4_auditability_evidence.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    if not label_cal.empty:
        for label, grp in label_cal.groupby("recalibrated_label"):
            ax.plot(grp["inner_threshold"], grp["gold_supported_fraction"], marker="o", label=label)
    ax.set_xlabel("Inner-label threshold")
    ax.set_ylabel("Gold-supported assignment fraction")
    ax.set_title("Confidence-label enrichment sensitivity")
    ax.legend(frameon=False, fontsize=7)
    ax.grid(color="#E8ECEF")
    fig.savefig(FIGURES / "fig5_confidence_calibration.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    if not case.empty:
        x = np.arange(len(case))
        ax.plot(x, case["topology_score"], marker="o", label="Topology")
        ax.plot(x, case["semantic_score"], marker="o", label="Semantic")
        ax.plot(x, case["go_score"], marker="o", label="GO")
        ax.set_xticks(x)
        ax.set_xticklabels(case["community_id"].astype(str), rotation=0)
    ax.set_xlabel("YKR018C assigned module")
    ax.set_ylabel("Evidence score")
    ax.legend(frameon=False)
    ax.grid(color="#E8ECEF")
    fig.savefig(FIGURES / "fig6_case_study_ykr018c.pdf", bbox_inches="tight")
    plt.close(fig)


def write_manifest() -> None:
    manifest = {
        "timestamp_local": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cwd": str(ROOT),
        "seeds": SEEDS,
        "commands": [
            "python3 scripts/high_rank_rerun.py",
            "cd manuscript/high_rank_version && pdflatex -interaction=nonstopmode -halt-on-error trace_ppi_high_rank.tex && bibtex trace_ppi_high_rank && pdflatex -interaction=nonstopmode -halt-on-error trace_ppi_high_rank.tex && pdflatex -interaction=nonstopmode -halt-on-error trace_ppi_high_rank.tex",
        ],
        "notes": "Network access was not used; unavailable datasets/baselines are explicitly marked in feasibility tables.",
    }
    (RESULTS / "result_manifest.json").write_text(json.dumps(manifest, indent=2))


def main() -> None:
    ensure_dirs()
    gold = load_gold()
    datasets, data_inv = dataset_inventory(gold)
    data_inv.to_csv(TABLES / "dataset_inventory_mapping_losses.csv", index=False)
    write_baseline_feasibility().to_csv(TABLES / "baseline_feasibility.csv", index=False)

    methods = ["MCL", "MCL+overlap", "TRACE-PPI", "SLPA", "CFinder-like", "Link communities", "ClusterONE-like"]
    rows = []
    memberships = []
    for dname, (graph, go, _) in datasets.items():
        for method in methods:
            seeds = SEEDS if method == "SLPA" else [42]
            for seed in seeds:
                row, mem, _ = evaluate_method(dname, method, graph, go, gold, seed=seed)
                rows.append(row)
                if mem is not None and len(memberships) < 12:
                    memberships.append(mem.assign(source_method=method, source_dataset=dname))
    bench = pd.DataFrame(rows)
    bench.to_csv(TABLES / "multi_dataset_baseline_benchmark.csv", index=False)
    if memberships:
        pd.concat(memberships, ignore_index=True).to_csv(RESULTS / "sample_membership_assignments.csv", index=False)

    gavin_graph, gavin_go, _ = datasets["gavin_yeast"]
    sens, label_cal, assign = sensitivity(gavin_graph, gavin_go, gold)
    sens.to_csv(TABLES / "parameter_sensitivity.csv", index=False)
    label_cal.to_csv(TABLES / "confidence_label_threshold_sensitivity.csv", index=False)
    assign.to_csv(RESULTS / "trace_ppi_assignment_gold_support.csv", index=False)

    case = assign[assign["protein_id"] == "YKR018C"].copy()
    case["evidence_summary"] = case.apply(
        lambda r: (
            f"YKR018C assigned to module {int(r['community_id'])} as {r['membership_type']}; "
            f"topology={r['topology_score']:.2f}, semantic={r['semantic_score']:.2f}, GO={r['go_score']:.2f}; "
            f"gold-supported={bool(r['gold_supported'])}."
        ),
        axis=1,
    )
    case.to_csv(TABLES / "case_study_ykr018c_evidence_bundle.csv", index=False)

    make_figures(bench, sens, label_cal, case)
    write_manifest()
    shutil.copy2(ROOT / "manuscript" / "references.bib", MANUSCRIPT / "references.bib")
    print("High-rank rerun complete")
    print(bench[["dataset", "method", "seed", "status", "f1", "precision", "recall", "note"]].to_string(index=False))


if __name__ == "__main__":
    main()
