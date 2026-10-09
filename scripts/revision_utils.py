#!/usr/bin/env python3
"""Shared helpers for Scientific Reports revision analyses."""
from __future__ import annotations

import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Set, Tuple

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "results" / "matplotlib_cache"))
sys.path.insert(0, str(ROOT))

from trace_ppi.graph_io import filter_go, load_gavin, load_krogan
from trace_ppi.metrics_utils import jaccard
from trace_ppi.paths import DATA, RESULTS, ensure_dirs
from trace_ppi.reuse import load_gold_standard_csv, precision_recall_f1_mmr
from trace_ppi.v2_membership import overlap_fraction

REV_RESULTS = ROOT / "results" / "revision"
REV_TABLES = ROOT / "tables" / "revision"
REV_FIGURES = ROOT / "figures" / "revision"
REQUIRED_AUDIT_FIELDS = {
    "protein_id",
    "community_id",
    "membership_type",
    "membership_score",
    "topology_score",
    "semantic_score",
    "go_score",
}


def ensure_revision_dirs() -> None:
    ensure_dirs()
    for p in (REV_RESULTS, REV_TABLES, REV_FIGURES):
        p.mkdir(parents=True, exist_ok=True)


def load_config() -> dict:
    with open(ROOT / "configs" / "trace_ppi_final.yaml") as f:
        return yaml.safe_load(f)


def load_dataset(dataset: str):
    if dataset == "gavin":
        graph, go = load_gavin()
        return graph, filter_go(go)
    if dataset == "krogan":
        # Prefer the cleaned benchmark graph used for the reported Krogan results.
        from trace_ppi.paths import DATA, RESULTS
        from trace_ppi.reuse import GOLoader
        import networkx as nx
        import pandas as pd

        edge_path = RESULTS / "krogan" / "cleaned_graph_edges.csv"
        if edge_path.exists():
            edges = pd.read_csv(edge_path)
            graph = nx.Graph()
            for _, row in edges.iterrows():
                graph.add_edge(str(row["protein_a"]), str(row["protein_b"]), weight=float(row.get("weight", 1.0)))
            go_raw = GOLoader().load_from_gaf(str(DATA["go_txt"]), taxid=559292, use_symbol=True)
            nodes = set(graph.nodes())
            go = {p: terms for p, terms in go_raw.items() if p in nodes}
            return graph, filter_go(go)
        graph, go = load_krogan()
        return graph, filter_go(go)
    raise ValueError(dataset)


def load_gold() -> dict:
    return load_gold_standard_csv(str(DATA["gold_standard"]))


def clusters_from_pairs(pairs: Iterable[Tuple[int, str]]) -> Dict[int, Set[str]]:
    out: Dict[int, Set[str]] = defaultdict(set)
    for cid, protein in pairs:
        out[int(cid)].add(str(protein))
    return {k: v for k, v in out.items() if len(v) >= 2}


def clusters_from_membership(df: pd.DataFrame) -> Dict[int, Set[str]]:
    return clusters_from_pairs(zip(df["community_id"], df["protein_id"]))


def membership_from_clusters(clusters: Dict[int, Set[str]]) -> pd.DataFrame:
    rows = [
        {"community_id": cid, "protein_id": p}
        for cid, members in sorted(clusters.items())
        for p in sorted(members)
    ]
    return pd.DataFrame(rows)


def assignment_set(clusters: Dict[int, Set[str]]) -> Set[Tuple[int, str]]:
    return {(int(cid), str(p)) for cid, members in clusters.items() for p in members}


def protein_to_modules(clusters: Dict[int, Set[str]]) -> Dict[str, Set[int]]:
    mapping: Dict[str, Set[int]] = defaultdict(set)
    for cid, members in clusters.items():
        for p in members:
            mapping[str(p)].add(int(cid))
    return dict(mapping)


def audit_field_availability(membership: pd.DataFrame | None) -> float:
    if membership is None or membership.empty:
        return 0.0
    if not REQUIRED_AUDIT_FIELDS.issubset(set(membership.columns)):
        return 0.0
    return float(membership[list(REQUIRED_AUDIT_FIELDS)].notna().all(axis=1).mean())


def mean_module_size(clusters: Dict[int, Set[str]]) -> float:
    sizes = [len(c) for c in clusters.values() if len(c) >= 2]
    return float(np.mean(sizes)) if sizes else 0.0


def multi_membership_stats(clusters: Dict[int, Set[str]], method: str, dataset: str) -> dict:
    mem = membership_from_clusters(clusters)
    if mem.empty:
        return {
            "dataset": dataset,
            "method": method,
            "proteins_assigned": 0,
            "proteins_with_gt1_module": 0,
            "pct_with_gt1_module": 0.0,
            "mean_assignments_per_protein": 0.0,
            "median_assignments_per_protein": 0.0,
            "max_assignments_per_protein": 0,
            "total_assignments": 0,
            "modules": 0,
        }
    counts = mem.groupby("protein_id")["community_id"].nunique()
    n_multi = int((counts > 1).sum())
    return {
        "dataset": dataset,
        "method": method,
        "proteins_assigned": int(counts.shape[0]),
        "proteins_with_gt1_module": n_multi,
        "pct_with_gt1_module": 100.0 * n_multi / max(1, counts.shape[0]),
        "mean_assignments_per_protein": float(counts.mean()),
        "median_assignments_per_protein": float(counts.median()),
        "max_assignments_per_protein": int(counts.max()),
        "total_assignments": int(len(mem)),
        "modules": int(mem["community_id"].nunique()),
        "assignment_counts": counts,
    }


def gold_supported_mask(refined: pd.DataFrame, clusters: dict, gold: dict) -> pd.Series:
    gold_sets = [set(v) for v in gold.values() if len(v) >= 2]
    flags = []
    for _, r in refined.iterrows():
        protein = str(r["protein_id"])
        pred = clusters.get(int(r["community_id"]), set())
        supported = False
        for g in gold_sets:
            if protein in g and jaccard(pred, g) >= 0.5:
                supported = True
                break
        flags.append(supported)
    return pd.Series(flags, index=refined.index)


def normalize_label(value: str) -> str:
    return str(value).replace("_orbit", "").strip().lower()


def load_trace_assignments(dataset: str) -> pd.DataFrame:
    if dataset == "gavin":
        path = RESULTS / "communities" / "trace_ppi_refined_gavin.csv"
    else:
        path = RESULTS / "krogan" / "trace_ppi_refined.csv"
        if not path.exists():
            path = RESULTS / "communities" / "cosmos_v3_refined_krogan.csv"
    df = pd.read_csv(path)
    df["confidence_label"] = df["membership_type"].map(normalize_label)
    return df


def load_clusterone(dataset: str) -> Dict[int, Set[str]] | None:
    path = RESULTS / "baselines" / f"clusterone_{dataset}_modules.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return clusters_from_membership(df)


def trace_from_seed(
    selected: Dict[int, Set[str]],
    graph,
    go,
    emb_map: dict,
    params: dict,
) -> tuple[Dict[int, Set[str]], pd.DataFrame, dict]:
    from trace_ppi.event_horizon_v2 import refine_v2
    from trace_ppi.recall_safe_supplementation import supplement_modules_recall_safe
    from trace_ppi.v2_membership import modules_to_membership

    supplemented, stats = supplement_modules_recall_safe(selected, graph, go, emb_map, params)
    membership = modules_to_membership(supplemented, graph, go, emb_map)
    refined = refine_v2(membership, {**params, "preserve_all": True, "apply_orbit_labels": True})
    clusters = clusters_from_membership(refined)
    return clusters, refined, stats


def load_embeddings(dataset: str):
    from trace_ppi.cosmos_v2_runner import _load_embeddings

    prof_path = RESULTS / "evidence_profiles" / f"protein_profiles_{dataset}.csv"
    profiles = pd.read_csv(prof_path)
    _, _, _, emb_map = _load_embeddings(dataset, profiles)
    return emb_map


def evaluate_method(clusters: Dict[int, Set[str]], gold: dict, membership: pd.DataFrame | None = None) -> dict:
    metrics = precision_recall_f1_mmr(clusters, gold)
    return {
        **metrics,
        "mean_size": mean_module_size(clusters),
        "audit_field_availability": audit_field_availability(membership),
        "n_modules": len([c for c in clusters.values() if len(c) >= 2]),
        "n_assignments": sum(len(c) for c in clusters.values() if len(c) >= 2),
        "overlap_protein_fraction": overlap_fraction(membership_from_clusters(clusters)),
    }
