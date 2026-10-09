"""Baseline community detection for TRACE-PPI evaluation."""
from __future__ import annotations

import csv
import random
import subprocess
from pathlib import Path
from typing import Dict, Set

import networkx as nx

from .paths import TRACE_ROOT, RESULTS
from .reuse import MCLClustering, apply_overlap_reassignment, GOTFIDF, calculate_permanence_all_proteins


def clusters_dict_from_sets(communities: Dict[int, Set[str]]) -> Dict[int, Set[str]]:
    return {k: v for k, v in communities.items() if len(v) >= 2}


def mcl_only(graph: nx.Graph, inflation: float = 2.0) -> Dict[int, Set[str]]:
    mcl = MCLClustering(inflation=inflation)
    return clusters_dict_from_sets(mcl.cluster(graph))


def mcl_overlap_heuristic(
    graph: nx.Graph,
    go_map: Dict[str, Set[str]],
    inflation: float = 2.0,
    overlap_tau: float = 0.1,
) -> Dict[int, Set[str]]:
    clusters = mcl_only(graph, inflation=inflation)
    gt = GOTFIDF(clusters, go_map)
    perm = calculate_permanence_all_proteins(clusters, graph)
    return apply_overlap_reassignment(
        clusters, graph, go_map, gt, perm, alpha=0.5, overlap_tau=overlap_tau
    )


def simple_function_aware_heuristic(graph: nx.Graph, go_map: Dict[str, Set[str]]) -> Dict[int, Set[str]]:
    return mcl_overlap_heuristic(graph, go_map)


def clusterone_exact(
    graph: nx.Graph,
    jar_path: Path | None = None,
    min_size: int = 2,
    dataset: str = "gavin",
) -> Dict[int, Set[str]]:
    """Run the official ClusterONE JAR on the weighted Gavin graph.

    This is intentionally not a proxy implementation. If the official JAR is
    unavailable or fails, the caller should report infeasibility rather than
    labelling another method as ClusterONE.
    """
    jar = jar_path or TRACE_ROOT / "tools" / "clusterone" / "cluster_one-1.0.jar"
    if not jar.exists():
        raise FileNotFoundError(f"Official ClusterONE JAR not found: {jar}")

    out_dir = RESULTS / "baselines"
    out_dir.mkdir(parents=True, exist_ok=True)
    edge_path = out_dir / f"{dataset}_clusterone_input.tsv"
    raw_path = out_dir / f"clusterone_{dataset}_raw.csv"

    with edge_path.open("w") as f:
        for u, v, data in graph.edges(data=True):
            if u == v:
                continue
            weight = float(data.get("weight", 1.0))
            f.write(f"{u} {v} {weight:.8f}\n")

    cmd = [
        "java",
        "-jar",
        str(jar),
        "-f",
        "edge_list",
        "-F",
        "csv",
        "-s",
        str(min_size),
        str(edge_path),
    ]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    raw_path.write_text(proc.stdout)

    clusters: Dict[int, Set[str]] = {}
    reader = csv.DictReader(proc.stdout.splitlines())
    for i, row in enumerate(reader):
        members = {p for p in row.get("Members", "").replace(";", " ").split() if p}
        if len(members) >= min_size:
            clusters[i] = members

    pd_rows = [
        {"community_id": cid, "protein_id": protein}
        for cid, members in clusters.items()
        for protein in sorted(members)
    ]
    try:
        import pandas as pd

        pd.DataFrame(pd_rows).to_csv(out_dir / f"clusterone_{dataset}_modules.csv", index=False)
    except Exception:
        pass
    return clusters


def slpa(
    graph: nx.Graph,
    threshold: float = 0.10,
    iterations: int = 100,
    seed: int = 42,
) -> Dict[int, Set[str]]:
    """Speaker-listener label propagation algorithm (SLPA).

    The implementation follows the standard memory-based SLPA procedure and is
    used as a reproducible overlapping-community baseline/sensitivity check.
    """
    rng = random.Random(seed)
    nodes = [str(n) for n in graph.nodes()]
    neighbors = {str(n): [str(x) for x in graph.neighbors(n)] for n in graph.nodes()}
    memory = {n: {n: 1} for n in nodes}

    for _ in range(iterations):
        listeners = nodes[:]
        rng.shuffle(listeners)
        for listener in listeners:
            votes = []
            for speaker in neighbors.get(listener, []):
                labels, counts = zip(*memory[speaker].items())
                total = sum(counts)
                pick = rng.uniform(0, total)
                acc = 0.0
                chosen = labels[-1]
                for label, count in zip(labels, counts):
                    acc += count
                    if pick <= acc:
                        chosen = label
                        break
                votes.append(chosen)
            if not votes:
                continue
            label = max(set(votes), key=lambda x: (votes.count(x), x))
            memory[listener][label] = memory[listener].get(label, 0) + 1

    communities: Dict[str, Set[str]] = {}
    denom = iterations + 1
    for node, labels in memory.items():
        for label, count in labels.items():
            if count / denom >= threshold:
                communities.setdefault(label, set()).add(node)

    out: Dict[int, Set[str]] = {}
    seen = set()
    for members in communities.values():
        frozen = frozenset(members)
        if len(members) >= 2 and frozen not in seen:
            seen.add(frozen)
            out[len(out)] = members
    return out


def cosmos_to_clusters(refined) -> Dict[int, Set[str]]:
    from collections import defaultdict
    out: Dict[int, Set[str]] = defaultdict(set)
    for _, r in refined.iterrows():
        out[int(r["community_id"])].add(r["protein_id"])
    return clusters_dict_from_sets(out)
