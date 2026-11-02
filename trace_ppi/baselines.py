"""Comparison methods.

* MCL                         strict MCL (trace_ppi.mcl), modules of >= 2 proteins.
* MCL + overlap               MCL followed by a permanence- and GO TF-IDF-based overlap
                              reassignment (trace_ppi.overlap_baseline).
* MCL + overlap + suppl.      the same, followed by bounded supplementation with the fixed
                              settings in SUPPLEMENT_BASELINE.
* ClusterONE                  official cluster_one-1.0.jar (Nepusz et al. 2012).
* SLPA                        SLPA run once with the memory thresholded at several values
                              (identical RNG sequence).
* GE (reimplementation)       graph-entropy seed growth following the six-step description
                              in Jeong et al. (Entropy 2021) / Kenley & Cho (2011); this is
                              our reimplementation, not the authors' code.
"""
from __future__ import annotations

import csv
import math
import random
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Set

import networkx as nx

from .config import ROOT
from .mcl import run_mcl
from . import overlap_baseline
from .supplementation import supplement

CLUSTERONE_JAR = ROOT / "tools" / "clusterone" / "cluster_one-1.0.jar"


def renumber(cover: Iterable[Set[str]], min_size: int = 2) -> Dict[int, Set[str]]:
    out, seen = {}, set()
    for mem in cover:
        key = frozenset(map(str, mem))
        if len(key) >= min_size and key not in seen:
            seen.add(key)
            out[len(out)] = set(key)
    return out


def mcl_modules(G: nx.Graph, inflation: float = 2.0) -> Dict[int, Set[str]]:
    return run_mcl(G, inflation).modules(2)


def mcl_overlap(G, go, inflation: float = 2.0) -> Dict[int, Set[str]]:
    return renumber(overlap_baseline.mcl_overlap(G, go, inflation).values())


SUPPLEMENT_BASELINE = dict(max_extra_members_per_module=2, max_relative_size_increase=0.15, min_evidence_score=0.38,
                           min_go_support=0.25, min_topological_support=0.15, min_semantic_support=0.28,
                           size_penalty_lambda=0.08, preserve_precision_floor=0.24)


def mcl_overlap_supplementation(G, go, emb, inflation: float = 2.0) -> Dict[int, Set[str]]:
    base = mcl_overlap(G, go, inflation)
    out, _, _ = supplement(base, G, go, emb, SUPPLEMENT_BASELINE)
    return renumber(out.values())


def clusterone(G: nx.Graph, density: float | None = None, min_size: int = 2) -> Dict[int, Set[str]]:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "edges.txt"
        with open(path, "w") as f:
            for u, v, d in G.edges(data=True):
                if u != v:
                    f.write(f"{u} {v} {float(d.get('weight', 1.0)):.8f}\n")
        cmd = ["java", "-jar", str(CLUSTERONE_JAR), "-f", "edge_list", "-F", "csv", "-s", str(min_size)]
        if density is not None:
            cmd += ["-d", str(density)]
        proc = subprocess.run(cmd + [str(path)], check=True, capture_output=True, text=True)
    cover = []
    for row in csv.DictReader(proc.stdout.splitlines()):
        mem = {p for p in row.get("Members", "").replace(";", " ").split() if p}
        cover.append(mem)
    return renumber(cover, min_size)


def slpa_multi(G: nx.Graph, thresholds: Iterable[float], iterations: int = 100, seed: int = 42) -> Dict[float, Dict[int, Set[str]]]:
    """Speaker-listener label propagation (Xie et al. 2011) with a fixed seed."""
    rng = random.Random(seed)
    nodes = [str(n) for n in G.nodes()]
    neighbors = {str(n): [str(x) for x in G.neighbors(n)] for n in G.nodes()}
    memory = {n: {n: 1} for n in nodes}
    for _ in range(iterations):
        listeners = nodes[:]
        rng.shuffle(listeners)
        for listener in listeners:
            votes = []
            for speaker in neighbors.get(listener, []):
                labels, counts = zip(*memory[speaker].items())
                pick = rng.uniform(0, sum(counts))
                acc, chosen = 0.0, labels[-1]
                for label, count in zip(labels, counts):
                    acc += count
                    if pick <= acc:
                        chosen = label
                        break
                votes.append(chosen)
            if votes:
                label = max(set(votes), key=lambda x: (votes.count(x), x))
                memory[listener][label] = memory[listener].get(label, 0) + 1
    out = {}
    for t in thresholds:
        comms: Dict[str, Set[str]] = {}
        for node, labels in memory.items():
            for label, count in labels.items():
                if count / (iterations + 1) >= t:
                    comms.setdefault(label, set()).add(node)
        out[t] = renumber(comms.values())
    return out


# ---------------------------------------------------------------- graph entropy
def graph_entropy_clusters(G: nx.Graph) -> Dict[int, Set[str]]:
    """Unweighted graph-entropy (GE) clustering, reimplemented from the published steps:
    1 seed = highest-degree node not yet in an output cluster (ties: identifier);
    2 initial cluster = seed + neighbours;
    3 remove each neighbour of the seed (descending degree) if GE decreases;
    4 add each boundary node (descending degree) if GE decreases, repeating passes
      until no addition lowers GE;
    5 output the cluster if it has >= 2 nodes;
    6 repeat until no seed remains.
    Node entropy e(v) = -p_in log2 p_in - p_out log2 p_out, with p_in the fraction of v's
    edges into the cluster; GE is the sum over nodes (only cluster members and boundary
    nodes contribute)."""
    adj = {str(n): {str(x) for x in G.neighbors(n) if x != n} for n in G.nodes()}
    deg = {n: len(a) for n, a in adj.items()}

    def h(k_in: int, d: int) -> float:
        if d == 0 or k_in == 0 or k_in == d:
            return 0.0
        p = k_in / d
        return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))

    def delta(C: Set[str], kin: Dict[str, int], v: str, add: bool) -> float:
        """Change of GE when v is added to / removed from C (v's own entropy and its neighbours')."""
        s = 1 if add else -1
        before = h(kin.get(v, 0), deg[v])
        after = h(kin.get(v, 0), deg[v])  # v's own k_in does not change, only membership
        change = after - before
        for u in adj[v]:
            k = kin.get(u, 0)
            change += h(k + s, deg[u]) - h(k, deg[u])
        return change

    covered: Set[str] = set()
    used: Set[str] = set()
    clusters: List[Set[str]] = []
    order = sorted(adj, key=lambda n: (-deg[n], n))
    for seed in order:
        if seed in covered or seed in used or deg[seed] == 0:
            continue
        used.add(seed)
        C = {seed} | adj[seed]
        kin: Dict[str, int] = {}
        for c in C:
            for u in adj[c]:
                kin[u] = kin.get(u, 0) + 1
        for v in sorted(adj[seed], key=lambda n: (-deg[n], n)):
            if v in C and delta(C, kin, v, add=False) < 0:
                C.discard(v)
                for u in adj[v]:
                    kin[u] -= 1
        changed = True
        while changed:
            changed = False
            boundary = sorted({u for c in C for u in adj[c]} - C, key=lambda n: (-deg[n], n))
            for v in boundary:
                if delta(C, kin, v, add=True) < 0:
                    C.add(v)
                    for u in adj[v]:
                        kin[u] = kin.get(u, 0) + 1
                    changed = True
        if len(C) >= 2:
            clusters.append(set(C))
            covered |= C
    return renumber(clusters)
