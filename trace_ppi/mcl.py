"""Strict Markov clustering: a failed run is reported, never replaced by Louvain."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix


class MCLFailure(RuntimeError):
    pass


@dataclass
class MCLResult:
    regions: Dict[int, Set[str]]      # every node, including singleton regions
    inflation: float
    converged: bool
    n_singletons: int
    max_region: int

    def modules(self, min_size: int = 2) -> Dict[int, Set[str]]:
        return {k: v for k, v in self.regions.items() if len(v) >= min_size}


def run_mcl(graph: nx.Graph, inflation: float = 2.0, expansion: int = 2, loop_value: float = 1.0,
            iterations: int = 100) -> MCLResult:
    """Unweighted MCL via ``markov_clustering`` (van Dongen 2000) on the adjacency matrix.

    Nodes are ordered by sorted identifier so the output does not depend on file order.
    Isolated nodes are returned as singleton regions.
    """
    import markov_clustering as mc

    nodes: List[str] = sorted(map(str, graph.nodes()))
    if len(nodes) < 2:
        return MCLResult({i: {n} for i, n in enumerate(nodes)}, inflation, True, len(nodes), 1)
    idx = {n: i for i, n in enumerate(nodes)}
    rows, cols = [], []
    for u, v in graph.edges():
        if u == v:
            continue
        rows += [idx[str(u)], idx[str(v)]]
        cols += [idx[str(v)], idx[str(u)]]
    matrix = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(nodes), len(nodes)))
    matrix.data[:] = 1.0
    try:
        result = mc.run_mcl(matrix, expansion=expansion, inflation=inflation, loop_value=loop_value,
                            iterations=iterations)
        clusters = mc.get_clusters(result)
    except Exception as exc:  # pragma: no cover - reported, not hidden
        raise MCLFailure(f"markov_clustering failed at inflation {inflation}: {exc}") from exc
    # Convergence check: one further expansion/inflation step must not change the matrix.
    last = mc.normalize(mc.inflate(mc.expand(result, expansion), inflation))
    converged = bool(abs(last - result).max() < 1e-6) if hasattr(last, "max") else True
    assigned: Dict[str, int] = {}
    regions: Dict[int, Set[str]] = {}
    # MCL attractor systems can overlap in rare degenerate cases; a node keeps the first
    # (lowest-index) cluster so that the regions form a partition.
    for members in sorted((tuple(sorted(c)) for c in clusters), key=lambda c: (c[0], len(c))):
        free = [nodes[i] for i in members if nodes[i] not in assigned]
        if not free:
            continue
        rid = len(regions)
        regions[rid] = set(free)
        for n in free:
            assigned[n] = rid
    for n in nodes:
        if n not in assigned:
            rid = len(regions)
            regions[rid] = {n}
            assigned[n] = rid
    sizes = [len(v) for v in regions.values()]
    return MCLResult(regions, inflation, converged, sum(s == 1 for s in sizes), max(sizes))
