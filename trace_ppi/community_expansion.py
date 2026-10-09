"""Boundary expansion to recover recall."""
from __future__ import annotations

from typing import Dict, Set

import networkx as nx
import pandas as pd

from .reuse import GENERIC_GO


def expand_modules(
    graph: nx.Graph,
    modules: Dict[int, Set[str]],
    go_map: dict,
    emb: dict,
    expansion_threshold: float = 0.12,
    max_size: int = 80,
) -> pd.DataFrame:
    rows = []
    for cid, members in modules.items():
        mem = set(members)
        boundary = set()
        for u in mem:
            if u not in graph:
                continue
            for v in graph.neighbors(u):
                if v not in mem:
                    boundary.add(v)
        # seed core members
        for p in mem:
            rows.append(_row(cid, p, "core", 0.9, 0.9, 0.8, graph, go_map, emb, mem))
        if len(mem) >= max_size:
            continue
        while len(mem) < max_size and boundary:
            best_p, best_gain, best_parts = None, -1e9, (0, 0, 0)
            for p in boundary:
                nbr_in = sum(1 for n in graph.neighbors(p) if n in mem)
                topo_gain = nbr_in / max(1, len(mem))
                sem_gain = 0.0
                if p in emb:
                    vecs = [emb[m] for m in mem if m in emb]
                    if vecs:
                        import numpy as np
                        c = np.mean(vecs, axis=0)
                        v = emb[p]
                        sem_gain = float(np.dot(v, c) / (np.linalg.norm(v) * np.linalg.norm(c) + 1e-9))
                go_gain = 1.0 if (go_map.get(p, set()) - GENERIC_GO) & set().union(
                    *(go_map.get(m, set()) - GENERIC_GO for m in mem if m != p)
                ) else 0.0
                size_gain = 1.0 - len(mem) / max_size
                gain = 0.45 * topo_gain + 0.35 * sem_gain + 0.15 * go_gain + 0.05 * size_gain
                if gain > best_gain:
                    best_gain, best_p, best_parts = gain, p, (topo_gain, sem_gain, go_gain)
            if best_p is None:
                break
            topo_g, sem_g, go_g = best_parts
            add = (
                best_gain >= expansion_threshold
                or topo_g >= 0.15
                or sem_g >= 0.35
                or (go_g > 0 and topo_g > 0.02)
            )
            if not add:
                boundary.discard(best_p)
                if not boundary:
                    break
                continue
            if topo_g >= 0.2 and sem_g >= 0.2:
                mtype = "inner_orbit"
            elif topo_g >= 0.15 or sem_g >= 0.3:
                mtype = "inner_orbit"
            elif best_gain >= expansion_threshold:
                mtype = "outer_orbit"
            else:
                mtype = "uncertain_orbit"
            mem.add(best_p)
            rows.append(_row(cid, best_p, mtype, best_gain, topo_g, sem_g, graph, go_map, emb, mem))
            boundary |= set(graph.neighbors(best_p))
            boundary -= mem
    return pd.DataFrame(rows)


def _row(cid, p, mtype, gain, topo, sem, graph, go_map, emb, mem):
    return dict(
        community_id=cid,
        protein_id=p,
        membership_type=mtype,
        membership_score=float(gain),
        topology_score=float(topo),
        semantic_score=float(sem),
        go_score=float(
            bool((go_map.get(p, set()) - GENERIC_GO)
                 & set().union(*(go_map.get(m, set()) - GENERIC_GO for m in mem if m != p)))
        ),
        expansion_gain=float(gain),
    )
