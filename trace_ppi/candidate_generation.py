"""Broad candidate module generation for COSMOS-PPI v2."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from .baselines import mcl_only
from .black_hole_cores import discover_cores
from .metrics_utils import jaccard, oracle_upper_bound
from .reuse import GENERIC_GO


def _ego_nodes(graph: nx.Graph, center: str, hops: int, max_size: int = 100) -> Set[str]:
    center = str(center)
    if center not in graph:
        return set()
    seen = {center}
    frontier = {center}
    for _ in range(hops):
        nxt = set()
        for u in frontier:
            for v in graph.neighbors(u):
                if v not in seen:
                    seen.add(v)
                    nxt.add(v)
                if len(seen) >= max_size:
                    return seen
        frontier = nxt
        if not frontier:
            break
    return seen


def _greedy_expand(
    graph: nx.Graph,
    seed_nodes: Set[str],
    emb: Dict[str, np.ndarray],
    max_size: int = 100,
    min_gain: float = 0.01,
) -> Set[str]:
    module = set(seed_nodes)
    candidates = set()
    for u in module:
        candidates |= set(graph.neighbors(u))
    candidates -= module
    while len(module) < max_size and candidates:
        best_p, best_gain = None, -1e9
        for p in candidates:
            nbr_in = sum(1 for n in graph.neighbors(p) if n in module)
            topo = nbr_in / max(1, len(module))
            sem = 0.0
            if p in emb and module:
                vecs = [emb[m] for m in module if m in emb]
                if vecs:
                    c = np.mean(vecs, axis=0)
                    v = emb[p]
                    sem = float(np.dot(v, c) / (np.linalg.norm(v) * np.linalg.norm(c) + 1e-9))
            gain = 0.6 * topo + 0.4 * sem
            if gain > best_gain:
                best_gain, best_p = gain, p
        if best_p is None or best_gain < min_gain:
            break
        module.add(best_p)
        candidates |= set(graph.neighbors(best_p))
        candidates -= module
    return module


def _semantic_neighbors(
    graph: nx.Graph,
    center: str,
    emb: Dict[str, np.ndarray],
    k: int,
    min_edge: bool = True,
) -> Set[str]:
    if center not in emb:
        return {center}
    v = emb[center]
    sims = []
    for p, e in emb.items():
        if p == center:
            continue
        if min_edge and p not in graph.neighbors(center) and not any(graph.has_edge(p, n) for n in graph.neighbors(center)):
            # require at least weak support: edge to center or to a center neighbor
            if center in graph and p in graph:
                cn = set(graph.neighbors(center))
                if not (graph.has_edge(p, center) or any(graph.has_edge(p, x) for x in cn)):
                    continue
        sims.append((float(np.dot(v, e) / (np.linalg.norm(v) * np.linalg.norm(e) + 1e-9)), p))
    sims.sort(reverse=True)
    chosen = {center} | {p for _, p in sims[:k]}
    return chosen


def generate_candidates(
    graph: nx.Graph,
    go_map: Dict[str, Set[str]],
    profiles: pd.DataFrame,
    emb_index: Dict[str, int],
    embeddings: np.ndarray,
    cores: pd.DataFrame,
    dataset: str,
    semantic_k: int = 50,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    emb = {pid: embeddings[i] for pid, i in emb_index.items()}
    rows: List[dict] = []
    cand_sets: Dict[int, Set[str]] = {}
    cid = 0

    # A. MCL modules
    mcl = mcl_only(graph)
    for _, members in mcl.items():
        mem = {str(p) for p in members}
        cand_sets[cid] = mem
        for p in mem:
            rows.append(
                dict(
                    dataset=dataset,
                    candidate_id=cid,
                    source="mcl",
                    protein_id=p,
                    candidate_score=0.0,
                    core_protein="",
                    rank_within_candidate=0,
                )
            )
        cid += 1

    # B–D. Black-hole ego / greedy / semantic per core
    for _, core in cores.iterrows():
        cp = str(core["core_protein"])
        for source, nodes in [
            ("bh_ego1", _ego_nodes(graph, cp, 1, 100)),
            ("bh_ego2", _ego_nodes(graph, cp, 2, 100)),
            ("greedy_expand", _greedy_expand(graph, {cp}, emb, 100)),
            ("semantic_k", _semantic_neighbors(graph, cp, emb, semantic_k)),
        ]:
            mem = {str(p) for p in nodes if str(p) in graph}
            if len(mem) < 2:
                continue
            cand_sets[cid] = mem
            for p in mem:
                rows.append(
                    dict(
                        dataset=dataset,
                        candidate_id=cid,
                        source=source,
                        protein_id=p,
                        candidate_score=0.0,
                        core_protein=cp,
                        rank_within_candidate=0,
                    )
                )
            cid += 1

    # E. Hybrid MCL + ego
    for mid, mem_mcl in list(mcl.items()):
        if not mem_mcl:
            continue
        cp = next(iter(mem_mcl))
        ego = _ego_nodes(graph, str(cp), 2, 100)
        union = {str(p) for p in (set(mem_mcl) | ego)}
        if len(union) < 2:
            continue
        if any(jaccard(union, existing) > 0.85 for existing in cand_sets.values()):
            continue
        cand_sets[cid] = union
        for p in union:
            rows.append(
                dict(
                    dataset=dataset,
                    candidate_id=cid,
                    source="hybrid_mcl_ego",
                    protein_id=p,
                    candidate_score=0.0,
                    core_protein=str(cp),
                    rank_within_candidate=0,
                )
            )
        cid += 1

    df = pd.DataFrame(rows)
    stats = {
        "n_candidates": len(cand_sets),
        "n_rows": len(df),
        "protein_coverage": df["protein_id"].nunique() if not df.empty else 0,
        "by_source": df.groupby("source")["candidate_id"].nunique().to_dict() if not df.empty else {},
    }
    return df, cand_sets, stats
