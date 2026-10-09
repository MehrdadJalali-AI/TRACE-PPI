"""Black-hole module core discovery."""
from __future__ import annotations

from typing import Dict, List, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd

POSITIVE_WEIGHTS = {
    "weighted_degree_norm": 0.25,
    "local_clustering_norm": 0.15,
    "k_core_score_norm": 0.20,
    "semantic_neighborhood_coherence": 0.20,
    "evidence_richness_norm": 0.20,
}
LAMBDA_UNC = 0.05

assert abs(sum(POSITIVE_WEIGHTS.values()) - 1.0) < 1e-6


def semantic_neighborhood_coherence(graph: nx.Graph, p: str, emb: Dict[str, np.ndarray]) -> float:
    if p not in emb:
        return 0.0
    nbrs = [n for n in graph.neighbors(p) if n in emb]
    if not nbrs:
        return 0.0
    v = emb[p]
    sims = [
        float(np.dot(v, emb[nb]) / (np.linalg.norm(v) * np.linalg.norm(emb[nb]) + 1e-9))
        for nb in nbrs
    ]
    return float(np.mean(sims))


def discover_cores(
    graph: nx.Graph,
    profiles: pd.DataFrame,
    emb_index: Dict[str, int],
    embeddings: np.ndarray,
    min_sep_hops: int = 2,
    max_cores: int | None = None,
) -> pd.DataFrame:
    emb_map = {pid: embeddings[i] for pid, i in emb_index.items()}
    prof = profiles.set_index("protein_id")
    scores = []
    for p in graph.nodes():
        if str(p) not in prof.index and p not in prof.index:
            continue
        row = prof.loc[str(p) if str(p) in prof.index else p]
        snc = semantic_neighborhood_coherence(graph, p, emb_map)
        cp = (
            POSITIVE_WEIGHTS["weighted_degree_norm"] * row["weighted_degree_norm"]
            + POSITIVE_WEIGHTS["local_clustering_norm"] * row["local_clustering_norm"]
            + POSITIVE_WEIGHTS["k_core_score_norm"] * row["k_core_score_norm"]
            + POSITIVE_WEIGHTS["semantic_neighborhood_coherence"] * _minmax_scalar(snc)
            + POSITIVE_WEIGHTS["evidence_richness_norm"] * row["evidence_richness_norm"]
            - LAMBDA_UNC * row["annotation_uncertainty_norm"]
        )
        scores.append((p, cp, snc, row))
    scores.sort(key=lambda x: x[1], reverse=True)
    if max_cores is None:
        max_cores = max(30, min(80, graph.number_of_nodes() // 15))

    selected: List[str] = []
    selected_set: Set[str] = set()
    rows = []
    cid = 0
    lengths = dict(nx.all_pairs_shortest_path_length(graph, cutoff=min_sep_hops))

    for p, cp, snc, row in scores:
        if len(selected) >= max_cores:
            break
        too_close = False
        for q in selected:
            d = lengths.get(p, {}).get(q, 999)
            if d < min_sep_hops:
                too_close = True
                break
        if too_close:
            continue
        selected.append(p)
        selected_set.add(p)
        members = {p}
        for n in graph.neighbors(p):
            if graph[p][n].get("weight", 1.0) >= 0.3:
                members.add(n)
        rows.append(
            {
                "core_id": cid,
                "core_protein": p,
                "core_potential": cp,
                "weighted_degree": row["weighted_degree"],
                "local_clustering": row["local_clustering"],
                "k_core_score": row["k_core_score"],
                "semantic_neighborhood_coherence": snc,
                "evidence_richness": row["evidence_richness"],
                "initial_members": ";".join(sorted(str(m) for m in members)),
            }
        )
        cid += 1
    return pd.DataFrame(rows)


def _minmax_scalar(x: float) -> float:
    return max(0.0, min(1.0, x))
