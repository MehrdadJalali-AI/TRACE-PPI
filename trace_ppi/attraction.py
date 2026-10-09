"""Protein-to-community attraction scores."""
from __future__ import annotations

from typing import Dict, List, Set

import networkx as nx
import numpy as np
import pandas as pd

from .reuse import GENERIC_GO

LAMBDA = dict(
    topology=0.40,
    semantic=0.30,
    go=0.20,
    stability=0.00,
    uncertainty=0.10,
)


def community_embedding(members: List[str], emb: Dict[str, np.ndarray]) -> np.ndarray:
    vecs = [emb[m] for m in members if m in emb]
    if not vecs:
        return np.zeros(1)
    v = np.mean(vecs, axis=0)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def compute_attraction(
    graph: nx.Graph,
    cores: pd.DataFrame,
    profiles: pd.DataFrame,
    emb_index: Dict[str, int],
    embeddings: np.ndarray,
    go_map: Dict[str, Set[str]],
    stability_prior: Dict[tuple, float] | None = None,
) -> pd.DataFrame:
    emb = {pid: embeddings[i] for pid, i in emb_index.items()}
    prof = profiles.set_index("protein_id")
    communities: Dict[int, Set[str]] = {}
    for _, row in cores.iterrows():
        cid = int(row["core_id"])
        mem = set(str(row["initial_members"]).split(";")) if row["initial_members"] else {row["core_protein"]}
        communities[cid] = mem
    comm_emb = {cid: community_embedding(list(m), emb) for cid, m in communities.items()}

    rows = []
    nodes = list(graph.nodes())
    for p in nodes:
        for cid, members in communities.items():
            T = _topology(graph, p, members)
            S = _semantic(p, comm_emb[cid], emb)
            G = _go_overlap(p, members, go_map)
            R = stability_prior.get((p, cid), 0.0) if stability_prior else 0.0
            U = _uncertainty(p, members, go_map, prof, T)
            A = (
                LAMBDA["topology"] * T
                + LAMBDA["semantic"] * S
                + LAMBDA["go"] * G
                + LAMBDA["stability"] * R
                - LAMBDA["uncertainty"] * U
            )
            rows.append(
                dict(
                    protein_id=p,
                    community_id=cid,
                    attraction=A,
                    topology_score=T,
                    semantic_score=S,
                    go_score=G,
                    stability_score=R,
                    uncertainty_score=U,
                )
            )
    df = pd.DataFrame(rows)
    for col in ("attraction", "topology_score", "semantic_score", "go_score", "uncertainty_score"):
        df[col] = _norm01(df[col].values)
    return df


def _norm01(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def _topology(g: nx.Graph, p: str, members: Set[str]) -> float:
    if p not in g or not members:
        return 0.0
    nbrs = set(g.neighbors(p))
    hits = len(nbrs & members)
    return hits / max(1, len(members))


def _semantic(p: str, cemb: np.ndarray, emb: Dict[str, np.ndarray]) -> float:
    if p not in emb or cemb.size <= 1:
        return 0.0
    v = emb[p]
    return float(np.dot(v, cemb) / (np.linalg.norm(v) * np.linalg.norm(cemb) + 1e-9))


def _go_overlap(p: str, members: Set[str], go_map: Dict[str, Set[str]]) -> float:
    gp = (go_map.get(p, set()) - GENERIC_GO)
    if not gp:
        return 0.0
    union = set()
    for m in members:
        if m == p:
            continue
        union |= (go_map.get(m, set()) - GENERIC_GO)
    if not union:
        return 0.0
    return len(gp & union) / len(gp | union)


def _uncertainty(p, members, go_map, prof, T) -> float:
    gp = go_map.get(p, set()) - GENERIC_GO
    u = float(prof.loc[p, "annotation_uncertainty"]) if p in prof.index else 1.0
    if not gp:
        u = min(1.0, u + 0.3)
    if T < 0.1:
        u = min(1.0, u + 0.2)
    return u
