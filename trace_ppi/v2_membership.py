"""Build membership tables with per-protein evidence scores for COSMOS v2."""
from __future__ import annotations

from typing import Dict, Set

import networkx as nx
import numpy as np
import pandas as pd

from .reuse import GENERIC_GO


def _member_scores(
    protein: str,
    members: Set[str],
    graph: nx.Graph,
    go_map: dict,
    emb: dict,
    topo_weight: float = 1.0,
    sem_weight: float = 1.0,
) -> tuple[float, float, float]:
    mem = set(members)
    nbr_in = sum(1 for n in graph.neighbors(protein) if n in mem) if protein in graph else 0
    topo = min(1.0, nbr_in / max(1, len(mem) - 1))
    sem = 0.0
    if protein in emb:
        vecs = [emb[m] for m in sorted(mem) if m in emb and m != protein]
        if vecs:
            c = np.mean(vecs, axis=0)
            v = emb[protein]
            sem = float(np.dot(v, c) / (np.linalg.norm(v) * np.linalg.norm(c) + 1e-9))
    go_terms = go_map.get(protein, set()) - GENERIC_GO
    others = set()
    for m in mem:
        if m == protein:
            continue
        others |= go_map.get(m, set()) - GENERIC_GO
    shared = go_terms & others
    go_score = len(shared) / len(go_terms) if go_terms else 0.0
    topo *= topo_weight
    sem *= sem_weight
    return topo, sem, go_score


def provenance_record(dataset: str, inflation: float = 2.0) -> str:
    """Run-level provenance copied onto every exported assignment row."""
    return (
        f"dataset={dataset};mcl=unweighted_adjacency;inflation={inflation};"
        "overlap_tau=0.1;min_module_size=2;embedding=tfidf_svd_l2;"
        "labels=core:tau>=0.35&sigma>=0.25;inner:0.25;outer:0.12"
    )


def modules_to_membership(
    modules: Dict[int, Set[str]],
    graph: nx.Graph,
    go_map: dict,
    emb: dict,
    topo_weight: float = 1.0,
    sem_weight: float = 1.0,
    provenance: str | None = None,
) -> pd.DataFrame:
    rows = []
    for cid, members in sorted(modules.items()):
        mem = {str(m) for m in members}
        for p in sorted(mem):
            topo, sem, go_s = _member_scores(p, mem, graph, go_map, emb, topo_weight, sem_weight)
            rows.append(
                dict(
                    community_id=cid,
                    protein_id=p,
                    membership_type="core",
                    membership_score=0.5 * topo + 0.5 * sem,
                    topology_score=topo,
                    semantic_score=sem,
                    go_score=go_s,
                    expansion_gain=0.0,
                    **({} if provenance is None else {"provenance": provenance}),
                )
            )
    return pd.DataFrame(rows)


def overlap_fraction(membership: pd.DataFrame) -> float:
    if membership.empty:
        return 0.0
    counts = membership.groupby("protein_id")["community_id"].nunique()
    return float((counts > 1).mean())
