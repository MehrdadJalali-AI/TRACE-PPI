"""Evidence-based scoring of candidate modules."""
from __future__ import annotations

from typing import Dict, Set

import networkx as nx
import numpy as np
import pandas as pd

from .reuse import GENERIC_GO
from .metrics_utils import jaccard

WEIGHTS = dict(w1=0.22, w2=0.15, w3=0.18, w4=0.15, w5=0.12, w6=0.10, w7=0.08)


def _conductance(graph: nx.Graph, nodes: Set[str]) -> float:
    cut = 0
    vol_in = 0
    for u in nodes:
        if u not in graph:
            continue
        for v in graph.neighbors(u):
            w = graph[u][v].get("weight", 1.0)
            vol_in += w
            if v not in nodes:
                cut += w
    return cut / (vol_in + 1e-9)


def score_candidates(
    graph: nx.Graph,
    cand_sets: Dict[int, Set[str]],
    go_map: Dict[str, Set[str]],
    profiles: pd.DataFrame,
    emb: Dict[str, np.ndarray],
    cores: pd.DataFrame,
) -> pd.DataFrame:
    core_strength = {}
    if not cores.empty:
        for _, r in cores.iterrows():
            core_strength[str(r["core_protein"])] = float(r["core_potential"])

    prof = profiles.set_index("protein_id")
    rows = []
    for cid, members in cand_sets.items():
        mem = [m for m in members if m in graph]
        if len(mem) < 2:
            continue
        sub = graph.subgraph(mem)
        m = len(mem)
        density = 2 * sub.number_of_edges() / (m * (m - 1)) if m > 1 else 0.0
        conduct = _conductance(graph, set(mem))
        topo = min(1.0, density * (1.0 - min(1.0, conduct)))

        vecs = [emb[p] for p in mem if p in emb]
        if len(vecs) >= 2:
            c = np.mean(vecs, axis=0)
            sem = float(np.mean([np.dot(c, v) / (np.linalg.norm(c) * np.linalg.norm(v) + 1e-9) for v in vecs]))
        else:
            sem = 0.0

        go_terms = set()
        for p in mem:
            go_terms |= (go_map.get(p, set()) - GENERIC_GO)
        go_coh = 0.0
        if go_terms:
            hits = sum(1 for p in mem if go_map.get(p, set()) - GENERIC_GO)
            go_coh = hits / len(mem)

        bh = 0.0
        for p in mem:
            bh = max(bh, core_strength.get(p, 0.0))

        ideal = 20.0
        size_prior = float(np.exp(-((m - ideal) ** 2) / (2 * 15**2)))

        unc = float(prof.loc[mem, "annotation_uncertainty"].mean()) if all(p in prof.index for p in mem[:1]) else 0.5
        try:
            unc = float(prof.reindex(mem)["annotation_uncertainty"].fillna(0.5).mean())
        except Exception:
            unc = 0.5

        frag = 1.0 if m < 3 else 0.0
        overlap_plaus = min(1.0, go_coh + topo)

        score = (
            WEIGHTS["w1"] * topo
            + WEIGHTS["w2"] * bh
            + WEIGHTS["w3"] * sem
            + WEIGHTS["w4"] * go_coh
            + WEIGHTS["w5"] * size_prior
            - WEIGHTS["w6"] * unc
            - WEIGHTS["w7"] * frag
        )
        rows.append(
            dict(
                candidate_id=cid,
                size=m,
                topology_cohesion=topo,
                weighted_internal_density=density,
                conductance=conduct,
                black_hole_core_strength=bh,
                semantic_coherence=sem,
                non_generic_go_coherence=go_coh,
                evidence_richness=1.0 - unc,
                annotation_uncertainty=unc,
                size_prior_score=size_prior,
                overlap_plausibility_score=overlap_plaus,
                candidate_score=score,
            )
        )
    return pd.DataFrame(rows).sort_values("candidate_score", ascending=False)


def select_modules(
    scores: pd.DataFrame,
    cand_sets: Dict[int, Set[str]],
    max_modules: int = 120,
    jaccard_thresh: float = 0.65,
) -> Dict[int, Set[str]]:
    selected: Dict[int, Set[str]] = {}
    for _, row in scores.iterrows():
        if len(selected) >= max_modules:
            break
        cid = int(row["candidate_id"])
        mem = cand_sets.get(cid, set())
        if len(mem) < 2:
            continue
        if any(jaccard(mem, s) >= jaccard_thresh for s in selected.values()):
            continue
        selected[cid] = mem
    return selected
