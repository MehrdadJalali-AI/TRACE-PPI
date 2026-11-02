"""Bounded supplementation with an addition log.

The gate and gain use the fixed scoring in _member_scores. TRACE-PPI channel
scores are recomputed afterwards on the final module composition.
"""
from __future__ import annotations

from typing import Dict, List, Mapping, Set, Tuple

import networkx as nx
import numpy as np

GENERIC_GO = {"GO:0003674", "GO:0005575", "GO:0008150"}


def _member_scores(protein, members, graph, go_map, emb):
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
        if m != protein:
            others |= go_map.get(m, set()) - GENERIC_GO
    go_score = len(go_terms & others) / len(go_terms) if go_terms else 0.0
    return topo, sem, go_score


def _passes_gate(topo, sem, go_s, p) -> bool:
    return go_s >= p["min_go_support"] or topo >= p["min_topological_support"] or sem >= p["min_semantic_support"]


def supplement(modules: Mapping[int, Set[str]], graph: nx.Graph, go_map, emb, params: dict
               ) -> Tuple[Dict[int, Set[str]], List[dict], Dict[str, int]]:
    p = params
    stats = dict(proteins_added=0, modules_touched=0, stop_gate=0, stop_gain=0, reject_precision=0)
    log: List[dict] = []
    out: Dict[int, Set[str]] = {}
    for cid, members in modules.items():
        base = {str(m) for m in members}
        if len(base) < 2:
            out[cid] = base
            continue
        max_size = max(int(len(base) * (1.0 + p["max_relative_size_increase"])), len(base))
        max_allowed = min(max_size, len(base) + p["max_extra_members_per_module"])
        mem = set(base)
        boundary = {v for u in sorted(mem) if u in graph for v in sorted(graph.neighbors(u)) if v not in mem}
        added = 0
        while len(mem) < max_allowed and boundary:
            best_p, best_gain = None, -1e9
            for prot in sorted(boundary):
                topo, sem, go_s = _member_scores(prot, mem, graph, go_map, emb)
                gain = (-1.0 if not _passes_gate(topo, sem, go_s, p) else
                        0.40 * topo + 0.35 * sem + 0.25 * go_s - p["size_penalty_lambda"] / 20.0)
                gain -= p["size_penalty_lambda"] * (len(mem) / 20.0)
                if gain > best_gain:
                    best_gain, best_p = gain, prot
            if best_p is None or best_gain < p["min_evidence_score"]:
                if best_p is not None:
                    t, s, g = _member_scores(best_p, mem, graph, go_map, emb)
                    stats["stop_gate" if not _passes_gate(t, s, g, p) else "stop_gain"] += 1
                break
            topo, sem, go_s = _member_scores(best_p, mem, graph, go_map, emb)
            est_precision = 0.5 * topo + 0.35 * sem + 0.15 * go_s
            if est_precision < p["preserve_precision_floor"] and go_s < p["min_go_support"]:
                stats["reject_precision"] += 1
                boundary.discard(best_p)
                continue
            mem.add(best_p)
            boundary.discard(best_p)
            boundary |= {n for n in sorted(graph.neighbors(best_p)) if n not in mem}
            added += 1
            stats["proteins_added"] += 1
            log.append({"module_id": cid, "protein_id": best_p, "supplementation_gain": float(best_gain)})
        stats["modules_touched"] += int(added > 0)
        out[cid] = mem
    return out, log, stats
