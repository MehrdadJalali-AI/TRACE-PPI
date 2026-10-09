"""Recall-safe per-protein supplementation under a module size budget."""
from __future__ import annotations

from typing import Dict, Set, Tuple

import networkx as nx

from .v2_membership import _member_scores
from .reuse import GENERIC_GO

SUPPLEMENT_DEFAULTS = dict(
    max_extra_members_per_module=2,
    max_relative_size_increase=0.15,
    min_evidence_score=0.42,
    min_go_support=0.25,
    min_topological_support=0.12,
    min_semantic_support=0.28,
    size_penalty_lambda=0.08,
    preserve_precision_floor=0.24,
)


def _evidence_gain(topo: float, sem: float, go_s: float, params: dict) -> float:
    p = params
    passes = (
        go_s >= p["min_go_support"]
        or topo >= p["min_topological_support"]
        or sem >= p["min_semantic_support"]
    )
    if not passes:
        return -1.0
    raw = (
        p.get("topo_channel_weight", 0.40) * topo
        + p.get("sem_channel_weight", 0.35) * sem
        + p.get("go_channel_weight", 0.25) * go_s
    )
    size_pen = p["size_penalty_lambda"] * (1.0 / max(1.0, 20.0))  # applied per step below
    return raw - size_pen


def supplement_modules_recall_safe(
    modules: Dict[int, Set[str]],
    graph: nx.Graph,
    go_map: dict,
    emb: dict,
    params: dict | None = None,
) -> Tuple[Dict[int, Set[str]], Dict[str, int]]:
    """Add boundary/candidate proteins to existing modules without naive expansion."""
    p = {**SUPPLEMENT_DEFAULTS, **(params or {})}
    stats = dict(
        proteins_added=0,
        modules_touched=0,
        modules_rejected_size=0,
        stop_gate=0,
        stop_gain=0,
        reject_precision=0,
        stop_two_member_cap=0,
        stop_relative_size_cap=0,
    )
    out: Dict[int, Set[str]] = {}

    for cid, members in modules.items():
        base = {str(m) for m in members}
        base_size = len(base)
        if base_size < 2:
            out[cid] = base
            continue
        max_size = int(base_size * (1.0 + p["max_relative_size_increase"]))
        max_size = max(max_size, base_size)
        cap = base_size + p["max_extra_members_per_module"]
        max_allowed = min(max_size, cap)

        mem = set(base)
        boundary: Set[str] = set()
        for u in sorted(mem):
            if u not in graph:
                continue
            for v in sorted(graph.neighbors(u)):
                if v not in mem:
                    boundary.add(v)

        added_here = 0
        while len(mem) < max_allowed and boundary:
            best_p, best_gain = None, -1e9
            for prot in sorted(boundary):
                topo, sem, go_s = _member_scores(prot, mem, graph, go_map, emb)
                gain = _evidence_gain(topo, sem, go_s, p)
                gain -= p["size_penalty_lambda"] * (len(mem) / 20.0)
                if gain > best_gain:
                    best_gain, best_p = gain, prot
            if best_p is None or best_gain < p["min_evidence_score"]:
                if best_p is not None:
                    topo_b, sem_b, go_b = _member_scores(best_p, mem, graph, go_map, emb)
                    passes_gate = (
                        go_b >= p["min_go_support"]
                        or topo_b >= p["min_topological_support"]
                        or sem_b >= p["min_semantic_support"]
                    )
                    stats["stop_gate" if not passes_gate else "stop_gain"] += 1
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
            added_here += 1
            stats["proteins_added"] += 1

        if added_here:
            stats["modules_touched"] += 1
        if len(mem) > max_allowed:
            stats["modules_rejected_size"] += 1
        if boundary and len(mem) >= max_allowed:
            rel_cap = max(int(base_size * (1.0 + p["max_relative_size_increase"])), base_size)
            member_cap = base_size + p["max_extra_members_per_module"]
            if member_cap <= rel_cap:
                stats["stop_two_member_cap"] += 1
            else:
                stats["stop_relative_size_cap"] += 1
        out[cid] = mem

    return out, stats
