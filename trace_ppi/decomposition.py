"""Heterogeneity-guided deconvolution of one flagged region.

Alternatives compared for region R (all restricted to G[R]):

* no_split     D = {R}
* spectral     sign split of the Fiedler vector of the normalised Laplacian of G[R]
               (median split if a side has < m proteins; unavailable if still < m);
               a disconnected R is split into its connected components of size >= m.
* seed_expand  seeds chosen by non-maximum suppression on a seed score (mean
               percentile rank of coreness, degree and clustering in G[R]); each seed
               core is the seed plus its c best-connected neighbours; one expansion
               pass admits p in R \\ core with tau > 0 and S(p, core) >= the lower
               quartile of the core's leave-one-out scores; candidates with reciprocal
               overlap >= 0.8 are merged; candidates with < m proteins are dropped.

Internal objective (no reference input):

    coh(D)  = (1/|R|) sum_{p in R} max_{C in D, p in C} [ 0.5 dens(C) + 0.5 S(p, C) ]
    sep(D)  = size-weighted mean conductance of the parts within G[R] (0 for D = {R})
    frag(D) = (|D| - 1) / max(1, floor(|R| / m) - 1)
    Q(D)    = coh(D) - lambda_sep * sep(D) - lambda_frag * frag(D)

The best split alternative is accepted only if Q(split) - Q(no_split) > delta; ties
between split alternatives are broken by fewer parts, then by the order
spectral < seed_expand.  Accepted children are re-examined recursively when they
are still eligible and flagged, up to ``max_depth`` levels.
"""
from __future__ import annotations

from typing import Dict, List, Mapping, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from .channels import Evidence
from .heterogeneity import conductance, flag, region_signals


def _local_graph(region: Set[str], ev: Evidence) -> nx.Graph:
    H = nx.Graph()
    H.add_nodes_from(sorted(region))
    H.add_edges_from((p, q) for p in region for q in ev.adj[p] & region if p < q)
    return H


def spectral_split(region: Set[str], ev: Evidence, m: int) -> List[Set[str]] | None:
    H = _local_graph(region, ev)
    comps = [set(c) for c in nx.connected_components(H)]
    if len(comps) > 1:
        parts = sorted((c for c in comps if len(c) >= m), key=lambda c: (-len(c), min(c)))
        return parts if len(parts) >= 2 else None
    nodes = sorted(region)
    L = nx.normalized_laplacian_matrix(H, nodelist=nodes).toarray()
    vals, vecs = np.linalg.eigh(L)
    f = vecs[:, 1]
    # Fix the eigenvector sign deterministically (largest |entry| positive).
    if f[np.argmax(np.abs(f))] < 0:
        f = -f
    left = {nodes[i] for i, x in enumerate(f) if x <= 0}
    if min(len(left), len(region) - len(left)) < m:
        med = float(np.median(f))
        left = {nodes[i] for i, x in enumerate(f) if x <= med}
    right = set(region) - left
    if min(len(left), len(right)) < m:
        return None
    return sorted([left, right], key=lambda c: (-len(c), min(c)))


def _merge(cands: List[Set[str]], overlap: float) -> List[Set[str]]:
    out: List[Set[str]] = []
    for c in sorted(cands, key=lambda x: (-len(x), sorted(x))):
        for i, prev in enumerate(out):
            inter = len(c & prev)
            if min(inter / len(c), inter / len(prev)) >= overlap:
                out[i] = prev | c
                break
        else:
            out.append(set(c))
    return out


def seed_expand(region: Set[str], ev: Evidence, m: int, cfg: dict) -> List[Set[str]] | None:
    H = _local_graph(region, ev)
    core_no = nx.core_number(H)
    deg = dict(H.degree())
    clu = nx.clustering(H)
    score = {n: 0.0 for n in H}
    for values in (core_no, deg, clu):
        ranks = pd.Series(values, dtype=float).rank(method="average", pct=True)
        for n in H:
            score[n] += float(ranks[n]) / 3.0
    order = sorted(H, key=lambda n: (-score[n], n))
    k_max = max(2, int(np.floor(np.sqrt(len(region)))))
    c = int(cfg.get("seed_core_neighbours", 4))
    cores: List[Set[str]] = []
    covered: Set[str] = set()
    for s in order:
        if len(cores) >= k_max:
            break
        if s in covered:
            continue
        nb = sorted(H.neighbors(s), key=lambda v: (-len(set(H.neighbors(v)) & set(H.neighbors(s))), -score[v], v))
        core = {s, *nb[:c]}
        if len(core) >= m:
            cores.append(core)
            covered |= core
    if not cores:
        return None
    q = 0.25
    expanded = []
    for core in cores:
        st = ev.module(core)
        inc = [st.scores(p)["S"] for p in core]
        theta = float(np.quantile(inc, q))
        cand = set(core)
        for p in sorted(region - core):
            s = st.scores(p)
            if s["tau"] > 0 and s["S"] >= theta:
                cand.add(p)
        expanded.append(cand)
    merged = [x for x in _merge(expanded, float(cfg.get("merge_reciprocal_overlap", 0.8))) if len(x) >= m]
    return merged or None


def objective(parts: List[Set[str]], region: Set[str], ev: Evidence, m: int, cfg: dict) -> Dict[str, float]:
    states = [ev.module(c) for c in parts]
    dens = [s.density() for s in states]
    best = {}
    for c, st, d in zip(parts, states, dens):
        for p in c:
            val = 0.5 * d + 0.5 * st.scores(p)["S"]
            if val > best.get(p, -1.0):
                best[p] = val
    coh = sum(best.values()) / len(region)
    k = len(parts)
    if k == 1 and parts[0] == set(region):
        sep = 0.0
    else:
        sizes = np.array([len(c) for c in parts], dtype=float)
        phis = np.array([conductance(ev, c, region) for c in parts])
        sep = float((sizes * phis).sum() / sizes.sum())
    frag = (k - 1) / max(1, int(np.floor(len(region) / m)) - 1)
    lam_s = float(cfg.get("lambda_separation", 0.5))
    lam_f = float(cfg.get("lambda_fragmentation", 0.25))
    return {"k": k, "coherence": coh, "separation_penalty": sep, "fragmentation_penalty": frag,
            "coverage": len(best) / len(region), "Q": coh - lam_s * sep - lam_f * frag}


def choose(region: Set[str], ev: Evidence, m: int, cfg: dict, alternatives: List[str], forced: str | None = None):
    proposals: Dict[str, List[Set[str]] | None] = {"no_split": [set(region)]}
    if "spectral" in alternatives or forced == "spectral":
        proposals["spectral"] = spectral_split(region, ev, m)
    if "seed_expand" in alternatives or forced == "seed_expand":
        proposals["seed_expand"] = seed_expand(region, ev, m, cfg)
    scored = {}
    for name, parts in proposals.items():
        scored[name] = objective(parts, region, ev, m, cfg) if parts else None
    delta = float(cfg.get("min_improvement", 0.02))
    order = {"no_split": 0, "spectral": 1, "seed_expand": 2}
    if forced:
        chosen = forced if proposals.get(forced) else "no_split"
        reason = "forced_mechanism" if chosen == forced else "forced_mechanism_unavailable"
    else:
        splits = [n for n in proposals if n != "no_split" and scored[n] is not None]
        if not splits:
            chosen, reason = "no_split", "no_valid_split_alternative"
        else:
            best = sorted(splits, key=lambda n: (-round(scored[n]["Q"], 12), scored[n]["k"], order[n]))[0]
            gain = scored[best]["Q"] - scored["no_split"]["Q"]
            if gain > delta:
                chosen, reason = best, f"split_gain_{gain:.4f}_exceeds_delta"
            else:
                chosen, reason = "no_split", f"split_gain_{gain:.4f}_not_above_delta"
    return chosen, proposals[chosen], scored, reason


def deconvolve(region: Set[str], ev: Evidence, thresholds: Mapping[str, float], het_cfg: dict, dec_cfg: dict,
               m: int, min_split_region: int, region_id: int, alternatives: List[str], forced: str | None = None,
               depth: int = 0, path: str = "") -> Tuple[List[Tuple[Set[str], str, int]], List[dict]]:
    """Return leaf candidates as (members, method_path, depth) and audit rows."""
    chosen, parts, scored, reason = choose(region, ev, m, dec_cfg, alternatives, forced)
    audit = []
    for name, sc in scored.items():
        audit.append({"region_id": region_id, "node_path": path or "root", "depth": depth, "size": len(region),
                      "alternative": name, "available": sc is not None, "selected": name == chosen,
                      "decision_reason": reason, **(sc or {})})
    if chosen == "no_split":
        return [(set(region), path or "no_split", depth)], audit
    leaves = []
    for i, child in enumerate(parts):
        child_path = f"{path + '/' if path else ''}{chosen}[{i}]"
        recurse = False
        if depth + 1 < int(dec_cfg.get("max_depth", 2)) and len(child) >= min_split_region:
            sig = region_signals(child, ev)
            recurse = flag(sig, thresholds, het_cfg, min_split_region)["flagged"]
        if recurse:
            sub, sub_audit = deconvolve(child, ev, thresholds, het_cfg, dec_cfg, m, min_split_region, region_id,
                                        alternatives, forced, depth + 1, child_path)
            leaves += sub
            audit += sub_audit
        else:
            leaves.append((set(child), child_path, depth + 1))
    return leaves, audit
