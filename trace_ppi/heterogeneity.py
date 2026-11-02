"""Three-signal heterogeneity detection for coarse MCL regions (no reference input).

For a region R with induced subgraph H = G[R]:

* topology T(R)  = (Q(P) + mean_{C in P} (1 - phi_H(C))) / 2, where P is the
  greedy-modularity (Clauset-Newman-Moore) partition of H, Q its modularity and
  phi_H the conductance of each part within H; T = 0 when |P| < 2.
* semantic D(R)  = 1 - mean pairwise cosine of embeddings of annotated members
  (needs >= 3 annotated members, otherwise missing).
* support V(R)   = interquartile range of tau(p, R) over members (needs >= 4).

Each signal is compared with the upper quartile (by default) of that signal over
all eligible regions of the same network (|R| >= min_region_size_for_split); a
region is flagged when at least ``min_votes`` available signals reach their
thresholds.  These definitions were fixed before any reference-based evaluation;
the threshold population is the eligible regions, not all regions,
regions because ineligible regions carry structural zeros.
"""
from __future__ import annotations

from typing import Dict, Iterable, Mapping, Set

import networkx as nx
import numpy as np
import pandas as pd

from .channels import Evidence

SIGNALS = ("topology", "semantic", "support")


def conductance(ev: Evidence, part: Set[str], region: Set[str]) -> float:
    rest = region - part
    cut = sum(len(ev.adj[p] & rest) for p in part)
    vol_part = sum(len(ev.adj[p] & region) for p in part)
    vol_rest = sum(len(ev.adj[p] & region) for p in rest)
    if vol_part == 0:
        return 1.0
    if vol_rest == 0:
        return 0.0
    return cut / min(vol_part, vol_rest)


def region_signals(region: Set[str], ev: Evidence) -> Dict[str, float]:
    region = set(region)
    out = {"n": len(region), "topology": np.nan, "semantic": np.nan, "support": np.nan,
           "modularity": np.nan, "mean_part_separation": np.nan, "n_modularity_parts": 0}
    edges = [(p, q) for p in region for q in ev.adj[p] & region if p < q]
    if len(region) >= 3:
        H = nx.Graph()
        H.add_nodes_from(sorted(region))
        H.add_edges_from(edges)
        if H.number_of_edges():
            parts = [set(c) for c in nx.community.greedy_modularity_communities(H)]
            out["n_modularity_parts"] = len(parts)
            if len(parts) >= 2:
                q = float(nx.community.modularity(H, parts))
                sep = float(np.mean([1.0 - conductance(ev, c, region) for c in parts]))
                out.update(modularity=q, mean_part_separation=sep, topology=(q + sep) / 2.0)
            else:
                out.update(modularity=0.0, mean_part_separation=0.0, topology=0.0)
        else:
            out.update(topology=0.0)
    annotated = sorted(p for p in region if p in ev.annotated)
    if len(annotated) >= 3:
        arr = np.vstack([ev.emb[p] for p in annotated])
        arr = arr / np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), 1e-12)
        sims = arr @ arr.T
        out["semantic"] = float(1.0 - sims[np.triu_indices_from(sims, 1)].mean())
    if len(region) >= 4:
        state = ev.module(region)
        taus = [state.scores(p)["tau"] for p in region]
        out["support"] = float(np.quantile(taus, 0.75) - np.quantile(taus, 0.25))
    return out


def fit_thresholds(table: pd.DataFrame, cfg: dict, min_size: int) -> Dict[str, float]:
    pop = table[table.n >= min_size] if cfg.get("population", "eligible") == "eligible" else table
    q = float(cfg.get("quantile", 0.75))
    return {s: (float(pop[s].dropna().quantile(q)) if pop[s].notna().any() else np.inf) for s in SIGNALS}


def flag(signals: Mapping[str, float], thresholds: Mapping[str, float], cfg: dict, min_size: int) -> Dict[str, object]:
    active = cfg.get("signals", list(SIGNALS))
    votes = {}
    for s in SIGNALS:
        v = signals.get(s, np.nan)
        votes[s] = bool(s in active and not np.isnan(v) and v >= thresholds[s])
    n_votes = sum(votes.values())
    need = min(int(cfg.get("min_votes", 2)), len(active))
    eligible = signals["n"] >= min_size
    return {**{f"{s}_flag": votes[s] for s in SIGNALS}, "votes": n_votes, "eligible": eligible,
            "flagged": bool(eligible and n_votes >= need)}


def assess_regions(regions: Mapping[int, Set[str]], ev: Evidence, cfg: dict, min_size: int):
    rows = []
    for rid in sorted(regions):
        sig = region_signals(regions[rid], ev) if len(regions[rid]) >= 2 else {
            "n": len(regions[rid]), "topology": np.nan, "semantic": np.nan, "support": np.nan,
            "modularity": np.nan, "mean_part_separation": np.nan, "n_modularity_parts": 0}
        rows.append({"region_id": rid, **sig})
    table = pd.DataFrame(rows)
    thresholds = fit_thresholds(table, cfg, min_size)
    flags = [flag(r, thresholds, cfg, min_size) for r in table.to_dict("records")]
    table = pd.concat([table, pd.DataFrame(flags)], axis=1)
    for s in SIGNALS:
        table[f"{s}_threshold"] = thresholds[s]
    return table, thresholds
