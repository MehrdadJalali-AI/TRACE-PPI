"""Network-perturbation stability (diagnostic; adopted as a gate only by the predeclared rule).

Replicate b removes floor(rate * |E|) edges chosen uniformly without replacement
(numpy default_rng(seed + b), edges in sorted order), reruns the full pipeline
(MCL included) on the perturbed graph and aligns modules.  An accepted membership
(p, M) is recovered in replicate b when some perturbed module M' contains p and
J(M \\ {p}, M' \\ {p}) >= alignment_jaccard; excluding p prevents the evaluated
protein from supporting its own alignment.  Stability = recovered fraction.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Set

import networkx as nx
import numpy as np
import pandas as pd

from .pipeline import TraceResult, run_trace


def perturbed_graph(G: nx.Graph, rate: float, seed: int) -> nx.Graph:
    edges = sorted(tuple(sorted((str(u), str(v)))) for u, v in G.edges())
    rng = np.random.default_rng(seed)
    drop = set(map(int, rng.choice(len(edges), int(np.floor(rate * len(edges))), replace=False)))
    H = nx.Graph()
    H.add_nodes_from(G.nodes())
    H.add_edges_from(e for i, e in enumerate(edges) if i not in drop)
    return H


def _j(a: Set[str], b: Set[str]) -> float:
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def stability(ds, cfg, base: TraceResult, rate: float, B: int, seed: int = 1000, align: float = 0.5,
              variant: dict | None = None):
    acc = base.accepted[["protein_id", "module_id"]].drop_duplicates()
    base_mods = base.modules
    hits = defaultdict(int)
    split_children = {}
    for rid, grp in base.ledger[(base.ledger.region_split == True) & (base.ledger.status == "accepted")].groupby("region_id"):  # noqa: E712
        split_children[int(rid)] = [base_mods[m] for m in sorted(grp.module_id.unique()) if m in base_mods]
    split_recovered = {rid: 0 for rid in split_children}
    runs = []
    for b in range(B):
        H = perturbed_graph(ds.graph, rate, seed + b)
        r = run_trace(ds, cfg, variant=variant, graph=H)
        by_p: Dict[str, List[Set[str]]] = defaultdict(list)
        for mem in r.modules.values():
            for p in mem:
                by_p[p].append(mem)
        for p, mid in acc.itertuples(index=False):
            ref = base_mods[mid] - {p}
            if any(_j(ref, m - {p}) >= align for m in by_p.get(p, ())):
                hits[(p, mid)] += 1
        pert = list(r.modules.values())
        for rid, kids in split_children.items():
            matched = {i for i, k in enumerate(kids) if any(_j(k, m) >= align for m in pert)}
            split_recovered[rid] += int(len(matched) >= 2)
        runs.append({"replicate": b, "rate": rate, "n_modules": len(r.modules), "n_flagged": int(r.regions.flagged.sum()),
                     "n_split_regions": int(r.ledger[r.ledger.region_split == True].region_id.nunique())})  # noqa: E712
    mem = acc.copy()
    mem["stability"] = [hits[(p, m)] / B for p, m in zip(mem.protein_id, mem.module_id)]
    mem["rate"] = rate
    splits = pd.DataFrame([{"region_id": rid, "rate": rate, "split_recovery": v / B} for rid, v in split_recovered.items()]) \
        if split_recovered else pd.DataFrame(columns=["region_id", "rate", "split_recovery"])
    return mem, pd.DataFrame(runs), splits
