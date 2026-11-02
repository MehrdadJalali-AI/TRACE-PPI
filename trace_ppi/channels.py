"""Assignment-level evidence channels.

For protein p and candidate module C, with C' = C \\ {p}:

* tau(p, C)   = |N(p) & C'| / |C'|                         (topology, in [0, 1])
* sigma(p, C) = max(0, cos(e_p, mean_{q in C', A(q) != {}} e_q))  (semantic, in [0, 1])
* g(p, C)     = |A(p) & U_{q in C'} A(q)| / |A(p)|          (GO, in [0, 1])
* S(p, C)     = w_tau * tau + w_sigma * sigma + w_g * g

Edge cases: |C'| = 0 gives tau = sigma = g = 0.  If A(p) is empty (no non-generic
GO annotation), sigma and g are 0 and the row is flagged ``annotation_missing``;
missing annotation is treated as absent evidence, not as evidence of absence.
Negative cosines are clipped to 0 (no semantic support).  sigma and g both derive
from GO annotations and are therefore related channels, not independent evidence.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, Mapping, Set

import networkx as nx
import numpy as np


class Evidence:
    def __init__(self, graph: nx.Graph, go: Mapping[str, Set[str]], embeddings: Mapping[str, np.ndarray],
                 weights: Mapping[str, float]):
        self.adj: Dict[str, Set[str]] = {str(n): {str(x) for x in graph.neighbors(n) if x != n} for n in graph.nodes()}
        self.go = {str(p): set(t) for p, t in go.items()}
        self.emb = {str(p): np.asarray(v, dtype=np.float64) for p, v in embeddings.items()}
        self.annotated = {p for p, t in self.go.items() if t}
        total = sum(max(0.0, float(weights[k])) for k in ("topology", "semantic", "go"))
        self.w = {k: max(0.0, float(weights[k])) / total for k in ("topology", "semantic", "go")} if total else \
            {"topology": 0.0, "semantic": 0.0, "go": 0.0}

    def module(self, members: Iterable[str]) -> "ModuleState":
        return ModuleState(self, members)


class ModuleState:
    """Cached sums for fast leave-one-out channel scores against a fixed member set."""

    def __init__(self, ev: Evidence, members: Iterable[str]):
        self.ev = ev
        self.members = frozenset(map(str, members))
        dim = len(next(iter(ev.emb.values()))) if ev.emb else 0
        self.emb_sum = np.zeros(dim)
        self.n_annotated = 0
        self.term_counts: Counter = Counter()
        for q in self.members:
            if q in ev.annotated:
                self.emb_sum += ev.emb[q]
                self.n_annotated += 1
                self.term_counts.update(ev.go[q])

    def scores(self, p: str) -> Dict[str, float]:
        ev = self.ev
        inside = p in self.members
        others = len(self.members) - (1 if inside else 0)
        missing = p not in ev.annotated
        if others <= 0:
            return {"tau": 0.0, "sigma": 0.0, "go": 0.0, "S": 0.0, "n_neighbours": 0, "annotation_missing": missing}
        nbrs = len(ev.adj.get(p, set()) & self.members)
        tau = nbrs / others
        sigma = 0.0
        g = 0.0
        if not missing:
            vec = self.emb_sum - (ev.emb[p] if inside else 0.0)
            n_ann = self.n_annotated - (1 if inside else 0)
            if n_ann > 0:
                norm = float(np.linalg.norm(vec))
                if norm > 0:
                    sigma = max(0.0, float(np.dot(ev.emb[p], vec)) / (norm * max(float(np.linalg.norm(ev.emb[p])), 1e-12)))
            terms = ev.go[p]
            shared = sum(1 for t in terms if self.term_counts.get(t, 0) - (1 if inside else 0) > 0)
            g = shared / len(terms)
        w = ev.w
        S = w["topology"] * tau + w["semantic"] * sigma + w["go"] * g
        return {"tau": tau, "sigma": min(1.0, sigma), "go": g, "S": S, "n_neighbours": nbrs,
                "annotation_missing": missing}

    def shared_terms(self, p: str, top: int = 5) -> str:
        """GO terms of p also carried by another member, most frequent first."""
        inside = p in self.members
        terms = [(self.term_counts.get(t, 0) - (1 if inside else 0), t) for t in self.ev.go.get(p, ())]
        terms = sorted((c, t) for c, t in terms if c > 0)
        terms.sort(key=lambda x: (-x[0], x[1]))
        return ";".join(t for _, t in terms[:top])

    def density(self) -> float:
        n = len(self.members)
        if n < 2:
            return 0.0
        e = sum(len(self.ev.adj.get(p, set()) & self.members) for p in self.members) / 2
        return e / (n * (n - 1) / 2)
