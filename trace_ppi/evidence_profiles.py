"""Build multi-evidence protein profiles."""
from __future__ import annotations

from typing import Dict, List, Set

import networkx as nx
import numpy as np
import pandas as pd

from .reuse import GENERIC_GO
from .graph_io import text_profile


def _minmax(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def build_profiles(
    graph: nx.Graph,
    go_map: Dict[str, Set[str]],
    dataset: str,
    alias_map: Dict[str, str] | None = None,
) -> pd.DataFrame:
    nodes = [str(n) for n in graph.nodes()]
    kcore = nx.core_number(graph)
    rows: List[dict] = []
    for p in nodes:
        terms = go_map.get(p, go_map.get(int(p) if p.isdigit() else p, set()))
        if not isinstance(terms, set):
            terms = set()
        non_generic = terms - GENERIC_GO
        neighbors = [str(n) for n in graph.neighbors(int(p) if p.isdigit() and int(p) in graph else p)]
        node_key = int(p) if p.isdigit() and int(p) in graph else p
        wdeg = sum(graph[node_key][n].get("weight", 1.0) for n in graph.neighbors(node_key))
        lc = nx.clustering(graph, node_key, weight="weight") if graph.degree(node_key) else 0.0
        aliases = (alias_map or {}).get(p, "")
        richness = len(non_generic) + (1 if wdeg > 0 else 0)
        uncertainty = 1.0 if not non_generic else max(0.0, 1.0 - min(1.0, len(non_generic) / 5.0))
        rows.append(
            {
                "protein_id": p,
                "dataset": dataset,
                "aliases": aliases,
                "go_terms": ";".join(sorted(terms)),
                "go_term_names_if_available": "",
                "go_namespaces": "",
                "non_generic_go_terms": ";".join(sorted(non_generic)),
                "neighbor_ids": ";".join(neighbors[:200]),
                "weighted_degree": wdeg,
                "local_clustering": lc,
                "k_core_score": kcore.get(node_key, 0),
                "text_profile": text_profile(p, non_generic or terms, aliases),
                "evidence_richness": richness,
                "annotation_uncertainty": uncertainty,
            }
        )
    df = pd.DataFrame(rows)
    for col in ("weighted_degree", "local_clustering", "k_core_score", "evidence_richness"):
        df[f"{col}_norm"] = _minmax(df[col].astype(float).values)
    df["annotation_uncertainty_norm"] = _minmax(df["annotation_uncertainty"].astype(float).values)
    return df
