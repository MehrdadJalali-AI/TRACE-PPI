"""Load PPI graphs and GO annotations for COSMOS-PPI."""
from __future__ import annotations

import logging
from typing import Dict, Set, Tuple

import networkx as nx

from .paths import DATA
from .reuse import GavinLoader, STRINGLoader, GOLoader, load_string_to_locus, remap_graph_nodes, GENERIC_GO
from .krogan_loader import load_krogan_from_biogrid

logger = logging.getLogger(__name__)


def load_gavin() -> Tuple[nx.Graph, Dict[str, Set[str]]]:
    g = GavinLoader(normalize=True).load(str(DATA["gavin_ppi"]))
    go = GOLoader().load_from_gaf(str(DATA["go_txt"]), taxid=559292, use_symbol=True)
    nodes = set(g.nodes())
    go = {p: terms for p, terms in go.items() if p in nodes}
    return g, go


def load_krogan() -> Tuple[nx.Graph, Dict[str, Set[str]]]:
    try:
        g, go, _ = load_krogan_from_biogrid(write_reports=True)
        return g, go
    except FileNotFoundError:
        logger.warning("BioGRID TAB3 file missing; reconstructing Krogan graph from cleaned edges.")
        from .paths import RESULTS
        import pandas as pd

        edge_path = RESULTS / "krogan" / "cleaned_graph_edges.csv"
        if not edge_path.exists():
            raise
        edges = pd.read_csv(edge_path)
        g = nx.Graph()
        for _, row in edges.iterrows():
            g.add_edge(str(row["protein_a"]), str(row["protein_b"]), weight=float(row.get("weight", 1.0)))
        go_raw = GOLoader().load_from_gaf(str(DATA["go_txt"]), taxid=559292, use_symbol=True)
        nodes = set(g.nodes())
        go = {p: terms for p, terms in go_raw.items() if p in nodes}
        return g, go


def load_string(threshold: int = 700) -> Tuple[nx.Graph, Dict[str, Set[str]]]:
    loader = STRINGLoader(taxid=4932, cache_dir=str(DATA["string_aliases"].parent), threshold=threshold)
    g, _ = loader.load_from_download(data_dir=str(REPO_ROOT := DATA["string_ppi"].parent))
    g.remove_edges_from(nx.selfloop_edges(g))
    go_raw = GOLoader().load_from_gaf(str(DATA["go_txt"]), taxid=559292, use_symbol=True)
    if DATA["string_aliases"].exists():
        locus_map = load_string_to_locus(str(DATA["string_aliases"]))
        g = remap_graph_nodes(g, locus_map)
    go: Dict[str, Set[str]] = {}
    for node in g.nodes():
        sid = f"4932.{node}" if not str(node).startswith("4932.") else node
        if node in go_raw:
            go[node] = go_raw[node]
        elif sid in go_raw:
            go[node] = go_raw[sid]
    return g, go


def filter_go(go: Dict[str, Set[str]], drop_generic: bool = True) -> Dict[str, Set[str]]:
    out = {}
    for p, terms in go.items():
        t = set(terms)
        if drop_generic:
            t -= GENERIC_GO
        out[p] = t
    return out


def text_profile(protein: str, go_terms: Set[str], aliases: str = "") -> str:
    parts = [protein]
    if aliases:
        parts.append(aliases)
    if go_terms:
        parts.append(" ".join(sorted(go_terms)))
    return " ".join(parts)
