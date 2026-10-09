"""Map STRING protein IDs to SGD systematic locus names (Y-format)."""

import gzip
import re
from pathlib import Path
from typing import Dict

LOCUS_RE = re.compile(r'^Y[A-Z][A-Z][0-9]+[A-Z]$')


def load_string_to_locus(aliases_gz: str) -> Dict[str, str]:
    """Build mapping 4932.YAL001C -> YAL001C from STRING aliases file."""
    mapping: Dict[str, str] = {}
    path = Path(aliases_gz)
    open_fn = gzip.open if str(path).endswith('.gz') else open
    with open_fn(path, 'rt', encoding='utf-8', errors='replace') as f:
        next(f, None)
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            string_id, alias, _source = parts[0], parts[1], parts[2]
            if LOCUS_RE.match(alias):
                mapping.setdefault(string_id, alias)
            elif string_id.startswith('4932.') and LOCUS_RE.match(string_id.split('.', 1)[1]):
                mapping.setdefault(string_id, string_id.split('.', 1)[1])
    return mapping


def remap_graph_nodes(graph, string_to_locus: Dict[str, str]):
    """Relabel NetworkX graph nodes to systematic locus names where possible."""
    import networkx as nx

    relabel = {}
    for node in graph.nodes():
        if node in string_to_locus:
            relabel[node] = string_to_locus[node]
        elif isinstance(node, str) and node.startswith('4932.'):
            suffix = node.split('.', 1)[1]
            relabel[node] = suffix if LOCUS_RE.match(suffix) else node
        else:
            relabel[node] = node
    return nx.relabel_nodes(graph, relabel, copy=True)
