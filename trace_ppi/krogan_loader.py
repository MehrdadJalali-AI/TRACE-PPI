"""Krogan 2006 yeast PPI loader using local BioGRID TAB3 records."""
from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Set, Tuple

import networkx as nx
import pandas as pd

from .paths import DATA, RESULTS
from .reuse import GENERIC_GO, GOLoader

KROGAN_PUBMED = "PUBMED:16554755"
YEAST_TAXID = "559292"
YEAST_ORF_RE = re.compile(r"^Y[A-P][LR][0-9]{3}[CW](?:-[A-Z])?$")


def _valid_orf(value: str) -> bool:
    return bool(value and YEAST_ORF_RE.match(value.strip()))


def _score_to_weight(value: str) -> float:
    try:
        if value and value != "-":
            return float(value)
    except ValueError:
        pass
    return 1.0


def load_krogan_from_biogrid(
    biogrid_path: Path | None = None,
    go_path: Path | None = None,
    write_reports: bool = True,
) -> Tuple[nx.Graph, Dict[str, Set[str]], dict]:
    """Load the Krogan Nature 2006 yeast interaction set from BioGRID.

    The local BioGRID TAB3 file contains the Krogan et al. Nature 2006 records
    as ``PUBMED:16554755``.  We keep physical yeast--yeast interactions, map
    interactors to SGD systematic ORF identifiers, remove self-loops and
    duplicate undirected edges, and preserve numeric scores when BioGRID
    provides them.  Krogan/BioGRID records in the local file mostly have no
    numeric confidence score, so those edges receive weight 1.0.
    """
    biogrid_path = Path(biogrid_path or DATA["krogan_biogrid"])
    go_path = Path(go_path or DATA["go_txt"])
    out_dir = RESULTS / "krogan"
    out_dir.mkdir(parents=True, exist_ok=True)

    graph = nx.Graph()
    raw_rows = 0
    publication_rows = 0
    physical_rows = 0
    duplicate_edges = 0
    self_loops = 0
    invalid_identifier_rows = 0
    unmapped = Counter()
    seen_edges = set()

    with biogrid_path.open(errors="replace", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            raw_rows += 1
            if row.get("Publication Source") != KROGAN_PUBMED:
                continue
            publication_rows += 1
            if row.get("Experimental System Type") != "physical":
                continue
            if row.get("Organism ID Interactor A") != YEAST_TAXID or row.get("Organism ID Interactor B") != YEAST_TAXID:
                continue
            physical_rows += 1

            a = (row.get("Systematic Name Interactor A") or "").strip()
            b = (row.get("Systematic Name Interactor B") or "").strip()
            if not _valid_orf(a):
                unmapped[a or row.get("Official Symbol Interactor A", "")] += 1
            if not _valid_orf(b):
                unmapped[b or row.get("Official Symbol Interactor B", "")] += 1
            if not (_valid_orf(a) and _valid_orf(b)):
                invalid_identifier_rows += 1
                continue
            if a == b:
                self_loops += 1
                continue

            edge = tuple(sorted((a, b)))
            weight = _score_to_weight(row.get("Score", ""))
            if edge in seen_edges:
                duplicate_edges += 1
                if graph[edge[0]][edge[1]].get("weight", 1.0) < weight:
                    graph[edge[0]][edge[1]]["weight"] = weight
                continue
            seen_edges.add(edge)
            graph.add_edge(edge[0], edge[1], weight=weight, source="Krogan2006_BioGRID")

    graph.remove_edges_from(nx.selfloop_edges(graph))

    go_raw = GOLoader().load_from_gaf(str(go_path), taxid=559292, use_symbol=True)
    nodes = set(graph.nodes())
    go = {p: set(go_raw.get(p, set())) - GENERIC_GO for p in nodes}
    go_nodes = sum(1 for p in nodes if go.get(p))

    edge_rows = [
        {"protein_a": u, "protein_b": v, "weight": float(data.get("weight", 1.0))}
        for u, v, data in sorted(graph.edges(data=True))
    ]
    pd.DataFrame(edge_rows).to_csv(out_dir / "cleaned_graph_edges.csv", index=False)

    report = {
        "dataset": "krogan",
        "source_file": str(biogrid_path),
        "publication_filter": KROGAN_PUBMED,
        "raw_biogrid_rows": raw_rows,
        "publication_rows": publication_rows,
        "physical_yeast_rows": physical_rows,
        "invalid_identifier_rows": invalid_identifier_rows,
        "self_loops_removed": self_loops,
        "duplicate_edges_removed": duplicate_edges,
        "clean_nodes": graph.number_of_nodes(),
        "clean_edges": graph.number_of_edges(),
        "edge_weight_policy": "numeric BioGRID Score when available, otherwise 1.0",
        "go_nodes_with_nongeneric_terms": go_nodes,
        "go_coverage": go_nodes / graph.number_of_nodes() if graph.number_of_nodes() else 0.0,
    }

    if write_reports:
        pd.DataFrame([report]).to_csv(out_dir / "cleaning_report.csv", index=False)
        pd.DataFrame(
            [
                {
                    "dataset": "krogan",
                    "mapped_nodes": graph.number_of_nodes(),
                    "unmapped_identifier_rows": invalid_identifier_rows,
                    "identifier_policy": "Systematic Name Interactor A/B must be SGD ORF-like identifiers",
                }
            ]
        ).to_csv(out_dir / "id_mapping_report.csv", index=False)
        pd.DataFrame(
            [
                {
                    "dataset": "krogan",
                    "nodes": graph.number_of_nodes(),
                    "nodes_with_nongeneric_go": go_nodes,
                    "go_coverage": report["go_coverage"],
                    "generic_roots_removed": ";".join(sorted(GENERIC_GO)),
                }
            ]
        ).to_csv(out_dir / "go_coverage_report.csv", index=False)
        pd.DataFrame(
            [{"identifier": key, "count": value} for key, value in unmapped.most_common()]
        ).to_csv(out_dir / "unmapped_proteins.csv", index=False)

    return graph, go, report
