"""
Baseline runners exporting a common assignment format:
community_id, protein_id, membership_score, method, dataset, seed
"""

import logging
from typing import Dict, List, Optional, Set, Tuple
import networkx as nx
import pandas as pd

from src.mcl_clustering import MCLClustering
from src.permanence import calculate_permanence_all_proteins
from src.membership_overlap import (
    apply_overlap_reassignment,
    calculate_functional_dependency,
    calculate_membership,
    build_all_fd_scores,
)
from src.go_tfidf import GOTFIDF
from src.lea.optimize import optimize_communities

logger = logging.getLogger(__name__)

BASELINE_METHODS = [
    'mcl_only',
    'mcl_overlap_heuristic',
    'mcl_permanence_only',
    'mcl_go_only',
    'leaf_ppi_no_lea',
    'leaf_ppi_full',
]


def clusters_to_rows(
    clusters: Dict[int, Set[str]],
    method: str,
    dataset: str,
    seed: int,
    membership_scores: Optional[Dict[Tuple[str, int], float]] = None,
) -> List[dict]:
    rows = []
    for cid, members in clusters.items():
        for protein in members:
            score = 1.0
            if membership_scores is not None:
                score = membership_scores.get((protein, cid), 1.0)
            rows.append({
                'community_id': cid,
                'protein_id': protein,
                'membership_score': score,
                'method': method,
                'dataset': dataset,
                'seed': seed,
            })
    return rows


def export_assignments(rows: List[dict], path: str) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _mcl_clusters(graph: nx.Graph, inflation: float = 2.0) -> Dict[int, Set[str]]:
    return MCLClustering(inflation=inflation).cluster(graph)


def run_baseline(
    method: str,
    graph: nx.Graph,
    dataset: str,
    seed: int = 42,
    protein_go_terms: Optional[Dict[str, Set[str]]] = None,
    go_tfidf: Optional[GOTFIDF] = None,
    mcl_inflation: float = 2.0,
    alpha: float = 0.5,
    overlap_tau: float = 0.1,
    transfer_tau: float = 0.0,
    lea_evaluations: int = 500,
    lea_population: int = 30,
) -> Tuple[Dict[int, Set[str]], List[dict]]:
    """
    Run one baseline variant and return clusters plus export rows.
    """
    if method not in BASELINE_METHODS:
        raise ValueError(f'Unknown baseline: {method}')

    initial = _mcl_clusters(graph, mcl_inflation)
    protein_go_terms = protein_go_terms or {}
    go_tfidf = go_tfidf or GOTFIDF(initial, protein_go_terms)
    permanence_scores = calculate_permanence_all_proteins(initial, graph)
    fd_scores = build_all_fd_scores(initial, protein_go_terms, go_tfidf)

    if method == 'mcl_only':
        return initial, clusters_to_rows(initial, method, dataset, seed)

    if method == 'mcl_overlap_heuristic':
        # Topology-only overlap: alpha=1, ignore GO
        clusters = apply_overlap_reassignment(
            initial, graph, protein_go_terms, go_tfidf, permanence_scores,
            alpha=1.0, overlap_tau=overlap_tau, transfer_tau=transfer_tau,
            fd_scores=fd_scores, normalize_scores=True,
        )
        return clusters, clusters_to_rows(clusters, method, dataset, seed)

    if method == 'mcl_permanence_only':
        clusters = apply_overlap_reassignment(
            initial, graph, protein_go_terms, go_tfidf, permanence_scores,
            alpha=1.0, overlap_tau=overlap_tau, transfer_tau=transfer_tau,
            fd_scores=fd_scores, normalize_scores=True,
        )
        return clusters, clusters_to_rows(clusters, method, dataset, seed)

    if method == 'mcl_go_only':
        clusters = apply_overlap_reassignment(
            initial, graph, protein_go_terms, go_tfidf, permanence_scores,
            alpha=0.0, overlap_tau=overlap_tau, transfer_tau=transfer_tau,
            fd_scores=fd_scores, normalize_scores=True,
        )
        return clusters, clusters_to_rows(clusters, method, dataset, seed)

    if method == 'leaf_ppi_no_lea':
        clusters = apply_overlap_reassignment(
            initial, graph, protein_go_terms, go_tfidf, permanence_scores,
            alpha=alpha, overlap_tau=overlap_tau, transfer_tau=transfer_tau,
            fd_scores=fd_scores, normalize_scores=True,
        )
        return clusters, clusters_to_rows(clusters, method, dataset, seed)

    if method == 'leaf_ppi_full':
        best_solution, _, clusters = optimize_communities(
            graph, initial, protein_go_terms, go_tfidf, permanence_scores,
            population_size=lea_population,
            max_evaluations=lea_evaluations,
            random_seed=seed,
        )
        opt_alpha = float(best_solution[0])
        memb = {}
        for cid, members in clusters.items():
            for p in members:
                memb[(p, cid)] = calculate_membership(
                    p, members, cid, graph, protein_go_terms, go_tfidf,
                    permanence_scores, opt_alpha, fd_scores=fd_scores, normalize_scores=True,
                )
        return clusters, clusters_to_rows(clusters, method, dataset, seed, memb)

    raise RuntimeError(method)
