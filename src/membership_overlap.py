"""
Membership and functional dependency calculations.
Implements functional dependency fd (Eq.2) and normalized Membership (Eq.4).
Handles overlapping community assignment.
"""

import logging
from typing import Dict, Set, List, Tuple, Optional
import networkx as nx

from src.normalization import build_score_maps

logger = logging.getLogger(__name__)


def calculate_functional_dependency(protein: str, cluster: Set[str],
                                   protein_go_terms: Dict[str, Set[str]],
                                   go_tfidf,
                                   cluster_id: int) -> float:
    """Functional dependency fd(p,c) from shared GO TF-IDF terms."""
    if protein not in protein_go_terms:
        return 0.0

    protein_terms = protein_go_terms[protein]
    if not protein_terms:
        return 0.0

    fd_score = 0.0
    for go_term in protein_terms:
        fd_score += go_tfidf.get_tfidf(cluster_id, go_term)

    return fd_score / len(protein_terms)


def build_all_fd_scores(
    clusters: Dict[int, Set[str]],
    protein_go_terms: Dict[str, Set[str]],
    go_tfidf,
) -> Dict[Tuple[str, int], float]:
    """Precompute raw FD for every protein in every cluster."""
    fd_scores = {}
    all_proteins = set()
    for cluster in clusters.values():
        all_proteins.update(cluster)
    for cid, cluster in sorted(clusters.items()):
        for protein in sorted(all_proteins):
            fd_scores[(protein, cid)] = calculate_functional_dependency(
                protein, cluster, protein_go_terms, go_tfidf, cid
            )
    return fd_scores


def calculate_membership(
    protein: str,
    cluster: Set[str],
    cluster_id: int,
    graph: nx.Graph,
    protein_go_terms: Dict[str, Set[str]],
    go_tfidf,
    permanence_scores: Dict[str, Dict[int, float]],
    alpha: float = 0.5,
    fd_scores: Optional[Dict[Tuple[str, int], float]] = None,
    norm_perm: Optional[Dict[Tuple[str, int], float]] = None,
    norm_fd: Optional[Dict[Tuple[str, int], float]] = None,
    normalize_scores: bool = True,
) -> float:
    """
    Membership(p, C_i) = alpha * norm_perm(p, C_i) + (1 - alpha) * norm_fd(p, C_i).

    When normalize_scores is False, uses raw permanence and FD (legacy).
    """
    perm = permanence_scores.get(protein, {}).get(cluster_id, 0.0)
    if fd_scores is not None:
        fd = fd_scores.get((protein, cluster_id), 0.0)
    else:
        fd = calculate_functional_dependency(
            protein, cluster, protein_go_terms, go_tfidf, cluster_id
        )

    if normalize_scores and norm_perm is not None and norm_fd is not None:
        perm = norm_perm.get((protein, cluster_id), 0.0)
        fd = norm_fd.get((protein, cluster_id), 0.0)
    elif normalize_scores and (norm_perm is None or norm_fd is None):
        # Fallback: local min-max over this protein's clusters
        keys = [(protein, cid) for cid in permanence_scores.get(protein, {})]
        if not keys:
            keys = [(protein, cluster_id)]
        raw_perm = {k: permanence_scores.get(k[0], {}).get(k[1], 0.0) for k in keys}
        raw_fd = {k: (fd_scores or {}).get(k, fd) for k in keys}
        from src.normalization import minmax_normalize
        np_map = minmax_normalize(raw_perm)
        nf_map = minmax_normalize(raw_fd)
        perm = np_map.get((protein, cluster_id), perm)
        fd = nf_map.get((protein, cluster_id), fd)

    return alpha * perm + (1.0 - alpha) * fd


def calculate_intra_extra_links(protein: str, cluster: Set[str],
                                graph: nx.Graph) -> Tuple[int, int]:
    if protein not in graph:
        return (0, 0)
    neighbors = set(graph.neighbors(protein))
    return len(neighbors & cluster), len(neighbors - cluster)


def find_emax_cluster(protein: str, clusters: Dict[int, Set[str]],
                     graph: nx.Graph) -> int:
    if protein not in graph:
        return -1
    neighbors = set(graph.neighbors(protein))
    max_connections = 0
    emax_cluster_id = -1
    for cluster_id, cluster in clusters.items():
        if protein in cluster:
            continue
        connections = len(neighbors & cluster)
        if connections > max_connections:
            max_connections = connections
            emax_cluster_id = cluster_id
    return emax_cluster_id


def apply_overlap_reassignment(
    clusters: Dict[int, Set[str]],
    graph: nx.Graph,
    protein_go_terms: Dict[str, Set[str]],
    go_tfidf,
    permanence_scores: Dict[str, Dict[int, float]],
    alpha: float = 0.5,
    overlap_tau: float = 0.1,
    transfer_tau: float = 0.0,
    fd_scores: Optional[Dict[Tuple[str, int], float]] = None,
    normalize_scores: bool = True,
) -> Dict[int, Set[str]]:
    """Apply overlapping reassignment using normalized membership scores."""
    updated_clusters = {cid: set(proteins) for cid, proteins in sorted(clusters.items())}
    all_proteins = set()
    for cluster in clusters.values():
        all_proteins.update(cluster)

    if fd_scores is None:
        fd_scores = build_all_fd_scores(clusters, protein_go_terms, go_tfidf)

    cluster_ids = sorted(updated_clusters.keys())
    norm_perm, norm_fd = (None, None)
    if normalize_scores:
        norm_perm, norm_fd = build_score_maps(
            all_proteins, cluster_ids, permanence_scores, fd_scores
        )

    logger.info(f"Applying overlap reassignment for {len(all_proteins)} proteins...")

    for protein in sorted(all_proteins):
        current_clusters = [cid for cid, cluster in sorted(updated_clusters.items())
                            if protein in cluster]
        if not current_clusters:
            continue

        current_memberships = {}
        for cid in current_clusters:
            cluster = updated_clusters[cid]
            current_memberships[cid] = calculate_membership(
                protein, cluster, cid, graph, protein_go_terms, go_tfidf,
                permanence_scores, alpha, fd_scores, norm_perm, norm_fd, normalize_scores,
            )

        for cluster_id, cluster in sorted(updated_clusters.items()):
            if cluster_id in current_clusters:
                continue
            test_cluster = cluster | {protein}
            memb_if_added = calculate_membership(
                protein, test_cluster, cluster_id, graph, protein_go_terms, go_tfidf,
                permanence_scores, alpha, fd_scores, norm_perm, norm_fd, normalize_scores,
            )
            max_current = max(current_memberships.values()) if current_memberships else 0.0
            if memb_if_added - max_current > overlap_tau:
                updated_clusters[cluster_id].add(protein)

        for cid in current_clusters:
            cluster = updated_clusters[cid]
            intra_links, extra_links = calculate_intra_extra_links(protein, cluster, graph)
            if extra_links > intra_links and transfer_tau >= 0:
                emax_cid = find_emax_cluster(protein, updated_clusters, graph)
                if emax_cid != -1 and emax_cid != cid:
                    emax_cluster = updated_clusters[emax_cid]
                    emax_intra, _ = calculate_intra_extra_links(protein, emax_cluster, graph)
                    if emax_intra > intra_links:
                        updated_clusters[cid].discard(protein)
                        updated_clusters[emax_cid].add(protein)

    return updated_clusters
