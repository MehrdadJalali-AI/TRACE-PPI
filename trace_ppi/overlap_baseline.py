"""Overlap-reassignment baseline (MCL + overlap; MCL + overlap + supplementation).

MCL clusters are post-processed by a permanence- and GO TF-IDF-based reassignment that can
give proteins additional memberships (Chakraborty et al. 2014 permanence).  The code is kept
verbatim so that the baseline numbers are reproducible.
"""
from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

# ---- normalization
def minmax_normalize(values: Dict[Tuple[str, int], float]) -> Dict[Tuple[str, int], float]:
    """Min-max normalize a sparse (protein, cluster_id) score map to [0, 1]."""
    if not values:
        return {}
    arr = np.array(list(values.values()), dtype=float)
    vmin, vmax = float(arr.min()), float(arr.max())
    if vmax - vmin < 1e-12:
        return {k: 0.0 for k in values}
    return {k: (v - vmin) / (vmax - vmin) for k, v in values.items()}


def build_score_maps(
    proteins: Iterable[str],
    cluster_ids: Iterable[int],
    permanence_scores: Dict[str, Dict[int, float]],
    fd_scores: Dict[Tuple[str, int], float],
) -> Tuple[Dict[Tuple[str, int], float], Dict[Tuple[str, int], float]]:
    """Collect raw permanence and FD for all (protein, cluster) pairs to normalize."""
    perm_map = {}
    fd_map = {}
    for protein in sorted(proteins):
        for cid in sorted(cluster_ids):
            perm_map[(protein, cid)] = permanence_scores.get(protein, {}).get(cid, 0.0)
            fd_map[(protein, cid)] = fd_scores.get((protein, cid), 0.0)
    return minmax_normalize(perm_map), minmax_normalize(fd_map)


# ---- go tfidf
logger = logging.getLogger(__name__)


class GOTFIDF:
    """
    Calculate TF-IDF importance scores for GO terms in clusters.
    """
    
    def __init__(self, clusters: Dict[int, Set[str]], 
                 protein_go_terms: Dict[str, Set[str]]):
        """
        Initialize TF-IDF calculator.
        
        Args:
            clusters: Dict mapping cluster_id to set of protein IDs
            protein_go_terms: Dict mapping protein ID to set of GO term IDs
        """
        self.clusters = clusters
        self.protein_go_terms = protein_go_terms
        self.num_clusters = len(clusters)
        
        # Build term frequency and document frequency
        self._compute_tf_idf()
    
    def _compute_tf_idf(self):
        """Compute TF-IDF scores for all GO terms in all clusters."""
        # Term frequency (TF): frequency of GO term in cluster
        # Document frequency (DF): number of clusters containing the term
        # IDF = log(N / DF) where N is total number of clusters
        
        self.tf = defaultdict(lambda: defaultdict(int))  # cluster_id -> go_term -> count
        self.df = defaultdict(int)  # go_term -> number of clusters containing it
        
        # Count term frequencies per cluster
        for cluster_id, proteins in sorted(self.clusters.items()):
            cluster_go_terms = set()
            for protein in sorted(proteins):
                if protein in self.protein_go_terms:
                    for go_term in sorted(self.protein_go_terms[protein]):
                        self.tf[cluster_id][go_term] += 1
                        cluster_go_terms.add(go_term)
            
            # Update document frequency
            for go_term in sorted(cluster_go_terms):
                self.df[go_term] += 1
        
        # Compute TF-IDF scores
        self.tfidf_scores = defaultdict(lambda: defaultdict(float))
        
        for cluster_id in sorted(self.clusters):
            cluster_size = len(self.clusters[cluster_id])
            if cluster_size == 0:
                continue
            
            for go_term, tf_count in sorted(self.tf[cluster_id].items()):
                # TF: normalized frequency in cluster
                tf = tf_count / cluster_size
                
                # IDF: inverse document frequency
                df = self.df[go_term]
                idf = math.log(self.num_clusters / df) if df > 0 else 0
                
                # TF-IDF score (Eq.3)
                self.tfidf_scores[cluster_id][go_term] = tf * idf
    
    def get_tfidf(self, cluster_id: int, go_term: str) -> float:
        """
        Get TF-IDF score for a GO term in a cluster.
        
        Args:
            cluster_id: Cluster ID
            go_term: GO term ID
            
        Returns:
            TF-IDF score
        """
        return self.tfidf_scores.get(cluster_id, {}).get(go_term, 0.0)
    
    def get_top_terms(self, cluster_id: int, top_k: int = 10) -> List[tuple]:
        """
        Get top-k GO terms by TF-IDF score for a cluster.
        
        Args:
            cluster_id: Cluster ID
            top_k: Number of top terms to return
            
        Returns:
            List of (go_term, tfidf_score) tuples, sorted by score descending
        """
        if cluster_id not in self.tfidf_scores:
            return []
        
        terms_scores = list(self.tfidf_scores[cluster_id].items())
        terms_scores.sort(key=lambda x: x[1], reverse=True)
        return terms_scores[:top_k]
    
    def get_all_scores(self) -> Dict[int, Dict[str, float]]:
        """
        Get all TF-IDF scores for all clusters.
        
        Returns:
            Dict mapping cluster_id to dict of go_term -> tfidf_score
        """
        return dict(self.tfidf_scores)


# ---- permanence
logger = logging.getLogger(__name__)


def calculate_permanence(protein: str, cluster: Set[str], graph: nx.Graph) -> float:
    """
    Calculate permanence for a protein in a cluster (Eq.1).
    
    Permanence(p, c) = (I(p) / E_max(p)) - (1 - C_in(p))
    
    Where:
    - I(p): internal connections (edges to proteins in same cluster)
    - E_max(p): maximum external connections to any single cluster
    - C_in(p): clustering coefficient of internal neighbors
    
    Args:
        protein: Protein ID
        cluster: Set of protein IDs in the cluster
        graph: NetworkX graph
        
    Returns:
        Permanence score
    """
    if protein not in graph:
        return 0.0
    
    neighbors = set(graph.neighbors(protein))
    if not neighbors:
        return 0.0
    
    # Internal connections: neighbors in the same cluster
    internal_neighbors = neighbors & cluster
    I_p = len(internal_neighbors)
    
    # External connections: neighbors not in this cluster
    external_neighbors = neighbors - cluster
    
    if not external_neighbors:
        # No external connections - fully internal
        if I_p > 0:
            return 1.0
        return 0.0
    
    # E_max: maximum external connections to any single cluster
    # For now, we'll use the count of external neighbors as approximation
    # In full implementation, would need to know all clusters
    E_max_p = len(external_neighbors)
    
    # Clustering coefficient of internal neighbors
    # C_in(p) = fraction of edges between internal neighbors
    if I_p < 2:
        C_in_p = 0.0
    else:
        internal_edges = 0
        internal_list = sorted(internal_neighbors)
        for i, n1 in enumerate(internal_list):
            for n2 in internal_list[i+1:]:
                if graph.has_edge(n1, n2):
                    internal_edges += 1
        max_possible = I_p * (I_p - 1) / 2
        C_in_p = internal_edges / max_possible if max_possible > 0 else 0.0
    
    # Calculate permanence (Eq.1)
    if E_max_p == 0:
        permanence = (I_p / max(1, len(neighbors))) - (1 - C_in_p)
    else:
        permanence = (I_p / E_max_p) - (1 - C_in_p)
    
    return permanence


def calculate_permanence_all_proteins(clusters: Dict[int, Set[str]], 
                                      graph: nx.Graph) -> Dict[str, Dict[int, float]]:
    """
    Calculate permanence for all proteins in all clusters.
    
    Args:
        clusters: Dict mapping cluster_id to set of protein IDs
        graph: NetworkX graph
        
    Returns:
        Dict mapping protein_id to dict of cluster_id -> permanence score
    """
    permanence_scores = {}
    
    # Build reverse mapping: protein -> clusters it belongs to
    protein_clusters = {}
    for cluster_id, proteins in sorted(clusters.items()):
        for protein in sorted(proteins):
            if protein not in protein_clusters:
                protein_clusters[protein] = []
            protein_clusters[protein].append(cluster_id)
    
    # Calculate permanence for each protein in each cluster
    for protein, cluster_ids in sorted(protein_clusters.items()):
        permanence_scores[protein] = {}
        for cluster_id in sorted(cluster_ids):
            cluster = clusters[cluster_id]
            perm = calculate_permanence(protein, cluster, graph)
            permanence_scores[protein][cluster_id] = perm
    
    return permanence_scores


# ---- membership overlap
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


# ---- entry points
def mcl_clusters_discovery_order(graph: nx.Graph, inflation: float = 2.0) -> Dict[int, Set[str]]:
    """Strict MCL on the unweighted adjacency; clusters keep the order returned by
    markov_clustering over graph.nodes(); singletons are dropped but ids are kept."""
    import markov_clustering as mc
    from scipy.sparse import lil_matrix

    nodes = list(graph.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    matrix = lil_matrix((len(nodes), len(nodes)), dtype=float)
    for u, v in graph.edges():
        matrix[idx[u], idx[v]] = 1.0
        matrix[idx[v], idx[u]] = 1.0
    clusters = mc.get_clusters(mc.run_mcl(matrix.tocsr(), inflation=inflation))
    return {cid: {nodes[i] for i in members} for cid, members in enumerate(clusters) if len(members) >= 2}


def mcl_overlap(graph: nx.Graph, go_map: Dict[str, Set[str]], inflation: float = 2.0,
                overlap_tau: float = 0.1) -> Dict[int, Set[str]]:
    """MCL + overlap.  Cluster ids follow markov_clustering's output order, except when that
    partition has more than max(50, n/3) clusters, in which case the strict-MCL region order of
    trace_ppi.mcl is used.  Both orders contain the same MCL clusters; only the ids differ."""
    from .mcl import run_mcl

    if len(run_mcl(graph, inflation).regions) > max(50, graph.number_of_nodes() // 3):
        clusters = run_mcl(graph, inflation).modules(2)
    else:
        clusters = mcl_clusters_discovery_order(graph, inflation)
    gt = GOTFIDF(clusters, go_map)
    perm = calculate_permanence_all_proteins(clusters, graph)
    return apply_overlap_reassignment(clusters, graph, go_map, gt, perm, alpha=0.5, overlap_tau=overlap_tau)
