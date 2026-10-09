"""
Permanence calculation for community detection.
Implements Eq.1 from main.docx.

Permanence measures how well a protein belongs to its community based on
internal vs external connections.
"""

import logging
from typing import Dict, Set
import networkx as nx

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
