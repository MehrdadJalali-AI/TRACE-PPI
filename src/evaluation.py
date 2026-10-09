"""
Evaluation metrics for community detection.
Includes structural and biological metrics.
"""

import logging
from typing import Dict, Set, Optional, List
import networkx as nx
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def calculate_intra_density(cluster: Set[str], graph: nx.Graph) -> float:
    """
    Calculate intra-cluster density.
    
    Args:
        cluster: Set of protein IDs in cluster
        graph: NetworkX graph
        
    Returns:
        Density (0-1)
    """
    if len(cluster) < 2:
        return 0.0
    
    cluster_list = list(cluster)
    edges = 0
    for i, p1 in enumerate(cluster_list):
        for p2 in cluster_list[i+1:]:
            if graph.has_edge(p1, p2):
                edges += 1
    
    max_possible = len(cluster) * (len(cluster) - 1) / 2
    return edges / max_possible if max_possible > 0 else 0.0


def calculate_inter_density(cluster: Set[str], all_clusters: Dict[int, Set[str]],
                            graph: nx.Graph) -> float:
    """
    Calculate inter-cluster density (connections to other clusters).
    
    Args:
        cluster: Set of protein IDs in cluster
        all_clusters: All clusters
        graph: NetworkX graph
        
    Returns:
        Average inter-cluster density
    """
    other_proteins = set()
    for cid, c in all_clusters.items():
        if c != cluster:
            other_proteins.update(c)
    
    if len(other_proteins) == 0:
        return 0.0
    
    inter_edges = 0
    for protein in cluster:
        neighbors = set(graph.neighbors(protein))
        inter_edges += len(neighbors & other_proteins)
    
    max_possible = len(cluster) * len(other_proteins)
    return inter_edges / max_possible if max_possible > 0 else 0.0


def calculate_conductance(cluster: Set[str], graph: nx.Graph) -> float:
    """
    Calculate conductance of a cluster.
    
    Conductance = cut_size / min(vol(S), vol(V\S))
    
    Args:
        cluster: Set of protein IDs in cluster
        graph: NetworkX graph
        
    Returns:
        Conductance value
    """
    if len(cluster) == 0:
        return 1.0
    
    # Cut size: edges between cluster and rest
    cut_size = 0
    cluster_volume = 0
    
    for protein in cluster:
        neighbors = set(graph.neighbors(protein))
        cluster_volume += len(neighbors)
        cut_size += len(neighbors - cluster)
    
    rest_volume = graph.number_of_edges() * 2 - cluster_volume
    
    min_volume = min(cluster_volume, rest_volume)
    if min_volume == 0:
        return 1.0
    
    return cut_size / min_volume


def calculate_overlapping_modularity(clusters: Dict[int, Set[str]], 
                                     graph: nx.Graph) -> float:
    """
    Calculate overlapping modularity approximation.
    
    Uses the method from Nicosia et al. (2009) for overlapping communities.
    
    Args:
        clusters: Dict mapping cluster_id to set of proteins
        graph: NetworkX graph
        
    Returns:
        Modularity value
    """
    m = graph.number_of_edges()
    if m == 0:
        return 0.0
    
    # Calculate membership matrix (protein -> clusters)
    protein_memberships = {}
    for cluster_id, cluster in clusters.items():
        for protein in cluster:
            if protein not in protein_memberships:
                protein_memberships[protein] = []
            protein_memberships[protein].append(cluster_id)
    
    # Normalize memberships (soft assignment)
    for protein in protein_memberships:
        num_clusters = len(protein_memberships[protein])
        if num_clusters > 0:
            protein_memberships[protein] = [1.0 / num_clusters] * num_clusters
    
    modularity = 0.0
    
    for cluster_id, cluster in clusters.items():
        cluster_list = list(cluster)
        for i, p1 in enumerate(cluster_list):
            memb1 = 1.0 / len([c for c in clusters.values() if p1 in c]) if p1 in cluster else 0.0
            
            for p2 in cluster_list[i+1:]:
                memb2 = 1.0 / len([c for c in clusters.values() if p2 in c]) if p2 in cluster else 0.0
                
                # Expected edge probability
                deg1 = graph.degree(p1) if p1 in graph else 0
                deg2 = graph.degree(p2) if p2 in graph else 0
                expected = (deg1 * deg2) / (2 * m) if m > 0 else 0
                
                # Actual edge
                actual = 1.0 if graph.has_edge(p1, p2) else 0.0
                
                modularity += memb1 * memb2 * (actual - expected)
    
    return modularity / m if m > 0 else 0.0


def calculate_mean_fd_per_cluster(clusters: Dict[int, Set[str]],
                                  protein_go_terms: Dict[str, Set[str]],
                                  go_tfidf) -> float:
    """
    Calculate mean functional dependency per cluster.
    
    Args:
        clusters: Dict mapping cluster_id to set of proteins
        protein_go_terms: Dict mapping protein ID to GO terms
        go_tfidf: GOTFIDF instance
        
    Returns:
        Mean FD across all clusters
    """
    from src.membership_overlap import calculate_functional_dependency
    
    fd_sum = 0.0
    cluster_count = 0
    
    for cluster_id, cluster in clusters.items():
        if len(cluster) == 0:
            continue
        
        cluster_fd_sum = 0.0
        for protein in cluster:
            fd = calculate_functional_dependency(
                protein, cluster, protein_go_terms, go_tfidf, cluster_id
            )
            cluster_fd_sum += fd
        
        avg_fd = cluster_fd_sum / len(cluster) if len(cluster) > 0 else 0.0
        fd_sum += avg_fd
        cluster_count += 1
    
    return fd_sum / cluster_count if cluster_count > 0 else 0.0


def evaluate_clusters(clusters: Dict[int, Set[str]],
                     graph: nx.Graph,
                     protein_go_terms: Dict[str, Set[str]],
                     go_tfidf,
                     gold_standard: Optional[Dict[int, Set[str]]] = None) -> pd.DataFrame:
    """
    Evaluate clusters using multiple metrics.
    
    Args:
        clusters: Dict mapping cluster_id to set of proteins
        graph: NetworkX graph
        protein_go_terms: Dict mapping protein ID to GO terms
        go_tfidf: GOTFIDF instance
        gold_standard: Optional gold standard clusters for comparison
        
    Returns:
        DataFrame with evaluation metrics
    """
    logger.info("Evaluating clusters...")
    
    metrics = {}
    
    # Structural metrics
    intra_densities = []
    inter_densities = []
    conductances = []
    
    for cluster_id, cluster in clusters.items():
        if len(cluster) == 0:
            continue
        
        intra_densities.append(calculate_intra_density(cluster, graph))
        inter_densities.append(calculate_inter_density(cluster, clusters, graph))
        conductances.append(calculate_conductance(cluster, graph))
    
    metrics['intra_density_mean'] = np.mean(intra_densities) if intra_densities else 0.0
    metrics['inter_density_mean'] = np.mean(inter_densities) if inter_densities else 0.0
    metrics['conductance_mean'] = np.mean(conductances) if conductances else 1.0
    
    # Overlapping modularity
    metrics['overlapping_modularity'] = calculate_overlapping_modularity(clusters, graph)
    
    # Biological metrics
    metrics['mean_fd_per_cluster'] = calculate_mean_fd_per_cluster(
        clusters, protein_go_terms, go_tfidf
    )
    
    # Cluster statistics
    cluster_sizes = [len(c) for c in clusters.values() if len(c) > 0]
    metrics['num_clusters'] = len(cluster_sizes)
    metrics['num_singletons'] = len([s for s in cluster_sizes if s == 1])
    metrics['mean_cluster_size'] = np.mean(cluster_sizes) if cluster_sizes else 0.0
    metrics['max_cluster_size'] = max(cluster_sizes) if cluster_sizes else 0
    metrics['min_cluster_size'] = min(cluster_sizes) if cluster_sizes else 0
    
    # Gold standard comparison (if provided)
    if gold_standard is not None:
        precision, recall, f1 = calculate_precision_recall_f1(clusters, gold_standard)
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1_score'] = f1
        
        nmi = calculate_overlapping_nmi(clusters, gold_standard)
        metrics['overlapping_nmi'] = nmi
    
    return pd.DataFrame([metrics])


def calculate_precision_recall_f1(predicted: Dict[int, Set[str]],
                                  gold: Dict[int, Set[str]]) -> tuple:
    """
    Calculate Precision, Recall, and F1 for overlapping communities.
    
    Args:
        predicted: Predicted clusters
        gold: Gold standard clusters
        
    Returns:
        (precision, recall, f1)
    """
    # Convert to protein -> cluster sets mapping
    pred_memberships = {}
    for cid, cluster in predicted.items():
        for protein in cluster:
            if protein not in pred_memberships:
                pred_memberships[protein] = set()
            pred_memberships[protein].add(cid)
    
    gold_memberships = {}
    for cid, cluster in gold.items():
        for protein in cluster:
            if protein not in gold_memberships:
                gold_memberships[protein] = set()
            gold_memberships[protein].add(cid)
    
    # Calculate Jaccard similarity for each protein's cluster assignments
    all_proteins = set(pred_memberships.keys()) | set(gold_memberships.keys())
    
    similarities = []
    for protein in all_proteins:
        pred_clusters = pred_memberships.get(protein, set())
        gold_clusters = gold_memberships.get(protein, set())
        
        if len(pred_clusters) == 0 and len(gold_clusters) == 0:
            similarities.append(1.0)
        elif len(pred_clusters) == 0 or len(gold_clusters) == 0:
            similarities.append(0.0)
        else:
            intersection = len(pred_clusters & gold_clusters)
            union = len(pred_clusters | gold_clusters)
            jaccard = intersection / union if union > 0 else 0.0
            similarities.append(jaccard)
    
    precision = np.mean(similarities) if similarities else 0.0
    
    # For recall, we check how well gold clusters are recovered
    recall = precision  # Simplified - same as precision for symmetric case
    
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1


def calculate_overlapping_nmi(predicted: Dict[int, Set[str]],
                              gold: Dict[int, Set[str]]) -> float:
    """
    Calculate Overlapping Normalized Mutual Information (ONMI).
    
    Simplified implementation - full ONMI is more complex.
    
    Args:
        predicted: Predicted clusters
        gold: Gold standard clusters
        
    Returns:
        ONMI value (0-1)
    """
    # This is a simplified version
    # Full ONMI requires more sophisticated computation
    
    # Get all proteins
    all_proteins = set()
    for cluster in predicted.values():
        all_proteins.update(cluster)
    for cluster in gold.values():
        all_proteins.update(cluster)
    
    n = len(all_proteins)
    if n == 0:
        return 0.0
    
    # Calculate mutual information
    # Simplified: use Jaccard-based approximation
    precision, recall, f1 = calculate_precision_recall_f1(predicted, gold)
    
    # Use F1 as approximation for NMI
    return f1

