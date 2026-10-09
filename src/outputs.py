"""
Output generation for community detection results.
Creates all required CSV files.
"""

import logging
import json
from typing import Dict, Set
import pandas as pd
import os

logger = logging.getLogger(__name__)


def save_initial_clusters(clusters: Dict[int, Set[str]], output_path: str):
    """
    Save initial MCL clusters to CSV.
    
    Args:
        clusters: Dict mapping cluster_id to set of proteins
        output_path: Output file path
    """
    logger.info(f"Saving initial clusters to {output_path}...")
    
    rows = []
    for cluster_id, proteins in clusters.items():
        for protein in proteins:
            rows.append({
                'cluster_id': cluster_id,
                'protein_id': protein
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(rows)} protein-cluster assignments")


def save_go_term_importance(go_tfidf, output_path: str):
    """
    Save GO term TF-IDF importance scores.
    
    Args:
        go_tfidf: GOTFIDF instance
        output_path: Output file path
    """
    logger.info(f"Saving GO term importance to {output_path}...")
    
    rows = []
    all_scores = go_tfidf.get_all_scores()
    
    for cluster_id, term_scores in all_scores.items():
        for go_term, tfidf_score in term_scores.items():
            rows.append({
                'cluster_id': cluster_id,
                'go_term': go_term,
                'tfidf_score': tfidf_score
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(rows)} GO term-cluster scores")


def save_protein_membership(clusters: Dict[int, Set[str]],
                            graph,
                            protein_go_terms: Dict[str, Set[str]],
                            go_tfidf,
                            permanence_scores: Dict[str, Dict[int, float]],
                            alpha: float,
                            output_path: str):
    """
    Save protein membership details.
    
    Args:
        clusters: Dict mapping cluster_id to set of proteins
        graph: NetworkX graph
        protein_go_terms: Dict mapping protein ID to GO terms
        go_tfidf: GOTFIDF instance
        permanence_scores: Pre-computed permanence scores
        alpha: Membership weight parameter
        output_path: Output file path
    """
    logger.info(f"Saving protein membership to {output_path}...")
    
    from src.membership_overlap import (
        calculate_membership, calculate_intra_extra_links,
        find_emax_cluster, build_all_fd_scores,
    )

    fd_scores = build_all_fd_scores(clusters, protein_go_terms, go_tfidf)
    rows = []

    for cluster_id, cluster in clusters.items():
        for protein in cluster:
            perm = permanence_scores.get(protein, {}).get(cluster_id, 0.0)
            from src.membership_overlap import calculate_functional_dependency
            fd = calculate_functional_dependency(
                protein, cluster, protein_go_terms, go_tfidf, cluster_id
            )
            membership = calculate_membership(
                protein, cluster, cluster_id, graph,
                protein_go_terms, go_tfidf, permanence_scores, alpha,
                fd_scores=fd_scores, normalize_scores=True,
            )
            
            intra, extra = calculate_intra_extra_links(protein, cluster, graph)
            
            # Find E_max cluster
            emax_cid = find_emax_cluster(protein, clusters, graph)
            
            rows.append({
                'protein_id': protein,
                'cluster_id': cluster_id,
                'permanence': perm,
                'fd': fd,
                'membership': membership,
                'intra': intra,
                'extra': extra,
                'emax': emax_cid
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(rows)} protein membership records")


def save_optimized_clusters(clusters: Dict[int, Set[str]], 
                           permanence_scores: Dict[str, Dict[int, float]],
                           protein_go_terms: Dict[str, Set[str]],
                           go_tfidf,
                           graph,
                           alpha: float,
                           output_path: str):
    """
    Save optimized clusters with membership scores.
    
    Args:
        clusters: Optimized clusters
        permanence_scores: Pre-computed permanence scores
        protein_go_terms: Dict mapping protein ID to GO terms
        go_tfidf: GOTFIDF instance
        graph: NetworkX graph
        alpha: Membership weight parameter
        output_path: Output file path
    """
    logger.info(f"Saving optimized clusters to {output_path}...")
    
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.membership_overlap import calculate_membership, build_all_fd_scores

    fd_scores = build_all_fd_scores(clusters, protein_go_terms, go_tfidf)
    rows = []

    for cluster_id, cluster in clusters.items():
        for protein in cluster:
            membership = calculate_membership(
                protein, cluster, cluster_id, graph,
                protein_go_terms, go_tfidf, permanence_scores, alpha,
                fd_scores=fd_scores, normalize_scores=True,
            )
            
            rows.append({
                'cluster_id': cluster_id,
                'protein_id': protein,
                'membership_score': membership
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(rows)} optimized cluster assignments")


def save_overlap_summary(clusters: Dict[int, Set[str]], output_path: str):
    """
    Save overlap summary.
    
    Args:
        clusters: Dict mapping cluster_id to set of proteins
        output_path: Output file path
    """
    logger.info(f"Saving overlap summary to {output_path}...")
    
    # Build protein -> clusters mapping
    protein_clusters = {}
    for cluster_id, cluster in clusters.items():
        for protein in cluster:
            if protein not in protein_clusters:
                protein_clusters[protein] = []
            protein_clusters[protein].append(cluster_id)
    
    rows = []
    for protein, cluster_ids in protein_clusters.items():
        rows.append({
            'protein_id': protein,
            'num_clusters': len(cluster_ids),
            'clusters_json': json.dumps(cluster_ids)
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved overlap summary for {len(rows)} proteins")


def save_evaluation_results(evaluation_df: pd.DataFrame, output_path: str):
    """
    Save evaluation results.
    
    Args:
        evaluation_df: DataFrame with evaluation metrics
        output_path: Output file path
    """
    logger.info(f"Saving evaluation results to {output_path}...")
    evaluation_df.to_csv(output_path, index=False)
    logger.info("Evaluation results saved")

