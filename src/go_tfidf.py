"""
TF-IDF calculation for GO terms in clusters.
Implements Eq.3 from main.docx.
"""

import logging
from typing import Dict, Set, List
from collections import defaultdict
import math

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
