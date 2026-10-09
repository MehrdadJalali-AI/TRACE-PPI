"""
MCL (Markov Cluster Algorithm) clustering for initial community detection.
"""

import logging
import subprocess
import tempfile
import os
import networkx as nx
from typing import Dict, List, Set

logger = logging.getLogger(__name__)


class MCLClustering:
    """
    MCL clustering wrapper.
    Requires MCL to be installed: https://micans.org/mcl/
    """
    
    def __init__(self, inflation: float = 2.0):
        """
        Initialize MCL clustering.
        
        Args:
            inflation: MCL inflation parameter (higher = more clusters)
        """
        self.inflation = inflation
    
    def cluster(self, graph: nx.Graph) -> Dict[int, Set[str]]:
        """
        Run MCL clustering on graph.
        
        Args:
            graph: NetworkX graph
            
        Returns:
            clusters: Dict mapping cluster_id to set of protein IDs
        """
        logger.info(f"Running MCL clustering (inflation={self.inflation})...")
        
        # Check if MCL is available
        try:
            subprocess.run(['mcl', '--version'], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("MCL binary not found. Trying Python markov_clustering.")
            py_clusters = self._python_mcl_clustering(graph)
            if py_clusters:
                return py_clusters
            logger.warning("Python MCL failed. Using Louvain fallback.")
            return self._fallback_clustering(graph)
        
        # Write graph to temporary file in MCL format
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as tmp_in:
            # Write edges with weights
            for u, v, data in graph.edges(data=True):
                weight = data.get('weight', 1.0)
                tmp_in.write(f"{u}\t{v}\t{weight}\n")
            tmp_in_path = tmp_in.name
        
        tmp_out_path = tmp_in_path + '.out'
        
        try:
            # Run MCL
            cmd = ['mcl', tmp_in_path, '--abc', '-I', str(self.inflation), '-o', tmp_out_path]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Parse MCL output
            clusters = {}
            cluster_id = 0
            
            with open(tmp_out_path, 'r') as f:
                for line in f:
                    proteins = line.strip().split('\t')
                    if proteins and proteins[0]:
                        clusters[cluster_id] = set(proteins)
                        cluster_id += 1
            
            logger.info(f"MCL found {len(clusters)} clusters")
            return clusters
            
        except subprocess.CalledProcessError as e:
            logger.error(f"MCL failed: {e.stderr}")
            return self._fallback_clustering(graph)
        finally:
            # Cleanup
            if os.path.exists(tmp_in_path):
                os.remove(tmp_in_path)
            if os.path.exists(tmp_out_path):
                os.remove(tmp_out_path)
    
    def _python_mcl_clustering(self, graph: nx.Graph) -> Dict[int, Set[str]]:
        """Run MCL via the markov_clustering package (no system mcl binary)."""
        try:
            import markov_clustering as mc
            from scipy.sparse import lil_matrix
        except ImportError:
            return {}

        nodes = list(graph.nodes())
        if len(nodes) < 2:
            return {0: set(nodes)} if nodes else {}

        idx = {n: i for i, n in enumerate(nodes)}
        n = len(nodes)
        matrix = lil_matrix((n, n), dtype=float)
        # markov_clustering is unstable on continuous socioaffinity weights;
        # use unweighted adjacency (standard MCL on PPI support graphs).
        for u, v in graph.edges():
            i, j = idx[u], idx[v]
            matrix[i, j] = 1.0
            matrix[j, i] = 1.0

        try:
            result = mc.run_mcl(matrix.tocsr(), inflation=self.inflation)
            cluster_lists = mc.get_clusters(result)
        except Exception as e:
            logger.warning(f"markov_clustering failed: {e}")
            return {}

        clusters: Dict[int, Set[str]] = {}
        for cluster_id, member_indices in enumerate(cluster_lists):
            proteins = {nodes[i] for i in member_indices}
            if proteins:
                clusters[cluster_id] = proteins
        # Reject near-degenerate partitions (one protein per cluster).
        if len(clusters) > max(50, len(nodes) // 3):
            logger.warning(
                f"Python MCL returned {len(clusters)} clusters on {len(nodes)} nodes; "
                "treating as failure."
            )
            return {}
        logger.info(f"Python MCL found {len(clusters)} clusters")
        return clusters

    def _fallback_clustering(self, graph: nx.Graph) -> Dict[int, Set[str]]:
        """
        Fallback to NetworkX community detection if MCL is not available.
        Uses Louvain algorithm as approximation.
        """
        logger.info("Using Louvain algorithm as fallback...")
        try:
            import community.community_louvain as community_louvain
            partition = community_louvain.best_partition(graph)
            
            clusters = {}
            for node, cluster_id in partition.items():
                if cluster_id not in clusters:
                    clusters[cluster_id] = set()
                clusters[cluster_id].add(node)
            
            logger.info(f"Louvain found {len(clusters)} clusters")
            return clusters
        except ImportError:
            logger.warning("python-louvain not available. Using simple connected components.")
            clusters = {}
            for i, component in enumerate(nx.connected_components(graph)):
                clusters[i] = component
            logger.info(f"Found {len(clusters)} connected components")
            return clusters

