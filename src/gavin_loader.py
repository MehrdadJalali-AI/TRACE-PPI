"""
Gavin PPI network loader.
Parses weighted socio-affinity PPI networks.
"""

import logging
import networkx as nx
import re
from typing import Tuple

logger = logging.getLogger(__name__)


def _valid_protein_id(protein_id: str) -> bool:
    """Return True for usable protein identifiers in the Gavin edge list.

    The local Gavin file contains a small number of spreadsheet-corrupted
    entries such as ``#NAME?`` and identifiers with embedded spaces. These are
    not valid protein identifiers for CYC2008 matching and also break exact
    external tools such as ClusterONE, so they are skipped consistently.
    """
    if not protein_id or protein_id == "#NAME?":
        return False
    if re.search(r"\s", protein_id):
        return False
    return True


class GavinLoader:
    """
    Load weighted PPI network from Gavin socio-affinity file.
    """
    
    def __init__(self, normalize: bool = True):
        """
        Initialize Gavin loader.
        
        Args:
            normalize: Whether to normalize weights to [0, 1]
        """
        self.normalize = normalize
    
    def load(self, gavin_file: str) -> nx.Graph:
        """
        Load weighted PPI network from Gavin file.
        
        Format:
        s	d	description
        YKL144C	YPR110C	0.397689
        ...
        
        Args:
            gavin_file: Path to Gavin PPI file
            
        Returns:
            graph: NetworkX graph with weighted edges
        """
        graph = nx.Graph()
        
        logger.info(f"Loading Gavin PPI network from {gavin_file}...")
        
        weights = []
        
        with open(gavin_file, 'r') as f:
            # Skip header line
            header = next(f).strip()
            logger.debug(f"Header: {header}")
            
            edge_count = 0
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    protein1 = parts[0].strip()
                    protein2 = parts[1].strip()
                    if not (_valid_protein_id(protein1) and _valid_protein_id(protein2)):
                        logger.debug(f"Skipping line with malformed protein id: {line.strip()}")
                        continue
                    
                    # Try to parse weight - handle various formats
                    try:
                        weight_str = parts[2].strip()
                        # Remove any non-numeric prefixes/suffixes
                        weight_str = weight_str.split()[-1] if ' ' in weight_str else weight_str
                        weight = float(weight_str)
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Skipping line with invalid weight: {line.strip()}")
                        continue
                    
                    weights.append(weight)
                    graph.add_edge(protein1, protein2, weight=weight)
                    edge_count += 1
                    
                    if edge_count % 10000 == 0:
                        logger.debug(f"Loaded {edge_count} edges...")
        
        logger.info(f"Loaded {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        
        # Normalize weights if requested
        if self.normalize and weights:
            min_weight = min(weights)
            max_weight = max(weights)
            weight_range = max_weight - min_weight
            
            if weight_range > 0:
                logger.info(f"Normalizing weights from [{min_weight:.4f}, {max_weight:.4f}] to [0, 1]")
                for u, v, data in graph.edges(data=True):
                    normalized_weight = (data['weight'] - min_weight) / weight_range
                    graph[u][v]['weight'] = normalized_weight
            else:
                logger.warning("All weights are identical, skipping normalization")
        
        return graph
