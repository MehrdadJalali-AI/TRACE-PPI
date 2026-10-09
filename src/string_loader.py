"""
STRING database loader for PPI networks.
Supports both download mode (parsing files) and REST API mode.
"""

import os
import gzip
import logging
import requests
from typing import Dict, Set, Tuple, Optional
import networkx as nx
import pandas as pd

logger = logging.getLogger(__name__)


class STRINGLoader:
    """
    Load PPI networks from STRING database.
    """
    
    STRING_BASE_URL = "https://string-db.org/api"
    STRING_DOWNLOAD_BASE = "https://stringdb-static.org/download"
    
    def __init__(self, taxid: int, cache_dir: str = "cache", threshold: int = 700):
        """
        Initialize STRING loader.
        
        Args:
            taxid: NCBI taxonomy ID (e.g., 4932 for S. cerevisiae)
            cache_dir: Directory for caching downloaded files
            threshold: Combined score threshold (0-1000)
        """
        self.taxid = taxid
        self.cache_dir = cache_dir
        self.threshold = threshold
        os.makedirs(cache_dir, exist_ok=True)
        
    def load_from_download(self, data_dir: Optional[str] = None) -> Tuple[nx.Graph, Dict[str, str]]:
        """
        Load PPI network from downloaded STRING files.
        
        Args:
            data_dir: Directory containing STRING files. If None, uses cache_dir.
            
        Returns:
            graph: NetworkX graph with protein interactions
            protein_aliases: Dict mapping STRING IDs to UniProt IDs
        """
        data_dir = data_dir or self.cache_dir
        
        # Try to find links file (check both compressed and uncompressed, and root dir)
        links_file = None
        possible_paths = [
            os.path.join(data_dir, f"{self.taxid}.protein.links.detailed.v11.5.txt.gz"),
            os.path.join(data_dir, f"{self.taxid}.protein.links.detailed.v11.5.txt"),
            os.path.join(".", f"{self.taxid}.protein.links.detailed.v11.5.txt.gz"),
            os.path.join(".", f"{self.taxid}.protein.links.detailed.v11.5.txt"),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                links_file = path
                break
        
        if links_file is None:
            raise FileNotFoundError(
                f"STRING links file not found. Checked:\n" + "\n".join(f"  - {p}" for p in possible_paths) +
                f"\nPlease download from: {self.STRING_DOWNLOAD_BASE}/protein.links.detailed.v11.5/{self.taxid}.protein.links.detailed.v11.5.txt.gz"
            )
        
        # Try to find aliases file (optional - can work without it)
        aliases_file = None
        alias_paths = [
            os.path.join(data_dir, f"{self.taxid}.protein.aliases.v11.5.txt.gz"),
            os.path.join(data_dir, f"{self.taxid}.protein.aliases.v11.5.txt"),
            os.path.join(".", f"{self.taxid}.protein.aliases.v11.5.txt.gz"),
            os.path.join(".", f"{self.taxid}.protein.aliases.v11.5.txt"),
        ]
        
        for path in alias_paths:
            if os.path.exists(path):
                aliases_file = path
                break
        
        if aliases_file is None:
            logger.warning("Aliases file not found. Will use STRING IDs directly.")
            protein_aliases = {}
        else:
            # Load aliases
            protein_aliases = self._load_aliases(aliases_file)
        
        logger.info(f"Loading STRING network from files (threshold={self.threshold})...")
        logger.info(f"Using links file: {links_file}")
        if aliases_file:
            logger.info(f"Using aliases file: {aliases_file}")
        
        # Load network
        graph = self._load_network(links_file, protein_aliases)
        
        logger.info(f"Loaded network: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        return graph, protein_aliases
    
    def _load_aliases(self, aliases_file: str) -> Dict[str, str]:
        """Load protein aliases and map to UniProt IDs."""
        aliases = {}
        uniprot_mapping = {}
        
        logger.info(f"Loading aliases from {aliases_file}...")
        
        # Check if file is compressed
        open_func = gzip.open if aliases_file.endswith('.gz') else open
        mode = 'rt' if aliases_file.endswith('.gz') else 'r'
        
        with open_func(aliases_file, mode) as f:
            next(f)  # Skip header
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    string_id = parts[0]
                    alias = parts[1]
                    source = parts[2]
                    
                    # Prioritize UniProt IDs
                    if source == "UniProt_AC" or source == "UniProt_ID":
                        if string_id not in uniprot_mapping:
                            uniprot_mapping[string_id] = alias
                    
                    aliases[string_id] = alias
        
        logger.info(f"Loaded {len(uniprot_mapping)} UniProt mappings")
        return uniprot_mapping
    
    def _load_network(self, links_file: str, protein_aliases: Dict[str, str]) -> nx.Graph:
        """Load PPI network from links file."""
        graph = nx.Graph()
        
        logger.info(f"Loading network from {links_file}...")
        
        # Check if file is compressed
        open_func = gzip.open if links_file.endswith('.gz') else open
        mode = 'rt' if links_file.endswith('.gz') else 'r'
        
        with open_func(links_file, mode) as f:
            next(f)  # Skip header
            edge_count = 0
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 10:
                    protein1 = parts[0]
                    protein2 = parts[1]
                    combined_score = int(parts[9])
                    
                    if combined_score >= self.threshold:
                        # Use UniProt ID if available, otherwise STRING ID
                        p1_id = protein_aliases.get(protein1, protein1)
                        p2_id = protein_aliases.get(protein2, protein2)
                        
                        graph.add_edge(p1_id, p2_id, weight=combined_score / 1000.0)
                        edge_count += 1
                        
                        if edge_count % 100000 == 0:
                            logger.debug(f"Loaded {edge_count} edges...")
        
        return graph
    
    def load_from_api(self, protein_list: Optional[list] = None) -> Tuple[nx.Graph, Dict[str, str]]:
        """
        Load PPI network from STRING REST API.
        
        Args:
            protein_list: List of protein identifiers. If None, uses all proteins for taxid.
            
        Returns:
            graph: NetworkX graph
            protein_aliases: Dict mapping STRING IDs to identifiers
        """
        logger.info("Loading network from STRING API...")
        
        if protein_list is None:
            # Get all proteins for this organism
            url = f"{self.STRING_BASE_URL}/json/get_string_ids"
            params = {
                "identifiers": "",
                "species": self.taxid,
                "limit": 1
            }
            # This is a simplified version - full implementation would fetch all proteins
            raise NotImplementedError("Full API implementation requires pagination")
        
        # Get interactions
        url = f"{self.STRING_BASE_URL}/json/network"
        params = {
            "identifiers": "%0d".join(protein_list[:100]),  # API limit
            "species": self.taxid,
            "required_score": self.threshold
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        graph = nx.Graph()
        protein_aliases = {}
        
        for interaction in data:
            p1 = interaction['preferredName_A']
            p2 = interaction['preferredName_B']
            score = interaction.get('score', 0)
            
            if score * 1000 >= self.threshold:
                graph.add_edge(p1, p2, weight=score)
        
        return graph, protein_aliases

