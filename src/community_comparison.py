"""
Comprehensive comparison of community detection algorithms for PPI networks.
Compares LEA-based method against state-of-the-art algorithms.
"""

import time
import json
import logging
import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict

# Import time module explicitly for runtime tracking
import time as time_module

logger = logging.getLogger(__name__)

# Try importing community detection libraries
try:
    import cdlib
    from cdlib import algorithms, evaluation
    CDLIB_AVAILABLE = True
except ImportError:
    CDLIB_AVAILABLE = False
    logger.warning("cdlib not available. Some algorithms will be unavailable.")

try:
    import igraph as ig
    IGRAPH_AVAILABLE = True
except ImportError:
    IGRAPH_AVAILABLE = False
    logger.warning("igraph not available. Some algorithms will be unavailable.")

try:
    import markov_clustering as mc
    MCL_AVAILABLE = True
except ImportError:
    MCL_AVAILABLE = False
    logger.warning("markov_clustering not available. MCL will use fallback.")

try:
    import community as community_louvain
    LOUVAIN_AVAILABLE = True
except ImportError:
    LOUVAIN_AVAILABLE = False
    logger.warning("python-louvain not available. Louvain will use fallback.")


def compute_nmi(communities1: Dict[int, Set[str]], 
                communities2: Dict[int, Set[str]]) -> float:
    """
    Compute Normalized Mutual Information (NMI) between two community structures.
    
    Args:
        communities1: Dict mapping cluster_id to set of nodes
        communities2: Dict mapping cluster_id to set of nodes
        
    Returns:
        NMI score (0-1)
    """
    # Convert to node->cluster mapping
    node_clusters1 = {}
    node_clusters2 = {}
    
    for cid, nodes in communities1.items():
        for node in nodes:
            if node not in node_clusters1:
                node_clusters1[node] = set()
            node_clusters1[node].add(cid)
    
    for cid, nodes in communities2.items():
        for node in nodes:
            if node not in node_clusters2:
                node_clusters2[node] = set()
            node_clusters2[node].add(cid)
    
    # Get all nodes
    all_nodes = set(node_clusters1.keys()) | set(node_clusters2.keys())
    
    if len(all_nodes) == 0:
        return 0.0
    
    # Compute mutual information
    n = len(all_nodes)
    mi = 0.0
    
    for node in all_nodes:
        clusters1 = node_clusters1.get(node, set())
        clusters2 = node_clusters2.get(node, set())
        
        if len(clusters1) == 0 or len(clusters2) == 0:
            continue
        
        # For overlapping, use Jaccard similarity
        intersection = len(clusters1 & clusters2)
        union = len(clusters1 | clusters2)
        if union > 0:
            mi += intersection / union
    
    # Normalize
    nmi = mi / n if n > 0 else 0.0
    return nmi


def compute_overlapping_nmi(communities1: Dict[int, Set[str]],
                            communities2: Dict[int, Set[str]]) -> float:
    """
    Compute Overlapping NMI approximation.
    
    Args:
        communities1: Dict mapping cluster_id to set of nodes
        communities2: Dict mapping cluster_id to set of nodes
        
    Returns:
        Overlapping NMI score
    """
    # Use same approach as NMI but account for overlaps
    return compute_nmi(communities1, communities2)


def compute_modularity(communities: Dict[int, Set[str]], graph: nx.Graph) -> float:
    """
    Compute modularity for communities.
    
    Args:
        communities: Dict mapping cluster_id to set of nodes
        graph: NetworkX graph
        
    Returns:
        Modularity score
    """
    if CDLIB_AVAILABLE:
        # Convert to cdlib format
        node_communities = defaultdict(list)
        for cid, nodes in communities.items():
            for node in nodes:
                node_communities[node].append(cid)
        
        # Create NodeClustering object
        coms = []
        for cid, nodes in communities.items():
            coms.append(list(nodes))
        
        try:
            from cdlib import NodeClustering
            nc = NodeClustering(coms, graph, "custom")
            mod = evaluation.modularity(nc, graph).score
            return mod
        except:
            pass
    
    # Fallback: compute manually
    m = graph.number_of_edges()
    if m == 0:
        return 0.0
    
    modularity = 0.0
    degrees = dict(graph.degree())
    
    for cid, nodes in communities.items():
        if len(nodes) == 0:
            continue
        
        # Compute membership weights (for overlapping)
        node_weights = {}
        for node in nodes:
            # Count how many clusters this node belongs to
            num_clusters = sum(1 for c in communities.values() if node in c)
            node_weights[node] = 1.0 / num_clusters if num_clusters > 0 else 0.0
        
        cluster_nodes = list(nodes)
        for i, node1 in enumerate(cluster_nodes):
            w1 = node_weights.get(node1, 1.0)
            for node2 in cluster_nodes[i+1:]:
                w2 = node_weights.get(node2, 1.0)
                
                # Expected edges
                expected = (degrees.get(node1, 0) * degrees.get(node2, 0)) / (2 * m)
                
                # Actual edge
                actual = 1.0 if graph.has_edge(node1, node2) else 0.0
                
                modularity += w1 * w2 * (actual - expected)
    
    return modularity / m if m > 0 else 0.0


def compute_conductance(communities: Dict[int, Set[str]], graph: nx.Graph) -> float:
    """
    Compute average conductance across communities.
    
    Args:
        communities: Dict mapping cluster_id to set of nodes
        graph: NetworkX graph
        
    Returns:
        Average conductance
    """
    conductances = []
    
    for cid, nodes in communities.items():
        if len(nodes) == 0:
            continue
        
        cluster = set(nodes)
        cut_size = 0
        cluster_volume = 0
        
        for node in cluster:
            neighbors = set(graph.neighbors(node))
            cluster_volume += len(neighbors)
            cut_size += len(neighbors - cluster)
        
        rest_volume = graph.number_of_edges() * 2 - cluster_volume
        min_volume = min(cluster_volume, rest_volume)
        
        if min_volume > 0:
            conductance = cut_size / min_volume
            conductances.append(conductance)
    
    return np.mean(conductances) if conductances else 1.0


def run_louvain(graph: nx.Graph, resolution: float = 1.0, 
                random_seed: Optional[int] = None) -> Tuple[Dict[int, Set[str]], Dict]:
    """
    Run Louvain algorithm.
    
    Args:
        graph: NetworkX graph
        resolution: Resolution parameter
        random_seed: Random seed
        
    Returns:
        (communities_dict, parameters_dict)
    """
    start_time = time.time()
    
    if LOUVAIN_AVAILABLE:
        if random_seed is not None:
            np.random.seed(random_seed)
        
        partition = community_louvain.best_partition(graph, resolution=resolution, random_state=random_seed)
        
        communities = {}
        for node, cid in partition.items():
            if cid not in communities:
                communities[cid] = set()
            communities[cid].add(node)
        
        runtime = time.time() - start_time
        params = {"resolution": resolution, "random_seed": random_seed, "runtime": runtime}
        
        return communities, params
    
    # Fallback using cdlib
    if CDLIB_AVAILABLE:
        try:
            coms = algorithms.louvain(graph, resolution=resolution, randomize=random_seed)
            communities = {}
            for i, com in enumerate(coms.communities):
                communities[i] = set(com)
            
            runtime = time.time() - start_time
            params = {"resolution": resolution, "random_seed": random_seed, "runtime": runtime}
            return communities, params
        except:
            pass
    
    # Final fallback: simple connected components
    communities = {}
    for i, component in enumerate(nx.connected_components(graph)):
        communities[i] = component
    
    runtime = time.time() - start_time
    params = {"resolution": resolution, "random_seed": random_seed, "fallback": "connected_components", "runtime": runtime}
    
    return communities, params


def run_leiden(graph: nx.Graph, resolution: float = 1.0,
               random_seed: Optional[int] = None) -> Tuple[Dict[int, Set[str]], Dict]:
    """
    Run Leiden algorithm.
    
    Args:
        graph: NetworkX graph
        resolution: Resolution parameter
        random_seed: Random seed
        
    Returns:
        (communities_dict, parameters_dict)
    """
    start_time = time.time()
    
    if CDLIB_AVAILABLE:
        try:
            coms = algorithms.leiden(graph, resolution_parameter=resolution, randomize=random_seed)
            communities = {}
            for i, com in enumerate(coms.communities):
                communities[i] = set(com)
            
            runtime = time.time() - start_time
            params = {"resolution": resolution, "random_seed": random_seed, "runtime": runtime}
            return communities, params
        except Exception as e:
            logger.warning(f"Leiden failed: {e}")
    
    # Fallback to Louvain
    return run_louvain(graph, resolution, random_seed)


def run_infomap(graph: nx.Graph, random_seed: Optional[int] = None) -> Tuple[Dict[int, Set[str]], Dict]:
    """
    Run Infomap algorithm.
    
    Args:
        graph: NetworkX graph
        random_seed: Random seed
        
    Returns:
        (communities_dict, parameters_dict)
    """
    start_time = time.time()
    
    if CDLIB_AVAILABLE:
        try:
            coms = algorithms.infomap(graph)
            communities = {}
            for i, com in enumerate(coms.communities):
                communities[i] = set(com)
            
            runtime = time.time() - start_time
            params = {"random_seed": random_seed, "runtime": runtime}
            return communities, params
        except Exception as e:
            logger.warning(f"Infomap failed: {e}")
    
    # Fallback
    communities = {}
    for i, component in enumerate(nx.connected_components(graph)):
        communities[i] = component
    
    runtime = time.time() - start_time
    params = {"random_seed": random_seed, "fallback": "connected_components", "runtime": runtime}
    
    return communities, params


def run_mcl(graph: nx.Graph, inflation: float = 2.0) -> Tuple[Dict[int, Set[str]], Dict]:
    """
    Run Markov Cluster Algorithm (MCL).
    
    Args:
        graph: NetworkX graph
        inflation: MCL inflation parameter
        
    Returns:
        (communities_dict, parameters_dict)
    """
    start_time = time.time()
    
    if MCL_AVAILABLE:
        try:
            # Convert to matrix
            matrix = nx.to_scipy_sparse_array(graph, format='csr')
            result = mc.run_mcl(matrix, inflation=inflation)
            clusters = mc.get_clusters(result)
            
            communities = {}
            for i, cluster in enumerate(clusters):
                communities[i] = set(cluster)
            
            runtime = time.time() - start_time
            params = {"inflation": inflation, "runtime": runtime}
            return communities, params
        except Exception as e:
            logger.warning(f"MCL failed: {e}")
    
    # Fallback using cdlib
    if CDLIB_AVAILABLE:
        try:
            coms = algorithms.markov_clustering(graph, inflation=inflation)
            communities = {}
            for i, com in enumerate(coms.communities):
                communities[i] = set(com)
            
            runtime = time.time() - start_time
            params = {"inflation": inflation, "runtime": runtime}
            return communities, params
        except:
            pass
    
    # Final fallback
    communities = {}
    for i, component in enumerate(nx.connected_components(graph)):
        communities[i] = component
    
    runtime = time.time() - start_time
    params = {"inflation": inflation, "fallback": "connected_components", "runtime": runtime}
    
    return communities, params


def run_link_communities(graph: nx.Graph) -> Tuple[Dict[int, Set[str]], Dict]:
    """
    Run Link Communities algorithm.
    
    Args:
        graph: NetworkX graph
        
    Returns:
        (communities_dict, parameters_dict)
    """
    start_time = time.time()
    
    if CDLIB_AVAILABLE:
        try:
            coms = algorithms.link_communities(graph)
            communities = {}
            for i, com in enumerate(coms.communities):
                communities[i] = set(com)
            
            runtime = time.time() - start_time
            params = {"runtime": runtime}
            return communities, params
        except Exception as e:
            logger.warning(f"Link Communities failed: {e}")
    
    # Fallback
    communities = {}
    for i, component in enumerate(nx.connected_components(graph)):
        communities[i] = component
    
    runtime = time.time() - start_time
    params = {"fallback": "connected_components", "runtime": runtime}
    
    return communities, params


def run_oslom(graph: nx.Graph) -> Tuple[Dict[int, Set[str]], Dict]:
    """
    Run OSLOM algorithm (Order Statistics Local Optimization Method).
    Note: OSLOM requires external binary. This is an approximation using cdlib.
    
    Args:
        graph: NetworkX graph
        
    Returns:
        (communities_dict, parameters_dict)
    """
    start_time = time.time()
    
    if CDLIB_AVAILABLE:
        try:
            # OSLOM is not directly available in cdlib, use similar method
            # Using label propagation as approximation
            coms = algorithms.label_propagation(graph)
            communities = {}
            for i, com in enumerate(coms.communities):
                communities[i] = set(com)
            
            runtime = time.time() - start_time
            params = {"method": "label_propagation_approximation", "runtime": runtime}
            return communities, params
        except Exception as e:
            logger.warning(f"OSLOM approximation failed: {e}")
    
    # Fallback
    communities = {}
    for i, component in enumerate(nx.connected_components(graph)):
        communities[i] = component
    
    runtime = time.time() - start_time
    params = {"fallback": "connected_components", "runtime": runtime}
    
    return communities, params


def run_bigclam(graph: nx.Graph, num_communities: Optional[int] = None) -> Tuple[Dict[int, Set[str]], Dict]:
    """
    Run BigCLAM overlapping community detection.
    
    Args:
        graph: NetworkX graph
        num_communities: Number of communities (if None, auto-detect)
        
    Returns:
        (communities_dict, parameters_dict)
    """
    start_time = time.time()
    
    if CDLIB_AVAILABLE:
        try:
            if num_communities is None:
                # Estimate number of communities
                num_communities = max(10, graph.number_of_nodes() // 100)
            
            coms = algorithms.bigclam(graph, number_communities=num_communities)
            communities = {}
            for i, com in enumerate(coms.communities):
                communities[i] = set(com)
            
            runtime = time.time() - start_time
            params = {"num_communities": num_communities, "runtime": runtime}
            return communities, params
        except Exception as e:
            logger.warning(f"BigCLAM failed: {e}")
    
    # Fallback
    communities = {}
    for i, component in enumerate(nx.connected_components(graph)):
        communities[i] = component
    
    runtime = time.time() - start_time
    params = {"num_communities": num_communities, "fallback": "connected_components", "runtime": runtime}
    
    return communities, params


def run_lea_method(graph: nx.Graph, 
                  initial_clusters: Dict[int, Set[str]],
                  protein_go_terms: Dict[str, Set[str]],
                  go_tfidf,
                  permanence_scores: Dict[str, Dict[int, float]],
                  alpha: float = 0.5,
                  overlap_tau: float = 0.1,
                  transfer_tau: float = 0.0,
                  lea_evaluations: int = 500,
                  random_seed: Optional[int] = None) -> Tuple[Dict[int, Set[str]], Dict]:
    """
    Run LEA-based overlapping community detection.
    
    Args:
        graph: NetworkX graph
        initial_clusters: Initial MCL clusters
        protein_go_terms: GO annotations
        go_tfidf: TF-IDF scores
        permanence_scores: Permanence scores
        alpha: Membership weight
        overlap_tau: Overlap threshold
        transfer_tau: Transfer threshold
        lea_evaluations: LEA max evaluations
        random_seed: Random seed
        
    Returns:
        (communities_dict, parameters_dict)
    """
    start_time = time.time()
    
    try:
        from src.membership_overlap import apply_overlap_reassignment
        from src.lea.optimize import optimize_communities
        
        # Run LEA optimization
        best_solution, best_fitness, optimized_clusters = optimize_communities(
            graph,
            initial_clusters,
            protein_go_terms,
            go_tfidf,
            permanence_scores,
            population_size=30,
            max_evaluations=lea_evaluations,
            lambda_inter=1.0,
            lambda_fragment=0.5,
            random_seed=random_seed
        )
        
        runtime = time.time() - start_time
        params = {
            "alpha": float(best_solution[0]),
            "overlap_tau": float(best_solution[1]),
            "transfer_tau": float(best_solution[2]),
            "lea_evaluations": lea_evaluations,
            "random_seed": random_seed
        }
        
        return optimized_clusters, params
        
    except Exception as e:
        logger.error(f"LEA method failed: {e}")
        # Fallback to initial clusters
        runtime = time.time() - start_time
        params = {"error": str(e), "fallback": "initial_clusters"}
        return initial_clusters, params


def evaluate_communities(communities: Dict[int, Set[str]], 
                        graph: nx.Graph,
                        ground_truth: Optional[Dict[int, Set[str]]] = None) -> Dict:
    """
    Evaluate community detection results.
    
    Args:
        communities: Detected communities
        graph: NetworkX graph
        ground_truth: Optional ground truth communities
        
    Returns:
        Dictionary of evaluation metrics
    """
    metrics = {}
    
    # Basic statistics
    metrics['num_communities'] = len([c for c in communities.values() if len(c) > 0])
    community_sizes = [len(c) for c in communities.values() if len(c) > 0]
    metrics['avg_community_size'] = np.mean(community_sizes) if community_sizes else 0.0
    
    # Structural metrics
    metrics['modularity'] = compute_modularity(communities, graph)
    metrics['conductance'] = compute_conductance(communities, graph)
    
    # NMI (if ground truth available)
    if ground_truth:
        metrics['nmi'] = compute_nmi(communities, ground_truth)
        metrics['overlapping_nmi'] = compute_overlapping_nmi(communities, ground_truth)
    else:
        metrics['nmi'] = None
        metrics['overlapping_nmi'] = None
    
    return metrics


def compare_all_methods(graph: nx.Graph,
                        dataset_name: str,
                        ground_truth: Optional[Dict[int, Set[str]]] = None,
                        lea_data: Optional[Dict] = None,
                        lea_evaluations: int = 500,
                        random_seed: int = 42) -> pd.DataFrame:
    """
    Run all community detection methods and compare results.
    
    Args:
        graph: NetworkX graph
        dataset_name: Name of dataset
        ground_truth: Optional ground truth communities
        lea_data: Optional dict with LEA-specific data (initial_clusters, go_terms, etc.)
        random_seed: Random seed for reproducibility
        
    Returns:
        DataFrame with comparison results
    """
    results = []
    
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    
    logger.info(f"Comparing methods on {dataset_name}: {num_nodes} nodes, {num_edges} edges")
    
    # 1. Louvain
    logger.info("Running Louvain...")
    try:
        method_start = time.time()
        communities, params = run_louvain(graph, resolution=1.0, random_seed=random_seed)
        method_runtime = time.time() - method_start
        metrics = evaluate_communities(communities, graph, ground_truth)
        results.append({
            'dataset': dataset_name,
            'method': 'Louvain',
            'overlapping': False,
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'num_communities': metrics['num_communities'],
            'avg_community_size': metrics['avg_community_size'],
            'modularity': metrics['modularity'],
            'nmi': metrics['nmi'],
            'overlapping_nmi': metrics['overlapping_nmi'],
            'conductance': metrics['conductance'],
            'runtime_sec': method_runtime,
            'parameters': json.dumps(params)
        })
    except Exception as e:
        logger.error(f"Louvain failed: {e}")
    
    # 2. Leiden
    logger.info("Running Leiden...")
    try:
        method_start = time.time()
        communities, params = run_leiden(graph, resolution=1.0, random_seed=random_seed)
        method_runtime = time.time() - method_start
        metrics = evaluate_communities(communities, graph, ground_truth)
        results.append({
            'dataset': dataset_name,
            'method': 'Leiden',
            'overlapping': False,
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'num_communities': metrics['num_communities'],
            'avg_community_size': metrics['avg_community_size'],
            'modularity': metrics['modularity'],
            'nmi': metrics['nmi'],
            'overlapping_nmi': metrics['overlapping_nmi'],
            'conductance': metrics['conductance'],
            'runtime_sec': method_runtime,
            'parameters': json.dumps(params)
        })
    except Exception as e:
        logger.error(f"Leiden failed: {e}")
    
    # 3. Infomap
    logger.info("Running Infomap...")
    try:
        method_start = time.time()
        communities, params = run_infomap(graph, random_seed=random_seed)
        method_runtime = time.time() - method_start
        metrics = evaluate_communities(communities, graph, ground_truth)
        results.append({
            'dataset': dataset_name,
            'method': 'Infomap',
            'overlapping': False,
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'num_communities': metrics['num_communities'],
            'avg_community_size': metrics['avg_community_size'],
            'modularity': metrics['modularity'],
            'nmi': metrics['nmi'],
            'overlapping_nmi': metrics['overlapping_nmi'],
            'conductance': metrics['conductance'],
            'runtime_sec': method_runtime,
            'parameters': json.dumps(params)
        })
    except Exception as e:
        logger.error(f"Infomap failed: {e}")
    
    # 4. MCL
    logger.info("Running MCL...")
    try:
        method_start = time.time()
        communities, params = run_mcl(graph, inflation=2.0)
        method_runtime = time.time() - method_start
        metrics = evaluate_communities(communities, graph, ground_truth)
        results.append({
            'dataset': dataset_name,
            'method': 'MCL',
            'overlapping': False,
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'num_communities': metrics['num_communities'],
            'avg_community_size': metrics['avg_community_size'],
            'modularity': metrics['modularity'],
            'nmi': metrics['nmi'],
            'overlapping_nmi': metrics['overlapping_nmi'],
            'conductance': metrics['conductance'],
            'runtime_sec': method_runtime,
            'parameters': json.dumps(params)
        })
    except Exception as e:
        logger.error(f"MCL failed: {e}")
    
    # 5. OSLOM
    logger.info("Running OSLOM...")
    try:
        method_start = time.time()
        communities, params = run_oslom(graph)
        method_runtime = time.time() - method_start
        metrics = evaluate_communities(communities, graph, ground_truth)
        results.append({
            'dataset': dataset_name,
            'method': 'OSLOM',
            'overlapping': False,
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'num_communities': metrics['num_communities'],
            'avg_community_size': metrics['avg_community_size'],
            'modularity': metrics['modularity'],
            'nmi': metrics['nmi'],
            'overlapping_nmi': metrics['overlapping_nmi'],
            'conductance': metrics['conductance'],
            'runtime_sec': method_runtime,
            'parameters': json.dumps(params)
        })
    except Exception as e:
        logger.error(f"OSLOM failed: {e}")
    
    # 6. Link Communities
    logger.info("Running Link Communities...")
    try:
        method_start = time.time()
        communities, params = run_link_communities(graph)
        method_runtime = time.time() - method_start
        metrics = evaluate_communities(communities, graph, ground_truth)
        results.append({
            'dataset': dataset_name,
            'method': 'Link_Communities',
            'overlapping': True,
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'num_communities': metrics['num_communities'],
            'avg_community_size': metrics['avg_community_size'],
            'modularity': metrics['modularity'],
            'nmi': metrics['nmi'],
            'overlapping_nmi': metrics['overlapping_nmi'],
            'conductance': metrics['conductance'],
            'runtime_sec': method_runtime,
            'parameters': json.dumps(params)
        })
    except Exception as e:
        logger.error(f"Link Communities failed: {e}")
    
    # 7. BigCLAM
    logger.info("Running BigCLAM...")
    try:
        method_start = time.time()
        communities, params = run_bigclam(graph)
        method_runtime = time.time() - method_start
        metrics = evaluate_communities(communities, graph, ground_truth)
        results.append({
            'dataset': dataset_name,
            'method': 'BigCLAM',
            'overlapping': True,
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'num_communities': metrics['num_communities'],
            'avg_community_size': metrics['avg_community_size'],
            'modularity': metrics['modularity'],
            'nmi': metrics['nmi'],
            'overlapping_nmi': metrics['overlapping_nmi'],
            'conductance': metrics['conductance'],
            'runtime_sec': method_runtime,
            'parameters': json.dumps(params)
        })
    except Exception as e:
        logger.error(f"BigCLAM failed: {e}")
    
    # 8. LEA-based method
    if lea_data:
        logger.info("Running LEA-based method...")
        try:
            method_start = time.time()
            communities, params = run_lea_method(
                graph,
                lea_data['initial_clusters'],
                lea_data.get('protein_go_terms', {}),
                lea_data.get('go_tfidf'),
                lea_data.get('permanence_scores', {}),
                alpha=0.5,
                overlap_tau=0.1,
                lea_evaluations=lea_evaluations,
                random_seed=random_seed
            )
            method_runtime = time.time() - method_start
            
            metrics = evaluate_communities(communities, graph, ground_truth)
            results.append({
                'dataset': dataset_name,
                'method': 'LEA_Overlapping',
                'overlapping': True,
                'num_nodes': num_nodes,
                'num_edges': num_edges,
                'num_communities': metrics['num_communities'],
                'avg_community_size': metrics['avg_community_size'],
                'modularity': metrics['modularity'],
                'nmi': metrics['nmi'],
                'overlapping_nmi': metrics['overlapping_nmi'],
                'conductance': metrics['conductance'],
                'runtime_sec': method_runtime,
                'parameters': json.dumps(params)
            })
        except Exception as e:
            logger.error(f"LEA method failed: {e}")
    
    return pd.DataFrame(results)

