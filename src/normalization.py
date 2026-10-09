"""
Score normalization for combining permanence and functional dependency.

LEAF-PPI uses per-evaluation min-max normalization (Option A) so that
M(p, C_i) = alpha * norm_perm(p, C_i) + (1 - alpha) * norm_fd(p, C_i)
with both components on [0, 1] before weighting.
"""

from typing import Dict, Iterable, Tuple
import numpy as np


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
