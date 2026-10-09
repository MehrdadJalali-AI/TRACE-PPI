"""
Gold-standard complex matching metrics for overlapping community detection.
"""

from typing import Dict, Set, List, Tuple
import itertools
import numpy as np
import pandas as pd


def clusters_to_protein_sets(clusters: Dict[int, Set[str]]) -> List[Set[str]]:
    return [c for c in clusters.values() if len(c) >= 2]


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def best_matching_pairs(
    predicted: List[Set[str]],
    gold: List[Set[str]],
    threshold: float = 0.0,
) -> List[Tuple[int, int, float]]:
    """Greedy maximum-weight matching on Jaccard similarity."""
    pairs = []
    for i, p in enumerate(predicted):
        for j, g in enumerate(gold):
            jac = jaccard(p, g)
            if jac >= threshold:
                pairs.append((jac, i, j))
    pairs.sort(reverse=True)
    used_p, used_g, matched = set(), set(), []
    for jac, i, j in pairs:
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        matched.append((i, j, jac))
    return matched


def precision_recall_f1_mmr(
    predicted: Dict[int, Set[str]],
    gold: Dict[int, Set[str]],
    jaccard_threshold: float = 0.5,
) -> Dict[str, float]:
    pred_list = clusters_to_protein_sets(predicted)
    gold_list = clusters_to_protein_sets(gold)
    if not pred_list or not gold_list:
        return {
            'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'mmr': 0.0,
            'coverage': 0.0, 'sensitivity': 0.0, 'ppv': 0.0, 'accuracy': 0.0,
        }

    matched = best_matching_pairs(pred_list, gold_list, threshold=jaccard_threshold)
    tp = len(matched)
    precision = tp / len(pred_list)
    recall = tp / len(gold_list)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    mmr = sum(j for _, _, j in matched) / len(gold_list)

    gold_proteins = set().union(*gold_list)
    pred_proteins = set().union(*pred_list)
    covered = set()
    for i, j, _ in matched:
        covered |= pred_list[i] & gold_list[j]
    coverage = len(covered) / len(gold_proteins) if gold_proteins else 0.0
    sensitivity = recall
    ppv = precision
    accuracy = len(covered) / len(gold_proteins | pred_proteins) if (gold_proteins | pred_proteins) else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mmr': mmr,
        'coverage': coverage,
        'sensitivity': sensitivity,
        'ppv': ppv,
        'accuracy': accuracy,
    }


def omega_index_approx(predicted: Dict[int, Set[str]], gold: Dict[int, Set[str]]) -> float:
    """Simplified Omega-like agreement from mean best Jaccard per gold complex."""
    pred_list = clusters_to_protein_sets(predicted)
    gold_list = clusters_to_protein_sets(gold)
    if not gold_list:
        return 0.0
    scores = []
    for g in gold_list:
        best = max((jaccard(p, g) for p in pred_list), default=0.0)
        scores.append(best)
    return float(np.mean(scores))


def evaluate_against_gold(
    predicted: Dict[int, Set[str]],
    gold: Dict[int, Set[str]],
) -> pd.Series:
    metrics = precision_recall_f1_mmr(predicted, gold)
    metrics['omega_index'] = omega_index_approx(predicted, gold)
    return pd.Series(metrics)


def load_gold_standard_csv(path: str) -> Dict[int, Set[str]]:
    df = pd.read_csv(path)
    gold = {}
    for _, row in df.iterrows():
        cid = row['cluster_id']
        pid = str(row['protein_id'])
        gold.setdefault(cid, set()).add(pid)
    return gold
