"""
GO hold-out validation to reduce circularity: train on BP, evaluate MF+CC coherence.
"""

from typing import Dict, Set, List, Tuple
import random
import numpy as np
import pandas as pd

BP_PREFIX = 'GO:'  # filtered by aspect column in GAF when available


def split_go_by_aspect(
    protein_go_terms: Dict[str, Set[str]],
    aspect_map: Dict[str, str],
    train_aspects: Set[str] = frozenset({'P', 'biological_process'}),
    eval_aspects: Set[str] = frozenset({'F', 'C', 'molecular_function', 'cellular_component'}),
) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    """Split GO terms per protein by ontology aspect."""
    train, held = {}, {}
    for protein, terms in protein_go_terms.items():
        train[protein] = {t for t in terms if aspect_map.get(t, 'P') in train_aspects}
        held[protein] = {t for t in terms if aspect_map.get(t, 'P') in eval_aspects}
    return train, held


def random_holdout_per_protein(
    protein_go_terms: Dict[str, Set[str]],
    holdout_fraction: float = 0.2,
    seed: int = 42,
) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    rng = random.Random(seed)
    train, held = {}, {}
    for protein, terms in protein_go_terms.items():
        terms = list(terms)
        if len(terms) < 2:
            train[protein] = set(terms)
            held[protein] = set()
            continue
        k = max(1, int(len(terms) * holdout_fraction))
        held_set = set(rng.sample(terms, k))
        train[protein] = set(terms) - held_set
        held[protein] = held_set
    return train, held


def community_heldout_coherence(
    clusters: Dict[int, Set[str]],
    heldout_go: Dict[str, Set[str]],
) -> pd.DataFrame:
    """Fraction of held-out GO terms shared within each community."""
    rows = []
    for cid, members in clusters.items():
        if len(members) < 2:
            continue
        term_sets = [heldout_go.get(p, set()) for p in members]
        union = set().union(*term_sets) if term_sets else set()
        if not union:
            continue
        shared = set.intersection(*[s for s in term_sets if s]) if term_sets else set()
        rows.append({
            'community_id': cid,
            'n_proteins': len(members),
            'n_heldout_terms': len(union),
            'coherence': len(shared) / len(union) if union else 0.0,
        })
    return pd.DataFrame(rows)


def summarize_holdout(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty:
        return {'mean_coherence': 0.0, 'n_communities': 0}
    return {
        'mean_coherence': float(df['coherence'].mean()),
        'n_communities': int(len(df)),
    }
