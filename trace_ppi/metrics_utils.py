"""Shared clustering metrics and gold-standard helpers."""
from __future__ import annotations

from typing import Dict, List, Set, Tuple

import numpy as np


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def clusters_from_membership(df) -> Dict[int, Set[str]]:
    out: Dict[int, Set[str]] = {}
    cid_col = "community_id" if "community_id" in df.columns else "candidate_id"
    for _, r in df.iterrows():
        cid = int(r[cid_col])
        out.setdefault(cid, set()).add(str(r["protein_id"]))
    return {k: v for k, v in out.items() if len(v) >= 2}


def oracle_upper_bound(
    candidates: Dict[int, Set[str]],
    gold: Dict[int, Set[str]],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Best possible recall if we pick the best candidate per reference complex."""
    gold_list = [g for g in gold.values() if len(g) >= 2]
    cand_list = [c for c in candidates.values() if len(c) >= 2]
    if not gold_list or not cand_list:
        return {"oracle_recall": 0.0, "oracle_precision_upper": 0.0, "oracle_f1_upper": 0.0}
    matched_gold = 0
    matched_cand = set()
    for g in gold_list:
        best = max((jaccard(g, c), i) for i, c in enumerate(cand_list))
        if best[0] >= threshold:
            matched_gold += 1
            matched_cand.add(best[1])
    oracle_recall = matched_gold / len(gold_list)
    oracle_precision = len(matched_cand) / len(cand_list) if cand_list else 0.0
    f1 = (
        2 * oracle_precision * oracle_recall / (oracle_precision + oracle_recall)
        if (oracle_precision + oracle_recall)
        else 0.0
    )
    return {
        "oracle_recall": oracle_recall,
        "oracle_precision_upper": oracle_precision,
        "oracle_f1_upper": f1,
        "n_candidates": len(cand_list),
        "n_gold": len(gold_list),
    }


def split_gold_complexes(
    gold: Dict[int, Set[str]], seed: int = 42, train_frac: float = 0.6, val_frac: float = 0.2
) -> Tuple[Dict[int, Set[str]], Dict[int, Set[str]], Dict[int, Set[str]]]:
    rng = np.random.default_rng(seed)
    ids = [k for k, v in gold.items() if len(v) >= 2]
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train_ids = set(ids[:n_train])
    val_ids = set(ids[n_train : n_train + n_val])
    test_ids = set(ids[n_train + n_val :])
    train = {k: gold[k] for k in train_ids}
    val = {k: gold[k] for k in val_ids}
    test = {k: gold[k] for k in test_ids}
    return train, val, test
