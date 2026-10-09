"""Candidate-, module-, and membership-level evaluation for overlapping covers."""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Dict, Mapping, Set

import numpy as np
import pandas as pd

from .metrics_utils import jaccard


def _valid(cover: Mapping[int, Set[str]]) -> Dict[object, Set[str]]:
    """Retain caller identifiers: CYC2008 IDs are strings, module IDs integers."""
    return {k: {str(x) for x in v} for k, v in cover.items() if len(v) >= 2}


def candidate_coverage(candidates: Mapping[int, Set[str]], gold: Mapping[int, Set[str]], match_jaccard: float = .5) -> tuple[dict, pd.DataFrame]:
    """Measure whether candidate construction can expose each reference complex."""
    candidates, gold = _valid(candidates), _valid(gold)
    rows = []
    for gid, reference in gold.items():
        scores = [(cid, jaccard(reference, candidate), len(reference & candidate) / len(reference)) for cid, candidate in candidates.items()]
        best = max(scores, key=lambda x: x[1]) if scores else (None, 0.0, 0.0)
        substantial = [cid for cid, _, recall in scores if recall >= .5]
        rows.append({"reference_complex_id": gid, "best_candidate_id": best[0], "best_jaccard": best[1],
                     "best_reference_coverage": best[2], "candidate_count_with_50pct_reference": len(substantial),
                     "status": "covered" if best[1] >= match_jaccard else ("partial" if best[2] > 0 else "no_candidate")})
    detail = pd.DataFrame(rows)
    merge_count = 0
    for candidate in candidates.values():
        hits = [g for g in gold.values() if len(candidate & g) / len(g) >= .5]
        merge_count += int(len(hits) >= 2)
    summary = {
        "candidate_recall": float((detail.best_jaccard >= match_jaccard).mean()) if len(detail) else 0.0,
        "candidate_coverage": float(detail.best_reference_coverage.mean()) if len(detail) else 0.0,
        "candidate_purity": float(np.mean([max((jaccard(c, g) for g in gold.values()), default=0.0) for c in candidates.values()])) if candidates else 0.0,
        "candidate_fragmentation": float(detail.candidate_count_with_50pct_reference.mean()) if len(detail) else 0.0,
        "candidate_merge_rate": merge_count / max(1, len(candidates)),
        "n_candidates": len(candidates), "n_reference": len(gold),
        "no_candidate_references": int((detail.status == "no_candidate").sum()) if len(detail) else 0,
        "partial_candidate_references": int((detail.status == "partial").sum()) if len(detail) else 0,
    }
    return summary, detail


def _best_reference_map(predicted: Mapping[int, Set[str]], gold: Mapping[int, Set[str]], min_jaccard: float = .25) -> Dict[int, int]:
    """Map each predicted module to its best reference independently.

    This many-to-one mapping permits fragmented predicted candidates to be
    counted as such and avoids forcing a one-to-one module score onto the
    assignment-level task.
    """
    mapping = {}
    for cid, module in _valid(predicted).items():
        gid, score = max(((gid, jaccard(module, ref)) for gid, ref in _valid(gold).items()), key=lambda x: x[1], default=(None, 0.0))
        if gid is not None and score >= min_jaccard:
            mapping[cid] = gid
    return mapping


def membership_pairs(predicted: Mapping[int, Set[str]], gold: Mapping[int, Set[str]], min_jaccard: float = .25) -> tuple[Set[tuple[str, int]], Set[tuple[str, int]], Dict[int, int]]:
    """Build reference-keyed predicted and gold (protein, complex) pairs."""
    gold = _valid(gold)
    mapping = _best_reference_map(predicted, gold, min_jaccard=min_jaccard)
    pred_pairs = {(str(p), mapping[cid]) for cid, members in _valid(predicted).items() if cid in mapping for p in members}
    gold_pairs = {(str(p), gid) for gid, members in gold.items() for p in members}
    return pred_pairs, gold_pairs, mapping


def _prf(predicted: Set[tuple[str, int]], truth: Set[tuple[str, int]]) -> dict:
    correct = predicted & truth
    precision = len(correct) / len(predicted) if predicted else 0.0
    recall = len(correct) / len(truth) if truth else 0.0
    return {"precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "correct": len(correct), "false": len(predicted - truth), "missed": len(truth - predicted)}


def evaluate_membership_pairs(predicted: Mapping[int, Set[str]], gold: Mapping[int, Set[str]], min_jaccard: float = .25) -> dict:
    pred_pairs, gold_pairs, mapping = membership_pairs(predicted, gold, min_jaccard)
    result = _prf(pred_pairs, gold_pairs)
    return {f"membership_{k}": v for k, v in result.items()} | {"mapped_predicted_modules": len(mapping)}


def evaluate_shared_proteins(predicted: Mapping[int, Set[str]], gold: Mapping[int, Set[str]], min_jaccard: float = .25) -> dict:
    pred_pairs, gold_pairs, _ = membership_pairs(predicted, gold, min_jaccard)
    truth_by_protein: Dict[str, Set[int]] = defaultdict(set)
    pred_by_protein: Dict[str, Set[int]] = defaultdict(set)
    for p, c in gold_pairs:
        truth_by_protein[p].add(c)
    for p, c in pred_pairs:
        pred_by_protein[p].add(c)
    shared = {p for p, memberships in truth_by_protein.items() if len(memberships) >= 2}
    true_shared_pairs = {(p, c) for p, c in gold_pairs if p in shared}
    predicted_shared_pairs = {(p, c) for p, c in pred_pairs if p in shared and len(pred_by_protein[p]) >= 2}
    result = _prf(predicted_shared_pairs, true_shared_pairs)
    correct_secondary = len(predicted_shared_pairs & true_shared_pairs)
    return {
        **{f"shared_membership_{k}": v for k, v in result.items()},
        "n_reference_shared_proteins": len(shared),
        "n_true_shared_proteins_recovered": len({p for p, _ in predicted_shared_pairs & true_shared_pairs}),
        "correct_secondary_memberships": correct_secondary,
        "false_secondary_memberships": len(predicted_shared_pairs - true_shared_pairs),
        "missed_secondary_memberships": len(true_shared_pairs - predicted_shared_pairs),
    }


def omega_index(predicted: Mapping[int, Set[str]], gold: Mapping[int, Set[str]]) -> float:
    """Adjusted Omega index based on agreement in shared-community counts."""
    nodes = sorted(set().union(*_valid(predicted).values(), *_valid(gold).values()))
    if len(nodes) < 2:
        return 0.0
    pred_by = defaultdict(set)
    gold_by = defaultdict(set)
    for cid, members in _valid(predicted).items():
        for p in members: pred_by[p].add(cid)
    for gid, members in _valid(gold).items():
        for p in members: gold_by[p].add(gid)
    total = len(nodes) * (len(nodes) - 1) // 2
    observed = defaultdict(int); pred_hist = defaultdict(int); gold_hist = defaultdict(int)
    for a, b in combinations(nodes, 2):
        pc = len(pred_by[a] & pred_by[b]); gc = len(gold_by[a] & gold_by[b])
        pred_hist[pc] += 1; gold_hist[gc] += 1
        if pc == gc: observed[pc] += 1
    obs = sum(observed.values()) / total
    expected = sum(pred_hist[k] * gold_hist[k] for k in set(pred_hist) | set(gold_hist)) / (total * total)
    return (obs - expected) / (1 - expected) if expected < 1 else 0.0


def overlapping_nmi(predicted: Mapping[int, Set[str]], gold: Mapping[int, Set[str]]) -> float:
    """Cover NMI from binary membership vectors (LFK-style conditional entropy)."""
    nodes = sorted(set().union(*_valid(predicted).values(), *_valid(gold).values()))
    if not nodes:
        return 0.0
    n = len(nodes)
    def entropy(x: Set[str]) -> float:
        p = len(x) / n
        return -sum(v * np.log2(v) for v in (p, 1 - p) if v > 0)
    def cond(x: Set[str], y: Set[str]) -> float:
        table = [len(x & y), len(x - y), len(y - x), n - len(x | y)]
        hx = entropy(x)
        if hx == 0: return 0.0
        hy = entropy(y)
        joint = 0.0
        for count in table:
            if count:
                p = count / n; joint -= p * np.log2(p)
        return max(0.0, joint - hy)
    pcover, gcover = list(_valid(predicted).values()), list(_valid(gold).values())
    if not pcover or not gcover: return 0.0
    hpg = np.mean([min(cond(p, g) / max(entropy(p), 1e-12) for g in gcover) for p in pcover])
    hgp = np.mean([min(cond(g, p) / max(entropy(g), 1e-12) for p in pcover) for g in gcover])
    return float(1.0 - 0.5 * (hpg + hgp))


def merged_cluster_cases(
    coarse: Mapping[int, Set[str]], original: Mapping[int, Set[str]], deconvolved: Mapping[int, Set[str]],
    gold: Mapping[int, Set[str]], provenance: Mapping[int, Mapping[str, object]] | None = None,
) -> pd.DataFrame:
    """Stress-test MCL clusters that substantially contain multiple references."""
    rows = []
    for cid, coarse_members in _valid(coarse).items():
        refs = [(gid, ref) for gid, ref in _valid(gold).items() if len(ref & coarse_members) / len(ref) >= .5]
        if len(refs) < 2:
            continue
        ref_ids = [gid for gid, _ in refs]
        shared = set().union(*(a & b for _, a in refs for _, b in refs if a is not b))
        new_local = {mid: m for mid, m in _valid(deconvolved).items() if (provenance or {}).get(mid, {}).get("coarse_cluster_id") == cid}
        def recovery(modules):
            return float(np.mean([max((jaccard(ref, m) for m in modules.values()), default=0.0) for _, ref in refs]))
        # Membership metrics are restricted to just the reference complexes in
        # the merged coarse location, which directly tests the reviewer's case.
        reference_subset = dict(refs)
        orig_m = evaluate_membership_pairs(original, reference_subset)
        new_m = evaluate_membership_pairs(new_local or deconvolved, reference_subset)
        rows.append({"initial_mcl_cluster": cid, "reference_complexes_contained": ";".join(map(str, ref_ids)),
                     "n_reference_complexes": len(refs), "shared_proteins": ";".join(sorted(shared)),
                     "n_shared_proteins": len(shared), "original_trace_recovery": recovery(original),
                     "new_deconvolution_recovery": recovery(new_local or deconvolved),
                     "original_membership_precision": orig_m["membership_precision"], "original_membership_recall": orig_m["membership_recall"],
                     "new_membership_precision": new_m["membership_precision"], "new_membership_recall": new_m["membership_recall"],
                     "new_subcomplexes": len(new_local)})
    return pd.DataFrame(rows)
