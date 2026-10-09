"""Biological and evidence-quality metrics for predicted modules."""
from __future__ import annotations

from typing import Dict, Set

import numpy as np
import pandas as pd

from .reuse import GENERIC_GO
from .v2_membership import overlap_fraction


def module_size_stats(clusters: Dict[int, Set[str]]) -> dict:
    sizes = [len(c) for c in clusters.values() if len(c) >= 2]
    if not sizes:
        return dict(mean_size=0.0, median_size=0.0, num_modules=0)
    return dict(
        mean_size=float(np.mean(sizes)),
        median_size=float(np.median(sizes)),
        num_modules=len(sizes),
    )


def go_coherence_score(modules: Dict[int, Set[str]], go_map: dict) -> float:
    scores = []
    for mem in modules.values():
        if len(mem) < 2:
            continue
        terms = [go_map.get(p, set()) - GENERIC_GO for p in mem]
        if not any(terms):
            continue
        inter = set.intersection(*terms) if all(terms) else set()
        union = set().union(*terms)
        scores.append(len(inter) / len(union) if union else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def functional_specificity(modules: Dict[int, Set[str]], go_map: dict) -> float:
    """Higher when modules use fewer generic-only proteins."""
    vals = []
    for mem in modules.values():
        if len(mem) < 2:
            continue
        specific = sum(1 for p in mem if (go_map.get(p, set()) - GENERIC_GO))
        vals.append(specific / len(mem))
    return float(np.mean(vals)) if vals else 0.0


def modules_with_nongeneric_go(modules: Dict[int, Set[str]], go_map: dict) -> float:
    if not modules:
        return 0.0
    ok = sum(
        1
        for mem in modules.values()
        if len(mem) >= 2 and any((go_map.get(p, set()) - GENERIC_GO) for p in mem)
    )
    n = len([m for m in modules.values() if len(m) >= 2])
    return ok / n if n else 0.0


def evidence_bundle_completeness(membership: pd.DataFrame) -> float:
    """Fraction of assignments with the required exported audit fields present.

    This is a documentation-coverage metric. A score of zero in an evidence
    channel can be meaningful evidence, so completeness should not require
    every channel to be positive.
    """
    if membership.empty:
        return 0.0
    required = {
        "protein_id",
        "community_id",
        "membership_type",
        "membership_score",
        "topology_score",
        "semantic_score",
        "go_score",
    }
    if not required.issubset(membership.columns):
        return 0.0
    complete_rows = membership[list(required)].notna().all(axis=1)
    return float(complete_rows.mean())


def compute_all_evidence_metrics(
    clusters: Dict[int, Set[str]],
    go_map: dict,
    membership: pd.DataFrame | None = None,
    runtime_sec: float = 0.0,
) -> dict:
    sizes = module_size_stats(clusters)
    ovl = overlap_fraction(membership) if membership is not None and not membership.empty else 0.0
    if membership is None or membership.empty:
        # build pseudo overlap from clusters
        rows = []
        for cid, mem in clusters.items():
            for p in mem:
                rows.append({"protein_id": p, "community_id": cid})
        membership = pd.DataFrame(rows, columns=["protein_id", "community_id"])
        ovl = overlap_fraction(membership)

    n_proteins = len(set().union(*clusters.values())) if clusters else 0
    if membership.empty or "protein_id" not in membership.columns or "community_id" not in membership.columns:
        n_overlap = 0
    else:
        multi = membership.groupby("protein_id")["community_id"].nunique()
        n_overlap = int((multi > 1).sum())

    return {
        **sizes,
        "n_proteins_assigned": n_proteins,
        "n_overlapping_proteins": n_overlap,
        "overlap_protein_fraction": ovl,
        "go_coherence": go_coherence_score(clusters, go_map),
        "functional_specificity": functional_specificity(clusters, go_map),
        "modules_with_nongeneric_go_fraction": modules_with_nongeneric_go(clusters, go_map),
        "evidence_bundle_completeness": evidence_bundle_completeness(membership)
        if membership is not None and not membership.empty
        else 0.0,
        "runtime_sec": runtime_sec,
    }
