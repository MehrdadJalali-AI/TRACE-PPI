#!/usr/bin/env python3
"""Quantify how TRACE-PPI refinement changes the initial MCL seed."""
from __future__ import annotations

import json
from collections import defaultdict

import pandas as pd

from revision_utils import (
    REV_RESULTS,
    REV_TABLES,
    assignment_set,
    trace_from_seed,
    ensure_revision_dirs,
    load_config,
    load_dataset,
    load_embeddings,
    protein_to_modules,
)


def _compare_stages(mcl, overlap, trace):
    mcl_asg = assignment_set(mcl)
    ov_asg = assignment_set(overlap)
    trace_asg = assignment_set(trace)
    mcl_map = protein_to_modules(mcl)
    ov_map = protein_to_modules(overlap)

    added_overlap_asg = ov_asg - mcl_asg
    proteins_with_added_overlap = {p for _, p in added_overlap_asg if p in mcl_map and (ov_map.get(p, set()) & mcl_map[p])}

    transferred_proteins = set()
    transfer_ops = 0
    for p, old_mods in mcl_map.items():
        new_mods = ov_map.get(p, set())
        left = old_mods - new_mods
        joined = new_mods - old_mods
        if left:
            transferred_proteins.add(p)
            transfer_ops += len(left)

    supp_asg = trace_asg - ov_asg
    proteins_added_supp = {p for _, p in supp_asg}

    def _frozen(clusters):
        return {cid: frozenset(mem) for cid, mem in clusters.items() if len(mem) >= 2}

    mcl_f = _frozen(mcl)
    trace_f = _frozen(trace)
    changed = 0
    for cid, mem in mcl_f.items():
        if trace_f.get(cid, frozenset()) != mem:
            changed += 1
    extra_trace = set(trace_f) - set(mcl_f)
    changed += len(extra_trace)
    n_final = max(1, len(trace_f))
    return {
        "initial_mcl_modules": len(mcl_f),
        "final_trace_modules": len(trace_f),
        "proteins_with_added_overlap": len(proteins_with_added_overlap),
        "added_overlap_assignments": len(added_overlap_asg),
        "proteins_transferred": len(transferred_proteins),
        "transfer_operations": int(transfer_ops),
        "proteins_added_by_supplementation": len(proteins_added_supp),
        "supplementation_assignments": len(supp_asg),
        "modules_changed_relative_to_mcl": changed,
        "pct_modules_changed": 100.0 * changed / n_final,
        "total_assignments_before": len(mcl_asg),
        "total_assignments_after": len(trace_asg),
    }


def main() -> None:
    ensure_revision_dirs()
    from trace_ppi import baselines

    cfg = load_config()
    supplement = dict(cfg.get("supplement", {}))
    rows = []
    details = {}
    for dataset in ("gavin", "krogan"):
        graph, go = load_dataset(dataset)
        emb = load_embeddings(dataset)
        mcl = baselines.mcl_only(graph, inflation=2.0)
        overlap = baselines.mcl_overlap_heuristic(graph, go, inflation=2.0)
        trace_clusters, trace_mem, supp_stats = trace_from_seed(overlap, graph, go, emb, supplement)
        stats = _compare_stages(mcl, overlap, trace_clusters)
        stats.update({"dataset": dataset, "seed_inflation": 2.0, **{f"supplement_{k}": v for k, v in supp_stats.items()}})
        rows.append(stats)
        trace_mem.to_csv(REV_RESULTS / f"{dataset}_trace_from_mcl2_membership.csv", index=False)
        pd.DataFrame(rows).to_csv(REV_TABLES / "table_refinement_impact.csv", index=False)
        pd.DataFrame(rows).to_csv(REV_RESULTS / "refinement_impact.csv", index=False)
        details[dataset] = stats
        print(dataset, json.dumps(stats, indent=2))

    df = pd.DataFrame(rows)
    df.to_csv(REV_TABLES / "table_refinement_impact.csv", index=False)
    df.to_csv(REV_RESULTS / "refinement_impact.csv", index=False)


if __name__ == "__main__":
    main()
