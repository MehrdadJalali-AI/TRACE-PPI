#!/usr/bin/env python3
"""Compare MCL, MCL+overlap, and TRACE-PPI at inflation 2.0 and 3.0."""
from __future__ import annotations

import pandas as pd

from revision_utils import (
    REV_RESULTS,
    REV_TABLES,
    trace_from_seed,
    ensure_revision_dirs,
    evaluate_method,
    load_config,
    load_dataset,
    load_embeddings,
    load_gold,
    membership_from_clusters,
)


def main() -> None:
    ensure_revision_dirs()
    from trace_ppi import baselines

    gold = load_gold()
    cfg = load_config()
    supplement = dict(cfg.get("supplement", {}))
    rows = []
    datasets = ("gavin", "krogan")
    inflations = (2.0, 3.0)
    for dataset in datasets:
        graph, go = load_dataset(dataset)
        emb = load_embeddings(dataset)
        for inflation in inflations:
            mcl = baselines.mcl_only(graph, inflation=inflation)
            mcl_metrics = evaluate_method(mcl, gold)
            rows.append(
                {
                    "dataset": dataset,
                    "seed_inflation": inflation,
                    "method": "MCL",
                    **mcl_metrics,
                    "notes": "Python MCL fallback; unweighted adjacency",
                }
            )

            overlap = baselines.mcl_overlap_heuristic(graph, go, inflation=inflation)
            ov_metrics = evaluate_method(overlap, gold, membership_from_clusters(overlap))
            rows.append(
                {
                    "dataset": dataset,
                    "seed_inflation": inflation,
                    "method": "MCL+overlap",
                    **ov_metrics,
                    "notes": "Function-aware overlap on the MCL seed",
                }
            )

            trace_clusters, trace_mem, supp_stats = trace_from_seed(overlap, graph, go, emb, supplement)
            trace_metrics = evaluate_method(trace_clusters, gold, trace_mem)
            rows.append(
                {
                    "dataset": dataset,
                    "seed_inflation": inflation,
                    "method": "TRACE-PPI",
                    **trace_metrics,
                    "notes": f"Recall-safe supplement added {supp_stats.get('proteins_added', 0)} proteins",
                }
            )
            trace_mem.to_csv(
                REV_RESULTS / f"{dataset}_trace_inflation{str(inflation).replace('.', '_')}_membership.csv",
                index=False,
            )
            print(dataset, inflation, rows[-3]["method"], rows[-3]["f1"], rows[-2]["f1"], rows[-1]["f1"])

    keep = [
        "dataset",
        "seed_inflation",
        "method",
        "f1",
        "precision",
        "recall",
        "mean_size",
        "audit_field_availability",
        "n_modules",
        "n_assignments",
        "notes",
    ]
    df = pd.DataFrame(rows)
    df[keep].to_csv(REV_TABLES / "table_tuned_seed_sensitivity.csv", index=False)
    df.to_csv(REV_RESULTS / "tuned_seed_sensitivity.csv", index=False)


if __name__ == "__main__":
    main()
