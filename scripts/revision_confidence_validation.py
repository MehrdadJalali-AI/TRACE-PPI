#!/usr/bin/env python3
"""Assignment-level Core versus non-Core gold-support validation."""
from __future__ import annotations

import pandas as pd
from scipy.stats import fisher_exact

from revision_utils import (
    REV_RESULTS,
    REV_TABLES,
    clusters_from_membership,
    ensure_revision_dirs,
    gold_supported_mask,
    load_trace_assignments,
    load_gold,
)


def _counts(df: pd.DataFrame) -> dict:
    core = df["is_core"]
    gold = df["gold_supported"]
    core_yes = int((core & gold).sum())
    core_n = int(core.sum())
    non_yes = int((~core & gold).sum())
    non_n = int((~core).sum())
    table = [[core_yes, core_n - core_yes], [non_yes, non_n - non_yes]]
    odds, pval = fisher_exact(table, alternative="two-sided")
    return {
        "core_gold_supported": core_yes,
        "core_total": core_n,
        "noncore_gold_supported": non_yes,
        "noncore_total": non_n,
        "core_fraction": core_yes / core_n if core_n else 0.0,
        "noncore_fraction": non_yes / non_n if non_n else 0.0,
        "odds_ratio": float(odds),
        "p_value": float(pval),
        "test_used": "Fisher exact (two-sided)",
        "contingency_core_yes_no": f"{core_yes}/{core_n - core_yes}",
        "contingency_noncore_yes_no": f"{non_yes}/{non_n - non_yes}",
    }


def main() -> None:
    ensure_revision_dirs()
    gold = load_gold()
    rows = []
    label_rows = []
    for dataset in ("gavin", "krogan"):
        df = load_trace_assignments(dataset)
        clusters = clusters_from_membership(df)
        df = df.copy()
        df["gold_supported"] = gold_supported_mask(df, clusters, gold)
        df["is_core"] = df["confidence_label"] == "core"
        stats = _counts(df)
        stats["dataset"] = dataset
        rows.append(stats)
        by_label = (
            df.groupby("confidence_label")
            .agg(
                n_assignments=("protein_id", "size"),
                gold_supported=("gold_supported", "sum"),
                gold_supported_fraction=("gold_supported", "mean"),
                topology_mean=("topology_score", "mean"),
                semantic_mean=("semantic_score", "mean"),
                membership_mean=("membership_score", "mean"),
            )
            .reset_index()
        )
        by_label["dataset"] = dataset
        label_rows.append(by_label)
        df[["community_id", "protein_id", "confidence_label", "gold_supported", "topology_score", "semantic_score", "go_score"]].to_csv(
            REV_RESULTS / f"{dataset}_assignment_gold_support.csv",
            index=False,
        )
        print(dataset, stats)

    out = pd.DataFrame(rows)
    out.to_csv(REV_TABLES / "table_core_vs_noncore_validation.csv", index=False)
    out.to_csv(REV_RESULTS / "core_vs_noncore_validation.csv", index=False)
    pd.concat(label_rows, ignore_index=True).to_csv(REV_RESULTS / "label_gold_support_by_dataset.csv", index=False)


if __name__ == "__main__":
    main()
