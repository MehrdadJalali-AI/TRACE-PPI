#!/usr/bin/env python3
"""Sensitivity of confidence labels and Core enrichment to threshold shifts."""
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


def assign_labels(topo, sem, core_t=0.35, core_s=0.25, inner=0.25, outer=0.12) -> str:
    if topo >= core_t and sem >= core_s:
        return "core"
    if topo >= inner or sem >= inner:
        return "inner"
    if topo >= outer or sem >= outer:
        return "outer"
    return "uncertain"


def _core_enrichment(labels, gold_flags):
    is_core = labels == "core"
    core_yes = int((is_core & gold_flags).sum())
    core_n = int(is_core.sum())
    non_yes = int((~is_core & gold_flags).sum())
    non_n = int((~is_core).sum())
    if core_n == 0 or non_n == 0:
        return core_yes, core_n, non_yes, non_n, float("nan"), float("nan")
    odds, pval = fisher_exact([[core_yes, core_n - core_yes], [non_yes, non_n - non_yes]])
    return core_yes, core_n, non_yes, non_n, float(odds), float(pval)


def main() -> None:
    ensure_revision_dirs()
    gold = load_gold()
    settings = [
        dict(name="default", core_t=0.35, core_s=0.25, inner=0.25, outer=0.12),
        dict(name="inner_0.18", core_t=0.35, core_s=0.25, inner=0.18, outer=0.12),
        dict(name="inner_0.32", core_t=0.35, core_s=0.25, inner=0.32, outer=0.12),
        dict(name="core_relaxed", core_t=0.30, core_s=0.20, inner=0.25, outer=0.12),
        dict(name="core_stricter", core_t=0.40, core_s=0.30, inner=0.25, outer=0.12),
        dict(name="outer_0.08", core_t=0.35, core_s=0.25, inner=0.25, outer=0.08),
        dict(name="outer_0.16", core_t=0.35, core_s=0.25, inner=0.25, outer=0.16),
        dict(name="all_minus_0.05", core_t=0.30, core_s=0.20, inner=0.20, outer=0.07),
        dict(name="all_plus_0.05", core_t=0.40, core_s=0.30, inner=0.30, outer=0.17),
    ]
    rows = []
    for dataset in ("gavin", "krogan"):
        df = load_trace_assignments(dataset)
        clusters = clusters_from_membership(df)
        gold_flags = gold_supported_mask(df, clusters, gold).to_numpy()
        topo = df["topology_score"].fillna(0).astype(float).to_numpy()
        sem = df["semantic_score"].fillna(0).astype(float).to_numpy()
        for s in settings:
            labels = pd.Series(
                [assign_labels(t, se, s["core_t"], s["core_s"], s["inner"], s["outer"]) for t, se in zip(topo, sem)]
            )
            core_yes, core_n, non_yes, non_n, odds, pval = _core_enrichment(labels, gold_flags)
            vc = labels.value_counts()
            rows.append(
                {
                    "dataset": dataset,
                    "setting": s["name"],
                    "core_topology_threshold": s["core_t"],
                    "core_semantic_threshold": s["core_s"],
                    "inner_threshold": s["inner"],
                    "outer_threshold": s["outer"],
                    "n_core": int(vc.get("core", 0)),
                    "n_inner": int(vc.get("inner", 0)),
                    "n_outer": int(vc.get("outer", 0)),
                    "n_uncertain": int(vc.get("uncertain", 0)),
                    "core_gold_supported": core_yes,
                    "core_total": core_n,
                    "noncore_gold_supported": non_yes,
                    "noncore_total": non_n,
                    "core_fraction": core_yes / core_n if core_n else 0.0,
                    "noncore_fraction": non_yes / non_n if non_n else 0.0,
                    "odds_ratio": odds,
                    "p_value": pval,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(REV_TABLES / "table_label_threshold_sensitivity.csv", index=False)
    out.to_csv(REV_RESULTS / "label_threshold_sensitivity.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
