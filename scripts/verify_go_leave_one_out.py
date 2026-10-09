#!/usr/bin/env python3
"""Rerun TRACE-PPI after leave-one-out GO support and compare to stored outputs.

GO support excludes the scored protein. Supplementation scores boundary proteins
that are not yet members, so module membership should be unchanged. This script
checks that claim and writes the regenerated channel table and case-study GO.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import yaml
from scipy.stats import fisher_exact

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from trace_ppi.cosmos_v2_runner import run_v2
from trace_ppi.graph_io import load_gavin
from trace_ppi.metrics_utils import jaccard
from trace_ppi.paths import DATA
from trace_ppi.reuse import load_gold_standard_csv, precision_recall_f1_mmr
from trace_ppi import baselines
from trace_ppi.v2_membership import provenance_record
from revision_utils import gold_supported_mask, normalize_label

OUT = ROOT / "results" / "revision"
OUT.mkdir(parents=True, exist_ok=True)
CFG = yaml.safe_load((ROOT / "configs" / "trace_ppi_final.yaml").read_text())
SUPP = CFG.get("supplement", {})
PARAMS = {
    "base_method": "mcl_overlap",
    "skip_expansion": True,
    "preserve_all": True,
    "recall_safe_supplement": True,
    "apply_orbit_labels": True,
    **SUPP,
}


def _sets(df: pd.DataFrame) -> dict:
    return {
        frozenset(sub.protein_id.astype(str))
        for _, sub in df.groupby("community_id")
        if len(sub) >= 2
    }


def _keyed(df: pd.DataFrame) -> dict:
    modules = {
        int(cid): frozenset(sub.protein_id.astype(str))
        for cid, sub in df.groupby("community_id")
    }
    out = {}
    for _, r in df.iterrows():
        key = (str(r.protein_id), modules[int(r.community_id)])
        out[key] = r
    return out


def _channel_table(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["lab"] = d.membership_type.map(normalize_label)
    sizes = d.groupby("community_id").protein_id.transform("size")
    d = d[sizes >= 2]
    d["mc"] = (
        (d.topology_score.astype(float) > 0).astype(int)
        + (d.semantic_score.astype(float) > 0).astype(int)
        + (d.go_score.astype(float) > 0).astype(int)
    ) >= 2
    g = (
        d.groupby("lab")
        .agg(
            n=("lab", "size"),
            tau=("topology_score", "mean"),
            sigma=("semantic_score", "mean"),
            go=("go_score", "mean"),
            multi=("mc", "mean"),
        )
        .reindex(["core", "inner", "outer", "uncertain"])
    )
    return g


def _or(df: pd.DataFrame, clusters: dict, gold: dict) -> dict:
    d = df.copy()
    sizes = d.groupby("community_id").protein_id.transform("size")
    d = d[sizes >= 2].copy()
    flags = gold_supported_mask(d, clusters, gold)
    labels = d.membership_type.map(normalize_label)
    core = labels.eq("core").to_numpy()
    gold_yes = flags.to_numpy()
    core_yes = int((core & gold_yes).sum())
    core_n = int(core.sum())
    non_yes = int((~core & gold_yes).sum())
    non_n = int((~core).sum())
    odds, pval = fisher_exact([[core_yes, core_n - core_yes], [non_yes, non_n - non_yes]])
    return {
        "core_gold": core_yes,
        "core_n": core_n,
        "non_gold": non_yes,
        "non_n": non_n,
        "odds_ratio": float(odds),
        "p": float(pval),
        "labels": labels.value_counts().to_dict(),
    }


def _matchable(clusters: dict, gold: dict) -> dict:
    cand = [(i, set(m)) for i, m in clusters.items() if len(m) >= 2]
    n = 0
    ge = 0
    partial = 0
    zero = 0
    for g in gold.values():
        if len(g) < 2:
            continue
        n += 1
        best = 0.0
        for _, c in cand:
            best = max(best, jaccard(set(g), c))
        if best == 0:
            zero += 1
        elif best < 0.5:
            partial += 1
        else:
            ge += 1
    return {"n_gold": n, "ge_0.5": ge, "partial": partial, "zero": zero,
            "fraction_ge_0.5": ge / n if n else None}


def main() -> None:
    gold = load_gold_standard_csv(str(DATA["gold_standard"]))
    report = {}
    stored_paths = {
        "gavin": ROOT / "results" / "communities" / "trace_ppi_refined_gavin.csv",
        "krogan": ROOT / "results" / "krogan" / "trace_ppi_refined.csv",
    }
    for dataset in ("gavin", "krogan"):
        stored = pd.read_csv(stored_paths[dataset])
        fresh = run_v2(dataset, variant="recall_safe_orbits", params=PARAMS, write_outputs=False)
        refined = fresh["refined"]
        clusters = fresh["clusters"]
        metrics = precision_recall_f1_mmr(clusters, gold)
        old_sets = _sets(stored)
        new_sets = _sets(refined)
        old_k = _keyed(stored)
        new_k = _keyed(refined)
        shared = set(old_k) & set(new_k)
        label_mismatch = 0
        tau_mismatch = 0
        sig_mismatch = 0
        for key in shared:
            a, b = old_k[key], new_k[key]
            if normalize_label(a.membership_type) != normalize_label(b.membership_type):
                label_mismatch += 1
            if abs(float(a.topology_score) - float(b.topology_score)) > 1e-8:
                tau_mismatch += 1
            if abs(float(a.semantic_score) - float(b.semantic_score)) > 1e-6:
                sig_mismatch += 1
        # write leave-one-out GO back onto the stored export, preserving other columns
        go_by_key = {k: float(new_k[k].go_score) for k in shared}
        stored = stored.copy()
        modules = {
            int(cid): frozenset(sub.protein_id.astype(str))
            for cid, sub in stored.groupby("community_id")
        }
        stored["go_score"] = [
            go_by_key.get((str(r.protein_id), modules[int(r.community_id)]), float(r.go_score))
            for _, r in stored.iterrows()
        ]
        if "provenance" not in stored.columns:
            stored["provenance"] = provenance_record(dataset)
        stored.to_csv(stored_paths[dataset], index=False)
        channels = _channel_table(stored)
        channels.to_csv(OUT / f"channel_means_{dataset}.csv")
        enrichment = _or(stored, {i: set(s) for i, s in enumerate(old_sets)}, gold)
        # clusters for OR must use the stored community ids
        stored_clusters = {
            int(cid): set(sub.protein_id.astype(str))
            for cid, sub in stored.groupby("community_id")
        }
        enrichment = _or(stored, stored_clusters, gold)
        report[dataset] = {
            "f1": metrics["f1"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "n_modules_fresh": len(new_sets),
            "n_modules_stored": len(old_sets),
            "module_sets_equal": old_sets == new_sets,
            "n_assignments_shared": len(shared),
            "n_assignments_stored": len(old_k),
            "n_assignments_fresh": len(new_k),
            "label_mismatches": label_mismatch,
            "tau_mismatches": tau_mismatch,
            "sigma_mismatches": sig_mismatch,
            "channels": channels.round(6).reset_index().to_dict(orient="records"),
            "enrichment": enrichment,
            "metrics_full": {k: (float(v) if hasattr(v, "real") else v) for k, v in metrics.items() if k in ("f1", "precision", "recall", "mean_size", "num_modules")},
        }
        print(dataset, "modules_equal", old_sets == new_sets, "F1", round(metrics["f1"], 6),
              "P", round(metrics["precision"], 6), "R", round(metrics["recall"], 6),
              "label_mm", label_mismatch, "OR", round(enrichment["odds_ratio"], 4),
              "labels", enrichment["labels"], flush=True)
        print(channels.round(3).to_string(), flush=True)

    # MCL seed matchable fraction on Gavin, same Jaccard categories as the oracle
    graph, _ = load_gavin()
    mcl = baselines.mcl_only(graph, inflation=2.0)
    report["gavin_mcl_seed_matchable"] = _matchable(mcl, gold)
    print("MCL seed matchable", report["gavin_mcl_seed_matchable"], flush=True)

    # Case-study bundles: recompute GO on the stored high-rank membership
    from trace_ppi.reuse import GENERIC_GO
    from trace_ppi.graph_io import filter_go
    _, go = load_gavin()
    go = filter_go(go)
    hr_path = ROOT / "results" / "high_rank_rerun" / "trace_ppi_assignment_gold_support.csv"
    hr = pd.read_csv(hr_path)
    mems = {int(cid): set(sub.protein_id.astype(str)) for cid, sub in hr.groupby("community_id")}

    def go_only(p, members):
        terms = go.get(p, set()) - GENERIC_GO
        others = set()
        for m in members:
            if m == p:
                continue
            others |= go.get(m, set()) - GENERIC_GO
        return (len(terms & others) / len(terms)) if terms else 0.0

    wanted = hr[hr.protein_id.isin(["YKR018C", "YIL161W"])].copy()
    wanted["go_leave_one_out"] = [
        go_only(str(r.protein_id), mems[int(r.community_id)]) for _, r in wanted.iterrows()
    ]
    wanted["tier"] = wanted.membership_type.map(normalize_label)
    cols = ["protein_id", "community_id", "tier", "topology_score", "semantic_score", "go_leave_one_out"]
    case = wanted[cols].sort_values(["protein_id", "community_id"])
    case.to_csv(OUT / "case_study_go.csv", index=False)
    report["case_study"] = case.round(6).to_dict(orient="records")
    print(case.to_string(index=False), flush=True)
    (OUT / "go_leave_one_out_verification.json").write_text(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
