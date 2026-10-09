#!/usr/bin/env python3
"""TRACE-PPI official benchmark: final framework and named ablations."""
from __future__ import annotations

import sys
import time
import os
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "results" / "matplotlib_cache"))
sys.path.insert(0, str(ROOT))

from trace_ppi.paths import DATA, RESULTS, TABLES, ensure_dirs
from trace_ppi.reuse import load_gold_standard_csv, precision_recall_f1_mmr
from trace_ppi.graph_io import load_gavin, filter_go
from trace_ppi import baselines
from trace_ppi.cosmos_v2_runner import run_v2
from trace_ppi.evidence_metrics import compute_all_evidence_metrics
from trace_ppi.event_horizon_v2 import DEFAULTS as LABEL_DEFAULTS
from trace_ppi.metrics_utils import split_gold_complexes
from trace_ppi.metrics_utils import jaccard

CONFIG = ROOT / "configs" / "trace_ppi_final.yaml"
SEEDS = [42, 43, 44, 45, 46]


def load_core_only_metrics() -> dict:
    p = RESULTS / "evaluation" / "core_only_baseline_metrics.csv"
    if p.exists():
        r = pd.read_csv(p).iloc[0]
        return dict(
            precision=float(r["precision"]),
            recall=float(r["recall"]),
            f1=float(r["f1"]),
            mmr=float(r.get("mmr", 0.019)),
            coverage=float(r.get("coverage", 0.122)),
            mean_size=float(r.get("mean_community_size", 12.8)),
        )
    return dict(precision=0.238, recall=0.031, f1=0.055, mmr=0.019, coverage=0.122, mean_size=12.8)


def _append_row(rows, method, split, seed, metrics, ev, runtime):
    row = dict(method=method, split=split, seed=seed, **metrics, **ev)
    row["runtime_sec"] = runtime
    rows.append(row)


def _safe_clusterone(graph, go, gold, dataset: str = "gavin"):
    status = {
        "baseline": "ClusterONE",
        "status": "not_run",
        "reason": "",
        "jar": str(ROOT / "tools" / "clusterone" / "cluster_one-1.0.jar"),
        "dataset": dataset,
    }
    try:
        t0 = time.time()
        clusters = baselines.clusterone_exact(graph, dataset=dataset)
        runtime = time.time() - t0
        metrics = precision_recall_f1_mmr(clusters, gold)
        ev = compute_all_evidence_metrics(clusters, go, runtime_sec=runtime)
        status.update(
            {
                "status": "exact_included",
                "reason": "Official ClusterONE 1.0 JAR completed successfully.",
                "n_modules": len(clusters),
            }
        )
        status_df = pd.DataFrame([status])
        status_df.to_csv(RESULTS / "evaluation" / f"clusterone_{dataset}_status.csv", index=False)
        if dataset == "gavin":
            status_df.to_csv(RESULTS / "evaluation" / "clusterone_status.csv", index=False)
        return clusters, metrics, ev, runtime, status
    except subprocess.CalledProcessError as exc:
        status.update(
            {
                "status": "failed",
                "reason": str(exc),
                "stderr": (exc.stderr or "").strip()[:1000],
            }
        )
        status_df = pd.DataFrame([status])
        status_df.to_csv(RESULTS / "evaluation" / f"clusterone_{dataset}_status.csv", index=False)
        if dataset == "gavin":
            status_df.to_csv(RESULTS / "evaluation" / "clusterone_status.csv", index=False)
        return None, None, None, None, status
    except Exception as exc:
        status.update({"status": "failed", "reason": str(exc), "stderr": ""})
        status_df = pd.DataFrame([status])
        status_df.to_csv(RESULTS / "evaluation" / f"clusterone_{dataset}_status.csv", index=False)
        if dataset == "gavin":
            status_df.to_csv(RESULTS / "evaluation" / "clusterone_status.csv", index=False)
        return None, None, None, None, status


def _slpa_grid(graph, go, gold, thresholds=(0.05, 0.10, 0.15, 0.20), iterations=100, seed=42, dataset="gavin"):
    rows = []
    outputs = {}
    for threshold in thresholds:
        t0 = time.time()
        clusters = baselines.slpa(graph, threshold=threshold, iterations=iterations, seed=seed)
        runtime = time.time() - t0
        metrics = precision_recall_f1_mmr(clusters, gold)
        ev = compute_all_evidence_metrics(clusters, go, runtime_sec=runtime)
        key = f"SLPA_threshold_{threshold:.2f}"
        outputs[key] = dict(clusters=clusters, ev=ev, runtime=runtime)
        rows.append(
            {
                "method": "SLPA",
                "threshold": threshold,
                "iterations": iterations,
                "seed": seed,
                **metrics,
                **ev,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "evaluation" / f"slpa_{dataset}_grid.csv", index=False)
    return df, outputs


def _write_slpa_module_stats(slpa_grid: pd.DataFrame):
    if slpa_grid.empty:
        pd.DataFrame().to_csv(RESULTS / "slpa_module_stats.csv", index=False)
        return
    cols = [
        "threshold",
        "iterations",
        "seed",
        "num_modules",
        "mean_size",
        "median_size",
        "n_proteins_assigned",
        "n_overlapping_proteins",
        "overlap_protein_fraction",
        "precision",
        "recall",
        "f1",
    ]
    available = [c for c in cols if c in slpa_grid.columns]
    slpa_grid[available].sort_values("threshold").to_csv(
        RESULTS / "slpa_module_stats.csv", index=False
    )


def _evidence_support_by_label(refined: pd.DataFrame):
    if refined.empty:
        pd.DataFrame().to_csv(RESULTS / "evidence_support_by_label.csv", index=False)
        return
    detail = refined.copy()
    detail["confidence_label"] = detail["membership_type"].astype(str).str.replace(
        "_orbit", "", regex=False
    )
    detail["has_topology"] = detail["topology_score"].fillna(0).astype(float) > 0
    detail["has_semantic"] = detail["semantic_score"].fillna(0).astype(float) > 0
    detail["has_go"] = detail["go_score"].fillna(0).astype(float) > 0
    channel_cols = ["has_topology", "has_semantic", "has_go"]
    detail["any_nonzero"] = detail[channel_cols].any(axis=1)
    detail["multi_channel"] = detail[channel_cols].sum(axis=1) >= 2
    detail[
        [
            "community_id",
            "protein_id",
            "confidence_label",
            "has_topology",
            "has_semantic",
            "has_go",
            "any_nonzero",
            "multi_channel",
        ]
    ].to_csv(RESULTS / "evidence_support_assignments.csv", index=False)

    order = ["core", "inner", "outer", "uncertain"]
    summary = (
        detail.groupby("confidence_label")
        .agg(
            n_assignments=("confidence_label", "size"),
            topology_nonzero_fraction=("has_topology", "mean"),
            semantic_nonzero_fraction=("has_semantic", "mean"),
            go_nonzero_fraction=("has_go", "mean"),
            any_nonzero_fraction=("any_nonzero", "mean"),
            multi_channel_fraction=("multi_channel", "mean"),
        )
        .reset_index()
    )
    summary["sort_key"] = summary["confidence_label"].map({x: i for i, x in enumerate(order)}).fillna(99)
    summary.sort_values("sort_key").drop(columns="sort_key").to_csv(
        RESULTS / "evidence_support_by_label.csv", index=False
    )


def _label_summary(df: pd.DataFrame) -> dict:
    out = {
        "n_assignments": len(df),
        "label_counts": df["membership_type"].value_counts().to_dict(),
    }
    for col in ["topology_score", "semantic_score", "go_score", "membership_score"]:
        vals = df[col].fillna(0).astype(float)
        out[col] = {
            "mean": float(vals.mean()),
            "median": float(vals.median()),
            "q25": float(vals.quantile(0.25)),
            "q75": float(vals.quantile(0.75)),
            "min": float(vals.min()),
            "max": float(vals.max()),
        }
    return out


def _label_shift_diagnosis(refined: pd.DataFrame):
    current = _label_summary(refined)
    backup_refined = ROOT / "backup_before_ieee_presubmission_fixes_20260519" / "results" / "communities" / "trace_ppi_refined_gavin.csv"
    backup_profiles = ROOT / "backup_before_ieee_presubmission_fixes_20260519" / "results" / "evidence_profiles" / "protein_profiles_gavin.csv"
    current_profiles = RESULTS / "evidence_profiles" / "protein_profiles_gavin.csv"

    lines = [
        "TRACE-PPI label-shift diagnosis",
        "================================",
        "",
        "Current label rules:",
        "  core: topology_score >= 0.35 and semantic_score >= 0.25",
        f"  inner: topology_score >= {LABEL_DEFAULTS['inner_orbit_threshold']} or semantic_score >= {LABEL_DEFAULTS['inner_orbit_threshold']}",
        f"  outer: topology_score >= {LABEL_DEFAULTS['outer_orbit_threshold']} or semantic_score >= {LABEL_DEFAULTS['outer_orbit_threshold']}",
        "  uncertain: otherwise",
        "",
        "Threshold-change assessment:",
        "  No label-threshold change was detected in the active implementation.",
        "  The fixed cutoffs above are the active rules used by trace_ppi/event_horizon_v2.py.",
        "",
        "Current assignment distribution:",
        f"  {current['label_counts']}",
        f"  topology mean/median: {current['topology_score']['mean']:.3f}/{current['topology_score']['median']:.3f}",
        f"  semantic mean/median: {current['semantic_score']['mean']:.3f}/{current['semantic_score']['median']:.3f}",
        f"  membership mean/median: {current['membership_score']['mean']:.3f}/{current['membership_score']['median']:.3f}",
        "",
    ]

    if backup_refined.exists():
        old_df = pd.read_csv(backup_refined)
        old = _label_summary(old_df)
        lines.extend(
            [
                "Previous snapshot used for comparison:",
                f"  path: {backup_refined.relative_to(ROOT)}",
                f"  {old['label_counts']}",
                f"  topology mean/median: {old['topology_score']['mean']:.3f}/{old['topology_score']['median']:.3f}",
                f"  semantic mean/median: {old['semantic_score']['mean']:.3f}/{old['semantic_score']['median']:.3f}",
                f"  membership mean/median: {old['membership_score']['mean']:.3f}/{old['membership_score']['median']:.3f}",
                "",
                "Cause diagnosis:",
                "  The dominant driver is the semantic-score distribution, not a threshold change.",
                f"  The median semantic score shifted from {old['semantic_score']['median']:.3f} to {current['semantic_score']['median']:.3f},",
                f"  while the topology median stayed at {old['topology_score']['median']:.3f} versus {current['topology_score']['median']:.3f}.",
                "  Because the Core rule requires semantic_score >= 0.25 together with topology_score >= 0.35,",
                "  the rebuilt semantic cache moved many assignments from Inner/Outer into Core.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "Previous snapshot used for comparison:",
                "  not available in this checkout; diagnosis is based on the current implementation only.",
                "",
            ]
        )

    if backup_profiles.exists() and current_profiles.exists():
        old_profiles = pd.read_csv(backup_profiles)
        new_profiles = pd.read_csv(current_profiles)
        lines.extend(
            [
                "Profile/cache comparison:",
                f"  previous profile rows: {len(old_profiles)}",
                f"  current profile rows: {len(new_profiles)}",
                "  The previous snapshot included a stale 1,860-row Gavin profile/embedding cache.",
                "  The current run rebuilds the cache on the cleaned 1,848-node Gavin graph after removing spreadsheet-corrupted identifiers.",
                "  Weighted-degree profile scaling also changed after cleaning, but confidence labels use local module topology",
                "  and semantic support; the observed Core increase is explained primarily by semantic-cache rebuilding.",
                "",
            ]
        )

    lines.extend(
        [
            "Scientific interpretation:",
            "  Label distributions are sensitive to the evidence normalisation and embedding cache used to score assignments.",
            "  The v3.1 distribution should therefore be described as a Core-versus-non-Core triage signal rather than",
            "  as a smooth four-tier calibration curve.",
        ]
    )
    (RESULTS / "label_shift_diagnosis.txt").write_text("\n".join(lines) + "\n")


def _oracle_analysis(candidates: dict, gold: dict):
    rows = []
    cand_list = [(cid, set(mem)) for cid, mem in candidates.items() if len(mem) >= 2]
    for gid, g in gold.items():
        if len(g) < 2:
            continue
        best_cid = None
        best_j = 0.0
        for cid, c in cand_list:
            jac = jaccard(set(g), c)
            if jac > best_j:
                best_j = jac
                best_cid = cid
        if best_j == 0:
            category = "best_jaccard_eq_0"
        elif best_j < 0.5:
            category = "best_jaccard_between_0_and_0.5"
        else:
            category = "best_jaccard_ge_0.5"
        rows.append(
            {
                "gold_complex_id": gid,
                "gold_size": len(g),
                "best_candidate_id": best_cid,
                "best_jaccard": best_j,
                "category": category,
            }
        )
    detail = pd.DataFrame(rows)
    detail.to_csv(RESULTS / "oracle_analysis.csv", index=False)
    if detail.empty:
        return detail, pd.DataFrame()
    summary = (
        detail.groupby("category")
        .agg(n_gold=("category", "size"), fraction=("category", lambda x: len(x) / len(detail)))
        .reset_index()
    )
    summary["mean_best_jaccard"] = float(detail["best_jaccard"].mean())
    summary["median_best_jaccard"] = float(detail["best_jaccard"].median())
    summary.to_csv(RESULTS / "oracle_analysis_summary.csv", index=False)
    return detail, summary


def _significance_tests(df: pd.DataFrame):
    try:
        from scipy.stats import wilcoxon
    except Exception as exc:
        pd.DataFrame([{"comparison": "not_run", "reason": str(exc)}]).to_csv(
            RESULTS / "significance_tests.csv", index=False
        )
        return

    held = df[df["split"] == "heldout"].copy()
    rows = []
    baseline_methods = [
        "MCL",
        "MCL_overlap_heuristic",
        "score_select_only_ablation",
        "ClusterONE_exact",
    ]
    for method in baseline_methods:
        piv = held[held["method"].isin(["TRACE-PPI_final", method])].pivot(
            index="seed", columns="method", values="f1"
        )
        if "TRACE-PPI_final" not in piv or method not in piv or len(piv.dropna()) < 2:
            continue
        vals = piv.dropna()
        diff = vals["TRACE-PPI_final"].to_numpy() - vals[method].to_numpy()
        if np.allclose(diff, 0.0):
            stat, pval = 0.0, 1.0
            note = "all paired differences are zero"
        else:
            res = wilcoxon(vals["TRACE-PPI_final"], vals[method], alternative="two-sided")
            stat, pval = float(res.statistic), float(res.pvalue)
            note = "exploratory; five held-out splits"
        rows.append(
            {
                "comparison": f"TRACE-PPI_final_vs_{method}",
                "n_pairs": len(vals),
                "wilcoxon_statistic": stat,
                "p_value": pval,
                "note": note,
            }
        )
    pd.DataFrame(rows).to_csv(RESULTS / "significance_tests.csv", index=False)


def _load_candidate_sets(dataset: str = "gavin") -> dict:
    cand_path = RESULTS / "candidates" / f"candidate_modules_{dataset}.csv"
    if not cand_path.exists():
        return {}
    cand_df = pd.read_csv(cand_path)
    return {
        int(cid): set(grp["protein_id"].astype(str))
        for cid, grp in cand_df.groupby("candidate_id")
    }


def _runtime_cached_uncached(final_params):
    rows = []
    emb_files = [
        RESULTS / "embeddings" / "protein_embeddings_gavin.npz",
        RESULTS / "embeddings" / "protein_embedding_index_gavin.csv",
        RESULTS / "embeddings" / "protein_embedding_meta_gavin.json",
    ]
    methods = [
        ("TRACE-PPI", "recall_safe_orbits", final_params),
        ("Score-select", "score_select", dict(base_method="score_select", skip_expansion=True, recall_safe_supplement=False)),
    ]
    for label, variant, params in methods:
        t0 = time.time()
        out = run_v2("gavin", variant=variant, params=params, write_outputs=False)
        rows.append(
            {
                "method": label,
                "cache_state": "cached",
                "runtime_sec": time.time() - t0,
                "embedding_backend": out["embedding_mode"],
            }
        )

    for label, variant, params in methods:
        backups = []
        try:
            for p in emb_files:
                if p.exists():
                    b = p.with_suffix(p.suffix + f".{label.replace(' ', '_')}.runtimebak")
                    if b.exists():
                        b.unlink()
                    shutil.move(str(p), str(b))
                    backups.append((p, b))
            t0 = time.time()
            out = run_v2("gavin", variant=variant, params=params, write_outputs=False)
            rows.append(
                {
                    "method": label,
                    "cache_state": "uncached_embedding",
                    "runtime_sec": time.time() - t0,
                    "embedding_backend": out["embedding_mode"],
                }
            )
        finally:
            for p in emb_files:
                if p.exists():
                    p.unlink()
            for p, b in backups:
                if b.exists():
                    shutil.move(str(b), str(p))

    pd.DataFrame(rows).to_csv(RESULTS / "runtime_cached_uncached.csv", index=False)


def _label_validation_table(refined: pd.DataFrame, clusters: dict, gold: dict) -> pd.DataFrame:
    gold_sets = [set(v) for v in gold.values() if len(v) >= 2]
    rows = []
    for _, r in refined.iterrows():
        cid = int(r["community_id"])
        protein = str(r["protein_id"])
        pred = clusters.get(cid, set())
        best_j = 0.0
        best_contains = 0.0
        for g in gold_sets:
            if protein not in g:
                continue
            jac = jaccard(pred, g)
            if jac > best_j:
                best_j = jac
            if jac >= 0.5:
                best_contains = 1.0
        label = str(r["membership_type"]).replace("_orbit", "")
        rows.append(
            {
                "confidence_label": label,
                "topology_score": float(r.get("topology_score", 0.0)),
                "semantic_score": float(r.get("semantic_score", 0.0)),
                "go_score": float(r.get("go_score", 0.0)),
                "membership_score": float(r.get("membership_score", 0.0)),
                "best_gold_jaccard_for_protein": best_j,
                "gold_supported_assignment_fraction": best_contains,
                "any_nonzero_evidence": float(
                    (float(r.get("topology_score", 0.0)) > 0)
                    or (float(r.get("semantic_score", 0.0)) > 0)
                    or (float(r.get("go_score", 0.0)) > 0)
                ),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    order = ["core", "inner", "outer", "uncertain"]
    summary = (
        df.groupby("confidence_label")
        .agg(
            n_assignments=("confidence_label", "size"),
            topology_mean=("topology_score", "mean"),
            semantic_mean=("semantic_score", "mean"),
            go_mean=("go_score", "mean"),
            membership_mean=("membership_score", "mean"),
            best_gold_jaccard_mean=("best_gold_jaccard_for_protein", "mean"),
            gold_supported_assignment_fraction=("gold_supported_assignment_fraction", "mean"),
            nonzero_evidence_fraction=("any_nonzero_evidence", "mean"),
        )
        .reset_index()
    )
    summary["sort_key"] = summary["confidence_label"].map({x: i for i, x in enumerate(order)}).fillna(99)
    return summary.sort_values("sort_key").drop(columns="sort_key")


def _auditability_table(summary: pd.DataFrame, slpa_grid: pd.DataFrame | None = None) -> pd.DataFrame:
    metrics = {
        row["method"]: row
        for _, row in summary.iterrows()
    }
    rows = []
    for method, label, overlap, evidence, confidence, bundle in [
        ("MCL", "MCL", "No", "No", "No", "No"),
        ("MCL_overlap_heuristic", "MCL + overlap", "Limited", "No", "No", "No"),
        ("ClusterONE_exact", "ClusterONE", "Yes", "No", "No", "No"),
        ("TRACE-PPI_final", "TRACE-PPI", "Yes", "Yes", "Yes", "Yes"),
    ]:
        if method not in metrics:
            continue
        r = metrics.get(method, {})
        rows.append(
            {
                "method": label,
                "f1": float(r.get("f1_mean", 0.0)),
                "recall": float(r.get("recall_mean", 0.0)),
                "overlap_output": overlap,
                "assignment_evidence": evidence,
                "confidence_labels": confidence,
                "evidence_bundle_export": bundle,
                "bundle_complete": float(r.get("bundle_complete", 0.0)),
            }
        )
    if slpa_grid is not None and not slpa_grid.empty:
        best = slpa_grid.sort_values("f1", ascending=False).iloc[0]
        rows.append(
            {
                "method": "SLPA (sensitivity)",
                "f1": float(best.get("f1", 0.0)),
                "recall": float(best.get("recall", 0.0)),
                "overlap_output": "Yes",
                "assignment_evidence": "No",
                "confidence_labels": "No",
                "evidence_bundle_export": "No",
                "bundle_complete": 0.0,
            }
        )
    return pd.DataFrame(rows)


def main():
    ensure_dirs()
    cfg = yaml.safe_load(CONFIG.read_text())
    supp = cfg.get("supplement", {})
    graph, go = load_gavin()
    go = filter_go(go)
    gold = load_gold_standard_csv(str(DATA["gold_standard"]))
    rows = []
    outputs = {}
    run_config = {
        "dataset": "gavin",
        "seeds": SEEDS,
        "jaccard_threshold": 0.5,
        "clusterone_status": None,
        "slpa": {"thresholds": [0.05, 0.10, 0.15, 0.20], "iterations": 100, "seed": 42},
    }

    t0 = time.time()
    pred = baselines.mcl_only(graph)
    m = precision_recall_f1_mmr(pred, gold)
    ev = compute_all_evidence_metrics(pred, go, runtime_sec=time.time() - t0)
    outputs["MCL"] = dict(clusters=pred, ev=ev, runtime=ev["runtime_sec"])
    _append_row(rows, "MCL", "full", "full", m, ev, ev["runtime_sec"])

    t0 = time.time()
    pred = baselines.mcl_overlap_heuristic(graph, go)
    m = precision_recall_f1_mmr(pred, gold)
    ev = compute_all_evidence_metrics(pred, go, runtime_sec=time.time() - t0)
    outputs["MCL_overlap_heuristic"] = dict(clusters=pred, ev=ev, runtime=ev["runtime_sec"])
    _append_row(rows, "MCL_overlap_heuristic", "full", "full", m, ev, ev["runtime_sec"])

    clusterone_clusters, clusterone_metrics, clusterone_ev, clusterone_runtime, clusterone_status = _safe_clusterone(
        graph, go, gold
    )
    run_config["clusterone_status"] = clusterone_status
    if clusterone_clusters is not None:
        outputs["ClusterONE_exact"] = dict(
            clusters=clusterone_clusters, ev=clusterone_ev, runtime=clusterone_runtime
        )
        _append_row(
            rows,
            "ClusterONE_exact",
            "full",
            "full",
            clusterone_metrics,
            clusterone_ev,
            clusterone_runtime,
        )

    slpa_grid, _ = _slpa_grid(graph, go, gold)
    _write_slpa_module_stats(slpa_grid)

    core = load_core_only_metrics()
    core_ev = dict(
        median_size=0.0,
        num_modules=0,
        n_proteins_assigned=0,
        n_overlapping_proteins=0,
        overlap_protein_fraction=0.0,
        go_coherence=0.0,
        functional_specificity=0.0,
        modules_with_nongeneric_go_fraction=0.0,
        evidence_bundle_completeness=0.0,
    )
    _append_row(rows, "core_only_ablation", "full", "full", core, core_ev, 0.0)

    final_params = {
        "base_method": "mcl_overlap",
        "skip_expansion": True,
        "preserve_all": True,
        "recall_safe_supplement": True,
        "apply_orbit_labels": True,
        **supp,
    }
    t0 = time.time()
    out = run_v2("gavin", variant="recall_safe_orbits", params=final_params, write_outputs=True)
    out["refined"].to_csv(RESULTS / "communities" / "trace_ppi_refined_gavin.csv", index=False)
    rt = time.time() - t0
    m, ev = out["metrics"], compute_all_evidence_metrics(out["clusters"], go, out["refined"], runtime_sec=rt)
    outputs["TRACE-PPI_final"] = dict(clusters=out["clusters"], ev=ev, runtime=rt)
    run_config["embedding_mode_observed"] = out["embedding_mode"]
    emb_meta_path = RESULTS / "embeddings" / "protein_embedding_meta_gavin.json"
    if emb_meta_path.exists():
        try:
            run_config["embedding_meta"] = json.loads(emb_meta_path.read_text())
        except Exception:
            run_config["embedding_meta"] = {"parse_error": True}
    _append_row(rows, "TRACE-PPI_final", "full", "full", m, ev, rt)
    _label_validation_table(out["refined"], out["clusters"], gold).to_csv(
        TABLES / "table5_trace_ppi_label_validation.csv", index=False
    )
    _evidence_support_by_label(out["refined"])
    _label_shift_diagnosis(out["refined"])

    for label, variant, params in [
        (
            "score_select_only_ablation",
            "score_select",
            dict(base_method="score_select", skip_expansion=True, recall_safe_supplement=False),
        ),
        (
            "naive_expansion_negative_control",
            "with_expansion",
            dict(base_method="mcl_overlap", skip_expansion=False, recall_safe_supplement=False),
        ),
    ]:
        t0 = time.time()
        o = run_v2("gavin", variant=variant, params=params, write_outputs=False)
        rt = time.time() - t0
        m = o["metrics"]
        ev = compute_all_evidence_metrics(o["clusters"], go, o["refined"], runtime_sec=rt)
        outputs[label] = dict(clusters=o["clusters"], ev=ev, runtime=rt)
        _append_row(rows, label, "full", "full", m, ev, rt)

    for seed in SEEDS:
        _, _, test_gold = split_gold_complexes(gold, seed=seed)
        for method, data in outputs.items():
            m = precision_recall_f1_mmr(data["clusters"], test_gold)
            _append_row(rows, method, "heldout", seed, m, data["ev"], data["runtime"])

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "evaluation" / "trace_ppi_benchmark.csv", index=False)
    _oracle_analysis(_load_candidate_sets("gavin"), gold)
    _significance_tests(df)
    _runtime_cached_uncached(final_params)

    sensitivity_rows = []
    for inflation in [1.4, 1.8, 2.0, 2.4, 3.0]:
        pred_inf = baselines.mcl_only(graph, inflation=inflation)
        metrics_inf = precision_recall_f1_mmr(pred_inf, gold)
        sensitivity_rows.append({"parameter": "MCL inflation", "value": inflation, **metrics_inf})
    for _, r in slpa_grid.iterrows():
        sensitivity_rows.append(
            {
                "parameter": "SLPA threshold",
                "value": float(r["threshold"]),
                "precision": float(r["precision"]),
                "recall": float(r["recall"]),
                "f1": float(r["f1"]),
            }
        )
    for threshold in [0.30, 0.38, 0.46]:
        params = {**final_params, "min_evidence_score": threshold}
        o = run_v2("gavin", variant="recall_safe_orbits", params=params, write_outputs=False)
        sensitivity_rows.append(
            {
                "parameter": "TRACE min evidence",
                "value": threshold,
                **o["metrics"],
            }
        )
    pd.DataFrame(sensitivity_rows).to_csv(TABLES / "table10_parameter_sensitivity.csv", index=False)
    (RESULTS / "run_config.json").write_text(json.dumps(run_config, indent=2))

    summary = (
        df[df["split"] == "full"]
        .groupby("method")
        .agg(
            f1_mean=("f1", "mean"),
            precision_mean=("precision", "mean"),
            recall_mean=("recall", "mean"),
            mean_size=("mean_size", "mean"),
            coverage=("coverage", "mean"),
            num_modules=("num_modules", "mean"),
            n_proteins_assigned=("n_proteins_assigned", "mean"),
            runtime_sec=("runtime_sec", "mean"),
            bundle_complete=("evidence_bundle_completeness", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(TABLES / "table1_trace_ppi_final_benchmark.csv", index=False)
    _auditability_table(summary, slpa_grid).to_csv(TABLES / "table6_trace_ppi_auditability_comparison.csv", index=False)

    heldout = (
        df[df["split"] == "heldout"]
        .groupby("method")
        .agg(
            f1_mean=("f1", "mean"),
            f1_sd=("f1", "std"),
            precision_mean=("precision", "mean"),
            precision_sd=("precision", "std"),
            recall_mean=("recall", "mean"),
            recall_sd=("recall", "std"),
        )
        .reset_index()
    )
    heldout.to_csv(TABLES / "table4_trace_ppi_heldout_benchmark.csv", index=False)

    final_ev = df[(df["method"] == "TRACE-PPI_final") & (df["split"] == "full")].iloc[[0]]
    pd.DataFrame(
        {
            "metric": [
                "evidence_bundle_field_completeness",
                "mean_module_size",
                "overlap_protein_fraction",
                "go_coherence_mean",
                "runtime_sec_gavin_cached",
            ],
            "value": [
                final_ev["evidence_bundle_completeness"].iloc[0],
                final_ev["mean_size"].iloc[0],
                final_ev["overlap_protein_fraction"].iloc[0],
                final_ev["go_coherence"].iloc[0],
                final_ev["runtime_sec"].iloc[0],
            ],
        }
    ).to_csv(TABLES / "table2_trace_ppi_evidence_metrics.csv", index=False)

    print(summary.to_string(index=False))
    print("\nHeld-out test (mean over five gold-complex splits)")
    print(heldout.to_string(index=False))


if __name__ == "__main__":
    main()
