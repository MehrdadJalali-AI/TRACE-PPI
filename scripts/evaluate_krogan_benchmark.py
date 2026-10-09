#!/usr/bin/env python3
"""Run the reproducible Krogan 2006 second-dataset benchmark."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "results" / "matplotlib_cache"))
sys.path.insert(0, str(ROOT))

from trace_ppi.paths import CONFIGS, RESULTS, TABLES, ensure_dirs
from trace_ppi.graph_io import filter_go, load_krogan
from trace_ppi import baselines
from trace_ppi.cosmos_v2_runner import run_v2
from trace_ppi.evidence_metrics import compute_all_evidence_metrics
from trace_ppi.reuse import load_gold_standard_csv, precision_recall_f1_mmr
from scripts.evaluate_trace_ppi_final import (
    _label_validation_table,
    _safe_clusterone,
    _slpa_grid,
)

DATASET = "krogan"
OUT = RESULTS / DATASET
CONFIG = CONFIGS / "trace_ppi_final.yaml"
GOLD = ROOT / "data" / "gold_standards" / "cyc2008_yeast.csv"


def _append_row(rows, method, metrics, ev, runtime):
    row = {"dataset": DATASET, "method": method, **metrics, **ev}
    row["runtime_sec"] = runtime
    rows.append(row)


def _core_only_from_cores(graph):
    core_path = RESULTS / "cores" / f"black_hole_cores_{DATASET}.csv"
    if not core_path.exists():
        return {}
    cores = pd.read_csv(core_path)
    clusters = {}
    for i, row in cores.iterrows():
        members = {
            str(x)
            for x in str(row.get("initial_members", "")).split(";")
            if x and str(x) in graph
        }
        if len(members) >= 2:
            clusters[int(i)] = members
    return clusters


def _evidence_support_by_label(refined: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    detail = refined.copy()
    if detail.empty:
        out = pd.DataFrame()
        out.to_csv(out_dir / "auditability_summary.csv", index=False)
        return out
    detail["confidence_label"] = detail["membership_type"].astype(str).str.replace(
        "_orbit", "", regex=False
    )
    detail["has_topology"] = detail["topology_score"].fillna(0).astype(float) > 0
    detail["has_semantic"] = detail["semantic_score"].fillna(0).astype(float) > 0
    detail["has_go"] = detail["go_score"].fillna(0).astype(float) > 0
    channel_cols = ["has_topology", "has_semantic", "has_go"]
    detail["any_nonzero"] = detail[channel_cols].any(axis=1)
    detail["multi_channel"] = detail[channel_cols].sum(axis=1) >= 2
    detail.to_csv(out_dir / "assignment_evidence_support.csv", index=False)

    order = ["core", "inner", "outer", "uncertain"]
    summary = (
        detail.groupby("confidence_label")
        .agg(
            assignments=("confidence_label", "size"),
            topology_nonzero_fraction=("has_topology", "mean"),
            semantic_nonzero_fraction=("has_semantic", "mean"),
            go_nonzero_fraction=("has_go", "mean"),
            nonzero_evidence_fraction=("any_nonzero", "mean"),
            multi_channel_fraction=("multi_channel", "mean"),
            topology_mean=("topology_score", "mean"),
            semantic_mean=("semantic_score", "mean"),
            go_mean=("go_score", "mean"),
            membership_mean=("membership_score", "mean"),
        )
        .reset_index()
    )
    summary["dataset"] = DATASET
    summary["sort_key"] = summary["confidence_label"].map({x: i for i, x in enumerate(order)}).fillna(99)
    summary = summary.sort_values("sort_key").drop(columns="sort_key")
    summary.to_csv(out_dir / "auditability_summary.csv", index=False)
    return summary


def _write_cross_dataset_auditability(krogan_audit: pd.DataFrame):
    rows = []
    gavin_path = RESULTS / "evidence_support_by_label.csv"
    if gavin_path.exists():
        g = pd.read_csv(gavin_path)
        g["dataset"] = "gavin"
        rename = {
            "n_assignments": "assignments",
            "any_nonzero_fraction": "nonzero_evidence_fraction",
        }
        g = g.rename(columns=rename)
        rows.append(g)
    if not krogan_audit.empty:
        rows.append(krogan_audit)
    if rows:
        pd.concat(rows, ignore_index=True, sort=False).to_csv(
            RESULTS / "cross_dataset_auditability.csv", index=False
        )


def main():
    ensure_dirs()
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(CONFIG.read_text())
    supp = cfg.get("supplement", {})
    gold = load_gold_standard_csv(str(GOLD))
    graph, go = load_krogan()
    go = filter_go(go)

    rows = []
    run_config = {
        "dataset": DATASET,
        "source": "BioGRID TAB3 records filtered to PUBMED:16554755 (Krogan et al., Nature 2006)",
        "gold_standard": str(GOLD),
        "jaccard_threshold": 0.5,
    }

    for method, label, fn in [
        ("MCL", "MCL", lambda: baselines.mcl_only(graph)),
        ("MCL_overlap_heuristic", "MCL + overlap", lambda: baselines.mcl_overlap_heuristic(graph, go)),
    ]:
        t0 = time.time()
        clusters = fn()
        runtime = time.time() - t0
        metrics = precision_recall_f1_mmr(clusters, gold)
        ev = compute_all_evidence_metrics(clusters, go, runtime_sec=runtime)
        _append_row(rows, method, metrics, ev, runtime)

    clusterone_clusters, clusterone_metrics, clusterone_ev, clusterone_runtime, clusterone_status = _safe_clusterone(
        graph, go, gold, dataset=DATASET
    )
    run_config["clusterone_status"] = clusterone_status
    if clusterone_clusters is not None:
        _append_row(rows, "ClusterONE_exact", clusterone_metrics, clusterone_ev, clusterone_runtime)

    slpa_grid, _ = _slpa_grid(graph, go, gold, dataset=DATASET)
    slpa_grid.to_csv(OUT / "slpa_sensitivity.csv", index=False)
    if not slpa_grid.empty:
        best = slpa_grid.sort_values("f1", ascending=False).iloc[0]
        slpa_metrics = {
            "precision": float(best["precision"]),
            "recall": float(best["recall"]),
            "f1": float(best["f1"]),
            "mmr": float(best.get("mmr", 0.0)),
            "coverage": float(best.get("coverage", 0.0)),
        }
        slpa_ev = {
            "mean_size": float(best.get("mean_size", 0.0)),
            "median_size": float(best.get("median_size", 0.0)),
            "num_modules": float(best.get("num_modules", 0.0)),
            "n_proteins_assigned": float(best.get("n_proteins_assigned", 0.0)),
            "n_overlapping_proteins": float(best.get("n_overlapping_proteins", 0.0)),
            "overlap_protein_fraction": float(best.get("overlap_protein_fraction", 0.0)),
            "go_coherence": float(best.get("go_coherence", 0.0)),
            "functional_specificity": float(best.get("functional_specificity", 0.0)),
            "modules_with_nongeneric_go_fraction": float(best.get("modules_with_nongeneric_go_fraction", 0.0)),
            "evidence_bundle_completeness": 0.0,
        }
        _append_row(rows, "SLPA_best_sensitivity", slpa_metrics, slpa_ev, float(best.get("runtime_sec", 0.0)))

    final_params = {
        "base_method": "mcl_overlap",
        "skip_expansion": True,
        "preserve_all": True,
        "recall_safe_supplement": True,
        "apply_orbit_labels": True,
        **supp,
    }
    t0 = time.time()
    out = run_v2(DATASET, variant="recall_safe_orbits", params=final_params, write_outputs=True)
    runtime = time.time() - t0
    metrics = out["metrics"]
    ev = compute_all_evidence_metrics(out["clusters"], go, out["refined"], runtime_sec=runtime)
    _append_row(rows, "TRACE-PPI_final", metrics, ev, runtime)
    out["refined"].to_csv(OUT / "trace_ppi_refined.csv", index=False)
    _label_validation_table(out["refined"], out["clusters"], gold).to_csv(
        OUT / "confidence_label_validation.csv", index=False
    )
    krogan_audit = _evidence_support_by_label(out["refined"], OUT)

    for method, variant, params in [
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
        o = run_v2(DATASET, variant=variant, params=params, write_outputs=False)
        runtime = time.time() - t0
        ev = compute_all_evidence_metrics(o["clusters"], go, o["refined"], runtime_sec=runtime)
        _append_row(rows, method, o["metrics"], ev, runtime)

    core_clusters = _core_only_from_cores(graph)
    core_metrics = precision_recall_f1_mmr(core_clusters, gold) if core_clusters else {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "mmr": 0.0,
        "coverage": 0.0,
    }
    core_ev = compute_all_evidence_metrics(core_clusters, go, runtime_sec=0.0)
    _append_row(rows, "core_only_ablation", core_metrics, core_ev, 0.0)

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "benchmark_summary.csv", index=False)
    summary.to_csv(TABLES / "table_krogan_benchmark_summary.csv", index=False)
    _write_cross_dataset_auditability(krogan_audit)
    (OUT / "run_config.json").write_text(json.dumps(run_config, indent=2))
    print(summary[["method", "f1", "precision", "recall", "mean_size", "overlap_protein_fraction", "runtime_sec", "evidence_bundle_completeness"]].to_string(index=False))


if __name__ == "__main__":
    main()
