"""End-to-end TRACE-PPI deconvolution experiment and reproducible outputs."""
from __future__ import annotations

import json
import platform
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Mapping, Set

import networkx as nx
import numpy as np
import pandas as pd

from . import baselines
from .cosmos_v2_runner import _load_embeddings
from .deconvolution import assign_overlapping_memberships, deconvolve_cluster, fit_heterogeneity
from .graph_io import filter_go, load_gavin, load_krogan
from .metrics_utils import clusters_from_membership
from .overlap_evaluation import (
    candidate_coverage, evaluate_membership_pairs, evaluate_shared_proteins,
    merged_cluster_cases, omega_index, overlapping_nmi,
)
from .paths import RESULTS, ensure_dirs
from .recall_safe_supplementation import supplement_modules_recall_safe
from .reuse import load_gold_standard_csv, precision_recall_f1_mmr


def _load(dataset: str):
    graph, go = load_gavin() if dataset == "gavin" else load_krogan()
    go = filter_go(go)
    profile_path = RESULTS / "evidence_profiles" / f"protein_profiles_{dataset}.csv"
    if not profile_path.exists():
        from .evidence_profiles import build_profiles
        profiles = build_profiles(graph, go, dataset)
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        profiles.to_csv(profile_path, index=False)
    profiles = pd.read_csv(profile_path)
    _, embedding_mode, _, embeddings = _load_embeddings(dataset, profiles)
    return graph, go, embeddings, embedding_mode


def _membership_clusters(df: pd.DataFrame) -> Dict[int, Set[str]]:
    return clusters_from_membership(df) if not df.empty else {}


def construct_deconvolved_candidates(
    coarse: Mapping[int, Set[str]], graph: nx.Graph, evidence: Mapping[str, object],
    heterogeneity_quantile: float = .75, weights: dict | None = None,
    disable_heterogeneity: bool = False,
) -> tuple[Dict[int, Set[str]], Dict[int, dict], pd.DataFrame]:
    """Decompose only heterogeneous MCL clusters and retain the rest intact."""
    heterogeneity = fit_heterogeneity(coarse, graph, evidence, quantile=heterogeneity_quantile)
    by_id = heterogeneity.set_index("cluster_id").to_dict("index") if not heterogeneity.empty else {}
    candidates: Dict[int, Set[str]] = {}
    metadata: Dict[int, dict] = {}
    next_id = 0
    for coarse_id, members in sorted(coarse.items()):
        decision = by_id.get(coarse_id, {})
        heterogeneous = bool(decision.get("heterogeneous", False)) or disable_heterogeneity
        if heterogeneous:
            output = deconvolve_cluster(set(members), graph, evidence, weights=weights)
            local_candidates = output["candidates"]
            local_cores = output["cores"]
            method = output["method"]
        else:
            local_candidates = [set(members)]
            local_cores = [set(members)]
            method = "retained_homogeneous"
        for i, candidate in enumerate(local_candidates):
            if len(candidate) < 2:
                continue
            candidates[next_id] = set(map(str, candidate))
            metadata[next_id] = {
                "coarse_cluster_id": int(coarse_id), "method": method,
                "core": set(local_cores[min(i, len(local_cores) - 1)]),
                "heterogeneous": heterogeneous,
                "heterogeneity_score": float(decision.get("heterogeneity_score", 0.0)),
            }
            next_id += 1
    return candidates, metadata, heterogeneity


def _decorate_provenance(df: pd.DataFrame, dataset: str, params: dict) -> pd.DataFrame:
    if df.empty:
        return df
    record = json.dumps({"dataset": dataset, "random_seed": 42, **params}, sort_keys=True)
    out = df.copy()
    out["provenance"] = record
    return out


def _method_row(method: str, clusters: Mapping[int, Set[str]], gold: Mapping[int, Set[str]]) -> dict:
    return {"method": method, **precision_recall_f1_mmr(clusters, gold),
            "omega_index": omega_index(clusters, gold), "overlapping_nmi": overlapping_nmi(clusters, gold),
            "n_modules": len(clusters), "n_assignments": sum(map(len, clusters.values()))}


def _fast_method_row(method: str, clusters: Mapping[int, Set[str]], gold: Mapping[int, Set[str]]) -> dict:
    """Module metrics for repeated sensitivity rows (cover NMI is reported once)."""
    return {"method": method, **precision_recall_f1_mmr(clusters, gold),
            "n_modules": len(clusters), "n_assignments": sum(map(len, clusters.values()))}


def _write_synthetic(results_dir: Path, weights: dict) -> pd.DataFrame:
    """Controlled overlap test independent of incomplete biological reference sets."""
    rows = []
    for n_complexes in (2, 3):
        for overlap in (.10, .30, .50):
            for noise in (.02, .10):
                graph = nx.Graph(); truth = {}
                shared = {f"s{i}" for i in range(max(1, int(12 * overlap)))}
                for k in range(n_complexes):
                    own = {f"c{k}_{i}" for i in range(12)}
                    members = own | shared; truth[k] = members
                    for a in members:
                        for b in members:
                            if a < b and np.random.default_rng(42 + k).random() < .78:
                                graph.add_edge(a, b)
                rng = np.random.default_rng(123)
                nodes = list(graph.nodes())
                for i, a in enumerate(nodes):
                    for b in nodes[i + 1:]:
                        if not graph.has_edge(a, b) and rng.random() < noise:
                            graph.add_edge(a, b)
                evidence = {"go": {}, "embeddings": {}}
                # The whole synthetic graph is intentionally one coarse cluster.
                out = deconvolve_cluster(set(graph.nodes()), graph, evidence, weights={"topology": 1, "semantic": 0, "go": 0})
                pred = {i: c for i, c in enumerate(out["candidates"])}
                mem = evaluate_membership_pairs(pred, truth)
                shared_m = evaluate_shared_proteins(pred, truth)
                rows.append({"n_true_complexes": n_complexes, "overlap_fraction": overlap, "noise_probability": noise,
                             "n_recovered_complexes": len(pred), "method": out["method"], **mem, **shared_m})
    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "synthetic_overlap_test.csv", index=False)
    return df


def run_deconvolution_experiment(dataset: str = "gavin", output_root: Path | None = None) -> dict:
    """Run the new stages and write the reviewer-requested structured outputs."""
    ensure_dirs()
    root = output_root or RESULTS
    root.mkdir(parents=True, exist_ok=True)
    run_dir = root / "deconvolution"
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    graph, go, embeddings, embedding_mode = _load(dataset)
    gold = load_gold_standard_csv(str(__import__("trace_ppi.paths", fromlist=["DATA"]).DATA["gold_standard"]))
    evidence = {"go": go, "embeddings": embeddings}
    coarse = baselines.mcl_only(graph)
    original = baselines.mcl_overlap_heuristic(graph, go)
    candidates, metadata, heterogeneity = construct_deconvolved_candidates(coarse, graph, evidence)
    membership = assign_overlapping_memberships(candidates, graph, evidence, metadata)
    membership = _decorate_provenance(membership, dataset, {"heterogeneity_quantile": .75, "threshold_mode": "adaptive", "weights": "0.45/0.35/0.20"})
    deconvolved = _membership_clusters(membership)

    # Bounded supplementation remains downstream of deconvolution.  It is not
    # allowed to create a separate module and retains its original safety caps.
    supplemented, supplement_stats = supplement_modules_recall_safe(deconvolved, graph, go, embeddings)
    supplemented_membership = assign_overlapping_memberships(supplemented, graph, evidence, metadata)
    supplemented_membership = _decorate_provenance(supplemented_membership, dataset, {"stage": "bounded_supplementation", "stats": supplement_stats})
    final_clusters = _membership_clusters(supplemented_membership)

    candidate_old, candidate_old_detail = candidate_coverage(coarse, gold)
    candidate_new, candidate_new_detail = candidate_coverage(candidates, gold)
    coverage = pd.DataFrame([{"construction": "original_mcl", **candidate_old}, {"construction": "deconvolution", **candidate_new}])
    coverage.to_csv(root / "candidate_coverage.csv", index=False)
    pd.concat([candidate_old_detail.assign(construction="original_mcl"), candidate_new_detail.assign(construction="deconvolution")]).to_csv(run_dir / "candidate_coverage_detail.csv", index=False)

    module_rows = [_method_row("MCL", coarse, gold), _method_row("Original TRACE-PPI", original, gold), _method_row("TRACE-PPI deconvolution", final_clusters, gold)]
    # Existing overlapping baselines are run on exactly the same graph.  The
    # official ClusterONE JAR is optional, but an absence is made explicit.
    slpa_clusters = baselines.slpa(graph, seed=42)
    module_rows.append(_method_row("SLPA", slpa_clusters, gold))
    try:
        clusterone = baselines.clusterone_exact(graph, dataset=dataset)
        module_rows.append(_method_row("ClusterONE", clusterone, gold))
    except Exception as exc:
        module_rows.append({"method": "ClusterONE", "status": f"unavailable: {type(exc).__name__}"})
    module_metrics = pd.DataFrame(module_rows)
    module_metrics.to_csv(root / "module_metrics.csv", index=False)
    module_metrics.to_csv(root / "baseline_comparison.csv", index=False)

    membership_rows = []
    shared_rows = []
    covers = {"MCL": coarse, "Original TRACE-PPI": original, "TRACE-PPI deconvolution": final_clusters, "SLPA": slpa_clusters}
    for method, cover in covers.items():
        membership_rows.append({"method": method, **evaluate_membership_pairs(cover, gold)})
        shared_rows.append({"method": method, **evaluate_shared_proteins(cover, gold)})
    pd.DataFrame(membership_rows).to_csv(root / "membership_metrics.csv", index=False)
    pd.DataFrame(shared_rows).to_csv(root / "shared_membership_metrics.csv", index=False)

    stress = merged_cluster_cases(coarse, original, final_clusters, gold, metadata)
    stress.to_csv(root / "merged_cluster_cases.csv", index=False)

    # Ablations use the same candidate construction.  The stages are explicit
    # so the incremental effect can be inspected instead of inferred.
    a2 = candidates
    a3 = deconvolved
    a4 = supplemented
    ablations = [
        ("A0 MCL only", coarse), ("A1 MCL + old reassignment", original),
        ("A2 MCL + deconvolution", a2), ("A3 + overlap refinement", a3),
        ("A4 + bounded supplementation", a4), ("A5 full + evidence tiers", final_clusters),
    ]
    ablation_rows = []
    for name, cover in ablations:
        ablation_rows.append({"ablation": name, **_method_row(name, cover, gold), **evaluate_membership_pairs(cover, gold), **evaluate_shared_proteins(cover, gold)})
    pd.DataFrame(ablation_rows).to_csv(root / "ablation_results.csv", index=False)

    sensitivity_rows = []
    # Candidate construction is independent of the membership rule; cache each
    # quantile's candidates before sweeping adaptive versus fixed assignment.
    sensitivity_constructions = {
        q: construct_deconvolved_candidates(coarse, graph, evidence, heterogeneity_quantile=q)
        for q in (.65, .75, .85)
    }
    for quantile, (cands, meta, _) in sensitivity_constructions.items():
        for mode in ("adaptive", "fixed"):
            mem = assign_overlapping_memberships(cands, graph, evidence, meta, threshold_mode=mode)
            cover = _membership_clusters(mem)
            counts = mem.groupby("protein_id").community_id.nunique() if not mem.empty else pd.Series(dtype=float)
            sensitivity_rows.append({"factor": "heterogeneity_quantile/threshold_mode", "heterogeneity_quantile": quantile,
                                     "threshold_mode": mode, "mean_multi_membership": float(counts.mean()) if len(counts) else 0.0,
                                     "false_secondary_membership_rate": evaluate_shared_proteins(cover, gold)["shared_membership_false"] / max(1, len(mem)),
                                     **_fast_method_row("sensitivity", cover, gold), **evaluate_membership_pairs(cover, gold), **evaluate_shared_proteins(cover, gold)})
    for name, weights in (("topology_only", {"topology": 1, "semantic": 0, "go": 0}),
                          ("semantic_only", {"topology": 0, "semantic": 1, "go": 0}),
                          ("topology_semantic", {"topology": .45, "semantic": .35, "go": .20})):
        # Weights affect the membership score and mechanism quality.  The
        # topology/semantic rows therefore rebuild candidates; the default row
        # reuses the main construction exactly.
        cands, meta, _ = (candidates, metadata, heterogeneity) if name == "topology_semantic" else construct_deconvolved_candidates(coarse, graph, evidence, weights=weights)
        mem = assign_overlapping_memberships(cands, graph, evidence, meta, weights=weights)
        cover = _membership_clusters(mem)
        sensitivity_rows.append({"factor": "evidence_weights", "setting": name, **_fast_method_row("sensitivity", cover, gold), **evaluate_membership_pairs(cover, gold), **evaluate_shared_proteins(cover, gold)})
    # Controls requested by the reviewer: split all clusters and turn off the
    # shared rule (margin 1.0 keeps only equal-strength duplicate support).
    for name, disable_hetero, margin in (("without_heterogeneity_detection", True, .85), ("without_shared_protein_rule", False, 1.0)):
        cands, meta, _ = construct_deconvolved_candidates(coarse, graph, evidence, disable_heterogeneity=disable_hetero)
        mem = assign_overlapping_memberships(cands, graph, evidence, meta, shared_margin=margin)
        cover = _membership_clusters(mem)
        sensitivity_rows.append({"factor": "control", "setting": name, **_fast_method_row("sensitivity", cover, gold), **evaluate_membership_pairs(cover, gold), **evaluate_shared_proteins(cover, gold)})
    pd.DataFrame(sensitivity_rows).to_csv(root / "threshold_sensitivity.csv", index=False)

    membership.to_csv(root / "deconvolution_memberships.csv", index=False)
    supplemented_membership.to_csv(root / "final_deconvolution_memberships.csv", index=False)
    heterogeneity.to_csv(root / "heterogeneity_scores.csv", index=False)
    provenance = {
        "dataset": dataset, "random_seed": 42, "embedding_backend": embedding_mode,
        "heterogeneity_quantile": .75, "membership_weights": {"topology": .45, "semantic": .35, "go": .20},
        "membership_threshold": "candidate-specific incumbent-score lower quartile", "shared_margin": .85,
        "supplementation": supplement_stats, "runtime_seconds": time.time() - started,
        "python": platform.python_version(), "graph_nodes": graph.number_of_nodes(), "graph_edges": graph.number_of_edges(),
    }
    try:
        provenance["commit_hash"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parents[1], text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        provenance["commit_hash"] = "unavailable (working directory is not a Git checkout)"
    (root / "experiment_log.json").write_text(json.dumps(provenance, indent=2, sort_keys=True))
    _write_synthetic(root, {"topology": 1, "semantic": 0, "go": 0})
    return {"module_metrics": module_metrics, "membership": membership, "final_clusters": final_clusters,
            "heterogeneity": heterogeneity, "stress": stress, "provenance": provenance}
