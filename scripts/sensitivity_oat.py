#!/usr/bin/env python3
"""One-at-a-time sensitivity of the reported TRACE-PPI path.

Perturbs overlap threshold, supplementation gain, the three supplementation
gates together, the size-penalty coefficient, the precision floor, and the
supplementation channel weights. BH weights and the greedy 0.6/0.4 candidate
expansion are not perturbed: they feed only the auxiliary candidate pool.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml
from scipy.stats import fisher_exact

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from trace_ppi.baselines import mcl_overlap_heuristic
from trace_ppi.cosmos_v2_runner import _load_embeddings
from trace_ppi.event_horizon_v2 import refine_v2
from trace_ppi.graph_io import filter_go, load_gavin, load_krogan
from trace_ppi.metrics_utils import clusters_from_membership
from trace_ppi.paths import DATA, RESULTS
from trace_ppi.recall_safe_supplementation import supplement_modules_recall_safe
from trace_ppi.reuse import load_gold_standard_csv, precision_recall_f1_mmr
from trace_ppi.v2_membership import modules_to_membership
from revision_utils import gold_supported_mask, normalize_label

CFG = yaml.safe_load((ROOT / "configs" / "trace_ppi_final.yaml").read_text())
BASE = dict(CFG["supplement"])


def _renorm(weights):
    total = sum(weights)
    return tuple(w / total for w in weights)


def _settings():
    rows = [("baseline", {})]
    rows.append(("overlap_tau", {"overlap_tau": round(0.1 * 0.8, 6)}))
    rows.append(("overlap_tau", {"overlap_tau": round(0.1 * 1.2, 6)}))
    rows.append(("min_evidence_score", {"min_evidence_score": round(0.38 * 0.8, 6)}))
    rows.append(("min_evidence_score", {"min_evidence_score": round(0.38 * 1.2, 6)}))
    rows.append((
        "supplement_gates",
        {
            "min_go_support": round(0.25 * 0.8, 6),
            "min_topological_support": round(0.15 * 0.8, 6),
            "min_semantic_support": round(0.28 * 0.8, 6),
        },
    ))
    rows.append((
        "supplement_gates",
        {
            "min_go_support": round(0.25 * 1.2, 6),
            "min_topological_support": round(0.15 * 1.2, 6),
            "min_semantic_support": round(0.28 * 1.2, 6),
        },
    ))
    rows.append(("size_penalty_lambda", {"size_penalty_lambda": round(0.08 * 0.8, 6)}))
    rows.append(("size_penalty_lambda", {"size_penalty_lambda": round(0.08 * 1.2, 6)}))
    rows.append(("preserve_precision_floor", {"preserve_precision_floor": round(0.24 * 0.8, 6)}))
    rows.append(("preserve_precision_floor", {"preserve_precision_floor": round(0.24 * 1.2, 6)}))
    plus = _renorm((0.40 + 0.05, 0.35 + 0.05, 0.25 + 0.05))
    minus = _renorm((0.40 - 0.05, 0.35 - 0.05, 0.25 - 0.05))
    rows.append(("channel_weights", {
        "topo_channel_weight": plus[0],
        "sem_channel_weight": plus[1],
        "go_channel_weight": plus[2],
    }))
    rows.append(("channel_weights", {
        "topo_channel_weight": minus[0],
        "sem_channel_weight": minus[1],
        "go_channel_weight": minus[2],
    }))
    return rows


def _enrichment(refined, clusters, gold):
    d = refined.copy()
    sizes = d.groupby("community_id").protein_id.transform("size")
    d = d[sizes >= 2].copy()
    flags = gold_supported_mask(d, clusters, gold).to_numpy()
    labels = d.membership_type.map(normalize_label)
    core = labels.eq("core").to_numpy()
    core_yes = int((core & flags).sum())
    core_n = int(core.sum())
    non_yes = int((~core & flags).sum())
    non_n = int((~core).sum())
    odds, pval = fisher_exact([[core_yes, core_n - core_yes], [non_yes, non_n - non_yes]])
    return core_n, float(odds), float(pval), int(len(d))


def _evaluate(graph, go, emb, gold, overlap_cache, overlap_tau, params):
    if overlap_tau not in overlap_cache:
        raw = mcl_overlap_heuristic(graph, go, inflation=2.0, overlap_tau=overlap_tau)
        overlap_cache[overlap_tau] = {
            i: {str(x) for x in mem}
            for i, mem in enumerate(raw.values())
            if len(mem) >= 2
        }
    selected, _ = supplement_modules_recall_safe(
        {i: set(mem) for i, mem in overlap_cache[overlap_tau].items()},
        graph, go, emb, params,
    )
    expanded = modules_to_membership(selected, graph, go, emb)
    refined = refine_v2(expanded, {**params, "preserve_all": True})
    clusters = clusters_from_membership(refined)
    metrics = precision_recall_f1_mmr(clusters, gold)
    core_n, odds, pval, n_assign = _enrichment(refined, clusters, gold)
    return metrics, core_n, odds, pval, n_assign


def main():
    gold = load_gold_standard_csv(str(DATA["gold_standard"]))
    out_rows = []
    for dataset, loader in (("gavin", load_gavin), ("krogan", load_krogan)):
        graph, go = loader()
        go = filter_go(go)
        profiles = pd.read_csv(RESULTS / "evidence_profiles" / f"protein_profiles_{dataset}.csv")
        _, _, _, emb = _load_embeddings(dataset, profiles)
        cache = {}
        for factor, override in _settings():
            params = {**BASE, **{k: v for k, v in override.items() if k != "overlap_tau"}}
            tau = float(override.get("overlap_tau", 0.1))
            metrics, core_n, odds, pval, n_assign = _evaluate(
                graph, go, emb, gold, cache, tau, params
            )
            row = {
                "dataset": dataset,
                "factor": factor,
                "setting": ";".join(f"{k}={override[k]:.6g}" for k in override) or "default",
                "f1": metrics["f1"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "assignments": n_assign,
                "core_assignments": core_n,
                "odds_ratio": odds,
                "fisher_p": pval,
            }
            out_rows.append(row)
            print(
                f"{dataset} {factor} {row['setting']} "
                f"F1={metrics['f1']:.4f} P={metrics['precision']:.4f} R={metrics['recall']:.4f} "
                f"n={n_assign} core={core_n} OR={odds:.3f}",
                flush=True,
            )
    df = pd.DataFrame(out_rows)
    dest = ROOT / "tables" / "revision" / "sensitivity_oat.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    print("wrote", dest)


if __name__ == "__main__":
    main()
