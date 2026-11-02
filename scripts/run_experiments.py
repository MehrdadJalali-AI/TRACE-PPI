#!/usr/bin/env python3
"""Run all TRACE-PPI revision experiments for one dataset.

    python3 scripts/run_experiments.py --dataset gavin   (run first: selects tuned baselines)
    python3 scripts/run_experiments.py --dataset krogan
    python3 scripts/run_experiments.py --dataset bioplex

Baseline tuning: MCL inflation, SLPA threshold and ClusterONE density are selected by
module F1 on Gavin (the designated development dataset) and applied unchanged to Krogan
and BioPlex.  On Gavin itself these tuned rows are therefore optimistic (tuned on the
evaluation reference).  TRACE-PPI uses the frozen configuration throughout.
"""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trace_ppi import baselines as B  # noqa: E402
from trace_ppi.config import config_hash, load_config, override  # noqa: E402
from trace_ppi.datasets import Dataset, load_dataset, load_gavin_graph, load_reference, sha256, PATHS  # noqa: E402
from trace_ppi.evaluation import (assignment_support, bootstrap_membership_f1, bootstrap_module_f1,  # noqa: E402
                                  core_enrichment, evaluate, overmerging)
from trace_ppi.pipeline import run_trace  # noqa: E402
from trace_ppi.stability import stability  # noqa: E402

OUT = ROOT / "results" / "revised"
INFLATIONS = [1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
SLPA_T = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
C1_DENSITY = [None, 0.2, 0.3, 0.4, 0.5, 0.6]


def weights_oat():
    base = {"topology": 0.45, "semantic": 0.35, "go": 0.20}
    out = []
    for k in base:
        for d in (-0.10, 0.10):
            w = dict(base)
            w[k] = base[k] + d
            rest = 1 - w[k]
            others = [o for o in base if o != k]
            tot = sum(base[o] for o in others)
            for o in others:
                w[o] = base[o] * rest / tot
            out.append((f"{k}{d:+.2f}", {o: round(v, 4) for o, v in w.items()}))
    return out


SENSITIVITY = (
    [("mcl.inflation", v) for v in [1.4, 1.6, 1.8, 2.5, 3.0, 4.0]]
    + [("heterogeneity.quantile", v) for v in [0.50, 0.65, 0.85, 0.90]]
    + [("heterogeneity.min_votes", v) for v in [1, 3]]
    + [("decomposition.lambda_separation", v) for v in [0.0, 0.25, 1.0]]
    + [("decomposition.lambda_fragmentation", v) for v in [0.0, 0.10, 0.50]]
    + [("decomposition.min_improvement", v) for v in [0.0, 0.05]]
    + [("assignment.quantile", v) for v in [0.10, 0.40, 0.50]]
    + [("assignment.weights", w) for _, w in weights_oat()]
    + [("assignment.guard_min_channel", v) for v in [0.0, 0.20, 0.25]]
    + [("assignment.guard_min_neighbours", 0)]
    + [("supplementation.min_evidence_score", v) for v in [0.304, 0.456]]
    + [("supplementation.max_extra_members_per_module", v) for v in [0, 4]]
)

VARIANTS = {
    "TRACE-PPI (full)": {},
    "TRACE-PPI without deconvolution": {"deconvolution": False},
    "TRACE-PPI topology-only heterogeneity": {"heterogeneity_signals": ["topology"]},
    "TRACE-PPI global admission threshold": {"threshold_mode": "global"},
    "TRACE-PPI seed-and-expand only": {"forced": "seed_expand"},
    "TRACE-PPI spectral only": {"forced": "spectral"},
    "TRACE-PPI without supplementation": {"supplementation": False},
}


def timed(fn, *a, **k):
    t = time.time()
    out = fn(*a, **k)
    return out, time.time() - t


def no_annotation(ds: Dataset) -> Dataset:
    return Dataset(ds.name, ds.graph, {p: set() for p in ds.go}, ds.embeddings, ds.labels, ds.meta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["gavin", "krogan", "bioplex"])
    ap.add_argument("--stability-replicates", type=int, default=None)
    ap.add_argument("--skip-sensitivity", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    out = OUT / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    log = {"dataset": args.dataset, "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "config_hash": config_hash(cfg)}

    ds, t_load = timed(load_dataset, args.dataset, cfg)
    ref, ref_info = load_reference(args.dataset, ds)
    nodes = set(map(str, ds.graph.nodes()))
    log.update(load_seconds=t_load, dataset_meta={k: v for k, v in ds.meta.items() if k != "uniprot_map"},
               reference=ref_info)
    rows = []

    def record(method, family, cover, runtime, extra=None):
        r = {"dataset": args.dataset, "method": method, "family": family, "runtime_seconds": runtime,
             **evaluate(cover, ref, nodes, cfg), **(extra or {})}
        rows.append(r)
        print(f"{method:55s} F1={r['module_f1']:.3f} mF1={r['membership_f1']:.3f} n={r['n_modules']}", flush=True)
        return r

    covers = {}
    # ---------------------------------------------------------------- TRACE-PPI
    main_res, t_main = timed(run_trace, ds, cfg)
    covers["TRACE-PPI (full)"] = main_res.modules
    n_rev = int((main_res.ledger.status == "review_suggestion").sum())
    record("TRACE-PPI (full)", "TRACE-PPI", main_res.modules, t_main,
           {"review_suggestions": n_rev, "review_proteins": main_res.review.protein_id.nunique()})
    support = assignment_support(main_res.ledger, main_res.modules, ref, cfg["evaluation"]["membership_mapping_jaccard"])
    support.to_csv(out / "trace_ledger.csv", index=False)
    main_res.regions.to_csv(out / "trace_regions.csv", index=False)
    main_res.splits.to_csv(out / "trace_split_audit.csv", index=False)
    pd.DataFrame([{"module_id": m, "protein_id": p} for m, mem in main_res.modules.items() for p in sorted(mem)]
                 ).to_csv(out / "trace_modules.csv", index=False)
    (out / "trace_provenance.json").write_text(json.dumps(main_res.provenance, indent=2, default=str))

    for name, var in list(VARIANTS.items())[1:]:
        res, t = timed(run_trace, ds, cfg, var)
        covers[name] = res.modules
        record(name, "TRACE-PPI ablation", res.modules, t,
               {"review_suggestions": int((res.ledger.status == "review_suggestion").sum())})
    res, t = timed(run_trace, no_annotation(ds), cfg)
    covers["TRACE-PPI without semantic/GO evidence"] = res.modules
    record("TRACE-PPI without semantic/GO evidence", "TRACE-PPI ablation", res.modules, t)

    # ---------------------------------------------------------------- baselines
    mcl, t = timed(B.mcl_modules, ds.graph, 2.0)
    covers["MCL (I=2.0)"] = mcl
    record("MCL (I=2.0)", "baseline", mcl, t)
    ov, t = timed(B.mcl_overlap, ds.graph, ds.go, 2.0)
    covers["MCL + overlap"] = ov
    record("MCL + overlap", "baseline", ov, t)
    eo, t = timed(B.mcl_overlap_supplementation, ds.graph, ds.go, ds.embeddings, 2.0)
    covers["MCL + overlap + supplementation"] = eo
    record("MCL + overlap + supplementation", "baseline", eo, t)
    ge, t = timed(B.graph_entropy_clusters, ds.graph)
    covers["GE (reimplementation)"] = ge
    record("GE (reimplementation of Jeong et al.)", "baseline", ge, t)

    weighted = load_gavin_graph() if args.dataset == "gavin" else ds.graph
    grids = []
    for inf in INFLATIONS:
        try:
            c, t = timed(B.mcl_modules, ds.graph, inf)
            grids.append({"parameter": "mcl_inflation", "value": inf, "status": "ok", "runtime_seconds": t,
                          **evaluate(c, ref, nodes, cfg)})
        except Exception as exc:  # reported, never replaced
            grids.append({"parameter": "mcl_inflation", "value": inf, "status": f"failed: {exc}"})
    slpa, t_slpa = timed(B.slpa_multi, ds.graph, SLPA_T)
    for th, c in slpa.items():
        grids.append({"parameter": "slpa_threshold", "value": th, "status": "ok", "runtime_seconds": t_slpa,
                      **evaluate(c, ref, nodes, cfg)})
    c1 = {}
    for d in C1_DENSITY:
        c, t = timed(B.clusterone, weighted, d)
        c1[d] = c
        grids.append({"parameter": "clusterone_density", "value": "default" if d is None else d, "status": "ok",
                      "runtime_seconds": t, **evaluate(c, ref, nodes, cfg)})
    grid = pd.DataFrame(grids)
    grid.insert(0, "dataset", args.dataset)
    grid.to_csv(out / "baseline_parameter_grids.csv", index=False)

    sel_path = OUT / "baseline_selection_gavin.json"
    if args.dataset == "gavin":
        sel = {}
        for par in ("mcl_inflation", "slpa_threshold", "clusterone_density"):
            g = grid[(grid.parameter == par) & (grid.status == "ok")]
            best = g.loc[g.module_f1.idxmax()]   # first grid value attaining the maximum
            sel[par] = best["value"]
        sel_path.write_text(json.dumps(sel, indent=2, default=str))
    sel = json.loads(sel_path.read_text())
    log["baseline_selection_from_gavin"] = sel
    inf = float(sel["mcl_inflation"])
    c, t = timed(B.mcl_modules, ds.graph, inf)
    covers["Tuned MCL"] = c
    record(f"Tuned MCL (I={inf}, selected on Gavin)", "baseline", c, t)
    th = float(sel["slpa_threshold"])
    covers["SLPA"] = slpa[th]
    record(f"SLPA (t={th}, selected on Gavin)", "baseline", slpa[th], t_slpa)
    covers["ClusterONE"] = c1[None]
    record("ClusterONE (default)", "baseline", c1[None], float(grid[(grid.parameter == "clusterone_density") & (grid.value == "default")].runtime_seconds.iloc[0]))
    dsel = None if sel["clusterone_density"] == "default" else float(sel["clusterone_density"])
    covers["Tuned ClusterONE"] = c1[dsel]
    record(f"ClusterONE (d={sel['clusterone_density']}, selected on Gavin)", "baseline", c1[dsel], 0.0)
    tr_tuned, t = timed(run_trace, ds, override(cfg, {"mcl.inflation": inf}))
    covers["TRACE-PPI from tuned MCL seed"] = tr_tuned.modules
    record(f"TRACE-PPI from tuned MCL seed (I={inf}; sensitivity)", "TRACE-PPI sensitivity", tr_tuned.modules, t)

    # ---------------------------------------------------------------- stability
    st_cfg = cfg["stability"]
    Bst = args.stability_replicates or int(st_cfg["replicates"])
    stab_rows, run_rows, split_rows = [], [], []
    for rate in ([0.05, 0.10, 0.20] if args.dataset != "bioplex" else [0.10]):
        reps = Bst if rate == st_cfg["edge_removal_fraction"] else max(5, Bst // 2)
        mem, runs, splits = stability(ds, cfg, main_res, rate, reps, 1000, st_cfg["alignment_jaccard"])
        stab_rows.append(mem); run_rows.append(runs); split_rows.append(splits)
        print(f"stability rate={rate} reps={reps} mean={mem.stability.mean():.3f}", flush=True)
    stab = pd.concat(stab_rows)
    stab.to_csv(out / "stability_memberships.csv", index=False)
    pd.concat(run_rows).to_csv(out / "stability_runs.csv", index=False)
    pd.concat(split_rows).to_csv(out / "stability_splits.csv", index=False)
    gate_rows = []
    for rate, g in stab.groupby("rate"):
        for thr in (0.3, 0.5, 0.7):
            keep = set(map(tuple, g[g.stability >= thr][["protein_id", "module_id"]].values))
            filt = {}
            for m, mem in main_res.modules.items():
                kept = {p for p in mem if (p, m) in keep}
                if len(kept) >= 2:
                    filt[m] = kept
            ev_ = evaluate(filt, ref, nodes, cfg)
            gate_rows.append({"rate": rate, "stability_threshold": thr, **ev_})
            if rate == st_cfg["edge_removal_fraction"] and thr == st_cfg["gate_threshold"]:
                covers["TRACE-PPI with stability gate"] = filt
                record("TRACE-PPI with stability gate (diagnostic)", "TRACE-PPI ablation", filt, 0.0)
    gate = pd.DataFrame(gate_rows)
    gate.to_csv(out / "stability_gate.csv", index=False)
    full_row = next(r for r in rows if r["method"] == "TRACE-PPI (full)")
    g0 = gate[(gate.rate == st_cfg["edge_removal_fraction"]) & (gate.stability_threshold == st_cfg["gate_threshold"])].iloc[0]
    decision = {"dataset": args.dataset,
                "precision_gain": float(g0.membership_precision - full_row["membership_precision"]),
                "f1_change": float(g0.membership_f1 - full_row["membership_f1"]),
                "coverage_change": float(g0.accepted_coverage - full_row["accepted_coverage"])}
    decision["rule_satisfied"] = bool(decision["precision_gain"] >= st_cfg["decision_min_precision_gain"]
                                      and decision["f1_change"] >= 0)
    decision["is_decision_dataset"] = args.dataset == st_cfg["decision_dataset"]
    (out / "stability_gate_decision.json").write_text(json.dumps(decision, indent=2))

    # --------------------------------------------------------------- inference
    bcfg = cfg["evaluation"]
    nb = int(bcfg["bootstrap_replicates"])
    keyset = ["TRACE-PPI (full)", "MCL + overlap + supplementation", "MCL (I=2.0)", "Tuned MCL", "MCL + overlap",
              "ClusterONE", "SLPA", "GE (reimplementation)", "TRACE-PPI without deconvolution"]
    sub = {k: covers[k] for k in keyset}
    bm = bootstrap_module_f1(sub, ref, bcfg["module_jaccard"], nb, 7, "TRACE-PPI (full)")
    bb = bootstrap_membership_f1(sub, ref, nodes, bcfg["membership_mapping_jaccard"], nb, 7, "TRACE-PPI (full)")
    bm.merge(bb, on="method").to_csv(out / "bootstrap_ci.csv", index=False)
    enr = {"TRACE-PPI": core_enrichment(support, nb, 11)}
    pd.DataFrame([{"dataset": args.dataset, "method": k, **v} for k, v in enr.items()]).to_csv(out / "core_enrichment.csv", index=False)
    tiers = support[support.status.isin(["accepted", "review_suggestion"])].groupby("tier").agg(
        n=("protein_id", "size"), reference_supported=("reference_supported", "mean"),
        mean_tau=("tau", "mean"), mean_sigma=("sigma", "mean"), mean_go=("go", "mean"), mean_S=("S", "mean")).reset_index()
    tiers.insert(0, "dataset", args.dataset)
    tiers.to_csv(out / "tier_summary.csv", index=False)
    om = overmerging(main_res.mcl_regions, main_res.modules, main_res.ledger, ref, nodes)
    om.to_csv(out / "overmerging_regions.csv", index=False)
    split_audit = main_res.splits
    root_rows = split_audit[split_audit.depth == 0] if len(split_audit) else split_audit
    summ = {"mcl_regions_ge2": int((om["size"] >= 2).sum()), "eligible_regions": int(main_res.regions.eligible.sum()),
            "overmerged_regions": int(om.overmerged.sum()),
            "overmerged_flagged": int((om.overmerged & om.flagged).sum()),
            "overmerged_split": int((om.overmerged & om.split).sum()),
            "overmerged_separated_ge2": int((om.overmerged & (om.contained_separated >= 2)).sum()),
            "merge_attributable_false_memberships": int(om.merge_attributable_false_memberships.sum()),
            "flagged_regions": int(main_res.regions.flagged.sum()),
            "split_regions": int(om.split.sum()),
            "flagged_rejected_no_split": int(main_res.regions.flagged.sum() - om.split.sum()),
            "fragmentation_errors": int(om.fragmentation_error.sum()),
            "split_regions_matching_reference_before_split": int((om.split & om.region_matches_reference).sum())}
    (out / "overmerging_summary.json").write_text(json.dumps(summ, indent=2))

    # ------------------------------------------------------------- sensitivity
    if not args.skip_sensitivity:
        sens = []
        for key, val in SENSITIVITY:
            c2 = override(cfg, {key: val})
            res, t = timed(run_trace, ds, c2)
            r = evaluate(res.modules, ref, nodes, cfg)
            sens.append({"dataset": args.dataset, "parameter": key, "value": json.dumps(val), "runtime_seconds": t,
                         "flagged": int(res.regions.flagged.sum()),
                         "split_regions": int(res.ledger[res.ledger.region_split == True].region_id.nunique()),  # noqa: E712
                         "review_suggestions": int((res.ledger.status == "review_suggestion").sum()), **r})
            print(f"sens {key}={val}: F1={r['module_f1']:.3f} mF1={r['membership_f1']:.3f}", flush=True)
        pd.DataFrame(sens).to_csv(out / "sensitivity.csv", index=False)

    bench = pd.DataFrame(rows)
    bench.to_csv(out / "benchmark.csv", index=False)
    log["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    log["peak_rss_mb_process"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2
    log["input_sha256"] = {k: sha256(p) for k, p in PATHS.items() if p.exists() and
                           ((args.dataset == "bioplex") == (k in ("bioplex", "goa_human", "complexportal_human")))}
    (out / "run_log.json").write_text(json.dumps(log, indent=2, default=str))


if __name__ == "__main__":
    main()
