"""TRACE-PPI end-to-end pipeline (module construction only; no reference input).

Steps: strict MCL localisation -> three-signal heterogeneity assessment ->
no-split / seed-and-expand / spectral selection with recursion -> adaptive
overlapping assignment (candidate-specific lower-quartile threshold + absolute
guard; review suggestions for topology-free semantic/GO evidence) -> bounded
supplementation -> recomputation of channels, guard and tiers on the final
module composition -> ledger and provenance export.
"""
from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass, field
from typing import Dict, List, Set

import networkx as nx
import numpy as np
import pandas as pd

from . import __version__
from .channels import Evidence
from .config import config_hash
from .datasets import Dataset
from .decomposition import deconvolve
from .heterogeneity import assess_regions
from .mcl import run_mcl
from .supplementation import supplement


@dataclass
class TraceResult:
    modules: Dict[int, Set[str]]
    ledger: pd.DataFrame            # accepted memberships, review suggestions and rejections
    regions: pd.DataFrame           # one row per MCL region with heterogeneity signals
    splits: pd.DataFrame            # objective components of every alternative considered
    candidates: Dict[int, Set[str]]
    provenance: dict
    mcl_regions: Dict[int, Set[str]] = field(default_factory=dict)

    @property
    def accepted(self) -> pd.DataFrame:
        return self.ledger[self.ledger.status == "accepted"]

    @property
    def review(self) -> pd.DataFrame:
        return self.ledger[self.ledger.status == "review_suggestion"]


def tier(tau: float, sigma: float, cfg: dict) -> str:
    c = cfg["tiers"]
    if tau >= c["core"]["tau"] and sigma >= c["core"]["sigma"]:
        return "Core"
    if tau >= c["inner"] or sigma >= c["inner"]:
        return "Inner"
    if tau >= c["outer"] or sigma >= c["outer"]:
        return "Outer"
    return "Uncertain"


def guard(s: dict, a: dict, enabled: bool = True) -> bool:
    if not enabled:
        return True
    return s["n_neighbours"] >= int(a["guard_min_neighbours"]) and max(s["tau"], s["sigma"]) >= float(a["guard_min_channel"])


def is_review(s: dict, a: dict) -> bool:
    return s["n_neighbours"] == 0 and s["sigma"] >= float(a["review_min_semantic"]) and s["go"] >= float(a["review_min_go"])


def run_trace(ds: Dataset, cfg: dict, variant: dict | None = None, regions: Dict[int, Set[str]] | None = None,
              graph: nx.Graph | None = None) -> TraceResult:
    """``variant`` keys: deconvolution (bool), heterogeneity_signals (list), forced (None|spectral|seed_expand),
    threshold_mode (candidate|global), guard (bool), supplementation (bool)."""
    t0 = time.time()
    v = {"deconvolution": True, "heterogeneity_signals": None, "forced": None, "threshold_mode": None,
         "guard": True, "supplementation": None, **(variant or {})}
    G = graph if graph is not None else ds.graph
    a, mcfg = cfg["assignment"], cfg["module"]
    ev = Evidence(G, ds.go, ds.embeddings, a["weights"])
    m_child = int(mcfg["min_split_child_size"])
    m_region = int(mcfg["min_region_size_for_split"])
    het_cfg = dict(cfg["heterogeneity"])
    if v["heterogeneity_signals"]:
        het_cfg["signals"] = list(v["heterogeneity_signals"])
        het_cfg["min_votes"] = min(int(het_cfg["min_votes"]), len(het_cfg["signals"]))
    timing = {}

    # 1. Coarse localisation.
    t = time.time()
    if regions is None:
        mcl = run_mcl(G, cfg["mcl"]["inflation"], cfg["mcl"]["expansion"], cfg["mcl"]["self_loop_value"])
        regions, mcl_info = mcl.regions, {"converged": mcl.converged, "n_regions": len(mcl.regions),
                                           "n_singletons": mcl.n_singletons, "max_region": mcl.max_region}
    else:
        mcl_info = {"n_regions": len(regions), "supplied": True}
    timing["mcl"] = time.time() - t

    # 2. Heterogeneity assessment.
    t = time.time()
    region_table, thresholds = assess_regions(regions, ev, het_cfg, m_region)
    timing["heterogeneity"] = time.time() - t

    # 3. Decomposition.
    t = time.time()
    alternatives = list(cfg["decomposition"]["alternatives"])
    cand: Dict[int, Set[str]] = {}
    cand_meta: Dict[int, dict] = {}
    split_rows: List[dict] = []
    rinfo = region_table.set_index("region_id").to_dict("index")
    for rid in sorted(regions):
        R = {str(x) for x in regions[rid]}
        if len(R) < int(mcfg["min_module_size"]):
            continue
        info = rinfo[rid]
        if info["flagged"] and v["deconvolution"]:
            leaves, audit = deconvolve(R, ev, thresholds, het_cfg, cfg["decomposition"], m_child, m_region, rid,
                                       alternatives, v["forced"])
            split_rows += audit
            method = "split" if len(leaves) > 1 else "no_split"
        else:
            leaves = [(R, "not_flagged" if info["eligible"] else "ineligible", 0)]
            method = "not_assessed"
        for members, path, depth in leaves:
            cid = len(cand)
            cand[cid] = set(members)
            cand_meta[cid] = {"region_id": rid, "region_size": len(R), "region_flagged": bool(info["flagged"]),
                              "region_split": len(leaves) > 1, "decomposition_path": path, "depth": depth,
                              "selected_method": (path.split("[")[0] if len(leaves) > 1 else
                                                  ("no_split" if info["flagged"] and v["deconvolution"] else path))}
    timing["decomposition"] = time.time() - t

    # 4. Adaptive overlapping assignment, region by region.
    t = time.time()
    q = float(a["quantile"])
    mode = v["threshold_mode"] or a["threshold_mode"]
    states = {cid: ev.module(c) for cid, c in cand.items()}
    inc_scores = {cid: [states[cid].scores(p)["S"] for p in sorted(c)] for cid, c in cand.items()}
    pooled = [s for vals in inc_scores.values() for s in vals]
    theta_global = float(np.quantile(pooled, q)) if pooled else 0.0
    theta, theta_src = {}, {}
    for cid, vals in inc_scores.items():
        if mode == "global":
            theta[cid], theta_src[cid] = theta_global, "global"
        elif len(vals) >= int(a["min_incumbents_for_quantile"]):
            theta[cid], theta_src[cid] = float(np.quantile(vals, q)), "candidate_quartile"
        else:
            theta[cid], theta_src[cid] = theta_global, "global_fallback_lt4_incumbents"
    by_region: Dict[int, List[int]] = {}
    for cid, meta in cand_meta.items():
        by_region.setdefault(meta["region_id"], []).append(cid)
    rows: List[dict] = []
    accepted: Dict[int, Set[str]] = {cid: set() for cid in cand}
    for rid, cids in sorted(by_region.items()):
        R = {str(x) for x in regions[rid]}
        for p in sorted(R):
            got = []
            best = None
            for cid in cids:
                s = states[cid].scores(p)
                incumbent = p in cand[cid]
                ok = guard(s, a, v["guard"])
                if ok and (best is None or s["S"] > best[1]["S"]):
                    best = (cid, s)
                if incumbent and ok:
                    status, reason = "accepted", "incumbent_passes_guard"
                    origin = "mcl_region_retained" if not cand_meta[cid]["region_split"] else "decomposition_incumbent"
                elif not incumbent and ok and s["S"] >= theta[cid]:
                    status, reason, origin = "accepted", "score_at_or_above_candidate_threshold", "adaptive_overlap"
                elif is_review(s, a):
                    status, reason, origin = ("review_suggestion", "semantic_and_go_support_without_topology",
                                              "incumbent" if incumbent else "adaptive_overlap")
                else:
                    if not incumbent and s["n_neighbours"] == 0:
                        continue  # no evidence of any kind worth recording
                    status = "rejected"
                    reason = ("guard_failed" if not ok else "below_candidate_threshold")
                    origin = "incumbent" if incumbent else "adaptive_overlap"
                rows.append({"protein_id": p, "candidate_id": cid, "status": status, "reason": reason,
                             "origin": origin, "theta": theta[cid], "threshold_source": theta_src[cid],
                             "guard_pass": ok, "stage": "assignment", **{f"assign_{k}": s[k] for k in ("tau", "sigma", "go", "S")}})
                if status == "accepted":
                    got.append(cid)
                    accepted[cid].add(p)
            if not got and best is not None and a.get("best_supported_fallback", True):
                cid, s = best
                accepted[cid].add(p)
                for r in rows[::-1]:
                    if r["protein_id"] == p and r["candidate_id"] == cid:
                        r.update(status="accepted", reason="best_supported_fallback", origin="best_supported_fallback")
                        break
    timing["assignment"] = time.time() - t

    # 5. Module formation: drop modules < min size; merge identical member sets.
    min_size = int(mcfg["min_module_size"])
    modules: Dict[int, Set[str]] = {}
    seen: Dict[frozenset, int] = {}
    cand_to_module: Dict[int, int] = {}
    for cid in sorted(accepted):
        mem = accepted[cid]
        if len(mem) < min_size:
            continue
        key = frozenset(mem)
        if key in seen:
            cand_to_module[cid] = seen[key]
            continue
        mid = len(modules)
        modules[mid] = set(mem)
        seen[key] = mid
        cand_to_module[cid] = mid

    # 6. Bounded supplementation (after deconvolution and initial assignment).
    t = time.time()
    sup_cfg = dict(cfg["supplementation"])
    do_sup = sup_cfg.get("enabled", True) if v["supplementation"] is None else v["supplementation"]
    sup_log: List[dict] = []
    sup_stats = {}
    if do_sup:
        modules, sup_log, sup_stats = supplement(modules, G, ds.go, ds.embeddings, sup_cfg)
    timing["supplementation"] = time.time() - t
    supplemented = {(r["module_id"], r["protein_id"]): r["supplementation_gain"] for r in sup_log}

    # 7. Final recomputation of channels, guard and tiers.
    t = time.time()
    module_to_cands: Dict[int, List[int]] = {}
    for cid, mid in cand_to_module.items():
        module_to_cands.setdefault(mid, []).append(cid)
    assign_index = {(r["protein_id"], r["candidate_id"]): r for r in rows}
    final_rows = []
    final_modules: Dict[int, Set[str]] = {}
    for mid in sorted(modules):
        st = ev.module(modules[mid])
        keep = set()
        cids = module_to_cands.get(mid, [])
        meta = cand_meta[cids[0]] if cids else {}
        for p in sorted(modules[mid]):
            s = st.scores(p)
            ok = guard(s, a, v["guard"])
            src = next((assign_index[(p, c)] for c in cids if (p, c) in assign_index), None)
            if (mid, p) in supplemented:
                origin, reason0 = "supplementation", "bounded_supplementation"
            else:
                origin, reason0 = (src["origin"], src["reason"]) if src else ("unknown", "")
            if ok:
                status, reason = "accepted", reason0
                keep.add(p)
            elif is_review(s, a):
                status, reason = "review_suggestion", "guard_failed_after_final_recompute"
            else:
                status, reason = "rejected", "guard_failed_after_final_recompute"
            final_rows.append({"protein_id": p, "module_id": mid, "candidate_id": cids[0] if cids else -1,
                               "status": status, "reason": reason, "origin": origin, "stage": "final",
                               "theta": theta[cids[0]] if cids else np.nan,
                               "threshold_source": theta_src[cids[0]] if cids else "",
                               "guard_pass": ok, "tau": s["tau"], "sigma": s["sigma"], "go": s["go"], "S": s["S"],
                               "n_neighbours_in_module": s["n_neighbours"], "annotation_missing": s["annotation_missing"],
                               "shared_go_terms": st.shared_terms(p), "module_size": len(modules[mid]),
                               "supplementation_gain": supplemented.get((mid, p), np.nan), **meta})
        if len(keep) >= min_size:
            final_modules[mid] = keep
        else:
            for r in final_rows:
                if r["module_id"] == mid and r["status"] == "accepted":
                    r.update(status="rejected", reason="module_below_min_size_after_recompute")
    # Review suggestions and rejections from the assignment stage that never entered a module.
    in_module = {(r["protein_id"], r["candidate_id"]) for r in final_rows}
    for r in rows:
        if r["status"] in ("review_suggestion", "rejected") and (r["protein_id"], r["candidate_id"]) not in in_module:
            mid = cand_to_module.get(r["candidate_id"], -1)
            final_rows.append({**{k: r[k] for k in ("protein_id", "candidate_id", "status", "reason", "origin",
                                                    "theta", "threshold_source", "guard_pass")},
                               "module_id": mid, "stage": "assignment",
                               "tau": r["assign_tau"], "sigma": r["assign_sigma"], "go": r["assign_go"], "S": r["assign_S"],
                               **cand_meta[r["candidate_id"]]})
    ledger = pd.DataFrame(final_rows)
    ledger["tier"] = [tier(t_, s_, cfg) if st_ == "accepted" else ("Review" if st_ == "review_suggestion" else "")
                      for t_, s_, st_ in zip(ledger.tau, ledger.sigma, ledger.status)]
    acc = ledger[ledger.status == "accepted"]
    counts = acc.groupby("protein_id").module_id.nunique()
    ledger["n_accepted_memberships"] = ledger.protein_id.map(counts).fillna(0).astype(int)
    het_cols = ["topology", "semantic", "support", "votes", "topology_flag", "semantic_flag", "support_flag"]
    het = region_table.set_index("region_id")[het_cols].add_prefix("heterogeneity_")
    ledger = ledger.join(het, on="region_id")
    timing["final_recompute"] = time.time() - t

    prov = {"method": "TRACE-PPI", "version": __version__, "dataset": ds.name, "config_hash": config_hash(cfg),
            "variant": {k: v[k] for k in v}, "random_seed": cfg["random_seed"], "mcl": {**cfg["mcl"], **mcl_info},
            "heterogeneity_thresholds": thresholds, "theta_global": theta_global,
            "supplementation_stats": sup_stats, "timing_seconds": {**timing, "total": time.time() - t0},
            "python": platform.python_version(), "networkx": nx.__version__, "numpy": np.__version__,
            "pandas": pd.__version__, "graph_nodes": G.number_of_nodes(), "graph_edges": G.number_of_edges(),
            "embedding_backend": ds.meta.get("embedding_backend")}
    ledger["provenance"] = f"TRACE-PPI {__version__}|{ds.name}|config={prov['config_hash']}|" \
                           f"mcl_I={cfg['mcl']['inflation']}|w={a['weights']['topology']}/{a['weights']['semantic']}/{a['weights']['go']}|q={q}|mode={mode}"
    splits = pd.DataFrame(split_rows)
    return TraceResult(modules=final_modules, ledger=ledger, regions=region_table, splits=splits,
                       candidates=cand, provenance=prov, mcl_regions=regions)
