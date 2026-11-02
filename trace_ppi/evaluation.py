"""Evaluation against reference complexes (never called by module construction).

Module level   greedy one-to-one matching of predicted modules and reference complexes
               in decreasing Jaccard order, accepting pairs with J >= 0.5; precision = matched modules / modules, recall = matched
               complexes / complexes.
Membership     every predicted module is mapped to its single best-Jaccard reference
               complex (ties: lowest identifier) when J >= 0.25; a predicted pair
               (p, ref(C)) is correct when p belongs to ref(C).  Reference pairs are
               restricted to proteins present in the network.  Unmapped modules
               contribute no predicted pairs and are counted by ``unmapped_modules``.
Shared proteins network proteins in >= 2 reference complexes; their reference pairs and
               the predicted pairs of proteins predicted in >= 2 modules.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, Set, Tuple

import numpy as np
import pandas as pd


def jaccard(a: Set[str], b: Set[str]) -> float:
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def _valid(cover: Mapping) -> Dict:
    return {k: set(map(str, v)) for k, v in cover.items() if len(v) >= 2}


def _index(ref: Mapping[str, Set[str]]) -> Dict[str, List[str]]:
    by = defaultdict(list)
    for gid, mem in ref.items():
        for p in mem:
            by[p].append(gid)
    return by


def jaccard_pairs(pred: Mapping[int, Set[str]], ref: Mapping[str, Set[str]], min_j: float) -> List[Tuple[float, int, str]]:
    by = _index(ref)
    out = []
    for cid, mem in pred.items():
        cands = {g for p in mem for g in by.get(p, ())}
        for g in cands:
            j = jaccard(mem, ref[g])
            if j >= min_j:
                out.append((j, cid, g))
    return out


def module_metrics(pred, ref, thr: float = 0.5) -> dict:
    """Ties are broken by sorting pairs by (J, i, j) in reverse, with i, j the
    positions of modules and complexes in insertion order."""
    pred, ref = _valid(pred), _valid(ref)
    pi = {c: i for i, c in enumerate(pred)}
    gi = {g: j for j, g in enumerate(ref)}
    pairs = sorted(((j, pi[c], gi[g], c, g) for j, c, g in jaccard_pairs(pred, ref, thr)), reverse=True)
    used_p, used_g, matched = set(), set(), []
    for j, _, _, c, g in pairs:
        if c in used_p or g in used_g:
            continue
        used_p.add(c); used_g.add(g); matched.append((c, g, j))
    P = len(matched) / len(pred) if pred else 0.0
    R = len(matched) / len(ref) if ref else 0.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    sizes = [len(v) for v in pred.values()]
    return {"module_precision": P, "module_recall": R, "module_f1": F, "n_modules": len(pred),
            "n_matched": len(matched), "mean_size": float(np.mean(sizes)) if sizes else 0.0,
            "median_size": float(np.median(sizes)) if sizes else 0.0, "max_size": max(sizes) if sizes else 0,
            "_matched_pred": used_p, "_matched_ref": used_g}


def best_map(pred, ref, min_j: float = 0.25) -> Dict[int, str]:
    best: Dict[int, Tuple[float, str]] = {}
    for j, c, g in jaccard_pairs(_valid(pred), _valid(ref), min_j):
        cur = best.get(c)
        if cur is None or j > cur[0] or (j == cur[0] and str(g) < str(cur[1])):
            best[c] = (j, g)
    return {c: g for c, (j, g) in best.items()}


def membership_metrics(pred, ref, nodes: Set[str], min_j: float = 0.25) -> dict:
    pred, ref = _valid(pred), _valid(ref)
    mp = best_map(pred, ref, min_j)
    pp = {(p, mp[c]) for c, mem in pred.items() if c in mp for p in mem}
    gp = {(p, g) for g, mem in ref.items() for p in mem if p in nodes}
    correct = pp & gp
    P = len(correct) / len(pp) if pp else 0.0
    R = len(correct) / len(gp) if gp else 0.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    by_g = defaultdict(set)
    for p, g in gp:
        by_g[p].add(g)
    shared = {p for p, gs in by_g.items() if len(gs) >= 2}
    pred_by = defaultdict(set)
    for p, g in pp:
        pred_by[p].add(g)
    sp = {(p, g) for p, g in pp if p in shared and len(pred_by[p]) >= 2}
    sg = {(p, g) for p, g in gp if p in shared}
    sc = sp & sg
    sP = len(sc) / len(sp) if sp else 0.0
    sR = len(sc) / len(sg) if sg else 0.0
    return {"membership_precision": P, "membership_recall": R, "membership_f1": F,
            "membership_pairs_predicted": len(pp), "membership_pairs_correct": len(correct),
            "unmapped_modules": len(pred) - len(mp),
            "shared_precision": sP, "shared_recall": sR, "shared_f1": 2 * sP * sR / (sP + sR) if sP + sR else 0.0,
            "n_reference_shared_proteins": len(shared), "shared_pairs_correct": len(sc)}


def cover_stats(pred, nodes: Set[str]) -> dict:
    pred = _valid(pred)
    counts = defaultdict(int)
    for mem in pred.values():
        for p in mem:
            counts[p] += 1
    multi = sum(1 for c in counts.values() if c >= 2)
    return {"n_assignments": int(sum(counts.values())), "proteins_assigned": len(counts),
            "accepted_coverage": len(counts) / len(nodes) if nodes else 0.0,
            "overlapping_proteins": multi, "overlap_assignments": int(sum(c for c in counts.values() if c >= 2))}


def evaluate(pred, ref, nodes: Set[str], cfg: dict) -> dict:
    m = module_metrics(pred, ref, cfg["evaluation"]["module_jaccard"])
    m = {k: v for k, v in m.items() if not k.startswith("_")}
    return {**m, **membership_metrics(pred, ref, nodes, cfg["evaluation"]["membership_mapping_jaccard"]),
            **cover_stats(pred, nodes)}


# ----------------------------------------------------------------------- bootstrap
def _module_indicators(pred, ref, thr):
    m = module_metrics(pred, ref, thr)
    pv = np.array([c in m["_matched_pred"] for c in sorted(_valid(pred))], float)
    rv = {g: (g in m["_matched_ref"]) for g in _valid(ref)}
    return pv, rv


def _f1(p, r):
    return 2 * p * r / (p + r) if p + r else 0.0


def bootstrap_module_f1(covers: Mapping[str, Mapping], ref, thr: float, B: int, seed: int, base: str) -> pd.DataFrame:
    """Percentile CIs for module F1 and for the paired difference to ``base``.
    Reference complexes are resampled jointly for all methods (recall); predicted
    modules are resampled independently per method (precision); matching is fixed."""
    rng = np.random.default_rng(seed)
    gids = sorted(_valid(ref))
    ind = {name: _module_indicators(c, ref, thr) for name, c in covers.items()}
    boots = {name: [] for name in covers}
    for _ in range(B):
        gi = rng.integers(0, len(gids), len(gids))
        for name, (pv, rv) in ind.items():
            rvec = np.array([rv[gids[i]] for i in gi], float)
            pi = rng.integers(0, len(pv), len(pv)) if len(pv) else []
            boots[name].append(_f1(pv[pi].mean() if len(pv) else 0.0, rvec.mean()))
    rows = []
    for name in covers:
        b = np.array(boots[name]); d = b - np.array(boots[base])
        rows.append({"method": name, "f1_ci_low": np.quantile(b, .025), "f1_ci_high": np.quantile(b, .975),
                     f"diff_vs_{base}_mean": d.mean(), f"diff_vs_{base}_ci_low": np.quantile(d, .025),
                     f"diff_vs_{base}_ci_high": np.quantile(d, .975)})
    return pd.DataFrame(rows)


def bootstrap_membership_f1(covers, ref, nodes, min_j, B, seed, base) -> pd.DataFrame:
    """Cluster bootstrap: predicted modules resampled per method (precision), reference
    complexes resampled jointly (recall); pairs inside a module/complex move together."""
    rng = np.random.default_rng(seed)
    ref_v = _valid(ref)
    gids = sorted(ref_v)
    data = {}
    for name, c in covers.items():
        c = _valid(c)
        mp = best_map(c, ref_v, min_j)
        mods = sorted(c)
        pc = np.array([sum(1 for p in c[m] if m in mp and p in ref_v[mp[m]]) for m in mods], float)
        pn = np.array([len(c[m]) if m in mp else 0 for m in mods], float)
        pp = {(p, mp[m]) for m in mods if m in mp for p in c[m]}
        rc = np.array([sum(1 for p in ref_v[g] if p in nodes and (p, g) in pp) for g in gids], float)
        data[name] = (pc, pn, rc)
        # note: duplicate (p, g) pairs from two modules mapped to one complex are rare and
        # counted per module here; point estimates use the set-based definition.
    rn = np.array([sum(1 for p in ref_v[g] if p in nodes) for g in gids], float)
    boots = {n: [] for n in covers}
    for _ in range(B):
        gi = rng.integers(0, len(gids), len(gids))
        R_den = rn[gi].sum()
        for name, (pc, pn, rc) in data.items():
            mi = rng.integers(0, len(pc), len(pc))
            P = pc[mi].sum() / pn[mi].sum() if pn[mi].sum() else 0.0
            R = rc[gi].sum() / R_den if R_den else 0.0
            boots[name].append(_f1(P, R))
    rows = []
    for name in covers:
        b = np.array(boots[name]); d = b - np.array(boots[base])
        rows.append({"method": name, "mf1_ci_low": np.quantile(b, .025), "mf1_ci_high": np.quantile(b, .975),
                     f"mdiff_vs_{base}_mean": d.mean(), f"mdiff_vs_{base}_ci_low": np.quantile(d, .025),
                     f"mdiff_vs_{base}_ci_high": np.quantile(d, .975)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------- assignment support
def assignment_support(ledger: pd.DataFrame, modules, ref, min_j: float = 0.25) -> pd.DataFrame:
    """Mark each accepted membership / review suggestion as reference-supported when
    the protein belongs to the module's best-mapped reference complex."""
    ref_v = _valid(ref)
    mp = best_map(_valid(modules), ref_v, min_j)
    out = ledger.copy()
    out["mapped_reference"] = out.module_id.map(mp)
    out["reference_supported"] = [bool(isinstance(g, str) and p in ref_v.get(g, set()))
                                  for p, g in zip(out.protein_id, out.mapped_reference)]
    by = _index(ref_v)
    out["protein_in_any_reference"] = out.protein_id.map(lambda p: p in by)
    return out


def core_enrichment(support: pd.DataFrame, B: int, seed: int) -> dict:
    """Core vs non-Core odds ratio with a module-cluster bootstrap CI (assignments in
    the same module are dependent, so modules are the resampling unit)."""
    from scipy.stats import fisher_exact
    acc = support[support.status == "accepted"].copy()
    acc["core"] = acc.tier == "Core"
    a = int((acc.core & acc.reference_supported).sum()); b = int((acc.core & ~acc.reference_supported).sum())
    c = int((~acc.core & acc.reference_supported).sum()); d = int((~acc.core & ~acc.reference_supported).sum())
    orr, p = fisher_exact([[a, b], [c, d]])
    rng = np.random.default_rng(seed)
    groups = [g for _, g in acc.groupby("module_id")]
    stats = np.array([[int((g.core & g.reference_supported).sum()), int((g.core & ~g.reference_supported).sum()),
                       int((~g.core & g.reference_supported).sum()), int((~g.core & ~g.reference_supported).sum())]
                      for g in groups])
    ors = []
    for _ in range(B):
        s = stats[rng.integers(0, len(stats), len(stats))].sum(0) + 0.5
        ors.append((s[0] * s[3]) / (s[1] * s[2]))
    return {"core_supported": a, "core_total": a + b, "noncore_supported": c, "noncore_total": c + d,
            "core_fraction": a / (a + b) if a + b else np.nan, "noncore_fraction": c / (c + d) if c + d else np.nan,
            "odds_ratio": orr, "fisher_p_descriptive": p, "or_ci_low": float(np.quantile(ors, .025)),
            "or_ci_high": float(np.quantile(ors, .975))}


# ------------------------------------------------------------------- overmerging
def overmerging(regions: Mapping[int, Set[str]], trace_modules, trace_ledger: pd.DataFrame, ref, nodes,
                coverage: float = 0.5) -> pd.DataFrame:
    """For each MCL region (>= 2 proteins): which reference complexes it substantially
    contains (>= 50% of the complex's network proteins), whether TRACE-PPI split it,
    and whether the split separated the complexes or fragmented a single one."""
    ref_v = _valid(ref)
    ref_net = {g: {p for p in m if p in nodes} for g, m in ref_v.items()}
    ref_net = {g: m for g, m in ref_net.items() if len(m) >= 2}
    by = _index(ref_net)
    region_modules = defaultdict(list)
    meta = trace_ledger[trace_ledger.status == "accepted"].drop_duplicates("module_id").set_index("module_id")
    for mid, mem in trace_modules.items():
        if mid in meta.index:
            region_modules[int(meta.loc[mid, "region_id"])].append(mem)
    split_regions = set(trace_ledger.loc[trace_ledger.region_split == True, "region_id"].astype(int))  # noqa: E712
    flagged_regions = set(trace_ledger.loc[trace_ledger.region_flagged == True, "region_id"].astype(int))  # noqa: E712
    rows = []
    for rid, R in regions.items():
        if len(R) < 2:
            continue
        cands = {g for p in R for g in by.get(p, ())}
        contained = sorted(g for g in cands if len(ref_net[g] & R) / len(ref_net[g]) >= coverage)
        best_region_j = max((jaccard(R, ref_v[g]) for g in cands), default=0.0)
        mods = region_modules.get(rid, [])
        def best_child(g):
            return max((jaccard(mm, ref_v[g]) for mm in mods), default=0.0)
        # False memberships of the unsplit region attributable to merging: proteins of the
        # region that belong to a different contained complex than the region's best match.
        merge_fp = 0
        if len(contained) >= 2:
            best_g = max(contained, key=lambda g: (jaccard(R, ref_v[g]), g))
            merge_fp = len((R - ref_v[best_g]) & set().union(*(ref_v[g] for g in contained if g != best_g)))
        rows.append({"region_id": rid, "size": len(R), "n_contained_reference": len(contained),
                     "contained_reference": ";".join(contained), "overmerged": len(contained) >= 2,
                     "region_best_jaccard": best_region_j, "region_matches_reference": best_region_j >= 0.5,
                     "flagged": rid in flagged_regions, "split": rid in split_regions, "n_trace_modules": len(mods),
                     "merge_attributable_false_memberships": merge_fp,
                     "contained_separated": sum(1 for g in contained if best_child(g) >= 0.5) if rid in split_regions else 0,
                     "fragmentation_error": bool(rid in split_regions and best_region_j >= 0.5 and
                                                 all(best_child(g) < 0.5 for g in cands if jaccard(R, ref_v[g]) >= 0.5))})
    return pd.DataFrame(rows)
