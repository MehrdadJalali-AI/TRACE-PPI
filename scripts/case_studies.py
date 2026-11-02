#!/usr/bin/env python3
"""Systematic case studies selected by a declared rule from saved TRACE-PPI outputs.

Selection rule (declared before inspection of individual cases; applied to Gavin, the
primary yeast dataset, and to BioPlex for the human shared-protein case):
  C1 split        every MCL region that TRACE-PPI split (all are reported);
  C2 rejected     the flagged region containing the largest number of reference complexes
                  for which "no split" was selected (ties: larger region, lower id);
  C3 missed       the largest overmerged region that was not flagged (detector miss);
  C4 shared       the protein with >= 2 accepted memberships having the most
                  reference-supported memberships (ties: higher mean S, then identifier);
                  one per dataset (Gavin, BioPlex);
  C5 ambiguous    the review suggestion with the highest sigma + g (ties: identifier).
Literature: Complex Portal entries matched to the case modules (best Jaccard) and the
PubMed identifiers they cite; titles are resolved through Europe PMC to verify identity.
These records document plausibility against curated literature, not experimental or
expert validation.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from trace_ppi.config import load_config  # noqa: E402
from trace_ppi.datasets import load_dataset, load_reference, PATHS  # noqa: E402

REV = ROOT / "results" / "revised"
OUT = REV / "case_studies"


def cp_table(name):
    path = PATHS["complexportal_human"] if name == "bioplex" else PATHS["complexportal_yeast"]
    df = pd.read_csv(path, sep="\t", dtype=str)
    df["pubmed"] = df["Cross references"].fillna("").str.findall(r"pubmed:(\d+)")
    return df.set_index("#Complex ac")


def europepmc_titles(pmids):
    out = {}
    for p in pmids:
        url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:{p}%20AND%20SRC:MED&format=json&resultType=lite"
        for _ in range(3):
            try:
                r = json.load(urllib.request.urlopen(url, timeout=30))["resultList"]["result"]
                out[p] = (f"{r[0].get('authorString', '')} {r[0].get('title', '')} {r[0].get('journalTitle', '')} "
                          f"{r[0].get('pubYear', '')}" if r else "NOT FOUND")
                break
            except Exception:
                time.sleep(2)
        else:
            out[p] = "LOOKUP FAILED"
    return out


def sgd_symbols():
    out = {}
    with open(PATHS["sgd_gaf"], encoding="latin-1") as fh:
        for line in fh:
            if line.startswith("!"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) > 10 and f[10]:
                out.setdefault(f[10].split("|")[0], f[2])
    return out


def jac(a, b):
    return len(a & b) / len(a | b) if a | b else 0.0


def best_refs(members, ref, k=3):
    s = sorted(((jac(members, m), g) for g, m in ref.items() if members & m), reverse=True)
    return [(g, j) for j, g in s[:k]]


def main():
    cfg = load_config()
    OUT.mkdir(parents=True, exist_ok=True)
    cases, rows = [], []
    for name in ("gavin", "bioplex"):
        d = REV / name
        led = pd.read_csv(d / "trace_ledger.csv", dtype={"protein_id": str})
        mods = pd.read_csv(d / "trace_modules.csv", dtype={"protein_id": str}).groupby("module_id").protein_id.apply(set).to_dict()
        om = pd.read_csv(d / "overmerging_regions.csv")
        regions = pd.read_csv(d / "trace_regions.csv").set_index("region_id")
        stab = pd.read_csv(d / "stability_memberships.csv", dtype={"protein_id": str})
        stab = stab[stab.rate == cfg["stability"]["edge_removal_fraction"]].set_index(["protein_id", "module_id"]).stability
        ds = load_dataset(name, cfg)
        ref, _ = load_reference(name, ds)
        cp = cp_table(name)
        sym = {k: str(v) for k, v in ds.labels.items()} if name == "bioplex" else sgd_symbols()
        acc = led[led.status == "accepted"].copy()
        acc["stability"] = [stab.get((p, m), float("nan")) for p, m in zip(acc.protein_id, acc.module_id)]
        sel = []
        if name == "gavin":
            for rid in sorted(om[om.split].region_id):
                sel.append(("C1 split region", "region", int(rid)))
            rej = om[om.flagged & ~om.split].sort_values(["n_contained_reference", "size", "region_id"], ascending=[False, False, True])
            if len(rej):
                sel.append(("C2 flagged, split rejected", "region", int(rej.iloc[0].region_id)))
            miss = om[om.overmerged & ~om.flagged].sort_values(["size", "region_id"], ascending=[False, True])
            if len(miss):
                sel.append(("C3 overmerged, not flagged", "region", int(miss.iloc[0].region_id)))
            rv = led[led.status == "review_suggestion"].assign(sg=lambda x: x.sigma + x.go).sort_values(["sg", "protein_id"], ascending=[False, True])
            if len(rv):
                sel.append(("C5 review suggestion", "protein", (rv.iloc[0].protein_id, int(rv.iloc[0].module_id))))
        multi = acc[acc.n_accepted_memberships >= 2]
        g = multi.groupby("protein_id").agg(sup=("reference_supported", "sum"), S=("S", "mean")).reset_index()
        g = g.sort_values(["sup", "S", "protein_id"], ascending=[False, False, True])
        if len(g):
            sel.append(("C4 shared protein", "protein", (g.iloc[0].protein_id, None)))
        for label, kind, key in sel:
            case = {"dataset": name, "case": label}
            if kind == "region":
                rid = key
                rr = regions.loc[rid]
                omr = om.set_index("region_id").loc[rid]
                child = acc[acc.region_id == rid]
                case.update(region_id=rid, region_size=int(rr.n), topology=rr.topology, semantic=rr.semantic,
                            support=rr.support, votes=int(rr.votes), flagged=bool(rr.flagged), split=bool(omr.split),
                            contained_reference=omr.contained_reference, n_contained=int(omr.n_contained_reference),
                            region_best_jaccard=omr.region_best_jaccard)
                for mid in sorted(child.module_id.unique()):
                    mem = mods.get(mid, set())
                    br = best_refs(mem, ref)
                    sub = child[child.module_id == mid]
                    rows.append({"dataset": name, "case": label, "region_id": rid, "module_id": int(mid), "module_size": len(mem),
                                 "best_reference": ";".join(f"{gg}({cp.loc[gg, 'Recommended name'] if gg in cp.index else ''}) J={j:.2f}" for gg, j in br),
                                 "median_tau": sub.tau.median(), "median_sigma": sub.sigma.median(), "median_go": sub.go.median(),
                                 "core_fraction": (sub.tier == "Core").mean(), "reference_supported_fraction": sub.reference_supported.mean(),
                                 "median_stability": sub.stability.median(), "selected_method": sub.selected_method.iloc[0],
                                 "decomposition_path": sub.decomposition_path.iloc[0]})
                split_rows = pd.read_csv(d / "trace_split_audit.csv")
                sr = split_rows[(split_rows.region_id == rid) & (split_rows.depth == 0)]
                case["alternatives"] = sr[["alternative", "k", "coherence", "separation_penalty", "fragmentation_penalty", "Q", "selected"]].round(3).to_dict("records")
                case["pubmed"] = sorted({p for gg in omr.contained_reference.split(";") if gg in cp.index for p in cp.loc[gg, "pubmed"]})[:6] \
                    if isinstance(omr.contained_reference, str) else []
                case["reference_names"] = {gg: cp.loc[gg, "Recommended name"] for gg in str(omr.contained_reference).split(";") if gg in cp.index}
            else:
                pid, mid = key
                sub = led[(led.protein_id == pid) & (led.status.isin(["accepted", "review_suggestion"]))]
                if mid is not None:
                    sub = sub[sub.module_id == mid]
                case.update(protein_id=pid, symbol=sym.get(pid, ""))
                pmids = set()
                for r in sub.itertuples():
                    mem = mods.get(r.module_id, set()) if r.status == "accepted" else set()
                    br = best_refs(mem | {pid}, ref) if mem else best_refs({pid}, ref, 3)
                    for gg, _ in br[:1]:
                        if gg in cp.index:
                            pmids |= set(cp.loc[gg, "pubmed"][:3])
                    rows.append({"dataset": name, "case": label, "protein_id": pid, "symbol": sym.get(pid, ""),
                                 "module_id": int(r.module_id), "status": r.status, "tier": r.tier, "tau": r.tau, "sigma": r.sigma,
                                 "go": r.go, "S": r.S, "theta": r.theta, "guard_pass": r.guard_pass, "origin": r.origin,
                                 "reason": r.reason, "reference_supported": r.reference_supported,
                                 "stability": stab.get((pid, r.module_id), float("nan")), "module_size": len(mem),
                                 "shared_go_terms": r.shared_go_terms if isinstance(getattr(r, "shared_go_terms", ""), str) else "",
                                 "best_reference": ";".join(f"{gg}({cp.loc[gg, 'Recommended name'] if gg in cp.index else ''}) J={j:.2f}" for gg, j in br)})
                case["pubmed"] = sorted(pmids)[:6]
            cases.append(case)
    allp = sorted({p for c in cases for p in c.get("pubmed", [])})
    titles = europepmc_titles(allp)
    for c in cases:
        c["literature"] = {p: titles.get(p) for p in c.get("pubmed", [])}
    (OUT / "case_studies.json").write_text(json.dumps(cases, indent=2, default=str))
    pd.DataFrame(rows).to_csv(OUT / "case_study_records.csv", index=False)
    print(json.dumps(cases, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
