#!/usr/bin/env python3
"""Generate every revised manuscript table, figure and number from saved outputs.

    python3 scripts/make_manuscript_assets.py --out <manuscript>/generated

Writes LaTeX table fragments (tab_*.tex, supp_*.tex), figures (fig_*.pdf) and
``numbers.tex`` (\\newcommand macros used for every result quoted in the prose).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REV = ROOT / "results" / "revised"
DS = ["gavin", "krogan", "bioplex"]
DSN = {"gavin": "Gavin", "krogan": "Krogan", "bioplex": "BioPlex"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "pdf.fonttype": 42, "svg.fonttype": "none"})

MAIN_METHODS = [
    ("TRACE-PPI (full)", "TRACE-PPI"),
    ("MCL + overlap + supplementation", "MCL + overlap + suppl."),
    ("MCL (I=2.0)", "MCL ($I=2.0$)"),
    ("Tuned MCL", "Tuned MCL$^{a}$"),
    ("MCL + overlap", "MCL + overlap"),
    ("ClusterONE (default)", "ClusterONE"),
    ("SLPA", "SLPA$^{a}$"),
    ("GE (reimplementation of Jeong et al.)", "GE (reimpl.)$^{b}$"),
]
ABLATIONS = [
    ("TRACE-PPI (full)", "Full (three-signal, selection)"),
    ("TRACE-PPI without deconvolution", "Without deconvolution"),
    ("TRACE-PPI topology-only heterogeneity", "Topology-only heterogeneity"),
    ("TRACE-PPI seed-and-expand only", "Seed-and-expand only (forced)"),
    ("TRACE-PPI spectral only", "Spectral only (forced)"),
    ("TRACE-PPI global admission threshold", "Global admission threshold"),
    ("TRACE-PPI without semantic/GO evidence", "Without semantic/GO evidence"),
    ("TRACE-PPI without supplementation", "Without supplementation"),
    ("TRACE-PPI with stability gate (diagnostic)", "With stability gate"),
]


def f3(x):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def f2(x):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.2f}"


def pct(x):
    return f"{100 * x:.1f}"


def intc(x):
    return f"{int(x):,}".replace(",", "{,}")


def find(b: pd.DataFrame, key: str) -> pd.Series:
    if key == "Tuned MCL":
        return b[b.method.str.startswith("Tuned MCL")].iloc[0]
    if key == "SLPA":
        return b[b.method.str.startswith("SLPA")].iloc[0]
    return b[b.method == key].iloc[0]


class Macros:
    def __init__(self):
        self.items = {}

    def __setitem__(self, k, v):
        assert re.fullmatch(r"[A-Za-z]+", k), k
        self.items[k] = v

    def write(self, path):
        path.write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(self.items.items())))


def load(ds):
    d = REV / ds
    out = {"bench": pd.read_csv(d / "benchmark.csv"), "boot": pd.read_csv(d / "bootstrap_ci.csv"),
           "enr": pd.read_csv(d / "core_enrichment.csv").iloc[0], "tiers": pd.read_csv(d / "tier_summary.csv"),
           "om": json.loads((d / "overmerging_summary.json").read_text()), "omr": pd.read_csv(d / "overmerging_regions.csv"),
           "gate": pd.read_csv(d / "stability_gate.csv"), "decision": json.loads((d / "stability_gate_decision.json").read_text()),
           "stab": pd.read_csv(d / "stability_memberships.csv"), "runs": pd.read_csv(d / "stability_runs.csv"),
           "sens": pd.read_csv(d / "sensitivity.csv") if (d / "sensitivity.csv").exists() else None,
           "grid": pd.read_csv(d / "baseline_parameter_grids.csv"), "log": json.loads((d / "run_log.json").read_text()),
           "prov": json.loads((d / "trace_provenance.json").read_text()), "ledger": pd.read_csv(d / "trace_ledger.csv"),
           "regions": pd.read_csv(d / "trace_regions.csv")}
    return out


def table(path, caption, label, colspec, header, rows, size="\\scriptsize", resize=True, notes=None):
    body = "\n".join(" & ".join(r) + r"\\" if not isinstance(r, str) else r for r in rows)
    inner = f"\\begin{{tabular}}{{{colspec}}}\n\\toprule\n{header}\\\\\n\\midrule\n{body}\n\\bottomrule\n\\end{{tabular}}"
    if resize:
        inner = f"\\resizebox{{\\textwidth}}{{!}}{{{inner}}}"
    note = f"\n\\par\\smallskip{{\\footnotesize {notes}}}" if notes else ""
    place = "H" if path.name.startswith("supp_") else "!ht"
    path.write_text(f"\\begin{{table}}[{place}]\n\\centering\n\\caption{{{caption}}}\\label{{{label}}}\n{size}\n{inner}{note}\n\\end{{table}}\n")


def display_name(m):
    """Method label as stored in the result files."""
    return m


def esc(s):
    s = "" if s is None or (isinstance(s, float) and np.isnan(s)) else str(s)
    for a, b in [("\\", "\\textbackslash{}"), ("&", "\\&"), ("%", "\\%"), ("_", "\\_"), ("#", "\\#"), ("$", "\\$"),
                 ("{", "\\{"), ("}", "\\}"), ("~", "\\textasciitilde{}"), ("^", "\\textasciicircum{}")]:
        s = s.replace(a, b)
    return s


def case_assets(out):
    cdir = REV / "case_studies"
    if not (cdir / "case_studies.json").exists():
        return
    cases = json.loads((cdir / "case_studies.json").read_text())
    cases = sorted(cases, key=lambda c: (c["case"][:2], list(DSN).index(c["dataset"])))
    alt_name = {"no_split": "no split", "spectral": "spectral", "seed_expand": "seed-and-expand"}
    rec = pd.read_csv(cdir / "case_study_records.csv")
    rows = []
    for c in cases:
        ds = DSN[c["dataset"]]
        sub = rec[(rec.dataset == c["dataset"]) & (rec.case == c["case"])]
        if "region_id" in c:
            sub = sub[sub.region_id == c["region_id"]]
            sig = f"T={c['topology']:.2f}, D={c['semantic']:.2f}, V={c['support']:.2f}; votes {c['votes']}"
            alts = {a["alternative"]: a["Q"] for a in c.get("alternatives", [])}
            q = ", ".join(f"{alt_name.get(k, k)} {v:.3f}" for k, v in alts.items()) if alts else "not evaluated (not flagged)"
            obj = f"region {c['region_id']} ({c['region_size']} proteins, {c['n_contained']} contained complexes)"
            outcome = (f"split into {len(sub)} modules" if c["split"] else "kept") + f"; Q: {q}"
            rows.append([esc(c["case"][:2]), ds, esc(obj), esc(sig), esc(outcome)])
        else:
            pid = c["protein_id"] + (f" ({c['symbol']})" if c.get("symbol") else "")
            ev = "; ".join(f"M{int(r.module_id)}: {r.status.replace('_', ' ')}, {r.tier if isinstance(r.tier, str) else '--'}, "
                           f"$\\tau$={r.tau:.2f}, $\\sigma$={r.sigma:.2f}, g={r.go:.2f}" for r in sub.itertuples())
            nsup = int((sub.reference_supported == True).sum())
            outcome = f"{len(sub)} record{'s' if len(sub) != 1 else ''}; {nsup} reference-supported"
            rows.append([esc(c["case"][:2]), ds, esc(pid), ev.replace("_", "\\_"), esc(outcome)])
    table(out / "tab_cases.tex",
          "Case studies selected by the declared rule (Section~\\ref{sec:case-rule}). T, D, V: topology, semantic and "
          "support heterogeneity; Q: internal selection objective of each alternative. Full records, matched Complex Portal "
          "entries and verified literature are in Supplementary Section~\\ref{S-sec:s-cases}.",
          "tab:cases", "ll>{\\raggedright\\arraybackslash}p{3.2cm}>{\\raggedright\\arraybackslash}p{5.4cm}>{\\raggedright\\arraybackslash}p{3.2cm}", "Rule & Data & Object & Recorded evidence & Outcome", rows,
          size="\\scriptsize", resize=False)
    parts = []
    for c in cases:
        ds = DSN[c["dataset"]]
        head = c["case"] + " (" + ds + ")"
        parts.append(f"\\paragraph{{{esc(head)}}}")
        sub = rec[(rec.dataset == c["dataset"]) & (rec.case == c["case"])]
        if "region_id" in c:
            sub = sub[sub.region_id == c["region_id"]]
            names = "; ".join(f"{esc(k)} ({esc(v)})" for k, v in c.get("reference_names", {}).items())
            parts.append(f"Region {c['region_id']}, {c['region_size']} proteins. Contained Complex Portal complexes: {names or 'none'}. "
                         f"Best region Jaccard {c['region_best_jaccard']:.2f}.")
            lines = [[str(int(r.module_id)), str(int(r.module_size)), esc(r.best_reference).replace(";", "; "), f"{r.median_tau:.2f}", f"{r.median_sigma:.2f}",
                      f"{r.median_go:.2f}", f"{r.core_fraction:.2f}", f"{r.reference_supported_fraction:.2f}",
                      "--" if pd.isna(r.median_stability) else f"{r.median_stability:.2f}"] for r in sub.itertuples()]
            hdr = "Module & n & Best Complex Portal match (Jaccard) & $\\tilde\\tau$ & $\\tilde\\sigma$ & $\\tilde g$ & Core & Sup. & Stab."
        else:
            parts.append(f"Protein {esc(c['protein_id'])}" + (f" ({esc(c['symbol'])})" if c.get("symbol") else "") + ".")
            lines = [[str(int(r.module_id)), esc(r.status.replace("_", " ")), esc(r.tier if isinstance(r.tier, str) else "--"),
                      f"{r.tau:.2f}", f"{r.sigma:.2f}", f"{r.go:.2f}", f"{r.S:.2f}", f"{r.theta:.2f}", esc(r.best_reference).replace(";", "; ")]
                     for r in sub.itertuples()]
            hdr = "Module & Status & Tier & $\\tau$ & $\\sigma$ & $g$ & $S$ & $\\theta$ & Best Complex Portal match (Jaccard)"
        cols = "rr>{\\raggedright\\arraybackslash}p{6.2cm}rrrrrr" if "region_id" in c else "rllrrrrr>{\\raggedright\\arraybackslash}p{5.5cm}"
        body = "\n".join(" & ".join(l) + r"\\" for l in lines)
        parts.append(f"\\par\\noindent{{\\scriptsize\\begin{{tabular}}{{{cols}}}\n\\toprule\n{hdr}\\\\\n\\midrule\n{body}\n\\bottomrule\n\\end{{tabular}}}}\\par\\smallskip")
        if c["case"].startswith("C5"):
            parts.append("For a review suggestion the protein is not a member of the module, so the last column lists the reference complexes that contain the protein itself.")
        lit = c.get("literature", {})
        if lit:
            parts.append("Literature cited by the matched Complex Portal entries (identity verified through Europe PMC): " +
                         "; ".join(f"PMID {p}: {esc(t)}" for p, t in lit.items()) + ".")
    (out / "supp_cases.tex").write_text("\n\n".join(parts) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    M = Macros()
    R = {ds: load(ds) for ds in DS}
    cfg = yaml.safe_load((ROOT / "configs" / "trace_ppi_frozen.yaml").read_text())

    # ------------------------------------------------------------- Table 1: data
    rows = []
    for ds in DS:
        lg = R[ds]["log"]
        meta, ref = lg["dataset_meta"], lg["reference"]
        gf = meta["go_filter"]
        removed = gf.get("excluded_assigned_by_reference_curator", 0) + gf.get("excluded_cites_reference_identifier", 0)
        led = R[ds]["ledger"]
        nodes = meta["nodes"]
        shared = int(R[ds]["bench"].iloc[0].n_reference_shared_proteins)
        refname = "Complex Portal yeast (frozen 610)" if ds != "bioplex" else "Complex Portal human"
        nref = ref["n_complexes"]
        rows.append([DSN[ds], "S. cerevisiae" if ds != "bioplex" else "H. sapiens",
                     {"gavin": "TAP-MS socio-affinity", "krogan": "TAP-MS (BioGRID)", "bioplex": "AP-MS, HEK293T"}[ds],
                     intc(nodes), intc(meta["edges"]), intc(meta["annotated_nodes"]), intc(removed), refname, intc(nref), intc(shared)])
        M[f"{DSN[ds]}Nodes"] = intc(nodes)
        M[f"{DSN[ds]}Edges"] = intc(meta["edges"])
        M[f"{DSN[ds]}GOremoved"] = intc(removed)
        M[f"{DSN[ds]}RefComplexes"] = intc(nref)
    bi = R["bioplex"]["log"]["reference"]
    M["BioPlexRefMembersNotInNetwork"] = intc(bi.get("member_not_in_network", 0))
    M["BioPlexGOAssignedByCP"] = intc(R["bioplex"]["log"]["dataset_meta"]["go_filter"].get("excluded_assigned_by_reference_curator", 0))
    M["BioPlexGOCitesCPX"] = intc(R["bioplex"]["log"]["dataset_meta"]["go_filter"].get("excluded_cites_reference_identifier", 0))
    M["GavinGOCitesCPX"] = intc(R["gavin"]["log"]["dataset_meta"]["go_filter"].get("excluded_cites_reference_identifier", 0))
    table(out / "tab_datasets.tex",
          "Networks, annotation and reference complexes. GO annotations removed: annotations assigned by the Complex Portal curators or citing a Complex Portal identifier, removed before any module construction. Shared proteins: network proteins belonging to two or more reference complexes.",
          "tab:data", "lllrrrrlrr",
          "Network & Organism & Assay & Proteins & Edges & GO-annotated & GO removed & Reference & Complexes & Shared proteins",
          rows)

    # ----------------------------------------------------------- Table 2: benchmark
    rows = []
    for ds in DS:
        b, boot = R[ds]["bench"], R[ds]["boot"].set_index("method")
        rows.append(f"\\multicolumn{{10}}{{l}}{{\\textit{{{DSN[ds]}}}}}\\\\")
        for key, lab in MAIN_METHODS:
            r = find(b, key)
            bk = {"ClusterONE (default)": "ClusterONE", "GE (reimplementation of Jeong et al.)": "GE (reimplementation)"}.get(key, key)
            ci = boot.loc[bk] if bk in boot.index else None
            fci = f"{f3(r.module_f1)} [{f3(ci.f1_ci_low)}, {f3(ci.f1_ci_high)}]" if ci is not None else f3(r.module_f1)
            mci = f"{f3(r.membership_f1)} [{f3(ci.mf1_ci_low)}, {f3(ci.mf1_ci_high)}]" if ci is not None else f3(r.membership_f1)
            rows.append([lab, f3(r.module_precision), f3(r.module_recall), fci, f3(r.membership_precision),
                         f3(r.membership_recall), mci, f3(r.shared_f1), intc(r.n_modules), intc(r.overlapping_proteins)])
            tag = {"TRACE-PPI (full)": "Trace", "MCL + overlap + supplementation": "MclOvSup", "MCL (I=2.0)": "Mcl", "Tuned MCL": "TunedMcl",
                   "MCL + overlap": "MclOv", "ClusterONE (default)": "Cone", "SLPA": "Slpa",
                   "GE (reimplementation of Jeong et al.)": "Ge"}[key]
            p = DSN[ds] + tag
            M[p + "Fone"] = f3(r.module_f1)
            M[p + "P"] = f3(r.module_precision)
            M[p + "R"] = f3(r.module_recall)
            M[p + "MFone"] = f3(r.membership_f1)
            M[p + "MP"] = f3(r.membership_precision)
            M[p + "MR"] = f3(r.membership_recall)
            M[p + "SFone"] = f3(r.shared_f1)
            M[p + "Modules"] = intc(r.n_modules)
            M[p + "Overlap"] = intc(r.overlapping_proteins)
            M[p + "Cov"] = pct(r.accepted_coverage)
            if ci is not None:
                M[p + "FoneCI"] = f"{f3(ci.f1_ci_low)}--{f3(ci.f1_ci_high)}"
                M[p + "MFoneCI"] = f"{f3(ci.mf1_ci_low)}--{f3(ci.mf1_ci_high)}"
                dcol = [c for c in boot.columns if c.startswith("diff_vs_") and c.endswith("_mean")][0]
                base = dcol[len("diff_vs_"):-len("_mean")]
                M[p + "Diff"] = f"{ci[f'diff_vs_{base}_mean']:+.3f}"
                M[p + "DiffCI"] = f"{ci[f'diff_vs_{base}_ci_low']:+.3f} to {ci[f'diff_vs_{base}_ci_high']:+.3f}"
                M[p + "MDiff"] = f"{ci[f'mdiff_vs_{base}_mean']:+.3f}"
                M[p + "MDiffCI"] = f"{ci[f'mdiff_vs_{base}_ci_low']:+.3f} to {ci[f'mdiff_vs_{base}_ci_high']:+.3f}"
    for ds in DS:
        r = R[ds]["bench"][R[ds]["bench"].method.str.startswith("TRACE-PPI from tuned MCL seed")].iloc[0]
        M[DSN[ds] + "TraceTunedFone"] = f3(r.module_f1)
        M[DSN[ds] + "TraceTunedMFone"] = f3(r.membership_f1)
    sel = R["gavin"]["log"]["baseline_selection_from_gavin"]
    M["TunedInflation"] = f"{float(sel['mcl_inflation']):.1f}"
    M["TunedSlpa"] = f"{float(sel['slpa_threshold']):.2f}"
    table(out / "tab_benchmark.tex",
          "Module-level and membership-level agreement with reference complexes. Brackets: 95\\% bootstrap intervals (reference complexes resampled jointly across methods, predicted modules resampled within method, matching held fixed). Shared F1: membership F1 restricted to reference proteins in two or more complexes. Overlap: proteins in two or more predicted modules.",
          "tab:benchmark", "lrrlrrlrrr",
          "Method & Module P & Module R & Module F1 [95\\% CI] & Memb. P & Memb. R & Memb. F1 [95\\% CI] & Shared F1 & Modules & Overlap",
          rows,
          notes=f"$^{{a}}$Parameter selected by module F1 on Gavin (MCL $I={M.items['TunedInflation']}$, SLPA $t={M.items['TunedSlpa']}$) and applied unchanged to Krogan and BioPlex; the Gavin rows are therefore optimistic. ClusterONE defaults were also best on Gavin. $^{{b}}$Our reimplementation of graph-entropy clustering from the published description (not the authors' code).")

    # -------------------------------------------------- Table 3: overmerging / decomposition
    rows = []
    for ds in DS:
        om, prov = R[ds]["om"], R[ds]["prov"]
        regs = R[ds]["regions"]
        rows.append([DSN[ds], intc(om["mcl_regions_ge2"]), intc(om["eligible_regions"]), intc(om["overmerged_regions"]),
                     intc(om["merge_attributable_false_memberships"]), intc(om["flagged_regions"]), intc(om["overmerged_flagged"]),
                     intc(om["split_regions"]), f"{om['flagged_rejected_no_split']}/{om['flagged_regions']}",
                     intc(om["overmerged_separated_ge2"]), intc(om["fragmentation_errors"])])
        for k, v in om.items():
            M[DSN[ds] + "".join(w.capitalize() for w in re.split(r"[_\d]+", k) if w)] = intc(v)
        M[DSN[ds] + "MclRegionsAll"] = intc(prov["mcl"]["n_regions"])
        M[DSN[ds] + "MclSingletons"] = intc(prov["mcl"]["n_singletons"])
        M[DSN[ds] + "MclMaxRegion"] = intc(prov["mcl"]["max_region"])
        el = regs[regs.eligible]
        M[DSN[ds] + "EligibleMedianSize"] = f"{el.n.median():.0f}"
        omr = R[ds]["omr"]
        M[DSN[ds] + "OvermergedPct"] = pct(om["overmerged_regions"] / om["mcl_regions_ge2"])
        M[DSN[ds] + "OvermergedFlaggedPct"] = pct(om["overmerged_flagged"] / max(1, om["overmerged_regions"]))
        big = omr[omr["size"] >= 6]
        M[DSN[ds] + "OvermergedEligible"] = intc((big.overmerged).sum())
    table(out / "tab_overmerging.tex",
          "Overmerging in MCL regions ($I=2.0$) and the response of TRACE-PPI. A region is overmerged when it contains at least half of the network proteins of two or more reference complexes (evaluation only). Merge-attributable false memberships: region members belonging to another contained complex than the region's best-matching complex. Separated: overmerged regions whose split recovers two or more contained complexes at $J\\geq0.5$. Fragmentation errors: split regions that matched a complex at $J\\geq0.5$ before splitting but not after.",
          "tab:overmerging", "lrrrrrrrrrr",
          "Network & Regions ($\\geq2$) & Eligible ($\\geq6$) & Overmerged & Merge FPs & Flagged & Overmerged flagged & Split & Split rejected & Separated & Fragmentation",
          rows)

    # ------------------------------------------------------------ Table 4: ablations
    rows = []
    for key, lab in ABLATIONS:
        cells = [lab]
        for ds in DS:
            b = R[ds]["bench"]
            r = b[b.method.str.startswith(key)]
            if not len(r):
                cells += ["--", "--", "--"]
                continue
            r = r.iloc[0]
            cells += [f3(r.module_f1), f3(r.membership_f1), pct(r.accepted_coverage)]
            tag = re.sub(r"[^A-Za-z]", "", lab.split("(")[0].title())
            M[DSN[ds] + "Abl" + tag + "Fone"] = f3(r.module_f1)
            M[DSN[ds] + "Abl" + tag + "MFone"] = f3(r.membership_f1)
            M[DSN[ds] + "Abl" + tag + "Cov"] = pct(r.accepted_coverage)
            M[DSN[ds] + "Abl" + tag + "Modules"] = intc(r.n_modules)
        rows.append(cells)
    table(out / "tab_ablation.tex",
          "Ablations of TRACE-PPI under the frozen configuration. Forced variants split every flagged region with the named method when it yields a valid split, bypassing the internal selection objective. Cov.: proteins with at least one accepted membership, as a percentage of network proteins. The stability gate keeps memberships recovered in at least 50\\% of 20 replicates with 10\\% of edges removed.",
          "tab:ablation", "lrrrrrrrrr",
          " & \\multicolumn{3}{c}{Gavin} & \\multicolumn{3}{c}{Krogan} & \\multicolumn{3}{c}{BioPlex}\\\\\n\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}\nVariant & Mod. F1 & Memb. F1 & Cov. & Mod. F1 & Memb. F1 & Cov. & Mod. F1 & Memb. F1 & Cov.",
          rows)

    # --------------------------------------------------------------- Table 5: tiers
    rows = []
    for ds in DS:
        t = R[ds]["tiers"].set_index("tier")
        e = R[ds]["enr"]
        st = R[ds]["stab"]
        st = st[st.rate == cfg["stability"]["edge_removal_fraction"]]
        led = R[ds]["ledger"]
        acc = led[led.status == "accepted"][["protein_id", "module_id", "tier"]]
        st = st.merge(acc, on=["protein_id", "module_id"], how="left")
        sbt = st.groupby("tier").stability.mean()
        for tier in ["Core", "Inner", "Outer", "Review"]:
            if tier not in t.index:
                continue
            r = t.loc[tier]
            rows.append([DSN[ds] if tier == "Core" else "", tier if tier != "Review" else "Review suggestion", intc(r.n),
                         f3(r.reference_supported), f2(r.mean_tau), f2(r.mean_sigma), f2(r.mean_go),
                         f3(sbt.get(tier, np.nan)) if tier != "Review" else "--",
                         (f"{e.odds_ratio:.2f} [{e.or_ci_low:.2f}, {e.or_ci_high:.2f}]" if tier == "Core" else "")])
            M[DSN[ds] + tier + "N"] = intc(r.n)
            M[DSN[ds] + tier + "Sup"] = f3(r.reference_supported)
            M[DSN[ds] + tier + "Stab"] = f3(sbt.get(tier, np.nan))
        M[DSN[ds] + "OR"] = f"{e.odds_ratio:.2f}"
        M[DSN[ds] + "ORCI"] = f"{e.or_ci_low:.2f}--{e.or_ci_high:.2f}"
        M[DSN[ds] + "CoreFrac"] = f3(e.core_fraction)
        M[DSN[ds] + "NonCoreFrac"] = f3(e.noncore_fraction)
        rv = led[led.status == "review_suggestion"]
        M[DSN[ds] + "ReviewN"] = intc(len(rv))
        M[DSN[ds] + "ReviewProteins"] = intc(rv.protein_id.nunique())
        M[DSN[ds] + "ReviewSup"] = f3(rv.reference_supported.mean()) if len(rv) else "--"
        M[DSN[ds] + "MeanStab"] = f3(st.stability.mean())
    table(out / "tab_tiers.tex",
          "Evidence tiers against reference support. Supported: fraction of accepted memberships (or review suggestions) whose protein belongs to the reference complex best matched by the module ($J\\geq0.25$). Stability: mean recovery over 20 replicates with 10\\% edge removal. Odds ratio: Core versus all non-Core accepted memberships, 95\\% interval from a module-cluster bootstrap. Tiers are operational categories, not calibrated probabilities.",
          "tab:tiers", "llrrrrrrl",
          "Network & Tier & $n$ & Supported & mean $\\tau$ & mean $\\sigma$ & mean $g$ & Stability & Core vs non-Core OR [95\\% CI]",
          rows)

    # ----------------------------------------------------------- stability gate decision
    for ds in DS:
        dec = R[ds]["decision"]
        M[DSN[ds] + "GatePrecGain"] = f"{dec['precision_gain']:+.3f}"
        M[DSN[ds] + "GateFoneChange"] = f"{dec['f1_change']:+.3f}"
        M[DSN[ds] + "GateCovChange"] = f"{100 * dec['coverage_change']:+.1f}"
        st = R[ds]["stab"]
        for rate, g in st.groupby("rate"):
            M[DSN[ds] + "Stab" + {0.05: "Five", 0.1: "Ten", 0.2: "Twenty"}[round(rate, 2)]] = f3(g.stability.mean())
        runs = R[ds]["runs"]
        r10 = runs[runs.rate == 0.1]
        M[DSN[ds] + "PertSplitMean"] = f"{r10.n_split_regions.mean():.1f}"
        M[DSN[ds] + "PertFlaggedMean"] = f"{r10.n_flagged.mean():.1f}"
        sp = pd.read_csv(REV / ds / "stability_splits.csv")
        M[DSN[ds] + "SplitRecovery"] = f3(sp[sp.rate == 0.1].split_recovery.mean()) if len(sp[sp.rate == 0.1]) else "--"

    # ----------------------------------------------------------- runtime / memory
    for ds in DS:
        prov = R[ds]["prov"]
        M[DSN[ds] + "TraceSeconds"] = f"{prov['timing_seconds']['total']:.2f}"
        M[DSN[ds] + "PeakMB"] = f"{R[ds]['log']['peak_rss_mb_process']:.0f}"
    rt = ROOT / "results" / "revised" / "runtime_memory.csv"
    if rt.exists():
        t = pd.read_csv(rt)
        rows = []
        for ds in DS:
            g = t[t.dataset == ds]
            if len(g):
                rows.append([DSN[ds], f"{g.wall_seconds.mean():.2f} $\\pm$ {g.wall_seconds.std():.2f}",
                             f"{g.peak_rss_mb.mean():.0f} $\\pm$ {g.peak_rss_mb.std():.0f}", f"{g.trace_seconds.mean():.2f}",
                             f"{g.mcl_seconds.mean():.2f}", str(len(g))])
                M[DSN[ds] + "WallSeconds"] = f"{g.wall_seconds.mean():.1f}"
                M[DSN[ds] + "WallMB"] = f"{g.peak_rss_mb.mean():.0f}"
        table(out / "supp_runtime.tex",
              "End-to-end TRACE-PPI runtime and peak resident memory in fresh processes (data loading, embedding, pipeline and export; mean $\\pm$ SD). Apple M2 Pro, 32 GB, Python 3.9.6.",
              "tab:s-runtime", "lrrrrr", "Network & Wall time (s) & Peak RSS (MB) & Pipeline (s) & of which MCL (s) & Runs", rows, size="\\small", resize=False)

    # ----------------------------------------------------- Supplementary tables
    # S: frozen configuration
    def flat(d, pre=""):
        for k, v in d.items():
            if isinstance(v, dict):
                yield from flat(v, f"{pre}{k}.")
            else:
                yield f"{pre}{k}", v
    rows = [[k.replace("_", "\\_"), str(v).replace("_", "\\_").replace("%", "\\%")] for k, v in flat(cfg) if not k.startswith("meta")]
    lines = "\n".join(f"{a} & {b}\\\\" for a, b in rows)
    (out / "supp_config.tex").write_text(
        "{\\small\\begin{longtable}{p{0.48\\linewidth}p{0.46\\linewidth}}\n\\caption{Frozen TRACE-PPI configuration (\\texttt{configs/trace\\_ppi\\_frozen.yaml}, committed before any reference-based evaluation).}\\label{tab:s-config}\\\\\n\\toprule Parameter & Value\\\\\\midrule\\endfirsthead\n\\toprule Parameter & Value\\\\\\midrule\\endhead\n"
        f"{lines}\n\\bottomrule\n\\end{{longtable}}}}\n")

    # S: full benchmark
    rows = []
    for ds in DS:
        b = R[ds]["bench"]
        rows.append(f"\\multicolumn{{12}}{{l}}{{\\textit{{{DSN[ds]}}}}}\\\\")
        for r in b.itertuples():
            rows.append([display_name(r.method).replace("&", "\\&"), f3(r.module_precision), f3(r.module_recall), f3(r.module_f1), f3(r.membership_precision),
                         f3(r.membership_recall), f3(r.membership_f1), f3(r.shared_f1), intc(r.n_modules), f"{r.mean_size:.1f}", intc(r.max_size),
                         pct(r.accepted_coverage)])
    table(out / "supp_benchmark_full.tex", "All methods and variants, all metrics. Coverage: proteins with at least one accepted membership, percentage of network proteins.",
          "tab:s-full", "lrrrrrrrrrrr", "Method & Mod. P & Mod. R & Mod. F1 & Memb. P & Memb. R & Memb. F1 & Shared F1 & Modules & Mean size & Max & Cov.", rows)

    # S: baseline grids
    rows = []
    for ds in DS:
        g = R[ds]["grid"]
        rows.append(f"\\multicolumn{{7}}{{l}}{{\\textit{{{DSN[ds]}}}}}\\\\")
        for r in g.itertuples():
            ok = str(r.status) == "ok"
            rows.append([r.parameter.replace("_", " "), str(r.value), f3(r.module_f1) if ok else "failed", f3(r.membership_f1) if ok else "--",
                         intc(r.n_modules) if ok else "--", f"{r.mean_size:.1f}" if ok else "--", f"{r.runtime_seconds:.2f}" if ok else "--"])
    table(out / "supp_grids.tex", "Baseline parameter grids. Strict MCL never falls back to another algorithm. Selection used Gavin module F1 only.",
          "tab:s-grids", "llrrrrr", "Parameter & Value & Mod. F1 & Memb. F1 & Modules & Mean size & Time (s)", rows, size="\\scriptsize", resize=False)

    # S: sensitivity
    rows = []
    for ds in DS:
        s = R[ds]["sens"]
        if s is None:
            continue
        rows.append(f"\\multicolumn{{8}}{{l}}{{\\textit{{{DSN[ds]}}} (frozen: Mod. F1 {f3(find(R[ds]['bench'], 'TRACE-PPI (full)').module_f1)}, Memb. F1 {f3(find(R[ds]['bench'], 'TRACE-PPI (full)').membership_f1)})}}\\\\")
        for r in s.itertuples():
            v = r.value
            try:
                vv = json.loads(v)
                if isinstance(vv, dict):
                    v = "/".join(f"{x:.3f}".rstrip("0") for x in vv.values())
            except Exception:
                pass
            rows.append([r.parameter.replace("_", "\\_"), str(v), intc(r.flagged), intc(r.split_regions), intc(r.review_suggestions),
                         f3(r.module_f1), f3(r.membership_f1), pct(r.accepted_coverage)])
    (out / "supp_sensitivity.tex").write_text(
        "{\\scriptsize\\setlength{\\tabcolsep}{2.5pt}\\begin{longtable}{llrrrrrr}\n\\caption{One-at-a-time sensitivity analysis around the frozen configuration (reported, not used for selection). Weights are topology/semantic/GO after renormalisation.}\\label{tab:s-sensitivity}\\\\\n\\toprule Parameter & Value & Flagged & Split & Review & Mod. F1 & Memb. F1 & Cov.\\\\\\midrule\\endfirsthead\n\\toprule Parameter & Value & Flagged & Split & Review & Mod. F1 & Memb. F1 & Cov.\\\\\\midrule\\endhead\n"
        + "\n".join((" & ".join(r) + "\\\\") if not isinstance(r, str) else r for r in rows) + "\n\\bottomrule\n\\end{longtable}}\n")
    for ds in DS:
        s = R[ds]["sens"]
        if s is None:
            continue
        for par, tag in [("assignment.quantile", "Q"), ("mcl.inflation", "Inf"), ("decomposition.lambda_fragmentation", "Lf"),
                         ("heterogeneity.min_votes", "Votes")]:
            g = s[s.parameter == par]
            M[DSN[ds] + "Sens" + tag + "FoneRange"] = f"{g.module_f1.min():.3f}--{g.module_f1.max():.3f}"
            M[DSN[ds] + "Sens" + tag + "MFoneRange"] = f"{g.membership_f1.min():.3f}--{g.membership_f1.max():.3f}"
            M[DSN[ds] + "Sens" + tag + "SplitRange"] = f"{g.split_regions.min()}--{g.split_regions.max()}"
        w = s[s.parameter == "assignment.weights"]
        M[DSN[ds] + "SensWFoneRange"] = f"{w.module_f1.min():.3f}--{w.module_f1.max():.3f}"
        M[DSN[ds] + "SensWMFoneRange"] = f"{w.membership_f1.min():.3f}--{w.membership_f1.max():.3f}"
        lf0 = s[(s.parameter == "decomposition.lambda_fragmentation") & (s.value == "0.0")]
        if len(lf0):
            M[DSN[ds] + "SensLfZeroSplit"] = intc(lf0.split_regions.iloc[0])
            M[DSN[ds] + "SensLfZeroFone"] = f3(lf0.module_f1.iloc[0])
        v1 = s[(s.parameter == "heterogeneity.min_votes") & (s.value == "1")]
        if len(v1):
            M[DSN[ds] + "SensVoteOneFlagged"] = intc(v1.flagged.iloc[0])
            M[DSN[ds] + "SensVoteOneSplit"] = intc(v1.split_regions.iloc[0])
            M[DSN[ds] + "SensVoteOneFone"] = f3(v1.module_f1.iloc[0])
        g0 = s[(s.parameter == "assignment.guard_min_neighbours") & (s.value == "0")]
        if len(g0):
            M[DSN[ds] + "SensNoNbrGuardMFone"] = f3(g0.membership_f1.iloc[0])
            M[DSN[ds] + "SensNoNbrGuardMP"] = f3(g0.membership_precision.iloc[0])

    # S: stability gate grid
    rows = []
    for ds in DS:
        g = R[ds]["gate"]
        full = find(R[ds]["bench"], "TRACE-PPI (full)")
        rows.append(f"\\multicolumn{{7}}{{l}}{{\\textit{{{DSN[ds]}}} (no gate: Memb. P {f3(full.membership_precision)}, Memb. F1 {f3(full.membership_f1)}, Cov. {pct(full.accepted_coverage)})}}\\\\")
        for r in g.itertuples():
            rows.append([f"{r.rate:.2f}", f"{r.stability_threshold:.1f}", f3(r.membership_precision), f3(r.membership_recall), f3(r.membership_f1),
                         f3(r.module_f1), pct(r.accepted_coverage)])
    table(out / "supp_stability.tex",
          "Stability gate evaluated as a filter on accepted memberships (diagnostic). The predeclared adoption rule (Gavin membership precision $+0.02$ without a membership F1 loss at 10\\% removal, threshold 0.5) was not met; stability is therefore reported, not used for acceptance.",
          "tab:s-stability", "rrrrrrr", "Edge removal & Threshold & Memb. P & Memb. R & Memb. F1 & Mod. F1 & Cov.", rows, size="\\scriptsize", resize=False)

    # S: provenance
    rows = []
    sums = {}
    for ds in DS:
        sums.update(R[ds]["log"]["input_sha256"])
    files = {"gavin_ppi": ("Gavin et al. 2006 socio-affinity network", "publication supplement"),
             "biogrid": ("BioGRID tab3 (S. cerevisiae), filtered to PubMed 16554755", "MIT"),
             "sgd_gaf": ("SGD GO annotations (GAF, 2020-04-06)", "CC BY 4.0"),
             "yeast_reference": ("Complex Portal S. cerevisiae, frozen 610-complex file", "CC0"),
             "bioplex": ("BioPlex 3.0 HEK293T network (Dec 2019)", "see bioplex.hms.harvard.edu"),
             "goa_human": ("GOA human GAF (generated 2026-05-21)", "CC BY 4.0"),
             "complexportal_human": ("Complex Portal 9606.tsv (Last-Modified 2026-10-17)", "CC0")}
    for k, (desc, lic) in files.items():
        if k in sums:
            rows.append([desc, lic, "\\texttt{" + sums[k][:16] + "\\ldots}"])
    table(out / "supp_provenance.tex", "Frozen inputs (downloaded or verified 2026-11-02). Full SHA-256 checksums are in \\texttt{data\\_manifest/} and each \\texttt{run\\_log.json}.",
          "tab:s-provenance", "p{7cm}ll", "Input & Licence & SHA-256 (prefix)", rows, size="\\scriptsize", resize=False)

    # ----------------------------------------------------- Figures
    # Fig 2: overmerging scatter
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))
    for ax, ds in zip(axes, DS):
        o = R[ds]["omr"]
        o = o[o["size"] >= 2]
        rng = np.random.default_rng(0)
        jit = rng.uniform(-0.18, 0.18, len(o))
        base = ~o.flagged
        ax.scatter(o["size"][base], o.n_contained_reference[base] + jit[base.values], s=6, c="#9aa5b1", lw=0, label="not flagged")
        fl = o.flagged & ~o.split
        ax.scatter(o["size"][fl], o.n_contained_reference[fl] + jit[fl.values], s=14, facecolors="none", edgecolors="#c0392b", lw=0.8, label="flagged, kept")
        sp = o.split
        ax.scatter(o["size"][sp], o.n_contained_reference[sp] + jit[sp.values], s=26, marker="D", c="#1f6f43", lw=0, label="split")
        ax.set_xscale("log")
        ax.axhline(1.5, color="#555", lw=0.5, ls=":")
        ax.set_title(DSN[ds], fontsize=8)
        ax.set_xlabel("MCL region size (proteins)")
        if ds == "gavin":
            ax.set_ylabel("reference complexes contained")
    axes[0].legend(frameon=False, fontsize=6, loc="upper left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_overmerging.{ext}", dpi=300)
    plt.close(fig)

    # Fig 3: tiers support + stability
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5))
    tiers = ["Core", "Inner", "Outer", "Review"]
    w = 0.26
    for i, ds in enumerate(DS):
        t = R[ds]["tiers"].set_index("tier")
        vals = [t.reference_supported.get(k, np.nan) for k in tiers]
        ns = [t.n.get(k, 0) for k in tiers]
        x = np.arange(len(tiers)) + (i - 1) * w
        axes[0].bar(x, vals, w, label=DSN[ds], color=["#34495e", "#7f8c8d", "#bdc3c7"][i])
        for xx, v, n in zip(x, vals, ns):
            axes[0].text(xx, (0 if np.isnan(v) else v) + 0.01, str(int(n)), ha="center", fontsize=5, rotation=90)
    axes[0].set_xticks(range(len(tiers)))
    axes[0].set_xticklabels(["Core", "Inner", "Outer", "Review\nsuggestion"])
    axes[0].set_ylabel("reference-supported fraction")
    axes[0].legend(frameon=False, fontsize=6)
    for i, ds in enumerate(DS):
        st = R[ds]["stab"]
        st = st[st.rate == 0.1].merge(R[ds]["ledger"][R[ds]["ledger"].status == "accepted"][["protein_id", "module_id", "tier"]], on=["protein_id", "module_id"])
        data = [st[st.tier == k].stability.values for k in ["Core", "Inner"]]
        pos = np.array([0, 1]) + (i - 1) * w
        bp = axes[1].boxplot(data, positions=pos, widths=w * 0.9, patch_artist=True, showfliers=False)
        for p in bp["boxes"]:
            p.set_facecolor(["#34495e", "#7f8c8d", "#bdc3c7"][i])
        for med in bp["medians"]:
            med.set_color("white" if i == 0 else "black")
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(["Core", "Inner"])
    axes[1].set_ylabel("membership stability (10% edges removed)")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_tiers.{ext}", dpi=300)
    plt.close(fig)

    # Supplementary figure: benchmark bars (graphical duplicate of Table 2)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)
    for ax, ds in zip(axes, DS):
        b = R[ds]["bench"]
        labs, vals = [], []
        for key, lab in MAIN_METHODS:
            r = find(b, key)
            labs.append(re.sub(r"\$.*?\$", "", lab))
            vals.append([r.module_precision, r.module_recall, r.module_f1])
        vals = np.array(vals)
        y = np.arange(len(labs))
        for j, (nm, c) in enumerate(zip(["P", "R", "F1"], ["#95a5a6", "#bdc3c7", "#2c3e50"])):
            ax.barh(y + (j - 1) * 0.26, vals[:, j], 0.26, color=c, label=nm)
        ax.set_yticks(y)
        ax.set_yticklabels(labs, fontsize=6)
        ax.invert_yaxis()
        ax.set_title(DSN[ds], fontsize=8)
        ax.set_xlabel("module-level score")
    axes[0].legend(frameon=False, fontsize=6, loc="lower right")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"figS_benchmark.{ext}", dpi=300)
    plt.close(fig)

    # Supplementary figure: sensitivity
    pars = [("mcl.inflation", "MCL inflation"), ("assignment.quantile", "admission quantile"),
            ("decomposition.lambda_fragmentation", "$\\lambda_{frag}$"), ("heterogeneity.quantile", "heterogeneity quantile")]
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.1))
    ref_vals = {"mcl.inflation": 2.0, "assignment.quantile": 0.25, "decomposition.lambda_fragmentation": 0.25, "heterogeneity.quantile": 0.75}
    for ax, (p, lab) in zip(axes, pars):
        for i, ds in enumerate(DS):
            s = R[ds]["sens"]
            if s is None:
                continue
            g = s[s.parameter == p].copy()
            g["v"] = g.value.astype(float)
            full = find(R[ds]["bench"], "TRACE-PPI (full)")
            g = pd.concat([g, pd.DataFrame([{"v": ref_vals[p], "membership_f1": full.membership_f1, "module_f1": full.module_f1}])]).sort_values("v")
            ax.plot(g.v, g.module_f1, "-o", ms=2.5, lw=0.9, color=["#34495e", "#7f8c8d", "#c0392b"][i], label=DSN[ds])
            ax.plot(g.v, g.membership_f1, "--s", ms=2.0, lw=0.7, color=["#34495e", "#7f8c8d", "#c0392b"][i])
        ax.axvline(ref_vals[p], color="#999", lw=0.5, ls=":")
        ax.set_xlabel(lab)
    axes[0].set_ylabel("F1 (solid: module; dashed: membership)")
    axes[0].legend(frameon=False, fontsize=5.5)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"figS_sensitivity.{ext}", dpi=300)
    plt.close(fig)

    case_assets(out)
    M.write(out / "numbers.tex")
    print(f"wrote {len(M.items)} macros and tables/figures to {out}")


if __name__ == "__main__":
    main()
