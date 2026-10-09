#!/usr/bin/env python3
"""Case-study rows from the same Gavin export as the tier and channel tables.

The previous case-study table was read from results/high_rank_rerun, whose
module ids are not the reported partition. This script reads
results/communities/trace_ppi_refined_gavin.csv and checks that the Gavin
Uncertain rows reproduce the published channel means.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "tables" / "revision" / "case_study_from_final_gavin.csv"
SRC = ROOT / "results" / "communities" / "trace_ppi_refined_gavin.csv"

# Table tab:channels, Gavin Uncertain, three decimals.
EXPECTED_UNCERTAIN = {"n": 3, "tau": 0.033, "sigma": 0.000, "go": 0.000}


def main() -> None:
    df = pd.read_csv(SRC)
    df["tier"] = df["membership_type"].astype(str).str.replace("_orbit", "", regex=False)
    sizes = df.groupby("community_id")["protein_id"].transform("size")
    reported = df[sizes >= 2].copy()

    uncertain = reported[reported["tier"] == "uncertain"]
    means = {
        "n": int(len(uncertain)),
        "tau": round(float(uncertain["topology_score"].mean()), 3),
        "sigma": round(float(uncertain["semantic_score"].mean()), 3),
        "go": round(float(uncertain["go_score"].mean()), 3),
    }
    if means != EXPECTED_UNCERTAIN:
        raise SystemExit(f"Gavin Uncertain means {means} != {EXPECTED_UNCERTAIN}")

    # Mixed evidence: one Core row (tau >= 0.35) and one Inner row with tau = 0.
    case = reported[reported["protein_id"].isin(["YBR181C", "YKR057W"])].copy()
    if case.groupby("protein_id").size().min() < 2:
        raise SystemExit("case-study proteins must have at least two memberships")
    case = case.sort_values(["protein_id", "community_id"])
    out = case[
        ["protein_id", "community_id", "tier", "topology_score", "semantic_score", "go_score"]
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print("uncertain_ok", means)
    print(out.to_string(index=False))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
