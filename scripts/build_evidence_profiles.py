#!/usr/bin/env python3
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))
from trace_ppi.paths import RESULTS, REPORTS, ensure_dirs
from trace_ppi.graph_io import load_gavin, load_krogan, load_string, filter_go
from trace_ppi.evidence_profiles import build_profiles

def audit(df, name):
    n = len(df)
    ng = (df["non_generic_go_terms"].str.len() > 0).sum()
    REPORTS.mkdir(exist_ok=True)
    with open(REPORTS / "evidence_profile_audit.md", "a") as f:
        f.write(f"\n## {name}\n- proteins: {n}\n- with non-generic GO: {ng}\n")

def main():
    ensure_dirs()
    open(REPORTS / "evidence_profile_audit.md", "w").write("# Evidence profile audit\n")
    for ds, loader in [("gavin", load_gavin), ("krogan", load_krogan)]:  # STRING optional (large)
        # , ("string", load_string)
        g, go = loader()
        go = filter_go(go)
        df = build_profiles(g, go, ds)
        df.to_csv(RESULTS / "evidence_profiles" / f"protein_profiles_{ds}.csv", index=False)
        audit(df, ds)

if __name__ == "__main__":
    main()
