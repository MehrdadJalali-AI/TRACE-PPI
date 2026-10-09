#!/usr/bin/env python3
import sys
from pathlib import Path
import pandas as pd
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))
from trace_ppi.paths import RESULTS, REPORTS, ensure_dirs
from trace_ppi.graph_io import load_gavin, load_string, filter_go
from trace_ppi.evidence_profiles import build_profiles
from trace_ppi.black_hole_cores import discover_cores

def run(ds, loader):
    g, go = loader()
    go = filter_go(go)
    prof = build_profiles(g, go, ds)
    emb = np.load(RESULTS / "embeddings" / f"protein_embeddings_{ds}.npz")["embeddings"]
    idx = pd.read_csv(RESULTS / "embeddings" / f"protein_embedding_index_{ds}.csv")
    emap = {r.protein_id: int(r.index) for r in idx.itertuples()}
    cores = discover_cores(g, prof, emap, emb)
    cores.to_csv(RESULTS / "cores" / f"black_hole_cores_{ds}.csv", index=False)
    return len(cores)

def main():
    ensure_dirs()
    n = []
    for ds, loader in [("gavin", load_gavin), ("string", load_string)]:
        n.append((ds, run(ds, loader)))
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "core_discovery_report.md").write_text("# Core discovery\n" + "\n".join(f"- {d}: {c} cores" for d, c in n))

if __name__ == "__main__":
    main()
