#!/usr/bin/env python3
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from trace_ppi.paths import RESULTS, REPORTS, ensure_dirs
from trace_ppi.semantic_embeddings import embed_profiles

def main():
    ensure_dirs()
    lines = ["# Semantic embedding report\n"]
    for ds in ("gavin", "krogan", "string"):
        p = RESULTS / "evidence_profiles" / f"protein_profiles_{ds}.csv"
        if not p.exists():
            continue
        prof = pd.read_csv(p)
        emb, mode, pids = embed_profiles(prof)
        np.savez_compressed(RESULTS / "embeddings" / f"protein_embeddings_{ds}.npz", embeddings=emb)
        pd.DataFrame({"protein_id": pids, "index": range(len(pids))}).to_csv(
            RESULTS / "embeddings" / f"protein_embedding_index_{ds}.csv", index=False
        )
        meta = {
            "dataset": ds,
            "embedding_backend": mode,
            "embedding_dim": int(emb.shape[1]),
            "n_proteins": int(len(pids)),
            "cache_status": "rebuilt",
        }
        (RESULTS / "embeddings" / f"protein_embedding_meta_{ds}.json").write_text(json.dumps(meta, indent=2))
        lines.append(f"## {ds}\n- mode: {mode}\n- dim: {emb.shape[1]}\n- proteins: {len(pids)}\n")
    (REPORTS / "semantic_embedding_report.md").write_text("\n".join(lines))

if __name__ == "__main__":
    main()
