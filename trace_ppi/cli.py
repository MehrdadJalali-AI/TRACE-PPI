"""Command-line interface.

    trace-ppi run --dataset gavin --out results/run_gavin [--config configs/trace_ppi_frozen.yaml]
    python3 -m trace_ppi run --dataset bioplex --out /tmp/bioplex

Module construction only: no reference complexes are read.  Outputs: modules.csv,
ledger.csv (accepted memberships, review suggestions, rejections), regions.csv,
split_audit.csv and provenance.json.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from .config import FROZEN_CONFIG, load_config
from .datasets import load_dataset
from .pipeline import run_trace


def main(argv=None):
    ap = argparse.ArgumentParser(prog="trace-ppi")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="build TRACE-PPI modules and the assignment ledger")
    r.add_argument("--dataset", required=True, choices=["gavin", "krogan", "bioplex"])
    r.add_argument("--config", default=str(FROZEN_CONFIG))
    r.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    cfg = load_config(a.config)
    ds = load_dataset(a.dataset, cfg)
    t1 = time.time()
    res = run_trace(ds, cfg)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"module_id": m, "protein_id": p} for m, mem in res.modules.items() for p in sorted(mem)]).to_csv(out / "modules.csv", index=False)
    res.ledger.to_csv(out / "ledger.csv", index=False)
    res.regions.to_csv(out / "regions.csv", index=False)
    res.splits.to_csv(out / "split_audit.csv", index=False)
    prov = {**res.provenance, "load_seconds": t1 - t0, "wall_seconds": time.time() - t0}
    (out / "provenance.json").write_text(json.dumps(prov, indent=2, default=str))
    print(json.dumps({"modules": len(res.modules), "accepted": int((res.ledger.status == "accepted").sum()),
                      "review_suggestions": int((res.ledger.status == "review_suggestion").sum()),
                      "wall_seconds": round(prov["wall_seconds"], 2)}))


if __name__ == "__main__":
    main()
