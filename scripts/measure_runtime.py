#!/usr/bin/env python3
"""End-to-end runtime and peak memory of `trace-ppi run` in fresh processes (3 per dataset).

    python3 scripts/measure_runtime.py   ->  results/revised/runtime_memory.csv
"""
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
rows = []
for ds in ["gavin", "krogan", "bioplex"]:
    for rep in range(3):
        with tempfile.TemporaryDirectory() as tmp:
            env = {**os.environ, "PYTHONHASHSEED": "0", "PYTHONPATH": str(ROOT)}
            p = subprocess.run(["/usr/bin/time", "-l", sys.executable, "-m", "trace_ppi", "run", "--dataset", ds, "--out", tmp],
                               cwd=ROOT, env=env, capture_output=True, text=True, check=True)
            rss = int(re.search(r"(\d+)\s+maximum resident set size", p.stderr).group(1))
            wall = float(re.search(r"([\d.]+)\s+real", p.stderr).group(1))
            prov = json.loads(Path(tmp, "provenance.json").read_text())
            rows.append({"dataset": ds, "replicate": rep, "wall_seconds": wall, "peak_rss_mb": rss / 1024 ** 2,
                         "trace_seconds": prov["timing_seconds"]["total"], "mcl_seconds": prov["timing_seconds"]["mcl"],
                         "load_seconds": prov["load_seconds"]})
            print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(ROOT / "results" / "revised" / "runtime_memory.csv", index=False)
