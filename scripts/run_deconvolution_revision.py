#!/usr/bin/env python3
"""Reproduce the TRACE-PPI intra-cluster deconvolution revision analyses."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "results" / "matplotlib_cache"))
sys.path.insert(0, str(ROOT))

from trace_ppi.deconvolution_pipeline import run_deconvolution_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("gavin", "krogan"), default="gavin")
    args = parser.parse_args()
    result = run_deconvolution_experiment(args.dataset)
    print(f"Completed {args.dataset}: {len(result['final_clusters'])} final complexes")
    print(f"Outputs: {ROOT / 'results'}")


if __name__ == "__main__":
    main()
