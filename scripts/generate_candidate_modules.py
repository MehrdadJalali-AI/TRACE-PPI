#!/usr/bin/env python3
"""Generate broad candidate modules (Step 4)."""
import sys
from pathlib import Path

COSMOS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COSMOS_ROOT))
sys.path.insert(0, str(COSMOS_ROOT.parent))

from trace_ppi.cosmos_v2_runner import run_v2

if __name__ == "__main__":
    run_v2("gavin", write_outputs=True)
