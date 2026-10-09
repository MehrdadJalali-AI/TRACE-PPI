"""Event horizon v2: confidence typing with minimal removal."""
from __future__ import annotations

import pandas as pd

DEFAULTS = dict(
    remove_threshold=0.18,
    inner_orbit_threshold=0.25,
    outer_orbit_threshold=0.12,
    core_percentile=70.0,
)


def refine_v2(expanded: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    p = {**DEFAULTS, **(params or {})}
    preserve = bool(p.get("preserve_all", False))
    rows = []
    for _, r in expanded.iterrows():
        topo = float(r.get("topology_score", 0))
        sem = float(r.get("semantic_score", 0))
        mtype = r.get("membership_type", "outer_orbit")
        if topo >= 0.35 and sem >= 0.25:
            mtype = "core"
        elif topo >= p["inner_orbit_threshold"] or sem >= p["inner_orbit_threshold"]:
            mtype = "inner_orbit"
        elif topo >= p["outer_orbit_threshold"] or sem >= p["outer_orbit_threshold"]:
            mtype = "outer_orbit"
        else:
            mtype = "uncertain_orbit"
        if not preserve and topo < p["remove_threshold"] and sem < p["remove_threshold"]:
            continue
        rows.append({**r.to_dict(), "membership_type": mtype})
    return pd.DataFrame(rows)
