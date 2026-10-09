"""Template-based evidence bundles."""
from __future__ import annotations

import pandas as pd

from .reuse import GENERIC_GO


def build_bundles(
    refined: pd.DataFrame,
    stability: pd.DataFrame,
    cores: pd.DataFrame,
    go_map: dict,
    dataset: str,
) -> pd.DataFrame:
    core_map = cores.set_index("core_id")["core_protein"].to_dict()
    stab = {(r.protein_id, int(r.community_id)): r.membership_frequency for r in stability.itertuples()}
    rows = []
    for _, r in refined.iterrows():
        p = r["protein_id"]
        cid = int(r["community_id"])
        mtype = r.get("membership_type", "outer_orbit")
        terms = sorted((go_map.get(p, set()) - GENERIC_GO))[:5]
        n_topo = int(float(r.get("topology_score", 0)) * 10)
        s = float(r.get("semantic_score", 0))
        freq = stab.get((p, cid), 0.0)
        summary = (
            f"Protein {p} is assigned to community {cid} as {mtype} because it has weighted connectivity "
            f"support approx {n_topo} to community members, semantic similarity {s:.3f} to the community "
            f"evidence profile, shared non-generic GO support {', '.join(terms) or 'none'}, and stability "
            f"frequency {freq:.3f} across perturbation views."
        )
        rows.append(
            dict(
                dataset=dataset,
                protein_id=p,
                community_id=cid,
                membership_type=mtype,
                membership_score=float(r.get("membership_score", r.get("attraction", 0))),
                topology_score=float(r.get("topology_score", 0)),
                semantic_score=float(r.get("semantic_score", 0)),
                go_score=float(r.get("go_score", 0)),
                stability_score=freq,
                uncertainty_score=float(r.get("uncertainty_score", 0)),
                top_neighbor_support=n_topo,
                top_non_generic_go_terms=";".join(terms),
                community_core_proteins=core_map.get(cid, ""),
                evidence_summary=summary,
            )
        )
    return pd.DataFrame(rows)
