"""COSMOS-PPI v2/v3: candidates → score → select → recall-safe supplement → refine."""
from __future__ import annotations

import json
from typing import Dict, Optional, Set

import numpy as np
import pandas as pd

from .paths import RESULTS, DATA, ensure_dirs
from .graph_io import load_gavin, load_krogan, load_string, filter_go
from .evidence_profiles import build_profiles
from .semantic_embeddings import embed_profiles
from .black_hole_cores import discover_cores
from .candidate_generation import generate_candidates
from .baselines import mcl_only, mcl_overlap_heuristic
from .candidate_scoring import score_candidates, select_modules
from .community_expansion import expand_modules
from .event_horizon_v2 import refine_v2
from .recall_safe_supplementation import supplement_modules_recall_safe, SUPPLEMENT_DEFAULTS
from .metrics_utils import clusters_from_membership, jaccard, oracle_upper_bound
from .reuse import load_gold_standard_csv, precision_recall_f1_mmr
from .v2_membership import modules_to_membership, overlap_fraction


DEFAULT_PARAMS = dict(
    max_modules=400,
    semantic_k=50,
    expansion_threshold=0.12,
    max_community_size=45,
    jaccard_select=0.70,
    base_method="mcl_overlap",
    skip_expansion=True,
    preserve_all=True,
    supplement_candidates=False,
    recall_safe_supplement=False,
    apply_orbit_labels=True,
    supplement_max=15,
    supplement_min_score=0.62,
    supplement_max_jaccard=0.45,
    topo_weight=1.0,
    sem_weight=1.0,
    **SUPPLEMENT_DEFAULTS,
)


def _load_embeddings(dataset: str, profiles: pd.DataFrame):
    emb_path = RESULTS / "embeddings" / f"protein_embeddings_{dataset}.npz"
    idx_path = RESULTS / "embeddings" / f"protein_embedding_index_{dataset}.csv"
    meta_path = RESULTS / "embeddings" / f"protein_embedding_meta_{dataset}.json"
    if emb_path.exists() and idx_path.exists():
        data = np.load(emb_path)
        emb = data["embeddings"]
        pids = pd.read_csv(idx_path)["protein_id"].astype(str).tolist()
        emb_map = {pid: emb[i] for i, pid in enumerate(pids)}
        backend = "cached_unknown_backend"
        if meta_path.exists():
            try:
                backend = json.loads(meta_path.read_text()).get("embedding_backend", backend)
            except Exception:
                pass
        return emb, f"cached:{backend}", pids, emb_map
    emb, mode, pids = embed_profiles(profiles)
    np.savez_compressed(emb_path, embeddings=emb)
    pd.DataFrame({"protein_id": pids, "index": range(len(pids))}).to_csv(idx_path, index=False)
    meta_path.write_text(
        json.dumps(
            {
                "dataset": dataset,
                "embedding_backend": mode,
                "embedding_dim": int(emb.shape[1]),
                "n_proteins": int(len(pids)),
                "cache_status": "created_by_run_v2",
            },
            indent=2,
        )
    )
    emb_map = {pid: emb[i] for i, pid in enumerate(pids)}
    return emb, mode, pids, emb_map


def _select_base_modules(
    base_method: str,
    graph,
    go,
    scores: pd.DataFrame,
    cand_sets: Dict[int, Set[str]],
    max_modules: int,
    jaccard_select: float,
) -> Dict[int, Set[str]]:
    if base_method == "mcl":
        raw = mcl_only(graph)
    elif base_method == "mcl_overlap":
        raw = mcl_overlap_heuristic(graph, go)
    elif base_method == "score_select":
        raw = select_modules(scores, cand_sets, max_modules=max_modules, jaccard_thresh=jaccard_select)
    else:
        raw = mcl_overlap_heuristic(graph, go)
    return {i: {str(x) for x in mem} for i, mem in enumerate(raw.values()) if len(mem) >= 2}


def _supplement_modules_whole(
    selected: Dict[int, Set[str]],
    scores: pd.DataFrame,
    cand_sets: Dict[int, Set[str]],
    max_add: int,
    min_score: float,
    max_jaccard: float,
) -> Dict[int, Set[str]]:
    out = dict(selected)
    next_id = max(out.keys(), default=-1) + 1
    added = 0
    for _, row in scores.iterrows():
        if added >= max_add:
            break
        cid = int(row["candidate_id"])
        mem = cand_sets.get(cid, set())
        if len(mem) < 3 or len(mem) > 35:
            continue
        if float(row["candidate_score"]) < min_score:
            continue
        if any(jaccard(mem, s) >= max_jaccard for s in out.values()):
            continue
        out[next_id] = {str(x) for x in mem}
        next_id += 1
        added += 1
    return out


def run_v2(
    dataset: str = "gavin",
    params: Optional[dict] = None,
    variant: str = "full",
    write_outputs: bool = True,
    gold_subset: Optional[dict] = None,
) -> dict:
    ensure_dirs()
    p = {**DEFAULT_PARAMS, **(params or {})}
    variant_map = {
        "no_expansion": dict(skip_expansion=True, recall_safe_supplement=False),
        "with_expansion": dict(skip_expansion=False, recall_safe_supplement=False),
        "topology_only": dict(base_method="mcl_overlap", skip_expansion=True, topo_weight=1.0, sem_weight=0.0),
        "semantic_only": dict(base_method="mcl_overlap", skip_expansion=True, topo_weight=0.0, sem_weight=1.0),
        "score_select": dict(base_method="score_select", skip_expansion=True, recall_safe_supplement=False),
        "mcl_base": dict(base_method="mcl", skip_expansion=True),
        "recall_safe_supplement": dict(
            skip_expansion=True, recall_safe_supplement=True, apply_orbit_labels=False
        ),
        "recall_safe_orbits": dict(
            skip_expansion=True, recall_safe_supplement=True, apply_orbit_labels=True, preserve_all=True
        ),
        "full": dict(skip_expansion=True, recall_safe_supplement=False, apply_orbit_labels=True),
    }
    p.update(variant_map.get(variant, {}))

    if dataset == "gavin":
        graph, go = load_gavin()
    elif dataset == "krogan":
        graph, go = load_krogan()
    else:
        graph, go = load_string()
    go = filter_go(go)

    prof_path = RESULTS / "evidence_profiles" / f"protein_profiles_{dataset}.csv"
    profiles = pd.read_csv(prof_path) if prof_path.exists() else build_profiles(graph, go, dataset)
    if write_outputs and not prof_path.exists():
        profiles.to_csv(prof_path, index=False)

    emb, mode, pids, emb_map = _load_embeddings(dataset, profiles)

    core_path = RESULTS / "cores" / f"black_hole_cores_{dataset}.csv"
    if core_path.exists():
        cores = pd.read_csv(core_path)
    else:
        idx = {pid: i for i, pid in enumerate(pids)}
        cores = discover_cores(graph, profiles, idx, emb, max_cores=80)
        if write_outputs:
            cores.to_csv(core_path, index=False)
    idx = {pid: i for i, pid in enumerate(pids)}

    cand_path = RESULTS / "candidates" / f"candidate_modules_{dataset}.csv"
    if cand_path.exists():
        cand_df = pd.read_csv(cand_path)
        cand_sets = {
            int(cid): set(grp["protein_id"].astype(str))
            for cid, grp in cand_df.groupby("candidate_id")
        }
        gen_stats = {"n_candidates": len(cand_sets), "from_cache": True}
    else:
        cand_df, cand_sets, gen_stats = generate_candidates(
            graph, go, profiles, idx, emb, cores, dataset, semantic_k=p["semantic_k"]
        )
        if write_outputs:
            cand_df.to_csv(cand_path, index=False)

    score_path = RESULTS / "candidates" / f"candidate_scores_{dataset}.csv"
    scores = (
        pd.read_csv(score_path)
        if score_path.exists()
        else score_candidates(graph, cand_sets, go, profiles, emb_map, cores)
    )
    if write_outputs and not score_path.exists():
        scores.to_csv(score_path, index=False)

    gold = gold_subset if gold_subset is not None else load_gold_standard_csv(str(DATA["gold_standard"]))
    oracle = oracle_upper_bound(cand_sets, gold)

    selected = _select_base_modules(
        p["base_method"], graph, go, scores, cand_sets, p["max_modules"], p["jaccard_select"]
    )
    supplement_stats = {}
    if p.get("supplement_candidates"):
        selected = _supplement_modules_whole(
            selected,
            scores,
            cand_sets,
            p["supplement_max"],
            p["supplement_min_score"],
            p["supplement_max_jaccard"],
        )
    if p.get("recall_safe_supplement"):
        selected, supplement_stats = supplement_modules_recall_safe(
            selected, graph, go, emb_map, p
        )

    if p.get("skip_expansion"):
        expanded = modules_to_membership(
            selected,
            graph,
            go,
            emb_map,
            topo_weight=p["topo_weight"],
            sem_weight=p["sem_weight"],
        )
    else:
        expanded = expand_modules(
            graph,
            selected,
            go,
            emb_map,
            expansion_threshold=p["expansion_threshold"],
            max_size=p["max_community_size"],
        )

    out_tag = "v3" if p.get("recall_safe_supplement") else "v2"
    if write_outputs:
        expanded.to_csv(RESULTS / "communities" / f"cosmos_{out_tag}_expanded_{dataset}.csv", index=False)

    if p.get("apply_orbit_labels", True):
        refined = refine_v2(expanded, p)
    else:
        refined = expanded.copy()

    if write_outputs:
        refined.to_csv(RESULTS / "communities" / f"cosmos_{out_tag}_refined_{dataset}.csv", index=False)

    clusters = clusters_from_membership(refined)
    metrics = precision_recall_f1_mmr(clusters, gold)
    ovl = overlap_fraction(refined)

    return dict(
        dataset=dataset,
        variant=variant,
        embedding_mode=mode,
        gen_stats=gen_stats,
        oracle=oracle,
        n_candidates=len(cand_sets),
        n_selected=len(selected),
        n_assigned=refined["protein_id"].nunique(),
        overlap_fraction=ovl,
        supplement_stats=supplement_stats,
        metrics=metrics,
        clusters=clusters,
        refined=refined,
        params=p,
        graph=graph,
        go=go,
    )
