"""Local, evidence-aware deconvolution of coarse MCL clusters.

The functions in this module deliberately operate *inside* an initial MCL
cluster.  MCL is consequently a localisation device, not the final complex
hypothesis.  All thresholds that decide whether a cluster is heterogeneous or
which boundary proteins are retained are estimated from the local or
dataset-wide score distributions; no reference complex is inspected.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, Mapping, Set

import networkx as nx
import numpy as np
import pandas as pd

from .reuse import GENERIC_GO


def _cosine(a, b) -> float:
    if a is None or b is None:
        return 0.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


def _rank01(values: Mapping[object, float]) -> Dict[object, float]:
    """Average percentile ranks, stable in the presence of tied scores."""
    if not values:
        return {}
    series = pd.Series(values, dtype=float)
    return series.rank(method="average", pct=True).to_dict()


def membership_components(
    protein: str,
    members: Iterable[str],
    graph: nx.Graph,
    go_map: Mapping[str, Set[str]],
    embeddings: Mapping[str, np.ndarray] | None = None,
) -> Dict[str, float]:
    """Return normalised topology, semantic, and GO support for ``(p, C)``."""
    community = {str(x) for x in members} - {str(protein)}
    if not community:
        return {"topology": 0.0, "semantic": 0.0, "go": 0.0}
    p = str(protein)
    neighbors = set(graph.neighbors(p)) if p in graph else set()
    # A fraction of the smaller available neighbourhood is less biased toward
    # large candidate complexes than raw internal degree.
    topology = len(neighbors & community) / max(1, min(len(neighbors), len(community)))
    p_terms = set(go_map.get(p, set())) - GENERIC_GO
    c_terms = set().union(*(set(go_map.get(x, set())) - GENERIC_GO for x in community))
    go = len(p_terms & c_terms) / len(p_terms | c_terms) if (p_terms | c_terms) else 0.0
    semantic = 0.0
    if embeddings and p in embeddings:
        vectors = [embeddings[x] for x in community if x in embeddings]
        if vectors:
            # Cosine is mapped to [0, 1], so it can be combined transparently.
            semantic = max(0.0, min(1.0, (_cosine(embeddings[p], np.mean(vectors, axis=0)) + 1.0) / 2.0))
    return {"topology": float(topology), "semantic": float(semantic), "go": float(go)}


def score_membership(
    protein: str,
    candidate_complex: Iterable[str],
    graph: nx.Graph,
    evidence: Mapping[str, object],
    weights: Mapping[str, float] | None = None,
) -> Dict[str, float]:
    """Auditable membership score M(p,C) with normalised evidence channels.

    ``weights`` defaults to equal topology and semantic support, with a small
    independent GO contribution.  The runner reports sensitivity to these
    weights rather than treating the default as a fitted biological constant.
    """
    w = {"topology": 0.45, "semantic": 0.35, "go": 0.20, **(weights or {})}
    total = sum(max(0.0, float(w[k])) for k in ("topology", "semantic", "go"))
    w = {k: max(0.0, float(w[k])) / total for k in w}
    parts = membership_components(
        str(protein), candidate_complex, graph,
        evidence.get("go", {}), evidence.get("embeddings", {}),
    )
    score = sum(w[k] * parts[k] for k in parts)
    return {**parts, "membership_score": float(score)}


def _cluster_features(
    cluster_id: int,
    cluster: Set[str],
    graph: nx.Graph,
    evidence: Mapping[str, object],
) -> Dict[str, float]:
    nodes = {str(x) for x in cluster if x in graph}
    local = graph.subgraph(nodes).copy()
    base = {"cluster_id": cluster_id, "n_nodes": len(nodes), "n_edges": local.number_of_edges(),
            "topology_modularity": 0.0, "cut_separation": 0.0,
            "semantic_dispersion": 0.0, "support_multimodality": 0.0}
    if len(nodes) < 5 or local.number_of_edges() < len(nodes) - 1:
        return base
    try:
        groups = list(nx.community.greedy_modularity_communities(local))
    except Exception:
        groups = [set(nodes)]
    if len(groups) > 1:
        try:
            base["topology_modularity"] = float(nx.community.modularity(local, groups))
        except Exception:
            pass
        cuts = []
        for g in groups:
            rest = nodes - set(g)
            if g and rest:
                # Disconnected local fragments can have zero volume on one
                # side, for which conductance is undefined rather than strong
                # evidence of a meaningful split.
                volume_g = sum(local.degree(x) for x in g)
                volume_rest = sum(local.degree(x) for x in rest)
                if min(volume_g, volume_rest) > 0:
                    cuts.append(1.0 - float(nx.algorithms.cuts.conductance(local, set(g), rest)))
        base["cut_separation"] = float(np.mean(cuts)) if cuts else 0.0
    emb = evidence.get("embeddings", {})
    vectors = [emb[x] for x in nodes if x in emb]
    if len(vectors) >= 3:
        arr = np.vstack(vectors)
        unit = arr / np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), 1e-12)
        sims = unit @ unit.T
        base["semantic_dispersion"] = float(1.0 - np.mean(sims[np.triu_indices_from(sims, 1)]))
    supports = []
    for p in nodes:
        supports.append(membership_components(p, nodes, graph, evidence.get("go", {}), emb)["topology"])
    if len(supports) >= 4:
        # Bimodality proxy: large separation between lower and upper support
        # quantiles signals a core-plus-boundary or mixed-core cluster.
        base["support_multimodality"] = float(np.quantile(supports, .75) - np.quantile(supports, .25))
    return base


def fit_heterogeneity(
    clusters: Mapping[int, Set[str]], graph: nx.Graph, evidence: Mapping[str, object],
    quantile: float = 0.75,
) -> pd.DataFrame:
    """Fit dataset-relative heterogeneity criteria for all coarse clusters.

    Three independent criteria are retained: modular structure/cut separation,
    semantic dispersion, and variation in node-level topological support.  A
    cluster is sent to deconvolution only when at least two criteria lie in the
    upper empirical tail.  The supplied quantile is recorded in the run log and
    swept by the experiment script.
    """
    rows = [_cluster_features(int(cid), set(members), graph, evidence) for cid, members in clusters.items()]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    criteria = {
        "topology": (df["topology_modularity"] + df["cut_separation"]) / 2.0,
        "semantic": df["semantic_dispersion"],
        "support": df["support_multimodality"],
    }
    for name, values in criteria.items():
        threshold = float(values.quantile(quantile))
        df[f"heterogeneity_{name}"] = values
        df[f"heterogeneity_{name}_threshold"] = threshold
        df[f"heterogeneity_{name}_flag"] = values >= threshold
    flags = [c for c in df.columns if c.endswith("_flag")]
    df["heterogeneity_votes"] = df[flags].sum(axis=1)
    df["heterogeneity_score"] = df[["heterogeneity_topology", "heterogeneity_semantic", "heterogeneity_support"]].rank(pct=True).mean(axis=1)
    df["heterogeneous"] = (df["n_nodes"] >= 5) & (df["heterogeneity_votes"] >= 2)
    return df


def detect_heterogeneity(cluster: Set[str], graph: nx.Graph, evidence: Mapping[str, object]) -> Dict[str, object]:
    """Convenience interface for a single cluster; the pipeline fits jointly."""
    result = fit_heterogeneity({0: set(cluster)}, graph, evidence)
    return result.iloc[0].to_dict() if not result.empty else {"heterogeneous": False}


def _seed_cores(local: nx.Graph, embeddings: Mapping[str, np.ndarray] | None = None) -> list[Set[str]]:
    """Density/k-core/weighted-degree seed selection with deterministic merging."""
    if local.number_of_nodes() < 3:
        return [set(local.nodes())]
    core = nx.core_number(local) if local.number_of_edges() else {n: 0 for n in local}
    degree = dict(local.degree(weight="weight"))
    density = {n: nx.clustering(local, n) for n in local}
    score = {}
    for name, values in (("core", core), ("degree", degree), ("density", density)):
        ranks = _rank01(values)
        for n in local:
            score[n] = score.get(n, 0.0) + ranks.get(n, 0.0) / 3.0
    # Local maxima are seeds.  A seed core is its densest closed ego network;
    # maximal cliques reinforce real dense cores when they are available.
    ordered = sorted(local, key=lambda n: (-score[n], str(n)))
    seeds = [n for n in ordered if all(score[n] >= score.get(v, 0.0) for v in local.neighbors(n))]
    seeds = seeds[: max(2, min(12, int(np.sqrt(local.number_of_nodes())) + 1))]
    cores = []
    for seed in seeds:
        ego = set(local.neighbors(seed)) | {seed}
        ranked = sorted(ego, key=lambda n: (-score[n], str(n)))
        candidate = {seed, *ranked[: max(2, min(len(ranked), 5))]}
        if len(candidate) >= 2:
            cores.append(candidate)
    for clique in nx.find_cliques(local):
        if len(clique) >= 3:
            cores.append(set(clique))
    return _merge_near_duplicates(cores, overlap=0.80)


def _merge_near_duplicates(sets: Iterable[Set[str]], overlap: float = 0.80) -> list[Set[str]]:
    """Merge seed expansions with predeclared reciprocal-overlap rule."""
    out: list[Set[str]] = []
    for candidate in sorted((set(x) for x in sets if len(x) >= 2), key=lambda x: (-len(x), sorted(x))):
        for i, previous in enumerate(out):
            reciprocal = min(len(candidate & previous) / len(candidate), len(candidate & previous) / len(previous))
            if reciprocal >= overlap:
                out[i] |= candidate
                break
        else:
            out.append(candidate)
    return out


def _expand_cores(
    cores: Iterable[Set[str]], local: nx.Graph, evidence: Mapping[str, object],
    weights: Mapping[str, float] | None,
) -> list[Set[str]]:
    """One evidence-aware expansion step using each candidate's own score distribution."""
    expanded = []
    all_nodes = set(map(str, local.nodes()))
    for core in cores:
        candidate = set(core)
        outside = all_nodes - candidate
        scored = [(p, score_membership(p, candidate, local, evidence, weights)) for p in outside]
        incumbent = [score_membership(p, candidate, local, evidence, weights)["membership_score"] for p in candidate]
        # Candidate-specific lower quartile avoids one global, brittle cutoff.
        threshold = float(np.quantile(incumbent, .25)) if incumbent else 0.0
        for p, s in scored:
            if s["topology"] > 0 and s["membership_score"] >= threshold:
                candidate.add(p)
        expanded.append(candidate)
    return _merge_near_duplicates(expanded, overlap=0.80)


def _spectral_cores(local: nx.Graph) -> list[Set[str]]:
    """A deterministic Fiedler-vector cut, retained only when both sides exist."""
    nodes = list(sorted(map(str, local.nodes())))
    if len(nodes) < 5 or local.number_of_edges() < len(nodes) - 1:
        return [set(nodes)]
    try:
        matrix = nx.normalized_laplacian_matrix(local, nodelist=nodes).toarray()
        values, vectors = np.linalg.eigh(matrix)
        fiedler = vectors[:, 1]
        left = {nodes[i] for i, x in enumerate(fiedler) if x <= 0}
        right = set(nodes) - left
        if min(len(left), len(right)) < 2:
            median = float(np.median(fiedler))
            left = {nodes[i] for i, x in enumerate(fiedler) if x <= median}
            right = set(nodes) - left
        return [x for x in (left, right) if len(x) >= 2] or [set(nodes)]
    except Exception:
        return [set(nodes)]


def _candidate_quality(candidates: Iterable[Set[str]], local: nx.Graph, evidence: Mapping[str, object], weights) -> float:
    vals = []
    for candidate in candidates:
        if len(candidate) < 2:
            continue
        density = nx.density(local.subgraph(candidate))
        support = np.mean([score_membership(p, candidate, local, evidence, weights)["membership_score"] for p in candidate])
        vals.append(0.5 * density + 0.5 * support)
    return float(np.mean(vals)) if vals else 0.0


def deconvolve_cluster(
    cluster: Set[str], graph: nx.Graph, evidence: Mapping[str, object],
    weights: Mapping[str, float] | None = None,
    mechanism: str = "auto",
) -> Dict[str, object]:
    """Compare seed-and-expand and local spectral deconvolution in one cluster."""
    local = graph.subgraph({str(x) for x in cluster if x in graph}).copy()
    if len(local) < 5:
        return {"method": "retained", "candidates": [set(local.nodes())], "cores": [set(local.nodes())], "quality": 0.0}
    seed_cores = _seed_cores(local, evidence.get("embeddings", {}))
    seed_candidates = _expand_cores(seed_cores, local, evidence, weights)
    spectral_cores = _spectral_cores(local)
    spectral_candidates = _expand_cores(spectral_cores, local, evidence, weights)
    alternatives = {
        "seed_expand": (seed_cores, seed_candidates),
        "spectral": (spectral_cores, spectral_candidates),
    }
    quality = {name: _candidate_quality(cands, local, evidence, weights) for name, (_, cands) in alternatives.items()}
    chosen = mechanism if mechanism in alternatives else max(quality, key=lambda x: (quality[x], x == "seed_expand"))
    cores, candidates = alternatives[chosen]
    return {
        "method": chosen, "candidates": candidates, "cores": cores,
        "quality": quality[chosen], "alternative_quality": quality,
        "n_seed_candidates": len(seed_candidates), "n_spectral_candidates": len(spectral_candidates),
    }


def assign_overlapping_memberships(
    candidates: Mapping[int, Set[str]], graph: nx.Graph, evidence: Mapping[str, object],
    candidate_meta: Mapping[int, Mapping[str, object]] | None = None,
    weights: Mapping[str, float] | None = None,
    threshold_mode: str = "adaptive",
    fixed_threshold: float = 0.35,
    shared_margin: float = 0.85,
) -> pd.DataFrame:
    """Assign one or more memberships without a forced exclusive choice.

    Adaptive thresholds are the lower quartile of each candidate's incumbent
    score distribution.  A shared role requires support above both local
    thresholds and within ``shared_margin`` of the protein's best support.
    """
    rows = []
    meta = candidate_meta or {}
    by_parent: Dict[object, list[int]] = defaultdict(list)
    for cid in candidates:
        by_parent[meta.get(cid, {}).get("coarse_cluster_id", cid)].append(cid)
    for parent, ids in by_parent.items():
        local_nodes = set().union(*(candidates[cid] for cid in ids))
        all_scores = {p: {} for p in local_nodes}
        thresholds = {}
        for cid in ids:
            inc = [score_membership(p, candidates[cid], graph, evidence, weights)["membership_score"] for p in candidates[cid]]
            thresholds[cid] = fixed_threshold if threshold_mode == "fixed" else float(np.quantile(inc, .25)) if inc else 0.0
            for p in local_nodes:
                all_scores[p][cid] = score_membership(p, candidates[cid], graph, evidence, weights)
        for p, scores in all_scores.items():
            best = max(s["membership_score"] for s in scores.values()) if scores else 0.0
            accepted = [cid for cid, s in scores.items() if p in candidates[cid] or (s["topology"] > 0 and s["membership_score"] >= thresholds[cid])]
            accepted = [cid for cid in accepted if scores[cid]["membership_score"] >= best * shared_margin or p in candidates[cid]]
            if not accepted and scores:
                accepted = [max(scores, key=lambda c: scores[c]["membership_score"])]
            for cid in accepted:
                s = scores[cid]
                core = set(meta.get(cid, {}).get("core", set()))
                role = "core" if p in core else ("shared" if len(accepted) >= 2 else "boundary")
                rows.append({
                    "community_id": cid, "protein_id": p, "membership_role": role,
                    "membership_score": s["membership_score"], "topology_score": s["topology"],
                    "semantic_score": s["semantic"], "go_score": s["go"],
                    "membership_threshold": thresholds[cid], "threshold_mode": threshold_mode,
                    "coarse_cluster_id": parent, "deconvolution_method": meta.get(cid, {}).get("method", "retained"),
                })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Tiers describe assignment support, independently of structural role.
    q_high, q_mid, q_low = df["membership_score"].quantile([.75, .50, .25]).tolist()
    df["evidence_tier"] = np.select(
        [df.membership_score >= q_high, df.membership_score >= q_mid, df.membership_score >= q_low],
        ["high", "moderate", "weak"], default="uncertain",
    )
    df["membership_type"] = df["evidence_tier"]
    return df.sort_values(["community_id", "protein_id"]).reset_index(drop=True)
