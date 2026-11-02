"""Input loading for the three evaluated networks.

Reference complexes are loaded by a separate function and are never passed to
module construction.  GO annotations assigned by the reference curator (Complex
Portal) or citing a Complex Portal identifier are excluded before any evidence
channel is built, so the reference cannot enter the GO or semantic channels.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .config import ROOT

RAW = ROOT / "data" / "raw"
GOLD = ROOT / "data" / "gold_standards"
PATHS = {
    "gavin_ppi": RAW / "gavin2006_socioaffinities_rescaled.txt",
    "biogrid": RAW / "biogrid_scerivise.tab3.txt",
    "sgd_gaf": RAW / "GO.txt",
    "yeast_reference": GOLD / "complexportal_yeast_610.csv",
    "bioplex": RAW / "human" / "BioPlex_293T_Network_10K_Dec_2019.tsv",
    "goa_human": RAW / "human" / "goa_human.gaf.gz",
    "complexportal_human": RAW / "complexportal" / "9606.tsv",
    "complexportal_yeast": RAW / "complexportal" / "559292.tsv",   # names/literature only; members from the frozen 610 file
}
GENERIC_GO = {"GO:0003674", "GO:0005575", "GO:0008150"}
YEAST_ORF_RE = re.compile(r"^Y[A-P][LR][0-9]{3}[CW](?:-[A-Z])?$|^Q[0-9]{4}$")
KROGAN_ORF_RE = re.compile(r"^Y[A-P][LR][0-9]{3}[CW](?:-[A-Z])?$")
KROGAN_PUBMED = "PUBMED:16554755"
UNIPROT_RE = re.compile(r"^([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})")


@dataclass
class Dataset:
    name: str
    graph: nx.Graph
    go: Dict[str, Set[str]]
    embeddings: Dict[str, np.ndarray]
    labels: Dict[str, str] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    @property
    def annotated(self) -> Set[str]:
        return {p for p, t in self.go.items() if t}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------------- GO
def parse_gaf(path: Path, id_mode: str, taxid: str | None, exclude_assigned_by: Iterable[str] = ("ComplexPortal",),
              exclude_prefix: str = "CPX-", strict_not: bool = True) -> Tuple[Dict[str, Set[str]], dict]:
    """Parse a GAF file.  ``id_mode='sgd'`` indexes DB_Object_ID, symbol and ORF-like
    synonyms; ``id_mode='uniprot'`` indexes the accession only."""
    opener = gzip.open if str(path).endswith(".gz") else open
    out: Dict[str, Set[str]] = defaultdict(set)
    stats = Counter()
    excluded_by = set(exclude_assigned_by)
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("!"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            stats["rows"] += 1
            qualifier, go_id = parts[3], parts[4]
            if not go_id.startswith("GO:"):
                continue
            if taxid is not None:
                taxon = parts[12] if len(parts) > 12 else ""
                if not taxon or not any(taxid in t for t in taxon.split("|")):
                    stats["other_taxon"] += 1
                    continue
            if qualifier == "NOT" or (strict_not and "NOT" in qualifier.split("|")):
                stats["not_qualifier"] += 1
                continue
            assigned_by = parts[14] if len(parts) > 14 else ""
            if assigned_by in excluded_by:
                stats["excluded_assigned_by_reference_curator"] += 1
                continue
            if exclude_prefix and any(exclude_prefix in (parts[i] if len(parts) > i else "") for i in (5, 7, 15)):
                stats["excluded_cites_reference_identifier"] += 1
                continue
            if id_mode == "sgd":
                ids = {parts[1]}
                if parts[2]:
                    ids.add(parts[2])
                if len(parts) > 10 and parts[10]:
                    ids |= {s for s in re.split(r"[|,;\s]+", parts[10]) if YEAST_ORF_RE.match(s)}
            else:
                ids = {parts[1].split("-")[0]}
            for pid in ids:
                if pid:
                    out[pid].add(go_id)
            stats["kept"] += 1
    return dict(out), dict(stats)


# ------------------------------------------------------------------------ networks
def load_gavin_graph(path: Path = PATHS["gavin_ppi"]) -> nx.Graph:
    """Drop malformed identifiers, self-loops and duplicate edges."""
    graph = nx.Graph()
    weights = []
    with open(path) as handle:
        next(handle)
        for line in handle:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            a, b = parts[0].strip(), parts[1].strip()
            if any((not x) or x == "#NAME?" or re.search(r"\s", x) for x in (a, b)):
                continue
            try:
                w_str = parts[2].strip()
                w = float(w_str.split()[-1] if " " in w_str else w_str)
            except (ValueError, IndexError):
                continue
            weights.append(w)
            graph.add_edge(a, b, weight=w)
    lo, hi = min(weights), max(weights)
    if hi > lo:
        for u, v, d in graph.edges(data=True):
            d["weight"] = (d["weight"] - lo) / (hi - lo)
    return graph


def load_krogan_graph(path: Path = PATHS["biogrid"]) -> Tuple[nx.Graph, dict]:
    """Keep physical interactions between yeast proteins with systematic ORF identifiers."""
    graph = nx.Graph()
    stats = Counter()
    with open(path, errors="replace", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            stats["raw_rows"] += 1
            if row.get("Publication Source") != KROGAN_PUBMED:
                continue
            if row.get("Experimental System Type") != "physical":
                continue
            if row.get("Organism ID Interactor A") != "559292" or row.get("Organism ID Interactor B") != "559292":
                continue
            stats["physical_yeast_rows"] += 1
            a = (row.get("Systematic Name Interactor A") or "").strip()
            b = (row.get("Systematic Name Interactor B") or "").strip()
            if not (KROGAN_ORF_RE.match(a) and KROGAN_ORF_RE.match(b)):
                stats["invalid_identifier_rows"] += 1
                continue
            if a == b:
                stats["self_loops"] += 1
                continue
            try:
                w = float(row.get("Score", "")) if row.get("Score", "-") not in ("", "-") else 1.0
            except ValueError:
                w = 1.0
            u, v = sorted((a, b))
            if graph.has_edge(u, v):
                stats["duplicate_edges"] += 1
                graph[u][v]["weight"] = max(graph[u][v]["weight"], w)
                continue
            graph.add_edge(u, v, weight=w)
    return graph, dict(stats)


def load_bioplex_graph(path: Path = PATHS["bioplex"]) -> Tuple[nx.Graph, Dict[str, Set[str]], Dict[str, str], dict]:
    """BioPlex 3.0 HEK293T network; nodes are Entrez Gene IDs, weight = pInt."""
    df = pd.read_csv(path, sep="\t", dtype=str)
    graph = nx.Graph()
    uniprots: Dict[str, Set[str]] = defaultdict(set)
    symbols: Dict[str, str] = {}
    stats = Counter(rows=len(df))
    for row in df.itertuples(index=False):
        a, b = str(row.GeneA), str(row.GeneB)
        for gene, acc, sym in ((a, row.UniprotA, row.SymbolA), (b, row.UniprotB, row.SymbolB)):
            m = UNIPROT_RE.match(str(acc)) if isinstance(acc, str) else None
            if m:
                uniprots[gene].add(m.group(1))
            symbols.setdefault(gene, str(sym))
        if a == b:
            stats["self_loops"] += 1
            continue
        if graph.has_edge(a, b):
            stats["duplicate_edges"] += 1
            continue
        graph.add_edge(a, b, weight=float(row.pInt))
    return graph, dict(uniprots), symbols, dict(stats)


# ------------------------------------------------------------------- embeddings
def text_profile(protein: str, terms: Set[str]) -> str:
    return " ".join([protein] + sorted(terms)) if terms else protein


def tfidf_svd_embeddings(nodes: List[str], go: Dict[str, Set[str]], dim: int = 64, seed: int = 42) -> Dict[str, np.ndarray]:
    """TF-IDF plus truncated SVD embeddings. Sentence-BERT is not used."""
    texts = [text_profile(p, go.get(p, set())) for p in nodes]
    X = TfidfVectorizer(max_features=8000, ngram_range=(1, 2), min_df=1).fit_transform(texts)
    k = min(dim, max(2, X.shape[1] - 1), X.shape[0] - 1)
    emb = normalize(TruncatedSVD(n_components=k, random_state=seed).fit_transform(X)).astype(np.float32)
    return {p: emb[i] for i, p in enumerate(nodes)}


# ---------------------------------------------------------------------- datasets
def load_dataset(name: str, config: dict) -> Dataset:
    ann = config.get("annotation", {})
    excl = ann.get("exclude_assigned_by", ["ComplexPortal"])
    prefix = ann.get("exclude_reference_identifier_prefix", "CPX-")
    strict_not = ann.get("exclude_not_qualifier", True)
    meta: dict = {"dataset": name}
    labels: Dict[str, str] = {}
    if name in ("gavin", "krogan"):
        if name == "gavin":
            graph = load_gavin_graph()
            meta["inputs"] = {"ppi": str(PATHS["gavin_ppi"].name)}
        else:
            graph, stats = load_krogan_graph()
            meta["cleaning"] = stats
            meta["inputs"] = {"ppi": str(PATHS["biogrid"].name), "publication": KROGAN_PUBMED}
        raw_go, go_stats = parse_gaf(PATHS["sgd_gaf"], "sgd", "559292", excl, prefix, strict_not)
        meta["inputs"]["go"] = PATHS["sgd_gaf"].name
        nodes = set(graph.nodes())
        go = {p: set(raw_go.get(p, set())) - GENERIC_GO for p in graph.nodes()}
        meta["go_filter"] = go_stats
    elif name == "bioplex":
        graph, uniprots, labels, stats = load_bioplex_graph()
        meta["cleaning"] = stats
        raw_go, go_stats = parse_gaf(PATHS["goa_human"], "uniprot", "9606", excl, prefix, strict_not)
        go = {}
        for p in graph.nodes():
            terms = set()
            for acc in uniprots.get(p, ()):
                terms |= raw_go.get(acc, set())
            go[p] = terms - GENERIC_GO
        meta["go_filter"] = go_stats
        meta["inputs"] = {"ppi": PATHS["bioplex"].name, "go": PATHS["goa_human"].name}
        meta["uniprot_map"] = {p: sorted(v) for p, v in uniprots.items()}
    else:
        raise ValueError(name)
    graph.remove_edges_from(list(nx.selfloop_edges(graph)))
    nodes = [str(n) for n in graph.nodes()]
    emb_cfg = config.get("embedding", {})
    embeddings = tfidf_svd_embeddings(nodes, go, emb_cfg.get("dimensions", 64), emb_cfg.get("svd_random_state", 42))
    meta.update({"nodes": graph.number_of_nodes(), "edges": graph.number_of_edges(),
                 "annotated_nodes": sum(1 for p in nodes if go.get(p)),
                 "embedding_backend": "tfidf_svd_64_l2"})
    return Dataset(name=name, graph=graph, go=go, embeddings=embeddings, labels=labels, meta=meta)


# --------------------------------------------------------------------- references
def load_reference(name: str, dataset: Dataset | None = None) -> Tuple[Dict[str, Set[str]], dict]:
    """Reference complexes, used only by evaluation code."""
    if name in ("gavin", "krogan"):
        df = pd.read_csv(PATHS["yeast_reference"], dtype=str)
        ref: Dict[str, Set[str]] = {}
        for cid, pid in zip(df.cluster_id, df.protein_id):   # first-occurrence order
            ref.setdefault(cid, set()).add(str(pid))
        ref = {k: v for k, v in ref.items() if len(v) >= 2}
        return ref, {"reference": "EBI Complex Portal, S. cerevisiae (610-complex frozen file)",
                     "file": PATHS["yeast_reference"].name, "n_complexes": len(ref)}
    cp = pd.read_csv(PATHS["complexportal_human"], sep="\t", dtype=str)
    col = "Identifiers (and stoichiometry) of molecules in complex"
    raw = {r["#Complex ac"]: str(r[col]) for _, r in cp.iterrows()}

    def members(ac: str, seen: Set[str]) -> Set[str]:
        out: Set[str] = set()
        for token in raw.get(ac, "").split("|"):
            ident = re.sub(r"\(\d+\)$", "", token.strip())
            if ident.startswith("CPX-"):
                if ident not in seen:
                    out |= members(ident, seen | {ident})
                continue
            m = UNIPROT_RE.match(ident)
            if m and not ident.startswith(("CHEBI", "URS")):
                out.add(m.group(1))
        return out

    acc_to_gene: Dict[str, Set[str]] = defaultdict(set)
    if dataset is not None:
        for gene, accs in dataset.meta.get("uniprot_map", {}).items():
            for a in accs:
                acc_to_gene[a].add(gene)
    stats = Counter()
    ref: Dict[str, Set[str]] = {}
    for ac in sorted(raw):
        accs = members(ac, {ac})
        ids = set()
        for a in accs:
            genes = acc_to_gene.get(a, set())
            if len(genes) == 1:
                ids.add(next(iter(genes)))
                stats["member_mapped"] += 1
            elif len(genes) > 1:
                stats["member_ambiguous"] += 1
                ids.add(f"uniprot:{a}")
            else:
                stats["member_not_in_network"] += 1
                ids.add(f"uniprot:{a}")
        if len(ids) >= 2:
            ref[ac] = ids
        else:
            stats["complexes_lt2_proteins"] += 1
    return ref, {"reference": "EBI Complex Portal, Homo sapiens (curated, non-predicted)",
                 "file": PATHS["complexportal_human"].name, "n_complexes": len(ref), **stats}
