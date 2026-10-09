"""Loaders and metrics for TRACE-PPI (local src package)."""
import sys
from pathlib import Path

TRACE_ROOT = Path(__file__).resolve().parents[1]
if str(TRACE_ROOT) not in sys.path:
    sys.path.insert(0, str(TRACE_ROOT))

from src.gavin_loader import GavinLoader  # noqa: E402
from src.string_loader import STRINGLoader  # noqa: E402
from src.go_loader import GOLoader  # noqa: E402
from src.string_id_mapper import load_string_to_locus, remap_graph_nodes  # noqa: E402
from src.validation.gold_standard_metrics import (  # noqa: E402
    precision_recall_f1_mmr,
    load_gold_standard_csv,
)
from src.mcl_clustering import MCLClustering  # noqa: E402
from src.membership_overlap import apply_overlap_reassignment  # noqa: E402
from src.go_tfidf import GOTFIDF  # noqa: E402
from src.permanence import calculate_permanence_all_proteins  # noqa: E402
from src.normalization import minmax_normalize  # noqa: E402

GENERIC_GO = {"GO:0003674", "GO:0005575", "GO:0008150"}

__all__ = [
    "GavinLoader",
    "STRINGLoader",
    "GOLoader",
    "load_string_to_locus",
    "remap_graph_nodes",
    "precision_recall_f1_mmr",
    "load_gold_standard_csv",
    "MCLClustering",
    "apply_overlap_reassignment",
    "GOTFIDF",
    "calculate_permanence_all_proteins",
    "minmax_normalize",
    "GENERIC_GO",
    "TRACE_ROOT",
]
