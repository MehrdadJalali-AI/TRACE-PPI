"""Repository paths for TRACE-PPI."""
from pathlib import Path

TRACE_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = TRACE_ROOT / "data"

DATA = {
    "gavin_ppi": DATA_DIR / "gavin2006_socioaffinities_rescaled.txt",
    "krogan_biogrid": DATA_DIR / "biogrid_scerivise.tab3.txt",
    "string_ppi": DATA_DIR / "4932.protein.links.detailed.v11.5.txt",
    "go_txt": DATA_DIR / "GO.txt",
    "string_aliases": DATA_DIR / "cache" / "4932.protein.aliases.v11.5.txt.gz",
    "gold_standard": DATA_DIR / "gold_standards" / "cyc2008_yeast.csv",
}

RESULTS = TRACE_ROOT / "results"
TABLES = TRACE_ROOT / "tables"
FIGURES = TRACE_ROOT / "figures"
REPORTS = TRACE_ROOT / "reports"
CONFIGS = TRACE_ROOT / "configs"

# Backward-compatible alias used by legacy module names
COSMOS_ROOT = TRACE_ROOT


def ensure_dirs() -> None:
    for sub in (
        "evidence_profiles",
        "embeddings",
        "cores",
        "attraction",
        "candidates",
        "communities",
        "stability",
        "evidence_bundles",
        "optimization",
        "evaluation",
        "ablation",
    ):
        (RESULTS / sub).mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
