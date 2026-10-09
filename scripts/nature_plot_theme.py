"""Scientific Reports / Nature-style plotting defaults for TRACE-PPI figures."""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
REV_FIG = FIG / "revision"
TABLES = ROOT / "tables"
RESULTS = ROOT / "results"

METHOD_COLORS = {
    "TRACE-PPI": "#0072B2",
    "MCL": "#4D4D4D",
    "MCL+overlap": "#009E73",
    "ClusterONE": "#D55E00",
    "SLPA": "#CC79A7",
    "SLPA (t=0.05)": "#CC79A7",
    "SLPA best grid": "#CC79A7",
    "Score-select": "#E69F00",
    "Naive expansion": "#999999",
    "Core-only": "#56B4E9",
}

LABEL_COLORS = {
    "Core": "#0072B2",
    "Inner": "#56B4E9",
    "Outer": "#E69F00",
    "Uncertain": "#999999",
    "core": "#0072B2",
    "inner": "#56B4E9",
    "outer": "#E69F00",
    "uncertain": "#999999",
}

CHANNEL_COLORS = {
    "topology": "#4D4D4D",
    "semantic": "#0072B2",
    "GO": "#009E73",
}

INK = "#222222"
MUTED = "#555555"
GRID = "#E6E6E6"


def method_color(name: str) -> str:
    key = str(name).strip()
    if key in METHOD_COLORS:
        return METHOD_COLORS[key]
    if key.lower().startswith("slpa"):
        return METHOD_COLORS["SLPA"]
    return "#666666"


def apply_nature_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "axes.grid": False,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def light_ygrid(ax) -> None:
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def light_xgrid(ax) -> None:
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
