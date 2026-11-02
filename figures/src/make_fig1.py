"""TRACE-PPI Figure 1 - publication workflow figure (double column, 180 mm).

    python3 figures/src/make_fig1.py   ->  figures/Figure1_TRACE-PPI.{pdf,svg,png,tiff}
"""
from pathlib import Path
import numpy as np, networkx as nx, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import (Circle, Wedge, Polygon, FancyBboxPatch, FancyArrowPatch,
                                Rectangle, Arc)
from scipy.spatial import ConvexHull
from scipy.interpolate import splprep, splev

plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 7,
                     "pdf.fonttype": 42, "svg.fonttype": "none"})
rng = np.random.default_rng(11)
OUT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------- palette
BLUE, ORNG, TEAL, PURP = "#3B6FB6", "#E0893A", "#2A9D8F", "#8C6BB1"
RED, GREY, INK, MUTE = "#D1495B", "#9AA5B1", "#1E2933", "#5B6875"
MODS = [BLUE, ORNG, TEAL, PURP]
CARD = "#F6F8FA"

W, H = 180, 128                                   # mm canvas
fig = plt.figure(figsize=(W / 25.4, H / 25.4))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")


def mix(c, t):                                    # blend colour c toward white by t
    c = np.array(matplotlib.colors.to_rgb(c)); return tuple(c + (1 - c) * t)


def text(x, y, s, size=7, **k):
    d = dict(ha="left", va="center", color=INK, fontsize=size, zorder=20); d.update(k)
    return ax.text(x, y, s, **d)


def rbox(x, y, w, h, fc, ec="none", lw=0.6, r=2.0, z=1, **k):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                       fc=fc, ec=ec, lw=lw, zorder=z, **k)
    ax.add_patch(p); return p


def sphere(x, y, r, col, z=6):
    """Shaded protein bead."""
    ax.add_patch(Circle((x, y), r, fc=mix(col, -0.0), ec=mix(col, 0) if False else "white",
                        lw=0.35, zorder=z))
    ax.add_patch(Circle((x, y), r, fc=col, ec="none", zorder=z))
    for k in range(1, 6):
        t = k / 6
        ax.add_patch(Circle((x - 0.32 * r * t, y + 0.32 * r * t), r * (1 - 0.8 * t),
                            fc=mix(col, 0.12 + 0.55 * t), ec="none", alpha=0.55, zorder=z + 0.01 * k))
    ax.add_patch(Circle((x, y), r, fc="none", ec=matplotlib.colors.to_hex(np.array(
        matplotlib.colors.to_rgb(col)) * 0.72), lw=0.3, zorder=z + 0.1))


def pie(x, y, r, cols, z=7):
    """Shared protein: split bead showing dual membership."""
    n = len(cols)
    for i, c in enumerate(cols):
        ax.add_patch(Wedge((x, y), r, 90 + 360 * i / n, 90 + 360 * (i + 1) / n, fc=c,
                           ec="white", lw=0.5, zorder=z))
    ax.add_patch(Circle((x - 0.3 * r, y + 0.3 * r), 0.35 * r, fc="white", alpha=0.45,
                        ec="none", zorder=z + 0.1))
    ax.add_patch(Circle((x, y), r * 1.32, fc="none", ec=RED, lw=0.7, zorder=z + 0.2))


def helix(x0, y0, L, col, amp=1.6, turns=3.0, lw=2.2, z=12):
    """Cartoon alpha-helix ribbon."""
    t = np.linspace(0, 1, 160)
    xs = x0 + L * t; ys = y0 + amp * np.sin(2 * np.pi * turns * t)
    depth = np.cos(2 * np.pi * turns * t)
    for i in range(len(t) - 1):
        d = depth[i]
        c = mix(col, 0.35) if d < 0 else col
        ax.plot(xs[i:i + 2], ys[i:i + 2], color=c, lw=lw * (0.45 + 0.55 * abs(d)),
                solid_capstyle="round", zorder=z + (0.5 if d > 0 else 0))


def blob(points, pad, fc, ec, alpha=0.18, lw=0.8, ls="-", z=3):
    """Smooth hull around a set of points."""
    ang = np.linspace(0, 2 * np.pi, 14, endpoint=False)
    P = np.vstack([p + pad * np.c_[np.cos(ang), np.sin(ang)] for p in points])
    hull = P[ConvexHull(P).vertices]
    hull = np.vstack([hull, hull[:1]])
    tck, _ = splprep([hull[:, 0], hull[:, 1]], s=len(hull) * 0.15, per=True)
    xs, ys = splev(np.linspace(0, 1, 200), tck)
    ax.add_patch(Polygon(np.c_[xs, ys], closed=True, fc=fc, alpha=alpha, ec="none", zorder=z))
    ax.add_patch(Polygon(np.c_[xs, ys], closed=True, fc="none", ec=ec, lw=lw, ls=ls, zorder=z + 0.1))


def chevron(x, y, col="#B8C2CC"):
    ax.add_patch(Polygon([[x - 1.2, y + 3], [x + 1.6, y], [x - 1.2, y - 3], [x - 0.2, y]],
                         closed=True, fc=col, ec="none", zorder=5))


def badge(x, y, r, col, num=None):
    ax.add_patch(Circle((x, y), r, fc=col, ec="none", zorder=15))
    if num is not None:
        text(x, y - 0.05, str(num), 6.5, ha="center", color="white", fontweight="bold")


def panel_label(x, y, s, title, sub=None):
    text(x, y, s, 11, fontweight="bold")
    text(x + 5, y, title, 8.5, fontweight="bold")
    if sub: text(x + 5 + 1.62 * len(title) * 0.98, y, sub, 6.8, color=MUTE, style="italic")


# ---------------------------------------------------------------- toy interactome
centres = [(0.24, 0.73), (0.70, 0.76), (0.27, 0.25), (0.74, 0.27)]
sizes = [7, 7, 7, 6]
G = nx.Graph(); pos = {}; comm = {}; groups = []; n = 0
for ci, (s, c) in enumerate(zip(sizes, centres)):
    g = list(range(n, n + s)); groups.append(g); n += s
    sub = nx.Graph()
    sub.add_nodes_from(g)
    for a, b in zip(g, g[1:] + g[:1]): sub.add_edge(a, b)
    for i in g:
        for j in g:
            if i < j and rng.random() < 0.45: sub.add_edge(i, j)
    p = nx.spring_layout(sub, seed=ci + 3, k=0.9)
    P = np.array([p[u] for u in g]); P = (P - P.mean(0)) / np.abs(P).max() * 0.15
    for u, q in zip(g, P): pos[u] = np.array(c) + q; comm[u] = ci
    G.add_edges_from(sub.edges())
S0, S1 = n, n + 1                                     # shared proteins
pos[S0] = np.array([0.475, 0.74]); pos[S1] = np.array([0.50, 0.27])
comm[S0] = comm[S1] = -1
for u in (S0, S1): G.add_node(u)
near = lambda gr, q, k: sorted(gr, key=lambda v: np.linalg.norm(pos[v] - q))[:k]
for v in near(groups[0], pos[S0], 3) + near(groups[1], pos[S0], 3): G.add_edge(S0, v)
for v in near(groups[2], pos[S1], 3) + near(groups[3], pos[S1], 3): G.add_edge(S1, v)
G.add_edge(near(groups[0], np.array([.25, .5]), 1)[0], near(groups[2], np.array([.25, .5]), 1)[0])
G.add_edge(near(groups[1], np.array([.75, .5]), 1)[0], near(groups[3], np.array([.75, .5]), 1)[0])
G.add_edge(near(groups[0], np.array([.5, .5]), 1)[0], near(groups[3], np.array([.5, .5]), 1)[0])

P_ = np.array([pos[u] for u in pos]); lo, hi = P_.min(0), P_.max(0)
for u in pos: pos[u] = 0.05 + 0.90 * (pos[u] - lo) / (hi - lo)
region = {u: (0 if comm[u] in (0, 1) or u == S0 else 1 if comm[u] == 2 or u == S1 else 2) for u in G}
REG_COL = ["#6E7FA8", TEAL, PURP]                    # MCL regions (R1 = merged, heterogeneous)


def draw_graph(x0, y0, s, node_col, edge_col, r=1.25, shared_pie=None, fade=None):
    T = lambda u: (x0 + pos[u][0] * s, y0 + pos[u][1] * s)
    for u, v in G.edges():
        (a, b), (c, d) = T(u), T(v)
        col, lw, al = edge_col(u, v)
        if fade and (fade(u) or fade(v)): al *= 0.35
        ax.plot([a, c], [b, d], color=col, lw=lw, alpha=al, zorder=4, solid_capstyle="round")
    for u in G:
        x, y = T(u)
        if shared_pie and u in shared_pie:
            pie(x, y, r * 1.1, shared_pie[u]); continue
        col = node_col(u)
        if fade and fade(u): col = mix(col, 0.6)
        sphere(x, y, r, col)
    return T


# ================================================================ PANEL A
AY = 70                                               # bottom of snapshot cards
CW, CH = 39.5, 50
CX = [3.0, 48.0, 93.0, 137.5]
panel_label(2.5, 125.0, "a", "Module construction", "   PPI graph and GO annotations only - no reference complexes")
for x in CX: rbox(x, AY, CW, CH, CARD, "#E1E6EB", 0.5, r=2.2)
for x in CX[:3]: chevron(x + CW + 2.5, AY + CH / 2)
GS = 31; GX = lambda x: x + (CW - GS) / 2; GY = AY + 3.5

# --- 1 input --------------------------------------------------------------
x = CX[0]
T = draw_graph(GX(x), GY, GS, lambda u: GREY, lambda u, v: ("#AEB7C1", 0.45, 0.9))
# magnifier: one protein -> ribbon
zx, zy = T(groups[1][0])
cx, cy, cr = x + CW - 9.5, AY + CH - 9.5, 7.2
ax.plot([zx, cx - cr * 0.7], [zy, cy - cr * 0.7], color=MUTE, lw=0.5, ls=(0, (2, 1.2)), zorder=11)
ax.add_patch(Circle((zx, zy), 2.1, fc="none", ec=MUTE, lw=0.5, zorder=11))
ax.add_patch(Circle((cx, cy), cr, fc="white", ec=MUTE, lw=0.6, zorder=11))
helix(cx - 5.2, cy + 1.6, 10.4, BLUE, amp=1.25, turns=2.5, lw=2.0)
helix(cx - 5.2, cy - 2.6, 10.4, ORNG, amp=1.25, turns=2.5, lw=2.0)
for xx in np.linspace(cx - 3.5, cx + 3.5, 3):
    ax.plot([xx, xx], [cy - 1.1, cy + 0.2], color=MUTE, lw=0.45, ls=(0, (1, 0.8)), zorder=13)
text(x + 2, AY + CH - 3.5, "Input interactome", 7, fontweight="bold")
text(x + 2, AY + CH - 7.0, "proteins + physical\ninteractions", 5.8, color=MUTE, va="top", linespacing=1.15)
# GO chips
for i, (lab_, c) in enumerate([("BP", BLUE), ("MF", ORNG), ("CC", TEAL)]):
    rbox(x + 2.2 + i * 6.6, AY + CH - 15.0, 5.8, 2.9, mix(c, 0.82), mix(c, 0.3), 0.4, r=1.2, z=8)
    text(x + 5.1 + i * 6.6, AY + CH - 13.55, "GO:" + lab_, 4.9, ha="center", color=mix(c, -0.2) if False else c, fontweight="bold")

# --- 2 MCL ---------------------------------------------------------------
x = CX[1]
T = draw_graph(GX(x), GY, GS, lambda u: REG_COL[region[u]],
               lambda u, v: ((REG_COL[region[u]], 0.55, 0.8) if region[u] == region[v]
                             else ("#C3CBD3", 0.4, 0.8)))
for k in range(3):
    pts = np.array([T(u) for u in G if region[u] == k])
    blob(pts, 2.6, REG_COL[k], REG_COL[k], alpha=0.13, lw=0.6, ls=(0, (2.5, 1.5)))
text(x + 2, AY + CH - 3.5, "Coarse localisation", 7, fontweight="bold")
text(x + 2, AY + CH - 7.0, "strict MCL (I = 2.0)\n→ search regions R", 5.8, color=MUTE, va="top", linespacing=1.15)
rx, ry = T(groups[0][0])

# --- 3 heterogeneity + decomposition ---------------------------------------
x = CX[2]
split = lambda u: BLUE if (comm[u] == 0) else ORNG if (comm[u] == 1 or u == S0) else REG_COL[region[u]]
T = draw_graph(GX(x), GY, GS, split,
               lambda u, v: ((split(u), 0.55, 0.8) if split(u) == split(v) else ("#C3CBD3", 0.4, 0.8)),
               fade=lambda u: region[u] != 0)
pts = np.array([T(u) for u in G if region[u] == 0])
blob(pts, 2.8, "#FFFFFF", RED, alpha=0.0, lw=0.8, ls=(0, (3, 1.6)))
cutx = (T(groups[0][0])[0] * 0 + np.mean([T(u)[0] for u in groups[0]]) * 0.5 + np.mean([T(u)[0] for u in groups[1]]) * 0.5) - 1.5
ys0 = [T(u)[1] for u in G if region[u] == 0]
ax.plot([cutx, cutx + 1.0], [min(ys0) - 3.5, max(ys0) + 3.5], color=RED, lw=0.9, zorder=9)
text(cutx + 1.5, min(ys0) - 2.6, "Fiedler cut", 5.0, color=RED, style="italic")
# signal gauge
gx, gy = x + 2.0, AY + CH - 7.6
for k, (s_, v) in enumerate([("T", 0.88), ("D", 0.80), ("V", 0.40)]):
    yy = gy - k * 2.3
    text(gx, yy, s_, 5.2, fontweight="bold", color=MUTE, style="italic")
    ax.add_patch(Rectangle((gx + 2, yy - 0.6), 12, 1.2, fc="#E3E8ED", ec="none", zorder=8))
    ax.add_patch(Rectangle((gx + 2, yy - 0.6), 12 * v, 1.2, fc=RED if v > .7 else "#B3BDC7", ec="none", zorder=9))
    ax.plot([gx + 2 + 12 * 0.7] * 2, [yy - 0.95, yy + 0.95], color=INK, lw=0.6, zorder=10)
text(gx + 15.5, gy - 2.3, "2 / 3 votes\n→ split", 5.4, va="center", linespacing=1.1, color=RED, fontweight="bold")
text(x + 2, AY + CH - 3.5, "Test & decompose", 7, fontweight="bold")

# --- 4 overlapping assignment ---------------------------------------------
x = CX[3]
mods = lambda u: MODS[comm[u]] if comm[u] >= 0 else RED
T = draw_graph(GX(x), GY, GS, mods,
               lambda u, v: ((mods(u), 0.55, 0.85) if comm[u] == comm[v] and comm[u] >= 0
                             else ((mods(v) if comm[u] < 0 else mods(u)), 0.55, 0.85)
                             if (comm[u] < 0 or comm[v] < 0) else ("#C3CBD3", 0.4, 0.8)),
               shared_pie={S0: [BLUE, ORNG], S1: [TEAL, PURP]})
for k in range(4):
    sh = S0 if k < 2 else S1
    pts = np.array([T(u) for u in groups[k]] + [T(sh)])
    blob(pts, 2.4, MODS[k], MODS[k], alpha=0.14, lw=0.6)
text(x + 2, AY + CH - 3.5, "Overlapping modules", 7, fontweight="bold")
text(x + 2, AY + CH - 7.0, "evidence-scored\nmemberships + tiers", 5.8, color=MUTE, va="top", linespacing=1.15)

# step numbers
for i, x in enumerate(CX):
    badge(x + CW / 2, AY, 2.3, INK, i + 1)

# evidence equation band ---------------------------------------------------
ey = 61.0
rbox(3.0, ey - 4.6, 177 - 3.0, 9.2, "white", "#D5DCE3", 0.5, r=2.0)
text(6.0, ey, "Membership evidence", 7, fontweight="bold")
text(36.5, ey, r"$S(p,C)\;=$", 8)
cx0 = 52
for lab_, w_, c, desc in [(r"$\tau$", "0.45", BLUE, "topology"), (r"$\sigma$", "0.35", ORNG, "semantic"),
                          (r"$g$", "0.20", TEAL, "GO terms")]:
    rbox(cx0, ey - 3.0, 21, 6.0, mix(c, 0.86), mix(c, 0.35), 0.4, r=1.5, z=3)
    text(cx0 + 1.6, ey, f"{w_}", 7, color=INK)
    text(cx0 + 7.2, ey + 0.1, lab_, 9, color=c, fontweight="bold")
    text(cx0 + 10.6, ey, desc, 5.8, color=MUTE)
    cx0 += 21
    if cx0 < 110: text(cx0 + 0.6, ey, "+", 8, color=MUTE); cx0 += 3.0
text(123, ey + 1.6, r"admit if $S \geq \theta_C$ (region q$_{0.25}$)", 6.0, color=INK)
text(123, ey - 1.8, "and topology evidence guard passes", 6.0, color=MUTE)

# ================================================================ PANEL B
BY = 34.0


def icon_card(x, y, w, col, title, desc, draw):
    ax.add_patch(Circle((x + 5.2, y), 4.4, fc=mix(col, 0.86), ec=mix(col, 0.35), lw=0.5, zorder=6))
    draw(x + 5.2, y, col)
    text(x + 11.5, y + 2.0, title, 6.8, fontweight="bold")
    text(x + 11.5, y - 1.0, desc, 5.6, color=MUTE, va="top", linespacing=1.15)


def ic_bars(x, y, c):
    for j, v in enumerate([0.9, 0.75, 0.35]):
        ax.add_patch(Rectangle((x - 2.2 + j * 1.6, y - 2.0), 1.1, 4.0 * v, fc=c, ec="none", zorder=8,
                               alpha=1 if v > .5 else .45))


def ic_tree(x, y, c):
    for dx in (-1.8, 1.8): ax.plot([x, x + dx], [y + 1.4, y - 1.2], color=c, lw=0.8, zorder=8)
    for p in [(x, y + 1.6), (x - 1.8, y - 1.4), (x + 1.8, y - 1.4)]:
        ax.add_patch(Circle(p, 0.85, fc=c, ec="white", lw=0.3, zorder=9))


def ic_ledger(x, y, c):
    for j in range(4):
        ax.add_patch(Rectangle((x - 2.4, y + 1.3 - j * 1.1), 4.8, 0.75, fc=c, alpha=1 - j * 0.2, ec="none", zorder=8))


def ic_stab(x, y, c):
    for j in range(3):
        ax.add_patch(Circle((x - 1.0 + j * 1.0, y), 1.9, fc="none", ec=c, lw=0.6, zorder=8))


def ic_ref(x, y, c):
    for j, (dx, dy) in enumerate([(-1.4, 1.0), (0.4, 1.4), (1.6, -0.2), (-0.6, -1.0), (0.6, -1.6)]):
        sphere(x + dx, y + dy, 0.85, c, z=8)


def ic_venn(x, y, c):
    ax.add_patch(Circle((x - 1.0, y), 2.0, fc=BLUE, alpha=0.35, ec="none", zorder=8))
    ax.add_patch(Circle((x + 1.0, y), 2.0, fc=c, alpha=0.45, ec="none", zorder=8))


def ic_merge(x, y, c):
    ax.add_patch(Circle((x, y), 2.7, fc="none", ec=c, lw=0.6, ls=(0, (1.5, 1)), zorder=8))
    sphere(x - 1.0, y + 0.5, 0.8, c, z=9); sphere(x + 1.0, y - 0.6, 0.8, mix(c, 0.4), z=9)


def ic_dist(x, y, c):
    xs = np.linspace(-2.8, 2.8, 60)
    ax.fill_between(x + xs, y - 1.8, y - 1.8 + 3.6 * np.exp(-xs ** 2 / 1.4), color=c, alpha=0.6, lw=0, zorder=8)
    ax.plot([x - 1.6, x + 1.6], [y - 2.3] * 2, color=INK, lw=0.7, zorder=9)


panel_label(2.5, 50.5, "b", "Diagnostics exported with every run")
D = "#D08A3C"
for k, (t, d, f) in enumerate([("Region record", "signals, thresholds\nand votes per region", ic_bars),
                               ("Split audit", "objective terms of each\nalternative + reason", ic_tree),
                               ("Membership ledger", "τ, σ, g, S, guard, tier\nand provenance", ic_ledger),
                               ("Perturbation stability", "10% edge removal × 20,\nfull reruns", ic_stab)]):
    icon_card(3.0 + k * 44.5, 41.5, 40, D, t, d, f)

# ================================================================ firewall
fy = 31.0
ax.plot([3, 177], [fy, fy], color=RED, lw=0.8, ls=(0, (5, 2.5)), zorder=5)
rbox(58, fy - 2.8, 64, 5.6, "white", RED, 0.6, r=2.8, z=6)
# padlock
lx, ly = 63.0, fy - 0.6
ax.add_patch(Rectangle((lx - 1.5, ly - 1.3), 3.0, 2.4, fc=RED, ec="none", zorder=8))
ax.add_patch(Arc((lx, ly + 1.1), 2.0, 2.4, theta1=0, theta2=180, ec=RED, lw=0.8, zorder=8))
text(65.6, fy, "Reference complexes never flow upward into a or b", 6.3, color=RED, fontweight="bold")

# ================================================================ PANEL C
panel_label(2.5, 23.5, "c", "Evaluation only", "   reference complexes enter here")
E = "#2F8F6A"
items = [("Reference", "Complex Portal,\nfrozen + checksummed", ic_ref),
         ("Agreement", "1:1 Jaccard ≥ 0.5;\nmodule P / R / F1", ic_venn),
         ("Overmerging", "regions holding\n≥ 2 complexes", ic_merge),
         ("Tier support", "Core vs non-Core\nagreement", ic_bars),
         ("Uncertainty", "paired bootstrap;\nsensitivity grids", ic_dist)]
for k, (t, d, f) in enumerate(items):
    xk = 3.0 + k * 35.4
    icon_card(xk, 12.0, 33, E, t, d, f)

# thin frame rules between panels
for yy in (55.0,):
    ax.plot([3, 177], [yy, yy], color="#E1E6EB", lw=0.6, zorder=1)

for ext in ["pdf", "svg"]:
    fig.savefig(OUT / f"Figure1_TRACE-PPI.{ext}")
fig.savefig(OUT / "Figure1_TRACE-PPI.png", dpi=600)
fig.savefig(OUT / "Figure1_TRACE-PPI.tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
