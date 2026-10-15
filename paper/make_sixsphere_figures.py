#!/usr/bin/env python3
"""
Figures for supplementary_material_sixsphere.tex: the six-sphere's cusp fibre
and the Einstein monotile.

Every geometric claim a figure makes is asserted before it is drawn, so a
figure that renders is a figure whose picture agrees with the exact
arithmetic in pyCICY.monotile and pyCICY.sixsphere:

  ss_hat_kites       the Hat is eight whole kites of the a:b = sqrt3:1 lattice
  ss_dictionary      the A2 triangulation and the Laves substrate coincide, and
                     the Laves bonds are exactly the cover relations
  ss_torus           the substrate rolled up on R^2/B0 Z^2: 1 centre, 3 edge
                     midpoints, 2 vertices, 6 + 6 bonds per fundamental domain
  ss_funnel_classes  the data-was-forced half: the funnel at B = 5 (numbers
                     read from figures/sixsphere_facts.tex) and the four
                     GL4(Z) classes by the |p| values they can reach
  tikz_dp6_hexagon.tex   the dP6 hexagon and its gluing, from cusp_fibre()

Each figure is written as PDF (for LaTeX) and PNG (for the README).

Usage
-----
    python3 paper/make_sixsphere_figures.py             # all
    python3 paper/make_sixsphere_figures.py hat torus   # a subset
"""

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
OUT = os.path.join(HERE, "figures")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from matplotlib.path import Path

# -- style -------------------------------------------------------------------
# Categorical slots 1-3 of the reference palette; the three site kinds also
# differ in marker shape, so identity never rests on colour alone.
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#b9b8b2"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
HAT_FILL = "#d6e4f7"
KIND = {
    "centre": dict(marker="o", s=34, c=BLUE, label="hexagon centre = component ($dP_6$)"),
    "edge": dict(marker="s", s=24, c=ORANGE, label="edge midpoint = double curve"),
    "vertex": dict(marker="^", s=34, c=AQUA, label="hexagon vertex = triple point"),
}

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK2, "text.color": INK,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "savefig.dpi": 200,
})

S3 = 3 ** 0.5
A, B = S3, 1.0          # the Hat point a : b = sqrt3 : 1
E1 = np.array([3.0, S3])  # 2a(cos 30, sin 30): image of e1 in the plane
E2 = np.array([0.0, 2 * S3])  # 2a(0, 1): image of e2


def _u(deg):
    return np.array([np.cos(np.radians(deg)), np.sin(np.radians(deg))])


def _save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        p = os.path.join(OUT, "%s.%s" % (name, ext))
        fig.savefig(p, bbox_inches="tight", facecolor="white")
        print("wrote", os.path.relpath(p, os.path.dirname(HERE)))
    plt.close(fig)


# -- the substrate, numerically ----------------------------------------------

def laves(R=4):
    """Sites and bonds of the [3.4.6.4] lattice at a:b = sqrt3:1 with centres
    at i*E1 + j*E2, |i|,|j| <= R. Sites are keyed by rounded coordinates."""
    key = lambda p: (round(float(p[0]), 6) + 0.0, round(float(p[1]), 6) + 0.0)
    sites, kind, bonds = {}, {}, []

    def add(p, k):
        sites.setdefault(key(p), np.array(p, dtype=float))
        kind[key(p)] = k
        return key(p)

    for i in range(-R, R + 1):
        for j in range(-R, R + 1):
            c = i * E1 + j * E2
            kc = add(c, "centre")
            for k in range(6):
                m = c + A * _u(30 + 60 * k)
                km = add(m, "edge")
                bonds.append((kc, km))
                for dv in (-30, 30):      # the two hexagon vertices at the ends
                    v = c + 2 * _u(30 + 60 * k + dv)
                    bonds.append((km, add(v, "vertex")))
    bonds = sorted({tuple(b) for b in bonds})
    return sites, kind, bonds, key


def check_against_exact():
    """The numerical lattice agrees with the exact one in pyCICY.monotile."""
    from pyCICY import monotile as Mo
    from pyCICY.monotile import Quad
    sites, kind, bonds, key = laves()
    P = Mo.laves_patch(a=Quad(0, 1, 3), b=Quad(1, 0, 3), rings=1)
    q = lambda x: float(x.p) + float(x.q) * S3
    for (x, y), k in zip(P["sites"], P["kind"]):
        kk = key((q(x), q(y)))
        assert kk in sites and kind[kk] == k, ("site mismatch", kk, k)
    ours = {frozenset((a, b)) for a, b in bonds}
    for i, j, _ in P["bonds"]:
        a = key((q(P["sites"][i][0]), q(P["sites"][i][1])))
        b = key((q(P["sites"][j][0]), q(P["sites"][j][1])))
        assert frozenset((a, b)) in ours, ("bond mismatch", a, b)
    return len(P["sites"]), len(P["bonds"])


def draw_lattice(ax, sites, kind, bonds, window, bond_kw=None, markers=True,
                 ms_scale=1.0):
    x0, x1, y0, y1 = window
    inw = lambda p: x0 <= p[0] <= x1 and y0 <= p[1] <= y1
    bond_kw = bond_kw or dict(color=MUTED, lw=0.7, zorder=1)
    for a, b in bonds:
        if inw(sites[a]) or inw(sites[b]):
            ax.plot([sites[a][0], sites[b][0]], [sites[a][1], sites[b][1]],
                    **bond_kw)
    if markers:
        pad = 0.25
        inm = lambda p: (x0 + pad <= p[0] <= x1 - pad
                         and y0 + pad <= p[1] <= y1 - pad)
        for k, st in KIND.items():
            pts = np.array([sites[s] for s in sites if kind[s] == k and inm(sites[s])])
            if len(pts):
                ax.scatter(pts[:, 0], pts[:, 1], marker=st["marker"],
                           s=st["s"] * ms_scale, c=st["c"], zorder=4,
                           edgecolors="white", linewidths=0.6)


def legend_row(fig, ax):
    handles = [plt.Line2D([], [], ls="", marker=st["marker"], color=st["c"],
                          markersize=6, label=st["label"]) for st in KIND.values()]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, -0.02))


# -- 1. the Hat as eight kites -------------------------------------------------

HAT_HEX = [(0, 0), (-1, -1), (0, -2), (2, -2), (2, -1), (4, -2), (5, -1),
           (4, 0), (3, 0), (2, 2), (0, 3), (0, 2), (-1, 2)]


def hat_outline():
    return np.array([(x + 0.5 * y, S3 / 2 * y) for x, y in HAT_HEX])


def hat_kites(poly_pts, c0=(0.0, -2 * S3)):
    """The eight kites of the lattice (centres c0 + Z E1 + Z E2) inside the
    Hat. Asserts the count, that every kite lies in the Hat, and that the
    areas add up."""
    poly = Path(poly_pts)
    inside = lambda p, t=1e-6: poly.contains_point(p, radius=t) or \
        poly.contains_point(p, radius=-t)
    kites = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            c = np.array(c0) + i * E1 + j * E2
            for k in range(6):
                q = [c, c + A * _u(30 + 60 * k), c + 2 * _u(60 * k + 60),
                     c + A * _u(90 + 60 * k)]
                if poly.contains_point(sum(q) / 4):
                    kites.append(q)
    assert len(kites) == 8, len(kites)
    assert all(inside(p) for k in kites for p in k), "a kite leaves the Hat"
    x, y = poly_pts[:, 0], poly_pts[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    assert abs(area - 8 * A * B) < 1e-9, area       # kite area = a*b
    return kites


def figure_hat():
    check_against_exact()
    sites, kind, bonds, _ = laves()
    P = hat_outline()
    kites = hat_kites(P)
    fig, ax = plt.subplots(figsize=(6.6, 4.9))
    window = (P[:, 0].min() - 2.6, P[:, 0].max() + 2.6,
              P[:, 1].min() - 2.6, P[:, 1].max() + 2.6)
    draw_lattice(ax, sites, kind, bonds, window)
    for q in kites:
        ax.add_patch(Polygon(q, closed=True, fc=HAT_FILL, ec="white", lw=1.6,
                             zorder=2))
    ax.add_patch(Polygon(P, closed=True, fc="none", ec=INK, lw=1.8, zorder=3,
                         joinstyle="round"))
    ax.set_xlim(window[0], window[1])
    ax.set_ylim(window[2], window[3])
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("The Hat is eight kites of the Laves substrate "
                 "($a:b=\\sqrt{3}:1$)", fontsize=10, color=INK)
    legend_row(fig, ax)
    _save(fig, "ss_hat_kites")


# -- 2. A2 triangulation and its Hasse diagram ---------------------------------

def figure_dictionary():
    """(a) the cells of the A2 triangulation; (b) the Laves substrate, whose
    bonds are the cover relations vertex < edge < triangle."""
    check_against_exact()
    R = 3
    cell = {}      # rounded position -> (kind, cell id)
    cover = set()
    key = lambda p: (round(float(p[0]), 6) + 0.0, round(float(p[1]), 6) + 0.0)
    pos = lambda v: v[0] * E1 + v[1] * E2
    tris, edges = [], set()
    for x in range(-R, R + 1):
        for y in range(-R, R + 1):
            v = (x, y)
            for t in ((v, (x + 1, y), (x, y + 1)),
                      ((x + 1, y), (x, y + 1), (x + 1, y + 1))):
                tris.append(t)
                for a, b in ((0, 1), (1, 2), (0, 2)):
                    edges.add(tuple(sorted((t[a], t[b]))))
    for v in {p for t in tris for p in t}:
        cell[key(pos(v))] = "centre"
    for e in edges:
        cell[key((pos(e[0]) + pos(e[1])) / 2)] = "edge"
        for v in e:
            cover.add(frozenset((key(pos(v)), key((pos(e[0]) + pos(e[1])) / 2))))
    for t in tris:
        g = key(sum(pos(v) for v in t) / 3)
        cell[g] = "vertex"
        for a, b in ((0, 1), (1, 2), (0, 2)):
            e = tuple(sorted((t[a], t[b])))
            cover.add(frozenset((g, key((pos(e[0]) + pos(e[1])) / 2))))
    sites, kind, bonds, _ = laves(R=6)
    inner = lambda p: np.hypot(*p) < 8.0
    n = 0
    for s in sites:
        if inner(sites[s]):
            assert cell.get(s) == kind[s], ("cell/site mismatch", s)
            n += 1
    m = 0
    for a, b in bonds:
        if inner(sites[a]) and inner(sites[b]):
            assert frozenset((a, b)) in cover, ("bond is not a cover relation", a, b)
            m += 1
    print("  dictionary: %d sites and %d bonds agree" % (n, m))

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.3))
    win = (-8.2, 8.2, -7.4, 7.4)
    star = [np.array(p) for p in
            ((1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1))]
    hexagon = [2 * _u(60 * k) for k in range(6)]

    ax = axes[0]
    for t in tris:
        P = np.array([pos(v) for v in t])
        ax.add_patch(Polygon(P, closed=True, fc="none", ec=MUTED, lw=0.9,
                             zorder=1))
    ax.add_patch(Polygon(np.array(hexagon), closed=True, fc=HAT_FILL,
                         ec="none", zorder=0))
    inw = lambda p: win[0] <= p[0] <= win[1] and win[2] <= p[1] <= win[3]
    for k, st in KIND.items():
        pts = np.array([sites[s] for s in sites if kind[s] == k and inw(sites[s])])
        ax.scatter(pts[:, 0], pts[:, 1], marker=st["marker"], s=st["s"],
                   c=st["c"], zorder=4, edgecolors="white", linewidths=0.6)
    ax.set_title("(a) the $A_2$ triangulation: vertices, edges, triangles",
                 fontsize=9.5)

    ax = axes[1]
    draw_lattice(ax, sites, kind, bonds, win)
    ax.add_patch(Polygon(np.array(hexagon), closed=True, fc=HAT_FILL,
                         ec="none", zorder=0))
    ax.set_title("(b) its Hasse diagram: the Laves substrate", fontsize=9.5)
    for ax in axes:
        ax.set_xlim(win[0], win[1])
        ax.set_ylim(win[2], win[3])
        ax.set_aspect("equal")
        ax.axis("off")
    legend_row(fig, axes[0])
    _save(fig, "ss_dictionary")


# -- 3. rolled up on the torus ---------------------------------------------------

def torus_data():
    """The fundamental domain of B0 Z^2 = <E1, E2> (|det B0| = 1): one
    representative of each site and each bond orbit, in (u, v) coordinates,
    with the counts asserted against the paper's table."""
    sites, kind, bonds, key = laves(R=3)
    M = np.array([E1, E2]).T
    Minv = np.linalg.inv(M)
    uv = lambda p: Minv @ p
    frac = lambda p: tuple(np.round(uv(p) - np.floor(uv(p) + 1e-9), 6) % 1.0)
    reps = {}
    for s, p in sites.items():
        reps.setdefault((kind[s], frac(p)), []).append(p)
    count = {k: sum(1 for kk, _ in reps if kk == k) for k in KIND}
    assert count == {"centre": 1, "edge": 3, "vertex": 2}, count
    rank = {"centre": 0, "edge": 1, "vertex": 2}
    orbits = {}
    for a, b in bonds:
        a_, b_ = (a, b) if rank[kind[a]] < rank[kind[b]] else (b, a)
        d = np.round(sites[b_] - sites[a_], 6)
        o = (kind[a_], frac(sites[a_]), kind[b_], tuple(d + 0.0))
        orbits.setdefault(o, (uv(sites[a_]), uv(sites[b_]) - uv(sites[a_])))
    ce = sum(1 for o in orbits if o[0] == "centre")
    ev = sum(1 for o in orbits if o[0] == "edge")
    assert (ce, ev) == (6, 6), (ce, ev)
    return reps, orbits, Minv


def figure_torus():
    reps, orbits, Minv = torus_data()
    fig = plt.figure(figsize=(9.6, 4.6))

    # (a) the flat fundamental domain, with the identifications
    ax = fig.add_subplot(1, 2, 1)
    dom = np.array([[0, 0], E1, E1 + E2, E2])
    sites, kind, bonds, _ = laves(R=3)
    ext = (-0.7, E1[0] + 0.7, -0.7, (E1 + E2)[1] + 0.7)
    draw_lattice(ax, sites, kind, bonds, ext,
                 bond_kw=dict(color="#e4e3de", lw=0.7, zorder=1),
                 markers=False)
    ax.add_patch(Polygon(dom, closed=True, fc=HAT_FILL, ec="none", alpha=0.7,
                         zorder=0))
    ax.add_patch(Polygon(dom, closed=True, fc="none", ec=INK, lw=1.3, zorder=3))
    M = np.array([E1, E2]).T
    for s, p in sites.items():
        uvp = np.linalg.inv(M) @ p
        if -1e-6 <= uvp[0] <= 1 + 1e-6 and -1e-6 <= uvp[1] <= 1 + 1e-6:
            st = KIND[kind[s]]
            rep = (uvp[0] < 1 - 1e-6) and (uvp[1] < 1 - 1e-6)
            ax.scatter([p[0]], [p[1]], marker=st["marker"], s=st["s"] * 1.3,
                       c=st["c"] if rep else "white", edgecolors=st["c"],
                       linewidths=1.1, zorder=5)
    # identification arrows: single on the e1 sides, double on the e2 sides
    def arrow(p, q, n):
        for t in (0.42, 0.58)[:n]:
            a = p + (q - p) * (t - 0.04)
            b = p + (q - p) * (t + 0.04)
            ax.annotate("", xy=b, xytext=a, zorder=6,
                        arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.0,
                                        mutation_scale=9))
    arrow(dom[0], dom[1], 1)
    arrow(dom[3], dom[2], 1)
    arrow(dom[0], dom[3], 2)
    arrow(dom[1], dom[2], 2)
    ax.text(0.5 * (E1 + E2)[0], -0.55,
            "1 component, 3 double curves, 2 triple points",
            ha="center", va="top", fontsize=8.5, color=INK2, zorder=7,
            bbox=dict(fc="white", ec="none", pad=1.5))
    ax.set_xlim(ext[0] - 0.3, ext[1] + 0.3)
    ax.set_ylim(ext[2] - 1.2, ext[3] + 0.3)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("(a) one fundamental domain of $B_0\\mathbb{Z}^2$",
                 fontsize=9.5)

    # (b) rolled up
    ax3 = fig.add_subplot(1, 2, 2, projection="3d")
    ax3.computed_zorder = False     # draw in the zorder given, not by depth
    Rt, rt = 2.0, 0.85
    elev, azim = 52.0, -68.0
    view = np.array([np.cos(np.radians(elev)) * np.cos(np.radians(azim)),
                     np.cos(np.radians(elev)) * np.sin(np.radians(azim)),
                     np.sin(np.radians(elev))])

    def torus(u, v, lift=1.0):
        u, v = np.asarray(u) * 2 * np.pi, np.asarray(v) * 2 * np.pi
        r = rt * lift
        return np.array([(Rt + r * np.cos(v)) * np.cos(u),
                         (Rt + r * np.cos(v)) * np.sin(u),
                         r * np.sin(v)])

    def visible(u, v):
        u, v = np.asarray(u) * 2 * np.pi, np.asarray(v) * 2 * np.pi
        n = np.array([np.cos(v) * np.cos(u), np.cos(v) * np.sin(u), np.sin(v)])
        return np.tensordot(view, n, axes=1) > 0.02

    uu, vv = np.meshgrid(np.linspace(0, 1, 90), np.linspace(0, 1, 40))
    X, Y, Z = torus(uu, vv)
    ax3.plot_surface(X, Y, Z, color="#e3ebf5", alpha=1.0, linewidth=0,
                     antialiased=True, shade=False, zorder=0)
    ax3.plot_wireframe(X, Y, Z, rstride=5, cstride=9, color="#c9d4e3",
                       linewidth=0.4, zorder=1)
    for (k0, p0, k1, d), (uv0, duv) in orbits.items():
        t = np.linspace(0, 1, 40)
        pts = uv0[:, None] + duv[:, None] * t[None, :]
        P = torus(pts[0], pts[1], 1.03)
        vis = visible(pts[0], pts[1])
        P = np.where(vis[None, :], P, np.nan)
        col = BLUE if k0 == "centre" else ORANGE
        ax3.plot(P[0], P[1], P[2], color=col if k0 == "centre" else AQUA,
                 lw=2.0, zorder=3, solid_capstyle="round")
    for (k, f), ps in reps.items():
        uvv = np.array(f)
        if not visible(uvv[0], uvv[1]):
            continue
        P = torus(uvv[0], uvv[1], 1.06)
        st = KIND[k]
        ax3.scatter([P[0]], [P[1]], [P[2]], marker=st["marker"],
                    s=st["s"] * 2.0, c=st["c"], edgecolors="white",
                    linewidths=0.7, depthshade=False, zorder=6)
    ax3.view_init(elev=elev, azim=azim)
    ax3.set_box_aspect((1, 1, 0.55))
    ax3.set_axis_off()
    ax3.set_title("(b) rolled up: the cusp fibre $W$ combinatorially "
                  "($6-12+6=0$)", fontsize=9.5)
    legend_row(fig, ax)
    _save(fig, "ss_torus")


# -- 4. the funnel and the four classes ------------------------------------------

def funnel_numbers():
    """Read the stage counts at B = 5 from the generated facts file, so the
    figure cannot disagree with the paper's table."""
    path = os.path.join(OUT, "sixsphere_facts.tex")
    txt = open(path).read()
    get = lambda name: int(re.search(r"\\SS%s\}\{(\d+)\}" % name, txt).group(1))
    vals = [("candidates tried", get("Tried")),
            ("$T_1^3=T_2^4=I$", get("Orders")),
            ("unipotent cusp, $N^2=0$", get("Cusp")),
            ("coinvariants exactly $\\mathbb{Z}$", get("Coinv")),
            ("invariant closure $=\\ker\\gamma$", get("Closure")),
            ("$|\\det B_0|=1$", get("Unimod"))]
    assert [v for _, v in vals] == [87846, 49446, 272, 84, 84, 32], vals
    return vals


#: the |p| values (as min(x, 12-x)) each GL4(Z)-class of 8 can reach, and the
#: least finite |p| in the class (Proposition 2)
CLASSES = [({1, 5}, 1, "contains Alp\u00f6ge's $T_1, T_2$"),
           ({0, 6}, 6, ""), ({2, 4}, 2, ""), ({3}, 3, "")]


def figure_funnel_classes():
    vals = funnel_numbers()
    assert sorted(set().union(*[c[0] for c in CLASSES])) == list(range(7))
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.0, 3.5),
                                 gridspec_kw=dict(width_ratios=[1.25, 1]))
    n = len(vals)
    y = np.arange(n)[::-1]
    ax.barh(y, [v for _, v in vals], height=0.56, color=BLUE, zorder=2)
    ax.set_xscale("log")
    ax.set_xlim(10, 6e5)
    ax.set_yticks(y)
    ax.set_yticklabels([l for l, _ in vals], fontsize=8.5, color=INK)
    for yi, (_, v) in zip(y, vals):
        ax.text(v * 1.18, yi, "{:,}".format(v), va="center", fontsize=8.5,
                color=INK)
    ax.set_xlabel("survivors at $B=5$ (log scale)")
    ax.grid(axis="x", color="#e4e3de", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title("(a) the funnel: 87,846 candidates to 32", fontsize=9.5)

    rows = np.arange(len(CLASSES))[::-1]
    for r, (res, low, note) in zip(rows, CLASSES):
        hl = bool(note)
        for p in range(7):
            on = p in res
            bx.scatter([p], [r], s=130 if on else 34,
                       c=BLUE if (on and hl) else (INK2 if on else "#d9d8d2"),
                       marker="o", zorder=3, edgecolors="none")
        bx.text(6.75, r, "least finite $|p|=%d$" % low, va="center",
                fontsize=8, color=INK2)
        if hl:
            bx.text(-0.45, r + 0.43, note, fontsize=8, color=BLUE, va="bottom")
    bx.set_yticks(rows)
    bx.set_yticklabels(["class %s" % c for c in "ABCD"], fontsize=8.5,
                       color=INK)
    bx.set_xticks(range(7))
    bx.set_xlim(-0.6, 9.6)
    bx.set_ylim(-0.6, len(CLASSES) - 0.2)
    bx.set_xlabel("$|p|$ values the class can reach, taken mod 12")
    for s in ("top", "right", "left"):
        bx.spines[s].set_visible(False)
    bx.tick_params(axis="y", length=0)
    bx.set_title("(b) four conjugacy classes of 8; only one reaches $|p|=1$",
                 fontsize=9.5)
    fig.tight_layout(w_pad=2.0)
    _save(fig, "ss_funnel_classes")


# -- 5. the dP6 hexagon, as TikZ -------------------------------------------------

TIKZ = r"""\documentclass[tikz,border=6pt]{standalone}
\usetikzlibrary{decorations.markings,shapes.geometric,positioning}
\definecolor{cE}{HTML}{2A78D6}   % component
\definecolor{cD}{HTML}{EB6834}   % double curve
\definecolor{cT}{HTML}{1BAF7A}   % triple point
\definecolor{ink}{HTML}{52514E}
\tikzset{
  node font=\small,
  side/.style={cD,line width=1.7pt},
  lab/.style={ink,font=\footnotesize},
  comp/.style={circle,fill=cE,draw=white,thick,inner sep=3pt},
  dbl/.style={rectangle,fill=cD,draw=white,thick,inner sep=3pt},
  tp/.style={regular polygon,regular polygon sides=3,fill=cT,draw=cT,thick,inner sep=2.4pt},
  tq/.style={regular polygon,regular polygon sides=3,fill=white,draw=cT,thick,inner sep=2.4pt},
  hasse/.style={ink!70,thick},
  one/.style={decoration={markings,mark=at position 0.5 with {\arrow{>}}},postaction=decorate},
  two/.style={decoration={markings,mark=between positions 0.40 and 0.60 step 0.20 with {\arrow{>}}},postaction=decorate},
  three/.style={decoration={markings,mark=between positions 0.34 and 0.66 step 0.16 with {\arrow{>}}},postaction=decorate}}
% Generated by paper/make_sixsphere_figures.py from sixsphere.cusp_fibre():
% glued sides @GLUED@; corner classes @CORNERS@.
\begin{document}
\begin{tikzpicture}
  % left: the component E_0, six (-1)-curves; glued sides carry equal arrow counts
@SIDES@
@CORNERS_TEX@
  \node[ink] at (0,0) {$E_0\cong dP_6$};
  \node[lab,align=center] at (0,-3.9) {six $(-1)$-curves $D_i$, $D_i^2=-1$;\\
    corners are the triple points $P,Q$};
  % right: its Hasse diagram, ranks 1 / 3 / 2
  \begin{scope}[xshift=8.4cm]
    \node[comp] (E) at (0,2.4) {};  \node[lab,right=5pt of E] {component $E_0$};
@HASSE@
    \node[lab,align=center] at (0,-3.9) {$e=1-3+2=0$ on the dual complex;\\
      each double curve meets $P$ and $Q$: $(-1)+(-1)+2=0$};
  \end{scope}
\end{tikzpicture}
\end{document}
"""


def tikz_source():
    from pyCICY import sixsphere as S
    W = S.cusp_fibre()
    rays = S.HEXAGON_RAYS
    glued = W["glued_sides"]
    # the picture below draws opposite sides as the glued pair and alternates
    # the two triple points around the corners; check both against the data
    for r, s in glued:
        assert abs(rays.index(s) - rays.index(r)) == 3, (r, s)
    assert [k[0] for k in W["corner_classes"]] == ["lower", "upper"] * 3
    assert (W["components"], W["double_curves"], W["triple_points"]) == (1, 3, 2)
    style = {}
    for n, (r, s) in enumerate(glued):
        style[r] = style[s] = ("one", "two", "three")[n]
    name = {}
    for n, (r, s) in enumerate(glued):
        name[r] = name[s] = "abc"[n]
    R = 3.0
    sides, corners = [], []
    for i, r in enumerate(rays):
        p, q = R * _u(60 * i - 30), R * _u(60 * i + 30)
        if i >= 3:                    # draw the second of a pair reversed, so
            p, q = q, p               # glued arrows are parallel translates
        mid = R * np.cos(np.radians(30)) * _u(60 * i)
        out = 1.0 * R * 0.0 + (R * np.cos(np.radians(30)) + 0.62)
        sides.append("  \\draw[side,%s] (%.4f,%.4f) -- (%.4f,%.4f);"
                     % (style[r], p[0], p[1], q[0], q[1]))
        sides.append("  \\node[lab] at (%.3f,%.3f) {$D_{%s}$};"
                     % (out * np.cos(np.radians(60 * i)),
                        out * np.sin(np.radians(60 * i)), name[r]))
    for i, (cls, _) in enumerate(W["corner_classes"]):
        c = R * _u(60 * i + 30)
        st = "tp" if cls == "lower" else "tq"
        corners.append("  \\node[%s] at (%.4f,%.4f) {};" % (st, c[0], c[1]))
    hasse = []
    for x, n in zip((-2.4, 0.0, 2.4), "abc"):
        hasse.append("    \\node[dbl] (D%s) at (%.1f,0) {};" % (n, x))
        hasse.append("    \\node[lab,below=4pt of D%s] {$D_%s$};" % (n, n))
    hasse.append("    \\node[tp] (P) at (-1.2,-2.1) {};")
    hasse.append("    \\node[tq] (Q) at (1.2,-2.1) {};")
    hasse.append("    \\node[lab,left=5pt of P] {$P$};")
    hasse.append("    \\node[lab,right=5pt of Q] {$Q$};")
    for n in "abc":
        hasse.append("    \\draw[hasse] (E) -- (D%s) -- (P);" % n)
        hasse.append("    \\draw[hasse] (D%s) -- (Q);" % n)
    return (TIKZ.replace("@GLUED@", str(glued))
            .replace("@CORNERS@", str([k[0] for k in W["corner_classes"]]))
            .replace("@SIDES@", "\n".join(sides))
            .replace("@CORNERS_TEX@", "\n".join(corners))
            .replace("@HASSE@", "\n".join(hasse)))


def figure_tikz():
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "tikz_dp6_hexagon.tex")
    with open(path, "w") as fh:
        fh.write(tikz_source())
    print("wrote", os.path.relpath(path, os.path.dirname(HERE)))
    # The supplement includes the compiled PDF, and the README the PNG.
    import shutil
    import subprocess
    if shutil.which("pdflatex") is None:
        print("  pdflatex not found: compile %s by hand" % os.path.basename(path))
        return
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                    os.path.basename(path)], cwd=OUT, check=True,
                   stdout=subprocess.DEVNULL)
    for ext in ("aux", "log"):
        try:
            os.remove(os.path.join(OUT, "tikz_dp6_hexagon." + ext))
        except OSError:
            pass
    print("wrote", os.path.relpath(path[:-3] + "pdf", os.path.dirname(HERE)))
    if shutil.which("pdftoppm"):
        subprocess.run(["pdftoppm", "-png", "-r", "130", "-singlefile",
                        path[:-3] + "pdf", path[:-4]], check=True)
        print("wrote", os.path.relpath(path[:-3] + "png", os.path.dirname(HERE)))


# -- driver --------------------------------------------------------------------------

FIGS = {"hat": figure_hat, "dictionary": figure_dictionary,
        "torus": figure_torus, "funnel": figure_funnel_classes,
        "tikz": figure_tikz}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    want = argv or list(FIGS)
    for w in want:
        if w not in FIGS:
            raise SystemExit("unknown figure %r; choose from %s"
                             % (w, ", ".join(FIGS)))
    for w in want:
        print("[%s]" % w)
        FIGS[w]()


if __name__ == "__main__":
    main()
