#!/usr/bin/env python3
"""
The dual complex of the cusp fibre W: a triangulated torus.

The components of W are the vertices of the A2 triangulation, its double curves
the edges and its triple points the triangles, taken modulo B0 Z^2. The dual
complex of W therefore has one vertex (the dP6 component), three edges (the
double curves) and two triangles (the triple points), and

    V - E + F = 1 - 3 + 2 = 0:

it is the torus, in the standard way, as a square cut along a diagonal.

    ss_dual_complex   (a) the A2 triangulation with its classes mod B0 Z^2,
                      (b) the same complex on the rolled-up torus.

Every count and incidence drawn is asserted against sixsphere.cusp_fibre()
first. The three double curves are named as in tikz_dp6_hexagon: D_a runs along
e1, D_b along e2 - e1, D_c along e2; P and Q are the 'lower' and 'upper'
triangles.

    python3 paper/make_sixsphere_dual_complex.py
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.colors import to_rgba

import make_sixsphere_figures as F
from make_sixsphere_hat_torus import _torus, _visible

E1, E2 = F.E1, F.E2
P_TINT, Q_TINT = "#d3eee3", "#e4e0f4"
# the three double curves: direction in the (e1, e2) basis, label, line style
CURVES = [("D_a", (1, 0), "-"), ("D_b", (-1, 1), (0, (5, 2.5))),
          ("D_c", (0, 1), (0, (1, 2.4)))]


def check_against_cusp_fibre():
    from pyCICY import sixsphere as S
    W = S.cusp_fibre()
    assert (W["components"], W["double_curves"], W["triple_points"]) == (1, 3, 2)
    assert W["dual_complex_euler"] == 0 == 1 - 3 + 2
    inc = W["curves_through_triple_points"]
    directions = sorted(k[0] for k in inc)
    assert directions == sorted(d for _, d, _ in CURVES), directions
    for k, tris in inc.items():
        # every double curve lies on both triangle classes
        assert sorted(t[0] for t in tris) == ["lower", "upper"], (k, tris)


def figure_dual():
    check_against_cusp_fibre()
    fig = plt.figure(figsize=(10.4, 4.7))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.06)
    pos = lambda u, v: u * E1 + v * E2

    # (a) the plane
    ax = fig.add_subplot(gs[0])
    rng = range(-2, 4)
    for i in rng:
        for j in rng:
            lower = np.array([pos(i, j), pos(i + 1, j), pos(i, j + 1)])
            upper = np.array([pos(i + 1, j), pos(i, j + 1), pos(i + 1, j + 1)])
            ax.add_patch(Polygon(lower, closed=True, fc=P_TINT, ec="none",
                                 zorder=0))
            ax.add_patch(Polygon(upper, closed=True, fc=Q_TINT, ec="none",
                                 zorder=0))
    for i in rng:
        for j in rng:
            for (name, d, ls), col in zip(CURVES, (F.ORANGE,) * 3):
                a, b = pos(i, j), pos(i + d[0], j + d[1])
                ax.plot([a[0], b[0]], [a[1], b[1]], color=col, lw=1.8, ls=ls,
                        solid_capstyle="round", dash_capstyle="round", zorder=2)
    pts = np.array([pos(i, j) for i in rng for j in rng])
    ax.scatter(pts[:, 0], pts[:, 1], s=40, c=F.BLUE, edgecolors="white",
               linewidths=0.7, zorder=4)
    for i in rng:
        for j in rng:
            g = pos(i + 1 / 3, j + 1 / 3)
            h = pos(i + 2 / 3, j + 2 / 3)
            ax.scatter([g[0]], [g[1]], marker="^", s=36, c=F.AQUA,
                       edgecolors=F.AQUA, linewidths=1.0, zorder=3)
            ax.scatter([h[0]], [h[1]], marker="^", s=36, c="white",
                       edgecolors=F.AQUA, linewidths=1.2, zorder=3)
    dom = np.array([pos(0, 0), pos(1, 0), pos(1, 1), pos(0, 1)])
    ax.add_patch(Polygon(dom, closed=True, fc="none", ec=F.INK, lw=2.0,
                         zorder=5))
    box = dict(fc="white", ec="none", pad=1.6, alpha=0.95)
    arrow = dict(arrowstyle="-", color=F.ORANGE, lw=0.9)
    for name, at, to in (("D_a", pos(0.5, 0), pos(0.55, -0.62)),
                         ("D_c", pos(0, 0.5), pos(-0.62, 0.5)),
                         ("D_b", pos(0.5, 0.5), pos(-0.15, 1.38))):
        ax.annotate("$%s$" % name, xy=tuple(at), xytext=tuple(to), fontsize=9.5,
                    color=F.ORANGE, ha="center", va="center", bbox=box,
                    arrowprops=arrow, zorder=8)
    for nm, u, to in (("P", 1 / 3, pos(1 / 3, 1 / 3) + np.array([-1.15, -1.25])),
                      ("Q", 2 / 3, pos(2 / 3, 2 / 3) + np.array([1.25, 1.05]))):
        ax.annotate("$%s$" % nm, xy=tuple(pos(u, u)), xytext=tuple(to),
                    fontsize=9.5, color=F.INK, ha="center", va="center",
                    arrowprops=dict(arrowstyle="-", color=F.INK2, lw=0.8),
                    bbox=box, zorder=8)
    ax.set_xlim(pos(-0.9, 0)[0], pos(2.0, 1.0)[0] + 0.4)
    ax.set_ylim(pos(0, -0.55)[1] - 0.3, pos(1.0, 1.9)[1])
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("(a) the $A_2$ triangulation, modulo $B_0\\mathbb{Z}^2$",
                 fontsize=9.5)

    # (b) rolled up
    ax3 = fig.add_subplot(gs[1], projection="3d")
    ax3.computed_zorder = False
    elev, azim = 50.0, -62.0
    view = np.array([np.cos(np.radians(elev)) * np.cos(np.radians(azim)),
                     np.cos(np.radians(elev)) * np.sin(np.radians(azim)),
                     np.sin(np.radians(elev))])
    uu, vv = np.meshgrid(np.linspace(0, 1, 181), np.linspace(0, 1, 73))
    X, Y, Z = _torus(uu, vv)
    uc = (uu[:-1, :-1] + uu[1:, 1:]) / 2
    vc = (vv[:-1, :-1] + vv[1:, 1:]) / 2
    fc = np.where(((uc + vc) < 1)[..., None], to_rgba(P_TINT), to_rgba(Q_TINT))
    ax3.plot_surface(X, Y, Z, facecolors=fc, rstride=1, cstride=1,
                     linewidth=0, antialiased=False, shade=False, zorder=0)
    t = np.linspace(0, 1, 80)
    loops = {"D_a": (t, 0 * t), "D_c": (0 * t, t), "D_b": (1 - t, t)}
    for name, d, ls in CURVES:
        u, v = loops[name]
        P3 = _torus(u, v, 1.03)
        P3 = np.where(_visible(u, v, view)[None, :], P3, np.nan)
        ax3.plot(P3[0], P3[1], P3[2], color=F.ORANGE, lw=2.6, ls=ls,
                 solid_capstyle="round", dash_capstyle="round", zorder=3)
    for (u, v, mk, fc) in ((1 / 3, 1 / 3, "^", F.AQUA), (2 / 3, 2 / 3, "^", "white")):
        if _visible(np.array(u), np.array(v), view):
            P3 = _torus(u, v, 1.06)
            ax3.scatter([P3[0]], [P3[1]], [P3[2]], marker=mk, s=70, c=fc,
                        edgecolors=F.AQUA, linewidths=1.3, depthshade=False,
                        zorder=6)
    P3 = _torus(0.0, 0.0, 1.06)
    ax3.scatter([P3[0]], [P3[1]], [P3[2]], marker="o", s=70, c=F.BLUE,
                edgecolors="white", linewidths=0.9, depthshade=False, zorder=7)
    ax3.view_init(elev=elev, azim=azim)
    ax3.set_box_aspect((1, 1, 0.55), zoom=1.25)
    ax3.set_axis_off()
    ax3.set_title("(b) rolled up: 1 vertex, 3 edges, 2 triangles",
                  fontsize=9.5)

    handles = [
        plt.Line2D([], [], marker="o", ls="", color=F.BLUE, ms=6.5,
                   label="vertex: the component $E_0$"),
        plt.Line2D([], [], color=F.ORANGE, lw=2.2, label="edges: double curves $D_a$ (solid), $D_b$ (dashed), $D_c$ (dotted)"),
        plt.Line2D([], [], marker="^", ls="", color=F.AQUA, ms=7,
                   label="triangles: triple points $P$ (filled), $Q$ (hollow)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, -0.045))
    fig.text(0.5, 0.965, "$V-E+F=1-3+2=0$: the dual complex of $W$ is a torus",
             ha="center", fontsize=10.5, color=F.INK)
    F._save(fig, "ss_dual_complex")


if __name__ == "__main__":
    figure_dual()
