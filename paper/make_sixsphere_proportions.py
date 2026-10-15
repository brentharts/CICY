#!/usr/bin/env python3
"""
The Hat, the Spectre and the Turtle on the six-sphere's torus.

Theorem 1 of supplementary_material_sixsphere.tex says the monotile substrate
is the Hasse diagram of the A2 triangulation at the Hat point a : b = sqrt3 : 1
and only there; the paper exhibits two other proportions, the Spectre's and one
within 2% of the Hat's. This script adds the Turtle, the Hat's mirror partner,
and rolls each substrate up on its own period lattice.

For a tile Tile(a, b) the substrate is the [3.4.6.4]-type lattice whose
hexagon-centre-to-edge-midpoint bond is the tile's b and whose
edge-midpoint-to-vertex bond is its a (so the Hat, Tile(1, sqrt3), sits on the
a : b = sqrt3 : 1 lattice of ss_hat_kites):

    Hat      Tile(1, sqrt3)   substrate a : b = sqrt3 : 1
    Spectre  Tile(1, 1)       substrate a : b = 1 : 1
    Turtle   Tile(sqrt3, 1)   substrate a : b = 1 : sqrt3

Every number in the figure is computed here, and the substrates are checked
against pyCICY.monotile.laves_patch before anything is drawn:

  * sites per period domain, by kind, and the bonds' orbits;
  * the valence of each site class;
  * cross-kind coincidences (a site of one kind on a site of another);
  * whether sites and bonds are exactly the cells and cover relations of the
    A2 triangulation.

Tile outlines come from brentharts/spectre (spectre.py); see
make_sixsphere_hat_torus.py for how it is located.

    python3 paper/make_sixsphere_proportions.py
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
from matplotlib.path import Path

import make_sixsphere_figures as F
from make_sixsphere_hat_torus import load_spectre, HAT_FILL, MYSTIC_FILL

S3 = F.S3
SPECTRE_FILL = "#e9e7df"
RANK = {"centre": 0, "edge": 1, "vertex": 2}
KEYR = lambda p: (round(float(p[0]), 6) + 0.0, round(float(p[1]), 6) + 0.0)


# -- a substrate at any proportions ------------------------------------------

class Substrate(object):
    """Hexagon centres on the lattice <E1, E2> = <2a u30, 2a u90>, six edge
    midpoints at distance a at 30 + 60k degrees, and from each midpoint two
    vertex sites at distance b, perpendicular. Nodes are (kind, position), so a
    site of one kind sitting on a site of another stays visible."""

    def __init__(self, a, b, R=5):
        self.a, self.b = a, b
        self.E1, self.E2 = 2 * a * F._u(30), 2 * a * F._u(90)
        M = np.array([self.E1, self.E2]).T
        self.Mi = np.linalg.inv(M)
        self.xy, self.bonds = {}, set()
        for i in range(-R, R + 1):
            for j in range(-R, R + 1):
                c = i * self.E1 + j * self.E2
                nc = self._node("centre", c)
                for k in range(6):
                    m = c + a * F._u(30 + 60 * k)
                    nm = self._node("edge", m)
                    self.bonds.add(frozenset((nc, nm)))
                    for dv in (-90, 90):
                        v = m + b * F._u(30 + 60 * k + dv)
                        self.bonds.add(frozenset((nm, self._node("vertex", v))))
        self.deg = {}
        for bd in self.bonds:
            for n in bd:
                self.deg[n] = self.deg.get(n, 0) + 1

    def _node(self, kind, p):
        n = (kind, KEYR(p))
        self.xy.setdefault(n, np.array(p, dtype=float))
        return n

    def frac(self, p):
        u = self.Mi @ np.asarray(p)
        return tuple(np.round(u - np.floor(u + 1e-9), 6) % 1.0)

    def quotient(self):
        """Per period domain: node classes by kind, bond orbits, valences,
        cross-kind coincidences."""
        cls = {}
        for n, p in self.xy.items():
            cls.setdefault((n[0], self.frac(p)), []).append(n)
        counts = {k: sum(1 for kk, _ in cls if kk == k) for k in RANK}
        orbits = set()
        for bd in self.bonds:
            x, y = sorted(bd, key=lambda n: (RANK[n[0]], n[1]))
            orbits.add((x[0], self.frac(self.xy[x]), y[0],
                        tuple(np.round(self.xy[y] - self.xy[x], 6) + 0.0)))
        # valence at a position, all kinds together; and of each kind alone
        pos_deg, kind_deg = {}, {}
        for n, p in self.xy.items():
            f = self.frac(p)
            pos_deg[f] = pos_deg.get(f, 0) + self.deg.get(n, 0)
            kind_deg.setdefault(n[0], set()).add(self.deg.get(n, 0))
        merged = {f for f in {self.frac(p) for p in self.xy.values()}
                  if len({n[0] for n, p in self.xy.items() if self.frac(p) == f}) > 1}
        return dict(counts=counts, orbits=len(orbits), merged=len(merged),
                    kind_deg={k: sorted(v) for k, v in kind_deg.items()},
                    pos_deg=pos_deg)


def a2_cells_and_covers(sub, R=4):
    """Cells of the A2 triangulation on the lattice <E1, E2>, and its cover
    relations vertex < edge < triangle, as sets of (kind, position)."""
    pos = lambda i, j: i * sub.E1 + j * sub.E2
    cells, covers = set(), set()
    for i in range(-R, R + 1):
        for j in range(-R, R + 1):
            for t in (((i, j), (i + 1, j), (i, j + 1)),
                      ((i + 1, j), (i, j + 1), (i + 1, j + 1))):
                g = sum(pos(*v) for v in t) / 3
                ng = ("vertex", KEYR(g))
                cells.add(ng)
                for p, q in ((0, 1), (1, 2), (0, 2)):
                    m = (pos(*t[p]) + pos(*t[q])) / 2
                    nm = ("edge", KEYR(m))
                    cells.add(nm)
                    covers.add(frozenset((nm, ng)))
                    for v in (t[p], t[q]):
                        nv = ("centre", KEYR(pos(*v)))
                        cells.add(nv)
                        covers.add(frozenset((nv, nm)))
    return cells, covers


def is_hasse_of_A2(sub, radius_periods=2.2):
    cells, covers = a2_cells_and_covers(sub)
    r = radius_periods * np.linalg.norm(sub.E1)
    near = lambda n: np.hypot(*sub.xy[n]) < r if n in sub.xy else False
    cell_xy = lambda n: np.array(n[1])
    mine = {n for n in sub.xy if np.hypot(*sub.xy[n]) < r}
    theirs = {n for n in cells if np.hypot(*cell_xy(n)) < r}
    sites_ok = mine == theirs
    bmine = {b for b in sub.bonds if all(np.hypot(*sub.xy[n]) < r for n in b)}
    btheirs = {b for b in covers if all(np.hypot(*cell_xy(n)) < r for n in b)}
    return sites_ok and bmine == btheirs


def check_against_exact(sub, a_q, b_q):
    """Every site and every bond of pyCICY.monotile.laves_patch (exact, Q(sqrt3))
    is in this substrate, as positions."""
    from pyCICY import monotile as Mo
    P = Mo.laves_patch(a=a_q, b=b_q, rings=2)
    q = lambda x: float(x.p) + float(x.q) * S3
    pts = [(q(x), q(y)) for x, y in P["sites"]]
    mine = {n[1] for n in sub.xy}
    assert all(KEYR(p) in mine for p in pts), "an exact site is missing"
    pb = {frozenset((KEYR(sub.xy[a_]), KEYR(sub.xy[b_])))
          for a_, b_ in map(tuple, sub.bonds)}
    for i, j, _ in P["bonds"]:
        key = frozenset((KEYR(pts[i]), KEYR(pts[j])))
        assert key in pb, "an exact bond is missing"
    return len(P["sites"]), len(P["bonds"])


# -- tiles on the Hat-point lattice -------------------------------------------

def kites_on_hat_lattice(P, lat):
    """Number of whole kites of the Hat-point lattice that make up the polygon
    P, in some placement; None if there is no such placement."""
    A = F.A
    for refl in (1, -1):
        Q0 = P * np.array([1.0, refl])
        for rot in range(0, 360, 30):
            th = np.radians(rot)
            R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
            Q = Q0 @ R.T
            for site in lat.sites:
                if abs(site[0]) > 6 or abs(site[1]) > 6:
                    continue
                Tr = np.array(site) - Q[0]
                Z = Q + Tr
                if not all(lat.key(p) in lat.sites for p in Z):
                    continue
                path = Path(Z)
                inside = [k for k, c in zip(lat.kites, lat.cent)
                          if path.contains_point(c)]
                if not inside:
                    continue
                if all(path.contains_point(p, radius=1e-6) or
                       path.contains_point(p, radius=-1e-6)
                       for k in inside for p in k):
                    x, y = Z[:, 0], Z[:, 1]
                    ar = 0.5 * abs(x @ np.roll(y, -1) - y @ np.roll(x, 1))
                    ar = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
                    if abs(ar - len(inside) * A * F.B) < 1e-6:
                        return len(inside)
    return None


# -- the figure -------------------------------------------------------------------

def figure_proportions():
    from pyCICY.monotile import Quad
    spectre = load_spectre()
    F.check_against_exact()
    one, r3 = Quad(1, 0, 3), Quad(0, 1, 3)
    cases = [
        dict(name="Hat", tile=(1.0, S3), tile_txt="Tile$(1,\\sqrt{3})$",
             ab=(S3, 1.0), q=(r3, one), ratio="$\\sqrt{3}:1$", fill=HAT_FILL),
        dict(name="Spectre", tile=(1.0, 1.0), tile_txt="Tile$(1,1)$",
             ab=(1.0, 1.0), q=(one, one), ratio="$1:1$", fill=SPECTRE_FILL),
        dict(name="Turtle", tile=(S3, 1.0), tile_txt="Tile$(\\sqrt{3},1)$",
             ab=(1.0, S3), q=(one, r3), ratio="$1:\\sqrt{3}$", fill=MYSTIC_FILL),
    ]
    from make_sixsphere_hat_torus import Lattice
    lat = Lattice(R=8)
    for c in cases:
        sub = Substrate(*c["ab"])
        c["sub"] = sub
        c["exact"] = check_against_exact(sub, *c["q"])
        c["quo"] = sub.quotient()
        c["hasse"] = is_hasse_of_A2(sub)
        P = spectre.get_spectre_points(*c["tile"])
        c["poly"] = P
        c["kites"] = kites_on_hat_lattice(P, lat)
        q = c["quo"]
        print("%-8s sites/domain %s  bond orbits %d  cross-kind positions %d  "
              "valences %s  Hasse(A2): %s  kites on Hat lattice: %s"
              % (c["name"], q["counts"], q["orbits"], q["merged"], q["kind_deg"],
                 c["hasse"], c["kites"]))
    hat, spe, tur = cases
    # what the figure will say, asserted so a change in the code cannot drift
    assert hat["hasse"] and not spe["hasse"] and not tur["hasse"]
    assert hat["quo"]["counts"] == {"centre": 1, "edge": 3, "vertex": 2}
    assert spe["quo"]["counts"] == {"centre": 1, "edge": 3, "vertex": 6}
    assert spe["quo"]["kind_deg"]["vertex"] == [1] and spe["quo"]["merged"] == 0
    assert tur["quo"]["counts"]["vertex"] == 1 and tur["quo"]["merged"] == 1
    assert hat["kites"] == 8 and tur["kites"] == 10 and spe["kites"] is None

    fig = plt.figure(figsize=(11.8, 10.6))
    gs = fig.add_gridspec(3, 3, height_ratios=[0.85, 1.25, 1.25],
                          hspace=0.34, wspace=0.08)
    KS = F.KIND
    for col, c in enumerate(cases):
        sub, a = c["sub"], c["ab"][0]
        norm = 2 * a                      # one period = 1 in every column

        # row 1: the tile
        ax = fig.add_subplot(gs[0, col])
        P = c["poly"] / norm
        ax.add_patch(Polygon(P, closed=True, fc=c["fill"], ec=F.INK, lw=1.6,
                             joinstyle="round"))
        ax.set_xlim(P[:, 0].min() - 0.12, P[:, 0].max() + 0.12)
        ax.set_ylim(P[:, 1].min() - 0.12, P[:, 1].max() + 0.12)
        ax.set_aspect("equal")
        ax.axis("off")
        note = ("%d whole kites of the $\\sqrt{3}\\!:\\!1$ lattice" % c["kites"]
                if c["kites"] else "not a union of kites of the $\\sqrt{3}\\!:\\!1$ lattice")
        ax.set_title("%s, %s\n%s" % (c["name"], c["tile_txt"], note), fontsize=9.5)

        # row 2: the exact substrate
        ax = fig.add_subplot(gs[1, col])
        win = 1.75
        inw = lambda p: np.hypot(*p) < win
        for bd in sub.bonds:
            n1, n2 = tuple(bd)
            p1, p2 = sub.xy[n1] / norm, sub.xy[n2] / norm
            if inw(p1) or inw(p2):
                kinds = {n1[0], n2[0]}
                col_ = "#8fd0b8" if "vertex" in kinds else "#bdbcb5"
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=col_, lw=0.8,
                        zorder=1)
        merged = {}
        for n, p in sub.xy.items():
            merged.setdefault(n[1], set()).add(n[0])
        for kind in ("vertex", "edge", "centre"):
            pts = np.array([sub.xy[n] / norm for n in sub.xy
                            if n[0] == kind and inw(sub.xy[n] / norm)])
            if not len(pts):
                continue
            st = KS[kind]
            both = np.array([len(merged[n[1]]) > 1 for n in sub.xy
                             if n[0] == kind and inw(sub.xy[n] / norm)])
            if kind == "vertex" and both.any():
                ax.scatter(pts[both][:, 0], pts[both][:, 1], marker="^", s=190,
                           facecolors="none", edgecolors=F.AQUA, linewidths=1.6,
                           zorder=3)
                pts = pts[~both]
                if not len(pts):
                    continue
            ax.scatter(pts[:, 0], pts[:, 1], marker=st["marker"],
                       s=st["s"] * 0.8, c=st["c"], edgecolors="white",
                       linewidths=0.5, zorder=4 if kind == "centre" else 3)
        ax.set_xlim(-win, win)
        ax.set_ylim(-win, win)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title("substrate at $a:b=%s$" % c["ratio"].strip("$"), fontsize=9.5)

        # row 3: rolled up on one period domain
        ax = fig.add_subplot(gs[2, col])
        E1, E2 = sub.E1 / norm, sub.E2 / norm
        dom = np.array([[0, 0], E1, E1 + E2, E2])
        ax.add_patch(Polygon(dom, closed=True, fc="#f6f6f2", ec="none", zorder=0))
        zone = lambda p: (-0.04 <= (sub.Mi @ (p * norm))[0] <= 1.04
                          and -0.04 <= (sub.Mi @ (p * norm))[1] <= 1.04)
        near = lambda p: (-0.45 <= (sub.Mi @ (p * norm))[0] <= 1.45
                          and -0.45 <= (sub.Mi @ (p * norm))[1] <= 1.45)
        for bd in sub.bonds:
            n1, n2 = tuple(bd)
            p1, p2 = sub.xy[n1] / norm, sub.xy[n2] / norm
            if near(p1) and near(p2):
                kinds = {n1[0], n2[0]}
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                        color="#8fd0b8" if "vertex" in kinds else "#bdbcb5",
                        lw=0.9, zorder=1)
        ax.add_patch(Polygon(dom, closed=True, fc="none", ec=F.INK, lw=1.5,
                             zorder=5))
        for n, p in sub.xy.items():
            pn = p / norm
            if not zone(pn):
                continue
            u = sub.Mi @ p
            rep = (u[0] < 1 - 1e-6) and (u[1] < 1 - 1e-6) and \
                  (u[0] > -1e-6) and (u[1] > -1e-6)
            st = KS[n[0]]
            both = len(merged[n[1]]) > 1 and n[0] == "vertex"
            if both:
                ax.scatter([pn[0]], [pn[1]], marker="^", s=170,
                           facecolors="none", edgecolors=F.AQUA,
                           linewidths=1.6 if rep else 1.0, zorder=5)
                continue
            ax.scatter([pn[0]], [pn[1]], marker=st["marker"], s=st["s"] * 1.0,
                       c=st["c"] if rep else "white", edgecolors=st["c"],
                       linewidths=1.0, zorder=6)
        q = c["quo"]
        ax.set_xlim(min(0, E2[0]) - 0.4, E1[0] + 0.4)
        ax.set_ylim(-0.4, (E1 + E2)[1] + 0.4)
        ax.set_aspect("equal")
        ax.axis("off")
        v = q["counts"]["vertex"]
        vv = q["kind_deg"]["vertex"]
        title = ("rolled up: %d centre, %d edges,\n%d vertex site%s"
                 % (q["counts"]["centre"], q["counts"]["edge"], v,
                    "" if v == 1 else "s"))
        verdict = ("Hasse of $A_2$: yes (valence %s)" % vv[-1]
                   if c["hasse"] else
                   "Hasse of $A_2$: no (%s)" % (
                       "corners apart, valence 1" if q["merged"] == 0
                       else "vertices on centres"))
        ax.set_title(title + "\n" + verdict, fontsize=9)

    handles = [plt.Line2D([], [], ls="", marker=KS[k]["marker"], color=KS[k]["c"],
                          ms=6.5, label=KS[k]["label"]) for k in KS]
    handles.append(plt.Line2D([], [], ls="", marker="^", markerfacecolor="none",
                              markeredgecolor=F.AQUA, markeredgewidth=1.5, ms=9,
                              label="vertex site on a centre"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, 0.045))
    fig.text(0.5, 0.025, "Periods are normalised to 1 in every column; hollow "
             "marks in the bottom row are images of a representative on the "
             "domain's boundary.", ha="center", fontsize=8, color=F.INK2)
    F._save(fig, "ss_proportions")


if __name__ == "__main__":
    figure_proportions()
