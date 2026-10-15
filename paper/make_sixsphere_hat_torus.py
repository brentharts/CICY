#!/usr/bin/env python3
"""
Rolling an aperiodic Hat tiling up on the six-sphere's torus.

supplementary_material_sixsphere.tex (Theorem 1 and Section 5.3) says the cusp
fibre W is the Laves substrate rolled up on R^2 / B0 Z^2, and that no Hat tiling
descends to it, because an aperiodic tiling is invariant under no translation
lattice. The substrate half is computed in pyCICY. This script computes the
other half, on a genuine aperiodic tiling.

The tiling comes from brentharts/spectre (spectre.py), whose substitution
system generates Tile(a, b) for any proportions; at a, b = 1, sqrt3 it is the
Hat. Before anything is drawn the script asserts that

  * every tile vertex lies on a site of the [3.4.6.4] Laves lattice at
    a : b = sqrt3 : 1 (the lattice of ss_hat_kites);
  * every Hat is exactly 8 whole kites of it, and the 'mystic' tile 10;

and then, for every lattice bond, asks whether it is a tile boundary (its two
adjacent kites belong to different tiles). For a tiling invariant under the
period lattice B0 Z^2 every one of the 12 bond orbits of the torus would be a
boundary always or never. The figures show that none is.

    ss_hat_rollup   (a) the patch on the lattice  (b) the torus, each bond
                    orbit weighted by how often it is a tile boundary
                    (c) the 12 boundary fractions
    ss_hat_overlap  how much the patch agrees with each of its own lattice
                    translates

The spectre repository is found through $SPECTRE_DIR, or as ../spectre next to
this repository, or as external/spectre inside it:

    git clone https://github.com/brentharts/spectre.git ../spectre
    python3 paper/make_sixsphere_hat_torus.py
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
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Polygon
from matplotlib.path import Path

import make_sixsphere_figures as F

S3, A = F.S3, F.A
BLUE, VIOLET = F.BLUE, "#6250d6"
HAT_FILL, MYSTIC_FILL = "#d6e4f7", "#dcd8f5"
# one hue, light to dark: the boundary fraction of an orbit
RAMP = LinearSegmentedColormap.from_list(
    "frac", ["#e1ebf7", "#9dc0ea", "#2a78d6", "#143f78"])

PATCH_RADIUS = 75.0     # keep tiles whose centre is within this of the origin
LATTICE_R = 40          # lattice centres i*E1 + j*E2, |i|, |j| <= LATTICE_R


# -- the spectre repository ------------------------------------------------

def load_spectre():
    cands = [os.environ.get("SPECTRE_DIR"),
             os.path.join(os.path.dirname(ROOT), "spectre"),
             os.path.join(ROOT, "external", "spectre")]
    for d in cands:
        if d and os.path.isfile(os.path.join(d, "spectre.py")):
            sys.path.insert(0, d)
            import spectre
            return spectre
    raise SystemExit(
        "spectre.py not found. Clone https://github.com/brentharts/spectre.git "
        "next to this repository, or set SPECTRE_DIR.")


# -- the lattice, once ---------------------------------------------------------

class Lattice(object):
    """Sites, kites and bonds of the Hat-point Laves lattice, with the maps
    needed to fold a bond onto the torus."""

    def __init__(self, R=LATTICE_R):
        self.sites, self.kind, _, self.key = F.laves(R=R)
        self.R = R
        M = np.array([F.E1, F.E2]).T
        self.Mi = np.linalg.inv(M)
        self.rank = {"centre": 0, "edge": 1, "vertex": 2}
        kites = []
        for i in range(-R, R + 1):
            for j in range(-R, R + 1):
                c = i * F.E1 + j * F.E2
                for k in range(6):
                    kites.append([c, c + A * F._u(30 + 60 * k),
                                  c + 2 * F._u(60 * k + 60),
                                  c + A * F._u(90 + 60 * k)])
        self.kites = kites
        self.cent = np.array([sum(q) / 4 for q in kites])
        self.kbonds = {}
        for ki, q in enumerate(kites):
            ks = [self.key(p) for p in q]
            for a, b in ((0, 1), (1, 2), (2, 3), (3, 0)):
                self.kbonds.setdefault(tuple(sorted((ks[a], ks[b]))),
                                       []).append(ki)

    def frac(self, p):
        u = self.Mi @ p
        return tuple(np.round(u - np.floor(u + 1e-9), 6) % 1.0)

    def orbit(self, ka, kb):
        """The torus orbit of a bond: the same key F.torus_data() uses."""
        a, b = (ka, kb) if self.rank[self.kind[ka]] < self.rank[self.kind[kb]] \
            else (kb, ka)
        return (self.kind[a], self.frac(self.sites[a]), self.kind[b],
                tuple(np.round(self.sites[b] - self.sites[a], 6) + 0.0))


# -- the Hat patch ---------------------------------------------------------------

def hat_patch(spectre, lat, n):
    """Tiles of the n-th substitution step at the Hat proportions, moved onto
    the lattice. Returns (polygons, labels, owner) where owner[k] is the tile
    covering kite k (-1: none). Asserts the facts listed in the docstring."""
    tiles = spectre.buildSpectreTiles(n, 1.0, S3, rotation=30)
    placed = []
    tiles["Delta"].forEachTile(lambda T, l: placed.append((T.copy(), l)))
    pts = spectre.get_spectre_points(1.0, S3)
    myst = spectre.get_spectre_points(S3, 1.0)
    polys = [((myst if l == "Gamma2" else pts) @ T[:, :2].T + T[:, 2], l)
             for T, l in placed]
    # register: some vertex of tile 0 is a lattice site of some kind
    P0 = polys[0][0]
    offs = [np.zeros(2)] + [A * F._u(30 + 60 * k) for k in range(6)] \
        + [2 * F._u(60 * k) for k in range(6)]
    c0 = None
    for v in P0:
        for off in offs:
            if all(lat.key(p - (v - off)) in lat.sites for p in P0):
                c0 = v - off
                break
        if c0 is not None:
            break
    assert c0 is not None, "tile 0 does not sit on the lattice"
    polys = [(P - c0, l) for P, l in polys]
    polys = [(P, l) for P, l in polys if np.hypot(*P.mean(0)) < PATCH_RADIUS]
    assert all(lat.key(p) in lat.sites for P, _ in polys for p in P), \
        "a tile vertex is off the lattice"
    owner = -np.ones(len(lat.kites), int)
    for t, (P, _) in enumerate(polys):
        lo, hi = P.min(0) - 1, P.max(0) + 1
        m = np.where((lat.cent[:, 0] > lo[0]) & (lat.cent[:, 0] < hi[0]) &
                     (lat.cent[:, 1] > lo[1]) & (lat.cent[:, 1] < hi[1]))[0]
        owner[m[Path(P).contains_points(lat.cent[m])]] = t
    cnt = np.bincount(owner[owner >= 0], minlength=len(polys))
    for t, (_, l) in enumerate(polys):
        assert cnt[t] == (10 if l == "Gamma2" else 8), \
            ("tile is not whole kites", l, cnt[t])
    return [P for P, _ in polys], [l for _, l in polys], owner


def boundary_map(lat, owner):
    """bond -> True/False (is a tile boundary), over the bonds both of whose
    kites are covered by the patch."""
    beta = {}
    for b, ks in lat.kbonds.items():
        if len(ks) == 2 and owner[ks[0]] >= 0 and owner[ks[1]] >= 0:
            beta[b] = bool(owner[ks[0]] != owner[ks[1]])
    return beta


def orbit_fractions(lat, beta, torus_keys):
    tot, bd = {}, {}
    for b, v in beta.items():
        o = lat.orbit(*b)
        tot[o] = tot.get(o, 0) + 1
        bd[o] = bd.get(o, 0) + v
    assert set(tot) == set(torus_keys), "orbit keys disagree with torus_data"
    return {o: bd[o] / tot[o] for o in tot}, tot


def overlap(lat, beta, rng=5):
    """Agreement of the patch's boundary pattern with its own translate by
    i*E1 + j*E2: the fraction of bonds b, with b and b + t both in the patch,
    on which 'is a boundary' agrees. 1 for every t would be a tiling invariant
    under the lattice."""
    out = {}
    for i in range(-rng, rng + 1):
        for j in range(-rng, rng + 1):
            t = i * F.E1 + j * F.E2
            agree = n = 0
            for (a, b), v in beta.items():
                pa, pb = lat.sites[a] + t, lat.sites[b] + t
                ka, kb = lat.key(pa), lat.key(pb)
                w = beta.get(tuple(sorted((ka, kb))))
                if w is None:
                    continue
                n += 1
                agree += (v == w)
            out[(i, j)] = (agree / n if n else float("nan"), n)
    return out


# -- figures -------------------------------------------------------------------------

def _torus(u, v, lift=1.0, Rt=2.0, rt=0.85):
    u, v = np.asarray(u) * 2 * np.pi, np.asarray(v) * 2 * np.pi
    r = rt * lift
    return np.array([(Rt + r * np.cos(v)) * np.cos(u),
                     (Rt + r * np.cos(v)) * np.sin(u), r * np.sin(v)])


def _visible(u, v, view):
    u, v = np.asarray(u) * 2 * np.pi, np.asarray(v) * 2 * np.pi
    n = np.array([np.cos(v) * np.cos(u), np.cos(v) * np.sin(u), np.sin(v)])
    return np.tensordot(view, n, axes=1) > 0.02


def figure_rollup(lat, polys, labels, frac3, frac4, n_tiles):
    reps, orbits, _ = F.torus_data()
    fig = plt.figure(figsize=(12.2, 4.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 0.95], wspace=0.2)

    # (a) the patch on the lattice
    ax = fig.add_subplot(gs[0])
    r = 15.5
    shown = [(P, l) for P, l in zip(polys, labels) if np.hypot(*P.mean(0)) < r]
    for P, l in shown:
        ax.add_patch(Polygon(P, closed=True,
                             fc=MYSTIC_FILL if l == "Gamma2" else HAT_FILL,
                             ec=F.INK, lw=1.1, joinstyle="round", zorder=2))
    win = (-r - 2.5, r + 2.5, -r - 2.5, r + 2.5)
    F.draw_lattice(ax, lat.sites, lat.kind,
                   [b for b in lat.kbonds if len(lat.kbonds[b]) == 2],
                   win, bond_kw=dict(color="#c9c8c2", lw=0.5, zorder=1),
                   markers=False)
    dom = np.array([[0, 0], F.E1, F.E1 + F.E2, F.E2])
    ax.add_patch(Polygon(dom, closed=True, fc="none", ec=F.ORANGE, lw=2.0,
                         ls=(0, (4, 2)), zorder=5))
    ax.annotate("one $B_0\\mathbb{Z}^2$ domain", xy=tuple(0.5 * (F.E1 + F.E2)),
                xytext=(-17.6, 16.6), color=F.ORANGE, fontsize=8.5, ha="left",
                va="center", zorder=7,
                bbox=dict(fc="white", ec="none", pad=2.0),
                arrowprops=dict(arrowstyle="-", color=F.ORANGE, lw=1.0))
    ax.set_xlim(win[0], win[1])
    ax.set_ylim(win[2], win[3])
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("(a) a Hat patch on the lattice", fontsize=9.5)
    h = [plt.Line2D([], [], marker="s", ls="", color=HAT_FILL, mec=F.INK,
                    ms=8, label="Hat (8 kites)"),
         plt.Line2D([], [], marker="s", ls="", color=MYSTIC_FILL, mec=F.INK,
                    ms=8, label="mystic tile (10 kites)")]
    ax.legend(handles=h, loc="lower center", frameon=False, fontsize=8,
              ncol=2, bbox_to_anchor=(0.5, -0.07))

    # (b) rolled up, each orbit weighted by its boundary fraction
    ax3 = fig.add_subplot(gs[1], projection="3d")
    ax3.computed_zorder = False
    elev, azim = 52.0, -68.0
    view = np.array([np.cos(np.radians(elev)) * np.cos(np.radians(azim)),
                     np.cos(np.radians(elev)) * np.sin(np.radians(azim)),
                     np.sin(np.radians(elev))])
    uu, vv = np.meshgrid(np.linspace(0, 1, 90), np.linspace(0, 1, 40))
    X, Y, Z = _torus(uu, vv)
    ax3.plot_surface(X, Y, Z, color="#eef2f8", alpha=1.0, linewidth=0,
                     antialiased=True, shade=False, zorder=0)
    ax3.plot_wireframe(X, Y, Z, rstride=5, cstride=9, color="#d6dde8",
                       linewidth=0.35, zorder=1)
    for key, (uv0, duv) in orbits.items():
        p = frac4[key]
        t = np.linspace(0, 1, 40)
        pts = uv0[:, None] + duv[:, None] * t[None, :]
        P = _torus(pts[0], pts[1], 1.03)
        P = np.where(_visible(pts[0], pts[1], view)[None, :], P, np.nan)
        ax3.plot(P[0], P[1], P[2], color=RAMP(p), lw=2.4 + 5.0 * p, zorder=3,
                 solid_capstyle="round")
    ax3.view_init(elev=elev, azim=azim)
    ax3.set_box_aspect((1, 1, 0.55), zoom=1.22)
    ax3.set_axis_off()
    ax3.set_title("(b) rolled up: each bond orbit shaded by how\noften it "
                  "is a tile boundary", fontsize=9.5)
    sm = plt.cm.ScalarMappable(cmap=RAMP, norm=plt.Normalize(0, 1))
    cax = fig.add_axes([0.435, 0.07, 0.17, 0.022])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_ticks([0, 0.5, 1])
    cb.set_ticklabels(["never", "", "always"])
    cb.outline.set_visible(False)
    cax.tick_params(labelsize=8, length=0)

    # (c) the twelve fractions
    bx = fig.add_subplot(gs[2])
    order = sorted(orbits, key=lambda k: (k[0] != "centre", frac4[k]))
    xs = np.arange(len(order))
    bx.axhspan(-0.05, 0.0, color="#e4e3de", lw=0)
    bx.axhspan(1.0, 1.05, color="#e4e3de", lw=0)
    bx.axhline(0, color=F.MUTED, lw=0.8)
    bx.axhline(1, color=F.MUTED, lw=0.8)
    bx.scatter(xs, [frac3[k] for k in order], s=34, facecolors="white",
               edgecolors=BLUE, linewidths=1.2, zorder=3,
               label="smaller patch (%d tiles)" % n_tiles[0])
    bx.scatter(xs, [frac4[k] for k in order], s=34, c=BLUE, zorder=4,
               label="larger patch (%d tiles)" % n_tiles[1])
    bx.text(5.5, 1.075, "a tiling invariant under $B_0\\mathbb{Z}^2$:\nevery "
            "orbit at 0 or 1", ha="center", va="bottom", fontsize=8,
            color=F.INK2)
    bx.set_xticks([2.5, 8.5])
    bx.set_xticklabels(["centre–edge\nbonds", "edge–vertex\nbonds"],
                       fontsize=8.5)
    bx.axvline(5.5, ymin=0.04, ymax=0.62, color="#e4e3de", lw=0.8)
    bx.set_xlim(-0.8, 11.8)
    bx.set_ylim(-0.05, 1.22)
    bx.set_yticks([0, 0.25, 0.5, 0.75, 1])
    bx.set_ylabel("fraction that are tile boundaries")
    for s in ("top", "right"):
        bx.spines[s].set_visible(False)
    bx.legend(loc="center", frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.74))
    bx.set_title("(c) the twelve orbits, at two patch sizes", fontsize=9.5)
    F._save(fig, "ss_hat_rollup")


def figure_overlap(lat, ov, rng=5):
    fig, ax = plt.subplots(figsize=(5.0, 5.7))
    ids = list(ov)
    pts = np.array([i * F.E1 + j * F.E2 for i, j in ids])
    vals = np.array([ov[k][0] for k in ids])
    nz = np.array([k != (0, 0) for k in ids])
    best = ids[int(np.nanargmax(np.where(nz, vals, -1)))]
    sc = ax.scatter(pts[:, 0], pts[:, 1], c=vals, cmap=RAMP, vmin=0.45,
                    vmax=1.0, s=150, edgecolors="white", linewidths=0.8,
                    zorder=3)
    ring = lambda t, col, r: ax.scatter([t[0]], [t[1]], s=r, facecolors="none",
                                        edgecolors=col, lw=1.6, zorder=4)
    ring(np.zeros(2), F.INK, 400)
    ring(F.E1, F.ORANGE, 300)
    ring(F.E2, F.ORANGE, 300)
    pb = best[0] * F.E1 + best[1] * F.E2
    ring(pb, F.INK2, 300)
    box = dict(fc="white", ec="none", pad=1.6, alpha=0.92)
    ax.annotate("identity", xy=(0, 0), xytext=(-12.5, -4.5), fontsize=8.5,
                color=F.INK, ha="center", bbox=box, zorder=7,
                arrowprops=dict(arrowstyle="-", color=F.INK2, lw=0.7))
    ax.annotate("$E_1$: %.2f" % ov[(1, 0)][0], xy=tuple(F.E1),
                xytext=(11.5, -3.5), fontsize=8.5, color=F.ORANGE,
                ha="center", bbox=box, zorder=7,
                arrowprops=dict(arrowstyle="-", color=F.ORANGE, lw=0.8))
    ax.annotate("$E_2$: %.2f" % ov[(0, 1)][0], xy=tuple(F.E2),
                xytext=(-9.5, 9.0), fontsize=8.5, color=F.ORANGE,
                ha="center", bbox=box, zorder=7,
                arrowprops=dict(arrowstyle="-", color=F.ORANGE, lw=0.8))
    ax.annotate("best: $%d E_1 %+d E_2$, %.2f" % (best[0], best[1],
                                                 ov[best][0]),
                xy=tuple(pb), xytext=(pb[0] - 15.5, pb[1] + 4.0),
                fontsize=8.5, color=F.INK2, ha="center", bbox=box, zorder=7,
                arrowprops=dict(arrowstyle="-", color=F.INK2, lw=0.8))
    ax.set_aspect("equal")
    ax.axis("off")
    cb = fig.colorbar(sc, ax=ax, fraction=0.05, pad=0.01, shrink=0.8)
    cb.set_label("agreement of the boundary pattern with its translate",
                 fontsize=8.5)
    cb.outline.set_visible(False)
    ax.set_title("Which lattice translations does the Hat patch survive?",
                 fontsize=10, color=F.INK)
    fig.text(0.5, 0.025, "a tiling invariant under $B_0\\mathbb{Z}^2$ would "
             "show 1.00 at every point", ha="center", fontsize=8.5,
             color=F.INK2)
    F._save(fig, "ss_hat_overlap")
    return float(np.nanmax(vals[nz])), float(np.nanmin(vals[nz])), best


# -- driver ------------------------------------------------------------------------------

def main():
    spectre = load_spectre()
    F.check_against_exact()
    lat = Lattice()
    _, orbits, _ = F.torus_data()
    torus_keys = set(orbits)
    res = {}
    for n in (3, 4):
        polys, labels, owner = hat_patch(spectre, lat, n)
        beta = boundary_map(lat, owner)
        fr, tot = orbit_fractions(lat, beta, torus_keys)
        res[n] = (polys, labels, owner, beta, fr)
        v = np.array(list(fr.values()))
        print("n=%d: %d tiles (%d hats, %d mystic), %d bonds; orbit "
              "fractions %.3f .. %.3f" % (n, len(polys),
                                          labels.count("Hat") + sum(1 for l in labels if l != "Gamma2"),
                                          labels.count("Gamma2"), len(beta),
                                          v.min(), v.max()))
        assert 0.0 < v.min() and v.max() < 1.0, "an orbit is always/never a boundary"
    polys, labels, owner, beta, fr4 = res[4]
    fr3 = res[3][4]
    figure_rollup(lat, polys, labels, fr3, fr4,
                  (len(res[3][0]), len(res[4][0])))
    ov = overlap(lat, beta)
    best, low, at = figure_overlap(lat, ov)
    print("overlap: best non-identity %.3f at %s, lowest %.3f" % (best, at, low))
    assert best < 1.0


if __name__ == "__main__":
    main()
