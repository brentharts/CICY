#!/usr/bin/env python3
"""
Two six-sphere tori for the mystic tiles of the Hat tiling at two substitution
steps (71 tiles, 8 of them mystic, Gamma2).

  (a) the tiling folded onto the torus: every mystic is 10 whole kites of the
      Hat-point lattice, and each kite lands on one of the 6 kites of the torus
      (the domain B0 Z^2 has area 6 sqrt3, a kite sqrt3). Mystics are drawn
      translucent, so a kite covered by several mystics reads darker.
  (b) the fibre connections: the plane displacement between neighbouring
      mystics, drawn as the path it winds along on the torus. Each mystic's
      centroid is one point of the torus; the plane is the universal cover and
      the fibre over a torus point is its orbit under B0 Z^2.

Facts printed and asserted: tile and mystic counts, kites per mystic (10), and
that the folded centroids are 8 points whose pairwise displacements are not
lattice vectors (so no two mystics are translates of one another's position).

    python3 paper/make_sixsphere_mystic_fibres.py   (needs ../spectre)
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import make_sixsphere_figures as F
import make_sixsphere_hat_torus as H
from make_sixsphere_hat_torus import _torus, _visible

N_STEPS = 2
E1_, E2_ = F.E1, F.E2
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6",
       "#c0408a", "#b8892a", "#3aa0b8", "#7a8a2a"]


def fold_kite(lat, kite):
    """uv corners of a kite, continuous around its own centroid."""
    c = sum(kite) / 4
    base = lat.Mi @ c
    f = np.floor(base)
    return [lat.Mi @ p - f for p in kite]


def densify(uv, k=7):
    out = []
    for a, b in zip(uv, uv[1:] + uv[:1]):
        for s in np.linspace(0, 1, k, endpoint=False):
            out.append(a + s * (b - a))
    return np.array(out)


def torus_axes(fig, spec, elev=50.0, azim=-62.0):
    ax = fig.add_subplot(spec, projection="3d")
    ax.computed_zorder = False
    view = np.array([np.cos(np.radians(elev)) * np.cos(np.radians(azim)),
                     np.cos(np.radians(elev)) * np.sin(np.radians(azim)),
                     np.sin(np.radians(elev))])
    uu, vv = np.meshgrid(np.linspace(0, 1, 181), np.linspace(0, 1, 73))
    X, Y, Z = _torus(uu, vv)
    ax.plot_surface(X, Y, Z, color="#eceae4", linewidth=0, antialiased=False,
                    shade=False, zorder=0)
    t = np.linspace(0, 1, 120)
    for a, b in ((t, 0 * t), (0 * t, t)):      # the two period circles
        P = _torus(a, b, 1.01)
        P = np.where(_visible(a, b, view)[None, :], P, np.nan)
        ax.plot(*P, color=F.INK2, lw=0.9, zorder=1)
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect((1, 1, 0.55), zoom=1.25)
    ax.set_axis_off()
    return ax, view


EXPECT = {2: (71, 8), 3: (443, 51)}
DCURVE = {(0.5, 0.0): "D_a", (0.0, 0.5): "D_c", (0.5, 0.5): "D_b"}


def main(N_STEPS=2, name="ss_mystic_fibres"):
    spectre = H.load_spectre()
    lat = H.Lattice()
    polys, labels, owner = H.hat_patch(spectre, lat, N_STEPS)
    my = [t for t, l in enumerate(labels) if l == "Gamma2"]
    print("n=%d: %d tiles, %d mystic" % (N_STEPS, len(polys), len(my)))
    assert (len(polys), len(my)) == EXPECT[N_STEPS]
    cen = {t: polys[t].mean(0) for t in my}
    kites = {t: [lat.kites[k] for k in np.where(owner == t)[0]] for t in my}
    assert all(len(v) == 10 for v in kites.values())
    uv = {t: lat.Mi @ cen[t] % 1.0 for t in my}
    # fibre classes: mystics that are lattice translates of one another
    # (same centroid mod B0 Z^2 and the same vertex set after translation)
    def same(a, b):
        d = lat.Mi @ (cen[b] - cen[a])
        if np.abs(d - np.round(d)).max() > 1e-6:
            return False
        t = np.round(d)[0] * F.E1 + np.round(d)[1] * F.E2
        return {lat.key(p + t) for p in polys[a]} == {lat.key(p) for p in polys[b]}
    cls = []
    for t in my:
        for c in cls:
            if same(c[0], t):
                c.append(t); break
        else:
            cls.append([t])
    print("translate classes of the mystics:", sorted(len(c) for c in cls))
    def info(t):
        e = polys[t][1] - polys[t][0]
        ang = round(float(np.degrees(np.arctan2(e[1], e[0]))) % 360, 1)
        k = lat.key(polys[t][0])
        assert lat.kind[k] == "edge"
        return ang, DCURVE[tuple(float(x) for x in lat.frac(lat.sites[k]))]
    cinfo = [{info(t) for t in c} for c in cls]
    assert all(len(x) == 1 for x in cinfo), "a class mixes orientations"
    cinfo = [next(iter(x)) for x in cinfo]
    print("classes (edge angle, anchor double curve):", sorted(cinfo))
    edges = {tuple(sorted((c[i], c[j]))) for c in cls for i in range(len(c))
             for j in range(i + 1, len(c))}
    print("fibre edges:", len(edges))

    fig = plt.figure(figsize=(13.0, 4.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.12, 1, 1], wspace=0.02)

    # (a) one mystic, flat: its 10 kites coloured by the torus kite they fold to
    from matplotlib.patches import Polygon as MPoly
    axa = fig.add_subplot(gs[0])
    TINT = ["#d6e4f7", "#fbdcd0", "#d3eee3", "#e4e0f4", "#f4e3ef", "#f1e8cc"]
    t0 = cls[0][0]
    own = np.where(owner == t0)[0]
    mult = np.zeros(6, int)
    for ki in own:
        mult[ki % 6] += 1
    c0 = polys[t0].mean(0)
    for ki, q in enumerate(lat.kites):
        if np.hypot(*(lat.cent[ki] - c0)) < 7.5:
            on = owner[ki] == t0
            axa.add_patch(MPoly(np.array(q), closed=True,
                                fc=TINT[ki % 6] if on else "#f6f5f1",
                                ec="#bdbab2" if not on else F.INK2,
                                lw=0.6 if not on else 1.0, zorder=1))
            if on:
                m = np.array(q).mean(0)
                axa.text(m[0], m[1], str(ki % 6 + 1), ha="center", va="center",
                         fontsize=9, color=F.INK, zorder=4)
    axa.add_patch(MPoly(polys[t0], closed=True, fc="none", ec=F.INK, lw=2.2,
                        zorder=3))
    for i in range(-3, 4):                       # Lambda-translates of the domain
        for j in range(-3, 4):
            o = i * E1_ + j * E2_
            P = np.array([o, o + E1_, o + E1_ + E2_, o + E2_])
            if np.hypot(*(P.mean(0) - c0)) < 9:
                axa.add_patch(MPoly(P, closed=True, fc="none", ec=F.BLUE,
                                    lw=1.0, ls=(0, (4, 3)), zorder=2))
    axa.set_xlim(c0[0] - 7.4, c0[0] + 7.4)
    axa.set_ylim(c0[1] - 6.6, c0[1] + 6.6)
    axa.set_aspect("equal")
    axa.axis("off")
    axa.set_title("(a) one mystic: 10 kites on 6 torus kites,\nmultiplicities (%s)" % ",".join(map(str, mult)), fontsize=9.5)
    print("one mystic folds with multiplicities", mult.tolist())

    # (b), (c): the six fibre points, seen from above and from below
    t = np.linspace(0, 1, 120)
    loops = {"D_a": (t, 0 * t), "D_c": (0 * t, t), "D_b": (1 - t, t)}
    ls = {"D_a": "-", "D_b": (0, (5, 2.5)), "D_c": (0, (1, 2.4))}
    mid = {"D_a": (0.5, 0.0), "D_c": (0.0, 0.5), "D_b": (0.5, 0.5)}
    for spec, elev, azim, ttl in ((gs[1], -55.0, -180.0, "(b) torus, view 1"),
                                  (gs[2], -55.0, -70.0, "(c) torus, view 2")):
        ax2, view = torus_axes(fig, spec, elev=elev, azim=azim)
        for nm, (u, v) in loops.items():
            P = _torus(u, v, 1.03)
            P = np.where(_visible(u, v, view)[None, :], P, np.nan)
            ax2.plot(*P, color=F.ORANGE, lw=2.2, ls=ls[nm], zorder=2)
        for k, c in enumerate(cls):
            ang, dn = cinfo[k]
            u0, v0 = mid[dn]
            du = 0.03 * (1 if ang < 180 else -1)
            if not _visible(u0 + du, v0, view):
                continue
            p = _torus(u0 + du, v0, 1.08)
            ax2.scatter([p[0]], [p[1]], [p[2]], s=95, c=PAL[k], depthshade=False,
                        edgecolors="white", linewidths=1.0, zorder=6)
        ax2.set_title(ttl + ": anchors on the double curves", fontsize=9.5)

    fig.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=PAL[k], ms=6,
                                   label="%d° on $%s$ (x%d)" % (round(cinfo[k][0]), cinfo[k][1], len(c)))
                        for k, c in enumerate(cls)],
               loc="lower center", ncol=min(6, len(cls)), frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, -0.01))
    fig.text(0.5, 0.96, "Mystic (Turtle) tiles of the Hat tiling, %d substitution steps: "
             "%d tiles, %d mystic" % (N_STEPS, len(polys), len(my)), ha="center", fontsize=10.5, color=F.INK)
    F._save(fig, name)


if __name__ == "__main__":
    main(2, "ss_mystic_fibres")
    main(3, "ss_mystic_fibres_n3")
