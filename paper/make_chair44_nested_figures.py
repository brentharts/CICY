#!/usr/bin/env python3
"""
Figures for AlpogeSixSphere.md, Part VIII: Chair44 nearly descends to nested
tori, and the Hat does not.

Every number a figure shows is asserted against pyCICY.chair44_nested before
it is drawn, so a figure that renders is a figure whose picture agrees with the
exact arithmetic:

  c44_nested_defect   where a level-6 Chair44 patch disagrees with its own
                      translate by 2^M e_x, in one z-slice: a thin set whose
                      share falls by about four at each step (the label field
                      is shown first; its colour is the rotation, nothing more)
  c44_nested_curve    (a) the defect 1 - agreement against the proved bound
                      c(M)/8^M, with the Hat's best translate for contrast;
                      (b) the agreement of a level-5 patch with every
                      in-plane translate; (c) the 2-adic valuations of the
                      near-periods; (d) the undetermined addresses, which grow
                      like 2^M while the volume grows like 8^M

Each figure is written as PDF and PNG.

    python3 paper/make_chair44_nested_figures.py
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
OUT = os.path.join(HERE, "figures")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

from pyCICY import chair44_nested as C

# -- style: the reference palette, as in make_sixsphere_figures.py -------------
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#b9b8b2", "#e6e5e0"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
RAMP = LinearSegmentedColormap.from_list("frac", ["#e1ebf7", "#9dc0ea", "#2a78d6", "#143f78"])

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK2, "text.color": INK,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 9.5, "axes.titleweight": "regular", "axes.titlelocation": "left",
    "savefig.dpi": 200,
})

HAT_BEST = 0.796        # AlpogeSixSphere.md, Part VII: best non-identity agreement at four steps


def _save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        p = os.path.join(OUT, "%s.%s" % (name, ext))
        fig.savefig(p, bbox_inches="tight", facecolor="white")
        print("wrote", os.path.relpath(p, os.path.dirname(HERE)))
    plt.close(fig)


def fig_defect(N=6, z=32):
    G = C.grid(N)
    side = G.shape[0]
    plane = G[:, :, z]
    assert (plane >= 0).all(), "the chosen slice should lie in the full bottom layer"
    fig, axes = plt.subplots(1, 4, figsize=(11.6, 3.35))

    # (a) the label field: colour = rotation index, nothing more
    ax = axes[0]
    rot = (plane // 7).T
    ax.imshow(rot, origin="lower", cmap="twilight", interpolation="nearest")
    ax.set_title("(a) the tiling, slice z = %d" % z)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.text(0.02, 0.97, "colour = rotation (24)", transform=ax.transAxes, va="top",
            fontsize=7.5, color="white", bbox=dict(fc="#0b0b0bAA", ec="none", pad=1.5))

    # (b-d) where the field disagrees with its translate by 2^M e_x
    table = C.coincidence_table(N)
    for ax, M in zip(axes[1:], (3, 4, 5)):
        v = (2 ** M, 0, 0)
        D = C.disagreement_map(G, v)[:, :, z]
        # overlap in this slice: x and x + 2^M both inside the patch
        overlap = np.zeros_like(D)
        overlap[: side - 2 ** M, :] = True
        count, total = int(D.sum()), int(overlap.sum())
        a_full, o_full = C.agreement(G, v)
        bound = 1 - table[M]["non_coincident"] / 8 ** M
        # the slice lies inside the patch; its disagreement share must respect the same proved bound
        assert (total - count) / total >= bound - 1e-9
        img = np.where(overlap.T, np.where(D.T, 2, 1), 0)
        ax.imshow(img, origin="lower", cmap=ListedColormap(["#ecebe6", "#ffffff", BLUE]),
                  vmin=0, vmax=2, interpolation="nearest")
        ax.set_title("(%s) translate by (%d, 0, 0)" % ("bcd"[M - 3], 2 ** M))
        ax.set_xlabel("x\n%.1f%% differ (proved at most %.1f%%)" % (100 * count / total, 100 * (1 - bound)))
        ax.set_yticks([])
    for ax in axes:
        ax.set_xticks([0, 32, 64, 96, 128] if ax is axes[0] else [0, 32, 64, 96, 128])
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    axes[0].set_yticks([0, 32, 64, 96, 128])
    fig.text(0.5, -0.04,
             "Blue: cells whose label differs from the label 2^M cells away along x.  Grey: no partner in the patch.  "
             "The disagreement set thins by about a factor four each step.",
             ha="center", fontsize=8, color=INK2)
    fig.tight_layout()
    _save(fig, "c44_nested_defect")


def fig_curve(N=6, scan_N=5):
    na = C.nested_agreement(N)
    xs = [r for r in na["rows"] if r["v"][1:] == (0, 0)]
    Ms = [r["M"] for r in xs]
    defect = [1 - r["fraction"] for r in xs]
    bound = [1 - float(r["bound"]) for r in xs]
    assert all(d <= b + 1e-12 for d, b in zip(defect, bound))
    ts = C.translate_scan(scan_N)
    table = C.coincidence_table(12)

    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.4))

    # (a) the defect against the proved bound
    ax = axes[0, 0]
    Mgrid = np.arange(1, 9)
    ax.semilogy(Ms, defect, "o-", color=BLUE, lw=2, ms=7, label="measured, level-%d patch" % N, zorder=3)
    ax.semilogy(Mgrid, [min(1.0, table[m]["non_coincident"] / 8 ** m) for m in Mgrid], "s--", color=ORANGE,
                lw=1.6, ms=6, mfc="white", label=r"proved bound $c(M)/8^M$")
    ax.semilogy(Mgrid, 28.0 / 4.0 ** Mgrid, ":", color=INK2, lw=1.4, label=r"$28\cdot 4^{-M}$")
    ax.axhline(1 - HAT_BEST, color=MUTED, lw=1.6)
    ax.text(8.05, 1 - HAT_BEST, "reference level, not a series:\nHat, best non-identity\ntranslate at 4 steps\n(Part VII, other footing)",
            va="center", fontsize=7.5, color=INK2)
    ax.set_xlim(0.7, 8.0); ax.set_ylim(2e-4, 1.4)
    ax.set_xlabel("M   (translation by $2^M e_x$, torus index $8^M$)")
    ax.set_ylabel("share of cells that disagree")
    ax.set_title("(a) disagreement with the translate")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.grid(color=GRID, lw=0.6)

    # (b) agreement with every in-plane translate
    ax = axes[0, 1]
    r = ts["radius"]
    S = ts["slice_z0"]
    quality = np.where(S > 0, -np.log10(np.maximum(1.0 - S, 1e-3)), np.nan)
    xs_, ys_ = np.meshgrid(np.arange(-r, r + 1), np.arange(-r, r + 1), indexing="ij")
    ok = np.isfinite(quality)
    order = np.argsort(quality[ok])                      # strongest drawn last, on top
    im = ax.scatter(xs_[ok][order], ys_[ok][order], c=quality[ok][order], cmap=RAMP, vmin=0, vmax=2.2,
                    s=16, marker="s", linewidths=0)
    ax.set_xlim(-r - 1, r + 1); ax.set_ylim(-r - 1, r + 1); ax.set_aspect("equal")
    for g in (-32, 32):
        ax.axvline(g, color=GRID, lw=0.8, zorder=0); ax.axhline(g, color=GRID, lw=0.8, zorder=0)
    ax.set_xlabel("$v_x$"); ax.set_ylabel("$v_y$")
    ax.set_title("(b) how nearly the translate $(v_x, v_y, 0)$ is a period")
    cb = fig_colorbar(ax, im)
    cb.set_label(r"$-\log_{10}$(share that disagree);  2 = 99% agree", fontsize=8)
    ax.text(0.02, 0.02, "level-%d patch; blank = no agreement, or overlap under 1/8" % scan_N,
            transform=ax.transAxes, fontsize=7.5, color=INK2)

    # (c) 2-adic valuation of the near-periods
    ax = axes[1, 0]
    ks = [3, 4, 5]
    width = 0.26
    for j, (thr, col) in enumerate(zip((0.5, 0.9, 0.99), (MUTED, BLUE, ORANGE))):
        bv = ts["by_threshold"][thr]["by_valuation"]
        ax.bar(np.array([0, 1, 2, 3, 4, 5]) + (j - 1) * width,
               [bv.get(k, 0) for k in range(6)], width * 0.92, color=col,
               label="agreement $\\geq$ %s" % thr)
    ax.set_yscale("symlog", linthresh=1)
    ax.set_xticks(range(6))
    ax.set_xlabel("2-adic valuation of the translation vector")
    ax.set_ylabel("number of translates")
    ax.set_title("(c) near-periods are $2^M$-divisible")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.grid(color=GRID, lw=0.6, axis="y")
    assert min(ts["by_threshold"][0.9]["by_valuation"]) >= 3

    # (d) the undetermined addresses against the volume
    ax = axes[1, 1]
    Mx = np.arange(1, 13)
    c = [table[m]["non_coincident"] for m in Mx]
    E = [table[m]["excess"] for m in Mx]
    assert all(c[i] == 28 * 2 ** int(m) - 48 and E[i] == 140 * 2 ** int(m) + 48 for i, m in enumerate(Mx))
    ax.semilogy(Mx, [8.0 ** m for m in Mx], ":", color=MUTED, lw=1.8, label="all addresses, $8^M$")
    ax.semilogy(Mx, E, "s-", color=ORANGE, lw=1.8, ms=5, label="label excess $E(M)=140\\cdot2^M+48$")
    ax.semilogy(Mx, c, "o-", color=BLUE, lw=1.8, ms=5, label="undetermined $c(M)=28\\cdot2^M-48$")
    ax.set_xlabel("M"); ax.set_ylabel("addresses in the $2^M$ cube")
    ax.set_title("(d) the undetermined set has dimension 1")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(color=GRID, lw=0.6)

    fig.tight_layout()
    _save(fig, "c44_nested_curve")


def fig_colorbar(ax, im):
    return plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)


if __name__ == "__main__":
    which = sys.argv[1:] or ["defect", "curve"]
    if "defect" in which:
        fig_defect()
    if "curve" in which:
        fig_curve()
