#!/usr/bin/env python3
"""
Generate figures/cosmo_*.pdf and figures/cosmology_facts.tex for paper/supplementary_material_cosmology.tex.

Every number quoted in that supplement is written here from the pyCICY modules, so the prose and the computations
cannot drift apart.  Three figures:

  cosmo_horizons.pdf   the horizons of the pancake (fibre: logarithmic; base: finite) and of Mixmaster
                       (finite on every axis through 40 epochs of the BKL map), rebuilt from pyCICY.horizons;
  cosmo_kasner.pdf     the Kasner exponents against u, with the horizon threshold p = 1;
  cosmo_moduli.pdf     the moduli-driven exponent budget: circles of exponents in the plane sum p = 1 for several
                       |s|^2, and the conserved |s|^2 along a numerical geodesic of the bicubic.

Run:  python3 paper/make_cosmology_figures.py     (from the repository root)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pyCICY import kasner as kn
from pyCICY import pancake as pk
from pyCICY import horizons as hz
from pyCICY import moduli_kasner as mk
from pyCICY.theories import cosmology as C

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)
FACTS = {}


def fact(name, value, fmt="%.4g"):
    FACTS[name] = value if isinstance(value, str) else fmt % value


def fig_horizons():
    sol = pk.pancake_solution(1.0, 3.0, 2.0, 0.5, t_max=6.0)
    h = hz.pancake_horizons(sol)
    summ = hz.summary()
    ent = C.MixmasterBKL().horizon(n_epochs=40)
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
    ax[0].semilogx(h["t"], h["fibre"], label="fibre (collapsing)")
    ax[0].semilogx(h["t"], h["base"], label="base")
    ax[0].axhline(np.pi, ls=":", c="gray")
    ax[0].set_xlabel("t"); ax[0].set_ylabel("comoving null angle"); ax[0].legend(frameon=False, fontsize=8)
    ax[0].set_title("pancake horizons", fontsize=9)
    ax[1].semilogx(h["t"], h["fibre_circumference"])
    ax[1].set_xlabel("t"); ax[1].set_ylabel("fibre circumference"); ax[1].set_title("fibre circumference (-> 0 at t = 0)", fontsize=9)
    ep = np.arange(1, len(ent["log10_h"]) + 1)
    for i in range(3):
        ax[2].plot(ep, [r[i] for r in ent["log10_h"]], label="axis %d" % i)
    ax[2].set_xlabel("Kasner epoch"); ax[2].set_ylabel(r"$\log_{10}$ cumulative horizon")
    ax[2].set_title("Mixmaster, u0 = pi", fontsize=9); ax[2].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "cosmo_horizons.pdf")); plt.close(fig)
    fact("PancakeContactTime", summ["t_contact_base"])
    fact("PancakePiBOverTwo", summ["pi_b0_over_2"])
    fact("PancakeFibreTurns", summ["fibre_turns_t1"])
    fact("PancakeTin", summ["t_in"], "%.1e")
    fact("MixmasterEpochs", str(len(ent["log10_h"])))
    fact("MixmasterLogHFinal", max(ent["log10_h"][-1]))
    fact("MixmasterLogHChange", max(abs(a - b) for a, b in zip(ent["log10_h"][-1], ent["log10_h"][-2])), "%.2e")


def fig_kasner():
    u = np.linspace(1.0, 6.0, 400)
    p = np.array([[float(x) for x in kn.kasner_exponents(v)] for v in u])
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    for i in range(3):
        ax.plot(u, p[:, i], label=r"$p_%d$" % (i + 1))
    ax.axhline(1.0, ls=":", c="gray"); ax.axhline(0.0, ls=":", c="gray")
    ax.set_xlabel("u"); ax.set_ylabel("Kasner exponent"); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "cosmo_kasner.pdf")); plt.close(fig)
    fact("KasnerMaxP", p.max())
    fact("KasnerMinP", p.min())


def fig_moduli():
    d = mk.triple_numbers([[2, 3], [2, 3]])
    th = np.linspace(0, 2 * np.pi, 400)
    # plane sum p = 1: basis e1 = (1,-1,0)/sqrt2, e2 = (1,1,-2)/sqrt6 about (1/3,1/3,1/3)
    e1, e2 = np.array([1, -1, 0]) / np.sqrt(2), np.array([1, 1, -2]) / np.sqrt(6)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    for s2 in (0, 1 / 6, 1 / 3, 1 / 2, 2 / 3):
        r = np.sqrt(max(2 / 3 - s2, 0))
        ax[0].plot(r * np.cos(th), r * np.sin(th), label=r"$|s|^2=%.3g$" % s2)
        if s2 == 0:
            pts = np.array([mk.exponents_from_speed2(0, t) for t in th])
            assert np.allclose(pts.sum(1), 1) and np.allclose((pts ** 2).sum(1), 1)
    ax[0].plot([0], [0], "k.")
    ax[0].set_aspect("equal"); ax[0].legend(frameon=False, fontsize=7)
    ax[0].set_title(r"exponents in the plane $\sum p=1$", fontsize=9)
    g = mk.moduli_geodesic(d, [1.0, 1.0], [0.3, -0.2], (0, 6), n=60)
    ax[1].plot(g["u"], g["speed2"] / g["speed2"][0])
    ax[1].plot(g["u"], g["t"][:, 0] / g["t"][0, 0], "--", label=r"$t^1$")
    ax[1].plot(g["u"], g["t"][:, 1] / g["t"][0, 1], ":", label=r"$t^2$")
    ax[1].set_xlabel(r"$u=\ln(t-t_s)$"); ax[1].legend(frameon=False, fontsize=8)
    ax[1].set_title(r"bicubic: $|s|^2$ conserved (solid)", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "cosmo_moduli.pdf")); plt.close(fig)
    fact("ModuliSpeedDrift", float(np.ptp(g["speed2"]) / g["speed2"][0]), "%.1e")
    # independent route
    s = np.array([0.2, 0.1]); s2 = float(s @ mk.moduli_metric(d, [1, 1]) @ s)
    p = mk.exponents_from_speed2(s2, 0.4)
    r = mk.einstein_scalar_check(d, [1, 1], s, p, t_end=50.0)
    fact("ELExponentDev", float(np.abs(r["p"] - np.array(p)).max()), "%.1e")
    fact("ELConstraint", abs(r["constraint"]), "%.1e")
    fact("DilationNorm", mk.speed2(d, [0.5, 1.1], [0.5, 1.1]), "%.12g")
    fact("QuinticMetricAtTwo", mk.moduli_metric(mk.triple_numbers([[4, 5]]), [2.0])[0, 0], "%.6g")


def main():
    fig_horizons(); fig_kasner(); fig_moduli()
    lines = ["% generated by paper/make_cosmology_figures.py -- do not edit"]
    for k, v in sorted(FACTS.items()):
        lines.append("\\providecommand{\\CF%s}{}\\renewcommand{\\CF%s}{%s}" % (k, k, v))
    with open(os.path.join(OUT, "cosmology_facts.tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
