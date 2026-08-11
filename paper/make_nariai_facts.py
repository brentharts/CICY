#!/usr/bin/env python3
"""
Generate figures/nariai_facts.tex and figures/nariai_ratio.pdf for
paper/supplementary_material_nariai.tex.

Every number quoted in that paper is written here, so the prose and the
computations cannot drift apart. Three packet families are pushed through the
negativity-ratio measurement:

  * the contour packet chi(u) = (s+iu)^{-2} of pyCICY.theories.nariai, whose
    phase distribution is exactly uniform (harmonic cancellation), so its
    ratio equals the phase-average closed form R(r) up to quadrature;
  * the VERBATIM Gaussian spectral packet of the source paper's companion
    script appendix_b_verification.py (w0 = 3, sigma = 0.7, same grid), so
    that the correspondence with the published numbers is nailed on a dense
    r grid rather than at the two points the paper quotes;
  * a low-winding Gaussian (w0 = 0.8), where the carrier turns over only a
    few cycles under the envelope, the higher circular moments are no longer
    negligible, and the measured ratio visibly departs from R(r) while still
    respecting the sharp bound e^{-2r}.

Run:  python3 paper/make_nariai_facts.py     (from the repository root)
"""

import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from pyCICY.theories import nariai as N

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)


# ---------------------------------------------------------------------------
# packet constructions
# ---------------------------------------------------------------------------

def gaussian_dchi(w0, sg, u):
    """The source paper's packet, construction copied from its script.

    chi(u) = int_0^inf dw a(w) e^{-i w u}, a(w) Gaussian at w0 of width sg,
    evaluated on the same frequency grid (0, 12] x 2400 as
    appendix_b_verification.py so that the w0 = 3 column reproduces the
    published numbers by construction.
    """
    w = np.linspace(1e-4, 12.0, 2400)
    aw = np.exp(-(w - w0) ** 2 / (2.0 * sg ** 2))
    coef = (-1j * w * aw) * (w[1] - w[0])
    out = np.empty(u.shape, dtype=complex)
    for i0 in range(0, len(u), 4000):
        blk = u[i0:i0 + 4000]
        out[i0:i0 + 4000] = np.exp(-1j * np.outer(blk, w)) @ coef
    return out


def ratio_from_dchi(dchi, u, r, theta):
    """D(W)/positive-part for a given horizon packet."""
    t = (2.0 * math.sinh(r) ** 2 * np.abs(dchi) ** 2
         - math.sinh(2.0 * r) * np.real(np.exp(-1j * theta) * dchi ** 2))
    pos = float(np.trapezoid(np.where(t > 0, t, 0.0), u))
    neg = -float(np.trapezoid(np.where(t < 0, t, 0.0), u))
    return neg / pos


# ---------------------------------------------------------------------------
# the survey
# ---------------------------------------------------------------------------

def survey():
    u = np.linspace(-40.0, 40.0, 32001)          # the script's grid
    thetas = [0.0, 1.0, math.pi / 2, math.pi]
    r_grid = [round(0.1 * k, 1) for k in range(1, 21)]        # 0.1 .. 2.0

    packets = {
        "contour": N.ContourPacket(s=0.8).dchi(u),
        "gauss3": gaussian_dchi(3.0, 0.7, u),
        "gauss08": gaussian_dchi(0.8, 0.7, u),
    }
    moments = {k: N.circular_moments(d, u, mmax=4)
               for k, d in packets.items()}

    dev = {k: [] for k in packets}
    bound_margin_min = math.inf
    rows = []
    for r in r_grid:
        want = N.universal_negativity_ratio(r)
        bound = N.negativity_bound(r)
        row = {"r": r, "R": want, "bound": bound}
        for k, d in packets.items():
            vals = [ratio_from_dchi(d, u, r, th) for th in thetas]
            m = max(abs(v - want) for v in vals)
            dev[k].append(m)
            row[k] = max(vals)
            worst = max(vals)
            if bound - worst < bound_margin_min:
                bound_margin_min = bound - worst
        rows.append(row)

    facts = {
        "RGridMin": r_grid[0], "RGridMax": r_grid[-1],
        "RGridN": len(r_grid), "ThetaN": len(thetas),
        "DevContour": max(dev["contour"]),
        "DevGaussThree": max(dev["gauss3"]),
        "DevGaussLow": max(dev["gauss08"]),
        "MomOneContour": moments["contour"][0],
        "MomTwoContour": moments["contour"][1],
        "MomOneGaussThree": moments["gauss3"][0],
        "MomTwoGaussThree": moments["gauss3"][1],
        "MomTwoGaussLow": moments["gauss08"][1],
        "BoundMarginMin": bound_margin_min,
        "RatioHalf": N.universal_negativity_ratio(0.5),
        "RatioOne": N.universal_negativity_ratio(1.0),
        "BoundHalf": N.negativity_bound(0.5),
        "BoundOne": N.negativity_bound(1.0),
    }

    # the four-routes table for the coherent sector
    a = 0.1
    routes = []
    for n in (1, 2, 3):
        c = N.coherent_entropy(n, a)
        q = N.coherent_entropy_quadrature(n, a)
        ar = N.coherent_entropy_araki(n, a)
        jt = N.jt_area_response(n, a)
        routes.append({"n": n, "closed": c["S_rel"], "quad": q, "araki": ar,
                       "jt_ratio": jt["ratio"],
                       "frac": str(c["longitudinal_integral"])})
    facts["RouteMaxQuadDev"] = max(
        abs(t["quad"] - t["closed"]) / t["closed"] for t in routes)
    facts["RouteMaxArakiDev"] = max(
        abs(t["araki"] - t["closed"]) / t["closed"] for t in routes)
    facts["RouteMaxJTDev"] = max(abs(t["jt_ratio"] - 1.0) for t in routes)

    # contour packet exact functionals + vanishing
    p = N.ContourPacket(s=0.8)
    i1n, i2n = p.i1_i2_numeric()
    facts["IOneExact"] = p.i1_exact()
    facts["ITwoRatio"] = abs(i2n) / p.i1_exact()
    facts["JacobianDev"] = p.boost_jacobian_check()["rel_diff"]

    # one-mode sum rule at general angle
    worst = 0.0
    for w_m, r in ((0.5, 0.3), (1.0, 0.8), (2.0, 1.5)):
        for th in (0.0, math.pi / 3, math.pi):
            m = N.one_mode_sum_rule(w_m, r, th)
            worst = max(worst,
                        abs(m["S_rel"] - m["closed_form"]),
                        abs(m["delta_S_vN"]))
    facts["OneModeDev"] = worst

    return facts, rows, dev, r_grid, moments


def write_facts(facts):
    def fmt(v):
        if isinstance(v, int):
            return str(v)
        if isinstance(v, float):
            if v == 0.0:
                return "0"
            if 0.01 <= abs(v) < 1e4:
                return "%.4g" % v
            return "\\ensuremath{%s}" % (
                "%.1e" % v).replace("e-0", "\\times 10^{-").replace(
                "e-", "\\times 10^{-").replace("e+0", "\\times 10^{").replace(
                "e+", "\\times 10^{") + "}"
        return str(v)

    lines = ["% generated by paper/make_nariai_facts.py -- do not edit",
             "% providecommand+renewcommand so this works whether or not",
             "% the fallback definitions in the .tex have been read first"]
    for k, v in sorted(facts.items()):
        lines.append("\\providecommand{\\NF%s}{}\\renewcommand{\\NF%s}{%s}"
                     % (k, k, fmt(v)))
    path = os.path.join(OUT, "nariai_facts.tex")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(OUT, "nariai_facts.json"), "w") as f:
        json.dump(facts, f, indent=1, default=str)
    print("wrote", path)


def figure(rows, dev, r_grid):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.6))

    rr = np.linspace(0.05, 2.0, 400)
    ax1.plot(rr, [N.universal_negativity_ratio(r) for r in rr], "k-",
             lw=1.4, label=r"phase average $\mathcal{R}(r)$")
    ax1.plot(rr, np.exp(-2.0 * rr), "k--", lw=1.1,
             label=r"sharp bound $e^{-2r}$")
    ax1.plot([row["r"] for row in rows], [row["contour"] for row in rows],
             "o", ms=4, mfc="none", label="contour packet")
    ax1.plot([row["r"] for row in rows], [row["gauss3"] for row in rows],
             "s", ms=4, mfc="none",
             label=r"paper's Gaussian ($\omega_0=3$)")
    ax1.plot([row["r"] for row in rows], [row["gauss08"] for row in rows],
             "^", ms=4, mfc="none",
             label=r"low-winding ($\omega_0=0.8$)")
    ax1.set_xlabel(r"squeeze parameter $r$")
    ax1.set_ylabel(r"$D(W)\,/\,$positive part")
    ax1.set_yscale("log")
    ax1.legend(fontsize=7.5, frameon=False)

    ax2.semilogy(r_grid, dev["contour"], "o-", ms=3.5, lw=0.8,
                 label="contour packet")
    ax2.semilogy(r_grid, dev["gauss3"], "s-", ms=3.5, lw=0.8,
                 label=r"Gaussian $\omega_0=3$")
    ax2.semilogy(r_grid, dev["gauss08"], "^-", ms=3.5, lw=0.8,
                 label=r"Gaussian $\omega_0=0.8$")
    ax2.set_xlabel(r"squeeze parameter $r$")
    ax2.set_ylabel(r"$\max_\theta\,|$ratio$-\mathcal{R}(r)|$")
    ax2.legend(fontsize=7.5, frameon=False)

    fig.tight_layout()
    path = os.path.join(OUT, "nariai_ratio.pdf")
    fig.savefig(path)
    print("wrote", path)


def main():
    facts, rows, dev, r_grid, moments = survey()
    write_facts(facts)
    try:
        figure(rows, dev, r_grid)
    except Exception as e:                                       # noqa: BLE001
        print("figure skipped:", e)
    print(json.dumps({k: ("%.3g" % v if isinstance(v, float) else v)
                      for k, v in facts.items()}, indent=1))


if __name__ == "__main__":
    main()
