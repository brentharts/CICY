#!/usr/bin/env python3
"""Verification suite for pyCICY.theories.spectre.

Sections
  1. the substitution matrix and its exact spectrum (with the rank
     correction: rank M = 5, rank M^2 = 4, not diagonalizable)
  2. unit arithmetic in Z[sqrt15]
  3. the Perron eigenvector, the multiplets, and their three mechanisms
  4. charges, census, ledger, Galois echo
  5. the frequency module generates Z[sqrt15] (with the corrected identity)
  6. exact geometry: the area form, the two-route census, det = -Q_-
  7. the order parameter in closed form; perturbation axes
  8. the knot layer: Alexander twice, signature, the unknotting sandwich,
     the mirror selection rule, the binding spectrum
  9. the dilation dictionary
 10. categories, refusals, registry
"""

import math
import os
import sys
import time
from fractions import Fraction as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from pyCICY.theories import spectre as S
from pyCICY.theories import get as get_theory
from pyCICY.theories.ftheory import NoSuchTheory

FAILURES = []


def check(label, got, want):
    ok = got == want
    if not ok:
        FAILURES.append("%s: got %r want %r" % (label, got, want))
    return ok


def check_true(label, cond):
    if not cond:
        FAILURES.append(label)
    return cond


def check_close(label, got, want, tol=1e-9):
    ok = abs(got - want) <= tol
    if not ok:
        FAILURES.append("%s: %r vs %r" % (label, got, want))
    return ok


def main():
    t0 = time.time()

    # ------------------------------------------------------------------
    # 1. matrix and spectrum
    # ------------------------------------------------------------------
    M = S.substitution_matrix()
    check("nine species", len(S.SPECIES), 9)
    idx = {s: i for i, s in enumerate(S.SPECIES)}
    for sp in ("Gamma", "Delta", "Sigma"):
        check("row %s is all ones" % sp, M[idx[sp]], [1] * 9)
    check_true("Phi row is NOT all ones (why Phi is outside the triplet)",
               M[idx["Phi"]] != [1] * 9)
    check("Theta row = Gamma indicator", M[idx["Theta"]],
          [1 if s == "Gamma" else 0 for s in S.SPECIES])
    check("Lambda row = Sigma indicator", M[idx["Lambda"]],
          [1 if s == "Sigma" else 0 for s in S.SPECIES])
    check("column sums are 7 or 8 (Gamma parent has 7 children)",
          sorted(set(sum(M[i][j] for i in range(9)) for j in range(9))),
          [7, 8])
    coeffs = S.charpoly()          # raises if it fails to factor
    check("charpoly degree", len(coeffs), 10)
    r = S.matrix_rank()
    check("rank M = 5 (correcting the source paper's 4)", r["rank"], 5)
    check("rank M^2 = 4", r["rank_squared"], 4)
    check("kernel dim 4, generalized kernel dim 5",
          (r["kernel_dim"], r["generalized_kernel_dim"]), (4, 5))
    check_true("M is not diagonalizable (one Jordan pair at zero)",
               not r["diagonalizable"])

    # ------------------------------------------------------------------
    # 2. unit arithmetic
    # ------------------------------------------------------------------
    d = S.inflation_data()
    check("Pell fundamental (4,1)", d["pell"], (4, 1))
    check_close("lambda = (sqrt6+sqrt10)/2", d["lambda"],
                d["lambda_check"], 1e-14)
    check("hexagonal channel: (lambda - 1/lambda)^2 = 6",
          d["hexagonal_channel_sq"], 6)
    check("pentagonal channel: (lambda + 1/lambda)^2 = 10",
          d["pentagonal_channel_sq"], 10)
    check("minimal polynomial x^4 - 8 x^2 + 1", d["min_poly"],
          (1, 0, -8, 0, 1))
    lam2 = S.Quad(4, 1, 15)
    check_true("unit inverse exact",
               lam2 * S.Quad(4, -1, 15) == S.Quad(1, 0, 15))
    q = S.Quad(2, 3, 15) / S.Quad(1, 1, 15)
    check_true("field division exact",
               q * S.Quad(1, 1, 15) == S.Quad(2, 3, 15))

    # ------------------------------------------------------------------
    # 3. Perron eigenvector and mechanisms
    # ------------------------------------------------------------------
    v = S.perron_frequencies()     # internally asserts M v = lam^2 v etc.
    g = S.Quad(4, -1, 15)
    check_true("triplet at g", v["Gamma"] == v["Delta"] == v["Sigma"] == g)
    check_true("doublet Theta=Lambda at g^2",
               v["Theta"] == v["Lambda"] == g * g)
    check_true("accidental doublet Pi=Xi", v["Pi"] == v["Xi"])
    check_close("Phi level float", float(v["Phi"]), 0.2217668, 1e-6)
    mech = S.degeneracy_mechanisms()
    check_true("conservation rows", mech["conservation_rows_all_ones"])
    check_true("indicator rows", mech["theta_row_is_gamma_indicator"]
               and mech["lambda_row_is_sigma_indicator"])
    check("Aut(M) trivial", mech["automorphism_group_order"], 1)

    # ------------------------------------------------------------------
    # 4. charges, census, ledger, echo
    # ------------------------------------------------------------------
    S.charges()                    # raises unless u M = +-u exactly
    series = S.census_series(5)
    check("totals 9, 71, 559, 4401",
          [row["total"] for row in series[:4]], [9, 71, 559, 4401])
    check("N_Gamma = 1, 8, 63, 496",
          [row["N_Gamma"] for row in series[:4]], [1, 8, 63, 496])
    check_true("Q+ = -1 at every depth",
               all(row["Q_plus"] == -1 for row in series))
    check("Q- alternates +1, -1, ...",
          [row["Q_minus"] for row in series[:4]], [1, -1, 1, -1])
    check_true("conservation in the counts",
               all(row["conservation"] for row in series))
    check_true("generational ledger N_Theta(n) = N_Gamma(n-1)",
               all(row["ledger"] for row in series))
    echo = S.galois_echo()
    check_true("Galois echo contracts at lambda^-4 (within 20%)",
               all(abs(rr - echo["prediction"]) < 0.2 * echo["prediction"]
                   for rr in echo["ratios"][-2:]))

    # ------------------------------------------------------------------
    # 5. the frequency module
    # ------------------------------------------------------------------
    fm = S.frequency_module()      # asserts both identities internally
    check_true("span is all of Z[sqrt15]", fm["span_is_full_ring"])
    check("the corrected identity", fm["corrected_identity"],
          "1 - v_Gamma = sqrt15 - 3")
    # the paper's printed identity is off by 3: 4 - v_Gamma - 3 = 1 - v_G
    check_true("paper's printed form is not sqrt15 (off by 3)",
               S.Quad(1, 0, 15) - g != S.Quad(0, 1, 15))
    check_true("... but equals sqrt15 - 3",
               S.Quad(1, 0, 15) - g == S.Quad(-3, 1, 15))

    # ------------------------------------------------------------------
    # 6. exact geometry
    # ------------------------------------------------------------------
    f = S.tile_area_form()         # asserts coefficients and endpoints
    check_true("A(1,sqrt3) = 8 sqrt3", f["A(1,sqrt3)"] == S.Quad(0, 8, 3))
    check_true("A(sqrt3,1) = 10 sqrt3", f["A(sqrt3,1)"] == S.Quad(0, 10, 3))
    S.mystic_asymmetry(1, 2)       # asserts sqrt3 (b^2-a^2) exactly
    S.mystic_asymmetry(S.Quad(1, 0, 3), S.Quad(0, 1, 3))
    geo = S.geometric_census(4)
    check_true("geometric census matches matrix census (two routes)",
               all(row["matches_matrix_route"] for row in geo))
    check_true("every patch is single-handed (strict chirality census)",
               all(row["single_handed"] for row in geo))
    check("determinants alternate (-1)^depth",
          [row["det"] for row in geo], [-1.0, 1.0, -1.0, 1.0])
    check("depth-4 patch: 4401 tiles, zero mirror tiles",
          (geo[3]["total"], geo[3]["single_handed"]), (4401, True))
    check_true("det = -Q_- at every depth (the parity avatar, with phase)",
               all(row["det_equals_minus_Qminus"] for row in geo))

    # ------------------------------------------------------------------
    # 7. order parameter and perturbation axes
    # ------------------------------------------------------------------
    op = S.order_parameter()
    check_true("max splitting = 10g/(8+10g) exactly",
               op["max_splitting"] == (10 * g) / (S.Quad(8, 0, 15) + 10 * g))
    check_close("splitting decimal 0.13702 (paper measured 0.137)",
                op["max_splitting_float"], 0.1370166, 1e-6)
    pert = S.perturbation_susceptibilities(n_ensembles=120, seed=1)
    check_true("accidental doublet splits in every ensemble",
               pert["accidental_always_splits"])
    check_true("triplet rigid under conservation-preserving perturbations",
               pert["triplet_rigid"])
    check_true("hierarchy: accidental splits >> triplet",
               pert["accidental_median"] > 1e3 * pert["triplet_median"])

    # ------------------------------------------------------------------
    # 8. the knot layer
    # ------------------------------------------------------------------
    ms = S.mystic_saturation()
    check("14/14 slots mismatch under role swap", ms["mismatches"], 14)
    xc = S.x_charges()
    check_true("all X-charges even", xc["all_even"])
    check_true("binding levels {0,1,2,7}", xc["levels_are_0_1_2_7"])
    rho = S.x_density(F(58, 100))
    check_true("x-density lands in (0, 1)", 0 < float(rho) < 1)
    for k in (3, 5, 7, 9):
        t = S.torus_knot(k)        # asserts both routes internally
        check("sigma(T(2,%d)) = -(k-1)" % k, t["signature"], -(k - 1))
        check("u sandwich closes at (k-1)/2 for k=%d" % k,
              (t["u_lower_from_signature"],
               t["u_upper_from_crossing_induction"]),
              ((k - 1) // 2, (k - 1) // 2))
    t7 = S.torus_knot(7)
    check("same-handed sum: signature forces additivity",
          (t7["signature_same_handed_sum"], t7["u_same_handed_sum_forced"]),
          (-12, 6))
    check("mirror sum: signature bound lost", t7["signature_mirror_sum"], 0)
    check_true("BH deficit is at k = 7", t7["mirror_deficit_known"])
    b = S.binding_spectrum()
    check("additive content 42 per tile", b["additive_content"], 42)
    check("deficit levels 2 delta {0,1,2,7}",
          [x // 2 for x in b["deficit_levels"]], [0, 1, 2, 7])
    check("mystic is the maximal binder", b["masses"]["Gamma2"], 42 - 14)
    check_true("sterile quartet + Phi0 cannot bind",
               set(b["sterile"]) == {"Theta", "Pi", "Xi", "Psi", "Phi0"})
    check("alexander of 7_1 alternates", t7["alexander"],
          [1, -1, 1, -1, 1, -1, 1])

    # ------------------------------------------------------------------
    # 9. the dictionary
    # ------------------------------------------------------------------
    dd = S.dilation_dictionary()
    check_close("tau_* = log lambda / 2 pi", dd["tau_star"], 0.1642031,
                1e-6)
    check_true("register identity e^{4 pi tau} = lambda^2",
               dd["register_identity"])
    check_close("KMS strip holds 6.0885 steps", dd["kms_steps_per_strip"],
                6.0885, 2e-3)
    check_close("log-period log(4+sqrt15)", dd["log_period"],
                math.log(4 + math.sqrt(15)), 1e-12)

    # ------------------------------------------------------------------
    # 10. categories, refusals, registry
    # ------------------------------------------------------------------
    th = get_theory("spectre-substrate")()
    check("registry key resolves", th.key, "spectre-substrate")
    sp = th.spectrum()
    check("five zero modes in the spectrum dict", sp["zero_modes"], 5)
    check("multiplicities (1,1,3,2,2)", sp["multiplicities"],
          (1, 1, 3, 2, 2))
    for meth in ("band_structure", "crossing_change", "holomorphic_yukawa"):
        try:
            getattr(th, meth)()
            check_true("%s refuses" % meth, False)
        except NoSuchTheory:
            check_true("%s raises NoSuchTheory (not a metric problem)"
                       % meth, True)
        except Exception:
            check_true("%s raises the *right* exception" % meth, False)
    missing = th.missing_for_physical()
    check_true("open problems listed (p, delta, intertwiner, complexity)",
               len(missing) >= 4)
    check_true("describe() runs", len(th.describe()) > 100)

    dt = time.time() - t0
    if FAILURES:
        print("test_spectre: %d FAILURES in %.1fs" % (len(FAILURES), dt))
        for msg in FAILURES:
            print("  FAIL:", msg)
        sys.exit(1)
    print("test_spectre: all checks passed in %.1fs" % dt)


if __name__ == "__main__":
    main()
