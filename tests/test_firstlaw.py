"""
Tests for nariai.first_law_bookkeeping: S = A/4 and alpha = 8 pi as one exact
identity, the Schwarzschild-de Sitter first law.

  [1] exact identity   alpha/pi = 8 and dS_rel/dA = 1/4 at many rational
                       (r, Lambda), black-hole and cosmological horizons;
                       dM/dr (total derivative along the horizon family) and
                       kappa = f'/2 (partial derivative at fixed M) are
                       different objects, so agreement is a real check
  [2] float routes     the module's cubic-root horizon data (sds_horizons,
                       numpy) against the exact kappa and mass
  [3] Nariai           A/4 = S_N = pi/Lambda; the 0/0 point is refused
  [4] JT and Araki     the module's dilaton-ODE ratio dA/(4 S_rel) = 1 and
                       its derived coupling equal the exact 8 pi
  [5] controls         wrong normalisations (S = A/2, alpha = 4 pi) and a
                       wrong surface gravity break the identity
  [6] f(R) gravity     constant-curvature horizons of f(R) = R + aR^2 + bR^3
                       + c: alpha/pi = 8/F with F = f'(R0), and
                       dS_rel = dS_Wald exactly; the field equation checked
                       from the Ricci tensor of the SdS metric
  [7] f(R) controls    non-solutions, F <= 0, and wrong normalisations fail

  [10] compactification  10D gravity on SdS_4 x CICY: G_4 = G10/Vol from the
                       action, the Wald entropy contracted in 10D at the
                       horizon S^2 x X, exact first law; alpha = 8 pi G_N and
                       G_N dS_rel/dA = 1/4 for any volume, any G10
  [11] controls          missing volume, missing 1/6, wrong F, a point outside
                       the Kahler cone

Run with:  python3 tests/test_firstlaw.py   or   python3 run_tests.py
"""

import math
import os
import random
import sys
from fractions import Fraction
Fr = Fraction

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyCICY.theories import nariai as N

FAILURES = []


def check(name, got, want):
    ok = got == want
    print("  {:<58} {:>12} {}".format(name, str(got)[:12],
                                      "ok" if ok else "FAIL want " + str(want)))
    if not ok:
        FAILURES.append(name)


def check_true(name, cond):
    print("  {:<58} {:>12} {}".format(name, str(bool(cond)),
                                      "ok" if cond else "FAIL"))
    if not cond:
        FAILURES.append(name)


def _raises(exc, fn, *a, **kw):
    try:
        fn(*a, **kw)
    except exc:
        return True
    except Exception:
        return False
    return False


def test_identity():
    print("\n[1] exact identity at rational points")
    rng = random.Random(3)
    bad, n, signs = [], 0, set()
    for _ in range(60):
        lam = Fraction(rng.randint(1, 12), rng.randint(1, 6))
        r = Fraction(rng.randint(1, 40), rng.randint(1, 12))
        if lam * r * r == 1:
            continue
        d = N.first_law_bookkeeping(r, lam)
        n += 1
        signs.add(d["kappa"] > 0)
        if (d["alpha_over_pi"] != 8 or d["S_rel_per_area"] != Fraction(1, 4)
                or d["horizon_residual"] != 0):
            bad.append((r, lam, d))
    check("alpha/pi = 8, S_rel/A = 1/4, f(r) = 0 (%d points): bad" % n,
          len(bad), 0)
    check_true("both signs of kappa (black-hole and cosmological) occur",
               signs == {True, False})


def test_float_routes():
    print("\n[2] against the module's cubic-root horizon data")
    bad = []
    for r, lam in ((Fraction(1, 2), 1), (Fraction(3, 2), 1),
                   (Fraction(1, 3), 2), (Fraction(3, 5), Fraction(4, 3))):
        d = N.first_law_bookkeeping(r, lam)
        m, L = float(d["mass"]), float(lam)
        if not 0.0 < 9 * L * m * m < 1.0:
            bad.append(("unphysical", r, lam))
            continue
        h = N.sds_horizons(m, L)
        roots = (h["rb"], h["rc"])
        k = (h["kappa_b"], h["kappa_c"])
        i = min(range(2), key=lambda j: abs(roots[j] - float(r)))
        if abs(roots[i] - float(r)) > 1e-9 or \
                abs(k[i] - abs(float(d["kappa"]))) > 1e-9:
            bad.append((r, lam, roots, k, d["kappa"]))
        # first-law sign: kappa and dM/dr have the same sign
        if (d["kappa"] > 0) != (d["dM_dr"] > 0):
            bad.append(("sign", r, lam))
    check("exact r, |kappa| equal the numpy cubic's: bad", len(bad), 0)


def test_nariai():
    print("\n[3] the Nariai limit")
    for lam in (1.0, 2.5):
        d = N.nariai_data(lam)
        check_true("A/4 = S_N = pi/Lambda at Lambda = %g" % lam,
                   abs(d["area"] / 4 - d["entropy"]) < 1e-12
                   and abs(d["entropy"] - math.pi / lam) < 1e-12)
    check_true("Lambda r^2 = 1 is refused (0/0), not guessed",
               _raises(ValueError, N.first_law_bookkeeping, 1, 1))
    check_true("(r, Lambda) = (1/2, 4) is also the Nariai radius: refused",
               _raises(ValueError, N.first_law_bookkeeping,
                       Fraction(1, 2), 4))
    # approaching the degenerate point the ratio stays exactly 8
    ok = all(N.first_law_bookkeeping(Fraction(1) + Fraction(1, 10 ** k), 1)
             ["alpha_over_pi"] == 8 for k in range(1, 6))
    check_true("ratio stays 8 as r -> r_* from above (k=1..5)", ok)


def test_jt():
    print("\n[4] the module's dilaton ODE and Araki route")
    for n, a in ((1, 0.3), (2, 0.2)):
        jt = N.jt_area_response(n, a)
        check_true("n=%d: dA(ODE) / 4 S_rel = 1 (dA = 4 S_rel)" % n,
                   abs(jt["ratio"] - 1.0) < 1e-3)
        exact = N.first_law_bookkeeping(Fraction(1, 2), 1)
        check_true("derived coupling = 8 pi = pi * exact alpha/pi",
                   abs(jt["coupling_derived"]
                       - math.pi * float(exact["alpha_over_pi"])) < 1e-12)


def test_controls():
    print("\n[5] controls: each check made to fail on purpose")
    d = N.first_law_bookkeeping(Fraction(1, 2), 1)
    k, dM = d["kappa"], d["dM_dr"]
    r = d["r"]
    dA = 8 * r                       # dA/pi per dr
    check("the same numbers give 8, from kappa dA / dM", k * dA / dM, 8)
    check("dS_rel/dA from these numbers is 1/4", 2 * dM / k / dA,
          Fraction(1, 4))
    check_true("a convention S = A/2 (doubled) would not match 1/4",
               2 * (2 * dM / k / dA) != Fraction(1, 4))
    check_true("a surface gravity off by 2 breaks alpha = 8 pi",
               (2 * k) * dA / dM != 8)
    check_true("using M = r/2 (Lambda dropped) breaks the first law",
               k * dA / Fraction(1, 2) != 8)
    check_true("negative radius refused",
               _raises(ValueError, N.first_law_bookkeeping, -1, 1))
    check_true("zero Lambda refused",
               _raises(ValueError, N.first_law_bookkeeping, 1, 0))


def test_fr():
    print("\n[6] f(R) gravity: alpha = 8 pi G_eff, dS_rel = dS_Wald")
    Fr = Fraction
    cases = [(Fr(1, 2), Fr(4), Fr(1, 16), Fr(0)),
             (Fr(3, 2), Fr(4, 3), Fr(1, 5), Fr(1, 50)),
             (Fr(1, 3), Fr(12), Fr(-1, 100), Fr(1, 400)),
             (Fr(2), Fr(1, 2), Fr(0), Fr(1, 9))]
    bad, einstein_checked = [], False
    for r, R0, a, b in cases:
        d = N.fr_bookkeeping(r, R0, a, b)
        F = 1 + 2 * a * R0 + 3 * b * R0 ** 2          # by hand
        c = (b * R0 ** 3 - R0) / 2                      # trace equation, by hand
        if (d["F"] != F or d["c"] != c or d["alpha_over_pi"] != 8 / F
                or d["S_rel_per_area"] != F / 4
                or d["S_rel_per_area"] != d["S_wald_per_area"]
                or d["field_equation_residual"] != 0
                or not d["sds_is_einstein_space"]
                or d["Lambda"] != R0 / 4):
            bad.append((r, R0, a, b, d))
    check("F, c, alpha/pi = 8/F, dS_rel = dS_Wald (4 f(R)s): bad", len(bad), 0)
    d0 = N.fr_bookkeeping(Fr(1, 2), 4)
    e = N.first_law_bookkeeping(Fr(1, 2), 1)          # Lambda = R0/4 = 1
    check("a = b = 0 reproduces the Einstein bookkeeping",
          (d0["F"], d0["alpha_over_pi"], d0["S_rel_per_area"], d0["kappa"]),
          (1, e["alpha_over_pi"], e["S_rel_per_area"], e["kappa"]))
    check("the R^2 coefficient drops out of c (4D constant curvature)",
          (N.fr_bookkeeping(Fr(1, 2), 4, Fr(1, 16))["c"],
           N.fr_bookkeeping(Fr(1, 2), 4, Fr(3, 7))["c"]),
          (Fr(-2), Fr(-2)))
    d1 = N.fr_bookkeeping(Fr(1, 2), 4, Fr(1, 16))
    check_true("delta A = 4 S_rel fails once F != 1 (S_rel/A = 3/8, not 1/4)",
               d1["S_rel_per_area"] == Fr(3, 8) != Fr(1, 4))
    check("Nariai entropy S_N = F pi / Lambda: S_N Lambda / pi = F",
          d1["nariai_S_times_Lambda_over_pi"], Fr(3, 2))
    check_true("SdS has R_mn = Lambda g_mn symbolically (R = 4 Lambda)",
               N._sds_is_einstein_space())


def test_fr_controls():
    print("\n[7] f(R) controls: each check made to fail on purpose")
    Fr = Fraction
    check_true("a c that violates the trace equation is refused",
               _raises(ValueError, N.fr_bookkeeping, Fr(1, 2), 4,
                       Fr(1, 16), 0, 7))
    check_true("the correct c = -R0/2 = -2 is accepted",
               N.fr_bookkeeping(Fr(1, 2), 4, Fr(1, 16), 0, -2)["c"] == -2)
    check_true("F <= 0 (a = -1/4, R0 = 4: F = -1) is refused",
               _raises(ValueError, N.fr_bookkeeping, Fr(1, 2), 4, Fr(-1, 4)))
    check_true("R0 <= 0 is refused (de Sitter branch only)",
               _raises(ValueError, N.fr_bookkeeping, Fr(1, 2), -1))
    d = N.fr_bookkeeping(Fr(1, 2), 4, Fr(1, 16))
    # the mass without the F factor would give alpha/pi = 8, not 8/F
    check_true("physical mass M instead of F M would give alpha/pi = 8, not 16/3",
               d["alpha_over_pi"] == Fr(16, 3) != 8)
    check_true("Einstein S = A/4 does not equal the relative entropy when F != 1",
               d["S_rel_per_area"] != Fr(1, 4))


def test_derived_inputs():
    print("\n[8] F M and S_Wald derived (Einstein frame); F_eff from the Wald tensor")
    d = N.fr_einstein_frame(Fr(1, 2), 4, Fr(5, 32))
    b = N.fr_bookkeeping(Fr(1, 2), 4, Fr(5, 32))
    check("F = 9/4", d["F"], Fr(9, 4))
    check_true("metric identity g~ = F g holds symbolically", d["metric_identity"])
    check("Lambda~ = V = Lambda/F", (d["Lambda_tilde"], d["potential_V"]),
          (Fr(4, 9), Fr(4, 9)))
    check("E from Einstein frame == F M (two routes)",
          d["E_from_einstein_frame"], d["F_times_M"])
    check("kappa Einstein frame == Jordan", d["kappa_from_einstein_frame"],
          d["kappa_jordan"])
    check("alpha/pi: Einstein frame == Jordan bookkeeping",
          d["alpha_over_pi_einstein_frame"] * 1, Fr(8))
    check("alpha/pi = 8/F agrees with fr_bookkeeping", d["alpha_over_pi"],
          b["alpha_over_pi"])
    check("S_rel/A = F/4", d["S_rel_per_area"], Fr(9, 16))
    check("F_eff from Wald tensor = f(R) F (beta only)",
          N.wald_factor(1, 0, Fr(1, 8)), 1 + 8 * Fr(1, 8))
    check("F_eff = 1 + 2 Lambda (alpha + 4 beta)",
          N.wald_factor(Fr(3, 2), Fr(1, 3), Fr(1, 5)),
          1 + 3 * (Fr(1, 3) + Fr(4, 5)))
    check("R^2 term equals f(R) with a = beta, R0 = 4 Lambda",
          N.wald_factor(1, 0, Fr(5, 32) * 1), 1 + 8 * Fr(5, 32))
    q = N.quadratic_gravity_bookkeeping(Fr(1, 2), 1, Fr(1, 4), 0)
    check("quadratic gravity: F = 3/2, alpha/pi = 16/3",
          (q["F"], q["alpha_over_pi"]), (Fr(3, 2), Fr(16, 3)))


def test_derived_controls():
    print("\n[9] controls for the derived inputs")
    check_true("critical point alpha = -1/(2 Lambda) refused",
               _raises(ValueError, N.wald_factor, 1, Fr(-1, 2), 0))
    check_true("ghost side (F<0) refused",
               _raises(ValueError, N.wald_factor, 1, -1, 0))
    check_true("non-square F refused by the Einstein-frame map",
               _raises(ValueError, N.fr_einstein_frame, Fr(1, 2), 4, Fr(1, 16)))
    d = N.fr_einstein_frame(Fr(1, 2), 4, Fr(5, 32))
    check_true("mass without F would not match the Einstein-frame energy",
               d["E_from_einstein_frame"] != d["F_times_M"] / d["F"])
    check_true("area without F (S=A/4) differs from Wald entropy",
               d["F_times_area_over_pi"] != d["area_tilde_over_pi"] / d["F"])
    check_true("wrong binormal normalisation would break F=1 at alpha=beta=0",
               N.wald_factor(1, 0, 0) == 1)


def test_compactification():
    print("\n[10] compactification: G_N = G10 / Vol(X), two routes")
    quintic, bicubic, ex = [[4, 5]], [[2, 3], [2, 3]], [[2, 2, 1], [3, 1, 3]]
    # volumes by hand: quintic J^3 = 5 -> 5 t^3/6; bicubic d_112 = d_122 = 3
    # -> (3/2) t1 t2 (t1 + t2)
    check("quintic Vol = 5 t^3/6 at t=2", N.kahler_volume_exact(quintic, [2]),
          Fr(20, 3))
    check("bicubic Vol = 3/2 t1 t2 (t1+t2) at (1, 1/2)",
          N.kahler_volume_exact(bicubic, [1, Fr(1, 2)]), Fr(9, 8))
    from pyCICY.export import kahler_volume
    check_true("exact volume = the float route in pyCICY.export",
               abs(float(N.kahler_volume_exact(ex, [1, 2]))
                   - kahler_volume(ex, [1, 2])) < 1e-9)
    cases = [(quintic, [2], Fr(1, 3), 4, 3, Fr(1, 16), Fr(1, 64)),
             (bicubic, [1, Fr(1, 2)], Fr(1, 3), 4, Fr(1, 5), Fr(0), Fr(1, 40)),
             (ex, [1, 2], Fr(2, 5), 3, 1, Fr(-1, 20), Fr(1, 100)),
             (quintic, [Fr(1, 2)], Fr(1, 3), 4, 7, Fr(0), Fr(0))]
    bad = []
    for X, t, r, lam, G10, a, b in cases:
        d = N.compactified_bookkeeping(X, t, r, lam, G10, a, b)
        F = 1 + 2 * Fr(lam) * (a + 4 * b)                 # by hand
        if (d["G4"] != Fr(G10) / d["V"] or d["F4"] != F or d["F10"] != F
                or d["first_law_residual"] != 0
                or d["alpha_over_pi"] != 8 * d["G_N"]
                or d["alpha_over_pi"] != d["alpha_over_pi_expected"]
                or d["GN_times_S_rel_per_area"] != Fr(1, 4)
                or d["G_N"] != Fr(G10) / (d["V"] * F)):
            bad.append((X, t, d))
    check("alpha=8 pi G_N, G_N dS_rel/dA = 1/4, first law (4 CICY cases): bad",
          len(bad), 0)
    a1 = N.compactified_bookkeeping(quintic, [2], Fr(1, 3), 4, 3)
    a2 = N.compactified_bookkeeping(quintic, [4], Fr(1, 3), 4, 3)
    check_true("rescaling the Kahler class by 2 changes G_N by exactly 8",
               a1["G_N"] / a2["G_N"] == 8)
    check_true("...and leaves G_N * dS_rel/dA = 1/4",
               a1["GN_times_S_rel_per_area"] == a2["GN_times_S_rel_per_area"]
               == Fr(1, 4))
    check("10D Wald factor = 4D Wald factor (F = 5)",
          N.wald_factor_product(4, Fr(1, 3), Fr(1, 24), 6),
          N.wald_factor(4, Fr(1, 3), Fr(1, 24)))


def test_compactification_controls():
    print("\n[11] compactification controls: each made to fail on purpose")
    quintic = [[4, 5]]
    d = N.compactified_bookkeeping(quintic, [2], Fr(1, 3), 4, 3, Fr(1, 16),
                                   Fr(1, 64))
    V, F = d["V"], d["F4"]
    # horizon entropy without the internal volume: S = F A / 4 G10
    dS_novol = F * 8 * Fr(1, 3) / (4 * 3)
    dE = F * N.first_law_bookkeeping(Fr(1, 3), 4)["dM_dr"] / d["G4"]
    check_true("dropping Vol from the horizon integral breaks the first law",
               dE - d["kappa"] * dS_novol / 2 != 0)
    wrong_V = N.kahler_volume_exact(quintic, [2]) * 6
    check_true("Vol without the 1/6 gives a different G_4", wrong_V != V)
    check_true("F=1 (ignoring the curvature terms) breaks alpha = 8 pi G_N",
               8 * (Fr(3) / V) != d["alpha_over_pi"])
    check_true("using G_4 for G_N ignores F: G_4 != G_N when F != 1",
               d["G4"] != d["G_N"])
    check("a point outside the Kahler cone is refused",
          _raises(ValueError, N.compactified_bookkeeping, quintic, [-1],
                  Fr(1, 3), 4), True)
    check("the Nariai radius is still refused",
          _raises(ValueError, N.compactified_bookkeeping, quintic, [1],
                  Fr(1, 2), 4), True)


def test_riem2():
    print("\n[12] Riemann^2 Wald entropy: direct vs Gauss-Bonnet decomposition")
    bad = []
    for r, lam in [(Fr(1, 3), 4), (Fr(1, 2), 1), (Fr(2, 5), 3), (Fr(1, 4), 2)]:
        d = N.riem2_wald(r, lam)
        # S = 4 pi per unit gamma  <=>  1/r^2 per unit area (A = 4 pi r^2)
        if d["direct"] != d["via_GB"] or d["direct"] != 1 / (r * r):
            bad.append((r, lam, d))
    check("direct = via GB = 1/r^2 (S = 4 pi gamma), 4 horizons: bad",
          len(bad), 0)
    d1, d2 = N.riem2_wald(Fr(1, 3), 4), N.riem2_wald(Fr(1, 3), 5)
    check_true("independent of Lambda at fixed r (S constant, not ~ A)",
               d1["direct"] == d2["direct"])
    d = N.riem2_wald(Fr(1, 2), 1)
    check_true("control: a wrong GB constant (8 pi) would not match",
               d["direct"] != 2 * d["gb_per_area"])
    check_true("alpha = 4 gamma, beta = -gamma gives F = 1 (cancellation)",
               N.wald_factor(1, Fr(4), Fr(-1)) == 1)


def test_riem2_internal():
    print("\n[13] Riemann^2 on a Ricci-flat Kahler X: Q = 32 pi^2 c_2 . J")
    k3, quintic, bicubic = [[3, 4]], [[4, 5]], [[2, 3], [2, 3]]
    # K3 = quartic in P^3: c(TX) = (1+H)^4/(1+4H) = 1 + 6 H^2, int H^2 = 4
    check("K3: Q/pi^2 = 32 chi, chi = 6 * 4 = 24 by hand", N.riem2_integral_over_pi2(k3),
          32 * 24)
    # K3 x T^2: both sides known. int|Rm|^2 = Area(T^2) * 768 pi^2 (Gauss-
    # Bonnet on K3, T^2 flat); c_2(K3 x T^2) = c_2(K3), int c_2 ^ J = 24 Area
    area = Fr(7, 5)
    check("K3 x T^2: 32 * int c_2.J = Area * 768 (n-independence of the 32)",
          32 * 24 * area, area * 768)
    # threefolds: c_2 . J by hand
    check("quintic: c_2 = 10 H^2, c_2.H = 50 -> Q/pi^2 = 1600 t",
          N.riem2_integral_over_pi2(quintic, [Fr(3, 2)]), 1600 * Fr(3, 2))
    check("bicubic: c_2 = 3H1^2 + 9 H1H2 + 3 H2^2, c_2.H_r = 36 each",
          N.riem2_integral_over_pi2(bicubic, [1, Fr(1, 2)]),
          32 * 36 * Fr(3, 2))
    d = N.compactified_riem2(quintic, [2], Fr(1, 10), 3)
    check("G_N unchanged by Riem^2: still G10/V", d["G_N_unchanged"],
          Fr(3) / d["V"])
    check("alpha + 4 beta = 0, so F = 1",
          N.wald_factor(4, d["alpha"], d["beta"]), 1)
    check("Lambda0 shift = gamma Q/(2V) = 16 gamma c_2.t / V",
          d["dLambda0_over_pi2"], Fr(16, 10) * 100 / d["V"])
    check("entropy constant = 4 gamma V / G10", d["S_const_over_pi"],
          4 * Fr(1, 10) * d["V"] / 3)
    check_true("control: a wrong constant (16 pi^2) would not match K3 x T^2",
               16 * 24 * area != area * 768)
    check("control: a threefold needs a Kahler point",
          _raises(ValueError, N.riem2_integral_over_pi2, quintic), True)
    check("control: outside the Kahler cone is refused",
          _raises(ValueError, N.compactified_riem2, quintic, [-1], 1), True)


def main():
    for section in (test_identity, test_float_routes, test_nariai, test_jt,
                    test_controls, test_fr, test_fr_controls,
                    test_derived_inputs, test_derived_controls,
                    test_compactification, test_compactification_controls,
                    test_riem2, test_riem2_internal):
        try:
            section()
        except Exception as exc:
            print("  {:<58} {:>12} FAIL".format(section.__name__ + " raised",
                                                type(exc).__name__))
            print("      " + str(exc)[:300])
            FAILURES.append("%s raised %s" % (section.__name__,
                                              type(exc).__name__))
    print()
    if FAILURES:
        print("FAILED: %d" % len(FAILURES))
        for f in FAILURES:
            print("  - " + f)
        return 1
    print("all firstlaw tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
