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

Run with:  python3 tests/test_firstlaw.py   or   python3 run_tests.py
"""

import math
import os
import random
import sys
from fractions import Fraction

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


def main():
    for section in (test_identity, test_float_routes, test_nariai, test_jt,
                    test_controls, test_fr, test_fr_controls):
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
