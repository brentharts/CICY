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


def main():
    for section in (test_identity, test_float_routes, test_nariai, test_jt,
                    test_controls):
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
