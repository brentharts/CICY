"""
Tests for pyCICY.enumerative.

Everything here is checked against a published value or against an existing,
independently implemented part of pyCICY:

  [1] Chern polynomial   vs CIPro section 2.3 (arXiv:2606.27588), dP2
  [2] Hilbert series     vs CIPro section 2.4, CICY 7821
  [3] GV invariants      vs the classical instanton numbers of the five
                            one-parameter models (Hosono, Klemm, Theisen, Yau)
  [4] cross-checks       Euler characteristic and c_2 against the routes
                            already in the package

Section [3] also checks integrality. The Gopakumar-Vafa invariants come out
of a rational power series computation with no reason to be integers unless
the computation is right, so integrality is a real test and not decoration.

Run with:  python3 tests/test_enumerative.py
       or: python3 run_tests.py  (runs every suite)
"""

import os
import sys
from fractions import Fraction

# Prefer the source tree over any installed copy of pyCICY.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyCICY import enumerative as E
from pyCICY import transitions as T

FAILURES = []


def check(name, got, want):
    ok = got == want
    print("  {:<56} {:>14} {}".format(name, str(got)[:14],
                                      "ok" if ok else "FAIL want " + str(want)))
    if not ok:
        FAILURES.append(name)


def check_true(name, cond):
    print("  {:<56} {:>14} {}".format(name, str(bool(cond)),
                                      "ok" if cond else "FAIL"))
    if not cond:
        FAILURES.append(name)


# --------------------------------------------------------------------------
print("\n[1] Chern polynomial (CIPro section 2.3, dP2)")
# CIPro prints 1 + J1 + J2 + 2 J1 J2 + J3 + 2 J1 J3 + J2 J3 for
# DimPs {2,1,1}, Conf {{1,1},{1,0},{0,1}}.
dp2 = [[2, 1, 1], [1, 1, 0], [1, 0, 1]]
c = E.chern_polynomial(dp2)
expected = {
    (0, 0, 0): 1,
    (1, 0, 0): 1, (0, 1, 0): 1, (0, 0, 1): 1,
    (1, 1, 0): 2, (1, 0, 1): 2, (0, 1, 1): 1,
}
check("term count", len(c), len(expected))
for e, v in sorted(expected.items()):
    check("coefficient of %s" % (e,), c.get(e), Fraction(v))
check_true("no J1^2 term (killed by J^{n+1} = 0 only for P^1)",
           (0, 2, 0) not in c and (0, 0, 2) not in c)

# c_0 = 1 and c_1 = 0 for a Calabi-Yau
quintic_c = E.chern_polynomial([[4, 5]])
check("quintic c_0", quintic_c[(0,)], Fraction(1))
check_true("quintic c_1 vanishes", quintic_c.get((1,), Fraction(0)) == 0)
check("quintic c_2", quintic_c[(2,)], Fraction(10))
# c_3 = -40 J^3; the Euler characteristic -200 is int_X c_3 = -40 * 5,
# the extra factor being the degree of the defining equation.
check("quintic c_3", quintic_c[(3,)], Fraction(-40))
check("chi = int_X c_3", E.euler_from_chern([[4, 5]]), -200)

# --------------------------------------------------------------------------
print("\n[2] Hilbert series (CIPro section 2.4, CICY 7821)")
# CIPro prints (1 - t1 t2)(1 - t1 t2^2)^2 / ((1 - t1)^3 (1 - t2)^5).
cicy7821 = [[2, 1, 1, 1], [4, 2, 2, 1]]
h = E.hilbert_series(cicy7821)
check("denominator exponents", h["denominator"], [3, 5])
check("numerator multidegrees", sorted(h["numerator"]),
      [(1, 1), (1, 2), (1, 2)])
check("rendered form", E.hilbert_series_str(cicy7821),
      "(1 - t1 t2)(1 - t1 t2^2)^2 / ((1 - t1)^3 (1 - t2)^5)")

# The quintic: (1 - t^5) / (1 - t)^5
check("quintic Hilbert series", E.hilbert_series_str([[4, 5]]),
      "(1 - t^5) / ((1 - t)^5)".replace("t^5", "t1^5").replace("(1 - t)",
                                                               "(1 - t1)"))

# Degree-1 piece of the quintic coordinate ring has the 5 homogeneous
# coordinates; degree 5 has binom(9,4) = 126 monomials minus the 1 relation.
coeffs = E.hilbert_coefficients([[4, 5]], 5)
check("quintic degree 1", coeffs[(1,)], 5)
check("quintic degree 2", coeffs[(2,)], 15)
check("quintic degree 5 (126 monomials - 1 relation)", coeffs[(5,)], 125)

# For CICY 7821, the number of polynomials of each defining multidegree must
# be at least one, or the equation could not be written down.
coeffs = E.hilbert_coefficients(cicy7821, 4)
for q in set(h["numerator"]):
    check_true("multidegree %s is populated" % (q,), coeffs.get(q, 0) > 0)

# --------------------------------------------------------------------------
print("\n[3] Gopakumar-Vafa invariants of the five one-parameter models")
# Classical genus-zero instanton numbers.
KNOWN = {
    "P^4[5]":       ([[4, 5]],          [2875, 609250, 317206375, 242467530000]),
    "P^5[3,3]":     ([[5, 3, 3]],       [1053, 52812, 6424326, 1139448384]),
    "P^5[2,4]":     ([[5, 2, 4]],       [1280, 92288, 15655168, 3883902528]),
    "P^6[2,2,3]":   ([[6, 2, 2, 3]],    [720, 22428, 1611504, 168199200]),
    "P^7[2,2,2,2]": ([[7, 2, 2, 2, 2]], [512, 9728, 416256, 25703936]),
}
KAPPA = {"P^4[5]": 5, "P^5[3,3]": 9, "P^5[2,4]": 8,
         "P^6[2,2,3]": 12, "P^7[2,2,2,2]": 16}

for name, (conf, expected) in sorted(KNOWN.items()):
    g = E.gv_invariants(conf, max_degree=len(expected))
    check("%s name" % name, g["name"], name)
    check("%s triple intersection kappa" % name, g["kappa"], KAPPA[name])
    for d, want in enumerate(expected, start=1):
        check("%s n_%d" % (name, d), g["invariants"][d], want)
    check_true("%s invariants are integers" % name,
               all(isinstance(v, int) for v in g["invariants"].values()))

# Multi-parameter models are refused rather than silently mishandled.
for conf, desc in [([[2, 3], [2, 3]], "two projective factors"),
                   ([[1, 1, 1], [4, 1, 4]], "the split quintic")]:
    try:
        E.gv_invariants(conf)
        check_true("%s rejected" % desc, False)
    except ValueError as exc:
        check_true("%s rejected" % desc, "one-parameter" in str(exc))

try:
    E.gv_invariants([[4, 4]])
    check_true("non-Calabi-Yau rejected", False)
except ValueError:
    check_true("non-Calabi-Yau rejected", True)

# --------------------------------------------------------------------------
print("\n[4] Cross-checks against the rest of the package")
CONFS = [[[4, 5]], [[2, 3], [2, 3]], [[1, 2], [1, 2], [1, 2], [1, 2]],
         [[1, 1, 1], [4, 1, 4]], [[1, 1, 1], [4, 2, 3]], [[3, 4], [1, 2]],
         [[2, 3], [1, 2], [1, 2]], [[2, 1, 1, 1], [4, 2, 2, 1]]]

import logging
logging.disable(logging.CRITICAL)
from pyCICY import CICY

euler_ok = c2_ok = 0
for conf in CONFS:
    # Euler characteristic by integrating c_3, against pyCICY's own value
    # from triple intersection numbers.
    mine = E.euler_from_chern(conf)
    theirs = CICY(conf).euler_characteristic()
    if mine == theirs:
        euler_ok += 1
    else:
        FAILURES.append("euler %s" % (conf,))
        print("  euler mismatch %s: %s vs %s" % (conf, mine, theirs))

    # Degree-2 part of the Chern polynomial against -ch_2 from transitions.
    cp = E.chern_polynomial(conf)
    deg2 = {}
    for e, coeff in cp.items():
        if sum(e) == 2:
            idx = [i for i, k in enumerate(e) for _ in range(k)]
            key = (idx[0], idx[1])
            deg2[key] = deg2.get(key, Fraction(0)) + coeff
    negch2 = {k: -v for k, v in T.chern_character_2(conf).items()}
    if deg2 == negch2:
        c2_ok += 1
    else:
        FAILURES.append("c2 %s" % (conf,))
        print("  c2 mismatch %s: %s vs %s" % (conf, deg2, negch2))

check("Euler from c_3 agrees with pyCICY", euler_ok, len(CONFS))
check("c_2 from Chern polynomial agrees with -ch_2", c2_ok, len(CONFS))

# CICY 7821's published invariants, as a final anchor.
X = CICY(cicy7821)
check("CICY 7821 h^{1,1} (CIPro: 2)", float(X.h[2]), 2.0)
check("CICY 7821 h^{2,1} (CIPro: 58)", float(X.h[1]), 58.0)
check("CICY 7821 Euler (CIPro: -112)", X.euler_characteristic(), -112)

# --------------------------------------------------------------------------
print("\n" + "=" * 72)
if FAILURES:
    print("FAILED ({}): {}".format(len(FAILURES), ", ".join(FAILURES)))
    sys.exit(1)
print("ALL TESTS PASSED on Python {}".format(sys.version.split()[0]))
