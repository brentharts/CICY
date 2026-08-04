"""
Tests for pyCICY.apolynomial.

The module ties the two ends of the package together: knots on one side,
quantized curves on the other, joined by the A-polynomial and the AJ
conjecture. The claims worth checking are:

  [1] the colored Jones formula for torus knots is normalised correctly
      (J_1 = 1) and, at N = 2, reproduces pyCICY.knots.jones exactly. This is
      the strongest test in the suite: a representation-theoretic sum over
      2N-1 terms against a sum over 2^n Kauffman states, computed by
      completely separate code, agreeing coefficient for coefficient;
  [2] the Laurent division underlying it is exact, and fails loudly when it
      should;
  [3] the recorded A-polynomials have the right Newton polygons, and their
      edge slopes reproduce the known boundary slopes -- 6 for the trefoil,
      which is pq, and +-4 for the figure-eight;
  [4] the A-polynomial Newton polygons are generally *not* reflexive, so the
      quantized A-polynomial is not the mirror curve of a local Calabi-Yau.
      What is shared with pyCICY.quantum_curve is the quantization rule, and
      the test records the distinction rather than blurring it;
  [5] the AJ conjecture holds for the trefoil as far as this can check it:
      the classical limit of the annihilating operators is divisible by the
      A-polynomial. Section [5] takes about twenty seconds; pass --quick to
      skip it.

Run with:  python3 tests/test_apolynomial.py
       or: python3 run_tests.py  (runs every suite)
"""

import os
import sys
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyCICY import apolynomial as A
from pyCICY import knots as K
from pyCICY import quantum_curve as Q
from pyCICY import toric as T

FAILURES = []
QUICK = "--quick" in sys.argv


def check(name, got, want):
    ok = got == want
    print("  {:<58} {:>10} {}".format(name, str(got)[:10],
                                      "ok" if ok else "FAIL want " + str(want)))
    if not ok:
        FAILURES.append(name)


def check_true(name, cond):
    print("  {:<58} {:>10} {}".format(name, str(bool(cond)),
                                      "ok" if cond else "FAIL"))
    if not cond:
        FAILURES.append(name)


TORUS = [((2, 3), "3_1"), ((2, 5), "5_1"), ((2, 7), "7_1"), ((3, 4), "8_19")]

# --------------------------------------------------------------------- [1]

print("\n[1] colored Jones, against the independent Kauffman computation")
for (s, t), name in TORUS:
    check("T(%d,%d): J_1 = 1" % (s, t),
          str(A.colored_jones_torus(s, t, 1)), "1")
    cj = A.colored_jones_torus(s, t, 2)
    ref = K.from_name(name).jones()
    check_true("T(%d,%d): J_2 equals jones(%s) exactly" % (s, t, name),
               cj == ref)
    check_true("T(%d,%d): and is not merely the mirror" % (s, t),
               cj != ref.invert_variable() or ref.is_palindromic())
for N in range(1, 7):
    cj = A.colored_jones_torus(2, 3, N)
    check_true("trefoil J_%d is a Laurent polynomial with integer coefficients"
               % N, all(isinstance(c, int) for c in cj.c.values()))
    check_true("trefoil J_%d evaluates to 1 at q = 1" % N,
               abs(cj.evaluate(1) - 1) < 1e-9)
# the colours grow in span, as they must
spans = [A.colored_jones_torus(2, 3, N).degrees() for N in range(1, 6)]
check_true("the trefoil's colours have strictly growing span",
           all(spans[i + 1][1] - spans[i + 1][0] > spans[i][1] - spans[i][0]
               for i in range(len(spans) - 1)))
try:
    A.colored_jones_torus(2, 4, 3)
    check_true("T(2,4) rejected as a link", False)
except ValueError:
    check_true("T(2,4) rejected as a link", True)
try:
    A.colored_jones_torus(2, 3, 0)
    check_true("colour zero rejected", False)
except ValueError:
    check_true("colour zero rejected", True)

# --------------------------------------------------------------------- [2]

print("\n[2] the Laurent division is exact")
check("(x^2 - 1) / (x - 1) = x + 1",
      A.laurent_divide({2: 1, 0: -1}, {1: 1, 0: -1}), {1: 1, 0: 1})
check("division by a monomial shifts",
      A.laurent_divide({3: 2, 1: 4}, {1: 2}), {2: 1, 0: 2})
check("exact division of a square",
      A.laurent_divide({2: 1, 1: 2, 0: 1}, {1: 1, 0: 1}), {1: 1, 0: 1})
try:
    A.laurent_divide({0: 1}, {1: 1, 0: -1}, guard=200)
    check_true("a non-terminating division raises", False)
except RuntimeError:
    check_true("a non-terminating division raises", True)
try:
    A.laurent_divide({0: 1}, {})
    check_true("an empty denominator raises", False)
except ZeroDivisionError:
    check_true("an empty denominator raises", True)

# --------------------------------------------------------------------- [3]

print("\n[3] A-polynomials, Newton polygons and boundary slopes")
for (s, t), name in TORUS:
    Ap = A.torus_apolynomial(s, t)
    check("T(%d,%d): A = 1 + L M^{pq}" % (s, t), Ap,
          {(0, 0): 1, (1, s * t): 1})
    check("T(%d,%d): recorded entry agrees" % (s, t), A.apolynomial(name), Ap)
    check("T(%d,%d): boundary slope is pq" % (s, t),
          A.boundary_slopes(Ap), [Fraction(s * t)])
check("figure-eight boundary slopes are +-4",
      A.boundary_slopes(A.apolynomial("4_1")), [Fraction(-4), Fraction(4)])
check("figure-eight Newton polygon",
      A.newton_polygon(A.apolynomial("4_1")),
      [(0, 4), (1, 0), (2, 4), (1, 8)])
check_true("the figure-eight A-polynomial is quadratic in L",
           max(i for i, _ in A.apolynomial("4_1")) == 2)
check("the abelian factor is L - 1", A.abelian_factor(),
      {(1, 0): 1, (0, 0): -1})
prod = A.multiply(A.torus_apolynomial(2, 3), A.abelian_factor())
check_true("including the abelian factor raises the L-degree",
           max(i for i, _ in prod) == 2)
check_true("multiplication is what sympy says",
           A.to_sympy(prod) ==
           __import__("sympy").expand(A.to_sympy(A.torus_apolynomial(2, 3))
                                      * A.to_sympy(A.abelian_factor())))
try:
    A.apolynomial("6_2")
    check_true("an unrecorded knot raises", False)
except KeyError:
    check_true("an unrecorded knot raises", True)

# --------------------------------------------------------------------- [4]

print("\n[4] the shared object is the quantization rule, not the geometry")
poly = A.newton_polygon(A.apolynomial("4_1"))
check_true("the figure-eight Newton polygon is not reflexive",
           not T.is_reflexive(poly))
check_true("no reflexive polygon matches it",
           not any(T.equivalent(poly, T.polygon(nm)) for nm in T.NAMED
                   if len(T.polygon(nm)) == len(poly)))
curve = A.to_quantum_curve(A.apolynomial("4_1"), name="4_1")
check_true("the A-polynomial quantizes to a QuantumCurve",
           isinstance(curve, Q.QuantumCurve))
check_true("no hop sits at the origin", (0, 0) not in curve.points)
# the figure-eight A-polynomial has no constant term -- its lowest L power
# is L^1 M^0 -- so nothing is dropped for it; the trefoil's 1 + L M^6 does
check("the figure-eight keeps every monomial", len(curve.points),
      len(A.apolynomial("4_1")))
check_true("the figure-eight has no constant term",
           (0, 0) not in A.apolynomial("4_1"))
check_true("the trefoil does have one", (0, 0) in A.apolynomial("3_1"))
tref_curve = A.to_quantum_curve(A.apolynomial("3_1"), name="3_1")
check("the trefoil loses its constant term", len(tref_curve.points),
      len(A.apolynomial("3_1")) - 1)
H = curve.bloch_matrix(1, 3, 0.2, 0.5)
check("the Bloch matrix has the right shape", H.shape, (3, 3))
check_true("and is Hermitian", __import__("numpy").allclose(H, H.conj().T))
try:
    A.to_quantum_curve({(0, 0): 1})
    check_true("quantizing a bare constant raises", False)
except ValueError:
    check_true("quantizing a bare constant raises", True)

# --------------------------------------------------------------------- [5]

print("\n[5] the AJ conjecture for the trefoil")
if QUICK:
    print("  (skipped, --quick)")
else:
    import sympy as sp

    js = {N: A.colored_jones_torus(2, 3, N) for N in range(1, 8)}
    terms, ns = A.find_recursion(js, dL=1, dQ=2, jlo=-4, jhi=4, nmax=5)
    check("no operator of L-degree 1 within small bounds", len(ns), 0)

    rep = A.verify_aj()
    check("the trefoil is the knot under test", rep["knot"], "T(2,3)")
    check_true("an annihilating operator was found", rep["found"])
    check("its L-degree is 3", rep["L_degree"], 3)
    check_true("the nullspace is more than one dimensional",
               rep["nullspace_dim"] > 1)
    check_true("the A-polynomial divides the classical limit",
               rep["a_polynomial_divides"])
    check("the A-polynomial recovered is 1 + L M^6",
          sp.expand(rep["a_polynomial"]),
          sp.expand(A.to_sympy(A.torus_apolynomial(2, 3))))
    L, M = sp.symbols("L M")
    check("the extra factor is (L-1)^2 (M^2-1)",
          sp.expand(rep["extra_factor"]),
          sp.expand((L - 1) ** 2 * (M ** 2 - 1)))
    check_true("the abelian component L-1 appears in the gcd",
               sp.rem(sp.Poly(sp.expand(rep["gcd_of_classical_limits"]), L),
                      sp.Poly(L - 1, L)).is_zero)
    check_true("the report is explicit that minimality is not claimed",
               "not a proven minimum" in rep["note"])


print("\n" + "-" * 72)
if FAILURES:
    print("FAILED (%d):" % len(FAILURES))
    for f in FAILURES:
        print("  " + f)
    sys.exit(1)
print("test_apolynomial: all checks passed")
