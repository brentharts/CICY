r"""
pyCICY.centralcharge -- exact large-volume central charges on CICY threefolds.

Why this module exists
----------------------
A Bridgeland stability condition on D^b(X) is a central charge
Z : K_num(X) -> C together with a slicing. For a Calabi-Yau threefold the
charges expected at large volume are

    Z_ord(E)  = - \int e^{-(B + i w)}  ch(E)                  (ordinary)
    Z_sqrt(E) = - \int e^{-(B + i w)}  ch(E) sqrt(td X)       (square-root Todd)

with w = t H the Kaehler class and sqrt(td X) = 1 + c_2(X)/24 on a threefold
with trivial canonical bundle. These are the charges of

    OpenAI, "Prescribed large-volume charges on threefolds with trivial
    canonical bundle", 24 Sept 2026 (family 055 of the OpenAI math release),
    OpenAI, "A Gepner stability condition on every smooth quintic threefold",
    24 Sept 2026,

which announce that stability conditions with exactly these charges exist at
every sufficiently large volume (and, for the quintic, one with the Gepner
symmetry of phase shift 2/5, Toda's conjecture).

This module computes the charges, exactly, from data the package already has:
the triple intersection numbers d_rst and the numbers \int c_2(TX) J_r of a
favourable CICY, and the Chern character of a virtual sum of line bundles
(the same presentation as :mod:`pyCICY.bundles`).

Conventions are those of the paper: Z_ord(E) = -ch_3^B + (1/2) w^2 ch_1^B
+ i (w ch_2^B - w^3 ch_0 / 6), with ch^B = e^{-B} ch. Everything is rational
arithmetic (``fractions.Fraction``); floats are refused, not rounded.

Ledger
------
Computed exactly, and each by two routes sharing no code (tests/test_centralcharge.py):

* Z_ord and Z_sqrt of an object given by its Chern character: a closed-form
  expansion in ch^B_k against a truncated-ring expansion of the exponential
  in the variables J_r integrated with d_rst.
* Covariance Z_{B+L}(E (x) O(L)) = Z_B(E): twisting the summands against
  shifting B.
* The Euler form and chi(E) by Hirzebruch-Riemann-Roch against the Leray
  computation ``CICY.line_co_euler`` on each summand.
* For h^{1,1} = 1 the autoequivalence Phi = T_O o (- (x) O(1)) acting on
  K_num (x) Q: Phi preserves the Euler form, its order, its characteristic
  polynomial and the eigenvector central charge Z_G with Z_G o Phi =
  exp(2 pi i k/n) Z_G are computed.

Not computed, and not claimed
-----------------------------
* That (Z, heart) is a stability condition. That is the cited theorem
  (unrefereed, not formalised in Lean in the release). This module computes
  the charge; the slicing, the support property and finite Harder-Narasimhan
  filtrations are not touched.
* The volume threshold T beyond which the theorem applies. The paper gives
  no value this module could use, so no charge here is certified to lie on a
  stability condition; ``line_bundle_phase`` therefore says "assuming the line
  bundle is tilt-stable" in its result.
* Whether a given w is ample. Entries >= 0 in the basis of ambient Kaehler
  generators (restricted classes) are accepted as ample; other ample classes
  are not recognised and are refused.
* That Z_G lies on a stability condition: the Gepner statement itself is the
  announced theorem. Only its K-theory shadow is computed.
* Anything for non-favourable configurations.
"""

from fractions import Fraction
from itertools import product as _product

import numpy as np

from .pyCICY import CICY

__all__ = ["Cx", "NotAmple", "CY3", "Object", "z_ordinary", "z_sqrt_td",
           "z_from_ring", "euler_pairing", "chi", "chi_from_cohomology",
           "line_bundle_phase", "gepner_matrix", "gepner_report"]


class NotAmple(ValueError):
    """Raised when the Kaehler class is not recognised as ample."""


def _frac(x):
    """Exact Fraction from int/Fraction/str. Floats are refused."""
    if isinstance(x, bool):
        raise TypeError("booleans are not numbers here")
    if isinstance(x, (int, np.integer)):
        return Fraction(int(x))
    if isinstance(x, Fraction):
        return x
    if isinstance(x, str):
        return Fraction(x)
    raise TypeError("exact rational input required, got %r; pass an int, a "
                    "Fraction or a string such as '3/2'" % (x,))


def _vec(v, n):
    out = [_frac(x) for x in v]
    if len(out) != n:
        raise ValueError("vector %r should have length %d" % (v, n))
    return out


class Cx(object):
    """Exact complex number re + i im with Fraction parts."""

    __slots__ = ("re", "im")

    def __init__(self, re=0, im=0):
        self.re = _frac(re)
        self.im = _frac(im)

    def __add__(self, o):
        o = o if isinstance(o, Cx) else Cx(o)
        return Cx(self.re + o.re, self.im + o.im)

    __radd__ = __add__

    def __neg__(self):
        return Cx(-self.re, -self.im)

    def __sub__(self, o):
        return self + (-(o if isinstance(o, Cx) else Cx(o)))

    def __mul__(self, o):
        o = o if isinstance(o, Cx) else Cx(o)
        return Cx(self.re * o.re - self.im * o.im,
                  self.re * o.im + self.im * o.re)

    __rmul__ = __mul__

    def __eq__(self, o):
        o = o if isinstance(o, Cx) else Cx(o)
        return self.re == o.re and self.im == o.im

    def __hash__(self):
        return hash((self.re, self.im))

    def __complex__(self):
        return complex(float(self.re), float(self.im))

    def __repr__(self):
        return "Cx(%s, %s)" % (self.re, self.im)

    def phase(self):
        """arg(Z)/pi in (-1, 1], as a float; Z = 0 raises."""
        if self.re == 0 and self.im == 0:
            raise ValueError("the zero charge has no phase")
        import math
        return math.atan2(float(self.im), float(self.re)) / math.pi


# ---------------------------------------------------------------------------
# the threefold
# ---------------------------------------------------------------------------

class CY3(object):
    """Intersection data of a Calabi-Yau threefold in a basis of H^2.

    ``d`` is the symmetric array d_rst of integers, ``c2`` the vector
    \\int c_2(TX) J_r. Build from a favourable CICY with :meth:`from_cicy`,
    or pass the numbers directly (useful for tests and for non-CICY inputs).
    """

    def __init__(self, d, c2, name=None):
        d = np.array(d, dtype=object)
        if d.ndim != 3 or not (d.shape[0] == d.shape[1] == d.shape[2]):
            raise ValueError("d must be a cubic array d_rst")
        h = d.shape[0]
        self.h = h
        self.d = [[[_frac(d[r, s, t]) for t in range(h)]
                   for s in range(h)] for r in range(h)]
        for r, s, t in _product(range(h), repeat=3):
            if not (self.d[r][s][t] == self.d[s][r][t] == self.d[r][t][s]):
                raise ValueError("d_rst is not symmetric")
        self.c2 = _vec(c2, h)
        self.name = name

    @classmethod
    def from_cicy(cls, X):
        if not isinstance(X, CICY):
            X = CICY(X, log=3)
        if X.nfold != 3:
            raise ValueError("a threefold is required")
        if not X.fav:
            raise ValueError("favourable configuration required: classes "
                             "must be integer vectors in a basis of H^2")
        d = np.array(X.triple_intersection(), dtype=float)
        c2 = np.array(X.second_chern(), dtype=float)
        di, ci = np.rint(d), np.rint(c2)
        if np.max(np.abs(d - di)) > 1e-6 or np.max(np.abs(c2 - ci)) > 1e-6:
            raise ArithmeticError("intersection numbers are not integers")
        out = cls(di.astype(int), ci.astype(int))
        out.cicy = X
        return out

    # trilinear form and friends ---------------------------------------
    def cubic(self, a, b, c):
        h = self.h
        return sum(self.d[r][s][t] * a[r] * b[s] * c[t]
                   for r in range(h) for s in range(h) for t in range(h)
                   if self.d[r][s][t] != 0)

    def quad(self, a, b):
        """The vector r -> d_rst a^s b^t."""
        h = self.h
        return [sum(self.d[r][s][t] * a[s] * b[t]
                    for s in range(h) for t in range(h)) for r in range(h)]

    def dot(self, u, v):
        return sum(a * b for a, b in zip(u, v))


class Object(object):
    """A class in K_num (x) Q through its Chern character.

    ``rank``, ``c1`` (vector in the basis of H^2), ``ch2`` (the vector
    \\int ch_2 J_r) and ``ch3`` (the number \\int ch_3).
    """

    def __init__(self, X, rank, c1, ch2, ch3):
        self.X = X
        self.rank = _frac(rank)
        self.c1 = _vec(c1, X.h)
        self.ch2 = _vec(ch2, X.h)
        self.ch3 = _frac(ch3)

    @classmethod
    def from_lines(cls, X, summands, signs=None):
        """The virtual sum sum_a eps_a O(L_a)."""
        signs = [1] * len(summands) if signs is None else signs
        r = Fraction(0)
        c1 = [Fraction(0)] * X.h
        ch2 = [Fraction(0)] * X.h
        ch3 = Fraction(0)
        for L, e in zip(summands, signs):
            L = _vec(L, X.h)
            e = _frac(e)
            r += e
            c1 = [a + e * b for a, b in zip(c1, L)]
            q = X.quad(L, L)
            ch2 = [a + e * b / 2 for a, b in zip(ch2, q)]
            ch3 += e * X.cubic(L, L, L) / 6
        return cls(X, r, c1, ch2, ch3)

    @classmethod
    def point(cls, X):
        """The skyscraper sheaf O_x: ch = (0, 0, 0, 1)."""
        return cls(X, 0, [0] * X.h, [0] * X.h, 1)

    # arithmetic ----------------------------------------------------------
    def __add__(self, o):
        return Object(self.X, self.rank + o.rank,
                      [a + b for a, b in zip(self.c1, o.c1)],
                      [a + b for a, b in zip(self.ch2, o.ch2)],
                      self.ch3 + o.ch3)

    def __neg__(self):
        return Object(self.X, -self.rank, [-a for a in self.c1],
                      [-a for a in self.ch2], -self.ch3)

    def __sub__(self, o):
        return self + (-o)

    def scale(self, k):
        k = _frac(k)
        return Object(self.X, k * self.rank, [k * a for a in self.c1],
                      [k * a for a in self.ch2], k * self.ch3)

    def shift(self, k):
        """E[k]: the class (-1)^k [E]."""
        return self if k % 2 == 0 else -self

    def twist(self, L):
        """E (x) O(L): ch -> ch e^L."""
        X = self.X
        L = _vec(L, X.h)
        qL = X.quad(L, L)
        ch1L = X.quad(L, self.c1)           # d(L, c1, .)
        c1 = [a + self.rank * b for a, b in zip(self.c1, L)]
        ch2 = [a + b + self.rank * q / 2
               for a, b, q in zip(self.ch2, ch1L, qL)]
        ch3 = (self.ch3 + X.dot(L, self.ch2) + X.cubic(L, L, self.c1) / 2
               + self.rank * X.cubic(L, L, L) / 6)
        return Object(X, self.rank, c1, ch2, ch3)

    def vector(self):
        """Coordinates (rank, c1, ch2, ch3) as a flat list."""
        return [self.rank] + list(self.c1) + list(self.ch2) + [self.ch3]

    # B-twisted character -------------------------------------------------
    def ch_B(self, B):
        """(ch0, ch1^B, ch2^B vector, ch3^B) for e^{-B} ch."""
        X = self.X
        B = _vec(B, X.h)
        r = self.rank
        ch1 = [a - r * b for a, b in zip(self.c1, B)]
        Bc = X.quad(B, self.c1)
        BB = X.quad(B, B)
        ch2 = [a - b + r * c / 2 for a, b, c in zip(self.ch2, Bc, BB)]
        ch3 = (self.ch3 - X.dot(B, self.ch2) + X.cubic(B, B, self.c1) / 2
               - r * X.cubic(B, B, B) / 6)
        return r, ch1, ch2, ch3


# ---------------------------------------------------------------------------
# central charges
# ---------------------------------------------------------------------------

def _check_ample(X, omega):
    w = _vec(omega, X.h)
    if any(x < 0 for x in w) or all(x == 0 for x in w):
        raise NotAmple("w = %s: entries must be >= 0 and not all zero (a "
                       "non-negative combination of the ambient Kaehler "
                       "generators); other ample classes are not recognised"
                       % [str(x) for x in w])
    if X.cubic(w, w, w) <= 0:
        raise NotAmple("w^3 <= 0")
    return w


def z_ordinary(E, B, omega):
    r"""Z_ord(E) = -ch3^B + (1/2) w^2 ch1^B + i (w ch2^B - w^3 ch0 / 6)."""
    X = E.X
    w = _check_ample(X, omega)
    r, ch1, ch2, ch3 = E.ch_B(B)
    re = -ch3 + X.dot(X.quad(w, w), ch1) / 2
    im = X.dot(w, ch2) - X.cubic(w, w, w) * r / 6
    return Cx(re, im)


def z_sqrt_td(E, B, omega):
    r"""Z_sqrt(E) = -\int e^{-B-iw} ch(E) (1 + c2/24)
    = Z_ord(E) - (1/24) c2 . (ch1^B - i w ch0)."""
    X = E.X
    w = _check_ample(X, omega)
    r, ch1, _, _ = E.ch_B(B)
    corr = Cx(X.dot(X.c2, ch1), -r * X.dot(X.c2, w)) * Fraction(-1, 24)
    return z_ordinary(E, B, omega) + corr


def z_from_ring(X, summands, signs, B, omega, sqrt_td=True):
    r"""The same charge from a different computation, for cross-checking.

    Expands -e^{-B-iw} sum_a eps_a e^{L_a}(1 + c2/24) as a truncated power
    series in the variables x_r = J_r with sympy and integrates the degree 3
    part by x_r x_s x_t -> d_rst. It shares no code with :func:`z_ordinary`
    / :func:`z_sqrt_td`, which use closed-form expansions in ch^B_k.
    Only virtual sums of line bundles are accepted.
    """
    import sympy as sp
    w = _check_ample(X, omega)
    B = _vec(B, X.h)
    xs = sp.symbols("x0:%d" % X.h)
    lam = sp.Symbol("lam")          # grading variable
    C = sp.Symbol("C")              # the degree 2 class c2(TX)

    def R(f):
        return sp.Rational(f.numerator, f.denominator)

    total = 0
    for L, e in zip(summands, signs):
        L = _vec(L, X.h)
        lin = sum((R(L[r]) - R(B[r]) - sp.I * R(w[r])) * xs[r]
                  for r in range(X.h))
        series = sum((lam * lin) ** k / sp.factorial(k) for k in range(4))
        td = 1 + (lam ** 2 * C / 24 if sqrt_td else 0)
        total += R(_frac(e)) * series * td
    deg3 = sp.expand(sp.expand(total).coeff(lam, 3))
    poly = sp.Poly(deg3, *xs, C)
    val = 0
    for mon, coef in poly.terms():
        idx = [r for r, p in enumerate(mon[:-1]) for _ in range(p)]
        if mon[-1] == 0:
            val += coef * R(X.d[idx[0]][idx[1]][idx[2]])
        else:
            val += coef * R(X.c2[idx[0]])
    val = sp.expand(-val)
    re, im = sp.re(val), sp.im(val)
    return Cx(Fraction(int(re.p), int(re.q)), Fraction(int(im.p), int(im.q)))


# ---------------------------------------------------------------------------
# Euler form
# ---------------------------------------------------------------------------

def _dual(E):
    """E^vee: ch_k -> (-1)^k ch_k."""
    return Object(E.X, E.rank, [-a for a in E.c1], E.ch2, -E.ch3)


def _product_ch(E, F):
    """Chern character of E (x) F (rank, c1, ch2, ch3)."""
    X = E.X
    h = X.h
    r = E.rank * F.rank
    c1 = [E.rank * b + F.rank * a for a, b in zip(E.c1, F.c1)]
    cross = X.quad(E.c1, F.c1)
    ch2 = [E.rank * b + F.rank * a + c
           for a, b, c in zip(E.ch2, F.ch2, cross)]
    ch3 = (E.rank * F.ch3 + F.rank * E.ch3 + X.dot(E.c1, F.ch2)
           + X.dot(F.c1, E.ch2))
    return Object(X, r, c1, ch2, ch3)


def chi(E):
    r"""chi(E) = \int ch(E) td(X) = ch3 + c2 . ch1 / 12 (HRR on a CY3)."""
    return E.ch3 + E.X.dot(E.X.c2, E.c1) / 12


def euler_pairing(E, F):
    r"""chi(E, F) = \int ch(E^vee) ch(F) td(X)."""
    return chi(_product_ch(_dual(E), F))


def chi_from_cohomology(X, summands, signs=None):
    """chi of a virtual line-bundle sum from ``CICY.line_co_euler`` (Leray).

    Shares no code with :func:`chi`."""
    signs = [1] * len(summands) if signs is None else signs
    cy = X.cicy if isinstance(X, CY3) else X
    return sum(int(e) * int(round(cy.line_co_euler(list(map(int, L)))))
               for L, e in zip(summands, signs))


# ---------------------------------------------------------------------------
# phases of line bundles
# ---------------------------------------------------------------------------

def line_bundle_phase(X, D, B, omega):
    r"""Position of O(D) in the double-tilted heart, *assuming it is
    tilt-stable* (not proved here), and its phase.

    Returns ``(k, Z, phase)``: the shift k in {0, 1, 2} such that O(D)[k]
    lies in the heart A_{B,w}, the charge Z_ord(O(D)[k]) and arg(Z)/pi in
    (0, 1]. k follows the first tilt (mu_{H,B} = w^2 (D - B) / w^3 > 0 puts
    O(D) in T, otherwise in F so O(D)[1] in the tilt) then the sign of
    nu = Im Z / (w^2 ch1^B) with the strict-positive / non-positive rule
    of the paper. The charge, the slope and nu are exact; the stability
    assumption is the only unproved input.
    """
    E = Object.from_lines(X, [D])
    w = _check_ample(X, omega)
    B_ = _vec(B, X.h)
    ww = X.quad(w, w)
    mu_num = X.dot(ww, [d - b for d, b in zip(_vec(D, X.h), B_)])
    in_T = mu_num > 0
    Z = z_ordinary(E, B_, w)
    base = E if in_T else -E          # object of the first tilt (class)
    Zb = z_ordinary(base, B_, w)
    den = X.dot(ww, base.ch_B(B_)[1])
    if den == 0:
        nu_pos = Zb.im > 0
    else:
        nu_pos = (Zb.im / den) > 0
    if in_T:
        k = 0 if nu_pos else 1
    else:
        k = 1 if nu_pos else 2
    Zk = Z if k % 2 == 0 else -Z
    ph = Zk.phase()
    if ph <= 0:
        ph += 2          # reported for inspection; callers test 0 < ph <= 1
    return k, Zk, ph


# ---------------------------------------------------------------------------
# the Gepner autoequivalence on K-theory (h11 = 1)
# ---------------------------------------------------------------------------

def _mat_mul(A, Bm):
    n = len(A)
    return [[sum(A[i][k] * Bm[k][j] for k in range(n)) for j in range(n)]
            for i in range(n)]


def gepner_matrix(X):
    r"""The matrix of Phi = T_O o (- (x) O(1)) on K_num (x) Q for h11 = 1.

    Basis: the coordinates (ch0, ch1, ch2, ch3) as in :meth:`Object.vector`
    (a column of ``M`` is the image of the corresponding basis vector, so
    v -> M v). T_O(F) has class [F] - chi(O, F) [O].
    """
    if X.h != 1:
        raise ValueError("gepner_matrix is written for h^{1,1} = 1")
    one = Object.from_lines(X, [[0]])
    basis = [Object(X, 1, [0], [0], 0), Object(X, 0, [1], [0], 0),
             Object(X, 0, [0], [1], 0), Object(X, 0, [0], [0], 1)]
    cols = []
    for e in basis:
        t = e.twist([1])
        img = t - one.scale(euler_pairing(one, t))
        cols.append(img.vector())
    return [[cols[j][i] for j in range(4)] for i in range(4)]


def gepner_report(X, max_order=40):
    r"""Exact K-theory data of Phi for h11 = 1.

    Returns a dict with ``matrix``, ``order`` (smallest n <= max_order with
    Phi^n = +-1, and the sign, or None), ``preserves_euler`` (Phi^T G Phi = G
    for the Euler form G, in the coordinates above), ``charpoly`` (sympy
    polynomial in x), and ``phi`` / ``euler_form`` as sympy matrices.
    """
    import sympy as sp
    M = gepner_matrix(X)
    Ms = sp.Matrix(4, 4, lambda i, j: sp.Rational(M[i][j].numerator,
                                                  M[i][j].denominator))
    basis = [Object(X, 1, [0], [0], 0), Object(X, 0, [1], [0], 0),
             Object(X, 0, [0], [1], 0), Object(X, 0, [0], [0], 1)]
    G = sp.Matrix(4, 4, lambda i, j: sp.Rational(
        euler_pairing(basis[i], basis[j]).numerator,
        euler_pairing(basis[i], basis[j]).denominator))
    pres = (Ms.T * G * Ms == G)
    order = None
    P = sp.eye(4)
    for n in range(1, max_order + 1):
        P = P * Ms
        if P == sp.eye(4):
            order = (n, +1)
            break
        if P == -sp.eye(4):
            order = (n, -1)
            break
    x = sp.Symbol("x")
    cp = sp.factor(Ms.charpoly(x).as_expr())
    return {"matrix": M, "euler_form": G, "preserves_euler": bool(pres),
            "order": order, "charpoly": cp, "phi": Ms}
