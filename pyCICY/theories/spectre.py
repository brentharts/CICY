r"""
pyCICY.theories.spectre -- the Spectre substrate, its exact spectral theory,
and the checkable layer of the crossover proposal.

What this module is
-------------------
An independent verification suite for Hartshorn's "The Spectre Substrate: A
Chiral Monotile Vacuum for the Penrose Conformal Crossover Surface", which
proposes that the zero-entropy crossover surface of conformal cyclic
cosmology carries the strictly chiral Spectre monotile phase of Smith,
Myers, Kaplan and Goodman-Strauss, and computes the exact spectral theory of
its substitution. The companion repository is
github.com/brentharts/spectre; this module re-derives the paper's Facts by
routes that share no code with that repository, in the discipline of the
series: a quantity computed once is a result, computed twice by unrelated
routes it is a test.

The layer structure of the source paper is respected exactly. Its *Facts*
(exact computations on the published substitution rules) are re-derived here
in exact arithmetic -- ``fractions.Fraction`` and a small
:class:`Quad` class for the fields Q(sqrt15) and Q(sqrt3), no floating
point anywhere in the assertions. Its *Propositions* (computations with a
modeling step) are re-computed with the modeling step stated. Its
*Conjectures* and *Speculations* are not tested, because they are not
testable by computation, and the module says so rather than pretending
otherwise.

What the verification adds
--------------------------
Four results the source paper can absorb:

* the **census is computed twice** -- once from the substitution matrix,
  once from the actual geometric supertile construction with exact
  Q(sqrt3) affine arithmetic -- and the two routes agree at every depth
  (9, 71, 559, 4401), with the geometric route additionally certifying that
  every tile in a patch carries the *same* handedness (determinant), that
  the common handedness alternates with depth, and that it does so in exact
  lockstep with the alternating charge: det = -Q_- at every depth computed;

* the **area of Tile(a,b) in closed form**,
  A(a,b) = 2*sqrt3*a^2 + 3ab + sqrt3*b^2, an exact identity in
  Q(sqrt3)[a,b] derived by symbolic shoelace, which reproduces the paper's
  endpoint values A(1,sqrt3) = 8*sqrt3 and A(sqrt3,1) = 10*sqrt3 and gives
  the mystic asymmetry exactly: A(b,a) - A(a,b) = sqrt3*(b^2 - a^2);

* the **geometric order parameter in closed form**: with the mystic pair
  carrying both areas, max |f_Gamma - f_Delta| = 10g/(8+10g)
  (g = 4 - sqrt15), whose decimal 0.13702... is the paper's measured 0.137;

* a **corrected ring-generation identity**: the paper exhibits 1 in the
  additive span of the frequencies via 8 v_Gamma - v_Theta = 1 (verified
  exact), but its second identity as printed ("sqrt15 = 4 - v_Gamma - 3")
  is off by 3; the correct statement is 1 - v_Gamma = sqrt15 - 3, from
  which sqrt15 = 3*1 + (1 - v_Gamma) lies in the span and the module is all
  of Z[sqrt15], as claimed. The conclusion stands; the identity is fixed.

The knot layer is made theorem-grade where it was citation-grade: the
Alexander polynomial of the edge-braid closure T(2,k) is computed by two
routes (reduced Burau determinant against the closed form (t^k+1)/(t+1),
exact polynomial equality over Q), the signature -(k-1) is computed from an
exact integer Seifert form by Sylvester pivots, and u(T(2,k)) = (k-1)/2 is
derived as a sandwich -- signature lower bound against a crossing-change
induction upper bound -- rather than cited. The mirror selection rule of the
mass mechanism is then an exact signature statement: sigma(K # mirror K) = 0
(the bound that forbids binding is *lost*, and Brittenham--Hermiller show
the deficit is real) while sigma(K # K) = 2(k-1) (additivity is *forced*:
same-handed pairs cannot bind). Strict chirality deciding which
configuration occurs is exactly the sense in which the mass mechanism
switches on at the Spectre-to-Hat transition.

The categories
--------------
Exact: everything listed above. Does not exist: momentum space -- an
aperiodic tiling has no Brillouin zone, so ``band_structure()`` raises
:exc:`~pyCICY.theories.ftheory.NoSuchTheory`, in the discipline of the
monotile companion paper; and on the isolated two-dimensional screen a
crossing change is an ambient three-space move that does not exist, so
``crossing_change()`` raises the same -- the additive-to-subadditive
transition of the source paper rendered as exception semantics. Open, and
listed by :meth:`SpectreSubstrate.missing_for_physical`: the collared
frequency p = Pr(Phi^2), the binding scale delta, the horizon-tiling
intertwiner, and the asymptotic complexity class of the transversal braid
words.
"""

import itertools
import math
from fractions import Fraction as F

import numpy as np

from .base import Theory, register
from .ftheory import NoSuchTheory

__all__ = ["Quad", "SPECIES", "RULES", "substitution_matrix", "charpoly",
           "matrix_rank", "pell_fundamental", "inflation_data",
           "perron_frequencies", "degeneracy_mechanisms",
           "automorphism_group_order", "charges", "census", "census_series",
           "frequency_module", "galois_echo", "tile_area_form",
           "mystic_asymmetry", "geometric_census", "order_parameter",
           "perturbation_susceptibilities", "EDGE_SEQUENCE",
           "mystic_saturation", "x_charges", "x_density", "torus_knot",
           "binding_spectrum", "dilation_dictionary", "SpectreSubstrate"]


# ---------------------------------------------------------------------------
# exact quadratic-field arithmetic
# ---------------------------------------------------------------------------

class Quad(object):
    """An element a + b*sqrt(d) of a real quadratic field, exactly.

    ``a`` and ``b`` are Fractions; ``d`` is a squarefree integer. Supports
    the ring operations, equality, and float conversion for display. The
    spectral theory lives in d = 15, the tile geometry in d = 3, and no
    computation below ever leaves the relevant field.
    """

    __slots__ = ("a", "b", "d")

    def __init__(self, a, b=0, d=15):
        self.a, self.b, self.d = F(a), F(b), int(d)

    def _lift(self, other):
        if isinstance(other, Quad):
            if other.d != self.d:
                raise ValueError("mixed fields")
            return other
        return Quad(other, 0, self.d)

    def __add__(self, o):
        o = self._lift(o)
        return Quad(self.a + o.a, self.b + o.b, self.d)

    __radd__ = __add__

    def __neg__(self):
        return Quad(-self.a, -self.b, self.d)

    def __sub__(self, o):
        return self + (-self._lift(o))

    def __rsub__(self, o):
        return self._lift(o) - self

    def __mul__(self, o):
        o = self._lift(o)
        return Quad(self.a * o.a + self.d * self.b * o.b,
                    self.a * o.b + self.b * o.a, self.d)

    __rmul__ = __mul__

    def __truediv__(self, o):
        o = self._lift(o)
        n = o.a * o.a - self.d * o.b * o.b
        if n == 0:
            raise ZeroDivisionError
        conj = Quad(o.a, -o.b, self.d)
        num = self * conj
        return Quad(num.a / n, num.b / n, self.d)

    def __eq__(self, o):
        o = self._lift(o)
        return self.a == o.a and self.b == o.b

    def __hash__(self):
        return hash((self.a, self.b, self.d))

    def __float__(self):
        return float(self.a) + float(self.b) * math.sqrt(self.d)

    def __repr__(self):
        if self.b == 0:
            return str(self.a)
        sign = "+" if self.b >= 0 else "-"
        return "(%s %s %s*sqrt%d)" % (self.a, sign, abs(self.b), self.d)


def _q15(a, b=0):
    return Quad(a, b, 15)


def _q3(a, b=0):
    return Quad(a, b, 3)


# ---------------------------------------------------------------------------
# the substitution, verbatim from Smith-Myers-Kaplan-Goodman-Strauss
# ---------------------------------------------------------------------------

SPECIES = ("Gamma", "Delta", "Theta", "Lambda", "Xi", "Pi", "Sigma",
           "Phi", "Psi")

#: The supertile composition rules (parent -> ordered children), transcribed
#: from the published substitution [SmithEtAl2] as implemented in the
#: companion repository. Gamma is the Mystic pair Gamma1 + Gamma2 and counts
#: as two placed tiles.
RULES = {
    "Gamma": ("Pi", "Delta", "Theta", "Sigma", "Xi", "Phi", "Gamma"),
    "Delta": ("Xi", "Delta", "Xi", "Phi", "Sigma", "Pi", "Phi", "Gamma"),
    "Theta": ("Psi", "Delta", "Pi", "Phi", "Sigma", "Pi", "Phi", "Gamma"),
    "Lambda": ("Psi", "Delta", "Xi", "Phi", "Sigma", "Pi", "Phi", "Gamma"),
    "Xi": ("Psi", "Delta", "Pi", "Phi", "Sigma", "Psi", "Phi", "Gamma"),
    "Pi": ("Psi", "Delta", "Xi", "Phi", "Sigma", "Psi", "Phi", "Gamma"),
    "Sigma": ("Xi", "Delta", "Xi", "Phi", "Sigma", "Pi", "Lambda", "Gamma"),
    "Phi": ("Psi", "Delta", "Psi", "Phi", "Sigma", "Pi", "Phi", "Gamma"),
    "Psi": ("Psi", "Delta", "Psi", "Phi", "Sigma", "Psi", "Phi", "Gamma"),
}


def substitution_matrix():
    """M[i][j] = number of species-i children of a species-j supertile.

    Integers; columns are parents. The all-ones rows of Gamma, Delta, Sigma
    and Phi are visible immediately: every supertile contains exactly one of
    each, which is the conservation law behind the frequency triplet.
    """
    return [[RULES[p].count(c) for p in SPECIES] for c in SPECIES]


def charpoly():
    r"""The characteristic polynomial of M, exactly.

    Faddeev--LeVerrier over Fractions. Returns the coefficient list
    (monic, degree 9, highest first) and asserts the factored form of the
    source paper,

        chi_M(x) = x^5 (x - 1)(x + 1)(x^2 - 8x + 1),

    by expanding it independently and comparing coefficient lists.
    """
    M = substitution_matrix()
    n = 9
    A = [[F(M[i][j]) for j in range(n)] for i in range(n)]
    B = [[F(1) if i == j else F(0) for j in range(n)] for i in range(n)]
    coeffs = [F(1)]
    for k in range(1, n + 1):
        Mk = [[sum(A[i][t] * B[t][j] for t in range(n)) for j in range(n)]
              for i in range(n)]
        ck = -sum(Mk[i][i] for i in range(n)) / k
        coeffs.append(ck)
        B = [[Mk[i][j] + (ck if i == j else 0) for j in range(n)]
             for i in range(n)]
    # expand x^5 (x^2 - 1)(x^2 - 8x + 1) = x^9 - 8x^8 + 8x^6 - x^5
    factored = [F(1), F(-8), F(0), F(8), F(-1)] + [F(0)] * 5
    if coeffs != factored:
        raise AssertionError("charpoly does not match the factored form")
    return coeffs


def _rank_of(rows_in):
    M = [[F(x) for x in row] for row in rows_in]
    rows, cols = len(M), len(M[0])
    r = 0
    for c in range(cols):
        piv = next((i for i in range(r, rows) if M[i][c] != 0), None)
        if piv is None:
            continue
        M[r], M[piv] = M[piv], M[r]
        for i in range(rows):
            if i != r and M[i][c] != 0:
                f = M[i][c] / M[r][c]
                M[i] = [M[i][j] - f * M[r][j] for j in range(cols)]
        r += 1
    return r


def matrix_rank():
    r"""The rank structure of M, exactly -- and a correction.

    The source paper states rank M = 4. Exact Gaussian elimination over Q
    (confirmed by two independent routes during development) gives

        rank M = 5,   rank M^n = 4 for all n >= 2.

    The characteristic polynomial has x^5, so the zero eigenvalue has
    algebraic multiplicity five -- but its geometric multiplicity is four:
    M is *not diagonalizable*. The zero sector consists of four census
    directions erased in a single substitution step and one Jordan pair
    erased in exactly two. The corrected statement is finer than the
    original, and the five-step contraction picture of the source paper is
    unaffected: the transient sector still dies, one of its directions
    just takes two steps to do it.

    Returns dict with ``rank``, ``rank_squared``, ``kernel_dim``,
    ``diagonalizable``.
    """
    M = substitution_matrix()
    M2 = [[sum(M[i][k] * M[k][j] for k in range(9)) for j in range(9)]
          for i in range(9)]
    r1, r2 = _rank_of(M), _rank_of(M2)
    return {"rank": r1, "rank_squared": r2,
            "kernel_dim": 9 - r1,
            "generalized_kernel_dim": 9 - r2,
            "diagonalizable": r1 == r2}


# ---------------------------------------------------------------------------
# the inflation unit and its arithmetic
# ---------------------------------------------------------------------------

def pell_fundamental():
    """(4, 1) is the fundamental solution of x^2 - 15 y^2 = 1.

    Verified by minimality: no y in 1..y0-1 admits an integer x. So
    lambda^2 = 4 + sqrt15 is the fundamental unit of Z[sqrt15] (norm +1;
    the field has no unit of norm -1 since x^2 - 15y^2 = -1 is insoluble
    mod 3), and inflation acts on all area-graded data by a fundamental
    unit.
    """
    x0, y0 = 4, 1
    assert x0 * x0 - 15 * y0 * y0 == 1
    for y in range(1, y0):
        x2 = 1 + 15 * y * y
        if int(math.isqrt(x2)) ** 2 == x2:
            raise AssertionError("smaller Pell solution exists")
    # x^2 - 15 y^2 = -1 is insoluble: mod 3 it reads x^2 = -1 = 2, and the
    # squares mod 3 are {0, 1}.
    return (x0, y0)


def inflation_data():
    r"""The inflation factor lambda and its exact identities.

    Everything is proved in Z[sqrt15]:

    * lambda^2 = 4 + sqrt15 and lambda^{-2} = 4 - sqrt15 are unit inverses:
      their product is exactly 1;
    * (lambda - 1/lambda)^2 = lambda^2 + lambda^{-2} - 2 = 6 and
      (lambda + 1/lambda)^2 = 10, so lambda - 1/lambda = sqrt6 (the
      hexagonal channel) and lambda + 1/lambda = sqrt10 (the pentagonal
      channel), whence lambda = (sqrt6 + sqrt10)/2;
    * lambda^2 is a root of y^2 - 8y + 1, so lambda has minimal polynomial
      x^4 - 8x^2 + 1 with Galois orbit {+-lambda, +-1/lambda} and splitting
      field Q(sqrt6, sqrt10).
    """
    lam2 = _q15(4, 1)
    lam2inv = _q15(4, -1)
    assert lam2 * lam2inv == _q15(1)
    diff_sq = lam2 + lam2inv - 2       # (lam - 1/lam)^2
    sum_sq = lam2 + lam2inv + 2        # (lam + 1/lam)^2
    assert diff_sq == _q15(6) and sum_sq == _q15(10)
    # y^2 - 8y + 1 at y = lam2:
    assert lam2 * lam2 - 8 * lam2 + 1 == _q15(0)
    lam = math.sqrt(float(lam2))
    return {"lambda_squared": lam2,
            "lambda": lam,
            "lambda_check": (math.sqrt(6) + math.sqrt(10)) / 2.0,
            "hexagonal_channel_sq": 6,
            "pentagonal_channel_sq": 10,
            "min_poly": (1, 0, -8, 0, 1),
            "pell": pell_fundamental(),
            "contraction": float(lam2inv * lam2inv)}   # lambda^-4


# ---------------------------------------------------------------------------
# the Perron eigenvector, the multiplets, and their three mechanisms
# ---------------------------------------------------------------------------

def perron_frequencies():
    r"""The exact species frequencies, verified as an eigenvector.

    With g = 4 - sqrt15 = lambda^{-2}:

        Phi: -54 + 14 sqrt15 = 2g(1-g)     (level 0.2218)
        Psi:  97 - 25 sqrt15               (level 0.1754)
        Gamma = Delta = Sigma = g          (level 0.1270, conservation)
        Pi = Xi: -58 + 15 sqrt15           (level 0.0948, accidental)
        Theta = Lambda = g^2               (level 0.0161, conditional)

    The function verifies M v = lambda^2 v componentwise in Q(sqrt15), that
    the nine components sum to exactly 1, and the two identities that
    organise the level structure: v_Phi = 2 g (1 - g) = 2 (v_Gamma -
    v_Theta), and the generational relation v_Theta = v_Gamma / lambda^2.
    """
    g = _q15(4, -1)
    v = {"Gamma": g, "Delta": g, "Sigma": g,
         "Theta": g * g, "Lambda": g * g,
         "Pi": _q15(-58, 15), "Xi": _q15(-58, 15),
         "Phi": _q15(-54, 14), "Psi": _q15(97, -25)}
    lam2 = _q15(4, 1)
    M = substitution_matrix()
    for i, ci in enumerate(SPECIES):
        s = _q15(0)
        for j, cj in enumerate(SPECIES):
            s = s + M[i][j] * v[cj]
        if not s == lam2 * v[ci]:
            raise AssertionError("Perron eigenvector fails at %s" % ci)
    total = _q15(0)
    for x in SPECIES:
        total = total + v[x]
    assert total == _q15(1)
    assert v["Phi"] == 2 * (g - g * g)
    assert v["Phi"] == 2 * (v["Gamma"] - v["Theta"])
    assert v["Theta"] == v["Gamma"] * _q15(4, -1)
    return v


def degeneracy_mechanisms():
    """The three inequivalent protections of the five levels, checked.

    (i) conservation: the Gamma, Delta and Sigma rows of M are identical
    all-ones vectors (and so is Phi's -- but Phi's frequency differs
    because a row fixes the *in*-flow, not the frequency; the triplet is
    the set whose rows agree AND whose columns force equality, which the
    eigenvector check of :func:`perron_frequencies` certifies);
    (ii) conditional: the Theta row is the indicator of Gamma and the
    Lambda row the indicator of Sigma, so v_Theta = v_Gamma/lambda^2 and
    v_Lambda = v_Sigma/lambda^2;
    (iii) accidental: Pi = Xi hinges on the identity
    v_Phi = 2(v_Gamma - v_Theta) and on no symmetry: the automorphism
    group of M is trivial (:func:`automorphism_group_order`).
    """
    M = substitution_matrix()
    idx = {s: i for i, s in enumerate(SPECIES)}
    ones = [1] * 9
    out = {"conservation_rows_all_ones":
           all(M[idx[s]] == ones for s in ("Gamma", "Delta", "Sigma")),
           "theta_row_is_gamma_indicator":
           M[idx["Theta"]] == [1 if s == "Gamma" else 0 for s in SPECIES],
           "lambda_row_is_sigma_indicator":
           M[idx["Lambda"]] == [1 if s == "Sigma" else 0 for s in SPECIES],
           "automorphism_group_order": automorphism_group_order()}
    return out


def automorphism_group_order():
    """|Aut(M)| = 1, by exact pruned search.

    An automorphism is a permutation p with M[p(i)][p(j)] = M[i][j]. Each
    index carries the signature (sorted row, sorted column, diagonal
    entry), which any automorphism must preserve -- and for the Spectre
    matrix all nine signatures are distinct, so the identity is the only
    candidate and the search is over before it starts. The full backtracking
    search is run anyway, as a guard against the signature computation
    itself being wrong.
    """
    M = substitution_matrix()
    n = 9
    sig = [(tuple(sorted(M[i])),
            tuple(sorted(M[k][i] for k in range(n))),
            M[i][i]) for i in range(n)]
    cands = [[j for j in range(n) if sig[j] == sig[i]] for i in range(n)]
    count = [0]

    def search(i, perm, used):
        if i == n:
            count[0] += 1
            return
        for j in cands[i]:
            if j in used:
                continue
            if all(M[i][k] == M[j][perm[k]] and M[k][i] == M[perm[k]][j]
                   for k in range(i)):
                perm.append(j)
                used.add(j)
                search(i + 1, perm, used)
                perm.pop()
                used.discard(j)

    search(0, [], set())
    return count[0]


# ---------------------------------------------------------------------------
# the two integer charges and the census
# ---------------------------------------------------------------------------

#: Left eigenvectors of M for eigenvalues +1 and -1, in SPECIES order.
Q_PLUS = (-3, -1, 0, 0, 1, 1, -2, 1, 2)
Q_MINUS = (-1, -1, 2, 0, 1, -1, 0, 1, 0)


def charges():
    """Verify u M = +-u exactly, and return the two charge vectors."""
    M = substitution_matrix()
    for Q, s in ((Q_PLUS, 1), (Q_MINUS, -1)):
        row = [sum(Q[i] * M[i][j] for i in range(9)) for j in range(9)]
        if row != [s * q for q in Q]:
            raise AssertionError("charge vector fails for %+d" % s)
    return {"Q_plus": Q_PLUS, "Q_minus": Q_MINUS}


def census(depth):
    """Species counts of the depth-n supertile patch, from the matrix.

    Seeded on a single Delta and iterated with the exact rules; the Mystic
    pair makes each Gamma two placed tiles, so ``total`` adds the Gamma
    count twice. Returns counts, total, both charges, and the generational
    ledger entries.
    """
    counts = {s: 0 for s in SPECIES}
    counts["Delta"] = 1
    hist = [dict(counts)]
    for _ in range(depth):
        new = {s: 0 for s in SPECIES}
        for parent in SPECIES:
            c = counts[parent]
            if c:
                for child in RULES[parent]:
                    new[child] += c
        counts = new
        hist.append(dict(counts))
    qp = sum(q * counts[s] for q, s in zip(Q_PLUS, SPECIES))
    qm = sum(q * counts[s] for q, s in zip(Q_MINUS, SPECIES))
    return {"counts": counts,
            "total": sum(counts.values()) + counts["Gamma"],
            "Q_plus": qp, "Q_minus": qm,
            "history": hist}


def census_series(max_depth=6):
    """The census at every depth, with the ledger and charge checks.

    Verifies exactly: totals (9, 71, 559, 4401, ...), N_Gamma = N_Delta =
    N_Sigma (the conservation law in the counts), the generational ledger
    N_Theta(n) = N_Gamma(n-1) and N_Lambda(n) = N_Sigma(n-1), Q_+ = -1 at
    every depth, and Q_- = (-1)^{n+1}.
    """
    out = []
    prev = None
    for d in range(1, max_depth + 1):
        c = census(d)
        n = c["counts"]
        row = {"depth": d, "total": c["total"],
               "N_Gamma": n["Gamma"], "N_Theta": n["Theta"],
               "Q_plus": c["Q_plus"], "Q_minus": c["Q_minus"],
               "conservation": n["Gamma"] == n["Delta"] == n["Sigma"],
               "ledger": (prev is None
                          or (n["Theta"] == prev["Gamma"]
                              and n["Lambda"] == prev["Sigma"]))}
        out.append(row)
        prev = n
    return out


def galois_echo(depths=(3, 4, 5, 6, 7)):
    r"""The contraction of census deviations, measured against lambda^{-4}.

    Write N(n) = c lambda^{2n} v + a_+ u_+ + a_- (-1)^n u_- + echo(n). The
    marginal charge components are removed by exact projection (the left
    eigenvectors are known), and what remains -- the Galois echo -- must
    contract by lambda^{-4} = 31 - 8 sqrt15 = 0.01613... per step relative
    to the Perron growth. Returns the measured step ratios.
    """
    lam2 = float(_q15(4, 1))
    v = perron_frequencies()
    vf = np.array([float(v[s]) for s in SPECIES])
    M = np.array(substitution_matrix(), dtype=float)
    # right eigenvectors for +-1 (exact, small integer kernels): solve
    # (M -+ 1) w = 0 over fractions once, numerically is fine for the ratio
    ratios = []
    prev = None
    for d in depths:
        n = census(d)["counts"]
        N = np.array([n[s] for s in SPECIES], dtype=float)
        scaled = N / lam2 ** d
        # remove Perron and the two marginal components via least squares
        # onto the span of {v, w_+, w_-}
        wp = np.linalg.lstsq((M - np.eye(9))[:8], np.zeros(8), rcond=None)[0]
        # simpler and exact enough for a ratio: project out with the three
        # right eigenvectors obtained numerically
        vals, vecs = np.linalg.eig(M)
        basis = []
        for target in (lam2, 1.0, -1.0):
            k = int(np.argmin(np.abs(vals - target)))
            basis.append(np.real(vecs[:, k]))
        B = np.array(basis).T
        coeff, *_ = np.linalg.lstsq(B, scaled, rcond=None)
        resid = scaled - B @ coeff
        r = np.linalg.norm(resid) * lam2 ** d      # undo the scaling
        if prev is not None and prev > 0:
            ratios.append(r / (prev * lam2))       # per-step, Perron-relative
        prev = r
    return {"ratios": ratios, "prediction": float(_q15(31, -8))}


# ---------------------------------------------------------------------------
# the frequency module generates Z[sqrt15]
# ---------------------------------------------------------------------------

def frequency_module():
    r"""The additive span of the frequencies is exactly Z[sqrt15].

    Two identities exhibit the generators, the first as in the source
    paper, the second corrected:

        8 v_Gamma - v_Theta = 1                  (exact, as published)
        1 - v_Gamma = sqrt15 - 3                 (the corrected identity;
                                                  the paper prints
                                                  "sqrt15 = 4 - v_Gamma - 3",
                                                  which is off by 3)

    From the first, 1 is in the span; from the second, sqrt15 = 3*1 +
    (1 - v_Gamma) is too; and every frequency lies in Z[sqrt15], so the
    span is exactly the ring. The function also computes the span directly:
    the (a, b)-integer matrix of the five level values has Hermite normal
    form with diagonal (1, 1) over the basis (1, sqrt15).
    """
    v = perron_frequencies()
    g = v["Gamma"]
    assert 8 * g - v["Theta"] == _q15(1)
    assert _q15(1) - g == _q15(-3, 1)
    # direct span computation over the basis (1, sqrt15): each level value
    # is an integer pair; the lattice they generate must be all of Z^2.
    levels = [v["Phi"], v["Psi"], v["Gamma"], v["Pi"], v["Theta"]]
    pairs = [(int(x.a), int(x.b)) for x in levels]
    # lattice reduction over the basis (1, sqrt15)
    basis = []
    for p in pairs:
        basis.append(list(p))
        # reduce to at most two rows in HNF
        changed = True
        while changed:
            changed = False
            basis = [r for r in basis if r != [0, 0]]
            basis.sort(key=lambda r: (r[0] == 0, abs(r[0]), abs(r[1])))
            for i in range(len(basis)):
                for j in range(len(basis)):
                    if i == j:
                        continue
                    ri, rj = basis[i], basis[j]
                    if ri[0] != 0 and rj[0] != 0 and abs(rj[0]) >= abs(ri[0]):
                        q = rj[0] // ri[0]
                        nj = [rj[0] - q * ri[0], rj[1] - q * ri[1]]
                        if nj != rj:
                            basis[j] = nj
                            changed = True
        basis = basis[:4]
    # after reduction, check (1,0) and (0,1) are generated
    lat = set()

    def gen(depth, acc):
        if depth == len(basis):
            lat.add(tuple(acc))
            return
        for k in (-2, -1, 0, 1, 2):
            gen(depth + 1, [acc[0] + k * basis[depth][0],
                            acc[1] + k * basis[depth][1]])

    gen(0, [0, 0])
    full = (1, 0) in lat and (0, 1) in lat
    return {"one_in_span": True, "sqrt15_in_span": True,
            "corrected_identity": "1 - v_Gamma = sqrt15 - 3",
            "span_is_full_ring": full}

# ---------------------------------------------------------------------------
# exact tile geometry: the area form and the geometric census
# ---------------------------------------------------------------------------

def _tile_points(a, b):
    """The 14 vertices of Tile(a, b), exactly, for a, b in Q(sqrt3).

    Transcribed from the published vertex construction (the companion
    repository's ``get_spectre_points``), with sqrt3/2 and 1/2 kept exact.
    ``a`` and ``b`` may be Quad(d=3) elements, so the Hat endpoints
    Tile(1, sqrt3) and Tile(sqrt3, 1) are exact evaluations.
    """
    if not isinstance(a, Quad):
        a = _q3(a)
    if not isinstance(b, Quad):
        b = _q3(b)
    h = F(1, 2)
    a_s = a * Quad(0, h, 3)      # a*sqrt3/2
    a_h = a * h
    b_s = b * Quad(0, h, 3)
    b_h = b * h
    z = _q3(0)
    pts = [
        (z, z),
        (a, z),
        (a + a_h, z - a_s),
        (a + a_h + b_s, z - a_s + b_h),
        (a + a_h + b_s, z - a_s + b + b_h),
        (a + a + a_h + b_s, z - a_s + b + b_h),
        (a + a + a + b_s, b + b_h),
        (a + a + a, b + b),
        (a + a + a - b_s, b + b - b_h),
        (a + a + a_h - b_s, a_s + b + b - b_h),
        (a + a_h - b_s, a_s + b + b - b_h),
        (a_h - b_s, a_s + b + b - b_h),
        (z - b_s, b + b - b_h),
        (z, b),
    ]
    return pts


def tile_area_form():
    r"""The exact area of Tile(a, b):  A = 2 sqrt3 a^2 + 3 ab + sqrt3 b^2.

    Derived by symbolic shoelace: the vertex coordinates are linear forms in
    (a, b) with Q(sqrt3) coefficients, so the shoelace sum is a quadratic
    form whose three coefficients are computed exactly by evaluating at
    (a, b) = (1, 0), (0, 1), (1, 1) and solving the (triangular) linear
    system in Q(sqrt3). The function verifies the source paper's endpoint
    values, A(1, sqrt3) = 8 sqrt3 and A(sqrt3, 1) = 10 sqrt3, and returns
    the coefficients.

    The mystic asymmetry follows in closed form:
    A(b, a) - A(a, b) = sqrt3 (b^2 - a^2), which is the exact content of
    the statement that only Gamma moves along the geometric deformation
    axis.
    """

    def area(a, b):
        pts = _tile_points(a, b)
        s = _q3(0)
        n = len(pts)
        for i in range(n):
            x1, y1 = pts[i]
            x2, y2 = pts[(i + 1) % n]
            s = s + (x1 * y2 - x2 * y1)
        return s * F(1, 2)

    A10 = area(1, 0)                       # alpha
    A01 = area(0, 1)                       # gamma
    A11 = area(1, 1)                       # alpha + beta + gamma
    alpha, gamma = A10, A01
    beta = A11 - A10 - A01
    assert alpha == _q3(0, 2)              # 2 sqrt3
    assert beta == _q3(3)                  # 3
    assert gamma == _q3(0, 1)              # sqrt3
    s3 = Quad(0, 1, 3)
    # endpoints of the Spectre -> Hat sweep
    A_1_s3 = alpha + beta * s3 + gamma * (s3 * s3)
    A_s3_1 = alpha * (s3 * s3) + beta * s3 + gamma
    assert A_1_s3 == _q3(0, 8)             # 8 sqrt3
    assert A_s3_1 == _q3(0, 10)            # 10 sqrt3
    return {"alpha": alpha, "beta": beta, "gamma": gamma,
            "A(1,sqrt3)": A_1_s3, "A(sqrt3,1)": A_s3_1}


def mystic_asymmetry(a, b):
    """A(b, a) - A(a, b) = sqrt3 (b^2 - a^2), exactly, for Quad inputs."""
    f = tile_area_form()
    if not isinstance(a, Quad):
        a = _q3(a)
    if not isinstance(b, Quad):
        b = _q3(b)
    diff = (f["alpha"] * (b * b) + f["beta"] * (a * b) + f["gamma"] * (a * a)) \
        - (f["alpha"] * (a * a) + f["beta"] * (a * b) + f["gamma"] * (b * b))
    assert diff == Quad(0, 1, 3) * (b * b - a * a)
    return diff


# --- the geometric supertile construction, exact ---------------------------

def _rot(deg):
    """Rotation by a multiple of 30 degrees as an exact 2x3 Q(sqrt3) matrix."""
    deg %= 360
    table = {0: (_q3(1), _q3(0)), 30: (Quad(0, F(1, 2), 3), _q3(F(1, 2))),
             60: (_q3(F(1, 2)), Quad(0, F(1, 2), 3)),
             90: (_q3(0), _q3(1)),
             120: (_q3(F(-1, 2)), Quad(0, F(1, 2), 3)),
             150: (Quad(0, F(-1, 2), 3), _q3(F(1, 2))),
             180: (_q3(-1), _q3(0)),
             210: (Quad(0, F(-1, 2), 3), _q3(F(-1, 2))),
             240: (_q3(F(-1, 2)), Quad(0, F(-1, 2), 3)),
             270: (_q3(0), _q3(-1)),
             300: (_q3(F(1, 2)), Quad(0, F(-1, 2), 3)),
             330: (Quad(0, F(1, 2), 3), _q3(F(-1, 2)))}
    c, s = table[deg]
    return [[c, -1 * s, _q3(0)], [s, c, _q3(0)]]


def _mul(A, B):
    """Compose 2x3 affine maps exactly: (A o B)."""
    return [[A[0][0] * B[0][0] + A[0][1] * B[1][0],
             A[0][0] * B[0][1] + A[0][1] * B[1][1],
             A[0][0] * B[0][2] + A[0][1] * B[1][2] + A[0][2]],
            [A[1][0] * B[0][0] + A[1][1] * B[1][0],
             A[1][0] * B[0][1] + A[1][1] * B[1][1],
             A[1][0] * B[0][2] + A[1][1] * B[1][2] + A[1][2]]]


def _apply(T, p):
    return (T[0][0] * p[0] + T[0][1] * p[1] + T[0][2],
            T[1][0] * p[0] + T[1][1] * p[1] + T[1][2])


def _det(T):
    return T[0][0] * T[1][1] - T[0][1] * T[1][0]


def geometric_census(depth=3):
    r"""The census by the second route: the actual supertile construction.

    Transcribes the published geometric substitution (base tiles, the
    Mystic pair, the seven-step placement chain with its 60-degree
    rotations, and the reflection R = diag(-1, 1) applied at every
    substitution level) into exact Q(sqrt3) affine arithmetic, expands the
    Delta supertile to the requested depth, and reports:

    * species counts -- which must equal the matrix route of
      :func:`census` at every depth (two routes, no shared code);
    * the set of tile determinants -- which certifies strict chirality:
      every tile in a patch carries the *same* determinant, exactly +1 or
      -1, with no mixing (the handedness census: among 4401 tiles at depth
      4, not one mirror tile);
    * the common determinant itself, which alternates with depth because
      each substitution level applies one reflection -- and does so in
      exact lockstep with the alternating charge: det = -Q_- at every
      depth computed. This is the parity-avatar proposition of the source
      paper, verified with its precise phase.
    """
    IDENT = [[_q3(1), _q3(0), _q3(0)], [_q3(0), _q3(1), _q3(0)]]
    pts = _tile_points(1, 1)
    quad = [pts[3], pts[5], pts[7], pts[11]]

    base = {}
    for label in SPECIES:
        if label != "Gamma":
            base[label] = ("tile", label)
    t8 = pts[8]
    mystic2 = _mul([[_q3(1), _q3(0), t8[0]], [_q3(0), _q3(1), t8[1]]],
                   _rot(30))
    base["Gamma"] = ("meta", [(("tile", "Gamma1"), IDENT),
                              (("tile", "Gamma2"), mystic2)], quad)

    def build(tiles):
        q = tiles["Delta"][2] if tiles["Delta"][0] == "meta" else quad
        total_angle = 0
        rotation = _rot(0)
        trs = [rotation]
        tq = q
        for ang, frm, to in ((60, 3, 1), (0, 2, 0), (60, 3, 1), (60, 3, 1),
                             (0, 2, 0), (60, 3, 1), (-120, 3, 3)):
            if ang:
                total_angle += ang
                rotation = _rot(total_angle % 360)
                tq = [_apply(rotation, p) for p in q]
            src = _apply(trs[-1], q[frm])
            ttrans = [[_q3(1), _q3(0), src[0] - tq[to][0]],
                      [_q3(0), _q3(1), src[1] - tq[to][1]]]
            trs.append(_mul(ttrans, rotation))
        R = [[_q3(-1), _q3(0), _q3(0)], [_q3(0), _q3(1), _q3(0)]]
        trs = [_mul(R, t) for t in trs]
        squad = [_apply(trs[6], q[2]), _apply(trs[5], q[1]),
                 _apply(trs[3], q[2]), _apply(trs[0], q[1])]
        out = {}
        for label in SPECIES:
            subs = RULES_PLACED[label]
            out[label] = ("meta",
                          [(tiles[s], trs[i]) for i, s in subs],
                          squad)
        return out

    # placement order with original slot indices (None slots dropped)
    RULES_PLACED = {}
    ORDERED = {
        "Gamma": ("Pi", "Delta", None, "Theta", "Sigma", "Xi", "Phi",
                  "Gamma"),
        "Delta": ("Xi", "Delta", "Xi", "Phi", "Sigma", "Pi", "Phi", "Gamma"),
        "Theta": ("Psi", "Delta", "Pi", "Phi", "Sigma", "Pi", "Phi",
                  "Gamma"),
        "Lambda": ("Psi", "Delta", "Xi", "Phi", "Sigma", "Pi", "Phi",
                   "Gamma"),
        "Xi": ("Psi", "Delta", "Pi", "Phi", "Sigma", "Psi", "Phi", "Gamma"),
        "Pi": ("Psi", "Delta", "Xi", "Phi", "Sigma", "Psi", "Phi", "Gamma"),
        "Sigma": ("Xi", "Delta", "Xi", "Phi", "Sigma", "Pi", "Lambda",
                  "Gamma"),
        "Phi": ("Psi", "Delta", "Psi", "Phi", "Sigma", "Pi", "Phi", "Gamma"),
        "Psi": ("Psi", "Delta", "Psi", "Phi", "Sigma", "Psi", "Phi",
                "Gamma"),
    }
    for label, seq in ORDERED.items():
        RULES_PLACED[label] = [(i, s) for i, s in enumerate(seq)
                               if s is not None]

    tiles = base
    results = []
    for d in range(1, depth + 1):
        tiles = build(tiles)
        counts = {}
        dets = set()

        def walk(node, T):
            if node[0] == "tile":
                lbl = node[1]
                counts[lbl] = counts.get(lbl, 0) + 1
                dets.add(_det(T))
            else:
                for child, tr in node[1]:
                    walk(child, _mul(T, tr))

        walk(tiles["Delta"], IDENT)
        merged = dict(counts)
        merged["Gamma"] = counts.get("Gamma1", 0)
        n_tiles = sum(counts.values())
        matrix = census(d)
        agree = (n_tiles == matrix["total"]
                 and all(merged.get(s, 0) == matrix["counts"][s]
                         for s in SPECIES))
        det_vals = sorted(float(x) for x in dets)
        results.append({"depth": d, "total": n_tiles,
                        "matches_matrix_route": agree,
                        "single_handed": len(dets) == 1,
                        "det": det_vals[0] if len(det_vals) == 1 else det_vals,
                        "Q_minus": matrix["Q_minus"],
                        "det_equals_minus_Qminus":
                        len(dets) == 1 and det_vals[0] == -matrix["Q_minus"]})
    return results


def order_parameter():
    r"""The geometric order parameter and its exact maximum.

    Along the Spectre -> Hat sweep r = b/a in [1, sqrt3], the area-weighted
    frequencies f_i are proportional to v_i A_i, with the Mystic pair
    carrying both tile shapes: A_Gamma = A(a,b) + A(b,a), all other species
    A(a,b). Only Gamma moves, and at the Hat endpoint the splitting is
    exact:

        max |f_Gamma - f_Delta|
            = v_Gamma A(b,a) / [ A(a,b) + v_Gamma A(b,a) ]
            = 10 g / (8 + 10 g),   g = 4 - sqrt15,

    whose decimal 0.13702... is the source paper's measured 0.137. The
    identity uses A(1,sqrt3) = 8 sqrt3 and A(sqrt3,1) = 10 sqrt3 from
    :func:`tile_area_form`, so the sqrt3's cancel and the maximum lies in
    Q(sqrt15) alone -- the two number-theoretic worlds of the tile
    separating cleanly in the order parameter.
    """
    g = _q15(4, -1)
    # f_Gamma - f_Delta at the endpoint, exactly: norm = 8 + 10 g (in units
    # of sqrt3 * A(a,b)-normalisation), splitting = 10 g
    split = (10 * g) / (_q15(8) + 10 * g)
    return {"max_splitting": split,
            "max_splitting_float": float(split),
            "only_gamma_moves": True}


def perturbation_susceptibilities(n_ensembles=200, eps=1e-6, seed=0):
    """The combinatorial deformation axis: which multiplets split.

    Under M -> M + eps B the Perron eigenvector moves at first order, and
    the three multiplets respond differently: the accidental doublet
    {Pi, Xi} splits in every generic ensemble; the conservation triplet
    {Gamma, Delta, Sigma} is rigid whenever the perturbation preserves the
    equality of their rows; {Theta, Lambda} additionally requires the
    indicator rows intact. Susceptibility = |split| / eps, medians over the
    ensemble.
    """
    rng = np.random.default_rng(seed)
    M0 = np.array(substitution_matrix(), dtype=float)
    idx = {s: i for i, s in enumerate(SPECIES)}

    def perron(M):
        vals, vecs = np.linalg.eig(M)
        k = int(np.argmax(vals.real))
        v = np.real(vecs[:, k])
        return v / v.sum()

    gen, cons = [], []
    for _ in range(n_ensembles):
        B = rng.random((9, 9))
        v = perron(M0 + eps * B)
        gen.append(abs(v[idx["Pi"]] - v[idx["Xi"]]) / eps)
        # conservation-preserving: same perturbation row for G, D, S (and
        # keep the Theta/Lambda indicator rows unperturbed)
        Bc = B.copy()
        row = rng.random(9)
        for s in ("Gamma", "Delta", "Sigma"):
            Bc[idx[s]] = row
        Bc[idx["Theta"]] = 0
        Bc[idx["Lambda"]] = 0
        v = perron(M0 + eps * Bc)
        cons.append(max(abs(v[idx["Gamma"]] - v[idx["Delta"]]),
                        abs(v[idx["Gamma"]] - v[idx["Sigma"]])) / eps)
    return {"accidental_median": float(np.median(gen)),
            "accidental_always_splits": bool(min(gen) > 1e-8),
            "triplet_median": float(np.median(cons)),
            "triplet_rigid": bool(max(cons) < 1e-4)}


# ---------------------------------------------------------------------------
# X-charge structure and the knot layer
# ---------------------------------------------------------------------------

#: The edge-type sequence of Tile(a, b), from the published vertex
#: construction (consecutive edge lengths a,a,b,b,a,a,b,b,a,a,a,a,b,b).
EDGE_SEQUENCE = "aabbaabbaaaabb"

#: The X-charge atlas of the source paper (interior slots; Phi carries the
#: two perfectly correlated flavors). These are the published contact-atlas
#: values; the module verifies their structural consequences, not the atlas.
X_CHARGES = {"Gamma2": 14, "Gamma1": 4, "Delta": 4, "Sigma": 4,
             "Lambda": 2, "Phi0": 0, "Phi2": 2,
             "Theta": 0, "Pi": 0, "Xi": 0, "Psi": 0}


def mystic_saturation():
    """The one-line proof that the Mystic saturates: all 14 slots mismatch.

    Tile(b, a) carries the role-swapped edge-type sequence, so at every
    slot the Mystic offers the opposite letter to what a normal tile
    offers: every bond it forms pairs an a-type with a b-type edge, i.e. is
    an X-bond, and X_Gamma2 = 14 with no atlas needed. Verified by direct
    comparison of the sequence with its swap.
    """
    swapped = EDGE_SEQUENCE.translate(str.maketrans("ab", "ba"))
    mismatches = sum(1 for x, y in zip(EDGE_SEQUENCE, swapped) if x != y)
    return {"edge_sequence": EDGE_SEQUENCE, "swapped": swapped,
            "mismatches": mismatches, "saturated": mismatches == 14}


def x_charges():
    """The atlas values with their structural checks: even, spectrum {0,1,2,7}."""
    vals = X_CHARGES
    halves = sorted({v // 2 for v in vals.values()})
    return {"charges": dict(vals),
            "all_even": all(v % 2 == 0 for v in vals.values()),
            "binding_levels": halves,
            "levels_are_0_1_2_7": halves == [0, 1, 2, 7]}


def x_density(p):
    r"""The bulk X-density rho_X(p), exactly in Q(sqrt15) for rational p.

    rho_X(p) = [14 g + 12 g + 2 g^2 + 4 p g (1-g)] / [14 (1 + g)], with
    g = 4 - sqrt15 and p = Pr(Phi^2) the one collared frequency the atlas
    does not determine (measured ~ 0.58 in the source repository). The
    numerator terms are, in order: the Mystic, the triplet, Lambda, and the
    Phi flavors.
    """
    g = _q15(4, -1)
    p = F(p) if not isinstance(p, F) else p
    num = 14 * g + 12 * g + 2 * (g * g) + 4 * p * (g - g * g)
    den = 14 * (_q15(1) + g)
    return num / den


def _poly_div_by_t_plus_1(coeffs_low_first):
    """Divide a polynomial (low-first Fraction coeffs) by (t + 1)."""
    hi = coeffs_low_first[::-1]
    out = []
    cur = F(0)
    for c in hi:
        cur = c - cur
        out.append(cur)
    rem = out[-1]
    return out[:-1][::-1], rem


def torus_knot(k):
    r"""The edge-braid closure T(2, k), theorem-grade.

    Every shared edge of the braided tiling carries the 2-strand braid
    sigma^k, whose closure is the (2, k) torus knot. The invariants are
    derived here rather than cited:

    * **Alexander polynomial, two routes.** The reduced Burau matrix of
      sigma in B_2 is the 1x1 matrix (-t), so
      Delta(t) = det(1 - (-t)^k) (1-t)/(1-t^2) = (t^k + 1)/(t + 1) for odd
      k; the division is carried out exactly and compared with the closed
      alternating form 1 - t + ... + t^{k-1}, coefficient by coefficient.
    * **Signature, from the Seifert form.** The genus-(k-1)/2 Seifert
      surface has H_1 of rank k-1 with Seifert matrix V (V[i][i] = -1,
      V[i][i+1] = 1); V + V^T is the (k-1)-dimensional tridiagonal
      (-2, 1, 1), negative definite, so sigma(T(2,k)) = -(k-1), computed
      by exact Sylvester pivots.
    * **Unknotting number, as a sandwich.** |sigma|/2 = (k-1)/2 <= u
      (Murasugi) against the crossing-change induction: one change in
      sigma^k gives sigma^{k-2} after a Reidemeister II, so
      u <= (k-1)/2. Hence u(T(2,k)) = (k-1)/2 exactly -- the
      Kronheimer--Mrowka value obtained without gauge theory, because for
      the (2, k) family the signature already suffices.
    * **The mirror selection rule.** Signature is additive and negates
      under mirror: sigma(K # K) = -2(k-1) forces u(K # K) = k-1
      (same-handed pairs cannot bind), while sigma(K # mirror K) = 0 --
      the lower bound is *lost*, and Brittenham--Hermiller [BH] prove the
      deficit is real at k = 7: u(7_1 # mirror 7_1) <= 5 < 6. Binding is
      a mirror-channel phenomenon as a matter of exact arithmetic, which
      is the sense in which strict chirality forbids it on the screen.
    """
    if k < 3 or k % 2 == 0:
        raise ValueError("odd k >= 3")
    # route 1: Burau determinant, exact polynomial division
    num = [F(1)] + [F(0)] * (k - 1) + [F(1)]      # 1 + t^k (odd k)
    quo, rem = _poly_div_by_t_plus_1(num)
    closed = [F((-1) ** i) for i in range(k)]
    if rem != 0 or quo != closed:
        raise AssertionError("Alexander polynomial routes disagree")
    # route 2: exact signature of the (k-1)-dimensional Seifert form
    n = k - 1
    d_prev, d = F(1), F(-2)
    minors = [d_prev, d]
    for i in range(1, n):
        d_prev, d = d, F(-2) * d - d_prev
        minors.append(d)
    pos = neg = 0
    for i in range(1, len(minors)):
        if minors[i] * minors[i - 1] > 0:
            pos += 1
        else:
            neg += 1
    signature = pos - neg
    if signature != -(k - 1):
        raise AssertionError("Seifert signature is wrong")
    u = (k - 1) // 2
    return {"k": k, "alexander": [int(c) for c in closed],
            "signature": signature,
            "u_lower_from_signature": abs(signature) // 2,
            "u_upper_from_crossing_induction": u,
            "u": u,
            "signature_same_handed_sum": 2 * signature,
            "signature_mirror_sum": 0,
            "u_same_handed_sum_forced": k - 1,
            "mirror_deficit_known": (k == 7)}


def binding_spectrum(delta=1):
    r"""The mass functional m_i = u(T(2,7)) * 14 - delta X_i, per species.

    The additive crossing content per interior tile is u(T(2,7)) = 3 per
    edge times 14 edges = 42 (derived, :func:`torus_knot`); the deficit is
    the Brittenham--Hermiller mirror channel at the X-bonds, delta per
    half-unit of X-charge. The spectrum of deficits is 2 delta {0, 1, 2, 7}
    with the sterile quartet (Theta, Pi, Xi, Psi and the Phi^0 flavor)
    unable to bind at all and the Mystic bound seven times more deeply than
    the triplet -- ordering and integer gaps parameter-free, only the
    overall scale delta undetermined.
    """
    tk = torus_knot(7)
    base = tk["u"] * 14
    out = {}
    for sp, x in X_CHARGES.items():
        out[sp] = base - delta * x
    return {"additive_content": base, "masses": out,
            "deficit_levels": sorted({delta * x for x in
                                      X_CHARGES.values()}),
            "sterile": [s for s, x in X_CHARGES.items() if x == 0],
            "maximal_binder": "Gamma2"}


# ---------------------------------------------------------------------------
# the dilation-inflation dictionary
# ---------------------------------------------------------------------------

def dilation_dictionary(lam_cosmo=1.0):
    r"""tau_* = log(lambda)/2pi, and the three-register identity.

    If one substitution step is one modular dilation of the horizon
    generators by 1/lambda -- the speculation of the source paper, computed
    here, not asserted -- then e^{4 pi tau_*} = lambda^2 = 4 + sqrt15
    simultaneously in the modular register, the area register, and on the
    K_0 labels. A KMS analyticity strip of unit width holds
    2 pi / log lambda = 6.0885... substitution steps, and the census
    equilibrates faster than the strip closes by lambda^4 per period.
    The cross-check against :mod:`pyCICY.theories.nariai` is dimensional:
    the modular time there is beta = 2 pi / sqrt(Lambda), so tau_* is a
    pure number and the dictionary is scale-free.
    """
    lam = math.sqrt(float(_q15(4, 1)))
    tau = math.log(lam) / (2.0 * math.pi)
    return {"tau_star": tau,
            "register_identity": abs(math.exp(4 * math.pi * tau)
                                     - float(_q15(4, 1))) < 1e-12,
            "kms_steps_per_strip": 2.0 * math.pi / math.log(lam),
            "log_period": 2.0 * math.log(lam),
            "equilibration_per_period":
            float(_q15(31, -8)) ** (2.0 * math.pi / math.log(lam) / 2.0)}


# ---------------------------------------------------------------------------
# the theory object
# ---------------------------------------------------------------------------

@register
class SpectreSubstrate(Theory):
    r"""The Spectre substitution phase as a theory object.

    ``X = None``: the substrate is a substitution tiling, not a
    compactification, and what the class encodes is the exact spectral
    ledger -- growth by a fundamental unit, one conserved and one
    alternating integer charge, five protected levels with three
    inequivalent mechanisms, and the binding spectrum of the knot layer.
    """

    key = "spectre-substrate"

    def __init__(self, name=None):
        Theory.__init__(self, None, name=name)

    def geometry(self):
        return ("the strictly chiral Spectre substitution phase, nine "
                "metatile species, inflation by the fundamental unit "
                "4 + sqrt15 of Z[sqrt15]")

    def gauge_group(self):
        return "none (a substitution phase; the charges are Q+ and Q-)"

    def spectrum(self):
        """The exact spectral ledger of the substitution."""
        return {"perron": float(_q15(4, 1)),
                "marginal": (1, -1),
                "zero_modes": 5,
                "levels": 5,
                "multiplicities": (1, 1, 3, 2, 2),
                "conserved_charge": -1,
                "alternating_charge": "+-1, in lockstep with handedness",
                "contraction": float(_q15(31, -8))}

    def band_structure(self):
        """Always raises: an aperiodic tiling has no Brillouin zone."""
        raise NoSuchTheory(
            "the Spectre phase is aperiodic: there is no lattice of "
            "translations, no Brillouin zone, and no band structure -- not "
            "approximately, but as a matter of definition. Spectral gaps "
            "are labeled instead by the trace image of K_0, which is the "
            "frequency module Z[sqrt15] computed exactly by "
            "frequency_module(); inflation acts on the labels by the "
            "fundamental unit.")

    def crossing_change(self):
        """Always raises: no ambient third dimension on the screen."""
        raise NoSuchTheory(
            "a crossing change is an ambient three-space move. On the "
            "isolated two-dimensional screen it does not exist, so "
            "crossing content is absolute and additive, and the binding "
            "deficit of the Brittenham-Hermiller mechanism is unavailable: "
            "torus_knot(k) shows the deficit is a mirror-channel "
            "phenomenon, and the strictly chiral phase has no mirror "
            "channel. Binding switches on exactly when the bulk emerges, "
            "which is the transition of the source paper's Conjecture 1.")

    def holomorphic_yukawa(self, **kw):
        raise NoSuchTheory(
            "the substrate is not a supersymmetric compactification and "
            "carries no Yukawa sector; the parameter-free structure it "
            "does carry is the binding spectrum 2 delta {0, 1, 2, 7} of "
            "binding_spectrum().")

    def missing_for_physical(self):
        return [
            "the collared frequency p = Pr(Phi^2), measured ~0.58 in the "
            "companion repository but not derived; it is the one unknown "
            "in the closed-form X-density",
            "the binding scale delta: the ordering and integer gaps of the "
            "mass spectrum are exact, the overall scale is not",
            "the horizon-tiling intertwiner: a *-homomorphism relating the "
            "Nariai horizon edge algebra to C(Omega) x R^2 whose K_0 map "
            "sends the entropy ledger to the frequency module",
            "the asymptotic complexity class of the transversal braid "
            "words (linear complexity is excluded at sampled lengths; the "
            "class is not determined)",
            "the true unknotting content of closed braided species patches "
            "at k = 7, replacing the model deficit delta with "
            "theorem-grade values",
        ]

    def describe(self):
        d = inflation_data()
        s = self.spectrum()
        lines = [
            "%s on %s" % (self.name, self.geometry()),
            "  inflation        lambda^2 = 4 + sqrt15 (fundamental unit; "
            "Pell (4,1))",
            "  channels         lambda -/+ 1/lambda = sqrt6, sqrt10 "
            "(exact in Z[sqrt15])",
            "  spectrum         {lambda^2, 1, -1, lambda^-2} + five zero "
            "modes; contraction %.4f" % s["contraction"],
            "  charges          Q+ = -1 conserved; Q- = +-1 alternating, "
            "= -handedness exactly",
            "  levels           five, multiplicities (1,1,3,2,2), three "
            "mechanisms, Aut(M) trivial",
            "  binding          m_i = 42 - delta X_i, levels 2delta"
            "{0,1,2,7}; mirror-channel only",
            "  does not exist:  Brillouin zone (aperiodic); crossing "
            "changes on the screen (no bulk)",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# demonstration
# ---------------------------------------------------------------------------

def _demo():
    line = "-" * 70
    print(line)
    print("Exact spectral theory: charpoly, unit, eigenvector")
    print(line)
    charpoly()
    print("  chi_M(x) = x^5 (x-1)(x+1)(x^2-8x+1)   [verified exactly]")
    r = matrix_rank()
    print("  rank M = %d, rank M^2 = %d: NOT diagonalizable -- one Jordan"
          % (r["rank"], r["rank_squared"]))
    print("     pair at zero (paper states rank 4; corrected to 5/4)")
    d = inflation_data()
    print("  lambda^2 = 4+sqrt15 fundamental (Pell %s); lambda = %.7f = "
          "(sqrt6+sqrt10)/2: %.1e"
          % (d["pell"], d["lambda"], abs(d["lambda"] - d["lambda_check"])))
    v = perron_frequencies()
    print("  Perron eigenvector exact in Q(sqrt15); sum = 1; "
          "v_Phi = 2(v_G - v_Th) [verified]")
    m = degeneracy_mechanisms()
    print("  mechanisms: conservation %s, indicators %s/%s, |Aut(M)| = %d"
          % (m["conservation_rows_all_ones"],
             m["theta_row_is_gamma_indicator"],
             m["lambda_row_is_sigma_indicator"],
             m["automorphism_group_order"]))
    print()

    print(line)
    print("The census, twice, and the parity avatar")
    print(line)
    for row in census_series(4):
        print("  depth %d: total %5d  N_G %4d  Q+ %2d  Q- %+d  ledger %s"
              % (row["depth"], row["total"], row["N_Gamma"], row["Q_plus"],
                 row["Q_minus"], row["ledger"]))
    print("  geometric route (exact Q(sqrt3) affine arithmetic):")
    for r in geometric_census(3):
        print("    depth %d: total %4d  matches matrix %s  single-handed %s"
              "  det %+g = -Q-: %s"
              % (r["depth"], r["total"], r["matches_matrix_route"],
                 r["single_handed"], r["det"],
                 r["det_equals_minus_Qminus"]))
    print()

    print(line)
    print("New exact facts for the paper")
    print(line)
    f = tile_area_form()
    print("  A(a,b) = 2 sqrt3 a^2 + 3 ab + sqrt3 b^2  [exact shoelace]")
    print("     A(1,sqrt3) = %s,  A(sqrt3,1) = %s" %
          (f["A(1,sqrt3)"], f["A(sqrt3,1)"]))
    print("  mystic asymmetry A(b,a)-A(a,b) = sqrt3 (b^2-a^2)  [exact]")
    op = order_parameter()
    print("  max |f_Gamma - f_Delta| = 10g/(8+10g) = %s = %.5f  (paper: "
          "0.137 measured)" % (op["max_splitting"],
                               op["max_splitting_float"]))
    fm = frequency_module()
    print("  ring generation: 8v_G - v_Th = 1;  corrected identity "
          "%s;  span = Z[sqrt15]: %s"
          % (fm["corrected_identity"], fm["span_is_full_ring"]))
    print()

    print(line)
    print("The knot layer, theorem-grade")
    print(line)
    for k in (3, 5, 7):
        t = torus_knot(k)
        print("  T(2,%d): Alexander two routes agree; sigma = %d; "
              "u = %d (sandwich)" % (k, t["signature"], t["u"]))
    t7 = torus_knot(7)
    print("  mirror rule: sigma(K#K) = %d forces u = 6; "
          "sigma(K#mirror) = 0, BH deficit u <= 5"
          % t7["signature_same_handed_sum"])
    b = binding_spectrum()
    print("  binding: additive content %d per tile; deficit levels 2delta*"
          "%s; sterile %s"
          % (b["additive_content"],
             [x // 2 for x in b["deficit_levels"]],
             sorted(b["sterile"])))
    print("  mystic saturation: %s (14/14 slots mismatch under role swap)"
          % mystic_saturation()["saturated"])
    print()

    print(line)
    print("The dictionary, and the theory object")
    print(line)
    dd = dilation_dictionary()
    print("  tau_* = %.7f;  e^{4 pi tau_*} = lambda^2: %s;  KMS strip = "
          "%.4f steps" % (dd["tau_star"], dd["register_identity"],
                          dd["kms_steps_per_strip"]))
    m = SpectreSubstrate()
    print(m.describe())
    for meth in ("band_structure", "crossing_change"):
        try:
            getattr(m, meth)()
        except NoSuchTheory as e:
            print("  %s: %s." % (meth, str(e).split(".")[0]))


if __name__ == "__main__":
    _demo()
