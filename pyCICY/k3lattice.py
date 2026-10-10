r"""
pyCICY.k3lattice -- is the transcendental quadratic space of a K3 surface on
the "quadratic locus"?  An exact decision, over Q.

Where the question comes from
-----------------------------
An OpenAI manuscript of 30 September 2026, *Algebraic Kuga-Satake
correspondences and Hodge conjectures on a K3 quadratic locus*
(github.com/openai/math, family 032), states in its abstract that the rational
Hodge and generalised Hodge conjectures hold, in every cohomological degree,
for every self-power of a projective complex K3 surface

    "whose transcendental quadratic space, with its cup-product form, admits a
     rational isometric embedding in  U_Q^{+2} (+) <-1>^4",

and that this includes every ample P-polarised K3 surface, Picard jumps
included, for  P = U (+) D8(-1) (+) D4(-1).

What this module is, and is not
-------------------------------
It is not a check of that theorem. Algebraicity of Kuga-Satake tensors, and
hence of Hodge classes on self-powers, is exactly the statement the manuscript
claims and this package cannot test. Three sibling manuscripts from the same
release (on split abelian eightfolds, Kuga-Satake correspondences for K3
surfaces, and products of K3 surfaces) were withdrawn on 6 October 2026 for a
sign error in one shared step; the preprint above is not on that list, but this
module has not checked whether it uses the withdrawn construction.

What *is* exact is the hypothesis. "Does this quadratic space embed in that
one?" is a finite computation over Q, decided by the Hasse-Minkowski theorem:
two rational quadratic forms are isometric iff they have the same rank,
discriminant, signature and Hasse invariant at every prime, and a form with
prescribed local invariants exists iff the product formula holds and a pair of
small-rank local conditions is met (Serre, *A Course in Arithmetic*, IV.3.3,
Prop. 7). So  V  embeds in  W  iff the would-be orthogonal complement  V'  has
admissible invariants -- which this module computes, rather than searches for.

The ledger
----------
    exact              the Hilbert symbol at every place; the invariants of a
                       rational form; isometry over Q; embedding of one
                       rational form in another, with the obstruction named
                       (rank, signature, a local condition at a named prime, or
                       the product formula); the transcendental space of a K3
                       from its Picard lattice; the quadratic-locus verdict;
                       double intersection numbers of a CICY K3.
    two routes         the complement's invariants from the Hasse-invariant
                       formula and from an explicit orthogonal complement
                       (:func:`explicit_complement`, linear algebra over Q).
    not computed       algebraicity of anything. Whether a given K3 exists with
                       a given Picard lattice (Nikulin's integral embedding
                       criterion is not implemented; only the *rational* one,
                       which is all the quoted hypothesis asks for). The
                       generic Picard rank of a CICY K3 beyond what its ambient
                       divisors span.
    does not exist     a certificate that a K3 is *outside* the locus from a
                       sublattice of its Picard lattice: see
                       :func:`cicy_k3_verdict`.

The rank bound
--------------
H^2(K3, Z) = U^3 (+) E8(-1)^2 has rank 22 and signature (3, 19). The
transcendental space T has rank 22 - rho and signature (2, 20 - rho). The
target space U^2 (+) <-1>^4 has rank 8 and signature (2, 6), so an embedding
needs 22 - rho <= 8, that is **rho >= 14**. The generic quartic (rho = 1) is
therefore nowhere near the locus, and the P-polarised family is the extreme
case: rank(P) = 2 + 8 + 4 = 14, so T and the target have equal rank and the
embedding is an isometry T_Q ~ U^2 (+) <-1>^4. (It is: D4 has index 2 in Z^4,
so D4 (x) Q is <1>^4, and D8 (x) Q = E8 (x) Q.)

Conventions
-----------
Quadratic spaces are given by symmetric rational Gram matrices; lattices enter
only through their rational span. The real place is written ``INFINITY`` (= 0);
every other place is a prime. The Hasse invariant is
eps_p = prod_{i<j} (a_i, a_j)_p for a diagonalisation <a_1, ..., a_n>, and the
discriminant is the product of the a_i modulo squares.

References
----------
J.-P. Serre, *A Course in Arithmetic*, Chapters III and IV.
J. W. S. Cassels, *Rational Quadratic Forms*, Chapters 4 and 6.
OpenAI Math release (github.com/openai/math), family 032 and history.md.
"""

from fractions import Fraction
from functools import lru_cache

from sympy import Matrix, factorint

__all__ = [
    "INFINITY", "DegenerateForm", "NotAK3Lattice", "AmbientDegenerate",
    "square_class", "hilbert", "diagonalize",
    "QSpace", "Embedding", "embeds", "complement", "explicit_complement",
    "gram_of", "lattice_U", "cartan_A", "cartan_D", "cartan_E8", "negate",
    "direct_sum", "identity_form",
    "MIN_PICARD_RANK", "k3_h2", "k3_locus_form", "transcendental_space",
    "Verdict", "locus_verdict", "double_intersection", "ambient_picard_gram",
    "cicy_k3_verdict",
]

INFINITY = 0

#: the least Picard rank a K3 on the quadratic locus can have (see module doc)
MIN_PICARD_RANK = 14


class DegenerateForm(ValueError):
    """The Gram matrix is singular. Refused rather than quotiented."""


class NotAK3Lattice(ValueError):
    """The input cannot be (the rational span of) a Picard lattice."""


class AmbientDegenerate(ValueError):
    """The ambient divisor classes of a CICY K3 span a degenerate subspace.

    The rank of their span inside NS(X) is then unknown from the Gram matrix
    alone, so no verdict is offered.
    """


# --------------------------------------------------------------------------
# exact arithmetic helpers
# --------------------------------------------------------------------------

def _frac(x):
    """An exact rational from an int, Fraction or sympy Rational.

    Floats are accepted only when integral: a float Gram matrix is a sign that
    some earlier step was not exact, and this module does not repair that.
    """
    if isinstance(x, Fraction):
        return x
    if hasattr(x, "p") and hasattr(x, "q"):
        return Fraction(int(x.p), int(x.q))
    if isinstance(x, float):
        if x != int(x):
            raise TypeError("non-integral float %r: exact input required" % x)
        return Fraction(int(x))
    return Fraction(x)


def square_class(q):
    """The squarefree integer in the class of the nonzero rational ``q``
    modulo squares of rationals (sign kept).

    n/d is congruent to n*d modulo squares, so only an integer is factored.
    """
    q = _frac(q)
    if q == 0:
        raise ValueError("0 has no square class")
    n = q.numerator * q.denominator
    sign = -1 if n < 0 else 1
    r = 1
    for p, e in factorint(abs(n)).items():
        if e % 2:
            r *= p
    return sign * r


def _val_unit(a, p):
    """a = p^v * u with u prime to p (a a nonzero integer)."""
    v = 0
    while a % p == 0:
        a //= p
        v += 1
    return v, a


def _legendre(u, p):
    """(u/p) for an odd prime p and u prime to p."""
    return 1 if pow(u % p, (p - 1) // 2, p) == 1 else -1


def _hilbert_odd(a, b, p):
    al, u = _val_unit(a, p)
    be, v = _val_unit(b, p)
    s = 0
    if (al * be) % 2 and ((p - 1) // 2) % 2:
        s ^= 1
    if be % 2 and _legendre(u, p) == -1:
        s ^= 1
    if al % 2 and _legendre(v, p) == -1:
        s ^= 1
    return -1 if s else 1


def _hilbert_2(a, b):
    al, u = _val_unit(a, 2)
    be, v = _val_unit(b, 2)

    def eps(x):
        return ((x - 1) // 2) % 2

    def omega(x):
        return ((x * x - 1) // 8) % 2

    s = (eps(u) * eps(v) + al * omega(v) + be * omega(u)) % 2
    return -1 if s else 1


def hilbert(a, b, p):
    """The Hilbert symbol (a, b)_p of nonzero rationals: +1 iff
    a x^2 + b y^2 = z^2 has a nontrivial solution over Q_p.

    ``p`` is a prime, or ``INFINITY`` for the real place.
    """
    a, b = square_class(a), square_class(b)
    if p == INFINITY:
        return -1 if (a < 0 and b < 0) else 1
    if p == 2:
        return _hilbert_2(a, b)
    return _hilbert_odd(a, b, p)


def _is_p_adic_square(m, p):
    """Whether the nonzero integer m is a square in Q_p."""
    v, u = _val_unit(m, p)
    if v % 2:
        return False
    if p == 2:
        return u % 8 == 1
    return _legendre(u, p) == 1


def _primes_of(*ints):
    """{2} together with every prime dividing one of the integers."""
    ps = {2}
    for m in ints:
        ps |= set(factorint(abs(m)))
    return ps


def diagonalize(G):
    """Diagonal entries of a rational diagonalisation of the symmetric matrix
    ``G`` (congruence over Q). Raises :class:`DegenerateForm` if G is singular.
    """
    n = len(G)
    A = [[_frac(x) for x in row] for row in G]
    for i in range(n):
        if len(A[i]) != n:
            raise ValueError("Gram matrix is not square")
        for j in range(i):
            if A[i][j] != A[j][i]:
                raise ValueError("Gram matrix is not symmetric")
    diag = []
    active = list(range(n))
    while active:
        piv = next((i for i in active if A[i][i] != 0), None)
        if piv is None:
            pair = next(((i, j) for i in active for j in active
                         if i < j and A[i][j] != 0), None)
            if pair is None:
                raise DegenerateForm("singular Gram matrix")
            i, j = pair
            # e_i -> e_i + e_j makes the (i, i) entry 2 A[i][j] != 0
            for k in range(n):
                A[i][k] += A[j][k]
            for k in range(n):
                A[k][i] += A[k][j]
            piv = i
        d = A[piv][piv]
        diag.append(d)
        active.remove(piv)
        for k in active:
            c = A[k][piv] / d
            if c != 0:
                for m in range(n):
                    A[k][m] -= c * A[piv][m]
                for m in range(n):
                    A[m][k] -= c * A[m][piv]
    return diag


# --------------------------------------------------------------------------
# rational quadratic spaces, by their invariants
# --------------------------------------------------------------------------

class QSpace(object):
    """A nondegenerate rational quadratic space, recorded by its invariants:
    rank ``n``, signature ``(pos, neg)``, discriminant ``d`` (a squarefree
    integer) and the finite set ``minus`` of primes at which the Hasse
    invariant is -1. The Hasse invariant is +1 at every prime outside
    ``minus``, which is why a finite set is enough.
    """

    __slots__ = ("n", "pos", "neg", "d", "minus")

    def __init__(self, n, pos, neg, d, minus=()):
        if pos < 0 or neg < 0 or pos + neg != n:
            raise ValueError("signature (%d, %d) is not a rank %d form"
                             % (pos, neg, n))
        d = square_class(d) if n else 1
        if (d < 0) != (neg % 2 == 1):
            raise ValueError("sign of the discriminant contradicts the "
                             "signature")
        self.n, self.pos, self.neg, self.d = n, pos, neg, d
        self.minus = frozenset(int(p) for p in minus)

    # -- construction -------------------------------------------------------
    @classmethod
    def from_diag(cls, entries):
        sq = [square_class(a) for a in entries]
        n = len(sq)
        pos = sum(1 for a in sq if a > 0)
        d = 1
        for a in sq:
            d = square_class(d * a)
        minus = set()
        for p in _primes_of(*sq):
            s = 1
            for i in range(n):
                for j in range(i + 1, n):
                    s *= hilbert(sq[i], sq[j], p)
            if s == -1:
                minus.add(p)
        return cls(n, pos, n - pos, d, minus)

    @classmethod
    def from_gram(cls, G):
        if len(G) == 0:
            return cls(0, 0, 0, 1)
        return cls.from_diag(diagonalize(G))

    # -- invariants ---------------------------------------------------------
    def eps(self, p):
        """The Hasse invariant at the place ``p`` (prime or ``INFINITY``)."""
        if p == INFINITY:
            return -1 if (self.neg * (self.neg - 1) // 2) % 2 else 1
        return -1 if p in self.minus else 1

    def product_formula(self):
        """prod_v eps_v == 1 over all places. True of every form that exists."""
        k = len(self.minus) + (1 if self.eps(INFINITY) == -1 else 0)
        return k % 2 == 0

    def is_isometric(self, other):
        """Isometry over Q (Hasse-Minkowski)."""
        return (self.n, self.pos, self.neg, self.d, self.minus) == \
               (other.n, other.pos, other.neg, other.d, other.minus)

    def orth_sum(self, other):
        """The orthogonal sum: eps = eps eps' (d, d')_p."""
        d = square_class(self.d * other.d)
        primes = set(self.minus) | set(other.minus) | \
            _primes_of(self.d, other.d)
        minus = set()
        for p in primes:
            if (self.eps(p) * other.eps(p)
                    * hilbert(self.d, other.d, p)) == -1:
                minus.add(p)
        return QSpace(self.n + other.n, self.pos + other.pos,
                      self.neg + other.neg, d, minus)

    def __repr__(self):
        return ("QSpace(rank=%d, sig=(%d,%d), d=%d, eps=-1 at %s)"
                % (self.n, self.pos, self.neg, self.d,
                   sorted(self.minus) or "no prime"))


# --------------------------------------------------------------------------
# embedding
# --------------------------------------------------------------------------

class Embedding(object):
    """The answer to "does V embed in W?": ``embeds``, the named
    ``obstruction`` when it does not, and the ``complement`` (the invariants
    of V' with V (+) V' = W) when it does."""

    __slots__ = ("embeds", "obstruction", "complement")

    def __init__(self, embeds, obstruction=None, complement=None):
        self.embeds = embeds
        self.obstruction = obstruction
        self.complement = complement

    def __bool__(self):
        return bool(self.embeds)

    def __repr__(self):
        if self.embeds:
            return "Embedding(True, complement=%r)" % (self.complement,)
        return "Embedding(False, %r)" % (self.obstruction,)


def embeds(V, W, _skip_rank2_condition=False):
    """Whether the rational quadratic space ``V`` embeds isometrically in ``W``.

    By Witt cancellation this holds iff there is a space V' with
    V (+) V' ~ W. Its invariants are forced:

        rank  = n_W - n_V,    sig = sig_W - sig_V,    d' = d_W d_V,
        eps'_p = eps_p(W) eps_p(V) (d_V, d')_p,

    and a space with those invariants exists iff the product formula holds and,
    in rank 1, no Hasse invariant is -1, and, in rank 2, eps'_p = +1 wherever
    d' = -1 in Q_p^*/squares (Serre, IV.3.3, Prop. 7). Nothing else constrains
    a form of rank >= 3.

    ``_skip_rank2_condition`` exists only so the tests can show that the
    sums-of-three-squares check fails without it.
    """
    if V.n > W.n:
        return Embedding(False, "rank: %d > %d" % (V.n, W.n))
    if V.pos > W.pos or V.neg > W.neg:
        return Embedding(False, "signature: (%d,%d) does not fit in (%d,%d)"
                         % (V.pos, V.neg, W.pos, W.neg))
    k = W.n - V.n
    pos, neg = W.pos - V.pos, W.neg - V.neg
    dp = square_class(W.d * V.d) if k else 1

    if k == 0:
        if V.is_isometric(W):
            return Embedding(True, None, QSpace(0, 0, 0, 1))
        return Embedding(False, "equal rank: the spaces are not isometric "
                         "(different discriminant or Hasse invariants)")

    primes = set(W.minus) | set(V.minus) | _primes_of(V.d, dp)
    minus = set()
    for p in sorted(primes):
        if W.eps(p) * V.eps(p) * hilbert(V.d, dp, p) == -1:
            minus.add(p)
    comp = QSpace(k, pos, neg, dp, minus)

    if k == 1 and comp.minus:
        p = min(comp.minus)
        return Embedding(False, "local at p=%d: a rank-1 complement has no "
                         "Hasse invariant, but this one is -1" % p)
    if k == 2 and not _skip_rank2_condition:
        for p in sorted(comp.minus):
            if _is_p_adic_square(-dp, p):
                return Embedding(
                    False, "local at p=%d: the rank-2 complement has "
                    "discriminant -1 in Q_%d^*/squares, which forces "
                    "Hasse invariant +1, but this one is -1" % (p, p))
    if not comp.product_formula():
        return Embedding(False, "product formula: the Hasse invariants of "
                         "the complement multiply to -1")
    return Embedding(True, None, comp)


def complement(W, V):
    """The invariants of the orthogonal complement of ``V`` in ``W``; raises
    ``ValueError`` naming the obstruction if ``V`` does not embed."""
    e = embeds(V, W)
    if not e:
        raise ValueError("V does not embed in W: " + e.obstruction)
    return e.complement


def gram_of(W, vectors):
    """The Gram matrix of ``vectors`` (rows) in the space with Gram ``W``."""
    Wm, Vm = Matrix(W), Matrix(vectors)
    G = Vm * Wm * Vm.T
    return [[_frac(G[i, j]) for j in range(G.cols)] for i in range(G.rows)]


def explicit_complement(W, vectors):
    """The Gram matrix of the orthogonal complement of the span of ``vectors``
    in the space with Gram ``W``, by linear algebra over Q.

    This is the second route to the complement. It shares nothing with the
    Hasse-invariant formula in :func:`embeds`, only the diagonaliser that turns
    the resulting Gram matrix into invariants.
    """
    Wm, Vm = Matrix(W), Matrix(vectors)
    ns = (Vm * Wm).nullspace()
    if not ns:
        return []
    B = Matrix.hstack(*ns)
    G = B.T * Wm * B
    return [[_frac(G[i, j]) for j in range(G.cols)] for i in range(G.rows)]


# --------------------------------------------------------------------------
# lattices
# --------------------------------------------------------------------------

def lattice_U():
    """The hyperbolic plane."""
    return [[0, 1], [1, 0]]


def _cartan(n, edges):
    G = [[0] * n for _ in range(n)]
    for i in range(n):
        G[i][i] = 2
    for i, j in edges:
        G[i][j] = G[j][i] = -1
    return G


def cartan_A(n):
    """The A_n root lattice (positive definite)."""
    return _cartan(n, [(i, i + 1) for i in range(n - 1)])


def cartan_D(n):
    """The D_n root lattice, n >= 4 (positive definite)."""
    if n < 4:
        raise ValueError("D_n needs n >= 4")
    return _cartan(n, [(i, i + 1) for i in range(n - 2)] + [(n - 3, n - 1)])


def cartan_E8():
    """The E8 lattice, Bourbaki labelling (positive definite, unimodular)."""
    edges = [(0, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (1, 3)]
    return _cartan(8, edges)


def negate(G):
    return [[-x for x in row] for row in G]


def direct_sum(*Gs):
    n = sum(len(G) for G in Gs)
    out = [[0] * n for _ in range(n)]
    o = 0
    for G in Gs:
        for i in range(len(G)):
            for j in range(len(G)):
                out[o + i][o + j] = G[i][j]
        o += len(G)
    return out


def identity_form(n, sign=1):
    return [[sign if i == j else 0 for j in range(n)] for i in range(n)]


def k3_h2():
    """Gram matrix of H^2(K3, Z) = U^3 (+) E8(-1)^2, rank 22, signature (3,19)."""
    return direct_sum(lattice_U(), lattice_U(), lattice_U(),
                      negate(cartan_E8()), negate(cartan_E8()))


def k3_locus_form():
    """Gram matrix of the target space U^2 (+) <-1>^4 of the quadratic locus."""
    return direct_sum(lattice_U(), lattice_U(), identity_form(4, -1))


@lru_cache(maxsize=None)
def _h2_space():
    return QSpace.from_gram(k3_h2())


@lru_cache(maxsize=None)
def _locus_space():
    return QSpace.from_gram(k3_locus_form())


# --------------------------------------------------------------------------
# the K3 question
# --------------------------------------------------------------------------

def _as_space(N):
    if isinstance(N, QSpace):
        return N
    rows = [[_frac(x) for x in row] for row in N]
    for i, row in enumerate(rows):
        if row[i].denominator == 1 and int(row[i]) % 2:
            raise NotAK3Lattice("the form is not even: entry (%d,%d) = %s"
                                % (i, i, row[i]))
        if row[i].denominator != 1:
            raise NotAK3Lattice("a Picard lattice has integral entries")
    return QSpace.from_gram(rows)


def transcendental_space(N):
    """The rational transcendental space T of a K3 surface whose Picard lattice
    has Gram matrix (or invariants) ``N``: the orthogonal complement of N in
    H^2(K3, Q).

    Raises :class:`NotAK3Lattice` if N does not have signature (1, rho-1) or
    cannot embed in H^2 rationally. That is a necessary condition for N to be
    a Picard lattice, not a sufficient one: Nikulin's integral criterion is
    not applied.
    """
    S = _as_space(N)
    if S.pos != 1:
        raise NotAK3Lattice("a Picard lattice has signature (1, rho-1); "
                            "this has (%d, %d)" % (S.pos, S.neg))
    e = embeds(S, _h2_space())
    if not e:
        raise NotAK3Lattice("does not embed in H^2(K3, Q): " + e.obstruction)
    return e.complement


class Verdict(object):
    """``status`` is "in", "out" or "undetermined"; ``reason`` says why;
    ``transcendental`` is the space T the verdict was computed from."""

    __slots__ = ("status", "reason", "transcendental", "picard_rank")

    def __init__(self, status, reason, transcendental, picard_rank):
        self.status, self.reason = status, reason
        self.transcendental, self.picard_rank = transcendental, picard_rank

    def __repr__(self):
        return "Verdict(%r, %r)" % (self.status, self.reason)


def locus_verdict(N, exact=True):
    """Whether a K3 with Picard lattice ``N`` lies on the quadratic locus.

    ``exact=True``: N is the whole Picard lattice (rationally), so both answers
    are decided. ``exact=False``: N is only known to lie inside the Picard
    lattice. Then T is a subspace of the T computed here, and a subspace of a
    space that embeds embeds, so "in" is still certain; but "does not embed"
    says nothing, and the verdict is "undetermined".
    """
    S = _as_space(N)
    T = transcendental_space(S)
    e = embeds(T, _locus_space())
    if e:
        return Verdict("in", "T embeds in U^2 + <-1>^4; complement %r"
                       % (e.complement,), T, S.n)
    if exact:
        return Verdict("out", e.obstruction, T, S.n)
    return Verdict("undetermined",
                   "the known classes give " + e.obstruction + "; the true "
                   "Picard lattice may be larger", T, S.n)


# --------------------------------------------------------------------------
# CICY K3
# --------------------------------------------------------------------------

def _config(cicy):
    M = getattr(cicy, "M", cicy)
    rows = [[int(_frac(x)) for x in row] for row in M]
    if not rows or any(len(r) != len(rows[0]) for r in rows):
        raise ValueError("ragged configuration matrix")
    return rows


def double_intersection(cicy):
    """The matrix  d_rs = int_X J_r J_s  of a CICY surface, computed in the
    ambient cohomology ring Q[J_1..J_m]/(J_i^(n_i+1)) with
    int_A prod J_i^(n_i) = 1.

    Accepts a configuration matrix (rows ``[n_i, q_i^1, ..., q_i^K]``) or a
    :class:`pyCICY.CICY`. Entirely independent of the 3-fold routines in
    ``pyCICY.pyCICY``, which do not cover surfaces.
    """
    rows = _config(cicy)
    m, K = len(rows), len(rows[0]) - 1
    ns = [r[0] for r in rows]
    if sum(ns) - K != 2:
        raise ValueError("not a surface: dim = %d" % (sum(ns) - K))
    poly = {tuple([0] * m): 1}
    for a in range(K):
        new = {}
        for mon, c in poly.items():
            for i in range(m):
                q = rows[i][1 + a]
                if q == 0:
                    continue
                e = list(mon)
                e[i] += 1
                if e[i] > ns[i]:
                    continue
                t = tuple(e)
                new[t] = new.get(t, 0) + c * q
        poly = new
    G = [[0] * m for _ in range(m)]
    for r in range(m):
        for s in range(m):
            e = list(ns)
            e[r] -= 1
            e[s] -= 1
            if min(e) < 0:
                continue
            G[r][s] = poly.get(tuple(e), 0)
    return G


def ambient_picard_gram(cicy):
    """The Gram matrix of the restricted ambient divisor classes of a CICY K3.

    Three exact consistency checks are made, none of which the construction
    imposes: the matrix is symmetric, its diagonal is even (H^2 of a K3 is an
    even lattice), and its signature is (1, r - 1) (the Hodge index theorem).
    A failure means the configuration is not a K3 or the arithmetic is wrong.
    """
    G = double_intersection(cicy)
    if any(G[i][i] % 2 for i in range(len(G))):
        raise ValueError("odd self-intersection on a K3: not an even lattice")
    try:
        S = QSpace.from_gram(G)
    except DegenerateForm:
        raise AmbientDegenerate(
            "the restricted ambient classes span a degenerate subspace; "
            "their rank in NS(X) cannot be read off the Gram matrix")
    if S.pos != 1:
        raise ValueError("Hodge index theorem violated: signature (%d, %d)"
                         % (S.pos, S.neg))
    return G


def cicy_k3_verdict(cicy, exact=False):
    """Quadratic-locus verdict for the generic member of a CICY K3 family,
    from the Picard lattice its ambient divisors span.

    That lattice is only a *lower bound* on the generic Picard lattice, so by
    default this can certify membership ("in") but never exclusion. Ambient
    classes number at most the count of projective factors, which is far below
    the rank 14 the locus requires; so for an ordinary CICY K3 the honest
    answer is "undetermined", and a larger Picard lattice has to come from
    outside the configuration matrix. Pass ``exact=True`` only when the
    ambient classes are known to span the whole Picard lattice.
    """
    return locus_verdict(ambient_picard_gram(cicy), exact=exact)
