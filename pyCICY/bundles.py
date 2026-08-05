r"""
pyCICY.bundles -- holomorphic vector bundles beyond the standard embedding.

Why this module exists
----------------------
:mod:`pyCICY.phenomenology` computes the heterotic spectrum for the *standard
embedding* V = TX, where the low energy gauge group is E_6 and the charged
matter is counted by Hodge numbers alone. Its docstring lists what the
configuration matrix does not determine, and the first item on that list is
the choice of gauge bundle V when it is not the standard embedding.

This module supplies that choice. The construction is the one of

    L. B. Anderson, J. Gray, A. Lukas, E. Palti,
    "Heterotic Line Bundle Standard Models",
    Phys. Rev. D 84 (2011) 106005, arXiv:1106.4804,
    JHEP 06 (2012) 113, arXiv:1202.1757,

in which V is a sum of line bundles

    V = \bigoplus_{a=1}^{n} O_X(L_a) ,      \sum_a L_a = 0 ,

with structure group S(U(1)^n) inside SU(n) inside E_8. Because every summand
is a line bundle, every cohomology group needed for the spectrum is a sum of
line bundle cohomologies, which :meth:`pyCICY.CICY.line_co` already computes.
Nothing here is approximated: the spectrum is exact wherever ``line_co`` is.

The other standard construction, the monad

    0 -> V -> \bigoplus_i O(b_i) -> \bigoplus_j O(c_j) -> 0 ,

is included because its Chern character is the same signed sum of
exponentials, so the topology costs no extra code. Its *cohomology* does cost
extra, and is not claimed; see "Scope" below.

Topology
--------
Both constructions present V as a virtual sum of line bundles, a formal
combination \sum_a eps_a O(L_a) with eps_a = +1 or -1. For such a thing

    ch(V) = \sum_a eps_a exp(L_a . J) ,

so on a favourable threefold with triple intersection numbers d_rst,

    rk V             = \sum_a eps_a
    c_1(V)^r         = \sum_a eps_a L_a^r
    \int ch_2(V) J_r = (1/2) \sum_a eps_a d_rst L_a^s L_a^t
    \int ch_3(V)     = (1/6) \sum_a eps_a d_rst L_a^r L_a^s L_a^t

and when c_1(V) = 0, which is the SU(n) condition, c_2(V) = -ch_2(V) and
the Atiyah-Singer index reduces to

    ind(V) = \sum_q (-1)^q h^q(X, V) = \int ch_3(V) ,

the term (1/12) c_1(V) c_2(TX) dropping out. :func:`index` computes that
polynomial expression; :meth:`Bundle.index_from_cohomology` computes the same
number by summing ``line_co_euler`` over the summands. They share no code and
the tests require them to agree, in the manner of ``node_validation``
elsewhere in this package.

Three conditions
----------------
A line bundle sum is a candidate heterotic model when three things hold.

*c_1(V) = 0*, so the structure group sits in SU(n). Checked exactly.

*Anomaly cancellation.* Without five-branes the heterotic Bianchi identity
requires c_2(TX) - c_2(V) to be an effective class. What is checked here is
the necessary condition that its integral against each Kahler cone generator
be non-negative,

    \int (c_2(TX) - c_2(V)) J_r >= 0    for all r,

which on a favourable CICY, where the J_r generate the Kahler cone, follows
from effectiveness but does not imply it. :func:`anomaly` returns that
verdict and says so in its ``sufficient`` field. Deciding effectiveness
properly means knowing the Mori cone, and this package does not.

*Poly-stability.* A sum of line bundles is never stable. It is poly-stable
exactly on the locus in the Kahler cone where all n slopes vanish
simultaneously,

    mu(L_a) = d_rst L_a^r t^s t^t = 0    for every a,

and only there does it solve the hermitian Yang-Mills equation and give a
supersymmetric vacuum. Since \sum_a L_a = 0 the slopes sum to zero, so there
are n-1 independent conditions on the h^{1,1}-1 directions of the projective
Kahler cone. :func:`stability_locus` looks for a solution numerically inside
the positive orthant. A solution found is a genuine point of the cone; a
solution not found within the search is reported as not found, not as
non-existent.

The spectrum
------------
For structure group SU(5) the commutant in E_8 is SU(5), and the
decomposition of the adjoint gives, in the conventions of arXiv:1202.1757,

    n(10)     = h^1(X, V)
    n(10-bar) = h^1(X, V*)
    n(5-bar)  = h^1(X, \Lambda^2 V)
    n(5)      = h^1(X, \Lambda^2 V*)
    n(1)      = h^1(X, V \otimes V*)

Every one of these is computable for a line bundle sum, because

    V*            = \bigoplus_a O(-L_a)
    \Lambda^2 V   = \bigoplus_{a<b} O(L_a + L_b)
    V \otimes V*  = \bigoplus_{a,b} O(L_a - L_b) .

Serre duality on a Calabi-Yau threefold gives h^2(V) = h^1(V*), so

    ind(V) = -h^1(V) + h^1(V*) = -( n(10) - n(10-bar) ) = -n_gen ,

and three generations upstairs of a quotient by a freely acting Gamma is
ind(V) = -3|Gamma|. That is the same counting :mod:`pyCICY.symmetries` does
for the standard embedding, now for a bundle that can actually break E_8
to something with the Standard Model in it.

:func:`scan` searches line bundle sums by index alone, which needs only the
triple intersection numbers and is therefore fast, and hands the survivors on
to the cohomology, which is not.

Scope
-----
* Monad bundles get their topology and index here but **not** their
  cohomology. h^q of a monad follows from the long exact sequence only up to
  the connecting map, whose rank is a property of the maps and not of the
  degrees; :meth:`Monad.cohomology_bounds` returns the bounds the sequence
  gives and refuses to guess the rest.
* Everything assumes X is a **favourable** threefold, so that H^2(X) is
  spanned by restrictions of the ambient Kahler forms and a line bundle is a
  list of integers. On a non-favourable configuration the constructor raises.
* Stability is checked for the *sum*. Whether the model is consistent after
  quotienting by Gamma, which needs an equivariant structure on V, belongs to
  a later module and is not decided here.
* No Yukawa couplings. Those are cup products and need holomorphic
  representatives, not just dimensions.
"""

import itertools
import logging
import time as _time

import numpy as np

from .pyCICY import CICY

logger = logging.getLogger('pyCICY.bundles')

__all__ = [
    "Bundle", "LineBundleSum", "Monad",
    "NotFavourable", "NotABundle",
    "chern_character", "index", "anomaly", "slope", "slope_is_definite", "slope_candidates", "slope_subsets_definite", "stability_locus",
    "scan",
]


class NotABundle(ValueError):
    """Raised when a monad's degrees make the defining sequence impossible."""


class NotFavourable(ValueError):
    """Raised when a construction needs a favourable configuration."""


def _as_cicy(X):
    """Accept a CICY object or a configuration matrix."""
    if isinstance(X, CICY):
        return X
    return CICY(X, log=3)


def _triple(X):
    t = np.array(X.triple_intersection(), dtype=float)
    return t


def _c2_tangent(X):
    """Return the vector \\int c_2(TX) J_r."""
    return np.array(X.second_chern(), dtype=float)


# ---------------------------------------------------------------------------
# topology of a virtual sum of line bundles
# ---------------------------------------------------------------------------

def chern_character(triple, summands, signs=None):
    r"""
    Chern character data of a virtual sum of line bundles.

    Parameters
    ----------
    triple : array (h11, h11, h11)
        The triple intersection numbers d_rst.
    summands : list of integer vectors
        The line bundles L_a, each of length h^{1,1}.
    signs : list of +-1, optional
        eps_a, defaulting to all +1.

    Returns
    -------
    dict with keys ``rank``, ``c1`` (vector), ``ch2`` (vector, the integrals
    \\int ch_2 J_r) and ``ch3`` (scalar, \\int ch_3).

    Notes
    -----
    These are the coefficients of ch(V) = sum_a eps_a exp(L_a . J) integrated
    against the obvious classes. No assumption that c_1 vanishes is made
    here; :func:`index` and :func:`anomaly` do make it and say so.
    """
    d = np.asarray(triple, dtype=float)
    L = np.asarray(summands, dtype=float)
    if L.ndim == 1:
        L = L.reshape(1, -1)
    eps = np.ones(len(L)) if signs is None else np.asarray(signs, dtype=float)

    rank = float(eps.sum())
    c1 = eps @ L
    # (1/2) sum_a eps_a d_rst L^s L^t
    ch2 = 0.5 * np.einsum('a,rst,as,at->r', eps, d, L, L)
    ch3 = (1.0 / 6.0) * np.einsum('a,rst,ar,as,at->', eps, d, L, L, L)
    return {"rank": rank, "c1": c1, "ch2": ch2, "ch3": ch3}


def index(X, summands, signs=None):
    r"""
    The Atiyah-Singer index of a virtual sum of line bundles.

    For c_1(V) = 0 this is \\int ch_3(V). Otherwise the full

        ind(V) = \\int ch_3(V) + (1/12) \\int c_1(V) c_2(TX)

    is returned, so the function is correct for bundles that are not SU(n).
    """
    X = _as_cicy(X)
    data = chern_character(_triple(X), summands, signs)
    c2 = _c2_tangent(X)
    return data["ch3"] + float(np.dot(data["c1"], c2)) / 12.0


def anomaly(X, summands, signs=None):
    r"""
    The heterotic Bianchi identity, as far as it can be checked here.

    Returns a dict with the vector ``surplus``\\ :math:`_r =
    \\int (c_2(TX) - c_2(V)) J_r`, a boolean ``ok`` that is True when every
    component is non-negative, and ``sufficient=False``.

    The last field is not decoration. Non-negativity of these integrals is
    implied by effectiveness of c_2(TX) - c_2(V) but does not imply it, so a
    True verdict is a necessary condition passed and nothing more. Deciding
    effectiveness needs the Mori cone.
    """
    X = _as_cicy(X)
    data = chern_character(_triple(X), summands, signs)
    if np.any(np.abs(data["c1"]) > 1e-9):
        logger.warning('anomaly(): c_1(V) != 0, so c_2(V) = (1/2)c_1^2 - ch_2 '
                       'and the surplus below is not the SU(n) one.')
        c2V = 0.5 * np.einsum('rst,s,t->r', _triple(X), data["c1"],
                              data["c1"]) - data["ch2"]
    else:
        c2V = -data["ch2"]
    surplus = _c2_tangent(X) - c2V
    return {"surplus": surplus,
            "ok": bool(np.all(surplus >= -1e-9)),
            "sufficient": False}


def slope(X, line, t):
    r"""
    The slope mu(L) = d_rst L^r t^s t^t at a Kahler parameter t.

    Divided by the volume this is the honest slope; the normalisation is
    irrelevant to the vanishing locus, which is all that is used.
    """
    X = _as_cicy(X)
    d = _triple(X)
    return float(np.einsum('rst,r,s,t->', d, np.asarray(line, dtype=float),
                           np.asarray(t, dtype=float),
                           np.asarray(t, dtype=float)))


def slope_is_definite(triple, line, tol=1e-12):
    r"""
    Whether mu(L) is of one sign throughout the interior of the Kahler cone.

    On a favourable CICY the Kahler cone is the positive orthant, so with
    M^{st} = d_rst L^r symmetrised, mu(L) = t^T M t is a quadratic form
    evaluated on strictly positive vectors. If every entry of M is >= 0 and
    one is positive then mu > 0 on the interior; likewise with the signs
    reversed. Only when M has entries of both signs can mu vanish somewhere
    inside.

    The one case that needs care is M = 0, which means mu vanishes
    *identically*. A trivial summand O_X is the obvious instance. That is
    compatible with poly-stability rather than an obstruction to it, so it
    returns False, not True.

    This is a necessary condition on each summand and it is pure sign
    arithmetic, which makes it the right thing to apply to a whole candidate
    pool before any optimisation runs; see :func:`scan`.
    """
    M = np.einsum('rst,r->st', np.asarray(triple, dtype=float),
                  np.asarray(line, dtype=float))
    M = M + M.T
    if not np.any(np.abs(M) > tol):
        return False                      # mu identically zero: allowed
    return not (np.any(M < -tol) and np.any(M > tol))


def slope_candidates(triple, cand, tol=1e-12):
    """Boolean mask over an array of charge vectors: which ones *could*
    appear in a poly-stable sum. Vectorised :func:`slope_is_definite`."""
    d = np.asarray(triple, dtype=float)
    c = np.asarray(cand, dtype=float)
    M = np.einsum('rst,ar->ast', d, c)
    M = M + np.transpose(M, (0, 2, 1))
    pos = (M > tol).any(axis=(1, 2))
    neg = (M < -tol).any(axis=(1, 2))
    zero = ~(pos | neg)
    return zero | (pos & neg)


def slope_subsets_definite(triple, summands, tol=1e-12):
    r"""
    An exact, cheap obstruction to poly-stability.

    At a common zero t* of all the slopes, *every* partial sum of slopes
    vanishes too: for any subset S,

        sum_{a in S} mu(L_a)(t*) = t*^T ( sum_{a in S} M_a ) t* = 0 ,

    with M_a the symmetrised d_rst L_a^r. So if for some subset that summed
    matrix is non-zero and has no negative entry, the form is strictly
    positive on the open positive orthant and no common zero can exist.
    Likewise with the signs reversed.

    Returns True when such a subset is found, i.e. when the bundle is
    definitely *not* poly-stable anywhere in the Kahler cone. Returning False
    means no obstruction of this kind, not that a solution exists.

    Subsets of size one are the per-summand test that :func:`slope_is_definite`
    performs and that :func:`slope_candidates` applies to a whole pool. The
    larger subsets are what make this worth doing: on the tetraquadric at
    charge 2 the size-one test alone leaves most of the search box standing,
    while the full subset test removes all of it, in 35 microseconds per
    bundle against roughly 30 milliseconds for the numerical search. The
    whole subset lattice is 2^n - 2 sign tests, which for the ranks of
    interest is at most thirty.
    """
    d = np.asarray(triple, dtype=float)
    L = np.asarray(summands, dtype=float)
    M = np.einsum('rst,ar->ast', d, L)
    M = M + np.transpose(M, (0, 2, 1))
    n = len(L)
    for k in range(1, n):
        for c in itertools.combinations(range(n), k):
            A = M[list(c)].sum(axis=0)
            if not np.any(np.abs(A) > tol):
                continue                  # identically zero: no obstruction
            if not (np.any(A < -tol) and np.any(A > tol)):
                return True
    return False


def stability_locus(X, summands, tries=24, seed=0, tol=1e-7):
    r"""
    Look for a point of the Kahler cone where every slope vanishes.

    A sum of line bundles is poly-stable exactly on the common zero locus of
    the n slopes, and nowhere else, so this is the condition for the model to
    solve the hermitian Yang-Mills equation.

    The search minimises sum_a mu(L_a)^2 over the positive orthant t^r > 0,
    restricted to the unit sphere to remove the overall scaling (the slopes
    are homogeneous of degree two, so only the direction matters). Since
    sum_a L_a = 0 the slopes sum to zero identically and one condition is
    redundant.

    Returns a dict with ``found``, the Kahler parameter ``t``, and the
    ``residual``. A ``found`` of False means no solution was located by this
    search from these starting points, which is weaker than a proof that none
    exists, and the docstring rather than the return value is where that
    caveat belongs.
    """
    from scipy.optimize import minimize

    X = _as_cicy(X)
    d = _triple(X)
    L = np.asarray(summands, dtype=float)
    n = d.shape[0]
    rng = np.random.default_rng(seed)

    # Cheap necessary condition first, and it removes almost everything.
    # mu(L_a) = t^T M_a t with M_a = d_rst L_a^r symmetrised. If M_a has no
    # negative entry, mu(L_a) >= 0 on the positive orthant with equality only
    # on its boundary, so there is no interior zero and no common zero
    # either. This is the sign test of CICY.l_slope, applied per summand
    # before any optimisation runs.
    if slope_subsets_definite(d, L):
        return {"found": False, "t": np.zeros(n), "residual": np.inf,
                "reason": "some subset of the summands has a definite "
                          "slope sum on the cone"}

    def residual(t):
        mu = np.einsum('rst,ar,s,t->a', d, L, t, t)
        return float(np.dot(mu, mu))

    best = None
    for _ in range(tries):
        t0 = rng.random(n) + 0.2
        t0 /= np.linalg.norm(t0)
        res = minimize(lambda u: residual(np.abs(u)), t0, method='Nelder-Mead',
                       options={'xatol': 1e-10, 'fatol': 1e-14,
                                'maxiter': 20000, 'maxfev': 20000})
        t = np.abs(res.x)
        nrm = np.linalg.norm(t)
        if nrm < 1e-12:
            continue
        t = t / nrm
        r = residual(t)
        if best is None or r < best[1]:
            best = (t, r)
        if r < tol and np.all(t > 1e-6):
            break

    t, r = best if best is not None else (np.ones(n) / np.sqrt(n), np.inf)
    interior = bool(np.all(t > 1e-6))
    return {"found": bool(r < tol and interior), "t": t, "residual": r}


# ---------------------------------------------------------------------------
# bundles
# ---------------------------------------------------------------------------

class Bundle(object):
    r"""
    A virtual sum of line bundles on a favourable CICY threefold.

    This is the common base of :class:`LineBundleSum` (all signs +1) and
    :class:`Monad` (the two terms of the defining sequence with opposite
    signs). Everything on this class is topology: rank, Chern data, index,
    anomaly, slopes. Cohomology lives on the subclasses, because only one of
    them has any.
    """

    def __init__(self, X, summands, signs=None, name=None):
        self.X = _as_cicy(X)
        if not self.X.fav:
            raise NotFavourable(
                'bundles requires a favourable configuration, so that a line '
                'bundle is a list of integers in a basis of H^2(X). This one '
                'is not favourable.')
        if self.X.nfold != 3:
            raise ValueError('bundles is written for threefolds; this '
                             'configuration is a %d-fold.' % self.X.nfold)
        self.summands = [list(map(int, L)) for L in summands]
        for L in self.summands:
            if len(L) != self.X.len:
                raise ValueError('line bundle %r has the wrong length for '
                                 'h^{1,1} = %d' % (L, self.X.len))
        self.signs = ([1] * len(self.summands) if signs is None
                      else [int(s) for s in signs])
        self.name = name
        self._ch = chern_character(_triple(self.X), self.summands, self.signs)

    # -- topology ----------------------------------------------------------

    @property
    def rank(self):
        return int(round(self._ch["rank"]))

    @property
    def c1(self):
        """c_1(V) as an integer vector in the basis of H^2(X)."""
        return np.rint(self._ch["c1"]).astype(int)

    @property
    def ch2(self):
        r"""The vector \\int ch_2(V) J_r."""
        return self._ch["ch2"]

    @property
    def c2(self):
        r"""The vector \\int c_2(V) J_r. Assumes c_1(V) = 0."""
        if np.any(self.c1 != 0):
            raise ValueError('c_2 accessor assumes c_1(V) = 0; use '
                             'chern_character for the general case.')
        return -self._ch["ch2"]

    @property
    def is_su(self):
        """True when c_1(V) = 0, so the structure group lies in SU(rank)."""
        return bool(np.all(self.c1 == 0))

    def index(self):
        """The index, from intersection theory."""
        return index(self.X, self.summands, self.signs)

    def index_from_cohomology(self):
        """
        The same index, from ``line_co_euler`` on each summand.

        Independent of :meth:`index`: that one contracts triple intersection
        numbers, this one runs the Leray spectral sequence. The tests require
        them to agree.
        """
        total = 0.0
        for eps, L in zip(self.signs, self.summands):
            total += eps * float(self.X.line_co_euler(L))
        return total

    def anomaly(self):
        """See :func:`anomaly`."""
        return anomaly(self.X, self.summands, self.signs)

    def slopes(self, t):
        """The slope of each summand at a Kahler parameter t."""
        return np.array([slope(self.X, L, t) for L in self.summands])

    def stability_locus(self, **kw):
        """See :func:`stability_locus`."""
        return stability_locus(self.X, self.summands, **kw)

    def describe(self):
        head = self.name or self.__class__.__name__
        return ('%s  rank=%d  c1=%s  ind=%s  anomaly_ok=%s'
                % (head, self.rank, list(self.c1),
                   round(self.index(), 6), self.anomaly()["ok"]))

    def __repr__(self):
        return '<%s rank %d on %s>' % (self.__class__.__name__, self.rank,
                                       self.X.M.tolist())


class LineBundleSum(Bundle):
    r"""
    V = \\bigoplus_a O_X(L_a), the heterotic line bundle construction.

    Every cohomology group the spectrum needs is a direct sum of line bundle
    cohomologies, so all of it is exactly computable. That is the whole
    reason this construction is used: it turns model building into a search
    over integer vectors.

    Example
    -------
    >>> from pyCICY import bundles
    >>> V = bundles.LineBundleSum([[1,2],[1,2],[1,2],[1,2]],
    ...                           [[1,1,-1,-1],[-1,1,1,-1],[1,-1,1,-1],
    ...                            [-1,-1,-1,1],[0,0,0,2]])
    >>> V.is_su
    True
    """

    def __init__(self, X, summands, name=None):
        Bundle.__init__(self, X, summands, None, name)

    # -- the bundles built out of V ---------------------------------------

    def dual(self):
        """V*, the sum of the O(-L_a)."""
        return LineBundleSum(self.X, [[-x for x in L] for L in self.summands],
                             name=(self.name + '*') if self.name else None)

    def wedge2(self):
        r"""\\Lambda^2 V = \\bigoplus_{a<b} O(L_a + L_b)."""
        out = []
        for a, b in itertools.combinations(range(len(self.summands)), 2):
            out.append([u + v for u, v in zip(self.summands[a],
                                              self.summands[b])])
        return LineBundleSum(self.X, out)

    def endomorphisms(self, traceless=True):
        r"""
        V \\otimes V* = \\bigoplus_{a,b} O(L_a - L_b).

        With ``traceless`` the n diagonal summands O(0) are dropped, which is
        what counts the bundle moduli and the U(1) charged singlets rather
        than the n trivial factors. The n-1 surviving trivial directions of
        S(U(1)^n) are a matter of the structure group, not of cohomology, and
        are not subtracted here.
        """
        out = []
        n = len(self.summands)
        for a in range(n):
            for b in range(n):
                if traceless and a == b:
                    continue
                out.append([u - v for u, v in zip(self.summands[a],
                                                  self.summands[b])])
        return LineBundleSum(self.X, out)

    # -- cohomology --------------------------------------------------------

    def cohomology(self, SpaSM=False):
        """
        The vector (h^0, h^1, h^2, h^3) of V, summed over the line bundles.

        Exact: the cohomology of a direct sum is the direct sum of the
        cohomologies. The cost is one ``line_co`` per summand and dominates
        everything else in this module.
        """
        total = np.zeros(4, dtype=int)
        for L in self.summands:
            total += np.array(self.X.line_co(L, SpaSM=SpaSM), dtype=int)
        return total

    def summand_cohomology(self, SpaSM=False):
        """Per-summand cohomology, as a list of (L, (h^0..h^3)) pairs."""
        return [(L, tuple(int(x) for x in self.X.line_co(L, SpaSM=SpaSM)))
                for L in self.summands]

    def su5_spectrum(self, SpaSM=False):
        r"""
        The SU(5) GUT spectrum, in the conventions of arXiv:1202.1757.

            n(10)     = h^1(V)          n(10-bar) = h^1(V*)
            n(5-bar)  = h^1(\\Lambda^2 V)  n(5)   = h^1(\\Lambda^2 V*)
            n(1)      = h^1(V \\otimes V*)

        Also returned are ``generations`` = n(10) - n(10-bar), which must
        equal -ind(V) and is cross-checked against it, and ``h0``, ``h3`` of
        V. A non-zero h^0(V) or h^3(V) is fatal to the model: it obstructs
        stability, since a stable bundle of vanishing slope has no sections.

        This is only meaningful for rank 5. For other ranks the commutant is
        a different group and the decomposition above is the wrong one, so
        the method raises rather than returning numbers under a false label.
        """
        if self.rank != 5:
            raise ValueError('su5_spectrum needs rank 5; this bundle has '
                             'rank %d, whose commutant in E_8 is not SU(5).'
                             % self.rank)
        if not self.is_su:
            raise ValueError('su5_spectrum needs c_1(V) = 0.')

        hV = self.cohomology(SpaSM=SpaSM)
        hVd = self.dual().cohomology(SpaSM=SpaSM)
        hW = self.wedge2().cohomology(SpaSM=SpaSM)
        hWd = self.wedge2().dual().cohomology(SpaSM=SpaSM)
        hE = self.endomorphisms().cohomology(SpaSM=SpaSM)

        gen = int(hV[1] - hVd[1])
        ind = self.index()
        consistent = abs(gen + ind) < 1e-6

        if not consistent:
            logger.warning('su5_spectrum: n_gen = %d but -ind(V) = %s; the '
                           'cohomology and the index disagree.', gen, -ind)

        return {"n10": int(hV[1]), "n10bar": int(hVd[1]),
                "n5bar": int(hW[1]), "n5": int(hWd[1]),
                "n1": int(hE[1]),
                "generations": gen,
                "index": ind,
                "index_consistent": consistent,
                "h0": int(hV[0]), "h3": int(hV[3])}


class Monad(Bundle):
    r"""
    V defined by 0 -> V -> B -> C -> 0 with B, C sums of line bundles.

    The Chern character is ch(B) - ch(C), which is why this shares a base
    class with :class:`LineBundleSum`, and the index is therefore free. The
    cohomology is not: see :meth:`cohomology_bounds`.
    """

    def __init__(self, X, B, C, name=None):
        summands = list(B) + list(C)
        signs = [1] * len(B) + [-1] * len(C)
        Bundle.__init__(self, X, summands, signs, name)
        self.B = [list(map(int, L)) for L in B]
        self.C = [list(map(int, L)) for L in C]

    def cohomology_bounds(self, SpaSM=False, stable=False):
        r"""
        What the long exact sequence gives, and no more.

        From 0 -> V -> B -> C -> 0,

            0 -> H^0(V) -> H^0(B) -> H^0(C) -> H^1(V) -> H^1(B) -> ...

        so with f_q the connecting map H^q(B) -> H^q(C) of rank r_q,

            h^q(V) = h^q(B) - r_q + (h^{q-1}(C) - r_{q-1}) .

        The ranks r_q depend on the actual maps and not on the degrees, so
        they are not determined by the data this class holds. Returned are
        the interval for each h^q obtained by letting each r_q range over its
        possible values, together with the exact alternating sum. When an
        interval has width zero the sequence has determined that h^q outright.

        A monad is only a bundle when the map B -> C is surjective as a sheaf
        map. That is a genericity condition on the coefficients and is not
        checked here -- but one *necessary* consequence of it is, because it
        is free: exactness at the right-hand end forces H^3(B) -> H^3(C) to be
        onto, so h^3(B) < h^3(C) makes the sequence impossible and
        :exc:`NotABundle` is raised. Without that check the formula below
        returns a negative h^3, which is how the case was found: B = O(1)^3
        and C = O + O(3) on the quintic gives h^3(V) = -1, because the trivial
        summand of C contributes h^3(O_X) = 1 by Serre duality and B has
        nothing to map onto it.

        With ``stable=True`` the bounds are tightened by imposing
        h^0(V) = h^3(V) = 0, which holds for any slope-stable bundle with
        c_1(V) = 0 on a Calabi-Yau threefold: a section would give a map
        O_X -> V from a sheaf of the same slope, contradicting stability, and
        h^3(V) = h^0(V*) vanishes by the same argument on the dual. This is
        an assumption about the bundle, not a consequence of the degrees, so
        it is off by default and flagged in the return value.
        """
        hB = np.zeros(4, dtype=int)
        for L in self.B:
            hB += np.array(self.X.line_co(L, SpaSM=SpaSM), dtype=int)
        hC = np.zeros(4, dtype=int)
        for L in self.C:
            hC += np.array(self.X.line_co(L, SpaSM=SpaSM), dtype=int)

        if hB[3] < hC[3]:
            raise NotABundle(
                "h^3(B) = %d < h^3(C) = %d, so H^3(B) -> H^3(C) cannot be "
                "surjective and 0 -> V -> B -> C -> 0 cannot be exact. This "
                "monad does not define a bundle." % (hB[3], hC[3]))

        # r_q = rank of H^q(B) -> H^q(C), with 0 <= r_q <= min(h^q B, h^q C).
        # The sequence terminates ... -> H^3(V) -> H^3(B) -> H^3(C) -> 0, so
        # f_3 is surjective and r_3 = h^3(C) is forced, not free.
        rmin = [0, 0, 0, int(hC[3])]
        rmax = [int(min(hB[q], hC[q])) for q in range(3)] + [int(hC[3])]

        lo, hi = [], []
        for q in range(4):
            prevC = int(hC[q - 1]) if q > 0 else 0
            prev_rmin = rmin[q - 1] if q > 0 else 0
            prev_rmax = rmax[q - 1] if q > 0 else 0
            # h^q(V) = (h^{q-1}(C) - r_{q-1}) + (h^q(B) - r_q)
            lo.append((prevC - prev_rmax) + (int(hB[q]) - rmax[q]))
            hi.append((prevC - prev_rmin) + (int(hB[q]) - rmin[q]))
        if stable:
            # h^0(V) = 0 forces H^0(B) -> H^0(C) injective, so r_0 = h^0(B);
            # h^3(V) = 0 forces r_2 = h^2(C) + h^3(B) - h^3(C). Substituting
            # both leaves r_1 as the only freedom, and then
            #
            #   h^1 = h^0(C) - h^0(B) + h^1(B) - r_1
            #   h^2 = h^1(C) - r_1 + h^2(B) - h^2(C) - h^3(B) + h^3(C)
            #
            # whose difference h^2 - h^1 is independent of r_1 and equals the
            # index, as it must. So the two intervals have the same width,
            # min(h^1(B), h^1(C)), and are rigidly offset by ind(V).
            if hB[0] > hC[0]:
                raise NotABundle(
                    "h^0(V) = 0 would need H^0(B) -> H^0(C) injective, but "
                    "h^0(B) = %d exceeds h^0(C) = %d. A stable bundle with "
                    "c_1(V) = 0 cannot arise from this monad."
                    % (hB[0], hC[0]))
            # h^3(V) = 0 forces r_2 to one specific value, which must lie in
            # the range a rank can occupy. When it does not -- and it does not
            # for about one monad in ninety -- h^3(V) is pinned above zero by
            # the sequence itself and no stable bundle can arise, however the
            # coefficients are chosen. Without this test the returned interval
            # escapes the unconditional one, which is impossible for a
            # tightening and is how the case was found.
            r2_forced = int(hC[2]) + int(hB[3]) - int(hC[3])
            if not (0 <= r2_forced <= min(int(hB[2]), int(hC[2]))):
                raise NotABundle(
                    "h^3(V) = 0 would force rank(H^2(B) -> H^2(C)) = %d, "
                    "outside its possible range [0, %d]. The sequence pins "
                    "h^3(V) above zero, so no slope-stable bundle with "
                    "c_1 = 0 arises from this monad."
                    % (r2_forced, min(int(hB[2]), int(hC[2]))))
            ind = int(self.index())
            base = int(hC[0]) - int(hB[0]) + int(hB[1])
            lo[0] = hi[0] = 0
            lo[3] = hi[3] = 0
            lo[1], hi[1] = base - rmax[1], base - rmin[1]
            lo[2], hi[2] = lo[1] + ind, hi[1] + ind
            if lo[1] < 0 or lo[2] < 0:
                raise NotABundle(
                    "imposing h^0(V) = h^3(V) = 0 forces a negative "
                    "cohomology dimension (h^1 in [%d, %d], h^2 in [%d, %d]), "
                    "so no stable bundle arises from this monad"
                    % (lo[1], hi[1], lo[2], hi[2]))

        out = {"hB": hB, "hC": hC,
               "bounds": list(zip(lo, hi)),
               "index": self.index(),
               "assumed_stable": bool(stable),
               "determined": [a == b for a, b in zip(lo, hi)]}
        return out


# ---------------------------------------------------------------------------
# searching
# ---------------------------------------------------------------------------

def scan(X, rank=5, charge=2, generations=3, symmetry_order=1,
         require_anomaly=True, require_stability=False, limit=100000,
         max_seconds=None, keep=None, as_objects=False, progress=None):
    r"""
    Search sums of ``rank`` line bundles for candidate models.

    The filter is applied in order of cost, which is the only way a search
    like this finishes:

    1. c_1(V) = 0, imposed by construction on the last summand rather than
       tested, so the loop runs over rank-1 free vectors;
    2. the index, which needs only triple intersection numbers;
    3. the anomaly surplus, also intersection numbers;
    4. optionally the poly-stability locus, which is a small optimisation;
    5. cohomology, which is *not* done here. Survivors are handed back for
       the caller to feed to :meth:`LineBundleSum.su5_spectrum` at leisure.

    Results are returned as plain lists of integer summands unless
    ``as_objects`` is set. That is not a stylistic choice: on the
    tetraquadric at charge 2 with ind(V) = -24 there are already more than
    80000 sums passing c_1 = 0, the index and the anomaly, and building a
    :class:`LineBundleSum` for each costs more than the entire search. The
    topological conditions are weak; poly-stability and cohomology are what
    actually cut the count down, and both are expensive enough to be worth
    applying deliberately rather than by default.

    ``limit`` is a hard cap on the number of models returned, and it has a
    finite default on purpose. The topological conditions alone are weak: on
    the tetraquadric at charge 2 with ind(V) = -6 they admit millions of
    sums, and an uncapped search exhausts memory before it exhausts the box.
    Hitting the cap logs a warning, because a truncated list is a different
    object from a complete one and silently returning the first hundred
    thousand would misrepresent it.

    ``max_seconds`` is a wall clock budget. The box grows like
    (2*charge+1)^(h11*rank) and there is no useful way to predict from the
    outside how long a given corner of it takes, so a search that must
    terminate should say by when. Exceeding the budget logs a warning and
    returns what was found, in the same truncated-not-complete sense as
    ``limit``.

    ``keep`` is an optional predicate on the list of summands, applied inside
    the loop alongside the other filters. It is the place to put a cheap
    model-specific condition -- no repeated summands, no trivial summand, a
    required equivariant structure -- so that it cuts the count before
    anything expensive sees it.

    ``generations`` and ``symmetry_order`` combine as ind(V) = -3|Gamma| in
    the sense of arXiv:1202.1757: the quotient by a freely acting Gamma
    divides the count, so upstairs one wants
    ind(V) = -generations * symmetry_order.

    The search is over the box |L^r| <= ``charge``. It is exhaustive within
    that box and says nothing about outside it.

    Notes
    -----
    Deduplication is by the sorted tuple of summands, so models differing
    only by the ordering of the line bundles are counted once. Models related
    by a symmetry of the configuration matrix are *not* identified; that
    would want :func:`pyCICY.transitions.canonical_key` lifted to act on
    bundles, which is a natural next step and is not done.
    """
    X = _as_cicy(X)
    if not X.fav:
        raise NotFavourable('scan requires a favourable configuration.')
    h11 = X.len
    target = -abs(generations) * abs(symmetry_order)
    d = _triple(X)
    c2X = _c2_tangent(X)

    if rank < 4:
        raise ValueError('scan is written for rank >= 4.')

    # Every quantity the filter uses is *additive over summands*: c_1, the
    # vector \int ch_2 J_r and the scalar 6 \int ch_3 are each an integer sum
    # of per-summand contributions. So this is a subset-sum problem, and
    # enumerating tuples is the wrong algorithm: at charge 2 on the
    # tetraquadric there are 6 x 10^9 of them. Instead the rank summands are
    # split as (rank-4) + 2 + 2, all pairs are hashed on their (c_1, 6 ch_3)
    # signature, and each outer choice is matched against that table in one
    # vectorised pass.
    cand = np.array(list(itertools.product(range(-charge, charge + 1),
                                           repeat=h11)), dtype=np.int64)
    if require_stability:
        # Poly-stability needs *every* summand to have a slope that changes
        # sign in the cone (or vanishes identically), and that is a property
        # of the summand alone. So it filters the pool, not the assembled
        # sums -- which is the difference between running the optimiser a few
        # thousand times and running it never at all for most of the box.
        mask = slope_candidates(d, cand)
        logger.info('slope pre-filter keeps %d of %d candidate line bundles',
                    int(mask.sum()), len(cand))
        cand = cand[mask]
        if len(cand) == 0:
            return []
    N = len(cand)
    candf = cand.astype(float)
    ch2_c = 0.5 * np.einsum('rst,as,at->ar', d, candf, candf)
    ch3_6 = np.rint(np.einsum('rst,ar,as,at->a', d, candf, candf, candf)
                    ).astype(np.int64)          # = 6 * ch_3
    ch2_hi = ch2_c.max(axis=0)
    target6 = int(round(6 * target))

    # -- all pairs, with the anomaly bound applied early --------------------
    ii, jj = np.triu_indices(N)
    if require_stability:
        # The subset obstruction is inherited: if the two members of a pair
        # already sum to a definite slope form, no completion of that pair
        # can be poly-stable. Applying it here, vectorised over the whole
        # pair table, is what keeps the search finite -- testing assembled
        # sums instead means paying for every one of them, and on the
        # tetraquadric at charge 2 there are of order 10^8.
        Mc = np.einsum('rst,ar->ast', d, cand.astype(float))
        Mc = Mc + np.transpose(Mc, (0, 2, 1))
        Mp = Mc[ii] + Mc[jj]
        nz = (np.abs(Mp) > 1e-12).any(axis=(1, 2))
        mixed = ((Mp > 1e-12).any(axis=(1, 2))
                 & (Mp < -1e-12).any(axis=(1, 2)))
        alive = mixed | ~nz
        logger.info('slope pair filter keeps %d of %d pairs',
                    int(alive.sum()), len(ii))
        ii, jj = ii[alive], jj[alive]
    p_c1 = cand[ii] + cand[jj]
    p_ch3 = ch3_6[ii] + ch3_6[jj]
    p_ch2 = ch2_c[ii] + ch2_c[jj]
    if require_anomaly:
        # rank-2 summands remain after this pair, each contributing at most
        # ch2_hi, so a pair failing this can never be completed.
        viable = np.all(c2X + p_ch2 + (rank - 2) * ch2_hi >= -1e-9, axis=1)
        ii, jj = ii[viable], jj[viable]
        p_c1, p_ch3, p_ch2 = p_c1[viable], p_ch3[viable], p_ch2[viable]

    # -- encode (c_1, 6 ch_3) as one integer, then sort --------------------
    span = rank * charge                      # |c_1^r| <= rank * charge
    base = 2 * span + 1
    lo3, hi3 = int(p_ch3.min()), int(p_ch3.max())
    m3 = hi3 - lo3 + 1

    def encode(c1v, ch3v):
        code = np.zeros(len(c1v), dtype=np.int64)
        for r in range(h11):
            code = code * base + (c1v[:, r] + span)
        return code * m3 + (ch3v - lo3)

    ok = np.all(np.abs(p_c1) <= span, axis=1)
    ii, jj, p_c1, p_ch3, p_ch2 = ii[ok], jj[ok], p_c1[ok], p_ch3[ok], p_ch2[ok]
    p_key = encode(p_c1, p_ch3)
    order = np.argsort(p_key, kind='stable')
    p_key_s = p_key[order]
    uniq, first = np.unique(p_key_s, return_index=True)
    bounds = np.append(first, len(p_key_s))
    u_c1 = p_c1[order][first]
    u_ch3 = p_ch3[order][first]

    # -- outer loop over the remaining rank-4 summands ---------------------
    outer = itertools.combinations_with_replacement(range(N), rank - 4)
    seen = set()
    out = []
    count = 0
    tick = 0
    started = _time.time()
    truncated = None

    for pick in outer:
        if truncated:
            break
        count += 1
        if progress is not None and count % 5000 == 0:
            progress(count, len(out))
        if max_seconds is not None and count % 64 == 0:
            if _time.time() - started > max_seconds:
                truncated = 'budget'
                break
        o_c1 = cand[list(pick)].sum(axis=0) if pick else np.zeros(h11,
                                                                  dtype=np.int64)
        o_ch3 = int(ch3_6[list(pick)].sum()) if pick else 0
        o_ch2 = (ch2_c[list(pick)].sum(axis=0) if pick
                 else np.zeros(h11))

        # the two pairs must together carry -o_c1 and target6 - o_ch3
        need_c1 = -o_c1 - u_c1
        need_ch3 = (target6 - o_ch3) - u_ch3
        good = np.all(np.abs(need_c1) <= span, axis=1) & \
            (need_ch3 >= lo3) & (need_ch3 <= hi3)
        if not good.any():
            continue
        want = encode(need_c1[good], need_ch3[good])
        pos = np.searchsorted(uniq, want)
        pos = np.clip(pos, 0, len(uniq) - 1)
        hit = uniq[pos] == want
        if not hit.any():
            continue

        left = np.flatnonzero(good)[hit]
        right = pos[hit]
        for a, b in zip(left, right):
            if uniq[a] > uniq[b]:          # each unordered split once
                continue
            if truncated:
                break
            for u in range(bounds[a], bounds[a + 1]):
                if truncated:
                    break
                for v in range(bounds[b], bounds[b + 1]):
                    # The budget is checked here, in the innermost loop, and
                    # not only in the outer one. For rank 5 the outer loop has
                    # a single element, so all of the work is inside; a check
                    # that only fires between outer choices would never fire
                    # at all.
                    tick += 1
                    if max_seconds is not None and tick % 256 == 0:
                        if _time.time() - started > max_seconds:
                            truncated = 'budget'
                            break
                    # Filters run *before* the duplicate test, deliberately.
                    # The same multiset of summands is reachable by several
                    # splits, so a set of everything seen is a set of every
                    # candidate, and on the tetraquadric at charge 2 that is
                    # tens of millions of tuples and gigabytes of memory. The
                    # filters are cheap and cut hard, so re-testing the odd
                    # duplicate costs far less than remembering all of them.
                    tot = o_ch2 + p_ch2[order[u]] + p_ch2[order[v]]
                    if require_anomaly and np.any(c2X + tot < -1e-9):
                        continue
                    idx = tuple(sorted(list(pick) + [ii[order[u]],
                                                     jj[order[u]],
                                                     ii[order[v]],
                                                     jj[order[v]]]))
                    summands = [list(map(int, cand[k])) for k in idx]
                    if require_stability:
                        # The sign obstruction first: it is exact and about a
                        # thousand times cheaper than the search, and on the
                        # examples tried it removes everything the search
                        # would have removed anyway.
                        if slope_subsets_definite(d, summands):
                            continue
                        if not stability_locus(X, summands)["found"]:
                            continue
                    if keep is not None and not keep(summands):
                        continue
                    if idx in seen:
                        continue
                    seen.add(idx)
                    out.append(LineBundleSum(X, summands) if as_objects
                               else summands)
                    if len(out) >= limit:
                        logger.warning(
                            'scan() stopped at limit=%d; the result is a '
                            'truncation of the search box, not all of it. '
                            'Tighten the box, or pass require_stability=True '
                            'or a keep= filter, rather than raising the '
                            'limit: the purely topological conditions are '
                            'weak and admit very large families.', limit)
                        return out
    if truncated == 'budget':
        logger.warning(
            'scan() stopped after %.1fs on a max_seconds=%s budget, having '
            'covered %d of the outer choices. The result is a truncation of '
            'the search box, not all of it.',
            _time.time() - started, max_seconds, count)
    return out
