r"""
pyCICY.theories.orientifold -- Type IIB orientifolds of a Calabi-Yau threefold.

The last of the three constructions the subpackage docstring named as missing,
and the one that sits closest to the geometry the rest of the package already
computes. An orientifold is a quotient of Type IIB by

    Omega_p (-1)^{F_L} sigma

with sigma a holomorphic involution of X. Everything four-dimensional follows
from how sigma acts on cohomology, and that action is what this module
computes -- exactly, by two independent routes that have to agree.

The two kinds
-------------
A holomorphic involution acts on the holomorphic three-form by a sign, and the
sign decides everything:

    sigma^* Omega = -Omega     O3 and O7 planes, fixed points and divisors
    sigma^* Omega = +Omega     O5 and O9 planes, fixed curves and all of X

That is not two conventions for the same thing. The dimensions of the fixed
components are forced by the sign -- even complex codimension for the first
case, odd for the second -- so the two are distinguished by geometry and the
module checks the agreement rather than assuming it. See
:meth:`SignInvolution.consistent`.

Two routes to the equivariant Hodge numbers
-------------------------------------------
The orientifold spectrum is a function of the split

    h^{1,1} = h^{1,1}_+ + h^{1,1}_-,    h^{2,1} = h^{2,1}_+ + h^{2,1}_-

and this module gets it twice.

The first route is the topological Lefschetz fixed point theorem. For a
holomorphic involution of a Calabi-Yau threefold,

    chi(Fix sigma) = 2 + 2 (h11_+ - h11_-) - tr(sigma | H^3)

and the trace on H^3 is fixed by the Omega sign together with
h21_+ - h21_-. On a favourable CICY the action on H^{1,1} is known -- the
ambient hyperplane classes span it and a sign-flip involution fixes each of
them -- so the only unknown is h21_+ - h21_-, and the fixed locus determines
it. Computing chi(Fix) means computing Euler characteristics of complete
intersections in products of projective spaces, which
:func:`complete_intersection_euler` does from Chern classes.

The second route, for a hypersurface, is to count monomials.
:func:`hypersurface_moduli_split` grades the Jacobian ring by the involution
and reads off the split directly. It shares nothing with the first route --
one is a fixed-point theorem, the other is linear algebra on monomials -- and
they agree.

The subtlety they agree *through* is worth naming, because getting it wrong
would leave both routes self-consistent and both wrong.
:math:`H^{2,1} \cong H^1(X, T_X) \otimes H^{3,0}`, and the polynomial
deformations grade :math:`H^1(T_X)`, not :math:`H^{2,1}`. When
:math:`\sigma^* \Omega = -\Omega` the two gradings are opposite, so the
invariant deformations are the *anti*-invariant part of :math:`H^{2,1}`. On
the quintic with one sign flip that is the difference between (38, 63) and
(63, 38), and the fixed-point theorem is what says which.

Sen's limit
-----------
:class:`SenLimit` connects this to :mod:`pyCICY.theories.ftheory`. F-theory
on an elliptic threefold over a base B degenerates, at weak coupling, to Type
IIB on the double cover of B branched over a curve in |-2K_B|, which is the
O7-plane. Three things fall out and all three are checked:

  * the double cover is a K3 for *every* rational base, since
    chi = 2 chi(B) - chi(O7) = 2(3+T) + 2(9-T) = 24 with the T cancelling
  * the D7 tadpole closes: the Whitney brane is in class -8K_B, and with its
    orientifold image that is 16 c_1 = 8 [O7]
  * the perturbative brane rules reproduce Kodaira. A stack of n D7-branes on
    top of the O7 gives so(2n), and the fibre type there is I_{n-4}^*, which
    :func:`~pyCICY.theories.ftheory.kodaira_type` independently says carries
    so(2n-8+8) = so(2n)

The last is the sharpest check in the module: one side is a count of Chan-Paton
factors, the other is the vanishing order of a discriminant, and neither knows
about the other.

What is not here
----------------
Fluxes. The D3 tadpole is stated as the total charge that has to be cancelled,
which the F-theory fourfold gives exactly as chi/24, but the decomposition
into O3-plane, D7 and flux contributions is not implemented and
:meth:`Orientifold.d3_tadpole` says so rather than returning a partial sum.
Nor are the Yukawa couplings, which in four dimensions exist -- unlike the
six-dimensional F-theory case -- and are simply not computed here.

References
----------
Sen, Orientifold limit of F-theory vacua, Phys. Rev. D55 (1997) 7345.
Grimm and Louis, The effective action of N=1 Calabi-Yau orientifolds,
    Nucl. Phys. B699 (2004) 387.
Blumenhagen, Kors, Lust and Stieberger, Four-dimensional string
    compactifications with D-branes, orientifolds and fluxes,
    Phys. Rept. 445 (2007) 1.
Collinucci, Denef and Esole, D-brane deconstructions in IIB orientifolds,
    JHEP 0902 (2009) 005.
"""

import itertools

import numpy as np

from .base import NeedsMetric, Theory, register
from . import ftheory as FT

__all__ = ["complete_intersection_euler", "SignInvolution", "Orientifold",
           "SenLimit", "hypersurface_moduli_split", "brane_stack"]


# ---------------------------------------------------------------------------
# Euler characteristics of complete intersections
# ---------------------------------------------------------------------------
#
# The fixed locus of an involution is a complete intersection in a product of
# projective spaces, but not a Calabi-Yau one -- it is a surface, a curve or a
# set of points -- so pyCICY.CICY cannot be used for it. The Chern class
# computation is short enough to do directly.


def _truncated_ring(dims):
    """Exponent bounds for H_i, since H_i^{n_i+1} = 0 in the cohomology."""
    return [int(d) for d in dims]


def _mul(a, b, dims):
    out = {}
    for ea, ca in a.items():
        for eb, cb in b.items():
            e = tuple(x + y for x, y in zip(ea, eb))
            if any(e[i] > dims[i] for i in range(len(dims))):
                continue
            out[e] = out.get(e, 0) + ca * cb
    return {e: c for e, c in out.items() if c}


def _add(a, b):
    out = dict(a)
    for e, c in b.items():
        out[e] = out.get(e, 0) + c
    return {e: c for e, c in out.items() if c}


def _scale(a, k):
    return {e: c * k for e, c in a.items() if c * k}


def _one(dims):
    return {tuple([0] * len(dims)): 1}


def _linear(coeffs, dims):
    """The class sum_i coeffs[i] H_i."""
    out = {}
    for i, c in enumerate(coeffs):
        if c:
            e = [0] * len(dims)
            e[i] = 1
            out[tuple(e)] = int(c)
    return out


def _power(a, n, dims):
    out = _one(dims)
    for _ in range(int(n)):
        out = _mul(out, a, dims)
    return out


def _inverse(a, dims):
    """1 / (1 + x) as 1 - x + x^2 - ..., valid since x is nilpotent here."""
    x = {e: c for e, c in a.items() if any(e)}
    total = _one(dims)
    term = _one(dims)
    for _ in range(sum(dims)):
        term = _scale(_mul(term, x, dims), -1)
        if not term:
            break
        total = _add(total, term)
    return total


def complete_intersection_euler(dims, degrees):
    r"""Euler characteristic of a complete intersection in a product of P^n.

    For Y the intersection of K hypersurfaces of multidegrees ``degrees`` in
    A = prod_i P^{n_i}, the total Chern class of Y is

        c(T_Y) = prod_i (1 + H_i)^{n_i + 1} / prod_a (1 + sum_i q^a_i H_i)

    restricted to Y, and chi(Y) is the degree-dim(Y) piece integrated against
    the class of Y, which is prod_a (sum_i q^a_i H_i).

    Parameters
    ----------
    dims : list of int
        The n_i of the ambient factors.
    degrees : list of list of int
        One multidegree per hypersurface. May be empty, in which case Y is
        the ambient space itself.

    Returns
    -------
    int
        chi(Y), or 0 when the equations over-determine Y and it is generically
        empty.

    Examples
    --------
    The quintic threefold, and a quintic surface and curve:

    >>> complete_intersection_euler([4], [[5]])
    -200
    >>> complete_intersection_euler([3], [[5]])
    55
    >>> complete_intersection_euler([2], [[5]])
    -10

    A quartic surface in P^3 is a K3:

    >>> complete_intersection_euler([3], [[4]])
    24

    With no equations the answer is the ambient Euler characteristic:

    >>> complete_intersection_euler([1, 2], [])
    6
    """
    dims = [int(d) for d in dims]
    degrees = [list(map(int, q)) for q in degrees]
    if not dims:
        return 1
    for q in degrees:
        if len(q) != len(dims):
            raise ValueError("each multidegree needs one entry per ambient "
                             "factor; got %s for %d factors" % (q, len(dims)))
    d = sum(dims) - len(degrees)
    if d < 0:
        return 0                       # more equations than dimensions

    top = _one(dims)
    for i, n in enumerate(dims):
        e = [0] * len(dims)
        e[i] = 1
        top = _mul(top, _power(_add(_one(dims), {tuple(e): 1}), n + 1, dims),
                   dims)
    cls = _one(dims)
    for q in degrees:
        lin = _linear(q, dims)
        top = _mul(top, _inverse(_add(_one(dims), lin), dims), dims)
        cls = _mul(cls, lin, dims)

    total = _mul(top, cls, dims)
    return int(total.get(tuple(dims), 0))


# ---------------------------------------------------------------------------
# involutions
# ---------------------------------------------------------------------------


class SignInvolution(object):
    r"""An involution of a CICY acting by sign flips on the coordinates.

    In each ambient factor P^{n_i} a subset of the homogeneous coordinates is
    sent to minus itself. The defining polynomials are taken to be invariant,
    which is a condition on their monomials and is what makes X itself
    invariant.

    This is deliberately narrower than "every involution of a CICY". Actions
    that permute the ambient factors are also involutions, and they act
    non-trivially on H^{1,1}, but they also permute the defining polynomials,
    and a polynomial that is exchanged with another restricts to the fixed
    locus in a way that depends on a sign not determined by the configuration
    matrix. The sign-flip case has no such ambiguity: everything below is
    fixed by the configuration and the choice of which coordinates flip.

    Parameters
    ----------
    conf : configuration matrix, or CICY
    flips : list of list of int
        Per ambient factor, the indices of the coordinates that change sign.
        A factor with an empty list is untouched.
    name : str, optional

    Attributes
    ----------
    minus : list of int
        How many coordinates flip in each factor.

    Examples
    --------
    The quintic with one coordinate flipped, which is an O3/O7 involution:

    >>> s = SignInvolution([[4, 5]], [[4]])
    >>> s.omega_sign()
    -1
    >>> s.hodge_split()["h21"]
    (38, 63)

    With two flipped it becomes O5/O9 instead:

    >>> SignInvolution([[4, 5]], [[3, 4]]).omega_sign()
    1
    """

    def __init__(self, conf, flips, poly_signs=None, name=None):
        from ..pyCICY import CICY
        self.X = conf if isinstance(conf, CICY) else CICY(
            np.asarray(conf, dtype=int).tolist())
        self.M = np.asarray(self.X.M, dtype=int)
        self.dims = [int(d) for d in self.M[:, 0]]
        self.degrees = [[int(self.M[i][a + 1]) for i in range(len(self.dims))]
                        for a in range(self.M.shape[1] - 1)]
        if len(flips) != len(self.dims):
            raise ValueError("flips needs one list per ambient factor; got "
                             "%d for %d factors" % (len(flips), len(self.dims)))
        self.flips = [sorted(set(int(k) for k in f)) for f in flips]
        for i, f in enumerate(self.flips):
            if f and (f[0] < 0 or f[-1] > self.dims[i]):
                raise ValueError(
                    "factor %d is P^%d with coordinates 0..%d; cannot flip %s"
                    % (i, self.dims[i], self.dims[i], f))
        self.minus = [len(f) for f in self.flips]
        self.plus = [self.dims[i] + 1 - self.minus[i]
                     for i in range(len(self.dims))]

        # A factor with every coordinate flipped, or none, is untouched: the
        # overall sign is the projective scaling. If that is true of every
        # factor the map is the identity on the ambient and there is no
        # involution to speak of, whatever it does to the polynomials.
        if all(m in (0, self.dims[i] + 1) for i, m in enumerate(self.minus)):
            raise ValueError(
                "flipping every coordinate of a factor, or none, is the "
                "identity on that factor, since the overall sign is the "
                "projective scaling. This choice is the identity on the "
                "whole ambient space and is not an involution of X.")

        self.signs = self._resolve_signs(poly_signs)
        self.name = name or "sign involution %s" % (self.flips,)

    def _monomial_parities(self, a):
        """Which parities of flipped-coordinate count a polynomial can have.

        A monomial of multidegree q^a uses some number of flipped coordinates
        in each factor, and the involution multiplies it by minus one to that
        total. So the polynomial splits by parity, and which parities are
        available at all is fixed by the multidegrees: a factor with q^a_i
        degree and no unflipped coordinates has no choice.
        """
        reach = {0}
        for i in range(len(self.dims)):
            q = self.degrees[a][i]
            lo = 0 if self.plus[i] else q          # forced all-flipped
            hi = q if self.minus[i] else 0         # forced none-flipped
            here = set()
            for k in range(lo, hi + 1):
                if k > q or (k and not self.minus[i]) or \
                        (k < q and not self.plus[i]):
                    continue
                here.add(k % 2)
            if not here:
                return set()
            reach = {(r + h) % 2 for r in reach for h in here}
        return reach

    def _resolve_signs(self, poly_signs):
        """Fix eps_a in p_a(sigma x) = eps_a p_a(x), checking it is possible.

        A polynomial invariant under the involution is one whose monomials
        all use an even number of flipped coordinates; anti-invariant is all
        odd. Either can define an invariant X, since X is the zero locus and
        does not see the sign of p, so both are legitimate orientifolds and
        they are different ones. When the multidegrees leave both open the
        default is the invariant choice; when only one parity has monomials
        at all, that one is forced.
        """
        n = len(self.degrees)
        if poly_signs is None:
            out = []
            for a in range(n):
                reach = self._monomial_parities(a)
                if not reach:
                    raise ValueError(
                        "no monomial of multidegree %s exists for this "
                        "coordinate split" % (self.degrees[a],))
                out.append(1 if 0 in reach else -1)
            return out
        if len(poly_signs) != n:
            raise ValueError("poly_signs needs one entry per defining "
                             "polynomial; got %d for %d"
                             % (len(poly_signs), n))
        out = []
        for a, e in enumerate(poly_signs):
            e = int(e)
            if e not in (1, -1):
                raise ValueError("each polynomial sign is +1 or -1")
            want = 0 if e > 0 else 1
            if want not in self._monomial_parities(a):
                raise ValueError(
                    "polynomial %d cannot have sign %+d: no monomial of "
                    "multidegree %s uses an %s number of flipped coordinates"
                    % (a, e, self.degrees[a],
                       "even" if want == 0 else "odd"))
            out.append(e)
        return out

    def __repr__(self):
        return "<%s on %s>" % (self.name, self.M.tolist())

    # -- the sign on Omega -------------------------------------------------

    def omega_sign(self):
        r"""The sign in sigma^* Omega = s Omega.

        The holomorphic three-form is a residue of the ambient measure
        divided by the defining polynomials. A diagonal sigma multiplies the
        measure of factor i by det(g_i) = (-1)^{minus_i}, and divides by
        eps_a for each polynomial, so

            s = prod_i (-1)^{minus_i} * prod_a eps_a .

        Negative means O3/O7, positive means O5/O9.

        The polynomial signs are not decoration. Flipping a set of
        coordinates and flipping its complement are the *same* map on
        projective space, and the two descriptions give opposite
        ``sum_i minus_i``; what puts them back together is that they also give
        opposite eps_a. Dropping the second factor would make the sign depend
        on how the same involution was written down.
        """
        s = -1 if sum(self.minus) % 2 else 1
        for e in self.signs:
            s *= e
        return s

    def oplane_type(self):
        """``"O3/O7"`` or ``"O5/O9"``, from :meth:`omega_sign`."""
        return "O3/O7" if self.omega_sign() < 0 else "O5/O9"

    # -- polynomials on the fixed locus -----------------------------------

    def _vanishes_on(self, choice, a):
        r"""Whether polynomial ``a`` restricts to zero on a fixed component.

        On the component where factor i keeps only its minus-coordinates, a
        monomial of the polynomial uses q^a_i of them. Invariance requires
        every monomial to use an even total number of minus-coordinates, so if

            sum over factors kept on the minus side of q^a_i

        is odd, no invariant monomial survives the restriction and the
        polynomial vanishes identically on that component -- which means it
        imposes no condition there, and the whole component lies inside X.

        This is not an edge case. It is what makes an O5-plane an O5-plane:
        on the two-flip quintic the line {x_0 = x_1 = x_2 = 0} lies entirely
        in X for exactly this reason, and that curve is the fixed locus.
        """
        s = sum(self.degrees[a][i] for i in range(len(self.dims))
                if choice[i] == "-")
        want = 0 if self.signs[a] > 0 else 1
        return s % 2 != want

    def fixed_components(self):
        r"""The components of the fixed locus, with their Euler characteristics.

        The fixed locus of a diagonal involution in P^{n} is the disjoint
        union of the two coordinate subspaces P^{plus-1} and P^{minus-1}, so
        in a product the ambient fixed locus has one component per choice of
        side per factor. Each is a smaller product of projective spaces, and
        the part of it lying in X is the complete intersection of whichever
        defining polynomials do not vanish identically on it.

        Returns
        -------
        list of dict
            ``sides`` (a tuple of ``"+"``/``"-"``), ``ambient`` dimensions,
            ``equations`` kept, ``dim`` (complex dimension in X), ``euler``,
            and ``oplane`` -- ``"O3"``, ``"O5"``, ``"O7"`` or ``"O9"``.
            Components that are generically empty are omitted.
        """
        out = []
        options = []
        for i in range(len(self.dims)):
            side = []
            if self.plus[i] > 0:
                side.append("+")
            if self.minus[i] > 0:
                side.append("-")
            options.append(side)

        for choice in itertools.product(*options):
            sub = [(self.plus[i] if choice[i] == "+" else self.minus[i]) - 1
                   for i in range(len(self.dims))]
            keep = [self.degrees[a] for a in range(len(self.degrees))
                    if not self._vanishes_on(choice, a)]
            # A factor contributing P^{-1} is the empty set, so the whole
            # component is empty. This cannot happen given the loop above,
            # which only offers a side when it has coordinates, but the guard
            # keeps the arithmetic below honest.
            if any(k < 0 for k in sub):
                continue
            dim = sum(sub) - len(keep)
            if dim < 0:
                continue                        # over-determined, empty
            chi = complete_intersection_euler(sub, keep)
            if dim == 0 and chi == 0:
                continue
            out.append({"sides": choice, "ambient": sub,
                        "equations": keep, "dim": dim, "euler": chi,
                        "inside_X": not keep,
                        "oplane": {0: "O3", 1: "O5", 2: "O7",
                                   3: "O9"}.get(dim, "dim %d" % dim)})
        return out

    def degeneracies(self):
        r"""Ways this involution forces X to be something other than generic.

        When every defining polynomial vanishes identically on a component of
        the ambient fixed locus, that whole coordinate subspace lies inside X,
        and two things can go wrong.

        If the component has the dimension of X, then X contains it, so X is
        not irreducible -- there is no smooth Calabi-Yau with this involution
        at all. On the quintic, flipping four coordinates makes every
        invariant monomial divisible by the fifth, so the polynomial
        factorises.

        If the component is a divisor in X, then X contains a linear subspace
        of the ambient as a divisor. That is a Noether-Lefschetz jump: the
        class is not in the lattice the ambient generates, so h^{1,1} is
        larger than the configuration matrix says and the CICY Hodge numbers
        do not apply to this X. On the quintic, flipping three coordinates
        forces X to contain a plane, and a quintic containing a plane has
        h^{1,1} = 2 rather than 1.

        Both are reported rather than silently worked around, because in
        either case a Hodge number taken from the configuration matrix would
        be a number for a different manifold.

        Returns
        -------
        list of dict
            Empty when the involution is generic.
        """
        out = []
        for c in self.fixed_components():
            if not c["inside_X"]:
                continue
            amb = sum(c["ambient"])
            if amb >= self.X.nfold:
                out.append({"kind": "reducible", "component": c,
                            "note": "every defining polynomial vanishes on a "
                                    "subspace of dimension %d, which is at "
                                    "least dim X = %d, so X is not "
                                    "irreducible"
                                    % (amb, int(self.X.nfold))})
            elif amb == self.X.nfold - 1:
                out.append({"kind": "picard jump", "component": c,
                            "note": "X contains the ambient subspace P^%s as "
                                    "a divisor, so h^{1,1} exceeds the value "
                                    "the configuration matrix gives"
                                    % (c["ambient"],)})
        return out

    def is_generic(self):
        """Whether :meth:`degeneracies` is empty."""
        return not self.degeneracies()

    def fixed_euler(self):
        """chi of the whole fixed locus, the sum over its components."""
        return sum(c["euler"] for c in self.fixed_components())

    def consistent(self):
        r"""Whether the fixed dimensions match the sign on Omega.

        A holomorphic involution with sigma^* Omega = -Omega has fixed
        components of even complex codimension, so on a threefold they are
        points and divisors: O3 and O7. With the plus sign they have odd
        codimension, so curves and the whole space: O5 and O9. The two
        computations here are independent -- one counts flipped coordinates,
        the other intersects polynomials with coordinate subspaces -- so the
        agreement is a check rather than a definition.
        """
        want = {-1: (0, 2), 1: (1, 3)}[self.omega_sign()]
        return all(c["dim"] in want for c in self.fixed_components())

    # -- the equivariant Hodge numbers -------------------------------------

    def hodge_split(self):
        r"""(h^{1,1}_+, h^{1,1}_-) and (h^{2,1}_+, h^{2,1}_-).

        By the topological Lefschetz fixed point theorem,

            chi(Fix sigma) = sum_k (-1)^k tr(sigma | H^k)
                           = 2 + 2 (h11_+ - h11_-) - tr(sigma | H^3)

        using that the traces on H^2 and H^4 agree by Poincare duality. On
        H^3 the holomorphic three-form and its conjugate contribute twice the
        Omega sign, and H^{2,1} with H^{1,2} contribute twice
        h21_+ - h21_-, so

            chi(Fix) = 2 + 2 (h11_+ - h11_-) - 2 s - 2 (h21_+ - h21_-)

        with s the sign from :meth:`omega_sign`. Everything but
        h21_+ - h21_- is known, and h21_+ + h21_- is h^{2,1}, so both follow.

        Requires a favourable X, where H^{1,1} descends from the ambient and
        a sign-flip involution therefore acts on it trivially. On a
        non-favourable X there are classes the configuration matrix does not
        see and the action on them is not determined here, so the method
        refuses.

        Returns
        -------
        dict
            ``h11``, ``h21`` (each a ``(plus, minus)`` pair), ``omega_sign``,
            ``fixed_euler``, ``oplanes``.
        """
        bad = self.degeneracies()
        if bad:
            raise ValueError(
                "this involution does not leave a generic X: %s. The Hodge "
                "numbers of the configuration matrix therefore do not "
                "describe the invariant manifold, and splitting them would "
                "split the wrong numbers. See degeneracies()."
                % "; ".join(b["kind"] for b in bad))
        if not self.X.fav:
            raise NotImplementedError(
                "this CICY is not favourable, so H^{1,1} is not spanned by "
                "the ambient hyperplane classes and the involution's action "
                "on the extra classes is not determined by the configuration "
                "matrix. The fixed locus and the Omega sign are still "
                "available; the Hodge split is not.")

        h11 = int(self.X.h[2])
        h21 = int(self.X.h[1])
        # Sign flips fix every ambient hyperplane class, so on a favourable X
        # the whole of H^{1,1} is invariant. A factor-permuting involution
        # would not have this property, which is the other reason this class
        # is restricted to sign flips.
        h11_plus, h11_minus = h11, 0

        s = self.omega_sign()
        chi = self.fixed_euler()
        # chi = 2 + 2(h11+ - h11-) - 2 s - 2 (h21+ - h21-)
        two_u = 2 + 2 * (h11_plus - h11_minus) - 2 * s - chi
        if two_u % 2:
            raise ValueError(
                "the Lefschetz number gives a half-integer for "
                "h21_+ - h21_-, which cannot happen; chi(Fix) = %d" % chi)
        u = two_u // 2
        if (h21 + u) % 2:
            raise ValueError(
                "h^{2,1} = %d and h21_+ - h21_- = %d have different parity, "
                "so no integer split exists. Either the involution does not "
                "preserve X or the fixed locus is not what was computed."
                % (h21, u))
        h21_plus = (h21 + u) // 2
        h21_minus = (h21 - u) // 2
        if h21_plus < 0 or h21_minus < 0:
            raise ValueError(
                "the split came out negative, (%d, %d), which is not a "
                "cohomology" % (h21_plus, h21_minus))
        return {"h11": (h11_plus, h11_minus),
                "h21": (h21_plus, h21_minus),
                "omega_sign": s, "fixed_euler": chi,
                "oplanes": self.oplane_counts()}

    def oplane_counts(self):
        """How many fixed components of each O-plane type, and their chi."""
        out = {}
        for c in self.fixed_components():
            k = c["oplane"]
            e = out.setdefault(k, {"components": 0, "euler": 0})
            e["components"] += 1
            e["euler"] += c["euler"]
        return out


def hypersurface_moduli_split(dims, degree, flips, poly_sign=None):
    r"""The complex structure split by counting monomials. Hypersurfaces only.

    An independent route to ``h^{2,1}_\pm`` for a hypersurface, sharing
    nothing with :meth:`SignInvolution.hodge_split`. The complex structure
    deformations of a hypersurface of multidegree ``degree`` in
    ``prod_i P^{n_i}`` are the monomials of that multidegree, modulo the
    overall scale and the reparametrisations of the ambient factors:

        h^{2,1} = (number of monomials) - 1 - sum_i ((n_i + 1)^2 - 1) .

    The involution grades both the monomials and the reparametrisations, so
    it grades the quotient.

    The catch, and the reason this function exists as a *check* rather than as
    the implementation: these deformations grade ``H^1(X, T_X)``, and
    ``H^{2,1} = H^1(T_X) tensor H^{3,0}``. When ``sigma^* Omega = -Omega`` the
    two gradings are opposite. The returned ``h21`` has the twist applied;
    ``deformations`` has it not, so both are visible.

    Returns
    -------
    dict
        ``h21`` as a ``(plus, minus)`` pair, ``deformations`` the untwisted
        split of ``H^1(T_X)``, ``monomials``, ``reparametrisations``,
        ``omega_sign``.

    Examples
    --------
    The quintic with one flip, agreeing with the fixed-point theorem:

    >>> hypersurface_moduli_split([4], [5], [[4]])["h21"]
    (38, 63)
    """
    dims = [int(d) for d in dims]
    degree = [int(q) for q in degree]
    flips = [sorted(set(int(k) for k in f)) for f in flips]
    if not (len(dims) == len(degree) == len(flips)):
        raise ValueError("dims, degree and flips need the same length")

    nminus = [len(f) for f in flips]
    nplus = [dims[i] + 1 - nminus[i] for i in range(len(dims))]
    if all(m in (0, dims[i] + 1) for i, m in enumerate(nminus)):
        raise ValueError(
            "this flip pattern is the identity on the ambient space; see "
            "SignInvolution for why")

    # Monomials of the given multidegree, graded by the parity of how many
    # flipped coordinates they use.
    mon = [0, 0]
    for i, n in enumerate(dims):
        counts = _monomials_by_parity(n + 1, degree[i], nminus[i])
        if i == 0:
            mon = list(counts)
        else:
            mon = [mon[0] * counts[0] + mon[1] * counts[1],
                   mon[0] * counts[1] + mon[1] * counts[0]]
    if mon[0] == 0 and mon[1] == 0:
        raise ValueError("no monomials of that multidegree")

    # eps, the sign in p(sigma x) = eps p(x). Both signs give a legitimate and
    # different orientifold when both parities of monomial exist; when only
    # one exists it is forced. See SignInvolution._resolve_signs.
    if poly_sign is None:
        eps = 1 if mon[0] else -1
    else:
        eps = int(poly_sign)
        if eps not in (1, -1):
            raise ValueError("poly_sign is +1 or -1")
        if mon[0 if eps > 0 else 1] == 0:
            raise ValueError(
                "no monomial of multidegree %s has the parity that sign "
                "%+d requires" % (degree, eps))
    if eps < 0:
        mon = [mon[1], mon[0]]                   # grade by agreement with eps
    s = (-1 if sum(nminus) % 2 else 1) * eps

    # The same degeneracy the fixed locus sees: if the defining polynomial
    # vanishes identically on a coordinate subspace of dimension dim X or
    # dim X - 1, then X is reducible or its Picard number jumps, and the
    # count below is a count for a manifold other than the one asked about.
    dimX = sum(dims) - 1
    for sides in itertools.product(*[[s_ for s_ in ("+", "-")
                                      if (nplus[i] if s_ == "+"
                                          else nminus[i])]
                                     for i in range(len(dims))]):
        parity = sum(degree[i] for i in range(len(dims))
                     if sides[i] == "-") % 2
        if parity == (0 if eps > 0 else 1):
            continue                             # restricts, imposes a condition
        amb = sum((nplus[i] if sides[i] == "+" else nminus[i]) - 1
                  for i in range(len(dims)))
        if amb >= dimX - 1:
            raise ValueError(
                "the defining polynomial vanishes identically on a "
                "coordinate subspace of dimension %d, with dim X = %d, so X "
                "is reducible or contains an ambient divisor and its Hodge "
                "numbers are not the generic ones. SignInvolution."
                "degeneracies() reports the same thing." % (amb, dimX))

    # The reparametrisation algebra gl(n_i+1) is graded by conjugation: the
    # blocks preserving the two coordinate sets are invariant, the off-blocks
    # are not. One overall scale is removed, and it is invariant.
    rep = [0, 0]
    for i, n in enumerate(dims):
        m = nminus[i]
        p = n + 1 - m
        rep[0] += p * p + m * m
        rep[1] += 2 * p * m
    # gl(n_i+1) minus its scalar, once per factor: the reparametrisation
    # group is prod_i PGL(n_i+1), not GL of the whole thing. The scalar is
    # the identity matrix, which conjugation fixes, so it comes off the
    # invariant side.
    rep[0] -= len(dims)
    mon_minus_scale = [mon[0] - 1, mon[1]]       # the scale of the polynomial

    defo = (mon_minus_scale[0] - rep[0], mon_minus_scale[1] - rep[1])
    if defo[0] < 0 or defo[1] < 0:
        raise ValueError("the deformation count came out negative, %s" % (defo,))

    h21 = defo if s > 0 else (defo[1], defo[0])
    return {"h21": h21, "deformations": defo, "monomials": tuple(mon),
            "reparametrisations": tuple(rep), "omega_sign": s,
            "h21_total": defo[0] + defo[1]}


def _monomials_by_parity(nvars, degree, nminus):
    """(even, odd) counts of degree-d monomials by how many flipped vars."""
    even = odd = 0
    nplus = nvars - nminus
    for k in range(degree + 1):                  # k flipped coordinates used
        a = _n_monomials(nminus, k)
        b = _n_monomials(nplus, degree - k)
        if k % 2:
            odd += a * b
        else:
            even += a * b
    return even, odd


def _n_monomials(nvars, degree):
    """Monomials of the given degree in the given number of variables."""
    if degree < 0:
        return 0
    if nvars == 0:
        return 1 if degree == 0 else 0
    from math import comb
    return comb(degree + nvars - 1, nvars - 1)


# ---------------------------------------------------------------------------
# the theory
# ---------------------------------------------------------------------------


@register
class Orientifold(Theory):
    r"""Type IIB on a Calabi-Yau threefold orientifold.

    Parameters
    ----------
    involution : SignInvolution
        Or a ``(conf, flips)`` pair, which is passed to
        :class:`SignInvolution`.
    name : str, optional

    Examples
    --------
    >>> o = Orientifold(SignInvolution([[4, 5]], [[4]]))
    >>> o.gauge_group()
    'determined by the D7-brane configuration, which is a separate choice'
    >>> s = o.spectrum()
    >>> s["chiral"], s["vectors"]
    (65, 38)
    """

    key = "type-iib-orientifold"

    def __init__(self, involution, name=None):
        if not isinstance(involution, SignInvolution):
            conf, flips = involution
            involution = SignInvolution(conf, flips)
        Theory.__init__(self, involution.X, name=name)
        self.involution = involution

    def geometry(self):
        return "%s / %s" % (self.X.M.tolist(), self.involution.oplane_type())

    def gauge_group(self):
        """Not fixed by the geometry: it is a choice of D7-brane stacks."""
        return ("determined by the D7-brane configuration, which is a "
                "separate choice")

    def spectrum(self):
        r"""The closed string spectrum in four dimensions. Exact.

        For an O3/O7 orientifold the massless closed string sector is, in
        N=1 language,

            h^{1,1}_+   chiral, the Kahler moduli paired with C_4
            h^{1,1}_-   chiral, from B_2 and C_2
            h^{2,1}_-   chiral, the complex structure moduli
            h^{2,1}_+   vector, U(1)s from C_4
            1           chiral, the axio-dilaton

        This is bookkeeping on the equivariant Hodge numbers and nothing more,
        which is why it is exact. The *open* string sector -- the gauge group
        and the charged matter -- is not determined by the involution: it is a
        choice of D7-branes subject to the tadpole condition, and
        :meth:`d7_tadpole` states that condition rather than solving it.

        Raises
        ------
        NotImplementedError
            For an O5/O9 involution. The multiplet assignment there is
            different and is not implemented; the Hodge split and the O-plane
            content are still available from the involution.
        """
        if self.involution.omega_sign() > 0:
            raise NotImplementedError(
                "this is an O5/O9 involution, where the four-dimensional "
                "multiplet assignment differs from the O3/O7 case and is not "
                "implemented here. involution.hodge_split() still gives the "
                "equivariant Hodge numbers, and fixed_components() the "
                "O-planes.")
        h = self.involution.hodge_split()
        h11p, h11m = h["h11"]
        h21p, h21m = h["h21"]
        return {"h11_plus": h11p, "h11_minus": h11m,
                "h21_plus": h21p, "h21_minus": h21m,
                "chiral": h11p + h11m + h21m + 1,
                "vectors": h21p,
                "kahler moduli": h11p,
                "complex structure moduli": h21m,
                "axio-dilaton": 1,
                "oplanes": h["oplanes"]}

    def d7_tadpole(self):
        r"""The D7-brane tadpole condition. Stated, not solved.

        The O7-planes carry D7-brane charge -8 in units where a brane and its
        orientifold image count as two, so the branes must satisfy

            sum_a N_a ( [D_a] + [D_a'] ) = 8 [O7]

        in H^2(X). This is a condition on a choice not made here -- the
        divisor classes the D7-branes wrap -- so the method reports the charge
        to be cancelled rather than a configuration cancelling it.
        """
        comps = [c for c in self.involution.fixed_components()
                 if c["oplane"] == "O7"]
        return {"o7_components": len(comps),
                "o7_euler": sum(c["euler"] for c in comps),
                "condition": "sum_a N_a ([D_a] + [D_a']) = 8 [O7]",
                "note": "the D7-brane classes are a choice this module does "
                        "not make; see SenLimit for the configuration that "
                        "F-theory picks out at weak coupling"}

    def d3_tadpole(self):
        """Always raises. The decomposition needs flux data.

        The total D3-brane charge to be cancelled is exact and F-theory gives
        it as chi(X_4)/24, which
        :meth:`pyCICY.theories.ftheory.FTheory4D.d3_tadpole` computes. Its
        decomposition into O3-plane, D7-brane and flux contributions needs
        the flux, which this package does not carry.
        """
        raise NeedsMetric(
            "the D3 tadpole splits into contributions from the O3-planes, "
            "the curvature of the D7-brane divisors, and the three-form "
            "flux. The flux is a choice this package does not carry, so no "
            "number is returned. The total charge is exact on the F-theory "
            "side as chi(X_4)/24; see FTheory4D.d3_tadpole().",
            missing=self.missing_for_physical())

    def holomorphic_yukawa(self, **kw):
        """Not implemented, and unlike the six-dimensional case they exist.

        Four-dimensional N=1 does admit a superpotential, so an orientifold
        with D7-branes has Yukawa couplings: they come from triple overlaps of
        open string wavefunctions on the intersections of brane divisors, with
        worldsheet instanton corrections. That is a computation this module
        does not do. Contrast
        :meth:`pyCICY.theories.ftheory.FTheory6D.holomorphic_yukawa`, where
        the couplings do not exist at all.
        """
        raise NotImplementedError(
            "Yukawa couplings in a four-dimensional orientifold come from "
            "overlaps of open string wavefunctions on intersecting D7-brane "
            "divisors, plus worldsheet instantons. They exist and are not "
            "computed here, which is a missing feature rather than an "
            "obstruction -- and a different statement from the "
            "six-dimensional F-theory case, where NoSuchTheory says there is "
            "nothing to compute.")

    def missing_for_physical(self):
        return [
            "a choice of D7-brane divisor classes satisfying the tadpole "
            "condition, which fixes the gauge group and the charged matter",
            "the three-form flux, without which the D3 tadpole cannot be "
            "decomposed and no moduli are stabilised",
        ] + Theory.missing_for_physical(self)

    def describe(self):
        inv = self.involution
        lines = ["%s on %s" % (self.name, self.geometry()),
                 "  sigma^* Omega = %+d, so %s planes"
                 % (inv.omega_sign(), inv.oplane_type()),
                 "  fixed locus chi = %d, consistent = %s"
                 % (inv.fixed_euler(), inv.consistent())]
        for c in inv.fixed_components():
            amb = " x ".join("P^%d" % d for d in c["ambient"]) or "a point"
            lines.append("    %-3s  %-20s %d equations, chi = %d%s"
                         % (c["oplane"], amb, len(c["equations"]),
                            c["euler"],
                            "   (lies inside X)" if c["inside_X"] else ""))
        try:
            h = inv.hodge_split()
            lines.append("  h^{1,1} = %d + %d,  h^{2,1} = %d + %d  (+ , -)"
                         % (h["h11"] + h["h21"]))
        except NotImplementedError as e:
            lines.append("  Hodge split: %s" % str(e).split(".")[0])
        try:
            s = self.spectrum()
            lines.append("  %d chiral multiplets, %d vector multiplets"
                         % (s["chiral"], s["vectors"]))
        except NotImplementedError as e:
            lines.append("  spectrum: %s" % str(e).split(".")[0])
        lines.append("  gauge group:        %s" % self.gauge_group())
        lines.append("  not computable here; needs")
        for m in self.missing_for_physical():
            lines.append("     - %s" % m)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Sen's weak coupling limit
# ---------------------------------------------------------------------------


def brane_stack(n, on_o7=False):
    r"""The gauge algebra of a stack of n D7-branes, and its Kodaira type.

    Perturbatively the answer is Chan-Paton bookkeeping: n branes away from
    the orientifold plane give u(n), and n on top of an O7-plane give so(2n),
    the doubling because the image branes coincide with the originals.

    Non-perturbatively the same configuration is a degeneration of the
    elliptic fibration, and the two answers must agree. A stack of n branes
    away from the O7 is a type I_n fibre, which carries su(n). A stack of n on
    the O7 is I_{n-4}^*, which carries so(2(n-4) + 8) = so(2n) -- the shift by
    four being the four branes it takes to cancel the O7 charge locally.

    Returns
    -------
    dict
        ``algebra``, ``kodaira``, ``kodaira_algebra`` and ``agree``. The
        third is read off :func:`pyCICY.theories.ftheory.kodaira_type`, which
        knows nothing about branes.

    Examples
    --------
    >>> r = brane_stack(4, on_o7=True)
    >>> r["algebra"], r["kodaira"], r["agree"]
    ('so(8)', 'I_0*', True)
    >>> brane_stack(5)["kodaira"]
    'I_5'
    """
    n = int(n)
    if n < 1:
        raise ValueError("a stack needs at least one brane")
    if on_o7:
        if n < 4:
            raise ValueError(
                "fewer than four D7-branes on an O7-plane leaves negative "
                "local charge, and the Weierstrass model there is not in "
                "minimal form. Four is the number that cancels the O7 and "
                "gives so(8).")
        alg = "so(%d)" % (2 * n)
        k = FT.kodaira_type(2, 3, 6 + (n - 4))
    else:
        alg = "su(%d)" % n if n > 1 else None
        k = FT.kodaira_type(0, 0, n)
    ka = k["algebra_split"]
    return {"algebra": alg, "kodaira": k["type"], "kodaira_algebra": ka,
            "agree": ka == alg,
            "u1": not on_o7,
            "note": ("perturbatively u(%d); the u(1) is not visible in the "
                     "Kodaira type, which sees only the non-abelian part. A "
                     "single brane has no non-abelian factor at all, and I_1 "
                     "correspondingly carries none." % n)
                    if not on_o7 else ""}


class SenLimit(object):
    r"""The weak coupling limit of F-theory over a base surface.

    Tuning the Weierstrass coefficients to

        f = -3 h^2 + eps eta ,     g = -2 h^3 + eps h eta - eps^2 psi

    and taking eps to zero drives the axio-dilaton to weak coupling almost
    everywhere. The discriminant degenerates to eps^2 h^2 (eta^2 - h psi), a
    double zero on {h = 0} -- the O7-plane -- and a single brane on the rest.
    The physical space is the double cover of B branched over the O7.

    Since ``f`` is a section of -4K_B and ``h^2`` must be too, the O7-plane
    sits in the class -2K_B, and that one fact drives everything below.

    Parameters
    ----------
    base : ftheory.Base or str
        The base surface of the F-theory model.

    Examples
    --------
    >>> s = SenLimit("P2")
    >>> s.o7_class()
    [-6]
    >>> s.branch_genus()
    10
    >>> s.double_cover_euler()
    24
    """

    def __init__(self, base):
        self.base = base if isinstance(base, FT.Base) else FT._named_base(base)

    def __repr__(self):
        return "<Sen limit of F-theory over %s>" % self.base.name

    def o7_class(self):
        """[O7] = -2 K_B, the branch divisor of the double cover."""
        return [-2 * int(k) for k in self.base.K]

    def branch_genus(self):
        r"""The genus of the O7 curve, which is K^2 + 1.

        By adjunction on C = -2K, 2g - 2 = C.(C + K) = (-2K).(-K) = 2K^2.
        On P^2 the O7 is a sextic, of genus 10.
        """
        return self.base.genus(self.o7_class())

    def double_cover_euler(self):
        r"""chi of the double cover of B branched over the O7. Always 24.

        A double cover branched over C has
        chi = 2 chi(B) - chi(C), and for a rational base
        chi(B) = 3 + T while chi(C) = 2 - 2(K^2 + 1) = -2(9 - T), so

            chi = 2(3 + T) + 2(9 - T) = 24

        with the tensor multiplet count cancelling. The Sen limit of *every*
        six-dimensional F-theory model is Type IIB on a K3, whatever the base
        was. That is not put in anywhere; it comes out of K^2 = 9 - T.
        """
        chi_B = self.base.chi_top
        g = self.branch_genus()
        return 2 * chi_B - (2 - 2 * g)

    def double_cover_is_k3(self):
        """Whether the branched cover is a K3, i.e. has chi = 24 and K = 0.

        The canonical class of the double cover branched over C in |2L| is
        the pullback of K_B + L. Here L = -K_B, so it vanishes: the cover is
        Calabi-Yau, and a Calabi-Yau surface with chi = 24 is a K3.
        """
        return self.double_cover_euler() == 24

    def whitney_class(self):
        r"""The class of the D7-brane, -8 K_B.

        With eta a section of -4K and psi of -6K, the brane eta^2 = h psi
        sits in |-8K|. It is a single irreducible brane, not a stack -- the
        "Whitney umbrella" -- so it carries no non-abelian gauge symmetry,
        which is why the generic F-theory model over a smooth base has none
        either.
        """
        return [-8 * int(k) for k in self.base.K]

    def d7_tadpole(self):
        r"""Whether the D7 charge cancels the O7 charge.

        The condition is sum_a N_a([D_a] + [D_a']) = 8 [O7]. The Whitney
        brane is a single brane, so it and its image contribute twice its
        class, 2 * (-8K) = -16K, and 8 [O7] = 8 * (-2K) = -16K. It closes,
        and it closes for every base at once because both sides are multiples
        of the same canonical class.
        """
        left = [2 * c for c in self.whitney_class()]
        right = [8 * c for c in self.o7_class()]
        return {"branes": left, "oplanes": right, "cancels": left == right,
                "condition": "sum_a N_a ([D_a] + [D_a']) = 8 [O7]"}

    def summary(self):
        b = self.base
        return {"base": b.name, "T": b.T, "K2": b.K2,
                "o7_class": self.o7_class(),
                "o7_genus": self.branch_genus(),
                "double_cover_euler": self.double_cover_euler(),
                "is_k3": self.double_cover_is_k3(),
                "whitney_class": self.whitney_class(),
                "d7_tadpole": self.d7_tadpole()["cancels"]}
