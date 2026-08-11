r"""
pyCICY.theories.mtheory -- M-theory compactifications.

The gap this fills
------------------
:mod:`pyCICY.theories` says that M-theory compactifications "would go here
too" and are not implemented. This module implements the three that the
existing machinery already determines, and is explicit that the fourth thing
one would want -- a chiral four-dimensional model -- is not among them.

M-theory has no dilaton and no string coupling, so there is no weak-coupling
expansion to organise the answers. What replaces it is that eleven-dimensional
supergravity plus the M2/M5 brane spectrum is fixed, and the four-dimensional
physics follows from the topology of the internal space alone. Every quantity
below is therefore an integer or a rational number read off from a
configuration matrix, in the same sense as everywhere else in this package,
and the same wall stands in the same place: **the metric is still absent.**

The three cases
---------------
``m-theory-cy3-5d``
    M-theory on a Calabi-Yau threefold gives five dimensions with eight
    supercharges. The vector multiplet moduli space is *cubic*, governed by

        F(t) = (1/6) d_rst t^r t^s t^t ,

    the triple intersection numbers -- which :meth:`pyCICY.CICY.
    triple_intersection` already computes exactly. This is the cleanest case
    in the whole package: the prepotential is not an approximation to
    anything, it is the exact two-derivative action, and the Chern-Simons
    levels are literally the intersection numbers.

``m-theory-g2``
    M-theory on a seven-dimensional space of G_2 holonomy gives four
    dimensions with N=1. There is no CICY of G_2 holonomy, but there is a
    standard construction that reuses one: Joyce's *barely G_2* quotient
    ``(X x S^1)/sigma`` with ``sigma`` an antiholomorphic involution acting on
    the circle by inversion. :class:`~pyCICY.theories.orientifold.
    SignInvolution` already computes the eigenvalue split of the cohomology,
    which is exactly the input the Betti numbers need.

``m-theory-cy4-3d``
    M-theory on a Calabi-Yau fourfold gives three dimensions with N=2, and the
    tadpole ``chi/24`` -- which becomes the D3-brane tadpole of the dual
    F-theory model on the base.

A fourth category, again
------------------------
:mod:`pyCICY.theories.ftheory` introduced quantities that *do not exist*
rather than being unavailable, and both of the first two cases here are in
that category, for two different reasons.

Five-dimensional theories with eight supercharges admit no superpotential, so
:class:`MTheory5D` has no Yukawa couplings at all and says so with
:exc:`~pyCICY.theories.ftheory.NoSuchTheory`, not with
:exc:`~pyCICY.theories.base.NeedsMetric`.

The G_2 case is sharper and is the reason this module is worth writing. On a
*smooth* G_2 manifold the four-dimensional theory has gauge group ``U(1)^b_2``
and ``b_3`` neutral chiral multiplets, and **that is all it can ever have**:
non-abelian gauge symmetry requires codimension-four ADE singularities and
chiral matter requires isolated conical singularities, neither of which a
smooth manifold has. So the absence of chiral matter here is not a limitation
of this package. It is a theorem, and :exc:`NoChiralMatter` states it as one.
The quotient itself is singular along the fixed locus and does get gauge
symmetry from it; what that costs is described in
:meth:`BarelyG2.singular_locus`.

And a bridge
------------
:class:`~pyCICY.theories.orientifold.SenLimit` checks the orientifold rules
against F-theory. The analogous check here is the M-theory/F-theory duality
itself: M-theory on an elliptic threefold is F-theory on the base on a circle,
so the five-dimensional multiplet counts must reproduce the Hodge numbers that
:meth:`~pyCICY.theories.ftheory.FTheory6D.hodge_numbers` derives from anomaly
cancellation. :func:`circle_reduction_of_6d` does that comparison. The two
routes share no code and no reasoning -- one is a spectral sequence on a
configuration matrix, the other is the cancellation of a six-dimensional
gravitational anomaly -- and they agree.

Horava-Witten
-------------
:func:`horava_witten_scales` is the one place where a number with units
appears. Strongly coupled E_8 x E_8 heterotic string theory is M-theory on an
interval, and Witten's observation is that this fixes the embarrassment of the
weakly coupled case, where the measured Newton constant and the measured
unified coupling cannot both be accommodated. The eleventh dimension is the
resolution: it comes out *larger* than the Calabi-Yau, which is the one
qualitative prediction of the construction. That connects to
:mod:`pyCICY.theories.moduli`, where ``alpha_GUT`` was a free parameter, and
it is worth saying plainly that this does not make it less free -- it converts
it into a statement about a radius.
"""

import math

import numpy as np

from .base import NeedsMetric, Theory, register
from .ftheory import NoSuchTheory

__all__ = ["NoChiralMatter", "MTheory5D", "BarelyG2", "MTheoryG2",
           "MTheory3D", "prepotential_coefficients", "prepotential",
           "gauge_couplings", "m5_string_tension", "barely_g2_betti",
           "circle_reduction_of_6d", "horava_witten_scales"]


class NoChiralMatter(NoSuchTheory):
    """Raised when chiral matter is asked of a smooth G_2 compactification.

    A separate exception from :exc:`~pyCICY.theories.base.NeedsMetric` because
    it reports a different fact. ``NeedsMetric`` means the number exists and
    this package cannot reach it. This means the number is zero for a reason
    that no amount of computation will change: the four-dimensional spectrum
    of M-theory on a smooth G_2 manifold is ``b_2`` abelian vector multiplets
    and ``b_3`` neutral chiral multiplets, with no charged states, because the
    charged states come from singularities and there are none.
    """


# ---------------------------------------------------------------------------
# M-theory on a Calabi-Yau threefold: five dimensions, eight supercharges
# ---------------------------------------------------------------------------

def prepotential_coefficients(X):
    """The triple intersection numbers ``d_rst``, as an integer array.

    The prepotential of the five-dimensional vector multiplets is
    ``F = d_rst t^r t^s t^t / 6`` and these are its exact coefficients. They
    are also the Chern-Simons levels: the eleven-dimensional term
    ``C ^ G ^ G`` reduces to ``d_rst A^r ^ F^s ^ F^t``, so the levels of a
    five-dimensional gauge theory obtained this way are intersection numbers
    of the geometry, which is why they are integers.
    """
    d = np.asarray(X.triple_intersection())
    rounded = np.rint(d).astype(int)
    if not np.allclose(d, rounded, atol=1e-6):
        raise ValueError("triple intersection numbers are not integral; "
                         "the configuration is probably not a threefold")
    return rounded


def prepotential(X, t=None):
    """Evaluate ``F(t) = d_rst t^r t^s t^t / 6``, or return the coefficients.

    Parameters
    ----------
    X : CICY
    t : sequence of float, optional
        Kahler moduli. With ``t`` omitted the coefficient array is returned.

    Notes
    -----
    Exact and classical at once, which is unusual. In the type IIA language
    this same cubic term receives worldsheet instanton corrections; in
    M-theory those corrections are absent, because the string coupling that
    would organise them is the radius of a circle that has been decompactified
    away. What is written here is the whole two-derivative vector multiplet
    action, not its leading term.
    """
    d = prepotential_coefficients(X)
    if t is None:
        return d
    t = np.asarray(t, dtype=float)
    if t.shape != (d.shape[0],):
        raise ValueError("expected %d Kahler moduli, got %d"
                         % (d.shape[0], t.size))
    return float(np.einsum("rst,r,s,t->", d, t, t, t)) / 6.0


def gauge_couplings(X, t):
    r"""The matrix ``a_rs = d_rst t^t``, and its signature.

    This is the intersection form of the divisors restricted by ``t``, and it
    is *not* positive definite -- a point worth stating, because the obvious
    guess is that a gauge kinetic matrix ought to be. By the Hodge index
    theorem, for ``t`` inside the Kahler cone ``a_rs`` has signature

        (1, h^{1,1} - 1) ,

    one positive direction and the rest negative. The positive direction is
    the overall volume; the genuine gauge kinetic term for the ``h^{1,1} - 1``
    physical vector multiplets is the *restriction* of ``-a_rs`` to the
    surface of fixed volume, and that restriction is positive definite. So
    Lorentzian signature here is the healthy case, and anything else means
    ``t`` is outside the cone.

    Returns
    -------
    dict
        ``matrix``, ``eigenvalues``, ``volume`` (the prepotential, which is
        the Calabi-Yau volume), ``signature`` as a ``(positive, negative)``
        pair, and ``lorentzian``, the Hodge index condition.

    Notes
    -----
    Lorentzian signature is necessary for ``t`` to be in the Kahler cone but
    not sufficient; the cone is cut out by the effective curves and divisors,
    which this package does not compute. A ``True`` here is evidence, not a
    certificate, and :meth:`MTheory5D.enhancement_note` says what finding the
    actual walls would take.
    """
    d = prepotential_coefficients(X)
    t = np.asarray(t, dtype=float)
    a = np.einsum("rst,t->rs", d, t)
    eig = np.linalg.eigvalsh(a)
    pos = int(np.sum(eig > 1e-9))
    neg = int(np.sum(eig < -1e-9))
    return {"matrix": a,
            "eigenvalues": eig,
            "volume": prepotential(X, t),
            "signature": (pos, neg),
            "lorentzian": pos == 1 and neg == len(eig) - 1}


def m5_string_tension(X, divisor, t):
    r"""Tension of the string from an M5-brane on a divisor.

    An M5-brane wrapping a divisor ``D = D_r J^r`` in a threefold leaves a
    string in five dimensions with tension proportional to the volume of the
    divisor,

        T(D)  =  (1/2) d_rst D^r t^s t^t ,

    an exact quadratic form in the moduli with integer coefficients. Together
    with the M2-brane states from wrapped curves these are the BPS objects of
    the five-dimensional theory, and their charges take values in
    ``H_4`` and ``H_2`` of the threefold, both lattices the configuration
    matrix determines.
    """
    d = prepotential_coefficients(X)
    D = np.asarray(divisor, dtype=float)
    t = np.asarray(t, dtype=float)
    return 0.5 * float(np.einsum("rst,r,s,t->", d, D, t, t))


@register
class MTheory5D(Theory):
    r"""M-theory on a Calabi-Yau threefold: five dimensions, eight supercharges.

    The spectrum is entirely topological. Reducing eleven-dimensional
    supergravity on ``X`` gives

        gravity multiplet     1
        vector multiplets     h^{1,1}(X) - 1
        hypermultiplets       h^{2,1}(X) + 1

    the vector multiplets from ``C_3`` expanded on the ``(1,1)`` forms, minus
    one because the overall volume sits in a hypermultiplet instead; the extra
    hypermultiplet is that universal one. No index theorem is needed and no
    bundle is chosen -- unlike every heterotic model in this package, the
    geometry is the whole input.

    Examples
    --------
    >>> from pyCICY import CICY
    >>> m = MTheory5D(CICY([[4, 5]]))
    >>> m.spectrum()["vector_multiplets"]
    0
    >>> m.spectrum()["hypermultiplets"]
    102
    >>> prepotential(m.X, [1.0])
    0.8333333333333334
    """

    key = "m-theory-cy3-5d"

    def __init__(self, X, name=None):
        Theory.__init__(self, X, name=name)
        if self.X is None:
            raise ValueError("M-theory on a threefold needs a threefold")
        if getattr(self.X, "nfold", 3) != 3:
            raise ValueError("expected a Calabi-Yau threefold, got an "
                             "%d-fold" % self.X.nfold)

    def hodge_numbers(self):
        """``(h^{1,1}, h^{2,1})`` of the threefold."""
        return (int(self.X.h[2]), int(self.X.h[1]))

    def gauge_group(self):
        """``U(1)^{h^{1,1}-1}``, plus the graviphoton.

        Non-abelian enhancement needs the divisors to shrink and a singularity
        to appear, which is a point on the boundary of the Kahler cone rather
        than a different geometry; :meth:`enhancement_note` says what would be
        required to find those points.
        """
        n = self.hodge_numbers()[0] - 1
        if n == 0:
            return "U(1) (graviphoton only)"
        return "U(1)^%d x U(1)_graviphoton" % n

    def spectrum(self):
        """The five-dimensional massless spectrum. Exact, from Hodge numbers."""
        h11, h21 = self.hodge_numbers()
        return {"vector_multiplets": h11 - 1,
                "hypermultiplets": h21 + 1,
                "gravity_multiplets": 1,
                "real_scalars": (h11 - 1) + 4 * (h21 + 1),
                "supercharges": 8}

    def prepotential(self, t=None):
        """See :func:`prepotential`."""
        return prepotential(self.X, t)

    def chern_simons_levels(self):
        """The levels ``c_rst = d_rst``. Integers, exactly."""
        return prepotential_coefficients(self.X)

    def higher_derivative_coefficient(self):
        r"""The coefficient vector ``c_2(X)_r`` of the ``A ^ tr R ^ R`` term.

        The one-loop Chern-Simons term of the five-dimensional theory is fixed
        by the second Chern class, ``c_{2,r}/48``, and
        :meth:`pyCICY.CICY.second_chern` computes it exactly. It is the second
        piece of topological data the five-dimensional action needs and the
        only one beyond the intersection numbers.
        """
        return np.rint(np.asarray(self.X.second_chern())).astype(int)

    def gauge_couplings(self, t):
        """See :func:`gauge_couplings`."""
        return gauge_couplings(self.X, t)

    def holomorphic_yukawa(self, **kw):
        """Always raises: there is no superpotential in five dimensions.

        Eight supercharges forbid one. This is the same category of answer as
        :exc:`~pyCICY.theories.ftheory.NoSuchTheory` in the six-dimensional
        F-theory case and for the same reason -- too much supersymmetry --
        and it is a different statement from the coupling being unavailable.
        The cubic prepotential is *not* a superpotential; it is the exact
        two-derivative action, and :meth:`prepotential` returns it.
        """
        raise NoSuchTheory(
            "a five-dimensional theory with eight supercharges admits no "
            "superpotential, so there are no Yukawa couplings to compute. "
            "This is not a gap: the full two-derivative vector multiplet "
            "action is the cubic prepotential, which prepotential() returns "
            "exactly.")

    def enhancement_note(self):
        """What non-abelian enhancement would need."""
        return [
            "the Kahler cone of X, whose walls are where curves or divisors "
            "shrink; this package computes intersection numbers but not the "
            "cone",
            "the type of the singularity reached at each wall, which fixes "
            "the enhanced ADE algebra",
            "for chiral matter in four dimensions, a further circle and a "
            "flux, or a different construction entirely -- five-dimensional "
            "theories with eight supercharges are never chiral",
        ]

    def describe(self):
        s = self.spectrum()
        h11, h21 = self.hodge_numbers()
        lines = ["%s on %s" % (self.name, self.geometry()),
                 "  Hodge numbers    h^{1,1} = %d, h^{2,1} = %d" % (h11, h21),
                 "  gauge group      %s" % self.gauge_group(),
                 "  vector multiplets %d" % s["vector_multiplets"],
                 "  hypermultiplets   %d" % s["hypermultiplets"],
                 "  prepotential     F = d_rst t^r t^s t^t / 6, exact",
                 "  Chern-Simons     c_rst = d_rst, integral",
                 "  Yukawa couplings none: eight supercharges forbid a "
                 "superpotential"]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# M-theory on a G_2 manifold: four dimensions, N = 1
# ---------------------------------------------------------------------------

def barely_g2_betti(h11_plus, h11_minus, h21):
    r"""Betti numbers of ``(X x S^1)/sigma`` from the eigenvalue split.

    Let ``sigma`` be an antiholomorphic involution of the threefold acting on
    the circle by ``theta -> -theta``. Cohomology of the quotient is the
    invariant part of the cohomology of the cover,

        H^k(Y)  =  H^k(X)^{sigma}  (+)  H^{k-1}(X)^{-sigma} ^ dtheta ,

    ``dtheta`` being odd. With ``H^1(X) = 0`` this gives

        b_2(Y) = h^{1,1}_-,        b_3(Y) = 1 + h^{2,1} + h^{1,1}_+ ,

    where the subscripts are the eigenvalues of the *holomorphic* involution
    ``tau`` with ``sigma = c . tau`` and ``c`` the real structure. The swap is
    not a typo: an antiholomorphic map sends ``J`` to ``-J``, so ``sigma^* =
    -tau^*`` on ``H^{1,1}`` and the ``sigma``-invariant two-forms are the
    ``tau``-anti-invariant ones. That is also the consistency check on the
    construction, since the G_2 form is

        phi = dtheta ^ J + Re Omega

    and ``J`` must be odd to pair with ``dtheta``, which it is, and
    ``Re Omega`` must be even, which requires ``tau^* Omega = +Omega``.

    The odd part of ``H^3(X)`` is fixed independently: an antiholomorphic
    involution is anti-symplectic on ``H^3(X, R)``, so its fixed subspace is
    Lagrangian and has dimension ``h^{2,1} + 1`` whatever ``tau`` is. The two
    derivations agree, and ``b_2 + b_3 = 1 + h^{1,1} + h^{2,1}``.

    Returns
    -------
    dict
        ``b2``, ``b3``, the full ``betti`` list, and ``euler`` (zero, as it
        must be for any seven-manifold).
    """
    b2 = int(h11_minus)
    b3 = 1 + int(h21) + int(h11_plus)
    betti = [1, 0, b2, b3, b3, b2, 0, 1]
    euler = sum((-1) ** k * b for k, b in enumerate(betti))
    return {"b2": b2, "b3": b3, "betti": betti, "euler": euler}


class BarelyG2(object):
    r"""The G_2 orbifold ``(X x S^1)/sigma``, from a sign involution of ``X``.

    Joyce's construction, sometimes called *barely* G_2 because the holonomy
    sits inside ``SU(3) x Z_2`` rather than filling out G_2 -- which has a
    physical consequence, noted in :meth:`holonomy_note`, and does not affect
    the Betti numbers or the four-dimensional supersymmetry.

    Parameters
    ----------
    involution : SignInvolution
        A *holomorphic* involution ``tau`` of the threefold with
        ``tau^* Omega = +Omega``. The antiholomorphic ``sigma`` used in the
        quotient is ``tau`` composed with complex conjugation of the ambient
        coordinates, which requires the defining polynomials to have real
        coefficients -- true of the standard representatives, and the reason
        the sign-flip involutions of :mod:`pyCICY.theories.orientifold` are
        the right input.

    Raises
    ------
    ValueError
        If ``tau^* Omega = -Omega``. Then ``Re Omega`` is odd, the G_2 form is
        not invariant, and the quotient is an orientifold rather than a G_2
        space. The two constructions use the same involutions and are told
        apart by exactly this sign, so the check is worth making loudly.

    Examples
    --------
    >>> from pyCICY.theories.orientifold import SignInvolution
    >>> g = BarelyG2(SignInvolution([[4, 5]], [[3, 4]]))
    >>> g.betti()["b2"], g.betti()["b3"]
    (0, 103)
    """

    def __init__(self, involution, name=None):
        self.involution = involution
        self.name = name or "barely G_2"
        if involution.omega_sign() != 1:
            raise ValueError(
                "this involution has tau^* Omega = -Omega, so Re Omega is "
                "odd under sigma and the G_2 three-form is not invariant. "
                "That is an O3/O7 orientifold involution, not a G_2 one; see "
                "pyCICY.theories.orientifold. A G_2 quotient needs "
                "omega_sign() == +1.")
        try:
            self._split = involution.hodge_split()
        except ValueError as e:
            # A different refusal, and worth not conflating with the sign
            # test above. Some sign involutions force X to contain a whole
            # ambient subspace, so X is not the generic complete intersection
            # its configuration matrix describes and the Hodge numbers on the
            # cover are wrong before any quotient is taken. The G_2 side has
            # nothing to add to that; it just must not paper over it.
            raise ValueError(
                "the eigenvalue split of this involution is unavailable, so "
                "the Betti numbers of the quotient are too: %s" % e)

    def __repr__(self):
        b = self.betti()
        return "<BarelyG2 b2=%d b3=%d>" % (b["b2"], b["b3"])

    def hodge_split(self):
        """The eigenvalue split of the underlying threefold cohomology."""
        return self._split

    def betti(self):
        """Betti numbers of the seven-manifold. See :func:`barely_g2_betti`."""
        h11p, h11m = self._split["h11"]
        h21 = sum(self._split["h21"])
        return barely_g2_betti(h11p, h11m, h21)

    def moduli(self):
        r"""The G_2 moduli: ``b_3`` of them, and they are *real*.

        The metric moduli of a G_2 manifold are deformations of the associative
        three-form, so they live in ``H^3(Y, R)``, and each pairs with a
        period of ``C_3`` to form one complex scalar. There is no split into
        Kahler and complex structure moduli and no special geometry: the
        moduli space is not a product, which is the structural difference
        between four-dimensional M-theory on G_2 and everything else in this
        package.
        """
        b = self.betti()
        return {"real_metric_moduli": b["b3"],
                "chiral_multiplets": b["b3"],
                "vector_multiplets": b["b2"],
                "note": "one complex scalar per three-cycle, from the metric "
                        "modulus and the period of C_3; no Kahler/complex "
                        "structure split exists"}

    def singular_locus(self):
        r"""What sits at the fixed locus, and why it is not computed here.

        The quotient is free only if ``sigma`` has no fixed points, and an
        antiholomorphic involution generically does: its fixed locus is the
        real locus ``L = X^{sigma}``, a special Lagrangian three-cycle. On
        ``X x S^1`` the fixed set of ``(sigma, theta -> -theta)`` is
        ``L x {0, pi}``, of dimension three inside seven, so of codimension
        four, and the group acts as ``-1`` on all four normal directions. That
        is ``R^4 / Z_2``, the ``A_1`` singularity, and it carries an ``SU(2)``
        gauge multiplet.

        So the singular quotient is *better* physics than the smooth case --
        it has non-abelian gauge symmetry. Counting it needs the number of
        connected components of the real locus of ``X``, and that is a
        genuinely different computation from anything in this package: the
        real locus depends on the actual coefficients of the defining
        polynomials, not only on the configuration matrix, and the number of
        components is not a topological invariant of ``X``. Two real forms of
        the same threefold can differ.

        There is still no chiral matter, however many ``A_1`` loci there are.
        Chiral matter needs isolated conical singularities, which this
        construction does not produce.
        """
        return {"model": "R^4/Z_2 along L x {0, pi}, L the real locus of X",
                "gauge_enhancement": "one SU(2) per connected component of L",
                "chiral_matter": False,
                "not_computed": [
                    "the number of connected components of the real locus, "
                    "which depends on the coefficients of the defining "
                    "polynomials and not only on the configuration matrix",
                    "whether the real locus is non-empty at all for a given "
                    "choice of real coefficients",
                ]}

    def holonomy_note(self):
        """Why 'barely'.

        The holonomy of this quotient lies in ``SU(3) x Z_2 < G_2`` rather
        than being all of ``G_2``. The four-dimensional supersymmetry is N=1
        either way, so the spectrum above is unaffected, but the construction
        inherits the complex structure of ``X``, and quantities that a generic
        G_2 manifold would not possess -- a preferred ``J``, a holomorphic
        volume form -- survive on the cover. Membrane instanton counting is
        where the difference is felt.
        """
        return ("holonomy in SU(3) x Z_2 < G_2; N=1 in four dimensions is "
                "unaffected, but the space is not a generic G_2 manifold")


@register
class MTheoryG2(Theory):
    r"""M-theory on a G_2 manifold: four dimensions, N=1.

    Parameters
    ----------
    space : BarelyG2, or SignInvolution, or a ``(b2, b3)`` pair
        The internal seven-manifold. A pair of Betti numbers is accepted so
        that a G_2 manifold from some other construction -- a twisted
        connected sum, say -- can be used without a threefold anywhere.

    Examples
    --------
    >>> from pyCICY.theories.orientifold import SignInvolution
    >>> m = MTheoryG2(SignInvolution([[4, 5]], [[3, 4]]))
    >>> m.spectrum()["chiral_multiplets"]
    103
    >>> m.gauge_group()
    'U(1)^0, i.e. none'
    """

    key = "m-theory-g2"

    def __init__(self, space, name=None):
        Theory.__init__(self, None, name=name)
        if isinstance(space, BarelyG2):
            self.space = space
        elif isinstance(space, (tuple, list)) and len(space) == 2:
            self.space = None
            self._betti = {"b2": int(space[0]), "b3": int(space[1])}
        else:
            self.space = BarelyG2(space)
        if self.space is not None:
            self._betti = self.space.betti()

    def geometry(self):
        if self.space is None:
            return "a G_2 manifold with b_2 = %d, b_3 = %d" % (
                self._betti["b2"], self._betti["b3"])
        return "(X x S^1)/sigma with X = %s" % (
            self.space.involution.M.tolist(),)

    def betti(self):
        return dict(self._betti)

    def gauge_group(self):
        """``U(1)^{b_2}``, from ``C_3`` on the two-cycles.

        Non-abelian factors are absent for a smooth G_2 manifold as a matter
        of principle; see :meth:`~BarelyG2.singular_locus` for what the
        orbifold points add.
        """
        b2 = self._betti["b2"]
        if b2 == 0:
            return "U(1)^0, i.e. none"
        return "U(1)^%d" % b2

    def spectrum(self):
        """The four-dimensional N=1 spectrum. Exact, from Betti numbers.

        ``b_2`` abelian vector multiplets and ``b_3`` chiral multiplets, all
        of them neutral moduli. The chiral index is zero, which is the point
        of :exc:`NoChiralMatter`.
        """
        return {"vector_multiplets": self._betti["b2"],
                "chiral_multiplets": self._betti["b3"],
                "charged_chiral_multiplets": 0,
                "chiral_index": 0,
                "supercharges": 4}

    def chiral_matter(self):
        """Always raises. Smooth G_2 compactifications are non-chiral."""
        raise NoChiralMatter(
            "M-theory on a smooth G_2 manifold has no charged chiral matter. "
            "Non-abelian gauge symmetry requires codimension-four ADE "
            "singularities and chiral matter requires isolated conical "
            "singularities; a smooth manifold has neither, so the answer is "
            "zero as a theorem rather than unavailable as a limitation. See "
            "BarelyG2.singular_locus() for what the orbifold fixed points "
            "supply, which is gauge symmetry but still not chirality.")

    def holomorphic_yukawa(self, **kw):
        """Always raises: there is no charged matter to couple."""
        raise NoChiralMatter(
            "there are no charged chiral multiplets in this compactification, "
            "so there are no Yukawa couplings. The superpotential is "
            "generated by membrane instantons on associative three-cycles and "
            "involves only the neutral moduli; this package does not count "
            "associative cycles.")

    def missing_for_physical(self):
        return [
            "codimension-four ADE singularities, for non-abelian gauge "
            "symmetry",
            "isolated conical singularities of the right type, for chiral "
            "matter -- without them the spectrum is non-chiral as a theorem",
            "the number of associative three-cycles and their volumes, for "
            "the membrane instanton superpotential",
            "the G_2 metric, which no exact method provides and which is "
            "harder than the Calabi-Yau case: there is no Yau theorem for "
            "G_2 holonomy, only Joyce's existence results",
        ]

    def describe(self):
        b = self._betti
        s = self.spectrum()
        lines = ["%s on %s" % (self.name, self.geometry()),
                 "  Betti numbers    b_2 = %d, b_3 = %d" % (b["b2"], b["b3"]),
                 "  gauge group      %s" % self.gauge_group(),
                 "  vector multiplets %d" % s["vector_multiplets"],
                 "  chiral multiplets %d (all neutral moduli)"
                 % s["chiral_multiplets"],
                 "  chiral index      0, and not because it was not computed:",
                 "     a smooth G_2 compactification is non-chiral as a "
                 "theorem"]
        if self.space is not None:
            sl = self.space.singular_locus()
            lines.append("  orbifold locus   %s" % sl["model"])
            lines.append("     -> %s" % sl["gauge_enhancement"])
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# M-theory on a Calabi-Yau fourfold: three dimensions, N = 2
# ---------------------------------------------------------------------------

@register
class MTheory3D(Theory):
    r"""M-theory on a Calabi-Yau fourfold: three dimensions, N=2.

    The interesting quantity is the tadpole. The eleven-dimensional
    Chern-Simons term and its one-loop correction give

        chi(X)/24  =  N_{M2}  +  (1/2) int G ^ G ,

    so the Euler characteristic of the fourfold -- an integer the
    configuration matrix determines -- fixes how many membranes and how much
    four-form flux the vacuum must carry. Two consequences follow with no
    further input, and both are checked by :meth:`tadpole`:

    * ``chi`` must be divisible by 6, since ``G`` is quantised so that
      ``G + c_2(X)/2`` is integral and the flux term is then a multiple of
      ``1/8`` at worst;
    * on an elliptic fourfold this is the D3-brane tadpole of the dual
      four-dimensional F-theory model, which
      :meth:`~pyCICY.theories.ftheory.FTheory4D.d3_tadpole` computes from the
      base. The two agree, being the same integer reached by different routes.
    """

    key = "m-theory-cy4-3d"

    def __init__(self, X, name=None):
        Theory.__init__(self, X, name=name)
        if self.X is None:
            raise ValueError("M-theory on a fourfold needs a fourfold")
        if getattr(self.X, "nfold", 4) != 4:
            raise ValueError("expected a Calabi-Yau fourfold, got an "
                             "%d-fold" % self.X.nfold)

    def euler(self):
        return int(round(self.X.euler_characteristic()))

    def gauge_group(self):
        return "U(1)^%d" % int(self.X.h[3])

    def spectrum(self):
        """The three-dimensional N=2 spectrum, before flux."""
        h11 = int(self.X.h[3])
        h31 = int(self.X.h[1])
        h21 = int(self.X.h[2])
        return {"vector_multiplets": h11,
                "chiral_multiplets": h31 + h21,
                "complex_structure_moduli": h31,
                "supercharges": 4}

    def tadpole(self):
        r"""``chi/24``, and what it must be balanced against."""
        chi = self.euler()
        return {"euler": chi,
                "chi_over_24": chi / 24.0,
                "integral": chi % 24 == 0,
                "half_integral_flux_allowed": chi % 24 != 0,
                "relation": "chi/24 = N_M2 + (1/2) int G ^ G",
                "note": "when chi/24 is not an integer the shortfall is made "
                        "up by half-integral flux, which is allowed exactly "
                        "when c_2(X) is not even -- the Witten quantisation "
                        "condition G + c_2/2 in H^4(X, Z)"}

    def holomorphic_yukawa(self, **kw):
        raise NeedsMetric(
            "the three-dimensional superpotential is generated by M5-branes "
            "wrapping divisors of arithmetic genus one, and depends on the "
            "flux. This package carries no flux data and does not enumerate "
            "the divisors, so there is no coupling to report.",
            missing=self.missing_for_physical())


# ---------------------------------------------------------------------------
# the bridge: M-theory on an elliptic threefold is F-theory on a circle
# ---------------------------------------------------------------------------

def circle_reduction_of_6d(model6d):
    r"""Check the six-dimensional F-theory spectrum against M-theory.

    M-theory on an elliptic Calabi-Yau threefold ``X`` over a base ``B`` is
    F-theory on ``B`` compactified on a circle, in the limit where the fibre
    shrinks. So the five-dimensional multiplet counts must be reproducible in
    two ways, and they use nothing in common:

    * from M-theory, ``n_V = h^{1,1}(X) - 1`` and ``n_H = h^{2,1}(X) + 1``;
    * from F-theory, the six-dimensional spectrum on a circle. Six-dimensional
      vectors give ``rank(G)`` five-dimensional vectors on the Coulomb branch,
      each tensor multiplet gives one more, and the Kaluza-Klein vector gives
      the last, so ``n_V = rank(G) + T + 1``. Hypermultiplets are unaffected
      by the circle, so ``n_H = H``.

    The six-dimensional side is fixed by anomaly cancellation --
    ``H - V + 29 T = 273``, an equation about six-dimensional physics with no
    geometry in it. The threefold Hodge numbers are fixed by the resolution of
    the singular fibres. That these agree is the content of the duality, and
    :meth:`~pyCICY.theories.ftheory.FTheory6D.hodge_numbers` is where this
    package already encodes it; this function makes the five-dimensional
    reading explicit and checks the two against each other.

    Parameters
    ----------
    model6d : FTheory6D

    Returns
    -------
    dict
        The counts from each side and whether they match.
    """
    s = model6d.spectrum()
    h11, h21 = model6d.hodge_numbers()

    n_V_geometry = h11 - 1
    n_H_geometry = h21 + 1

    rank = 0
    from .ftheory import algebra_data
    for alg, _, _ in model6d.gauge:
        rank += algebra_data(alg)["rank"]
    rank += model6d.abelian_rank()
    n_V_anomaly = rank + int(s["T"]) + 1
    n_H_anomaly = float(s["H_neutral"])

    return {"h11": h11, "h21": h21,
            "n_V_from_geometry": n_V_geometry,
            "n_V_from_6d_spectrum": n_V_anomaly,
            "n_H_from_geometry": n_H_geometry,
            "n_H_from_6d_spectrum": n_H_anomaly,
            "rank": rank, "T": int(s["T"]),
            "agrees": (n_V_geometry == n_V_anomaly
                       and abs(n_H_geometry - n_H_anomaly) < 1e-9)}


# ---------------------------------------------------------------------------
# Horava-Witten: the strongly coupled heterotic string
# ---------------------------------------------------------------------------

#: Newton's constant in GeV^-2.
G_NEWTON = 6.70883e-39


def horava_witten_scales(alpha_gut=1.0 / 25.0, m_gut=2.0e16,
                         g_newton=G_NEWTON):
    r"""Solve for the eleven-dimensional scale and the interval length.

    Strongly coupled ``E_8 x E_8`` heterotic string theory is M-theory on
    ``X x S^1/Z_2``, with one ``E_8`` on each end of the interval. Matching
    the four-dimensional couplings to the eleven-dimensional ones gives, in
    Witten's normalisation,

        alpha_GUT  =  (4 pi kappa^2)^{2/3} / (2 V) ,
        G_N        =  kappa^2 / (16 pi^2 rho V) ,

    with ``kappa`` the eleven-dimensional gravitational coupling, ``V`` the
    Calabi-Yau volume and ``pi rho`` the length of the interval. Two equations
    and three unknowns, so one input is needed: the compactification scale,
    taken here as ``V = m_gut^{-6}``.

    Why this is the point of the construction
    -----------------------------------------
    In the *weakly* coupled heterotic string the same two quantities are not
    independent -- both are set by the string coupling and the volume -- and
    the measured values cannot be fitted: the prediction for ``G_N`` comes out
    too large by a factor of roughly twenty, the long-standing discrepancy
    Witten's paper set out to address. At strong coupling ``rho`` is a new
    parameter and the fit is possible. What comes out is the qualitative
    result worth having: ``1/rho`` is *smaller* than every other scale in the
    problem, so the eleventh dimension is larger than the Calabi-Yau and there
    is an energy range in which the world is five-dimensional.

    Returns
    -------
    dict
        ``kappa_squared``, ``m11``, ``inverse_rho``, ``rho_over_v16`` (how
        many times longer the interval is than the Calabi-Yau), and
        ``eleventh_dimension_larger``.

    Notes
    -----
    The O(1) factors in the two matching relations are convention-dependent
    and differ between sources by factors of ``2`` and ``pi``; the *ordering*
    of the scales, which is the physical statement, is not sensitive to them.
    Nothing here is a prediction of ``alpha_GUT``: it is an input, exactly as
    in :mod:`pyCICY.theories.moduli`, and what the construction buys is a
    translation of that free parameter into a radius.
    """
    V = m_gut ** -6.0
    kappa_sq = (2.0 * V * alpha_gut) ** 1.5 / (4.0 * math.pi)
    rho = kappa_sq / (16.0 * math.pi ** 2 * g_newton * V)
    m11 = ((2.0 * math.pi) ** 8 / (2.0 * kappa_sq)) ** (1.0 / 9.0)
    inv_rho = 1.0 / rho
    v16 = V ** (1.0 / 6.0)
    return {"kappa_squared": kappa_sq,
            "m11": m11,
            "inverse_rho": inv_rho,
            "m_gut": m_gut,
            "rho_over_v16": rho / v16,
            "eleventh_dimension_larger": rho > v16,
            "ordering": "1/rho < M_GUT < M_11" if inv_rho < m_gut < m11
                        else "not the expected ordering"}


# ---------------------------------------------------------------------------
# demonstration
# ---------------------------------------------------------------------------

def _demo():
    from ..pyCICY import CICY
    from .orientifold import SignInvolution
    from .ftheory import FTheory6D

    line = "-" * 68

    print(line)
    print("M-theory on a Calabi-Yau threefold: five dimensions")
    print(line)
    for conf, label in ([[4, 5]], "quintic"), ([[2, 2, 1], [3, 1, 3]], "two-parameter model [[2,2,1],[3,1,3]]"):
        m = MTheory5D(CICY(conf), name="M-theory on the %s" % label)
        print(m.describe())
        d = m.chern_simons_levels()
        n = d.shape[0]
        t = np.ones(n)
        g = gauge_couplings(m.X, t)
        print("  at t = (1,...,1):  volume = %.4f, signature of d_rst t^t = "
              "%s, Hodge index: %s"
              % (g["volume"], g["signature"],
                 "ok" if g["lorentzian"] else "t is outside the Kahler cone"))
        print("  c_2 . J_r        = %s" % m.higher_derivative_coefficient())
        try:
            m.holomorphic_yukawa()
        except NoSuchTheory as e:
            print("  Yukawa           %s" % str(e).split(".")[0])
        print()

    print(line)
    print("M-theory on a G_2 manifold: four dimensions, N=1")
    print(line)
    tau = SignInvolution([[4, 5]], [[3, 4]])
    print("involution of the quintic, omega_sign = %+d" % tau.omega_sign())
    g = BarelyG2(tau)
    print("  threefold split  h^{1,1} = %s, h^{2,1} = %s"
          % (g.hodge_split()["h11"], g.hodge_split()["h21"]))
    b = g.betti()
    print("  Betti numbers    %s" % b["betti"])
    print("  Euler character  %d (zero, as any seven-manifold must be)"
          % b["euler"])
    m = MTheoryG2(g, name="M-theory")
    print(m.describe())
    try:
        m.chiral_matter()
    except NoChiralMatter as e:
        print("  chiral matter:   %s." % str(e).split(".")[0])
    print()

    print(line)
    print("The duality check: M-theory on an elliptic threefold")
    print(line)
    for base in ["P2", "F3", "F12"]:
        f = FTheory6D(base)
        c = circle_reduction_of_6d(f)
        print("  base %-4s gauge %-8s  n_V: %2d (geometry) vs %2d (anomaly)"
              "   n_H: %4d vs %4d   %s"
              % (base, f.gauge_group(), c["n_V_from_geometry"],
                 c["n_V_from_6d_spectrum"], c["n_H_from_geometry"],
                 c["n_H_from_6d_spectrum"],
                 "agree" if c["agrees"] else "DISAGREE"))
    print()

    print(line)
    print("Horava-Witten: where the eleventh dimension sits")
    print(line)
    hw = horava_witten_scales()
    print("  M_11             %.3e GeV" % hw["m11"])
    print("  M_GUT            %.3e GeV" % hw["m_gut"])
    print("  1/rho            %.3e GeV" % hw["inverse_rho"])
    print("  interval / CY    %.2f times longer" % hw["rho_over_v16"])
    print("  ordering         %s" % hw["ordering"])
    print()
    print("  alpha_GUT is an input here, not a prediction. What the")
    print("  construction converts it into is a radius, and the radius is")
    print("  the largest length in the problem.")


if __name__ == "__main__":
    _demo()
