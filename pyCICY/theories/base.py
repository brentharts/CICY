r"""
pyCICY.theories.base -- a common interface for string constructions.

Why a subpackage
----------------
Everything in this package so far is one construction: heterotic E_8 x E_8 with
a holomorphic bundle on a Calabi-Yau threefold. That is not the only way to get
four-dimensional physics out of a Calabi-Yau, and the modules underneath --
intersection numbers, cohomology, group actions, enumerative invariants -- are
not specific to it. This subpackage is where other constructions go, sharing
that machinery and declaring for themselves what they can and cannot compute.

The interface is deliberately shaped around one distinction, because it is the
distinction that matters and the one most easily blurred:

**Holomorphic quantities are exact. Physical quantities need a metric.**

A holomorphic Yukawa coupling is a cup product of cohomology classes landing in
H^3(X, O) = C. It is quasi-topological: it depends on the complex structure and
the bundle, not on the Kahler metric, and for the cases below it is an integer
or a power series with integer coefficients. The *physical* Yukawa coupling is
that number divided by the norms of the fields, and those norms are integrals
of harmonic representatives against the Ricci-flat metric. Nothing in this
package computes them, and :exc:`NeedsMetric` says so rather than returning a
plausible float.

Adding a theory
---------------
Subclass :class:`Theory` and implement :meth:`spectrum` and
:meth:`holomorphic_yukawa`. Leave :meth:`physical_yukawa` alone unless the
construction genuinely determines it; the base class raises with an
explanation, which is the correct behaviour for every construction currently
here and probably for most that follow.

:meth:`missing_for_physical` is the other thing worth implementing. It returns
the list of ingredients a physical coupling would need, so that a caller can
report what is missing rather than discovering it as an exception. That list is
also the specification for whoever wants to close the gap.
"""

import numpy as np

__all__ = ["Theory", "NeedsMetric", "registry", "register", "get"]


class NeedsMetric(NotImplementedError):
    """Raised when a quantity requires the Ricci-flat metric.

    An exception rather than a sentinel, for the same reason
    :exc:`pyCICY.phenomenology.MassRatioNotComputable` is: a function that
    returned ``None`` or a placeholder would sooner or later be tabulated next
    to a measured constant, and nothing in the table would distinguish a
    computed number from an invented one.

    Carries ``missing``, the list of ingredients that would be needed, so the
    caller can report the gap precisely.
    """

    def __init__(self, message, missing=None):
        NotImplementedError.__init__(self, message)
        self.missing = list(missing or [])


class Theory(object):
    """Base class for a string construction on a Calabi-Yau threefold.

    Parameters
    ----------
    X : CICY or configuration matrix
    name : str, optional
    """

    #: Short identifier used by :func:`register` and :func:`get`.
    key = None

    def __init__(self, X, name=None):
        from ..pyCICY import CICY
        if X is None:
            # Not every construction compactifies on a CICY. The generic
            # F-theory background is a hypersurface in a weighted projective
            # bundle, which is not a complete intersection in a product of
            # projective spaces; the physics is still computed from the same
            # interface, so the geometry is allowed to be absent and named by
            # geometry() instead.
            self.X = None
        else:
            self.X = X if isinstance(X, CICY) else CICY(
                np.asarray(X, dtype=int).tolist())
        self.name = name or self.__class__.__name__

    def geometry(self):
        """A label for the compactification geometry, used by :meth:`describe`."""
        if self.X is None:
            return "a geometry that is not a CICY"
        return self.X.M.tolist()

    # -- what the construction gives --------------------------------------

    def gauge_group(self):
        """The four-dimensional gauge group, as a string."""
        raise NotImplementedError

    def spectrum(self):
        """The massless chiral spectrum. Exact, from index theory."""
        raise NotImplementedError

    def holomorphic_yukawa(self, **kw):
        """The holomorphic Yukawa couplings. Exact where implemented."""
        raise NotImplementedError

    # -- what it does not give --------------------------------------------

    def missing_for_physical(self):
        """Ingredients needed to turn holomorphic couplings into physical ones.

        The default list is the honest one for every construction here. It is
        also a specification: supply these and the physical coupling follows.
        """
        return [
            "the Ricci-flat metric on X, which no exact method provides and "
            "which numerical packages (cymetric, cymyc) approximate",
            "harmonic representatives of the cohomology classes carrying the "
            "matter fields, not merely their dimensions",
            "the Kahler potential for the matter field kinetic terms, from "
            "which the field normalisations follow",
            "for a quotient model, all of the above equivariantly, on X/Gamma "
            "rather than on X",
        ]

    def physical_yukawa(self, **kw):
        """Always raises. See :meth:`missing_for_physical`.

        A physical Yukawa coupling is the holomorphic one divided by the norms
        of the three fields, and those norms are integrals of harmonic forms
        against the Ricci-flat metric. The holomorphic factor is exact; the
        normalisation is not available here at any precision, not even a poor
        one, so no number is returned.
        """
        raise NeedsMetric(
            "the physical Yukawa coupling is the holomorphic one divided by "
            "the field normalisations, which are integrals against the "
            "Ricci-flat metric. This package computes the holomorphic factor "
            "exactly and the normalisation not at all. See "
            "missing_for_physical() for what would be required.",
            missing=self.missing_for_physical())

    def fermion_masses(self, **kw):
        """Always raises. Masses need the couplings *and* the vevs."""
        raise NeedsMetric(
            "a fermion mass is a physical Yukawa coupling times a Higgs "
            "vacuum expectation value. Neither factor is available: the "
            "coupling needs the metric, and the vev needs moduli "
            "stabilisation, which is a separate problem this package does not "
            "address.",
            missing=self.missing_for_physical()
            + ["moduli stabilisation, to fix the Higgs vacuum expectation "
               "value"])

    # -- reporting ---------------------------------------------------------

    def describe(self):
        """A summary separating what is exact from what is not."""
        lines = ["%s on %s" % (self.name, self.geometry()),
                 "  gauge group      %s" % self.gauge_group()]
        try:
            for k, v in sorted(self.spectrum().items()):
                lines.append("  %-16s %s" % (k, v))
        except NotImplementedError:
            lines.append("  spectrum         not implemented")
        try:
            y = self.holomorphic_yukawa()
            lines.append("  holomorphic Yukawa: %s" % (y.get("summary", y),))
        except (NotImplementedError, NeedsMetric) as e:
            lines.append("  holomorphic Yukawa: %s" % str(e).split(".")[0])
        lines.append("  physical Yukawa:    not computable here; needs")
        for m in self.missing_for_physical():
            lines.append("     - %s" % m)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# a small registry, so that adding a theory is one line
# ---------------------------------------------------------------------------

registry = {}


def register(cls):
    """Class decorator adding a theory to :data:`registry` under its ``key``."""
    if not getattr(cls, "key", None):
        raise ValueError("a registered theory needs a key")
    registry[cls.key] = cls
    return cls


def get(key):
    """Look up a registered theory by key."""
    if key not in registry:
        raise KeyError(
            "no theory registered as %r; available: %s"
            % (key, sorted(registry)))
    return registry[key]
