r"""
pyCICY.theories.surface -- a base class for amplitudes, beside the one for
compactifications.

Why a sibling rather than a subclass
------------------------------------
Every construction in this subpackage so far answers one question: what
four-dimensional physics comes out of this geometry. :class:`~.base.Theory`
therefore has ``gauge_group``, ``spectrum`` and ``holomorphic_yukawa``.

None of those three means anything for a leading singularity.
:class:`~.nariai.NariaiEntropic` already stretched the interface by setting
``X = None`` and keeping only the ledger, and the subpackage docstring says as
much. Three more classes for which the interface's own verbs are meaningless
would be evidence that the interface is wrong, not that the physics is
unusual.

So amplitudes get their own base, with its own verbs -- :meth:`leading_singularity`,
:meth:`cuts`, :meth:`soft_limit` -- and share with :class:`~.base.Theory` the
things that are genuinely common: :data:`~.base.registry`, so that one lookup
still tells you what any construction in this package claims;
:exc:`~.base.NeedsMetric`, for the rare amplitude quantity that really does
want a metric; and :exc:`~.ftheory.NoSuchTheory`, for a quantity that does not
exist rather than one not computed.

The shared machinery underneath is graphs, curves on surfaces and tropical
fans -- not intersection numbers and cohomology. That is the honest reason
for the split.

The ledger, now with four entries
---------------------------------
The package has always separated three things: exact; *needs the Ricci-flat
metric* (:exc:`~.base.NeedsMetric`); *does not exist*
(:exc:`~.ftheory.NoSuchTheory`, :exc:`~.mtheory.NoChiralMatter`,
:exc:`~.nariai.TypeIIIFactor`).

Amplitudes force a fourth, :exc:`NotAnalytic`. A contour prescription is not
unavailable, not metric-dependent and not non-existent: it is *true by
argument*. That an integration contour renders a finite integral everywhere
in kinematic space is a theorem, and a theorem is not the kind of thing this
package computes. It can build the contour, count its pieces, and evaluate an
example -- but an example is not the statement, and the difference is exactly
the one this ledger exists to keep.

:exc:`NeedsIntegration` is *not* a fifth category. It is the amplitude-side
:exc:`~.base.NeedsMetric`: a missing ingredient, named, with the list of what
would supply it.
"""

from .base import NeedsMetric, register, get, registry     # noqa: F401
from .ftheory import NoSuchTheory                          # noqa: F401

__all__ = ["SurfaceTheory", "NotAnalytic", "NeedsIntegration",
           "NeedsMetric", "NoSuchTheory", "register", "get", "registry"]


class NotAnalytic(NotImplementedError):
    """Raised for a statement that is true by argument, not by arithmetic.

    The fourth entry in the ledger. Convergence of a contour, finiteness of a
    remainder, validity of an expansion outside the region it was derived in:
    each is proved in the literature and checkable here only in examples.
    Returning an example and calling it the statement would be the same
    mistake as returning a plausible float for a physical Yukawa coupling.

    Carries ``checkable``, the list of things this package *can* do about the
    statement, so a refusal still tells the caller where to go next.
    """

    def __init__(self, message, checkable=None):
        NotImplementedError.__init__(self, message)
        self.checkable = list(checkable or [])


class NeedsIntegration(NotImplementedError):
    """Raised for a quantity that exists but requires integration or resummation.

    The amplitude-side analogue of :exc:`~.base.NeedsMetric`, and the same
    category, not a new one: an exact integrand is in hand and the missing
    ingredient is named. Loop integration, phase-space integration, infrared
    subtraction and resummation of a perturbative series all land here.

    Carries ``missing``, which is also the specification for closing the gap.
    """

    def __init__(self, message, missing=None):
        NotImplementedError.__init__(self, message)
        self.missing = list(missing or [])


class SurfaceTheory(object):
    """Base class for an amplitude computed from curves on a surface.

    There is no ``X``. The data is a graph or a surface -- a fatgraph, a
    triangulated disk, a punctured disk -- and subclasses describe it through
    :meth:`surface` rather than through a configuration matrix.

    Parameters
    ----------
    surface : str, optional
        A label for the surface or graph the amplitude lives on.
    name : str, optional
    """

    #: Short identifier, shared with :data:`~.base.registry`.
    key = None

    def __init__(self, surface=None, name=None):
        self._surface = surface
        self.name = name or self.__class__.__name__

    def surface(self):
        """A label for the surface the curves live on."""
        return self._surface or "a surface this subclass has not described"

    # -- the verbs ---------------------------------------------------------

    def leading_singularity(self, **kw):
        """The maximal residue: a polynomial in kinematic variables.

        Exact where implemented -- it is a counting problem, not an integral.
        """
        raise NotImplementedError

    def cuts(self, **kw):
        """Residues of the integrand on a channel going on shell."""
        raise NotImplementedError

    def soft_limit(self, **kw):
        """The behaviour of the integrand when some momenta go soft."""
        raise NotImplementedError

    # -- the ledger --------------------------------------------------------

    def exact_content(self):
        """What this construction computes exactly. Subclasses should override."""
        return []

    def declined(self):
        """What it refuses, and why. Subclasses should override."""
        return []

    def missing_for_observable(self):
        """Ingredients needed to turn an exact integrand into a measurement.

        The default list is the honest one for every amplitude module here,
        and it is a specification rather than an apology.
        """
        return [
            "loop integration of the integrand, which this package does not "
            "perform at any precision",
            "an infrared subtraction, since the loop integral diverges and "
            "the divergence cancels only against real emission",
            "phase-space integration against the measurement's cuts",
            "for a hadronic process, parton distributions, which are not "
            "perturbative at all",
        ]

    def cross_section(self, **kw):
        """Always raises. See :meth:`missing_for_observable`."""
        raise NeedsIntegration(
            "a cross-section is an integrated, infrared-subtracted, "
            "phase-space-averaged object. This package computes exact "
            "integrands and residues and performs none of those three "
            "integrations. See missing_for_observable() for what would be "
            "required.",
            missing=self.missing_for_observable())

    # -- reporting ---------------------------------------------------------

    def describe(self):
        """A summary separating what is exact from what is refused."""
        lines = ["%s on %s" % (self.name, self.surface())]
        exact = self.exact_content()
        if exact:
            lines.append("  exact:")
            for e in exact:
                lines.append("     - %s" % e)
        declined = self.declined()
        if declined:
            lines.append("  declined:")
            for d in declined:
                lines.append("     - %s" % d)
        lines.append("  no cross-section here; that would need")
        for m in self.missing_for_observable():
            lines.append("     - %s" % m)
        return "\n".join(lines)
