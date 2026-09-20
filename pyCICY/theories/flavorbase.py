r"""
pyCICY.theories.flavorbase -- a base class for theories of flavour.

Why a third sibling
-------------------
:class:`~.base.Theory` answers "what four-dimensional physics comes out of this
geometry" and has ``gauge_group``, ``spectrum`` and ``holomorphic_yukawa``.
:class:`~.surface.SurfaceTheory` answers "what is the value of this amplitude"
and has ``leading_singularity``, ``cuts`` and ``soft_limit``.

A theory of flavour asks something different again: *given* the Standard Model
gauge group and matter content, what structure in the Yukawa matrices accounts
for the observed masses, mixings and CP violation? The object is a pair of
3x3 matrices and a pattern of zeros in them, and the verbs are
:meth:`FlavorTheory.texture`, :meth:`FlavorTheory.rephasing_invariant`,
:meth:`FlavorTheory.predicted_angle` and :meth:`FlavorTheory.deviation`.

It was tempting to put this under :class:`~.base.Theory` instead, since for
once its verbs nearly apply -- there is a gauge group and there is a quark
spectrum. The reason not to is ``holomorphic_yukawa``, which in this package
means a cup product of cohomology classes, quasi-topological and exact.
A Yukawa texture is a phenomenological ansatz for the same matrix. Making one
method mean both would blur precisely the distinction the interface exists to
keep, so flavour gets its own base and shares what is genuinely shared: the
registry, so one lookup still reports what any construction here claims, and
the exception hierarchy.

The ledger
----------
Nothing new. :exc:`NeedsFit` is the flavour-side
:exc:`~.base.NeedsMetric` -- the same category, a named missing ingredient --
and it exists because a great deal of what a flavour paper reports is the
output of a numerical fit to experimental data. A fit is not exact arithmetic
and, worse, its *completeness* cannot be certified: a scan finds the minima it
finds. This package will run such a scan when asked, but it reports the box it
searched, in the same way :func:`pyCICY.bundles.scan` does, and it never lets
a truncated result look exhaustive.

What is exact in this corner is the combinatorics -- which textures exist,
which are related by permutations, which loops carry the phase, which
determinants can be made real -- and the algebra of perturbative
diagonalisation. Those need no fit at all, and they are where the modules
under this base do their work.
"""

from .base import NeedsMetric, register, get, registry     # noqa: F401
from .ftheory import NoSuchTheory                          # noqa: F401

__all__ = ["FlavorTheory", "NeedsFit", "NeedsMetric", "NoSuchTheory",
           "register", "get", "registry"]


class NeedsFit(NotImplementedError):
    """Raised for a quantity that requires a numerical fit to data.

    The flavour-side analogue of :exc:`~.base.NeedsMetric`, and the same
    category rather than a new one: the quantity exists, an exact structure is
    in hand, and what is missing is named.

    Two things make a fit different from a calculation and both are worth
    stating rather than hiding. It is numerical, so it is not exact
    arithmetic. And it is a search, so its results depend on which minima were
    found -- a scan that reports "these are the solutions" is really reporting
    "these are the solutions my scan found in this box".

    Carries ``missing``, and where a scan *is* available, ``available``, which
    names the function that will run it and report its own box.
    """

    def __init__(self, message, missing=None, available=None):
        NotImplementedError.__init__(self, message)
        self.missing = list(missing or [])
        self.available = available


class FlavorTheory(object):
    """Base class for a theory of quark flavour.

    There is no ``X`` and no surface. The data is a pattern of non-zero
    entries in the two Yukawa matrices, together with whatever the theory
    assumes about their phases.

    Parameters
    ----------
    name : str, optional
    """

    #: Short identifier, shared with :data:`~.base.registry`.
    key = None

    def __init__(self, name=None):
        self.name = name or self.__class__.__name__

    # -- the verbs ---------------------------------------------------------

    def texture(self):
        """The pattern of non-zero Yukawa entries. Exact, combinatorial."""
        raise NotImplementedError

    def rephasing_invariant(self):
        """The combination of entries whose phase cannot be redefined away.

        The only physical phase in the Lagrangian, and -- for the
        constructions here -- the thing the whole prediction turns on.
        """
        raise NotImplementedError

    def predicted_angle(self, **kw):
        """Which angle of the unitarity triangle the texture fixes, and to what."""
        raise NotImplementedError

    def deviation(self, **kw):
        """The calculable departure from the leading-order value.

        The leading order is where the pattern is visible; the deviation is
        where it becomes a prediction that an experiment can refute.
        """
        raise NotImplementedError

    # -- the ledger --------------------------------------------------------

    def exact_content(self):
        """What this construction computes exactly. Subclasses should override."""
        return []

    def declined(self):
        """What it refuses, and why. Subclasses should override."""
        return []

    def missing_for_ultraviolet(self):
        """What would be needed to derive the texture rather than assume it.

        A texture is an ansatz. Saying so explicitly is the point of this
        list: the pattern of zeros and the discrete phases are inputs here,
        and a theory that produced them would supply these.
        """
        return [
            "a symmetry, or a geography, or an anomalous-dimension structure "
            "that forces the particular entries to vanish rather than merely "
            "allowing it",
            "a mechanism of spontaneous CP violation whose vacuum fixes the "
            "phases to the assumed discrete values",
            "an account of why the surviving entries have the magnitudes they "
            "do, which is the hierarchy problem of flavour and is not "
            "addressed by a texture at all",
        ]

    def ultraviolet_completion(self, **kw):
        """Always raises. See :meth:`missing_for_ultraviolet`."""
        raise NeedsFit(
            "a texture is an ansatz for the Yukawa matrices, not a derivation "
            "of them. This package computes what a given texture implies and "
            "does not supply the theory that would produce it.",
            missing=self.missing_for_ultraviolet())

    # -- reporting ---------------------------------------------------------

    def describe(self):
        """A summary separating what is exact from what is refused."""
        lines = ["%s" % self.name]
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
        lines.append("  the texture is an ansatz; a theory of it would need")
        for m in self.missing_for_ultraviolet():
            lines.append("     - %s" % m)
        return "\n".join(lines)
