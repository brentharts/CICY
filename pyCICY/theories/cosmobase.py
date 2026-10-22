r"""
pyCICY.theories.cosmobase -- the fourth base class: theories of a cosmological singularity.

`Theory` asks what four-dimensional physics comes out of a geometry. `SurfaceTheory` asks what an amplitude equals.
`FlavorTheory` asks what structure in the Yukawa matrices accounts for the measured masses. A cosmological solution
asks something else again: what does the geometry do at the end of time -- which axes collapse, whether the
curvature blows up, what a light ray can reach -- so it gets its own verbs (`exponents`, `curvature`,
`singularity_type`, `horizon`) and shares only the registry and the exception hierarchy.

The ledger is the same one, and each entry says where it sits:

* exact: Kasner exponents, the BKL map, closed-form curvature, the pancake series and its Kretschmann scalar at
  t = 0, the Kasner horizon rule, the FLRW horizon entropy and its second-law criterion;
* numerical, with the check named: the Mixmaster solution (Bianchi IX integrated in ln V, constraint monitored),
  the sharp-wall BKL map for the horizon sum (calibrated against the integration it replaces);
* does not exist (:exc:`~pyCICY.theories.ftheory.NoSuchTheory`): a finite-curvature *vacuum* pancake, because the
  only vacuum exponent set with one collapsing axis and two finite ones, (1, 0, 0), is flat spacetime;
* needs data or a fit (:exc:`~pyCICY.theories.flavorbase.NeedsFit`): any statement that a toy cosmology matches
  observations. The toys here are our own, not fitted, and not reproductions of the models that inspired them.

What is deliberately not here is the Jaynes-Cummings sensor and "complexity scalar" of Quantum Gravity Lab, the
project these modules were ported from. They map K(t) to a number through a stated modelling choice; that is a
reasonable experiment and not an exact result, and this package's discipline is to keep the two apart.
"""

from .base import register
from .ftheory import NoSuchTheory
from .flavorbase import NeedsFit

__all__ = ["CosmologyTheory", "NoSuchTheory", "NeedsFit"]


class CosmologyTheory(object):
    """Base class for a cosmological solution or toy, with no compactification geometry attached."""

    #: Short identifier used by :func:`register` and :func:`get`.
    key = None

    def __init__(self, name=None):
        self.name = name or self.__class__.__name__

    def geometry(self):
        """A label for the spacetime, used by :meth:`describe`."""
        raise NotImplementedError

    # -- verbs ---------------------------------------------------------------

    def exponents(self, **kw):
        """Kasner-type exponents of the scale factors at the singularity, where defined."""
        raise NotImplementedError

    def curvature(self, **kw):
        """Kretschmann scalar (or the best exact statement about it)."""
        raise NotImplementedError

    def singularity_type(self, **kw):
        """Morphology of the singularity: point, cigar, barrel, pancake or none."""
        raise NotImplementedError

    def horizon(self, **kw):
        """What a null ray can reach: the particle-horizon statement for this solution."""
        raise NotImplementedError

    # -- ledger -------------------------------------------------------------

    def exact_content(self):
        return []

    def declined(self):
        return []

    def missing_for_observation(self):
        """What a comparison with data would need. Default: all of it."""
        return ["a likelihood and the data it is evaluated on (SNe Ia, BAO, CMB distances)",
                "a fit with its parameter box stated: a scan reports the minima it found in the box it searched"]

    def physical_prediction(self, **kw):
        """Always raises: no cosmology here is compared with observations."""
        raise NeedsFit(
            "no cosmology in this package is fitted to data. The solutions are exact; a statement that one of "
            "them describes the observed universe needs a likelihood, data and a stated parameter box. See "
            "missing_for_observation().", missing=self.missing_for_observation())

    def describe(self):
        lines = ["%s on %s" % (self.name, self.geometry()), "  exact:"]
        lines += ["     - %s" % e for e in self.exact_content()]
        lines.append("  declined:")
        lines += ["     - %s" % d for d in self.declined()]
        return "\n".join(lines)
