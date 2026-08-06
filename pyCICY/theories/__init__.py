"""
pyCICY.theories -- string constructions on a Calabi-Yau threefold.

Each module here is one way of getting four-dimensional physics out of the same
geometry, sharing the exact machinery underneath and declaring for itself what
it can compute. See :mod:`pyCICY.theories.base` for the interface and for the
one distinction it is built around: holomorphic quantities are exact, physical
quantities need the Ricci-flat metric.

Currently implemented:

    heterotic-standard-embedding   V = TX, E_6, Yukawa couplings exact
    heterotic-line-bundle          V a sum of line bundles, SU(5) and the
                                   Standard Model, spectrum exact

Type IIA and IIB orientifolds, F-theory and M-theory compactifications would go
here too. They are not implemented.
"""

from .base import Theory, NeedsMetric, registry, register, get
from .heterotic import StandardEmbedding, LineBundleModel
from . import yukawa

__all__ = ["Theory", "NeedsMetric", "registry", "register", "get",
           "StandardEmbedding", "LineBundleModel", "yukawa"]
