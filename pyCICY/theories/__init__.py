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

The Yukawa side is layered by how much of the class each step needs: `yukawa`
decides the texture from dimensions, `representatives` labels the Koszul origin
and rules products out from that label, `cocycles` writes the class down as a
monomial and returns the integer, and `differentials` computes the spectral
sequence for the classes that are not single monomials. None of the four needs
a metric, and none of the four gives a physical coupling.

Type IIA and IIB orientifolds, F-theory and M-theory compactifications would go
here too. They are not implemented.
"""

from .base import Theory, NeedsMetric, registry, register, get
from .heterotic import StandardEmbedding, LineBundleModel
from . import yukawa
from . import representatives
from . import cocycles
from . import differentials

__all__ = ["Theory", "NeedsMetric", "registry", "register", "get",
           "StandardEmbedding", "LineBundleModel", "yukawa",
           "representatives", "cocycles", "differentials"]
