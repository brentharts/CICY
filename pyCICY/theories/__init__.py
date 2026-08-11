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
    f-theory-6d                    elliptic threefold over a surface, six
                                   dimensional (1,0), spectrum exact from
                                   anomaly cancellation
    f-theory-4d                    elliptic fourfold, four dimensional N=1,
                                   D3 tadpole exact and spectrum flux
                                   dependent
    type-iib-orientifold           a holomorphic involution of X, O3/O7 or
                                   O5/O9, equivariant Hodge numbers and the
                                   closed string spectrum exact
    m-theory-cy3-5d                five dimensions with eight supercharges,
                                   spectrum and cubic prepotential exact
    m-theory-g2                    four dimensions N=1 on a barely G_2
                                   quotient, Betti numbers and spectrum exact
    m-theory-cy4-3d                three dimensions N=2, chi/24 tadpole exact

The Yukawa side is layered by how much of the class each step needs: `yukawa`
decides the texture from dimensions, `representatives` labels the Koszul origin
and rules products out from that label, `cocycles` writes the class down as a
monomial and returns the integer, and `differentials` computes the spectral
sequence for the classes that are not single monomials. None of the four needs
a metric, and none of the four gives a physical coupling.

`ftheory` adds a third category to the two the interface was built around.
Beside quantities that are exact and quantities that need a metric there are
quantities that do not exist: six-dimensional (1,0) supersymmetry forbids a
superpotential, so an F-theory compactification to six dimensions has no
Yukawa couplings at all, and :exc:`~pyCICY.theories.ftheory.NoSuchTheory` says
that rather than returning zero or blaming the metric.

`orientifold` reaches the same physics from the other side, and the two meet:
:class:`~pyCICY.theories.orientifold.SenLimit` takes an F-theory base to its
weak coupling limit, where the D7-brane rules of the orientifold have to
reproduce the Kodaira fibre types of the fibration. They do.

`mtheory` is the third module to need a category beyond exact and
needs-a-metric, and it needs two. A five-dimensional theory with eight
supercharges has no superpotential, so :class:`~pyCICY.theories.mtheory.
MTheory5D` raises :exc:`~pyCICY.theories.ftheory.NoSuchTheory` exactly as the
six-dimensional F-theory case does. And M-theory on a *smooth* G_2 manifold
has no charged chiral matter at all -- gauge symmetry needs codimension-four
ADE singularities, chirality needs isolated conical ones --- so the zero that
:exc:`~pyCICY.theories.mtheory.NoChiralMatter` reports is a theorem rather
than a gap.

It meets `ftheory` from the other side too, and this is the second place two
constructions in this package check each other: M-theory on an elliptic
threefold is F-theory on the base on a circle, so the five-dimensional
multiplet counts read off the Hodge numbers must equal the ones read off the
six-dimensional anomaly conditions. :func:`~pyCICY.theories.mtheory.
circle_reduction_of_6d` compares them. They agree.

Type IIA orientifolds would go here too. They are not implemented.
"""

from .base import Theory, NeedsMetric, registry, register, get
from .heterotic import StandardEmbedding, LineBundleModel
from .ftheory import Base, FTheory6D, FTheory4D, NoSuchTheory
from .orientifold import Orientifold, SignInvolution, SenLimit
from .mtheory import (MTheory5D, MTheoryG2, MTheory3D, BarelyG2,
                      NoChiralMatter, circle_reduction_of_6d,
                      horava_witten_scales)
from . import yukawa
from . import representatives
from . import cocycles
from . import differentials
from . import ftheory
from . import orientifold
from . import mtheory

__all__ = ["Theory", "NeedsMetric", "registry", "register", "get",
           "StandardEmbedding", "LineBundleModel", "yukawa",
           "representatives", "cocycles", "differentials",
           "Base", "FTheory6D", "FTheory4D", "NoSuchTheory", "ftheory",
           "Orientifold", "SignInvolution", "SenLimit", "orientifold",
           "MTheory5D", "MTheoryG2", "MTheory3D", "BarelyG2",
           "NoChiralMatter", "circle_reduction_of_6d",
           "horava_witten_scales", "mtheory"]
