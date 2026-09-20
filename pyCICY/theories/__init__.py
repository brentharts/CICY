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
    nariai-entropic                entropic gravity on the Nariai horizon:
                                   relative entropy, Clausius relation, and
                                   the Einstein coupling 8pi exact; the von
                                   Neumann entropy does not exist (type III)
    spectre-substrate              the chiral Spectre monotile phase proposed
                                   for the conformal crossover surface:
                                   substitution spectrum, census, charges and
                                   tile geometry exact in Q(sqrt15) and
                                   Q(sqrt3); the crossover itself is a
                                   conjecture and is not tested
    crossover-parity               the observational layer of the same
                                   proposal: the rotation algebra exact, the
                                   log-periodic band-power search a stated
                                   modeling step, the chirality of the tensor
                                   sector not yet observable at all
    gluon-leading-singularity      maximal residues of pure-gluon amplitudes
                                   as a covering problem on a fatgraph: the
                                   three-point vertex, the extension balance,
                                   the closed-curve exponent and the n-gon
                                   coefficients exact; cuts need a phase-space
                                   integration and fermion loops have no
                                   settled sign rule
    soft-factorisation             the infrared structure of a Feynman
                                   integrand in Schwinger space: the Symanzik
                                   polynomials from the graph Laplacian,
                                   the divergent rays, the worldline
                                   variables and the hard-soft factorisation
                                   exact; the finiteness of the subtracted
                                   remainder is a theorem and the soft
                                   anomalous dimension a resummed series
    cuts-and-contours              unitarity cuts of a one-loop surface
                                   integral: the residues at any mass level
                                   and the piece count of the Pochhammer
                                   contour exact, including the level at
                                   which a candidate stringy integral stops
                                   being unitary; convergence of the contour
                                   is a theorem and is declined

Not every construction here is a compactification any more, and one of them is
not even shaped like `Theory`. `surface` adds a sibling base class,
:class:`~pyCICY.theories.surface.SurfaceTheory`, for amplitudes: same
registry, same exception hierarchy, different verbs
(`leading_singularity`, `cuts`, `soft_limit`), because `gauge_group` and
`spectrum` mean nothing for a maximal residue. It also adds the ledger's
fourth entry, :exc:`~pyCICY.theories.surface.NotAnalytic`, for a statement
that is true by argument rather than by arithmetic -- the convergence of a
contour, say -- which is neither exact, nor waiting on a metric, nor
non-existent. `surfaceology` is the machinery underneath: curves on a disk,
u-variables from F-polynomials, and the u-equations as the independent check
that the parametrisation is the right one. `softgraph` is the other half:
graphs rather than surfaces, with U and F built from the Laplacian and checked
against spanning trees and spanning 2-forests. The two amplitude modules also
check each other in the small way that matters, by declining each other's
verbs: a leading singularity is a covering problem and a soft limit is a
tropical one, and neither pretends to the other. `contours` is the third, and
the only construction in this package whose headline result is a failure: a
candidate stringy completion that satisfies unitarity at six mass levels and
cannot satisfy it at the seventh. Reproducing a contradiction is a stronger
test of an implementation than reproducing a value.

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

`spectre` and `parity` are the two halves of one proposal and are the third
pair in the package that checks itself: `spectre` computes the substitution's
spectral theory exactly, including the log-period log(4 + sqrt15) that has no
free parameter in it, and `parity` takes that same number to real band powers
and asks whether it is there. The frequency the second searches at is not
fitted; it is the first one's output.

Beside the constructions are the modules that take an exact spectrum
downstream: `running` to the beta coefficients and the QCD scale, `couplings`
to unification and what can honestly be said about 137, `moduli` to the
racetrack dilaton, and `etg_foreground` to the foreground that would sit
underneath the parity verdicts if it were real.

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
from .nariai import NariaiEntropic, TypeIIIFactor
from . import nariai
from .spectre import SpectreSubstrate
from . import spectre
from .parity import CrossoverParityProbe
from . import parity
from . import moduli
from . import running
from . import couplings
from . import etg_foreground
from .surface import SurfaceTheory, NotAnalytic, NeedsIntegration
from . import surface
from .gluons import GluonLeadingSingularity
from . import gluons
from .softgraph import SoftFactorisation
from . import softgraph
from .contours import CutsAndContours
from . import contours
from . import surfaceology

__all__ = ["Theory", "NeedsMetric", "registry", "register", "get",
           "StandardEmbedding", "LineBundleModel", "yukawa",
           "representatives", "cocycles", "differentials",
           "Base", "FTheory6D", "FTheory4D", "NoSuchTheory", "ftheory",
           "Orientifold", "SignInvolution", "SenLimit", "orientifold",
           "MTheory5D", "MTheoryG2", "MTheory3D", "BarelyG2",
           "NoChiralMatter", "circle_reduction_of_6d",
           "horava_witten_scales", "mtheory",
           "NariaiEntropic", "TypeIIIFactor", "nariai",
           "SpectreSubstrate", "spectre",
           "CrossoverParityProbe", "parity",
           "moduli", "running", "couplings", "etg_foreground",
           "SurfaceTheory", "NotAnalytic", "NeedsIntegration", "surface",
           "GluonLeadingSingularity", "gluons", "surfaceology",
           "SoftFactorisation", "softgraph",
           "CutsAndContours", "contours"]
