r"""
pyCICY.theories.contours -- unitarity cuts of a surface integral, and the
combinatorics of its integration contour.

The third of the amplitude modules, and the one whose sharpest result is a
*failure*. There are many surface integrals that reduce to the same field
theory at low energies and differ only in how they behave in the ultraviolet.
Figueiredo and Skowronek show that unitarity on massive thresholds is a
stringent test of that freedom: read the residues of the loop integrand
directly, match them against three-point couplings fixed at tree level, and
most of the candidates die.

Reproducing a contradiction is a better test of an implementation than
reproducing a value. A wrong residue routine will usually still produce *some*
polynomial; it will not usually produce a system of constraints that is
satisfiable at four levels and inconsistent at the fifth.

What this module computes
-------------------------
Exact, and the centre of the module: leading singularities of the one-loop
two-point integrand as residues in the positive coordinates. Because a curve
with ``q`` self-intersections first contributes at order ``y**q``, each massive
threshold is a *finite* residue computation rather than a truncation of an
infinite product, which is what makes any of this arithmetic rather than
analysis.

:func:`dhat_leading_singularity` does this for the ``Dhat`` integral, and
:func:`unitarity_constraints` runs the matching: levels (1,0) and (2,0) force
the two open-curve exponents to vanish, level (1,1) then forces ``d = 4``, and
with those four constraints in hand level (2,1) cannot be matched by anything.
The system is inconsistent, and the function returns *how* it is inconsistent
rather than a boolean.

:func:`baby_leading_singularity` does the same for the truncated "baby"
integrals, which fail differently and more cheaply: three different mass levels
all return ``-Delta1 - Delta2``, and the tree-side values at those levels are
not equal to each other.

Also exact, and much simpler: the combinatorics of the generalised Pochhammer
contour. The number of sheets, tubes and tori is a count of non-crossing chord
sets, which is to say of faces of the associahedron, and
:func:`associahedron_faces` computes it. The top face count is the Catalan
number, which the tests check.

What it refuses
---------------
That a contour renders the integral finite everywhere in kinematic space is a
theorem about convergence, not an arithmetic fact, and
:exc:`~.surface.NotAnalytic` says so -- the module can build the contour and
count its pieces and evaluate an example, and an example is not the statement.
That is the fourth ledger entry doing exactly the work it was added for.

Numerical evaluation at large kinematics gets the same refusal for a different
reason: the answer there is a fine cancellation between two exponentially large
contributions, so finite precision is the binding constraint and no amount of
care in this package would change that.

Reference
---------
Figueiredo and Skowronek, *Cuts and contours*, JHEP 12 (2025) 024,
arXiv:2506.05456. Equation and table numbers below are that paper's.
"""

from itertools import combinations

import sympy as sp

from .surface import SurfaceTheory, NotAnalytic, NeedsIntegration, register
from .surfaceology import chords, crossing

__all__ = ["CutsAndContours", "residue2", "dhat_integrand",
           "dhat_leading_singularity", "closed_curve_factor",
           "baby_leading_singularity", "tachyon_level_one_residue",
           "unitarity_constraints", "associahedron_faces",
           "pochhammer_pieces", "minimal_cutoff", "PAPER_TABLE_1"]

Y1, Y2 = sp.symbols("y_1_0 y_2_0", positive=True)
X11, X22, X12 = sp.symbols("X11 X22 X12")
DELTA1, DELTA2 = sp.symbols("Delta1 Delta2")
DIM = sp.Symbol("d")
ALPHA0 = sp.Symbol("alpha0")


# ---------------------------------------------------------------------------
# residues in two variables
# ---------------------------------------------------------------------------

def residue2(expr, n1, n2, y1=Y1, y2=Y2):
    """The coefficient of ``y1**n1 y2**n2``, exactly.

    A leading singularity at mass levels ``(n1, n2)`` is the residue of the
    integrand at ``y1 = y2 = 0`` with poles of those orders, which is this
    coefficient. Exponents stay symbolic throughout -- the expansion is the
    binomial series in the positive coordinates, not a numerical one.
    """
    s = sp.series(expr, y2, 0, n2 + 1).removeO()
    s = sp.series(sp.expand(s), y1, 0, n1 + 1).removeO()
    return sp.simplify(sp.expand(s).coeff(y1, n1).coeff(y2, n2))


# ---------------------------------------------------------------------------
# the Dhat integrand
# ---------------------------------------------------------------------------

def dhat_integrand(n1, n2, x11=X11, x22=X22, y1=Y1, y2=Y2):
    """The ``Dhat`` one-loop two-point integrand at mass levels (n1, n2).

    The paper's (5.17), with the ``1/y**(1+n)`` poles already stripped, so
    that the leading singularity is the plain coefficient of
    ``y1**n1 y2**n2``.

    ``Dhat`` is a particularly simple stringy completion: it distinguishes the
    two orientations of a spiral around the puncture and includes no closed or
    self-intersecting curves at all. That simplicity is what makes it a clean
    test case -- there are few enough parameters that the constraints cannot
    all be absorbed.
    """
    return ((1 + y1) ** (-n1 - n2 - x11 - x22)
            * (1 + y2) ** (-n1 - n2 - x11 - x22)
            * (1 + y1 + y1 * y2) ** (x11 + 2 * n1)
            * (1 + y2 + y1 * y2) ** (x22 + 2 * n2))


def dhat_leading_singularity(n1, n2, x11=X11, x22=X22):
    """The ``Dhat`` leading singularity at levels (n1, n2). Exact."""
    return sp.factor(residue2(dhat_integrand(n1, n2, x11, x22), n1, n2))


# ---------------------------------------------------------------------------
# closed curves, and the truncated "baby" integrals
# ---------------------------------------------------------------------------

def closed_curve_factor(delta1=DELTA1, delta2=DELTA2, order=8, y1=Y1, y2=Y2):
    """The contribution of the closed curves winding around the puncture.

    For an exponent ``Delta(q) = Delta1 q + Delta2 q**2``, the infinite
    products over windings collapse to closed forms -- the paper's (5.6) --
    the first to ``1 - y1 y2`` and the second to a Dedekind eta function
    written as an infinite product.

    ``order`` truncates that product. It is not an approximation in the
    dangerous sense: each further factor first contributes at a higher order
    in ``y1 y2``, so any residue below that order is already exact. The test
    checks independence of ``order`` rather than taking this on trust.
    """
    prod = sp.Integer(1) - y1 * y2
    for k in range(2, order):
        prod *= (1 - (y1 * y2) ** k) ** 2
    return (1 - y1 * y2) ** delta1 * prod ** delta2


def baby_leading_singularity(n1, n2, delta1=DELTA1, delta2=DELTA2, order=8):
    """Leading singularity of a truncated "baby" surface integral.

    Keeps only curves that do not self-intersect, plus the closed curves. With
    the open-curve exponents at their homology values the whole level
    dependence sits in the closed curves, and the answer at several different
    levels comes out the same -- which is already enough to fail, since the
    tree side does not.
    """
    e = (((1 + Y1) / (1 + Y2)) ** (n1 - n2)
         * closed_curve_factor(delta1, delta2, order))
    return sp.simplify(residue2(e, n1, n2))


def tachyon_level_one_residue(x22=X22, alpha0=ALPHA0):
    """The (1,0) residue that fixes an open-curve exponent, the paper's (5.10).

    Returns ``1 - X22 - alpha0``. Matching it against the glued three-point
    ansatz forces ``X22 = 0`` for any Regge intercept, which is also what
    homology says that curve's momentum should be: it starts and ends at the
    same boundary point and carries none.
    """
    e = ((1 + Y1) ** (1 - x22 - alpha0)
         * (1 + Y2) ** (-1 - x22 - alpha0)
         * (1 + Y2 + Y1 * Y2) ** (x22 + alpha0))
    return sp.simplify(residue2(e, 1, 0))


# ---------------------------------------------------------------------------
# the table, and the contradiction
# ---------------------------------------------------------------------------

#: Table 1 of the paper. The left column is *derived* by
#: :func:`dhat_leading_singularity` and appears here only for comparison; the
#: right column is the glued three-point ansatz with couplings matched to the
#: tree-level string amplitude, which is reference data -- deriving it needs
#: the tree-level coupling extraction at level two, which this module does not
#: do.
PAPER_TABLE_1 = {
    (0, 0): {"dhat": "1", "ansatz": "1"},
    (1, 0): {"dhat": "1 - X22", "ansatz": "1"},
    (2, 0): {"dhat": "(1 - X22)*(2 - X22)/2", "ansatz": "1"},
    (1, 1): {"dhat": "4 + X11 + X22 + X11*X22", "ansatz": "d"},
    (2, 1): {"dhat": "(4 - 9*X22 - 3*X22**2 - X11*X22 - X11*X22**2)/2",
             "ansatz": "29*d/32 - 13/16"},
}


#: The order in which the levels are matched, including the mirror levels.
#: The paper's Table 1 lists only ``n1 >= n2`` and states that the mirror rows
#: follow by exchanging the two curve exponents; the integrand is symmetric
#: under swapping the two coordinates together with the two exponents, so
#: :func:`dhat_leading_singularity` produces the mirrored rows by itself and
#: only the tree-side value has to be carried across.
MATCHING_ORDER = [(0, 0), (1, 0), (0, 1), (2, 0), (0, 2), (1, 1), (2, 1)]

#: Mirror levels and the row of Table 1 their tree-side value comes from.
MIRROR_OF = {(0, 1): (1, 0), (0, 2): (2, 0)}


def _ansatz(level):
    row = PAPER_TABLE_1[MIRROR_OF.get(level, level)]
    return sp.sympify(row["ansatz"], locals={"d": DIM})


def unitarity_constraints():
    """Match the ``Dhat`` residues against the tree-side values, level by level.

    Returns a record of how the matching proceeds and where it breaks. The
    sequence is the paper's: the massless-massive levels and their mirrors fix
    both open-curve exponents, the first doubly-massive level then fixes the
    space-time dimension, and the next level after that has nothing left to
    adjust and does not match.

    The mirror levels matter. Without them only one of the two exponents is
    determined, the dimension is never pinned, and the system looks
    satisfiable -- which is a good illustration of why a failure has to be
    reproduced carefully rather than asserted.

    The point of returning the whole sequence rather than a boolean is that
    the *shape* of the failure is the result: several constraints satisfied
    and the next one impossible is a much stronger statement than
    "inconsistent".
    """
    steps = []
    solution = {}

    for level in MATCHING_ORDER:
        lhs = sp.expand(dhat_leading_singularity(*level).subs(solution))
        rhs = sp.expand(_ansatz(level).subs(solution))
        eq = sp.expand(lhs - rhs)
        if eq == 0:
            steps.append({"level": level, "status": "already satisfied",
                          "lhs": lhs, "rhs": rhs, "fixed": {}})
            continue
        unknowns = sorted(eq.free_symbols & {X11, X22, DIM},
                          key=lambda s: s.name)
        sols = sp.solve(eq, unknowns, dict=True) if unknowns else []
        if sols and len(sols[0]) == len(unknowns):
            fixed = dict(sols[0])
            solution.update(fixed)
            steps.append({"level": level, "status": "fixes a parameter",
                          "lhs": lhs, "rhs": rhs, "fixed": fixed})
        elif sols:
            steps.append({"level": level, "status": "under-determined",
                          "lhs": lhs, "rhs": rhs, "fixed": {}})
        else:
            steps.append({"level": level, "status": "no solution",
                          "lhs": lhs, "rhs": rhs, "fixed": {}})

    failed = [s for s in steps if s["status"] == "no solution"]
    return {"steps": steps,
            "solution": solution,
            "consistent": not failed,
            "first_failure": failed[0]["level"] if failed else None}


# ---------------------------------------------------------------------------
# the contour, as combinatorics
# ---------------------------------------------------------------------------

def associahedron_faces(n, codim):
    """Sets of ``codim`` pairwise non-crossing chords of the n-gon.

    The faces of the associahedron, counted directly: a set of compatible
    chords is a partial triangulation, and the full triangulations -- codim
    ``n - 3`` -- are its vertices, so that count is the Catalan number. Which
    is a check on the enumeration, not an input to it.
    """
    cs = chords(n)
    total = 0
    for sub in combinations(cs, codim):
        if all(not crossing(a, b) for a, b in combinations(sub, 2)):
            total += 1
    return total


def pochhammer_pieces(n):
    """How many sheets, tubes and tori the generalised Pochhammer contour has.

    One sheet per subset of the chords, since the phases are in one-to-one
    correspondence with subsets of propagators; then the codimension-one
    boundaries of those sheets are glued in pairs by tubes and the
    codimension-two boundaries in fours by tori.

    At five points this is 32 sheets, 80 tubes and 40 tori, which is the
    count the paper gives.
    """
    c = len(chords(n))
    sheets = 2 ** c
    out = {"chords": c, "sheets": sheets}
    if c >= 1:
        out["tubes"] = associahedron_faces(n, 1) * sheets // 2
    if c >= 2:
        out["tori"] = associahedron_faces(n, 2) * sheets // 4
    return out


def minimal_cutoff(n):
    """The smallest common ``R*`` for which no branch cut is crossed.

    The binding constraint is the F-polynomial with the most monomials: every
    one of its ``n - 3`` non-constant terms must stay below ``1/(n-3)`` in
    modulus, which happens once every deformed coordinate has real part above
    ``log(n - 3)``.

    Exact as a bound. That the contour built with it converges everywhere in
    kinematic space is a different statement, and
    :meth:`CutsAndContours.contour_is_finite` refuses it.
    """
    if n <= 4:
        return sp.Integer(0)          # any positive cutoff will do
    return sp.log(n - 3)


# ---------------------------------------------------------------------------
# the theory
# ---------------------------------------------------------------------------

@register
class CutsAndContours(SurfaceTheory):
    """Unitarity cuts of a one-loop surface integral, and its contour.

    The verb here is :meth:`cuts`: residues of the integrand, taken directly,
    with no integration anywhere. :meth:`leading_singularity` is the same
    computation at the maximal residue.
    """

    key = "cuts-and-contours"

    def __init__(self, integrand="dhat", name=None):
        SurfaceTheory.__init__(
            self, surface="the once-punctured disk with two marked points",
            name=name)
        self.integrand = integrand

    def cuts(self, n1=0, n2=0, integrand=None):
        """The leading singularity on the cut at mass levels (n1, n2). Exact."""
        kind = integrand or self.integrand
        if kind == "dhat":
            return dhat_leading_singularity(n1, n2)
        if kind == "baby":
            return baby_leading_singularity(n1, n2)
        raise ValueError("unknown integrand %r; try 'dhat' or 'baby'" % kind)

    def leading_singularity(self, n1=0, n2=0, integrand=None):
        return self.cuts(n1, n2, integrand)

    def unitarity(self):
        """Run the level-by-level matching. See :func:`unitarity_constraints`."""
        return unitarity_constraints()

    def contour_is_finite(self, **kw):
        """Always raises: convergence is a theorem, not an arithmetic fact."""
        raise NotAnalytic(
            "that the contour renders the integral finite everywhere in "
            "kinematic space is a statement about convergence, proved by an "
            "argument about the whole moduli space. This module can build the "
            "contour, bound the cutoff that keeps it off the branch cuts, "
            "count its pieces and evaluate an example -- and an example is "
            "not the statement.",
            checkable=["the minimal cutoff R* > log(n-3), as an exact bound "
                       "on where the F-polynomials can vanish",
                       "the piece count of the generalised Pochhammer "
                       "contour, as faces of the associahedron",
                       "a single numerical evaluation at moderate kinematics"])

    def evaluate(self, **kw):
        """Always raises: at large kinematics this is a precision problem."""
        raise NotAnalytic(
            "numerical evaluation of the deformed contour at large negative "
            "kinematics is a fine cancellation between two exponentially "
            "large contributions, so the binding constraint is finite "
            "precision rather than method. Choosing the cutoff per cone "
            "mitigates it and cannot remove it, and past some point a saddle "
            "point approximation is the honest tool.",
            checkable=["the cutoff conditions per cone, which say exactly "
                       "which kinematic limits can be mitigated and which "
                       "cannot"])

    def alpha_prime_expansion(self, **kw):
        """Always raises: an integral, not a residue."""
        raise NeedsIntegration(
            "the low-energy expansion of the amplitude is an integral over "
            "the positive coordinates. This module computes residues of the "
            "integrand, which need no integration at all, and stops there.",
            missing=["integration over the positive coordinates in each cone "
                     "of the fan",
                     "a contour prescription wherever a planar variable is "
                     "negative"])

    def exact_content(self):
        return [
            "leading singularities of the one-loop two-point integrand as "
            "residues, exactly, at any mass level",
            "the level-by-level unitarity matching, including the level at "
            "which the Dhat integral becomes inconsistent",
            "the piece count of the generalised Pochhammer contour, as faces "
            "of the associahedron",
            "the minimal cutoff log(n-3) that keeps the deformation off the "
            "branch cuts",
        ]

    def declined(self):
        return [
            "that the contour converges everywhere (see contour_is_finite)",
            "numerical evaluation at large kinematics, which is a precision "
            "problem (see evaluate)",
            "the alpha' expansion, which needs integration",
        ]
