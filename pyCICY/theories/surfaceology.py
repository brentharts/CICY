r"""
pyCICY.theories.surfaceology -- curves on a disk, exactly.

The machinery every amplitude module in this package sits on. A tree-level
colour-ordered amplitude is an integral over a surface -- the disk with n
marked points on its boundary -- and the objects that carry the physics are
*curves* on that surface, one per chord (i,j). Each curve has a kinematic
variable X_{i,j}, the planar Mandelstam, and a u-variable u_{i,j} in [0,1]
which vanishes exactly where that channel goes on shell.

Nothing here needs a metric, a regulator or a floating-point number. A
u-variable is a ratio of polynomials in positive coordinates y with integer
coefficients, and everything this module computes is an identity between such
ratios, checked with sympy over the rationals.

Two routes, as always
---------------------
The u-variables are built here from the F-polynomials, which is a closed form
read off the ladder triangulation. That construction is *not* self-checking:
a wrong index convention would give wrong u's that still look like u's.

So the check is the u-equations,

    u_X + prod_{X'} u_{X'}^{#(X,X')} = 1,

with #(X,X') the number of times the two curves cross. These are the defining
equations of the space -- an algebraic presentation of Teichmueller space --
and they share no code with the F-polynomial formula: one is a product of
ratios, the other a statement about which chords of a polygon interleave.
:func:`u_equation_residuals` requires every residual to be identically zero as
a rational function, not zero at sampled points.

That check is what fixes the index conventions. The boundary behaviour of the
F-polynomials is stated in the literature for a particular range of indices;
here F_{i,j} is simply 1 when its sum is empty, and y_{1,k} is 0 outside the
n-3 genuine coordinates. Those two conventions are not asserted, they are the
ones under which the u-equations hold, and the test says so.

References
----------
Figueiredo and Skowronek, *Cuts and contours*, JHEP 12 (2025) 024,
arXiv:2506.05456, section 2.1, for the positive parametrisation and the
F-polynomials; Arkani-Hamed, Frost, Salvatori, Plamondon and Thomas for the
counting problem the parametrisation comes from.
"""

import sympy as sp

__all__ = ["chords", "crossing", "y_symbols", "f_polynomial", "u_variable",
           "u_variables", "u_equation_residuals", "koba_nielsen",
           "planar_variables", "nonplanar_variable"]


# ---------------------------------------------------------------------------
# the combinatorics of a polygon: which chords exist, and which cross
# ---------------------------------------------------------------------------

def chords(n):
    """The chords of the n-gon: pairs (i,j) that are not boundary edges.

    There are n(n-3)/2 of them, one per propagator that can appear in the
    colour-ordered amplitude, and they are the curves on the disk.
    """
    out = []
    for i in range(1, n + 1):
        for j in range(i + 2, n + 1):
            if (i, j) != (1, n):
                out.append((i, j))
    return out


def crossing(c1, c2):
    """Number of times chords c1 and c2 cross inside the disk: 0 or 1.

    Two chords of a polygon cross exactly when their endpoints interleave
    around the boundary. At tree level this is all the u-equations need; at
    loop level the same exponent can be any non-negative integer, because
    curves may wind around a puncture.
    """
    (a, b), (c, d) = c1, c2
    return 1 if (a < c < b < d) or (c < a < d < b) else 0


# ---------------------------------------------------------------------------
# the positive parametrisation
# ---------------------------------------------------------------------------

def y_symbols(n):
    """The n-3 positive coordinates of the ladder triangulation.

    The underlying triangulation is the ray-like one, T = {(1,3), ..., (1,n-1)},
    whose dual fatgraph is the ladder. One coordinate per chord in T.
    """
    return {k: sp.Symbol("y_1_%d" % k, positive=True)
            for k in range(3, n)}


def _y(y, k):
    """y_{1,k} inside its range, and 0 outside it.

    Outside the range there is no chord, so no coordinate; every monomial
    that would involve one is absent. Writing it as zero rather than special
    casing the sums keeps f_polynomial a single expression, and the
    u-equations are what certify the choice.
    """
    return y.get(k, sp.Integer(0))


def f_polynomial(i, j, y):
    """F_{i,j}: one polynomial per pair, with integer coefficients.

    F_{i,j} = 1 + sum_{m=i+2}^{j} prod_{k=m}^{j} y_{1,k},

    so it is 1 whenever the sum is empty (j < i+2), and 1 again whenever every
    monomial involves a coordinate outside the range.
    """
    total = sp.Integer(1)
    for m in range(i + 2, j + 1):
        term = sp.Integer(1)
        for k in range(m, j + 1):
            term *= _y(y, k)
        total += term
    return sp.expand(total)


def u_variable(i, j, y):
    """The u-variable of the curve (i,j), as a ratio of F-polynomials."""
    if i == 1:
        return sp.cancel(_y(y, j) * f_polynomial(1, j - 1, y)
                         / f_polynomial(1, j, y))
    return sp.cancel(f_polynomial(i - 1, j, y) * f_polynomial(i, j - 1, y)
                     / (f_polynomial(i, j, y) * f_polynomial(i - 1, j - 1, y)))


def u_variables(n, y=None):
    """All u-variables of the n-point disk, keyed by chord."""
    y = y_symbols(n) if y is None else y
    return {c: u_variable(c[0], c[1], y) for c in chords(n)}


def u_equation_residuals(n, y=None):
    """u_X + prod_{X'} u_{X'}^{#(X,X')} - 1, for every chord X.

    Zero for every chord, identically in the y. This is the independent route:
    :func:`u_variable` knows about F-polynomials and nothing about which chords
    interleave; :func:`crossing` knows the interleaving and nothing about
    F-polynomials.
    """
    y = y_symbols(n) if y is None else y
    u = u_variables(n, y)
    out = {}
    for c in chords(n):
        prod = sp.Integer(1)
        for d in chords(n):
            if d != c and crossing(c, d):
                prod *= u[d]
        out[c] = sp.simplify(sp.together(u[c] + prod - 1))
    return out


# ---------------------------------------------------------------------------
# kinematics
# ---------------------------------------------------------------------------

def planar_variables(n):
    """Symbols X_{i,j}, one per chord."""
    return {c: sp.Symbol("X_%d_%d" % c) for c in chords(n)}


def nonplanar_variable(i, j, X):
    """c_{i,j} = X_{i,j} + X_{i+1,j+1} - X_{i,j+1} - X_{i+1,j}.

    The exponent of an F-polynomial in the y-space integrand. It is minus
    twice a dot product of external momenta, hence the name: it is the piece
    of the kinematics that is not a planar pole.
    """
    def g(a, b):
        a, b = min(a, b), max(a, b)
        return X.get((a, b), sp.Integer(0))
    return g(i, j) + g(i + 1, j + 1) - g(i, j + 1) - g(i + 1, j)


def koba_nielsen(n, y=None, X=None):
    """The Koba-Nielsen factor prod_C u_C^{X_C}, as an unexpanded product.

    The tree-level string integrand is dlog(y) times this. Returned
    symbolically so callers can take residues in y, which is how every
    singularity of the amplitude is reached.
    """
    y = y_symbols(n) if y is None else y
    X = planar_variables(n) if X is None else X
    u = u_variables(n, y)
    out = sp.Integer(1)
    for c in chords(n):
        out *= u[c] ** X[c]
    return out
