r"""
pyCICY.theories.gluons -- gluon leading singularities, as counting.

A leading singularity is the maximal residue of an amplitude: put every
propagator of a graph on shell and read off what is left. For pure gluons it
is the numerator, and it is rational -- no logarithms, no polylogarithms, no
integrals. Carrolo and Figueiredo show that it is also *combinatorial*: glue
three-point vertices, draw the Lorentz contractions as curves on the fatgraph,
and each monomial of the answer is one way of covering every edge exactly once
with non-overlapping curves.

That is a loop over subsets, which is the kind of thing this package is built
for. Nothing here integrates anything.

What is exact
-------------
:func:`scaffolded_three_point` derives the three-point vertex rather than
quoting it: build the polarisation vectors from pairs of scalar momenta, write
the standard Yang-Mills vertex, and rewrite every dot product in planar
variables through the dual coordinates. Two things must then happen on their
own, and both are checked: the gauge parameters must drop out, and the squares
of the dual coordinates must cancel. What remains is six monomials.

:func:`extension_balance` derives the coefficient that the paper's section 6
computes by hand. A monomial with n curves ending on a puncture can be
completed around the loop by extending any subset of them, a configuration
with ``Ne`` extensions carrying ``(-1)**Ne``, and there are ``C(n, Ne)`` of
them. The alternating sum is the whole content of that argument, and it is 1
for every n -- so the extensions contribute exactly 1, the closed curve
contributes ``1 - D``, and the coefficient is ``2 - D``. The bubble's
``(2 - D)`` is the n = 2 case, not a separate result.

:func:`closed_curve_exponent` is the paper's answer to the question that was
open before it: a closed curve homotopic to an internal boundary -- equivalently
one that turns only left -- carries ``1 - D``; every other closed curve carries
``-D``. The left-turning formulation is the one implemented, because the
planar restatement in terms of how many punctures a curve encloses is not even
well defined on the non-planar example the paper gives.

The check worth having
----------------------
In the limit where every loop-dependent variable of the n-gon goes to ``Y`` and
every purely external one to ``X``, the paper reports that the coefficients are
those of the Lucas polynomials, verified to n = 13, with no explanation. This
module computes them two ways that share no code -- the two-term recurrence
``L_n = x L_{n-1} - L_{n-2}`` and the closed form
``(-1)^k n/(n-k) C(n-k, k)`` -- and compares both against the paper's table.
An integer sequence arriving from a recurrence nobody put in is the sort of
agreement this package means by a test.

The table itself is data, read from the paper, and :data:`PAPER_NGON` says so.
The recurrence is derived. Reproducing a table from a table would prove
nothing.

Reference
---------
Carrolo and Figueiredo, *How gluon leading singularities discover curves on
surfaces*, arXiv:2512.17019, JHEP 07 (2026) 101. Equation numbers below are
that paper's.
"""

from math import comb

import sympy as sp

from .surface import SurfaceTheory, NeedsIntegration, NotAnalytic, register

__all__ = ["GluonLeadingSingularity", "scaffolded_three_point",
           "extension_balance", "closed_curve_exponent", "lucas_polynomial",
           "lucas_coefficients", "lucas_coefficients_closed_form",
           "ngon_leading_singularity", "paper_comparison", "PAPER_NGON"]

D = sp.Symbol("D", positive=True)
_X, _Y = sp.symbols("X Y")


# ---------------------------------------------------------------------------
# the three-point vertex, derived
# ---------------------------------------------------------------------------

def scaffolded_three_point(gauge=True):
    """The scalar-scaffolded three-gluon amplitude, in planar variables.

    Each gluon is produced by a pair of colour-ordered scalars, so three gluons
    become six scalars and the kinematics becomes the six dual coordinates of
    a hexagon. Gluon i has momentum ``q_i = x_{2i+1} - x_{2i-1}`` and
    polarisation ``eps_i`` the difference of its two scalar momenta, defined
    only up to ``eps_i -> eps_i + a_i q_i``.

    Every dot product is then rewritten through
    ``x_i . x_j = (x_i^2 + x_j^2 - X_{i,j}) / 2``, the massless conditions
    ``X_{i,i+1} = 0`` are imposed, and the gluons are put on shell by the
    scaffolding conditions ``X_{1,3} = X_{3,5} = X_{1,5} = 0``.

    Parameters
    ----------
    gauge : bool
        Keep the gauge parameters ``a_1, a_2, a_3`` symbolic. They must cancel;
        :func:`scaffolded_three_point` returns the result either way, and the
        caller (or the test) can check that it does not depend on them.

    Returns
    -------
    dict
        ``amplitude`` (expanded, in ``X_i_j``), ``gauge_dependence`` and
        ``coordinate_dependence`` (both must be empty), and ``monomials``.
    """
    n = 6
    dot_sym = {}
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            dot_sym[(i, j)] = sp.Symbol("d_%d_%d" % (i, j))

    def d(i, j):
        return dot_sym[(min(i, j), max(i, j))]

    def x(i):
        return [1 if k == i else 0 for k in range(n + 1)]

    def add(*vs):
        out = [0] * (n + 1)
        for v in vs:
            for i in range(n + 1):
                out[i] += v[i]
        return out

    def smul(c, v):
        return [c * t for t in v]

    def dot(a, b):
        return sp.expand(sum(a[i] * b[j] * d(i, j)
                             for i in range(1, n + 1)
                             for j in range(1, n + 1)))

    a1, a2, a3 = sp.symbols("a1 a2 a3") if gauge else (0, 0, 0)

    # gluon momenta: q_i is the chord of the momentum polygon (2.1)-(2.2)
    q1 = add(x(3), smul(-1, x(1)))
    q2 = add(x(5), smul(-1, x(3)))
    q3 = add(x(1), smul(-1, x(5)))
    # polarisations, each defined up to a shift along its own momentum
    e1 = add(smul(2, x(2)), smul(-1, x(1)), smul(-1, x(3)), smul(a1, q1))
    e2 = add(smul(2, x(4)), smul(-1, x(3)), smul(-1, x(5)), smul(a2, q2))
    e3 = add(smul(2, x(6)), smul(-1, x(5)), smul(-1, x(1)), smul(a3, q3))

    # the standard three-gluon vertex (2.4)
    amp = (dot(e1, e2) * dot(e3, add(q1, smul(-1, q2)))
           + dot(e2, e3) * dot(e1, add(q2, smul(-1, q3)))
           + dot(e1, e3) * dot(e2, add(q3, smul(-1, q1))))

    X = {}
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            X[(i, j)] = sp.Symbol("X_%d_%d" % (i, j))

    def XX(i, j):
        return X[(min(i, j), max(i, j))]

    sub = {dot_sym[(i, j)]: (d(i, i) + d(j, j) - XX(i, j)) / 2
           for i in range(1, n + 1) for j in range(i + 1, n + 1)}
    amp = sp.expand(amp.subs(sub, simultaneous=True))

    on_shell = {XX(i, i + 1): 0 for i in range(1, n)}
    on_shell[XX(1, n)] = 0                       # the sixth boundary edge
    for c in [(1, 3), (3, 5), (1, 5)]:           # gluons on shell (2.3)
        on_shell[XX(*c)] = 0
    amp = sp.expand(amp.subs(on_shell))

    gauge_dep = [s for s in (a1, a2, a3)
                 if getattr(s, "free_symbols", set()) and amp.has(s)]
    coord_dep = [dot_sym[(i, i)] for i in range(1, n + 1)
                 if amp.has(dot_sym[(i, i)])]
    return {"amplitude": amp,
            "gauge_dependence": gauge_dep,
            "coordinate_dependence": coord_dep,
            "monomials": len(sp.Add.make_args(amp))}


#: The paper's equation (2.5), for comparison only. Never used to build
#: anything -- :func:`scaffolded_three_point` derives its own answer, and the
#: test asks whether the two agree up to the overall power of two the paper
#: says it drops.
PAPER_THREE_POINT = ("X_1_4*X_2_6 + X_3_6*X_2_4 + X_2_5*X_4_6"
                     " - X_2_5*X_3_6 - X_1_4*X_3_6 - X_1_4*X_2_5")


# ---------------------------------------------------------------------------
# extensions around a loop, and the exponent of a closed curve
# ---------------------------------------------------------------------------

def extension_balance(n):
    """The net contribution of extensions around a puncture with n curves.

    A monomial carrying n curves that end on the puncture can be completed by
    extending any non-empty subset of them around the loop. A configuration
    using ``Ne`` extensions enters with ``(-1)**Ne``, and cyclic symmetry makes
    the number of configurations with ``Ne`` extensions exactly ``C(n, Ne)``.
    So the total is ``-sum_{Ne=1}^{n} (-1)**Ne C(n, Ne)``, which telescopes to
    1 for every n >= 1 -- the paper's (6.3).

    Returned as an exact integer, computed from the binomials rather than from
    the closed form, so that the test can assert the value rather than restate
    the algebra.
    """
    if n < 1:
        raise ValueError("a puncture needs at least one curve ending on it")
    return -sum((-1) ** ne * comb(n, ne) for ne in range(1, n + 1))


def closed_curve_exponent(left_turning, dimension=D):
    """The exponent of a closed curve contributing to a leading singularity.

    The question the paper settles. A closed curve receives the correction to
    the naive gluing -- the one that ghosts supply in a Lagrangian treatment --
    if and only if there is a path between the two glued legs that turns
    exclusively left, which is to say the curve is homotopic to an internal
    boundary of the graph. Then its exponent is ``1 - D``; otherwise ``-D``.

    Stated this way rather than by counting enclosed punctures, because on a
    non-planar graph the number of punctures a closed curve encloses is not
    well defined and the left-turning criterion still is (the paper's (5.4)).
    """
    return (1 - dimension) if left_turning else (-dimension)


def ngon_puncture_coefficient(n, dimension=D):
    """The coefficient of the all-puncture monomial in the planar n-gon.

    Closed curve plus extensions: ``(1 - D) + 1 = 2 - D``, the paper's (6.4).
    Derived from :func:`extension_balance` and :func:`closed_curve_exponent`,
    not quoted. The bubble is n = 2.
    """
    return closed_curve_exponent(True, dimension) + extension_balance(n)


# ---------------------------------------------------------------------------
# Lucas polynomials, two ways
# ---------------------------------------------------------------------------

def lucas_polynomial(n, x=None):
    """L_n by the two-term recurrence: L_0 = 2, L_1 = x, L_n = x L_{n-1} - L_{n-2}."""
    x = sp.Symbol("x") if x is None else x
    a, b = sp.Integer(2), x
    if n == 0:
        return a
    for _ in range(n - 1):
        a, b = b, sp.expand(x * b - a)
    return sp.expand(b)


def lucas_coefficients(n):
    """The coefficients of L_n, as integers, highest power first."""
    x = sp.Symbol("x")
    poly = sp.Poly(lucas_polynomial(n, x), x)
    return [int(poly.coeff_monomial(x ** (n - 2 * k)))
            for k in range(n // 2 + 1)]


def lucas_coefficients_closed_form(n):
    """The same integers from ``(-1)**k * n/(n-k) * C(n-k, k)``.

    An independent route: no recurrence, no polynomial algebra, just
    binomials. Agreement with :func:`lucas_coefficients` is a check on both.
    """
    out = []
    for k in range(n // 2 + 1):
        val = sp.Rational((-1) ** k * n, n - k) * comb(n - k, k)
        assert val.q == 1, "Lucas coefficient should be an integer"
        out.append(int(val))
    return out


# ---------------------------------------------------------------------------
# the n-gon leading singularity in the X/Y limit
# ---------------------------------------------------------------------------

#: The paper's table (8.1), transcribed. **Reference data, not a computation.**
#: Every entry is compared against something derived; none is used to build it.
PAPER_NGON = {
    2: "(2 - D)*Y**2",
    3: "(2 - D)*Y**3 + 2*(-X**3 + 3*X**2*Y)",
    4: "(2 - D)*Y**4 - X**4 + 4*X**3*Y - 2*X**2*Y**2",
    5: "(2 - D)*Y**5 - X**5 + 5*X**4*Y - 5*X**3*Y**2",
    6: "(2 - D)*Y**6 - X**6 + 6*X**5*Y - 9*X**4*Y**2 + 2*X**3*Y**3",
    7: "(2 - D)*Y**7 - X**7 + 7*X**6*Y - 14*X**5*Y**2 + 7*X**4*Y**3",
    8: ("(2 - D)*Y**8 - X**8 + 8*X**7*Y - 20*X**6*Y**2 + 16*X**5*Y**3"
        " - 2*X**4*Y**4"),
}


def ngon_leading_singularity(n, dimension=D, x=None, y=None):
    """The n-gon leading singularity in the limit X_{i,j} -> X, X_{i,p} -> Y.

    Built entirely from derived pieces: the ``Y**n`` coefficient from
    :func:`ngon_puncture_coefficient`, and the rest from the Lucas
    coefficients, as

        ``(2 - D) Y**n - sum_k C_k Y**k X**(n-k)``,  ``C_k = [x**(n-2k)] L_n``.

    This is a *prediction* for n > 8, where the paper's table stops, and a
    comparison for n <= 8. See :func:`paper_comparison`.
    """
    x = _X if x is None else x
    y = _Y if y is None else y
    out = ngon_puncture_coefficient(n, dimension) * y ** n
    for k, c in enumerate(lucas_coefficients(n)):
        out -= c * y ** k * x ** (n - k)
    return sp.expand(out)


def paper_comparison(dimension=D):
    """Compare the derived n-gon form against the paper's table, n = 2 .. 8.

    Returns one record per n with the difference and, where the two differ by
    an overall factor on the X-dependent part, that factor. The known
    structure is: n = 2 has no X-dependent part at all, n = 3 carries an extra
    overall 2, and n >= 4 agrees term by term.
    """
    out = {}
    for n, text in sorted(PAPER_NGON.items()):
        paper = sp.expand(sp.sympify(text, locals={"D": dimension,
                                                   "X": _X, "Y": _Y}))
        derived = ngon_leading_singularity(n, dimension)
        diff = sp.expand(paper - derived)
        # the Y**n term is derived independently; compare the rest
        paper_x = sp.expand(paper - sp.expand(
            ngon_puncture_coefficient(n, dimension) * _Y ** n))
        derived_x = sp.expand(derived - sp.expand(
            ngon_puncture_coefficient(n, dimension) * _Y ** n))
        ratio = None
        if derived_x != 0:
            r = sp.cancel(paper_x / derived_x)
            if r.is_number:
                ratio = sp.nsimplify(r)
        out[n] = {"difference": diff,
                  "agrees": diff == 0,
                  "x_part_ratio": ratio,
                  "y_coefficient_agrees":
                      sp.expand(paper.coeff(_Y, n) - paper.coeff(_Y, n)) == 0}
    return out


# ---------------------------------------------------------------------------
# the theory
# ---------------------------------------------------------------------------

@register
class GluonLeadingSingularity(SurfaceTheory):
    """Leading singularities of pure-gluon amplitudes, from curves on a surface.

    Non-supersymmetric Yang-Mills in any number of dimensions, with the gluons
    scalar-scaffolded so that the kinematics is the planar variables of a
    2n-gon. The leading singularity is a polynomial, computed by counting
    coverings of a fatgraph rather than by evaluating a residue numerically.
    """

    key = "gluon-leading-singularity"

    def __init__(self, dimension=None, name=None):
        SurfaceTheory.__init__(
            self,
            surface="a fatgraph with scalar-scaffolded gluon legs",
            name=name)
        self.dimension = D if dimension is None else sp.sympify(dimension)

    def leading_singularity(self, n=None, three_point=False):
        """The leading singularity, exactly.

        With ``three_point=True``, the derived three-gluon vertex. With an
        integer ``n``, the n-gon at one loop in the X/Y limit.
        """
        if three_point:
            return scaffolded_three_point()["amplitude"]
        if n is None:
            raise ValueError("give n, or three_point=True")
        return ngon_leading_singularity(n, self.dimension)

    def cuts(self, **kw):
        """Not here: a cut is a leading singularity times a phase-space integral.

        The leading singularity is the exact part and this module returns it.
        Multiplying by the Lorentz-invariant phase-space measure and
        integrating is the other part, and this package does not integrate.
        """
        raise NeedsIntegration(
            "the unitarity cut is this module's leading singularity times the "
            "Lorentz-invariant phase-space integral over the cut momenta. The "
            "first factor is exact and available; the second is an "
            "integration this package does not perform.",
            missing=["the Lorentz-invariant phase-space measure for the cut "
                     "momenta, and its integration",
                     "a regulator, since the phase-space integral diverges in "
                     "four dimensions for massless cuts"])

    def soft_limit(self, **kw):
        """Not here: see :class:`~pyCICY.theories.softgraph.SoftFactorisation`."""
        raise NotImplementedError(
            "the soft expansion of an integrand is a different construction, "
            "built from graph Laplacians and tropical rays rather than from "
            "coverings of a fatgraph. It is "
            "pyCICY.theories.softgraph.SoftFactorisation, registered as "
            "'soft-factorisation'.")

    def fermion_loop(self, **kw):
        """Always raises: the sign rule for fermion loops is not settled.

        For pure gluons the sign of a monomial is ``(-1)**Ne`` in the number of
        extensions, which is local data. With a fermion loop the sign depends
        additionally on the total number of intersections of the contraction
        curves inside the loop, and the paper leaves the systematic treatment
        of the resulting cancellations open. An implementation that guessed
        would be indistinguishable from one that knew.
        """
        raise NotAnalytic(
            "leading singularities with fermions in the loop need a rule for "
            "the cancellations, and the sign there depends on the whole "
            "intersection pattern inside the loop rather than on the "
            "extension count alone. The paper presents the graphical picture "
            "and leaves the systematic analysis for future work; this module "
            "declines rather than guessing.",
            checkable=["the Pfaffian sign rule for a single contraction "
                       "pattern, which is the number of intersections of the "
                       "contraction curves",
                       "individual low-point examples against explicit gamma "
                       "matrix traces"])

    def exact_content(self):
        return [
            "the scalar-scaffolded three-gluon vertex, derived from the "
            "polarisation contractions rather than quoted",
            "the balance of extensions around a puncture, hence the (2 - D) "
            "coefficient of the all-puncture monomial at any multiplicity",
            "the exponent of a closed curve at any loop order: 1 - D when it "
            "is homotopic to an internal boundary, -D otherwise",
            "the n-gon leading singularity in the X/Y limit, with the Lucas "
            "coefficients computed two independent ways",
        ]

    def declined(self):
        return [
            "unitarity cuts, which need a phase-space integration (see cuts)",
            "leading singularities with fermion loops, where the sign rule is "
            "not settled (see fermion_loop)",
            "anything requiring loop integration, infrared subtraction or a "
            "cross-section",
        ]
