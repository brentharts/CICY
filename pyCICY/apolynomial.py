r"""
pyCICY.apolynomial -- A-polynomials, colored Jones, and the AJ conjecture.

Closing the loop
----------------
This package computes two things that look unrelated. :mod:`pyCICY.knots`
computes Jones polynomials of knots; :mod:`pyCICY.quantum_curve` quantizes
the mirror curve of a local Calabi-Yau into a difference operator. The
object that connects them is a plane curve in (C*)^2 together with its
quantization, and on the knot side that curve is the A-polynomial.

The A-polynomial A(M, L) of a knot cuts out the SL(2,C) character variety of
the knot complement, with M and L the eigenvalues of the meridian and
longitude. The AJ conjecture of Garoufalidis says the colored Jones
polynomials J_N obey a q-difference equation

    A-hat(Q, L; q) J_N = 0 ,     L J_N = J_{N+1},  Q J_N = q^N J_N,

whose operators satisfy L Q = q Q L -- the same Weyl algebra that
:mod:`pyCICY.quantum_curve` uses, with q = e^{i hbar} -- and that setting
q = 1 and Q = M^2 recovers the classical A-polynomial.

So both ends of the package quantize a curve in (C*)^2 by the same rule. The
Newton polygon is the shared combinatorial object, and
:func:`to_quantum_curve` hands a knot's A-polynomial to the very same
machinery that eats a toric diagram.

What is computed here
---------------------
*Colored Jones for torus knots*, from the closed formula (Rosso-Jones, in the
form used by Hikami and Lovejoy, arXiv:1409.6243 eq. 3.1),

    J_N(T(s,t); q) = q^{st(1-N^2)/4} / (q^{N/2} - q^{-N/2})
                     * sum_j q^{st j^2} ( q^{-(s+t)j + 1/2} - q^{-(s-t)j-1/2} )

over j = -(N-1)/2, ..., (N-1)/2. This is normalised so J_1 = 1, and it is a
completely independent route to the same invariant that
:mod:`pyCICY.knots` computes by counting states of the Kauffman bracket. The
tests check that J_2 reproduces :meth:`pyCICY.knots.Knot.jones` exactly for
T(2,3), T(2,5), T(2,7) and T(3,4) = 8_19 -- a representation-theoretic sum
against a sum over 2^n smoothings, agreeing term for term.

*The recursion*, searched for numerically. :func:`find_recursion` looks for
operators annihilating a table of colored Jones polynomials, by linear
algebra over the coefficients. :func:`classical_limit` then sets q = 1.
For the trefoil the smallest L-degree admitting a solution is 3, matching
what is known for its non-commutative A-polynomial, and the classical limit
of the whole nullspace has greatest common divisor

    (L - 1)^2 (M^2 - 1) (L M^6 + 1) ,

which contains the trefoil's A-polynomial exactly: the abelian factor L - 1
and the geometric factor 1 + L M^{pq} with pq = 6.

Two honest caveats about that computation. The extra factors are real and
expected: the classical limit of an annihilating operator contains the
A-polynomial but need not equal it. And the "smallest L-degree" is smallest
*within the search bounds* -- widen or narrow the ranges for the Q- and
q-degrees and the answer moves. :func:`find_recursion` therefore takes those
bounds explicitly and never claims minimality. The cost grows steeply with
the knot: the trefoil takes about twenty seconds, and T(2,5) needs bounds
large enough that it is not practical here.

*A-polynomials* themselves are quoted, not derived. For torus knots the
geometric factor is 1 + L M^{pq} (Cooper, Culler, Gillet, Long and Shalen);
the figure-eight entry is from Borot and Eynard, arXiv:1205.2261 eq. 200.
Computing an A-polynomial from scratch means eliminating variables from the
gluing equations of a triangulation, which is a different project.

Newton polygons and boundary slopes
-----------------------------------
The slopes of the edges of the Newton polygon of A are boundary slopes of
incompressible surfaces in the knot complement (Cooper, Culler, Gillet, Long
and Shalen). :func:`boundary_slopes` reads them off, and they come out right:
6 for the trefoil, which is pq, and +-4 for the figure-eight. Since the
Newton polygon is also exactly what :class:`pyCICY.quantum_curve.QuantumCurve`
consumes, the same few lattice points serve as a topological invariant of a
knot and as the hopping set of a lattice model.

That coincidence should not be oversold. A toric diagram is a *reflexive*
polygon and an A-polynomial's Newton polygon generally is not, so the
quantized A-polynomial is not the mirror curve of any local Calabi-Yau, and
:func:`to_quantum_curve` says so. What is genuinely shared is the
quantization rule, not the geometry.
"""

from fractions import Fraction

from . import knots as _knots
from . import quantum_curve as _qc
from . import toric as _toric

__all__ = [
    "colored_jones_torus", "laurent_divide",
    "A_POLYNOMIALS", "torus_apolynomial", "apolynomial", "abelian_factor",
    "multiply", "to_sympy", "newton_polygon", "boundary_slopes",
    "to_quantum_curve", "find_recursion", "classical_limit", "verify_aj",
]


# --------------------------------------------------------- colored Jones

def laurent_divide(num, den, guard=100000):
    """Exact Laurent division of ``{exponent: coefficient}`` dicts.

    Raises if the division does not terminate, which would mean the quotient
    is not a Laurent polynomial.
    """
    num = {e: Fraction(c) for e, c in num.items() if c}
    den = {e: Fraction(c) for e, c in den.items() if c}
    if not den:
        raise ZeroDivisionError("empty denominator")
    dtop = max(den)
    dlead = den[dtop]
    quot = {}
    steps = 0
    while num:
        steps += 1
        if steps > guard:
            raise RuntimeError("Laurent division did not terminate; the "
                               "quotient is not a Laurent polynomial")
        ntop = max(num)
        c = num[ntop] / dlead
        shift = ntop - dtop
        quot[shift] = quot.get(shift, 0) + c
        for e, dc in den.items():
            k = e + shift
            num[k] = num.get(k, 0) - c * dc
            if not num[k]:
                del num[k]
    return {e: (int(c) if c.denominator == 1 else c)
            for e, c in quot.items() if c}


def colored_jones_torus(s, t, N):
    """Normalized colored Jones of the right-handed torus knot T(s,t).

    Returns a :class:`pyCICY.knots.Laurent` in q, normalised so that
    ``J_1 = 1`` and ``J_2`` is the ordinary Jones polynomial. Exponents are
    carried internally in units of q^{1/4}, since the summand has quarter
    powers that only cancel at the end.
    """
    if N < 1:
        raise ValueError("colour N must be at least one")
    from math import gcd
    if gcd(s, t) != 1:
        raise ValueError("T(%d,%d) is a link, not a knot" % (s, t))

    num = {}
    prefactor = Fraction(s * t * (1 - N * N), 4)
    j = Fraction(-(N - 1), 2)
    while j <= Fraction(N - 1, 2):
        base = s * t * j * j + prefactor
        for e, c in ((base - (s + t) * j + Fraction(1, 2), 1),
                     (base - (s - t) * j - Fraction(1, 2), -1)):
            f = e * 4
            if f.denominator != 1:
                raise AssertionError("unexpected exponent %s" % e)
            num[int(f)] = num.get(int(f), 0) + c
        j += 1

    quot = laurent_divide(num, {2 * N: 1, -2 * N: -1})
    out = {}
    for e, c in quot.items():
        if e % 4:
            raise AssertionError("non-integer power of q survived: %s/4" % e)
        out[e // 4] = c
    return _knots.Laurent(out)


# ---------------------------------------------------------- A-polynomials
#
# Represented as {(i, j): coefficient} meaning coefficient * L^i * M^j.

def torus_apolynomial(p, q):
    """Geometric factor of the A-polynomial of T(p,q): ``1 + L M^{pq}``.

    Cooper, Culler, Gillet, Long and Shalen. The abelian factor L - 1 is not
    included; multiply by :func:`abelian_factor` if it is wanted.
    """
    from math import gcd
    if gcd(p, q) != 1:
        raise ValueError("T(%d,%d) is a link, not a knot" % (p, q))
    return {(0, 0): 1, (1, p * q): 1}


def abelian_factor():
    """The factor ``L - 1`` from the abelian component of the character variety."""
    return {(1, 0): 1, (0, 0): -1}


def multiply(a, b):
    """Product of two A-polynomials in the dict representation."""
    out = {}
    for (i1, j1), c1 in a.items():
        for (i2, j2), c2 in b.items():
            k = (i1 + i2, j1 + j2)
            out[k] = out.get(k, 0) + c1 * c2
    return {k: c for k, c in out.items() if c}


# Quoted, not derived. The torus entries follow the closed form above; the
# figure-eight geometric factor is Borot and Eynard, arXiv:1205.2261 eq. 200,
#     l^2 m^4 + l(-m^8 + m^6 + 2 m^4 + m^2 - 1) + m^4 .
A_POLYNOMIALS = {
    "3_1": torus_apolynomial(2, 3),
    "5_1": torus_apolynomial(2, 5),
    "7_1": torus_apolynomial(2, 7),
    "8_19": torus_apolynomial(3, 4),
    "4_1": {(2, 4): 1,
            (1, 8): -1, (1, 6): 1, (1, 4): 2, (1, 2): 1, (1, 0): -1,
            (0, 4): 1},
}


def apolynomial(name, include_abelian=False):
    """A-polynomial of a named knot, as ``{(i, j): coefficient}``."""
    if name not in A_POLYNOMIALS:
        raise KeyError("no A-polynomial recorded for %r; known: %s"
                       % (name, ", ".join(sorted(A_POLYNOMIALS))))
    A = dict(A_POLYNOMIALS[name])
    return multiply(A, abelian_factor()) if include_abelian else A


def to_sympy(A, symbols=("L", "M")):
    """The A-polynomial as a sympy expression."""
    import sympy as sp
    L, M = sp.symbols(" ".join(symbols))
    return sp.expand(sum(c * L ** i * M ** j for (i, j), c in A.items()))


# ------------------------------------------------ polygons and slopes

def newton_polygon(A):
    """Vertices of the Newton polygon of an A-polynomial, in (L, M) exponents."""
    return _toric.convex_hull(list(A))


def boundary_slopes(A):
    """Edge slopes dM/dL of the Newton polygon, as sorted Fractions.

    By Cooper, Culler, Gillet, Long and Shalen these are boundary slopes of
    incompressible surfaces in the knot complement. For the trefoil the
    single edge gives 6 = pq; for the figure-eight the four edges give +-4.
    """
    hull = newton_polygon(A)
    if len(hull) < 2:
        return []
    out = set()
    n = len(hull)
    pairs = ([(hull[i], hull[(i + 1) % n]) for i in range(n)] if n > 2
             else [(hull[0], hull[1])])
    for a, b in pairs:
        dl = b[0] - a[0]
        dm = b[1] - a[1]
        if dl:
            out.add(Fraction(dm, dl))
    return sorted(out)


def to_quantum_curve(A, name=None, drop_constant=True):
    """Quantize an A-polynomial with the same machinery as a mirror curve.

    The AJ operators satisfy L Q = q Q L, which is the Weyl algebra that
    :class:`pyCICY.quantum_curve.QuantumCurve` quantizes with q = e^{i hbar}.
    Feeding the Newton polygon of A to that class therefore applies the same
    quantization rule to the knot's curve.

    This is a statement about the *rule*, not about the geometry. A toric
    diagram is a reflexive polygon; the Newton polygon of an A-polynomial
    generally is not, so the resulting operator is not the quantized mirror
    curve of any local Calabi-Yau, and the spectra should not be read as a
    Hofstadter problem for one. Use :func:`pyCICY.toric.is_reflexive` on the
    polygon to see the difference for a given knot.

    The constant term sits at the origin, which is an on-site energy rather
    than a hop, and is dropped by default.
    """
    pts = [k for k in A if not (drop_constant and k == (0, 0))]
    if not pts:
        raise ValueError("nothing left to quantize after dropping the "
                         "constant term")
    coeffs = [A[k] for k in pts]
    return _qc.QuantumCurve(pts, coeffs=coeffs, name=name)


# ------------------------------------------------------------ the recursion

def find_recursion(js, dL, dQ, jlo, jhi, nmax):
    """Search for q-difference operators annihilating a colored Jones table.

    Looks for ``A-hat = sum_k a_k(Q, q) L^k`` with
    ``a_k(Q, q) = sum_{i,j} c_{k,i,j} Q^i q^j``, imposing

        sum_{k,i,j} c_{k,i,j} q^{iN+j} J_{N+k}(q) = 0

    for N = 1 .. nmax. Returns ``(terms, nullspace)``, where ``terms`` lists
    the ``(k, i, j)`` labelling the columns.

    The answer depends on the search bounds. An operator of a given L-degree
    may need a Q- or q-degree outside the window and will then be missed, so
    a failure to find one at L-degree d is *not* a proof that none exists.
    Nothing here should be read as computing a minimal-order recursion.

    Parameters
    ----------
    js : dict
        ``{N: Laurent}`` or ``{N: {exponent: coefficient}}``.
    dL, dQ : int
        Maximum powers of L and of Q.
    jlo, jhi : int
        Range of the free power of q.
    nmax : int
        Largest colour used to generate equations.
    """
    import sympy as sp

    tab = {}
    for N, v in js.items():
        tab[N] = dict(v.c) if hasattr(v, "c") else dict(v)

    terms = [(k, i, j)
             for k in range(dL + 1)
             for i in range(dQ + 1)
             for j in range(jlo, jhi + 1)]
    col_of = {t: c for c, t in enumerate(terms)}

    rows = {}
    for N in range(1, nmax + 1):
        if any((N + k) not in tab for k in range(dL + 1)):
            break
        for (k, i, j) in terms:
            for e, c in tab[N + k].items():
                key = (N, e + i * N + j)
                rows.setdefault(key, {})
                col = col_of[(k, i, j)]
                rows[key][col] = rows[key].get(col, 0) + c
    if not rows:
        return terms, []

    ordered = [rows[key] for key in sorted(rows)]
    M = sp.zeros(len(ordered), len(terms))
    for r, row in enumerate(ordered):
        for col, v in row.items():
            M[r, col] = v
    return terms, M.nullspace()


def _operator(terms, vec):
    out = {}
    for (k, i, j), v in zip(terms, vec):
        if v != 0:
            out.setdefault(k, {})[(i, j)] = v
    return out


def classical_limit(terms, vec):
    """Set q -> 1 and Q -> M^2 in an operator, giving a polynomial in L, M."""
    import sympy as sp
    L, M = sp.symbols("L M")
    op = _operator(terms, list(vec))
    expr = 0
    for k, coeffs in op.items():
        a = sum(c * M ** (2 * i) for (i, j), c in coeffs.items())
        expr += a * L ** k
    return sp.expand(expr)


def verify_aj(s=2, t=3, dL=3, dQ=4, jw=8, nmax=13, max_vectors=6):
    """Verify the AJ conjecture for a torus knot, as far as it can be verified.

    Computes colored Jones polynomials, searches for annihilating operators,
    takes the greatest common divisor of their classical limits, and checks
    that the known A-polynomial divides it.

    The gcd properly contains the A-polynomial: for the trefoil it comes out
    as ``(L-1)^2 (M^2-1) (L M^6 + 1)``, whose last factor is the geometric
    A-polynomial and whose ``L-1`` is the abelian component. The additional
    factors are an artefact of taking any annihilating operator rather than
    the minimal one, and are reported rather than divided away.

    The defaults are the trefoil, which takes roughly twenty seconds. Larger
    knots need larger bounds and get expensive quickly.
    """
    import sympy as sp
    L, M = sp.symbols("L M")

    js = {N: colored_jones_torus(s, t, N) for N in range(1, nmax + dL + 2)}
    terms, ns = find_recursion(js, dL=dL, dQ=dQ, jlo=-jw, jhi=jw, nmax=nmax)
    if not ns:
        return {"tessellation": (s, t), "found": False, "nullspace_dim": 0,
                "L_degree": dL, "note": "no operator within these bounds; "
                                        "this does not prove none exists"}

    limits = [classical_limit(terms, list(v)) for v in ns[:max_vectors]]
    g = limits[0]
    for e in limits[1:]:
        g = sp.gcd(g, e)
    target = sp.expand(to_sympy(torus_apolynomial(s, t)))
    quotient, remainder = sp.div(sp.Poly(sp.expand(g), L),
                                 sp.Poly(target, L))
    divides = sp.simplify(remainder.as_expr()) == 0
    return {
        "knot": "T(%d,%d)" % (s, t),
        "found": True,
        "L_degree": dL,
        "nullspace_dim": len(ns),
        "gcd_of_classical_limits": sp.factor(g),
        "a_polynomial": sp.factor(target),
        "a_polynomial_divides": bool(divides),
        "extra_factor": sp.factor(sp.simplify(g / target)) if divides else None,
        "note": ("the L-degree is the smallest that admits a solution within "
                 "the given search bounds, not a proven minimum"),
    }
