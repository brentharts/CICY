r"""
pyCICY.sixsphere_cosmology -- two further matches between the monotile's numbers and a cosmological singularity,
stated as what they are: exact, generic, and not connections.

The six-sphere paper (paper/supplementary_material_sixsphere.tex) already separates identities of integers from
connections.  This module adds two matches of that kind, with the exact computation behind each.

1. THE BKL MAP.  The Mixmaster map  u -> u - 1 (u >= 2),  u -> 1/(u - 1) (1 < u < 2)  is the Gauss map, so a closed
   orbit is a periodic continued fraction and its multiplier is the square of the unit of the matrix the partial
   quotients multiply to.  The Spectre inflation lambda^2 = 4 + sqrt15 is that unit for the period (6,1); the orbit of
   u* = 3 + sqrt15 closes after 7 epochs (eras of 6 and 1) with multiplier lambda^4.  The perimeter growth
   nu = 2 + sqrt5 = [4;4,...] is the one-era orbit of 4 epochs, multiplier -nu^2.  Hence the outline dimension
   D = ln nu / ln lambda is 2 ln|m_nu| / ln m_lambda.  All of that holds for EVERY periodic continued fraction
   (checked over a grid), so it is a theorem about continued fractions and not a property of the monotile.  The
   Kasner denominator D(u) = 1 + u + u^2 = |u - zeta_3|^2 is an Eisenstein norm, and D(u*) = 7 lambda^2: a one-off,
   rational over the unit only at this point of this orbit, left unexplained.

2. THE PANCAKE.  The collapsing fibre of the pancake winds (2/a0) ln(t/t_in) (the horizon coefficient 2/a0, quoted
   from pyCICY.horizons) over an Euler-angle period 4 pi (identification not checked).  One substitution level, a
   rescaling by lambda, is then ln(lambda)/(2 pi a0) fibre turns, and 2 pi a0 / ln(lambda) = 6.090 levels make one
   turn at a0 = 1: near the 6 of Z/6, not 6; a0 = 0.985 would give exactly 6, a tuned value.  Numerology, asserted
   as a miss.  Separately and exactly: the 60-degree rotation R of the hexagonal cusp torus has stabilisers 6, 2, 3 on
   the six site classes, so the torus modulo Z/6 is the orbifold sphere S^2(2,3,6) with chi = 0 = chi(T)/6, the
   Euclidean (2,3,6) the paper already names as the substrate's symmetry.  Z/6 in the fibre circle gives the lens space
   L(6,1); sharing the group is the whole overlap and no map is constructed.

What it does not say: that the tiling has dynamics, that the six-sphere has a cosmology, or that either match
explains anything.  :func:`status` returns every check; a failing check stops the facts from being written.
"""

from fractions import Fraction

import sympy as sp

__all__ = ["bkl_step", "closed_orbit", "periodic_u", "unit_of", "dimension_two_ways", "levels_per_turn",
           "a0_for_levels_per_turn", "stabilisers", "status"]

S15, S5 = sp.sqrt(15), sp.sqrt(5)
LAMBDA2 = 4 + S15
LAMBDA = sp.sqrt(LAMBDA2)
NU = 2 + S5
R = ((0, -1), (1, 1))       # the 60-degree rotation of the hexagonal lattice in the basis (E1, E2)
SITES = [(Fraction(0), Fraction(0)), (Fraction(1, 2), Fraction(0)), (Fraction(0), Fraction(1, 2)),
         (Fraction(1, 2), Fraction(1, 2)), (Fraction(1, 3), Fraction(1, 3)), (Fraction(2, 3), Fraction(2, 3))]


def bkl_step(u):
    """One epoch of the BKL map, exact: (u', du'/du)."""
    if sp.N(u, 60) >= 2:
        return sp.radsimp(sp.simplify(u - 1)), sp.Integer(1)
    return sp.radsimp(sp.simplify(1 / (u - 1))), sp.simplify(-1 / (u - 1) ** 2)


def closed_orbit(u0, max_epochs=40):
    """(epochs of u0 until it returns, era lengths, multiplier).  An inversion ends an era."""
    us, ders, u = [u0], [], u0
    for _ in range(max_epochs):
        u, d = bkl_step(u)
        ders.append(d)
        if sp.simplify(u - u0) == 0:
            eras, run = [], 0
            for d_ in ders:
                run += 1
                if d_ != 1:
                    eras.append(run)
                    run = 0
            assert run == 0
            return us, eras, sp.radsimp(sp.simplify(sp.prod(ders)))
        us.append(u)
    raise ValueError("no closed orbit within %d epochs" % max_epochs)


def periodic_u(a, b):
    """[a; b, a, b, ...] as an exact surd."""
    x = sp.Symbol("x", positive=True)
    return [s for s in sp.solve(sp.Eq(x, a + 1 / (b + 1 / x)), x) if sp.N(s) > 0][0]


def unit_of(a, b):
    M = sp.Matrix([[a, 1], [1, 0]]) * sp.Matrix([[b, 1], [1, 0]])
    return max(M.eigenvals(), key=lambda e: sp.N(e))


def dimension_two_ways():
    """(ln nu / ln lambda, 2 ln|m_nu| / ln m_lambda) from the two closed orbits: equal."""
    _, _, m_nu = closed_orbit(NU)
    _, _, m_lam = closed_orbit(3 + S15)
    return (float(sp.log(NU) / sp.log(LAMBDA)), float(2 * sp.log(abs(m_nu)) / sp.log(m_lam)))


def levels_per_turn(a0):
    """2 pi a0 / ln(lambda): substitution levels per fibre turn of the pancake."""
    return 2 * sp.pi * sp.sympify(a0) / sp.log(LAMBDA)


def a0_for_levels_per_turn(N):
    return sp.sympify(N) * sp.log(LAMBDA) / (2 * sp.pi)


def _mat_pow(A, n):
    out = ((1, 0), (0, 1))
    for _ in range(n):
        out = tuple(tuple(sum(out[i][k] * A[k][j] for k in range(2)) for j in range(2)) for i in range(2))
    return out


def _act(A, p):
    q = tuple(sum(A[i][k] * p[k] for k in range(2)) for i in range(2))
    return tuple(c - (c.numerator // c.denominator) for c in q)


def stabilisers():
    """({site: stabiliser order in <R>}, {orbit: stabiliser order}) on the six site classes."""
    st = {p: sum(1 for k in range(6) if _act(_mat_pow(R, k), p) == p) for p in SITES}
    orbits = {frozenset(_act(_mat_pow(R, k), p) for k in range(6)): st[p] for p in SITES}
    return st, orbits


def status():
    """Every check, and the numbers the paper quotes.  A failing check is reported in ``checks``."""
    checks, f = {}, {}
    us, eras, mult = closed_orbit(3 + S15)
    M = sp.Matrix([[6, 1], [1, 0]]) * sp.Matrix([[1, 1], [1, 0]])
    x = sp.Symbol("x")
    checks["spectre orbit: 7 epochs, eras (6,1)"] = len(us) == 7 and eras == [6, 1]
    checks["spectre multiplier = lambda^4"] = sp.simplify(mult - LAMBDA2 ** 2) == 0
    checks["lambda^2 from [[7,6],[1,1]], x^2 - 8x + 1"] = (
        M == sp.Matrix([[7, 6], [1, 1]]) and sp.expand(M.charpoly(x).as_expr() - (x ** 2 - 8 * x + 1)) == 0)
    u2, eras2, mult2 = closed_orbit(NU)
    checks["nu orbit: 4 epochs, one era, multiplier -nu^2"] = (
        len(u2) == 4 and eras2 == [4] and sp.simplify(mult2 + NU ** 2) == 0)
    checks["nu = phi^3"] = sp.simplify(sp.GoldenRatio ** 3 - NU) == 0
    d1, d2 = dimension_two_ways()
    checks["D two ways"] = abs(d1 - d2) < 1e-12 and abs(d1 - 1.3993) < 5e-5
    grid = ((1, 6), (2, 3), (3, 1), (1, 4), (2, 5), (4, 3))
    ok = True
    for a, b in grid:
        _, e, m = closed_orbit(periodic_u(a, b))
        ok = ok and sorted(e) == sorted([a, b]) and sp.simplify(abs(m) - unit_of(a, b) ** 2) == 0
    checks["every periodic continued fraction closes (grid)"] = ok
    # D(u*) = 7 lambda^2, a one-off
    checks["D(u*) = 7 lambda^2"] = sp.simplify(1 + us[0] + us[0] ** 2 - 7 * LAMBDA2) == 0
    other = [(a, b) for a, b in ((1, 6), (2, 3), (3, 2), (2, 5))
             if not sp.radsimp((1 + periodic_u(a, b) + periodic_u(a, b) ** 2) / unit_of(a, b)).is_rational]
    checks["D = 7 epsilon is a one-off (irrational at the other starts)"] = len(other) == 4
    eisen = sp.Symbol("u", real=True)
    z3 = sp.Rational(-1, 2) + sp.sqrt(3) * sp.I / 2
    checks["1 + u + u^2 = |u - zeta_3|^2"] = sp.simplify(
        sp.expand(sp.Abs(eisen - z3) ** 2, complex=True) - (1 + eisen + eisen ** 2)) == 0
    # pancake clock
    N1 = sp.N(levels_per_turn(1), 30)
    a06 = sp.N(a0_for_levels_per_turn(6), 30)
    checks["fibre clock misses 6"] = abs(N1 - 6) > 0.08
    checks["a0 giving 6 is tuned (not 1)"] = abs(a06 - 1) > 0.01 and sp.simplify(levels_per_turn(a0_for_levels_per_turn(6)) - 6) == 0
    # orbifold
    st, orbits = stabilisers()
    checks["stabilisers 6,2,2,2,3,3"] = sorted(st.values()) == [2, 2, 2, 3, 3, 6]
    checks["orbits (2,3,6), orbit-stabiliser"] = sorted(orbits.values()) == [2, 3, 6] and all(
        len(o) * m == 6 for o, m in orbits.items())
    chi = 2 - sum(1 - Fraction(1, m) for m in orbits.values())
    checks["orbifold chi = 0"] = chi == 0
    checks["R^6 = 1, R^3 = -1"] = _mat_pow(R, 6) == ((1, 0), (0, 1)) and _mat_pow(R, 3) == ((-1, 0), (0, -1))
    f.update(Period=len(us), Eras=",".join(str(e) for e in eras), NuPeriod=len(u2), NuEras=",".join(str(e) for e in eras2),
             GridChecked=len(grid), Dim="%.4f" % d1, LevelsPerTurn="%.3f" % float(N1),
             LevelsMissPercent="%.1f" % float(100 * (N1 - 6) / 6), AzeroSix="%.3f" % float(a06),
             StabOrders=", ".join(str(m) for m in sorted(orbits.values(), reverse=True)),
             OrbifoldChi=int(chi), MultNum="%.2f" % float(mult), NuMultNum="%.2f" % float(-mult2),
             OneOffStarts=len(other))
    return dict(checks=checks, facts=f)
