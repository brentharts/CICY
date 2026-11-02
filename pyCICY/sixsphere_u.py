"""
u-variables from F-polynomials, the punctured disk, and why the torus resists
(Part XIV-c).  Companion to :mod:`pyCICY.sixsphere_surface` and
``FigueiredoSurfaceology.md`` (the disk modules ``pyCICY.theories.surfaceology``).

The method needs no formula for u.  With principal coefficients every cluster
variable has an F-polynomial F_g(y) and a g-vector.  Put y_i = t^{m_i}, t -> 0.
Binary geometry says u_g ~ t^{nu_g(m)}, where nu_g(m) is the coordinate of m on
the ray -g_g of the g-fan cone that contains m (and 0 off the cone).  Asking
that nu_g be a combination of the valuations of the F-polynomials,

    nu_g(m) = sum_h E[g,h] * min_{mon in F_h} <mon, m>  +  <mu_g, m> ,

is an exact integer linear problem; its solution is u_g = y^mu * prod_h F_h^E.
Three things are then checked, none of them put in:

  * the same method reproduces the hexagon's u-variables of the disk module;
  * on the punctured square (type D4, the one-loop disk, tagged arcs among its
    16 cluster variables) the u-equations hold exactly with integer exponents
    that are 0 exactly on compatible pairs;
  * on the once-punctured torus (the Markov quiver) the same ansatz has no
    solution, or none that is canonical -- see :func:`torus_ansatz`.
"""

import itertools
import random
from fractions import Fraction

import numpy as np
import sympy as sp

from . import sixsphere_surface as SF

# exchange matrices ---------------------------------------------------------
B_A2 = sp.Matrix([[0, 1], [-1, 0]])
B_A3 = sp.Matrix([[0, 1, 0], [-1, 0, 1], [0, -1, 0]])
#: oriented 4-cycle: the fan triangulation of the once-punctured square
B_D4 = sp.Matrix([[0, 1, 0, -1], [-1, 0, 1, 0], [0, -1, 0, 1], [1, 0, -1, 0]])


def _mutate_extended(Bt, k):
    out = Bt.copy()
    for i in range(Bt.rows):
        for j in range(Bt.cols):
            if i == k or j == k:
                out[i, j] = -Bt[i, j]
            else:
                out[i, j] = Bt[i, j] + (abs(Bt[i, k]) * Bt[k, j] + Bt[i, k] * abs(Bt[k, j])) // 2
    return out


def principal_data(B, depth=None):
    """All cluster variables (principal coefficients) within ``depth``
    mutations, or the whole exchange graph if ``depth`` is None (finite type).

    Returns (data, clusters): data[v] = {"expr", "F", "g", "mons"}, clusters a
    list of tuples of variable indices."""
    n = B.rows
    X = sp.symbols("x0:%d" % n, positive=True)
    Y = sp.symbols("y0:%d" % n, positive=True)
    key = lambda s: frozenset(sp.srepr(e) for e in s)                   # noqa: E731
    Bt0 = B.col_join(sp.eye(n))
    seeds = {key(X): (tuple(X), Bt0)}
    layer, d = [(tuple(X), Bt0)], 0
    while layer and (depth is None or d < depth):
        d += 1
        nxt = []
        for s, Bt in layer:
            for k in range(n):
                full = list(s) + list(Y)
                col = [Bt[i, k] for i in range(2 * n)]
                pos = sp.Mul(*[full[i] ** max(col[i], 0) for i in range(2 * n)])
                neg = sp.Mul(*[full[i] ** max(-col[i], 0) for i in range(2 * n)])
                new = list(s)
                new[k] = sp.expand(sp.cancel((pos + neg) / s[k]))
                kk = key(new)
                if kk not in seeds:
                    seeds[kk] = (tuple(new), _mutate_extended(Bt, k))
                    nxt.append(seeds[kk])
        layer = nxt
    ids, data, clusters = {}, [], []
    for s, _ in seeds.values():
        cl = []
        for e in s:
            r = sp.srepr(e)
            if r not in ids:
                ids[r] = len(data)
                P = sp.Poly(sp.expand(e * sp.Mul(*X) ** 60), *(tuple(X) + tuple(Y)))
                mon = P.monoms()[0]
                a, c = [mon[i] - 60 for i in range(n)], mon[n:]
                g = tuple(a[i] - sum(B[i, j] * c[j] for j in range(n)) for i in range(n))
                F = sp.expand(e.subs({x: 1 for x in X}))
                data.append({"expr": e, "F": F, "g": g,
                             "mons": sp.Poly(F, *Y).monoms()})
            cl.append(ids[r])
        clusters.append(tuple(cl))
    return data, clusters


def _valuation_system(data, clusters, n, grid, only_covered, sign=-1):
    nonunit = [v for v in range(len(data)) if data[v]["F"] != 1]
    cones = []
    for cl in clusters:
        R = np.array([[sign * data[v]["g"][i] for v in cl] for i in range(n)], float)
        if abs(np.linalg.det(R)) > 1e-9:
            cones.append((cl, np.linalg.inv(R)))
    pts, rows, keep = [m for m in itertools.product(range(-grid, grid + 1), repeat=n) if any(m)], [], []
    nu = np.zeros((len(pts), len(data)))
    for pi, m in enumerate(pts):
        mv = np.array(m, float)
        for cl, Rinv in cones:
            a = Rinv @ mv
            if (a > -1e-9).all():
                for v, av in zip(cl, a):
                    nu[pi, v] = max(av, 0)
                keep.append(pi)
                break
    if only_covered:
        pts = [pts[i] for i in keep]
        nu = nu[keep]
    V = np.array([[min(float(np.dot(mon, m)) for mon in data[v]["mons"]) for v in nonunit]
                  for m in pts])
    A = np.hstack([V, np.array(pts, float)])
    return nonunit, pts, nu, A, len(keep)


def solve_u(B, depth=None, grid=2, only_covered=False, sign=-1):
    """Fit u_g = y^mu prod F_h^E for every variable g.

    Returns a dict with ``data``, ``clusters``, ``nonunit``, ``E`` (list of
    integer coefficient vectors, None where the ansatz is inconsistent or not
    integral), ``consistent``, ``integral``, ``rank`` and ``unknowns``."""
    n = B.rows
    data, clusters = principal_data(B, depth)
    nonunit, pts, nu, A, covered = _valuation_system(data, clusters, n, grid, only_covered, sign)
    rank = int(np.linalg.matrix_rank(A))
    E, consistent, integral = {}, {}, {}
    for v in range(len(data)):
        sol = np.linalg.lstsq(A, nu[:, v], rcond=None)[0]
        consistent[v] = bool(np.abs(A @ sol - nu[:, v]).max() < 1e-8)
        coef = np.rint(sol)
        integral[v] = bool(np.abs(sol - coef).max() < 1e-6)
        E[v] = [int(c) for c in coef] if consistent[v] and integral[v] else None
    return {"data": data, "clusters": clusters, "nonunit": nonunit, "E": E,
            "consistent": consistent, "integral": integral, "rank": rank,
            "unknowns": A.shape[1], "points": len(pts), "covered": covered}


def _mon(y, m):
    r = Fraction(1)
    for yi, e in zip(y, m):
        r *= Fraction(yi) ** e
    return r


def u_values(sol, y):
    """Exact values of all u_g at the rational point y."""
    data, nonunit, E = sol["data"], sol["nonunit"], sol["E"]
    FV = {v: sum(_mon(y, m) for m in data[v]["mons"]) for v in nonunit}
    out = {}
    for g, coef in E.items():
        val = Fraction(1)
        for i, c in enumerate(coef):
            if c:
                val *= (FV[nonunit[i]] if i < len(nonunit) else Fraction(y[i - len(nonunit)])) ** c
        out[g] = val
    return out


def u_equation_exponents(sol, trials=80, seed=1):
    """The exponents C[g,h] of the u-equations u_g + prod_h u_h^C[g,h] = 1,
    found from log(1-u_g) = sum_h C log u_h by least squares at random rational
    points, rounded, and then verified exactly at fresh points.

    Returns (C, exact) with ``exact`` True if every u-equation holds exactly at
    12 independent rational points."""
    n, N = len(sol["data"][0]["g"]), len(sol["data"])
    rnd = random.Random(seed)
    pts = [[Fraction(rnd.randint(1, 9), rnd.randint(1, 9)) for _ in range(n)] for _ in range(trials)]
    U = [u_values(sol, y) for y in pts]
    L = np.array([[np.log(float(u[h])) for h in range(N)] for u in U])
    C = np.zeros((N, N))
    for g in range(N):
        b = np.array([np.log(1 - float(u[g])) for u in U])
        mask = [h for h in range(N) if h != g]
        C[g, mask] = np.linalg.lstsq(L[:, mask], b, rcond=None)[0]
    Cr = np.rint(C).astype(int)
    rnd = random.Random(99)
    exact = np.abs(C - Cr).max() < 1e-6
    for _ in range(12):
        y = [Fraction(rnd.randint(1, 20), rnd.randint(1, 20)) for _ in range(n)]
        u = u_values(sol, y)
        for g in range(N):
            rhs = Fraction(1)
            for h in range(N):
                if Cr[g, h]:
                    rhs *= u[h] ** int(Cr[g, h])
            exact &= (u[g] + rhs == 1)
    return Cr, bool(exact)


def finite_type_report(B):
    """Solve and verify the u-equations of a finite-type cluster algebra."""
    sol = solve_u(B, depth=None, grid=2)
    N = len(sol["data"])
    unique = sol["rank"] == sol["unknowns"]
    ok = all(sol["E"][v] is not None for v in range(N))
    Cr, exact = u_equation_exponents(sol)
    comp = np.zeros((N, N), bool)
    for cl in sol["clusters"]:
        for a in cl:
            for b in cl:
                comp[a, b] = a != b
    zero_on_compat = bool(((Cr == 0) == (comp | np.eye(N, dtype=bool))).all())
    terms = max(sum(1 for i, c in enumerate(sol["E"][v]) if c and i < len(sol["nonunit"]))
                for v in range(N)) if ok else None
    return {"variables": N, "clusters": len(sol["clusters"]), "unique": unique,
            "all_solved": ok, "u_equations_exact": exact,
            "C_symmetric": bool((Cr == Cr.T).all()), "C_zero_iff_compatible": zero_on_compat,
            "C_values": sorted(set(Cr.flatten().tolist())), "max_F_factors": terms,
            "exponents_pm1": ok and all(abs(c) <= 1 for v in range(N) for c in sol["E"][v]),
            "sol": sol, "C": Cr}


def hexagon_match():
    """The method on A3 against the disk module's u-variables of the hexagon,
    as multisets of exact values under the relabelling y_k <-> y_{n-k}."""
    from .theories import surfaceology as DS
    sol = solve_u(B_A3, depth=None, grid=2)
    y = DS.y_symbols(6)
    us = DS.u_variables(6, y)
    keys = sorted(y)
    rnd = random.Random(5)
    vals = [Fraction(rnd.randint(1, 9), rnd.randint(1, 9)) for _ in range(3)]
    mine = sorted(sp.Rational(v.numerator, v.denominator) for v in u_values(sol, vals).values())
    for perm in itertools.permutations(range(3)):
        sub = {y[k]: sp.Rational(vals[perm[i]].numerator, vals[perm[i]].denominator)
               for i, k in enumerate(keys)}
        if sorted(sp.nsimplify(u.subs(sub)) for u in us.values()) == mine:
            return {"match": True, "relabelling": perm, "variables": len(mine)}
    return {"match": False, "variables": len(mine)}


def punctured_square():
    """Type D4: the oriented 4-cycle is the fan triangulation of the
    once-punctured square, the four initial variables are the plain arcs P_i
    to the puncture, and the d-vector (= row of C on the initial variables)
    sorts the other twelve into 4 notched arcs N_j (zero at j, ones elsewhere)
    and 8 chords (cyclic intervals of length 1 and 2).  Reads off u(P_i), u(N_j).
    """
    rep = finite_type_report(B_D4)
    sol, C = rep["sol"], rep["C"]
    data, E, nonunit = sol["data"], sol["E"], sol["nonunit"]
    init = [v for v in range(len(data)) if data[v]["F"] == 1]
    d = {v: tuple(int(C[v, i]) for i in init) for v in range(len(data)) if v not in init}
    notched = {v: d[v].index(0) for v in d if sorted(d[v]) == [0, 1, 1, 1]}
    chords = {v: tuple(i for i in range(4) if d[v][i]) for v in d if v not in notched}
    shapes = sorted(len(s) for s in chords.values())

    def factors(v):
        pos = [nonunit[i] for i, c in enumerate(E[v]) if i < len(nonunit) and c == 1]
        neg = [nonunit[i] for i, c in enumerate(E[v]) if i < len(nonunit) and c == -1]
        ys = [i - len(nonunit) for i, c in enumerate(E[v]) if i >= len(nonunit) and c]
        return pos, neg, ys

    flip_ok = True
    for i, p in enumerate(init):
        pos, neg, ys = factors(p)
        want_den = [v for v, j in notched.items() if j == (i + 1) % 4]
        want_num = [v for v, s in chords.items()
                    if len(s) == 2 and set(s) == {(i + 2) % 4, (i + 3) % 4}]
        flip_ok &= (neg == want_den and pos == want_num and not ys)
    n_ok = True
    for v, j in notched.items():
        pos, neg, ys = factors(v)
        want_num = [w for w, s in chords.items()
                    if len(s) == 2 and set(s) == {(j + 2) % 4, (j + 3) % 4}]
        n_ok &= (neg == [v] and pos == want_num and ys == [(j + 1) % 4] and E[v][len(nonunit) + (j + 1) % 4] == 1)
    return {"report": rep, "plain": len(init), "notched": len(notched),
            "chords": len(chords), "chord_shapes": shapes,
            "plain_flips_tag_and_shifts": bool(flip_ok),
            "notched_pattern": bool(n_ok)}


def torus_ansatz(depth=3, grid=3):
    """The same ansatz on the once-punctured torus (the Markov quiver).

    (a) Demanding nu = 0 off the g-fan (as in finite type, where the fan is
        complete and there is no 'off'): count the variables with no solution.
    (b) Fitting on the g-fan only: is the solution unique?  (rank against the
        number of unknowns; a rank deficit means the fit is not canonical.)
    """
    B = SF.dual_quiver()
    a = solve_u(B, depth=depth, grid=grid, only_covered=False)
    b = solve_u(B, depth=depth, grid=grid, only_covered=True)
    N = len(a["data"])
    init = [v for v in range(N) if a["data"][v]["F"] == 1]
    return {"depth": depth, "variables": N,
            "off_fan_unsolved": sum(1 for v in range(N) if a["E"][v] is None),
            "initial_unsolved": all(a["E"][v] is None for v in init),
            "fan_only_unknowns": b["unknowns"], "fan_only_rank": b["rank"],
            "covered_points": a["covered"], "points": a["points"]}
