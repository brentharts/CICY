r"""
pyCICY.sixsphere_search -- which lattice data can build a complex six-sphere?

Alpoge's construction stands on a representation of the (3, 4, inf)
triangle group on Z^4 (:mod:`pyCICY.sixsphere`). Two questions follow, and
both are finite enough to compute:

1. **Which triangle groups?** Could (2, 3, inf), (3, 6, inf) or (4, 6, inf)
   carry the same construction?
2. **Which lattice data?** Within (3, 4, inf), is the paper's pair (T1, T2)
   one member of a family, or forced?

The class searched
------------------
Everything here is restricted to the *flag class*: pairs (T1, T2) in
SL(4, Z) of the shape Lemma 2.7 (iii) proves the paper's matrices have. Each
generator fixes gamma, preserves <gamma, u, w>, and acts trivially on the top
quotient, so in the basis (gamma, u, w, delta)

    T = [[1, r, s], [0, M, c], [0, 0, 1]],   M in SL(2, Z), r a row, c a column,

with M the Kodaira monodromy of the elliptic surface behind tau (Part II).
Data outside the flag class is not searched, and nothing here says anything
about it.

Answer 1: only (3, 4, inf)
--------------------------
* A finite-order element of SL(2, Z) has order 1, 2, 3, 4 or 6, so the
  elliptic orders m_j lie in {2, 3, 4, 6} (enumerated, not quoted).
* The only element of order 2 is -I, which is central; the cusp
  M0 = (M1 M2)^{-1} can then never be unipotent. Every (2, m) is out.
* Of the remaining hyperbolic pairs, only (3, 4), (3, 6) and (4, 6) admit a
  middle pair with a *unipotent* cusp. For (3, 3) and (6, 6) the product is
  parabolic only as -(unipotent), an I_2* cusp; for (4, 4) not at all.
  (Observed, stable across boxes.)
* The Theorem 7.17 presentation gives, for general orders,
  p = m1 m2 l0 - m2 l1 - m1 l2, divisible by gcd(m1, m2). So |p| = 1 needs
  coprime orders, which leaves (3, 4) alone. (A congruence: proved.)

Answer 2: the paper's data, and three near misses
-------------------------------------------------
Block-unipotent conjugation acts by r -> r + x(M - I), c -> c + (I - M)y
with the *same* x, y for both generators (derived symbolically by
:func:`conjugation_action`). Since M2 - I is invertible over Q with
|det| = 2, T2's r and c reduce to 2 x 2 residue classes, and nothing is left
to normalise T1: its r and c are free, and the extensions form infinite
families. :func:`funnel` searches them in a stated box and applies the S^6
conditions in stages, reporting the count at every stage:

    T1^3 = T2^4 = I  ->  unipotent cusp, N^2 = 0, rank N = 2
      ->  coinvariants exactly Z  ->  invariant closure of Lambda_tor = ker gamma
      ->  |det B0| = 1 (one dP6 at the cusp)

The survivors fall into four GL(4, Z) conjugacy classes (:func:`classify`).
They are distinguished by an invariant -- the set of |p| values reachable by
admissible twists, taken mod m1 m2 -- so their distinctness is proved, not
searched for; within a class the conjugators are explicit and verified.
Exactly one class can reach |p| = 1, and it is conjugate to the paper's
matrices (:func:`paper_conjugator` returns the matrix). The other three
bottom out at |p| = 3, 2 and 6 (the last also reaching p = 0).

What is proved, what is observed
--------------------------------
Proved: the finite-order list; the -I obstruction; the gcd congruence; the
distinctness of the four classes; every membership and the conjugator to the
paper's data. Observed: the emptiness of (3, 3), (4, 4), (6, 6) for unipotent
cusps, and the completeness of the four classes -- :func:`stability` shows the
final count unchanged as the box grows while earlier stages keep growing, but
a finite box is not a proof. Declined: everything analytic. Whether the three
near-miss classes support period functions, and so a complex manifold at
all, is not a lattice question.
"""

import itertools
from math import gcd

import numpy as np
from sympy import Matrix, symbols, linsolve, eye, zeros, Rational
from sympy.matrices.normalforms import invariant_factors

from . import sixsphere as S

__all__ = ["sl2_finite_orders", "middle_pairs", "triangle_group_survey",
           "gcd_obstruction", "conjugation_action", "residue_reps",
           "solve_corner", "funnel", "admissible_ells", "p_residues",
           "classify", "find_conjugator", "paper_conjugator", "stability",
           "status_search", "stored_constants_agree", "CLASS_REP",
           "PAPER_CONJUGATOR", "MIRROR_CONJUGATOR",
           # connections
           "laves_is_cusp_hasse", "laves_cusp_dictionary", "lattice_mirror",
           "dual_lattice", "field_comparison", "coincidences",
           "euclidean_triangle_groups", "hofstadter_link",
           "status_connections"]

_I2 = np.eye(2, dtype=np.int64)
_I4 = np.eye(4, dtype=np.int64)

#: The representative, in the normal form of :func:`funnel`, of the one class
#: that reaches |p| = 1; the conjugator carrying the paper's (T1, T2) onto
#: it; and the conjugator from (T1, T2) to (T1^-1, T2^-1). Stored so that
#: :mod:`pyCICY.sixsphere_lean` can state them without running the search;
#: :func:`stored_constants_agree` recomputes all three from scratch.
CLASS_REP = (Matrix([[1, -3, 0, -2], [0, -1, 1, -2], [0, -1, 0, 0],
                     [0, 0, 0, 1]]),
             Matrix([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 1],
                     [0, 0, 0, 1]]))
PAPER_CONJUGATOR = Matrix([[1, 3, -3, -2], [0, 1, 0, 0], [0, 0, 1, -1],
                           [0, 0, 0, -1]])
MIRROR_CONJUGATOR = Matrix([[1, 6, -6, -2], [0, 0, 1, -1], [0, 1, 0, -1],
                            [0, 0, 0, -1]])


def _mpow(T, k):
    P = np.eye(len(T), dtype=np.int64)
    for _ in range(k):
        P = P @ T
    return P


def _order(M, bound=12):
    P = np.eye(len(M), dtype=np.int64)
    for k in range(1, bound + 1):
        P = P @ M
        if (P == np.eye(len(M), dtype=np.int64)).all():
            return k
    return None


# ---------------------------------------------------------------------------
# Answer 1: which triangle groups
# ---------------------------------------------------------------------------

def _sl2_elements(box):
    out = {}
    for a, b, c, d in itertools.product(range(-box, box + 1), repeat=4):
        if a * d - b * c != 1:
            continue
        M = np.array([[a, b], [c, d]], dtype=np.int64)
        o = _order(M)
        if o:
            out.setdefault(o, []).append(M)
    return out


def sl2_finite_orders(box=4):
    """The orders of finite-order elements of SL(2, Z), found by enumeration.

    {1, 2, 3, 4, 6}, the classical answer, with -I the only element of order 2.
    """
    els = _sl2_elements(box)
    return {"orders": sorted(els),
            "order_2_elements": [M.tolist() for M in els.get(2, [])]}


def middle_pairs(m1, m2, box=4):
    """Pairs (M1, M2) of orders (m1, m2) with M1 M2 = +-(unipotent) != +-I.

    Returns the pairs whose product is unipotent (an I_n cusp) and, separately,
    those whose product is minus a unipotent (I_n*), each with its n.
    """
    els = _sl2_elements(box)
    plus, minus = [], []
    for M1 in els.get(m1, []):
        for M2 in els.get(m2, []):
            P = M1 @ M2
            t = int(np.trace(P))
            if t == 2 and not (P == _I2).all():
                n = int(np.gcd.reduce(np.abs((P - _I2).ravel())))
                plus.append((M1, M2, n))
            if t == -2 and not (P == -_I2).all():
                n = int(np.gcd.reduce(np.abs((P + _I2).ravel())))
                minus.append((M1, M2, n))
    return {"unipotent": plus, "minus_unipotent": minus}


def gcd_obstruction(m1, m2):
    r"""|p| over all admissible twists, for general orders, by exhaustion mod m1 m2.

    p = m1 m2 l0 - m2 l1 - m1 l2 (Theorem 7.17's relation matrix with general
    orders); admissible means gcd(l_j, m_j) = 1. Runs over every residue
    class, so the minimum is exact, and it equals gcd(m1, m2) -- in
    particular |p| = 1 is reachable iff the orders are coprime.
    """
    n = m1 * m2
    values = set()
    for l1 in range(n):
        if gcd(l1, m1) != 1:
            continue
        for l2 in range(n):
            if gcd(l2, m2) != 1:
                continue
            x = (-m2 * l1 - m1 * l2) % n
            values.add(min(x, n - x))
    positive = sorted(v for v in values if v > 0)
    return {"m": (m1, m2), "gcd": gcd(m1, m2),
            "min_positive_|p|": positive[0] if positive else None,
            "reachable_residues": sorted(values),
            "all_divisible_by_gcd": all(v % gcd(m1, m2) == 0 for v in values)}


def triangle_group_survey(box=4):
    """Every hyperbolic (m1, m2, inf) with m_j in {2, 3, 4, 6}, and its verdict."""
    rows = []
    for m1, m2 in itertools.combinations_with_replacement((2, 3, 4, 6), 2):
        if 1 / m1 + 1 / m2 >= 1:
            continue
        mp = middle_pairs(m1, m2, box)
        g = gcd_obstruction(m1, m2)
        plus_n = sorted({n for _, _, n in mp["unipotent"]})
        minus_n = sorted({n for _, _, n in mp["minus_unipotent"]})
        if not mp["unipotent"]:
            verdict = ("no unipotent cusp" if not mp["minus_unipotent"]
                       else "cusp only I_n* (sign)")
        elif g["min_positive_|p|"] != 1:
            verdict = "|p| >= %d (gcd)" % g["min_positive_|p|"]
        else:
            verdict = "candidate"
        rows.append({"m": (m1, m2), "unipotent_pairs": len(mp["unipotent"]),
                     "cusp_n": plus_n,
                     "minus_unipotent_pairs": len(mp["minus_unipotent"]),
                     "star_cusp_n": minus_n, "gcd": g["gcd"],
                     "min_|p|": g["min_positive_|p|"], "verdict": verdict})
    return {"box": box, "rows": rows,
            "candidates": [r["m"] for r in rows if r["verdict"] == "candidate"]}


# ---------------------------------------------------------------------------
# Answer 2: the extensions within (3, 4, inf)
# ---------------------------------------------------------------------------

def _blk(r, M, c, s):
    T = np.eye(4, dtype=np.int64)
    T[0, 1:3] = r
    T[1:3, 1:3] = M
    T[1:3, 3] = c
    T[0, 3] = s
    return T


def conjugation_action():
    r"""How U = [[1, x, z], [0, I, y], [0, 0, 1]] acts on (r, M, c), symbolically.

    Returns the increments r' - r and c' - c and whether M is unchanged:
    r' = r + x(M - I), c' = c + (I - M) y. The same x, y act on every
    generator at once, which is what makes T1's data unnormalisable once T2's
    is fixed.
    """
    from sympy import simplify, expand
    x0, x1, y0, y1, z, r0, r1, c0, c1, s = symbols(
        "x0 x1 y0 y1 z r0 r1 c0 c1 s")
    a, b, c, d = symbols("a b c d")

    def blk(r, M, cc, s_):
        return Matrix([[1, r[0], r[1], s_], [0, M[0, 0], M[0, 1], cc[0]],
                       [0, M[1, 0], M[1, 1], cc[1]], [0, 0, 0, 1]])

    M = Matrix([[a, b], [c, d]])
    T = blk((r0, r1), M, (c0, c1), s)
    U = blk((x0, x1), eye(2), (y0, y1), z)
    C = simplify(U * T * U.inv())
    dr = [expand(C[0, 1] - r0), expand(C[0, 2] - r1)]
    dc = [expand(C[1, 3] - c0), expand(C[2, 3] - c1)]
    xr = Matrix([[x0, x1]]) * (M - eye(2))
    yc = (eye(2) - M) * Matrix([y0, y1])
    return {"dr": dr, "dc": dc,
            "dr = x(M - I)": [expand(u - v) for u, v in zip(dr, xr)] == [0, 0],
            "dc = (I - M)y": [expand(u - v) for u, v in zip(dc, yc)] == [0, 0],
            "M unchanged": simplify(C[1:3, 1:3] - M) == zeros(2)}


def residue_reps(M, side="row", box=3):
    """Representatives of Z^2 / Z^2 (M - I) (rows) or Z^2 / (I - M) Z^2 (columns)."""
    D = (M - _I2) if side == "row" else (_I2 - M)
    reps = []

    def same(v, w):
        A = D.T if side == "row" else D
        sol = np.linalg.solve(A.astype(float), (v - w).astype(float))
        return np.allclose(sol, np.round(sol))

    for v in itertools.product(range(box), repeat=2):
        v = np.array(v, dtype=np.int64)
        if not any(same(v, w) for w in reps):
            reps.append(v)
    return reps


def solve_corner(r, M, c, m):
    """The unique corner s with T^m = I, or None if it is not an integer."""
    f = _mpow(_blk(r, M, c, 0), m)[0, 3]      # corner of T^m is m s + f
    if f % m:
        return None
    T = _blk(r, M, c, -f // m)
    return T if (_mpow(T, m) == _I4).all() else None


def _inv_int(T):
    Ti = np.round(np.linalg.inv(T)).astype(np.int64)
    if not (Ti @ T == _I4).all():
        raise ArithmeticError("not unimodular")
    return Ti


def _sym(T):
    return Matrix(np.asarray(T).tolist())


def _free(A, m, v):
    """Freeness of x -> A x + v/m, as in :func:`sixsphere.log_transform_free`,
    for a general order m."""
    I = eye(4)
    if A * v != v:
        return False
    for k in range(1, m):
        tot, P = zeros(4, 1), I
        for _ in range(k):
            tot += P * v
            P = P * A
        ann = S.int_kernel((A ** k - I).T)
        if all(Rational(x).q == 1 for x in ann * (tot / m)):
            return False
    return True


def _closure_is_ker_gamma(A1, A2, tor):
    L = tor
    while True:
        new = L
        for A in (A1, A2):
            new = new.col_join((A * L.T).T)
        new = S.saturate_basis(new)
        if new.rows == L.rows and S.same_lattice(new, L):
            break
        L = new
    return S.same_lattice(L, Matrix([[0, 1, 0, 0], [0, 0, 1, 0],
                                     [0, 0, 0, 1]]))


def funnel(box=5, m1=3, m2=4, M2=None, m1_box=6):
    r"""Search the flag-class extensions in a box; count survivors stage by stage.

    M2 is fixed (default: the paper's middle, S = [[0, -1], [1, 0]]); every
    order-4 element is conjugate to S in GL(2, Z), and the middle
    conjugations in GL(4, Z) are allowed. M1 runs over the order-m1 elements
    in ``m1_box`` whose product with M2 is unipotent with n = 1. T2's (r, c)
    run over residue representatives; T1's over [-box, box]^4.
    """
    M2 = np.array([[0, -1], [1, 0]], dtype=np.int64) if M2 is None else M2
    M1s = [M1 for M1, M2_, n in middle_pairs(m1, m2, m1_box)["unipotent"]
           if (M2_ == M2).all() and n == 1]
    rreps, creps = residue_reps(M2, "row"), residue_reps(M2, "col")
    counts = {"tried": 0, "orders": 0, "cusp": 0, "coinvariants_Z": 0,
              "closure_ker_gamma": 0, "unimodular_B0": 0}
    survivors = []
    for M1 in M1s:
        for r2 in rreps:
            for c2 in creps:
                T2 = solve_corner(r2, M2, c2, m2)
                if T2 is None:
                    continue
                for r1 in itertools.product(range(-box, box + 1), repeat=2):
                    for c1 in itertools.product(range(-box, box + 1), repeat=2):
                        counts["tried"] += 1
                        T1 = solve_corner(np.array(r1), M1, np.array(c1), m1)
                        if T1 is None:
                            continue
                        counts["orders"] += 1
                        N = _inv_int(T1 @ T2) - _I4
                        if not (N @ N == 0).all() or np.linalg.matrix_rank(N) != 2:
                            continue
                        counts["cusp"] += 1
                        T1s, T2s = _sym(T1), _sym(T2)
                        I = eye(4)
                        f = sorted(abs(int(x)) for x in
                                   invariant_factors((T1s - I).row_join(T2s - I)))
                        if f != [0, 1, 1, 1]:
                            continue
                        counts["coinvariants_Z"] += 1
                        A1, A2 = T1s.inv().T, T2s.inv().T
                        M0 = (T1s * T2s).T           # A(T0) = (T0^{-1})^t, T0^{-1} = T1 T2
                        tor = S.int_kernel(M0 - I)
                        if not _closure_is_ker_gamma(A1, A2, tor):
                            continue
                        counts["closure_ker_gamma"] += 1
                        detB0 = 1
                        for x in invariant_factors(M0 - I):
                            if x != 0:
                                detB0 *= abs(int(x))
                        if detB0 != 1:
                            continue
                        counts["unimodular_B0"] += 1
                        survivors.append((T1, T2))
    return {"box": box, "m": (m1, m2), "M1_choices": [M.tolist() for M in M1s],
            "T2_residue_reps": {"r": [v.tolist() for v in rreps],
                                "c": [v.tolist() for v in creps]},
            "counts": counts, "survivors": survivors}


def admissible_ells(T1, T2, m1=3, m2=4, box=3):
    """gamma(v) for admissible twist vectors v at each point (freeness computed)."""
    T1s, T2s = _sym(T1), _sym(T2)
    out = []
    for A, m in ((T1s.inv().T, m1), (T2s.inv().T, m2)):
        F = S.int_kernel(A - eye(4))
        vals = set()
        for co in itertools.product(range(-box, box + 1), repeat=F.rows):
            v = zeros(4, 1)
            for i, cf in enumerate(co):
                v += cf * F.row(i).T
            if _free(A, m, v):
                vals.add(int(v[0]))
        out.append(vals)
    return out


def p_residues(ells, m1=3, m2=4):
    """The reachable |p| mod m1 m2 (as min(x, n - x)): a conjugacy invariant."""
    n = m1 * m2
    return frozenset(min((-m2 * a - m1 * b) % n, n - (-m2 * a - m1 * b) % n)
                     for a in ells[0] for b in ells[1])


def find_conjugator(Ta, Tb, coeff_box=2):
    """An integer P, det +-1, with P Ta_j P^{-1} = Tb_j, or None if not found.

    Solves the linear system P Ta_j = Tb_j P exactly and searches its integer
    points in a box; a None is *not* a proof of non-conjugacy (the invariant
    of :func:`p_residues` is what proves that).
    """
    P = Matrix(4, 4, symbols("p0:16"))
    eqs = []
    for A, B in zip(Ta, Tb):
        eqs += list(P * _sym(A) - _sym(B) * P)
    sol = list(linsolve(eqs, list(P)))[0]
    free = sorted(set().union(*[e.free_symbols for e in sol]), key=str)
    for co in itertools.product(range(-coeff_box, coeff_box + 1),
                                repeat=len(free)):
        Pm = Matrix(4, 4, [e.subs(dict(zip(free, co))) for e in sol])
        if all(x == int(x) for x in Pm) and abs(Pm.det()) == 1:
            return Pm
    return None


def classify(survivors, m1=3, m2=4):
    """Group survivors into conjugacy classes; verify every membership.

    First by the invariant (reachable |p| residues), which proves classes
    with different invariants are distinct; then, within an invariant, by
    explicit conjugators, each verified.
    """
    classes = []
    for T1, T2 in survivors:
        inv = p_residues(admissible_ells(T1, T2, m1, m2), m1, m2)
        placed = False
        for cl in classes:
            if cl["invariant"] != inv:
                continue
            P = find_conjugator((T1, T2), cl["rep"])
            if P is not None:
                cl["members"].append({"T": (T1, T2), "P": P})
                placed = True
                break
        if not placed:
            classes.append({"invariant": inv, "rep": (T1, T2),
                            "members": [{"T": (T1, T2), "P": eye(4)}]})
    for cl in classes:
        rep = tuple(_sym(T) for T in cl["rep"])
        cl["verified"] = all(
            m["P"] * _sym(m["T"][0]) * m["P"].inv() == rep[0]
            and m["P"] * _sym(m["T"][1]) * m["P"].inv() == rep[1]
            for m in cl["members"])
        cl["reaches_|p|=1"] = 1 in cl["invariant"]
    return sorted(classes, key=lambda c: sorted(c["invariant"]))


def paper_conjugator(classes=None, box=5):
    """The explicit integer P, det +-1, carrying the paper's (T1, T2) onto the
    representative of the class that reaches |p| = 1."""
    if classes is None:
        classes = classify(funnel(box)["survivors"])
    target = [c for c in classes if c["reaches_|p|=1"]]
    if len(target) != 1:
        return {"unique_class": False, "P": None}
    rep = target[0]["rep"]
    paper = (np.array(S.T1.tolist(), dtype=np.int64),
             np.array(S.T2.tolist(), dtype=np.int64))
    P = find_conjugator(paper, rep)
    ok = (P is not None
          and P * S.T1 * P.inv() == _sym(rep[0])
          and P * S.T2 * P.inv() == _sym(rep[1]))
    return {"unique_class": True, "P": P, "det": P.det() if P is not None
            else None, "verified": ok,
            "paper_invariant": sorted(p_residues(admissible_ells(*paper)))}


def stored_constants_agree(box=5):
    """Recompute CLASS_REP, PAPER_CONJUGATOR and MIRROR_CONJUGATOR."""
    cls = classify(funnel(box)["survivors"])
    rep = [c for c in cls if c["reaches_|p|=1"]][0]["rep"]
    pc = paper_conjugator(cls)
    return {"class_rep": (_sym(rep[0]), _sym(rep[1])) == CLASS_REP,
            "paper_conjugator": pc["P"] == PAPER_CONJUGATOR,
            "mirror_conjugator": lattice_mirror()["P"] == MIRROR_CONJUGATOR}


def stability(boxes=(5, 8)):
    """The funnel's counts as the box grows: earlier stages grow, the last not."""
    return {b: funnel(b)["counts"] for b in boxes}


def status_search(box=5):
    """Run every Part VII check; pass/fail per item."""
    sv = triangle_group_survey()
    ca = conjugation_action()
    fn = funnel(box)
    cls = classify(fn["survivors"])
    pc = paper_conjugator(cls)
    checks = {
        "finite orders in SL2(Z) = {1,2,3,4,6}; -I the only involution":
            sl2_finite_orders()["orders"] == [1, 2, 3, 4, 6]
            and sl2_finite_orders()["order_2_elements"] == [[[-1, 0], [0, -1]]],
        "only (3,4,inf) survives the survey": sv["candidates"] == [(3, 4)],
        "|p| divisible by gcd(m1, m2), min = gcd":
            all(gcd_obstruction(a, b)["min_positive_|p|"] == gcd(a, b)
                for a, b in ((3, 4), (3, 6), (4, 6))),
        "conjugation action r + x(M-I), c + (I-M)y":
            ca["dr = x(M - I)"] and ca["dc = (I - M)y"] and ca["M unchanged"],
        "T2 data reduces to 2 x 2 residue classes":
            len(fn["T2_residue_reps"]["r"]) == 2
            and len(fn["T2_residue_reps"]["c"]) == 2,
        "funnel ends at 32 survivors": fn["counts"]["unimodular_B0"] == 32,
        "four classes, pairwise distinct by invariant":
            len(cls) == 4 and len({c["invariant"] for c in cls}) == 4,
        "every membership verified": all(c["verified"] for c in cls),
        "exactly one class reaches |p| = 1":
            sum(c["reaches_|p|=1"] for c in cls) == 1,
        "the paper's data is conjugate to it (explicit P, verified)":
            pc["verified"] and abs(pc["det"]) == 1,
    }
    return {"checks": checks, "all_pass": all(checks.values()),
            "counts": fn["counts"], "box": box,
            "classes": [{"invariant": sorted(c["invariant"]),
                         "size": len(c["members"])} for c in cls]}


# ===========================================================================
# Connections to other theories in the package
# ===========================================================================
#
# Each candidate connection is settled by a computation, in the manner of
# ninelink.monotile_compatibility ("the nines are unrelated"): two are
# structural, two are coincidences, and the rest are exact comparisons.

def _cells_A2_in_plane(R, a):
    """A2 vertices, edge midpoints and triangle centroids, embedded as the
    monotile's hexagon centres (e1 -> 2a(cos 30, sin 30), e2 -> 2a(0, 1))."""
    from fractions import Fraction as Fr
    from .monotile import Quad

    def emb(p):
        x, y = p
        return (a * Quad(0, 1, 3) * x, a * x + a * Quad(2, 0, 3) * y)

    def key(pt):
        return (pt[0].p, pt[0].q, pt[1].p, pt[1].q)

    kinds, inc = {}, set()
    for x in range(-R, R + 1):
        for y in range(-R, R + 1):
            kinds[key(emb((x, y)))] = "centre"
            for d in S.EDGE_DIRECTIONS:
                mid = (x + Fr(d[0], 2), y + Fr(d[1], 2))
                kinds[key(emb(mid))] = "edge"
                inc.add(frozenset((key(emb((x, y))), key(emb(mid)))))
                inc.add(frozenset((key(emb((x + d[0], y + d[1]))),
                                   key(emb(mid)))))
            for _, tri in S.a2_triangles((x, y)).items():
                c = (sum(Fr(v[0]) for v in tri) / 3,
                     sum(Fr(v[1]) for v in tri) / 3)
                kinds[key(emb(c))] = "vertex"
                for i in range(3):
                    p, q = tri[i], tri[(i + 1) % 3]
                    mid = (Fr(p[0] + q[0], 2), Fr(p[1] + q[1], 2))
                    inc.add(frozenset((key(emb(mid)), key(emb(c)))))
    return kinds, inc, key


def laves_is_cusp_hasse(rings=2):
    r"""The monotile's kite lattice is the Hasse diagram of the cusp fan.

    :func:`pyCICY.monotile.laves_patch` builds the [3.4.6.4] Laves lattice on
    which every Tile(a, b) -- the Hat, the Spectre -- is a polykite. At the
    regular proportions a = sqrt3 b its sites are, exactly in Q(sqrt3), the
    cells of the A2 triangulation of Part III: hexagon centres are A2
    vertices, edge midpoints are edge midpoints, hexagon vertices are
    triangle centroids. Its bonds are exactly the cover relations vertex <
    edge < triangle. So W, the cusp fibre, is this lattice rolled up on
    R^2 / B0 Z^2.
    """
    from . import monotile as Mo
    from .monotile import Quad
    a, b = Quad(0, 1, 3), Quad(1, 0, 3)
    P = Mo.laves_patch(a=a, b=b, rings=rings)
    kinds, inc, key = _cells_A2_in_plane(rings + 3, a)
    patch = {key(s): P["kind"][i] for i, s in enumerate(P["sites"])}
    bonds = {frozenset((key(P["sites"][i]), key(P["sites"][j])))
             for i, j, _ in P["bonds"]}
    inc_in_patch = {e for e in inc if e <= set(patch)}
    return {"sites": len(patch),
            "sites_are_A2_cells_of_the_right_kind":
                all(k in kinds and kinds[k] == v for k, v in patch.items()),
            "incidences": len(inc_in_patch), "bonds": len(bonds),
            "bonds_are_exactly_the_incidences": inc_in_patch == bonds}


def laves_cusp_dictionary():
    """Per fundamental domain of B0 Z^2: Laves features against strata of W."""
    c = S.cusp_fibre()
    hexagon_sides = 6
    kites = 6                                 # (vertex in triangle) flags
    return {
        "hexagon centre (valence 6) = component (dP6)": (1, c["components"]),
        "edge midpoint (valence 4) = double curve": (3, c["double_curves"]),
        "hexagon vertex (valence 3) = triple point": (2, c["triple_points"]),
        "kite = corner of the dP6 hexagon": (kites, len(
            S.star_of_origin()["triangles"])),
        "centre-edge bond = (-1)-curve of the hexagon": (hexagon_sides, len(
            S.hexagon_surface()["self_intersections"])),
        "edge-vertex bond = double curve through a triple point": (
            6, sum(len(p) for p in c["curves_through_triple_points"].values())),
        "e(W) = hexagon vertices per domain": (2, S.euler_W_stratified()),
        "V - E + F of the rolled-up lattice (a torus)": (0, 6 - 12 + 6),
    }


def lattice_mirror():
    r"""Is the lattice data chiral? No: an explicit conjugator, det 1.

    Reversing the base's orientation reverses every meridian, replacing the
    marking by its inverse (Remark 3.18): (T1, T2) -> (T1^{-1}, T2^{-1}).
    The two pairs are conjugate in SL(4, Z), so the monodromy cannot tell X
    from its orientation-reversed counterpart. That is a statement about the
    lattice, not a proof that X is biholomorphic to its conjugate. Returned
    also as a :mod:`pyCICY.chirality` record.
    """
    from .chirality import _record
    Ta = (np.array(S.T1.tolist(), dtype=np.int64),
          np.array(S.T2.tolist(), dtype=np.int64))
    Tb = (np.array(S.T1.inv().tolist(), dtype=np.int64),
          np.array(S.T2.inv().tolist(), dtype=np.int64))
    P = find_conjugator(Ta, Tb)
    ok = (P is not None and P * S.T1 * P.inv() == S.T1.inv()
          and P * S.T2 * P.inv() == S.T2.inv())
    rec = _record("complex six-sphere", "Alpoge (3,4,inf) lattice data",
                  "base orientation reversal", (1, 5), (1, 5), 12,
                  fixed=ok, detected=True, asymmetry=0,
                  note="(T1, T2) ~ (T1^-1, T2^-1) by an explicit P; "
                       "|p| residues {1, 5} are symmetric")
    return {"P": P, "det": P.det() if P is not None else None,
            "verified": ok, "achiral": ok, "record": rec}


def dual_lattice():
    r"""V and its dual Lambda = V^*: isomorphic over Q, not over Z.

    Q0 intertwines T with A = T^{-t} over Q (it is invariant), but det Q0 =
    36. Every integral intertwiner lies in a rank-2 lattice (s, t) and has
    determinant 36 s^4, which is never +-1: the monodromy lattice is not
    self-dual. A sharper form of 'no polarisation' -- not even a
    non-positive principal one exists.
    """
    from sympy import factor, Poly, gcd_list, ilcm, symbols as sy
    m = S.monodromy()
    P = Matrix(4, 4, sy("p0:16"))
    eqs = []
    for A, B in ((S.T1, m["A1"]), (S.T2, m["A2"])):
        eqs += list(P * A - B * P)
    sol = list(linsolve(eqs, list(P)))[0]
    free = sorted(set().union(*[e.free_symbols for e in sol]), key=str)
    basis = []
    for f in free:
        basis.append([e.subs({g: (1 if g == f else 0) for g in free})
                      for e in sol])
    Bm = Matrix(basis)
    d = ilcm(*[Rational(x).q for x in Bm])
    Z = S.saturate((Bm * d).applyfunc(int).tolist())
    s, t = sy("s t")
    gen = Matrix(4, 4, list(s * Z.row(0) + t * Z.row(1)))
    det = factor(gen.det())
    return {"Q0_intertwines": all(S.Q0 * T == A * S.Q0 for T, A in
                                  ((S.T1, m["A1"]), (S.T2, m["A2"]))),
            "det_Q0": S.Q0.det(), "intertwiner_rank": Z.rows,
            "det_form": det,
            "content": gcd_list(Poly(gen.det(), s, t).coeffs()),
            "self_dual_over_Z": False if gcd_list(
                Poly(gen.det(), s, t).coeffs()) > 1 else None}


def field_comparison():
    r"""Quadratic square classes of the six-sphere's fields and the monotiles'.

    Signed classes (:func:`ninelink.square_classes`, with the sign kept).
    The six-sphere's elliptic values tau(z1) = rho, tau(z2) = i generate
    Q(sqrt-3, i) = Q(zeta_12); its lattice form b = 6w^2 - d^2 has isotropic
    slopes +-sqrt6. The monotiles' tile geometry is in Q(sqrt3), the Hat's
    inflation in Q(sqrt5), the Spectre's lambda = (sqrt6 + sqrt10)/2 in
    Q(sqrt6, sqrt10).
    """
    from .theories.ninelink import square_classes as sq
    from sympy import sqrt, simplify
    F = {"S6 periods Q(sqrt-3, i)": set(sq([-3, -1])),
         "S6 lattice Q(sqrt6)": set(sq([6])),
         "monotile geometry Q(sqrt3)": set(sq([3])),
         "Hat inflation Q(sqrt5)": set(sq([5])),
         "Spectre inflation Q(sqrt6, sqrt10)": set(sq([6, 10]))}
    per, lat, geo, hat, spe = F.values()
    return {"classes": {k: sorted(v) for k, v in F.items()},
            "periods & geometry": sorted(per & geo),
            "periods & Hat": sorted(per & hat),
            "periods & Spectre": sorted(per & spe),
            "lattice & Spectre": sorted(lat & spe),
            "lambda^2 = 4 + sqrt15": simplify(
                ((sqrt(6) + sqrt(10)) / 2) ** 2 - 4 - sqrt(15)) == 0}


def coincidences():
    r"""Two resemblances, each settled as unrelated.

    * sqrt6. b's isotropic slopes and the Spectre's lambda share the class 6.
      The six-sphere's 6 is Q0(u, w), forced by the -6 in T1 (the 6 mu shape
      of Part II); the Spectre's is from lambda^2 = 4 + sqrt15. Nothing
      downstream of either uses the other.
    * valence 4 and order 4. The Laves valences (6, 4, 3) echo the orbifold
      orders (3, 4, inf), but the valence-4 sites are the double curves,
      which sit over the cusp p0 -- not over p2, the order-4 point.
    """
    fc = field_comparison()
    return {"sqrt6_shared": fc["lattice & Spectre"] == [1, 6],
            "sqrt6_origin_S6": "Q0(u, w) = 6, from T1[0][2] = %d"
                               % S.T1[0, 2],
            "sqrt6_origin_Spectre": "lambda^2 = 4 + sqrt15",
            "valence4_sites_lie_over": "p0 (the cusp): they are the double "
                                       "curves of W",
            "order4_point": "p2, a multiple fibre 4 S2 with no double curves"}


def euclidean_triangle_groups(bound=12):
    """(p, q, r) with 1/p + 1/q + 1/r = 1; none contains both 3 and 4, so the
    (3, 4, inf) base is necessarily hyperbolic, while the monotile's
    substrate symmetry (2, 3, 6) is Euclidean."""
    from fractions import Fraction as Fr
    eu = [(p, q, r) for p in range(2, bound) for q in range(p, bound)
          for r in range(q, bound) if Fr(1, p) + Fr(1, q) + Fr(1, r) == 1]
    return {"euclidean": eu,
            "any_with_3_and_4": any(3 in t and 4 in t for t in eu)}


def hofstadter_link():
    """The cusp component's polygon is toric's B3 (= dP6), whose hoppings are
    the triangular lattice's: the mirror curve of the component is the
    triangular-lattice Hofstadter model of :mod:`pyCICY.quantum_curve`."""
    from . import toric
    hops = set(toric.hoppings(toric.polygon("B3")))
    return {"B3_hoppings": sorted(hops),
            "equal_to_hexagon_rays": hops == set(S.HEXAGON_RAYS),
            "twelve": toric.twelve(toric.polygon("B3"))}


def status_connections():
    """Run every connection check; pass/fail per item."""
    lv = laves_is_cusp_hasse()
    dic = laves_cusp_dictionary()
    mir = lattice_mirror()
    du = dual_lattice()
    fc = field_comparison()
    co = coincidences()
    from sympy import symbols as sy
    s = sy("s")
    checks = {
        "Laves sites are the A2 cells, kinds matching (exact, Q(sqrt3))":
            lv["sites_are_A2_cells_of_the_right_kind"],
        "Laves bonds are exactly the A2 cover relations":
            lv["bonds_are_exactly_the_incidences"],
        "per-domain dictionary with W agrees":
            all(a == b for a, b in dic.values()),
        "lattice data achiral: (T1,T2) ~ (T1^-1,T2^-1), det 1":
            mir["verified"] and mir["det"] == 1,
        "V not self-dual over Z: det of intertwiners = 36 s^4":
            du["det_form"] == 36 * s ** 4 and du["Q0_intertwines"],
        "period field Q(zeta12) = classes {-3,-1,1,3}":
            fc["classes"]["S6 periods Q(sqrt-3, i)"] == [-3, -1, 1, 3],
        "shares Q(sqrt3) with monotile geometry, nothing with inflations":
            fc["periods & geometry"] == [1, 3] and fc["periods & Hat"] == [1]
            and fc["periods & Spectre"] == [1],
        "the sqrt6 coincidence, recorded": co["sqrt6_shared"]
            and fc["lambda^2 = 4 + sqrt15"],
        "no Euclidean triangle group has both 3 and 4":
            not euclidean_triangle_groups()["any_with_3_and_4"],
        "cusp component = toric B3, triangular-lattice hoppings":
            hofstadter_link()["equal_to_hexagon_rays"],
    }
    return {"checks": checks, "all_pass": all(checks.values())}
