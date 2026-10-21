#!/usr/bin/env python3
"""
Conjecture 1 for every substitution level at once: an exact finite automaton.

make_sixsphere_mystic_levels.py counts the mystic fibre classes on whole patches
of up to n = 7 steps (2.1 million tiles) and finds six each time from n = 3.
This script explains why, and for all n, by exact arithmetic.

The structure of the substitution (spectre.py, buildSupertiles)
---------------------------------------------------------------
At level m the eight child placements Theta_0..Theta_7 are rigid motions
    Theta_k = (R . Rot_k,  R . t_k),     R = the mirror x -> -x,
whose rotation parts Rot_k (multiples of 60 degrees, a fixed list) do NOT depend
on m. Only the translations do, and they are fixed by the four-point 'quad' Q^{m-1}
of the level below:
    t_k = Theta_{k-1}(Q^{m-1}[from_k]) - Rot_k Q^{m-1}[to_k],
    Q^m = (Theta_6 Q[2], Theta_5 Q[1], Theta_3 Q[2], Theta_0 Q[1]).
All nine supertile labels at one level use the same eight Theta_k; the label only
chooses which child tile goes in each slot.

Why it is finite
----------------
The rotations by 60 degrees and the mirror x -> -x map the period lattice
Lambda = Z E1 + Z E2 to itself, as integer matrices in the basis (E1, E2):
    rho = [[0,-1],[1,1]],   mirror = [[-1,0],[1,1]].
So every formula above is well defined on the torus R^2 / Lambda, and the quad
Q^m and the translations t_k^m, reduced mod Lambda, evolve by an exact
deterministic map on a finite set (the base quad is in (1/6) Z^2). The set of
placements of the mystic Gamma2 inside a level-m supertile of label X,
    W_m(X) = { (g, c) : g in the 12-element group, c in R^2/Lambda },
obeys W_m(X) = U_k Theta_k^m . W_{m-1}(child_k(X)),
with W_0(Gamma) = {(I, 0)} (the Gamma2 inside the level-0 Gamma) and W_0 empty
for the other labels. The pair (Q^m mod Lambda, W_m) therefore runs through a
finite set, so it is eventually periodic, and one pass until the first repeat
settles every n.

What is read off W_n(Delta)
---------------------------
Gamma2 is placed in the Gamma frame by B2 = (rotation 30 deg, + points[8]), so its
anchor and first edge there are x_a = B2(myst[0]) = (1/2, 1/2) and d0 = (1/2, 0)
in the basis (E1, E2). For (g, c) in W_n(Delta):
    anchor = g x_a + c,     first-edge direction = g d0.
The base frame is not the lattice: the 14 vertices of the base Hat (and the Gamma2
vertices after B2) all lie on the Laves lattice shifted by o_b = (0, 1/2), found
here by brute force and unique. Registration, i.e. the lattice offset of the whole
level-n patch, is o_g = g o_b + c, and the assertion that it is the same for every
(g, c) in W_n(Delta) is exactly the statement that every tile vertex of the patch
is a site. The class of a mystic is (reflection, first-edge angle, anchor - o_g mod
Lambda), and it is the edge-midpoint site of the double curve D_a, D_c or D_b.

    python3 paper/make_sixsphere_mystic_automaton.py
"""

import os
import sys
from fractions import Fraction as Fr
from math import floor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import numpy as np

import make_sixsphere_figures as F

# ---- integer matrices on (E1, E2)-coordinates -----------------------------------

I2 = ((1, 0), (0, 1))
RHO = ((0, -1), (1, 1))          # rotation by 60 degrees
MIR = ((-1, 0), (1, 1))          # x -> -x


def mm(A, B):
    return tuple(tuple(sum(A[i][k] * B[k][j] for k in range(2)) for j in range(2))
                 for i in range(2))


def det(A):
    return A[0][0] * A[1][1] - A[0][1] * A[1][0]


def rot(deg):
    assert deg % 60 == 0
    r = I2
    for _ in range((deg // 60) % 6):
        r = mm(RHO, r)
    return r


def ap(A, v):
    return (A[0][0] * v[0] + A[0][1] * v[1], A[1][0] * v[0] + A[1][1] * v[1])


def add(u, v):
    return (u[0] + v[0], u[1] + v[1])


def sub(u, v):
    return (u[0] - v[0], u[1] - v[1])


def red(v):
    """reduce mod Lambda: coordinates into [0, 1)."""
    return tuple(x - floor(x) for x in v)


B = np.array([F.E1, F.E2]).T          # columns E1, E2
BI = np.linalg.inv(B)


def real(A):
    return B @ np.array(A, float) @ BI


def _check_matrices():
    for k in range(6):
        th = np.radians(60 * k)
        want = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        assert np.allclose(real(rot(60 * k)), want), "rotation matrix in (E1,E2)"
    assert np.allclose(real(MIR), np.array([[-1.0, 0.0], [0.0, 1.0]])), "mirror matrix"
    # both preserve Lambda: integer matrices with determinant +-1
    assert all(abs(det(rot(60 * k))) == 1 for k in range(6)) and abs(det(MIR)) == 1


# ---- the base data from spectre.py, exactly ---------------------------------------

S3 = 3 ** 0.5
SITES6 = {(0, 0), (0, 3), (3, 0), (3, 3), (2, 2), (4, 4)}      # site classes, units 1/6
EDGE = {(0, 3): "D_a", (3, 0): "D_c", (3, 3): "D_b"}
# what the numeric scripts call them: (1/2, 0) is D_a, (0, 1/2) is D_c, (1/2, 1/2) is D_b
EDGE = {(Fr(1, 2), Fr(0)): "D_a", (Fr(0), Fr(1, 2)): "D_c", (Fr(1, 2), Fr(1, 2)): "D_b"}


def to_uv_exact(p):
    x = BI @ np.asarray(p, float) * 6
    r = np.round(x)
    assert np.abs(x - r).max() < 1e-9, ("not a multiple of 1/6", x)
    return (Fr(int(r[0]), 6), Fr(int(r[1]), 6))


def base_data(spectre):
    pts = spectre.get_spectre_points(1.0, S3)
    myst = spectre.get_spectre_points(S3, 1.0)
    H = [to_uv_exact(p) for p in pts]
    # brute-force the offset making all base-Hat points sites (unique)
    cands = []
    for a in range(6):
        for b in range(6):
            if all((int(red(sub(h, (Fr(a, 6), Fr(b, 6))))[0] * 6) % 6,
                    int(red(sub(h, (Fr(a, 6), Fr(b, 6))))[1] * 6) % 6) in SITES6 for h in H):
                cands.append((Fr(a, 6), Fr(b, 6)))
    assert len(cands) == 1, cands
    ob = cands[0]
    th = np.radians(30)
    R30 = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    M2 = [to_uv_exact(R30 @ p + pts[8]) for p in myst]
    assert all((int(red(sub(m, ob))[0] * 6) % 6, int(red(sub(m, ob))[1] * 6) % 6) in SITES6
               for m in M2), "Gamma2 is not registered like the Hat"
    quad = [H[i] for i in (3, 5, 7, 11)]
    xa = M2[0]
    d0 = sub(M2[1], M2[0])
    return dict(ob=ob, quad=quad, xa=xa, d0=d0)


# ---- one level of buildSupertiles, exactly -----------------------------------------------

STEPS = ((60, 3, 1), (0, 2, 0), (60, 3, 1), (60, 3, 1), (0, 2, 0), (60, 3, 1), (-120, 3, 3))
SUBST = {
    "Gamma":  ("Pi", "Delta", None, "Theta", "Sigma", "Xi", "Phi", "Gamma"),
    "Delta":  ("Xi", "Delta", "Xi", "Phi", "Sigma", "Pi", "Phi", "Gamma"),
    "Theta":  ("Psi", "Delta", "Pi", "Phi", "Sigma", "Pi", "Phi", "Gamma"),
    "Lambda": ("Psi", "Delta", "Xi", "Phi", "Sigma", "Pi", "Phi", "Gamma"),
    "Xi":     ("Psi", "Delta", "Pi", "Phi", "Sigma", "Psi", "Phi", "Gamma"),
    "Pi":     ("Psi", "Delta", "Xi", "Phi", "Sigma", "Psi", "Phi", "Gamma"),
    "Sigma":  ("Xi", "Delta", "Xi", "Phi", "Sigma", "Pi", "Lambda", "Gamma"),
    "Phi":    ("Psi", "Delta", "Psi", "Phi", "Sigma", "Pi", "Phi", "Gamma"),
    "Psi":    ("Psi", "Delta", "Psi", "Phi", "Sigma", "Psi", "Phi", "Gamma"),
}
LABELS = list(SUBST)


def level(quad):
    """(thetas, super_quad) for the quad of the level below; thetas[k] = (g, t), t mod Lambda."""
    thetas = [(I2, (Fr(0), Fr(0)))]
    total = 0
    g = I2
    tq = list(quad)
    for ang, fr, to in STEPS:
        if ang != 0:
            total += ang
            g = rot(total)
            tq = [ap(g, q) for q in quad]
        gp, tp = thetas[-1]
        t = sub(add(ap(gp, quad[fr]), tp), tq[to])
        thetas.append((g, red(t)))
    thetas = [(mm(MIR, g), red(ap(MIR, t))) for g, t in thetas]
    sq = [add(ap(thetas[6][0], quad[2]), thetas[6][1]),
          add(ap(thetas[5][0], quad[1]), thetas[5][1]),
          add(ap(thetas[3][0], quad[2]), thetas[3][1]),
          add(ap(thetas[0][0], quad[1]), thetas[0][1])]
    return thetas, [red(q) for q in sq]


def step_W(W, thetas):
    """W_m(X) = U_k Theta_k . W_{m-1}(child_k(X))."""
    out = {}
    for X in LABELS:
        s = set()
        for k, Y in enumerate(SUBST[X]):
            if Y is None:
                continue
            gk, tk = thetas[k]
            for g, c in W[Y]:
                s.add((mm(gk, g), red(add(tk, ap(gk, c)))))
        out[X] = frozenset(s)
    return out


# ---- reading classes off W_n(root) --------------------------------------------------------

def classes(Wroot, data):
    """{(reflected, angle, anchor class): count of placements}; asserts the registration
    offset g.o_b + c is the same for every placement (= every tile vertex is a site)."""
    ob, xa, d0 = data["ob"], data["xa"], data["d0"]
    d0r = real(I2) @ (B @ np.array([float(d0[0]), float(d0[1])]))
    out = {}
    og = None
    for g, c in sorted(Wroot, key=repr):
        o = red(add(ap(g, ob), c))
        if og is None:
            og = o
        assert o == og, ("registration offset differs between placements", o, og)
        a = red(sub(add(ap(g, xa), c), og))
        assert a in EDGE, ("anchor is not an edge-midpoint site", a)
        v = real(g) @ d0r
        ang = round(float(np.degrees(np.arctan2(v[1], v[0]))) % 360.0, 3)
        if ang >= 359.999:
            ang = 0.0
        assert abs(ang / 30.0 - round(ang / 30.0)) < 1e-6
        key = (det(g) < 0, ang, EDGE[a])
        out[key] = out.get(key, 0) + 1
    return out


def step_counts(C, thetas):
    """The same recursion with multiplicities: C_m(X)[(g, c)] = number of Gamma2 tiles."""
    out = {}
    for X in LABELS:
        d = {}
        for k, Y in enumerate(SUBST[X]):
            if Y is None:
                continue
            gk, tk = thetas[k]
            for (g, c), n in C[Y].items():
                key = (mm(gk, g), red(add(tk, ap(gk, c))))
                d[key] = d.get(key, 0) + n
        out[X] = d
    return out


def class_sizes(Croot, data):
    """{(reflected, angle, double curve): number of mystics}, from a counted placement dict."""
    ob, xa, d0 = data["ob"], data["xa"], data["d0"]
    d0r = B @ np.array([float(d0[0]), float(d0[1])])
    out, og = {}, None
    for (g, c), n in Croot.items():
        o = red(add(ap(g, ob), c))
        og = o if og is None else og
        assert o == og
        a = red(sub(add(ap(g, xa), c), og))
        v = real(g) @ d0r
        ang = round(float(np.degrees(np.arctan2(v[1], v[0]))) % 360.0, 3)
        ang = 0.0 if ang >= 359.999 else ang
        key = (det(g) < 0, ang, EDGE[a])
        out[key] = out.get(key, 0) + n
    return out


def sizes_by_level(max_level, data):
    """class sizes at every level 1..max_level for the root Delta (exact integers)."""
    quad = data["quad"]
    C = {X: {} for X in LABELS}
    C["Gamma"] = {(I2, (Fr(0), Fr(0))): 1}
    res = []
    for m in range(1, max_level + 1):
        thetas, quad = level(quad)
        C = step_counts(C, thetas)
        res.append((m, class_sizes(C["Delta"], data)))
    return res


SITENAME = {(Fr(0), Fr(0)): "E0 (hexagon centre)", (Fr(1, 2), Fr(0)): "D_a",
            (Fr(0), Fr(1, 2)): "D_c", (Fr(1, 2), Fr(1, 2)): "D_b",
            (Fr(1, 3), Fr(1, 3)): "P", (Fr(2, 3), Fr(2, 3)): "Q"}


def hat_classes_by_level(max_level, data, pts_uv):
    """The same count for the Hat itself (every tile except Gamma2): the anchor is the
    first vertex of the Hat polygon, (0, 0) in its own frame, and the first edge is
    pts[1] - pts[0]. A class is (reflected, angle, anchor site class mod Lambda); the
    bound is 12 angles x 6 site classes x 2 = 144. Returns [(level, {class: size})]."""
    xh = pts_uv[0]
    d0 = sub(pts_uv[1], pts_uv[0])
    d0r = B @ np.array([float(d0[0]), float(d0[1])])
    ob = data["ob"]
    quad = data["quad"]
    C = {X: {(I2, (Fr(0), Fr(0))): 1} for X in LABELS}      # one Hat in each level-0 supertile
    res = []
    for m in range(1, max_level + 1):
        thetas, quad = level(quad)
        C = step_counts(C, thetas)
        out, og = {}, None
        for (g, c), n in C["Delta"].items():
            o = red(add(ap(g, ob), c))
            og = o if og is None else og
            assert o == og, "registration differs between Hats"
            a = red(sub(add(ap(g, xh), c), og))
            assert a in SITENAME, ("Hat anchor is not a site", a)
            v = real(g) @ d0r
            ang = round(float(np.degrees(np.arctan2(v[1], v[0]))) % 360.0, 3)
            ang = 0.0 if ang >= 359.999 else ang
            key = (det(g) < 0, ang, SITENAME[a])
            out[key] = out.get(key, 0) + n
        res.append((m, out))
    return res


def six_set(data, table):
    """the six (angle, double curve) pairs seen at level >= 3."""
    return sorted({(k[1], k[2]) for m, cl in table if m >= 3 for k in cl})


def roots_report(data, levels=(3, 4, 5, 6)):
    """classes of every root label X, as (angle, double curve) sets, per level."""
    quad = data["quad"]
    W = {X: frozenset() for X in LABELS}
    W["Gamma"] = frozenset([(I2, (Fr(0), Fr(0)))])
    out = {}
    for m in range(1, max(levels) + 1):
        thetas, quad = level(quad)
        W = step_W(W, thetas)
        if m in levels:
            out[m] = {X: sorted({(k[1], k[2]) for k in classes(W[X], data)}) for X in LABELS
                      if W[X]}
    return out


def run(max_level=60, verbose=True):
    _check_matrices()
    import make_sixsphere_hat_torus as H
    spectre = H.load_spectre()
    data = base_data(spectre)
    quad = data["quad"]
    W = {X: frozenset() for X in LABELS}
    W["Gamma"] = frozenset([(I2, (Fr(0), Fr(0)))])
    seen = {}
    table = []
    state = (tuple(red(q) for q in quad), tuple(W[X] for X in LABELS))
    seen[state] = 0
    period = None
    for m in range(1, max_level + 1):
        thetas, quad = level(quad)
        W = step_W(W, thetas)
        cl = classes(W["Delta"], data)
        # the six classes, as a set of (angle, double curve) (reflection recorded apart)
        table.append((m, cl))
        state = (tuple(quad), tuple(W[X] for X in LABELS))
        if state in seen:
            period = (seen[state], m)
            break
        seen[state] = m
    return table, period, data


def main():
    table, period, data = run()
    print("base offset o_b =", tuple(str(x) for x in data["ob"]),
          "  x_a =", tuple(str(x) for x in data["xa"]), "  d0 =", tuple(str(x) for x in data["d0"]))
    print("\nlevel  placements  classes  reflected?  (angle, double curve)")
    for m, cl in table:
        refl = {k[0] for k in cl}
        pairs = sorted((k[1], k[2]) for k in cl)
        print("%4d  %10d  %7d  %-10s  %s" % (m, sum(cl.values()), len(cl),
              "all" if refl == {True} else ("none" if refl == {False} else "mixed"),
              " ".join("%d%s" % (a, d[-1]) for a, d in pairs)))
    if period:
        print("\nthe state (quad mod Lambda, all placement sets) first repeats: level %d = level %d"
              % (period[1], period[0]))
        print("so levels >= %d repeat with period %d; the table above is therefore the whole story."
              % (period[0], period[1] - period[0]))
    else:
        print("\nno repeat found within the level bound")

    six = six_set(data, table)
    print("\nthe six classes (angle, double curve):", " ".join("%d%s" % (a, d[-1]) for a, d in six))
    # invariance under the 12-element group: rho and the mirror permute the six
    def act(g, a, d):
        pt = {"D_a": (Fr(1, 2), Fr(0)), "D_c": (Fr(0), Fr(1, 2)), "D_b": (Fr(1, 2), Fr(1, 2))}[d]
        q = red(ap(g, pt))
        v = real(g) @ np.array([np.cos(np.radians(a)), np.sin(np.radians(a))])
        ang = round(float(np.degrees(np.arctan2(v[1], v[0]))) % 360.0, 3)
        return (0.0 if ang >= 359.999 else ang), EDGE[q]
    inv = all(act(g, a, d) in six for g in (rot(60), MIR) for a, d in six)
    print("the set of six is invariant under rotation by 60 degrees and the mirror:", inv)

    print("\nclass sizes (number of mystic tiles) on the whole level-n patch, root Delta:")
    for m, cl in sizes_by_level(12, data):
        if m < 3:
            continue
        print("  n=%2d  total %-9d  %s" % (m, sum(cl.values()), " ".join(
            "%d%s:%d" % (k[1], k[2][-1], v) for k, v in sorted(cl.items(), key=lambda kv: kv[0][1]))))

    print("\nthe Hat itself (every tile but the mystic): classes of (reflected, angle, anchor site):")
    import make_sixsphere_hat_torus as H
    pts = H.load_spectre().get_spectre_points(1.0, S3)
    pts_uv = [to_uv_exact(p) for p in pts]
    for m, cl in hat_classes_by_level(10, data, pts_uv):
        refl = sorted({k[0] for k in cl})
        print("  n=%2d  %d Hats, %3d classes (bound 144), reflections %s, "
              "angles %d, anchor sites %d" % (m, sum(cl.values()), len(cl), refl,
                                              len({k[1] for k in cl}), len({k[2] for k in cl})))

    print("\nevery root label X, classes (angle, double curve) at levels 3..6:")
    rr = roots_report(data)
    for m, d in rr.items():
        for X, pairs in d.items():
            sub6 = set(pairs) <= set(six)
            print("  level %d  %-7s %d classes%s" % (m, X, len(pairs), "" if sub6 else "  NOT a subset of the six"))


if __name__ == "__main__":
    main()
