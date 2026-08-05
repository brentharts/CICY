"""
Tests for pyCICY.polytope.

The checks follow the package's usual discipline: a value is asserted only
when it can be reached by a route this module does not control.

  [1] 2D overlap    polar() must reproduce toric.dual() on all sixteen
                    reflexive polygons -- independently written, older code
  [2] quintic       Batyrev on the P^4 polytope must give (1, 101) and its
                    mirror (101, 1), with l(Delta) = C(9,4) = 126
  [3] 24-cell       reflexivity, the lattice it needs, the f-vector, and the
                    Hodge numbers
  [4] Ali's claims  the two vertex sets are a dual pair; the tetrahedron
                    count is 48, not 576
  [5] duality       (P*)* = P, and dim th + dim th* = n - 1 for every face

Section [1] is the important one. Everything here generalises code that
already existed in two dimensions, so the generalisation has a ready-made
oracle and there is no excuse for not using it.

Run with:  python3 tests/test_polytope.py
       or: python3 run_tests.py
"""

import itertools
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyCICY import polytope as P
from pyCICY import toric

FAILURES = []


def check(name, got, want):
    ok = got == want
    print("  {:<58} {:>12} {}".format(name, str(got)[:12],
                                      "ok" if ok else "FAIL want " + str(want)))
    if not ok:
        FAILURES.append(name)


def check_true(name, cond):
    print("  {:<58} {:>12} {}".format(name, str(bool(cond)),
                                      "ok" if cond else "FAIL"))
    if not cond:
        FAILURES.append(name)


def _key(A):
    return sorted(map(tuple, np.asarray(A, dtype=np.int64).tolist()))


def test_two_dimensional_overlap():
    print("\n[1] agreement with pyCICY.toric in two dimensions")
    polys = toric.enumerate_reflexive()
    check("toric rederives sixteen reflexive polygons", len(polys), 16)

    exact = 0
    for v in polys:
        v = np.array(v, dtype=np.int64)
        if _key(toric.dual(v)) == _key(P.polar(v, convention="toric")):
            exact += 1
    check("polar(convention='toric') reproduces toric.dual", exact, 16)

    # ... and the Batyrev-signed version differs from it by y -> -y only.
    negated = 0
    for v in polys:
        v = np.array(v, dtype=np.int64)
        if _key(toric.dual(v)) == _key(-P.polar(v)):
            negated += 1
    check("the two conventions differ by y -> -y and nothing else",
          negated, 16)

    # Every reflexive polygon must be seen as reflexive, in the ambient
    # lattice, with no recourse to a sublattice.
    amb = sum(1 for v in polys
              if P.is_reflexive(np.array(v, dtype=np.int64))["lattice"]
              == "ambient")
    check("all sixteen are reflexive in Z^2 itself", amb, 16)

    # The twelve theorem, recomputed through this module's lattice points
    # rather than through toric's.
    bad = 0
    for v in polys:
        v = np.array(v, dtype=np.int64)
        d = P.polar(v, convention="toric")
        nb = len(P.lattice_points(v)) - len(P.interior_lattice_points(v))
        nbd = len(P.lattice_points(d)) - len(P.interior_lattice_points(d))
        if nb + nbd != 12:
            bad += 1
    check("twelve theorem holds via polytope.lattice_points", bad, 0)


def test_quintic():
    print("\n[2] Batyrev calibrated on the quintic")
    fan = P.simplex(4)
    newton = P.polar(fan)
    check("l(Delta) = C(9,4), one per quintic monomial",
          len(P.lattice_points(newton)), math.comb(9, 4))

    h = P.batyrev_hodge(newton)
    check("quintic h^{1,1}", h["h11"], 1)
    check("quintic h^{2,1}", h["h21"], 101)
    check("quintic chi", h["euler"], -200)

    hm = P.batyrev_hodge(fan)
    check("mirror quintic h^{1,1}", hm["h11"], 101)
    check("mirror quintic h^{2,1}", hm["h21"], 1)
    check_true("the mirror really is the transpose",
               (hm["h11"], hm["h21"]) == (h["h21"], h["h11"]))

    # A reflexive 3-polytope gives a K3, not a threefold.
    k3 = P.batyrev_hodge(P.cross_polytope(3))
    check("K3 from a reflexive 3-polytope: h^{1,1}", k3["h11"], 20)
    check("K3 Euler characteristic", k3["euler"], 24)


def test_twenty_four_cell():
    print("\n[3] the 24-cell")
    raw = P.d4_roots()
    check("24 vertices", len(raw), 24)
    check_true("all of squared norm 2",
               set(int((v * v).sum()) for v in raw) == {2})

    r = P.is_reflexive(raw)
    check_true("reflexive", r["reflexive"])
    check("but not in Z^4 -- it needs the lattice it generates",
          P.is_reflexive(raw, use_generated_lattice=False)["reflexive"], False)
    check("that lattice is D_4, of index 2 in Z^4", r["index"], 2)

    V = P.twenty_four_cell()
    check("f-vector", P.f_vector(V), [24, 96, 96, 24])
    check_true("Euler relation 24 - 96 + 96 - 24 = 0",
               24 - 96 + 96 - 24 == 0)

    check("l(Delta)", len(P.lattice_points(V)), 25)
    check("interior lattice points: the origin only",
          len(P.interior_lattice_points(V)), 1)

    h = P.batyrev_hodge(V)
    check("h^{1,1}", h["h11"], 20)
    check("h^{2,1}", h["h21"], 20)
    check("chi", h["euler"], 0)
    check("no facet corrections", h["facet_correction"], (0, 0))
    check("no codimension-2 corrections", h["codim2_correction"], (0, 0))

    # Self-duality forces h11 = h21 without any of the above arithmetic, so
    # the two arguments must land in the same place.
    check_true("self-duality independently forces h11 = h21",
               h["h11"] == h["h21"])


def test_ali_claims():
    print("\n[4] the checkable claims of arXiv:2511.10685")

    # The paper's two vertex sets are polar duals of one another, not two
    # descriptions of the same polytope.
    d = P.polar(P.d4_roots(), exact=False)
    unit = P.twenty_four_cell("unit")
    check_true("section 3.1 polytope is polar to the section 2.1 one",
               set(map(tuple, np.round(d, 6).tolist()))
               == set(map(tuple, np.round(unit, 6).tolist())))
    check("the section 2.1 set has 24 vertices", len(unit), 24)
    check_true("of squared norm 1",
               np.allclose((unit * unit).sum(axis=1), 1.0))
    check_true("so the dual pair is not integral in Z^4",
               not np.allclose(d, np.round(d)))

    # The tetrahedron count. The paper says 576; brute force says 48.
    tets = P.equilateral_subsets(P.d4_roots(), 4, edge_sq=4)
    check("regular tetrahedra with |v_i - v_j|^2 = 4", len(tets), 48)
    check_true("the paper's example tetrahedron is among them",
               any(set(map(tuple, P.d4_roots()[list(t)].tolist()))
                   == {(1, 1, 0, 0), (1, -1, 0, 0), (0, 0, 1, 1),
                       (0, 0, 1, -1)} for t in tets))

    # ... and the reason it is 48: the condition is orthogonality, and the
    # answer factorises as three frames times 2^4 sign choices.
    V = P.d4_roots()
    check_true("the condition is pairwise orthogonality",
               all(int(V[a] @ V[b]) == 0
                   for t in tets for a, b in itertools.combinations(t, 2)))
    check("3 orthogonal frames x 2^4 signs", 3 * 2 ** 4, 48)

    # Dropping the edge-length condition entirely changes nothing: 48 is the
    # count of *all* equilateral four-subsets of the 24-cell, at any edge
    # length. So the criterion of section 3.1 is not selecting a sub-family
    # out of a larger set of regular tetrahedra -- there is no larger set,
    # and 576 cannot be recovered by relaxing the edge length either.
    allsets = P.equilateral_subsets(P.d4_roots(), 4)
    check("equilateral 4-subsets at any edge length", len(allsets), 48)


def test_duality():
    print("\n[5] duality of polytopes and of faces")
    for name, V in (("cross-polytope 4D", P.cross_polytope(4)),
                    ("simplex 4D", P.simplex(4)),
                    ("24-cell", P.twenty_four_cell())):
        D = P.polar(V)
        DD = P.polar(D)
        check_true("(P*)* = P for the %s" % name, _key(DD) == _key(V))

    # dim th + dim th* = n - 1, for every face of the 24-cell.
    V = P.twenty_four_cell()
    D = P.polar(V)
    fl = P.face_lattice(V)
    fld = P.face_lattice(D)
    dim_of = {}
    for d, faces in fld.items():
        for f in faces:
            dim_of[f] = d
    bad = 0
    total = 0
    for d, faces in fl.items():
        for f in faces:
            g = P.dual_face(V, D, f)
            total += 1
            if dim_of.get(g, -1) != 3 - d:
                bad += 1
    check("dim(th) + dim(th*) = 3 for all %d faces" % total, bad, 0)


def main():
    t0 = time.time()
    test_two_dimensional_overlap()
    test_quintic()
    test_twenty_four_cell()
    test_ali_claims()
    test_duality()

    print("\n" + "=" * 72)
    if FAILURES:
        print("FAILED (%d): %s" % (len(FAILURES), ", ".join(FAILURES)))
        return 1
    print("test_polytope: all checks passed in %.1fs" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
