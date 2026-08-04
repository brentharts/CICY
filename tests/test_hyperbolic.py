"""
Tests for pyCICY.hyperbolic.

The module implements the geometry and the automorphic Bloch theory of
Maciejko and Rayan, "Automorphic Bloch theorems for hyperbolic lattices",
PNAS 119(9) e2116869119 (2022). Almost everything here is a closed-form
expression that is easy to misremember, so the suite derives rather than
trusts:

  [1] the circumradius is found numerically, by demanding the cell's
      interior angle really be 2 pi / q, and only then compared with the
      closed form. The inradius and edge length are checked against
      distances measured in the disk. (All three of these formulas are
      commonly confused with one another; the first draft of this module had
      them permuted, and this section is what caught it.)
  [2] SU(1,1) really acts by isometries: distances are preserved.
  [3] the side-pairing translations pair into inverses and satisfy the
      relator of the regular 4g-gon -- and the *canonical* surface word does
      not hold on these generators, which is checked explicitly so that the
      distinction is not quietly lost.
  [4] Gauss-Bonnet confirms the genus independently: the cell area equals
      4 pi (g - 1) = 2 pi |chi|.
  [5] the boundary fraction of a flake tends to (p-2)/(p-1), not to zero.
  [6] the Weyl pair has scalar commutators, is irreducible, and the
      representations built from it satisfy the relator and reduce to the
      abelian sector at N = 1.

Run with:  python3 tests/test_hyperbolic.py
       or: python3 run_tests.py  (runs every suite)
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyCICY import hyperbolic as H

FAILURES = []


def check(name, got, want):
    ok = got == want
    print("  {:<58} {:>10} {}".format(name, str(got)[:10],
                                      "ok" if ok else "FAIL want " + str(want)))
    if not ok:
        FAILURES.append(name)


def check_true(name, cond):
    print("  {:<58} {:>10} {}".format(name, str(bool(cond)),
                                      "ok" if cond else "FAIL"))
    if not cond:
        FAILURES.append(name)


def check_close(name, got, want, tol=1e-9):
    ok = abs(float(got) - float(want)) <= tol
    print("  {:<58} {:>10.3e} {}".format(name, abs(float(got) - float(want)),
                                         "ok" if ok else "FAIL"))
    if not ok:
        FAILURES.append(name)


TILINGS = [(8, 8), (7, 3), (3, 7), (5, 4), (4, 5), (12, 12), (6, 4), (16, 16)]

# --------------------------------------------------------------------- [1]

print("\n[1] the geometry is derived, not assumed")
check_true("{4,4} is euclidean, not hyperbolic", not H.exists(4, 4))
check_true("{3,6} is euclidean, not hyperbolic", not H.exists(3, 6))
check_true("{5,3} is spherical, not hyperbolic", not H.exists(5, 3))
check_true("{7,3} is hyperbolic", H.exists(7, 3))
for p, q in TILINGS:
    check_close("{%d,%d}: closed-form R matches the numerical solve" % (p, q),
                H.circumradius(p, q), H.solve_circumradius(p, q), tol=1e-7)
    check_close("{%d,%d}: interior angle is 2 pi / q" % (p, q),
                H.interior_angle(p, H.circumradius(p, q)),
                2 * math.pi / q, tol=1e-9)
    check_true("{%d,%d}: inradius is less than circumradius" % (p, q),
               H.inradius(p, q) < H.circumradius(p, q))

    verts = H.cell_vertices(p, q)
    check("{%d,%d}: cell has p vertices" % (p, q), len(verts), p)
    check_close("{%d,%d}: vertices sit at the circumradius" % (p, q),
                H.distance(verts[0]), H.circumradius(p, q), tol=1e-9)
    check_close("{%d,%d}: edge length" % (p, q),
                H.distance(verts[0], verts[1]), H.edge_length(p, q), tol=1e-9)
    # the inradius is the distance to the geodesic midpoint of an edge
    to0 = lambda z: (z - verts[0]) / (1 - np.conj(verts[0]) * z)
    back = lambda z: (z + verts[0]) / (1 + np.conj(verts[0]) * z)
    w = to0(verts[1])
    mid = back(np.tanh(H.distance(verts[0], verts[1]) / 4.0) * w / abs(w))
    check_close("{%d,%d}: inradius reaches the edge midpoint" % (p, q),
                H.distance(mid), H.inradius(p, q), tol=1e-9)
try:
    H.circumradius(4, 4)
    check_true("non-hyperbolic input is rejected", False)
except ValueError:
    check_true("non-hyperbolic input is rejected", True)

# --------------------------------------------------------------------- [2]

print("\n[2] SU(1,1) acts by isometries")
rng = np.random.default_rng(0)
zs = [complex(*(0.7 * rng.uniform(-1, 1, 2))) for _ in range(6)]
mats = [H.translation(1.3, 0.4), H.translation(0.7, -2.1), H.rotation(0.9),
        H.translation(2.0, 1.0) @ H.rotation(-0.3)]
for i, M in enumerate(mats):
    check_close("matrix %d has unit determinant" % i,
                abs(np.linalg.det(M)), 1.0, tol=1e-9)
    check_true("matrix %d inverts correctly" % i,
               np.allclose(M @ H.inverse(M), np.eye(2), atol=1e-9))
    for a, b in zip(zs, zs[1:]):
        check_close("matrix %d preserves a distance" % i,
                    H.distance(H.apply(M, a), H.apply(M, b)),
                    H.distance(a, b), tol=1e-8)
check_close("translation moves the origin the right distance",
            H.distance(H.apply(H.translation(1.7, 0.3), 0j)), 1.7, tol=1e-9)
check_close("rotation fixes the origin",
            abs(H.apply(H.rotation(1.1), 0j)), 0.0, tol=1e-12)

# --------------------------------------------------------------------- [3]

print("\n[3] the Fuchsian generators and the relator")
for g in (2, 3, 4):
    p = 4 * g
    gens = H.generators(p)
    check("{%d,%d}: p generators" % (p, p), len(gens), p)
    for k in range(p):
        check_true("{%d,%d}: generator %d pairs with its opposite" % (p, p, k),
                   np.allclose(gens[k] @ gens[(k + p // 2) % p],
                               np.eye(2), atol=1e-8))
    check_true("{%d,%d}: the opposite-side relator holds" % (p, p),
               H.relator_holds(p))
    check_true("{%d,%d}: relator residual is tiny" % (p, p),
               H.relator_residual(p) < 1e-8)
    check("{%d,%d}: relator word has length p" % (p, p),
          len(H.relator_word(p)), p)
    check_true("{%d,%d}: relator alternates in sign" % (p, p),
               [e for _, e in H.relator_word(p)][:4] == [1, -1, 1, -1])

# The canonical surface word does NOT hold on these generators. If this ever
# starts passing, the relator convention has silently changed.
gens = H.generators(8)
canonical = np.eye(2, dtype=complex)
for k in (0, 1, 4, 5, 2, 3, 6, 7):          # a b a^-1 b^-1 c d c^-1 d^-1
    canonical = canonical @ gens[k]
check_true("the canonical commutator word is NOT the octagon relator",
           not np.allclose(canonical, np.eye(2), atol=1e-6)
           and not np.allclose(canonical, -np.eye(2), atol=1e-6))

# --------------------------------------------------------------------- [4]

print("\n[4] Gauss-Bonnet confirms the genus")
for g in (2, 3, 4, 5):
    p = 4 * g
    check("{%d,%d} has genus %d" % (p, p, g), H.genus(p), g)
    check_close("{%d,%d}: area equals 4 pi (g-1)" % (p, p),
                H.cell_area(p, p), 4 * math.pi * (g - 1), tol=1e-9)
    check_close("{%d,%d}: area equals 2 pi |chi|" % (p, p),
                H.cell_area(p, p), 2 * math.pi * abs(2 - 2 * g), tol=1e-9)
for p, q in TILINGS:
    check_true("{%d,%d}: area is positive" % (p, q), H.cell_area(p, q) > 0)
try:
    H.genus(7)
    check_true("genus of a non-{4g,4g} tiling is rejected", False)
except ValueError:
    check_true("genus of a non-{4g,4g} tiling is rejected", True)

# --------------------------------------------------------------------- [5]

print("\n[5] flakes, and why they cannot stand in for the bulk")
sizes = [len(H.flake(8, 8, d)) for d in (0, 1, 2, 3)]
check("flake sizes for {8,8}", sizes, [1, 9, 65, 457])
check_true("each ring is (p-1) times the previous one",
           all(abs((sizes[i + 1] - sizes[i])
                   / float(sizes[i] - sizes[i - 1]) - 7.0) < 1e-9
               for i in (1, 2)))
A, pts = H.flake_adjacency(8, 8, 2)
check("adjacency is square", A.shape, (len(pts), len(pts)))
check_true("adjacency is symmetric", np.allclose(A, A.T))
check_true("the central cell has p neighbours", A.sum(axis=1).max() == 8)
check_true("no cell exceeds p neighbours", A.sum(axis=1).max() <= 8)
for p in (8, 12, 16):
    frac = H.boundary_fraction(p, p, 3)
    check_close("{%d,%d}: boundary fraction approaches (p-2)/(p-1)" % (p, p),
                frac, H.boundary_fraction_limit(p), tol=2e-3)
    check_true("{%d,%d}: boundary fraction does not vanish" % (p, p),
               frac > 0.8)
check_true("deeper flakes do not reduce the boundary fraction much",
           abs(H.boundary_fraction(8, 8, 3) - H.boundary_fraction(8, 8, 2))
           < 0.01)
E = H.flake_spectrum(8, 8, 2)
check("flake spectrum has one eigenvalue per cell", len(E), 65)
check_true("flake spectrum is real and bounded by the degree",
           float(np.max(np.abs(E))) <= 8.0 + 1e-9)

# --------------------------------------------------------------------- [6]

print("\n[6] the Weyl pair and the higher-dimensional sectors")
for N in (2, 3, 4, 5, 6):
    X, Z, omega = H.weyl_pair(N)
    eye = np.eye(N, dtype=complex)
    check_true("N=%d: X is unitary" % N, np.allclose(X @ X.conj().T, eye))
    check_true("N=%d: Z is unitary" % N, np.allclose(Z @ Z.conj().T, eye))
    check_true("N=%d: X^N = 1" % N,
               np.allclose(np.linalg.matrix_power(X, N), eye))
    check_true("N=%d: Z^N = 1" % N,
               np.allclose(np.linalg.matrix_power(Z, N), eye))
    ok = True
    for a in range(N):
        for b in range(N):
            Xa = np.linalg.matrix_power(X, a)
            Zb = np.linalg.matrix_power(Z, b)
            c = Xa @ Zb @ np.linalg.inv(Xa) @ np.linalg.inv(Zb)
            if not np.allclose(c, omega ** (-a * b) * eye, atol=1e-8):
                ok = False
    check_true("N=%d: [X^a, Z^b] = omega^{-ab} times the identity" % N, ok)
    # irreducibility: the commutant of {X, Z} is the scalars
    rows = []
    for i in range(N):
        for j in range(N):
            E_ = np.zeros((N, N), dtype=complex)
            E_[i, j] = 1
            rows.append(np.concatenate([(X @ E_ - E_ @ X).ravel(),
                                        (Z @ E_ - E_ @ Z).ravel()]))
    sv = np.linalg.svd(np.array(rows).T)[1]
    commutant = N * N - int(np.sum(sv > 1e-9))
    check("N=%d: commutant of the Weyl pair is one-dimensional" % N,
          commutant, 1)

for g in (2, 3):
    for N in (1, 2, 3, 4, 5):
        U = H.weyl_rep(N, g)
        check("N=%d g=%d: 2g matrices returned" % (N, g), len(U), 2 * g)
        check_true("N=%d g=%d: representation satisfies the relator" % (N, g),
                   H.rep_is_valid(U))
check_true("the trivial representation is valid", H.rep_is_valid(H.weyl_rep(1)))
try:
    H.weyl_rep(3, g=1)
    check_true("genus below two is rejected", False)
except ValueError:
    check_true("genus below two is rejected", True)

print("\n[6b] the abelian sector, and the reduction to it")
rng = np.random.default_rng(3)
for _ in range(6):
    theta = rng.uniform(0, 2 * np.pi, size=4)
    Hm = H.bloch_hamiltonian(H.weyl_rep(1, 2), theta)
    check_close("N=1 Bloch Hamiltonian is 2t sum cos k",
                float(Hm.real[0, 0]), H.abelian_energy(theta), tol=1e-12)
for N in (2, 3, 4):
    theta = rng.uniform(0, 2 * np.pi, size=4)
    Hm = H.bloch_hamiltonian(H.weyl_rep(N, 2), theta)
    check("N=%d Hamiltonian is N x N" % N, Hm.shape, (N, N))
    check_true("N=%d Hamiltonian is Hermitian" % N,
               np.allclose(Hm, Hm.conj().T, atol=1e-12))
check_close("abelian bandwidth at genus 2", H.abelian_bandwidth(2), 16.0)
check_close("abelian bandwidth at genus 3", H.abelian_bandwidth(3), 24.0)
Ea = H.abelian_spectrum(2, nk=8)
check_close("abelian band reaches +4g", float(Ea.max()), 8.0, tol=1e-9)
check_close("abelian band reaches -4g", float(Ea.min()), -8.0, tol=1e-9)
centres, counts = H.abelian_dos(2, nk=8, bins=40)
check("dos returns matching arrays", len(centres), len(counts))
check_true("dos is non-negative", bool((counts >= 0).all()))

print("\n[6c] the sectors do not agree, and are not claimed to")
rep = H.compare_sectors(g=2, dims=(1, 2, 3), depth=2, samples=200)
check("comparison covers three sectors plus a flake", len(rep["rows"]), 4)
check("comparison records the genus", rep["genus"], 2)
check("comparison records the tessellation", rep["tessellation"], (8, 8))
abelian = rep["rows"][0]
others = rep["rows"][1:3]
check_true("the abelian band is the widest",
           all(o["max"] <= abelian["max"] + 1e-9 for o in others))
check_true("the higher sectors are strictly narrower",
           all(o["max"] < abelian["max"] - 1e-6 for o in others))
check_true("the flake is narrower still",
           rep["rows"][3]["max"] < others[0]["max"])
check_true("the flake row reports its boundary fraction",
           rep["rows"][3]["boundary_fraction"] > 0.8)

found = H.search_reps(2, g=2, limit=4)
check_true("the search turns up further representations", len(found) > 0)
for exps in found:
    X, Z, _ = H.weyl_pair(2)
    mats = [np.linalg.matrix_power(X, a) @ np.linalg.matrix_power(Z, b)
            for a, b in exps]
    check_true("a searched representation satisfies the relator",
               H.rep_is_valid(mats))


print("\n" + "-" * 72)
if FAILURES:
    print("FAILED (%d):" % len(FAILURES))
    for f in FAILURES:
        print("  " + f)
    sys.exit(1)
print("test_hyperbolic: all checks passed")
