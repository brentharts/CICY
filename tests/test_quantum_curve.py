"""
Tests for pyCICY.quantum_curve.

The module quantizes the mirror curve of a local toric Calabi-Yau into a
magnetic Bloch matrix, following Sugimoto, "Calabi-Yau geometry and electrons
on 2d lattices", arXiv:1701.01561, and the earlier local F_0 observation of
Hatsuda, Katsura and Tachikawa. The claims worth checking are:

  [1] the matrix is Hermitian, has q bands, and at flux 1/2 on the square
      lattice reproduces the closed-form Harper result exactly;
  [2] local F_0 gives the square lattice and local B_3 the triangular one,
      which is the content of the correspondence;
  [3] gap Chern numbers come out as the textbook TKNN values, and the
      Diophantine equation they solve actually holds;
  [4] the spectrum is symmetric under E -> -E exactly for the bipartite
      polygons -- checked against all sixteen, not asserted;
  [5] E(Phi) = -E(1-Phi) holds universally, while E(Phi) = E(1-Phi) holds
      exactly in the bipartite cases;
  [6] reflecting the Newton polygon does *not* move the spectrum, so the
      spectral notion of chirality is genuinely bipartiteness and not
      polygon reflection. This is a negative result and is recorded as one.

Run with:  python3 tests/test_quantum_curve.py
       or: python3 run_tests.py  (runs every suite)
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyCICY import toric as T
from pyCICY import quantum_curve as Q

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
    ok = np.allclose(got, want, atol=tol)
    print("  {:<58} {:>10} {}".format(name, "{:.3e}".format(
        float(np.max(np.abs(np.asarray(got) - np.asarray(want))))),
        "ok" if ok else "FAIL"))
    if not ok:
        FAILURES.append(name)


# --------------------------------------------------------------------- [1]

print("\n[1] the magnetic Bloch matrix")
h = Q.harper()
for q in (2, 3, 5, 7):
    H = h.bloch_matrix(1, q, 0.31, 0.77)
    check("Harper matrix is %dx%d" % (q, q), H.shape, (q, q))
    check_close("Harper matrix Hermitian at q=%d" % q, H, H.conj().T)
    check_true("eigenvalues real at q=%d" % q,
               np.allclose(np.linalg.eigvals(H).imag, 0, atol=1e-9))
check("q bands at Phi=3/7", h.bands(3, 7, nk=4).shape[1], 7)
check_close("hbar at Phi=1/4", h.hbar(1, 4), np.pi / 2)

# The square lattice at half flux: E = +- 2 sqrt(cos^2 k1 + cos^2 k2).
for k1, k2 in ((0.3, 0.7), (1.1, 2.4), (0.0, 0.0)):
    w = 2 * np.sqrt(np.cos(k1) ** 2 + np.cos(k2) ** 2)
    got = np.sort(np.linalg.eigvalsh(h.bloch_matrix(1, 2, k1, k2)))
    check_close("Harper Phi=1/2 at k=(%.1f,%.1f) vs closed form" % (k1, k2),
                got, [-w, w])

check_close("square lattice at zero field has bandwidth 8",
            h.spectrum(0, 1, nk=24).max() - h.spectrum(0, 1, nk=24).min(),
            8.0, tol=2e-2)

# --------------------------------------------------------------------- [2]

print("\n[2] the correspondence: polygon -> lattice")
check("local F_0 gives four hops", sorted(Q.from_polygon("F0").points),
      [(-1, 0), (0, -1), (0, 1), (1, 0)])
check("local B_3 gives six hops", sorted(Q.from_polygon("B3").points),
      [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)])
check("local P^2 gives three hops", len(Q.from_polygon("P2").points), 3)
check_true("vertices_only drops non-vertex boundary points",
           len(Q.from_polygon("T4", vertices_only=True).points)
           < len(Q.from_polygon("T4").points))
try:
    Q.QuantumCurve([(0, 0), (1, 0)])
    check_true("origin rejected as a hop", False)
except ValueError:
    check_true("origin rejected as a hop", True)
try:
    Q.QuantumCurve([(1, 0), (0, 1)], coeffs=[1.0])
    check_true("mismatched coefficients rejected", False)
except ValueError:
    check_true("mismatched coefficients rejected", True)

# --------------------------------------------------------------------- [3]

print("\n[3] gap Chern numbers (TKNN)")


def labels(curve, p, q, nk=14):
    return [(g["filled"], g["chern"]) for g in curve.gap_labels(p, q, nk=nk)]


check("F0 gap labels at Phi=1/3", labels(h, 1, 3), [(1, 1), (2, -1)])
check("F0 gap labels at Phi=1/4", labels(h, 1, 4), [(1, 1), (3, -1)])
check("F0 gap labels at Phi=2/5", labels(h, 2, 5),
      [(1, -2), (2, 1), (3, -1), (4, 2)])
check("F0 gap labels at Phi=1/5", labels(h, 1, 5),
      [(1, 1), (2, 2), (3, -2), (4, -1)])

for q in (3, 4, 5, 7, 8):
    for p in range(1, q):
        if np.gcd(p, q) != 1:
            continue
        for r in range(1, q):
            t = Q.QuantumCurve.chern_number(r, p, q)
            check_true("Diophantine r=%d at %d/%d: |t| <= q/2" % (r, p, q),
                       abs(t) <= q / 2)
            check_true("Diophantine r=%d at %d/%d: r = qs + pt solvable"
                       % (r, p, q), (r - p * t) % q == 0)
try:
    Q.QuantumCurve.chern_number(1, 2, 4)
    check_true("non-reduced flux rejected", False)
except ValueError:
    check_true("non-reduced flux rejected", True)

# --------------------------------------------------------------------- [4]

print("\n[4] spectral chirality tracks bipartiteness, over all sixteen")
mismatch = []
for nm in sorted(T.NAMED):
    c = Q.from_polygon(nm)
    bip = c.is_bipartite()
    sym = (c.spectral_asymmetry(1, 3, nk=10) < 1e-8
           and c.spectral_asymmetry(2, 5, nk=10) < 1e-8)
    if bip != sym:
        mismatch.append(nm)
check("polygons where bipartite != E->-E symmetric", mismatch, [])
check("bipartite polygons", sorted(nm for nm in T.NAMED
                                   if Q.from_polygon(nm).is_bipartite()),
      ["F0", "T4"])
check_true("F0 spectrum symmetric", not Q.from_polygon("F0").is_spectrally_chiral())
check_true("B3 spectrum asymmetric", Q.from_polygon("B3").is_spectrally_chiral())
check_true("P2 spectrum asymmetric", Q.from_polygon("P2").is_spectrally_chiral())
check_true("B3 is centrally symmetric yet spectrally chiral",
           Q.from_polygon("B3").is_centrally_symmetric()
           and Q.from_polygon("B3").is_spectrally_chiral())
check_true("T4 is spectrally symmetric yet not centrally symmetric",
           not Q.from_polygon("T4").is_centrally_symmetric()
           and not Q.from_polygon("T4").is_spectrally_chiral())

# --------------------------------------------------------------------- [5]

print("\n[5] the flux reflection Phi -> 1 - Phi")
for nm in ("F0", "B3", "P2", "T4", "T9"):
    c = Q.from_polygon(nm)
    anti = plain = 0
    total = 0
    for p, q in Q.farey(7):
        a = np.sort(c.spectrum(p, q, nk=6))
        b = np.sort(c.spectrum(q - p, q, nk=6))
        total += 1
        if np.allclose(a, -b[::-1], atol=1e-7):
            anti += 1
        if np.allclose(a, b, atol=1e-7):
            plain += 1
    check("%s: E(Phi) = -E(1-Phi) everywhere" % nm, anti, total)
    check("%s: E(Phi) = E(1-Phi) iff bipartite" % nm,
          plain == total, c.is_bipartite())

# --------------------------------------------------------------------- [6]

print("\n[6] reflecting the polygon does not move the spectrum")
for nm in ("P2", "B3", "F1", "T9", "dP2"):
    c = Q.from_polygon(nm)
    m = c.mirror()
    check_close("%s: mirror spectrum equals original" % nm,
                np.sort(c.spectrum(1, 3, nk=8)),
                np.sort(m.spectrum(1, 3, nk=8)), tol=1e-9)
check_true("B3 hopping set is its own reflection",
           set(Q.from_polygon("B3").mirror().points)
           == set(Q.from_polygon("B3").points))
check_true("P2 hopping set is not its own reflection",
           set(Q.from_polygon("P2").mirror().points)
           != set(Q.from_polygon("P2").points))

# --------------------------------------------------------------------- [7]

print("\n[7] butterflies and bookkeeping")
check("farey(5) count", len(Q.farey(5)), 9)      # 1 + 2 + 2 + 4
check_true("farey pairs are coprime and ordered",
           all(np.gcd(p, q) == 1 and 0 < p < q for p, q in Q.farey(9)))
f, e = Q.butterfly("F0", qmax=7, nk=4)
check("butterfly arrays are parallel", len(f), len(e))
check_true("butterfly flux in (0,1)", bool((f > 0).all() and (f < 1).all()))
check_true("butterfly energies bounded by total hopping",
           float(np.abs(e).max()) <= 4.0 + 1e-9)
fb, eb = Q.butterfly("B3", qmax=7, nk=4)
check_true("B3 butterfly energies bounded by six hops",
           float(np.abs(eb).max()) <= 6.0 + 1e-9)
centres, counts = h.dos(1, 5, nk=8, bins=50)
check("dos returns matching arrays", len(centres), len(counts))
check_true("dos is non-negative", bool((counts >= 0).all()))
check_true("describe mentions the polygon name",
           "B3" in Q.from_polygon("B3").describe())
check_true("gaps are strictly positive width",
           all(g["width"] > 0 for g in h.gap_labels(2, 5, nk=12)))
check_true("band edges are ordered",
           all(lo <= hi for lo, hi in h.band_edges(2, 5, nk=8)))


print("\n" + "-" * 72)
if FAILURES:
    print("FAILED (%d):" % len(FAILURES))
    for f_ in FAILURES:
        print("  " + f_)
    sys.exit(1)
print("test_quantum_curve: all checks passed")
