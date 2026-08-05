"""
Tests for pyCICY.bundles.

The design principle here is the one used elsewhere in this package: wherever
a number can be reached by two routes that share no code, compute it both ways
and require agreement, rather than asserting a value that was produced by the
code being tested.

  [1] index            intersection theory (contract d_rst with the charges)
                       vs the Leray spectral sequence (line_co_euler on each
                       summand). No shared code at all.
  [2] index again      the alternating sum of the actually computed
                       cohomology, a third route.
  [3] Serre duality    h^q(V) = h^{3-q}(V*) on a Calabi-Yau threefold, which
                       line_co knows nothing about and must satisfy anyway.
  [4] spectrum         n(10) - n(10-bar) must equal -ind(V), which ties the
                       SU(5) decomposition back to [1].
  [5] stability        the cheap exact sign obstruction must never reject a
                       bundle the numerical search accepts.
  [6] monads           the long exact sequence bounds must contain the value
                       forced by the index.

Everything runs on the quintic, CICY 7833 and the tetraquadric, whose Hodge
data is cheap, and every scan is boxed by an explicit limit or budget. The
whole suite is seconds, not minutes; that is deliberate, since the expensive
parts of this module are exactly the parts a test suite must not invoke
casually.

Run with:  python3 tests/test_bundles.py
       or: python3 run_tests.py  (runs every suite)
"""

import itertools
import os
import sys
import time

import numpy as np

# Prefer the source tree over any installed copy of pyCICY.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyCICY import CICY
from pyCICY import bundles as B

FAILURES = []


def check(name, got, want):
    ok = got == want
    print("  {:<56} {:>14} {}".format(name, str(got)[:14],
                                      "ok" if ok else "FAIL want " + str(want)))
    if not ok:
        FAILURES.append(name)


def check_true(name, cond):
    print("  {:<56} {:>14} {}".format(name, str(bool(cond)),
                                      "ok" if cond else "FAIL"))
    if not cond:
        FAILURES.append(name)


def check_close(name, got, want, tol=1e-9):
    ok = abs(float(got) - float(want)) < tol
    print("  {:<56} {:>14} {}".format(name, "%.10g" % got,
                                      "ok" if ok else "FAIL want " + str(want)))
    if not ok:
        FAILURES.append(name)


def check_raises(name, exc, fn, *a, **kw):
    try:
        fn(*a, **kw)
    except exc:
        print("  {:<56} {:>14} ok".format(name, "raised"))
        return
    except Exception as e:                                   # noqa: BLE001
        print("  {:<56} {:>14} FAIL wrong exception {!r}".format(
            name, "raised", e))
        FAILURES.append(name)
        return
    print("  {:<56} {:>14} FAIL no exception".format(name, "-"))
    FAILURES.append(name)


# ---------------------------------------------------------------------------
# manifolds, built once
# ---------------------------------------------------------------------------

QUINTIC = CICY([[4, 5]])
M7833 = CICY([[2, 2, 1], [3, 1, 3]])
TETRA = CICY([[1, 2], [1, 2], [1, 2], [1, 2]])

# A poly-stable rank 5 sum on the tetraquadric with ind(V) = -6, found by
# scan() and verified below by every route the module has.
TETRA_MODEL = [[-2, -2, -1, 2], [-2, 1, 0, 0], [1, -2, 1, 0],
               [1, 1, -1, 0], [2, 2, 1, -2]]


def test_topology():
    print("\n[1] Chern data and the two routes to the index")

    V = B.LineBundleSum(TETRA, TETRA_MODEL)
    check("rank", V.rank, 5)
    check("c1(V) = 0", list(V.c1), [0, 0, 0, 0])
    check_true("is_su", V.is_su)
    check_close("ind(V), intersection theory", V.index(), -6.0)
    check_close("ind(V), Leray (line_co_euler)", V.index_from_cohomology(), -6.0)

    # The agreement above is the point of the test, so state it as its own
    # assertion rather than leaving it implicit in two equal numbers.
    check_true("the two index routes agree",
               abs(V.index() - V.index_from_cohomology()) < 1e-9)

    # ... and on other manifolds, with bundles chosen only to be traceless.
    cases = [
        (M7833, [[1, -1], [2, -2], [-3, 3], [0, 0], [0, 0]]),
        (M7833, [[2, 1], [-1, 0], [0, -2], [1, 1], [-2, 0]]),
        (QUINTIC, [[1], [1], [-1], [-1], [0]]),
        (TETRA, [[1, -1, 0, 0], [0, 1, -1, 0], [0, 0, 1, -1], [-1, 0, 0, 1],
                 [0, 0, 0, 0]]),
    ]
    for i, (X, S) in enumerate(cases):
        W = B.LineBundleSum(X, S)
        check_true("index routes agree, case %d" % i,
                   abs(W.index() - W.index_from_cohomology()) < 1e-9)

    # rank and c1 of a virtual sum, the Monad case, without any bundle
    ch = B.chern_character(np.array(TETRA.triple_intersection()),
                           [[1, 0, 0, 0], [0, 1, 0, 0], [1, 1, 0, 0]],
                           signs=[1, 1, -1])
    check_close("virtual rank = 1 + 1 - 1", ch["rank"], 1.0)
    check("virtual c1 = 0", [int(round(x)) for x in ch["c1"]], [0, 0, 0, 0])

    # c_2 accessor refuses when c_1 is not zero, rather than returning -ch_2
    bad = B.LineBundleSum(TETRA, [[1, 0, 0, 0], [0, 0, 0, 0]])
    check_true("c1 != 0 detected", not bad.is_su)
    check_raises("c2 raises when c1 != 0", ValueError, lambda: bad.c2)


def test_cohomology():
    print("\n[2] cohomology, Serre duality, and a third route to the index")

    V = B.LineBundleSum(TETRA, TETRA_MODEL)
    h = V.cohomology()
    alt = int(h[0] - h[1] + h[2] - h[3])
    check("h^*(V)", list(map(int, h)), [0, 24, 18, 0])
    check_close("alternating sum = ind(V)", alt, V.index())

    # Serre duality on a CY threefold: h^q(V) = h^{3-q}(V*). line_co has no
    # notion of duality, so this is a genuine constraint on it.
    hd = V.dual().cohomology()
    for q in range(4):
        check("Serre h^%d(V) = h^%d(V*)" % (q, 3 - q),
              int(h[q]), int(hd[3 - q]))

    # A poly-stable bundle of vanishing slope has no sections and, by Serre,
    # no h^3 either. Both must vanish for the model to make sense.
    check("h^0(V) = 0", int(h[0]), 0)
    check("h^3(V) = 0", int(h[3]), 0)


def test_spectrum():
    print("\n[3] the SU(5) spectrum")

    V = B.LineBundleSum(TETRA, TETRA_MODEL)
    sp = V.su5_spectrum()
    check("n(10)", sp["n10"], 24)
    check("n(10-bar)", sp["n10bar"], 18)
    check("generations = n(10) - n(10-bar)", sp["generations"], 6)
    check_close("generations = -ind(V)", sp["generations"], -V.index())
    check_true("index_consistent", sp["index_consistent"])
    check_true("n(5-bar) - n(5) = generations",
               sp["n5bar"] - sp["n5"] == sp["generations"])

    # Six generations upstairs is three downstairs of a freely acting group
    # of order two, which is the whole reason the scan targets -3|Gamma|.
    check("three generations for |Gamma| = 2", sp["generations"] // 2, 3)

    # The decomposition is rank specific and must refuse other ranks rather
    # than relabelling the wrong cohomology groups.
    W = B.LineBundleSum(TETRA, [[1, -1, 0, 0], [-1, 1, 0, 0], [0, 0, 0, 0]])
    check_raises("su5_spectrum rejects rank 3", ValueError, W.su5_spectrum)

    # Lambda^2 of a rank 5 bundle has rank 10, and V (x) V* traceless has 20.
    check("rank of Lambda^2 V", V.wedge2().rank, 10)
    check("rank of traceless V (x) V*", V.endomorphisms().rank, 20)


def test_anomaly():
    print("\n[4] anomaly cancellation")

    V = B.LineBundleSum(TETRA, TETRA_MODEL)
    a = V.anomaly()
    check_true("surplus non-negative", a["ok"])
    check_true("reported as necessary only", a["sufficient"] is False)
    check("surplus", [int(round(x)) for x in a["surplus"]], [2, 8, 2, 44])

    # c_2(TX) is the same object the rest of the package computes.
    check("c2(TX) from CICY", [int(x) for x in TETRA.second_chern()],
          [24, 24, 24, 24])

    # The check must be able to fail, or it is doing no work. It is worth
    # being precise about why it can. ch_2(V)_r = (1/2) d_rst sum_a k^s k^t
    # with d_rst >= 0 and sum_a k_a k_a positive semi-definite, which looks
    # as though the surplus could never drop below c_2(TX) -- but a positive
    # semi-definite matrix has negative off-diagonal entries, k = (1,-1,0,0)
    # being the simplest, so the contraction against d_rst is not sign
    # definite and the condition genuinely constrains.
    fails = B.LineBundleSum(TETRA, [[3, -3, 0, 0], [-3, 3, 0, 0],
                                    [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    a2 = fails.anomaly()
    check_true("an anomaly-violating bundle is detected", not a2["ok"])
    check("its surplus", [int(round(x)) for x in a2["surplus"]],
          [24, 24, -12, -12])

    # ... and the same charges at half the size do not violate it, so the
    # test is sensitive to the charges rather than to the shape.
    ok2 = B.LineBundleSum(TETRA, [[2, -2, 0, 0], [-2, 2, 0, 0],
                                  [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    check_true("the same shape at charge 2 passes", ok2.anomaly()["ok"])


def test_stability():
    print("\n[5] slopes and poly-stability")

    d = np.array(TETRA.triple_intersection())

    # The degenerate case that matters: mu identically zero is compatible
    # with poly-stability, not an obstruction to it.
    check_true("trivial summand is not 'definite'",
               not B.slope_is_definite(d, [0, 0, 0, 0]))
    check_true("a positive summand is definite",
               B.slope_is_definite(d, [1, 1, 1, 1]))

    # The pool filter is the vectorised version of the same test.
    cand = np.array(list(itertools.product(range(-1, 2), repeat=4)))
    mask = B.slope_candidates(d, cand)
    byhand = np.array([not B.slope_is_definite(d, c) for c in cand])
    check_true("slope_candidates agrees with slope_is_definite",
               bool(np.array_equal(mask, byhand)))

    # A hand-built symmetric example that must be poly-stable at t = (1,1,1,1).
    S = [[1, -1, 0, 0], [0, 1, -1, 0], [0, 0, 1, -1], [-1, 0, 0, 1],
         [0, 0, 0, 0]]
    V = B.LineBundleSum(TETRA, S)
    mu = V.slopes([1.0, 1.0, 1.0, 1.0])
    check_true("slopes vanish at the symmetric point",
               bool(np.all(np.abs(mu) < 1e-9)))
    loc = V.stability_locus(tries=40)
    check_true("stability_locus finds it", loc["found"])
    check_true("and at an interior point", bool(np.all(loc["t"] > 1e-6)))

    # The scan model, found by the search and verified independently here.
    W = B.LineBundleSum(TETRA, TETRA_MODEL)
    locW = W.stability_locus(tries=40)
    check_true("scan model is poly-stable", locW["found"])
    check_true("its slopes really do vanish there",
               bool(np.all(np.abs(W.slopes(locW["t"])) < 1e-6)))

    # [5] proper: the cheap obstruction must never reject something the
    # numerical search accepts. A false positive here would silently delete
    # valid models from every scan.
    rng = np.random.default_rng(7)
    disagreements = 0
    tested = 0
    for _ in range(120):
        first = rng.integers(-2, 3, size=(4, 4))
        S = np.vstack([first, -first.sum(axis=0)])
        obstructed = B.slope_subsets_definite(d, S)
        found = B.stability_locus(TETRA, S, tries=6)["found"]
        tested += 1
        if obstructed and found:
            disagreements += 1
    check("obstruction never rejects a solvable bundle (%d tried)" % tested,
          disagreements, 0)

    # And it must actually obstruct something, or it is doing no work.
    check_true("obstruction fires on a definite summand",
               B.slope_subsets_definite(
                   d, [[1, 1, 1, 1], [-1, -1, -1, -1], [0, 0, 0, 0],
                       [0, 0, 0, 0], [0, 0, 0, 0]]))
    check_true("obstruction does not fire on the scan model",
               not B.slope_subsets_definite(d, TETRA_MODEL))


def test_monad():
    print("\n[6] monads")

    # 0 -> V -> B -> C -> 0 on the tetraquadric, rank 5 - 1 = 4.
    Bch = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
           [1, 1, 1, 1]]
    Cch = [[2, 2, 2, 2]]
    V = B.Monad(TETRA, Bch, Cch)
    check("rank(V) = rank(B) - rank(C)", V.rank, 4)
    check("c1(V)", list(V.c1), [0, 0, 0, 0])
    check_true("index routes agree for the monad",
               abs(V.index() - V.index_from_cohomology()) < 1e-9)

    bd = V.cohomology_bounds()
    lo = [a for a, _ in bd["bounds"]]
    hi = [b for _, b in bd["bounds"]]
    check_true("bounds are ordered", all(a <= b for a, b in bd["bounds"]))

    # The alternating sum of *any* admissible choice inside the bounds equals
    # the index, because the index is exact on the sequence. Check that the
    # index lies between the extreme alternating sums the bounds permit.
    alt_lo = lo[0] - hi[1] + lo[2] - hi[3]
    alt_hi = hi[0] - lo[1] + hi[2] - lo[3]
    check_true("index lies inside the bounds' alternating range",
               alt_lo - 1e-9 <= V.index() <= alt_hi + 1e-9)

    check_true("determined flags match zero-width intervals",
               bd["determined"] == [a == b for a, b in bd["bounds"]])

    check_true("cohomology_bounds returns hB and hC",
               len(bd["hB"]) == 4 and len(bd["hC"]) == 4)

    # -- the two impossibility checks, on the cases that found them ---------

    # h^3(B) < h^3(C) makes the sequence inexact: the trivial summand of C
    # contributes h^3(O_X) = 1 by Serre duality and B has nothing to map onto
    # it. Without the check the formula returns h^3(V) = -1.
    bad = B.Monad(QUINTIC, [[1], [1], [1]], [[0], [3]])
    check_raises("h^3(B) < h^3(C) is refused", B.NotABundle,
                 bad.cohomology_bounds)

    # Imposing h^3(V) = 0 forces one specific rank for H^2(B) -> H^2(C),
    # which need not be attainable. When it is not, no stable bundle arises.
    pinned = B.Monad(M7833, [[0, 1], [2, 0], [0, 2], [0, 0]],
                     [[1, 1], [2, 1], [-1, 1]])
    ok = pinned.cohomology_bounds()
    check("that monad pins h^3(V) at 1", ok["bounds"][3], (1, 1))
    check_raises("so stable=True is refused", B.NotABundle,
                 pinned.cohomology_bounds, stable=True)

    # -- a sweep, since one monad on one manifold proves very little --------

    rng = np.random.default_rng(1)
    tested = neg = idx_out = contain = alt = nb = st = 0
    for X in (QUINTIC, M7833, TETRA):
        h11 = X.len
        for _ in range(400):
            nbnd = int(rng.integers(3, 6))
            ncnd = nbnd - int(rng.integers(1, 3))
            Bc = rng.integers(0, 3, size=(nbnd, h11))
            Cc = (rng.integers(0, 3, size=(ncnd - 1, h11)) if ncnd > 1
                  else np.zeros((0, h11), dtype=int))
            Cc = np.vstack([Cc, Bc.sum(0) - Cc.sum(0)])
            try:
                M = B.Monad(X, Bc, Cc)
            except Exception:                                    # noqa: BLE001
                continue
            try:
                r = M.cohomology_bounds()
            except B.NotABundle:
                nb += 1
                continue
            except Exception:                                    # noqa: BLE001
                continue
            tested += 1
            lo = [a for a, _ in r["bounds"]]
            hi = [b for _, b in r["bounds"]]
            if min(lo) < 0:
                neg += 1
            if not (lo[0] - hi[1] + lo[2] - hi[3] <= r["index"]
                    <= hi[0] - lo[1] + hi[2] - lo[3]):
                idx_out += 1
            try:
                rs = M.cohomology_bounds(stable=True)
                st += 1
            except B.NotABundle:
                continue
            l2 = [a for a, _ in rs["bounds"]]
            h2 = [b for _, b in rs["bounds"]]
            if min(l2) < 0:
                neg += 1
            # A tightening must lie inside what it tightens.
            if any(l2[q] < lo[q] or h2[q] > hi[q] for q in range(4)):
                contain += 1
            if not (l2[0] - h2[1] + l2[2] - h2[3] <= r["index"]
                    <= h2[0] - l2[1] + h2[2] - l2[3]):
                alt += 1

    check_true("swept %d monads (%d refused, %d stable-admissible)"
               % (tested, nb, st), tested > 800)
    check("no negative cohomology dimension", neg, 0)
    check("index always inside the alternating range", idx_out, 0)
    check("stable bounds always inside the general ones", contain, 0)
    check("stable bounds always consistent with the index", alt, 0)

    # The tightening must actually tighten, or it is doing nothing.
    narrower = 0
    for Bc, Cc in (([[1, 0], [0, 1], [1, 1], [0, 0]], [[1, 1], [1, 1]]),
                   ([[1, 1], [1, 1], [2, 2]], [[2, 2], [2, 2]])):
        try:
            M = B.Monad(M7833, Bc, Cc)
            g = M.cohomology_bounds()
            t = M.cohomology_bounds(stable=True)
        except B.NotABundle:
            continue
        if sum(b - a for a, b in t["bounds"]) < sum(b - a for a, b in g["bounds"]):
            narrower += 1
    check_true("stable=True narrows the intervals", narrower > 0)
    check_true("and says so in the return value",
               pinned.cohomology_bounds()["assumed_stable"] is False)


def test_monad_scan():
    print("\n[6b] scan_monads")

    # positivity_ok is a real filter with a documented meaning.
    # Positivity needs every summand of C to *strictly* exceed every summand
    # of B in at least one direction. A B summand equal to a C summand fails
    # it, which is why [1,1,1,1] cannot appear on both sides.
    good = B.Monad(TETRA, [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],
                           [0, 0, 0, 1], [1, 1, 0, 0]],
                   [[1, 1, 1, 1], [2, 2, 2, 2]])
    check_true("a positive monad passes positivity_ok", good.positivity_ok())
    equal = B.Monad(TETRA, [[1, 1, 1, 1], [0, 1, 0, 0], [0, 0, 1, 0],
                            [0, 0, 0, 1], [1, 1, 0, 0]],
                    [[1, 1, 1, 1], [2, 2, 2, 2]])
    check_true("a B summand equal to a C summand fails it",
               not equal.positivity_ok())
    neg = B.Monad(TETRA, [[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],
                          [1, 1, 1, 1], [1, 1, 1, 1]],
                  [[1, 1, 1, 1], [1, 1, 1, 1]])
    check_true("a negative charge fails it", not neg.positivity_ok())

    # The scans run on CICY 7833 rather than the tetraquadric: h^{1,1} = 2
    # instead of 4 makes the box small enough for a test suite, and the
    # conditions being checked do not care which manifold they hold on.
    t0 = time.time()
    found = B.scan_monads(M7833, rank=4, nC=2, charge=3, generations=3,
                          symmetry_order=2, limit=12, max_seconds=45)
    dt = time.time() - t0
    check_true("scan_monads finds monads (%d in %.0fs)" % (len(found), dt),
               len(found) > 0)

    bad_c1 = bad_ind = bad_pos = 0
    for Bc, Cc in found:
        M = B.Monad(M7833, Bc, Cc)
        if np.any(M.c1 != 0):
            bad_c1 += 1
        if abs(M.index() + 6) > 1e-9:
            bad_ind += 1
        if not M.positivity_ok():
            bad_pos += 1
    check("every monad has c1 = 0", bad_c1, 0)
    check("every monad has ind(V) = -3|Gamma| = -6", bad_ind, 0)
    check("every monad is positive", bad_pos, 0)
    check_true("and every one survives cohomology_bounds",
               all(_bounds_ok(M7833, Bc, Cc) for Bc, Cc in found))

    # Neither target is reachable in the charge-2 box, and that box completes
    # rather than being truncated, so the emptiness is a real statement.
    for so in (1, 2):
        empty = B.scan_monads(M7833, rank=4, nC=2, charge=2, generations=3,
                              symmetry_order=so, max_seconds=45)
        check("no monad at charge 2 with |Gamma| = %d" % so, len(empty), 0)

    # Every hit here has a trivial summand in B, and that is worth asserting
    # rather than filtering away. O_X passes positivity -- all its charges are
    # zero, so every summand of C exceeds it somewhere -- and a V with a
    # trivial summand has a smaller structure group than advertised, so none
    # of these is a model. The degeneracy is not incidental to this box: it
    # is all of it.
    def no_trivial(Bc, Cc):
        return all(any(x) for x in Bc)

    check("hits with a trivial summand in B",
          sum(1 for Bc, Cc in found if not no_trivial(Bc, Cc)), len(found))

    # Running the same scan with that predicate as keep= therefore returns
    # nothing, which would make an "the filter works" assertion vacuous. So
    # the hook is checked directly instead, with a predicate that must reject
    # everything and a small budget.
    rejected = B.scan_monads(M7833, rank=4, nC=2, charge=3, generations=3,
                             symmetry_order=2, keep=lambda Bc, Cc: False,
                             limit=12, max_seconds=8)
    check("a keep= that rejects everything returns nothing", len(rejected), 0)


def _bounds_ok(X, Bc, Cc):
    try:
        B.Monad(X, Bc, Cc).cohomology_bounds()
        return True
    except B.NotABundle:
        return False


def test_scan():
    print("\n[7] scan: correctness and its budgets")

    # Every model the scan returns must satisfy the conditions it filters on.
    t0 = time.time()
    found = B.scan(TETRA, rank=5, charge=1, generations=3, symmetry_order=2,
                   limit=300)
    dt = time.time() - t0
    check_true("scan returns something at charge 1", len(found) > 0)
    d = np.array(TETRA.triple_intersection())
    c2X = np.array(TETRA.second_chern())
    bad_c1 = bad_ind = bad_anom = 0
    for S in found:
        V = B.LineBundleSum(TETRA, S)
        if np.any(V.c1 != 0):
            bad_c1 += 1
        if abs(V.index() + 6) > 1e-9:
            bad_ind += 1
        if not V.anomaly()["ok"]:
            bad_anom += 1
    check("every model has c1 = 0", bad_c1, 0)
    check("every model has ind(V) = -3|Gamma| = -6", bad_ind, 0)
    check("every model passes the anomaly", bad_anom, 0)
    check_true("scan at charge 1 is fast (%.1fs)" % dt, dt < 30)

    # No rank 5 sum on the tetraquadric within charge 1 has ind = -3. This is
    # a real statement, not an empty search: the only non-zero entry of d_rst
    # is 2, so 6*ind is even and ind = -3 is unreachable. It is why three
    # generations there needs a freely acting quotient, exactly as
    # phenomenology.generation_survey finds for the standard embedding.
    none3 = B.scan(TETRA, rank=5, charge=1, generations=3, symmetry_order=1,
                   limit=10)
    check("no ind = -3 model at charge 1", len(none3), 0)

    # limit is honoured exactly.
    capped = B.scan(TETRA, rank=5, charge=1, generations=3, symmetry_order=2,
                    limit=17)
    check("limit is respected", len(capped), 17)

    # max_seconds is honoured, on a box far too large to finish.
    t0 = time.time()
    B.scan(TETRA, rank=5, charge=2, generations=3, symmetry_order=2,
           require_stability=True, max_seconds=4)
    dt = time.time() - t0
    check_true("max_seconds is respected (%.1fs for a 4s budget)" % dt,
               dt < 20)

    # keep= is applied, and cuts.
    def no_repeats(S):
        return len(set(map(tuple, S))) == len(S)

    # Compare against an uncapped count, since two runs that both hit the
    # same limit are trivially equal and would make this check vacuous.
    everything = B.scan(TETRA, rank=5, charge=1, generations=3,
                        symmetry_order=2, limit=100000)
    filtered = B.scan(TETRA, rank=5, charge=1, generations=3,
                      symmetry_order=2, keep=no_repeats, limit=100000)
    check_true("keep= filter is applied",
               all(no_repeats(S) for S in filtered))
    check_true("keep= filter actually removes models (%d -> %d)"
               % (len(everything), len(filtered)),
               len(filtered) < len(everything))

    # With stability on, every survivor must genuinely be poly-stable.
    stable = B.scan(TETRA, rank=5, charge=2, generations=3, symmetry_order=2,
                    require_stability=True, limit=12, max_seconds=60)
    bad = 0
    for S in stable:
        if not B.stability_locus(TETRA, S, tries=40)["found"]:
            bad += 1
    check("every require_stability model is poly-stable", bad, 0)
    check_true("and the sign obstruction agrees on all of them",
               all(not B.slope_subsets_definite(d, S) for S in stable))


def test_guards():
    print("\n[8] guards")

    # A genuinely non-favourable CICY: h^{1,1} = 19 but only three ambient
    # factors, so H^2(X) is not spanned by their restrictions and a line
    # bundle is not a list of three integers.
    nonfav = CICY([[1, 1, 1], [2, 0, 3], [2, 3, 0]])
    check_true("the test manifold really is non-favourable", not nonfav.fav)
    check_raises("non-favourable configuration refused", B.NotFavourable,
                 B.LineBundleSum, nonfav,
                 [[1, -1, 0], [-1, 1, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
    check_raises("wrong charge length refused", ValueError,
                 B.LineBundleSum, TETRA, [[1, -1], [-1, 1]])
    check_raises("scan refuses rank < 4", ValueError,
                 B.scan, TETRA, 3, 1)
    V = B.LineBundleSum(TETRA, TETRA_MODEL)
    check_true("repr works", "LineBundleSum" in repr(V))
    check_true("describe works", "rank=5" in V.describe())


def main():
    t0 = time.time()
    test_topology()
    test_cohomology()
    test_spectrum()
    test_anomaly()
    test_stability()
    test_monad()
    test_monad_scan()
    test_scan()
    test_guards()

    print("\n" + "=" * 72)
    if FAILURES:
        print("FAILED (%d): %s" % (len(FAILURES), ", ".join(FAILURES)))
        return 1
    print("test_bundles: all checks passed in %.1fs" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
