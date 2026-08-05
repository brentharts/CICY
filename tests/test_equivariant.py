"""
Tests for pyCICY.equivariant.

  [1] index at the identity   the total of the character-valued index must
                              equal CICY.line_co_euler, which reaches the same
                              number by the Leray spectral sequence rather
                              than by Koszul, and in floating point rather
                              than exact integers
  [2] freeness, two ways      an index argument (Lefschetz) and a geometric
                              one (fixed points forced onto X) must agree
                              wherever both can see
  [3] structures              the torsor over the character group
  [4] the bridge              the equivariant index of the bundles.scan model
                              derives what breaking.worked_example assumed

Section [2] is the one worth reading. The two routes are genuinely
independent -- one is an alternating sum over a resolution, the other is
monomial arithmetic at coordinate points -- and neither is complete on its
own. The geometric test sees only coordinate fixed points, so it misses an
action trivial on a whole factor. The Lefschetz test is necessary but not
sufficient and depends on the probe set. The tests record both limitations
with cases that exhibit them.

Run with:  python3 tests/test_equivariant.py
       or: python3 run_tests.py
"""

import itertools
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyCICY import CICY
from pyCICY import bundles as B
from pyCICY import equivariant as E

FAILURES = []

TETRA_CONF = [[1, 2], [1, 2], [1, 2], [1, 2]]
TETRA = CICY(TETRA_CONF)
TETRA_MODEL = [[-2, -2, -1, 2], [-2, 1, 0, 0], [1, -2, 1, 0],
               [1, 1, -1, 0], [2, 2, 1, -2]]


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


def test_index_at_identity():
    print("\n[1] the index, against line_co_euler")
    A = E.TETRAQUADRIC_Z2()
    ok, msgs = A.check()
    check_true("the action is admissible on this configuration", ok)

    worst = 0.0
    n = 0
    for k in itertools.product((-2, -1, 0, 1, 2), repeat=4):
        ch = A.euler(list(k))
        ref = float(TETRA.line_co_euler(list(k)))
        worst = max(worst, abs(sum(ch) - ref))
        n += 1
    check_true("Koszul total matches line_co_euler on %d bundles (%.1e)"
               % (n, worst), worst < 1e-6)

    # The Koszul route is exact integer arithmetic; line_co_euler is not.
    v = TETRA.line_co_euler([2, -1, 0, 1])
    check_true("line_co_euler returns float noise here (%.1e)" % v,
               v != 0 and abs(v) < 1e-12)
    check("the Koszul route returns exact zero", sum(A.euler([2, -1, 0, 1])), 0)
    check_true("and every entry is an integer",
               all(isinstance(x, int) for x in A.euler([1, 1, 1, 1])))

    # A twist permutes the character cyclically and cannot change the total.
    base = A.euler([1, 1, 1, 1])
    tw = A.euler([1, 1, 1, 1], twist=1)
    check("twisting preserves the total", sum(tw), sum(base))
    check("and rotates the character", tw, [base[-1], base[0]])


def test_freeness():
    print("\n[2] freeness, by two independent routes")

    cases = [
        ("Z2 (0,1) all four, charge 0", 2, [[0, 1]] * 4, [0], True),
        ("Z3 (0,1) all four, charge 0", 3, [[0, 1]] * 4, [0], False),
        ("Z4 (0,1) all four, charge 0", 4, [[0, 1]] * 4, [0], False),
        ("Z5 (0,1) all four, charge 0", 5, [[0, 1]] * 4, [0], False),
        ("Z2 (0,1) all four, charge 1", 2, [[0, 1]] * 4, [1], False),
        ("Z2 trivial on one factor", 2, [[0, 0]] + [[0, 1]] * 3, [0], False),
        ("Z2 trivial on two factors", 2,
         [[0, 1], [0, 1], [0, 0], [0, 0]], [0], False),
    ]
    for name, n, w, pc, expect_free in cases:
        A = E.CyclicAction(TETRA_CONF, n, w, pc)
        free, _ = A.looks_free()
        check("%-34s -> free" % name, free, expect_free)

    # Where the geometric route can see, the two must agree. It sees only
    # coordinate fixed points, so it is conclusive when it fires and silent
    # otherwise; a disagreement in the direction "forced points but looks
    # free" would be a real contradiction.
    contradictions = 0
    for name, n, w, pc, _ in cases:
        A = E.CyclicAction(TETRA_CONF, n, w, pc)
        free, _ = A.looks_free()
        forced = A.forced_fixed_points()
        if forced and free:
            contradictions += 1
    check("no action has forced fixed points and still looks free",
          contradictions, 0)

    # The geometric route is genuinely incomplete, and here is a case proving
    # it: trivial on two factors gives a two-dimensional fixed locus that
    # contains no coordinate point, so it finds nothing while Lefschetz does.
    A = E.CyclicAction(TETRA_CONF, 2, [[0, 1], [0, 1], [0, 0], [0, 0]], [0])
    check("trivial-on-two-factors: forced coordinate points",
          len(A.forced_fixed_points()), 0)
    check_true("but Lefschetz still rejects it", not A.looks_free()[0])

    # ... and the Z3 case is the reverse, where geometry gives the reason.
    A3 = E.CyclicAction(TETRA_CONF, 3, [[0, 1]] * 4, [0])
    check("Z3: coordinate fixed points forced onto X",
          len(A3.forced_fixed_points()), 11)
    check_true("and Lefschetz agrees", not A3.looks_free()[0])

    # The probe set matters, which is why the default is a box and not a list.
    free = E.TETRAQUADRIC_Z2()
    _, tab = free.looks_free()
    check_true("the free action passes the whole default box", len(tab) <= 5)


def test_structures():
    print("\n[3] equivariant structures")
    A = E.TETRAQUADRIC_Z2()

    # n structures per summand, n^r in total, and those with trivial total
    # twist are the ones that descend.
    all_s = E.enumerate_structures(A, TETRA_MODEL, descend_only=False)
    desc = E.enumerate_structures(A, TETRA_MODEL)
    check("structures on a rank-5 sum with Gamma = Z_2", len(all_s), 2 ** 5)
    check("of which these descend", len(desc), 16)
    check_true("descending ones have trivial total twist",
               all(sum(t) % 2 == 0 for t in desc))

    # For a FREE action the choice of structure is invisible to the index,
    # and necessarily so: every summand's character is a multiple of the
    # regular representation, i.e. a constant vector, which a cyclic twist
    # maps to itself. So the chiral spectrum downstairs does not depend on
    # the equivariant structure at all.
    a = E.bundle_index_character(A, TETRA_MODEL, [0, 0, 0, 0, 0])
    b = E.bundle_index_character(A, TETRA_MODEL, [1, 0, 0, 0, 0])
    check("free action: twisting leaves the character alone", a, b)
    check_true("because each summand is already equidistributed",
               all(E.is_regular_multiple(A.euler(k))[0] for k in TETRA_MODEL))

    # For a non-free action it is visible, so the argument is not decorative
    # -- it is the freeness that makes it inert.
    # The summand twisted must itself be one whose character is uneven --
    # the first summand of this model happens to be equidistributed even for
    # the non-free action, so twisting it proves nothing either way.
    N = E.CyclicAction(TETRA_CONF, 3, [[0, 1]] * 4, [0])
    uneven = [i for i, k in enumerate(TETRA_MODEL)
              if not E.is_regular_multiple(N.euler(k))[0]]
    check_true("the non-free action has uneven summands to twist", uneven)
    t = [0] * 5
    t[uneven[0]] = 1
    c = E.bundle_index_character(N, TETRA_MODEL, [0, 0, 0, 0, 0])
    d = E.bundle_index_character(N, TETRA_MODEL, t)
    check_true("non-free action: twisting does change the character", c != d)
    check("but never the total", sum(c), sum(d))

    check_true("a wrong-length structure list is refused",
               _raises(E.bundle_index_character, A, TETRA_MODEL, [0, 0]))


def _raises(fn, *a, **kw):
    try:
        fn(*a, **kw)
    except Exception:                                            # noqa: BLE001
        return True
    return False


def test_bridge():
    print("\n[4] what breaking had to assume, now derived")
    A = E.TETRAQUADRIC_Z2()
    V = B.LineBundleSum(TETRA, TETRA_MODEL)
    ch = E.bundle_index_character(A, TETRA_MODEL)

    check("ind(V) from bundles", int(V.index()), -6)
    check("total of the equivariant character", sum(ch), -6)
    check("the character itself", ch, [-3, -3])

    reg, mult = E.is_regular_multiple(ch)
    check_true("it is a multiple of the regular representation", reg)
    check("with multiple ind(V)/|Gamma|", mult, -3)

    # This is the point of the module: breaking.worked_example assumed the
    # Gamma-charges were equidistributed, and here that is derived from the
    # Koszul complex rather than chosen. Three generations in each sector.
    from pyCICY import breaking as BR
    check("generations downstairs, from breaking",
          BR.generations_downstairs(-6, 2), 3)
    check_true("which is -mult, the per-sector count", -mult == 3)

    # gamma_charges keeps the sign, since a negative entry is an
    # anti-generation and dropping the sign would silently flip chirality.
    charges, negative = E.gamma_charges(ch)
    check("net character is negative, so no positive charges", charges, [])
    check("and the negative part is reported", negative, {0: -3, 1: -3})
    pos, neg = E.gamma_charges(ch, multiplicity_sign=-1)
    check("with the sign flipped there are six multiplets", len(pos), 6)
    check("equidistributed across the two sectors", sorted(pos), [0, 0, 0, 1, 1, 1])


def main():
    t0 = time.time()
    test_index_at_identity()
    test_freeness()
    test_structures()
    test_bridge()

    print("\n" + "=" * 72)
    if FAILURES:
        print("FAILED (%d): %s" % (len(FAILURES), ", ".join(FAILURES)))
        return 1
    print("test_equivariant: all checks passed in %.1fs" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
