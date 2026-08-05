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


def test_permutation_action():
    print("\n[2b] actions that permute the ambient factors")

    # The regression oracle this class was built against: with sigma the
    # identity it must reproduce CyclicAction exactly, since every cycle has
    # length one and the product over cycles becomes a product over factors.
    A = E.CyclicAction(TETRA_CONF, 2, [[0, 1]] * 4, [0])
    P = E.PermutationAction(TETRA_CONF, 2, [0, 1, 2, 3], [[0, 1]] * 4, [0])
    bad = sum(1 for k in itertools.product((-2, -1, 0, 1, 2), repeat=4)
              if A.euler(list(k)) != P.euler(list(k)))
    check("sigma = identity reproduces CyclicAction on 625 bundles", bad, 0)

    # Only sigma-invariant bundles carry an equivariant structure at all.
    Q = E.PermutationAction(TETRA_CONF, 2, [1, 0, 3, 2], [[0, 0]] * 4, [0])
    check_true("a sigma-invariant bundle is accepted",
               Q.is_invariant([1, 1, 2, 2]))
    check_true("a non-invariant one is not", not Q.is_invariant([1, 0, 2, 2]))
    check_true("and euler refuses it", _raises(Q.euler, [1, 0, 2, 2]))
    check("invariant charges in [-2,2]: one per cycle, not per factor",
          len(Q.invariant_charges(-2, 2)), 25)

    # check_order composes n maps, not n times the cycle length -- those are
    # different group elements, and the earlier version used the wrong one.
    # It accepted a Z_4 action as a Z_2 one, and the integrality of the
    # multiplicities did not catch it, which is why euler consults it.
    wrong = E.PermutationAction(TETRA_CONF, 2, [1, 0, 3, 2],
                                [[0, 0], [0, 1], [0, 0], [0, 1]], [0])
    ok, msgs = wrong.check_order()
    check_true("a Z_4 action declared as Z_2 is rejected", not ok)
    check_true("with a message naming the factor", "factor 0" in msgs[0])
    check_true("and euler refuses to compute for it",
               _raises(wrong.euler, [1, 1, 1, 1]))

    right = E.PermutationAction(TETRA_CONF, 2, [1, 0, 3, 2], [[0, 1]] * 4, [0])
    check_true("a genuine Z_2 passes the order check", right.check_order()[0])

    # A configuration whose degrees are not sigma-invariant cannot work,
    # since this class does not permute the defining polynomials.
    check_true("a non-invariant degree column is refused",
               _raises(E.PermutationAction, [[1, 2], [1, 1]], 2, [1, 0],
                       [[0, 0], [0, 0]], [0]))
    check_true("a dimension-changing permutation is refused",
               _raises(E.PermutationAction, [[1, 2], [2, 2]], 2, [1, 0],
                       [[0, 0], [0, 0, 0]], [0]))

    # No cyclic action permuting the factors of the tetraquadric is free.
    # Fixed points need an eigenvector of the composite map around each cycle,
    # and a linear map always has one; requiring g^n = id forces that
    # composite to be scalar, so the fixed locus is positive-dimensional and
    # X cannot avoid it. Exhaustively over every valid Z_2:
    valid = free = 0
    for perm in itertools.permutations(range(4)):
        if list(perm) == [0, 1, 2, 3]:
            continue
        for w in itertools.product(itertools.product(range(2), repeat=2),
                                   repeat=4):
            for pc in range(2):
                try:
                    Z = E.PermutationAction(TETRA_CONF, 2, list(perm),
                                            [list(x) for x in w], [pc])
                except Exception:                                # noqa: BLE001
                    continue
                if not Z.check_order()[0]:
                    continue
                valid += 1
                if Z.looks_free(probes=Z.invariant_charges(-1, 1))[0]:
                    free += 1
    check_true("valid Z_2 factor-permuting actions examined (%d)" % valid,
               valid > 1000)
    check("none of them is free", free, 0)


def test_abelian_and_free_permuting():
    print("\n[2c] abelian groups, and a free factor-permuting action")

    # Regression: one generator must reproduce PermutationAction exactly.
    P = E.PermutationAction(TETRA_CONF, 2, [0, 1, 2, 3], [[0, 1]] * 4, [0])
    A = E.AbelianAction(TETRA_CONF, [2], [[0, 1, 2, 3]], [[[0, 1]] * 4], [[0]])
    check_true("AbelianAction validates", A.check()[0])
    bad = sum(1 for k in itertools.product((-2, -1, 0, 1, 2), repeat=4)
              if [A.euler(list(k))[(c,)] for c in range(2)] != P.euler(list(k)))
    check("one generator reproduces PermutationAction on 625 bundles", bad, 0)

    # The consistency checks are separate conditions and each must bite.
    bad_order = E.AbelianAction(TETRA_CONF, [2], [[1, 0, 3, 2]],
                                [[[0, 0], [0, 1], [0, 0], [0, 1]]], [[0]])
    check_true("a mis-declared order is caught", not bad_order.check()[0])
    check_true("and euler refuses it", _raises(bad_order.euler, [1, 1, 1, 1]))

    # invariant_charges works on orbits of the whole group, not one generator.
    two = E.AbelianAction(TETRA_CONF, [2, 2], [[1, 0, 2, 3], [0, 1, 3, 2]],
                          [[[0, 0]] * 4, [[0, 0]] * 4], [[0], [0]])
    check("two transpositions leave two orbits, so 25 invariant charges",
          len(two.invariant_charges(-2, 2)), 25)

    # -- the free factor-permuting action ---------------------------------

    # Last session established that no cyclic action permuting the factors of
    # the tetraquadric can be free, because g^n = id forces the composite
    # around each cycle to be scalar and a scalar composite has a
    # positive-dimensional fixed locus. The escape is an action whose SQUARE
    # is a non-scalar phase -- which makes it order 4, not 2.
    F = E.PermutationAction(TETRA_CONF, 4, [1, 0, 3, 2],
                            [[0, 0], [0, 2], [0, 0], [0, 2]], [0])
    check_true("it is a genuine Z_4", F.check_order()[0])
    free, _ = F.looks_free()
    check_true("and it is free, unlike every Z_2 of this shape", free)

    # Its index totals must still agree with line_co_euler.
    ks = F.invariant_charges(-2, 2)
    worst = max(abs(sum(F.euler(k)) - float(TETRA.line_co_euler(k)))
                for k in ks)
    check_true("totals match line_co_euler on %d invariant bundles (%.1e)"
               % (len(ks), worst), worst < 1e-6)
    check_true("every character is equidistributed over Z_4",
               all(E.is_regular_multiple(F.euler(k))[0] for k in ks))

    # The same shape at order 2 -- scalar square -- is not free, which is the
    # contrast the whole argument rests on.
    G = E.PermutationAction(TETRA_CONF, 2, [1, 0, 3, 2], [[0, 1]] * 4, [0])
    check_true("the order-2 version is valid", G.check_order()[0])
    check_true("but not free", not G.looks_free()[0])


def test_polynomial_permutations():
    print("\n[2d] permuting the defining polynomials")

    # pi = identity must reproduce the previous behaviour exactly.
    A = E.CyclicAction(TETRA_CONF, 2, [[0, 1]] * 4, [0])
    P = E.PermutationAction(TETRA_CONF, 2, [0, 1, 2, 3], [[0, 1]] * 4, [0])
    bad = sum(1 for k in itertools.product((-2, -1, 0, 1, 2), repeat=4)
              if A.euler(list(k)) != P.euler(list(k)))
    check("pi = identity is unchanged, 625 bundles", bad, 0)

    # -- the oracle -------------------------------------------------------
    #
    # The sign from the induced permutation of each wedge factor is the part
    # that cannot be checked at the identity, so it needs an independent
    # route. Here is one. On a configuration whose two defining polynomials
    # have identical multidegree, swapping them is -- in the eigenbasis
    # p_+- = p_1 +- p_2 -- exactly the phase-only action with charges 0 and 1,
    # which the already-tested code handles. The two must agree.
    ORACLE = [[1, 1, 1]] * 5          # a favourable CY3, chi = -80
    ident = list(range(5))
    swapped = E.PermutationAction(ORACLE, 2, ident, [[0, 0]] * 5, [0, 0],
                                  polynomial_perm=[1, 0])
    diagonalised = E.PermutationAction(ORACLE, 2, ident, [[0, 0]] * 5, [0, 1])
    ks = [[k[0]] * 3 + [k[1]] * 2
          for k in itertools.product((-2, -1, 0, 1, 2), repeat=2)]
    check("swapping two polynomials equals the diagonalised phase action",
          sum(1 for k in ks if swapped.euler(k) != diagonalised.euler(k)), 0)

    # ... and the oracle is sensitive to the sign, which is the whole point.
    # Forcing it wrong breaks 16 of the 25, while the identity-total check
    # below passes either way and would have shipped the bug.
    orig = E._restricted_sign
    try:
        E._restricted_sign = lambda perm, S: 1
        broken = E.PermutationAction(ORACLE, 2, ident, [[0, 0]] * 5, [0, 0],
                                     polynomial_perm=[1, 0])
        wrong = sum(1 for k in ks if broken.euler(k) != diagonalised.euler(k))
        check_true("a forced +1 sign is caught by the oracle (%d of %d)"
                   % (wrong, len(ks)), wrong > 0)
        X = CICY(ORACLE)
        worst = max(abs(sum(broken.euler(k)) - float(X.line_co_euler(k)))
                    for k in ks)
        check_true("but NOT by the total at the identity (%.1e)" % worst,
                   worst < 1e-6)
    finally:
        E._restricted_sign = orig

    # The sign itself, directly: a transposition inside an invariant subset.
    check("sign of a transposition on its own support",
          E._restricted_sign([1, 0], (0, 1)), -1)
    check("sign of the identity", E._restricted_sign([0, 1], (0, 1)), 1)
    check("a 3-cycle is even", E._restricted_sign([1, 2, 0], (0, 1, 2)), 1)

    # -- compatibility ----------------------------------------------------
    #
    # Permuting the polynomials must be compatible with permuting the factors:
    # d[sigma(i)][pi(a)] = d[i][a]. Neither permutation alone need preserve
    # the degree matrix, only the pair.
    check_true("swapping polynomials alone is refused when degrees differ",
               _raises(E.PermutationAction, [[1, 1, 2], [1, 2, 1]], 2, [0, 1],
                       [[0, 0], [0, 0]], [0, 0], polynomial_perm=[1, 0]))
    both = E.PermutationAction([[1, 1, 2], [1, 2, 1]], 2, [1, 0],
                               [[0, 0], [0, 0]], [0, 0], polynomial_perm=[1, 0])
    check_true("but swapping factors and polynomials together is fine",
               both is not None)
    check_true("a non-permutation is refused",
               _raises(E.PermutationAction, ORACLE, 2, ident, [[0, 0]] * 5,
                       [0, 0], polynomial_perm=[0, 0]))

    # Both permutations non-trivial at once, checked the only way available.
    Q = E.PermutationAction(ORACLE, 2, [1, 0, 3, 2, 4], [[0, 0]] * 5, [0, 0],
                            polynomial_perm=[1, 0])
    X = CICY(ORACLE)
    worst = max(abs(sum(Q.euler(k)) - float(X.line_co_euler(k)))
                for k in Q.invariant_charges(-2, 2))
    check_true("sigma and pi together: totals still match line_co_euler "
               "(%.1e)" % worst, worst < 1e-6)


def test_abelian_polynomial_permutations():
    print("\n[2e] polynomial permutations through AbelianAction")

    ORACLE = [[1, 1, 1]] * 5
    ident = list(range(5))
    ks = [[k[0]] * 3 + [k[1]] * 2
          for k in itertools.product((-2, -1, 0, 1, 2), repeat=2)]
    X = CICY(ORACLE)

    # One generator must reproduce PermutationAction, pi included.
    P = E.PermutationAction(ORACLE, 2, ident, [[0, 0]] * 5, [0, 0],
                            polynomial_perm=[1, 0])
    A = E.AbelianAction(ORACLE, [2], [ident], [[[0, 0]] * 5], [[0, 0]],
                        polynomial_perms=[[1, 0]])
    check("one generator with pi reproduces PermutationAction",
          sum(1 for k in ks
              if [A.euler(k)[(c,)] for c in range(2)] != P.euler(k)), 0)

    # The same diagonalisation oracle, through the abelian path.
    D = E.AbelianAction(ORACLE, [2], [ident], [[[0, 0]] * 5], [[0, 1]])
    check("swap equals the diagonalised phase action, via AbelianAction",
          sum(1 for k in ks if A.euler(k) != D.euler(k)), 0)

    # ... and it is still sensitive here, which is the point of repeating it.
    orig = E._restricted_sign
    try:
        E._restricted_sign = lambda perm, S: 1
        broken = E.AbelianAction(ORACLE, [2], [ident], [[[0, 0]] * 5],
                                 [[0, 0]], polynomial_perms=[[1, 0]])
        wrong = sum(1 for k in ks if broken.euler(k) != D.euler(k))
        worst = max(abs(sum(broken.euler(k).values())
                        - float(X.line_co_euler(k))) for k in ks)
        check_true("a forced +1 sign is caught (%d of %d)" % (wrong, len(ks)),
                   wrong > 0)
        check_true("and still not by the identity total (%.1e)" % worst,
                   worst < 1e-6)
    finally:
        E._restricted_sign = orig

    # A genuine two-generator case: one swaps the polynomials, one phases.
    G = E.AbelianAction(ORACLE, [2, 2], [ident, ident],
                        [[[0, 0]] * 5, [[0, 1]] * 5], [[0, 0], [0, 0]],
                        polynomial_perms=[[1, 0], [0, 1]])
    ok, msgs = G.check()
    check_true("Z_2 x Z_2 with a polynomial swap validates", ok)
    check("its order", G.order, 4)
    worst = max(abs(sum(G.euler(k).values()) - float(X.line_co_euler(k)))
                for k in G.invariant_charges(-2, 2))
    check_true("totals match line_co_euler (%.1e)" % worst, worst < 1e-6)
    check("and the character has one entry per element of the dual group",
          len(G.euler([1, 1, 1, 1, 1])), 4)

    # Non-commuting polynomial permutations must be refused. Two transpositions
    # on three polynomials do not commute, and nothing about the factors would
    # reveal it -- the factor permutations here are both the identity.
    H = E.AbelianAction([[1, 1, 1, 1]] * 7, [2, 2],
                        [list(range(7))] * 2, [[[0, 0]] * 7] * 2,
                        [[0, 0, 0], [0, 0, 0]],
                        polynomial_perms=[[1, 0, 2], [0, 2, 1]])
    ok2, msgs2 = H.check()
    check_true("non-commuting polynomial permutations are caught", not ok2)
    check_true("with a message naming the polynomials",
               "defining polynomials" in msgs2[0])
    check_true("and euler refuses", _raises(H.euler, [1] * 7))

    # Compatibility is per generator.
    check_true("an incompatible generator is refused",
               _raises(E.AbelianAction, [[1, 1, 2], [1, 2, 1]], [2],
                       [[0, 1]], [[[0, 0], [0, 0]]], [[0, 0]],
                       polynomial_perms=[[1, 0]]))


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
    test_permutation_action()
    test_abelian_and_free_permuting()
    test_polynomial_permutations()
    test_abelian_polynomial_permutations()
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
