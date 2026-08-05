"""
Tests for pyCICY.breaking.

  [1] branching     computed from the hypercharge generator, and checked
                    against pyCICY.flavor.SM_HYPERCHARGES -- a table written
                    for a different paper that never referred to this one
  [2] anomaly       Tr(Y) over 10 + 5bar, by a route different from flavor's
  [3] Wilson lines  enumeration against the closed form n - gcd(n, 5)
  [4] projection    the worked example, and the consistency conditions
  [5] boundary      that the module refuses to invent what it cannot derive

Section [5] is unusual and deliberate. The central limitation of this module
is that the Gamma-representation on cohomology is not a function of anything
the package holds, so project() takes it as an argument. A test suite can at
least check that the argument genuinely drives the answer -- if two very
different charge assignments gave the same spectrum, the input would be
decorative.

Run with:  python3 tests/test_breaking.py
       or: python3 run_tests.py
"""

import math
import os
import sys
import time
from fractions import Fraction as F

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyCICY import breaking as B

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


def check_raises(name, exc, fn, *a, **kw):
    try:
        fn(*a, **kw)
    except exc:
        print("  {:<58} {:>12} ok".format(name, "raised"))
        return
    except Exception as e:                                       # noqa: BLE001
        print("  {:<58} {:>12} FAIL wrong exception {!r}".format(
            name, "raised", e))
        FAILURES.append(name)
        return
    print("  {:<58} {:>12} FAIL no exception".format(name, "-"))
    FAILURES.append(name)


def test_branching():
    print("\n[1] SU(5) -> SU(3) x SU(2) x U(1)")
    Y = B.hypercharge_generator()
    check("Y is traceless", sum(Y), F(0))
    check("Y in the 5", [str(y) for y in Y],
          ["-1/3", "-1/3", "-1/3", "1/2", "1/2"])

    ten = {n: (Y, m) for n, d, Y, m in B.branching("10")}
    check("10 contains (3,2)_{1/6}", str(ten["(3, 2)"][0]), "1/6")
    check("10 contains (3bar,1)_{-2/3}", str(ten["(3bar, 1)"][0]), "-2/3")
    check("10 contains (1,1)_{1}", str(ten["(1, 1)"][0]), "1")
    check("state count of the 10", sum(m for _, _, _, m in B.branching("10")), 10)
    check("state count of the 5bar",
          sum(m for _, _, _, m in B.branching("5bar")), 5)
    check("state count of the 24",
          sum(m for _, _, _, m in B.branching("24")), 24)

    ok, table = B.verify_against_flavor()
    check_true("branching agrees with flavor.SM_HYPERCHARGES", ok)
    for name, y, field, target, sign, good in table:
        check_true("  %-11s Y=%-5s <-> %s" % (name, y, field), good)


def test_anomaly():
    print("\n[2] anomaly cancellation")
    check("Tr(Y) over 10 + 5bar", B.anomaly_trace_of_generation(), F(0))

    # The same zero, reached in flavor.py by summing over fields with colour
    # multiplicities rather than by tracing over SU(5) representations.
    from pyCICY import flavor
    check("flavor reaches the same zero differently",
          flavor.anomaly_trace(), F(0))

    # Not vacuous: the individual pieces are large and of both signs.
    pieces = [Y * m for _, _, Y, m in B.branching("10") + B.branching("5bar")]
    check_true("the contributions do not vanish individually",
               any(p != 0 for p in pieces))
    check_true("and come with both signs",
               any(p > 0 for p in pieces) and any(p < 0 for p in pieces))


def test_wilson():
    print("\n[3] Wilson lines")
    bad = 0
    for n in range(1, 40):
        if len(B.wilson_lines(n)) != B.wilson_line_count(n):
            bad += 1
    check("enumeration matches n - gcd(n,5) for n up to 39", bad, 0)

    check("Z_1 admits none", B.wilson_line_count(1), 0)
    check("Z_5 admits none", B.wilson_line_count(5), 0)
    check_true("and those are the only two orders that fail",
               all(B.wilson_line_count(n) > 0
                   for n in range(1, 60) if n not in (1, 5)))

    check("the smallest order that works", B.minimal_order(), 2)
    check("Z_2 has exactly one", B.wilson_lines(2), [(0, 1)])
    check("and it breaks to the Standard Model",
          B.unbroken_group(0, 1, 2), "SU(3) x SU(2) x U(1)")
    check_true("a central Wilson line breaks nothing",
               B.unbroken_group(0, 0, 5) == "SU(5)")

    # Every enumerated line really does break to the SM, and the determinant
    # condition is enforced rather than assumed.
    bad = 0
    for n in (2, 3, 4, 6, 7, 10):
        for p, q in B.wilson_lines(n):
            if not B.breaks_to_sm(p, q, n):
                bad += 1
            if (3 * p + 2 * q) % n != 0:
                bad += 1
    check("all enumerated lines break to the SM with det = 1", bad, 0)

    try:
        B.unbroken_group(1, 1, 4)      # 3 + 2 = 5, not 0 mod 4
        check_true("a determinant violation is rejected", False)
    except ValueError:
        check_true("a determinant violation is rejected", True)


def test_projection():
    print("\n[4] the quotient spectrum")
    r = B.worked_example()
    check("generations downstairs", r["generations"], 3)
    check("which is -ind(V)/|Gamma| = 6/2", r["expected_generations"], 3)
    check_true("consistent", r["consistent"])
    check("hypercharge anomaly downstairs", r["anomaly"], F(0))

    # Doublet-triplet splitting really happens, and is due to the Wilson line.
    charges = {"10": [0] * 12 + [1] * 12, "10bar": [0] * 9 + [1] * 9,
               "5bar": [1] * 3, "5": [1] * 3}
    dt = B.doublet_triplet_split(charges, 2, (0, 1))
    check("surviving Higgs doublets", dt["doublets"], 6)
    check("surviving colour triplets", dt["triplets"], 0)
    check_true("split", dt["split"])

    # ... and it is the *asymmetry* of the Higgs charges that does it. With
    # the charges split evenly both survive and the mechanism is invisible.
    sym = dict(charges, **{"5bar": [0] * 3 + [1] * 3, "5": [0] * 3 + [1] * 3})
    dt2 = B.doublet_triplet_split(sym, 2, (0, 1))
    check("symmetric charges leave triplets behind", dt2["triplets"], 3)
    check_true("so the split is not automatic", not dt2["split"])

    # The anomaly must be weighted by the width of each piece. Dropping the
    # width gives a number that looks like an anomaly and is not, so check
    # the widths are what the branching says.
    check("width of a surviving (3,2)", r["widths"][("10", "(3, 2)", F(1, 6))], 6)
    check("width of a surviving (1,1)", r["widths"][("10", "(1, 1)", F(1))], 1)

    # A non-divisible index is refused rather than rounded.
    try:
        B.generations_downstairs(-7, 2)
        check_true("a non-divisible index is refused", False)
    except ValueError:
        check_true("a non-divisible index is refused", True)
    check("and a divisible one is not", B.generations_downstairs(-6, 2), 3)


def test_chiral_spectrum():
    print("\n[4b] the chiral spectrum, derived end to end")
    from pyCICY import equivariant as E

    MODEL = [[-2, -2, -1, 2], [-2, 1, 0, 0], [1, -2, 1, 0],
             [1, 1, -1, 0], [2, 2, 1, -2]]
    A = E.TETRAQUADRIC_Z2()
    r = B.chiral_spectrum(A, MODEL, wilson=(0, 1))

    check("ind(V) upstairs", r["index_V"], -6)
    check("generations downstairs", r["generations"], 3)
    check("which is -ind(V)/|Gamma|",
          B.generations_downstairs(r["index_V"], r["gamma_order"]), 3)
    check("hypercharge anomaly", r["anomaly"], F(0))

    # Complete SU(5) generations: every piece has the same multiplicity.
    mults = set(r["spectrum"].values())
    check("every Standard Model piece has the same net multiplicity",
          mults, {3})

    # ... and that is forced. For a free action each index character is a
    # multiple of the regular representation, so it is constant, so every
    # Wilson shift lands on the same multiplicity and the Wilson line cannot
    # split the chiral spectrum at all.
    check_true("all three characters are equidistributed",
               all(r["equidistributed"].values()))
    same = 0
    for w in [(0, 1), None]:
        if B.chiral_spectrum(A, MODEL, wilson=w)["spectrum"] == r["spectrum"]:
            same += 1
    check("the Wilson line makes no difference to the chiral content",
          same, 2)

    # This is not a contradiction with doublet_triplet_split, which acts on
    # the *non-chiral* content -- the vector-like pairs an index cannot see.
    # Both statements hold at once, and the tests assert them side by side.
    charges = {"10": [0] * 12 + [1] * 12, "10bar": [0] * 9 + [1] * 9,
               "5bar": [1] * 3, "5": [1] * 3}
    dt = B.doublet_triplet_split(charges, 2, (0, 1))
    check_true("the Wilson line still splits the vector-like content",
               dt["split"])

    # The widths must match the branching, since the anomaly is weighted by
    # them and pairing them wrongly would give a plausible non-zero number.
    check("width of a (3,2)", r["widths"][("10", "(3, 2)", F(1, 6))], 6)
    check("width of a (1,2)", r["widths"][("5bar", "(1, 2)", F(-1, 2))], 2)

    # The character of V (x) V* is the singlet index, and it vanishes: the
    # singlets are non-chiral, so no count of them is determined here.
    check("V (x) V* has vanishing index", sum(r["character_endomorphisms"]), 0)


def test_other_gut_groups():
    print("\n[4c] SO(10) and E_6")
    from pyCICY import equivariant as E
    from pyCICY import bundles as BU
    from pyCICY import CICY

    check("rank 3 commutant", B.gut_group(3), "E_6")
    check("rank 4 commutant", B.gut_group(4), "SO(10)")
    check("rank 5 commutant", B.gut_group(5), "SU(5)")
    check_raises("other ranks are refused, not labelled", ValueError,
                 B.gut_group, 6)

    X = CICY([[1, 2], [1, 2], [1, 2], [1, 2]])
    A = E.TETRAQUADRIC_Z2()

    # -- the two identities, which are free consistency checks -------------
    rng = np.random.default_rng(0)
    bad4 = bad3 = 0
    for _ in range(200):
        f = rng.integers(-3, 4, size=(3, 4))
        V4 = BU.LineBundleSum(X, np.vstack([f, -f.sum(axis=0)]))
        if abs(V4.wedge2().index()) > 1e-9:
            bad4 += 1
        g = rng.integers(-3, 4, size=(2, 4))
        V3 = BU.LineBundleSum(X, np.vstack([g, -g.sum(axis=0)]))
        if abs(V3.wedge2().index() + V3.index()) > 1e-9:
            bad3 += 1
    check("rank 4: ind(Lambda^2 V) = 0, the 6 of SU(4) being self-dual",
          bad4, 0)
    check("rank 3: ind(Lambda^2 V) = -ind(V), Lambda^2 V being V*", bad3, 0)

    # -- SO(10) ------------------------------------------------------------
    M4 = [[-2, -2, -2, -1], [-2, -1, -1, -1], [2, 1, 2, 0], [2, 2, 1, 2]]
    r4 = B.chiral_spectrum(A, M4)
    check("rank 4 model gives SO(10)", r4["gut_group"], "SO(10)")
    check("three 16s", r4["spectrum"][("SO(10)", "16", "V")], 3)
    check("the 10 is vector-like, so no index counts it",
          r4["spectrum"][("SO(10)", "10", "wedge2")], 0)
    check("and its character vanishes identically",
          sum(r4["character_wedge2"]), 0)
    check("only the 16 is chiral", r4["chiral_only"], ["16"])

    # A Wilson line in the SU(5) hypercharge direction is the wrong group
    # theory here, so it is refused rather than applied to the wrong thing.
    check_raises("a Wilson line is refused for rank 4", ValueError,
                 B.chiral_spectrum, A, M4, wilson=(0, 1))

    # -- E_6 ---------------------------------------------------------------
    M3 = [[-2, -2, -2, -1], [-1, 0, 1, 0], [3, 2, 1, 1]]
    V3 = BU.LineBundleSum(X, M3)
    check("the rank 3 model has ind(V) = -6", int(V3.index()), -6)
    r3 = B.chiral_spectrum(A, M3)
    check("rank 3 model gives E_6", r3["gut_group"], "E_6")
    check("three 27s", r3["spectrum"][("E_6", "27", "V")], 3)
    check_true("char(Lambda^2 V) = -char(V) at character level, not just total",
               [-x for x in r3["character_V"]] == r3["character_wedge2"])
    check_raises("a Wilson line is refused for rank 3", ValueError,
                 B.chiral_spectrum, A, M3, wilson=(0, 1))

    # -- rank 5 is unchanged ----------------------------------------------
    M5 = [[-2, -2, -1, 2], [-2, 1, 0, 0], [1, -2, 1, 0],
          [1, 1, -1, 0], [2, 2, 1, -2]]
    r5 = B.chiral_spectrum(A, M5, wilson=(0, 1))
    check("rank 5 still gives SU(5)", r5["gut_group"], "SU(5)")
    check("and still three generations", r5["generations"], 3)

    # All three ranks agree on the generation count, which is -ind(V)/|Gamma|
    # and has nothing to do with the GUT group.
    check_true("all three ranks give three generations",
               r3["generations"] == r4["generations"] == r5["generations"] == 3)


def test_boundary():
    print("\n[5] the input really drives the answer")
    base = {"10": [0] * 12 + [1] * 12, "10bar": [0] * 9 + [1] * 9,
            "5bar": [1] * 3, "5": [1] * 3}
    a = B.project(base, 2, wilson=(0, 1), index=-6)

    # A different equivariant structure -- same topology, same index, same
    # Wilson line -- must give a different spectrum. If it did not, the
    # Gamma-charges would be decorative and the module would be pretending to
    # compute something it is not.
    other = dict(base, **{"10": [0] * 24, "10bar": [0] * 18})
    b = B.project(other, 2, wilson=(0, 1), index=-6)
    check_true("different Gamma-charges give a different spectrum",
               a["spectrum"] != b["spectrum"])
    check("and the second is not three generations", b["generations"], 6)
    check_true("so it is correctly reported inconsistent",
               b["consistent"] is False)

    # The Wilson line also matters: no Wilson line means no splitting.
    c = B.project(base, 2, wilson=None, index=-6)
    check_true("dropping the Wilson line changes the spectrum",
               c["spectrum"] != a["spectrum"])


def main():
    t0 = time.time()
    test_branching()
    test_anomaly()
    test_wilson()
    test_projection()
    test_chiral_spectrum()
    test_other_gut_groups()
    test_boundary()

    print("\n" + "=" * 72)
    if FAILURES:
        print("FAILED (%d): %s" % (len(FAILURES), ", ".join(FAILURES)))
        return 1
    print("test_breaking: all checks passed in %.1fs" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
