"""
Tests for pyCICY.chair44_nested (Part VIII: Chair44 nearly descends to nested
tori, and the Hat does not).

  [1] substitution   the eight child poses tile every rotated doubled chair;
                     168 labels, column sums 8, primitive at power 3
  [2] coincidence    depth 3 with 336 of 512 addresses; 209 label sets, 41 of
                     them non-singleton; the counts c(M) and E(M)
  [3] closed forms   c(M) = 28*2^M - 48, E(M) = 140*2^M + 48 for all M >= 1,
                     by the finite recurrence certificate
  [4] descent        agreement of a level-6 patch (1.8 million cells) with its
                     2^M translates against the proved bound 1 - c(M)/8^M
  [5] translates     every translate at once (FFT): where the near-periods are
  [6] undetermined   six or 24 rotations at undetermined addresses; the six
                     are a dihedral coset about a body diagonal
  [7] not a link     Kuhn's triangulation of Z^3 against the cube group; the
                     S4 cover of the (3, 4, inf) orbifold
  [8] controls       a perturbed child pose is rejected; a tighter bound than
                     the proved one is violated; a wrong closed form fails;
                     too few terms prove nothing; odd shifts agree nowhere;
                     a matrix outside the Kuhn group is rejected
  [9] ledger         every quantity carries a status

Run with:  python3 tests/test_chair44_nested.py   (about 15 s)
       or: python3 run_tests.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fractions import Fraction

from pyCICY import chair44_nested as C

FAILURES = []


def check(name, got, want):
    ok = got == want
    print("  {:<66} {:>12} {}".format(name, str(got)[:12], "ok" if ok else "FAIL want " + str(want)))
    if not ok:
        FAILURES.append(name)


def check_true(name, cond):
    print("  {:<66} {:>12} {}".format(name, str(bool(cond)), "ok" if cond else "FAIL"))
    if not cond:
        FAILURES.append(name)


_STATUS = None
IDENT_ALT = ((0, 1, 2), (1, 1, 1))


def status():
    global _STATUS
    if _STATUS is None:
        _STATUS = C.status()
    return _STATUS


def test_substitution():
    print("\n[1] the substitution")
    labels, sigma, _ = C._sub()
    check_true("the eight children tile every rotated doubled chair", C.verify_children())
    check("rotations", len(C.rotations()), 24)
    check("labels (24 rotations x 7 cells)", len(labels), 168)
    check("column sums of sigma", C.column_sums(sigma), [8])
    check("primitive exponent", C.primitive_exponent(sigma), 3)
    check("cells of the chair", len(C.CELLS), 7)


def test_coincidence():
    print("\n[2] coincidence")
    co = C.coincidence_table(8)
    first = next(r["M"] for r in co if r["coincident"] > 0)
    check("least coincidence depth", first, 3)
    check("coincident addresses at depth 3", co[3]["coincident"], 336)
    check("non-coincident words c(M), M = 0..8", [r["non_coincident"] for r in co],
          [1, 8, 64, 176, 400, 848, 1744, 3536, 7120])
    check("excess E(M), M = 0..8", [r["excess"] for r in co],
          [167, 328, 608, 1168, 2288, 4528, 9008, 17968, 35888])
    order, _ = C.subset_automaton()
    check("reachable label sets", len(order), 209)
    check("non-singleton label sets", sum(1 for A in order if len(A) > 1), 41)
    check("largest label set is the whole alphabet", max(len(A) for A in order), 168)


def test_closed_forms():
    print("\n[3] closed forms")
    cf = C.closed_forms_proved()
    check("non-singleton states", cf["non_singleton_states"], 41)
    check("terms needed / checked", (cf["terms_needed"], cf["terms_checked"]), (43, 44))
    check_true("c(M) = 28*2^M - 48 from M = 1", cf["c_closed_form"])
    check_true("E(M) = 140*2^M + 48 from M = 1", cf["E_closed_form"])
    check_true("proved", cf["proved"])
    factors, T, text = C.transfer_charpoly()
    check("characteristic polynomial of the transfer matrix", factors, {0: 5, 2: 4, 1: 32})
    check("... as typeset", text, "x^{5}(x-1)^{32}(x-2)^{4}")
    check("transfer matrix size", len(T), 41)
    co = C.coincidence_table(6)
    check_true("the formula fails at M = 0 (c(0) = 1, formula gives -20)",
              co[0]["non_coincident"] != 28 - 48)


def test_descent():
    print("\n[4] descent to the nested tori (level-6 patch)")
    na = C.nested_agreement(6)
    check("cells in the patch", na["cells"], 7 * 8 ** 6)
    for r in na["rows"]:
        if r["v"][1:] == (0, 0):
            check_true("M=%d along x: %.5f >= bound %.5f (proved)" % (r["M"], r["fraction"], float(r["bound"])),
                       r["holds"])
    check_true("the inequality holds on all 15 rows (3 axes, M = 1..5)",
              len(na["rows"]) == 15 and all(r["holds"] for r in na["rows"]))
    x = {r["M"]: r for r in na["rows"] if r["v"][1:] == (0, 0)}
    check("M=1 agreement along x", x[1]["agree"], 0)
    check_true("M=3 measured 0.6835", abs(x[3]["fraction"] - 0.68352) < 1e-4)
    check_true("M=4 measured 0.9102", abs(x[4]["fraction"] - 0.91023) < 1e-4)
    check_true("M=5 measured 0.9761", abs(x[5]["fraction"] - 0.97612) < 1e-4)
    check_true("disagreement at M falls by about a factor 4 per step",
              all(3.0 < (1 - x[M]["fraction"]) / (1 - x[M + 1]["fraction"]) < 5.0 for M in (3, 4)))


def test_translates():
    print("\n[5] every translate (level-5 patch, FFT)")
    ts = status()["scan"]
    check("cells", ts["cells"], 7 * 8 ** 5)
    check_true("rounding error below 1e-3", ts["fft_rounding_error"] < 1e-3)
    th = ts["by_threshold"]
    check("translates with agreement >= 0.99", th[0.99]["count"], 6)
    check("... their valuations", th[0.99]["by_valuation"], {5: 6})
    check("translates with agreement >= 0.9", th[0.9]["count"], 342)
    check("... their valuations", th[0.9]["by_valuation"], {3: 166, 4: 152, 5: 24})
    check_true("none with valuation below 3", min(th[0.9]["by_valuation"]) >= 3)
    check_true("agreement >= 0.5 reaches valuation 2 but not 1",
              min(th[0.5]["by_valuation"]) == 2)
    check_true("best translate is a corner of the patch lattice",
              all(abs(c) == 32 for c in ts["best"][0]["v"]))


def test_undetermined():
    print("\n[6] the undetermined addresses")
    prof = C.undetermined_profile(12)
    check("label-set sizes at M = 12", sorted(prof), [6, 24])
    check("addresses with six rotations", prof[6], 114624)
    check("addresses with all 24", prof[24], 16)
    check("... the 24-set count is the same at M = 8", C.undetermined_profile(8)[24], 16)
    d = C.six_is_a_dihedral_coset()
    check("six-label sets", d["n_sets"], 28)
    check_true("each is a coset of a dihedral subgroup about a body diagonal", d["all_dihedral_cosets"])


def test_not_a_link():
    print("\n[7] two things that are not connections")
    kv = C.kuhn_vs_cube()
    check("simplices per domain", kv["simplices"], [1, 7, 12, 6])
    check("Euler characteristic", kv["euler"], 0)
    check("round a vertex (4!)", kv["around_vertex"], 24)
    check("cube group preserving the complex", (kv["cube_preserving"], kv["cube_group"]), (12, 48))
    check("rotations preserving it", kv["rotations_preserving"], 6)
    check("automorphism group, and its meet with the cube group", (kv["automorphism_group"], kv["intersection"]), (48, 12))
    check("mod 2, cube", kv["mod2_cube"], {"fixed_nonzero": 1, "orbit_sizes": [1, 3, 3]})
    check("mod 2, Kuhn", kv["mod2_kuhn"], {"fixed_nonzero": 0, "orbit_sizes": [3, 4]})
    oc = C.orbifold_cover()
    check("orbifold chi of (3, 4, inf)", oc["orbifold_chi"], Fraction(-5, 12))
    check("S4 cover: chi, cusps, genus", (oc["cover_chi"], oc["cusps"], oc["genus"]), (-10, 12, 0))


def test_controls():
    print("\n[8] controls")
    # a perturbed child pose must be rejected (translation off by one)
    good = C.CHILDREN
    try:
        C.CHILDREN = good[:7] + ((good[7][0], (good[7][1][0] + 1,) + good[7][1][1:]),)
        check_true("a child moved by one cell no longer tiles", not C.verify_children())
        C.CHILDREN = good[:7] + ((IDENT_ALT, good[7][1]),)
        check_true("a child given a different rotation no longer tiles", not C.verify_children())
        # the shape check cannot see handedness: swapping x and z is a mirror symmetry of the
        # undecorated chair, so the mirrored child fills the same seven cells.  That blindness is
        # the reason the decorated tile is needed, and why chirality is a separate statement.
        C.CHILDREN = good[:7] + ((((0, 1, 2), (-1, 1, 1)), good[7][1]),)
        check_true("a mirror-image child still tiles (the undecorated chair is mirror symmetric)",
                   C.verify_children())
    finally:
        C.CHILDREN = good
    check_true("restored", C.verify_children())
    # a tighter bound than the proved one is false
    na = C.nested_agreement(5, axes=((1, 0, 0),))
    row = [r for r in na["rows"] if r["M"] == 3][0]
    tight = Fraction(8 ** 3 - C.c_of(3) // 2, 8 ** 3)
    check_true("a bound with c(M)/2 would be violated at M = 3", row["fraction"] < float(tight))
    check_true("... and the proved bound is not vacuous: agreement < 1", row["fraction"] < 1)
    # a wrong closed form fails
    co = C.coincidence_table(8)
    check_true("c(M) = 28*2^M - 47 is false", not all(r["non_coincident"] == 28 * 2 ** r["M"] - 47 for r in co[1:]))
    check_true("E(M) = 140*2^M + 47 is false", not all(r["excess"] == 140 * 2 ** r["M"] + 47 for r in co[1:]))
    # too few terms prove nothing
    check_true("10 terms do not prove the closed form", not C.closed_forms_proved(terms=10)["proved"])
    # odd shifts agree nowhere on whole cells
    G = C.grid(5)
    for v in ((1, 0, 0), (3, 0, 0), (5, 0, 0), (0, 1, 0), (0, 0, 7)):
        a, o = C.agreement(G, v)
        check("agreement at the odd shift %s" % (v,), a, 0)
    # the identity shift agrees everywhere
    a, o = C.agreement(G, (0, 0, 0))
    check_true("the zero shift agrees everywhere", a == o and o == 7 * 8 ** 5)
    # a matrix outside the Kuhn group
    kv = C.kuhn_vs_cube()
    check_true("the cube's 4-fold rotation is not in the Kuhn group (12 < 48)", kv["cube_preserving"] < kv["cube_group"])


def test_ledger():
    print("\n[9] ledger")
    led = C.ledger()
    check_true("every row has a status", all(len(r) == 2 and r[1] for r in led))
    words = ("exact", "proved", "quoted", "declined", "coincidence", "measured")
    check_true("every status uses the ledger's vocabulary",
              all(any(w in r[1] for w in words) for r in led))
    check_true("declined rows present (hull, fourfold)", sum("declined" in r[1] for r in led) >= 2)
    check_true("status() passes", status()["all_pass"])


if __name__ == "__main__":
    for t in (test_substitution, test_coincidence, test_closed_forms, test_descent, test_translates,
              test_undetermined, test_not_a_link, test_controls, test_ledger):
        t()
    print()
    if FAILURES:
        print("FAILED (%d): %s" % (len(FAILURES), "; ".join(FAILURES)))
        sys.exit(1)
    print("all passed")
