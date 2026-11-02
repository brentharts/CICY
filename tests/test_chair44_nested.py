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
  [6b] tubes         the undetermined set is four diagonal tubes (M <= 7)
  [6c] interface     cell-level contact counts F, B, C of the level-n patch (n <= 6)
                     against the coincidence count, which has no area term
  [6d] approximants  tight-binding tori: the lift differs from the next torus on
                     140*2^M - 232 sites; rank, counting functions, six choices,
                     Bragg intensities
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


def test_scan_counts():
    print("\n[5b] the size-matched scan summary")
    N = 4
    ts, sc = C.translate_scan(N), C.translate_counts(N)
    check("same patch size", sc["cells"], ts["cells"])
    check_true("the FFT is exact in integers", sc["fft_rounding_error"] < 1e-6)
    for thr in (0.5, 0.9, 0.99):
        check("agreement >= %.2f counts agree with translate_scan" % thr,
              sc["counts"][thr], ts["by_threshold"][thr]["count"])
    check_true("best agreement at overlap >= 1/8 matches translate_scan",
               abs(sc["best"][8]["fraction"] - ts["best"][0]["fraction"]) < 1e-12)
    fr = [sc["best"][f]["fraction"] for f in (2, 4, 8, 16)]
    check_true("the best agreement does not decrease as the overlap rule loosens",
               all(a <= b + 1e-12 for a, b in zip(fr, fr[1:])))
    check_true("chance level is small and positive", 0 < sc["chance"] < 0.05)
    check_true("a count of translates at a looser rule is no smaller",
               sc["best"][16]["translates"] >= sc["best"][8]["translates"]
               >= sc["best"][2]["translates"])


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


def test_tubes():
    print("\n[6b] the undetermined set is four diagonal tubes")
    tb = C.undetermined_is_diagonal_tubes(7)
    check("levels checked", [r["M"] for r in tb], list(range(1, 8)))
    check_true("union of the four tubes = undetermined set, M <= 7", all(r["equal"] for r in tb))
    check("cells per tube at M = 5", tb[4]["cells"], [7 * 32 - 6] * 4)
    check("overlap (cells counted four times, less once: 8 * 3)", {r["overlap"] for r in tb}, {24})
    check_true("c(M) = 4(7*2^M - 6) - 24 = 28*2^M - 48", all(r["c_formula"] for r in tb))
    check_true("E(M) = 140*2^M + 48; 5 per six-set, 23 per 24-set from M = 2",
               all(r["excess_formula"] for r in tb))
    check("at M = 1 the 8 undetermined addresses are 42-label sets: E", tb[0]["excess"], 8 * 41)


def test_interface():
    print("\n[6c] interface of the level-n patch against the coincidence count")
    ic = C.interface_counts(6)
    check("contact pairs C, n = 1..6", [r["C"] for r in ic], [16, 183, 1663, 14079, 115711, 937983])
    check("cell faces between chairs F, n = 1..4", [r["F"] for r in ic[:4]], [48, 576, 5376, 46080])
    check("boundary faces B, n = 1..4", [r["B"] for r in ic[:4]], [96, 384, 1536, 6144])
    check_true("F = 12(8^n - 4^n) and B = 24*4^n for n <= 6", all(r["F_formula"] and r["B_formula"] for r in ic))
    check_true("C = (29/8) 8^n - 3*4^n - 1 for n <= 6", all(r["C_formula"] for r in ic))
    check_true("no overlap, and the patch is 7*8^n cells", all(r["overlap"] == 0 and r["filled"] for r in ic))
    check_true("8 C_n - C_(n+1) = -(12*4^n + 7), n <= 5", all(r["defect_formula"] for r in ic[:-1]))
    de = C.defect_exponents(8)
    check("transfer-matrix roots (root: multiplicity)", de["roots"], {0: 5, 1: 32, 2: 4})
    check_true("no root 4, so no area term in c(M)", de["no_area_root"])
    r8 = de["rows"][-1]
    check("line fraction / surface fraction = 7 c(M) / (24 * 4^M) at M = 8", r8["ratio"],
          Fraction(7 * (28 * 2 ** 8 - 48), 24 * 4 ** 8))
    check_true("... which times 2^M is within 1% of 49/6", abs(float(r8["ratio"]) * 2 ** 8 / (49 / 6) - 1) < 0.01)
    # control: the interface count does have an area term (B = 24*4^n), c(M) does not
    check_true("control: B / 4^n is constant but c(M) / 4^M falls",
               len({r["B"] // 4 ** r["n"] for r in ic}) == 1
               and C.c_of(8) / 4 ** 8 < C.c_of(4) / 4 ** 4 / 10)


def test_approximants():
    print("\n[6d] periodic approximants on the nested tori")
    check("the six chair-stack labels", C.chair_stack_labels(), [0, 49, 77, 84, 112, 161])
    # the label field is the real tiling: sigma^n of the root cells reproduces the level-n patch labels
    labels = C._sub()[0]
    rots = C.rotations()
    G, t = C.level_poses(3)
    idf = rots.index(C.IDENTITY)
    ok = True
    neg = (G.sum(axis=2) < 0)
    Cc = C.np.array(C.CELLS)
    cells = C.np.einsum('nij,kj->nki', G, Cc) + t[:, None, :] - neg[:, None, :].astype(int)
    fidx = {C.frame_matrix(f).tobytes(): i for i, f in enumerate(rots)}
    pos = {}
    for i in range(len(G)):
        f = fidx[G[i].tobytes()]
        for k in range(7):
            pos[tuple(int(x) for x in cells[i, k])] = 7 * f + k
    for k, c in enumerate(C.CELLS):
        g = C.label_field(3, 7 * idf + k)
        for x in C.itertools.product(range(8), repeat=3):
            ok = ok and pos[tuple(8 * c[i] + x[i] for i in range(3))] == g[x]
    check_true("sigma^3 of the root cells reproduces the level-3 patch labels", ok)
    owner = {tuple(int(x) for x in cells[i, k]): i for i in range(len(G)) for k in range(7)}
    T = C._same_chair_tables()
    bad = 0
    for p, i in owner.items():
        for ax in range(3):
            q = list(p)
            q[ax] += 1
            q = tuple(q)
            if q in owner and bool(T[ax][pos[p], pos[q]]) != (owner[q] == i):
                bad += 1
    check("same-chair test against the chair partition of the level-3 patch: mismatches", bad, 0)
    ap = C.approximant_defect(7, 5, 3)
    check("|S|, M = 1..7", [r["S"] for r in ap], [140 * 2 ** M - 232 for M in range(1, 8)])
    check_true("|S| <= 8 c(M) for M <= 7", all(r["S_within_bound"] for r in ap))
    check("rank D, M = 1..5", [r["rank"] for r in ap if "rank" in r], [12 * 2 ** M for M in range(1, 6)])
    check("largest counting-function difference, M = 1..3", [r["max_dN"] for r in ap if "max_dN" in r], [4, 4, 9])
    check_true("... within 2|S|", all(r["dN_within_bound"] for r in ap if "max_dN" in r))
    check_true("the lift's spectrum is the union over the eight half-period twists",
               all(r["lift_is_twisted_union"] for r in ap if "max_dN" in r))
    sx = C.six_choices(6)
    check("tube size at M = 6 (7*2^M - 6)", sx[-1]["tube"], 442)
    check_true("six labels differ pairwise exactly on the (1,1,1) tube, distinct there",
               all(r["pair_is_tube"] and r["distinct_on_tube"] for r in sx))
    df = C.diffraction(7)
    check("intensity at k = 0 is the density squared, 1/49 (M = 7, to 5 places)", round(df[-1]["grid"][0], 5), round(1 / 49, 5))
    check("total intensity at M = 7, to 5 places (density 1/7)", round(df[-1]["total"], 5), round(1 / 7, 5))
    check_true("amplitude change <= |S|/8^(M+1) at every level", all(r["amp_within_bound"] for r in df if "amp" in r))
    gaps = [1 / 7 - df[-1]["grid"][m] for m in range(2, 6)]
    check_true("the weight beyond the 2^M0 grid falls by a factor of 6 or more per level (M0 = 2..5)",
               all(a / b > 6 for a, b in zip(gaps, gaps[1:])))
    # controls: the factor 8 (one c(M) per block) is needed, and the untwisted spectrum alone is not the lift's
    check_true("control: |S| exceeds c(M), so the bound needs the eight blocks (M >= 3)",
               all(r["S"] > C.c_of(r["M"]) for r in ap if r["M"] >= 3))
    g3 = C.label_field(3, C.chair_stack_labels()[0])
    e0 = C.np.linalg.eigvalsh(C.approximant_hamiltonian(g3).toarray())
    e1 = C.np.linalg.eigvalsh(C.approximant_hamiltonian(g3, twist=(1, 1, 1)).toarray())
    check_true("control: the periodic and the (1,1,1)-twisted spectra differ", not C.np.allclose(e0, e1))


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
              test_scan_counts, test_undetermined, test_tubes, test_interface, test_approximants,
              test_not_a_link, test_controls, test_ledger):
        t()
    print()
    if FAILURES:
        print("FAILED (%d): %s" % (len(FAILURES), "; ".join(FAILURES)))
        sys.exit(1)
    print("all passed")
