"""
Tests for pyCICY.theories.ftheory.

F-theory is worth testing differently from the heterotic modules, because in
six dimensions the physics is overdetermined: anomaly cancellation, the
geometry of the base, and the Hodge numbers of the elliptic threefold are
three routes to the same numbers, and each can be checked against the other
two. Most of what follows is that cross-checking rather than comparison with a
stored table.

  [1] kodaira       the fibre classification, including that it refuses
                    vanishing orders that Delta = 4f^3 + 27g^2 cannot produce
  [2] derived       the six matter-free algebras and their curves, derived
                    from the anomaly conditions, against the tabulated
                    non-Higgsable clusters they must reproduce
  [3] matter        multiplicities solved from anomalies, against the standard
                    results: 2N fundamentals on a -2 curve, N+8 and one
                    antisymmetric on a -1 curve, half a 56 for e7 on -7
  [4] bases         K^2 = 10 - h^{1,1}, genus by adjunction, and the moduli
                    count from Riemann-Roch against 272 - 29T from the
                    gravitational anomaly -- two unrelated computations that
                    must agree
  [5] spectra       whole models: P^2 giving (2, 272), F_12 giving (11, 491),
                    every anomaly residual zero
  [6] duality       F_n against E8 x E8 on K3 with (12+n, 12-n) instantons
  [7] absent        that six-dimensional Yukawa couplings raise NoSuchTheory
                    and not NeedsMetric, because the reason is different
  [8] fibrations    obvious fibrations of a configuration matrix, against
                    hand-checkable cases and against Hodge numbers computed
                    by the rest of the package
  [9] fourfolds     the D3 tadpole, exact, and the spectrum refusing

Run with:  python3 tests/test_ftheory.py
       or: python3 run_tests.py
"""

import os
import sys
import time
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyCICY import CICY
from pyCICY import theories as T
from pyCICY.theories import ftheory as FT

FAILURES = []


def check(name, got, want):
    ok = got == want
    print("  {:<58} {:>14} {}".format(name, str(got)[:14],
                                      "ok" if ok else "FAIL want " + str(want)))
    if not ok:
        FAILURES.append(name)


def check_true(name, cond):
    print("  {:<58} {:>14} {}".format(name, str(bool(cond)),
                                      "ok" if cond else "FAIL"))
    if not cond:
        FAILURES.append(name)


def _raises(exc, fn, *a, **kw):
    try:
        fn(*a, **kw)
    except exc:
        return True
    except Exception:                                            # noqa: BLE001
        return False
    return False


# ---------------------------------------------------------------------------


def test_kodaira():
    print("\n[1] Kodaira's classification of singular fibres")

    # The table, read off the vanishing orders of (f, g, Delta).
    cases = [((0, 0, 0), "I_0", None),
             ((0, 0, 1), "I_1", None),
             ((0, 0, 5), "I_5", "su(5)"),
             ((1, 1, 2), "II", None),
             ((1, 2, 3), "III", "su(2)"),
             ((2, 2, 4), "IV", "su(3)"),
             ((2, 3, 6), "I_0*", "so(8)"),
             ((2, 3, 8), "I_2*", "so(12)"),
             ((3, 4, 8), "IV*", "e6"),
             ((3, 5, 9), "III*", "e7"),
             ((4, 5, 10), "II*", "e8")]
    for orders, name, alg in cases:
        r = FT.kodaira_type(*orders)
        check("f,g,D = %-10s -> %s" % (str(orders), name), r["type"], name)
        if alg:
            check("   carries", r["algebra_split"], alg)

    # The rank of the split algebra is what h^{1,1} of the resolved threefold
    # counts, so it is worth having right.
    check("rank of so(12) from I_2*", FT.kodaira_type(2, 3, 8)["rank_split"], 6)
    check("rank of e8 from II*", FT.kodaira_type(4, 5, 10)["rank_split"], 8)

    # Non-minimal: not a fibre type, a signal to blow up the base.
    nm = FT.kodaira_type(4, 6, 12)
    check("(4,6,12) is non-minimal", nm["type"], "non-minimal")
    check_true("and is flagged as such", not nm["minimal"])

    # The consistency check is not decoration. Delta = 4f^3 + 27g^2 vanishes to
    # order min(3 ord f, 2 ord g), with equality unless the leading terms
    # cancel -- which they cannot when the two orders differ. So these are
    # arithmetically impossible and must not be silently classified.
    check_true("ord(Delta) below the bound is refused",
               _raises(ValueError, FT.kodaira_type, 1, 1, 1))
    check_true("and above it when no cancellation is possible",
               _raises(ValueError, FT.kodaira_type, 0, 1, 4))
    check_true("negative orders refused",
               _raises(ValueError, FT.kodaira_type, -1, 0, 0))

    # The stronger statement, and the one that says the two halves of the
    # function agree: no entry of Kodaira's table is rejected by the
    # arithmetic. Every (a, b, c) that names a fibre type either has
    # 3a = 2b, so the leading terms of Delta can cancel and c may exceed the
    # bound, or has c exactly min(3a, 2b). Scan and confirm.
    classified, mismatches = 0, []
    for a in range(6):
        for b in range(8):
            for c in range(16):
                try:
                    r = FT.kodaira_type(a, b, c, check=False)
                except ValueError:
                    continue
                if not r["minimal"]:
                    continue
                classified += 1
                try:
                    FT.kodaira_type(a, b, c)
                except ValueError:
                    mismatches.append((a, b, c))
    check_true("the scan found the table", classified > 30)
    check("no Kodaira type is rejected by the Delta arithmetic",
          mismatches, [])


def test_matter_free_derivation():
    print("\n[2] the matter-free algebras, derived")

    # Set every matter multiplicity to zero and the two anomaly conditions
    # become two independent formulas for D.D. They agree for six algebras and
    # at integer values for those same six -- which is exactly the tabulated
    # list of matter-free non-Higgsable clusters. Nothing about the table is
    # used in deriving this.
    derived = FT.matter_free_algebras()
    check("six algebras need no charged matter", len(derived), 6)
    for alg, n in [("su(3)", -3), ("so(8)", -4), ("f4", -5), ("e6", -6),
                   ("e7", -8), ("e8", -12)]:
        check("%-6s sits on a %d curve" % (alg, n), derived.get(alg), n)

    # And they are the tabulated ones, at the tabulated self-intersections.
    for n, alg in [(3, "su(3)"), (4, "so(8)"), (5, "f4"), (6, "e6"),
                   (8, "e7"), (12, "e8")]:
        check("table: -%d carries %s" % (n, alg), FT.NON_HIGGSABLE[n], alg)
        check_true("   and the derivation puts it there",
                   derived.get(alg) == -n)
        check("   with no charged matter", FT.non_higgsable(n)["matter"], {})

    # The one non-Higgsable cluster with matter, and it is half a
    # hypermultiplet: the 56 of e7 is pseudo-real, so half of one is a
    # consistent representation content and the anomaly conditions pick it.
    seven = FT.non_higgsable(7)
    check("-7 carries e7", seven["algebra"], "e7")
    check("with half a 56", seven["matter"], {"56": Fraction(1, 2)})

    # -1 and -2 force nothing.
    check("-1 forces no algebra", FT.non_higgsable(1)["algebra"], None)
    check("-2 forces no algebra", FT.non_higgsable(2)["algebra"], None)

    # -9, -10, -11 give e8 but on a base that has to be blown up first.
    for n in (9, 10, 11):
        r = FT.non_higgsable(n)
        check("-%d gives e8 after %d blowups" % (n, 12 - n),
              (r["algebra"], r["blowups"]), ("e8", 12 - n))
    check("-12 needs no blowup", FT.non_higgsable(12)["blowups"], 0)

    # Below -12 no elliptic Calabi-Yau exists at all.
    check_true("a -13 curve is refused", _raises(ValueError,
                                                 FT.non_higgsable, 13))

    # g2 and su(2) fail the integrality of the same two formulas, so they
    # cannot appear without matter. The failure is the physics, not an error.
    check_true("g2 is not matter-free", "g2" not in derived)
    check_true("su(2) is not matter-free", "su(2)" not in derived)


def test_matter():
    print("\n[3] matter content from the anomaly conditions")

    # The standard results for su(N), all three from the same linear system.
    for N in (4, 5, 6, 8):
        m = FT.matter_content("su(%d)" % N, -2)["matter"]
        check("su(%d) on a -2 curve: 2N fundamentals" % N,
              m, {"fund": Fraction(2 * N)})
    for N in (4, 5, 6, 8):
        m = FT.matter_content("su(%d)" % N, -1)["matter"]
        check("su(%d) on a -1 curve: N+8 fund, 1 antisym" % N,
              (m["fund"], m["antisym"]), (Fraction(N + 8), Fraction(1)))
    for N in (4, 6, 9):
        m = FT.matter_content("su(%d)" % N, 0)["matter"]
        check("su(%d) on a 0 curve: 16 fund, 2 antisym" % N,
              (m["fund"], m["antisym"]), (Fraction(16), Fraction(2)))

    # su(2) and su(3) have no independent quartic Casimir, so they run through
    # a different branch of the group theory data and are worth checking apart.
    check("su(2) on a -1 curve: 10 doublets",
          FT.matter_content("su(2)", -1)["matter"], {"fund": Fraction(10)})
    check("su(3) on a 0 curve: 18 fundamentals",
          FT.matter_content("su(3)", 0)["matter"], {"fund": Fraction(18)})

    # so(N) with vectors only is possible on a -4 curve and nowhere else: the
    # quartic condition fixes D.D = -4 independently of the multiplicity.
    check("so(10) on a -4 curve: N-8 vectors",
          FT.matter_content("so(10)", -4)["matter"], {"vector": Fraction(2)})
    check("so(14) on a -4 curve: N-8 vectors",
          FT.matter_content("so(14)", -4)["matter"], {"vector": Fraction(6)})
    check_true("so(10) on a -3 curve has no vector-only solution",
               _raises(ValueError, FT.matter_content, "so(10)", -3))

    # Genus enters as adjoint hypermultiplets, one per unit of genus.
    m = FT.matter_content("su(5)", 0, genus=1)["matter"]
    check("su(5) on a genus-1 0-curve: one adjoint", m, {"adj": Fraction(1)})
    m = FT.matter_content("su(5)", 4, genus=1)["matter"]
    check("su(5) on a genus-1 4-curve keeps the adjoint",
          m.get("adj"), Fraction(1))

    # A negative multiplicity is not a spectrum, and saying so is the point.
    check_true("a negative multiplicity is refused",
               _raises(ValueError, FT.matter_content, "su(12)", -3))

    # e8 has no matter representation below the adjoint, so the system has no
    # unknowns and becomes a pure consistency check: only a -12 curve works.
    check("e8 has no unknowns and fits a -12 curve",
          FT.matter_content("e8", -12)["matter"], {})
    check_true("and fails on -11", _raises(ValueError,
                                           FT.matter_content, "e8", -11))

    # Feeding a spectrum back in must give zero residuals in every condition.
    r = FT.check_anomalies("su(5)", -1, 0,
                           {"fund": 13, "antisym": 1})
    check_true("residuals vanish for su(5) on -1", r["ok"])
    check("all three conditions were used", len(r["residuals"]), 3)
    bad = FT.check_anomalies("su(5)", -1, 0, {"fund": 12, "antisym": 1})
    check_true("and do not vanish for a wrong multiplicity", not bad["ok"])

    # The charged hypermultiplet count is what the gravitational anomaly sees.
    check("e7 with half a 56 is 28 charged hypers",
          FT.matter_content("e7", -7)["charged_dim"], Fraction(28))


def test_bases():
    print("\n[4] bases, and two routes to the same moduli count")

    for b in ([FT.Base.P2()] + [FT.Base.hirzebruch(n) for n in range(6)]
              + [FT.Base.del_pezzo(k) for k in range(9)]):
        check_true("%-6s has K^2 = 10 - h^{1,1}" % b.name, b.consistent())

    check("P^2: K^2", FT.Base.P2().K2, 9)
    check("P^2: T", FT.Base.P2().T, 0)
    check("F_5: K^2 is 8 for every Hirzebruch",
          FT.Base.hirzebruch(5).K2, 8)
    check("dP_8: T", FT.Base.del_pezzo(8).T, 8)

    # The section of F_n has self-intersection -n; the fibre has zero. This is
    # where the non-Higgsable algebra lives.
    F7 = FT.Base.hirzebruch(7)
    check("F_7 section squares to -7", F7.dot([1, 0], [1, 0]), -7)
    check("F_7 fibre squares to 0", F7.dot([0, 1], [0, 1]), 0)
    check("section meets fibre once", F7.dot([1, 0], [0, 1]), 1)
    check("both are rational curves", (F7.genus([1, 0]), F7.genus([0, 1])),
          (0, 0))

    # Adjunction on P^2: a degree-d plane curve has genus (d-1)(d-2)/2.
    P2 = FT.Base.P2()
    for d in range(1, 7):
        check("degree %d plane curve has genus %d"
              % (d, (d - 1) * (d - 2) // 2),
              P2.genus([d]), (d - 1) * (d - 2) // 2)

    # The check this section exists for. The number of complex structure
    # moduli of the generic Weierstrass model is a Riemann-Roch count on the
    # base: sections of -4K and -6K, less the automorphisms and the rescaling.
    # The gravitational anomaly, a one-loop condition in six dimensions, says
    # the answer must be 272 - 29T. Nothing connects the two computations
    # except that both are true.
    for b in ([FT.Base.P2()] + [FT.Base.hirzebruch(n) for n in range(3)]
              + [FT.Base.del_pezzo(k) for k in range(9)]):
        check("%-6s Riemann-Roch moduli = 272 - 29T" % b.name,
              FT.weierstrass_moduli(b), 272 - 29 * b.T)

    # The F_2 case is the delicate one and worth naming. Its automorphism
    # group is one dimension larger than F_0's, which would undercount the
    # moduli by one; the compensation is that F_2 deforms to F_0, so
    # h^1(T_B) = 1. Using the Euler characteristic of T_B rather than h^0
    # alone is what makes this come out.
    check("chi(T_B) is 6 for every Hirzebruch surface",
          [FT.Base.hirzebruch(n).chi_tangent() for n in range(4)],
          [6, 6, 6, 6])
    check("and 8 for P^2", FT.Base.P2().chi_tangent(), 8)

    # h^0(-4K) on P^2 is h^0(O(12)) = 91, by hand.
    check("h^0(P^2, -4K) = h^0(O(12))", P2.h0_anticanonical(4), 91)
    check("h^0(P^2, -6K) = h^0(O(18))", P2.h0_anticanonical(6), 190)

    check_true("dP_9 is refused", _raises(ValueError, FT.Base.del_pezzo, 9))
    check_true("an asymmetric intersection form is refused",
               _raises(ValueError, FT.Base, [[0, 1], [2, 0]], [1, 1]))

    # A hand-built base carries no `kind`, so nothing downstream mistakes it
    # for a named surface. Give one the intersection form of F_4 and the name
    # of something else: the numbers follow the form, and the Hirzebruch
    # special cases stay switched off.
    fake = FT.Base([[-4, 1], [1, 0]], [-2, -6], name="F_4 by hand")
    check("built by hand, K^2 still 8", fake.K2, 8)
    check("and the section still squares to -4", fake.dot([1, 0], [1, 0]), -4)
    check_true("but kind is unset", fake.kind is None)
    check("so no gauge algebra is assumed",
          FT.FTheory6D(fake).gauge_group(), "trivial")
    check_true("and no heterotic dual is claimed",
               FT.FTheory6D(fake).heterotic_dual() is None)
    # Put so(8) on that section by hand and the answers match F_4 exactly.
    byhand = FT.FTheory6D(fake, gauge=[("so(8)", [1, 0])])
    check("naming the algebra reproduces F_4",
          byhand.hodge_numbers(), FT.FTheory6D("F4").hodge_numbers())


def test_spectra():
    print("\n[5] whole models")

    # The two anchors of the six-dimensional landscape.
    m = FT.FTheory6D("P2")
    s = m.spectrum()
    check("P^2: (T, V, H)", (s["T"], s["V"], s["H"]), (0, 0, 273))
    check("P^2: Hodge numbers of the elliptic threefold",
          m.hodge_numbers(), (2, 272))
    check("P^2: chi", m.euler_characteristic(), -540)

    # F_12 is the far end: the -12 section carries e8, which is the largest
    # non-Higgsable algebra there is, and the threefold is the well known
    # (11, 491).
    m = FT.FTheory6D("F12")
    s = m.spectrum()
    check("F_12: gauge algebra", m.gauge_group(), "e8")
    check("F_12: (T, V, H)", (s["T"], s["V"], s["H"]), (1, 248, 492))
    check("F_12: Hodge numbers", m.hodge_numbers(), (11, 491))
    check("F_12: chi", m.euler_characteristic(), -960)
    check("F_12: rank 8 gives 8 extra divisors",
          m.hodge_numbers()[0] - FT.Base.hirzebruch(12).h11 - 1, 8)

    # The Hirzebruch series, where the non-Higgsable algebra grows with n.
    expect = {0: ("trivial", (3, 243)), 1: ("trivial", (3, 243)),
              2: ("trivial", (3, 243)), 3: ("su(3)", (5, 251)),
              4: ("so(8)", (7, 271)), 5: ("f4", (7, 295)),
              6: ("e6", (9, 321)), 7: ("e7", (10, 348)),
              8: ("e7", (10, 376)), 12: ("e8", (11, 491))}
    for n, (alg, hodge) in sorted(expect.items()):
        m = FT.FTheory6D("F%d" % n)
        check("F_%-2d %-8s hodge" % (n, alg), (m.gauge_group(),
                                               m.hodge_numbers()),
              (alg, hodge))

    # F_7 is the only one of these with charged matter, and it is half a
    # hypermultiplet in the 56, so 28 charged states.
    s7 = FT.FTheory6D("F7").spectrum()
    check("F_7 has 28 charged hypermultiplets", s7["H_charged"], 28)
    check("F_8 has none", FT.FTheory6D("F8").spectrum()["H_charged"], 0)

    # Every anomaly, gravitational and gauge, in every model above.
    good = [n for n in range(13) if n not in (9, 10, 11)]
    for spec in ["P2", "dP0", "dP4", "dP8"] + ["F%d" % n for n in good]:
        m = FT.FTheory6D(spec)
        r = m.check_anomalies()
        check_true("%-4s cancels every anomaly" % spec, r["ok"])
        check("   H - V + 29T - 273", r["gravitational"], 0)

    # F_9, F_10 and F_11 are not bases. The -n section carries points where
    # the Weierstrass model is non-minimal, and those have to be blown up
    # before there is a Calabi-Yau to compactify on -- after which the surface
    # is no longer Hirzebruch. The class refuses rather than reporting a
    # spectrum for a geometry that does not exist.
    for n in (9, 10, 11):
        check_true("F_%d is refused as a base" % n,
                   _raises(ValueError, FT.FTheory6D, "F%d" % n))
    check_true("and so is F_13, where no blowup helps",
               _raises(ValueError, FT.FTheory6D, "F13"))

    # A model built by hand rather than from the non-Higgsable defaults.
    # su(5) on a -1 curve of dP_3, with the matter derived.
    dP3 = FT.Base.del_pezzo(3)
    m = FT.FTheory6D(dP3, gauge=[("su(5)", [0, 1, 0, 0])])
    check("hand-built su(5) on a -1 curve of dP_3",
          m.gauge_group(), "su(5)")
    check("   matter is 13 fundamentals and an antisymmetric",
          (m.gauge[0][2]["fund"], m.gauge[0][2]["antisym"]),
          (Fraction(13), Fraction(1)))
    check_true("   and it is anomaly free", m.check_anomalies()["ok"])
    s = m.spectrum()
    check("   charged hypers", s["H_charged"], 13 * 5 + 10)
    check("   V = dim su(5)", s["V"], 24)

    # Two factors at once, on curves that do not meet, so no bifundamentals.
    m2 = FT.FTheory6D(FT.Base.del_pezzo(3),
                      gauge=[("su(5)", [0, 1, 0, 0]),
                             ("su(3)", [0, 0, 1, 0])])
    check("two factors", m2.gauge_group(), "su(5) x su(3)")
    check_true("still anomaly free", m2.check_anomalies()["ok"])
    check("rank adds", m2.spectrum()["rank"], 6)

    # An explicitly wrong spectrum must be caught rather than absorbed.
    bad = FT.FTheory6D(FT.Base.del_pezzo(3),
                       gauge=[("su(5)", [0, 1, 0, 0], {"fund": 12,
                                                       "antisym": 1})])
    check_true("a wrong multiplicity fails the check",
               not bad.check_anomalies()["ok"])

    check_true("an unreadable base name is refused",
               _raises(ValueError, FT.FTheory6D, "K3"))


def test_duality():
    print("\n[6] heterotic duality")

    # F-theory on the elliptic threefold over F_n is dual to E8 x E8 on K3
    # with instanton numbers (12+n, 12-n). The end of the range is the check
    # worth making: F_12 means no instantons in the second E8, so that E8 is
    # unbroken -- and independently, the -12 section of F_12 carries e8.
    for n in [k for k in range(13) if k not in (9, 10, 11)]:
        d = FT.FTheory6D("F%d" % n).heterotic_dual()
        check("F_%-2d <-> instantons %s" % (n, (12 + n, 12 - n)),
              d["instantons"], (12 + n, 12 - n))

    # The three bases that need blowing up are exactly the three instanton
    # numbers 1, 2, 3 -- too few to break E8, and on the heterotic side the
    # small instanton transition that produces extra tensor multiplets is the
    # same event as the blowup on the F-theory side.
    check("F_9, F_10, F_11 correspond to 3, 2, 1 instantons",
          [12 - n for n in (9, 10, 11)], [3, 2, 1])
    check("and to 3, 2, 1 blowups",
          [FT.non_higgsable(n)["blowups"] for n in (9, 10, 11)], [3, 2, 1])
    d12 = FT.FTheory6D("F12").heterotic_dual()
    check("no instantons in the second E8", d12["instantons"][1], 0)
    check("so that E8 is unbroken", d12["unbroken_from_second_E8"], "e8")
    check("and the geometry says the same",
          FT.FTheory6D("F12").gauge_group(), "e8")

    # Below n = 3 there are enough instantons to break the second E8 entirely.
    for n in range(3):
        check("F_%d leaves nothing unbroken" % n,
              FT.FTheory6D("F%d" % n).heterotic_dual()
              ["unbroken_from_second_E8"], None)

    check_true("P^2 has no Hirzebruch dual to report",
               FT.FTheory6D("P2").heterotic_dual() is None)


def test_absent_couplings():
    print("\n[7] couplings that do not exist")

    m = FT.FTheory6D("F4")

    # Three different reasons a number can be unavailable, and the package
    # should not blur them. Here the reason is supersymmetry: six-dimensional
    # (1,0) admits no superpotential, so there is no Yukawa coupling to
    # compute, approximate or look up.
    check_true("holomorphic_yukawa raises NoSuchTheory",
               _raises(FT.NoSuchTheory, m.holomorphic_yukawa))
    check_true("physical_yukawa too", _raises(FT.NoSuchTheory,
                                              m.physical_yukawa))
    check_true("fermion_masses too", _raises(FT.NoSuchTheory,
                                             m.fermion_masses))

    # It is not the metric's fault, and the exception must not say it is.
    check_true("and it is not a NeedsMetric",
               not _raises(T.NeedsMetric, m.holomorphic_yukawa))
    check_true("but it is still a NotImplementedError, so callers that "
               "catch broadly still work",
               _raises(NotImplementedError, m.holomorphic_yukawa))

    # The contrast: the heterotic line bundle model has couplings that exist
    # and are not implemented, which is a NotImplementedError but not a
    # NoSuchTheory.
    lb = T.LineBundleModel([[4, 5]], [[1], [1], [1], [1], [-4]])
    check_true("the heterotic coupling exists but is unimplemented",
               _raises(NotImplementedError, lb.holomorphic_yukawa))
    check_true("and is not reported as non-existent",
               not _raises(FT.NoSuchTheory, lb.holomorphic_yukawa))

    # What a metric would be needed for here is the moduli space, not the
    # couplings, and describe() should say so.
    missing = m.missing_for_physical()
    check_true("the metric is wanted for the hypermultiplet moduli space",
               any("hypermultiplet moduli" in s for s in missing))
    check_true("and the list says the Yukawas are not on it",
               any("in six dimensions there are none" in s for s in missing))
    d = m.describe()
    check_true("describe() names the gauge algebra", "so(8)" in d)
    check_true("describe() shows the anomaly cancelling",
               "H - V + 29T - 273 = 0" in d)
    check_true("describe() says there are no Yukawas",
               "none exist in six dimensions" in d)

    # The registry picked up both new theories.
    check("four theories registered", sorted(T.registry),
          ["f-theory-4d", "f-theory-6d", "heterotic-line-bundle",
           "heterotic-standard-embedding"])
    check_true("get() finds the six-dimensional one",
               T.get("f-theory-6d") is FT.FTheory6D)


def test_fibrations():
    print("\n[8] obvious fibrations of a configuration matrix")

    # The quintic is not fibred: one row, one column, nothing to split.
    check("the quintic has no obvious fibration",
          FT.obvious_fibrations([[4, 5]]), [])

    # The (3,3) hypersurface in P^2 x P^2 fibres in plane cubics over P^2,
    # two ways, one for each projection.
    f = FT.obvious_fibrations([[2, 3], [2, 3]])
    check("(3,3) in P^2 x P^2: two fibrations", len(f), 2)
    check("   fibre is a plane cubic", f[0]["fibre"], [[2, 3]])
    check("   base is P^2", f[0]["base_name"], "P^2")

    # The tetraquadric fibres over P^1 x P^1 in six ways, one per choice of
    # two of the four P^1 factors for the fibre.
    tetra = [[1, 2], [1, 2], [1, 2], [1, 2]]
    f = FT.obvious_fibrations(tetra)
    check("tetraquadric: six elliptic fibrations", len(f), 6)
    check_true("all over P^1 x P^1",
               all(x["base_name"] == "P^1 x P^1" for x in f))
    check_true("all with a (2,2) curve as fibre",
               all(x["fibre"] == [[1, 2], [1, 2]] for x in f))

    # It is also K3 fibred, four ways, one per P^1 left as the base.
    f2 = FT.obvious_fibrations(tetra, fibre_dim=2)
    check("tetraquadric: four K3 fibrations", len(f2), 4)
    check_true("each over a P^1",
               all(x["base_name"] == "P^1" for x in f2))

    # The fibre block of a Calabi-Yau is automatically Calabi-Yau, because its
    # rows already sum correctly once the base columns are zero. Check it.
    for conf in ([[2, 3], [2, 3]], tetra, [[1, 2, 0], [1, 0, 2], [3, 2, 2]]):
        for x in FT.obvious_fibrations(conf):
            rows = x["fibre"]
            ok = all(sum(r[1:]) == r[0] + 1 for r in rows)
            check_true("fibre of %s is Calabi-Yau" % (conf,), ok)
            break

    check_true("is_obviously_fibred agrees",
               FT.is_obviously_fibred(tetra)
               and not FT.is_obviously_fibred([[4, 5]]))

    # And the caution the docstring insists on: an obvious fibration is a
    # genus-one fibration, not a Weierstrass model. The (3,3) hypersurface
    # fibres over P^2 and has Hodge numbers (2, 83), computed here by the rest
    # of the package; the Weierstrass model over the same base has (2, 272).
    # Same base, same fibre type, different manifolds.
    X = CICY([[2, 3], [2, 3]])
    check("(3,3) in P^2 x P^2 has h^{1,1} = 2", int(X.h[2]), 2)
    check("and h^{2,1} = 83", int(X.h[1]), 83)
    check("the Weierstrass model over P^2 has 272",
          FT.FTheory6D("P2").hodge_numbers()[1], 272)
    check_true("so they are different manifolds",
               (int(X.h[2]), int(X.h[1])) != FT.FTheory6D("P2").hodge_numbers())

    # The sweep, against the published number. Anderson, Gao, Gray and Lee
    # report 7837 of the 7890 CICY threefolds admitting an obvious genus-one
    # fibration; the criterion here is implemented from the definition, so
    # agreement is a check on the implementation rather than a lookup.
    path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "data", "cicylist.json")
    if os.path.exists(path):
        import json
        with open(path) as fh:
            entries = json.load(fh)["entries"]
        check("the published list has 7890 entries", len(entries), 7890)
        n = sum(1 for e in entries if FT.obvious_fibrations(e["conf"]))
        check("7837 have an obvious genus-one fibration", n, 7837)
    else:
        print("  {:<58} {:>14} {}".format(
            "the published list is absent, sweep skipped", "-", "skip"))


def test_fourfolds():
    print("\n[9] fourfolds and the D3 tadpole")

    # The (3,4) hypersurface in P^2 x P^3: a Calabi-Yau fourfold, elliptically
    # fibred over P^3 by projection.
    m = FT.FTheory4D([[2, 3], [3, 4]])
    check_true("obviously fibred", m.is_elliptically_fibred())
    check("over P^3", m.fibrations[0]["base_name"], "P^3")
    check("fibre is a plane cubic", m.fibrations[0]["fibre"], [[2, 3]])

    t = m.d3_tadpole()
    check("chi", t["chi"], 2016)
    check("chi/24", t["tadpole"], Fraction(84))
    check_true("integral, so integral flux is consistent", t["integral"])

    # The spectrum is not available and the reason is the flux, which is a
    # choice this package does not carry rather than a computation it declines
    # to do.
    check_true("spectrum raises", _raises(T.NeedsMetric, m.spectrum))
    try:
        m.spectrum()
    except T.NeedsMetric as e:
        check_true("and names the flux as what is missing",
                   any("G-flux" in s for s in e.missing))

    # A threefold is not a fourfold, and the error should say which theory to
    # use instead.
    check_true("a threefold is refused",
               _raises(ValueError, FT.FTheory4D, [[4, 5]]))


def main():
    t0 = time.time()
    test_kodaira()
    test_matter_free_derivation()
    test_matter()
    test_bases()
    test_spectra()
    test_duality()
    test_absent_couplings()
    test_fibrations()
    test_fourfolds()

    print("\n" + "=" * 72)
    if FAILURES:
        print("FAILED (%d): %s" % (len(FAILURES), ", ".join(FAILURES)))
        return 1
    print("test_ftheory: all checks passed in %.1fs" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
