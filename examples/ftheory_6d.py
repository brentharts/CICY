#!/usr/bin/env python3
"""
F-theory in six dimensions: what the anomalies determine, and what they do not.

    python3 examples/ftheory_6d.py
    python3 examples/ftheory_6d.py --base F7
    python3 examples/ftheory_6d.py --gauge 'su(5)' --divisor 0,1,0,0 --base dP3
    python3 examples/ftheory_6d.py --only clusters

F-theory on an elliptically fibred Calabi-Yau threefold gives six-dimensional
N=(1,0) supergravity, and that theory is unusually rigid. Anomaly cancellation
is not a consistency check applied after the spectrum is known; it is strong
enough to *produce* the spectrum from the divisor classes. Everything printed
below is integer or rational arithmetic on the intersection form of a surface.
There is no cohomology computation anywhere in this script and no metric.

The sections, in the order they build on each other:

    1  the derivation      which algebras can live on a curve with no charged
                           matter, from the anomaly conditions alone, and that
                           the answer is the tabulated non-Higgsable clusters
    2  matter              multiplicities solved from the same conditions
    3  the base            two independent counts of the same moduli
    4  a survey            the Hirzebruch and del Pezzo bases end to end
    5  a chosen model      an SU(5) put on a curve by hand
    6  what is missing     the three ways a number can be unavailable

The last section is the point of the exercise as much as the first. This
package is careful about the difference between a quantity that is exact, one
that needs the Ricci-flat metric, and one that does not exist -- and six
dimensions supplies the third kind, because (1,0) supersymmetry forbids a
superpotential and so forbids Yukawa couplings outright.
"""

import argparse
import sys

sys.path.insert(0, __file__.rsplit("/", 2)[0])

from pyCICY.theories import ftheory as FT


def banner(text):
    print("\n" + text)
    print("-" * len(text))


# ---------------------------------------------------------------------------


def derivation():
    banner("1. Which algebras need no charged matter")

    print("""\
Six-dimensional anomaly cancellation gives, for a simple algebra on an
irreducible divisor D,

    lam ( A_adj - sum_R x_R A_R )   =  6 K . D
    lam^2 ( sum_R x_R C_R - C_adj ) =  3 D . D

Set every x_R to zero on a rational curve and each becomes a formula for the
self-intersection:

    D.D = -2 - lam A_adj / 6      and      D.D = -lam^2 C_adj / 3 .

These are different functions of the group theory. They agree for very few
algebras, and at an integer for fewer still. Everything else is forced to
carry matter.
""")

    derived = FT.matter_free_algebras()
    print("  %-8s %8s %8s %8s %8s" % ("algebra", "lam", "A_adj", "C_adj",
                                      "D.D"))
    for name in ["su(2)", "su(3)", "su(5)", "so(8)", "so(10)", "g2", "f4",
                 "e6", "e7", "e8"]:
        d = FT.algebra_data(name)
        _, A, _, C = d["reps"][d["adjoint"]]
        n_A = -2 - FT.F(d["lam"]) * A / 6
        n_C = -FT.F(d["lam"]) ** 2 * C / 3
        verdict = ("D.D = %d" % derived[name] if name in derived
                   else "needs matter (%s vs %s)" % (n_A, n_C))
        print("  %-8s %8s %8s %8s   %s" % (name, d["lam"], A, C, verdict))

    print("\nSix survive. Compare Morrison and Taylor's table of the algebras"
          "\nforced on a curve of self-intersection -n in any base:\n")
    for n in range(1, 13):
        r = FT.non_higgsable(n)
        alg = r["algebra"] or "-"
        matter = ", ".join("%s x %s" % (v, k)
                           for k, v in sorted(r["matter"].items())) or "none"
        extra = ("  (%d blowup%s first)" % (r["blowups"],
                 "" if r["blowups"] == 1 else "s")) if r["blowups"] else ""
        print("  -%-3d %-8s matter: %-12s%s" % (n, alg, matter, extra))
    print("""
The six matter-free entries are exactly the six that survived above, at
exactly the same self-intersections. Nothing in the derivation looked at the
table. The one entry with matter, -7, is the interesting case: the anomaly
conditions there have the unique solution x = 1/2 for the 56 of e7, and half a
hypermultiplet is a real thing, because the 56 is pseudo-real.""")


def matter():
    banner("2. Matter multiplicities, solved")

    print("""\
The same two conditions, plus the quartic one where the algebra has an
independent quartic Casimir, determine the multiplicities outright. These are
the standard results, and none of them is stored anywhere in the package.
""")
    print("  %-10s %-24s" % ("D.D", "su(6) matter"))
    for n in (-2, -1, 0, 1, 2):
        m = FT.matter_content("su(6)", n)["matter"]
        print("  %-10d %s" % (n, ", ".join("%s x %s" % (v, k)
                                           for k, v in sorted(m.items()))))
    print("""
    -2 gives 2N fundamentals, -1 gives N + 8 and one antisymmetric,
     0 gives 16 and two: the familiar table, out of the linear algebra.
""")
    print("  %-14s %s" % ("genus 1, D.D = 4:",
                          FT.matter_content("su(6)", 4, genus=1)["matter"]))
    print("  a curve of genus g carries g adjoints, and the conditions know it")

    print("\n  and where there is no solution, that is the physics:")
    for args in [("so(10)", -3), ("e8", -11), ("su(12)", -3)]:
        try:
            FT.matter_content(*args)
        except ValueError as e:
            print("    %-12s D.D = %-4d %s" % (args[0], args[1],
                                               str(e).split(".")[0][:70]))


def base_counts():
    banner("3. Two routes to the same number")

    print("""\
The complex structure moduli of the generic Weierstrass model are a
Riemann-Roch count on the base: the coefficients f in H^0(-4K) and g in
H^0(-6K), less the automorphisms of B and the rescaling (f, g) -> (t^4 f,
t^6 g). The gravitational anomaly H - V + 29T = 273 is a one-loop condition in
six dimensions and says the answer must be 272 - 29T. The two computations
have nothing to do with each other.
""")
    print("  %-7s %4s %4s %8s %8s %8s %10s %10s"
          % ("base", "K^2", "T", "chi(T_B)", "h0(-4K)", "h0(-6K)",
             "moduli", "272-29T"))
    bases = ([FT.Base.P2()] + [FT.Base.hirzebruch(n) for n in range(3)]
             + [FT.Base.del_pezzo(k) for k in range(1, 9)])
    for b in bases:
        print("  %-7s %4d %4d %8d %8d %8d %10d %10d"
              % (b.name, b.K2, b.T, b.chi_tangent(), b.h0_anticanonical(4),
                 b.h0_anticanonical(6), FT.weierstrass_moduli(b),
                 272 - 29 * b.T))
    print("""
F_2 is the delicate row. Its automorphism group is one dimension larger than
F_0's, which on its own would undercount the moduli by one; the compensation
is that F_2 deforms to F_0, so h^1(T_B) = 1. Using chi(T_B) rather than
h^0(T_B) is what keeps the column at 243 across the Hirzebruch surfaces.""")


def survey():
    banner("4. The Hirzebruch and del Pezzo bases, end to end")

    print("  %-6s %3s %-8s %5s %5s %6s %6s   %-14s %8s"
          % ("base", "T", "algebra", "V", "H", "charged", "rank",
             "(h11, h21)", "chi"))
    specs = (["P2"] + ["F%d" % n for n in range(9)] + ["F12"]
             + ["dP%d" % k for k in range(1, 9)])
    for spec in specs:
        try:
            m = FT.FTheory6D(spec)
        except ValueError as e:
            print("  %-6s   refused: %s" % (spec, str(e).split(":")[0]))
            continue
        s = m.spectrum()
        h = m.hodge_numbers()
        assert m.check_anomalies()["ok"]
        print("  %-6s %3d %-8s %5d %5d %6d %6d   %-14s %8d"
              % (spec, s["T"], m.gauge_group(), s["V"], s["H"],
                 s["H_charged"], s["rank"], "(%d, %d)" % h,
                 m.euler_characteristic()))

    print("""
Every row cancels every anomaly, gravitational and gauge; the assert above
would have stopped the script otherwise.

F_9, F_10 and F_11 are missing on purpose. Their sections carry points where
the Weierstrass model is non-minimal, so they are not bases until those points
are blown up -- after which the surface is not Hirzebruch. On the heterotic
side those are the cases with 3, 2 and 1 instantons in the second E8, too few
to fit, and the small instanton transition that fixes it is the same event as
the blowup here.
""")
    print("  heterotic duals (E8 x E8 on K3, instanton numbers):")
    for n in [0, 4, 8, 12]:
        d = FT.FTheory6D("F%d" % n).heterotic_dual()
        print("    F_%-3d  %-10s  algebra the duality gives: %s"
              % (n, d["instantons"], d["unbroken_from_second_E8"] or "nothing"))
    print("""
    F_12 is the check worth making. No instantons in the second E8 means it is
    untouched, so the dual must have an unbroken E8 -- and independently, the
    -12 section of F_12 carries e8. Two derivations, one answer.""")


def chosen(base_name, algebra, divisor):
    banner("5. A model chosen rather than forced")

    base = FT._named_base(base_name)
    D = [int(x) for x in divisor.split(",")]
    if len(D) != base.h11:
        raise SystemExit("the divisor needs %d entries for %s, got %d"
                         % (base.h11, base.name, len(D)))
    print("  base %s, divisor %s, D.D = %d, genus %d"
          % (base.name, D, base.dot(D, D), base.genus(D)))
    m = FT.FTheory6D(base, gauge=[(algebra, D)])
    print()
    print(m.describe())
    r = m.check_anomalies()
    print("\n  anomaly residuals:")
    print("    gravitational          %s" % r["gravitational"])
    for g in r["gauge"]:
        for lab, res in zip(g["labels"], g["residuals"]):
            print("    %-38s %s" % (lab, res))


def kodaira():
    banner("6. Kodaira's table, and what the fibre carries")

    print("  %-14s %-8s %-12s %-12s" % ("(f, g, Delta)", "fibre", "split",
                                        "non-split"))
    for orders in [(0, 0, 0), (0, 0, 1), (0, 0, 5), (1, 1, 2), (1, 2, 3),
                   (2, 2, 4), (2, 3, 6), (2, 3, 8), (3, 4, 8), (3, 5, 9),
                   (4, 5, 10), (4, 6, 12)]:
        r = FT.kodaira_type(*orders)
        print("  %-14s %-8s %-12s %-12s"
              % (str(orders), r["type"], r["algebra_split"] or "-",
                 r["algebra_nonsplit"] or "-"))
    print("""
The last row is not a fibre type. ord(f) >= 4 with ord(g) >= 6 means the
Weierstrass model is non-minimal: there is no Calabi-Yau there until the base
is blown up. Reporting it as a gauge algebra would be reporting physics for a
geometry that does not exist.""")


def spinors_and_intersections():
    banner("8. so(N) spinors, and matter where divisors meet")

    print("""\
The spinor trace coefficients are not tabulated. The weights of a spinor are
(+-1/2, ..., +-1/2), so summing over sign patterns gives the traces directly,
and the index comes out as the dimension over eight for both even and odd
rank. so(8) is the exception: there the quartic expansion reaches a term
involving all four signs, which is the Pfaffian, and triality then makes the
spinors indistinguishable from the vector.
""")
    print("  %-8s %-6s %-8s %-8s %-8s %s"
          % ("algebra", "dim", "A", "B", "C", "reality"))
    for N in (7, 8, 9, 10, 11, 12, 13, 14, 16):
        d, A, B, C = FT.algebra_data("so(%d)" % N)["reps"]["spinor"]
        print("  %-8s %-6d %-8s %-8s %-8s %s"
              % ("so(%d)" % N, d, A, B, C, FT.reality("so(%d)" % N, "spinor")))

    print("""
With spinors available so(N) is no longer confined to the -4 curve, where the
vectors alone would put it. The reality column is what says which
multiplicities are allowed: half a hypermultiplet exists only for a
pseudo-real representation.
""")
    print("  %-9s %s" % ("D.D", "matter"))
    for N in (10, 12, 14):
        for n in (-4, -3, -2, 0):
            try:
                m = FT.matter_content("so(%d)" % N, n)["matter"]
                text = ", ".join("%s x %s" % (v, k)
                                 for k, v in sorted(m.items()))
            except ValueError as e:
                text = "refused: " + str(e).split(",")[-1].strip()[:46]
            print("  so(%-2d) %3d  %s" % (N, n, text))

    banner("   and bifundamentals")
    print("""\
Every defining representation tabulated here has index one, so the mixed
anomaly condition b_i . b_j = sum x A A makes the intersection number the
multiplicity outright. The states are not new: a bifundamental looks, to each
factor alone, like copies of that factor's defining representation, so the
spectrum has to remove the overlap rather than add anything. Getting that
wrong breaks the gravitational anomaly, which is how it gets caught.
""")
    B = FT.Base.hirzebruch(0)
    print("  %-20s %-7s %-8s %-10s %-9s %s"
          % ("gauge", "b_i.b_j", "shared", "H_charged", "(h11,h21)", "anomaly"))
    for a, b in [("su(5)", "su(3)"), ("su(6)", "su(4)"), ("so(10)", "su(2)"),
                 ("su(8)", "su(2)")]:
        m = FT.FTheory6D(B, gauge=[(a, [1, 0]), (b, [0, 1])])
        s = m.spectrum()
        bif = m.bifundamentals()[0]
        print("  %-20s %-7d %-8d %-10d %-9s %s"
              % ("%s x %s" % (a, b), bif["multiplicity"],
                 s["bifundamental_states"], s["H_charged"],
                 "%s" % (m.hodge_numbers(),),
                 "cancels" if m.check_anomalies()["ok"] else "FAILS"))


def fourfolds():
    banner("9. Four dimensions, and the box the flux lives in")

    print("""\
An elliptic fourfold over a threefold base gives four-dimensional N=1. Two
computations of its Euler characteristic, sharing nothing: h^{3,1} is a moduli
count on the base, and chi = 6(8 + h11 + h31 - h21) follows because X is a
Calabi-Yau fourfold; separately chi = 12 c_1 c_2 + 360 c_1^3 is a Chern number
of the fibration.
""")
    print("  %-16s %-7s %-7s %-5s %-7s %-9s %-9s %s"
          % ("base", "c1c2", "c1^3", "h11", "h31", "chi", "Chern", "agree"))
    for dims in ([3], [1, 2], [1, 1, 1]):
        B = FT.ProductBase(dims)
        h = FT.fourfold_hodge(B)
        print("  %-16s %-7d %-7d %-5d %-7d %-9d %-9d %s"
              % (B.name, B.chern_number([1, 2]), B.chern_number([1, 1, 1]),
                 h["h11"], h["h31"], h["euler"], h["euler_chern"],
                 "yes" if h["agree"] else "NO"))

    print("""
On P^3 both give 23328, so the D3 tadpole is 972. The flux is then bounded
without being determined:
""")
    m = FT.FTheory4D.over([3])
    f = m.flux()
    print("    chi/24 = %s, so N_D3 + (1/2) int G ^ G = %s"
          % (f["tadpole"], f["tadpole"]))
    print("    with N_D3 >= 0 this gives  int G ^ G <= %s" % f["max_GG"])
    for c in f["conditions"]:
        print("      - %s" % c)
    print("""
None of that is a spectrum. The chiral index is an index twisted by G, and G
is a choice this package does not carry, so spectrum() raises rather than
returning a number that would look like a prediction.""")


def clusters_and_sections():
    banner("10. Clusters that span several curves, and extra sections")

    print("""\
A curve too shallow to force anything on its own can still be part of a
cluster that forces something collectively, because the intersections tie the
Weierstrass model on one curve to its neighbour's. There are three such
clusters, and the matter is derived rather than tabulated.

The normalisation of the mixed condition is what they test. It is
b_i . b_j = lam_i lam_j sum x A A, and the lambdas are not decoration: on the
(-3, -2) cluster the g2's own conditions give it exactly one 7, while a full
(7, 2) would need two. With the lambdas the multiplicity is one half, needing
exactly the one 7 there is.
""")
    for c in FT.NON_HIGGSABLE_CLUSTERS:
        r = FT.cluster(c["curves"], c["edges"])
        print("  %-14s %s" % (c["name"],
                              " + ".join(a or "(nothing)"
                                         for _, a in c["curves"])))
        for k, (n, alg) in enumerate(c["curves"]):
            m = ", ".join("%s x %s" % (v, kk)
                          for kk, v in sorted(r["matter"][k].items()))
            print("      curve %d, self-intersection %3d: %s"
                  % (k, n, m or "no matter"))
        for sh in r["shared"]:
            print("      shared: %s x (%s, %s) between %s and %s"
                  % (sh["multiplicity"], sh["reps"][0], sh["reps"][1],
                     sh["algebras"][0], sh["algebras"][1]))
        print("      V = %d, rank = %d, charged = %s, closes: %s\n"
              % (r["dim_V"], r["rank"], r["charged"], r["consistent"]))

    print("""\
The three-curve cluster is the sharper case, because the middle curve's matter
is claimed twice. so(7) on a -3 curve is given two spinors by its own
conditions and each su(2) neighbour claims one, exactly. Note also which
representation carries the shared matter: the spinor, not the vector, because
on a -3 curve so(7) has no vectors at all. Both have index one, so the choice
is made by what the curve carries.
""")

    banner("   extra sections")
    print("""\
A rational section beyond the zero section adds a U(1) and a divisor. The
abelian conditions are the non-abelian ones with the adjoint dropped and
lam A_R replaced by q^2:

    sum_q x_q q^2 = 6 c_1 . b ,      sum_q x_q q^4 = 3 b . b

with b the height pairing. Two conditions, so two charges are determined and
more are not. Setting every multiplicity to zero forces b = 0, which is to say
a U(1) with no charged matter is not there.
""")
    print("  %-10s %-28s %s" % ("height", "charged matter", "model"))
    for h in range(2, 14, 2):
        try:
            mm = FT.FTheory6D("P2", abelian=[{"height": [h]}])
            u = mm.abelian[0]["matter"]
            text = ", ".join("%s of charge %d" % (v, q)
                             for q, v in sorted(u.items()))
            print("  %-10s %-28s (h11, h21) = %s"
                  % ("%dH" % h, text, mm.hodge_numbers()))
        except ValueError:
            print("  %-10s %-28s -" % ("%dH" % h, "no integer spectrum"))
    print("""
Only some height pairings admit a spectrum at all; the rest want fractional
numbers of states and are refused. Against the model with no extra section,
(h11, h21) = (2, 272), the section shows up in both entries: one more divisor,
and the charged states removed from the neutral count.""")


def missing():
    banner("7. Three ways a number can be unavailable")

    m = FT.FTheory6D("F4")
    print("""\
The package already separated quantities that are exact from quantities that
need the Ricci-flat metric. Six dimensions adds a third case, and the three
should not produce the same exception.
""")
    for label, fn in [
            ("F-theory 6d, Yukawa coupling", m.holomorphic_yukawa)]:
        try:
            fn()
        except FT.NoSuchTheory as e:
            print("  %s\n    NoSuchTheory: %s\n" % (label, _wrap(str(e))))

    from pyCICY import theories as T
    lb = T.LineBundleModel([[4, 5]], [[1], [1], [1], [1], [-4]])
    try:
        lb.holomorphic_yukawa()
    except NotImplementedError as e:
        print("  heterotic line bundle, holomorphic Yukawa\n"
              "    NotImplementedError: %s\n" % _wrap(str(e)))
    try:
        lb.physical_yukawa()
    except T.NeedsMetric as e:
        print("  heterotic line bundle, physical Yukawa\n"
              "    NeedsMetric: %s\n" % _wrap(str(e)))

    print("  the first does not exist, the second is not implemented, the "
          "third is\n  obstructed. Only the third is the metric's fault.")
    print("\n  what a metric would buy in six dimensions:")
    for s in m.missing_for_physical():
        print("    - %s" % _wrap(s, indent=6))


def _wrap(text, width=68, indent=4):
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    lines.append(cur)
    return ("\n" + " " * indent).join(lines)


# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--base", default="dP3",
                   help="base for the chosen model, e.g. P2, F7, dP3")
    p.add_argument("--gauge", default="su(5)",
                   help="gauge algebra for the chosen model")
    p.add_argument("--divisor", default="0,1,0,0",
                   help="divisor class, comma separated, in the base's basis")
    p.add_argument("--only", default=None,
                   choices=["clusters", "matter", "counts", "survey",
                            "model", "kodaira", "spinors",
                            "fourfolds", "clusters2", "missing"],
                   help="run one section instead of all of them")
    a = p.parse_args()

    sections = {"clusters": derivation, "matter": matter,
                "spinors": spinors_and_intersections, "fourfolds": fourfolds,
                "clusters2": clusters_and_sections,
                "counts": base_counts, "survey": survey,
                "model": lambda: chosen(a.base, a.gauge, a.divisor),
                "kodaira": kodaira, "missing": missing}
    order = ["clusters", "matter", "counts", "survey", "model", "kodaira",
             "spinors", "fourfolds", "clusters2", "missing"]
    for key in ([a.only] if a.only else order):
        sections[key]()
    print()


if __name__ == "__main__":
    main()
