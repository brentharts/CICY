"""
Tests for pyCICY.theories.orientifold.

The module computes one thing that matters -- the equivariant Hodge split of
an involution -- and computes it two ways that share no code. Most of what
follows is running both and requiring them to agree, plus the cases where
they must both *refuse*, which turn out to be the interesting ones.

  [1] euler         Euler characteristics of complete intersections in
                    products of projective spaces, against hand-checkable
                    values, since the fixed locus is one of these
  [2] involution    the sign on Omega, the fixed components, and that the
                    fixed dimensions match the sign rather than being told to
  [3] agreement     Lefschetz against monomial counting, on every involution
                    of the quintic, the bicubic and the tetraquadric
  [4] complement    flipping a coordinate set and flipping its complement are
                    the same map, and must give the same orientifold
  [5] degenerate    the involutions that force X to contain an ambient
                    subspace, where the configuration's Hodge numbers stop
                    describing X, and both routes have to refuse
  [6] spectrum      the closed string multiplet counting, and the O5/O9 case
                    declining rather than guessing
  [7] sen           the weak coupling limit: a K3 for every base, the D7
                    tadpole closing, and the brane rules reproducing Kodaira
  [8] euler_cross   three routes to the Euler characteristic of the elliptic
                    threefold, agreeing exactly where the model is smooth

Run with:  python3 tests/test_orientifold.py
       or: python3 run_tests.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyCICY import theories as T
from pyCICY.theories import ftheory as FT
from pyCICY.theories import orientifold as O

FAILURES = []

QUINTIC = [[4, 5]]
BICUBIC = [[2, 3], [2, 3]]
TETRA = [[1, 2], [1, 2], [1, 2], [1, 2]]


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


def test_euler():
    print("\n[1] Euler characteristics of complete intersections")

    # The fixed locus of an involution is a complete intersection in a product
    # of projective spaces and is almost never Calabi-Yau, so CICY cannot be
    # used on it. These are the cases that can be checked by hand.
    check("quintic threefold", O.complete_intersection_euler([4], [[5]]), -200)
    check("quartic surface in P^3 is a K3",
          O.complete_intersection_euler([3], [[4]]), 24)
    check("quintic surface in P^3",
          O.complete_intersection_euler([3], [[5]]), 55)

    # A degree-d surface in P^3 has chi = d^3 - 4d^2 + 6d, and a plane curve
    # of degree d has genus (d-1)(d-2)/2. Both are standard and independent of
    # the Chern class computation being tested.
    for d in range(1, 8):
        check("degree %d surface in P^3" % d,
              O.complete_intersection_euler([3], [[d]]),
              d ** 3 - 4 * d ** 2 + 6 * d)
    for d in range(1, 8):
        g = (d - 1) * (d - 2) // 2
        check("degree %d plane curve, genus %d" % (d, g),
              O.complete_intersection_euler([2], [[d]]), 2 - 2 * g)

    # With no equations the answer is the ambient Euler characteristic, which
    # is the product of the n_i + 1.
    check("P^4", O.complete_intersection_euler([4], []), 5)
    check("P^1 x P^2", O.complete_intersection_euler([1, 2], []), 6)
    check("P^1 x P^1 x P^1", O.complete_intersection_euler([1, 1, 1], []), 8)

    # In dimension zero the formula returns the intersection number, which is
    # the product of the degrees. This is the case the O3-planes need.
    check("two conics in P^2 meet in 4 points",
          O.complete_intersection_euler([2], [[2], [2]]), 4)
    check("three quadrics in P^3 meet in 8",
          O.complete_intersection_euler([3], [[2], [2], [2]]), 8)

    # Over-determined is empty, not an error.
    check("more equations than dimensions is empty",
          O.complete_intersection_euler([1], [[2], [2]]), 0)

    # And it agrees with the rest of the package where they overlap: the
    # bicubic's Euler characteristic from Chern classes against 2(h11 - h21).
    from pyCICY import CICY
    X = CICY(BICUBIC)
    check("bicubic, against CICY",
          O.complete_intersection_euler([2, 2], [[3, 3]]),
          int(X.euler_characteristic()))

    check_true("a multidegree of the wrong length is refused",
               _raises(ValueError, O.complete_intersection_euler,
                       [2, 2], [[3]]))


def test_involution():
    print("\n[2] the involution, its sign and its fixed locus")

    s = O.SignInvolution(QUINTIC, [[4]])
    check("one flip gives sigma^* Omega = -Omega", s.omega_sign(), -1)
    check("so O3 and O7 planes", s.oplane_type(), "O3/O7")
    check("two flips give +Omega",
          O.SignInvolution(QUINTIC, [[3, 4]]).omega_sign(), 1)
    check("and O5 and O9 planes",
          O.SignInvolution(QUINTIC, [[3, 4]]).oplane_type(), "O5/O9")

    # The fixed locus of the one-flip quintic: the hyperplane {x_4 = 0} meets
    # X in a quintic surface, and the point [0:0:0:0:1] lies on X because no
    # invariant quintic has an x_4^5 term.
    comps = s.fixed_components()
    check("two fixed components", len(comps), 2)
    check("an O7, the quintic surface", (comps[0]["oplane"],
                                         comps[0]["euler"]), ("O7", 55))
    check("and one O3 point", (comps[1]["oplane"], comps[1]["euler"]),
          ("O3", 1))
    check("chi of the fixed locus", s.fixed_euler(), 56)

    # This is the check, not the definition. The dimensions of the fixed
    # components are computed by intersecting polynomials with coordinate
    # subspaces; the Omega sign is computed by counting flipped coordinates.
    # That even codimension goes with the minus sign is a theorem, and the two
    # computations have to bear it out.
    for flips in ([[4]], [[3, 4]], [[2, 3, 4]], [[0]], [[0, 1]]):
        inv = O.SignInvolution(QUINTIC, flips)
        check_true("flips %s: dimensions match the Omega sign" % (flips,),
                   inv.consistent())
    for flips in ([[2], [2]], [[1, 2], [1, 2]], [[2], []]):
        inv = O.SignInvolution(BICUBIC, flips)
        check_true("bicubic %s: dimensions match" % (flips,),
                   inv.consistent())

    # Flipping every coordinate of a factor is the projective scaling, so an
    # involution built only out of those is the identity and there is nothing
    # to orientifold.
    check_true("flipping all five coordinates is the identity",
               _raises(ValueError, O.SignInvolution, QUINTIC,
                       [[0, 1, 2, 3, 4]]))
    check_true("and so is flipping none",
               _raises(ValueError, O.SignInvolution, QUINTIC, [[]]))
    check_true("a coordinate index out of range is refused",
               _raises(ValueError, O.SignInvolution, QUINTIC, [[7]]))

    # The sign in p(sigma x) = eps p(x) is usually free -- both choices give a
    # legitimate and different orientifold -- but not always. When the flipped
    # factor is one the polynomial has no degree in, every monomial uses an
    # even number of flipped coordinates and eps is forced to be +1. Here the
    # first polynomial has degree zero in the only flipped factor.
    FORCED = [[1, 2, 0], [1, 0, 2], [1, 1, 1], [1, 1, 1], [1, 1, 1]]
    inv = O.SignInvolution(FORCED, [[], [0], [], [], []])
    check("the first polynomial admits only even monomials",
          inv._monomial_parities(0), {0})
    check("the second admits both", inv._monomial_parities(1), {0, 1})
    check("so the default resolves to (+1, +1)", inv.signs, [1, 1])
    check_true("and the impossible sign is refused",
               _raises(ValueError, O.SignInvolution, FORCED,
                       [[], [0], [], [], []], [-1, 1]))
    check_true("as is a sign that is neither +1 nor -1",
               _raises(ValueError, O.SignInvolution, QUINTIC, [[4]], [0]))
    check_true("and a wrong number of signs",
               _raises(ValueError, O.SignInvolution, QUINTIC, [[4]], [1, 1]))

    # Where both signs are possible, both are built and they differ.
    up = O.SignInvolution(TETRA, [[1], [], [], []], poly_signs=[1])
    dn = O.SignInvolution(TETRA, [[1], [], [], []], poly_signs=[-1])
    check("both signs give involutions with opposite Omega sign",
          (up.omega_sign(), dn.omega_sign()), (-1, 1))


def test_agreement():
    print("\n[3] Lefschetz against counting monomials")

    # Two routes with nothing in common. One is the topological fixed point
    # theorem, fed by Euler characteristics of complete intersections. The
    # other grades the monomials of the defining polynomial and the
    # reparametrisations of the ambient. They must give the same split.
    cases = [
        (QUINTIC, [4], [[4]]),
        (QUINTIC, [4], [[3, 4]]),
        (QUINTIC, [4], [[0]]),
        (QUINTIC, [4], [[1, 2]]),
        (BICUBIC, [2, 2], [[2], [2]]),
        (BICUBIC, [2, 2], [[1, 2], [1, 2]]),
        (TETRA, [1, 1, 1, 1], [[1], [], [], []]),
        (TETRA, [1, 1, 1, 1], [[1], [1], [], []]),
        (TETRA, [1, 1, 1, 1], [[1], [1], [1], []]),
        (TETRA, [1, 1, 1, 1], [[1], [1], [1], [1]]),
        (TETRA, [1, 1, 1, 1], [[0], [0], [0], []]),
    ]
    degree = {id(QUINTIC): [5], id(BICUBIC): [3, 3],
              id(TETRA): [2, 2, 2, 2]}
    for conf, dims, flips in cases:
        inv = O.SignInvolution(conf, flips)
        if inv.degeneracies():
            continue
        a = inv.hodge_split()
        b = O.hypersurface_moduli_split(dims, degree[id(conf)], flips)
        check("%s %s: the two routes agree" % (conf[0], flips),
              a["h21"], b["h21"])
        check("   and on the total", sum(a["h21"]), int(inv.X.h[1]))
        check("   and on the Omega sign", a["omega_sign"], b["omega_sign"])

    # The quintic with one flip, worked out by hand in both directions. The
    # fixed locus is a quintic surface plus a point, chi = 56, and the
    # Lefschetz number 4 + 2(1) - 2u = 56 gives u = -25, so (38, 63).
    inv = O.SignInvolution(QUINTIC, [[4]])
    check("quintic, one flip", inv.hodge_split()["h21"], (38, 63))

    # The twist that makes the two agree, isolated. The monomials grade
    # H^1(T_X); H^{2,1} is that tensored with H^{3,0}, on which sigma acts by
    # the Omega sign. With the minus sign the gradings are opposite, so the
    # invariant deformations are the anti-invariant part of H^{2,1}. Getting
    # this backwards leaves both routes self-consistent and both wrong.
    m = O.hypersurface_moduli_split([4], [5], [[4]])
    check("deformations of H^1(T_X)", m["deformations"], (63, 38))
    check("and H^{2,1}, twisted by Omega", m["h21"], (38, 63))
    check_true("the twist is exactly the reversal",
               m["h21"] == m["deformations"][::-1])
    # With the plus sign there is no twist.
    m2 = O.hypersurface_moduli_split([4], [5], [[3, 4]])
    check("with +Omega there is no twist", m2["h21"], m2["deformations"])

    # h^{1,1} is untouched by a sign flip on a favourable X: the ambient
    # hyperplane classes span it and each is fixed.
    for conf, flips in [(QUINTIC, [[4]]), (BICUBIC, [[2], [2]]),
                        (TETRA, [[1], [], [], []])]:
        inv = O.SignInvolution(conf, flips)
        h11 = inv.hodge_split()["h11"]
        check("%s: h^{1,1} is entirely invariant" % (conf[0],),
              h11, (int(inv.X.h[2]), 0))


def test_complement():
    print("\n[4] a coordinate set and its complement are the same map")

    # Flipping S and flipping the complement of S differ by an overall sign,
    # which is the projective scaling, so they are the same involution of the
    # ambient. They give different values of sum_i minus_i, and the thing that
    # puts them back together is that they also give opposite polynomial
    # signs. If omega_sign ignored the polynomial sign, the answer would
    # depend on how the same map was written down.
    a = O.SignInvolution(QUINTIC, [[0]], poly_signs=[1])
    b = O.SignInvolution(QUINTIC, [[1, 2, 3, 4]], poly_signs=[-1])
    check("the two descriptions differ in flipped count",
          (sum(a.minus), sum(b.minus)), (1, 4))
    check("but agree on the Omega sign",
          (a.omega_sign(), b.omega_sign()), (-1, -1))
    check("and on chi of the fixed locus",
          (a.fixed_euler(), b.fixed_euler()), (56, 56))
    check("and on the Hodge split",
          a.hodge_split()["h21"], b.hodge_split()["h21"])
    check("and on the O-plane content",
          a.oplane_counts(), b.oplane_counts())

    # The other pairing of signs is the same map too, and it is the one that
    # makes the polynomial factorise.
    c = O.SignInvolution(QUINTIC, [[0]], poly_signs=[-1])
    d = O.SignInvolution(QUINTIC, [[1, 2, 3, 4]], poly_signs=[1])
    check("both descriptions of the other choice are reducible",
          ([x["kind"] for x in c.degeneracies()],
           [x["kind"] for x in d.degeneracies()]),
          (["reducible"], ["reducible"]))

    # Same on the bicubic, where the complement is taken in one factor only.
    e = O.SignInvolution(BICUBIC, [[2], [2]])
    f = O.SignInvolution(BICUBIC, [[0, 1], [2]], poly_signs=[-1])
    check("bicubic: same Omega sign",
          (e.omega_sign(), f.omega_sign()),
          (e.omega_sign(), e.omega_sign()))
    check("bicubic: same fixed locus", e.fixed_euler(), f.fixed_euler())
    check("bicubic: same split", e.hodge_split()["h21"],
          f.hodge_split()["h21"])


def test_degenerate():
    print("\n[5] the involutions that change the manifold")

    # Flipping three of the quintic's coordinates makes every invariant
    # monomial vanish on the plane spanned by those three, so X contains a
    # plane. A quintic containing a plane is a Noether-Lefschetz jump: the
    # plane's class is not in the lattice the ambient hyperplane generates, so
    # h^{1,1} is 2 rather than 1, and the configuration matrix's Hodge numbers
    # are numbers for a different manifold.
    inv = O.SignInvolution(QUINTIC, [[2, 3, 4]])
    bad = inv.degeneracies()
    check("three flips: X contains an ambient divisor", len(bad), 1)
    check("   reported as a Picard jump", bad[0]["kind"], "picard jump")
    check_true("   and the split refuses", _raises(ValueError,
                                                   inv.hodge_split))
    check_true("   while the fixed locus is still available",
               inv.fixed_euler() == 8)
    # The monomial route has to refuse for the same reason, not agree wrongly.
    check_true("   the monomial route refuses too",
               _raises(ValueError, O.hypersurface_moduli_split,
                       [4], [5], [[2, 3, 4]]))

    # Four flips is worse: every invariant monomial is divisible by the
    # remaining coordinate, so the polynomial factorises and X is not
    # irreducible.
    inv = O.SignInvolution(QUINTIC, [[1, 2, 3, 4]])
    check("four flips: X is reducible",
          [b["kind"] for b in inv.degeneracies()], ["reducible"])
    check_true("   and the split refuses",
               _raises(ValueError, inv.hodge_split))

    # The bicubic has the same phenomenon with a single flip in one factor.
    inv = O.SignInvolution(BICUBIC, [[2], []])
    check_true("bicubic, one flip in one factor, is degenerate",
               not inv.is_generic())

    # And the cases that are fine are reported as fine.
    for conf, flips in [(QUINTIC, [[4]]), (QUINTIC, [[3, 4]]),
                        (BICUBIC, [[2], [2]]),
                        (TETRA, [[1], [], [], []])]:
        check_true("%s %s is generic" % (conf[0], flips),
                   O.SignInvolution(conf, flips).is_generic())

    # A curve inside X is not a problem: a generic quintic contains 2875
    # lines, so containing one changes nothing about h^{1,1}. Only a divisor
    # does. The two-flip quintic contains a line and is generic.
    inv = O.SignInvolution(QUINTIC, [[3, 4]])
    inside = [c for c in inv.fixed_components() if c["inside_X"]]
    check("two flips: a line lies inside X", len(inside), 1)
    check("   of dimension one", inside[0]["dim"], 1)
    check_true("   which is not a degeneracy", inv.is_generic())


def test_spectrum():
    print("\n[6] the closed string spectrum")

    o = O.Orientifold(O.SignInvolution(QUINTIC, [[4]]))
    s = o.spectrum()
    check("h^{1,1} split", (s["h11_plus"], s["h11_minus"]), (1, 0))
    check("h^{2,1} split", (s["h21_plus"], s["h21_minus"]), (38, 63))
    # chiral = h11_+ + h11_- + h21_- + 1 = 1 + 0 + 63 + 1
    check("chiral multiplets", s["chiral"], 65)
    check("vector multiplets", s["vectors"], 38)
    check("the axio-dilaton is one of them", s["axio-dilaton"], 1)
    check("O-planes", sorted(s["oplanes"]), ["O3", "O7"])

    # An O5/O9 involution has a different multiplet assignment, which is not
    # implemented. It declines rather than reporting the O3/O7 answer.
    o5 = O.Orientifold(O.SignInvolution(QUINTIC, [[3, 4]]))
    check_true("O5/O9 declines to give a spectrum",
               _raises(NotImplementedError, o5.spectrum))
    check_true("but the Hodge split is still there",
               o5.involution.hodge_split()["h21"] == (53, 48))

    # The gauge group is an open string question and is not determined by the
    # involution; the tadpole is stated rather than solved.
    check_true("the gauge group is declared to be a separate choice",
               "separate choice" in o.gauge_group())
    t = o.d7_tadpole()
    check("one O7 component", t["o7_components"], 1)
    check_true("with the condition stated", "8 [O7]" in t["condition"])

    # The D3 tadpole needs flux, which the package does not carry.
    check_true("d3_tadpole raises", _raises(T.NeedsMetric, o.d3_tadpole))

    # Yukawa couplings here are a missing feature, unlike the six-dimensional
    # F-theory case where they do not exist. The exceptions must differ.
    check_true("Yukawas are unimplemented",
               _raises(NotImplementedError, o.holomorphic_yukawa))
    check_true("but not declared non-existent",
               not _raises(FT.NoSuchTheory, o.holomorphic_yukawa))
    check_true("whereas in six dimensions they are",
               _raises(FT.NoSuchTheory,
                       FT.FTheory6D("P2").holomorphic_yukawa))

    check("registered", "type-iib-orientifold" in T.registry, True)
    check_true("and reachable", T.get("type-iib-orientifold") is O.Orientifold)
    d = o.describe()
    check_true("describe mentions the O-plane type", "O3/O7" in d)
    check_true("describe shows the fixed locus", "fixed locus chi = 56" in d)


def test_sen():
    print("\n[7] Sen's weak coupling limit")

    # The O7-plane sits at h = 0 with h^2 a section of -4K, so [O7] = -2K.
    s = O.SenLimit("P2")
    check("O7 on P^2 is a sextic", s.o7_class(), [6])
    check("of genus 10", s.branch_genus(), 10)

    # The double cover of the base branched over the O7 is a K3 for every
    # base, because chi = 2(3 + T) + 2(9 - T) and the T cancels. Nothing puts
    # this in; it follows from K^2 = 10 - h^{1,1} on a rational surface.
    bases = (["P2"] + ["F%d" % n for n in range(9)] + ["F12"]
             + ["dP%d" % k for k in range(1, 9)])
    for spec in bases:
        lim = O.SenLimit(spec)
        check("%-5s double cover has chi = 24" % spec,
              lim.double_cover_euler(), 24)
        check_true("   so it is a K3", lim.double_cover_is_k3())

    # The branch genus does vary, even though the cover does not.
    check("the O7 genus is K^2 + 1, so it falls with T",
          [O.SenLimit("dP%d" % k).branch_genus() for k in range(1, 9)],
          [9, 8, 7, 6, 5, 4, 3, 2])

    # The D7 tadpole closes for every base: the Whitney brane is in -8K, and
    # with its image that is 8 [O7] = -16K.
    for spec in bases:
        lim = O.SenLimit(spec)
        r = lim.d7_tadpole()
        check_true("%-5s D7 charge cancels the O7 charge" % spec, r["cancels"])
    check("on P^2 both sides are -16K", O.SenLimit("P2").d7_tadpole()["branes"],
          [48])

    # The sharpest check in the module. Perturbatively, n D7-branes on an
    # O7-plane give so(2n) by Chan-Paton counting. Non-perturbatively the same
    # configuration is a fibre of type I_{n-4}^*, and ftheory.kodaira_type --
    # which knows nothing about branes -- says that carries so(2n-8+8).
    for n in range(4, 13):
        r = O.brane_stack(n, on_o7=True)
        check("%2d branes on the O7: %s from both sides" % (n, r["algebra"]),
              (r["algebra"], r["agree"]), ("so(%d)" % (2 * n), True))
        check("   via fibre type", r["kodaira"], "I_%d*" % (n - 4))
    for n in range(2, 10):
        r = O.brane_stack(n)
        check("%2d branes off the O7: %s from both sides" % (n, r["algebra"]),
              (r["algebra"], r["agree"]), ("su(%d)" % n, True))
        check("   via fibre type", r["kodaira"], "I_%d" % n)

    # Four is the number of branes that cancels the O7 charge locally, and it
    # is the number that gives the smooth I_0^* fibre. Fewer is not a
    # configuration.
    check("four branes on the O7 give so(8) and I_0*",
          (O.brane_stack(4, True)["algebra"],
           O.brane_stack(4, True)["kodaira"]), ("so(8)", "I_0*"))
    check_true("three is refused", _raises(ValueError, O.brane_stack, 3, True))
    check("a single brane has no non-abelian factor",
          O.brane_stack(1)["algebra"], None)
    check("   and I_1 carries none either",
          O.brane_stack(1)["kodaira_algebra"], None)
    check_true("   so they agree", O.brane_stack(1)["agree"])


def test_euler_cross():
    print("\n[8] three routes to the Euler characteristic")

    # The Hodge numbers of the elliptic threefold come from the anomaly; the
    # Chern classes give chi = -60 K^2 directly. Where the generic Weierstrass
    # model is smooth the two must agree.
    smooth = ["P2", "F0", "F1", "F2"] + ["dP%d" % k for k in range(1, 9)]
    for spec in smooth:
        m = FT.FTheory6D(spec)
        check("%-5s spectrum and Chern classes agree" % spec,
              m.euler_characteristic(), FT.weierstrass_euler(m.base))
        check_true("   and the model is smooth", m.gauge_group() == "trivial")

    # Where it is not smooth they must differ, and by the resolution of the
    # singular fibre. Reporting agreement there would mean one of them was
    # not computing what it says.
    for spec in ["F3", "F4", "F5", "F6", "F7", "F8", "F12"]:
        m = FT.FTheory6D(spec)
        check_true("%-5s is singular, so the two differ" % spec,
                   m.euler_characteristic() != FT.weierstrass_euler(m.base))
    check("F_12 differs by the e8 resolution",
          FT.FTheory6D("F12").euler_characteristic()
          - FT.weierstrass_euler(FT.Base.hirzebruch(12)), -480)

    # And the Chern class route is base-independent in the way K^2 is: every
    # Hirzebruch surface gives -480 because every one has K^2 = 8.
    check("all Hirzebruch bases give -480",
          sorted({FT.weierstrass_euler(FT.Base.hirzebruch(n))
                  for n in range(9)}), [-480])


def main():
    t0 = time.time()
    test_euler()
    test_involution()
    test_agreement()
    test_complement()
    test_degenerate()
    test_spectrum()
    test_sen()
    test_euler_cross()

    print("\n" + "=" * 72)
    if FAILURES:
        print("FAILED (%d): %s" % (len(FAILURES), ", ".join(FAILURES)))
        return 1
    print("test_orientifold: all checks passed in %.1fs" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
