"""
Tests for pyCICY.theories.mtheory.

The module computes very little that is hard, and that is the point: almost
everything is a Hodge number rearranged, or the triple intersection numbers
under a new name. What is worth testing is therefore not the arithmetic but
the *identifications* -- that the rearrangements are the right ones, that the
two independent routes to the same five-dimensional spectrum agree, and that
the things declared impossible are declared impossible for the stated reason
rather than by accident.

  [1] prepotential   the cubic is the intersection form: integrality, full
                     symmetry in its three indices, and the quintic's 5/6
  [2] fivedim        the 5d multiplet counting, the gauge kinetic matrix as
                     the Kahler cone condition, and M5-brane tensions
  [3] g2betti        the barely G_2 Betti numbers: b_2 + b_3 = 1 + h11 + h21,
                     vanishing Euler characteristic, Poincare duality, and
                     the eigenvalue swap that an antiholomorphic involution
                     forces
  [4] g2refuses      the omega_sign = -1 involutions must be refused; they
                     are orientifolds, and the two constructions differ by
                     exactly that sign
  [5] g2spectrum     the 4d N=1 content, and NoChiralMatter raised as a
                     theorem rather than as a missing computation
  [6] duality        M-theory on an elliptic threefold against F-theory on a
                     circle, over every base the package knows; the two sides
                     share no code
  [7] threedim       chi/24 on fourfolds, and that it agrees with the D3
                     tadpole of the dual F-theory model
  [8] categories     the three kinds of unavailable answer stay distinct:
                     NoSuchTheory, NoChiralMatter and NeedsMetric are raised
                     in the right places and never confused for each other
  [9] hw             Horava-Witten: the scale ordering, and its insensitivity
                     to the convention-dependent O(1) factors

Run with:  python3 tests/test_mtheory.py
       or: python3 run_tests.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from pyCICY import CICY
from pyCICY import theories as T
from pyCICY.theories import ftheory as FT
from pyCICY.theories import mtheory as M
from pyCICY.theories import orientifold as O
from pyCICY.theories.base import NeedsMetric

FAILURES = []

QUINTIC = [[4, 5]]
BICUBIC = [[2, 3], [2, 3]]
TETRA = [[1, 2], [1, 2], [1, 2], [1, 2]]
TWOPARAM = [[2, 2, 1], [3, 1, 3]]
SEXTIC4 = [[5, 6]]


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


def check_close(name, got, want, tol=1e-9):
    ok = abs(got - want) < tol
    print("  {:<58} {:>14} {}".format(name, "%.6g" % got,
                                      "ok" if ok else "FAIL want %.6g" % want))
    if not ok:
        FAILURES.append(name)


def _raises(exc, fn, *a, **kw):
    try:
        fn(*a, **kw)
    except exc:
        return True
    except Exception:                                            # noqa: BLE001
        return False
    return False


def banner(text):
    print("\n" + text)
    print("-" * len(text))


# ---------------------------------------------------------------------------
# [1] the prepotential is the intersection form
# ---------------------------------------------------------------------------

def test_prepotential():
    banner("[1] the prepotential is the intersection form, exactly")

    for conf, name in ((QUINTIC, "quintic"), (BICUBIC, "bicubic"),
                       (TETRA, "tetraquadric"), (TWOPARAM, "two-parameter")):
        X = CICY(conf)
        d = M.prepotential_coefficients(X)

        # An intersection number is an intersection number: integral, and
        # symmetric in all three indices because the wedge product of even
        # forms is.
        check_true("%-13s d_rst integral" % name, d.dtype.kind == "i")
        check_true("%-13s d_rst totally symmetric" % name,
                   np.array_equal(d, d.transpose(1, 0, 2))
                   and np.array_equal(d, d.transpose(0, 2, 1)))
        check_true("%-13s agrees with triple_intersection()" % name,
                   np.allclose(d, np.asarray(X.triple_intersection())))

    # The quintic has one Kahler modulus and d_111 = 5, so F(1) = 5/6. This is
    # the whole two-derivative vector multiplet action of the five-dimensional
    # theory, not the leading term of one.
    X = CICY(QUINTIC)
    check("quintic d_111", int(M.prepotential_coefficients(X)[0, 0, 0]), 5)
    check_close("quintic F(t=1) = 5/6", M.prepotential(X, [1.0]), 5.0 / 6.0)

    # Homogeneity of degree three, which is what makes the moduli space a
    # projective cubic hypersurface rather than anything more complicated.
    Y = CICY(TWOPARAM)
    t = np.array([1.3, 0.7])
    check_close("F is homogeneous of degree 3",
                M.prepotential(Y, 2 * t), 8 * M.prepotential(Y, t), 1e-9)

    # Wrong number of moduli is an error, not a silent broadcast.
    check_true("wrong modulus count refused",
               _raises(ValueError, M.prepotential, Y, [1.0]))


# ---------------------------------------------------------------------------
# [2] five dimensions
# ---------------------------------------------------------------------------

def test_fivedim():
    banner("[2] M-theory on a threefold: five dimensions, eight supercharges")

    m = M.MTheory5D(CICY(QUINTIC))
    s = m.spectrum()
    check("quintic vector multiplets  h11 - 1", s["vector_multiplets"], 0)
    check("quintic hypermultiplets    h21 + 1", s["hypermultiplets"], 102)
    check("quintic supercharges", s["supercharges"], 8)

    # The one-loop Chern-Simons coefficient is the second Chern class, and on
    # the quintic c_2 . J = 50.
    check("quintic c_2 . J", int(m.higher_derivative_coefficient()[0]), 50)

    # h11 = 1 means no vector multiplets at all: the only gauge field is the
    # graviphoton, which sits in the gravity multiplet.
    check_true("quintic has only the graviphoton",
               "graviphoton" in m.gauge_group())

    n = M.MTheory5D(CICY(TWOPARAM))
    check("two-parameter vector multiplets", n.spectrum()["vector_multiplets"], 1)
    check("two-parameter hypermultiplets", n.spectrum()["hypermultiplets"], 60)

    # The matrix d_rst t^t is NOT positive definite, and expecting it to be is
    # the natural mistake. The Hodge index theorem says it has signature
    # (1, h11 - 1) on the Kahler cone: one volume direction, and the physical
    # gauge kinetic term is minus the restriction to fixed volume.
    g = m.gauge_couplings([1.0])
    check("quintic signature", g["signature"], (1, 0))
    check_close("   and volume = F(t)", g["volume"], 5.0 / 6.0)

    for t in ([1.0, 1.0], [1.0, 6.0], [3.0, 1.0], [0.5, 2.0]):
        gg = n.gauge_couplings(t)
        check("two-parameter signature at t = %s" % (t,),
              gg["signature"], (1, 1))
    check_true("   which is the Hodge index condition",
               n.gauge_couplings([1.0, 1.0])["lorentzian"])

    # The signature is a real condition, not something every symmetric matrix
    # satisfies. Note it cannot be broken by flipping the sign of t, since
    # a_rs is linear in t and negating it negates every eigenvalue -- for two
    # moduli (1,1) is preserved. It takes a genuinely different direction, and
    # on the tetraquadric there is one.
    tetra = M.MTheory5D(CICY(TETRA))
    check("tetraquadric signature at t = (1,1,1,1)",
          tetra.gauge_couplings([1.0] * 4)["signature"], (1, 3))
    check("   and (2,2) somewhere outside the cone",
          tetra.gauge_couplings([0.377, -0.396, 1.921, 0.315])["signature"],
          (2, 2))

    # An M5-brane on a divisor gives a string whose tension is the divisor
    # volume, quadratic in t with integer coefficients.
    check_close("M5 on the quintic hyperplane, t = 1",
                M.m5_string_tension(m.X, [1.0], [1.0]), 2.5)
    check_close("   scales as t^2",
                M.m5_string_tension(m.X, [1.0], [3.0]), 22.5)

    # Only threefolds. A fourfold has no five-dimensional reduction of this
    # kind and must be refused rather than producing an array of the wrong
    # rank.
    check_true("a fourfold is refused",
               _raises(ValueError, M.MTheory5D, CICY(SEXTIC4)))


# ---------------------------------------------------------------------------
# [3] the barely G_2 Betti numbers
# ---------------------------------------------------------------------------

def test_g2betti():
    banner("[3] barely G_2: the Betti numbers of (X x S^1)/sigma")

    # The identity that has to hold whatever the involution: the invariant and
    # anti-invariant parts together are all of H^*(X), so
    #     b_2 + b_3 = h11_- + 1 + h21 + h11_+ = 1 + h11 + h21.
    for h11p, h11m, h21 in [(1, 0, 101), (2, 3, 60), (0, 5, 19), (7, 7, 7)]:
        b = M.barely_g2_betti(h11p, h11m, h21)
        check("b2 + b3 = 1 + h11 + h21  (%d,%d,%d)" % (h11p, h11m, h21),
              b["b2"] + b["b3"], 1 + h11p + h11m + h21)
        # Every closed odd-dimensional manifold has vanishing Euler
        # characteristic. Getting this for free is a check that the Betti
        # list is Poincare dual to itself, not an independent fact.
        check("   Euler characteristic vanishes", b["euler"], 0)
        check_true("   Poincare duality b_k = b_{7-k}",
                   b["betti"] == b["betti"][::-1])
        check("   b_1 = 0, so the quotient is not a product with a circle",
              b["betti"][1], 0)

    # The eigenvalue swap. An antiholomorphic sigma has sigma^* = -tau^* on
    # H^{1,1}, so b_2 counts the tau-ANTI-invariant classes. The quintic has
    # h11 = 1 and its single Kahler class is tau-even, hence sigma-odd, hence
    # contributes to b_3 and not to b_2. If the split were taken the other way
    # round b_2 would come out 1 and the G_2 form dtheta ^ J + Re Omega would
    # not be invariant.
    g = M.BarelyG2(O.SignInvolution(QUINTIC, [[3, 4]]))
    b = g.betti()
    check("quintic quotient b_2 (= h11_-, not h11_+)", b["b2"], 0)
    check("quintic quotient b_3", b["b3"], 103)
    check("   full Betti list", b["betti"], [1, 0, 0, 103, 103, 0, 0, 1])

    # The same identity through the class rather than the free function.
    split = g.hodge_split()
    check("   b2 + b3 against the threefold Hodge numbers",
          b["b2"] + b["b3"],
          1 + sum(split["h11"]) + sum(split["h21"]))

    # b_3 has a second derivation that does not use tau at all: an
    # antiholomorphic involution is anti-symplectic on H^3(X, R), so its fixed
    # subspace is Lagrangian, of dimension h21 + 1. Adding the odd two-forms
    # gives b_3, and the two routes must agree.
    h11p, h11m = split["h11"]
    h21 = sum(split["h21"])
    check("   b_3 from the Lagrangian argument agrees",
          b["b3"], (h21 + 1) + h11p)


# ---------------------------------------------------------------------------
# [4] the involutions that are not G_2 involutions
# ---------------------------------------------------------------------------

def test_g2refuses():
    banner("[4] omega_sign = -1 is an orientifold, and must be refused")

    # Re Omega has to be invariant for the G_2 three-form to survive. An
    # involution with tau^* Omega = -Omega is an O3/O7 orientifold involution
    # instead. The two constructions consume the same objects and are told
    # apart by exactly this sign, so a silent acceptance here would produce
    # confident Betti numbers for a space that is not a G_2 manifold.
    bad = O.SignInvolution(QUINTIC, [[4]])
    check("the one-flip quintic involution has omega_sign", bad.omega_sign(), -1)
    check_true("   and BarelyG2 refuses it",
               _raises(ValueError, M.BarelyG2, bad))

    good = O.SignInvolution(QUINTIC, [[3, 4]])
    check("the two-flip involution has omega_sign", good.omega_sign(), 1)
    check_true("   and is accepted", isinstance(M.BarelyG2(good), M.BarelyG2))

    # Across a range of involutions of several threefolds. There are two
    # independent reasons to refuse -- the wrong sign on Omega, and an
    # involution that forces X to contain an ambient subspace so that the
    # configuration's Hodge numbers stop describing it -- and an involution is
    # accepted exactly when neither applies. Conflating them would let a
    # degenerate configuration through on the strength of its sign.
    tried = accepted = wrong_sign = degenerate = 0
    for conf, flips in [(QUINTIC, [[0]]), (QUINTIC, [[0, 1]]),
                        (QUINTIC, [[0, 1, 2]]), (QUINTIC, [[0, 1, 2, 3]]),
                        (BICUBIC, [[0], []]), (BICUBIC, [[0, 1], []]),
                        (BICUBIC, [[0], [1]]), (BICUBIC, [[0, 1], [0, 1]]),
                        (TETRA, [[0], [], [], []]),
                        (TETRA, [[0], [0], [], []])]:
        inv = O.SignInvolution(conf, flips)
        tried += 1
        sign_ok = inv.omega_sign() == 1
        try:
            inv.hodge_split()
            split_ok = True
        except ValueError:
            split_ok = False
        want = sign_ok and split_ok
        got = not _raises(ValueError, M.BarelyG2, inv)
        accepted += bool(got)
        wrong_sign += (not sign_ok)
        degenerate += (sign_ok and not split_ok)
        if got != want:
            check_true("involution %s %s misclassified" % (conf, flips), False)
    check_true("all %d involutions classified correctly" % tried, True)
    check_true("   %d accepted, %d wrong sign, %d degenerate: all three occur"
               % (accepted, wrong_sign, degenerate),
               accepted and wrong_sign and degenerate)

    # The degenerate refusal must say what it is about, not be mistaken for
    # the sign test. These have omega_sign = +1 and are still refused.
    bad_split = O.SignInvolution(QUINTIC, [[0, 1, 2, 3]])
    check("a degenerate involution still has omega_sign",
          bad_split.omega_sign(), 1)
    try:
        M.BarelyG2(bad_split)
        check_true("   but is refused anyway", False)
    except ValueError as e:
        check_true("   but is refused, for the other reason",
                   "Betti numbers" in str(e) and "Omega" not in str(e))


# ---------------------------------------------------------------------------
# [5] the four-dimensional spectrum, and the theorem
# ---------------------------------------------------------------------------

def test_g2spectrum():
    banner("[5] four dimensions, N=1, and no chiral matter as a theorem")

    m = M.MTheoryG2(O.SignInvolution(QUINTIC, [[3, 4]]))
    s = m.spectrum()
    check("vector multiplets = b_2", s["vector_multiplets"], 0)
    check("chiral multiplets = b_3", s["chiral_multiplets"], 103)
    check("supercharges", s["supercharges"], 4)
    check("charged chiral multiplets", s["charged_chiral_multiplets"], 0)
    check("chiral index", s["chiral_index"], 0)

    # The zero above is reported as a theorem, and asking for it directly has
    # to say so rather than returning the number. A construction that returned
    # 0 quietly would be indistinguishable from one that had not looked.
    check_true("chiral_matter() raises NoChiralMatter",
               _raises(M.NoChiralMatter, m.chiral_matter))
    check_true("holomorphic_yukawa() raises NoChiralMatter",
               _raises(M.NoChiralMatter, m.holomorphic_yukawa))

    # It is a NoSuchTheory, not a NeedsMetric. The distinction is the whole
    # point of the base class.
    check_true("NoChiralMatter is a NoSuchTheory",
               issubclass(M.NoChiralMatter, FT.NoSuchTheory))
    check_true("   and is not a NeedsMetric",
               not issubclass(M.NoChiralMatter, NeedsMetric))

    # A G_2 manifold from elsewhere, given only as Betti numbers, with no
    # threefold in sight. The class has to work without one.
    tcs = M.MTheoryG2((23, 204), name="a twisted connected sum")
    check("a (b2, b3) pair alone gives a spectrum",
          tcs.spectrum()["vector_multiplets"], 23)
    check("   and the gauge group", tcs.gauge_group(), "U(1)^23")

    # The moduli are real three-form deformations paired with C_3 periods:
    # one chiral multiplet each, and no Kahler / complex structure split.
    g = M.BarelyG2(O.SignInvolution(QUINTIC, [[3, 4]]))
    check("moduli count matches b_3",
          g.moduli()["chiral_multiplets"], g.betti()["b3"])

    # The orbifold locus is described but its component count is declined,
    # because the real locus depends on the coefficients of the defining
    # polynomials and not on the configuration matrix.
    sl = g.singular_locus()
    check_true("the singular locus gives SU(2), not chirality",
               sl["chiral_matter"] is False and "SU(2)" in sl["gauge_enhancement"])
    check_true("   and the component count is declined, not guessed",
               len(sl["not_computed"]) >= 1)


# ---------------------------------------------------------------------------
# [6] the duality, which is the only real cross-check here
# ---------------------------------------------------------------------------

def test_duality():
    banner("[6] M-theory on an elliptic threefold = F-theory on a circle")

    # The two sides share no code and no reasoning. On one, h^{1,1} and
    # h^{2,1} of the resolved threefold; on the other, the six-dimensional
    # spectrum fixed by H - V + 29 T = 273, an equation with no geometry in
    # it. Reducing on a circle they must give the same five-dimensional
    # multiplet counts.
    bases = (["P2", "F0", "F1", "F2", "F3", "F4", "F5", "F6", "F8", "F12"]
             + ["dP%d" % k for k in range(1, 9)])
    agreed = 0
    for spec in bases:
        f = FT.FTheory6D(spec)
        c = M.circle_reduction_of_6d(f)
        ok = c["agrees"]
        agreed += bool(ok)
        if not ok:
            check_true("%-5s duality holds" % spec, False)
    check("every base agrees on n_V and n_H", agreed, len(bases))

    # Spot values, so that a uniform sign error could not pass the loop above.
    c = M.circle_reduction_of_6d(FT.FTheory6D("P2"))
    check("P2:  n_V (only the KK vector)", c["n_V_from_geometry"], 1)
    check("P2:  n_H", c["n_H_from_geometry"], 273)
    c = M.circle_reduction_of_6d(FT.FTheory6D("F12"))
    check("F12: n_V = rank(e8) + T + 1", c["n_V_from_geometry"], 10)
    check("   from the 6d side too", c["n_V_from_6d_spectrum"], 10)
    check("   rank and tensor count", (c["rank"], c["T"]), (8, 1))

    # The relation is n_V = rank + T + 1 in general, so raising the rank by
    # one raises h^{1,1} by one. Checking it as a difference catches an error
    # in the constant that the absolute values might hide.
    a = M.circle_reduction_of_6d(FT.FTheory6D("F3"))
    b = M.circle_reduction_of_6d(FT.FTheory6D("F4"))
    check("rank difference tracks h^{1,1} difference",
          b["h11"] - a["h11"], b["rank"] - a["rank"])


# ---------------------------------------------------------------------------
# [7] three dimensions and the tadpole
# ---------------------------------------------------------------------------

def test_threedim():
    banner("[7] M-theory on a fourfold: chi/24")

    m = M.MTheory3D(CICY(SEXTIC4))
    t = m.tadpole()
    check("sextic fourfold Euler characteristic", t["euler"], 2610)
    check_close("   chi/24", t["chi_over_24"], 108.75)

    # 2610/24 is not an integer, so this vacuum cannot be built from membranes
    # alone: it needs half-integral flux, which is exactly what the Witten
    # quantisation condition permits when c_2 is not even. Reporting the
    # fraction rather than rounding it is the point.
    check_true("   so it is not integral", not t["integral"])
    check_true("   and half-integral flux is required",
               t["half_integral_flux_allowed"])

    check("sextic complex structure moduli",
          m.spectrum()["complex_structure_moduli"], 426)

    # A threefold has no such reduction.
    check_true("a threefold is refused",
               _raises(ValueError, M.MTheory3D, CICY(QUINTIC)))

    # And the tadpole of the M-theory reduction on a fourfold is the D3
    # tadpole of the dual four-dimensional F-theory model: the same rational
    # number, reached from an Euler characteristic on one side and from the
    # Chern numbers of a threefold base on the other. The bases here are
    # threefolds, since a four-dimensional F-theory model needs one.
    for dims in ([3], [1, 2], [1, 1, 1]):
        base = FT.ProductBase(dims)
        f = FT.FTheory4D.over(base)
        d3 = f.d3_tadpole()
        chi = FT.fourfold_euler(base)
        check("%-9s fourfold chi" % base.name, int(chi), int(d3["chi"]))
        check_close("%-9s chi/24 = D3 tadpole" % base.name,
                    chi / 24.0, float(d3["tadpole"]))
        # The M-theory statement of the same equation, on the same number.
        check_true("%-9s the two tadpole reports agree on integrality"
                   % base.name,
                   bool(d3["integral"]) == (int(chi) % 24 == 0))


# ---------------------------------------------------------------------------
# [8] the three kinds of unavailable answer stay distinct
# ---------------------------------------------------------------------------

def test_categories():
    banner("[8] exact, does-not-exist, and needs-a-metric stay apart")

    five = M.MTheory5D(CICY(QUINTIC))
    g2 = M.MTheoryG2(O.SignInvolution(QUINTIC, [[3, 4]]))
    three = M.MTheory3D(CICY(SEXTIC4))

    # Five dimensions: eight supercharges forbid a superpotential. This is not
    # a missing computation and must not be reported as one.
    check_true("5d yukawa raises NoSuchTheory",
               _raises(FT.NoSuchTheory, five.holomorphic_yukawa))
    check_true("   and not NeedsMetric",
               not _raises(NeedsMetric, five.holomorphic_yukawa))

    # G_2: no charged matter to couple, again a theorem.
    check_true("G_2 yukawa raises NoChiralMatter",
               _raises(M.NoChiralMatter, g2.holomorphic_yukawa))

    # Three dimensions: the superpotential exists, from M5-branes on divisors,
    # and is genuinely unavailable here. So this one IS a NeedsMetric.
    check_true("3d yukawa raises NeedsMetric",
               _raises(NeedsMetric, three.holomorphic_yukawa))
    check_true("   and not NoSuchTheory",
               not _raises(FT.NoSuchTheory, three.holomorphic_yukawa))

    # The physical coupling is unavailable everywhere, always with a list of
    # what would be needed rather than a placeholder number.
    for name, th in [("5d", five), ("3d", three)]:
        try:
            th.physical_yukawa()
            check_true("%s physical_yukawa refuses" % name, False)
        except NeedsMetric as e:
            check_true("%s physical_yukawa refuses with a list" % name,
                       len(e.missing) >= 3)

    # The G_2 case has its own list, and it has to mention the thing that
    # makes it harder than the Calabi-Yau case: there is no Yau theorem.
    missing = " ".join(g2.missing_for_physical()).lower()
    check_true("G_2 missing list names the singularities",
               "singular" in missing)
    check_true("   and that G_2 metrics have no Yau theorem",
               "yau" in missing)

    # Registration, so `get` finds them.
    for key in ["m-theory-cy3-5d", "m-theory-g2", "m-theory-cy4-3d"]:
        check_true("%s is registered" % key, T.get(key) is not None)

    # describe() must not raise on any of them; it is the reporting path and
    # it calls everything.
    for name, th in [("5d", five), ("G_2", g2)]:
        try:
            th.describe()
            check_true("%s describe() runs" % name, True)
        except Exception as e:                                   # noqa: BLE001
            check_true("%s describe() runs (%s)" % (name, e), False)


# ---------------------------------------------------------------------------
# [9] Horava-Witten
# ---------------------------------------------------------------------------

def test_hw():
    banner("[9] Horava-Witten: where the eleventh dimension sits")

    hw = M.horava_witten_scales()
    check_true("M_11 is above the GUT scale", hw["m11"] > hw["m_gut"])
    check_true("1/rho is below the GUT scale", hw["inverse_rho"] < hw["m_gut"])
    check_true("   so the eleventh dimension is the largest length",
               hw["eleventh_dimension_larger"])
    check("the ordering", hw["ordering"], "1/rho < M_GUT < M_11")

    # The physical statement is the ordering, and it must survive the O(1)
    # factors that differ between conventions. Perturbing both relations by up
    # to a factor of a few, in the form of moving alpha_GUT and M_GUT over
    # their plausible ranges, must not disturb it.
    ok = True
    for alpha in [1.0 / 20, 1.0 / 25, 1.0 / 30]:
        for mg in [1.0e16, 2.0e16, 3.0e16]:
            h = M.horava_witten_scales(alpha_gut=alpha, m_gut=mg)
            ok = ok and h["eleventh_dimension_larger"]
    check_true("ordering survives the plausible parameter range", ok)

    # And it is a genuine consequence rather than an identity: it has to be
    # breakable, or the test above would be checking nothing. Pushing the
    # compactification scale UP towards the Planck scale breaks it -- the
    # interval shrinks below the Calabi-Yau -- which is the right direction,
    # since the whole point is that the observed hierarchy between M_GUT and
    # the Planck scale is what opens the interval up.
    check_true("   but is not true by construction",
               not M.horava_witten_scales(m_gut=1.0e18)[
                   "eleventh_dimension_larger"])
    check_true("   and the crossover is above the GUT scale, not below",
               M.horava_witten_scales(m_gut=1.0e13)[
                   "eleventh_dimension_larger"])

    # Newton's constant is an input and must come back out. Inverting the
    # second matching relation is the check that the solve is consistent.
    import math
    V = hw["m_gut"] ** -6.0
    rho = 1.0 / hw["inverse_rho"]
    g_back = hw["kappa_squared"] / (16.0 * math.pi ** 2 * rho * V)
    check_true("G_N is recovered from the solution",
               abs(g_back / M.G_NEWTON - 1.0) < 1e-9)


def main():
    t0 = time.time()
    test_prepotential()
    test_fivedim()
    test_g2betti()
    test_g2refuses()
    test_g2spectrum()
    test_duality()
    test_threedim()
    test_categories()
    test_hw()

    print("\n" + "=" * 72)
    if FAILURES:
        print("FAILED (%d): %s" % (len(FAILURES), ", ".join(FAILURES)))
        return 1
    print("test_mtheory: all checks passed in %.1fs" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
