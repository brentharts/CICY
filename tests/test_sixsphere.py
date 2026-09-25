"""
Tests for pyCICY.sixsphere (Part I: the lattice layer) and
pyCICY.theories.hopf.

  [1] monodromy     orders, determinants, the unipotent cusp, A1 A2 M0 = I
  [2] lattices      fixed, invariant and torsion sublattices, coinvariants,
                    the invariant closure of Lambda_tor
  [3] forms         Q0 and eta solved for separately; eta = *Q0; b = diag(6,-1)
  [4] twists        freeness derived and checked against the paper's rule;
                    |p| against the Seifert H_1, by Smith form
  [5] controls      each check made to fail on purpose: a check that cannot
                    fail is not a test
  [6] registry      the theory is registered and every physics verb raises
                    NoSuchTheory, never NeedsMetric

Part II (Section 3: the period laws):

  [7] laws          closure under g1^3 = g2^4 = 1 and the cusp law; the
                    laws re-derived from T1, T2, T0 alone; the automorphy
                    factor; values at the elliptic points; local sections
  [8] real layer    det_R Pi = Im tau D and invariance of D; the Hodge form
                    on F^1, Lagrangian, signature (1, 1)
  [9] kodaira       the surface behind tau: IV*, III, I_1 by ftheory on the
                    Weierstrass orders, and independently by the lattice
                    subquotient plus Noether's formula
  [10] controls     a wrong law breaks closure; a wrong matrix entry breaks
                    the 6 mu shape; a quadratic twist keeps j and changes a
                    fibre; without Noether four combinations survive

Part III (Section 4: the toric filling at the cusp):

  [11] fan          unimodular cones, charts with t = z0 z1 z2, the volume
                    form, pi_1 of the toric model
  [12] E0           the star of a vertex is dP6: by pyCICY.toric, and from
                    the fan (six (-1)-curves, Noether)
  [13] W            the quotient: one component, opposite sides glued, two
                    triple points on every double curve, e(W) = 2 by strata
                    and from the normalisation, Friedman's triple-point
                    formula
  [14] d            |det B0| = d for d = 1..5: d components, e(W) = 2d both
                    ways, the dual complex a torus
  [15] controls     a scaled cone is not unimodular; on the P^2 fan the
                    triple-point formula fails; W sees only the lattice B0 Z^2

Part IV (Sections 7 and 9: topology and invariants):

  [16] topology     |p| three ways including the full presentation of
                    Theorem 7.17; b(W) and H_*(M) from exterior powers of the
                    cusp monodromy; the multiple fibres (Lemma 7.13, Prop 7.14)
  [17] ledger       HRR from Chern roots, chi(O) twice, the Hodge numbers up
                    to one parameter, Frolicher, HKP; K_X in Pic; the Hodge
                    bundle degree; the C^* fixed locus and its weights
  [18] controls     flipping one sign of Lemma 7.16 gives Z/7; an even l2
                    gives torsion in H_1(S_2); HRR on P^3 gives chi(T) = 15;
                    multiplicities (3, 3) would put torsion in Pic; the
                    cocharacter e1 fixes a different curve

Part V (the finite layer in Lean 4, pyCICY.sixsphere_lean):

  [19] python       every fact computed; the Lean data generated from the
                    same objects; the invariance hypotheses recomputed
  [20] kernel       if a lean is available (PYCICY_LEAN or PATH): the file
                    compiles, every fact machine-checked on standard axioms,
                    and five controls -- a false fact, a sorry, a
                    native_decide, a declared axiom, a tampered matrix --
                    are all refused.
                    Without a kernel: nothing is claimed as checked.

Run with:  python3 tests/test_sixsphere.py
       or: python3 run_tests.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sympy import Matrix, eye, symbols, cancel

from pyCICY import sixsphere as S
from pyCICY import theories as T

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


def _raises(exc, fn, *a, **kw):
    try:
        fn(*a, **kw)
    except exc:
        return True
    except Exception:
        return False
    return False


def test_monodromy():
    print("\n[1] monodromy")
    o = S.orders()
    check("det T1", o["det_T1"], 1)
    check("det T2", o["det_T2"], 1)
    check("order T1", o["order_T1"], 3)
    check("order T2", o["order_T2"], 4)
    check("T0 has infinite order", o["order_T0"], None)
    check_true("N != 0 and N^2 = 0", o["N_nonzero"] and o["N_squared_zero"])
    check_true("ker N = im N", o["ker_N_equals_im_N"])
    check_true("T1 T2 T0 = I", o["T1T2T0_is_I"])
    check_true("A1 A2 M0 = I", o["A1A2M0_is_I"])
    check_true("gamma o A = gamma for A1, A2, M0", o["gamma_invariant"])
    m = S.monodromy()
    # the paper prints T0 and M0 explicitly: compare entry by entry
    check("T0 as printed (Lemma 2.2)", m["T0"],
          Matrix([[1, 0, 0, 1], [0, 1, -1, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))
    check("A1 as printed (Lemma 2.4)", m["A1"],
          Matrix([[1, 0, 0, 0], [6, 0, 1, 0], [-6, -1, -1, 0],
                  [-2, 1, 0, 1]]))
    check("A2 as printed (Lemma 2.4)", m["A2"],
          Matrix([[1, 0, 0, 0], [0, 0, -1, 0], [-6, 1, 0, 0],
                  [3, 0, 1, 1]]))
    # the same matrices appear in the Lean formalisation (Solution.lean);
    # M0 there, row-major:
    check("M0 matches the Lean formalisation", m["M0"],
          Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 1, 1, 0], [-1, 0, 0, 1]]))


def test_lattices():
    print("\n[2] lattices")
    fl = S.fixed_lattices()
    check_true("Lambda^A1 = <eps, delta^>", fl["Lambda_A1_is_<eps,delta^>"])
    check_true("Lambda^A2 = <eps', delta^>", fl["Lambda_A2_is_<eps',delta^>"])
    check("gamma(eps)", fl["gamma(eps)"], 1)
    check("gamma(eps')", fl["gamma(eps')"], 1)
    iv = S.invariant_vectors()
    check_true("V^G = Z gamma", iv["V^G_is_Z_gamma"])
    check_true("Lambda^G = Z delta^", iv["Lambda^G_is_Z_delta^"])
    for side in ("V", "Lambda"):
        c = S.coinvariants(side)
        check("coinvariants %s: invariant factors" % side,
              c["invariant_factors"], [1, 1, 1, 0])
        check("coinvariants %s: free rank" % side, c["free_rank"], 1)
    tl = S.torsion_lattice()
    check_true("Lambda_tor = ker(M0 - I)", tl["ker_is_<w^,delta^>"])
    check_true("Lambda_tor = im(M0 - I)", tl["im_is_<w^,delta^>"])
    check("B0", S.B0(), Matrix([[0, 1], [-1, 0]]))
    check("det B0", S.B0().det(), 1)
    check_true("invariant closure of Lambda_tor = ker gamma",
               S.invariant_closure()["equals_ker_gamma"])


def test_forms():
    print("\n[3] forms")
    F = S.invariant_forms()
    check("number of invariant alternating forms", len(F), 1)
    check_true("generator is +-Q0", F[0] in (S.Q0, -S.Q0))
    B = S.invariant_bivectors()
    check("number of invariant bivectors", len(B), 1)
    check_true("generator is +-eta", B[0] in (S.ETA, -S.ETA))
    check("eta ^ eta / vol", S.wedge_square(S.ETA), 12)
    check_true("eta = *Q0 (two solves, one answer)",
               S.hodge_star(S.Q0) in (S.ETA, -S.ETA))
    b = S.b_form()
    check("b Gram on (w, delta)", b["gram"], Matrix([[6, 0], [0, -1]]))
    check("b signature", b["signature"], (1, 1))
    check("b discriminant", b["discriminant"], -6)
    check_true("b symmetric and vanishing on ker N",
               b["symmetric_on_V"] and b["vanishes_on_ker_N"])
    check("polarisable", b["polarisable"], False)


def test_twists():
    print("\n[4] twists")
    check("(l0, l1, l2)", S.twist_integers(), (0, 1, -1))
    check_true("log transform at p1 free (v1 = eps)",
               S.log_transform_free(1, S.EPS)[0])
    check_true("log transform at p2 free (v2 = -eps')",
               S.log_transform_free(2, -S.EPS_PRIME)[0])
    rule = True
    for a in range(-6, 7):
        for d in (-2, 0, 1, 5):
            f1 = S.log_transform_free(1, a * S.EPS + d * S.DELTA_HAT)[0]
            f2 = S.log_transform_free(2, a * S.EPS_PRIME + d * S.DELTA_HAT)[0]
            rule &= (f1 == (a % 3 != 0)) and (f2 == (a % 2 != 0))
    check_true("freeness rule derived: 3 !| l1, l2 odd (52 twists each)",
               rule)
    # which power fails is informative: an even l2 is caught at k = 2
    per = dict(S.log_transform_free(2, 2 * S.EPS_PRIME)[1])
    check("l2 = 2: fixed point appears at k = 2 only",
          (per[1], per[2], per[3]), (False, True, False))
    check_true("non-fixed twist vector rejected",
               _raises(ValueError, S.log_transform_free, 1, S.EPS_PRIME))
    check("|p| for the paper's twists", S.pi1_order(0, 1, -1), 1)
    check("|p| for X' (v2 = +eps')", S.pi1_order(*S.twist_integers(
        v2=S.EPS_PRIME)), 7)
    sa = S.seifert_agreement(box=4)
    check("closed form vs Seifert H_1: disagreements over 729",
          len(sa["disagreements"]), 0)
    check("S^3 case: |H_1| of Seifert (3,4) with (0,1,-1)",
          sa["S3_case"]["order"], 1)


def test_controls():
    print("\n[5] negative controls")
    # a sign flip in the closed form must be caught by the Smith-form route
    wrong = [(l0, l1, l2) for l0 in range(-3, 4) for l1 in range(-3, 4)
             for l2 in range(-3, 4)
             if abs(12 * l0 + 4 * l1 - 3 * l2)
             != S.seifert_h1(3, 4, -l1, -l2, l0)["order"]]
    check_true("sign-flipped formula disagrees somewhere", len(wrong) > 0)
    # lattice equality is not vacuous
    check_true("<eps, delta^> != <2 eps, delta^>",
               not S.same_lattice(Matrix([list(S.EPS), list(S.DELTA_HAT)]),
                                  Matrix([list(2 * S.EPS),
                                          list(S.DELTA_HAT)])))
    # saturation is not the identity
    check("saturation of <2 e1> is <e1>", S.saturate([[2, 0, 0, 0]]),
          Matrix([[1, 0, 0, 0]]))
    # a perturbed T1 loses order 3, and with it the free product relation
    Tbad = S.T1.copy()
    Tbad[0, 2] = -5
    check("perturbed T1 has no finite order", S._order(Tbad), None)
    # Q0 with Q0(u, w) = 5 instead of 6 is no longer invariant: the solve
    # for invariant forms is what pins the 6, not the paper's say-so
    Qbad = S._antisym([0, 0, 1, 5, 0, 0])
    check_true("Q0 with (u,w) = 5 is not T1-invariant",
               S.T1.T * Qbad * S.T1 != Qbad)
    # Hodge star is not trivially +-identity on the forms
    check_true("*Q0 != +-Q0", S.hodge_star(S.Q0) not in (S.Q0, -S.Q0))


def test_registry():
    print("\n[6] registry")
    check_true("registered", "complex-six-sphere" in T.registry)
    X = T.get("complex-six-sphere")()
    check("X is None", X.X, None)
    for verb in ("spectrum", "holomorphic_yukawa", "physical_yukawa",
                 "fermion_masses"):
        check_true("%s raises NoSuchTheory" % verb,
                   _raises(T.NoSuchTheory, getattr(X, verb)))
        try:
            getattr(X, verb)()
        except T.NeedsMetric:
            FAILURES.append("%s blamed the metric" % verb)
        except Exception:
            pass
    check_true("describe lists exact and declined",
               "exact:" in X.describe() and "declined:" in X.describe())
    check_true("lattice() passes", X.lattice()["all_pass"])


def test_laws():
    print("\n[7] period laws")
    cl = S.closure()
    check_true("g1^3 acts trivially on (tau, mu, beta)", cl["g1^3_trivial"])
    check_true("g2^4 acts trivially on (tau, mu, beta)", cl["g2^4_trivial"])
    check_true("g1 g2 acts by (tau+1, mu, beta-1)", cl["g0_law"])
    check("cocycle sums (paper's route)", S.cocycle_sums(),
          {"g1": 0, "g2": 0})
    la = S.lattice_agreement()
    for g in ("g1", "g2", "g0"):
        check_true("%s: law forced by the integer matrix" % g,
                   la[g]["law_ok"])
        check_true("%s: Z(gz) keeps the 6 mu shape" % g, la[g]["shape_ok"])
        check_true("%s: R_g as printed" % g, la[g]["R_ok"])
    jt = S.jtilde_cocycle()
    check("jtilde over the g1 orbit", jt["g1_orbit"], 1)
    check("jtilde over the g2 orbit", jt["g2_orbit"], 1)
    check("jtilde(g0)", jt["g0"], 1)
    check_true("det R_g = 1/jtilde(g)", jt["det_R_is_1/jtilde"])
    ep = S.elliptic_points()
    for k in ("tau(z1)_is_rho", "tau(z2)_is_i", "mu(z1)=(2-rho)/3",
              "mu(z2)=(1-i)/2", "6(1-mu)^2 = 2 tau at z1",
              "6 mu^2 = -3 tau at z2"):
        check_true(k, ep[k])
    for k, v in S.local_sections().items():
        check_true("local section: " + k, v)


def test_real_layer():
    print("\n[8] real layer")
    for k, v in S.lattice_condition().items():
        check_true(k, v)
    h = S.hodge_form()
    check_true("F^1 is Q0-Lagrangian", h["lagrangian"])
    check_true("Hodge Gram = -[[12 Im tau, 12 Im mu], [., 2 Im beta]]",
               h["gram_as_printed"])
    check_true("det Gram = 24 Im tau D", h["det = 24 Im tau D"])
    # D < 0 and Im tau > 0 give det < 0: one positive, one negative
    # eigenvalue, whatever c0 is -- the analytic face of b = diag(6, -1)


def test_kodaira():
    print("\n[9] kodaira")
    for k, v in S.kodaira_table_check().items():
        check_true("table entry %s agrees with ftheory" % k, v)
    w = S.weierstrass_route()
    check_true("j = 1728 t", w["j_is_1728t"])
    check("fibre at t = 0", w["fibres"][0]["type"], "IV*")
    check("fibre at t = 1", w["fibres"][1]["type"], "III")
    check("fibre at t = oo", w["fibres"]["oo"]["type"], "I_1")
    check("Euler numbers sum", w["euler_sum"], 12)
    l = S.lattice_route()
    check_true("subquotient is multiplicative", l["homomorphism"])
    check_true("subquotient reproduces the tau laws (third route)",
               all(l["tau_laws"].values()))
    check("candidates at t = 0", sorted(l["candidates"][0]), ["IV", "IV*"])
    check("candidates at t = 1", sorted(l["candidates"][1]), ["III", "III*"])
    check("candidates at t = oo", l["candidates"]["oo"], ["I_1"])
    check("Noether selects", l["noether_selects"], [("IV*", "III", "I_1")])
    check_true("the two routes agree", S.kodaira_crosscheck()["agree"])


def test_controls_periods():
    print("\n[10] Part II controls")
    tau, mu, beta = S.TAU, S.MU, S.BETA
    bad = dict(S.LAWS)
    bad["g1"] = (bad["g1"][0], bad["g1"][1],
                 beta + 3 - 6 * (1 - mu) ** 2 / tau)       # +3 for +2
    check_true("beta law with +3 no longer closes under g1^3",
               S.law(["g1"] * 3, bad) != (tau, mu, beta))
    T = S.T1.copy()
    T[0, 2] = -5
    check("T1 with -5 for -6 breaks the 6 mu shape",
          S.laws_from_lattice(T)["shape_ok"], False)
    t = symbols("t")
    g2, g3 = S.WEIERSTRASS
    g2t, g3t = g2 * (t - 1) ** 2, g3 * (t - 1) ** 3
    check("quadratic twist by (t-1) keeps j",
          cancel(1728 * g2t ** 3 / (g2t ** 3 - 27 * g3t ** 2)), 1728 * t)
    from pyCICY.theories.ftheory import kodaira_type
    f, g = -g2t / 4, -g3t / 4
    Dt = 4 * f ** 3 + 27 * g ** 2
    check("...and turns the fibre at t = 1 into III*",
          kodaira_type(S._ord_at(f, 1, 6), S._ord_at(g, 1, 9),
                       S._ord_at(Dt, 1, 18))["type"], "III*")
    check("without Noether, combinations surviving the monodromy",
          len(S.lattice_route()["combinations"]), 4)
    check_true("status_periods passes", S.status_periods()["all_pass"])


def test_fan():
    print("\n[11] fan")
    fc = S.fan_checks(radius=3)
    check("cone determinants", fc["dets"], {"lower": {1}, "upper": {-1}})
    check_true("t = z0 z1 z2 in every chart",
               fc["t = z0 z1 z2 in every chart"])
    check_true("dx1 dx2 dt/(x1 x2) = +- dz0 dz1 dz2 (K_N0 trivial)",
               fc["Omega nowhere vanishing"])
    check("pi_1(Y_F): invariant factors of the ray matrix",
          S.toric_pi1()["invariant_factors"], [1, 1, 1])


def test_E0():
    print("\n[12] E0")
    hs = S.hexagon_surface()
    check("rays around 0 (paper's order)", hs["rays"], S.HEXAGON_RAYS)
    # copied from Solution.lean, ToricComponent.hexagonRay
    lean = ((1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1))
    check_true("same order as the Lean formalisation", hs["rays"] == lean)
    check("consecutive determinants", hs["consecutive_dets"], [1] * 6)
    check("self-intersections", hs["self_intersections"], [-1] * 6)
    check("pyCICY.toric name (points blown up)", hs["toric_name"], "B3")
    check_true("toric: smooth", hs["toric_smooth"])
    check("toric: degree K^2", hs["toric_degree"], 6)
    check("e(E0) = number of rays", hs["euler"], 6)
    check_true("Noether K^2 + e = 12", hs["noether"])


def test_W():
    print("\n[13] W")
    c = S.cusp_fibre()
    check("components", c["components"], 1)
    check("double curves", c["double_curves"], 3)
    check("triple points", c["triple_points"], 2)
    check("glued hexagon sides", c["glued_sides"],
          [((-1, 0), (1, 0)), ((-1, 1), (1, -1)), ((0, -1), (0, 1))])
    k = c["corner_classes"]
    check("distinct triple points among six corners", len(set(k)), 2)
    check_true("corners alternate around the hexagon",
               all(k[i] != k[(i + 1) % 6] for i in range(6)))
    check_true("each double curve meets both triple points",
               all(len(p) == 2
                   for p in c["curves_through_triple_points"].values()))
    check("e(W) by torus-orbit strata", S.euler_W_stratified(), 2)
    en = S.euler_W_normalisation()
    check("e(W) = (6 - 6) + e(D)", (en["e(normalisation)"],
                                    en["e(hexagon cycles)"], en["e(D)"]),
          (6, 6, 2))
    check("three double curves through each triple point",
          en["curves_through_each_point"], [3])
    check("e(W) from the normalisation", en["e(W)"], 2)
    tp = S.triple_point_formula()
    check("triple-point formula (C|V)^2 + (C|V')^2 + T_C per curve",
          sorted(set(tp["per_curve"].values())), [0])


def test_det_scan():
    print("\n[14] |det B0| = d")
    for r in S.det_scan():
        d = r["d"]
        check("d=%d: components" % d, r["components"], d)
        check("d=%d: double curves, triple points" % d,
              (r["double_curves"], r["triple_points"]), (3 * d, 2 * d))
        check("d=%d: e(W) strata, normalisation" % d,
              (r["e_strata"], r["e_normalisation"]), (2 * d, 2 * d))
        check("d=%d: dual complex V - E + F" % d, r["dual_complex_euler"], 0)
        check_true("d=%d: triple-point formula" % d,
                   r["triple_point_formula"])


def test_controls_cusp():
    print("\n[15] Part III controls")
    check("a cone over a doubled triangle has det 4",
          S.cone_matrix(((0, 0), (2, 0), (0, 2))).det(), 4)
    p2 = ((1, 0), (0, 1), (-1, -1))
    check("P^2 fan: self-intersections +1", S.self_intersections(p2),
          [1, 1, 1])
    si = S.self_intersections(p2)
    check("...so (C|V)^2 + (C|V')^2 + T_C with +1 curves, T_C = 2",
          si[0] + si[1] + 2, 4)
    check_true("degenerate B0 rejected",
               _raises(ValueError, S.cusp_fibre, [[1, 2], [2, 4]]))
    cI = S.cusp_fibre(Matrix([[1, 0], [0, 1]]))
    c0 = S.cusp_fibre()
    check_true("W depends only on the lattice B0 Z^2, not on B0",
               (cI["components"], cI["double_curves"], cI["triple_points"],
                cI["glued_sides"])
               == (c0["components"], c0["double_curves"],
                   c0["triple_points"], c0["glued_sides"]))
    check_true("status_cusp passes", S.status_cusp()["all_pass"])


def test_topology():
    print("\n[16] topology (Section 7)")
    p3 = S.pi1_three_routes()
    check("|p|: closed form = Seifert = presentation, disagreements",
          len(p3["disagreements"]), 0)
    check_true("...over %d admissible twists and l0" % p3["checked"],
               p3["checked"] > 100)
    h = S.h1_from_presentation()
    check("H_1(X) from the six-generator presentation", h["order"], 1)
    ch = S.cusp_homology()
    check("b_q(W) = rank (wedge^q V)^T0", ch["b(W)"], [1, 2, 4, 2, 1])
    check("e(W) from the Betti numbers", ch["euler(W)"], 2)
    for q, ok in ch["invariants_as_printed"].items():
        check_true("(wedge^%d V)^T0 as printed in Prop. 7.12" % q, ok)
    for k in ch["specialisation_kernels"]:
        check_true("q=%d: ker of specialisation saturated, rank %d"
                   % (k["q"], k["rank"]),
                   k["saturated"] and k["rank=C(4,q)-b_q"])
    check("ranks of H_*(M), Wang", ch["ranks H_*(M)"], [1, 3, 6, 6, 3, 1])
    for j, want in ((1, (3, -9, 1, 3, 1)), (2, (4, -4, 2, 2, 2))):
        mf = S.multiple_fibre(j)
        check_true("p%d: (A_j - I) Lambda = ker gamma cap ker psi_j" % j,
                   mf["image_is_kerg_cap_kerpsi"] and mf["coinvariants_Z2"])
        check_true("p%d: H_1(S_j) torsion-free" % j, mf["H1(S)_torsion_free"])
        check("p%d: (k1, det2, k2, |det13|, k3)" % j,
              (mf["k1"], mf["det2"], mf["k2"], abs(mf["det13"]), mf["k3"]),
              want)


def test_ledger():
    print("\n[17] invariants ledger (Section 9)")
    from sympy import Symbol
    c1, c2, c3 = Symbol("c1"), Symbol("c2"), Symbol("c3")
    g = S.hrr_threefold()["general"]
    check("HRR chi(O)", g["O"], c1 * c2 / 24)
    check("HRR chi(T) (textbook: c1^3/2 - 19 c1c2/24 + c3/2)",
          g["T"], c1 ** 3 / 2 - 19 * c1 * c2 / 24 + c3 / 2)
    check_true("Serre: chi(Omega^3) = -chi(O), chi(Omega^2) = -chi(Omega^1)",
               g["Omega3"] == -g["O"] and g["Omega2"] == -g["Omega1"])
    onX = {k: v.subs(c3, 2) for k, v in S.hrr_threefold()["on_X"].items()}
    check("on X: chi(O, Om1, Om2, Om3, T)",
          [onX[k] for k in ("O", "Omega1", "Omega2", "Omega3", "T")],
          [0, -1, 1, 0, 1])
    le = S.leray_h0q()
    check("h^{0,q} from the direct images", le["h0q"], [1, 1, 0, 0])
    check("chi(O) from Leray (HRR gives 0)", le["chi(O)"], 0)
    hn = S.hodge_numbers()
    check("free parameters in the Hodge numbers", len(hn["free"]), 1)
    check("h11 - h12", hn["h11_minus_h12"], 1)
    check("sum (-1)^(p+q) h^{p,q}", hn["euler"], 2)
    check_true("h10 = 0 but h01 = 1: no Hodge symmetry",
               hn["table"][(1, 0)] == 0 and hn["table"][(0, 1)] == 1)
    check_true("Frolicher does not degenerate at E1", hn["b1<h01"])
    for k, v in S.hkp_constraints().items():
        check_true("HKP: " + k, v)
    pa = S.picard_arithmetic()
    check("Pic subgroup <H,S1,S2>: invariant factors", pa["invariant_factors"],
          [1, 1])
    check("K_X in units of u", pa["K"], -6)
    check("classical formula would give", pa["classical"], 5)
    check("2 S1 + S2 in units of u (pull-backs: 12Z)", pa["difference"], 11)
    check("normal forms of K_X", pa["normal_forms"], [(-1, 0, 2)])
    hb = S.hodge_bundle_degree()
    check("deg f_* omega via Euler numbers / 12", hb["via_euler"], 1)
    check("a_j", hb["a"], (2, 1))
    check("12 x local exponents = fibre Euler numbers",
          hb["12*exponents"], [8, 3, 1])
    cs = S.cstar_fixed_locus()
    check("C^*: fixed vertex strata", cs["fixed_vertex_strata"], 0)
    check("C^*: fixed edge strata (direction)",
          [e[0] for e in cs["fixed_edge_strata"]], [(0, 1)])
    check("C^*: fixed triple points", cs["fixed_triple_points"], 2)
    check("C^*: weights (normal, tangent, normal)",
          cs["weights_at_triple_points"], [(-1, 0, 1)])
    check("e(fixed locus) = e(X)", cs["euler_fixed"], 2)


def test_controls_invariants():
    print("\n[18] Part IV controls")
    check("Lemma 7.16 sign flipped at p2 only: |H_1| becomes",
          S.h1_from_presentation(S.EPS, S.EPS_PRIME)["order"], 7)
    check_true("l2 = 2 (not admissible): H_1(S_2) has torsion",
               not S.multiple_fibre(2, 2 * S.EPS_PRIME)["H1(S)_torsion_free"])
    from sympy import Symbol
    g = S.hrr_threefold()["general"]
    P3 = {Symbol("c1"): 4, Symbol("c2"): 6, Symbol("c3"): 4}
    check("HRR on P^3: chi(T) = dim PGL_4", g["T"].subs(P3), 15)
    check("HRR on P^3: chi(O), chi(Omega^1)",
          (g["O"].subs(P3), g["Omega1"].subs(P3)), (1, -1))
    from sympy.matrices.normalforms import invariant_factors
    check("multiplicities (3,3) would give torsion in <H,S1,S2>",
          [abs(int(x)) for x in invariant_factors(
              Matrix([[3, 0, -1], [0, 3, -1]]))], [1, 3])
    other = S.cstar_fixed_locus(cochar=(1, 0))
    check("cocharacter e1 fixes the e1-curve instead",
          [e[0] for e in other["fixed_edge_strata"]], [(1, 0)])
    check_true("status_invariants passes", S.status_invariants()["all_pass"])


def test_lean_python():
    print("\n[19] Lean facts: the Python half")
    from pyCICY import sixsphere_lean as L
    py = L.check_python()
    check("number of facts", len(py), 34)
    check_true("every fact computed True", all(v is True for v in py.values()))
    src = L.lean_source()
    check_true("every fact is in the emitted source",
               all(("theorem %s" % f.name) in src for f in L.FACTS))
    check_true("every fact has an axiom query",
               all(("#print axioms SixSphere.%s" % f.name) in src
                   for f in L.FACTS))
    check_true("the Lean T1 is generated from sixsphere.T1",
               ("def T1 : Mat := %s" % L.lean_mat(S.T1)) in src)
    import re
    code = re.sub(r"/-.*?-/", "", src, flags=re.S)       # drop block comments
    code = re.sub(r"--[^\n]*", "", code)                 # and line comments
    check_true("no sorry, native_decide or axiom in the code",
               not re.search(r"\b(sorry|native_decide|axiom)\b", code))


def test_lean_kernel():
    print("\n[20] Lean facts: the kernel")
    from pyCICY import sixsphere_lean as L
    from pyCICY.theories import nariai_lean as NL
    if NL.lean_executable() is None:
        st = L.status()
        check("toolchain", st["toolchain"], False)
        check_true("without a kernel nothing is claimed machine-checked",
                   not any(v["machine_checked"] for v in st["facts"].values()))
        print("  (no lean found: kernel checks and controls skipped, "
              "not passed)")
        return
    st = L.status()
    check("the kernel accepts the file", st["lean_ok"], True)
    check_true("every fact machine-checked", st["all_machine_checked"])
    check_true("standard axioms only (propext, Quot.sound, Classical.choice)",
               all(v["standard_axioms_only"] for v in st["facts"].values()))
    check("choice-free facts", st["n_choice_free"], 29)
    uses_choice = sorted(k for k, v in st["facts"].items()
                         if not v["choice_free"])
    check("facts using Classical.choice (omega/grind, quantified)",
          uses_choice, sorted(["invariant_forms_unique",
                               "invariant_bivectors_unique", "p_coprime_12",
                               "fan_unimodular", "K_normal_form_unique"]))
    base = L.lean_source()
    tail = "\n\nend SixSphere"

    def with_extra(thm, name):
        return (base.replace(tail, "\n\n" + thm + tail, 1)
                + "#print axioms SixSphere.%s\n" % name)

    r = L.check_lean(source=with_extra(
        "theorem bogus : mpow T1 2 = I4 := by decide", "bogus"))
    check("control: a false fact (T1^2 = I) fails to compile", r["ok"], False)
    r = L.check_lean(source=with_extra(
        "theorem cheat : mpow T1 2 = I4 := by sorry", "cheat"))
    check_true("control: a sorry is caught by the axiom report",
               r["ok"] is False and "cheat" in r["unsound"])
    r = L.check_lean(source=with_extra(
        "theorem trusted : mpow T1 3 = I4 := by native_decide", "trusted"))
    check_true("control: native_decide (its auxiliary axiom) is refused",
               r["ok"] is False and "trusted" in r["unsound"])
    r = L.check_lean(source=with_extra(
        "axiom cheatAx : mpow T1 2 = I4\ntheorem declared : mpow T1 2 = I4 := "
        "cheatAx", "declared"))
    check_true("control: a user-declared axiom is refused",
               r["ok"] is False and "declared" in r["unsound"])
    tampered = base.replace("def T1 : Mat := %s" % L.lean_mat(S.T1),
                            "def T1 : Mat := [[1, 0, -5, 2], [0, -1, 1, 1], "
                            "[0, -1, 0, 1], [0, 0, 0, 1]]", 1)
    r = L.check_lean(source=tampered)
    check_true("control: T1 with -5 for -6 breaks the kernel check",
               r["ok"] is False and len(r["errors"]) > 0)


def main():
    test_monodromy()
    test_lattices()
    test_forms()
    test_twists()
    test_controls()
    test_registry()
    test_laws()
    test_real_layer()
    test_kodaira()
    test_controls_periods()
    test_fan()
    test_E0()
    test_W()
    test_det_scan()
    test_controls_cusp()
    test_topology()
    test_ledger()
    test_controls_invariants()
    test_lean_python()
    test_lean_kernel()
    print()
    if FAILURES:
        print("FAILED: %d" % len(FAILURES))
        for f in FAILURES:
            print("  - " + f)
        return 1
    print("all sixsphere tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
