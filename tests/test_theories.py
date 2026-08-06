"""
Tests for pyCICY.theories.

The subpackage exists to keep one distinction sharp, so the tests are organised
around it:

  [1] interface     the registry, and that the base class refuses physical
                    quantities rather than approximating them
  [2] exact         the standard-embedding Yukawa coupling, which is a triple
                    intersection number classically and an integer series with
                    instanton corrections -- no metric anywhere
  [3] convergence   the instanton sum only converges for small q, and the
                    diagnostic must say so rather than returning a partial sum
                    that looks like an answer
  [4] not exact     the line bundle model, where the holomorphic coupling is
                    unimplemented (a missing feature) and the physical one is
                    obstructed (a different thing), and the two must not be
                    conflated

Run with:  python3 tests/test_theories.py
       or: python3 run_tests.py
"""

import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyCICY import CICY
from pyCICY import bundles as BU
from pyCICY import equivariant as E
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
    except Exception:                                            # noqa: BLE001
        return False
    return False


QUINTIC = [[4, 5]]
TETRA = [[1, 2], [1, 2], [1, 2], [1, 2]]
M7833 = [[2, 2, 1], [3, 1, 3]]
MODEL = [[-2, -2, -1, 2], [-2, 1, 0, 0], [1, -2, 1, 0],
         [1, 1, -1, 0], [2, 2, 1, -2]]


def test_interface():
    print("\n[1] the interface")
    check("two theories registered", sorted(T.registry),
          ["heterotic-line-bundle", "heterotic-standard-embedding"])
    check_true("get() finds one",
               T.get("heterotic-standard-embedding") is T.StandardEmbedding)
    check_true("and refuses an unknown key",
               _raises(KeyError, T.get, "type-iia"))

    se = T.StandardEmbedding(QUINTIC)
    check("gauge group", se.gauge_group(), "E_6")

    # Physical quantities raise. An exception rather than a placeholder, so
    # that nothing can be tabulated next to a measured constant by accident.
    check_true("physical_yukawa raises", _raises(T.NeedsMetric,
                                                 se.physical_yukawa))
    check_true("fermion_masses raises", _raises(T.NeedsMetric,
                                                se.fermion_masses))

    # The exception carries what would be needed, so a caller can report the
    # gap rather than discovering it.
    try:
        se.physical_yukawa()
    except T.NeedsMetric as e:
        check_true("and names the missing ingredients", len(e.missing) >= 4)
        check_true("including the metric",
                   any("Ricci-flat" in m for m in e.missing))
        check_true("and harmonic representatives",
                   any("harmonic" in m for m in e.missing))
    try:
        se.fermion_masses()
    except T.NeedsMetric as e:
        check_true("masses additionally need moduli stabilisation",
                   any("stabilis" in m for m in e.missing))

    check_true("describe() mentions what is not computable",
               "not computable here" in se.describe())


def test_exact_yukawa():
    print("\n[2] the standard-embedding Yukawa, exactly")
    se = T.StandardEmbedding(QUINTIC)
    X = CICY(QUINTIC)

    s = se.spectrum()
    check("27 families = h^{2,1}", s["27"], int(X.h[1]))
    check("27-bar families = h^{1,1}", s["27bar"], int(X.h[2]))
    check("net generations = -chi/2",
          s["generations"], -X.euler_characteristic() // 2)

    # The classical coupling is the triple intersection number. On the quintic
    # that is the integer 5, computed from the configuration matrix with no
    # metric and no approximation.
    y = se.holomorphic_yukawa()
    check("quintic classical Yukawa", float(y["classical"].ravel()[0]), 5.0)
    check_true("which is exactly d_rst",
               np.allclose(y["classical"],
                           np.asarray(X.triple_intersection())))

    # And on a multi-parameter manifold it is the whole tensor.
    tq = T.StandardEmbedding(TETRA)
    yt = tq.holomorphic_yukawa()
    check("tetraquadric classical Yukawa is 4x4x4", yt["classical"].shape,
          (4, 4, 4))
    check_true("with entries 0 and 2 only",
               set(np.unique(yt["classical"]).tolist()) == {0.0, 2.0})

    # Instanton corrections, from the genus-zero invariants. Integers.
    r = se.holomorphic_yukawa(q=1e-5, max_degree=5)
    check("degree-1 invariant is 2875", r["invariants"][1], 2875)
    check("degree-2 invariant is 609250", r["invariants"][2], 609250)
    check("degree-3 invariant is 317206375", r["invariants"][3], 317206375)
    check_true("the corrected coupling exceeds the classical one",
               r["quantum"] > 5.0)

    # As q -> 0 the instantons switch off and the classical value is recovered.
    tail = [se.holomorphic_yukawa(q=q, max_degree=5)["quantum"]
            for q in (1e-4, 1e-5, 1e-6)]
    check_true("q -> 0 recovers d_111 = 5 (%.6f -> %.6f)" % (tail[0], tail[-1]),
               abs(tail[-1] - 5.0) < abs(tail[0] - 5.0) < 1.0)
    check_true("and monotonically", tail[0] > tail[1] > tail[2] > 5.0)

    # Multi-parameter instanton corrections are not implemented, and say so
    # rather than silently returning the classical answer as if corrected.
    check_true("instantons refused for h^{1,1} > 1",
               _raises(NotImplementedError, tq.holomorphic_yukawa, q=1e-5))


def test_convergence():
    print("\n[3] the instanton sum only converges for small q")
    se = T.StandardEmbedding(QUINTIC)

    # The invariants grow faster than q^d shrinks unless q is genuinely small.
    # At q = 0.01 the degree-five term alone is about 3e6 and the partial sum
    # is meaningless; the flag has to say so, or a caller reads 3030588 as a
    # Yukawa coupling.
    bad = se.holomorphic_yukawa(q=0.01, max_degree=5)
    check_true("q = 0.01 is flagged as not converging", not bad["converging"])
    check_true("because the last term dominates (%.2g vs %.2g)"
               % (bad["last_term"], bad["quantum"]),
               bad["last_term"] > 0.5 * abs(bad["quantum"]))
    check_true("and the summary says so", "NOT converging" in bad["summary"])

    good = se.holomorphic_yukawa(q=1e-4, max_degree=5)
    check_true("q = 1e-4 converges", good["converging"])
    check_true("with a negligible last term (%.2g)" % good["last_term"],
               good["last_term"] < 1e-3 * abs(good["quantum"]))
    check_true("and the summary agrees", "converging" in good["summary"])

    # Term-by-term output, so the truncation is inspectable rather than a
    # single opaque number.
    check("terms are reported per degree", sorted(good["terms"]),
          [1, 2, 3, 4, 5])
    check_true("and decrease with degree",
               all(good["terms"][d] >= good["terms"][d + 1]
                   for d in range(1, 5)))


def test_line_bundle_model():
    print("\n[4] the line bundle model, where it is not exact")
    A = E.TETRAQUADRIC_Z2()
    m = T.LineBundleModel(TETRA, MODEL, action=A, wilson=(0, 1))

    check_true("the Wilson line is reflected in the gauge group",
               "SU(3) x SU(2) x U(1)" in m.gauge_group())
    s = m.spectrum()
    check("three generations", s["generations"], 3)
    check("anomaly free", int(s["anomaly"]), 0)

    # Without the group action, the spectrum is on the cover and says so.
    m0 = T.LineBundleModel(TETRA, MODEL)
    check("index on the cover", m0.spectrum()["index"], -6)
    check_true("and it is labelled as such", "cover" in m0.spectrum()["note"])

    # The two kinds of unavailability must not be conflated. The holomorphic
    # coupling is a missing feature -- it is quasi-topological and would be
    # exact given cohomology representatives. The physical one is obstructed
    # by the absence of a metric. Different exceptions, different reasons.
    check_true("holomorphic coupling is NotImplementedError",
               _raises(NotImplementedError, m.holomorphic_yukawa))
    check_true("physical coupling is NeedsMetric",
               _raises(T.NeedsMetric, m.physical_yukawa))
    try:
        m.holomorphic_yukawa()
    except NotImplementedError as e:
        msg = str(e)
        check_true("and the reason is representatives, not the metric",
                   "representatives" in msg and "quasi-topological" in msg)
    check_true("its missing list leads with representatives",
               "representatives" in m.missing_for_physical()[0])

    # The standard embedding does not need them, which is the contrast.
    se = T.StandardEmbedding(TETRA)
    check_true("the standard embedding needs no representatives",
               all("representatives" not in x
                   for x in se.missing_for_physical()[:1]))


def test_yukawa_texture():
    print("\n[5] Yukawa selection rules and texture")
    from pyCICY.theories import yukawa as Y
    from pyCICY import bundles as BU

    check("up-type patterns, one per pair", len(Y.up_type_patterns()), 10)
    check("down-type patterns = 5 x 3, the epsilon structure",
          len(Y.down_type_patterns()), 15)
    check_true("down-type is an SU(5) statement",
               _raises(ValueError, Y.down_type_patterns, 4))

    # Every down-type pattern uses all five indices exactly once, which is
    # what makes the charges cancel when sum_a L_a = 0.
    bad = 0
    for a, bc, de in Y.down_type_patterns():
        if sorted([a] + list(bc) + list(de)) != [0, 1, 2, 3, 4]:
            bad += 1
    check("every down pattern uses all five indices", bad, 0)

    t = Y.texture(TETRA, MODEL)
    check("all up patterns are charge-allowed",
          t["summary"]["up"]["charge_allowed"], 10)
    check("all down patterns are charge-allowed",
          t["summary"]["down"]["charge_allowed"], 15)

    # The dimensions used must reproduce the spectrum computed elsewhere, or
    # the texture is being read off the wrong cohomology groups. Three
    # independent totals, from su5_spectrum, which shares no code with this.
    X = CICY(TETRA)
    sp = BU.LineBundleSum(X, MODEL).su5_spectrum()
    h1 = lambda v: int(np.asarray(X.line_co(list(map(int, v))))[1])  # noqa: E731
    k = np.asarray(MODEL)
    import itertools as it
    check("sum of 10 dimensions = n10",
          sum(h1(k[a]) for a in range(5)), sp["n10"])
    check("sum of 5-bar dimensions = n5bar",
          sum(h1(k[a] + k[b]) for a, b in it.combinations(range(5), 2)),
          sp["n5bar"])
    check("sum of 5 dimensions = n5",
          sum(h1(-(k[a] + k[b])) for a, b in it.combinations(range(5), 2)),
          sp["n5"])

    # For this model every pattern is a texture zero: charge-allowed but with
    # a zero-dimensional group somewhere. That is a real and rather damning
    # statement about the model -- it has no holomorphic Yukawa couplings at
    # all -- and it is exact.
    check("no up-type coupling survives", t["summary"]["up"]["present"], 0)
    check("no down-type coupling survives", t["summary"]["down"]["present"], 0)
    check("so all 25 patterns are texture zeros",
          t["summary"]["up"]["texture_zeros"]
          + t["summary"]["down"]["texture_zeros"], 25)
    bad = sum(1 for r in t["down"] if min(r["dimensions"]) > 0)
    check("and each has a vanishing group", bad, 0)

    # Reached through the theory object as well.
    m = T.LineBundleModel(TETRA, MODEL, action=E.TETRAQUADRIC_Z2(),
                          wilson=(0, 1))
    check_true("LineBundleModel exposes the texture",
               m.yukawa_texture()["summary"] == t["summary"])

    # The texture says nothing about coupling *strengths*, and must not
    # pretend to: no record carries a value.
    check_true("no record contains a coupling value",
               all("value" not in r and "strength" not in r
                   for r in t["down"]))


def test_viable_triples():
    print("\n[6] can a manifold support a coupling at all?")
    from pyCICY.theories import yukawa as Y
    from pyCICY import bundles as BU

    # A necessary condition on the manifold, asked before any model is built.
    v = Y.viable_triples(TETRA, charge=2, limit=50)
    check_true("the tetraquadric admits viable triples at charge 2", len(v) > 0)

    # Every returned triple must satisfy both conditions it claims.
    X = CICY(TETRA)
    d = np.asarray(X.triple_intersection())
    bad_sum = bad_h1 = bad_slope = 0
    for a, b, c in v[:20]:
        if np.any(np.asarray(a) + np.asarray(b) + np.asarray(c)):
            bad_sum += 1
        for x in (a, b, c):
            if int(np.asarray(X.line_co(list(x)))[1]) == 0:
                bad_h1 += 1
            if BU.slope_is_definite(d, x):
                bad_slope += 1
    check("every triple has charges summing to zero", bad_sum, 0)
    check("every member has h^1 > 0", bad_h1, 0)
    check("and a non-definite slope", bad_slope, 0)

    # The slope filter is not decoration. Without it the first triple found on
    # the tetraquadric contains (0,0,-2,0), whose slope is one-signed, so no
    # bundle containing it can be poly-stable -- 9390 models built around that
    # pair contained exactly zero stable ones.
    check_true("(0,0,-2,0) has a definite slope",
               BU.slope_is_definite(d, [0, 0, -2, 0]))
    check_true("so it is excluded when require_slope is on",
               all([0, 0, -2, 0] not in t for t in v))

    # The answer depends on the box, and the docstring says so. CICY 7833 has
    # none at charge 4 and three at charge 5, which is a statement about the
    # box as much as the manifold.
    check("CICY 7833 has no viable triple at charge 4",
          len(Y.viable_triples(M7833, charge=4)), 0)
    check_true("but does at charge 5",
               len(Y.viable_triples(M7833, charge=5)) > 0)


def test_running():
    print("\n[7] from the spectrum to the gauge couplings")
    from pyCICY.theories import running as RUN
    from fractions import Fraction as Fr

    # Calibration: the formulas must reproduce the MSSM at Standard Model
    # content, or they are not the right formulas.
    check("MSSM one-loop coefficients", RUN.beta_coefficients(3, 1),
          RUN.MSSM_BETAS)
    check("b_3 = -9 + 2 n_g", RUN.beta_coefficients(3, 1)[2], Fr(-3))
    check("an extra Higgs pair moves b_2 but not b_3",
          (RUN.beta_coefficients(3, 2)[1], RUN.beta_coefficients(3, 2)[2]),
          (Fr(2), Fr(-3)))

    # The sharpest thing the topology says about hadrons: whether QCD confines
    # at all is decided by the generation count, which is an index.
    check("three generations confine", RUN.confines(3), True)
    check("four still do (b_3 = -1)", RUN.confines(4), True)
    check("five do not (b_3 = +1)", RUN.confines(5), False)
    check_true("and the scale is then refused, not invented",
               _raises(ValueError, RUN.lambda_qcd, n_generations=5))

    # Vector-like colour changes the running, which is why heterotic models
    # with extra matter cannot use the MSSM numbers.
    check("a vector-like 3+3bar shifts b_3 by one",
          RUN.beta_coefficients(3, 1, {"3+3bar": 1})[2], Fr(-2))

    # The sensitivity, which is the honest reason this is hard: the exact
    # integer sits in an exponent multiplied by an unknown.
    s = RUN.sensitivity()
    check_true("d ln(Lambda)/d ln(alpha) is about 52 (%.1f)"
               % s["dlnLambda_dlnalpha"],
               50 < s["dlnLambda_dlnalpha"] < 55)
    lo = RUN.lambda_qcd(alpha_gut=1 / 25.)["lambda_qcd"]
    hi = RUN.lambda_qcd(alpha_gut=1 / 18.)["lambda_qcd"]
    check_true("a 30%% change in alpha spans >6 orders of magnitude (%.0e)"
               % (hi / lo), hi / lo > 1e6)

    # Vector-like matter, now computable, and its consequence.
    check("a 10+10bar pair shifts every b by 3",
          RUN.beta_with_vectorlike(3, 1, vectorlike_10=1)[2], Fr(0))
    check_true("so a single light pair destroys asymptotic freedom",
               not RUN.vectorlike_verdict(3, vectorlike_10=1)["confines"])
    check("three generations tolerate none",
          RUN.vectorlike_verdict(3)["max_light_10_pairs"], 0)

    # Complete multiplets cancel from the differences, so the unification
    # prediction is untouched even as confinement is destroyed.
    from pyCICY.theories import couplings as CO
    b_bare = RUN.beta_coefficients(3, 1)
    b_vl = RUN.beta_with_vectorlike(3, 1, vectorlike_10=9)
    check("differences are unchanged by complete multiplets",
          b_vl[0] - b_vl[1], b_bare[0] - b_bare[1])
    check_true("hence the same alpha_3 prediction",
               abs(CO.predict_alpha3(*b_vl)[0]
                   - CO.predict_alpha3(*b_bare)[0]) < 1e-12)
    check_true("but b_3 has flipped sign (%s)" % b_vl[2], b_vl[2] > 0)

    # The chain returns no number, deliberately.
    c = RUN.mass_ratio_chain()
    check("four factors in m_p/m_e", len(c["factors"]), 4)
    check_true("none carries a value",
               all(f["value"] is None or "b_3" in str(f["value"])
                   for f in c["factors"]))
    check("three are unavailable",
          sum(1 for f in c["factors"] if f["status"] != "exact"), 4)
    check_true("and the reasons are distinct",
               len({f["status"] for f in c["factors"]}) >= 3)

    # Reached from a model, using its exact generation count.
    m = T.LineBundleModel(TETRA, MODEL, action=E.TETRAQUADRIC_Z2(),
                          wilson=(0, 1))
    check("the model's own beta coefficients", m.beta_coefficients(),
          RUN.MSSM_BETAS)


def test_moduli():
    print("\n[8] stabilising the dilaton")
    from pyCICY.theories import moduli as MO
    from fractions import Fraction as Fr

    check("condensation exponent for SU(N) is 8 pi^2/N",
          abs(MO.condensation_exponent(8) - 8 * math.pi ** 2 / 8) < 1e-12,
          True)
    check_true("a single condensate is refused: no racetrack",
               _raises(ValueError, MO.racetrack_dilaton, 7, 7))

    # The closed form for the minimum drops the K_S term. It should be a good
    # approximation, and if it is not the closed-form exponent below is not
    # the right one either.
    d = MO.racetrack_dilaton(7, 8, ratio=10.0)
    e = MO.racetrack_dilaton(7, 8, ratio=10.0, exact=True)
    rel = abs(d["re_s"] - e["re_s"]) / d["re_s"]
    check_true("closed form agrees with the full F-term to %.1e" % rel,
               rel < 5e-3)

    # The point of the module: the double exponential becomes a power law,
    # with the exponent an exact rational in three topological integers.
    check("exponent for SU(7)xSU(8), three generations",
          MO.qcd_scale_exponent(7, 8), Fr(56, 3))
    check("and it is N1 N2 / (|b3| (N2 - N1))",
          MO.qcd_scale_exponent(6, 8), Fr(6 * 8, 3 * 2))
    check_true("it is exact, not a float",
               isinstance(MO.qcd_scale_exponent(5, 10), Fr))
    check_true("and diverges as N2 -> N1, the runaway",
               _raises(ValueError, MO.qcd_scale_exponent, 7, 7))
    check_true("no confinement means no scale to set",
               _raises(ValueError, MO.qcd_scale_exponent, 7, 8,
                       n_generations=5))

    # The generation count enters the exponent, so the visible sector and the
    # hidden sector are not independent.
    check_true("four generations change the exponent",
               MO.qcd_scale_exponent(7, 8, n_generations=4)
               != MO.qcd_scale_exponent(7, 8, n_generations=3))

    # alpha_GUT is now an output rather than an input.
    r = MO.qcd_scale_from_racetrack(7, 8, ratio=10.0)
    check_true("SU(7)xSU(8), R=10 gives alpha_GUT near 1/20 (1/%.1f)"
               % (1 / r["alpha_gut"]), 18 < 1 / r["alpha_gut"] < 23)
    check_true("and Lambda within a factor of a few of 0.2 GeV (%.3f)"
               % r["lambda_qcd"], 0.02 < r["lambda_qcd"] < 2.0)

    # Both factors live in the hidden E_8, of rank eight, so
    # rank(SU(N1) x SU(N2)) = N1 + N2 - 2 <= 8. An earlier version of this
    # module omitted the constraint and reported SU(7)xSU(8) and friends as
    # viable; they are of rank thirteen and do not embed.
    check_true("SU(7)xSU(8) has rank 13 and is excluded",
               not MO.rank_allowed(7, 8))
    check_true("SU(4)xSU(5) has rank 7 and is allowed",
               MO.rank_allowed(4, 5))
    hits = MO.viable_hidden_groups()
    check_true("every surviving group is rank-allowed",
               all(MO.rank_allowed(h["N1"], h["N2"]) for h in hits))
    check_true("and the window is narrow (%d groups)" % len(hits),
               0 < len(hits) < 10)

    # The largest exponent available inside E_8 is 20/3.
    best = max((MO.qcd_scale_exponent(n1, n2)
                for n1 in range(2, 10) for n2 in range(n1 + 1, 10)
                if MO.rank_allowed(n1, n2)))
    check("the largest rank-allowed exponent", best, Fr(20, 3))
    check_true("attained by SU(4)xSU(5)",
               MO.qcd_scale_exponent(4, 5) == best)
    check_true("which needs the most modest ratio (%.3g)"
               % MO.required_ratio(4, 5),
               MO.required_ratio(4, 5) < MO.required_ratio(3, 4))

    # Inverting the running: where the dilaton must sit is fixed by the
    # generation count and the observed scale, and does NOT depend on the
    # hidden group. The hidden group only sets which R lands there.
    d = MO.dilaton_from_scale()
    check_true("inverted alpha_GUT is 1/20.2 (1/%.1f)" % (1 / d["alpha_gut"]),
               19.5 < 1 / d["alpha_gut"] < 21)
    check_true("and is independent of the hidden group",
               d["depends_on_hidden_group"] is False)
    check_true("the same for every rank-allowed group",
               len({round(1 / MO.racetrack_dilaton(
                   n1, n2, MO.required_ratio(n1, n2))["alpha_gut"], 1)
                   for n1 in range(2, 8) for n2 in range(n1 + 1, 8)
                   if MO.rank_allowed(n1, n2)}) == 1)

    # A line bundle sum cannot supply a racetrack, and says so.
    hc = MO.hidden_commutant(4)
    check("SU(4) leaves SO(10)", hc["commutant"], "SO(10)")
    check("with one non-abelian factor", hc["nonabelian_factors"], 1)
    check_true("so no racetrack from a line bundle sum",
               hc["racetrack_possible"] is False)
    check_true("an untabulated rank is refused",
               _raises(ValueError, MO.hidden_commutant, 9))

    # The anomaly surplus is reported as the hidden-sector budget, not used to
    # enumerate hidden bundles, and says so.
    from pyCICY import bundles as BU
    V = BU.LineBundleSum(CICY(TETRA), MODEL)
    hc = MO.hidden_group_constraint(V.anomaly()["surplus"])
    check_true("the surplus is effective", hc["effective"])
    check_true("and is labelled a budget, not an answer",
               "not implemented" in hc["note"])

    # Rank counting is necessary. The hidden bundle competes with the
    # racetrack group for the eight ranks of E_8.
    check("a rank-2 hidden bundle leaves 7", MO.rank_budget(2), 7)
    check("a rank-5 one leaves 4", MO.rank_budget(5), 4)

    # But rank counting is not sufficient, and this is where the module
    # tightened. Of the maximal-rank subgroups of E_8 only A4 x A4 is a
    # product of two SU factors, and its ranks are equal, so every racetrack
    # must come from breaking inside it. SU(4) then cannot appear: surviving
    # as SU(5-n) with n = 1 means a trivial hidden bundle and no breaking.
    ok45, _ = MO.embeds_in_e8(4, 5)
    check_true("SU(4)xSU(5) does not embed by these chains, though its rank "
               "would allow it", not ok45)
    check_true("SU(3)xSU(5) does", MO.embeds_in_e8(3, 5)[0])
    check_true("and SU(5)xSU(5) is maximal rank", MO.embeds_in_e8(5, 5)[0])

    reach = MO.reachable_racetracks()
    check("three racetracks are reachable", len(reach), 3)
    check_true("all with N <= 5", all(r["N2"] <= 5 for r in reach))
    check("the largest reachable exponent is 5/2",
          max(r["exponent"] for r in reach), Fr(5, 2))
    check_true("attained by SU(3)xSU(5)",
               reach[0]["N1"] == 3 and reach[0]["N2"] == 5)

    # The conclusion, and it is not encouraging: even the best case needs the
    # two condensation scales to differ by seven orders of magnitude.
    best = reach[0]
    check_true("which still needs R ~ 1e7 (%.2e)" % best["ratio_required"],
               1e6 < best["ratio_required"] < 1e9)
    check_true("and alpha_GUT is 1/20.2 regardless",
               19.5 < best["alpha_gut_inv"] < 21)

    # The scan turns the budget into an enumeration.
    hs = MO.hidden_scan(CICY(TETRA), V.anomaly()["surplus"], rank=4,
                        charge=1, limit=30)
    check_true("the budget admits hidden bundles of rank 4",
               len(hs["bundles"]) > 0)
    check("which leave SO(10)", hs["commutant"], "SO(10)")
    check_true("but still no racetrack", hs["racetrack_possible"] is False)
    bad = 0
    d3 = np.asarray(CICY(TETRA).triple_intersection(), dtype=float)
    for k in hs["bundles"]:
        k = np.asarray(k)
        if np.any(k.sum(axis=0)):
            bad += 1
        c2 = -0.5 * np.einsum("rst,as,at->r", d3, k, k)
        if np.any(c2 > np.asarray(V.anomaly()["surplus"]) + 1e-9):
            bad += 1
    check("every hidden bundle has c1 = 0 and fits the budget", bad, 0)


def test_couplings():
    print("\n[9] gauge coupling unification and the number 137")
    from pyCICY.theories import couplings as C
    from pyCICY.theories import running as RUN
    from fractions import Fraction as Fr

    # The classic prediction, whose only theory input is the b_i.
    a3, _ = C.predict_alpha3(*RUN.beta_coefficients(3, 1))
    check_true("MSSM predicts alpha_3(M_Z) = %.4f, measured %.4f"
               % (a3, C.ALPHA3_MZ), abs(a3 - C.ALPHA3_MZ) < 0.005)
    sm = C.predict_alpha3(Fr(41, 10), Fr(-19, 6), Fr(-7))[0]
    check_true("the Standard Model spectrum misses badly (%.4f)" % sm,
               abs(sm - C.ALPHA3_MZ) > 0.03)

    # Unification scale and coupling come out at the textbook values.
    u = C.unification_point(*RUN.beta_coefficients(3, 1))
    check_true("M_G = %.2g GeV, near 2e16" % u["m_gut"],
               1e16 < u["m_gut"] < 4e16)
    check_true("alpha_G = 1/%.1f, near 1/24" % u["alpha_gut_inv"],
               22 < u["alpha_gut_inv"] < 27)

    # THE CAVEAT. A complete generation contributes equally to every b_i, so
    # it cancels from all the differences, and the prediction depends only on
    # those. The quantity this package computes best is the one that drops out.
    preds = [C.predict_alpha3(*RUN.beta_coefficients(ng, 1))[0]
             for ng in (3, 4, 5)]
    check_true("alpha_3 is identical for 3, 4 and 5 generations",
               max(preds) - min(preds) < 1e-12)
    d34 = (RUN.beta_coefficients(3, 1)[0] - RUN.beta_coefficients(3, 1)[1],
           RUN.beta_coefficients(4, 1)[0] - RUN.beta_coefficients(4, 1)[1])
    check("because b_1 - b_2 does not depend on n_g", d34[0], d34[1])

    # What it *is* sensitive to is the vector-like sector, which an index
    # cannot see.
    two_h = C.predict_alpha3(*RUN.beta_coefficients(3, 2))[0]
    check_true("a second Higgs pair changes it drastically (%.3f)" % two_h,
               abs(two_h - a3) > 0.5)
    exotic = C.predict_alpha3(*RUN.beta_coefficients(3, 1, {"3+3bar": 1}))[0]
    check_true("as does one vector-like triplet (%.4f)" % exotic,
               abs(exotic - a3) > 0.04)

    # Electroweak quantities.
    check_true("sin^2 theta_W = %.4f, measured 0.2312" % C.sin2_theta_w(),
               abs(C.sin2_theta_w() - 0.2312) < 0.002)
    check_true("alpha_em^-1(M_Z) = %.2f, measured %.2f"
               % (C.alpha_em_inverse_mz(), C.ALPHA_EM_INV_MZ),
               abs(C.alpha_em_inverse_mz() - C.ALPHA_EM_INV_MZ) < 0.5)

    # And 137: reachable only with measured hadronic input.
    z = C.alpha_em_inverse_zero()
    check_true("running to zero gives %.2f against 137.036" % z,
               abs(z - 137.036) < 1.5)
    check_true("but Delta alpha is an input, not a computation",
               _raises(ValueError, C.alpha_em_inverse_zero, 127.95, 1.5))

    chain = C.fine_structure_chain()
    check("five steps to 137", len(chain["steps"]), 5)
    check("the last one has no value",
          chain["steps"][-1]["value"], None)
    check_true("and names the hadronic dispersion relation",
               "dispersion" in chain["steps"][-1]["reason"])
    check_true("exactly one step is labelled exact",
               sum(1 for st in chain["steps"] if st["status"] == "exact") == 1)


def test_mass_terms():
    print("\n[10] can the vector-like matter be lifted?")
    from pyCICY.theories import yukawa as Y

    r = Y.mass_terms(TETRA, MODEL)
    check("10s come from three summands", r["tens"], [1, 2, 4])
    check("10bars from one", r["antitens"], [0])
    check_true("singlets exist (%d types)" % r["n_singlet_types"],
               r["n_singlet_types"] > 0)

    # h^1(O_X) = h^{0,1} = 0 on a Calabi-Yau threefold, so there is no neutral
    # singlet and hence no diagonal mass term for any line bundle model.
    check_true("no neutral singlet exists", not r["neutral_singlet_exists"])

    check("no pair is liftable", r["liftable"], 0)
    check("three pairs, all trapped", r["unliftable"], 3)
    check_true("and the verdict says so",
               "no vector-like pair can be lifted" in r["verdict"])

    # The obstruction is the same one that kills the Yukawa couplings. Taking
    # the singlet (c,d) = (b,a) gives exactly the needed charge, so the
    # condition is a charge-conserving triple with h^1 > 0 throughout --
    # exactly viable_triples.
    X = CICY(TETRA)
    k = np.asarray(MODEL)
    bad = 0
    for a in r["tens"]:
        for b in r["antitens"]:
            trip = [k[a], -k[b], k[b] - k[a]]
            if np.any(sum(trip)):
                bad += 1
            h1 = [int(np.asarray(X.line_co(list(map(int, t))))[1])
                  for t in trip]
            # the pair is liftable exactly when all three are non-zero
            liftable = all(x > 0 for x in h1)
            rec = [p for p in r["pairs"] if p["pair"] == (a, b)][0]
            if liftable != rec["liftable"]:
                bad += 1
    check("the triple sums to zero and predicts liftability", bad, 0)

    # And it is systematic, not this model's bad luck.
    S = BU.scan(CICY(TETRA), rank=5, charge=1, generations=3,
                symmetry_order=2, limit=60)
    lift = sum(1 for s in S if Y.mass_terms(TETRA, s)["liftable"] > 0)
    check("no scanned model has a liftable pair (of %d)" % len(S), lift, 0)


def main():
    t0 = time.time()
    test_interface()
    test_exact_yukawa()
    test_convergence()
    test_line_bundle_model()
    test_yukawa_texture()
    test_viable_triples()
    test_running()
    test_moduli()
    test_couplings()
    test_mass_terms()

    print("\n" + "=" * 72)
    if FAILURES:
        print("FAILED (%d): %s" % (len(FAILURES), ", ".join(FAILURES)))
        return 1
    print("test_theories: all checks passed in %.1fs" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
