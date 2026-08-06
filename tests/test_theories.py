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

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyCICY import CICY
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


def main():
    t0 = time.time()
    test_interface()
    test_exact_yukawa()
    test_convergence()
    test_line_bundle_model()
    test_yukawa_texture()
    test_viable_triples()

    print("\n" + "=" * 72)
    if FAILURES:
        print("FAILED (%d): %s" % (len(FAILURES), ", ".join(FAILURES)))
        return 1
    print("test_theories: all checks passed in %.1fs" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
