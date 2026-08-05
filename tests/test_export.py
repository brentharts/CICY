"""
Tests for pyCICY.export.

The bridge to cymetric and cymyc has to be tested in two halves, because the
target packages are not dependencies of this one and will usually be absent.

  [1]-[4] run always. They test the thing this package is responsible for:
          that the monomials have the right multidegree, that the
          Gamma-invariant ones are exactly those of the declared charge, and
          that the emitted tuples have the shapes the target APIs document.

  [5]     runs only if cymetric or cymyc is importable, and is the part that
          actually matters: hand the export to their code and check the points
          it generates lie on our hypersurface, and that the group action
          preserves it.

Section [5] is skipped rather than failed when the packages are missing, and
says so, since a skipped integration test and a passing one are different
things.

Run with:  python3 tests/test_export.py
       or: python3 run_tests.py
"""

import itertools
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyCICY import CICY
from pyCICY import equivariant as E
from pyCICY import export as X_

FAILURES = []
SKIPPED = []

TETRA = [[1, 2], [1, 2], [1, 2], [1, 2]]


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


def test_layout():
    print("\n[1] coordinates and multidegrees")
    check("ambient vector", list(X_.ambient_vector(TETRA)), [1, 1, 1, 1])
    check("coordinate blocks, n_i + 1 each",
          X_.coordinate_blocks(TETRA), [(0, 2), (2, 4), (4, 6), (6, 8)])

    m = X_.monomials(TETRA, 0)
    check("the quartic has 3^4 monomials", m.shape, (81, 8))

    # Every monomial must have exactly the right degree in every factor --
    # this is the condition the target packages assume and never check.
    bad = 0
    conf = np.array(TETRA)
    deg = conf[:, 1:].T[0]
    for row in m:
        for i, (s, e) in enumerate(X_.coordinate_blocks(TETRA)):
            if row[s:e].sum() != deg[i]:
                bad += 1
    check("every monomial has the right multidegree", bad, 0)

    # A second configuration with unequal degrees, to catch an off-by-one that
    # a symmetric example would hide.
    conf2 = [[2, 3], [2, 3]]
    m2 = X_.monomials(conf2, 0)
    check("bicubic monomial count", m2.shape, (100, 6))
    check("bicubic coordinate blocks",
          X_.coordinate_blocks(conf2), [(0, 3), (3, 6)])


def test_charges():
    print("\n[2] Gamma-charges of monomials")
    A = E.TETRAQUADRIC_Z2()
    ch = X_.monomial_charges(TETRA, 0, A)
    check("one charge per monomial", len(ch), 81)
    check("charges are 0 or 1", sorted(set(ch.tolist())), [0, 1])
    check("41 monomials are invariant", int((ch == 0).sum()), 41)

    inv, mask = X_.invariant_monomials(TETRA, 0, A)
    check("invariant_monomials agrees", inv.shape, (41, 8))

    # The charge must be computed the same way equivariant.py computes it, or
    # the exported polynomial would not be the one the index assumed.
    weights = [w for row in A.weights for w in row]
    bad = 0
    for row, c in zip(X_.monomials(TETRA, 0), ch):
        if (sum(int(r) * int(w) for r, w in zip(row, weights)) % 2) != c:
            bad += 1
    check("charges match the weight sum directly", bad, 0)

    # And the declared polynomial charge must be admissible, or there is no
    # invariant hypersurface at all.
    check_true("the declared charge is admissible",
               0 in A.admissible_polynomial_charges(0))


def test_polynomials():
    print("\n[3] defining polynomials")
    A = E.TETRAQUADRIC_Z2()
    mons, coeffs = X_.defining_polynomials(TETRA, action=A, seed=1)
    check("one polynomial per hypersurface", len(mons), 1)
    check("invariant polynomial uses 41 monomials", len(mons[0]), 41)
    check("one coefficient per monomial", len(coeffs[0]), len(mons[0]))
    check_true("coefficients are complex by default",
               np.iscomplexobj(coeffs[0]))

    mons2, _ = X_.defining_polynomials(TETRA, action=None, seed=1)
    check("without an action, all 81 are used", len(mons2[0]), 81)

    # Reproducible from the seed, since both target packages treat the
    # coefficients as given data.
    _, c1 = X_.defining_polynomials(TETRA, action=A, seed=7)
    _, c2 = X_.defining_polynomials(TETRA, action=A, seed=7)
    check_true("the same seed gives the same coefficients",
               np.allclose(c1[0], c2[0]))
    _, c3 = X_.defining_polynomials(TETRA, action=A, seed=8)
    check_true("a different seed does not", not np.allclose(c1[0], c3[0]))

    # An unrealisable charge must be refused rather than yielding an empty
    # polynomial, which would look like a manifold and be nothing.
    class _Bad(object):
        weights = A.weights
        n = 2
        polynomial_charges = [1]
        conf = A.conf
    try:
        X_.defining_polynomials(TETRA, action=_Bad())
        ok = _has_charge_one(A)
    except ValueError:
        ok = True
    check_true("an unrealisable declared charge is handled", ok)


def _has_charge_one(A):
    return 1 in A.admissible_polynomial_charges(0)


def test_smoothness():
    print("\n[3b] the smoothness filter")
    A = E.TETRAQUADRIC_Z2()
    mons = X_.monomials(TETRA, 0)

    # A single monomial is a union of hyperplanes with multiplicity, singular
    # all along the intersections. If the check cannot see that, it sees
    # nothing.
    i = [k for k, r in enumerate(mons) if list(r) == [2, 0, 2, 0, 2, 0, 2, 0]][0]
    c = np.zeros(len(mons), dtype=complex)
    c[i] = 1.0
    ok, on_X, drops = X_.is_smooth_over_Fp(TETRA, [mons], [c], samples=8000)
    check_true("a single monomial is detected as singular", not ok)
    check_true("and the rank drops at every point of X found (%d/%d)"
               % (drops, on_X), on_X > 0 and drops == on_X)

    # A polynomial divisible by x_00^2 is non-reduced along x_00 = 0.
    rng = np.random.default_rng(0)
    c2 = np.zeros(len(mons), dtype=complex)
    for k, r in enumerate(mons):
        if list(r)[:2] == [2, 0]:
            c2[k] = rng.integers(1, 9) + 1j * rng.integers(1, 9)
    check_true("a non-reduced polynomial is detected",
               not X_.is_smooth_over_Fp(TETRA, [mons], [c2], samples=8000)[0])

    # The reduction must be faithful, or the check tests a different
    # polynomial. Floats are refused rather than rounded: rounding the real
    # part of 0.3 + 0.7i gives 0, which deletes the monomial, and an earlier
    # version of this function reported a false singularity for exactly that
    # reason on the first seed it was pointed at.
    check_true("float coefficients are refused",
               _raises(X_.is_smooth_over_Fp, TETRA, [mons],
                       [np.random.default_rng(0).normal(size=len(mons))],
                       samples=200))
    check_true("and a prime with no square root of -1 is refused",
               _raises(X_.is_smooth_over_Fp, TETRA, [mons], [c], p=103,
                       samples=200))

    # A generic one passes, and finds points to test on.
    rng3 = np.random.default_rng(5)
    c3 = (rng3.integers(1, 13, size=len(mons))
          + 1j * rng3.integers(1, 13, size=len(mons))).astype(complex)
    ok3, on3, dr3 = X_.is_smooth_over_Fp(TETRA, [mons], [c3], samples=8000)
    check_true("a generic polynomial passes", ok3)
    check_true("having found %d points of X to test" % on3, on3 > 20)
    check("with no rank drops", dr3, 0)

    # The export path now runs this by default, so a singular draw cannot
    # reach a metric package silently.
    # Every seed in a range must pass, since a generic member of the family
    # is smooth. Failures here mean the check is wrong, not the geometry.
    good = 0
    for sd in range(6):
        try:
            X_.defining_polynomials(TETRA, action=A, seed=sd, samples=4000)
            good += 1
        except ValueError:
            pass
    check("six consecutive seeds all give smooth polynomials", good, 6)

    m, cc = X_.defining_polynomials(TETRA, action=A, seed=3)
    check_true("defining_polynomials checks by default", len(m) == 1)
    check_true("and its coefficients are Gaussian integers",
               np.allclose(cc[0], np.rint(cc[0].real) + 1j * np.rint(cc[0].imag)))
    check_true("and the check can be turned off",
               len(X_.defining_polynomials(TETRA, action=A, seed=3,
                                           check_smooth=False)[0]) == 1)


def test_formats():
    print("\n[4] the two output formats")
    A = E.TETRAQUADRIC_Z2()

    d = X_.to_cymetric(TETRA, action=A, seed=3)
    check("cymetric keys", sorted(d), ["ambient", "coefficients", "kmoduli",
                                       "monomials"])
    check("ambient", list(d["ambient"]), [1, 1, 1, 1])
    check("one kmodulus per ambient factor", len(d["kmoduli"]), 4)
    check_true("monomials is a list of arrays",
               isinstance(d["monomials"], list)
               and d["monomials"][0].dtype == np.int64)

    (mons, cy_dim, km, amb), coeffs = X_.to_cymyc(TETRA, action=A, seed=3)
    check("cymyc cy_dim", cy_dim, 3)
    check("cymyc ambient", list(amb), [1, 1, 1, 1])
    check_true("cymyc kmoduli are complex64", km.dtype == np.complex64)

    # cy_dim must be sum(n_i) - (number of equations), and a second example
    # keeps that from being a coincidence of the tetraquadric.
    (_, cy2, _, _), _ = X_.to_cymyc([[2, 3], [2, 3]], seed=0)
    check("bicubic cy_dim", cy2, 3)
    (_, cy3, _, _), _ = X_.to_cymyc([[4, 5]], seed=0)
    check("quintic cy_dim", cy3, 3)

    src = X_.poly_spec_source(TETRA, action=A, name="tetraquadric_z2")
    check_true("poly_spec source mentions the function name",
               "def tetraquadric_z2()" in src)
    check_true("and is valid Python", _compiles(src))
    ns = {}
    exec(compile(src, "<generated>", "exec"), ns)
    m2, cy4, km4, amb4 = ns["tetraquadric_z2"]()
    check("the generated source round-trips: cy_dim", cy4, 3)
    check("and the monomials", m2[0].shape, (41, 8))


def _compiles(src):
    try:
        compile(src, "<generated>", "exec")
        return True
    except SyntaxError:
        return False


MODEL = [[-2, -2, -1, 2], [-2, 1, 0, 0], [1, -2, 1, 0],
         [1, 1, -1, 0], [2, 2, 1, -2]]


def test_kmoduli():
    print("\n[4b] Kahler moduli from the stability locus")
    from pyCICY import bundles as BU

    V = BU.LineBundleSum(CICY(TETRA), MODEL)

    # The point of the whole function: at kmoduli = ones the bundle is not
    # stable, so a metric computed there is a metric at a point in moduli
    # space where the model does not exist.
    at_ones = np.abs(np.asarray(V.slopes(np.ones(4)))).max()
    check_true("the slopes do NOT vanish at kmoduli = ones (%.1f)" % at_ones,
               at_ones > 1.0)

    r = X_.kmoduli_from_stability(TETRA, MODEL)
    check_true("but they do at the stability point (%.1e)"
               % np.abs(r["slopes"]).max(), np.abs(r["slopes"]).max() < 1e-6)
    check_true("which is in the interior of the cone",
               bool((r["kmoduli"] > 0).all()))

    # And it is not merely a rescaling of ones -- the direction differs.
    cos = float(np.dot(np.ones(4), r["kmoduli"])
                / np.linalg.norm(np.ones(4)) / np.linalg.norm(r["kmoduli"]))
    check_true("and is not a rescaling of ones (cos = %.3f)" % cos, cos < 0.99)

    # Every normalisation must keep the slopes vanishing, since rescaling a
    # solution of a homogeneous condition is still a solution. If one of them
    # did not, the scaling would be wrong.
    for norm in ("volume", "sum", "max", "raw"):
        rr = X_.kmoduli_from_stability(TETRA, MODEL, normalisation=norm)
        check_true("normalisation=%-7s keeps the slopes zero" % norm,
                   np.abs(rr["slopes"]).max() < 1e-6)
    check_true("normalisation='volume' really gives volume 1",
               abs(X_.kmoduli_from_stability(
                   TETRA, MODEL, normalisation="volume")["volume"] - 1) < 1e-9)
    check_true("normalisation='sum' sums to the number of moduli",
               abs(X_.kmoduli_from_stability(
                   TETRA, MODEL,
                   normalisation="sum")["kmoduli"].sum() - 4) < 1e-9)
    check_true("an unknown normalisation is refused",
               _raises(X_.kmoduli_from_stability, TETRA, MODEL,
                       normalisation="nonsense"))

    # A bundle with no stability locus has no correct Kahler point, so the
    # function raises instead of falling back to a default.
    unstable = [[1, 1, 1, 1], [-1, -1, -1, -1], [0, 0, 0, 0],
                [0, 0, 0, 0], [0, 0, 0, 0]]
    check_true("an unstable bundle is refused, not defaulted",
               _raises(X_.kmoduli_from_stability, TETRA, unstable))

    # The exporters accept it, and refuse it without the bundle.
    d = X_.to_cymetric(TETRA, kmoduli="stability", summands=MODEL)
    check_true("to_cymetric(kmoduli='stability') works",
               np.allclose(d["kmoduli"], r["kmoduli"]))
    check_true("and needs the summands",
               _raises(X_.to_cymetric, TETRA, kmoduli="stability"))
    check_true("an unknown named option is refused",
               _raises(X_.to_cymetric, TETRA, kmoduli="whatever",
                       summands=MODEL))
    check_true("the default is still ones",
               np.allclose(X_.to_cymetric(TETRA)["kmoduli"], np.ones(4)))


def _raises(fn, *a, **kw):
    try:
        fn(*a, **kw)
    except Exception:                                            # noqa: BLE001
        return True
    return False


def test_quotient_conditions():
    print("\n[4c] conditions for the quotient to exist")
    A = E.TETRAQUADRIC_Z2()

    # Gamma must preserve Omega or X/Gamma is not Calabi-Yau at all. This is
    # independent of freeness: neither implies the other, and the tests below
    # exhibit both failures separately.
    check("omega character of the free Z_2", X_.omega_character(TETRA, A), 0)
    check_true("so it preserves Omega",
               X_.preserves_holomorphic_form(TETRA, A))

    notCY = E.CyclicAction(TETRA, 2, [[0, 1]] * 4, [1])
    check_true("a charge-1 quartic breaks Omega",
               not X_.preserves_holomorphic_form(TETRA, notCY))

    # preserves Omega but is not free
    z4 = E.CyclicAction(TETRA, 4, [[0, 1]] * 4, [0])
    check_true("Z_4 preserves Omega", X_.preserves_holomorphic_form(TETRA, z4))
    check_true("but is not free", not z4.looks_free()[0])

    # The textbook case: the free Z_5 on the quintic, x_i -> zeta^i x_i, which
    # must satisfy both. Weights sum to 10, vanishing mod 5, against a
    # charge-0 quintic.
    q = E.CyclicAction([[4, 5]], 5, [[0, 1, 2, 3, 4]], [0])
    check("omega character of the quintic Z_5",
          X_.omega_character([[4, 5]], q), 0)
    check_true("and it is free",
               q.looks_free(probes=[[k] for k in range(-3, 4)])[0])

    # The group as matrices, which is all a metric package needs.
    G = X_.group_matrices(A)
    check("one matrix per element", len(G), 2)
    check_true("the identity is present",
               any(np.allclose(g, np.eye(8)) for g in G))
    check_true("and the group closes",
               all(any(np.allclose(a @ b, c) for c in G) for a in G for b in G))
    check_true("the non-trivial element is an involution",
               np.allclose(G[1] @ G[1], np.eye(8)))

    # The Kahler class must descend too.
    r = X_.kmoduli_from_stability(TETRA, MODEL)
    check_true("kmoduli are invariant under a non-permuting action",
               X_.kmoduli_are_invariant(A, r["kmoduli"]))
    perm = E.PermutationAction(TETRA, 2, [1, 0, 3, 2], [[0, 1]] * 4, [0])
    check_true("but not under one that permutes factors",
               not X_.kmoduli_are_invariant(perm, r["kmoduli"]))
    check_true("and they are, if constant on the orbits",
               X_.kmoduli_are_invariant(perm, np.array([2., 2., 5., 5.])))

    rep = X_.quotient_report(TETRA, A, summands=MODEL)
    check_true("the quotient report passes for this model", rep["ok"])
    check_true("and fails when Omega is broken",
               not X_.quotient_report(TETRA, notCY)["ok"])


def test_integration():
    print("\n[5] integration with cymetric / cymyc")
    A = E.TETRAQUADRIC_Z2()
    # the Z_2 acts as x_{i,1} -> -x_{i,1} on each factor
    g = np.array([1, -1, 1, -1, 1, -1, 1, -1])

    def evalp(z, m, c):
        return np.array([(c * np.prod(zi[None, :] ** m, axis=1)).sum()
                         for zi in z])

    try:
        import warnings
        import logging
        warnings.filterwarnings("ignore")
        logging.disable(logging.INFO)
        from cymetric.pointgen.pointgen_cicy import CICYPointGenerator
    except Exception as e:                                       # noqa: BLE001
        print("  cymetric not importable (%s); skipping" % type(e).__name__)
        SKIPPED.append("cymetric")
        CICYPointGenerator = None

    if CICYPointGenerator is not None:
        args = X_.to_cymetric(TETRA, action=A, seed=3)
        pg = CICYPointGenerator(**args, verbose=3)
        z = pg.generate_points(200)
        m, c = args["monomials"][0], args["coefficients"][0]
        on = float(np.abs(evalp(z, m, c)).max())
        check_true("cymetric points lie on our hypersurface (%.1e)" % on,
                   on < 1e-6)

        # The reason the bridge exists: the action must preserve X, or the
        # metric would be computed on a manifold with no symmetry and the
        # quotient the model needs would not exist.
        moved = float(np.abs(evalp(z * g, m, c)).max())
        check_true("and Gamma preserves it (%.1e)" % moved, moved < 1e-6)

        # The contrast, which is what makes the check non-vacuous.
        args0 = X_.to_cymetric(TETRA, action=None, seed=3)
        pg0 = CICYPointGenerator(**args0, verbose=3)
        z0 = pg0.generate_points(100)
        m0, c0 = args0["monomials"][0], args0["coefficients"][0]
        on0 = float(np.abs(evalp(z0, m0, c0)).max())
        moved0 = float(np.abs(evalp(z0 * g, m0, c0)).max())
        check_true("a random polynomial also gives points on itself (%.1e)"
                   % on0, on0 < 1e-6)
        check_true("but Gamma does NOT preserve it (%.1e)" % moved0,
                   moved0 > 1e-3)

        # A cross-package check of our intersection numbers. cymetric's
        # get_volume_from_intersections returns int_X J^3 = d_rst t^r t^s t^t
        # -- six times our kappa, since we carry the 1/3!. Everything else
        # must agree exactly, and it does: our numbers come from the Leray
        # spectral sequence on the configuration matrix, theirs from an
        # independent computation.
        worst = 0.0
        rng2 = np.random.default_rng(1)
        for conf in (TETRA, [[2, 3], [2, 3]], [[4, 5]]):
            pgv = CICYPointGenerator(**X_.to_cymetric(conf, seed=2),
                                     verbose=3)
            for _ in range(5):
                tt = rng2.random(len(conf)) + 0.3
                theirs = pgv.get_volume_from_intersections(tt)
                ours = 6.0 * X_.kahler_volume(conf, tt)
                worst = max(worst, abs(theirs - ours) / max(1.0, abs(ours)))
        check_true("our triple intersections match cymetric's (%.1e)" % worst,
                   worst < 1e-9)

        # The equivariant generator on the fork: the point set must be closed
        # under Gamma, the measure unchanged, and a non-invariant polynomial
        # rejected rather than quietly accepted.
        try:
            from cymetric.pointgen import EquivariantCICYPointGenerator
        except ImportError:
            print("  EquivariantCICYPointGenerator not present; skipping")
            SKIPPED.append("cymetric-equivariant")
            EquivariantCICYPointGenerator = None

        if EquivariantCICYPointGenerator is not None:
            eargs = X_.to_cymetric(TETRA, action=A, kmoduli="stability",
                                   summands=MODEL, seed=3, include_group=True)
            plain = {k: v for k, v in eargs.items() if k != "group_matrices"}
            pe = EquivariantCICYPointGenerator(**eargs, verbose=3)
            pp = CICYPointGenerator(**plain, verbose=3)
            we = pe.generate_point_weights(500)
            wp = pp.generate_point_weights(500)
            check("the orbit doubles the sample", len(we), 2 * len(wp))

            # Comparing the two generators' weight *sums* would compare two
            # Monte Carlo estimates on different random samples, which differ
            # by sampling noise and tests nothing. The exact property is that
            # the measure is Gamma-invariant: a point and its image carry the
            # same weight. Rows 0..n-1 are the identity images and rows n..2n-1
            # are their images under the second group element.
            half = len(we) // 2
            wa, wb = we["weight"][:half], we["weight"][half:]
            rel = float(np.max(np.abs(wa - wb)
                               / np.maximum(np.abs(wa), 1e-30)))
            check_true("a point and its image carry the same weight (%.1e)"
                       % rel, rel < 1e-9)

            zz = we["point"]
            gg = eargs["group_matrices"][1]
            img = pe._rescale_to_patch(zz @ gg.T)
            dist = np.min(np.abs(img[:, None, :] - zz[None, :, :]).max(-1),
                          axis=1).max()
            check_true("the point set is Gamma-closed (%.1e)" % dist,
                       dist < 1e-8)
            check_true("invariance verified on the sample (%.1e)"
                       % pe.verify_invariance(zz[:50]),
                       pe.verify_invariance(zz[:50]) < 1e-6)

            bad = X_.to_cymetric(TETRA, action=None, seed=3)
            bad["group_matrices"] = eargs["group_matrices"]
            check_true("a non-invariant polynomial is rejected",
                       _raises(lambda: EquivariantCICYPointGenerator(
                           **bad, verbose=3).generate_point_weights(50)))

            # Which space the weights integrate over. The augmented sample is
            # a sample of the cover, so integrals against it are |Gamma| times
            # the quotient answer -- a trap, since nothing downstream knows.
            pq = EquivariantCICYPointGenerator(**eargs, measure="quotient",
                                               verbose=3)
            wq = pq.generate_point_weights(500)
            ratio = float(we["weight"].sum() / wq["weight"].sum())
            check_true("measure='quotient' divides by |Gamma| (%.4f)" % ratio,
                       abs(ratio - pe.gamma_order) < 1e-9)
            check("gamma_order is exposed", pe.gamma_order, 2)
            check_true("vol_quotient is vol(X)/|Gamma|",
                       abs(pe.vol_quotient
                           - pe.get_volume_from_intersections(pe.kmoduli) / 2)
                       < 1e-9)
            check_true("an unknown measure is refused",
                       _raises(lambda: EquivariantCICYPointGenerator(
                           **eargs, measure="nonsense", verbose=3)))

    try:
        from cymyc import alg_geo
        import jax.numpy as jnp
    except Exception as e:                                       # noqa: BLE001
        print("  cymyc not importable (%s); skipping" % type(e).__name__)
        SKIPPED.append("cymyc")
        return

    (mons, cy_dim, km, amb), coeffs = X_.to_cymyc(TETRA, action=A, seed=3)
    rng = np.random.default_rng(0)
    zz = rng.normal(size=8) + 1j * rng.normal(size=8)
    theirs = complex(alg_geo.evaluate_poly(jnp.array(zz), jnp.array(mons[0]),
                                           jnp.array(coeffs[0])))
    ours = complex((coeffs[0] * np.prod(zz[None, :] ** mons[0],
                                        axis=1)).sum())
    # JAX defaults to float32, so the tolerance is theirs and not ours.
    rel = abs(theirs - ours) / max(1.0, abs(ours))
    check_true("cymyc.evaluate_poly agrees with our own (%.1e relative)" % rel,
               rel < 1e-4)


def main():
    t0 = time.time()
    test_layout()
    test_charges()
    test_polynomials()
    test_smoothness()
    test_formats()
    test_kmoduli()
    test_quotient_conditions()
    test_integration()

    print("\n" + "=" * 72)
    if FAILURES:
        print("FAILED (%d): %s" % (len(FAILURES), ", ".join(FAILURES)))
        return 1
    msg = "test_export: all checks passed in %.1fs" % (time.time() - t0)
    if SKIPPED:
        msg += "  (integration skipped: %s not installed)" % ", ".join(SKIPPED)
    print(msg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
