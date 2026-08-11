"""
Tests for pyCICY.theories.nariai.

The module exists to check a paper, so the tests check the checker: every
quantity the module computes by an independent route is required to agree
with the paper's closed forms, and every refusal is required to be the right
kind of refusal. The recurring pattern is one number, several derivations,
no shared code -- the same discipline test_mtheory applies to the F-theory
duality.

  [1] geometry       the exact SdS cubic against the Ginsparg-Perry
                     expansion: root splitting, both surface gravities
                     vanishing linearly (why the Bousso-Hawking rescaling is
                     forced), and the antipodal area balance at O(eps^2)
  [2] nariai         the invariants of the degenerate solution, the KK tower
                     as the physical regulator, and the self-generated
                     timescale hierarchy, including at the observed Lambda
  [3] rational       the closed form (38)-(39) as an exact identity in
                     fractions.Fraction -- no floating point anywhere
  [4] routes         one entropy, four computations: closed form, direct
                     quadrature of (31), the Araki symplectic route (29)
                     with no integration by parts, and the JT dilaton ODE,
                     which must return delta A = 4 S_rel with 8 pi derived
  [5] clausius       S_rel = beta E exactly, the ledger equation, and the
                     per-nat area shift equal to 1/S_N
  [6] vanishing      I_2 = 0 on the contour packet, where the answer is a
                     theorem of complex analysis rather than a small number;
                     the boost-Jacobian identity; the 0-or-1 wedge-locality
                     dichotomy
  [7] budget         squeezed negativity windows exist, are exactly prepaid,
                     the total is thermal and theta-independent, and the
                     running ledger's dips respect the budget
  [8] ratio_law      the universal funded-depth ratio found by this module's
                     own cross-checks: packet- and angle-independent,
                     matching the phase-average closed form and reproducing
                     the paper's Gaussian-packet values
  [9] one_mode       the sum rule S_rel = beta dE at general squeezing
                     angle, the closed form (66), the Shale/entropy
                     inequivalence on a tower, and additivity
 [10] categories     TypeIIIFactor for the von Neumann entropy, NoSuchTheory
                     for the absent Yukawa sector, registration, describe()

Run with:  python3 tests/test_nariai.py
       or: python3 run_tests.py
"""

import math
import os
import sys
import time
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from pyCICY import theories as T
from pyCICY.theories import nariai as N
from pyCICY.theories.base import NeedsMetric
from pyCICY.theories.ftheory import NoSuchTheory

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


def check_close(name, got, want, tol=1e-9):
    ok = abs(got - want) <= tol * max(1.0, abs(want))
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
# [1] the exact cubic against the expansion
# ---------------------------------------------------------------------------

def test_geometry():
    banner("[1] Schwarzschild-de Sitter: exact roots vs Ginsparg-Perry")

    # A nondegenerate member: two horizons, two temperatures. The inequality
    # of the surface gravities is why generic SdS has no global KMS state.
    d = N.sds_horizons(mass=0.2, lam=1.0)
    check_true("two horizons, rb < rc", 0 < d["rb"] < d["rc"])
    check_true("their surface gravities differ",
               abs(d["kappa_b"] - d["kappa_c"]) > 1e-3)
    # f vanishes on both roots -- the roots are actually roots.
    for r in (d["rb"], d["rc"]):
        f = 1.0 - 2.0 * 0.2 / r - r ** 2 / 3.0
        check_close("   f(r) = 0 at r = %.4f" % r, f, 0.0, 1e-12)

    # The degenerate slice: the expansion is confirmed, not assumed. The
    # ratios to the first-order forms must approach 1, and quadratically.
    d1 = N.degenerate_family(0.01)
    d2 = N.degenerate_family(0.001)
    check_close("rb -> r*(1 - eps) [eps=1e-2]", d1["rb_over_expansion"],
                1.0, 2e-4)
    check_close("rb -> r*(1 - eps) [eps=1e-3]", d2["rb_over_expansion"],
                1.0, 2e-6)
    check_true("   convergence is quadratic in eps",
               abs(d2["rb_over_expansion"] - 1.0)
               < 0.02 * abs(d1["rb_over_expansion"] - 1.0))

    # Both surface gravities vanish linearly, with unit coefficient in
    # units of eps sqrt(Lambda): the quantitative content of the statement
    # that a construction anchored to the SdS time sees zero temperature,
    # and that the Bousso-Hawking rescaling is forced rather than chosen.
    check_close("kappa_b / (eps sqrtL) -> 1", d2["kappa_b_over_eps"],
                1.0, 1e-2)
    check_close("kappa_c / (eps sqrtL) -> 1", d2["kappa_c_over_eps"],
                1.0, 1e-2)

    # The antipodal balance: at first order the two areas shift equally and
    # oppositely, so the imbalance is O(eps^2) -- ratio 4 under eps -> eps/2.
    i1 = N.degenerate_family(0.02)["area_imbalance"]
    i2 = N.degenerate_family(0.01)["area_imbalance"]
    check_close("area imbalance scales as eps^2", i1 / i2, 4.0, 5e-3)
    # ...while the individual radius shifts scale as eps: halving eps halves
    # them. First order versus second order is the content of the balance.
    s1 = abs(N.degenerate_family(0.02)["rb"] - N.degenerate_family(0.02)["r_star"])
    s2 = abs(N.degenerate_family(0.01)["rb"] - N.degenerate_family(0.01)["r_star"])
    check_close("   individual shifts scale as eps, not eps^2",
                s1 / s2, 2.0, 5e-3)


# ---------------------------------------------------------------------------
# [2] the invariants, the tower, and the hierarchy
# ---------------------------------------------------------------------------

def test_nariai():
    banner("[2] Nariai data, the KK regulator, and the hierarchy")

    d = N.nariai_data(1.0)
    check_close("area 4pi/Lambda", d["area"], 4.0 * math.pi)
    check_close("entropy pi/Lambda", d["entropy"], math.pi)
    check_close("kappa = sqrt(Lambda)", d["kappa"], 1.0)
    check_close("beta * T = 1", d["beta"] * d["temperature"], 1.0)

    # The tower: every l >= 1 mode is principal series with m^2/Lambda >= 2,
    # so the transverse continuum of Rindler is a discrete, exponentially
    # decoupling tower -- the physical regulator behind every finite sum.
    tower = N.kk_tower(lam=1.0, bare_mass=0.0, lmax=5)
    check("l = 0 is the light mode", tower[0]["m2"], 0.0)
    check_close("l = 1 has m^2/Lambda = 2", tower[1]["m2_over_lambda"], 2.0)
    check_true("every l >= 1 is principal series",
               all(t["principal_series"] for t in tower[1:]))
    supp = [t["boltzmann_suppression"] for t in tower[1:]]
    check_true("suppression is monotone in l",
               all(supp[i + 1] < supp[i] for i in range(len(supp) - 1)))
    check_true("   and roughly e^{-pi l}",
               supp[3] / supp[2] < math.exp(-math.pi) * 1.5)

    # The hierarchy is the entropy itself, and at the observed Lambda it is
    # the famous 1e122. The classical drift time is infinite: epsilon labels
    # exact static solutions, so the instability is O(hbar) from the start.
    t = N.timescales(1.0)
    check_close("t_inst/t_mod = S_N/pi", t["hierarchy"], 1.0)
    check_true("classical drift time is infinite",
               t["t_classical"] == math.inf)
    check_close("per-nat area shift = 1/S_N",
                t["per_nat_area_shift"], 1.0 / math.pi)
    obs = N.timescales(N.OBSERVED_LAMBDA_PLANCK)
    check_true("observed universe: S_N ~ 1e122",
               1e121 < obs["entropy"] < 2e122)
    check_true("   nucleation exponent is -S_N",
               obs["nucleation_exponent"] == -obs["entropy"])


# ---------------------------------------------------------------------------
# [3] the closed form as exact arithmetic
# ---------------------------------------------------------------------------

def test_rational():
    banner("[3] the longitudinal integral (38) as a rational identity")

    # int x^{2n-1} (n-x)^2 e^{-2x} dx = n (2n-1)! / (2 4^n), proved by
    # expanding and using int x^m e^{-2x} = m!/2^{m+1}. Both sides in exact
    # Fractions; coherent_entropy() itself asserts this, so here we recompute
    # the Gamma route independently and also check the first few values.
    fact = math.factorial
    for n in range(1, 9):
        closed = Fraction(n * fact(2 * n - 1), 2 * 4 ** n)
        gamma = (Fraction(n * n * fact(2 * n - 1), 4 ** n)
                 - Fraction(n * fact(2 * n), 4 ** n)
                 + Fraction(fact(2 * n + 1), 4 * 4 ** n))
        check_true("n = %d: Gamma expansion equals closed form (%s)"
                   % (n, closed), closed == gamma)
    check("n = 1 value", N.coherent_entropy(1, 1.0)["longitudinal_integral"],
          Fraction(1, 8))
    check("n = 3 value", N.coherent_entropy(3, 1.0)["longitudinal_integral"],
          Fraction(45, 16))

    # And the paper's headline number: S_rel(n=1) = pi^2 a^2 / Lambda.
    c = N.coherent_entropy(1, 0.5, lam=2.0)
    check_close("S_rel(n=1) = pi^2 a^2 / Lambda",
                c["S_rel"], math.pi ** 2 * 0.25 / 2.0, 1e-14)
    check_true("the family starts at n = 1",
               _raises(ValueError, N.coherent_entropy, 0, 1.0))


# ---------------------------------------------------------------------------
# [4] one entropy, four routes
# ---------------------------------------------------------------------------

def test_routes():
    banner("[4] closed form, quadrature, Araki route, JT dilaton ODE")

    a = 0.1
    for n in (1, 2, 3, 4):
        c = N.coherent_entropy(n, a)["S_rel"]
        q = N.coherent_entropy_quadrature(n, a)
        ar = N.coherent_entropy_araki(n, a)
        # The quadrature route is Eq. (31): stress tensor against the
        # modular weight. The Araki route is Eq. (29): the symplectic form
        # on (delta phi, phi), both derivative terms computed, no
        # integration by parts. Agreement of the three tests the
        # manipulation (30) numerically.
        check_close("n=%d quadrature matches closed form" % n, q, c, 1e-6)
        check_close("n=%d Araki route matches closed form" % n, ar, c, 1e-6)

        # The JT theorem: integrate the dilaton ODE and compare with 4 S_rel.
        # The two sides share no code; the ratio is the theorem, and the
        # coupling 8 pi is derived by the reduction rather than imposed.
        jt = N.jt_area_response(n, a)
        check_close("n=%d dilaton ODE gives delta A = 4 S_rel" % n,
                    jt["ratio"], 1.0, 1e-5)
    check("   with the derived coupling",
          N.jt_area_response(1, a)["coupling_derived"], 8.0 * math.pi)

    # Scaling: S_rel is quadratic in the amplitude (a classical pulse's
    # energy) and inversely proportional to Lambda (the transverse area).
    s1 = N.coherent_entropy(2, 0.1, lam=1.0)["S_rel"]
    check_close("S_rel scales as a^2",
                N.coherent_entropy(2, 0.2, lam=1.0)["S_rel"] / s1, 4.0, 1e-12)
    check_close("S_rel scales as 1/Lambda",
                N.coherent_entropy(2, 0.1, lam=4.0)["S_rel"] / s1, 0.25,
                1e-12)


# ---------------------------------------------------------------------------
# [5] Clausius, the ledger, and the self-generated hierarchy
# ---------------------------------------------------------------------------

def test_clausius():
    banner("[5] S_rel = beta E, the ledger, and one nat per 1/S_N")

    c = N.coherent_entropy(1, 0.3, lam=1.0)
    # The Clausius relation is exact by construction of the dictionary, so
    # test the *independent* content: the Killing energy against the paper's
    # closed form E = pi a^2 / (2 sqrt(Lambda)) for n = 1.
    check_close("Killing energy pi a^2 / 2 sqrtL",
                c["killing_energy"], math.pi * 0.09 / 2.0, 1e-14)
    check_close("beta E = S_rel", c["clausius"], c["S_rel"], 1e-14)
    check_close("delta A = 4 S_rel", c["delta_A"], 4.0 * c["S_rel"], 1e-14)

    # The ledger: capacity plus record, finite numbers only, and the
    # per-nat shift is exactly 1/S_N -- the framework computing the size of
    # its own domain of validity.
    d = N.nariai_data(1.0)
    check_close("ledger total = pi/Lambda + S_rel",
                c["ledger_total"], d["entropy"] + c["S_rel"], 1e-14)
    check_close("per-nat shift = 1/S_N",
                c["per_nat_shift"], 1.0 / d["entropy"], 1e-14)
    check_true("coherent flux is pointwise non-negative",
               c["pointwise_flux_nonnegative"])


# ---------------------------------------------------------------------------
# [6] the vanishing theorem on the contour packet
# ---------------------------------------------------------------------------

def test_vanishing():
    banner("[6] I_2 = 0: contour-exact, on a different packet")

    for s in (0.5, 0.8, 2.0):
        p = N.ContourPacket(s=s)
        i1n, i2n = p.i1_i2_numeric()
        check_close("s=%.1f I_1 matches 3pi/(2 k s^5)" % s,
                    i1n, p.i1_exact(), 1e-8)
        check_true("s=%.1f |I_2|/I_1 below 1e-12 (contour: exactly 0)" % s,
                   abs(i2n) / p.i1_exact() < 1e-12)

    # The mechanism: the modular weight is the boost Jacobian, (62).
    p = N.ContourPacket(s=0.8)
    check_true("boost-Jacobian identity to 1e-12",
               p.boost_jacobian_check()["rel_diff"] < 1e-12)

    # The dichotomy: 0 for wedge-adapted data, 1 for boost-blind data, so
    # the ratio is a binary wedge-locality detector -- and tanh r >= 1 being
    # unreachable, no wedge-local squeeze of a real mode exists.
    check("real packet has |I_2|/I_1", p.real_packet_ratio(), 1.0)
    check_true("   which exceeds tanh r for every finite r",
               p.real_packet_ratio() > math.tanh(50.0) - 1e-15)


# ---------------------------------------------------------------------------
# [7] the anti-evaporation budget
# ---------------------------------------------------------------------------

def test_budget():
    banner("[7] negativity windows: real, bounded, exactly prepaid")

    totals = []
    for theta in (0.0, 1.0, math.pi / 2, math.pi):
        b = N.squeezed_flux_budget(0.5, theta)
        check_true("th=%.2f windows exist and are funded" % theta,
                   b["windows_exist"] and b["budget_respected"])
        check_close("th=%.2f total is the thermal value" % theta,
                    b["total"], b["prediction"], 1e-8)
        totals.append(b["total"])
    # theta moves the negativity around but not the total: I_2 = 0 at work.
    check_true("the total is theta-independent",
               max(totals) - min(totals) < 1e-8 * totals[0])

    # The running ledger: R(-inf) is the thermal total, dips exist, and no
    # dip exceeds the funded depth. This is the strongest statement the
    # kinematics admits about the teleological-horizon problem.
    rl = N.running_ledger(0.5, math.pi)
    check_close("R(-inf) = 2 sinh^2 r I_1", rl["total"], rl["prediction"],
                1e-8)
    check_true("dips exist", rl["largest_dip"] > 0.0)
    check_true("every dip within the budget", rl["dip_within_budget"])


# ---------------------------------------------------------------------------
# [8] the universal ratio law
# ---------------------------------------------------------------------------

def test_ratio_law():
    banner("[8] the funded-depth ratio is a function of r alone")

    # Found by this module's cross-checks: the ratio D(W)/positive-part is
    # packet- and angle-independent at fixed r, equal to the phase-average
    # closed form, and reproduces the paper's Gaussian-packet numbers.
    for r in (0.3, 0.5, 1.0, 1.5):
        want = N.universal_negativity_ratio(r)
        for s in (0.5, 2.0):
            for theta in (0.0, 1.0, math.pi):
                b = N.squeezed_flux_budget(r, theta, N.ContourPacket(s=s))
                got = b["funded_depth"] / b["positive_part"]
                if abs(got - want) > 2e-5 * max(1.0, want):
                    check_close("r=%.1f s=%.1f th=%.2f ratio"
                                % (r, theta, s), got, want, 2e-5)
        check_close("r=%.1f ratio = phase-average closed form" % r,
                    N.squeezed_flux_budget(
                        r, 1.0, N.ContourPacket(s=0.8))["funded_depth"]
                    / N.squeezed_flux_budget(
                        r, 1.0, N.ContourPacket(s=0.8))["positive_part"],
                    want, 2e-5)

    # The paper's Section 9.2 values, from its Gaussian packet, are the same
    # function: 0.209 at r = 0.5 and 0.044 at r = 1.0.
    check_close("paper's 0.209 at r = 0.5",
                N.universal_negativity_ratio(0.5), 0.209, 2e-3)
    check_close("paper's 0.044 at r = 1.0",
                N.universal_negativity_ratio(1.0), 0.044, 1e-2)

    # Limits: unsqueezed states have no flux to be negative (ratio -> 1 is
    # the empty statement), deep squeezes are almost all particles.
    check_true("monotone decreasing in r",
               all(N.universal_negativity_ratio(r1)
                   > N.universal_negativity_ratio(r2)
                   for r1, r2 in [(0.1, 0.3), (0.3, 1.0), (1.0, 2.0)]))
    check_true("-> 0 for deep squeezes",
               N.universal_negativity_ratio(5.0) < 1e-3)

    # The sharp bound from I_2 = 0 alone: D(W) <= e^{-2r} * positive part,
    # by the convex-majorant argument. Strictly above the phase-average
    # value, respected by every packet and angle, and exponentially
    # stronger than the paper's budget D <= P.
    for r in (0.3, 0.5, 1.0, 1.5):
        check_true("bound e^{-2r} lies above the phase-average value r=%.1f"
                   % r,
                   N.negativity_bound(r) > N.universal_negativity_ratio(r))
        for theta in (0.0, 1.0, math.pi):
            b = N.squeezed_flux_budget(r, theta, N.ContourPacket(s=0.7))
            check_true("D <= e^{-2r} P at r=%.1f th=%.2f" % (r, theta),
                       b["funded_depth"]
                       <= N.negativity_bound(r) * b["positive_part"])

    # The mechanism: I_2 = 0 is the vanishing of the first circular moment
    # of the phase distribution; the contour packet additionally has its
    # low harmonics killed by the 6-fold winding (sum_k cos^4(x + k pi/3)
    # is constant), so its phase distribution is exactly uniform and its
    # ratio is exactly the phase-average value.
    p = N.ContourPacket(s=0.8)
    u = np.linspace(-300.0, 300.0, 600001)
    moms = N.circular_moments(p.dchi(u), u, mmax=4)
    check_true("first circular moment vanishes (= the vanishing theorem)",
               moms[0] < 1e-10)
    check_true("higher moments vanish too (exact uniformity)",
               max(moms[1:]) < 1e-8)
    x = np.linspace(0.0, 2.0 * math.pi, 4001)
    ident = sum(np.cos(x + k * math.pi / 3.0) ** 4 for k in range(3))
    check_true("sum_k cos^4(x + k pi/3) = 9/8 identically",
               float(np.max(np.abs(ident - 9.0 / 8.0))) < 1e-12)


# ---------------------------------------------------------------------------
# [9] the one-mode sum rule and the tower
# ---------------------------------------------------------------------------

def test_one_mode():
    banner("[9] S_rel = beta dE at general angle, and the mode tower")

    for w, r in ((0.5, 0.3), (1.0, 0.8), (2.0, 1.5)):
        base = None
        for theta in (0.0, math.pi / 3, math.pi):
            m = N.one_mode_sum_rule(w, r, theta)
            check_close("w=%.1f r=%.1f th=%.2f S_rel = beta dE"
                        % (w, r, theta), m["S_rel"], m["beta_dE"], 1e-12)
            check_close("   and the closed form (66)",
                        m["S_rel"], m["closed_form"], 1e-12)
            check_true("   with dS_vN = 0 and nu invariant",
                       abs(m["delta_S_vN"]) < 1e-12
                       and m["nu_invariant"] < 1e-12)
            if base is None:
                base = m["S_rel"]
            else:
                check_close("   theta drops out", m["S_rel"], base, 1e-12)

    # Additivity, and the inequivalence of the two convergence conditions:
    # a tower with sinh^2 r_k = 1/k^2 and omega_k = k^2 is Shale-summable
    # while its entropy partial sums grow linearly -- an implementable
    # squeeze of infinite relative entropy, the ledger recording infinity.
    modes = [(1.0, 0.5), (2.0, 0.3), (3.0, 0.1)]
    total = N.lifted_sum_rule(modes)["S_rel"]
    parts = sum(N.lifted_sum_rule([m])["S_rel"] for m in modes)
    check_close("the sum rule is additive over modes", total, parts, 1e-12)

    def tower(K):
        return [(float(k * k), math.asinh(1.0 / k)) for k in range(1, K + 1)]
    s100 = N.lifted_sum_rule(tower(100))
    s200 = N.lifted_sum_rule(tower(200))
    check_true("Shale sum converges (sum 1/k^2)",
               s200["shale_sum"] - s100["shale_sum"] < 0.006)
    check_true("entropy partial sums grow without bound",
               s200["S_rel"] - s100["S_rel"] > 0.9 * (s100["S_rel"] / 100)
               * 100 * 0.5)
    check_true("   at roughly linear rate",
               s200["S_rel"] / s100["S_rel"] > 1.8)


# ---------------------------------------------------------------------------
# [10] the categories
# ---------------------------------------------------------------------------

def test_categories():
    banner("[10] the right refusals, and the theory object")

    m = N.NariaiEntropic()

    # The von Neumann entropy is undefined on a type III factor: a
    # structural absence, so a NoSuchTheory subclass, never a NeedsMetric.
    check_true("von_neumann_entropy raises TypeIIIFactor",
               _raises(N.TypeIIIFactor, m.von_neumann_entropy))
    check_true("TypeIIIFactor is a NoSuchTheory",
               issubclass(N.TypeIIIFactor, NoSuchTheory))
    check_true("   and is not a NeedsMetric",
               not issubclass(N.TypeIIIFactor, NeedsMetric))

    # There is no Yukawa sector: not a compactification, nothing to couple.
    check_true("holomorphic_yukawa raises NoSuchTheory",
               _raises(NoSuchTheory, m.holomorphic_yukawa))

    # The ledger equation assembles finite columns with interference zero.
    led = m.ledger(coherent=[(1, 0.1), (2, 0.05)], squeezes=[(1.0, 0.8)])
    check("interference column is exactly zero", led["interference"], 0.0)
    check_close("the columns add to the total",
                led["total"],
                led["capacity"] + led["classical_events"]
                + led["quantum_events"], 1e-12)
    check_true("every column is finite",
               all(math.isfinite(v) for v in led.values()))

    # Branch selection: the coherent sector cannot shrink a pierced cut;
    # squeezed windows exist and stay inside the prepaid budget.
    b = m.branch_selection()
    check_true("coherent flux cannot shrink the pierced cut",
               not b["coherent_can_shrink_pierced_cut"])
    check_true("anti-evaporation certifies non-coherence",
               b["anti_evaporation_needs_noncoherence"])
    check_true("squeezed windows exist, total positive, budget held",
               b["squeezed_windows_exist"] and b["squeezed_total_positive"]
               and b["budget_respected"])

    # Registration and reporting.
    check_true("registered under its key",
               T.get("nariai-entropic") is N.NariaiEntropic)
    check_true("Lambda <= 0 refused",
               _raises(ValueError, N.NariaiEntropic, 0.0))
    try:
        m.describe()
        check_true("describe() runs", True)
    except Exception as e:                                       # noqa: BLE001
        check_true("describe() runs (%s)" % e, False)
    check_true("the open problems name the type III lift",
               any("type III" in x for x in m.missing_for_physical()))
    check_true("   and the microstate question",
               any("microstate" in x for x in m.missing_for_physical()))


def main():
    t0 = time.time()
    test_geometry()
    test_nariai()
    test_rational()
    test_routes()
    test_clausius()
    test_vanishing()
    test_budget()
    test_ratio_law()
    test_one_mode()
    test_categories()

    print("\n" + "=" * 72)
    if FAILURES:
        print("FAILED (%d): %s" % (len(FAILURES), ", ".join(FAILURES)))
        return 1
    print("test_nariai: all checks passed in %.1fs" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
