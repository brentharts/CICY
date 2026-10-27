"""
Tests for pyCICY.sixsphere_theta (Part XI: a theta function whose multiplier
is the holonomy character chi of Part IX).

  [1] multiplier    F(g x) = e(chi(g)/12) F(x), no automorphy factor, on the
                    generators of G(Z), on T1, T2 and on random words
  [2] alpha         Euclid (S/T word) against the eta^2 multiplier
  [3] index 12      <T1, T2> fixes F; g fixes F iff chi(g) = 0; F^12 is
                    G(Z)-invariant; z^k reach all 12 multipliers
  [4] uniqueness    80 nearby candidates, only theta_1/eta with e(beta/12)
                    is fixed by T1 and T2 (the 76 others are the control)
  [5] zeros         F = 0 on mu in Z + tau Z, nonzero at the half periods
  [6] proof         classical theta/eta laws + the beta shift simplify to
                    the constant chi/12 on the 5 generators (exact); the laws
                    themselves against the series; control without the shift

Run with:  python3 tests/test_sixsphere_theta.py   or   python3 run_tests.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyCICY import sixsphere_theta as T

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


def test_multiplier():
    print("\n[1] the multiplier of F is chi")
    mc = T.multiplier_check(n_words=24)
    want = {"L(S)": 9, "L(T)": 1, "h(1,0,0)": 6, "h(0,1,0)": 6, "z": 1,
            "T1": 0, "T2": 0}
    for k, v in mc["generators"].items():
        check("chi(%s)" % k, v["chi"], want[k])
        check_true("F(%s x) / F(x) = e(chi/12), modulus 1" % k, v["ok"])
    check("random words agreeing", mc["words_ok"], mc["words"])


def test_alpha():
    print("\n[2] alpha two ways")
    at = T.alpha_two_ways(n=24)
    check("Euclid = eta^2 multiplier on random SL(2,Z) words", at["agree"], at["of"])
    check("alpha(S) from eta^2", T.alpha_eta2(T.G._ST)[0], 9)
    check("alpha(T) from eta^2", T.alpha_eta2(T.G._TT)[0], 1)


def test_index():
    print("\n[3] index 12 as an invariance group")
    sh = T.stabiliser_is_holonomy()
    check_true("every word in T1, T2 fixes F", sh["holonomy_fixes_F"])
    check_true("g fixes F iff chi(g) = 0", sh["fixed_iff_chi_zero"])
    check_true("F^12 invariant under G(Z)", sh["F12_invariant"])
    check("multipliers of z^k, k = 0..11", sh["cosets"], list(range(12)))


def test_uniqueness():
    print("\n[4] uniqueness among 80 candidates")
    cs = T.candidate_search()
    check("candidates", cs["candidates"], 80)
    check("survivors (12k, sign of mu, eta power, theta index)",
          sorted(cs["survivors"]), [(1.0, -1, -1, 1), (1.0, 1, -1, 1)])
    check_true("control: e(beta/6) theta_1/eta is not fixed by T1",
               not T.wrong_index_control()["fixed_by_T1"])


def test_zeros():
    print("\n[5] the zero divisor")
    zd = T.zero_divisor()
    check_true("|F| < 1e-15 on mu in Z + tau Z", zd["max_on_lattice"] < 1e-15)
    check_true("|F| > 1e-3 at the three half periods",
               zd["min_at_half_periods"] > 1e-3)


def test_proof():
    print("\n[6] the proof by generators")
    pr = T.proof_by_generators()
    want = {"L(S)": "-1/4", "L(T)": "1/12", "h(1,0,0)": "1/2",
            "h(0,1,0)": "1/2", "z": "1/12"}
    for k, v in pr.items():
        check("exponent for %s (exact)" % k, str(v["exponent"]), want[k])
        check_true("12 * exponent = chi(%s) mod 12" % k, v["ok"])
    check_true("classical theta/eta laws against the series",
               T.classical_laws_check()["ok"])
    for k in ("L(S)", "h(0,1,0)"):
        check_true("control: %s exponent without the beta shift is not constant" % k,
                   T.simplify(T._classical_exponent(k, T.S.TAU, T.S.MU)).free_symbols)


def main():
    for section in (test_multiplier, test_alpha, test_index, test_uniqueness,
                    test_zeros, test_proof):
        try:
            section()
        except Exception as exc:
            print("  {:<58} {:>12} FAIL".format(section.__name__ + " raised",
                                                type(exc).__name__))
            print("      " + str(exc)[:300])
            FAILURES.append("%s raised %s" % (section.__name__,
                                              type(exc).__name__))
    print()
    if FAILURES:
        print("FAILED: %d" % len(FAILURES))
        for f in FAILURES:
            print("  - " + f)
        return 1
    print("all sixsphere_theta tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
