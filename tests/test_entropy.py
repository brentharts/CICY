"""
Tests for pyCICY.entropy: D4-D0 black holes, Cardy against the attractor.

  [1] exact data       D, c_L, c_R, c_2.P by hand on the quintic (known
                       c_L = 55, c_R = 30 for P = H) and exact identities
  [2] attractor        minimising |Z|^2 e^K with centralcharge.z_ordinary
                       reproduces (S/pi)^2 = 4 D q_0, on three threefolds,
                       from several random starts
  [3] corrections      Cardy / leading = 1 + c_2.P/(6D) exactly; the
                       Todd-corrected attractor gives 4 D (q_0 - c_2.P/24);
                       the two corrections differ
  [4] scaling          P -> kP gives delta -> delta/k^2 and the ratio -> 1
  [5] controls         wrong normalisations and bad charges break or are
                       refused
  [6] published list   c_2 against data/cicylist.json on a random sample

Run with:  python3 tests/test_entropy.py   or   python3 run_tests.py
"""

import math
import os
import sys
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyCICY import centralcharge as CC
from pyCICY import entropy as En

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


QUINTIC = CC.CY3.from_cicy([[4, 5]])
M7833 = CC.CY3.from_cicy([[2, 2, 1], [3, 1, 3]])
TETRA = CC.CY3.from_cicy([[1, 2]] * 4)
CASES = [("quintic", QUINTIC, [2], 200), ("7833", M7833, [2, 3], 500),
         ("tetraquadric", TETRA, [1, 2, 1, 2], 900)]


def close(a, b, rel=1e-8):
    return abs(a - b) <= rel * max(abs(a), abs(b), 1.0)


def test_exact():
    print("\n[1] exact data")
    d = En.d4d0(QUINTIC, [1], 100)
    check("quintic, P = H: D = 5/6", d["D"], Fraction(5, 6))
    check("quintic, P = H: c_L = 55", d["cL"], 55)
    check("quintic, P = H: c_R = 30", d["cR"], 30)
    check("quintic, P = H: c_2.P = 50", d["c2P"], 50)
    check("quintic, P = H: delta = 10", d["delta"], 10)
    bad = []
    for name, X, P, q in CASES:
        d = En.d4d0(X, P, q)
        if d["cL"] != 6 * d["D"] + d["c2P"] or \
                d["cR"] != 6 * d["D"] + d["c2P"] / 2:
            bad.append(name)
        if d["s2_leading"] != 4 * d["D"] * q:
            bad.append(name + " leading")
        if d["s2_cardy"] != 4 * d["cL"] * q / 6:
            bad.append(name + " cardy")
    check("c_L, c_R and the (S/pi)^2 formulas, three CICYs: bad", bad, [])
    check_true("S = pi sqrt((S/pi)^2) for both entropies",
               close(d["S_leading"], math.pi * math.sqrt(float(d["s2_leading"])))
               and close(d["S_cardy"],
                         math.pi * math.sqrt(float(d["s2_cardy"]))))


def test_attractor():
    print("\n[2] the attractor reproduces (S/pi)^2 = 4 D q_0")
    for name, X, P, q in CASES:
        d = En.d4d0(X, P, q)
        a = En.attractor(X, P, q)
        check_true("%s: attractor^2 = 4 D q0 (%.3f vs %.3f)"
                   % (name, a["s2"], float(d["s2_leading"])),
                   close(a["s2"], float(d["s2_leading"]), 1e-7))
        check_true("%s: six starts agree (spread %.1e)" % (name, a["spread"]),
                   a["spread"] < 1e-6)
    # the attractor sits at w proportional to P for D4-D0 (q_A = 0)
    a = En.attractor(M7833, [2, 3], 500)
    check_true("7833: minimiser w is proportional to P = (2,3)",
               close(a["omega"][1] / a["omega"][0], 1.5, 1e-5))


def test_corrections():
    print("\n[3] the two corrections")
    for name, X, P, q in CASES:
        d = En.d4d0(X, P, q)
        check("%s: Cardy/leading = 1 + c2.P/(6D) exactly" % name,
              d["ratio_cardy_over_leading"], 1 + d["delta"])
        b = En.attractor(X, P, q, sqrt_td=True)
        check_true("%s: Todd attractor^2 = 4 D (q0 - c2.P/24)" % name,
                   close(b["s2"], float(d["s2_todd_shift"]), 1e-7))
        check_true("%s: Cardy and Todd values differ" % name,
                   d["s2_cardy"] != d["s2_todd_shift"])


def test_scaling():
    print("\n[4] large charge: P -> kP")
    deltas, ratios = [], []
    for k in (1, 2, 4, 8, 16):
        d = En.d4d0(QUINTIC, [k], 10 ** 6)
        deltas.append(d["delta"])
        ratios.append(d["ratio_cardy_over_leading"])
    check("quintic delta(k) = 10 / k^2 for k = 1,2,4,8,16", deltas,
          [Fraction(10, k * k) for k in (1, 2, 4, 8, 16)])
    check_true("ratio decreases monotonically towards 1",
               all(a > b for a, b in zip(ratios, ratios[1:]))
               and ratios[-1] < Fraction(104, 100))
    d = En.d4d0(QUINTIC, [16], 10 ** 8)       # c_L = 21280
    check("regime q0/c_L at P = 16 H, q0 = 10^8", d["regime"],
          Fraction(10 ** 8, 21280))
    check_true("... which is well inside the Cardy regime", d["regime"] > 100)


def test_list_sample():
    print("\n[6] a sample of the published CICY list")
    import json
    import random
    path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "data", "cicylist.json")
    entries = [e for e in json.load(open(path))["entries"]
               if e["h11"] == len(e["conf"])]
    rng = random.Random(9)
    sample = rng.sample(entries, 40)
    bad_c2, bad_delta, deltas = [], [], []
    for e in sample:
        X = CC.CY3.from_cicy(e["conf"])
        if [Fraction(x) for x in e["c2"]] != X.c2:
            bad_c2.append(e["num"])
        d = En.d4d0(X, [1] * X.h, 1)
        deltas.append(d["delta"])
        if d["delta"] <= 0 or d["cL"] != 6 * d["D"] + d["c2P"]:
            bad_delta.append(e["num"])
    check("c_2 computed here equals the published c_2 (40 CICYs): bad",
          bad_c2, [])
    check("delta > 0 and c_L consistent at P = sum J_r: bad", bad_delta, [])
    check_true("delta at the unit class stays above 0.1 (full survey: "
               "min 0.1007 over 4871)", min(deltas) > 0.1)


def test_controls():
    print("\n[5] controls: each check made to fail on purpose")
    d = En.d4d0(QUINTIC, [2], 200)
    a = En.attractor(QUINTIC, [2], 200)
    check_true("a wrong normalisation S = A/2 would miss the attractor",
               not close(2 * a["s2"], float(d["s2_leading"]), 1e-3))
    check_true("without the 4/3 in e^{-K} the attractor changes",
               not close(a["s2"] * 3 / 4, float(d["s2_leading"]), 1e-3))
    check_true("q0 <= 0 refused", _raises(En.NotCardyRegime, En.d4d0,
                                          QUINTIC, [1], 0))
    check_true("D <= 0 refused (P = 0)", _raises(Exception, En.d4d0,
                                                 QUINTIC, [0], 5))
    check_true("a non-ample P is refused",
               _raises(CC.NotAmple, En.d4d0, M7833, [-1, 2], 5))
    check_true("floats are refused", _raises(TypeError, En.d4d0,
                                             QUINTIC, [1.5], 5))
    b = En.attractor(QUINTIC, [2], 200, sqrt_td=True)
    check_true("Todd attractor is not the leading one",
               not close(b["s2"], a["s2"], 1e-4))


def main():
    for section in (test_exact, test_attractor, test_corrections,
                    test_scaling, test_list_sample, test_controls):
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
    print("all entropy tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
