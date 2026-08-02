"""
Tests for pyCICY.smoothness and the published-list handling in
pyCICY.cicylist.

The smoothness criterion is the Jacobian rank condition of CIPro section 2.2
(arXiv:2606.27588), checked exhaustively over finite fields. Section [2]
tests the two traps that approach falls into, because both produce a
confidently wrong answer rather than an error:

  * a prime dividing a degree makes every point look singular;
  * a single random draw over a small field need not be generic.

Run with:  python3 tests/test_smoothness.py
       or: python3 run_tests.py  (runs every suite)
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyCICY import cicylist as L
from pyCICY import smoothness as S

FAILURES = []


def check(name, got, want):
    ok = got == want
    print("  {:<58} {:>10} {}".format(name, str(got)[:10],
                                      "ok" if ok else "FAIL want " + str(want)))
    if not ok:
        FAILURES.append(name)


def check_true(name, cond):
    print("  {:<58} {:>10} {}".format(name, str(bool(cond)),
                                      "ok" if cond else "FAIL"))
    if not cond:
        FAILURES.append(name)


# --------------------------------------------------------------------------
print("\n[1] Coordinates and monomials")
dp2 = [[2, 1, 1], [1, 1, 0], [1, 0, 1]]
dims, slices, n_vars = S.coordinate_layout(dp2)
check("dimensions", dims, [2, 1, 1])
check("coordinate count", n_vars, 7)
check("slices", slices, [(0, 3), (3, 5), (5, 7)])
check("names", S.default_names(dp2),
      ["x0", "x1", "x2", "y0", "y1", "z0", "z1"])

# Monomials of multidegree (1,1,0): 3 choices in P^2 times 2 in the first P^1.
mons = S.monomials(dp2, [1, 1, 0])
check("monomials of multidegree (1,1,0)", len(mons), 6)
check_true("x0 y0 is among them", (1, 0, 0, 1, 0, 0, 0) in mons)
check("rendered monomial", S.monomial_string((1, 0, 0, 1, 0, 0, 0),
                                             S.default_names(dp2)), "x0 y0")
# Quintic: monomials of degree 5 in 5 variables is binom(9,4) = 126.
check("quintic degree-5 monomials", len(S.monomials([[4, 5]], [5])), 126)

# --------------------------------------------------------------------------
print("\n[2] The two traps of finite-field testing")
quintic = [[4, 5]]
fermat = {tuple(5 if j == i else 0 for j in range(5)): 1 for i in range(5)}

# Trap one: p | degree makes derivatives vanish identically.
bad = S.is_smooth(quintic, equations=[fermat], primes=(5,),
                  allow_bad_primes=True)
check_true("Fermat quintic looks singular over F_5", bad["smooth"] is False)
refused = S.is_smooth(quintic, equations=[fermat], primes=(5,))
check_true("that prime is refused by default", refused["smooth"] is None)
check_true("and the reason is recorded",
           "divides the degrees" in refused["per_prime"][5]["skipped"])
good = S.is_smooth(quintic, equations=[fermat], primes=(7, 11))
check_true("Fermat quintic is smooth over F_7 and F_11", good["smooth"])

# Trap two: one random draw over a small field need not be generic.
singular_draws = 0
for seed in range(12):
    eqs = S.random_equations(dp2, 5, seed=seed)
    if S.singular_points(dp2, eqs, 5, limit=1)["singular"]:
        singular_draws += 1
check_true("some random members over F_5 are genuinely singular",
           singular_draws > 0)
check_true("but the generic member is still found smooth",
           S.is_smooth(dp2, primes=(5, 7), trials=6)["smooth"])

# --------------------------------------------------------------------------
print("\n[3] The CIPro section 2.2 example")
# dp2 with s1 = x0 y0 and s2 = x0 z1 is singular: both vanish identically on
# x0 = 0, and there the two differentials are proportional.
s1 = {(1, 0, 0, 1, 0, 0, 0): 1}          # x0 y0
s2 = {(1, 0, 0, 0, 0, 0, 1): 1}          # x0 z1
result = S.is_smooth(dp2, equations=[s1, s2], primes=(5, 7))
check_true("CIPro's dP2 example is singular", result["smooth"] is False)
witness = result["singular_example"]
check_true("a witness point is returned", witness is not None)
check("the witness has x0 = 0", witness[0], 0)

# --------------------------------------------------------------------------
print("\n[4] Deliberately singular and generic families")
# A quintic missing one variable is singular at the corresponding point.
degenerate = {tuple(5 if j == i else 0 for j in range(5)): 1 for i in range(4)}
res = S.is_smooth(quintic, equations=[degenerate], primes=(7,))
check_true("quintic without x4 is singular", res["smooth"] is False)
check("singular exactly at [0:0:0:0:1]", res["singular_example"],
      (0, 0, 0, 0, 1))

for name, conf in [("quintic", [[4, 5]]),
                   ("bicubic", [[2, 3], [2, 3]]),
                   ("tetraquadric", [[1, 2], [1, 2], [1, 2], [1, 2]]),
                   ("split quintic", [[1, 1, 1], [4, 1, 4]])]:
    r = S.is_smooth(conf, primes=(7,), trials=4)
    check_true("generic %s is smooth" % name, r["smooth"])
    check_true("%s: not claimed as proof" % name, r["proof"] is False)

# The budget is enforced rather than running unbounded.
try:
    S.singular_points(quintic, [fermat], 7, max_points=10)
    check_true("point budget enforced", False)
except ValueError as exc:
    check_true("point budget enforced", "budget" in str(exc))

# --------------------------------------------------------------------------
print("\n[5] Published list: parsing and comparison")
# Three real entries from the Oxford list, including number 14, which is the
# Schoen manifold that the splitting construction finds independently.
SAMPLE = [
    {"num": 1,
     "conf": [[1, 1, 1, 0, 0, 0, 0], [1, 0, 0, 1, 0, 0, 1],
              [1, 0, 0, 0, 0, 1, 1], [1, 1, 0, 0, 1, 0, 0],
              [1, 1, 0, 0, 0, 0, 1], [2, 0, 0, 1, 2, 0, 0],
              [2, 0, 1, 0, 0, 2, 0]],
     "h11": 15, "h21": 15, "euler": 0},
    {"num": 14, "conf": [[1, 1, 1], [2, 0, 3], [2, 3, 0]],
     "h11": 19, "h21": 19, "euler": 0},
    {"num": 31, "conf": [[2, 3, 0], [3, 0, 4]],
     "h11": 0, "h21": 0, "euler": 0},
]

tmp = tempfile.mkdtemp(prefix="pycicy-pub-")
path = os.path.join(tmp, "cicylist.json")
with open(path, "w") as fh:
    json.dump({"entries": SAMPLE, "count": len(SAMPLE)}, fh)

entries = L.load_published_list(path)
check("entries loaded", len(entries), 3)

report = L.compare_to_published(entries)
check("checked", report["checked"], 2)
check("agree", report["agree"], 2)
check("disagreements", len(report["disagree"]), 0)
check("errors", len(report["errors"]), 0)
# Entry 31 is a direct product, recorded with the placeholder h11 = h21 = 0.
check("direct products skipped", report["skipped_products"], 1)

# Without skipping, the placeholder entry must be reported rather than
# silently counted as agreeing.
report2 = L.compare_to_published(entries, skip_products=False)
check("without skipping, all three are checked", report2["checked"], 3)

# A missing file must explain where the data comes from.
try:
    L.load_published_list(os.path.join(tmp, "absent.json"))
    check_true("missing file explained", False)
except FileNotFoundError as exc:
    check_true("missing file explained",
               "fetch_cicy_list" in str(exc))

# The Schoen manifold is reachable by splitting, so coverage must see it.
web = L.split_web(max_depth=2, max_configs=400)
cov = L.published_coverage(entries, web)
check("published entries considered", cov["published"], 3)
check_true("the Schoen manifold is covered by the web", cov["in_both"] >= 1)

import shutil
shutil.rmtree(tmp, ignore_errors=True)

# --------------------------------------------------------------------------
print("\n[6] The fetch script parses the published format")
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import fetch_cicy_list as F

RAW = """Num    : 14
NumPs  : 3
NumPol : 2
Eta    : 0
H11    : 19
H21    : 19
C2     : {0, 36, 36}
Redun  : {0, 0, 0, 0, 0}
{1, 1}
{0, 3}
{3, 0}

Num    : 31
NumPs  : 2
NumPol : 2
Eta    : 0
H11    : 0
H21    : 0
C2     : {72, 0}
Redun  : {0, 0, 0, 0, 0}
{3, 0}
{0, 4}
"""
parsed = F.parse(RAW)
check("blocks parsed", len(parsed), 2)
# The source omits the projective space dimensions; they are reconstructed
# from the Calabi-Yau condition n_i = sum_a q^i_a - 1.
check("dimensions reconstructed", parsed[0]["conf"],
      [[1, 1, 1], [2, 0, 3], [2, 3, 0]])
check("Hodge numbers read", (parsed[0]["h11"], parsed[0]["h21"]), (19, 19))
check("Euler read", parsed[0]["euler"], 0)
check("c2 read from the header, not the matrix", parsed[0]["c2"], [0, 36, 36])
check("second block", parsed[1]["conf"], [[2, 3, 0], [3, 0, 4]])

problems = F.validate(parsed)
check_true("a two-entry sample fails the 7890 count check",
           any("7890" in p or "expected" in p for p in problems))

# --------------------------------------------------------------------------
print("\n" + "=" * 72)
if FAILURES:
    print("FAILED ({}): {}".format(len(FAILURES), ", ".join(FAILURES)))
    sys.exit(1)
print("ALL TESTS PASSED on Python {}".format(sys.version.split()[0]))
