"""
Tests for pyCICY.additivity.

The questions here are prompted by Brittenham and Hermiller, "Unknotting
number is not additive under connected sum", arXiv:2506.24088. Unknotting
number is a minimal move count, and they show the count for a composite is
not determined by its factors. The split web has the same shape -- objects,
an elementary move, distinguished trivial objects -- so the analogous
questions can be asked and, unlike in knot theory, answered exhaustively.

The answers are checked here rather than assumed:

  [1] split depth is forced to be K - 1, so no path dependence is possible;
  [2] every contraction path has that same length and greedy is optimal;
  [3] the total node count along a chain is fixed by chi, so it cannot vary;
  [4] the decomposition of that total *does* vary between chains.

Run with:  python3 tests/test_additivity.py
       or: python3 run_tests.py  (runs every suite)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyCICY import additivity as A
from pyCICY import cicylist as L
from pyCICY import transitions as T

FAILURES = []


def check(name, got, want):
    ok = got == want
    print("  {:<56} {:>12} {}".format(name, str(got)[:12],
                                      "ok" if ok else "FAIL want " + str(want)))
    if not ok:
        FAILURES.append(name)


def check_true(name, cond):
    print("  {:<56} {:>12} {}".format(name, str(bool(cond)),
                                      "ok" if cond else "FAIL"))
    if not cond:
        FAILURES.append(name)


WEB = L.split_web(max_depth=3, max_configs=1200)

# --------------------------------------------------------------------------
print("\n[1] Split depth is forced by the shape of the matrix")
check("quintic is a seed", A.split_depth([[4, 5]]), 0)
check("its split is one move away", A.split_depth([[1, 1, 1], [4, 1, 4]]), 1)
check("every seed has depth 0",
      sorted({A.split_depth(s) for s in L.SEEDS}), [0])

bad = [r for r in L.web_nodes(WEB)
       if A.split_depth(r["conf"]) != r["depth"]]
check("configurations where depth != K-1", len(bad), 0)
check_true("a useful number were checked", len(L.web_nodes(WEB)) > 500)

# --------------------------------------------------------------------------
print("\n[2] Contraction paths: greedy is optimal, and every path is equal")
seeds = A.seed_keys()
check("five seeds", len(seeds), 5)

mismatched = greedy_suboptimal = stuck = spread = 0
sample = [r for r in L.web_nodes(WEB) if r["depth"] <= 3][:120]
for rec in sample:
    info = A.contraction_paths(rec["conf"])
    if info["stuck"]:
        stuck += 1
        continue
    if info["min"] != rec["depth"]:
        mismatched += 1
    if info["greedy"] != info["min"]:
        greedy_suboptimal += 1
    if info["max"] != info["min"]:
        spread += 1

check("contraction paths that get stuck", stuck, 0)
check("minimum length != split depth", mismatched, 0)
check("greedy worse than optimal", greedy_suboptimal, 0)
check("paths of differing length", spread, 0)
check_true("a sample was actually examined", len(sample) >= 50)

# There is no Bernhard-Jablan style failure available here: greedy cannot go
# wrong because all routes have the same length by construction.
info = A.contraction_paths([[1, 0, 1, 1], [1, 1, 0, 1], [4, 1, 2, 2]])
check("this configuration has several contraction routes",
      info["paths"] > 1, True)
check("all of them take the same number of moves", info["min"], info["max"])

# --------------------------------------------------------------------------
print("\n[3] The total node count is fixed by the Euler characteristic")
survey = A.survey(WEB)
check("configurations whose totals disagree",
      len(survey["total_mismatches"]), 0)
check_true("many configurations were surveyed", survey["checked"] > 500)

# Spot check the telescoping directly: total N along a chain must equal
# (chi(X_R) - chi(X_D)) / 2 with X_D a seed.
from pyCICY import cache as C
checked = 0
for key, rec in list(WEB["nodes"].items()):
    if key in seeds or rec.get("euler") is None:
        continue
    ms = A.decompositions(WEB, key)
    if not ms:
        continue
    total = sum(next(iter(ms)))
    # every chain starts at some seed; the seed's chi must make this work out
    seed_eulers = {WEB["nodes"][k]["euler"] for k in seeds}
    if not any(rec["euler"] - e == 2 * total for e in seed_eulers):
        FAILURES.append("telescoping %s" % (rec["conf"],))
        print("   telescoping failed for", rec["conf"])
    checked += 1
    if checked >= 60:
        break
check_true("telescoping verified on a sample", checked >= 40)

# --------------------------------------------------------------------------
print("\n[4] The decomposition of that total is NOT fixed")
check_true("some configurations admit several decompositions",
           survey["multiple_decompositions"] > 0)
check_true("and it is a substantial fraction, not a curiosity",
           survey["multiple_decompositions"] > 100)
check_true("single steps differ by a lot in some cases",
           survey["max_single_step_spread"] > 10)

# The published example from the module docstring.
target = None
for key, rec in WEB["nodes"].items():
    if rec["conf"] == [[1, 0, 1, 1], [1, 1, 0, 1], [4, 1, 2, 2]]:
        target = key
        break
check_true("the documented example is in the web", target is not None)
if target is not None:
    ms = sorted(A.decompositions(WEB, target))
    check("its decompositions", ms, [(12, 36), (16, 32)])
    check("both sum to the same total", len({sum(m) for m in ms}), 1)
    check("that total", sum(ms[0]), 48)

# --------------------------------------------------------------------------
print("\n" + "=" * 72)
if FAILURES:
    print("FAILED ({}): {}".format(len(FAILURES), ", ".join(FAILURES)))
    sys.exit(1)
print("ALL TESTS PASSED on Python {}".format(sys.version.split()[0]))
