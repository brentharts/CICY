"""
Tests for pyCICY.cache and pyCICY.cicylist.

The split web is generated the way the CICY threefold list was originally
compiled: from the configurations with a single defining equation, by
repeated P^1 splitting with duplicates removed (Candelas, Dale, Lutken,
Schimmrigk, Nucl. Phys. B298 (1988) 493).

Section [4] checks that landmark manifolds turn up where they should,
including the quintic -> split edge with N = 16 from section 1.1 of
arXiv:2512.18124.

Section [2] is the one that matters for the cache: cached and uncached runs
must produce identical results. A cache that is fast but wrong is worse than
no cache.

Run with:  python3 tests/test_cicylist.py
       or: python3 run_tests.py  (runs every suite)
"""

import os
import shutil
import sys
import tempfile

# Prefer the source tree over any installed copy of pyCICY.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyCICY import cache as C
from pyCICY import cicylist as L
from pyCICY import transitions as T

FAILURES = []


def check(name, got, want):
    ok = got == want
    print("  {:<56} {:>12} {}".format(name, str(got), "ok" if ok else
                                      "FAIL want " + str(want)))
    if not ok:
        FAILURES.append(name)


def check_true(name, cond):
    print("  {:<56} {:>12} {}".format(name, str(bool(cond)),
                                      "ok" if cond else "FAIL"))
    if not cond:
        FAILURES.append(name)


TMP = tempfile.mkdtemp(prefix="pycicy-test-")
CACHE_PATH = os.path.join(TMP, "test.sqlite")

# --------------------------------------------------------------------------
print("\n[1] Cache basics")
cache = C.Cache(path=CACHE_PATH)
check("starts empty", cache.info()["entries"], 0)

r1 = C.hodge([[4, 5]], cache=cache)
check("quintic h^{1,1}", r1["h11"], 1.0)
check("quintic h^{2,1}", r1["h21"], 101.0)
check("quintic euler", r1["euler"], -200)
check("one miss recorded", cache.stats()["misses"], 1)

r2 = C.hodge([[4, 5]], cache=cache)
check("second lookup is a hit", cache.stats()["hits"], 1)
check_true("hit returns the same value", r1 == r2)
check("one entry stored", cache.info()["entries"], 1)

# Keying is on the normal form, so a relabelled configuration must reuse the
# entry rather than recompute.
before = cache.stats()["hits"]
C.hodge([[1, 1, 1], [4, 1, 4]], cache=cache)
C.hodge([[4, 1, 4], [1, 1, 1]], cache=cache)          # rows swapped
C.hodge([[1, 1, 1], [4, 4, 1]], cache=cache)          # columns swapped
check("relabellings share one cache entry",
      cache.stats()["hits"] - before, 2)
check("still only two distinct entries", cache.info()["entries"], 2)

# A disabled cache must always miss and never write.
off = C.Cache(path=CACHE_PATH, enabled=False)
C.hodge([[4, 5]], cache=off)
check("disabled cache misses", off.stats()["misses"], 1)
check("disabled cache writes nothing", off.stats()["writes"], 0)

check("clear empties the cache", cache.clear(), 2)
check("cache is empty after clear", cache.info()["entries"], 0)
cache.close()

# Errors are cached rather than retried on every run.
cache = C.Cache(path=CACHE_PATH)
bad = C.hodge([[1, 1, 1], [1, 1, 1], [1, 1, 1]], cache=cache)  # a CY 1-fold
check_true("unsupported dimension recorded as an error", "error" in bad)
check_true("error mentions the dimension", "fold" in bad["error"])

# --------------------------------------------------------------------------
print("\n[2] Cached and uncached results agree")
warm = C.Cache(path=os.path.join(TMP, "warm.sqlite"))
cold = C.Cache(path=os.path.join(TMP, "cold.sqlite"), enabled=False)
confs = [[[4, 5]], [[2, 3], [2, 3]], [[1, 2], [1, 2], [1, 2], [1, 2]],
         [[1, 1, 1], [4, 1, 4]], [[1, 1, 1], [4, 2, 3]]]
a = [C.hodge(c, cache=warm) for c in confs]
a2 = [C.hodge(c, cache=warm) for c in confs]      # all hits this time
b = [C.hodge(c, cache=cold) for c in confs]
check_true("second pass was entirely hits", warm.stats()["hits"] == len(confs))
check_true("cached == freshly computed", a == b)
check_true("cache is deterministic across passes", a == a2)

# --------------------------------------------------------------------------
print("\n[3] Seeds")
check("five single-equation seeds", len(L.SEEDS), 5)
for s in L.SEEDS:
    info = T.check_configuration(s)
    check_true("%s is a CY threefold" % (s,),
               info["calabi_yau"] and info["dim_X"] == 3
               and info["warnings"] == [])
check_true("every seed has exactly one defining equation",
           all(len(s[0]) - 1 == 1 for s in L.SEEDS))
check_true("no seed is itself a split",
           all(not T.is_contractible(s) for s in L.SEEDS))

# --------------------------------------------------------------------------
print("\n[4] Split web")
web = L.split_web(max_depth=2, max_configs=400,
                  cache=C.Cache(path=os.path.join(TMP, "web.sqlite")))
recs = L.web_nodes(web)
check_true("web is non-trivial", len(recs) > 100)
check_true("no configuration failed to evaluate",
           all(r.get("error") is None for r in recs))
check_true("every configuration is a CY threefold",
           all(T.check_configuration(r["conf"])["dim_X"] == 3
               and T.check_configuration(r["conf"])["calabi_yau"]
               for r in recs))
check_true("seeds are at depth 0",
           sum(1 for r in recs if r["depth"] == 0) == len(L.SEEDS))
check_true("normal forms are unique across the web",
           len({T.canonical_key(r["conf"]) for r in recs}) == len(recs))

pairs = {(r["h11"], r["h21"]) for r in recs}
for name, pair in [("quintic", (1.0, 101.0)),
                   ("split quintic", (2.0, 86.0)),
                   ("bicubic", (2.0, 83.0)),
                   ("tetraquadric", (4.0, 68.0)),
                   ("Schoen (split bicubic)", (19.0, 19.0))]:
    check_true("%s present" % name, pair in pairs)

# The edge of arXiv:2512.18124 section 1.1 must be in the web with N = 16.
qk = T.canonical_key([[4, 5]])
sk = T.canonical_key([[1, 1, 1], [4, 1, 4]])
edge = [e for e in web["edges"] if e["parent"] == qk and e["child"] == sk]
check_true("quintic -> split quintic edge exists", bool(edge))
if edge:
    check("its node count", edge[0]["nodes"], 16)
    check_true("it is an effective split", edge[0]["effective"])

check_true("every edge records a node count",
           all(isinstance(e["nodes"], int) and e["nodes"] >= 0
               for e in web["edges"]))
check_true("effective flag agrees with N > 0",
           all(e["effective"] == (e["nodes"] > 0) for e in web["edges"]))

# All CICY threefolds have chi <= 0; the published list spans [-200, 0].
eulers = [r["euler"] for r in recs]
check_true("no configuration has chi > 0", max(eulers) <= 0)
check_true("chi stays within the published range", min(eulers) >= -200)

# Dropping ineffective splits must not add anything.
lean = L.split_web(max_depth=2, max_configs=400, include_ineffective=False,
                   cache=C.Cache(path=os.path.join(TMP, "lean.sqlite")))
check_true("ineffective splits carry N = 0 only",
           all(e["nodes"] > 0 for e in lean["edges"]))
check_true("excluding them yields no more configurations",
           len(lean["nodes"]) <= len(web["nodes"]))

# --------------------------------------------------------------------------
print("\n[5] Contraction walks back to a seed")
chain = L.contract_to_seed([[1, 1, 1], [4, 1, 4]])
check("chain length", len(chain), 2)
check("chain ends at the quintic", chain[-1], [[4, 5]])
check_true("chain end has no contractible row",
           not T.is_contractible(chain[-1]))

deep = T.split(T.split([[4, 5]], 0, [1]), 1, [0, 1])
chain = L.contract_to_seed(deep)
check("depth-2 chain length", len(chain), 3)
check("depth-2 chain ends at the quintic", chain[-1], [[4, 5]])

check_true("a seed contracts to itself",
           L.contract_to_seed([[4, 5]]) == [[[4, 5]]])

# --------------------------------------------------------------------------
print("\n[6] Survey aggregation")
s = L.survey(web)
check("survey counts every configuration", s["configurations"], len(recs))
check("no failures", s["failed"], 0)
check_true("distinct Hodge pairs counted", s["distinct_hodge_pairs"] == len(pairs))
check_true("depths add up", sum(s["by_depth"].values()) == len(recs))
check_true("h11 within the published range [0, 19]",
           0 <= s["h11_range"][0] and s["h11_range"][1] <= 19)
check_true("h21 within the published range [0, 101]",
           0 <= s["h21_range"][0] and s["h21_range"][1] <= 101)

# --------------------------------------------------------------------------
print("\n[7] List loading and validation")
literal = os.path.join(TMP, "list_literal.txt")
with open(literal, "w") as f:
    f.write("[[4,5]]\n[[1,1,1],[4,1,4]]\n[[2,3],[2,3]]\n")
got = L.load_list(literal)
check("literal format parsed", got, [[[4, 5]], [[1, 1, 1], [4, 1, 4]],
                                     [[2, 3], [2, 3]]])

blocks = os.path.join(TMP, "list_blocks.txt")
with open(blocks, "w") as f:
    f.write("4 5\n\n1 1 1\n4 1 4\n\n2 3\n2 3\n")
check("block format parsed", L.load_list(blocks), got)
check("limit respected", len(L.load_list(literal, limit=2)), 2)

report = L.check_list(got)
check("all valid", report["valid"], 3)
check("no duplicates", report["duplicates"], [])
check("distinct count", report["distinct"], 3)

report = L.check_list(got + [[[4, 1, 4], [1, 1, 1]]])   # a relabelling
check("relabelled duplicate detected", len(report["duplicates"]), 1)

report = L.check_list([[[4, 4]]])                        # not Calabi-Yau
check("non-CY entry rejected", report["valid"], 0)
check_true("reason recorded", bool(report["invalid"]))

shutil.rmtree(TMP, ignore_errors=True)

# --------------------------------------------------------------------------
print("\n" + "=" * 72)
if FAILURES:
    print("FAILED ({}): {}".format(len(FAILURES), ", ".join(FAILURES)))
    sys.exit(1)
print("ALL TESTS PASSED on Python {}".format(sys.version.split()[0]))
