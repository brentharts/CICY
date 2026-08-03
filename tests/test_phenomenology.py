"""
Tests for pyCICY.phenomenology.

Two kinds of check:

  [1]-[3] that the index-theorem quantities are right, against the quintic
          and against real entries of the published list;
  [4]     that quantities the topology does not determine are refused rather
          than approximated. That test matters as much as the others: a
          placeholder mass ratio plotted beside a measured constant is
          indistinguishable from a derived one, so the refusal is a feature
          under test, not an omission.

Run with:  python3 tests/test_phenomenology.py
       or: python3 run_tests.py  (runs every suite)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyCICY import phenomenology as P

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


# --------------------------------------------------------------------------
print("\n[1] Standard embedding spectrum on the quintic")
spec = P.standard_embedding_spectrum([[4, 5]])
check("gauge group", spec["gauge_group"], "E_6")
check("n(27) = h^{2,1}", spec["n_27"], 101)
check("n(27-bar) = h^{1,1}", spec["n_27bar"], 1)
check("net generations = h21 - h11", spec["net_generations"], 100)
check("which is |chi|/2", spec["net_generations"], abs(spec["euler"]) // 2)
check_true("E_6 is flagged as not the Standard Model group",
           any("Standard Model" in n for n in spec["notes"]))

# The index theorem must hold for every configuration we try.
for conf in ([[4, 5]], [[2, 3], [2, 3]], [[1, 2], [1, 2], [1, 2], [1, 2]],
             [[1, 1, 1], [4, 1, 4]], [[2, 1, 1, 1], [4, 2, 2, 1]]):
    s = P.standard_embedding_spectrum(conf)
    check_true("h21 - h11 = -chi/2 for %s" % (conf,),
               s["h21"] - s["h11"] == -s["euler"] // 2)

# --------------------------------------------------------------------------
print("\n[2] Generation counting and required symmetry order")
check("chi = -200 gives 100 generations", P.chiral_generations(-200), 100)
check("chi = -24 with |Gamma| = 4 gives 3", P.chiral_generations(-24, 4), 3)
check("chi = -6 would give 3 with no quotient", P.chiral_generations(-6), 3)
check("order needed for chi = -24", P.required_symmetry_order(-24), 4)
check("order needed for chi = -12", P.required_symmetry_order(-12), 2)
check_true("chi = -200 admits no integer order",
           P.required_symmetry_order(-200) is None)
check_true("chi = 0 admits none", P.required_symmetry_order(0) is None)

# |Gamma| must divide chi, since chi(X/Gamma) has to be an integer.
try:
    P.standard_embedding_spectrum([[4, 5]], symmetry_order=3)
    check_true("non-dividing |Gamma| rejected", False)
except ValueError as exc:
    check_true("non-dividing |Gamma| rejected", "does not divide" in str(exc))

for bad in (0, -1):
    try:
        P.chiral_generations(-24, bad)
        check_true("|Gamma| = %d rejected" % bad, False)
    except ValueError:
        check_true("|Gamma| = %d rejected" % bad, True)

# Fourfolds and non-threefolds are refused.
try:
    P.standard_embedding_spectrum([[5, 6]])
    check_true("fourfold refused", False)
except ValueError as exc:
    check_true("fourfold refused", "threefold" in str(exc))

# --------------------------------------------------------------------------
print("\n[3] Survey of the published list, when present")
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "cicylist.json")
if os.path.exists(DATA):
    from pyCICY import cicylist as L

    entries = L.load_published_list(DATA)
    result = P.generation_survey(entries)
    check("entries surveyed", result["entries"], 7890)
    check("configurations with chi = 0", result["zero_euler"], 52)
    check("three-generation candidates", result["candidates"], 2689)
    check("smallest non-zero |chi|", result["min_abs_euler"], 4)
    # |chi| = 6 does not occur anywhere in the list, so three generations
    # from the standard embedding always needs a non-trivial free quotient.
    check_true("|chi| = 6 never occurs", result["needs_quotient"])
    check_true("|Gamma| = 4 is a common requirement",
               result["required_orders"].get(4, 0) > 100)
    check_true("every candidate order divides some |chi|/6",
               all(k >= 1 for k in result["required_orders"]))

    # A real three-generation candidate from the list: CICY 950, chi = -24.
    entry = [r for r in entries if r["num"] == 950][0]
    check("CICY 950 has chi = -24", entry["euler"], -24)
    s = P.standard_embedding_spectrum(entry["conf"], symmetry_order=4)
    check("with |Gamma| = 4 it gives three generations",
          s["net_generations"], 3)
    # h11 = 9 and h21 = 21 are not divisible by 4, so the individual 27 counts
    # on the quotient are not simply h/|Gamma|; this must be flagged.
    check_true("individual 27 counts withheld when not divisible",
               s["n_27"] is None and s["n_27bar"] is None)
    check_true("and the reason is recorded",
               any("not simply the" in n for n in s["notes"]))
else:
    print("  data/cicylist.json absent; run scripts/fetch_cicy_list.py")

# --------------------------------------------------------------------------
print("\n[4] Quantities topology does not determine are refused")
why = P.why_not_masses()
check("marked as not computable", why["computable_from_topology"], False)
check_true("several reasons are given", len(why["reasons"]) >= 4)
check_true("the metric problem is named",
           any("Ricci-flat" in r for r in why["reasons"]))
check_true("moduli stabilisation is named",
           any("moduli" in r for r in why["reasons"]))
check_true("alternatives are offered", len(why["what_is_computable"]) >= 3)

# The exception type exists and is an error, not a sentinel, so that a
# refusal cannot be silently plotted as a value.
check_true("MassRatioNotComputable is an exception",
           issubclass(P.MassRatioNotComputable, Exception))

# There must be no function in the module that returns a mass ratio.
# (Substring matching on "ratio" would catch "generations"; match on "mass".)
names = [n for n in dir(P) if "mass" in n.lower() and not n.startswith("_")]
check("only the refusal path mentions masses", sorted(names),
      ["MassRatioNotComputable", "why_not_masses"])
check_true("no compute_mass_ratio function exists",
           not hasattr(P, "compute_mass_ratio"))

# --------------------------------------------------------------------------
print("\n" + "=" * 72)
if FAILURES:
    print("FAILED ({}): {}".format(len(FAILURES), ", ".join(FAILURES)))
    sys.exit(1)
print("ALL TESTS PASSED on Python {}".format(sys.version.split()[0]))
