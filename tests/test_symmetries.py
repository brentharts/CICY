"""
Tests for pyCICY.mathematica and pyCICY.symmetries.

The symmetry data itself is not bundled -- it lives in the Mathematica
version of the Oxford CICY list and is fetched by
scripts/fetch_symmetries.py -- so sections [1] and [2] test the parser and
the group-order table directly, and section [3] tests the model selection
against a synthetic symmetry table whose answer is known by construction.
Section [4] runs against the real file if it is present.

The central safety property under test is that an unrecognised group name
raises rather than defaulting. A name silently treated as order 1 would
inflate the count of viable three-generation models, which is precisely the
error this code exists to avoid.

Run with:  python3 tests/test_symmetries.py
       or: python3 run_tests.py  (runs every suite)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyCICY import mathematica as MM
from pyCICY import symmetries as S

FAILURES = []
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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
print("\n[1] Mathematica expression parser")
check("list", MM.loads("{1,2,3}"), [1, 2, 3])
check("nested list", MM.loads("{{1,1},{0,3}}"), [[1, 1], [0, 3]])
check("negative integer", MM.loads("-17"), -17)
check("real", MM.loads("2.5"), 2.5)
check("string", MM.loads('"hi"'), "hi")
check("comments skipped", MM.loads("{1, (* note *) 2}"), [1, 2])
check("nested comments skipped", MM.loads("{1 (* a (* b *) c *), 2}"), [1, 2])

rule = MM.loads("Num -> 14")
check_true("rule parsed", isinstance(rule, MM.Rule))
check("rule lhs", str(rule.lhs), "Num")
check("rule rhs", rule.rhs, 14)

from fractions import Fraction
check("exact rational", MM.loads("3/4"), Fraction(3, 4))

call = MM.loads("f[1, {2,3}]")
check_true("function call parsed", isinstance(call, MM.Expr))
check("call head", call.head, "f")
check("call arity", len(call.args), 2)

# A data file wraps its table in an assignment; that must not be a syntax
# error, and the right-hand side must be reachable.
assigned = MM.loads("CICYlist = {{Num -> 1}}")
check_true("assignment parsed", isinstance(assigned, MM.Expr))
check("assignment head", assigned.head, "Set")
check("assignment rhs", len(assigned.args[1]), 1)

check("multiple statements",
      len(MM.loads("a = {1}; b = {2};", all_expressions=True)), 2)

# Symbols and strings must stay distinguishable, since group names arrive as
# bare symbols.
check_true("bare symbol is a Symbol",
           isinstance(MM.loads("Z5"), MM.Symbol))
check_true("quoted name is a plain str, not a Symbol",
           not isinstance(MM.loads('"Z5"'), MM.Symbol))

# The real file records group actions as polynomials in the homogeneous
# coordinates, so infix arithmetic must parse. Nothing is evaluated.
poly = MM.loads("{x16^2*x2*x6, 0, 0}")
check("polynomial entry parsed", len(poly), 3)
check_true("it became a Times tree",
           isinstance(poly[0], MM.Expr) and poly[0].head == "Times")
check_true("power is a Power tree",
           isinstance(MM.loads("x^2"), MM.Expr)
           and MM.loads("x^2").head == "Power")
check_true("sum is a Plus tree", MM.loads("a + b").head == "Plus")
check_true("unary minus on a symbol", MM.loads("-x1").head == "Minus")
check("negative literals stay literals", MM.loads("{1,-2,3}"), [1, -2, 3])
check_true("division of symbols", MM.loads("x/y").head == "Divide")
check("exact rational still folds", MM.loads("3/4"), Fraction(3, 4))
check_true("rules still parse alongside arithmetic",
           isinstance(MM.loads("x1 -> -x2"), MM.Rule))

# Constructs found in the real cicylist.m, added after it failed to parse.
check_true("parenthesised group",
           MM.loads("(a+b)*c").head == "Times")
check_true("unary minus on a parenthesised product",
           MM.loads("-(x10*x2*x7*x8)").head == "Minus")
check("polynomial with a leading negated group",
      len(MM.loads("{0, -(x10*x2*x7*x8) + x1*x7*x8*x9, 0}")), 3)
check_true("Part is not read as a function call",
           MM.loads("v[[1]]").head == "Part")
check_true("a genuine call is still a call", MM.loads("f[x]").head == "f")
check_true("repeated application", MM.loads("f[x][y]").head == "Apply")
check("Mathematica export exponent", MM.loads("1.2*^-5"), 1.2e-5)
check_true("slots and pure functions",
           MM.loads("#1 + 1 &").head == "Function")
check_true("comparisons", MM.loads("x1 == x2").head == "Equal")

for bad in ("{1, 2", "{1,,2}", "@@", "{1} {2}"):
    try:
        MM.loads(bad)
        check_true("%r rejected" % bad, False)
    except MM.MathematicaSyntaxError:
        check_true("%r rejected" % bad, True)

# --------------------------------------------------------------------------
print("\n[2] Group orders")
for name, order in [("Z2", 2), ("Z5", 5), ("Z_4", 4), ("Z3xZ3", 9),
                    ("Z2xZ2xZ2", 8), ("Q8", 8), ("Dic3", 12), ("A4", 12),
                    ("S3", 6), ("12", 12), (8, 8), ("Trivial", 1)]:
    check("order of %r" % (name,), S.group_order(name), order)

check("multiplication sign accepted", S.group_order("Z2\u00d7Z4"), 8)

for bad in ("Frobenius42", "", "SomeGroup", "Z0"):
    try:
        S.group_order(bad)
        check_true("%r raises" % bad, False)
    except S.UnknownGroup:
        check_true("%r raises" % bad, True)

# The key property: an unknown name must not become order 1.
orders, unknown = S.symmetry_orders({"symmetries": ["Z4", "Mystery"]})
check("known orders extracted", orders, [4])
check("unknown names reported", unknown, ["Mystery"])
try:
    S.symmetry_orders({"symmetries": ["Mystery"]}, strict=True)
    check_true("strict mode raises", False)
except S.UnknownGroup:
    check_true("strict mode raises", True)

check("absent symmetry field", S.symmetry_orders({}), ([], []))
check("empty symmetry field", S.symmetry_orders({"symmetries": []}), ([], []))

# The layout of the source file is not under our control, so the group name
# may sit under a rule. Fields holding the coordinate action must be skipped,
# or coordinate symbols would be reported as unknown groups.
rec = {"symmetries": [[
    {"rule": ["Group", "Z3"]},
    {"rule": ["Action", [[{"head": "Times", "args": ["x1", 2]}, 0],
                         [0, "x2"]]]},
]]}
orders, unknown = S.symmetry_orders(rec)
check("group read from a rule", orders, [3])
check("coordinate action not mistaken for groups", unknown, [])
check("group under an alternative key",
      S.symmetry_orders({"symmetries": [{"rule": ["Symmetry", "Q8"]}]})[0],
      [8])

# --------------------------------------------------------------------------
print("\n[2b] Group order from generator matrices")
# The Oxford file records no group names: each symmetry is a freeness flag
# plus explicit generators acting on the homogeneous coordinates.
check("order of <diag(-1,1)>", S.matrix_group_order([[[-1, 0], [0, 1]]]), 2)
check("order of <rot90>", S.matrix_group_order([[[0, -1], [1, 0]]]), 4)
check("order of <diag(-1,1), diag(1,-1)>",
      S.matrix_group_order([[[-1, 0], [0, 1]], [[1, 0], [0, -1]]]), 4)
check("order of the trivial generator",
      S.matrix_group_order([[[1, 0], [0, 1]]]), 1)
# An infinite group must be reported as undetermined, never as a number: a
# spurious order would create a spurious three-generation model.
check_true("infinite order returns None",
           S.matrix_group_order([[[1, 1], [0, 1]]]) is None)
check_true("empty generators return None",
           S.matrix_group_order([]) is None)
check_true("ragged generators return None",
           S.matrix_group_order([[[1, 0], [0, 1, 0]]]) is None)

# The shape the real file uses: {True, {{generators}}}.
rec = {"symmetries": [["True", [[[[-1, 0], [0, 1]]]]]]}
check("order read from the file's shape", S.symmetry_orders(rec)[0], [2])
check_true("the freeness flag is not read as a group name",
           S.symmetry_orders({"symmetries": ["True"]}) == ([], []))
check("generator matrices are found at depth",
      len(S.generator_matrices([["True", [[[[1, 0], [0, 1]]]]]])), 1)

# --------------------------------------------------------------------------
print("\n[3] Model selection against a synthetic table")
# chi = -24 needs |Gamma| = 4; chi = -12 needs 2; chi = -200 needs nothing
# integral, so it is not a candidate at all.
cicys = [
    {"num": 1, "euler": -24},
    {"num": 2, "euler": -12},
    {"num": 3, "euler": -200},
    {"num": 4, "euler": -24},
    {"num": 5, "euler": -36},
]
syms = [
    {"num": 1, "symmetries": ["Z4"]},          # exactly what is needed
    {"num": 2, "symmetries": ["Z4"]},          # order 4, need 2: divisible
    {"num": 3, "symmetries": ["Z5"]},          # irrelevant, not a candidate
    {"num": 4, "symmetries": []},              # candidate, no group
    {"num": 5, "symmetries": ["Mystery"]},     # unknown name
]
res = S.three_generation_models(cicys, syms)
check("exact matches", res["n_exact"], 1)
check("matches by divisibility", res["n_by_divisibility"], 1)
check("candidates lacking a group", res["n_without"], 2)
check("unknown names counted", res["unknown_groups"], {"Mystery": 1})
check("chi = -200 excluded entirely",
      all(e["num"] != 3 for e in res["exact"] + res["by_divisibility"]
          + res["candidates_without_symmetry"]), True)
check("exact match is entry 1", res["exact"][0]["num"], 1)
check("its required order", res["exact"][0]["order"], 4)

# Divisibility must be reported separately, never folded into the exact
# count: a subgroup of the right order exists for abelian groups but is an
# extra step in general.
check_true("divisible case is not counted as exact",
           res["by_divisibility"][0]["num"] == 2)

report = S.coverage_report(cicys, syms)
check("CICYs considered", report["cicys"], 5)
check("with a non-trivial symmetry", report["with_nontrivial_symmetry"], 3)
check("order distribution", report["order_distribution"], {4: 2, 5: 1})

# A missing data file must explain where the data comes from.
try:
    S.load_symmetries(os.path.join(ROOT, "data", "definitely-absent.json"))
    check_true("missing file explained", False)
except FileNotFoundError as exc:
    check_true("missing file explained", "fetch_symmetries" in str(exc))

# --------------------------------------------------------------------------
print("\n[4] Against the real data, when present")
SYM = os.path.join(ROOT, "data", "symmetries.json")
CICY = os.path.join(ROOT, "data", "cicylist.json")
if os.path.exists(SYM) and os.path.exists(CICY):
    from pyCICY import cicylist as L
    from pyCICY import phenomenology as P

    entries = L.load_published_list(CICY)
    syms = S.load_symmetries(SYM)

    # The Mathematica file carries 31 records beyond the 7890 of the text
    # list, numbered 7891-7921 with the symmetry field set to "unknown".
    check("records in the Mathematica file", len(syms), 7921)
    unknown = [r for r in syms if r["symmetries"] == "unknown"]
    check("records marked unknown", len(unknown), 31)
    check_true("they are the ones beyond 7890",
               all(r["num"] > 7890 for r in unknown))

    with_data = [r for r in syms
                 if isinstance(r["symmetries"], list) and r["symmetries"]]
    # Braun's classification: 195 CICYs admit freely acting symmetries.
    check("CICYs carrying symmetry data", len(with_data), 195)

    total = 0
    problems = 0
    orders = {}
    for rec in with_data:
        parsed, probs = S.parse_symmetry_records(rec)
        total += len(parsed)
        problems += len(probs)
        for item in parsed:
            orders[item.order] = orders.get(item.order, 0) + 1
    # Lukas and Mishra, CMP 379 (2020) 847, quote 1695 quotients arising
    # from Braun's classification.
    check("symmetry records in total", total, 1695)
    check_true("every record yielded a GAP order", sum(orders.values()) == 1695)
    # One record in the file has a GAP order that disagrees with its own
    # abelian invariants (order 8 against invariants [2, 8]). That is an
    # inconsistency in the source data; it is surfaced, not resolved.
    check("records failing their own consistency check", problems, 1)
    check_true("group orders are all at least 2", min(orders) >= 2)

    # The Tian-Yau manifold: CICY 536, chi = -18, with a free Z_3 giving
    # chi = -6 and hence three generations. This is the first
    # three-generation Calabi-Yau ever constructed, and it should fall out
    # of the search rather than being put in by hand.
    ty = [r for r in entries if r["num"] == 536][0]
    check("CICY 536 configuration", ty["conf"], [[3, 3, 0, 1], [3, 0, 3, 1]])
    check("its Euler characteristic", ty["euler"], -18)
    check("order needed for three generations",
          P.required_symmetry_order(ty["euler"], 3), 3)
    ty_syms = [r for r in syms if r["num"] == 536][0]
    parsed, _ = S.parse_symmetry_records(ty_syms)
    check_true("a group of order 3 is recorded for it",
               3 in {x.order for x in parsed})

    # The full search: of the configurations whose Euler characteristic
    # permits three generations, how many carry a group of exactly the
    # required order.
    by_num = {r["num"]: r for r in syms}
    hits = []
    for rec in entries:
        need = P.required_symmetry_order(rec["euler"], 3)
        if need is None:
            continue
        sym = by_num.get(rec["num"])
        if not sym:
            continue
        parsed, _ = S.parse_symmetry_records(sym)
        if need in {x.order for x in parsed}:
            hits.append(rec["num"])
    check_true("the Tian-Yau manifold is among the hits", 536 in hits)
    check_true("the search is highly selective", 0 < len(hits) < 20)
    print("  three-generation models found: %s" % sorted(hits))
else:
    print("  data absent; run scripts/fetch_cicy_list.py and "
          "scripts/fetch_symmetries.py")
    print("  (this section is skipped, not failed)")

# --------------------------------------------------------------------------
print("\n" + "=" * 72)
if FAILURES:
    print("FAILED ({}): {}".format(len(FAILURES), ", ".join(FAILURES)))
    sys.exit(1)
print("ALL TESTS PASSED on Python {}".format(sys.version.split()[0]))
