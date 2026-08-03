"""
Regression tests for pyCICY on modern Python / NumPy.

Run with:  python3 tests/test_pycicy.py
       or: python3 run_tests.py  (runs every suite)
(or: pytest test_pycicy.py)

These cover the breakages found when bringing the package up to
Python 3.12 / NumPy 2.x, plus correctness checks against values from
the CICY literature.
"""

import os
import sys

# Prefer the source tree over any installed copy of pyCICY.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import sys

import numpy as np

from pyCICY import CICY

logging.disable(logging.CRITICAL)

FAILURES = []


def check(name, got, want, tol=1e-9):
    ok = abs(float(got) - float(want)) < tol
    print("  {:<52} {:>12} {}".format(name, str(got), "ok" if ok else "FAIL want " + str(want)))
    if not ok:
        FAILURES.append(name)


def check_true(name, cond):
    print("  {:<52} {:>12} {}".format(name, str(bool(cond)), "ok" if cond else "FAIL"))
    if not cond:
        FAILURES.append(name)


# --------------------------------------------------------------------------
print("\n[1] Hodge numbers / Euler characteristic vs. literature")
# (name, configuration, h^{1,1}, h^{2,1})
LIT = [
    ("quintic",      [[4, 5]],                                  1, 101),
    ("bicubic",      [[2, 3], [2, 3]],                          2,  83),
    ("tetraquadric", [[1, 2], [1, 2], [1, 2], [1, 2]],          4,  68),
]
for name, conf, h11, h21 in LIT:
    M = CICY(conf)
    check(name + " h^{1,1}", M.h[2], h11)
    check(name + " h^{2,1}", M.h[1], h21)
    check(name + " euler", M.euler_characteristic(), 2 * (h11 - h21))

# --------------------------------------------------------------------------
print("\n[2] Docstring reference values, CICY([[2,2,1],[3,1,3]])")
M = CICY([[2, 2, 1], [3, 1, 3]])
check("c2(0,1)", M.c2(0, 1), 2.5)
check_true("second_chern_all == [[1,2.5],[2.5,3]]",
           np.allclose(M.second_chern_all(), [[1.0, 2.5], [2.5, 3.0]]))
check("euler_characteristic", M.euler_characteristic(), -114)
check("hodge_data h^{2,1}", M.hodge_data()[1], 59)
check("hodge_data h^{1,1}", M.hodge_data()[2], 2.0)

# --------------------------------------------------------------------------
print("\n[3] euler_characteristic is cached rounded (was: raw float on reuse)")
M = CICY([[1, 2], [1, 2], [1, 2], [1, 2]])
vals = [M.euler_characteristic() for _ in range(3)]
check_true("three calls all give exactly -128", all(v == -128 for v in vals))
check_true("returns an integer type", isinstance(vals[0], (int, np.integer)))

# --------------------------------------------------------------------------
print("\n[4] hodge_data() does not corrupt self.fav / logger (was: permanent)")
M = CICY([[1, 2], [1, 2], [1, 2], [1, 2]])
fav_before = M.fav
first = list(M.hodge_data())
second = list(M.hodge_data())
check_true("self.fav preserved across hodge_data()", M.fav == fav_before is True)
check_true("hodge_data() is idempotent", first == second)
check_true("hodge_data() == cached self.h", [float(x) for x in first] == [float(x) for x in M.h])
check_true("logger level restored", logging.getLogger("pyCICY.pyCICY").level == 0)

# --------------------------------------------------------------------------
print("\n[5] np.int removal: CICYs needing _single_map now construct")
# This configuration crashed in __init__ under NumPy >= 1.24 with
# AttributeError: module 'numpy' has no attribute 'int'
T = CICY([[1, 2, 0, 0, 0], [1, 0, 2, 0, 0], [1, 0, 0, 2, 0],
          [1, 0, 0, 0, 2], [3, 1, 1, 1, 1]])
check("T h^{1,1}", T.h[2], 5)
check("T h^{2,1}", T.h[1], 37)
check("T euler", T.euler_characteristic(), -64)
check_true("T euler == 2(h11-h21)",
           T.euler_characteristic() == 2 * (float(T.h[2]) - float(T.h[1])))

# --------------------------------------------------------------------------
print("\n[6] Line bundle cohomology")
M = CICY([[1, 2], [1, 2], [1, 2], [1, 2]])
check_true("line_co([1,2,-4,1]) == [0,36,0,0]",
           list(M.line_co([1, 2, -4, 1])) == [0, 36, 0, 0])
for L in ([2, -3, 1, 1], [-5, 2, 3, -1], [4, -6, 2, 3]):
    co = M.line_co(L)
    idx = co[0] - co[1] + co[2] - co[3]
    check("index(L=%s) == line_co_euler" % (L,), idx, M.line_co_euler(L))

# --------------------------------------------------------------------------
print("\n[7] Leray matches its documented example")
M2 = CICY([[2, 2, 1], [3, 1, 3]])
E, origin = M2.Leray(M2._line_to_BBW([3, -4]))
expected = [[0, 0, 0, [[3, 0]], 0, 0],
            [0, 0, 0, [[1, -1], [2, -3]], 0, 0],
            [0, 0, 0, [[0, -4]], 0, 0]]
got = [[(np.array(c).tolist() if c != 0 else 0) for c in row] for row in E]
check_true("Leray E_1 table", got == expected)

# --------------------------------------------------------------------------
print("\n[8] Misc API smoke test")
M = CICY([[1, 2], [1, 2], [1, 2], [1, 2]])
for name, fn in [
    ("first_chern", lambda: M.first_chern()),
    ("second_chern", lambda: M.second_chern()),
    ("third_chern", lambda: M.third_chern()),
    ("triple_intersection", lambda: M.triple_intersection()),
    ("is_favourable", lambda: M.is_favourable()),
    ("is_directproduct", lambda: M.is_directproduct()),
    ("line_index", lambda: M.line_index()),
    ("l_slope", lambda: M.l_slope([1, 2, -4, 1])),
    ("line_slope", lambda: M.line_slope()),
    ("def_poly", lambda: M.def_poly()),
    ("drst", lambda: M.drst(0, 1, 2)),
]:
    try:
        fn()
        check_true(name, True)
    except Exception as exc:
        print("  {:<52} FAIL {}: {}".format(name, type(exc).__name__, exc))
        FAILURES.append(name)

# --------------------------------------------------------------------------
print("\n[9] Chern class sums are exact (no float32 drift)")
M = CICY([[1, 2], [1, 2], [1, 2], [1, 2]])
c3 = M.third_chern()
d = M.triple_intersection()
e = np.einsum("rst,rst", d, c3)
check_true("einsum(d,c3) within 1e-9 of integer -128", abs(e - (-128)) < 1e-9)

# --------------------------------------------------------------------------
print("\n[10] CY fourfolds (sextic in P^5)")
F = CICY([[5, 6]])
check("nfold", F.nfold, 4)
check("h^{3,1}", F.h[1], 426)
check("h^{2,1}", F.h[2], 0)
check("h^{1,1}", F.h[3], 1)
check("h^{2,2}", F.h[4], 1752)
check("euler", F.euler_characteristic(), 2610)
h11, h21, h31, h22 = float(F.h[3]), float(F.h[2]), float(F.h[1]), float(F.h[4])
check("4 + 2h11 - 4h21 + 2h31 + h22",
      4 + 2 * h11 - 4 * h21 + 2 * h31 + h22, 2610)

# --------------------------------------------------------------------------
print("\n[11] Unsupported dimensions fail clearly (was: UnboundLocalError)")
for conf, fold in ([[1, 1, 1], [1, 1, 1], [1, 1, 1]], 1), ([[3, 2, 2]], 1), ([[2, 4, 1]], 0):
    try:
        CICY(conf)
        print("  {:<52} {:>12} FAIL expected ValueError".format(str(conf)[:50], ""))
        FAILURES.append(str(conf))
    except ValueError as exc:
        check_true("%s -> ValueError (%d-fold)" % (str(conf)[:30], fold),
                   "fold" in str(exc))

# --------------------------------------------------------------------------
print("\n" + "=" * 72)
if FAILURES:
    print("FAILED ({}): {}".format(len(FAILURES), ", ".join(FAILURES)))
    sys.exit(1)
print("ALL TESTS PASSED on Python {}, NumPy {}".format(
    sys.version.split()[0], np.__version__))
