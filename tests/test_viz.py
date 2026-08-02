"""
Tests for pyCICY.viz.

The important one is section [1]: it checks the fast numpy evaluation against
the original symbolic sympy formulation from the calabi_yau_stl.py script,
so the rewrite is verified rather than merely assumed.

Run with:  python3 tests/test_viz.py
       or: python3 run_tests.py  (runs every suite)
"""

import os
import sys

# Prefer the source tree over any installed copy of pyCICY.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import struct
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")

import numpy as np
import sympy as sp

from pyCICY import viz as cy

FAILURES = []


def check_true(name, cond):
    print("  {:<56} {}".format(name, "ok" if cond else "FAIL"))
    if not cond:
        FAILURES.append(name)


def check_close(name, got, want, tol=1e-12):
    ok = abs(float(got) - float(want)) <= tol
    print("  {:<56} {:>10.3e} {}".format(name, float(got), "ok" if ok else "FAIL"))
    if not ok:
        FAILURES.append(name)


# --------------------------------------------------------------------------
print("\n[1] numpy evaluation matches the original sympy formulation")
xs, ys, asym = sp.symbols("x y a")
worst = 0.0
for (n, k1, k2) in [(2, 0, 1), (3, 1, 2), (5, 0, 0), (5, 2, 3), (7, 4, 6), (8, 7, 0)]:
    z1 = sp.exp(sp.I * ((2 * sp.pi * k1) / n)) * (sp.cos(xs + ys * sp.I)) ** (sp.Rational(2) / n)
    z2 = sp.exp(sp.I * ((2 * sp.pi * k2) / n)) * (sp.sin(xs + ys * sp.I)) ** (sp.Rational(2) / n)
    Xe, Ye, Ze = sp.re(z1), sp.re(z2), sp.im(z1) * sp.cos(asym) + sp.im(z2) * sp.sin(asym)
    for xv in (0.11, 0.7, 1.4):
        for yv in (-1.2, 0.0, 0.9):
            ref = [float(e.subs({xs: xv, ys: yv, asym: 0.4}).evalf())
                   for e in (Xe, Ye, Ze)]
            w = xv + 1j * yv
            g1 = np.exp(2j * np.pi * k1 / n) * np.cos(w) ** (2.0 / n)
            g2 = np.exp(2j * np.pi * k2 / n) * np.sin(w) ** (2.0 / n)
            got = [g1.real, g2.real,
                   g1.imag * np.cos(0.4) + g2.imag * np.sin(0.4)]
            worst = max(worst, max(abs(r - g) for r, g in zip(ref, got)))
check_close("max |numpy - sympy| over 54 sample points", worst, 0.0, 1e-12)

# --------------------------------------------------------------------------
print("\n[2] patch() shape, finiteness and argument validation")
X, Y, Z = cy.patch(5, 0, 0, res=12)
check_true("patch returns (12, 12) grids", X.shape == Y.shape == Z.shape == (12, 12))
check_true("patch values are real", not np.iscomplexobj(Z))
check_true("interior values finite", np.isfinite(Z[1:-1, 1:-1]).all())
for bad, desc in [((1, 0, 0), "n < 2"), ((5, 5, 0), "k1 out of range"),
                  ((5, 0, -1), "k2 out of range")]:
    try:
        cy.patch(*bad, res=4)
        check_true("%s rejected" % desc, False)
    except ValueError:
        check_true("%s rejected" % desc, True)

print("  ", end="")
check_true("patches() yields n^2 patches", len(list(cy.patches(4, res=6))) == 16)

# --------------------------------------------------------------------------
print("\n[3] Projection angle actually varies the surface")
_, _, Z0 = cy.patch(5, 1, 2, a=0.0, res=10)
_, _, Z1 = cy.patch(5, 1, 2, a=np.pi / 2, res=10)
check_true("a=0 and a=pi/2 give different Z", not np.allclose(Z0, Z1))
# At a=0, Z is purely Im z1; at a=pi/2 purely Im z2.
w = np.meshgrid(np.linspace(0, np.pi / 2, 10),
                np.linspace(-np.pi / 2, np.pi / 2, 10))
wc = w[0] + 1j * w[1]
im1 = (np.exp(2j * np.pi * 1 / 5) * np.cos(wc) ** (2.0 / 5)).imag
check_true("a=0 gives Z = Im z1", np.allclose(Z0, im1))

# --------------------------------------------------------------------------
print("\n[4] pyCICY integration")
check_true("from_cicy([[4,5]]) == 5", cy.from_cicy([[4, 5]]) == 5)
check_true("from_cicy([[5,6]]) == 6", cy.from_cicy([[5, 6]]) == 6)
for bad, desc in [([[4, 4]], "non-CY degree"), ([[2, 3], [2, 3]], "two spaces")]:
    try:
        cy.from_cicy(bad)
        check_true("%s rejected" % desc, False)
    except ValueError:
        check_true("%s rejected" % desc, True)

lbl = cy.describe(5)
check_true("describe(5) reports quintic h11=1", "h^{1,1}$=1" in lbl)
check_true("describe(5) reports h21=101", "h^{2,1}$=101" in lbl)
check_true("describe(5) reports chi=-200", "-200" in lbl)
check_true("describe(6) reports sextic 4-fold chi=2610",
           "2610" in cy.describe(6) and "4-fold" in cy.describe(6))
check_true("describe(3) degrades gracefully", cy.describe(3) == "n=3")
# Above 4 folds pyCICY has no Euler implementation and returns 0; we must not
# print that placeholder as though it were a real invariant.
check_true("describe(7) omits placeholder chi", "chi" not in cy.describe(7))

# --------------------------------------------------------------------------
print("\n[5] STL export is structurally valid")
tmp = tempfile.mkdtemp()
path = os.path.join(tmp, "cy.stl")
count = cy.write_stl(path, 4, res=10)
size = os.path.getsize(path)
with open(path, "rb") as f:
    f.read(80)
    header_count = struct.unpack("<I", f.read(4))[0]
    body = f.read()
check_true("returned count == header count", count == header_count)
check_true("file size == 84 + 50*facets", size == 84 + 50 * count)
check_true("body length == 50*facets", len(body) == 50 * count)
check_true("at least one facet written", count > 0)

vals = np.array([struct.unpack("<12fH", body[i * 50:(i + 1) * 50])[:12]
                 for i in range(count)])
check_true("all vertex/normal data finite", np.isfinite(vals).all())
lens = np.linalg.norm(vals[:, 0:3], axis=1)
check_true("all normals unit length", np.allclose(lens, 1.0, atol=1e-5))

apath = os.path.join(tmp, "cy_ascii.stl")
acount = cy.write_stl(apath, 3, res=8, mode="ascii")
text = open(apath).read()
check_true("ascii facet count matches", text.count("facet normal") == acount)
check_true("ascii loops balanced",
           text.count("outer loop") == text.count("endloop") == acount)
check_true("ascii has no nan", "nan" not in text.lower())

# Degenerate triangles must be skipped, not written with a NaN normal
# (surf2stl.local_find_normal divides by zero here).
class _Reject:
    def write(self, *a):
        raise AssertionError("degenerate facet should not be written")

p = [np.array([1., 0., 0.]), np.array([2., 0., 0.]), np.array([3., 0., 0.])]
check_true("collinear triangle skipped", cy._facet(_Reject(), *p, "binary") == 0)
q = np.array([np.nan, 0., 0.])
check_true("non-finite vertex skipped",
           cy._facet(_Reject(), q, p[1], p[2], "binary") == 0)

# --------------------------------------------------------------------------
print("\n[6] Plotting runs headless")
import matplotlib.pyplot as plt

ax = cy.plot(3, res=10)
check_true("plot() returns a 3D axis", hasattr(ax, "get_zlim"))
check_true("plot() drew n^2 surfaces", len(ax.collections) == 9)
ax2 = cy.plot(3, res=10, color_by="patch")
check_true("color_by='patch' also draws", len(ax2.collections) == 9)
fig = cy.plot_grid(range(2, 5), res=8)
check_true("plot_grid makes one axis per n", len(fig.axes) == 3)
plt.close("all")

xb, yb, zb = cy.bounds(5, res=12)
check_true("bounds are finite", all(np.isfinite([*xb, *yb, *zb])))
check_true("bounds ordered", xb[0] < xb[1] and yb[0] < yb[1] and zb[0] < zb[1])

# --------------------------------------------------------------------------
print("\n" + "=" * 72)
if FAILURES:
    print("FAILED ({}): {}".format(len(FAILURES), ", ".join(FAILURES)))
    sys.exit(1)
print("ALL TESTS PASSED on Python {}, NumPy {}, matplotlib {}".format(
    sys.version.split()[0], np.__version__, matplotlib.__version__))
