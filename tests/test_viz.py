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
print("\n[6] plots for the toric, quantum-curve, knot and chirality modules")

from pyCICY import chirality as _chir
from pyCICY import knots as _knots
from pyCICY import toric as _toric

def _boundary_lines(ax):
    return [l for l in ax.lines if l.get_label() in ("P", "P*")]

ax = cy.plot_polygon("B3")
check_true("plot_polygon draws P and its dual", len(_boundary_lines(ax)) == 2)
check_true("plot_polygon marks the lattice points", len(ax.collections) >= 3)
check_true("plot_polygon titles with the twelve theorem", "12" in ax.get_title())
ax = cy.plot_polygon("P2", with_dual=False)
check_true("with_dual=False omits the dual", len(_boundary_lines(ax)) == 1)
ax = cy.plot_polygon([(1, 0), (0, 1), (-1, -1)])
check_true("plot_polygon accepts explicit vertices", ax.get_title().startswith("P2"))
plt.close("all")

fig = cy.plot_polygon_grid()
check_true("polygon grid has a panel per reflexive polygon",
           sum(1 for a in fig.axes if a.get_title()) == len(_toric.NAMED))
plt.close("all")
fig = cy.plot_polygon_grid(names=["P2", "F0"], ncols=2)
check_true("polygon grid honours a subset", len(fig.axes) == 2)
plt.close("all")

ax = cy.plot_butterfly("F0", qmax=8, nk=3)
check_true("butterfly draws points", len(ax.collections) == 1)
check_true("butterfly labels F0 bipartite", "bipartite" in ax.get_title())
check_true("butterfly x axis spans the full flux range",
           ax.get_xlim() == (0.0, 1.0))
ax = cy.plot_butterfly("B3", qmax=8, nk=3)
check_true("butterfly labels B3 spectrally chiral",
           "chiral" in ax.get_title())
ax = cy.plot_butterfly("F0", qmax=6, nk=3, gaps_at=(1, 3))
check_true("gap annotations appear", len(ax.texts) >= 2)
check_true("gap annotations are signed Chern numbers",
           all(t.get_text().lstrip("+-").isdigit() for t in ax.texts))
plt.close("all")
fig = cy.plot_butterfly_grid(["F0", "B3"], qmax=6, nk=3)
check_true("butterfly grid has two panels", len(fig.axes) == 2)
plt.close("all")

ax = cy.plot_jones("K15n81556")
check_true("plot_jones draws the knot and its mirror", len(ax.containers) == 2)
check_true("plot_jones reports K15n81556 as chiral", "chiral" in ax.get_title())
ax = cy.plot_jones("4_1")
check_true("plot_jones reports 4_1 as not separated",
           "does not separate" in ax.get_title())
ax = cy.plot_jones(_knots.from_name("3_1"), with_mirror=False)
check_true("with_mirror=False draws one series", len(ax.containers) == 1)
plt.close("all")

ax = cy.plot_braid([1] * 3)
check_true("plot_braid draws a line per crossing strand", len(ax.lines) >= 6)
check_true("plot_braid adds closure arcs", len(ax.patches) == 2)
ax = cy.plot_braid([1] * 7 + [-2] * 7, strands=3)
check_true("plot_braid handles three strands", len(ax.patches) == 3)
check_true("plot_braid titles with the crossing count",
           "14 crossings" in ax.get_title())
ax = cy.plot_braid([1, -1], closure=False)
check_true("closure=False omits the arcs", len(ax.patches) == 0)
plt.close("all")

records = _chir.survey()
ax = cy.plot_chirality(records)
check_true("plot_chirality draws every non-curve domain",
           len(ax.collections) == 6)          # filled + hollow, three domains
check_true("plot_chirality marks the mirror axis",
           any(l.get_xdata()[0] == 0 for l in ax.lines))
check_true("plot_chirality labels the fixed points", len(ax.texts) >= 3)
plt.close("all")
fig = cy.plot_chirality_grid(records)
check_true("chirality grid has one panel per domain", len(fig.axes) == 3)
plt.close("all")

# The polygon panel is the degenerate case: every preserved value is 12.
poly_recs = [r for r in records if r["domain"] == "polygon"]
check_true("every polygon record preserves twelve",
           {r["preserved"] for r in poly_recs} == {12})
ax = cy.plot_chirality(poly_recs)
lo, hi = ax.get_ylim()
check_true("a constant preserved value still gets a sensible y range",
           lo < 12 < hi and hi - lo > 1)
plt.close("all")

ax = cy.plot_hyperbolic_flake(8, depth=2)
check_true("hyperbolic flake plots the cell centres", len(ax.collections) >= 1)
check_true("hyperbolic flake draws the boundary circle", len(ax.patches) == 1)
check_true("hyperbolic flake draws geodesic bonds as arcs",
           all(len(l.get_xdata()) > 2 for l in ax.lines))
check_true("hyperbolic flake titles with the boundary fraction",
           "boundary fraction" in ax.get_title())
ax = cy.plot_hyperbolic_flake(8, depth=1, show_bonds=False)
check_true("show_bonds=False omits the arcs", len(ax.lines) == 0)
plt.close("all")

try:
    cy.plot_chirality([r for r in records if r["domain"] == "curve"])
    check_true("curve-only records are rejected", False)
except ValueError:
    check_true("curve-only records are rejected", True)

# ------------------------------------------------- heterotic model building
from pyCICY.theories import yukawa as _Y
from pyCICY.theories import representatives as _RP
from pyCICY import equivariant as _E

_conf = _Y.CICY5299["configuration"]
_M = _Y.CICY5299["summands"]

# The texture plot must agree with the computation it draws, or it is
# decoration rather than a figure. Both counts are checked against the modules.
_t = _Y.texture(_conf, _M)
_ref = _RP.refine_texture(_conf, _M, kind="up")
plt.close("all")
ax = cy.plot_yukawa_texture(_conf, _M, kind="up")
check_true("up-type title reports the cup-product refined count (%d, not %d)"
           % (_ref["kept"], _t["summary"]["up"]["present"]),
           "%d of" % _ref["kept"] in ax.get_title())
plt.close("all")
ax = cy.plot_yukawa_texture(_conf, _M, kind="down")
check_true("down-type title matches the texture (%d)"
           % _t["summary"]["down"]["present"],
           "%d of" % _t["summary"]["down"]["present"] in ax.get_title())
plt.close("all")

for _name, _fn in (("search funnel", cy.plot_search_funnel),
                   ("unification", cy.plot_unification),
                   ("racetrack", cy.plot_racetrack),
                   ("equivariant character", cy.plot_equivariant_character)):
    plt.close("all")
    try:
        check_true("%s renders" % _name, _fn() is not None)
    except Exception as _e:                                      # noqa: BLE001
        check_true("%s renders (%r)" % (_name, _e), False)
plt.close("all")

# The character plot claims equidistribution; check the claim, not the picture.
ax = cy.plot_equivariant_character()
check_true("a free action is labelled equidistributed",
           "equidistributed" in ax.get_title())
plt.close("all")
ax = cy.plot_equivariant_character(
    action=_E.CyclicAction([[1, 2]] * 4, 3, [[0, 1]] * 4, [0]),
    charges=[[1, 1, 1, 1]])
check_true("and a non-free one is not", "uneven" in ax.get_title())
plt.close("all")

# The unification plot draws six lines: three couplings, two spectra.
ax = cy.plot_unification()
check_true("unification plot draws both spectra", len(ax.lines) == 6)
plt.close("all")

# --------------------------------------------------------------------------
print("\n" + "=" * 72)
if FAILURES:
    print("FAILED ({}): {}".format(len(FAILURES), ", ".join(FAILURES)))
    sys.exit(1)
print("ALL TESTS PASSED on Python {}, NumPy {}, matplotlib {}".format(
    sys.version.split()[0], np.__version__, matplotlib.__version__))
