"""
pyCICY.viz -- visualise the classic Calabi-Yau cross-section, fast.

This is a rewrite of the standalone calabi_yau_stl.py script. The original built every patch as a
symbolic sympy expression and then reached into sympy's private plotting
internals (``p.backend(p)``, ``backend.parent``, ``_process_series``) to move
the result onto a matplotlib axis. Those internals were removed in newer
sympy, so the script no longer runs at all.

Here the surface is evaluated directly with numpy complex arithmetic, which
is both public API and dramatically faster. The formulation is identical to
the original to machine precision (see tests/test_viz.py).

The surface
-----------
For the Fermat hypersurface z1^n + z2^n = 1 one takes the n^2 patches

    z1 = exp(2*pi*i*k1/n) * cos(x + i*y)^(2/n)
    z2 = exp(2*pi*i*k2/n) * sin(x + i*y)^(2/n)

with k1, k2 in 0..n-1, over x in [0, pi/2], y in [-pi/2, pi/2], and projects
the 4 real dimensions down to 3 via

    X = Re z1,  Y = Re z2,  Z = Im z1 * cos(a) + Im z2 * sin(a)

where ``a`` is the projection angle.

Plots for the other modules
---------------------------
Besides the Fermat cross-section, this module draws the objects the rest of
the package computes:

    plot_polygon, plot_polygon_grid   reflexive polygons and their polar duals
    plot_butterfly, ..._grid          Hofstadter spectra of quantized curves
    plot_jones                        Jones polynomial against its mirror
    plot_braid                        braid closures, e.g. 7_1 # m7_1
    plot_chirality                    the cross-domain asymmetry plot
    plot_hyperbolic_flake             a {p,q} tiling in the Poincare disk
    plot_apolynomial                  Newton polygon of a knot A-polynomial
    plot_colored_jones                colored Jones coefficients by colour
    plot_yukawa_texture               allowed, forbidden and vanishing couplings
    plot_search_funnel                narrowing to a viable model
    plot_unification                  one-loop running of the gauge couplings
    plot_racetrack                    condensation ratios the racetracks need
    plot_equivariant_character        index characters, free versus not

The last is the unifying one. Every chirality record carries an *asymmetry*,
the combination of its invariant pair that negates under the mirror
operation, and a *preserved* quantity that does not. Plotting one against the
other puts mirror partners symmetrically about zero and fixed points on the
axis. For Calabi-Yau threefolds this is precisely the conventional Hodge plot
of :func:`plot_hodge`, with chi/2 horizontally, so that figure turns out to
be one panel of a family that also covers knots and reflexive polygons.

Relation to pyCICY
------------------
The degree-n Fermat hypersurface in P^(n-1) is the CICY with configuration
[[n-1, n]] -- so n=5 is the quintic, CICY([[4, 5]]). :func:`from_cicy` maps a
configuration matrix to the corresponding n, and :func:`describe` pulls the
Hodge data straight out of pyCICY so plots can be labelled with real
topological invariants rather than just an integer.
"""

import argparse
import math
import struct
from fractions import Fraction as Fraction_

import numpy as np

__all__ = [
    "patch", "patches", "bounds", "from_cicy", "describe",
    "plot", "plot_grid", "write_stl", "DEFAULT_ANGLE",
    "plot_hodge", "plot_node_counts",
    "plot_polygon", "plot_polygon_grid",
    "plot_butterfly", "plot_butterfly_grid",
    "plot_jones", "plot_braid", "plot_chirality", "plot_chirality_grid",
    "plot_hyperbolic_flake", "plot_apolynomial", "plot_colored_jones",
    "plot_yukawa_texture", "plot_search_funnel", "plot_unification",
    "plot_racetrack", "plot_equivariant_character",
]

DEFAULT_ANGLE = 0.4


# ---------------------------------------------------------------- geometry

def patch(n, k1, k2, a=DEFAULT_ANGLE, res=30):
    """Evaluate a single (k1, k2) patch of the degree-n surface.

    Parameters
    ----------
    n : int
        Degree of the hypersurface (n >= 2). n=5 is the quintic.
    k1, k2 : int
        Patch indices, each in 0..n-1.
    a : float
        Projection angle mixing Im z1 and Im z2 into the third axis.
    res : int
        Grid resolution per parameter direction.

    Returns
    -------
    X, Y, Z : ndarray, shape (res, res)
    """
    if n < 2:
        raise ValueError("n must be >= 2, got %r" % (n,))
    if not (0 <= k1 < n and 0 <= k2 < n):
        raise ValueError("k1, k2 must lie in 0..%d, got (%r, %r)" % (n - 1, k1, k2))

    x = np.linspace(0.0, np.pi / 2, res)
    y = np.linspace(-np.pi / 2, np.pi / 2, res)
    x, y = np.meshgrid(x, y)

    w = x + 1j * y
    # Principal branch, matching sympy's convention.
    z1 = np.exp(2j * np.pi * k1 / n) * np.cos(w) ** (2.0 / n)
    z2 = np.exp(2j * np.pi * k2 / n) * np.sin(w) ** (2.0 / n)

    X = z1.real
    Y = z2.real
    Z = z1.imag * math.cos(a) + z2.imag * math.sin(a)
    return X, Y, Z


def patches(n, a=DEFAULT_ANGLE, res=30):
    """Yield ``(k1, k2, X, Y, Z)`` for all n^2 patches of the surface."""
    for k1 in range(n):
        for k2 in range(n):
            X, Y, Z = patch(n, k1, k2, a, res)
            yield k1, k2, X, Y, Z


def bounds(n, a=DEFAULT_ANGLE, res=30):
    """Return ``((xmin, xmax), (ymin, ymax), (zmin, zmax))`` over the surface.

    Useful for giving every subplot a common, correctly scaled aspect ratio.
    """
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for _, _, X, Y, Z in patches(n, a, res):
        for i, arr in enumerate((X, Y, Z)):
            finite = arr[np.isfinite(arr)]
            if finite.size:
                lo[i] = min(lo[i], finite.min())
                hi[i] = max(hi[i], finite.max())
    return tuple((float(lo[i]), float(hi[i])) for i in range(3))


# ------------------------------------------------------------- pyCICY glue

def from_cicy(conf):
    """Map a CICY configuration matrix to the degree n of its Fermat model.

    Only configurations of the form ``[[n-1, n]]`` -- a single degree-n
    hypersurface in P^(n-1) -- have a Fermat cross-section of this kind.

    >>> from_cicy([[4, 5]])
    5
    """
    rows = [list(r) for r in conf]
    if len(rows) != 1 or len(rows[0]) != 2:
        raise ValueError(
            "Only a single hypersurface in a single projective space has a "
            "Fermat cross-section of this form; got %r" % (conf,))
    dim, deg = int(rows[0][0]), int(rows[0][1])
    if deg != dim + 1:
        raise ValueError(
            "Configuration %r is not Calabi-Yau: a degree-%d hypersurface in "
            "P^%d needs degree %d for c_1 = 0." % (conf, deg, dim, dim + 1))
    return deg


def describe(n):
    """Return a label for the degree-n surface, using pyCICY when it applies.

    pyCICY only handles Calabi-Yau 2-, 3- and 4-folds, so for other n we fall
    back to a plain label rather than pretending we have invariants.
    """
    base = "n=%d" % n
    try:
        import logging

        try:
            from .pyCICY import CICY
        except ImportError:
            # Allows running this file directly, not just as pyCICY.viz
            from pyCICY import CICY
    except ImportError:
        return base

    # pyCICY logs under the bare name 'pyCICY', and hodge_data() calls
    # logger.setLevel() internally, so raising the level here would just be
    # overwritten. A filter is not affected by setLevel, so use one.
    cy_logger = logging.getLogger("pyCICY")
    mute = lambda record: False  # noqa: E731
    cy_logger.addFilter(mute)
    try:
        M = CICY([[n - 1, n]])
        if M.nfold == 3:
            return "%s  $h^{1,1}$=%g  $h^{2,1}$=%g  $\\chi$=%d" % (
                base, float(M.h[2]), float(M.h[1]), M.euler_characteristic())
        if M.nfold == 4:
            return "%s  (CY 4-fold)  $\\chi$=%d" % (
                base, M.euler_characteristic())
        # pyCICY has no Hodge/Euler implementation above 4 folds; it returns a
        # placeholder 0 there, so report the dimension only.
        return "%s  (CY %d-fold)" % (base, M.nfold)
    except Exception:
        # n < 4 gives a CY 0-/1-fold, which pyCICY rejects by design.
        return base
    finally:
        cy_logger.removeFilter(mute)


# ------------------------------------------------------------------ plotting

def plot_hodge(records, ax=None, color_by="depth", cmap="viridis",
               annotate=(), title=None, alpha=0.75):
    """Hodge plot of a set of Calabi-Yau threefolds.

    The conventional presentation for the CICY list: the Euler characteristic
    chi = 2(h^{1,1} - h^{2,1}) horizontally against h^{1,1} + h^{2,1}
    vertically, so mirror pairs sit symmetrically about chi = 0.

    Parameters
    ----------
    records : iterable of dict
        Each needs ``h11`` and ``h21``; ``depth`` and ``favourable`` are used
        for colouring when present. This is the shape produced by
        :func:`pyCICY.cicylist.web_nodes`.
    color_by : {'depth', 'favourable', 'none'}
        'depth' shades by how many splits were needed to reach the
        configuration; 'favourable' contrasts favourable descriptions
        against the rest.
    annotate : iterable of (h11, h21, label)
        Points to label, for calling out specific manifolds.
    """
    import matplotlib.pyplot as plt

    recs = [r for r in records
            if r.get("h11") is not None and r.get("h21") is not None]
    if not recs:
        raise ValueError("no records with Hodge numbers to plot")

    if ax is None:
        fig = plt.figure(figsize=(9, 6))
        ax = fig.add_subplot(111)

    h11 = np.array([float(r["h11"]) for r in recs])
    h21 = np.array([float(r["h21"]) for r in recs])
    chi = 2 * (h11 - h21)
    height = h11 + h21

    if color_by == "depth" and all("depth" in r for r in recs):
        c = np.array([r["depth"] for r in recs])
        sc = ax.scatter(chi, height, c=c, cmap=cmap, s=26, alpha=alpha,
                        edgecolors="none")
        cb = ax.figure.colorbar(sc, ax=ax)
        cb.set_label("splits from a seed")
    elif color_by == "favourable" and all("favourable" in r for r in recs):
        fav = np.array([bool(r["favourable"]) for r in recs])
        ax.scatter(chi[fav], height[fav], s=26, alpha=alpha,
                   label="favourable", edgecolors="none")
        ax.scatter(chi[~fav], height[~fav], s=26, alpha=alpha,
                   label="not favourable", edgecolors="none")
        ax.legend(frameon=False)
    else:
        ax.scatter(chi, height, s=26, alpha=alpha, edgecolors="none")

    ax.axvline(0, color="0.6", lw=0.8, zorder=0)
    ax.set_xlabel(r"$\chi = 2(h^{1,1} - h^{2,1})$")
    ax.set_ylabel(r"$h^{1,1} + h^{2,1}$")
    ax.set_title(title if title is not None
                 else "%d configurations, %d distinct Hodge pairs"
                 % (len(recs), len({(a, b) for a, b in zip(h11, h21)})))

    for a, b, label in annotate:
        ax.annotate(label, (2 * (a - b), a + b),
                    textcoords="offset points", xytext=(6, 6), fontsize=9)
        ax.scatter([2 * (a - b)], [a + b], s=70, facecolors="none",
                   edgecolors="crimson", linewidths=1.4, zorder=5)
    return ax


def plot_node_counts(edges, ax=None, bins=30, title=None):
    """Distribution of node counts N over the split edges of a web.

    N is the number of nodes of the singular variety joining the two sides of
    the conifold transition. N = 0 marks an ineffective split, which changes
    no topology at all, so those are separated out rather than buried in the
    first bin.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig = plt.figure(figsize=(9, 5))
        ax = fig.add_subplot(111)

    counts = [int(e["nodes"]) for e in edges]
    zero = sum(1 for n in counts if n == 0)
    positive = [n for n in counts if n > 0]

    if positive:
        ax.hist(positive, bins=bins, alpha=0.85, edgecolor="none")
    ax.set_xlabel("nodes $N$ of the singular variety")
    ax.set_ylabel("split edges")
    ax.set_title(title if title is not None
                 else "%d effective splits (plus %d ineffective, $N=0$)"
                 % (len(positive), zero))
    return ax


def plot(n, a=DEFAULT_ANGLE, res=30, ax=None, cmap="viridis",
         elev=30, azim=45, title=None, equal_aspect=True,
         color_by="z", alpha=1.0, axis_off=False):
    """Draw the degree-n surface on a 3D axis and return that axis.

    color_by : {'z', 'patch'}
        'z' shades every vertex by height, which brings out the structure of
        the surface. 'patch' gives each of the n^2 patches one flat colour,
        which makes the patch decomposition legible instead.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    if ax is None:
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection="3d")

    cm = plt.get_cmap(cmap)
    total = n * n
    zb = bounds(n, a, res)[2]
    norm = Normalize(vmin=zb[0], vmax=zb[1])

    for idx, (k1, k2, X, Y, Z) in enumerate(patches(n, a, res)):
        kw = dict(rstride=1, cstride=1, linewidth=0, antialiased=False,
                  alpha=alpha)
        if color_by == "patch":
            ax.plot_surface(X, Y, Z, color=cm(idx / max(total - 1, 1)),
                            shade=True, **kw)
        else:
            ax.plot_surface(X, Y, Z, facecolors=cm(norm(Z)),
                            shade=False, **kw)

    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title if title is not None else describe(n))
    if axis_off:
        ax.set_axis_off()

    if equal_aspect:
        xb, yb, zb2 = bounds(n, a, res)
        ax.set_xlim(*xb)
        ax.set_ylim(*yb)
        ax.set_zlim(*zb2)
        try:
            ax.set_box_aspect([hi - lo for lo, hi in (xb, yb, zb2)])
        except AttributeError:  # matplotlib < 3.3
            pass
    return ax


def plot_grid(ns=range(2, 9), a=DEFAULT_ANGLE, res=30, cmap="viridis",
              ncols=4, figsize=(16, 8)):
    """Reproduce the original multi-panel figure, one panel per n."""
    import matplotlib.pyplot as plt

    ns = list(ns)
    nrows = int(math.ceil(len(ns) / ncols))
    fig = plt.figure(figsize=figsize)
    for g, n in enumerate(ns):
        ax = fig.add_subplot(nrows, ncols, g + 1, projection="3d")
        plot(n, a=a, res=res, ax=ax, cmap=cmap)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------- STL out

def write_stl(filename, n, a=DEFAULT_ANGLE, res=30, mode="binary"):
    """Write the whole degree-n surface to a single STL file.

    The original script emitted n^2 separate files, one per patch, so the
    surface could not be opened as one object. This writes all patches into
    one mesh. Facets touching a non-finite vertex are skipped, matching
    surf2stl's behaviour at the branch points.
    """
    if mode != "ascii":
        mode = "binary"

    title = "Calabi-Yau n=%d a=%.4f by calabi_yau.py" % (n, a)
    nfacets = 0

    with open(filename, "wb" if mode != "ascii" else "w") as f:
        if mode == "ascii":
            f.write("solid %s\n" % title)
        else:
            f.write(title.ljust(80).encode("ascii")[:80])
            f.write(struct.pack("<I", 0))  # placeholder, patched below

        for _, _, X, Y, Z in patches(n, a, res):
            for i in range(Z.shape[0] - 1):
                for j in range(Z.shape[1] - 1):
                    quad = [(i, j), (i, j + 1), (i + 1, j + 1), (i + 1, j)]
                    p = [np.array([X[r, c], Y[r, c], Z[r, c]]) for r, c in quad]
                    nfacets += _facet(f, p[0], p[1], p[2], mode)
                    nfacets += _facet(f, p[2], p[3], p[0], mode)

        if mode == "ascii":
            f.write("endsolid %s\n" % title)
        else:
            f.seek(80, 0)
            f.write(struct.pack("<I", nfacets))

    return nfacets


def _facet(f, p1, p2, p3, mode):
    """Write one triangle; return 1 if written, 0 if degenerate/non-finite."""
    if not (np.isfinite(p1).all() and np.isfinite(p2).all()
            and np.isfinite(p3).all()):
        return 0
    v = np.cross(p2 - p1, p3 - p1)
    norm = math.sqrt(float(np.dot(v, v)))
    if norm == 0.0 or not math.isfinite(norm):
        return 0  # degenerate triangle: zero-area, no well-defined normal
    nvec = v / norm

    if mode == "ascii":
        f.write("facet normal %.7f %.7f %.7f\n" % tuple(nvec))
        f.write("outer loop\n")
        for p in (p1, p2, p3):
            f.write("vertex %.7f %.7f %.7f\n" % tuple(p))
        f.write("endloop\n")
        f.write("endfacet\n")
    else:
        f.write(struct.pack("<3f", *nvec))
        for p in (p1, p2, p3):
            f.write(struct.pack("<3f", *p))
        f.write(struct.pack("<H", 0))
    return 1


# ---------------------------------------------------------------------- CLI

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("-n", type=int, default=5,
                    help="degree of the hypersurface (default 5, the quintic)")
    ap.add_argument("--cicy", type=str, default=None,
                    help="CICY configuration instead of -n, e.g. '[[4,5]]'")
    ap.add_argument("-a", "--angle", type=float, default=DEFAULT_ANGLE,
                    help="projection angle (default %.1f)" % DEFAULT_ANGLE)
    ap.add_argument("-r", "--res", type=int, default=30,
                    help="grid resolution per patch (default 30)")
    ap.add_argument("--grid", action="store_true",
                    help="plot a panel per n instead of a single surface")
    ap.add_argument("--range", type=int, nargs=2, default=(2, 9),
                    metavar=("LO", "HI"), help="n range for --grid")
    ap.add_argument("--cmap", default="viridis")
    ap.add_argument("-o", "--out", default=None, help="save figure to this path")
    ap.add_argument("--stl", default=None, help="also write this STL file")
    ap.add_argument("--stl-mode", choices=("binary", "ascii"), default="binary")
    args = ap.parse_args(argv)

    n = args.n
    if args.cicy:
        import ast
        n = from_cicy(ast.literal_eval(args.cicy))
        print("configuration %s -> n = %d" % (args.cicy, n))

    import matplotlib
    if args.out:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if args.grid:
        fig = plot_grid(range(args.range[0], args.range[1]),
                        a=args.angle, res=args.res, cmap=args.cmap)
    else:
        ax = plot(n, a=args.angle, res=args.res, cmap=args.cmap)
        fig = ax.figure

    if args.stl:
        cnt = write_stl(args.stl, n, a=args.angle, res=args.res,
                        mode=args.stl_mode)
        print("wrote %d facets to %s" % (cnt, args.stl))

    if args.out:
        fig.savefig(args.out, dpi=140, bbox_inches="tight")
        print("wrote %s" % args.out)
    else:
        plt.show()


if __name__ == "__main__":
    main()


# ------------------------------------------------- polygons, curves, knots
#
# Plots for pyCICY.toric, pyCICY.quantum_curve, pyCICY.knots and
# pyCICY.chirality. matplotlib is imported inside each function, as
# everywhere else in this module, so that importing pyCICY.viz stays cheap.

def _polygon_verts(polygon):
    from . import toric as _toric
    if isinstance(polygon, str):
        return _toric.polygon(polygon), polygon
    verts = _toric.convex_hull([tuple(v) for v in polygon])
    return verts, _toric.classify(verts)["name"]


def plot_polygon(polygon, ax=None, with_dual=True, annotate=True, title=None,
                 legend=True):
    """A reflexive polygon, its lattice points, and optionally its polar dual.

    The dual is the Batyrev mirror, and drawing the two together makes the
    twelve theorem visible: the boundary lattice points of P and of P* always
    number twelve between them, however they are shared out.

    Parameters
    ----------
    polygon : str or sequence of (m, n)
        A name from :data:`pyCICY.toric.NAMED`, or explicit vertices.
    with_dual : bool
        Overlay the polar dual.
    """
    import matplotlib.pyplot as plt
    from . import toric as _toric

    verts, name = _polygon_verts(polygon)
    if ax is None:
        ax = plt.figure(figsize=(4.2, 4.2)).add_subplot(111)

    def _draw(vs, color, label, lw, ls):
        pts = list(vs) + [vs[0]]
        ax.plot([p[0] for p in pts], [p[1] for p in pts],
                color=color, lw=lw, ls=ls, label=label, zorder=3)

    _draw(verts, "tab:blue", "P", 1.8, "-")
    inner = _toric.lattice_points(verts)
    ax.scatter([p[0] for p in inner], [p[1] for p in inner],
               s=16, color="tab:blue", zorder=4)
    ax.scatter([v[0] for v in verts], [v[1] for v in verts],
               s=52, facecolors="none", edgecolors="tab:blue", zorder=5)

    if with_dual:
        dual = _toric.dual(verts)
        _draw(dual, "tab:red", "P*", 1.4, "--")
        dpts = _toric.lattice_points(dual)
        ax.scatter([p[0] for p in dpts], [p[1] for p in dpts],
                   s=12, color="tab:red", alpha=0.8, zorder=4)
        if legend:
            ax.legend(frameon=False, fontsize=8, loc="best")

    ax.scatter([0], [0], marker="+", s=70, color="k", zorder=6)
    ax.axhline(0, color="0.85", lw=0.6, zorder=0)
    ax.axvline(0, color="0.85", lw=0.6, zorder=0)
    ax.set_aspect("equal")
    ax.grid(True, color="0.93", lw=0.5)

    if title is None and name:
        b, bd, tot = _toric.twelve(verts)
        bits = ["%s" % name]
        if annotate:
            bits.append(r"$\partial P{=}%d$, $\partial P^*{=}%d$, sum ${=}%d$"
                        % (b, bd, tot))
        title = "\n".join(bits)
    if title:
        ax.set_title(title, fontsize=9)
    return ax


def plot_polygon_grid(names=None, ncols=4, with_dual=True, figsize=None):
    """All sixteen reflexive polygons, or a chosen subset, on one figure."""
    import matplotlib.pyplot as plt
    from . import toric as _toric

    names = list(_toric.NAMED) if names is None else list(names)
    nrows = int(math.ceil(len(names) / float(ncols)))
    figsize = figsize or (3.0 * ncols, 3.1 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    for ax, name in zip(axes.ravel(), names):
        plot_polygon(name, ax=ax, with_dual=with_dual, legend=False)
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    handles = [plt.Line2D([], [], color="tab:blue", lw=1.8, label="$P$"),
               plt.Line2D([], [], color="tab:red", lw=1.4, ls="--",
                          label="$P^*$")]
    if with_dual:
        fig.legend(handles=handles, frameon=False, fontsize=9,
                   loc="lower center", ncol=2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def plot_butterfly(curve, qmax=25, nk=6, ax=None, color="k", size=0.6,
                   title=None, gaps_at=None):
    """The Hofstadter spectrum of a quantized mirror curve.

    Energies are plotted against the flux Phi = hbar / 2 pi, sweeping every
    reduced fraction with denominator up to ``qmax``. Local F_0 gives the
    classic square-lattice butterfly; local B_3 gives the triangular one,
    which is visibly slanted because E(Phi) = -E(1-Phi) holds without the
    spectrum itself being symmetric in E.

    Parameters
    ----------
    gaps_at : (p, q), optional
        Annotate the gap Chern numbers at this flux.
    """
    import matplotlib.pyplot as plt
    from . import quantum_curve as _qc

    if isinstance(curve, str):
        curve = _qc.from_polygon(curve)
    if ax is None:
        ax = plt.figure(figsize=(5.2, 4.6)).add_subplot(111)

    flux, energy = curve.butterfly(qmax=qmax, nk=nk)
    ax.scatter(flux, energy, s=size, c=color, marker=".", linewidths=0,
               alpha=0.55)
    ax.set_xlabel(r"$\Phi = \hbar / 2\pi$")
    ax.set_ylabel("$E$")
    ax.set_xlim(0, 1)

    if gaps_at is not None:
        p, q = gaps_at
        for g in curve.gap_labels(p, q, nk=max(nk, 12)):
            mid = 0.5 * (g["lower"] + g["upper"])
            ax.annotate("%+d" % g["chern"], (p / q, mid), fontsize=7,
                        color="tab:red", ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                  ec="tab:red", lw=0.5, alpha=0.85))

    if title is None:
        bip = curve.is_bipartite()
        title = "%s -- %s" % (curve.name or "curve",
                              "bipartite" if bip else "spectrally chiral")
    ax.set_title(title, fontsize=10)
    return ax


def plot_butterfly_grid(names=("F0", "B3"), qmax=25, nk=6, figsize=None,
                        gaps_at=None):
    """Several butterflies side by side."""
    import matplotlib.pyplot as plt

    names = list(names)
    figsize = figsize or (4.6 * len(names), 4.4)
    fig, axes = plt.subplots(1, len(names), figsize=figsize, squeeze=False)
    for ax, name in zip(axes[0], names):
        plot_butterfly(name, qmax=qmax, nk=nk, ax=ax, gaps_at=gaps_at)
    fig.tight_layout()
    return fig


def plot_jones(knot, ax=None, with_mirror=True, title=None):
    """Jones polynomial coefficients, against those of the mirror image.

    Chirality shows up as the failure of the two to coincide. This is the
    picture behind the observation of Wang and Zhang, arXiv:2507.14265, that
    K15n81556 and the knot in the second Brittenham-Hermiller diagram are
    mirror images rather than the same knot.
    """
    import matplotlib.pyplot as plt
    from . import knots as _knots

    if isinstance(knot, str):
        knot = _knots.from_name(knot)
    if ax is None:
        ax = plt.figure(figsize=(5.6, 3.4)).add_subplot(111)

    v = knot.jones()
    series = [(v, "tab:blue", "$V(t)$", -0.18)]
    if with_mirror:
        series.append((knot.mirror().jones(), "tab:red",
                       "$V(1/t)$ (mirror)", 0.18))

    width = 0.34 if with_mirror else 0.6
    for poly, color, label, off in series:
        exps = sorted(poly.c)
        ax.bar([e + off for e in exps], [poly.c[e] for e in exps],
               width=width, color=color, label=label, alpha=0.85)

    ax.axhline(0, color="0.4", lw=0.8)
    ax.axvline(0, color="0.85", lw=0.6, zorder=0)
    ax.set_xlabel("power of $t$")
    ax.set_ylabel("coefficient")
    ax.legend(frameon=False, fontsize=8)
    if title is None:
        title = "%s -- %s" % (
            knot.name or "knot",
            "chiral" if knot.is_chiral() else "Jones does not separate it")
    ax.set_title(title, fontsize=10)
    return ax


def _smoothstep(t):
    return t * t * (3.0 - 2.0 * t)


def plot_braid(word, strands=None, ax=None, closure=True, title=None,
               gap=0.22):
    """Draw a braid word, optionally with its closure arcs.

    Generator ``k`` is the crossing of strands k and k+1 in which the lower
    strand passes over, matching :func:`pyCICY.knots.from_braid`; ``-k`` is
    its inverse. The under-strand is drawn broken at the crossing.

    The connected sum of two braid closures is the juxtaposition of their
    words on one more strand than the two together, so ``[1]*7 + [-2]*7`` on
    three strands is 7_1 # m7_1, the knot of arXiv:2506.24088.
    """
    import matplotlib.pyplot as plt
    from matplotlib.path import Path
    import matplotlib.patches as mpatches

    word = [int(g) for g in word]
    if strands is None:
        strands = (max(abs(g) for g in word) + 1) if word else 1
    if ax is None:
        ax = plt.figure(figsize=(0.8 * max(len(word), 1) + 2.2,
                                 0.8 * strands + 1.2)).add_subplot(111)

    ts = np.linspace(0.0, 1.0, 40)
    for col, g in enumerate(word):
        k = abs(g) - 1
        for pos in range(strands):
            if pos not in (k, k + 1):
                ax.plot([col, col + 1], [pos, pos], color="0.25", lw=1.6,
                        solid_capstyle="round", zorder=2)
        for start, end in ((k, k + 1), (k + 1, k)):
            xs = col + ts
            ys = start + (end - start) * _smoothstep(ts)
            # the generator's sign says which of the two is on top
            over = (start == k) if g > 0 else (start == k + 1)
            if over:
                ax.plot(xs, ys, color="0.15", lw=1.8, zorder=4,
                        solid_capstyle="round")
            else:
                keep = (np.abs(ts - 0.5) > gap)
                seg, cur = [], []
                for x, y, ok in zip(xs, ys, keep):
                    if ok:
                        cur.append((x, y))
                    elif cur:
                        seg.append(cur)
                        cur = []
                if cur:
                    seg.append(cur)
                for piece in seg:
                    ax.plot([q[0] for q in piece], [q[1] for q in piece],
                            color="0.15", lw=1.8, zorder=3,
                            solid_capstyle="round")

    length = len(word)
    if closure and strands:
        top = strands - 1
        for i in range(strands):
            # nested arcs: the lowest strand takes the widest loop
            d = 0.45 + 0.45 * (strands - 1 - i)
            h = top + 0.60 + 0.45 * (strands - 1 - i)
            verts = [(length, i), (length + d, i), (length + d, h),
                     (-d, h), (-d, i), (0, i)]
            codes = [Path.MOVETO] + [Path.LINETO] * (len(verts) - 1)
            ax.add_patch(mpatches.PathPatch(
                Path(verts, codes), fill=False, color="0.55", lw=1.2,
                ls="-", zorder=1))

    ax.set_aspect("equal")
    ax.axis("off")
    if title is None:
        title = "braid closure, {} strands, {} crossings".format(
            strands, len(word))
    ax.set_title(title, fontsize=9)
    return ax


_DOMAIN_STYLE = {
    "knot": ("tab:blue", "o", "knots"),
    "polygon": ("tab:green", "s", "reflexive polygons"),
    "cicy": ("tab:red", "^", "Calabi-Yau threefolds"),
}


def plot_chirality(records=None, ax=None, annotate=True, title=None,
                   xscale="linear"):
    """The cross-domain chirality plot.

    Horizontally the *asymmetry*, the combination of each object's invariant
    pair that negates under its mirror operation; vertically the quantity the
    mirror preserves. Mirror partners therefore sit symmetrically about zero
    and objects fixed by their involution sit on the axis, whatever domain
    they come from.

    Restricted to the Calabi-Yau records this is the conventional Hodge plot
    of :func:`plot_hodge`, since the asymmetry there is chi/2 and the
    preserved quantity is h^{1,1} + h^{2,1}.

    Quantized curves are omitted: their involution has no invariant pair,
    because it leaves the spectrum exactly where it was.
    """
    import matplotlib.pyplot as plt
    from . import chirality as _chir

    if records is None:
        records = _chir.survey()
    usable = [r for r in records if r.get("asymmetry") is not None
              and isinstance(r.get("preserved"), int)]
    if not usable:
        raise ValueError("no records with an asymmetry to plot")

    if ax is None:
        ax = plt.figure(figsize=(7.6, 5.0)).add_subplot(111)

    for domain, (color, marker, label) in _DOMAIN_STYLE.items():
        rs = [r for r in usable if r["domain"] == domain]
        if not rs:
            continue
        ax.scatter([r["asymmetry"] for r in rs], [r["preserved"] for r in rs],
                   c=color, marker=marker, s=44, alpha=0.85,
                   edgecolors="none", label=label)
        # the mirror partners, hollow
        ax.scatter([-r["asymmetry"] for r in rs], [r["preserved"] for r in rs],
                   facecolors="none", edgecolors=color, marker=marker, s=44,
                   alpha=0.5, linewidths=0.8)

    if annotate:
        # several fixed objects can land on the same point -- the four
        # self-dual polygons all sit at (0, 12) -- so group before labelling
        groups = {}
        for r in usable:
            if r["fixed"] and r.get("name"):
                groups.setdefault((r["asymmetry"], r["preserved"]),
                                  []).append(str(r["name"]))
        for (x, y), names in groups.items():
            ax.annotate(", ".join(sorted(names)), (x, y),
                        textcoords="offset points", xytext=(6, 3),
                        fontsize=7, color="0.3")

    if xscale == "symlog":
        ax.set_xscale("symlog", linthresh=10)
    heights = {r["preserved"] for r in usable}
    if len(heights) == 1:                 # e.g. polygons, always 12
        v = heights.pop()
        ax.set_ylim(v - 3, v + 3)
    ax.axvline(0, color="0.6", lw=0.9, zorder=0)
    ax.set_xlabel("asymmetry (negated by the mirror)")
    ax.set_ylabel("preserved by the mirror")
    ax.legend(frameon=False, fontsize=8)
    n_fixed = sum(1 for r in usable if r["fixed"])
    ax.set_title(title if title is not None
                 else "%d objects, %d fixed by their mirror (on the axis); "
                      "filled = object, hollow = its mirror"
                      % (len(usable), n_fixed), fontsize=9)
    return ax


def plot_chirality_grid(records=None, figsize=None):
    """One chirality panel per domain, each on its own scale.

    The combined :func:`plot_chirality` is dominated by the Calabi-Yau
    points, whose asymmetry reaches a hundred while knots and polygons live
    within about ten. Splitting by domain keeps the shared structure -- the
    mirror is a reflection about zero, fixed points lie on the axis -- while
    letting each domain be legible.
    """
    import matplotlib.pyplot as plt
    from . import chirality as _chir

    if records is None:
        records = _chir.survey()
    domains = [d for d in ("knot", "polygon", "cicy")
               if any(r["domain"] == d for r in records)]
    figsize = figsize or (4.4 * len(domains), 4.0)
    fig, axes = plt.subplots(1, len(domains), figsize=figsize, squeeze=False)
    for ax, domain in zip(axes[0], domains):
        rs = [r for r in records if r["domain"] == domain]
        plot_chirality(rs, ax=ax, title=_DOMAIN_STYLE[domain][2])
        ax.legend().set_visible(False)
    fig.tight_layout()
    return fig


def plot_hyperbolic_flake(p, q=None, depth=2, ax=None, title=None,
                          show_bonds=True):
    """A finite patch of the {p,q} tessellation in the Poincare disk.

    Cell centres are shaded by their distance from the origin in the
    breadth-first order, and the unit circle is drawn for reference. The
    picture makes the point that :func:`pyCICY.hyperbolic.boundary_fraction`
    quantifies: the cells crowd towards the boundary circle, so most of them
    are always near the rim.
    """
    import matplotlib.pyplot as plt
    from . import hyperbolic as _hyp

    q = p if q is None else q
    if ax is None:
        ax = plt.figure(figsize=(5.4, 5.4)).add_subplot(111)

    cells = _hyp.flake(p, q, depth)
    pts = list(cells)
    lv = np.array([cells[z] for z in pts], dtype=float)

    circle = plt.Circle((0, 0), 1.0, fill=False, color="0.6", lw=1.0)
    ax.add_patch(circle)

    if show_bonds:
        # bonds between adjacent cell centres, drawn as true geodesics:
        # in the disk these are circular arcs, not chords
        gens = _hyp.generators(p, q)
        index = {_hyp._key(z): z for z in pts}
        drawn = set()
        for z in pts:
            for M in gens:
                w = index.get(_hyp._key(_hyp.apply(M, z)))
                if w is None:
                    continue
                key = tuple(sorted([_hyp._key(z), _hyp._key(w)]))
                if key in drawn:
                    continue
                drawn.add(key)
                arc = _hyp.geodesic(z, w, n=24)
                ax.plot(arc.real, arc.imag, color="0.78", lw=0.5, zorder=1)

    sc = ax.scatter([z.real for z in pts], [z.imag for z in pts],
                    c=lv, cmap="viridis", s=14, zorder=3, edgecolors="none")
    cb = ax.figure.colorbar(sc, ax=ax, fraction=0.046)
    cb.set_label("rings from the centre")

    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal")
    ax.axis("off")
    frac = _hyp.boundary_fraction(p, q, depth)
    ax.set_title(title if title is not None
                 else "{%d,%d} cell centres, %d cells, boundary fraction %.3f"
                      % (p, q, len(pts), frac), fontsize=10)
    return ax


def plot_apolynomial(name, ax=None, title=None, annotate_slopes=True):
    """Newton polygon of a knot's A-polynomial, with its boundary slopes.

    The edge slopes are boundary slopes of incompressible surfaces in the
    knot complement (Cooper, Culler, Gillet, Long and Shalen), and the same
    lattice points are what :class:`pyCICY.quantum_curve.QuantumCurve`
    consumes as a hopping set -- which is the tie between the knot side of
    this package and the quantized-curve side.
    """
    import matplotlib.pyplot as plt
    from . import apolynomial as _ap

    A = _ap.apolynomial(name) if isinstance(name, str) else name
    label = name if isinstance(name, str) else "A"
    if ax is None:
        ax = plt.figure(figsize=(4.6, 4.2)).add_subplot(111)

    pts = sorted(A)
    hull = _ap.newton_polygon(A)
    closed = list(hull) + [hull[0]] if len(hull) > 2 else list(hull)
    ax.plot([p[0] for p in closed], [p[1] for p in closed],
            color="tab:blue", lw=1.6, zorder=2)
    ax.scatter([p[0] for p in pts], [p[1] for p in pts],
               s=[26 + 14 * min(abs(A[p]), 3) for p in pts],
               color="tab:blue", zorder=4)
    for p in pts:
        ax.annotate(str(A[p]), p, textcoords="offset points",
                    xytext=(5, 4), fontsize=7, color="0.35")

    if annotate_slopes and len(hull) > 1:
        n = len(hull)
        pairs = ([(hull[i], hull[(i + 1) % n]) for i in range(n)] if n > 2
                 else [(hull[0], hull[1])])
        for a, b in pairs:
            if b[0] == a[0]:
                continue
            slope = Fraction_(b[1] - a[1], b[0] - a[0])
            mid = (0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]))
            ax.annotate("%s" % slope, mid, textcoords="offset points",
                        xytext=(-16, -2), fontsize=8, color="tab:red")

    ax.set_xlabel("power of $L$")
    ax.set_ylabel("power of $M$")
    # exponents are integers, so do not let matplotlib invent 0.25 steps
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, color="0.93", lw=0.5)
    slopes = ", ".join(str(s) for s in _ap.boundary_slopes(A))
    ax.set_title(title if title is not None
                 else "%s: boundary slopes %s" % (label, slopes), fontsize=10)
    return ax


def plot_colored_jones(s, t, nmax=6, ax=None, title=None):
    """Coefficients of the colored Jones polynomials of a torus knot.

    One row per colour N, with the power of q horizontally and the sign and
    size of the coefficient shown by the marker. The span grows
    quadratically in N, which is the behaviour the degree conjecture
    describes and which drives the asymptotics behind the volume conjecture.
    """
    import matplotlib.pyplot as plt
    from . import apolynomial as _ap

    if ax is None:
        ax = plt.figure(figsize=(7.0, 3.8)).add_subplot(111)

    for N in range(1, nmax + 1):
        poly = _ap.colored_jones_torus(s, t, N)
        exps = sorted(poly.c)
        vals = [poly.c[e] for e in exps]
        pos = [e for e, v in zip(exps, vals) if v > 0]
        neg = [e for e, v in zip(exps, vals) if v < 0]
        sz = lambda es: [18 + 16 * (abs(poly.c[e]) - 1) for e in es]
        ax.scatter(pos, [N] * len(pos), s=sz(pos), color="tab:blue",
                   marker="o", zorder=3)
        ax.scatter(neg, [N] * len(neg), s=sz(neg), color="tab:red",
                   marker="s", zorder=3)
        lo, hi = poly.degrees()
        ax.plot([lo, hi], [N, N], color="0.85", lw=1.0, zorder=1)

    ax.set_xlabel("power of $q$")
    ax.set_ylabel("colour $N$")
    ax.set_yticks(range(1, nmax + 1))
    ax.scatter([], [], color="tab:blue", marker="o", label="positive")
    ax.scatter([], [], color="tab:red", marker="s", label="negative")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.set_title(title if title is not None
                 else r"$J_N(T(%d,%d))$, span growing quadratically in $N$"
                      % (s, t), fontsize=10)
    return ax


# ------------------------------------------------- heterotic model building

def plot_yukawa_texture(conf, summands, ax=None, kind="down", title=None,
                        refine=True):
    """The Yukawa texture as a grid of allowed, forbidden and vanishing.

    Three outcomes, and the distinction between them is the content:

    * **charge-forbidden** -- the line bundles do not cancel, so the cup
      product does not land in ``H^3(O_X)``. Symmetry.
    * **texture zero** -- charge-allowed but some cohomology group is
      zero-dimensional, so there is no field to couple. Geometry.
    * **present** -- survives both, and with ``refine`` also survives the
      cup-product rules of :mod:`pyCICY.theories.representatives`, which can
      kill a coupling the dimensions call present.

    Drawing them together makes the point that a table of numbers does not:
    almost everything is a texture zero, and the survivors are sparse.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from .theories import yukawa as _y

    if ax is None:
        ax = plt.figure(figsize=(6.0, 3.4)).add_subplot(111)
    t = _y.texture(conf, summands, kind=kind)
    recs = t[kind]

    killed = set()
    if refine and kind == "up":
        from .theories import representatives as _rp
        try:
            rr = _rp.refine_texture(conf, summands, kind="up")
            killed = {r["pattern"] for r in rr["records"]
                      if r["status"] == "vanishes"}
        except Exception:                                        # noqa: BLE001
            killed = set()

    codes, labels = [], []
    for r in recs:
        if not r["charge_allowed"]:
            codes.append(0)
        elif not r["present"]:
            codes.append(1)
        elif r["pattern"] in killed:
            codes.append(2)
        else:
            codes.append(3)
        labels.append(" ".join(r["pattern"]))

    cmap = ListedColormap(["0.92", "tab:red", "tab:orange", "tab:green"])
    arr = np.array(codes)[None, :]
    # a short strip, not a tall block: the row carries no information and a
    # square-ish cell reads as a grid of outcomes rather than a heat map
    ax.imshow(arr, cmap=cmap, vmin=0, vmax=3, aspect=0.6)
    ax.set_yticks([])
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    glyph = {0: "-", 1: "0", 2: "v", 3: "*"}
    for i, c in enumerate(codes):
        ax.text(i, 0, glyph[c], ha="center", va="center", fontsize=7,
                color="0.4" if c == 0 else "w")

    # only show legend entries that actually occur, so the key describes the
    # picture rather than the general scheme
    scheme = [("charge-forbidden", "0.92", 0), ("texture zero", "tab:red", 1),
              ("cup product vanishes", "tab:orange", 2),
              ("present", "tab:green", 3)]
    used = [(lab, col) for lab, col, code in scheme if code in set(codes)]
    handles = [plt.Rectangle((0, 0), 1, 1, color=col) for _, col in used]
    ax.legend(handles, [lab for lab, _ in used],
              frameon=False, fontsize=7, ncol=min(len(used), 3),
              loc="lower center", bbox_to_anchor=(0.5, 1.02))
    n_present = sum(1 for c in codes if c == 3)
    if title is None:
        title = "%s-type Yukawa texture: %d of %d survive" % (
            kind, n_present, len(codes))
    ax.set_title(title, fontsize=9, pad=22)
    return ax


def plot_search_funnel(counts=None, ax=None, title=None):
    """The search for a viable model, stage by stage, on a log scale.

    Two phases, and they count different things, so the plot separates them
    rather than running them into one misleading series. The first counts
    *manifolds*: how many favourable threefolds admit a viable triple at all.
    The second counts *models* on one surviving manifold: how many rank-five
    sums pass each condition.

    The shape is the argument. Within each phase the drop is severe, and the
    binding constraint is not the topology --- the index and anomaly admit
    enormous families --- but joint poly-stability.
    """
    import matplotlib.pyplot as plt

    if counts is None:
        counts = [("favourable CY3\n($\\leq$3 factors)", 111, "manifolds"),
                  ("admits a\nviable triple", 36, "manifolds"),
                  ("$c_1{=}0$, index,\nanomaly", 1500, "models"),
                  ("jointly\npoly-stable", 4, "models"),
                  ("a coupling\nsurvives", 4, "models"),
                  ("vector-like\nliftable", 4, "models")]
    if ax is None:
        ax = plt.figure(figsize=(6.8, 3.4)).add_subplot(111)
    names = [c[0] for c in counts]
    vals = [max(c[1], 0.5) for c in counts]
    phase = [c[2] for c in counts]
    colors = ["tab:blue" if p == "manifolds" else "tab:orange"
              for p in phase]
    colors[-1] = "tab:green"
    ax.bar(range(len(vals)), vals, color=colors)
    ax.set_yscale("log")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=7)
    for i, c in enumerate(counts):
        ax.text(i, max(c[1], 0.5) * 1.25, str(c[1]), ha="center", fontsize=8)

    # the phases count different objects, so mark the boundary
    split = next((i for i, p in enumerate(phase) if p == "models"), None)
    if split:
        ax.axvline(split - 0.5, color="0.5", lw=0.9, ls=":")
        ax.text(split / 2.0 - 0.5, ax.get_ylim()[1] * 0.45, "manifolds",
                ha="center", fontsize=7, color="tab:blue")
        ax.text((split + len(vals) - 1) / 2.0, ax.get_ylim()[1] * 0.45,
                "models on one manifold", ha="center", fontsize=7,
                color="tab:orange")
    ax.set_ylabel("count (log)", fontsize=8)
    ax.grid(True, axis="y", color="0.93", lw=0.5)
    ax.set_title(title or "Narrowing to a viable model", fontsize=9)
    return ax


def plot_unification(betas=None, ax=None, title=None, mz=91.1876,
                     a1_inv=59.0, a2_inv=29.57, a3=0.1181):
    """One-loop running of the three gauge couplings.

    The three lines meet at a point only for the right spectrum, and whether
    they do is decided by the one-loop coefficients, which come from the
    chiral spectrum. Drawing the supersymmetric and non-supersymmetric cases
    together shows the discriminator directly: one meets, the other misses by
    a wide margin.
    """
    import matplotlib.pyplot as plt
    from fractions import Fraction as _F
    from .theories import running as _r

    if ax is None:
        ax = plt.figure(figsize=(5.4, 3.6)).add_subplot(111)
    sets = betas or [("MSSM", _r.beta_coefficients(3, 1), "-"),
                     ("Standard Model",
                      (_F(41, 10), _F(-19, 6), _F(-7)), "--")]
    t = np.linspace(0, 40, 200)          # t = ln(mu/M_Z)
    colors = ["tab:blue", "tab:orange", "tab:green"]
    for name, b, ls in sets:
        starts = [a1_inv, a2_inv, 1.0 / a3]
        for i, (bi, s0) in enumerate(zip(b, starts)):
            ax.plot(t, s0 - float(bi) * t / (2 * np.pi), ls=ls,
                    color=colors[i], lw=1.3,
                    label=(r"$\alpha_%d^{-1}$ (%s)" % (i + 1, name)))
    ax.set_xlabel(r"$\ln(\mu/M_Z)$", fontsize=8)
    ax.set_ylabel(r"$\alpha_i^{-1}$", fontsize=8)
    ax.set_ylim(0, 65)
    ax.grid(True, color="0.93", lw=0.5)
    ax.legend(frameon=False, fontsize=6, ncol=2)
    ax.set_title(title or "Gauge coupling unification", fontsize=9)
    return ax


def plot_racetrack(ax=None, title=None, lambda_qcd=0.2, m_gut=5.0e17):
    """Condensation-scale ratio required by each reachable racetrack.

    Only three ``SU(N_1) x SU(N_2)`` embed in the hidden ``E_8`` by the
    standard chains, and the ratio each needs spans nine orders of magnitude.
    The plot makes the cost of the reachable options visible at a glance.
    """
    import matplotlib.pyplot as plt
    from .theories import moduli as _m

    if ax is None:
        ax = plt.figure(figsize=(5.0, 3.0)).add_subplot(111)
    reach = _m.reachable_racetracks(lambda_qcd=lambda_qcd, m_gut=m_gut)
    names = ["SU(%d)x SU(%d)\n$p=%s$" % (r["N1"], r["N2"], r["exponent"])
             for r in reach]
    vals = [r["ratio_required"] for r in reach]
    ax.bar(range(len(vals)), vals, color="tab:purple")
    ax.set_yscale("log")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=7)
    for i, v in enumerate(vals):
        ax.text(i, v * 1.4, "%.1e" % v, ha="center", fontsize=7)
    ax.set_ylabel("condensation ratio $R$ required", fontsize=8)
    ax.grid(True, axis="y", color="0.93", lw=0.5)
    ax.set_title(title or
                 r"Reachable racetracks ($\alpha_{\rm GUT}\approx 1/20$ for all)",
                 fontsize=9)
    return ax


def plot_equivariant_character(action=None, charges=None, ax=None, title=None):
    """Index characters of a group action, showing equidistribution.

    For a freely acting group every Lefschetz number vanishes, so the
    character is a multiple of the regular representation --- a flat bar
    chart. A non-free action is visibly uneven. The flatness is the whole
    reason the equivariant structure does not affect the chiral spectrum.
    """
    import matplotlib.pyplot as plt
    from . import equivariant as _e

    if ax is None:
        ax = plt.figure(figsize=(5.4, 3.0)).add_subplot(111)
    if action is None:
        action = _e.TETRAQUADRIC_Z2()
    if charges is None:
        charges = [[1, 1, 1, 1], [-2, -2, -1, 2], [2, 2, 1, -2]]
    width = 0.8 / len(charges)
    n = action.n
    for j, k in enumerate(charges):
        ch = action.euler(list(k))
        ax.bar(np.arange(n) + j * width - 0.4, ch, width=width,
               label="$k=%s$" % (list(k),))
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xticks(range(n))
    ax.set_xlabel(r"$\Gamma$-charge", fontsize=8)
    ax.set_ylabel("index multiplicity", fontsize=8)
    ax.legend(frameon=False, fontsize=6)
    ax.grid(True, axis="y", color="0.93", lw=0.5)
    free = all(_e.is_regular_multiple(action.euler(list(k)))[0]
               for k in charges)
    ax.set_title(title or ("Equivariant index characters -- %s"
                           % ("equidistributed (free action)" if free
                              else "uneven (not free)")), fontsize=9)
    return ax
