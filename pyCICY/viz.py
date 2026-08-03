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

import numpy as np

__all__ = [
    "patch", "patches", "bounds", "from_cicy", "describe",
    "plot", "plot_grid", "write_stl", "DEFAULT_ANGLE",
    "plot_hodge", "plot_node_counts",
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
