r"""
pyCICY.hyperbolic -- hyperbolic lattices and automorphic Bloch theory.

The setting
-----------
Maciejko and Rayan, "Automorphic Bloch theorems for hyperbolic lattices",
PNAS 119(9) e2116869119 (2022), give a Bloch theory for lattices in the
hyperbolic plane. On a {p,q} tessellation the ordinary Bloch theorem fails,
because the translation group is a Fuchsian group and is not abelian. The
resolution is to impose periodic boundary conditions by a normal subgroup,
which compactifies the lattice onto a genus-g Riemann surface, and to expand
in *automorphic* functions: eigenstates transform under translations by a
unitary representation of the Fuchsian group. One-dimensional
representations give an ordinary Brillouin zone, the Jacobian torus
T^{2g}, and higher-dimensional irreducible representations give further
sectors which no torus can capture.

This module implements the geometry, the abelian sector exactly, and an
explicit family of higher-dimensional irreducible representations. What it
does *not* do is enumerate the normal subgroups of a Fuchsian group, which
is the step that in general wants GAP or Magma; see "Scope" below.

Geometry, and a warning about the standard formulas
---------------------------------------------------
For a regular {p,q} tessellation, which exists in the hyperbolic plane
exactly when (p-2)(q-2) > 4, the relevant lengths are

    cosh R      = cot(pi/p) cot(pi/q)      centre to vertex
    cosh r      = cos(pi/q) / sin(pi/p)    centre to edge midpoint
    cosh(l/2)   = cos(pi/p) / sin(pi/q)    half an edge

These three are easy to confuse with one another, and the tests derive them
rather than trusting them: :func:`solve_circumradius` finds R numerically by
demanding that the interior angle of the cell really be 2 pi / q, and the
test suite checks the closed forms against it across a range of tilings.

Translations and the genus
--------------------------
Isometries are represented by SU(1,1) matrices acting on the Poincare disk
by Moebius transformations. For the {4g, 4g} tessellations -- the case in
which the lattice has one site per unit cell, so that the tessellation is its
own Bravais lattice -- the 4g translations through the edge midpoints, each
of hyperbolic length 2r, pair up into inverses and satisfy

    gamma_0 gamma_1^{-1} gamma_2 gamma_3^{-1} ... gamma_{4g-1}^{-1} = 1 ,

the relator of the regular 4g-gon with opposite sides identified. It is worth
being explicit that this is *not* the canonical surface relation
prod_i [a_i, b_i] = 1: the two presentations describe the same group but in
different generators, and using the canonical word on these generators gives
a matrix nowhere near the identity. :func:`relator_residual` evaluates the
correct word, and Gauss-Bonnet independently confirms the genus, since the
cell area (p-2) pi - 2 pi p / q equals 4 pi (g - 1) = 2 pi |chi| exactly.

The abelian sector
------------------
With one orbital per cell and uniform hopping over the 4g neighbours, a
one-dimensional representation sends gamma_j to a phase and the Bloch
Hamiltonian collapses to a number,

    E(k) = 2 t sum_{j=1}^{2g} cos k_j ,     k in T^{2g} ,

which is the band structure of a 2g-dimensional hypercubic lattice. The
hyperbolic lattice's abelian Brillouin zone is a torus of twice the genus,
and its density of states is the hypercubic one.

Higher-dimensional sectors
--------------------------
For an N-dimensional representation the Hamiltonian is an N x N matrix

    H = t sum_{j=0}^{2g-1} ( e^{i theta_j} U_j + h.c. ) ,

with the U_j unitary matrices obeying the relator above. :func:`weyl_pair`
builds the clock and shift matrices X, Z with

    [X^a, Z^b] = omega^{-ab} 1 ,     omega = exp(2 pi i / N),

whose commutator is a scalar; that is what makes explicit solutions
possible. :func:`weyl_rep` returns the solution (U_0, U_1, U_2, U_3, ...) =
(Z, 1, X, Z^{-1}, 1, ...), which satisfies the relator exactly and is
irreducible because X and Z already generate the full algebra on C^N.
Scalar phases cancel from the relator, since every generator appears once
with each sign, so the twists theta_j range freely over T^{2g} and each N
gives a whole family. :func:`search_reps` finds the other solutions built
from powers of X and Z.

Why any of this is necessary
----------------------------
:func:`flake` builds a finite patch of the lattice and
:func:`boundary_fraction` measures how much of it is boundary. In the
hyperbolic plane the answer does not go to zero: for {8,8} it sits near 0.86
at every depth, because the number of cells grows exponentially with the
radius and so a fixed fraction of them always lies on the rim. A finite
patch is therefore never a good approximation to the bulk, which is exactly
why periodic boundary conditions and an automorphic Bloch theory are needed
rather than a large-flake extrapolation. :func:`compare_sectors` puts the
flake spectrum next to the abelian band and the higher-dimensional ones so
the three can be seen not to agree.

Scope
-----
Implemented: the geometry of any {p,q}; the Fuchsian generators, relator and
genus for the {4g,4g} family; the abelian sector exactly; explicit
N-dimensional irreducible representations and their Hamiltonians; finite
flakes.

Not implemented: enumeration of the normal subgroups of a Fuchsian group,
which is what would give the *complete* set of irreducible representations
and hence the full spectrum. That is a computational group theory problem
better handed to GAP, and nothing here should be read as computing the whole
hyperbolic band structure. The sectors this module produces are genuine
sectors; they are not claimed to exhaust anything.
"""

import cmath
import math

import numpy as np

__all__ = [
    "exists", "cell_area", "circumradius", "inradius", "edge_length",
    "solve_circumradius", "interior_angle",
    "translation", "rotation", "apply", "inverse", "distance", "cell_vertices",
    "geodesic",
    "genus", "generators", "relator_word", "relator_residual", "relator_holds",
    "flake", "flake_adjacency", "boundary_fraction",
    "boundary_fraction_limit", "flake_spectrum",
    "abelian_energy", "abelian_spectrum", "abelian_dos", "abelian_bandwidth",
    "weyl_pair", "weyl_rep", "rep_is_valid", "search_reps",
    "bloch_hamiltonian", "sector_spectrum", "compare_sectors",
]

_TOL = 1e-9


# ------------------------------------------------------------------ geometry

def exists(p, q):
    """True when {p,q} tiles the hyperbolic plane, i.e. (p-2)(q-2) > 4."""
    return (p - 2) * (q - 2) > 4


def _check(p, q):
    if not exists(p, q):
        raise ValueError(
            "{%d,%d} is not hyperbolic: (p-2)(q-2) = %d, which must exceed 4"
            % (p, q, (p - 2) * (q - 2)))


def cell_area(p, q):
    """Hyperbolic area of one cell, by Gauss-Bonnet.

    A p-gon with interior angles 2 pi / q has area (p-2) pi - 2 pi p / q.
    """
    _check(p, q)
    return (p - 2) * math.pi - 2.0 * math.pi * p / q


def circumradius(p, q):
    """Distance from a cell centre to a vertex: cosh R = cot(pi/p) cot(pi/q)."""
    _check(p, q)
    return math.acosh(1.0 / (math.tan(math.pi / p) * math.tan(math.pi / q)))


def inradius(p, q):
    """Distance from a cell centre to an edge midpoint.

    ``cosh r = cos(pi/q) / sin(pi/p)``. Adjacent cell centres are 2r apart,
    which is the translation length of the side-pairing generators.
    """
    _check(p, q)
    return math.acosh(math.cos(math.pi / q) / math.sin(math.pi / p))


def edge_length(p, q):
    """Length of one edge: cosh(l/2) = cos(pi/p) / sin(pi/q)."""
    _check(p, q)
    return 2.0 * math.acosh(math.cos(math.pi / p) / math.sin(math.pi / q))


def cell_vertices(p, q):
    """The p vertices of the cell centred at the origin, in the disk."""
    rad = math.tanh(circumradius(p, q) / 2.0)
    return [rad * cmath.exp(2j * math.pi * k / p) for k in range(p)]


def interior_angle(p, radius):
    """Interior angle of the regular p-gon whose circumradius is ``radius``.

    Used to derive the circumradius rather than assume it: the correct value
    is the one making this equal to 2 pi / q.
    """
    rad = math.tanh(radius / 2.0)
    v = [rad * cmath.exp(2j * math.pi * k / p) for k in range(p)]
    to_origin = lambda z: (z - v[0]) / (1 - np.conj(v[0]) * z)
    return abs(cmath.phase(to_origin(v[1]) / to_origin(v[-1])))


def solve_circumradius(p, q, hi=50.0):
    """Find R numerically by bisection on ``interior_angle(p, R) = 2 pi / q``.

    Independent of :func:`circumradius`, and the two are checked against each
    other in the tests.
    """
    _check(p, q)
    target = 2.0 * math.pi / q
    lo, hi = 1e-9, hi
    f = lambda R: interior_angle(p, R) - target
    if f(lo) * f(hi) > 0:
        raise ValueError("no bracketing interval for {%d,%d}" % (p, q))
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(lo) * f(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


# --------------------------------------------------------------- isometries

def translation(d, theta=0.0):
    """SU(1,1) matrix translating the origin a distance ``d`` in direction ``theta``."""
    c, s = math.cosh(d / 2.0), math.sinh(d / 2.0)
    return np.array([[c, s * cmath.exp(1j * theta)],
                     [s * cmath.exp(-1j * theta), c]], dtype=complex)


def rotation(angle):
    """SU(1,1) matrix rotating the disk about the origin."""
    return np.array([[cmath.exp(1j * angle / 2.0), 0.0],
                     [0.0, cmath.exp(-1j * angle / 2.0)]], dtype=complex)


def apply(M, z):
    """Act on a point of the Poincare disk by the Moebius transformation."""
    return (M[0, 0] * z + M[0, 1]) / (M[1, 0] * z + M[1, 1])


def inverse(M):
    """Inverse of a unit-determinant 2x2 matrix, via the adjugate.

    Written as [[d, -b], [-c, a]] rather than by conjugating entries. The
    conjugated form happens to agree for a pure translation, whose diagonal
    is real, but is wrong for a rotation or any composite -- which is what
    section [2] of the test suite checks.
    """
    return np.array([[M[1, 1], -M[0, 1]],
                     [-M[1, 0], M[0, 0]]], dtype=complex)


def distance(z, w=0.0):
    """Hyperbolic distance between two points of the disk."""
    return 2.0 * np.arctanh(abs((z - w) / (1 - np.conj(w) * z)))


def geodesic(z, w, n=40):
    """Sample the hyperbolic geodesic from ``z`` to ``w`` in the disk.

    Geodesics of the Poincare disk are circular arcs meeting the boundary at
    right angles, not straight chords, so they cannot be drawn by joining
    endpoints. This maps ``z`` to the origin, where geodesics through the
    origin *are* straight, samples the segment, and maps back.
    """
    to0 = lambda u: (u - z) / (1 - np.conj(z) * u)
    back = lambda u: (u + z) / (1 + np.conj(z) * u)
    end = to0(w)
    if abs(end) < 1e-15:
        return np.array([z] * n)
    d = 2.0 * np.arctanh(abs(end))
    unit = end / abs(end)
    # parametrised by arc length, so the halfway sample really is the
    # hyperbolic midpoint rather than the Euclidean one
    return np.array([back(np.tanh(t * d / 2.0) * unit)
                     for t in np.linspace(0.0, 1.0, n)])


# ------------------------------------------------- Fuchsian group and genus

def genus(p):
    """Genus of the surface obtained from the {p,p} cell, for p = 4g."""
    if p % 4:
        raise ValueError("genus is defined here for {4g,4g}; got p = %d" % p)
    return p // 4


def generators(p, q=None):
    """The p side-pairing translations of the {p,q} tessellation.

    Each carries the cell at the origin onto one of its p neighbours, by a
    translation of length 2r through the corresponding edge midpoint. They
    pair into inverses, ``generators[k]`` with ``generators[k + p//2]``.
    """
    q = p if q is None else q
    d = 2.0 * inradius(p, q)
    return [translation(d, math.pi / p + 2.0 * math.pi * k / p)
            for k in range(p)]


def relator_word(p):
    """Indices of the relator of the regular p-gon with opposite sides paired.

    The word is gamma_0 gamma_1^{-1} gamma_2 gamma_3^{-1} ..., returned as a
    list of ``(index, exponent)``. This is deliberately not the canonical
    surface word prod [a_i, b_i]; see the module docstring.
    """
    return [(k, 1 if k % 2 == 0 else -1) for k in range(p)]


def relator_residual(p, q=None):
    """How far the relator word is from the identity, in matrix norm."""
    gens = generators(p, q)
    prod = np.eye(2, dtype=complex)
    for k, e in relator_word(p):
        prod = prod @ (gens[k] if e > 0 else inverse(gens[k]))
    return min(float(np.max(np.abs(prod - np.eye(2)))),
               float(np.max(np.abs(prod + np.eye(2)))))


def relator_holds(p, q=None, tol=1e-6):
    """True when the side-pairing generators satisfy the relator."""
    return relator_residual(p, q) < tol


# ------------------------------------------------------------------- flakes

def _key(z, places=7):
    """Rounded coordinates, so points can be deduplicated in a dict.

    Cell centres of a hyperbolic tiling crowd towards the boundary of the
    disk as the depth grows, so the rounding is a genuine limit on how deep
    a flake can go before distinct cells become indistinguishable in double
    precision. :func:`flake` reports the depth actually reached.
    """
    return (round(z.real, places), round(z.imag, places))


def flake(p, q=None, depth=2, places=7):
    """Cell centres of a finite patch, as ``{point: depth}``.

    Breadth-first from the origin, applying the side-pairing translations.
    Deduplication is by rounded coordinates, which keeps the construction
    linear in the number of cells rather than quadratic.
    """
    q = p if q is None else q
    gens = generators(p, q)
    seen = {0j: 0}
    keys = {_key(0j, places): 0j}
    frontier = [0j]
    for level in range(depth):
        nxt = []
        for z in frontier:
            for M in gens:
                w = apply(M, z)
                k = _key(w, places)
                if k not in keys:
                    keys[k] = w
                    seen[w] = level + 1
                    nxt.append(w)
        frontier = nxt
        if not frontier:
            break
    return seen


def flake_adjacency(p, q=None, depth=2, points=None):
    """Adjacency matrix of a flake, together with its ordered points."""
    q = p if q is None else q
    pts = list(points if points is not None else flake(p, q, depth))
    gens = generators(p, q)
    index = {_key(z): i for i, z in enumerate(pts)}
    n = len(pts)
    A = np.zeros((n, n))
    for i, z in enumerate(pts):
        for M in gens:
            j = index.get(_key(apply(M, z)))
            if j is not None:
                A[i, j] = 1.0
    return A, pts


def boundary_fraction(p, q=None, depth=2):
    """Fraction of flake cells with fewer than p neighbours inside the flake.

    This does not tend to zero as the flake grows. The number of cells grows
    exponentially with the radius, so a fixed fraction always sits on the
    rim; for {8,8} the value hovers near 0.86 at every depth. It is the
    quantitative reason a finite patch cannot stand in for the bulk, and
    hence the reason automorphic Bloch theory is needed at all.
    """
    A, pts = flake_adjacency(p, q, depth)
    deg = A.sum(axis=1)
    interior = int(np.sum(deg >= p - _TOL))
    return 1.0 - interior / float(len(pts))


def boundary_fraction_limit(p):
    """Limiting boundary fraction of a large {p,p} flake, ``(p-2)/(p-1)``.

    Each ring of the flake is larger than the previous one by a factor of
    p - 1, so a flake of depth n holds about (p-1)^n cells of which about
    (p-1)^{n-1} are interior. The fraction on the rim therefore tends to
    (p-2)/(p-1) rather than to zero: 6/7 for {8,8}, 10/11 for {12,12}.
    Verified against :func:`boundary_fraction` in the tests.
    """
    return (p - 2) / float(p - 1)


def flake_spectrum(p, q=None, depth=2, t=1.0):
    """Eigenvalues of the hopping Hamiltonian on a finite flake."""
    A, _ = flake_adjacency(p, q, depth)
    return np.linalg.eigvalsh(t * A)


# ---------------------------------------------------------- abelian sector

def abelian_energy(k, t=1.0):
    """E(k) = 2 t sum_j cos k_j on the Jacobian torus T^{2g}."""
    return 2.0 * t * float(np.sum(np.cos(np.asarray(k, dtype=float))))


def abelian_spectrum(g, nk=12, t=1.0, seed=None):
    """Sample E(k) over T^{2g}.

    A regular grid costs nk^{2g} points, which is already 20736 at genus 2
    and nk = 12, so for higher genus the sampling is random unless the grid
    is small enough to be affordable.
    """
    dim = 2 * g
    if nk ** dim <= 200000:
        axes = np.linspace(0.0, 2.0 * np.pi, nk, endpoint=False)
        grids = np.meshgrid(*([axes] * dim), indexing="ij")
        return 2.0 * t * sum(np.cos(G) for G in grids).ravel()
    rng = np.random.default_rng(seed)
    ks = rng.uniform(0.0, 2.0 * np.pi, size=(200000, dim))
    return 2.0 * t * np.cos(ks).sum(axis=1)


def abelian_bandwidth(g, t=1.0):
    """Full width of the abelian band, 8 g |t|, spanning +- 4 g t."""
    return 8.0 * g * abs(t)


def abelian_dos(g, nk=12, bins=120, t=1.0, seed=None):
    """Density of states of the abelian sector, as (centres, counts)."""
    E = abelian_spectrum(g, nk=nk, t=t, seed=seed)
    counts, edges = np.histogram(E, bins=bins, density=True)
    return 0.5 * (edges[:-1] + edges[1:]), counts


# ------------------------------------------------------- higher-dim sectors

def weyl_pair(N):
    """The clock and shift matrices (X, Z) on C^N, and omega.

    They are unitary, satisfy ``[X^a, Z^b] = omega^{-ab} 1``, and generate an
    irreducible algebra. The commutator being a *scalar* is what allows the
    surface relator to be solved in closed form.
    """
    if N < 1:
        raise ValueError("N must be positive")
    omega = cmath.exp(2j * math.pi / N)
    Z = np.diag([omega ** j for j in range(N)]).astype(complex)
    X = np.roll(np.eye(N, dtype=complex), 1, axis=0)
    return X, Z, omega


def weyl_rep(N, g=2):
    """An N-dimensional representation obeying the {4g,4g} relator.

    Returns the 2g independent matrices ``U_0, ..., U_{2g-1}``; the remaining
    2g generators are their inverses. The construction is
    ``(Z, 1, X, Z^{-1}, 1, ..., 1)``: the relator collapses because the
    trailing identities drop out and ``Z X Z . Z^{-1} X^{-1} Z^{-1} = 1``.

    For N > 1 the representation is irreducible, since X and Z already
    generate the whole matrix algebra on C^N. It is not faithful, and for
    g > 2 it factors through the genus-two quotient; :func:`search_reps`
    turns up many others.
    """
    if g < 2:
        raise ValueError("genus must be at least 2")
    X, Z, _ = weyl_pair(N)
    eye = np.eye(N, dtype=complex)
    U = [Z, eye.copy(), X, np.conj(Z).T.copy()]
    U += [eye.copy() for _ in range(2 * g - 4)]
    return U[:2 * g]


def rep_is_valid(U, tol=1e-8):
    """Check unitarity and the relator for a list of 2g matrices."""
    U = [np.asarray(M, dtype=complex) for M in U]
    if len(U) % 2:
        raise ValueError("expected 2g matrices, got %d" % len(U))
    N = U[0].shape[0]
    eye = np.eye(N, dtype=complex)
    for M in U:
        if not np.allclose(M @ M.conj().T, eye, atol=tol):
            return False
    inv = [M.conj().T for M in U]
    word = eye.copy()
    for j in range(len(U)):                 # gamma_0 gamma_1^-1 ...
        word = word @ (U[j] if j % 2 == 0 else inv[j])
    for j in range(len(U)):                 # then the inverse half
        word = word @ (inv[j] if j % 2 == 0 else U[j])
    return bool(np.allclose(word, eye, atol=tol))


def search_reps(N, g=2, limit=None, nonabelian_only=True):
    """Solutions of the relator built from powers of the Weyl pair.

    Each generator is taken to be ``X^a Z^b``; the search runs over the
    exponents. Returns a list of exponent tuples
    ``((a_0, b_0), ..., (a_{2g-1}, b_{2g-1}))``.
    """
    import itertools

    X, Z, _ = weyl_pair(N)

    def U(a, b):
        return (np.linalg.matrix_power(X, a) @ np.linalg.matrix_power(Z, b))

    out = []
    for exps in itertools.product(range(N), repeat=4 * g):
        pairs = tuple((exps[2 * j], exps[2 * j + 1]) for j in range(2 * g))
        mats = [U(a, b) for a, b in pairs]
        if not rep_is_valid(mats):
            continue
        if nonabelian_only and len(mats) >= 3:
            if np.allclose(mats[0] @ mats[2], mats[2] @ mats[0]):
                continue
        out.append(pairs)
        if limit is not None and len(out) >= limit:
            break
    return out


def bloch_hamiltonian(U, theta, t=1.0):
    """H = t sum_j ( e^{i theta_j} U_j + h.c. ), an N x N Hermitian matrix.

    With one-dimensional U_j this reduces to ``2 t sum_j cos theta_j``, the
    abelian sector.
    """
    U = [np.asarray(M, dtype=complex) for M in U]
    theta = np.asarray(theta, dtype=float)
    if len(theta) != len(U):
        raise ValueError("got %d matrices but %d twists" % (len(U), len(theta)))
    N = U[0].shape[0]
    H = np.zeros((N, N), dtype=complex)
    for M, th in zip(U, theta):
        term = t * cmath.exp(1j * th) * M
        H += term + term.conj().T
    return H


def sector_spectrum(N, g=2, samples=4000, t=1.0, seed=0, U=None):
    """Eigenvalues of the N-dimensional sector, sampled over the twists."""
    U = weyl_rep(N, g) if U is None else U
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(samples):
        theta = rng.uniform(0.0, 2.0 * np.pi, size=len(U))
        out.append(np.linalg.eigvalsh(bloch_hamiltonian(U, theta, t=t)))
    return np.sort(np.concatenate(out))


def compare_sectors(g=2, dims=(1, 2, 3), depth=2, samples=2000, t=1.0,
                    seed=0):
    """Abelian band, higher-dimensional sectors and a finite flake, together.

    The three do not agree, and are not meant to: the abelian sector is only
    the one-dimensional part of the representation theory, the flake is a
    finite patch whose boundary never becomes negligible, and the
    N-dimensional sectors here are particular representations rather than a
    complete set.
    """
    p = 4 * g
    rows = []
    for N in dims:
        E = (abelian_spectrum(g, nk=10, t=t) if N == 1
             else sector_spectrum(N, g=g, samples=samples, t=t, seed=seed))
        rows.append({
            "sector": "abelian (T^{%d})" % (2 * g) if N == 1
                      else "%d-dimensional" % N,
            "dim": N,
            "min": float(np.min(E)),
            "max": float(np.max(E)),
            "mean": float(np.mean(E)),
            "n_samples": int(E.size),
        })
    Ef = flake_spectrum(p, p, depth=depth, t=t)
    frac = boundary_fraction(p, p, depth=depth)
    rows.append({
        "sector": "flake (depth %d)" % depth,
        "dim": None,
        "min": float(np.min(Ef)),
        "max": float(np.max(Ef)),
        "mean": float(np.mean(Ef)),
        "n_samples": int(Ef.size),
        "boundary_fraction": frac,
    })
    return {
        "genus": g,
        "tessellation": (p, p),
        "abelian_bandwidth": abelian_bandwidth(g, t),
        "rows": rows,
    }
