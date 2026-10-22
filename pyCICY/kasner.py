"""
pyCICY.kasner -- the Kasner vacuum cosmology, with two independent symbolic routes to its curvature.

The (3+1)-dimensional Kasner metric is

    ds^2 = -dt^2 + t^(2 p1) dx^2 + t^(2 p2) dy^2 + t^(2 p3) dz^2        (c = 1)

an exact, spatially flat, anisotropic *vacuum* solution.  It is vacuum (Ricci = 0)
exactly when the exponents satisfy the two Kasner conditions

    p1 + p2 + p3     = 1      (a plane)
    p1^2 + p2^2 + p3^2 = 1    (a sphere)

Consequences checked here (and in tests/test_kasner.py):

  * Ricci tensor vanishes for every admissible (p1, p2, p3).
  * Riemann vanishes only for (0, 0, 1) -- flat spacetime in disguise.
  * Otherwise one exponent is negative: the volume grows like t while one
    direction contracts.  Isotropy (1/3, 1/3, 1/3) violates the sphere condition,
    so it is not vacuum.
  * Kretschmann scalar  K = R_abcd R^abcd = -16 p1 p2 p3 / t^4  >= 0, which
    diverges at t -> 0: a curvature singularity, not a coordinate artefact.

Parametrisation (Lifshitz-Khalatnikov), u >= 1:

    p1 = -u / (1+u+u^2),   p2 = (1+u) / (1+u+u^2),   p3 = u (1+u) / (1+u+u^2)

which satisfies both conditions identically and covers every admissible set up to
permutation.  u = 1 gives (-1/3, 2/3, 2/3); u -> infinity gives (0, 0, 1).

Two independent routes to the curvature, as everywhere in this package:

    sympy_curvature      Christoffel symbols, Riemann and Ricci written out in sympy for a diagonal metric
    einsteinpy_curvature the same tensors from EinsteinPy's symbolic module

and the closed form `kretschmann_closed_form` is the third check on both.  `kasner_geodesic` integrates timelike
worldlines exactly (conserved momenta).  Ported from Quantum Gravity Lab (CC0), with its sensor coupling left behind:
a Jaynes-Cummings sensor fed by K(t) is a modelling choice, not an exact result.
"""

from __future__ import annotations

import importlib.util
from typing import Sequence, Tuple

import numpy as np
from scipy.integrate import solve_ivp


def has_einsteinpy() -> bool:
    return importlib.util.find_spec("einsteinpy") is not None


# --------------------------------------------------------------------------
#  Exponents
# --------------------------------------------------------------------------

def kasner_exponents(u) -> Tuple:
    """
    Kasner exponents (p1 <= 0 <= p2 <= p3) for the parameter u >= 1.

    Works with ints/floats and with exact types (fractions.Fraction, sympy
    Rational): the arithmetic is generic, so exact input gives exact exponents.
    """
    if u < 1:
        raise ValueError("u must be >= 1 (u < 1 gives the same exponents, reordered)")
    d = 1 + u + u * u
    return (-u / d, (1 + u) / d, u * (1 + u) / d)


def kasner_conditions(p: Sequence) -> Tuple:
    """(sum p_i, sum p_i^2): both equal 1 for a vacuum Kasner solution."""
    return sum(p), sum(pi * pi for pi in p)


def is_vacuum_kasner(p: Sequence, tol: float = 1e-12) -> bool:
    s1, s2 = kasner_conditions(p)
    return abs(s1 - 1) <= tol and abs(s2 - 1) <= tol


# --------------------------------------------------------------------------
#  Singularity morphology
# --------------------------------------------------------------------------

def singularity_type(p: Sequence[float], direction: str = "singularity", tol: float = 1e-12) -> str:
    """
    Morphology of the scale factors a_i = t^p_i at the end of time selected by `direction`:

        "singularity"  t -> 0+   (the Big Bang / Big Crunch end)
        "late"         t -> infinity

    An axis with a_i -> 0 is "collapsing", a_i -> infinity "stretching", p_i = 0 "finite".  Then

        point    all three collapse
        cigar    two collapse, one stretches
        barrel   two collapse, one stays finite
        pancake  exactly one collapses (the others stretch or stay finite)
        none     no axis collapses

    This classifies the scale factors only.  It says nothing about curvature: (0, 0, 1) is a "pancake" in this
    sense but is flat spacetime (check `is_flat_kasner`), and the "late" end of any vacuum Kasner solution
    has a collapsing axis but K -> 0, so it is not a singularity.  The literature's wording differs
    (Dechant et al.: one radius -> 0, the others finite; Cotsakis & Yefremov: two expand, one contracts);
    "exactly one collapses" covers both.
    """
    if direction not in ("singularity", "late"):
        raise ValueError("direction must be 'singularity' or 'late'")
    sign = 1.0 if direction == "singularity" else -1.0       # t -> infinity is t -> 0 with p -> -p
    q = [sign * float(x) for x in p]
    collapsing = sum(x > tol for x in q)
    stretching = sum(x < -tol for x in q)
    if collapsing == 3:
        return "point"
    if collapsing == 2:
        return "cigar" if stretching == 1 else "barrel"
    if collapsing == 1:
        return "pancake"
    return "none"


def is_flat_kasner(p: Sequence[float], tol: float = 1e-12) -> bool:
    """True when a vacuum Kasner solution is flat: the exponents are a permutation of (0, 0, 1)."""
    return is_vacuum_kasner(p, tol) and sorted(abs(float(x)) for x in p)[1] <= tol


# --------------------------------------------------------------------------
#  Closed forms (no EinsteinPy needed)
# --------------------------------------------------------------------------

def kretschmann_closed_form(p: Sequence, t):
    """K = -16 p1 p2 p3 / t^4 for vacuum Kasner exponents p (valid only when is_vacuum_kasner)."""
    p1, p2, p3 = p
    return -16 * p1 * p2 * p3 / t ** 4


def kasner_volume_exponent(p: Sequence):
    """Spatial volume element ~ t^(p1+p2+p3); equals t^1 for vacuum Kasner."""
    return sum(p)


# --------------------------------------------------------------------------
#  EinsteinPy symbolic verification
# --------------------------------------------------------------------------

def kasner_metric(p: Sequence):
    """
    EinsteinPy MetricTensor for the Kasner metric with exponents p, in
    coordinates (t, x, y, z).  Returns (metric, (t, x, y, z)).
    """
    if not has_einsteinpy():
        raise ImportError("EinsteinPy is required: pip install einsteinpy")
    import sympy as sp
    from einsteinpy.symbolic import MetricTensor

    t, x, y, z = sp.symbols("t x y z", positive=True)
    g = sp.diag(-1, t ** (2 * p[0]), t ** (2 * p[1]), t ** (2 * p[2]))
    return MetricTensor(g.tolist(), [t, x, y, z], config="ll"), (t, x, y, z)


def einsteinpy_curvature(p: Sequence) -> dict:
    """
    Curvature of the Kasner metric with exponents p, from EinsteinPy's symbolic
    tensors.  Pass exact exponents (sympy Rational) to get exact zeros.

    Returns a dict:
        ricci_is_zero    bool
        riemann_is_zero  bool
        kretschmann      sympy expression in t
        t                the sympy symbol t
    """
    import sympy as sp
    from einsteinpy.symbolic import ChristoffelSymbols, RicciTensor, RiemannCurvatureTensor

    metric, (t, _, _, _) = kasner_metric(p)
    riem = RiemannCurvatureTensor.from_christoffels(ChristoffelSymbols.from_metric(metric))
    ricci = RicciTensor.from_riemann(riem)

    ricci_zero = sp.simplify(sp.Matrix(ricci.tensor().tolist())) == sp.zeros(4, 4)
    r_llll = riem.change_config("llll").tensor()
    riemann_zero = all(sp.simplify(r_llll[a, b, c, d]) == 0
                       for a in range(4) for b in range(4) for c in range(4) for d in range(4))

    # K = R_abcd R^abcd; the metric is diagonal, so raising indices is a product of 1/g_ii
    g = metric.tensor()
    kretschmann = 0
    for a in range(4):
        for b in range(4):
            for c in range(4):
                for d in range(4):
                    v = r_llll[a, b, c, d]
                    if v != 0:
                        kretschmann += v * v / (g[a, a] * g[b, b] * g[c, c] * g[d, d])
    return dict(ricci_is_zero=bool(ricci_zero), riemann_is_zero=bool(riemann_zero),
                kretschmann=sp.simplify(kretschmann), t=t)


# --------------------------------------------------------------------------
#  Timelike geodesics (own integrator; EinsteinPy's solver covers only
#  Schwarzschild / Kerr)
# --------------------------------------------------------------------------

def diagonal_curvature(g_diag, coords) -> dict:
    """
    Riemann, Ricci and Kretschmann of the diagonal metric diag(g_diag) in `coords`, written out in sympy with no
    other package: Gamma^a_bc from the metric, R^a_bcd from Gamma, R_bd by contraction.  Slow for large
    expressions but exact; it is the second route that checks `einsteinpy_curvature`.
    Returns dict(ricci_is_zero, riemann_is_zero, kretschmann).
    """
    import sympy as sp
    n = len(coords)
    g_diag = [sp.sympify(x) for x in g_diag]      # a Python int would make 1 / x a float
    g = sp.diag(*g_diag)
    ginv = sp.diag(*[1 / x for x in g_diag])
    Gam = [[[sum(ginv[a, d] * (sp.diff(g[d, b], coords[c]) + sp.diff(g[d, c], coords[b])
                                - sp.diff(g[b, c], coords[d])) for d in range(n)) / 2
             for c in range(n)] for b in range(n)] for a in range(n)]
    def riem_up(a, b, c, d):
        r = sp.diff(Gam[a][b][d], coords[c]) - sp.diff(Gam[a][b][c], coords[d])
        r += sum(Gam[a][c][e] * Gam[e][b][d] - Gam[a][d][e] * Gam[e][b][c] for e in range(n))
        return sp.simplify(r)
    R = {(a, b, c, d): riem_up(a, b, c, d) for a in range(n) for b in range(n) for c in range(n) for d in range(n)}
    ricci = sp.Matrix(n, n, lambda b, d: sp.simplify(sum(R[(a, b, a, d)] for a in range(n))))
    kret = 0
    for (a, b, c, d), v in R.items():
        if v != 0:                       # R_abcd = g_aa R^a_bcd, and R^abcd = R_abcd / (g_aa g_bb g_cc g_dd)
            kret += (g[a, a] * v) ** 2 / (g[a, a] * g[b, b] * g[c, c] * g[d, d])
    return dict(ricci_is_zero=ricci == sp.zeros(n, n), riemann_is_zero=all(v == 0 for v in R.values()),
                kretschmann=sp.simplify(kret))


def sympy_curvature(p: Sequence) -> dict:
    """
    Curvature of the Kasner metric with exponents p from `diagonal_curvature`, no EinsteinPy.  Pass exact
    (sympy Rational) exponents to get exact zeros.  Keys as `einsteinpy_curvature`, plus t.
    """
    import sympy as sp
    t, x, y, z = sp.symbols("t x y z", positive=True)
    out = diagonal_curvature([-1, t ** (2 * p[0]), t ** (2 * p[1]), t ** (2 * p[2])], [t, x, y, z])
    out["t"] = t
    return out


def kasner_geodesic(p: Sequence[float], k: Sequence[float], t0: float, t_end: float,
                    n_out: int = 400, rtol: float = 1e-11, atol: float = 1e-13) -> dict:
    """
    Timelike geodesic (unit mass, c = 1) of the Kasner metric with exponents p.

    The metric does not depend on x, y, z, so the covariant momenta
    k_i = g_ii u^i = t^(2 p_i) dx^i/dtau are conserved.  The normalisation
    u.u = -1 then fixes

        gamma = u^t = dt/dtau = sqrt(1 + sum_i k_i^2 t^(-2 p_i))
        dx^i/dt     = k_i t^(-2 p_i) / gamma ,        dtau/dt = 1 / gamma

    which is integrated in t.  If t_end < t0 the worldline runs toward the
    singularity (the contracting, Big-Crunch branch); tau still increases along it.

    Returns a dict of arrays on an even grid of t between t0 and t_end:
        t, tau (proper time since t0), x (n, 3), gamma, speed
    where `speed` = sqrt(1 - 1/gamma^2) is the speed relative to the comoving
    observers (-> 1 as the singularity is approached along a contracting path).
    """
    p = np.asarray(p, dtype=float)
    k = np.asarray(k, dtype=float)
    if t0 <= 0.0 or t_end <= 0.0:
        raise ValueError("t0 and t_end must be positive")
    sgn = -1.0 if t_end < t0 else 1.0       # tau must increase even when t decreases

    def gamma_of(t):
        return np.sqrt(1.0 + np.sum(k ** 2 * t ** (-2.0 * p)))

    def rhs(t, y):
        g = gamma_of(t)
        return sgn * np.concatenate([[1.0 / g], k * t ** (-2.0 * p) / g])

    t_eval = np.linspace(t0, t_end, n_out)
    sol = solve_ivp(rhs, [t0, t_end], np.zeros(4), t_eval=t_eval,
                    method="DOP853", rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(f"Kasner geodesic integration failed: {sol.message}")
    gamma = np.array([gamma_of(t) for t in sol.t])
    return dict(t=sol.t, tau=sol.y[0], x=sol.y[1:].T, gamma=gamma,
                speed=np.sqrt(np.clip(1.0 - 1.0 / gamma ** 2, 0.0, 1.0)))
