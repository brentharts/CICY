"""
pyCICY.tidal -- tidal tensors for the cosmological spacetimes, and what they say about "pancake" singularities.

For comoving observers in a diagonal Bianchi universe  ds^2 = -dt^2 + sum a_i^2 sigma_i^2  the geodesic-deviation
equation in the orthonormal frame is  xi_i'' = (a_i'' / a_i) xi_i .  The tidal eigenvalues are therefore

    lambda_i = a_i'' / a_i        (> 0 stretches along axis i, < 0 compresses)

with trace  sum_i lambda_i = -R_00  (zero in vacuum).  Examples:

    Kasner          lambda_i = p_i (p_i - 1) / t^2     -> diverges as t -> 0 (every vacuum Kasner is a cigar there)
    Schwarzschild   (+2, -1, -1) GM / (c^2 r^3)        -> radial stretch, transverse squeeze
    pancake         finite at t = 0 (pancake)      -> no diverging tides, although the volume vanishes

The tidal strength is T = sqrt(sum lambda_i^2).  For a static vacuum field K = 8 T^2 (Schwarzschild:
8 * 6 (GM)^2/r^6 = 48 (GM)^2/r^6); in general K = 8 (E.E - B.B), so K = 8 T^2 - 8 B.B with B the magnetic Weyl
part, which is nonzero in Mixmaster.  Both statements are tested.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from . import pancake as qp


def tidal_from_scale_factors(a, add) -> np.ndarray:
    """lambda_i = a_i'' / a_i for scale factors a and second derivatives add (shape (..., 3))."""
    return np.asarray(add, dtype=float) / np.asarray(a, dtype=float)


def kasner_tidal(p: Sequence[float], t) -> np.ndarray:
    """Comoving tidal eigenvalues of Kasner, p_i (p_i - 1) / t^2, shape (len(t), 3) for array t."""
    p = np.asarray(p, dtype=float)
    t = np.atleast_1d(np.asarray(t, dtype=float))
    return (p * (p - 1.0))[None, :] / t[:, None] ** 2


def schwarzschild_tidal(r, G: float = 1.0, M: float = 1.0, c: float = 1.0) -> np.ndarray:
    """
    (+2, -1, -1) GM / (c^2 r^3) for a freely falling observer: (radial, transverse, transverse), shape (n, 3).
    These are eigenvalues of R_0i0i in 1/length^2 (so K = 8 sum lambda^2 holds); the Newtonian tidal
    acceleration per unit separation is c^2 times this.
    """
    r = np.atleast_1d(np.asarray(r, dtype=float))
    base = G * M / (c ** 2 * r ** 3)
    return np.stack([2.0 * base, -base, -base], axis=-1)


def pancake_tidal(sol) -> np.ndarray:
    """Comoving tidal eigenvalues R_i'' / R_i along a `pancake.pancake_solution`, shape (n, 3)."""
    return tidal_from_scale_factors(sol["R"], sol["Rdd"])


def pancake_tidal_at_zero(b0: float, f0: float, m: float, Lam: float = 0.0, kappa: float = 1.0) -> np.ndarray:
    """
    Tidal eigenvalues at the pancake itself: (4/b0^2, e, e) with
    e = R_2''/R_2 = 2 b2 / b0 = Lam/2 + kappa m^2 f0^2/4 - 2/b0^2.
    Finite for every choice of the free data.
    """
    c = qp.pancake_series(1.0, b0, f0, m, Lam, kappa)
    e = 2.0 * c["b2"] / b0
    return np.array([4.0 / b0 ** 2, e, e])


def tidal_strength(lam) -> np.ndarray:
    """T = sqrt(sum_i lambda_i^2); a frame-independent size of the tidal tensor."""
    return np.sqrt(np.sum(np.asarray(lam, dtype=float) ** 2, axis=-1))
