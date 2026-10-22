"""
pyCICY.flrw -- a TOY flat-FLRW expansion history with a sigmoid-blended "entropy" dark term, compared with
LCDM and with two models from the literature, plus an exact bounce toy.  Ported from Quantum Gravity Lab (CC0).

THIS IS OUR OWN TOY, not a reproduction of anyone's model.  It is inspired by the Fairfolio project
"LCDM+S" (L. Desre), which describes a sigmoid blend of two Friedmann solutions but publishes no equations or
code, so nothing here can be said to match it.  The pieces are chosen for being checkable:

  toy ("LCDM+S toy")    E^2(a) = Om a^-3 + Or a^-4 + (1 - Om - Or) g(a),   g(a) = s(a) / s(1)
                         s(a)  = 1 / (1 + (a_t / a)^k)                (sigmoid in ln a, width 1/k)
      The dark term switches on around a = a_t.  g -> 1 as a_t -> 0, which recovers flat LCDM exactly; E(1) = 1.
      Because g grows with a the dark term is phantom-like (w < -1) while the sigmoid is rising.

  Tsallis horizon       E = [Om a^-3 + Or a^-4 + OL]^(1 / (4 - 2 beta))        beta = 1 is LCDM
      (modified Friedmann equation from S = gamma A^beta; Shahhoseini, Malekjani, Khodam-Mohammadi, arXiv:2501.03655)

  Pade deceleration     E^2 = Om (1+z)^3 + (1 - Om) (1 + z + q0 z)^2           q0 = -1 is LCDM
      (Capozziello, D'Agostino, Luongo, arXiv:2202.03300)

Horizon thermodynamics.  For flat FLRW the apparent horizon is r = 1/H, so S = pi / (G H^2) (hbar = c = k_B = 1)
and dS/dt = -2 pi H' / (G H^3).  The second law dS/dt >= 0 is therefore H' <= 0, i.e. deceleration q >= -1, i.e.
the total fluid does not go phantom.  `second_law_holds` checks that on a history; the toy can violate it.

Units: H0 = 1 (times in 1/H0, distances in c/H0) unless h0 is given.  Flat, spatial curvature zero.
Kretschmann of flat FLRW: K = 12 [ (a''/a)^2 + H^4 ].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.integrate import quad


# --------------------------------------------------------------------------
#  Models (all return E^2 = H^2 / H0^2 as a function of the scale factor a)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ToyParams:
    Om: float = 0.3
    Or: float = 0.0
    a_t: float = 0.0          # sigmoid centre; 0 gives LCDM
    k: float = 4.0            # sigmoid steepness in ln a


def _s(a, a_t: float, k: float):
    a = np.asarray(a, dtype=float)
    if a_t <= 0.0:
        return np.ones_like(a)
    return 1.0 / (1.0 + (a_t / a) ** k)


def lcdm_E2(a, Om: float = 0.3, Or: float = 0.0):
    a = np.asarray(a, dtype=float)
    return Om * a ** -3 + Or * a ** -4 + (1.0 - Om - Or)


def toy_E2(a, p: ToyParams = ToyParams()):
    a = np.asarray(a, dtype=float)
    g = _s(a, p.a_t, p.k) / _s(1.0, p.a_t, p.k)
    return p.Om * a ** -3 + p.Or * a ** -4 + (1.0 - p.Om - p.Or) * g


def toy_dlnE2_dlna(a, p: ToyParams = ToyParams()):
    """d ln E^2 / d ln a, analytic (ds/dln a = k s (1 - s))."""
    a = np.asarray(a, dtype=float)
    s = _s(a, p.a_t, p.k)
    dg = 0.0 if p.a_t <= 0.0 else p.k * (1.0 - s) * s / _s(1.0, p.a_t, p.k)
    dE2 = -3.0 * p.Om * a ** -3 - 4.0 * p.Or * a ** -4 + (1.0 - p.Om - p.Or) * dg
    return dE2 / toy_E2(a, p)


def tsallis_E2(a, beta: float, Om: float = 0.3, Or: float = 0.0):
    a = np.asarray(a, dtype=float)
    return (Om * a ** -3 + Or * a ** -4 + (1.0 - Om - Or)) ** (1.0 / (2.0 - beta))


def pade_E2(a, q0: float, Om: float = 0.3):
    z = 1.0 / np.asarray(a, dtype=float) - 1.0
    return Om * (1.0 + z) ** 3 + (1.0 - Om) * (1.0 + z + q0 * z) ** 2


# --------------------------------------------------------------------------
#  Kinematics from any E^2(a)
# --------------------------------------------------------------------------

def dlnE2_dlna_numeric(E2, a, h: float = 1e-5):
    """Central difference in ln a of ln E^2, for models without an analytic derivative."""
    a = np.asarray(a, dtype=float)
    return (np.log(E2(a * np.exp(h))) - np.log(E2(a * np.exp(-h)))) / (2.0 * h)


def deceleration(E2, a, dlnE2=None):
    """q = -1 - H'/H^2 = -1 - (1/2) d ln E^2 / d ln a  (so q >= -1 is H' <= 0)."""
    d = dlnE2_dlna_numeric(E2, a) if dlnE2 is None else dlnE2
    return -1.0 - 0.5 * d


def cosmic_time(E2, a, h0: float = 1.0):
    """t(a) = int_0^a da' / (a' H(a')), in units of 1/H0 when h0 = 1 (starts at the big bang, a = 0)."""
    f = lambda x: 1.0 / (x * h0 * np.sqrt(E2(x)))
    a = np.atleast_1d(np.asarray(a, dtype=float))
    return np.array([quad(f, 0.0, x, epsabs=1e-12, epsrel=1e-11, limit=200)[0] for x in a])


def luminosity_distance(E2, z, h0: float = 1.0):
    """d_L = (1 + z) int_0^z dz' / (h0 E(z')) in c/H0 units (flat)."""
    z = np.atleast_1d(np.asarray(z, dtype=float))
    f = lambda x: 1.0 / (h0 * np.sqrt(E2(1.0 / (1.0 + x))))
    return (1.0 + z) * np.array([quad(f, 0.0, x, epsabs=1e-13, epsrel=1e-12)[0] for x in z])


def distance_modulus(E2, z, h0: float = 1.0):
    """mu = 5 log10(d_L) + const; the constant cancels in differences between models at fixed h0."""
    return 5.0 * np.log10(luminosity_distance(E2, z, h0))


def horizon_entropy(E2, a, h0: float = 1.0, G: float = 1.0):
    """S = pi / (G H^2) for the flat apparent horizon r = 1/H (hbar = c = k_B = 1)."""
    return np.pi / (G * h0 ** 2 * np.asarray(E2(a), dtype=float))


def entropy_rate_sign(E2, a, dlnE2=None):
    """dS/dt is proportional to -H'/H^3, with H' = (H^2 / 2) d ln E^2/d ln a: positive iff q > -1."""
    d = dlnE2_dlna_numeric(E2, a) if dlnE2 is None else dlnE2
    return -0.5 * d / np.sqrt(np.asarray(E2(a), dtype=float))


def second_law_holds(E2, a, dlnE2=None, tol: float = 1e-12) -> bool:
    return bool(np.all(entropy_rate_sign(E2, a, dlnE2) >= -tol))


def flrw_kretschmann(H, Hdot):
    """K = 12 [ (a''/a)^2 + H^4 ] with a''/a = H' + H^2 (flat FLRW; de Sitter gives 24 H^4)."""
    H, Hdot = np.asarray(H, dtype=float), np.asarray(Hdot, dtype=float)
    return 12.0 * ((Hdot + H ** 2) ** 2 + H ** 4)


# --------------------------------------------------------------------------
#  History table
# --------------------------------------------------------------------------

def flrw_history(E2, a, h0: float = 1.0, dlnE2=None) -> dict:
    """Arrays a, z, t, H, q, dS_sign, K along a grid of scale factors (ascending), as a dict."""
    a = np.asarray(a, dtype=float)
    d = dlnE2_dlna_numeric(E2, a) if dlnE2 is None else dlnE2
    H = h0 * np.sqrt(E2(a))
    Hdot = 0.5 * H ** 2 * d
    return dict(a=a, z=1.0 / a - 1.0, t=cosmic_time(E2, a, h0), H=H, q=-1.0 - 0.5 * d,
                dS_sign=-0.5 * d / np.sqrt(E2(a)), K=flrw_kretschmann(H, Hdot))


# --------------------------------------------------------------------------
#  A non-singular bounce toy (dust with a maximum density)
# --------------------------------------------------------------------------
#
# E^2(a) = x (1 - x / rho_c),  x = Om a^-3 : the effective-Friedmann form used in loop quantum cosmology, where the
# density saturates at rho_c and the universe bounces at a_b = (Om / rho_c)^(1/3).  This is a generic non-singular
# bounce, NOT the "quantum pressure" mechanism of Gaztanaga (Phys. Rev. D 113, 2026, ScienceDaily summary), whose
# equations I have not read.  It is here as a finite-curvature crossing like the pancake's.
# With u = a^3, u' = 3 sqrt(Om) sqrt(u - u_b), so a^3 = a_b^3 + (9/4) Om t^2 with t = 0 at the bounce (exact).

def bounce_E2(a, Om: float = 0.3, rho_c: float = 10.0):
    x = Om * np.asarray(a, dtype=float) ** -3
    return x * (1.0 - x / rho_c)


def bounce_scale_factor(t, Om: float = 0.3, rho_c: float = 10.0, h0: float = 1.0):
    """Exact a(t) through the bounce (t = 0 at a = a_b): a^3 = a_b^3 + (9/4) Om h0^2 t^2."""
    a_b3 = Om / rho_c
    return (a_b3 + 2.25 * Om * (h0 * np.asarray(t, dtype=float)) ** 2) ** (1.0 / 3.0)


def bounce_hubble(t, Om: float = 0.3, rho_c: float = 10.0, h0: float = 1.0):
    """H(t) = u' / (3 u) with u = a^3 = a_b^3 + (9/4) Om h0^2 t^2, i.e. 1.5 Om h0^2 t / a^3 (odd in t)."""
    a = bounce_scale_factor(t, Om, rho_c, h0)
    return 1.5 * Om * h0 ** 2 * np.asarray(t, dtype=float) / a ** 3


def bounce_hubble_dot(t, Om: float = 0.3, rho_c: float = 10.0, h0: float = 1.0):
    """H' = 1.5 Om h0^2 (1 / a^3 - 4.5 Om h0^2 t^2 / a^6) ; equals 1.5 rho_c h0^2 at the bounce."""
    a3 = bounce_scale_factor(t, Om, rho_c, h0) ** 3
    t = np.asarray(t, dtype=float)
    return 1.5 * Om * h0 ** 2 * (1.0 / a3 - 3.0 * 1.5 * Om * h0 ** 2 * t ** 2 / a3 ** 2)
