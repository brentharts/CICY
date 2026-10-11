r"""
pyCICY.entropy -- D4-D0 black-hole entropy on a CICY threefold, two routes.

Why this module exists
----------------------
The semiclassical derivation of Einstein's equations from relative entropy
(Dorau-Much; see :mod:`pyCICY.theories.nariai`) assumes S = A/4, i.e. the
coupling alpha = 8 pi. In a Calabi-Yau compactification that normalisation is
not exact: higher-derivative terms, controlled by c_2(TX), correct the area
law. :mod:`pyCICY.theories.nariai` checks the Einstein-gravity case exactly;
this module computes the size of the correction for a given CICY from the
same data :mod:`pyCICY.centralcharge` already uses.

The black holes are the 4D BPS black holes of M-theory / type IIA on X with
D4 charge P = p^r J_r (a divisor class) and D0 charge q_0, no D2 or D6 charge
(the "D4-D0" slice). Two routes to the entropy:

* **Microscopic (Cardy).** Maldacena-Strominger-Witten: the M5-brane on P
  gives a 2D CFT with
      c_L = 6 D + c_2 . P ,     c_R = 6 D + c_2 . P / 2 ,
      D = (1/6) d_rst p^r p^s p^t ,
  and S = 2 pi sqrt(c_L q_0 / 6) at large q_0. Exact in (S/pi)^2.
* **Macroscopic (attractor).** At the attractor point the entropy is
  S = A/4 = pi |Z|^2 with Z the central charge normalised by the Kaehler
  potential, e^{-K} = (4/3) d(w, w, w). Here |Z|^2 e^K is minimised over the
  Kaehler class w with :func:`pyCICY.centralcharge.z_ordinary` (and with
  :func:`~pyCICY.centralcharge.z_sqrt_td` for the Todd-corrected charge) on
  the class with ch = (0, P, 0, -q_0). This route knows nothing of the CFT.

At leading order they agree, (S/pi)^2 = 4 D q_0, which is the check that
S = A/4 and the normalisation of Z are consistent. The Cardy value exceeds
the leading one by exactly the factor 1 + c_2.P / (6 D), the first
higher-derivative (Wald-type) correction; the Todd-corrected attractor
shifts q_0 by c_2.P/24 instead. These two corrections are *different*, and
this module reports both without identifying them.

Ledger
------
Exact (rational arithmetic): D, c_L, c_R, c_2.P, the correction fraction
delta = c_2.P/(6D), and (S/pi)^2 for the Cardy and leading formulas.
Numerical, by minimisation: the attractor values (tolerance stated in the
tests). Not computed, and not claimed:

* the microstate counting itself -- the MSW CFT is a derivation with
  assumptions (P very ample, the Cardy regime q_0 >> c_L), not a theorem
  here; ``regime`` reports q_0/c_L but nothing checks very-ampleness;
* D2 and D6 charges, and any formula beyond the D4-D0 slice;
* that the Todd-corrected attractor equals the Cardy value: it does not,
  and the difference is reported;
* sub-leading corrections beyond the first (non-perturbative terms, the
  q_0 shift conventions of MSW versus supergravity).
"""

import math
from fractions import Fraction

from . import centralcharge as CC

__all__ = ["d4d0", "attractor", "NotCardyRegime"]


class NotCardyRegime(ValueError):
    """Raised when a requested charge has no real entropy (D <= 0 or q0 <= 0)."""


def d4d0(X, P, q0):
    r"""Exact D4-D0 data and the Cardy and leading-order (S/pi)^2.

    ``X`` is a :class:`pyCICY.centralcharge.CY3`, ``P`` an integer vector,
    ``q0`` a positive rational. Returns a dict of Fractions (``s2`` entries
    are (S/pi)^2) plus the float entropies. ``regime`` is q0 / c_L: the Cardy
    formula is a large-charge statement, trustworthy only for regime >> 1.
    """
    P = CC._vec(P, X.h)
    q0 = CC._frac(q0)
    CC._check_ample(X, P)               # nonneg, not all zero, w^3 > 0
    D = X.cubic(P, P, P) / 6
    c2P = X.dot(X.c2, P)
    if D <= 0 or q0 <= 0:
        raise NotCardyRegime("need D = d p^3 / 6 > 0 and q0 > 0; got D=%s, "
                             "q0=%s" % (D, q0))
    cL = 6 * D + c2P
    cR = 6 * D + c2P / 2
    s2_lead = 4 * D * q0
    s2_cardy = 4 * (cL / 6) * q0
    return {"D": D, "c2P": c2P, "cL": cL, "cR": cR,
            "delta": c2P / (6 * D),
            "s2_leading": s2_lead, "s2_cardy": s2_cardy,
            "ratio_cardy_over_leading": s2_cardy / s2_lead,
            "s2_todd_shift": 4 * D * (q0 - c2P / 24),
            "S_leading": math.pi * math.sqrt(float(s2_lead)),
            "S_cardy": math.pi * math.sqrt(float(s2_cardy)),
            "regime": q0 / cL}


def attractor(X, P, q0, sqrt_td=False, starts=6, seed=0):
    r"""Minimise |Z|^2 e^K over the Kaehler class: S/pi at the attractor.

    The class is ch = (0, P, 0, -q0) and the charge is Z_ord (or Z_sqrt with
    ``sqrt_td``), evaluated at B = 0. S/pi = min |Z|^2 / ((4/3) w^3), found
    numerically by Nelder-Mead in log w from several starts. Returns a dict
    with ``S_over_pi`` (the minimum), ``s2`` (its square, (S/pi)^2),
    ``omega`` (the minimiser) and ``spread`` (max - min of S/pi over the
    starts, as a convergence check).
    """
    import random
    from scipy.optimize import minimize

    P = CC._vec(P, X.h)
    q0 = CC._frac(q0)
    E = CC.Object(X, 0, P, [0] * X.h, -q0)
    zf = CC.z_sqrt_td if sqrt_td else CC.z_ordinary
    B = [0] * X.h

    def f(u):
        w = [Fraction(math.exp(x)) for x in u]
        z = complex(zf(E, B, w))
        vol = float(X.cubic(w, w, w))
        if vol <= 0:
            return 1e300
        return abs(z) ** 2 / (4.0 / 3.0 * vol)

    rng = random.Random(seed)
    vals, best = [], None
    for k in range(starts):
        u0 = [rng.uniform(-1.0, 2.0) for _ in range(X.h)]
        r = minimize(f, u0, method="Nelder-Mead",
                     options={"xatol": 1e-10, "fatol": 1e-13, "maxiter": 4000})
        vals.append(r.fun)
        if best is None or r.fun < best.fun:
            best = r
    return {"S_over_pi": min(vals), "s2": min(vals) ** 2, "omega": [math.exp(x) for x in best.x],
            "spread": max(vals) - min(vals)}
