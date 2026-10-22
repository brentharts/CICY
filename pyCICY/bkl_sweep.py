"""
pyCICY.bkl_sweep -- scan the Mixmaster starting parameter u and summarise the Kasner epochs and the bounces.

Why a per-u start time.  `bkl.initial_state` starts in a Kasner epoch and sets the first bounce near
`t_bounce`.  The curvature term of the negative-exponent axis at the start is (t_bounce / t0)^(4 |p_1|) / 2, and
p_1 = -u / (1 + u + u^2) -> 0 as u grows, so a fixed t0 gives a first epoch that is not clean for large u (for
u = 8.5 and t0 = 1e6 the term is already 3e-3 and the first epoch is missing).  `clean_t0` picks t0 so that term
is `tol`; with it every u in the scan starts in a clean epoch.  (`bkl.clean_t0`, which `Mixmaster` also
uses to warn about unclean starts.)

Analytic expectations used as checks (all tested):
  * Kasner epoch u has exponents (-u, 1+u, u(1+u)) / (1+u+u^2) and Weyl magnitude K = 16 u^2 (1+u)^2 /
    (1+u+u^2)^3 / t^4, which falls from 64/27 at u = 1 toward 0 as u -> infinity (the (0, 0, 1) flat limit);
  * after each bounce u -> u - 1 (u >= 2) or 1/(u - 1) (new era);
  * the measured exponents of every epoch match the Landau-Lifshitz map.
Bounces are the peaks of the curvature term above `epoch_threshold` (the same threshold that separates the
epochs), because the fixed 5% of `Mixmaster.bounce_times` misses the weak bounces at large u and a threshold
relative to the strongest bounce misses weaker early ones.  Every row therefore has n_epochs = n_bounces + 1,
which is checked.  Ported from Quantum Gravity Lab (CC0) without its sensor columns.
"""


from __future__ import annotations

from typing import Iterable, List

import numpy as np

from . import bkl as bk
from . import kasner as qk


def kasner_weyl_coefficient(u: float) -> float:
    """K t^4 for the Kasner epoch with parameter u: -16 p1 p2 p3 = 16 u^2 (1+u)^2 / (1+u+u^2)^3."""
    return 16.0 * u * u * (1.0 + u) ** 2 / (1.0 + u + u * u) ** 3


clean_t0 = bk.clean_t0     # re-exported: defined next to Mixmaster, which warns when a start is not clean


def _epoch_exponent_error(mm: bk.Mixmaster, u: float, epochs) -> float:
    p = np.array(qk.kasner_exponents(float(u)))
    err = 0.0
    for e in epochs:
        err = max(err, float(np.max(np.abs(np.sort(e["v"]) - np.sort(p)))))
        p = bk.bkl_transition(p)
    return err


def mixmaster_sweep(u_values: Iterable[float], t_bounce: float = 1.0, t_end: float = 1e-60,
                    epoch_threshold: float = 2e-3) -> List[dict]:
    """
    One row per starting u:
        u, p1, p2, p3, t0, K_coeff (analytic K t^4), K_t4_measured (mid first epoch),
        n_bounces, n_epochs, first_bounce_t, bounce_strength (max over the run of the curvature term t^2 R3),
        exponent_error (max over the epochs found vs the Landau-Lifshitz map).
    Returns a list of dicts, one per u.
    """
    rows = []
    for u in u_values:
        p = qk.kasner_exponents(float(u))
        t0 = clean_t0(u, t_bounce)
        mm = bk.Mixmaster(float(u), t0, t_bounce, t_end)
        epochs = mm.epochs(n_grid=60000, threshold=epoch_threshold)
        t_grid = np.geomspace(mm.t0, mm.t_end, 6000)
        strength = float(np.max(mm.potential_fraction(t_grid)))
        bounces = mm.bounce_times(threshold=epoch_threshold)
        row = dict(u=float(u), p1=p[0], p2=p[1], p3=p[2], t0=t0, K_coeff=kasner_weyl_coefficient(float(u)),
                   n_bounces=len(bounces), n_epochs=len(epochs),
                   first_bounce_t=float(bounces[0]) if len(bounces) else np.nan,
                   bounce_strength=strength,
                   exponent_error=_epoch_exponent_error(mm, u, epochs))
        if len(epochs):
            e0 = epochs[0]
            t_mid = float(np.sqrt(e0["t_start"] * e0["t_start"] * np.exp(-e0["dln_t"])))
            row["K_t4_measured"] = float(mm.weyl_magnitude(t_mid)[0] * t_mid ** 4)
        else:
            row["K_t4_measured"] = np.nan
        rows.append(row)
    return rows
