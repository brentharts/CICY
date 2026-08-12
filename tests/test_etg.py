#!/usr/bin/env python3
"""Verification suite for pyCICY.theories.etg_foreground.

Sections
  1. reproduction of Gjergo-Kroupa (2025): timescales, energy densities,
     dust-temperature consistency, and the Eq. (5) arithmetic slip
  2. the spectral wall: FIRAS limits on the graybody variant
  3. the anisotropy wall: the Poisson shot-noise floor
  4. (RUN_DATA=1) stability of the layer-two bound under a shot nuisance
"""

import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyCICY.theories import etg_foreground as E

FAILURES = []


def check(label, got, want):
    if got != want:
        FAILURES.append("%s: got %r want %r" % (label, got, want))


def check_true(label, cond):
    if not cond:
        FAILURES.append(label)


def check_close(label, got, want, rel=0.02):
    if abs(got - want) > rel * abs(want):
        FAILURES.append("%s: %r vs %r" % (label, got, want))


def main():
    t0 = time.time()

    # ------------------------------------------------------------------
    # 1. reproduction
    # ------------------------------------------------------------------
    r = E.gk_reproduction()
    check_close("tau_down at 1e11.5 Msun reproduces 440 Myr",
                r["tau_down_11p5_Myr"][0], 440, 0.02)
    check_close("tau_down at 1e12 Msun reproduces 340 Myr",
                r["tau_down_12_Myr"][0], 340, 0.02)
    check_close("U_CMB,0 reproduces 4.17e-14 J/m^3",
                r["U_CMB0_Jm3"][0], 4.17e-14, 0.01)
    check_close("U at recombination reproduces 0.06 J/m^3",
                r["U_CMB_EoR_Jm3"][0], 0.06, 0.02)
    check_close("the conservative fraction is 1.4%",
                r["conservative_fraction"][0], 0.0141, 0.01)
    check_close("dust temperature closes at ~49 K",
                r["T_dust_em_K"][0], 48.7, 0.01)
    # the slip: Eq (5) as printed evaluates to 17.75, not 16.5
    check_close("Eq (5) computed value", r["z_f_eq5_computed"], 17.75,
                0.001)
    check_true("the slip is real (>1 apart from the printed value)",
               r["eq5_slip"])
    check_true("...and the 21-cm-window conclusion is unaffected",
               not r["conclusion_affected"])

    # ------------------------------------------------------------------
    # 2. FIRAS
    # ------------------------------------------------------------------
    f = E.firas_limit()
    fx = f["f_max_by_beta_d"]
    check_true("all graybody variants bounded below the claimed 1.4%",
               all(v < 0.014 for v in fx.values()))
    check_true("physical dust (beta_d = 1.5-2) bounded below 5e-4",
               fx[1.5] < 5e-4 and fx[2.0] < 5e-4)
    check_true("the weakest case (beta_d ~ 1, near the dT degeneracy) "
               "is still 5x below the claim", fx[1.0] < 0.014 / 5)

    # ------------------------------------------------------------------
    # 3. shot noise
    # ------------------------------------------------------------------
    s = E.shot_noise_limit()
    check_close("~6 sources per Planck beam (GK's own count)",
                s["sources_per_planck_beam"], 5.8, 0.05)
    check_true("claimed 1.4% overpredicts ell~3000 power by > 1e5",
               s["overprediction_factor"] > 1e5)
    check_true("blackbody variant bounded at f < 5e-5",
               s["f_max"] < 5e-5)
    check_true("claimed / allowed is a factor of several hundred",
               300 < s["claimed_over_allowed"] < 1000)
    # the floor scales as it must: doubling N halves C_shot
    s2 = E.shot_noise_limit(n_sources=8.8e7)
    check_close("Poisson scaling f_max ~ sqrt(N)",
                s2["f_max"] / s["f_max"], math.sqrt(2.0), 0.01)

    # verdict logic (no data needed)
    v = E.impact_on_verdicts()
    check_true("sigma ratings unchanged", not v["sigma_ratings_changed"])

    # ------------------------------------------------------------------
    # 4. layer-two stability (gated)
    # ------------------------------------------------------------------
    if os.environ.get("RUN_DATA") == "1":
        lw = E.layer_two_with_shot()
        check_true("layer-two bound stable under shot nuisance "
                   "(UL within 20% of 0.0071)",
                   lw["upper95_amplitude"] < 0.0071 * 1.2)
    else:
        print("  (shot-nuisance GLS skipped; set RUN_DATA=1 to run)")

    dt = time.time() - t0
    if FAILURES:
        print("test_etg: %d FAILURES in %.1fs" % (len(FAILURES), dt))
        for msg in FAILURES:
            print("  FAIL:", msg)
        sys.exit(1)
    print("test_etg: all checks passed in %.1fs" % dt)


if __name__ == "__main__":
    main()
