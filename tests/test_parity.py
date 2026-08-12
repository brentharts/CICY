#!/usr/bin/env python3
"""Verification suite for pyCICY.theories.parity.

Sections
  1. the exact rotation algebra (identities at machine precision)
  2. the substrate frequency (parameter-free, tied to the spectre module)
  3. the measurement table and the verdict logic
  4. categories, refusals, registry
  5. (optional, RUN_DATA=1) the layer-two search on real band powers

The data search needs candl, candl_data and camb and runs CAMB four
times; it is gated behind the environment variable so the default suite
stays fast, and the gated section re-verifies the null result quoted in
the paper: joint |A| at omega* consistent with zero.
"""

import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from pyCICY.theories import parity as P
from pyCICY.theories import spectre as S
from pyCICY.theories import get as get_theory
from pyCICY.theories.ftheory import NoSuchTheory

FAILURES = []


def check(label, got, want):
    if got != want:
        FAILURES.append("%s: got %r want %r" % (label, got, want))


def check_true(label, cond):
    if not cond:
        FAILURES.append(label)


def check_close(label, got, want, tol=1e-9):
    if abs(got - want) > tol:
        FAILURES.append("%s: %r vs %r" % (label, got, want))


def main():
    t0 = time.time()

    # ------------------------------------------------------------------
    # 1. rotation algebra
    # ------------------------------------------------------------------
    inv = P.rotation_invariants(beta_deg=0.7, n=128, seed=3)
    check_true("EE+BB rotation-invariant", inv["sum_invariant"] < 1e-12)
    check_true("TE^2+TB^2 rotation-invariant",
               inv["te_tb_invariant"] < 1e-12)
    check_true("rotate then unrotate is the identity",
               inv["round_trip"] < 1e-12)
    check_true("EB winds at sin(4 beta)/2 exactly",
               inv["eb_slope_check"] < 1e-14)
    # small-angle content: at beta = 0.277 deg the EB/EE ratio is ~1%
    b = 0.277
    check_close("sin(4 beta)/2 at the measured angle",
                0.5 * math.sin(4 * math.radians(b)), 0.00967, 1e-4)
    # a rotation cannot create parity-odd power from nothing at beta=0
    out = P.rotation_mixing([1.0], [0.1], [0.5], 0.0)
    check("beta = 0 is the identity",
          (out["EB"][0], out["TB"][0]), (0.0, 0.0))

    # ------------------------------------------------------------------
    # 2. the substrate frequency
    # ------------------------------------------------------------------
    check_close("log-period is log(4+sqrt15)", P.LOG_PERIOD,
                math.log(4 + math.sqrt(15)), 1e-14)
    check_close("omega* = 2 pi / log-period", P.OMEGA_STAR,
                2 * math.pi / P.LOG_PERIOD, 1e-14)
    # tied to the spectre module's exact unit, not typed independently
    lam2 = S.inflation_data()["lambda_squared"]
    check_close("frequency inherits the fundamental unit exactly",
                P.LOG_PERIOD, math.log(float(lam2)), 1e-14)
    check_close("one modulation cycle spans a factor lambda^2 in k",
                math.exp(2 * math.pi / P.OMEGA_STAR), float(lam2), 1e-10)

    # ------------------------------------------------------------------
    # 3. measurements and the verdict
    # ------------------------------------------------------------------
    meas = P.birefringence_measurements()
    check_true("at least six published entries", len(meas) >= 6)
    heads = [m for m in meas if m.get("headline")]
    check("exactly one headline measurement", len(heads), 1)
    check("headline is the 2026 joint analysis",
          heads[0]["arxiv"], "2608.06480")
    check_close("headline beta", heads[0]["beta_deg"], 0.277, 1e-9)
    st = P.birefringence_status()
    check_true("face-value significance ~4.8 sigma",
               4.5 < st["significance_face_value"] < 5.1)
    check_true("dust-robust significance quoted separately",
               st["significance_dust_robust"] == 3.5)
    check_true("verdict states the consistent branch is active",
               "consistent branch is active" in st["verdict"])
    check_true("caveats include the propagation-vs-primordial gap",
               any("Chern-Simons" in c or "propagation" in c
                   for c in st["caveats"]))
    ten = P.tensor_chirality_status()
    check_close("r bound", ten["r_upper_95"], 0.036, 1e-12)

    # ------------------------------------------------------------------
    # 3b. the substrate prediction, exactly
    # ------------------------------------------------------------------
    pr = P.substrate_beta_prediction()
    g = S.Quad(4, -1, 15)
    pred_quad = (S.Quad(20, 0, 15) * g) / (S.Quad(8, 0, 15) + 10 * g)
    from fractions import Fraction as Fr
    check("beta_pred rationalization 70/67 - (40/201) sqrt15",
          (pred_quad.a, pred_quad.b), (Fr(70, 67), Fr(-40, 201)))
    check("module quotes the exact pair",
          tuple(pr["beta_pred_exact"]), (Fr(70, 67), Fr(-40, 201)))
    check_close("beta_pred decimal", pr["beta_pred_deg"], 0.2740332, 1e-6)
    check_close("beta_pred = 2 x order parameter",
                pr["beta_pred_deg"],
                2 * S.order_parameter()["max_splitting_float"], 1e-12)
    check_true("pull against 2026 joint below 0.1 sigma",
               abs(pr["pull_sigma"]) < 0.1)
    check_true("prediction is labeled speculation-grade",
               "speculation" in pr["grade"])
    ta = P.transfer_assumptions()
    check_close("one period is a factor lambda^2",
                ta["first_harmonic_period_factor"],
                float(S.inflation_data()["lambda_squared"]), 1e-10)
    check_true("amplitude and phase declared free",
               "amplitude" in ta["free"] and "phase" in ta["free"])
    fc = P.calibration_forecast()
    check_true("SO-era 0.05 deg tests the prediction at >5 sigma",
               next(r for r in fc["rows"]
                    if r["sigma_cal_deg"] == 0.05)
               ["detect_beta_sigma"] > 5.0)
    check_true("LiteBIRD-class separates 0.274 from 0.342 at >5 sigma",
               next(r for r in fc["rows"]
                    if r["sigma_cal_deg"] == 0.01)
               ["discriminate_pred_vs_0p342"] > 5.0)

    # ------------------------------------------------------------------
    # 4. categories, refusals, registry
    # ------------------------------------------------------------------
    th = get_theory("crossover-parity")()
    check("registry key", th.key, "crossover-parity")
    spec = th.spectrum()
    check_close("theory object carries omega*", spec["omega_star"],
                P.OMEGA_STAR, 1e-14)
    try:
        th.tensor_chirality()
        check_true("tensor chirality refuses", False)
    except NoSuchTheory:
        pass
    except Exception:
        check_true("tensor chirality raises the right exception", False)
    check_true("open problems listed", len(th.missing_for_physical()) >= 3)
    check_true("describe() renders the parity verdict",
               "0.277" in th.describe())

    # ------------------------------------------------------------------
    # 5. optional: the search on real data
    # ------------------------------------------------------------------
    if os.environ.get("RUN_DATA") == "1":
        r = P.layer_two_search("SPT3G_2018_TTTEEE_lite")
        check("SPT-3G bin count", r["n_bins"], 123)
        check_true("SPT-3G amplitude consistent with zero (dchi2 < 9)",
                   r["delta_chi2_2dof"] < 9.0)
        check_true("upper limit is finite and sub-10%",
                   0 < r["upper95_amplitude"] < 0.10)
        if os.path.isdir(P._PLANCK_LITE_DIR):
            rp = P.planck_layer_two()
            check("Planck bin count 613", rp["n_bins"], 613)
            check_true("Planck null at omega* (dchi2 < 9)",
                       rp["delta_chi2_2dof"] < 9.0)
            check_true("Planck UL95 below 1%",
                       rp["upper95_amplitude"] < 0.01)
        else:
            print("  (planck_lite data absent; run "
                  "scripts/get_external_data.sh)")
    else:
        print("  (layer-two data search skipped; set RUN_DATA=1 to run)")

    dt = time.time() - t0
    if FAILURES:
        print("test_parity: %d FAILURES in %.1fs" % (len(FAILURES), dt))
        for msg in FAILURES:
            print("  FAIL:", msg)
        sys.exit(1)
    print("test_parity: all checks passed in %.1fs" % dt)


if __name__ == "__main__":
    main()
