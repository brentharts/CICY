#!/usr/bin/env python3
"""Generate figures/crossover_facts.tex for
paper/supplementary_material_crossover.tex.

Runs the full data confrontation of pyCICY.theories.parity -- the
birefringence verdict, the substrate beta candidate, the layer-two GLS
searches on SPT-3G 2018, ACT DR6 and Planck plik-lite band powers, the
joint combinations, the DSI second harmonic, and the calibration-era
forecast -- and writes every number quoted in the paper. Requires camb,
candl, candl_data, and external/planck_lite_data
(scripts/get_external_data.sh). Runtime ~3 minutes.
"""

import json
import math
import os
import sys

import candl
candl_version = [int(a) for a in candl.__version__.split('.')]
if candl_version[0] <= 2 and candl_version[1] <= 2 and candl_version[2] < 1:
    print('monkey patching jax for older candl')
    print(candl, candl.__version__)
    ## allows older version of candl and newer version of jax to be compatible
    import jax.numpy as jnp
    # Convert lists to JAX arrays before jnp.atleast_1d checks types
    _orig_atleast_1d = jnp.atleast_1d
    jnp.atleast_1d = lambda *arys: _orig_atleast_1d(
        *[jnp.asarray(a) if isinstance(a, (list, tuple)) else a for a in arys]
    )

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyCICY.theories import parity as P

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)


def main():
    # resumable: reload any previously computed facts so an interrupted
    # run continues where it left off (delete the json to force fresh)
    facts = {}
    ck = os.path.join(OUT, "crossover_facts.json")
    if os.path.exists(ck):
        with open(ck) as f:
            facts = {k: v for k, v in json.load(f).items()}
        facts = {k: (float(v) if isinstance(v, str) and
                     v.replace(".", "").replace("-", "").replace("e",
                     "").replace("+", "").isdigit() else v)
                 for k, v in facts.items()}

    def save():
        with open(ck, "w") as f:
            json.dump(facts, f, indent=1, default=str)

    # parity channel
    st = P.birefringence_status()
    head = st["headline"]
    facts["BetaMeasured"] = head["beta_deg"]
    facts["BetaSigma"] = head["sigma"]
    facts["BetaZFace"] = st["significance_face_value"]
    facts["BetaZDust"] = st["significance_dust_robust"]

    # the candidate
    pr = P.substrate_beta_prediction()
    facts["BetaPred"] = pr["beta_pred_deg"]
    facts["BetaPull"] = abs(pr["pull_sigma"])

    # layer two, per dataset
    if "ULPlanck" in facts:
        spt = act = pl = None
    else:
        spt = P.layer_two_search("SPT3G_2018_TTTEEE_lite")
        act = P.layer_two_search("ACT_DR6_TTTEEE")
        pl = P.planck_layer_two()
        for tag, r in (("SPT", spt), ("ACT", act), ("Planck", pl)):
            facts["Amp%s" % tag] = r["amplitude"]
            facts["SigAmp%s" % tag] = r["sigma_amp"]
            facts["DChi%s" % tag] = r["delta_chi2_2dof"]
            facts["UL%s" % tag] = r["upper95_amplitude"]
            facts["NBins%s" % tag] = r["n_bins"]
        facts["ChiFidPlanck"] = pl["chi2_fiducial"]

        duo = P.combined_search()
        tri = P.combined_search(datasets=("planck_lite",
                                          "SPT3G_2018_TTTEEE_lite",
                                          "ACT_DR6_TTTEEE"))
        facts["AmpDuo"] = duo["amplitude"]
        facts["ULDuo"] = duo["upper95_amplitude"]
        facts["AmpTri"] = tri["amplitude"]
        facts["SigAmpTri"] = tri["sigma_amp"]
        facts["DChiTri"] = tri["delta_chi2_2dof"]
        facts["ULTri"] = tri["upper95_amplitude"]
        save()

    # robustness: six smooth directions
    if "ULTriRobust" not in facts:
        tri_r = P.combined_search(datasets=("planck_lite",
                                            "SPT3G_2018_TTTEEE_lite",
                                            "ACT_DR6_TTTEEE"),
                                  robust=True)
        facts["ULTriRobust"] = tri_r["upper95_amplitude"]
        save()

    # the BB channel and the wired tensor archive
    if "ULBB" not in facts:
        bb = P.layer_two_search("SPTpol_BB_lite")
        facts["ULBB"] = bb["upper95_amplitude"]
        facts["NBinsBB"] = bb["n_bins"]
        save()
    try:
        if "BKChiralSN" in facts:
            raise RuntimeError("cached")
        bk = P.Bk18Reader()
        facts["BKChiralSN"] = bk.chiral_forecast()["total_sn"]
        facts["BKNSpectra"] = len(bk.order)
        facts["BKNBins"] = bk.nbins
    except Exception:
        pass

    # the DSI second harmonic
    if "ULHarmTwo" not in facts:
        h2 = P.planck_layer_two(omega=2 * P.OMEGA_STAR)
        facts["ULHarmTwo"] = h2["upper95_amplitude"]
        facts["DChiHarmTwo"] = h2["delta_chi2_2dof"]
        save()

    # the GK foreground, priced
    from pyCICY.theories import etg_foreground as E
    if "ULTriShot" not in facts:
        gk = E.gk_reproduction()
        facts["GKZfPrinted"] = gk["z_f_eq5_as_printed"]
        facts["GKZfComputed"] = gk["z_f_eq5_computed"]
        fir = E.firas_limit()
        facts["GKFirasBest"] = min(fir["f_max_by_beta_d"].values())
        facts["GKFirasWorst"] = max(fir["f_max_by_beta_d"].values())
        shot = E.shot_noise_limit()
        facts["GKShotFmax"] = shot["f_max"]
        facts["GKOverpred"] = shot["overprediction_factor"]
        facts["GKClaimedOverAllowed"] = shot["claimed_over_allowed"]
        facts["GKPerBeam"] = shot["sources_per_planck_beam"]
        lw = E.layer_two_with_shot()
        facts["ULTriShot"] = lw["upper95_amplitude"]
        save()

    # frequencies
    facts["OmegaStar"] = P.OMEGA_STAR
    facts["LogPeriod"] = P.LOG_PERIOD
    facts["PeriodFactor"] = math.exp(P.LOG_PERIOD)

    # forecast
    fc = P.calibration_forecast()
    tags = {0.5: "Half", 0.277: "Today", 0.1: "PointOne",
            0.05: "SO", 0.01: "LB"}
    for row in fc["rows"]:
        tag = tags[row["sigma_cal_deg"]]
        facts["Detect%s" % tag] = row["detect_beta_sigma"]
        facts["Disc%s" % tag] = row["discriminate_pred_vs_0p342"]

    # tensor channel
    facts["RUpper"] = P.tensor_chirality_status()["r_upper_95"]

    def fmt(v):
        if isinstance(v, int):
            return str(v)
        if isinstance(v, float):
            if v == 0:
                return "0"
            if 0.01 <= abs(v) < 1e4:
                return "%.4g" % v
            m, e = ("%.1e" % v).split("e")
            return "\\ensuremath{%s\\times 10^{%d}}" % (m, int(e))
        return str(v)

    lines = ["% generated by paper/make_crossover_facts.py -- do not edit"]
    for k, v in sorted(facts.items()):
        lines.append("\\providecommand{\\CF%s}{}\\renewcommand{\\CF%s}{%s}"
                     % (k, k, fmt(v)))
    path = os.path.join(OUT, "crossover_facts.tex")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(OUT, "crossover_facts.json"), "w") as f:
        json.dump(facts, f, indent=1, default=str)
    print("wrote", path)
    print(json.dumps({k: (("%.4g" % v) if isinstance(v, float) else v)
                      for k, v in sorted(facts.items())}, indent=1))


if __name__ == "__main__":
    main()
