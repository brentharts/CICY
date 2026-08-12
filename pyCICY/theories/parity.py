r"""
pyCICY.theories.parity -- the observational layer of the crossover proposal:
parity violation in the CMB, and the log-periodic search at the substrate's
parameter-free frequency.

What this module is
-------------------
The Spectre-substrate paper makes two falsifiable statements. Conjecture 1
predicts a parity-asymmetric crossover: primordial chiral gravitational
waves or parity-odd CMB correlations of primordial origin are consistent
with it, exact parity-evenness to observational limits disfavors it. Its
falsifiability section predicts a log-periodic modulation of primordial
statistics with period log(lambda^2) = log(4 + sqrt15) = 2.0634... in
log k -- the fundamental unit of the substrate acting on scales, with no
free frequency.

This module confronts both statements with real data:

* the **parity channel** collects the published isotropic cosmic
  birefringence measurements (the EB/TB rotation angle beta), states their
  shared-systematic structure honestly, and renders the verdict the data
  currently support;
* the **log-periodic channel** is a direct search: CAMB computes CMB
  spectra from a primordial spectrum modulated at the substrate frequency,
  and the templates are fitted to real band powers (SPT-3G 2018 TT/TE/EE,
  ACT DR6 TT/TE/EE) through the candl likelihood machinery [candl], with
  smooth directions (amplitude, tilt, calibration) marginalized so the
  quasi-broadband modulation cannot hide in them.

The exact layer -- the rotation algebra of a uniformly birefringent sky --
is implemented with the same discipline as the rest of the series: the
mixing identities are exact trigonometry, verified as identities, and the
invariant (EE + BB is rotation-blind while EB sees 4 beta) is checked
rather than assumed.

The epistemic categories
------------------------
Exact: the rotation algebra; the template frequency omega* = 2 pi /
log(4 + sqrt15) (no free parameter). Measured, with a stated systematic:
beta. A modeling step, stated: the GLS template fit (linearized templates,
fiducial cosmology fixed, nuisance directions marginalized). Not yet
observable: the chirality of the tensor sector -- there is no detected
primordial gravitational-wave background whose handedness could be
measured, and the module says so instead of manufacturing a number.
"""

import math
import os

import numpy as np

from .base import Theory, register
from .ftheory import NoSuchTheory

__all__ = ["rotation_mixing", "rotation_invariants", "OMEGA_STAR",
           "LOG_PERIOD", "birefringence_measurements",
           "birefringence_status", "tensor_chirality_status",
           "camb_fiducial", "modulation_templates", "layer_two_search",
           "combined_search", "frequency_scan", "substrate_beta_prediction",
           "transfer_assumptions", "harmonic_search",
           "calibration_forecast", "PlanckLite", "planck_layer_two",
           "CrossoverParityProbe"]

# ---------------------------------------------------------------------------
# the exact layer: rotation algebra and the substrate frequency
# ---------------------------------------------------------------------------

#: The paper's parameter-free log-period in ln k: log(lambda^2).
LOG_PERIOD = math.log(4.0 + math.sqrt(15.0))          # 2.0634371...

#: The corresponding angular frequency in ln k.
OMEGA_STAR = 2.0 * math.pi / LOG_PERIOD               # 3.0452...


def rotation_mixing(cl_ee, cl_bb, cl_te, beta_deg, cl_eb=None, cl_tb=None):
    r"""The exact spectra of a sky uniformly rotated by beta.

    With E' = E cos(2b) - B sin(2b), B' = E sin(2b) + B cos(2b):

        EE' = EE cos^2(2b) + BB sin^2(2b) - EB sin(4b)
        BB' = EE sin^2(2b) + BB cos^2(2b) + EB sin(4b)
        EB' = (EE - BB) sin(4b)/2 + EB cos(4b)
        TE' = TE cos(2b) - TB sin(2b)
        TB' = TE sin(2b) + TB cos(2b)

    Exact identities, not small-angle expansions. This is the map whose
    inversion turns a measured EB into the angle beta; the miscalibration
    degeneracy is that an instrument rotation alpha produces the same map
    on the CMB but acts with its own angle on the (unrotated-at-emission)
    foregrounds, which is what the Minami--Komatsu method exploits.
    """
    b = math.radians(beta_deg)
    c2, s2 = math.cos(2 * b), math.sin(2 * b)
    c4, s4 = math.cos(4 * b), math.sin(4 * b)
    ee = np.asarray(cl_ee, dtype=float)
    bb = np.asarray(cl_bb, dtype=float)
    te = np.asarray(cl_te, dtype=float)
    eb = np.zeros_like(ee) if cl_eb is None else np.asarray(cl_eb)
    tb = np.zeros_like(te) if cl_tb is None else np.asarray(cl_tb)
    return {"EE": ee * c2 ** 2 + bb * s2 ** 2 - eb * s4,
            "BB": ee * s2 ** 2 + bb * c2 ** 2 + eb * s4,
            "EB": 0.5 * (ee - bb) * s4 + eb * c4,
            "TE": te * c2 - tb * s2,
            "TB": te * s2 + tb * c2}


def rotation_invariants(beta_deg=0.35, n=64, seed=0):
    """Verify the algebra as identities on random spectra.

    EE + BB is rotation-invariant; EB winds at 4 beta; TE^2 + TB^2 is
    invariant; and rotating by beta then -beta is the identity map. All
    verified to machine precision on random positive spectra.
    """
    rng = np.random.default_rng(seed)
    ee, bb = rng.random(n) + 1.0, rng.random(n) * 0.1
    te = rng.standard_normal(n)
    out = rotation_mixing(ee, bb, te, beta_deg)
    back = rotation_mixing(out["EE"], out["BB"], out["TE"], -beta_deg,
                           cl_eb=out["EB"], cl_tb=out["TB"])
    return {"sum_invariant": float(np.max(np.abs(out["EE"] + out["BB"]
                                                 - ee - bb))),
            "te_tb_invariant": float(np.max(np.abs(
                out["TE"] ** 2 + out["TB"] ** 2 - te ** 2))),
            "round_trip": float(max(np.max(np.abs(back["EE"] - ee)),
                                    np.max(np.abs(back["BB"] - bb)),
                                    np.max(np.abs(back["TE"] - te)),
                                    np.max(np.abs(back["EB"])),
                                    np.max(np.abs(back["TB"])))),
            "eb_slope_check": float(abs(
                (out["EB"][0] / (ee[0] - bb[0]))
                - 0.5 * math.sin(4 * math.radians(beta_deg))))}


# ---------------------------------------------------------------------------
# the parity channel: what has actually been measured
# ---------------------------------------------------------------------------

def birefringence_measurements():
    """The published isotropic birefringence angles, with provenance.

    All entries use the foreground-EB (Minami--Komatsu) method or angle
    calibration priors; the shared systematic is the modeling of Galactic
    dust EB and the absolute polarization-angle calibration, and the
    entries are NOT mutually independent (Planck data recur). The list is
    for orientation; the headline number is the 2026 joint analysis.
    """
    return [
        {"label": "Planck PR3 (Minami-Komatsu 2020)",
         "beta_deg": 0.35, "sigma": 0.14, "arxiv": "2011.11254"},
        {"label": "Planck PR4 (Diego-Palazuelos+ 2022)",
         "beta_deg": 0.30, "sigma": 0.11, "arxiv": "2201.07682"},
        {"label": "WMAP9 + Planck PR4 (Eskilt-Komatsu 2022)",
         "beta_deg": 0.342, "sigma": 0.0925, "arxiv": "2205.13962",
         "note": "3.6 sigma; asymmetric errors +0.094/-0.091"},
        {"label": "Cosmoglobe DR1 WMAP+LFI reprocessing (2023)",
         "beta_deg": 0.26, "sigma": 0.10, "arxiv": "2305.02268"},
        {"label": "ACT DR6 mean array rotation (Louis+ 2025)",
         "beta_deg": 0.20, "sigma": 0.08, "arxiv": "2503.14452",
         "note": "interpreted by ACT as mean detector rotation <psi>"},
        {"label": "ACT DR6 cosmological beta (Diego-Palazuelos-Komatsu "
                  "2025)", "beta_deg": 0.215, "sigma": 0.074,
         "arxiv": "2509.13654", "note": "2.9 sigma"},
        {"label": "ACT DR6 + Planck PR4 joint (Eskilt 2026)",
         "beta_deg": 0.277, "sigma": 0.057, "arxiv": "2608.06480",
         "note": "4.8 sigma at face-value instrument priors; "
                 "dust-robustness variant still excludes 0 at 3.5 sigma",
         "headline": True},
    ]


def birefringence_status():
    """The verdict the parity data currently support, stated carefully.

    Returns the headline measurement, its significance, and the mapping
    onto the paper's Conjecture 1 -- including what the measurement does
    NOT establish.
    """
    meas = birefringence_measurements()
    head = next(m for m in meas if m.get("headline"))
    z = head["beta_deg"] / head["sigma"]
    return {
        "headline": head,
        "significance_face_value": z,
        "significance_dust_robust": 3.5,
        "verdict": (
            "The sky is currently measured to be parity-ODD in EB at "
            "%.1f sigma (face-value instrument priors; 3.5 sigma under "
            "conservative dust modeling): beta = %.3f +/- %.3f deg. "
            "Exact parity-evenness is NOT what the data show, so the "
            "disfavoring branch of Conjecture 1 is not triggered; the "
            "consistent branch is active." % (z, head["beta_deg"],
                                              head["sigma"])),
        "caveats": [
            "isotropic birefringence is parity violation in photon "
            "propagation (e.g. a Chern-Simons coupling), not uniquely a "
            "primordial/crossover asymmetry; Conjecture 1 gains "
            "consistency, not confirmation",
            "the measurement chain shares two systematics: Galactic dust "
            "EB modeling and absolute polarization-angle calibration; "
            "the entries are correlated through reused Planck data",
            "the conjecture as stated is qualitative: it predicts a "
            "nonzero parity asymmetry of primordial origin but not the "
            "magnitude or sign of beta; a quantitative substrate "
            "prediction for beta is an open problem to add to the list",
        ],
    }


def tensor_chirality_status():
    """The gravitational-wave channel: not yet observable, and why.

    The chirality of the primordial tensor background would appear in TB
    and EB correlations sourced by gravitational waves. No primordial
    tensor background has been detected: BICEP/Keck 2021 bound
    r_0.05 < 0.036 (95%). A handedness measurement of an undetected
    background does not exist; this channel becomes live if and when r is
    detected.
    """
    return {"r_upper_95": 0.036, "reference": "BICEP/Keck XIII, "
            "PRL 127, 151301 (2021)",
            "status": "chirality unobservable until the tensor background "
                      "itself is detected"}


# ---------------------------------------------------------------------------
# the log-periodic channel: templates and the search
# ---------------------------------------------------------------------------

_PLANCK18 = dict(H0=67.36, ombh2=0.02237, omch2=0.1200, tau=0.0544,
                 As=2.1e-9, ns=0.9649)


def _camb_pars(lmax=4600):
    import camb
    p = camb.set_params(H0=_PLANCK18["H0"], ombh2=_PLANCK18["ombh2"],
                        omch2=_PLANCK18["omch2"], tau=_PLANCK18["tau"],
                        As=_PLANCK18["As"], ns=_PLANCK18["ns"],
                        lmax=lmax, lens_potential_accuracy=1)
    return p


def _dls_from_pars(p, lmax):
    import camb
    res = camb.get_results(p)
    cl = res.get_cmb_power_spectra(p, CMB_unit="muK",
                                   spectra=["total"])["total"]
    out = {}
    for i, key in enumerate(("TT", "EE", "BB", "TE")):
        arr = np.zeros(lmax + 1)
        n = min(lmax + 1, cl.shape[0])
        arr[:n] = cl[:n, i]
        out[key] = arr
    return out


def camb_fiducial(lmax=4600):
    """Fiducial Planck-2018 lensed spectra (Dl, muK^2, from ell 0)."""
    return _dls_from_pars(_camb_pars(lmax), lmax)


def _modulated_pk(amplitude, phase, kmin=5e-6, kmax=8.0, nk=4096):
    """P(k) = As (k/k0)^(ns-1) [1 + A cos(omega* ln(k/k0) + phase)]."""
    k0 = 0.05
    ks = np.logspace(math.log10(kmin), math.log10(kmax), nk)
    base = _PLANCK18["As"] * (ks / k0) ** (_PLANCK18["ns"] - 1.0)
    mod = 1.0 + amplitude * np.cos(OMEGA_STAR * np.log(ks / k0) + phase)
    return ks, base * mod


def modulation_templates(lmax=4600, amplitude=0.04, omega=None):
    r"""d Dl / dA for the two quadratures of the log-periodic modulation.

    The primordial spectrum is modulated at the substrate frequency
    (or ``omega`` if given), CAMB projects it through the full transfer
    functions -- no k ~ l/D approximation -- and the finite-difference
    templates per unit modulation amplitude are returned together with the
    tilt derivative d Dl / d ns used to protect the fit from absorbing
    smooth directions.
    """
    global OMEGA_STAR
    om_save = OMEGA_STAR
    if omega is not None:
        OMEGA_STAR = omega        # used inside _modulated_pk
    try:
        fid = camb_fiducial(lmax)
        temps = {}
        for name, phase in (("cos", 0.0), ("sin", 0.5 * math.pi)):
            p = _camb_pars(lmax)
            ks, pk = _modulated_pk(amplitude, phase)
            p.set_initial_power_table(
                ks, pk, effective_ns_for_nonlinear=_PLANCK18["ns"])
            dls = _dls_from_pars(p, lmax)
            temps[name] = {s: (dls[s] - fid[s]) / amplitude
                           for s in ("TT", "EE", "BB", "TE")}
        p = _camb_pars(lmax)
        p.InitPower.set_params(As=_PLANCK18["As"],
                               ns=_PLANCK18["ns"] + 0.01)
        dns = _dls_from_pars(p, lmax)
        temps["dns"] = {s: (dns[s] - fid[s]) / 0.01
                        for s in ("TT", "EE", "BB", "TE")}
        return fid, temps
    finally:
        OMEGA_STAR = om_save


# --- the substrate prediction, the transfer assumption, the forecast -----

def substrate_beta_prediction():
    r"""A quantitative candidate for beta from the substrate, exactly.

    SPECULATION-GRADE, and labeled as such. The substrate's only intrinsic
    chiral order parameter with a preferred magnitude is the geometric
    splitting maximum of the Spectre->Hat sweep,

        A_max = 10 g / (8 + 10 g) = 35/67 - (20/201) sqrt(15)
              = 0.1370166...,       g = 4 - sqrt(15),

    an exact number of the tiling (pyCICY.theories.spectre,
    order_parameter()), computed for the deformation analysis before any
    comparison with polarization data was contemplated. The candidate
    identification is

        beta_pred = 2 * A_max  [degrees]
                  = 70/67 - (40/201) sqrt(15)  degrees
                  = 0.2740332... degrees,

    where the factor 2 is the spin-2 convention (the polarization plane
    rotates by beta while the Stokes vector rotates by 2 beta; the
    substrate order parameter is identified with the Stokes-level
    asymmetry) and the unit postulate -- the dimensionless order parameter
    read in DEGREES -- is the entire modeling step, stated rather than
    hidden. Against the 2026 joint measurement beta = 0.277 +/- 0.057 deg
    the candidate sits at 0.05 sigma. That is either a coincidence among
    small numbers or a prediction; the 0.05-degree calibration era will
    decide, which is exactly what a speculation is for.
    """
    import fractions
    F = fractions.Fraction
    a_max_rational = (F(35, 67), F(-20, 201))         # a + b sqrt15
    beta_rational = (F(70, 67), F(-40, 201))
    a_max = float(a_max_rational[0]) + float(a_max_rational[1]) * math.sqrt(15)
    beta = 2.0 * a_max
    head = next(m for m in birefringence_measurements()
                if m.get("headline"))
    pull = (head["beta_deg"] - beta) / head["sigma"]
    return {"order_parameter_exact": a_max_rational,
            "beta_pred_exact": beta_rational,
            "beta_pred_deg": beta,
            "measured": head,
            "pull_sigma": pull,
            "grade": "speculation (unit postulate: degrees; factor 2: "
                     "Stokes convention)",
            "falsified_if": "|beta - 0.2740| > 5 sigma_cal once "
                            "sigma_cal ~ 0.05 deg (Simons Observatory "
                            "goal) or 0.01 deg class (LiteBIRD)"}


def transfer_assumptions():
    r"""The substrate -> P(k) transfer, made explicit.

    The search template is not arbitrary: if the crossover statistics
    inherit the substitution's discrete scale invariance k -> lambda^2 k,
    then any imprint on the primordial spectrum is a periodic function of
    ln k with period log(lambda^2),

        P(k) = P_0(k) * F(ln k mod log lambda^2),
        F(x) = 1 + sum_m [a_m cos(m omega* x) + b_m sin(m omega* x)],

    and the searched amplitude is the leading Fourier mode |c_1| of F.
    The modeling choices, in decreasing order of necessity:

    (i) discrete scale invariance itself (the content of the paper's
        layer-two prediction; gives the frequency, no amplitude);
    (ii) the imprint is multiplicative on P(k) (minimal coupling of the
        substrate modulation to curvature perturbations);
    (iii) truncation at the first harmonic (testable: the m = 2 harmonic
        sits at 2 omega*, and the same machinery bounds it --
        harmonic_search()).

    What DSI does not fix: the amplitude, the phase, and whether F acts
    on the spectrum or its logarithm (indistinguishable at first order).
    """
    return {"frequency_fixed_by": "discrete scale invariance under "
                                  "k -> lambda^2 k",
            "free": ["amplitude", "phase", "harmonic content"],
            "harmonics_at": [OMEGA_STAR, 2 * OMEGA_STAR, 3 * OMEGA_STAR],
            "first_harmonic_period_factor": float(math.exp(LOG_PERIOD))}


def harmonic_search(dataset="ACT_DR6_TTTEEE", m=2):
    """Bound the m-th DSI harmonic with the same machinery."""
    return layer_two_search(dataset, omega=m * OMEGA_STAR)


def calibration_forecast(sigma_cals=(0.5, 0.277, 0.1, 0.05, 0.01)):
    r"""What the calibration era decides, as arithmetic.

    For an absolute polarization-angle calibration sigma_cal (degrees)
    and negligible statistical error (the satellite/SO regime), the
    significance of the current central value and the discrimination
    between beta_pred = 0.2740 and beta = 0 are pure ratios. The 2026
    joint measurement is calibration-prior dominated; the entries below
    say when the question moves from priors to instrumental fact. Quoted
    program goals: Simons Observatory targets ~0.1-0.05 deg absolute
    calibration; LiteBIRD's requirement is of order 0.01 deg class for
    its birefringence science case [LiteBIRD forecast, JCAP 07 (2025)
    083].
    """
    pred = substrate_beta_prediction()["beta_pred_deg"]
    rows = []
    for sc in sigma_cals:
        rows.append({"sigma_cal_deg": sc,
                     "detect_beta_sigma": pred / sc,
                     "discriminate_pred_vs_zero": pred / sc,
                     "discriminate_pred_vs_0p342": abs(0.342 - pred) / sc})
    return {"beta_pred": pred, "rows": rows}


# --- Planck plik-lite: the two extra octaves ------------------------------

_PLANCK_LITE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "external", "planck_lite_data")


class PlanckLite(object):
    """Adapter for the Planck 2018 plik-lite band powers [PlanckLitePy].

    Reads the foreground-marginalized binned TT (215 bins, ell 30-2508),
    TE and EE (199 bins each, ell 30-1996) band powers, covariance and
    bin weights vendored from github.com/heatherprince/planck-lite-py
    into ``external/planck_lite_data``. Exposes the same three pieces the
    GLS core needs: ``data_bandpowers``, ``covariance`` and a Dl-binning
    function. Extends the search band from ell ~ 325 (SPT-3G) down to
    ell = 30: two more octaves of ln k, slightly over one additional
    period of the substrate modulation.
    """

    name = "Planck 2018 plik-lite TT/TE/EE"
    ell_max = 2508

    def __init__(self, data_dir=None):
        from scipy.io import FortranFile
        d = data_dir or _PLANCK_LITE_DIR
        pl = os.path.join(d, "planck2018_plik_lite") + os.sep
        self.plmin = 30
        self.nbintt, self.nbinte, self.nbinee = 215, 199, 199
        bval, X_data, X_sig = np.genfromtxt(
            pl + "cl_cmb_plik_v22.dat", unpack=True)
        self.data_bandpowers = X_data          # binned Cl, muK^2
        self.blmin = np.loadtxt(pl + "blmin.dat").astype(int)
        self.blmax = np.loadtxt(pl + "blmax.dat").astype(int)
        self.bin_w = np.loadtxt(pl + "bweight.dat")
        f = FortranFile(pl + "c_matrix_plik_v22.dat", "r")
        n = self.nbintt + self.nbinte + self.nbinee
        cov = f.read_reals(dtype=float).reshape((n, n))
        for i in range(n):
            for j in range(i, n):
                cov[i, j] = cov[j, i]
        self.covariance = cov

    def bin_dls(self, dls):
        """Bin theory Dl dict (arrays from ell 0) into the data ordering."""
        out = np.zeros(self.nbintt + self.nbinte + self.nbinee)
        for spec, nb, off in (("TT", self.nbintt, 0),
                              ("TE", self.nbinte, self.nbintt),
                              ("EE", self.nbinee,
                               self.nbintt + self.nbinte)):
            dl = dls[spec]
            ls = np.arange(dl.size)
            fac = np.where(ls > 0, ls * (ls + 1) / (2.0 * np.pi), 1.0)
            cl = dl / fac
            for i in range(nb):
                lo = self.blmin[i] + self.plmin
                hi = self.blmax[i] + self.plmin + 1
                out[off + i] = np.sum(
                    cl[lo:hi] * self.bin_w[self.blmin[i]:
                                           self.blmax[i] + 1])
        return out


def planck_layer_two(omega=None, _cache={}):
    """The layer-two GLS on the Planck plik-lite band powers."""
    key = ("planck", omega)
    if key in _cache:
        return _cache[key]
    like = PlanckLite()
    fid, temps = modulation_templates(lmax=like.ell_max + 2, omega=omega)
    b_fid = like.bin_dls(fid)
    resid = like.data_bandpowers - b_fid
    cols = []
    for name in ("cos", "sin"):
        d = {sp: fid[sp] + temps[name][sp] for sp in fid}
        cols.append(like.bin_dls(d) - b_fid)
    cols.append(b_fid.copy())                          # lnAs
    d = {sp: fid[sp] + temps["dns"][sp] * 0.01 for sp in fid}
    cols.append((like.bin_dls(d) - b_fid) / 0.01)      # ns
    d = {sp: fid[sp] * (1.02 if sp == "EE" else (1.01 if sp == "TE"
                                                 else 1.0)) for sp in fid}
    cols.append((like.bin_dls(d) - b_fid) / 0.01)      # Pcal
    X = np.array(cols).T
    C = like.covariance
    F = X.T @ np.linalg.solve(C, X)
    theta = np.linalg.solve(F, X.T @ np.linalg.solve(C, resid))
    cov = np.linalg.inv(F)
    a, ca = theta[:2], cov[:2, :2]
    chi2 = float(a @ np.linalg.solve(ca, a))
    amp = float(np.hypot(*a))
    sig = float(np.sqrt(0.5 * np.trace(ca)))
    out = {"dataset": like.name, "n_bins": int(like.data_bandpowers.size),
           "omega": omega or OMEGA_STAR,
           "A_cos": float(a[0]), "A_sin": float(a[1]),
           "sigma_cos": float(np.sqrt(ca[0, 0])),
           "sigma_sin": float(np.sqrt(ca[1, 1])),
           "cov_cs": float(ca[0, 1]),
           "amplitude": amp, "sigma_amp": sig,
           "delta_chi2_2dof": chi2,
           "upper95_amplitude": amp + 1.96 * sig,
           "chi2_fiducial": float(resid @ np.linalg.solve(C, resid))}
    _cache[key] = out
    return out


def _default_nuisance(like):
    """Central values of the likelihood's priors for required nuisances."""
    vals = {}
    for pr in getattr(like, "priors", []):
        names = pr.par_names if hasattr(pr, "par_names") else []
        centres = np.atleast_1d(np.asarray(pr.central_value)).ravel() \
            if hasattr(pr, "central_value") else []
        for nm, cv in zip(names, centres):
            vals[nm] = float(cv)
    for nm in getattr(like, "required_nuisance_parameters", []):
        vals.setdefault(nm, 1.0)
    return vals


def _binned(like, dls, nuisance):
    params = dict(nuisance)
    params["Dl"] = {k: v[2 : like.ell_max + 1] for k, v in dls.items()}
    model = like.get_model_specs(params)
    return np.asarray(like.bin_model_specs(model)).ravel()


def layer_two_search(dataset="SPT3G_2018_TTTEEE_lite", lmax=None,
                     amplitude=0.04, omega=None, _cache={}):
    r"""Fit the substrate's log-periodic template to real band powers.

    Generalized least squares in band-power space: the design matrix holds
    the two modulation quadratures plus three smooth nuisance directions
    (overall amplitude = the fiducial spectrum itself, the tilt derivative,
    and the temperature/polarization calibration split), the noise is the
    likelihood's full band-power covariance, and the reported quantities
    are the profiled quadrature amplitudes, their covariance, the joint
    amplitude, its significance (2 dof), and the 95% upper limit on the
    modulation amplitude |A| at the substrate frequency.
    """
    import candl
    import candl_data
    key = (dataset, omega)
    if key in _cache:
        return _cache[key]
    like = candl.Like(getattr(candl_data, dataset))
    lmax = lmax or like.ell_max
    fid, temps = modulation_templates(lmax=max(lmax, like.ell_max),
                                      omega=omega)
    nuis = _default_nuisance(like)
    b_fid = _binned(like, fid, nuis)
    data = np.asarray(like.data_bandpowers).ravel()
    resid = data - b_fid

    cols, names = [], []
    for name in ("cos", "sin"):
        d = {s: fid[s] + temps[name][s] for s in fid}
        cols.append(_binned(like, d, nuis) - b_fid)
        names.append(name)
    # smooth nuisance directions
    cols.append(b_fid.copy())                       # d/dlnAs ~ spectrum
    names.append("lnAs")
    d = {s: fid[s] + temps["dns"][s] * 0.01 for s in fid}
    cols.append((_binned(like, d, nuis) - b_fid) / 0.01)
    names.append("ns")
    # polarization-vs-temperature relative calibration direction
    d = {s: fid[s] * (1.02 if s in ("EE",) else (1.01 if s == "TE"
                                                 else 1.0)) for s in fid}
    cols.append((_binned(like, d, nuis) - b_fid) / 0.01)
    names.append("Pcal")

    X = np.array(cols).T
    C = np.asarray(like.covariance)
    Ci_X = np.linalg.solve(C, X)
    F = X.T @ Ci_X
    theta = np.linalg.solve(F, X.T @ np.linalg.solve(C, resid))
    cov = np.linalg.inv(F)
    a = theta[:2]
    ca = cov[:2, :2]
    chi2 = float(a @ np.linalg.solve(ca, a))
    amp = float(np.hypot(*a))
    # 95% upper limit on |A|: profile over phase, Rayleigh-like; use the
    # conservative circularised sigma
    sig = float(np.sqrt(0.5 * np.trace(ca)))
    upper95 = amp + 1.96 * sig
    chi2_fid = float(resid @ np.linalg.solve(C, resid))
    out = {"dataset": like.name, "n_bins": int(data.size),
           "omega": omega or OMEGA_STAR,
           "A_cos": float(a[0]), "A_sin": float(a[1]),
           "sigma_cos": float(np.sqrt(ca[0, 0])),
           "sigma_sin": float(np.sqrt(ca[1, 1])),
           "cov_cs": float(ca[0, 1]),
           "amplitude": amp, "sigma_amp": sig,
           "delta_chi2_2dof": chi2,
           "significance_1d_equiv": math.sqrt(max(chi2 - 2.0, 0.0)),
           "upper95_amplitude": upper95,
           "chi2_fiducial": chi2_fid,
           "nuisance_directions": names[2:]}
    _cache[key] = out
    return out


def combined_search(datasets=("SPT3G_2018_TTTEEE_lite",
                              "ACT_DR6_TTTEEE"), omega=None):
    """Inverse-Fisher combination of the quadrature fits across datasets.

    The band powers of SPT-3G 2018 and ACT DR6 are independent
    measurements (different instruments, sky masks and noise), so the
    (A_cos, A_sin) estimates combine by summing their Fisher matrices.
    Returns the joint amplitude, its significance, and the joint 95%
    upper limit at the substrate frequency.
    """
    Fs, ms = [], []
    per = []
    for ds in datasets:
        if ds == "planck_lite":
            r = planck_layer_two(omega=omega)
        else:
            r = layer_two_search(ds, omega=omega)
        per.append(r)
        cov = np.array([[r["sigma_cos"] ** 2, r.get("cov_cs", 0.0)],
                        [r.get("cov_cs", 0.0), r["sigma_sin"] ** 2]])
        Fi = np.linalg.inv(cov)
        Fs.append(Fi)
        ms.append(Fi @ np.array([r["A_cos"], r["A_sin"]]))
    F = sum(Fs)
    a = np.linalg.solve(F, sum(ms))
    cov = np.linalg.inv(F)
    chi2 = float(a @ F @ a)
    sig = float(np.sqrt(0.5 * np.trace(cov)))
    amp = float(np.hypot(*a))
    return {"datasets": [r["dataset"] for r in per],
            "omega": omega or OMEGA_STAR,
            "A_cos": float(a[0]), "A_sin": float(a[1]),
            "amplitude": amp, "sigma_amp": sig,
            "delta_chi2_2dof": chi2,
            "upper95_amplitude": amp + 1.96 * sig,
            "per_dataset": per}


def frequency_scan(dataset="SPT3G_2018_TTTEEE_lite",
                   omegas=(1.5, 2.0, 2.5, OMEGA_STAR, 3.5, 4.5, 6.0)):
    """The amplitude at a grid of frequencies, to contextualise omega*.

    A detection AT the substrate frequency is only meaningful against the
    amplitude landscape at neighbouring frequencies; a limit at omega* is
    strengthened by showing the machinery responds elsewhere.
    """
    return [layer_two_search(dataset, omega=w) for w in omegas]


# ---------------------------------------------------------------------------
# the theory object
# ---------------------------------------------------------------------------

@register
class CrossoverParityProbe(Theory):
    """The data-facing layer of the Spectre-substrate proposal."""

    key = "crossover-parity"

    def __init__(self, name=None):
        Theory.__init__(self, None, name=name)

    def geometry(self):
        return ("the observational sky: CMB polarization band powers and "
                "the published parity-odd correlation measurements")

    def spectrum(self):
        return {"log_period": LOG_PERIOD, "omega_star": OMEGA_STAR,
                "prediction": "log-periodic modulation of primordial "
                              "statistics at omega*, parameter-free"}

    def parity_verdict(self):
        return birefringence_status()

    def tensor_chirality(self):
        raise NoSuchTheory(
            "the chirality of the primordial tensor background is not an "
            "observable of any existing dataset: no such background has "
            "been detected (r < 0.036 at 95%), and the handedness of an "
            "undetected signal does not exist as a measurement. The "
            "channel opens if and when r is detected.")

    def missing_for_physical(self):
        return [
            "a quantitative substrate prediction for beta: Conjecture 1 "
            "is qualitative, and the measured 0.277 +/- 0.057 deg awaits "
            "a derived number to compare against",
            "the transfer from substrate modulation to curvature "
            "perturbations: the layer-two search assumes the log-period "
            "imprints multiplicatively on P(k), the minimal choice",
            "polarization-angle calibration at the 0.05 deg level "
            "(planned: Simons Observatory, LiteBIRD) to move the parity "
            "verdict from priors-dependent to instrumental fact",
        ]

    def describe(self):
        s = birefringence_status()
        return "\n".join([
            "%s" % self.name,
            "  parity channel   beta = %.3f +/- %.3f deg (%.1f sigma "
            "face value, 3.5 dust-robust)" % (
                s["headline"]["beta_deg"], s["headline"]["sigma"],
                s["significance_face_value"]),
            "  tensor channel   closed until r detected (r < 0.036)",
            "  layer two        log-periodic search at omega* = %.4f "
            "per ln k (period log(4+sqrt15))" % OMEGA_STAR,
        ])
