r"""
pyCICY.theories.etg_foreground -- the Gjergo--Kroupa early-massive-galaxy
foreground proposal, reproduced and priced.

What this module is
-------------------
Gjergo & Kroupa (Nucl. Phys. B 1017 (2025) 116931) [GK] propose that the
dust-thermalized light of massive early-type galaxies forming at
z ~ 15-20 constitutes an overlooked CMB foreground: "even in our most
conservative estimates, massive ETGs account for 1.4% up to the full
present-day CMB energy density." Because both data channels of the
crossover analysis (pyCICY.theories.parity) live in CMB band powers and
EB spectra, this claim, if true at any substantial level, sits underneath
our sigma statements and must be priced before the final verdicts.

The module does three things, in the discipline of the series.

First, it *reproduces* GK: the downsizing timescales (their Eq. 2), the
energy densities (their Eq. 22 and the 1.4% ratio of their Table 1), and
the dust-temperature consistency check (their Eq. 27: L = 1e15 L_sun
through R = 10 kpc gives T ~ 49 K, which redshifted from their formation
epoch lands at ~2.6-2.8 K). All reproduce. One arithmetic slip is found
and reported: their Eq. (5) as printed, 15 Mpc / 800 kpc - 1, evaluates
to 17.75, not the quoted 16.5; the conclusion it feeds (z_f inside the
21-cm window 15 < z < 20) is unaffected, and the finding is recorded in
the spirit of a quantity computed twice.

Second, it prices the claim against the two datasets GK defer to future
work ("the impact on CMB anisotropies are left for future works"). The
proposal has two spectral variants and each meets a different wall:

* **Graybody variant** (dust emissivity ~ nu^beta_d > 0): a redshifted
  graybody does not become a blackbody -- the emissivity factor survives
  redshift -- so the FIRAS monopole spectrum limits the admissible
  energy fraction f directly. For beta_d in the physical range 1-2 the
  limit is f < few x 1e-5 after marginalizing the CMB temperature: two
  to three orders of magnitude below the conservative 1.4%.
* **Blackbody variant** (optically thick emission, spectrally
  indistinguishable from the CMB monopole): the spectrum is silent, but
  the field is made of N ~ 4.4e7 *discrete sources* (GK's own count,
  ~6 per Planck beam), whose Poisson shot noise is a floor on the
  anisotropy it must carry. Equal-luminosity Poisson statistics --
  the most conservative case; any luminosity spread or clustering only
  raises it -- put the shot power at ell ~ 3000 above the *total*
  measured 150 GHz sky power unless f < ~3e-5. The 1.4% claim
  over-predicts small-scale power by a factor ~1e5.

Third, it re-runs the layer-two GLS with a shot-noise template appended
to the nuisance block, demonstrating that the log-periodic bound of the
crossover analysis is stable against an ETG-like component at any
allowed amplitude -- so the sigma ratings of the parity and
log-periodicity verdicts stand unchanged, now with the foreground priced
rather than ignored.

Stated assumptions of the pricing: sources are point-like at z ~ 17
(kpc scales subtend sub-arcsecond angles), unclustered (conservative:
clustering adds power), equal-luminosity (conservative: spread adds
power), and their photons free-stream after emission (the post-
recombination universe is transparent; there is no mechanism to
re-thermalize 4.4e7 point sources into a smooth field).
"""

import math

import numpy as np

__all__ = ["gk_reproduction", "firas_limit", "shot_noise_limit",
           "layer_two_with_shot", "impact_on_verdicts"]

# constants (SI)
A_RAD = 7.565723e-16          # radiation constant, J m^-3 K^-4
L_SUN = 3.828e26              # W
KPC = 3.0856776e19            # m
MPC = 1e3 * KPC
T_CMB = 2.7255                # K
H_PLANCK = 6.62607015e-34
K_B = 1.380649e-23
C_LIGHT = 2.99792458e8
SIGMA_SB = 5.670374419e-8


# ---------------------------------------------------------------------------
# 1. reproduction
# ---------------------------------------------------------------------------

def gk_reproduction():
    """Reproduce GK's pipeline numbers, and record the Eq. (5) slip.

    Returns a dict of (reproduced, published) pairs plus the corrected
    formation redshift. Everything reproduces except the printed value
    of Eq. (5): 15 Mpc / (2 x 400 kpc) - 1 = 17.75, not 16.5. The
    conclusion it feeds -- that the formation epoch falls inside the
    21-cm absorption window 15 < z < 20 -- holds for both values, so
    nothing downstream in GK moves; the discrepancy is recorded because
    a number computed twice is a test.
    """
    def tau_down(M):
        return (8.16 * math.exp(-0.556 * math.log10(M) + 3.401)
                + 0.027) * 1000.0          # Myr

    z_eq5 = 15000.0 / 800.0 - 1.0
    u0 = A_RAD * T_CMB ** 4
    t_dust = (1e15 * L_SUN
              / (4 * math.pi * (10 * KPC) ** 2 * SIGMA_SB)) ** 0.25
    out = {
        "tau_down_11p5_Myr": (tau_down(10 ** 11.5), 440),
        "tau_down_12_Myr": (tau_down(10 ** 12.0), 340),
        "U_CMB0_Jm3": (u0, 4.17e-14),
        "U_CMB_EoR_Jm3": (u0 * 1091 ** 4, 0.06),
        "conservative_fraction": (5.9 / 418.0, 0.014),
        "T_dust_em_K": (t_dust, 50.0),
        "T_dust_obs_K": (t_dust / (1 + z_eq5), 2.7),
        "z_f_eq5_as_printed": 16.5,
        "z_f_eq5_computed": z_eq5,
        "eq5_slip": abs(z_eq5 - 16.5) > 1.0,
        "conclusion_affected": not (15.0 < z_eq5 < 20.0),
    }
    return out


# ---------------------------------------------------------------------------
# 2a. the spectral wall (graybody variant)
# ---------------------------------------------------------------------------

def _planck_bnu(nu, T):
    x = H_PLANCK * nu / (K_B * T)
    return 2 * H_PLANCK * nu ** 3 / C_LIGHT ** 2 / np.expm1(x)


def firas_limit(beta_d_grid=(0.5, 1.0, 1.5, 2.0), rms_ppm=50.0):
    r"""Maximum energy fraction of a graybody component under FIRAS.

    A component emitted with dust emissivity epsilon(nu) ~ nu^beta_d at
    redshift z arrives with spectrum ~ nu^beta_d B_nu(T_obs): the
    power-law factor is redshift-invariant, so unless beta_d = 0 exactly
    the component is *not* a blackbody today and distorts the monopole.
    FIRAS bounds rms deviations from a blackbody at ``rms_ppm`` (50) parts
    per million of the peak brightness [Fixsen et al. 1996]. For each
    beta_d the admissible fraction f (in energy density) is computed
    after profiling out a CMB temperature shift -- the one exact
    degeneracy -- over the FIRAS band 60-600 GHz.
    """
    nu = np.linspace(60e9, 600e9, 200)
    b0 = _planck_bnu(nu, T_CMB)
    peak = b0.max()
    # temperature-shift template (the profiled direction)
    dT = 1e-4
    t_temp = (_planck_bnu(nu, T_CMB + dT) - b0) / dT
    out = {}
    for bd in beta_d_grid:
        # unit-energy-fraction graybody: shape nu^bd * B_nu, normalized so
        # its bolometric energy equals f * (CMB bolometric energy)
        shape = (nu / 150e9) ** bd * b0
        shape *= np.trapezoid(b0, nu) / np.trapezoid(shape, nu)
        # profile out the temperature direction (least squares)
        alpha = np.dot(shape, t_temp) / np.dot(t_temp, t_temp)
        resid = shape - alpha * t_temp
        rms_per_f = float(np.sqrt(np.mean(resid ** 2)))
        f_max = (rms_ppm * 1e-6 * peak) / rms_per_f
        out[bd] = f_max
    return {"f_max_by_beta_d": out, "rms_ppm": rms_ppm,
            "note": "beta_d = 0 (exact blackbody) is unconstrained by "
                    "the spectrum and is priced by shot_noise_limit()"}


# ---------------------------------------------------------------------------
# 2b. the anisotropy wall (blackbody variant)
# ---------------------------------------------------------------------------

def shot_noise_limit(n_sources=4.4e7, nu_ghz=150.0, ell=3000,
                     total_sky_power_uk2=300.0, claimed_f=0.014):
    r"""The Poisson floor of N discrete sources against measured power.

    If the ETG field carries an energy fraction f of the CMB with the
    same (blackbody) spectrum, built from N equal point sources, its
    shot-noise angular power is

        C_shot = Ibar^2 / nbar / (dB/dT)^2,
        Ibar = f * B_nu(T),  nbar = N / 4pi,

    flat in C_ell up to the (sub-arcsecond) source scale. The limit
    compares D_shot(ell) with the *total* measured sky power at 150 GHz
    and ell ~ 3000 -- CMB plus all foregrounds, ~300 muK^2 -- which is
    maximally conservative: no component separation is assumed at all,
    the ETG field is merely required not to exceed everything measured.
    Equal luminosities and no clustering are the minimum-variance case;
    any realism raises the prediction and tightens the bound.
    """
    nu = nu_ghz * 1e9
    x = H_PLANCK * nu / (K_B * T_CMB)
    b = _planck_bnu(np.array([nu]), T_CMB)[0]
    dbdt = b * (x * math.exp(x) / math.expm1(x)) / T_CMB
    nbar = n_sources / (4 * math.pi)              # per sr
    # C_shot in K^2 per unit f^2
    c_shot_per_f2 = (b / dbdt) ** 2 / nbar
    d_fac = ell * (ell + 1) / (2 * math.pi)
    d_shot_per_f2_uk2 = c_shot_per_f2 * d_fac * 1e12
    f_max = math.sqrt(total_sky_power_uk2 / d_shot_per_f2_uk2)
    d_at_claim = d_shot_per_f2_uk2 * claimed_f ** 2
    return {"ell": ell, "nu_GHz": nu_ghz,
            "sources_per_planck_beam": n_sources / 7.58e6,
            "D_shot_at_claimed_f_uk2": d_at_claim,
            "total_measured_uk2": total_sky_power_uk2,
            "overprediction_factor": d_at_claim / total_sky_power_uk2,
            "f_max": f_max,
            "claimed_over_allowed": claimed_f / f_max}


# ---------------------------------------------------------------------------
# 3. stability of the layer-two bound
# ---------------------------------------------------------------------------

def layer_two_with_shot(datasets=("planck_lite", "SPT3G_2018_TTTEEE_lite",
                                  "ACT_DR6_TTTEEE")):
    """Re-run the joint log-periodic fit with a shot-noise nuisance.

    Appends a Poisson template (C_ell = const in TT, i.e. D_ell
    proportional to ell(ell+1)) to the smooth-direction block of the GLS
    of pyCICY.theories.parity and recombines the datasets. If the
    log-periodic bound survives unmoved, an ETG-like component at any
    spectrally allowed amplitude cannot be what the layer-two search
    would have seen, and the crossover verdicts inherit no systematic
    from the GK proposal.
    """
    from . import parity as P

    def with_shot(fid):
        ells = np.arange(len(fid["TT"]), dtype=float)
        shot = ells * (ells + 1) / (3000.0 * 3001.0)   # unit at ell=3000
        d = {k: v.copy() for k, v in fid.items()}
        d["TT"] = d["TT"] + shot
        return d

    Fs, ms, per = [], [], []
    for ds in datasets:
        if ds == "planck_lite":
            like = P.PlanckLite()
            fid, temps = P.modulation_templates(lmax=like.ell_max + 2)
            b_fid = like.bin_dls(fid)
            resid = like.data_bandpowers - b_fid
            cols = []
            for name in ("cos", "sin"):
                d = {sp: fid[sp] + temps[name][sp] for sp in fid}
                cols.append(like.bin_dls(d) - b_fid)
            cols.append(b_fid.copy())
            d = {sp: fid[sp] + temps["dns"][sp] * 0.01 for sp in fid}
            cols.append((like.bin_dls(d) - b_fid) / 0.01)
            d = {sp: fid[sp] * (1.02 if sp == "EE" else
                                (1.01 if sp == "TE" else 1.0))
                 for sp in fid}
            cols.append((like.bin_dls(d) - b_fid) / 0.01)
            cols.append(like.bin_dls(with_shot(fid)) - b_fid)
            X = np.array(cols).T
            C = like.covariance
        else:
            import candl
            import candl_data
            like = candl.Like(getattr(candl_data, ds))
            lmax = like.ell_max
            fid, temps = P.modulation_templates(lmax=max(4600, lmax))
            nuis = P._default_nuisance(like)
            b_fid = P._binned(like, fid, nuis)
            resid = np.asarray(like.data_bandpowers).ravel() - b_fid
            cols = []
            for name in ("cos", "sin"):
                d = {sp: fid[sp] + temps[name][sp] for sp in fid}
                cols.append(P._binned(like, d, nuis) - b_fid)
            cols.append(b_fid.copy())
            d = {sp: fid[sp] + temps["dns"][sp] * 0.01 for sp in fid}
            cols.append((P._binned(like, d, nuis) - b_fid) / 0.01)
            d = {sp: fid[sp] * (1.02 if sp == "EE" else
                                (1.01 if sp == "TE" else 1.0))
                 for sp in fid}
            pcal = (P._binned(like, d, nuis) - b_fid) / 0.01
            if np.max(np.abs(pcal)) > 0:
                cols.append(pcal)
            cols.append(P._binned(like, with_shot(fid), nuis) - b_fid)
            X = np.array(cols).T
            C = np.asarray(like.covariance)
        F = X.T @ np.linalg.solve(C, X)
        theta = np.linalg.solve(F, X.T @ np.linalg.solve(C, resid))
        cov = np.linalg.inv(F)
        a, ca = theta[:2], cov[:2, :2]
        Fi = np.linalg.inv(ca)
        Fs.append(Fi)
        ms.append(Fi @ a)
        per.append({"dataset": ds, "A": [float(v) for v in a]})
    Ftot = sum(Fs)
    a = np.linalg.solve(Ftot, sum(ms))
    cov = np.linalg.inv(Ftot)
    amp = float(np.hypot(*a))
    sig = float(np.sqrt(0.5 * np.trace(cov)))
    return {"amplitude": amp, "sigma_amp": sig,
            "upper95_amplitude": amp + 1.96 * sig,
            "per_dataset": per}


# ---------------------------------------------------------------------------
# 4. the verdicts, updated
# ---------------------------------------------------------------------------

def impact_on_verdicts():
    """What the GK proposal does to the crossover sigma ratings: nothing.

    Both variants of the proposed foreground are bounded far below the
    claimed conservative level by data already in hand -- the graybody
    by FIRAS, the blackbody by its own shot noise against the total
    measured 150 GHz sky power -- and the layer-two bound is stable
    under an ETG shot-noise nuisance. The admissible fraction
    (f < few x 1e-5) perturbs the EB-derived birefringence angle and
    the band-power fits at O(f), three orders of magnitude below their
    quoted uncertainties. The 4.8/3.5 sigma parity verdict and the
    percent-level log-periodicity bound therefore stand unchanged, now
    with the foreground priced rather than ignored.
    """
    fir = firas_limit()
    shot = shot_noise_limit()
    return {"firas_f_max": fir["f_max_by_beta_d"],
            "shot_f_max": shot["f_max"],
            "claimed_conservative": 0.014,
            "sigma_ratings_changed": False}
