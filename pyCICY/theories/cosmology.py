r"""
pyCICY.theories.cosmology -- cosmological singularities as registry entries.

Five entries, four of them not compactifications (``X = None``, as for `NariaiEntropic` and
`ComplexSixSphere`), sharing the new base class :class:`~.cosmobase.CosmologyTheory`:

    kasner-vacuum    the Kasner solution: exponents exact, curvature by three independent routes (the closed form,
                     a sympy Riemann tensor and EinsteinPy), singularity morphology, horizon rule
    mixmaster-bkl    vacuum Bianchi IX: the BKL map exact in rational arithmetic, the curvature numerical with the
                     Hamiltonian constraint monitored, and the horizon result that contradicts Misner's hope
    pancake-bigbang  the non-singular pancake of Dechant, Lasenby and Hobson: series, K(0) in closed form, horizons
    flrw-toys        flat FLRW: LCDM, a sigmoid toy, a coupled-black-hole toy, Tsallis and Pade models from the
                     literature, and an exact bounce, with horizon entropy and its second-law criterion

The modules underneath (:mod:`pyCICY.kasner`, :mod:`~pyCICY.bkl`, :mod:`~pyCICY.pancake`, :mod:`~pyCICY.horizons`,
:mod:`~pyCICY.tidal`, :mod:`~pyCICY.flrw`, :mod:`~pyCICY.ccbh`, :mod:`~pyCICY.bkl_sweep`) were ported from Quantum
Gravity Lab (CC0). The sensor coupling of that project was left behind on purpose; see
:mod:`~pyCICY.theories.cosmobase`.
"""

from fractions import Fraction

import numpy as np

from .base import register
from .cosmobase import CosmologyTheory, NoSuchTheory, NeedsFit
from .. import kasner as _kn
from .. import bkl as _bkl
from .. import pancake as _pk
from .. import horizons as _hz
from .. import flrw as _fl
from .. import ccbh as _cc
from .. import moduli_kasner as _mk

__all__ = ["KasnerVacuum", "MixmasterBKL", "PancakeBigBang", "FLRWToys", "ModuliKasner"]


def _exact(u):
    """u as a Fraction when it is an int, a Fraction or a decimal string, else float."""
    if isinstance(u, (int, Fraction)):
        return Fraction(u)
    if isinstance(u, str):
        return Fraction(u)
    return float(u)


# ---------------------------------------------------------------------------
#  Kasner
# ---------------------------------------------------------------------------

@register
class KasnerVacuum(CosmologyTheory):
    """The vacuum Kasner solution ds^2 = -dt^2 + sum t^(2 p_i) dx_i^2 with exponents from the parameter u >= 1."""

    key = "kasner-vacuum"

    def __init__(self, u=2, name=None):
        CosmologyTheory.__init__(self, name=name)
        self.u = _exact(u)

    def geometry(self):
        return "Kasner vacuum, u = %s, a_i = t^p_i with sum p = sum p^2 = 1" % (self.u,)

    def exponents(self, **kw):
        """Exact exponents (Fractions for rational u) in the order negative, middle, large."""
        return _kn.kasner_exponents(self.u)

    def curvature(self, t=None, **kw):
        """K t^4 = -16 p1 p2 p3 (exact), and K(t) when t is given."""
        p = self.exponents()
        coeff = -16 * p[0] * p[1] * p[2]
        return coeff if t is None else coeff / t ** 4

    def verify_curvature(self, einsteinpy=True):
        """
        Compare the closed form with a sympy Riemann tensor, and with EinsteinPy's. Returns a dict with the three
        Kretschmann coefficients (K t^4) and whether Ricci vanishes on each symbolic route.
        """
        import sympy as sp
        p = tuple(sp.Rational(x.numerator, x.denominator) if isinstance(x, Fraction) else sp.nsimplify(x)
                  for x in self.exponents())
        out = dict(closed=sp.nsimplify(self.curvature()))
        c = _kn.sympy_curvature(p)
        out["sympy"] = sp.simplify(c["kretschmann"] * c["t"] ** 4)
        out["sympy_ricci_zero"] = c["ricci_is_zero"]
        if einsteinpy:
            e = _kn.einsteinpy_curvature(p)
            out["einsteinpy"] = sp.simplify(e["kretschmann"] * e["t"] ** 4)
            out["einsteinpy_ricci_zero"] = e["ricci_is_zero"]
        return out

    def singularity_type(self, direction="singularity", **kw):
        return _kn.singularity_type([float(x) for x in self.exponents()], direction)

    def horizon(self, **kw):
        """Per axis: the horizon type and the proper horizon t / (1 - p), exact."""
        p = self.exponents()
        return dict(types=tuple(_hz.horizon_type(float(x)) for x in p),
                    proper_over_t=tuple((1 / (1 - x)) for x in p),
                    statement="comoving horizon t^(1-p)/(1-p), finite for every axis of a vacuum Kasner solution "
                              "with u > 1; it diverges only at p = 1, the flat (1, 0, 0)")

    def exact_content(self):
        return [
            "the exponents (-u, 1+u, u(1+u)) / (1+u+u^2), exact for rational u, with both Kasner conditions",
            "K = -16 p1 p2 p3 / t^4 by three routes: the closed form, a sympy Riemann tensor written out from the "
            "metric, and EinsteinPy's symbolic tensors; Ricci = 0 on both symbolic routes",
            "Riemann = 0 only for (0, 0, 1): that exponent set is flat spacetime",
            "singularity morphology (point, cigar, barrel, pancake) of the scale factors at either end of time",
            "the horizon rule: comoving horizon t^(1-p)/(1-p), finite for p < 1",
            "timelike geodesics from the conserved covariant momenta (numerical quadrature of exact equations)",
        ]

    def declined(self):
        return [
            "that the Kasner solution describes a physical early universe: vacuum, anisotropic, spatially flat",
            "any comparison with observation (NeedsFit)",
        ]


# ---------------------------------------------------------------------------
#  Mixmaster
# ---------------------------------------------------------------------------

@register
class MixmasterBKL(CosmologyTheory):
    """Vacuum Bianchi IX: a sequence of Kasner epochs joined by bounces (BKL / Mixmaster)."""

    key = "mixmaster-bkl"

    def __init__(self, u="37/10", name=None):
        CosmologyTheory.__init__(self, name=name)
        self.u = _exact(u)

    def geometry(self):
        return "vacuum Bianchi IX (Mixmaster), starting Kasner parameter u = %s" % (self.u,)

    def epochs(self, n):
        """
        The first n Kasner epochs from the BKL map, in exact rational arithmetic when u is rational: u, era and
        exponents. Stops early, without error, when the orbit reaches u = 1 or 2 (a rational u terminates).
        """
        out, u, era = [], self.u, 0
        for _ in range(n):
            p = _kn.kasner_exponents(u)
            out.append(dict(u=u, era=era, exponents=p, K_coefficient=-16 * p[0] * p[1] * p[2]))
            try:
                u, new_era = _bkl.bkl_u_map(u)
            except ValueError:
                break
            era += int(new_era)
        return out

    def exponents(self, **kw):
        return self.epochs(1)[0]["exponents"]

    def solve(self, t_bounce=1.0, t_end=1e-30):
        """The numerical Bianchi IX solution (:class:`pyCICY.bkl.Mixmaster`) from a clean first epoch."""
        u = float(self.u)
        return _bkl.Mixmaster(u, _bkl.clean_t0(u, t_bounce), t_bounce, t_end)

    def curvature(self, t=None, **kw):
        """Numerical: the Weyl magnitude along the integrated solution. Exact only inside an epoch."""
        mm = self.solve()
        tt = np.geomspace(mm.t0, mm.t_end, 200) if t is None else np.atleast_1d(t)
        return tt, mm.weyl_magnitude(tt)

    def singularity_type(self, **kw):
        return "point-like and oscillatory: no single morphology (the roles of the axes rotate every bounce)"

    def horizon(self, n_epochs=40, theta=0.5):
        """
        The horizon sum over n_epochs Kasner epochs for an irrational start (pi), calibrated on the numerical
        solution; returns per-axis log10 of the cumulative comoving horizon. It converges: every axis has a
        finite horizon under the criterion int dt / a_i. See :mod:`pyCICY.horizons` for what that criterion is.
        """
        import mpmath as mp
        u = float(np.pi)
        mm = _bkl.Mixmaster(u, _bkl.clean_t0(u), 1.0, 1e-40)
        (_, A1) = _hz.epoch_amplitudes(mm)[0]
        return _hz.mixmaster_horizon_map(mp.pi, A1, float(np.log(mm.t0)), n_epochs, theta=theta)

    def exact_content(self):
        return [
            "the BKL map in exact rational arithmetic: u -> u - 1 for u >= 2, 1/(u - 1) for 1 < u < 2, and the "
            "per-axis Landau-Lifshitz rule, checked against each other",
            "K t^4 = 16 u^2 (1+u)^2 / (1+u+u^2)^3 in each epoch",
            "the Kretschmann scalar of Bianchi IX as a closed expression in a, a', a'', validated against a "
            "brute-force Riemann tensor",
            "the structure of the horizon result: d ln(t/a_i)/d ln t = 1 - p_i > 0 in every vacuum epoch",
        ]

    def declined(self):
        return [
            "the Bianchi IX solution itself and its Weyl curvature: numerical, integrated in ln V with the "
            "Hamiltonian constraint monitored (not exact)",
            "the infinite epoch sequence: handled by a sharp-wall map calibrated against the integration to within "
            "a factor of order 2 in the horizon; a modelling step, not a theorem",
            "that int dt / a_i is the right horizon criterion on Bianchi IX: it is exact for Bianchi I and used "
            "here as a proxy; real null geodesics of Bianchi IX were not integrated",
            "Misner's own argument, about the Hubble horizon of a momentarily static axis, which is a different "
            "quantity; and the measure of circumnavigating initial conditions (Chitre 1972, abstract only read)",
            "chaos of the map as a measure statement (the Gauss map): not tested here",
        ]


# ---------------------------------------------------------------------------
#  Pancake
# ---------------------------------------------------------------------------

@register
class PancakeBigBang(CosmologyTheory):
    """
    The non-singular pancake Big Bang: a massive scalar field in biaxial Bianchi IX, R_1 = t (a0 + a2 t^2 + ...),
    R_2 = R_3 = b0 + b2 t^2 + ..., finite curvature at t = 0 although the volume vanishes (arXiv:0809.4335).
    """

    key = "pancake-bigbang"

    def __init__(self, a0=1.0, b0=3.0, f0=2.0, m=0.5, Lam=0.0, kappa=1.0, name=None):
        CosmologyTheory.__init__(self, name=name)
        self.params = dict(a0=a0, b0=b0, f0=f0, m=m, Lam=Lam, kappa=kappa)

    def geometry(self):
        return "biaxial Bianchi IX with a massive scalar field, a0=%g b0=%g f0=%g m=%g" % (
            self.params["a0"], self.params["b0"], self.params["f0"], self.params["m"])

    def exponents(self, **kw):
        """The scale-factor behaviour at t = 0: R_1 ~ t (exponent 1), R_2 = R_3 -> b0 (exponent 0)."""
        return (Fraction(1), Fraction(0), Fraction(0))

    def series(self):
        P = self.params
        return _pk.pancake_series(P["a0"], P["b0"], P["f0"], P["m"], P["Lam"], P["kappa"])

    def curvature(self, **kw):
        """K(0) in closed form: 16 e^2 + (128 - 48 a0^2) / b0^4 with e = Lam/2 + kappa m^2 f0^2/4 - 2/b0^2."""
        P = self.params
        return _pk.kretschmann_at_pancake(P["b0"], P["f0"], P["m"], P["Lam"], P["kappa"], P["a0"])

    def solution(self, t_max=6.0):
        P = self.params
        return _pk.pancake_solution(P["a0"], P["b0"], P["f0"], P["m"], P["Lam"], P["kappa"], t_max=t_max)

    def singularity_type(self, **kw):
        return _kn.singularity_type([float(x) for x in self.exponents()])

    def horizon(self, t_max=6.0):
        """Fibre: logarithmic coefficient 2/a0, divergent. Base: finite, antipodal contact time from the solution."""
        hz = _hz.pancake_horizons(self.solution(t_max))
        return dict(fibre_log_coefficient=_hz.fibre_log_coefficient(self.params["a0"]),
                    base_contact_time=_hz.base_contact_time(hz),
                    statement="the collapsing fibre has a logarithmically divergent horizon round a circle of "
                              "vanishing circumference; the open base keeps a finite one")

    def vacuum_pancake(self):
        """Always raises: there is no finite-curvature vacuum pancake."""
        raise NoSuchTheory(
            "the only vacuum Kasner exponents with one collapsing axis and two finite ones are (1, 0, 0), which is "
            "flat spacetime: a coordinate singularity, with K = 0. A pancake with finite, non-zero curvature at "
            "t = 0 needs matter (here a massive scalar field); it is not a vacuum solution. This is an absence, "
            "not a missing computation.")

    def exact_content(self):
        return [
            "the odd-parity series coefficients a2 = 2 a0 / (3 b0^2), b2, f2 solved from the field equations; the "
            "Hamiltonian constraint holds identically at t = 0",
            "K(0) = 16 e^2 + (128 - 48 a0^2) / b0^4, checked against a brute-force Riemann tensor, with a0 != 1 "
            "and at the paper's parameters",
            "the tidal eigenvalues at t = 0, (4/b0^2, e, e): finite for every choice of the free data",
            "the horizon rule: logarithmic coefficient 2/a0 along the collapsing fibre",
            "winding of x-geodesics: d x / d ln t -> 2/a0, proper time spent near the pancake ~ a0 t1^2 / (2|L|)",
        ]

    def declined(self):
        return [
            "the solution away from t = 0: numerical (DOP853), constraint monitored",
            "the period of the fibre coordinate (4 pi on the standard S^3): not checked against the paper's "
            "identification",
            "the paper's value of Lambda, which the page read did not give (set to 0)",
            "the triaxial perturbation's late-time behaviour beyond the runs made",
            "inflation, reheating or any observational statement (NeedsFit)",
        ]


# ---------------------------------------------------------------------------
#  Flat FLRW toys
# ---------------------------------------------------------------------------

@register
class FLRWToys(CosmologyTheory):
    """
    Flat FLRW histories E^2(a): LCDM, our sigmoid toy, our coupled-black-hole toy, and a Tsallis and a Pade model
    from the literature, plus an exact bounce. The toys are our own, are not fitted and do not reproduce the
    models that inspired them.
    """

    key = "flrw-toys"

    MODELS = ("lcdm", "toy", "tsallis", "pade", "bounce", "ccbh")

    def __init__(self, name=None, **params):
        CosmologyTheory.__init__(self, name=name)
        self.params = params

    def geometry(self):
        return "flat FLRW, E^2(a) from one of %s" % (", ".join(self.MODELS),)

    def model(self, which="lcdm", **kw):
        """E^2(a) as a callable, for `which` in MODELS. Keyword arguments override the defaults."""
        kw = dict(self.params, **kw)
        if which == "lcdm":
            return lambda a: _fl.lcdm_E2(a, kw.get("Om", 0.3), kw.get("Or", 0.0))
        if which == "toy":
            tp = _fl.ToyParams(Om=kw.get("Om", 0.3), Or=kw.get("Or", 0.0), a_t=kw.get("a_t", 0.4), k=kw.get("k", 4.0))
            return lambda a: _fl.toy_E2(a, tp)
        if which == "tsallis":
            return lambda a: _fl.tsallis_E2(a, kw.get("beta", 1.1), kw.get("Om", 0.3), kw.get("Or", 0.0))
        if which == "pade":
            return lambda a: _fl.pade_E2(a, kw.get("q0", -0.6), kw.get("Om", 0.3))
        if which == "bounce":
            return lambda a: _fl.bounce_E2(a, kw.get("Om", 0.3), kw.get("rho_c", 10.0))
        if which == "ccbh":
            bg = _cc.ccbh_background(kw.get("Om", 0.3), kw.get("eps", 3.7), kw.get("k", 3.0), kw.get("h", 0.7))
            return bg.E2
        raise KeyError("unknown model %r; available: %s" % (which, ", ".join(self.MODELS)))

    def _grid(self, which, a, n, **kw):
        if a is not None:
            return np.asarray(a, dtype=float)
        if which == "bounce":        # E^2 < 0 below the bounce a_b = (Om / rho_c)^(1/3)
            kw = dict(self.params, **kw)
            a_b = (kw.get("Om", 0.3) / kw.get("rho_c", 10.0)) ** (1.0 / 3.0)
            return np.geomspace(1.001 * a_b, 1.0, n)
        return np.geomspace(0.1, 1.0, n)

    def second_law(self, which="lcdm", a=None, **kw):
        """
        dS/dt >= 0 for the horizon entropy S = pi/(G H^2) on a grid of a: equivalent to q >= -1. The expanding
        branch of the bounce toy violates it just after the bounce, where H' > 0 (q < -1).
        """
        a = self._grid(which, a, 200, **kw)
        return _fl.second_law_holds(self.model(which, **kw), a)

    def curvature(self, which="lcdm", a=None, **kw):
        """Kretschmann scalar K = 12 [(a''/a)^2 + H^4] along a grid of a (numerical derivative of E^2)."""
        a = self._grid(which, a, 200, **kw)
        return _fl.flrw_history(self.model(which, **kw), a)["K"]

    def bounce_curvature(self, rho_c=10.0, h0=1.0):
        """The exact Kretschmann scalar at the bounce of the dust bounce toy: 27 (rho_c h0^2)^2."""
        return 27.0 * (rho_c * h0 ** 2) ** 2

    def singularity_type(self, which="lcdm", **kw):
        return "none" if which == "bounce" else "point"

    def horizon(self, which="lcdm", a=None, **kw):
        """The apparent-horizon entropy S = pi / (G H^2) on a grid of a, exact in E^2."""
        a = self._grid(which, a, 50, **kw)
        return _fl.horizon_entropy(self.model(which, **kw), a)

    def fit_to_data(self, **kw):
        raise NeedsFit("fitting a toy to supernova, BAO or CMB data is not implemented, and any comparison would be "
                       "of a toy we wrote ourselves.", missing=self.missing_for_observation())

    def exact_content(self):
        return [
            "LCDM, the toy, the Tsallis and Pade models as closed-form E^2(a); the toy reduces to LCDM exactly "
            "as a_t -> 0, and eps = 0 gives LCDM in the coupled-black-hole toy",
            "flat-FLRW Kretschmann scalar 12 [(a''/a)^2 + H^4]; de Sitter gives 24 H^4",
            "horizon entropy S = pi / (G H^2) and the second-law criterion dS/dt >= 0 <=> q >= -1",
            "the dust bounce E^2 = x (1 - x/rho_c): a^3 = a_b^3 + (9/4) Om t^2 exactly, K(0) = 27 (rho_c h0^2)^2, "
            "finite curvature through a vanishing-density-gradient crossing",
            "single black hole growth m ~ a^k and spin ~ a^(-2k), and the density exponent k - 3 (w = -1 at k = 3)",
        ]

    def declined(self):
        return [
            "that any of these describes the observed expansion: no fit, no data (NeedsFit)",
            "the sigmoid toy and the coupled-black-hole toy are our own constructions: not reproductions of the "
            "Fairfolio 'LCDM+S' project (equations unpublished) nor of Croker et al. / the DESI analysis",
            "the star formation history is the Madau-Dickinson form, quoted, not refit; the coupled-black-hole toy "
            "needs a conversion efficiency of about 3.7 times all stellar mass for Omega_DE = 0.7, so it does not "
            "by itself explain dark energy in this bookkeeping",
            "the bounce is a generic effective-Friedmann bounce, not the quantum-pressure mechanism of Gaztanaga",
        ]


# ---------------------------------------------------------------------------
#  Kinetic-dominated Kahler moduli
# ---------------------------------------------------------------------------

@register
class ModuliKasner(CosmologyTheory):
    """
    Bianchi I cosmology whose only matter is the Kahler moduli of a CICY rolling freely. The one entry here that
    takes a geometry: the triple intersection numbers fix the moduli-space metric, and with it the exponent budget
    sum p = 1, sum p^2 = 1 - |s|^2, 0 <= |s|^2 <= 2/3.
    """

    key = "moduli-kasner"

    def __init__(self, X=None, name=None):
        from ..pyCICY import CICY
        CosmologyTheory.__init__(self, name)
        if X is None:
            raise ValueError("ModuliKasner needs a CICY (or configuration matrix) for its intersection numbers")
        self.X = X if isinstance(X, CICY) else CICY(np.asarray(X, dtype=int).tolist())
        self.d = _mk.triple_numbers(self.X)

    def geometry(self):
        return "Bianchi I x Kahler moduli of %s (h11 = %d), classical large-volume K, axions fixed" % (
            self.X.M.tolist(), self.d.shape[0])

    def metric(self, t):
        return _mk.moduli_metric(self.d, t)

    def exponents(self, speed2=0, theta=0.0, **kw):
        """Kasner exponents for moduli kinetic energy |s|^2 (a point on the circle of exponents)."""
        return _mk.exponents_from_speed2(speed2, theta)

    def budget(self):
        return _mk.max_speed2()

    def curvature(self, p=None, speed2=0, theta=0.0, **kw):
        """K (t - t_s)^4 for the exponents p (default: those of `speed2`, `theta`); exact for rational p."""
        return _mk.kretschmann_coefficient(p if p is not None else self.exponents(speed2, theta))

    def singularity_type(self, speed2=0, theta=0.0, **kw):
        return _kn.singularity_type([float(x) for x in self.exponents(speed2, theta)], "singularity")

    def horizon(self, speed2=0, theta=0.0, **kw):
        from ..horizons import horizon_type
        return [horizon_type(x) for x in self.exponents(speed2, theta)]

    def geodesic(self, t0, s0, u_span=(0.0, 3.0), **kw):
        return _mk.moduli_geodesic(self.d, t0, s0, u_span, **kw)

    def exact_content(self):
        return [
            "the Kahler-moduli metric G_ij = -3 kappa_ij / V + (9/2) kappa_i kappa_j / V^2 (also in rational "
            "arithmetic), its third derivative, and the Christoffel symbols, checked against sympy",
            "the exponent budget: sum p = 1, sum p^2 = 1 - |s|^2, real exponents iff |s|^2 <= 2/3, the "
            "isotropic stiff fluid saturating it",
            "the dilation t -> lambda t is Killing with G(xi, xi) = 3/2, so rays are geodesics and "
            "lambda ~ (t - t_s)^sigma needs |sigma| <= 2/3",
            "the Kretschmann coefficient for any exponents, checked against a symbolic Riemann tensor",
        ]

    def declined(self):
        return [
            "geodesics for h11 > 1 are numerical (|s| conserved to integrator accuracy, and an independent "
            "Euler-Lagrange integration of the minisuperspace Lagrangian agrees)",
            "alpha' corrections, complex-structure and bundle moduli, axions, a potential, and the heterotic "
            "dilaton beyond the product-metric rule: the truncation is classical large-volume only",
            "whether kinetic domination is reached from any initial condition: that is moduli stabilisation",
            "any comparison with observation (NeedsFit)",
        ]
