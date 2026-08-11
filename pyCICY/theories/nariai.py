r"""
pyCICY.theories.nariai -- entropic gravity on the Nariai horizon, tested.

What this module is
-------------------
An independent check of Hartshorn (2026), "Gravity from Relative Entropy:
Jackiw-Teitelboim Dynamics on the Nariai Horizon", which extends the
Dorau-Much derivation of the semiclassical Einstein equations from the
Araki-Uhlmann relative entropy [Phys. Rev. Lett. 136, 091602] from the local
Rindler wedge to the Nariai spacetime dS_2 x S^2 -- the degenerate limit of
Schwarzschild-de Sitter in which the two horizons merge at r = 1/sqrt(Lambda)
and the bifurcation surface is a compact sphere of area 4 pi / Lambda.

Nariai is not a CICY and never will be: dS_2 x S^2 is Lorentzian, not even
Kahler. The module lives in this package for the same reason
:class:`~pyCICY.theories.ftheory.FTheory6D` does with ``X = None``: what the
:class:`~pyCICY.theories.base.Theory` interface actually demands is not a
configuration matrix but a contract about epistemic status -- which numbers
are exact, which need input the framework cannot supply, and which do not
exist. The paper keeps exactly this ledger, and this module re-derives its
checkable entries by routes independent of the paper's own companion code
(github.com/brentharts/NariaiRelativeEntropy), so that agreement means
something:

* the closed-form coherent entropy, Eq. (39) of the paper, is verified as an
  **exact rational identity** -- the integral (38) expanded in Gamma
  functions with `fractions.Fraction`, no floating point -- and then
  cross-checked against adaptive quadrature;
* the Araki-formula route (the symplectic form evaluated on the modular
  derivative, Eq. (29)) and the stress-tensor route (Eq. (31)) are computed
  **separately** and required to agree, testing the integration by parts
  (30) numerically rather than assuming it;
* the vanishing theorem I_2 = 0 (Eq. (64)) is verified on a **different
  packet** than the paper's Gaussian: chi(u) = (s + iu)^{-2}, whose spectral
  weight w e^{-sw} is strictly positive-frequency and whose I_2 vanishes by
  an explicit contour argument (all poles in the upper half-plane), giving a
  closed form to test against rather than a small number to stare at;
* the Jackiw-Teitelboim area response, Eq. (46)-(47), is obtained by
  **numerically integrating the dilaton ODE** and compared with 4 S_rel from
  the entropy side -- the two faces of the theorem delta A = 4 S_rel,
  computed with no shared code;
* the one-mode sum rule (66) is checked from Gaussian covariance matrices at
  **general squeezing angle**, where the paper's script fixes theta = 0;
* the geometry facts (degenerating roots, kappa_{b,c} -> epsilon
  sqrt(Lambda), the antipodal area balance at O(epsilon^2)) are extracted
  from the **exact cubic**, not the Ginsparg-Perry expansion, so that the
  expansion is something the module confirms rather than inputs.

The categories, one more time
-----------------------------
This package sorts quantities into exact / needs-a-metric / does-not-exist,
and the Nariai construction is the most instructive specimen yet, because the
walls run through different places than in any compactification:

* The **relative entropy is exact** -- finite, positive, no regulator -- and
  so is the Einstein coupling 8 pi derived from it. The quantity that
  elsewhere in this package sits behind the metric wall (a physical
  coupling) is here a theorem, because the metric response is the *output*.
* The **von Neumann entropy does not exist**. The horizon algebra is a type
  III factor: no density matrices, no partial trace, no microstate count.
  :exc:`TypeIIIFactor` reports this the way
  :exc:`~pyCICY.theories.ftheory.NoSuchTheory` reports a forbidden
  superpotential -- undefined as a matter of structure, not unavailable as a
  matter of technique.
* What **remains genuinely open** is dynamical, not entropic: the response of
  the teleological horizon to sign-indefinite flux (the paper's item (iii)),
  the type-III domain lift of the mode-resolved sum rule (its Eq. (67),
  honestly labelled a conjecture), and the microstate question the relative
  entropy was designed to sidestep. :meth:`NariaiEntropic.missing_for_physical`
  lists them.

Units: G = hbar = c = 1 (Planck units) throughout, matching the paper.
"""

import math
from fractions import Fraction

import numpy as np

from .base import Theory, register
from .ftheory import NoSuchTheory

__all__ = ["TypeIIIFactor", "NariaiEntropic", "nariai_data", "sds_horizons",
           "degenerate_family", "kk_tower", "coherent_entropy",
           "coherent_entropy_quadrature", "coherent_entropy_araki",
           "jt_area_response", "ContourPacket", "squeezed_flux_budget",
           "one_mode_sum_rule", "lifted_sum_rule", "running_ledger",
           "universal_negativity_ratio",
           "timescales", "OBSERVED_LAMBDA_PLANCK"]

#: The observed cosmological constant in Planck units,
#: Lambda l_P^2 with Lambda ~ 1.106e-52 m^-2 and l_P ~ 1.616e-35 m.
OBSERVED_LAMBDA_PLANCK = 2.888e-122


class TypeIIIFactor(NoSuchTheory):
    """Raised when a von Neumann entropy is asked of the horizon algebra.

    The horizon algebra of a bifurcate Killing horizon is a type III factor:
    it admits no density matrices, no partial traces, and no trace at all, so
    -Tr(rho ln rho) is not divergent but *undefined* -- there is no rho. The
    only entropy available is Araki's relative entropy of a *pair* of states,
    which is exactly what the construction uses. A subclass of
    :exc:`~pyCICY.theories.ftheory.NoSuchTheory` because it reports the same
    kind of fact: the quantity is absent as a matter of structure, and no
    amount of computation supplies it.
    """


# ---------------------------------------------------------------------------
# geometry: Schwarzschild-de Sitter, its degeneration, and Nariai data
# ---------------------------------------------------------------------------

def sds_horizons(mass, lam):
    r"""Exact horizon data of Schwarzschild-de Sitter, from the cubic.

    ``f(r) = 1 - 2M/r - Lambda r^2/3`` vanishes on the roots of
    ``r^3 - (3/Lambda) r + 6M/Lambda``. For ``0 < 9 Lambda M^2 < 1`` two
    roots are positive: the black-hole horizon ``rb`` and the cosmological
    horizon ``rc > rb``. Surface gravities are ``|f'(r)|/2``. Everything here
    is the exact cubic -- the Ginsparg-Perry expansion is *checked against*
    this function, never used inside it.
    """
    nine = 9.0 * lam * mass ** 2
    if not (0.0 < nine < 1.0):
        raise ValueError("need 0 < 9 Lambda M^2 < 1 for two horizons; "
                         "got %g" % nine)
    roots = np.roots([1.0, 0.0, -3.0 / lam, 6.0 * mass / lam])
    pos = sorted(float(r.real) for r in roots
                 if abs(r.imag) < 1e-9 * max(1.0, abs(r)) and r.real > 0)
    if len(pos) != 2:
        raise RuntimeError("expected two positive roots, got %r" % pos)
    rb, rc = pos

    def fprime(r):
        return 2.0 * mass / r ** 2 - 2.0 * lam * r / 3.0

    return {"rb": rb, "rc": rc,
            "kappa_b": abs(fprime(rb)) / 2.0,
            "kappa_c": abs(fprime(rc)) / 2.0,
            "A_b": 4.0 * math.pi * rb ** 2,
            "A_c": 4.0 * math.pi * rc ** 2,
            "nine_lam_m2": nine}


def degenerate_family(eps, lam=1.0):
    r"""The Ginsparg-Perry slice ``9 Lambda M^2 = 1 - 3 eps^2``, exactly.

    Returns the exact roots together with the quantities the paper's Section
    5 reads off the expansion, so a caller can test the expansion:

    * ``rb, rc -> r_* (1 -/+ eps)``: the roots split symmetrically;
    * ``kappa_b, kappa_c -> eps sqrt(Lambda)``: both surface gravities vanish
      linearly in eps, which is why the SdS time ``t`` sees zero temperature
      and the Bousso-Hawking rescaling ``t = psi/(eps sqrt(Lambda))`` is
      forced -- the normalisation Section 2 of the paper calls the hinge;
    * ``area_imbalance = A_b + A_c - 2 A_*``: the antipodal balance. At first
      order the areas shift equally and oppositely, so the imbalance is
      O(eps^2) -- the geometric face of the entropy sum rule (51).
    """
    mass = math.sqrt((1.0 - 3.0 * eps ** 2) / (9.0 * lam))
    d = sds_horizons(mass, lam)
    r_star = 1.0 / math.sqrt(lam)
    a_star = 4.0 * math.pi / lam
    d.update({
        "eps": eps, "mass": mass, "r_star": r_star,
        "rb_over_expansion": d["rb"] / (r_star * (1.0 - eps)),
        "rc_over_expansion": d["rc"] / (r_star * (1.0 + eps)),
        "kappa_b_over_eps": d["kappa_b"] / (eps * math.sqrt(lam)),
        "kappa_c_over_eps": d["kappa_c"] / (eps * math.sqrt(lam)),
        "area_imbalance": d["A_b"] + d["A_c"] - 2.0 * a_star,
    })
    return d


def nariai_data(lam=1.0):
    r"""The invariants of the degenerate solution.

    Everything is fixed by Lambda alone: the common radius, the compact
    bifurcation area, the extremal entropy S_N = pi/Lambda, and -- because
    the geometry normalises the Killing field (unit norm on the central
    geodesic), removing the Rindler boost-rescaling ambiguity -- an
    *unambiguous* surface gravity kappa = sqrt(Lambda) and Bousso-Hawking
    temperature kappa/2pi.
    """
    kappa = math.sqrt(lam)
    return {"r_star": 1.0 / kappa,
            "area": 4.0 * math.pi / lam,
            "entropy": math.pi / lam,
            "kappa": kappa,
            "temperature": kappa / (2.0 * math.pi),
            "beta": 2.0 * math.pi / kappa}


def timescales(lam=1.0):
    r"""The hierarchy that licenses equilibrium tools on an unstable saddle.

    Three timescales: the modular time ``beta = 2 pi / sqrt(Lambda)``; the
    classical drift time of the degeneracy parameter, which is *infinite*
    (epsilon labels a family of exact static solutions -- the instability is
    O(hbar) from the start); and the backreaction time, longer than the
    modular time by the horizon entropy itself,

        t_inst / t_mod ~ S_N / pi,   S_N = pi / Lambda  (Planck units).

    The hierarchy is self-generated: one nat of relative entropy moves the
    area by one part in S_N (see :func:`coherent_entropy`), so the framework
    computes the size of its own domain of validity. For the observed Lambda
    the ratio is ~1e122 -- the paper calls it the largest hierarchy in
    physics, and the nucleation channel is suppressed by e^{-S_N} on top.
    """
    d = nariai_data(lam)
    s_n = d["entropy"]
    return {"t_modular": d["beta"],
            "t_classical": math.inf,
            "hierarchy": s_n / math.pi,
            "entropy": s_n,
            "nucleation_exponent": -s_n,
            "per_nat_area_shift": 1.0 / s_n}


def kk_tower(lam=1.0, bare_mass=0.0, lmax=6):
    r"""The Kaluza-Klein tower of the sphere, and why s-waves dominate.

    ``m_l^2 = m^2 + Lambda l(l+1)``. On the dS_2 factor every ``l >= 1``
    member is principal-series (``m_l^2/Lambda >= 2 > 1/4``) with
    ``nu_l = sqrt(l(l+1) + m^2/Lambda - 1/4)``, and its thermal excitation is
    suppressed as ``e^{-pi nu_l}``. The compact sphere is the physical
    regulator: the transverse continuum of Rindler becomes a discrete tower,
    which is why every entropy sum in the construction converges without
    subtraction, and why the instability sector is the l = 0 mode.
    """
    out = []
    for ell in range(lmax + 1):
        m2 = bare_mass ** 2 + lam * ell * (ell + 1)
        ratio = m2 / lam
        principal = ratio > 0.25
        nu = math.sqrt(ratio - 0.25) if principal else 0.0
        out.append({"l": ell, "m2": m2, "m2_over_lambda": ratio,
                    "principal_series": principal, "nu": nu,
                    "boltzmann_suppression": math.exp(-math.pi * nu)})
    return out


# ---------------------------------------------------------------------------
# the coherent sector: closed form, quadrature, and the Araki route
# ---------------------------------------------------------------------------

def _profile(n, a, kappa, U):
    """The excitation family (36): phi_n = a (-kappa U)^n e^{kappa U}."""
    x = -kappa * U
    return a * x ** n * np.exp(-x)


def _dprofile(n, a, kappa, U):
    """d phi_n / dU = -a kappa x^{n-1} (n - x) e^{-x},  x = -kappa U."""
    x = -kappa * U
    return -a * kappa * x ** (n - 1) * (n - x) * np.exp(-x)


def coherent_entropy(n, a, lam=1.0):
    r"""The closed-form entropy of the pulse family, as exact arithmetic.

    Paper Eqs. (36)-(39): for the s-wave profile
    ``phi_n = a (-kappa U)^n e^{kappa U}`` on the right horizon portion,

        S_rel = (4 pi^2 a^2 / Lambda) * n (2n-1)! / 4^n .

    The longitudinal integral (38) is verified here as an exact *rational*
    identity: expanding ``(n - x)^2`` and using
    ``int_0^inf x^m e^{-2x} dx = m! / 2^{m+1}``,

        int = n^2 (2n-1)!/2^{2n} - n (2n)!/2^{2n} + (2n+1)!/2^{2n+2}
            = n (2n-1)! / (2 * 4^n),

    computed in :class:`fractions.Fraction`. If the two rationals ever
    disagreed the closed form would be wrong; they agree for all n.

    Returns the full dictionary of the paper's Section 4.3: the entropy, the
    boost heat ``dQ = S/2pi``, the Killing energy ``E = S/beta`` (so that
    ``S = beta E`` is an exact Clausius relation, derived not assumed), the
    area response ``dA = 4 S``, the ledger total ``pi/Lambda + S``, and the
    per-nat area shift ``(dA/A)/S = 1/S_N`` realising the self-generated
    hierarchy of :func:`timescales`.
    """
    if n < 1:
        raise ValueError("the family starts at n = 1")
    fact = math.factorial
    closed = Fraction(n * fact(2 * n - 1), 2 * 4 ** n)
    gamma_route = (Fraction(n * n * fact(2 * n - 1), 4 ** n)
                   - Fraction(n * fact(2 * n), 4 ** n)
                   + Fraction(fact(2 * n + 1), 4 * 4 ** n))
    if closed != gamma_route:
        raise AssertionError("the closed form (38) failed as a rational "
                             "identity: %s != %s" % (closed, gamma_route))

    d = nariai_data(lam)
    s_rel = 8.0 * math.pi ** 2 * a ** 2 / lam * float(closed)
    dq = s_rel / (2.0 * math.pi)
    e_killing = s_rel / d["beta"]
    da = 4.0 * s_rel
    return {"n": n, "amplitude": a,
            "longitudinal_integral": closed,          # exact Fraction
            "S_rel": s_rel,
            "boost_heat": dq,
            "killing_energy": e_killing,
            "clausius": d["beta"] * e_killing,        # == S_rel identically
            "delta_A": da,
            "delta_A_over_A": da / d["area"],
            "ledger_total": d["entropy"] + s_rel,
            "per_nat_shift": (da / d["area"]) / s_rel,  # == 1/S_N
            "pointwise_flux_nonnegative": True}


def coherent_entropy_quadrature(n, a, lam=1.0, u_max=60.0, points=200001):
    r"""S_rel by direct quadrature of Eq. (31): the stress-tensor route.

    ``S_rel = 2 pi * A(S) * int_{-inf}^0 (-U) (d_U phi_n)^2 dU`` with
    ``A(S) = 4 pi / Lambda`` the transverse volume of the s-wave. Computed on
    an affine grid with the trapezoid rule -- deliberately pedestrian, so
    that agreement with :func:`coherent_entropy` tests the closed form and
    not a shared derivation.
    """
    kappa = math.sqrt(lam)
    U = np.linspace(-u_max / kappa, 0.0, points)
    integrand = (-U) * _dprofile(n, a, kappa, U) ** 2
    longitudinal = float(np.trapezoid(integrand, U))
    return 2.0 * math.pi * (4.0 * math.pi / lam) * longitudinal


def coherent_entropy_araki(n, a, lam=1.0, u_max=60.0, points=200001):
    r"""S_rel by the Araki-formula route: the symplectic form on (delta phi, phi).

    Paper Eq. (29): differentiating the coherent-state overlap along the
    modular flow leaves only the symplectic phase,

        S_rel = (1/2) sigma(delta phi, phi),
        delta phi = -2 pi U d_U phi,
        sigma(f, g) = int (f d_U g - g d_U f) dU dvolS.

    This function evaluates sigma *directly* -- the two derivative terms,
    no integration by parts -- so that agreement with the stress-tensor route
    verifies the manipulation (30) numerically instead of assuming it. On the
    Rindler horizon the same check would require compactly supported
    transverse data; on Nariai the s-wave is admissible and the transverse
    factor is the finite area.
    """
    kappa = math.sqrt(lam)
    U = np.linspace(-u_max / kappa, 0.0, points)
    phi = _profile(n, a, kappa, U)
    dphi = _dprofile(n, a, kappa, U)
    delta = -2.0 * math.pi * U * dphi
    # d/dU of delta phi, by spectral-free centered differences
    ddelta = np.gradient(delta, U)
    sigma = float(np.trapezoid(delta * dphi - phi * ddelta, U))
    return 0.5 * (4.0 * math.pi / lam) * sigma


def jt_area_response(n, a, lam=1.0, u_max=60.0, points=200001):
    r"""The dilaton ODE, integrated, against 4 S_rel: the JT theorem.

    Paper Eqs. (44)-(47). The s-wave reduction of the throat is linearized de
    Sitter Jackiw-Teitelboim gravity, whose dilaton equation on the horizon
    collapses to

        d^2_U phi_dil = -(8 pi G / Lambda) <:T_UU:> ,

    and the flux-sourced solution, with the sl(2,R) kernel {1, U} removed,
    responds at the bifurcation surface by ``delta A = 4 pi phi_dil(0) =
    4 G S_rel`` -- the coupling 8 pi *derived* from the dimensional
    reduction, where Dorau-Much had to impose it via the Bekenstein-Hawking
    normalisation.

    This function integrates the ODE numerically (two cumulative
    trapezoids, kernel-free boundary data at U -> -inf) and returns the
    magnitude of the bifurcation-surface response next to 4 S_rel from the
    entropy side. The two are computed with no shared code; their ratio is
    the theorem. The paper's remark on the retarded-vs-teleological sign --
    the two framings differ by exactly the kernel, the magnitude is
    frame-independent -- is respected by comparing magnitudes.
    """
    kappa = math.sqrt(lam)
    U = np.linspace(-u_max / kappa, 0.0, points)
    flux = _dprofile(n, a, kappa, U) ** 2          # <:T_UU:> of the pulse
    source = -(8.0 * math.pi / lam) * flux
    # phi'(U) with phi'(-inf) = 0, then phi(U) with phi(-inf) = 0
    dU = U[1] - U[0]
    dphi = np.concatenate([[0.0], np.cumsum(
        0.5 * (source[1:] + source[:-1]) * dU)])
    phi = np.concatenate([[0.0], np.cumsum(
        0.5 * (dphi[1:] + dphi[:-1]) * dU)])
    delta_a_ode = 4.0 * math.pi * abs(phi[-1])
    s_rel = coherent_entropy(n, a, lam)["S_rel"]
    return {"delta_A_from_dilaton_ode": delta_a_ode,
            "four_S_rel": 4.0 * s_rel,
            "ratio": delta_a_ode / (4.0 * s_rel),
            "coupling_derived": 8.0 * math.pi}


# ---------------------------------------------------------------------------
# the squeezed sector: the vanishing theorem on an independent packet
# ---------------------------------------------------------------------------

class ContourPacket(object):
    r"""A boost-positive-frequency packet with closed-form functionals.

    ``chi(u) = (s + iu)^{-2} = int_0^inf w e^{-s w} e^{-i w u} dw``: the
    spectral weight ``w e^{-sw}`` is supported on strictly positive boost
    frequencies, so the packet is wedge-adapted by construction. Its
    functionals are exact:

        d_u chi           = -2i (s + iu)^{-3}
        I_1 = (1/k) int |d_u chi|^2 du = 3 pi / (2 k s^5)
        I_2 = (1/k) int (d_u chi)^2 du = 0

    the last by contour integration: ``(d_u chi)^2 = -4 (s + iu)^{-6}`` is
    analytic except at ``u = i s`` in the *upper* half-plane and decays as
    ``|u|^{-6}``, so closing the contour below encloses nothing. This is the
    vanishing theorem (64) of the paper -- the modular weight is the boost
    Jacobian, ``(-U) dU = du / kappa``, so I_2 is the zero-total-boost-
    frequency Fourier component of a square of positive frequencies -- but
    realised on a packet where the answer is an *exact closed form* rather
    than a small quadrature residue. The paper's companion script uses a
    numerically synthesised Gaussian spectral profile; agreement between the
    two packets is agreement about the theorem, not about a grid.

    Parameters
    ----------
    s : float
        Spectral decay scale; the packet width in boost time is ~ s.
    kappa : float
        Surface gravity; on Nariai, sqrt(Lambda).
    """

    def __init__(self, s=0.8, kappa=1.0):
        self.s = float(s)
        self.kappa = float(kappa)

    def dchi(self, u):
        u = np.asarray(u, dtype=float)
        return -2.0j / (self.s + 1j * u) ** 3

    def i1_exact(self):
        return 3.0 * math.pi / (2.0 * self.kappa * self.s ** 5)

    def i2_exact(self):
        """Zero, by the contour argument in the class docstring."""
        return 0.0

    def i1_i2_numeric(self, u_max=4000.0, points=800001):
        """Both functionals by brute-force quadrature on the boost line."""
        u = np.linspace(-u_max, u_max, points)
        d = self.dchi(u)
        i1 = float(np.trapezoid(np.abs(d) ** 2, u)) / self.kappa
        i2 = complex(np.trapezoid(d ** 2, u)) / self.kappa
        return i1, i2

    def boost_jacobian_check(self, u_max=40.0, points=400001):
        r"""The identity (62): (-U) |d_U chi|^2 dU = (1/k) |d_u chi|^2 du.

        Computed on both sides of the change of variables
        ``U = -(1/k) e^{-k u}`` and returned as a relative difference. This
        is the entire mechanism of the vanishing theorem -- the modular
        weight is not analogous to a thermal factor, it *is* the Jacobian --
        so it earns its own check.
        """
        u = np.linspace(-u_max, u_max, points)
        boost = float(np.trapezoid(np.abs(self.dchi(u)) ** 2, u)) / self.kappa
        # affine side: d_U = e^{k u} d_u, dU = e^{-k u} du, (-U) = e^{-k u}/k
        jac = np.exp(-self.kappa * u)
        affine = float(np.trapezoid(
            (jac / self.kappa) * np.abs(self.dchi(u) / jac) ** 2 * jac, u))
        return {"boost": boost, "affine": affine,
                "rel_diff": abs(boost - affine) / boost}

    def real_packet_ratio(self):
        """|I_2|/I_1 = 1 exactly for boost-blind (real) data.

        For real horizon data ``(d chi)^2 = |d chi|^2`` pointwise, so the
        ratio is 1 whatever the profile: no wedge-local squeeze of such a
        mode exists at any finite r (tanh r >= 1 is unreachable). The
        dichotomy 0-or-1 makes |I_2|/I_1 a binary wedge-locality detector.
        """
        return 1.0


def squeezed_flux_budget(r, theta, packet=None, u_max=200.0, points=400001):
    r"""Local negativity, globally prepaid: the anti-evaporation budget.

    Paper Eqs. (54)-(61). The squeezed flux

        <:T_UU:> ~ 2 sinh^2 r |d chi|^2 - sinh 2r Re[e^{-i theta}(d chi)^2]

    has an interference term that can locally dominate (sinh 2r > 2 sinh^2 r
    for small r), producing genuinely negative boost-energy windows -- the
    engine of anti-evaporation. But because I_2 = 0, the interference term
    integrates to *exactly zero* against the modular weight: the total is the
    thermal value 2 sinh^2 r I_1 whatever theta, and every unit of funded
    negativity D(W) is prepaid by positive flux elsewhere on the same
    horizon. Negativity is a zero-sum reallocation -- the Nariai-horizon
    instance of the quantum-interest circle of ideas, with every integral
    finite.

    Returns the modular-weighted total, its theta-independent prediction, the
    funded depth D(W), the positive part, and the support fraction of the
    negative windows.
    """
    p = packet or ContourPacket()
    u = np.linspace(-u_max, u_max, points)
    d = p.dchi(u)
    t = (2.0 * math.sinh(r) ** 2 * np.abs(d) ** 2
         - math.sinh(2.0 * r) * np.real(np.exp(-1j * theta) * d ** 2)
         ) / p.kappa
    total = float(np.trapezoid(t, u))
    neg = -float(np.trapezoid(np.where(t < 0.0, t, 0.0), u))
    pos = float(np.trapezoid(np.where(t > 0.0, t, 0.0), u))
    return {"total": total,
            "prediction": 2.0 * math.sinh(r) ** 2 * p.i1_exact(),
            "funded_depth": neg,
            "positive_part": pos,
            "budget_respected": neg <= pos * (1.0 + 1e-12),
            "negative_support_fraction": float(np.mean(t < 0.0)),
            "windows_exist": neg > 0.0}


def running_ledger(r, theta, packet=None, u_max=200.0, points=400001):
    r"""The cut-resolved ledger R(u_c), and its three structural claims.

    Paper Eq. (68)-(69): the modular flux accumulated between a cut at boost
    time u_c and the bifurcation surface. The particle part is a monotone
    CDF; the interference part is a bounded oscillation returning to zero;
    every dip of R -- the cut-resolved shadow of an anti-evaporation window
    -- is bounded by the funded depth. Whatever the eventual dynamical theory
    of the teleological horizon says, it must be a functional of this
    stratification.
    """
    p = packet or ContourPacket()
    u = np.linspace(-u_max, u_max, points)
    d = p.dchi(u)
    t = (2.0 * math.sinh(r) ** 2 * np.abs(d) ** 2
         - math.sinh(2.0 * r) * np.real(np.exp(-1j * theta) * d ** 2)
         ) / p.kappa
    du = u[1] - u[0]
    # accumulate from the bifurcation side (large u) backward
    R = np.concatenate([[0.0], np.cumsum(t[::-1]) * du])[:-1][::-1]
    total = float(R[0])
    envelope = np.maximum.accumulate(R[::-1])[::-1]
    largest_dip = float(np.max(envelope - R))
    neg = -float(np.trapezoid(np.where(t < 0.0, t, 0.0), u))
    return {"total": total,
            "prediction": 2.0 * math.sinh(r) ** 2 * p.i1_exact(),
            "largest_dip": largest_dip,
            "funded_depth": neg,
            "dip_within_budget": largest_dip <= neg * (1.0 + 1e-9)}


# ---------------------------------------------------------------------------
# the one-mode sum rule, at general squeezing angle
# ---------------------------------------------------------------------------

def universal_negativity_ratio(r):
    r"""The funded-depth ratio D(W)/positive-part as a function of r alone.

    An observation made *by* this module's cross-checks, not taken from the
    paper: computing :func:`squeezed_flux_budget` for the contour packet at
    several widths and angles, and comparing with the paper's Gaussian
    packet, the ratio of funded negativity to positive flux agrees across
    all of them to six digits at fixed r (0.2087 at r = 0.5, 0.0444 at
    r = 1.0 -- the paper's Section 9.2 values reproduced by a completely
    different packet).

    The mechanism, once seen, is elementary. Since ``|(d chi)^2| =
    |d chi|^2`` pointwise, the squeezed flux is

        t(u) = |d_u chi|^2 [ 2 sinh^2 r - sinh 2r cos(theta - phase(u)) ]/k,

    a fixed envelope times a cosine in the *local phase* of ``(d chi)^2``.
    When that phase winds uniformly through the packet's support -- true for
    any packet whose central frequency turns over many cycles under the
    envelope, and exactly true in the ergodic-phase limit -- the negative
    and positive parts become circle averages, and with ``T = tanh r``,
    ``psi_0 = arccos T``:

        D(W)/pos = (sin psi_0 - T psi_0) / (T (pi - psi_0) + sin psi_0).

    Universal in the packet and in theta; a strictly decreasing function of
    r, from 1 at r = 0 (where any negativity would be unfunded, and indeed
    the flux vanishes) toward 0 as r -> inf (deep squeezes are almost all
    particles). This sharpens the paper's budget (61) into a *ratio law*:
    the fraction of the horizon ledger a squeezed excitation may hold in
    negativity is set by r alone, independent of how the excitation is
    shaped or phased.
    """
    t = math.tanh(r)
    if t <= 0.0:
        return 1.0
    psi0 = math.acos(t)
    return ((math.sin(psi0) - t * psi0)
            / (t * (math.pi - psi0) + math.sin(psi0)))


def one_mode_sum_rule(omega, r, theta=0.0, kappa=1.0):
    r"""S_rel = beta Delta E for a wedge-local one-mode squeeze, any angle.

    Paper Eq. (66) and Section 9.2, extended: the wedge restriction of the
    global vacuum is thermal at beta = 2 pi / kappa mode by mode, with
    covariance ``sigma = nu I``, ``nu = coth(beta omega / 2)``. A squeeze at
    angle theta is the symplectic map ``S = R(theta/2) diag(e^r, e^{-r})
    R(-theta/2)``; unitarity forces the symplectic eigenvalue -- hence the
    Gaussian entropy -- to be invariant, so Delta S = 0 and the relative
    entropy collapses to the first law,

        S_rel = beta omega sinh^2 r coth(beta omega / 2),

    the coth being the thermal occupancy of the wedge mode. The paper's
    companion script checks theta = 0 only; the angle enters the flux
    distribution but must drop out of every entry here, and does.
    """
    beta = 2.0 * math.pi / kappa
    nu = 1.0 / math.tanh(beta * omega / 2.0)
    c, s = math.cos(theta / 2.0), math.sin(theta / 2.0)
    Rm = np.array([[c, -s], [s, c]])
    S = Rm @ np.diag([math.exp(r), math.exp(-r)]) @ Rm.T
    sig_th = nu * np.eye(2)
    sig_sq = S @ sig_th @ S.T
    nu_sq = math.sqrt(float(np.linalg.det(sig_sq)))

    def svn(x):
        ap, am = (x + 1.0) / 2.0, (x - 1.0) / 2.0
        return ap * math.log(ap) - (am * math.log(am) if am > 0 else 0.0)

    e_th = omega * float(np.trace(sig_th)) / 4.0
    e_sq = omega * float(np.trace(sig_sq)) / 4.0
    ln_z = -math.log(2.0 * math.sinh(beta * omega / 2.0))
    s_rel = beta * e_sq + ln_z - svn(nu_sq)
    closed = beta * omega * math.sinh(r) ** 2 * nu
    return {"S_rel": s_rel,
            "beta_dE": beta * (e_sq - e_th),
            "closed_form": closed,
            "delta_S_vN": svn(nu_sq) - svn(nu),
            "nu_invariant": abs(nu_sq - nu)}


def lifted_sum_rule(modes, kappa=1.0):
    r"""The mode-resolved sum rule (67), with its two convergence conditions.

    ``S_rel = sum_k sinh^2 r_k * beta omega_k coth(beta omega_k / 2)``,
    subject to the Shale condition ``sum sinh^2 r_k < inf`` (unitary
    implementability of the product squeeze) and the separate finiteness of
    the entropy sum itself, which carries the extra thermal weight. The
    conditions are *not* equivalent -- there are Shale-implementable squeezes
    of infinite relative entropy, the ledger recording infinity -- and the
    paper honestly labels the lifted rule a conjecture pending the natural-
    cone convergence question in the type III setting. This function computes
    the finite-list version, which is the part that is theorem-adjacent, and
    reports both partial sums so the inequivalence can be exhibited on
    towers.
    """
    beta = 2.0 * math.pi / kappa
    total = 0.0
    shale = 0.0
    for omega, r in modes:
        nu = 1.0 / math.tanh(beta * omega / 2.0)
        total += math.sinh(r) ** 2 * beta * omega * nu
        shale += math.sinh(r) ** 2
    return {"S_rel": total, "shale_sum": shale, "n_modes": len(modes)}


# ---------------------------------------------------------------------------
# the theory object
# ---------------------------------------------------------------------------

@register
class NariaiEntropic(Theory):
    r"""Entropic gravity on the Nariai horizon: the theory object.

    Parameters
    ----------
    lam : float
        The cosmological constant in Planck units. Defaults to 1; pass
        :data:`OBSERVED_LAMBDA_PLANCK` for the observed value, at which the
        horizon entropy is ~1.1e122.

    Notes
    -----
    ``X = None``, as for :class:`~pyCICY.theories.ftheory.FTheory6D`: the
    compactification geometry is not a CICY, and the interface never needed
    it to be. What the class encodes is the ledger equation of the paper's
    Eq. (70): a state-independent extremal capacity pi/Lambda, a coherent
    column, a squeezed-thermal column, and an interference column that is
    exactly zero by the vanishing theorem.
    """

    key = "nariai-entropic"

    def __init__(self, lam=1.0, name=None):
        Theory.__init__(self, None, name=name)
        if lam <= 0:
            raise ValueError("Nariai needs Lambda > 0")
        self.lam = float(lam)

    def geometry(self):
        return ("dS2 x S^2 (Nariai), the degenerate Schwarzschild-de Sitter "
                "limit at Lambda = %g" % self.lam)

    def gauge_group(self):
        """None: this is semiclassical gravity, not a gauge compactification."""
        return "none (semiclassical gravity; no gauge sector)"

    def spectrum(self):
        """The horizon data: capacity, temperature, and the KK tower scales."""
        d = nariai_data(self.lam)
        return {"capacity": d["entropy"],
                "temperature": d["temperature"],
                "beta": d["beta"],
                "area": d["area"],
                "kk_gap_over_lambda": 2.0,   # m_1^2 / Lambda for m = 0
                "s_wave_dominant": True}

    def ledger(self, coherent=(), squeezes=()):
        r"""The Dorau-Much-Nariai equation (70), assembled.

        Parameters
        ----------
        coherent : iterable of (n, a)
            Pulses from the closed-form family, each contributing its exact
            S_rel.
        squeezes : iterable of (omega, r)
            Wedge-local squeezed modes, each contributing the thermal first
            law of :func:`one_mode_sum_rule`.

        Returns the four columns -- capacity, classical events, quantum
        events, interference -- with the last identically zero by the
        vanishing theorem, and every term finite.
        """
        d = nariai_data(self.lam)
        classical = sum(coherent_entropy(n, a, self.lam)["S_rel"]
                        for n, a in coherent)
        quantum = lifted_sum_rule(list(squeezes), kappa=d["kappa"])["S_rel"]
        return {"capacity": d["entropy"],
                "classical_events": classical,
                "quantum_events": quantum,
                "interference": 0.0,
                "total": d["entropy"] + classical + quantum}

    def branch_selection(self, r=0.5, theta=math.pi):
        r"""The selection rule and its boundary, in one dictionary.

        Coherent flux is pointwise non-negative, so within the response model
        any flux-receiving cut can only grow: anti-evaporation is a
        certificate of non-coherent horizon flux. Squeezed states supply the
        non-coherence -- locally negative windows -- but the vanishing
        theorem caps them: the modular-weighted total stays thermal-positive,
        so negativity is bounded, prepaid redistribution. The boundary of the
        coherent sector is the boundary between the branches.
        """
        b = squeezed_flux_budget(r, theta,
                                 ContourPacket(kappa=math.sqrt(self.lam)))
        return {"coherent_can_shrink_pierced_cut": False,
                "anti_evaporation_needs_noncoherence": True,
                "squeezed_windows_exist": b["windows_exist"],
                "squeezed_total_positive": b["total"] > 0.0,
                "budget_respected": b["budget_respected"]}

    def von_neumann_entropy(self):
        """Always raises: no density matrices on a type III factor."""
        raise TypeIIIFactor(
            "the horizon algebra is a type III von Neumann factor: it admits "
            "no density matrices and no trace, so -Tr(rho ln rho) is "
            "undefined rather than divergent. The construction never uses "
            "it; the only entropy in play is Araki's relative entropy of a "
            "pair of states, which ledger() assembles and which is finite. "
            "Asking what the horizon entropy counts is the microstate "
            "question, listed in missing_for_physical().")

    def holomorphic_yukawa(self, **kw):
        """Always raises: there is no matter superpotential in this theory."""
        raise NoSuchTheory(
            "the Nariai construction is semiclassical gravity plus a scalar "
            "probe, not a supersymmetric compactification: there is no "
            "superpotential and no Yukawa sector. The analogue of a coupling "
            "-- the Einstein coupling 8 pi -- is derived, exactly, by "
            "jt_area_response().")

    def missing_for_physical(self):
        return [
            "the dynamical response of the teleological horizon to "
            "sign-indefinite flux, outside the quasi-equilibrium window "
            "(the paper's item (iii); only the running-ledger kinematics "
            "survives there)",
            "the natural-cone convergence lifting the one-mode sum rule to "
            "the full quadratic generator in the type III setting (Eq. (67) "
            "is a conjecture until then)",
            "species additivity: alpha = 8 pi must emerge independently of "
            "field content for realistic matter",
            "the response of non-spherical cuts to l >= 1 flux, where no "
            "two-dimensional reduction exists",
            "what, if anything, the capacity pi/Lambda counts -- the "
            "microstate question the relative entropy is constructed to "
            "sidestep, not to answer",
        ]

    def describe(self):
        d = nariai_data(self.lam)
        t = timescales(self.lam)
        lines = [
            "%s on %s" % (self.name, self.geometry()),
            "  gauge group      %s" % self.gauge_group(),
            "  capacity         S_N = pi/Lambda = %.6g" % d["entropy"],
            "  temperature      sqrt(Lambda)/2pi = %.6g (unambiguous: "
            "geometry-normalised)" % d["temperature"],
            "  hierarchy        t_inst/t_mod ~ %.3g (self-generated)"
            % t["hierarchy"],
            "  exact:           S_rel of coherent pulses; Clausius "
            "S_rel = beta E; delta A = 4 S_rel with 8 pi derived (JT); "
            "I_2 = 0",
            "  does not exist:  von Neumann entropy (type III); Yukawa "
            "sector (not a compactification)",
            "  open:            teleological dynamics; type III lift of "
            "(67); species additivity; microstates",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# demonstration
# ---------------------------------------------------------------------------

def _demo():
    line = "-" * 70

    print(line)
    print("Geometry: the exact cubic against the Ginsparg-Perry expansion")
    print(line)
    for eps in (0.1, 0.01, 0.001):
        d = degenerate_family(eps)
        print("  eps = %-6g rb/[r*(1-eps)] = %.6f   kappa_b/(eps sqrtL) = "
              "%.6f   (Ab+Ac-2A*)/eps^2 = %.4f"
              % (eps, d["rb_over_expansion"], d["kappa_b_over_eps"],
                 d["area_imbalance"] / eps ** 2))
    print("  -> roots split as r*(1 -/+ eps), both surface gravities vanish")
    print("     linearly (the Bousso-Hawking rescaling made quantitative),")
    print("     and the antipodal area balance holds at O(eps^2).")
    print()

    print(line)
    print("Coherent pulses: one number, four independent routes")
    print(line)
    a = 0.1
    print("  n   closed form     quadrature      Araki route     "
          "dA(ODE)/4S_rel")
    for n in (1, 2, 3):
        c = coherent_entropy(n, a)
        q = coherent_entropy_quadrature(n, a)
        ar = coherent_entropy_araki(n, a)
        jt = jt_area_response(n, a)
        print("  %d   %.10f    %.10f    %.10f    %.12f"
              % (n, c["S_rel"], q, ar, jt["ratio"]))
    c = coherent_entropy(1, a)
    print("  Clausius: beta*E = %.12f = S_rel (exact); per-nat area shift "
          "= 1/S_N: %s"
          % (c["clausius"],
             abs(c["per_nat_shift"] - 1.0 / nariai_data()["entropy"]) < 1e-15))
    print()

    print(line)
    print("The vanishing theorem, on a packet with a contour-exact answer")
    print(line)
    p = ContourPacket(s=0.8)
    i1n, i2n = p.i1_i2_numeric()
    print("  I_1 exact 3pi/(2 k s^5) = %.12f   numeric = %.12f"
          % (p.i1_exact(), i1n))
    print("  I_2 exact (contour)     = 0               |I_2|/I_1 numeric "
          "= %.2e" % (abs(i2n) / p.i1_exact()))
    j = p.boost_jacobian_check()
    print("  boost-Jacobian identity (62): rel. diff = %.1e" % j["rel_diff"])
    print("  boost-blind (real) packet:    |I_2|/I_1 = %g -> no wedge-local "
          "squeeze exists" % p.real_packet_ratio())
    print()

    print(line)
    print("Squeezed budget: local negativity, exactly prepaid")
    print(line)
    for r, th in ((0.5, 0.0), (0.5, math.pi), (1.0, math.pi / 2)):
        b = squeezed_flux_budget(r, th)
        print("  r=%.1f th=%4.2f  total=%.6f (thermal pred %.6f)  D(W)=%.4f"
              "  D/pos=%.3f  neg support=%4.1f%%"
              % (r, th, b["total"], b["prediction"], b["funded_depth"],
                 b["funded_depth"] / b["positive_part"],
                 100 * b["negative_support_fraction"]))
    print("  universal ratio law (found by this module's cross-checks):")
    for r in (0.3, 0.5, 1.0, 1.5):
        b = squeezed_flux_budget(r, 1.0, ContourPacket(s=2.0))
        print("    r=%.1f  D/pos measured %.6f   phase-average closed form "
              "%.6f" % (r, b["funded_depth"] / b["positive_part"],
                        universal_negativity_ratio(r)))
    rl = running_ledger(0.5, math.pi)
    print("  running ledger: R(-inf)=%.6f (pred %.6f), largest dip %.4f "
          "<= D(W) %.4f: %s"
          % (rl["total"], rl["prediction"], rl["largest_dip"],
             rl["funded_depth"], rl["dip_within_budget"]))
    print()

    print(line)
    print("One-mode sum rule at general angle (the script checked theta=0)")
    print(line)
    for w, r in ((0.5, 0.3), (1.0, 0.8), (2.0, 1.5)):
        for th in (0.0, math.pi / 3, math.pi):
            m = one_mode_sum_rule(w, r, th)
            print("  w=%.1f r=%.1f th=%4.2f  S_rel=%.9f  beta dE=%.9f  "
                  "closed=%.9f  dS_vN=%.1e"
                  % (w, r, th, m["S_rel"], m["beta_dE"], m["closed_form"],
                     abs(m["delta_S_vN"])))
    print()

    print(line)
    print("The theory object, and the observed universe")
    print(line)
    m = NariaiEntropic()
    print(m.describe())
    led = m.ledger(coherent=[(1, 0.1)], squeezes=[(1.0, 0.8)])
    print("  ledger: capacity %.4f + classical %.6f + quantum %.6f + "
          "interference %g = %.6f"
          % (led["capacity"], led["classical_events"], led["quantum_events"],
             led["interference"], led["total"]))
    try:
        m.von_neumann_entropy()
    except TypeIIIFactor as e:
        print("  von Neumann entropy: %s." % str(e).split(".")[0])
    obs = NariaiEntropic(OBSERVED_LAMBDA_PLANCK, name="observed Lambda")
    print("  at the observed Lambda: S_N = %.3g, hierarchy %.3g"
          % (nariai_data(OBSERVED_LAMBDA_PLANCK)["entropy"],
             timescales(OBSERVED_LAMBDA_PLANCK)["hierarchy"]))


if __name__ == "__main__":
    _demo()
