r"""
pyCICY.theories.couplings -- gauge coupling unification, and the number 137.

About 137
---------
``alpha^{-1}(0) = 137.035999...`` is the electromagnetic coupling in the
Thomson limit, and it is a recurring target for claims that some structure
"explains" it. It is worth being clear at the outset what kind of number it is,
because the answer decides what a compactification could possibly say about it.

It is **not** a fundamental constant of a unified theory. It is a
*low-energy, scheme-dependent* value of a running coupling, related to the
short-distance coupling by

    alpha^{-1}(0)  =  alpha^{-1}(M_Z) * (1 - Delta alpha)^{-1} ... (see below)

where ``Delta alpha`` receives a hadronic contribution that is **not
calculable from first principles**: it is extracted from measured
``e^+e^- -> hadrons`` cross sections through a dispersion relation. So even a
theory that predicted the short-distance couplings exactly could not produce
137.036 without that measured input. Any derivation of 137 from pure geometry
is, at best, a derivation of something else.

What *is* predicted, and it is a real prediction, is the relation between the
couplings at the electroweak scale given unification. That prediction uses the
one-loop coefficients, which come from the massless spectrum, which
:mod:`pyCICY.breaking` computes exactly from an index. So the geometry does
have something sharp to say here --- just not about the digits of 137.

The prediction
--------------
One-loop running from a common value at ``M_G``,

    1/alpha_i(M_Z)  =  1/alpha_G  +  (b_i / 2 pi) ln(M_G / M_Z) ,

is three equations in two unknowns, so eliminating ``alpha_G`` and ``M_G``
leaves one relation among the three measured couplings:

    1/alpha_3  =  (b_3 - b_2)/(b_1 - b_2) * 1/alpha_1
                + (b_1 - b_3)/(b_1 - b_2) * 1/alpha_2 .

With ``alpha_1^{-1} = 59.0`` and ``alpha_2^{-1} = 29.57`` at ``M_Z``:

    MSSM spectrum, b = (33/5, 1, -3)      ->  alpha_3(M_Z) = 0.117
    Standard Model, b = (41/10, -19/6, -7) ->  alpha_3(M_Z) = 0.071
    measured                               ->  alpha_3(M_Z) = 0.118

The supersymmetric spectrum lands within one per cent at one loop; the
non-supersymmetric one is out by forty. That is a genuine discriminator, and
its only theoretical input is the set of ``b_i`` --- integers and simple
rationals determined by the chiral spectrum.

An important caveat, and it cuts against the obvious hope
---------------------------------------------------------
A complete generation contributes ``2 n_g`` to *every* ``b_i``, so it cancels
from all the differences ``b_i - b_j``, and the prediction above depends only
on those. Therefore

    **the unification prediction is independent of the number of
    generations.**

Four generations give exactly the same ``alpha_3(M_Z)`` as three. The quantity
this package computes most reliably --- the chiral generation count, an index
--- is precisely the one that drops out. What the prediction is sensitive to
is the *incomplete* multiplets: Higgs doublets, and any vector-like exotics.
Adding a second Higgs pair moves the prediction from ``0.117`` to ``0.786``,
and one vector-like colour triplet moves it to ``0.067``.

That is worth stating plainly because it relocates the difficulty. Those
multiplets are vector-like, so they live in exactly the sector an index cannot
see --- the same sector that
:func:`pyCICY.theories.breaking.chiral_spectrum` leaves undetermined and that
:mod:`pyCICY.theories.yukawa` cannot reach either. Gauge coupling unification
is sensitive to the part of the spectrum this package cannot compute, and
insensitive to the part it can.
"""

import math
from fractions import Fraction as F

__all__ = [
    "ALPHA1_INV_MZ", "ALPHA2_INV_MZ", "ALPHA3_MZ", "ALPHA_EM_INV_MZ",
    "ALPHA_EM_INV_0", "DELTA_ALPHA", "M_Z",
    "predict_alpha3", "sin2_theta_w", "alpha_em_inverse_mz",
    "alpha_em_inverse_zero", "unification_point", "fine_structure_chain",
]

#: Measured inputs at the Z pole, GUT normalisation for ``alpha_1``.
ALPHA1_INV_MZ = 59.0
ALPHA2_INV_MZ = 29.57
ALPHA3_MZ = 0.1181
ALPHA_EM_INV_MZ = 127.95
ALPHA_EM_INV_0 = 137.035999
DELTA_ALPHA = 0.0590
M_Z = 91.1876


def predict_alpha3(b1, b2, b3, a1_inv=ALPHA1_INV_MZ, a2_inv=ALPHA2_INV_MZ):
    r"""
    Predict ``alpha_3(M_Z)`` from unification and the other two couplings.

    Eliminating ``alpha_G`` and ``M_G`` from the three one-loop equations
    leaves one relation, so with two couplings measured the third is
    determined. The only theoretical input is the ``b_i``.

    Returns ``(alpha3, alpha3_inv)``.
    """
    b1, b2, b3 = F(b1), F(b2), F(b3)
    if b1 == b2:
        raise ValueError(
            "b_1 = b_2, so the two equations are degenerate and nothing is "
            "predicted; this happens only for contrived spectra")
    w1 = float((b3 - b2) / (b1 - b2))
    w2 = float((b1 - b3) / (b1 - b2))
    inv = w1 * a1_inv + w2 * a2_inv
    if inv <= 0:
        raise ValueError(
            "the prediction gives 1/alpha_3 = %.3f <= 0, which is not a "
            "coupling; this spectrum does not unify sensibly" % inv)
    return 1.0 / inv, inv


def sin2_theta_w(a2_inv=ALPHA2_INV_MZ, aem_inv=ALPHA_EM_INV_MZ):
    r"""
    The weak mixing angle, ``sin^2 theta_W = alpha_em / alpha_2``.

    At ``M_Z`` this is ``29.57 / 127.95 = 0.2312``, matching the measured
    value. At the unification scale the same quantity is ``3/8`` for any
    spectrum with a standard ``SU(5)`` embedding of hypercharge --- a group
    theory statement, independent of the geometry --- and the running between
    them is what the ``b_i`` control.
    """
    return a2_inv / aem_inv


def alpha_em_inverse_mz(a1_inv=ALPHA1_INV_MZ, a2_inv=ALPHA2_INV_MZ):
    r"""
    ``1/alpha_em = 1/alpha_2 + (5/3) / alpha_1``.

    From ``1/e^2 = 1/g^2 + 1/g'^2`` with ``alpha_1 = (5/3) alpha_Y``. Gives
    ``127.90`` against a measured ``127.95``.
    """
    return a2_inv + (5.0 / 3.0) * a1_inv


def alpha_em_inverse_zero(aem_inv_mz=ALPHA_EM_INV_MZ,
                          delta_alpha=DELTA_ALPHA):
    r"""
    Run the electromagnetic coupling down to the Thomson limit.

    ``alpha(M_Z) = alpha(0) / (1 - Delta alpha)``, so
    ``alpha^{-1}(0) = alpha^{-1}(M_Z) / (1 - Delta alpha)``.

    **``Delta alpha`` is measured, not computed.** It splits into a leptonic
    part, which is perturbatively calculable, and a hadronic part which is
    not: ``Delta alpha_had`` comes from ``e^+e^- -> hadrons`` data through a
    dispersion relation, because the relevant momenta are in the
    non-perturbative regime of QCD. This is the step at which any programme of
    deriving 137.036 from geometry necessarily stops, and it stops for reasons
    of strong-coupling dynamics rather than of insufficient topology.

    Note also the scheme dependence: the on-shell and MS-bar values of
    ``alpha^{-1}(M_Z)`` differ by about one unit, so "the" value at ``M_Z`` is
    not a single number until a scheme is fixed.
    """
    if not 0.0 <= delta_alpha < 1.0:
        raise ValueError("Delta alpha must lie in [0, 1)")
    return aem_inv_mz / (1.0 - delta_alpha)


def unification_point(b1, b2, b3, a1_inv=ALPHA1_INV_MZ,
                      a2_inv=ALPHA2_INV_MZ, mz=M_Z):
    r"""
    The scale and coupling at which ``alpha_1`` and ``alpha_2`` meet.

    From the first two equations,

        ln(M_G / M_Z) = 2 pi (1/alpha_1 - 1/alpha_2) / (b_1 - b_2) ,

    and ``alpha_G`` follows. Whether ``alpha_3`` also passes through that point
    is the content of :func:`predict_alpha3`; here it is an output rather than
    an assumption.
    """
    b1, b2 = F(b1), F(b2)
    if b1 == b2:
        raise ValueError("b_1 = b_2: the couplings never meet")
    ln_ratio = 2 * math.pi * (a1_inv - a2_inv) / float(b1 - b2)
    if ln_ratio <= 0:
        raise ValueError(
            "the couplings meet below M_Z (ln(M_G/M_Z) = %.2f), so this "
            "spectrum does not unify in the ultraviolet" % ln_ratio)
    m_gut = mz * math.exp(ln_ratio)
    a_gut_inv = a1_inv - float(b1) * ln_ratio / (2 * math.pi)
    return {"m_gut": m_gut, "alpha_gut": 1.0 / a_gut_inv,
            "alpha_gut_inv": a_gut_inv, "ln_ratio": ln_ratio}


def fine_structure_chain(n_generations=3, n_higgs_pairs=1, extra=None):
    r"""
    What stands between the geometry and 137.036, factor by factor.

    Returns a list of records with ``step``, ``status`` and ``reason``, in the
    manner of :func:`pyCICY.theories.running.mass_ratio_chain`, and no number
    for the final value.

    The summary is that exactly one step is exact, one is a genuine and
    successful prediction, and the last is not available even in principle
    from a compactification: the hadronic vacuum polarisation is measured
    input.
    """
    from . import running

    b1, b2, b3 = running.beta_coefficients(n_generations, n_higgs_pairs, extra)
    a3, a3_inv = predict_alpha3(b1, b2, b3)
    steps = [
        {"step": "the one-loop coefficients b_1, b_2, b_3",
         "status": "exact",
         "value": "(%s, %s, %s)" % (b1, b2, b3),
         "reason": "functions of the chiral spectrum, which is an index"},
        {"step": "alpha_3(M_Z) from unification and two measured couplings",
         "status": "predicted",
         "value": "%.4f (measured %.4f)" % (a3, ALPHA3_MZ),
         "reason": "one relation survives eliminating alpha_G and M_G; the "
                   "only theory input is the b_i"},
        {"step": "sin^2 theta_W at M_Z",
         "status": "predicted",
         "value": "%.4f" % sin2_theta_w(),
         "reason": "alpha_em/alpha_2, with 3/8 at unification by group theory"},
        {"step": "alpha_em^{-1}(M_Z)",
         "status": "predicted, and scheme dependent",
         "value": "%.2f (measured %.2f)" % (alpha_em_inverse_mz(),
                                            ALPHA_EM_INV_MZ),
         "reason": "1/alpha_2 + (5/3)/alpha_1; on-shell and MS-bar differ by "
                   "about one unit"},
        {"step": "alpha_em^{-1}(0) = 137.036",
         "status": "needs measured hadronic input",
         "value": None,
         "reason": "the running from M_Z to zero momentum needs "
                   "Delta alpha_had, which is extracted from e+e- -> hadrons "
                   "data by a dispersion relation because the momenta are "
                   "non-perturbative. No compactification supplies it."},
    ]
    return {"steps": steps,
            "betas": {"b1": b1, "b2": b2, "b3": b3},
            "alpha3_predicted": a3,
            "alpha3_measured": ALPHA3_MZ,
            "summary": "137.036 is a low-energy value of a running coupling, "
                       "not a fundamental constant; the geometry predicts the "
                       "short-distance relations and the last step needs "
                       "measured strong-interaction data"}
