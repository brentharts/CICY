r"""
pyCICY.theories.running -- from the exact spectrum to the gauge couplings.

Why this module exists
----------------------
The proton-to-electron mass ratio is the standard target for "can a
compactification predict anything". It is worth writing the chain out, because
doing so shows which links this package can close exactly, which it cannot
close at all, and --- the useful part --- that they are not the same links one
might guess.

    m_e  =  y_e v / sqrt(2)
    m_p  ~  Lambda_QCD  =  M_GUT exp( -2 pi / ( |b_3| alpha_GUT ) )

so

    m_p / m_e  ~  (M_GUT / v) exp( -2 pi / (|b_3| alpha_GUT) ) / y_e .

The proton mass is not a Yukawa coupling at all. It is a *dimensional
transmutation* scale: almost all of it is QCD binding energy, set by where the
strong coupling diverges, which depends on the unified coupling and on the
one-loop coefficient ``b_3``. And ``b_3`` is a function of the massless
spectrum -- which :mod:`pyCICY.breaking` computes **exactly**, from index
theory, with no metric anywhere.

So one factor in the ratio is exactly determined by the topology. The others
are not, and this module is explicit about which is which rather than
producing a number.

What is exact here
------------------
The one-loop coefficients. For a supersymmetric spectrum with ``n_g``
generations and ``n_H`` Higgs doublet pairs,

    b_1 = 2 n_g + (3/5) n_H          (GUT-normalised hypercharge)
    b_2 = -6 + 2 n_g + n_H
    b_3 = -9 + 2 n_g

reproducing the MSSM values ``(33/5, 1, -3)`` at ``n_g = 3``, ``n_H = 1``.
These are rational numbers computed from integers that came out of an index
theorem. :func:`beta_coefficients` computes them and
:func:`confines` reports the consequence that matters here: ``b_3 < 0`` is
asymptotic freedom, and

    n_g <= 4   =>  b_3 < 0,  QCD confines, there is a proton
    n_g >= 5   =>  b_3 >= 0, it does not, and there is no proton to weigh

a statement about the chiral spectrum, hence about the index of a bundle,
hence about the configuration matrix. It is the one place where the geometry
speaks directly to the proton mass.

What is not
-----------
Everything else in the ratio. :func:`mass_ratio_chain` returns the
factorisation with each factor labelled, and the labels are the point: three
of the four are unavailable, for three different reasons, and no amount of
work on the configuration matrix addresses any of them. See
:func:`pyCICY.theories.base.Theory.missing_for_physical`.
"""

import math
from fractions import Fraction as F

__all__ = [
    "beta_coefficients", "confines", "unification_scale", "lambda_qcd",
    "mass_ratio_chain", "sensitivity", "MSSM_BETAS",
]

#: The MSSM one-loop coefficients, for calibration.
MSSM_BETAS = (F(33, 5), F(1), F(-3))


def beta_coefficients(n_generations=3, n_higgs_pairs=1, extra=None):
    r"""
    One-loop beta function coefficients from the massless spectrum.

    Parameters
    ----------
    n_generations : int
        Net chiral generations. From :func:`pyCICY.breaking.chiral_spectrum`
        this is ``-ind(V)/|Gamma|``, an exact integer.
    n_higgs_pairs : int
        Number of Higgs doublet pairs. **Not** determined by an index: the
        Higgs is vector-like, so its multiplicity lives in exactly the sector
        an index cannot see. Supplied by the caller, and the default of one is
        an assumption rather than a computation.
    extra : dict, optional
        Additional vector-like matter, as ``{'3+3bar': n, '2+2bar': n}``.
        Heterotic models routinely carry some, and it changes the running.

    Returns
    -------
    ``(b_1, b_2, b_3)`` as exact ``Fraction``s.

    Notes
    -----
    Supersymmetric normalisation throughout, and ``b_1`` in the GUT convention
    where unification predicts ``g_1 = g_2 = g_3``. The formulas reproduce
    :data:`MSSM_BETAS` at the Standard Model content, which is the calibration.
    """
    ng = int(n_generations)
    nh = int(n_higgs_pairs)
    b1 = 2 * ng + F(3, 5) * nh
    b2 = -6 + 2 * ng + nh
    b3 = -9 + 2 * ng
    if extra:
        # a vector-like 3 + 3bar contributes 1 to b_3 and 2/5 to b_1;
        # a vector-like doublet pair contributes 1 to b_2 and 3/5 to b_1
        n3 = int(extra.get("3+3bar", 0))
        n2 = int(extra.get("2+2bar", 0))
        b3 = b3 + n3
        b1 = b1 + F(2, 5) * n3 + F(3, 5) * n2
        b2 = b2 + n2
    return b1, b2, b3


def confines(n_generations=3, extra=None):
    r"""
    Whether QCD is asymptotically free, hence whether a proton exists.

    ``b_3 = -9 + 2 n_g`` (plus vector-like colour), so the sign flips between
    four and five generations:

    ======  ======  =================================
    n_g     b_3     consequence
    ======  ======  =================================
    3       -3      confines; the observed case
    4       -1      confines, barely
    5       +1      no confinement, no proton
    ======  ======  =================================

    This is the sharpest statement the topology makes about hadron masses. The
    generation count comes from an index, exactly, so a compactification either
    permits a proton or does not, and the configuration matrix decides.
    """
    _, _, b3 = beta_coefficients(n_generations, 1, extra)
    return bool(b3 < 0)


def unification_scale(alpha_gut=1.0 / 25.0, m_string=5.0e17):
    """The scale at which the couplings are taken to unify.

    Returned as given: in the heterotic string this is tied to the string
    scale and hence to the volume of the compactification and the dilaton
    vacuum expectation value, neither of which this package determines. The
    default is a conventional value, not a prediction.
    """
    return {"alpha_gut": alpha_gut, "m_gut": m_string,
            "determined_here": False,
            "why": "the unified coupling is the dilaton vev and the scale is "
                   "set by the compactification volume; both are moduli, and "
                   "moduli stabilisation is not addressed here"}


def lambda_qcd(alpha_gut=1.0 / 25.0, m_gut=5.0e17, n_generations=3,
               n_higgs_pairs=1, extra=None):
    r"""
    The QCD scale by one-loop running, ``Lambda = M exp(-2 pi / (|b_3| alpha))``.

    The exponent is where the exactly-known ingredient enters: ``b_3`` comes
    from the spectrum and therefore from the index. Everything else in the
    formula is a modulus.

    Returns a dict with ``lambda_qcd``, the ``b_3`` used, and an explicit note
    that the inputs are assumptions. One loop and no thresholds; this is the
    structure of the dependence, not a precision calculation.
    """
    _, _, b3 = beta_coefficients(n_generations, n_higgs_pairs, extra)
    if b3 >= 0:
        raise ValueError(
            "b_3 = %s is not negative, so the strong coupling does not grow "
            "in the infrared, QCD does not confine, and there is no Lambda "
            "and no proton. With %d generations that is the prediction."
            % (b3, n_generations))
    lam = m_gut * math.exp(-2 * math.pi / (abs(float(b3)) * alpha_gut))
    return {"lambda_qcd": lam, "b3": b3, "alpha_gut": alpha_gut,
            "m_gut": m_gut,
            "exact_part": "b_3, from the chiral spectrum",
            "assumed_part": "alpha_gut and m_gut, both moduli"}


def mass_ratio_chain(n_generations=3, n_higgs_pairs=1, extra=None):
    r"""
    The proton-to-electron mass ratio, factor by factor, with each labelled.

    Returns a list of records, one per factor of

        m_p / m_e  ~  (M_GUT / v) * exp(-2 pi / (|b_3| alpha_GUT)) / y_e ,

    each carrying ``factor``, ``status`` (``'exact'``, ``'needs metric'``,
    ``'needs moduli stabilisation'`` or ``'not determined by topology'``) and
    a ``reason``.

    The function deliberately returns no number. Three of the four factors are
    unavailable, for three different reasons, and multiplying an exact one by
    three guesses would produce something that looked like a prediction. What
    it does provide is an audit: exactly which quantities would have to be
    supplied, and by what kind of calculation, for the ratio to follow.
    """
    b1, b2, b3 = beta_coefficients(n_generations, n_higgs_pairs, extra)
    out = [
        {"factor": "exp(-2 pi / (|b_3| alpha_GUT)), the QCD scale",
         "status": "exact in b_3, assumed in alpha_GUT",
         "value": "b_3 = %s" % b3,
         "reason": "b_3 = -9 + 2 n_g follows from the chiral spectrum, which "
                   "is an index and therefore exact. alpha_GUT is the dilaton "
                   "vev."},
        {"factor": "M_GUT, the unification scale",
         "status": "needs moduli stabilisation",
         "value": None,
         "reason": "set by the compactification volume, a Kahler modulus"},
        {"factor": "v, the electroweak scale",
         "status": "needs moduli stabilisation",
         "value": None,
         "reason": "requires the supersymmetry breaking mechanism and the "
                   "Higgs potential, neither addressed here"},
        {"factor": "y_e, the electron Yukawa coupling",
         "status": "needs metric",
         "value": None,
         "reason": "the holomorphic coupling is quasi-topological and its "
                   "vanishing pattern is exact (see theories.yukawa); its "
                   "value needs cohomology representatives, and the physical "
                   "coupling additionally needs the Ricci-flat metric for the "
                   "field normalisations"},
    ]
    return {"factors": out,
            "betas": {"b1": b1, "b2": b2, "b3": b3},
            "confines": bool(b3 < 0),
            "exact_factors": sum(1 for f in out if f["status"] == "exact"),
            "summary": "of %d factors, %d is exact only in part and %d are "
                       "unavailable; no number is returned"
                       % (len(out), 1, len(out) - 1)}


def sensitivity(alpha_gut=1.0 / 25.0, n_generations=3, n_higgs_pairs=1,
                extra=None):
    r"""
    How hard the prediction actually is, quantified.

    The QCD scale depends exponentially on the unified coupling,

        Lambda = M exp( -2 pi / ( |b_3| alpha_GUT ) ) ,

    so the logarithmic derivative is

        d ln Lambda / d ln alpha_GUT  =  2 pi / ( |b_3| alpha_GUT ) ,

    which at ``alpha_GUT = 1/25`` and ``b_3 = -3`` is about **52**. A one per
    cent error in the unified coupling moves the QCD scale, and hence the
    proton mass, by fifty per cent. Over the plausible range the scale spans

        alpha_GUT = 1/25  ->  Lambda ~ 9 x 10^-6 GeV
        alpha_GUT = 1/20  ->  Lambda ~ 0.3 GeV        (about right)
        alpha_GUT = 1/18  ->  Lambda ~ 21 GeV

    seven orders of magnitude across a thirty per cent change in the input.

    This is worth stating plainly because it inverts the naive picture of what
    stands between a compactification and a mass ratio. The exactly known
    ingredient -- ``b_3``, an integer from an index theorem -- sits in an
    exponent multiplied by a quantity nobody can compute. Knowing it exactly
    buys nothing until ``alpha_GUT`` is known to roughly a tenth of a per cent,
    which requires moduli stabilisation, not more topology.

    Returns a dict with the derivative and a small table of the dependence.
    """
    _, _, b3 = beta_coefficients(n_generations, n_higgs_pairs, extra)
    if b3 >= 0:
        raise ValueError("b_3 = %s >= 0: no confinement scale to be sensitive"
                         % b3)
    d = 2 * math.pi / (abs(float(b3)) * alpha_gut)
    table = []
    for inv in (25.0, 22.0, 20.0, 18.0):
        a = 1.0 / inv
        table.append((inv, 5.0e17 * math.exp(
            -2 * math.pi / (abs(float(b3)) * a))))
    return {"dlnLambda_dlnalpha": d,
            "b3": b3,
            "percent_per_percent": d,
            "table": table,
            "moral": "b_3 is exact and sits in an exponent multiplied by a "
                     "modulus; the exactness is amplified into irrelevance "
                     "until alpha_GUT is known to about 0.1 per cent"}
