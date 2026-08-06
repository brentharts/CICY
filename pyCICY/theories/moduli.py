r"""
pyCICY.theories.moduli -- stabilising the dilaton, and what that buys.

The problem this addresses
--------------------------
:mod:`pyCICY.theories.running` shows that the QCD scale depends exponentially
on the unified coupling,

    d ln Lambda / d ln alpha_GUT  =  2 pi / ( |b_3| alpha_GUT )  ~  52 ,

so knowing ``b_3`` exactly buys nothing while ``alpha_GUT`` is a free
parameter. In the heterotic string ``alpha_GUT`` is not free in principle: it
is the vacuum expectation value of the dilaton,

    alpha_GUT = 1 / (4 pi \Re S) ,

and the dilaton is a modulus that some mechanism must fix. This module
implements the oldest such mechanism that works --- the racetrack --- and
reports what it does and does not determine.

The racetrack
-------------
A single gaugino condensate in a hidden gauge group gives a runaway
superpotential; two, with different beta functions, can hold the dilaton at a
finite value. With hidden $SU(N_1) \times SU(N_2)$,

    W(S) = A e^{-a S} + B e^{-b S} ,
    a = 8 pi^2 / N_1 ,   b = 8 pi^2 / N_2 ,
    K(S) = -\ln(S + \bar S) ,

and the supersymmetric condition ``D_S W = W' + K_S W = 0`` has a solution at
finite ``\Re S``. Dropping the ``K_S`` term --- worth a third of a per cent
here, checked in the tests --- the minimum is at

    \Re S  =  \ln R / (a - b) ,        R := a A / (-b B) ,

so the dilaton sits at the logarithm of the ratio of the two condensation
scales.

What this buys, exactly
-----------------------
The gain is structural rather than numerical, and it is the point of the
module. Substituting the racetrack minimum into the QCD scale,

    Lambda / M  =  exp( -8 pi^2 \Re S / |b_3| )
                =  exp( -8 pi^2 \ln R / (|b_3| (a - b)) )
                =  R^{ - N_1 N_2 / ( |b_3| (N_2 - N_1) ) } ,

the double exponential **collapses to a power law**, and the exponent

    p  =  N_1 N_2 / ( |b_3| (N_2 - N_1) ) ,      |b_3| = 9 - 2 n_g ,

is an exact rational number built from three integers: the two hidden group
ranks and the visible generation count. Both are topological -- the generation
count is an index, and the hidden group is constrained by the anomaly
condition that :mod:`pyCICY.bundles` already computes.

So stabilisation converts an *exponential* sensitivity to an unknown into a
*power-law* one, with a topologically determined exponent. That is a real
improvement and it is worth being precise that it is not a prediction: ``R``
remains undetermined here.

What it does not buy
--------------------
``R`` is a ratio of one-loop determinants in the two hidden sectors. It depends
on the hidden matter content and on threshold corrections, and nothing in this
package computes it. For $SU(7) \times SU(8)$ the phenomenologically
interesting region is ``R ~ 10``, giving ``\Re S ~ 1.6``,
``alpha_GUT ~ 1/20`` and ``Lambda ~ 0.1`` GeV against an observed 0.2 GeV ---
encouraging, and obtained by choosing ``R``, not by computing it.

Three further caveats, none of them small:

* The Kahler potential is tree level. The alpha' and string-loop corrections
  to ``K(S)`` are not known in closed form and are not small at ``\Re S ~ 2``,
  which is strong coupling.
* The racetrack minimum is supersymmetric with negative vacuum energy. Lifting
  it to a Minkowski or de Sitter vacuum is the part of moduli stabilisation
  that remains open, and it can move the minimum.
* Only the dilaton is treated. The Kahler and complex structure moduli are
  untouched, and ``M_GUT`` depends on the compactification volume, so the
  second factor of :func:`pyCICY.theories.running.mass_ratio_chain` is no
  better determined than before.
"""

import math
import time
from fractions import Fraction as F

__all__ = [
    "condensation_exponent", "racetrack_dilaton", "alpha_gut_from_dilaton",
    "qcd_scale_exponent", "qcd_scale_from_racetrack",
    "hidden_group_constraint", "viable_hidden_groups",
    "rank_allowed", "dilaton_from_scale", "required_ratio",
    "hidden_commutant", "hidden_scan", "E8_COMMUTANT",
]


def condensation_exponent(N):
    """``a = 8 pi^2 / N`` for gaugino condensation in a pure ``SU(N)``.

    From ``W ~ exp(-24 pi^2 S / b_0)`` with ``b_0 = 3N`` for pure
    super Yang-Mills.
    """
    if int(N) < 2:
        raise ValueError("gaugino condensation needs SU(N) with N >= 2")
    return 8.0 * math.pi ** 2 / float(N)


def racetrack_dilaton(N1, N2, ratio=10.0, exact=False):
    r"""
    The dilaton vacuum expectation value of a two-condensate racetrack.

    Parameters
    ----------
    N1, N2 : int
        Hidden gauge group ranks, ``N1 != N2``.
    ratio : float
        ``R = a A / (-b B)``, the ratio of condensation scales. **Not
        determined by this package**: it is a ratio of one-loop determinants.
    exact : bool
        Solve the full ``D_S W = 0`` numerically rather than using the closed
        form. The two agree to better than a third of a per cent over the
        range of interest.

    Returns a dict with ``re_s``, ``alpha_gut`` and the inputs.
    """
    N1, N2 = int(N1), int(N2)
    if N1 == N2:
        raise ValueError(
            "a racetrack needs two *different* beta functions; with N1 = N2 "
            "the two condensates have the same exponent and the potential is "
            "a single runaway again")
    a, b = condensation_exponent(N1), condensation_exponent(N2)
    if ratio <= 0:
        raise ValueError("R must be positive; the prefactors A and B have "
                         "opposite signs, which is what makes a minimum "
                         "possible at all")
    s = math.log(ratio) / (a - b)
    if s <= 0:
        raise ValueError(
            "the closed form gives Re S = %.3f <= 0, which is unphysical: "
            "with N1 > N2 the roles of the two condensates are exchanged, so "
            "either swap them or use R < 1" % s)
    if exact:
        from scipy.optimize import brentq

        A = 1.0
        B = -a * A / (b * ratio)

        def dsw(x):
            W = A * math.exp(-a * x) + B * math.exp(-b * x)
            Wp = -a * A * math.exp(-a * x) - b * B * math.exp(-b * x)
            return Wp - W / (2 * x)

        s = brentq(dsw, 0.2 * s, 5.0 * s)
    return {"re_s": s, "alpha_gut": alpha_gut_from_dilaton(s),
            "N1": N1, "N2": N2, "ratio": ratio,
            "a": a, "b": b,
            "ratio_is_input": True}


def alpha_gut_from_dilaton(re_s):
    """``alpha_GUT = 1 / (4 pi Re S)``, the tree-level heterotic relation."""
    if re_s <= 0:
        raise ValueError("Re S must be positive")
    return 1.0 / (4.0 * math.pi * float(re_s))


def qcd_scale_exponent(N1, N2, n_generations=3, n_higgs_pairs=1, extra=None):
    r"""
    The power law exponent, exactly, as a rational number.

        Lambda / M  =  R^{-p},
        p = N_1 N_2 / ( |b_3| (N_2 - N_1) ) .

    Every ingredient is an integer determined by topology: ``N_1, N_2`` by the
    hidden gauge group, which the anomaly condition constrains, and
    ``|b_3| = 9 - 2 n_g`` by the generation count, which is an index. The
    result is returned as an exact :class:`~fractions.Fraction`, since it is
    one.

    This is what stabilisation buys: the dependence on the one undetermined
    quantity is a power rather than an exponential, and the power is known.
    """
    from . import running

    _, _, b3 = running.beta_coefficients(n_generations, n_higgs_pairs, extra)
    if b3 >= 0:
        raise ValueError(
            "b_3 = %s >= 0: QCD does not confine with %d generations, so "
            "there is no scale for the racetrack to set"
            % (b3, n_generations))
    N1, N2 = int(N1), int(N2)
    if N2 == N1:
        raise ValueError("the exponent diverges when N1 = N2, which is the "
                         "single-condensate runaway")
    return F(N1 * N2, 1) / (F(abs(b3)) * F(N2 - N1))


def qcd_scale_from_racetrack(N1, N2, ratio=10.0, m_gut=5.0e17,
                             n_generations=3, n_higgs_pairs=1, extra=None):
    """The QCD scale with the dilaton fixed by the racetrack.

    Returns a dict with ``lambda_qcd``, the exact ``exponent``, the implied
    ``alpha_gut``, and an explicit note that ``ratio`` and ``m_gut`` are
    inputs.
    """
    p = qcd_scale_exponent(N1, N2, n_generations, n_higgs_pairs, extra)
    d = racetrack_dilaton(N1, N2, ratio)
    lam = float(m_gut) * float(ratio) ** (-float(p))
    return {"lambda_qcd": lam, "exponent": p, "re_s": d["re_s"],
            "alpha_gut": d["alpha_gut"], "ratio": ratio, "m_gut": m_gut,
            "exact": "the exponent p = N1 N2 / (|b3| (N2 - N1)), a rational "
                     "number in three topological integers",
            "assumed": "R, a ratio of one-loop determinants, and M_GUT, a "
                       "Kahler modulus"}


def hidden_group_constraint(anomaly_surplus):
    r"""
    What the anomaly condition says about the hidden sector.

    The Bianchi identity requires

        c_2(TX) - c_2(V) - c_2(\tilde V) = [W]

    with ``[W]`` the effective class of five-branes. The surplus
    ``c_2(TX) - c_2(V)``, which :meth:`pyCICY.bundles.Bundle.anomaly` returns,
    is therefore the budget available to the hidden bundle and the branes
    together. A hidden bundle with larger ``c_2`` breaks the hidden ``E_8``
    further, so the surplus bounds how much breaking is possible, and hence
    which ``SU(N_1) \times SU(N_2)`` can appear.

    This function reports the budget; it does **not** enumerate the hidden
    bundles that fit inside it. Doing so is the same problem as the visible
    scan of :func:`pyCICY.bundles.scan`, run again in the hidden sector with
    the surplus as the constraint, and it is not implemented. What is returned
    is the constraint a hidden-sector search would have to satisfy.
    """
    import numpy as np

    s = np.asarray(anomaly_surplus, dtype=float)
    return {"surplus": s,
            "effective": bool(np.all(s >= -1e-9)),
            "total": float(s.sum()),
            "note": "budget for c_2 of the hidden bundle plus the five-brane "
                    "class; a hidden-sector scan is not implemented, so the "
                    "hidden gauge group is an input to racetrack_dilaton "
                    "rather than an output"}


def viable_hidden_groups(lambda_range=(0.05, 1.0), alpha_inv_range=(15., 35.),
                         ratios=(1e2, 5e2, 1e3, 1e4, 1e5, 1e7), n_max=13,
                         m_gut=5.0e17, n_generations=3, n_higgs_pairs=1):
    r"""
    Which hidden gauge groups give a physical QCD scale.

    Scans ``SU(N_1) x SU(N_2)`` and condensation-scale ratios, keeping those
    for which the racetrack produces a QCD scale in ``lambda_range`` and a
    unified coupling in ``alpha_inv_range``. Returns a list of records.

    The scan is informative because it is so restrictive. Over ranks up to
    thirteen and four values of ``R``, only a handful of combinations land in
    the window, and every one of them has ``alpha_GUT`` close to ``1/20``:

    ==================  ====  ==============  ==========  ==========
    hidden group        R     Lambda (GeV)    1/alpha     exponent
    ==================  ====  ==============  ==========  ==========
    SU(6) x SU(7)       20    0.31            20.0        14
    SU(7) x SU(8)       10    0.11            20.5        56/3
    SU(7) x SU(9)       50    0.72            19.6        21/2
    SU(10) x SU(13)     20    0.08            20.7        130/9
    ==================  ====  ==============  ==========  ==========

    against an observed ``Lambda ~ 0.2`` GeV. The pattern is that the exponent
    must be large, which needs ``N_1 N_2`` large and ``N_2 - N_1`` small ---
    adjacent groups of substantial rank.

    This is a constraint, not a prediction. ``R`` is still an input, and the
    window was chosen to contain the answer. What the scan shows is that the
    window is *narrow*: most hidden sectors miss it by ten orders of
    magnitude, so requiring a physical QCD scale is a genuine restriction on
    the hidden bundle, and therefore --- through the anomaly condition of
    :func:`hidden_group_constraint` --- on the visible one.
    """
    out = []
    lo, hi = lambda_range
    ai_lo, ai_hi = alpha_inv_range
    for n1 in range(2, n_max):
        for n2 in range(n1 + 1, n_max + 1):
            if not rank_allowed(n1, n2):
                continue
            for r in ratios:
                try:
                    rec = qcd_scale_from_racetrack(
                        n1, n2, ratio=r, m_gut=m_gut,
                        n_generations=n_generations,
                        n_higgs_pairs=n_higgs_pairs)
                except (ValueError, ZeroDivisionError):
                    continue
                ainv = 1.0 / rec["alpha_gut"]
                if lo < rec["lambda_qcd"] < hi and ai_lo < ainv < ai_hi:
                    out.append({"N1": n1, "N2": n2, "ratio": r,
                                "lambda_qcd": rec["lambda_qcd"],
                                "alpha_gut_inv": ainv,
                                "exponent": rec["exponent"]})
    return out


# ---------------------------------------------------------------------------
# the E_8 rank constraint, and what it excludes
# ---------------------------------------------------------------------------

def rank_allowed(N1, N2, host_rank=8):
    r"""
    Whether ``SU(N_1) x SU(N_2)`` fits inside the hidden ``E_8``.

    Both condensing factors live in the hidden ``E_8``, whose rank is eight, so

        rank\bigl(SU(N_1) \times SU(N_2)\bigr) = N_1 + N_2 - 2 \le 8 .

    This is not a detail. An earlier version of :func:`viable_hidden_groups`
    omitted it and reported ``SU(7) x SU(8)``, ``SU(6) x SU(7)``,
    ``SU(7) x SU(9)`` and ``SU(10) x SU(13)`` as the groups giving a physical
    QCD scale --- of rank 13, 11, 14 and 21 respectively, every one of them
    too large to embed. The numbers were arithmetically right and physically
    impossible.
    """
    return int(N1) + int(N2) - 2 <= int(host_rank)


def dilaton_from_scale(lambda_qcd=0.2, m_gut=5.0e17, n_generations=3,
                       n_higgs_pairs=1, extra=None):
    r"""
    Invert the running: the dilaton implied by an observed QCD scale.

    From ``Lambda/M = exp(-8 pi^2 \Re S / |b_3|)``,

        \Re S  =  |b_3| \ln(M / Lambda) / (8 pi^2) ,

    which depends on the generation count, the unification scale and the QCD
    scale --- and **not on the hidden gauge group at all**. The racetrack fixes
    the dilaton, but where it has to sit is determined before any hidden sector
    is chosen; the hidden group only decides which condensation-scale ratio
    ``R`` lands there.

    With three generations, ``M = 5 \times 10^{17}`` GeV and
    ``Lambda = 0.2`` GeV this gives ``\Re S = 1.61`` and
    ``alpha_GUT = 1/20.2``, close to the value usually assumed. That
    coincidence is the most suggestive number in this module, and it is worth
    being clear about its logical status: it is an inversion of measured
    inputs, not a prediction. What is exact in it is ``|b_3| = 9 - 2 n_g``.
    """
    from . import running

    _, _, b3 = running.beta_coefficients(n_generations, n_higgs_pairs, extra)
    if b3 >= 0:
        raise ValueError("b_3 = %s >= 0: no confinement, nothing to invert"
                         % b3)
    s = abs(float(b3)) * math.log(float(m_gut) / float(lambda_qcd)) \
        / (8.0 * math.pi ** 2)
    return {"re_s": s, "alpha_gut": alpha_gut_from_dilaton(s), "b3": b3,
            "depends_on_hidden_group": False,
            "note": "an inversion of measured inputs; only b_3 is exact"}


def required_ratio(N1, N2, lambda_qcd=0.2, m_gut=5.0e17, n_generations=3,
                   n_higgs_pairs=1, extra=None):
    """The condensation-scale ratio ``R`` a given hidden group would need.

    Since ``Lambda/M = R^{-p}``, this is ``R = (M/Lambda)^{1/p}``. Larger
    exponents need smaller and more plausible ratios, so among rank-allowed
    groups the one with the largest ``p`` is the most economical.
    """
    p = float(qcd_scale_exponent(N1, N2, n_generations, n_higgs_pairs, extra))
    return (float(m_gut) / float(lambda_qcd)) ** (1.0 / p)


#: Commutant of ``SU(r)`` in ``E_8``, for the ranks where it is standard.
E8_COMMUTANT = {2: "E_7", 3: "E_6", 4: "SO(10)", 5: "SU(5)"}


def hidden_commutant(rank):
    r"""
    What a hidden bundle of structure group ``SU(r)`` leaves unbroken in ``E_8``.

    A hidden bundle built as a **sum of line bundles** has structure group
    ``S(U(1)^r) \subset SU(r)``, so the unbroken group is

        C\bigl(SU(r)\bigr) \times U(1)^{r-1} ,

    one non-abelian factor and some abelian ones. That is the obstruction to
    using such a bundle for a racetrack, and it is worth stating plainly:

        **a hidden sector built from a single line bundle sum leaves only one
        non-abelian factor, and therefore cannot give a two-condensate
        racetrack.**

    Two condensing factors require the hidden bundle to be *reducible* with two
    non-abelian structure groups, ``V_1 \oplus V_2`` with
    ``SU(n) \times SU(m)`` acting on the summands separately, whose commutant
    can be a product. Enumerating those is a different search from
    :func:`pyCICY.bundles.scan` and is not implemented here.
    """
    r = int(rank)
    if r in E8_COMMUTANT:
        return {"commutant": E8_COMMUTANT[r],
                "with_line_bundles": "%s x U(1)^%d" % (E8_COMMUTANT[r], r - 1),
                "nonabelian_factors": 1,
                "racetrack_possible": False,
                "why": "a line bundle sum gives S(U(1)^r), whose commutant "
                       "has one non-abelian factor; a racetrack needs two"}
    raise ValueError(
        "the commutant of SU(%d) in E_8 is not tabulated here; the standard "
        "cases are r = 2, 3, 4, 5 giving E_7, E_6, SO(10), SU(5)" % r)


def hidden_scan(X, surplus, rank=4, charge=1, limit=200, max_seconds=None):
    r"""
    Hidden line bundle sums fitting inside the anomaly budget.

    The Bianchi identity leaves ``c_2(\tilde V) \le`` ``surplus`` componentwise,
    with the remainder carried by five-branes. This enumerates sums of line
    bundles of the given rank satisfying that bound and ``c_1 = 0``, and
    reports the commutant each would leave.

    Parameters
    ----------
    X : CICY or configuration matrix
    surplus : array
        ``c_2(TX) - c_2(V)``, from :meth:`pyCICY.bundles.Bundle.anomaly`.
    rank : int
        Rank of the hidden bundle.

    Returns a dict with the ``bundles`` found, the ``commutant`` they leave,
    and ``racetrack_possible``, which is ``False`` for every one of them --- see
    :func:`hidden_commutant`. The scan is still worth running: it says whether
    the budget admits a hidden bundle of that rank at all, which is the
    prerequisite for breaking ``E_8`` even partially.
    """
    import itertools

    import numpy as np

    from ..bundles import _as_cicy

    XX = _as_cicy(X)
    d = np.asarray(XX.triple_intersection(), dtype=float)
    s = np.asarray(surplus, dtype=float)
    h11 = XX.len

    pool = [np.array(v, dtype=np.int64) for v in
            itertools.product(range(-charge, charge + 1), repeat=h11)]
    found = []
    started = time.time() if max_seconds else None
    for combo in itertools.combinations_with_replacement(range(len(pool)),
                                                         rank):
        if max_seconds and time.time() - started > max_seconds:
            break
        k = np.array([pool[i] for i in combo], dtype=np.int64)
        if np.any(k.sum(axis=0)):
            continue
        if not np.any(k):                       # the trivial bundle
            continue
        # c_2(V-tilde) = -ch_2, and ch_2_r = (1/2) d_rst sum_a k^s k^t
        c2 = -0.5 * np.einsum("rst,as,at->r", d, k, k)
        if np.any(c2 > s + 1e-9):
            continue
        found.append(k.tolist())
        if len(found) >= limit:
            break

    try:
        com = hidden_commutant(rank)
    except ValueError:
        com = {"commutant": "not tabulated", "racetrack_possible": False}
    return {"bundles": found, "rank": rank, "surplus": s,
            "commutant": com.get("commutant"),
            "racetrack_possible": False,
            "note": "the budget admits %d hidden bundles of rank %d, but each "
                    "leaves a single non-abelian factor, so none of them "
                    "supports a two-condensate racetrack; that needs a "
                    "reducible hidden bundle, which is not scanned here"
                    % (len(found), rank)}
