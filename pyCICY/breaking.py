r"""
pyCICY.breaking -- Wilson lines, and what they do to a heterotic SU(5) model.

Where this sits
---------------
:mod:`pyCICY.bundles` produces candidate models: rank-5 sums of line bundles
on a favourable CICY, poly-stable, anomaly-free, with ind(V) = -3|Gamma|. What
it reports is an SU(5) grand unified spectrum *upstairs*, on the covering
space, together with the arithmetic statement that the generation count is
divisible by |Gamma|. It never constructs the quotient.

That leaves a gap. "Three generations after quotienting by a freely acting
group of order two" is, at that stage, a division, not a model. The physics
that turns SU(5) into SU(3) x SU(2) x U(1) is a Wilson line on X/Gamma, and
whether the surviving spectrum is the Standard Model depends on how Gamma acts
on the cohomology -- which the configuration matrix does not determine. This
module supplies the part that is group theory and is honest about the part
that is not.

What is determined, and computed here
-------------------------------------
*The branching.* Under SU(5) -> SU(3) x SU(2) x U(1)_Y,

    10   ->  (3, 2)_{1/6}  +  (3bar, 1)_{-2/3}  +  (1, 1)_{1}
    5bar ->  (3bar, 1)_{1/3}  +  (1, 2)_{-1/2}

which is one Standard Model generation, in the left-handed conjugate
convention. :func:`branching` computes this from the hypercharge generator
rather than tabulating it, and :func:`verify_against_flavor` checks the
resulting hypercharges against :data:`pyCICY.flavor.SM_HYPERCHARGES`, which
were entered independently from a different paper for a different purpose.
The two must agree up to conjugation, and do.

*Which Wilson lines break to the Standard Model.* A Wilson line in the
hypercharge direction is W = diag(a, a, a, b, b) with a^3 b^2 = 1, of order
dividing |Gamma|. Its commutant in SU(5) is S(U(3) x U(2)) = SU(3) x SU(2) x
U(1) when a != b, and all of SU(5) when a = b, in which case W is central and
breaks nothing. :func:`wilson_lines` enumerates them for a cyclic group and
:func:`unbroken_group` names the commutant.

The count has a closed form, found by enumeration and then proved in the
docstring of :func:`wilson_line_count`: for Gamma = Z_n there are exactly

    n - gcd(n, 5)

Wilson lines breaking SU(5) to the Standard Model. It vanishes precisely for
n = 1 and n = 5. So a Z_5 quotient -- which is otherwise a perfectly good
freely acting symmetry, and does divide the generation count by five -- cannot
break the GUT group at all, and Z_2 is the smallest that can. That is the
smallest order the tetraquadric models of :mod:`pyCICY.bundles` use.

What is NOT determined, and is not computed here
------------------------------------------------
The downstairs spectrum needs the representation of Gamma on each upstairs
cohomology group. That is not a function of the configuration matrix, the
charges, or anything else this package holds: it requires an *equivariant
structure* on each line bundle, a lift of the Gamma action to the total space,
and different lifts give different answers on the same topological data.

So :func:`project` takes the Gamma-representation content as an argument. It
computes what survives, given that content; it does not invent it. Passing
made-up multiplicities produces a made-up spectrum, and the function says so
rather than presenting the result as derived. This is the same line
:mod:`pyCICY.phenomenology` draws around the Yukawa couplings, and for the
same reason: a plausible-looking number here would be indistinguishable in a
table from a computed one.

What :func:`project` *is* good for is the consistency conditions. Given a
proposed assignment of Gamma-charges it checks that the surviving generation
count matches -ind(V)/|Gamma| from :mod:`pyCICY.bundles`, that the hypercharge
anomaly still cancels downstairs, and that the colour-triplet partner of the
Higgs doublet is projected out -- doublet-triplet splitting, which in this
construction is a condition on Gamma-charges rather than a tuning.
"""

import itertools
import math
from fractions import Fraction as F

__all__ = [
    "hypercharge_generator", "branching", "verify_against_flavor",
    "wilson_lines", "wilson_line_count", "unbroken_group", "breaks_to_sm",
    "minimal_order", "project", "doublet_triplet_split", "worked_example",
    "anomaly_trace_of_generation",
    "generations_downstairs", "chiral_spectrum",
]


# ---------------------------------------------------------------------------
# SU(5) branching
# ---------------------------------------------------------------------------

def hypercharge_generator():
    """Y in the fundamental 5 of SU(5), as exact rationals.

    Y = diag(-1/3, -1/3, -1/3, 1/2, 1/2), traceless as a generator must be.
    The normalisation is fixed by requiring that the lepton doublet inside the
    5bar carry Y = -1/2, which is the convention of
    :data:`pyCICY.flavor.SM_HYPERCHARGES`.
    """
    return [F(-1, 3)] * 3 + [F(1, 2)] * 2


def branching(rep="10"):
    r"""
    Decompose an SU(5) representation into SU(3) x SU(2) x U(1)_Y.

    Supported: ``'5'``, ``'5bar'``, ``'10'``, ``'10bar'``, ``'24'``. The
    hypercharges are computed from :func:`hypercharge_generator` -- for the 10,
    the antisymmetric square, as Y_i + Y_j over pairs -- rather than tabulated,
    so the arithmetic is visible.

    Returns a list of ``(name, (dim3, dim2), Y, multiplicity)``, where the
    multiplicity is the number of states, dim3 * dim2.
    """
    Y = hypercharge_generator()
    if rep in ("5", "5bar"):
        s = 1 if rep == "5" else -1
        return [("(3%s, 1)" % ("" if s > 0 else "bar"), (3, 1), s * Y[0], 3),
                ("(1, 2)", (1, 2), s * Y[3], 2)]
    if rep in ("10", "10bar"):
        s = 1 if rep == "10" else -1
        cc = [("(3bar, 1)", (3, 1), s * (Y[0] + Y[1]), 3),
              ("(3, 2)", (3, 2), s * (Y[0] + Y[3]), 6),
              ("(1, 1)", (1, 1), s * (Y[3] + Y[4]), 1)]
        return cc
    if rep == "24":
        return [("(8, 1)", (8, 1), F(0), 8),
                ("(1, 3)", (1, 3), F(0), 3),
                ("(1, 1)", (1, 1), F(0), 1),
                ("(3, 2)", (3, 2), F(-5, 6), 6),
                ("(3bar, 2)", (3, 2), F(5, 6), 6)]
    raise ValueError("unsupported representation %r" % rep)


def verify_against_flavor():
    r"""
    Check the branching against :data:`pyCICY.flavor.SM_HYPERCHARGES`.

    A generation sits in 10 + 5bar as left-handed conjugates, so the
    hypercharges here are minus those of the fields in the flavour module:

        (3, 2)_{1/6}     <-> qL with Y = +1/6      (same, the doublet is left-handed)
        (3bar, 1)_{-2/3} <-> uR with Y = +2/3      (conjugate)
        (1, 1)_{1}       <-> eR with Y = -1        (conjugate)
        (3bar, 1)_{1/3}  <-> dR with Y = -1/3      (conjugate)
        (1, 2)_{-1/2}    <-> lL with Y = -1/2      (same, left-handed)

    The two tables were written for different papers and never referred to one
    another, so agreement is a real check. Returns ``(ok, table)``.
    """
    from . import flavor

    pieces = branching("10") + branching("5bar")
    want = {
        "(3, 2)": ("qL", 1),
        "(3bar, 1)": None,          # appears twice, resolved by hypercharge
        "(1, 1)": ("eR", -1),
        "(1, 2)": ("lL", 1),
    }
    table = []
    ok = True
    for name, dims, Y, mult in pieces:
        if name == "(3bar, 1)":
            field, sign = ("uR", -1) if Y < 0 else ("dR", -1)
        else:
            field, sign = want[name]
        target = flavor.SM_HYPERCHARGES[field][0]
        good = (Y == sign * target)
        table.append((name, Y, field, target, sign, good))
        ok = ok and good
    return ok, table


def anomaly_trace_of_generation():
    """Tr(Y) over 10 + 5bar. Must vanish, and does, by a different route
    from :func:`pyCICY.flavor.anomaly_trace`: there it is a sum over fields
    with colour multiplicities, here a trace over SU(5) representations."""
    return sum(Y * mult for _, _, Y, mult in
               branching("10") + branching("5bar"))


# ---------------------------------------------------------------------------
# Wilson lines
# ---------------------------------------------------------------------------

def wilson_lines(n):
    r"""
    Wilson lines of a cyclic group Z_n breaking SU(5) to the Standard Model.

    In the hypercharge direction a Wilson line is W = diag(a, a, a, b, b) with
    a, b n-th roots of unity and det W = a^3 b^2 = 1. Writing a = w^p, b = w^q
    with w = exp(2 pi i / n), the determinant condition is

        3p + 2q = 0   (mod n)

    and W breaks SU(5) if and only if a != b, i.e. p != q. When a = b the
    condition forces a^5 = 1 and W is central in SU(5), so it commutes with
    everything and breaks nothing.

    Returns the list of ``(p, q)``.
    """
    if n < 1:
        raise ValueError("n must be a positive integer")
    return [(p, q) for p in range(n) for q in range(n)
            if (3 * p + 2 * q) % n == 0 and p != q]


def wilson_line_count(n):
    r"""
    The number of SM-breaking Wilson lines of Z_n, in closed form:

        #{ (p, q) : 3p + 2q = 0 mod n, p != q }  =  n - gcd(n, 5) .

    *Proof.* The congruence 3p + 2q = 0 (mod n) has, for each p, a solution
    set in q that is a coset of the subgroup of order gcd(2, n) -- but it is
    cleaner to count the whole solution set: the map (p, q) -> 3p + 2q from
    (Z_n)^2 to Z_n is surjective, since gcd(2, 3, n) = 1, so its kernel has
    exactly n elements. Among those, the ones with p = q satisfy 5p = 0
    (mod n), of which there are gcd(n, 5). Subtracting gives n - gcd(n, 5).

    The count therefore vanishes exactly when n = gcd(n, 5), i.e. when n
    divides 5: n = 1 and n = 5. A Z_5 quotient divides the generation count by
    five and still cannot break the GUT group, because every Wilson line
    compatible with the determinant condition is central. :func:`minimal_order`
    records that Z_2 is the smallest group that works.

    Checked against :func:`wilson_lines` by direct enumeration in the tests.
    """
    return n - math.gcd(n, 5)


def unbroken_group(p, q, n):
    """The commutant in SU(5) of the Wilson line diag(w^p, w^p, w^p, w^q, w^q).

    Returns a string. The block structure is all there is to it: a matrix
    commuting with a diagonal matrix having two distinct eigenvalues, of
    multiplicity 3 and 2, is block diagonal, and inside SU(5) that is
    S(U(3) x U(2)) = SU(3) x SU(2) x U(1).
    """
    if (3 * p + 2 * q) % n != 0:
        raise ValueError("det W != 1: 3p + 2q must vanish mod n")
    if (p - q) % n == 0:
        return "SU(5)"
    return "SU(3) x SU(2) x U(1)"


def breaks_to_sm(p, q, n):
    """Whether this Wilson line breaks SU(5) to exactly the Standard Model group."""
    return unbroken_group(p, q, n) == "SU(3) x SU(2) x U(1)"


def minimal_order():
    """The smallest |Gamma| admitting an SM-breaking Wilson line.

    It is 2, realised by W = diag(1, 1, 1, -1, -1). That is exactly the
    symmetry order the tetraquadric models found by
    :func:`pyCICY.bundles.scan` require, so the two halves of the construction
    are compatible at the smallest possible group -- which is not automatic:
    Z_5 would have been the naive guess from ind(V) = -3|Gamma| and it is one
    of the two orders that provably cannot work.
    """
    n = 1
    while wilson_line_count(n) == 0:
        n += 1
    return n


# ---------------------------------------------------------------------------
# the quotient spectrum
# ---------------------------------------------------------------------------

def generations_downstairs(index, gamma_order):
    """N_gen on X/Gamma, from the upstairs index. Raises if not integral.

    This is the arithmetic :mod:`pyCICY.bundles` already performs; it is
    repeated here so that :func:`project` has something to check its input
    against.
    """
    n = -int(index)
    if n % gamma_order != 0:
        raise ValueError(
            "ind(V) = %s is not divisible by |Gamma| = %d, so this bundle does "
            "not descend with an integral generation count"
            % (index, gamma_order))
    return n // gamma_order


def project(charges, gamma_order, wilson=None, index=None):
    r"""
    The surviving spectrum on X/Gamma, **given** the Gamma-charges upstairs.

    Parameters
    ----------
    charges : dict
        For each multiplet name -- ``'10'``, ``'10bar'``, ``'5bar'``, ``'5'``,
        ``'1'`` -- a list of the Gamma-charges of the upstairs cohomology
        states, as integers mod ``gamma_order``. **This is the input that the
        topology does not supply.** It encodes the equivariant structure on the
        bundle, which is a choice of lift of the Gamma action and not a
        function of the charges or the configuration matrix.
    gamma_order : int
        |Gamma|, for a cyclic Gamma.
    wilson : (p, q), optional
        The Wilson line. Its effect is to shift the charge of each SM piece by
        the hypercharge-weighted amount, so that different pieces of one SU(5)
        multiplet survive or not independently -- which is the entire point of
        a Wilson line and the mechanism behind doublet-triplet splitting.
    index : int, optional
        ``ind(V)`` upstairs. When given, the surviving generation count is
        checked against ``-index/gamma_order`` and a mismatch is reported.

    Returns
    -------
    dict
        ``spectrum`` mapping each SM piece to its surviving multiplicity,
        ``widths`` giving the number of states in each piece,
        ``generations``, ``anomaly`` (the hypercharge trace downstairs,
        weighted by those widths), and ``consistent`` when ``index`` was
        supplied.

    Notes
    -----
    A state survives when its total charge -- the upstairs Gamma-charge plus
    the Wilson line shift -- vanishes mod |Gamma|, since only Gamma-invariant
    states descend to the quotient.

    Nothing here derives ``charges``. Supplying charges that do not come from
    an actual equivariant structure gives a spectrum that is arithmetically
    consistent and physically meaningless, and no function in this module can
    tell the difference. What the return value *does* certify is internal
    consistency: the generation count against the index, and the vanishing of
    the hypercharge anomaly.
    """
    n = int(gamma_order)
    if n < 1:
        raise ValueError("gamma_order must be positive")

    pieces = {}
    for rep in ("10", "10bar", "5bar", "5"):
        if rep in charges:
            for name, dims, Y, mult in branching(rep):
                pieces.setdefault((rep, name, Y, mult), [])

    shift = _wilson_shift(wilson, n) if wilson else {}

    spectrum = {}
    widths = {}
    for rep, states in charges.items():
        if rep == "1":
            spectrum[("1", "(1, 1)", F(0))] = sum(
                1 for c in states if c % n == 0)
            continue
        for name, dims, Y, mult in branching(rep):
            s = shift.get(Y, 0)
            surviving = sum(1 for c in states if (c + s) % n == 0)
            spectrum[(rep, name, Y)] = surviving
            widths[(rep, name, Y)] = mult

    tens = sum(v for k, v in spectrum.items() if k[0] == "10")
    tenbars = sum(v for k, v in spectrum.items() if k[0] == "10bar")

    # The anomaly is a trace over *states*, so each surviving multiplet
    # contributes its full width: a surviving (3,2) is six states, not one.
    # Omitting the width gives a number that looks like an anomaly and is not.
    anomaly = sum(F(k[2]) * v * widths.get(k, 1) for k, v in spectrum.items())
    out = {"spectrum": spectrum,
           "widths": widths,
           "generations": None,
           "anomaly": anomaly,
           "consistent": None}

    # generations from the (1,1)_1 piece of the 10, the right-handed electron,
    # which is the cleanest single counter since it has multiplicity one.
    e_up = spectrum.get(("10", "(1, 1)", F(1)), 0)
    e_dn = spectrum.get(("10bar", "(1, 1)", F(-1)), 0)
    out["generations"] = e_up - e_dn

    if index is not None:
        try:
            want = generations_downstairs(index, n)
            out["consistent"] = (out["generations"] == want)
            out["expected_generations"] = want
        except ValueError as e:
            out["consistent"] = False
            out["error"] = str(e)
    return out


def worked_example():
    r"""
    A consistent case, end to end, on the tetraquadric model of :mod:`bundles`.

    That model has ind(V) = -6, n(10) = 24 and n(10-bar) = 18 upstairs, and is
    poly-stable with h^0 = h^3 = 0. Quotienting by a freely acting Z_2 should
    leave three generations.

    The Gamma-charges below are *chosen*, not derived -- there is no
    equivariant structure in this package to derive them from -- but they are
    chosen to be the ones a free Z_2 action would give if it split each
    cohomology group as evenly as the counts allow: 12 of the 24 tens
    invariant, 9 of the 18 anti-tens. What the example demonstrates is that
    the consistency conditions then close: three generations, matching
    -ind(V)/|Gamma|, with the hypercharge anomaly vanishing.

    The Higgs charges are deliberately *not* symmetric. With W = (0, 1) the
    weak doublet inside the 5bar shifts by -q = 1 while its colour triplet
    shifts by -p = 0, so giving every Higgs multiplet charge 1 keeps the
    doublets and projects the triplets out entirely. Splitting the charges
    evenly instead -- three of each -- would leave both surviving in equal
    numbers and hide the mechanism, which is worth knowing when reading a
    spectrum that looks symmetric.

    Returns the dict from :func:`project`.
    """
    charges = {
        "10": [0] * 12 + [1] * 12,
        "10bar": [0] * 9 + [1] * 9,
        "5bar": [1] * 3,
        "5": [1] * 3,
    }
    return project(charges, 2, wilson=(0, 1), index=-6)


def _wilson_shift(wilson, n):
    r"""Charge shift induced on each SM piece by a Wilson line.

    The Wilson line acts on a state of hypercharge Y by a phase determined by
    its position in the SU(5) multiplet. In the parametrisation
    W = diag(w^p, w^p, w^p, w^q, w^q), a state built from k indices in the
    colour block and l in the weak block picks up w^{kp + lq}, and (k, l) is
    determined by Y within each multiplet.
    """
    p, q = wilson
    return {
        F(1, 6): (p + q) % n,        # (3,2) of the 10
        F(-2, 3): (2 * p) % n,       # (3bar,1) of the 10
        F(1): (2 * q) % n,           # (1,1) of the 10
        F(1, 3): (-p) % n,           # (3bar,1) of the 5bar
        F(-1, 2): (-q) % n,          # (1,2) of the 5bar
        F(-1, 6): (-p - q) % n,
        F(2, 3): (-2 * p) % n,
        F(-1): (-2 * q) % n,
        F(-1, 3): p % n,
        F(1, 2): q % n,
    }


def doublet_triplet_split(charges, gamma_order, wilson):
    r"""
    Whether the Wilson line separates the Higgs doublet from its colour triplet.

    A Higgs sits in a 5 (and 5bar) of SU(5), which contains a weak doublet
    (1, 2)_{-1/2} and a colour triplet (3bar, 1)_{1/3}. The triplet mediates
    proton decay and must not survive. Upstairs the two are one irreducible
    multiplet and cannot be separated; the Wilson line shifts their charges by
    different amounts -- ``-q`` and ``-p`` respectively -- so a choice with
    p != q can keep one and project out the other.

    Returns a dict with the surviving ``doublets`` and ``triplets`` and a
    boolean ``split``. The mechanism is a condition on p, q and the
    Gamma-charges, not a tuning of a continuous parameter, which is the usual
    argument for preferring this construction to a four-dimensional GUT.
    """
    res = project(charges, gamma_order, wilson=wilson)
    doub = sum(v for k, v in res["spectrum"].items() if k[1] == "(1, 2)")
    trip = sum(v for k, v in res["spectrum"].items() if k[1] == "(3bar, 1)"
               and k[2] in (F(1, 3), F(-1, 3)))
    return {"doublets": doub, "triplets": trip,
            "split": bool(doub > 0 and trip == 0)}


def chiral_spectrum(action, summands, wilson=None, X=None):
    r"""
    The chiral spectrum on X/Gamma, **derived** rather than supplied.

    :func:`project` takes the Gamma-charges as an argument because the
    representation of Gamma on each cohomology group is not a function of the
    topology. That remains true. But the *chiral* part -- the net content, an
    index -- is determined, and :mod:`pyCICY.equivariant` computes it. This
    function joins the two, so that the pipeline
    :func:`pyCICY.bundles.scan` -> equivariant index -> Standard Model
    spectrum runs end to end with nothing chosen by hand.

    Parameters
    ----------
    action : a :class:`pyCICY.equivariant.CyclicAction`, ``PermutationAction``
        or ``AbelianAction`` on the same manifold.
    summands : list of charge vectors
        The rank-5 line bundle sum, as :class:`pyCICY.bundles.LineBundleSum`
        takes it.
    wilson : (p, q), optional
        The Wilson line, as in :func:`project`.
    X : CICY, optional
        Only needed to build the wedge and endomorphism bundles; inferred from
        ``action.conf`` when omitted.

    Returns
    -------
    dict
        ``spectrum`` mapping each Standard Model piece to its net chiral
        multiplicity downstairs, ``generations``, ``anomaly``, and
        ``equidistributed`` recording whether the index characters were
        constant.

    The result for a free action, and why
    ------------------------------------
    The 10 of SU(5) sits in H^1(V) and the 5-bar in H^1(Lambda^2 V), so the
    net number of each is read off ind(V) and ind(Lambda^2 V). A Wilson line
    shifts the Gamma-charge of each Standard Model piece within its multiplet
    by a different amount -- that is how it splits them -- and the surviving
    multiplicity is the one at the shifted charge.

    For a *free* Gamma every one of those characters is a multiple of the
    regular representation, i.e. constant, so **every shift lands on the same
    multiplicity** and the Wilson line cannot split the chiral spectrum at
    all. What comes out is complete SU(5) generations, -ind(V)/|Gamma| of
    them, whatever Wilson line is chosen.

    That is the correct physics and it is worth being explicit that it is
    derived here rather than assumed: doublet-triplet splitting is necessarily
    a statement about *vector-like* pairs, the non-chiral content that an
    index cannot see. :func:`doublet_triplet_split` operates on exactly that
    content, which is why it still takes Gamma-charges as input and why
    :func:`worked_example` supplies them by hand. The two functions are not in
    tension; they describe different halves of the spectrum, and only one half
    is determined by topology.
    """
    from . import bundles as _bundles
    from . import equivariant as _eq

    n = int(getattr(action, "n", getattr(action, "order", 1)))
    if X is None:
        from .pyCICY import CICY
        X = CICY(action.conf.tolist())
    V = _bundles.LineBundleSum(X, summands)

    def character(bundle_summands):
        ch = _eq.bundle_index_character(action, bundle_summands)
        if isinstance(ch, dict):                 # AbelianAction
            keys = sorted(ch)
            return [ch[k] for k in keys]
        return list(ch)

    cV = character(V.summands)
    cW = character(V.wedge2().summands)
    cE = character(V.endomorphisms().summands)

    equid = {"V": len(set(cV)) == 1,
             "wedge2": len(set(cW)) == 1,
             "endomorphisms": len(set(cE)) == 1}

    shift = _wilson_shift(wilson, len(cV)) if wilson else {}
    spectrum = {}
    widths = {}
    for rep, chars in (("10", cV), ("5bar", cW)):
        for name, dims, Y, mult in branching(rep):
            s = shift.get(Y, 0) % len(chars)
            key = (rep, name, Y)
            spectrum[key] = -chars[(-s) % len(chars)]
            widths[key] = mult

    gens = spectrum.get(("10", "(1, 1)", F(1)), 0)
    # Weighted by the number of states in each piece, as in project(): a
    # surviving (3,2) is six states, not one. Pairing the multiplicities to
    # the spectrum by key rather than by iteration order, since the two are
    # built in different loops and a zip between them would be silently
    # order-dependent.
    anomaly = sum(F(k[2]) * v * widths[k] for k, v in spectrum.items())
    return {"spectrum": spectrum,
            "widths": widths,
            "generations": gens,
            "anomaly": anomaly,
            "equidistributed": equid,
            "index_V": sum(cV),
            "character_V": cV,
            "character_wedge2": cW,
            "character_endomorphisms": cE,
            "gamma_order": n}
