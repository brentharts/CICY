r"""
pyCICY.phenomenology -- what the topology of a CICY does, and does not,
determine about four-dimensional physics.

This module exists to draw a line carefully, because the line is easy to
blur. Some quantities of phenomenological interest follow from the
configuration matrix alone by index theorems. Others do not follow from it at
all, at any level of computational effort, because they depend on data the
configuration matrix does not contain.

What is determined by the topology
----------------------------------
For the heterotic string with the *standard embedding*, V = TX, the low
energy gauge group is E_6 and the charged matter is counted by Hodge numbers:

    n(27)      = h^{2,1}(X),
    n(27-bar)  = h^{1,1}(X),
    net chiral generations = h^{2,1} - h^{1,1} = -chi(X)/2 .

The last equality is the index theorem for the Dirac operator twisted by V,
and it is why the Euler characteristic is the first thing a phenomenologist
computes. Quotienting by a freely acting discrete symmetry Gamma divides it:

    net generations on X/Gamma = |chi(X)| / (2 |Gamma|) .

Three generations therefore requires |chi| = 6|Gamma|. Since no CICY in the
published list has |chi| = 6 (:func:`generation_survey` checks this on the
data), a three-generation standard-embedding model always needs a non-trivial
freely acting quotient, which is also what supplies the Wilson lines that
break E_6 down towards the Standard Model group.

What is NOT determined by the topology
--------------------------------------
Masses and couplings are not. The proton-to-electron mass ratio, the fine
structure constant, and the fermion mass hierarchy are not functions of the
configuration matrix, and no amount of computation here will produce them.
They depend on:

* the *moduli* -- the continuous Kahler and complex structure parameters
  fixing the size and shape of X. Physical Yukawa couplings are the
  holomorphic couplings divided by the Kahler normalisation of the fields,
  and the Kahler potential depends on the Ricci-flat metric, for which no
  closed form is known on any compact CY threefold;
* the dilaton vacuum expectation value, which sets the gauge coupling;
* the mechanism of supersymmetry breaking, which sets the overall scale;
* the choice of gauge bundle V when it is not the standard embedding, and of
  Wilson lines, neither of which is topology alone.

Stabilising the moduli is an unsolved problem across the landscape. Anything
in this module that would require them raises rather than returning a number,
because a plausible-looking number here would be worse than no number at all:
it would be indistinguishable in a plot from a computed one.

:func:`why_not_masses` returns that explanation in structured form so a
caller can display it instead of a fabricated value.
"""

__all__ = [
    "standard_embedding_spectrum", "chiral_generations",
    "required_symmetry_order", "generation_survey", "why_not_masses",
    "MassRatioNotComputable",
]


class MassRatioNotComputable(NotImplementedError):
    """Raised when asked for a quantity topology cannot determine.

    Deliberately an exception rather than a sentinel value. A function that
    returned ``None`` or a placeholder float would sooner or later be plotted
    next to a measured constant, and at that point nothing in the figure
    would distinguish a computed number from an invented one.
    """


def _hodge(conf, cache=None):
    from .cache import hodge as _h
    data = _h(conf, cache=cache)
    if data.get("error"):
        raise ValueError("cannot evaluate %r: %s" % (conf, data["error"]))
    if data.get("nfold") != 3:
        raise ValueError("heterotic generation counting applies to Calabi-Yau "
                         "threefolds; this is a %s-fold" % data.get("nfold"))
    return data


def standard_embedding_spectrum(conf, symmetry_order=1, cache=None):
    r"""Charged spectrum of the heterotic standard embedding on X or X/Gamma.

    With V = TX the gauge group is E_6, the 27s are counted by h^{2,1} and
    the 27-bars by h^{1,1}, so the net number of chiral generations is
    h^{2,1} - h^{1,1} = -chi/2. On a free quotient X/Gamma every count is
    divided by |Gamma|.

    Parameters
    ----------
    symmetry_order : int
        |Gamma| for a freely acting symmetry. Must divide chi, since
        chi(X/Gamma) = chi(X)/|Gamma| has to be an integer; this is a
        necessary condition on Gamma, not a sufficient one. Whether a free
        action of that order actually exists is a separate question, settled
        for the CICY list by Braun, arXiv:1003.3235, and not decided here.

    Returns
    -------
    dict with ``gauge_group``, ``n_27``, ``n_27bar``, ``net_generations``,
    ``euler``, ``symmetry_order``, and ``notes``.

    Example
    -------
    >>> s = standard_embedding_spectrum([[4, 5]])
    >>> s["n_27"], s["n_27bar"], s["net_generations"]
    (101, 1, 100)
    """
    data = _hodge(conf, cache=cache)
    h11 = int(round(data["h11"]))
    h21 = int(round(data["h21"]))
    euler = int(data["euler"])

    if symmetry_order < 1:
        raise ValueError("symmetry order must be at least 1")
    if euler % symmetry_order:
        raise ValueError(
            "|Gamma| = %d does not divide chi = %d, so chi(X/Gamma) would not "
            "be an integer and no free action of that order can exist"
            % (symmetry_order, euler))

    notes = []
    if symmetry_order > 1:
        notes.append(
            "counts on the quotient assume a freely acting Gamma of this "
            "order exists; existence is not checked here")
    else:
        notes.append(
            "E_6 is not the Standard Model group; breaking it requires "
            "Wilson lines, which need a non-trivial fundamental group and "
            "hence a free quotient")

    if h11 % symmetry_order or h21 % symmetry_order:
        notes.append(
            "h^{1,1} or h^{2,1} is not divisible by |Gamma|; the individual "
            "27 and 27-bar counts on the quotient are not simply the "
            "quotients of these numbers, only the net difference is")

    return {
        "gauge_group": "E_6",
        "n_27": h21 // symmetry_order if h21 % symmetry_order == 0 else None,
        "n_27bar": h11 // symmetry_order if h11 % symmetry_order == 0 else None,
        "net_generations": abs(euler) // (2 * symmetry_order),
        "euler": euler,
        "h11": h11,
        "h21": h21,
        "symmetry_order": symmetry_order,
        "notes": notes,
    }


def chiral_generations(euler, symmetry_order=1):
    """Net chiral generations from the Euler characteristic.

    >>> chiral_generations(-200)
    100
    >>> chiral_generations(-24, 4)
    3
    """
    if symmetry_order < 1:
        raise ValueError("symmetry order must be at least 1")
    if euler % (2 * symmetry_order):
        raise ValueError(
            "chi = %d is not divisible by 2|Gamma| = %d, so the index is not "
            "an integer" % (euler, 2 * symmetry_order))
    return abs(euler) // (2 * symmetry_order)


def required_symmetry_order(euler, generations=3):
    """Order a freely acting Gamma must have to give this many generations.

    Returns None when no integer order works.

    >>> required_symmetry_order(-24)
    4
    >>> required_symmetry_order(-200)          # 100 is not divisible by 3
    """
    if euler == 0:
        return None
    total = abs(euler)
    if total % (2 * generations):
        return None
    return total // (2 * generations)


def generation_survey(entries, generations=3):
    """Survey a list of CICYs for three-generation candidates.

    ``entries`` is the record list from
    :func:`pyCICY.cicylist.load_published_list`, or anything with ``euler``
    keys.

    Returns counts, the distribution of required |Gamma|, and the smallest
    |chi| observed, which is what decides whether a quotient is needed at
    all.
    """
    import collections

    eulers = [int(r["euler"]) for r in entries]
    nonzero = [abs(e) for e in eulers if e != 0]
    orders = collections.Counter()
    candidates = 0
    for e in eulers:
        order = required_symmetry_order(e, generations)
        if order is not None:
            candidates += 1
            orders[order] += 1

    return {
        "entries": len(entries),
        "zero_euler": sum(1 for e in eulers if e == 0),
        "candidates": candidates,
        "required_orders": dict(orders),
        "min_abs_euler": min(nonzero) if nonzero else None,
        "needs_quotient": (2 * generations) not in nonzero,
        "generations": generations,
    }


def why_not_masses(quantity="proton-to-electron mass ratio"):
    """Structured explanation of why a mass or coupling is not computable here.

    Returned rather than printed so a caller can render it, and so that the
    reason travels with the refusal instead of being lost.
    """
    return {
        "quantity": quantity,
        "computable_from_topology": False,
        "reasons": [
            "physical Yukawa couplings are holomorphic couplings divided by "
            "the Kahler normalisation of the fields, and that normalisation "
            "depends on the Ricci-flat metric, which is not known in closed "
            "form on any compact Calabi-Yau threefold",
            "the overall mass scale depends on the supersymmetry breaking "
            "mechanism, which is not fixed by the compactification geometry",
            "the gauge coupling depends on the dilaton vacuum expectation "
            "value, a modulus",
            "all of the above require the moduli to be stabilised, which is "
            "an open problem across the landscape",
            "for a bundle other than the standard embedding, the spectrum "
            "depends on the choice of V and of Wilson lines, which are extra "
            "data beyond the configuration matrix",
        ],
        "what_is_computable": [
            "the net number of chiral generations, |chi| / (2|Gamma|)",
            "the 27 and 27-bar counts for the standard embedding, h^{2,1} "
            "and h^{1,1}",
            "the order |Gamma| a freely acting symmetry would need for a "
            "given generation number",
            "topological constraints such as the anomaly condition on c_2",
        ],
    }
