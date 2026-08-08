r"""
pyCICY.theories.ftheory -- F-theory compactifications.

The subpackage docstring used to end by saying that F-theory "would go here
too. [It is] not implemented." This module is that gap closed for the
six-dimensional case, and it is worth saying at the start why F-theory fits
the interface so cleanly and where it strains it.

What F-theory is, for present purposes
--------------------------------------
Type IIB with a varying axio-dilaton tau, geometrised as the complex structure
of an auxiliary torus fibred over the physical space. The total space is
elliptically fibred, Calabi-Yau, and one dimension higher than the IIB
background. Fibring over a complex *surface* B gives a Calabi-Yau threefold
and six-dimensional N=(1,0) supergravity, which is what this module computes.

Everything here is exact, and for a sharper reason than elsewhere in the
package. Six-dimensional (1,0) supergravity is anomalous unless a set of
polynomial identities in the divisor classes hold, and those identities are
strong enough to *determine* the spectrum from the geometry. They are integer
and rational arithmetic on the intersection form of a surface. No cohomology
computation, no metric, no approximation.

The three categories
--------------------
The package distinguishes quantities that are exact from quantities that need
the Ricci-flat metric. F-theory in six dimensions adds a third category that
the heterotic modules never had occasion to name:

    exact              the spectrum, the gauge algebra, the Hodge numbers of
                       the elliptic threefold, the anomaly conditions
    needs a metric     the hypermultiplet moduli space metric, the physical
                       normalisation of the kinetic terms
    does not exist     the Yukawa couplings

The last one is not a gap. Six-dimensional (1,0) supersymmetry forbids a
superpotential: hypermultiplets have no cubic coupling to write down, so
:meth:`FTheory6D.holomorphic_yukawa` raises rather than returning zero, and it
raises with a different message from :class:`~pyCICY.theories.base.NeedsMetric`
because the reason is different. A coupling that is absent by supersymmetry
and a coupling that is present but uncomputable should not produce the same
exception.

What is tabulated and what is derived
-------------------------------------
The Kodaira table (:func:`kodaira_type`) and the non-Higgsable cluster list
(:data:`NON_HIGGSABLE`) are quoted results, from Kodaira's classification of
singular fibres and from Morrison and Taylor's classification of bases. The
group-theory coefficients in :data:`_ALGEBRA` are quoted too.

The matter content is *not* tabulated. :func:`matter_content` solves the
anomaly conditions for the multiplicities, and the answers it returns for the
standard cases -- SU(N) on a -1 curve with N+8 fundamentals and one
antisymmetric, E7 on a -7 curve with half a 56 -- come out of the linear
algebra rather than out of a lookup. :func:`matter_free_algebras` likewise
*derives* the six matter-free non-Higgsable algebras, and the fact that it
reproduces the tabulated {su(3), so(8), f4, e6, e7, e8} at self-intersections
{-3, -4, -5, -6, -8, -12} is a check on the group theory, not an input to it.

Connection to CICYs
-------------------
The generic Weierstrass model over a base is a hypersurface in a bundle over
B with fibre P^{2,3,1}, which is not a product of projective spaces, so it is
not a CICY and does not appear in the list this package is built around. What
CICYs do have is *obvious* fibrations in the sense of Anderson, Gao, Gray and
Lee: a splitting of the configuration matrix in which a block of rows is zero
outside a block of columns, so that the block is itself a Calabi-Yau one-fold
and the complement is a surface. :func:`obvious_fibrations` finds these, and
:class:`FTheory4D` uses the same routine one dimension up.

The distinction there is worth keeping: an obvious fibration gives a genus-one
fibration, which need not have a section. The (3,3) hypersurface in P^2 x P^2
fibres in elliptic curves over P^2 and has Hodge numbers (2, 83); the
Weierstrass model over the same base has (2, 272). They are different
manifolds, and only the second is an F-theory background in the naive sense.
:func:`obvious_fibrations` reports the fibration, not a claim about sections.

References
----------
Kodaira, On compact analytic surfaces II, Ann. Math. 77 (1963) 563.
Morrison and Vafa, Compactifications of F-theory on Calabi-Yau threefolds I,
    II, Nucl. Phys. B473 (1996) 74, B476 (1996) 437.
Kumar, Morrison and Taylor, Global aspects of the space of 6D N=1
    supergravities, JHEP 1011 (2010) 118.
Morrison and Taylor, Classifying bases for 6D F-theory models, Cent. Eur. J.
    Phys. 10 (2012) 1072.
Anderson, Gao, Gray and Lee, Fibrations in CICY threefolds, JHEP 1710 (2017)
    077.
"""

from fractions import Fraction

import numpy as np

from .base import NeedsMetric, Theory, register

__all__ = ["Base", "FTheory6D", "FTheory4D", "NoSuchTheory",
           "kodaira_type", "KODAIRA", "NON_HIGGSABLE",
           "algebra_data", "matter_content", "check_anomalies", "reality",
           "weierstrass_euler", "ProductBase",
           "fourfold_euler", "fourfold_hodge",
           "matter_free_algebras", "weierstrass_moduli",
           "obvious_fibrations", "is_obviously_fibred"]


class NoSuchTheory(NotImplementedError):
    """Raised for a quantity that does not exist, rather than one not computed.

    The package already separates "exact" from "needs the Ricci-flat metric",
    via :exc:`~pyCICY.theories.base.NeedsMetric`. Six-dimensional (1,0)
    supergravity forces a third case: a Yukawa coupling in six dimensions is
    not unavailable, it is forbidden by supersymmetry. Returning zero would be
    numerically right and conceptually wrong, and raising
    :exc:`NeedsMetric` would blame the metric for an absence that has nothing
    to do with it.
    """


# ---------------------------------------------------------------------------
# Kodaira's classification of singular fibres
# ---------------------------------------------------------------------------

#: Kodaira fibre types, keyed by the name, with the vanishing orders of
#: (f, g, Delta) that produce them and the gauge algebra carried by the
#: divisor. Where the algebra depends on the monodromy the split and
#: non-split cases are both given; see :func:`kodaira_type`.
KODAIRA = {
    "I_0":  {"split": None,        "nonsplit": None},
    "I_n":  {"split": "su(n)",     "nonsplit": "sp(floor(n/2))"},
    "II":   {"split": None,        "nonsplit": None},
    "III":  {"split": "su(2)",     "nonsplit": "su(2)"},
    "IV":   {"split": "su(3)",     "nonsplit": "su(2)"},
    "I_0*": {"split": "so(8)",     "nonsplit": "so(7) or g2"},
    "I_n*": {"split": "so(2n+8)",  "nonsplit": "so(2n+7)"},
    "IV*":  {"split": "e6",        "nonsplit": "f4"},
    "III*": {"split": "e7",        "nonsplit": "e7"},
    "II*":  {"split": "e8",        "nonsplit": "e8"},
}


def kodaira_type(ord_f, ord_g, ord_delta, check=True):
    r"""The Kodaira fibre type from the vanishing orders of f, g and Delta.

    In a Weierstrass model y^2 = x^3 + f x + g the singular fibres sit over
    the vanishing locus of Delta = 4 f^3 + 27 g^2, and the type of degeneration
    -- hence the gauge algebra on that divisor -- is fixed by how fast f, g and
    Delta vanish along it.

    Parameters
    ----------
    ord_f, ord_g, ord_delta : int
        Orders of vanishing along the divisor.
    check : bool, optional
        Verify the orders against Delta = 4 f^3 + 27 g^2. Since
        ord(Delta) >= min(3 ord f, 2 ord g) always, with equality unless the
        two terms cancel at leading order, an ``ord_delta`` below that bound is
        impossible and one above it when ``3 ord_f != 2 ord_g`` is impossible
        too. Both raise :exc:`ValueError`.

    Returns
    -------
    dict
        ``type``, ``algebra_split``, ``algebra_nonsplit``, ``rank_split``,
        ``minimal`` and ``note``.

    Notes
    -----
    A model with ord(f) >= 4 and ord(g) >= 6 is *non-minimal*: it is not the
    Weierstrass form of a Calabi-Yau with canonical singularities, and the base
    has to be blown up before the model means anything. The function reports
    this rather than assigning a fibre type to it.

    Examples
    --------
    >>> kodaira_type(0, 0, 0)["type"]
    'I_0'
    >>> kodaira_type(4, 5, 10)["algebra_split"]
    'e8'
    >>> kodaira_type(2, 3, 8)["algebra_split"]
    'so(12)'
    """
    a, b, c = int(ord_f), int(ord_g), int(ord_delta)
    if a < 0 or b < 0 or c < 0:
        raise ValueError("vanishing orders are non-negative")

    if check:
        bound = min(3 * a, 2 * b)
        if c < bound:
            raise ValueError(
                "ord(Delta) = %d is below min(3 ord f, 2 ord g) = %d, which "
                "Delta = 4 f^3 + 27 g^2 cannot do" % (c, bound))
        if 3 * a != 2 * b and c != bound:
            raise ValueError(
                "with 3 ord f = %d and 2 ord g = %d the two terms of Delta "
                "vanish to different orders, so ord(Delta) must equal %d, not "
                "%d; a higher order needs the leading terms to cancel, which "
                "needs 3 ord f = 2 ord g" % (3 * a, 2 * b, bound, c))

    if a >= 4 and b >= 6:
        return {"type": "non-minimal", "algebra_split": None,
                "algebra_nonsplit": None, "rank_split": 0, "minimal": False,
                "note": "ord(f) >= 4 and ord(g) >= 6: the model is not in "
                        "minimal Weierstrass form and the base must be blown "
                        "up before it describes a Calabi-Yau"}

    def out(name, split=None, nonsplit=None, note=""):
        return {"type": name, "algebra_split": split,
                "algebra_nonsplit": nonsplit,
                "rank_split": _rank_of(split), "minimal": True, "note": note}

    if c == 0:
        if a and b:
            raise ValueError(
                "with ord(f) = %d and ord(g) = %d both positive, Delta = "
                "4 f^3 + 27 g^2 vanishes too, so ord(Delta) = 0 is impossible "
                "and there is no smooth fibre here" % (a, b))
        return out("I_0", note="smooth fibre, no gauge algebra")
    if a == 0 and b == 0:
        if c == 1:
            return out("I_1", note="nodal fibre, no gauge algebra")
        return out("I_%d" % c, "su(%d)" % c, "sp(%d)" % (c // 2),
                   "split gives su(%d), non-split sp(%d)" % (c, c // 2))
    if a >= 1 and b == 1 and c == 2:
        return out("II", note="cuspidal fibre, no gauge algebra")
    if a == 1 and b >= 2 and c == 3:
        return out("III", "su(2)", "su(2)")
    if a >= 2 and b == 2 and c == 4:
        return out("IV", "su(3)", "su(2)",
                   "split gives su(3), non-split su(2)")
    # I_0* needs ord(Delta) = 6, which given ord(f) >= 2 and ord(g) >= 3
    # forces min(3 ord f, 2 ord g) = 6: one of the two must sit at the
    # minimum. Deeper vanishing of both pushes Delta past 6 and the fibre is
    # something else.
    if c == 6 and a >= 2 and b >= 3 and (a == 2 or b == 3):
        return out("I_0*", "so(8)", "so(7) or g2",
                   "so(8) split, so(7) semi-split, g2 non-split")
    if a == 2 and b == 3 and c > 6:
        n = c - 6
        return out("I_%d*" % n, "so(%d)" % (2 * n + 8), "so(%d)" % (2 * n + 7),
                   "split gives so(%d), non-split so(%d)"
                   % (2 * n + 8, 2 * n + 7))
    if a >= 3 and b == 4 and c == 8:
        return out("IV*", "e6", "f4", "split gives e6, non-split f4")
    if a == 3 and b >= 5 and c == 9:
        return out("III*", "e7", "e7")
    if a >= 4 and b == 5 and c == 10:
        return out("II*", "e8", "e8")

    raise ValueError(
        "(ord f, ord g, ord Delta) = (%d, %d, %d) is not a Kodaira type. The "
        "orders are individually possible but do not occur together in the "
        "classification." % (a, b, c))


# ---------------------------------------------------------------------------
# Lie algebra data for the anomaly conditions
# ---------------------------------------------------------------------------
#
# For a representation R, tr_R F^2 = A_R tr F^2 and tr_R F^4 = B_R tr F^4 +
# C_R (tr F^2)^2, where tr is in the defining representation: the fundamental
# for su and sp, the vector for so, and 27, 56, 248, 26, 7 for e6, e7, e8, f4,
# g2. B_R only makes sense when the algebra has an independent quartic Casimir
# -- su(N >= 4), so(N >= 7), sp(N) -- and is None otherwise, in which case the
# quartic trace has been folded into C_R. lam is the normalisation factor
# lambda of Kumar-Morrison-Taylor: 1, 2, 1, 6, 12, 60, 6, 2 for su, so, sp,
# e6, e7, e8, f4, g2.

F = Fraction


def _rank_of(name):
    if not name:
        return 0
    try:
        return algebra_data(name)["rank"]
    except (ValueError, KeyError):
        return 0


def algebra_data(name):
    r"""Dimension, rank, normalisation and representation data for an algebra.

    Parameters
    ----------
    name : str
        ``"su(5)"``, ``"so(10)"``, ``"sp(2)"``, ``"e6"``, ``"e7"``, ``"e8"``,
        ``"f4"``, ``"g2"``.

    Returns
    -------
    dict
        ``dim``, ``rank``, ``lam``, ``adjoint`` (the name of the adjoint
        representation in ``reps``) and ``reps``, a dict mapping a name to
        ``(dim, A, B, C)`` with ``B`` None when the algebra has no independent
        quartic Casimir.
    """
    s = str(name).strip().lower().replace(" ", "")
    if s in ("e6", "e_6"):
        return {"dim": 78, "rank": 6, "lam": 6, "adjoint": "adj",
                "reps": {"27": (27, F(1), None, F(1, 12)),
                         "adj": (78, F(4), None, F(1, 2))}}
    if s in ("e7", "e_7"):
        return {"dim": 133, "rank": 7, "lam": 12, "adjoint": "adj",
                "reps": {"56": (56, F(1), None, F(1, 24)),
                         "adj": (133, F(3), None, F(1, 6))}}
    if s in ("e8", "e_8"):
        return {"dim": 248, "rank": 8, "lam": 60, "adjoint": "adj",
                "reps": {"adj": (248, F(1), None, F(1, 100))}}
    if s in ("f4", "f_4"):
        return {"dim": 52, "rank": 4, "lam": 6, "adjoint": "adj",
                "reps": {"26": (26, F(1), None, F(1, 12)),
                         "adj": (52, F(3), None, F(5, 12))}}
    if s in ("g2", "g_2"):
        return {"dim": 14, "rank": 2, "lam": 2, "adjoint": "adj",
                "reps": {"7": (7, F(1), None, F(1, 4)),
                         "adj": (14, F(4), None, F(5, 2))}}

    kind, n = _parse_classical(s)
    if kind == "su":
        if n < 2:
            raise ValueError("su(%d) is not a simple algebra" % n)
        if n == 2:
            # No independent quartic Casimir: tr_2 F^4 = (1/2)(tr_2 F^2)^2,
            # so the quartic traces collapse into C.
            return {"dim": 3, "rank": 1, "lam": 1, "adjoint": "adj",
                    "reps": {"fund": (2, F(1), None, F(1, 2)),
                             "adj": (3, F(4), None, F(8))}}
        if n == 3:
            return {"dim": 8, "rank": 2, "lam": 1, "adjoint": "adj",
                    "reps": {"fund": (3, F(1), None, F(1, 2)),
                             "adj": (8, F(6), None, F(9))}}
        return {"dim": n * n - 1, "rank": n - 1, "lam": 1, "adjoint": "adj",
                "reps": {"fund": (n, F(1), F(1), F(0)),
                         "antisym": (n * (n - 1) // 2, F(n - 2), F(n - 8),
                                     F(3)),
                         "sym": (n * (n + 1) // 2, F(n + 2), F(n + 8), F(3)),
                         "adj": (n * n - 1, F(2 * n), F(2 * n), F(6))}}
    if kind == "so":
        if n < 7:
            raise ValueError(
                "so(%d) is not tabulated here; below so(7) the algebra is "
                "isomorphic to an su or sp algebra, which should be used "
                "instead" % n)
        reps = {"vector": (n, F(1), F(1), F(0)),
                "adj": (n * (n - 1) // 2, F(n - 2), F(n - 8), F(3))}
        reps["spinor"] = _spinor(n)
        return {"dim": n * (n - 1) // 2, "rank": n // 2, "lam": 2,
                "adjoint": "adj", "reps": reps}
    if kind == "sp":
        if n < 1:
            raise ValueError("sp(%d) is not a simple algebra" % n)
        return {"dim": n * (2 * n + 1), "rank": n, "lam": 1, "adjoint": "adj",
                "reps": {"fund": (2 * n, F(1), F(1), F(0)),
                         "antisym": (n * (2 * n - 1) - 1, F(2 * n - 2),
                                     F(2 * n - 8), F(3)),
                         "adj": (n * (2 * n + 1), F(2 * n + 2), F(2 * n + 8),
                                 F(3))}}
    raise ValueError("unknown algebra %r" % (name,))


def _spinor(n):
    r"""Trace coefficients of the so(N) spinor, from its weights.

    Not quoted from a table. The weights of a spinor are
    (+-1/2, ..., +-1/2) in the orthogonal basis, so putting the field strength
    in the Cartan with eigenvalues x_1 ... x_r and summing over sign patterns
    gives the traces directly. Writing t_2 and t_4 for the vector traces,
    which are 2 sum x_i^2 and 2 sum x_i^4,

        tr_S F^2 = (d_S / 8) t_2
        tr_S F^4 = -(d_S / 16) t_4 + (3 d_S / 64) t_2^2

    with d_S the dimension of the spinor. The odd terms drop out of the sign
    sum, and the cross terms drop out because summing a product of an
    incomplete set of signs over all patterns gives zero. Both the even and
    odd rank cases land on the same formula once written in terms of d_S,
    which is the reason for writing it that way.

    so(8) is the exception, and it is the one place a table is needed. There
    the incomplete-set argument fails: the quartic expansion reaches a term
    involving all four signs at once, which does not cancel. That term is the
    Pfaffian, the extra invariant so(8) has and no other so(N) does, and
    triality is the statement that the two spinors are then indistinguishable
    from the vector. So they take the vector's coefficients.

    Returns
    -------
    (dim, A, B, C)
    """
    n = int(n)
    if n == 8:
        return (8, F(1), F(1), F(0))
    d = 2 ** ((n - 1) // 2)
    return (d, F(d, 8), F(-d, 16), F(3 * d, 64))


def reality(name, rep):
    r"""Whether a representation is real, complex or pseudo-real.

    This decides which multiplicities are allowed. A hypermultiplet in a
    pseudo-real representation can be halved -- the reality condition can be
    imposed on half of it -- and that is why e7 on a -7 curve carries half a
    56 and why the answer is exact rather than a rounding artefact. For a
    real or complex representation there is no such thing as half a
    hypermultiplet, so a fractional multiplicity there means the
    configuration does not exist.

    The so(N) spinors follow the eightfold pattern: real for N congruent to
    0, 1 or 7 mod 8, pseudo-real for 3, 4 or 5, complex for 2 or 6.
    """
    data = algebra_data(name)
    if rep not in data["reps"]:
        raise ValueError("%s has no representation %r" % (name, rep))
    if rep == data["adjoint"]:
        return "real"
    s = str(name).strip().lower().replace(" ", "")
    if s in ("e6", "e_6"):
        return "complex"
    if s in ("e7", "e_7"):
        return "pseudoreal"
    if s in ("f4", "f_4", "g2", "g_2"):
        return "real"
    kind, n = _parse_classical(s)
    if kind == "sp":
        return "pseudoreal" if rep == "fund" else "real"
    if kind == "su":
        if n == 2:
            return "pseudoreal" if rep == "fund" else "real"
        if rep == "antisym" and n == 4:
            return "real"                      # the 6 of su(4) = vector of so(6)
        return "complex"
    if kind == "so":
        if rep == "vector":
            return "real"
        m = n % 8
        if m in (0, 1, 7):
            return "real"
        if m in (3, 4, 5):
            return "pseudoreal"
        return "complex"
    return "complex"


def _parse_classical(s):
    for kind in ("su", "so", "sp"):
        if s.startswith(kind):
            rest = s[len(kind):].strip("()_")
            try:
                return kind, int(rest)
            except ValueError:
                raise ValueError("cannot read a rank out of %r" % (s,))
    raise ValueError("unknown algebra %r" % (s,))


#: Default matter representations searched by :func:`matter_content`, chosen
#: so that the anomaly system has as many unknowns as it has independent
#: equations for each algebra. Spinors of so(N) are deliberately absent: their
#: dimension and trace coefficients depend on N mod 8 and are not tabulated
#: here, so a model needing them is reported as unsolved rather than solved
#: wrongly.
DEFAULT_REPS = {
    "su2": ["fund"], "su3": ["fund"],
    "su": ["fund", "antisym"],
    "so": ["vector", "spinor"],
    "sp": ["fund", "antisym"],
    "e6": ["27"], "e7": ["56"], "e8": [], "f4": ["26"], "g2": ["7"],
}


def _default_reps(name, data):
    s = str(name).strip().lower().replace(" ", "")
    if s in DEFAULT_REPS:
        return list(DEFAULT_REPS[s])
    for kind in ("su", "so", "sp"):
        if s.startswith(kind):
            reps = [r for r in DEFAULT_REPS[kind] if r in data["reps"]]
            # so(8) is the one place a default has to be trimmed. Triality
            # makes its spinors carry the vector's trace coefficients exactly,
            # so a spinor column would duplicate the vector column and leave
            # the anomaly system underdetermined -- not because the physics is
            # ambiguous but because the two representations are
            # indistinguishable to it. Ask for them explicitly if you want the
            # split; the total is what the conditions fix.
            if s.startswith("so") and _parse_classical(s)[1] == 8:
                reps = [r for r in reps if r != "spinor"]
            return reps
    raise ValueError("no default representations for %r" % (name,))


# ---------------------------------------------------------------------------
# the six-dimensional anomaly conditions
# ---------------------------------------------------------------------------


def anomaly_equations(name, self_intersection, genus=0, reps=None):
    r"""The gauge anomaly conditions as a linear system in the multiplicities.

    For a simple algebra on an irreducible divisor D in the base, with matter
    multiplicities x_R, anomaly cancellation in six dimensions requires

        lam ( A_adj - sum_R x_R A_R )   =  6 K . D
        lam^2 ( sum_R x_R C_R - C_adj ) =  3 D . D
        B_adj                           =  sum_R x_R B_R

    the third only when the algebra has an independent quartic Casimir. The
    adjoint appears on both sides of the first two: once as the vector
    multiplet, and once as matter with multiplicity equal to the genus of D,
    since a curve of genus g carries g adjoint hypermultiplets.

    ``K . D`` is not independent: the adjunction formula fixes it as
    ``2g - 2 - D.D``.

    Returns
    -------
    (rows, rhs, unknowns)
        ``rows`` a list of coefficient lists over :class:`~fractions.Fraction`,
        ``rhs`` the right-hand sides, ``unknowns`` the representation names in
        the order of the columns.
    """
    data = algebra_data(name)
    reps = list(reps) if reps is not None else _default_reps(name, data)
    adj = data["adjoint"]
    unknown = [r for r in reps if r != adj]
    for r in unknown:
        if r not in data["reps"]:
            raise ValueError("%s has no representation %r; it has %s"
                             % (name, r, sorted(data["reps"])))

    lam = F(data["lam"])
    g = int(genus)
    n = int(self_intersection)
    KD = 2 * g - 2 - n                      # adjunction

    _, A_adj, B_adj, C_adj = data["reps"][adj]

    rows, rhs = [], []

    # lam ( A_adj - g A_adj - sum x_R A_R ) = 6 K.D
    rows.append([lam * data["reps"][r][1] for r in unknown])
    rhs.append(lam * A_adj * (1 - g) - 6 * KD)

    # lam^2 ( g C_adj + sum x_R C_R - C_adj ) = 3 D.D
    rows.append([lam * lam * data["reps"][r][3] for r in unknown])
    rhs.append(lam * lam * C_adj * (1 - g) + 3 * n)

    # B_adj = g B_adj + sum x_R B_R, only where the quartic Casimir is there
    if B_adj is not None and all(data["reps"][r][2] is not None
                                 for r in unknown):
        rows.append([data["reps"][r][2] for r in unknown])
        rhs.append(B_adj * (1 - g))

    return rows, rhs, unknown


def _solve_exact(rows, rhs):
    """Exact Gaussian elimination. Returns the unique solution, or raises.

    Rational throughout, so a half-hypermultiplet comes out as
    ``Fraction(1, 2)`` and not as ``0.49999999999999994``.
    """
    m, ncol = len(rows), (len(rows[0]) if rows else 0)
    A = [[F(x) for x in row] + [F(rhs[i])] for i, row in enumerate(rows)]
    pivots = []
    r = 0
    for c in range(ncol):
        p = next((i for i in range(r, m) if A[i][c] != 0), None)
        if p is None:
            continue
        A[r], A[p] = A[p], A[r]
        inv = A[r][c]
        A[r] = [v / inv for v in A[r]]
        for i in range(m):
            if i != r and A[i][c] != 0:
                f = A[i][c]
                A[i] = [A[i][j] - f * A[r][j] for j in range(ncol + 1)]
        pivots.append(c)
        r += 1
        if r == m:
            break
    for i in range(r, m):
        if all(A[i][j] == 0 for j in range(ncol)) and A[i][ncol] != 0:
            raise ValueError("the anomaly conditions have no solution")
    if len(pivots) < ncol:
        raise ValueError(
            "the anomaly conditions do not determine the multiplicities: "
            "%d independent equations for %d unknowns" % (len(pivots), ncol))
    x = [F(0)] * ncol
    for i, c in enumerate(pivots):
        x[c] = A[i][ncol]
    return x


def matter_content(name, self_intersection, genus=0, reps=None):
    r"""The matter on a divisor, from anomaly cancellation alone.

    Six-dimensional anomaly cancellation is restrictive enough to fix the
    charged spectrum once the algebra and the divisor class are given. There
    is nothing to look up.

    Parameters
    ----------
    name : str
        The gauge algebra, e.g. ``"su(5)"``.
    self_intersection : int
        D . D in the base.
    genus : int, optional
        The genus of D; it contributes ``genus`` adjoint hypermultiplets.
    reps : list of str, optional
        Which representations to allow. The default per algebra is
        :data:`DEFAULT_REPS`.

    Returns
    -------
    dict
        ``matter`` mapping representation name to multiplicity (a
        :class:`~fractions.Fraction`, so half-hypermultiplets are exact),
        ``charged_dim`` the total number of charged hypermultiplets,
        ``algebra``, ``dim``, ``rank``.

    Raises
    ------
    ValueError
        If the conditions have no solution, do not determine one, or force a
        negative multiplicity. All three are meaningful: they say the algebra
        cannot sit on that divisor with those representations.

    Examples
    --------
    SU(N) on a -1 curve, the standard N + 8 fundamentals and one
    antisymmetric:

    >>> m = matter_content("su(5)", -1)["matter"]
    >>> int(m["fund"]), int(m["antisym"])
    (13, 1)

    E7 on a -7 curve, where the answer is half a hypermultiplet:

    >>> matter_content("e7", -7)["matter"]["56"]
    Fraction(1, 2)
    """
    data = algebra_data(name)
    rows, rhs, unknown = anomaly_equations(name, self_intersection, genus,
                                           reps)
    if unknown:
        x = _solve_exact(rows, rhs)
    else:
        x = []
        for row, b in zip(rows, rhs):
            if b != 0:
                raise ValueError(
                    "%s admits no matter in the representations considered, "
                    "and the anomaly conditions are then violated on a curve "
                    "of genus %d and self-intersection %d"
                    % (name, genus, self_intersection))

    matter = {}
    for r, v in zip(unknown, x):
        if v < 0:
            raise ValueError(
                "anomaly cancellation forces %s multiplicity %s for %s on a "
                "genus-%d curve of self-intersection %d, which is not a "
                "spectrum" % (r, v, name, genus, self_intersection))
        if v != 0:
            if v.denominator == 2 and reality(name, r) != "pseudoreal":
                raise ValueError(
                    "anomaly cancellation wants %s of %s with multiplicity "
                    "%s, but %s is %s, not pseudo-real, so there is no half "
                    "hypermultiplet to have. The configuration does not "
                    "exist." % (r, name, v, r, reality(name, r)))
            if v.denominator > 2:
                raise ValueError(
                    "anomaly cancellation wants %s of %s with multiplicity "
                    "%s. Multiplicities are integers, or halves for a "
                    "pseudo-real representation; nothing smaller is a "
                    "spectrum." % (r, name, v))
            matter[r] = v
    if genus:
        matter[data["adjoint"]] = F(genus)

    charged = sum(data["reps"][r][0] * v for r, v in matter.items())
    return {"algebra": name, "matter": matter,
            "charged_dim": charged,
            "dim": data["dim"], "rank": data["rank"]}


def check_anomalies(name, self_intersection, genus, matter):
    """Residuals of the gauge anomaly conditions for a given spectrum.

    The counterpart of :func:`matter_content`: instead of solving for the
    multiplicities, take them as given and report by how much each condition
    fails. Every residual zero means the spectrum is anomaly free.

    Returns
    -------
    dict
        ``residuals`` (a list of Fractions), ``ok``, and ``labels``.
    """
    data = algebra_data(name)
    reps = list(matter)
    rows, rhs, unknown = anomaly_equations(name, self_intersection, genus,
                                           reps + [data["adjoint"]])
    res = []
    for row, b in zip(rows, rhs):
        res.append(sum((c * F(matter[r]) for c, r in zip(row, unknown)),
                       F(0)) - F(b))
    labels = ["A: lam (A_adj - sum x A) = 6 K.D",
              "C: lam^2 (sum x C - C_adj) = 3 D.D",
              "B: B_adj = sum x B"][:len(res)]
    return {"residuals": res, "ok": all(r == 0 for r in res),
            "labels": labels}


def matter_free_algebras(candidates=None):
    r"""Which algebras can sit on a rational curve with no charged matter.

    This is the derivation behind :data:`NON_HIGGSABLE`. Set every
    multiplicity to zero and the two anomaly conditions become two independent
    determinations of the self-intersection,

        D.D = -2 - lam A_adj / 6   and   D.D = -lam^2 C_adj / 3 ,

    which agree only for a few algebras, and only at integer values for fewer
    still. Everything else needs charged matter to cancel its anomalies.

    Returns
    -------
    dict
        Algebra name -> self-intersection, for the algebras that survive.

    Examples
    --------
    >>> sorted(matter_free_algebras().items(), key=lambda kv: kv[1])
    [('e8', -12), ('e7', -8), ('e6', -6), ('f4', -5), ('so(8)', -4), ('su(3)', -3)]
    """
    if candidates is None:
        candidates = (["su(%d)" % n for n in range(2, 13)]
                      + ["so(%d)" % n for n in range(7, 17)]
                      + ["sp(%d)" % n for n in range(1, 7)]
                      + ["g2", "f4", "e6", "e7", "e8"])
    out = {}
    for name in candidates:
        try:
            data = algebra_data(name)
        except ValueError:
            continue
        adj = data["adjoint"]
        _, A_adj, _, C_adj = data["reps"][adj]
        lam = F(data["lam"])
        n_A = -2 - lam * A_adj / 6
        n_C = -lam * lam * C_adj / 3
        if n_A == n_C and n_A.denominator == 1:
            out[name] = int(n_A)
    return out


#: Non-Higgsable clusters on a single rational curve, from Morrison and
#: Taylor. A curve of self-intersection -n in the base forces this algebra on
#: the generic Weierstrass model: there is no complex structure for which the
#: fibre is less singular, so the gauge symmetry cannot be Higgsed away.
#:
#: The matter is not stored here; :func:`matter_content` derives it, and for
#: n in {3, 4, 5, 6, 8, 12} the answer is that there is none. The -7 case is
#: the interesting one: e7 with half a hypermultiplet in the 56.
NON_HIGGSABLE = {
    1: None, 2: None,
    3: "su(3)", 4: "so(8)", 5: "f4", 6: "e6", 7: "e7", 8: "e7",
    9: "e8", 10: "e8", 11: "e8", 12: "e8",
}


def non_higgsable(n):
    """The algebra and matter forced on a curve of self-intersection -n.

    Parameters
    ----------
    n : int
        Positive, so the curve has self-intersection ``-n``.

    Returns
    -------
    dict
        ``algebra`` (None for n <= 2), ``matter``, ``blowups``, ``note``.

    Notes
    -----
    For n in 9, 10, 11 the algebra is e8, but the Weierstrass model is
    non-minimal at ``12 - n`` points of the curve, which must be blown up
    before the model is a Calabi-Yau. Only n = 12 gives e8 on an unmodified
    base. ``blowups`` records this.
    """
    n = int(n)
    if n < 1:
        raise ValueError("a non-Higgsable cluster needs a curve of negative "
                         "self-intersection; got -%d" % n)
    if n > 12:
        raise ValueError(
            "a curve of self-intersection -%d cannot sit in the base of an "
            "elliptic Calabi-Yau threefold: below -12 the Weierstrass model "
            "is non-minimal along the whole curve" % n)
    alg = NON_HIGGSABLE[n]
    if alg is None:
        return {"algebra": None, "matter": {}, "blowups": 0,
                "note": "a curve of self-intersection -%d forces no gauge "
                        "algebra" % n}
    # For n in 9, 10, 11 the anomaly conditions have no solution on the curve
    # as it stands, and that is the correct answer rather than a failure: the
    # Weierstrass model is non-minimal at 12 - n points, and those points must
    # be blown up before the model is a Calabi-Yau at all. Each blowup drops
    # the self-intersection by one, so the curve that finally carries e8 has
    # self-intersection -12, where the conditions do have a solution.
    blowups = max(0, 12 - n) if n >= 9 else 0
    matter = matter_content(alg, -(n + blowups))["matter"]
    note = ""
    if blowups:
        note = ("the Weierstrass model is non-minimal at %d points of this "
                "curve; the base must be blown up there, after which the "
                "curve has self-intersection -12 and the anomaly conditions "
                "close" % blowups)
    return {"algebra": alg, "matter": matter, "blowups": blowups,
            "self_intersection_after_blowup": -(n + blowups),
            "note": note}


# ---------------------------------------------------------------------------
# bases
# ---------------------------------------------------------------------------


class Base(object):
    r"""A complex surface, as an intersection form and a canonical class.

    F-theory in six dimensions needs remarkably little about the base: the
    intersection pairing on H^2, the canonical class, and which classes are
    effective. The first two are all that the anomaly conditions use, and they
    are what this class carries.

    Parameters
    ----------
    intersection : 2-d array of int
        The pairing D_i . D_j in a chosen basis of divisor classes.
    canonical : list of int
        K_B in the same basis.
    name : str, optional

    Attributes
    ----------
    h11 : int
        The rank of the basis, so the number of tensor multiplets is
        ``h11 - 1``.

    Notes
    -----
    The class does not verify that a surface with the given form exists. It
    does check the numerical consequence that matters, ``K^2 = 10 - h^{1,1}``,
    which holds for the rational surfaces that F-theory bases are, and warns
    through :meth:`consistent` rather than refusing, since a non-rational base
    is a legitimate if unusual thing to want.
    """

    #: How the base was built: None for a hand-supplied form, otherwise
    #: ``"P2"``, ``"hirzebruch"`` or ``"del_pezzo"``. Code that needs to know
    #: whether a surface is Hirzebruch -- the non-Higgsable section, the
    #: heterotic dual -- reads this rather than parsing :attr:`name`, so that
    #: a hand-built base with an unlucky name is not mistaken for one.
    kind = None

    #: The n of F_n or the k of dP_k, when :attr:`kind` says there is one.
    parameter = None

    def __init__(self, intersection, canonical, name=None):
        self.form = np.asarray(intersection, dtype=object)
        if self.form.ndim != 2 or self.form.shape[0] != self.form.shape[1]:
            raise ValueError("the intersection form must be square")
        if not np.array_equal(self.form, self.form.T):
            raise ValueError("the intersection form must be symmetric")
        self.K = np.asarray(canonical, dtype=object)
        if self.K.shape != (self.form.shape[0],):
            raise ValueError("K must have one entry per basis divisor")
        self.h11 = int(self.form.shape[0])
        self.name = name or "base"

    # -- constructors ------------------------------------------------------

    @classmethod
    def P2(cls):
        """The projective plane. h^{1,1} = 1, K = -3H, K^2 = 9, T = 0."""
        b = cls([[1]], [-3], "P^2")
        b.kind = "P2"
        return b

    @classmethod
    def hirzebruch(cls, n):
        r"""The Hirzebruch surface F_n.

        Basis (s, f) with s the section of self-intersection -n, f the fibre.
        K = -2s - (n+2)f, K^2 = 8, T = 1. For n >= 3 the section is a curve of
        self-intersection -n and carries a non-Higgsable gauge algebra.
        """
        n = int(n)
        if n < 0:
            raise ValueError("F_n needs n >= 0")
        b = cls([[-n, 1], [1, 0]], [-2, -(n + 2)], "F_%d" % n)
        b.kind, b.parameter = "hirzebruch", n
        return b

    @classmethod
    def del_pezzo(cls, k):
        r"""The del Pezzo surface dP_k, P^2 blown up at k generic points.

        Basis (H, E_1, ..., E_k) with H^2 = 1, E_i^2 = -1, K = -3H + sum E_i,
        K^2 = 9 - k, T = k.
        """
        k = int(k)
        if not 0 <= k <= 8:
            raise ValueError("dP_k is a del Pezzo surface only for 0 <= k <= 8")
        form = [[0] * (k + 1) for _ in range(k + 1)]
        form[0][0] = 1
        for i in range(1, k + 1):
            form[i][i] = -1
        b = cls(form, [-3] + [1] * k, "dP_%d" % k)
        b.kind, b.parameter = "del_pezzo", k
        return b

    # -- the arithmetic ----------------------------------------------------

    def dot(self, D1, D2):
        """The intersection number D1 . D2."""
        a = np.asarray(D1, dtype=object)
        b = np.asarray(D2, dtype=object)
        return int(a.dot(self.form).dot(b))

    @property
    def K2(self):
        """K_B . K_B, equal to 9 - T for a rational surface."""
        return self.dot(self.K, self.K)

    @property
    def T(self):
        """The number of tensor multiplets, h^{1,1}(B) - 1."""
        return self.h11 - 1

    @property
    def chi_top(self):
        """The topological Euler characteristic, 2 + h^{1,1} for rational B."""
        return 2 + self.h11

    def genus(self, D):
        """The arithmetic genus of a divisor, from 2g - 2 = D.(D + K)."""
        two_g_minus_2 = self.dot(D, D) + self.dot(D, self.K)
        if two_g_minus_2 % 2:
            raise ValueError("D.(D+K) = %d is odd, so D has no integral genus"
                             % two_g_minus_2)
        return two_g_minus_2 // 2 + 1

    def consistent(self):
        """Whether K^2 = 10 - h^{1,1}, as it is for every rational surface."""
        return self.K2 == 10 - self.h11

    def h0_anticanonical(self, n):
        r"""h^0(B, -nK), by Riemann-Roch.

        For a rational surface with -nK having no higher cohomology,

            h^0(-nK) = chi(O) + (1/2)(-nK).(-nK - K) = 1 + n(n+1) K^2 / 2 .

        The Weierstrass coefficients f and g are sections of -4K and -6K, so
        this counts them, and it is where the 272 - 29T of the generic model
        comes from.
        """
        n = int(n)
        return 1 + n * (n + 1) * self.K2 // 2

    def chi_tangent(self):
        r"""chi(T_B) = h^0(T_B) - h^1(T_B) + h^2(T_B) = (7K^2 - 5 chi_top)/6.

        This is the number of Weierstrass coefficients that are not moduli:
        the automorphisms of B act on them, and the deformations of B itself
        give some back. Using the Euler characteristic rather than h^0(T_B)
        alone is what makes the count come out right for F_2, whose
        automorphism group is one dimension larger than F_0's and which
        compensates by deforming to F_0.
        """
        num = 7 * self.K2 - 5 * self.chi_top
        if num % 6:
            raise ValueError("chi(T_B) is not integral for this base")
        return num // 6

    def __repr__(self):
        return "<Base %s, h^{1,1}=%d, K^2=%d, T=%d>" % (
            self.name, self.h11, self.K2, self.T)


class ProductBase(object):
    r"""A base that is a product of projective spaces, of any dimension.

    :class:`Base` carries a surface as an intersection form, which is all six
    dimensional F-theory needs. Four dimensional F-theory compactifies on an
    elliptic fourfold, so the base is a threefold, and the quantities that
    matter there are Chern numbers rather than a single intersection matrix.
    Products of projective spaces are enough to reach the standard examples
    and are simple enough to do exactly: the cohomology is generated by one
    hyperplane class per factor with H_i^{n_i+1} = 0, and

        c(B) = prod_i (1 + H_i)^{n_i + 1} .

    Parameters
    ----------
    dims : list of int
        The n_i, so ``[3]`` is P^3 and ``[1, 2]`` is P^1 x P^2.

    Examples
    --------
    >>> B = ProductBase([3])
    >>> B.dim, B.h11, B.chern_number([1, 1, 1]), B.chern_number([1, 2])
    (3, 1, 64, 24)
    """

    def __init__(self, dims, name=None):
        self.dims = [int(d) for d in dims]
        if any(d < 1 for d in self.dims):
            raise ValueError("each factor needs positive dimension")
        self.dim = sum(self.dims)
        self.h11 = len(self.dims)
        self.name = name or " x ".join("P^%d" % d for d in self.dims)

    def __repr__(self):
        return "<ProductBase %s, dim %d>" % (self.name, self.dim)

    # -- the cohomology ring, truncated ------------------------------------

    def _mul(self, a, b):
        out = {}
        for ea, ca in a.items():
            for eb, cb in b.items():
                e = tuple(x + y for x, y in zip(ea, eb))
                if any(e[i] > self.dims[i] for i in range(self.h11)):
                    continue
                out[e] = out.get(e, 0) + ca * cb
        return {e: c for e, c in out.items() if c}

    def chern(self):
        """The total Chern class, as a dict from exponents to coefficients."""
        total = {tuple([0] * self.h11): 1}
        for i, n in enumerate(self.dims):
            e = [0] * self.h11
            e[i] = 1
            h = {tuple([0] * self.h11): 1, tuple(e): 1}
            factor = {tuple([0] * self.h11): 1}
            for _ in range(n + 1):
                factor = self._mul(factor, h)
            total = self._mul(total, factor)
        return total

    def chern_class(self, k):
        """The degree-k part of the total Chern class."""
        return {e: c for e, c in self.chern().items() if sum(e) == k}

    def integrate(self, poly):
        """The coefficient of prod_i H_i^{n_i}, which is the integral."""
        return int(poly.get(tuple(self.dims), 0))

    def chern_number(self, parts):
        """An integral of a product of Chern classes, e.g. ``[1, 2]`` for c1 c2."""
        if sum(parts) != self.dim:
            raise ValueError(
                "c_%s has degree %d on a %d-fold; the integrand must have "
                "degree %d" % (parts, sum(parts), self.dim, self.dim))
        out = {tuple([0] * self.h11): 1}
        for k in parts:
            out = self._mul(out, self.chern_class(int(k)))
        return self.integrate(out)

    # -- the data the Weierstrass model needs ------------------------------

    def h0_anticanonical(self, k):
        r"""h^0(B, -kK), a product of binomials.

        With -K = sum_i (n_i + 1) H_i, the sections of -kK are the
        multi-homogeneous polynomials of degree k(n_i + 1) in each factor.
        """
        from math import comb
        out = 1
        for n in self.dims:
            out *= comb(k * (n + 1) + n, n)
        return out

    def chi_tangent(self):
        r"""chi(T_B) = sum_i ((n_i + 1)^2 - 1), the automorphisms.

        A product of projective spaces has no deformations and no higher
        cohomology of its tangent sheaf, so the Euler characteristic is just
        h^0, which is the dimension of prod_i PGL(n_i + 1).
        """
        return sum((n + 1) ** 2 - 1 for n in self.dims)

    @property
    def K2(self):
        """c_1^2 for a surface, so that a two-factor product matches Base."""
        if self.dim != 2:
            raise ValueError("K^2 is a surface quantity; this base has "
                             "dimension %d" % self.dim)
        return self.chern_number([1, 1])

    @property
    def chi_top(self):
        """The topological Euler characteristic, prod_i (n_i + 1)."""
        out = 1
        for n in self.dims:
            out *= n + 1
        return out


def fourfold_euler(base):
    r"""chi of the smooth Weierstrass fourfold over a threefold base.

        chi(X) = 12 int_B c_1 c_2 + 360 int_B c_1^3

    On P^3 that is 12 x 24 + 360 x 64 = 23328, and the D3-brane tadpole
    chi/24 is 972.
    """
    if base.dim != 3:
        raise ValueError("an elliptic fourfold needs a threefold base; this "
                         "one has dimension %d" % base.dim)
    return 12 * base.chern_number([1, 2]) + 360 * base.chern_number([1, 1, 1])


def fourfold_hodge(base):
    r"""Hodge numbers of the smooth Weierstrass fourfold, and a check on them.

    The dictionary is the same as in six dimensions, one dimension up:

        h^{1,1}(X) = h^{1,1}(B) + 1
        h^{3,1}(X) = h^0(-4K) + h^0(-6K) - chi(T_B) - 1
        h^{2,1}(X) = 0     for a base with no odd cohomology

    and then chi is determined, because a Calabi-Yau fourfold satisfies

        chi = 6 (8 + h^{1,1} + h^{3,1} - h^{2,1}) .

    That is a completely different computation from :func:`fourfold_euler`,
    which is a Chern number of the fibration. On P^3 the moduli count gives
    h^{3,1} = 969 + 2925 - 15 - 1 = 3878 and hence chi = 6 x 3888 = 23328,
    which is what the Chern numbers give. The agreement is the check.

    Returns
    -------
    dict
        ``h11``, ``h21``, ``h31``, ``h22``, ``euler``, ``euler_chern`` and
        ``agree``.
    """
    if base.dim != 3:
        raise ValueError("this is the fourfold case; the base is a threefold")
    h11 = base.h11 + 1
    h21 = 0
    h31 = (base.h0_anticanonical(4) + base.h0_anticanonical(6)
           - base.chi_tangent() - 1)
    h22 = 2 * (22 + 2 * h11 + 2 * h31 - h21)
    chi = 6 * (8 + h11 + h31 - h21)
    chern = fourfold_euler(base)
    return {"h11": h11, "h21": h21, "h31": h31, "h22": h22,
            "euler": chi, "euler_chern": chern, "agree": chi == chern,
            "d3_tadpole": Fraction(chern, 24)}


def weierstrass_euler(base):
    r"""chi of the smooth Weierstrass threefold over ``base``, which is -60 K^2.

    An independent route to the Euler characteristic: not through the Hodge
    numbers, and not through the anomaly, but from the Chern classes of the
    elliptic fibration. For a smooth Weierstrass model over a surface B,

        chi(X) = -60 int_B c_1(B)^2 = -60 K_B^2 .

    On P^2 that is -540 and on any Hirzebruch surface -480, which is what
    :meth:`FTheory6D.euler_characteristic` gives from the spectrum.

    The agreement holds exactly where the generic model is smooth. Once the
    base forces a gauge algebra -- F_n for n >= 3 -- the Weierstrass model is
    singular and the smooth Calabi-Yau is its resolution, which has different
    Chern classes and a different Euler characteristic. F_12 gives -960 from
    the spectrum against -480 here, and the difference is the resolution of
    the e8 fibre. So this function is a check on the smooth cases and a
    statement about the singular ones, not a shortcut for either.
    """
    return -60 * base.K2


def weierstrass_moduli(base):
    r"""Complex structure moduli of the generic Weierstrass model over ``base``.

    The Weierstrass model is fixed by f in H^0(-4K) and g in H^0(-6K), modulo
    the automorphisms of the base and the rescaling (f, g) -> (t^4 f, t^6 g).
    So

        n_moduli = h^0(-4K) + h^0(-6K) - chi(T_B) - 1 .

    On P^2 that is 91 + 190 - 8 - 1 = 272, and in general it is 272 - 29T,
    which is exactly what the gravitational anomaly demands of a model with no
    gauge group. The agreement is not built in anywhere: the left side is
    Riemann-Roch on a surface and the right side is a one-loop condition in
    six dimensions. :func:`pyCICY.theories.ftheory` tests it.
    """
    return (base.h0_anticanonical(4) + base.h0_anticanonical(6)
            - base.chi_tangent() - 1)


# ---------------------------------------------------------------------------
# the theory
# ---------------------------------------------------------------------------


@register
class FTheory6D(Theory):
    r"""F-theory on an elliptic Calabi-Yau threefold: six-dimensional (1,0).

    Parameters
    ----------
    base : Base or str
        The base surface. A string is passed to :meth:`Base.hirzebruch` or
        :meth:`Base.del_pezzo` or :meth:`Base.P2` by name: ``"P2"``,
        ``"F3"``, ``"dP4"``.
    gauge : list of (algebra, divisor) or (algebra, divisor, matter), optional
        Gauge algebras and the divisor classes they sit on. When the matter is
        omitted it is derived from the anomaly conditions. When ``gauge`` is
        omitted entirely the non-Higgsable content of the base is used, which
        is the generic model.
    name : str, optional

    Notes
    -----
    ``self.X`` is None. The generic Weierstrass model is a hypersurface in a
    P^{2,3,1} bundle over B, which is not a complete intersection in a product
    of projective spaces and so is not a CICY. That is a statement about the
    ambient space, not about the physics: everything this class computes comes
    from the base and the anomaly conditions, and none of it needs the
    threefold as a CICY. Use :func:`obvious_fibrations` for the CICYs that do
    fibre in elliptic curves.

    Examples
    --------
    >>> m = FTheory6D("P2")
    >>> m.spectrum()["T"], m.spectrum()["H"], m.spectrum()["V"]
    (0, 273, 0)
    >>> m.hodge_numbers()
    (2, 272)

    The base F_12, where the -12 section forces e8:

    >>> m = FTheory6D("F12")
    >>> m.gauge_group()
    'e8'
    >>> m.hodge_numbers()
    (11, 491)
    """

    key = "f-theory-6d"

    def __init__(self, base, gauge=None, name=None):
        Theory.__init__(self, None, name=name)
        self.base = base if isinstance(base, Base) else _named_base(base)
        if gauge is None:
            gauge = _generic_gauge(self.base)
        self.gauge = []
        for item in gauge:
            if len(item) == 2:
                alg, D = item
                matter = matter_content(alg, self.base.dot(D, D),
                                        self.base.genus(D))["matter"]
            elif len(item) == 3:
                alg, D, matter = item
                matter = {k: F(v) for k, v in dict(matter).items()}
            else:
                raise ValueError("each gauge factor is (algebra, divisor) or "
                                 "(algebra, divisor, matter)")
            self.gauge.append((alg, list(D), matter))

    # -- reporting ---------------------------------------------------------

    def geometry(self):
        return "the generic elliptic threefold over %s" % self.base.name

    def gauge_group(self):
        """The six-dimensional gauge algebra, as a string."""
        if not self.gauge:
            return "trivial"
        return " x ".join(a for a, _, _ in self.gauge)

    # -- matter where the divisors meet ------------------------------------

    def bifundamentals(self):
        r"""Matter localised where two gauge divisors intersect.

        The mixed anomaly condition for two simple factors is

            b_i . b_j = sum_{RS} x^{ij}_{RS} A_R A_S

        and the defining representation of every algebra tabulated here --
        the fundamental of su and sp, the vector of so, the 27, 56, 26 and 7
        of e6, e7, f4 and g2 -- has A = 1. So a pair of gauge divisors
        meeting in ``b_i . b_j`` points carries that many bifundamentals of
        the two defining representations, and the intersection number *is*
        the multiplicity.

        These are not extra states. They are already inside the per-divisor
        counts: a bifundamental (d_i, d_j) looks, to factor i alone, like d_j
        copies of its defining representation. That is why
        :meth:`spectrum` has to subtract the overlap rather than add
        anything, and why the count of plain fundamentals on each divisor is
        what is left after the shared ones are removed.

        Returns
        -------
        list of dict
            ``factors`` (the pair of indices), ``algebras``, ``multiplicity``,
            ``reps``, ``dim`` (states per bifundamental) and ``total``.
        """
        out = []
        for i in range(len(self.gauge)):
            for j in range(i + 1, len(self.gauge)):
                mult = self.base.dot(self.gauge[i][1], self.gauge[j][1])
                if mult == 0:
                    continue
                if mult < 0:
                    raise ValueError(
                        "the gauge divisors %s and %s have intersection "
                        "number %d. A negative intersection between distinct "
                        "irreducible divisors is not a matter multiplicity, "
                        "so these cannot both carry gauge algebras as given."
                        % (self.gauge[i][1], self.gauge[j][1], mult))
                di, ri = self._defining(i)
                dj, rj = self._defining(j)
                out.append({"factors": (i, j),
                            "algebras": (self.gauge[i][0], self.gauge[j][0]),
                            "reps": (ri, rj), "multiplicity": mult,
                            "dim": di * dj, "total": mult * di * dj})
        return out

    def _defining(self, i):
        """(dimension, name) of the defining representation of factor i."""
        data = algebra_data(self.gauge[i][0])
        for r in ("fund", "vector", "27", "56", "26", "7"):
            if r in data["reps"]:
                return data["reps"][r][0], r
        raise ValueError(
            "%s has no representation of index one below the adjoint, so "
            "bifundamental matter with it is not covered here"
            % self.gauge[i][0])

    # -- the exact part ----------------------------------------------------

    def spectrum(self):
        r"""The massless spectrum. Exact, from anomaly cancellation.

        Six-dimensional (1,0) supergravity is anomalous unless

            H - V + 29 T = 273 ,

        with H hypermultiplets, V vector multiplets and T tensor multiplets.
        T and V come from the geometry, H_charged from the anomaly conditions
        on each gauge divisor, and the neutral hypermultiplets are then not a
        separate computation: they are what the gravitational anomaly says
        they must be.

        Returns
        -------
        dict
            ``T``, ``V``, ``H``, ``H_charged``, ``H_neutral``, ``rank``,
            ``gravitational_anomaly`` (zero when cancelled).
        """
        T = self.base.T
        V = 0
        rank = 0
        H_charged = F(0)
        for alg, D, matter in self.gauge:
            data = algebra_data(alg)
            V += data["dim"]
            rank += data["rank"]
            for r, x in matter.items():
                H_charged += data["reps"][r][0] * F(x)
        # Matter at an intersection has been counted once by each of the two
        # divisors that meet there, so remove the duplicate. Adding it in
        # instead would break the gravitational anomaly, which is the check
        # that catches this if it is got wrong.
        shared = 0
        for b in self.bifundamentals():
            shared += b["total"]
        H_charged -= shared
        if H_charged.denominator != 1:
            raise ValueError(
                "the charged hypermultiplet count came out as %s. Half "
                "multiplicities are legitimate for pseudo-real "
                "representations, but the total number of hypermultiplets "
                "must be an integer." % H_charged)
        H_charged = int(H_charged)
        H = 273 - 29 * T + V
        return {"T": T, "V": V, "H": H, "H_charged": H_charged,
                "H_neutral": H - H_charged, "rank": rank,
                "bifundamental_states": int(shared),
                "gravitational_anomaly": H - V + 29 * T - 273}

    def hodge_numbers(self):
        r"""(h^{1,1}, h^{2,1}) of the elliptic threefold.

        The dictionary between the six-dimensional spectrum and the geometry:

            h^{1,1}(X) = h^{1,1}(B) + 1 + rank(G) + rk(MW)
            h^{2,1}(X) = H_neutral - 1

        The first counts the base divisors, the zero section, and one
        exceptional divisor per Cartan generator of the resolved fibre. The
        second is the complex structure moduli, the neutral hypermultiplets
        minus the universal one containing the overall volume.

        The Mordell-Weil rank is taken to be zero, which is the generic case;
        a model with extra sections has extra U(1)s and this understates
        h^{1,1}.
        """
        s = self.spectrum()
        return (self.base.h11 + 1 + s["rank"], s["H_neutral"] - 1)

    def euler_characteristic(self):
        """chi = 2 (h^{1,1} - h^{2,1}) of the elliptic threefold."""
        h11, h21 = self.hodge_numbers()
        return 2 * (h11 - h21)

    def check_anomalies(self):
        """Every anomaly condition, gravitational and gauge, as residuals.

        Returns
        -------
        dict
            ``gravitational``, ``gauge`` (a list, one entry per factor) and
            ``ok``.
        """
        s = self.spectrum()
        gauge = []
        for alg, D, matter in self.gauge:
            r = check_anomalies(alg, self.base.dot(D, D),
                                self.base.genus(D), matter)
            r["algebra"] = alg
            r["divisor"] = list(D)
            gauge.append(r)
        # The mixed condition, and the consistency it demands: the shared
        # matter has to fit inside what each divisor was independently told to
        # carry. If a divisor's own anomaly conditions give fewer copies of
        # its defining representation than the intersections require, the two
        # statements contradict and the model does not exist.
        mixed = []
        for b in self.bifundamentals():
            ok = True
            for k, side in zip(b["factors"], (0, 1)):
                _, _, matter = self.gauge[k]
                _, rep = self._defining(k)
                have = F(matter.get(rep, 0))
                need = sum(F(c["multiplicity"]) * c["dim"]
                           / algebra_data(self.gauge[k][0])["reps"][rep][0]
                           for c in self.bifundamentals()
                           if k in c["factors"])
                if have < need:
                    ok = False
            b = dict(b)
            b["ok"] = ok
            mixed.append(b)
        return {"gravitational": s["gravitational_anomaly"],
                "gauge": gauge, "mixed": mixed,
                "ok": s["gravitational_anomaly"] == 0
                      and all(g["ok"] for g in gauge)
                      and all(m["ok"] for m in mixed)}

    def heterotic_dual(self):
        r"""The heterotic dual, when the base is a Hirzebruch surface.

        F-theory on an elliptic threefold over F_n is dual to the E8 x E8
        heterotic string on K3 with instanton numbers (12 + n, 12 - n). The
        check worth making is at the end of the range: F_12 gives (24, 0), no
        instantons in the second E8, hence an unbroken E8 -- which is exactly
        the non-Higgsable algebra on the -12 section.

        Returns
        -------
        dict or None
            None when the base is not a Hirzebruch surface.
        """
        if self.base.kind != "hirzebruch":
            return None
        n = self.base.parameter
        return {"instantons": (12 + n, 12 - n),
                "unbroken_from_second_E8": NON_HIGGSABLE.get(n),
                "note": "E8 x E8 heterotic on K3 with instanton numbers "
                        "(%d, %d)" % (12 + n, 12 - n)}

    # -- the part that does not exist --------------------------------------

    def holomorphic_yukawa(self, **kw):
        """Always raises :exc:`NoSuchTheory`. There are none in six dimensions.

        Not a gap. Six-dimensional (1,0) supersymmetry admits no
        superpotential: hypermultiplets sit in quaternionic multiplets with no
        gauge-invariant cubic holomorphic coupling available. There is nothing
        to compute, and returning zero would put a physical statement and a
        missing feature in the same slot.

        The four-dimensional case is different, and different in the ordinary
        way: F-theory on a Calabi-Yau fourfold does have Yukawa couplings, they
        are localised at points where matter curves meet, and computing them
        needs G-flux data this package does not carry. See :class:`FTheory4D`.
        """
        raise NoSuchTheory(
            "six-dimensional (1,0) supersymmetry forbids a superpotential, so "
            "this compactification has no Yukawa couplings at all. This is "
            "not a limitation of the computation: contrast "
            "LineBundleModel.holomorphic_yukawa(), which is a coupling that "
            "exists and is not implemented, and physical_yukawa(), which is a "
            "coupling that exists and needs the metric.")

    def physical_yukawa(self, **kw):
        """Always raises :exc:`NoSuchTheory`, for the same reason."""
        return self.holomorphic_yukawa(**kw)

    def fermion_masses(self, **kw):
        """Always raises. In six dimensions there is no Yukawa to make one."""
        raise NoSuchTheory(
            "a fermion mass needs a Yukawa coupling and a vacuum expectation "
            "value; in six dimensions there is no Yukawa coupling to begin "
            "with.")

    def missing_for_physical(self):
        """What a metric would still be needed for, which is not the couplings.

        The spectrum, the gauge algebra and the anomaly conditions are exact.
        What the metric governs here is the hypermultiplet moduli space and
        the normalisation of the kinetic terms, neither of which this package
        computes and neither of which any of the numbers above depend on.
        """
        return [
            "the metric on the hypermultiplet moduli space, which is "
            "quaternionic-Kahler and receives corrections this package does "
            "not compute",
            "the normalisation of the kinetic terms, hence any statement "
            "about physical couplings rather than the spectrum",
            "note that the Yukawa couplings are *not* on this list: in six "
            "dimensions there are none, which is a different statement",
        ]

    def describe(self):
        """A summary, separating the exact spectrum from what it omits."""
        s = self.spectrum()
        h11, h21 = self.hodge_numbers()
        lines = ["%s on %s" % (self.name, self.geometry()),
                 "  gauge algebra    %s" % self.gauge_group(),
                 "  T = %d   V = %d   H = %d  (%d charged, %d neutral)"
                 % (s["T"], s["V"], s["H"], s["H_charged"], s["H_neutral"]),
                 "  H - V + 29T - 273 = %d" % s["gravitational_anomaly"],
                 "  elliptic threefold  h^{1,1} = %d, h^{2,1} = %d, chi = %d"
                 % (h11, h21, 2 * (h11 - h21))]
        for alg, D, matter in self.gauge:
            m = ", ".join("%s x %s" % (v, k) for k, v in sorted(matter.items()))
            lines.append("  %-8s on %s, D^2 = %d, matter: %s"
                         % (alg, list(D), self.base.dot(D, D), m or "none"))
        lines.append("  Yukawa couplings:   none exist in six dimensions")
        lines.append("  not computable here; needs")
        for m in self.missing_for_physical():
            lines.append("     - %s" % m)
        return "\n".join(lines)


def _named_base(spec):
    s = str(spec).strip().replace(" ", "")
    low = s.lower()
    if low in ("p2", "p^2", "cp2"):
        return Base.P2()
    if low.startswith("f") and low[1:].lstrip("_").isdigit():
        return Base.hirzebruch(int(low[1:].lstrip("_")))
    if low.startswith("dp") and low[2:].lstrip("_").isdigit():
        return Base.del_pezzo(int(low[2:].lstrip("_")))
    raise ValueError(
        "cannot read %r as a base; use 'P2', 'F<n>', 'dP<k>' or a Base "
        "instance" % (spec,))


def _generic_gauge(base):
    """Non-Higgsable content of a Hirzebruch or del Pezzo base.

    Only the Hirzebruch surfaces have a curve forced to negative
    self-intersection below -2 in this basis, namely the section of F_n for
    n >= 3. The del Pezzo surfaces have no curve below -1 and so no
    non-Higgsable gauge symmetry, and neither does P^2.
    """
    if base.kind == "hirzebruch":
        n = base.parameter
        if n > 12:
            raise ValueError(
                "F_%d has a section of self-intersection -%d, below the -12 "
                "at which the Weierstrass model goes non-minimal along the "
                "whole curve. There is no elliptic Calabi-Yau over it."
                % (n, n))
        if n in (9, 10, 11):
            raise ValueError(
                "F_%d is not a base for a minimal Weierstrass model: the "
                "model is non-minimal at %d points of the -%d section, which "
                "must be blown up first. The result is no longer a Hirzebruch "
                "surface, so build it as a Base explicitly. The good "
                "Hirzebruch bases are F_0 to F_8 and F_12."
                % (n, 12 - n, n))
        alg = NON_HIGGSABLE.get(n)
        if alg is not None:
            return [(alg, [1, 0])]          # the section s, with s.s = -n
    return []


# ---------------------------------------------------------------------------
# four dimensions
# ---------------------------------------------------------------------------


@register
class FTheory4D(Theory):
    r"""F-theory on an elliptic Calabi-Yau fourfold: four-dimensional N=1.

    This one takes a CICY, because elliptically fibred CICY fourfolds do exist
    and the package can compute their Euler characteristic. What it computes
    is the D3-brane tadpole,

        N_D3 + (1/2) int G ^ G = chi(X) / 24 ,

    an exact integer condition on the flux. Everything past that -- the chiral
    spectrum, the Yukawa couplings, moduli stabilisation -- needs the flux G
    itself, which is a choice of a four-form class this package does not carry
    and cannot enumerate.

    The class exists to make that boundary explicit rather than to hide it.
    :meth:`spectrum` raises, and says what would be needed.

    Parameters
    ----------
    X : CICY or configuration matrix
        A Calabi-Yau fourfold. Whether it is elliptically fibred is checked
        with :func:`obvious_fibrations` and reported, not assumed.
    """

    key = "f-theory-4d"

    @classmethod
    def over(cls, base, name=None):
        """Build the generic Weierstrass fourfold over a threefold base.

        The alternative to handing in a CICY. The generic model is a
        hypersurface in a P^{2,3,1} bundle, which is not a complete
        intersection in a product of projective spaces, so it has no
        configuration matrix -- but its Hodge numbers, Euler characteristic
        and tadpole all follow from the base, exactly.
        """
        obj = cls.__new__(cls)
        Theory.__init__(obj, None, name=name)
        obj.base = base if isinstance(base, ProductBase) else ProductBase(base)
        if obj.base.dim != 3:
            raise ValueError("an elliptic fourfold needs a threefold base")
        obj.fibrations = []
        return obj

    def __init__(self, X, name=None):
        Theory.__init__(self, X, name=name)
        self.base = None
        if self.X.nfold != 4:
            raise ValueError(
                "F-theory in four dimensions compactifies on a Calabi-Yau "
                "fourfold; this configuration is a %d-fold. For a threefold "
                "the compactification is six-dimensional, see FTheory6D."
                % self.X.nfold)
        self.fibrations = obvious_fibrations(self.X.M.tolist())

    def geometry(self):
        if self.base is not None:
            return "the generic elliptic fourfold over %s" % self.base.name
        return self.X.M.tolist()

    def gauge_group(self):
        """Not determined by the fourfold alone."""
        if self.base is not None:
            return ("trivial for the generic Weierstrass model over a smooth "
                    "base; a gauge algebra needs a divisor tuned to a "
                    "Kodaira type")
        return ("determined by the singular fibres over the base threefold, "
                "which the configuration matrix does not resolve")

    def hodge_numbers(self):
        """Hodge numbers of the generic Weierstrass fourfold. Base only."""
        if self.base is None:
            raise NotImplementedError(
                "Hodge numbers of a general Calabi-Yau fourfold beyond h^{1,1} "
                "and h^{3,1} are not computed here; use FTheory4D.over(base) "
                "for the generic Weierstrass model, where they follow from "
                "the base.")
        return fourfold_hodge(self.base)

    def flux(self):
        r"""What the G-flux has to satisfy. Constraints, not a choice.

        Three conditions bound the flux without determining it, and all three
        are exact:

        *Quantisation.* G + c_2(X)/2 must be an integral class. So the flux
        is integrally quantised only when c_2(X) is even, and otherwise
        carries a half-integral shift. The same arithmetic shows in
        :meth:`d3_tadpole`: a non-integral chi/24 is the signal.

        *The tadpole.* N_D3 + (1/2) int G ^ G = chi/24 with N_D3 a
        non-negative integer, so

            int_X G ^ G  <=  chi / 12 ,

        which is a finite bound on an infinite-looking choice. On P^3 that is
        1944.

        *Supersymmetry.* G must be primitive and of Hodge type (2,2), which
        is a condition on the complex structure, not a numerical one.

        The chiral spectrum is an index twisted by G, so none of this gives a
        spectrum. It gives the box the flux lives in.
        """
        chi = self.euler()
        t = Fraction(chi, 24)
        return {"chi": chi, "tadpole": t,
                "max_GG": Fraction(chi, 12),
                "integrally_quantised": t.denominator == 1,
                "conditions": [
                    "G + c_2(X)/2 integral (Witten quantisation)",
                    "N_D3 + (1/2) int G ^ G = chi/24, with N_D3 >= 0",
                    "G primitive and of type (2,2)"],
                "note": "these bound the flux; they do not choose it, and the "
                        "chiral spectrum is an index twisted by the choice"}

    def euler(self):
        """chi of the fourfold, from the CICY or from the base."""
        if self.base is not None:
            return fourfold_euler(self.base)
        return int(self.X.euler_characteristic())

    def is_elliptically_fibred(self):
        """Whether an obvious genus-one fibration was found. Necessary, not
        sufficient: F-theory also wants a section, which this does not test."""
        return bool(self.fibrations)

    def d3_tadpole(self):
        r"""chi(X)/24, the D3-brane tadpole. Exact.

        The tadpole condition is N_D3 + (1/2) int_X G ^ G = chi(X)/24 with
        N_D3 a non-negative integer. So chi must be divisible by 24 when the
        flux is integrally quantised; when it is not, the flux is
        half-integrally shifted, which is a real and allowed possibility, so a
        non-integral value here is reported rather than treated as an error.

        Returns
        -------
        dict
            ``chi``, ``tadpole``, ``integral``, ``note``.
        """
        chi = self.euler()
        t = Fraction(chi, 24)
        return {"chi": chi, "tadpole": t, "integral": t.denominator == 1,
                "note": ("chi/24 is an integer, so integrally quantised flux "
                         "is consistent" if t.denominator == 1 else
                         "chi/24 = %s is not an integer, so the flux must "
                         "carry the half-integral shift by c_2(X)" % t)}

    def spectrum(self):
        """Always raises. The chiral spectrum is a function of the flux."""
        raise NeedsMetric(
            "the chiral spectrum of a four-dimensional F-theory model is an "
            "index twisted by the G-flux, which is a choice of a class in "
            "H^4(X) subject to quantisation and tadpole conditions. This "
            "package carries no flux data, so there is no spectrum to report. "
            "The tadpole itself, d3_tadpole(), is exact.",
            missing=self.missing_for_physical())

    def missing_for_physical(self):
        return [
            "a choice of G-flux in H^{2,2}(X), quantised so that G + c_2(X)/2 "
            "is integral, satisfying the D3 tadpole and primitivity",
            "the resolution of the singular fibres, which the configuration "
            "matrix does not record",
            "the matter curves and their intersection points, where the "
            "Yukawa couplings localise",
        ] + Theory.missing_for_physical(self)


# ---------------------------------------------------------------------------
# obvious fibrations of a CICY
# ---------------------------------------------------------------------------


def obvious_fibrations(conf, fibre_dim=1):
    r"""Fibrations visible in the configuration matrix itself.

    Following Anderson, Gao, Gray and Lee: reorder the rows and columns of a
    configuration matrix into

        [ F  0 ]
        [ C  B ]

    so that the top block of rows has zero entries outside the left block of
    columns. Then the top block is itself a complete intersection in its own
    ambient factors, the bottom block defines the base, and the manifold fibres
    over the base with the top block as fibre. When the fibre block has
    dimension one it is a Calabi-Yau one-fold -- an elliptic curve -- and the
    fibration is a genus-one fibration.

    The Calabi-Yau condition on the fibre block is automatic and worth seeing
    why: each row of the full matrix sums to its ambient dimension plus one,
    and a fibre row is zero on every base column, so its entries in the fibre
    columns already sum correctly. The fibre of an elliptically fibred CICY
    is Calabi-Yau because the total space is.

    Parameters
    ----------
    conf : configuration matrix
    fibre_dim : int, optional
        1 for a genus-one fibration, 2 for a K3 or abelian surface fibration.

    Returns
    -------
    list of dict
        One per splitting, with ``fibre_rows``, ``fibre_cols``, ``base_rows``,
        ``base_cols``, ``fibre`` and ``base`` (sub-configurations),
        ``fibre_dim``, ``base_dim`` and ``base_name`` when the base is
        recognisable.

    Notes
    -----
    A genus-one fibration need not have a section, and this function makes no
    claim that it does. The (3,3) hypersurface in P^2 x P^2 is found here, and
    its fibres are plane cubics with no marked point; the Weierstrass model
    over the same base is a different manifold with different Hodge numbers.

    The search is over subsets of rows, which fixes the columns: a base column
    is one that vanishes on every fibre row. That is 2^(number of rows), so up
    to 2^15 for the CICY list, which is fast but not free.

    Examples
    --------
    The (3,3) hypersurface in P^2 x P^2, elliptically fibred over P^2:

    >>> [f["base_name"] for f in obvious_fibrations([[2, 3], [2, 3]])]
    ['P^2']
    """
    M = np.asarray(conf, dtype=int)
    dims = [int(d) for d in M[:, 0]]
    D = M[:, 1:]
    nrow, ncol = D.shape
    total = int(sum(dims)) - ncol
    fibre_dim = int(fibre_dim)
    base_dim = total - fibre_dim
    if base_dim < 1 or nrow < 2:
        return []

    # For each column, the set of rows on which it is non-zero, as a bitmask.
    # A column is a base column exactly when it misses every fibre row.
    nz = [0] * ncol
    for j in range(ncol):
        for i in range(nrow):
            if D[i, j]:
                nz[j] |= 1 << i
    if any(z == 0 for z in nz):
        return []                          # a column that constrains nothing

    # The dimension bound prunes the search hard. The fibre block satisfies
    # sum(dims over fibre rows) - (number of fibre columns) = fibre_dim, and
    # there are at most ncol fibre columns, so the fibre rows carry at most
    # fibre_dim + ncol dimensions between them. On a configuration with many
    # P^1 factors that cuts 2^15 subsets down to a few thousand.
    cap = fibre_dim + ncol
    subsets = []

    def walk(start, mask, dsum):
        if mask:
            subsets.append((mask, dsum))
        for i in range(start, nrow):
            if dsum + dims[i] <= cap:
                walk(i + 1, mask | (1 << i), dsum + dims[i])

    walk(0, 0, 0)

    out = []
    full = (1 << nrow) - 1
    for mask, dsum in subsets:
        if mask == full:
            continue
        fc = [j for j in range(ncol) if nz[j] & mask]
        if dsum - len(fc) != fibre_dim:
            continue
        bc = [j for j in range(ncol) if not (nz[j] & mask)]
        fr = [i for i in range(nrow) if mask >> i & 1]
        br = [i for i in range(nrow) if not (mask >> i & 1)]
        fibre = [[dims[i]] + [int(D[i, j]) for j in fc] for i in fr]
        # The fibre of a Calabi-Yau fibration is Calabi-Yau, and here that is
        # a consequence rather than an assumption: a fibre row is zero on
        # every base column, so its degrees in the fibre columns already sum
        # to the whole row sum. Checking it costs nothing and makes the
        # function honest on input that is not Calabi-Yau to begin with.
        if any(sum(r[1:]) != r[0] + 1 for r in fibre):
            continue
        base = [[dims[i]] + [int(D[i, j]) for j in bc] for i in br]
        bdim = sum(dims[i] for i in br) - len(bc)
        out.append({"fibre_rows": fr, "fibre_cols": fc,
                    "base_rows": br, "base_cols": bc,
                    "fibre": fibre, "base": base,
                    "fibre_dim": fibre_dim, "base_dim": bdim,
                    "base_name": _recognise(base)})
    return out


def is_obviously_fibred(conf, fibre_dim=1):
    """Whether :func:`obvious_fibrations` finds at least one splitting."""
    return bool(obvious_fibrations(conf, fibre_dim))


def _recognise(conf):
    """Name a base configuration when it is a product of projective spaces.

    A base with no equations of its own is just its ambient factors, which
    covers the cases that matter for the fibrations found in the CICY list:
    P^2, P^1 x P^1 and P^3. Anything with a defining equation is left
    unnamed rather than guessed at.
    """
    M = np.asarray(conf, dtype=int)
    if M.shape[1] > 1:
        return None
    return " x ".join("P^%d" % int(d) for d in M[:, 0])
