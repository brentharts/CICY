r"""
pyCICY.theories.cocycles -- explicit representatives, and a number.

What this adds
--------------
:mod:`pyCICY.theories.representatives` labels a cohomology class by its Koszul
origin --- an index set ``S`` and a list of which projective factors carry top
degree --- and decides from that whether a cup product vanishes. It says so
itself: the overall normalisation is not computed, and what is returned is a
boolean.

On a product of projective spaces the classes are explicit enough to multiply,
so the boolean can be upgraded to an integer. That is what this module does.

The Cech basis
--------------
On ``P^n`` the two non-vanishing cohomologies of ``O(m)`` both have a monomial
basis:

    H^0(P^n, O(m)),  m >= 0
        the degree-``m`` monomials ``x^a``, all ``a_i >= 0``, ``|a| = m``;

    H^n(P^n, O(m)),  m <= -n-1
        the Laurent monomials ``x^a`` with *every* ``a_i <= -1`` and
        ``|a| = m``.

The second is the Cech class of the top open cover, and its dimension
``C(-m-1, n)`` is exactly what :func:`representatives.ambient_degree_dims`
counts --- so this is that function's basis, not a different model of it.

Multiplication is multiplication of monomials, followed by the projection that
kills anything leaving the range: a product landing with some exponent ``>= -1``
in a top-degree factor is zero in cohomology. Serre duality on ``P^n`` picks out
``H^n(O(-n-1)) = C`` with generator ``1/(x_0 ... x_n)``, so a triple product
that reaches the top is a *number*: the coefficient of that generator.

Kunneth multiplies these factor by factor, and the Koszul side contributes
``e_{S_1} ^ e_{S_2} ^ e_{S_3}``, which is ``+-e_{0} ^ ... ^ e_{K-1}`` when the
index sets are disjoint and exhaust the defining polynomials, and zero
otherwise. The sign is the sign of the sorting permutation.

What the number means, and what it does not
-------------------------------------------
The result is exact and integral, but it is a coupling only up to normalisation,
in two distinct senses that are worth keeping apart:

**One overall constant, shared by the whole model.** Identifying
``H^{top}(A, Lambda^K N^*)`` with ``H^3(X, O_X) = C`` is a choice of generator,
and rescaling a basis vector of any one cohomology group rescales every coupling
it appears in. So an individual entry is not meaningful on its own; *ratios* of
couplings that share their groups, the *relative signs*, and the *rank* of the
resulting mass matrix are. Those are the quantities this returns.

**The Kahler normalisation is absent entirely.** These are holomorphic
couplings. Turning them into physical Yukawa couplings divides by the matter
field kinetic terms, which need the Ricci-flat metric and are not available
here --- the same wall :mod:`pyCICY.theories.base` draws everywhere else.

The sign convention is the ordering the caller supplies: the Koszul wedge is
computed on the three index sets in the order given, so permuting the arguments
can flip the sign. :func:`coupling` reports the magnitude, which does not depend
on that, alongside the signed value, which does.

Scope
-----
As in :mod:`representatives`, only classes with a unique Koszul origin are
handled. Where that fails this module declines and says which of the two ways it
failed --- an empty group or a genuine mixture --- rather than reporting both as
the same thing.
"""

import itertools
from math import comb

import numpy as np

from .representatives import ambient_degree_dims, koszul_origin

__all__ = [
    "cech_basis", "ambient_basis", "top_generator", "koszul_sign",
    "class_origin", "coupling", "coupling_tensor", "model_couplings",
]


def cech_basis(n, m):
    r"""
    An explicit monomial basis of ``H^*(P^n, O(m))``.

    Returns ``(degree, monomials)`` where each monomial is a tuple of ``n+1``
    integer exponents. ``degree`` is ``0`` for ``m >= 0``, ``n`` for
    ``m <= -n-1``, and ``None`` in the gap, where the cohomology vanishes and
    the monomial list is empty.

    The length of the list agrees with
    :func:`representatives.ambient_degree_dims` by construction; this is that
    count made explicit.
    """
    n = int(n)
    m = int(m)
    if m >= 0:
        mons = [c for c in _compositions(m, n + 1)]
        return 0, mons
    if m <= -n - 1:
        # every exponent <= -1: write a_i = -b_i - 1 with b_i >= 0, so that
        # sum b_i = -m - (n+1) and the count is C(-m-1, n) as it must be
        mons = [tuple(-b - 1 for b in c)
                for c in _compositions(-m - (n + 1), n + 1)]
        return n, mons
    return None, []


def _compositions(total, parts):
    """All ``parts``-tuples of non-negative integers summing to ``total``."""
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in _compositions(total - first, parts - 1):
            yield (first,) + rest


def ambient_basis(dims, k):
    r"""
    A Kunneth basis of ``H^*(A, O(k))`` on ``A = P^{n_1} x ... x P^{n_m}``.

    Returns ``(degrees, basis)``. ``degrees`` is the per-factor cohomological
    degree, and each basis element is a tuple of per-factor exponent tuples.
    An empty basis means the group vanishes.
    """
    dims = [int(d) for d in dims]
    k = [int(x) for x in k]
    per = [cech_basis(n, m) for n, m in zip(dims, k)]
    if any(d is None for d, _ in per):
        return None, []
    degrees = tuple(d for d, _ in per)
    basis = [tuple(t) for t in itertools.product(*[mons for _, mons in per])]
    return degrees, basis


def top_generator(dims):
    """The generator of ``H^{top}(A, O(-n_1-1, ...))``: all exponents ``-1``."""
    return tuple(tuple([-1] * (int(n) + 1)) for n in dims)


def koszul_sign(subsets, K):
    r"""
    The sign of ``e_{S_1} ^ e_{S_2} ^ ... ``, or ``0``.

    Zero when the index sets overlap or fail to exhaust ``range(K)``; otherwise
    the sign of the permutation sorting their concatenation, which is the
    coefficient of ``e_0 ^ ... ^ e_{K-1}``.
    """
    flat = [a for S in subsets for a in S]
    if len(set(flat)) != len(flat) or set(flat) != set(range(int(K))):
        return 0
    # sign of the permutation sorting `flat`, by counting inversions
    inv = sum(1 for i in range(len(flat)) for j in range(i + 1, len(flat))
              if flat[i] > flat[j])
    return -1 if inv % 2 else 1


def class_origin(conf, k, degree=1, verify=True):
    r"""
    The unique Koszul origin of ``H^degree(X, O(k))``, with its basis.

    Returns a dict with ``status`` one of:

    ``"unique"``
        one contributing term; the dict carries ``S``, the per-factor
        ``degrees``, and an explicit monomial ``basis``.
    ``"empty"``
        no term contributes --- the cohomology group is zero. This is *not* a
        mixture, and telling the two apart matters: an empty group means the
        coupling is exactly absent, while a mixture means the question is open.
    ``"mixed"``
        several terms contribute, the classes mix, the spectral sequence
        differentials matter, and no explicit representative is claimed.
    ``"differentials"``
        one term contributes, but its dimension disagrees with the true
        ``h^degree(X, O(k))``, so the sequence has not degenerated here and the
        monomials span the ``E_1`` term rather than the cohomology.

    ``verify`` controls that last check, which costs one :meth:`CICY.line_co`
    call. Turning it off is faster and unsound; it exists for callers that have
    already established degeneration by other means.
    """
    conf = np.asarray(conf, dtype=int)
    dims = conf[:, 0]
    recs = [r for r in koszul_origin(conf, k) if r["degree_on_X"] == int(degree)]
    if not recs:
        return {"status": "empty", "S": None, "basis": [],
                "reason": "no Koszul term contributes in degree %d, so "
                          "H^%d(X, O(k)) = 0" % (int(degree), int(degree))}
    if len(recs) > 1:
        return {"status": "mixed", "S": [r["S"] for r in recs], "basis": [],
                "reason": "%d Koszul terms contribute, so the classes are a "
                          "mixture and the differentials matter" % len(recs)}
    r = recs[0]
    degrees, basis = ambient_basis(dims, r["twisted_charge"])
    out = {"status": "unique", "S": r["S"], "degrees": degrees,
           "basis": basis, "dimension": len(basis),
           "twisted_charge": r["twisted_charge"], "reason": None}
    if not verify:
        return out

    # Uniqueness *within* one degree is not enough. The Koszul spectral
    # sequence still has differentials, and they can cut this term down before
    # it reaches H^*(X) -- in which case the monomials above span the E_1 term
    # but not the cohomology, and are not representatives of anything. The
    # honest test is against an engine that already resolves the sequence.
    from ..pyCICY import CICY
    h = CICY(conf.tolist()).line_co(list(np.asarray(k, dtype=int)))
    true_dim = int(h[int(degree)])
    if true_dim != len(basis):
        return {"status": "differentials", "S": r["S"], "basis": [],
                "dimension": None, "e1_dimension": len(basis),
                "true_dimension": true_dim,
                "reason": "the E_1 term has dimension %d but h^%d(X, O(k)) = "
                          "%d, so the spectral sequence differentials act and "
                          "these monomials are not representatives"
                          % (len(basis), int(degree), true_dim)}
    return out


def _multiply(dims, monomials):
    r"""
    Multiply Cech monomials factor by factor, or return ``None``.

    ``None`` means the product is zero in cohomology: in each factor the
    exponents simply add, and the result must be the top generator, every
    exponent ``-1``. Anything else has left the range of the Cech complex.
    """
    out = []
    for i, n in enumerate(dims):
        e = [0] * (int(n) + 1)
        for mon in monomials:
            for j, x in enumerate(mon[i]):
                e[j] += x
        out.append(tuple(e))
    return tuple(out)


def coupling(conf, charges, degree=1, verify=True):
    r"""
    The exact coupling of three cohomology classes, as an integer tensor.

    ``charges`` is three line bundle charge vectors summing to zero, so that
    the product lands in ``H^3(X, O_X) = C``.

    Returns a dict with:

    ``tensor``
        an integer array of shape ``(d_1, d_2, d_3)``, the coefficient of the
        top generator for each triple of basis elements;
    ``value``
        the single entry when all three groups are one-dimensional, else
        ``None``;
    ``magnitude``
        ``abs(value)``, which does not depend on the argument ordering;
    ``vanishes``
        whether the tensor is identically zero.

    A non-zero entry is a holomorphic coupling up to the model-wide
    normalisation described in this module's docstring. It is not a physical
    Yukawa coupling, which needs the metric.
    """
    charges = [np.asarray(c, dtype=int) for c in charges]
    if len(charges) != 3:
        raise ValueError("a Yukawa coupling is a product of three classes")
    if np.any(sum(charges)):
        raise ValueError(
            "the charges sum to %s, not zero, so the product does not land in "
            "H^3(O_X) and there is no coupling to analyse"
            % (sum(charges)).tolist())

    conf = np.asarray(conf, dtype=int)
    dims = [int(d) for d in conf[:, 0]]
    K = conf.shape[1] - 1

    origins = [class_origin(conf, c, degree, verify=verify) for c in charges]

    # An empty group is an exact statement, not a failure: if any participating
    # cohomology vanishes there is nothing to multiply and the coupling is
    # absent. Only a genuine mixture leaves the question open.
    empty = [i for i, o in enumerate(origins) if o["status"] == "empty"]
    if empty:
        return {"tensor": np.zeros((0, 0, 0), dtype=int), "value": 0,
                "magnitude": 0, "vanishes": True,
                "reason": "class %d lies in a zero-dimensional group, so the "
                          "coupling is absent" % empty[0],
                "koszul_sign": 0, "shape": (0, 0, 0)}

    mixed = [i for i, o in enumerate(origins)
             if o["status"] in ("mixed", "differentials")]
    if mixed:
        i = mixed[0]
        raise ValueError("class %d has no explicit representative: %s"
                         % (i, origins[i]["reason"]))

    sgn = koszul_sign([o["S"] for o in origins], K)
    shape = tuple(o["dimension"] for o in origins)

    if sgn == 0:
        return {"tensor": np.zeros(shape, dtype=int), "value": 0,
                "magnitude": 0, "vanishes": True,
                "reason": "the Koszul indices %s do not wedge to the top term"
                          % ([o["S"] for o in origins],),
                "koszul_sign": 0, "shape": shape}

    # the ambient degrees must also reach H^top(A)
    total_q = [sum(o["degrees"][i] for o in origins) for i in range(len(dims))]
    if total_q != dims:
        return {"tensor": np.zeros(shape, dtype=int), "value": 0,
                "magnitude": 0, "vanishes": True,
                "reason": "the ambient degrees sum to %s, not H^top(A) = %s"
                          % (total_q, dims),
                "koszul_sign": sgn, "shape": shape}

    top = top_generator(dims)
    tensor = np.zeros(shape, dtype=int)
    for idx in itertools.product(*[range(d) for d in shape]):
        mons = [origins[a]["basis"][idx[a]] for a in range(3)]
        prod = _multiply(dims, mons)
        if prod == top:
            tensor[idx] = sgn

    value = int(tensor.reshape(-1)[0]) if tensor.size == 1 else None
    return {"tensor": tensor, "value": value,
            "magnitude": abs(value) if value is not None else None,
            "vanishes": not tensor.any(), "reason": None,
            "koszul_sign": sgn, "shape": shape}


def coupling_tensor(conf, charges, degree=1, verify=True):
    """The integer tensor of :func:`coupling`, on its own."""
    return coupling(conf, charges, degree, verify=verify)["tensor"]


def model_couplings(conf, summands, kind="up"):
    r"""
    Every up-type coupling of a line bundle model, with its exact value.

    Runs :func:`coupling` over the charge-allowed patterns and returns the
    signed values. Turning these into a mass matrix additionally requires
    choosing which ``5`` is the Higgs, which is a property of the Wilson line
    and the breaking pattern rather than of the geometry, so no rank is claimed
    here.

    Patterns whose classes genuinely mix are reported with a status rather than
    a number; patterns killed by a zero-dimensional group are reported as
    vanishing, because that is what they are.
    """
    conf = np.asarray(conf, dtype=int)
    k = np.asarray(summands, dtype=int)
    if kind != "up":
        raise ValueError("only up-type couplings are implemented here; the "
                         "down-type index structure needs the epsilon tensor")

    records = []
    for a, b in itertools.combinations(range(len(k)), 2):
        trio = [k[a], k[b], -(k[a] + k[b])]
        pattern = ("10_%d" % a, "10_%d" % b, "5_%d%d" % (a, b))
        try:
            res = coupling(conf, trio)
        except ValueError as e:
            records.append({"pattern": pattern, "status": "no representative",
                            "value": None, "reason": str(e)})
            continue
        records.append({"pattern": pattern,
                        "status": "vanishes" if res["vanishes"] else "present",
                        "value": res["value"], "magnitude": res["magnitude"],
                        "koszul_sign": res["koszul_sign"],
                        "reason": res["reason"]})

    present = [r for r in records if r["status"] == "present"]
    return {"records": records, "present": len(present),
            "vanishing": sum(1 for r in records if r["status"] == "vanishes"),
            "undecided": sum(1 for r in records
                             if r["status"] == "no representative"),
            "values": {r["pattern"]: r["value"] for r in present},
            "note": "values are exact up to one model-wide normalisation and "
                    "carry no Kahler factor; ratios, relative signs and rank "
                    "are the meaningful content"}
