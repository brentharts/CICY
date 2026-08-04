r"""
pyCICY.chirality -- one mirror operation, four kinds of object.

The claim
---------
Three of the mirror operations this package deals with have the same shape.
Each is an involution, each swaps a pair of integer invariants, and each
preserves the sum or span of that pair:

    domain                 involution          swapped pair        preserved
    ------------------------------------------------------------------------
    knot                   mirror image        degrees of V(t)     span of V
    reflexive polygon      polar duality       (#dP, #dP*)         12
    Calabi-Yau threefold   mirror symmetry     (h^{1,1}, h^{2,1})  h^11 + h^21

For knots the mirror image sends V(t) to V(1/t), so the pair of extreme
degrees (a, b) goes to (-b, -a) and the span b - a is fixed. For reflexive
polygons the Batyrev mirror is polar duality, and the twelve theorem says the
boundary point counts of P and P* sum to twelve however they are exchanged.
For Calabi-Yau threefolds mirror symmetry exchanges the two Hodge numbers and
therefore negates the Euler characteristic while fixing their sum. The
:func:`survey` function tabulates all three side by side, and the test suite
checks the swap-and-preserve law in each domain rather than asserting it.

Each record also carries an ``asymmetry``: the combination of the pair that
*negates* under the involution, namely the sum of the extreme Jones degrees,
the difference of the two boundary counts, and h^{1,1} - h^{2,1}, which is
half the Euler characteristic. Plotting asymmetry against the preserved
quantity puts mirror partners symmetrically about zero and fixed points on
the axis; for the Calabi-Yau domain that plot is exactly the conventional
Hodge plot, and :func:`pyCICY.viz.plot_chirality` draws it for all domains at
once. Note that on the knot side a vanishing asymmetry is a weaker statement
than a palindromic Jones polynomial: it says only that the range of degrees
is symmetric, so ``asymmetry`` is a coarser detector than ``fixed``.

An object is *chiral* when it is not fixed by its involution. Fixed points
exist in every domain and this module finds them: the amphichiral knots of
the table, the four self-dual reflexive polygons, and the CICY entries with
h^{1,1} = h^{2,1}.

The fourth domain, and a warning
--------------------------------
The quantized mirror curves of :mod:`pyCICY.quantum_curve` are the awkward
case, and they are included precisely because they are awkward. The obvious
involution there is reflection of the Newton polygon, (m,n) -> (-m,-n), but
that operation leaves the spectrum *exactly* unchanged, so no spectral
invariant can detect it. :func:`curve_chirality` reports ``detected`` as
``None`` for this reason rather than reporting ``False``, which would suggest
the curve had been shown to be achiral.

What the lattice spectrum does see is a different property altogether,
bipartiteness, which is a condition on the polygon modulo two and is neither
implied by nor implies invariance under reflection. It is reported alongside,
under its own name, so that the two are not confused. The moral is that
"mirror" is not one operation wearing four hats; the resemblance is real for
three of these domains and breaks down for the fourth, and a uniform
interface is only honest if it says so.

A note on detection
-------------------
Throughout, ``detected`` being ``False`` means the chosen invariant did not
distinguish the object from its mirror, not that the object is achiral. The
Jones polynomial in particular is a sufficient but not necessary test for
chirality: a chiral knot may still have a palindromic Jones polynomial.
``fixed`` is reserved for the cases where the object itself, and not merely
an invariant of it, is genuinely unmoved.

The CICY list is chiral as a set
--------------------------------
:func:`cicy_list_chirality` asks whether the published list of CICY
threefolds is closed under mirror symmetry, and it is emphatically not:
of the distinct Hodge pairs occurring in the list, only the self-mirror ones
have their partner present. The reason is visible in the data. Euler numbers
in the list are all non-positive, h^{1,1} never exceeds 19 while h^{2,1}
reaches 101, so a mirror partner would need Hodge numbers outside the range
the construction produces at all. The mirror of a CICY is essentially never
a CICY, which is why :func:`mirror` returns Hodge data for this domain and
not a configuration matrix.
"""

import os

from . import knots as _knots
from . import quantum_curve as _qc
from . import toric as _toric

__all__ = [
    "DOMAINS", "chirality", "mirror", "mirror_pair", "mirror_invariant",
    "knot_chirality", "polygon_chirality", "curve_chirality", "cicy_chirality",
    "survey", "cicy_list_chirality", "format_survey",
]


DOMAINS = {
    "knot": {
        "involution": "mirror image (reverse every crossing)",
        "detector": "Jones polynomial under t -> 1/t",
        "pair": "extreme degrees of V",
        "preserved": "span of V",
    },
    "polygon": {
        "involution": "polar duality (the Batyrev mirror)",
        "detector": "GL(2,Z) class of P against P*",
        "pair": "boundary lattice points of P and of P*",
        "preserved": "their sum, which is 12",
    },
    "cicy": {
        "involution": "mirror symmetry",
        "detector": "Hodge numbers",
        "pair": "(h^{1,1}, h^{2,1})",
        "preserved": "h^{1,1} + h^{2,1}",
    },
    "curve": {
        "involution": "reflection of the Newton polygon",
        "detector": "none: the spectrum is invariant under it",
        "pair": None,
        "preserved": "the whole spectrum",
    },
}


def _record(domain, name, involution, pair, mirror_pair_, preserved,
            fixed, detected, note=None, asymmetry=None, **extra):
    rec = {
        "domain": domain,
        "name": name,
        "involution": involution,
        "pair": pair,
        "mirror_pair": mirror_pair_,
        "preserved": preserved,
        "asymmetry": asymmetry,
        "fixed": fixed,
        "detected": detected,
        "note": note,
    }
    rec.update(extra)
    return rec


# ------------------------------------------------------------------- knots

def knot_chirality(knot):
    """Chirality of a knot, detected by the Jones polynomial.

    The mirror image sends V(t) to V(1/t), so the extreme degrees (a, b) of
    V go to (-b, -a) and the span is preserved. ``detected`` is False when
    V is palindromic, which does not prove the knot amphichiral.
    """
    if isinstance(knot, str):
        knot = _knots.from_name(knot)
    v = knot.jones()
    vm = knot.mirror().jones()
    a, b = v.degrees()
    am, bm = vm.degrees()
    palindromic = v.is_palindromic()
    return _record(
        "knot", knot.name, DOMAINS["knot"]["involution"],
        (a, b), (am, bm), b - a,
        asymmetry=a + b,
        fixed=palindromic, detected=not palindromic,
        note=("Jones detects chirality" if not palindromic else
              "Jones does not separate this knot from its mirror; "
              "that is not a proof of amphichirality"),
        jones=v, jones_mirror=vm,
        determinant=knot.determinant(),
        crossings=len(knot), writhe=knot.writhe(),
    )


# ---------------------------------------------------------------- polygons

def polygon_chirality(polygon):
    """Chirality of a reflexive polygon under polar duality.

    Polar duality is the Batyrev mirror. It exchanges the boundary lattice
    point counts of P and P*, whose sum is always twelve, and its fixed
    points are the four self-dual reflexive polygons.
    """
    if isinstance(polygon, str):
        name, verts = polygon, _toric.polygon(polygon)
    else:
        verts = _toric.convex_hull([tuple(v) for v in polygon])
        name = _toric.classify(verts)["name"]
    b, bd, total = _toric.twelve(verts)
    dual = _toric.dual(verts)
    self_dual = _toric.equivalent(verts, dual)
    dual_nm = _toric.dual_name(name) if name else None
    return _record(
        "polygon", name, DOMAINS["polygon"]["involution"],
        (b, bd), (bd, b), total,
        asymmetry=b - bd,
        fixed=self_dual, detected=not self_dual,
        note=("self-dual" if self_dual else "dual is {}".format(dual_nm)),
        dual_name=dual_nm, dual=dual, degree=_toric.degree(verts),
        smooth=_toric.is_smooth(verts),
        # the separate, non-mirror axis; see the module docstring
        bipartite=_toric.is_bipartite(verts),
        centrally_symmetric=_toric.is_centrally_symmetric(verts),
    )


# ----------------------------------------------------------- quantum curves

def curve_chirality(curve, p=1, q=3, nk=12, tol=1e-8):
    """The awkward case: reflection of a quantized curve is undetectable.

    Reflecting the Newton polygon leaves the spectrum exactly where it was,
    so ``detected`` is ``None`` rather than ``False``. The genuinely
    measurable asymmetry of the spectrum is bipartiteness, reported here
    under ``spectrally_chiral``, and it is a different property: local B_3 is
    fixed by reflection and spectrally chiral, while T4 is not fixed by
    reflection and is spectrally symmetric.
    """
    if isinstance(curve, str):
        curve = _qc.from_polygon(curve)
    reflected = curve.mirror()
    fixed = set(reflected.points) == set(curve.points)
    asym = curve.spectral_asymmetry(p, q, nk=nk)
    return _record(
        "curve", curve.name, DOMAINS["curve"]["involution"],
        None, None, "the spectrum",
        asymmetry=None,
        fixed=fixed, detected=None,
        note=("the spectrum is invariant under reflection, so no spectral "
              "invariant can detect this involution; see spectrally_chiral "
              "for the property the spectrum does see"),
        hops=len(curve.points),
        spectral_asymmetry=asym,
        spectrally_chiral=asym > tol,
        bipartite=curve.is_bipartite(),
        centrally_symmetric=curve.is_centrally_symmetric(),
        flux=(p, q),
    )


# -------------------------------------------------------------------- CICYs

def cicy_chirality(conf=None, hodge=None, name=None):
    """Chirality of a Calabi-Yau threefold under mirror symmetry.

    Pass either a configuration matrix, whose Hodge numbers are then
    computed through :func:`pyCICY.cache.hodge`, or an explicit
    ``hodge=(h11, h21)`` pair. Mirror symmetry exchanges the two Hodge
    numbers, fixes their sum and negates the Euler characteristic.

    The mirror manifold itself is not constructed. For the published list it
    is almost never a CICY at all; see :func:`cicy_list_chirality`.
    """
    if hodge is None:
        if conf is None:
            raise ValueError("give a configuration matrix or hodge=(h11, h21)")
        from . import cache as _cache
        data = _cache.hodge(conf)
        if data.get("error"):
            raise ValueError("could not compute Hodge data: " + data["error"])
        h11, h21 = int(data["h11"]), int(data["h21"])
    else:
        h11, h21 = (int(x) for x in hodge)
    euler = 2 * (h11 - h21)
    self_mirror = h11 == h21
    return _record(
        "cicy", name, DOMAINS["cicy"]["involution"],
        (h11, h21), (h21, h11), h11 + h21,
        asymmetry=h11 - h21,
        fixed=self_mirror, detected=not self_mirror,
        note=("self-mirror Hodge numbers" if self_mirror
              else "mirror has Hodge numbers ({}, {})".format(h21, h11)),
        h11=h11, h21=h21, euler=euler, euler_mirror=-euler,
        configuration=conf,
    )


# ------------------------------------------------------------- the dispatch

def chirality(obj, kind=None, **kw):
    """Chirality record for any supported object.

    Instances of :class:`pyCICY.knots.Knot` and
    :class:`pyCICY.quantum_curve.QuantumCurve` are recognised directly. Raw
    lists are ambiguous -- a list of pairs could be a Newton polygon or a
    configuration matrix -- so pass ``kind`` as one of ``'knot'``,
    ``'polygon'``, ``'curve'`` or ``'cicy'`` for those.
    """
    if kind is None:
        if isinstance(obj, _knots.Knot):
            kind = "knot"
        elif isinstance(obj, _qc.QuantumCurve):
            kind = "curve"
        elif isinstance(obj, str):
            if obj in _knots.KNOTS:
                kind = "knot"
            elif obj in _toric.NAMED or obj in _toric.ALIASES:
                kind = "polygon"
            else:
                raise ValueError("unknown name {!r}".format(obj))
        else:
            raise ValueError(
                "cannot tell what {!r} is; pass kind='polygon', 'cicy', "
                "'knot' or 'curve'".format(type(obj).__name__))
    if kind == "knot":
        return knot_chirality(obj, **kw)
    if kind == "polygon":
        return polygon_chirality(obj, **kw)
    if kind == "curve":
        return curve_chirality(obj, **kw)
    if kind == "cicy":
        if isinstance(obj, (list, tuple)):
            return cicy_chirality(conf=obj, **kw)
        return cicy_chirality(**kw)
    raise ValueError("unknown kind {!r}".format(kind))


def mirror(obj, kind=None, **kw):
    """The mirror of an object, in whatever sense its domain supports.

    Knots and quantized curves come back as objects; polygons come back as
    the polar dual; a Calabi-Yau comes back as Hodge data only, because the
    mirror of a CICY is generally not a CICY.
    """
    rec = chirality(obj, kind=kind, **kw)
    if rec["domain"] == "knot":
        k = obj if isinstance(obj, _knots.Knot) else _knots.from_name(obj)
        return k.mirror()
    if rec["domain"] == "curve":
        c = obj if isinstance(obj, _qc.QuantumCurve) else _qc.from_polygon(obj)
        return c.mirror()
    if rec["domain"] == "polygon":
        return rec["dual"]
    return {"h11": rec["h21"], "h21": rec["h11"], "euler": rec["euler_mirror"]}


def mirror_pair(obj, kind=None, **kw):
    """The pair of integers the mirror operation swaps, or ``None``."""
    return chirality(obj, kind=kind, **kw)["pair"]


def mirror_invariant(obj, kind=None, **kw):
    """The quantity the mirror operation preserves."""
    return chirality(obj, kind=kind, **kw)["preserved"]


# ------------------------------------------------------------------ surveys

def survey(knots=None, polygons=None, hodge_pairs=None, curves=None):
    """Chirality records across all four domains."""
    if knots is None:
        knots = sorted(_knots.KNOTS)
    if polygons is None:
        polygons = list(_toric.NAMED)
    if curves is None:
        curves = ["F0", "B3", "P2", "T4"]
    if hodge_pairs is None:
        # a few landmark CICY threefolds, Hodge numbers quoted from the list
        hodge_pairs = [("quintic", (1, 101)), ("bicubic", (2, 83)),
                       ("tetraquadric", (4, 68)), ("split of quintic", (2, 86)),
                       ("self-mirror example", (15, 15))]
    out = []
    out += [knot_chirality(n) for n in knots]
    out += [polygon_chirality(n) for n in polygons]
    out += [curve_chirality(n) for n in curves]
    out += [cicy_chirality(hodge=h, name=n) for n, h in hodge_pairs]
    return out


def format_survey(records=None):
    """The cross-domain table as a printable string."""
    if records is None:
        records = survey()
    lines = ["{:<10} {:<20} {:>14} {:>14} {:>10} {:>8}".format(
        "domain", "name", "pair", "mirror pair", "preserved", "fixed"),
        "-" * 80]
    for r in records:
        pair = "-" if r["pair"] is None else str(tuple(r["pair"]))
        mp = "-" if r["mirror_pair"] is None else str(tuple(r["mirror_pair"]))
        pres = r["preserved"]
        pres = pres if isinstance(pres, int) else "-"
        lines.append("{:<10} {:<20} {:>14} {:>14} {:>10} {:>8}".format(
            r["domain"], str(r["name"])[:20], pair, mp, pres,
            "yes" if r["fixed"] else "-"))
    return "\n".join(lines)


def _default_list_path():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(here, "data", "cicylist.json")


def cicy_list_chirality(path=None):
    """Is the published CICY threefold list closed under mirror symmetry?

    Collects the distinct (h^{1,1}, h^{2,1}) pairs occurring in the list and
    asks, for each, whether the swapped pair also occurs. Returns counts and
    the ranges that explain the answer.

    Entries recording h^{1,1} = h^{2,1} = 0 are excluded and counted
    separately. Zero is not a possible Hodge number for a Calabi-Yau
    threefold; in the published file it is a sentinel for "not given", used
    on the configurations that are products, such as the elliptic curve times
    a quartic K3 of entry 31, whose second Chern class is likewise recorded
    with a zero. Counting them as self-mirror would inflate the answer with
    twenty-two manifolds whose Hodge numbers the list never supplied.
    """
    from . import cicylist as _cl
    path = path or _default_list_path()
    entries = _cl.load_published_list(path)
    counts = {}
    degenerate = 0
    for e in entries:
        h11, h21 = int(e["h11"]), int(e["h21"])
        if h11 == 0 or h21 == 0:
            degenerate += 1
            continue
        counts[(h11, h21)] = counts.get((h11, h21), 0) + 1
    pairs = set(counts)
    self_mirror = sorted(p for p in pairs if p[0] == p[1])
    with_partner = sorted(p for p in pairs if (p[1], p[0]) in pairs)
    without = sorted(p for p in pairs if (p[1], p[0]) not in pairs)
    return {
        "path": path,
        "n_manifolds": len(entries),
        "n_degenerate": degenerate,
        "n_usable": len(entries) - degenerate,
        "n_pairs": len(pairs),
        "self_mirror_pairs": self_mirror,
        "pairs_with_mirror_partner": with_partner,
        "n_pairs_with_partner": len(with_partner),
        "n_pairs_without_partner": len(without),
        "manifolds_with_partner": sum(counts[p] for p in with_partner),
        "nontrivial_mirror_pairs": [p for p in with_partner if p[0] != p[1]],
        "h11_range": (min(p[0] for p in pairs), max(p[0] for p in pairs)),
        "h21_range": (min(p[1] for p in pairs), max(p[1] for p in pairs)),
        "euler_range": (min(2 * (a - b) for a, b in pairs),
                        max(2 * (a - b) for a, b in pairs)),
        "closed_under_mirror": len(without) == 0,
    }
