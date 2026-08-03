r"""
pyCICY.symmetries -- freely acting discrete symmetries and what they sharpen.

:mod:`pyCICY.phenomenology` can say that a CICY *admits* three generations
for some symmetry order |Gamma| = |chi|/6, but not whether a freely acting
group of that order actually exists. That is the content of

    V. Braun, "On Free Quotients of Complete Intersection Calabi-Yau
    Manifolds", JHEP 04 (2011) 005, arXiv:1003.3235,

whose results ship in the Mathematica version of the Oxford CICY list.
``scripts/fetch_symmetries.py`` downloads and extracts them into
``data/symmetries.json``; this module consumes that file and turns

    "N configurations admit three generations for some |Gamma|"

into

    "N configurations carry a freely acting group of the required order".

Group orders
------------
The symmetry data records group *names*. :func:`group_order` maps the usual
ones to their orders. Names it does not recognise are reported rather than
guessed at or dropped: an unrecognised name silently treated as order 1 would
inflate the count of viable models, which is exactly the error this module
exists to avoid.

A caveat on subgroups
---------------------
If Gamma acts freely then so does any subgroup, so a group of order divisible
by the required r also supplies a free action of order r whenever a subgroup
of order r exists. That is automatic for abelian Gamma, and most of the
groups here are abelian, but it is not automatic in general. Both counts are
reported separately by :func:`three_generation_models`, exact matches and
matches by divisibility, so the weaker inference is never folded silently
into the stronger one.
"""

import json
import os
import re

__all__ = [
    "group_order", "load_symmetries", "symmetry_orders",
    "three_generation_models", "coverage_report", "UnknownGroup",
    "generator_matrices", "matrix_group_order",
    "parse_symmetry_records", "SymmetryRecord",
]


class UnknownGroup(ValueError):
    """Raised when a group name cannot be assigned an order."""


# Orders of the non-cyclic groups that appear in this classification. Cyclic
# groups and direct products are handled structurally by group_order.
_NAMED_ORDERS = {
    "Q8": 8,          # quaternion group
    "D4": 8,          # dihedral of order 8 (Mathematica's D4 convention)
    "D8": 8,
    "Dic3": 12,       # dicyclic
    "Dic5": 20,
    "A4": 12,
    "S3": 6,
    "S4": 24,
    "H8": 8,
    "F20": 20,
    "SL2Z3": 24,
    "Z2xQ8": 16,
}

_CYCLIC = re.compile(r"^Z_?(\d+)$", re.IGNORECASE)


def group_order(name):
    """Order of a discrete group given by name.

    Handles cyclic groups ``Zn``, direct products written with ``x``,
    ``X``, ``*`` or the multiplication sign, a table of named non-abelian
    groups, and bare integers.

    >>> group_order("Z5")
    5
    >>> group_order("Z3xZ3")
    9
    >>> group_order("Q8")
    8

    Unrecognised names raise rather than defaulting, so that a name this
    table does not know cannot quietly become order 1.
    """
    if isinstance(name, int):
        if name < 1:
            raise UnknownGroup("group order must be positive, got %r" % name)
        return name

    text = str(name).strip()
    if not text:
        raise UnknownGroup("empty group name")

    if text in ("1", "Trivial", "Id", "None"):
        return 1

    # direct products
    parts = re.split(r"[x\u00d7*]", text)
    if len(parts) > 1:
        order = 1
        for part in parts:
            order *= group_order(part)
        return order

    match = _CYCLIC.match(text)
    if match:
        value = int(match.group(1))
        if value < 1:
            raise UnknownGroup("bad cyclic group %r" % text)
        return value

    if text in _NAMED_ORDERS:
        return _NAMED_ORDERS[text]

    if text.isdigit():
        return int(text)

    raise UnknownGroup(
        "unrecognised group name %r; add it to _NAMED_ORDERS in "
        "pyCICY/symmetries.py rather than letting it default" % text)


# --------------------------------------------------- generator matrices
#
# The Oxford Mathematica file does not record group names. Each symmetry is
# stored as a freeness flag together with explicit generator matrices acting
# on the homogeneous coordinates, e.g.
#
#     {True, {{{-1,0,...},{0,1,...},...}, ...}}
#
# so the order of Gamma has to be computed from the generators rather than
# looked up. Entries are frequently roots of unity, which appear in the
# parsed data as function calls rather than numbers; those cases are reported
# as undetermined instead of being approximated.


def _is_int_matrix(value):
    return (isinstance(value, list) and value
            and all(isinstance(row, list) and row
                    and all(isinstance(x, int) for x in row)
                    for row in value)
            and all(len(row) == len(value[0]) for row in value))


def generator_matrices(value, out=None, depth=0):
    """Collect integer generator matrices from a parsed symmetry field.

    Returns a list of matrices, each a list of equal-length integer rows.
    Nested structure is walked, so the exact depth at which the matrices sit
    does not have to be known in advance. Non-integer entries, such as roots
    of unity, mean the matrix is skipped and reported by
    :func:`matrix_group_order` as undetermined.
    """
    if out is None:
        out = []
    if depth > 10:
        return out
    if _is_int_matrix(value):
        out.append([list(r) for r in value])
        return out
    if isinstance(value, list):
        for item in value:
            generator_matrices(item, out, depth + 1)
    elif isinstance(value, dict):
        if "rule" in value:
            generator_matrices(value["rule"][1], out, depth + 1)
        elif "args" in value:
            for a in value["args"]:
                generator_matrices(a, out, depth + 1)
    return out


def _matmul(a, b):
    n = len(a)
    m = len(b[0])
    k = len(b)
    return tuple(tuple(sum(a[i][t] * b[t][j] for t in range(k))
                       for j in range(m)) for i in range(n))


def matrix_group_order(generators, max_order=4096):
    """Order of the finite matrix group generated by ``generators``.

    Closure by breadth-first multiplication. Returns None if the group has
    not closed by ``max_order`` elements, which is treated as "not
    determined" rather than as a number: an infinite or merely large group
    must not be reported as though it were the order of a free action.
    """
    generators = [g for g in generators if g]
    if not generators:
        return None
    n = len(generators[0])
    if any(len(g) != n or any(len(r) != n for r in g) for g in generators):
        return None

    identity = tuple(tuple(1 if i == j else 0 for j in range(n))
                     for i in range(n))
    gens = [tuple(tuple(r) for r in g) for g in generators]

    seen = {identity}
    frontier = [identity]
    while frontier:
        nxt = []
        for element in frontier:
            for g in gens:
                product = _matmul(element, g)
                if product not in seen:
                    if len(seen) >= max_order:
                        return None
                    seen.add(product)
                    nxt.append(product)
        frontier = nxt
    return len(seen)


# ------------------------------------------------- the Oxford record format
#
# Each symmetry in the Mathematica file is a list of five fields:
#
#   0  freeness flag, "True"
#   1  {generators, {order, index}}   -- the GAP SmallGroup identifier
#   2  {"True", {invariants}} if the group is abelian, else {"False"}
#   3  further per-projective-space data
#   4  further per-projective-space data
#
# The order is therefore read directly from the GAP identifier rather than
# reconstructed from the generator matrices, which is both exact and much
# cheaper. For abelian groups the product of the invariants gives an
# independent check on that order, and disagreements are surfaced rather
# than resolved silently.


class SymmetryRecord(object):
    """One freely acting symmetry as recorded in the Oxford data."""

    __slots__ = ("order", "gap_index", "free", "abelian", "invariants",
                 "consistent")

    def __init__(self, order, gap_index, free, abelian, invariants):
        self.order = order
        self.gap_index = gap_index
        self.free = free
        self.abelian = abelian
        self.invariants = invariants
        if abelian and invariants:
            product = 1
            for value in invariants:
                product *= value
            self.consistent = (product == order)
        else:
            self.consistent = None      # no independent check available

    def __repr__(self):
        return "SymmetryRecord(order=%s, gap=[%s,%s], abelian=%s)" % (
            self.order, self.order, self.gap_index, self.abelian)


def _truthy(value):
    return str(value) == "True"


def parse_symmetry_records(record):
    """Structured view of the symmetries recorded for one CICY.

    Returns ``(records, problems)``: a list of :class:`SymmetryRecord` and a
    list of strings describing anything that did not fit the expected shape.
    Nothing is guessed at; an entry that does not match is reported.
    """
    syms = record.get("symmetries")
    out = []
    problems = []

    if syms is None or isinstance(syms, str) or not isinstance(syms, list):
        if isinstance(syms, str):
            problems.append("symmetry field is the string %r" % syms)
        return out, problems

    for entry in syms:
        if not (isinstance(entry, list) and len(entry) >= 3):
            problems.append("symmetry entry has %s fields, expected at least 3"
                            % (len(entry) if isinstance(entry, list) else "?"))
            continue
        free = _truthy(entry[0])
        gid = entry[1][1] if (isinstance(entry[1], list) and len(entry[1]) > 1) \
            else None
        if not (isinstance(gid, list) and len(gid) == 2
                and all(isinstance(x, int) for x in gid)):
            problems.append("no GAP identifier in symmetry entry")
            continue
        order, index = gid

        abelian = False
        invariants = None
        field2 = entry[2]
        if isinstance(field2, list) and field2:
            abelian = _truthy(field2[0])
            if abelian and len(field2) > 1 and isinstance(field2[1], list) \
                    and all(isinstance(x, int) for x in field2[1]):
                invariants = list(field2[1])

        item = SymmetryRecord(order, index, free, abelian, invariants)
        if item.consistent is False:
            problems.append(
                "GAP order %d disagrees with abelian invariants %r"
                % (order, invariants))
        out.append(item)

    return out, problems


def load_symmetries(path):
    """Load ``data/symmetries.json`` produced by the fetch script."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            "%s not found. Braun's symmetry data lives in the Mathematica "
            "version of the CICY list, which is distributed by its authors "
            "and not bundled here; run\n"
            "    python3 scripts/fetch_symmetries.py\n"
            "to download and extract it." % path)
    with open(path) as fh:
        payload = json.load(fh)
    return payload["entries"] if isinstance(payload, dict) else payload


# Rule left-hand sides under which a group name is likely to sit. The layout
# of the source file is not fixed by anything we control, so the search is by
# name rather than by position.
_GROUP_KEYS = ("group", "symmetry", "quotient", "name", "type", "gamma")

# Keys whose contents are explicitly not group names, e.g. the polynomial
# action on homogeneous coordinates. Descending into these would turn
# coordinate symbols into spurious "unknown group" reports.
_SKIP_KEYS = ("action", "matrix", "generator", "map", "poly", "conf", "perm")

# Values that are flags rather than group names. The Oxford data stores a
# freeness flag alongside the generators.
_FLAGS = ("True", "False", "Null", "None")


def _collect_group_names(value, out, depth=0):
    """Walk an extracted symmetry field and gather plausible group names."""
    if depth > 8 or value is None:
        return
    if isinstance(value, str):
        if value not in _FLAGS:
            out.append(value)
        return
    if isinstance(value, int):
        out.append(value)
        return
    if isinstance(value, list):
        for item in value:
            _collect_group_names(item, out, depth + 1)
        return
    if isinstance(value, dict):
        if "rule" in value:
            lhs, rhs = value["rule"]
            key = str(lhs).lower()
            if any(k in key for k in _SKIP_KEYS):
                return
            if any(k in key for k in _GROUP_KEYS):
                _collect_group_names(rhs, out, depth + 1)
            else:
                _collect_group_names(rhs, out, depth + 1)
            return
        if "head" in value:
            # a function call: its head is often the group name, e.g. Z[3]
            out.append(value["head"])
            return


def symmetry_orders(record, strict=False):
    """Orders of the freely acting groups recorded for one CICY.

    The symmetry field may be a bare list of names, or a list of rule sets
    such as ``{Group -> Z3, Action -> {...}}``. Both are handled: the search
    is by rule name, and fields holding the coordinate action are skipped so
    that coordinate symbols are not mistaken for group names.

    Returns ``(orders, unknown)``: the orders that could be determined, and
    the names that could not. With ``strict=True`` an unknown name raises.
    """
    syms = record.get("symmetries")
    if syms is None:
        return [], []

    # The Oxford data carries a GAP SmallGroup identifier, which gives the
    # order exactly; use it when present.
    structured, _problems = parse_symmetry_records(record)
    if structured:
        return sorted({r.order for r in structured if r.free}), []

    # Otherwise fall back to generator matrices, then to names.
    matrices = generator_matrices(syms)
    if matrices:
        order = matrix_group_order(matrices)
        if order is not None and order > 1:
            return [order], []
        return [], (["<matrix generators, order undetermined>"]
                    if order is None else [])

    names = []
    _collect_group_names(syms, names)

    orders = []
    unknown = []
    for name in names:
        try:
            orders.append(group_order(name))
        except UnknownGroup:
            if strict:
                raise
            unknown.append(str(name))
    return orders, unknown


def three_generation_models(cicy_entries, symmetry_entries, generations=3):
    """Which CICYs carry a free action of exactly the order needed.

    Parameters
    ----------
    cicy_entries : list
        Records from :func:`pyCICY.cicylist.load_published_list`, carrying
        ``num`` and ``euler``.
    symmetry_entries : list
        Records from :func:`load_symmetries`, carrying ``num`` and
        ``symmetries``.

    Returns
    -------
    dict
        ``exact``: entries where a recorded group has order exactly
        |chi| / (2 * generations).
        ``by_divisibility``: entries where some recorded group has order a
        multiple of it, so a subgroup of the right order exists provided the
        group has one. Reported separately because that is an extra step,
        automatic for abelian groups but not in general.
        ``candidates_without_symmetry``: entries whose Euler characteristic
        permits the generation count but for which no suitable group is
        recorded.
        ``unknown_groups``: names that could not be assigned an order.
    """
    from .phenomenology import required_symmetry_order

    by_num = {r["num"]: r for r in symmetry_entries}

    exact = []
    divisible = []
    without = []
    unknown = {}

    for rec in cicy_entries:
        need = required_symmetry_order(int(rec["euler"]), generations)
        if need is None:
            continue
        sym = by_num.get(rec["num"])
        orders, bad = symmetry_orders(sym) if sym else ([], [])
        for name in bad:
            unknown[name] = unknown.get(name, 0) + 1

        if need in orders:
            exact.append({"num": rec["num"], "euler": rec["euler"],
                          "order": need})
        elif any(o % need == 0 for o in orders if o):
            divisible.append({"num": rec["num"], "euler": rec["euler"],
                              "required": need, "available": sorted(orders)})
        else:
            without.append({"num": rec["num"], "euler": rec["euler"],
                            "required": need, "available": sorted(orders)})

    return {
        "generations": generations,
        "exact": exact,
        "by_divisibility": divisible,
        "candidates_without_symmetry": without,
        "unknown_groups": unknown,
        "n_exact": len(exact),
        "n_by_divisibility": len(divisible),
        "n_without": len(without),
    }


def coverage_report(cicy_entries, symmetry_entries):
    """How much of the list carries any recorded freely acting symmetry."""
    by_num = {r["num"]: r for r in symmetry_entries}
    with_any = 0
    total_groups = 0
    order_counts = {}
    unknown = {}

    for rec in cicy_entries:
        sym = by_num.get(rec["num"])
        if not sym:
            continue
        orders, bad = symmetry_orders(sym)
        for name in bad:
            unknown[name] = unknown.get(name, 0) + 1
        if orders:
            with_any += 1
            total_groups += len(orders)
            for o in orders:
                order_counts[o] = order_counts.get(o, 0) + 1

    return {
        "cicys": len(cicy_entries),
        "with_symmetry_record": len(by_num),
        "with_nontrivial_symmetry": with_any,
        "total_group_entries": total_groups,
        "order_distribution": dict(sorted(order_counts.items())),
        "unknown_groups": unknown,
    }
