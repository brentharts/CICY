r"""
pyCICY.knots -- knot diagrams, the Jones polynomial, and chirality.

Why this lives here
-------------------
:mod:`pyCICY.additivity` already asks what the failure of additivity of the
unknotting number,

    Brittenham and Hermiller, "Unknotting number is not additive under
    connected sum", arXiv:2506.24088,

looks like for the split web of CICY configurations. That module reasons
about the *analogy*. This one computes the knot side directly, because a
short follow-up to Brittenham and Hermiller turns out to be checkable in a
few lines:

    Wang and Zhang, "A remark on the counterexample to the unknotting number
    conjecture", arXiv:2507.14265,

observed that the two diagrams of K15n81556 appearing in the Brittenham and
Hermiller argument do not represent the same knot but a chiral knot and its
mirror image, and that this can be seen from the Jones polynomial.
:func:`chirality_report` reproduces exactly that computation, and it is a
regression test in ``tests/test_knots.py``.

The wider motivation is that mirroring a knot is the same formal move as
mirror symmetry elsewhere in this package. On the knot side it is
V(t) -> V(1/t); on the local Calabi-Yau side of :mod:`pyCICY.quantum_curve`
it is the reflection of the Newton polygon; on the compact side it is the
exchange of Hodge numbers. Whether the object is fixed by that move is a
question one can ask uniformly, and the answers differ in instructive ways.

Representation
--------------
A diagram is a list of PD (planar diagram) crossings together with a sign for
each. A crossing is a 4-tuple (a, b, c, d) of arc labels read
counter-clockwise starting from the incoming under-strand, so that

    a = under-in,   c = under-out,
    d = over-in and b = over-out   when the sign is +1,
    b = over-in and d = over-out   when the sign is -1.

When signs are not supplied they are inferred from the arc numbering by
:func:`infer_signs`, which is valid whenever arcs are labelled consecutively
along the orientation of the knot, as they are in the standard tables. The
rule was calibrated against, and agrees with, every diagram in :data:`KNOTS`.

Nothing here needs SnapPy, Sage or spherogram. The diagrams in :data:`KNOTS`
are stored as data, including the fifteen-crossing census knot K15n81556 that
the Brittenham-Hermiller argument passes through; :func:`from_census` is an
optional convenience for pulling further diagrams if SnapPy happens to be
installed.

Conventions
-----------
The Kauffman bracket is normalised so that the closure of the positive braid
sigma_1^3 has V = -t^-4 + t^-3 + t^-1, which agrees with the Knot Atlas entry
for 3_1. The unknot has V = 1.

The diagrams in :data:`KNOTS` are normalised to have non-negative writhe.
Standard tables do not fix the chirality of a knot canonically, so some rule
is needed if ``from_name`` and :func:`torus_knot` are to agree; this one
makes them agree, and the tests check that ``from_name("3_1")`` and
``torus_knot(2, 3)`` have the same Jones polynomial, likewise for 5_1, 7_1
and 8_19 = T(3,4). K15n81556 already has writhe +1 and is stored exactly as
it comes from the census, since its chirality is the whole point of it.

What is computed and what is quoted
-----------------------------------
Jones polynomials, determinants, chirality, connected sums and crossing
changes are computed from the diagram. Unknotting numbers are *not*
computed -- they are minimal move counts over all diagrams and are not
accessible this way. :data:`UNKNOTTING` quotes them from the literature with
attribution, and :func:`unknotting_search` performs an honest but strictly
diagram-dependent search which can only ever establish upper bounds; see its
docstring for why it does not, and cannot, rediscover the
Brittenham-Hermiller result by brute force.
"""

import itertools as it
from collections import defaultdict

__all__ = [
    "Laurent", "Knot", "KNOTS", "UNKNOTTING",
    "from_pd", "from_braid", "torus_knot", "from_name", "from_census",
    "infer_signs", "unknot", "chirality_report", "additivity_report",
    "unknotting_search",
]


# ------------------------------------------------------------------- Laurent

class Laurent(object):
    """A Laurent polynomial in one variable, backed by {exponent: coefficient}."""

    __slots__ = ("c",)

    def __init__(self, coeffs=None):
        c = {}
        for e, v in (coeffs or {}).items():
            if v:
                c[int(e)] = v
        self.c = c

    @classmethod
    def monomial(cls, exponent=0, coeff=1):
        return cls({exponent: coeff})

    def __bool__(self):
        return bool(self.c)

    def __eq__(self, other):
        if isinstance(other, (int, float)):
            other = Laurent({0: other})
        return isinstance(other, Laurent) and self.c == other.c

    def __hash__(self):
        return hash(tuple(sorted(self.c.items())))

    def __add__(self, other):
        out = dict(self.c)
        for e, v in other.c.items():
            out[e] = out.get(e, 0) + v
        return Laurent(out)

    def __neg__(self):
        return Laurent({e: -v for e, v in self.c.items()})

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Laurent({e: v * other for e, v in self.c.items()})
        out = defaultdict(int)
        for e1, v1 in self.c.items():
            for e2, v2 in other.c.items():
                out[e1 + e2] += v1 * v2
        return Laurent(out)

    __rmul__ = __mul__

    def shift(self, k):
        """Multiply by t^k."""
        return Laurent({e + k: v for e, v in self.c.items()})

    def invert_variable(self):
        """t -> 1/t, which is what mirroring a knot does to its Jones polynomial."""
        return Laurent({-e: v for e, v in self.c.items()})

    def is_palindromic(self):
        return self.c == self.invert_variable().c

    def evaluate(self, t):
        return sum(v * t ** e for e, v in self.c.items())

    def degrees(self):
        return (min(self.c), max(self.c)) if self.c else (0, 0)

    def __str__(self):
        if not self.c:
            return "0"
        parts = []
        for e in sorted(self.c):
            v = self.c[e]
            sign = "-" if v < 0 else ("+" if parts else "")
            a = abs(v)
            if e == 0:
                term = str(a)
            else:
                coeff = "" if a == 1 else str(a)
                term = "{}t^{}".format(coeff, e)
            parts.append((sign + " " if sign else "") + term)
        return " ".join(parts).strip()

    __repr__ = __str__


# --------------------------------------------------------------- sign rule

def infer_signs(pd):
    """Crossing signs from a PD code whose arcs run consecutively.

    With 2n arcs labelled along the orientation, the over-strand of a
    crossing (a, b, c, d) runs d -> b when the crossing is positive and
    b -> d when it is negative, so comparing b and d modulo 2n decides the
    sign. Raises ValueError when the labelling does not permit the test.
    """
    m = 2 * len(pd)
    out = []
    for a, b, c, d in pd:
        if b == (d + 1) % m:
            out.append(1)
        elif d == (b + 1) % m:
            out.append(-1)
        else:
            raise ValueError(
                "cannot infer the sign of crossing {}; arc labels are not "
                "consecutive along the knot, so pass signs explicitly"
                .format((a, b, c, d)))
    return out


# -------------------------------------------------------------- union-find

def _find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _union(parent, a, b):
    ra, rb = _find(parent, a), _find(parent, b)
    if ra != rb:
        parent[ra] = rb


# -------------------------------------------------------------------- Knot

class Knot(object):
    """A knot presented by a planar diagram.

    Parameters
    ----------
    pd : sequence of 4-tuples
        PD crossings, counter-clockwise from the incoming under-strand.
    signs : sequence of +-1, optional
        Crossing signs. Inferred by :func:`infer_signs` when omitted.
    name : str, optional
    """

    def __init__(self, pd, signs=None, name=None):
        self.pd = [tuple(int(x) for x in c) for c in pd]
        for c in self.pd:
            if len(c) != 4:
                raise ValueError("a PD crossing needs four arc labels, got %r" % (c,))
        if signs is None:
            signs = infer_signs(self.pd) if self.pd else []
        signs = [int(s) for s in signs]
        if len(signs) != len(self.pd):
            raise ValueError("got %d crossings but %d signs"
                             % (len(self.pd), len(signs)))
        if any(s not in (1, -1) for s in signs):
            raise ValueError("crossing signs must be +1 or -1")
        self.signs = signs
        self.name = name

    # ------------------------------------------------------------ basics

    def __len__(self):
        return len(self.pd)

    @property
    def n_crossings(self):
        return len(self.pd)

    @property
    def arcs(self):
        return sorted({a for c in self.pd for a in c})

    def writhe(self):
        return sum(self.signs)

    def _ends(self):
        """Map each arc to (outgoing (crossing, slot), incoming (crossing, slot))."""
        out, inc = {}, {}
        for i, (c, s) in enumerate(zip(self.pd, self.signs)):
            over_in, over_out = (3, 1) if s == 1 else (1, 3)
            inc[c[0]] = (i, 0)
            out[c[2]] = (i, 2)
            inc[c[over_in]] = (i, over_in)
            out[c[over_out]] = (i, over_out)
        return out, inc

    def n_components(self):
        """Number of link components; a knot diagram has exactly one."""
        out, inc = self._ends()
        if set(out) != set(inc):
            raise ValueError("diagram is not a closed oriented link")
        # follow each arc through its crossing to the next arc
        nxt = {}
        for arc, (i, slot) in inc.items():
            c, s = self.pd[i], self.signs[i]
            if slot == 0:
                nxt[arc] = c[2]
            else:
                nxt[arc] = c[1] if s == 1 else c[3]
        seen, comps = set(), 0
        for arc in self.arcs:
            if arc in seen:
                continue
            comps += 1
            cur = arc
            while cur not in seen:
                seen.add(cur)
                cur = nxt[cur]
        return comps

    def is_knot(self):
        return self.n_components() == 1

    # --------------------------------------------------------- operations

    def crossing_change(self, i):
        """Swap over and under at crossing ``i``. An involution."""
        pd = list(self.pd)
        signs = list(self.signs)
        a, b, c, d = pd[i]
        if signs[i] == 1:
            pd[i] = (d, a, b, c)
            signs[i] = -1
        else:
            pd[i] = (b, c, d, a)
            signs[i] = 1
        nm = None if self.name is None else self.name + "[x%d]" % i
        return Knot(pd, signs, name=nm)

    def crossing_changes(self, indices):
        """Apply crossing changes at every index in ``indices``."""
        k = self
        for i in indices:
            k = k.crossing_change(i)
        return k

    def mirror(self):
        """The mirror image: every crossing changed."""
        k = self.crossing_changes(range(len(self.pd)))
        k.name = None if self.name is None else "m" + self.name
        return k

    def relabel(self, offset):
        """Shift every arc label by ``offset``."""
        return Knot([tuple(a + offset for a in c) for c in self.pd],
                    self.signs, name=self.name)

    def connected_sum(self, other, arc=None, other_arc=None):
        """The connected sum of two knots, spliced respecting orientation.

        One arc of each diagram is cut and the free ends cross-joined, the
        outgoing end of each to the incoming end of the other, so the result
        is again a single component.
        """
        if not self.pd:
            return Knot(other.pd, other.signs, name=other.name)
        if not other.pd:
            return Knot(self.pd, self.signs, name=self.name)
        shift = max(self.arcs) + 1
        rhs = other.relabel(shift)
        x = self.arcs[0] if arc is None else arc
        y = (rhs.arcs[0] if other_arc is None else other_arc + shift)

        out1, inc1 = self._ends()
        out2, inc2 = rhs._ends()
        u = max(self.arcs + rhs.arcs) + 1
        v = u + 1

        pd = [list(c) for c in self.pd] + [list(c) for c in rhs.pd]
        n1 = len(self.pd)
        i, s = out1[x]
        pd[i][s] = u                      # x leaves -> becomes u
        i, s = inc2[y]
        pd[n1 + i][s] = u                 # -> enters where y entered
        i, s = out2[y]
        pd[n1 + i][s] = v                 # y leaves -> becomes v
        i, s = inc1[x]
        pd[n1 * 0 + i][s] = v             # -> enters where x entered

        nm = None
        if self.name and other.name:
            nm = "{} # {}".format(self.name, other.name)
        return Knot([tuple(c) for c in pd], self.signs + rhs.signs, name=nm)

    __add__ = connected_sum

    # ----------------------------------------------------------- invariants

    def kauffman_bracket(self):
        """The Kauffman bracket as a :class:`Laurent` polynomial in A."""
        n = len(self.pd)
        if n == 0:
            return Laurent({0: 1})
        arcs = self.arcs
        idx = {a: i for i, a in enumerate(arcs)}
        total = defaultdict(int)
        loop_factor = Laurent({2: -1, -2: -1})       # -A^2 - A^-2
        for state in range(1 << n):
            parent = list(range(len(arcs)))
            a_count = 0
            for j, (p, q, r, s) in enumerate(self.pd):
                if (state >> j) & 1:                  # A-smoothing: p-q, r-s
                    a_count += 1
                    _union(parent, idx[p], idx[q])
                    _union(parent, idx[r], idx[s])
                else:                                 # B-smoothing: p-s, q-r
                    _union(parent, idx[p], idx[s])
                    _union(parent, idx[q], idx[r])
            loops = len({_find(parent, i) for i in range(len(arcs))})
            poly = Laurent({2 * a_count - n: 1})
            for _ in range(loops - 1):
                poly = poly * loop_factor
            for e, v in poly.c.items():
                total[e] += v
        return Laurent(total)

    def jones(self):
        """The Jones polynomial V(t), normalised to the Knot Atlas convention."""
        br = self.kauffman_bracket()
        w = self.writhe()
        sign = -1 if w % 2 else 1
        out = defaultdict(int)
        for e, v in br.c.items():
            a = e - 3 * w
            if a % 4:
                raise ValueError("half-integer powers: this is a link, not a knot")
            out[a // 4] += sign * v
        return Laurent(out)

    def determinant(self):
        """|V(-1)|, the determinant of the knot."""
        return abs(int(round(self.jones().evaluate(-1))))

    def is_chiral(self):
        """True when the Jones polynomial distinguishes the knot from its mirror.

        A sufficient but not necessary test: a knot whose Jones polynomial is
        palindromic may still be chiral, so ``False`` means "not detected as
        chiral by this invariant", not "amphichiral".
        """
        return not self.jones().is_palindromic()

    def jones_is_trivial(self):
        """True when V = 1. Necessary but not sufficient for being unknotted."""
        return self.jones() == Laurent({0: 1})

    # ----------------------------------------------------------------- misc

    def describe(self):
        j = self.jones()
        return ("{:<12} {:>3} crossings  writhe {:>+3}  det {:>4}  "
                "chiral {:<5}  V = {}".format(
                    self.name or "?", len(self.pd), self.writhe(),
                    self.determinant(), str(self.is_chiral()), j))

    def __repr__(self):
        return "Knot({} crossings, name={!r})".format(len(self.pd), self.name)


# ------------------------------------------------------------- constructors

def from_pd(pd, signs=None, name=None):
    return Knot(pd, signs, name)


def unknot():
    """The zero-crossing diagram."""
    return Knot([], [], name="0_1")


def from_braid(word, strands=None, name=None):
    r"""Closure of a braid word.

    ``word`` is a sequence of non-zero integers, ``k`` meaning the generator
    sigma_k and ``-k`` its inverse, with strands numbered from one. The
    closure is taken by joining the right-hand ends back to the left.
    """
    word = [int(g) for g in word]
    if any(g == 0 for g in word):
        raise ValueError("braid generators are numbered from one")
    if strands is None:
        strands = max(abs(g) for g in word) + 1 if word else 1
    if strands < 2:
        return unknot()

    cur = list(range(strands))
    nxt = strands
    pd, signs = [], []
    for g in word:
        k = abs(g) - 1
        if k + 1 >= strands:
            raise ValueError("generator %d needs more than %d strands" % (g, strands))
        p, q = cur[k], cur[k + 1]
        r, s = nxt, nxt + 1
        nxt += 2
        if g > 0:
            # left strand passes over: over-in p, over-out s; under q -> r.
            # sign +1 puts over-in in slot d and over-out in slot b.
            pd.append((q, s, r, p))
            signs.append(1)
        else:
            # left strand passes under: under p -> s; over q -> r.
            # sign -1 puts over-in in slot b and over-out in slot d.
            pd.append((p, q, s, r))
            signs.append(-1)
        cur[k], cur[k + 1] = r, s

    # close the braid: identify the final labels with the initial ones
    rename = {cur[i]: i for i in range(strands)}
    pd = [tuple(rename.get(a, a) for a in c) for c in pd]
    return Knot(pd, signs, name=name)


def torus_knot(p, q):
    """The (p, q) torus knot as the closure of (sigma_1 ... sigma_{p-1})^q."""
    from math import gcd
    if gcd(p, q) != 1:
        raise ValueError("(%d,%d) is a link, not a knot" % (p, q))
    word = list(range(1, p)) * q
    return from_braid(word, strands=p, name="T({},{})".format(p, q))


def from_name(name):
    """A knot from the built-in table."""
    if name not in KNOTS:
        raise KeyError("unknown knot {!r}; known: {}".format(
            name, ", ".join(sorted(KNOTS))))
    pd = KNOTS[name]
    return Knot(pd, name=name)


def from_census(name):
    """Pull a diagram from the SnapPy census, if SnapPy is installed.

    Only needed for diagrams outside :data:`KNOTS`. The census knot used by
    Brittenham and Hermiller, K15n81556, is already in the table so that the
    interesting case works with no optional dependency at all.
    """
    if name in KNOTS:
        return from_name(name)
    try:
        import snappy
        import spherogram
    except ImportError:                                  # pragma: no cover
        raise ImportError(
            "from_census needs snappy and spherogram; "
            "pip install snappy spherogram snappy_15_knots")
    dt = snappy.HTLinkExteriors[name].DT_code()          # pragma: no cover
    link = spherogram.Link("DT:" + str(dt))              # pragma: no cover
    return Knot(link.PD_code(),                          # pragma: no cover
                [c.sign for c in link.crossings], name=name)


# ---------------------------------------------------------------- the table
#
# PD codes with arcs numbered consecutively along the knot, so that signs are
# inferable. Rolfsen names follow the Knot Atlas, and K15n81556 is the
# fifteen-crossing census knot through which the Brittenham-Hermiller
# argument passes.

KNOTS = {
    "3_1": [(2, 0, 3, 5), (0, 4, 1, 3), (4, 2, 5, 1)],
    "4_1": [(7, 4, 0, 5), (3, 0, 4, 1), (1, 7, 2, 6), (5, 3, 6, 2)],
    "5_1": [(4, 0, 5, 9), (0, 6, 1, 5), (6, 2, 7, 1), (2, 8, 3, 7),
            (8, 4, 9, 3)],
    "5_2": [(4, 0, 5, 9), (0, 6, 1, 5), (8, 2, 9, 1), (2, 8, 3, 7),
            (6, 4, 7, 3)],
    "6_1": [(6, 11, 7, 0), (0, 5, 1, 6), (10, 2, 11, 1), (2, 10, 3, 9),
            (8, 4, 9, 3), (4, 8, 5, 7)],
    "6_2": [(11, 7, 0, 6), (7, 1, 8, 0), (1, 9, 2, 8), (5, 3, 6, 2),
            (3, 10, 4, 11), (9, 4, 10, 5)],
    "6_3": [(8, 11, 9, 0), (0, 4, 1, 3), (6, 2, 7, 1), (2, 8, 3, 7),
            (4, 9, 5, 10), (10, 5, 11, 6)],
    "7_1": [(6, 0, 7, 13), (0, 8, 1, 7), (8, 2, 9, 1), (2, 10, 3, 9),
            (10, 4, 11, 3), (4, 12, 5, 11), (12, 6, 13, 5)],
    "7_2": [(10, 0, 11, 13), (0, 10, 1, 9), (8, 2, 9, 1), (2, 8, 3, 7),
            (6, 4, 7, 3), (4, 12, 5, 11), (12, 6, 13, 5)],
    "7_3": [(8, 0, 9, 13), (0, 8, 1, 7), (6, 2, 7, 1), (2, 10, 3, 9),
            (10, 4, 11, 3), (4, 12, 5, 11), (12, 6, 13, 5)],
    "7_4": [(13, 7, 0, 6), (5, 1, 6, 0), (1, 11, 2, 10), (9, 3, 10, 2),
            (3, 9, 4, 8), (11, 5, 12, 4), (7, 13, 8, 12)],
    "7_5": [(4, 0, 5, 13), (0, 4, 1, 3), (8, 2, 9, 1), (2, 10, 3, 9),
            (10, 6, 11, 5), (6, 12, 7, 11), (12, 8, 13, 7)],
    "7_6": [(13, 9, 0, 8), (7, 1, 8, 0), (1, 12, 2, 13), (11, 2, 12, 3),
            (3, 7, 4, 6), (9, 5, 10, 4), (5, 11, 6, 10)],
    "7_7": [(13, 7, 0, 6), (5, 1, 6, 0), (1, 11, 2, 10), (9, 3, 10, 2),
            (3, 12, 4, 13), (7, 4, 8, 5), (11, 8, 12, 9)],
    "8_19": [(15, 5, 0, 4), (5, 1, 6, 0), (10, 2, 11, 1), (2, 14, 3, 13),
             (3, 9, 4, 8), (11, 7, 12, 6), (7, 13, 8, 12), (14, 10, 15, 9)],
    "8_20": [(6, 0, 7, 15), (11, 1, 12, 0), (1, 9, 2, 8), (2, 14, 3, 13),
             (14, 4, 15, 3), (4, 9, 5, 10), (10, 5, 11, 6), (7, 12, 8, 13)],
    "K15n81556": [(2, 0, 3, 29), (0, 10, 1, 9), (10, 2, 11, 1), (14, 3, 15, 4),
                  (4, 24, 5, 23), (24, 6, 25, 5), (6, 15, 7, 16),
                  (16, 7, 17, 8), (8, 17, 9, 18), (18, 11, 19, 12),
                  (25, 12, 26, 13), (13, 22, 14, 23), (19, 27, 20, 26),
                  (27, 21, 28, 20), (21, 29, 22, 28)],
}

# Unknotting numbers quoted from the literature, not computed here. The
# torus knot values are Kronheimer and Mrowka's resolution of the Milnor
# conjecture, u(T(p,q)) = (p-1)(q-1)/2; the small knots are classical table
# values. The K15n81556 entry is the upper bound used by Brittenham and
# Hermiller, arXiv:2506.24088.
UNKNOTTING = {
    "3_1": 1, "4_1": 1, "5_1": 2, "5_2": 1, "6_1": 1, "6_2": 1, "6_3": 1,
    "7_1": 3, "7_2": 1, "7_3": 2, "7_4": 2, "7_5": 2, "7_6": 1, "7_7": 1,
    "8_19": 3, "8_20": 1,
}


# ------------------------------------------------------------------ reports

def chirality_report(name="K15n81556"):
    """Reproduce the Wang-Zhang chirality check.

    Wang and Zhang, arXiv:2507.14265, note that the two diagrams of
    K15n81556 in the Brittenham-Hermiller argument represent a chiral knot
    and its mirror image rather than the same knot, and that the Jones
    polynomial sees this. That is what is evaluated here.
    """
    k = from_name(name) if isinstance(name, str) else name
    v = k.jones()
    vm = k.mirror().jones()
    return {
        "name": k.name,
        "crossings": len(k),
        "writhe": k.writhe(),
        "determinant": k.determinant(),
        "jones": v,
        "jones_mirror": vm,
        "mirror_is_inverse": vm == v.invert_variable(),
        "palindromic": v.is_palindromic(),
        "chiral": not v.is_palindromic(),
    }


def additivity_report():
    """Bookkeeping for the Brittenham-Hermiller connected sum 7_1 # m7_1.

    The Jones polynomial is multiplicative under connected sum and that is
    checked here. The unknotting numbers are quoted, not derived: u is a
    minimum over all diagrams and is not something a single diagram can
    settle. The point of the entry is the inequality

        u(7_1 # m7_1) <= 5 < 6 = u(7_1) + u(m7_1),

    which is the content of arXiv:2506.24088.
    """
    k = from_name("7_1")
    mk = k.mirror()
    total = k.connected_sum(mk)
    return {
        "left": k.name,
        "right": "m7_1",
        "sum_crossings": len(total),
        "sum_components": total.n_components(),
        "jones_left": k.jones(),
        "jones_right": mk.jones(),
        "jones_sum": total.jones(),
        "jones_multiplicative": total.jones() == k.jones() * mk.jones(),
        "u_left_quoted": UNKNOTTING["7_1"],
        "u_right_quoted": UNKNOTTING["7_1"],
        "u_sum_naive": 2 * UNKNOTTING["7_1"],
        "u_sum_upper_bound_BH": 5,
        "additive": False,
        "reference": "Brittenham and Hermiller, arXiv:2506.24088",
    }


def unknotting_search(knot, max_changes=2, limit=None):
    """Search for a set of crossing changes making the Jones polynomial trivial.

    This gives upper bounds only, and weak ones. Two independent reasons:

    * It works in a single fixed diagram. The unknotting number is a minimum
      over *all* diagrams of the knot, and the whole difficulty of
      arXiv:2506.24088 is that the economical route for 7_1 # m7_1 is not
      visible in the obvious diagram. Running this on the fourteen-crossing
      connected sum will not recover their bound of five, and that is the
      expected outcome rather than a bug.
    * V = 1 is necessary but not sufficient for being unknotted, so a hit is
      evidence and not proof.

    Cost grows as C(n, k) * 2^n bracket evaluations, so it is easy to make
    this take hours; ``limit`` caps the number of subsets tried.
    """
    n = len(knot)
    tried = 0
    for k in range(0, max_changes + 1):
        for combo in it.combinations(range(n), k):
            if limit is not None and tried >= limit:
                return {"found": None, "tried": tried, "exhausted": False}
            tried += 1
            if knot.crossing_changes(combo).jones_is_trivial():
                return {"found": list(combo), "changes": k,
                        "tried": tried, "exhausted": False}
    return {"found": None, "tried": tried, "exhausted": True}
