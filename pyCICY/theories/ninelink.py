r"""
pyCICY.theories.ninelink -- nine-link Yukawa textures, exactly.

Flavour space is ten-dimensional: the two 3x3 complex Yukawa matrices carry
36 real parameters, the U(3)^3 flavour rotations remove 26 of them, and what
survives is six masses, three mixing angles and one phase. Arkani-Hamed,
Figueiredo, Hall and Manzari observe that the same ten numbers can be carried
by Yukawa matrices with exactly nine non-zero entries between them and a
single irremovable phase -- *nine-link textures* -- and that when every such
texture is fitted to the data, the fitted phase piles up at multiples of pi/8
rather than spreading out.

The reason it piles up is not a fact about fits. For hierarchical matrices,
perturbative diagonalisation makes one angle of the unitarity triangle equal
to that phase at leading order, but only when texture zeros switch off one of
the mixings; otherwise numerator and denominator share a term and the angle
collapses to zero. So the clustering is a consequence of which links are
absent, which is combinatorics.

What is exact here
------------------
The whole link-diagram layer, with no fitting anywhere:

- the enumeration of nine-link textures, from the conditions alone: nine
  links, both matrices full rank, the diagram connected, the (3,3) entries
  present;
- their orbits under permutations of the three quark doublets and of the two
  sets of singlets;
- the unique closed loop of each diagram, and the rephasing-invariant
  monomial it carries, with the conjugation rule for links pointing into a
  doublet;
- the perfect matchings of each diagram, which are the terms of the
  determinant, and hence whether the phase can be placed so that
  ``arg det(Yu Yd) = 0`` -- the strong-CP question, answered by graph theory;
- the unitarity-triangle angles of any given pair of Yukawa matrices, by
  exact diagonalisation.

Two counts drop out of the enumeration that are not in the paper: there are
**1592** nine-link textures, falling into **36** orbits under permutations.
The paper's counts -- 156 equivalence classes, then (29, 35, 35) at fixed
phase -- are counts of *fits*, not of diagrams, and one diagram supports
several inequivalent fits, so the two are not in tension. They are also not
comparable, which is worth saying plainly.

What is not
-----------
Everything numerical: the scan over textures, the histogram of fitted phases,
the equivalence classes of fits, and the predicted ellipses. Those come from
a chi-squared minimisation against experimental data, and a scan reports the
minima it found in the box it searched. :exc:`~.flavorbase.NeedsFit` says so.

Reference
---------
Arkani-Hamed, Figueiredo, Hall and Manzari, *The Very Nearly Right Theory of
Flavor*, arXiv:2607.27315. Equation numbers below are that paper's, with
``S`` prefixes for the supplementary material.
"""

from itertools import permutations

import numpy as np
import sympy as sp

from .flavorbase import FlavorTheory, NeedsFit, register

__all__ = ["Texture", "enumerate_textures", "permutation_orbits", "census",
           "strong_cp_census", "ckm_from_yukawas", "unitarity_angles",
           "jarlskog", "example_texture", "example_check",
           "example_two_texture", "example_two_anomaly", "z8_ratio",
           "cycle_rank", "independent_phases", "square_classes",
           "cyclotomic_overlap", "monotile_compatibility",
           "yukawa_triangle_ratios", "pi_over_4_condition", "NineLinkTexture",
           "PDG"]


#: Central values at M_Z from the paper's Table S2, used for the arithmetic
#: checks below. Reference data, never fitted to here.
PDG = {
    "yu": 7.04e-6, "yc": 3.56e-3, "yt": 0.967,
    "yd": 1.54e-5, "ys": 3.06e-4, "yb": 1.630e-2,
    "Vus": 0.22517, "Vub": 0.003763, "Vcb": 0.04189,
    "Vcd": 0.22503, "Vtd": 0.00863, "Vts": 0.04117,
    "alpha_deg": 84.1, "beta_deg": 22.6, "gamma_deg": 66.4,
}


# ---------------------------------------------------------------------------
# link diagrams
# ---------------------------------------------------------------------------

def _cells(mask):
    """The (row, column) positions present in a 9-bit mask."""
    return [(k // 3, k % 3) for k in range(9) if mask >> k & 1]


def _perfect_matchings(mask):
    """Permutations sigma with every ``(i, sigma(i))`` present.

    These are the terms of the determinant: one per perfect matching of the
    bipartite graph. A matrix with no perfect matching is singular for every
    choice of entries, so this is also the full-rank test.
    """
    cells = set(_cells(mask))
    return [p for p in permutations(range(3))
            if all((i, p[i]) in cells for i in range(3))]


class Texture(object):
    """A nine-link texture: two masks, and everything that follows from them.

    The masks are 9-bit integers, bit ``3*i + j`` for entry ``(i, j)``. All
    the structure below -- the loop, the invariant, the determinant terms --
    is a property of the diagram and needs no numerical values at all.
    """

    def __init__(self, u_mask, d_mask):
        self.u = int(u_mask)
        self.d = int(d_mask)

    def __repr__(self):
        return "Texture(u=%s, d=%s)" % (sorted(self.cells("u")),
                                        sorted(self.cells("d")))

    def __eq__(self, other):
        return (self.u, self.d) == (other.u, other.d)

    def __hash__(self):
        return hash((self.u, self.d))

    # -- basics ------------------------------------------------------------

    def cells(self, sector):
        return _cells(self.u if sector == "u" else self.d)

    def links(self):
        """All nine links, as ``(sector, i, j)``."""
        return ([("u", i, j) for i, j in self.cells("u")]
                + [("d", i, j) for i, j in self.cells("d")])

    def split(self):
        """How the nine links divide between the two matrices."""
        return len(self.cells("u")), len(self.cells("d"))

    def matrix(self, sector, entries=None):
        """A sympy matrix for this sector, with symbols on the present links."""
        M = sp.zeros(3, 3)
        for i, j in self.cells(sector):
            M[i, j] = (entries[(sector, i, j)] if entries
                       else sp.Symbol("%s%d%d" % (sector, i + 1, j + 1),
                                      positive=True))
        return M

    # -- the graph ---------------------------------------------------------

    def _adjacency(self):
        adj = {}
        for s, i, j in self.links():
            a, b = ("q", i), (s, j)
            adj.setdefault(a, []).append((b, (s, i, j)))
            adj.setdefault(b, []).append((a, (s, i, j)))
        return adj

    def is_connected(self):
        """All nine nodes in one piece.

        With nine links and nine nodes, connected is the same as having
        exactly one independent loop -- the cycle rank is ``E - V + C``, so
        connectivity forces it to 1. That is why a single rephasing invariant
        and a connected diagram are the same condition, which is neater than
        checking for loops directly.
        """
        adj = self._adjacency()
        if len(adj) != 9:
            return False
        start = next(iter(adj))
        seen, stack = {start}, [start]
        while stack:
            x = stack.pop()
            for y, _ in adj[x]:
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
        return len(seen) == 9

    def is_valid(self):
        """Nine links, both sectors full rank, connected, both (3,3) present."""
        if bin(self.u).count("1") + bin(self.d).count("1") != 9:
            return False
        if not (self.u >> 8 & 1) or not (self.d >> 8 & 1):
            return False
        if not _perfect_matchings(self.u) or not _perfect_matchings(self.d):
            return False
        return self.is_connected()

    def loop(self):
        """The links of the unique closed loop, as a set.

        Found by stripping nodes of degree one until nothing else can be
        removed; what remains is the cycle.
        """
        adj = self._adjacency()
        deg = {n: len(v) for n, v in adj.items()}
        live = set(self.links())
        changed = True
        while changed:
            changed = False
            for n in list(deg):
                if deg[n] == 1:
                    for m, e in adj[n]:
                        if e in live:
                            live.discard(e)
                            deg[n] -= 1
                            deg[m] -= 1
                            changed = True
        return live

    def loop_walk(self):
        """The loop as an ordered walk, alternating doublets and singlets.

        Returns a list of ``(sector, i, j, conjugated)``. A link is conjugated
        when the walk traverses it *into* a doublet node, which is the rule
        that makes the product rephasing invariant.
        """
        live = self.loop()
        adj = {}
        for s, i, j in live:
            a, b = ("q", i), (s, j)
            adj.setdefault(a, []).append((b, (s, i, j)))
            adj.setdefault(b, []).append((a, (s, i, j)))
        start = next(n for n in adj if n[0] == "q")
        walk, node, used = [], start, set()
        while True:
            nxt = None
            for m, e in adj[node]:
                if e not in used:
                    nxt, edge = m, e
                    break
            if nxt is None:
                break
            used.add(edge)
            walk.append((edge[0], edge[1], edge[2], nxt[0] == "q"))
            node = nxt
            if node == start and len(used) == len(live):
                break
        return walk

    def rephasing_invariant(self, entries=None):
        """The loop monomial whose phase cannot be rotated away.

        With ``entries`` given as a dict of complex numbers or sympy
        expressions, returns the monomial itself; otherwise a symbolic one.
        """
        out = sp.Integer(1)
        for s, i, j, conj in self.loop_walk():
            val = (entries[(s, i, j)] if entries
                   else sp.Symbol("%s%d%d" % (s, i + 1, j + 1)))
            out *= sp.conjugate(val) if conj else val
        return out

    # -- determinants, and strong CP ---------------------------------------

    def determinant_terms(self, sector):
        """The perfect matchings of one sector: the terms of its determinant."""
        return _perfect_matchings(self.u if sector == "u" else self.d)

    def phase_placements(self):
        """Loop links on which the phase leaves ``arg det(Yu Yd)`` zero.

        Each determinant term is a perfect matching. If the phase sits on a
        link that belongs to no matching of its own sector, that sector's
        determinant never sees it and stays real. If every matching contains
        the link, the determinant is real times a phase, which is not the same
        thing at all.
        """
        out = []
        for s, i, j in sorted(self.loop()):
            if all(p[i] != j for p in self.determinant_terms(s)):
                out.append((s, i, j))
        return out

    def strong_cp_safe(self):
        """Whether the phase can be placed so the determinant stays real."""
        return bool(self.phase_placements())

    def is_diagonal(self, sector):
        mask = self.u if sector == "u" else self.d
        return sorted(_cells(mask)) == [(0, 0), (1, 1), (2, 2)]


# ---------------------------------------------------------------------------
# enumeration
# ---------------------------------------------------------------------------

_CACHE = {}


def enumerate_textures():
    """Every nine-link texture, from the conditions alone.

    Not a table: the loop runs over all pairs of masks and applies full rank,
    connectivity and the (3,3) requirement. Cached, because it is the input
    to everything else here.
    """
    if "textures" not in _CACHE:
        full = [m for m in range(512) if _perfect_matchings(m)]
        out = []
        for u in full:
            nu = bin(u).count("1")
            if not 3 <= nu <= 6 or not (u >> 8 & 1):
                continue
            for d in full:
                if bin(d).count("1") != 9 - nu or not (d >> 8 & 1):
                    continue
                t = Texture(u, d)
                if t.is_connected():
                    out.append(t)
        _CACHE["textures"] = out
    return _CACHE["textures"]


def _act(t, pq, pu, pd):
    u = d = 0
    for i, j in t.cells("u"):
        u |= 1 << (3 * pq[i] + pu[j])
    for i, j in t.cells("d"):
        d |= 1 << (3 * pq[i] + pd[j])
    return Texture(u, d)


def permutation_orbits():
    """Orbits under ``S3_Q x S3_uc x S3_dc``, the relabelling of the fields.

    Two textures related by permuting the quark doublets and the singlets are
    the same theory written differently. The orbits are computed on the
    diagrams; the paper's equivalence classes are of *fits* and are a
    different and larger count, since one diagram supports several minima.
    """
    if "orbits" not in _CACHE:
        S = list(permutations(range(3)))
        # orbits are taken on the unconstrained set, so the group acts freely
        # on the whole of it; the (3,3) condition is a choice of representative
        full = [m for m in range(512) if _perfect_matchings(m)]
        allt = []
        for u in full:
            nu = bin(u).count("1")
            if not 3 <= nu <= 6:
                continue
            for d in full:
                if bin(d).count("1") != 9 - nu:
                    continue
                t = Texture(u, d)
                if t.is_connected():
                    allt.append(t)
        seen, orbits = set(), []
        for t in allt:
            if (t.u, t.d) in seen:
                continue
            orb = {(_act(t, pq, pu, pd).u, _act(t, pq, pu, pd).d)
                   for pq in S for pu in S for pd in S}
            seen |= orb
            orbits.append(orb)
        _CACHE["orbits"] = orbits
    return _CACHE["orbits"]


def census():
    """Counts that follow from the enumeration, with nothing fitted."""
    ts = enumerate_textures()
    from collections import Counter
    return {
        "textures": len(ts),
        "splits": dict(Counter(t.split() for t in ts)),
        "loop_lengths": dict(Counter(len(t.loop()) for t in ts)),
        "orbits": len(permutation_orbits()),
    }


def strong_cp_census():
    """How many textures admit a real determinant, and what the failures are.

    The paper reports that of the textures surviving its fits, only five fail,
    and that all five are diagonal in one sector. Over the *full* enumeration
    that second statement does not hold, and this function returns the numbers
    rather than the claim: the failures are exactly the textures whose loop
    meets every perfect matching, and only some of those are diagonal.
    """
    ts = enumerate_textures()
    safe = [t for t in ts if t.strong_cp_safe()]
    bad = [t for t in ts if not t.strong_cp_safe()]
    single = [t for t in ts
              if len(t.determinant_terms("u")) == 1
              and len(t.determinant_terms("d")) == 1]
    return {
        "total": len(ts),
        "safe": len(safe),
        "failures": len(bad),
        "failures_diagonal_in_a_sector":
            sum(1 for t in bad if t.is_diagonal("u") or t.is_diagonal("d")),
        "single_matching_in_both": len(single),
        "safe_equals_single_matching":
            {(t.u, t.d) for t in safe} == {(t.u, t.d) for t in single},
    }


# ---------------------------------------------------------------------------
# from Yukawa matrices to the unitarity triangle
# ---------------------------------------------------------------------------

def ckm_from_yukawas(Yu, Yd):
    """The CKM matrix, by exact diagonalisation rather than by expansion.

    Only the left-handed rotations are physical; they diagonalise ``Y Y^dag``,
    and the CKM matrix is their product. Eigenvalues are sorted lightest
    first so the generation labelling is fixed.
    """
    def left(Y):
        w, U = np.linalg.eigh(np.asarray(Y) @ np.asarray(Y).conj().T)
        return U[:, np.argsort(w)].conj().T
    return left(Yu) @ left(Yd).conj().T


def unitarity_angles(V, degrees=False):
    """``(alpha, beta, gamma)`` from a CKM matrix, as defined in (1).

    Rephasing invariant, so the arbitrary phases left over by the
    diagonalisation drop out.
    """
    a = np.angle(-(V[2, 0] * np.conj(V[2, 2])) / (V[0, 0] * np.conj(V[0, 2])))
    b = np.angle(-(V[1, 0] * np.conj(V[1, 2])) / (V[2, 0] * np.conj(V[2, 2])))
    g = np.angle(-(V[0, 0] * np.conj(V[0, 2])) / (V[1, 0] * np.conj(V[1, 2])))
    return tuple(np.degrees(x) for x in (a, b, g)) if degrees else (a, b, g)


def jarlskog(V):
    """The Jarlskog invariant, twice the area of any unitarity triangle."""
    return float(np.imag(V[0, 0] * V[1, 1] * np.conj(V[0, 1])
                         * np.conj(V[1, 0])))


# ---------------------------------------------------------------------------
# the worked examples, checked against exact diagonalisation
# ---------------------------------------------------------------------------

def example_texture(which, eps=0.1, coefficients=None, phase_on=None):
    """One of the paper's three worked textures, as explicit matrices.

    ``which`` is 1, 2 or 3 for the textures of (S38), (S43) and (S48), whose
    leading-order phases are ``pi/2``, ``-pi/8`` and ``5pi/8`` and which
    predict ``alpha``, ``beta`` and ``gamma`` respectively.

    ``coefficients`` supplies the order-one numbers multiplying each power of
    ``eps``. They must be *generic*: setting them all to one makes two
    singular values degenerate in example 2 and the diagonalisation
    ill-defined, which is a property of the example rather than of the method.

    ``phase_on`` overrides where the phase sits. See
    :func:`example_check` for why example 2 needs it.
    """
    c = coefficients or {}

    def k(name, default=1.0):
        return c.get(name, default)

    e = eps
    if which == 1:
        Yu = np.array([[0, 1j * k("u12") * e ** 4, 0],
                       [k("u21") * e ** 5, k("u22") * e ** 3, 0],
                       [0, 0, k("u33")]], dtype=complex)
        Yd = np.array([[0, k("d12") * e ** 5, 0],
                       [k("d21") * e ** 5, k("d22") * e ** 4,
                        k("d23") * e ** 3],
                       [0, 0, k("d33") * e ** 2]], dtype=complex)
        return Yu, Yd
    if which == 2:
        ph = np.exp(-1j * np.pi / 8)
        where = phase_on or ("d", 0, 2)
        def f(s, i, j):
            return ph if (s, i, j) == where else 1.0
        Yu = np.array([[k("u11") * e ** 5, 0, k("u13") * e ** 2],
                       [0, k("u22") * e ** 2, 0],
                       [0, 0, k("u33")]], dtype=complex)
        Yd = np.array([[k("d11") * e ** 5, 0, k("d13") * e ** 4 * f("d", 0, 2)],
                       [0, 0, k("d23") * e ** 3],
                       [0, k("d32") * e ** 2 * f("d", 2, 1),
                        k("d33") * e ** 2]], dtype=complex)
        return Yu, Yd
    if which == 3:
        ph = np.exp(1j * 5 * np.pi / 8)
        Yu = np.array([[k("u11") * e ** 5, 0, 0],
                       [0, k("u22") * e ** 2, k("u23") * e],
                       [0, 0, k("u33")]], dtype=complex)
        Yd = np.array([[0, ph * k("d12") * e ** 4, k("d13") * e ** 4],
                       [0, k("d22") * e ** 3, 0],
                       [k("d31") * e ** 3, 0, k("d33") * e ** 2]],
                      dtype=complex)
        return Yu, Yd
    raise ValueError("which must be 1, 2 or 3")


def _generic(seed=7):
    rng = np.random.default_rng(seed)
    names = ["u11", "u12", "u13", "u21", "u22", "u23", "u31", "u32", "u33",
             "d11", "d12", "d13", "d21", "d22", "d23", "d31", "d32", "d33"]
    return {n: float(rng.uniform(0.6, 1.6)) for n in names}


def example_check(which, eps=0.1, coefficients=None, phase_on=None):
    """Compare exact diagonalisation against the paper's closed form.

    The paper derives each angle to next-to-leading order and expresses the
    correction in terms of CKM elements and mass ratios. That is checkable
    without repeating the expansion: diagonalise the same matrices exactly,
    compute the angle, and evaluate the closed form on the CKM matrix the
    diagonalisation produced. The residual must vanish as ``eps`` does.

    Returns the exact angle, the closed form, and the residual.
    """
    coefficients = coefficients or _generic()
    Yu, Yd = example_texture(which, eps, coefficients, phase_on)
    V = ckm_from_yukawas(Yu, Yd)
    a, b, g = unitarity_angles(V)
    if which == 1:
        exact, closed = a, np.pi / 2 - abs(V[2, 0] * V[0, 2]
                                           / (V[1, 2] * V[2, 1]))
    elif which == 2:
        ys, yb = sorted(np.linalg.svd(Yd)[1])[1:]
        exact = b
        closed = (np.pi / 8 - 0.5 * np.sqrt(2 - np.sqrt(2))
                  * abs(V[0, 1] / V[2, 0]) * (ys ** 2 / yb ** 2))
    else:
        exact = abs(g)                      # the orientation of phi is a
        closed = (3 * np.pi / 8            # convention; compare magnitudes
                  - 0.5 * np.sqrt(2 + np.sqrt(2))
                  * abs(V[1, 2] * V[0, 2] / V[0, 1]))
    return {"exact": exact, "closed_form": closed,
            "residual": exact - closed, "eps": eps}


def example_two_texture():
    """The link diagram of the paper's second worked example, as a Texture."""
    u = 0
    for i, j in [(0, 0), (0, 2), (1, 1), (2, 2)]:
        u |= 1 << (3 * i + j)
    d = 0
    for i, j in [(0, 0), (0, 2), (1, 2), (2, 1), (2, 2)]:
        d |= 1 << (3 * i + j)
    return Texture(u, d)


def example_two_anomaly(eps=(0.1, 0.05, 0.025), seed=7):
    """What happens to the second worked example, reported rather than fixed.

    Two separate findings, both computed:

    *The printed phase is not in the loop.* The diagram of (S43) has nine
    links on nine nodes, so exactly one closed loop, and it is
    ``{Yu13, Yu33, Yd13, Yd33}``. The entry the paper marks with the phase,
    ``Yd32``, is not in it. A phase there is removable by rephasing, and
    diagonalising the matrices as printed gives ``beta = 0`` exactly, not
    ``pi/8``.

    *Moving it into the loop fixes the leading order and not the correction.*
    Placing the phase on ``Yd13`` -- the down-type loop entry with the
    smallest indices, which is the paper's own stated rule -- gives
    ``beta = pi/8`` at leading order, as advertised. But the printed
    next-to-leading correction then does not describe the residual: the exact
    departure from ``pi/8`` falls like ``eps^4`` while the quoted correction
    is of order ``eps^2``, so their difference does not vanish with ``eps``.

    The first of these looks like a transcription slip. The second may follow
    from it -- with the phase in a different link the subleading structure is
    a different calculation -- or may be independent. This function returns
    the numbers and takes no view.

    Compare :func:`example_check` on examples 1 and 3, where the residual
    falls like ``eps^4`` and the closed forms are reproduced.
    """
    t = example_two_texture()
    loop = sorted(t.loop())
    coeffs = _generic(seed)
    as_printed, relocated = [], []
    for e in eps:
        r = example_check(2, eps=e, coefficients=coeffs,
                          phase_on=("d", 2, 1))
        as_printed.append({"eps": e, "beta": r["exact"]})
        r = example_check(2, eps=e, coefficients=coeffs,
                          phase_on=("d", 0, 2))
        relocated.append({"eps": e, "beta": r["exact"],
                          "departure_from_pi_8": r["exact"] - np.pi / 8,
                          "closed_form_residual": r["residual"]})
    return {"loop": loop,
            "printed_phase_link": ("d", 2, 1),
            "printed_phase_in_loop": ("d", 2, 1) in t.loop(),
            "as_printed": as_printed,
            "relocated_to": ("d", 0, 2),
            "relocated": relocated}


def cycle_rank(texture):
    """The number of independent loops: ``E - V + C``. Exact.

    The physical content of a texture's phases is a graph invariant. Every
    link carries a phase, every field can be rephased, and the phases that
    survive are exactly the independent cycles of the link diagram -- the
    first homology of the graph. So the count of irremovable CP phases is
    ``links - nodes + components``, with no reference to the entries at all.

    Nine-link textures are the case where this equals one, which is what
    makes their single phase the whole of CP violation. Written this way the
    statement generalises: a texture with more links has more phases, one per
    extra independent loop, and the paper's analysis of deformations away
    from nine links is an analysis of what those extra loops do.
    """
    links = texture.links()
    nodes = set()
    adj = {}
    for s, i, j in links:
        a, b = ("q", i), (s, j)
        nodes |= {a, b}
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    seen, comps = set(), 0
    for n in nodes:
        if n in seen:
            continue
        comps += 1
        stack = [n]
        seen.add(n)
        while stack:
            x = stack.pop()
            for y in adj.get(x, []):
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
    return len(links) - len(nodes) + comps


def independent_phases(texture):
    """The number of irremovable CP phases. Same as :func:`cycle_rank`."""
    return cycle_rank(texture)


# ---------------------------------------------------------------------------
# does the monotile's arithmetic reach the flavour phases?
# ---------------------------------------------------------------------------

def _squarefree(n):
    """The squarefree part of n, keeping its sign: -12 -> -3, 12 -> 3.

    The sign matters for imaginary fields: Q(sqrt -3, i) has the four
    classes {1, -1, -3, 3}, and dropping signs would report {1, 3} -- the
    same as the real field Q(sqrt 3).
    """
    sign = -1 if int(n) < 0 else 1
    n = abs(int(n))
    out, d = 1, 2
    while d * d <= n:
        e = 0
        while n % d == 0:
            n //= d
            e += 1
        if e % 2:
            out *= d
        d += 1
    return sign * (out * n if n > 1 else out)


def square_classes(generators):
    """The squarefree classes of the multiquadratic field they generate.

    A field ``Q(sqrt(a), sqrt(b))`` has quadratic subfields indexed by the
    group generated by ``a`` and ``b`` modulo squares, and ``sqrt(k)`` lies in
    it exactly when ``k`` is in that group. Elementary, exact, and enough to
    settle the question below without any numerics.
    """
    cls = {1}
    for g in generators:
        cls |= {_squarefree(c * g) for c in cls}
    for _ in range(3):
        cls |= {_squarefree(a * b) for a in cls for b in cls}
    return sorted(cls)


def cyclotomic_overlap(m, n):
    """``Q(zeta_m) cap Q(zeta_n) = Q(zeta_gcd(m,n))``.

    The discrete phases two symmetries can both support are the ones in the
    intersection of their cyclotomic fields, and that intersection is
    governed by the greatest common divisor. Which turns a vague question
    about whether two structures are compatible into an integer.
    """
    from math import gcd
    g = gcd(int(m), int(n))
    return {"m": int(m), "n": int(n), "intersection_order": g,
            "phases": "multiples of 2*pi/%d" % g}


def monotile_compatibility():
    """Can a Spectre-monotile vacuum supply the flavour phases? Mostly not.

    Both structures are built on *nine* things -- nine links here, nine
    substitution species in :mod:`pyCICY.theories.spectre` -- and the
    temptation to read something into that is the reason this function
    exists. The nines are unrelated: nine links is ten flavour observables
    minus the one phase, and nine species is the number of metatiles the
    substitution happens to need. The substitution matrix has 56 non-zero
    entries and row sums up to 16; no texture's diagram is anything like it.

    The arithmetic settles it properly. The special angles need ``sqrt 2``:
    ``tan(pi/8) = sqrt2 - 1``, and the triangle's side ratios are
    ``cot(pi/8) = 1 + sqrt2`` and ``1/cos(pi/8)``. The substitution's
    inflation factor is ``lambda = (sqrt6 + sqrt10)/2`` with
    ``lambda^2 = 4 + sqrt15``, so its field is ``Q(sqrt6, sqrt10)``, whose
    quadratic subfields are exactly ``Q(sqrt6)``, ``Q(sqrt10)`` and
    ``Q(sqrt15)``. ``sqrt2`` is not among them, and neither is ``sqrt3`` --
    the monotile's own placement field -- which sits in the geometry rather
    than in the inflation.

    One corner survives, and it is the interesting part.
    ``Q(zeta_8) cap Q(zeta_12) = Q(zeta_4) = Q(i)``: a vacuum whose phases are
    twelve-fold, as the monotile's geometry is, and a texture wanting
    eight-fold phases share exactly the multiples of ``pi/2``. That is
    ``alpha``, and only ``alpha`` -- the peak with nine equivalence classes,
    the right-triangle one, the one whose textures are also the ones whose
    determinants are naturally real. So a monotile-flavoured spontaneous CP
    violation could deliver ``alpha = pi/2`` and could not deliver
    ``beta = pi/8`` or ``gamma = 3pi/8``.

    And it says what would be needed instead: ``lcm(8, 12) = 24``, so a
    twenty-four-fold vacuum contains both, ``Q(zeta_24) ⊃ Q(zeta_8)``.
    """
    inflation = square_classes([6, 10])
    return {
        "inflation_field": "Q(sqrt6, sqrt10)",
        "inflation_square_classes": inflation,
        "needs_sqrt2_for_tan_pi_8": True,
        "sqrt2_in_inflation_field": _squarefree(2) in inflation,
        "sqrt3_in_inflation_field": _squarefree(3) in inflation,
        "sqrt15_in_inflation_field": _squarefree(15) in inflation,
        "eightfold_meets_twelvefold": cyclotomic_overlap(8, 12),
        "sixteenfold_meets_twelvefold": cyclotomic_overlap(16, 12),
        "eightfold_meets_twentyfourfold": cyclotomic_overlap(8, 24),
        "angles_a_twelvefold_vacuum_can_fix": ["alpha = pi/2"],
        "angles_it_cannot": ["beta = pi/8", "gamma = 3pi/8"],
        "smallest_vacuum_containing_both": 24,
    }


# ---------------------------------------------------------------------------
# the special triangle, and the pi/4 accident
# ---------------------------------------------------------------------------

def yukawa_triangle_ratios():
    """The three side ratios of the ``(pi/2, pi/8, 3pi/8)`` triangle. Exact.

    Derived from the right triangle rather than quoted: with the right angle
    at ``alpha``, the two legs are ``|Vtd Vtb*|`` and ``|Vud Vub*|`` and the
    hypotenuse is ``|Vcd Vcb*|``, so ``|R_alpha| = cot(beta)``,
    ``|R_beta| = 1/cos(beta)`` and ``|R_gamma| = sin(beta)``.

    Returns exact surds, including the identity ``cot(pi/8) = 1 + sqrt 2``.
    """
    b = sp.pi / 8
    return {"R_alpha": sp.simplify(1 / sp.tan(b)),
            "R_beta": sp.simplify(1 / sp.cos(b)),
            "R_gamma": sp.simplify(sp.sin(b)),
            "R_gamma_as_cos": sp.simplify(sp.cos(3 * sp.pi / 8)),
            "cot_pi_8": sp.simplify(sp.nsimplify(1 / sp.tan(sp.pi / 8)))}


def z8_ratio():
    """The ratio a Z8 vacuum would give, evaluated exactly.

    The paper sketches a mechanism in which two flavon vevs of equal
    magnitude and phases differing by ``pi/4`` produce
    ``i (1 - e^{i pi/4}) / (1 + e^{i pi/4})``, which it identifies with
    ``i tan(pi/8)``. Exact arithmetic gives ``(1 - e^{i t})/(1 + e^{i t}) =
    -i tan(t/2)``, so the bracket is ``-i tan(pi/8)`` and the whole expression
    is ``tan(pi/8)`` -- real, hence a degenerate triangle rather than a right
    one.

    Returned as computed, with the target alongside, rather than adjusted to
    agree. The difference is a single factor of ``i``.
    """
    z = (1 - sp.exp(sp.I * sp.pi / 4)) / (1 + sp.exp(sp.I * sp.pi / 4))
    return {"bracket": sp.simplify(sp.expand_complex(z)),
            "bracket_is_minus_i_tan": sp.simplify(
                sp.expand_complex(z) + sp.I * sp.tan(sp.pi / 8)) == 0,
            "i_times_bracket": sp.simplify(sp.expand_complex(sp.I * z)),
            "target": sp.I * sp.tan(sp.pi / 8),
            "agrees": sp.simplify(sp.expand_complex(sp.I * z)
                                  - sp.I * sp.tan(sp.pi / 8)) == 0}


def pi_over_4_condition(data=None):
    """The coincidence behind the ``phi ~ pi/4`` peak, checked against data.

    The peak is not connected to the unitarity triangle at all. It requires
    ``(ys/yb)^2 |Vus/Vub|^2 ~ sqrt 2``, which is a statement about measured
    quantities and nothing else. Returns the measured combination, the target,
    and the discrepancy.
    """
    d = data or PDG
    val = (d["ys"] / d["yb"]) ** 2 * (d["Vus"] / d["Vub"]) ** 2
    target = float(sp.sqrt(2))
    return {"measured": val, "target": target,
            "relative_error": abs(val - target) / target}


# ---------------------------------------------------------------------------
# the theory
# ---------------------------------------------------------------------------

@register
class NineLinkTexture(FlavorTheory):
    """A nine-link Yukawa texture and what it implies for the CKM triangle."""

    key = "nine-link-texture"

    def __init__(self, texture=None, name=None):
        FlavorTheory.__init__(self, name=name)
        self.tex = texture

    def texture(self):
        """The link diagram, or the whole enumeration if none was given."""
        if self.tex is None:
            return census()
        return {"links": self.tex.links(), "split": self.tex.split(),
                "loop": sorted(self.tex.loop()),
                "strong_cp_safe": self.tex.strong_cp_safe()}

    def rephasing_invariant(self):
        if self.tex is None:
            raise ValueError("give a texture")
        return self.tex.rephasing_invariant()

    def predicted_angle(self, which=1, **kw):
        """The angle a worked example predicts, by exact diagonalisation."""
        return example_check(which, **kw)

    def deviation(self, which=1, **kw):
        """The calculable departure from the special value."""
        r = example_check(which, **kw)
        special = {1: np.pi / 2, 2: np.pi / 8, 3: 3 * np.pi / 8}[which]
        return {"from_exact": r["exact"] - special,
                "from_closed_form": r["closed_form"] - special,
                "eps": r["eps"]}

    def histogram(self, **kw):
        """Always raises: the phase histogram is the output of a fit."""
        raise NeedsFit(
            "the clustering of the rephasing invariant around multiples of "
            "pi/8 is the output of a chi-squared fit of every texture to the "
            "flavour data. This module computes the combinatorics that "
            "explains the clustering and does not run the fit; and a scan "
            "that did would report the minima it found in the box it "
            "searched, not the solutions.",
            missing=["a chi-squared minimisation of ten parameters against "
                     "the seventeen observables of Table S2",
                     "a scan over starting points and phase windows, since "
                     "each texture has several local minima",
                     "a confidence-level threshold, which decides which fits "
                     "count as viable and therefore all the reported counts"],
            available="the exact layer: enumerate_textures, census, "
                      "strong_cp_census, example_check")

    def exact_content(self):
        return [
            "the enumeration of nine-link textures from the conditions "
            "alone, and their orbits under relabelling the quark fields",
            "the unique closed loop of each diagram and the "
            "rephasing-invariant monomial it carries",
            "the perfect matchings, hence the determinant terms, hence "
            "whether the phase can be placed to leave arg det(Yu Yd) zero",
            "the unitarity-triangle angles of given Yukawa matrices, by "
            "exact diagonalisation rather than by expansion",
            "the side ratios of the (pi/2, pi/8, 3pi/8) triangle, and the "
            "arithmetic of the pi/4 coincidence",
        ]

    def declined(self):
        return [
            "the phase histogram and every count derived from it, which are "
            "outputs of a fit (see histogram)",
            "the predicted ellipses of the paper's figures, for the same "
            "reason",
            "any claim to have derived the texture rather than assumed it",
        ]
