r"""
pyCICY.chair44_nested -- Chair44 nearly descends to nested tori, and the Hat does not.

Part VII of the six-sphere write-up (AlpogeSixSphere.md) rolls a Hat tiling up on
the cusp torus R^2 / B0 Z^2 and finds that it does not descend: every one of the
twelve bond orbits is a tile boundary sometimes and not always, and no lattice
translate of the patch agrees with it everywhere (best non-identity agreement 0.796
at four steps). That is what one expects of an aperiodic tiling and of an inflation
that is not a lattice map.

Tsiokos's Chair44 (the 3D "einstein" tile, arXiv 2609.19214 and 2609.23783, repository
ioannist/six-birds-tiles, CC BY 4.0) is the opposite kind of object. Its inflation is
the *integer* 2, so level-M supertiles are lattice objects: they sit on the sublattice
2^M Z^3 of index 8^M, and the tiling is exhausted by the nested tori R^3 / 2^M Z^3.
This module computes how well it descends to each of them, exactly.

What is computed
----------------
1. **The substitution.**  24 rotations x 7 unit cells = 168 labels; sigma sends a label
   to the eight labels of its 2 x 2 x 2 block.  The eight child poses are re-checked to
   tile the doubled chair (:func:`verify_children`).  Column sums 8, primitive at power 3.
2. **Coincidence, with its rate** (:func:`coincidence_table`).  After reading M digits of
   a position inside a level-M block, the set of labels still possible is tracked by a
   subset automaton (209 reachable sets, 41 of them with more than one label).  Then
          non-coincident addresses   c(M) = 28*2^M - 48            (M >= 1)
          label excess  sum(|set|-1) E(M) = 140*2^M + 48           (M >= 1)
   These are *proved* by :func:`closed_forms_proved`: both sequences are read off a 41-state
   transfer matrix, so a difference from the closed form is a linear recurrence of order
   at most 43 and vanishes if it vanishes on 43 consecutive terms; 44 are checked.
3. **The descent inequality** (:func:`nested_agreement`).  For any two labels the fraction
   of addresses of the 2^M cube at which their M-fold images differ is at most c(M)/8^M.
   A patch that is a union of whole level-M blocks therefore agrees with its translate by
   2^M v on at least 1 - c(M)/8^M of the overlap.  The inequality is a theorem; the table
   measures the actual agreement on a patch of 7*8^N cells and shows how close to the bound
   it comes.
4. **Which translates are near-periods** (:func:`translate_scan`): every translate of a
   patch at once (FFT, integer counts), tabulated by the 2-adic valuation of the vector.
5. **What the undetermined addresses are** (:func:`undetermined_profile`,
   :func:`six_is_a_dihedral_coset`): at nearly all addresses where the label is not yet
   determined, exactly six rotations remain, and they form a coset of an order-6 dihedral
   subgroup (the stabiliser of a body diagonal); at a fixed small number all 24 remain.
6. **Two things that are not connections** (:func:`kuhn_vs_cube`): the 3D analogue of the
   cusp triangulation (Kuhn's triangulation of Z^3) has the right counts, but the cube's
   rotation group does not preserve it, and the group that does is not conjugate to the
   cube's in GL(3, Z).

What is NOT claimed
-------------------
* Pure-point diffraction and limit-periodicity of Chair44 follow from the coincidence only
  through the cited Lee-Moody / Schlottmann theorem (T3, recorded upstream); nothing here
  re-proves that, and whether every Chair44 tiling lies in the substitution hull is open
  upstream.  Chair44's aperiodicity is Tsiokos's claimed proof, quoted.
* The six-sphere does not contain Chair44.  Alpoge's cusp fibre is two-dimensional and
  hexagonal; this module builds no complex fourfold.  The dictionary d = |det B0| <-> index
  of the supertile lattice is an analogy of counting, stated as such.
* The Hat comparison is a contrast of kind.  The Hat's inflation is irrational and gives no
  nested lattices; its 0.796 was measured on Laves-lattice translates of a four-step patch,
  not on the footing used here.

Run ``python3 -m pyCICY.chair44_nested`` for the report.
"""

import itertools
from collections import Counter
from fractions import Fraction

import numpy as np

# ---------------------------------------------------------------------------
# 1. the substitution
# ---------------------------------------------------------------------------
#: the seven unit cells of the chair: the 2 x 2 x 2 cube minus its far corner
CELLS = [c for c in itertools.product((0, 1), repeat=3) if c != (1, 1, 1)]
#: child positions inside a doubled cell, in the order the digits are read
BITS = list(itertools.product((0, 1), repeat=3))
IDENTITY = ((0, 1, 2), (1, 1, 1))

#: the eight child poses (frame, translation) of the doubled chair, as recorded in
#: brentharts/spectre (symchair44.CHILDREN) from Tsiokos's candidate certificate;
#: a frame is (perm, signs), (G x)_i = s_i x_{p_i}.  verify_children() re-checks them.
CHILDREN = (
    (((0, 1, 2), (1, 1, 1)), (0, 0, 0)),
    (((0, 1, 2), (1, 1, 1)), (1, 1, 1)),
    (((0, 1, 2), (-1, -1, 1)), (4, 4, 0)),
    (((0, 2, 1), (1, -1, 1)), (0, 4, 0)),
    (((1, 0, 2), (1, 1, -1)), (0, 0, 4)),
    (((1, 2, 0), (-1, 1, -1)), (4, 0, 4)),
    (((2, 0, 1), (1, -1, -1)), (0, 4, 4)),
    (((2, 1, 0), (-1, 1, 1)), (4, 0, 0)),
)


def frame_apply(frame, x):
    p, s = frame
    return tuple(s[i] * x[p[i]] for i in range(3))


def frame_mul(f, g):
    """f o g as a frame."""
    (p, s), (q, r) = f, g
    return tuple(q[p[i]] for i in range(3)), tuple(s[i] * r[p[i]] for i in range(3))


def frame_matrix(frame):
    p, s = frame
    m = np.zeros((3, 3), dtype=int)
    for i in range(3):
        m[i, p[i]] = s[i]
    return m


def frame_det(frame):
    p, s = frame
    inversions = sum(1 for i in range(3) for j in range(i + 1, 3) if p[i] > p[j])
    return (-1) ** inversions * s[0] * s[1] * s[2]


def rotations():
    """The 24 proper rotations of the cube, as frames."""
    return [(p, s) for p in itertools.permutations(range(3))
            for s in itertools.product((1, -1), repeat=3)
            if frame_det((p, s)) == 1]


def cell_of(frame, t, c):
    """Lower corner of the unit cell that the native cell c occupies under pose (frame, t)."""
    x = frame_apply(frame, c)
    return tuple(x[i] + t[i] - (1 if frame[1][i] < 0 else 0) for i in range(3))


def refine_pose(pose):
    """The eight children of a pose, at twice the scale."""
    G, t = pose
    return [(frame_mul(G, H), tuple(2 * t[i] + frame_apply(G, u)[i] for i in range(3)))
            for H, u in CHILDREN]


def verify_children():
    """Each rotated chair, doubled, is tiled exactly by its eight children (56 = 8 x 7 cells)."""
    ok = True
    for f in rotations():
        owner = {}
        for G, t in refine_pose((f, (0, 0, 0))):
            for c in CELLS:
                cell = cell_of(G, t, c)
                if cell in owner:
                    return False
                owner[cell] = G
        doubled = {tuple(2 * q[i] + a[i] for i in range(3))
                   for c in CELLS for q in [cell_of(f, (0, 0, 0), c)] for a in BITS}
        ok = ok and set(owner) == doubled and len(owner) == 56
    return ok


def substitution():
    """(labels, sigma): labels = (frame, native cell index), sigma[label][bit] = child label."""
    frames = rotations()
    labels = [(f, k) for f in frames for k in range(7)]
    index = {l: i for i, l in enumerate(labels)}
    sigma = [[None] * 8 for _ in labels]
    for f in frames:
        owner = {}
        for G, t in refine_pose((f, (0, 0, 0))):
            for k, c in enumerate(CELLS):
                owner[cell_of(G, t, c)] = (G, k)
        for kk, c in enumerate(CELLS):
            p = cell_of(f, (0, 0, 0), c)
            for b, a in enumerate(BITS):
                sigma[index[(f, kk)]][b] = index[owner[tuple(2 * p[i] + a[i] for i in range(3))]]
    return labels, sigma


_SUB = None


def _sub():
    global _SUB
    if _SUB is None:
        labels, sigma = substitution()
        _SUB = (labels, sigma, np.array(sigma, dtype=np.int32))
    return _SUB


def primitive_exponent(sigma, limit=8):
    n = len(sigma)
    step = [set(r) for r in sigma]
    cur = [set(s) for s in step]
    for power in range(1, limit + 1):
        if all(len(c) == n for c in cur):
            return power
        cur = [set().union(*(step[j] for j in c)) for c in cur]
    return None


def column_sums(sigma):
    cols = Counter(j for row in sigma for j in row)
    return sorted(set(cols[j] for j in range(len(sigma))))


# ---------------------------------------------------------------------------
# 2. coincidence and its rate
# ---------------------------------------------------------------------------
_AUTOMATON = None


def subset_automaton():
    """Reachable label sets under A -> {sigma[j][b] : j in A}, from the full set."""
    global _AUTOMATON
    if _AUTOMATON is None:
        sigma = _sub()[1]
        full = frozenset(range(len(sigma)))
        state, order, trans = {full: 0}, [full], []
        i = 0
        while i < len(order):
            row = []
            for b in range(8):
                B = frozenset(sigma[j][b] for j in order[i])
                if B not in state:
                    state[B] = len(order)
                    order.append(B)
                row.append(state[B])
            trans.append(row)
            i += 1
        _AUTOMATON = (order, trans)
    return _AUTOMATON


def coincidence_table(Mmax=8):
    """Per depth M: addresses 8^M, coincident, non-coincident c(M), excess E(M), set-size histogram."""
    order, trans = subset_automaton()
    size = [len(A) for A in order]
    cnt, rows = {0: 1}, []
    for M in range(Mmax + 1):
        hist = Counter()
        for s, c in cnt.items():
            hist[size[s]] += c
        rows.append(dict(M=M, addresses=8 ** M,
                         coincident=hist.get(1, 0),
                         non_coincident=sum(v for k, v in hist.items() if k > 1),
                         excess=sum(v * (k - 1) for k, v in hist.items()),
                         histogram=dict(sorted(hist.items()))))
        nxt = {}
        for s, c in cnt.items():
            for b in range(8):
                nxt[trans[s][b]] = nxt.get(trans[s][b], 0) + c
        cnt = nxt
    return rows


def c_of(M):
    """c(M): digit words of length M that have not coincided (exact, from the automaton)."""
    return coincidence_table(M)[M]["non_coincident"]


def closed_forms_proved(terms=44):
    """c(M) = 28*2^M - 48 and E(M) = 140*2^M + 48 for all M >= 1, by a finite certificate.

    Non-singleton label sets can only lead to non-singleton or singleton sets, never the
    reverse, so c and E are linear functionals of T^M e_0 with T the transfer matrix on the
    non-singleton states.  A difference from a closed form with characteristic roots 2 and 1
    is therefore a linear recurrence of order <= n + 2 (n = number of non-singleton states);
    it is identically zero from M = 1 as soon as it vanishes on n + 2 consecutive terms.
    """
    order, _ = subset_automaton()
    n = sum(1 for A in order if len(A) > 1)
    rows = coincidence_table(terms)
    need = n + 2
    c_ok = all(r["non_coincident"] == 28 * 2 ** r["M"] - 48 for r in rows[1:])
    e_ok = all(r["excess"] == 140 * 2 ** r["M"] + 48 for r in rows[1:])
    return dict(non_singleton_states=n, terms_checked=terms, terms_needed=need,
                c_closed_form=c_ok, E_closed_form=e_ok,
                proved=bool(c_ok and e_ok and terms >= need))


def transfer_charpoly():
    """Characteristic polynomial of the transfer matrix on the non-singleton label sets, factored.

    Returns (factors, string): factors = {root: multiplicity} when it splits over Z.  The
    spectral radius is the largest root; here exactly 2, so the undetermined set (c(M) ~ 2^M
    inside 8^M addresses) has box dimension log2(2) = 1 in the 2-adic internal space.
    """
    import sympy as sp
    order, trans = subset_automaton()
    ns = [k for k in range(len(order)) if len(order[k]) > 1]
    idx = {k: i for i, k in enumerate(ns)}
    T = [[0] * len(ns) for _ in ns]
    for k in ns:
        for b in range(8):
            d = trans[k][b]
            if d in idx:
                T[idx[k]][idx[d]] += 1
    x = sp.symbols("x")
    poly = sp.Matrix(T).charpoly(x).as_expr()
    _, fl = sp.factor_list(poly)
    factors = {}
    for f, m in fl:
        root = sp.solve(f, x)
        if len(root) == 1 and sp.degree(f, x) == 1:
            factors[int(root[0])] = int(m)
        else:
            factors[str(f)] = int(m)
    text = "".join(("x" if r == 0 else "(x%+d)" % -r if isinstance(r, int) else "(%s)" % r)
                   + ("^{%d}" % m if m > 1 else "") for r, m in sorted(factors.items(), key=lambda t: str(t[0])))
    return factors, T, text


def undetermined_profile(M=12):
    """Label-set sizes at the non-coincident addresses of the 2^M cube."""
    h = coincidence_table(M)[M]["histogram"]
    return {k: v for k, v in h.items() if k > 1}


def six_is_a_dihedral_coset():
    """Every 6-label set is six rotations forming a coset of an order-6 subgroup of S4 = O,
    and that subgroup is dihedral (three half-turns, two third-turns) about a body diagonal."""
    labels = _sub()[0]
    order, _ = subset_automaton()
    rots = rotations()
    mats = {f: frame_matrix(f) for f in rots}
    by_key = {tuple(m.flatten()): f for f, m in mats.items()}
    inv = lambda f: by_key[tuple(mats[f].T.flatten())]
    prod = lambda a, b: by_key[tuple((mats[a] @ mats[b]).flatten())]

    def order_of(f):
        k, g = 1, f
        while tuple(mats[g].flatten()) != tuple(np.eye(3, dtype=int).flatten()):
            g, k = prod(f, g), k + 1
        return k

    sets6 = [A for A in order if len(A) == 6]
    stats = Counter()
    for A in sets6:
        R = {labels[i][0] for i in A}
        quotient = {prod(a, inv(b)) for a in R for b in R}
        orders = Counter(order_of(g) for g in quotient)
        diag = [d for d in itertools.product((1, -1), repeat=3)
                if all(tuple(mats[g] @ np.array(d)) in (d, tuple(-x for x in d)) for g in quotient)]
        stats[(len(R), len(quotient), tuple(sorted(orders.items())), len(diag) > 0)] += 1
    return dict(n_sets=len(sets6), classes=dict(stats),
                all_dihedral_cosets=set(stats) == {(6, 6, ((1, 1), (2, 3), (3, 2)), True)})


# ---------------------------------------------------------------------------
# 3. nested tori: agreement of the patch with its translates
# ---------------------------------------------------------------------------
def patch(N):
    """Cells (positions, labels) of the level-N patch: the chair scaled by 2^N, 7*8^N cells."""
    sig = _sub()[2]
    labels = _sub()[0]
    root = [labels.index((IDENTITY, k)) for k in range(7)]
    P = np.array(CELLS, dtype=np.int64)
    L = np.array(root, dtype=np.int32)
    for _ in range(N):
        P = np.concatenate([2 * P + np.array(a) for a in BITS])
        L = np.concatenate([sig[L, b] for b in range(8)])
    return P, L


def grid(N):
    """Dense label grid of the level-N patch (-1 = empty), side 2^(N+1)."""
    P, L = patch(N)
    n = 2 ** (N + 1)
    G = np.full((n, n, n), -1, dtype=np.int16)
    G[P[:, 0], P[:, 1], P[:, 2]] = L
    return G


def agreement(G, v):
    """(cells whose label equals that of the cell at +v, cells with both present)."""
    sa, sb = [], []
    for c, n in zip(v, G.shape):
        if abs(c) >= n:
            return 0, 0
        sa.append(slice(max(0, -c), n - max(0, c)))
        sb.append(slice(max(0, c), n - max(0, -c)))
    A, B = G[tuple(sa)], G[tuple(sb)]
    both = (A >= 0) & (B >= 0)
    return int((both & (A == B)).sum()), int(both.sum())


def disagreement_map(G, v):
    """Boolean grid: cells present at x and at x+v whose labels differ."""
    out = np.zeros(G.shape, dtype=bool)
    sa, sb = [], []
    for c, n in zip(v, G.shape):
        sa.append(slice(max(0, -c), n - max(0, c)))
        sb.append(slice(max(0, c), n - max(0, -c)))
    A, B = G[tuple(sa)], G[tuple(sb)]
    out[tuple(sa)] = (A >= 0) & (B >= 0) & (A != B)
    return out


def nested_agreement(N=6, axes=((1, 0, 0), (0, 1, 0), (0, 0, 1))):
    """For v = 2^M e and M = 1..N-1: measured agreement, and the proved bound 1 - c(M)/8^M."""
    G = grid(N)
    table = coincidence_table(max(N, 3))
    rows = []
    for M in range(1, N):
        c = table[M]["non_coincident"]
        for e in axes:
            v = tuple(2 ** M * x for x in e)
            a, o = agreement(G, v)
            bound = Fraction(8 ** M - c, 8 ** M)
            rows.append(dict(M=M, v=v, agree=a, overlap=o,
                             fraction=a / o, bound=bound,
                             holds=a * 8 ** M >= o * (8 ** M - c)))
    return dict(N=N, cells=int((G >= 0).sum()), rows=rows)


def valuation(n):
    if n == 0:
        return 99
    k = 0
    while n % 2 == 0:
        n //= 2
        k += 1
    return k


def translate_scan(N=5, overlap_fraction=8, top=25):
    """Agreement of the level-N patch with *every* translate (FFT, integer counts)."""
    G = grid(N)
    side = G.shape[0]
    pad = 2 * side
    ax = (0, 1, 2)
    acc = np.zeros((pad, pad, pad // 2 + 1))
    for lab in range(len(_sub()[0])):
        ind = (G == lab).astype(np.float64)
        if not ind.any():
            continue
        F = np.fft.rfftn(ind, s=(pad,) * 3, axes=ax)
        acc += F.real ** 2 + F.imag ** 2
    corr = np.fft.irfftn(acc, s=(pad,) * 3, axes=ax)
    Fm = np.fft.rfftn((G >= 0).astype(np.float64), s=(pad,) * 3, axes=ax)
    ov = np.fft.irfftn(Fm.real ** 2 + Fm.imag ** 2, s=(pad,) * 3, axes=ax)
    err = max(np.abs(corr - np.rint(corr)).max(), np.abs(ov - np.rint(ov)).max())
    corr, ov = np.rint(corr).astype(np.int64), np.rint(ov).astype(np.int64)
    n = int((G >= 0).sum())
    r = np.arange(-side + 1, side)
    X, Y, Z = np.meshgrid(r, r, r, indexing="ij")
    C, O = corr[X % pad, Y % pad, Z % pad], ov[X % pad, Y % pad, Z % pad]
    ok = (O >= n // overlap_fraction) & ~((X == 0) & (Y == 0) & (Z == 0))
    frac = np.where(ok, C / np.maximum(O, 1), 0.0)
    val = np.minimum(np.minimum(*[np.vectorize(valuation)(A) for A in (X, Y)]),
                     np.vectorize(valuation)(Z))
    best = []
    for f in np.argsort(-frac, axis=None)[:top]:
        i = np.unravel_index(f, frac.shape)
        best.append(dict(v=(int(X[i]), int(Y[i]), int(Z[i])), fraction=float(frac[i]),
                         overlap=int(O[i]), valuation=int(val[i])))
    by_threshold = {}
    for thr in (0.5, 0.9, 0.99):
        sel = ok & (frac >= thr)
        by_threshold[thr] = dict(count=int(sel.sum()),
                                 by_valuation=dict(sorted(Counter(val[sel].tolist()).items())))
    # slice v_z = 0 for the figures
    zc = side - 1
    return dict(N=N, cells=n, fft_rounding_error=float(err), best=best, by_threshold=by_threshold,
                slice_z0=frac[:, :, zc].copy(), radius=side - 1)


def translate_counts(N=6, fractions=(2, 4, 8, 16), thresholds=(0.5, 0.9, 0.99)):
    """The same scan as :func:`translate_scan`, summarised for a size-matched comparison.

    Returns the number of cells, the number of labels present, the chance level (the
    probability that two cells drawn independently carry the same label), the number
    of translates scanned (overlap at least an eighth of the patch, zero excluded), how
    many of them agree on at least each threshold, and, for each overlap fraction
    ``1/f``, the best agreement with its translate and the overlap there. This is the
    Chair44 side of the Hat comparison in ``paper/make_sixsphere_hat_fft.py``: the two
    scans use the same rule, so the counts and the best values can be set side by side
    (the Hat's translates form a plane lattice, Chair44's a three-dimensional one, so
    the counts of translates scanned are not comparable but the best values are).
    """
    G = grid(N)
    side = G.shape[0]
    pad = 2 * side
    ax = (0, 1, 2)
    acc = np.zeros((pad, pad, pad // 2 + 1))
    for lab in range(len(_sub()[0])):
        ind = (G == lab).astype(np.float64)
        if not ind.any():
            continue
        F = np.fft.rfftn(ind, s=(pad,) * 3, axes=ax)
        acc += F.real ** 2 + F.imag ** 2
    corr = np.fft.irfftn(acc, s=(pad,) * 3, axes=ax)
    Fm = np.fft.rfftn((G >= 0).astype(np.float64), s=(pad,) * 3, axes=ax)
    ov = np.fft.irfftn(Fm.real ** 2 + Fm.imag ** 2, s=(pad,) * 3, axes=ax)
    err = max(np.abs(corr - np.rint(corr)).max(), np.abs(ov - np.rint(ov)).max())
    corr, ov = np.rint(corr).astype(np.int64), np.rint(ov).astype(np.int64)
    n = int((G >= 0).sum())
    nz = np.ones(corr.shape, bool)
    nz[0, 0, 0] = False
    frac = corr / np.maximum(ov, 1)
    labels, freq = np.unique(G[G >= 0], return_counts=True)
    chance = float(((freq / n) ** 2).sum())
    ok = nz & (ov >= n // 8)
    counts = {thr: int((ok & (frac >= thr)).sum()) for thr in thresholds}
    best = {}
    for f in fractions:
        m = nz & (ov >= n // f)
        a = np.unravel_index(np.argmax(np.where(m, frac, -1.0)), frac.shape)
        v = tuple(int(((x + pad // 2) % pad) - pad // 2) for x in a)
        best[f] = dict(fraction=float(frac[a]), v=v, overlap=int(ov[a]), translates=int(m.sum()))
    return dict(N=N, cells=n, labels=int(len(labels)), chance=chance, scanned=int(ok.sum()),
                counts=counts, best=best, fft_rounding_error=float(err))


# ---------------------------------------------------------------------------
# 4. the 3D cusp analogue, and why it is not a connection
# ---------------------------------------------------------------------------
def _det3(g):
    return int(round(np.linalg.det(np.array(g))))


def kuhn_vs_cube():
    """Kuhn's triangulation of Z^3 against the cube group.

    The 2D version of this triangulation is the A2 triangulation behind the cusp fan.
    Counts per fundamental domain, the number of simplices round a vertex, and the
    linear symmetries: the cube group O_h (48), the automorphism group of the complex (48),
    their intersection, and a mod-2 invariant separating them up to GL(3, Z) conjugacy.
    """
    B = list(itertools.product((0, 1), repeat=3))
    leq = lambda a, b: all(x <= y for x, y in zip(a, b))
    nz = [v for v in B if any(v)]

    def chains(k):
        out = 0
        for ds in itertools.combinations(nz, k):
            ds = sorted(ds, key=sum)
            if all(leq(ds[i], ds[i + 1]) and ds[i] != ds[i + 1] for i in range(k - 1)):
                out += 1
        return out

    f = [1] + [chains(k) for k in (1, 2, 3)]
    D = {tuple(s * c for c in d) for d in nz for s in (1, -1)}
    apply = lambda g, v: tuple(sum(g[i][j] * v[j] for j in range(3)) for i in range(3))
    Oh = []
    for p in itertools.permutations(range(3)):
        for s in itertools.product((1, -1), repeat=3):
            g = [[0] * 3 for _ in range(3)]
            for i in range(3):
                g[i][p[i]] = s[i]
            Oh.append(tuple(map(tuple, g)))
    preserves = lambda g: {apply(g, d) for d in D} == D
    Dl = sorted(D)
    K = []
    for c1, c2, c3 in itertools.product(Dl, repeat=3):
        g = tuple(tuple(c[i] for c in (c1, c2, c3)) for i in range(3))
        if abs(_det3(g)) == 1 and preserves(g):
            K.append(g)

    def mod2(G):
        vecs = [v for v in itertools.product((0, 1), repeat=3) if any(v)]
        img = lambda g, v: tuple(x % 2 for x in apply(g, v))
        fixed = [v for v in vecs if all(img(g, v) == v for g in G)]
        orbits, seen = [], set()
        for v in vecs:
            if v not in seen:
                o = {img(g, v) for g in G}
                seen |= o
                orbits.append(len(o))
        return dict(fixed_nonzero=len(fixed), orbit_sizes=sorted(orbits))

    rot = [g for g in Oh if _det3(g) == 1]
    return dict(simplices=f, euler=sum((-1) ** i * x for i, x in enumerate(f)),
                around_vertex=24, plane_simplices=(1, 3, 2),
                cube_group=len(Oh), cube_preserving=sum(preserves(g) for g in Oh),
                rotations_preserving=sum(preserves(g) for g in rot),
                automorphism_group=len(K), intersection=len(set(K) & set(Oh)),
                mod2_cube=mod2(Oh), mod2_kuhn=mod2(K))


def orbifold_cover():
    """Riemann-Hurwitz for the S4 cover of the (3, 4, inf) orbifold with the cusp sent to a
    half-turn: genus 0, 12 cusps.  Group theory only; nothing ties it to the six-sphere."""
    chi = 2 - (1 - Fraction(1, 3)) - (1 - Fraction(1, 4)) - 1
    n, cusps = 24, 12
    cover = n * chi
    return dict(orbifold_chi=chi, cover_chi=cover, cusps=cusps, genus=(2 - (cover + cusps)) / 2)


# ---------------------------------------------------------------------------
# status and ledger
# ---------------------------------------------------------------------------
def status(N=6, scan_N=5):
    labels, sigma, _ = _sub()
    co = coincidence_table(8)
    cf = closed_forms_proved()
    na = nested_agreement(N)
    ts = translate_scan(scan_N)
    prof = undetermined_profile(12)
    dih = six_is_a_dihedral_coset()
    kv = kuhn_vs_cube()
    oc = orbifold_cover()
    first = next(r["M"] for r in co if r["coincident"] > 0)
    tcp = transfer_charpoly()[0]
    th = ts["by_threshold"]
    checks = {
        "the eight child poses tile every rotated doubled chair (56 cells)": verify_children(),
        "168 labels; every column of sigma sums to 8; primitive at power 3":
            len(labels) == 168 and column_sums(sigma) == [8] and primitive_exponent(sigma) == 3,
        "least coincidence depth 3, with 336 of 512 addresses coincident":
            first == 3 and co[3]["coincident"] == 336,
        "reachable label sets: 209, of which 41 non-singleton":
            len(subset_automaton()[0]) == 209
            and sum(1 for A in subset_automaton()[0] if len(A) > 1) == 41,
        "c(M) = 28*2^M - 48 and E(M) = 140*2^M + 48 for all M >= 1 (finite certificate)":
            cf["proved"],
        "transfer matrix: characteristic polynomial x^5 (x-2)^4 (x-1)^32, spectral radius 2":
            tcp == {0: 5, 2: 4, 1: 32},
        "descent inequality: agreement >= 1 - c(M)/8^M on every row": all(r["holds"] for r in na["rows"]),
        "agreement is 0 at v = 2e and below 1/2 at v = 4e":
            all(r["fraction"] == 0 for r in na["rows"] if r["M"] == 1)
            and all(r["fraction"] < 0.5 for r in na["rows"] if r["M"] == 2),
        "every translate with agreement >= 0.9 has 2-adic valuation >= 3":
            min(th[0.9]["by_valuation"]) >= 3,
        "the FFT counts are integers to rounding error < 1e-3": ts["fft_rounding_error"] < 1e-3,
        "undetermined addresses carry 6 or 24 rotations, and 6 dominates":
            set(prof) == {6, 24} and prof[6] > 1000 * prof[24],
        "each 6-set is a coset of a dihedral subgroup about a body diagonal":
            dih["all_dihedral_cosets"],
        "Kuhn complex of Z^3: 1, 7, 12, 6 simplices, chi = 0": kv["simplices"] == [1, 7, 12, 6]
            and kv["euler"] == 0,
        "the cube group does not preserve it: 12 of 48, 6 of 24 rotations":
            kv["cube_preserving"] == 12 and kv["rotations_preserving"] == 6,
        "its automorphism group has order 48, meeting the cube group in 12":
            kv["automorphism_group"] == 48 and kv["intersection"] == 12,
        "the two order-48 groups differ mod 2, so are not GL(3, Z)-conjugate":
            kv["mod2_cube"] != kv["mod2_kuhn"] and kv["mod2_cube"]["fixed_nonzero"] == 1
            and kv["mod2_kuhn"]["fixed_nonzero"] == 0,
        "S4 cover of the (3, 4, inf) orbifold: genus 0, 12 cusps":
            oc["genus"] == 0 and oc["cusps"] == 12,
    }
    return dict(checks=checks, all_pass=all(checks.values()), nested=na, scan=ts,
                table=co, closed_forms=cf, profile=prof, kuhn=kv)


def ledger():
    """Every quantity and its status, in the six-sphere ledger's five words."""
    return [
        ("child poses tile the doubled chair; 168 labels; column sum 8; primitive at power 3",
         "exact; poses quoted from the upstream certificate, tiling re-verified"),
        ("least coincidence depth 3, 336 of 512 addresses; 209 label sets, 41 non-singleton",
         "exact; agrees with upstream THEOREM.md"),
        ("c(M) = 28*2^M - 48, M >= 1", "exact; proved by a finite recurrence certificate"),
        ("E(M) = sum(|set| - 1) = 140*2^M + 48, M >= 1", "exact; same certificate"),
        ("descent inequality: agreement of a patch with its 2^M v translate >= 1 - c(M)/8^M",
         "proved (counting); measured on a 1.8-million-cell patch"),
        ("agreement 0.684, 0.910, 0.976 at M = 3, 4, 5 along an axis", "exact integer counts"),
        ("translates with agreement >= 0.9 all have valuation >= 3 (level-5 patch)",
         "exact, one patch; not proved for all patches"),
        ("pure-point / limit-periodic structure of Chair44",
         "quoted: Lee-Moody / Schlottmann import, upstream T3"),
        ("every Chair44 tiling lies in the substitution hull", "declined: open upstream"),
        ("Chair44 aperiodic; einstein property", "quoted: Tsiokos's claimed proof"),
        ("transfer matrix on the 41 non-singleton sets: x^5 (x-2)^4 (x-1)^32; spectral radius 2",
         "exact"),
        ("undetermined addresses carry 6 or 24 rotations; 6-sets are dihedral cosets",
         "exact (finite automaton), M = 12 for the counts"),
        ("the six candidates and the six mystic classes", "coincidence, no mechanism found"),
        ("Kuhn complex of Z^3: 1, 7, 12, 6; 24 simplices round a vertex", "exact"),
        ("the cube group does not preserve it; the groups that do are not GL(3, Z)-conjugate",
         "exact (mod-2 invariant); 24 = 24 is a coincidence"),
        ("S4 cover of the (3, 4, inf) orbifold: genus 0, 12 cusps",
         "exact group theory; unrelated to the six-sphere's monodromy"),
        ("Hat: best non-identity agreement 0.796 (four steps)",
         "quoted from Part VII; different footing, a contrast of kind"),
        ("a complex fourfold with Chair44 cusp fibre", "declined: not constructed, not claimed"),
    ]


def report():
    st = status()
    print("Chair44 and the nested tori R^3 / 2^M Z^3")
    for k, v in st["checks"].items():
        print("  [%s] %s" % ("ok" if v else "FAIL", k))
    print("\n  M  axis agreement  bound")
    for r in st["nested"]["rows"]:
        if r["v"][1:] == (0, 0):
            print("  %d  %.5f        %.5f" % (r["M"], r["fraction"], float(r["bound"])))
    print("\n  all pass:", st["all_pass"])
    return st["all_pass"]


if __name__ == "__main__":
    raise SystemExit(0 if report() else 1)
