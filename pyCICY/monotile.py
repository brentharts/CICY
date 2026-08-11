r"""
pyCICY.monotile -- the Hat--Spectre monotile family, exactly.

In 2023 Smith, Myers, Kaplan and Goodman-Strauss found the first aperiodic
monotile: a single 13-sided shape, the Hat, that tiles the plane but only
aperiodically. The Hat is one member of a continuous family Tile$(a,b)$,
parametrised by two edge lengths, and recent work maps quantum models onto
the vertices of these tilings and uses the parameter
$\ell = a/(a+b)$ as a *control knob for the quantum geometric tensor*: moving
along the family drives real-space topological transitions with no momentum
space anywhere in sight.

That last clause is what makes this a natural module for a package built on
exact arithmetic. An aperiodic tiling has no Brillouin zone, so its
topological invariants must be computed in real space --- and the real-space
invariant of choice, the spectral localizer index, is *the signature of a
matrix*. A signature is a count of pivot signs, decidable exactly over any
ordered field. The whole chain

    tile geometry -> vertex coordinates -> Hamiltonian -> localizer -> index

stays inside $\mathbb{Q}(\sqrt3)$: the tiles live on the Laves [3.4.6.4] kite
lattice whose bond directions are multiples of $30^\circ$, so every
coordinate, every hopping amplitude and every localizer entry is
$p + q\sqrt3$ with $p, q$ rational, and the index at the end is an integer
computed with no floating point and no tolerance.

What is derived rather than tabulated
-------------------------------------
The substitution combinatorics of the Hat tiling lives in a second quadratic
field, $\mathbb{Q}(\sqrt5)$. The metatile substitution matrix is quoted from
Smith et al.; everything after it is derived:

  * the inflation factor is the Perron eigenvalue, and it comes out
    $\varphi^4 = (7 + 3\sqrt5)/2$ with $\varphi$ the golden ratio ---
    an exact eigenvalue of an integer matrix, verified by exact division of
    the characteristic polynomial;
  * the metatile frequencies are the Perron eigenvector, computed over
    $\mathbb{Q}(\sqrt5)$;
  * the ratio of unreflected to reflected hats is then
    $\varphi^4 : 1$ --- the tiling is *almost* chiral, with one anti-hat per
    $\varphi^4 \approx 6.854$ hats, and the number is an exact element of
    $\mathbb{Q}(\sqrt5)$, not a decimal;
  * aperiodicity itself: a tiling with a period has rational tile
    frequencies, and the derived frequencies are irrational, so there is no
    period. The famous headline about the Hat is, at this level, a statement
    about the irrationality of an eigenvector.

The named tiles
---------------
Tile$(a,b)$ depends only on the ratio, and $\ell = a/(a+b)$ names the family
members: the Chevron at $\ell = 0$, the Hat at $\ell = 1/(1+\sqrt3)$, the
Spectre at $\ell = 1/2$, the Turtle at $\ell = \sqrt3/(1+\sqrt3)$ and the
Comet at $\ell = 1$. Exchanging $a \leftrightarrow b$ reflects the tile, so
$\ell \mapsto 1-\ell$ is a mirror: the Hat and the Turtle are each other's
mirror partners in the parameter --- their $\ell$ values are algebraic
conjugates under $\sqrt3 \mapsto -\sqrt3$ up to this reflection --- and the
Spectre is the fixed point, which is the parameter-space shadow of its
special role as the strictly chiral member. This slots into
:mod:`pyCICY.chirality`, where every other domain in the package already has
its mirror map.

What is exact, what is quoted, what is absent
---------------------------------------------
Exact: the geometry, the substitution combinatorics, the frequencies, the
aperiodicity argument, and the spectral localizer index of any finite patch.
Quoted: the substitution matrix itself and the hats-per-metatile counts,
which are combinatorial facts about the published construction. Absent: the
generation of large aperiodic vertex patches by substitution (the finite
patches here are cut from the Laves lattice that underlies every Tile$(a,b)$,
which is the correct substrate but is periodic --- the module says so rather
than presenting it as more), spectral functions, and anything requiring a
thermodynamic limit.

References
----------
Smith, Myers, Kaplan and Goodman-Strauss, An aperiodic monotile,
    arXiv:2303.10798.
Smith, Myers, Kaplan and Goodman-Strauss, A chiral aperiodic monotile,
    arXiv:2305.17743.
Schirmann, Franca, Flicker and Grushin, Physical properties of an aperiodic
    monotile, Phys. Rev. Lett. 132 (2024) 086402.
Roche Carrasco, Schirmann, Mordret and Grushin, Family of aperiodic tilings
    with tunable quantum geometric tensor, arXiv:2505.13304.
Loring and Schulz-Baldes, Finite volume calculation of K-theory invariants,
    New York J. Math. 23 (2017) 1111.
"""

from fractions import Fraction as F

__all__ = ["Quad", "SQRT3", "SQRT5", "named_tiles", "tile_ell", "mirror_ell",
           "SUBSTITUTION", "HATS_PER_METATILE", "inflation_factor",
           "metatile_frequencies", "hat_chirality", "is_aperiodic",
           "laves_patch", "qwz_hamiltonian", "localizer_signature",
           "localizer_index", "phase_scan"]


# ---------------------------------------------------------------------------
# exact quadratic fields
# ---------------------------------------------------------------------------


class Quad(object):
    r"""An element $p + q\sqrt d$ of $\mathbb{Q}(\sqrt d)$, with exact sign.

    The whole module rests on two facts about these fields: they are closed
    under the arithmetic a Hamiltonian needs, and they are *ordered*, with a
    decidable order. The sign of $p + q\sqrt d$ is settled by comparing
    $p^2$ with $d q^2$, which is a comparison of rationals; that is what
    makes a matrix signature over the field computable with no floating
    point, and the signature is the topological invariant.
    """

    __slots__ = ("p", "q", "d")

    def __init__(self, p, q=0, d=3):
        self.p = F(p)
        self.q = F(q)
        self.d = int(d)

    # -- arithmetic --------------------------------------------------------

    def _coerce(self, other):
        if isinstance(other, Quad):
            if other.d != self.d:
                raise ValueError("cannot mix Q(sqrt %d) and Q(sqrt %d)"
                                 % (self.d, other.d))
            return other
        return Quad(other, 0, self.d)

    def __add__(self, o):
        o = self._coerce(o)
        return Quad(self.p + o.p, self.q + o.q, self.d)

    __radd__ = __add__

    def __neg__(self):
        return Quad(-self.p, -self.q, self.d)

    def __sub__(self, o):
        return self + (-self._coerce(o))

    def __rsub__(self, o):
        return self._coerce(o) - self

    def __mul__(self, o):
        o = self._coerce(o)
        return Quad(self.p * o.p + self.d * self.q * o.q,
                    self.p * o.q + self.q * o.p, self.d)

    __rmul__ = __mul__

    def inverse(self):
        n = self.p * self.p - self.d * self.q * self.q
        if n == 0:
            raise ZeroDivisionError("zero has no inverse in Q(sqrt %d)"
                                    % self.d)
        return Quad(self.p / n, -self.q / n, self.d)

    def __truediv__(self, o):
        return self * self._coerce(o).inverse()

    def __rtruediv__(self, o):
        return self._coerce(o) * self.inverse()

    # -- order -------------------------------------------------------------

    def sign(self):
        """The exact sign, by comparing p^2 with d q^2."""
        if self.q == 0:
            return (self.p > 0) - (self.p < 0)
        if self.p == 0:
            return (self.q > 0) - (self.q < 0)
        if self.p > 0 and self.q > 0:
            return 1
        if self.p < 0 and self.q < 0:
            return -1
        # opposite signs: the larger of p^2 and d q^2 decides
        big_p = self.p * self.p > self.d * self.q * self.q
        if self.p > 0:
            return 1 if big_p else -1
        return -1 if big_p else 1

    def __eq__(self, o):
        try:
            o = self._coerce(o)
        except (ValueError, TypeError):
            return NotImplemented
        return self.p == o.p and self.q == o.q

    def __ne__(self, o):
        r = self.__eq__(o)
        return NotImplemented if r is NotImplemented else not r

    def __lt__(self, o):
        return (self - self._coerce(o)).sign() < 0

    def __le__(self, o):
        return (self - self._coerce(o)).sign() <= 0

    def __gt__(self, o):
        return (self - self._coerce(o)).sign() > 0

    def __ge__(self, o):
        return (self - self._coerce(o)).sign() >= 0

    def __hash__(self):
        return hash((self.p, self.q, self.d))

    def conjugate(self):
        r"""The Galois conjugate, $\sqrt d \mapsto -\sqrt d$."""
        return Quad(self.p, -self.q, self.d)

    def is_rational(self):
        return self.q == 0

    def __float__(self):
        return float(self.p) + float(self.q) * self.d ** 0.5

    def __repr__(self):
        if self.q == 0:
            return str(self.p)
        return "(%s + %s sqrt%d)" % (self.p, self.q, self.d)


SQRT3 = Quad(0, 1, 3)
SQRT5 = Quad(0, 1, 5)
PHI = (Quad(1, 0, 5) + SQRT5) / 2                       # the golden ratio
PHI4 = PHI * PHI * PHI * PHI                            # (7 + 3 sqrt5)/2


# ---------------------------------------------------------------------------
# the family, by parameter
# ---------------------------------------------------------------------------


def tile_ell(a, b):
    r"""$\ell = a/(a+b)$, the parameter naming a member of the family."""
    a, b = _q3(a), _q3(b)
    return a / (a + b)


def mirror_ell(ell):
    r"""The mirror map $\ell \mapsto 1-\ell$, i.e. Tile$(a,b) \mapsto$
    Tile$(b,a)$, which reflects the tile."""
    return Quad(1, 0, 3) - _q3(ell)


def _q3(x):
    return x if isinstance(x, Quad) else Quad(x, 0, 3)


def named_tiles():
    r"""The named members, with exact $\ell \in \mathbb{Q}(\sqrt3)$.

    $1/(1+\sqrt3)$ rationalises to $(\sqrt3-1)/2$, and
    $\sqrt3/(1+\sqrt3)$ to $(3-\sqrt3)/2$; the two sum to one, which is the
    statement that the Hat and the Turtle are mirror partners. The Spectre
    sits at the fixed point of the mirror, which is the parameter-space
    shadow of its special role: it is the member that tiles with a single
    handedness.
    """
    one = Quad(1, 0, 3)
    hat = one / (one + SQRT3)
    return {
        "Chevron": Quad(0, 0, 3),
        "Hat": hat,
        "Spectre": Quad(F(1, 2), 0, 3),
        "Turtle": SQRT3 / (one + SQRT3),
        "Comet": one,
    }


# ---------------------------------------------------------------------------
# the substitution system
# ---------------------------------------------------------------------------

#: The metatile substitution of Smith, Myers, Kaplan and Goodman-Strauss:
#: the Hat tiling is generated by four supertiles H, T, P, F, and one
#: inflation step replaces each by the counts below (columns are the parent,
#: rows the children). This matrix is quoted; everything computed from it is
#: derived, and the two published numbers it must reproduce -- the inflation
#: factor phi^4 and the hat:anti-hat ratio phi^4 : 1 -- are what the tests
#: check.
SUBSTITUTION = ((3, 1, 2, 2),
                (1, 0, 0, 0),
                (3, 0, 1, 1),
                (3, 0, 2, 3))

METATILES = ("H", "T", "P", "F")

#: Hats contained in each metatile, as (total, reflected). The H supertile
#: carries the tiling's entire supply of anti-hats: one reflected hat among
#: its four.
HATS_PER_METATILE = {"H": (4, 1), "T": (1, 0), "P": (2, 0), "F": (2, 0)}


def _charpoly():
    """Characteristic polynomial of the substitution matrix, over Z.

    Computed by exact Leverrier--Faddeev so nothing numerical enters; the
    matrix is 4x4 and the coefficients are small.
    """
    n = 4
    M = [[F(SUBSTITUTION[i][j]) for j in range(n)] for i in range(n)]
    coeffs = [F(1)]
    A = [row[:] for row in M]
    for k in range(1, n + 1):
        c = -sum(A[i][i] for i in range(n)) / k
        coeffs.append(c)
        if k < n:
            for i in range(n):
                A[i][i] += c
            A = [[sum(M[i][t] * A[t][j] for t in range(n)) for j in range(n)]
                 for i in range(n)]
    return coeffs                      # x^4 + c1 x^3 + c2 x^2 + c3 x + c4


def inflation_factor():
    r"""The Perron eigenvalue of the substitution, exactly.

    The characteristic polynomial is computed over $\mathbb{Z}$ and the
    candidate $\varphi^4 = (7+3\sqrt5)/2$ is verified as a root by exact
    evaluation in $\mathbb{Q}(\sqrt5)$ --- not by numerical eigensolving.
    The polynomial factors as $(x^2-7x+1)(x^2-1)$: the first factor has the
    conjugate pair $\varphi^{\pm4}$, whose product is $1$ because the
    substitution matrix has determinant one on that block, and the second
    contributes $\pm1$.

    Returns
    -------
    dict
        ``value`` ($\varphi^4$ as a :class:`Quad` over $\sqrt5$),
        ``charpoly`` (integer coefficients, monic, degree first),
        ``is_root`` and ``is_phi4``.
    """
    coeffs = _charpoly()
    lam = PHI4
    acc = Quad(0, 0, 5)
    for c in coeffs:                   # Horner, exactly
        acc = acc * lam + Quad(c, 0, 5)
    seven = Quad(7, 0, 5)
    phi4_check = lam * lam - seven * lam + 1        # root of x^2 - 7x + 1
    return {"value": lam,
            "charpoly": [int(c) if c.denominator == 1 else c for c in coeffs],
            "is_root": acc == Quad(0, 0, 5),
            "is_phi4": phi4_check == Quad(0, 0, 5)}


def metatile_frequencies():
    r"""The relative frequencies of H, T, P, F, over $\mathbb{Q}(\sqrt5)$.

    The Perron right-eigenvector: solve $(M - \varphi^4) v = 0$ by exact
    elimination in $\mathbb{Q}(\sqrt5)$ and normalise the entries to sum to
    one. Every entry is irrational, which is the seed of
    :func:`is_aperiodic`.
    """
    lam = PHI4
    n = 4
    A = [[Quad(SUBSTITUTION[i][j], 0, 5) for j in range(n)] for i in range(n)]
    for i in range(n):
        A[i][i] = A[i][i] - lam
    # exact kernel of a rank-3 matrix over Q(sqrt5)
    v = _kernel_1d(A)
    total = v[0] + v[1] + v[2] + v[3]
    freqs = [x / total for x in v]
    return dict(zip(METATILES, freqs))


def _kernel_1d(A):
    """One kernel vector of a square matrix over a Quad field, exactly."""
    n = len(A)
    M = [row[:] for row in A]
    pivots = []
    r = 0
    for c in range(n):
        p = next((i for i in range(r, n) if M[i][c] != Quad(0, 0, M[i][c].d)),
                 None)
        if p is None:
            continue
        M[r], M[p] = M[p], M[r]
        inv = M[r][c].inverse()
        M[r] = [x * inv for x in M[r]]
        for i in range(n):
            if i != r and M[i][c] != Quad(0, 0, M[i][c].d):
                f = M[i][c]
                M[i] = [M[i][j] - f * M[r][j] for j in range(n)]
        pivots.append(c)
        r += 1
    free = [c for c in range(n) if c not in pivots]
    if len(free) != 1:
        raise ValueError("expected a one-dimensional kernel; the eigenvalue "
                         "is not simple or is not an eigenvalue")
    fc = free[0]
    d = A[0][0].d
    v = [Quad(0, 0, d)] * n
    v[fc] = Quad(1, 0, d)
    for i, pc in enumerate(pivots):
        v[pc] = -M[i][fc]
    return v


def hat_chirality():
    r"""The ratio of unreflected to reflected hats: $\varphi^4 : 1$, derived.

    Weight the metatile frequencies by the hat counts of
    :data:`HATS_PER_METATILE`. Every anti-hat in the tiling sits inside an H
    supertile, so the ratio is

        (4 f_H + f_T + 2 f_P + 2 f_F - f_H) / f_H

    and it comes out exactly $\varphi^4$. The Hat tiling is *almost* chiral:
    one reflected tile per $\varphi^4 \approx 6.854$ unreflected ones, and
    the imbalance is not a decimal but an element of $\mathbb{Q}(\sqrt5)$.
    The Spectre, at the mirror-fixed point of the family, needs no reflected
    tiles at all.

    Returns
    -------
    dict
        ``ratio`` (a :class:`Quad`), ``is_phi4``, ``reflected_fraction``
        (of all hats, exactly $1/(1+\varphi^4)$).
    """
    f = metatile_frequencies()
    total = Quad(0, 0, 5)
    refl = Quad(0, 0, 5)
    for name, (t, r) in HATS_PER_METATILE.items():
        total = total + Quad(t, 0, 5) * f[name]
        refl = refl + Quad(r, 0, 5) * f[name]
    ratio = (total - refl) / refl
    return {"ratio": ratio, "is_phi4": ratio == PHI4,
            "reflected_fraction": refl / total}


def is_aperiodic():
    r"""Why the Hat tiling has no period, from the frequencies alone.

    A tiling with a period is a periodic arrangement of a fundamental
    domain, and in a fundamental domain every metatile occurs an integer
    number of times, so every relative frequency is rational. The derived
    frequencies are elements of $\mathbb{Q}(\sqrt5)$ with non-zero
    irrational part, so no period exists. The headline fact about the Hat
    is, at this level of description, the irrationality of an eigenvector
    --- the same shape of argument that quasicrystal diffraction rests on.

    Returns
    -------
    dict
        ``aperiodic``, ``witness`` (a frequency), ``rational_part``,
        ``irrational_part``.
    """
    f = metatile_frequencies()
    for name in METATILES:
        if not f[name].is_rational():
            return {"aperiodic": True, "witness": name,
                    "rational_part": f[name].p,
                    "irrational_part": f[name].q}
    return {"aperiodic": False, "witness": None,
            "rational_part": None, "irrational_part": None}


# ---------------------------------------------------------------------------
# the Laves lattice, and exact real-space topology on it
# ---------------------------------------------------------------------------


def laves_patch(a=None, b=None, rings=1):
    r"""A finite patch of the [3.4.6.4] Laves kite lattice, exactly.

    Every Tile$(a,b)$ is a union of eight kites of this lattice --- the Hat
    made headlines as an "einstein" but is, underneath, a polykite --- so
    the lattice is the substrate all the tilings share, and its two bond
    lengths *are* the parameters $a$ and $b$: hexagon centre to edge
    midpoint, and edge midpoint to hexagon vertex, meeting at right angles.
    Bond directions are multiples of $30^\circ$, whose cosines lie in
    $\tfrac12\mathbb{Z}[\sqrt3]$, so every vertex coordinate is exact.

    This patch is periodic, and the docstring says so: generating a genuine
    aperiodic vertex patch requires implementing the substitution geometry,
    which this module does not do. What the patch supplies is the correct
    local geometry --- the sites, bonds and bond angles that any Tile$(a,b)$
    model inherits --- and a testbed on which the exact-signature machinery
    below is demonstrated end to end.

    Parameters
    ----------
    a, b : Quad or rational, optional
        The two bond lengths. Default $a = 1$, $b = 1$ (the Spectre's
        proportions; the Hat's are $a : b = 1 : \sqrt3$).
    rings : int
        0 for a single hexagon's worth of kites, 1 to add its six
        neighbours, and so on.

    Returns
    -------
    dict
        ``sites`` (list of exact (x, y) pairs), ``bonds`` (list of
        (i, j, direction) with the direction as an exact unit vector), and
        ``kind`` per site: ``"centre"``, ``"edge"`` or ``"vertex"``.
    """
    a = _q3(1 if a is None else a)
    b = _q3(1 if b is None else b)
    half = Quad(F(1, 2), 0, 3)
    # unit vectors at 30 degree multiples: (cos, sin) in (1/2) Z[sqrt3]
    unit = []
    for k in range(12):
        c = {0: Quad(1, 0, 3), 1: half * SQRT3, 2: half,
             3: Quad(0, 0, 3), 4: -half, 5: -half * SQRT3,
             6: Quad(-1, 0, 3), 7: -half * SQRT3, 8: -half,
             9: Quad(0, 0, 3), 10: half, 11: half * SQRT3}[k]
        s = {0: Quad(0, 0, 3), 1: half, 2: half * SQRT3,
             3: Quad(1, 0, 3), 4: half * SQRT3, 5: half,
             6: Quad(0, 0, 3), 7: -half, 8: -half * SQRT3,
             9: Quad(-1, 0, 3), 10: -half * SQRT3, 11: -half}[k]
        unit.append((c, s))

    two_a = a + a
    centres = [(Quad(0, 0, 3), Quad(0, 0, 3))]
    if rings >= 1:
        for k in range(1, 12, 2):          # neighbours across edge midpoints
            centres.append((two_a * unit[k][0], two_a * unit[k][1]))

    sites, kind, index = [], [], {}

    def add(pt, what):
        if pt not in index:
            index[pt] = len(sites)
            sites.append(pt)
            kind.append(what)
        return index[pt]

    bonds = []
    for (cx, cy) in centres:
        ci = add((cx, cy), "centre")
        for k in range(1, 12, 2):          # edge midpoints at odd multiples
            ex = cx + a * unit[k][0]
            ey = cy + a * unit[k][1]
            ei = add((ex, ey), "edge")
            bonds.append((ci, ei, unit[k]))
            for turn in (3, 9):            # +-90 degrees to the two vertices
                kk = (k + turn) % 12
                vx = ex + b * unit[kk][0]
                vy = ey + b * unit[kk][1]
                vi = add((vx, vy), "vertex")
                bonds.append((ei, vi, unit[kk]))

    seen = set()
    uniq = []
    for i, j, d in bonds:
        key = (min(i, j), max(i, j))
        if key in seen:
            continue
        seen.add(key)
        uniq.append((i, j, d))
    return {"sites": sites, "bonds": uniq, "kind": kind}


def qwz_hamiltonian(patch, M, t=1):
    r"""A two-orbital Chern-insulator model on a patch, exactly.

    The Qi--Wu--Zhang model transplanted to a graph, which is the standard
    move for amorphous and quasicrystalline topology: on-site term
    $M\sigma_z$, and along a bond of direction $(\cos\theta, \sin\theta)$
    the hopping

        $T(\theta) = \tfrac{t}{2}\bigl[i(\cos\theta\,\sigma_x
                     + \sin\theta\,\sigma_y) - \sigma_z\bigr]$ .

    On the Laves patch every $\cos\theta$ and $\sin\theta$ is in
    $\tfrac12\mathbb{Z}[\sqrt3]$, so with rational $M$ and $t$ the
    Hamiltonian is a complex Hermitian matrix whose real and imaginary
    parts are both exact.

    Returns
    -------
    (Hre, Him)
        Two square matrices of :class:`Quad` entries with
        $H = H_{re} + i H_{im}$, $H_{re}$ symmetric and $H_{im}$
        antisymmetric.
    """
    M = _q3(M)
    t = _q3(t)
    half_t = t * Quad(F(1, 2), 0, 3)
    n = 2 * len(patch["sites"])
    zero = Quad(0, 0, 3)
    Hre = [[zero] * n for _ in range(n)]
    Him = [[zero] * n for _ in range(n)]

    for s in range(len(patch["sites"])):
        Hre[2 * s][2 * s] = M                    # +M on orbital 1
        Hre[2 * s + 1][2 * s + 1] = -M           # -M on orbital 2

    for (i, j, (c, s)) in patch["bonds"]:
        # T = (t/2) [ i (c sx + s sy) - sz ]
        # rows/cols: (site, orbital); T acts c_i^dag T c_j, plus h.c.
        # i*c*sx: entries (0,1) and (1,0) get i * c * t/2  -> imaginary part
        # i*s*sy: sy = [[0,-i],[i,0]]; i*sy = [[0,1],[-1,0]] -> real part
        # -sz:    diagonal -1, +1 -> real part
        blocks_re = [[-half_t, half_t * s],
                     [-half_t * s, half_t]]
        blocks_im = [[zero, half_t * c],
                     [half_t * c, zero]]
        for aa in range(2):
            for bb in range(2):
                r_, c_ = 2 * i + aa, 2 * j + bb
                Hre[r_][c_] = Hre[r_][c_] + blocks_re[aa][bb]
                Him[r_][c_] = Him[r_][c_] + blocks_im[aa][bb]
                # h.c.: H_{ji} = conj(H_ij)^T
                Hre[c_][r_] = Hre[c_][r_] + blocks_re[aa][bb]
                Him[c_][r_] = Him[c_][r_] - blocks_im[aa][bb]
    return Hre, Him


def localizer_signature(Hre, Him, patch, x0=0, y0=0, E0=0, kappa=F(1, 2)):
    r"""The spectral localizer signature, exactly.

    The localizer of Loring and Schulz-Baldes bundles the Hamiltonian with
    the position operators,

        $L = \begin{pmatrix} H - E_0 & \kappa\,\Pi \\
             \kappa\,\Pi^\dagger & -(H - E_0) \end{pmatrix}$,
        $\qquad \Pi = (X - x_0) - i (Y - y_0)$,

    and the local topological index at $(x_0, y_0, E_0)$ is half the
    signature of this Hermitian matrix. *Half the signature of a matrix*:
    that is the entire invariant, and a signature is a count of pivot signs
    in an $LDL^T$ decomposition, computable over any ordered field with no
    eigenvalue ever taken. The complex Hermitian localizer is doubled to a
    real symmetric matrix $[[A, -B], [B, A]]$, which doubles the signature
    again, and the elimination runs in $\mathbb{Q}(\sqrt3)$ throughout.

    Returns
    -------
    dict
        ``signature`` of the real doubled matrix, ``index`` (an integer,
        the local Chern number when the gaps are open), ``size``, and
        ``zero_pivots`` (non-zero means the localizer gap is closed at this
        point and the index is undefined there --- reported, not fudged).
    """
    x0, y0 = _q3(x0), _q3(y0)
    E0, kappa = _q3(E0), _q3(kappa)
    n = len(Hre)
    zero = Quad(0, 0, 3)

    # position operators on the doubled (site, orbital) index
    X = [zero] * n
    Y = [zero] * n
    for s, (px, py) in enumerate(patch["sites"]):
        X[2 * s] = X[2 * s + 1] = px - x0
        Y[2 * s] = Y[2 * s + 1] = py - y0

    # localizer L = [[H-E, k(X - iY)], [k(X + iY), -(H-E)]], complex
    m = 2 * n
    Lre = [[zero] * m for _ in range(m)]
    Lim = [[zero] * m for _ in range(m)]
    for i in range(n):
        for j in range(n):
            h_re = Hre[i][j] - (E0 if i == j else zero)
            Lre[i][j] = h_re
            Lim[i][j] = Him[i][j]
            Lre[n + i][n + j] = -h_re
            Lim[n + i][n + j] = -Him[i][j]
        Lre[i][n + i] = kappa * X[i]
        Lim[i][n + i] = -kappa * Y[i]
        Lre[n + i][i] = kappa * X[i]
        Lim[n + i][i] = kappa * Y[i]

    # real symmetric doubling [[A, -B], [B, A]]
    N = 2 * m
    S = [[zero] * N for _ in range(N)]
    for i in range(m):
        for j in range(m):
            S[i][j] = Lre[i][j]
            S[m + i][m + j] = Lre[i][j]
            S[i][m + j] = -Lim[i][j]
            S[m + i][j] = Lim[i][j]

    sig, zeros = _signature(S)
    idx = None
    if zeros == 0 and sig % 4 == 0:
        idx = sig // 4                     # doubling twice quarters it
    return {"signature": sig, "index": idx, "size": N, "zero_pivots": zeros}


def _signature(S):
    """Signature of a symmetric Quad matrix by LDL^T with symmetric pivoting.

    Returns (positive - negative, zero_pivots). Exact throughout: the only
    operations are field arithmetic and the decidable sign of a Quad.

    When every remaining diagonal entry is zero but an off-diagonal one is
    not, a congruence fixes it: adding row j to row i and column j to column
    i is S -> P^T S P with P invertible, which Sylvester's law says leaves
    the signature alone, and it puts 2 S_ij on the diagonal. That is both
    simpler and easier to trust than eliminating around a two-by-two
    hyperbolic block, and the tests hit this path on the hand-checkable
    matrix [[0, 1], [1, 0]].
    """
    n = len(S)
    A = [row[:] for row in S]
    zero = Quad(0, 0, 3) if n == 0 or A[0][0].d == 3 else Quad(0, 0, A[0][0].d)
    pos = neg = zeros = 0
    k = 0
    while k < n:
        p = next((i for i in range(k, n) if A[i][i].sign() != 0), None)
        if p is None:
            # all remaining diagonal entries vanish; find any non-zero
            # off-diagonal entry and congruence it onto the diagonal
            hit = None
            for i in range(k, n):
                for j in range(i + 1, n):
                    if A[i][j].sign() != 0:
                        hit = (i, j)
                        break
                if hit:
                    break
            if hit is None:
                zeros += n - k
                break
            i, j = hit
            for c in range(n):                   # row i += row j
                A[i][c] = A[i][c] + A[j][c]
            for r in range(n):                   # col i += col j
                A[r][i] = A[r][i] + A[r][j]
            continue                             # retry the pivot search
        if p != k:
            A[k], A[p] = A[p], A[k]
            for row in A:
                row[k], row[p] = row[p], row[k]
        d = A[k][k]
        if d.sign() > 0:
            pos += 1
        else:
            neg += 1
        inv = d.inverse()
        for i in range(k + 1, n):
            if A[i][k].sign() == 0:
                continue
            f = A[i][k] * inv
            for j in range(k + 1, n):
                A[i][j] = A[i][j] - f * A[k][j]
        for i in range(k + 1, n):
            A[k][i] = zero
            A[i][k] = zero
        k += 1
    return pos - neg, zeros


def localizer_index(M, kappa=F(1, 2), rings=1, a=None, b=None,
                    x0=0, y0=0, E0=0):
    """Convenience: patch, Hamiltonian and localizer index in one call."""
    patch = laves_patch(a=a, b=b, rings=rings)
    Hre, Him = qwz_hamiltonian(patch, M)
    return localizer_signature(Hre, Him, patch, x0=x0, y0=y0, E0=E0,
                               kappa=kappa)


def phase_scan(Ms, kappa=F(1, 2), rings=1, a=None, b=None):
    """The localizer index over a list of exact mass values.

    Returns a list of (M, index) pairs. The interesting content is where the
    index jumps: between two sampled masses with different indices there is
    an exact topological transition, bracketed by rationals.
    """
    patch = laves_patch(a=a, b=b, rings=rings)
    out = []
    for M in Ms:
        Hre, Him = qwz_hamiltonian(patch, M)
        r = localizer_signature(Hre, Him, patch, kappa=kappa)
        out.append((M, r["index"]))
    return out
