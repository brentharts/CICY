r"""
pyCICY.twistor -- twistor geometry and tree-level scattering amplitudes.

Witten's observation is that the perturbative expansion of gauge theory, which
in Feynman diagrams is a swamp, becomes simple when written in twistor
space: an $n$-gluon tree amplitude localises on a holomorphic curve of degree
$d = k - 1$, and the amplitudes themselves are rational functions of spinor
brackets rather than of momenta. The point for a package like this one is
that "rational function of spinor brackets" means *exactly computable*. There
is no integral, no metric and no approximation anywhere in a tree amplitude.

What this module computes, and how it is checked
------------------------------------------------
Everything here is exact arithmetic over the rationals, and almost everything
is computed twice.

    spinor kinematics   exact rational lambda and lambda-tilde with momentum
                        conservation imposed by construction, not tested for
    Parke-Taylor        the closed form for maximally-helicity-violating
                        amplitudes
    BCFW recursion      the same amplitudes built from three-point ones,
                        which must agree with Parke-Taylor and does
    relations           cyclicity, reflection, U(1) decoupling and the rank
                        of the space of colour orderings

The last is the sharpest. The $(n-1)!$ colour orderings of a tree amplitude
are not independent: Kleiss-Kuijf relations cut them to $(n-2)!$ and BCJ
relations to $(n-3)!$. Those are statements about the rank of a matrix of
rational numbers, and :func:`ordering_rank` computes that rank exactly rather
than checking a handful of identities.

Complexified kinematics
-----------------------
The spinors are taken independent: lambda-tilde is not the conjugate of
lambda. That is not a convenience, it is the correct setting. Real Lorentzian
three-point kinematics is degenerate --- every bracket vanishes --- so BCFW
recursion, which builds everything from three-point amplitudes, only exists
after complexification. Working over the rationals with independent spinors
is exactly split-signature kinematics, where the three-point amplitudes are
honest nonzero rational numbers and the whole recursion closes without ever
leaving the field.

Momentum conservation is then a linear condition. Choosing the lambdas
freely, the tildes must satisfy sum_i lambda_i^a lambda-tilde_i^b = 0 for each
pair of indices, which says each column of lambda-tilde lies in the kernel of
a 2-by-n matrix. That kernel has dimension n - 2 and is computed exactly, so
the kinematics is momentum-conserving by construction and
:meth:`Kinematics.check` is a test of the code rather than of the numbers.

The geometry
------------
Twistor space is $\mathbb{P}^3$; compactified complexified Minkowski space is
the Grassmannian $G(2,4)$, which under the Pl\"ucker embedding is the Klein
quadric, a single quadric hypersurface in $\mathbb{P}^5$. That is a
configuration matrix, so the machinery this package already has for complete
intersections applies to it directly: :func:`twistor_geometry` computes the
Euler characteristics of the twistor double fibration from the same Chern
class routine used for orientifold fixed loci, and gets 4, 12 and 6 --- the
last being the number of Schubert cells of $G(2,4)$.

What is not here
----------------
Loops. The tree amplitudes below are exact and complete; the loop expansion
needs integrals over regions that are not determined by the rational data
here, and no loop amplitude is computed. The amplituhedron is represented
only by the combinatorics of its cells, :func:`positroid_cells`, not by its
geometry. And the twistor-string correspondence enters only as the degree
formula :func:`mhv_degree`, which is a statement the module records rather
than derives.

References
----------
Parke and Taylor, An amplitude for n-gluon scattering, Phys. Rev. Lett. 56
    (1986) 2459.
Witten, Perturbative gauge theory as a string theory in twistor space,
    Commun. Math. Phys. 252 (2004) 189.
Britto, Cachazo, Feng and Witten, Direct proof of tree-level recursion
    relation in Yang-Mills, Phys. Rev. Lett. 94 (2005) 181602.
Bern, Carrasco and Johansson, New relations for gauge-theory amplitudes,
    Phys. Rev. D78 (2008) 085011.
Arkani-Hamed, Bourjaily, Cachazo, Goncharov, Postnikov and Trnka,
    Grassmannian Geometry of Scattering Amplitudes, CUP (2016).
Atiyah, Dunajski and Mason, Twistor theory at fifty: from contour integrals
    to twistor strings, Proc. Roy. Soc. A473 (2017) 20170530.
"""

import itertools
import random as _random
from fractions import Fraction as F

__all__ = ["Kinematics", "angle", "square", "parke_taylor", "anti_mhv",
           "tree_amplitude", "ordering_rank", "u1_decoupling_residual", "bcj_residual",
           "decorated_permutations", "positroid_cells", "cell_dimension",
           "twistor_geometry", "mhv_degree", "penrose_helicity"]


# ---------------------------------------------------------------------------
# exact spinor kinematics
# ---------------------------------------------------------------------------


def angle(a, b):
    r"""The holomorphic bracket $\langle ab \rangle = a^1 b^2 - a^2 b^1$."""
    return F(a[0]) * F(b[1]) - F(a[1]) * F(b[0])


def square(a, b):
    r"""The anti-holomorphic bracket $[ab]$, the same contraction on tildes."""
    return F(a[0]) * F(b[1]) - F(a[1]) * F(b[0])


class Kinematics(object):
    r"""Momentum-conserving massless kinematics, exactly over the rationals.

    A massless momentum in four dimensions is a rank-one bispinor
    $p^{a\dot b} = \lambda^a \tilde\lambda^{\dot b}$, and momentum
    conservation is $\sum_i \lambda_i^a \tilde\lambda_i^{\dot b} = 0$. With
    the two spinors independent -- complexified, or split signature -- that is
    a linear condition, so a configuration can be *constructed* rather than
    solved for numerically.

    Parameters
    ----------
    lam : list of pairs
        The $\lambda_i$, one pair of rationals per leg.
    lamt : list of pairs
        The $\tilde\lambda_i$. Must satisfy momentum conservation; use
        :meth:`random` to get a conserving pair.

    Examples
    --------
    >>> k = Kinematics.random(5, seed=1)
    >>> k.check()
    True
    >>> k.n
    5
    """

    def __init__(self, lam, lamt, check=True):
        self.lam = [tuple(F(x) for x in v) for v in lam]
        self.lamt = [tuple(F(x) for x in v) for v in lamt]
        if len(self.lam) != len(self.lamt):
            raise ValueError("lambda and lambda-tilde need the same length")
        self.n = len(self.lam)
        if self.n < 3:
            raise ValueError("an amplitude needs at least three legs")
        if check and not self.check():
            raise ValueError(
                "these spinors do not conserve momentum; the tildes must lie "
                "in the kernel of the 2-by-n matrix of lambdas. "
                "Kinematics.random builds a conserving configuration.")

    # -- construction ------------------------------------------------------

    @classmethod
    def random(cls, n, seed=0, bound=6, kind="holomorphic"):
        r"""A random momentum-conserving configuration, exact.

        The lambdas are chosen freely. Each column of lambda-tilde must then
        lie in the kernel of the 2-by-n matrix whose columns are the lambdas,
        a space of dimension $n - 2$; a random rational vector in that kernel
        is a valid choice, and the configuration conserves momentum by
        construction.

        Generic position matters: an amplitude has poles where consecutive
        brackets vanish, so the constructor rejects and redraws any
        configuration with a vanishing bracket rather than dividing by zero
        later.

        Three points are the exception, and the exception is the physics. For
        $n = 3$ the kernel above is one-dimensional, so both columns of
        lambda-tilde are proportional to a single vector and every square
        bracket vanishes identically. There is no generic three-point
        configuration with both sets of brackets non-zero, over any field:
        momentum conservation forbids it. The two three-point amplitudes
        therefore live at different kinematic points, and ``kind`` selects
        which --- ``"holomorphic"`` for non-zero angle brackets, which
        supports the MHV amplitude, or ``"antiholomorphic"`` for the
        conjugate. This is exactly the degeneracy that makes the three-point
        amplitude special, and the reason BCFW needs complex momenta at all.
        """
        n = int(n)
        if n < 3:
            raise ValueError("an amplitude needs at least three legs; got %d"
                             % n)
        if n == 3:
            return cls._three_point_kinematics(seed, bound, kind)
        rng = _random.Random(seed)
        for _ in range(400):
            lam = [(F(rng.randint(1, bound)), F(rng.randint(1, bound)))
                   for _ in range(n)]
            if any(angle(lam[i], lam[j]) == 0
                   for i in range(n) for j in range(i + 1, n)):
                continue
            basis = _kernel([[lam[i][0] for i in range(n)],
                             [lam[i][1] for i in range(n)]])
            if len(basis) != n - 2:
                continue
            cols = []
            ok = True
            for _c in range(2):
                v = [F(0)] * n
                for b in basis:
                    c = F(rng.randint(-bound, bound))
                    v = [x + c * y for x, y in zip(v, b)]
                cols.append(v)
            lamt = [(cols[0][i], cols[1][i]) for i in range(n)]
            if any(square(lamt[i], lamt[j]) == 0
                   for i in range(n) for j in range(i + 1, n)):
                ok = False
            if not ok:
                continue
            k = cls(lam, lamt, check=False)
            if k.check():
                return k
        raise RuntimeError("failed to build generic kinematics for n = %d" % n)

    @classmethod
    def _three_point_kinematics(cls, seed, bound, kind):
        rng = _random.Random(seed)
        for _ in range(400):
            v = [(F(rng.randint(1, bound)), F(rng.randint(1, bound)))
                 for _ in range(3)]
            if any(angle(v[i], v[j]) == 0 for i in range(3)
                   for j in range(i + 1, 3)):
                continue
            basis = _kernel([[v[i][0] for i in range(3)],
                             [v[i][1] for i in range(3)]])
            if len(basis) != 1:
                continue
            w = basis[0]
            c1, c2 = F(rng.randint(1, bound)), F(rng.randint(1, bound))
            other = [(c1 * w[i], c2 * w[i]) for i in range(3)]
            if kind == "holomorphic":
                lam, lamt = v, other
            elif kind == "antiholomorphic":
                lam, lamt = other, v
            else:
                raise ValueError("kind is 'holomorphic' or 'antiholomorphic'")
            k = cls(lam, lamt, check=False)
            if k.check():
                return k
        raise RuntimeError("failed to build three-point kinematics")

    # -- the invariants ----------------------------------------------------

    def angle(self, i, j):
        r"""$\langle ij \rangle$, with legs numbered from one."""
        return angle(self.lam[i - 1], self.lam[j - 1])

    def square(self, i, j):
        r"""$[ij]$, with legs numbered from one."""
        return square(self.lamt[i - 1], self.lamt[j - 1])

    def s(self, *legs):
        r"""The Mandelstam $(\sum p_i)^2$, as the determinant of the bispinor."""
        P = [[F(0), F(0)], [F(0), F(0)]]
        for i in legs:
            l, t = self.lam[i - 1], self.lamt[i - 1]
            for a in range(2):
                for b in range(2):
                    P[a][b] += l[a] * t[b]
        return P[0][0] * P[1][1] - P[0][1] * P[1][0]

    def check(self):
        r"""Whether $\sum_i \lambda_i \tilde\lambda_i = 0$, exactly."""
        for a in range(2):
            for b in range(2):
                tot = sum(self.lam[i][a] * self.lamt[i][b]
                          for i in range(self.n))
                if tot != 0:
                    return False
        return True

    def schouten_residual(self, i, j, k, l):
        r"""The Schouten identity, which two-component spinors must satisfy.

        $\langle ij\rangle\langle kl\rangle + \langle ik\rangle\langle lj
        \rangle + \langle il\rangle\langle jk\rangle = 0$ holds because there
        is no antisymmetric three-index tensor in two dimensions. It is not
        imposed anywhere; that it comes out zero tests the bracket.
        """
        return (self.angle(i, j) * self.angle(k, l)
                + self.angle(i, k) * self.angle(l, j)
                + self.angle(i, l) * self.angle(j, k))

    def momentum_residual(self, k, l):
        r"""$\sum_i \langle ki\rangle [il]$, zero by momentum conservation."""
        return sum(self.angle(k, i) * self.square(i, l)
                   for i in range(1, self.n + 1))

    def __repr__(self):
        return "<Kinematics n=%d, exact over Q>" % self.n


def _kernel(rows):
    """Exact kernel basis of a matrix over the rationals."""
    m = len(rows)
    ncol = len(rows[0]) if m else 0
    A = [[F(x) for x in r] for r in rows]
    pivots, r = [], 0
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
                A[i] = [A[i][j] - f * A[r][j] for j in range(ncol)]
        pivots.append(c)
        r += 1
        if r == m:
            break
    free = [c for c in range(ncol) if c not in pivots]
    basis = []
    for fc in free:
        v = [F(0)] * ncol
        v[fc] = F(1)
        for i, pc in enumerate(pivots):
            v[pc] = -A[i][fc]
        basis.append(v)
    return basis


def rank(rows):
    """Exact rank of a matrix of rationals."""
    if not rows:
        return 0
    return len(rows[0]) - len(_kernel(rows))


# ---------------------------------------------------------------------------
# closed-form amplitudes
# ---------------------------------------------------------------------------


def parke_taylor(lam, negatives, order=None):
    r"""The Parke-Taylor amplitude, exact.

    For a colour-ordered tree amplitude with exactly two negative-helicity
    gluons $i$ and $j$,

        A = \langle ij \rangle^4 /
            (\langle 12\rangle \langle 23\rangle \cdots \langle n1\rangle) .

    One line, and it replaces a number of Feynman diagrams that grows
    faster than exponentially. It depends on the holomorphic spinors alone,
    which is the statement that an MHV amplitude is supported on a degree-one
    curve in twistor space -- a line.

    Parameters
    ----------
    lam : list of pairs, or Kinematics
    negatives : pair of int
        The two negative-helicity legs, numbered from one.
    order : list of int, optional
        The colour ordering; the identity by default.
    """
    lam = lam.lam if isinstance(lam, Kinematics) else lam
    n = len(lam)
    order = list(order) if order else list(range(1, n + 1))
    if len(negatives) != 2:
        raise ValueError("Parke-Taylor is for exactly two negative helicities")
    i, j = negatives
    num = angle(lam[i - 1], lam[j - 1]) ** 4
    den = F(1)
    for a in range(len(order)):
        b = (a + 1) % len(order)
        den *= angle(lam[order[a] - 1], lam[order[b] - 1])
    if den == 0:
        raise ZeroDivisionError("a consecutive bracket vanishes in this "
                                "ordering; the kinematics is not generic")
    return num / den


def anti_mhv(lamt, positives, order=None):
    r"""The conjugate amplitude, with exactly two positive helicities.

    The same expression with square brackets. Parity exchanges the two, and
    an amplitude computed by :func:`tree_amplitude` with $n-2$ negative
    helicities must agree with this.
    """
    lamt = lamt.lamt if isinstance(lamt, Kinematics) else lamt
    n = len(lamt)
    order = list(order) if order else list(range(1, n + 1))
    if len(positives) != 2:
        raise ValueError("this is for exactly two positive helicities")
    i, j = positives
    num = square(lamt[i - 1], lamt[j - 1]) ** 4
    den = F(1)
    for a in range(len(order)):
        b = (a + 1) % len(order)
        den *= square(lamt[order[a] - 1], lamt[order[b] - 1])
    if den == 0:
        raise ZeroDivisionError("a consecutive square bracket vanishes")
    return num / den


# ---------------------------------------------------------------------------
# BCFW recursion
# ---------------------------------------------------------------------------


def _three_point(lam, lamt, hel):
    """The two three-point amplitudes, which seed everything else.

    At three points momentum conservation forces either all the square
    brackets or all the angle brackets to vanish, so the amplitude is
    holomorphic or anti-holomorphic. Which one is fixed by the helicities.
    These exist only for complex momenta, which is why the recursion needs
    the spinors independent.
    """
    nm = hel.count(-1)
    if nm == 2:
        i, j = [a for a in range(3) if hel[a] == -1]
        num = angle(lam[i], lam[j]) ** 4
        den = (angle(lam[0], lam[1]) * angle(lam[1], lam[2])
               * angle(lam[2], lam[0]))
        return num / den if den != 0 else F(0)
    if nm == 1:
        i, j = [a for a in range(3) if hel[a] == 1]
        num = square(lamt[i], lamt[j]) ** 4
        den = (square(lamt[0], lamt[1]) * square(lamt[1], lamt[2])
               * square(lamt[2], lamt[0]))
        return num / den if den != 0 else F(0)
    return F(0)


def _bispinor(lam, lamt, legs):
    P = [[F(0), F(0)], [F(0), F(0)]]
    for i in legs:
        for a in range(2):
            for b in range(2):
                P[a][b] += lam[i][a] * lamt[i][b]
    return P


def _det(P):
    return P[0][0] * P[1][1] - P[0][1] * P[1][0]


def _factorise(P):
    """Write a rank-one bispinor as lambda times lambda-tilde, exactly."""
    if _det(P) != 0:
        raise ValueError("bispinor is not rank one; it cannot be an on-shell "
                         "momentum")
    for a in range(2):
        if P[a][0] != 0 or P[a][1] != 0:
            # row a is (lam^a) * lamt, so take lamt proportional to it
            lamt = (P[a][0], P[a][1])
            if P[a][0] != 0:
                lam = (P[0][0] / P[a][0], P[1][0] / P[a][0])
            else:
                lam = (P[0][1] / P[a][1], P[1][1] / P[a][1])
            return lam, lamt
    raise ValueError("the zero bispinor has no factorisation")


def tree_amplitude(lam, lamt, hel, _depth=0):
    r"""A colour-ordered tree amplitude, by BCFW recursion. Exact.

    Britto, Cachazo, Feng and Witten's recursion deforms two of the external
    momenta by a complex parameter, uses the fact that the deformed amplitude
    vanishes at large deformation, and reconstructs it from its poles. Each
    residue factorises into two amplitudes with fewer legs, so the whole tree
    expansion follows from the three-point amplitudes, which are fixed by
    Lorentz invariance alone.

    Parameters
    ----------
    lam, lamt : list of pairs
        Spinors, one per leg, in the colour order.
    hel : list of int
        Helicities, $+1$ or $-1$, one per leg.

    Returns
    -------
    Fraction
        The amplitude, exactly.

    Notes
    -----
    The shift used is $\tilde\lambda_i \to \tilde\lambda_i - z
    \tilde\lambda_j$, $\lambda_j \to \lambda_j + z \lambda_i$ on a
    cyclically adjacent pair with helicities $(-,+)$, which is the choice for
    which the deformed amplitude falls off at large $z$. Such a pair always
    exists once both helicities occur, and the ordering is rotated to put it
    at the ends; a colour-ordered amplitude is cyclic, so the rotation costs
    nothing.

    Amplitudes with fewer than two legs of either helicity vanish for
    $n \geq 4$. That is returned rather than derived, and is the one piece of
    input beyond the three-point amplitudes.
    """
    n = len(hel)
    if n != len(lam) or n != len(lamt):
        raise ValueError("spinors and helicities must have the same length")
    if n < 3:
        raise ValueError("a tree amplitude needs at least three legs")
    if n == 3:
        return _three_point(lam, lamt, hel)
    nm = hel.count(-1)
    if nm < 2 or n - nm < 2:
        return F(0)
    if _depth > 40:
        raise RuntimeError("BCFW recursion did not terminate")

    # Rotate so that legs n and 1 are a (-, +) adjacent pair.
    shift = None
    for r in range(n):
        if hel[(r - 1) % n] == -1 and hel[r % n] == 1:
            shift = r
            break
    if shift is None:
        raise RuntimeError("no adjacent (-, +) pair, which cannot happen "
                           "once both helicities are present")
    idx = [(shift + a) % n for a in range(n)]
    lam = [lam[a] for a in idx]
    lamt = [lamt[a] for a in idx]
    hel = [hel[a] for a in idx]
    # Now leg n (index n-1) has helicity -1 and leg 1 (index 0) has +1.

    total = F(0)
    for m in range(2, n - 1):
        left = list(range(0, m))            # legs 1..m, containing leg 1
        P = _bispinor(lam, lamt, left)
        P2 = _det(P)
        if P2 == 0:
            continue                        # no pole in this channel
        # The shift lambda_1 -> lambda_1 + z lambda_n moves P by z R.
        R = [[lam[n - 1][a] * lamt[0][b] for b in range(2)] for a in range(2)]
        c = (P[0][0] * R[1][1] + R[0][0] * P[1][1]
             - P[0][1] * R[1][0] - R[0][1] * P[1][0])
        if c == 0:
            continue
        z = -P2 / c
        Ph = [[P[a][b] + z * R[a][b] for b in range(2)] for a in range(2)]
        lp, ltp = _factorise(Ph)

        lam_s = list(lam)
        lamt_s = list(lamt)
        lam_s[0] = (lam[0][0] + z * lam[n - 1][0],
                    lam[0][1] + z * lam[n - 1][1])
        lamt_s[n - 1] = (lamt[n - 1][0] - z * lamt[0][0],
                         lamt[n - 1][1] - z * lamt[0][1])

        for h in (-1, 1):
            # Left: legs 1..m and the internal line carrying -P, helicity -h.
            lamL = [lam_s[a] for a in left] + [(-lp[0], -lp[1])]
            lamtL = [lamt_s[a] for a in left] + [ltp]
            helL = [hel[a] for a in left] + [-h]
            # Right: the internal line then legs m+1..n.
            right = list(range(m, n))
            lamR = [lp] + [lam_s[a] for a in right]
            lamtR = [ltp] + [lamt_s[a] for a in right]
            helR = [h] + [hel[a] for a in right]

            nmL, nmR = helL.count(-1), helR.count(-1)
            if len(helL) >= 4 and (nmL < 2 or len(helL) - nmL < 2):
                continue
            if len(helR) >= 4 and (nmR < 2 or len(helR) - nmR < 2):
                continue
            AL = tree_amplitude(lamL, lamtL, helL, _depth + 1)
            if AL == 0:
                continue
            AR = tree_amplitude(lamR, lamtR, helR, _depth + 1)
            if AR == 0:
                continue
            total += AL * AR / P2
    return total


# ---------------------------------------------------------------------------
# relations among colour orderings
# ---------------------------------------------------------------------------


def u1_decoupling_residual(kin, negatives):
    r"""The photon decoupling identity, which must vanish.

    Summing a colour-ordered amplitude over the cyclic insertions of one leg
    into the ordering of the rest gives zero:

        A(1,2,3,\dots,n) + A(2,1,3,\dots,n) + \dots + A(2,3,\dots,1,n) = 0 .

    It follows from the $U(1)$ inside $U(N)$ decoupling from the rest, and it
    is the simplest of the relations that cut the $(n-1)!$ orderings down.
    Here it is evaluated on Parke-Taylor amplitudes and must be exactly zero.
    """
    n = kin.n
    rest = list(range(2, n + 1))
    total = F(0)
    # n - 1 insertions, not n: putting leg 1 after leg n gives back the
    # ordering with it in front, since a colour-ordered amplitude is cyclic,
    # and including both leaves the sum equal to the amplitude rather than to
    # zero.
    for pos in range(len(rest)):
        order = rest[:pos] + [1] + rest[pos:]
        total += parke_taylor(kin, negatives, order)
    return total


def bcj_residual(kin, negatives):
    r"""The fundamental Bern-Carrasco-Johansson relation, which must vanish.

        \sum_{i=2}^{n-1} \Bigl( \sum_{j=2}^{i} s_{1j} \Bigr)
        A(2, \dots, i, 1, i+1, \dots, n) \;=\; 0 .

    Unlike :func:`u1_decoupling_residual` the coefficients depend on the
    momenta, through $s_{1j} = \langle 1j
angle [1j]$. That is the reason
    BCJ cuts the independent orderings further than Kleiss-Kuijf does, and
    also the reason :func:`ordering_rank` cannot see it --- see the note
    there.
    """
    n = kin.n
    rest = list(range(2, n + 1))
    total = F(0)
    for i in range(2, n):
        coef = sum(kin.angle(1, j) * kin.square(1, j) for j in range(2, i + 1))
        order = ([x for x in rest if x <= i] + [1]
                 + [x for x in rest if x > i])
        total += coef * parke_taylor(kin, negatives, order)
    return total


def ordering_rank(kin, negatives, relation="kk"):
    r"""The rank of the space spanned by the colour orderings. Exact.

    The $(n-1)!$ cyclically inequivalent orderings of a tree amplitude are
    not independent. Reflection and the Kleiss-Kuijf relations reduce the
    independent set to $(n-2)!$; the Bern-Carrasco-Johansson relations reduce
    it further to $(n-3)!$.

    This computes a rank rather than checking identities: the amplitude is
    evaluated for every ordering at many independent kinematic points, and
    the rank of the resulting matrix over the rationals is returned. Nothing
    about the relations is used as input.

    What comes back is $(n-2)!$, and the reason is worth stating because it
    is a limitation of the method rather than of the relations. A rank taken
    across kinematic points can only see relations whose coefficients are
    *constant*. Kleiss-Kuijf relations have coefficients $\pm 1$, so they hold
    at every point simultaneously and the rank drops to $(n-2)!$. The
    Bern-Carrasco-Johansson relations have coefficients built from Mandelstam
    invariants, which vary from point to point, so no single linear relation
    holds across the matrix and the further reduction to $(n-3)!$ is
    invisible here. To see BCJ one must work at a fixed kinematic point with
    the momentum-dependent coefficients supplied, which is what
    :func:`bcj_residual` does.

    Parameters
    ----------
    kin : Kinematics or list of Kinematics
        Kinematic points. At least $(n-2)!$ of them are needed for the rank
        to saturate; fewer bounds the rank by the number of points, which is
        reported so the shortfall is visible rather than silent.
    negatives : pair of int
    relation : str
        ``"kk"``, ``"bcj"`` or ``"cyclic"``, setting the ``expected`` field.

    Returns
    -------
    dict
        ``rank``, ``orderings``, ``points``, ``expected``, ``agrees`` and
        ``saturated``, the last being whether enough points were supplied for
        the rank to be meaningful.
    """
    points = kin if isinstance(kin, (list, tuple)) else [kin]
    n = points[0].n
    orders = [[1] + list(p) for p in itertools.permutations(range(2, n + 1))]
    rows = []
    for k in points:
        rows.append([parke_taylor(k, negatives, o) for o in orders])
    r = rank(rows)
    expected = {"bcj": _fact(n - 3), "kk": _fact(n - 2),
                "cyclic": _fact(n - 1)}[relation]
    return {"rank": r, "orderings": len(orders), "points": len(points),
            "expected": expected, "agrees": r == expected,
            "saturated": len(points) > r}


def _fact(k):
    out = 1
    for i in range(2, k + 1):
        out *= i
    return out


# ---------------------------------------------------------------------------
# the positive Grassmannian
# ---------------------------------------------------------------------------


def decorated_permutations(n):
    r"""Decorated permutations of $[n]$: permutations with coloured fixed points.

    Postnikov's classification says the cells of the totally non-negative
    Grassmannian $G(k,n)_{\geq 0}$ are in bijection with decorated
    permutations of $[n]$ having $k$ anti-exceedances. On-shell diagrams of
    planar $\mathcal{N}=4$ super Yang-Mills are labelled by the same objects,
    which is why the combinatorics belongs in a module about amplitudes.

    A decorated permutation is a bijection of $[n]$ together with a colour,
    ``+1`` or ``-1``, on each fixed point. Yields ``(perm, colours)`` with
    ``perm`` a tuple in one-line notation and ``colours`` a dict on the fixed
    points.
    """
    n = int(n)
    for perm in itertools.permutations(range(1, n + 1)):
        fixed = [i for i in range(1, n + 1) if perm[i - 1] == i]
        for signs in itertools.product((1, -1), repeat=len(fixed)):
            yield perm, dict(zip(fixed, signs))


def anti_exceedances(perm, colours):
    r"""The number of anti-exceedances, which is the $k$ of the cell.

    An anti-exceedance is an $i$ with $\pi^{-1}(i) > i$, together with each
    fixed point coloured $-1$ (a "loop"). This is the statistic that grades
    decorated permutations by which Grassmannian they belong to.
    """
    n = len(perm)
    inv = [0] * (n + 1)
    for i in range(1, n + 1):
        inv[perm[i - 1]] = i
    count = sum(1 for i in range(1, n + 1) if inv[i] > i)
    count += sum(1 for i, s in colours.items() if s == -1)
    return count


def positroid_cells(k, n):
    r"""The number of cells of $G(k,n)_{\geq 0}$, by counting.

    Returns
    -------
    int
        Decorated permutations of $[n]$ with $k$ anti-exceedances.

    Notes
    -----
    Summed over $k$ this must equal the total number of decorated
    permutations, and that total has a closed form: it is
    $\sum_{j=0}^{n} n!/j!$, the number of arrangements of $n$ objects. The
    identity is not obvious from either side --- one counts permutations with
    coloured fixed points, the other counts ordered selections --- and
    :func:`positroid_total_check` verifies it.
    """
    k, n = int(k), int(n)
    return sum(1 for perm, col in decorated_permutations(n)
               if anti_exceedances(perm, col) == k)


def positroid_total_check(n):
    """Total cells over all k, against sum_j n!/j!. Returns (counted, closed)."""
    n = int(n)
    counted = sum(1 for _ in decorated_permutations(n))
    fac = [1]
    for i in range(1, n + 1):
        fac.append(fac[-1] * i)
    closed = sum(fac[n] // fac[j] for j in range(n + 1))
    return counted, closed


def cell_dimension(k, n):
    r"""The dimension of the top cell of $G(k,n)_{\geq 0}$, which is $k(n-k)$."""
    return int(k) * (int(n) - int(k))


# ---------------------------------------------------------------------------
# twistor geometry
# ---------------------------------------------------------------------------


def twistor_geometry():
    r"""The twistor double fibration, as complete intersections.

    Twistor space is $\mathbb{P}^3$. Compactified complexified Minkowski
    space is the Grassmannian of two-planes in $\mathbb{C}^4$, which the
    Pl\"ucker embedding realises as a single quadric in $\mathbb{P}^5$ --- the
    Klein quadric --- so it has a configuration matrix, ``[[5, 2]]``, and the
    Chern class machinery this package already uses for orientifold fixed loci
    applies to it unchanged.

    The Euler characteristics come out 4, 12 and 6. The last is the number of
    Schubert cells of $G(2,4)$, indexed by the partitions fitting in a
    two-by-two box, and the middle is the twelve cells of the flag manifold
    $F(1,3;4)$, the $(1,1)$ hypersurface in
    $\mathbb{P}^3 \times \mathbb{P}^{3*}$ expressing the incidence
    $Z^\alpha W_\alpha = 0$.

    Returns
    -------
    list of dict
        ``name``, ``configuration``, ``dim``, ``euler`` and ``note``.
    """
    from .theories.orientifold import complete_intersection_euler as chi
    out = [
        {"name": "twistor space PT", "configuration": [[3]], "dim": 3,
         "euler": chi([3], []),
         "note": "P^3; the physical twistor space is this minus a line, on "
                 "which the Penrose transform is a first cohomology"},
        {"name": "incidence F(1,3;4)", "configuration": [[3, 1], [3, 1]],
         "dim": 5, "euler": chi([3, 3], [[1, 1]]),
         "note": "the (1,1) hypersurface Z^a W_a = 0 in P^3 x P^3*"},
        {"name": "Minkowski G(2,4)", "configuration": [[5, 2]], "dim": 4,
         "euler": chi([5], [[2]]),
         "note": "the Klein quadric; its Euler characteristic is the number "
                 "of Schubert cells, C(4,2)"},
    ]
    return out


def mhv_degree(k, loops=0):
    r"""The degree of the twistor-space curve an amplitude localises on.

    Witten's result: an $\mathrm{N}^{k-2}\mathrm{MHV}$ amplitude at $\ell$
    loops is supported on a holomorphic curve in twistor space of degree

        d = (k - 1) + \ell ,

    of genus at most $\ell$. For $k = 2$ at tree level $d = 1$: an MHV
    amplitude lives on a line, which is the geometric content of
    Parke-Taylor depending only on the holomorphic spinors.

    This is recorded rather than derived; nothing else in the module uses it.
    """
    k, loops = int(k), int(loops)
    if k < 2:
        raise ValueError("k is at least two; fewer than two negative "
                         "helicities gives a vanishing tree amplitude")
    return {"k": k, "loops": loops, "degree": k - 1 + loops,
            "max_genus": loops,
            "label": "MHV" if k == 2 else "N^%dMHV" % (k - 2)}


def penrose_helicity(h):
    r"""The Penrose transform, and why it needs an open twistor space.

    A free massless field of helicity $h$ on Minkowski space corresponds to a
    class in $H^1$ of twistor space with values in $\mathcal{O}(-2h-2)$. The
    twist is the content: helicity is read off the homogeneity degree of a
    function on twistor space, so the whole spectrum of massless free fields
    is a single sheaf cohomology with a varying line bundle.

    The caution the module records is that the cohomology cannot be taken on
    $\mathbb{P}^3$. By Bott's formula $H^1(\mathbb{P}^3, \mathcal{O}(m)) = 0$
    for every $m$, so the transform on compact twistor space would give
    nothing at all. It is taken on $\mathbb{P}^3$ minus a line, or on a
    tubular neighbourhood of a real slice, and that open space is not
    something this package's cohomology machinery covers.

    Returns
    -------
    dict
        ``helicity``, ``bundle``, ``h1_on_P3`` and ``note``.
    """
    h = F(h)
    twist = -2 * h - 2
    if twist.denominator != 1:
        raise ValueError("helicity must be a half-integer; got %s" % h)
    return {"helicity": h, "bundle": "O(%d)" % int(twist),
            "h1_on_P3": 0,
            "note": "H^1(P^3, O(m)) vanishes for every m by Bott, so the "
                    "Penrose transform is a cohomology of an open twistor "
                    "space, not of P^3"}
