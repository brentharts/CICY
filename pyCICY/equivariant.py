r"""
pyCICY.equivariant -- group actions on line bundles, and the index they carry.

The gap this closes
-------------------
:mod:`pyCICY.breaking` computes what survives on X/Gamma *given* the
representation of Gamma on the upstairs cohomology, and says plainly that it
cannot derive that representation: it needs an equivariant structure, a lift
of the Gamma action to the total space of each line bundle, which is extra
data beyond the configuration matrix and the charges.

This module supplies it, for the case that is exactly computable.

What is exactly computable, and why
-----------------------------------
The individual groups H^q(X, L) as Gamma-representations require running the
Leray spectral sequence equivariantly, and the ranks of its differentials are
not determined by the degrees -- the same obstruction that makes
:meth:`pyCICY.bundles.Monad.cohomology_bounds` return intervals.

The *alternating sum* is different. Characters are additive on exact
sequences, so the character-valued index

    ind_Gamma(L) = sum_q (-1)^q ch( H^q(X, L) )  in  R(Gamma)

can be computed from any equivariant resolution, with no spectral sequence
anywhere. The Koszul resolution of O_X inside the ambient product of
projective spaces,

    0 -> Lambda^K N* -> ... -> N* -> O_A -> O_X -> 0 ,   N* = sum_a O_A(-d_a),

does it: every term is a sum of ambient line bundles, whose equivariant Euler
characteristics factorise over the projective factors by Kunneth, and Gamma
acts on the a-th summand of N* by the inverse of the character of the a-th
defining polynomial. So

    ind_Gamma(O_X(k)) = sum_{S subset {1..K}} (-1)^{|S|}
                        (prod_{a in S} c_a^{-1}) chi_Gamma(A, O_A(k - sum_S d_a))

is a finite sum of computable terms, in exact integer arithmetic.

That is enough for the physics. The chiral spectrum -- the net number of
generations in each Gamma-charge sector, which is what determines what
survives the quotient and the Wilson line -- is an index. It is precisely the
part of the cohomology that does not depend on the unknown ranks.

What is *not* computed: h^1 and h^2 separately as Gamma-representations. The
non-chiral content, vector-like pairs that can lift, is out of reach here for
the same reason it is out of reach in :mod:`pyCICY.bundles`.

Equivariant structures are a choice
-----------------------------------
Once the Gamma action on the ambient coordinates is fixed, O_A(k) carries a
canonical linearisation, and every other differs from it by a character of
Gamma. So the equivariant structures on a line bundle form a torsor over the
character group -- for cyclic Gamma of order n, there are exactly n of them,
and :meth:`CyclicAction.twist` moves between them.

That is the moduli :mod:`pyCICY.breaking` warned about, now explicit and
finite. A rank-5 sum of line bundles has n^5 equivariant structures, of which
the ones with trivial total twist descend to the quotient;
:func:`bundle_index_character` takes the choice as an argument and
:func:`enumerate_structures` lists them.

**And for a free action the choice does not matter.** This was not expected
and is worth stating plainly. If Gamma acts freely then every Lefschetz
number vanishes, so the index character of *each summand* is a multiple of
the regular representation -- a constant vector. A twist permutes that vector
cyclically, and a constant vector is fixed by any permutation. So the
equivariant index, and with it the entire chiral spectrum downstairs, is
independent of which equivariant structure is chosen. The choice shows up
only in h^1 and h^2 separately, that is in the vector-like pairs, which the
index cannot see in any case.

The practical consequence for :mod:`pyCICY.breaking` is that the missing
ingredient it flagged was missing only for the non-chiral sector. The
generation count, and its distribution over Gamma-charges, is determined --
and is uniform. ``tests/test_equivariant.py`` checks both halves: that
twisting is inert for the free action, and that it is *not* inert for a
non-free one, so the argument is doing something rather than being ignored.

The freeness check
------------------
For g in Gamma acting *freely*, the holomorphic Lefschetz fixed point formula
has no fixed points to sum over, so the Lefschetz number

    L(g) = sum_q (-1)^q tr( g | H^q(X, L) )

vanishes for every g != e. Vanishing of all of those is exactly the statement
that ind_Gamma(L) is an integer multiple of the regular representation -- the
charges are equidistributed. :func:`is_regular_multiple` tests it.

This is a necessary condition for freeness, not a sufficient one, and it is
checked here as a diagnostic on the action rather than asserted. It is also a
result worth having in its own right: it *derives* the equidistribution that
:func:`pyCICY.breaking.worked_example` had to assume by hand.

Scope
-----
* Cyclic Gamma acting diagonally on the homogeneous coordinates, by phases.
  Actions permuting coordinates within a factor, or permuting the ambient
  factors, are not implemented; the latter would also require the charges of
  the line bundle to be permutation-invariant before an equivariant structure
  exists at all.
* Freeness is diagnosed, not decided. Deciding it needs the defining
  polynomials, not just their degrees and characters, and this package works
  with configuration matrices.
* The character of the defining polynomials is an input. Whether a polynomial
  of a given multidegree and character exists is checkable from the monomials
  and :meth:`CyclicAction.admissible_polynomial_charges` checks it; whether
  the particular one cutting out X has that character is not.
"""

import itertools

import numpy as np

__all__ = [
    "CyclicAction", "regular_representation", "is_regular_multiple",
    "bundle_index_character", "enumerate_structures", "gamma_charges",
    "TETRAQUADRIC_Z2",
]


def _convolve(a, b, n):
    """Product of two characters, as multiplicities over Z_n irreps."""
    out = [0] * n
    for i, x in enumerate(a):
        if not x:
            continue
        for j, y in enumerate(b):
            if y:
                out[(i + j) % n] += x * y
    return out


class CyclicAction(object):
    r"""
    A Z_n action on a CICY, by phases on the ambient homogeneous coordinates.

    Parameters
    ----------
    conf : configuration matrix
        The usual pyCICY form: row i is ``[n_i, d_i1, ..., d_iK]``.
    order : int
        n, the order of the cyclic group.
    weights : list of lists
        ``weights[i][j]`` is the exponent w with which the generator acts on
        the j-th homogeneous coordinate of the i-th projective factor,
        x -> zeta^w x with zeta a primitive n-th root of unity. Factor i needs
        ``n_i + 1`` weights.
    polynomial_charges : list of int
        The charge of each defining polynomial: the generator multiplies the
        a-th polynomial by zeta^{c_a}. This is data about the polynomials, not
        about their degrees, and is an input.

    Notes
    -----
    Weights are only meaningful modulo n, and shifting all weights of one
    factor by a constant rescales the coordinates of that factor by a global
    phase, which acts trivially on the projective space but does change the
    linearisation of O(k). :meth:`normalised` puts them in a canonical form so
    that two descriptions of the same action compare equal.
    """

    def __init__(self, conf, order, weights, polynomial_charges):
        self.conf = np.asarray(conf, dtype=int)
        self.n = int(order)
        if self.n < 1:
            raise ValueError("order must be a positive integer")
        self.dims = self.conf[:, 0]
        self.degrees = self.conf[:, 1:].T          # degrees[a] = multidegree
        self.K = self.degrees.shape[0]
        if len(weights) != len(self.dims):
            raise ValueError(
                "need one weight list per ambient factor: %d given, %d needed"
                % (len(weights), len(self.dims)))
        for i, w in enumerate(weights):
            if len(w) != self.dims[i] + 1:
                raise ValueError(
                    "factor %d is P^%d and needs %d weights, got %d"
                    % (i, self.dims[i], self.dims[i] + 1, len(w)))
        self.weights = [[int(x) % self.n for x in w] for w in weights]
        if len(polynomial_charges) != self.K:
            raise ValueError(
                "need one charge per defining polynomial: %d given, %d needed"
                % (len(polynomial_charges), self.K))
        self.polynomial_charges = [int(c) % self.n for c in polynomial_charges]

    def __repr__(self):
        return "<CyclicAction Z_%d weights=%s poly_charges=%s>" % (
            self.n, self.weights, self.polynomial_charges)

    # -- ambient ----------------------------------------------------------

    def _chi_projective(self, k, w):
        r"""
        Equivariant Euler characteristic of O(k) on a single P^d.

        Only two of the cohomology groups can be non-zero, and both have
        monomial bases on which the action is diagonal:

            k >= 0        H^0 has basis the degree-k monomials
            -d <= k < 0   everything vanishes
            k <= -d-1     H^d has basis the Laurent monomials of degree k with
                          every exponent at most -1, contributing (-1)^d

        Writing such a Laurent monomial as x^{-gamma-1} with gamma >= 0 and
        |gamma| = -k-d-1 gives the shift by -sum(w) below: that shift is the
        character of the canonical bundle, and forgetting it is the easiest
        way to get an answer that is right at the identity and wrong
        everywhere else.
        """
        d = len(w) - 1
        out = [0] * self.n
        if k >= 0:
            for mono in itertools.combinations_with_replacement(range(d + 1), k):
                out[sum(w[j] for j in mono) % self.n] += 1
        elif k <= -d - 1:
            total = sum(w) % self.n
            sign = (-1) ** d
            for mono in itertools.combinations_with_replacement(
                    range(d + 1), -k - d - 1):
                c = (-sum(w[j] for j in mono) - total) % self.n
                out[c] += sign
        return out

    def ambient_euler(self, kvec):
        """Equivariant Euler characteristic of O_A(k) on the ambient product.

        Kunneth: the character is the product over factors, which on
        multiplicity vectors is a convolution.
        """
        kvec = list(kvec)
        if len(kvec) != len(self.dims):
            raise ValueError("k needs one entry per ambient factor")
        out = [0] * self.n
        out[0] = 1
        for k, w in zip(kvec, self.weights):
            out = _convolve(out, self._chi_projective(k, w), self.n)
        return out

    # -- on X, by Koszul --------------------------------------------------

    def euler(self, kvec, twist=0):
        r"""
        The character-valued index of O_X(k), as Z_n irrep multiplicities.

        ``twist`` selects the equivariant structure: the canonical one is
        ``twist=0`` and the others are its translates by characters, so
        ``euler(k, j)`` is ``euler(k, 0)`` shifted cyclically by j.

        Entry ``c`` of the result is
        ``sum_q (-1)^q dim H^q(X, L)_c``, the *net* multiplicity of charge c.
        It can be negative, as any index can.

        The total, ``sum(euler(k))``, must equal
        :meth:`pyCICY.CICY.line_co_euler` -- separate code, a separate formula,
        and floating point where this is exact integer arithmetic.
        """
        out = [0] * self.n
        for r in range(self.K + 1):
            for S in itertools.combinations(range(self.K), r):
                kk = [int(kvec[i]) - int(sum(self.degrees[a][i] for a in S))
                      for i in range(len(self.dims))]
                ch = self.ambient_euler(kk)
                shift = (-sum(self.polynomial_charges[a] for a in S)) % self.n
                sign = (-1) ** r
                for c in range(self.n):
                    out[(c + shift) % self.n] += sign * ch[c]
        if twist:
            t = int(twist) % self.n
            out = [out[(c - t) % self.n] for c in range(self.n)]
        return out

    # -- consistency ------------------------------------------------------

    def admissible_polynomial_charges(self, a):
        """Charges a defining polynomial of multidegree ``d_a`` could carry.

        A polynomial is a Gamma-eigenvector only if all its monomials share a
        charge, so the possible charges are those actually realised by some
        monomial of that multidegree. A charge outside this set describes no
        polynomial at all.
        """
        deg = self.degrees[a]
        out = [0] * self.n
        out[0] = 1
        for i, k in enumerate(deg):
            per = [0] * self.n
            for mono in itertools.combinations_with_replacement(
                    range(self.dims[i] + 1), int(k)):
                per[sum(self.weights[i][j] for j in mono) % self.n] += 1
            out = _convolve(out, per, self.n)
        return sorted(c for c in range(self.n) if out[c] > 0)

    def check(self):
        """Validate the action against the configuration.

        Checks that each declared polynomial charge is realisable by some
        monomial of the right multidegree. Returns ``(ok, messages)``.
        """
        msgs = []
        ok = True
        for a in range(self.K):
            allowed = self.admissible_polynomial_charges(a)
            if self.polynomial_charges[a] not in allowed:
                ok = False
                msgs.append(
                    "polynomial %d has declared charge %d, but no monomial of "
                    "multidegree %s carries it; admissible charges are %s"
                    % (a, self.polynomial_charges[a],
                       list(self.degrees[a]), allowed))
        return ok, msgs

    def looks_free(self, probes=None):
        r"""
        The necessary condition for a free action, tested on several bundles.

        If Gamma acts freely then every g != e has no fixed points, so the
        holomorphic Lefschetz number L(g) vanishes and the equivariant index
        is a multiple of the regular representation -- the charges are
        equidistributed. Failure of that is a proof the action is *not* free.
        Success is evidence, not proof: the condition is necessary only.

        A passing result depends on the probe set, and the default was
        originally too narrow to notice it. An action trivial on two of the
        four tetraquadric factors has a two-dimensional fixed locus and is
        plainly not free, yet it passed a handful of hand-picked probes; over
        the box below it fails 180 of 625. The genuinely free action passes
        all 625. So the default sweeps a small box rather than a short list,
        and the docstring says what a pass is worth.

        Returns ``(looks_free, table)``, the table holding the failures when
        there are any and the first few probes otherwise.
        """
        if probes is None:
            m = len(self.dims)
            probes = [list(k) for k in
                      itertools.product((-2, -1, 0, 1, 2), repeat=m)]
        failures = []
        sample = []
        for k in probes:
            ch = self.euler(k)
            reg, mult = is_regular_multiple(ch)
            if reg:
                if len(sample) < 5:
                    sample.append((list(k), ch, True, mult))
            else:
                failures.append((list(k), ch, False, mult))
        return (not failures), (failures if failures else sample)

    def forced_fixed_points(self):
        r"""
        Ambient fixed points that the polynomial charges force onto X.

        A diagonal action fixes the coordinate points of the ambient -- those
        with a single non-zero homogeneous coordinate in each factor. At such
        a point every monomial vanishes except the pure power of the chosen
        coordinates. If that monomial's charge differs from the declared
        charge of a defining polynomial, it cannot appear in that polynomial,
        so the polynomial vanishes there and the fixed point *lies on X*. The
        action is then certainly not free.

        This is a sufficient condition for non-freeness, reached from the
        geometry, and it is completely independent of the Lefschetz argument
        behind :meth:`looks_free`, which is reached from an index. The two
        must agree, and ``tests/test_equivariant.py`` requires it.

        Returns the list of forced points, each as a tuple of one coordinate
        index per ambient factor.
        """
        out = []
        ranges = [range(int(d) + 1) for d in self.dims]
        for choice in itertools.product(*ranges):
            on_X = True
            for a in range(self.K):
                c = sum(int(self.degrees[a][i]) * self.weights[i][choice[i]]
                        for i in range(len(self.dims))) % self.n
                if c == self.polynomial_charges[a]:
                    on_X = False       # this polynomial need not vanish here
                    break
            if on_X:
                out.append(tuple(choice))
        return out

    def twist(self, character):
        """A different equivariant structure, differing by the given character.

        Returns a function ``k -> euler(k, character)``. Kept as a method so
        that the n available structures on a line bundle are visible as such
        rather than implicit in an integer argument.
        """
        t = int(character) % self.n

        def structure(kvec):
            return self.euler(kvec, twist=t)

        structure.character = t
        return structure


# ---------------------------------------------------------------------------
# representation-theoretic helpers
# ---------------------------------------------------------------------------

def regular_representation(n, multiple=1):
    """The regular representation of Z_n, or a multiple of it."""
    return [int(multiple)] * int(n)


def is_regular_multiple(character):
    """Whether a character is an integer multiple of the regular representation.

    Returns ``(is_multiple, multiple)``. Equidistribution of the charges is
    equivalent to the vanishing of every Lefschetz number L(g) for g != e,
    which is what a free action forces.
    """
    ch = list(character)
    if not ch:
        return True, 0
    first = ch[0]
    return all(c == first for c in ch), first


def bundle_index_character(action, charges, structures=None):
    r"""
    The character-valued index of a sum of line bundles.

    Parameters
    ----------
    action : CyclicAction
    charges : list of charge vectors
        The summands, as in :class:`pyCICY.bundles.LineBundleSum`.
    structures : list of int, optional
        One equivariant structure per summand, as a character of Gamma.
        Defaults to the canonical one on each. **This is the choice that the
        topology does not make**; :func:`enumerate_structures` lists the
        possibilities.

    Returns the summed character, whose total is ``ind(V)``.
    """
    if structures is None:
        structures = [0] * len(charges)
    if len(structures) != len(charges):
        raise ValueError("one equivariant structure per summand is needed")
    out = [0] * action.n
    for k, t in zip(charges, structures):
        ch = action.euler(k, twist=t)
        for c in range(action.n):
            out[c] += ch[c]
    return out


def enumerate_structures(action, charges, descend_only=True):
    r"""
    The equivariant structures available on a sum of line bundles.

    Each summand admits ``n`` of them, so a rank-r sum admits ``n^r``. With
    ``descend_only`` the list is restricted to those whose total twist
    vanishes, which is the condition for the induced structure on
    ``Lambda^r V = O_X`` to be the trivial one -- necessary for the bundle to
    descend to the quotient with an SU(r) structure group intact.

    Returns the list of tuples. For a rank-5 sum with Gamma = Z_2 there are
    32 structures and 16 that descend, which is the size of the choice
    :mod:`pyCICY.breaking` cannot make for itself.
    """
    n = action.n
    r = len(charges)
    out = []
    for t in itertools.product(range(n), repeat=r):
        if descend_only and sum(t) % n != 0:
            continue
        out.append(t)
    return out


def gamma_charges(character, multiplicity_sign=1):
    r"""
    Convert an index character into the charge list :func:`pyCICY.breaking.project` wants.

    ``breaking.project`` takes a list of Gamma-charges, one per multiplet.
    An index gives *net* multiplicities, which may be negative: a negative
    entry means the anti-multiplet dominates in that charge sector. This
    helper expands the positive part into a list and reports the negative part
    separately rather than discarding it, since the sign is the chirality and
    losing it would silently turn anti-generations into generations.

    Returns ``(charges, negative_part)``.
    """
    charges = []
    negative = {}
    for c, m in enumerate(character):
        m = int(m) * multiplicity_sign
        if m > 0:
            charges.extend([c] * m)
        elif m < 0:
            negative[c] = m
    return charges, negative


# ---------------------------------------------------------------------------
# a worked action
# ---------------------------------------------------------------------------

def TETRAQUADRIC_Z2():
    r"""
    A free Z_2 on the tetraquadric, the manifold :mod:`pyCICY.bundles` scans.

    The generator acts on each of the four P^1 factors by
    [x_0 : x_1] -> [x_0 : -x_1], and the defining quartic is taken to have
    charge 0. The ambient fixed locus is the 16 points with each coordinate
    either [1:0] or [0:1]; a charge-0 quartic contains the monomial
    x_{10}^2 x_{20}^2 x_{30}^2 x_{40}^2, which is non-zero at every one of
    them, so a generic such quartic misses the fixed locus entirely and the
    action on X is free.

    |Gamma| = 2 is the order that :func:`pyCICY.breaking.minimal_order`
    identifies as the smallest admitting a Wilson line that breaks SU(5) to
    the Standard Model, and the order the three-generation models of
    :func:`pyCICY.bundles.scan` require. The three parts of the construction
    meet here.
    """
    conf = [[1, 2], [1, 2], [1, 2], [1, 2]]
    return CyclicAction(conf, 2, [[0, 1]] * 4, [0])
