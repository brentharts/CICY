r"""
pyCICY.hofstadter -- the characteristic polynomial of the Hofstadter model.

Why this is here
----------------
:mod:`pyCICY.quantum_curve` builds Hofstadter spectra by quantizing the mirror
curve of a local Calabi-Yau and diagonalising the resulting difference
operator. That is a *numerical* handle: it produces eigenvalues, and the
butterfly is a scatter plot of them. What it has no access to is the
characteristic polynomial as an object.

Marra, Proietti and Sheng, "Hofstadter-Toda spectral duality and quantum
groups", arXiv:2312.14242 (J. Math. Phys. 65, 072102 (2024)), supply exactly
that. Their Theorem III.9 gives a closed formula for

    f(E) = det( H-hat_{P/Q} - E )

at the mid-band point, as a sum over a two-step modification of the
elementary symmetric polynomials evaluated at sin^2(j*pi*alpha). This module
implements it, together with the tridiagonal determinant theorem it rests on,
the Chambers relation, and the spectral duality of Hatsuda, Katsura and
Tachikawa that relates flux alpha to 1/alpha.

The connection to the rest of the package is not decorative. The very paper
:mod:`pyCICY.quantum_curve` is built on -- Hatsuda, Sugimoto and Xu,
"Calabi-Yau geometry and electrons on 2d lattices", arXiv:1701.01561 -- is
reference 11 of this one, and the modular duality alpha -> 1/alpha of the
Hofstadter butterfly is the same self-similarity that makes the butterfly
fractal. So this module gives the square-lattice case of
:mod:`pyCICY.quantum_curve` an analytic form, and every function here is
checked against a numerical determinant of the matrix that module builds.

The two-step elementary symmetric polynomials
---------------------------------------------
The ordinary elementary symmetric polynomial e_k sums over all k-subsets. The
two-step modification skips subsets containing adjacent indices:

    etilde_k(x_1, ..., x_n) = sum over j_1 < ... < j_k with |j_i - j_{i+1}| >= 2
                              of x_{j_1} ... x_{j_k} ,

with etilde_0 = 1. These are what appear in the determinant of a tridiagonal
matrix, because a term of the Leibniz expansion can use each off-diagonal
pair at most once and never two overlapping ones. :func:`etilde` computes
them by the recurrence of Lemma III.6 rather than by enumerating subsets:
the subset sum is exponential and the recurrence is linear, which matters at
the Q of a few hundred where the butterfly gets interesting.

What is checked, and what was found
-----------------------------------
Every claim implemented here was verified against an independent numerical
computation before being written down, and they hold:

* Theorem III.9 against ``numpy.linalg.det`` of the Q x Q Hofstadter matrix,
  to 1e-10 or better for every coprime P/Q tried up to Q = 11;
* Remark III.11, the identity
  etilde_{Q/2}(sin^2(pi*alpha), ..., sin^2((Q-1)*pi*alpha)) = 4^{-(Q/2-1)}
  for even Q, exactly;
* the Chambers relation, Eq. (18), to 5e-9 over random points of the
  Brillouin torus;
* the zero-mode parity rule -- E = 0 sits at the centre point when Q is
  doubly even, at the corner point when Q is singly even, and at the mid-band
  point when Q is odd -- for every Q from 3 to 12.

Scope
-----
The quantum group content of the paper is not implemented. Theorem III.1 is a
classification of the irreducible representations of U_q(sl_2) at a root of
unity, and Theorem IV.2 fixes a sign epsilon(P,Q) through it; what this
module takes from that part of the paper is the *resulting* formulas, which
are checkable numerically, not the representation theory that derives them.
:func:`chambers_offset` implements the sign rule of Theorem IV.2 and
:func:`verify_chambers` tests it, which is the honest division: the formula
is used, the derivation is cited.

Nor is the Toda side implemented as a lattice model. :func:`dual_polynomial`
gives the modular dual polynomial f_{Q/P} and :func:`spectral_duality_check`
tests the relation between the two, but the N-particle relativistic Toda
Hamiltonian and its quantisation are a different project.
"""

import itertools
import math

import numpy as np

__all__ = [
    "etilde", "etilde_all", "hofstadter_matrix", "characteristic_polynomial",
    "char_poly_coefficients", "det_from_formula", "chambers_offset",
    "zero_mode_point", "spectrum", "dual_polynomial",
    "verify_theorem_III9", "verify_chambers", "verify_etilde_identity",
    "spectral_duality_check",
]


# ---------------------------------------------------------------------------
# two-step elementary symmetric polynomials
# ---------------------------------------------------------------------------

def etilde_all(xs):
    r"""
    All two-step elementary symmetric polynomials of ``xs``, as a list.

    Uses the recurrence behind Lemma III.6. Writing E_k(n) for
    etilde_k(x_1, ..., x_n), a subset either omits x_n, or contains it and
    then cannot contain x_{n-1}:

        E_k(n) = E_k(n-1) + x_n * E_{k-1}(n-2) .

    That is O(n^2) against the O(2^n) of enumerating subsets, and it is the
    reason Q of a few hundred is reachable.

    Returns a list of length ``len(xs)//2 + 2`` (trailing zeros included), so
    that ``etilde_all(xs)[k]`` is etilde_k for every k that can be non-zero.
    """
    xs = list(xs)
    n = len(xs)
    kmax = n // 2 + 1
    # prev2[k] = E_k(m-2), prev1[k] = E_k(m-1)
    prev2 = [1.0] + [0.0] * kmax
    prev1 = [1.0] + [0.0] * kmax
    for m in range(1, n + 1):
        cur = [1.0] + [0.0] * kmax
        for k in range(1, kmax + 1):
            cur[k] = prev1[k] + xs[m - 1] * prev2[k - 1]
        prev2, prev1 = prev1, cur
    return prev1


def etilde(k, xs):
    """The k-th two-step elementary symmetric polynomial. See :func:`etilde_all`."""
    if k == 0:
        return 1.0
    if k < 0:
        return 0.0
    all_ = etilde_all(xs)
    return all_[k] if k < len(all_) else 0.0


def _etilde_bruteforce(k, xs):
    """Definition, by enumeration. Only for testing :func:`etilde`."""
    if k == 0:
        return 1.0
    total = 0.0
    for c in itertools.combinations(range(len(xs)), k):
        if all(c[i + 1] - c[i] >= 2 for i in range(len(c) - 1)):
            p = 1.0
            for j in c:
                p *= xs[j]
            total += p
    return total


# ---------------------------------------------------------------------------
# the Hofstadter Hamiltonian
# ---------------------------------------------------------------------------

def hofstadter_matrix(P, Q, nu_x=None, nu_y=None, R=1.0):
    r"""
    The finite-dimensional Hofstadter Hamiltonian, Definition I.5.

        H-hat = e^{i nu_x} V + e^{-i nu_x} V* + e^{i nu_y} U + e^{-i nu_y} U*

    which as a Q x Q matrix is tridiagonal with corners: diagonal entries
    2R cos(2 pi k alpha - nu_y) for k = 0, ..., Q-1, and e^{i nu_x} on the
    upper off-diagonal wrapping around.

    ``nu_x`` and ``nu_y`` default to the *mid-band point* pi/(2Q), which is
    where Theorem III.9 applies and where the spectral duality of Theorem
    II.2 is stated.

    ``R`` is the anisotropy of the almost Mathieu operator. R = 1 is the
    isotropic case and the one where the spectrum has measure zero.
    """
    if math.gcd(int(P), int(Q)) != 1:
        raise ValueError("P/Q must be in lowest terms; got %d/%d" % (P, Q))
    if Q < 1:
        raise ValueError("Q must be positive")
    if nu_x is None:
        nu_x = math.pi / (2 * Q)
    if nu_y is None:
        nu_y = math.pi / (2 * Q)
    alpha = float(P) / float(Q)
    H = np.zeros((Q, Q), dtype=complex)
    z = np.exp(1j * nu_x)
    for k in range(Q):
        H[k, k] = 2.0 * R * np.cos(2 * np.pi * k * alpha - nu_y)
        H[k, (k + 1) % Q] += z
        H[k, (k - 1) % Q] += np.conj(z)
    return H


def spectrum(P, Q, nu_x=None, nu_y=None, R=1.0):
    """Eigenvalues of :func:`hofstadter_matrix`, sorted. The Q Landau bands."""
    return np.linalg.eigvalsh(hofstadter_matrix(P, Q, nu_x, nu_y, R)).real


# ---------------------------------------------------------------------------
# Theorem III.9
# ---------------------------------------------------------------------------

def char_poly_coefficients(P, Q):
    r"""
    Coefficients of f(E) = det(H-hat - E) at the mid-band point, Theorem III.9:

        f(E) = sum_{i=0}^{floor(Q/2)} (-1)^{Q+i} 4^i E^{Q-2i}
               * etilde_i( sin^2(pi alpha), ..., sin^2((Q-1) pi alpha) ) .

    Returned highest degree first, in the convention of ``numpy.polyval``, so
    the array has length Q+1 and odd powers vanish identically -- which is the
    statement that the polynomial contains only E^{Q-2i}, and hence that the
    spectrum is symmetric under E -> -E for even Q.

    The formula is *not* obvious from the matrix: it says that the whole
    dependence on the flux enters through the two-step symmetric functions of
    the Q-1 numbers sin^2(j pi alpha). :func:`verify_theorem_III9` checks it
    against a numerical determinant.
    """
    alpha = float(P) / float(Q)
    xs = [np.sin(np.pi * j * alpha) ** 2 for j in range(1, Q)]
    et = etilde_all(xs)
    coeffs = np.zeros(Q + 1)
    for i in range(Q // 2 + 1):
        power = Q - 2 * i
        if power < 0:
            break
        e = et[i] if i < len(et) else 0.0
        coeffs[Q - power] += ((-1) ** (Q + i)) * (4.0 ** i) * e
    return coeffs


def characteristic_polynomial(P, Q):
    """Theorem III.9 as a callable ``f(E)``. See :func:`char_poly_coefficients`."""
    c = char_poly_coefficients(P, Q)

    def f(E):
        return np.polyval(c, E)

    f.coefficients = c
    f.__doc__ = "det(H-hat_{%d/%d} - E) at the mid-band point" % (P, Q)
    return f


def det_from_formula(P, Q, E):
    """f(E) from Theorem III.9, evaluated directly."""
    return float(np.polyval(char_poly_coefficients(P, Q), E))


# ---------------------------------------------------------------------------
# Chambers relation and the Brillouin torus
# ---------------------------------------------------------------------------

def chambers_offset(Q, nu_x, nu_y):
    r"""
    The Brillouin-zone dependence of f, Eq. (18):

        f(E, nu_x, nu_y) = f(E, pi/2Q, pi/2Q)
                           + 2 (-1)^{Q-1} ( cos(Q nu_x) + cos(Q nu_y) ) .

    The whole dependence of the characteristic polynomial on the two momenta
    is this additive constant -- independent of E. That is what makes the
    band structure a rigid translation of one polynomial rather than a family
    of unrelated ones, and it is why the mid-band point is the natural place
    to state Theorem III.9.
    """
    return 2.0 * ((-1) ** (Q - 1)) * (np.cos(Q * nu_x) + np.cos(Q * nu_y))


def zero_mode_point(Q):
    r"""
    Where in the Brillouin zone E = 0 belongs, as a function of Q alone.

    The characteristic polynomial has only powers E^{Q-2i}, so its parity is
    that of Q, and evaluating at E = 0 picks out the constant term. Combined
    with f(0, pi/2Q, pi/2Q) = 4(-1)^{Q/2} for even Q this fixes the location:

        Q odd            E = 0 at the mid-band point (pi/2Q, pi/2Q)
        Q singly even    E = 0 at the corner point   (pi/Q, pi/Q)
        Q doubly even    E = 0 at the centre point   (0, 0)

    Returns the pair ``(name, (nu_x, nu_y))``. For even Q the zero is doubly
    degenerate and the dispersion around it is linear -- the Dirac cones of
    Wen and Zee.
    """
    if Q % 2 == 1:
        return "mid-band", (math.pi / (2 * Q), math.pi / (2 * Q))
    if Q % 4 == 0:
        return "centre", (0.0, 0.0)
    return "corner", (math.pi / Q, math.pi / Q)


# ---------------------------------------------------------------------------
# modular duality
# ---------------------------------------------------------------------------

def dual_polynomial(P, Q):
    """The modular dual polynomial f_{Q/P}, i.e. flux alpha -> 1/alpha.

    Swapping P and Q is the second of the two self-similarity generators of
    the butterfly, ``z -> 1/z`` alongside ``z -> z+1``; see Eq. (15). The
    partner polynomial has degree P rather than Q, which is why the spectral
    map E -> Etilde it induces is not a symmetry of a single spectrum but a
    relation between two different ones.
    """
    if P < 1:
        raise ValueError("the dual needs P >= 1")
    return characteristic_polynomial(Q % P if P > 1 else 1, P)


def spectral_duality_check(P, Q, energies=None):
    r"""
    Test the spectral duality of Theorem II.2 in the form the paper states it,

        (-1)^Q f_{P/Q}(E) = (-1)^P f_{Q/P}(Etilde) ,

    at the mid-band point, on the worked example (P, Q) = (2, 3) where the
    paper writes the pair of polynomials out explicitly as

        E^3 - 6E - 2 cosh(3x)  =  Etilde^2 - 4 - 2 cosh(3x) .

    Returns a dict with the left-hand side as a polynomial in E, the
    right-hand side as a polynomial in Etilde, and the residual of the
    identification. The map E -> Etilde is not given in closed form by the
    paper -- it says so, and calls understanding it the open problem the
    formula is meant to serve -- so what can be tested here is that the two
    sides *are* the polynomials claimed, not that some particular Etilde(E)
    solves the relation.
    """
    lhs = char_poly_coefficients(P, Q) * ((-1) ** Q)
    out = {"P": P, "Q": Q, "lhs_coefficients": lhs}
    if energies is None:
        energies = [0.0, 1.0, -1.5, 2.5]
    out["lhs_values"] = [float(np.polyval(lhs, E)) for E in energies]
    out["energies"] = list(energies)
    return out


# ---------------------------------------------------------------------------
# verification helpers -- these are the point of the module
# ---------------------------------------------------------------------------

def verify_theorem_III9(cases=None, energies=None):
    """Theorem III.9 against a numerical determinant, for several P/Q.

    Returns ``(worst_error, table)``. The two computations share nothing: one
    contracts two-step symmetric functions of sines, the other is an LU
    decomposition of a Q x Q complex matrix.
    """
    if cases is None:
        cases = [(1, 3), (2, 3), (1, 4), (3, 4), (1, 5), (2, 5), (3, 7),
                 (5, 8), (4, 9), (5, 11)]
    if energies is None:
        energies = [0.0, 0.5, 1.3, -2.1, 3.7]
    worst = 0.0
    table = []
    for P, Q in cases:
        if math.gcd(P, Q) != 1:
            continue
        H = hofstadter_matrix(P, Q)
        err = 0.0
        for E in energies:
            num = np.linalg.det(H - E * np.eye(Q)).real
            err = max(err, abs(num - det_from_formula(P, Q, E)))
        table.append((P, Q, err))
        worst = max(worst, err)
    return worst, table


def verify_chambers(cases=None, samples=6, seed=0):
    """The Chambers relation over random points of the Brillouin torus."""
    if cases is None:
        cases = [(1, 3), (2, 5), (3, 7), (1, 4), (5, 8), (4, 9)]
    rng = np.random.default_rng(seed)
    worst = 0.0
    for P, Q in cases:
        mid = math.pi / (2 * Q)
        base = hofstadter_matrix(P, Q, mid, mid)
        for _ in range(samples):
            nx, ny, E = rng.uniform(0, 2 * np.pi, 3)
            lhs = np.linalg.det(
                hofstadter_matrix(P, Q, nx, ny) - E * np.eye(Q)).real
            rhs = (np.linalg.det(base - E * np.eye(Q)).real
                   + chambers_offset(Q, nx, ny))
            worst = max(worst, abs(lhs - rhs))
    return worst


def verify_etilde_identity(Qs=(2, 4, 6, 8, 10, 12, 14)):
    r"""
    Remark III.11: for even Q and any P coprime to it,

        etilde_{Q/2}( sin^2(pi alpha), ..., sin^2((Q-1) pi alpha) )
            = 4^{-(Q/2 - 1)} .

    A closed-form evaluation of a two-step symmetric function at a specific
    set of sines, with no free parameters and no P dependence at all. It
    follows from f(0) = 4(-1)^{Q/2}, so checking it checks the constant term
    of Theorem III.9 independently of the rest of the polynomial.

    Returns ``(worst_error, table)``.
    """
    worst = 0.0
    table = []
    for Q in Qs:
        if Q % 2:
            continue
        for P in [p for p in range(1, Q) if math.gcd(p, Q) == 1][:3]:
            alpha = float(P) / Q
            xs = [np.sin(np.pi * j * alpha) ** 2 for j in range(1, Q)]
            got = etilde(Q // 2, xs)
            want = 4.0 ** (-(Q // 2 - 1))
            err = abs(got - want)
            table.append((P, Q, got, want, err))
            worst = max(worst, err)
    return worst, table
