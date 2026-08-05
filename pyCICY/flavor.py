r"""
pyCICY.flavor -- the 24-cell flavour construction, implemented as stated.

What this is
------------
An implementation of the Standard Model content of

    A. F. Ali, "Quantum Spacetime Imprints: The 24-Cell, Standard Model
    Symmetry and its Flavor Mixing", arXiv:2511.10685,

whose geometric backbone -- the 24-cell as a reflexive polytope, its polar
dual, and its tetrahedral subsets -- lives in :mod:`pyCICY.polytope`. This
module carries the physics: the hypercharge functional, the projection to a
tetrahedron, the resulting A_4 neutrino mass matrix and tribimaximal mixing,
and the estimates of theta_13 and the Cabibbo angle.

Everything here is implemented **as the paper states it**, with the paper's
own numbers, and then checked. Where a stated result does not follow from the
stated inputs, the function computes what the inputs give and says so in its
docstring and its return value, rather than quietly substituting a parameter
that makes the number come out right. Three such cases are recorded below.
This is the same policy :mod:`pyCICY.apolynomial` follows when it reports the
leftover factor in the classical limit of the AJ recursion instead of
suppressing it.

What checks out exactly
-----------------------
*The hypercharge functional.* For each generation g the paper gives a vector
h_Y^(g) and an offset eps_g such that Y_g(alpha) = <alpha, h_Y^(g)> + eps_g
reproduces the five Standard Model hypercharges on five vertices of the
24-cell. In exact rational arithmetic all fifteen values are reproduced, on
fifteen distinct vertices of the sixteen in V_2. The unused one is
(-1, 1, -1, 1). The hypercharge trace vanishes generation by generation, so
the U(1)_Y anomaly cancels.

*The tetrahedral geometry.* Projecting a regular tetrahedron of the 24-cell
to the three-dimensional affine hull and normalising gives a Gram matrix with
every off-diagonal entry exactly -1/3. The resulting matrix J has eigenvalues
{1/3, 4/3, 4/3}, and U_TBM diagonalises it, giving theta_12 = 35.26 degrees,
theta_23 = 45 degrees, theta_13 = 0 -- tribimaximal mixing.

Three things that do not follow, and are reported as such
---------------------------------------------------------
*The hypercharge functional is an interpolation, not a prediction, for
generations 1 and 2.* The functional has 4 components of h_Y plus eps_g, so
five free parameters against five target hypercharges. :func:`fit_rank`
computes the rank of the 5 x 5 system and finds it full for all three
generations, so a unique solution exists for *any* choice of five affinely
independent vertices, whatever the targets. The Standard Model values are
being fitted, not derived.

*Generation 3 is the exception, and there is a reason.* The paper sets
eps_3 = 0, leaving four parameters for five constraints -- and the system is
nevertheless consistent, which is a genuine property of that vertex choice.
:func:`epsilon_zero_analysis` locates it precisely. The five generation-3
vertices carry exactly one linear dependency, with null vector
c = (-1, 1, -1, -2, 1), and consistency is exactly the vanishing of c . Y.
Imposing only the three Yukawa-invariance relations -- Y_L = Y_eR + Y_H,
Y_uR = Y_q + Y_H, Y_dR = Y_q - Y_H, all with one Higgs hypercharge -- reduces
that combination to -4 Y_H - 2 Y_eR, which vanishes precisely because
Y_eR = -1 and Y_H = 1/2. So the coincidence is real but is a statement about
the Standard Model hypercharges, not about the 24-cell. :func:`epsilon_zero_census`
scans all 437760 (five-subset, assignment) pairs and finds 14592 of them,
3.33%, admit an eps = 0 solution; the generation-3 choice is one of a family
of about one in thirty, not a unique geometric selection.

*The Minimal Distortion Principle has nothing to minimise.* The paper defines
D(Pi) = sum_{i<j} | ||Pi(v_i) - Pi(v_j)|| - ||v_i - v_j|| | and takes eta to
measure the residual distortion of the optimal projection, then uses
eta ~ 0.02 to drive both theta_13 and the Cabibbo angle. But four points
always span an affine subspace of dimension at most three, so the orthogonal
projection onto that hull is an *isometry* and D = 0 identically.
:func:`mdp_distortion` returns it, and gets 2e-15. Whatever eta is, it is not
the distortion of this projection; the tetrahedron is embedded exactly.

*The Cabibbo arithmetic.* The paper writes

    theta_C ~ kappa_q eta ||v_1 - v_2|| ~ sqrt(2/3) x (0.02-0.03) x 2
            ~ 0.22-0.26 ~ 12.6-15 degrees.

Evaluating the middle expression gives 0.033-0.049, i.e. 1.9-2.8 degrees. The
stated 0.22 needs eta ~ 0.135, some six times the eta ~ 0.022 the same paper
requires for theta_13. :func:`cabibbo_angle` evaluates the formula as written
and returns both the value and the eta that would be needed for the quoted
answer, so the two can be compared rather than one being assumed. There is no
v2 of the paper as of this writing.

Scope
-----
The F_4 branching, the spinfoam edge lengths, the flux stabilisation and the
collider estimates are not implemented. Neither is the T' representation
theory: :func:`cabibbo_angle` evaluates the paper's scalar estimate, which is
where its number comes from, and does not construct the binary tetrahedral
group or its doublets. The full chi-squared fit of the neutrino sector that
the paper describes as future work is likewise not attempted; what is here is
the ideal limit and the first-order shift away from it.
"""

import itertools
import math

import numpy as np

__all__ = [
    "SM_HYPERCHARGES", "GENERATION_VERTICES", "HYPERCHARGE_DATA",
    "hypercharge", "verify_hypercharges", "anomaly_trace",
    "fit_rank", "epsilon_zero_analysis", "epsilon_zero_census",
    "tetrahedron_projection", "mdp_distortion", "gram_matrix",
    "J_matrix", "U_TBM", "tbm_angles", "mixing_angles",
    "neutrino_mass_matrix", "theta13_from_strain", "cabibbo_angle",
]

from fractions import Fraction as _F


# ---------------------------------------------------------------------------
# hypercharges
# ---------------------------------------------------------------------------

#: Standard Model hypercharges, and the multiplicity of each species.
SM_HYPERCHARGES = {
    "lL": (_F(-1, 2), 2),     # left-handed lepton doublet
    "qL": (_F(1, 6), 6),      # left-handed quark doublet, 2 x 3 colours
    "eR": (_F(-1), 1),        # right-handed charged lepton
    "uR": (_F(2, 3), 3),      # right-handed up-type quark
    "dR": (_F(-1, 3), 3),     # right-handed down-type quark
}

_ORDER = ("lL", "qL", "eR", "uR", "dR")

#: Vertex assignments, as the integer tuples a with alpha = a/2.
GENERATION_VERTICES = {
    1: {"lL": (1, 1, 1, 1), "qL": (1, 1, -1, -1), "eR": (-1, -1, -1, -1),
        "uR": (1, -1, 1, -1), "dR": (1, -1, -1, 1)},
    2: {"lL": (1, 1, 1, -1), "qL": (1, 1, -1, 1), "eR": (-1, -1, 1, 1),
        "uR": (1, -1, 1, 1), "dR": (-1, 1, 1, 1)},
    3: {"lL": (1, -1, -1, -1), "qL": (-1, 1, -1, -1), "eR": (-1, -1, -1, 1),
        "uR": (-1, 1, 1, -1), "dR": (-1, -1, 1, -1)},
}

#: ``(h_Y, epsilon)`` per generation, with kappa_g absorbed into both.
HYPERCHARGE_DATA = {
    1: ((_F(3, 2), _F(-1, 3), _F(1, 6), _F(-5, 6)), _F(-3, 4)),
    2: ((_F(5, 3), _F(2, 3), _F(7, 6), _F(11, 6)), _F(-4, 3)),
    3: ((_F(1, 3), _F(1), _F(1, 2), _F(-1, 6)), _F(0)),
}


def hypercharge(generation, vertex):
    r"""
    Y_g(alpha) = <alpha, h_Y^(g)> + eps_g, in exact rational arithmetic.

    ``vertex`` is the integer tuple a, with alpha = a/2 the actual vertex of
    the 24-cell in the normalisation of section 2.1 (squared norm 1). The
    normalisation constant kappa_g of the paper cancels between h_Y and eps
    and is absorbed here; it plays no role in any value.
    """
    h, eps = HYPERCHARGE_DATA[generation]
    a = [_F(int(x), 2) for x in vertex]
    return sum(a[i] * h[i] for i in range(4)) + eps


def verify_hypercharges():
    """Every hypercharge of every generation, checked exactly.

    Returns ``(all_ok, table)`` with one row per (generation, species) giving
    the computed and target values. Exact rationals throughout, so an
    agreement here is an identity and not a tolerance.
    """
    table = []
    ok = True
    for g in (1, 2, 3):
        for name in _ORDER:
            got = hypercharge(g, GENERATION_VERTICES[g][name])
            want = SM_HYPERCHARGES[name][0]
            table.append((g, name, got, want, got == want))
            ok = ok and got == want
    return ok, table


def anomaly_trace():
    r"""
    Tr(Y) = sum over the fermions of a generation of multiplicity times Y.

    Vanishing of this trace is the condition for the U(1)_Y gravitational
    anomaly to cancel. It is a statement about the Standard Model hypercharges
    alone and holds for every generation identically, since the three
    generations carry the same hypercharges on different vertices. Returned as
    an exact ``Fraction``.
    """
    return sum(mult * Y for Y, mult in SM_HYPERCHARGES.values())


def distinct_vertices():
    """The set of vertices used across all three generations.

    Fifteen of the sixteen elements of V_2, the unused one being
    (-1, 1, -1, 1). Returned as ``(vertices, unused)``.
    """
    used = set()
    for g in (1, 2, 3):
        used.update(GENERATION_VERTICES[g].values())
    allv = set(itertools.product([1, -1], repeat=4))
    return used, allv - used


# ---------------------------------------------------------------------------
# is it a fit?
# ---------------------------------------------------------------------------

def fit_rank(generation):
    r"""
    The rank of the linear system that determines h_Y and eps.

    The functional is linear in the five unknowns (h_1, ..., h_4, eps), and
    there are five target hypercharges, so the question of whether the
    Standard Model values are *derived* or merely *accommodated* is the
    question of whether that 5 x 5 system is invertible. It is, for all three
    generations. A full-rank system has a unique solution for any right-hand
    side whatever, so the hypercharges are being fitted.

    Returns a dict with ``rank``, ``unique`` and ``solution``.
    """
    import sympy as sp

    V = GENERATION_VERTICES[generation]
    A = sp.Matrix([[sp.Rational(V[n][i], 2) for i in range(4)] + [1]
                   for n in _ORDER])
    b = sp.Matrix([sp.Rational(SM_HYPERCHARGES[n][0].numerator,
                               SM_HYPERCHARGES[n][0].denominator)
                   for n in _ORDER])
    rank = A.rank()
    out = {"rank": int(rank), "unique": rank == 5, "solution": None}
    if rank == 5:
        out["solution"] = list(A.solve(b))
    return out


def epsilon_zero_analysis(generation=3):
    r"""
    Why generation 3 can set eps = 0, and generations 1 and 2 cannot.

    With eps forced to zero the system is five equations in four unknowns and
    is generically inconsistent. It is consistent exactly when the unique
    linear dependency among the five vertices is matched by the same
    dependency among the five target hypercharges: if sum_i c_i alpha_i = 0
    then consistency requires sum_i c_i Y_i = 0.

    For generation 3 the null vector is c = (-1, 1, -1, -2, 1) in the order
    (lL, qL, eR, uR, dR), and c . Y = 0. Substituting only the Yukawa
    invariance relations Y_L = Y_eR + Y_H, Y_uR = Y_q + Y_H, Y_dR = Y_q - Y_H
    turns that combination into -4 Y_H - 2 Y_eR, so it vanishes because
    Y_eR = -1 and Y_H = 1/2. The coincidence is about the Standard Model
    hypercharge assignments, not about the geometry of the 24-cell.

    Returns a dict with ``consistent``, the null vector ``c``, the value of
    ``c_dot_Y``, and the solved ``h`` when one exists.
    """
    import sympy as sp

    V = GENERATION_VERTICES[generation]
    A = sp.Matrix([[sp.Rational(V[n][i], 2) for i in range(4)] for n in _ORDER])
    b = sp.Matrix([sp.Rational(SM_HYPERCHARGES[n][0].numerator,
                               SM_HYPERCHARGES[n][0].denominator)
                   for n in _ORDER])
    ns = A.T.nullspace()
    c = [sp.nsimplify(x) for x in ns[0]] if ns else None
    cdotY = sum(c[i] * b[i] for i in range(5)) if c else None
    aug = A.row_join(b)
    consistent = A.rank() == aug.rank()
    out = {"consistent": bool(consistent), "c": c, "c_dot_Y": cdotY,
           "rank": int(A.rank()), "rank_augmented": int(aug.rank()), "h": None}
    if consistent:
        out["h"] = list(A.solve_least_squares(b))
    return out


def epsilon_zero_census():
    r"""
    How special the generation-3 choice is, by exhaustion.

    Scans every five-element subset of the sixteen vertices of V_2 and every
    assignment of the five species to them, and counts those admitting an
    eps = 0 solution. Returns ``(hits, total, fraction)``.

    The answer is 14592 of 437760, or 3.33%. So the generation-3 property is
    genuine but common at the level of about one in thirty; it is not a
    geometric selection principle picking out a unique configuration.

    This takes a few seconds. It is a census, not something to call in a loop.
    """
    import sympy as sp

    V = [np.array(v) for v in itertools.product([1, -1], repeat=4)]
    tgt = [sp.Rational(SM_HYPERCHARGES[n][0].numerator,
                       SM_HYPERCHARGES[n][0].denominator) for n in _ORDER]
    hits = 0
    total = 0
    for sub in itertools.combinations(range(16), 5):
        M = sp.Matrix([list(V[i]) for i in sub])
        if M.rank() < 4:
            continue
        ns = M.T.nullspace()
        if not ns:
            continue
        c = ns[0]
        for perm in itertools.permutations(range(5)):
            total += 1
            if sum(c[k] * tgt[perm[k]] for k in range(5)) == 0:
                hits += 1
    return hits, total, (hits / total if total else 0.0)


# ---------------------------------------------------------------------------
# the tetrahedron, the projection, and TBM
# ---------------------------------------------------------------------------

#: The representative tetrahedron of section 3.1.
TETRAHEDRON = np.array([[1, 1, 0, 0], [1, -1, 0, 0],
                        [0, 0, 1, 1], [0, 0, 1, -1]], dtype=float)


def tetrahedron_projection(V=None, normalise=True):
    """Project four points of R^4 into the affine hull of their centroid.

    Centre, build an orthonormal basis of the three-dimensional span, and take
    coordinates. With ``normalise`` the results are put on the unit sphere,
    which is what the Gram matrix of -1/3 refers to.
    """
    V = TETRAHEDRON if V is None else np.asarray(V, dtype=float)
    W = V - V.mean(axis=0)
    Q, _ = np.linalg.qr(W[:3].T)
    P = W @ Q
    if normalise:
        P = P / np.linalg.norm(P, axis=1, keepdims=True)
    return P


def mdp_distortion(V=None):
    r"""
    The distortion functional D(Pi) = sum_{i<j} | ||Pi(v_i)-Pi(v_j)|| - ||v_i-v_j|| |.

    Returns its value at the orthogonal projection onto the affine hull.

    It is zero, to machine precision, and necessarily so: any four points span
    an affine subspace of dimension at most three, and orthogonal projection
    onto that subspace is an isometry of it. There is nothing for a Minimal
    Distortion Principle to minimise here, and in particular the parameter eta
    that the paper uses for theta_13 and the Cabibbo angle cannot be the
    residual distortion of this map.
    """
    V = TETRAHEDRON if V is None else np.asarray(V, dtype=float)
    P = tetrahedron_projection(V, normalise=False)
    return float(sum(abs(np.linalg.norm(P[i] - P[j])
                         - np.linalg.norm(V[i] - V[j]))
                     for i, j in itertools.combinations(range(len(V)), 2)))


def gram_matrix(V=None):
    """Gram matrix of the normalised projected tetrahedron. Off-diagonals -1/3."""
    P = tetrahedron_projection(V)
    return P @ P.T


def J_matrix():
    """The ideal tetrahedral matrix J: 1 on the diagonal, -1/3 off it.

    This is the 3 x 3 block the neutrino mass matrix is built from, i.e. the
    Gram matrix of three of the four tetrahedron directions.
    """
    J = np.full((3, 3), -1.0 / 3.0)
    np.fill_diagonal(J, 1.0)
    return J


def U_TBM():
    """The tribimaximal mixing matrix, in the paper's ordering."""
    return np.array([
        [math.sqrt(2 / 3), math.sqrt(1 / 3), 0.0],
        [-math.sqrt(1 / 6), math.sqrt(1 / 3), math.sqrt(1 / 2)],
        [-math.sqrt(1 / 6), math.sqrt(1 / 3), -math.sqrt(1 / 2)],
    ])


def tbm_angles():
    """The tribimaximal mixing angles in degrees, from :func:`U_TBM`.

    theta_12 = arcsin(1/sqrt3) = 35.26 degrees, theta_23 = 45, theta_13 = 0.
    The measured theta_12 is about 33.4 degrees and theta_13 about 8.5, so TBM
    is a starting point that needs a perturbation, which is what eta supplies.
    """
    return mixing_angles(U_TBM())


def mixing_angles(U):
    """Extract (theta_12, theta_13, theta_23) in degrees from a mixing matrix.

    Standard PMNS parametrisation: s13 = |U_13|, t12 = |U_12|/|U_11|,
    t23 = |U_23|/|U_33|.
    """
    U = np.abs(np.asarray(U, dtype=float))
    th13 = math.degrees(math.asin(min(1.0, U[0, 2])))
    th12 = math.degrees(math.atan2(U[0, 1], U[0, 0]))
    th23 = math.degrees(math.atan2(U[1, 2], U[2, 2]))
    return th12, th13, th23


def neutrino_mass_matrix(alpha=1.0, eta=0.0, C=None):
    """M_nu = alpha J + eta C, the paper's Eq. (FullNuMass).

    ``C`` is a real symmetric traceless perturbation; the default is zero, so
    the default return is the ideal tetrahedral matrix and the mixing is
    exactly tribimaximal.
    """
    M = alpha * J_matrix()
    if eta and C is not None:
        C = np.asarray(C, dtype=float)
        if not np.allclose(C, C.T):
            raise ValueError("C must be symmetric")
        if abs(np.trace(C)) > 1e-12:
            raise ValueError("C must be traceless")
        M = M + eta * C
    return M


def theta13_from_strain(eps13, eta):
    r"""
    The reactor angle from an off-diagonal (1,3) strain, as the paper estimates it:

        theta_13 ~ |eps_13| / (3 sqrt(3) eta) ,

    in radians, from first-order perturbation theory about the degenerate TBM
    limit. Returns the angle in degrees.

    The paper's worked case is eta = 0.022 and eps_13 = 0.017, quoted as
    giving theta_13 = 8.5 degrees; this function evaluates the formula so that
    the two can be compared.
    """
    if eta == 0:
        raise ValueError("eta must be non-zero: the estimate divides by it")
    return math.degrees(abs(eps13) / (3 * math.sqrt(3) * eta))


def cabibbo_angle(eta=0.02, kappa_q=None, edge=2.0):
    r"""
    The Cabibbo angle from the paper's scalar estimate,

        tan(theta_C) ~ kappa_q eta ||v_1 - v_2|| ,

    with kappa_q = sqrt(2/3) and ||v_1 - v_2|| = 2 as the paper specifies.

    Returns a dict with ``tan_theta``, ``degrees``, and ``eta_for_quoted``,
    the value of eta that the paper's own quoted answer of tan(theta_C) ~ 0.22
    would require.

    **The stated arithmetic does not close.** The paper writes

        sqrt(2/3) x (0.02-0.03) x 2 ~ 0.22-0.26 ~ 12.6-15 degrees,

    but sqrt(2/3) x 0.02 x 2 = 0.0327 and sqrt(2/3) x 0.03 x 2 = 0.0490,
    i.e. 1.9 to 2.8 degrees. Reaching tan(theta_C) = 0.2250, the measured
    value, needs eta = 0.138 -- about six times the eta = 0.022 the same paper
    requires in :func:`theta13_from_strain` for the reactor angle. Since the
    paper describes eta as a universal distortion shared by the quark and
    lepton sectors, the two determinations are in tension by that factor.

    This function returns what the formula gives. The discrepancy is stated
    rather than absorbed, because absorbing it would mean silently using an
    eta the paper never writes.
    """
    if kappa_q is None:
        kappa_q = math.sqrt(2.0 / 3.0)
    t = kappa_q * eta * edge
    measured = 0.2250          # sin(theta_C), the Wolfenstein lambda
    return {"tan_theta": t,
            "degrees": math.degrees(math.atan(t)),
            "eta_for_quoted": measured / (kappa_q * edge),
            "measured_sin_theta_C": measured,
            "measured_degrees": math.degrees(math.asin(measured))}
