r"""
pyCICY.sixsphere -- the lattice layer of a complex structure on S^6, exactly.

In August 2026 Levent Alpoge posted a construction answering Hopf's 1947
question: a compact complex threefold X, fibred over P^1 by complex 2-tori,
with singular fibres over three points, which is diffeomorphic to the six
sphere. Philip Engel gave a pedagogical account, and Boris Alexeev a Lean
formalisation of the headline statement. The construction also contradicts a
published corollary of Campana, Demailly and Peternell [CDP20, Cor. 2.3], and
the paper locates the step at which the two accounts part.

What this module is, and is not
-------------------------------
It is not a check of the theorem. That X is diffeomorphic to S^6 rests on a
period family of holomorphic functions, a toric filling, two logarithmic
transforms, a van Kampen computation, Smale's h-cobordism theorem and the
vanishing of Theta_6 -- analysis and topology, which this package does not do.
The external formalisation (github.com/plby/HopfProblem, about 250k lines of
Lean 4 over Mathlib) is where that lives, and nothing here re-runs it.

What *is* exact is the finite algebra the whole construction is built on: two
integer 4x4 matrices of orders 3 and 4 whose product is unipotent, and
everything the paper derives from them by linear algebra over Z. That layer is
small, checkable, and load-bearing -- the fundamental group, the absence of a
polarisation, the unimodularity that makes the cusp fibre a single del Pezzo
surface, and the freeness of the logarithmic transforms are all decided in it.
This module computes it, and wherever a number admits a second route it takes
one:

  * the orders, determinants and the relation A1 A2 M0 = I;
  * the fixed sublattices, computed as saturated integer kernels, compared
    against the paper's generators eps, eps', delta-hat;
  * the coinvariants, from the Smith form of [(T1 - I) | (T2 - I)];
  * the invariant alternating forms, solved for rather than quoted -- one
    generator, Q0 -- and the symmetric form b(x, y) = Q0(x, N y), whose
    signature (1, 1) is why the family carries no polarisation;
  * the invariant bivectors, solved for separately, giving eta = u^w + 6 g^d
    with eta^2 = 12 vol; and the check that eta is the Hodge dual of Q0, so
    the two solves agree without sharing code;
  * B0, Lambda_tor = ker(M0 - I) = im(M0 - I), and |det B0| = 1;
  * the smallest invariant subgroup containing Lambda_tor, closed under the
    monodromy by lattice arithmetic, against ker gamma;
  * the freeness of the logarithmic transforms, *derived* from a fixed-point
    computation on the real torus. The paper states the rule (3 does not
    divide l1; l2 odd); here it comes out of the arithmetic;
  * |p| = |12 l0 - 4 l1 - 3 l2|, the order of pi_1 quoted from Section 7, set
    against the order of H_1 of the Seifert fibred space over S^2(3, 4)
    computed from its presentation by Smith form -- the reading the paper
    offers in its footnote and says the body does not use.

The last check is honest about its scope: it confirms that the paper's closed
form is Orlik's Seifert formula in disguise. It does not prove that pi_1(X) is
cyclic of that order; the van Kampen argument of Section 7 does, and is
declined here.

Conventions follow the paper: V = Z^4 with basis (gamma, u, w, delta),
Lambda = V* with the dual basis, matrices act on columns, and
A(T) = (T^{-1})^t.

References
----------
L. Alpoge, *A compact complex threefold fibred by tori over the projective
line, and the six-sphere*, https://alpo.ge/s6.pdf (2026).
P. Engel, pedagogical account, https://philip-engel.github.io/S6.pdf (2026).
B. Alexeev, Lean formalisation, https://github.com/plby/HopfProblem (2026).
P. Orlik, *Seifert Manifolds*, LNM 291 (1972).
F. Campana, J.-P. Demailly, Th. Peternell, *Rationally connected manifolds
and semipositivity of the Ricci curvature*, corrigendum (2020) [CDP20].
"""

from sympy import Matrix, eye, zeros, Rational, ilcm, symbols, linsolve
from sympy.matrices.normalforms import invariant_factors, smith_normal_decomp

__all__ = [
    "BASIS", "T1", "T2", "dual", "monodromy", "orders",
    "saturate", "int_kernel", "same_lattice", "fixed_lattice",
    "EPS", "EPS_PRIME", "DELTA_HAT", "fixed_lattices", "invariant_vectors",
    "coinvariants", "invariant_forms", "Q0", "b_form",
    "invariant_bivectors", "ETA", "pfaffian", "wedge_square", "hodge_star",
    "B0", "torsion_lattice", "invariant_closure",
    "gamma", "twist_integers", "log_transform_free",
    "pi1_order", "seifert_h1", "seifert_agreement",
    "ledger", "status",
    # Part II
    "TAU", "MU", "BETA", "LAWS", "PHI", "law", "closure", "cocycle_sums",
    "period_matrix", "laws_from_lattice", "R_PRINTED", "lattice_agreement",
    "jtilde_cocycle", "elliptic_points", "local_sections", "D_function",
    "lattice_condition", "hodge_form",
    "KODAIRA_MONODROMY", "WEIERSTRASS", "kodaira_table_check",
    "weierstrass_route", "subquotient", "lattice_route", "kodaira_crosscheck",
    "status_periods",
    # Part III
    "HEXAGON_RAYS", "EDGE_DIRECTIONS", "a2_triangles", "a2_patch",
    "cone_matrix", "fan_checks", "toric_pi1", "star_of_origin",
    "self_intersections", "hexagon_surface", "cusp_fibre", "euler_W_stratified",
    "euler_W_normalisation", "triple_point_formula", "det_scan",
    "status_cusp",
    # Part IV
    "compound", "h1_from_presentation", "pi1_three_routes",
    "invariant_exterior", "cusp_homology", "PSI", "multiple_fibre",
    "hrr_threefold", "DIRECT_IMAGES", "leray_h0q", "hodge_numbers",
    "hkp_constraints", "picard_arithmetic", "hodge_bundle_degree",
    "cstar_fixed_locus", "status_invariants",
]

#: The ordered basis of V; Lambda carries the dual basis with hats.
BASIS = ("gamma", "u", "w", "delta")

#: The two monodromy generators of Definition 2.1 (columns are images).
T1 = Matrix([[1, 0, -6, 2], [0, -1, 1, 1], [0, -1, 0, 1], [0, 0, 0, 1]])
T2 = Matrix([[1, 6, 0, -3], [0, 0, -1, 1], [0, 1, 0, 0], [0, 0, 0, 1]])

#: Vectors of Lambda in the dual basis (gamma^, u^, w^, delta^).
EPS = Matrix([1, 2, -4, 0])
EPS_PRIME = Matrix([1, 3, -3, 0])
DELTA_HAT = Matrix([0, 0, 0, 1])

_I4 = eye(4)


# ---------------------------------------------------------------------------
# integer lattice utilities
# ---------------------------------------------------------------------------

def _integral(v):
    """Scale a rational vector to a primitive integer one."""
    v = Matrix(v)
    d = 1
    for x in v:
        d = ilcm(d, Rational(x).q)
    w = (v * d).applyfunc(int)
    g = 0
    for x in w:
        g = abs(int(x)) if g == 0 else _gcd(g, abs(int(x)))
    return w / g if g > 1 else w


def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def saturate(rows):
    r"""The saturation (span_Q rows) \cap Z^n, as a matrix of basis rows.

    From the Smith decomposition D = U P V: the row space of P is that of
    D V^{-1}, so the first r rows of the unimodular V^{-1} are a Z-basis of
    its saturation.
    """
    P = Matrix(rows)
    if P.rows == 0:
        return P
    r = P.rank()
    D, U, V = smith_normal_decomp(P)
    return V.inv()[:r, :]


def int_kernel(M):
    """A Z-basis (as rows) of ker(M) \\cap Z^n, saturated."""
    ns = Matrix(M).nullspace()
    if not ns:
        return zeros(0, Matrix(M).cols)
    return saturate([list(_integral(v)) for v in ns])


def same_lattice(A, B):
    """Whether two sets of row vectors span the same sublattice of Z^n.

    Each spans the other: every row of one is an integer combination of the
    rows of the other, checked by exact solve and integrality.
    """
    A, B = Matrix(A), Matrix(B)
    if A.rank() != B.rank():
        return False

    def inside(X, Y):
        for i in range(X.rows):
            sol, params = Y.T.gauss_jordan_solve(X.row(i).T)
            sol = sol.subs({p: 0 for p in params})
            # a rational solution exists; integrality of *some* solution is
            # what matters, and for a basis Y the solution is unique
            if Y.rank() == Y.rows and any(Rational(x).q != 1 for x in sol):
                return False
        return True

    Ab, Bb = saturate_basis(A), saturate_basis(B)
    return inside(Ab, Bb) and inside(Bb, Ab)


def saturate_basis(A):
    """A Z-basis of the lattice spanned by the rows of A (not saturated).

    Integer row reduction; used by :func:`same_lattice` and
    :func:`invariant_closure`.
    """
    R = [[int(x) for x in Matrix(A).row(i)] for i in range(Matrix(A).rows)]
    n = Matrix(A).cols
    r = 0
    for c in range(n):
        while True:
            live = [i for i in range(r, len(R)) if R[i][c] != 0]
            if len(live) <= 1:
                break
            p = min(live, key=lambda i: abs(R[i][c]))
            for i in live:
                if i != p:
                    q = R[i][c] // R[p][c]
                    R[i] = [a - q * b for a, b in zip(R[i], R[p])]
        live = [i for i in range(r, len(R)) if R[i][c] != 0]
        if not live:
            continue
        R[r], R[live[0]] = R[live[0]], R[r]
        if R[r][c] < 0:
            R[r] = [-a for a in R[r]]
        r += 1
    return Matrix(R[:r]) if r else zeros(0, n)


# ---------------------------------------------------------------------------
# Section 2: the monodromy matrices
# ---------------------------------------------------------------------------

def dual(T):
    """A(T) = (T^{-1})^t, the contragredient action on Lambda = V*."""
    return Matrix(T).inv().T


def monodromy():
    """T0, N and the dual matrices A1, A2, M0 (Lemmas 2.2 and 2.4)."""
    T0 = (T1 * T2).inv()
    return {"T0": T0, "N": T0 - _I4,
            "A1": dual(T1), "A2": dual(T2), "M0": dual(T0)}


def _order(M, bound=24):
    P = Matrix(M)
    for k in range(1, bound + 1):
        if P == _I4:
            return k
        P = P * M
    return None       # infinite, or larger than the bound


def orders():
    """Lemma 2.2: determinants, exact orders, and the unipotent product."""
    m = monodromy()
    N = m["N"]
    return {
        "det_T1": T1.det(), "det_T2": T2.det(),
        "order_T1": _order(T1), "order_T2": _order(T2),
        "order_T0": _order(m["T0"]),                  # None: infinite
        "N_squared_zero": N * N == zeros(4),
        "N_nonzero": N != zeros(4),
        "ker_N_equals_im_N": same_lattice(int_kernel(N),
                                          saturate(N.T.tolist())),
        "T1T2T0_is_I": T1 * T2 * m["T0"] == _I4,
        "A1A2M0_is_I": m["A1"] * m["A2"] * m["M0"] == _I4,
        "gamma_invariant": all((Matrix([[1, 0, 0, 0]]) * m[k])
                               == Matrix([[1, 0, 0, 0]])
                               for k in ("A1", "A2", "M0")),
    }


def fixed_lattice(M):
    """ker(M - I) \\cap Z^4 as saturated basis rows."""
    return int_kernel(Matrix(M) - _I4)


def fixed_lattices():
    """Lemma 2.6 and 2.7: the fixed sublattices, against the paper's generators."""
    m = monodromy()
    L1, L2 = fixed_lattice(m["A1"]), fixed_lattice(m["A2"])
    return {
        "Lambda_A1": L1, "Lambda_A2": L2,
        "Lambda_A1_is_<eps,delta^>":
            same_lattice(L1, Matrix([list(EPS), list(DELTA_HAT)])),
        "Lambda_A2_is_<eps',delta^>":
            same_lattice(L2, Matrix([list(EPS_PRIME), list(DELTA_HAT)])),
        "gamma(eps)": EPS[0], "gamma(eps')": EPS_PRIME[0],
    }


def invariant_vectors():
    """V^G and Lambda^G (Lemma 2.7 (i), (ii)): Z gamma and Z delta-hat."""
    m = monodromy()
    VG = int_kernel((T1 - _I4).col_join(T2 - _I4))
    LG = int_kernel((m["A1"] - _I4).col_join(m["A2"] - _I4))
    return {"V^G": VG, "Lambda^G": LG,
            "V^G_is_Z_gamma": same_lattice(VG, Matrix([[1, 0, 0, 0]])),
            "Lambda^G_is_Z_delta^": same_lattice(LG, Matrix([list(DELTA_HAT)]))}


def coinvariants(side="V"):
    """Invariant factors of [(T1 - I) | (T2 - I)] (Lemma 2.7 (iii)).

    The cokernel of that 4x8 matrix is V_G. Three unit factors and a missing
    fourth mean V_G = Z, free of rank one.
    """
    if side == "V":
        A, B = T1, T2
    else:
        m = monodromy()
        A, B = m["A1"], m["A2"]
    M = (A - _I4).row_join(B - _I4)
    f = [abs(int(x)) for x in invariant_factors(M)]
    rank = sum(1 for x in f if x != 0)     # sympy lists a missing factor as 0
    return {"invariant_factors": f, "free_rank": 4 - rank,
            "torsion": [x for x in f if x > 1]}


# ---------------------------------------------------------------------------
# the invariant forms: two solves that share no code
# ---------------------------------------------------------------------------

def _antisym(q):
    return Matrix([[0, q[0], q[1], q[2]],
                   [-q[0], 0, q[3], q[4]],
                   [-q[1], -q[3], 0, q[5]],
                   [-q[2], -q[4], -q[5], 0]])


def _solve_invariant(transform):
    q = symbols("q0:6")
    Q = _antisym(q)
    eqs = []
    for T in (T1, T2):
        eqs += list(transform(T, Q) - Q)
    sol = list(linsolve(eqs, q))[0]
    free = sorted(set().union(*[x.free_symbols for x in sol]), key=str)
    basis = []
    for s in free:
        v = [x.subs({t: (1 if t == s else 0) for t in free}) for x in sol]
        basis.append(list(_integral(v)))
    return [_antisym(b) for b in saturate(basis).tolist()]


def invariant_forms():
    """G-invariant alternating forms on V: Q with T^t Q T = Q for T1, T2."""
    return _solve_invariant(lambda T, Q: T.T * Q * T)


def invariant_bivectors():
    """G-invariant bivectors in Lambda^2 V: E with T E T^t = E for T1, T2."""
    return _solve_invariant(lambda T, E: T * E * T.T)


#: Q0 of Lemma 2.8: Q0(gamma, delta) = 1, Q0(u, w) = 6.
Q0 = _antisym([0, 0, 1, 6, 0, 0])

#: eta = u^w + 6 gamma^delta, the invariant bivector of Sections 7 and 9.
ETA = _antisym([0, 0, 6, 1, 0, 0])


def pfaffian(E):
    """Pf of a 4x4 antisymmetric matrix."""
    return E[0, 1] * E[2, 3] - E[0, 2] * E[1, 3] + E[0, 3] * E[1, 2]


def wedge_square(E):
    """eta ^ eta = 2 Pf(E) vol, for the bivector with coefficient matrix E."""
    return 2 * pfaffian(E)


def hodge_star(Q):
    """The combinatorial Hodge dual on Lambda^2 of a rank-4 lattice.

    (*Q)_{ij} = (1/2) eps_{ijkl} Q_{kl}; with det T = 1 it intertwines the
    actions on forms and on bivectors, so it must carry Q0 to +-eta.
    """
    from sympy import LeviCivita
    S = zeros(4)
    for i in range(4):
        for j in range(4):
            S[i, j] = sum(LeviCivita(i, j, k, l) * Q[k, l]
                          for k in range(4) for l in range(4)) / 2
    return S


def b_form():
    """b(x, y) = Q0(x, N y) on V / ker N in the basis (w, delta) (Lemma 2.8).

    A polarised degeneration needs this form definite; it is diag(6, -1).
    """
    N = monodromy()["N"]
    B = Q0 * N
    G = Matrix([[B[2, 2], B[2, 3]], [B[3, 2], B[3, 3]]])
    ev = G.eigenvals()
    pos = sum(k for v, k in ev.items() if v > 0)
    neg = sum(k for v, k in ev.items() if v < 0)
    return {"gram": G, "symmetric_on_V": B == B.T,
            "vanishes_on_ker_N": all(B[i, j] == 0 for i in (0, 1)
                                     for j in range(4)),
            "signature": (pos, neg), "discriminant": G.det(),
            "polarisable": pos == 0 or neg == 0}


# ---------------------------------------------------------------------------
# the cusp: Lambda_tor and B0
# ---------------------------------------------------------------------------

def torsion_lattice():
    """Lemma 2.6: Lambda_tor = ker(M0 - I) = im(M0 - I) = <w^, delta^>."""
    M0 = monodromy()["M0"]
    K = int_kernel(M0 - _I4)
    Im = saturate_basis((M0 - _I4).T)
    std = Matrix([[0, 0, 1, 0], [0, 0, 0, 1]])
    return {"ker": K, "im": Im,
            "ker_is_<w^,delta^>": same_lattice(K, std),
            "im_is_<w^,delta^>": same_lattice(Im, std)}


def B0():
    """The map Lambda/Lambda_tor -> Lambda_tor induced by M0 - I.

    In the bases (gamma^-bar, u^-bar) and (w^, delta^): the columns are the
    (w^, delta^)-coordinates of (M0 - I) gamma^ and (M0 - I) u^.
    """
    D = monodromy()["M0"] - _I4
    return Matrix([[D[2, 0], D[2, 1]], [D[3, 0], D[3, 1]]])


def invariant_closure():
    """Lemma 2.7 (iv): the smallest <A1, A2>-invariant subgroup containing
    Lambda_tor, closed by iterating the monodromy, against ker gamma."""
    m = monodromy()
    L = Matrix([[0, 0, 1, 0], [0, 0, 0, 1]])
    while True:
        new = L
        for A in (m["A1"], m["A2"]):
            new = new.col_join((A * L.T).T)
        new = saturate_basis(new)
        if new.rows == L.rows and same_lattice(new, L):
            break
        L = new
    kerg = Matrix([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    return {"closure": L, "equals_ker_gamma": same_lattice(L, kerg)}


# ---------------------------------------------------------------------------
# the twists: freeness derived, the fundamental group two ways
# ---------------------------------------------------------------------------

def gamma(v):
    """The functional gamma on Lambda: the gamma^-coefficient."""
    return int(Matrix(v)[0])


def twist_integers(v1=None, v2=None, l0=0):
    """(l0, l1, l2) = (l0, gamma(v1), gamma(v2)); defaults v1 = eps, v2 = -eps'."""
    v1 = EPS if v1 is None else Matrix(v1)
    v2 = -EPS_PRIME if v2 is None else Matrix(v2)
    return (l0, gamma(v1), gamma(v2))


def log_transform_free(j, v):
    r"""Whether the Z/m_j action x -> A_j x + v/m_j on the real torus is free.

    Derived, not quoted. The k-th power moves x to A^k x + c_k with
    c_k = (I + A + ... + A^{k-1}) v / m. It has a fixed point on
    (Lambda tensor R)/Lambda iff c_k lies in W_k + Lambda, W_k the real image
    of A^k - I. Take a saturated integer basis P of the annihilator of W_k;
    then P maps Lambda onto the integer points, so the test is P c_k integral.

    Returns (free, per_power) with per_power listing (k, has_fixed_point).
    The paper states the outcome -- free iff 3 does not divide gamma(v1),
    resp. gamma(v2) is odd; :func:`status` checks this over a box.
    """
    m = monodromy()
    A = {1: m["A1"], 2: m["A2"]}[j]
    mj = {1: 3, 2: 4}[j]
    v = Matrix(v)
    if A * v != v:
        raise ValueError("a twist vector must lie in Lambda^{A_%d}" % j)
    per = []
    for k in range(1, mj):
        S = zeros(4, 1)
        P = _I4
        for _ in range(k):
            S += P * v
            P = P * A
        c = S / mj
        P_ann = int_kernel((A ** k - _I4).T)
        per.append((k, all(Rational(x).q == 1 for x in P_ann * c)))
    return (not any(f for _, f in per)), per


def pi1_order(l0, l1, l2):
    """|p| = |12 l0 - 4 l1 - 3 l2|, the order of pi_1(X) (Section 7).

    **Quoted.** The formula is the output of the paper's van Kampen
    computation, which this package does not reproduce. 0 means infinite.
    """
    return abs(12 * l0 - 4 * l1 - 3 * l2)


def seifert_h1(a1, a2, b1, b2, e0):
    r"""H_1 of the Seifert fibred space over S^2(a1, a2) from its presentation.

    pi_1 = < c1, c2, h | h central, c1^{a1} h^{b1}, c2^{a2} h^{b2},
    c1 c2 = h^{e0} >; abelianised, the relation matrix on (c1, c2, h) has rows
    (a1, 0, b1), (0, a2, b2), (1, 1, -e0), and its Smith form gives H_1.
    Returns the invariant factors and the order (0 for infinite).
    """
    R = Matrix([[a1, 0, b1], [0, a2, b2], [1, 1, -e0]])
    f = [abs(int(x)) for x in invariant_factors(R)]
    order = 0 if len(f) < 3 or 0 in f else 1
    if order:
        for x in f:
            order *= x
    return {"invariant_factors": f, "order": order,
            "cyclic": sum(1 for x in f if x > 1) <= 1}


def seifert_agreement(box=4):
    """|p| against |H_1| of the (3, 4) Seifert space with b_j = -l_j, e0 = l0.

    Two routes that share no code: a closed form, and a Smith form. Agreement
    over the box is what the paper's footnote asserts; it identifies the
    formula, not pi_1(X).
    """
    bad = []
    n = 0
    for l0 in range(-box, box + 1):
        for l1 in range(-box, box + 1):
            for l2 in range(-box, box + 1):
                n += 1
                h = seifert_h1(3, 4, -l1, -l2, l0)
                if h["order"] != pi1_order(l0, l1, l2) or not h["cyclic"]:
                    bad.append((l0, l1, l2))
    return {"checked": n, "disagreements": bad,
            "S3_case": seifert_h1(3, 4, -1, 1, 0)}


# ---------------------------------------------------------------------------
# the ledger
# ---------------------------------------------------------------------------

def ledger():
    """What this module computes, and the status of everything around it."""
    return [
        ("orders, determinants, T0 unipotent with N^2 = 0", "exact, computed"),
        ("fixed and invariant sublattices", "exact, computed; matches paper"),
        ("coinvariants V_G = Lambda_G = Z", "exact, Smith form"),
        ("invariant alternating forms: Z Q0", "exact, solved for"),
        ("invariant bivectors: Z eta, eta^2 = 12 vol", "exact, second solve"),
        ("eta = *Q0", "exact; the two solves agree"),
        ("b = diag(6, -1): no polarisation", "exact, computed"),
        ("Lambda_tor = ker = im (M0 - I), det B0 = 1", "exact, computed"),
        ("invariant closure of Lambda_tor is ker gamma", "exact, computed"),
        ("freeness of the logarithmic transforms", "exact, derived here"),
        ("|p| = |12 l0 - 4 l1 - 3 l2|",
         "quoted from Sec. 7; equals |H_1| of the (3,4) Seifert space"),
        ("H_1(X) = Z/|p|",
         "exact, third route: the full presentation of Theorem 7.17 "
         "abelianised on six generators (Part IV)"),
        ("pi_1(X) abelian, the presentation itself",
         "quoted: van Kampen over the pieces (Theorem 7.17)"),
        ("transformation laws of tau, mu, beta; closure under g1^3, g2^4",
         "exact, symbolic identities"),
        ("the laws as forced by T1, T2 (Pi(gz) = R Pi M)",
         "exact, second route; agrees with the printed laws"),
        ("values at z1, z2 and vanishing of phi_j", "exact, in Q(rho), Q(i)"),
        ("det_R Pi = Im tau D, D invariant, Hodge form (1,1)",
         "exact, real-symbolic identities"),
        ("fibre types IV*, III, I_1 of the surface behind tau",
         "exact, two routes: ftheory on the Weierstrass orders, and the "
         "lattice subquotient with Noether's formula"),
        ("existence of tau, mu, beta; choice of c0 with D < 0",
         "declined (NotAnalytic): lifting and torsor arguments"),
        ("the A2 fan: unimodular cones, charts, K_N0 trivial, pi_1(Y) = 1",
         "exact, computed"),
        ("E0 = dP6: six (-1)-curves, degree 6",
         "exact, two routes: pyCICY.toric and the fan"),
        ("W: one component, opposite sides glued, two triple points",
         "exact, lattice orbits of cells"),
        ("e(W) = 2", "exact, two routes: strata and normalisation"),
        ("triple-point formula on every double curve",
         "exact; formula quoted (Friedman 1983), values computed"),
        ("|det B0| = d: d components, e(W) = 2d",
         "exact, computed for d = 1..5 (Remark 4.9)"),
        ("freeness and properness of the Lambda action at the cusp",
         "declined (NotAnalytic): estimates on log|t| against C(t)"),
        ("b(W) = (1,2,4,2,1); ranks of H_*(M) = (1,3,6,6,3,1)",
         "exact, invariants of exterior powers of T0, M0"),
        ("multiple fibres: coinvariants, H_1(S_j), indices of Prop. 7.14",
         "exact; unimodularity of H^2(S_j) quoted"),
        ("chi(O, Omega^1, Omega^2, Omega^3, T) = (0, -1, 1, 0, 1)",
         "exact, HRR from Chern roots; c1 = c2 = 0 and c3 = 2 quoted"),
        ("chi(O) = 0", "exact, two routes: HRR and Leray"),
        ("Hodge numbers", "exact up to one integer: h11 = h22 = h12 + 1; "
         "h12 open (Remark 9.21)"),
        ("K_X = -6u in <H, S1, S2> = Z u; classical formula impossible",
         "exact Picard arithmetic; K_X = f^*O(-1) + 2 S2 quoted"),
        ("deg f_* omega = 1", "exact, two routes: Kodaira Euler numbers "
         "and the local exponents a_j/m_j"),
        ("C^* fixed locus in the cusp model: one P^1, weights (+1, -1)",
         "exact, toric; no fixed points elsewhere quoted"),
        ("direct images R^q f_* O, h^{p,0} = 0, a(X) = 1",
         "quoted (Theorem 9.1): analysis on the fibres"),
        ("30 facts of the finite layer",
         "machine-checked: Lean 4 kernel, Mathlib-free, standard axioms "
         "only, 25 choice-free (pyCICY.sixsphere_lean; Part V)"),
        ("X diffeomorphic to S^6",
         "declined: analysis and topology. Machine-checked externally in "
         "github.com/plby/HopfProblem, not re-run here"),
    ]


def status(box=3):
    """Run every check in this module and report pass/fail per item."""
    o = orders()
    fl = fixed_lattices()
    iv = invariant_vectors()
    forms = invariant_forms()
    bivs = invariant_bivectors()
    b = b_form()
    tl = torsion_lattice()
    cl = invariant_closure()
    free_rule = all(
        log_transform_free(1, a * EPS + d * DELTA_HAT)[0] == (a % 3 != 0)
        and log_transform_free(2, a * EPS_PRIME + d * DELTA_HAT)[0]
        == (a % 2 != 0)
        for a in range(-box, box + 1) for d in (0, 1))
    sa = seifert_agreement(box)
    checks = {
        "orders 3 and 4, det 1": (o["order_T1"], o["order_T2"],
                                  o["det_T1"], o["det_T2"]) == (3, 4, 1, 1),
        "T0 unipotent, N^2 = 0, ker N = im N":
            o["order_T0"] is None and o["N_squared_zero"]
            and o["ker_N_equals_im_N"],
        "A1 A2 M0 = I, gamma invariant":
            o["A1A2M0_is_I"] and o["gamma_invariant"],
        "fixed lattices": fl["Lambda_A1_is_<eps,delta^>"]
            and fl["Lambda_A2_is_<eps',delta^>"],
        "invariant vectors": iv["V^G_is_Z_gamma"]
            and iv["Lambda^G_is_Z_delta^"],
        "coinvariants Z": coinvariants("V")["free_rank"] == 1
            and coinvariants("Lambda")["free_rank"] == 1
            and not coinvariants("V")["torsion"],
        "unique invariant form Q0":
            len(forms) == 1 and forms[0] in (Q0, -Q0),
        "unique invariant bivector eta, eta^2 = 12":
            len(bivs) == 1 and bivs[0] in (ETA, -ETA)
            and wedge_square(ETA) == 12,
        "eta = *Q0": hodge_star(Q0) in (ETA, -ETA),
        "b = diag(6,-1), not polarisable":
            b["gram"] == Matrix([[6, 0], [0, -1]]) and not b["polarisable"],
        "Lambda_tor = ker = im": tl["ker_is_<w^,delta^>"]
            and tl["im_is_<w^,delta^>"],
        "B0 unimodular": abs(B0().det()) == 1,
        "closure of Lambda_tor is ker gamma": cl["equals_ker_gamma"],
        "twists (0, 1, -1), both free":
            twist_integers() == (0, 1, -1)
            and log_transform_free(1, EPS)[0]
            and log_transform_free(2, -EPS_PRIME)[0],
        "freeness rule derived (3 | l1 or l2 even => not free)": free_rule,
        "|p| = 1; X' with v2 = +eps' has |p| = 7":
            pi1_order(*twist_integers()) == 1
            and pi1_order(*twist_integers(v2=EPS_PRIME)) == 7,
        "|p| = |H_1(Seifert(3,4))| over the box":
            not sa["disagreements"] and sa["S3_case"]["order"] == 1,
    }
    return {"checks": checks, "all_pass": all(checks.values()),
            "ledger": ledger()}


# ===========================================================================
# Part II -- Section 3: the period laws, as identities
# ===========================================================================
#
# The period functions tau, mu, beta on the upper half plane are existence
# theorems (a lifting argument, and two torsors under line bundles on P^1),
# and nothing here constructs them. But every *transformation law* they are
# required to satisfy is a rational identity in three symbols, and those
# identities carry the whole consistency of the construction: that the laws
# close up under g1^3 = g2^4 = 1, that they are the ones the integer matrices
# force, that the lattice stays a lattice, and that the Hodge form has the
# signature it has. All of that is checked here in exact symbolic arithmetic.
#
# Convention (Definition 3.1): for a function f and g in Delta, f o g is
# expressed through (tau, mu, beta) at z. Then f o (g h) = (f o g) o h, so the
# substitution for a word g h is F_g evaluated at F_h.

from sympy import (symbols as _symbols, cancel as _cancel, simplify as _simplify,
                   expand_complex as _expand_complex, I as _I, re as _re,
                   im as _im, sqrt as _sqrt, Poly as _Poly, quo as _quo,
                   expand as _expand, solve as _solve)

TAU, MU, BETA = _symbols("tau mu beta")
_X = (TAU, MU, BETA)

#: The printed laws (tau1)-(beta1) of Definition 3.1.
LAWS = {
    "g1": ((TAU - 1) / TAU, (1 - MU) / TAU,
           BETA + 2 - 6 * (1 - MU) ** 2 / TAU),
    "g2": (-1 / TAU, 1 + MU / TAU, BETA - 3 - 6 * MU ** 2 / TAU),
}

#: phi_j, the inhomogeneous terms of the beta laws: beta o g_j = beta + phi_j.
PHI = {"g1": 2 - 6 * (1 - MU) ** 2 / TAU, "g2": -3 - 6 * MU ** 2 / TAU}


def _subst(exprs, images):
    return tuple(_cancel(e.subs(dict(zip(_X, images)), simultaneous=True))
                 for e in exprs)


def law(word, laws=None):
    """(tau, mu, beta) o w for a word w, a sequence of 'g1', 'g2'."""
    laws = LAWS if laws is None else laws
    F = _X
    for g in word:
        F = _subst(F, laws[g])
    return F


def closure():
    """The laws are consistent with the relations of Delta.

    g1^3 and g2^4 act trivially (Propositions 3.11 and 3.13 in one stroke:
    closure of the beta law *is* the vanishing of the cocycle sums), and
    g1 g2 acts by (tau + 1, mu, beta - 1), so g0 = (g1 g2)^{-1} acts by
    (tau - 1, mu, beta + 1), the parabolic law of the cusp.
    """
    return {"g1^3": law(["g1"] * 3), "g2^4": law(["g2"] * 4),
            "g1g2": law(["g1", "g2"]),
            "g1^3_trivial": law(["g1"] * 3) == _X,
            "g2^4_trivial": law(["g2"] * 4) == _X,
            "g0_law": law(["g1", "g2"]) == (TAU + 1, MU, BETA - 1)}


def cocycle_sums():
    """Proposition 3.13 by the paper's own route: sum_k phi_j o g_j^k = 0."""
    out = {}
    for g, m in (("g1", 3), ("g2", 4)):
        s = 0
        for k in range(m):
            s += _subst((PHI[g],), law([g] * k))[0]
        out[g] = _simplify(s)
    return out


def period_matrix(tau=TAU, mu=MU, beta=BETA):
    """Pi = [Z | I] with Z = [[6 mu, tau], [beta, mu]] (Definition 3.3)."""
    return Matrix([[6 * mu, tau, 1, 0], [beta, mu, 0, 1]])


def laws_from_lattice(T):
    r"""Derive the transformation law of g from its integer matrix alone.

    Pi(gz) = R_g(z) Pi(z) M_g with M_g = T^t, and Pi(gz) must again have
    right block I; so R_g is the inverse of the right block of Pi(z) T^t, and
    the left block of R_g Pi(z) T^t is Z(gz). Two things come out: the law,
    and a *consistency condition* the matrices did not have to satisfy --
    that Z(gz) again has the shape [[6 mu', tau'], [beta', mu']].
    """
    P = period_matrix() * Matrix(T).T
    R = P[:, 2:4].inv().applyfunc(_cancel)
    Z = (R * P[:, 0:2]).applyfunc(_cancel)
    return {"R": R, "Z": Z,
            "shape_ok": _cancel(Z[0, 0] - 6 * Z[1, 1]) == 0,
            "law": (Z[0, 1], Z[1, 1], Z[1, 0])}


#: R_g printed in Proposition 3.16.
R_PRINTED = {"g1": Matrix([[-1 / TAU, 0], [(1 - MU) / TAU, 1]]),
             "g2": Matrix([[1 / TAU, 0], [-MU / TAU, 1]])}


def lattice_agreement():
    """The printed laws against the ones the integer matrices force."""
    out = {}
    for g, T in (("g1", T1), ("g2", T2), ("g0", monodromy()["T0"])):
        d = laws_from_lattice(T)
        want = LAWS.get(g, (TAU - 1, MU, BETA + 1))
        out[g] = {
            "shape_ok": d["shape_ok"],
            "law_ok": all(_cancel(a - b) == 0 for a, b in zip(d["law"], want)),
            "R_ok": (g == "g0" and d["R"] == eye(2))
            or (g in R_PRINTED
                and (d["R"] - R_PRINTED[g]).applyfunc(_cancel) == zeros(2)),
        }
    return out


def jtilde_cocycle():
    """Lemma 3.8: the automorphy factor jtilde(g1) = -tau, jtilde(g2) = tau.

    Its products over the orbits of g1 and g2 are 1, jtilde(g0) = 1, and it is
    1/det R_g -- a third, independent appearance of the same factor.
    """
    jt = {"g1": -TAU, "g2": TAU}

    def along(word):
        # jtilde(g_1 ... g_k, z) = prod jtilde(g_i, g_{i+1}...g_k z)
        prod = 1
        for i, g in enumerate(word):
            prod *= _subst((jt[g],), law(word[i + 1:]))[0]
        return _cancel(prod)

    return {"g1_orbit": along(["g1"] * 3), "g2_orbit": along(["g2"] * 4),
            "g0": _cancel(1 / along(["g1", "g2"])),
            "det_R_is_1/jtilde": all(
                _cancel(R_PRINTED[g].det() - 1 / jt[g]) == 0 for g in jt)}


def elliptic_points():
    r"""Values at the fixed points z1, z2 (Theorem 3.4 (i), (ii)).

    tau(z1), tau(z2) are the upper-half-plane fixed points of the tau laws;
    mu(z_j) is then forced by the mu law; and phi_j vanishes at z_j, which is
    what makes the beta problem locally solvable there. Returned exactly, in
    Q(rho) and Q(i).
    """
    out = {}
    for g in ("g1", "g2"):
        tfix = [s for s in _solve(_sympify_eq(LAWS[g][0] - TAU), TAU)
                if _im(s) > 0]
        t0 = tfix[0]
        m0 = [s for s in _solve(LAWS[g][1].subs(TAU, t0) - MU, MU)][0]
        out[g] = {"tau": _simplify(t0), "mu": _simplify(m0),
                  "phi": _simplify(_expand_complex(
                      PHI[g].subs({TAU: t0, MU: m0})))}
    rho = (1 + _sqrt(3) * _I) / 2
    out["tau(z1)_is_rho"] = _simplify(out["g1"]["tau"] - rho) == 0
    out["tau(z2)_is_i"] = _simplify(out["g2"]["tau"] - _I) == 0
    out["mu(z1)=(2-rho)/3"] = _simplify(out["g1"]["mu"] - (2 - rho) / 3) == 0
    out["mu(z2)=(1-i)/2"] = _simplify(out["g2"]["mu"] - (1 - _I) / 2) == 0
    out["6(1-mu)^2 = 2 tau at z1"] = _simplify(_expand_complex(
        6 * (1 - out["g1"]["mu"]) ** 2 - 2 * out["g1"]["tau"])) == 0
    out["6 mu^2 = -3 tau at z2"] = _simplify(_expand_complex(
        6 * out["g2"]["mu"] ** 2 + 3 * out["g2"]["tau"])) == 0
    return out


def _sympify_eq(e):
    return _cancel(e * TAU)          # clear the denominator tau


def local_sections():
    """The explicit local solutions of Propositions 3.11 and 3.13.

    mu_1 = (2 - tau)/3 and mu_2 = (1 - tau)/2 satisfy the mu laws at z1, z2;
    beta_j = (1/m_j) sum_k k (phi_j o g_j^k) telescopes to beta o g_j - beta
    = phi_j; and at the cusp mu = 0, beta = -tau satisfy the g0 law.
    """
    out = {}
    mus = {"g1": (2 - TAU) / 3, "g2": (1 - TAU) / 2}
    for g, mj in mus.items():
        lhs = mj.subs(TAU, LAWS[g][0])                      # mu_j o g
        rhs = LAWS[g][1].subs(MU, mj)                       # law applied
        out["mu_%s" % g] = _cancel(lhs - rhs) == 0
    for g, m in (("g1", 3), ("g2", 4)):
        # generic mu: the telescoping uses only the cocycle relation
        bj = sum(k * _subst((PHI[g],), law([g] * k))[0] for k in range(m)) / m
        bj_g = _subst((bj,), LAWS[g])[0]
        out["beta_%s_telescopes" % g] = _cancel(bj_g - bj - PHI[g]) == 0
    g0 = (TAU - 1, MU, BETA + 1)
    out["cusp_mu=0"] = True                                 # mu o g0 = mu
    out["cusp_beta=-tau"] = _cancel((-TAU).subs(TAU, g0[0]) - (-TAU + 1)) == 0
    return out


# -- the real layer: lattice condition and the Hodge form ------------------

_a, _b, _c, _d, _e, _f = _symbols("a b c d e f", real=True)
_TAU_R, _MU_R, _BETA_R = _a + _I * _b, _c + _I * _d, _e + _I * _f


def D_function(tau=_TAU_R, mu=_MU_R, beta=_BETA_R):
    """D = Im beta - 6 (Im mu)^2 / Im tau (condition (beta3))."""
    return _im(beta) - 6 * _im(mu) ** 2 / _im(tau)


def _real(e):
    return _simplify(_expand_complex(e))


def lattice_condition():
    r"""Lemma 3.14, both halves, in real symbols tau = a+ib, mu = c+id, beta = e+if.

    det_R Pi = Im tau * D, so L(z) is a lattice iff D != 0; and D is
    invariant under g1 and g2. Checked as identities, not at sample points.
    """
    Pi = period_matrix(_TAU_R, _MU_R, _BETA_R)
    rows = []
    for i in range(2):
        rows.append([_re(x) for x in Pi.row(i)])
        rows.append([_im(x) for x in Pi.row(i)])
    detR = Matrix(rows).det()
    D0 = D_function()
    out = {"detR_Pi = Im tau * D": _real(detR - _b * D0) == 0}
    sub = {TAU: _TAU_R, MU: _MU_R, BETA: _BETA_R}
    for g in ("g1", "g2"):
        t, m, B = (x.subs(sub, simultaneous=True) for x in LAWS[g])
        out["D o %s = D" % g] = _real(D_function(t, m, B) - D0) == 0
    return out


def hodge_form():
    r"""Remark 3.23: F^1 is Q0-Lagrangian and h = i Q0(., conj .) on it.

    sigma_1 = 6 mu gamma + tau u + w, sigma_2 = beta gamma + mu u + delta.
    Gram -[[12 Im tau, 12 Im mu], [12 Im mu, 2 Im beta]], determinant
    24 Im tau * D < 0 under (beta3): signature (1, 1) for either sign of Q0,
    so no choice of c0 gives a polarisation. This is the analytic face of
    b = diag(6, -1) from Part I.
    """
    s1 = Matrix([6 * _MU_R, _TAU_R, 1, 0])
    s2 = Matrix([_BETA_R, _MU_R, 0, 1])
    S_ = (s1, s2)
    lag = _real((s1.T * Q0 * s2)[0])
    H = Matrix(2, 2, lambda i, j: _real(
        _I * (S_[i].T * Q0 * S_[j].conjugate())[0]))
    want = -Matrix([[12 * _b, 12 * _d], [12 * _d, 2 * _f]])
    return {"lagrangian": lag == 0, "gram": H,
            "gram_as_printed": (H - want).applyfunc(_real) == zeros(2),
            "det = 24 Im tau D": _real(H.det() - 24 * _b * D_function()) == 0}


# ===========================================================================
# Part II -- the elliptic surface behind tau, and Kodaira two ways
# ===========================================================================
#
# Remark 3.12: tau is the period map of the rational elliptic surface
#     y^2 = 4 x^3 - 3 t^3 (t-1) x - t^4 (t-1)^2,     j = 1728 t,
# with fibres IV*, III, I_1 over t = 0, 1, infinity. j alone fixes the surface
# only up to quadratic twist, and a twist changes the fibre types (it
# multiplies the local monodromy by -1). So there are two genuinely different
# routes to the three fibre types:
#
#   Weierstrass  the vanishing orders of f, g, Delta, classified by
#                pyCICY.theories.ftheory.kodaira_type -- code written for
#                F-theory, knowing nothing of this construction;
#   lattice      mod gamma, sigma_1 = tau u + w, so <u, w> = <gamma,u,w>/<gamma>
#                carries the elliptic curve's H_1 with its SL_2(Z) sign. Its
#                monodromy classes (order, trace) narrow each point to a pair
#                {X, X*}, and Noether's formula e = 12 chi(O) -- the Euler
#                numbers must sum to a multiple of 12 -- picks one combination.
#
# The lattice route never sees the Weierstrass model; the Weierstrass route
# never sees T1, T2.

#: Kodaira's table (quoted: Kodaira 1963; BHPV Table V.6): local monodromy in
#: SL_2(Z) up to conjugacy, and Euler number of the fibre.
KODAIRA_MONODROMY = {
    "I_0":  (Matrix([[1, 0], [0, 1]]), 0),
    "I_1":  (Matrix([[1, 1], [0, 1]]), 1),
    "II":   (Matrix([[1, 1], [-1, 0]]), 2),
    "III":  (Matrix([[0, 1], [-1, 0]]), 3),
    "IV":   (Matrix([[0, 1], [-1, -1]]), 4),
    "I_0*": (Matrix([[-1, 0], [0, -1]]), 6),
    "IV*":  (Matrix([[-1, -1], [1, 0]]), 8),
    "III*": (Matrix([[0, -1], [1, 0]]), 9),
    "II*":  (Matrix([[0, -1], [1, 1]]), 10),
}

#: Vanishing orders (f, g, Delta) realising each type, for the table check.
_KODAIRA_ORDERS = {"I_1": (0, 0, 1), "II": (1, 1, 2), "III": (1, 2, 3),
                   "IV": (2, 2, 4), "I_0*": (2, 3, 6), "IV*": (3, 4, 8),
                   "III*": (3, 5, 9), "II*": (4, 5, 10)}

_T = _symbols("t")
#: g2, g3 of the Weierstrass model of Remark 3.12.
WEIERSTRASS = (3 * _T ** 3 * (_T - 1), _T ** 4 * (_T - 1) ** 2)


def kodaira_table_check():
    """The quoted Euler numbers equal ord(Delta), as classified by ftheory.

    Guards the quoted table against a transcription slip: for each type, the
    representative orders go through ftheory.kodaira_type and must come back
    as that type, with Euler number = ord Delta.
    """
    from .theories.ftheory import kodaira_type
    out = {}
    for name, o in _KODAIRA_ORDERS.items():
        k = kodaira_type(*o)["type"]
        out[name] = (k == name) and KODAIRA_MONODROMY[name][1] == o[2]
    return out


def _order_in_sl2(M, bound=12):
    P = Matrix(M)
    for k in range(1, bound + 1):
        if P == eye(2):
            return k
        P = P * M
    return None


def _ord_at(p, pt, deg):
    p = _Poly(_expand(p), _T)
    if pt == "oo":
        return deg - p.degree()
    k, q = 0, p
    while q.eval(pt) == 0:
        q = _Poly(_quo(q.as_expr(), _T - pt), _T)
        k += 1
    return k


def weierstrass_route():
    """Fibre types from vanishing orders, through ftheory.kodaira_type.

    y^2 = 4x^3 - g2 x - g3 becomes y'^2 = x^3 + f x + g with f = -g2/4,
    g = -g3/4 (y = 2y'); orders at infinity from the degrees 4, 6, 12 of
    f, g, Delta as sections over P^1.
    """
    from .theories.ftheory import kodaira_type
    g2, g3 = WEIERSTRASS
    f, g = -g2 / 4, -g3 / 4
    Delta = 4 * f ** 3 + 27 * g ** 2
    j = _cancel(1728 * g2 ** 3 / (g2 ** 3 - 27 * g3 ** 2))
    fibres = {}
    for pt in (0, 1, "oo"):
        o = (_ord_at(f, pt, 4), _ord_at(g, pt, 6), _ord_at(Delta, pt, 12))
        fibres[pt] = {"orders": o, "type": kodaira_type(*o)["type"]}
    euler = sum(KODAIRA_MONODROMY[fibres[p]["type"]][1] for p in fibres)
    return {"j": j, "j_is_1728t": _cancel(j - 1728 * _T) == 0,
            "fibres": fibres, "euler_sum": euler}


def subquotient(T):
    """The action of T on <gamma, u, w>/<gamma>, in the basis (u, w)."""
    T = Matrix(T)
    return Matrix([[T[1, 1], T[1, 2]], [T[2, 1], T[2, 2]]])


def lattice_route():
    r"""Fibre types from the lattice alone, plus Noether's formula.

    1. <gamma,u,w> is T-stable and gamma is fixed, so each T acts on the
       quotient; the action is multiplicative, and on the period (tau, 1) it
       reproduces the tau laws -- a third route to (tau1)-(tau2).
    2. (order, trace) of each subquotient matrix, and for a unipotent one the
       n of I_n = gcd of the entries of M - I, narrow each point to the
       Kodaira types with that class.
    3. Orientation conventions invert monodromy; order and trace do not see
       that, which is why the result is a pair {X, X*} and not a type.
    4. The Euler numbers of the fibres of an elliptic surface sum to
       12 chi(O_S), a positive multiple of 12. Exactly one combination passes.
    """
    m = monodromy()
    mats = {0: subquotient(T1), 1: subquotient(T2), "oo": subquotient(m["T0"])}
    # the laws of tau, from the action on (tau, 1)
    tau_laws = {}
    for key, g in ((0, "g1"), (1, "g2")):
        v = mats[key] * Matrix([TAU, 1])
        tau_laws[g] = _cancel(v[0] / v[1] - LAWS[g][0]) == 0
    v0 = mats["oo"] * Matrix([TAU, 1])
    tau_laws["g0"] = _cancel(v0[0] / v0[1] - (TAU - 1)) == 0
    hom = subquotient(T1 * T2) == subquotient(T1) * subquotient(T2)

    def cls(M):
        return (_order_in_sl2(M), M.trace())

    candidates = {}
    for pt, M in mats.items():
        o, tr = cls(M)
        cands = []
        for name, (K, e) in KODAIRA_MONODROMY.items():
            if name == "I_0":
                continue
            if cls(K) == (o, tr):
                cands.append(name)
        if o is None and tr == 2:                     # unipotent: I_n
            n = 0
            for x in (M - eye(2)):
                n = _gcd(n, abs(int(x)))
            cands = ["I_%d" % n] if n == 1 else ["I_%d (not tabulated)" % n]
        candidates[pt] = cands
    combos = []
    for a in candidates[0]:
        for b in candidates[1]:
            for c in candidates["oo"]:
                e = sum(KODAIRA_MONODROMY[x][1] for x in (a, b, c))
                combos.append(((a, b, c), e))
    passing = [c for c, e in combos if e > 0 and e % 12 == 0]
    return {"matrices": mats, "tau_laws": tau_laws, "homomorphism": hom,
            "candidates": candidates, "combinations": combos,
            "noether_selects": passing}


def kodaira_crosscheck():
    """Both routes, and whether they agree."""
    w = weierstrass_route()
    l = lattice_route()
    wt = tuple(w["fibres"][p]["type"] for p in (0, 1, "oo"))
    return {"weierstrass": wt, "lattice": l["noether_selects"],
            "agree": l["noether_selects"] == [wt],
            "euler_sum": w["euler_sum"]}


def status_periods():
    """Run every Part II check; pass/fail per item."""
    cl = closure()
    cs = cocycle_sums()
    la = lattice_agreement()
    jt = jtilde_cocycle()
    ep = elliptic_points()
    ls = local_sections()
    lc = lattice_condition()
    hf = hodge_form()
    kt = kodaira_table_check()
    kc = kodaira_crosscheck()
    lr = lattice_route()
    checks = {
        "g1^3 = g2^4 = 1 on (tau, mu, beta)":
            cl["g1^3_trivial"] and cl["g2^4_trivial"],
        "g0 = (g1 g2)^-1 acts by (tau-1, mu, beta+1)": cl["g0_law"],
        "cocycle sums vanish (Prop. 3.13)": cs == {"g1": 0, "g2": 0},
        "laws derived from T1, T2, T0 match the printed ones":
            all(v["law_ok"] for v in la.values()),
        "Z(gz) keeps the shape [[6 mu', tau'], [beta', mu']]":
            all(v["shape_ok"] for v in la.values()),
        "R_g as printed; R_g0 = I": all(v["R_ok"] for v in la.values()),
        "jtilde closes on orbits, jtilde(g0) = 1, det R = 1/jtilde":
            jt["g1_orbit"] == 1 and jt["g2_orbit"] == 1 and jt["g0"] == 1
            and jt["det_R_is_1/jtilde"],
        "tau(z1) = rho, tau(z2) = i":
            ep["tau(z1)_is_rho"] and ep["tau(z2)_is_i"],
        "mu(z1) = (2-rho)/3, mu(z2) = (1-i)/2":
            ep["mu(z1)=(2-rho)/3"] and ep["mu(z2)=(1-i)/2"],
        "phi_j vanishes at z_j":
            ep["6(1-mu)^2 = 2 tau at z1"] and ep["6 mu^2 = -3 tau at z2"],
        "local sections of the mu and beta problems": all(ls.values()),
        "det_R Pi = Im tau * D; D invariant": all(lc.values()),
        "F^1 Lagrangian; Hodge Gram as printed, det = 24 Im tau D":
            hf["lagrangian"] and hf["gram_as_printed"]
            and hf["det = 24 Im tau D"],
        "Kodaira table: Euler = ord Delta, via ftheory": all(kt.values()),
        "subquotient <u,w> reproduces the tau laws":
            all(lr["tau_laws"].values()) and lr["homomorphism"],
        "Weierstrass: j = 1728 t; IV*, III, I_1; Euler sum 12":
            kc["weierstrass"] == ("IV*", "III", "I_1")
            and kc["euler_sum"] == 12
            and weierstrass_route()["j_is_1728t"],
        "lattice + Noether select the same three types": kc["agree"],
    }
    return {"checks": checks, "all_pass": all(checks.values())}


# ===========================================================================
# Part III -- Section 4: the toric filling at the cusp
# ===========================================================================
#
# At the cusp the monodromy is unipotent with Lambda_tor = ker = im (M0 - I),
# and the fibre is filled in, after Mumford, by a quotient of an infinite
# smooth toric threefold Y. Its fan is the cone over the A_2 triangulation of
# R^2 x {1}: vertices Z^2, edges in the three directions e1, e2, e2 - e1, and
# two triangles per unit square. Lambda-bar = Lambda/Lambda_tor acts through
# translations by B0 Lambda-bar, and W = t^{-1}(0)/Lambda.
#
# All of the combinatorics is exact and done here: unimodularity of every
# cone, the charts and the holomorphic volume form, the star of a vertex as
# the hexagonal fan of the degree-6 del Pezzo (identified by pyCICY.toric,
# which calls it dP3 = B3 by the number of blown-up points), the
# identifications the quotient makes, and e(W) by two routes. Then B0 is made
# a parameter, and the paper's Remark 4.9 -- |det B0| = d gives d components
# and e(W) = 2d -- is computed rather than read.
#
# Not here: that the Lambda action is free and properly discontinuous and the
# quotient proper over the disc (Theorem 4.5 (a)-(c)). Those are estimates on
# log|t| against a holomorphic C(t), and are declined.

#: The six rays around a vertex, in the cyclic order of Lemma 4.2 (iv); the
#: Lean formalisation's ToricComponent.hexagonRay uses the same order.
HEXAGON_RAYS = ((1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1))

#: The three edge directions of T_{A2}, one per pair +-.
EDGE_DIRECTIONS = ((1, 0), (0, 1), (-1, 1))


def _add(a, b):
    return (a[0] + b[0], a[1] + b[1])


def a2_triangles(v):
    """The two triangles based at v: 'lower' {v, v+e1, v+e2}, 'upper'
    {v+e1, v+e2, v+e1+e2}."""
    e1, e2 = (1, 0), (0, 1)
    return {("lower", v): (v, _add(v, e1), _add(v, e2)),
            ("upper", v): (_add(v, e1), _add(v, e2), _add(_add(v, e1), e2))}


def a2_patch(radius=2):
    """Cells of T_{A2} based at vertices in [-radius, radius]^2."""
    verts, edges, tris = [], [], {}
    for x in range(-radius, radius + 1):
        for y in range(-radius, radius + 1):
            v = (x, y)
            verts.append(v)
            for d in EDGE_DIRECTIONS:
                edges.append((v, _add(v, d)))
            tris.update(a2_triangles(v))
    return verts, edges, tris


def cone_matrix(tri):
    """Rows (v_i, 1): the primitive ray generators of the cone over tri."""
    return Matrix([[v[0], v[1], 1] for v in tri])


def fan_checks(radius=2):
    r"""Lemma 4.2 (i)-(iii) on a patch.

    Every triangle cone is unimodular (det = +1 lower, -1 upper); every ray
    is at height 1, so t vanishes to order exactly 1 on every E_v and W is
    reduced; the dual basis of each cone sums to (0, 0, 1), so t = z0 z1 z2
    in every chart; and the change of basis to the torus coordinates has
    det +-1, so dx1 dx2 dt / (x1 x2) = +- dz0 dz1 dz2 extends as a
    nowhere-vanishing 3-form (Theorem 4.5 (e): K_{N0} trivial).
    """
    _, _, tris = a2_patch(radius)
    dets, chart_ok, omega_ok = {}, True, True
    for (kind, v), tri in tris.items():
        M = cone_matrix(tri)
        dets.setdefault(kind, set()).add(M.det())
        dual_basis = M.inv()                 # columns m_i: <m_i, (v_j,1)> = delta
        chart_ok &= sum((dual_basis[:, i] for i in range(3)),
                        zeros(3, 1)) == Matrix([0, 0, 1])
        omega_ok &= abs(dual_basis.det()) == 1
    return {"dets": dets,
            "unimodular": all(abs(d) == 1 for s in dets.values() for d in s),
            "t = z0 z1 z2 in every chart": chart_ok,
            "Omega nowhere vanishing": omega_ok,
            "rays at height 1": True}


def toric_pi1(radius=1):
    """pi_1 of the finite subfans: N'/<rays> by Smith form (Fulton 3.2).

    Trivial once the rays (0,1), (e1,1), (e2,1) are present; Corollary 4.8
    passes to the increasing union.
    """
    verts = [(x, y) for x in range(-radius, radius + 1)
             for y in range(-radius, radius + 1)]
    R = Matrix([[v[0], v[1], 1] for v in verts])
    f = [abs(int(x)) for x in invariant_factors(R)]
    return {"invariant_factors": f, "trivial": f == [1, 1, 1]}


def star_of_origin():
    """The six triangles containing 0, and their rays in cyclic order."""
    star = []
    for x in (-1, 0):
        for y in (-1, 0):
            for key, tri in a2_triangles((x, y)).items():
                if (0, 0) in tri:
                    star.append((key, tri))
    rays = sorted({v for _, tri in star for v in tri if v != (0, 0)},
                  key=lambda r: __import__("math").atan2(r[1], r[0]) % (
                      2 * __import__("math").pi))
    return {"triangles": star, "rays": tuple(rays)}


def self_intersections(rays):
    """D_i^2 for a complete smooth 2d fan with rays in cyclic order.

    v_{i-1} + v_{i+1} = b_i v_i and D_i^2 = -b_i (Fulton, Sec. 2.5).
    """
    out = []
    n = len(rays)
    for i in range(n):
        s = _add(rays[i - 1], rays[(i + 1) % n])
        r = rays[i]
        b = s[0] // r[0] if r[0] else s[1] // r[1]
        out.append(-b if (b * r[0], b * r[1]) == s else None)
    return out


def hexagon_surface():
    r"""The component E_0: the toric surface of the star of a vertex.

    Identified two ways. pyCICY.toric classifies the polygon of rays (it is
    reflexive and smooth, degree K^2 = 6). Independently, from the fan: the
    six rays in cyclic order have consecutive determinants +1 (a complete
    smooth fan), each ray v_i satisfies v_{i-1} + v_{i+1} = b_i v_i with
    D_i^2 = -b_i = -1 (six (-1)-curves, the anticanonical hexagon), e = #rays
    = 6, and Noether K^2 + e = 12 closes the circle.
    """
    from . import toric
    st = star_of_origin()
    rays = st["rays"]
    n = len(rays)
    consec = [Matrix([rays[i], rays[(i + 1) % n]]).det() for i in range(n)]
    selfint = self_intersections(rays)
    cl = toric.classify(list(rays))
    euler = n
    return {"rays": rays, "matches_paper_and_lean": rays == HEXAGON_RAYS,
            "consecutive_dets": consec, "self_intersections": selfint,
            "toric_name": cl["name"], "toric_smooth": cl["smooth"],
            "toric_degree": cl["degree"], "euler": euler,
            "noether": cl["degree"] + euler == 12}


# -- the quotient by B0 Lambda-bar -----------------------------------------

def _reduce(v, Binv):
    """Canonical key of v in Z^2 / B Z^2: the fractional part of B^{-1} v.

    ``Binv`` is B^{-1} as a 2x2 tuple of Fractions, precomputed once.
    """
    from fractions import Fraction
    x0 = Binv[0][0] * v[0] + Binv[0][1] * v[1]
    x1 = Binv[1][0] * v[0] + Binv[1][1] * v[1]
    return (x0 - (x0.numerator // x0.denominator),
            x1 - (x1.numerator // x1.denominator))


def _binv(B):
    from fractions import Fraction
    Bi = B.inv()
    return tuple(tuple(Fraction(int(Rational(Bi[i, j]).p),
                                int(Rational(Bi[i, j]).q)) for j in range(2))
                 for i in range(2))


def cusp_fibre(B=None):
    r"""The central fibre W = t^{-1}(0) / (B Z^2), combinatorially.

    Orbits of vertices, edges and triangles of T_{A2} under translation by the
    lattice B Z^2 are the irreducible components, the double curves and the
    triple points of W. Returned: their counts; which hexagon sides of E_0
    are glued; how the six hexagon corners of E_0 map to triple points; which
    double curves pass through which triple points; and the dual complex.
    """
    B = B0() if B is None else Matrix(B)
    d = abs(B.det())
    if d == 0:
        raise ValueError("B0 must be non-degenerate (Remark 4.9 (a))")
    Bi = _binv(B)
    # enumerate a patch large enough to meet every coset
    R = int(max(abs(x) for x in B)) * 2 + 2
    verts, edges, tris = a2_patch(R)
    vkeys = {_reduce(v, Bi) for v in verts}
    ekeys = {(_sub(e[1], e[0]), _reduce(e[0], Bi)) for e in edges}
    tkeys = {(k[0], _reduce(k[1], Bi)) for k in tris}
    # the hexagon of E_0: side {0, r} glued to side {0, r'} iff same orbit
    side_key = {}
    for r in HEXAGON_RAYS:
        base, dirn = ((0, 0), r) if r in EDGE_DIRECTIONS else (r, _neg(r))
        side_key[r] = (dirn, _reduce(base, Bi))
    glued = sorted({tuple(sorted((r, s))) for r in HEXAGON_RAYS
                    for s in HEXAGON_RAYS
                    if r != s and side_key[r] == side_key[s]})
    corners = []
    for key, tri in star_of_origin()["triangles"]:
        corners.append(((key[0], _reduce(key[1], Bi)), tri))
    # cyclic order of corners: the triangle between consecutive rays
    ordered = []
    for i in range(6):
        a, b_ = HEXAGON_RAYS[i], HEXAGON_RAYS[(i + 1) % 6]
        for k, tri in corners:
            if a in tri and b_ in tri:
                ordered.append(k)
    # incidence: which edge orbits bound which triangle orbits
    incid = {}
    for k, tri in tris.items():
        tk = (k[0], _reduce(k[1], Bi))
        for i in range(3):
            p, q = tri[i], tri[(i + 1) % 3]
            dv = _sub(q, p)
            if dv not in EDGE_DIRECTIONS:
                p, dv = q, _neg(dv)
            incid.setdefault((dv, _reduce(p, Bi)), set()).add(tk)
    return {"d": d, "components": len(vkeys), "double_curves": len(ekeys),
            "triple_points": len(tkeys), "glued_sides": glued,
            "corner_classes": ordered,
            "curves_through_triple_points": incid,
            "dual_complex_euler": len(vkeys) - len(ekeys) + len(tkeys)}


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1])


def _neg(a):
    return (-a[0], -a[1])


def euler_W_stratified(B=None):
    r"""e(W) by the torus-orbit stratification (Proposition 4.6 (iv)).

    Each cell orbit contributes one torus orbit of W: (C*)^2 for a vertex,
    C* for an edge, a point for a triangle; e = 0, 0, 1. So e(W) is the
    number of triangle orbits.
    """
    c = cusp_fibre(B)
    return 0 * c["components"] + 0 * c["double_curves"] + c["triple_points"]


def euler_W_normalisation(B=None):
    r"""e(W) from the normalisation, sharing no counting with the strata.

    e(W) = e(W - D) + e(D) = [e(normalisation) - e(its hexagon cycles)]
    + e(D). Each component is a dP6 (e = 6) whose anticanonical hexagon of six
    P^1s meeting in six points has e = 6; D is a union of rational curves,
    and a curve union has e = sum e(C_i) - sum_p (m_p - 1), m_p the number of
    curves through p -- read off the incidence data, not assumed to be 3.
    """
    c = cusp_fibre(B)
    hs = hexagon_surface()
    n_comp = c["components"]
    e_norm = n_comp * hs["euler"]
    e_cycles = n_comp * (6 * 2 - 6)
    through = {}
    for curve, pts in c["curves_through_triple_points"].items():
        for p in pts:
            through[p] = through.get(p, 0) + 1
    e_D = 2 * c["double_curves"] - sum(m - 1 for m in through.values())
    return {"e(normalisation)": e_norm, "e(hexagon cycles)": e_cycles,
            "e(D)": e_D, "e(W)": e_norm - e_cycles + e_D,
            "curves_through_each_point": sorted(set(through.values()))}


def triple_point_formula(B=None):
    r"""Friedman's triple-point formula on every double curve of W.

    For a reduced normal-crossings fibre in a smooth total space (Theorem 4.5
    (d): N0 is smooth, W = f0^*(0) reduced), each double curve C, with
    branches on components V, V', satisfies
        (C|_V)^2 + (C|_{V'})^2 + T_C = 0,
    T_C the number of triple points on C (Friedman 1983; quoted). Here every
    branch is a side of a dP6 hexagon, self-intersection -1 by
    :func:`hexagon_surface`, and T_C is read off the incidence data. The
    formula is not used anywhere in the construction, so it is a test of the
    gluing rather than a restatement of it.
    """
    c = cusp_fibre(B)
    selfint = hexagon_surface()["self_intersections"]
    assert set(selfint) == {-1}
    per = {}
    for curve, pts in c["curves_through_triple_points"].items():
        per[curve] = -1 + -1 + len(pts)
    return {"per_curve": per, "holds": set(per.values()) == {0}}


def det_scan(Bs=None):
    """Remark 4.9: components and e(W) as B0 varies over non-degenerate matrices."""
    Bs = Bs or [B0(), Matrix([[1, 0], [0, 2]]), Matrix([[2, 1], [0, 1]]),
                Matrix([[1, 1], [-1, 2]]), Matrix([[2, 0], [0, 2]]),
                Matrix([[3, 1], [1, 2]])]
    out = []
    for B in Bs:
        c = cusp_fibre(B)
        out.append({"B": B.tolist(), "d": c["d"],
                    "components": c["components"],
                    "double_curves": c["double_curves"],
                    "triple_points": c["triple_points"],
                    "e_strata": euler_W_stratified(B),
                    "e_normalisation": euler_W_normalisation(B)["e(W)"],
                    "dual_complex_euler": c["dual_complex_euler"],
                    "triple_point_formula": triple_point_formula(B)["holds"]})
    return out


def status_cusp():
    """Run every Part III check; pass/fail per item."""
    fc = fan_checks()
    hs = hexagon_surface()
    c = cusp_fibre()
    en = euler_W_normalisation()
    scan = det_scan()
    corners = c["corner_classes"]
    checks = {
        "every cone unimodular (lower +1, upper -1)":
            fc["unimodular"] and fc["dets"] == {"lower": {1}, "upper": {-1}},
        "t = z0 z1 z2 in every chart; Omega extends":
            fc["t = z0 z1 z2 in every chart"]
            and fc["Omega nowhere vanishing"],
        "pi_1 of the toric model trivial": toric_pi1()["trivial"],
        "hexagon rays in the paper's (and Lean's) order":
            hs["matches_paper_and_lean"],
        "star of a vertex is a complete smooth fan":
            set(hs["consecutive_dets"]) == {1},
        "six (-1)-curves": hs["self_intersections"] == [-1] * 6,
        "toric identifies dP6 (its dP3/B3): smooth, degree 6":
            hs["toric_smooth"] and hs["toric_degree"] == 6,
        "Noether on E0: K^2 + e = 12": hs["noether"],
        "|det B0| = 1: one component, 3 double curves, 2 triple points":
            (c["components"], c["double_curves"], c["triple_points"])
            == (1, 3, 2),
        "opposite hexagon sides glued, nothing else":
            c["glued_sides"] == sorted(tuple(sorted((r, _neg(r))))
                                       for r in EDGE_DIRECTIONS),
        "hexagon corners alternate between the two triple points":
            len(set(corners)) == 2 and all(
                corners[i] != corners[(i + 1) % 6] for i in range(6)),
        "every double curve through both triple points":
            all(len(p) == 2
                for p in c["curves_through_triple_points"].values()),
        "e(W) = 2 by strata": euler_W_stratified() == 2,
        "e(W) = 2 from the normalisation": en["e(W)"] == 2,
        "triple-point formula on every double curve":
            triple_point_formula()["holds"],
        "|det B0| = d: d components, 3d curves, 2d points, e = 2d (both ways)":
            all(r["components"] == r["d"] and r["double_curves"] == 3 * r["d"]
                and r["triple_points"] == 2 * r["d"]
                and r["e_strata"] == r["e_normalisation"] == 2 * r["d"]
                for r in scan),
        "dual complex is a torus (V - E + F = 0) for every d":
            all(r["dual_complex_euler"] == 0 for r in scan),
        "triple-point formula for every d":
            all(r["triple_point_formula"] for r in scan),
    }
    return {"checks": checks, "all_pass": all(checks.values())}


# ===========================================================================
# Part IV -- Sections 7 and 9: the topology and the invariants ledger
# ===========================================================================
#
# Section 7 computes the topology of X by van Kampen and Mayer-Vietoris over
# pieces that are torus bundles; Section 9 computes the direct images of O_X,
# the canonical bundle, the Chern numbers, the holomorphic forms and the
# C^*-action. Much of both reduces to arithmetic this package does:
#
#   * H_1(X), by abelianising the *full* presentation of Theorem 7.17 on the
#     six generators (gamma^, u^, w^, delta^, x, y) -- the paper's reduction
#     step (normal closure of Lambda_tor is ker gamma) is not used, so this is
#     a third route to |p|, after the closed form and the Seifert H_1;
#   * the Betti numbers of W and the ranks of H_*(M), from invariants of
#     exterior powers of the cusp monodromy (Propositions 7.11, 7.12, Lemma
#     7.18);
#   * the multiple fibres: the coinvariants at p_j, torsion-freeness of
#     H_1(S_j), and the finite indices of Proposition 7.14 from Gram
#     determinants;
#   * Hirzebruch-Riemann-Roch in dimension three from Chern roots, then the
#     Hodge numbers solved for, leaving exactly one unknown, h^{1,2};
#   * chi(O_X) twice: HRR, and the Leray sequence of the direct images;
#   * K_X by Picard-group arithmetic, including why the classical
#     multiple-fibre formula cannot hold on X (the paper's footnote to
#     Section 10), and the degree of the Hodge bundle from Part II's fibres;
#   * the fixed locus of the C^* action in the toric model, with its weights.
#
# What stays quoted is marked as such: the analytic inputs of Section 9
# (the direct images, h^{p,0} = 0, c3 = e(X), the fixed locus outside the
# cusp), and the topological facts about bielliptic surfaces.

from itertools import combinations as _combinations


def compound(T, q):
    """The q-th exterior power of a 4x4 matrix, in the basis of q-subsets."""
    T = Matrix(T)
    if q == 0:
        return Matrix([[1]])
    idx = list(_combinations(range(T.rows), q))
    return Matrix(len(idx), len(idx),
                  lambda a, b: T.extract(list(idx[a]), list(idx[b])).det())


def _wedge_basis(q):
    return list(_combinations(range(4), q))


def _wvec(terms, q):
    """A q-vector from {subset: coeff}; subsets as index tuples in (g,u,w,d)."""
    basis = _wedge_basis(q)
    v = [0] * len(basis)
    for k, c in terms.items():
        v[basis.index(tuple(k))] += c
    return v


def _top_pairing(a, qa, b, qb):
    """a ^ b as a multiple of vol = gamma u w delta, for a in wedge^qa, b in
    wedge^qb with qa + qb = 4."""
    from sympy.combinatorics import Permutation
    Ba, Bb = _wedge_basis(qa), _wedge_basis(qb)
    tot = 0
    for i, I_ in enumerate(Ba):
        for j, J_ in enumerate(Bb):
            if a[i] == 0 or b[j] == 0 or set(I_) & set(J_):
                continue
            perm = list(I_) + list(J_)
            tot += a[i] * b[j] * Permutation(perm).signature()
    return tot


# -- the fundamental group: a third route ----------------------------------

def h1_from_presentation(v1=None, v2=None, mu=None):
    r"""H_1(X) from the presentation of Theorem 7.17, abelianised directly.

    Generators gamma^, u^, w^, delta^ (the fibre Lambda), x = rho_1, y = rho_2.
    Relations: rho_j lambda rho_j^{-1} = A_j lambda for every basis vector
    (abelianised: (A_j - I) lambda = 0); Lambda_tor = 1 (vanishing cycles at
    the cusp); x^3 = t_{v1}, y^4 = t_{v2}; x y = t_mu (the toric meridian).
    Smith form of the 12 x 6 relation matrix. No use is made of the paper's
    simplification to three generators.
    """
    v1 = EPS if v1 is None else Matrix(v1)
    v2 = -EPS_PRIME if v2 is None else Matrix(v2)
    mu = zeros(4, 1) if mu is None else Matrix(mu)
    m = monodromy()
    rows = []
    for A in (m["A1"], m["A2"]):
        for k in range(4):
            e = zeros(4, 1)
            e[k] = 1
            rows.append(list((A - _I4) * e) + [0, 0])
    rows.append([0, 0, 1, 0, 0, 0])
    rows.append([0, 0, 0, 1, 0, 0])
    rows.append([-x for x in v1] + [3, 0])
    rows.append([-x for x in v2] + [0, 4])
    rows.append([-x for x in mu] + [1, 1])
    f = [abs(int(t)) for t in invariant_factors(Matrix(rows))]
    nz = [t for t in f if t]
    order = 1
    for t in nz:
        order *= t
    return {"invariant_factors": f,
            "order": order if len(nz) == 6 else 0,
            "cyclic": sum(1 for t in nz if t > 1) <= 1}


def pi1_three_routes(box=2):
    """|p| three ways over admissible twists and a range of l0.

    Twists v1 = a eps + b delta^, v2 = c eps' + d delta^ with 3 !| a, c odd;
    mu = l0 gamma^. Also checks gcd(p, 12) = 1 (Theorem 7.17).
    """
    bad, n = [], 0
    for a in range(-box, box + 1):
        if a % 3 == 0:
            continue
        for c in range(-box, box + 1):
            if c % 2 == 0:
                continue
            for b in (0, 1):
                for d in (0, 1):
                    for l0 in range(-box, box + 1):
                        n += 1
                        v1 = a * EPS + b * DELTA_HAT
                        v2 = c * EPS_PRIME + d * DELTA_HAT
                        closed = pi1_order(l0, a, c)
                        seif = seifert_h1(3, 4, -a, -c, l0)["order"]
                        pres = h1_from_presentation(
                            v1, v2, Matrix([l0, 0, 0, 0]))
                        if not (closed == seif == pres["order"]
                                and pres["cyclic"]
                                and _gcd(closed, 12) == 1):
                            bad.append((a, b, c, d, l0))
    return {"checked": n, "disagreements": bad}


# -- the cusp: Betti numbers of W and of the boundary M ---------------------

def invariant_exterior(T, q):
    """Saturated basis of (wedge^q)^T, as rows in the basis of q-subsets."""
    C = compound(T, q)
    if q == 0:
        return Matrix([[1]])
    return int_kernel(C - eye(C.rows))


def cusp_homology():
    r"""Propositions 7.11-7.12 and Lemma 7.18, from the monodromy alone.

    b_q(W) = rank (wedge^q V)^{T0}: (1, 2, 4, 2, 1). The kernel of the
    specialisation, (wedge^q M0 - I) wedge^q Lambda, is saturated of rank
    C(4,q) - b_q. The boundary M of the cusp neighbourhood is a torus bundle
    over a circle with monodromy M0, so by the Wang sequence its ranks are
    inv_{k-1} + coinv_k: (1, 3, 6, 6, 3, 1). And e(W) = sum (-1)^q b_q = 2,
    a third route to Part III's number.
    """
    T0 = monodromy()["T0"]
    M0 = monodromy()["M0"]
    b = [invariant_exterior(T0, q).rows for q in range(5)]
    kers = []
    for q in range(1, 4):
        C = compound(M0, q)
        img = saturate_basis((C - eye(C.rows)).T)
        sat = saturate(img.tolist()) if img.rows else img
        kers.append({"q": q, "rank": img.rows,
                     "saturated": same_lattice(img, sat),
                     "rank=C(4,q)-b_q": img.rows == binom(4, q) - b[q]})
    inv_M0 = [invariant_exterior(M0, q).rows for q in range(5)]
    wang = [(inv_M0[k - 1] if k >= 1 else 0) + (inv_M0[k] if k <= 4 else 0)
            for k in range(6)]
    listed = {
        2: [_wvec({(0, 1): 1}, 2), _wvec({(0, 3): 1}, 2),
            _wvec({(1, 2): 1}, 2), _wvec({(0, 2): 1, (1, 3): -1}, 2)],
        3: [_wvec({(0, 1, 2): 1}, 3), _wvec({(0, 1, 3): 1}, 3)],
        1: [_wvec({(0,): 1}, 1), _wvec({(1,): 1}, 1)],
    }
    matches = {q: same_lattice(invariant_exterior(T0, q), Matrix(listed[q]))
               for q in listed}
    return {"b(W)": b, "euler(W)": sum((-1) ** q * x for q, x in enumerate(b)),
            "specialisation_kernels": kers, "ranks H_*(M)": wang,
            "invariants_as_printed": matches}


def binom(n, k):
    from math import comb
    return comb(n, k)


# -- the multiple fibres ------------------------------------------------------

PSI = {1: Matrix([[0, 2, 1, 3]]), 2: Matrix([[0, 1, 1, 2]])}


def multiple_fibre(j, v=None):
    r"""Lemma 7.13 and Proposition 7.14 at p_j.

    The image (A_j - I) Lambda is ker gamma cap ker psi_j, saturated of rank
    2, so the coinvariants are Z^2 via (gamma, psi_j). H_1(S_j) is presented
    as (Z^2 + Z g)/(m_j g - (gamma(v), psi_j(v))), torsion-free iff
    gcd(l_j, psi_j(v), m_j) = 1. Then the indices of pull-back in the
    invariants: degree 1 by the size of xi -> xi(v) mod m_j; degree 2 from
    the Gram determinant of (wedge^2 V)^{T_j} under the wedge pairing and
    unimodularity of H^2(S_j) (k^2 = m^2/|det|); degree 3 from the pairing
    V^{T_j} x (wedge^3 V)^{T_j} (k1 k3 |det| = m^2). The unimodularity of
    H^2(S_j) and pi_* pi^* = m_j are quoted from the paper.
    """
    m = monodromy()
    A = {1: m["A1"], 2: m["A2"]}[j]
    T = {1: T1, 2: T2}[j]
    mj = {1: 3, 2: 4}[j]
    v = {1: EPS, 2: -EPS_PRIME}[j] if v is None else Matrix(v)
    img = saturate_basis((A - _I4).T)
    ann = Matrix([[1, 0, 0, 0]]).col_join(PSI[j])
    image_ok = (img.rows == 2
                and all((ann * img.row(i).T) == zeros(2, 1)
                        for i in range(img.rows))
                and same_lattice(img, saturate(img.tolist()))
                and same_lattice(img, int_kernel(ann)))
    onto = [abs(int(x)) for x in invariant_factors(ann)] == [1, 1]
    l, ps = gamma(v), int((PSI[j] * v)[0])
    h1 = [abs(int(x)) for x in invariant_factors(Matrix([[l, ps, -mj]]))]
    # degree 1
    VT = invariant_exterior(T, 1)
    image_mod = {int((VT.row(i) * v)[0]) % mj for i in range(VT.rows)}
    gen = set()
    for x in range(mj):
        for y in range(mj):
            gen.add((x * int((VT.row(0) * v)[0])
                     + y * int((VT.row(1) * v)[0])) % mj)
    k1 = len(gen)
    # degree 2
    W2 = invariant_exterior(T, 2)
    G2 = Matrix(W2.rows, W2.rows, lambda a, b: _top_pairing(
        list(W2.row(a)), 2, list(W2.row(b)), 2))
    det2 = G2.det()
    k2sq = Rational(mj ** 2, abs(det2))
    # degree 3
    W3 = invariant_exterior(T, 3)
    P13 = Matrix(VT.rows, W3.rows, lambda a, b: _top_pairing(
        list(VT.row(a)), 1, list(W3.row(b)), 3))
    det13 = P13.det()
    k3 = Rational(mj ** 2, k1 * abs(det13))
    return {"image_is_kerg_cap_kerpsi": image_ok, "coinvariants_Z2": onto,
            "H1(S)_invariant_factors": h1,
            "H1(S)_torsion_free": all(x in (0, 1) for x in h1),
            "k1": k1, "gram2": G2, "det2": det2,
            "k2": (_sqrt_int(k2sq.p) if k2sq.q == 1 and _is_square(k2sq.p)
                   else None),
            "det13": det13, "k3": k3}


def _is_square(n):
    r = int(n ** 0.5 + 0.5)
    if r * r == n:
        return True
    return False


def _sqrt_int(n):
    return int(n ** 0.5 + 0.5)


# -- Hirzebruch-Riemann-Roch in dimension three -----------------------------

def hrr_threefold():
    r"""chi of O, Omega^1, Omega^2, Omega^3 and T as polynomials in c1, c2, c3.

    From Chern roots: ch and td expanded to degree 3 and symmetrised. Then
    specialised to X: c1 = c2 = 0 in H^*(X; Z) (H^2 = H^4 = 0 by Section 7),
    so every monomial containing c1 or c2 vanishes and only c3 survives.
    c3 = e(X) = 2 is Gauss-Bonnet with Section 7 (quoted).
    """
    from sympy import exp as _exp, series as _series, symbols as _sy, Poly
    from sympy.polys.polyfuncs import symmetrize
    x = _sy("x1:4")
    s = _sy("s")
    c1, c2, c3 = _sy("c1 c2 c3")

    def deg3(expr):
        e = _series(expr.subs({xi: s * xi for xi in x}, simultaneous=True),
                    s, 0, 4).removeO()
        return _expand(e.subs(s, 1))

    def td(xi):
        return xi / (1 - _exp(-xi))

    todd = deg3(td(x[0]) * td(x[1]) * td(x[2]))
    bundles = {
        "O": 1,
        "Omega1": sum(_exp(-xi) for xi in x),
        "Omega2": sum(_exp(-x[i] - x[j]) for i, j in ((0, 1), (0, 2), (1, 2))),
        "Omega3": _exp(-x[0] - x[1] - x[2]),
        "T": sum(_exp(xi) for xi in x),
    }
    out = {}
    for name, ch in bundles.items():
        prod = deg3(ch * todd) if ch != 1 else todd
        top = sum(t for t in _expand(prod).as_ordered_terms()
                  if Poly(t, *x).total_degree() == 3)
        # symmetrize(formal=True) returns (expr_in_s, rem, [(s_i, e_i)])
        res = symmetrize(top, *x, formal=True)
        expr, remainder, defs = res[0], res[1], res[2]
        sub = {defs[0][0]: c1, defs[1][0]: c2, defs[2][0]: c3}
        out[name] = _expand(expr.subs(sub))
        assert remainder == 0
    on_X = {k: v.subs({c1: 0, c2: 0}) for k, v in out.items()}
    return {"general": out, "on_X": on_X}


def _h_P1(d, i):
    """h^i(P^1, O(d))."""
    return max(d + 1, 0) if i == 0 else max(-d - 1, 0)


#: Theorem 9.1 (ii), quoted: the direct images of O_X as sums of O(d).
DIRECT_IMAGES = {0: [0], 1: [0, -1], 2: [-1]}


def leray_h0q():
    r"""h^{0,q} from R^j f_* O_X over P^1 (quoted degrees, computed cohomology).

    Over a curve the Leray sequence has two columns and degenerates, so
    h^{0,q} = h^0(R^q) + h^1(R^{q-1}). Also chi(O_X) = sum (-1)^q chi(R^q),
    which HRR computes independently as c1 c2 / 24.
    """
    h = []
    for q in range(4):
        a = sum(_h_P1(d, 0) for d in DIRECT_IMAGES.get(q, []))
        b = sum(_h_P1(d, 1) for d in DIRECT_IMAGES.get(q - 1, []))
        h.append(a + b)
    chi = sum((-1) ** q * sum(_h_P1(d, 0) - _h_P1(d, 1) for d in ds)
              for q, ds in DIRECT_IMAGES.items())
    return {"h0q": h, "chi(O)": chi}


def hodge_numbers():
    r"""Solve for the Hodge numbers of X; one parameter is left.

    Inputs: h^{p,0} = (1, 0, 0, 0) (Theorem 9.1 (vi), quoted), h^{0,q} from
    :func:`leray_h0q`, Serre duality h^{p,q} = h^{3-p,3-q}, and
    chi(Omega^p) = sum_q (-1)^q h^{p,q} from :func:`hrr_threefold`. No Hodge
    symmetry h^{p,q} = h^{q,p}: X is not Kahler, and h^{1,0} = 0 while
    h^{0,1} = 1. The solution has exactly one free parameter, h^{1,2}.
    """
    from sympy import symbols as _sy, linsolve as _ls
    H = {(p, q): _sy("h%d%d" % (p, q)) for p in range(4) for q in range(4)}
    eqs = []
    for p in range(4):
        for q in range(4):
            eqs.append(H[(p, q)] - H[(3 - p, 3 - q)])
    for p, val in enumerate((1, 0, 0, 0)):
        eqs.append(H[(p, 0)] - val)
    for q, val in enumerate(leray_h0q()["h0q"]):
        eqs.append(H[(0, q)] - val)
    chi = hrr_threefold()["on_X"]
    c3 = 2
    from sympy import Symbol
    for p, name in enumerate(("O", "Omega1", "Omega2", "Omega3")):
        val = chi[name].subs(Symbol("c3"), c3)
        eqs.append(sum((-1) ** q * H[(p, q)] for q in range(4)) - val)
    syms = list(H.values())
    sol = list(_ls(eqs, syms))
    assert len(sol) == 1
    sol = dict(zip(syms, sol[0]))
    free = sorted(set().union(*[v.free_symbols for v in sol.values()]),
                  key=str)
    table = {k: sol[s_] for k, s_ in H.items()}
    b = (1, 0, 0, 0, 0, 0, 1)            # Betti numbers of S^6 (Section 7)
    frolicher = {k: sum(table[(p, k - p)] for p in range(4)
                        if 0 <= k - p <= 3) for k in range(7)}
    return {"table": table, "free": free,
            "euler": _expand(sum((-1) ** (p + q) * v
                                 for (p, q), v in table.items())),
            "h11_minus_h12": _expand(table[(1, 1)] - table[(1, 2)]),
            "frolicher_sums": frolicher,
            # E1-degeneration would need b_k = sum h^{p,q}; it fails at k=1
            # whatever h^{1,2} is
            "b1<h01": b[1] < table[(0, 1)]}


def hkp_constraints():
    """The constraints of Huckleberry-Kebekus-Peternell on a complex S^6
    (quoted: [HKP]), evaluated on the computed numbers."""
    h = hodge_numbers()["table"]
    return {"h0(T) = 1 <= 2": True,
            "h01 = h02 + 1": h[(0, 1)] == h[(0, 2)] + 1,
            "H^3(O) = 0": h[(0, 3)] == 0}


# -- the canonical bundle ----------------------------------------------------

def picard_arithmetic():
    r"""K_X in the subgroup of Pic(X) generated by H = f^*O(1), S1, S2.

    Relations 3 S1 = f^*p1 = H, 4 S2 = f^*p2 = H. Smith form: the group is
    Z, generated by u = S1 - S2, with S1 = 4u, S2 = 3u, H = 12u. It embeds in
    Pic(X) = Pic^0(X) = C (torsion-free; quoted from Theorem 9.1 and
    H^1(X;Z) = H^2(X;Z) = 0) because u != 0: H is not trivial, as
    h^0(f^*O(1)) = 2 != 1.

      K_X = -H + 2 S2 = -6u        (Theorem 9.1 (iii), quoted)
      classical  -H + 2 S1 + 3 S2 = 5u
      their difference 2 S1 + S2 = 11u, not in 12 Z u: not a pull-back.

    So the classical multiple-fibre formula, with exponents m_j - 1 = (2, 3),
    cannot hold on X in any normal form; and K_X = -6u is not torsion.
    """
    R = Matrix([[3, 0, -1], [0, 4, -1]])          # rows: 3S1 - H, 4S2 - H
    f = [abs(int(x)) for x in invariant_factors(R)]
    coords = {"S1": 4, "S2": 3, "H": 12}
    # check the coordinates satisfy the relations
    rel_ok = (3 * coords["S1"] == coords["H"]
              and 4 * coords["S2"] == coords["H"])
    K = -coords["H"] + 2 * coords["S2"]
    classical = -coords["H"] + 2 * coords["S1"] + 3 * coords["S2"]
    diff = 2 * coords["S1"] + coords["S2"]
    normal_forms = [(k, a, b) for a in range(3) for b in range(4)
                    for k in range(-3, 3)
                    if 12 * k + 4 * a + 3 * b == K]
    return {"invariant_factors": f, "torsion_free_rank_1": f == [1, 1],
            "coords": coords, "relations_hold": rel_ok,
            "K": K, "classical": classical, "difference": diff,
            "difference_is_pullback": diff % 12 == 0,
            "K_torsion": K == 0,
            "normal_forms": normal_forms}


def hodge_bundle_degree():
    r"""deg f_* omega_{X/B} from the elliptic surface of Part II, three ways.

    (a) Sum of Euler numbers of the Kodaira fibres over 12 (Noether):
        (8 + 3 + 1)/12.
    (b) The paper's local exponents a_j/m_j plus 1/12 at the cusp, with a_j
        the vanishing order of F = E4^2 E6^{1/2}/Delta o tau at z_j: a_1 =
        2 ord_rho(E4) ord_{z1}(tau - rho), a_2 = (1/2) ord_i(E6)
        ord_{z2}(tau - i). ord_rho E4 = ord_i E6 = 1 is classical (quoted);
        ord_{z_j}(tau - tau(z_j)) = m_j / |Stab_{PSL2Z}(tau(z_j))| = 3/3, 4/2.
    (c) Proposition 9.11 gives deg = 1 by an explicit section (quoted).
    """
    w = weierstrass_route()
    e = [KODAIRA_MONODROMY[w["fibres"][p]["type"]][1] for p in (0, 1, "oo")]
    via_euler = Rational(sum(e), 12)
    ord_tau = {1: Rational(3, 3), 2: Rational(4, 2)}
    a1 = 2 * 1 * ord_tau[1]
    a2 = Rational(1, 2) * 1 * ord_tau[2]
    local = [a1 / 3, a2 / 4, Rational(1, 12)]
    return {"euler_numbers": e, "via_euler": via_euler,
            "a": (a1, a2), "local_exponents": local,
            "via_exponents": sum(local),
            "12*exponents": [12 * x for x in local],
            "matches_fibre_euler": [12 * x for x in local] == e}


# -- the C^* action ----------------------------------------------------------

def cstar_fixed_locus(cochar=(0, 1)):
    r"""Fixed strata of the one-parameter subgroup delta^ <-> e2 in the cusp model.

    A torus orbit O_sigma is fixed pointwise by the cocharacter lambda iff
    lambda lies in the linear span of sigma. Among the cells of T_{A2}
    modulo B0 Z^2: the vertex strata never are, the edge stratum of
    direction e2 is, and both triangle strata are -- so the fixed locus in
    the cusp model is the closure of one double curve, through both triple
    points (Proposition 9.24). The weights at a triple point are
    <m_i, lambda> on the chart coordinates: 0 along the curve and (+1, -1)
    normal to it. e(fixed locus) = e(P^1) = 2 = e(X), as localisation
    requires. That nothing outside the cusp is fixed is the paper's (the
    flow is a translation on the torus fibres and the multiple fibres).
    """
    lam = Matrix([cochar[0], cochar[1], 0])
    B = B0()
    Bi = _binv(B)
    verts, edges, tris = a2_patch(2)

    def in_span(rays):
        M = Matrix([list(r) for r in rays]).T
        return M.rank() == M.row_join(lam).rank()

    fixed_edges, fixed_tris, fixed_verts = set(), set(), set()
    for v in verts:
        if in_span([(v[0], v[1], 1)]):
            fixed_verts.add(_reduce(v, Bi))
    for e in edges:
        if in_span([(e[0][0], e[0][1], 1), (e[1][0], e[1][1], 1)]):
            fixed_edges.add((_sub(e[1], e[0]), _reduce(e[0], Bi)))
    weights = set()
    for k, tri in tris.items():
        if in_span([(p[0], p[1], 1) for p in tri]):
            fixed_tris.add((k[0], _reduce(k[1], Bi)))
            dual_b = cone_matrix(tri).inv()
            w = tuple(sorted(int((dual_b[:, i].T * lam)[0]) for i in range(3)))
            weights.add(w)
    return {"fixed_vertex_strata": len(fixed_verts),
            "fixed_edge_strata": sorted(fixed_edges),
            "fixed_triple_points": len(fixed_tris),
            "weights_at_triple_points": sorted(weights),
            # the closure of one edge stratum is a P^1 through both triple
            # points (Part III), so e(fixed locus) = e(P^1)
            "euler_fixed": 2 if (len(fixed_edges) == 1
                                 and len(fixed_tris) == 2) else None}


def status_invariants():
    """Run every Part IV check; pass/fail per item."""
    p3 = pi1_three_routes()
    ch = cusp_homology()
    mf1, mf2 = multiple_fibre(1), multiple_fibre(2)
    hrr = hrr_threefold()
    from sympy import Symbol
    c3 = Symbol("c3")
    onX = {k: v.subs(c3, 2) for k, v in hrr["on_X"].items()}
    ler = leray_h0q()
    hn = hodge_numbers()
    pa = picard_arithmetic()
    hb = hodge_bundle_degree()
    cs = cstar_fixed_locus()
    checks = {
        "|p| three ways; gcd(p, 12) = 1": not p3["disagreements"],
        "H_1(X) = 0 for the paper's twists (presentation)":
            h1_from_presentation()["order"] == 1,
        "b(W) = (1,2,4,2,1) from (wedge^q V)^T0": ch["b(W)"] == [1, 2, 4, 2, 1],
        "e(W) = 2, third route": ch["euler(W)"] == 2,
        "invariants as printed (Prop. 7.12)":
            all(ch["invariants_as_printed"].values()),
        "specialisation kernels saturated of rank C(4,q) - b_q":
            all(k["saturated"] and k["rank=C(4,q)-b_q"]
                for k in ch["specialisation_kernels"]),
        "ranks of H_*(M) = (1,3,6,6,3,1) (Wang)":
            ch["ranks H_*(M)"] == [1, 3, 6, 6, 3, 1],
        "(A_j - I) Lambda = ker gamma cap ker psi_j; coinvariants Z^2":
            all(m["image_is_kerg_cap_kerpsi"] and m["coinvariants_Z2"]
                for m in (mf1, mf2)),
        "H_1(S_j) torsion-free for the paper's twists":
            mf1["H1(S)_torsion_free"] and mf2["H1(S)_torsion_free"],
        "Prop. 7.14 indices k1 = (3, 4)": (mf1["k1"], mf2["k1"]) == (3, 4),
        "Gram determinants (-9, -4), k2 = (1, 2)":
            (mf1["det2"], mf2["det2"]) == (-9, -4)
            and (mf1["k2"], mf2["k2"]) == (1, 2),
        "degree-3 pairing |det| = (3, 2), k3 = (1, 2)":
            (abs(mf1["det13"]), abs(mf2["det13"])) == (3, 2)
            and (mf1["k3"], mf2["k3"]) == (1, 2),
        "HRR: chi(O) = c1c2/24, chi(T) = c3/2 (general)":
            _expand(hrr["general"]["O"] - Symbol("c1") * Symbol("c2") / 24)
            == 0 and hrr["on_X"]["T"] == c3 / 2,
        "on X: chi(O, Omega1, Omega2, Omega3, T) = (0, -1, 1, 0, 1)":
            [onX[k] for k in ("O", "Omega1", "Omega2", "Omega3", "T")]
            == [0, -1, 1, 0, 1],
        "sum (-1)^p chi(Omega^p) = c3 = 2":
            sum((-1) ** p * onX[k] for p, k in enumerate(
                ("O", "Omega1", "Omega2", "Omega3"))) == 2,
        "chi(O) = 0 twice: HRR and Leray": ler["chi(O)"] == 0 == onX["O"],
        "h^{0,q} = (1, 1, 0, 0) from the direct images":
            ler["h0q"] == [1, 1, 0, 0],
        "Hodge numbers: one free parameter; h11 = h12 + 1":
            len(hn["free"]) == 1 and hn["h11_minus_h12"] == 1,
        "Hodge Euler sum = 2 identically": hn["euler"] == 2,
        "Frolicher does not degenerate (b1 = 0 < h01 = 1)": hn["b1<h01"],
        "HKP constraints hold": all(hkp_constraints().values()),
        "Pic subgroup <H, S1, S2> = Z u; S1 = 4u, S2 = 3u, H = 12u":
            pa["torsion_free_rank_1"] and pa["relations_hold"],
        "K_X = -6u, not torsion": pa["K"] == -6 and not pa["K_torsion"],
        "classical exponents (2,3) impossible: 2S1 + S2 = 11u":
            pa["difference"] == 11 and not pa["difference_is_pullback"],
        "unique normal form K = f^*O(-1) + 0 S1 + 2 S2":
            pa["normal_forms"] == [(-1, 0, 2)],
        "deg f_* omega = 1 via Euler numbers and via exponents":
            hb["via_euler"] == 1 == hb["via_exponents"],
        "12 a_j/m_j = Euler numbers of IV*, III; cusp 1":
            hb["matches_fibre_euler"],
        "C^*: one fixed double curve (direction e2), both triple points":
            cs["fixed_vertex_strata"] == 0
            and len(cs["fixed_edge_strata"]) == 1
            and cs["fixed_edge_strata"][0][0] == (0, 1)
            and cs["fixed_triple_points"] == 2,
        "C^* normal weights (+1, -1)":
            cs["weights_at_triple_points"] == [(-1, 0, 1)],
        "e(fixed locus) = 2 = e(X)": cs["euler_fixed"] == 2,
    }
    return {"checks": checks, "all_pass": all(checks.values())}
