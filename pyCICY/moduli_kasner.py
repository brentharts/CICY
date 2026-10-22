r"""
pyCICY.moduli_kasner -- Kasner cosmology driven by the Kahler moduli of a CICY: the exponent budget.

The question.  Take the four-dimensional N = 1 theory from a Calabi-Yau compactification and let its Kahler moduli
roll with no potential (kinetic domination), in a Bianchi I universe  ds^2 = -dt^2 + sum_i a_i^2 dx_i^2.  What
Kasner exponents can the three large directions have, and what does that say about how the internal manifold
moves?  The answer needs only the triple intersection numbers d_ijk that this package computes exactly.

Setting (stated, not derived from a compactification here).  Units M_P = 1, action  R/2 - K_{i jbar} dT^i dTbar^j,
with the classical large-volume Kahler potential  K = -ln( d_ijk x^i x^j x^k ),  x^i = T^i + Tbar^i = 2 t^i,
and the axions Im T^i held at zero.  Because K depends only on T + Tbar, the slice Im T = 0 is totally geodesic, so
this truncation is consistent.  With G_ab = 2 K_{a bbar} the kinetic term is (1/2) G_ab dphi^a dphi^b, and in the
real coordinates t^i

    G_ij(t) = (1/2) d^2 K / dt^i dt^j = -3 kappa_ij / V + (9/2) kappa_i kappa_j / V^2 ,
    V = d_ijk t^i t^j t^k ,  kappa_ij = d_ijk t^k ,  kappa_i = d_ijk t^j t^k .

Derivation of the budget (every step is checked numerically in the tests, by a second route).
  1. A homogeneous scalar has T^i_j = p delta^i_j, so the shear equation is d(V H_i)/dt = 0, V = a1 a2 a3, and the
     moduli follow geodesics of G with  d(V |dphi/dt|)/dt = 0.  Hence H_i = k_i / V, and V' = sum k_i gives
     V = (sum k) (t - t_s), H_i = p_i / (t - t_s) with  sum p_i = 1.
  2. The Hamiltonian constraint  H1H2 + H2H3 + H1H3 = (1/2) G_ab dphi^a dphi^b  becomes
         sum_i p_i^2 = 1 - |s|^2 ,     s^a = (t - t_s) dphi^a/dt = dphi^a / d ln(t - t_s) ,
     with |s|^2 = G_ab s^a s^b conserved along the motion.  The moduli therefore move along a geodesic of G at constant
     speed |s| in the logarithmic time ln(t - t_s).  (|s| = 0 is vacuum Kasner.)
  3. (sum p)^2 - sum p^2 = 2 sum_{i<j} p_i p_j >= 0 and sum p^2 >= 1/3 for sum p = 1, so real exponents exist iff
         0 <= |s|^2 <= 2/3 ,
     with equality at p = (1/3, 1/3, 1/3), the isotropic stiff-fluid universe a ~ t^(1/3).  For fixed |s| the exponents
     lie on a circle of radius^2 = 2/3 - |s|^2 about (1/3, 1/3, 1/3) in the plane sum p = 1.
  4. The dilation xi = t^i d/dt^i is a Killing field of G with constant norm, G(xi, xi) = 3/2 (K shifts by a constant
     under t -> lambda t, and t^i t^j K_ij = -6 + 9 = 3 for every t).  Integral curves of a Killing field of
     constant norm are geodesics, so the rays t(u) = lambda(u) t_0 are exact solutions, and with
     lambda ~ (t - t_s)^sigma:   |s|^2 = (3/2) sigma^2 ,  so  |sigma| <= 2/3.
     In words: a kinetic-dominated Calabi-Yau cannot grow its linear size faster than (t - t_s)^(2/3); the isotropic
     value saturates it, and then the volume V_CY grows like (t - t_s)^2.

What is exact and what is not.  The metric, its third derivative, the budget, the dilation statement and the closed
form of the Kretschmann scalar are exact (the metric also in rational arithmetic).  Geodesics of G for several moduli
are numerical.  The truncation is not exact physics: it keeps the classical large-volume K with no alpha'
corrections, no bundle or complex-structure moduli, no potential and no axions; and, for heterotic models, the
dilaton enters as an extra product factor (:func:`product_speed2`, :func:`dilaton_speed2`, with its own budget
sigma_S^2 <= 4/3 and a link to the racetrack of :mod:`pyCICY.theories.moduli` in :func:`racetrack_kinetic_budget`).  Whether the loop (a kinetic-dominated rolling) is
reached from anywhere is a question of moduli stabilisation, which this package does not address.
"""

from fractions import Fraction

import numpy as np
from scipy.integrate import solve_ivp

__all__ = ["triple_numbers", "kahler_potential", "moduli_metric", "moduli_metric_exact", "metric_third_derivative",
           "christoffel", "speed2", "product_speed2", "max_speed2", "exponent_circle", "exponents_from_speed2",
           "dilation_sigma", "speed2_of_dilation", "kretschmann_coefficient", "moduli_geodesic",
           "einstein_scalar_check", "dilaton_speed2", "dilaton_sigma_max2", "joint_dilation_norm2",
           "joint_sigma_max2", "alpha_gut_ratio", "qcd_scale_ratio", "racetrack_kinetic_budget"]


def triple_numbers(X):
    """The integer triple intersection numbers d_ijk of a CICY (or configuration matrix) as an int array."""
    from .pyCICY import CICY
    XX = X if isinstance(X, CICY) else CICY(np.asarray(X, dtype=int).tolist())
    d = np.asarray(XX.triple_intersection(), dtype=float)
    if not np.allclose(d, np.round(d)):
        raise ValueError("non-integer intersection number")
    return np.round(d).astype(int)


def _kappa(d, t):
    t = np.asarray(t, dtype=float)
    V = float(np.einsum("ijk,i,j,k", d, t, t, t))
    return V, np.einsum("ijk,j,k->i", d, t, t), np.einsum("ijk,k->ij", d, t)


def kahler_potential(d, t):
    """K = -ln(d_ijk t^i t^j t^k) up to an additive constant; needs a positive volume."""
    V = _kappa(d, t)[0]
    if V <= 0:
        raise ValueError("the cubic form d t t t must be positive (inside the Kahler cone)")
    return -np.log(V)


def moduli_metric(d, t):
    """G_ij = -3 kappa_ij / V + (9/2) kappa_i kappa_j / V^2, the metric of the kinetic term (1/2) G dt dt."""
    V, ka, kij = _kappa(d, t)
    if V <= 0:
        raise ValueError("the cubic form d t t t must be positive (inside the Kahler cone)")
    return -3.0 * kij / V + 4.5 * np.outer(ka, ka) / V ** 2


def moduli_metric_exact(d, t):
    """The same G_ij as nested lists of Fractions, for rational t."""
    n = len(t)
    t = [Fraction(v) for v in t]
    V = sum(int(d[i, j, k]) * t[i] * t[j] * t[k] for i in range(n) for j in range(n) for k in range(n))
    ka = [sum(int(d[i, j, k]) * t[j] * t[k] for j in range(n) for k in range(n)) for i in range(n)]
    kij = [[sum(int(d[i, j, k]) * t[k] for k in range(n)) for j in range(n)] for i in range(n)]
    return [[-3 * kij[i][j] / V + Fraction(9, 2) * ka[i] * ka[j] / V ** 2 for j in range(n)] for i in range(n)]


def metric_third_derivative(d, t):
    """K_ijk = d^3 K / dt^i dt^j dt^k, closed form (checked against sympy in the tests)."""
    V, ka, kij = _kappa(d, t)
    return (-6.0 * d / V
            + 18.0 * (np.einsum("ij,k->ijk", kij, ka) + np.einsum("ik,j->ijk", kij, ka)
                      + np.einsum("jk,i->ijk", kij, ka)) / V ** 2
            - 54.0 * np.einsum("i,j,k->ijk", ka, ka, ka) / V ** 3)


def christoffel(d, t):
    """
    Gamma^a_bc of G.  G = K_ab / 2 is a Hessian metric, so Gamma_{d,bc} = K_bcd / 4 and
    Gamma^a_bc = (1/2) K^{ad} K_bcd with K^{ad} the inverse of K_ab = 2 G_ab.
    """
    G = moduli_metric(d, t)
    return 0.5 * np.einsum("ad,bcd->abc", np.linalg.inv(2.0 * G), metric_third_derivative(d, t))


def speed2(d, t, s):
    """|s|^2 = G_ab s^a s^b for the rate s^a = dt^a / d ln(t - t_s) of the moduli."""
    s = np.asarray(s, dtype=float)
    return float(s @ moduli_metric(d, t) @ s)


def product_speed2(speed2_kahler, dln_sdlnT=0.0):
    """
    With the heterotic dilaton, K gains -ln(S + Sbar) and the metric a factor G_SS = 1 / (2 s^2) (s = Re S), so
    |s|^2 = |s_Kahler|^2 + (1/2) (d ln s / d ln(t - t_s))^2.  Exact; the budget still reads |s|^2 <= 2/3.
    """
    return speed2_kahler + 0.5 * dln_sdlnT ** 2


def max_speed2():
    """The budget: real Kasner exponents exist iff |s|^2 <= 2/3 (exact)."""
    return Fraction(2, 3)


def exponent_circle(sp2):
    """radius^2 = 2/3 - |s|^2 of the circle of exponents about (1/3, 1/3, 1/3); raises if |s|^2 > 2/3."""
    r2 = Fraction(2, 3) - (sp2 if isinstance(sp2, Fraction) else Fraction(sp2).limit_denominator(10 ** 12))
    if r2 < 0:
        raise ValueError("|s|^2 = %s exceeds 2/3: no real Kasner exponents with this much moduli kinetic energy" % sp2)
    return r2


def exponents_from_speed2(sp2, theta=0.0):
    """
    The Kasner exponents p_i = 1/3 + sqrt(2/3 - |s|^2) sqrt(2/3) cos(theta + 2 pi i / 3), i = 0, 1, 2: both
    sum p = 1 and sum p^2 = 1 - |s|^2 hold identically.  theta selects the point on the circle.
    """
    r = float(exponent_circle(sp2)) ** 0.5
    return tuple(1.0 / 3.0 + (2.0 / 3.0) ** 0.5 * r * np.cos(theta + 2.0 * np.pi * i / 3.0) for i in range(3))


def dilation_sigma(sp2):
    """sigma >= 0 with lambda ~ (t - t_s)^sigma for a dilation ray of speed |s|^2: sigma = sqrt(2 |s|^2 / 3)."""
    return float(np.sqrt(2.0 * sp2 / 3.0))


def speed2_of_dilation(sigma):
    """|s|^2 = (3/2) sigma^2 for lambda ~ (t - t_s)^sigma (G(xi, xi) = 3/2).  |sigma| <= 2/3 by the budget."""
    return Fraction(3, 2) * Fraction(sigma) ** 2 if isinstance(sigma, (int, Fraction)) else 1.5 * sigma ** 2


def kretschmann_coefficient(p):
    """
    K (t - t_s)^4 for H_i = p_i / (t - t_s):  4 sum_i p_i^2 (p_i - 1)^2 + 4 sum_{i<j} p_i^2 p_j^2.
    Equals -16 p1 p2 p3 when sum p^2 = 1 (vacuum); checked against a symbolic Riemann tensor in the tests.
    """
    p = [Fraction(x) if isinstance(x, (int, Fraction)) else x for x in p]
    return (4 * sum(x * x * (x - 1) ** 2 for x in p)
            + 4 * sum(p[i] ** 2 * p[j] ** 2 for i in range(3) for j in range(i + 1, 3)))


# ---------------------------------------------------------------------------------------------------------------------
#  The heterotic dilaton: a product factor, and what it does to the budget
# ---------------------------------------------------------------------------------------------------------------------
#  K = -ln(S + Sbar) - ln(d x x x).  The dilaton part, as a function of s = Re S, is -ln s up to a constant, so its
#  metric is G_ss = (1/2) d^2(-ln s)/ds^2 = 1/(2 s^2): flat in ln s, with line element (1/2) (d ln s)^2.  The total
#  space is a Riemannian PRODUCT, so geodesics factorise: the Kahler speed |s_K|^2 and the dilaton rate
#  sigma_S = d ln s / d ln(t - t_s) are separately conserved, and
#       |s|^2 = |s_K|^2 + sigma_S^2 / 2 <= 2/3 .
#  The dilation xi = t^i d_i + s d_s is still Killing, now with G(xi, xi) = 3/2 + 1/2 = 2, so a joint rescaling
#  lambda ~ (t - t_s)^sigma needs 2 sigma^2 <= 2/3, i.e. sigma^2 <= 1/3; the dilaton alone has sigma_S^2 <= 4/3.

def dilaton_speed2(sigma_s):
    """|s|^2 contributed by a dilaton rolling as Re S ~ (t - t_s)^sigma_s: sigma_s^2 / 2 (exact)."""
    return Fraction(sigma_s) ** 2 / 2 if isinstance(sigma_s, (int, Fraction)) else 0.5 * sigma_s ** 2


def dilaton_sigma_max2():
    """sigma_S^2 <= 4/3 with the Kahler moduli at rest: the largest the dilaton exponent can be, squared (exact)."""
    return Fraction(4, 3)


def joint_dilation_norm2():
    """G(xi, xi) for the joint dilation (Kahler moduli and dilaton scaled together): 3/2 + 1/2 = 2."""
    return Fraction(2)


def joint_sigma_max2():
    """sigma^2 <= 1/3 for a joint rescaling of the Kahler moduli and Re S: 2 sigma^2 <= 2/3 (exact)."""
    return Fraction(1, 3)


def alpha_gut_ratio(sigma_s, F):
    """alpha_GUT(F) / alpha_GUT(1) = F^(-sigma_s) when Re S ~ (t - t_s)^sigma_s grows by a factor F (alpha = 1/(4 pi Re S))."""
    return float(F) ** (-float(sigma_s))


def qcd_scale_ratio(re_s0, sigma_s, F, b3_abs=3.0):
    """
    Lambda_QCD(F) / Lambda_QCD(1) at one loop, exact in the model.  With alpha = 1/(4 pi s) the exponent is
    2 pi / (|b_3| alpha) = 8 pi^2 s / |b_3|, so  Lambda ~ exp(-8 pi^2 s / |b_3|)  and, for s = s0 F^sigma_s,
        ratio = exp(-8 pi^2 s0 (F^sigma_s - 1) / |b_3|).
    Its logarithmic derivative in ln s is the sensitivity of running.sensitivity (about 52 at alpha = 1/25, b_3 = -3).
    """
    return float(np.exp(-8.0 * np.pi ** 2 * re_s0 * (float(F) ** float(sigma_s) - 1.0) / b3_abs))


def racetrack_kinetic_budget(N1, N2, ratio=10.0, F=2.0, b3_abs=3.0):
    """
    A THOUGHT EXPERIMENT, not dynamics.  Take Re S at the racetrack minimum of pyCICY.theories.moduli (a statement about
    a potential) and ask how far a dilaton with NO potential could move it if the Kahler moduli were at rest and the
    scale (t - t_s) grew by a factor F: the budget allows sigma_S up to +-2/sqrt(3).  Returns the extreme alpha_GUT and
    Lambda_QCD ratios.  The racetrack is what pins sigma_S = 0, which is why a stabilised dilaton leaves the Kahler
    budget |s_K|^2 <= 2/3 untouched; the numbers here say how much a failure of stabilisation could do, not that it
    happens.  The condensation ratio is an INPUT (see racetrack_dilaton), as is |b_3|.
    """
    from .theories.moduli import racetrack_dilaton
    rt = racetrack_dilaton(N1, N2, ratio)
    smax = float(dilaton_sigma_max2()) ** 0.5
    out = dict(re_s0=rt["re_s"], alpha_gut0=rt["alpha_gut"], sigma_max=smax, F=F)
    for tag, sg in (("up", smax), ("down", -smax)):
        out["alpha_ratio_" + tag] = alpha_gut_ratio(sg, F)
        out["lambda_ratio_" + tag] = qcd_scale_ratio(rt["re_s"], sg, F, b3_abs)
    return out


def moduli_geodesic(d, t0, s0, u_span, n=200, rtol=1e-12, atol=1e-14):
    """
    The moduli trajectory phi(u), u = ln(t - t_s), a geodesic of G from t0 with velocity s0 = dphi/du, numerical.
    Returns dict(u, t (n, h), s (n, h), speed2 (n,)): speed2 is conserved, which is the check.
    """
    t0 = np.asarray(t0, dtype=float)
    s0 = np.asarray(s0, dtype=float)
    h = len(t0)

    def rhs(u, y):
        tt, ss = y[:h], y[h:]
        return np.concatenate([ss, -np.einsum("abc,b,c->a", christoffel(d, tt), ss, ss)])

    u = np.linspace(u_span[0], u_span[1], n)
    sol = solve_ivp(rhs, [u[0], u[-1]], np.concatenate([t0, s0]), t_eval=u, method="DOP853", rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError("geodesic integration failed: " + sol.message)
    tt, ss = sol.y[:h].T, sol.y[h:].T
    return dict(u=u, t=tt, s=ss, speed2=np.array([speed2(d, tt[i], ss[i]) for i in range(n)]))


def einstein_scalar_check(d, t0, tdot0, p_direction, t_end=200.0, rtol=1e-12, atol=1e-14, dilaton=None):
    """
    The independent route.  The Euler-Lagrange equations of the Bianchi I minisuperspace Lagrangian

        L = (1/2) e^S [ sum_i xdot_i^2 - (sum_i xdot_i)^2 + G_ab(t) tdot^a tdot^b (+ sdot^2 / (2 s^2)) ] ,
        S = sum x_i ,  x_i = ln a_i

    (N = 1; the constraint is imposed on the initial data) are derived by sympy from L alone, with no use of the
    algebra above, and integrated.  `p_direction` is the initial exponent vector; its sum must be 1 and its sum of
    squares must equal 1 - |s|^2 for the given moduli velocity, which is what fixes the constraint.
    `dilaton = (s0, sdot0)` adds the heterotic dilaton Re S = s with the product metric 1/(2 s^2); then |s|^2 includes
    the dilaton rate.  Starting at t = 0 with t_s = -1, returns the exponents p_i = xdot_i (t - t_s), |s|^2, the
    Hamiltonian constraint at the end and the moduli (and dilaton, sigma_s = d ln s / d ln(t - t_s)) at the end.
    """
    import sympy as sp
    d = np.asarray(d)
    h = d.shape[0]
    nd = 0 if dilaton is None else 1
    xs = sp.symbols("x0:3")
    ts = sp.symbols("t0:%d" % h, positive=True)
    xd = sp.symbols("xd0:3")
    td = sp.symbols("td0:%d" % h)
    ss = sp.symbols("sdil", positive=True)
    sdd = sp.symbols("sdild")
    q, qd = list(xs) + list(ts) + ([ss] if nd else []), list(xd) + list(td) + ([sdd] if nd else [])
    n = 3 + h + nd
    V = sum(int(d[i, j, k]) * ts[i] * ts[j] * ts[k] for i in range(h) for j in range(h) for k in range(h))
    G = sp.hessian(-sp.log(V), ts) / 2
    A = sp.eye(3) - sp.ones(3, 3)
    M = sp.zeros(n, n)
    M[:3, :3] = sp.exp(sum(xs)) * A
    M[3:3 + h, 3:3 + h] = sp.exp(sum(xs)) * G
    if nd:
        M[n - 1, n - 1] = sp.exp(sum(xs)) / (2 * ss ** 2)
    L = sp.Rational(1, 2) * (sp.Matrix([qd]) * M * sp.Matrix(qd))[0]
    pv = [sp.diff(L, v) for v in qd]
    r = [sp.diff(L, q[a]) - sum(sp.diff(pv[a], q[k]) * qd[k] for k in range(n)) for a in range(n)]
    Mf, rf = sp.lambdify(q + qd, M, "numpy"), sp.lambdify(q + qd, r, "numpy")

    def rhs(_, y):
        Mn = np.array(Mf(*y[:n], *y[n:]), dtype=float)
        rn = np.array(rf(*y[:n], *y[n:]), dtype=float)
        return np.concatenate([y[n:], np.linalg.solve(Mn, rn)])

    q0 = [0.0, 0.0, 0.0] + list(np.asarray(t0, dtype=float)) + ([float(dilaton[0])] if nd else [])
    v0 = list(np.asarray(p_direction, dtype=float)) + list(np.asarray(tdot0, dtype=float)) + \
        ([float(dilaton[1])] if nd else [])
    sol = solve_ivp(rhs, [0.0, t_end], np.array(q0 + v0), method="DOP853", rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError("Einstein-scalar integration failed: " + sol.message)
    y = sol.y[:, -1]
    tt, xdot, tdot = y[3:3 + h], y[n:n + 3], y[n + 3:n + 3 + h]
    Gn = moduli_metric(d, tt)
    clock = t_end + 1.0
    kin = float(tdot @ Gn @ tdot)
    out = dict(t=tt)
    if nd:
        s_end, sdot_end = y[n - 1], y[2 * n - 1]
        kin_d = float(sdot_end ** 2 / (2 * s_end ** 2))
        out.update(s=s_end, sigma_s=float(sdot_end / s_end * clock))
        kin += kin_d
    out["p"] = xdot * clock
    out["speed2"] = kin * clock ** 2
    out["constraint"] = float(xdot[0] * xdot[1] + xdot[1] * xdot[2] + xdot[0] * xdot[2] - 0.5 * kin)
    return out
