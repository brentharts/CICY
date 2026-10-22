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
dilaton enters as an extra factor (:func:`product_speed2`).  Whether the loop (a kinetic-dominated rolling) is
reached from anywhere is a question of moduli stabilisation, which this package does not address.
"""

from fractions import Fraction

import numpy as np
from scipy.integrate import solve_ivp

__all__ = ["triple_numbers", "kahler_potential", "moduli_metric", "moduli_metric_exact", "metric_third_derivative",
           "christoffel", "speed2", "product_speed2", "max_speed2", "exponent_circle", "exponents_from_speed2",
           "dilation_sigma", "speed2_of_dilation", "kretschmann_coefficient", "moduli_geodesic",
           "einstein_scalar_check"]


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


def einstein_scalar_check(d, t0, tdot0, p_direction, t_end=200.0, rtol=1e-12, atol=1e-14):
    """
    The independent route.  The Euler-Lagrange equations of the Bianchi I minisuperspace Lagrangian

        L = (1/2) e^S [ sum_i xdot_i^2 - (sum_i xdot_i)^2 + G_ab(t) tdot^a tdot^b ] ,   S = sum x_i ,  x_i = ln a_i

    (N = 1; the constraint is imposed on the initial data) are derived by sympy from L alone, with no use of the
    algebra above, and integrated.  `p_direction` is the initial exponent vector; its sum must be 1 and its sum of
    squares must equal 1 - |s|^2 for the given moduli velocity, which is what fixes the constraint.
    Starting at t = 0 with t_s = -1, returns the exponents p_i = xdot_i (t - t_s), |s|^2 and the Hamiltonian
    constraint at the end, plus the moduli at the end.
    """
    import sympy as sp
    d = np.asarray(d)
    h = d.shape[0]
    xs = sp.symbols("x0:3")
    ts = sp.symbols("t0:%d" % h, positive=True)
    xd = sp.symbols("xd0:3")
    td = sp.symbols("td0:%d" % h)
    q, qd = list(xs) + list(ts), list(xd) + list(td)
    n = 3 + h
    V = sum(int(d[i, j, k]) * ts[i] * ts[j] * ts[k] for i in range(h) for j in range(h) for k in range(h))
    G = sp.hessian(-sp.log(V), ts) / 2
    A = sp.eye(3) - sp.ones(3, 3)
    M = sp.zeros(n, n)
    M[:3, :3] = sp.exp(sum(xs)) * A
    M[3:, 3:] = sp.exp(sum(xs)) * G
    L = sp.Rational(1, 2) * (sp.Matrix([qd]) * M * sp.Matrix(qd))[0]
    pv = [sp.diff(L, v) for v in qd]
    r = [sp.diff(L, q[a]) - sum(sp.diff(pv[a], q[k]) * qd[k] for k in range(n)) for a in range(n)]
    Mf, rf = sp.lambdify(q + qd, M, "numpy"), sp.lambdify(q + qd, r, "numpy")

    def rhs(_, y):
        Mn = np.array(Mf(*y[:n], *y[n:]), dtype=float)
        rn = np.array(rf(*y[:n], *y[n:]), dtype=float)
        return np.concatenate([y[n:], np.linalg.solve(Mn, rn)])

    y0 = np.concatenate([[0.0, 0.0, 0.0], np.asarray(t0, dtype=float), np.asarray(p_direction, dtype=float),
                         np.asarray(tdot0, dtype=float)])
    sol = solve_ivp(rhs, [0.0, t_end], y0, method="DOP853", rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError("Einstein-scalar integration failed: " + sol.message)
    y = sol.y[:, -1]
    tt, xdot, tdot = y[3:n], y[n:n + 3], y[n + 3:]
    Gn = moduli_metric(d, tt)
    p = xdot * (t_end + 1.0)
    s2 = float(tdot @ Gn @ tdot) * (t_end + 1.0) ** 2
    constraint = float(xdot[0] * xdot[1] + xdot[1] * xdot[2] + xdot[0] * xdot[2] - 0.5 * tdot @ Gn @ tdot)
    return dict(p=p, speed2=s2, constraint=constraint, t=tt)
