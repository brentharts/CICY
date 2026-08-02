r"""
pyCICY.enumerative -- Hilbert series, Chern polynomials and Gopakumar--Vafa
invariants.

These are the self-contained topological quantities of section 2 of

    L. B. Anderson, A. Constantin, J. Gray, Y.-H. He, S.-J. Lee, A. Lukas,
    "CIPro Package: Complete Intersections in Products of Projective Spaces
    and Line Bundles", arXiv:2606.27588.

Each is checkable against a published value, and each is checked in
tests/test_enumerative.py rather than merely computed.

Chern polynomial
----------------
From the tangent sequence 0 -> TX -> TA|_X -> E|_X -> 0 with
E = sum_a O_A(q_a),

    c(TX) = c(TA) / c(E)
          = prod_i (1 + J_i)^{n_i+1} / prod_a (1 + q_a . J),

truncated by J_i^{n_i+1} = 0 and by the dimension of X.

Hilbert series
--------------
The Koszul resolution of the structure sheaf by the defining equations gives
the multigraded Hilbert series of the coordinate ring as

    H(t) = prod_a (1 - t^{q_a}) / prod_i (1 - t_i)^{n_i+1},
    t^{q_a} = prod_i t_i^{q^i_a} .

Gopakumar--Vafa invariants
--------------------------
Implemented for the one-parameter models only: the five Calabi--Yau
threefolds that are complete intersections in a single projective space. The
route is the standard mirror-symmetry computation of

    S. Hosono, A. Klemm, S. Theisen, S.-T. Yau,
    "Mirror symmetry, mirror map and applications to Calabi-Yau hypersurfaces",
    Commun. Math. Phys. 167 (1995) 301, arXiv:hep-th/9308122;
    "Mirror symmetry, mirror map and applications to complete intersection
    Calabi-Yau spaces", Nucl. Phys. B433 (1995) 501, arXiv:hep-th/9406055.

CIPro obtains its GV invariants from code by Albrecht Klemm based on the same
references; this is an independent implementation, not a port, and it is
restricted to the one-parameter case. See :func:`gv_invariants` for what is
and is not covered.

All series arithmetic is done over exact rationals. That matters: the
Gopakumar--Vafa invariants must come out integral, and integrality is a
strong check that would be masked by floating point.
"""

from fractions import Fraction
from math import factorial

from .transitions import _as_matrix, dimensions, is_calabi_yau

__all__ = [
    "chern_polynomial", "chern_polynomial_str", "euler_from_chern",
    "hilbert_series", "hilbert_series_str", "hilbert_coefficients",
    "gv_invariants", "ONE_PARAMETER",
]


# The five Calabi-Yau threefolds that are complete intersections in a single
# projective space, i.e. the one-parameter models. Keys are the pyCICY
# configuration; values are the conventional name.
ONE_PARAMETER = {
    ((4, 5),): "P^4[5]",
    ((5, 3, 3),): "P^5[3,3]",
    ((5, 2, 4),): "P^5[2,4]",
    ((6, 2, 2, 3),): "P^6[2,2,3]",
    ((7, 2, 2, 2, 2),): "P^7[2,2,2,2]",
}


# ------------------------------------------------------- polynomial helpers

def _poly_mul(dims, a, b, max_total):
    """Multiply two polynomials in the J_i, truncating by J_i^{n_i+1} = 0."""
    out = {}
    for ea, ca in a.items():
        for eb, cb in b.items():
            if sum(ea) + sum(eb) > max_total:
                continue
            e = tuple(x + y for x, y in zip(ea, eb))
            if any(e[i] > dims[i] for i in range(len(dims))):
                continue
            out[e] = out.get(e, Fraction(0)) + ca * cb
    return {e: c for e, c in out.items() if c != 0}


def _poly_inv(dims, a, max_total):
    """Invert a polynomial with constant term 1."""
    zero = tuple(0 for _ in dims)
    if a.get(zero, Fraction(0)) != 1:
        raise ValueError("can only invert a series with constant term 1")
    rest = {e: c for e, c in a.items() if e != zero}
    # (1 + r)^{-1} = 1 - r + r^2 - ...
    out = {zero: Fraction(1)}
    term = {zero: Fraction(1)}
    for k in range(1, max_total + 1):
        term = _poly_mul(dims, term, rest, max_total)
        if not term:
            break
        sign = -1 if k % 2 else 1
        for e, c in term.items():
            out[e] = out.get(e, Fraction(0)) + sign * c
    return {e: c for e, c in out.items() if c != 0}


# ------------------------------------------------------- Chern polynomial

def chern_polynomial(conf, max_degree=None):
    r"""Total Chern class of the tangent bundle as a polynomial in the J_i.

    Computes c(TX) = prod_i (1+J_i)^{n_i+1} / prod_a (1 + q_a . J), truncated
    by J_i^{n_i+1} = 0 and by ``max_degree`` (the dimension of X by default).

    Returns
    -------
    dict
        ``{exponent tuple: Fraction}``. The entry with all exponents zero is
        the constant 1; total degree d entries are the components of c_d.

    Example
    -------
    The dP2 surface of CIPro section 2.3, which returns
    ``1 + J1 + J2 + 2 J1 J2 + J3 + 2 J1 J3 + J2 J3``. Here index 0 is the
    P^2 and indices 1, 2 the two P^1 factors.

    >>> c = chern_polynomial([[2, 1, 1], [1, 1, 0], [1, 0, 1]])
    >>> c[(1, 1, 0)], c[(1, 0, 1)], c[(0, 1, 1)]
    (Fraction(2, 1), Fraction(2, 1), Fraction(1, 1))
    """
    M = _as_matrix(conf)
    m, kk = M.shape[0], M.shape[1] - 1
    dims = [int(M[i, 0]) for i in range(m)]
    dim_x = dimensions(M)[2]
    top = dim_x if max_degree is None else max_degree
    top = max(0, min(top, sum(dims)))

    zero = tuple(0 for _ in range(m))
    numer = {zero: Fraction(1)}
    for i in range(m):
        factor = {zero: Fraction(1)}
        e = [0] * m
        e[i] = 1
        factor[tuple(e)] = Fraction(1)
        for _ in range(dims[i] + 1):
            numer = _poly_mul(dims, numer, factor, top)

    denom = {zero: Fraction(1)}
    for c in range(kk):
        factor = {zero: Fraction(1)}
        for i in range(m):
            q = int(M[i, 1 + c])
            if q:
                e = [0] * m
                e[i] = 1
                factor[tuple(e)] = Fraction(q)
        denom = _poly_mul(dims, denom, factor, top)

    return _poly_mul(dims, numer, _poly_inv(dims, denom, top), top)


def chern_polynomial_str(poly, names=None):
    """Render a Chern polynomial, ordered by total degree."""
    if not poly:
        return "0"
    m = len(next(iter(poly)))
    names = names or ["J%d" % (i + 1) for i in range(m)]
    terms = []
    for e in sorted(poly, key=lambda e: (sum(e), e)):
        c = poly[e]
        mono = []
        for i, k in enumerate(e):
            if k == 1:
                mono.append(names[i])
            elif k > 1:
                mono.append("%s^%d" % (names[i], k))
        body = " ".join(mono)
        if not body:
            terms.append(str(c if c.denominator != 1 else c.numerator))
            continue
        if c == 1:
            terms.append(body)
        elif c == -1:
            terms.append("-" + body)
        else:
            cc = c if c.denominator != 1 else c.numerator
            terms.append("%s %s" % (cc, body))
    return " + ".join(terms).replace("+ -", "- ")


def euler_from_chern(conf):
    r"""Euler characteristic by integrating the top Chern class.

    For a threefold, chi = int_X c_3(TX) = int_A c_3(TX) prod_a (q_a . J),
    picking out the coefficient of prod_i J_i^{n_i}.

    This is an independent route to a quantity pyCICY already computes from
    triple intersection numbers, so the two can be compared.

    >>> euler_from_chern([[4, 5]])
    -200
    """
    M = _as_matrix(conf)
    m = M.shape[0]
    kk = M.shape[1] - 1
    dims = [int(M[i, 0]) for i in range(m)]
    dim_x = dimensions(M)[2]

    top = sum(dims)
    poly = chern_polynomial(M, max_degree=top)
    c_top = {e: c for e, c in poly.items() if sum(e) == dim_x}

    for c in range(kk):
        factor = {}
        for i in range(m):
            q = int(M[i, 1 + c])
            if q:
                e = [0] * m
                e[i] = 1
                factor[tuple(e)] = Fraction(q)
        c_top = _poly_mul(dims, c_top, factor, top)

    value = c_top.get(tuple(dims), Fraction(0))
    if value.denominator != 1:
        raise ValueError("non-integral Euler characteristic %s" % value)
    return int(value)


# --------------------------------------------------------- Hilbert series

def hilbert_series(conf):
    r"""Multigraded Hilbert series of the coordinate ring.

    Returns the numerator and denominator of

        H(t) = prod_a (1 - t^{q_a}) / prod_i (1 - t_i)^{n_i+1}

    without expanding, since the factored form is the useful one.

    Returns
    -------
    dict
        ``numerator``: list of multidegrees q_a, one per defining equation,
        each standing for a factor ``(1 - t^{q_a})``.
        ``denominator``: list of exponents n_i + 1, one per projective
        factor, each standing for ``(1 - t_i)^{n_i+1}``.

    Example
    -------
    CICY 7821 of CIPro section 2.4, whose Hilbert series is published as
    ``(1 - t1 t2)(1 - t1 t2^2)^2 / ((1 - t1)^3 (1 - t2)^5)``.

    >>> h = hilbert_series([[2, 1, 1, 1], [4, 2, 2, 1]])
    >>> h["denominator"]
    [3, 5]
    >>> sorted(h["numerator"])
    [(1, 1), (1, 2), (1, 2)]
    """
    M = _as_matrix(conf)
    m, kk = M.shape[0], M.shape[1] - 1
    return {
        "numerator": [tuple(int(M[i, 1 + c]) for i in range(m))
                      for c in range(kk)],
        "denominator": [int(M[i, 0]) + 1 for i in range(m)],
    }


def hilbert_series_str(conf, names=None):
    """Render the Hilbert series in the factored form CIPro prints.

    >>> hilbert_series_str([[2, 1, 1, 1], [4, 2, 2, 1]])
    '(1 - t1 t2)(1 - t1 t2^2)^2 / ((1 - t1)^3 (1 - t2)^5)'
    """
    h = hilbert_series(conf)
    m = len(h["denominator"])
    names = names or ["t%d" % (i + 1) for i in range(m)]

    def mono(q):
        parts = []
        for i, k in enumerate(q):
            if k == 1:
                parts.append(names[i])
            elif k > 1:
                parts.append("%s^%d" % (names[i], k))
        return " ".join(parts) if parts else "1"

    counts = {}
    for q in h["numerator"]:
        counts[q] = counts.get(q, 0) + 1
    num = ""
    for q in sorted(counts):
        power = counts[q]
        num += "(1 - %s)%s" % (mono(q), "^%d" % power if power > 1 else "")

    den = " ".join("(1 - %s)^%d" % (names[i], h["denominator"][i])
                   for i in range(m))
    return "%s / (%s)" % (num or "1", den)


def hilbert_coefficients(conf, order):
    """Expand the Hilbert series to a given total order in the t_i.

    Returns ``{multidegree: coefficient}``, the dimension of the graded piece
    of the coordinate ring in each multidegree. Useful as a cross-check: the
    coefficient in degree q_a counts the polynomials available to write down
    the corresponding defining equation.
    """
    h = hilbert_series(conf)
    m = len(h["denominator"])

    # 1/(1-t_i)^{e} = sum_k binom(k+e-1, e-1) t_i^k
    series = {tuple(0 for _ in range(m)): 1}
    for i, e in enumerate(h["denominator"]):
        factor = {}
        for k in range(order + 1):
            key = [0] * m
            key[i] = k
            num = 1
            for j in range(e - 1):
                num = num * (k + e - 1 - j)
            factor[tuple(key)] = num // factorial(e - 1) if e > 1 else 1
        series = _series_mul(series, factor, order)

    for q in h["numerator"]:
        factor = {tuple(0 for _ in range(m)): 1}
        if sum(q) <= order:
            factor[tuple(q)] = -1
        series = _series_mul(series, factor, order)

    return {e: c for e, c in series.items() if c != 0}


def _series_mul(a, b, order):
    out = {}
    for ea, ca in a.items():
        for eb, cb in b.items():
            e = tuple(x + y for x, y in zip(ea, eb))
            if sum(e) > order:
                continue
            out[e] = out.get(e, 0) + ca * cb
    return out


# ------------------------------------------------- Gopakumar--Vafa invariants

def _t_mul(a, b, M):
    out = [Fraction(0)] * (M + 1)
    for i, x in enumerate(a):
        if x == 0:
            continue
        for j, y in enumerate(b):
            if i + j > M:
                break
            if y:
                out[i + j] += x * y
    return out


def _t_inv(a, M):
    if a[0] == 0:
        raise ValueError("series is not invertible")
    out = [Fraction(0)] * (M + 1)
    out[0] = 1 / a[0]
    for n in range(1, M + 1):
        s = Fraction(0)
        for k in range(1, n + 1):
            if k < len(a):
                s += a[k] * out[n - k]
        out[n] = -s / a[0]
    return out


def _t_exp(a, M):
    if a[0] != 0:
        raise ValueError("exp needs a series with zero constant term")
    out = [Fraction(0)] * (M + 1)
    out[0] = Fraction(1)
    for n in range(1, M + 1):
        s = Fraction(0)
        for k in range(1, n + 1):
            s += Fraction(k) * a[k] * out[n - k]
        out[n] = s / n
    return out


def _harmonic(n):
    return sum(Fraction(1, i) for i in range(1, n + 1))


def gv_invariants(conf, max_degree=6):
    r"""Genus-zero Gopakumar--Vafa invariants of a one-parameter model.

    Restricted to Calabi--Yau threefolds that are complete intersections in a
    *single* projective space, the five configurations listed in
    :data:`ONE_PARAMETER`. Those are exactly the CICYs with one Kahler
    parameter, which is what makes the mirror map a one-variable problem.

    Method (Hosono, Klemm, Theisen and Yau). With degrees d_a in P^N, set
    kappa = prod_a d_a and mu = prod_a d_a^{d_a}. The fundamental period is

        w_0(z) = sum_m [ prod_a (d_a m)! / (m!)^{N+1} ] z^m ,

    and the logarithmic solution gives the mirror map t = log z + sigma/w_0
    where sigma has coefficients w_0[m] ( sum_a d_a H_{d_a m} - (N+1) H_m ),
    H the harmonic numbers. The Euler--Mascheroni constants cancel precisely
    because sum_a d_a = N + 1, which is the Calabi--Yau condition. The Yukawa
    coupling is then

        K_ttt = kappa / [ (1 - mu z) (1 + z sigma'/w_0 ... )^3 w_0^2 ]

    expressed in q = e^t, and the invariants are read off from

        K_ttt = kappa + sum_d n_d d^3 q^d / (1 - q^d) .

    Returns
    -------
    dict
        ``name``, ``kappa``, ``mu``, and ``invariants`` mapping degree to
        n_d. Values are Python ints; a non-integral result raises, since
        integrality is one of the checks this computation has to pass.

    Raises
    ------
    ValueError
        If the configuration is not one of the five one-parameter models.
        Multi-parameter models need the full multi-variable mirror map, which
        is not implemented here; CIPro covers those through Klemm's code.

    Example
    -------
    The quintic, whose first three invariants are the classical 2875,
    609250 and 317206375.

    >>> gv_invariants([[4, 5]], max_degree=3)["invariants"]
    {1: 2875, 2: 609250, 3: 317206375}
    """
    M = _as_matrix(conf)
    if M.shape[0] != 1:
        raise ValueError(
            "Gopakumar-Vafa invariants are implemented only for the "
            "one-parameter models, which are complete intersections in a "
            "single projective space; this configuration has %d projective "
            "factors and so has more than one Kahler parameter."
            % M.shape[0])
    if not is_calabi_yau(M):
        raise ValueError("configuration is not Calabi-Yau")
    if dimensions(M)[2] != 3:
        raise ValueError("expected a threefold, got a %d-fold"
                         % dimensions(M)[2])

    key = (tuple(int(v) for v in M[0]),)
    name = ONE_PARAMETER.get(key)
    if name is None:
        raise ValueError(
            "%r is not one of the five one-parameter models %r"
            % (key[0], sorted(ONE_PARAMETER)))

    N = int(M[0, 0])
    degs = [int(v) for v in M[0, 1:]]

    kappa = 1
    for d in degs:
        kappa *= d
    mu = 1
    for d in degs:
        mu *= d ** d

    m_order = max_degree + 2
    w0 = [Fraction(0)] * (m_order + 1)
    sigma = [Fraction(0)] * (m_order + 1)
    for k in range(m_order + 1):
        num = 1
        for d in degs:
            num *= factorial(d * k)
        w0[k] = Fraction(num, factorial(k) ** (N + 1))
        bracket = (sum(Fraction(d) * _harmonic(d * k) for d in degs)
                   - Fraction(N + 1) * _harmonic(k))
        sigma[k] = w0[k] * bracket

    S = _t_mul(sigma, _t_inv(w0, m_order), m_order)      # S(0) = 0

    # q = z exp(S)
    eS = _t_exp(S, m_order)
    qz = [Fraction(0)] * (m_order + 2)
    for i, c in enumerate(eS):
        if i + 1 <= m_order + 1:
            qz[i + 1] = c

    # invert the mirror map: z as a series in q
    zq = [Fraction(0)] * (m_order + 2)
    zq[1] = Fraction(1)
    for n in range(2, m_order + 2):
        comp = [Fraction(0)] * (m_order + 2)
        pw = [Fraction(0)] * (m_order + 2)
        pw[0] = Fraction(1)
        for k in range(1, m_order + 2):
            pw = _t_mul(pw, zq, m_order + 1)
            if qz[k]:
                for i in range(m_order + 2):
                    comp[i] += qz[k] * pw[i]
        zq[n] -= comp[n]

    # K_ttt = kappa / [ (1 - mu z) (1 + z S')^3 w_0^2 ]
    zSp = [Fraction(0)] * (m_order + 1)
    for k in range(1, m_order + 1):
        zSp[k] = Fraction(k) * S[k]
    one_plus = [Fraction(0)] * (m_order + 1)
    one_plus[0] = Fraction(1)
    for k in range(m_order + 1):
        one_plus[k] += zSp[k]

    denom = _t_mul(_t_mul(one_plus, one_plus, m_order), one_plus, m_order)
    disc = [Fraction(0)] * (m_order + 1)
    disc[0] = Fraction(1)
    disc[1] = Fraction(-mu)
    denom = _t_mul(denom, disc, m_order)
    denom = _t_mul(denom, _t_mul(w0, w0, m_order), m_order)
    k_z = [Fraction(kappa) * c for c in _t_inv(denom, m_order)]

    # substitute z(q)
    k_q = [Fraction(0)] * (m_order + 1)
    k_q[0] += k_z[0]
    pw = [Fraction(0)] * (m_order + 1)
    pw[0] = Fraction(1)
    for k in range(1, m_order + 1):
        pw = _t_mul(pw, zq[:m_order + 1], m_order)
        for i in range(m_order + 1):
            k_q[i] += k_z[k] * pw[i]

    invariants = {}
    for d in range(1, max_degree + 1):
        s = Fraction(0)
        for e in range(1, d):
            if d % e == 0:
                s += invariants[e] * Fraction(e) ** 3
        value = (k_q[d] - s) / Fraction(d) ** 3
        if value.denominator != 1:
            raise ValueError(
                "non-integral Gopakumar-Vafa invariant n_%d = %s; the "
                "computation is wrong" % (d, value))
        invariants[d] = int(value)

    return {
        "name": name,
        "kappa": kappa,
        "mu": mu,
        "invariants": invariants,
    }
