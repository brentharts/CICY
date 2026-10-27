r"""
pyCICY.sixsphere_theta -- a theta function whose multiplier is the holonomy
character (Part XI).

Part IX proved, by a finite search, that the monodromy group <T1, T2> is the
kernel of a character chi : G(Z) -> Z/12 of the integral Jacobi group
G = SL(2, Z) |x Heis, with chi = alpha + f, alpha(S) = 9, alpha(T) = 1 and
f(c1, c2, s) = s - 6 q(c1, c2), q the odd theta characteristic. It also
remarked, without computing it, that alpha looks like the multiplier of eta^2
and f like an odd theta characteristic. This module turns that remark into a
second, analytic route to index 12.

On the period domain with coordinates (tau, mu, beta) (Alpoge's period matrix
Z = [[6 mu, tau], [beta, mu]], :mod:`pyCICY.sixsphere`), put

    F(tau, mu, beta) = e(beta / 12) theta_1(tau, mu) / eta(tau),

    e(x) = exp(2 pi i x).

theta_1 / eta has weight 0 and index 1/2; the Jacobi-form factor
e(-c mu^2 / 2(c tau + d)) that index 1/2 produces is exactly what the beta
coordinate absorbs, because beta transforms with -6 c mu^2 / (c tau + d)
(e.g. L(S): beta -> beta - 6 mu^2 / tau). The claims, each checked here:

1. **Multiplier = chi** (:func:`multiplier_check`). For every g in G(Z),
   acting by the law that the integer matrix alone forces
   (:func:`pyCICY.sixsphere.laws_from_lattice`),

       F(g . x) = e(chi(g) / 12) F(x),

   with no automorphy factor. Checked on the generators L(S), L(T), h(1,0,0),
   h(0,1,0), z and on T1, T2 themselves, then on random words.
2. **Index 12, analytically** (:func:`stabiliser_is_holonomy`). Hence the
   stabiliser of F in G(Z) is ker chi, which Part IX identified with
   <T1, T2>: the holonomy group is the invariance group of one explicit
   function, and F^12 is invariant under all of G(Z). alpha here is computed
   by a Euclid decomposition of M into S and T (:func:`alpha_sl2`) and
   checked against the eta^2 multiplier numerically -- the two share no code.
3. **Uniqueness** (:func:`candidate_search`). Among 80 candidates
   e(k beta) theta_j(tau, +-mu) eta^r with k = +-1/12, j = 1..4,
   r in {-1, 1, -3, 2, -2}, only k = 1/12, j = 1, r = -1 (both signs of mu,
   theta_1 being odd) is invariant under T1 and T2. This is the control that
   has to fail: the other 76 do.
4. **Proof** (:func:`proof_by_generators`). The identity of claim 1 holds
   as a theorem, not only numerically: on each of the five generators the
   classical transformation laws of theta_1 and eta (Mumford, Tata I),
   added to the beta shift of the law, give an exponent that sympy
   simplifies to a constant, equal to chi / 12 mod 1. The five generate
   G(Z), the action is a group action and chi a homomorphism, so it holds on
   every word. The five classical laws are themselves checked numerically
   against the theta series, as the quoted inputs.
5. **Zero divisor** (:func:`zero_divisor`). F vanishes exactly where
   theta_1 does, mu in Z + tau Z, and is nonzero at the other 2-torsion
   points. Its meaning on the period domain is left open.

Numerics are mpmath at 25 digits; theta_1 is its own series, not
``mpmath.jtheta``, whose nome argument q = e(tau / 2) picks a branch of
q^(1/4) that gives the wrong multiplier for T.
"""

import random

import mpmath as mp
from sympy import Matrix, Rational, eye, lambdify, nsimplify, simplify

from . import sixsphere as S
from . import sixsphere_gauge as G

mp.mp.dps = 25

#: A generic base point (tau, mu, beta) on the period domain.
BASE = (mp.mpc("0.13", "1.1"), mp.mpc("0.21", "0.17"), mp.mpc("0.3", "-0.4"))


def e(x):
    return mp.exp(2j * mp.pi * x)


def eta(t):
    """Dedekind eta, e(tau / 24) prod (1 - q^n)."""
    return e(t / 24) * mp.qp(e(t))


def theta(j, t, z):
    r"""Jacobi theta_j(tau, z) from its own series (j = 1..4).

    theta_1 = 2 sum_{n >= 0} (-1)^n e(tau (n + 1/2)^2 / 2) sin((2n + 1) pi z),
    and the others by the usual half-period shifts."""
    if j == 1:
        return 2 * mp.nsum(lambda n: (-1) ** int(n) * e(t * (n + 0.5) ** 2 / 2)
                           * mp.sin((2 * n + 1) * mp.pi * z), [0, mp.inf])
    if j == 2:
        return 2 * mp.nsum(lambda n: e(t * (n + 0.5) ** 2 / 2)
                           * mp.cos((2 * n + 1) * mp.pi * z), [0, mp.inf])
    if j == 3:
        return 1 + 2 * mp.nsum(lambda n: e(t * n ** 2 / 2)
                               * mp.cos(2 * n * mp.pi * z), [1, mp.inf])
    if j == 4:
        return 1 + 2 * mp.nsum(lambda n: (-1) ** int(n) * e(t * n ** 2 / 2)
                               * mp.cos(2 * n * mp.pi * z), [1, mp.inf])
    raise ValueError(j)


def F(t, m, b):
    """e(beta / 12) theta_1(tau, mu) / eta(tau)."""
    return e(b / 12) * theta(1, t, m) / eta(t)


# --------------------------------------------------------------------------
# alpha on all of SL(2, Z), two ways
# --------------------------------------------------------------------------

def alpha_sl2(M):
    """alpha(M) in Z/12 by Euclid: write M as a word in S and T and add
    alpha(S) = 9, alpha(T) = 1 (Part IX's presentation)."""
    M = Matrix(M)
    a = 0
    while M[1, 0] != 0:
        q = M[0, 0] // M[1, 0]
        M = G._TT ** (-q) * M
        a += q * G.ALPHA["T"]
        M = G._ST.inv() * M
        a += G.ALPHA["S"]
    if M[0, 0] == -1:          # -I = S^2
        M = -M
        a += 2 * G.ALPHA["S"]
    return (a + M[0, 1] * G.ALPHA["T"]) % 12


def alpha_eta2(M, t=BASE[0]):
    """alpha(M) read off the eta^2 multiplier numerically:
    eta(M tau)^2 = e(alpha / 12) (c tau + d) eta(tau)^2 -- an independent
    route that never decomposes M."""
    a, b, c, d = (int(x) for x in Matrix(M))
    r = eta((a * t + b) / (c * t + d)) ** 2 / ((c * t + d) * eta(t) ** 2)
    n = mp.re(12 * mp.log(r) / (2j * mp.pi))
    return int(mp.nint(n)) % 12, abs(r), abs(n - mp.nint(n))


def chi(g):
    """Part IX's chi with alpha supplied by :func:`alpha_sl2`."""
    M, _ = G.split(g)
    return G.chi(g, alpha_M=alpha_sl2(M))


# --------------------------------------------------------------------------
# 1. the multiplier
# --------------------------------------------------------------------------

def act(g, x=BASE):
    """g . x by the law the integer matrix forces."""
    law = S.laws_from_lattice(g)["law"]
    return lambdify((S.TAU, S.MU, S.BETA), law, "mpmath")(*x)


def multiplier(g, f=F, x=BASE):
    """(12 arg(f(g x) / f(x)) / 2 pi, |f(g x) / f(x)|)."""
    r = f(*act(g, x)) / f(*x)
    return mp.re(12 * mp.log(r) / (2j * mp.pi)), abs(r)


def _d12(a, b):
    return abs(((a - b) % 12 + 6) % 12 - 6)


GENERATORS = {"L(S)": G.levi(G._ST), "L(T)": G.levi(G._TT),
              "h(1,0,0)": G.heis(1, 0, 0), "h(0,1,0)": G.heis(0, 1, 0),
              "z": G.heis(0, 0, 1), "T1": S.T1, "T2": S.T2}


def random_words(n=20, seed=1, length=(2, 5)):
    rng = random.Random(seed)
    gens = list(GENERATORS.values())
    out = []
    for _ in range(n):
        g = Matrix(G._I4)
        for _ in range(rng.randint(*length)):
            g = g * rng.choice(gens)
        out.append(g)
    return out


def multiplier_check(n_words=20, tol=1e-12):
    """F(g x) / F(x) = e(chi(g) / 12) on the generators and on random words."""
    gens = {}
    for k, g in GENERATORS.items():
        n, a = multiplier(g)
        gens[k] = {"chi": chi(g), "measured": float(n), "abs": float(a),
                   "ok": _d12(n, chi(g)) < tol and abs(a - 1) < tol}
    words = []
    for g in random_words(n_words):
        n, a = multiplier(g)
        words.append(_d12(n, chi(g)) < tol and abs(a - 1) < tol)
    return {"generators": gens, "words_ok": sum(words), "words": len(words)}


# --------------------------------------------------------------------------
# 2. index 12 analytically
# --------------------------------------------------------------------------

def alpha_two_ways(n=20, seed=3):
    """alpha by Euclid against alpha from the eta^2 multiplier, on random
    SL(2, Z) words."""
    rng = random.Random(seed)
    gens = [G._ST, G._TT, G._ST.inv(), G._TT.inv()]
    agree = 0
    for _ in range(n):
        M = eye(2)
        for _ in range(rng.randint(1, 8)):
            M = M * rng.choice(gens)
        a, mod, err = alpha_eta2(M)
        agree += a == alpha_sl2(M) and abs(mod - 1) < 1e-12 and err < 1e-12
    return {"agree": agree, "of": n}


def stabiliser_is_holonomy(n_words=20, tol=1e-12):
    """F is fixed by T1, T2 and by every word in them; an element of G(Z) fixes
    F iff chi = 0; F^12 is fixed by everything."""
    rng = random.Random(7)
    hol = []
    for _ in range(n_words):
        g = Matrix(G._I4)
        for _ in range(rng.randint(1, 6)):
            g = g * rng.choice([S.T1, S.T2, S.T1.inv(), S.T2.inv()])
        n, a = multiplier(g)
        hol.append(_d12(n, 0) < tol and abs(a - 1) < tol)
    fixes, kernel, f12 = 0, 0, 0
    words = random_words(n_words, seed=11)
    for g in words:
        n, a = multiplier(g)
        fixed = _d12(n, 0) < tol and abs(a - 1) < tol
        fixes += fixed == (chi(g) == 0)
        kernel += chi(g) == 0
        f12 += _d12(12 * n, 0) < 1e-9
    # the cosets: the generator z = h(0, 0, 1) has chi = 1 and multiplier
    # e(1/12), so z^k for k = 0..11 represent all 12 cosets of ker chi
    cosets = sorted({int(mp.nint(multiplier(G.heis(0, 0, k))[0])) % 12
                     for k in range(12)})
    return {"holonomy_fixes_F": sum(hol) == len(hol),
            "fixed_iff_chi_zero": fixes == len(words),
            "kernel_elements_sampled": kernel,
            "F12_invariant": f12 == len(words),
            "cosets": cosets}


# --------------------------------------------------------------------------
# 3. uniqueness among nearby candidates (the control)
# --------------------------------------------------------------------------

def candidate_search(tol=1e-12):
    """e(k beta) theta_j(tau, sgn mu) eta(tau)^r for k = +-1/12, sgn = +-1,
    r in {-1, 1, -3, 2, -2}, j = 1..4: which are fixed by T1 and T2?"""
    fT1 = act(S.T1)
    fT2 = act(S.T2)
    survivors, n = [], 0
    for k in (mp.mpf(1) / 12, -mp.mpf(1) / 12):
        for sgn in (1, -1):
            for r in (-1, 1, -3, 2, -2):
                for j in (1, 2, 3, 4):
                    n += 1

                    def f(t, m, b):
                        return e(k * b) * theta(j, t, sgn * m) * eta(t) ** r
                    f0 = f(*BASE)
                    if all(abs(f(*y) / f0 - 1) < tol for y in (fT1, fT2)):
                        survivors.append((float(k * 12), sgn, r, j))
    return {"candidates": n, "survivors": survivors}


def wrong_index_control():
    """e(beta / 6) theta_1 / eta -- the beta coefficient doubled, so the index
    no longer matches -- is not fixed by T1."""
    def f(t, m, b):
        return e(b / 6) * theta(1, t, m) / eta(t)
    r = f(*act(S.T1)) / f(*BASE)
    return {"fixed_by_T1": abs(r - 1) < 1e-12, "abs": float(abs(r))}


# --------------------------------------------------------------------------
# 3b. the proof: classical laws + the beta shift, exactly
# --------------------------------------------------------------------------

def _classical_exponent(name, t, m):
    """log_e of (theta_1/eta)(g tau, g mu) / (theta_1/eta)(tau, mu), from the
    classical laws (quoted, Mumford Tata I, I.5 and I.7):
      theta_1(-1/t, m/t) = -i sqrt(-it) e(m^2/2t) theta_1(t, m),
      eta(-1/t) = sqrt(-it) eta(t);
      theta_1(t+1, m) = e(1/8) theta_1,  eta(t+1) = e(1/24) eta;
      theta_1(t, m+1) = -theta_1;  theta_1(t, m-t) = -e(m - t/2) theta_1."""
    return {"L(S)": -Rational(1, 4) + m ** 2 / (2 * t),
            "L(T)": Rational(1, 8) - Rational(1, 24),
            "h(1,0,0)": Rational(1, 2),
            "h(0,1,0)": Rational(1, 2) + m - t / 2,
            "z": 0}[name]


def proof_by_generators():
    """For each generator: classical exponent + (beta' - beta)/12, simplified
    exactly. It must be a constant c with 12 c = chi(g) mod 12."""
    out = {}
    t, m, b = S.TAU, S.MU, S.BETA
    for k in ("L(S)", "L(T)", "h(1,0,0)", "h(0,1,0)", "z"):
        law = S.laws_from_lattice(GENERATORS[k])["law"]
        c = simplify(_classical_exponent(k, t, m) + (law[2] - b) / 12)
        const = c.free_symbols == set()
        out[k] = {"exponent": c, "constant": const, "chi": chi(GENERATORS[k]),
                  "ok": const and (12 * nsimplify(c) - chi(GENERATORS[k])) % 12 == 0}
    return out


def classical_laws_check(tol=1e-20):
    """The quoted laws used by :func:`proof_by_generators`, against the
    series: max relative error over the five."""
    t, m, _ = BASE
    errs = [
        abs(theta(1, -1 / t, m / t) / (-1j * mp.sqrt(-1j * t) * e(m ** 2 / (2 * t))
                                       * theta(1, t, m)) - 1),
        abs(eta(-1 / t) / (mp.sqrt(-1j * t) * eta(t)) - 1),
        abs(theta(1, t + 1, m) / (e(mp.mpf(1) / 8) * theta(1, t, m)) - 1),
        abs(eta(t + 1) / (e(mp.mpf(1) / 24) * eta(t)) - 1),
        abs(theta(1, t, m + 1) / -theta(1, t, m) - 1),
        abs(theta(1, t, m - t) / (-e(m - t / 2) * theta(1, t, m)) - 1)]
    return {"max_rel_err": float(max(errs)), "ok": max(errs) < tol}


# --------------------------------------------------------------------------
# 4. the zero divisor
# --------------------------------------------------------------------------

def zero_divisor():
    """F at the lattice points mu = m + n tau (zero) and at the three other
    2-torsion points (nonzero)."""
    t, _, b = BASE
    at_lattice = max(abs(F(t, m + n * t, b)) for m in (-1, 0, 1, 2)
                     for n in (-1, 0, 1))
    half = min(abs(F(t, u + v * t, b))
               for u, v in ((0.5, 0), (0, 0.5), (0.5, 0.5)))
    return {"max_on_lattice": float(at_lattice), "min_at_half_periods": float(half)}


def status_theta():
    """Run every check of Part XI and print a summary; True if all pass."""
    checks = []

    def ok(name, cond):
        checks.append((name, bool(cond)))

    mc = multiplier_check()
    ok("F(g x) = e(chi(g)/12) F(x) on the 7 generators",
       all(v["ok"] for v in mc["generators"].values()))
    ok("... and on %d random words" % mc["words"], mc["words_ok"] == mc["words"])
    at = alpha_two_ways()
    ok("alpha by Euclid = alpha from the eta^2 multiplier",
       at["agree"] == at["of"])
    sh = stabiliser_is_holonomy()
    ok("<T1, T2> fixes F; g fixes F iff chi(g) = 0; F^12 invariant",
       sh["holonomy_fixes_F"] and sh["fixed_iff_chi_zero"] and sh["F12_invariant"])
    ok("z^k reach all 12 multipliers", sh["cosets"] == list(range(12)))
    cs = candidate_search()
    ok("80 candidates, survivors exactly theta_1 / eta with e(beta/12)",
       cs["candidates"] == 80 and sorted(cs["survivors"])
       == [(1.0, -1, -1, 1), (1.0, 1, -1, 1)])
    ok("control: e(beta/6) theta_1 / eta not fixed by T1",
       not wrong_index_control()["fixed_by_T1"])
    pr = proof_by_generators()
    ok("proof: classical laws + beta shift = chi/12 exactly, 5 generators",
       all(v["ok"] for v in pr.values()))
    ok("control: without the beta shift, L(S) and h(0,1,0) are not constant",
       all(simplify(_classical_exponent(k, S.TAU, S.MU)).free_symbols
           for k in ("L(S)", "h(0,1,0)")))
    ok("the quoted classical laws hold against the series",
       classical_laws_check()["ok"])
    zd = zero_divisor()
    ok("F = 0 on mu in Z + tau Z, nonzero at half periods",
       zd["max_on_lattice"] < 1e-15 and zd["min_at_half_periods"] > 1e-3)
    for name, c in checks:
        print("  {:<62} {}".format(name, "ok" if c else "FAIL"))
    return all(c for _, c in checks)


if __name__ == "__main__":
    import sys
    sys.exit(0 if status_theta() else 1)
