"""
Tests for pyCICY.k3lattice: the K3 quadratic locus, decided over Q.

  [1] hilbert       reciprocity, bilinearity, (a,-a) = (a,1-a) = 1, known values,
                    and an independent route: solubility of a x^2 + b y^2 = z^2
                    modulo p^k, which shares no code with the formulas
  [2] invariants    known isometries (U, E8 and D_n over Q, <1,1> ~ <2,2> ~ <5,5>),
                    invariance under random rational congruence, the product
                    formula for every form built
  [3] embedding     against theorems that do not mention Hilbert symbols:
                    Davenport-Cassels for sums of two and three squares,
                    Lagrange for four, universality of the hyperbolic plane
  [4] complement    the Hasse-invariant formula against an explicit orthogonal
                    complement, on random subspaces
  [5] K3 lattices   H^2, the rank bound, the P-polarised family in two routes
                    (E8 Cartan matrix; rational coordinates), a Picard jump,
                    and the cases that fall outside
  [6] CICY K3       double intersection numbers by hand, Hodge index, evenness,
                    row-permutation covariance, and the honest "undetermined"
  [7] controls      each check made to fail on purpose: a check that cannot fail
                    is not a test

Run with:  python3 tests/test_k3lattice.py
       or: python3 run_tests.py
"""

import os
import random
import sys
from fractions import Fraction
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sympy import Matrix, Rational

from pyCICY import CICY
from pyCICY import k3lattice as K

FAILURES = []


def check(name, got, want):
    ok = got == want
    print("  {:<58} {:>12} {}".format(name, str(got)[:12],
                                      "ok" if ok else "FAIL want " + str(want)))
    if not ok:
        FAILURES.append(name)


def check_true(name, cond):
    print("  {:<58} {:>12} {}".format(name, str(bool(cond)),
                                      "ok" if cond else "FAIL"))
    if not cond:
        FAILURES.append(name)


def _raises(exc, fn, *a, **kw):
    try:
        fn(*a, **kw)
    except exc:
        return True
    except Exception:
        return False
    return False


def S(*entries):
    return K.QSpace.from_diag(list(entries))


def _reciprocity_failures(pairs):
    bad = 0
    for a, b in pairs:
        prod = K.hilbert(a, b, K.INFINITY)
        for p in K._primes_of(a, b):
            prod *= K.hilbert(a, b, p)
        if prod != 1:
            bad += 1
    return bad


def _random_pairs(rng, n, bound=60):
    return [(rng.choice([-1, 1]) * rng.randint(1, bound),
             rng.choice([-1, 1]) * rng.randint(1, bound)) for _ in range(n)]


# an independent oracle for the Hilbert symbol --------------------------------

def _solvable_mod(a, b, p, k):
    """Whether a x^2 + b y^2 = z^2 has a primitive solution modulo p^k."""
    q = p ** k
    has_any, has_unit = set(), set()
    for z in range(q):
        v = z * z % q
        has_any.add(v)
        if z % p:
            has_unit.add(v)
    for x in range(q):
        for y in range(q):
            v = (a * x * x + b * y * y) % q
            if x % p or y % p:
                if v in has_any:
                    return True
            elif v in has_unit:
                return True
    return False


def test_hilbert():
    print("\n[1] hilbert")
    rng = random.Random(20261009)
    check("reciprocity on 400 random pairs: failures",
          _reciprocity_failures(_random_pairs(rng, 400)), 0)
    # bilinearity and the two identities, over rationals
    fails = 0
    for _ in range(150):
        a, a2, b = [Fraction(rng.randint(1, 40) * rng.choice([-1, 1]),
                             rng.randint(1, 9)) for _ in range(3)]
        for p in K._primes_of(a.numerator, a.denominator, a2.numerator,
                              a2.denominator, b.numerator, b.denominator):
            if K.hilbert(a * a2, b, p) != K.hilbert(a, b, p) * K.hilbert(a2, b, p):
                fails += 1
            if K.hilbert(a, -a, p) != 1:
                fails += 1
            if a != 1 and K.hilbert(a, 1 - a, p) != 1:
                fails += 1
    check("bilinearity, (a,-a)=1, (a,1-a)=1: failures", fails, 0)
    check("(-1,-1)_2", K.hilbert(-1, -1, 2), -1)
    check("(-1,-1)_inf", K.hilbert(-1, -1, K.INFINITY), -1)
    check("(2,5)_2", K.hilbert(2, 5, 2), -1)
    check("(2,7)_2", K.hilbert(2, 7, 2), 1)
    check("(3,3)_3", K.hilbert(3, 3, 3), -1)
    check("(5,5)_5", K.hilbert(5, 5, 5), 1)
    check("square classes: 18 and 1/8 and -50",
          (K.square_class(18), K.square_class(Fraction(1, 8)),
           K.square_class(-50)), (2, 2, -2))

    vals = [-3, -2, -1, 1, 2, 3, 5, 6, 7, 10]
    for p, k in ((2, 5), (3, 3), (5, 2), (7, 2)):
        wrong = [(a, b) for a in vals for b in vals
                 if (K.hilbert(a, b, p) == 1) != _solvable_mod(a, b, p, k)]
        check("formula vs solubility mod %d^%d, %d pairs: mismatches"
              % (p, k, len(vals) ** 2), len(wrong), 0)


def test_invariants():
    print("\n[2] invariants")
    check_true("<1,1> ~ <2,2>", S(1, 1).is_isometric(S(2, 2)))
    check_true("<1,1> ~ <5,5>", S(1, 1).is_isometric(S(5, 5)))
    check_true("<1,1> is not <3,3>", not S(1, 1).is_isometric(S(3, 3)))
    check_true("U ~ <1,-1> ~ <a,-a> for a = 1..20",
               all(S(1, -1).is_isometric(S(a, -a)) for a in range(1, 21))
               and K.QSpace.from_gram(K.lattice_U()).is_isometric(S(1, -1)))
    E8 = K.QSpace.from_gram(K.cartan_E8())
    check("E8: rank, signature, discriminant", (E8.n, E8.pos, E8.d), (8, 8, 1))
    check_true("E8 (x) Q ~ <1>^8", E8.is_isometric(S(*[1] * 8)))
    check_true("D_n (x) Q ~ <1>^n for n = 4..10",
               all(K.QSpace.from_gram(K.cartan_D(n)).is_isometric(S(*[1] * n))
                   for n in range(4, 11)))
    check_true("D8 (x) Q ~ E8 (x) Q",
               K.QSpace.from_gram(K.cartan_D(8)).is_isometric(E8))
    check("det of the Cartan matrices D8, D4, E8",
          (Matrix(K.cartan_D(8)).det(), Matrix(K.cartan_D(4)).det(),
           Matrix(K.cartan_E8()).det()), (4, 4, 1))

    # invariance under random rational congruence, and the product formula
    rng = random.Random(7)
    bad_iso = bad_pf = 0
    for _ in range(40):
        n = rng.choice([3, 4, 5])
        while True:
            A = [[rng.randint(-3, 3) for _ in range(n)] for _ in range(n)]
            G = [[A[i][j] + A[j][i] for j in range(n)] for i in range(n)]
            try:
                V = K.QSpace.from_gram(G)
                break
            except K.DegenerateForm:
                continue
        B = Matrix.eye(n)
        for _ in range(6):
            i, j = rng.sample(range(n), 2)
            E = Matrix.eye(n)
            E[i, j] = Rational(rng.randint(-3, 3), rng.randint(1, 4))
            B = B * E
        B = B * Matrix.diag(*[rng.choice([1, 2, 3, -1, 5]) for _ in range(n)])
        G2 = B.T * Matrix(G) * B
        V2 = K.QSpace.from_gram([[G2[i, j] for j in range(n)] for i in range(n)])
        bad_iso += not V.is_isometric(V2)
        bad_pf += not (V.product_formula() and V2.product_formula())
    check("congruent forms isometric (40 random): failures", bad_iso, 0)
    check("product formula on every form built: failures", bad_pf, 0)


def _sum_of_squares(n, k):
    r = int(n ** 0.5) + 1
    return any(sum(x * x for x in t) == n
               for t in product(range(r + 1), repeat=k))


def test_embedding():
    print("\n[3] embedding, against theorems that do not mention Hilbert symbols")
    N = range(1, 121)
    W2, W3, W4 = S(1, 1), S(1, 1, 1), S(1, 1, 1, 1)
    # rank-1 complement for two squares, rank-2 for three, rank-3 for four
    check("n in <1,1> iff a sum of two squares (Davenport-Cassels): mismatches",
          [n for n in N if bool(K.embeds(S(n), W2)) != _sum_of_squares(n, 2)], [])
    check("n in <1,1,1> iff a sum of three squares: mismatches",
          [n for n in N if bool(K.embeds(S(n), W3)) != _sum_of_squares(n, 3)], [])
    check("n in <1,1,1,1> for every n (Lagrange): failures",
          [n for n in N if not K.embeds(S(n), W4)], [])
    check("-n in <-1,-1,-1> iff n a sum of three squares: mismatches",
          [n for n in N if bool(K.embeds(S(-n), S(-1, -1, -1)))
           != _sum_of_squares(n, 3)], [])
    check("n in <1,-1> (U is universal), |n| <= 60: failures",
          [n for n in range(-60, 61) if n and not K.embeds(S(n), S(1, -1))], [])
    check_true("7 is refused in <1,1,1>, and the obstruction is local at p=2",
               K.embeds(S(7), W3).obstruction.startswith("local at p=2"))
    # the complement of <3> in <1,1> is <3> with Hasse invariant (3,-1)_p, which
    # is -1 at p = 2 and p = 3 (an even number of primes, as it must be); the
    # refusal names the smallest
    check("3 is refused in <1,1>: (3,-1)_p at p = 2, 3, inf",
          (K.hilbert(3, -1, 2), K.hilbert(3, -1, 3),
           K.hilbert(3, -1, K.INFINITY)), (-1, -1, 1))
    check_true("... and the refusal is a local one, at the smallest such prime",
               K.embeds(S(3), W2).obstruction.startswith("local at p=2"))
    check_true("signature obstruction: <1,1> does not fit in <1,-1,-1>",
               K.embeds(S(1, 1), S(1, -1, -1)).obstruction.startswith("signature"))
    check_true("rank obstruction: rank 3 into rank 2",
               K.embeds(S(1, 1, 1), W2).obstruction.startswith("rank"))
    check_true("a space embeds in itself, with a rank-0 complement",
               K.embeds(W3, W3).complement.n == 0)


def test_complement():
    print("\n[4] the complement: Hasse-invariant formula against linear algebra")
    rng = random.Random(4242)
    Wdiag = [1, 2, -3, 5, -7, 6]
    W = Matrix.diag(*Wdiag)
    WS = S(*Wdiag)
    bad = tried = 0
    while tried < 40:
        k = rng.choice([1, 2, 3])
        vecs = [[Rational(rng.randint(-4, 4), rng.randint(1, 3))
                 for _ in range(6)] for _ in range(k)]
        try:
            Vg = K.gram_of(W.tolist(), vecs)
            V = K.QSpace.from_gram(Vg)
        except K.DegenerateForm:
            continue
        if Matrix(vecs).rank() < k:
            continue
        tried += 1
        e = K.embeds(V, WS)
        C = K.QSpace.from_gram(K.explicit_complement(W.tolist(), vecs))
        if not (e and e.complement.is_isometric(C) and e.complement.n == 6 - k):
            bad += 1
        # and the sum of the two really is W
        if not V.orth_sum(C).is_isometric(WS):
            bad += 1
    check("40 random subspaces: formula = explicit complement, V + V' = W",
          bad, 0)


def _P():
    return K.direct_sum(K.lattice_U(), K.negate(K.cartan_D(8)),
                        K.negate(K.cartan_D(4)))


def test_k3():
    print("\n[5] K3 lattices")
    H = K.QSpace.from_gram(K.k3_h2())
    check("H^2(K3): rank, signature, discriminant",
          (H.n, H.pos, H.neg, H.d), (22, 3, 19, -1))
    L = K.QSpace.from_gram(K.k3_locus_form())
    check("target U^2 + <-1>^4: rank, signature", (L.n, L.pos, L.neg), (8, 2, 6))
    check("MIN_PICARD_RANK = 22 - 8", K.MIN_PICARD_RANK, 22 - L.n)

    # quartic: <4>
    T = K.transcendental_space([[4]])
    check("quartic: T rank and signature", (T.n, T.pos, T.neg), (21, 2, 19))
    v = K.locus_verdict([[4]], exact=True)
    check_true("quartic, exact: out, by rank", v.status == "out"
               and v.reason.startswith("rank"))

    # the P-polarised family, route one: Cartan matrices and the Hasse formula
    P = _P()
    Pn = K.QSpace.from_gram(P)
    check("P = U + D8(-1) + D4(-1): rank, signature", (Pn.n, Pn.pos, Pn.neg),
          (14, 1, 13))
    vP = K.locus_verdict(P)
    check("P-polarised: verdict", vP.status, "in")
    check_true("P-polarised: T is isometric to U^2 + <-1>^4 (equal rank)",
               vP.transcendental.is_isometric(L))

    # route two: explicit vectors in rational coordinates, no E8 Cartan matrix
    # H_Q = U^3 + <-1>^16 (E8 (x) Q ~ <1>^8, checked above); P embedded by roots
    W = K.direct_sum(K.lattice_U(), K.lattice_U(), K.lattice_U(),
                     K.identity_form(16, -1))

    def vec(i):
        v = [0] * 22
        v[i] = 1
        return v

    def root(a, b, sign):
        v = [0] * 22
        v[a], v[b] = 1, sign * 1
        return v

    rows = [vec(0), vec(1)]
    b1, b2 = 6, 14
    rows += [root(b1 + i, b1 + i + 1, -1) for i in range(7)] + \
            [root(b1 + 6, b1 + 7, +1)]
    rows += [root(b2 + i, b2 + i + 1, -1) for i in range(3)] + \
            [root(b2 + 2, b2 + 3, +1)]
    Pg = K.gram_of(W, rows)
    check("explicit roots reproduce Gram(P) exactly",
          [[int(x) for x in r] for r in Pg], P)
    Tx = K.QSpace.from_gram(K.explicit_complement(W, rows))
    check_true("route two: explicit complement ~ route one's T",
               Tx.is_isometric(vP.transcendental))
    check_true("route two: explicit complement ~ U^2 + <-1>^4",
               Tx.is_isometric(L))

    # a Picard jump keeps a K3 on the locus (T shrinks to a subspace)
    vJ = K.locus_verdict(K.direct_sum(P, [[-2]]))
    check("P + <-2> (rank 15): verdict", vJ.status, "in")
    check("P + <-2>: T has rank 7", vJ.transcendental.n, 7)

    # outside the locus
    vA = K.locus_verdict(K.direct_sum(K.lattice_U(), K.negate(K.cartan_E8()),
                                      K.negate(K.cartan_A(4))))
    check("U + E8(-1) + A4(-1) (rank 14): verdict", vA.status, "out")
    check_true("... because T has discriminant 5, not 1",
               vA.reason.startswith("equal rank") and vA.transcendental.d == 5)
    vR = K.locus_verdict(K.direct_sum(K.lattice_U(), K.negate(K.cartan_D(8)),
                                      K.negate(K.cartan_A(3))))
    check_true("rank 13: out, by rank",
               vR.status == "out" and vR.reason.startswith("rank"))
    check("rank 13 with exact=False: verdict",
          K.locus_verdict(K.direct_sum(K.lattice_U(), K.negate(K.cartan_D(8)),
                                       K.negate(K.cartan_A(3))),
                          exact=False).status, "undetermined")
    check("exact=False never says out: quartic", K.locus_verdict(
        [[4]], exact=False).status, "undetermined")

    # things that are not Picard lattices
    check_true("signature (2,0) is refused", _raises(
        K.NotAK3Lattice, K.transcendental_space, [[2, 0], [0, 2]]))
    check_true("an odd form is refused", _raises(
        K.NotAK3Lattice, K.transcendental_space, [[3]]))
    check_true("rank 26 cannot sit in H^2", _raises(
        K.NotAK3Lattice, K.transcendental_space,
        K.direct_sum(K.lattice_U(), K.negate(K.cartan_E8()),
                     K.negate(K.cartan_E8()), K.negate(K.cartan_E8()))))


def test_cicy():
    print("\n[6] CICY K3")
    quartic = [[3, 4]]
    p13 = [[1, 2], [1, 2], [1, 2]]
    p4 = [[1, 1, 1]] * 4
    check("quartic: J^2 = 4", K.double_intersection(quartic), [[4]])
    check("P4[2,3]: J^2 = 6", K.double_intersection([[4, 2, 3]]), [[6]])
    check("P5[2,2,2]: J^2 = 8", K.double_intersection([[5, 2, 2, 2]]), [[8]])
    check("(P1)^3 [2,2,2]: 2(J - I)", K.double_intersection(p13),
          [[0, 2, 2], [2, 0, 2], [2, 2, 0]])
    check("P1 x P2 [(2),(3)]: ((0,3),(3,2))", K.double_intersection(
        [[1, 2], [2, 3]]), [[0, 3], [3, 2]])
    check("(P1)^4 [1,1,1]^2: 2(J - I)", K.double_intersection(p4),
          [[0 if i == j else 2 for j in range(4)] for i in range(4)])
    for cfg in (quartic, p13, p4, [[1, 2], [2, 3]]):
        X = CICY(cfg)
        check("chi = 24, so %s is a K3" % (cfg,), int(X.euler_characteristic()), 24)
    check("accepts a CICY object", K.double_intersection(CICY(p13)),
          K.double_intersection(p13))
    # covariance under a permutation of the ambient factors
    G = K.double_intersection([[1, 2], [2, 3]])
    Gp = K.double_intersection([[2, 3], [1, 2]])
    check("row swap swaps the Gram matrix's rows and columns",
          Gp, [[G[1][1], G[1][0]], [G[0][1], G[0][0]]])
    check_true("a 3-fold is refused", _raises(ValueError, K.double_intersection,
                                              [[4, 5]]))

    for cfg in (quartic, p13, p4):
        Gm = K.ambient_picard_gram(cfg)
        Sp = K.QSpace.from_gram(Gm)
        check("Hodge index: signature (1, r-1) for %s" % (cfg,),
              (Sp.pos, Sp.neg), (1, len(Gm) - 1))
        check_true("even diagonal for %s" % (cfg,),
                   all(Gm[i][i] % 2 == 0 for i in range(len(Gm))))
        v = K.cicy_k3_verdict(cfg)
        check("verdict for %s" % (cfg,), v.status, "undetermined")
        check_true("T has rank 22 - r > 8, so it cannot embed",
                   v.transcendental.n == 22 - len(Gm) > 8)
    check("exact=True on the quartic", K.cicy_k3_verdict(quartic, exact=True).status,
          "out")


def test_controls():
    print("\n[7] controls: each check made to fail on purpose")
    rng = random.Random(20261009)
    pairs = _random_pairs(rng, 400)

    real = K._hilbert_2
    try:
        K._hilbert_2 = lambda a, b: 1
        broken = _reciprocity_failures(pairs)
    finally:
        K._hilbert_2 = real
    check_true("a Hilbert symbol that is always +1 at p=2 breaks reciprocity",
               broken > 0)

    try:
        K._hilbert_2 = lambda a, b: real(a, b) * real(1, 1) * \
            (-1 if (a % 8 == 5 and b % 2 == 0) else 1)
        broken = _reciprocity_failures(pairs)
    finally:
        K._hilbert_2 = real
    check_true("a wrong sign on one 2-adic case breaks reciprocity", broken > 0)
    check("and with the real symbol restored: failures",
          _reciprocity_failures(pairs), 0)

    W3 = S(1, 1, 1)
    check_true("without the rank-2 condition, 7 wrongly embeds in <1,1,1>",
               bool(K.embeds(S(7), W3, _skip_rank2_condition=True))
               and not _sum_of_squares(7, 3))
    wrong = [n for n in range(1, 121)
             if bool(K.embeds(S(n), W3, _skip_rank2_condition=True))
             != _sum_of_squares(n, 3)]
    check_true("... and the sums-of-three-squares oracle catches it (n=%s)"
               % (wrong[:4],), len(wrong) > 0)

    impossible = K.QSpace(2, 2, 0, 1, minus={3})
    check_true("a space with one Hasse invariant -1 violates the product formula",
               not impossible.product_formula())
    check_true("a sign of the discriminant that contradicts the signature",
               _raises(ValueError, K.QSpace, 2, 2, 0, -1))
    check_true("a singular Gram matrix is refused, not quotiented",
               _raises(K.DegenerateForm, K.diagonalize, [[1, 1], [1, 1]]))
    check_true("a non-integral float is refused", _raises(
        TypeError, K.QSpace.from_gram, [[0.5]]))

    real_di = K.double_intersection
    try:
        K.double_intersection = lambda cicy: [[0, 0], [0, 0]]
        refused = _raises(K.AmbientDegenerate, K.ambient_picard_gram, [[3, 4]])
    finally:
        K.double_intersection = real_di
    check_true("a degenerate ambient Gram matrix gets no verdict", refused)

    try:
        K.double_intersection = lambda cicy: [[3]]
        refused = _raises(ValueError, K.ambient_picard_gram, [[3, 4]])
    finally:
        K.double_intersection = real_di
    check_true("an odd self-intersection is flagged as not a K3", refused)

    try:
        K.double_intersection = lambda cicy: [[2, 0], [0, 2]]
        refused = _raises(ValueError, K.ambient_picard_gram, [[3, 4]])
    finally:
        K.double_intersection = real_di
    check_true("a definite ambient Gram violates the Hodge index theorem",
               refused)


def main():
    for section in (test_hilbert, test_invariants, test_embedding,
                    test_complement, test_k3, test_cicy, test_controls):
        try:
            section()
        except Exception as exc:  # an exception is a failure, not an abort
            print("  {:<58} {:>12} FAIL".format(section.__name__ + " raised",
                                                type(exc).__name__))
            print("      " + str(exc)[:200])
            FAILURES.append("%s raised %s" % (section.__name__,
                                              type(exc).__name__))
    print()
    if FAILURES:
        print("FAILED: %d" % len(FAILURES))
        for f in FAILURES:
            print("  - " + f)
        return 1
    print("all k3lattice tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
