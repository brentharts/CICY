r"""
pyCICY.sixsphere_lean -- the finite layer of the complex six-sphere, as Lean 4.

:mod:`pyCICY.sixsphere` computes the integer and rational layer of Alpoge's
construction and checks each number by a second route where one exists. This
module lifts the finite part of that layer one rung further:

    exact  <  computed twice by unrelated routes  <  machine-checked

It follows :mod:`pyCICY.theories.nariai_lean` exactly. Every :class:`Fact` is
a Lean theorem paired with the Python callable that computes the same claim;
every matrix or vector in the Lean file is *generated* from the same sympy
object :mod:`pyCICY.sixsphere` computes with, so the proof cannot drift onto
a different object; and the axiom report is parsed, never assumed.

What is proved
--------------
Concrete facts, by ``decide`` on integer matrices written as lists:
the orders of T1, T2, the unipotent cusp, the contragredients and
A1 A2 M0 = I, the fixed vectors, the invariant form and bivector, b =
diag(6, -1), B0, the freeness of the paper's logarithmic transforms, the
twist integers and |p|, the Kodaira subquotient, Noether's selection, the
hexagon of E0, the C^* weights, and the Picard coordinates.

Universally quantified facts, by ``omega`` or ``grind``:
  * uniqueness of the invariant alternating form and of the invariant
    bivector -- the linear systems are generated from the matrices by sympy
    and the Python half checks they are exactly the entries of
    T^t Q T - Q and T E T^t - E;
  * gcd(p, 12) = 1 for every admissible twist and every l0;
  * the Seifert and pi_1 relation determinants as identities in l0, l1, l2;
  * every cone of the *infinite* A2 fan is unimodular -- a statement about
    all x, y, not about a patch;
  * the unique normal form of K_X in Pic.

What is not
-----------
Smith normal forms (the passage from a relation determinant to a group
order), saturation of lattices, symbolic period laws, Riemann-Roch, and
everything analytic. Where a fact rests on one of those the note says which
half is Python-only. And the headline theorem is not here at all: it is
machine-checked, over Mathlib, in github.com/plby/HopfProblem.

The axiom policy
----------------
An allow-list, not a deny-list. A fact is machine-checked only if every
axiom in its parsed report is one of Lean's three standard axioms,
``propext``, ``Quot.sound`` and ``Classical.choice`` -- the same three the
HopfProblem Comparator permits. Anything else disqualifies, including axioms
nobody anticipated: Lean 4.33 compiles ``native_decide`` to a fresh
auxiliary axiom per proof (``<thm>._native.native_decide.ax_N``), not to
``Lean.ofReduceBool``, and a deny-list naming the latter lets it through.
The test suite's controls include exactly that case.

:mod:`~pyCICY.theories.nariai_lean` is stricter, admitting only ``propext``
and ``Quot.sound``, because its facts claim to be constructive. These claim
only to be proved; whether each is also choice-free is reported per fact, so
the stronger grade stays visible.
"""

from sympy import Matrix, symbols, expand, eye, zeros

from . import sixsphere as S
from .theories.nariai_lean import (Fact, check_source, lean_executable,
                                   NoToolchain)

__all__ = ["FACTS", "lean_source", "check_python", "check_lean", "status",
           "lean_mat", "lean_vec", "STANDARD_AXIOMS"]

NAMESPACE = "SixSphere"
FILENAME = "SixSphereFacts.lean"
#: The allow-list: Lean's standard axioms, as permitted by the HopfProblem
#: Comparator. Every other axiom disqualifies a fact.
STANDARD_AXIOMS = ("propext", "Quot.sound", "Classical.choice")


# ---------------------------------------------------------------------------
# rendering sympy objects as Lean literals
# ---------------------------------------------------------------------------

def lean_vec(v):
    return "[" + ", ".join(str(int(x)) for x in Matrix(v)) + "]"


def lean_mat(M):
    M = Matrix(M)
    return "[" + ", ".join(
        "[" + ", ".join(str(int(M[i, j])) for j in range(M.cols)) + "]"
        for i in range(M.rows)) + "]"


def _lean_expr(e):
    """A linear sympy expression with integer coefficients, as Lean."""
    e = expand(e)
    terms = []
    for sym, coeff in sorted(e.as_coefficients_dict().items(), key=str):
        c = int(coeff)
        terms.append("(%d)*%s" % (c, sym) if sym != 1 else "(%d)" % c)
    return " + ".join(terms) if terms else "0"


def _invariance_equations(transform):
    """The linear equations of invariance, from the matrices, by sympy."""
    q = symbols("q01 q02 q03 q12 q13 q23")
    Q = S._antisym(q)
    eqs = []
    for T in (S.T1, S.T2):
        D = transform(T, Q) - Q
        for i in range(4):
            for j in range(i + 1, 4):
                if expand(D[i, j]) != 0:
                    eqs.append(expand(D[i, j]))
    return q, eqs


_FORM_EQS = _invariance_equations(lambda T, Q: T.T * Q * T)
_BIV_EQS = _invariance_equations(lambda T, E: T * E * T.T)


def _hyps(eqs):
    return "\n    ".join("(h%d : %s = 0)" % (k, _lean_expr(e))
                         for k, e in enumerate(eqs))


# ---------------------------------------------------------------------------
# the data, generated
# ---------------------------------------------------------------------------

_m = S.monodromy()
_SUB = {k: S.subquotient(T) for k, T in
        (("T1", S.T1), ("T2", S.T2), ("T0", _m["T0"]))}
_LOWER0 = S.cone_matrix(((0, 0), (1, 0), (0, 1)))
_LOWER0_INV = _LOWER0.inv()
assert all(x == int(x) for x in _LOWER0_INV)


def _annihilator(j, k):
    """Saturated integer basis of the annihilator of im(A_j^k - I), and the
    vector sum_{i<k} A_j^i v_j, for the paper's twists."""
    A = {1: _m["A1"], 2: _m["A2"]}[j]
    v = {1: S.EPS, 2: -S.EPS_PRIME}[j]
    P = S.int_kernel((A ** k - eye(4)).T)
    Ssum = zeros(4, 1)
    Pw = eye(4)
    for _ in range(k):
        Ssum += Pw * v
        Pw = Pw * A
    return P, Ssum


_ANN = {(j, k): _annihilator(j, k)
        for j, mj in ((1, 3), (2, 4)) for k in range(1, mj)}


PREAMBLE = r"""/-
SixSphereFacts, emitted by pyCICY.sixsphere_lean.

The finite layer of L. Alpoge's complex threefold X diffeomorphic to S^6
(https://alpo.ge/s6.pdf): the monodromy lattice, the invariant forms, the
twists, the cusp fan, the Kodaira subquotient, and the Picard arithmetic.
Every matrix below is generated from the same object pyCICY.sixsphere
computes with.

Mathlib-free. Tactics: `decide`, `omega`, `grind`, `simp only`. No `sorry`,
no `native_decide`. The headline theorem is not here; it is formalised over
Mathlib in github.com/plby/HopfProblem.
-/

namespace SixSphere

abbrev Mat := List (List Int)
abbrev Vec := List Int

def dot (a b : Vec) : Int := (List.zipWith (· * ·) a b).foldl (· + ·) 0
def col (B : Mat) (j : Nat) : Vec := B.map (fun r => r.getD j 0)
def width (A : Mat) : Nat := (A.headD []).length
def transpose (A : Mat) : Mat := (List.range (width A)).map (col A)
def mul (A B : Mat) : Mat :=
  A.map (fun r => (List.range (width B)).map (fun j => dot r (col B j)))
def app (A : Mat) (v : Vec) : Vec := A.map (fun r => dot r v)
def sub (A B : Mat) : Mat := List.zipWith (List.zipWith (· - ·)) A B
def ident (n : Nat) : Mat :=
  (List.range n).map (fun i => (List.range n).map (fun j => if i == j then 1 else 0))
def zero (n : Nat) : Mat := (List.range n).map (fun _ => List.replicate n 0)
def mpow (A : Mat) : Nat → Mat
  | 0 => ident (A.length)
  | n + 1 => mul A (mpow A n)
def entry (A : Mat) (i j : Nat) : Int := (A.getD i []).getD j 0
def trace2 (A : Mat) : Int := entry A 0 0 + entry A 1 1
def det2 (a b c d : Int) : Int := a * d - b * c
def det3 (a b c d e f g h i : Int) : Int :=
  a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
def det3m (A : Mat) : Int :=
  det3 (entry A 0 0) (entry A 0 1) (entry A 0 2) (entry A 1 0) (entry A 1 1)
       (entry A 1 2) (entry A 2 0) (entry A 2 1) (entry A 2 2)
def minor (A : Mat) (j : Nat) : Mat := (A.drop 1).map (fun r => r.eraseIdx j)
def det4 (A : Mat) : Int :=
  entry A 0 0 * det3m (minor A 0) - entry A 0 1 * det3m (minor A 1)
  + entry A 0 2 * det3m (minor A 2) - entry A 0 3 * det3m (minor A 3)
/-- the Pfaffian of a 4x4 antisymmetric matrix; eta ^ eta = 2 Pf vol -/
def pf (E : Mat) : Int :=
  entry E 0 1 * entry E 2 3 - entry E 0 2 * entry E 1 3 + entry E 0 3 * entry E 1 2

-- the data: generated from pyCICY.sixsphere
def T1 : Mat := %(T1)s
def T2 : Mat := %(T2)s
def T0 : Mat := %(T0)s
def A1 : Mat := %(A1)s
def A2 : Mat := %(A2)s
def M0 : Mat := %(M0)s
def eps : Vec := %(eps)s
def epsp : Vec := %(epsp)s
def dhat : Vec := %(dhat)s
def Q0 : Mat := %(Q0)s
def eta : Mat := %(eta)s
def B0 : Mat := %(B0)s
def I4 : Mat := ident 4
def Z4 : Mat := zero 4
def N : Mat := sub T0 I4
"""


def _data():
    return {"T1": lean_mat(S.T1), "T2": lean_mat(S.T2),
            "T0": lean_mat(_m["T0"]), "A1": lean_mat(_m["A1"]),
            "A2": lean_mat(_m["A2"]), "M0": lean_mat(_m["M0"]),
            "eps": lean_vec(S.EPS), "epsp": lean_vec(S.EPS_PRIME),
            "dhat": lean_vec(S.DELTA_HAT), "Q0": lean_mat(S.Q0),
            "eta": lean_mat(S.ETA), "B0": lean_mat(S.B0())}


# ---------------------------------------------------------------------------
# the facts
# ---------------------------------------------------------------------------

def _ann_fact(j, k):
    P, Ssum = _ANN[(j, k)]
    A = "A%d" % j
    mj = {1: 3, 2: 4}[j]
    name = "log_transform_free_p%d_power%d" % (j, k)
    lean = r"""/-- At p%(j)d, the %(k)d-th power of the logarithmic transform has no fixed
point: P annihilates the image of A%(j)d^%(k)d - I, and P applied to
(I + ... + A^%(km)d) v is not divisible by %(mj)d. (Saturation of P is checked in
Python.) -/
theorem %(name)s :
    mul %(P)s (sub (mpow %(A)s %(k)d) I4) = List.replicate %(r)d [0, 0, 0, 0] ∧
    (app %(P)s %(S)s).any (fun x => x %% %(mj)d != 0) = true := by decide""" % {
        "j": j, "k": k, "km": k - 1, "mj": mj, "name": name,
        "P": lean_mat(P), "A": A, "S": lean_vec(Ssum), "r": P.rows}
    return Fact(name, lean,
                lambda: dict(S.log_transform_free(j, {1: S.EPS,
                                                      2: -S.EPS_PRIME}[j])[1])
                [k] is False,
                "freeness of the paper's twist, one power at a time")


def _facts():
    q = _FORM_EQS[0]
    F = [
        Fact("det_T1_T2",
             "theorem det_T1_T2 : det4 T1 = 1 ∧ det4 T2 = 1 := by decide",
             lambda: S.T1.det() == 1 and S.T2.det() == 1,
             "Lemma 2.2 (i)"),
        Fact("order_T1",
             "/-- T1 has order exactly 3 (3 is prime). -/\n"
             "theorem order_T1 : mpow T1 3 = I4 ∧ T1 ≠ I4 := by decide",
             lambda: S.orders()["order_T1"] == 3,
             "Lemma 2.2 (ii)"),
        Fact("order_T2",
             "/-- T2 has order exactly 4: T2^4 = I but T2^2 != I. -/\n"
             "theorem order_T2 : mpow T2 4 = I4 ∧ mpow T2 2 ≠ I4 := by decide",
             lambda: S.orders()["order_T2"] == 4,
             "Lemma 2.2 (ii)"),
        Fact("cusp_unipotent",
             "/-- T1 T2 T0 = I and T0 = I + N with N != 0, N^2 = 0. -/\n"
             "theorem cusp_unipotent :\n"
             "    mul (mul T1 T2) T0 = I4 ∧ N ≠ Z4 ∧ mul N N = Z4 := by decide",
             lambda: S.orders()["T1T2T0_is_I"] and S.orders()["N_squared_zero"]
             and S.orders()["N_nonzero"],
             "Lemma 2.2 (iii)"),
        Fact("contragredients",
             "/-- A_j = (T_j^{-1})^t, i.e. A_j^t T_j = I; and A1 A2 M0 = I. -/\n"
             "theorem contragredients :\n"
             "    mul (transpose A1) T1 = I4 ∧ mul (transpose A2) T2 = I4 ∧\n"
             "    mul (transpose M0) T0 = I4 ∧ mul (mul A1 A2) M0 = I4 := by decide",
             lambda: S.orders()["A1A2M0_is_I"],
             "Lemma 2.4"),
        Fact("gamma_invariant",
             "/-- gamma o A = gamma: the first row of each dual matrix is e0. -/\n"
             "theorem gamma_invariant :\n"
             "    A1.headD [] = [1,0,0,0] ∧ A2.headD [] = [1,0,0,0] ∧\n"
             "    M0.headD [] = [1,0,0,0] := by decide",
             lambda: S.orders()["gamma_invariant"],
             "Lemma 2.4; the coinvariant Z of Lemma 2.7"),
        Fact("fixed_vectors",
             "theorem fixed_vectors :\n"
             "    app A1 eps = eps ∧ app A1 dhat = dhat ∧ app A2 epsp = epsp ∧\n"
             "    app A2 dhat = dhat ∧ eps.headD 0 = 1 ∧ epsp.headD 0 = 1 := by decide",
             lambda: S.fixed_lattices()["Lambda_A1_is_<eps,delta^>"]
             and S.fixed_lattices()["Lambda_A2_is_<eps',delta^>"],
             "Lemma 2.6 (containment; saturation is Python-only)"),
        Fact("Q0_invariant",
             "theorem Q0_invariant :\n"
             "    mul (mul (transpose T1) Q0) T1 = Q0 ∧\n"
             "    mul (mul (transpose T2) Q0) T2 = Q0 := by decide",
             lambda: S.invariant_forms()[0] in (S.Q0, -S.Q0),
             "Lemma 2.8, existence"),
        Fact("invariant_forms_unique",
             "/-- Every integer solution of T^t Q T = Q (T = T1, T2) is a multiple\n"
             "of Q0. The hypotheses are the entries of T^t Q T - Q, generated by\n"
             "sympy from the same matrices. -/\n"
             "theorem invariant_forms_unique (%s : Int)\n    %s :\n"
             "    q01 = 0 ∧ q02 = 0 ∧ q13 = 0 ∧ q23 = 0 ∧ q12 = 6 * q03 := by omega"
             % (" ".join(str(x) for x in q), _hyps(_FORM_EQS[1])),
             lambda: _equations_match(_FORM_EQS, lambda T, Q: T.T * Q * T)
             and len(S.invariant_forms()) == 1,
             "Lemma 2.8, uniqueness"),
        Fact("eta_invariant",
             "/-- eta = u^w + 6 gamma^delta is invariant and eta^eta = 12 vol. -/\n"
             "theorem eta_invariant :\n"
             "    mul (mul T1 eta) (transpose T1) = eta ∧\n"
             "    mul (mul T2 eta) (transpose T2) = eta ∧ 2 * pf eta = 12 := by decide",
             lambda: S.wedge_square(S.ETA) == 12
             and S.invariant_bivectors()[0] in (S.ETA, -S.ETA),
             "Sections 7 and 9"),
        Fact("invariant_bivectors_unique",
             "theorem invariant_bivectors_unique (%s : Int)\n    %s :\n"
             "    q01 = 0 ∧ q02 = 0 ∧ q13 = 0 ∧ q23 = 0 ∧ q03 = 6 * q12 := by omega"
             % (" ".join(str(x) for x in _BIV_EQS[0]), _hyps(_BIV_EQS[1])),
             lambda: _equations_match(_BIV_EQS, lambda T, E: T * E * T.T)
             and len(S.invariant_bivectors()) == 1,
             "the second solve, now proved"),
        Fact("b_form_indefinite",
             "/-- b(x,y) = Q0(x, N y) on (w, delta): diag(6, -1), determinant -6,\n"
             "so indefinite: no polarisation. -/\n"
             "theorem b_form_indefinite :\n"
             "    entry (mul Q0 N) 2 2 = 6 ∧ entry (mul Q0 N) 3 3 = -1 ∧\n"
             "    entry (mul Q0 N) 2 3 = 0 ∧ entry (mul Q0 N) 3 2 = 0 ∧\n"
             "    det2 6 0 0 (-1) < 0 := by decide",
             lambda: S.b_form()["gram"] == Matrix([[6, 0], [0, -1]]),
             "Lemma 2.8, Remark 3.23"),
        Fact("B0_unimodular",
             "/-- B0 is read off M0 - I, and is unimodular. -/\n"
             "theorem B0_unimodular :\n"
             "    entry (sub M0 I4) 2 0 = entry B0 0 0 ∧ entry (sub M0 I4) 2 1 = entry B0 0 1 ∧\n"
             "    entry (sub M0 I4) 3 0 = entry B0 1 0 ∧ entry (sub M0 I4) 3 1 = entry B0 1 1 ∧\n"
             "    det2 (entry B0 0 0) (entry B0 0 1) (entry B0 1 0) (entry B0 1 1) = 1 := by decide",
             lambda: abs(S.B0().det()) == 1,
             "Lemma 2.6; one dP6 at the cusp"),
        Fact("twists_and_p",
             "/-- gamma(v1) = 1, gamma(v2) = -1; p = -1; for X', p = -7. -/\n"
             "theorem twists_and_p :\n"
             "    eps.headD 0 = 1 ∧ (epsp.map (fun x => -x)).headD 0 = -1 ∧\n"
             "    12 * 0 - 4 * 1 - 3 * (-1) = (-1 : Int) ∧\n"
             "    12 * 0 - 4 * 1 - 3 * 1 = (-7 : Int) := by decide",
             lambda: S.twist_integers() == (0, 1, -1)
             and S.pi1_order(0, 1, -1) == 1 and S.pi1_order(0, 1, 1) == 7,
             "Theorem 7.17"),
        Fact("p_coprime_12",
             "/-- For every admissible twist and every l0, p is prime to 12. -/\n"
             "theorem p_coprime_12 (l0 l1 l2 : Int) (h1 : l1 % 3 ≠ 0) (h2 : l2 % 2 = 1) :\n"
             "    (12*l0 - 4*l1 - 3*l2) % 2 ≠ 0 ∧ (12*l0 - 4*l1 - 3*l2) % 3 ≠ 0 := by omega",
             lambda: not S.pi1_three_routes(box=2)["disagreements"],
             "Theorem 7.17, the last line"),
        Fact("seifert_det",
             "/-- The H_1 relation determinant of the Seifert space over S^2(3,4). -/\n"
             "theorem seifert_det (b1 b2 e0 : Int) :\n"
             "    det3 3 0 b1 0 4 b2 1 1 (-e0) = -(12*e0 + 4*b1 + 3*b2) := by\n"
             "  simp only [det3]; omega",
             lambda: not S.seifert_agreement(box=2)["disagreements"],
             "the order is |det| by Smith form, which is Python-only"),
        Fact("pi1_relation_det",
             "/-- The relation matrix of Theorem 7.17 in the basis (x, c):\n"
             "rows (3, -l1) and (4, l2 - 4 l0); its determinant is -p. -/\n"
             "theorem pi1_relation_det (l0 l1 l2 : Int) :\n"
             "    det2 3 (-l1) 4 (l2 - 4*l0) = -(12*l0 - 4*l1 - 3*l2) := by\n"
             "  simp only [det2]; omega",
             lambda: S.h1_from_presentation()["order"] == 1,
             "Theorem 7.17"),
        Fact("kodaira_subquotient",
             "/-- On <u,w>: orders 3 and 4 with traces -1 and 0; unipotent with\n"
             "n = 1 at the cusp. -/\n"
             "theorem kodaira_subquotient :\n"
             "    mpow %s 3 = ident 2 ∧ %s ≠ ident 2 ∧ trace2 %s = -1 ∧\n"
             "    mpow %s 4 = ident 2 ∧ mpow %s 2 ≠ ident 2 ∧ trace2 %s = 0 ∧\n"
             "    trace2 %s = 2 ∧ sub %s (ident 2) = [[0, -1], [0, 0]] := by decide"
             % ((lean_mat(_SUB["T1"]),) * 3 + (lean_mat(_SUB["T2"]),) * 3
                + (lean_mat(_SUB["T0"]),) * 2),
             lambda: S.lattice_route()["candidates"]["oo"] == ["I_1"],
             "Part II, the lattice route"),
        Fact("noether_selects",
             "/-- Of {IV, IV*} x {III, III*} x {I_1}, only IV* + III + I_1 has Euler\n"
             "sum divisible by 12. With the cusp's sign forgotten ({I_1, I_1*}),\n"
             "divisibility is not enough -- 7 + 8 + 9 = 24 -- which is why the\n"
             "paper uses deg K = 1 (Prop. 9.11). -/\n"
             "theorem noether_selects :\n"
             "    (([4, 8].flatMap fun a => [3, 9].map fun b => a + b + 1).filter\n"
             "        (fun e => e % 12 == 0)) = [12] ∧\n"
             "    (([1, 7].flatMap fun c => [4, 8].flatMap fun a => [3, 9].map fun b =>\n"
             "        a + b + c).filter (fun e => e % 12 == 0)) = [12, 24] := by decide",
             lambda: S.lattice_route()["noether_selects"] == [("IV*", "III", "I_1")],
             "Part II and Prop. 9.11"),
        Fact("fan_unimodular",
             "/-- Every cone of the infinite A2 fan is unimodular: lower triangles\n"
             "have determinant +1 and upper -1, for all (x, y). -/\n"
             "theorem fan_unimodular (x y : Int) :\n"
             "    det3 x y 1 (x+1) y 1 x (y+1) 1 = 1 ∧\n"
             "    det3 (x+1) y 1 x (y+1) 1 (x+1) (y+1) 1 = -1 := by\n"
             "  simp only [det3]; constructor <;> grind",
             lambda: S.fan_checks()["dets"] == {"lower": {1}, "upper": {-1}},
             "Lemma 4.2 (i), on all of Z^2 rather than a patch"),
        Fact("hexagon_dP6",
             "/-- The six rays around a vertex: consecutive determinants 1, and\n"
             "v_{i-1} + v_{i+1} = v_i, so every boundary curve is a (-1)-curve. -/\n"
             "theorem hexagon_dP6 :\n"
             "    let r : List (Int × Int) := %s\n"
             "    (List.range 6).all (fun i =>\n"
             "      let a := r.getD i (0,0); let b := r.getD ((i+1) %% 6) (0,0)\n"
             "      let c := r.getD ((i+5) %% 6) (0,0)\n"
             "      det2 a.1 a.2 b.1 b.2 == 1 && (c.1 + b.1, c.2 + b.2) == a) = true := by\n"
             "  decide"
             % ("[" + ", ".join("(%d, %d)" % v for v in S.HEXAGON_RAYS) + "]"),
             lambda: S.hexagon_surface()["self_intersections"] == [-1] * 6
             and set(S.hexagon_surface()["consecutive_dets"]) == {1},
             "Lemma 4.2 (iv); the same order as the HopfProblem hexagonRay"),
        Fact("cstar_weights",
             "/-- At the triple point of the lower cone at 0, the dual basis (the\n"
             "chart coordinates) pairs with the cocharacter delta^ <-> (0,1,0) to\n"
             "weights (-1, 0, 1): tangent 0, normal (+1, -1). -/\n"
             "theorem cstar_weights :\n"
             "    mul %s %s = ident 3 ∧\n"
             "    app (transpose %s) [0, 1, 0] = [-1, 0, 1] := by decide"
             % (lean_mat(_LOWER0), lean_mat(_LOWER0_INV), lean_mat(_LOWER0_INV)),
             lambda: S.cstar_fixed_locus()["weights_at_triple_points"]
             == [(-1, 0, 1)],
             "Proposition 9.24"),
        Fact("picard_arithmetic",
             "/-- In <H, S1, S2> with 3 S1 = 4 S2 = H: S1 = 4u, S2 = 3u, H = 12u;\n"
             "K_X = -H + 2 S2 = -6u; the classical 2 S1 + S2 = 11u is not a multiple\n"
             "of 12u; and the minors of the relation matrix are coprime. -/\n"
             "theorem picard_arithmetic :\n"
             "    3 * 4 = (12 : Int) ∧ 4 * 3 = (12 : Int) ∧ -12 + 2 * 3 = (-6 : Int) ∧\n"
             "    (2 * 4 + 3 : Int) % 12 ≠ 0 ∧\n"
             "    det2 3 0 0 4 = 12 ∧ det2 3 (-1) 0 (-1) = -3 ∧ det2 0 (-1) 4 (-1) = 4 ∧\n"
             "    (0 * 12 + 1 * (-3) + 1 * 4 : Int) = 1 := by decide",
             lambda: S.picard_arithmetic()["K"] == -6
             and S.picard_arithmetic()["difference"] == 11,
             "Theorem 9.1 (iii) and the footnote to Section 10"),
        Fact("K_normal_form_unique",
             "/-- The only way to write K_X = -6u as k H + a S1 + b S2 with\n"
             "0 <= a < 3, 0 <= b < 4 is f^*O(-1) + 2 S2. -/\n"
             "theorem K_normal_form_unique (k a b : Int) (ha : 0 ≤ a ∧ a < 3)\n"
             "    (hb : 0 ≤ b ∧ b < 4) (h : 12*k + 4*a + 3*b = -6) :\n"
             "    k = -1 ∧ a = 0 ∧ b = 2 := by omega",
             lambda: S.picard_arithmetic()["normal_forms"] == [(-1, 0, 2)],
             "no normal form with exponents (2, 3) exists"),
        Fact("hodge_bundle_degree",
             "/-- 12 a_j / m_j are the Euler numbers 8 and 3, and with the cusp's 1\n"
             "the local exponents sum to deg f_* omega = 1. -/\n"
             "theorem hodge_bundle_degree :\n"
             "    12 * 2 = 8 * 3 ∧ 12 * 1 = 3 * 4 ∧ 8 + 3 + 1 = 12 := by decide",
             lambda: S.hodge_bundle_degree()["matches_fibre_euler"],
             "Proposition 9.11; arithmetic on computed exponents"),
    ]
    for (j, k) in sorted(_ANN):
        F.append(_ann_fact(j, k))
    F += _search_facts()
    return F


def _search_facts():
    """Part VII: the gcd obstruction, the paper conjugator, the mirror."""
    from . import sixsphere_search as X
    rep1, rep2 = X.CLASS_REP
    P, Pm = X.PAPER_CONJUGATOR, X.MIRROR_CONJUGATOR
    return [
        Fact("gcd_obstruction_3_6",
             "/-- For (3,6,inf), p = 18 l0 - 6 l1 - 3 l2 is divisible by 3 for\n"
             "every twist: |p| = 1 is impossible. -/\n"
             "theorem gcd_obstruction_3_6 (l0 l1 l2 : Int) :\n"
             "    (18*l0 - 6*l1 - 3*l2) % 3 = 0 := by omega",
             lambda: X.gcd_obstruction(3, 6)["min_positive_|p|"] == 3,
             "Part VII, the triangle-group survey"),
        Fact("gcd_obstruction_4_6",
             "/-- For (4,6,inf), p = 24 l0 - 6 l1 - 4 l2 is even for every twist. -/\n"
             "theorem gcd_obstruction_4_6 (l0 l1 l2 : Int) :\n"
             "    (24*l0 - 6*l1 - 4*l2) % 2 = 0 := by omega",
             lambda: X.gcd_obstruction(4, 6)["min_positive_|p|"] == 2,
             "Part VII, the triangle-group survey"),
        Fact("paper_conjugator",
             "/-- The explicit P (det -1) carrying the paper's (T1, T2) onto the\n"
             "representative of the only class reaching |p| = 1. -/\n"
             "theorem paper_conjugator :\n"
             "    mul %s T1 = mul %s %s ∧\n"
             "    mul %s T2 = mul %s %s ∧ det4 %s = -1 := by decide"
             % (lean_mat(P), lean_mat(rep1), lean_mat(P),
                lean_mat(P), lean_mat(rep2), lean_mat(P), lean_mat(P)),
             lambda: all(X.stored_constants_agree().values()),
             "Part VII; the search itself is Python-only"),
        Fact("lattice_achiral",
             "/-- Reversing the base orientation gives (T1^-1, T2^-1); an explicit\n"
             "P of determinant 1 conjugates (T1, T2) to it: the lattice data is\n"
             "achiral. -/\n"
             "theorem lattice_achiral :\n"
             "    mul T1 %s = I4 ∧ mul T2 %s = I4 ∧\n"
             "    mul %s T1 = mul %s %s ∧ mul %s T2 = mul %s %s ∧\n"
             "    det4 %s = 1 := by decide"
             % (lean_mat(S.T1.inv()), lean_mat(S.T2.inv()),
                lean_mat(Pm), lean_mat(S.T1.inv()), lean_mat(Pm),
                lean_mat(Pm), lean_mat(S.T2.inv()), lean_mat(Pm),
                lean_mat(Pm)),
             lambda: X.lattice_mirror()["verified"],
             "Part VII, connections"),
    ]


def _equations_match(eqs_pair, transform):
    """The Lean hypotheses are exactly the nonzero entries of the invariance
    equations, recomputed here from the matrices."""
    q, eqs = eqs_pair
    Q = S._antisym(q)
    again = []
    for T in (S.T1, S.T2):
        D = transform(T, Q) - Q
        for i in range(4):
            for j in range(i + 1, 4):
                if expand(D[i, j]) != 0:
                    again.append(expand(D[i, j]))
    return again == eqs and len(eqs) > 0


FACTS = _facts()


# ---------------------------------------------------------------------------
# emission and checking
# ---------------------------------------------------------------------------

def lean_source(facts=None):
    """The whole Lean file, as text."""
    facts = FACTS if facts is None else facts
    body = "\n\n".join(f.lean for f in facts)
    axioms = "\n".join("#print axioms %s.%s" % (NAMESPACE, f.name)
                       for f in facts)
    return (PREAMBLE % _data() + "\n" + body + "\n\nend " + NAMESPACE
            + "\n\n" + axioms + "\n")


def check_python(facts=None):
    """Evaluate every fact as :mod:`pyCICY.sixsphere` computes it."""
    facts = FACTS if facts is None else facts
    out = {}
    for f in facts:
        try:
            out[f.name] = bool(f.python())
        except Exception as exc:                                 # noqa: BLE001
            out[f.name] = "error: %s" % exc
    return out


def check_lean(path=None, timeout=900, require=False, source=None):
    """Kernel-check the file; see :func:`nariai_lean.check_source`."""
    return check_source(lean_source() if source is None else source,
                        NAMESPACE, FILENAME, path=path, timeout=timeout,
                        require=require, allowed=STANDARD_AXIOMS)


def status(lean_path=None):
    """Per fact: computed; machine-checked or not; axioms; choice-free or not.

    ``machine_checked`` is True only when a kernel ran, the file compiled,
    the fact's axiom report was read, and every axiom in it is on the
    allow-list :data:`STANDARD_AXIOMS`. Nothing here ever claims a check that
    did not run.
    """
    py = check_python()
    ln = check_lean(path=lean_path)
    out = {}
    for f in FACTS:
        ax = ln["axioms"].get(f.name)
        checked = bool(ln["available"] and ln["ok"] and ax is not None
                       and f.name not in ln.get("unsound", []))
        out[f.name] = {
            "computed": py.get(f.name),
            "machine_checked": checked,
            "axioms": ax,
            "standard_axioms_only": (ax is not None
                                     and all(a in STANDARD_AXIOMS for a in ax)),
            "choice_free": ax is not None and "Classical.choice" not in ax,
            "note": f.note,
        }
    return {"facts": out, "toolchain": ln["available"], "lean_ok": ln["ok"],
            "errors": ln.get("errors", []),
            "all_computed": all(v["computed"] is True for v in out.values()),
            "all_machine_checked": all(v["machine_checked"]
                                       for v in out.values()),
            "n_facts": len(out),
            "n_choice_free": sum(1 for v in out.values() if v["choice_free"])}
