r"""
pyCICY.theories.nariai_lean -- the finite algebra of the Nariai horizon, emitted
as Lean 4 and kernel-checked.

Why this module exists
----------------------
:mod:`pyCICY.theories.nariai` verifies the interference-functional vanishing
numerically: build a boost-positive-frequency packet, integrate, and read
``|I2|/I1 = 1e-15``. That is evidence, and it is the weakest kind of claim in
this package -- a float near zero, in a package whose entire discipline is
that a number computed twice by unrelated routes is a test and a number
computed once is not.

The vanishing is not really analysis. ``I2`` is the zero-frequency component
of ``(d_u chi)^2``, whose spectrum is the **sumset** of the packet's own
spectrum. So ``I2 = 0`` exactly when no two frequencies of the packet sum to
zero: never, for a finite set of positive integers; always, for a symmetric
one. That is a statement about finite sets of integers, and a finite statement
about integers can be *proved*, not measured.

So this module does two things. It computes the sumset layer exactly, in
Python, with integers and no floating point in any statement. And it emits the
same facts as Mathlib-free Lean 4 and asks the Lean kernel to check them,
which adds a status this package has not had before:

    exact  <  computed twice by unrelated routes  <  machine-checked

Not a new category of *quantity* -- the ledger's four entries are unchanged --
but a new grade of confidence in a claim that is already exact.

What is emitted
---------------
Only finite algebra: exponent arithmetic, sumsets of concrete and of
universally quantified integer lists, and the ring identities behind the
one-mode sum rule. The analysis stays where it belongs. A theorem about
improper integrals of distributions is not going to be settled by ``decide``,
and this module does not pretend otherwise.

The Lean is deliberately Mathlib-free, so it depends on nothing but the core
toolchain: ``decide`` for finite checks, ``omega`` for linear integer
arithmetic, ``simp`` with core lemmas for the rest. ``ring`` is a Mathlib
tactic and is not available, which is why the ring identities below are
written in the shapes ``omega`` can normalise.

Toolchain
---------
:func:`check_lean` needs a ``lean`` executable. If there is none it says so
and returns rather than raising, because emitting the file is useful even
where it cannot be checked -- but a *claim* of machine-checking is only made
when the kernel actually ran. :func:`status` reports both.

Reference
---------
B. S. Hartshorn, *Gravity from Relative Entropy: Jackiw-Teitelboim Dynamics on
the Nariai Horizon*, and its companion ``NariaiFacts.lean``, which proves the
same facts independently. That file is not read or copied here; it is what
makes this module's output checkable against something other than itself.
"""

import os
import shutil
import subprocess
import tempfile

import numpy as np

__all__ = ["Fact", "FACTS", "lean_source", "check_python", "check_lean",
           "check_source", "parse_axioms",
           "status", "sumset", "sumset_has_zero", "sumset_range",
           "spectrum_indices", "exact_packet", "functionals", "occupied_bins",
           "detector_ratio", "energy_gap", "NoToolchain"]


class NoToolchain(RuntimeError):
    """Raised only when a caller insists on a kernel check with no kernel."""


# ---------------------------------------------------------------------------
# the exact layer: sumsets of integer spectra
# ---------------------------------------------------------------------------

def spectrum_indices(jmin=20, jmax=40):
    """The integer frequency indices of the packet.

    Integers, not floats, and bounded below by ``jmin >= 1``. On a periodic
    grid these are grid harmonics, so the discrete sums that follow are exact
    Fourier coefficients rather than quadrature approximations, and "``I2`` is
    the zero bin" is an identity rather than a limit.
    """
    if jmin < 1:
        raise ValueError("a positive spectrum needs jmin >= 1")
    return list(range(jmin, jmax + 1))


def sumset(ws):
    """``{a + b : a, b in ws}``, as a sorted list of integers."""
    return sorted({a + b for a in ws for b in ws})


def sumset_has_zero(ws):
    """Whether any two elements sum to zero. The whole content of ``I2 = 0``."""
    return 0 in set(sumset(ws))


def sumset_range(ws):
    """Where the spectrum of ``(d_u chi)^2`` lives: ``[2 min, 2 max]``."""
    return 2 * min(ws), 2 * max(ws)


# ---------------------------------------------------------------------------
# the packet the facts are about
# ---------------------------------------------------------------------------

def exact_packet(jmin=20, jmax=40, length=2 * np.pi, points=2048,
                 centre=30.0, width=6.0):
    """``d_u chi`` on a periodic grid, from a strictly positive integer spectrum.

    Every frequency is a grid harmonic, so nothing is truncated and nothing is
    approximated: the spectrum is exactly the index list, as a property of
    integers.
    """
    idx = np.array(spectrum_indices(jmin, jmax), dtype=float)
    amp = np.exp(-((idx - centre) / width) ** 2)
    u = np.arange(points) * length / points
    w = 2 * np.pi * idx / length
    dchi = np.zeros(points, dtype=complex)
    for a, ww in zip(amp, w):
        dchi += (-1j * ww * a) * np.exp(-1j * ww * u)
    return u, dchi


def functionals(dchi, length=2 * np.pi):
    """``(I1, I2)``: the norm, and the interference functional."""
    n = len(dchi)
    return (float(np.sum(np.abs(dchi) ** 2) * length / n),
            complex(np.sum(dchi ** 2) * length / n))


def occupied_bins(dchi, tol=1e-8):
    """Which frequencies ``(d_u chi)^2`` actually carries, from its transform.

    Returned unaliased, so the result can be compared directly with
    :func:`sumset`. The comparison is the point: the bins are measured from
    the signal, the sumset is computed from the integers, and they agree.
    """
    f = np.fft.fft(dchi ** 2)
    n = len(f)
    big = np.abs(f) > tol * np.max(np.abs(f))
    return sorted(int(n - b) if b > n // 2 else int(b)
                  for b in np.where(big)[0])


def detector_ratio(eps):
    """``|I2|/I1`` for a negative-frequency admixture of weight ``eps``.

    The graded wedge-locality detector: 0 for boost-adapted data, 1 for
    boost-blind data, and ``2 eps / (1 + eps^2) = tanh(2 artanh eps)``
    between. The bound ``|I2|/I1 <= 1`` and its saturation only at
    ``eps = 1`` are the two facts the Lean layer proves.
    """
    return 2.0 * eps / (1.0 + eps ** 2)


def energy_gap(nu, s, t):
    """``nu(s^2 + t^2) - 2 nu s t = nu (s - t)^2``: the one-mode trace gap.

    Written in integers so the identity is exactly the one Lean checks. The
    physical content is that the gap is non-negative for every squeeze and
    vanishes only when there is none.
    """
    return nu * s * s + nu * t * t - (nu * (s * t) + nu * (s * t))


# ---------------------------------------------------------------------------
# facts: each one computed in Python and stated in Lean
# ---------------------------------------------------------------------------

class Fact(object):
    """One finite claim, in two languages.

    ``python`` is a callable returning True when the claim holds as computed
    here; ``lean`` is the theorem text whose proof the kernel will check. The
    pairing is the point: a fact that is only in Python is unproved, and a
    fact that is only in Lean might be about something other than what the
    package computes.
    """

    def __init__(self, name, lean, python, note=""):
        self.name = name
        self.lean = lean
        self.python = python
        self.note = note

    def __repr__(self):
        return "Fact(%s)" % self.name


_PACKET = spectrum_indices(20, 24)          # small enough for `decide`
_PACKET_LEAN = "[" + ",".join(str(k) for k in _PACKET) + "]"


PREAMBLE = r"""/-
NariaiFacts, emitted by pyCICY.theories.nariai_lean.

Mathlib-free: the only tactics used are `decide`, `omega`, `simp` with core
lemmas, and explicit term proofs. Nothing here depends on `sorryAx` or on
`Classical.choice`.

What is proved is the finite algebra of Appendix B: the exponent balance that
makes the modular weight cancel, the sumset mechanism behind the vanishing of
the interference functional, and the ring identities behind the one-mode sum
rule. The analysis is not here and is not claimed.
-/

namespace Nariai

/-- The three exponents of `exp(kappa u)` in the modular weight, the boost
Jacobian, and the squared first derivative. -/
def wExp : Int := -1
def derExp : Int := 2
def jacExp : Int := -1

/-- Whether any two entries of a spectrum sum to zero. This is the whole
content of the vanishing: `I2` is the zero-frequency component of the square,
whose spectrum is the sumset. -/
def sumsetHasZero (ws : List Int) : Bool :=
  ws.any (fun a => ws.any (fun b => a + b == 0))

/-- The packet's spectrum, as the integers it actually is. -/
def packetSpectrum : List Int := %s
""" % _PACKET_LEAN


FACTS = [
    Fact(
        "modular_weight_balances",
        r"""/-- The weight, the Jacobian and the squared derivative cancel:
in exponents of `exp(kappa u)` the three factors are `-1, +2, -1`. -/
theorem modular_weight_balances : wExp + derExp + jacExp = 0 := by decide""",
        lambda: (-1) + 2 + (-1) == 0,
        "the identity that makes the boost measure invariant"),

    Fact(
        "weight_without_derivatives_fails",
        r"""/-- And they do not cancel without the derivative factors, which is
why the naive form of the identity is false. -/
theorem weight_without_derivatives_fails : wExp + jacExp ≠ 0 := by decide""",
        lambda: (-1) + (-1) != 0,
        "the negative half: the check can fail, and does, if mis-stated"),

    Fact(
        "weight_without_derivatives_residue",
        r"""/-- The residue is exactly `-2`, which is the missing
`exp(2 kappa u)`. -/
theorem weight_without_derivatives_residue : wExp + jacExp = -2 := by decide""",
        lambda: (-1) + (-1) == -2,
        "and by how much it fails"),

    Fact(
        "positive_spectrum_avoids_zero",
        r"""/-- A strictly positive spectrum has no two elements summing to
zero, for every finite spectrum, not merely the one computed. -/
theorem positive_spectrum_avoids_zero (ws : List Int) (h : ∀ w ∈ ws, 0 < w) :
    sumsetHasZero ws = false := by
  simp [sumsetHasZero]
  intro a ha b hb
  have h1 := h a ha
  have h2 := h b hb
  omega""",
        lambda: all(not sumset_has_zero(spectrum_indices(j, j + 6))
                    for j in range(1, 40)),
        "the general theorem, quantified over all finite positive spectra"),

    Fact(
        "symmetric_spectrum_hits_zero",
        r"""/-- A symmetric spectrum always does, which is the boost-blind
case where the detector reads one. -/
theorem symmetric_spectrum_hits_zero (w : Int) :
    sumsetHasZero [w, -w] = true := by
  simp [sumsetHasZero]
  omega""",
        lambda: all(sumset_has_zero([w, -w]) for w in range(1, 40)),
        "the opposite endpoint, also for every w"),

    Fact(
        "packet_positive",
        r"""/-- The packet this package actually builds has a positive
spectrum, as a property of integers with no tail to truncate. -/
theorem packet_positive : ∀ w ∈ packetSpectrum, 0 < w := by decide""",
        lambda: all(w > 0 for w in _PACKET),
        "the hypothesis, discharged on the object computed"),

    Fact(
        "packet_avoids_zero",
        r"""/-- Hence its interference functional vanishes identically. -/
theorem packet_avoids_zero : sumsetHasZero packetSpectrum = false := by decide""",
        lambda: not sumset_has_zero(_PACKET),
        "the conclusion, for that same object"),

    Fact(
        "packet_sumset_lower_bound",
        r"""/-- And every frequency of the square is at least twice the
lowest frequency of the packet, so zero is not merely absent, it is far. -/
theorem packet_sumset_lower_bound :
    packetSpectrum.all (fun a => packetSpectrum.all (fun b => %d ≤ a + b)) = true := by
  decide""" % (2 * min(_PACKET)),
        lambda: min(sumset(_PACKET)) == 2 * min(_PACKET),
        "the quantitative version of the same statement"),

    Fact(
        "sq_nonneg",
        r"""/-- Squares are non-negative over `Int`, proved rather than
imported, since Mathlib is not available. -/
theorem sq_nonneg (x : Int) : 0 ≤ x * x := by
  rcases Int.le_total 0 x with h | h
  · exact Int.mul_nonneg h h
  · have h2 : 0 ≤ (-x) * (-x) := Int.mul_nonneg (by omega) (by omega)
    have e : (-x) * (-x) = x * x := by
      simp [Int.neg_mul, Int.mul_neg]
    omega""",
        lambda: all(x * x >= 0 for x in range(-50, 51)),
        "the one lemma everything below needs"),

    Fact(
        "detector_bounded",
        r"""/-- The wedge-locality detector never exceeds one. -/
theorem detector_bounded (p q : Int) : 2*(p*q) ≤ q*q + p*p := by
  have h : 0 ≤ (p - q) * (p - q) := sq_nonneg (p - q)
  simp [Int.mul_sub, Int.mul_comm] at h
  omega""",
        lambda: all(2 * p * q <= q * q + p * p
                    for p in range(-20, 21) for q in range(-20, 21)),
        "the bound |I2|/I1 <= 1"),

    Fact(
        "detector_saturates_iff_equal",
        r"""/-- And reaches one exactly on real data, where the positive and
negative parts have equal weight. -/
theorem detector_saturates_iff_equal (p q : Int) :
    2*(p*q) = q*q + p*p ↔ p = q := by
  constructor
  · intro h
    have hz : 0 ≤ (p - q) * (p - q) := sq_nonneg (p - q)
    simp [Int.mul_sub, Int.mul_comm] at hz
    have e : (p - q) * (p - q) = 0 := by
      simp [Int.mul_sub, Int.mul_comm]; omega
    rcases Int.mul_eq_zero.mp e with h0 | h0 <;> omega
  · intro h
    subst h
    omega""",
        lambda: all(((2 * p * q == q * q + p * p) == (p == q))
                    for p in range(-20, 21) for q in range(-20, 21)),
        "saturation only at eps = 1, the no-wedge-local-squeeze statement"),

    Fact(
        "energy_gap_identity",
        r"""/-- The one-mode trace gap, before any constraint. Written as a
sum rather than with a coefficient so it stays inside what `omega`
normalises. -/
theorem energy_gap_identity (nu s t : Int) :
    nu*s*s + nu*t*t - (nu*(s*t) + nu*(s*t)) = nu*((s-t)*(s-t)) := by
  simp [Int.mul_sub, Int.sub_mul, Int.mul_add, Int.add_mul,
        Int.mul_assoc, Int.mul_comm, Int.mul_left_comm]
  omega""",
        lambda: all(energy_gap(nu, s, t) == nu * (s - t) ** 2
                    for nu in range(-6, 7)
                    for s in range(-6, 7) for t in range(-6, 7)),
        "the closed form of the squeezed energy cost"),

    Fact(
        "energy_gap_nonneg",
        r"""/-- So a wedge-local squeeze never lowers the boost energy, for
any physical thermal state. -/
theorem energy_gap_nonneg (nu s t : Int) (h : 0 ≤ nu) :
    0 ≤ nu*((s-t)*(s-t)) :=
  Int.mul_nonneg h (sq_nonneg (s-t))""",
        lambda: all(nu * (s - t) ** 2 >= 0 for nu in range(0, 7)
                    for s in range(-6, 7) for t in range(-6, 7)),
        "positivity of the cost"),

    Fact(
        "energy_gap_zero_iff_trivial",
        r"""/-- And it vanishes only when there is no squeeze at all. -/
theorem energy_gap_zero_iff_trivial (nu s t : Int) (hnu : 0 < nu)
    (h : nu*((s-t)*(s-t)) = 0) : s = t := by
  rcases Int.mul_eq_zero.mp h with h0 | h0
  · omega
  · rcases Int.mul_eq_zero.mp h0 with h1 | h1 <;> omega""",
        lambda: all((nu * (s - t) ** 2 == 0) == (s == t)
                    for nu in range(1, 7)
                    for s in range(-6, 7) for t in range(-6, 7)),
        "the converse, which is what makes the bound sharp"),

    Fact(
        "squeeze_det_identity",
        r"""/-- The determinant of the squeezed covariance, which is what the
von Neumann entropy depends on. -/
theorem squeeze_det_identity (nu s t : Int) :
    (nu*s*s) * (nu*t*t) = (nu*nu)*((s*t)*(s*t)) := by
  simp [Int.mul_assoc, Int.mul_comm, Int.mul_left_comm]""",
        lambda: all((nu * s * s) * (nu * t * t)
                    == (nu * nu) * ((s * t) * (s * t))
                    for nu in range(-5, 6)
                    for s in range(-5, 6) for t in range(-5, 6)),
        "the algebra behind dS_vN = 0"),

    Fact(
        "squeeze_det_invariant",
        r"""/-- Squeezing is symplectic, so the determinant -- hence the
symplectic eigenvalue, hence the von Neumann entropy -- does not move. -/
theorem squeeze_det_invariant (nu s t : Int) (h : s*t = 1) :
    (nu*s*s) * (nu*t*t) = nu*nu := by
  rw [squeeze_det_identity, h]; simp""",
        lambda: all((nu * s * s) * (nu * t * t) == nu * nu
                    for nu in range(-5, 6) for s, t in ((1, 1), (-1, -1))),
        "the entropy is unchanged; only the energy moves"),
]


# ---------------------------------------------------------------------------
# emission and checking
# ---------------------------------------------------------------------------

def lean_source(facts=None):
    """The whole Lean file, as text."""
    facts = FACTS if facts is None else facts
    body = "\n\n".join(f.lean for f in facts)
    axioms = "\n".join("#print axioms Nariai.%s" % f.name for f in facts)
    return PREAMBLE + "\n" + body + "\n\nend Nariai\n\n" + axioms + "\n"


def check_python(facts=None):
    """Evaluate every fact as this package computes it.

    Returns the per-fact verdicts. A fact that fails here is a fact the Lean
    file would be proving about something other than what the package does,
    which is the failure mode worth catching.
    """
    facts = FACTS if facts is None else facts
    out = {}
    for f in facts:
        try:
            out[f.name] = bool(f.python())
        except Exception as exc:                                 # noqa: BLE001
            out[f.name] = "error: %s" % exc
    return out


def lean_executable():
    """Where the Lean kernel is, or None."""
    return (os.environ.get("PYCICY_LEAN")
            or shutil.which("lean"))


def parse_axioms(output, namespace):
    """Read ``#print axioms`` lines for theorems in ``namespace``.

    Lean prints one report per theorem, ``'Ns.name' depends on axioms: [a, b]``
    or ``'Ns.name' does not depend on any axioms``; a long list can wrap onto
    following lines, so a report is read until its closing bracket.
    """
    axioms = {}
    lines = output.splitlines()
    prefix = "'%s." % namespace
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line.startswith(prefix):
            continue
        name = line.split("'")[1].split(".", 1)[1]
        if "does not depend on any axioms" in line:
            axioms[name] = []
            continue
        text = line.split("axioms:", 1)[-1]
        while "]" not in text and i < len(lines):
            text += " " + lines[i].strip()
            i += 1
        deps = text.strip().strip("[]")
        axioms[name] = [d.strip() for d in deps.split(",") if d.strip()]
    return axioms


def check_source(src, namespace, filename, path=None, timeout=600,
                 require=False, allowed=("propext", "Quot.sound")):
    """Write a Lean file and ask the kernel to check it; the generic core.

    ``allowed`` is an allow-list: a theorem counts as machine-checked only if
    every axiom in its report is on it. It is an allow-list and not a
    deny-list on purpose. A deny-list of ``sorryAx`` and
    ``Lean.ofReduceBool`` misses what Lean 4.33 actually emits for
    ``native_decide`` -- a fresh auxiliary axiom per proof, named
    ``<thm>._native.native_decide.ax_N`` -- and misses a plain user-declared
    ``axiom``. The report is parsed, never assumed.
    """
    exe = lean_executable()
    if exe is None:
        if require:
            raise NoToolchain(
                "no lean executable found. Set PYCICY_LEAN or put lean on "
                "PATH. The file can still be emitted; what cannot be done "
                "without a kernel is claiming it checks.")
        return {"available": False, "ok": None, "axioms": {},
                "output": "", "source": src}
    target, tmp = path, None
    if target is None:
        tmp = tempfile.mkdtemp(prefix="pycicy-lean-")
        target = os.path.join(tmp, filename)
    with open(target, "w") as fh:
        fh.write(src)
    proc = subprocess.run([exe, target], capture_output=True, text=True,
                          timeout=timeout)
    out = (proc.stdout or "") + (proc.stderr or "")
    axioms = parse_axioms(out, namespace)
    bad = sorted(n for n, deps in axioms.items()
                 if any(d not in allowed for d in deps))
    errors = [l for l in out.splitlines() if ": error:" in l]
    return {"available": True,
            "ok": proc.returncode == 0 and not errors and not bad,
            "returncode": proc.returncode,
            "errors": errors,
            "axioms": axioms,
            "unsound": bad,
            "checked": sorted(axioms),
            "path": target,
            "output": out,
            "source": src}


def check_lean(path=None, timeout=600, require=False):
    """Write the Lean file and ask the kernel to check it.

    Returns a record with ``available`` (was there a toolchain), ``ok`` (did
    it compile), ``axioms`` (what each theorem depends on), and the raw
    output. With ``require=True`` the absence of a toolchain raises instead
    of being reported, for callers that want the check or nothing.

    The axiom report is the part that matters. A Lean file that compiles may
    still be resting on ``sorryAx``, and a proof resting on ``sorryAx`` is not
    a proof. This function reads the ``#print axioms`` lines and refuses to
    call anything machine-checked that mentions it -- or, for this module,
    ``Classical.choice``, since these facts claim to be constructive. The
    policy is an allow-list (``propext``, ``Quot.sound``), so an axiom this
    function has never heard of disqualifies too.
    """
    return check_source(lean_source(), "Nariai", "NariaiFacts.lean",
                        path=path, timeout=timeout, require=require,
                        allowed=("propext", "Quot.sound"))


def status(lean_path=None):
    """The grade of every fact: computed, and machine-checked or not.

    This is the new rung. The package already distinguishes exact from
    needs-a-metric from does-not-exist from true-by-argument. Those are
    categories of *quantity*. This is a grade of *confidence* in a claim that
    is already exact, and it is reported separately for that reason.
    """
    py = check_python()
    ln = check_lean(path=lean_path)
    out = {}
    for f in FACTS:
        checked = (ln["available"] and ln["ok"]
                   and f.name in ln["axioms"]
                   and f.name not in ln.get("unsound", []))
        out[f.name] = {
            "computed": py.get(f.name),
            "machine_checked": bool(checked),
            "axioms": ln["axioms"].get(f.name),
            "note": f.note,
        }
    return {"facts": out,
            "toolchain": ln["available"],
            "all_computed": all(v["computed"] is True for v in out.values()),
            "all_machine_checked": all(v["machine_checked"]
                                       for v in out.values())}
