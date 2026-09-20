"""
Tests for pyCICY.theories.nariai_lean.

The first module in the package whose claims are not merely computed twice but
*proved*. The tests therefore check three different things, and the third is
new:

  [1] sumset       the exact layer. I2 is the zero-frequency component of the
                   square of the packet, whose spectrum is the sumset of the
                   packet's own, so the vanishing is a statement about finite
                   sets of integers. Positive spectra never contain zero in
                   their sumset; symmetric ones always do
  [2] packet       and the integers describe the object actually built: the
                   occupied DFT bins of (d_u chi)^2, measured from the signal,
                   equal the sumset, computed from the indices. The measured
                   |I2|/I1 is roundoff for every jmin, because nothing is
                   being truncated
  [3] detector     the graded wedge-locality detector and the energy gap, as
                   integer identities, matching the closed forms
  [4] python       every emitted fact, evaluated as this package computes it.
                   A fact failing here would mean the Lean file proves
                   something about an object other than the one in use
  [5] lean         the emitted file itself: it parses, it contains a theorem
                   per fact, and where a toolchain exists the kernel accepts
                   it with no sorryAx and no Classical.choice. Without a
                   toolchain the test says so and does not claim otherwise
  [6] status       the new grade -- computed, and machine-checked or not --
                   reported separately from the ledger's categories, because
                   it is a grade of confidence rather than a kind of quantity

Run with:  python3 tests/test_nariai_lean.py
       or: python3 run_tests.py

To exercise the kernel, put `lean` on PATH or set PYCICY_LEAN. Without one,
the Lean section reports "no toolchain" and passes: emitting is useful even
where checking is impossible, and the module never claims a check it did not
run.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from pyCICY.theories import nariai_lean as NL

FAILURES = []


def check(name, got, want):
    ok = got == want
    print("  {:<58} {:>14} {}".format(name, str(got)[:14],
                                      "ok" if ok else "FAIL want " + str(want)))
    if not ok:
        FAILURES.append(name)


def check_true(name, cond):
    print("  {:<58} {:>14} {}".format(name, str(bool(cond)),
                                      "ok" if cond else "FAIL"))
    if not cond:
        FAILURES.append(name)


def note(text):
    print("  {:<58} {:>14} --".format(text, ""))


# ---------------------------------------------------------------------------
# [1] the sumset layer
# ---------------------------------------------------------------------------

def test_sumset():
    print("\n[1] the vanishing, as arithmetic on finite sets of integers")

    idx = NL.spectrum_indices(20, 24)
    check("the spectrum is integers", idx, [20, 21, 22, 23, 24])
    check("its sumset runs from twice the least", NL.sumset_range(idx),
          (40, 48))
    check_true("and does not contain zero", not NL.sumset_has_zero(idx))

    # the general statements, over many spectra rather than one
    check_true("no positive spectrum has zero in its sumset",
               all(not NL.sumset_has_zero(NL.spectrum_indices(j, j + 5))
                   for j in range(1, 60)))
    check_true("every symmetric pair does",
               all(NL.sumset_has_zero([w, -w]) for w in range(1, 60)))
    check_true("and a mixed spectrum does exactly when it is symmetric",
               NL.sumset_has_zero([3, -3, 7])
               and not NL.sumset_has_zero([3, 7, 11]))

    # the bound is sharp: the least element of the sumset is twice the least
    check_true("the least frequency of the square is twice the least of the "
               "packet",
               all(min(NL.sumset(NL.spectrum_indices(j, j + 5)))
                   == 2 * j for j in range(1, 30)))

    # a positive spectrum cannot be rescued into vanishing by adding to it
    check_true("adding a negative frequency breaks it",
               NL.sumset_has_zero(NL.spectrum_indices(3, 8) + [-5]))

    # and the requirement really is jmin >= 1
    try:
        NL.spectrum_indices(0, 5)
        check_true("jmin = 0 is rejected", False)
    except ValueError:
        check_true("jmin = 0 is rejected", True)


# ---------------------------------------------------------------------------
# [2] the packet the integers are about
# ---------------------------------------------------------------------------

def test_packet():
    print("\n[2] the integers describe the object actually built")

    for jmin in (1, 5, 20, 60):
        u, dchi = NL.exact_packet(jmin, jmin + 20, centre=jmin + 10.0)
        I1, I2 = NL.functionals(dchi)
        idx = NL.spectrum_indices(jmin, jmin + 20)
        check_true("jmin=%d: measured bins equal the computed sumset" % jmin,
                   NL.occupied_bins(dchi) == NL.sumset(idx))
        check_true("jmin=%d: |I2|/I1 is roundoff" % jmin,
                   abs(I2) / I1 < 1e-12)

    # the point of the exact spectrum: no degradation as jmin falls, because
    # nothing is being truncated
    ratios = []
    for jmin in (1, 60):
        _, dchi = NL.exact_packet(jmin, jmin + 20, centre=jmin + 10.0)
        I1, I2 = NL.functionals(dchi)
        ratios.append(abs(I2) / I1)
    check_true("and jmin = 1 is as clean as jmin = 60",
               max(ratios) < 1e-12 and max(ratios) / min(ratios) < 100)

    # I1 is not itself small, so the ratio means something
    _, dchi = NL.exact_packet(20, 40)
    I1, _ = NL.functionals(dchi)
    check_true("I1 is not small, so the ratio is not vacuous", I1 > 1.0)


# ---------------------------------------------------------------------------
# [3] the detector and the gap
# ---------------------------------------------------------------------------

def test_detector():
    print("\n[3] the graded detector, and the one-mode energy gap")

    check("no admixture gives zero", NL.detector_ratio(0.0), 0.0)
    check("full admixture gives one", NL.detector_ratio(1.0), 1.0)
    check_true("and between them it is tanh(2 artanh eps)",
               all(abs(NL.detector_ratio(e) - np.tanh(2 * np.arctanh(e)))
                   < 1e-15 for e in (0.1, 0.25, 0.5, 0.75, 0.9)))
    check_true("never exceeding one",
               all(NL.detector_ratio(e) <= 1.0 + 1e-15
                   for e in np.linspace(0, 1, 51)))

    check_true("the energy gap is nu (s - t)^2 exactly",
               all(NL.energy_gap(nu, s, t) == nu * (s - t) ** 2
                   for nu in range(-4, 5)
                   for s in range(-4, 5) for t in range(-4, 5)))
    check_true("non-negative for every physical state",
               all(NL.energy_gap(nu, s, t) >= 0 for nu in range(0, 5)
                   for s in range(-4, 5) for t in range(-4, 5)))
    check_true("and zero only when there is no squeeze",
               all((NL.energy_gap(nu, s, t) == 0) == (s == t)
                   for nu in range(1, 5)
                   for s in range(-4, 5) for t in range(-4, 5)))


# ---------------------------------------------------------------------------
# [4] every fact, as Python computes it
# ---------------------------------------------------------------------------

def test_python_facts():
    print("\n[4] every emitted fact, evaluated here")

    verdicts = NL.check_python()
    check("facts emitted", len(verdicts), len(NL.FACTS))
    for name in sorted(verdicts):
        check_true("computed: %s" % name, verdicts[name] is True)


# ---------------------------------------------------------------------------
# [5] the Lean file
# ---------------------------------------------------------------------------

def test_lean():
    print("\n[5] the emitted Lean, and the kernel if there is one")

    src = NL.lean_source()
    check_true("the file opens a namespace and closes it",
               "namespace Nariai" in src and "end Nariai" in src)
    check("a theorem per fact",
          sum(1 for f in NL.FACTS if ("theorem %s" % f.name) in src),
          len(NL.FACTS))
    check("an axiom report per fact",
          src.count("#print axioms Nariai."), len(NL.FACTS))
    check_true("no Mathlib import", "import Mathlib" not in src)
    check_true("and no sorry in the emitted text",
               "sorry" not in src.replace("sorryAx", ""))

    r = NL.check_lean()
    if not r["available"]:
        note("no lean toolchain: emitted but not checked (see module docs)")
        check_true("and the module says so rather than claiming a check",
                   r["ok"] is None)
        return

    check_true("the kernel accepts the file", r["ok"] is True)
    check("no compile errors", r["errors"], [])
    check("every theorem reported its axioms",
          sorted(r["axioms"]), sorted(f.name for f in NL.FACTS))
    check("nothing rests on sorryAx or Classical.choice", r["unsound"], [])

    # the finite ones should need no axioms at all
    for name in ("modular_weight_balances", "packet_avoids_zero",
                 "packet_positive", "packet_sumset_lower_bound"):
        check("%s needs no axioms" % name, r["axioms"].get(name), [])


# ---------------------------------------------------------------------------
# [6] the new grade
# ---------------------------------------------------------------------------

def test_status():
    print("\n[6] the grade: computed, and machine-checked or not")

    st = NL.status()
    check_true("every fact computes", st["all_computed"])
    check("one entry per fact", len(st["facts"]), len(NL.FACTS))
    check_true("each entry carries its note",
               all(v["note"] for v in st["facts"].values()))

    if st["toolchain"]:
        check_true("and every fact is machine-checked",
                   st["all_machine_checked"])
        check_true("with the axiom list recorded per fact",
                   all(v["axioms"] is not None
                       for v in st["facts"].values()))
    else:
        note("no toolchain: machine_checked is False throughout, correctly")
        check_true("nothing claims to be checked",
                   not st["all_machine_checked"])

    # the grade is orthogonal to the ledger: these are exact quantities that
    # have gained confidence, not quantities of a new kind
    check_true("machine-checked is reported separately from the ledger",
               set(st) == {"facts", "toolchain", "all_computed",
                           "all_machine_checked"})


def main():
    t0 = time.time()
    print("=" * 72)
    print("test_nariai_lean: the finite algebra, proved rather than measured")
    print("=" * 72)

    test_sumset()
    test_packet()
    test_detector()
    test_python_facts()
    test_lean()
    test_status()

    print("\n" + "=" * 72)
    if FAILURES:
        print("FAILED (%d): %s" % (len(FAILURES), ", ".join(FAILURES)))
        return 1
    print("test_nariai_lean: all checks passed in %.1fs" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
