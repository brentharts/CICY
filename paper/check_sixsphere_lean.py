#!/usr/bin/env python3
"""
Run the six-sphere Lean facts through the Lean kernel, on a machine that has Lean 4.

The facts of ``pyCICY.sixsphere_lean`` come in two groups:

  * FACTS       the facts the paper counts as machine-checked when this script passes;
  * CERT_FACTS  the 25 saturation / Smith-form / H_1 = 0 certificates, written without a
                Smith routine in Lean. Until this script has passed on them they are
                Python-verified only, and nothing should call them machine-checked.

For each group this emits the Lean file, runs ``lean`` on it, reads the ``#print axioms``
report for every fact, and prints a verdict. A fact counts only if the kernel ran, the
file compiled, its axiom report was read, and every axiom is on the allow-list
(propext, Quot.sound, Classical.choice). Nothing here claims a check that did not run.

    python3 paper/check_sixsphere_lean.py                  # both groups
    python3 paper/check_sixsphere_lean.py --certs          # the certificates only
    python3 paper/check_sixsphere_lean.py --write out/     # also write the .lean files

Exit status: 0 if the toolchain ran and every fact in the chosen groups is
machine-checked, 1 if some fact failed, 2 if there is no Lean toolchain.
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from pyCICY import sixsphere_lean as L                            # noqa: E402


def run(name, facts, write_dir):
    if write_dir:
        os.makedirs(write_dir, exist_ok=True)
        path = os.path.join(write_dir, "SixSphere%s.lean" % name)
        with open(path, "w") as fh:
            fh.write(L.lean_source(facts))
        print("wrote %s" % path)
    st = L.status(facts=facts)
    print("\n== %s: %d facts" % (name, st["n_facts"]))
    print("   python half all true : %s" % st["all_computed"])
    print("   Lean toolchain found : %s" % st["toolchain"])
    if not st["toolchain"]:
        print("   (no `lean` on the path or in ~/.elan; install Lean 4 with elan and rerun)")
        return 2
    print("   file compiled        : %s" % st["lean_ok"])
    bad = [(k, v) for k, v in st["facts"].items() if not v["machine_checked"]]
    nonstd = [k for k, v in st["facts"].items()
              if v["axioms"] is not None and not v["standard_axioms_only"]]
    print("   machine-checked      : %d of %d" % (st["n_facts"] - len(bad), st["n_facts"]))
    print("   choice-free          : %d of %d" % (st["n_choice_free"], st["n_facts"]))
    if nonstd:
        print("   NON-STANDARD AXIOMS  : %s" % ", ".join(nonstd))
    for k, v in bad:
        print("   NOT CHECKED: %s (python=%s, axioms=%s)" % (k, v["computed"], v["axioms"]))
    for e in st["errors"][:10]:
        print("   lean error: %s" % e)
    return 0 if st["all_machine_checked"] and st["all_computed"] and not nonstd else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--certs", action="store_true", help="only the certificate facts")
    ap.add_argument("--facts", action="store_true", help="only the main facts")
    ap.add_argument("--write", metavar="DIR", help="also write the .lean files into DIR")
    a = ap.parse_args(argv)
    groups = []
    if not a.certs:
        groups.append(("Facts", L.FACTS))
    if not a.facts:
        groups.append(("Certificates", L.CERT_FACTS))
    codes = [run(n, f, a.write) for n, f in groups]
    return 2 if 2 in codes else (1 if 1 in codes else 0)


if __name__ == "__main__":
    sys.exit(main())
