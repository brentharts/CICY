#!/usr/bin/env python3
"""Run every pyCICY test suite and report a combined result.

Each suite is a standalone script that exits non-zero on failure, so they are
run in subprocesses to keep one suite's state from leaking into the next.
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SUITES = ["tests/test_viz.py", "tests/test_pycicy.py",
          "tests/test_toric.py", "tests/test_quantum_curve.py",
          "tests/test_knots.py", "tests/test_chirality.py",
          "tests/test_hyperbolic.py", "tests/test_apolynomial.py", "tests/test_bundles.py",
          "tests/test_hofstadter.py",
          "tests/test_polytope.py",
          "tests/test_flavor.py",
          "tests/test_breaking.py",
          "tests/test_equivariant.py",
          "tests/test_export.py",
          "tests/test_theories.py",
          "tests/test_ftheory.py",
          "tests/test_orientifold.py"]


def main():
    failed = []
    for suite in SUITES:
        print("\n" + "#" * 72)
        print("# " + suite)
        print("#" * 72)
        rc = subprocess.call([sys.executable, os.path.join(HERE, suite)], cwd=HERE)
        if rc != 0:
            failed.append(suite)

    print("\n" + "#" * 72)
    if failed:
        print("SUITES FAILED: " + ", ".join(failed))
        return 1
    print("ALL SUITES PASSED (%d)" % len(SUITES))
    return 0


if __name__ == "__main__":
    sys.exit(main())
