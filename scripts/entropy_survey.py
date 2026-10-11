#!/usr/bin/env python3
"""Survey the D4-D0 correction fraction delta = c_2.P / (6 D) over the
favourable CICYs, at the canonical ample class P = J_1 + ... + J_h.

Usage: python3 scripts/entropy_survey.py [out.json]
Takes about 30 minutes for the 4874 favourable entries (20 s cap each; the
published run completed 4871 and skipped 3). Also records whether the c_2
computed here equals the published c_2 in data/cicylist.json.
"""
import json
import os
import signal
import sys
from fractions import Fraction

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pyCICY import entropy as En
from pyCICY.centralcharge import CY3


class _Timeout(BaseException):
    pass


def _alarm(*_):
    raise _Timeout()


def main(out="entropy_survey.json"):
    entries = json.load(open(os.path.join(ROOT, "data", "cicylist.json")))["entries"]
    signal.signal(signal.SIGALRM, _alarm)
    rows, skipped = [], 0
    for e in entries:
        if e["h11"] != len(e["conf"]):          # not favourable
            continue
        signal.alarm(20)
        try:
            X = CY3.from_cicy(e["conf"])
            r = En.d4d0(X, [1] * X.h, 1)
            rows.append({"num": e["num"], "h11": e["h11"], "D": str(r["D"]),
                         "c2P": str(r["c2P"]), "delta": float(r["delta"]),
                         "c2_matches_published":
                             [Fraction(x) for x in e["c2"]] == X.c2})
        except (_Timeout, Exception):
            skipped += 1
        finally:
            signal.alarm(0)
    json.dump(rows, open(out, "w"))
    print("%d done, %d skipped, c2 matches published: %d"
          % (len(rows), skipped, sum(r["c2_matches_published"] for r in rows)))


if __name__ == "__main__":
    main(*sys.argv[1:2])
