#!/usr/bin/env python3
"""
Conjecture 1 (six mystic fibre classes), tested on whole substitution patches.

The fibre classes of the mystic (Gamma2, the ten-kite Turtle) were computed in
make_sixsphere_mystic_fibres.py on patches of two to four steps, the larger a
radius-cut disc, by comparing mystics pairwise. Proposition 5 shows that is not
needed: a class is exactly a pair (first-edge angle, anchor vertex mod the
period lattice), so the classes of a patch are the distinct pairs, and a patch
of any size can be counted in one pass.

For every Gamma2 tile of the whole n-step patch (no radius cut) this records
    * the angle of the first edge of the placed tile, and whether the tile is a
      reflection (det of the linear part < 0);
    * the anchor v0, the first vertex, reduced modulo Lambda = Z E1 + Z E2, and
      checks that it is an edge-midpoint site, one of (1/2, 0), (0, 1/2),
      (1/2, 1/2), i.e. a double curve D_a, D_c, D_b of the cusp fibre;
and counts the distinct (reflection, angle, anchor) triples. Conjecture 1 says
six for every n >= 3: one per oriented double curve of the dual complex.

    python3 paper/make_sixsphere_mystic_levels.py          # n = 1 .. 5
    python3 paper/make_sixsphere_mystic_levels.py 6        # chosen steps
"""

import os
import sys
import time
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import numpy as np

import make_sixsphere_hat_torus as H
import make_sixsphere_hat_fft as HF

MI = HF.MI
DCURVE = {(0.5, 0.0): "D_a", (0.0, 0.5): "D_c", (0.5, 0.5): "D_b"}


def mystic_classes(spectre, n):
    placed, c0 = HF.placed_tiles(spectre, n)
    types = Counter(l for _, l, _ in placed)
    classes = defaultdict(int)
    refl = 0
    for T, l, P in placed:
        if l != "Gamma2":
            continue
        e = P[1] - P[0]
        ang = round(float(np.degrees(np.arctan2(e[1], e[0]))) % 360.0, 1)
        if ang >= 359.95:
            ang = 0.0
        det = float(np.linalg.det(T[:2, :2]))
        r = det < 0
        refl += r
        u = MI @ P[0]
        f = u - np.floor(u + 1e-9)
        h = np.round(2 * f) / 2.0
        assert np.abs(f - h).max() < 1e-6, ("anchor is not a half-lattice point", f)
        key = (float(h[0] % 1.0), float(h[1] % 1.0))
        assert key in DCURVE, ("anchor is not an edge-midpoint site", key)
        classes[(r, ang, DCURVE[key])] += 1
    return types, classes, refl


SITES = {(0.0, 0.0): "E0", (0.5, 0.0): "D_a", (0.0, 0.5): "D_c", (0.5, 0.5): "D_b",
         (1 / 3, 1 / 3): "P", (2 / 3, 2 / 3): "Q"}


def hat_classes(spectre, n):
    """The same count for the Hat (every tile but Gamma2): (reflected, angle of the first
    edge, class mod Lambda of the first vertex), on the whole n-step patch."""
    placed, c0 = HF.placed_tiles(spectre, n)
    classes = defaultdict(int)
    for T, l, P in placed:
        if l == "Gamma2":
            continue
        e = P[1] - P[0]
        ang = round(float(np.degrees(np.arctan2(e[1], e[0]))) % 360.0, 1)
        ang = 0.0 if ang >= 359.95 else ang
        u = MI @ P[0]
        f = (u - np.floor(u + 1e-9))
        k = [kk for kk in SITES if abs(f[0] - kk[0]) < 1e-6 and abs(f[1] - kk[1]) < 1e-6
             or (abs(f[0] - 1 - kk[0]) < 1e-6 and abs(f[1] - kk[1]) < 1e-6)
             or (abs(f[0] - kk[0]) < 1e-6 and abs(f[1] - 1 - kk[1]) < 1e-6)]
        assert len(k) == 1, ("anchor is not a site class", f)
        classes[(float(np.linalg.det(T[:2, :2])) < 0, ang, SITES[k[0]])] += 1
    return classes


def main(steps):
    spectre = H.load_spectre()
    if os.environ.get("HAT"):
        for n in steps:
            cl = hat_classes(spectre, n)
            print("Hat, n=%d: %d tiles, %d classes" % (n, sum(cl.values()), len(cl)))
            for k, v in sorted(cl.items(), key=lambda kv: (kv[0][1], kv[0][2])):
                print("   %s angle %5.1f  %-3s x%d" % ("refl" if k[0] else "    ", k[1], k[2], v))
        return
    for n in steps:
        t0 = time.time()
        types, classes, refl = mystic_classes(spectre, n)
        nm = types.get("Gamma2", 0)
        print("\nn = %d: %d tiles, %d mystic (%d reflected), %d classes  (%.1fs)"
              % (n, sum(types.values()), nm, refl, len(classes), time.time() - t0))
        for (r, ang, dc), cnt in sorted(classes.items(), key=lambda kv: (kv[0][1], kv[0][0])):
            print("   angle %5.1f  %s%s  x%d" % (ang, dc, "  (reflected)" if r else "", cnt))


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or [1, 2, 3, 4, 5])
