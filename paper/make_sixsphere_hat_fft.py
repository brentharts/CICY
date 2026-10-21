#!/usr/bin/env python3
"""
The Hat on the same footing as Chair44: agreement with *every* lattice translate.

Part VII measured how a Hat patch agrees with its own translates by the bond
pattern 'is a tile boundary', over |i|, |j| <= 5 only, and found 0.796 at best.
Part VIII measured Chair44 differently: a label at every unit cell, and the
agreement with every translate by FFT. This script does for the Hat what Part
VIII does for Chair44.

The unit cells are the kites of the [3.4.6.4] lattice, indexed (i, j, k): the
kite k = 0..5 round the hexagon centre i*E1 + j*E2. Translation by the period
lattice L = B0 Z^2 keeps k and moves (i, j), exactly as translation by 2^M v
moves a Chair44 cell. A kite's label is
    (k, tile type, linear part of the tile's placement, the kite's place in the tile)
so two kites carry the same label exactly when they are the same piece of the
same tile in the same pose. A coarser label drops the place in the tile.

For every translate t = (di, dj) the agreement is
    #{kites x : x and x+t covered, label(x) = label(x+t)} / #{x : x and x+t covered},
computed by FFT on integer indicators, with the rounding error asserted small,
for translates overlapping the patch in at least an eighth of its kites (the
rule of pyCICY.chair44_nested.translate_scan). The chance level, the sum of
squared label frequencies, is reported beside every number: a coarse label has
a high one, and agreement means little until it is subtracted.

    python3 paper/make_sixsphere_hat_fft.py            # steps 3, 4, 5
    python3 paper/make_sixsphere_hat_fft.py 4 6        # chosen steps
"""

import os
import sys
import time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import numpy as np
from matplotlib.path import Path

import make_sixsphere_figures as F
import make_sixsphere_hat_torus as H

S3, A = F.S3, F.A
E1, E2 = F.E1, F.E2
MI = np.linalg.inv(np.array([E1, E2]).T)


def kite_corners(i, j, k):
    c = i * E1 + j * E2
    return [c, c + A * F._u(30 + 60 * k), c + 2 * F._u(60 * k + 60),
            c + A * F._u(90 + 60 * k)]


def kite_centroids(R):
    """centroid of kite (i, j, k), array of shape (2R+1, 2R+1, 6, 2)."""
    r = np.arange(-R, R + 1)
    I, J = np.meshgrid(r, r, indexing="ij")
    C = I[..., None, None] * E1 + J[..., None, None] * E2
    out = np.zeros(I.shape + (6, 2))
    for k in range(6):
        q = [A * F._u(30 + 60 * k), 2 * F._u(60 * k + 60), A * F._u(90 + 60 * k)]
        out[:, :, k, :] = C[:, :, 0, :] + sum(q) / 4
    return out


def placed_tiles(spectre, n):
    """[(T, label, polygon)] of the n-th substitution step, moved onto the lattice
    (tile 0 registered exactly as make_sixsphere_hat_torus.hat_patch does)."""
    tiles = spectre.buildSpectreTiles(n, 1.0, S3, rotation=30)
    placed = []
    tiles["Delta"].forEachTile(lambda T, l: placed.append((T.copy(), l)))
    pts = spectre.get_spectre_points(1.0, S3)
    myst = spectre.get_spectre_points(S3, 1.0)
    shapes = {"Gamma2": myst}
    polys = [(shapes.get(l, pts) @ T[:, :2].T + T[:, 2]) for T, l in placed]
    lat = H.Lattice(R=8)
    offs = [np.zeros(2)] + [A * F._u(30 + 60 * k) for k in range(6)] \
        + [2 * F._u(60 * k) for k in range(6)]
    c0 = None
    P0 = polys[0]
    for v in P0:
        for off in offs:
            if all(lat.key(p - (v - off)) in lat.sites for p in P0):
                c0 = v - off
                break
        if c0 is not None:
            break
    assert c0 is not None, "tile 0 does not sit on the lattice"
    return [(T, l, P - c0) for (T, l), P in zip(placed, polys)], c0


def label_grid(placed, c0, radius):
    """(G_fine, G_coarse, ntiles): integer label of every kite covered by a tile
    whose centre lies within `radius`; -1 where none."""
    keep = [(T, l, P) for T, l, P in placed if np.hypot(*P.mean(0)) < radius]
    R = int(np.ceil(radius / 3.0)) + 6
    cent = kite_centroids(R)
    side = 2 * R + 1
    Gf = -np.ones((side, side, 6), np.int64)
    Gc = -np.ones((side, side, 6), np.int64)
    fine, coarse = {}, {}
    count = Counter()
    for T, l, P in keep:
        Tlin = T[:2, :2]
        Tinv = np.linalg.inv(Tlin)
        pose = tuple(np.round(Tlin.flatten(), 3) + 0.0)
        lo, hi = P.min(0) - 1, P.max(0) + 1
        box = np.array([[lo[0], lo[1]], [lo[0], hi[1]], [hi[0], lo[1]], [hi[0], hi[1]]])
        u = (MI @ box.T).T
        i0, i1 = int(np.floor(u[:, 0].min())) - 1, int(np.ceil(u[:, 0].max())) + 1
        j0, j1 = int(np.floor(u[:, 1].min())) - 1, int(np.ceil(u[:, 1].max())) + 1
        i0, j0 = max(i0, -R), max(j0, -R)
        i1, j1 = min(i1, R), min(j1, R)
        sub = cent[i0 + R:i1 + R + 1, j0 + R:j1 + R + 1]
        flat = sub.reshape(-1, 2)
        inside = Path(P).contains_points(flat).reshape(sub.shape[:3])
        ii, jj, kk = np.nonzero(inside)
        count[l] += len(ii)
        assert len(ii) == (10 if l == "Gamma2" else 8), ("tile not whole kites", l, len(ii))
        for a, b, k in zip(ii, jj, kk):
            local = Tinv @ (sub[a, b, k] - T[:2, 2])
            fk = (k, l, pose, tuple(np.round(local, 3) + 0.0))
            ck = (k, l, pose)
            Gf[i0 + a + R, j0 + b + R, k] = fine.setdefault(fk, len(fine))
            Gc[i0 + a + R, j0 + b + R, k] = coarse.setdefault(ck, len(coarse))
    return Gf, Gc, len(keep), R


def agreement_scan(G, overlap_fraction=8):
    """Agreement with every translate (di, dj), by FFT on integer indicators.
    Returns dict(frac, overlap, chance, n, err, side)."""
    side = G.shape[0]
    pad = 2 * side
    covered = G >= 0
    n = int(covered.sum())
    acc = np.zeros((pad, pad // 2 + 1))
    labels = np.unique(G[covered])
    freq = np.array([(G == L).sum() for L in labels], dtype=np.float64)
    for L in labels:
        ind = (G == L).any(axis=2).astype(np.float64)   # a label sits at one k only
        Fv = np.fft.rfft2(ind, s=(pad, pad))
        acc += Fv.real ** 2 + Fv.imag ** 2
    corr = np.fft.irfft2(acc, s=(pad, pad))
    ov = np.zeros((pad, pad // 2 + 1))
    for k in range(6):
        Fk = np.fft.rfft2(covered[:, :, k].astype(np.float64), s=(pad, pad))
        ov += Fk.real ** 2 + Fk.imag ** 2
    ov = np.fft.irfft2(ov, s=(pad, pad))
    err = max(np.abs(corr - np.rint(corr)).max(), np.abs(ov - np.rint(ov)).max())
    assert err < 1e-3, ("FFT rounding", err)
    corr, ov = np.rint(corr).astype(np.int64), np.rint(ov).astype(np.int64)
    r = np.arange(-side + 1, side)
    X, Y = np.meshgrid(r, r, indexing="ij")
    C, O = corr[X % pad, Y % pad], ov[X % pad, Y % pad]
    ok = (O >= n // overlap_fraction) & ~((X == 0) & (Y == 0))
    frac = np.where(ok, C / np.maximum(O, 1), 0.0)
    chance = float(((freq / n) ** 2).sum())
    frac_all = C / np.maximum(O, 1)
    return dict(frac=frac, frac_all=frac_all, ok=ok, X=X, Y=Y, O=O, chance=chance, n=n,
                err=float(err), side=side)


def lattice_norm(di, dj):
    v = di * E1 + dj * E2
    return float(np.hypot(*v))


def report(name, G, top=6):
    s = agreement_scan(G)
    frac, ok = s["frac"], s["ok"]
    best = np.argsort(-frac, axis=None)[:top]
    out = dict(name=name, cells=s["n"], chance=s["chance"], err=s["err"],
               translates=int(ok.sum()))
    out["thresholds"] = {t: int((ok & (frac >= t)).sum()) for t in (0.5, 0.9, 0.99)}
    out["best"] = []
    for f in best:
        a = np.unravel_index(f, frac.shape)
        out["best"].append(((int(s["X"][a]), int(s["Y"][a])), float(frac[a]),
                            int(s["O"][a]), lattice_norm(int(s["X"][a]), int(s["Y"][a]))))
    # best agreement in the |i|,|j| <= 5 window of Part VII, for comparison
    win = ok & (np.abs(s["X"]) <= 5) & (np.abs(s["Y"]) <= 5)
    out["best_window5"] = float(frac[win].max()) if win.any() else float("nan")
    # best agreement among translates that keep at least 1/f of the patch overlapped
    allfrac = np.where(s["O"] > 0, s["frac_all"], 0.0)
    nz = ~((s["X"] == 0) & (s["Y"] == 0))
    out["best_by_overlap"] = {}
    for f in (2, 4, 8, 16):
        m = nz & (s["O"] >= s["n"] // f)
        if m.any():
            a = np.unravel_index(np.argmax(np.where(m, allfrac, -1)), allfrac.shape)
            out["best_by_overlap"][f] = (float(allfrac[a]), (int(s["X"][a]), int(s["Y"][a])),
                                         int(s["O"][a]), int(m.sum()))
    return out


def main(steps):
    spectre = H.load_spectre()
    for n in steps:
        t0 = time.time()
        placed, c0 = placed_tiles(spectre, n)
        rmax = max(np.hypot(*P.mean(0)) for _, _, P in placed)
        radius = 75.0 if n == 4 else 0.7 * rmax
        Gf, Gc, nt, R = label_grid(placed, c0, radius)
        print("\n=== n = %d: %d tiles built, %d in the disc of radius %.0f (%.1fs) ==="
              % (n, len(placed), nt, radius, time.time() - t0))
        for name, G in (("fine label (k, tile, pose, kite in tile)", Gf),
                        ("coarse label (k, tile, pose)", Gc)):
            r = report(name, G)
            print("  %s" % name)
            print("    kites %d, labels chance level %.4f, translates scanned %d, FFT err %.1e"
                  % (r["cells"], r["chance"], r["translates"], r["err"]))
            print("    agreement >= 0.5 / 0.9 / 0.99: %d / %d / %d"
                  % tuple(r["thresholds"][t] for t in (0.5, 0.9, 0.99)))
            print("    best in |i|,|j|<=5: %.3f" % r["best_window5"])
            for f, (fr, tv, ov, cnt) in sorted(r["best_by_overlap"].items()):
                print("    overlap >= 1/%-2d (%5d translates): best %.3f at %s, overlap %d"
                      % (f, cnt, fr, tv, ov))
            for tv, fr, ov, nm in r["best"]:
                print("    best t=%-10s agreement %.3f  overlap %6d  |t|=%.1f" % (tv, fr, ov, nm))
        print("  (%.1fs)" % (time.time() - t0))


if __name__ == "__main__":
    steps = [int(a) for a in sys.argv[1:]] or [3, 4, 5]
    main(steps)
