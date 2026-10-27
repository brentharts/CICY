#!/usr/bin/env python3
"""
The Hat tiling as a subgraph of the cusp fibre's substrate, in a magnetic field
(Part X of AlpogeSixSphere.md).

pyCICY.cusp_hofstadter computes the substrate exactly: the [3.4.6.4] Laves
graph rolled up on the cusp torus, its Bloch bands (in A2 invariants) and the
Chern numbers of its Hofstadter gaps (FHS on the magnetic Bloch bundle, and
Kitaev's real-space formula on an open patch). This script brings in a genuine
aperiodic Hat tiling (spectre.py from brentharts/spectre, through
make_sixsphere_hat_torus) and asserts, before anything is drawn, that

  * every edge of every tile (Tile(1, sqrt3), 14 edges; the mystic Tile(sqrt3, 1))
    is one bond of the substrate, so the Hat's vertex graph is a subgraph of it;
  * with the substrate's Landau phases restricted to that subgraph, the flux
    through every Hat is 8 phi and through every mystic 10 phi (mod 1);
  * the Hat spectrum at phi + 1/2 equals the spectrum at phi (period 1/2),
    and at phi = 1/2 equals the zero-field spectrum, while the substrate
    patch's does not;

and then measures, reported as measured,

  * the zero-field density of states of the Hat graph against the substrate's
    (exact Bloch bands), the first panel of quantum-lattice's tiling figure
    seen from the six-sphere;
  * Kitaev's real-space Chern number over (phi, E_F) for the substrate patch
    and the Hat patch -- a "Chern butterfly" -- with the FHS integers of the
    substrate's Bloch bundle marked at a few rational fluxes.

    ss_hat_hofstadter   (a) zero-field DOS  (b) substrate Chern butterfly
                        (c) Hat Chern butterfly, phi in [0, 1/2]

    SPECTRE_DIR=../spectre python3 paper/make_sixsphere_hat_hofstadter.py [--quick]
"""

import os
import sys
from fractions import Fraction as Fr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import numpy as np

import make_sixsphere_hat_torus as T
from pyCICY import cusp_hofstadter as C


def hat_graph(level=3, R=30):
    """Vertices (exact lattice coordinates), edges, and tile cycles of a Hat
    patch; asserts that every tile edge is a substrate bond."""
    sp = T.load_spectre()
    lat = T.Lattice(R=R)
    polys, labels, _ = T.hat_patch(sp, lat, level)
    idx, pos = {}, []
    edges, cycles = set(), []

    def vid(p):
        k = lat.key(p)
        if k not in idx:
            uv = lat.Mi @ p
            exact = (Fr(round(uv[0] * 6), 6), Fr(round(uv[1] * 6), 6))
            assert np.allclose([float(exact[0]), float(exact[1])], uv, atol=1e-6)
            idx[k] = len(pos)
            pos.append(exact)
        return idx[k]

    for P, l in zip(polys, labels):
        cyc = [vid(p) for p in P]
        # counterclockwise orientation in the plane
        area = 0.5 * sum(P[i - 1][0] * P[i][1] - P[i][0] * P[i - 1][1]
                         for i in range(len(P)))
        if area < 0:
            cyc = cyc[::-1]
        cycles.append((cyc, l))
        for i in range(len(P)):
            a, b = P[i], P[(i + 1) % len(P)]
            assert tuple(sorted((lat.key(a), lat.key(b)))) in lat.kbonds, \
                "a tile edge is not a substrate bond"
            edges.add(tuple(sorted((vid(a), vid(b)))))
    return pos, sorted(edges), cycles


def face_fluxes(pos, cycles, phi):
    out = []
    for cyc, l in cycles:
        tot = sum(C.landau_phase(pos[cyc[i]], pos[cyc[(i + 1) % len(cyc)]], phi)
                  for i in range(len(cyc)))
        out.append((l, tot / (2 * np.pi)))
    return out


def checks(pos, edges, cycles, sub_pos, sub_bonds):
    deg = np.bincount(np.array(edges).ravel())
    out = {"vertices": len(pos), "edges": len(edges), "tiles": len(cycles),
           "degrees": sorted(set(deg.tolist()))}
    phi = 0.137
    ff = face_fluxes(pos, cycles, phi)
    out["face_flux_ok"] = all(
        abs(((f - (10 if l == "Gamma2" else 8) * phi) + 0.5) % 1 - 0.5) < 1e-9
        for l, f in ff)
    e = lambda P, B, f: np.linalg.eigvalsh(C.finite_hamiltonian(P, B, f))
    out["hat_period_half"] = float(np.abs(e(pos, edges, 0.137)
                                          - e(pos, edges, 0.637)).max())
    out["hat_half_is_zero"] = float(np.abs(e(pos, edges, 0.5)
                                           - e(pos, edges, 0.0)).max())
    out["substrate_half_vs_zero"] = float(np.abs(e(sub_pos, sub_bonds, 0.5)
                                                 - e(sub_pos, sub_bonds, 0.0)).max())
    return out


def chern_map(pos, edges, phis, energies, radius):
    """Kitaev's marker over a grid of (phi, E_F), and the spectra."""
    xy = C.to_xy(np.array([[float(a), float(b)] for a, b in pos]))
    c = xy.mean(0)
    M = np.zeros((len(energies), len(phis)))
    specs = []
    for j, phi in enumerate(phis):
        H = C.finite_hamiltonian(pos, edges, phi)
        w, v = np.linalg.eigh(H)
        specs.append(w)
        d = xy - c
        r = np.hypot(d[:, 0], d[:, 1])
        ang = np.mod(np.arctan2(d[:, 1], d[:, 0]), 2 * np.pi)
        ins = r < radius
        A = np.where(ins & (ang < 2 * np.pi / 3))[0]
        B = np.where(ins & (ang >= 2 * np.pi / 3) & (ang < 4 * np.pi / 3))[0]
        Cc = np.where(ins & (ang >= 4 * np.pi / 3))[0]
        for i, EF in enumerate(energies):
            o = v[:, w < EF]
            PA, PB, PC = o[A], o[B], o[Cc]
            PAB, PBC, PCA = PA @ PB.conj().T, PB @ PC.conj().T, PC @ PA.conj().T
            t1 = np.trace(PAB @ PBC @ PCA)
            t2 = np.trace(PCA.conj().T @ PBC.conj().T @ PAB.conj().T)
            M[i, j] = np.real(-12j * np.pi * (t1 - t2))
    return M, specs


def main(quick=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    level = 3
    pos, edges, cycles = hat_graph(level)
    sub_pos, sub_bonds = C.finite_patch(26.0)
    ck = checks(pos, edges, cycles, sub_pos, sub_bonds)
    print("Hat graph:", ck)
    assert ck["face_flux_ok"]
    assert ck["hat_period_half"] < 1e-9 and ck["hat_half_is_zero"] < 1e-9
    assert ck["substrate_half_vs_zero"] > 0.1

    # (a) zero-field DOS
    E_hat = np.linalg.eigvalsh(C.finite_hamiltonian(pos, edges, 0))
    nk = 120
    ks = 2 * np.pi * np.arange(nk) / nk
    E_sub = np.concatenate([np.linalg.eigvalsh(C.bloch(a, b))
                            for a in ks for b in ks])

    nphi = 13 if quick else 41
    nE = 61 if quick else 141
    phis_sub = np.linspace(0, 1, 2 * nphi - 1)
    phis_hat = np.linspace(0, 0.5, nphi)
    Es_sub = np.linspace(-4.4, 4.4, nE)
    Es_hat = np.linspace(-2.6, 2.6, nE)
    M_sub, sp_sub = chern_map(sub_pos, sub_bonds, phis_sub, Es_sub, 13.0)
    M_hat, sp_hat = chern_map(pos, edges, phis_hat, Es_hat, 13.0)
    print("Hat: max |Kitaev marker| over the grid = %.3f" % np.abs(M_hat).max())

    fhs = []
    for phi in (Fr(1, 3), Fr(1, 4), Fr(1, 6), Fr(2, 3), Fr(3, 4)):
        for g in C.gap_cherns(phi, nk=12, nkgap=12):
            if g["window"][1] - g["window"][0] > 0.15:
                fhs.append((float(phi), sum(g["window"]) / 2, g["chern_int"]))
    print("FHS gaps:", fhs)

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
    ax = axs[0]
    ax.hist(E_sub, bins=120, density=True, histtype="step", color="#2a78d6",
            label="substrate W (Bloch, exact bands)")
    ax.hist(E_hat, bins=120, density=True, histtype="step", color="#d6602a",
            label="Hat graph (level %d, %d sites)" % (level, len(pos)))
    ax.set_xlabel("E")
    ax.set_title("(a) zero field: the Hat is a subgraph of W")
    ax.legend(fontsize=8, loc="upper left")
    for ax, M, ph, Es, ttl, sp in (
            (axs[1], M_sub, phis_sub, Es_sub, "(b) substrate: Kitaev Chern number",
             sp_sub),
            (axs[2], M_hat, phis_hat, Es_hat,
             "(c) Hat: Kitaev marker, period 1/2 (no quantised gap)", sp_hat)):
        im = ax.imshow(np.clip(M, -3, 3), origin="lower", aspect="auto",
                       extent=(ph[0], ph[-1], Es[0], Es[-1]), cmap="RdBu_r",
                       vmin=-3, vmax=3, interpolation="nearest")
        for x, w in zip(ph, sp):
            ax.plot(np.full(len(w), x), w, ",", color="k", alpha=0.25)
        ax.set_xlabel("flux per kite  phi")
        ax.set_ylabel("E_F")
        ax.set_title(ttl, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    for x, y, c in fhs:
        axs[1].text(x, y, str(c), ha="center", va="center", fontsize=8,
                    color="k", weight="bold")
    fig.tight_layout()
    out = os.path.join(HERE, "figures", "ss_hat_hofstadter.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=130)
    print("wrote", out)
    return ck, fhs


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)
