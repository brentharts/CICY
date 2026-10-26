r"""
pyCICY.cusp_hofstadter -- electrons on the cusp fibre's substrate, and a
magnetic field as a line bundle on the cusp torus (Part X).

Part VII showed that the cusp fibre W of Alpoge's six-sphere is the
[3.4.6.4] Laves lattice rolled up on the torus R^2 / B0 Z^2: per fundamental
domain one hexagon centre C (a dP6 component), three edge midpoints E (the
double curves) and two hexagon vertices V (the triple points), with six C-E
and six E-V bonds. In lattice coordinates (u, v) for the basis E1, E2 the
sites are torsion points of the hexagonal torus: C = 0, the E's are the three
2-torsion points, the V's are (1/3, 1/3) and (2/3, 2/3).

This module puts a tight-binding electron on that graph -- one orbital per
site, hopping 1 per bond -- and asks the questions that the monotile
tight-binding study (quantum-lattice, ``ql_tiling_spectra.py``) asked of the
Hat, Spectre and Turtle tilings, whose vertex graphs are identical and whose
unit-hopping density of states is therefore one curve.

1. **The substrate's bands, exactly, in A2 invariants** (:func:`bloch_charpoly`).
   The Bloch Hamiltonian is 6 x 6 and bipartite (E against C + V), so its
   characteristic polynomial is a cubic in E^2. With x = e^{ik1}, y = e^{ik2}
   it is

       E^6 - (12 + s) E^4 - 3 (s - 6) E^2 - |Delta|^2,

       s = x + 1/x + y + 1/y + x/y + y/x          (the six roots of A2),
       Delta = (1 - x)(1 - y)(1 - x/y) = det B   (the A2 Weyl denominator,
                                                  up to a unit),

   B the 3 x 3 block from the E sites to the C and V sites. Both
   coefficients are Weyl-invariant functions of the A2 root system -- the
   triangulation behind the cusp fan of Part III. So E = 0 is reached exactly
   where Delta = 0: on the three mirror lines k1 = 0, k2 = 0, k1 = k2 of the
   Weyl group acting on the Brillouin torus. These are nodal *lines*, not
   Dirac points; at Gamma, E = 0 is fourfold and the other two levels are
   +-sqrt 18, and 18 is the row sum of B B^T there (degree 4, plus 2 x 5 from
   the two hexagon cliques, plus 2 x 2 from the two triangle cliques). The
   spectrum fills [-3 sqrt 2, 3 sqrt 2].

2. **The Hat graph is a subgraph of the substrate** (stated here; checked in
   ``paper/make_sixsphere_hat_hofstadter.py`` on generated patches). At the
   Hat's proportions every edge of Tile(1, sqrt 3) -- the 14-edge form used
   for the vertex graphs -- is one kite edge: C-E has length sqrt 3 and E-V
   length 1. So the Hat's tight-binding model is the substrate's with the
   bonds inside tiles deleted, and Part VII's boundary fractions are the
   densities of the twelve bond orbits in the Hat graph.

3. **A magnetic field as a line bundle, with no gauge chosen on the tiling**
   (:func:`landau_phase`, :func:`kite_fluxes`). A uniform field with flux phi
   through each kite is a connection on the substrate; on the cusp torus it
   is a line bundle only if the flux per domain, 6 phi, is an integer (its
   Chern number). Restricting the substrate's Peierls phases to a subgraph is
   restriction of a connection, so the Hat inherits its magnetic model from
   the substrate -- the structure-first order of Gomes's argument (Part IX,
   ``HOPF_WITHOUT_PRINCIPAL_BUNDLES.md``): there is no torus for the
   aperiodic tiling, so no principal bundle to start from, and nothing is
   lost.

4. **Periods** (:func:`flux_periods`). A tight-binding spectrum on a planar
   graph depends on the fluxes of its faces mod 1. The substrate's faces are
   kites, so its butterfly has period 1 in phi. The Hat graph's bounded faces
   are Hats (8 kites) and mystics (10 kites), so its butterfly has period
   1/gcd(8, 10) = 1/2: at phi = 1/2 the Hat model is gauge-equivalent to
   zero field, while the substrate carries pi flux through every kite. The
   Spectre is not a union of kites (Part VII), and has no such statement.

5. **Chern numbers two ways** (:func:`gap_cherns`, :func:`kitaev_chern`).
   For phi = p/q the magnetic Bloch bundle over the torus of a q x 1
   supercell has Chern numbers computed by Fukui-Hatsugai-Suzuki, from
   overlaps of eigenvectors -- a projector computation, gauge-free. Each gap
   satisfies the Streda / TKNN Diophantine relation rho = sigma * 6 phi + s
   with integer s (rho the states per domain below the gap). Independently,
   Kitaev's real-space formula on an open finite patch -- which needs no
   torus, no Bloch bundle and no periodicity -- reproduces the same integer
   in the bulk. That real-space route is the one that applies to the Hat.

What is not claimed: anything about the Hat's gaps beyond what the paper
script measures on finite patches (reported as measured); any physical
meaning of the electron model for the six-sphere, which knows nothing of
electrons. The substrate's spectrum and Chern numbers are exact or reach the
same integer by two routes.
"""

from fractions import Fraction as Fr
from math import floor, gcd, pi, sqrt

import numpy as np
from sympy import Matrix as _Matrix, symbols as _symbols, factor as _factor, \
    Poly as _Poly, expand as _expand, eye as _eye, simplify as _simplify

__all__ = [
    "SITES", "SITE_KIND", "BONDS", "KITES", "E1", "E2", "to_xy",
    "bloch", "bloch_charpoly", "nodal_lines", "band_ranges", "landau_phase", "kite_fluxes",
    "flux_periods", "magnetic_bloch", "chern_fhs", "gap_cherns",
    "finite_patch", "finite_hamiltonian", "kitaev_chern", "dos",
    "status_cusp_hofstadter",
]

#: Cartesian images of the lattice basis (the cusp lattice B0 Z^2).
E1 = np.array([3.0, sqrt(3.0)])
E2 = np.array([0.0, 2.0 * sqrt(3.0)])

_H = [(Fr(1, 2), Fr(0)), (Fr(0), Fr(1, 2)), (Fr(-1, 2), Fr(1, 2)),
      (Fr(-1, 2), Fr(0)), (Fr(0), Fr(-1, 2)), (Fr(1, 2), Fr(-1, 2))]
_V = [(Fr(2, 3), Fr(-1, 3)), (Fr(1, 3), Fr(1, 3)), (Fr(-1, 3), Fr(2, 3)),
      (Fr(-2, 3), Fr(1, 3)), (Fr(-1, 3), Fr(-1, 3)), (Fr(1, 3), Fr(-2, 3))]


def _reduce(p):
    R = (floor(p[0]), floor(p[1]))
    return (p[0] - R[0], p[1] - R[1]), R


def _build():
    sites, kind = [], []
    index = {}

    def sid(p, k):
        f, R = _reduce(p)
        if f not in index:
            index[f] = len(sites)
            sites.append(f)
            kind.append(k)
        return index[f], R

    bonds = set()
    kites = []
    c = (Fr(0), Fr(0))
    ic, _ = sid(c, "C")
    for k in range(6):
        m = (c[0] + _H[k][0], c[1] + _H[k][1])
        im_, Rm = sid(m, "E")
        bonds.add((ic, im_, Rm))
        for vv in (_V[k], _V[(k + 1) % 6]):
            iv, Rv = sid(vv, "V")
            bonds.add((im_, iv, (Rv[0] - Rm[0], Rv[1] - Rm[1])))
    # kites: C, E_k, V_{k+1}, E_{k+1} -- as absolute positions, counterclockwise
    for k in range(6):
        kites.append([c, _H[k], _V[(k + 1) % 6], _H[(k + 1) % 6]])
    return sites, kind, sorted(bonds), kites


#: Sites of one fundamental domain in (u, v), their kinds, and the bonds
#: (a, b, R): site a in cell 0 to site b in cell R.
SITES, SITE_KIND, BONDS, KITES = _build()


def to_xy(uv):
    """Cartesian position of a point given in lattice coordinates."""
    uv = np.asarray(uv, dtype=float)
    return uv[..., 0:1] * E1 + uv[..., 1:2] * E2


# ---------------------------------------------------------------------------
# 1. zero field
# ---------------------------------------------------------------------------

def bloch(k1, k2):
    """The 6 x 6 Bloch Hamiltonian at quasi-momentum (k1, k2), hopping 1."""
    n = len(SITES)
    H = np.zeros((n, n), complex)
    for a, b, R in BONDS:
        ph = np.exp(1j * (k1 * R[0] + k2 * R[1]))
        H[a, b] += ph
        H[b, a] += np.conj(ph)
    return H


def bloch_charpoly():
    r"""det(E - H(x, y)) exactly, x = e^{ik1}, y = e^{ik2}, against the A2 form.

    Returns the polynomial, whether it equals
    E^6 - (12 + s) E^4 - 3 (s - 6) E^2 - Delta Delta-bar (the last term is
    det H = -|det B|^2, bipartite), whether
    det B = +-Delta up to a monomial unit, the multiplicities at Gamma, and
    the E-sublattice matrix B B^T at k = 0 with its row sums.
    """
    x, y, E = _symbols("x y E")
    n = len(SITES)
    H = _Matrix.zeros(n, n)
    for a, b, R in BONDS:
        H[a, b] += x ** R[0] * y ** R[1]
        H[b, a] += x ** (-R[0]) * y ** (-R[1])
    p = _expand((E * _eye(n) - H).det())
    s = x + 1 / x + y + 1 / y + x / y + y / x
    Delta = (1 - x) * (1 - y) * (1 - x / y)
    Delta_bar = (1 - 1 / x) * (1 - 1 / y) * (1 - y / x)
    a2 = E ** 6 - (12 + s) * E ** 4 - 3 * (s - 6) * E ** 2 - Delta * Delta_bar
    a2_ok = _simplify(p - a2) == 0
    Eidx = [i for i, k in enumerate(SITE_KIND) if k == "E"]
    Oidx = [i for i, k in enumerate(SITE_KIND) if k != "E"]
    B = H.extract(Eidx, Oidx)
    ratio = _simplify(B.det() / Delta)
    unit = ratio.is_Mul or ratio.is_Pow or ratio.is_Symbol or ratio.is_number
    unit = bool(unit) and len(_Poly((ratio * x ** 3 * y ** 3).together()
                                    .as_numer_denom()[0], x, y).terms()) == 1
    B0 = B.subs({x: 1, y: 1})
    BBt = B0 * B0.T
    at_gamma = _Poly(p.subs({x: 1, y: 1}), E)
    return {"charpoly": p, "a2_form": a2_ok, "detB_over_Delta": ratio,
            "detB_is_unit_times_Delta": unit,
            "at_gamma": _factor(at_gamma.as_expr()),
            "BBt_at_0": BBt,
            "row_sums": [sum(BBt.row(i)) for i in range(BBt.rows)]}


def nodal_lines(nk=60, tol=1e-9):
    """Grid points with a zero eigenvalue, and whether all lie on k1 = 0,
    k2 = 0 or k1 = k2 (and every point of those lines has one)."""
    ks = 2 * pi * np.arange(nk) / nk
    on, off, missing = 0, 0, 0
    for i, a in enumerate(ks):
        for j, b in enumerate(ks):
            z = np.min(np.abs(np.linalg.eigvalsh(bloch(a, b)))) < 1e-7
            wall = i == 0 or j == 0 or i == j
            if z and wall:
                on += 1
            elif z:
                off += 1
            elif wall:
                missing += 1
    return {"on_walls": on, "off_walls": off, "walls_without_zero": missing}


def band_ranges(nk=48):
    """Min and max of each of the six bands on an nk x nk grid."""
    ks = 2 * pi * np.arange(nk) / nk
    E = np.array([np.linalg.eigvalsh(bloch(a, b)) for a in ks for b in ks])
    return [(float(E[:, i].min()), float(E[:, i].max())) for i in range(6)]


# ---------------------------------------------------------------------------
# 3. the field as a connection
# ---------------------------------------------------------------------------

def landau_phase(r1, r2, phi):
    r"""Peierls phase for a hop r1 -> r2 (lattice coordinates), flux phi per kite.

    Landau gauge A = Phi u dv with Phi = 6 phi the flux per domain (the
    domain has (u, v)-area 1): theta = 2 pi Phi (u1 + u2)/2 (v2 - v1).
    """
    Phi = 6 * phi
    return 2 * pi * float(Phi) * (float(r1[0]) + float(r2[0])) / 2 * \
        (float(r2[1]) - float(r1[1]))


def kite_fluxes(phi):
    """The phase around each kite, counterclockwise, divided by 2 pi."""
    out = []
    for q in KITES:
        tot = 0.0
        for i in range(4):
            tot += landau_phase(q[i], q[(i + 1) % 4], phi)
        out.append(tot / (2 * pi))
    return out


def flux_periods():
    """Face sizes in kites, and the flux period 1/gcd for each graph."""
    faces = {"substrate": (1,), "hat": (8, 10)}
    out = {}
    for k, v in faces.items():
        g = 0
        for f in v:
            g = gcd(g, f)
        out[k] = {"faces_in_kites": v, "period": Fr(1, g)}
    out["spectre"] = {"faces_in_kites": None, "period": None}
    return out


def magnetic_bloch(phi, k1, k2):
    r"""Bloch Hamiltonian of the q x 1 supercell for kite flux phi = p/q."""
    phi = Fr(phi)
    q = phi.denominator
    n = len(SITES)
    H = np.zeros((n * q, n * q), complex)
    for m in range(q):
        for a, b, R in BONDS:
            r1 = (SITES[a][0] + m, SITES[a][1])
            r2 = (SITES[b][0] + m + R[0], SITES[b][1] + R[1])
            mb = (m + R[0]) % q
            S1 = (m + R[0] - mb) // q
            th = landau_phase(r1, r2, phi)
            amp = np.exp(1j * (th + k1 * S1 + k2 * R[1]))
            i, j = a + n * m, b + n * mb
            H[i, j] += amp
            H[j, i] += np.conj(amp)
    return H


def chern_fhs(phi, n_occ, nk=24):
    r"""Fukui-Hatsugai-Suzuki Chern number of the lowest n_occ bands."""
    ks = 2 * pi * np.arange(nk) / nk
    U = np.empty((nk, nk), dtype=object)
    for i, a in enumerate(ks):
        for j, b in enumerate(ks):
            _, v = np.linalg.eigh(magnetic_bloch(phi, a, b))
            U[i, j] = v[:, :n_occ]
    # the supercell Hamiltonian is k-periodic (positions are not in the
    # phases), so the frame at 2 pi is the frame at 0: no boundary fix needed

    def link(A, B):
        d = np.linalg.det(A.conj().T @ B)
        return d / abs(d)

    F = 0.0
    for i in range(nk):
        for j in range(nk):
            a, b = U[i, j], U[(i + 1) % nk, j]
            c, d = U[(i + 1) % nk, (j + 1) % nk], U[i, (j + 1) % nk]
            F += np.angle(link(a, b) * link(b, c) * link(c, d) * link(d, a))
    return F / (2 * pi)


def gap_cherns(phi, nk=24, nkgap=24, min_gap=1e-3):
    r"""Every gap of the magnetic spectrum at kite flux phi: its energy window,
    the states per domain below it, its Chern number and the TKNN integer."""
    phi = Fr(phi)
    q = phi.denominator
    nb = 6 * q
    ks = 2 * pi * np.arange(nkgap) / nkgap
    E = np.array([np.linalg.eigvalsh(magnetic_bloch(phi, a, b))
                  for a in ks for b in ks])
    lo, hi = E.min(0), E.max(0)
    out = []
    for n in range(1, nb):
        if lo[n] - hi[n - 1] > min_gap:
            C = chern_fhs(phi, n, nk)
            rho = Fr(n, q)
            Ci = int(round(C))
            s = rho - Ci * 6 * phi
            out.append({"n_below": n, "window": (float(hi[n - 1]), float(lo[n])),
                        "rho": rho, "chern": C, "chern_int": Ci,
                        "tknn_s": s, "tknn_integer": s.denominator == 1})
    return out


# ---------------------------------------------------------------------------
# 5b. real space: open patches and Kitaev's formula
# ---------------------------------------------------------------------------

def finite_patch(radius=6.0):
    """Sites (lattice coordinates, exact) and bonds of an open substrate patch:
    every site within ``radius`` (Cartesian) of the origin, every bond between
    two such sites."""
    R = int(radius // 3) + 2
    pos, idx = [], {}
    bonds = set()
    for i in range(-R, R + 1):
        for j in range(-R, R + 1):
            for a, b, Rb in BONDS:
                p = (SITES[a][0] + i, SITES[a][1] + j)
                q = (SITES[b][0] + i + Rb[0], SITES[b][1] + j + Rb[1])
                if np.hypot(*to_xy(p)) > radius or np.hypot(*to_xy(q)) > radius:
                    continue
                for r in (p, q):
                    if r not in idx:
                        idx[r] = len(pos)
                        pos.append(r)
                bonds.add(tuple(sorted((idx[p], idx[q]))))
    return pos, sorted(bonds)


def finite_hamiltonian(pos, bonds, phi=0):
    """Hopping 1 on every bond with the substrate's Landau phases restricted."""
    n = len(pos)
    H = np.zeros((n, n), complex)
    for i, j in bonds:
        a = np.exp(1j * landau_phase(pos[i], pos[j], phi))
        H[j, i] += a
        H[i, j] += np.conj(a)
    return H


def kitaev_chern(H, pos, E_F, centre=(0.0, 0.0), radius=None):
    r"""Kitaev's real-space Chern number at Fermi energy E_F.

    nu = 12 pi i sum_{j in A, k in B, l in C} (P_jk P_kl P_lj - P_jl P_lk P_kj),
    A, B, C the three 120-degree sectors around ``centre`` within ``radius``,
    counterclockwise. No torus, no Bloch bundle: only the projector P.
    """
    w, v = np.linalg.eigh(H)
    occ = v[:, w < E_F]
    P = occ @ occ.conj().T
    xy = to_xy(np.array([[float(a), float(b)] for a, b in pos]))
    d = xy - np.asarray(centre)
    r = np.hypot(d[:, 0], d[:, 1])
    if radius is None:
        radius = 0.5 * r.max()
    ang = np.mod(np.arctan2(d[:, 1], d[:, 0]), 2 * pi)
    inside = r < radius
    A = np.where(inside & (ang < 2 * pi / 3))[0]
    B = np.where(inside & (ang >= 2 * pi / 3) & (ang < 4 * pi / 3))[0]
    C = np.where(inside & (ang >= 4 * pi / 3))[0]
    t1 = np.trace(P[np.ix_(A, B)] @ P[np.ix_(B, C)] @ P[np.ix_(C, A)])
    t2 = np.trace(P[np.ix_(A, C)] @ P[np.ix_(C, B)] @ P[np.ix_(B, A)])
    # sign: chosen so that it matches the FHS orientation of chern_fhs; the
    # relative signs and magnitudes across gaps are then the check
    return float(np.real(-12j * pi * (t1 - t2)))


def dos(energies, bins=120, rng=None):
    """Normalised histogram of a spectrum."""
    h, e = np.histogram(energies, bins=bins, range=rng, density=True)
    return h, e


def status_cusp_hofstadter():
    checks = []

    def ok(name, cond):
        checks.append((name, bool(cond)))

    ok("6 sites (C, 3E, 2V), 12 bonds per domain",
       len(SITES) == 6 and len(BONDS) == 12
       and sorted(SITE_KIND) == ["C", "E", "E", "E", "V", "V"])
    cp = bloch_charpoly()
    ok("charpoly = E^6 - (12+s)E^4 - 3(s-6)E^2 - |Delta_A2|^2", cp["a2_form"])
    ok("det B = unit x Weyl denominator", cp["detB_is_unit_times_Delta"])
    nl = nodal_lines(30)
    ok("E = 0 exactly on the three mirror lines",
       nl["off_walls"] == 0 and nl["walls_without_zero"] == 0)
    ok("B B^T row sums 18 at k = 0", cp["row_sums"] == [18, 18, 18])
    br = band_ranges(24)
    ok("spectrum fills [-3 sqrt 2, 3 sqrt 2]",
       abs(br[0][0] + sqrt(18)) < 1e-9 and abs(br[5][1] - sqrt(18)) < 1e-9)
    kf = kite_fluxes(Fr(1, 7))
    ok("every kite carries flux phi", all(abs(f - 1 / 7) < 1e-12 for f in kf))
    gc = gap_cherns(Fr(1, 3), nk=18, nkgap=18)
    ok("TKNN integer in every gap at phi = 1/3",
       gc and all(g["tknn_integer"] and abs(g["chern"] - g["chern_int"]) < 1e-6
                  for g in gc))
    for name, c in checks:
        print("  {:<62} {}".format(name, "ok" if c else "FAIL"))
    return all(c for _, c in checks)


if __name__ == "__main__":
    import sys
    sys.exit(0 if status_cusp_hofstadter() else 1)
