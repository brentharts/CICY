r"""
pyCICY.quantum_curve -- quantized mirror curves and Hofstadter spectra.

The construction
----------------
A local toric Calabi-Yau K_S has a mirror curve

    Sigma :  sum_{(m,n) in P} c_{mn} x^m y^n = 0 ,      x, y in C^* ,

whose Newton polygon P is the toric diagram of S (see :mod:`pyCICY.toric`).
Writing x = e^u, y = e^v and imposing

    [u, v] = i hbar

promotes the defining polynomial to an operator

    O = sum_{(m,n) in P} c_{mn} e^{m u + n v} ,

Weyl-ordered, so that e^{mu+nv} = e^{i pi hbar m n / 2} e^{mu} e^{nv}. This is
a difference operator, and its spectral problem is the object the refined
topological string solves.

Hatsuda, Katsura and Tachikawa observed that for local F_0 this eigenvalue
problem *is* Harper's equation, the tight-binding problem for an electron on
a square lattice in a magnetic field, with

    hbar = 2 pi Phi ,     Phi = magnetic flux per unit cell.

Sugimoto, "Calabi-Yau geometry and electrons on 2d lattices",
arXiv:1701.01561 (Phys. Rev. D 95, 086004), extended this: local B_3, the
three-point blow-up of local P^2, has a hexagonal Newton polygon and gives
electrons on a *triangular* lattice. The general statement is a dictionary,

    lattice points of P   <->  hopping vectors
    coefficients c_{mn}   <->  hopping amplitudes
    hbar / 2 pi           <->  magnetic flux per unit cell,

and this module evaluates the right-hand side.

How the matrix is built
-----------------------
At rational flux Phi = p/q the magnetic translations T_1, T_2, obeying
T_1 T_2 = e^{2 pi i Phi} T_2 T_1, close on a q-dimensional representation.
In Landau gauge, on a basis j = 0, ..., q-1 and at Bloch momentum (k_1, k_2),

    T_2 |j> = e^{i(k_2 + 2 pi Phi j)} |j> ,     T_1 |j> = e^{i k_1} |j+1 mod q> ,

and the operator becomes the q x q matrix

    H(k) = sum_{(m,n)} c_{mn} e^{i pi Phi m n} e^{i(m k_1 + n k_2)} T_1^m T_2^n .

Sweeping p/q and diagonalising produces the Hofstadter butterfly.

Spectral chirality
------------------
The spectrum is symmetric under E -> -E exactly when the lattice is
bipartite. At the level of the polygon that is a condition modulo two, not a
symmetry of P as a set: bipartiteness holds iff some f in (Z/2)^2 is odd on
every lattice point of P other than the origin, so that (-1)^{f(site)}
anticommutes with H. See :func:`pyCICY.toric.bipartite_functional`.

It is worth being careful here, because the tempting shortcut is wrong in
both directions. Local F_0 (square polygon, square lattice) is bipartite and
its spectrum is symmetric. Local B_3 (hexagon, triangular lattice) is
*centrally symmetric as a polygon* and yet not bipartite, because its hops
include (1,0), (0,1) and (1,-1) whose parities cannot all be made odd; its
spectrum is correspondingly asymmetric. Conversely T4 = P(1,1,2) is not
centrally symmetric but is bipartite, and its spectrum is symmetric. So the
spectral shadow of chirality tracks P mod 2 and not the reflection symmetry
of P. :meth:`QuantumCurve.spectral_asymmetry` measures it directly, and the
test suite checks the equivalence across all sixteen reflexive polygons.

Gap labels
----------
Each gap of the spectrum at flux p/q carries an integer Chern number, the
Hall conductance in units of e^2/h, fixed by the Diophantine equation of
Thouless, Kohmoto, Nightingale and den Nijs,

    r = q s_r + p t_r ,      |t_r| <= q/2 ,

with r the number of filled bands. :meth:`QuantumCurve.gap_labels` returns
these together with the measured gap, so the topological label and the
numerical gap it belongs to are reported side by side.
"""

import math

import numpy as np

from . import toric

__all__ = ["QuantumCurve", "from_polygon", "harper", "butterfly", "farey"]


class QuantumCurve(object):
    r"""The Weyl quantization of a curve with prescribed Newton polygon.

    Parameters
    ----------
    points : sequence of (m, n)
        Exponents of the monomials, i.e. the hopping vectors.
    coeffs : sequence, optional
        Coefficients c_{mn}. Defaults to all ones.
    name : str, optional
        Label carried through to :meth:`describe`.
    hermitian : bool
        If True (the default) the Hermitian part of the matrix is taken. For
        a centrally symmetric point set with real equal coefficients this is
        no restriction, the matrix is already Hermitian; for the others it
        selects the self-adjoint tight-binding model with that hopping set.
    """

    def __init__(self, points, coeffs=None, name=None, hermitian=True):
        self.points = [tuple(int(a) for a in p) for p in points]
        if (0, 0) in self.points:
            raise ValueError("the origin is a constant term, not a hop; "
                             "drop it or fold it into the energy")
        if coeffs is None:
            coeffs = [1.0] * len(self.points)
        coeffs = list(coeffs)
        if len(coeffs) != len(self.points):
            raise ValueError("got %d points but %d coefficients"
                             % (len(self.points), len(coeffs)))
        self.coeffs = [complex(c) for c in coeffs]
        self.name = name
        self.hermitian = bool(hermitian)

    # ---------------------------------------------------------- constructors

    @classmethod
    def from_polygon(cls, name_or_verts, coeffs=None, vertices_only=False,
                     **kw):
        """Build the quantized mirror curve of a reflexive polygon.

        By default every lattice point of P other than the origin contributes
        a monomial, which is the mirror curve proper. With
        ``vertices_only=True`` only the vertices do, giving the
        nearest-neighbour tight-binding model.
        """
        if isinstance(name_or_verts, str):
            verts = toric.polygon(name_or_verts)
            label = name_or_verts
        else:
            verts = [tuple(v) for v in name_or_verts]
            label = toric.classify(verts)["name"]
        pts = (toric.convex_hull(verts) if vertices_only
               else toric.hoppings(verts))
        return cls(pts, coeffs=coeffs, name=label, **kw)

    def mirror(self):
        """The curve with every exponent negated, (m,n) -> (-m,-n).

        The Newton-polygon shadow of orientation reversal. A centrally
        symmetric polygon gives back an identical curve.
        """
        return QuantumCurve([(-m, -n) for m, n in self.points],
                            coeffs=self.coeffs,
                            name=None if self.name is None else self.name + "*",
                            hermitian=self.hermitian)

    # ------------------------------------------------------------- the matrix

    @staticmethod
    def hbar(p, q):
        """hbar = 2 pi p / q, the Planck constant at flux p/q."""
        return 2.0 * math.pi * p / q

    def bloch_matrix(self, p, q, k1=0.0, k2=0.0):
        """The q x q magnetic Bloch matrix at flux Phi = p/q."""
        phi = p / q
        j = np.arange(q)
        H = np.zeros((q, q), dtype=complex)
        for (m, n), c in zip(self.points, self.coeffs):
            # Weyl ordering factor, then the Bloch phase
            amp = c * np.exp(1j * np.pi * phi * m * n) \
                    * np.exp(1j * (m * k1 + n * k2))
            H[(j + m) % q, j] += amp * np.exp(2j * np.pi * phi * n * j)
        if self.hermitian:
            H = 0.5 * (H + H.conj().T)
        return H

    # -------------------------------------------------------------- spectrum

    def bands(self, p, q, nk=16):
        """Sorted eigenvalues on an nk x nk momentum grid, shape (nk*nk, q)."""
        k1s = np.linspace(0.0, 2.0 * np.pi / q, nk, endpoint=False)
        k2s = np.linspace(0.0, 2.0 * np.pi, nk, endpoint=False)
        out = np.empty((nk * nk, q))
        i = 0
        for k1 in k1s:
            for k2 in k2s:
                H = self.bloch_matrix(p, q, k1, k2)
                out[i] = np.linalg.eigvalsh(H) if self.hermitian \
                    else np.sort(np.linalg.eigvals(H).real)
                i += 1
        return out

    def spectrum(self, p, q, nk=16):
        """Every eigenvalue found on the grid, flattened and sorted."""
        return np.sort(self.bands(p, q, nk).ravel())

    def band_edges(self, p, q, nk=16):
        """(min, max) of each of the q bands."""
        b = self.bands(p, q, nk)
        return list(zip(b.min(axis=0), b.max(axis=0)))

    def gaps(self, p, q, nk=16, tol=1e-9):
        """Open gaps as (r, lower_edge, upper_edge, width), r = filled bands."""
        edges = self.band_edges(p, q, nk)
        out = []
        for r in range(1, q):
            lo = edges[r - 1][1]
            hi = edges[r][0]
            if hi - lo > tol:
                out.append((r, float(lo), float(hi), float(hi - lo)))
        return out

    def dos(self, p, q, nk=16, bins=200, window=None):
        """Density of states histogram, returned as (bin_centres, counts)."""
        E = self.spectrum(p, q, nk)
        rng = window if window is not None else (E.min(), E.max())
        counts, edges = np.histogram(E, bins=bins, range=rng, density=True)
        return 0.5 * (edges[:-1] + edges[1:]), counts

    # ------------------------------------------------------------ gap labels

    @staticmethod
    def chern_number(r, p, q):
        r"""Solve the TKNN Diophantine equation r = q s + p t for t.

        The representative with |t| <= q/2 is returned; for even q and the
        half-filling ambiguity the positive one is chosen.
        """
        if math.gcd(p, q) != 1:
            raise ValueError("flux p/q must be in lowest terms")
        t = (r * pow(p, -1, q)) % q
        if t > q / 2:
            t -= q
        return int(t)

    def gap_labels(self, p, q, nk=16, tol=1e-9):
        """Every open gap with its Chern number.

        Returns a list of dicts with keys ``filled``, ``chern``, ``lower``,
        ``upper`` and ``width``.
        """
        return [{"filled": r, "chern": self.chern_number(r, p, q),
                 "lower": lo, "upper": hi, "width": w}
                for r, lo, hi, w in self.gaps(p, q, nk, tol)]

    # ------------------------------------------------------------- chirality

    def spectral_asymmetry(self, p=1, q=3, nk=12):
        """Largest deviation of the spectrum from symmetry under E -> -E.

        Zero (to numerical accuracy) for a bipartite lattice, i.e. for a
        centrally symmetric Newton polygon; strictly positive otherwise.
        """
        E = self.spectrum(p, q, nk)
        return float(np.max(np.abs(E + E[::-1])))

    def is_spectrally_chiral(self, p=1, q=3, nk=12, tol=1e-8):
        """True when the spectrum is not symmetric under E -> -E."""
        return self.spectral_asymmetry(p, q, nk) > tol

    def is_centrally_symmetric(self):
        """True if the hopping set is invariant under (m,n) -> (-m,-n).

        Kept distinct from :meth:`is_bipartite` on purpose: it is the latter,
        not this, that controls the E -> -E symmetry of the spectrum.
        """
        s = set(self.points)
        return all((-m, -n) in s for m, n in s)

    def bipartite_functional(self):
        """f in (Z/2)^2 odd on every hop, or ``None`` if there is none."""
        for f in ((1, 0), (0, 1), (1, 1)):
            if all((f[0] * m + f[1] * n) % 2 == 1 for m, n in self.points):
                return f
        return None

    def is_bipartite(self):
        """True when the lattice is bipartite, hence the spectrum symmetric."""
        return self.bipartite_functional() is not None

    # ---------------------------------------------------------- the butterfly

    def butterfly(self, qmax=25, nk=6, qmin=2):
        """Hofstadter butterfly as parallel arrays (flux, energy)."""
        fl, en = [], []
        for p, q in farey(qmax, qmin=qmin):
            E = self.spectrum(p, q, nk)
            fl.append(np.full(E.shape, p / q))
            en.append(E)
        return np.concatenate(fl), np.concatenate(en)

    # ------------------------------------------------------------------ misc

    def describe(self, p=1, q=3, nk=12):
        tag = self.name or "curve"
        E = self.spectrum(p, q, nk)
        return ("{}  {} hops  Phi={}/{}  E in [{:+.4f}, {:+.4f}]  "
                "asym={:.2e}".format(tag, len(self.points), p, q,
                                     E.min(), E.max(),
                                     self.spectral_asymmetry(p, q, nk)))

    def __repr__(self):
        return "QuantumCurve({!r}, name={!r})".format(self.points, self.name)


# --------------------------------------------------------------- module level

def farey(qmax, qmin=2):
    """Coprime pairs (p, q) with qmin <= q <= qmax and 0 < p < q."""
    return [(p, q) for q in range(qmin, qmax + 1)
            for p in range(1, q) if math.gcd(p, q) == 1]


def from_polygon(name_or_verts, **kw):
    """Convenience wrapper for :meth:`QuantumCurve.from_polygon`."""
    return QuantumCurve.from_polygon(name_or_verts, **kw)


def harper():
    """The Harper / almost-Mathieu operator, i.e. the local F_0 curve."""
    return QuantumCurve.from_polygon("F0")


def butterfly(name_or_verts, qmax=25, nk=6, **kw):
    """Butterfly of a named polygon in one call."""
    return from_polygon(name_or_verts, **kw).butterfly(qmax=qmax, nk=nk)
