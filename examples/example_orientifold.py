#!/usr/bin/env python3
"""
Type IIB orientifolds: involutions of a CICY, and Sen's limit of F-theory.

    python3 examples/orientifold.py
    python3 examples/orientifold.py --conf '[[2,3],[2,3]]'
    python3 examples/orientifold.py --only sen
    python3 examples/orientifold.py --scan

An orientifold is Type IIB quotiented by a holomorphic involution sigma of the
Calabi-Yau. Everything four-dimensional in the closed string sector follows
from how sigma splits the cohomology, and this script computes that split two
ways that share no code, so that agreement means something.

    1  the sign        sigma^* Omega = -Omega gives O3 and O7 planes,
                       +Omega gives O5 and O9, and the dimensions of the fixed
                       locus have to bear that out rather than being told it
    2  two routes      the Lefschetz fixed point theorem against a count of
                       monomials, and the twist by Omega that reconciles them
    3  degeneracies    the involutions that force X to contain a plane, where
                       the configuration matrix stops describing X and both
                       routes have to refuse
    4  the spectrum    four-dimensional N=1 multiplets from the split
    5  Sen's limit     F-theory at weak coupling: a K3 for every base, the D7
                       tadpole closing, and the brane rules reproducing the
                       Kodaira fibre types

Section 5 is where this meets pyCICY.theories.ftheory. A stack of n
D7-branes on an O7-plane gives so(2n) by counting Chan-Paton factors. The same
configuration is a fibre of type I_{n-4}^*, and the Kodaira table says that
carries so(2n). One side is open string bookkeeping and the other is the
vanishing order of a discriminant. Neither knows about the other.
"""

import argparse
import ast
import sys

sys.path.insert(0, __file__.rsplit("/", 2)[0])

from pyCICY import CICY
from pyCICY.theories import ftheory as FT
from pyCICY.theories import orientifold as O


def banner(text):
    print("\n" + text)
    print("-" * len(text))


DEGREE = {"[[4, 5]]": ([4], [5]),
          "[[2, 3], [2, 3]]": ([2, 2], [3, 3]),
          "[[1, 2], [1, 2], [1, 2], [1, 2]]": ([1, 1, 1, 1], [2, 2, 2, 2])}


def _hypersurface_data(conf):
    """dims and multidegree, when the configuration is a hypersurface."""
    key = str([list(r) for r in conf])
    if key in DEGREE:
        return DEGREE[key]
    if len(conf[0]) == 2:
        return [r[0] for r in conf], [r[1] for r in conf]
    return None, None


def the_sign(conf):
    banner("1. The sign on Omega, and the shape of the fixed locus")

    print("""\
A holomorphic involution multiplies the holomorphic three-form by a sign, and
the sign decides which O-planes appear. It is computed here by counting
flipped coordinates and polynomial signs -- nothing geometric. The dimensions
of the fixed components are computed separately, by intersecting the defining
polynomials with coordinate subspaces. That even complex codimension goes with
the minus sign is a theorem, so the two computations have to agree.
""")
    dims = [r[0] for r in conf]
    print("  %-22s %-6s %-8s %-30s %s"
          % ("flips", "Omega", "planes", "fixed components", "consistent"))
    seen = 0
    for pattern in _patterns(dims):
        try:
            inv = O.SignInvolution(conf, pattern)
        except ValueError:
            continue
        comps = inv.fixed_components()
        desc = ", ".join("%s(chi %d)" % (c["oplane"], c["euler"])
                         for c in comps)
        print("  %-22s %+6d %-8s %-30s %s"
              % (pattern, inv.omega_sign(), inv.oplane_type(), desc[:30],
                 inv.consistent()))
        seen += 1
        if seen >= 12:
            print("  ... (truncated)")
            break


def _patterns(dims, limit=40):
    """A few sign patterns: flip the first k coordinates of each factor."""
    import itertools
    ranges = [range(0, d + 1) for d in dims]
    out = []
    for choice in itertools.product(*ranges):
        if sum(choice) == 0:
            continue
        out.append([list(range(k)) for k in choice])
        if len(out) >= limit:
            break
    return out


def two_routes(conf):
    banner("2. Two routes to the equivariant Hodge numbers")

    dims, degree = _hypersurface_data(conf)
    print("""\
The first route is the Lefschetz fixed point theorem,

    chi(Fix sigma) = 2 + 2 (h11_+ - h11_-) - 2 s - 2 (h21_+ - h21_-)

with s the Omega sign. On a favourable CICY a sign flip fixes every ambient
hyperplane class, so h^{1,1} is entirely invariant and the only unknown is
h21_+ - h21_-, which chi(Fix) determines.

The second route counts monomials. The complex structure deformations of a
hypersurface are its monomials modulo the ambient reparametrisations, and the
involution grades both.
""")
    print("  %-22s %-8s %-10s %-14s %-14s %s"
          % ("flips", "Omega", "chi(Fix)", "Lefschetz", "monomials", "agree"))
    for pattern in _patterns([r[0] for r in conf], limit=14):
        try:
            inv = O.SignInvolution(conf, pattern)
        except ValueError:
            continue
        if inv.degeneracies():
            continue
        try:
            a = inv.hodge_split()["h21"]
        except (ValueError, NotImplementedError):
            continue
        b = None
        if dims is not None:
            try:
                b = O.hypersurface_moduli_split(dims, degree, pattern)["h21"]
            except ValueError:
                b = None
        print("  %-22s %+8d %-10d %-14s %-14s %s"
              % (pattern, inv.omega_sign(), inv.fixed_euler(), a,
                 b if b else "-", "yes" if b == a else
                 ("-" if b is None else "NO")))

    if dims is not None and conf == [[4, 5]]:
        m = O.hypersurface_moduli_split([4], [5], [[4]])
        print("""
The reconciliation worth seeing. On the quintic with one flip the monomial
count gives %s, and the fixed point theorem gives %s. They are not the same
numbers in the same order, and neither is wrong.

The monomials grade H^1(X, T_X), the deformations. But
H^{2,1} = H^1(T_X) tensor H^{3,0}, and sigma acts on H^{3,0} by the Omega
sign, which here is -1. So the two gradings are opposite, and the invariant
deformations are the *anti*-invariant part of H^{2,1}. Get that backwards and
both routes stay self-consistent and both are wrong; the fixed point theorem
is what says which way round it goes.""" % (m["deformations"], m["h21"]))


def degeneracies(conf):
    banner("3. When the involution changes the manifold")

    print("""\
Some involutions cannot be imposed on a generic X at all, and the module has
to notice rather than returning a number for a different manifold.

If every defining polynomial vanishes identically on a coordinate subspace,
that subspace lies inside X. When it is a divisor, X contains a linear
subspace of the ambient, whose class is not in the lattice the ambient
generates: h^{1,1} jumps and the configuration matrix's Hodge numbers describe
something else. When it has the dimension of X, the polynomial factorises and
X is not even irreducible.
""")
    print("  %-22s %-12s %s" % ("flips", "verdict", "why"))
    for pattern in _patterns([r[0] for r in conf], limit=20):
        try:
            inv = O.SignInvolution(conf, pattern)
        except ValueError as e:
            print("  %-22s %-12s %s" % (pattern, "not a map",
                                        str(e).split(".")[0][:44]))
            continue
        bad = inv.degeneracies()
        if not bad:
            continue
        print("  %-22s %-12s %s" % (pattern, bad[0]["kind"],
                                    bad[0]["note"][:46]))

    if conf == [[4, 5]]:
        print("""
On the quintic this is concrete. Flipping three coordinates forces every
invariant monomial to vanish on the plane they span, so X contains a plane --
and a quintic containing a plane has h^{1,1} = 2, not 1. Flipping four forces
every invariant monomial to be divisible by the fifth coordinate, so the
quintic factorises.""")


def spectrum(conf):
    banner("4. The four-dimensional spectrum")

    shown = 0
    for pattern in _patterns([r[0] for r in conf], limit=20):
        try:
            inv = O.SignInvolution(conf, pattern)
        except ValueError:
            continue
        if inv.degeneracies() or inv.omega_sign() > 0:
            continue
        o = O.Orientifold(inv)
        print()
        print(o.describe())
        shown += 1
        if shown >= 2:
            break
    if not shown:
        print("  no O3/O7 involution found for this configuration")
    print("""
The open string sector is absent on purpose. The gauge group and the charged
matter are a choice of D7-brane divisor classes subject to the tadpole
condition, not a consequence of the involution, and the module states the
condition rather than solving it. Which configuration F-theory picks out at
weak coupling is the next section.""")


def sen():
    banner("5. Sen's limit")

    print("""\
F-theory on an elliptic threefold over a base B degenerates at weak coupling
to Type IIB on the double cover of B branched over a curve in |-2K_B|, which
is the O7-plane. Since f is a section of -4K and equals -3h^2 to leading
order, the O7 class is forced to be -2K, and everything below follows from
that.
""")
    print("  %-6s %3s %4s %-12s %-8s %-9s %s"
          % ("base", "T", "K^2", "[O7]", "genus", "cover chi", "D7 tadpole"))
    bases = ["P2"] + ["F%d" % n for n in (0, 1, 2, 3, 8, 12)] \
        + ["dP%d" % k for k in (1, 4, 8)]
    for spec in bases:
        s = O.SenLimit(spec)
        d = s.summary()
        print("  %-6s %3d %4d %-12s %-8d %-9d %s"
              % (spec, d["T"], s.base.K2, str(d["o7_class"])[:12],
                 d["o7_genus"], d["double_cover_euler"],
                 "closes" if d["d7_tadpole"] else "FAILS"))

    print("""
Every cover is a K3, and that is derived rather than arranged:

    chi = 2 chi(B) - chi(O7) = 2(3 + T) - (2 - 2(K^2 + 1))
        = 2(3 + T) + 2(9 - T) = 24

with the tensor multiplet count cancelling between the two terms because
K^2 = 9 - T on a rational surface. The genus of the branch curve does depend
on the base -- 10 on P^2, 2 on dP_8 -- but the cover does not.

The D7 tadpole closes for the same structural reason. The brane at
eta^2 = h psi sits in |-8K|, and with its orientifold image that is -16K,
which is 8 [O7]. Both sides are multiples of the same canonical class, so it
holds for every base at once.
""")

    banner("   and the brane rules against Kodaira")
    print("""\
Perturbatively a stack of n D7-branes gives u(n), or so(2n) when it sits on
the O7-plane and its images coincide with it. Non-perturbatively the same
configuration is a degeneration of the elliptic fibration, with a fibre type
read off the vanishing orders of f, g and Delta. The two have to agree.
""")
    print("  %-28s %-10s %-10s %s"
          % ("configuration", "branes say", "fibre", "Kodaira says"))
    for n in range(2, 8):
        r = O.brane_stack(n)
        print("  %-28s %-10s %-10s %-10s %s"
              % ("%d branes, away from the O7" % n, r["algebra"],
                 r["kodaira"], r["kodaira_algebra"],
                 "agree" if r["agree"] else "DISAGREE"))
    for n in range(4, 10):
        r = O.brane_stack(n, on_o7=True)
        print("  %-28s %-10s %-10s %-10s %s"
              % ("%d branes, on the O7" % n, r["algebra"], r["kodaira"],
                 r["kodaira_algebra"], "agree" if r["agree"] else "DISAGREE"))
    print("""
The shift by four is the content. It takes four D7-branes to cancel the charge
of an O7-plane locally, which is why n branes on the O7 give a fibre of type
I_{n-4}^* rather than I_n^*, and why four of them give the smooth I_0^* with
so(8). Nothing in ftheory.kodaira_type knows about branes; it reads a table of
vanishing orders.
""")
    print("  and a third route to the Euler characteristic, for the record:")
    for spec in ["P2", "F0", "F2", "F3", "F12"]:
        m = FT.FTheory6D(spec)
        a, b = m.euler_characteristic(), FT.weierstrass_euler(m.base)
        print("    %-5s spectrum %6d   -60 K^2 %6d   %s"
              % (spec, a, b, "agree" if a == b else
                 "differ, by the resolution of the %s fibre"
                 % m.gauge_group()))


def scan():
    banner("A scan for O3/O7 involutions")

    confs = [[[4, 5]], [[2, 3], [2, 3]], [[1, 2], [1, 2], [1, 2], [1, 2]],
             [[1, 1, 1], [1, 1, 1], [3, 2, 2]], [[2, 2, 1], [3, 1, 3]]]
    print("  %-34s %-8s %-6s %-10s %s"
          % ("configuration", "flips", "Omega", "h21 split", "O-planes"))
    for conf in confs:
        found = 0
        for pattern in _patterns([r[0] for r in conf], limit=30):
            try:
                inv = O.SignInvolution(conf, pattern)
            except ValueError:
                continue
            if inv.degeneracies() or inv.omega_sign() > 0:
                continue
            try:
                h = inv.hodge_split()
            except (ValueError, NotImplementedError):
                continue
            print("  %-34s %-8s %+6d %-10s %s"
                  % (conf if not found else "", str(pattern)[:8],
                     inv.omega_sign(), str(h["h21"]),
                     ", ".join(sorted(h["oplanes"]))))
            found += 1
            if found >= 3:
                break
        if not found:
            print("  %-34s none found" % (conf,))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--conf", default="[[4,5]]",
                   help="configuration matrix to work through")
    p.add_argument("--only", default=None,
                   choices=["sign", "routes", "degenerate", "spectrum", "sen"])
    p.add_argument("--scan", action="store_true",
                   help="scan several configurations for O3/O7 involutions")
    a = p.parse_args()
    conf = [list(r) for r in ast.literal_eval(a.conf)]

    X = CICY(conf)
    print("Working on %s, h^{1,1} = %d, h^{2,1} = %d, chi = %d"
          % (conf, int(X.h[2]), int(X.h[1]), X.euler_characteristic()))

    sections = {"sign": lambda: the_sign(conf),
                "routes": lambda: two_routes(conf),
                "degenerate": lambda: degeneracies(conf),
                "spectrum": lambda: spectrum(conf),
                "sen": sen}
    order = ["sign", "routes", "degenerate", "spectrum", "sen"]
    for key in ([a.only] if a.only else order):
        sections[key]()
    if a.scan:
        scan()
    print()


if __name__ == "__main__":
    main()
