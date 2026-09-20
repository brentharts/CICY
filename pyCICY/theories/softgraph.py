r"""
pyCICY.theories.softgraph -- infrared structure from the geometry of Schwinger
space.

A Feynman integral in Schwinger parametrisation is an integral over one
positive variable per propagator, and everything about its infrared behaviour
that this package can reach is a statement about two polynomials built from
the graph: the Symanzik polynomials U and F. They come from the reduced graph
Laplacian by the matrix-tree theorem,

    U = (prod_e alpha_e) det L,     F = U [ p_v . p_w (L^-1)_{vw} - sum_e m_e^2 alpha_e ],

which makes the whole analysis linear algebra over the rationals. Figueiredo,
Gambuti and Hannesdottir use exactly this to show that the integrand factorises
into hard and soft parts under a soft scaling, and that -- written in the right
variables -- topologically distinct diagrams have the *same* soft integrand and
differ only in where they are integrated.

That last statement is the one worth having in a package like this, because it
is checkable and surprising: the planar and non-planar two-loop ladders are
different graphs with different Laplacians, and after the soft expansion, in
worldline variables, they agree identically.

Three routes to U, two to F
---------------------------
:meth:`Graph.U` is the determinant of the reduced Laplacian.
:meth:`Graph.U_spanning_trees` is the sum over spanning trees of the product of
the *complementary* edges. They share no code: one is linear algebra, the other
is an enumeration over subsets with a connectivity test. The third route is the
literature -- the closed forms for the one-loop vertex, which the tests check
against.

F likewise: :meth:`Graph.F` from the inverse Laplacian, and
:meth:`Graph.F_forests` from the sum over spanning 2-forests. Signature
conventions for F differ across the literature, and rather than assert one, the
sign here is the one under which the one-loop vertex reproduces the published
closed form. The test records that, so the convention is visible rather than
buried.

What is exact and what is not
-----------------------------
Exact: the polynomials, the Laplacian blocks, the tropical function of a ray,
the classification of a ray as logarithmically or power divergent, the
worldline variables and the identity that produces them, the leading term of
any expansion along a ray, and the ordered domains the different topologies
occupy.

Not exact, and refused: that the remainder left after subtracting the expanded
integrands is infrared finite is a theorem, not an arithmetic fact, and
:exc:`~.surface.NotAnalytic` says so. The exponentiated soft anomalous
dimension is a resummed perturbative series and gets
:exc:`~.surface.NeedsIntegration`. Neither is a number this module will
invent.

Reference
---------
Figueiredo, Gambuti and Hannesdottir, *Soft factorisation and exponentiation
from Schwinger-space geometry*, JHEP 05 (2026) 040, arXiv:2506.15603.
Equation numbers below are that paper's.
"""

from itertools import combinations

import sympy as sp

from .surface import SurfaceTheory, NeedsIntegration, NotAnalytic, register

__all__ = ["Graph", "SoftFactorisation", "one_loop_vertex", "planar_ladder",
           "nonplanar_ladder", "one_loop_F", "soft_integrand", "jet_block",
           "worldline_inverse", "scale_by_ray", "leading_in",
           "tropical_function", "ray_divergence", "search_soft_rays"]


# ---------------------------------------------------------------------------
# graphs
# ---------------------------------------------------------------------------

class Graph(object):
    """A Feynman graph, carrying enough structure to build U and F.

    Parameters
    ----------
    edges : list of (label, u, v, mass)
        ``mass`` is a sympy expression, zero for a massless line.
    momenta : dict
        vertex -> symbol for the external momentum entering there. The
        remaining momentum is understood to enter the vertex that is deleted
        to form the reduced Laplacian, so momentum conservation is automatic.
    dots : dict
        ``(p, q) -> expression`` for every dot product of external momenta.
        Symmetric; either order may be given.
    reduce_at : str
        the vertex deleted to form the reduced Laplacian. Conventionally a
        hard vertex, so that the soft blocks stay whole.
    """

    def __init__(self, edges, momenta=None, dots=None, reduce_at=None,
                 name=None):
        self.edges = [(lab, u, v, sp.sympify(m)) for lab, u, v, m in edges]
        self.momenta = dict(momenta or {})
        self.dots = dict(dots or {})
        self.name = name or "graph"
        verts = []
        for _, u, v, _ in self.edges:
            for w in (u, v):
                if w not in verts:
                    verts.append(w)
        self.vertices = verts
        self.reduce_at = reduce_at or verts[0]
        if self.reduce_at not in verts:
            raise ValueError("no such vertex: %r" % (self.reduce_at,))
        self.alpha = {lab: sp.Symbol(lab, positive=True)
                      for lab, _, _, _ in self.edges}

    # -- basic invariants --------------------------------------------------

    @property
    def loops(self):
        """E - V + 1, the number of independent loops."""
        return len(self.edges) - len(self.vertices) + 1

    def _kept(self):
        return [v for v in self.vertices if v != self.reduce_at]

    def dot(self, p, q):
        if (p, q) in self.dots:
            return sp.sympify(self.dots[(p, q)])
        if (q, p) in self.dots:
            return sp.sympify(self.dots[(q, p)])
        raise KeyError("no dot product given for %r . %r" % (p, q))

    # -- the Laplacian -----------------------------------------------------

    def laplacian(self, reduced=True):
        """The graph Laplacian, entries ``sum_e eta_ve eta_we / alpha_e``.

        Reduced by default: the full Laplacian is singular, since its rows sum
        to zero, and deleting one vertex is what makes it invertible.
        """
        verts = self._kept() if reduced else list(self.vertices)
        idx = {v: i for i, v in enumerate(verts)}
        n = len(verts)
        L = sp.zeros(n, n)
        for lab, u, v, _ in self.edges:
            a = 1 / self.alpha[lab]
            if u in idx:
                L[idx[u], idx[u]] += a
            if v in idx:
                L[idx[v], idx[v]] += a
            if u in idx and v in idx:
                L[idx[u], idx[v]] -= a
                L[idx[v], idx[u]] -= a
        return L, verts

    # -- Symanzik polynomials, route one: linear algebra -------------------

    def U(self):
        """U = (prod_e alpha_e) det L. A polynomial with integer coefficients."""
        if getattr(self, "_U", None) is None:
            L, _ = self.laplacian()
            prod = sp.Integer(1)
            for lab in self.alpha:
                prod *= self.alpha[lab]
            self._U = sp.expand(sp.cancel(prod * L.det(method="berkowitz")))
        return self._U

    def F(self):
        """F = U [ p.p (L^-1) - sum_e m^2 alpha_e ].

        Built through the adjugate rather than the inverse: ``U L^-1`` is
        ``(prod_e alpha_e) adj(L)``, which is polynomial, so no rational
        function is ever formed and nothing has to be cancelled afterwards.
        """
        if getattr(self, "_F", None) is None:
            L, verts = self.laplacian()
            prod = sp.Integer(1)
            for lab in self.alpha:
                prod *= self.alpha[lab]
            adj = L.adjugate(method="berkowitz")
            tot = sp.Integer(0)
            for i, v in enumerate(verts):
                for j, w in enumerate(verts):
                    if v in self.momenta and w in self.momenta:
                        tot += (self.dot(self.momenta[v], self.momenta[w])
                                * sp.cancel(prod * adj[i, j]))
            masses = sum(m ** 2 * self.alpha[lab]
                         for lab, _, _, m in self.edges)
            self._F = sp.expand(tot - masses * self.U())
        return self._F

    # -- route two: enumeration over forests -------------------------------

    def _components(self, edge_subset):
        """Connected components of the graph restricted to ``edge_subset``."""
        parent = {v: v for v in self.vertices}

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for lab in edge_subset:
            _, u, v, _ = self._edge(lab)
            ru, rv = find(u), find(v)
            if ru != rv:
                parent[ru] = rv
        out = {}
        for v in self.vertices:
            out.setdefault(find(v), []).append(v)
        return list(out.values())

    def _edge(self, label):
        for e in self.edges:
            if e[0] == label:
                return e
        raise KeyError(label)

    def _forests(self, pieces):
        """Spanning forests with ``pieces`` components, as edge-label tuples."""
        labels = [lab for lab, _, _, _ in self.edges]
        size = len(self.vertices) - pieces
        out = []
        for sub in combinations(labels, size):
            comps = self._components(sub)
            if len(comps) == pieces:                 # spanning and acyclic
                out.append((sub, comps))
        return out

    def U_spanning_trees(self):
        """U as a sum over spanning trees of the product of the other edges.

        The matrix-tree theorem, taken the other way round: no determinant, no
        inverse, just subsets and a union-find. Agreement with :meth:`U` is a
        check on both.
        """
        total = sp.Integer(0)
        for sub, _ in self._forests(1):
            term = sp.Integer(1)
            for lab in self.alpha:
                if lab not in sub:
                    term *= self.alpha[lab]
            total += term
        return sp.expand(total)

    def _component_momentum(self, comp):
        """Coefficients of the external momenta flowing into one component.

        Every vertex named in ``momenta`` contributes its own momentum; the
        deleted vertex carries minus their sum, which is how momentum
        conservation enters without being imposed by hand.
        """
        names = sorted(set(self.momenta.values()))
        coeff = {p: 0 for p in names}
        for v in comp:
            if v in self.momenta:
                coeff[self.momenta[v]] += 1
            if v == self.reduce_at:
                for p in names:
                    coeff[p] -= 1
        return coeff

    def F_forests(self):
        """F as a sum over spanning 2-forests, minus the internal mass term.

        For each 2-forest, the square of the external momentum flowing into
        one of its two components, times the product of the edges left out;
        then ``- (sum_e m_e^2 alpha_e) U``. No determinant and no inverse, so
        agreement with :meth:`F` is a check on both.
        """
        total = sp.Integer(0)
        for sub, comps in self._forests(2):
            coeff = self._component_momentum(comps[0])
            sq = sum(coeff[a] * coeff[b] * self.dot(a, b)
                     for a in coeff for b in coeff)
            term = sp.Integer(1)
            for lab in self.alpha:
                if lab not in sub:
                    term *= self.alpha[lab]
            total += sq * term
        masses = sum(m ** 2 * self.alpha[lab] for lab, _, _, m in self.edges)
        return sp.expand(total - masses * self.U())


# ---------------------------------------------------------------------------
# the graphs the paper works with
# ---------------------------------------------------------------------------

def _kinematics(m, s):
    """Two on-shell massive legs of mass m with invariant s = (p1+p2)^2."""
    return {("p1", "p1"): m ** 2,
            ("p2", "p2"): m ** 2,
            ("p1", "p2"): (s - 2 * m ** 2) / 2}


def one_loop_vertex(m=None, s=None):
    """The one-loop vertex: two massive lines and one soft photon.

    Three edges, one loop. The paper's (2.1): U is the sum of the three
    Schwinger parameters, and F collapses to ``s a11 a21 - m^2 (a11 + a21)^2``
    -- the same F as a bubble, but with a U that remembers the photon.
    """
    m = sp.Symbol("m", positive=True) if m is None else m
    s = sp.Symbol("s") if s is None else s
    edges = [("a11", "h0", "v11", m),
             ("a21", "h0", "v21", m),
             ("g1", "v11", "v21", 0)]
    return Graph(edges, momenta={"v11": "p1", "v21": "p2"},
                 dots=_kinematics(m, s), reduce_at="h0",
                 name="one-loop vertex")


def _beta():
    return sp.symbols("beta11 beta12 beta21 beta22", positive=True)


def planar_ladder(m=None, s=None):
    """The planar two-loop ladder: photon 1 inside photon 2."""
    m = sp.Symbol("m", positive=True) if m is None else m
    s = sp.Symbol("s") if s is None else s
    edges = [("a11", "h0", "v11", m), ("a12", "v11", "v12", m),
             ("a21", "h0", "v21", m), ("a22", "v21", "v22", m),
             ("g1", "v11", "v21", 0), ("g2", "v12", "v22", 0)]
    g = Graph(edges, momenta={"v12": "p1", "v22": "p2"},
              dots=_kinematics(m, s), reduce_at="h0", name="planar ladder")
    b11, b12, b21, b22 = _beta()
    # worldline distances from the hard vertex to each photon attachment
    g.worldline = {"a11": b11, "a12": b12 - b11,
                   "a21": b21, "a22": b22 - b21}
    g.domain = "0 < beta11 < beta12, 0 < beta21 < beta22"
    return g


def nonplanar_ladder(m=None, s=None):
    """The non-planar two-loop ladder: the photons cross.

    The same six edges as the planar ladder, attached the other way round on
    the second jet. A different graph, a different Laplacian, a different U --
    and, after the soft expansion in worldline variables, the same integrand.
    The difference has moved entirely into the domain: the ordering on the
    second jet is reversed, so the two diagrams fill complementary halves of
    the octant and their sum fills it once.
    """
    m = sp.Symbol("m", positive=True) if m is None else m
    s = sp.Symbol("s") if s is None else s
    edges = [("a11", "h0", "v11", m), ("a12", "v11", "v12", m),
             ("a21", "h0", "v21", m), ("a22", "v21", "v22", m),
             ("g1", "v11", "v22", 0), ("g2", "v12", "v21", 0)]
    g = Graph(edges, momenta={"v12": "p1", "v22": "p2"},
              dots=_kinematics(m, s), reduce_at="h0",
              name="non-planar ladder")
    b11, b12, b21, b22 = _beta()
    g.worldline = {"a11": b11, "a12": b12 - b11,
                   "a21": b22, "a22": b21 - b22}
    g.domain = "0 < beta11 < beta12, 0 < beta22 < beta21"
    return g


def one_loop_F(x, y, m=None, s=None):
    """The one-loop F in worldline variables: ``s x y - m^2 (x + y)^2``."""
    m = sp.Symbol("m", positive=True) if m is None else m
    s = sp.Symbol("s") if s is None else s
    return sp.expand(s * x * y - m ** 2 * (x + y) ** 2)


def soft_integrand(graph, ray, worldline=True):
    """The leading U and F along ``ray``, optionally in worldline variables.

    The object to compare between topologies. In Schwinger parameters the
    planar and non-planar ladders give different answers; in worldline
    variables they give the same one, and only the domain differs.
    """
    Ul, du = leading_in(*scale_by_ray(graph.U(), ray, graph.alpha))
    Fl, df = leading_in(*scale_by_ray(graph.F(), ray, graph.alpha))
    if worldline and getattr(graph, "worldline", None):
        sub = {graph.alpha[k]: v for k, v in graph.worldline.items()}
        Ul = sp.expand(Ul.subs(sub, simultaneous=True))
        Fl = sp.expand(Fl.subs(sub, simultaneous=True))
    return {"U": Ul, "U_degree": du, "F": Fl, "F_degree": df,
            "domain": getattr(graph, "domain", None)}


# ---------------------------------------------------------------------------
# worldline variables
# ---------------------------------------------------------------------------

def jet_block(length, alpha=None):
    """The tridiagonal Laplacian block of one massive line with ``length`` edges.

    The jet block of the soft Laplacian: nearest-neighbour couplings along a
    line of propagators stemming from the hard vertex.
    """
    a = ([sp.Symbol("alpha_%d" % (k + 1), positive=True)
          for k in range(length)] if alpha is None else list(alpha))
    J = sp.zeros(length, length)
    for i in range(length):
        J[i, i] += 1 / a[i]
        if i + 1 < length:
            J[i, i] += 1 / a[i + 1]
            J[i, i + 1] -= 1 / a[i + 1]
            J[i + 1, i] -= 1 / a[i + 1]
    return J, a


def worldline_inverse(length, alpha=None):
    """``(J^-1)_{vw} = beta_{min(v,w)}``, with ``beta_k = alpha_1 + ... + alpha_k``.

    Not a change of notation but a matrix identity, and the reason the
    worldline distance from the hard vertex is the natural variable: inverting
    the jet block produces it without being asked.

    Returns the inverse and the list of beta.
    """
    J, a = jet_block(length, alpha)
    beta = [sp.expand(sum(a[:k + 1])) for k in range(length)]
    return sp.simplify(J.inv()), beta


# ---------------------------------------------------------------------------
# rays, scalings and tropical counting
# ---------------------------------------------------------------------------

def scale_by_ray(expr, ray, alpha, t=None):
    """Apply ``alpha_e -> lambda^{-r_e} alpha_e``, written with ``t = 1/lambda``.

    The soft limit is ``lambda -> 0``, so in ``t`` the leading behaviour is the
    *highest* power, which is what :func:`leading_in` extracts.
    """
    t = sp.Symbol("t", positive=True) if t is None else t
    sub = {alpha[k]: t ** ray[k] * alpha[k] for k in ray}
    return sp.expand(expr.subs(sub, simultaneous=True)), t


def leading_in(expr, t):
    """The leading coefficient and degree of ``expr`` as a polynomial in ``t``."""
    p = sp.Poly(sp.expand(expr), t)
    d = p.degree()
    return sp.expand(p.coeff_monomial(t ** d)), d


def tropical_function(graph, ray, dimension=4):
    """The exponent counting the divergence of a ray, exactly.

    The Schwinger integrand is ``prod_e d alpha_e / U^{D/2} exp(i F / U)``.
    Along ``alpha_e -> lambda^{-r_e} alpha_e`` the measure contributes
    ``lambda^{-sum_e r_e}`` and the explicit ``U^{-D/2}`` contributes
    ``lambda^{+(D/2) deg U``, while ``F/U`` is unchanged at leading order
    whenever ``F`` and ``U`` scale with the same degree -- which is what makes
    a ray a *soft* one. So the integrand goes as ``lambda^{-Trop}`` with

        Trop(r) = sum_e r_e - (D/2) deg U,

    zero for a logarithmic divergence, positive for a power divergence.
    ``deg U`` is taken from the actual expansion rather than from a
    tropicalised formula, so the two cannot drift apart.

    Only meaningful when ``deg F == deg U``; :func:`ray_divergence` checks
    that before calling this.
    """
    D = sp.sympify(dimension)
    _, du = leading_in(*scale_by_ray(graph.U(), ray, graph.alpha))
    measure = sum(int(r) for r in ray.values())
    return sp.simplify(measure - D * du / 2)


def ray_divergence(graph, ray, dimension=4):
    """Classify a ray: ``'log'``, ``'power'``, ``'finite'``, or ``'not-soft'``.

    ``'not-soft'`` is returned when ``F`` and ``U`` do not scale together, so
    that ``F/U`` grows along the ray and the exponent oscillates rather than
    the integrand diverging -- the configuration is not a soft one and this
    counting does not apply to it.
    """
    U, F = graph.U(), graph.F()
    _, du = leading_in(*scale_by_ray(U, ray, graph.alpha))
    _, df = leading_in(*scale_by_ray(F, ray, graph.alpha))
    if df != du:
        return "not-soft"
    trop = tropical_function(graph, ray, dimension)
    if trop == 0:
        return "log"
    return "power" if trop > 0 else "finite"


def search_soft_rays(graph, bound=2, dimension=4):
    """Find the divergent soft rays by enumeration, within a stated bound.

    Not a substitute for the normal fan of the Newton polytope: it is a search
    over integer vectors with entries in ``0 .. bound``, and like every
    truncated search in this package it reports its own box rather than
    pretending to be exhaustive.

    Returns ``(rays, box)`` where ``box`` describes what was searched.
    """
    labels = sorted(graph.alpha)
    found = []
    seen = set()

    def rec(i, cur):
        if i == len(labels):
            if all(v == 0 for v in cur):
                return
            g = 0
            for v in cur:
                g = sp.igcd(g, v)
            key = tuple(v // g for v in cur) if g else tuple(cur)
            if key in seen:
                return
            seen.add(key)
            ray = dict(zip(labels, key))
            kind = ray_divergence(graph, ray, dimension)
            if kind in ("log", "power"):
                found.append((ray, kind))
            return
        for v in range(bound + 1):
            rec(i + 1, cur + [v])

    rec(0, [])
    box = "entries 0..%d on %d edges, up to overall scale" % (bound,
                                                              len(labels))
    return found, box


# ---------------------------------------------------------------------------
# the theory
# ---------------------------------------------------------------------------

@register
class SoftFactorisation(SurfaceTheory):
    """Infrared factorisation of a Feynman integrand in Schwinger space.

    The verb this construction implements is :meth:`soft_limit`; the leading
    singularity belongs to a different construction and is declined.
    """

    key = "soft-factorisation"

    def __init__(self, graph=None, name=None):
        SurfaceTheory.__init__(
            self, surface="a Feynman graph in Schwinger parametrisation",
            name=name)
        self.graph = graph or planar_ladder()

    def soft_limit(self, ray=None, graph=None):
        """The leading behaviour of U and F along a soft ray. Exact.

        Returns the leading coefficients and their degrees, from which the
        hard-soft factorisation can be read off directly.
        """
        g = graph or self.graph
        if ray is None:
            raise ValueError("give a ray, or use search_soft_rays to find one")
        Ul, du = leading_in(*scale_by_ray(g.U(), ray, g.alpha))
        Fl, df = leading_in(*scale_by_ray(g.F(), ray, g.alpha))
        return {"U_leading": Ul, "U_degree": du,
                "F_leading": Fl, "F_degree": df,
                "divergence": ray_divergence(g, ray)}

    def leading_singularity(self, **kw):
        raise NotImplementedError(
            "a leading singularity is a maximal residue, computed by counting "
            "coverings of a fatgraph. That is "
            "pyCICY.theories.gluons.GluonLeadingSingularity, not this "
            "construction.")

    def remainder_is_finite(self, **kw):
        """Always raises: this is a theorem, not an arithmetic fact."""
        raise NotAnalytic(
            "that the remainder left after subtracting the ray expansions is "
            "infrared finite is proved in the literature by an argument about "
            "the whole integration region. This module can exhibit the "
            "subtraction, verify that the subtracted integrand vanishes along "
            "each ray, and check examples -- but an example is not the "
            "statement.",
            checkable=["that the modified soft integrand dS_2 - dS_1 x dS_1 "
                       "vanishes at leading order along each contributing ray",
                       "the inclusion-exclusion over rays, including the ray "
                       "added by the blow-up, as exact combinatorics"])

    def soft_anomalous_dimension(self, **kw):
        """Always raises: a resummed perturbative series."""
        raise NeedsIntegration(
            "the soft anomalous dimension is the exponent of a resummed "
            "series, obtained by integrating the factorised soft integrand "
            "order by order. This module supplies the exact integrand and "
            "performs none of the integrations.",
            missing=["integration of the soft integrand over its domain, in "
                     "dimensional regularisation",
                     "an ultraviolet renormalisation scheme, since the "
                     "expanded integrand carries spurious ultraviolet "
                     "divergences that cancel against the infrared ones",
                     "resummation of the resulting series"])

    def exact_content(self):
        return [
            "the Symanzik polynomials from the graph Laplacian, and "
            "independently from spanning trees and spanning 2-forests",
            "the worldline variables, as the identity that inverting a jet "
            "block gives beta_{min(v,w)}",
            "the leading behaviour of U and F along any ray, and the "
            "classification of that ray as logarithmic, power or finite",
            "that the planar and non-planar ladders have the same soft "
            "integrand in worldline variables, and differ only in domain",
        ]

    def declined(self):
        return [
            "that the subtracted remainder is infrared finite, which is a "
            "theorem (see remainder_is_finite)",
            "the soft anomalous dimension, which is a resummed series (see "
            "soft_anomalous_dimension)",
            "leading singularities, which are a different construction",
        ]
