# What Curves on Surfaces Determine, and What Only Converges

## Exact methods for gluon leading singularities, soft factorisation, and unitarity cuts

### A Python implementation of three results of Figueiredo and collaborators, with a ledger of where exactness ends

---

## Summary

This document records what the `pyEXACT` package computes from three papers,
how each quantity is checked, and — with equal care — which of their results
the package refuses to produce.

The three papers share a formalism and a habit: they replace an integral by a
count. Leading singularities of gluon amplitudes become coverings of a
fatgraph; the infrared structure of a Feynman integral becomes block structure
in a graph Laplacian; unitarity cuts of a string integrand become residues of
a rational function in positive coordinates. None of the three needs a metric,
a regulator, a Monte Carlo, or a floating-point number, which is why they fit a
package built around exact integer and rational arithmetic.

Three modules implement them: `theories.gluons`, `theories.softgraph` and
`theories.contours`, sharing a base class `theories.surface.SurfaceTheory` and
the machinery in `theories.surfaceology`. Together they are 35 suites' worth of
tests away from being a demonstration: every number that can be reached two
ways is reached two ways, and the places where it cannot are named.

Four results are worth stating up front, because they are the ones where an
independent implementation earns its keep rather than restating a paper:

1. **An integer sequence arrives unbidden.** The n-gon leading singularity's
   coefficients, computed here from a covering count and a closed-curve rule,
   satisfy the Lucas recurrence `L_n = x L_{n-1} − L_{n-2}` — a two-term
   recurrence nothing in the derivation put in. This reproduces the papers'
   observation, which is offered there without explanation.
2. **The divergent rays are found, not told.** A search over integer vectors in
   a stated box returns exactly one soft ray for the one-loop vertex and
   exactly two for the planar ladder, matching the published table, and
   correctly reports that the non-planar ladder has only one — which is the
   reason it contributes a single pole.
3. **A contradiction reproduces, and nearly didn't.** The `D̂ₙ` stringy
   integral satisfies unitarity at six mass levels and fails at the seventh.
   Getting that right required the mirror levels; without them the same system
   looks satisfiable and yields a wrong space-time dimension. The near miss is
   kept in the test suite.
4. **U remembers the blob; the action forgets it.** Extending past ladder
   diagrams to soft webs containing unscaled subgraphs, the pinching argument
   turns out to be two claims rather than one, and only one of them is true of
   both objects.

---

## The ledger, and its fourth entry

Everything in this package is sorted into categories that say what kind of
thing a number is, and refusing is a first-class outcome rather than a failure.
Before the amplitude work there were three:

| category | exception | meaning |
| --- | --- | --- |
| exact | — | integer or rational arithmetic, computed |
| needs a metric | `NeedsMetric` | exists, requires the Ricci-flat metric, which no exact method supplies |
| does not exist | `NoSuchTheory`, `NoChiralMatter`, `TypeIIIFactor` | forbidden or undefined, not merely uncomputed |

Amplitudes forced a fourth, `NotAnalytic`:

> a statement that is **true by argument, not by arithmetic**.

That an integration contour renders an integral finite everywhere in kinematic
space is a theorem. The package can build the contour, bound the cutoff that
keeps it off the branch cuts, count its pieces, and evaluate an example — and
an example is not the statement. Returning one and calling it the theorem would
be the same mistake as returning a plausible float for a physical Yukawa
coupling.

`NeedsIntegration` was added alongside it and is deliberately *not* a fifth
category: it is the amplitude-side `NeedsMetric`. An exact integrand is in
hand, and what is missing is a named integration — loop, phase-space, or a
resummation.

Every refusal carries its own specification. `NotAnalytic` carries
`checkable`, the list of things the package *can* still do about the
statement; `NeedsIntegration` carries `missing`, which is the list of
ingredients that would close the gap.

---

## Why amplitudes needed their own base class

The `theories/` subpackage was built around one question: what
four-dimensional physics comes out of this geometry. Its base class has
`gauge_group`, `spectrum` and `holomorphic_yukawa`.

None of those three means anything for a leading singularity. The entropic
gravity module had already stretched the interface by setting `X = None` and
keeping only the ledger, but three more classes for which the interface's own
verbs are meaningless would be evidence that the interface is wrong, not that
the physics is unusual.

So amplitudes get a sibling rather than a subclass: `SurfaceTheory`, with
verbs `leading_singularity`, `cuts` and `soft_limit`, sharing with `Theory`
exactly what is genuinely common — the registry, so one lookup still reports
what any construction in the package claims, and the exception hierarchy. The
shared machinery underneath is graphs, curves on surfaces and tropical fans;
not intersection numbers and cohomology. That is the honest reason for the
split, and it is why the project is no longer called `pyCICY-X`.

---

## The machinery: curves on a disk

`theories.surfaceology` carries what the amplitude modules stand on: the
chords of an n-gon, their crossing numbers, the F-polynomials, and the
u-variables of the positive parametrisation.

The construction is a closed form, and a closed form is not self-checking: a
wrong index convention gives wrong u-variables that still look like
u-variables. So the check is the **u-equations**,

```
u_X + prod_{X'} u_{X'}^{#(X,X')} = 1,
```

which share no code with the construction. One route knows F-polynomials and
nothing about which chords of a polygon interleave; the other knows the
interleaving and nothing about F-polynomials. Every residual vanishes
identically as a rational function — not at sampled points — at n = 4, 5, 6.

That check did real work. The literature states the F-polynomial boundary case
for a different index range than the one needed here, and rather than guess, the
conventions adopted are the ones under which the u-equations hold: `F_{i,j}` is
1 when its sum is empty, and `y_{1,k}` is 0 outside the n−3 genuine
coordinates. The test records this, so the choice is visible rather than
buried. With those conventions the Koba–Nielsen exponents reproduce the
published five-point integrand factor by factor.

---

## Module one: gluon leading singularities as counting

*Carrôlo and Figueiredo,* How gluon leading singularities discover curves on
surfaces, *arXiv:2512.17019, JHEP 07 (2026) 101.*

A leading singularity is the maximal residue of an amplitude: put every
propagator on shell and read off what is left. For pure gluons it is rational,
and the paper shows it is combinatorial — glue three-point vertices, draw the
Lorentz contractions as curves on the fatgraph, and each monomial of the answer
is one way of covering every edge exactly once with non-overlapping curves.

### The three-point vertex, derived rather than quoted

`scaffolded_three_point()` builds the polarisation vectors from pairs of scalar
momenta, writes the standard Yang–Mills vertex, and rewrites every dot product
through the dual coordinates of the momentum polygon. Two things then have to
happen on their own, and both are checked rather than assumed:

- the gauge parameters `a₁, a₂, a₃` cancel — the polarisations are defined only
  up to a shift along their own momenta, and the answer must not know;
- the squares of the dual coordinates cancel — only differences of dual
  coordinates are physical, so their appearance in intermediate expressions
  must be spurious.

What remains is six monomials, agreeing with the published vertex up to an
overall factor of 4. The test asserts that the factor is *a power of two*
rather than hard-coding it, since the paper explicitly drops factors of two and
restores them as `2^(vertices)`.

### The (2−D) coefficient, derived rather than asserted

A monomial with `n` curves ending on a puncture can be completed around the
loop by extending any non-empty subset of them; a configuration with `Ne`
extensions carries `(−1)^Ne`, and cyclic symmetry makes the count of such
configurations `C(n, Ne)`. The whole content of that argument is an alternating
binomial sum,

```
-sum_{Ne=1}^{n} (-1)^Ne C(n, Ne) = 1     for every n >= 1,
```

so the extensions contribute exactly 1, the closed curve contributes `1 − D`,
and the coefficient is `2 − D`. The bubble's `(2 − D)` is therefore the n = 2
case of one identity, not a separate result, and the test says so by deriving
both from the same function.

### The closed-curve exponent

The question the paper settles: a closed curve homotopic to an internal
boundary — equivalently, one that turns only left — carries exponent `1 − D`;
every other closed curve carries `−D`.

The implementation uses the left-turning criterion rather than the planar
restatement in terms of how many punctures a curve encloses, because the paper
gives a non-planar graph on which the puncture count is not even well defined
and the left-turning criterion still is.

### The Lucas check

In the limit where every loop-dependent variable of the n-gon goes to `Y` and
every purely external one to `X`, the leading singularity organises as

```
LS_n = (2 - D) Y^n - sum_k C_k Y^k X^(n-k),     C_k = [x^(n-2k)] L_n(x).
```

The module computes those `C_k` two ways that share no code — the two-term
recurrence `L_n = x L_{n−1} − L_{n−2}`, and the binomial closed form
`(−1)^k n/(n−k) C(n−k, k)` — and they agree to n = 20. Against the paper's
table they agree term by term for n = 4 through 8, with the triangle carrying
the extra overall factor of two the paper notes.

**An observation worth recording.** Under this module's normalisation the box
agrees in *every* coefficient, including the `C₂` the paper's footnote
singles out as the one exception. The likeliest explanation is a difference in
how the sum is normalised rather than a disagreement about a number — but it is
written down here rather than smoothed over, because a silent agreement where a
paper reports a discrepancy is exactly the sort of thing that should be
visible.

### What this module refuses

- **Unitarity cuts** → `NeedsIntegration`. The leading singularity is the exact
  factor; multiplying by the Lorentz-invariant phase-space measure and
  integrating is the other factor, and this package does not integrate.
- **Fermion loops** → `NotAnalytic`. For pure gluons the sign of a monomial is
  `(−1)^Ne`, which is local data. With a fermion loop the sign depends
  additionally on the total number of intersections of the contraction curves
  inside the loop, and the paper presents the graphical picture while leaving
  the systematic treatment of the cancellations open. An implementation that
  guessed would be indistinguishable from one that knew.

---

## Module two: soft factorisation from graph Laplacians

*Figueiredo, Gambuti and Hannesdottir,* Soft factorisation and exponentiation
from Schwinger-space geometry, *arXiv:2506.15603, JHEP 05 (2026) 040.*

Everything about the infrared behaviour of a Feynman integral that this package
can reach is a statement about two polynomials built from the graph. By the
matrix-tree theorem,

```
U = (prod_e alpha_e) det L,    F = U [ p_v . p_w (L^-1)_{vw} - sum_e m_e^2 alpha_e ],
```

with `L` the reduced graph Laplacian. That makes the whole analysis linear
algebra over the rationals.

### Three routes, two to each polynomial

`U` comes from the determinant of the reduced Laplacian and, independently,
from a sum over spanning trees of the product of the complementary edges: one
is linear algebra, the other an enumeration over subsets with a union-find.
`F` comes from the adjugate — `U·L⁻¹` is `(∏ α)·adj(L)`, which is polynomial,
so no rational function is ever formed — and, independently, from a sum over
spanning 2-forests. All four agree on all three graphs tested.

The sign convention in `F` is **not asserted**. Metric signature and the
direction of the Fourier transform both enter it and the literature differs;
the convention used is the one under which the one-loop vertex reproduces the
published closed form, and the test records that rather than hiding it.

That check caught a real error. The first implementation of the 2-forest route
forgot that the deleted vertex carries the balancing momentum, so the two
routes disagreed. Neither route is privileged; the disagreement is what found
the bug.

### The rays, found by search

A ray is a scaling `alpha_e → lambda^(−r_e) alpha_e`. The integrand goes as
`lambda^(−Trop)` with

```
Trop(r) = sum_e r_e - (D/2) deg U,
```

valid when `F` and `U` scale with the same degree — which is what makes a ray a
*soft* one. `Trop = 0` is a logarithmic divergence, positive is a power
divergence.

Rather than reading the rays off the paper, `search_soft_rays` enumerates
integer vectors in a stated box and classifies each. The results:

| graph | rays found | which |
| --- | --- | --- |
| one-loop vertex | 1 | (1,1,2) |
| planar ladder | 2 | (0,1,0,1,0,2) and (1,1,1,1,2,2) |
| non-planar ladder | 1 | (1,1,1,1,2,2) only |

matching the published table, including the negative result: the non-planar
ladder's partial scaling is classified `not-soft`, which is the reason it
contributes a single pole where the planar ladder contributes two.

Like every truncated search in this package, the function reports its own box
rather than claiming to be exhaustive. It is a search over integer vectors with
entries 0…2, not a normal fan.

### The worldline variables, as an identity

`beta_{i,k} = alpha_{i,1} + … + alpha_{i,k}` is the distance along a worldline
from the hard vertex to the k-th photon attachment. It is not a change of
notation: inverting the tridiagonal jet block gives

```
(J^-1)_{vw} = beta_{min(v,w)}
```

without being asked. Checked symbolically for jets of length 1 through 5.

### The result worth having

In Schwinger parameters the planar and non-planar two-loop ladders have
different `F`. In worldline variables their **soft integrands are identical**,
and the entire difference has moved into the domain:

| | ordering |
| --- | --- |
| planar | `0 < β₁₁ < β₁₂`, `0 < β₂₁ < β₂₂` |
| non-planar | `0 < β₁₁ < β₁₂`, `0 < β₂₂ < β₂₁` |

Same on the first jet, reversed on the second, so the two diagrams fill
complementary halves of the octant and their sum fills it once. That is where
the `1/2` comes from, and at `ℓ` photons the `1/ℓ!` that produces the
exponential.

### Past the ladders: blobs, and what pinching actually says

A soft web can contain an unscaled connected subgraph — a *blob* — which the
ladders never have. The expansion treats it by pinching: at leading order the
blob enters only through which photons attach to it, so contracting it to a
point changes nothing. Formally, the inverse `(B + Γ_B)⁻¹` is dominated by a
rank-one piece divided by `Tr Γ_B = Σ 1/α` over the photons attaching to the
blob, which is independent of the blob's internal parameters.

`pinch(graph, vertices)` performs the contraction as a plain graph operation —
identify the vertices, delete the edges internal to them, keep the rest — so
the comparison is a statement about the expansion rather than about two graphs
drawn to agree.

Implementing it split the claim in two, and the split is the interesting part:

| | tree blob (one edge) | loop blob (triangle) |
| --- | --- | --- |
| leading `U` depends on blob internals | no | **yes** |
| leading action `F/U` depends on them | no | no |
| action equals the pinched graph's | yes | yes |

So the accurate statement is: **U remembers the blob; the action forgets it.**
A blob with a loop of its own contributes its own factor to `U`, exactly as the
factorisation into connected webs predicts — for the triangle,

```
U -> g1 * (g2 + g3) * (ab1 + ab2 + ab3),
```

one factor per connected web, with the blob-containing web contributing
`g2 + g3`, which is the blob pinched to a point. The action meanwhile splits
web by web in worldline variables,

```
V -> F_1(beta11, beta21)/g1 + F_1(beta12, beta22)/(g2 + g3),
```

with no blob parameter anywhere in it. Asserting blob-independence for both
objects would have been wrong, and the test distinguishes the two cases rather
than stating the stronger claim.

### What this module refuses

- **That the subtracted remainder is infrared finite** → `NotAnalytic`. The
  subtraction can be exhibited, the vanishing along each ray checked, and
  examples evaluated; the statement is about the whole integration region.
- **The soft anomalous dimension** → `NeedsIntegration`. A resummed
  perturbative series, with the integration, the ultraviolet scheme and the
  resummation all named.

---

## Module three: cuts, contours, and a failure of unitarity

*Figueiredo and Skowronek,* Cuts and contours, *arXiv:2506.05456,
JHEP 12 (2025) 024.*

The only construction in this package whose headline result is a failure. Many
surface integrals reduce to the same field theory at low energies and differ
only in the ultraviolet; unitarity on massive thresholds is a stringent test of
that freedom, and most candidates die.

### Why this is arithmetic and not analysis

A curve with `q` self-intersections first contributes at order `y^q`. So each
massive threshold is a **finite residue computation** rather than a truncation
of an infinite product — the integrand at level `(n₁, n₂)` only needs windings
up to `min(n₁, n₂)`. That is what makes the whole analysis exact.

### The table, and the mirror

The five published rows of the `D̂` leading singularities come out exactly, as
plain coefficients of `y₁^n₁ y₂^n₂`. The mirror rows are *produced* rather than
assumed: the integrand is symmetric under swapping the two coordinates together
with the two curve exponents, so `dhat_leading_singularity(0,1)` returns the
mirror of `(1,0)` on its own, and the test checks that it does.

### The contradiction

`unitarity_constraints()` runs the matching level by level and returns the
whole sequence, because the shape of the failure is the result:

| level | left side | right side | outcome |
| --- | --- | --- | --- |
| (0,0) | 1 | 1 | already satisfied |
| (1,0) | 1 − X₂₂ | 1 | fixes X₂₂ = 0 |
| (0,1) | 1 − X₁₁ | 1 | fixes X₁₁ = 0 |
| (2,0) | 1 | 1 | already satisfied |
| (0,2) | 1 | 1 | already satisfied |
| (1,1) | 4 | d | fixes d = 4 |
| (2,1) | 2 | 45/16 | **no solution** |

Six levels satisfied and the seventh impossible is a far stronger statement
than "inconsistent", which is why the function returns the steps rather than a
boolean.

**The near miss, kept as a test.** The first implementation matched only the
levels printed in the paper's table — which lists `n₁ ≥ n₂` and states the
mirror rule in prose rather than tabulating it. Without the mirror rows only
one exponent is ever determined, `(1,1)` solves for `X₁₁` in terms of `d`
instead of fixing `d`, and `(2,1)` then solves for `d = 90/29`. Every level
"solves"; the contradiction evaporates into a wrong number. The test suite now
asserts that the mirror-free version *does* look satisfiable and *does* give
the wrong dimension. A contradiction is only as good as the care taken
reproducing it.

**What is derived and what is cited.** The `D̂` column is computed here. The
tree-side column is reference data from the paper: deriving `29d/32 − 13/16`
needs the tree-level coupling extraction at level two with its degeneracies,
which this module does not do. The docstring says so rather than letting the
table look uniformly computed.

### The baby integrals fail differently

Truncated "baby" integrals — keeping only non-self-intersecting curves plus
closed curves with `Δ(q) = Δ₁q + Δ₂q²` — return `−Δ₁ − Δ₂` at levels (1,1),
(2,1) and (1,2) alike, while the tree side gives different values at those
levels. Cheaper to falsify, and the test also checks that the residues are
independent of where the eta-function product is truncated, since each further
factor first contributes at a higher order.

### The contour, as combinatorics

The pieces of the generalised Pochhammer contour are faces of the
associahedron: one sheet per subset of chords, tubes gluing codimension-one
boundaries in pairs, tori gluing codimension-two boundaries in fours.

| n | chords | sheets | tubes | tori |
| --- | --- | --- | --- | --- |
| 5 | 5 | 32 | 80 | 40 |

matching the paper's count, with the top face count coming out as the Catalan
number (5, 14, 42 for n = 5, 6, 7) — which was not put in. The cutoff
`R* > log(n−3)` that keeps the deformation off the branch cuts is an exact
bound, `log 2` at five points.

### What this module refuses

- **That the contour converges everywhere** → `NotAnalytic`. The fourth ledger
  entry doing exactly the work it was added for.
- **Numerical evaluation at large kinematics** → `NotAnalytic`, for a different
  reason: the answer is a fine cancellation between two exponentially large
  contributions, so finite precision is the binding constraint rather than
  method. Choosing the cutoff per cone mitigates it and cannot remove it.
- **The α′ expansion** → `NeedsIntegration`.

---

## The ledger, filled in

| quantity | status |
| --- | --- |
| scaffolded three-gluon vertex | exact, derived |
| extension balance, hence (2−D) at any multiplicity | exact, derived |
| closed-curve exponent at any loop order | exact, derived |
| n-gon coefficients, two independent routes | exact, derived |
| Symanzik polynomials, two routes each | exact, derived |
| divergent rays | exact, found by bounded search |
| worldline variables | exact, a matrix identity |
| planar = non-planar in worldline variables | exact, derived |
| blob pinching (action), blob factor (U) | exact, derived |
| cut residues at any mass level | exact, derived |
| the `D̂` unitarity contradiction | exact, derived |
| Pochhammer piece count, cutoff bound | exact, derived |
| tree-side couplings at level two | **cited**, not derived |
| fermion-loop sign rule | `NotAnalytic` — open in the literature |
| contour convergence | `NotAnalytic` — a theorem |
| large-kinematics evaluation | `NotAnalytic` — a precision problem |
| unitarity cuts, α′ expansion, cross-sections | `NeedsIntegration` |
| soft anomalous dimension | `NeedsIntegration` |

---

## Reproducing all of it

```bash
git clone https://github.com/brentharts/CICY.git && cd CICY
pip install -r requirements.txt

python3 tests/test_surface.py      # u-equations, three-point vertex, Lucas
python3 tests/test_softgraph.py    # Symanzik routes, rays, worldline, blobs
python3 tests/test_contours.py     # the table, the contradiction, the contour
python3 run_tests.py               # all 35 suites
```

```python
from pyCICY import theories as T
from pyCICY.theories import softgraph as SG

T.get("gluon-leading-singularity")().leading_singularity(n=8)
T.get("cuts-and-contours")().unitarity()["first_failure"]     # (2, 1)

SG.SoftFactorisation().pinching(SG.blob_web(blob_edges=3))
# {'blob_in_U': True, 'blob_in_action': False, 'action_agrees': True, ...}
```

Every refusal is reachable too, and carries its own specification:

```python
g = T.get("gluon-leading-singularity")()
try:
    g.fermion_loop()
except Exception as e:
    print(type(e).__name__, e.checkable)
```

---

## No Mathematica was read

Two of the three papers ship companion code as Mathematica notebooks — arXiv
ancillary material for the soft-factorisation paper, supplementary material for
cuts and contours — and the leading-singularity paper ships none. None of it
was consulted; none of it could have been, in the environment this was built
in, and the package's standing rule is no Mathematica anyway.

That is a feature of the exercise rather than a limitation of it. A quantity
computed once is a result. Computed twice, by unrelated routes, it is a test —
and the point of an independent Python implementation of a Mathematica result
is precisely that the two share no code.

---

## What remains open

- **The fermion-loop sign rule.** The one thing all three papers leave
  genuinely open, and the clearest next target: with a fermion loop the sign of
  a monomial depends on the whole intersection pattern inside the loop rather
  than on the extension count alone.
- **Tree-level coupling extraction at level two**, which would let the
  right-hand column of the unitarity table be derived rather than cited, and
  would make the contradiction self-contained.
- **The normal fan**, replacing the bounded ray search with a convex-hull
  computation, so that "these are the rays" stops being "these are the rays in
  this box".
- **Why Lucas.** The paper offers no explanation for the recurrence and neither
  does this implementation. It reproduces the pattern; it does not account for
  it.

---

## References

- Sérgio Carrôlo, Carolina Figueiredo (2025), *How gluon leading singularities
  discover curves on surfaces*, arXiv:2512.17019; JHEP 07 (2026) 101.
- Carolina Figueiredo, Giulio Gambuti, Holmfridur S. Hannesdottir (2026),
  *Soft factorisation and exponentiation from Schwinger-space geometry*,
  arXiv:2506.15603; JHEP 05 (2026) 040,
  https://doi.org/10.1007/JHEP05(2026)040.
- Carolina Figueiredo, Marcos Skowronek (2025), *Cuts and contours*,
  arXiv:2506.05456; JHEP 12 (2025) 024,
  https://doi.org/10.1007/JHEP12(2025)024.
- Nima Arkani-Hamed, Hadleigh Frost, Giulio Salvatori, Pierre-Guy Plamondon,
  Hugh Thomas, *All loop scattering as a counting problem*, arXiv:2309.15913;
  JHEP 08 (2025) 194 — the surfaceology formalism the three papers build on.
- Nima Arkani-Hamed, Qu Cao, Jin Dong, Carolina Figueiredo, Song He,
  *Scalar-scaffolded gluons and the combinatorial origins of Yang-Mills
  theory*, arXiv:2401.00041; JHEP 04 (2025) 078.
