## pyCICY Papers
- Supplementary material: figures for the Extended pyCICY package
    - https://doi.org/10.5281/zenodo.21798923

what follows is the short version.

### The sixteen reflexive polygons

![The sixteen reflexive polygons with their polar duals](paper/figures/fig_polygons.png)

Each panel is the toric diagram of a local Calabi-Yau `K_S`, drawn with its
lattice points (solid) and its polar dual `P*` (dashed). The dual is the
Batyrev mirror. Every title records the twelve theorem,
`#dP + #dP* = 12`, which holds in all sixteen cases however the boundary
points are shared out. The classification is not assumed:
`toric.verify_named()` rederives the list by brute force, and the five smooth
cases -- `P2`, `F0`, `F1`, `dP2` and `B3 = dP3` -- are *detected* by the
criterion that the vertex count equal the boundary count, not tabulated.

### Hofstadter spectra of quantized mirror curves

![Hofstadter butterflies for local F_0 and local B_3](paper/figures/fig_butterflies.png)

Quantizing the mirror curve of a local toric Calabi-Yau turns it into an
electron hopping on a 2d lattice in a magnetic field
([arXiv:1701.01561](https://arxiv.org/abs/1701.01561)): lattice points of the
Newton polygon are hopping vectors, and `hbar/2pi` is the flux per unit cell.
The square polygon of local `F_0` gives the square lattice and the classic
butterfly; the hexagonal polygon of local `B_3 = dP_3` gives the triangular
lattice. Annotations at `Phi = 1/3` are gap Chern numbers from the
Diophantine equation `r = q*s + p*t` with `|t| <= q/2`.

The triangular butterfly is visibly slanted. `E(Phi) = -E(1-Phi)` holds for
all sixteen geometries, but `E(Phi) = E(1-Phi)` only in the bipartite cases --
and bipartiteness is a condition on the polygon *modulo two*, not a reflection
symmetry of it. Only `F_0` and `T4 = P(1,1,2)` are bipartite, so fourteen of
sixteen spectra are asymmetric in `E`. `B_3` is centrally symmetric as a
polygon and still spectrally chiral; `T4` is the other way round.

### Chirality of K15n81556, and the additivity counterexample

![Jones polynomial of K15n81556 against its mirror, and 7_1 # m7_1 as a braid](paper/figures/fig_knots.png)

Upper: the Jones polynomial of the fifteen-crossing census knot `K15n81556`
against that of its mirror, coefficient by coefficient. The two are
reflections about `t^0` and do not coincide, so the knot is chiral. This
reproduces the observation of Wang and Zhang
([arXiv:2507.14265](https://arxiv.org/abs/2507.14265)) that the two diagrams
of `K15n81556` in the Brittenham-Hermiller argument
([arXiv:2506.24088](https://arxiv.org/abs/2506.24088)) are a chiral knot and
its mirror rather than the same knot. The determinant is 39 for both, so
detecting the chirality needs an invariant not symmetric under `t -> 1/t`.

Lower: the connected sum `7_1 # m7_1`, whose unknotting number breaks
additivity, as a braid closure. The connected sum of two braid closures is the
juxtaposition of their words on one more strand than the two together, so this
is the closure of `s1^7 s2^-7` on three strands; the two halves have visibly
opposite handedness.

### One mirror operation, three domains

![Chirality across knots, reflexive polygons and Calabi-Yau threefolds](paper/figures/fig_chirality.png)

Three mirror operations of the same shape: each is an involution that swaps a
pair of integers and preserves their sum or span. Horizontally the combination
the mirror negates, vertically the one it preserves, so mirror partners sit
symmetrically about zero (filled = object, hollow = its mirror) and objects
fixed by their involution lie on the axis. The right-hand panel is the
conventional Hodge plot in disguise, since its horizontal coordinate is
`chi/2`.

Quantized curves are absent deliberately: reflecting their Newton polygon
leaves the spectrum *exactly* unchanged, so no spectral invariant can detect
that involution, and the package reports it as undetermined rather than as
achiral.

### Hyperbolic lattices, and why finite patches fail

![A {8,8} flake in the Poincare disk, and its boundary fraction](paper/figures/fig_hyperbolic.png)

Left: a patch of the `{8,8}` tessellation in the Poincare disk. The marked
points are cell centres, which for the `{4g,4g}` family form the Bravais
lattice of a genus-`g` surface; bonds are drawn as true geodesics, circular
arcs meeting the boundary at right angles, not chords.

Right: the fraction of cells that are not fully coordinated, against flake
depth. It does not tend to zero. Each ring is `p-1` times the last, so the
rim fraction tends to `(p-2)/(p-1)` -- 6/7 for `{8,8}`, 10/11 for `{12,12}`
-- shown dashed. A finite patch is therefore never a good stand-in for the
bulk, which is the quantitative reason hyperbolic band theory needs periodic
boundary conditions and automorphic functions
([PNAS 119 e2116869119](https://doi.org/10.1073/pnas.2116869119)).

### A-polynomials and colored Jones

![A-polynomial Newton polygons and colored Jones of the trefoil](paper/figures/fig_apolynomial.png)

Upper: Newton polygons of the A-polynomials of the figure-eight and the
trefoil, annotated with the coefficient at each lattice point and with the
edge slopes in red. Those slopes are boundary slopes of incompressible
surfaces in the knot complement, and come out at `±4` and `6 = pq`.

Lower: the colored Jones polynomials of the trefoil from the Rosso-Jones
formula, one row per colour, marker shape giving the sign of the coefficient.
The span grows quadratically in `N`. The row `N = 2` is the ordinary Jones
polynomial and agrees coefficient for coefficient with the Kauffman-bracket
computation elsewhere in the package.

The same lattice points are what the quantized-curve machinery consumes as a
hopping set -- the concrete link between the two ends of the package. That
link is about the quantization *rule*, not the geometry: a toric diagram is a
reflexive polygon and an A-polynomial's Newton polygon is not.

### The original CICY figures

The split-web figures are described in the supplement and rebuilt by
`make figures`: `hodge_depth`, `hodge_favourable`, `node_counts`,
`node_validation`, `ch2_check`, `gv_invariants`, `additivity`, `generations`,
`web_growth` and `quintic_surface`. The most important is `node_validation`,
which checks the node count from ambient intersection theory against the count
inferred from the Euler characteristic -- two computations with no shared code
that must land on the same integer for every split, and do.

# pyCICY

A python CICY toolkit, which allows the computation of line bundle cohomologies over Complete Intersection Calabi Yau manifolds. It further contains functions for determining various topological quantities, such as Chern classes, triple intersection and Hodge numbers. Installation is straighforwad with pip

```console
pip install pyCICY
```


## Quickstart

Import the CICY object from the module

```python
from pyCICY import CICY
```

Next define a CICY, for example the tetraquadric:

```python
M = CICY([[1,2],[1,2],[1,2],[1,2]])
```

Now we are able to do some calculations, e.g.

```python
M.line_co([1,2,-4,1])
```

determines the hodge numbers of the line bundle L = O(1,2,-4,1).

Since the rank computation takes the most time we included [SpasM - github](http://github.com/cbouilla/spasm). The *rank_hybrid* executable of SpaSM has to be in your $PATH.

```python
T = CICY([[1,2,0,0,0],[1,0,2,0,0],[1,0,0,2,0],[1,0,0,0,2],[3,1,1,1,1]])
```

and do some computations:

```python
T.line_co([3,-4,2,3,5], SpaSM=True)
```

## Visualisation

Forked from: https://github.com/Kuo-TingKai/CalabiYauViz

`pyCICY.viz` renders the classic Calabi-Yau cross-section: the degree-n Fermat
hypersurface z1^n + z2^n = 1, projected from 4 real dimensions to 3.

```python
from pyCICY import viz

viz.plot(5)          # the quintic
viz.plot_grid(range(2, 9))
```

Because the degree-n Fermat hypersurface in P^(n-1) is exactly the CICY
`[[n-1, n]]`, the two halves of the package line up. `viz.describe` pulls the
real invariants out of the `CICY` object, so plots are labelled with computed
topology rather than just an index:

```python
>>> viz.from_cicy([[4, 5]])
5
>>> viz.describe(5)
'n=5  $h^{1,1}$=1  $h^{2,1}$=101  $\\chi$=-200'
```

The surface can also be exported as a single STL mesh:

```python
viz.write_stl('quintic.stl', 5, res=40)
```

There is a command line front end too:

```console
pycicy-viz --cicy '[[4,5]]' -r 40 -o quintic.png --stl quintic.stl
pycicy-viz --grid --range 2 9
```

Plotting needs matplotlib, which is an optional extra:

```console
pip install pyCICY[viz]
```

The rest of the package has no plotting dependency, and importing `pyCICY`
does not pull matplotlib in.

## Tests

```console
python3 run_tests.py
```

`tests/test_pycicy.py` checks the topological machinery against values from the
CICY literature (quintic, bicubic, tetraquadric, sextic fourfold) plus the
Euler identities. `tests/test_viz.py` checks the vectorised surface evaluation
against the original symbolic sympy formulation to machine precision, and
validates the STL output structurally. The full run takes a few minutes,
mostly in the line bundle cohomology checks.

## Local geometries and quantized mirror curves

`pyCICY.toric` and `pyCICY.quantum_curve` cover the *local* (non-compact,
toric) Calabi-Yau side, where the combinatorial backbone is a two-dimensional
reflexive polygon rather than a configuration matrix.

There are sixteen reflexive polygons up to `GL(2,Z)`. The table is not taken
on trust: `toric.verify_named()` rederives it by brute force, and the tests
check the twelve theorem, the duality involution and the identification of
the five smooth toric del Pezzo surfaces.

```python
from pyCICY import toric

toric.describe(toric.polygon('B3'))     # 'B3  6-gon  bdry=6  K^2=6  smooth'
toric.dual_name('P2')                   # 'T9'
toric.hoppings(toric.polygon('B3'))     # the six triangular-lattice neighbours
```

Quantizing the mirror curve of `K_S` turns it into an electron hopping on a
2d lattice in a magnetic field, following Sugimoto, *Calabi-Yau geometry and
electrons on 2d lattices*, [arXiv:1701.01561](https://arxiv.org/abs/1701.01561),
which generalises the local `F_0` / square lattice observation of Hatsuda,
Katsura and Tachikawa to local `B_3 = dP_3` and the triangular lattice. The
dictionary is: lattice points of the polygon are hopping vectors, and
`hbar/2pi` is the magnetic flux per unit cell.

```python
from pyCICY import quantum_curve as Q

Q.harper().gap_labels(2, 5)             # gap Chern numbers -2, 1, -1, 2
flux, energy = Q.butterfly('B3')        # the triangular-lattice butterfly
```

The spectrum is symmetric under `E -> -E` exactly when the lattice is
bipartite, which is a condition on the polygon *modulo two* and not a
reflection symmetry of it. Local `B_3` is centrally symmetric and still
spectrally chiral; `T4 = P(1,1,2)` is not centrally symmetric and is
spectrally symmetric. Only `F_0` and `T4` are bipartite, so fourteen of the
sixteen geometries are spectrally chiral. `E(Phi) = -E(1-Phi)` holds for all
sixteen. The survey script prints the whole table:

```console
python3 examples/toric_survey.py
python3 examples/toric_survey.py --flux 2/5 --gaps B3
python3 examples/toric_survey.py --plot F0 B3 --out /tmp
```

The overlap with the compact CICY side is deliberately narrow. The mirror
curve of `K_S` is an anticanonical curve of `S`, and when `S` is a product of
projective spaces that curve is a CICY one-fold, so `toric.anticanonical_cicy`
returns `[[2,3]]` for local `P^2` and `[[1,2],[1,2]]` for local `F_0`, and
`None` for the other fourteen. The local geometries are not themselves CICY
threefolds and the module does not pretend otherwise.

## Knots, chirality and unknotting number

`pyCICY.knots` computes the Jones polynomial from a planar diagram. It needs
no SnapPy, Sage or spherogram; the diagrams, including the fifteen-crossing
census knot `K15n81556`, are stored as data.

It exists because `pyCICY.additivity` already reasons by analogy about
Brittenham and Hermiller, *Unknotting number is not additive under connected
sum*, [arXiv:2506.24088](https://arxiv.org/abs/2506.24088), and the knot side
turns out to be directly checkable. Wang and Zhang,
[arXiv:2507.14265](https://arxiv.org/abs/2507.14265), observed that the two
diagrams of `K15n81556` in that argument represent a chiral knot and its
mirror image rather than the same knot, detected by the Jones polynomial:

```python
from pyCICY import knots

knots.from_name('3_1').jones()          # -t^-4 + t^-3 + t^-1
knots.torus_knot(2, 7).jones()          # agrees with from_name('7_1')
knots.chirality_report('K15n81556')     # chiral: True
```

The determinant of `K15n81556` is 39 for the knot and for its mirror, so
chirality needs an invariant that is not symmetric under `t -> 1/t`. Only
`4_1` and `6_3` in the table are not detected as chiral, and the Jones test
is sufficient but not necessary, so that means "not detected" rather than
"amphichiral".

Mirroring, connected sums, crossing changes and braid closures are all
supported. Unknotting numbers are *quoted* from the literature rather than
computed, since they are minima over all diagrams;
`knots.unknotting_search` gives diagram-dependent upper bounds only and its
docstring is explicit about why it cannot recover the Brittenham-Hermiller
bound by brute force.

```console
python3 examples/knot_chirality.py
python3 examples/knot_chirality.py --search 3     # slow, finds nothing, as expected
```

## Heterotic line bundle standard models

`pyCICY.bundles` supplies the piece `pyCICY.phenomenology` explicitly declines
to: a gauge bundle that is not the standard embedding. The construction is
that of Anderson, Gray, Lukas and Palti,
[arXiv:1106.4804](https://arxiv.org/abs/1106.4804) and
[arXiv:1202.1757](https://arxiv.org/abs/1202.1757), in which

    V = O(L_1) + ... + O(L_5),    sum_a L_a = 0,

so the structure group is `S(U(1)^5)` inside `SU(5)` inside `E_8`. Every
cohomology group the spectrum needs is then a sum of line bundle
cohomologies, which `CICY.line_co` already computes exactly.

```python
from pyCICY import CICY, bundles

X = CICY([[1,2],[1,2],[1,2],[1,2]])
V = bundles.LineBundleSum(X, [[-2,-2,-1,2],[-2,1,0,0],[1,-2,1,0],
                              [1,1,-1,0],[2,2,1,-2]])
V.index()                       # -6.0, from the triple intersection numbers
V.index_from_cohomology()       # -6.0, from the Leray spectral sequence
V.su5_spectrum()                # n(10)=24, n(10-bar)=18, n(5-bar)=54, ...
V.stability_locus()['found']    # True
```

The index is computed twice by code with nothing in common — one contraction
of `d_rst` against the charges, one run of the Leray spectral sequence per
summand — and a third time as the alternating sum of the computed
cohomology. Serre duality `h^q(V) = h^{3-q}(V*)`, of which `line_co` knows
nothing, is checked as a fourth constraint. This is the `node_validation`
discipline applied to bundles.

Monads `0 -> V -> B -> C -> 0` get their topology here too, but **not** their
cohomology. The `h^q(V)` follow from the long exact sequence only up to the
ranks of the maps `H^q(B) -> H^q(C)`, which depend on the morphism and not on
the degrees; `Monad.cohomology_bounds` returns the interval the sequence
gives and refuses to guess the rest.

### Why the ordering of the filters is the whole design

The conditions on a candidate model do not cost remotely the same, so `scan`
applies them in increasing order of price and stops before the expensive
ones:

| stage | cost per bundle |
| --- | --- |
| `c_1(V) = 0` | integer addition |
| index `= -3\|Gamma\|` | one tensor contraction |
| anomaly `c_2(TX) - c_2(V)` effective | one tensor contraction |
| poly-stability | 0.03 ms as a sign test, 30 ms as a numerical search |
| cohomology | 35 `line_co` calls for a rank 5 model |

Poly-stability needs all five slopes `mu(L_a) = d_rst L_a^r t^s t^t` to
vanish at one point of the Kahler cone. There, *every partial sum* of slopes
vanishes too, so if any subset of the summands has `sum_{a in S} M_a` with no
negative entry, that form is strictly positive on the open positive orthant
and no such point exists. The test is exact, it is at most thirty sign tests,
and it inherits to pairs — which is where `scan` applies it, to the pair
table before the sums are ever assembled. Without that, the tetraquadric at
charge 2 is of order `10^8` assemblies and the search does not finish.

On the tetraquadric with `|Gamma| = 2`, the funnel runs 200000 → 3 → 2:

```console
python3 examples/line_bundle_models.py --charge 2 --budget 12
make bundles CHARGE=2 ORDER=2
```

Two smaller results worth recording. No rank 5 line bundle sum on the
tetraquadric has `ind(V) = -3` within charge 2: the only non-zero entry of
`d_rst` there is 2, so `6*ind` is even. Three generations therefore needs a
freely acting quotient, which is the same conclusion
`phenomenology.generation_survey` reaches for the standard embedding by a
completely different route. And the anomaly condition is *not* vacuous, though
it nearly looks it: `ch_2(V)_r = (1/2) d_rst sum_a L^s L^t` has `d_rst >= 0`
and `sum_a L_a L_a` positive semi-definite, but a positive semi-definite
matrix has negative off-diagonal entries, so the contraction is not sign
definite and the condition genuinely cuts.

Every search takes an explicit charge box, a result `limit` and an optional
`max_seconds`, and warns when it returns a truncation rather than a complete
answer. That is deliberate: a truncated list and an exhaustive one are
different objects and should not be presented alike.

## Chirality across domains

`pyCICY.chirality` puts one interface over the mirror operations of the other
modules. Three of them turn out to have the same shape: an involution that
swaps a pair of integers and preserves their sum or span.

| domain | involution | swapped pair | preserved |
| --- | --- | --- | --- |
| knot | mirror image | extreme degrees of `V(t)` | span of `V` |
| reflexive polygon | polar duality (Batyrev) | `(#dP, #dP*)` | 12 |
| Calabi-Yau threefold | mirror symmetry | `(h^{1,1}, h^{2,1})` | `h^{1,1} + h^{2,1}` |

```python
from pyCICY import chirality

chirality.mirror_pair('B3')                 # (6, 6) -- self-dual
chirality.mirror_invariant('P2')            # 12
chirality.chirality('K15n81556')['fixed']   # False
print(chirality.format_survey())
```

The fixed points are found rather than tabulated: the amphichiral knots of
the table, the four self-dual reflexive polygons `B3`, `T6`, `Q6`, `P6`, and
the Hodge pairs with `h^{1,1} = h^{2,1}`. That the preserved quantity is
meaningful on the knot side is checked through the
Kauffman-Murasugi-Thistlethwaite theorem: the span of `V` equals the crossing
number exactly for the alternating knots.

The quantized mirror curve is included as the case where the analogy
**fails**. Reflecting its Newton polygon leaves the spectrum exactly
unchanged, so no spectral invariant can detect that involution and
`chirality` reports `detected=None` rather than `False`. What the spectrum
does see is bipartiteness, which is a logically independent property: all
four combinations of the two occur among the sixteen polygons, with `B3`
fixed by reflection yet spectrally chiral and `T4` the other way round.

Finally, `chirality.cicy_list_chirality()` asks whether the published list of
7890 CICY threefolds is closed under mirror symmetry. It is not: of the 265
distinct Hodge pairs, the only ones whose mirror also appears are the two
self-mirror pairs, so the list contains **no non-trivial mirror pair at all**.
`h^{1,1}` never exceeds 19 while `h^{2,1}` reaches 101, equivalently every
Euler characteristic in the list is non-positive. (22 entries record `0` for
both Hodge numbers; that is a sentinel for "not given" on the product
configurations, not a value, and it is excluded rather than counted as
self-mirror.)

```console
python3 examples/chirality_zoo.py
python3 examples/chirality_zoo.py --domain curve
```

## A-polynomials and the AJ conjecture

`pyCICY.apolynomial` closes the loop between the two ends of the package.
`pyCICY.knots` computes Jones polynomials; `pyCICY.quantum_curve` quantizes
mirror curves. The object joining them is a plane curve in `(C*)^2`, and on
the knot side that curve is the A-polynomial. The AJ conjecture says the
colored Jones polynomials obey a `q`-difference equation whose operators
satisfy `L Q = q Q L` -- the same Weyl algebra `quantum_curve` uses, with
`q = exp(i*hbar)` -- and that setting `q = 1`, `Q = M^2` recovers the
classical A-polynomial.

```python
from pyCICY import apolynomial as ap

ap.colored_jones_torus(2, 3, 3)     # third colour of the trefoil
ap.boundary_slopes(ap.apolynomial('4_1'))     # [-4, 4]
ap.verify_aj()                                # the trefoil, ~20s
```

Colored Jones for torus knots comes from the Rosso-Jones formula in the form
of [Hikami and Lovejoy](https://arxiv.org/abs/1409.6243). At `N = 2` it
reproduces `knots.jones()` **exactly** for T(2,3), T(2,5), T(2,7) and
T(3,4) = 8_19 -- a representation-theoretic sum against a sum over `2^n`
Kauffman states, from code sharing nothing, agreeing coefficient for
coefficient.

`find_recursion` searches for annihilating `q`-difference operators by linear
algebra on the colored Jones table, and `classical_limit` sets `q = 1`. For
the trefoil the smallest L-degree admitting a solution is 3, and the gcd of
the classical limits is `(L-1)^2 (M^2-1) (L M^6 + 1)`, which contains the
trefoil's A-polynomial exactly: the geometric factor `1 + L M^{pq}` with
`pq = 6`, and the abelian factor `L - 1`. Two caveats are stated in the code
rather than glossed: the leftover factor is expected, since the classical
limit of *an* annihilating operator contains the A-polynomial without
equalling it; and the L-degree is minimal only *within the search bounds*,
which `find_recursion` takes explicitly and never claims otherwise.

The edge slopes of the Newton polygon are boundary slopes of incompressible
surfaces (Cooper, Culler, Gillet, Long, Shalen), and they come out right:
`pq` for each torus knot, `+-4` for the figure-eight. Since the Newton
polygon is also what `QuantumCurve` consumes, `to_quantum_curve` hands a
knot straight to the lattice machinery. That is a statement about the
quantization *rule*, not the geometry: a toric diagram is a reflexive
polygon and an A-polynomial's Newton polygon is not, so the resulting
operator is **not** the mirror curve of any local Calabi-Yau, and the
docstring says so.

A-polynomials themselves are quoted with attribution, not derived --
computing one means eliminating variables from gluing equations, which is a
different project.

```console
python3 examples/aj_conjecture.py
python3 examples/aj_conjecture.py --skip-recursion
```

## Hyperbolic lattices

`pyCICY.hyperbolic` implements the automorphic Bloch theory of Maciejko and
Rayan, *Automorphic Bloch theorems for hyperbolic lattices*,
[PNAS 119(9) e2116869119](https://doi.org/10.1073/pnas.2116869119). On a
hyperbolic `{p,q}` lattice the ordinary Bloch theorem fails, because the
translation group is Fuchsian and not abelian; periodic boundary conditions
compactify the lattice onto a genus-`g` surface and eigenstates transform
under a unitary representation of that group.

```python
from pyCICY import hyperbolic as hyp

hyp.circumradius(8, 8), hyp.solve_circumradius(8, 8)   # closed form vs derived
hyp.relator_holds(8)          # the octagon side-pairing relator
hyp.cell_area(8, 8)           # 4*pi*(g-1), so genus 2 by Gauss-Bonnet
hyp.boundary_fraction(8, 8, 3)
hyp.compare_sectors(g=2, dims=(1, 2, 3))
```

Three things are worth flagging. The three standard length formulas
(`cosh R = cot(pi/p)cot(pi/q)`, `cosh r = cos(pi/q)/sin(pi/p)`,
`cosh(l/2) = cos(pi/p)/sin(pi/q)`) are easy to permute, so
`solve_circumradius` derives `R` numerically from the requirement that the
cell's interior angle be `2*pi/q`, and the tests compare the two. The
relator of the regular `4g`-gon with opposite sides identified is
`g0 g1^-1 g2 g3^-1 ... = 1`, which is **not** the canonical surface word
`prod_i [a_i, b_i] = 1`; both present the same group but in different
generators, and the tests assert that the canonical word does *not* hold on
these generators. And Gauss-Bonnet confirms the genus independently, since
the cell area equals `4*pi*(g-1) = 2*pi*|chi|`.

The abelian sector collapses to `E(k) = 2t sum_j cos k_j` on the Jacobian
torus `T^{2g}` -- a `2g`-dimensional hypercubic band. Higher-dimensional
sectors are built from clock and shift matrices, whose commutator
`[X^a, Z^b] = omega^{-ab}` is a *scalar*, which is what lets the relator be
solved in closed form.

`boundary_fraction` explains why any of this is needed: for a hyperbolic
flake it tends to `(p-2)/(p-1)`, not to zero (6/7 for `{8,8}`, 10/11 for
`{12,12}`), because each ring is `p-1` times the last. A finite patch is
never a good stand-in for the bulk.

What is **not** implemented is the enumeration of normal subgroups of a
Fuchsian group, which is what would give the complete set of irreducible
representations and hence the full spectrum; that is a job for GAP. The
sectors here are genuine sectors, but nothing claims they exhaust anything.

```console
python3 examples/hyperbolic_bloch.py
python3 examples/hyperbolic_bloch.py --genus 3 --plot /tmp
```


## Literature

The module has been developed in the context of the following papers:

- Yasuyuki Hatsuda, Yuji Sugimoto, Zhaojie Xu (2017) Calabi-Yau geometry and electrons on 2d lattices
  - https://arxiv.org/abs/1701.01561 
- Joseph Maciejko and Steven Rayan (2022) Automorphic Bloch theorems for hyperbolic lattices
  - https://www.pnas.org/doi/10.1073/pnas.2116869119 
- Chao Wang, Yimu Zhang (2025) A remark on the counterexample to the unknotting number conjecture
  - https://doi.org/10.48550/arXiv.2507.14265

- Lara B. Anderson, Andrei Constantin, James Gray, Yang-Hui He, Seung-Joo Lee (Jun 25, 2026) CIPro Package: Complete Intersections in Products of Projective Spaces and Line Bundles
  - https://arxiv.org/pdf/2606.27588
- Lara B. Anderson, James Gray, Sunit A. Patil, Caoimhín Scanlon (2025) Mapping moduli across heterotic conifolds
  - https://arxiv.org/pdf/2512.18124

- Larfors, Magdalena and Schneider, Robin (2019) Line bundle cohomologies on CICYs with Picard number two
  - arXiv; 1906.00392 hep-th; doi:10.1002/prop.201900083; Fortsch. Phys.

- Hubsch, Tristan (1994) Calabi-Yau manifolds: A Bestiary for physicists
  - World Scientific; 9789810219277, 981021927X
- Lara B. Anderson (2008) Heterotic and M-theory Compactifications for String Phenomenology
  - arXiv;hep-th;0808.3621 https://inspirehep.net/record/793857/files/arXiv:0808.3621.pdf



## Useful software
The SpaSM library can be found here: [github](http://github.com/cbouilla/spasm)
pyCICY works nicely with [Sage](http://www.sagemath.org/). Other useful packages for dealing with Calabi Yau manifolds in toric varieties are [cohomCalg](https://github.com/BenjaminJurke/cohomCalg/) and [PALP](http://hep.itp.tuwien.ac.at/~kreuzer/CY/CYpalp.html).
