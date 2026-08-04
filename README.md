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

## Figures

`pyCICY.viz` draws the objects the other modules compute, alongside the
original Fermat cross-section. matplotlib is imported lazily inside each
function, so importing `pyCICY.viz` stays cheap.

```python
from pyCICY import viz

viz.plot_polygon_grid()                        # all 16 polygons with their duals
viz.plot_butterfly_grid(['F0', 'B3'], gaps_at=(1, 3))
viz.plot_jones('K15n81556')                    # against its mirror
viz.plot_braid([1]*7 + [-2]*7, strands=3)      # 7_1 # m7_1
viz.plot_chirality_grid()                      # the cross-domain plot
```

`plot_chirality` is the unifying one. Every chirality record carries an
*asymmetry*, the combination of its invariant pair that the mirror negates,
and a *preserved* quantity that it does not. Plotting one against the other
puts mirror partners symmetrically about zero and fixed points on the axis.
Restricted to the Calabi-Yau records it is exactly the conventional Hodge
plot of `plot_hodge`, with `chi/2` horizontally -- so that familiar figure
turns out to be one panel of a family that also covers knots and reflexive
polygons. Quantized curves are omitted, since their involution has no
invariant pair at all.

The connected sum of two braid closures is the juxtaposition of their words
on one more strand than the two together, so `[1]*7 + [-2]*7` on three
strands is `7_1 # m7_1`, the knot of arXiv:2506.24088. `paper/make_figures.py`
emits `fig_polygons`, `fig_butterflies`, `fig_knots` and `fig_chirality`
along with the existing figures.

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

## Literature

The module has been developed in the context of the following papers:

- Lara B. Anderson, Andrei Constantin, James Gray, Yang-Hui He, Seung-Joo Lee (Jun 25, 2026) CIPro Package: Complete Intersections in Products of Projective Spaces and Line Bundles
  - https://arxiv.org/pdf/2606.27588
- Lara B. Anderson, James Gray, Sunit A. Patil, Caoimhín Scanlon (2025) Mapping moduli across heterotic conifolds
  - https://arxiv.org/pdf/2512.18124

```tex
@article{Larfors:2019sie,
    author = "Larfors, Magdalena and Schneider, Robin",
    title = "{Line bundle cohomologies on CICYs with Picard number two}",
    eprint = "1906.00392",
    archivePrefix = "arXiv",
    primaryClass = "hep-th",
    reportNumber = "UUITP-18/19",
    doi = "10.1002/prop.201900083",
    journal = "Fortsch. Phys.",
    volume = "67",
    number = "12",
    pages = "1900083",
    year = "2019"
}
````

Further literature can be found here:

```tex
@book{Hubsch:1992nu,
	author         = "Hubsch, Tristan",
	title          = "{Calabi-Yau manifolds: A Bestiary for physicists}",
	publisher      = "World Scientific",
	address        = "Singapore",
	year           = "1994",
	ISBN           = "9789810219277, 981021927X",
	SLACcitation   = "%%CITATION = INSPIRE-338506;%%"
}

@phdthesis{Anderson:2008ex,
	author         = "Anderson, Lara Briana",
	title          = "{Heterotic and M-theory Compactifications for String
	Phenomenology}",
	school         = "Oxford U.",
	url            = "https://inspirehep.net/record/793857/files/arXiv:0808.3621.pdf",
	year           = "2008",
	eprint         = "0808.3621",
	archivePrefix  = "arXiv",
	primaryClass   = "hep-th",
	SLACcitation   = "%%CITATION = ARXIV:0808.3621;%%"
}
```

The SpaSM library can be found here: [github](http://github.com/cbouilla/spasm)

```tex
@manual{spasm,
title = {{SpaSM}: a Sparse direct Solver Modulo $p$},
author = {The SpaSM group},
edition = {v1.2},
year = {2017},
note = {\url{http://github.com/cbouilla/spasm}}
}
```

## Useful software

pyCICY works nicely with [Sage](http://www.sagemath.org/). Other useful packages for dealing with Calabi Yau manifolds in toric varieties are [cohomCalg](https://github.com/BenjaminJurke/cohomCalg/) and [PALP](http://hep.itp.tuwien.ac.at/~kreuzer/CY/CYpalp.html).
