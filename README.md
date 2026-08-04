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
