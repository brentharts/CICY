# What Nine Links Determine, and What Only a Fit Can Say

## Exact methods for nine-link Yukawa textures, the unitarity triangle, and the strong CP problem

### A Python implementation of Arkani-Hamed, Figueiredo, Hall and Manzari, with three discrepancies recorded and one arithmetic question settled

---

## Summary

The angles of the CKM unitarity triangle sit close to (π/2, π/8, 3π/8). That
could be an accident, or it could be the visible end of a theory in which CP is
an exact symmetry broken by a vacuum with discrete phases. The obstacle to
telling the difference is that the unitarity triangle is a heavily processed
function of the Yukawa matrices, so simplicity in one need not show up as
simplicity in the other.

Arkani-Hamed, Figueiredo, Hall and Manzari find a setting where it does. Flavour
space is ten-dimensional, so the ten observables can be carried by Yukawa
matrices with exactly nine non-zero entries and one irremovable phase — *nine-link
textures*. Fit every such texture to the data and the fitted phase clusters at
multiples of π/8, and the reason is not a fact about fits: for hierarchical
matrices, texture zeros are what let one angle of the unitarity triangle *equal*
that phase at leading order. Beyond leading order the two differ by computable
amounts, which turns the pattern into sub-degree predictions that the next
generation of flavour measurements will confirm or kill.

The interesting layer is therefore combinatorial, and `pyCICY.theories.ninelink`
implements it with no fitting anywhere. What follows records what comes out,
including four things worth stating in their own right:

1. **Two counts that are not in the paper.** From the conditions alone there are
   **1592** nine-link textures, in **36** orbits under relabelling the quark
   fields.
2. **The strong-CP claim, and its true scope.** Over the full enumeration, the
   textures admitting a real determinant are *precisely* those with a single
   perfect matching in each sector — 1376 of 1592. The paper's stronger remark,
   that the failures are all diagonal in one sector, holds for its fitted subset
   and not in general.
3. **Three discrepancies**, each checkable in exact arithmetic: a phase placed
   outside its own loop, a next-to-leading correction that does not describe its
   residual, and a missing factor of *i*.
4. **The monotile question, settled.** Whether the Spectre monotile's arithmetic
   can supply the flavour phases is not a question about the number nine. It is a
   question about square classes and cyclotomic fields, and the answer is *no,
   except for π/2*.

---

## The third base class

The package already had two: `Theory`, for what four-dimensional physics comes
out of a geometry, and `SurfaceTheory`, for what an amplitude equals. A theory of
flavour asks a third thing — given the gauge group and the matter content, what
structure in the Yukawa matrices accounts for the measured masses, mixings and CP
violation — so it gets `FlavorTheory`, with verbs `texture`,
`rephasing_invariant`, `predicted_angle` and `deviation`.

Reusing `Theory` was tempting, because for once its verbs nearly apply: there is
a gauge group and there is a quark spectrum. The reason not to is
`holomorphic_yukawa`, which in this package means a cup product of cohomology
classes — quasi-topological, exact, computed. A Yukawa texture is a
phenomenological ansatz for the same matrix. One method meaning both would blur
exactly the distinction the interface exists to keep.

`NeedsFit` is *not* a fifth ledger category. It is the flavour-side
`NeedsMetric`: a named missing ingredient. Two things make a fit different from a
calculation, and both are worth saying. It is numerical. And it is a search, so a
scan reporting "these are the solutions" is really reporting "these are the
solutions my scan found in this box". The exception carries `missing`, and also
`available`, which names the exact layer that *is* computable — a refusal should
still tell you where to go.

---

## The enumeration

A nine-link texture is specified by which entries are present. The conditions are
that there are nine links in total, that both matrices are full rank, that the
diagram is connected, and that the (3,3) entries survive. `enumerate_textures`
applies them directly:

| quantity | value |
| --- | --- |
| nine-link textures | 1592 |
| orbits under `S₃^Q × S₃^{uᶜ} × S₃^{dᶜ}` | 36 |
| splits (4+5, 5+4) | 692 each |
| splits (3+6, 6+3) | 104 each |
| loop lengths | 4 (1272), 6 (320) |

The loop lengths come out 4 and 6, which is what the paper says. The counts of
diagrams are not comparable with the paper's 156 classes or its (29, 35, 35) at
fixed phase, because those are counts of *fits* — one diagram supports several
inequivalent minima, and the phase can be fixed to different multiples of π/8 on
the same diagram. Saying which count is which is the point of reporting both.

### Why connectivity is the whole condition

Nine links on nine nodes, and the cycle rank of a graph is `E − V + C`. So
connectivity forces exactly one independent loop; "a single closed loop carrying
the phase" and "the diagram is connected" are the same condition. That is a
tidier statement than checking for loops directly, and it generalises:

> **The number of irremovable CP phases in any Yukawa texture is `E − V + C`.**

Every link carries a phase, every field can be rephased, and what survives is the
first homology of the link diagram. `cycle_rank` computes it, and over the whole
enumeration it is 1 — which is what makes the nine-link case the one where a
single phase is the entirety of CP violation. Add one link and it becomes 2. The
paper's analysis of deformations away from nine links is, in this language, an
analysis of what the extra loops do.

The invariant itself is tested rather than assumed: assign an arbitrary phase to
each of the nine fields, rotate every entry accordingly, and require the argument
of the loop monomial not to move. It doesn't, for every texture tested.

---

## Strong CP is graph theory

Each term of `det(Yu)` is a perfect matching of that sector's diagram. If the
phase sits on a link belonging to no matching, that determinant never sees it and
stays real; if every matching contains the link, the determinant is real times a
phase, which is not the same thing. So whether `arg det(Yu Yd) = 0` can be
arranged is decided by matchings and loops, with no numbers involved.

Over the full enumeration:

| | count |
| --- | --- |
| textures admitting a real determinant | 1376 |
| textures that do not | 216 |
| of those, diagonal in one sector | 56 |
| textures with a single matching in each sector | 1376 |

The last two rows are the content. **Safe is exactly single-matching-in-both** —
the two sets coincide, which is a clean equivalence the module asserts rather
than a coincidence it notices. And the paper's remark that the failures are all
diagonal in one sector is true of the textures that survived its fits, but not in
general: of 216 failures only 56 are diagonal. `strong_cp_census` returns the
counts rather than the claim.

---

## Three discrepancies

This package's standing policy for implementing a paper is that where a stated
result does not follow from the stated inputs, the function computes what the
inputs give and says so. Three cases arose.

### The phase outside its own loop

The paper works three textures in detail, deriving next-to-leading corrections
for α, β and γ. Rather than repeat the expansion, the module diagonalises the
same matrices exactly and evaluates the published closed form on the CKM matrix
that produces. Examples 1 and 3 agree, with the residual falling like ε⁴.

Example 2 does not, and the reason is specific. Its diagram has nine links on
nine nodes, so exactly one loop, and that loop is `{Yu13, Yu33, Yd13, Yd33}`.
**The entry the paper marks with the phase, `Yd32`, is not in it.** A phase there
is removable by rephasing, and diagonalising the matrices as printed gives β = 0
exactly, not π/8.

Moving the phase to `Yd13` — the down-type loop entry with the smallest indices,
which is the paper's own stated placement rule — restores β = π/8 at leading
order, as advertised. But the printed correction then does not describe the
residual: the exact departure from π/8 falls like ε⁴, while the quoted correction
is of order ε², so their difference does not vanish with ε at all.

| ε | β as printed | β relocated | departure from π/8 | closed-form residual |
| --- | --- | --- | --- | --- |
| 0.100 | 0 | 0.392671 | −2.8·10⁻⁵ | +4.8·10⁻³ |
| 0.050 | 0 | 0.392697 | −1.7·10⁻⁶ | +2.4·10⁻³ |
| 0.025 | 0 | 0.392699 | −1.1·10⁻⁷ | +1.2·10⁻³ |

The first finding looks like a transcription slip. Whether the second follows from
it — with the phase in a different link the subleading structure is a different
calculation — or is independent, `example_two_anomaly` returns the numbers and
takes no view.

### A factor of *i* in the Z8 sketch

The paper sketches a vacuum in which two flavon vevs of equal magnitude with
phases differing by π/4 produce `i(1 − e^{iπ/4})/(1 + e^{iπ/4})`, identified with
`i·tan(π/8)` — which is what makes the Yukawa triangle a right triangle.

In exact arithmetic `(1 − e^{iθ})/(1 + e^{iθ}) = −i·tan(θ/2)`, so the bracket is
`−i·tan(π/8)` and the whole expression is `tan(π/8)`. Real. A real ratio is a
degenerate triangle, not a right one. A single factor of *i*, reported as
computed.

### The π/4 coincidence, at eleven percent

The peak at φ ≈ π/4 is unrelated to the unitarity triangle; it needs
`(y_s/y_b)²·|V_us/V_ub|² ≈ √2`, which is a statement about measured quantities
and nothing else. On the paper's own Table S2 central values this evaluates to
1.262 against 1.414 — about 11%, where the paper says 9%. Minor, and possibly a
difference in which determination is used, but the module reports its own number.

Against these, the parts that do reproduce: the leading-order value of every
worked example, the closed forms of examples 1 and 3, the loop lengths, and the
side ratios of the special triangle, which are derived from the right triangle
rather than quoted —

```
|R_alpha| = cot(pi/8)  = 1 + sqrt 2
|R_beta|  = 1/cos(pi/8)
|R_gamma| = sin(pi/8)  = cos(3 pi/8)
```

with `tan(3π/8) = 1 + √2` as an exact surd identity.

---

## The monotile question

The package already contains an exact treatment of the Spectre monotile's
substitution — `theories.spectre`, implementing the substrate of

> B. S. Hartshorn, *One Tile, Two Units: Area Inflation in Q(√15) and Boundary
> Inflation in Q(√5) for the Spectre Monotile*

— whose substitution acts on **nine species**. A nine-link texture has **nine
links**. The temptation to read something into that is exactly why the question
deserves an answer rather than a wink.

### The nines are unrelated

Nine links is ten flavour observables minus the one phase. Nine species is the
number of metatiles the Spectre substitution needs. Neither number constrains the
other, and the objects are not alike: the substitution matrix has 56 non-zero
entries and row sums as large as 16, while a link diagram has nine entries and
every row sum at most 3. No texture's diagram is anything like it, and the module
checks that rather than asserting it.

### The arithmetic, which is the real question

The special angles need √2. `tan(π/8) = √2 − 1`, and the triangle's side ratios
are `1 + √2` and `1/cos(π/8)`.

The substitution's inflation factor is `λ = (√6 + √10)/2`, with
`λ² = 4 + √15`, the fundamental unit of `Z[√15]`. Its field is `Q(√6, √10)`, a
multiquadratic field whose quadratic subfields are indexed by the square classes
generated by 6 and 10:

```
square classes: {1, 6, 10, 15}
```

so the quadratic subfields are exactly `Q(√6)`, `Q(√10)` and `Q(√15)`. **√2 is
not among them.** Neither is √3, the monotile's own placement field, which lives
in the geometry rather than in the inflation — only the product √15 appears.

So the substitution's unit arithmetic cannot supply the flavour phases. That is a
negative result, and it is exact: four elements in a set, checked by squarefree
parts.

### One corner survives, and it selects α

The question of which discrete phases two symmetries can both support is governed
by an intersection of cyclotomic fields, and that intersection is a greatest
common divisor:

| | intersection | shared phases |
| --- | --- | --- |
| `Q(ζ₈) ∩ Q(ζ₁₂)` | `Q(ζ₄) = Q(i)` | multiples of π/2 |
| `Q(ζ₁₆) ∩ Q(ζ₁₂)` | `Q(ζ₄)` | multiples of π/2 |
| `Q(ζ₈) ∩ Q(ζ₂₄)` | `Q(ζ₈)` | multiples of π/4 |

A vacuum whose phases are twelve-fold — as the monotile's geometry is — and a
texture wanting eight-fold phases share exactly the multiples of π/2.

That is α, and only α. Which is not nothing: the π/2 peak is the one with nine
equivalence classes, it is the right-triangle case, and its textures are the ones
whose determinants are most naturally real. **A monotile-flavoured spontaneous CP
violation could deliver α = π/2 and could not deliver β = π/8 or γ = 3π/8.**

And the same arithmetic says what would be needed instead. `lcm(8, 12) = 24`, so a
twenty-four-fold vacuum contains both structures: `Q(ζ₂₄) ⊃ Q(ζ₈)` and
`Q(ζ₂₄) ⊃ Q(ζ₁₂)`. If one wanted a monotile-adjacent origin for the full
(π/2, π/8, 3π/8) triangle, twenty-four-fold is the smallest place it could live.

This is the honest shape of the connection: not an identification, a constraint.
The two structures are compatible on a subgroup, incompatible off it, and the
subgroup is named.

---

## The ledger, filled in

| quantity | status |
| --- | --- |
| the enumeration of nine-link textures, and its orbits | exact, derived |
| loop lengths, and the loop of any diagram | exact, derived |
| the rephasing invariant, and its invariance under rephasing | exact, derived |
| the number of physical phases as `E − V + C` | exact, derived |
| perfect matchings as determinant terms | exact, derived |
| the strong-CP census, and the scope of the paper's claim | exact, derived |
| unitarity angles from given Yukawa matrices | exact, by diagonalisation |
| closed forms of worked examples 1 and 3 | exact, reproduced |
| worked example 2 | **does not reproduce**; the anomaly is reported |
| the Z8 ratio, and the π/4 coincidence | exact, reported as computed |
| triangle side ratios as surds | exact, derived |
| square classes and cyclotomic intersections | exact, derived |
| the phase histogram, and every count from it | `NeedsFit` |
| the predicted ellipses | `NeedsFit` |
| deriving a texture rather than assuming one | `NeedsFit` |

---

## Reproducing all of it

```bash
git clone https://github.com/brentharts/CICY.git && cd CICY
pip install -r requirements.txt
python3 tests/test_ninelink.py      # about a second
python3 run_tests.py                # all 36 suites
```

```python
from pyCICY.theories import ninelink as N

N.census()                  # 1592 textures, 36 orbits, loops of length 4 and 6
N.strong_cp_census()        # 1376 safe, and safe == single-matching-in-both
N.example_check(1)          # closed form against exact diagonalisation
N.example_two_anomaly()     # the phase that is not in its own loop
N.monotile_compatibility()  # square classes, and Q(z8) cap Q(z12) = Q(i)
```

---

## What remains open

- **Sorting the 36 orbits by which angle they predict.** The leading-order rule
  is combinatorial — which of `s^u₁₂`, `s^d₁₂`, `s₁₃` the zeros switch off
  decides whether the texture fixes α, β or γ. That partition can be computed
  without any fitting, and it would turn the enumeration into the paper's three
  families directly.
- **Deformations away from nine links**, which in this language is what happens
  as the cycle rank rises above one and the single phase becomes several.
- **The scan itself**, if it is ever wanted: implementable with the package's
  existing truncated-search discipline, reporting its box rather than its
  conclusions.
- **Whether example 2's correction is independently wrong**, or wrong only
  because the phase moved.

---

## References

- Nima Arkani-Hamed, Carolina Figueiredo, Lawrence J. Hall, Claudio Andrea
  Manzari (2026), *The Very Nearly Right Theory of Flavor*, arXiv:2607.27315.
- B. S. Hartshorn, *One Tile, Two Units: Area Inflation in Q(√15) and Boundary
  Inflation in Q(√5) for the Spectre Monotile* — the substitution implemented in
  `pyCICY.theories.spectre`, and the source of the nine species compared above.
- D. Smith, J. S. Myers, C. S. Kaplan, C. Goodman-Strauss (2023), *An aperiodic
  monotile* and *A chiral aperiodic monotile*.
- C. Jarlskog (1985), *Commutator of the quark mass matrices*, Phys. Rev. Lett.
  55, 1039 — the invariant measure of CP violation.
- G. C. Branco, D. Emmanuel-Costa, R. González Felipe (2000), *Texture zeros and
  weak basis transformations*, arXiv:hep-ph/9911418.
