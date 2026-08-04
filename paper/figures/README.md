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
viz.plot_hyperbolic_flake(8, depth=2)          # {8,8} in the Poincare disk
viz.plot_apolynomial('4_1')                    # Newton polygon and slopes
```

`paper/make_figures.py` writes every figure **twice**, as a PDF for the LaTeX
supplement and a PNG of the same content for this README, because GitHub
renders PNG inline in Markdown but will not render an embedded PDF. The PNGs
are tracked in the repository; the PDFs are not, and are rebuilt on demand.

```console
make figures        # everything, including the split web (slow)
make new-figures    # only the six below, no web build needed
make supplement     # figures, then the LaTeX document
```
