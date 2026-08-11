# Makefile for pyCICY
#
#   make            same as `make paper`
#   make figures    regenerate every plot used by the paper
#   make paper      figures, then pdflatex twice -> paper/supplementary_material.pdf
#   make test       run every test suite
#   make survey     print the split-web survey to the terminal
#   make clean      remove LaTeX build litter
#   make distclean  also remove generated figures and the PDF
#   make cache-info / cache-clear   inspect or empty the computation cache
#
# The paper is built with pdflatex run twice: the first pass writes the .aux
# file, the second resolves the table of contents and the cleveref/hyperref
# cross-references, which would otherwise show up as ?? in the PDF.
#
# Figures are regenerated from source rather than checked in. The heavy
# cohomology is memoised by pyCICY.cache (SQLite, under /tmp by default, see
# PYCICY_CACHE), so only the first build pays full price.

PYTHON      ?= python3
PDFLATEX    ?= pdflatex
PAPERDIR    := paper
FIGDIR      := $(PAPERDIR)/figures
TEX         := supplementary_material.tex
PDF         := $(PAPERDIR)/supplementary_material.pdf
FIGSCRIPT   := $(PAPERDIR)/make_figures.py

# Depth of the splitting survey used for the figures. Raising this makes the
# first build slower but explores more of the web; see section 6 of the paper.
DEPTH       ?= 3
COMPARE_LIMIT ?= 500
FLUX        ?= 1/3
GENUS       ?= 2
HFLUX ?= 3/7
CHARGE ?= 1
ORDER ?= 2
BUDGET ?= 20
MAX_CONFIGS ?= 1200
FIB_LIMIT   ?= 7890
ORI_CONF    ?= [[4,5]]
TWISTOR_N   ?= 6

# The figure script writes these; facts.json carries the scalars quoted in
# the prose so the text and the plots cannot drift apart.
FIGURES := $(FIGDIR)/hodge_depth.pdf \
           $(FIGDIR)/hodge_favourable.pdf \
           $(FIGDIR)/node_counts.pdf \
           $(FIGDIR)/node_validation.pdf \
           $(FIGDIR)/web_growth.pdf \
           $(FIGDIR)/quintic_surface.pdf \
           $(FIGDIR)/ch2_check.pdf \
           $(FIGDIR)/gv_invariants.pdf \
           $(FIGDIR)/additivity.pdf \
           $(FIGDIR)/facts.json $(FIGDIR)/facts.tex

# Sources whose modification should invalidate the figures.
PYSRC := $(wildcard pyCICY/*.py) $(FIGSCRIPT)

.PHONY: all paper papers strings-paper strings-facts twistor-paper twistor-facts monotile-paper monotile-facts figures test survey toric-survey ftheory ftheory-fibrations orientifold twistor monotile knot-chirality chirality hyperbolic aj new-figures supplement clean distclean cache-info cache-clear data symmetries compare help

all: paper

# ---------------------------------------------------------------- figures

figures: $(FIGDIR)/.stamp

$(FIGDIR)/.stamp: $(PYSRC)
	@echo "==> generating figures (depth $(DEPTH))"
	@mkdir -p $(FIGDIR)
	$(PYTHON) $(FIGSCRIPT) --outdir $(FIGDIR) --depth $(DEPTH) \
	    --max-configs $(MAX_CONFIGS)
	@touch $@

# ------------------------------------------------------------------ paper

paper: $(PDF)

# pdflatex is run twice on purpose; see the note at the top of this file.
# -interaction=nonstopmode keeps a bad reference from hanging the build, and
# the exit status is still checked so a real error fails the make.
$(PDF): $(PAPERDIR)/$(TEX) $(FIGDIR)/.stamp
	@echo "==> pdflatex pass 1/2"
	cd $(PAPERDIR) && $(PDFLATEX) -interaction=nonstopmode -halt-on-error $(TEX) >/dev/null
	@echo "==> pdflatex pass 2/2 (resolving cross-references)"
	cd $(PAPERDIR) && $(PDFLATEX) -interaction=nonstopmode -halt-on-error $(TEX) >/dev/null
	@echo "==> wrote $(PDF)"
	@if grep -q 'Reference.*undefined' $(PAPERDIR)/supplementary_material.log; then \
	    echo "WARNING: undefined references remain:"; \
	    grep 'Reference.*undefined' $(PAPERDIR)/supplementary_material.log | sort -u; \
	fi

# ------------------------------------------------------------------ other

# ------------------------------------------------------------ strings paper
#
# The strings paper has no figures; its numbers come from
# paper/make_strings_facts.py, which recomputes every quantity the text
# quotes. pdflatex is run twice for the same reason as above.

STRINGSTEX  := supplementary_material_strings.tex
STRINGSPDF  := $(PAPERDIR)/supplementary_material_strings.pdf
STRINGSFACTS := $(FIGDIR)/strings_facts.tex

strings-paper: $(STRINGSPDF)

strings-facts: $(STRINGSFACTS)

$(STRINGSFACTS): $(wildcard pyCICY/theories/*.py) $(PAPERDIR)/make_strings_facts.py
	@echo "==> recomputing the numbers quoted in the strings paper"
	@mkdir -p $(FIGDIR)
	$(PYTHON) $(PAPERDIR)/make_strings_facts.py --sections strings --outdir $(FIGDIR)

$(STRINGSPDF): $(PAPERDIR)/$(STRINGSTEX) $(STRINGSFACTS)
	@echo "==> pdflatex pass 1/2 (strings)"
	cd $(PAPERDIR) && $(PDFLATEX) -interaction=nonstopmode -halt-on-error $(STRINGSTEX) >/dev/null
	@echo "==> pdflatex pass 2/2 (strings)"
	cd $(PAPERDIR) && $(PDFLATEX) -interaction=nonstopmode -halt-on-error $(STRINGSTEX) >/dev/null
	@echo "==> wrote $(STRINGSPDF)"
	@if grep -q 'Reference.*undefined' $(PAPERDIR)/supplementary_material_strings.log; then \
	    echo "WARNING: undefined references remain:"; \
	    grep 'Reference.*undefined' $(PAPERDIR)/supplementary_material_strings.log | sort -u; \
	fi

# ------------------------------------------------------------ twistor paper
#
# Same discipline as the strings paper: no figures, and every number comes
# from make_strings_facts.py, which recomputes it from pyCICY.twistor.

TWISTORTEX   := supplementary_material_twistor.tex
TWISTORPDF   := $(PAPERDIR)/supplementary_material_twistor.pdf
TWISTORFACTS := $(FIGDIR)/twistor_facts.tex

twistor-paper: $(TWISTORPDF)

twistor-facts: $(TWISTORFACTS)

$(TWISTORFACTS): pyCICY/twistor.py $(PAPERDIR)/make_strings_facts.py
	@echo "==> recomputing the numbers quoted in the twistor paper"
	@mkdir -p $(FIGDIR)
	$(PYTHON) $(PAPERDIR)/make_strings_facts.py --sections twistor --outdir $(FIGDIR)

$(TWISTORPDF): $(PAPERDIR)/$(TWISTORTEX) $(TWISTORFACTS)
	@echo "==> pdflatex pass 1/2 (twistor)"
	cd $(PAPERDIR) && $(PDFLATEX) -interaction=nonstopmode -halt-on-error $(TWISTORTEX) >/dev/null
	@echo "==> pdflatex pass 2/2 (twistor)"
	cd $(PAPERDIR) && $(PDFLATEX) -interaction=nonstopmode -halt-on-error $(TWISTORTEX) >/dev/null
	@echo "==> wrote $(TWISTORPDF)"
	@if grep -q 'Reference.*undefined' $(PAPERDIR)/supplementary_material_twistor.log; then \
	    echo "WARNING: undefined references remain:"; \
	    grep 'Reference.*undefined' $(PAPERDIR)/supplementary_material_twistor.log | sort -u; \
	fi

# ---------------------------------------------------------- monotile paper
#
# Same discipline again: every number recomputed from pyCICY.monotile. This
# one is slower to regenerate (~2 min) because the localizer signatures are
# exact LDL^T eliminations over Q(sqrt3).

MONOTILETEX   := supplementary_material_monotile.tex
MONOTILEPDF   := $(PAPERDIR)/supplementary_material_monotile.pdf
MONOTILEFACTS := $(FIGDIR)/monotile_facts.tex

monotile-paper: $(MONOTILEPDF)

monotile-facts: $(MONOTILEFACTS)

$(MONOTILEFACTS): pyCICY/monotile.py $(PAPERDIR)/make_strings_facts.py
	@echo "==> recomputing the numbers quoted in the monotile paper"
	@mkdir -p $(FIGDIR)
	$(PYTHON) $(PAPERDIR)/make_strings_facts.py --sections monotile --outdir $(FIGDIR)

$(MONOTILEPDF): $(PAPERDIR)/$(MONOTILETEX) $(MONOTILEFACTS)
	@echo "==> pdflatex pass 1/2 (monotile)"
	cd $(PAPERDIR) && $(PDFLATEX) -interaction=nonstopmode -halt-on-error $(MONOTILETEX) >/dev/null
	@echo "==> pdflatex pass 2/2 (monotile)"
	cd $(PAPERDIR) && $(PDFLATEX) -interaction=nonstopmode -halt-on-error $(MONOTILETEX) >/dev/null
	@echo "==> wrote $(MONOTILEPDF)"
	@if grep -q 'Reference.*undefined' $(PAPERDIR)/supplementary_material_monotile.log; then \
	    echo "WARNING: undefined references remain:"; \
	    grep 'Reference.*undefined' $(PAPERDIR)/supplementary_material_monotile.log | sort -u; \
	fi

# All four papers.
papers: paper strings-paper twistor-paper monotile-paper

test:
	$(PYTHON) run_tests.py

# Download the published CICY three-fold list. The data is redistributed by
# its authors, not by pyCICY, so it is fetched rather than vendored.
data:
	$(PYTHON) scripts/fetch_cicy_list.py --outdir data

# Braun's freely acting symmetries, from the Mathematica version of the list.
symmetries:
	$(PYTHON) scripts/fetch_symmetries.py --outdir data

compare: data
	$(PYTHON) -c "from pyCICY import cicylist as L; \
	e = L.load_published_list('data/cicylist.json'); \
	r = L.compare_to_published(e, limit=$(COMPARE_LIMIT)); \
	print('checked', r['checked'], 'agree', r['agree'], \
	      'disagree', len(r['disagree']), 'errors', len(r['errors']))"

survey:
	$(PYTHON) examples/split_survey.py --depth $(DEPTH) \
	    --max-configs $(MAX_CONFIGS)

# The sixteen reflexive polygons, their local Calabi-Yau data and the
# spectra of the 2d lattice models they quantize to (arXiv:1701.01561).
toric-survey:
	$(PYTHON) examples/toric_survey.py --flux $(FLUX)

# Six-dimensional F-theory: the non-Higgsable clusters derived from anomaly
# cancellation, the Hirzebruch and del Pezzo bases end to end, and the three
# ways a number can be unavailable.
ftheory:
	$(PYTHON) examples/ftheory_6d.py

# How much of the CICY list fibres in elliptic curves, by the block criterion
# of Anderson, Gao, Gray and Lee. Needs `make data`.
ftheory-fibrations:
	$(PYTHON) examples/ftheory_fibrations.py --limit $(FIB_LIMIT)

# Type IIB orientifolds: involutions of a CICY computed two ways, and Sen's
# weak coupling limit of the F-theory models.
orientifold:
	$(PYTHON) examples/orientifold.py --conf '$(ORI_CONF)' --scan

# Twistor theory: exact rational scattering amplitudes, BCFW against
# Parke-Taylor, and the positive Grassmannian.
twistor:
	$(PYTHON) examples/twistor.py --n $(TWISTOR_N)

# The Hat-Spectre aperiodic monotile family: exact substitution combinatorics
# and real-space topology over Q(sqrt3).
monotile:
	$(PYTHON) examples/monotile.py

# Chirality of K15n81556 and the failure of additivity of the unknotting
# number (arXiv:2506.24088, arXiv:2507.14265). SEARCH=N also runs the
# crossing-change search, which is slow; N=3 takes a few minutes.
knot-chirality:
	$(PYTHON) examples/knot_chirality.py $(if $(SEARCH),--search $(SEARCH),)

# The cross-domain comparison: one mirror operation over knots, reflexive
# polygons, quantized curves and Calabi-Yau threefolds. DOMAIN=knot narrows it.
chirality:
	$(PYTHON) examples/chirality_zoo.py $(if $(DOMAIN),--domain $(DOMAIN),)

# Just the four new figures, without rebuilding the whole split web.
# A-polynomials, colored Jones and the AJ conjecture. SKIP_RECURSION=1
# omits the nullspace search, which is the slow part.
aj:
	$(PYTHON) examples/aj_conjecture.py $(if $(SKIP_RECURSION),--skip-recursion,)

# Hyperbolic lattices and automorphic Bloch theory (PNAS 2116869119).
hyperbolic:
	$(PYTHON) examples/hyperbolic_bloch.py --genus $(GENUS)

# Heterotic line bundle standard models. CHARGE is the scan box and BUDGET the
# wall clock allowance per stage; both default small, because the topological
# conditions alone admit very large families and the search box grows like
# (2*CHARGE+1)^(h11*rank).
bundles:
	$(PYTHON) examples/line_bundle_models.py --charge $(CHARGE) --order $(ORDER) --budget $(BUDGET)

# The Hofstadter characteristic polynomial (arXiv:2312.14242).
hofstadter:
	$(PYTHON) examples/hofstadter_duality.py --flux $(HFLUX)

# The 24-cell: reflexive polytope, Batyrev Hodge numbers, flavour claims.
polytope:
	$(PYTHON) examples/twentyfour_cell.py $(if $(NO_CENSUS),--no-census,)

# The LaTeX supplement describing every figure. Needs the figures first.
supplement: figures
	cd paper && pdflatex -interaction=nonstopmode supplementary_material.tex \
	  && pdflatex -interaction=nonstopmode supplementary_material.tex

new-figures:
	@$(PYTHON) -c "import matplotlib; matplotlib.use('Agg'); \
	  import importlib.util as u; \
	  s=u.spec_from_file_location('mf','paper/make_figures.py'); \
	  m=u.module_from_spec(s); s.loader.exec_module(m); \
	  import os; d='paper/figures'; os.makedirs(d, exist_ok=True); \
	  m.figure_polygons(d); m.figure_butterflies(d); \
	  m.figure_knots(d); m.figure_chirality(d); \
	  m.figure_hyperbolic(d); m.figure_apolynomial(d)"

cache-info:
	@$(PYTHON) -c "from pyCICY import cache; import json; \
	print(json.dumps(cache.cache_info(), indent=2))"

cache-clear:
	@$(PYTHON) -c "from pyCICY import cache; \
	print('removed', cache.clear_cache(), 'entries')"

clean:
	rm -f $(PAPERDIR)/*.aux $(PAPERDIR)/*.log $(PAPERDIR)/*.toc $(PAPERDIR)/*.out \
	      $(PAPERDIR)/*.toc $(PAPERDIR)/*.fls $(PAPERDIR)/*.fdb_latexmk
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	rm -rf build *.egg-info

distclean: clean
	rm -f $(PDF) $(STRINGSPDF) $(TWISTORPDF) $(MONOTILEPDF) $(FIGURES) $(FIGDIR)/.stamp
	rm -f $(STRINGSFACTS) $(FIGDIR)/strings_facts.json
	rm -f $(TWISTORFACTS) $(FIGDIR)/twistor_facts.json
	rm -f $(MONOTILEFACTS) $(FIGDIR)/monotile_facts.json
	-rmdir $(FIGDIR) 2>/dev/null || true

help:
	@echo "targets: all paper papers strings-paper strings-facts twistor-paper twistor-facts monotile-paper monotile-facts figures test survey toric-survey knot-chirality chirality hyperbolic aj bundles hofstadter polytope new-figures supplement clean distclean cache-info cache-clear"
	@echo "vars:    DEPTH=$(DEPTH) MAX_CONFIGS=$(MAX_CONFIGS) CHARGE=$(CHARGE) ORDER=$(ORDER) BUDGET=$(BUDGET) FLUX=$(FLUX) HFLUX=$(HFLUX) PYTHON=$(PYTHON)"
