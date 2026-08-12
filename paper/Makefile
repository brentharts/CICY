default:
	pdflatex supplementary_material_crossover.tex
	pdflatex supplementary_material_crossover.tex
	open supplementary_material_crossover.pdf
	
data:
	python make_crossover_facts.py
	
install:
	pip install candl-like --break-system-packages
	pip install jax jaxlib --break-system-packages
	pip install camb --break-system-packages
	git clone https://github.com/Lbalkenhol/candl_data.git
	cd candl_data
	pip install . --break-system-packages

