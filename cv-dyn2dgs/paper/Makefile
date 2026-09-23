# Build the CV-Dyn2DGS thesis.
#
#   make            xelatex (recommended - uses system Korean fonts)
#   make pdflatex   pdflatex route (needs a ko.TeX font set)
#   make tables     regenerate result tables from results/ (see ../scripts/make_tables.py)
#   make clean      remove build artefacts
#
# NOTE: this document has never been compiled. No LaTeX distribution was
# available in the environment where it was written, so typesetting errors may
# remain. Start with `make` and read the log.

MAIN    := main
TEX     := xelatex
BIBTEX  := bibtex
FLAGS   := -interaction=nonstopmode -halt-on-error -file-line-error

.PHONY: all pdflatex tables clean distclean check

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex preamble.tex refs.bib $(wildcard sections/*.tex) $(wildcard tables/*.tex)
	$(TEX) $(FLAGS) $(MAIN)
	-$(BIBTEX) $(MAIN)
	$(TEX) $(FLAGS) $(MAIN)
	$(TEX) $(FLAGS) $(MAIN)
	@echo
	@echo "Built $(MAIN).pdf. Check the log for unresolved references:"
	@grep -c "LaTeX Warning: Reference" $(MAIN).log || true

pdflatex:
	$(MAKE) TEX=pdflatex all

tables:
	cd .. && python3 scripts/make_tables.py results paper/tables

# Structural sanity check that needs no LaTeX at all.
check:
	cd .. && python3 scripts/check_paper.py

clean:
	rm -f *.aux *.log *.out *.toc *.bbl *.blg *.lof *.lot *.fls \
	      *.fdb_latexmk *.synctex.gz sections/*.aux

distclean: clean
	rm -f $(MAIN).pdf
