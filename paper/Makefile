MAIN = main

.PHONY: all figures clean

all: figures
	latexmk -pdf -interaction=nonstopmode $(MAIN).tex || \
	  (pdflatex -interaction=nonstopmode $(MAIN) && bibtex $(MAIN) && \
	   pdflatex -interaction=nonstopmode $(MAIN) && \
	   pdflatex -interaction=nonstopmode $(MAIN))

figures:
	mkdir -p figures
	-cp ../results/transfer_heatmap.png ../results/rsa.png figures/ 2>/dev/null

clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out \
	      $(MAIN).fls $(MAIN).fdb_latexmk
