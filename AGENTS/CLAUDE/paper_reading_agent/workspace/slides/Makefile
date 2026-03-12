TEX = slides
LATEX = pdflatex -interaction=nonstopmode

.PHONY: all clean

all: $(TEX).pdf

$(TEX).pdf: $(TEX).tex refs.bib
	$(LATEX) $(TEX)
	bibtex $(TEX) || true
	$(LATEX) $(TEX)
	$(LATEX) $(TEX)

clean:
	rm -f $(TEX).pdf $(TEX).aux $(TEX).bbl $(TEX).blg $(TEX).log $(TEX).out $(TEX).toc $(TEX).nav $(TEX).snm
