MAINFILE = nips_2019
EXTRA_FILES := $(shell echo "images/*") 

## Inner workings
OBJECTS = $(shell echo *.tex)
STYLES = $(shell echo *.sty)
BIB = $(shell echo *.bib)

OBJECTS_TEST = $(addsuffix .t, $(basename $(OBJECTS)))
STYLES_TEST = $(addsuffix .s, $(basename $(STYLES)))
BIB_TEST = bib
TESTS = $(addprefix make/, $(OBJECTS_TEST) $(STYLES_TEST) $(BIB_TEST))
TEMP2 := $(shell mkdir make 2>/dev/null)

.PHONY: all
all:
	pdflatex $(MAINFILE)
	bibtex $(MAINFILE)
	pdflatex $(MAINFILE)
	pdflatex $(MAINFILE)

.PHONY: clean
clean:
	-rm -f *.aux
	-rm -f *.log
	-rm -f *.toc
	-rm -f *.bbl
	-rm -f *.blg
	-rm -f *.out
	-rm -f make/bib

.PHONY: cleanall
cleanall: clean
	-rm -f *.pdf
	-rm -f *.ps
	-rm -f *.dvi
	-rm -rf ./make
