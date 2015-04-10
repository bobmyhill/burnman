# ./Makefile

# ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###

ECHOCMD:=/bin/echo -e
PDFLATEX:=pdflatex
BIBTEX:=bibtex
# ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###

TARGET:=MSH

# ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# 

default: main

main:
	@$(PDFLATEX) $(TARGET)
	@$(BIBTEX) $(TARGET)
	@$(PDFLATEX) $(TARGET)
	@$(PDFLATEX) $(TARGET)

.PHONY: clean

clean:
	@rm -f $(TARGET).aux \
	       $(TARGET).log \
	       $(TARGET).nav \
	       $(TARGET).out \
	       $(TARGET).snm \
	       $(TARGET).toc \
	       $(TARGET).vrb \
	       $(TARGET).pdf \
	       $(TARGET).dvi \
	       $(TARGET).ps  \
	       $(TARGET).bbl \
	       $(TARGET).blg \
	       $(TARGET).thm \
	       $(TARGET).brf \
	       missfont.log  \
	       x.log
	@rm -f *~
