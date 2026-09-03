LATEXMK ?= latexmk
BUILD_DIR := build
ARXIV_DIR := $(BUILD_DIR)/arxiv-source
ARXIV_ARCHIVE := $(BUILD_DIR)/arxiv-source.tar.gz

.PHONY: pdf arxiv-source clean

pdf:
	mkdir -p $(BUILD_DIR)
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error -output-directory=$(BUILD_DIR) main.tex
	cp $(BUILD_DIR)/main.pdf paper.pdf

arxiv-source: pdf
	rm -rf $(ARXIV_DIR)
	mkdir -p $(ARXIV_DIR)
	cp main.tex references.bib $(BUILD_DIR)/main.bbl $(ARXIV_DIR)/
	cp -R sections tables figures $(ARXIV_DIR)/
	COPYFILE_DISABLE=1 tar -czf $(ARXIV_ARCHIVE) -C $(ARXIV_DIR) .
	@echo "Created $(ARXIV_ARCHIVE)"

clean:
	$(LATEXMK) -C -output-directory=$(BUILD_DIR) main.tex
	rm -rf $(ARXIV_DIR) $(ARXIV_ARCHIVE)
