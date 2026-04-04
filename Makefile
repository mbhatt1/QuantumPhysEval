.PHONY: help install pooled paper

help:
	@echo "install  - pip install -e ."
	@echo "pooled   - build pooled GPT-4/GPT-5 snapshot artifacts"
	@echo "paper    - build paper/QuantumPhysEval.pdf"

install:
	pip install -e .

pooled:
	python scripts/build_pooled_snapshot.py --results-json artifacts/results/gpt4_family_results.json --results-json artifacts/results/gpt5_family_results.json

paper:
	cd paper && latexmk -pdf -interaction=nonstopmode QuantumPhysEval.tex

