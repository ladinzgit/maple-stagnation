.PHONY: install reproduce h1 h2 h3

install:
	pip install -r requirements.txt

reproduce: h1 h2 h3
	@echo "All analyses complete."

h1:
	jupyter nbconvert --to notebook --execute --inplace h1_clustering/feature_selection.ipynb
	jupyter nbconvert --to notebook --execute --inplace h1_clustering/h1_clustering.ipynb

h2:
	python h2_distribution/run_analysis.py
	jupyter nbconvert --to notebook --execute --inplace h2_distribution/h2_distribution.ipynb

h3:
	jupyter nbconvert --to notebook --execute --inplace h3_rule/h3_rule.ipynb
