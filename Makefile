# Notebooks excluded from the CI execution gate:
#  - 04 reads a private (gitignored) dataset.
#  - 05_eval only plots the benchmark CSVs written by the separate, slow
#    05_scalability.py run; those .results/ outputs are not committed.
#  - comparison / comparison_ARIMA hang on a cell under nbconvert
#    (no running event loop), stalling the CI job; run them locally.
SKIP_NOTEBOOKS := \
	examples/04_eco_pack_presov.ipynb \
	examples/05_scalability_eval.ipynb \
	examples/comparison.ipynb \
	examples/comparison_ARIMA.ipynb
PUBLIC_NOTEBOOKS := $(filter-out $(SKIP_NOTEBOOKS),$(wildcard examples/*.ipynb))

format:
	uv run pre-commit run --all-files

execute-notebooks:
	uv run jupyter nbconvert --execute --to notebook --inplace examples/*.ipynb --ExecutePreprocessor.timeout=-1

# Execution check used by CI on PRs to main: runs every notebook whose
# data ships with the repo (and reliably finishes), writing results to a
# temp dir so the worktree is never modified.
verify-notebooks:
	uv run jupyter nbconvert --execute --to notebook $(PUBLIC_NOTEBOOKS) --output-dir="$$(mktemp -d)" --ExecutePreprocessor.timeout=-1

render-notebooks:
	uv run jupyter nbconvert --to markdown examples/*.ipynb
