# Notebooks reading private (gitignored) datasets — runnable locally,
# impossible on CI runners.
PRIVATE_NOTEBOOKS := examples/04_eco_pack_presov.ipynb
PUBLIC_NOTEBOOKS := $(filter-out $(PRIVATE_NOTEBOOKS),$(wildcard examples/*.ipynb))

format:
	uv run pre-commit run --all-files

execute-notebooks:
	uv run jupyter nbconvert --execute --to notebook --inplace examples/*.ipynb --ExecutePreprocessor.timeout=-1

# Execution check used by CI on PRs to main: runs every notebook whose
# data ships with the repo, writing results to a temp dir so the
# worktree is never modified.
verify-notebooks:
	uv run jupyter nbconvert --execute --to notebook $(PUBLIC_NOTEBOOKS) --output-dir="$$(mktemp -d)" --ExecutePreprocessor.timeout=-1

render-notebooks:
	uv run jupyter nbconvert --to markdown examples/*.ipynb
