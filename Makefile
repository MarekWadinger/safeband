format:
	uv run pre-commit run --all-files

execute-notebooks:
	uv run jupyter nbconvert --execute --to notebook --inplace examples/*.ipynb --ExecutePreprocessor.timeout=-1

# Execution check for the pre-push gate: runs every example notebook but
# writes results to a temp dir, so the worktree is never modified.
verify-notebooks:
	uv run jupyter nbconvert --execute --to notebook examples/*.ipynb --output-dir="$$(mktemp -d)" --ExecutePreprocessor.timeout=-1

render-notebooks:
	uv run jupyter nbconvert --to markdown examples/*.ipynb
