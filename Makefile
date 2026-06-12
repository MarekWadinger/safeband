format:
	uv run pre-commit run --all-files

execute-notebooks:
	uv run jupyter nbconvert --execute --to notebook --inplace examples/*.ipynb --ExecutePreprocessor.timeout=-1

render-notebooks:
	uv run jupyter nbconvert --to markdown examples/*.ipynb
