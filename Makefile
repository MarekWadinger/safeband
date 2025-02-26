format:
	pre-commit run --all-files

execute-notebooks:
	jupyter nbconvert --execute --to notebook --inplace examples/*.ipynb --ExecutePreprocessor.timeout=-1

render-notebooks:
	jupyter nbconvert --to markdown examples/*.ipynb
