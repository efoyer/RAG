SGOINFRE_DIR = /sgoinfre/goinfre/Perso/efoyer/RAG
VENV_DIR = $(SGOINFRE_DIR)/.venv

install:
	@rm -rf .venv
	@mkdir -p $(VENV_DIR)
	@ln -sfn $(VENV_DIR) .venv
	uv sync

run:
	uv run python -m src

debug:
	uv run -m pdb -m src

clean:
	rm -rf __pycache__ .mypy_cache *.pyc src/__pycache__
	rm -rf .venv $(VENV_DIR)

lint:
	python3 -m flake8 src/ || status=$$?; \
	python3 -m mypy src/ --exclude=venv \
		--explicit-package-bases \
		--warn-return-any \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--check-untyped-defs \
		--follow-imports=skip
