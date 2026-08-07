install:
	uv venv --python 3.10
	uv add flake8 mypy fire tqdm
	uv sync

run:
	@clear
	@uv run -m src

debug:
	uv run -m pdb -m src

clean:
	rm -rf __pycache__ .mypy_cache *.pyc src/__pycache__

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
