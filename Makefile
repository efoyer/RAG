install:
	uv sync

run:
	uv run python -m src

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
