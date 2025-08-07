.PHONY: clean purge test lint check

clean:
	rm -rf conversations/*.json

purge: clean
	rm -rf models/*.gguf

test:
	pytest

lint:
	ruff format && ruff check

check: lint test
