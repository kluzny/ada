.PHONY: clean purge test lint format fix check

clean:
	rm -rf conversations/*.json
	rm -rf logs/*.log

purge: clean
	rm -rf models/*.gguf

test:
	pytest

format:
	ruff format

fix:
	ruff check --fix

lint: format fix

check: lint test
