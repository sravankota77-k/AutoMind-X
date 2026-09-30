.PHONY: check lint types security docs test demo bench compare profile
check: lint types security docs test
lint:
	ruff check . && ruff format --check .
types:
	mypy automind main.py
security:
	bandit -q -r automind main.py -c pyproject.toml
docs:
	interrogate -q automind main.py benchmarks examples
test:
	python -m pytest --cov=automind
demo:
	python main.py --quick --json
compare:
	python main.py --quick --compare-random 5 --no-plots
profile:
	python main.py --quick --profile --no-plots
bench:
	python benchmarks/bench.py --write
