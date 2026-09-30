# Contributing

Thank you for helping! Please read the [Code of Conduct](CODE_OF_CONDUCT.md) first.

## Set up

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"      # or: pip install -r requirements-dev.txt
pre-commit install           # optional: runs lint, types and secret checks on every commit
cp .env.example .env         # optional: local tuning knobs (never commit .env)
```

## Before opening a pull request

```bash
make check    # ruff + mypy + bandit + docstring gate + tests with coverage (same as CI)
```

## Standards

* **Typed and documented.** Every module, class and function has a docstring (`interrogate`
  enforces 95%+); public functions have type hints (`mypy` is clean).
* **Small functions.** Prefer helpers over long bodies (see how `explain.py` builds one section per helper).
* **Test every behaviour change.** Unit tests for logic, property tests for invariants, a CLI test for
  new flags. Coverage must stay at or above 90%.
* **Claims need evidence.** If you say something is faster or better, add a benchmark
  (`benchmarks/bench.py`) or a comparison (`--compare-random`) and report the honest result.
* **No secrets.** Configuration goes in `AUTOMIND_*` variables documented in `.env.example`.

## Extending the search space

Add a gene in `genome.py`, its choices in `config.py`, a mutation operator in `operators.py`, and make
sure `test_operators_keep_genomes_valid` still passes. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Reporting bugs

Open an issue with the command you ran, the `--seed`, your Python version and the full error text.
Security problems: see [SECURITY.md](SECURITY.md).
