# Changelog

## 1.2.0
### Added
- **Equal-budget random-search baseline** (`automind/baselines.py`, `--compare-random N`) with a paired
  multi-seed comparison and bootstrap confidence interval - measures whether the search really beats
  random sampling instead of assuming it.
- **Inference-latency profile** (batch ms/1000 samples, single-sample p50/p95) in the report and `summary.json`.
- **Profiling**: `--profile` writes a cProfile summary; `summary.json` records training-time share.
- **Persistent measurement cache** (`--cache FILE`): repeat runs skip retraining (settings-fingerprinted JSON).
- **Environment configuration**: `Config.from_env()` and `.env.example` (`AUTOMIND_*`, no secrets).
- Docs: `docs/` (architecture, usage, FAQ, SDG 9 impact, evaluation), `examples/`, `CODE_OF_CONDUCT.md`,
  `benchmarks/RESULTS.md`, pre-commit config, secret-scan job in CI, docstring gate (`interrogate`).
### Changed
- 100% docstring coverage; `explain()` split into one helper per report section (identical output).
- Benchmark script measures cold vs warm cache, sort scaling and latency; ruff rule-set widened.
- `.gitignore` now keeps `.env.example` tracked.

## 1.1.0
### Added
- Deterministic FLOPs-based runtime proxy (`--runtime-metric proxy`, default) -> reproducible runs.
- Parallel, deduplicated batch evaluation of offspring (`--n-jobs`), with serial fallback.
- Knee-point selection (`--select knee`) and deployment budgets (`--max-params`, `--max-cost`).
- Accessible `report.html`, `ALT_TEXT.md`, colour-blind-safe plots, `summary.json` (`--json`).
- Config validation, argparse validation, exit codes.
- Property-based tests (Hypothesis), CLI/end-to-end tests (59 tests, ~99% coverage).
- `pyproject.toml` (ruff, mypy, pytest), CI (lint, types, security scan, tests), Dockerfile, Dependabot.
### Changed
- NumPy-vectorised dominance computation; memoised hold-out/baseline training.
- `Genome` uses identity equality; logging replaces `print`.

## 1.0.0
- Initial NSGA-II + self-adaptive mutation architecture search with explanation report.
