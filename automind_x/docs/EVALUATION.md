# Evaluation and evidence

Everything below was produced by commands in this repository; numbers are from the run committed in
`outputs/` and `benchmarks/`. Nothing here is tuned to look better than it is.

## 1. Held-out test result (`python main.py --json`, seed 42)

| | AutoMind-X | Default MLP(100) | Ratio |
|---|---|---|---|
| Architecture | 8 / relu / lr 0.03 | 100 / relu / lr 0.001 | |
| Parameters | 195 | 2403 | **12.3x smaller** |
| Estimated training cost | 33 ms | 1060 ms | **31.8x cheaper** |
| Test accuracy | 0.794 | 0.814 | -0.020 (2 points lower) |

The test split is never seen during the search; the final model is retrained and scored on it.

## 2. Is it better than random search? (`--compare-random`, equal training budget)

Random search receives exactly as many unique trainings as AutoMind-X (182). Paired over 8 seeds,
default configuration (`benchmarks/random_search_comparison.txt`):

| Metric | AutoMind-X | Random | Mean diff | 95% bootstrap CI | Wins / ties / losses |
|---|---|---|---|---|---|
| Hypervolume (front quality) | 0.7605 | 0.7442 | **+0.0163** | [+0.0027, +0.0325] | 6 / 0 / 2 |
| Best fitness | 0.4399 | 0.4384 | +0.0015 | [-0.0049, +0.0078] | 3 / 2 / 3 |
| Best accuracy | 0.8665 | 0.8622 | +0.0043 | [-0.0015, +0.0103] | 6 / 0 / 2 |

**Reading it honestly.** AutoMind-X finds a *significantly better Pareto front* (hypervolume) than
random search at the default budget, but the effect is modest and the confidence interval only just
excludes zero. It does **not** find a significantly better single best model. At a very small budget
(quick config, about 48 trainings, 6 seeds) there was no clear difference (hypervolume -0.011,
CI [-0.050, +0.035]). Interpretation: evolutionary guidance pays off once the budget is large enough
for selection pressure to matter. Eight seeds is a small sample; use `--compare-random 30` to tighten it.

## 3. Efficiency (`python benchmarks/bench.py --write`, 1 CPU core)

| Measurement | Result |
|---|---|
| Profile (`--profile`) | training is **99.7%** of run time -> avoiding retraining is the main lever |
| Repeat run with `--cache` | 4.3 s -> 0.05 s (**~80x faster**, 48 trainings avoided) |
| Non-dominated sort, 400 / 1600 points | 33 -> 14 ms and 378 -> 198 ms after vectorising the inner loop (identical output, tested) |
| Parallel training (`--n-jobs -1`) | 1.0x on this single-core machine; expected to scale with cores |
| Identical architectures | trained once (memoised), duplicates within a batch removed |

### Inference latency (honest note)

The compact model is far smaller, but on this CPU its **single-sample latency is not lower** than the
default MLP's (p50 about 0.27-0.35 ms for both): scikit-learn's per-call overhead dominates at this
size. Batch throughput is slightly better (about 1.0-1.5 vs 1.2-1.9 ms per 1000 samples). The real
gains of the small model are memory, parameters and training cost; latency gains would need a leaner
inference runtime (for example exporting weights to NumPy or ONNX).

## 4. Code and test quality

| Check | Result |
|---|---|
| Tests | 85 passing (unit, property-based, determinism, CLI, end-to-end) |
| Coverage | 99% (gate: 90%) |
| ruff / mypy / bandit | clean |
| Docstring coverage (`interrogate`) | 100% (gate: 95%) |
| Reproducibility | same seed -> identical search (tested); parallel == serial (tested) |

## 5. Limitations

Synthetic tabular data and a plain MLP search space; small CPU networks; results for other datasets,
architectures (CNNs, transformers) or GPUs are not claimed. The cost objective is a FLOPs proxy by
default (reproducible) rather than measured time.
