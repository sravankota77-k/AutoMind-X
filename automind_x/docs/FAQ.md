# FAQ and troubleshooting

**Is a run reproducible?** Yes with the default `proxy` cost: the same `--seed` gives the identical
search (a test verifies it). With `--runtime-metric wallclock` timings vary between runs.

**`--n-jobs -1` is not faster on my machine.** Process start-up costs more than parallelism saves
for small batches (below 4 networks) or on a single core. Use `--n-jobs 1`, or run
`python benchmarks/bench.py` to measure your machine.

**The budget is "NOT satisfiable".** No architecture in the search space fits `--max-params` /
`--max-cost`. The report then shows the least-violating model. Loosen the budget or enlarge
`Config.neuron_choices`.

**Second run trained nothing.** That is the `--cache` file working: measurements are reused when the
data and training settings are identical (`fingerprint` in `evaluator.py`). Delete the file to reset.

**`error: Invalid Config: ...`** A value is out of range. The message names the field. Exit code is 2.

**Plots fail on a server.** Matplotlib is forced to the headless `Agg` backend, so no display is needed.

**Is my data used or uploaded?** No. The dataset is generated locally and nothing leaves the machine.

**Why is the chosen model less accurate than the default MLP?** By design AutoMind-X trades a little
accuracy for a much smaller, cheaper model. The hold-out section states the exact trade.
