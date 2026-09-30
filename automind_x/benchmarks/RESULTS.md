# Benchmark results

Machine: Linux-6.18.44-fc-v50-x86_64-with-glibc2.39, Python 3.12.3, 1 CPU core(s).
Numbers are machine specific; re-run `python benchmarks/bench.py --write` to refresh them.

| Benchmark | Result |
|---|---|
| Train 16 networks, serial | 2.70 s |
| Train 16 networks, `n_jobs=-1` | 2.54 s (1.06x) |
| Quick search, cold cache | 4.33 s |
| Quick search, warm cache | 0.05 s (79x faster, 48 trainings avoided) |
| Non-dominated sort, 400 points | 14.5 ms |
| Non-dominated sort, 1600 points | 197.6 ms |

## Inference latency

| Network | Parameters | Batch (ms / 1000 samples) | Single p50 (ms) | Single p95 (ms) |
|---|---|---|---|---|
| default MLP(100) | 2403 | 1.69 | 0.266 | 0.417 |
| compact MLP(16) | 387 | 1.45 | 0.355 | 0.493 |
