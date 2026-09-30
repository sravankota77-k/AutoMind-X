"""Reproducible micro-benchmarks for the efficiency claims in the README.

    python benchmarks/bench.py                # print a table
    python benchmarks/bench.py --write        # also write benchmarks/RESULTS.md

Measured (medians of several repeats, see ``REPEATS``):

1. serial vs parallel batch training (depends on core count; equal on a single core)
2. cold vs warm persistent cache (the big win: a repeat run trains nothing)
3. non-dominated sort scaling (vectorised dominance matrix)
4. inference latency of the default MLP vs a small evolved-size network
"""

from __future__ import annotations

import argparse
import os
import platform
import statistics
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from automind.config import Config
from automind.dataset import make_dataset
from automind.evaluator import Evaluator
from automind.evolution import AutoMindX
from automind.genome import Genome, random_genome
from automind.pareto import non_dominated_sort

REPEATS = 3


def _median(fn, repeats: int = REPEATS) -> float:
    """Median wall time (seconds) of ``fn()`` over ``repeats`` calls."""
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def time_batch(n_jobs: int, n: int = 16) -> float:
    """Seconds to train ``n`` random networks with ``n_jobs`` workers."""
    cfg = Config(n_jobs=n_jobs, max_iter=40, n_samples=1500)
    data = make_dataset(cfg)
    rng = np.random.default_rng(0)
    genomes = [random_genome(cfg, rng) for _ in range(n)]

    def run() -> None:
        Evaluator(data, cfg).evaluate_many([g.clone() for g in genomes])

    return _median(run, repeats=2)


def time_sort(n: int = 400) -> float:
    """Seconds to non-dominated-sort ``n`` random 3-objective points."""
    objs = np.random.default_rng(0).random((n, 3)).tolist()
    return _median(lambda: non_dominated_sort(objs), repeats=5)


def time_cache() -> tuple[float, float, int]:
    """Return (cold seconds, warm seconds, trainings avoided) for one quick search."""
    cfg = Config(pop_size=8, generations=5, n_samples=1000, max_iter=40, n_jobs=1)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "cache.json")
        t0 = time.perf_counter()
        cold = AutoMindX(cfg, verbose=False, cache_path=path).run()
        cold_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        warm = AutoMindX(cfg, verbose=False, cache_path=path).run()
        warm_s = time.perf_counter() - t0
    return cold_s, warm_s, cold.evaluator.n_trainings - warm.evaluator.n_trainings


def latency_table() -> list[tuple[str, int, float, float, float]]:
    """(name, params, batch ms/1k, single p50 ms, single p95 ms) for two reference networks."""
    cfg = Config(max_iter=40, n_samples=1500)
    ev = Evaluator(make_dataset(cfg), cfg)
    rows = []
    for name, layers in (("default MLP(100)", [100]), ("compact MLP(16)", [16])):
        g = Genome(layers=layers, activation="relu", lr=0.01, mutation_rate=0.0)
        ev.evaluate(g)
        lat = ev.latency(g)
        rows.append((name, g.params, lat.batch_ms_per_1k, lat.single_p50_ms, lat.single_p95_ms))
    return rows


def build_report() -> str:
    """Run every benchmark and return a Markdown report."""
    serial, parallel = time_batch(1), time_batch(-1)
    cold, warm, avoided = time_cache()
    lines = [
        "# Benchmark results",
        "",
        f"Machine: {platform.platform()}, Python {platform.python_version()}, {os.cpu_count()} CPU core(s).",
        "Numbers are machine specific; re-run `python benchmarks/bench.py --write` to refresh them.",
        "",
        "| Benchmark | Result |",
        "|---|---|",
        f"| Train 16 networks, serial | {serial:.2f} s |",
        f"| Train 16 networks, `n_jobs=-1` | {parallel:.2f} s ({serial / parallel:.2f}x) |",
        f"| Quick search, cold cache | {cold:.2f} s |",
        f"| Quick search, warm cache | {warm:.2f} s ({cold / warm:.0f}x faster, {avoided} trainings avoided) |",
        f"| Non-dominated sort, 400 points | {time_sort() * 1000:.1f} ms |",
        f"| Non-dominated sort, 1600 points | {time_sort(1600) * 1000:.1f} ms |",
        "",
        "## Inference latency",
        "",
        "| Network | Parameters | Batch (ms / 1000 samples) | Single p50 (ms) | Single p95 (ms) |",
        "|---|---|---|---|---|",
    ]
    for name, params, batch, p50, p95 in latency_table():
        lines.append(f"| {name} | {params} | {batch:.2f} | {p50:.3f} | {p95:.3f} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    """Print the report; with ``--write`` also save it next to this script."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--write", action="store_true", help="write benchmarks/RESULTS.md")
    args = ap.parse_args()
    report = build_report()
    print(report)
    if args.write:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "RESULTS.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
