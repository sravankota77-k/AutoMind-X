"""AutoMind-X entry point.

python main.py --quick       10-second demo
python main.py               full run
python main.py --help        all options
"""

from __future__ import annotations

import argparse
import cProfile
import csv
import io
import json
import logging
import os
import pstats
import sys
import time
from dataclasses import asdict

from automind.baselines import compare_with_random, format_comparison
from automind.config import RUNTIME_METRICS, SELECTION_STRATEGIES, Config
from automind.evolution import AutoMindX, Result
from automind.explain import explain, holdout_report, select_final
from automind.genome import Genome
from automind.report import write_html
from automind.visualize import make_all


def _positive_int(text: str) -> int:
    """Argparse type: parse an integer that must be >= 1."""
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def _positive_seeds(text: str) -> int:
    """Argparse type: number of seeds for ``--compare-random`` (1..50)."""
    value = _positive_int(text)
    if value > 50:
        raise argparse.ArgumentTypeError("at most 50 seeds")
    return value


def _safe_outdir(text: str) -> str:
    """Resolve the output directory and refuse anything that is an existing non-directory."""
    path = os.path.abspath(os.path.expanduser(text))
    if os.path.exists(path) and not os.path.isdir(path):
        raise argparse.ArgumentTypeError(f"{text!r} exists and is not a directory")
    return path


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface (see ``python main.py --help``)."""
    ap = argparse.ArgumentParser(
        description="AutoMind-X: explainable, self-adaptive, multi-objective neural architecture search"
    )
    ap.add_argument("--quick", action="store_true", help="small fast demo")
    ap.add_argument("--generations", type=_positive_int, help="number of generations")
    ap.add_argument("--pop", type=_positive_int, help="population size (>= 2)")
    ap.add_argument("--seed", type=int, help="random seed (default 42)")
    ap.add_argument("--outdir", type=_safe_outdir, help="output directory (default: $AUTOMIND_OUTDIR or 'outputs')")
    ap.add_argument(
        "--runtime-metric",
        choices=RUNTIME_METRICS,
        help="'proxy' = deterministic FLOPs estimate (default), 'wallclock' = measured seconds",
    )
    ap.add_argument(
        "--select",
        choices=SELECTION_STRATEGIES,
        dest="selection",
        help="final model choice: 'balanced' weighted fitness or 'knee' ideal-point distance",
    )
    ap.add_argument("--max-params", type=_positive_int, help="deployment budget: max model parameters")
    ap.add_argument("--max-cost", type=float, help="deployment budget: max training cost (pseudo-seconds)")
    ap.add_argument("--n-jobs", type=int, help="parallel workers for training (-1 = all cores, 1 = serial)")
    ap.add_argument("--cache", metavar="FILE", help="persist measurements (proxy metric) so repeat runs skip training")
    ap.add_argument(
        "--compare-random",
        type=_positive_seeds,
        metavar="SEEDS",
        help="also run an equal-budget random-search baseline over SEEDS seeds and report the paired difference",
    )
    ap.add_argument("--profile", action="store_true", help="write a cProfile summary to <outdir>/profile.txt")
    ap.add_argument("--no-plots", action="store_true", help="skip figure generation")
    ap.add_argument("--json", action="store_true", help="also write summary.json (machine-readable)")
    ap.add_argument("--quiet", action="store_true", help="suppress per-generation progress")
    return ap


def build_config(a: argparse.Namespace) -> Config:
    """Merge defaults < ``AUTOMIND_*`` environment < CLI flags into one validated Config."""
    cfg = Config.from_env(**({"outdir": a.outdir} if a.outdir else {}))
    if a.quick:
        cfg.pop_size, cfg.generations, cfg.n_samples, cfg.max_iter = 8, 5, 1000, 40
    for attr, val in (
        ("generations", a.generations),
        ("pop_size", a.pop),
        ("seed", a.seed),
        ("runtime_metric", a.runtime_metric),
        ("selection", a.selection),
        ("n_jobs", a.n_jobs),
        ("max_params", a.max_params),
        ("max_cost", a.max_cost),
    ):
        if val is not None:
            setattr(cfg, attr, val)
    cfg.validate()
    return cfg


def write_outputs(
    result: Result,
    cfg: Config,
    report: str,
    chosen: Genome,
    hold: tuple,
    write_json: bool,
    total_seconds: float = 0.0,
) -> None:
    """Write the text report, the architecture CSV and (optionally) summary.json."""
    os.makedirs(cfg.outdir, exist_ok=True)
    with open(os.path.join(cfg.outdir, "explanation.txt"), "w", encoding="utf-8") as f:
        f.write(report)
    with open(os.path.join(cfg.outdir, "all_architectures.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "uid",
                "generation",
                "layers",
                "activation",
                "lr",
                "accuracy",
                "runtime_s",
                "wall_time_s",
                "params",
                "fitness",
            ]
        )
        for g in result.archive:
            w.writerow(
                [
                    g.uid,
                    g.birth_gen,
                    "-".join(map(str, g.layers)),
                    g.activation,
                    g.lr,
                    f"{g.accuracy:.4f}",
                    f"{g.runtime:.4f}",
                    f"{g.wall_time:.4f}",
                    g.params,
                    f"{g.fitness:.4f}",
                ]
            )
    if write_json:
        test_acc, test_rt, test_params = hold
        summary = dict(
            seed=cfg.seed,
            budget=dict(max_params=cfg.max_params, max_cost=cfg.max_cost),
            runtime_metric=cfg.runtime_metric,
            selection=cfg.selection,
            selected=dict(
                layers=chosen.layers,
                activation=chosen.activation,
                lr=chosen.lr,
                val_accuracy=chosen.accuracy,
                params=chosen.params,
                found_in_generation=chosen.birth_gen,
            ),
            holdout=dict(test_accuracy=test_acc, cost_s=test_rt, params=test_params),
            latency=asdict(result.evaluator.latency(chosen)),
            timing=dict(
                total_s=round(total_seconds, 3),
                training_s=round(result.evaluator.train_seconds, 3),
                training_share=round(result.evaluator.train_seconds / total_seconds, 3) if total_seconds else None,
            ),
            trainings=result.evaluator.n_trainings,
            cache_hits=result.evaluator.cache_hits,
            disk_cache_hits=result.evaluator.disk_hits,
            final_hypervolume=result.history[-1]["hypervolume"],
        )
        with open(os.path.join(cfg.outdir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)


def main(argv: list[str] | None = None) -> int:
    """Run the CLI: search, select, explain, write outputs. Returns the process exit code (0 ok, 2 invalid input)."""
    a = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING if a.quiet else logging.INFO, format="%(message)s", stream=sys.stdout)
    try:
        cfg = build_config(a)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    profiler = cProfile.Profile() if a.profile else None
    t0 = time.time()
    if profiler:
        profiler.enable()
    result = AutoMindX(cfg, verbose=not a.quiet, cache_path=a.cache).run()
    if profiler:
        profiler.disable()
    search_seconds = time.time() - t0
    chosen = select_final(result)[0]
    hold = result.evaluator.holdout(chosen)
    report = explain(result) + "\n" + holdout_report(result, chosen)
    if a.compare_random:
        report += "\n\n" + format_comparison(compare_with_random(cfg, [cfg.seed + i for i in range(a.compare_random)]))
    write_outputs(result, cfg, report, chosen, hold, a.json, search_seconds)
    if profiler:
        _write_profile(profiler, cfg.outdir)
    paths = [] if a.no_plots else [*make_all(result, cfg.outdir), write_html(result, cfg.outdir)]

    print("\n" + report)
    print(
        f"\nSaved: {', '.join(os.path.basename(p) for p in paths)}{', ' if paths else ''}explanation.txt, "
        f"all_architectures.csv -> {cfg.outdir}/   (total {time.time() - t0:.1f}s, "
        f"{result.evaluator.n_trainings} networks trained, {result.evaluator.cache_hits} cache hits)"
    )
    return 0


def _write_profile(profiler: cProfile.Profile, outdir: str, top: int = 25) -> None:
    """Save the ``top`` most expensive calls (by cumulative time) to ``profile.txt``."""
    os.makedirs(outdir, exist_ok=True)
    buf = io.StringIO()
    pstats.Stats(profiler, stream=buf).sort_stats("cumulative").print_stats(top)
    with open(os.path.join(outdir, "profile.txt"), "w", encoding="utf-8") as f:
        f.write(buf.getvalue())


if __name__ == "__main__":
    sys.exit(main())
