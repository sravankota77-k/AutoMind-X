"""Random-search baseline and an equal-budget comparison against AutoMind-X.

Question answered: *does the NSGA-II + self-adaptive search find a better Pareto front than simply
sampling architectures at random, given the same number of trained networks?*

Fairness rules
--------------
* Same dataset, same search space, same training budget (``max_iter``) and same objectives.
* Random search receives exactly as many **unique trainings** as AutoMind-X used.
* Both are scored on the *validation* objectives with the same hypervolume routine and reference
  point, so the numbers are directly comparable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import Config
from .dataset import Dataset, make_dataset
from .evaluator import Evaluator, normalised_objectives
from .evolution import AutoMindX
from .genome import Genome, random_genome
from .pareto import hypervolume_mc, non_dominated_sort


@dataclass
class SearchOutcome:
    """Summary of one search run, comparable across methods."""

    method: str
    n_trainings: int
    hypervolume: float  # of the non-dominated front over everything evaluated (higher is better)
    best_fitness: float
    best_accuracy: float
    front_size: int
    archive: list = field(default_factory=list)


def _summarise(method: str, archive: list[Genome], evaluator: Evaluator, cfg: Config) -> SearchOutcome:
    """Reduce an archive of evaluated genomes to the comparable summary statistics."""
    objs = [g.objectives() for g in archive]
    front = [archive[i] for i in non_dominated_sort(objs)[0]]
    hv = hypervolume_mc([normalised_objectives(g.accuracy, g.runtime, g.params, cfg) for g in front])
    return SearchOutcome(
        method=method,
        n_trainings=evaluator.n_trainings,
        hypervolume=hv,
        best_fitness=max(g.fitness for g in archive),
        best_accuracy=max(g.accuracy for g in archive),
        front_size=len(front),
        archive=archive,
    )


def random_search(cfg: Config, n_trainings: int, data: Dataset | None = None) -> SearchOutcome:
    """Sample ``n_trainings`` *unique* random architectures and evaluate them.

    Uses the same ``random_genome`` sampler that initialises AutoMind-X, so the only difference
    between the two methods is the search strategy.
    """
    rng = np.random.default_rng(cfg.seed + 1)  # +1: independent stream from the evolutionary run
    ev = Evaluator(data if data is not None else make_dataset(cfg), cfg)
    seen: dict = {}
    attempts = 0
    while len(seen) < n_trainings and attempts < n_trainings * 50:
        g = random_genome(cfg, rng)
        seen.setdefault(g.key(), g)
        attempts += 1
    batch = list(seen.values())
    ev.evaluate_many(batch)
    return _summarise("random", batch, ev, cfg)


def evolutionary_outcome(cfg: Config, data: Dataset | None = None) -> SearchOutcome:
    """Run AutoMind-X and summarise it the same way as :func:`random_search`."""
    result = AutoMindX(cfg, data=data, verbose=False).run()
    return _summarise("automind-x", result.archive, result.evaluator, cfg)


@dataclass
class Comparison:
    """Paired, equal-budget comparison over several seeds."""

    seeds: list[int]
    evo: list[SearchOutcome]
    rnd: list[SearchOutcome]

    def diffs(self, attr: str) -> np.ndarray:
        """Per-seed ``AutoMind-X - random`` differences of one outcome attribute."""
        return np.array([getattr(e, attr) - getattr(r, attr) for e, r in zip(self.evo, self.rnd, strict=True)])

    def summary(self, attr: str = "hypervolume", n_boot: int = 4000, seed: int = 0) -> dict:
        """Mean paired difference, 95% bootstrap CI, and win/tie/loss counts for ``attr``."""
        d = self.diffs(attr)
        boot = np.random.default_rng(seed).choice(d, size=(n_boot, len(d)), replace=True).mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        return dict(
            metric=attr,
            mean_evo=float(np.mean([getattr(e, attr) for e in self.evo])),
            mean_random=float(np.mean([getattr(r, attr) for r in self.rnd])),
            mean_diff=float(d.mean()),
            ci95=(float(lo), float(hi)),
            wins=int((d > 1e-9).sum()),
            ties=int((np.abs(d) <= 1e-9).sum()),
            losses=int((d < -1e-9).sum()),
            significant=bool(lo > 0 or hi < 0),
        )


def compare_with_random(cfg: Config, seeds: list[int]) -> Comparison:
    """Run AutoMind-X and random search for every seed with an identical training budget."""
    evo, rnd = [], []
    for s in seeds:
        c = Config(**{**cfg.__dict__, "seed": s})
        data = make_dataset(c)
        e = evolutionary_outcome(c, data)
        evo.append(e)
        rnd.append(random_search(c, e.n_trainings, data))
    return Comparison(seeds=list(seeds), evo=evo, rnd=rnd)


def format_comparison(cmp: Comparison) -> str:
    """Plain-language table of the comparison (used by the CLI and the benchmark script)."""
    lines = [
        f"EQUAL-BUDGET COMPARISON vs RANDOM SEARCH  ({len(cmp.seeds)} seeds, "
        f"~{int(np.mean([e.n_trainings for e in cmp.evo]))} trainings per method)",
        f"  {'metric':<14}{'AutoMind-X':>12}{'random':>10}{'mean diff':>12}   95% CI            W/T/L",
    ]
    for attr in ("hypervolume", "best_fitness", "best_accuracy"):
        s = cmp.summary(attr)
        lines.append(
            f"  {attr:<14}{s['mean_evo']:>12.4f}{s['mean_random']:>10.4f}{s['mean_diff']:>+12.4f}"
            f"   [{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}]  {s['wins']}/{s['ties']}/{s['losses']}"
        )
    hv = cmp.summary("hypervolume")
    verdict = (
        "AutoMind-X is significantly better"
        if hv["significant"] and hv["mean_diff"] > 0
        else "random search is significantly better"
        if hv["significant"]
        else "no statistically clear difference"
    )
    lines.append(f"  -> hypervolume verdict: {verdict} (paired bootstrap, CI excludes 0 = significant)")
    return "\n".join(lines)
