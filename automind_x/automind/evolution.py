"""The evolution loop that ties everything together."""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field

import numpy as np

from .config import Config
from .dataset import Dataset, make_dataset
from .evaluator import Evaluator, normalised_objectives
from .genome import random_genome
from .meta import MetaController
from .operators import crossover, mutate, tournament_select
from .pareto import environmental_selection, hypervolume_mc

log = logging.getLogger(__name__)


@dataclass
class Result:
    """Everything a finished run produced: history, archive, lineage registry, final population."""

    cfg: Config
    data: Dataset
    evaluator: Evaluator
    meta: MetaController
    history: list = field(default_factory=list)  # one dict of statistics per generation
    gen_best: dict = field(default_factory=dict)  # generation -> best-fitness genome
    archive: list = field(default_factory=list)  # every unique architecture ever evaluated
    registry: dict = field(default_factory=dict)  # uid -> genome (for lineage tracing)
    final_population: list = field(default_factory=list)


class AutoMindX:
    """Self-adaptive NSGA-II architecture search (selection, crossover, mutation, meta-evolution)."""

    def __init__(self, cfg, data=None, verbose: bool = True, cache_path: str | None = None):
        """Set up data, evaluator and meta-controller. ``cache_path`` enables the persistent cache."""
        self.cfg, self.verbose = cfg, verbose
        self.rng = np.random.default_rng(cfg.seed)
        self.data = data if data is not None else make_dataset(cfg)
        self.evaluator = Evaluator(self.data, cfg, cache_path=cache_path)
        self.meta = MetaController(cfg)
        self._uid = itertools.count()
        self.archive: dict = {}  # architecture key -> first genome that discovered it
        self.registry: dict = {}

    # ------------------------------------------------------------------ helpers
    def _register(self, g, gen, parents=()):
        """Assign a unique id, birth generation and parents to a genome and remember it for lineage tracing."""
        g.uid, g.birth_gen, g.parents = next(self._uid), gen, tuple(parents)
        self.registry[g.uid] = g

    def _evaluate_batch(self, genomes):
        """Train all unseen architectures of a batch (in parallel when allowed) and archive them."""
        self.evaluator.evaluate_many(genomes)
        for g in genomes:
            self.archive.setdefault(g.key(), g)

    @staticmethod
    def _dedup(genomes):
        """Drop genomes whose architecture already appears earlier in the list."""
        seen, out = set(), []
        for g in genomes:
            if g.key() not in seen:
                seen.add(g.key())
                out.append(g)
        return out

    # ------------------------------------------------------------------ 1. population initialisation
    def initialise(self):
        """Create, evaluate and rank the random starting population."""
        pop = [random_genome(self.cfg, self.rng) for _ in range(self.cfg.pop_size)]
        for g in pop:
            self._register(g, 0)
        self._evaluate_batch(pop)
        return environmental_selection(self._dedup(pop), self.cfg.pop_size)

    # ------------------------------------------------------------------ 2-5. selection, crossover, mutation, meta
    def make_offspring(self, pop, gen):
        """Make one generation of offspring: selection, crossover, self-adaptive mutation, batched evaluation."""
        cfg, rng, meta = self.cfg, self.rng, self.meta
        offspring: list = []
        pending: set = set()  # pending = architectures already proposed this generation
        while len(offspring) < cfg.pop_size:
            p1 = tournament_select(pop, cfg.tournament_size, rng)  # selection
            p2 = tournament_select(pop, cfg.tournament_size, rng)
            if rng.random() < cfg.crossover_rate:  # crossover
                child = crossover(p1, p2, cfg, rng)
            else:
                child = p1.clone()
                child.notes = []
            child.mutation_rate = meta.mutate_rate(child.mutation_rate, rng)  # meta: evolve the rate gene
            rate = meta.effective_rate(child.mutation_rate)  # meta: population multiplier
            mutate(child, rate, cfg, rng)  # mutation
            for _ in range(6):  # avoid re-evaluating an architecture we already know
                if child.key() not in self.archive and child.key() not in pending:
                    break
                mutate(child, rate, cfg, rng)
            self._register(child, gen, (p1.uid, p2.uid))
            pending.add(child.key())
            offspring.append(child)
        self._evaluate_batch(offspring)  # fitness evaluation (batched)
        return offspring

    # ------------------------------------------------------------------ bookkeeping
    def _record(self, gen, pop, result):
        """Append this generation's statistics (fitness, hypervolume, mutation trace) and log a progress line."""
        cfg = self.cfg
        fits = [g.fitness for g in pop]
        best = max(pop, key=lambda g: g.fitness)
        front = [g for g in pop if g.rank == 0]
        hv = hypervolume_mc([normalised_objectives(g.accuracy, g.runtime, g.params, cfg) for g in front])
        result.history.append(
            dict(
                gen=gen,
                best_fitness=max(fits),
                mean_fitness=float(np.mean(fits)),
                worst_fitness=min(fits),
                best_accuracy=max(g.accuracy for g in pop),
                mean_accuracy=float(np.mean([g.accuracy for g in pop])),
                mean_runtime=float(np.mean([g.runtime for g in pop])),
                mean_params=float(np.mean([g.params for g in pop])),
                front_size=len(front),
                hypervolume=hv,
                mean_mutation_rate=float(np.mean([g.mutation_rate for g in pop])),
                multiplier=self.meta.multiplier,
                pop_fitness=fits,
                best_uid=best.uid,
                trainings=self.evaluator.n_trainings,
            )
        )
        result.gen_best[gen] = best
        if self.verbose:
            log.info(
                f"gen {gen:>2} | best fit {max(fits):+.4f} | best acc {result.history[-1]['best_accuracy']:.3f} "
                f"| front {len(front):>2} | HV {hv:.3f} | mut x{self.meta.multiplier:.2f} "
                f"| trained nets {self.evaluator.n_trainings}"
            )

    # ------------------------------------------------------------------ 6. main evolution loop
    def run(self) -> Result:
        """Run the full evolution loop and return a :class:`Result`."""
        cfg = self.cfg
        result = Result(cfg, self.data, self.evaluator, self.meta)
        pop = self.initialise()
        self._record(0, pop, result)
        self.meta.update(max(g.fitness for g in pop))

        for gen in range(1, cfg.generations + 1):
            offspring = self.make_offspring(pop, gen)
            pool = self._dedup(pop + offspring)
            pop = environmental_selection(pool, cfg.pop_size)  # NSGA-II survival
            champion = max(pool, key=lambda g: g.fitness)  # elitism guard
            if all(champion is not g for g in pop):
                pop[-1] = champion
            self._record(gen, pop, result)
            self.meta.update(max(g.fitness for g in pop))  # meta-evolution update

        result.final_population = pop
        result.archive = list(self.archive.values())
        result.registry = self.registry
        self.evaluator.save_disk_cache()
        return result
