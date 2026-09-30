"""Self-adaptive mutation + lightweight meta-evolution.

Two cooperating levels (both lightweight, no extra training cost):

1. INDIVIDUAL level (evolution of the mutation rate itself)
   Every genome carries a `mutation_rate` gene. Before a child mutates its architecture it first
   mutates its OWN rate (log-normal step, as in Evolution Strategies). Children whose rate happened
   to produce good architectures survive -> good rates are selected for. Architecture and mutation
   rate therefore co-evolve.

2. POPULATION level (self-adaptive controller)
   A global multiplier watches whether the population improved:
     improved  -> multiplier shrinks  (exploit: fine-tune what works)
     stagnated -> multiplier grows    (explore: escape local optimum)
   Effective rate used for mutation = individual gene x multiplier.
"""

from __future__ import annotations

import math

import numpy as np


class MetaController:
    """Two-level self-adaptive mutation: per-genome rate genes plus a population-wide exploit/explore multiplier."""

    def __init__(self, cfg, shrink: float = 0.85, grow: float = 1.25):
        """``shrink`` < 1 is applied on improvement (exploit), ``grow`` > 1 on stagnation (explore)."""
        self.cfg = cfg
        self.shrink, self.grow = shrink, grow
        self.multiplier = 1.0
        self.best: float | None = None
        self.stagnation = 0
        self.log: list[tuple[str, float]] = []  # (action, multiplier) after every update

    # ---- level 1: evolve the mutation-rate gene ----
    def mutate_rate(self, rate: float, rng: np.random.Generator) -> float:
        """Log-normal self-mutation of a genome's own rate gene, clipped to ``[mut_min, mut_max]``."""
        new = rate * math.exp(self.cfg.meta_tau * rng.normal())
        return float(np.clip(new, self.cfg.mut_min, self.cfg.mut_max))

    # ---- rate actually used this generation ----
    def effective_rate(self, gene_rate: float) -> float:
        """Rate actually used to mutate: gene x population multiplier, clipped."""
        return float(np.clip(gene_rate * self.multiplier, self.cfg.mut_min, self.cfg.mut_max))

    # ---- level 2: adapt from population improvement ----
    def update(self, best_fitness: float) -> str:
        """Adapt the multiplier from this generation's best fitness; returns ``init``, ``exploit`` or ``explore``."""
        cfg = self.cfg
        if self.best is None:
            self.best, action = best_fitness, "init"
        elif best_fitness > self.best + cfg.improve_eps:
            self.best, self.stagnation, action = best_fitness, 0, "exploit"
            self.multiplier *= self.shrink
        else:
            self.stagnation += 1
            action = "explore"
            self.multiplier *= self.grow
        self.multiplier = float(np.clip(self.multiplier, cfg.mult_min, cfg.mult_max))
        self.log.append((action, self.multiplier))
        return action
