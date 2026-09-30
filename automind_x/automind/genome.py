"""The Genome: one candidate neural network + its own mutation-rate gene."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np


@dataclass(eq=False)  # identity semantics: two genomes are equal only if they are the same object
class Genome:
    # ---- genes (what evolution changes) ----
    """One candidate network (genes) plus its own mutation-rate gene, bookkeeping and measured results."""

    layers: list[int]  # neurons per hidden layer, e.g. [64, 32]
    activation: str
    lr: float
    mutation_rate: float  # META gene: evolves together with the architecture

    # ---- bookkeeping (for explainability) ----
    uid: int = -1
    parents: tuple[int, ...] = ()
    birth_gen: int = 0
    notes: list[str] = field(default_factory=list)  # what happened when this genome was created

    # ---- results (filled by the evaluator / selection) ----
    accuracy: float = float("nan")
    runtime: float = float("nan")  # the runtime OBJECTIVE (proxy or wall-clock, see Config.runtime_metric)
    wall_time: float = float("nan")  # measured wall-clock seconds (informational)
    params: int = 0
    fitness: float = float("-inf")
    rank: int = 0
    crowding: float = 0.0

    def key(self):
        """Identity of the *architecture* (mutation rate excluded) - used for caching."""
        return (tuple(self.layers), self.activation, round(self.lr, 8))

    def describe(self) -> str:
        """Short human-readable architecture string, e.g. ``64-32 | relu | lr=0.01``."""
        return f"{'-'.join(map(str, self.layers))} | {self.activation} | lr={self.lr:g}"

    def objectives(self):
        """All objectives as MINIMISATION values: (error, runtime, params)."""
        return (1.0 - self.accuracy, self.runtime, float(self.params))

    def clone(self) -> Genome:
        """Deep-enough copy (layers and notes are duplicated so mutation cannot leak into the parent)."""
        return dataclasses.replace(self, layers=list(self.layers), notes=list(self.notes))


def random_genome(cfg, rng: np.random.Generator) -> Genome:
    """Sample a random valid genome from the search space defined in Config."""
    n_layers = int(rng.integers(cfg.min_layers, cfg.max_layers + 1))
    layers = [int(rng.choice(cfg.neuron_choices)) for _ in range(n_layers)]
    return Genome(
        layers=layers,
        activation=str(rng.choice(cfg.activations)),
        lr=float(rng.choice(cfg.lr_choices)),
        mutation_rate=cfg.init_mutation_rate,
        notes=["random init"],
    )
