"""Genetic operators: selection, crossover, mutation."""

from __future__ import annotations

import numpy as np

from .genome import Genome


# ----------------------------------------------------------------------------- selection
def tournament_select(pop, k: int, rng: np.random.Generator) -> Genome:
    """Pick k random individuals; the best one (lowest Pareto rank, then most isolated) wins."""
    idx = rng.choice(len(pop), size=min(k, len(pop)), replace=False)
    return min((pop[i] for i in idx), key=lambda g: (g.rank, -g.crowding))


# ----------------------------------------------------------------------------- crossover
def crossover(p1: Genome, p2: Genome, cfg, rng: np.random.Generator) -> Genome:
    """Layers: one-point crossover (head of p1 + tail of p2).
    Activation / learning-rate: inherited from a random parent.
    Mutation-rate gene: average of the parents (so good rates spread through the population)."""
    cut1 = int(rng.integers(1, len(p1.layers) + 1))
    cut2 = int(rng.integers(0, len(p2.layers)))
    layers = (p1.layers[:cut1] + p2.layers[cut2:])[: cfg.max_layers]
    return Genome(
        layers=layers,
        activation=p1.activation if rng.random() < 0.5 else p2.activation,
        lr=p1.lr if rng.random() < 0.5 else p2.lr,
        mutation_rate=(p1.mutation_rate + p2.mutation_rate) / 2.0,
        notes=["crossover"],
    )


# ----------------------------------------------------------------------------- mutation
def _neighbour(choices, value, rng):
    """Move one step up/down in an ordered list of choices (small, local change)."""
    i = choices.index(value) if value in choices else 0
    j = int(np.clip(i + rng.choice([-1, 1]), 0, len(choices) - 1))
    return choices[j]


def _apply(op: str, g: Genome, cfg, rng) -> str | None:
    """Apply one mutation operator. Returns a human-readable note, or None if not possible."""
    if op == "resize":
        i = int(rng.integers(len(g.layers)))
        new_size = int(_neighbour(list(cfg.neuron_choices), g.layers[i], rng))
        if new_size == g.layers[i]:
            return None
        note = f"resize layer {i + 1}: {g.layers[i]}->{new_size}"
        g.layers[i] = new_size
        return note
    if op == "add":
        if len(g.layers) >= cfg.max_layers:
            return None
        i = int(rng.integers(len(g.layers) + 1))
        size = int(rng.choice(cfg.neuron_choices))
        g.layers.insert(i, size)
        return f"add layer of {size} at position {i + 1}"
    if op == "remove":
        if len(g.layers) <= cfg.min_layers:
            return None
        i = int(rng.integers(len(g.layers)))
        size = g.layers.pop(i)
        return f"remove layer {i + 1} ({size} neurons)"
    if op == "activation":
        options = [a for a in cfg.activations if a != g.activation]
        new_act = str(rng.choice(options))
        note = f"activation {g.activation}->{new_act}"
        g.activation = new_act
        return note
    if op == "lr":
        new_lr = float(_neighbour(list(cfg.lr_choices), g.lr, rng))
        if new_lr == g.lr:
            return None
        note = f"lr {g.lr:g}->{new_lr:g}"
        g.lr = new_lr
        return note
    return None


OPS = ("resize", "add", "remove", "activation", "lr")


def mutate(g: Genome, rate: float, cfg, rng: np.random.Generator) -> Genome:
    """Each operator fires independently with probability `rate`.
    If nothing fired we force one, so a child is never a plain copy of its parent."""
    notes = []
    for op in OPS:
        if rng.random() < rate:
            n = _apply(op, g, cfg, rng)
            if n:
                notes.append(n)
    if not notes:
        for op in rng.permutation(OPS):
            n = _apply(str(op), g, cfg, rng)
            if n:
                notes.append(n)
                break
    g.notes = g.notes + notes
    return g
