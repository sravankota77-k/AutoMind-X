"""Central configuration. Every knob of the system lives here, and is validated on creation."""

from __future__ import annotations

import os
from dataclasses import dataclass

RUNTIME_METRICS = ("proxy", "wallclock")
SELECTION_STRATEGIES = ("balanced", "knee")

# Environment variables read by :meth:`Config.from_env` (documented in ``.env.example``).
ENV_PREFIX = "AUTOMIND_"
_ENV_FIELDS: dict[str, type] = {
    "seed": int,
    "pop_size": int,
    "generations": int,
    "n_jobs": int,
    "runtime_metric": str,
    "selection": str,
    "max_params": int,
    "max_cost": float,
    "outdir": str,
}


@dataclass
class Config:
    """Every tunable of AutoMind-X in one validated dataclass.

    Build it directly (``Config(pop_size=20)``), from the CLI (``main.build_config``) or from
    environment variables (:meth:`from_env`). Invalid values raise ``ValueError`` immediately.
    """

    # ---- reproducibility ----
    seed: int = 42

    # ---- synthetic dataset (scikit-learn make_classification) ----
    n_samples: int = 2500
    n_features: int = 20
    n_informative: int = 10
    n_redundant: int = 4
    n_classes: int = 3
    class_sep: float = 0.9
    flip_y: float = 0.03

    # ---- search space ----
    min_layers: int = 1
    max_layers: int = 4
    neuron_choices: tuple = (4, 8, 16, 32, 64, 128)
    activations: tuple = ("relu", "tanh", "logistic")
    lr_choices: tuple = (0.001, 0.003, 0.01, 0.03)

    # ---- training budget for each candidate network ----
    max_iter: int = 60

    # ---- runtime objective ----
    # "proxy"     : deterministic FLOPs-based estimate -> fully reproducible runs (default)
    # "wallclock" : measured seconds -> realistic but noisy
    runtime_metric: str = "proxy"
    proxy_flops_per_s: float = 1.0e9  # calibration of the FLOPs -> pseudo-seconds conversion

    # ---- efficiency ----
    n_jobs: int = -1  # parallel training of offspring (-1 = all cores, 1 = serial)

    # ---- genetic algorithm ----
    pop_size: int = 14
    generations: int = 12
    crossover_rate: float = 0.85
    tournament_size: int = 2

    # ---- self-adaptive mutation + meta-evolution ----
    init_mutation_rate: float = 0.35  # starting value of every individual's own gene
    mut_min: float = 0.05
    mut_max: float = 0.90
    meta_tau: float = 0.30  # step size for evolving the mutation-rate gene
    mult_min: float = 0.40  # bounds of the population-level multiplier
    mult_max: float = 2.50
    improve_eps: float = 1e-3  # what counts as "the population improved"

    # ---- scalar fitness (used for tracking, adaptation and final pick) ----
    w_acc: float = 0.60
    w_time: float = 0.20
    w_params: float = 0.20
    time_ref: float = 0.30  # seconds that count as "maximally slow"
    params_ref: float = 6000  # parameter count that counts as "maximally big"

    # ---- final model selection ----
    selection: str = "balanced"  # "balanced" (weighted fitness) or "knee" (closest to ideal point)
    max_params: int | None = None  # deployment budget: model may have at most this many parameters
    max_cost: float | None = None  # deployment budget: training cost (same unit as runtime) upper bound

    # ---- output ----
    outdir: str = "outputs"

    def __post_init__(self) -> None:
        """Validate immediately so an invalid Config can never exist."""
        self.validate()

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None, **overrides) -> Config:
        """Create a Config from ``AUTOMIND_*`` environment variables (see ``.env.example``).

        Explicit keyword ``overrides`` win over the environment. Unset variables keep defaults;
        no secret or credential is ever read - only tuning knobs.
        """
        env = os.environ if environ is None else environ
        values: dict = {}
        for name, typ in _ENV_FIELDS.items():
            raw = env.get(ENV_PREFIX + name.upper())
            if raw not in (None, ""):
                try:
                    values[name] = typ(raw)
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid Config: {ENV_PREFIX}{name.upper()}={raw!r} is not a valid {typ.__name__}"
                    ) from exc
        values.update(overrides)
        return cls(**values)

    def validate(self) -> None:
        """Fail fast with a clear message instead of crashing deep inside the search."""

        def need(cond: bool, msg: str) -> None:
            """Raise ``ValueError`` with a uniform message when ``cond`` is false."""
            if not cond:
                raise ValueError(f"Invalid Config: {msg}")

        need(self.n_samples >= 100, "n_samples must be >= 100")
        need(self.n_classes >= 2, "n_classes must be >= 2")
        need(
            self.n_informative + self.n_redundant <= self.n_features,
            "n_informative + n_redundant must be <= n_features",
        )
        need(self.n_informative >= 1, "n_informative must be >= 1")
        need(2**self.n_informative >= self.n_classes, "n_informative too small for n_classes")
        need(1 <= self.min_layers <= self.max_layers, "1 <= min_layers <= max_layers required")
        need(
            len(self.neuron_choices) > 0 and all(int(n) > 0 for n in self.neuron_choices),
            "neuron_choices must be positive integers",
        )
        need(len(self.activations) > 1, "need at least two activations")
        need(len(self.lr_choices) > 0 and all(x > 0 for x in self.lr_choices), "lr_choices must be > 0")
        need(self.max_iter >= 1, "max_iter must be >= 1")
        need(self.pop_size >= 2, "pop_size must be >= 2")
        need(self.generations >= 1, "generations must be >= 1")
        need(0.0 <= self.crossover_rate <= 1.0, "crossover_rate must be in [0, 1]")
        need(self.tournament_size >= 1, "tournament_size must be >= 1")
        need(0.0 < self.mut_min <= self.mut_max <= 1.0, "0 < mut_min <= mut_max <= 1 required")
        need(0.0 < self.mult_min <= self.mult_max, "0 < mult_min <= mult_max required")
        need(min(self.w_acc, self.w_time, self.w_params) >= 0.0, "fitness weights must be >= 0")
        need(self.time_ref > 0 and self.params_ref > 0, "time_ref and params_ref must be > 0")
        need(self.proxy_flops_per_s > 0, "proxy_flops_per_s must be > 0")
        need(self.runtime_metric in RUNTIME_METRICS, f"runtime_metric must be one of {RUNTIME_METRICS}")
        need(self.selection in SELECTION_STRATEGIES, f"selection must be one of {SELECTION_STRATEGIES}")
        need(self.max_params is None or self.max_params > 0, "max_params must be > 0")
        need(self.max_cost is None or self.max_cost > 0, "max_cost must be > 0")
        need(self.n_jobs != 0, "n_jobs must be non-zero (-1 = all cores)")
