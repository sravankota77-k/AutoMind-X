"""Fitness evaluation: train a candidate network and measure the 3 objectives."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import warnings
from collections.abc import Iterable
from dataclasses import asdict, dataclass

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier

log = logging.getLogger(__name__)

# (accuracy, runtime_objective, params, wall_time)
Measurement = tuple[float, float, int, float]


def count_params(n_in: int, layers, n_out: int) -> int:
    """Weights + biases of a fully connected network."""
    sizes = [n_in, *layers, n_out]
    return int(sum(a * b + b for a, b in zip(sizes[:-1], sizes[1:], strict=True)))


def proxy_runtime(params: int, n_train: int, epochs: int, flops_per_s: float) -> float:
    """Deterministic training-cost estimate in pseudo-seconds.

    Forward + backward pass ~ 6 FLOPs per parameter per sample, times the epochs actually run.
    Being hardware independent, it makes the whole search reproducible.
    """
    return 6.0 * params * n_train * max(1, epochs) / flops_per_s


def normalised_objectives(acc, runtime, params, cfg):
    """All three objectives scaled to [0, 1], lower = better."""
    return (1.0 - acc, min(1.0, runtime / cfg.time_ref), min(1.0, params / cfg.params_ref))


def fitness_terms(acc, runtime, params, cfg):
    """Return (accuracy reward, time penalty, size penalty)."""
    return (
        cfg.w_acc * acc,
        cfg.w_time * min(1.0, runtime / cfg.time_ref),
        cfg.w_params * min(1.0, params / cfg.params_ref),
    )


def scalar_fitness(acc, runtime, params, cfg) -> float:
    """Weighted score ``w_acc*acc - w_time*t - w_params*p`` (tracking, adaptation and final pick only)."""
    reward, t_pen, p_pen = fitness_terms(acc, runtime, params, cfg)
    return reward - t_pen - p_pen


def _fit_model(spec, data, cfg) -> tuple[MLPClassifier, float]:
    """Fit one MLP from a ``(layers, activation, lr)`` spec; return ``(model, wall_seconds)``."""
    layers, activation, lr = spec
    model = MLPClassifier(
        hidden_layer_sizes=tuple(layers),
        activation=activation,
        learning_rate_init=lr,
        max_iter=cfg.max_iter,
        early_stopping=True,
        n_iter_no_change=8,
        random_state=cfg.seed,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        t0 = time.perf_counter()
        model.fit(data.X_train, data.y_train)
        wall = time.perf_counter() - t0
    return model, wall


def _train_and_score(spec, data, cfg, eval_split: str = "val") -> Measurement:
    """Train one network. Module-level (picklable) so it can run in worker processes."""
    layers = spec[0]
    model, wall = _fit_model(spec, data, cfg)
    X, y = (data.X_val, data.y_val) if eval_split == "val" else (data.X_test, data.y_test)
    acc = float(model.score(X, y))
    params = count_params(data.n_features, layers, data.n_classes)
    if cfg.runtime_metric == "proxy":
        runtime = proxy_runtime(params, len(data.y_train), int(model.n_iter_), cfg.proxy_flops_per_s)
    else:
        runtime = wall
    return acc, runtime, params, wall


@dataclass(frozen=True)
class LatencyProfile:
    """Inference latency of a trained model on the held-out test split (milliseconds)."""

    batch_ms_per_1k: float  # time to classify 1000 samples in one batch call
    single_p50_ms: float  # median latency of a one-sample ``predict`` call
    single_p95_ms: float  # 95th-percentile latency of a one-sample ``predict`` call
    repeats: int


def fingerprint(cfg) -> str:
    """Short hash of every setting that changes a measurement -> safe key for the on-disk cache."""
    relevant = {
        k: v
        for k, v in asdict(cfg).items()
        if k
        in {
            "seed",
            "n_samples",
            "n_features",
            "n_informative",
            "n_redundant",
            "n_classes",
            "class_sep",
            "flip_y",
            "max_iter",
            "runtime_metric",
            "proxy_flops_per_s",
        }
    }
    blob = json.dumps(relevant, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


# Below this batch size, process start-up costs more than the parallel speed-up saves.
MIN_PARALLEL_BATCH = 4


class Evaluator:
    """Trains candidates, caches results by architecture, and evaluates batches in parallel."""

    def __init__(self, data, cfg, cache_path: str | None = None):
        """``cache_path`` (optional, proxy metric only) persists measurements as JSON between runs."""
        self.data, self.cfg = data, cfg
        self._cache: dict = {}
        self._holdout_cache: dict = {}
        self._latency_cache: dict = {}
        self.n_trainings = 0
        self.cache_hits = 0
        self.disk_hits = 0
        self.train_seconds = 0.0  # wall time spent inside ``_train_batch`` (profiling)
        self._cache_path = cache_path if cfg.runtime_metric == "proxy" else None
        self._fingerprint = fingerprint(cfg)
        self._load_disk_cache()

    # ------------------------------------------------------------------ optional persistent cache
    @staticmethod
    def _key_to_str(key) -> str:
        """Stable JSON string for an architecture key (used by the on-disk cache)."""
        return json.dumps([list(key[0]), key[1], key[2]])

    def _load_disk_cache(self) -> None:
        """Load cached measurements if the file exists and was produced under identical settings."""
        if not self._cache_path or not os.path.isfile(self._cache_path):
            return
        try:
            with open(self._cache_path, encoding="utf-8") as f:
                blob = json.load(f)
        except (OSError, ValueError):
            log.warning("ignoring unreadable cache file %s", self._cache_path)
            return
        if blob.get("fingerprint") != self._fingerprint:
            return  # produced under different settings -> not comparable
        self._disk: dict = blob.get("entries", {})

    def save_disk_cache(self) -> None:
        """Write all measurements made so far to ``cache_path`` (no-op when caching is off)."""
        if not self._cache_path:
            return
        entries = {self._key_to_str(k): list(v) for k, v in self._cache.items()}
        entries.update(getattr(self, "_disk", {}))
        os.makedirs(os.path.dirname(os.path.abspath(self._cache_path)), exist_ok=True)
        with open(self._cache_path, "w", encoding="utf-8") as f:
            json.dump({"fingerprint": self._fingerprint, "entries": entries}, f)

    # ------------------------------------------------------------------ single / batch evaluation
    def _apply(self, g, m: Measurement) -> None:
        """Copy a cached measurement onto a genome and compute its scalar fitness."""
        g.accuracy, g.runtime, g.params, g.wall_time = m
        g.fitness = scalar_fitness(g.accuracy, g.runtime, g.params, self.cfg)

    def evaluate(self, g):
        """Fill g.accuracy / runtime / params / fitness. Identical architectures are cached."""
        return self.evaluate_many([g])[0]

    def evaluate_many(self, genomes: Iterable) -> list:
        """Evaluate a batch; each *unique unseen* architecture is trained once, in parallel if allowed."""
        genomes = list(genomes)
        todo = {}
        disk = getattr(self, "_disk", {})
        for g in genomes:
            key = g.key()
            if key in self._cache or key in todo:
                continue
            stored = disk.get(self._key_to_str(key))
            if stored is not None:  # measured in an earlier run under identical settings
                self._cache[key] = (float(stored[0]), float(stored[1]), int(stored[2]), float(stored[3]))
                self.disk_hits += 1
                continue
            todo[key] = (tuple(g.layers), g.activation, g.lr)
        t0 = time.perf_counter()
        results = self._train_batch(list(todo.values()))
        self.train_seconds += time.perf_counter() - t0
        for key, m in zip(todo.keys(), results, strict=True):
            self._cache[key] = m
            self.n_trainings += 1
        self.cache_hits += len(genomes) - len(todo)
        for g in genomes:
            self._apply(g, self._cache[g.key()])
        return genomes

    def _train_batch(self, specs: list) -> list[Measurement]:
        """Train a list of specs, in parallel when allowed (proxy metric) and serially otherwise."""
        if not specs:
            return []
        jobs = self.cfg.n_jobs
        # Parallel wall-clock timings would be distorted by CPU contention -> serial in that mode.
        if len(specs) >= MIN_PARALLEL_BATCH and jobs != 1 and self.cfg.runtime_metric == "proxy":
            try:
                from joblib import Parallel, delayed

                return list(Parallel(n_jobs=jobs)(delayed(_train_and_score)(s, self.data, self.cfg) for s in specs))
            except Exception as exc:  # sandbox without multiprocessing, etc. -> degrade gracefully
                log.warning("parallel evaluation unavailable (%s); falling back to serial", exc)
        return [_train_and_score(s, self.data, self.cfg) for s in specs]

    # ------------------------------------------------------------------ hold-out + baseline
    def holdout(self, g):
        """Retrain and score on the untouched TEST split -> (test_acc, runtime, params). Memoised."""
        key = g.key()
        if key not in self._holdout_cache:
            acc, runtime, params, _ = _train_and_score(
                (tuple(g.layers), g.activation, g.lr), self.data, self.cfg, eval_split="test"
            )
            self._holdout_cache[key] = (acc, runtime, params)
        return self._holdout_cache[key]

    def latency(self, g, repeats: int = 200) -> LatencyProfile:
        """Measure inference latency of ``g`` (retrains once, memoised). Batch and single-sample."""
        key = g.key()
        if key not in self._latency_cache:
            model, _ = _fit_model((tuple(g.layers), g.activation, g.lr), self.data, self.cfg)
            X = self.data.X_test
            model.predict(X[:1])  # warm-up
            t0 = time.perf_counter()
            model.predict(X[:1000])
            batch_ms = (time.perf_counter() - t0) * 1000.0 * (1000.0 / min(1000, len(X)))
            singles = []
            for i in range(repeats):
                t0 = time.perf_counter()
                model.predict(X[i % len(X) : i % len(X) + 1])
                singles.append((time.perf_counter() - t0) * 1000.0)
            self._latency_cache[key] = LatencyProfile(
                batch_ms_per_1k=batch_ms,
                single_p50_ms=float(np.percentile(singles, 50)),
                single_p95_ms=float(np.percentile(singles, 95)),
                repeats=repeats,
            )
        return self._latency_cache[key]

    def baseline(self):
        """A plain 'default' MLP (100,) for comparison."""
        from .genome import Genome

        base = Genome(layers=[100], activation="relu", lr=0.001, mutation_rate=0.0)
        return base, self.holdout(base)
