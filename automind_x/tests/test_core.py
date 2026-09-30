import itertools

import numpy as np
from sklearn.neural_network import MLPClassifier

from automind.config import Config
from automind.dataset import make_dataset
from automind.evaluator import Evaluator, count_params, scalar_fitness
from automind.genome import Genome, random_genome
from automind.meta import MetaController
from automind.operators import crossover, mutate
from automind.pareto import crowding_distance, dominates, environmental_selection, hypervolume_mc, non_dominated_sort


def tiny_cfg(**kw):
    return Config(n_samples=300, pop_size=6, generations=3, max_iter=15, **kw)


def valid(g, cfg):
    return (
        cfg.min_layers <= len(g.layers) <= cfg.max_layers
        and all(n in cfg.neuron_choices for n in g.layers)
        and g.activation in cfg.activations
        and g.lr in cfg.lr_choices
    )


# ---------- pareto ----------
def test_dominates():
    assert dominates((1, 1, 1), (2, 2, 2))
    assert not dominates((1, 3, 1), (2, 2, 2))
    assert not dominates((1, 1, 1), (1, 1, 1))


def test_sort_and_crowding():
    objs = [(1, 4), (2, 2), (4, 1), (3, 3), (5, 5)]
    fronts = non_dominated_sort(objs)
    assert sorted(fronts[0]) == [0, 1, 2] and fronts[1] == [3] and fronts[2] == [4]
    d = crowding_distance(objs, fronts[0])
    assert d[0] == d[2] == float("inf") and np.isfinite(d[1])


def test_environmental_selection_keeps_front():
    gs = [
        Genome([8], "relu", 0.01, 0.3, accuracy=a, runtime=r, params=p)
        for a, r, p in [(0.9, 1, 100), (0.8, 2, 200), (0.95, 3, 300), (0.5, 9, 900)]
    ]
    surv = environmental_selection(gs, 3)
    assert len(surv) == 3 and gs[3] not in surv


def test_hypervolume_monotone():
    small = hypervolume_mc([(0.5, 0.5, 0.5)])
    big = hypervolume_mc([(0.5, 0.5, 0.5), (0.1, 0.9, 0.9)])
    assert 0 < small < big <= 1 and abs(small - 0.125) < 0.02


# ---------- genome / operators ----------
def test_operators_keep_genomes_valid():
    cfg, rng = Config(), np.random.default_rng(0)
    for _ in range(300):
        a, b = random_genome(cfg, rng), random_genome(cfg, rng)
        c = crossover(a, b, cfg, rng)
        n_before = len(c.notes)
        mutate(c, rng.random(), cfg, rng)
        assert valid(c, cfg) and len(c.notes) > n_before  # >=1 operator always applied


# ---------- evaluator ----------
def test_param_count_matches_sklearn():
    m = MLPClassifier(hidden_layer_sizes=(7, 5), max_iter=5, random_state=0)
    X = np.random.default_rng(0).normal(size=(60, 4))
    y = np.arange(60) % 3
    m.fit(X, y)
    real = sum(w.size for w in m.coefs_) + sum(b.size for b in m.intercepts_)
    assert real == count_params(4, [7, 5], 3)


def test_fitness_prefers_small_fast_accurate():
    cfg = Config()
    assert scalar_fitness(0.9, 0.1, 500, cfg) > scalar_fitness(0.9, 0.2, 500, cfg)
    assert scalar_fitness(0.9, 0.1, 500, cfg) > scalar_fitness(0.9, 0.1, 5000, cfg)
    assert scalar_fitness(0.95, 0.1, 500, cfg) > scalar_fitness(0.9, 0.1, 500, cfg)


def test_evaluator_caches():
    cfg = tiny_cfg()
    ev = Evaluator(make_dataset(cfg), cfg)
    g = random_genome(cfg, np.random.default_rng(1))
    ev.evaluate(g)
    acc = g.accuracy
    g2 = g.clone()
    ev.evaluate(g2)
    assert ev.n_trainings == 1 and ev.cache_hits == 1 and g2.accuracy == acc


# ---------- meta ----------
def test_meta_adapts_and_is_bounded():
    cfg = Config()
    m = MetaController(cfg)
    m.update(0.5)
    for _ in range(30):
        m.update(0.5)  # stagnation -> explore
    assert m.multiplier == cfg.mult_max
    m.update(0.9)  # improvement -> exploit
    assert m.multiplier < cfg.mult_max
    rng = np.random.default_rng(0)
    assert all(cfg.mut_min <= m.mutate_rate(0.3, rng) <= cfg.mut_max for _ in range(200))


# ---------- end to end ----------
def test_evolution_smoke_and_elitism():
    from automind.evolution import AutoMindX
    from automind.explain import explain, select_final

    cfg = tiny_cfg()
    r = AutoMindX(cfg, verbose=False).run()
    assert len(r.history) == cfg.generations + 1
    best = [h["best_fitness"] for h in r.history]
    assert all(b2 >= b1 - 1e-12 for b1, b2 in itertools.pairwise(best))  # elitism: never gets worse
    chosen, front = select_final(r)
    assert chosen in front and chosen.rank == 0
    text = explain(r)
    assert "WHY THIS ARCHITECTURE WAS SELECTED" in text and "META-EVOLUTION" in text


def test_same_seed_same_initial_population():
    """Generation 0 is fully seed-determined. Later generations can differ slightly because
    runtime is measured wall-clock time (documented caveat)."""
    from automind.evolution import AutoMindX

    keys = []
    for _ in range(2):
        e = AutoMindX(tiny_cfg(seed=7), verbose=False)
        keys.append([g.key() for g in e.initialise()])
    assert sorted(keys[0]) == sorted(keys[1])
