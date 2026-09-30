"""Property-based tests (Hypothesis), budget-aware selection, accessible report, evaluator memoisation."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import main as cli
from automind.config import Config
from automind.dataset import make_dataset
from automind.evaluator import MIN_PARALLEL_BATCH, Evaluator
from automind.evolution import AutoMindX
from automind.explain import explain, global_front, select_final, within_budget
from automind.genome import Genome, random_genome
from automind.pareto import dominance_matrix, dominates, non_dominated_sort
from automind.report import build_html, write_html

points = st.lists(st.tuples(*[st.floats(0, 1, allow_nan=False)] * 3), min_size=1, max_size=25)


# ---------- property-based: Pareto invariants ----------
@settings(max_examples=60, deadline=None)
@given(points)
def test_fronts_partition_and_first_front_is_non_dominated(objs):
    fronts = non_dominated_sort(objs)
    flat = sorted(i for f in fronts for i in f)
    assert flat == list(range(len(objs)))  # every index exactly once
    first = fronts[0]
    assert not any(dominates(objs[a], objs[b]) for a in first for b in first)
    for later in fronts[1:]:  # everything later is dominated by someone earlier
        for j in later:
            assert any(dominates(objs[i], objs[j]) for i in range(len(objs)))


@settings(max_examples=60, deadline=None)
@given(points)
def test_dominance_matrix_matches_scalar_dominates(objs):
    m = dominance_matrix(objs)
    for i in range(len(objs)):
        for j in range(len(objs)):
            assert bool(m[i, j]) == dominates(objs[i], objs[j])


@settings(max_examples=40, deadline=None)
@given(st.integers(0, 10_000))
def test_random_genomes_always_valid(seed):
    cfg = Config()
    g = random_genome(cfg, np.random.default_rng(seed))
    assert cfg.min_layers <= len(g.layers) <= cfg.max_layers
    assert g.activation in cfg.activations and g.lr in cfg.lr_choices


def test_dominance_matrix_empty():
    assert dominance_matrix([]).shape == (0, 0)


# ---------- budget-aware selection ----------
def tiny(**kw):
    return Config(n_samples=300, pop_size=6, generations=3, max_iter=15, n_jobs=1, **kw)


@pytest.fixture(scope="module")
def base_result():
    return AutoMindX(tiny(), verbose=False).run()


def test_budget_is_respected_when_feasible(base_result):
    smallest = min(g.params for g in global_front(base_result))
    cfg = base_result.cfg
    cfg.max_params = smallest + 1
    try:
        chosen, _ = select_final(base_result)
        assert chosen.params <= cfg.max_params and within_budget(chosen, cfg)
        assert "BUDGET" in explain(base_result) and "satisfied" in explain(base_result)
    finally:
        cfg.max_params = None


def test_impossible_budget_returns_least_violating(base_result):
    cfg = base_result.cfg
    cfg.max_params = 1
    try:
        chosen, front = select_final(base_result)
        assert chosen.params == min(g.params for g in front)
        assert "NOT satisfiable" in explain(base_result)
    finally:
        cfg.max_params = None


def test_cost_budget_filters(base_result):
    cfg = base_result.cfg
    cheapest = min(g.runtime for g in global_front(base_result))
    cfg.max_cost = cheapest * 1.0001
    try:
        chosen, _ = select_final(base_result)
        assert chosen.runtime <= cfg.max_cost
    finally:
        cfg.max_cost = None


def test_budget_config_validation():
    with pytest.raises(ValueError):
        Config(max_params=0)
    with pytest.raises(ValueError):
        Config(max_cost=-1.0)


def test_cli_budget_flag(tmp_path, capsys):
    out = tmp_path / "o"
    args = ["--quick", "--generations", "2", "--pop", "5", "--n-jobs", "1", "--quiet", "--no-plots"]
    assert cli.main([*args, "--max-params", "800", "--outdir", str(out)]) == 0
    assert "BUDGET" in (out / "explanation.txt").read_text()


# ---------- accessible HTML report ----------
def test_html_report_is_accessible_and_escaped(base_result):
    page = build_html(base_result)
    assert '<html lang="en">' in page and page.count("<h1>") == 1
    assert 'href="#main"' in page and "<caption>" in page and 'scope="col"' in page
    assert page.count("<img") == page.count("alt='") == 4  # every image has alt text
    assert "<script" not in page  # no JavaScript


def test_html_escapes_untrusted_text(base_result):
    g = base_result.final_population[0]
    old = g.activation
    g.activation = "<script>alert(1)</script>"
    try:
        assert "<script>alert(1)</script>" not in build_html(base_result)
    finally:
        g.activation = old


def test_write_html(tmp_path, base_result):
    path = write_html(base_result, str(tmp_path))
    assert path.endswith("report.html") and (tmp_path / "report.html").stat().st_size > 500


# ---------- evaluator memoisation & thresholds ----------
def test_holdout_is_memoised():
    cfg = tiny()
    ev = Evaluator(make_dataset(cfg), cfg)
    g = Genome([8], "relu", 0.01, 0.3)
    first = ev.holdout(g)
    assert ev.holdout(g) is first  # same cached tuple, no retraining
    base, res = ev.baseline()
    assert ev.baseline()[1] is res


def test_small_batches_skip_parallel(monkeypatch):
    cfg = Config(n_samples=300, max_iter=10, n_jobs=2)
    ev = Evaluator(make_dataset(cfg), cfg)
    called = []
    import joblib

    monkeypatch.setattr(joblib, "Parallel", lambda *a, **k: called.append(1))
    rng = np.random.default_rng(0)
    ev.evaluate_many([random_genome(cfg, rng) for _ in range(MIN_PARALLEL_BATCH - 1)])
    assert not called


def test_parallel_failure_falls_back_to_serial(monkeypatch):
    cfg = Config(n_samples=300, max_iter=10, n_jobs=2)
    ev = Evaluator(make_dataset(cfg), cfg)
    import joblib

    def boom(*a, **k):
        raise OSError("no multiprocessing here")

    monkeypatch.setattr(joblib, "Parallel", boom)
    rng = np.random.default_rng(0)
    gs = ev.evaluate_many([random_genome(cfg, rng) for _ in range(MIN_PARALLEL_BATCH + 1)])
    assert all(g.accuracy > 0 for g in gs)
