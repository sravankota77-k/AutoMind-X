"""Tests for v1.2: random-search baseline, env config, persistent cache, latency, profiling, CLI flags."""

import json
import os

import numpy as np
import pytest

import main as cli
from automind.baselines import (
    Comparison,
    SearchOutcome,
    compare_with_random,
    evolutionary_outcome,
    format_comparison,
    random_search,
)
from automind.config import Config
from automind.dataset import make_dataset
from automind.evaluator import Evaluator, LatencyProfile, fingerprint
from automind.evolution import AutoMindX
from automind.genome import Genome


def tiny(**kw):
    return Config(n_samples=300, pop_size=6, generations=2, max_iter=10, n_jobs=1, **kw)


# ---------- Config.from_env ----------
def test_from_env_reads_typed_values():
    cfg = Config.from_env({"AUTOMIND_SEED": "7", "AUTOMIND_MAX_COST": "0.5", "AUTOMIND_SELECTION": "knee"})
    assert cfg.seed == 7 and cfg.max_cost == 0.5 and cfg.selection == "knee"


def test_from_env_ignores_empty_and_unset():
    cfg = Config.from_env({"AUTOMIND_MAX_PARAMS": "", "AUTOMIND_SEED": ""})
    assert cfg.max_params is None and cfg.seed == 42


def test_from_env_overrides_win():
    assert Config.from_env({"AUTOMIND_SEED": "7"}, seed=9).seed == 9


@pytest.mark.parametrize(
    "env", [{"AUTOMIND_SEED": "abc"}, {"AUTOMIND_RUNTIME_METRIC": "nope"}, {"AUTOMIND_POP_SIZE": "1"}]
)
def test_from_env_rejects_bad_values(env):
    with pytest.raises(ValueError, match="Invalid Config"):
        Config.from_env(env)


def test_env_example_lists_every_supported_variable():
    from automind.config import _ENV_FIELDS, ENV_PREFIX

    env_file = os.path.join(os.path.dirname(__file__), "..", ".env.example")
    with open(env_file, encoding="utf-8") as f:
        text = f.read()
    for name in _ENV_FIELDS:
        assert f"{ENV_PREFIX}{name.upper()}=" in text


# ---------- random search baseline ----------
def test_random_search_uses_exact_budget_and_unique_architectures():
    cfg = tiny()
    out = random_search(cfg, 10)
    assert isinstance(out, SearchOutcome)
    assert out.n_trainings == 10 == len({g.key() for g in out.archive})
    assert 0.0 <= out.hypervolume <= 1.0 and out.front_size >= 1


def test_random_search_is_reproducible():
    a, b = random_search(tiny(), 8), random_search(tiny(), 8)
    assert a.hypervolume == b.hypervolume and a.best_fitness == b.best_fitness


def test_equal_budget_comparison_and_summary():
    cmp = compare_with_random(tiny(), seeds=[0, 1, 2])
    assert isinstance(cmp, Comparison)
    for e, r in zip(cmp.evo, cmp.rnd, strict=True):
        assert e.n_trainings == r.n_trainings  # same training budget = fair
    s = cmp.summary("hypervolume")
    assert s["wins"] + s["ties"] + s["losses"] == 3
    assert s["ci95"][0] <= s["mean_diff"] <= s["ci95"][1]
    text = format_comparison(cmp)
    assert "EQUAL-BUDGET COMPARISON" in text and "hypervolume" in text and "verdict" in text


def test_summary_flags_a_clear_win_and_a_clear_loss():
    def mk(hv):
        return SearchOutcome("m", 5, hv, hv, hv, 1)

    win = Comparison([0, 1, 2, 3], [mk(0.9)] * 4, [mk(0.5)] * 4)
    assert win.summary()["significant"] and win.summary()["wins"] == 4
    assert "AutoMind-X is significantly better" in format_comparison(win)
    loss = Comparison([0, 1, 2, 3], [mk(0.5)] * 4, [mk(0.9)] * 4)
    assert "random search is significantly better" in format_comparison(loss)
    tie = Comparison([0, 1], [mk(0.5)] * 2, [mk(0.5)] * 2)
    assert "no statistically clear difference" in format_comparison(tie)


def test_evolutionary_outcome_matches_direct_run():
    cfg = tiny()
    out = evolutionary_outcome(cfg)
    direct = AutoMindX(cfg, verbose=False).run()
    assert out.n_trainings == direct.evaluator.n_trainings


# ---------- latency ----------
def test_latency_profile_is_positive_ordered_and_memoised():
    cfg = tiny()
    ev = Evaluator(make_dataset(cfg), cfg)
    g = Genome(layers=[8], activation="relu", lr=0.01, mutation_rate=0.1)
    lat = ev.latency(g, repeats=20)
    assert isinstance(lat, LatencyProfile)
    assert lat.batch_ms_per_1k > 0 and 0 < lat.single_p50_ms <= lat.single_p95_ms
    assert ev.latency(g) is lat  # memoised


# ---------- persistent cache ----------
def test_fingerprint_changes_with_settings_only_that_matter():
    assert fingerprint(Config()) == fingerprint(Config(pop_size=20))  # search size does not change measurements
    assert fingerprint(Config()) != fingerprint(Config(seed=1))
    assert fingerprint(Config()) != fingerprint(Config(max_iter=10))


def test_disk_cache_roundtrip_skips_retraining(tmp_path):
    path = str(tmp_path / "cache.json")
    cfg = tiny()
    first = AutoMindX(cfg, verbose=False, cache_path=path).run()
    assert first.evaluator.n_trainings > 0 and os.path.isfile(path)
    second = AutoMindX(cfg, verbose=False, cache_path=path).run()
    assert second.evaluator.n_trainings == 0 and second.evaluator.disk_hits > 0
    assert [h["best_fitness"] for h in first.history] == [h["best_fitness"] for h in second.history]


def test_disk_cache_ignored_when_settings_differ_or_file_corrupt(tmp_path):
    path = tmp_path / "cache.json"
    AutoMindX(tiny(), verbose=False, cache_path=str(path)).run()
    other = AutoMindX(tiny(seed=5), verbose=False, cache_path=str(path)).run()
    assert other.evaluator.disk_hits == 0  # different fingerprint -> not reused
    path.write_text("{not json", encoding="utf-8")
    again = AutoMindX(tiny(), verbose=False, cache_path=str(path)).run()
    assert again.evaluator.n_trainings > 0  # corrupt file is ignored, not fatal


def test_disk_cache_disabled_for_wallclock(tmp_path):
    path = tmp_path / "cache.json"
    cfg = tiny(runtime_metric="wallclock")
    AutoMindX(cfg, verbose=False, cache_path=str(path)).run()
    assert not path.exists()  # wall-clock timings are not reproducible -> never cached


# ---------- CLI ----------
def test_cli_profile_json_latency_and_cache(tmp_path):
    out, cache = tmp_path / "out", tmp_path / "c.json"
    argv = ["--quick", "--quiet", "--no-plots", "--json", "--profile", "--cache", str(cache), "--outdir", str(out)]
    assert cli.main(argv) == 0
    assert (out / "profile.txt").read_text(encoding="utf-8").strip()
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["latency"]["single_p50_ms"] > 0
    assert 0 <= summary["timing"]["training_share"] <= 1.05
    assert "INFERENCE LATENCY" in (out / "explanation.txt").read_text(encoding="utf-8")
    assert cli.main(argv) == 0  # warm run
    warm = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert warm["trainings"] == 0 and warm["disk_cache_hits"] > 0


def test_cli_compare_random_appends_section(tmp_path):
    out = tmp_path / "o"
    argv = ["--generations", "2", "--pop", "4", "--quiet", "--no-plots", "--compare-random", "2", "--outdir", str(out)]
    # keep the run tiny via env (also exercises env -> CLI merge)
    os.environ["AUTOMIND_SEED"] = "3"
    try:
        assert cli.main(["--quick", *argv]) == 0
    finally:
        del os.environ["AUTOMIND_SEED"]
    assert "EQUAL-BUDGET COMPARISON" in (out / "explanation.txt").read_text(encoding="utf-8")


@pytest.mark.parametrize("bad", [["--compare-random", "0"], ["--compare-random", "99"]])
def test_cli_rejects_bad_seed_counts(bad):
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(bad)


def test_env_outdir_used_when_flag_absent(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOMIND_OUTDIR", str(tmp_path / "from_env"))
    cfg = cli.build_config(cli.build_parser().parse_args(["--quick"]))
    assert cfg.outdir.endswith("from_env") and np.isfinite(cfg.seed)


# ---------- vectorised non-dominated sort == naive definition ----------
def _naive_fronts(objs):
    """Reference: repeatedly peel off the points that no remaining point dominates."""
    from automind.pareto import dominates

    remaining, fronts = list(range(len(objs))), []
    while remaining:
        front = [i for i in remaining if not any(dominates(objs[j], objs[i]) for j in remaining if j != i)]
        fronts.append(sorted(front))
        remaining = [i for i in remaining if i not in front]
    return fronts


@pytest.mark.parametrize("n,dim,decimals", [(1, 3, None), (40, 2, None), (60, 3, None), (60, 3, 1)])
def test_non_dominated_sort_matches_naive_definition(n, dim, decimals):
    from automind.pareto import non_dominated_sort

    pts = np.random.default_rng(n + dim).random((n, dim))
    if decimals is not None:
        pts = np.round(pts, decimals)  # many ties / duplicates
    objs = pts.tolist()
    assert [sorted(f) for f in non_dominated_sort(objs)] == _naive_fronts(objs)
