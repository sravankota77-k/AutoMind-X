"""Extra tests: config validation, determinism, proxy runtime, parallel==serial, selection, CLI, outputs."""

import csv
import json

import numpy as np
import pytest

import main as cli
from automind.config import Config
from automind.dataset import make_dataset
from automind.evaluator import Evaluator, proxy_runtime
from automind.evolution import AutoMindX
from automind.explain import explain, global_front, holdout_report, knee_point, select_final
from automind.genome import Genome, random_genome
from automind.operators import mutate, tournament_select
from automind.pareto import crowding_distance, dominates, hypervolume_mc, non_dominated_sort


def tiny(**kw):
    return Config(n_samples=300, pop_size=6, generations=3, max_iter=15, **kw)


# ---------- config validation ----------
@pytest.mark.parametrize(
    "bad",
    [
        dict(pop_size=1),
        dict(generations=0),
        dict(crossover_rate=1.5),
        dict(runtime_metric="x"),
        dict(selection="x"),
        dict(min_layers=5, max_layers=2),
        dict(n_jobs=0),
        dict(mut_min=0.0),
        dict(n_informative=30),
        dict(time_ref=0),
        dict(lr_choices=(-1.0,)),
    ],
)
def test_config_rejects_invalid(bad):
    with pytest.raises(ValueError, match="Invalid Config"):
        Config(**bad)


def test_config_defaults_valid():
    Config().validate()


# ---------- proxy runtime ----------
def test_proxy_runtime_scales_and_is_deterministic():
    a = proxy_runtime(500, 1000, 10, 1e9)
    assert a == proxy_runtime(500, 1000, 10, 1e9)
    assert proxy_runtime(1000, 1000, 10, 1e9) == pytest.approx(2 * a)
    assert proxy_runtime(500, 1000, 0, 1e9) > 0  # epochs clamped to >= 1


# ---------- full reproducibility (proxy metric) ----------
def test_full_run_is_reproducible():
    runs = []
    for _ in range(2):
        r = AutoMindX(tiny(seed=3, n_jobs=1), verbose=False).run()
        runs.append(([h["best_fitness"] for h in r.history], sorted(g.key() for g in r.archive)))
    assert runs[0] == runs[1]


def test_different_seed_changes_search():
    a = AutoMindX(tiny(seed=1, n_jobs=1), verbose=False).run()
    b = AutoMindX(tiny(seed=2, n_jobs=1), verbose=False).run()
    assert sorted(g.key() for g in a.archive) != sorted(g.key() for g in b.archive)


def test_parallel_matches_serial():
    cfg_s, cfg_p = tiny(n_jobs=1), tiny(n_jobs=2)
    rng = np.random.default_rng(0)
    gs = [random_genome(cfg_s, rng) for _ in range(5)]
    gp = [g.clone() for g in gs]
    Evaluator(make_dataset(cfg_s), cfg_s).evaluate_many(gs)
    Evaluator(make_dataset(cfg_p), cfg_p).evaluate_many(gp)
    assert [(g.accuracy, g.runtime, g.params) for g in gs] == [(g.accuracy, g.runtime, g.params) for g in gp]


def test_batch_dedup_trains_each_architecture_once():
    cfg = tiny(n_jobs=1)
    ev = Evaluator(make_dataset(cfg), cfg)
    g = random_genome(cfg, np.random.default_rng(1))
    ev.evaluate_many([g.clone(), g.clone(), g.clone()])
    assert ev.n_trainings == 1 and ev.cache_hits == 2


def test_wallclock_metric_runs():
    cfg = tiny(runtime_metric="wallclock")
    ev = Evaluator(make_dataset(cfg), cfg)
    g = ev.evaluate(random_genome(cfg, np.random.default_rng(2)))
    assert g.runtime == g.wall_time > 0


# ---------- genome / operators ----------
def test_clone_is_independent_and_identity_equality():
    g = Genome([8, 4], "relu", 0.01, 0.3)
    c = g.clone()
    c.layers.append(16)
    c.notes.append("x")
    assert g.layers == [8, 4] and g.notes == [] and g != c and g == g


def test_tournament_prefers_lower_rank():
    a = Genome([4], "relu", 0.01, 0.3, rank=0, crowding=1.0)
    b = Genome([4], "tanh", 0.01, 0.3, rank=3, crowding=9.0)
    assert tournament_select([a, b], 2, np.random.default_rng(0)) is a


def test_mutation_records_notes_even_at_zero_rate():
    cfg, rng = Config(), np.random.default_rng(5)
    g = random_genome(cfg, rng)
    n = len(g.notes)
    mutate(g, 0.0, cfg, rng)
    assert len(g.notes) > n


# ---------- pareto edge cases ----------
def test_pareto_edge_cases():
    assert non_dominated_sort([]) == []
    assert non_dominated_sort([(1, 1)]) == [[0]]
    assert crowding_distance([(1, 1), (2, 2)], [0, 1]) == {0: float("inf"), 1: float("inf")}
    assert hypervolume_mc([]) == 0.0
    assert hypervolume_mc([(0.0, 0.0, 0.0)]) == pytest.approx(1.0)


# ---------- selection + explainability ----------
@pytest.fixture(scope="module")
def result():
    return AutoMindX(tiny(n_jobs=1), verbose=False).run()


def test_select_final_strategies(result):
    balanced, front = select_final(result, "balanced")
    knee, _ = select_final(result, "knee")
    assert balanced.rank == 0 and knee.rank == 0
    assert balanced is max(front, key=lambda g: g.fitness)
    assert knee is knee_point(front, result.cfg)


def test_global_front_is_non_dominated(result):
    front = global_front(result)
    assert front and not any(dominates(a.objectives(), b.objectives()) for a in front for b in front)


def test_explanation_sections_and_holdout(result):
    text = explain(result) + holdout_report(result, select_final(result)[0])
    for marker in (
        "[1] WHY IT QUALIFIES",
        "[2] TRADE-OFFS",
        "[3] COMPARISON",
        "[4] HOW IT EVOLVED",
        "[5] WHAT THE SEARCH LEARNED",
        "[6] META-EVOLUTION",
        "HOLD-OUT TEST",
        "Baseline",
    ):
        assert marker in text


def test_archive_has_unique_architectures(result):
    keys = [g.key() for g in result.archive]
    assert len(keys) == len(set(keys))


# ---------- CLI + outputs ----------
def test_cli_end_to_end(tmp_path):
    out = tmp_path / "out"
    code = cli.main(
        ["--quick", "--generations", "2", "--pop", "4", "--outdir", str(out), "--n-jobs", "1", "--json", "--quiet"]
    )
    assert code == 0
    for name in (
        "explanation.txt",
        "all_architectures.csv",
        "summary.json",
        "ALT_TEXT.md",
        "convergence.png",
        "fitness.png",
        "accuracy_vs_runtime.png",
        "pareto_front.png",
    ):
        assert (out / name).stat().st_size > 0, name
    summary = json.loads((out / "summary.json").read_text())
    assert 0.0 <= summary["holdout"]["test_accuracy"] <= 1.0
    with open(out / "all_architectures.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows and {"uid", "accuracy", "params", "fitness"} <= set(rows[0])


def test_cli_no_plots(tmp_path):
    out = tmp_path / "o"
    assert (
        cli.main(
            [
                "--quick",
                "--generations",
                "1",
                "--pop",
                "3",
                "--outdir",
                str(out),
                "--n-jobs",
                "1",
                "--no-plots",
                "--quiet",
            ]
        )
        == 0
    )
    assert not (out / "convergence.png").exists() and (out / "explanation.txt").exists()


@pytest.mark.parametrize("args", [["--generations", "0"], ["--pop", "-3"], ["--runtime-metric", "bogus"]])
def test_cli_rejects_bad_args(args):
    with pytest.raises(SystemExit) as e:
        cli.main(args)
    assert e.value.code == 2


def test_cli_rejects_pop_of_one(capsys):
    assert cli.main(["--pop", "1", "--quiet"]) == 2
    assert "Invalid Config" in capsys.readouterr().err


def test_cli_rejects_file_as_outdir(tmp_path):
    f = tmp_path / "afile"
    f.write_text("x")
    with pytest.raises(SystemExit):
        cli.main(["--outdir", str(f)])
