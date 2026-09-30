"""Explainability: WHY was the final architecture selected, and how does it compare with the past?"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .evaluator import fitness_terms, normalised_objectives
from .pareto import dominates, non_dominated_sort


def knee_point(front, cfg):
    """Pareto member closest (Euclidean) to the ideal point in normalised objective space.
    Needs no weights: it is the 'best compromise' of the front."""

    def dist(g):
        """Distance of a genome from the ideal point (0, 0, 0) in normalised objective space."""
        return float(np.linalg.norm(normalised_objectives(g.accuracy, g.runtime, g.params, cfg)))

    return min(front, key=dist)


def within_budget(g, cfg) -> bool:
    """True if the genome respects the optional deployment budgets."""
    return (cfg.max_params is None or g.params <= cfg.max_params) and (
        cfg.max_cost is None or g.runtime <= cfg.max_cost
    )


def select_final(result, strategy=None):
    """Final pick from the Pareto-optimal individuals, honouring optional deployment budgets.

    balanced : best weighted fitness (default)
    knee     : closest to the ideal point (weight-free)

    Budget handling: candidates over ``max_params`` / ``max_cost`` are excluded. Feasible models
    are searched in the whole archive's Pareto front if none of the final population fits. If
    nothing at all fits, the least-violating (smallest, then cheapest) model is returned.
    """
    cfg = result.cfg
    strategy = strategy or getattr(cfg, "selection", "balanced")
    front = [g for g in result.final_population if g.rank == 0]
    feasible = [g for g in front if within_budget(g, cfg)]
    if not feasible and (cfg.max_params is not None or cfg.max_cost is not None):
        feasible = [g for g in global_front(result) if within_budget(g, cfg)]
        if not feasible:
            return min(front, key=lambda g: (g.params, g.runtime)), front
    pool = feasible or front
    if strategy == "knee":
        return knee_point(pool, cfg), front
    return max(pool, key=lambda g: g.fitness), front


def global_front(result):
    """Pareto front (rank 0) of *every* architecture ever evaluated, not just the final population."""
    objs = [g.objectives() for g in result.archive]
    return [result.archive[i] for i in non_dominated_sort(objs)[0]]


def _row(g):
    """One aligned text line describing a genome: architecture, accuracy, cost, parameters."""
    return f"{g.describe():<34} acc={g.accuracy:.3f}  cost={g.runtime * 1000:7.1f} ms  params={g.params:>6}"


def _ratio(a, b):
    """Format ``a / b`` as ``x1.23`` (``n/a`` when ``b`` is zero)."""
    return f"x{a / b:.2f}" if b else "n/a"


def _lineage(result, g):
    """Walk back through the fitter parent at every step."""
    chain = [g]
    while chain[-1].parents:
        parents = [result.registry[u] for u in chain[-1].parents if u in result.registry]
        if not parents:
            break
        chain.append(max(parents, key=lambda p: p.fitness))
    return list(reversed(chain))


def _header(result, chosen) -> list[str]:
    """Title block: the selected model, budget status and discovery generation."""
    cfg = result.cfg
    lines = ["=" * 78, "AUTOMIND-X  |  WHY THIS ARCHITECTURE WAS SELECTED", "=" * 78, f"\nSELECTED: {_row(chosen)}"]
    if cfg.max_params is not None or cfg.max_cost is not None:
        ok = within_budget(chosen, cfg)
        lines.append(
            f"BUDGET  : max_params={cfg.max_params}, max_cost={cfg.max_cost}  ->  "
            f"{'satisfied' if ok else 'NOT satisfiable - least-violating model shown'}"
        )
    lines.append(f"          discovered in generation {chosen.birth_gen}")
    return lines


def _section_qualifies(result, chosen, front) -> list[str]:
    """[1] Why the model is Pareto-optimal and why it won among the finalists."""
    cfg, archive = result.cfg, result.archive
    beaten = sum(dominates(chosen.objectives(), a.objectives()) for a in archive)
    reward, t_pen, p_pen = fitness_terms(chosen.accuracy, chosen.runtime, chosen.params, cfg)
    how = "closest to the ideal point (knee)" if cfg.selection == "knee" else "the best balanced score"
    lines = [
        "\n[1] WHY IT QUALIFIES",
        f"  - It is Pareto-optimal: none of the {len(archive)} architectures tried is better in all three",
        f"    objectives at once. It strictly dominates {beaten} of them.",
        f"  - Among the {len(front)} Pareto-optimal finalists it is chosen for {how}:",
        f"      score = {reward:.3f} (accuracy x{cfg.w_acc}) - {t_pen:.3f} (time penalty) - {p_pen:.3f} (size penalty)"
        f" = {chosen.fitness:.3f}",
    ]
    for name, key, low_good in (
        ("accuracy", lambda g: g.accuracy, False),
        ("speed", lambda g: g.runtime, True),
        ("compactness", lambda g: g.params, True),
    ):
        ordered = sorted(front, key=key, reverse=not low_good)
        lines.append(f"      {name:<12} rank {ordered.index(chosen) + 1} of {len(front)} finalists")
    return lines


def _section_tradeoffs(chosen, front) -> list[str]:
    """[2] What you would give up or gain by picking another finalist."""
    lines = ["\n[2] TRADE-OFFS vs OTHER FINALISTS (what you would give up / gain by picking them)"]
    for label, alt in (
        ("most accurate", max(front, key=lambda g: g.accuracy)),
        ("fastest", min(front, key=lambda g: g.runtime)),
        ("smallest", min(front, key=lambda g: g.params)),
    ):
        if alt is chosen:
            lines.append(f"  - {label:<14}: the selected model IS the {label} finalist.")
        else:
            lines.append(
                f"  - {label:<14}: {alt.describe()}  ->  accuracy {alt.accuracy - chosen.accuracy:+.3f}, "
                f"runtime {_ratio(alt.runtime, chosen.runtime)}, params {_ratio(alt.params, chosen.params)}"
            )
    return lines


def _section_generations(result, chosen) -> list[str]:
    """[3] The selected model compared with the best individual of earlier generations."""
    lines = ["\n[3] COMPARISON WITH PREVIOUS GENERATIONS (best balanced individual of each generation)"]
    gens = sorted(result.gen_best)
    shown = sorted({0, gens[-1], *np.linspace(0, gens[-1], 5).astype(int).tolist()})
    for gnum in shown:
        lines.append(f"  gen {gnum:>2}: {_row(result.gen_best[gnum])}  fit={result.gen_best[gnum].fitness:+.3f}")
    g0, gprev = result.gen_best[0], result.gen_best[gens[-2]] if len(gens) > 1 else result.gen_best[0]
    lines.append(
        f"  vs generation 0 best : accuracy {chosen.accuracy - g0.accuracy:+.3f}, "
        f"runtime {_ratio(chosen.runtime, g0.runtime)}, params {_ratio(chosen.params, g0.params)}, "
        f"fitness {chosen.fitness - g0.fitness:+.3f}"
    )
    lines.append(
        f"  vs previous generation: accuracy {chosen.accuracy - gprev.accuracy:+.3f}, "
        f"runtime {_ratio(chosen.runtime, gprev.runtime)}, params {_ratio(chosen.params, gprev.params)}, "
        f"fitness {chosen.fitness - gprev.fitness:+.3f}"
    )
    h0, hN = result.history[0], result.history[-1]
    lines.append(
        f"  Pareto quality (hypervolume): {h0['hypervolume']:.3f} -> {hN['hypervolume']:.3f}   "
        f"| mean accuracy of population: {h0['mean_accuracy']:.3f} -> {hN['mean_accuracy']:.3f}"
    )
    return lines


def _section_lineage(result, chosen) -> list[str]:
    """[4] Ancestry of the selected model through its fitter parent."""
    lines = ["\n[4] HOW IT EVOLVED (ancestry through the fitter parent)"]
    for anc in _lineage(result, chosen):
        lines.append(f"  gen {anc.birth_gen:>2}: {anc.describe():<34} {'; '.join(anc.notes) or '-'}")
    return lines


def _section_learned(result) -> list[str]:
    """[5] Which activations and depths worked, across every architecture tried."""
    lines = ["\n[5] WHAT THE SEARCH LEARNED (across all architectures tried)"]
    by_act: dict = defaultdict(list)
    by_depth: dict = defaultdict(list)
    for a in result.archive:
        by_act[a.activation].append(a)
        by_depth[len(a.layers)].append(a)
    on_front = {id(f) for f in global_front(result)}
    for act, items in sorted(by_act.items()):
        lines.append(
            f"  activation {act:<9}: tried {len(items):>3}, mean acc {np.mean([i.accuracy for i in items]):.3f}, "
            f"on Pareto front {sum(id(i) in on_front for i in items)}"
        )
    for d, items in sorted(by_depth.items()):
        lines.append(
            f"  {d} hidden layer(s)   : tried {len(items):>3}, mean acc {np.mean([i.accuracy for i in items]):.3f}, "
            f"on Pareto front {sum(id(i) in on_front for i in items)}"
        )
    return lines


def _section_meta(result, chosen) -> list[str]:
    """[6] How the self-adaptive mutation behaved during the run."""
    mult = [h["multiplier"] for h in result.history]
    acts = [a for a, _ in result.meta.log]
    return [
        "\n[6] META-EVOLUTION (self-adaptive mutation)",
        f"  - population multiplier ranged {min(mult):.2f}..{max(mult):.2f}; "
        f"{acts.count('explore')} generations triggered EXPLORE (stagnation -> mutate more), "
        f"{acts.count('exploit')} triggered EXPLOIT (improvement -> mutate less).",
        f"  - mean evolved mutation-rate gene: {result.history[0]['mean_mutation_rate']:.3f} -> "
        f"{result.history[-1]['mean_mutation_rate']:.3f}; selected model carries gene = {chosen.mutation_rate:.3f}",
    ]


def explain(result) -> str:
    """Build the plain-language report explaining why the final architecture was selected."""
    chosen, front = select_final(result)
    lines = [
        *_header(result, chosen),
        *_section_qualifies(result, chosen, front),
        *_section_tradeoffs(chosen, front),
        *_section_generations(result, chosen),
        *_section_lineage(result, chosen),
        *_section_learned(result),
        *_section_meta(result, chosen),
    ]
    return "\n".join(lines)


def holdout_report(result, chosen) -> str:
    """Honest final check on the untouched test split, versus a default MLP baseline."""
    ev = result.evaluator
    acc, rt, params = ev.holdout(chosen)
    base, (b_acc, b_rt, b_params) = ev.baseline()
    lat, b_lat = ev.latency(chosen), ev.latency(base)
    return "\n".join(
        [
            "\nHOLD-OUT TEST (data the evolution never saw)",
            f"  AutoMind-X  {chosen.describe():<26} test acc={acc:.3f}  cost={rt * 1000:6.1f} ms  params={params}",
            f"  Baseline    {base.describe():<26} test acc={b_acc:.3f}  cost={b_rt * 1000:6.1f} ms  params={b_params}",
            f"  -> {_ratio(params, b_params)} parameters, {_ratio(rt, b_rt)} training cost, "
            f"accuracy {acc - b_acc:+.3f} vs baseline",
            "",
            "INFERENCE LATENCY (edge-deployment view, this machine)",
            f"  AutoMind-X  batch {lat.batch_ms_per_1k:6.2f} ms / 1000 samples | single sample "
            f"p50 {lat.single_p50_ms:.3f} ms, p95 {lat.single_p95_ms:.3f} ms",
            f"  Baseline    batch {b_lat.batch_ms_per_1k:6.2f} ms / 1000 samples | single sample "
            f"p50 {b_lat.single_p50_ms:.3f} ms, p95 {b_lat.single_p95_ms:.3f} ms",
        ]
    )
