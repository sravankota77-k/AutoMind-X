"""All plots. Uses the non-interactive Agg backend so it works on any machine.

Accessibility: colour-blind-safe Okabe-Ito palette, redundant encoding (marker shape and line
style in addition to colour), large readable fonts, and a plain-text description (ALT_TEXT.md)
for every figure so screen-reader users get the same information.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from .explain import global_front, select_final

# Okabe-Ito colour-blind-safe palette
BLUE, ORANGE, GREEN, VERMILION, PURPLE, GOLD, GREY = (
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#D55E00",
    "#CC79A7",
    "#F0E442",
    "#999999",
)
plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11, "legend.fontsize": 10})


def _save(fig, outdir: str, name: str) -> str:
    """Tidy layout, write the figure to ``outdir/name`` at 140 dpi, close it and return the path."""
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, name)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_convergence(result, outdir: str) -> str:
    """Plot best/mean fitness, hypervolume and the mutation-rate trace per generation."""
    h = result.history
    gens = [x["gen"] for x in h]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].plot(gens, [x["best_fitness"] for x in h], "o-", label="best fitness", color=BLUE)
    ax[0].set_xlabel("generation")
    ax[0].set_ylabel("best fitness", color=BLUE)
    ax[0].set_title("Convergence")
    ax2 = ax[0].twinx()
    ax2.plot(gens, [x["hypervolume"] for x in h], "s--", color=VERMILION, label="hypervolume")
    ax2.set_ylabel("hypervolume (Pareto quality)", color=VERMILION)
    ax[1].plot(gens, [x["mean_mutation_rate"] for x in h], "o-", color=VERMILION, label="mean mutation-rate gene")
    ax[1].plot(gens, [x["multiplier"] for x in h], "s--", color=PURPLE, label="population multiplier")
    ax[1].set_xlabel("generation")
    ax[1].set_title("Self-adaptive mutation & meta-evolution")
    ax[1].legend()
    ax[0].grid(alpha=0.3)
    ax[1].grid(alpha=0.3)
    return _save(fig, outdir, "convergence.png")


def plot_fitness(result, outdir: str) -> str:
    """Plot the fitness distribution of the population per generation."""
    h = result.history
    gens = np.array([x["gen"] for x in h])
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for x in h:
        ax.scatter([x["gen"]] * len(x["pop_fitness"]), x["pop_fitness"], s=12, color=GREY, alpha=0.4)
    ax.fill_between(gens, [x["worst_fitness"] for x in h], [x["best_fitness"] for x in h], alpha=0.15, color=BLUE)
    ax.plot(gens, [x["best_fitness"] for x in h], "o-", label="best", color=GREEN)
    ax.plot(gens, [x["mean_fitness"] for x in h], "s--", label="mean", color=BLUE)
    ax.plot(gens, [x["worst_fitness"] for x in h], "^:", label="worst", color=VERMILION)
    ax.set_xlabel("generation")
    ax.set_ylabel("fitness  (higher = better)")
    ax.set_title("Fitness across generations")
    ax.legend()
    ax.grid(alpha=0.3)
    return _save(fig, outdir, "fitness.png")


def plot_acc_vs_runtime(result, outdir: str) -> str:
    """Scatter accuracy against training cost for every architecture, highlighting the Pareto front."""
    arch = result.archive
    front = global_front(result)
    chosen, _ = select_final(result)
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    sc = ax.scatter(
        [a.runtime * 1000 for a in arch],
        [a.accuracy for a in arch],
        c=[a.birth_gen for a in arch],
        cmap="cividis",
        s=28,
        alpha=0.8,
    )
    plt.colorbar(sc, ax=ax, label="generation discovered")
    f = sorted(front, key=lambda a: a.runtime)
    ax.plot(
        [a.runtime * 1000 for a in f],
        [a.accuracy for a in f],
        "--",
        color=VERMILION,
        lw=1.5,
        label="Pareto front (all 3 objectives)",
    )
    ax.scatter(
        [chosen.runtime * 1000],
        [chosen.accuracy],
        marker="*",
        s=320,
        c=GOLD,
        edgecolors="k",
        zorder=5,
        label="selected",
    )
    ax.set_xlabel("training cost (ms)")
    ax.set_ylabel("validation accuracy")
    ax.set_title("Accuracy vs Runtime")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    return _save(fig, outdir, "accuracy_vs_runtime.png")


def plot_pareto(result, outdir: str) -> str:
    """Four-panel view of the three-objective Pareto front (3-D plus three 2-D projections)."""
    arch = result.archive
    front = global_front(result)
    chosen, _ = select_final(result)
    front_ids = {id(f) for f in front}
    rest = [a for a in arch if id(a) not in front_ids]
    fig = plt.figure(figsize=(13, 9))
    ax = fig.add_subplot(2, 2, 1, projection="3d")
    ax.scatter(
        [a.runtime * 1000 for a in rest],
        [np.log10(a.params) for a in rest],
        [a.accuracy for a in rest],
        c=GREY,
        s=12,
        marker="o",
        label="dominated",
    )
    ax.scatter(
        [a.runtime * 1000 for a in front],
        [np.log10(a.params) for a in front],
        [a.accuracy for a in front],
        c=VERMILION,
        s=35,
        marker="D",
        label="Pareto front",
    )
    ax.scatter(
        [chosen.runtime * 1000],
        [np.log10(chosen.params)],
        [chosen.accuracy],
        c=GOLD,
        s=220,
        marker="*",
        edgecolors="k",
        label="selected",
    )
    ax.set_xlabel("cost (ms)")
    ax.set_ylabel("log10 params")
    ax.set_zlabel("accuracy")
    ax.set_title("3-objective Pareto front")
    ax.legend(fontsize=8)

    def proj(pos, fx, fy, xl, yl, title, logx=False):
        """Draw one 2-D projection of the objective space on ``ax``."""
        a = fig.add_subplot(2, 2, pos)
        a.scatter([fx(g) for g in rest], [fy(g) for g in rest], c=GREY, s=14, marker="o")
        a.scatter([fx(g) for g in front], [fy(g) for g in front], c=VERMILION, s=32, marker="D")
        a.scatter([fx(chosen)], [fy(chosen)], c=GOLD, s=220, marker="*", edgecolors="k", zorder=5)
        if logx:
            a.set_xscale("log")
        a.set_xlabel(xl)
        a.set_ylabel(yl)
        a.set_title(title)
        a.grid(alpha=0.3)

    proj(2, lambda g: g.params, lambda g: g.accuracy, "parameters (log)", "accuracy", "Accuracy vs Size", True)
    proj(3, lambda g: g.runtime * 1000, lambda g: g.params, "cost (ms)", "parameters", "Size vs Runtime")
    proj(4, lambda g: g.runtime * 1000, lambda g: g.accuracy, "cost (ms)", "accuracy", "Accuracy vs Runtime")
    return _save(fig, outdir, "pareto_front.png")


def describe_figures(result) -> dict[str, str]:
    """Plain-language description of each figure (used for ALT_TEXT.md)."""
    h = result.history
    chosen, front = select_final(result)
    gf = global_front(result)
    return {
        "convergence.png": (
            f"Two line charts over {len(h) - 1} generations. Left: best fitness rises from "
            f"{h[0]['best_fitness']:.3f} to {h[-1]['best_fitness']:.3f} while hypervolume goes from "
            f"{h[0]['hypervolume']:.3f} to {h[-1]['hypervolume']:.3f}. Right: mean mutation-rate gene goes from "
            f"{h[0]['mean_mutation_rate']:.3f} to {h[-1]['mean_mutation_rate']:.3f}; the population multiplier ends at "
            f"{h[-1]['multiplier']:.2f}."
        ),
        "fitness.png": (
            f"Best, mean and worst fitness per generation with individual population members as grey dots. "
            f"Best fitness ends at {h[-1]['best_fitness']:.3f}, mean at {h[-1]['mean_fitness']:.3f}, "
            f"worst at {h[-1]['worst_fitness']:.3f}."
        ),
        "accuracy_vs_runtime.png": (
            f"Scatter of {len(result.archive)} evaluated architectures (validation accuracy vs training cost), "
            f"coloured by generation discovered. A dashed line joins the {len(gf)} Pareto-optimal points. "
            f"A star marks the selected model: {chosen.describe()} with accuracy {chosen.accuracy:.3f}."
        ),
        "pareto_front.png": (
            f"Four panels showing the three-objective trade-off (accuracy, cost, parameters). Grey circles are "
            f"dominated architectures, diamonds are the {len(gf)} Pareto-optimal ones, the star is the selected "
            f"model with {chosen.params} parameters."
        ),
    }


def write_alt_text(result, outdir: str) -> str:
    """Write ``ALT_TEXT.md`` with a text description of every figure (screen-reader friendly)."""
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "ALT_TEXT.md")
    lines: list[str] = ["# Figure descriptions (text alternatives)\n"]
    for name, text in describe_figures(result).items():
        lines.append(f"## {name}\n{text}\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def make_all(result, outdir: str) -> list[str]:
    """Create all four figures plus ``ALT_TEXT.md``; return their paths."""
    paths = [
        plot_convergence(result, outdir),
        plot_fitness(result, outdir),
        plot_acc_vs_runtime(result, outdir),
        plot_pareto(result, outdir),
    ]
    paths.append(write_alt_text(result, outdir))
    return paths
