"""Multi-objective machinery (NSGA-II): dominance, sorting, crowding, hypervolume.

All objectives are minimised. Dominance is computed with NumPy broadcasting (one O(n^2 * m)
vectorised pass instead of a Python double loop), which keeps large populations fast.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

INF = float("inf")


def dominates(a: Sequence[float], b: Sequence[float]) -> bool:
    """a dominates b if it is no worse in every objective and better in at least one (minimise)."""
    return all(x <= y for x, y in zip(a, b, strict=False)) and any(x < y for x, y in zip(a, b, strict=False))


def dominance_matrix(objs: Sequence[Sequence[float]]) -> np.ndarray:
    """Boolean matrix D where D[i, j] is True iff objs[i] dominates objs[j]."""
    arr = np.asarray(objs, dtype=float)
    if arr.size == 0:
        return np.zeros((0, 0), dtype=bool)
    le = np.all(arr[:, None, :] <= arr[None, :, :], axis=2)
    lt = np.any(arr[:, None, :] < arr[None, :, :], axis=2)
    return le & lt


def non_dominated_sort(objs: Sequence[Sequence[float]]) -> list[list[int]]:
    """Split indices into fronts: front 0 = Pareto-optimal, front 1 = next best, ..."""
    n = len(objs)
    if n == 0:
        return []
    dom = dominance_matrix(objs)
    dom_count = dom.sum(axis=0).astype(int)  # how many dominate me
    fronts: list[list[int]] = []
    current = [i for i in range(n) if dom_count[i] == 0]
    while current:
        fronts.append(current)
        nxt: list[int] = []
        for i in current:
            row = dom[i]
            dom_count[row] -= 1  # everything i dominates loses one dominator (vectorised, no inner loop)
            nxt.extend(np.flatnonzero(row & (dom_count == 0)).tolist())
        current = nxt
    return fronts


def crowding_distance(objs: Sequence[Sequence[float]], front: Sequence[int]) -> dict[int, float]:
    """How isolated each member is inside its front (bigger = more valuable for diversity)."""
    dist = dict.fromkeys(front, 0.0)
    if len(front) <= 2:
        return dict.fromkeys(front, INF)
    for k in range(len(objs[0])):
        srt = sorted(front, key=lambda i: objs[i][k])
        lo, hi = objs[srt[0]][k], objs[srt[-1]][k]
        dist[srt[0]] = dist[srt[-1]] = INF
        if hi - lo < 1e-12:
            continue
        for j in range(1, len(srt) - 1):
            dist[srt[j]] += (objs[srt[j + 1]][k] - objs[srt[j - 1]][k]) / (hi - lo)
    return dist


def environmental_selection(pool: list, size: int) -> list:
    """NSGA-II survival: fill the next generation front by front; break ties with crowding."""
    objs = [g.objectives() for g in pool]
    survivors: list = []
    for rank, front in enumerate(non_dominated_sort(objs)):
        crowd = crowding_distance(objs, front)
        for i in front:
            pool[i].rank, pool[i].crowding = rank, crowd[i]
        if len(survivors) + len(front) <= size:
            survivors += [pool[i] for i in front]
        else:
            need = size - len(survivors)
            survivors += [pool[i] for i in sorted(front, key=lambda i: -crowd[i])[:need]]
            break
    return survivors


def hypervolume_mc(points: Sequence[Sequence[float]], n_samples: int = 20000, seed: int = 0) -> float:
    """Monte-Carlo hypervolume in the unit cube (reference point = all ones, minimisation).
    Fixed seed => the same sample cloud each call => values comparable across generations."""
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        return 0.0
    samples = np.random.default_rng(seed).random((n_samples, pts.shape[1]))
    covered = np.zeros(n_samples, dtype=bool)
    for p in pts:
        covered |= np.all(samples >= p, axis=1)
    return float(covered.mean())
