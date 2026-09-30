"""Smallest useful example: run a search, print the chosen model and its explanation.

python examples/quickstart.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from automind.config import Config
from automind.evolution import AutoMindX
from automind.explain import explain, holdout_report, select_final

if __name__ == "__main__":
    cfg = Config(pop_size=8, generations=4, n_samples=1000, max_iter=40, seed=1, n_jobs=1)
    result = AutoMindX(cfg, verbose=False).run()
    chosen, front = select_final(result)
    print(f"Chosen: {chosen.describe()}  (val acc {chosen.accuracy:.3f}, {chosen.params} params)")
    print(f"Pareto finalists: {len(front)}")
    print(holdout_report(result, chosen))
    print()
    print(explain(result))
