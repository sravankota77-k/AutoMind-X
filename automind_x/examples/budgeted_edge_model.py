"""Find the best model that fits a tiny deployment budget (edge / low-power device scenario).

python examples/budgeted_edge_model.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from automind.config import Config
from automind.evolution import AutoMindX
from automind.explain import select_final, within_budget

if __name__ == "__main__":
    cfg = Config(pop_size=8, generations=5, n_samples=1000, max_iter=40, max_params=500, selection="knee", n_jobs=1)
    result = AutoMindX(cfg, verbose=False).run()
    chosen, _ = select_final(result)
    lat = result.evaluator.latency(chosen)
    print(f"Model: {chosen.describe()} | {chosen.params} parameters | budget met: {within_budget(chosen, cfg)}")
    print(f"Inference: {lat.single_p50_ms:.3f} ms per sample (median), {lat.batch_ms_per_1k:.2f} ms per 1000 samples")
