# API reference (short)

Every public function has a docstring; use `help(automind.<module>)` for details.

| Import | Purpose |
|---|---|
| `automind.config.Config` | validated settings; `Config.from_env()` |
| `automind.evolution.AutoMindX(cfg).run()` | run the search, returns `Result` |
| `automind.explain.select_final(result)` | `(chosen_genome, pareto_front)` honouring budgets |
| `automind.explain.explain(result)` / `holdout_report(result, chosen)` | plain-language reports |
| `automind.evaluator.Evaluator` | train + cache + hold-out + `latency(genome)` |
| `automind.baselines.compare_with_random(cfg, seeds)` | equal-budget paired comparison |
| `automind.pareto` | `non_dominated_sort`, `crowding_distance`, `environmental_selection`, `hypervolume_mc` |
| `automind.report.write_html`, `automind.visualize.make_all` | report and figures |

CLI: `python main.py --help`. Data model: `Genome` (`layers`, `activation`, `lr`, `mutation_rate`, plus
measured `accuracy`, `runtime`, `params`, `fitness`).
