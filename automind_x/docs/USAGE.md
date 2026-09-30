# Usage guide

## Install

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt                        # runtime only
pip install -r requirements-dev.txt                    # + tests, lint, types, security scan
```

Python 3.10 or newer. No API keys or accounts are needed.

## Five common tasks

| I want to... | Command |
|---|---|
| Try it in seconds | `python main.py --quick` |
| Run the full search and get every output | `python main.py` |
| Pick the best compromise without choosing weights | `python main.py --select knee` |
| Get the best model under a size limit | `python main.py --max-params 2000` |
| Check it beats random search | `python main.py --quick --compare-random 5` |
| Re-run instantly after a first run | `python main.py --cache .cache/m.json` |
| See where time goes | `python main.py --quick --profile` (writes `profile.txt`) |

## Configuration precedence

`CLI flag` > `AUTOMIND_*` environment variable > built-in default. Copy `.env.example` to `.env`,
edit, then `set -a; source .env; set +a`. See the file for every variable.

## Using it as a library

```python
from automind.config import Config
from automind.evolution import AutoMindX
from automind.explain import explain, select_final

cfg = Config(pop_size=10, generations=6, seed=1, n_jobs=1)
result = AutoMindX(cfg, verbose=False).run()
chosen, front = select_final(result)
print(chosen.describe(), chosen.accuracy, chosen.params)
print(explain(result))
```

More in [`examples/`](../examples).

## Outputs (`outputs/` by default)

| File | Contents |
|---|---|
| `explanation.txt` | Plain-language "why this model", hold-out test, latency, optional random-search comparison |
| `report.html` | Accessible report (landmarks, alt text, tables with captions, no JavaScript) |
| `summary.json` (`--json`) | Machine-readable: selected model, hold-out, latency, timing, cache statistics |
| `all_architectures.csv` | Every architecture evaluated |
| `*.png`, `ALT_TEXT.md` | Four figures and their text descriptions |
| `profile.txt` (`--profile`) | Top 25 calls by cumulative time |

## Exit codes

`0` success, `2` invalid input (bad flag value or bad configuration).

## Reading the results

* **Accuracy** is measured on validation data during search and on the test split for the final number.
* **Cost** is estimated training cost in pseudo-seconds (proxy) or measured seconds (wallclock).
* **Hypervolume** (0-1, higher is better) measures how much of the objective space the Pareto front
  dominates; it rises when the search finds better trade-offs.
