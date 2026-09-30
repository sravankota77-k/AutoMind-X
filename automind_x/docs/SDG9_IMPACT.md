# Alignment with UN SDG 9: Industry, Innovation and Infrastructure

**Primary target: 9.4** - upgrade infrastructure and industries to make them sustainable, with greater
resource-use efficiency and adoption of clean and environmentally sound technologies.
**Supporting target: 9.5** - enhance scientific research and upgrade technological capabilities.

## The link, stated plainly

Choosing a neural-network architecture is usually done by trial and error or by an AutoML tool that
returns one opaque "best" model, typically the most accurate and often far larger and more expensive to
train than necessary. AutoMind-X makes the **accuracy / training-cost / model-size trade-off explicit**
and lets an engineer choose a model that fits a real budget (`--max-params`, `--max-cost`), which is
what resource-constrained industrial and edge deployments need.

## Evidence (measured, see [EVALUATION.md](EVALUATION.md))

| SDG 9.4 concern | AutoMind-X measurement |
|---|---|
| Resource efficiency (compute) | chosen model trains at **~1/32 of the estimated cost** of a default MLP |
| Resource efficiency (memory) | **12.3x fewer parameters** (195 vs 2403) |
| Cost of accuracy | about **2 points** lower test accuracy - reported, not hidden |
| Avoiding wasted compute in the search itself | duplicate architectures trained once; `--cache` makes repeat runs ~80x faster |
| Reproducible, auditable engineering (9.5) | deterministic seeds, plain-language explanation, honest random-search comparison |

## How outputs map to the target

* `--max-params` / `--max-cost` -> "give me the best model that fits this device or budget".
* `explanation.txt` -> each decision is explained, supporting accountable industrial use.
* Compute proxy (FLOPs based) -> a hardware-independent way to compare training cost.
* Accessible `report.html` and alt text -> the results are usable by people with disabilities.

## What this does not claim

The project does not measure energy or CO2 directly; training cost is a FLOPs proxy. The benefit is
an enabler (smaller, cheaper models become easy to find), not a verified field impact. Results are on a
synthetic benchmark; real-world savings depend on the application.
