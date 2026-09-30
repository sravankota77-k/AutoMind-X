"""Accessible HTML report: semantic markup, text alternatives, keyboard/screen-reader friendly.

* one <h1>, ordered headings, <main>/<nav> landmarks, skip link, ``lang`` attribute
* every figure has meaningful alt text plus a visible caption
* data tables use <caption> and scoped <th>; contrast >= 7:1; no colour-only meaning; no JavaScript
* all user-derived text is HTML-escaped
"""

from __future__ import annotations

import html
import os

from .explain import explain, global_front, holdout_report, select_final
from .visualize import describe_figures

_CSS = """
body{font:18px/1.6 system-ui,sans-serif;max-width:60rem;margin:0 auto;padding:1rem;color:#111;background:#fff}
a{color:#0033cc}.skip{position:absolute;left:-999px}.skip:focus{left:0;background:#ffd;padding:.5rem}
table{border-collapse:collapse;width:100%}th,td{border:1px solid #444;padding:.35rem .5rem;text-align:left}
caption{font-weight:bold;text-align:left;padding:.4rem 0}pre{background:#f3f3f3;padding:1rem;overflow-x:auto}
figure{margin:1.5rem 0}img{max-width:100%;height:auto}:focus-visible{outline:3px solid #b30000;outline-offset:2px}
"""


def build_html(result) -> str:
    """Return the complete accessible HTML report as a string."""
    chosen, front = select_final(result)
    hold = holdout_report(result, chosen)
    esc = html.escape
    rows = "".join(
        f"<tr><th scope='row'>{esc(g.describe())}</th><td>{g.accuracy:.3f}</td>"
        f"<td>{g.runtime * 1000:.1f}</td><td>{g.params}</td></tr>"
        for g in sorted(front, key=lambda g: -g.accuracy)
    )
    figs = "".join(
        f"<figure><img src='{esc(name)}' alt='{esc(text)}'><figcaption>{esc(text)}</figcaption></figure>"
        for name, text in describe_figures(result).items()
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AutoMind-X report</title><style>{_CSS}</style></head><body>
<a class="skip" href="#main">Skip to main content</a>
<header><h1>AutoMind-X report</h1>
<nav aria-label="Sections"><a href="#selected">Selected model</a> | <a href="#front">Pareto finalists</a> |
<a href="#figures">Figures</a> | <a href="#explanation">Full explanation</a></nav></header>
<main id="main">
<section id="selected"><h2>Selected model</h2>
<p><strong>{esc(chosen.describe())}</strong> - validation accuracy {chosen.accuracy:.3f},
{chosen.params} parameters, found in generation {chosen.birth_gen}. The Pareto archive holds
{len(global_front(result))} non-dominated architectures out of {len(result.archive)} tried.</p>
<pre>{esc(hold)}</pre></section>
<section id="front"><h2>Pareto-optimal finalists</h2>
<table><caption>Finalists sorted by validation accuracy (cost in milliseconds of estimated training)</caption>
<thead><tr><th scope="col">Architecture</th><th scope="col">Accuracy</th><th scope="col">Cost (ms)</th>
<th scope="col">Parameters</th></tr></thead><tbody>{rows}</tbody></table></section>
<section id="figures"><h2>Figures</h2>{figs}</section>
<section id="explanation"><h2>Full explanation</h2><pre>{esc(explain(result))}</pre></section>
</main></body></html>
"""


def write_html(result, outdir: str) -> str:
    """Write ``report.html`` into ``outdir`` and return its path."""
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "report.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(build_html(result))
    return path
