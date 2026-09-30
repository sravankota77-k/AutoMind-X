# Security notes

* **No secrets.** The project needs no API keys, tokens or credentials and makes no network calls.
  `.env.example` documents the only environment variables (`AUTOMIND_*` tuning knobs, no credentials).
  `.gitignore` excludes `.env`, `.env.*` (except the template), `*.pem` and `*.key`; CI runs a
  secret scan (gitleaks) and `pre-commit` runs `detect-private-key`.
* **No unsafe deserialisation.** No `pickle`, `eval`, `exec`, `subprocess` or `shell=True` anywhere.
  Results and the optional cache are plain CSV / TXT / JSON / PNG. A cache file is only trusted if its
  settings fingerprint matches; a corrupt or foreign file is ignored, never executed.
* **Input validation.** `Config.validate()` rejects out-of-range settings (including values from
  environment variables); the CLI enforces positive integers, bounds `--compare-random`, and refuses an
  `--outdir` that is an existing file. Invalid input exits with code 2.
* **Bounded resources.** Search-space sizes, iterations and population are bounded by the config;
  parallelism is capped by `--n-jobs`.
* **Supply chain.** Dependencies are version-ranged in `requirements.txt`; Dependabot and `pip-audit`
  run in CI; the Docker image runs as a non-root user.
* **Static analysis.** `bandit` runs in CI and via `make security`.

## Reporting a vulnerability

Please open a private security advisory on the repository (or contact the maintainers privately).
Do not post exploit details in a public issue. We aim to acknowledge reports within 7 days.
