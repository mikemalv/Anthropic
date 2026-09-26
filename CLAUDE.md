# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project purpose

A Python demo project for **FinTechCo** (a fictional financial services company) that shows how
Claude Code speeds up quant and data-science work on public macroeconomic data from the
Federal Reserve Economic Data (FRED) API.

The demo runs live in a 10–15 minute segment of an executive meeting. Favor work that is
**fast, reliable, and easy to explain** over work that is clever. Every change should run
end to end in seconds.

Narrative arc (one question, answered end to end):
1. **Acquire**: pull series from FRED (e.g. `CPIAUCSL`, `UNRATE`, `FEDFUNDS`, `VIXCLS`, `DGS10`).
2. **Clean & explore**: align frequencies, handle missing values, compute YoY changes.
3. **Analyze / model**: e.g. the flattening Phillips Curve, or how Fed policy changes relate to volatility.
4. **Present**: an interactive dashboard a client-facing team could use.

## Audience

- **CTO**: security-conscious and focused on regulatory compliance. Cares about auditability, secrets handling, and reproducibility.
- **Head of Digital Transformation**: cares about productivity and time-to-value.
- **FinTechCo teams**: software engineers (Python/TS/Java), data scientists (Jupyter, pandas, scikit-learn), and SREs (Python, shell, Go).

Write code and output that a data scientist could pick up and extend, and that a CTO could audit.

## Tech stack

- Python 3.11+, managed with **uv** (`pyproject.toml` + `uv.lock`)
- Data: `pandas`, `numpy`, `fredapi` (or `requests` against the FRED REST API)
- Stats/modeling: `statsmodels`, `scikit-learn`
- Visualization: `plotly`; dashboard in `streamlit`
- Notebooks: Jupyter, for exploration only. Reusable logic lives in `src/`.
- Testing: `pytest`. Lint/format: `ruff`.

## Layout

```
src/fintechco_demo/
  fred_client.py    # FRED fetch + on-disk cache; the only module that calls the network
  cleaning.py       # frequency alignment, resampling, missing-data handling, transforms
  analysis.py       # regressions, rolling correlations, event studies
  charts.py         # plotly figure builders (pure functions: DataFrame -> Figure)
app/dashboard.py    # streamlit entry point
notebooks/          # exploratory notebooks
data/raw/           # cached FRED pulls (gitignored)
data/processed/     # cleaned outputs (gitignored)
tests/
```

## Commands

```bash
uv sync                                  # install deps
uv run pytest                            # run tests
uv run ruff check . && uv run ruff format .
uv run streamlit run app/dashboard.py    # launch dashboard
uv run jupyter lab                       # notebooks
```

## Security and compliance guardrails

These are non-negotiable. They exist because this demo stands in for work at a regulated financial institution.

- **Secrets**: read `FRED_API_KEY` from the environment (`.env`, loaded with `python-dotenv`). Never hardcode, print, log, or commit keys. `.env` is gitignored; keep `.env.example` up to date.
- **Data**: use only public FRED data. Never introduce customer data, PII, or anything resembling real account information.
- **Network**: the only external calls allowed are to `api.stlouisfed.org`. Ask before adding any other external service or dependency.
- **Reproducibility**: cache raw pulls in `data/raw/` with the series ID and pull date in the filename. Every chart and number must trace back to a series ID and date range.
- **No destructive operations**: don't delete cached data, rewrite git history, or run `rm -rf` without asking.

## Coding conventions

- Keep functions small and pure where possible. Only `fred_client.py` performs I/O.
- Use type hints on public functions and a one-line docstring naming the FRED series involved.
- Use `pd.DatetimeIndex` for all time series. State the frequency explicitly (`"MS"`, `"D"`, etc.).
- Convert to a common frequency before joining series. Never silently forward-fill across more than one period. Log how many rows were filled or dropped.
- Label units in column names (`cpi_yoy_pct`, `unrate_pct`, `fedfunds_pct`).
- Charts need a title, axis labels with units, and a source note (`Source: FRED, series X`).
- Treat statistical results as descriptive unless shown otherwise. When reporting a relationship, also report the sample period, N, and a caveat about correlation vs. causation.

## Workflow expectations for Claude

- Before a non-trivial change, state the plan in 2–4 bullets.
- After changing code, run `uv run pytest` and `uv run ruff check .`, and report the actual results.
- Prefer the cached data in `data/raw/`. Hit the API only when data is missing or explicitly requested, because live-demo reliability matters.
- If the FRED API fails or rate-limits, fall back to cached data and say so clearly.
- Summarize findings in plain English a non-technical executive can follow, then give the technical detail.
