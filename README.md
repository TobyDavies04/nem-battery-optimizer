# NEM Battery Dispatch Optimizer

A quantitative, linear-programming-based battery dispatch optimizer for the Australian
National Electricity Market (NEM), built as an ongoing project to develop optimization,
market, and time-series modelling skills relevant to energy trading.

## Project roadmap

See `docs/roadmap.docx` for the full phased plan. Summary:

- **Phase 1 — Battery arbitrage LP (perfect foresight).** Optimize charge/discharge
  schedule against historical NEM dispatch prices to maximize revenue.
- **Phase 2 — Rolling-horizon re-optimization.** Realistic re-solving using only
  information available "as of now" plus a short-horizon forecast, with degradation cost.
- **Phase 3 — Multi-market co-optimization (FCAS).** Extend the LP to co-optimize
  energy arbitrage with Frequency Control Ancillary Services bidding.
- **Phase 4 — Volatility & regime modelling (GARCH / Markov-switching).** Apply
  nonlinear time series methods to NEM price volatility and feed the signal into the
  rolling-horizon optimizer as a risk-adjustment.

## Data source

Historical NEM dispatch price, demand, and generator-level data from
[AEMO NEMWEB](https://nemweb.com.au). ([OpenNEM](https://opennem.org.au) used as a
reference/visual guide during development.)

## Repo structure

```
data/           raw and processed NEM data (not committed — see .gitignore)
notebooks/      exploratory analysis
src/            core Python modules (LP model, data fetching, backtesting, etc.)
plots/          generated figures
tests/          unit tests
docs/           project roadmap and notes
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Status

🚧 Phase 1 in progress.
