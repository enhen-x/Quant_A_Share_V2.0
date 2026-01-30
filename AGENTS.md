# Repository Guidelines

## Project Structure & Module Organization
- `src/` houses the core library modules (data sources, preprocessing, modeling, backtest, analysis, live trading).
- `scripts/` contains runnable entry points for data ingestion, feature building, training, backtests, and live trading workflows.
- `config/` stores configuration (notably `config/main.yaml`).
- `data/` is the working data area (raw, processed, and live-trading configs).
- `docs/` and `architecture/` contain reports, guides, and diagrams.
- `logs/` and `figures/` capture runtime outputs and visuals.

## Build, Test, and Development Commands
- `pip install -r requirements.txt` installs Python dependencies (Python 3.10+ required).
- `python scripts/date_landing/init_stock_pool.py` initializes the stock universe.
- `python scripts/date_landing/download_data.py` downloads market data.
- `python scripts/analisis/clean_and_check.py` cleans and validates data.
- `python scripts/feature_create/rebuild_features.py` rebuilds features/labels.
- `python scripts/model_train/run_walkforward.py` runs walk-forward training.
- `python scripts/back_test/run_backtest.py` executes backtests.
- `python scripts/back_test/run_recommendation.py` generates daily recommendations.
- `python scripts/live/run_auto_trading.py` starts simulated live trading (`--real` for real mode).

## Coding Style & Naming Conventions
- Python code uses 4-space indentation; keep functions and modules small and focused.
- Follow existing module layout under `src/` (e.g., `src/preprocessing/*`, `src/model/*`).
- Prefer descriptive, lowercase_with_underscores for functions and scripts (e.g., `run_walkforward.py`).

## Testing Guidelines
- No dedicated test framework or `tests/` directory is present.
- Use analysis and validation scripts under `scripts/analisis/` for sanity checks (e.g., `check_overfit.py`, `check_stress_test.py`).
- When adding tests, note the command and location in this file.

## Commit & Pull Request Guidelines
- Recent commits use concise, action-oriented prefixes like `docs:`, `Fix`, and `Update:`.
- Keep commits focused by feature or fix; include a short summary and key impact.
- PRs should describe scope, include relevant run commands, and link related issues or notes. Add screenshots for chart/visual updates when applicable (see `architecture/` outputs).

## Security & Configuration Tips
- Do not commit credentials; use `data/live_trading/config*.txt` examples as templates.
- Validate config changes in `config/main.yaml` before running long workflows.
