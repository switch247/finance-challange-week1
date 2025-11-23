#  Price Moves TO News Sentiment Co relation— Week 1 Challenge

Analyze the relationship between financial news sentiment and stock price movements and produce actionable insights.

---

## Project Overview

This repository contains compact code and notebooks to:

- load and inspect news and price data
- run simple sentiment scoring on headlines
- compute basic technical indicators for exploratory analysis
- run correlation and lag analysis between sentiment and daily returns
- produce figures used in the week-1 deliverable

This README is intentionally concise: it explains purpose, how to run the code, and where to find analyses.

---

## Business Objective

Combine qualitative news sentiment with quantitative price metrics to surface short-term signals useful for trading or monitoring.

---

## Dataset Overview

- News (sample columns): `headline`, `url`, `publisher`, `date` (timestamp), `stock`
- Prices (sample columns): `Open`, `High`, `Low`, `Close`, `Volume` (daily returns are derived)

Example sample files are in the `data/` directory. Large or external data sources should be kept out of Git and downloaded via scripts or referenced in configs.

---

## Folder Structure (this repo)

```
./
├─ data/                # sample CSVs (news + prices)
├─ scripts/             # loader, preprocess, model, plotting helpers
├─ notebook/            # EDA and analysis notebooks
├─ models/              # saved model artifacts
├─ tests/               # unit tests
├─ requirements.txt
└─ README.md
```

---

## Setup & Installation (PowerShell)

1) Create and activate a virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

2) Run tests:

```powershell
pytest -q
```

---

## Run examples (quick)

- Train the demo sentiment model (uses `data/raw_analyst_ratings.csv` by default):

```powershell
python -c "from scripts import model; model.train()"
```

- Evaluate a saved model:

```powershell
python -c "from scripts import model; model.evaluate_model()"
```

- Generate a small in-memory sample dataset for quick experiments:

```powershell
python -c "from scripts import utils; df = utils.generate_sample_df('Demo', n_days=7); print(df.head())"
```

If you prefer explicit CLI files, I can add `scripts/run_training.py` and `scripts/run_eval.py` as thin wrappers.

---

## Tasks Completed (in this repo)

- Starter EDA notebooks (`notebook/`)
- Basic sentiment demo pipeline and model save/load (`scripts/model.py`)
- Data loading and preprocessing helpers (`scripts/data_loader.py`, `scripts/preprocess.py`)

---

## Technologies

- Python, pandas, NumPy
- scikit-learn (demo model)
- Matplotlib / Seaborn for visualization
- Optional: TA-Lib / PyNance for technical indicators (install if needed)

---

## Key Starter Insights (illustrative)

- Negative headlines often align with short-term negative returns (needs per-stock quantification).
- Aggregating sentiment per day and combining it with indicators increases robustness.

---

## Next Steps (recommendations)

- Add small CLI entrypoints for common tasks (`train`, `evaluate`, `generate-sample`).
- Provide a `configs/default.yaml` to centralize file paths and parameters.
- Add CI to run `pytest` and basic linting on PRs.

---

If you'd like, I can now implement the CLI wrappers, update `requirements.txt`, or add a short `CONTRIBUTING.md`.
