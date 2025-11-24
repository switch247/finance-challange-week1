project/          # One repo = one real project
├── .gitignore
├── README.md                       # How to run, train, deploy, reproduce
├── LICENSE
├── pyproject.toml                  # Poetry / Hatch / PDM (2025 default)
├── Makefile                        # make train, make predict, make all, etc.
├── .env.example                    # Example env vars
├── .pre-commit-config.yaml

├── config/                         # All configuration lives here
│   ├── __init__.py
│   ├── logging.yaml                # Structured logging config
│   ├── settings.py                 # Pydantic v2 BaseSettings (loads .env)
│   └── hydra/                      # Optional: if you want Hydra later
│       └── config.yaml

├── data/
│   ├── raw/                        # gitignored + DVC/Pachyderm/LakeFS
│   ├── processed/
│   └── external/

├── notebooks/                      # Exploration only (committed)
│   ├── 01_eda.ipynb
│   ├── 02_features.ipynb
│   └── 99_scratch.ipynb

├── src/
│   └── customer_churn/             # Your actual installable package
│       ├── __init__.py
│       ├── __main__.py             # python -m customer_churn → CLI
│       │
│       ├── constants.py            # Paths, column names, seeds, etc.
│       ├── logger.py               # Configured logger = get_logger(__name__)
│       ├── seed.py                 # seed_everything(42) for reproducibility
│       │
│       ├── data/
│       │   ├── loader.py
│       │   └── preprocessing.py
│       │
│       ├── features/
│       │   ├── builder.py
│       │   └── transformers.py     # scikit-learn custom transformers
│       │
│       ├── models/
│       │   ├── train.py
│       │   ├── predict.py
│       │   ├── evaluate.py
│       │   └── model_registry.py   # Save/load with MLflow or custom
│       │
│       ├── pipelines/              # Full end-to-end pipelines
│       │   ├── training_pipeline.py   # used by scripts/train.py
│       │   └── inference_pipeline.py
│       │
│       ├── utils/                  # Reusable helpers (used everywhere)
│       │   ├── metrics.py
│       │   ├── plotting.py
│       │   ├── files.py            # save_pickle, load_pickle, etc.
│       │   └── timers.py
│       │
│       └── cli/                    # Click or Typer commands
│           ├── __init__.py
│           ├── train.py
│           └── predict.py

├── scripts/                        # ← These are the ones you actually run daily
│   ├── train.py                    # python scripts/train.py --config config/prod.yaml
│   ├── predict.py                  # python scripts/predict.py input.csv output.csv
│   ├── evaluate.py
│   ├── serve.py                    # FastAPI / BentoML / your model server
│   ├── backfill.py                 # Re-run historical predictions
│   ├── debug_data.py               # Quick sanity checks
│   └── mlflow_run.py               # One-off MLflow experiments

├── tests/
│   ├── unit/
│   │   ├── test_data.py
│   │   ├── test_features.py
│   │   └── test_models.py
│   └── integration/
│       └── test_full_pipeline.py

├── outputs/                        # gitignored
│   ├── models/
│   ├── predictions/
│   ├── figures/
│   └── reports/

├── experiments/                    # MLflow tracking folder (or W&B)
└── docs/                           # Optional Sphinx / MkDocs