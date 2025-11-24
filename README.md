# Financial News Sentiment Analysis

Analyze the relationship between financial news sentiment and stock price movements.

---

## Project Overview

This repository contains code to:
- Load and inspect news data.
- Perform Exploratory Data Analysis (EDA) including text analysis and topic modeling.
- Train and evaluate sentiment analysis models.

## Folder Structure

```
./
├── config/             # Configuration (logging, settings)
├── data/               # Data directory (raw, processed)
├── notebooks/          # Jupyter notebooks for analysis
├── scripts/            # Executable scripts (train.py)
├── src/
│   └── fnsa/           # Main package
│       ├── data/       # Data loading and preprocessing
│       ├── features/   # Feature engineering and text analysis
│       ├── models/     # Model training and evaluation
│       └── utils/      # Utilities (plotting)
├── tests/              # Unit tests
├── pyproject.toml      # Project configuration and dependencies
└── README.md
```

## Setup & Installation

This project uses [Poetry](https://python-poetry.org/) for dependency management.

1.  **Install Dependencies**:
    ```bash
    poetry install
    ```

2.  **Activate Virtual Environment**:
    ```bash
    poetry shell
    ```

## Usage

### 1. Exploratory Data Analysis
Run the notebook `notebooks/01_eda.ipynb` to perform comprehensive EDA, including:
- Descriptive statistics
- Publisher analysis
- Time series analysis
- Topic modeling

### 2. Train Model
Train the sentiment analysis model using the training script:

```bash
python scripts/train.py
```

Options:
- `--data_path`: Path to the input CSV file.
- `--model_path`: Path to save the trained model.
- `--evaluate`: Run evaluation after training.

Example:
```bash
python scripts/train.py --evaluate
```

## Configuration
- **Settings**: `config/settings.py` (manages paths and constants).
- **Logging**: `config/logging.yaml`.

## Technologies
- **Python 3.8+**
- **Pandas, NumPy**: Data manipulation.
- **Scikit-learn**: Machine learning (Logistic Regression, NMF, TF-IDF).
- **TextBlob**: Sentiment analysis.
- **Matplotlib, Seaborn**: Visualization.
