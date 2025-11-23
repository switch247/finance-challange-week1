import pandas as pd
import numpy as np
from typing import Optional

def generate_sample_df(country_name: str, n_days: int = 30, freq: str = "H", seed: Optional[int] = 42) -> pd.DataFrame:
    """Generate an in-memory sample DataFrame matching the dashboard's expected columns.

    Columns include headline, url, publisher, date, stock.
    """
    rng = np.random.default_rng(seed)
    periods = pd.date_range(end=pd.Timestamp.today().normalize(), periods=n_days * 24, freq=freq)

    # Sample headlines, publishers, stocks
    headlines = [
        "Stocks That Hit 52-Week Highs On Friday",
        "Stocks That Hit 52-Week Highs On Wednesday",
        "71 Biggest Movers From Friday",
        "46 Stocks Moving In Friday's Mid-Day Session",
        "B of A Securities Maintains Neutral on Agilent Technologies, Raises Price Target to $88",
        "CFRA Maintains Hold on Agilent Technologies, Lowers Price Target to $85"
    ]
    publishers = ["Benzinga Insights", "Lisa Levin", "Vick Meyer", "vishwanath@benzinga.com"]
    stocks = ["A", "B", "C"]  # Example stocks

    n_samples = len(periods)
    selected_headlines = rng.choice(headlines, size=n_samples)
    selected_publishers = rng.choice(publishers, size=n_samples)
    selected_stocks = rng.choice(stocks, size=n_samples)

    # Generate URLs based on headlines (simplified)
    urls = [f"https://www.example.com/news/{i}/" + h.lower().replace(" ", "-") for i, h in enumerate(selected_headlines)]

    df = pd.DataFrame({
        "headline": selected_headlines,
        "url": urls,
        "publisher": selected_publishers,
        "date": periods.strftime("%Y-%m-%d %H:%M:%S%z"),
        "stock": selected_stocks,
    })

    return df