import yfinance as yf
import talib
import pandas as pd
import matplotlib.pyplot as plt



def load_price_data(ticker: str, start: str = "2023-01-01", end: str = "2023-12-31") -> pd.DataFrame:
    """Load historical price data for a ticker.

    Tries to read a CSV from the local `data/raw` directory first; if not found, falls back to yfinance.
    """
    import os
    csv_path = os.path.join("data", "raw", f"{ticker}.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date")
    else:
        df = yf.download(ticker, start=start, end=end, progress=False)
        # Ensure a clean column layout
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
    df["Ticker"] = ticker
    return df


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add common TA-Lib technical indicators to the DataFrame.

    Expected columns: Open, High, Low, Close, Volume (or Adj Close).
    """
    price = df["Adj Close"] if "Adj Close" in df.columns else df["Close"]
    df["SMA_20"] = talib.SMA(price, timeperiod=20)
    df["SMA_50"] = talib.SMA(price, timeperiod=50)
    df["EMA_20"] = talib.EMA(price, timeperiod=20)
    df["RSI_14"] = talib.RSI(price, timeperiod=14)
    macd, macd_signal, macd_hist = talib.MACD(price, fastperiod=12, slowperiod=26, signalperiod=9)
    df["MACD"] = macd
    df["MACD_signal"] = macd_signal
    df["MACD_hist"] = macd_hist
    return df


def plot_price_and_sma(df: pd.DataFrame, ticker: str):
    """Plot closing price with SMA overlays."""
    price = df["Adj Close"] if "Adj Close" in df.columns else df["Close"]
    plt.figure(figsize=(14, 6))
    plt.plot(df.index, price, label="Close Price", linewidth=2)
    plt.plot(df.index, df["SMA_20"], label="SMA 20", linestyle="--")
    plt.plot(df.index, df["SMA_50"], label="SMA 50", linestyle="-.")
    plt.title(f"{ticker} – Price and SMA")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.show()


def plot_rsi(df: pd.DataFrame, ticker: str):
    """Plot RSI indicator with overbought/oversold levels."""
    plt.figure(figsize=(14, 4))
    plt.plot(df.index, df["RSI_14"], label="RSI 14", linewidth=2)
    plt.axhline(70, color="red", linestyle="--", label="Overbought (70)")
    plt.axhline(30, color="green", linestyle="--", label="Oversold (30)")
    plt.title(f"{ticker} – RSI")
    plt.xlabel("Date")
    plt.ylabel("RSI")
    plt.legend()
    plt.show()


def plot_macd(df: pd.DataFrame, ticker: str):
    """Plot MACD indicator with signal line and histogram."""
    plt.figure(figsize=(14, 4))
    plt.plot(df.index, df["MACD"], label="MACD", linewidth=2)
    plt.plot(df.index, df["MACD_signal"], label="Signal", linewidth=2)
    plt.bar(df.index, df["MACD_hist"], label="Histogram", width=0.8, alpha=0.3)
    plt.title(f"{ticker} – MACD")
    plt.xlabel("Date")
    plt.ylabel("MACD")
    plt.legend()
    plt.show()


def run_quantitative_analysis(ticker: str, start: str = "2023-01-01", end: str = "2023-12-31"):
    """High‑level helper to execute the full pipeline for a single ticker.
    """
    df = load_price_data(ticker, start, end)
    df = compute_indicators(df)
    plot_price_and_sma(df, ticker)
    plot_rsi(df, ticker)
    plot_macd(df, ticker)
    # Return the enriched DataFrame for further analysis if needed
    return df
