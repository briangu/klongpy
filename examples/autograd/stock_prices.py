"""
Helper to fetch stock price data from Yahoo Finance.
All math/optimization is done in Klong - this only fetches data.
"""
import yfinance as yf
import numpy as np


def fetchStockPrices(tickers, start, end):
    """Fetch daily closing prices for multiple stocks.

    Args:
        tickers: List of ticker symbols
        start: Start date (e.g., "2024-01-01")
        end: End date (e.g., "2024-12-31")

    Returns:
        List of price series (one per ticker), aligned to shortest length
    """
    series = []
    for ticker in tickers:
        data = yf.download(ticker, start=start, end=end, progress=False)
        close = data['Close'].values.flatten()
        series.append(close)
    min_len = min(len(s) for s in series)
    return [np.ascontiguousarray(s[:min_len]) for s in series]
