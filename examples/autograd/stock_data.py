"""
Simple helper to fetch stock data from Yahoo Finance.
"""
import yfinance as yf
import numpy as np


def fetch_returns(ticker, start, end):
    """Fetch daily returns for a stock."""
    data = yf.download(ticker, start=start, end=end, progress=False)
    close = data['Close'].values.flatten()  # Flatten to 1D
    # Calculate returns: (p[t] - p[t-1]) / p[t-1]
    returns = (close[1:] - close[:-1]) / close[:-1]
    return returns


def fetchStockData(start, end):
    """Fetch returns for AAPL, MSFT, GOOGL, JPM."""
    r0 = fetch_returns("AAPL", start, end)
    r1 = fetch_returns("MSFT", start, end)
    r2 = fetch_returns("GOOGL", start, end)
    r3 = fetch_returns("JPM", start, end)
    # Align to shortest length
    min_len = min(len(r0), len(r1), len(r2), len(r3))
    return [r0[:min_len], r1[:min_len], r2[:min_len], r3[:min_len]]
