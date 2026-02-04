"""
Helper to fetch stock price series from Yahoo Finance.
"""
import yfinance as yf
import numpy as np


def fetch_prices(ticker, start, end):
    """Fetch daily closing prices for a stock."""
    data = yf.download(ticker, start=start, end=end, progress=False)
    close = data['Close'].values.flatten()
    return close


def fetchStockPrices(tickers, start, end):
    """Fetch price series for multiple stocks.

    Args:
        tickers: List of ticker symbols
        start: Start date (e.g., "2024-01-01")
        end: End date (e.g., "2024-12-31")

    Returns:
        List of price series (one per ticker), aligned to shortest length
    """
    series = [fetch_prices(ticker, start, end) for ticker in tickers]
    min_len = min(len(s) for s in series)
    return [s[:min_len] for s in series]


def fetchStockReturns(tickers, start, end):
    """Fetch return series for multiple stocks.

    Args:
        tickers: List of ticker symbols
        start: Start date (e.g., "2024-01-01")
        end: End date (e.g., "2024-12-31")

    Returns:
        List of return series (one per ticker), aligned to shortest length
    """
    prices = fetchStockPrices(tickers, start, end)
    returns = []
    for p in prices:
        r = (p[1:] - p[:-1]) / p[:-1]
        # Ensure it's a contiguous array
        returns.append(np.ascontiguousarray(r))
    return returns


def ewma(alpha, series):
    """Calculate exponentially weighted moving average.

    Args:
        alpha: Smoothing factor (0 < alpha <= 1)
        series: Array of values (numpy or torch)

    Returns:
        Array of EWMA values
    """
    # Check if it's a torch tensor
    try:
        import torch
        is_torch = isinstance(series, torch.Tensor)
    except ImportError:
        is_torch = False

    if is_torch:
        result = torch.zeros_like(series)
    else:
        result = np.zeros_like(series)

    result[0] = series[0]
    for i in range(1, len(series)):
        result[i] = alpha * series[i] + (1 - alpha) * result[i-1]
    return result
