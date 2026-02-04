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


def compute_ewma_estimates(returns_list, alpha, end_idx):
    """Compute EWMA mean and variance estimates up to end_idx.

    Args:
        returns_list: List of return series (one per asset)
        alpha: EWMA smoothing parameter
        end_idx: Index to compute estimates up to (exclusive)

    Returns:
        (means, vols) - arrays of mean and volatility estimates
    """
    means = []
    vols = []

    for returns in returns_list:
        # Slice returns up to end_idx
        r = returns[:end_idx]

        # Compute EWMA of returns and squared returns
        m1 = ewma(alpha, r)
        m2 = ewma(alpha, r * r)

        # Extract final estimates
        mu = m1[-1]
        var = m2[-1] - mu * mu
        vol = np.sqrt(max(0, var))

        means.append(mu)
        vols.append(vol)

    return np.array(means), np.array(vols)


def rollingOptimize(returns_list, alpha=0.06, lookback=32, eta=0.1, nsteps=5):
    """Perform rolling walk-forward portfolio optimization.

    Args:
        returns_list: List of return series (one per asset)
        alpha: EWMA smoothing parameter
        lookback: Minimum days before starting optimization
        eta: Gradient ascent learning rate
        nsteps: Optimization steps per day

    Returns:
        Dictionary with:
            - daily_returns: Array of daily portfolio returns
            - cumulative_returns: Array of cumulative returns
            - weights: Array of daily weight vectors
            - sharpe_ratios: Array of daily Sharpe ratios
    """
    n_assets = len(returns_list)
    n_days = len(returns_list[0])

    # Initialize tracking
    weights_history = []
    portfolio_returns = []
    cumulative_returns = []
    sharpe_history = []

    # Start with equal weights
    w = np.ones(n_assets) / n_assets

    # Rolling optimization loop
    for day in range(lookback, n_days):
        # Compute EWMA estimates using data up to current day
        mu, vols = compute_ewma_estimates(returns_list, alpha, day)

        # Optimize weights using gradient ascent
        for _ in range(nsteps):
            # Sharpe ratio
            port_ret = np.sum(w * mu)
            port_vol = np.sqrt(np.sum((w ** 2) * (vols ** 2)))
            sharpe = port_ret / port_vol if port_vol > 0 else 0

            # Gradient of Sharpe ratio
            grad_ret = mu
            grad_vol = w * (vols ** 2) / port_vol if port_vol > 0 else np.zeros_like(w)
            grad_sharpe = (grad_ret * port_vol - port_ret * grad_vol) / (port_vol ** 2) if port_vol > 0 else np.zeros_like(w)

            # Gradient ascent step
            w_new = w + eta * grad_sharpe
            w_new = np.maximum(w_new, 0)  # No short selling
            w_sum = np.sum(w_new)
            w = w_new / w_sum if w_sum > 0 else w  # Normalize

        # Calculate portfolio return for this day
        day_returns = np.array([returns_list[i][day] for i in range(n_assets)])
        port_return = np.sum(w * day_returns)

        # Track results
        weights_history.append(w.copy())
        portfolio_returns.append(port_return)
        cumulative_returns.append(np.sum(portfolio_returns))
        sharpe_history.append(sharpe)

    # Return as list to avoid dictionary access issues in Klong
    return [
        np.array(portfolio_returns),       # [0] daily_returns
        np.array(cumulative_returns),      # [1] cumulative_returns
        np.array(weights_history),         # [2] weights
        np.array(sharpe_history)           # [3] sharpe_ratios
    ]
