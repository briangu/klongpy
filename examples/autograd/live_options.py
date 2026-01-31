"""
Fetch live option data from Yahoo Finance for Black-Scholes verification.
"""
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np


def get_option_data(ticker="AAPL"):
    """Get current stock price and a near-the-money option."""
    stock = yf.Ticker(ticker)

    # Get current stock price
    spot = stock.info.get('currentPrice') or stock.info.get('regularMarketPrice')

    # Get option chain for nearest expiration
    expirations = stock.options
    if not expirations:
        return None

    # Pick expiration ~30-60 days out for reasonable theta
    today = datetime.now()
    best_exp = None
    for exp in expirations:
        exp_date = datetime.strptime(exp, "%Y-%m-%d")
        days_out = (exp_date - today).days
        if 20 <= days_out <= 90:
            best_exp = exp
            break

    if not best_exp:
        best_exp = expirations[min(2, len(expirations)-1)]

    # Get option chain
    chain = stock.option_chain(best_exp)
    calls = chain.calls

    # Find near-the-money call
    calls = calls[calls['volume'] > 100]  # Filter for liquid options
    if len(calls) == 0:
        calls = chain.calls

    calls['moneyness'] = abs(calls['strike'] - spot)
    atm_call = calls.loc[calls['moneyness'].idxmin()]

    # Calculate time to expiry in years
    exp_date = datetime.strptime(best_exp, "%Y-%m-%d")
    time_to_expiry = (exp_date - today).days / 365.0

    # Risk-free rate (approximate with 3-month Treasury, Jan 2026)
    rate = 0.043  # ~4.3% current short-term rate

    return {
        'ticker': ticker,
        'spot': float(spot),
        'strike': float(atm_call['strike']),
        'time': float(time_to_expiry),
        'rate': rate,
        'impliedVol': float(atm_call['impliedVolatility']),
        'marketPrice': float(atm_call['lastPrice']),
        'marketDelta': float(atm_call.get('delta', 0)) if 'delta' in atm_call else None,
        'expiration': best_exp,
        'bid': float(atm_call['bid']),
        'ask': float(atm_call['ask'])
    }


def format_for_klong(data):
    """Print data in format easy to paste into Klong."""
    if data is None:
        print("Could not fetch option data")
        return

    print(f':\" Live Option Data for {data["ticker"]}\"')
    print(f"spot::{data['spot']}")
    print(f"strike::{data['strike']}")
    print(f"rate::{data['rate']}")
    print(f"time::{data['time']:.6f}")
    print(f"vol::{data['impliedVol']:.6f}")
    print()
    mid = (data['bid'] + data['ask']) / 2
    print(f"Expiration: {data['expiration']}")
    print(f"Market Price: ${data['marketPrice']:.2f} (bid: ${data['bid']:.2f}, ask: ${data['ask']:.2f})")
    print(f"Mid Price: ${mid:.2f}")


def fetchOptionData(ticker="AAPL"):
    """Return option data as list: [spot, strike, rate, time, vol, bid, ask, lastPrice]"""
    data = get_option_data(ticker)
    if data is None:
        return None
    return [
        data['spot'],
        data['strike'],
        data['rate'],
        data['time'],
        data['impliedVol'],
        data['bid'],
        data['ask'],
        data['marketPrice'],
        data['expiration']
    ]


if __name__ == "__main__":
    data = get_option_data("AAPL")
    if data:
        format_for_klong(data)
