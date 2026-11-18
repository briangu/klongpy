# Stocks Examples

Market data feeds and finance models.

## Simulated feed

`feed/` publishes simulated trades over IPC to a consumer that computes VWAP and a moving average. It needs only `pip install klongpy`. See [feed/README.md](feed/README.md).

```bash
cd feed
kgpy demo.kg
```

## Black-Scholes (Klong and Python)

```bash
kgpy options/black-scholes.kg
python options/black-scholes.py
```

## Yahoo Finance data and LSTM forecasting

These programs download prices over the network and need `pip install yfinance`; the LSTM programs also need `torch`.

```bash
kgpy yfinance/fetch.kg
kgpy yfinance/lstm.kg
kgpy yfinance/compact_lstm.kg
```
