# Stocks Examples

Market data ingestion and finance models.

## Requirements

```bash
pip install yfinance
```

For broker/data APIs:

```bash
pip install alpaca-py polygon-api-client requests
```

## Run

### Yahoo Finance data + stats

```bash
kgpy yfinance/fetch.kg
```

### LSTM forecasting (uses yfinance data)

```bash
kgpy yfinance/lstm.kg
kgpy yfinance/compact_lstm.kg
```

### Black-Scholes (Klong and Python)

```bash
kgpy options/black-scholes.kg
python options/black-scholes.py
```

### Alpaca data (requires API keys)

```bash
export ALPACA_API_KEY=...
export ALPACA_SECRET_KEY=...
kgpy alpaca/update_data.kg
```

### Polygon data (requires API key)

```bash
export POLYGON_API_KEY=...
kgpy polygon/update_data.kg
```
