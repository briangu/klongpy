# Simulated market feed

A ticker feed over KlongPy IPC with no broker, account or network: a seeded random walk publishes trades, and a consumer keeps the latest price per symbol and a volume-weighted average price (VWAP) and short moving average. It needs only `pip install klongpy`.

| File | Role |
|---|---|
| `feed.kg` | The source (`next()`, a seeded random walk over four symbols) and the server that publishes each trade to the clients subscribed to its symbol. |
| `consumer.kg` | Subscribes, keeps prices and sizes per symbol, and prints last price, VWAP, 5-trade moving average, volume and trade count when the run ends. |
| `demo.kg` | Runs both in one process with fixed settings. |

A trade is a dictionary with the keys `sym`, `t` (seconds since the start of the run), `px` and `qty`. The random numbers come from a small generator in `feed.kg`, so a seed gives the same trades on every run and on every backend.

## Run in two terminals

Run the programs from this directory: `demo.kg` loads the others by relative path. In the first terminal:

```bash
kgpy feed.kg -- 8888 20 60 7
```

The arguments are the port, the trades per second, the number of trades (0 for no end) and the seed. They are all optional, in that order. The feed keeps running while its server listens; press Ctrl-C to stop it.

In the second terminal:

```bash
kgpy consumer.kg -- 8888 AAPL NVDA
```

The arguments are the port and the symbols to subscribe to; with no symbols the consumer subscribes to all four (`*`). The source starts when the first client subscribes and ends after the last trade, so a consumer started later starts a new run. When a run ends the consumer prints its summary, closes its connection and exits.

## Run in one process

```bash
kgpy demo.kg
```

The demo publishes 200 trades at 400 a second with seed 7, and its consumer subscribes to AAPL, NVDA and TSLA. It prints the same output on every run (the port in the first line is the one given as an optional argument, 18888 by default):

```text
feed on port 18888: 400 trades/s, 200 trades, seed 7
feed: published 200 trades
summary of 200 trades published
AAPL     last   187.78 vwap   189.79 sma5   187.47 volume  27000 trades 53
NVDA     last   119.93 vwap   120.32 sma5   120.30 volume  19800 trades 44
TSLA     last   248.62 vwap   249.19 sma5   248.69 volume  25000 trades 49
```

The program ends after the last line: the consumer closes its connection, and the server stops when the last client has gone.

## Use a real source

`next()` in `feed.kg` is the only function that makes trades. To publish a real feed, remove the timer (`emit`, `begin`) and call `publish(trade)` for each trade your source delivers, with `trade` a dictionary that has the keys `sym`, `t`, `px` and `qty`. The server, the subscriptions and the consumer stay as they are.
