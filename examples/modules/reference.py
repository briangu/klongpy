#!/usr/bin/env python3
"""Reference for main.kg: the same calculation in plain Python, printing the text of EXPECTED.txt.

Floating point follows the order of evaluation of Klong (right to left), and the
sums are plain left-to-right loops. Run:  python3 reference.py > EXPECTED.txt
"""
import math

out = []
def p(s): out.append(s)

# ---- lib/rng.kg
M = 2147483648
class Rng:
    state = 1
    count = 0
def step():
    Rng.count += 1
    Rng.state = (1103515245 * Rng.state + 12345) % M
    return Rng.state
def reseed(n):
    Rng.state = n % M
    Rng.count = 0
def uniform(n): return [step() / M for _ in range(n)]
def noise(n): return [(u - 0.5) * 3.4641016151377544 for u in uniform(n)]

# ---- lib/stats.kg
def fsum(v):
    s = 0.0
    for a in v: s = s + a
    return s
def cum(v):
    c = [0.0]
    for a in v: c.append(c[-1] + a)
    return c
def mean(v): return fsum(v) / len(v)
def var(v):
    m = mean(v)
    return mean([(a - m) * (a - m) for a in v])
def std(v): return var(v) ** 0.5
def win(n, v): return [v[i:i + n] for i in range(len(v) - n + 1)]
def ma(n, v):
    c = cum(v)
    return [(a - b) / n for a, b in zip(c[n:], c[:-n])]
def ma4(v): return ma(4, v)
def returns(p_): return [a / b - 1 for a, b in zip(p_[1:], p_[:-1])]
def drawdown(p_):
    out_, m = [], None
    for a in p_:
        m = a if m is None or a > m else m
        out_.append((a - m) / m)
    return out_
def maxdd(p_): return min(drawdown(p_))

# ---- lib/series.kg
def sma(n, p_): return ma(n, p_)
def zscore(n, p_):
    a = p_[n - 1:]
    m = ma(n, p_)
    s = [std(w) for w in win(n, p_)]
    return [(x - y) / z for x, y, z in zip(a, m, s)]

# ---- book/portfolio.kg
def new(cash): return dict(cash=cash, pos=0.0, px=0.0, trades=0)
def equity(pf): return pf["cash"] + pf["pos"] * pf["px"]
def trade(pf, px, qty):
    q = new(pf["cash"] - px * qty)
    q["pos"] = pf["pos"] + qty
    q["px"] = px
    q["trades"] = pf["trades"] + (0 if qty == 0 else 1)
    return q
def mark(pf, px): return trade(pf, px, 0.0)

# ---- strategy/crossover.kg, meanrev.kg
def cross_signal(cfg, p_):
    f = cfg["strategy"]["params"]["crossover"]["fast"]
    s = cfg["strategy"]["params"]["crossover"]["slow"]
    slow, fast = ma(s, p_), ma(f, p_)[s - f:]
    return [0] * (s - 1) + [1 if a < b else 0 for a, b in zip(slow, fast)]
def cross_step(pf, row):
    px = row[0]
    m = mark(pf, px)
    q = (row[1] * math.floor(equity(m) / px)) - m["pos"]
    return trade(m, px, q)
def mean_signal(cfg, p_):
    w = cfg["strategy"]["params"]["meanrev"]["window"]
    b = cfg["strategy"]["params"]["meanrev"]["band"]
    z = zscore(w, p_)
    return [0] * (w - 1) + [(1 if a < -b else 0) - (1 if a > b else 0) for a in z]
def mean_step(pf, row):
    px = row[0]
    m = mark(pf, px)
    q = math.floor(row[1] * (0.5 * (equity(m) / px)))
    q = q - m["pos"]
    return trade(m, px, q)
def cross_describe(cfg):
    c = cfg["strategy"]["params"]["crossover"]
    return "crossover fast=%d slow=%d" % (c["fast"], c["slow"])
def mean_describe(cfg):
    c = cfg["strategy"]["params"]["meanrev"]
    return "meanrev window=%d band=%s" % (c["window"], c["band"])
REG = {"crossover": dict(name="crossover", signal=cross_signal, step=cross_step, describe=cross_describe),
       "meanrev": dict(name="meanrev", signal=mean_signal, step=mean_step, describe=mean_describe)}

# ---- config.kg
def defaults():
    return dict(run=dict(days=120, seed=7, cash=10000.0),
                feed=dict(symbols=["ALPHA", "BETA"], drift=0.0004, vol=0.012),
                strategy=dict(params=dict(crossover=dict(fast=5, slow=20), meanrev=dict(window=20, band=1.0))),
                report=dict(title="backtest"))

# ---- report/table.kg, Klong .fmt
def num(x, w, d):  return ("%" + str(w + d + 1) + "." + str(d) + "f") % x
def left(s, w): return s.ljust(w)
def right(s, w): return str(s).rjust(w)
def header():
    return "%s %s %s %s %s %s" % (left("symbol", 8), left("strategy", 10), right("equity", 13),
                                  right("return%", 8), right("maxdd%", 8), right("trades", 6))
def row(sym, strat, eq, ret, dd, n):
    return "%s %s %s %s %s %s" % (left(sym, 8), left(strat, 10), num(eq, 10, 2), num(ret, 5, 2), num(dd, 5, 2), right(n, 6))
def csvline(s): return ",".join('"%s"' % f for f in s.split(" "))

# ---- main.kg
p("== modules: a backtest ==")
p("[config] loaded"); p("[rng] loaded"); p("[feed] loaded")
p("[series] loaded"); p("[portfolio] loaded"); p("[crossover] loaded"); p("[meanrev] loaded"); p("[registry] loaded")
p("[table] loaded")

p("-- configuration")
cfg = defaults()
cfg["run"]["days"] = 90
cfg["strategy"]["params"]["crossover"]["fast"] = 4
snap = cfg["run"]
snap["seed"] = 11
p("days=%d seed=%d cash=%s" % (cfg["run"]["days"], cfg["run"]["seed"], cfg["run"]["cash"]))

p("-- shared state of rng")
p("draws before the feed: %d" % Rng.count)
reseed(cfg["run"]["seed"])
quotes = {}
for sym in cfg["feed"]["symbols"]:
    r_ = [cfg["feed"]["drift"] + cfg["feed"]["vol"] * z for z in noise(cfg["run"]["days"])]
    ps = [100.0]
    for r in r_: ps.append(ps[-1] * (1 + r))
    quotes[sym] = ps
p("draws after the feed: %d" % Rng.count)
uniform(1)
p("same instance by another path: %d" % Rng.count)
p("exports of rng: 4")
p("step is missing from the exports: 1")

p("-- qualified names")
v = quotes["ALPHA"]
r = returns(v)
m5 = [mean(w) for w in win(5, r)]
p("bars: %d, five-day windows: %d" % (len(v), len(m5)))
p("mean of the five-day means, in bps: %s" % num(10000 * mean(m5), 3, 2))
p("projection: %s" % repr(21.0 * 2.0))
p("as a value: %s" % repr(mean([1.0, 2.0, 6.0])))
xs = [1.0, 2.0, 6.0]
p("in a dictionary: %s %s" % (num(mean(xs), 1, 4), num(std(xs), 1, 4)))
p("spread: %s" % num(std(xs) / mean(xs), 1, 4))
p("avg price %s, worst drawdown %s%%" % (num(mean(v), 3, 2), num(100 * maxdd(v), 2, 2)))
ramp = [1.0 + i for i in range(16)]
p("ma4 agrees with ma: 1")

p("-- the strategies, chosen by name at run time")
def run(strat, cfg, sym, p_):
    sig = strat["signal"](cfg, p_)
    rows = [[a, float(b)] for a, b in zip(p_, sig)]
    pf = new(cfg["run"]["cash"])
    trail = [pf]
    for rw in rows:
        pf = strat["step"](pf, rw)
        trail.append(pf)
    curve = [equity(t) for t in trail]
    fin = trail[-1]
    ret = 100 * ((curve[-1] / cfg["run"]["cash"]) - 1)
    dd = 100 * maxdd(curve)
    s = "%s %s %s" % (sym, strat["name"], num(curve[-1], 1, 2))
    return dict(row=row(sym, strat["name"], curve[-1], ret, dd, fin["trades"]), csv=s)
def rowsfor(name):
    s = REG[name]
    p(s["describe"](cfg))
    return [run(s, cfg, x, quotes[x]) for x in cfg["feed"]["symbols"]]
results = []
for name in ["crossover", "meanrev"]:
    results += rowsfor(name)
p(header())
for x in results: p(x["row"])

p("-- csv: a classic library used through the new form, and imported lazily by table")
p("[table] importing csv")
out_csv = [csvline(x["csv"]) for x in results]
p(out_csv[0])
p("fields in the first line: %d" % len(out_csv[0].split('","')))

p("-- a classic file beside the modules")
p("legacy mean %s, stats mean %s" % (num((v[0] + v[-1]) / 2, 3, 2), num(mean(v), 3, 2)))
p("bps: %s" % repr(10000 * 0.25))
p("== done ==")
print("\n".join(out))
