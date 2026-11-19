# Examples

This page indexes the example programs in the `examples/` directory and shows a few snippets you can run in the REPL. For a first lesson in the language, use [Start Here](getting-started.md) and the [Tutorial](tutorial.md).

The snippets define functions with named parameters, `area(w;h)::{w*h}`, and use the short forms `x`, `y` and `z` for one-line functions given to adverbs. See [named parameters](language-additions.md#named-parameters).

## Example directories

Each directory is under [`examples/`](https://github.com/briangu/klongpy/tree/main/examples). The extra column names the install extra or package the example needs beyond `pip install klongpy`.

| Directory | What it shows | Needs |
|---|---|---|
| [1brc](https://github.com/briangu/klongpy/tree/main/examples/1brc) | A One Billion Row Challenge pipeline: files read in parallel by Python, statistics aggregated by a KlongPy function. | |
| [autograd](https://github.com/briangu/klongpy/tree/main/examples/autograd) | Gradients, optimizers, linear regression, neural nets, finance and differentiable physics. | `klongpy[torch]`; `yfinance` for the live options examples |
| [db](https://github.com/briangu/klongpy/tree/main/examples/db) | DuckDB table interop, a persisted table store, and a web and database server. | `klongpy[db,web]` |
| [embedded](https://github.com/briangu/klongpy/tree/main/examples/embedded) | Static-profile programs (a moving average, a FIR filter, a PID step) with their expected output, compiled to C as in [Compiling Klong to C](embedded.md). | |
| [hardware](https://github.com/briangu/klongpy/tree/main/examples/hardware) | A moving-average crossover strategy for the FPGA, with the scripts that measure it. See [Klong functions as FPGA hardware](hardware.md). | an FPGA toolchain and card |
| [deploy](https://github.com/briangu/klongpy/tree/main/examples/deploy) | A Docker image that runs `kgpy` on the script named by `KG_FILE_PATH`, and Kubernetes files. | Docker |
| [ipc](https://github.com/briangu/klongpy/tree/main/examples/ipc) | IPC clients and servers, broadcast and publish/subscribe with `.srv`, `.cli` and timers. | |
| [math](https://github.com/briangu/klongpy/tree/main/examples/math) | NumPy interop for FFT and trigonometry through `.pyf`. | |
| [ml](https://github.com/briangu/klongpy/tree/main/examples/ml) | K-nearest neighbours, cosine similarity, a transformer and an LSTM. | `torch` for the LSTM only |
| [modules](https://github.com/briangu/klongpy/tree/main/examples/modules) | A backtest split over twelve files with `.import` and `.export`: private state, qualified names `a.b`, and one program per module error. | |
| [python](https://github.com/briangu/klongpy/tree/main/examples/python) | Calling Python from Klong and Klong from Python; multiprocessing and threading. | |
| [stats_logging](https://github.com/briangu/klongpy/tree/main/examples/stats_logging) | Statistics ingested over IPC into table storage, with periodic flush and reporting. | `klongpy[db]`, `pytz` |
| [stocks](https://github.com/briangu/klongpy/tree/main/examples/stocks) | A simulated market feed published over IPC with VWAP and a moving average, Black-Scholes, and LSTM forecasting on Yahoo Finance data. | `yfinance` for the Yahoo Finance examples only; the feed needs nothing |
| [web](https://github.com/briangu/klongpy/tree/main/examples/web) | An HTTP server with GET and POST handlers. | `klongpy[web]` |

## Snippets

### Arithmetic

An expression is read from right to left.

```kgpy
?> 5+3*2 :" Expressions are evaluated from right to left: 3*2 and then + 5
11
```

Define sum and count functions over an array:

```kgpy
?> sum(v)::{+/v}  :" sum + over / the array v
:monad
?> sum([1 2 3])
6
?> count(v)::{#v}
:monad
?> count([1 2 3])
3
```

The average is the sum divided by the number of elements:

<!-- check-docs: continue -->
```kgpy
?> avg(v)::{sum(v)%count(v)} :" average is the sum divided by the number of elements
:monad
?> avg([1 2 3])
2.0
```

### Math on arrays

An operator applied to a whole array does the loop in the array library. `{x*x}'` calls a function once per element:

```kgpy
?> {x*x}'[1 2 3 4 5] :" square each element as we iterate over the array
[1 4 9 16 25]
```

`a*a` multiplies the arrays element by element in one step:

```kgpy
?> a::[1 2 3 4 5];a*a  :" a*a multiplies the arrays
[1 4 9 16 25]
```

The whole-array form is much faster on large arrays. [Performance](performance.md) has measurements.

### Python integration

Python data can be put into the interpreter, and Klong functions called on it:

```python
from klongpy import KlongInterpreter
import numpy as np

data = np.array([1, 2, 3, 4, 5])
klong = KlongInterpreter()
klong['data'] = data         # the Klong name data holds the array
klong('avg(v)::{(+/v)%#v}')  # define a function in Klong
r = klong('avg(data)')       # call it on the array
print(r)                     # 3.0
```

`.py` and `.pyf` import Python modules and functions, so Klong can call libraries that already exist. This uses the NumPy FFT:

```kgpy
?> .pyf("numpy";"fft");fft::.pya(fft;"fft")
:monad
?> signal::[0.0 1.0 0.0 -1.0] :" Example simple signal
[0.0 1.0 0.0 -1.0]
?> result::fft(signal)
[0j -2j 0j 2j]
```

### Tables and SQL

This needs `pip install "klongpy[db]"`. DuckDB queries the arrays of a table without copying them.

<!-- check-docs: requires duckdb pandas -->
```kgpy
?> .py("klongpy.db")
1
?> t::.table([["name" ["Alice" "Bob"]] ["age" [25 30]]])
 name age
Alice  25
  Bob  30
?> db::.db(:{;"T";t})
:db
?> db("select * from T where age > 27")
["Bob" 30]
```

### IPC and remote calls

An IPC connection to a server is a function. Calling a remote function by name runs it on the server with your arguments. The server here has an `avg` function.

Start a server in one terminal:

```kgpy
server?> avg(v)::{(+/v)%#v}
:monad
server?> .srv(8888)
1
```

Start the client and connect to the server as `f`. To pass parameters to a remote function, form a list of the function symbol followed by the parameters, such as `:avg,,!100`:

<!-- check-docs: continue -->
```kgpy
client?> f::.cli(8888) :" connect to the server
remote[localhost:8888]:fn
client?> f(:avg,,!100) :" call the remote function avg directly with the parameter !100
49.5
```

A local proxy for the remote function:

<!-- check-docs: continue -->
```kgpy
client?> myavg::f(:avg) :" reference the remote function by its symbol :avg and assign to a local variable called myavg
remote[localhost:8888]:fn:avg:monad
client?> myavg(!100) :" this runs on the server with !100 array passed to it as a parameter
49.5
```

A remote call can be wrapped so that a callback runs when it completes:

<!-- check-docs: continue -->
```kgpy
client?> afn::.async(myavg;{.pf("Avg calculated: {}";x)})
async::monad
client?> afn(!100)
Avg calculated: 49.5
1
```

### A web server

This needs `pip install "klongpy[web]"`.

Put this in `web.kg`. It adds one index handler:

```text
.py("klongpy.web")
data::!10
index(request)::{"Hello, Klong World! ",data}
.web(8888;:{;"/";index};:{})
.p("ready at http://localhost:8888")
```

Run it with `kgpy web.kg`. The run stays up while the server listens, until Ctrl-C or `.webc`:

```bash
$ kgpy web.kg
ready at http://localhost:8888
```

In another terminal:

```bash
$ curl http://localhost:8888
['Hello, Klong World! ' 0 1 2 3 4 5 6 7 8 9]
```

### Gradients

`f:>x` is the gradient of `f` at `x`. It is autograd on the torch backend (`kgpy --backend torch`, which needs `klongpy[torch]`) and numeric on NumPy, which is what the outputs below show:

```kgpy
?> f(t)::{t^2}      :" Define f(t) = t^2
:monad
?> f:>3            :" the derivative at 3 is 6, up to the numeric error of the NumPy backend
6.000000000838668
```

For a list input the gradient has one entry per element:

```kgpy
?> h(v)::{+/v^2}    :" h(v) = sum of squares
:monad
?> h:>[1.0 2.0 3.0]   :" Gradient: [2*1, 2*2, 2*3]
[2.000000000279556 3.9999999996709334 6.000000000838668]
```

Gradient descent on x^2 from 5.0, 100 steps of 0.1:

```kgpy
?> f(t)::{t^2}
:monad
?> s::5.0
5.0
?> {s::s-(0.1*f:>s)}'!100;s
1.0185179880729389e-09
```

Here `f:>s` is the gradient of `f` at `s`, by autograd on the torch backend and by finite differences on NumPy. The dyad `s∇f` and `.grad` are finite differences on every backend. The other spellings are in [PyTorch Backend and Autograd](torch_backend.md).

Linear regression and neural networks are in the [autograd examples](https://github.com/briangu/klongpy/tree/main/examples/autograd).

## See also

- [Start here](getting-started.md) and the [Tutorial](tutorial.md) for the language.
- [What's new in v2](whats-new-v2.md) for the changes of this release.
- [What runs where](what-runs-where.md) and [Native runtime](native/index.md) for running Klong without Python.
