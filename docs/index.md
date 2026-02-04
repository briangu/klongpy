# KlongPy: A High-Performance Array Language with Autograd

KlongPy is a Python adaptation of the [Klong](https://t3x.org/klong) [array language](https://en.wikipedia.org/wiki/Array_programming), offering high-performance vectorized operations. It prioritizes compatibility with Python, thus allowing seamless integration of Python's expansive ecosystem while retaining Klong's succinctness.

KlongPy backends include [NumPy](https://numpy.org/) and optional [PyTorch](https://pytorch.org/) (CPU, CUDA, and Apple MPS).
When PyTorch is enabled, automatic differentiation (autograd) is supported; otherwise, numeric differentiation is the default.

New to v0.7.0, KlongPy now brings gradient-based programming to an already-succinct array language, so you can differentiate compact array expressions directly. It's also a batteries-included system with IPC, DuckDB-backed database tooling, web/websocket support, and other integrations exposed seamlessly from the language.

Array languages like APL, K, and Q revolutionized finance by treating operations as data transformations, not loops. KlongPy brings this philosophy to machine learning: gradients become expressions you compose, not boilerplate you maintain. The result is a succinct mathematical-like notation that is automatically extended to machine learning.

Source code on GitHub: [briangu/klongpy](https://github.com/briangu/klongpy)

# Quick Install

```bash
# REPL + NumPy backend (pick one option below)
pip install "klongpy[repl]"
kgpy

# Enable torch backend (autograd + GPU)
pip install "klongpy[torch]"
kgpy --backend torch

# Everything (web, db, websockets, torch, repl)
pip install "klongpy[all]"
```

New users may want to read the [Quick Start](quick-start.md) guide and the
[REPL Reference](repl.md) to get familiar with the interactive environment.
Note: the REPL (`kgpy`) requires the `klongpy[repl]` extra (or `klongpy[all]`).

# Overview

KlongPy is a batteries-included platform with kdb+/Q-inspired features:

### Core Language
- **Vectorized Operations**: NumPy/PyTorch-powered bulk array operations
- **Automatic Differentiation**: Native `:>` operator for exact gradients
- **GPU Acceleration**: CUDA and Apple MPS support via PyTorch
- **Python Integration**: Import any Python library with `.py()` and `.pyf()`
- **[Speed](performance.md)**: High-performance vectorized computing on CPU or GPU

### Data Infrastructure (kdb+/Q-like)
- **[Fast Columnar Database](fast_columnar_database.md)**: Zero-copy DuckDB integration for SQL on arrays
- **[Inter-Process Communication](ipc_capabilities.md)**: Build ticker plants and distributed systems
- **[Table & Key-Value Store](table_and_key_value_stores.md)**: Persistent storage for tables and data
- **[Web Server](web_server.md)**: Built-in HTTP server for APIs and dashboards
- **[WebSockets](websockets.md)**: Connect to WebSocket servers and handle messages in KlongPy
- **[Timers](timer.md)**: Scheduled execution for periodic tasks

### Documentation
- **[PyTorch Backend & Autograd](torch_backend.md)**: Complete autograd reference
- **[Operator Reference](operators.md)**: All language operators
- **[Performance Guide](performance.md)**: Optimization tips

# Examples

Consider this simple Klong expression that computes an array's average: `(+/a)%#a`. Decoded, it means "sum of 'a' divided by the length of 'a'", as read from right to left.

Below, we define the function 'avg' and apply it to the array of 1 million integers (as defined by !1000000)

Let's try this in the KlongPy REPL:

```Bash
$ rlwrap kgpy

Welcome to KlongPy REPL v0.7.0
author: Brian Guarraci
repo  : https://github.com/briangu/klongpy
Ctrl-D or ]q to quit

?> avg::{(+/x)%#x}
:monad
?> avg(!1000000)
499999.5
```

Now let's time it (first, run it once, then 100 times):

```
?> ]T avg(!1000000)
total: 0.0032962500117719173 per: 0.0032962500117719173
?> ]T:100 avg(!1000000)
total: 0.10882879211567342 per: 0.0010882879211567343
```

We can also import Python custom or standard modules to use directly in Klong language.

```
?> .pyf("math";"pi")
1
?> pi
3.141592653589793
```

Here we import the fsum function from standard Python math library and redefine avg to use 'fsum':

```
?> .pyf("math";"fsum")
1
?> favg::{fsum(x)%#x}
:monad
?> favg(!1000000)
499999.5
```

Notice that using fsum is slower than using Klong '+/'.  This is because the '+/' operation is vectorized while fsum is not.

```
?> ]T favg(!1000000)
total: 0.050078875152394176 per: 0.050078875152394176
?> ]T:100 favg(!1000000)
total: 2.93945804098621 per: 0.029394580409862103
```

To use KlongPy within Python, here's a basic outline:

```python
from klongpy import KlongInterpreter

# instantiate the KlongPy interpreter
klong = KlongInterpreter()

# define average function in Klong (Note the '+/' (sum over) uses np.add.reduce under the hood)
klong('avg::{(+/x)%#x}')

# create a billion random uniform values [0,1)
data = np.random.rand(10**9)

# reference the 'avg' function in Klong interpreter and call it directly from Python.
r = klong['avg'](data)

print(f"avg={np.round(r,6)}")
```

And let's run a performance comparison between NumPy and PyTorch backends:

```python
import time
from klongpy import KlongInterpreter

# Use torch backend
klong = KlongInterpreter(backend='torch')
klong('avg::{(+/x)%#x}')

# Create large array
data = klong('!1000000')

start = time.perf_counter_ns()
r = klong['avg'](data)
stop = time.perf_counter_ns()

print(f"avg={float(r):.6f} in {round((stop - start) / (10**9), 6)} seconds")
print(f"Backend: {klong._backend.name}, Device: {klong._backend.device}")
```

Run with PyTorch:

    $ python3 example.py
    avg=499999.5 in 0.001234 seconds
    Backend: torch, Device: mps:0

## Why KlongPy?

### For Quants and Traders

Optimize portfolios with gradients in a language designed for arrays:

```klong
:" Portfolio optimization: gradient of Sharpe ratio"
returns::[0.05 0.08 0.03 0.10]      :" Annual returns per asset"
vols::[0.15 0.20 0.10 0.25]         :" Volatilities per asset"
w::[0.25 0.25 0.25 0.25]            :" Portfolio weights"

sharpe::{(+/x*returns)%((+/((x^2)*(vols^2)))^0.5)}
sg::sharpe:>w                       :" Gradient of Sharpe ratio"
.d("sharpe gradient="); .p(sg)
sharpe gradient=[0.07257738709449768 0.032256484031677246 0.11693036556243896 -0.22176480293273926]
```

### For ML Researchers

Neural networks in pure array notation:

```klong
:" Single-layer neural network with gradient descent"
.bkf(["exp"])
sigmoid::{1%(1+exp(0-x))}
forward::{sigmoid((w1*x)+b1)}
X::[0.5 1.0 1.5 2.0]; Y::[0.2 0.4 0.6 0.8]
w1::0.1; b1::0.1; lr::0.1
loss::{+/((forward'X)-Y)^2}

:" Train with multi-param gradients"
{grads::loss:>[w1 b1]; w1::w1-(lr*grads@0); b1::b1-(lr*grads@1)}'!1000
.d("w1="); .d(w1); .d(" b1="); .p(b1)
w1=1.74 b1=-2.17
```

### For Scientists

Express mathematics directly:

```klong
:" Gradient of f(x,y,z) = x^2 + y^2 + z^2 at [1,2,3]"
f::{+/x^2}
f:>[1 2 3]
[2.0 4.0 6.0]
```

Enable the PyTorch backend with `--backend torch` or programmatically via `KlongInterpreter(backend="torch", device="cuda")`.

See [PyTorch Backend & Autograd](torch_backend.md) for more details and the [autograd examples](https://github.com/briangu/klongpy/tree/main/examples/autograd) for complete examples including gradient descent and neural networks.

# Installation

### CPU

    $ pip3 install klongpy

### PyTorch Backend (recommended for autograd and GPU)

    $ pip3 install "klongpy[torch]"

Then use the `--backend` flag:

    $ python your_script.py  # (with KlongInterpreter(backend='torch'))
    $ kgpy --backend torch

### All application tools (db, web, REPL, websockets, etc.)

    $ pip3 install "klongpy[all]"


# REPL

KlongPy has a REPL similar to Klong's REPL.

```bash
$ pip3 install klongpy[repl]
$ rlwrap kgpy

Welcome to KlongPy REPL
author: Brian Guarraci
repo  : https://github.com/briangu/klongpy
Ctrl-C to quit

?> 1+1
2
>? "hello, world!"
hello, world!
?> prime::{&/x!:\2+!_x^1%2}
:monad
?> prime(4)
0
?> prime(251)
1
?> ]T prime(251)
total: 0.0004914579913020134 per: 0.0004914579913020134
```

Read about the [prime example here](https://t3x.org/klong/prime.html).


# Status

KlongPy is a superset of the Klong array language, passing all Klong integration tests plus additional test suites. The PyTorch backend provides GPU acceleration (CUDA, MPS) and automatic differentiation.

Ongoing development:
- Expanded torch backend coverage
- Additional built-in tools and integrations
- Improved error messages and debugging

Note: The torch backend does not support object dtype arrays or string operations. Use the numpy backend for these.

# Differences from Klong

KlongPy is effectively a superset of the Klong language, but has some key differences:

* **Infinite precision**: The main difference in this implementation of Klong is the lack of infinite precision. By using NumPy we are restricted to doubles.
* **Python integration**: The `.py()` and `.pyf()` commands allow direct import of Python modules into the current Klong context.
* **IPC**: KlongPy supports IPC between KlongPy processes.
* **Autograd**: Native gradient computation via the `:>` operator with PyTorch backend.

# Use Cases

- **Quantitative Finance**: Self-optimizing trading strategies, risk models, portfolio optimization
- **Machine Learning**: Neural networks, gradient descent, optimization in minimal code
- **Scientific Computing**: Physics simulations, numerical methods, data analysis
- **Time Series Analysis**: Signal processing, feature engineering, streaming data
- **Rapid Prototyping**: Express complex algorithms in few lines, then optimize

# Lineage and Inspiration

KlongPy stands on the shoulders of giants:

- **[APL](https://en.wikipedia.org/wiki/APL_(programming_language))** (1966): Ken Iverson's revolutionary notation
- **[J](https://www.jsoftware.com/)**: ASCII-friendly APL successor
- **[K/Q/kdb+](https://code.kx.com/)**: High-performance time series and trading systems
- **[Klong](https://t3x.org/klong)**: Nils M Holm's elegant, accessible array language
- **[NumPy](https://numpy.org/)**: The "Iverson Ghost" in Python's scientific stack
- **[PyTorch](https://pytorch.org/)**: Automatic differentiation and GPU acceleration

KlongPy combines Klong's simplicity with Python's ecosystem and PyTorch's autograd creating something new: an array language where gradients are first-class citizens.

# Related Projects

- [Klupyter](https://github.com/briangu/klupyter) - KlongPy in Jupyter Notebooks
- [VS Code Syntax Highlighting](https://github.com/briangu/klongpy-vscode)
- [Advent of Code Solutions](https://github.com/briangu/aoc)

# Development

```bash
git clone https://github.com/briangu/klongpy.git
cd klongpy
pip install -e ".[dev]"   # Install in editable mode with dev dependencies
python3 -m pytest tests/  # Run tests
```

# Acknowledgements

Huge thanks to [Nils M Holm](https://t3x.org) for creating Klong and writing the [Klong Book](https://t3x.org/klong/book.html), which made this project possible.
