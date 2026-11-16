
![Unit Tests](https://github.com/briangu/klongpy/workflows/Unit%20Tests/badge.svg)
[![Last Commit](https://img.shields.io/github/last-commit/briangu/klongpy)](https://img.shields.io/github/last-commit/briangu/klongpy)
[![Dependency Status](https://img.shields.io/librariesio/github/briangu/klongpy)](https://libraries.io/github/briangu/klongpy)
[![Open Issues](https://img.shields.io/github/issues-raw/briangu/klongpy)](https://github.com/briangu/klongpy/issues)
[![Repo Size](https://img.shields.io/github/repo-size/briangu/klongpy)](https://img.shields.io/github/repo-size/briangu/klongpy)
[![GitHub star chart](https://img.shields.io/github/stars/briangu/klongpy?style=social)](https://star-history.com/#briangu/klongpy)

[![Release Notes](https://img.shields.io/github/release/briangu/klongpy)](https://github.com/briangu/klongpy/releases)
[![Downloads](https://static.pepy.tech/badge/klongpy/month)](https://pepy.tech/project/klongpy)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# KlongPy: A High-Performance Array Language with Autograd

KlongPy is a Python adaptation of the [Klong](https://t3x.org/klong) [array language](https://en.wikipedia.org/wiki/Array_programming), offering high-performance vectorized operations. It prioritizes compatibility with Python, thus allowing seamless integration of Python's expansive ecosystem while retaining Klong's succinctness.

KlongPy backends include [NumPy](https://numpy.org/) and optional [PyTorch](https://pytorch.org/) (CPU, CUDA, and Apple MPS).
When PyTorch is enabled, automatic differentiation (autograd) is supported; otherwise, numeric differentiation is the default.

Full documentation: [https://klongpy.org](https://klongpy.org)

New to v0.7.0, KlongPy now brings gradient-based programming to an already-succinct array language, so you can differentiate compact array expressions directly. It's also a batteries-included system with IPC, DuckDB-backed database tooling, web/websocket support, and other integrations exposed seamlessly from the language.

Backends include NumPy and optional PyTorch (CPU, CUDA, and Apple MPS). When PyTorch is enabled, gradients use autograd; otherwise numeric differentiation is the default.

**PyTorch gradient descent (10+ lines):**
```python
import torch
x = torch.tensor(5.0, requires_grad=True)
optimizer = torch.optim.SGD([x], lr=0.1)
for _ in range(100):
    loss = x ** 2
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
print(x)  # ~0
```

**KlongPy gradient descent (2 lines):**
```klong
f::{x^2}; s::5.0
{s::s-(0.1*f:>s)}'!100   :" s -> 0"
```

Array languages like APL, K, and Q revolutionized finance by treating operations as data transformations, not loops. KlongPy brings this philosophy to machine learning: gradients become expressions you compose, not boilerplate you maintain. The result is a succint mathematical-like notation that is automatically extended to machine learning.

## Quick Install

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

## REPL

```bash
$ kgpy
Welcome to KlongPy REPL v0.7.0
Author: Brian Guarraci
Web: http://klongpy.org
Backend: torch (mps)
]h for help; Ctrl-D or ]q to quit

$>
```

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

## The Array Language Advantage

Array languages express *what* you want, not *how* to compute it. This enables automatic optimization:

| Operation | Python | KlongPy |
|-----------|--------|---------|
| Sum an array | `sum(a)` | `+/a` |
| Running sum | `np.cumsum(a)` | `+\a` |
| Dot product | `np.dot(a,b)` | `+/a*b` |
| Average | `sum(a)/len(a)` | `(+/a)%#a` |
| Gradient | *10+ lines* | `f:>x` |
| Multi-param grad | *20+ lines* | `loss:>[w b]` |
| Jacobian | *15+ lines* | `x∂f` |
| Optimizer | *10+ lines* | `{w::w-(lr*f:>w)}` |

KlongPy inherits from the [APL](https://en.wikipedia.org/wiki/APL_(programming_language)) family tree (APL → J → K/Q → Klong), adding Python integration and automatic differentiation.

## Performance

Run the included benchmark on any backend:

```bash
kgpy --backend torch --device cpu examples/bench_compiler.kg
kgpy --backend numpy examples/bench_compiler.kg
```

### Expression Compiler Benchmark (Apple M1 Mac Studio)

Both backends include an expression compiler that converts Klong ASTs to a backend-neutral IR, then generates platform-specific Python functions. Expressions compile once and are cached — subsequent calls pay only execution cost.

Times are per-call averages over 1000 iterations.

| Operation | Elements | NumPy | Torch CPU |
|-----------|------:|------:|----------:|
| **Arithmetic** | | | |
| `a+b` | 100K | 0.066 ms | 0.155 ms |
| `a*2+b` | 100K | 0.091 ms | 0.276 ms |
| `(a+b)*(a-b)` | 100K | 0.124 ms | 0.390 ms |
| `(a*2+b*3)%(a+1)-(b*c)` | 100K | 0.218 ms | 0.870 ms |
| **Lambdas** | | | |
| `{x+y}(a;b)` | 100K | 0.074 ms | 0.156 ms |
| `{(x+y)*(x-y)}(a;b)` | 100K | 0.129 ms | 0.390 ms |
| `{+/x*y}(a;b)` | 100K | 0.098 ms | 0.285 ms |
| **Reduce** | | | |
| `+/a` | 100K | 0.065 ms | 0.150 ms |
| `+/a*b` | 100K | 0.095 ms | 0.293 ms |
| **Scan** | | | |
| `+\ts` cumsum | 10K | 0.083 ms | 0.054 ms |
| `|\ts` running max | 10K | 24.63 ms | 0.068 ms |
| `(|\ts)-ts` drawdown | 10K | 24.65 ms | 0.072 ms |
| `|/(|\ts)-ts` max drawdown | 10K | 24.68 ms | 0.081 ms |
| **Real-world** | | | |
| `(+/p*s)%+/s` VWAP | 100K | 0.151 ms | 0.512 ms |
| `vwap::{(+/x*y)%+/y}; vwap(p;s)` | 100K | 0.151 ms | 0.489 ms |
| `a-(+/a)%#a` de-mean | 100K | 0.088 ms | 0.292 ms |

NumPy is faster for element-wise and reduce operations on CPU. Torch excels at scan operations where it compiles to native tensor methods (`cummax`, `cumsum`) — running max is **362x faster** than NumPy's interpreter fallback. Torch's full advantage appears on GPU (`--device cuda` or `--device mps`).

## Complete Feature Set

KlongPy is a batteries-included platform with kdb+/Q-inspired features:

### Core Language
- **Vectorized Operations**: NumPy/PyTorch-powered bulk array operations
- **Automatic Differentiation**: Native `:>` operator for exact gradients
- **GPU Acceleration**: CUDA and Apple MPS support via PyTorch
- **Python Integration**: Import any Python library with `.py()` and `.pyf()`

### Data Infrastructure (kdb+/Q-like)
- **[Fast Columnar Database](docs/fast_columnar_database.md)**: Zero-copy DuckDB integration for SQL on arrays
- **[Inter-Process Communication](docs/ipc_capabilities.md)**: Build ticker plants and distributed systems
- **[Table & Key-Value Store](docs/table_and_key_value_stores.md)**: Persistent storage for tables and data
- **[Web Server](docs/web_server.md)**: Built-in HTTP server for APIs and dashboards
- **[WebSockets](docs/websockets.md)**: Connect to WebSocket servers and handle messages in KlongPy
- **[Timers](docs/timer.md)**: Scheduled execution for periodic tasks

### Documentation
- **[Quick Start Guide](docs/quick-start.md)**: Get running in 5 minutes
- **[Tutorial](docs/tutorial.md)**: The language from the first expression to functions, adverbs, dictionaries and files
- **[Klong for NumPy Users](docs/from-numpy.md)** and **[Klong for q Users](docs/from-kdb.md)**: Translation tables and the differences that cause mistakes
- **[Friendlier Syntax](docs/language-additions.md)**: Destructuring, named parameters, evaluated dictionaries, `.fmt`
- **[Strict Mode](docs/strict.md)** and **[Lint](docs/lint.md)**: Catch mistakes early
- **[Compiling Klong to C](docs/embedded.md)**: A static subset of Klong that compiles to freestanding C99 for small targets
- **[PyTorch Backend & Autograd](docs/torch_backend.md)**: Complete autograd reference
- **[Operator Reference](docs/operators.md)**: All language operators
- **[Performance Guide](docs/performance.md)**: Optimization tips

Full documentation: [https://briangu.github.io/klongpy](https://briangu.github.io/klongpy)

## Security Model: Trusted Networks

KlongPy's network features assume every peer is trusted. Use them on a trusted network only.

- **IPC (`.srv`, `.cli`)**: a connected peer can run arbitrary Klong, and through it arbitrary Python (`.py`, `.pyf`) and shell commands (`.sys`). It can also call any function you have defined and assign any variable.
- **IPC wire format**: the default path speaks the kdb+ IPC protocol and does not unpickle anything from the network. Message size is limited (`.system` key `ipc_max_message`, 256 MiB by default). The legacy pickle mode (`KLONGPY_IPC=legacy`, or a `.srv` left on `auto` that sees an old client) still unpickles messages, which runs code chosen by the sender, and is deprecated. Use it on trusted networks only, or set `KLONGPY_IPC=kdb` to refuse it.
- **No encryption, optional login**: IPC credentials (`"host:port:user:password"`) and all data are sent in clear text and there is no TLS. A server accepts every client unless the program defines `.srv.auth`. `.web` has no login or access control.
- **Bind address**: `.srv(8888)` and `.web(8888;...)` listen on all interfaces (`0.0.0.0`). Anyone who can reach the port has the access above. `.web` routes run the Klong functions you register with the request parameters.
- **KVS and tables**: keys are used as file names under the store root without checking that they stay inside it, and values are pickled on disk. Treat the root directory as trusted storage and do not load stores you did not write.
- **`.db`**: the string you pass is executed as SQL by DuckDB as given.

To limit exposure:

- Bind to localhost: `.srv("127.0.0.1:8888")`, `.web("127.0.0.1:8080";get;post)`. `kgpy -s ARG` evaluates `.srv(ARG)`, so to bind to localhost pass a quoted string: `kgpy -s '"127.0.0.1:8888"'`.
- Firewall the ports, and never expose a KlongPy port to an untrusted network or the internet.
- Between hosts, use an SSH tunnel or VPN.

## Typing Special Characters

KlongPy uses Unicode operators for mathematical notation. Here's how to type them:

| Symbol | Name | Mac | Windows | Description |
|--------|------|-----|---------|-------------|
| `∇` | Nabla | `Option + v` then select, or Character Viewer | `Alt + 8711` (numpad) | Numeric gradient |
| `∂` | Partial | `Option + d` | `Alt + 8706` (numpad) | Jacobian operator |

**Mac Tips:**
- **Option + d** types `∂` directly
- For `∇`, open Character Viewer with **Ctrl + Cmd + Space**, search "nabla"
- Or simply copy-paste: `∇` `∂`

**Alternative:** Use the function equivalents that don't require special characters:
```klong
3∇f            :" Using nabla"
.grad(f;3)     :" Instead of 3∇f"
.jacobian(f;x) :" Instead of x∂f"
```

Note the argument order: `.grad` and `.jacobian` take the function first, while `∇` and `∂` take the point first.

## Syntax Cheat Sheet

Functions take up to 3 parameters, named `x`, `y`, `z`. A function has as many parameters as the highest of the three that its body uses. For more than three, or for names you choose, write `f(a;b;c;d)::{...}` (see [Friendlier Syntax](docs/language-additions.md#named-parameters)):

```klong
:" Operators (right to left evaluation)"
5+3*2           :" 11 (3*2 first, then +5)"
+/[1 2 3]       :" 6  (sum: + over /)"
*/[1 2 3]       :" 6  (product: * over /)"
#[1 2 3]        :" 3  (length)"
3|5             :" 5  (max)"
3&5             :" 3  (min)"

:" Functions"
avg::{(+/x)%#x}         :" Monad (1 arg)"
dot::{+/x*y}            :" Dyad (2 args)"
clip::{(x|y)&z}         :" Triad (3 args): min(max(x,y),z)"

:" Adverbs (modifiers)"
f::{x^2}
f'[1 2 3]               :" Each: apply f to each -> [1 4 9]"
+/[1 2 3]               :" Over: fold/reduce -> 6"
+\[1 2 3]               :" Scan: running fold -> [1 3 6]"
{x@0}@'[10 20 30]       :" Each-Index: enumerate -> [0 1 2]"

:" Evaluated arrays (expressions evaluated at construction)"
a::10; b::20
[;a;b;a+b]              :" -> [10 20 30]"

:" Autograd"
f::{x^2}
3∇f                     :" Numeric gradient at x=3 -> ~6.0"
f:>3                    :" Autograd (exact with torch) at x=3 -> 6.0"
f::{+/x^2}             :" Redefine f as sum-of-squares"
f:>[1 2 3]              :" Gradient -> [2 4 6]"

:" Multi-parameter gradients"
w::2.0; b::3.0
loss::{(w^2)+(b^2)}
loss:>[w b]             :" Gradients for both -> [4.0 6.0]"

:" Jacobian (for vector functions)"
g::{x^2}                :" Element-wise square"
[1 2]∂g                 :" Jacobian matrix -> [[2 0] [0 4]]"
```

## Examples

### 1. Basic Array Operations

```klong
?> a::[1 2 3 4 5]
[1 2 3 4 5]
?> a*a                    :" Element-wise square"
[1 4 9 16 25]
?> +/a                    :" Sum"
15
?> (*/a)                  :" Product"
120
?> avg::{(+/x)%#x}        :" Define average"
:monad
?> avg(a)
3.0
```

### 2. Gradient Descent

Minimize f(x) = (x-3)^2

(with PyTorch's autograd)
```klong
$ kgpy --backend torch
?> f::{(x-3)^2}
:monad
?> s::10.0; lr::0.1
0.1
?> {s::s-(lr*f:>s); s}'!10
[8.600000381469727 7.4800004959106445 6.584000587463379 5.8672003746032715 5.293760299682617 4.835008144378662 4.468006610870361 4.174405097961426 3.9395241737365723 3.751619338989258]
```

(Numerical differentiation)
```klong
$ kgpy
?> f::{(x-3)^2}
:monad
?> s::10.0; lr::0.1
0.1
?> {s::s-(lr*f:>s); s}'!10
[8.60000000104776 7.480000001637279 6.584000001220716 5.867200000887465 5.2937600006031005 4.835008000393373 4.4680064002611175 4.174405120173077 3.939524096109306 3.7516192768605094]
```

### 3. Linear Regression

```klong
:" Data: y = 2*x + 3 + noise"
X::[1 2 3 4 5]
Y::[5.1 6.9 9.2 10.8 13.1]

:" Model parameters"
w::0.0; b::0.0

:" Loss function"
mse::{(+/(((w*X)+b)-Y)^2)%#X}

:" Train with multi-parameter gradients"
lr::0.01
{grads::mse:>[w b]; w::w-(lr*grads@0); b::b-(lr*grads@1)}'!1000

.d("Learned: w="); .d(w); .d(" b="); .p(b)
Learned: w=2.01 b=2.97
```

### 4. Database Operations

<!-- check-docs: requires duckdb pandas -->
```klong
?> .py("klongpy.db")
1
?> t::.table([["name" ["Alice" "Bob" "Carol"]] ["age" [25 30 35]]])
 name age
Alice  25
  Bob  30
Carol  35
?> db::.db(:{},"T",,t)
:db
?> db("SELECT * FROM T WHERE age > 27")
[["Bob" 30] ["Carol" 35]]
```

### 5. IPC: Distributed Computing

**Server:**
```klong
?> avg::{(+/x)%#x}
:monad
?> .srv(8888)
1
```

**Client:**
```klong
?> f::.cli(8888)              :" Connect to server"
remote[localhost:8888]:fn
?> myavg::f(:avg)             :" Get remote function reference"
remote[localhost:8888]:fn:avg:monad
?> myavg(!1000000)            :" Execute on server"
499999.5
```

### 6. Web Server

```klong
.py("klongpy.web")
data::!10
index::{x; "Hello from KlongPy! Data: ",data}
get:::{}; get,"/",index
post:::{}
h::.web(8888;get;post)
.p("Server ready at http://localhost:8888")
```

```bash
$ curl http://localhost:8888
['Hello from KlongPy! Data: ' 0 1 2 3 4 5 6 7 8 9]
```

## Installation Options

### Basic Runtime (NumPy only)
```bash
pip install klongpy
```

### REPL Support
```bash
pip install "klongpy[repl]"
```

### With PyTorch Autograd (Recommended)
```bash
pip install "klongpy[torch]"
kgpy --backend torch          # Enable torch backend
```

### Web / DB / WebSockets Extras
```bash
pip install "klongpy[web]"
pip install "klongpy[db]"
pip install "klongpy[ws]"
```

### Full Installation (REPL, DB, Web, WebSockets, Torch)
```bash
pip install "klongpy[all]"
```

## Lineage and Inspiration

KlongPy stands on the shoulders of giants:

- **[APL](https://en.wikipedia.org/wiki/APL_(programming_language))** (1966): Ken Iverson's revolutionary notation
- **[J](https://www.jsoftware.com/)**: ASCII-friendly APL successor
- **[K/Q/kdb+](https://code.kx.com/)**: High-performance time series and trading systems
- **[Klong](https://t3x.org/klong)**: Nils M Holm's elegant, accessible array language
- **[NumPy](https://numpy.org/)**: The "Iverson Ghost" in Python's scientific stack
- **[PyTorch](https://pytorch.org/)**: Automatic differentiation and GPU acceleration

KlongPy combines Klong's simplicity with Python's ecosystem and PyTorch's autograd creating something new: an array language where gradients are first-class citizens.

## Use Cases

- **Quantitative Finance**: Self-optimizing trading strategies, risk models, portfolio optimization
- **Machine Learning**: Neural networks, gradient descent, optimization in minimal code
- **Scientific Computing**: Physics simulations, numerical methods, data analysis
- **Time Series Analysis**: Signal processing, feature engineering, streaming data
- **Rapid Prototyping**: Express complex algorithms in few lines, then optimize

## Status

KlongPy implements the Klong array language and runs the Klong test suite, plus additional test suites. It differs from Klong in a few ways, listed in the documentation. The PyTorch backend provides GPU acceleration (CUDA, MPS) and automatic differentiation.

Ongoing development:
- Expanded torch backend coverage
- Additional built-in tools and integrations
- Improved error messages and debugging

## Related Projects

- [Klupyter](https://github.com/briangu/klupyter) - KlongPy in Jupyter Notebooks
- [VS Code Syntax Highlighting](https://github.com/briangu/klongpy-vscode)
- [Advent of Code Solutions](https://github.com/briangu/aoc)

## Development

```bash
git clone https://github.com/briangu/klongpy.git
cd klongpy
pip install -e ".[dev]"   # Install in editable mode with dev dependencies
python3 -m pytest tests/  # Run tests
```

## Issues

This project does not accept direct issue submissions.

Please start with a GitHub Discussion.
Maintainers will promote validated discussions to Issues.

Active contributors may be invited to open issues directly.

## Contributors

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution workflow, discussion-first policy, and code standards.

## Documentation

```bash
# Install docs tooling
pip install -e ".[docs]"

# Build the site into ./site
mkdocs build

# Serve locally with live reload
mkdocs serve
```

## Acknowledgements

Huge thanks to [Nils M Holm](https://t3x.org) for creating Klong and writing the [Klong Book](https://t3x.org/klong/book.html), which made this project possible.
