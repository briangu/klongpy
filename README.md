
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

KlongPy brings automatic differentiation to array programming. Write gradient-based optimization in 2 lines instead of 20. Build self-learning trading systems, neural networks, and scientific computing applications with unprecedented conciseness.

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
f::{x^2}; x::5.0
{x::x-(0.1*f:>x)}'!100   :" x -> 0
```

**Or with custom optimizer (copy from examples/):**
```klong
.pyf("optimizers";"SGDOptimizer")
x::5.0; opt::SGDOptimizer(klong;["x"];:{["lr" 0.1]})
{opt({x^2})}'!100        :" x -> 0
```

This isn't just shorter—it's a fundamentally different way to express computation. Array languages like APL, K, and Q revolutionized finance and data analysis with their concise vectorized operations. KlongPy adds native autograd, making gradients first-class citizens in an array language.

## Quick Install

```bash
# REPL + NumPy backend (pick one option below)
pip install "klongpy[repl]"
kgpy

# Enable torch backend (autograd + GPU)
pip install "klongpy[torch]"
USE_TORCH=1 kgpy            # or KLONG_BACKEND=torch

# Everything (web, db, websockets, torch, repl)
pip install "klongpy[all]"
```

## Why KlongPy?

### For Quants and Traders

Build self-learning trading strategies in a language designed for time series:

```klong
:" Moving average crossover with learned parameters
sma::{(+/x)%#x}
signal::{(sma(n#x))-(sma(m#x))}  :" Difference of moving averages
loss::{+/(signal(prices)-returns)^2}

:" Learn optimal window sizes via gradient descent
loss:>n                           :" Gradient w.r.t. short window
loss:>m                           :" Gradient w.r.t. long window
```

### For ML Researchers

Neural networks in pure array notation:

```klong
:" Single layer: sigmoid(W*x + b)
sigmoid::{1%(1+exp(0-x))}
forward::{sigmoid((w1*x)+b1)}
loss::{+/(forward'X - Y)^2}

:" Train with multi-param gradients
{grads::loss:>[w1 b1]; w1::w1-(lr*grads@0); b1::b1-(lr*grads@1)}'!1000
```

### For Scientists

Express mathematics directly:

```klong
:" Gradient of f(x,y,z) = x^2 + y^2 + z^2 at [1,2,3]
f::{+/x^2}
f:>[1 2 3]    :" [2 4 6] - exact gradient via autograd
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

## Performance: NumPy vs PyTorch Backend

The PyTorch backend provides significant speedups for large arrays with GPU acceleration:

```
Benchmark                   NumPy (ms)   Torch (ms)      Speedup
----------------------------------------------------------------------
vector_add_1M                    0.327        0.065    5.02x (torch)
compound_expr_1M                 0.633        0.070    9.00x (torch)
sum_1M                           0.246        0.087    2.84x (torch)
grade_up_100K                    0.588        0.199    2.96x (torch)
enumerate_1M                     0.141        0.050    2.83x (torch)
```

*Benchmarks on Apple M1 with MPS. Run `python tests/perf_backend.py --compare` for your system.*

## Complete Feature Set

KlongPy isn't just an autograd experiment—it's a production-ready platform with kdb+/Q-inspired features:

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
- **[PyTorch Backend & Autograd](docs/torch_backend.md)**: Complete autograd reference
- **[Operator Reference](docs/operators.md)**: All language operators
- **[Performance Guide](docs/performance.md)**: Optimization tips

Full documentation: [https://briangu.github.io/klongpy](https://briangu.github.io/klongpy)

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
3∇f           :" Using nabla
.jacobian(f;x) :" Instead of x∂f
```

## Syntax Cheat Sheet

Functions take up to 3 parameters, always named `x`, `y`, `z`:

```klong
:" Operators (right to left evaluation)
5+3*2           :" 11 (3*2 first, then +5)
+/[1 2 3]       :" 6  (sum: + over /)
*/[1 2 3]       :" 6  (product: * over /)
#[1 2 3]        :" 3  (length)
|[3 1 2]        :" [1 2 3] (sort)
&[1 0 1]        :" [0 2] (where/indices of true)

:" Functions
avg::{(+/x)%#x}         :" Monad (1 arg)
dot::{+/x*y}            :" Dyad (2 args)
clip::{x|y&z}           :" Triad (3 args): min(max(x,y),z)

:" Adverbs (modifiers)
f'[1 2 3]               :" Each: apply f to each element
1 2 3 +'[10 20 30]      :" Each-pair: [11 22 33]
+/[1 2 3]               :" Over: fold/reduce
+\[1 2 3]               :" Scan: running fold [1 3 6]

:" Autograd
f::{x^2}
3∇f                     :" Numeric gradient at x=3 -> ~6.0
f:>3                    :" Autograd (exact with torch) at x=3 -> 6.0
f:>[1 2 3]              :" Gradient of sum-of-squares -> [2 4 6]

:" Multi-parameter gradients
w::2.0; b::3.0
loss::{(w^2)+(b^2)}
loss:>[w b]             :" Gradients for both -> [4.0 6.0]

:" Jacobian (for vector functions)
g::{x^2}                :" Element-wise square
[1 2]∂g                 :" Jacobian matrix -> [[2 0] [0 4]]
```

## Examples

### 1. Basic Array Operations

```klong
?> a::[1 2 3 4 5]
[1 2 3 4 5]
?> a*a                    :" Element-wise square
[1 4 9 16 25]
?> +/a                    :" Sum
15
?> (*/a)                  :" Product
120
?> avg::{(+/x)%#x}        :" Define average
:monad
?> avg(a)
3.0
```

### 2. Gradient Descent

```klong
:" Minimize f(x) = (x-3)^2
?> f::{(x-3)^2}
:monad
?> x::10.0; lr::0.1
0.1
?> {x::x-(lr*f:>x); x}'!10    :" 10 gradient steps
[8.6 7.48 6.584 5.867 5.294 4.835 4.468 4.175 3.940 3.752]
```

### 3. Linear Regression

```klong
:" Data: y = 2*x + 3 + noise
X::[1 2 3 4 5]
Y::[5.1 6.9 9.2 10.8 13.1]

:" Model parameters
w::0.0; b::0.0

:" Loss function
mse::{(+/(((w*X)+b)-Y)^2)%#X}

:" Train with multi-parameter gradients
lr::0.01
{grads::mse:>[w b]; w::w-(lr*grads@0); b::b-(lr*grads@1)}'!1000

.d("Learned: w="); .d(w); .d(" b="); .p(b)
:" Output: Learned: w=2.02 b=2.94
```

**Or with custom optimizer (copy from examples/autograd/optimizers.py):**
```klong
.pyf("optimizers";"AdamOptimizer")
w::0.0; b::0.0
opt::AdamOptimizer(klong;["w" "b"];:{["lr" 0.01]})
{opt(mse)}'!1000         :" Optimizer handles gradient computation
```

### 4. Database Operations

```klong
?> .py("klongpy.db")
?> t::.table([[\"name\" [\"Alice\" \"Bob\" \"Carol\"]] [\"age\" [25 30 35]]])
name  age
Alice  25
Bob    30
Carol  35
?> db::.db(:{},\"T\",t)
?> db(\"SELECT * FROM T WHERE age > 27\")
name  age
Bob    30
Carol  35
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
?> f::.cli(8888)              :" Connect to server
remote[localhost:8888]:fn
?> myavg::f(:avg)             :" Get remote function reference
remote[localhost:8888]:fn:avg:monad
?> myavg(!1000000)            :" Execute on server
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
USE_TORCH=1 kgpy              # Enable torch backend (or KLONG_BACKEND=torch)
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

KlongPy combines Klong's simplicity with Python's ecosystem and PyTorch's autograd—creating something new: an array language where gradients are first-class citizens.

## Use Cases

- **Quantitative Finance**: Self-optimizing trading strategies, risk models, portfolio optimization
- **Machine Learning**: Neural networks, gradient descent, optimization in minimal code
- **Scientific Computing**: Physics simulations, numerical methods, data analysis
- **Time Series Analysis**: Signal processing, feature engineering, streaming data
- **Rapid Prototyping**: Express complex algorithms in few lines, then optimize

## Status

KlongPy is a superset of the Klong array language, passing all Klong integration tests plus additional test suites. The PyTorch backend provides GPU acceleration (CUDA, MPS) and automatic differentiation.

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
