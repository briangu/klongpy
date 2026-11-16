# KlongPy: the Klong array language in Python

KlongPy is a Python implementation of [Klong](https://t3x.org/klong), an [array language](https://en.wikipedia.org/wiki/Array_programming). You write `+/a` for the sum of an array instead of a loop, and the work runs on [NumPy](https://numpy.org/) or, optionally, [PyTorch](https://pytorch.org/) (CPU, CUDA and Apple MPS). With the torch backend `f:>x` gives the exact gradient of a Klong function; with NumPy it is computed numerically. You can call Python from Klong and Klong from Python.

Source code on GitHub: [briangu/klongpy](https://github.com/briangu/klongpy)

New here? Read [Start Here](getting-started.md). If you have code for an earlier release, read [Migrating to v2](migrating-to-v2.md).

## Install

```bash
pip install klongpy              # the interpreter and the kgpy command (NumPy backend)
pip install "klongpy[torch]"     # the torch backend: GPU arrays and exact gradients
pip install "klongpy[db]"        # tables and SQL on arrays (pandas, DuckDB)
pip install "klongpy[web]"       # the .web HTTP server (aiohttp)
pip install "klongpy[ws]"        # WebSockets
pip install "klongpy[all]"       # all of the above
```

The REPL needs no extra: `kgpy` starts it, and `kgpy --backend torch` selects the torch backend. [Start Here](getting-started.md#install) says what each extra is for.

## A first session

An expression is read from right to left. `!5` is the list 0 to 4, `+/` sums a list, and `#` is the length:

```kgpy
?> +/1+!5
15
?> avg::{(+/x)%#x}
:monad
?> avg(!1000000)
499999.5
```

You can bring in Python functions. This imports `fsum` from the standard `math` module and uses it in a function:

```kgpy
?> .pyf("math";"fsum")
1
?> favg::{fsum(x)%#x}
:monad
?> favg(!1000)
499.5
```

From Python, a Klong function is a Python callable:

```python
import numpy as np
from klongpy import KlongInterpreter

klong = KlongInterpreter()
klong('avg::{(+/x)%#x}')
print(klong['avg'](np.arange(10.0)))   # 4.5
```

See [Python Integration](python_integration.md). `]T expression` at the REPL times an expression, and [Performance](performance.md) has measured timings for the backends.

## What is new in v2

[What's New in v2](whats-new-v2.md) has the full list with examples.

- **A friendlier language**: [destructuring, named parameters, dictionary literals, projections and arity from the highest argument](language-additions.md)
- **Readable [errors](errors.md)** with a position, a kind and the source line
- **[Strict mode](strict.md)** in three levels and **[a linter](lint.md)**: `kgpy --lint file.kg`
- **[A better REPL](repl.md)**: multi-line entry, completion, history and `it`
- **[kdb+ compatible IPC](ipc_capabilities.md)**: the default wire protocol is kdb+, with tables, timestamps and credentials
- **[The torch backend](torch_backend.md)** and exact gradients
- **[A conformance suite](conformance.md)** of 1270 cases that does not depend on one implementation
- **[Compiling a static subset of Klong to C](embedded.md)** for small targets
- **[A native Rust runtime](native/index.md)** that runs the full language without Python, including on an [ESP32-P4](native/esp32-p4.md)
- **Documentation that is checked by a tool**: the examples in these pages are run by `scripts/check_docs.py`

Planned and not available: hardware kernels on FPGA and a WebAssembly build. See the [roadmap](roadmap.md).

## Data and systems

- **[Fast Columnar Database](fast_columnar_database.md)**: tables and SQL on arrays with DuckDB
- **[Table and Key-Value Stores](table_and_key_value_stores.md)**: persistent storage for tables and data
- **[Inter-Process Communication](ipc_capabilities.md)**: servers, clients and callbacks
- **[Web Server](web_server.md)**, **[WebSockets](websockets.md)** and **[Timers](timer.md)**

## Documentation

- **[Start Here](getting-started.md)**, **[Quick Start](quick-start.md)** and **[Tutorial](tutorial.md)**: the language from the first expression to functions, adverbs, dictionaries and files
- **[Klong for NumPy Users](from-numpy.md)** and **[Klong for q Users](from-kdb.md)**: translation tables and the differences that cause mistakes
- **[Operators](operators.md)**, **[System Functions](system-functions.md)**, **[Errors](errors.md)** and **[REPL](repl.md)**
- **[Differences from Klong](differences.md)**, **[Release Notes](release-notes.md)** and **[Contributing](contributing.md)**

## Differences from Klong

KlongPy extends Klong. The list below is short; the cases where its behaviour differs, and the known bugs, are on [Differences from Klong](differences.md).

* **Number precision**: Reals are doubles, where Klong's are arbitrary precision. Integers are 64 bits wide and wrap around on overflow, as in NumPy; an operand beyond 64 bits (a long literal, or `2^64`) makes `+ - * % < > | &` exact, but `^` is computed in floating point.
* **Friendlier syntax**: Destructuring (`[a b]::v`), named parameters (`f(a;b;c;d)::{...}`), evaluated dictionaries (`:{;k;v}`), `d@k` on dictionaries, `.fmt`, `.grad`. Programs for classic Klong run unchanged apart from the cases in [Migrating to v2](migrating-to-v2.md). See [Friendlier Syntax](language-additions.md).
* **Function arity**: A function has as many arguments as the highest of `x`, `y` and `z` its body uses, so `{-x}` is a monad. A call with fewer arguments is a projection.
* **Strict mode and lint**: Three [strict levels](strict.md) and [`kgpy --lint`](lint.md) turn silent mistakes into errors or reports. The default is the classic behaviour.
* **Errors**: Errors carry a position and are [classed by kind](errors.md).
* **Python integration**: `.py` and `.pyf` import Python modules and functions into the current Klong context.
* **IPC**: `.srv` and `.cli` connect KlongPy processes, and q processes, over the kdb+ IPC protocol.
* **Autograd**: Gradients with the `:>` operator, exact with the torch backend.

The torch backend does not support object dtype arrays, so strings and mixed lists stay NumPy arrays on the host. See [PyTorch Backend](torch_backend.md).

## Lineage and inspiration

- **[APL](https://en.wikipedia.org/wiki/APL_(programming_language))** (1966): Ken Iverson's notation
- **[J](https://www.jsoftware.com/)**: an ASCII-friendly APL successor
- **[K/Q/kdb+](https://code.kx.com/)**: time series and trading systems
- **[Klong](https://t3x.org/klong)**: Nils M Holm's array language, and the [Klong Book](https://t3x.org/klong/book.html)
- **[NumPy](https://numpy.org/)** and **[PyTorch](https://pytorch.org/)**: the array backends

## Related projects

- [Klupyter](https://github.com/briangu/klupyter): KlongPy in Jupyter Notebooks
- [VS Code Syntax Highlighting](https://github.com/briangu/klongpy-vscode)
- [Advent of Code Solutions](https://github.com/briangu/aoc)

## Development

```bash
git clone https://github.com/briangu/klongpy.git
cd klongpy
pip install -e ".[dev]"   # Install in editable mode with dev dependencies
python3 -m pytest tests/  # Run tests
```

See [Contributing](contributing.md).

## Acknowledgements

Thanks to [Nils M Holm](https://t3x.org) for creating Klong and writing the Klong Book, which made this project possible.
