# KlongPy 2: one array language from GPU to FPGA to microcontroller

KlongPy is an implementation of [Klong](https://t3x.org/klong), an [array language](https://en.wikipedia.org/wiki/Array_programming), in Python. You write `+/a` for the sum of an array instead of a loop. It is for array and time-series work, for people coming from NumPy or kdb+/q, for gradient-based work, and for small devices. Source code: [briangu/klongpy](https://github.com/briangu/klongpy).

## Targets and status

| Target | Status | More |
|---|---|---|
| CPU, with NumPy | stable | [Start Here](getting-started.md) |
| GPU, with PyTorch and autograd | stable | [PyTorch Backend](torch_backend.md) |
| Native interpreter (Rust) on a desktop | passes conformance | [Native Runtime](native/index.md) |
| Compile to C | works for a subset of the language | [Compiling Klong to C](embedded.md) |
| Microcontrollers (native runtime) | experimental, one board | [Native Runtime status](native/index.md#status-what-has-and-has-not-been-proven) |
| FPGA | experimental, one board | [Klong as FPGA Hardware](hardware.md) |

A [conformance suite](conformance.md) of about 1,300 cases is what makes it the same language on every target.

## Install

```bash
pip install klongpy              # the interpreter and the kgpy command (NumPy backend)
pip install "klongpy[torch]"     # the torch backend: GPU arrays and gradients
pip install "klongpy[all]"       # torch, tables, web, WebSockets
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

From Python, a Klong function is a Python callable:

```python
import numpy as np
from klongpy import KlongInterpreter

klong = KlongInterpreter()
klong('avg::{(+/x)%#x}')
print(klong['avg'](np.arange(10.0)))   # 4.5
```

## Where to go

* **New to array languages:** [Start Here](getting-started.md), then the [Tutorial](tutorial.md) and [Examples](examples.md).
* **From NumPy:** [Klong for NumPy Users](from-numpy.md).
* **From kdb+ or q:** [Klong for q Users](from-kdb.md) and [IPC Capabilities](ipc_capabilities.md).
* **Upgrading from 0.7:** [What's New in v2](whats-new-v2.md), [Migrating to v2](migrating-to-v2.md) and [Release Notes](release-notes.md).
* **Embedding in Python:** [Python Integration](python_integration.md) and [PyTorch Backend](torch_backend.md).
* **Small devices and hardware:** [Native Runtime](native/index.md), [Compiling Klong to C](embedded.md) and [Klong as FPGA Hardware](hardware.md).
* **Performance:** [Performance](performance.md).
* **Language reference:** [Operators](operators.md), [System Functions](system-functions.md), [Errors](errors.md) and [Differences from Klong](differences.md).

## Security

KlongPy's network features assume every peer is trusted.

* A connected IPC peer can run arbitrary Klong, and through it Python and shell commands.
* A server accepts every client unless the program defines `.srv.auth`, and nothing is encrypted.
* The default protocol is kdb+ IPC and unpickles nothing. The 0.7 pickle protocol runs code chosen by the sender, is deprecated, and is refused by a server that has `.srv.auth` defined or `KLONGPY_IPC=kdb` set.
* `.srv` and `.web` listen on all interfaces unless you bind to localhost.

Use them on a trusted network only. See [IPC Capabilities](ipc_capabilities.md), and the security model in the [README](https://github.com/briangu/klongpy#security-model-trusted-networks).

## Lineage

KlongPy follows [APL](https://en.wikipedia.org/wiki/APL_(programming_language)), [J](https://www.jsoftware.com/), [K/Q/kdb+](https://code.kx.com/) and [Klong](https://t3x.org/klong), Nils M Holm's array language described in the [Klong Book](https://t3x.org/klong/book.html). Thanks to him for creating it. [NumPy](https://numpy.org/) and [PyTorch](https://pytorch.org/) are the array backends.

Related projects: [Klupyter](https://github.com/briangu/klupyter) (KlongPy in Jupyter), [VS Code syntax highlighting](https://github.com/briangu/klongpy-vscode) and [Advent of Code solutions](https://github.com/briangu/aoc). To build and test KlongPy, see [Contributing](contributing.md).
