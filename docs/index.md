# KlongPy 2: one array language from CPU and GPU to FPGA and microcontroller

KlongPy is an implementation of [Klong](https://t3x.org/klong), an [array language](https://en.wikipedia.org/wiki/Array_programming), in Python. You write `+/a` for the sum of an array instead of a loop. It is for array and time-series work, for people coming from NumPy or kdb+/q, for gradient-based work, and for small devices. Source code: [briangu/klongpy](https://github.com/briangu/klongpy).

## Targets and status

| Target | Status | More |
|---|---|---|
| CPU (NumPy backend) | stable | [Start Here](getting-started.md) |
| GPU (PyTorch backend, autograd) | stable on the CPU, which the automated tests cover; Apple MPS has been run by hand and CUDA not at all, and neither is in the automated tests | [PyTorch Backend and Autograd](torch_backend.md) |
| Native interpreter (desktop) | passes the conformance suite | [Native Runtime](native/index.md) |
| Compile to C (static profile) | works for a subset of the language; three examples run on Raspberry Pi Picos | [Compiling Klong to C](embedded.md) |
| Microcontrollers (native interpreter) | experimental; run on an ESP32-P4 and on two Raspberry Pi Picos | [Native Runtime status](native/index.md#status-what-has-and-has-not-been-proven) |
| FPGA kernels | experimental; run on one Alveo U250 | [Klong functions as FPGA hardware](hardware.md) |

KlongPy and the native Rust runtime run a [conformance suite](conformance.md) of about 1,500 cases. The C and FPGA paths compile a subset of the language, and that subset is checked against the interpreter.

## Install

```bash
pip install klongpy              # the interpreter and the kgpy command (NumPy backend)
pip install "klongpy[torch]"     # the torch backend: GPU arrays and gradients
pip install "klongpy[all]"       # torch, tables, web, WebSockets
```

The REPL needs no extra: `kgpy` starts it, and `kgpy --backend torch` selects the torch backend. [Start Here](getting-started.md#install) says what each extra is for.

## A first session

An expression is read from right to left. `!5` is the list 0 to 4, `+/` sums a list, `#` is the length, and `avg(v)::{...}` defines a function with the named parameter `v`:

```kgpy
?> +/1+!5
15
?> avg(v)::{(+/v)%#v}
:monad
?> avg(!1000000)
499999.5
```

From Python, a Klong function is a Python callable:

```python
import numpy as np
from klongpy import KlongInterpreter

klong = KlongInterpreter()
klong('avg(v)::{(+/v)%#v}')
print(klong['avg'](np.arange(10.0)))   # 4.5
```

## Where to go

* **New to array languages:** [Start Here](getting-started.md), then the [Tutorial](tutorial.md) and [Examples](examples.md).
* **From NumPy:** [Klong for NumPy Users](from-numpy.md).
* **From kdb+ or q:** [Klong for q Users](from-kdb.md) and [IPC Capabilities](ipc_capabilities.md).
* **Upgrading from 0.7:** [What's New in v2](whats-new-v2.md), [Migrating to v2](migrating-to-v2.md) and [Release Notes](release-notes.md).
* **Embedding in Python:** [Python Integration](python_integration.md) and [PyTorch Backend and Autograd](torch_backend.md).
* **Small devices and hardware:** [Native Runtime](native/index.md), [Compiling Klong to C](embedded.md) and [Klong functions as FPGA hardware](hardware.md).
* **Performance:** [Performance](performance.md).
* **Language reference:** [Operators](operators.md), [System Functions](system-functions.md), [Errors](errors.md) and [Differences from Klong](differences.md).

## Security

KlongPy's network features assume every peer is trusted.

* A connected IPC peer can run arbitrary Klong, and through it Python and shell commands.
* A server accepts every client unless the program defines `.srv.auth`, and nothing is encrypted.
* The protocol is kdb+ IPC and nothing is unpickled. The pickle protocol of 0.7 was removed in 2.0.
* `.srv` and `.web` listen on all interfaces unless you bind to localhost.

Use them on a trusted network only. See [IPC Capabilities](ipc_capabilities.md), and the security model in the [README](https://github.com/briangu/klongpy#security-model-trusted-networks).

## Lineage

KlongPy follows [APL](https://en.wikipedia.org/wiki/APL_(programming_language)), [J](https://www.jsoftware.com/), [K/Q/kdb+](https://code.kx.com/) and [Klong](https://t3x.org/klong), Nils M Holm's array language described in the [Klong Book](https://t3x.org/klong/book.html). Thanks to him for creating it. [NumPy](https://numpy.org/) and [PyTorch](https://pytorch.org/) are the array backends.

Related projects: [Klupyter](https://github.com/briangu/klupyter) (KlongPy in Jupyter), [VS Code syntax highlighting](https://github.com/briangu/klongpy-vscode) and [Advent of Code solutions](https://github.com/briangu/aoc). To build and test KlongPy, see [Contributing](contributing.md).
