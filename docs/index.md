# KlongPy 2: one array language from CPU and GPU to FPGA and microcontroller

KlongPy is an implementation of [Klong](https://t3x.org/klong), an [array language](https://en.wikipedia.org/wiki/Array_programming). You write `+/a` for the sum of an array instead of a loop. It is for array and time-series work, for people coming from NumPy or kdb+/q, for gradient-based work, and for small devices. Source code: [briangu/klongpy](https://github.com/briangu/klongpy).

KlongPy 2 runs Klong on one interpreter, written in Rust. Two commands start it:

| Command | What it is |
|---|---|
| `kg` | The standalone interpreter with column tables, embedded DuckDB and optional Metal compute on macOS. |
| `kgpy` | The same interpreter hosted in Python, with NumPy and PyTorch arrays, pandas and Arrow interchange, Python callables, web services and FPGA card bindings. |

The same interpreter is in firmware for microcontroller boards, and the tools `lint`, `check`, `compile c`, `fpga source` and `ast` run from either command. [What Runs Where](what-runs-where.md) says which part does what.

## Targets and status

The ticker plant with its journal implemented in Klong sustained **2,097,152 offered trades/s** on two AMD EPYC hosts connected by 100 GbE. Each batch synced to local NVMe and reached two subscribers before the publisher received its reply. Both one-minute runs verified 125,829,120 archived trades and met the 50 ms target for at least 99% of scheduled live and historical quotes. [Journal measurements](performance.md#klong-journal-on-100-gbe) · [FPGA analytics on committed trades](performance.md#fpga-analytics-on-committed-trades)

| Target | Status | More |
|---|---|---|
| CPU (`kg`, and `kgpy` with NumPy) | stable; passes the conformance suite | [Start Here](getting-started.md) |
| GPU | Automated acceptance covers native Metal and the Python tensor backend on Apple MPS. The Python backend also has CPU coverage. | [Native Metal](what-runs-where.md#native-metal) · [PyTorch Backend and Autograd](torch_backend.md) |
| Compile to C (static profile) | works for a subset of the language; compiled C has run with desktop compilers and, for three examples, on two Raspberry Pi Picos | [Compiling Klong to C](embedded.md) |
| Microcontrollers (the interpreter in firmware) | experimental; two RP2040 Picos passed the RC1 paired IPC self-test. Earlier firmware ran console tests on an ESP32-P4 | [Native Runtime status](native/index.md#status-what-has-and-has-not-been-proven) |
| FPGA kernels | experimental; run on one Alveo U250 | [Klong functions as FPGA hardware](hardware.md) |

For 2.0.0rc1, the [conformance suite](conformance.md) passed 2,220 cases under standalone `kg` and 2,199 applicable cases under Python-hosted `kgpy`. The serial console simulator passed 1,716 selected cases. Physical boards have [separate test results](native/index.md#status-what-has-and-has-not-been-proven). The C and FPGA compilers support the static subset of Klong.

## Install

```bash
pip install klongpy              # the kgpy and kg commands (NumPy arrays)
pip install "klongpy[torch]"     # the torch backend: GPU arrays and gradients
pip install "klongpy[all]"       # torch, pandas and DuckDB adapters, web, WebSockets
```

KlongPy 2 needs Python 3.11 to 3.14. The REPL needs no extra: `kgpy` or `kg` starts it, and `kgpy --backend torch` selects the torch backend. [Start Here](getting-started.md#install) says what each extra is for.

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

`kg` takes the same expressions. From Python, a Klong function is a Python callable:

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
* **Which command to run:** [What Runs Where](what-runs-where.md).
* **Splitting a program into files:** [Modules](language-additions.md#modules).
* **Upgrading from the pure-Python line:** [What's New in v2](whats-new-v2.md), [Migrating to v2](migrating-to-v2.md) and [Release Notes](release-notes.md).
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
