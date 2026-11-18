
![Unit Tests](https://github.com/briangu/klongpy/workflows/Unit%20Tests/badge.svg)
[![Last Commit](https://img.shields.io/github/last-commit/briangu/klongpy)](https://img.shields.io/github/last-commit/briangu/klongpy)
[![Dependency Status](https://img.shields.io/librariesio/github/briangu/klongpy)](https://libraries.io/github/briangu/klongpy)
[![Open Issues](https://img.shields.io/github/issues-raw/briangu/klongpy)](https://github.com/briangu/klongpy/issues)
[![Repo Size](https://img.shields.io/github/repo-size/briangu/klongpy)](https://img.shields.io/github/repo-size/briangu/klongpy)
[![GitHub star chart](https://img.shields.io/github/stars/briangu/klongpy?style=social)](https://star-history.com/#briangu/klongpy)

[![Release Notes](https://img.shields.io/github/release/briangu/klongpy)](https://github.com/briangu/klongpy/releases)
[![Downloads](https://static.pepy.tech/badge/klongpy/month)](https://pepy.tech/project/klongpy)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# KlongPy 2: one array language from CPU and GPU to FPGA and microcontroller

KlongPy is an implementation of [Klong](https://t3x.org/klong), an [array language](https://en.wikipedia.org/wiki/Array_programming), in Python. You write `+/a` for the sum of an array instead of a loop. It is for array and time-series work, for people coming from NumPy or kdb+/q, for gradient-based work, and for small devices.

Full documentation: [https://www.klongpy.org/](https://www.klongpy.org/). New here? Read [Start Here](docs/getting-started.md).

## Where it runs

| Target | Status | More |
|---|---|---|
| CPU (NumPy backend) | stable | [Start Here](docs/getting-started.md) |
| GPU (PyTorch backend, autograd) | stable on CPU tensors; the GPU devices (CUDA, Apple MPS) use the same code and are not part of the automated tests | [PyTorch Backend and Autograd](docs/torch_backend.md) |
| Native interpreter (desktop) | passes the conformance suite | [Native Runtime](docs/native/index.md) |
| Compile to C (static profile) | works for a subset of the language; three examples run on Raspberry Pi Picos | [Compiling Klong to C](docs/embedded.md) |
| Microcontrollers (native interpreter) | experimental; run on an ESP32-P4 and on two Raspberry Pi Picos | [Native Runtime status](docs/native/index.md#status-what-has-and-has-not-been-proven) |
| FPGA kernels | experimental; run on one Alveo U250 | [Klong functions as FPGA hardware](docs/hardware.md) |

KlongPy and the native Rust runtime run a [conformance suite](docs/conformance.md) of about 1,300 cases, each a program and its expected output. The C and FPGA paths compile a subset of the language, and that subset is checked against the interpreter.

## Install

```bash
pip install klongpy              # the interpreter and the kgpy command (NumPy backend)
pip install "klongpy[torch]"     # the torch backend: GPU arrays and gradients
pip install "klongpy[db]"        # tables and SQL on arrays (pandas, DuckDB)
pip install "klongpy[web]"       # the .web HTTP server (aiohttp)
pip install "klongpy[ws]"        # WebSockets
pip install "klongpy[all]"       # all of the above
```

The REPL needs no extra: `kgpy` starts it. `kgpy --backend torch` selects the torch backend.

## A first program

An expression is read from right to left. `!5` is the list 0 to 4, and `+/` sums a list. `avg(v)::{...}` defines a function with a named parameter. Type `kgpy` and then:

```kgpy
?> +/1+!5
15
?> avg(v)::{(+/v)%#v}
:monad
?> avg(1+!5)
3.0
```

A gradient is an operator: `f:>x` is the gradient of `f` at `x`. It is autograd on the torch backend and numeric (finite differences) on NumPy, which is what this run uses. Gradient descent on `x^2` from 5, 100 steps of size 0.1:

```kgpy
?> f::{x^2}
:monad
?> f:>3
6.000000000838668
?> s::5.0
5.0
?> {s::s-(0.1*f:>s)}'!100;s
1.0185179880729389e-09
```

The other spellings of the gradient and the Unicode operators are in [PyTorch Backend and Autograd](docs/torch_backend.md).

## What is new in 2

Version 2 adds the native runtime, the conformance suite, compiling to C, experimental FPGA kernels, strict modes, a linter, errors with positions, language additions (destructuring, named parameters, dictionary literals) and kdb+-protocol IPC. [What's New in v2](docs/whats-new-v2.md) is the tour, and [Migrating to v2](docs/migrating-to-v2.md) lists what changes for existing code.

## Staying on 0.7

Version 2 changes how some existing code behaves. To stay on the 0.7 line:

```bash
pip install "klongpy<2"
```

The documentation of each release line stays online: pick the version in the header of [klongpy.org](https://www.klongpy.org/). The README shown on a release's PyPI page, and at its tag in this repository, is the README of that release.

## Security model: trusted networks

KlongPy's network features assume every peer is trusted. Use them on a trusted network only.

- **IPC (`.srv`, `.cli`)**: a connected peer can run arbitrary Klong, and through it arbitrary Python (`.py`, `.pyf`) and shell commands (`.sys`). It can also call any function you have defined and assign any variable.
- **IPC wire format**: the kdb+ IPC protocol, and nothing from the network is unpickled. Message size is limited (`.system` key `ipc_max_message`, 256 MiB by default). The pickle protocol of 0.7 was removed in 2.0; a 0.7 peer cannot connect.
- **No encryption, optional login**: IPC credentials (`"host:port:user:password"`) and all data are sent in clear text and there is no TLS. A server accepts every client unless the program defines `.srv.auth`. `.web` has no login or access control.
- **Bind address**: `.srv(8888)` and `.web(8888;...)` listen on all interfaces (`0.0.0.0`). Anyone who can reach the port has the access above. `.web` routes run the Klong functions you register with the request parameters.
- **KVS and tables**: keys are used as file names under the store root without checking that they stay inside it, and values are pickled on disk. Treat the root directory as trusted storage and do not load stores you did not write.
- **`.db`**: DuckDB runs the string you pass as SQL, as given.

To limit exposure:

- Bind to localhost: `.srv("127.0.0.1:8888")`, `.web("127.0.0.1:8080";get;post)`. `kgpy -s ARG` evaluates `.srv(ARG)`, so to bind to localhost pass a quoted string: `kgpy -s '"127.0.0.1:8888"'`.
- Firewall the ports, and never expose a KlongPy port to an untrusted network or the internet.
- Between hosts, use an SSH tunnel or VPN.

See [IPC Capabilities](docs/ipc_capabilities.md) for the protocol and the credentials hook.

## Lineage

- **[APL](https://en.wikipedia.org/wiki/APL_(programming_language))** (1966): Ken Iverson's notation.
- **[J](https://www.jsoftware.com/)**: an ASCII-friendly successor of APL.
- **[K/Q/kdb+](https://code.kx.com/)**: time-series and trading systems.
- **[Klong](https://t3x.org/klong)**: Nils M Holm's array language, described in the [Klong Book](https://t3x.org/klong/book.html). Thanks to him for creating it.
- **[NumPy](https://numpy.org/)** and **[PyTorch](https://pytorch.org/)**: the array backends.

Related projects: [Klupyter](https://github.com/briangu/klupyter) (KlongPy in Jupyter), [VS Code syntax highlighting](https://github.com/briangu/klongpy-vscode) and [Advent of Code solutions](https://github.com/briangu/aoc).

## Contributing

This project does not accept direct issues. Start with a GitHub Discussion; maintainers promote validated discussions to issues. See [CONTRIBUTING.md](CONTRIBUTING.md), and [docs/contributing.md](docs/contributing.md) to build and test.

## Licence

MIT. See [LICENSE](LICENSE).
