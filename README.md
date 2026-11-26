
![Unit Tests](https://github.com/briangu/klongpy/workflows/Unit%20Tests/badge.svg)
[![Last Commit](https://img.shields.io/github/last-commit/briangu/klongpy)](https://img.shields.io/github/last-commit/briangu/klongpy)
[![Dependency Status](https://img.shields.io/librariesio/github/briangu/klongpy)](https://libraries.io/github/briangu/klongpy)
[![Open Issues](https://img.shields.io/github/issues-raw/briangu/klongpy)](https://github.com/briangu/klongpy/issues)
[![Repo Size](https://img.shields.io/github/repo-size/briangu/klongpy)](https://img.shields.io/github/repo-size/briangu/klongpy)
[![GitHub star chart](https://img.shields.io/github/stars/briangu/klongpy?style=social)](https://star-history.com/#briangu/klongpy)

[![Release Notes](https://img.shields.io/github/release/briangu/klongpy)](https://github.com/briangu/klongpy/releases)
[![Downloads](https://static.pepy.tech/badge/klongpy/month)](https://pepy.tech/project/klongpy)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# KlongPy 2: one array language from GPU to FPGA to microcontroller

KlongPy is an implementation of [Klong](https://t3x.org/klong), an [array language](https://en.wikipedia.org/wiki/Array_programming). You write `+/a` for the sum of an array instead of a loop. It is for array and time-series work, for people coming from NumPy or kdb+/q, for gradient-based work, and for small devices.

KlongPy 2 runs Klong on one interpreter, written in Rust. Two commands start it:

| Command | What it is |
|---|---|
| `kg` | A standalone native program with column tables, embedded DuckDB and optional Metal compute on macOS. |
| `kgpy` | The same engine hosted in Python, with NumPy and PyTorch arrays, pandas and Arrow interchange, Python callables, web services and FPGA card bindings. |

The same interpreter is in firmware for microcontroller boards, and the tools `lint`, `check`, `compile c`, `fpga source` and `ast` run from either command. [What Runs Where](docs/what-runs-where.md) says which part does what.

Full documentation: [https://www.klongpy.org/](https://www.klongpy.org/). New here? Read [Start Here](docs/getting-started.md).

The native ticker plant takes ordered trades over kdb+ IPC, serves live quotes and hands each completed trading day to a separate historical database. On two AMD EPYC hosts connected by 100 GbE, an experimental in-memory RDB sustained **3.01M trades/s with a synchronous journal** and **5.24M/s in RAM only**, each across two one-minute runs with two subscribers and both query clients active. Including rollover, the journaled runs averaged **1.99–2.05M/s**. [Throughput, query latency and missed reads](docs/performance.md#finding-the-sustained-rate) · [Run the sweep](examples/stocks/tickerplant/README.md#reproduce-the-in-memory-sweep)

## Where it runs

| Target | Status | More |
|---|---|---|
| CPU (`kg`, and `kgpy` with NumPy) | stable; passes the conformance suite | [Start Here](docs/getting-started.md) |
| GPU (Python backend and autograd) | The opt-in GPU acceptance tests have run on Apple MPS. The standalone Metal backend also passes the native MPS cases. | [GPU acceptance tests](tests/v2/README.md) · [PyTorch Backend and Autograd](docs/torch_backend.md) |
| Compile to C (static profile) | works for a subset of the language; compiled C has run with desktop compilers and, for three examples, on two Raspberry Pi Picos | [Compiling Klong to C](docs/embedded.md) |
| Microcontrollers (the interpreter in firmware) | experimental; two RP2040 Picos passed the RC1 paired IPC self-test. Earlier firmware ran console tests on an ESP32-P4 | [Native Runtime status](docs/native/index.md#status-what-has-and-has-not-been-proven) |
| FPGA kernels | experimental; run on one Alveo U250 | [Klong functions as FPGA hardware](docs/hardware.md) |

For 2.0.0rc1, the [conformance suite](docs/conformance.md) passed 2,220 cases under standalone `kg` and 2,199 applicable cases under Python-hosted `kgpy`. The serial console simulator passed 1,716 selected cases. Physical boards have [separate test results](docs/native/index.md#status-what-has-and-has-not-been-proven). The C and FPGA compilers support the static subset of Klong.

## Install

For standalone `kg`, run this from the repository root with Rust installed and Cargo's binary directory on your `PATH`.

```sh
cargo install --locked --path native/klong-cli
kg -e '+/!100'                   # 4950
```

Type `kg` for the REPL or `kg trades.kg` to run a script. Tools and coding agents can read `kg --capabilities` as JSON. Firmware builders can select a board and its features separately. [Native build guide](docs/native/building.md)

For the Python distribution, choose the packages you use. These commands install KlongPy 2.0.0rc1. While 2.0 is in prerelease, an unpinned `pip install klongpy` installs the stable 0.7 series instead.

```bash
python -m pip install "klongpy==2.0.0rc1"              # the kgpy and kg commands (NumPy arrays)
python -m pip install "klongpy[torch]==2.0.0rc1"     # the torch backend: GPU arrays and gradients
python -m pip install "klongpy[db]==2.0.0rc1"        # pandas, Python DuckDB, legacy key-value store
python -m pip install "klongpy[web]==2.0.0rc1"       # the .web HTTP server (aiohttp)
python -m pip install "klongpy[ws]==2.0.0rc1"        # WebSockets
python -m pip install "klongpy[all]==2.0.0rc1"       # all of the above
```

KlongPy 2 needs Python 3.11 to 3.14. The REPL needs no extra: `kgpy` or `kg` starts it. `kgpy --backend torch` selects the torch backend.

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

Version 2 brings one Rust engine to both commands, built-in tables and DuckDB, kdb+-protocol IPC, and conformance tests across targets. It adds modules, destructuring, named parameters, strict modes, diagnostic tools, compilation to C and experimental FPGA kernels. [What's New in v2](docs/whats-new-v2.md) is the tour, and [Migrating to v2](docs/migrating-to-v2.md) lists what changes for existing code.

## Staying on 0.7

Version 2 changes how some existing code behaves, and there is no pure-Python version of it. To stay on the pure-Python 0.7 line:

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
