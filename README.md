
![Unit Tests](https://github.com/briangu/klongpy/workflows/Unit%20Tests/badge.svg)
[![Last Commit](https://img.shields.io/github/last-commit/briangu/klongpy)](https://img.shields.io/github/last-commit/briangu/klongpy)
[![Dependency Status](https://img.shields.io/librariesio/github/briangu/klongpy)](https://libraries.io/github/briangu/klongpy)
[![Open Issues](https://img.shields.io/github/issues-raw/briangu/klongpy)](https://github.com/briangu/klongpy/issues)
[![Repo Size](https://img.shields.io/github/repo-size/briangu/klongpy)](https://img.shields.io/github/repo-size/briangu/klongpy)
[![GitHub star chart](https://img.shields.io/github/stars/briangu/klongpy?style=social)](https://star-history.com/#briangu/klongpy)

[![Release Notes](https://img.shields.io/github/release/briangu/klongpy)](https://github.com/briangu/klongpy/releases)
[![Downloads](https://static.pepy.tech/badge/klongpy/month)](https://pepy.tech/project/klongpy)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# KlongPy: the Klong array language in Python

KlongPy is a Python implementation of [Klong](https://t3x.org/klong), an [array language](https://en.wikipedia.org/wiki/Array_programming). You write `+/a` for the sum of an array instead of a loop, and the work runs on [NumPy](https://numpy.org/) or, optionally, [PyTorch](https://pytorch.org/) (CPU, CUDA and Apple MPS). With the torch backend `f:>x` gives the exact gradient of a Klong function; with NumPy it is computed numerically. You can call Python from Klong and Klong from Python.

Full documentation: [https://briangu.github.io/klongpy](https://briangu.github.io/klongpy). New here? Read [Start Here](docs/getting-started.md).

## Install

```bash
pip install klongpy              # the interpreter and the kgpy command (NumPy backend)
pip install "klongpy[torch]"     # the torch backend: GPU arrays and exact gradients
pip install "klongpy[db]"        # tables and SQL on arrays (pandas, DuckDB)
pip install "klongpy[web]"       # the .web HTTP server (aiohttp)
pip install "klongpy[ws]"        # WebSockets
pip install "klongpy[all]"       # all of the above
```

The REPL needs no extra: `kgpy` starts it. `kgpy --backend torch` selects the torch backend.

### Staying on version 1

Version 2 changes how some existing code behaves; [Migrating to v2](docs/migrating-to-v2.md) lists every change.
To keep the previous release line:

```bash
pip install "klongpy<2"
```

The documentation for each release line stays online: pick the version in the header of
[klongpy.org](https://www.klongpy.org/). The README of a release is the one shown on its PyPI page
and at its tag in this repository.

## A first session

An expression is read from right to left. `!5` is the list 0 to 4, and `+/` sums a list:

```kgpy
?> +/1+!5
15
?> avg::{(+/x)%#x}
:monad
?> avg(1+!5)
3.0
```

A gradient is an operator. This is the torch backend, which gives it exactly (the NumPy backend gives `6.000000000838668`):

<!-- check-docs: skip: shows the torch backend output (exact gradients), and torch is not installed where the docs are checked -->
```kgpy
?> f::{x^2}
:monad
?> f:>3
6.0
```

A step of gradient descent on `f` is `s::s-(0.1*f:>s)`, so minimising `x^2` from 5 is one line:

```klong
f::{x^2}; s::5.0
{s::s-(0.1*f:>s)}'!100   :" s is now close to 0"
```

## What is new in v2

[What's New in v2](docs/whats-new-v2.md) has the full list with examples. If you have code for an earlier release, read [Migrating to v2](docs/migrating-to-v2.md) first.

- **A friendlier language**: [destructuring, named parameters, dictionary literals, projections and arity from the highest argument](docs/language-additions.md)
- **Readable [errors](docs/errors.md)** with a position, a kind and the source line
- **[Strict mode](docs/strict.md)** in three levels and **[a linter](docs/lint.md)**: `kgpy --lint file.kg`
- **[A better REPL](docs/repl.md)**: multi-line entry, completion, history and `it`
- **[kdb+ compatible IPC](docs/ipc_capabilities.md)**: the default wire protocol is kdb+, with tables, timestamps and credentials
- **[The torch backend](docs/torch_backend.md)** and exact gradients
- **[A conformance suite](docs/conformance.md)** of 1277 cases that does not depend on one implementation
- **[Compiling a static subset of Klong to C](docs/embedded.md)** for small targets
- **[A native Rust runtime](docs/native/index.md)** that runs the full language without Python, including on an [ESP32-P4](docs/native/esp32-p4.md)
- **Documentation that is checked by a tool**: the examples in the documentation are run by `scripts/check_docs.py`

Planned and not available: hardware kernels on FPGA and a WebAssembly build. See the [roadmap](docs/roadmap.md).

## More examples

A table and a SQL query (needs `klongpy[db]`):

<!-- check-docs: requires duckdb pandas -->
```kgpy
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

Two processes. A server defines `avg` and serves it, and a client calls it over the kdb+ IPC protocol:

```kgpy
server?> avg::{(+/x)%#x}
:monad
server?> .srv(8888)
1
client?> f::.cli(8888)
remote[localhost:8888]:fn
client?> f("avg(!10)")
4.5
```

## Where to go next

- [Start Here](docs/getting-started.md), the [Quick Start](docs/quick-start.md) and the [Tutorial](docs/tutorial.md)
- [Klong for NumPy Users](docs/from-numpy.md) and [Klong for q Users](docs/from-kdb.md)
- [Python Integration](docs/python_integration.md) and [PyTorch Backend and Autograd](docs/torch_backend.md)
- [Fast Columnar Database](docs/fast_columnar_database.md), [Table and Key-Value Stores](docs/table_and_key_value_stores.md), [Web Server](docs/web_server.md), [WebSockets](docs/websockets.md) and [Timers](docs/timer.md)
- [Operators](docs/operators.md), [System Functions](docs/system-functions.md), [Performance](docs/performance.md) and [Differences from Klong](docs/differences.md)

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

## Lineage and Inspiration

KlongPy stands on the shoulders of giants:

- **[APL](https://en.wikipedia.org/wiki/APL_(programming_language))** (1966): Ken Iverson's revolutionary notation
- **[J](https://www.jsoftware.com/)**: ASCII-friendly APL successor
- **[K/Q/kdb+](https://code.kx.com/)**: High-performance time series and trading systems
- **[Klong](https://t3x.org/klong)**: Nils M Holm's elegant, accessible array language
- **[NumPy](https://numpy.org/)**: The "Iverson Ghost" in Python's scientific stack
- **[PyTorch](https://pytorch.org/)**: Automatic differentiation and GPU acceleration

KlongPy combines Klong's simplicity with Python's ecosystem and PyTorch's autograd creating something new: an array language where gradients are first-class citizens.

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
