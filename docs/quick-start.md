# Quick Start

Get KlongPy installed and run your first expression in a few steps. [Start Here](getting-started.md) is the longer guide for people new to array languages.

## Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.9 to 3.14
- `pip` for installing packages

## Setup Steps

It's a good idea to work inside a virtual environment so that the
dependencies for KlongPy don't interfere with other Python projects:

```bash
python3 -m venv venv
source venv/bin/activate
```

With the environment active, choose one of the install options below. The REPL has line editing, history and tab completion built in, so `rlwrap` is not needed, and it needs no extra package:

```bash
# Interpreter and the kgpy command (NumPy backend)
pip install klongpy

# Everything: torch, web, db, websockets
pip install "klongpy[all]"
```

The extras are `torch`, `db` (pandas and DuckDB), `web` (aiohttp) and `ws` (websockets). `klongpy[repl]` still works and installs nothing new. [Start Here](getting-started.md#install) says what each extra is for.

### PyTorch Backend (Optional)

For GPU arrays and automatic differentiation, install PyTorch:

```bash
pip install "klongpy[torch]"
```

Then enable the torch backend when running:

```bash
kgpy --backend torch
```

## Your First KlongPy Session

After installing, run `kgpy` to start the REPL. The banner shows the version and the backend, and `]h` lists the help topics. Type an expression after the `?> ` prompt and the REPL prints its value:

```kgpy
?> 1+1
2
?> +/!10
45
```

Quit with Ctrl-D or `]q`. To run a single expression without the REPL, use `kgpy -e '1+1'`, and to run a file, `kgpy file.kg`.

See the [REPL Reference](repl.md) for more information on commands and operations.

## Next Steps

Now that you've installed KlongPy and run a simple session, you're ready to dive deeper. Check out the following resources:

- [Start Here](getting-started.md) - install, a first session, a first script, and where to go next.
- [Tutorial](tutorial.md) - the language from the first expression to functions, adverbs, dictionaries and files.
- [Examples](examples.md) - hands-on code examples.
- [Python Integration](python_integration.md) - interop with Python modules and data.
- [PyTorch Backend & Autograd](torch_backend.md) - gradients, Jacobians, and compilation.
- [Operators](operators.md) - language operator reference.
- [Performance](performance.md) - benchmarking and backend tips.
- [What's New in v2](whats-new-v2.md) and [Migrating to v2](migrating-to-v2.md) - if you used an earlier release.

For any issues or questions, visit our [Issues](https://github.com/briangu/klongpy/issues) page on GitHub.
