# Quick Start

Welcome to KlongPy! Get up and running with KlongPy in just a few steps.

## Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.9 or higher (3.11 or 3.12 recommended)
- `pip` for installing packages

## Setup Steps

It's a good idea to work inside a virtual environment so that the
dependencies for KlongPy don't interfere with other Python projects:

```bash
python3 -m venv venv
source venv/bin/activate
```

With the environment active, choose one of the install options below
(and optionally `rlwrap` for a nicer REPL experience):

```bash
# Base runtime only (NumPy backend)
pip install klongpy

# REPL support (required for `kgpy`)
pip install "klongpy[repl]"

# Full feature set (REPL, torch, web, db, websockets)
pip install "klongpy[all]"
```

### PyTorch Backend (Optional)

For GPU acceleration and automatic differentiation, install PyTorch:

```bash
pip install "klongpy[torch]"
```

For nicer line editing, install `rlwrap` via your OS package manager (optional).

Then enable the torch backend when running:

```bash
kgpy --backend torch
```


## Setting Up Your First KlongPy Session

After installing KlongPy, you can start using it right away. Here’s how to set up a basic session:


```bash
$> rlwrap kgpy
Welcome to KlongPy REPL v0.7.0
author: Brian Guarraci
web   : http://klongpy.org
]h for help
Ctrl-D or ]q to quit

?> 1+1
2
?>
```

Version numbers may differ depending on the release you installed.

See the [REPL Reference](repl.md) for more information on commands and operations.

## Next Steps

Now that you've installed KlongPy and run a simple session, you're ready to dive deeper. Check out the following resources:

- [Examples](examples.md) - hands-on code examples.
- [Python Integration](python_integration.md) - interop with Python modules and data.
- [PyTorch Backend & Autograd](torch_backend.md) - gradients, Jacobians, and compilation.
- [Operators](operators.md) - language operator reference.
- [Performance](performance.md) - benchmarking and backend tips.

For any issues or questions, visit our [Issues](https://github.com/briangu/klongpy/issues) page on GitHub.

Thank you for choosing KlongPy, and happy coding!
