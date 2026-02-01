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

With the environment active, install KlongPy (and optionally `rlwrap`
for a nicer REPL experience):

```bash
pip install "klongpy[all]"
pip install rlwrap  # optional
```

### PyTorch Backend (Optional)

For GPU acceleration and automatic differentiation, install PyTorch:

```bash
pip install torch
```

Then enable the torch backend when running:

```bash
USE_TORCH=1 kgpy
```


## Setting Up Your First KlongPy Session

After installing KlongPy, you can start using it right away. Here’s how to set up a basic session:


```bash
$> rlwrap kgpy
Welcome to KlongPy REPL v0.6.0
author: Brian Guarraci
web   : http://klongpy.org
]h for help
ctrl-d or ]q to quit

?> 1+1
2
?>
```

See the [REPL Reference](repl.md) for more information on commands and operations.

## Next Steps

Now that you've installed KlongPy and run a simple session, you're ready to dive deeper. Check out the following resources:

- [Examples](examples.md) - for more hands-on code examples.
- [API Reference](api-reference.md) - for detailed documentation on KlongPy functions and classes.
- [Contribute](contribute.md) - for guidelines on how to contribute to KlongPy.

For any issues or questions, visit our [Issues](https://github.com/briangu/klongpy/issues) page on GitHub.

Thank you for choosing KlongPy, and happy coding!
