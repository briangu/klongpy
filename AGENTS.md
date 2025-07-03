# Repo guidelines

To run the test suite, simply run:

```
python3 -m unittest
```

# Klong Language Overview

Klong is a concise array programming language descended from the APL family.
Expressions are evaluated from right to left and most built-in operators are
single characters (e.g. `+`, `-`, `*`, `/`). Functions take at most three
parameters named `x`, `y`, and `z` and are classified as nilad (no args),
monad (one arg), dyad (two args), or triad (three args).
This design keeps syntax minimal while encouraging vectorized operations over
entire arrays.

# KlongPy Extensions

KlongPy implements Klong on top of Python. It uses NumPy for CPU execution and
optionally CuPy for GPU acceleration. The interpreter exposes a dictionary-like
context allowing easy exchange of data and functions between KlongPy and Python.
Python modules and functions can be imported directly via the `.py` and `.pyf`
commands.

Additional built-in modules extend Klong with practical tools:
- **Fast columnar database** via DuckDB integration for zero-copy SQL on NumPy arrays.
- **Inter-process communication** with `.cli` and `.clid` for remote procedure
  calls and distributed dictionaries.
- **Table/key-value store** for persistent data storage.
- **Timers and a web server** to build event-driven or web applications entirely
  in Klong.

These features combine Klong's terse syntax with Python's ecosystem, enabling
high-performance array programming for data analysis and distributed systems.
