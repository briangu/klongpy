# Python Integration Examples

These examples mix Python and KlongPy. Each ``.kg`` file loads the Python module next to it with a relative path, so run it from the directory that holds it.
From this directory:

```bash
$ kgpy hello.kg
hello, world
```

## Basic examples

* **hello.kg / hello.py** – loads a simple Python function and calls it.
* **hello_exports.kg / hello_exports.py** – exposes a Python function via
  ``klongpy_exports`` so that it can be called from Klong.

## multiprocessing

Demonstrates using Python's ``multiprocessing`` module.

* **pool.kg** – call a function that uses a ``Pool`` under the hood.
* **pool_async.kg** – asynchronous variant using ``.async`` in Klong.
* **callback.kg** – marshal a Klong callback to a worker process.
* **worker/** – load another Klong program inside a worker process.

Run from the `multiprocessing` directory:

```bash
$ cd multiprocessing
$ kgpy pool.kg
```

## threading

Examples that rely on ``multiprocessing.pool.ThreadPool``.

* **pool.kg** – parallel work in threads.
* **pool_async.kg** – asynchronous thread example.
* **callback.kg** – use a Klong callback on a thread.
* **callback_async.kg** – asynchronous callback version.

Run from the `threading` directory:

```bash
$ cd threading
$ kgpy pool.kg
```

## Calling from Python

Create a ``KlongInterpreter`` and load a Klong script. Run this from this directory, so that `hello.kg` finds `hello.py`:

```python
from klongpy import KlongInterpreter

klong = KlongInterpreter()
klong('.l("hello.kg")')
```

A ``.kg`` file pulls in its companion Python module using ``.py``:

```kg
.py("hello.py")
hello("world")
```
