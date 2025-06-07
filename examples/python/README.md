# Python Integration Examples

This directory contains small examples showing how to mix Python and KlongPy.
Run any of the ``.kg`` files with ``kgpy``:

```bash
$ kgpy hello.kg
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

Run with:

```bash
$ kgpy multiprocessing/pool.kg
```

## threading

Examples that rely on ``multiprocessing.pool.ThreadPool``.

* **pool.kg** – parallel work in threads.
* **pool_async.kg** – asynchronous thread example.
* **callback.kg** – use a Klong callback on a thread.
* **callback_async.kg** – asynchronous callback version.

Run with:

```bash
$ kgpy threading/pool.kg
```

## Calling from Python

Instantiate a ``KlongInterpreter`` and execute a Klong script:

```python
from klongpy import KlongInterpreter

klong = KlongInterpreter()
klong('.l("examples/python/hello.kg")')
```

A ``.kg`` file pulls in its companion Python module using ``.py``:

```kg
.py("hello.py")
hello("world")
```
