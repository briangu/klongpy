
![Unit Tests](https://github.com/briangu/klongpy/workflows/Unit%20Tests/badge.svg)

# KlongPy

KlongPy is a Python adaptation of the [Klong](https://t3x.org/klong) [array language](https://en.wikipedia.org/wiki/Array_programming), offering high-performance vectorized operations. It prioritizes compatibility with Python, thus allowing seamless integration of Python's expansive ecosystem while retaining Klong's succinctness. With the inclusion of [CuPy](https://github.com/cupy/cupy), KlongPy can operate using both CPU and GPU backends. It utilizes [NumPy](https://numpy.org/), an [Iverson Ghost](https://analyzethedatanotthedrivel.org/2018/03/31/NumPy-another-iverson-ghost/) descendant from APL, as its core. This means its runtime can target either GPU or CPU backends. The framework's foundation lies in [Nils M Holm](https://t3x.org)'s work, the original developer of Klong, who has authored a [Klong Book](https://t3x.org/klong/book.html). KlongPy is especially useful for data scientists, quantitative analysts, researchers, and programming language aficionados.

# Overview

KlongPy is both an Array Language runtime and a set of powerful tools for building high performance data analysis and distributed computing applications.  Some of the features include:

* [__Array Programming__](https://en.wikipedia.org/wiki/Array_programming): Based on [Klong](https://t3x.org/klong), a concise, expressive, and easy-to-understand array programming language. Its simple syntax and rich feature set make it an excellent tool for data scientists and engineers.
* [__Speed__](docs/performance.md): Designed for high-speed vectorized computing, enabling you to process large data sets quickly and efficiently on either CPU or GPU.
* [__Fast Columnar Database__](docs/fast_columnar_database.md): Includes integration with [DuckDb](http://duckdb.org), a super fast in-process columnar store that can operate directly on NumPy arrays with zero-copy.
* [__Inter-Process Communication (IPC)__](docs/ipc_capabilities.md): Includes built-in support for IPC, enabling easy communication between different processes and systems. Ticker plants and similar pipelines are easy to build.
* [__Table and Key-Value Store__](docs/table_and_key_value_stores.md): Includes a simple file-backed key value store that can be used to store database tables or raw key/value pairs.
* [__Python Integration__](docs/python_integration.md): Seamlessly compatible with Python and modules, allowing you to leverage existing Python libraries and frameworks.
* [__Web server__](docs/web_server.md): Includes a web server, making it easy to build sites backed by KlongPy capabilities.
* [__Timers__](docs/timer.md): Includes periodic timer facility to periodically perform tasks.

# Examples

Consider this simple Klong expression that computes an array's average: `(+/a)%#a`. Decoded, it means "sum of 'a' divided by the length of 'a'", as read from right to left.

Below, we define the function 'avg' and apply it to the array of 1 million integers (as defined by !1000000)

Let's try this in the KlongPy REPL:

```Bash
$ rlwrap kgpy

Welcome to KlongPy REPL v0.3.78
author: Brian Guarraci
repo  : https://github.com/briangu/klongpy
crtl-d or ]q to quit

?> avg::{(+/x)%#x}
:monad
?> avg(!1000000)
499999.5
```

Now let's time it (first, right it once, then 100 times):

```
?> ]T avg(!1000000)
total: 0.0032962500117719173 per: 0.0032962500117719173
?> ]T:100 avg(!1000000)
total: 0.10882879211567342 per: 0.0010882879211567343
```

We can also import Python custom or standard modules to use directly in Klong language.

```
?> .pyf("math";"pi")
1
?> pi
3.141592653589793
```

Here we import the fsum function from standard Python math library and redefine avg to use 'fsum':

```
?> .pyf("math";"fsum")
1
?> favg::{fsum(x)%#x}
:monad
?> favg(!1000000)
499999.5
```

Notice that using fsum is slower than using Klong '+/'.  This is because the '+/' operation is vectorized while fsum is not.

```
?> ]T favg(!1000000)
total: 0.050078875152394176 per: 0.050078875152394176
?> ]T:100 favg(!1000000)
total: 2.93945804098621 per: 0.029394580409862103
```

To use KlongPy within Python, here's a basic outline:

```python
from klongpy import KlongInterpreter

# instantiate the KlongPy interpeter
klong = KlongInterpreter()

# define average function in Klong (Note the '+/' (sum over) uses np.add.reduce under the hood)
klong('avg::{(+/x)%#x}')

# create a billion random uniform values [0,1)
data = np.random.rand(10**9)

# reference the 'avg' function in Klong interpeter and call it directly from Python.
r = klong['avg'](data)

print(f"avg={np.round(r,6)}")
```

And let's run a performance comparison between CPU and GPU backends:

```python
import time
from klongpy.backend import np
from klongpy import KlongInterpreter

klong = KlongInterpreter()
klong('avg::{(+/x)%#x}')

data = np.random.rand(10**9)

start = time.perf_counter_ns()
r = klong['avg'](data)
stop = time.perf_counter_ns()

print(f"avg={np.round(r,6)} in {round((stop - start) / (10**9), 6)} seconds")
```

Run (CPU)

    $ python3 tests/perf_avg.py
    avg=0.5 in 0.16936 seconds

Run (GPU)

    $ USE_GPU=1 python3 tests/perf_avg.py
    avg=0.500015 in 0.027818 seconds

# Installation

### CPU

    $ pip3 install klongpy

### GPU support

    Choose your CuPy prebuilt binary or from source.  Note, the [ROCM](docs/ROCM.md) support for CuPy is experimental and likely will have issues.

    'cupy' => build from source
    'cuda12x' => "cupy-cuda12x"
    'cuda11x' => "cupy-cuda11x"
    'cuda111' => "cupy-cuda111"
    'cuda110' => "cupy-cuda110"
    'cuda102' => "cupy-cuda102"
    'rocm-5-0' => "cupy-rocm-5-0"
    'rocm-4-3' => "cupy-rocm-4-3"

    $ pip3 install klongpy[cupy]

### All application tools (db, web, REPL, etc.)

    $ pip3 install klongpy[full]


# REPL

KlongPy has a REPL similar to Klong's REPL.

```bash
$ pip3 install klongpy[repl]
$ rlwrap kgpy

Welcome to KlongPy REPL
author: Brian Guarraci
repo  : https://github.com/briangu/klongpy
crtl-c to quit

?> 1+1
2
>? "hello, world!"
hello, world!
?> prime::{&/x!:\2+!_x^1%2}
:monad
?> prime(4)
0
?> prime(251)
1
?> ]T prime(251)
total: 0.0004914579913020134 per: 0.0004914579913020134
```

Read about the [prime example here](https://t3x.org/klong/prime.html).


# Status

KlongPy aims to be a complete implementation of klong.  It currently passes all of the integration tests provided by klong as well as additional suites.

Since CuPy is [not 100% compatible with NumPy](https://docs.cupy.dev/en/stable/user_guide/difference.html), there are currently some gaps in KlongPy between the two backends.  Notably, strings are supported in CuPy arrays so KlongPy GPU support currently is limited to math.

Primary ongoing work includes:

* Additional tools to make KlongPy applications more capable.
* Additional syntax error help
* Actively switch between CuPy and NumPy when incompatibilities are present
    * Work on CuPy kernels is in this branch: _cupy_reduce_kernels

# Differences from Klong

KlongPy is effectively a superset of the Klong language, but has some key differences:

    * Infinite precision: The main difference in this implementation of Klong is the lack of infinite precision.  By using NumPy we are restricted to doubles.
    * Python integration: Most notably, the ".py" command allows direct import of Python modules into the current Klong context.
    * IPC - KlongPy supports IPC between KlongPy processes.

# Related

 * [Klupyter - KlongPy in Jupyter Notebooks](https://github.com/briangu/klupyter)
 * [Advent Of Code '22](https://github.com/briangu/aoc/tree/main/22)


# Unused operators

The following operators are yet to be used:

```
:! :& :, :< :> :?
```

# Contribute

### Develop

    $ git clone https://github.com/briangu/klongpy.git
    $ cd klongpy
    $ python3 setup.py develop

### Running tests

```bash
python3 -m unittest
```


# Acknowledgement

HUGE thanks to Nils M Holm for his work on Klong and providing the foundations for this interesting project.

