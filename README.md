
![Unit Tests](https://github.com/briangu/klongpy/workflows/Unit%20Tests/badge.svg)

# KlongPy

KlongPy is a vectorized port of [Klong](https://t3x.org/klong), making it a blazingly fast [array language](https://en.wikipedia.org/wiki/Array_programming) that supports direct Python integration.

[NumPy](https://numpy.org/) is used as the runtime target because it itself is an [Iverson Ghost](https://analyzethedatanotthedrivel.org/2018/03/31/NumPy-another-iverson-ghost/), or rather a descendent of APL, making the mapping from Klong to NumPy *relatively* straightforward.

Using NumPy also means that, via [CuPy](https://github.com/cupy/cupy), both CPU and GPU backends are supported (where possible, see Status section).


Klong was created by Nils M Holm and he has also written a [Klong Book](https://t3x.org/klong/book.html).


# Related

 * [Advent Of Code '22](https://github.com/briangu/aoc/tree/main/22).
 * [Example Ticker Plant with streaming and datagrames](https://github.com/briangu/kdfs)


# Overview

KlongPy brings together the Klong terse array language notation with the performance of NumPy.  I wanted to use Klong but I also wanted it to be a fast as possible.  Bonus is the ability to mix Klong with Python libraries making it easy to pick and choose the tools as appropriate.

Here's simple example of mixing Python and KlongPy to compute average of a 1B entry array.

First, let's look at how average is computed in Klong:

```
{(+/x)%#x}
```

which directly translates to (right to left): length of x divides sum over x.

Now, with that in hand, we can try it in the REPL:

```
    Welcome to KlongPy REPL
    author: Brian Guarraci
    repo  : https://github.com/briangu/klongpy
    crtl-c to quit

    ?> avg::{(+/x)%#x}
    :monad
    ?> avg(100)
    49.5
```
Now let's time it.  First we'll run it once and see it takes about 374us, then we'll run it 100 times.
    ?> ]T avg(!100)
    0.0003741057589650154
    >? ]T:100 avg(!100)
    0.007682837080210447
```

Let's compare CPU vs GPU backends:

```python
import time
from klongpy.backend as np
from klongpy import KlongInterpreter

# instantiate the KlongPy interpeter
klong = KlongInterpreter()

# create a billion random uniform values [0,1)
data = np.random.rand(10**9)

# define average function in Klong
# Note the '+/' (sum over) uses np.add.reduce under the hood
klong('avg::{(+/x)%#x}')

# make generated data available in KlongPy as the 'data' variable
klong['data'] = data

# run Klong average function and return the result back to Python
start = time.perf_counter_ns()
r = klong('avg(data)')
stop = time.perf_counter_ns()
seconds = (stop - start) / (10**9)
print(f"avg={np.round(r,6)} in {round(seconds,6)} seconds")
```

Run (CPU)

    $ python3 tests/perf_avg.py
    avg=0.5 in 0.16936 seconds

Run (GPU)

    $ USE_GPU=1 python3 tests/perf_avg.py
    avg=0.500015 in 0.027818 seconds

# Python integration

KlongPy supports direct Python integration, making it easy to mix Klong with Python and use it in the most suitable scenarios.  For example, KlongPy can be part of an ML/Pandas workflow or be part of the website backend.

Extending KlongPy with custom functions and moving data in / out of the KlongPy interpeter is easy since the interpreter operates as a dictionary. The dictionary contents are the current KlongPy state.

Data generated elsewhere can be set in KlongPy and seamlessly accessed and processed via Klong language.  Also, Python lambdas or functions may be exposed directly as Klong functions, allowing easy extensions to the Klong language.

## Function example

```python
klong = KlongInterpreter()
klong['f'] = lambda x, y, z: x*1000 + y - z
r = klong('f(3; 10; 20)')
assert r == 2990
```

## Data example

```python
data = np.arange(10*9)
klong['data'] = data
r = klong('1+data')
assert r == 1 + data
```

Variables may be directly retrieved from KlongPy context:

```python
klong('Q::1+data')
Q = klong['Q']
print(Q)
```

# Performance

The Klong language is simple, so the overhead is low.  The bulk of the compute time will likely be spent in NumPy doing actual work.

Here's a contrived rough benchmark to show the magnitude differences between Python, KlongPy (CPU + GPU) and Numpy (CPU).

**Spoiler**: GPU-backed KlongPy is about 790x faster than naive Python and 36x faster than NumPy-backed KlongPy.

### Python

```python
def python_vec(number=100):
    r = timeit.timeit(lambda: [2 * (1 + x) for x in range(10000000)], number=number)
    return r/number
```

### KlongPy

```python
# NumPy and CuPy (CuPy is enabled via USE_GPU=1 environment variable
def klong_vec(number=100):
    klong = KlongInterpreter()
    r = timeit.timeit(lambda: klong("2*1+!10000000"), number=number)
    return r/number
```

### NumPy (explicit usage)

```python
def NumPy_vec(number=100):
    r = timeit.timeit(lambda: np.multiply(np.add(np.arange(10000000), 1), 2), number=number)
    return r/number
```

## Results

### CPU (AMD Ryzen 9 7950x)

    $ python3 tests/perf_vector.py
    Python: 0.369111s
    KlongPy USE_GPU=None: 0.017946s
    Numpy: 0.017896s
    Python / KlongPy => 20.568334
    Numpy / KlongPy => 0.997245

### GPU (Same CPU w/ NVIDIA GeForce RTX 3090)

    $ USE_GPU=1 python3 tests/perf_vector.py
    Python: 0.364893s
    KlongPy USE_GPU=1: 0.000461s
    NumPy: 0.017053s
    Python / KlongPy => 790.678069
    Numpy / KlongPy => 36.951443


# Installation

### CPU

    $ pip3 install klongpy

### GPU support

    Choose your CuPy prebuilt binary or from source:

    'cupy' => build from source
    'cuda12x' => "cupy-cuda12x"
    'cuda11x' => "cupy-cuda11x"
    'cuda111' => "cupy-cuda111"
    'cuda110' => "cupy-cuda110"
    'cuda102' => "cupy-cuda102"
    'rocm-5-0' => "cupy-rocm-5-0"
    'rocm-4-3' => "cupy-rocm-4-3"

    $ pip3 install klongpy[cupy]

### Everything

    $ pip3 install klongpy[cupy,repl]

### Develop

    $ git clone https://github.com/briangu/klongpy.git
    $ cd klongpy
    $ python3 setup.py develop


# REPL

KlongPy has a REPL similar to Klong.

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
    0.0005430681630969048

Read about the [prime example here](https://t3x.org/klong/prime.html).


# Status

KlongPy aims to be a complete implementation of klong.  It currently passes all of the integration tests provided by klong.

Since CuPy is [not 100% compatible with NumPy](https://docs.cupy.dev/en/stable/user_guide/difference.html), there are currently some gaps in KlongPy between the two backends.  Notably, strings are supported in CuPy arrays so KlongPy GPU support currently is limited to math.

Primary ongoing work includes:

* Actively switch between CuPy and NumPy when incompatibilities are present
* Additional syntax error help
* Additional tests to
    * ensure proper vectorization
    * increase Klong grammar coverage
* Make REPL (kgpy) compatible with original Klong (kg) REPL

# Differences from Klong

The main difference between Klong and KlongPy is that KlongPy doesn't infinite precision because it's backed by NumPy which is restricted to doubles.

# Running tests

```bash
python3 -m unittest
```

# Unused operators

The following operators are yet to be used:

```
:! :& :, :< :> :?
```

# Acknowledgement

HUGE thanks to Nils M Holm for his work on Klong and providing the foundations for this interesting project.

