
![Unit Tests](https://github.com/briangu/klongpy/workflows/Unit%20Tests/badge.svg)

# KlongPy

If you're intrigued by the world of [array languages](https://en.wikipedia.org/wiki/Array_programming) and love Python's versatility, meet KlongPy. This project marries the two, bringing Klong's elegant, terse syntax and the computational power of array programming to Python.

Born from the lineage of the [Klong](https://t3x.org/klong) array language, KlongPy leverages [NumPy](https://numpy.org/), an [Iverson Ghost](https://analyzethedatanotthedrivel.org/2018/03/31/NumPy-another-iverson-ghost/) that traces its roots back to APL, as its runtime target. This paves a relatively seamless path for mapping Klong constructs to NumPy, unleashing the performance benefits of this efficient, data-crunching Python library.

The charm of KlongPy? It integrates Python's extensive library ecosystem right into the Klong language. Whether it's feeding your machine learning model with Klong-processed data or mixing and matching Klong with other Python libraries for your data analysis workflow, KlongPy has you covered. And, with [CuPy](https://github.com/cupy/cupy) in play, you have the flexibility of both CPU and GPU backends at your disposal.

The project builds upon the work of [Nils M Holm](https://t3x.org), the creator of the Klong language, who has written a comprehensive [Klong Book](https://t3x.org/klong/book.html) for anyone interested in diving deeper. In short, if you're a data scientist, researcher, or just a programming language enthusiast, KlongPy may just be the next thing you want to check out.

# Overview

At its heart, KlongPy is about infusing Python's flexibility with the compact Klong array language, allowing you to tap into the performance of NumPy while writing code that's concise and powerful. 

Consider this simple Klong expression that computes an array's average: (+/a)%#a. Decoded, it means "sum of 'a' divided by the length of 'a'", from right to left.

Below, we define the function 'avg' and apply it to the array of integers from 0 -> 99 (as defined by !100)

Let's try this in the KlongPy REPL:

```Bash
$ rlwrap kgpy

Welcome to KlongPy REPL v0.3.78
author: Brian Guarraci
repo  : https://github.com/briangu/klongpy
crtl-d or ]q to quit

?> avg::{(+/x)%#x}
:monad
?> avg(!100)
49.49999999999999
```

We can also import Python modules to use directly in Klong language.  

Here we import the standard Python math library and redefine avg to use 'fsum':

```Bash
?> .py('math')
1
?> avg::{fsum(x)%#x}
:monad
?> avg(!100)
49.5
```

Now let's time it (first, right it once, then 100 times):

```
?> ]T avg(!100)
total: 2.710003172978759e-06 per: 2.710003172978759e-06
?> ]T:100 avg(!100)
total: 4.0871003875508904e-05 per: 4.0871003875508905e-07
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

# Python integration

Seamlessly blending Klong and Python is the cornerstone of KlongPy, enabling you to utilize each language where it shines brightest. For instance, you can integrate KlongPy into your ML/Pandas workflows, or deploy it as a powerhouse driving your website backend.

The charm of KlongPy lies in its dictionary-like interpreter that hosts the current KlongPy state, making it incredibly simple to extend KlongPy with custom functions and shuttle data in and out of the interpreter.

Imagine your data processed elsewhere, just set it into KlongPy and watch as the Klong language works its magic, accessing and manipulating your data with effortless ease. Even more, Python lambdas or functions can directly be exposed as Klong functions, adding an array of powerful tools to your Klong arsenal.

KlongPy indeed is a force multiplier, amplifying the power of your data operations.

## Function example

Call a Python function from Klong:

```python
from klongpy import KlongInterpreter
klong = KlongInterpreter()
klong['f'] = lambda x, y, z: x*1000 + y - z
r = klong('f(3; 10; 20)')
assert r == 2990
```

and vice versa, you can call a Klong function from Python:

```python
from klongpy import KlongInterpreter
klong = KlongInterpreter()
klong("f::{(x*1000) + y - z}")
r = klong['f'](3, 10, 20)
assert r == 2990
```

## Data example

Since the Klong interpreter context is dictionary-like, you can store values there for access in Klong:

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

## Python library access

Python functions, including lambdas, can be easily added to support common operations.  

In order to be consistent with Klong language, the paramters of Python functions may have at most three paramters and they must be x, y, and z.

```Python
from datetime import datetime
from klongpy import KlongInterpreter
klong = KlongInterpreter()
klong['strptime'] = lambda x: datetime.strptime(x, "%d %B, %Y")
klong("""
    a::strptime("21 June, 2018")
    .p(a)
    d:::{};d,"timestamp",a
    .p(d)
""")
```

prints the following dictionary to the console:

```
2018-06-21 00:00:00
{'timestamp': datetime.datetime(2018, 6, 21, 0, 0)}
```

You can go one step further and call back into Python from Klong with the result:

```Python
from datetime import datetime
from klongpy import KlongInterpreter
klong = KlongInterpreter()
klong['strptime'] = lambda x: datetime.strptime(x, "%d %B, %Y")
klong['myprint'] = lambda x: print(f"called from KlongPy: {x}")
klong("""
    a::strptime("21 June, 2018")
    myprint(a)
    d:::{};d,"timestamp",a
    myprint(d)
""")
```
outputs
```
called from KlongPy: 2018-06-21 00:00:00
called from KlongPy: {'timestamp': datetime.datetime(2018, 6, 21, 0, 0)}
```

## Loading Python Modules directly into KlongPy

KlongPy has the powerful ability to load Python modules directly. This can be extremely useful when you want to utilize the functionality offered by various Python libraries, and seamlessly integrate them into your KlongPy programs.  

Here is an example of how you can load a Python module into KlongPy:

```bash
$ rlwrap kgpy

Welcome to KlongPy REPL v0.3.76
author: Brian Guarraci
repo  : https://github.com/briangu/klongpy
crtl-d or ]q to quit

?> .py("math")
1
?> sqrt(64)
8.0
?> fsum(!100)
4950.0
```

In order to keep consistency with Klong 3-parameter function rules, KlongPy will attempt to remap loaded functions to use the x,y and z convention.  For example, in the Python math module, fsum is defined as fsum(seq), so KlongPy remaps this to fsum(x) so that it works within the runtime.


## Loading Custom Python Modules

Custom modules can be written for KlongPy in the same way as any Python module, the main
difference is that they don't need to be installed (e.g. via pip).

Simply create a directory with a __init__.py and appropriate files, as in:


```Python
# __init__.py
from .hello_world import hello
```

```Python
# hello_world.py

def hello():
    return "world!"

def not_exported():
    raise RuntimeError()
```

Now, you can import the module with the .py command and run the "hello" function.

```bash
$ rlwrap kgpy

Welcome to KlongPy REPL v0.3.76
author: Brian Guarraci
repo  : https://github.com/briangu/klongpy
crtl-d or ]q to quit

?> .py("tests/plugins/greetings")
1
?> hello()
world!
```


# Pandas DataFrame integration

This a work in progress, but it's very interesting to think about what DataFrames look like in Klong since they are basically a dictionary of columns.

For now, the following works where we convert a DataFrame into a dictionary of columns:

```Python
from klongpy import KlongInterpreter
import pandas as pd
import numpy as np

data = {'Name': ['Alice', 'Bob', 'Charlie', 'David'],
        'Age': [25, 30, 35, 40]}
df = pd.DataFrame(data)

klong = KlongInterpreter()
klong['df'] = {col: np.array(df[col]) for col in df.columns}
klong('df?"Name"') # ==> ['Alice', 'Bob', 'Charlie', 'David']
klong('df?"Age"')  # ==> [25, 30, 35, 40]
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

    Choose your CuPy prebuilt binary or from source.  Note, the ROCM support for CuPy is experimental and likely will have issues.

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
0.0005430681630969048
```

Read about the [prime example here](https://t3x.org/klong/prime.html).


# Status

KlongPy aims to be a complete implementation of klong.  It currently passes all of the integration tests provided by klong.

Since CuPy is [not 100% compatible with NumPy](https://docs.cupy.dev/en/stable/user_guide/difference.html), there are currently some gaps in KlongPy between the two backends.  Notably, strings are supported in CuPy arrays so KlongPy GPU support currently is limited to math.

Primary ongoing work includes:

* Actively switch between CuPy and NumPy when incompatibilities are present
    * Work on CuPy kernels is in this branch: _cupy_reduce_kernels
* Additional syntax error help
* Additional tests to
    * ensure proper vectorization
    * increase Klong grammar coverage
* Make REPL (kgpy) compatible with original Klong (kg) REPL


# Differences from Klong

While KlongPy aims to be 100% compatible with Klong language, the KlongPy system has some differences:

    * Infinite precision: The main difference in this implementation of Klong is the lack of infinite precision.  By using NumPy we are restricted to doubles.
    * Python integration: Most notably, the ".py" command allows direct import of Python modules into the current Klong context.
    * IPC - KlongPy will support IPC between KlongPy processes, allowing one KlongPy process to interact with other KlongPy processes over the network.

# Related

 * [Klupyter - KlongPy in Jupyter Notebooks](https://github.com/briangu/klupyter)
 * [Advent Of Code '22](https://github.com/briangu/aoc/tree/main/22)
 * [Example Ticker Plant with streaming and dataframes](https://github.com/briangu/kdfs)


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

