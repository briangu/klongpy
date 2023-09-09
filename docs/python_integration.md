# Python Integration


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
