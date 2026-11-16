# Python Integration


KlongPy runs inside Python. You can call Python functions from Klong, call Klong functions from Python, and move data between the two.

The interpreter object behaves like a dictionary that holds the current state: `klong['name'] = value` defines a name for Klong code, and `klong['name']` reads one back. A Python function or lambda stored this way is a Klong function.

## Function example

Call a Python function from Klong:

```console
$ python -W ignore <<'EOF'
> from klongpy import KlongInterpreter
> klong = KlongInterpreter()
> klong['f'] = lambda x, y, z: x*1000 + y - z
> r = klong('f(3; 10; 20)')
> print(r)
> EOF
2990
```

and vice versa, you can call a Klong function from Python:

```console
$ python -W ignore <<'EOF'
> from klongpy import KlongInterpreter
> klong = KlongInterpreter()
> klong("f::{(x*1000) + y - z}")
> r = klong['f'](3, 10, 20)
> print(r)
> EOF
2990
```

Arguments may also be passed by name. For a function written with `x`, `y` and `z` the names are `x`, `y` and `z`. A function defined with [named parameters](language-additions.md#named-parameters) takes any number of parameters, and its names are the ones in its definition:

```console
$ python -W ignore <<'EOF'
> from klongpy import KlongInterpreter
> from klongpy.errors import KlongArityError
> klong = KlongInterpreter()
> klong("f::{(x*1000) + y - z}")
> print(klong['f'](3, 10, z=20))
> klong("g(a;b;c;d)::{(a*1000)+b-c+d}")
> print(klong['g'](3, 10, 20, 1))
> print(klong['g'](3, 10, d=1, c=20))
> for call in (lambda: klong['f'](1, 2), lambda: klong['f'](1, 2, 3, 4),
>              lambda: klong['f'](1, 2, q=3), lambda: klong['f'](1, 2, 3, z=3)):
>     try:
>         call()
>     except KlongArityError as e:
>         print(e.kind)
> EOF
2990
2989
2989
arity
arity
arity
arity
```

A call from Python does not make a projection. Too few arguments, too many, an unknown name or a name given twice raises `KlongArityError` (a `KlongError`, kind `arity`), as the four calls above show.

## Data example

Since the Klong interpreter context is dictionary-like, you can store values there for access in Klong:

```console
$ python -W ignore <<'EOF'
> import numpy as np
> from klongpy import KlongInterpreter
> klong = KlongInterpreter()
> data = np.arange(10)
> klong['data'] = data
> print(klong('1+data'))
> EOF
[ 1  2  3  4  5  6  7  8  9 10]
```

Variables may be directly retrieved from KlongPy context:

```console
$ python -W ignore <<'EOF'
> import numpy as np
> from klongpy import KlongInterpreter
> klong = KlongInterpreter()
> klong['data'] = np.arange(5)
> klong('Q::1+data')
> Q = klong['Q']
> print(type(Q).__name__, Q)
> EOF
ndarray [1 2 3 4 5]
```

## Python library access

Python functions, including lambdas, can be easily added to support common operations.

In order to be consistent with the Klong language, the parameters of Python functions may have at most three parameters and they must be x, y, and z.

```console
$ python -W ignore <<'EOF'
> from datetime import datetime
> from klongpy import KlongInterpreter
> klong = KlongInterpreter()
> klong['strptime'] = lambda x: datetime.strptime(x, "%d %B, %Y")
> klong("""
>     a::strptime("21 June, 2018")
>     .p(a)
>     d:::{};d,"timestamp",a
>     .p(d)
> """)
> EOF
2018-06-21 00:00:00
:{[timestamp 2018-06-21 00:00:00]}
```

`.p` prints the dictionary in Klong form. A Python function sees the Python objects, as the next example shows.

You can go one step further and call back into Python from Klong with the result:

```console
$ python -W ignore <<'EOF'
> from datetime import datetime
> from klongpy import KlongInterpreter
> klong = KlongInterpreter()
> klong['strptime'] = lambda x: datetime.strptime(x, "%d %B, %Y")
> klong['myprint'] = lambda x: print(f"called from KlongPy: {x}")
> klong("""
>     a::strptime("21 June, 2018")
>     myprint(a)
>     d:::{};d,"timestamp",a
>     myprint(d)
> """)
> EOF
called from KlongPy: 2018-06-21 00:00:00
called from KlongPy: {'timestamp': datetime.datetime(2018, 6, 21, 0, 0)}
```

## Loading Python Modules directly into KlongPy

KlongPy has the powerful ability to load Python modules directly. This can be extremely useful when you want to utilize the functionality offered by various Python libraries, and seamlessly integrate them into your KlongPy programs.  

Here is an example of how you can load a Python module into KlongPy:

```kgpy
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

Simply create a directory with a `__init__.py` and appropriate files. The names that `__init__.py` imports are the functions that become available in Klong. Here the directory is `greetings`:

```console
$ mkdir greetings
$ cat > greetings/__init__.py <<'EOF'
> from .hello_world import hello
> EOF
$ cat > greetings/hello_world.py <<'EOF'
> def hello():
>     return "world!"
> 
> def not_exported():
>     raise RuntimeError()
> EOF
$ cat > main.kg <<'EOF'
> .py("greetings")
> .p(hello())
> EOF
$ kgpy main.kg
world!
```

`.py` takes a module name or the path of a directory that holds such a package. Only the names imported in `__init__.py` are defined in Klong; `not_exported` is not.

See also: [PyTorch backend](torch_backend.md), [System functions](system-functions.md) (`.py`, `.pyc`, `.pyf`) and the [roadmap](roadmap.md) for known limits.
