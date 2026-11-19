# Python Integration

The `kgpy` command and the `KlongInterpreter` class host the same Rust engine inside Python. Klong code can call Python functions, Python code can call Klong functions, and both read and write the same variables. This page is for someone embedding the interpreter in a Python program or using Python libraries from Klong.

The interpreter object behaves like a dictionary: `klong['name'] = value` defines a name for Klong code, and `klong['name']` reads one back. A Python function or lambda stored this way is a Klong function.

## The parameter rule

A Python function called from Klong takes at most three parameters, and they must be named `x`, `y` and `z`. Klong passes its arguments to the parameters with those names. A function defined as `lambda x, y: ...` works; `lambda a, b: ...` does not.

Known limit: a parameter with another name fails with a raw Python `TypeError` (`<lambda>() missing 2 required positional arguments: 'a' and 'b'`), not a Klong error, and the Python traceback is shown. Rename the parameters, or wrap the function: `klong['f'] = lambda x, y: g(x, y)`.

Functions loaded with `.py` and `.pyf` are remapped to this rule for you (see [Python library access](#python-library-access)).

## Types

Values cross the boundary as the objects in this table. Klong numbers and strings are plain Python values; lists are arrays of the active backend.

| Klong | Python on NumPy | Python on torch |
|---|---|---|
| integer `3` | `int` | `int` |
| real `1.5` | `float` | `float` |
| string `"ab"` | `str` | `str` |
| character `0cx` | `KGChar` | `KGChar` |
| symbol `:a` | `KGSym` | `KGSym` |
| list of numbers `[1 2 3]`, matrix `[[1 2][3 4]]` | `numpy.ndarray` | `torch.Tensor` |
| list of strings or mixed `["a" 1]` | `numpy.ndarray`, dtype `object` | `numpy.ndarray`, dtype `object` |
| dictionary `:{[1 2]}` | `dict` | `dict` |
| function | callable | callable |

A Python `int`, `float`, `str`, `dict`, `list`, `tuple` or `numpy.ndarray` stored with `klong['v'] = ...` is the Klong value of that name. A Python `list` or `tuple` is a list in Klong (`#v` is its length), and a Klong operation on it returns an array of the active backend. `bool` and `None` are kept as Python objects. A number that is only a literal, a stored value or a length (`3`, `#v`) comes back as `int` or `float`. The result of an operation that computes on numbers, such as arithmetic, a comparison, a reduction or an element taken from an array (`1+1`, `3>2`, `+/[1 2 3]`), is a NumPy scalar (`numpy.int64`, `numpy.float64`) on the NumPy backend and a 0-dimensional tensor on torch. Convert with `int(...)` or `float(...)` when a plain Python number matters.

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

A Klong function is a Python callable:

```console
$ python -W ignore <<'EOF'
> from klongpy import KlongInterpreter
> klong = KlongInterpreter()
> klong("f(a;b;c)::{(a*1000) + b - c}")
> r = klong['f'](3, 10, 20)
> print(r)
> EOF
2990
```

Arguments may be passed by name. A function defined with [named parameters](language-additions.md#named-parameters) takes any number of parameters, and its names are the ones in its definition. For a function written with `x`, `y` and `z` the names are `x`, `y` and `z`:

```console
$ python -W ignore <<'EOF'
> from klongpy import KlongInterpreter
> from klongpy.errors import KlongArityError
> klong = KlongInterpreter()
> klong("f(a;b;c)::{(a*1000) + b - c}")
> print(klong['f'](3, 10, c=20))
> klong("g(a;b;c;d)::{(a*1000)+b-c+d}")
> print(klong['g'](3, 10, 20, 1))
> print(klong['g'](3, 10, d=1, c=20))
> for call in (lambda: klong['f'](1, 2), lambda: klong['f'](1, 2, 3, 4),
>              lambda: klong['f'](1, 2, q=3), lambda: klong['f'](1, 2, 3, c=3)):
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

Store a value in the interpreter to use it in Klong:

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

Read a Klong variable back from Python:

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

A function imported with `.py` or `.pyf` that has optional parameters is called with every argument written in the call, however many: `.pyf("numpy";"linspace")` then `linspace(0;1;4;0)` passes the fourth argument too (`endpoint=0`). Too many or too few arguments are the Python `TypeError`, reported as a Klong error of kind `python`. A function with no optional parameters takes its required ones by position, at most three of them, and extra arguments are dropped (an arity error at [strict level 2](strict.md)). One with more than three required parameters is called with a list of the positional arguments and a dictionary of the keyword arguments, `f([1 2 3 4];:{})`, or through `.pyc`.

```console
$ python -W ignore <<'EOF'
> from datetime import datetime
> from klongpy import KlongInterpreter
> klong = KlongInterpreter()
> klong['strptime'] = lambda x: datetime.strptime(x, "%d %B, %Y")
> klong("""
>     a::strptime("21 June, 2018")
>     .p(a)
>     d:::{;"timestamp";a}
>     .p(d)
> """)
> EOF
2018-06-21 00:00:00
:{[timestamp 2018-06-21 00:00:00]}
```

`.p` prints the dictionary in Klong form. A Python function sees the Python objects, as the next example shows.

A Python function can also receive Python objects from Klong and print them:

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
>     d:::{;"timestamp";a}
>     myprint(d)
> """)
> EOF
called from KlongPy: 2018-06-21 00:00:00
called from KlongPy: {'timestamp': datetime.datetime(2018, 6, 21, 0, 0)}
```

## Loading Python modules

`.py("module")` loads a Python module and defines its public functions in Klong:

```kgpy
?> .py("math")
1
?> sqrt(64)
8.0
?> fsum(!100)
4950.0
```

To follow the [parameter rule](#the-parameter-rule), loaded functions are remapped to `x`, `y` and `z`. The Python `math.fsum(seq)` is called as `fsum(x)`.


## Loading custom Python modules

A custom module is an ordinary Python package that needs no installation. Create a directory with a `__init__.py` and appropriate files. The names that `__init__.py` imports are the functions that become available in Klong. Here the directory is `greetings`:

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

See also: [PyTorch Backend and Autograd](torch_backend.md), [System functions](system-functions.md) (`.py`, `.pyc`, `.pyf`) and the [roadmap](roadmap.md) for known limits.
