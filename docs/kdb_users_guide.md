# KlongPy for KDB Users

This guide helps KDB+/q developers quickly become productive with KlongPy. It highlights language similarities, shows how to connect to an existing kdb+ instance, and demonstrates how common q idioms translate into KlongPy.

## 1. Installing KlongPy

Install the full feature set (including optional modules):

```bash
pip3 install "klongpy[full]"
```

Optional GPU support can be enabled with `USE_GPU=1` and an appropriate CuPy package installed as noted in the [quick start](quick-start.md).

## 2. Launching the REPL

Use the bundled REPL to experiment with KlongPy expressions:

```bash
rlwrap kgpy
```

The REPL understands Klong syntax which shares many concepts with q. Functions use positional parameters `x`, `y`, and `z` and can be called like q verbs.

Example average function:

```kgpy
?> avg::{(+/x)%#x}
:monad
?> avg(!10)
4.5
```

## 3. Connecting to kdb+

KlongPy includes experimental support for q processes via the `qpython` library. Three system functions are provided:

- `.qcli(x)` – create a q client. `x` may be a port number (`5000`) or a string like `"host:port"`. The result is a remote function handle.
- `.qclid(x)` – similar to `.qcli` but returns a dictionary-style handle for getting and setting remote values.
- `.qclic(x)` – close a handle returned by `.qcli` or `.qclid`.

```kgpy
?> q::.qcli(5000)
q[localhost:5000]:fn
?> q("2+2")
4
?> d::.qclid(q)
q[localhost:5000]:dict
?> d?:version
3.6
```

Remote q functions appear as callable proxies. You can store them in variables and invoke them just like local functions.

```kgpy
?> sumq::q(:sum)
q[localhost:5000]:sum:fn
?> sumq(!5)
10
```

Use `.qclic` when finished:

```kgpy
?> .qclic(q)
1
```

## 4. Working with Tables

KlongPy offers a lightweight table module backed by DuckDB for fast, zero‑copy SQL over NumPy arrays. While not a full kdb+ replacement, it covers many everyday tasks.

```kgpy
?> .py("klongpy.db")
?> t::.table([["sym" ["A" "B"]] ["px" [10 20]]])
?> db::.db(:{},"T",t)
?> db("select * from T where px>15")
[sym px]
[B 20]
```

Tables behave similarly to q tables: columns may be indexed, appended to, or merged using the helper functions in `klongpy.db`.

## 5. IPC and Asynchronous Workflows

For distributed workflows reminiscent of q IPC, KlongPy provides its own IPC layer. Servers are started with `.srv(port)` and clients connect via `.cli(port)` or `.clid(port)`. Functions and dictionaries returned behave as local proxies.

```kgpy
# server
?> .srv(8888)

# client
?> f::.cli(8888)
remote[localhost:8888]:fn
?> f("avg::{(+/x)%#x}")
:monad
?> f(:avg,,!100)
49.5
```

The `.async(fn;cb)` utility wraps a function for asynchronous execution, invoking `cb` when the remote call completes.

## 6. Translating Common q Idioms

Below are a few examples of how familiar q expressions translate to KlongPy.

| q Expression | KlongPy Equivalent |
|--------------|-------------------|
| `avg til 10` | `(+/!10)%#10` |
| `count a` | `#a` |
| `a + b` | `a+b` |
| `select from t where sym=`A`` | `db("select * from t where sym = 'A'")` |
| Asynchronous call `h(`func;args)` | `.async(h(:func);cb)` |

KlongPy syntax is largely a superset of Klong, so most array and dictionary operations map directly to concise expressions.

## 7. Next Steps

- Explore the [examples](examples.md) folder for more scripts.
- Review [python_integration.md](python_integration.md) for mixing Python and KlongPy.
- Use the [IPC capabilities](ipc_capabilities.md) to build distributed systems.

KlongPy aims to provide a familiar environment for array‑oriented programming while embracing Python's ecosystem. With these tools, q developers can gradually adopt KlongPy for new projects or integrate it alongside existing kdb+ deployments.

