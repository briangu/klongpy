# Code Review Bug Report (2026-03-06)

Review scope:

- Runtime-verified against the core interpreter with `.venv312/bin/python` and selected pytest modules.
- Static review only for the web subsystem because `aiohttp` is not installed in the local venv.

## 1. Parser silently accepts trailing unmatched braces

Affected code:

- `klongpy/interpreter.py:499-508`

Repro:

```python
from klongpy import KlongInterpreter
k = KlongInterpreter()

k("a::1}}")                 # returns 1
k("aggs::{[a];a:::{}}}}")   # returns a function object
```

Actual behavior:

- Once `prog()` finishes a valid expression, it stops when the next token is not `;`.
- `exec()` returns the parsed expressions without rejecting unmatched trailing `}` tokens.
- Extra closing braces after an otherwise valid expression are ignored instead of raising a syntax error.

Expected behavior:

- Unmatched trailing `}` tokens should raise a parse error.

## 2. Parenthesized join with indexed values does not parse

Affected code:

- `klongpy/interpreter.py:437-439`
- `klongpy/interpreter.py:454-485`

Repro:

```python
from klongpy import KlongInterpreter
k = KlongInterpreter()
k("q::[3 8]")

k("q[0],q[-1]")     # works, returns [3 8]
k("(q[0],q[-1])")   # raises UnexpectedChar at ')'
```

Actual behavior:

- The same join expression parses at top level but fails when wrapped in parentheses.

Expected behavior:

- Parentheses should preserve the value of `q[0],q[-1]`, not make it unparsable.

## 3. Function argument parsing drops a string argument whose value is `";"`

Affected code:

- `klongpy/interpreter.py:317-368`

Repro:

```python
from klongpy import KlongInterpreter
k = KlongInterpreter()
k('f::{x,y}')
r = k('f("hello";";")')
print(type(r), r.args)
```

Actual behavior:

- The call returns a partially applied `KGCall` with `args == ['hello', None]`.
- The second argument is treated as an omitted projection slot instead of the string `";"`.

Expected behavior:

- `f("hello";";")` should evaluate to `"hello;"`.

## 4. Dyadic join misclassifies dictionary operands

Affected code:

- `klongpy/dyads.py:556-561`

Repro A:

```python
from klongpy import KlongInterpreter
k = KlongInterpreter()
k("b:::{[1 2]}")
k("c:::{[3 4]}")
k("b,c")   # raises KeyError: 1
```

Repro B:

```python
from klongpy import KlongInterpreter
k = KlongInterpreter()
k("A::[];A::A,:{}")   # raises TypeError: unhashable type: 'dict'
```

Actual behavior:

- Any dictionary on the left is treated as if the right-hand side were always a two-element tuple.
- Any dictionary on the right is treated as a dictionary update whenever the left operand has length 2.
- This breaks both dictionary merge-style joins and appending dictionary values into lists.

Expected behavior:

- `dict,dict` should merge dictionaries.
- `list,dict` should append the dictionary as a value when the left operand is not an update tuple.

## 5. Nested dictionary literals keep inner dictionaries unevaluated

Affected code:

- `klongpy/parser.py:245-248`

Repro A:

```python
from klongpy import KlongInterpreter
k = KlongInterpreter()
print(k(':{[1 :{[2 3]}]}'))
```

Actual behavior:

- The value is `{1: <KGCall ...>}` instead of `{1: {2: 3}}`.

Repro B:

```python
from klongpy import KlongInterpreter
k = KlongInterpreter()
k('c:::{["GET" :{["/" 2]}]}')
k('(c?"GET")?"/"')   # raises ValueError from numpy.where on a 0d array
```

Actual behavior:

- The outer dictionary stores a deferred `KGCall(copy_lambda, ...)` rather than the inner dictionary value.
- The next lookup runs against that unevaluated wrapper and falls into the generic array-search path.

Expected behavior:

- Nested dictionary literals should be fully materialized before being inserted into the outer dictionary.

## 6. Take flattens nested arrays instead of repeating elements

Affected code:

- `klongpy/dyads.py:1017-1032`

Repro:

```python
from klongpy import KlongInterpreter
k = KlongInterpreter()
print(k("(4)#[[0 0]]"))
```

Actual behavior:

- The result is a single row with eight zeros (`array([[0, 0, 0, 0, 0, 0, 0, 0]])`).

Expected behavior:

- `(4)#[[0 0]]` should repeat the nested element four times:
  `[[0 0] [0 0] [0 0] [0 0]]`.

Notes:

- `eval_dyad_take()` converts the input with `np.asarray(b)`, so `[[0 0]]` becomes a 2D numeric array and `tile()` repeats the inner scalars instead of list elements.

## 7. `CallbackEvent.trigger()` crashes when callbacks mutate subscriptions

Affected code:

- `klongpy/utils.py:33-35`

Repro:

```python
from klongpy.utils import CallbackEvent

event = CallbackEvent()

def cb():
    event.unsubscribe(cb)

event.subscribe(cb)
event.trigger()   # RuntimeError: Set changed size during iteration
```

Actual behavior:

- Iterating the live `set` makes self-unsubscribe and similar cleanup patterns crash.

Expected behavior:

- Triggering should iterate over a snapshot so subscribers can safely subscribe or unsubscribe during dispatch.

## 8. Web server shutdown/docs path is inconsistent, and startup logging is malformed

Affected code:

- `klongpy/web/sys_fn_web.py:63-65`
- `klongpy/web/sys_fn_web.py:87`
- `klongpy/web/sys_fn_web.py:112`
- `klongpy/web/sys_fn_web.py:150-156`

Status:

- Static review only; not runtime-tested locally because `aiohttp` is unavailable.

Findings:

- `logging.info("web server start @ ", x)` and the similar route log lines pass extra arguments without format placeholders. Python logging emits formatting errors for these calls.
- `.webc()` only shuts a server down when `x` is a `KGCall` whose `.a` is a `KGLambda`.
- `.web()` returns a raw `WebServerHandle`, and the docs/examples use `.webc(h)`, so the documented shutdown flow does not match the implementation.

Expected behavior:

- Startup/route logging should use formatted messages.
- `.webc()` should accept the `WebServerHandle` that `.web()` returns.
