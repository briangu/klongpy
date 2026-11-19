# Klong for q users

This page is for people who know q and want to read or write Klong. It maps the idioms, lists the differences that cause mistakes, and describes how a KlongPy program and a q process talk to each other over the kdb+ IPC protocol.

The q examples on this page were checked with [PeachQ](https://www.timestored.com/peachq/) 0.92, a third-party open-source implementation of q. PeachQ is not kdb+: where the page says what q prints, it means what PeachQ 0.92 printed, and the page makes no claim about kdb+ beyond that. Conditions: every q result shown is from PeachQ 0.92. The Klong examples run at [strict level 2](strict.md), except the sessions that need a q process.

kdb+ is a trademark of KX. KlongPy's IPC is an independent implementation, written from public documentation and Apache-licensed client code. It speaks the kdb+ IPC protocol and interoperates with q processes; it is not a KX product and uses no KX code.

If you have not used Klong at all, read the [tutorial](tutorial.md) first or alongside this page. The [operator reference](operators.md) is the lookup table.

## What carries over

Klong and q are both array languages in the APL line, and the shape of a program is the same.

- Evaluation is right to left and there is no operator precedence. `2*3+4` is 14 in both.
- A verb is a monad or a dyad by position. `-x` negates and `a-b` subtracts.
- Adverbs modify verbs: `'` each, `/` over, `\` scan. `+/x` sums, `+\x` is the running sum.
- A function takes its arguments as `x`, `y` and `z`, and its rank is the highest of them that its body uses.
- Lists are the basic data type, and most verbs apply to every element of a list.
- Strings are lists of characters, symbols are atoms, and a dictionary maps keys to values.

```q
q)2*3+4
14
q){x+z}[1;2;3]
4
```

<!-- check-docs: strict 2 -->
```kgpy
?> 2*3+4
14
?> {x+z}(1;2;3)
4
```

Klong is smaller than q. It has integers, floats (doubles), characters, strings, symbols, lists, dictionaries and functions. It has no boolean, no temporal types, no typed nulls, no q-sql and no namespaces. A file is a [module](language-additions.md#modules), a dictionary of the names it exports, and `a.b` reads the entry `b` of `a`.

## What is different

### A leading minus is a verb

In q, `-1+2` is 1: the `-1` is read as a number. In Klong a leading `-` is the negate monad and applies to everything to its right, so `-1+2` is `-(1+2)`, which is -3. A negative number is written inside a list (`[-1 2]`) or in parentheses. [Lint](lint.md) warns about the trap as W010.

```q
q)-1+2
1
q)(-1)+2
1
```

<!-- check-docs: strict 2 -->
```kgpy
?> -1+2
-3
?> (-1)+2
1
?> [-1 2]
[-1 2]
```

The same rule makes a slice from the end `(-2)#y` or `(0-2)#y`, not `-2#y`.

### Calls and indexing

Square brackets build lists in Klong. They never index or call. A call is `f(a;b)`, indexing is `a@i`, and applying a function with `@` works too: `f@x`. A one-argument call needs no space and a space before the parenthesis is a [lint](lint.md) warning (W014).

```q
q){x+y}[1;2]
3
q)3 1 2@1
1
```

<!-- check-docs: strict 2 -->
```kgpy
?> f(a;b)::{a+b}
:dyad
?> f(1;2)
3
?> [3 1 2]@1
1
?> f(1;)@10
11
```

A call with fewer arguments than the function takes is a projection, as in q. Klong writes the missing argument as an empty slot or leaves it off: `f(1;)` and `f(1)` both give a projection. A call with too many arguments drops the extras at strict levels 0 and 1 and is an arity error at level 2.

Assignment is `::` everywhere. In q, `::` inside a function assigns a global; in Klong it is plain assignment, and a name that is not a parameter, not declared as a local (`{[t];...}`) and not already global becomes a local of that call at level 0 and is an error at strict levels 1 and 2. A global is created by assigning at the top level. Outside functions, `d:::{...}` is the assignment `d::` followed by the dictionary literal `:{...}`. See [strict mode](strict.md).

<!-- check-docs: strict 2 -->
```kgpy
?> total::0
0
?> bump(n)::{total::total+n}
:monad
?> bump(5)
5
?> total
5
?> scaled(n)::{[t];t::n*2;t+1}
:monad
?> scaled(5)
11
```

Klong has [named parameters](language-additions.md#named-parameters), `f(a;b;c;d)::{...}`, with no limit on their number. The q names `x y z` also work in a bare `{...}`: `{x+y}` is a dyad, and it is the short form to give an adverb. [Destructuring](language-additions.md#destructuring) takes the place of q's `a:x 0;b:x 1`.

### No booleans

Comparisons return the integers 0 and 1. There is no `1b`, and no boolean vector type; the verbs that take a mask (`&`, `~`) take 0 and 1. The conditional is `:[c;a;b]`, where q has `$[c;a;b]`. An else-if chain is `:[c1;a:|c2;b;d]`.

```q
q)3>2
1b
q)$[1<2;"yes";"no"]
"yes"
```

<!-- check-docs: strict 2 -->
```kgpy
?> 3>2
1
?> :[1<2;"yes";"no"]
"yes"
?> sign(n)::{:[n<0;"neg":|n=0;"zero";"pos"]}
:monad
?> sign'[-5 0 7]
["neg" "zero" "pos"]
```

A condition is false for 0, for an empty list and for an empty string; everything else is true.

### Nulls: `:undefined`

q has a typed null for each type (`0N`, `0n`) and infinities (`0w`). Klong has one undefined value, written `:undefined`. It is what a division by zero gives, what a missing dictionary key gives, and what a failed conversion gives. Test for it with the `:_` monad.

```q
q)1%0
0w
q)0%0
0n
q)(`a`b!1 2)`zz
0N
```

<!-- check-docs: strict 2 -->
```kgpy
?> 1%0
:undefined
?> :_(1%0)
1
?> d:::{[:a 1] [:b 2]}
:{[:a 1] [:b 2]}
?> d@:zz
:undefined
?> :_d@:zz
1
?> 1:$"x"
:undefined
```

A list that holds `:undefined` can be printed and indexed, but arithmetic on it is an error. Nulls are not translated over IPC: see [the type table](#types-across-the-connection).

### Find, group and the dyad `?`

q's `x?y` returns the index of the first match, or the count of `x` when there is none. Klong's `x?y` returns the list of all matching indices, empty when there is none. Use `*` on the result for the first match.

```q
q)3 1 3 2 1?3
0
q)3 1 2?5
3
```

<!-- check-docs: strict 2 -->
```kgpy
?> z::[3 1 3 2 1]
[3 1 3 2 1]
?> z?3
[0 2]
?> z?7
[]
?> *z?3
0
```

`=z` is group, and it returns only the lists of indices, ordered by the first appearance of each value: `=[3 1 3 2 1]` is `[[0 2] [1 4] [3]]`. q's `group` returns a dictionary keyed by the values. `?z` is q's `distinct`, in order of first appearance, not sorted. For a random number use `.rn()`.

### Scan with a seed

A scan with a seed keeps the seed in Klong. q's does not.

```q
q)10 +\ 1 2 3
11 13 16
```

<!-- check-docs: strict 2 -->
```kgpy
?> 10+\[1 2 3]
[10 11 13 16]
```

### Strings, characters and symbols

A Klong string is a list of characters, like a q string, but a one-character string is still a string: `"a"` is a string of length 1 and the character is written `0ca`. In q `"a"` is a character atom, which matters over IPC (see below). A symbol is written `:abc` where q writes `` `abc ``, and a list of symbols is `[:a :b :c]`. A symbol held as data is never looked up as a variable.

<!-- check-docs: strict 2 -->
```kgpy
?> #"a"
1
?> [:a :b :c]
[:a :b :c]
?> foo::5
5
?> {x}'[:foo :bar]
[:foo :bar]
```

A list literal holds what is written in it. To evaluate expressions inside a list, start it with a semicolon: `[;1+1;2+2]` is q's `(1+1;2+2)`.

<!-- check-docs: strict 2 -->
```kgpy
?> [;1+1;2+2]
[2 4]
```

### Dictionaries

A dictionary literal lists `[key value]` pairs. Lookup is `d@k` or `d?k`, and a missing key gives `:undefined`. `d@[k1 k2]` looks up several keys. Joining a pair or another dictionary with `,` adds entries, and it changes the dictionary in place (q's `,` returns a new one). A dictionary literal whose keys and values are expressions is `:{;k;v;...}`. See [Language Additions](language-additions.md#dictionaries).

```q
q)d:`a`b!1 2
q)d,`c`d!3 4
a| 1
b| 2
c| 3
d| 4
```

<!-- check-docs: strict 2 -->
```kgpy
?> d:::{[:a 1] [:b 2]}
:{[:a 1] [:b 2]}
?> e::d
:{[:a 1] [:b 2]}
?> d,:{[:c 3] [:d 4]}
:{[:a 1] [:b 2] [:c 3] [:d 4]}
?> e
:{[:a 1] [:b 2] [:c 3] [:d 4]}
?> d@[:a :zz]
[1 :undefined]
?> *'d
[:a :b :c :d]
?> {x@1}'d
[1 2 3 4]
?> :{;:x;1+1;:y;2*3}
:{[:x 2] [:y 6]}
```

### Division

`%` is float division in both languages. `x div y` is `x:%y` and `x mod y` is `x!y`, with one difference for negative numbers: Klong truncates toward zero where q floors. `(-7):%2` is -3 in Klong and -4 in q, and `(-7)!2` is -1 in Klong and 1 in q.

## Translation table

Each group is one table of q expressions, their Klong forms and the q result (from PeachQ 0.92), followed by the Klong expressions run in a session. The Klong blocks continue each other, so the variables set in the first are available in the others. The q expressions use these definitions:

```q
x:3 1 2
y:1 2 3 4
z:3 1 3 2 1
w:(1 2;3 4 5)
m:(1 2 3;4 5 6)
d:`a`b!1 2
b:10110b
```

Where the Klong result differs from the q result in a table:

| Expression | q | Klong |
|---|---|---|
| `avg x` | `2f` | `2.0` |
| `group z` | a dictionary from value to positions | the list of position lists `[[0 2] [1 4] [3]]` |
| `z?3` | `0`, the first position | `[0 2]`, all positions |
| `z?7` | `5`, the count when not found | `[]` |
| `` d`zz `` | `0N` | `:undefined` |
| `1%0` | `0w` | `:undefined` |
| `-1+2` | `1` | `-3`, a leading minus is a verb |
| `3>2` | `1b` | `1` |
| `10 +\ 1 2 3` | `11 13 16` | `[10 11 13 16]`, the seed is included |


### Lists

| q | Klong | Result in q |
|---|---|---|
| `til 5` | `!5` | `0 1 2 3 4` |
| `count x` | `#x` | `3` |
| `sum x` | `+/x` | `6` |
| `sums x` | `+\x` | `3 4 6` |
| `avg x` | `(+/x)%#x` | `2f` |
| `where b` | `&b` | `0 2 3` |
| `iasc x` | `<x` | `1 2 0` |
| `x@iasc x` | `x@<x` | `1 2 3` |
| `x@idesc x` | `x@>x` | `3 2 1` |
| `distinct z` | `?z` | `3 1 2` |
| `reverse x` | `\|x` | `2 1 3` |
| `first x` | `*x` | `3` |
| `2#y` | `2#y` | `1 2` |
| `2_y` | `2_y` | `3 4` |
| `-2#y` | `(-2)#y` | `3 4` |
| `neg x` | `-x` | `-3 -1 -2` |
| `x,y` | `x,y` | `3 1 2 1 2 3 4` |
| `raze w` | `,/w` | `1 2 3 4 5` |
| `count each w` | `#'w` | `2 3` |
| `flip m` | `+m` | `1 4` / `2 5` / `3 6` |
| `2 3#til 6` | `[2 3]:^!6` | `0 1 2` / `3 4 5` |

The same expressions in KlongPy:

<!-- check-docs: strict 2 -->
```kgpy
?> x::[3 1 2]
[3 1 2]
?> y::[1 2 3 4]
[1 2 3 4]
?> z::[3 1 3 2 1]
[3 1 3 2 1]
?> w::[[1 2] [3 4 5]]
[[1 2] [3 4 5]]
?> m::[[1 2 3] [4 5 6]]
[[1 2 3] [4 5 6]]
?> d:::{[:a 1] [:b 2]}
:{[:a 1] [:b 2]}
?> b::[1 0 1 1 0]
[1 0 1 1 0]
?> !5
[0 1 2 3 4]
?> #x
3
?> +/x
6
?> +\x
[3 4 6]
?> (+/x)%#x
2.0
?> &b
[0 2 3]
?> <x
[1 2 0]
?> x@<x
[1 2 3]
?> x@>x
[3 2 1]
?> ?z
[3 1 2]
?> |x
[2 1 3]
?> *x
3
?> 2#y
[1 2]
?> 2_y
[3 4]
?> (-2)#y
[3 4]
?> -x
[-3 -1 -2]
?> x,y
[3 1 2 1 2 3 4]
?> ,/w
[1 2 3 4 5]
?> #'w
[2 3]
?> +m
[[1 4] [2 5] [3 6]]
?> [2 3]:^!6
[[0 1 2] [3 4 5]]
```

### Finding, grouping and dictionaries

| q | Klong | Result in q |
|---|---|---|
| `group z` | `=z` | `3\| 0 2` / `1\| 1 4` / `2\| ,3` |
| `z?3` | `z?3` | `0` |
| `z?7` | `z?7` | `5` |
| `` d`a `` | `d@:a` | `1` |
| `` d`zz `` | `d@:zz` | `0N` |

The same expressions in KlongPy:

<!-- check-docs: strict 2; continue -->
```kgpy
?> =z
[[0 2] [1 4] [3]]
?> z?3
[0 2]
?> z?7
[]
?> d@:a
1
?> d@:zz
:undefined
```

### Arithmetic, conditions and functions

| q | Klong | Result in q |
|---|---|---|
| `7 mod 3` | `7!3` | `1` |
| `7 div 2` | `7:%2` | `3` |
| `7%2` | `7%2` | `3.5` |
| `3>2` | `3>2` | `1b` |
| `1%0` | `1%0` | `0w` |
| `-1+2` | `-1+2` | `1` |
| `(-1)+2` | `(-1)+2` | `1` |
| `$[1<2;"yes";"no"]` | `:[1<2;"yes";"no"]` | `"yes"` |
| `f:{x+y};f[1;2]` | `f(a;b)::{a+b};f(1;2)` | `3` |
| `x@1` | `x@1` | `1` |

The same expressions in KlongPy:

<!-- check-docs: strict 2; continue -->
```kgpy
?> 7!3
1
?> 7:%2
3
?> 7%2
3.5
?> 3>2
1
?> 1%0
:undefined
?> -1+2
-3
?> (-1)+2
1
?> :[1<2;"yes";"no"]
"yes"
?> f(a;b)::{a+b};f(1;2)
3
?> x@1
1
```

### Adverbs

| q | Klong | Result in q |
|---|---|---|
| `{x*2} each x` | `{x*2}'x` | `6 2 4` |
| `max x` | `\|/x` | `3` |
| `min x` | `&/x` | `1` |
| `10 +/ 1 2 3` | `10+/[1 2 3]` | `16` |
| `10 +\ 1 2 3` | `10+\[1 2 3]` | `11 13 16` |
| `3 {x*2}/ 1` | `3{x*2}:*1` | `8` |
| `({x div 2}/) 100` | `{x:%2}:~100` | `0` |

The same expressions in KlongPy:

<!-- check-docs: strict 2; continue -->
```kgpy
?> {x*2}'x
[6 2 4]
?> |/x
3
?> &/x
1
?> 10+/[1 2 3]
16
?> 10+\[1 2 3]
[10 11 13 16]
?> 3{x*2}:*1
8
?> {x:%2}:~100
0
```


## Tables and queries

q has tables and q-sql in the core. Klong does not. With the database extras installed (`pip install "klongpy[db]"`), `klongpy.db` provides tables backed by pandas, and queries are SQL strings run by DuckDB over those tables. See [Columnar tables and SQL](fast_columnar_database.md) and [Table and Key-Value Stores](table_and_key_value_stores.md). There is no `select ... by` syntax, no keyed tables, no attributes, and no splayed or partitioned tables.

```q
q)t:([] sym:`a`b`a`b`a; px:1 2 3 4 5)
q)exec sum px by sym from t
a| 9
b| 6
q)exec px from t where px>2
3 4 5
q)exec sum px from t
15
q)cols t
`sym`px
```

<!-- check-docs: requires duckdb pandas; strict 2 -->
```kgpy
?> .py("klongpy.db")
1
?> t::.table([["sym" ["a" "b" "a" "b" "a"]] ["px" [1 2 3 4 5]]])
sym px
  a  1
  b  2
  a  3
  b  4
  a  5
?> .schema(t)
["sym" "px"]
?> t?"px"
[1 2 3 4 5]
?> db::.db(:{;"t";t})
:db
?> db("select sym, sum(px) as s from t group by sym order by sym")
[["a" 9.0] ["b" 6.0]]
?> db("select px from t where px>2")
[3 4 5]
?> db("select sum(px) from t")
15.0
```

A query returns plain Klong values, not a table: one column gives a list, several columns give a list of rows, one value gives a number. `t,"qty",,[10 20 30 40 50]` adds a column. DuckDB's `sum` of integers comes back as a float.

## Timers

q's `.z.ts` runs once per `\t` interval. Klong has `.timer(name;seconds;callback)`: a timer is a value, there may be many, and the callback's result decides whether it runs again (1 to continue, 0 to stop). `.timerc(t)` cancels one. The interval may be fractional. See [Timers](timer.md).

```q
n:0
.z.ts:{n+::1;show n;if[n>=3;system"t 0"]}
system"t 200"
```

That q script prints `1`, `2` and `3`, one per line, and then stops the timer. The Klong version prints the same:

```
n::0
cb::{n::n+1;.p(n);:[n<3;1;0]}
th::.timer("t";0.2;cb)
```

Run it with `kgpy -l file.kg`, which stays in the REPL while the timer fires. `kgpy file.kg` runs the file and exits without waiting for a timer, so it prints nothing for this program.

## Talking to q: IPC

KlongPy's `.srv`, `.cli`, `.clid`, `.clic`, `.clit` and `.async` speak the kdb+ IPC protocol. A Klong program can connect to a q process, and a q process can connect to a Klong server. The full description, including how functions cross the wire, is in [IPC Capabilities](ipc_capabilities.md). This section is what a q user needs.

Limits: no TLS, so a user name and password travel in clear text, and a peer that can reach the port can run Klong code in your interpreter, including `.py` and `.sys`. The other limits (message size, compression) are in [IPC Capabilities](ipc_capabilities.md#protocol-and-security).

Tested: KlongPy against PeachQ 0.92 in both directions, and against a protocol-level fake of a kdb+ server; not against a KX kdb+ process, so the behaviours below are those of PeachQ. The tests are in `tests/test_ipc_peachq.py` and run when `KLONG_TEST_PEACHQ` names a PeachQ executable.

### Addresses

| q | Klong | Meaning |
|---|---|---|
| `` hopen `:localhost:8888 `` | `.cli("localhost:8888")` or `.cli(8888)` | connect to a port, on localhost for the bare integer |
| `` hopen `:host:8888:user:pw `` | `.cli("host:8888:user:pw")` | with a user name and password, in clear text |
| `q -p 8888` | `.srv(8888)` or `kgpy -s 8888` | listen on a port |
| | `.srv(0)` | stop listening and drop the clients |
| `hclose h` | `.clic(h)` | close a connection |

### A Klong program calls a q process

`.cli` returns a function. Call it with a string to evaluate q source in the q process, or with a symbol and arguments to call a function that is defined there by name. A string is source text in the language of the process that receives it. The result comes back as a Klong value.

Start a q process that listens on port 8888, for example `q -p 8888`, and then, in a Klong session:

<!-- check-docs: skip: needs a running q process (PeachQ 0.92) -->
```kgpy
?> h::.cli("localhost:8888")
remote[localhost:8888]:fn
?> h("1+1")
2
?> h("til 5")
[0 1 2 3 4]
?> h("`a`b!1 2")
:{[:a 1] [:b 2]}
?> h("mean:{(sum x)%count x}")
:undefined
?> h(:mean,,[1 2 3 4])
2.5
?> h("mean 1 2 3 4")
2.5
```

`h(:mean,,[1 2 3 4])` is q's `h(`mean;1 2 3 4)`: the callee as a symbol, then its arguments. The `,,` wraps the list so that it is one argument. With two arguments write `h(:add,3,4)`. Assigning in q returns the generic null, which arrives as `:undefined`.

An error in q comes back as a Klong error of kind `remote` whose message is the text of the q signal. The connection stays open.

<!-- check-docs: skip: needs a running q process (PeachQ 0.92) -->
```kgpy
?> h("nosuch")
error: nosuch
?> h("1+`a")
error: type
?> h("1 2 3 + 4 5")
error: length
?> h("1+1")
2
```

A table from q arrives as a table when `klongpy.db` is loaded in the Klong process, and as a dictionary from column symbol to column array otherwise. `.clid` gives the remote context as a dictionary, which sets and gets variables in the q process:

<!-- check-docs: skip: needs a running q process (PeachQ 0.92) -->
```kgpy
?> .py("klongpy.db")
1
?> T::h("([] a:1 2 3; b:1.5 2.5 3.5)")
a   b
1 1.5
2 2.5
3 3.5
?> T?"b"
[1.5 2.5 3.5]
?> .schema(T)
["a" "b"]
?> d::.clid(h)
remote[localhost:8888]:dict
?> d,[:foo 2]
remote[localhost:8888]:dict
?> d?:foo
2
?> h("foo")
2
?> .clic(h)
1
```

### A q process calls a Klong server

Start the server with `kgpy -s 8888`, or call `.srv(8888)` in a program. A q client sends a string, which is Klong source, or a call by symbol. These were run against a KlongPy server that had `avg::{(+/x)%#x}`, `add::{x+y}`, `hello::{"hello, ",x}`, `data::[3 1 2]` and `total::100` defined:

```q
q)h:hopen `:localhost:8888
q)h"1+1"
2
q)h"!5"
0 1 2 3 4
q)h"avg([1 2 3 4])"
2.5
q)h(`avg;1 2 3 4)
2.5
q)h(`add;3;4)
7
q)h(`hello;"q")
"hello, q"
q)h`total
100
q)h"data@<data"
1 2 3
```

A string sent to a KlongPy server is Klong, not q. Klong error text is returned as the q error, with the wire name first:

```q
q)@[h;"nosuch(1)";{"caught: ",x}]
"caught: name: undefined function: nosuch"
q)@[h;"1+";{"caught: ",x}]
"caught: parse: missing right operand for +"
q)@[h;(`nosuch;1);{"caught: ",x}]
"caught: name: symbol not found: nosuch"
```

At the default strict level, text that is not Klong can come back as a value instead of an error. `h"til 5"` is the name `til` followed by a 5 that the server drops, so the answer is the symbol `` `til `` and the server prints a warning. Start the server at strict level 1 or 2 (`kgpy --strict=2 -s 8888`) and the same request is answered with a parse error, as it is for any program that is not valid Klong. See [strict mode](strict.md).

Variables can be set with a symbol call, and asynchronous messages (`neg[h]`) are accepted and run in order:

```q
q)neg[h]"hits::hits+1"
q)neg[h]"hits::hits+1"
q)h"hits"
2
q)h(`set;`bar;7)
q)h`bar
7
```

The server had `hits::0` defined. A one-character string is a character atom in q, so `h"T"` sends the character `T` and not a string of length 1; the server answers with that character. Write `h"T "` or `h enlist"T"` to send the name. If `T` is a `klongpy.db` table on the server, the answer is a table (type 98):

```q
q)type h enlist"T"
98h
q)cols h"T "
`sym`px
q)(h"T ")`px
1.5 2.5 3.5
```

Hooks and the client's view of the connection are covered below. An open connection from a q client can be given a user name and password with `.srv.auth`; see [Credentials](ipc_capabilities.md#credentials-srvauth).

### Types across the connection

One table for both directions, as observed against PeachQ 0.92. The type numbers are q's (negative for an atom, positive for a vector). The full table, with the 32-bit and character details, is in [IPC Capabilities](ipc_capabilities.md#types-across-the-wire).

| Klong value | q type the q process sees | q value | Klong value `.cli` returns |
|---|---|---|---|
| integer | `-7h`, long atom | long, int, short or boolean atom | integer (`1b` is `1`) |
| float | `-9h`, float atom | float atom | float |
| char | char atom | char atom (`"a"`) | char (`0ca`) |
| string | `10h`, char vector | char vector (`"abc"`) | string |
| symbol | `-11h`, symbol atom | symbol (`` `abc ``) | symbol (`:abc`) |
| integer or float array | `7h` or `9h`, vector | long or float vector | integer or float array |
| list of symbols | `11h`, symbol vector | symbol vector | list of symbols |
| list of strings, mixed list | `0h`, general list | general list | list (`(1;"ab";`c)` is `[1 "ab" :c]`) |
| dictionary | `99h` | dictionary | dictionary |
| function | `100h`, lambda; its source text is sent | function other than a lambda | an opaque value, `:kdb-function`, that can be sent back |
| `:undefined` | `101h`, the generic null | generic null `(::)` | `:undefined` |
| | | table | `Table` with `klongpy.db` loaded, else a dictionary of column arrays |
| | | keyed table | a two-element list of the key table and the value table |
| | | date, time and other temporal types | the underlying number (`2000.01.01` is `0`) |
| | | `0N`, `0n`, `0w` | not translated: `-9223372036854775808`, `nan`, `inf` |
| a Python object, a connection handle | not sent: a type error | | |

<!-- check-docs: skip: needs a running q process (PeachQ 0.92) -->
```kgpy
?> h::.cli("localhost:8888")
remote[localhost:8888]:fn
?> h("tp:{type x}")
:undefined
?> h(:tp,,5)
-7
?> h(:tp,,2.5)
-9
?> h(:tp,,"abc")
10
?> h(:tp,,:abc)
-11
?> h(:tp,,[1 2 3])
7
?> h(:tp,,[1.5 2.5])
9
?> h(:tp,,[:a :b])
11
?> h(:tp,,["ab" "cd"])
0
?> h(:tp,,[1 "a"])
0
?> h(:tp,,:{[:a 1] [:b 2]})
99
?> h(:tp,,{x+1})
100
?> h(:tp,,1%0)
101
?> h("0N")
-9223372036854775808
?> h("""a""")
0ca
?> h("2000.01.01")
0
```

A function sent to q is its Klong source text, parsed by q. It runs only if that text is also valid q.

### Between two KlongPy interpreters

The same calls work between two KlongPy interpreters; see the example in [IPC Capabilities](ipc_capabilities.md#example).

### Hooks

q's `.z.po` and `.z.pc` run when a client opens and closes a connection, and `.z.pw` checks credentials. The Klong names are `.srv.o`, `.srv.c` and `.srv.auth`. `.srv.e` is called for an error that nobody is waiting for. There is no equivalent of q's `.z.pg` and `.z.ps`: a request is always evaluated as Klong source or as a call by name. The arguments and examples are in [Server Callbacks](ipc_capabilities.md#server-callbacks) and [Credentials](ipc_capabilities.md#credentials-srvauth).

`.timer` replaces `.z.ts`, as shown above. A call that gets no answer in 30 seconds fails with a `timed out` error; `.clit(h;seconds)` changes the limit.

## What is missing

Compared with a q process:

- No q-sql (`select`, `update`, `exec`, `delete`) and no `by` clause; use SQL strings over `klongpy.db` tables.
- No typed nulls, no boolean, byte, short, int, real or temporal types. Temporal values from q arrive as plain numbers and `0N` is not null.
- No keyed tables in the q sense (an `.index` on a table makes an insert update the matching row, which is not the same), no attributes, no enumerations, and no splayed or partitioned tables.
- No `.z.pg`, `.z.ps`, `.z.pi` or other message handlers; see Hooks above.
- No true asynchronous send from Klong (q's `neg[h]`). `.async(f;cb)` sends a request and calls `cb` with the answer when it arrives; a KlongPy server does accept asynchronous messages from q clients.
- Nested callbacks through a q process are not supported.
- A function is sent as its source text. A projection arrives opaque and cannot be called by the receiver.

## See also

- [Tutorial](tutorial.md) for the language from the first expression.
- [Klong for NumPy users](from-numpy.md) for the Python array side.
- [IPC Capabilities](ipc_capabilities.md), [IPC and timers in the native runtime](native/ipc.md), [Timers](timer.md), [Columnar tables and SQL](fast_columnar_database.md).
- [Language Additions](language-additions.md) for destructuring, named parameters and dictionaries.
- [Strict mode](strict.md) and [Lint](lint.md) for the checks that catch the q habits above.
