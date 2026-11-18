# Tutorial

This page takes you from the first expression to functions, adverbs, dictionaries and files. It assumes you can program in some language and have never seen Klong. Every example is a transcript you can type at the REPL (`kgpy`); lines that start with `?> ` are what you type, and the line below is what the interpreter prints. The output shown is real.

All the code on this page is accepted at strict level 2, the strictest setting (see [Strict mode](strict.md)). The few examples that show a mistake say so and run at the default level.

Klong is a small array language by Nils Holm. KlongPy is a Python implementation of it that runs the same programs and adds a few conveniences, which are marked "KlongPy adds" below. Each one is shown beside the classic form, so a program written for any Klong runs here unchanged. The full reference is the [Klong manual](https://t3x.org/klong/klong-ref.txt.html); [Operators](operators.md) lists every operator.

## 1. Reading right to left

Klong has no operator precedence. An expression is read from right to left, and parentheses group:

<!-- check-docs: strict 2 -->
```kgpy
?> 5+3*2
11
?> (5+3)*2
16
?> 2*3+4
14
```

`5+3*2` is `5+(3*2)`: the `3*2` on the right is done first, then `5+6`. The rule has no exceptions, so you never have to remember a table of precedence. It also means that `2*3+4` is `2*(3+4)`, not `(2*3)+4`.

Read a longer expression the same way, one token at a time from the right. This one is the average of a list:

<!-- check-docs: continue -->
```kgpy
?> (+/[1 2 3])%#[1 2 3]
2.0
```

- `#[1 2 3]` is the number of elements, 3.
- `+/[1 2 3]` adds the elements: `/` is the adverb "over" and it makes `+` fold over the list, giving 6.
- `%` divides the left by the right: 6 over 3. `%` always gives a real number, so the result is `2.0`.

## 2. Values

### Numbers

Integers are 64 bits wide. Reals are double precision floats. A number is a real as soon as it has a decimal point or an exponent.

<!-- check-docs: strict 2 -->
```kgpy
?> 42
42
?> 2.5
2.5
?> 1e3
1000.0
?> 7%2
3.5
?> 7:%2
3
```

`%` is real division and `:%` is integer division.

An integer that does not fit in 64 bits wraps around when both operands are ordinary integers. An operand that is itself larger than 64 bits makes the arithmetic exact, so you can compute with large numbers. `^` of two integer scalars with a non-negative exponent is exact too (`2^64` above); a list operand wraps in 64 bits instead:

<!-- check-docs: continue -->
```kgpy
?> 9223372036854775807+1
-9223372036854775808
?> 2^64
18446744073709551616
?> (2^64)+1
18446744073709551617
```

### Lists

A list is written in square brackets with spaces between the elements. A list that mixes integers and reals becomes a list of reals. Lists can hold lists.

<!-- check-docs: strict 2 -->
```kgpy
?> [1 2 3]
[1 2 3]
?> [1 2.5 3]
[1.0 2.5 3.0]
?> [[1 2] [3 4]]
[[1 2] [3 4]]
?> []
[]
```

A list literal holds what is written in it, token by token; it does not evaluate anything. To build a list from expressions, put a `;` right after the opening bracket and separate the expressions with `;`:

<!-- check-docs: continue -->
```kgpy
?> [;1+1;3*3]
[2 9]
?> a::10
10
?> [;a;a+1]
[10 11]
```

### Strings, characters and symbols

A string is text in double quotes. A quote inside a string is written twice. A single character is written `0c` followed by the character. A symbol is a name with a colon in front:

<!-- check-docs: strict 2 -->
```kgpy
?> "hello"
"hello"
?> 0cx
0cx
?> :foo
:foo
?> "say ""hi"""
"say ""hi"""
```

A string is a list of characters, so list operators work on it:

<!-- check-docs: continue -->
```kgpy
?> #"hello"
5
?> "hello"@1
0ce
?> |"abc"
"cba"
?> "abc","def"
"abcdef"
?> 3#"abcdef"
"abc"
```

Symbols are names used as data, for instance as keys. A symbol stays a symbol; it is never looked up as a variable:

<!-- check-docs: continue -->
```kgpy
?> foo::5
5
?> :foo
:foo
?> {x}'[:foo :bar]
[:foo :bar]
?> :foo=:foo
1
```

The empty string `""`, the empty list `[]` and `0` are false in a condition; everything else is true (see [Conditionals](#7-conditionals)).

## 3. Operators by example

Every operator is a single glyph and it has two meanings: one with a single argument on its right (a monad) and one with two (a dyad). The position decides which. `-` is negate in `-a` and subtract in `a-3`:

<!-- check-docs: strict 2 -->
```kgpy
?> a::5
5
?> -a
-5
?> a-3
2
```

A few that you will use all the time:

<!-- check-docs: continue -->
```kgpy
?> !5
[0 1 2 3 4]
?> #[7 8 9]
3
?> [1 2],[3 4]
[1 2 3 4]
?> [10 20 30]@1
20
?> [10 20 30]@[0 2]
[10 30]
?> 2#[7 8 9]
[7 8]
?> 1_[7 8 9]
[8 9]
?> *[7 8 9]
7
?> |[1 2 3]
[3 2 1]
?> ?[3 1 3 2 1]
[3 1 2]
?> &[1 0 1 1]
[0 2 3]
?> <[30 10 20]
[1 2 0]
?> [30 10 20]@<[30 10 20]
[10 20 30]
?> 7|3
7
?> 7&3
3
?> 10!3
1
?> 2^10
1024
```

- `!n` counts from 0, `#a` is the length, `a,b` joins, `a@i` indexes from 0 (a list of indices gives a list).
- `n#a` takes `n` elements and `n_a` drops them. `*a` is the first element and `|a` reverses.
- `?a` is the distinct values in order of first appearance. `&a` is the positions that are true (the "where" of a 0/1 list).
- `<a` is the positions that would sort `a`. It is not the sorted list; to sort, index with it: `a@<a`.
- `a|b` is the larger and `a&b` the smaller. `a!b` is the remainder of `a` by `b`.

Arithmetic and comparison go through lists element by element, and a single number is applied to every element:

<!-- check-docs: continue -->
```kgpy
?> [1 2 3]+[10 20 30]
[11 22 33]
?> 2*[1 2 3]
[2 4 6]
?> [1 2 3]=[1 2 4]
[1 1 0]
?> +/[1 2 3]=[1 2 4]
2
?> [1 2 3]~[1 2 3]
1
```

`=` compares each pair of elements and gives a list of 0 and 1. `~` (match) compares the whole values and gives one result.

Lists of lists work the same way, and have a few operators of their own: `^a` is the shape, `+a` the transpose and `[2 3]:^a` reshapes.

<!-- check-docs: continue -->
```kgpy
?> m::[2 3]:^[1 2 3 4 5 6]
[[1 2 3] [4 5 6]]
?> ^m
[2 3]
?> +m
[[1 4] [2 5] [3 6]]
```

To change one element, use amend. `a:=v,i` gives a copy of `a` with `v` at index `i`:

<!-- check-docs: continue -->
```kgpy
?> [1 2 3 4 5]:=9,0
[9 2 3 4 5]
```

The operators are listed with their exact rules in [Operators](operators.md); `]h` at the REPL shows the help for one of them (see [The REPL](#12-the-repl)).

## 4. Functions

A function is code in braces. Inside it, `x`, `y` and `z` are the first, second and third argument. A function is called with its arguments in parentheses, separated by `;`:

<!-- check-docs: strict 2 -->
```kgpy
?> f::{x*2}
:monad
?> f(4)
8
?> g::{x+y}
:dyad
?> g(3;4)
7
```

`::` defines a name. The REPL shows a function as `:nilad`, `:monad`, `:dyad` or `:triad` for the number of arguments it takes. That number is the highest of `x`, `y` and `z` that the body uses, so `{y}` is a dyad and `{-x}` is a monad. A function takes at most three arguments this way; for more, see [named parameters](#named-parameters).

A function that takes no arguments is a nilad. It is called with empty parentheses:

<!-- check-docs: continue -->
```kgpy
?> count::0
0
?> tick::{count::count+1}
:nilad
?> tick()
1
?> tick()
2
```

Calling a function with fewer arguments than it takes gives a new function that waits for the rest. This is called a projection. Leave the last arguments off, or leave any argument empty and keep its `;`:

<!-- check-docs: continue -->
```kgpy
?> add::{x+y}
:dyad
?> inc::add(1)
:monad
?> inc(5)
6
?> dec::add(;-1)
:monad
?> dec(5)
4
```

(The REPL shows a projection by the number of arguments it still takes.) A projection made inside a function keeps the values its arguments had there, so `{add(x)}'[1 2 3]` is a list of three different projections. Calling a function with more arguments than it takes is an error at strict level 2, and at levels 0 and 1 the extra arguments are dropped:

<!-- check-docs: continue -->
```kgpy
?> {x}(1;2)
error: the function takes 1 argument, got 2
```

Inside a function, `.f` is the function itself, which is how you write recursion without a name:

<!-- check-docs: strict 2 -->
```kgpy
?> fact::{:[x<2;1;x*.f(x-1)]}
:monad
?> fact(10)
3628800
?> gcd::{:[y=0;x;.f(y;x!y)]}
:dyad
?> gcd(48;18)
6
```

(`:[c;a;b]` is the conditional, described in [section 7](#7-conditionals).) A function can be applied to every element of a list with the adverb `'`:

<!-- check-docs: continue -->
```kgpy
?> fact'[1 2 3 4 5]
[1 2 6 24 120]
```

Adverbs are the subject of [section 6](#6-adverbs).

## 5. Locals, globals, destructuring

An assignment to a name that exists, anywhere, changes that name. This is how a function updates a global:

<!-- check-docs: strict 2 -->
```kgpy
?> total::0
0
?> add::{total::total+x}
:monad
?> add(5)
5
?> add(7)
12
?> total
12
```

A function that needs working variables of its own declares them in a list at the start of its body, `{[a b]; ...}`. They exist only during the call and do not touch a global of the same name:

<!-- check-docs: continue -->
```kgpy
?> t::100
100
?> area::{[t];t::x*y;t+1}
:dyad
?> area(3;4)
13
?> t
100
```

Without the declaration, a function cannot create a name by assigning to it. At strict level 1 and 2 this is an error, so the typo or the forgotten declaration is found at once:

<!-- check-docs: continue -->
```kgpy
?> bad::{u::x*2;u+1}
:monad
?> bad(5)
error: undefined variable: u
```

At the default level (0) the same function runs, and `u` is a hidden local that disappears when the call ends, so it is not visible from outside. Declare your locals. [Strict mode](strict.md) lists what each level rejects.

### Destructuring

KlongPy adds assignment of a list to several names at once. The classic form indexes the list one name at a time:

<!-- check-docs: strict 2 -->
```kgpy
?> v::[10 20 30]
[10 20 30]
?> a::v@0;b::v@1;c::v@2
30
?> [p q r]::v
[10 20 30]
?> p+q+r
60
```

Both forms give the same names. The new form needs as many names as the list has elements, and says so when they differ:

<!-- check-docs: continue -->
```kgpy
?> [p q]::v
error: cannot destructure 3 values into 2 names
```

When it is the first statement of a function, the names are the function's locals and need no declaration:

<!-- check-docs: continue -->
```kgpy
?> pv1::{[r n a];r::x@0;n::x@1;a::x@2;a%(1+r)^n}
:monad
?> pv2::{[r n a]::x;a%(1+r)^n}
:monad
?> pv1([0.05 10 100])
61.39132535407592
?> pv2([0.05 10 100])
61.39132535407592
```

### Named parameters

A Klong function sees its arguments as `x`, `y` and `z`, and three is the most. KlongPy adds a definition form that names the parameters and has no limit on how many there are. The classic way to pass more than three values is one list and the destructuring above. With names, you write the call shape on the left of `::`:

<!-- check-docs: continue -->
```kgpy
?> pv3(r;n;a)::{a%(1+r)^n}
:triad
?> pv3(0.05;10;100)
61.39132535407592
?> sum4(a;b;c;d)::{a+b+c+d}
:fn/4
?> sum4(1;2;3;4)
10
```

The parameters are locals. Inside such a function `x`, `y` and `z` are not available, because the parameters replace them. (The REPL shows a function of four or more parameters as `:fn/N`, here `:fn/4`.) Fewer arguments give a projection, as before, and more are an error:

<!-- check-docs: continue -->
```kgpy
?> p4::sum4(1;;3;4)
:monad
?> p4(2)
10
?> sum4(1;2;3;4;5)
error: sum4 takes 4 arguments, got 5
```

[Language Additions](language-additions.md) gives the complete rules for both, including the head form in more detail and how to call named-parameter functions from Python.

## 6. Adverbs

An adverb is a glyph written right after a verb (a function or an operator, with no space between them). It changes how the verb is applied. The most common are each (`'`), over (`/`) and scan (`\`):

<!-- check-docs: strict 2 -->
```kgpy
?> {x*x}'[1 2 3]
[1 4 9]
?> +/[1 2 3 4]
10
?> +\[1 2 3 4]
[1 3 6 10]
?> {x+y}/[1 2 3]
6
?> |/[3 9 2]
9
?> ,/[[1 2] [3 4]]
[1 2 3 4]
```

- `f'a` applies `f` to each element of `a`.
- `f/a` folds `f` over `a`: `+/` is the sum, `|/` the maximum, `,/` joins a list of lists.
- `f\a` is the same fold, keeping every intermediate result.
- A left argument is the starting value of the fold: `10{x+y}/[1 2 3]` is 16.

<!-- check-docs: continue -->
```kgpy
?> 10{x+y}/[1 2 3]
16
```

With a list on the left, `'` takes one element from each side, so it zips:

<!-- check-docs: continue -->
```kgpy
?> [1 2 3]{x+y}'[10 20 30]
[11 22 33]
```

`:'` is each-pair. It applies a function to each pair of neighbours:

<!-- check-docs: continue -->
```kgpy
?> {x,y}:'[1 2 3 4]
[[1 2] [2 3] [3 4]]
?> {y-x}:'[1 4 9 16]
[3 5 7]
```

`@'` hands the function an `[index element]` pair, like `enumerate`:

<!-- check-docs: continue -->
```kgpy
?> {x@0}@'[10 20 30]
[0 1 2]
?> {(x@0)*(x@1)}@'[10 20 30]
[0 20 60]
```

### Loops

Klong has no loop statement. Repetition is an adverb. `:*` (iterate) applies a function a given number of times, and `\*` does the same and keeps each step:

<!-- check-docs: continue -->
```kgpy
?> 3{x*2}:*1
8
?> 3{x*2}\*1
[1 2 4 8]
```

`:~` (converge) applies a function until its result stops changing. With a condition function written before the body, `{condition}{body}:~start` is a while loop: it runs the body while the condition holds.

<!-- check-docs: continue -->
```kgpy
?> {x<100}{x*2}:~1
128
?> {(x+2%x)%2}:~1
1.4142156862745097
?> {_x%2}\~100
[100 50 25 12 6 3 1 0]
```

The first is a while loop (double until it reaches 100 or more). The second is Newton's iteration for the square root of 2. Converge stops when two successive values agree within a small tolerance (relative 1e-5), so the result agrees with the square root of 2, 1.4142135623730951, to five digits and no further. The third halves with integer division and keeps each step (`\~` is scan-converge).

For a longer example of the adverbs at work, `klongpy/lib/csv.kg` is a complete CSV reader written with them.

## 7. Conditionals

`:[c;a;b]` is "if `c` then `a` else `b`". Only the branch that is chosen is evaluated:

<!-- check-docs: strict 2 -->
```kgpy
?> :[1;"yes";"no"]
"yes"
?> :[0;"yes";"no"]
"no"
?> abs::{:[x<0;-x;x]}
:monad
?> abs(-7)
7
```

The condition is false if it is 0 (also `0.0`) or empty (`[]` or `""`). Everything else is true:

<!-- check-docs: continue -->
```kgpy
?> :[[];"yes";"no"]
"no"
?> :[""; "yes";"no"]
"no"
?> :[:foo;"yes";"no"]
"yes"
```

To test several cases in a row, write `:|` in place of the `;` before the next test. `:[c1;a:|c2;b;d]` means `:[c1;a;:[c2;b;d]]`:

<!-- check-docs: continue -->
```kgpy
?> sign::{:[x<0;"neg":|x=0;"zero";"pos"]}
:monad
?> sign'[-5 0 7]
["neg" "zero" "pos"]
```

The classic form nests the conditionals, and means the same:

<!-- check-docs: continue -->
```kgpy
?> sign2::{:[x<0;"neg";:[x=0;"zero";"pos"]]}
:monad
?> sign2'[-5 0 7]
["neg" "zero" "pos"]
```

## 8. Dictionaries

A dictionary maps keys to values. The literal is `:{[key value] [key value] ...}`. `d?k` looks a key up, and gives `:undefined` for a key that is not there. `:_` tests for it:

<!-- check-docs: strict 2 -->
```kgpy
?> d:::{["a" 1] ["b" 2]}
:{["a" 1] ["b" 2]}
?> d?"a"
1
?> d?"zz"
:undefined
?> :_d?"zz"
1
?> :_d?"a"
0
```

(`d:::{...}` is `d::` followed by the literal `:{...}`.) Keys can be strings, numbers or symbols. Adding an entry uses `,` with a `[key value]` pair. `,v` is the one-element list `[v]` and `k,[v]` is the pair `[k v]`, so `d,k,,v` adds the entry (read it from the right):

<!-- check-docs: continue -->
```kgpy
?> d::d,"c",,3
:{["a" 1] ["b" 2] ["c" 3]}
?> d::d,"a",,100
:{["a" 100] ["b" 2] ["c" 3]}
?> #d
3
```

Setting a key that exists replaces its value. An empty dictionary is `:{}`.

Keys that are written in a plain literal are data, as in a list literal, so a name there is a symbol and not its value:

<!-- check-docs: continue -->
```kgpy
?> n::5
5
?> :{["n" n]}
:{["n" :n]}
```

The classic way around it is to build the dictionary from an empty one:

<!-- check-docs: continue -->
```kgpy
?> e:::{}
:{}
?> e::e,"n",,n
:{["n" 5]}
```

KlongPy adds a literal whose keys and values are expressions, evaluated each time. Put a `;` right after `:{` and write `key;value;key;value`:

<!-- check-docs: continue -->
```kgpy
?> :{;"n";n;"double";n*2}
:{["n" 5] ["double" 10]}
?> mk::{:{;"n";x;"sq";x*x}}
:monad
?> mk(3)
:{["n" 3] ["sq" 9]}
```

It also adds `d@k`, which is the same as `d?k` (and `d@[k1 k2]` for a list of keys), and joining two dictionaries merges them, the right one winning:

<!-- check-docs: continue -->
```kgpy
?> cfg:::{;"name";"x";"size";12}
:{["name" "x"] ["size" 12]}
?> cfg@"size"
12
?> cfg@["size" "name"]
[12 "x"]
?> cfg@"zzz"
:undefined
?> cfg::cfg,:{["size" 99] ["depth" 3]}
:{["name" "x"] ["size" 99] ["depth" 3]}
```

One trap comes from the pair rule. `,` with a dictionary on the right and a list of two elements on the left joins the pair into the dictionary, so a dictionary does not append to a two-element list. Wrap it with `,,` to append:

<!-- check-docs: continue -->
```kgpy
?> l::[1 2]
[1 2]
?> l,:{["k" 1]}
:{["k" 1] [1 2]}
?> l,,:{["k" 1]}
[1 2 :{["k" 1]}]
```

`kgpy --lint` reports the first form as W020 (see [Lint](lint.md)). The remaining rules, including nested dictionaries, are in [Language Additions](language-additions.md).

## 9. Printing and strings

`.p` prints a value and a newline. `.d` prints without the newline, and strings are printed without their quotes. Both return their argument, so the REPL shows it after the printed text:

<!-- check-docs: strict 2 -->
```kgpy
?> .p("hello")
hello
"hello"
?> .d("a");.d("b");.p("")
ab
""
```

`$` turns a value into a string. To build a message, join strings with `,`:

<!-- check-docs: continue -->
```kgpy
?> $42
"42"
?> $[1 2 3]
["1" "2" "3"]
?> "n=",$42
"n=42"
```

`n$s` pads a value to a width. A positive width pads on the right, a negative one on the left, and a width of the form `5.2` also fixes the number of decimals:

<!-- check-docs: continue -->
```kgpy
?> 10$123
"123       "
?> (-10)$123
"       123"
?> 5.2$3.0
"    3.00"
```

KlongPy adds `.fmt`, which fills a template. `{}` takes the next argument, `{0}` and `{1}` take them by position, `{:8}` pads like `8$`, and `{{` and `}}` are literal braces. The arguments are a list, which is easiest to build with `[;...]` so that the items are expressions. `.pf` is `.fmt` followed by a print:

<!-- check-docs: continue -->
```kgpy
?> .fmt("{} + {} = {}";[;1;2;1+2])
"1 + 2 = 3"
?> .fmt("{1} before {0}";["a" "b"])
"b before a"
?> .fmt("{:6}|{:-6}|";[7 8])
"7     |     8|"
?> .fmt("{:5.2}";3.14159)
"    3.14"
?> .pf("total: {}";42)
total: 42
"total: 42"
```

The classic way to do the first one is `$` and `,`:

<!-- check-docs: continue -->
```kgpy
?> ($1),$" + ",($2),$" = ",$1+2
"1 + 2 = 3"
```

(Read it from the right: `$1+2` is `$(1+2)`, so `"3"`.) A single argument can be given without a list. To pass one list as the only argument, wrap it with `,`:

<!-- check-docs: continue -->
```kgpy
?> .fmt("{}";42)
"42"
?> .fmt("{}";,[1 2 3])
"[1 2 3]"
```

Placeholders and arguments must agree in number, and `{}` and `{0}` cannot be mixed:

<!-- check-docs: continue -->
```kgpy
?> .fmt("{} {}";[1])
error: .fmt: 2 placeholder(s) but 1 argument(s) in template '{} {}'
```

## 10. Files, modules, comments

A comment starts with `:"` and ends at the next `"`. It can be a whole line or follow code:

<!-- check-docs: strict 2 -->
```kgpy
?> 1+2 :" this is a comment"
3
```

`.comment("END")` ignores everything up to a line that is `END`. At the REPL the lines of such a block are typed after a `..` prompt (more in [The REPL](#12-the-repl)):

<!-- check-docs: continue -->
```kgpy
?> .comment("END")
.. none of this is evaluated: 1+
.. END
None
?> 2+2
4
```

`.l("file.kg")` loads and runs a Klong file. The name is looked up in the current directory, then in the directories of the `KLONGPATH` environment variable, and `.kg` may be left off. `KLONGPATH` defaults to `.:lib`. The standard library files (`math`, `csv`, `set`, `nstat` and others) are in the `klongpy/lib` directory of the package, which `kgpy` adds to `KLONGPATH` itself, so `.l("nstat")` works in the REPL and in `kgpy -e` without setting anything. A `KlongInterpreter` created from Python does not add it: there, set `KLONGPATH` to that directory, or copy the file you need, before you load one. The next example uses a file `lib.kg` with these two lines, and loads it:

```console
$ cat > lib.kg <<'EOF'
> shout.up::{x,"!"}
> limit::99
> EOF
$ cat lib.kg
shout.up::{x,"!"}
limit::99
$ kgpy -e '.l("lib.kg");.p(limit);shout.up("hey")'
99
"hey!"
```

`.l` returns the value of the last statement of the file.

A library brackets its definitions with `.module(:name)` and `.module(0)`. Inside the module its names refer to the module's own variables, so a program that later assigns the same name does not change what the library's functions use. The names are not hidden, though: after the module is closed they can still be used by their plain names (see [Modules](operators.md#modules)). By convention a library names what it exports `name.function`, as `csv.split` and `csv.read` in `klongpy/lib/csv.kg` do, so that its names do not clash with yours:

```kgpy
?> .module(:geom);geom.sq::{x*x};geom.cube::{x*geom.sq(x)};.module(0)
None
?> geom.cube(3)
27
```

## 11. Python interop

KlongPy is a Python package, so Klong code and Python code share one interpreter. From Klong, `.py("module")` imports a Python module and defines its public functions as Klong functions, and `.pyf("module";"name")` imports one name:

<!-- check-docs: strict 2 -->
```kgpy
?> .py("math")
1
?> sqrt(64)
8.0
?> fsum(!100)
4950.0
?> .pyf("numpy";["min" "max"])
1
?> max([1 5 2])
5
```

Python functions are called with the Klong rules: at most three arguments, and they are named `x`, `y` and `z`.

From Python, the interpreter is an object. Calling it runs Klong code, and indexing it reads and writes the variables. The values are in the comments.

```python
from klongpy import KlongInterpreter

klong = KlongInterpreter()
klong['f'] = lambda x, y, z: x*1000 + y - z    # a Python function becomes a Klong function
klong('f(3; 10; 20)')                          # 2990

klong("g::{(x*1000) + y - z}")
klong['g'](3, 10, 20)                          # 2990: a Klong function becomes a Python callable

klong("vol(a;b;c;d)::{a+b+c+d}")
klong['vol'](1, 2, d=4, c=3)                   # 10: named-parameter functions take keywords

klong['data'] = [1, 2, 3]
klong('+/data')                                # 6
klong('2^64')                                  # 18446744073709551616
```

[Python Integration](python_integration.md) covers modules of your own, NumPy arrays and more.

## 12. The REPL

`kgpy` starts the REPL. It reads one entry, evaluates it and prints the value. `]h` shows help for an operator, adverb or system function, by glyph or by name. This is a shell session, not a transcript:

```
$ printf ']h .p\n]h Over\n' | kgpy
?> .p(x)               [Print]

    Pretty-print the object "x" (like Display) and then print a
    newline sequence. Backend arrays are converted to numpy for cleaner output.
    Use .bkp() for raw backend-specific print.
?> f/a                 [Over]

    If "a" is a list, fold "f" over "a":
    ...
```

The REPL keeps the last printed value in `it`:

<!-- check-docs: strict 2 -->
```kgpy
?> 1+2
3
?> it*2
6
?> it+it
12
```

When an entry leaves a bracket, a string or a comment open, the REPL prompts with `..` and reads on, then evaluates the lines as one entry. That makes a long definition easy to type:

<!-- check-docs: continue -->
```kgpy
?> avg::{
.. (+/x)%#x
.. }
:monad
?> avg([2 4 9])
5.0
```

An empty line at the `..` prompt cancels the entry. [The REPL](repl.md) lists the other `]` commands, the history and completion, and [Errors](errors.md) explains the messages and exit codes.

## 13. Strict mode and common mistakes

Klong has a few places where text that looks wrong is accepted and means something else. KlongPy keeps that behaviour at its default level, so old programs run, and offers two stricter levels that reject the doubtful input. Select a level with `kgpy --strict=N`, with `KlongInterpreter(strict=N)` or with `.strict(N)`:

- Level 0 (the default) runs everything, and warns about input that it ignores.
- Level 1 makes that input a syntax error, and makes an assignment or a read of a name that is not defined inside a function a name error.
- Level 2 also makes the read of any undefined name, and extra arguments in a call, errors.

[Strict mode](strict.md) lists each rule. The static checker [`kgpy --lint`](lint.md) finds the same kinds of mistakes in a file without running it, and some more. The traps that catch most newcomers:

**Names cannot contain `_`.** `_` is the operator Drop, so `my_var` is `my _ var`:

```kgpy
?> my_var::5
error: cannot apply _ (Drop) to symbol and integer
```

At level 2 the same line stops earlier, on the undefined name `my`. Use `myVar` or `my.var`. (A dot is part of a name, which is how modules name things.)

**A leading minus negates everything to its right.** A minus sign joins its number only inside a list literal; anywhere else `-` is the operator, and `-1+2` is `-(1+2)`. Put parentheses around a negative number in an expression:

<!-- check-docs: strict 2 -->
```kgpy
?> -1+2
-3
?> (-1)+2
1
?> [-1 2]
[-1 2]
```

**A name in a list is a symbol.** `[a b]` holds the symbols `:a` and `:b`. To list the values, evaluate with `[;a;b]`:

<!-- check-docs: continue -->
```kgpy
?> a::1
1
?> b::2
2
?> [a b]
[:a :b]
?> [;a;b]
[1 2]
```

**A space before `(` drops the call.** `f (1)` is the function `f` followed by something left over. At level 0 the program runs on, with a warning, as if the `(1)` were not there; at level 1 and 2 it is a syntax error:

<!-- check-docs: continue -->
```kgpy
?> f::{x+1}
:monad
?> f(1)
2
?> f (1)
error: unexpected '(' after the end of the expression
```

**A function that only uses `y` takes two arguments.** `{y}` has arity 2, so `f::{y};f(5)` does not return 5 or an error; it is a projection that waits for `y`, and calling it with one more argument completes it:

<!-- check-docs: continue -->
```kgpy
?> g::{y}
:dyad
?> g(5)
:monad
?> g(5;6)
6
```

**Indexing uses `@`.** Square brackets build lists. `q[1]` does not index (see [Operators](operators.md#indexing-is-not-q1)), write `q@1`.

**Assigning inside a function.** Without a declaration, the assignment is an error at level 1 and 2 (section 5). At level 0 it makes a local that is gone after the call: `b` below is still undefined, and an undefined name evaluates to its own symbol, `:b`. This session starts at level 0 and then turns strict level 1 on:

```kgpy
?> g::{b::5}
:nilad
?> g()
5
?> b
:b
?> .strict(1)
1
?> h::{c::5}
:nilad
?> h()
error: undefined variable: c
```

For new code, use level 2 (`kgpy --strict=2`) and run `kgpy --lint` on the files. Both report the mistakes above before the program runs or gives a wrong answer. See [Strict mode](strict.md#which-level-to-use).

## Where to go next

- [Operators](operators.md): every operator, adverb and system function by name, with edge cases.
- [Language Additions](language-additions.md): the complete rules for the additions marked "KlongPy adds".
- [System functions](system-functions.md): files, channels, timers, IPC, the database and more.
- [Examples](examples.md): short programs.
- Coming from another tool? [From NumPy](from-numpy.md) and [From kdb+/q](from-kdb.md) map their idioms to Klong.
- [Start Here](getting-started.md) and [What's new in v2](whats-new-v2.md) for what v2 adds, and [Migrating to v2](migrating-to-v2.md) for existing programs.
- [Native runtime](native/index.md): the same language on small devices, with no Python.
