//! The profile document, docs/static-profile.md: the generator of the block between `<!-- BEGIN GENERATED -->` and
//! `<!-- END GENERATED -->`, ported from `klongpy/static/profile_doc.py` (`python -m klongpy.static.profile_doc`),
//! from the tables of `tables.rs`. `klong profile-doc [--check|--write] [PATH]`.
//!
//! The text around the block (`HEAD`, `TAIL`) is hand-written and kept here verbatim.
use std::io::Write;

use crate::tables::{self, Op};

pub const BEGIN: &str = "<!-- BEGIN GENERATED -->";
pub const END: &str = "<!-- END GENERATED -->";
pub const DEFAULT_PATH: &str = "docs/static-profile.md";

const USAGE: &str = "usage: klong profile-doc [--check [PATH] | --write [PATH]]";

/// A glyph as inline code that survives a Markdown table.
fn code(glyph: &str) -> String {
    format!(
        "<code>{}</code>",
        glyph.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;").replace('|', "&#124;")
    )
}

fn written(op: &Op) -> String {
    let g = code(op.glyph);
    match op.valence {
        "monad" => format!("{g}a"),
        "dyad" => format!("a{g}b"),
        "adverb1" => format!("f{g}a"),
        _ => format!("a f{g}b"),
    }
}

fn cell(text: &str) -> String {
    text.replace('|', "&#124;")
}

fn table(head: &[&str], rows: &[Vec<String>]) -> String {
    let mut out = vec![format!("| {} |", head.join(" | ")), format!("|{}", "---|".repeat(head.len()))];
    out.extend(rows.iter().map(|r| format!("| {} |", r.join(" | "))));
    out.join("\n")
}

fn op_rows(ops: &[&Op]) -> Vec<Vec<String>> {
    ops.iter()
        .map(|op| {
            let operands = if op.operands.is_empty() { "-".to_string() } else { op.operands.join(" , ") };
            vec![
                written(op),
                format!("`{}`", op.id),
                op.name.to_string(),
                operands,
                op.type_rule.unwrap_or("None").to_string(),
                op.shape_rule.unwrap_or("None").to_string(),
                cell(op.note),
            ]
        })
        .collect()
}

fn pairs(items: &[(&str, &str)]) -> Vec<Vec<String>> {
    items.iter().map(|(k, v)| vec![format!("`{k}`"), cell(v)]).collect()
}

/// The text between the markers (without them), ending in a newline.
pub fn generated_block() -> String {
    let allowed: Vec<&Op> = tables::OPTABLE.iter().filter(|o| o.allowed).collect();
    let rejected: Vec<&Op> = tables::OPTABLE.iter().filter(|o| !o.allowed).collect();
    let (adverbs, operators): (Vec<&Op>, Vec<&Op>) = allowed.iter().partition(|o| o.valence.starts_with("adverb"));
    let head = ["Written", "Row", "Name", "Operands", "Type rule", "Shape rule", "Note"];
    let mut parts: Vec<String> = Vec::new();
    parts.push(format!("### Operators\n\n{}", table(&head, &op_rows(&operators))));
    parts.push(format!(
        "### Adverbs\n\nThe verb `f` is an operator, the name of a function or extern, or a function literal.\n\n{}",
        table(&head, &op_rows(&adverbs))
    ));
    let forms: Vec<Vec<String>> =
        tables::FORMS.iter().map(|f| vec![cell(f.form), format!("`{}`", f.id), cell(f.note)]).collect();
    parts.push(format!("### Other forms\n\n{}", table(&["Form", "Row", "Rule"], &forms)));
    let rej: Vec<Vec<String>> = rejected
        .iter()
        .map(|op| vec![written(op), op.name.to_string(), format!("`{}`", op.rule.unwrap_or("None")), cell(op.note)])
        .collect();
    parts.push(format!(
        "### Operators and adverbs outside the profile\n\n{}",
        table(&["Written", "Name", "Error", "Why"], &rej)
    ));
    parts.push(format!("### Operand kinds\n\n{}", table(&["Kind", "Meaning"], &pairs(tables::OPERAND_KINDS))));
    parts.push(format!(
        "### Type rules\n\n{}",
        table(&["Rule", "Element type of the result"], &pairs(tables::TYPE_RULES))
    ));
    parts.push(format!("### Shape rules\n\n{}", table(&["Rule", "Shape of the result"], &pairs(tables::SHAPE_RULES))));
    // the profile's own rules: `RULES` also carries the streamable tier at its end
    let is_stream = |id: &str| tables::STREAM_RULES.iter().any(|s| s.id == id);
    let mut rule_rows: Vec<Vec<String>> = Vec::new();
    for (gid, title) in tables::GROUPS {
        for r in tables::RULES.iter().filter(|r| !is_stream(r.id) && r.group == *gid) {
            rule_rows.push(vec![format!("`{}`", r.id), title.to_string(), cell(r.title)]);
        }
    }
    parts.push(format!(
        "### Errors\n\nEvery rejection is a `static error` that ends in one of these rule ids.\n\n{}",
        table(&["Rule id", "Group", "Rejected"], &rule_rows)
    ));
    let stream_rows: Vec<Vec<String>> = tables::STREAM_RULES
        .iter()
        .map(|r| vec![format!("`{}`", r.id), cell(r.title), cell(r.hint.unwrap_or(""))])
        .collect();
    parts.push(format!(
        "### Rules for streaming kernels\n\n`kgpy check --streamable` reports these as the same kind of error, for a program \
that is inside the profile but would not take one record per clock cycle in an FPGA kernel \
(see [Extra rules for FPGA kernels](#extra-rules-for-fpga-kernels)).\n\n{}",
        table(&["Rule id", "Reported when", "Change"], &stream_rows)
    ));
    parts.join("\n\n") + "\n"
}

const HEAD: &str = r####"# The static profile

This page is the reference for the part of Klong that can be compiled ahead of
time to C or to an FPGA kernel: every type, every shape and every call target is
known before the program runs. It is for someone who writes such a program or
needs to know why the checker rejected one. A program inside the profile runs
unchanged under KlongPy and gives the same answers when compiled; the
[differences](#differences-from-klongpy) are few. For the tools around it see
[Compiling Klong to C](embedded.md); for FPGA kernels see
[Klong functions as FPGA hardware](hardware.md).

```
kgpy check prog.kg
```

checks a program against the profile without running it. Every problem is
reported, each with its position, the rule it breaks (the word in brackets)
and usually a hint:

```
prog.kg:3:5: static error: the count of n#a (Take) is not a compile-time constant: the result length would depend on run-time data [const-count]
  f::{y#x}
      ^^^
  hint: the count must be a literal, #a or ^a, or a global that no function assigns
```

The exit status is 1 if any file is outside the profile. `-v` lists what was
found (entries, externs, globals with their types, each function with the
argument types it was checked for); `--int64` checks integer constants
against a 64-bit `kg_int`. From Python: `klongpy.static.check_file(path)`
returns a report with `ok`, `diagnostics`, `functions`, `globals`, `externs`
and `entries`.

## Compile time and run time

A program is split by what kind of top-level statement it is.

| Top-level statement | When | Meaning |
|---|---|---|
| `name::{...}` or `name(p;q)::{...}` | run time | a function to compile, if reachable from an entry |
| `name::<any other expression>` | compile time | evaluated once by KlongPy on the host; the result becomes an initialised global |
| `.extern(...)` | compile time | declares a function the target provides |
| `.param(...)` | compile time | declares a global that is an input register of the compiled program |
| `.entry(...)` | compile time | marks a function as a C entry point and fixes its argument types |
| `.l("file")` | compile time | loads another file under the same rules |
| any other expression | error | top-level effects have no meaning on the target |

Compile-time expressions may use the whole language, including the library,
`.py` and file I/O. That is how tables are built: `coef::{...}'!64` runs on the
host and the 64 results are emitted as data. Only the resulting value has to
fit the profile.

Run-time functions must fit the profile. Only functions an entry reaches are
compiled and checked; a function used only by compile-time definitions may
use anything. A file with no `.entry` at all is checked as a library: every
function, with arguments of unknown type.

The program must parse at strict level 2 (`kgpy --strict=2`).

## Types

| Type | C type | Notes |
|---|---|---|
| `int` | `kg_int` = `int32_t` (default) or `int64_t` (`--int64`) | booleans are ints 0 and 1 |
| `float` | `kg_float` = `float` (default) or `double` (`--float64`) | |
| `fixed<W,I>` | `int32_t` (W up to 32) or `int64_t` | a W-bit fixed-point number with I integer bits; see [Fixed-point numbers](#fixed-point-numbers) |
| `int[d1,...,dk]`, `float[d1,...,dk]` | pointer to `kg_int`/`kg_float`, row-major | every `d` a compile-time positive integer, rank 1 to 4 |
| `str` | `const char *` | only as a literal argument to an extern |

A compile-time value is accepted as a global if it is an int, a float, or a
rectangular numeric array. Python ints outside the `kg_int` range, object
arrays, strings, symbols, dicts and functions are rejected with the position
of the definition.

A global that some compiled function assigns is state and lives in RAM; one
that no function assigns is a constant, lives in flash, and counts as a
compile-time constant wherever one is required (a take count, for example).

## What a run-time function may contain

| Group | Forms |
|---|---|
| Literals | integers, floats, numeric list literals, evaluated lists `[;a;b]` of same-shaped elements |
| Names | parameters (`x y z` or named), locals, head-destructured names, globals, other compiled functions, externs |
| Arithmetic dyads | `+ - * %` (`%` always gives float), `:%` integer divide, `!` remainder, `^` power, <code>&#124;</code> max, `&` min |
| Comparison dyads | `< > =`, giving int; `~` match on equal-shaped operands, giving int |
| Monads | `-` negate, `_` floor (gives int), `%` reciprocal, `~` not, `#` size and `^` shape (arrays only; `#` is the first dimension), `*` first, <code>&#124;</code> reverse, `+` transpose (rank 2), `!n` enumerate with constant `n`, `,` wrap as one-element list |
| Structure | `a@i` index with a scalar or an int array; `n#a` take and `n_a` drop with constant `n`, negative meaning from the end; take is rank 1 only; `a,b` join; `n:+a` rotate (rank 1); `s:^a` reshape with constant `s`; `a:=v,i` amend of a rank-1 array with one scalar value and either a scalar index or an index list of known length; other forms are rejected |
| Control | `:[c;a;b]` and its chain form `:[c1;a:|c2;b;else]`; statement sequences; `name::expr` to a local or a global |
| Adverbs | `f'a` each; `a f'b` each-2; `f/a` over; `a f/b` over with seed; `f\a` scan; `a f\b` scan with seed; `f:'a` each-pair; `n f:*a` iterate; `p f:~a` while |
| Calls | compiled functions and externs with all arguments supplied |

Rejected, each with a positioned error naming the rule:

- Anything whose result length depends on run-time data: `&a` where, `a?b` find, `?a` range, `=a` group, `<a` and `>a` grade (grade has a fixed length but is not in the profile, because it needs a sort routine in the runtime), `:#` split, `:_` cut, take or drop or reshape with a non-constant count.
- `f:~a` converge and the `\~` `\*` scan forms (result length unknown).
- Strings, chars, symbols, dicts, `:undefined` as run-time values.
- Functions as values: a function stored in a variable, returned, or passed as an argument other than directly as the verb of an adverb. Projections.
- Recursion, direct or mutual, including `.f`.
- `.E`, `.l`, `.py*`, channels and every other system function except `.extern`-declared names.
- A variable whose type or shape differs between two assignments, or a conditional whose branches have different shapes.
- Division of an int by zero when both are constants.

The tables below are generated from the checker's own data, so they cannot disagree with it. Each row and each
error has a stable id.

"####;

const TAIL: &str = r####"
## Fixed-point numbers

`fixed<W,I>` is a signed number of W bits, 2 to 64, of which I, 1 to W, are integer bits (the sign bit counts). It
is for programs whose arithmetic must be exact and cheap in hardware: an add is one cycle and a multiply is a
DSP block. It is written in a signature (`"fixed<32,16> -> fixed<64,32>"`) and made by two conversions that are
system functions evaluated by the compiler:

```
.fixed("fixed<32,16>";x)     :" an int, float or fixed-point value (or array) as that type "
.float(x)                    :" a fixed-point value (or array) as a float "
```

**Reference semantics.** The value of a number is `raw * 2^-F` where `raw` is an integer of W bits and
`F = W - I`. Every operation is defined on `raw`, and the interpreter stand-in, the C emitter and the hardware give
the same bits (the tests compare them over every pair of `fixed<8,4>` values and over random values of
`fixed<32,16>`, `fixed<64,32>` and others):

| Operation | Result `raw` |
|---|---|
| `a+b`, `a-b`, `-a` | the integer result, wrapped to W bits (two's complement) |
| `a*b` | `floor(a*b / 2^F)`, wrapped to W bits |
| `a%b` | `floor(a * 2^F / b)`, wrapped to W bits; `b` is not 0 (a zero divisor traps like an integer one); W is at most 32 |
| `a<b`, `a>b`, `a=b`, `a~b` | the comparison of the raw values, as an int |
| <code>a&#124;b</code>, `a&b` | the larger and the smaller raw value |
| `_a` | `floor(a / 2^F)` as an int (wrapped to `kg_int`) |
| `.fixed("fixed<W,I>";v)`, v an int | `v * 2^F`, wrapped |
| `.fixed("fixed<W,I>";v)`, v a float | `floor(v * 2^F)`, wrapped; nan, and anything beyond a 64-bit integer, give 0 |
| `.fixed("fixed<W,I>";v)`, v fixed-point | the fraction bits are shifted (a right shift floors), then wrapped |
| `.float(v)` | `raw * 2^-F`, exact up to 53 bits, rounded to `float` in compiled code |

Where this is written down once: `klongpy/static/fixed.py`. The C code (`klong_rt.h`) uses integers only: a
product of two operands of more than 32 bits is computed from 32-bit limbs, so `__int128` is not needed and a
32-bit target can use any W up to 64; division is limited to 32 bits because a 64-bit quotient needs a 128-bit
dividend. Hardware uses the same integer code (so that equality is by construction), not `ap_fixed`.

**What mixes.** Two operands of the same fixed-point type; or a fixed-point operand and an `int` that is a constant
that fits the integer part of the type (`x*2`, `x+1`). Everything else is an error with the rule `fixed`: two
different fixed-point types (convert one with `.fixed`), a float (write the float as `.fixed("fixed<..>";0.5)` or
convert the fixed-point value with `.float`), an int that is not a constant. `:%`, `!`, `^` and the monad `%` are
not defined on fixed-point numbers; `x|-x` is the absolute value. Reductions and scans of `+ | &` work and wrap like
the operators they apply; an int constant seed (`0+/x`) converts exactly.

Under KlongPy a program that uses `fixed<..>` runs through `klongpy.static.call_entry`, which evaluates the typed
program on the scaled integers (`klongpy/static/eval_ir.py`): the interpreter has no fixed-point type. A
fixed-point argument is a number or an array (or a `klongpy.static.fixed.FixedArray`), and the result is a float,
or with `raw=True` the scaled integers that compiled code works on.

## Parameters

```
.param("gain";"fixed<32,16>";0.5)      :" name, type, and the value it starts with (0 when omitted) "
```

declares a global that is an input register: the host writes it between runs and a run reads it. The type is a
scalar `int`, `float` or `fixed<W,I>`; the name is defined by `.param` itself and a compiled function may read it,
not assign it (rule `param`). It is not a compile-time constant, so a shape or a loop count cannot depend on it; a
parameter bounds a window only as "use the first p of a fixed maximum": `+/buf*(!8)<p`. Under KlongPy it is an
ordinary global; `klongpy.static.set_param(klong, "gain", 0.75)` sets it between calls (`call_entry` reads it at each
call). In C it is a field of the instance with a setter, `prog_set_gain(p, value)`; `prog_init` restores the
default. In an FPGA kernel it is a register or, with lanes, a table; see
[Extra rules for FPGA kernels](#extra-rules-for-fpga-kernels).

## Typing rules

Inference is forward and monomorphic. Every expression gets one type from the types of its parts.

- **Literals.** An integer literal is `int`, a literal with `.` or `e` is `float`. A list literal is an array of the common element type (any float makes it float).
- **Atomic dyads** (`+ - * :% ! | & < > =`). Scalar with scalar gives a scalar; scalar with array gives the array's shape; of two arrays the shorter shape must be a prefix of the longer, and the result has the longer shape. Otherwise it is a shape error at compile time. The element type is float if either operand is float; comparisons give int.
- **`%`** always float. **`:%`** needs int operands and gives int. **`!`** gives int for two ints and float otherwise.
- **`^`** gives int only when both operands are int and the exponent is a non-negative constant; otherwise float. (KlongPy returns an int whenever the result happens to be whole; the profile cannot depend on values.)
- **`_`** gives int. **`#a`** and **`^a`** are compile-time constants.
- **Index.** `a@i` with scalar `i` drops the first axis. With an int array `i` of shape `S`, the result shape is `S` followed by the rest of `a`'s shape.
- **Conditional.** The condition must be a scalar. Branch types are unified (int with float gives float); shapes must be equal.
- **Each.** `f'a` calls `f` on each item along the first axis; the result shape is `[n]` followed by `f`'s result shape.
- **Each-pair.** `f:'a` gives `[n-1]` followed by `f`'s result shape.
- **Constant indices** are checked at compile time; one out of range is an error.
- **Over and scan.** The accumulator's type is the fixed point of `acc = f(acc, item)`, starting from the first item (or the seed); if it does not stabilise after one promotion step (int to float), it is an error. Scan's result is `[n]` (or `[n+1]` with a seed, matching Klong) followed by the accumulator's shape. So `,/a` (flatten) is outside the profile: write `[n]:^a`.
- **Iterate and while.** The state's type and shape must be the same before and after the body.
- **Functions** are specialised per distinct tuple of argument types and shapes. An entry's argument types come from its `.entry` signature; everything else follows from call sites. A constant passed as an argument is not a constant inside the function.
- **Globals** take the type of their compile-time value. Assigning a different type or shape is an error. An int global assigned a float is an error, not a promotion (state must not change type), and so is a float global assigned an int.
- **Numeric promotion at assignment** to a local: the local's type is the join of all assignments to it (int with float gives float); shapes must agree.
- **Compile-time constants** are literals, `#a` and `^a`, globals that no compiled function assigns, and `+ - * :% !`, max and min of constants.

## Things to know when writing numeric Klong

- A leading `-` negates the whole expression to its right: `-2#a` is `-(2#a)`, `-7!2` is `-(7!2)`. Write `(-2)#a`.
- Negative counts mean "from the end": `(-2)#[1 2 3]` is `[2 3]`, `(-1)_[1 2 3]` is `[1 2]`, `(-1):+a` rotates left.
- Take cycles: `5#[1 2]` is `[1 2 1 2 1]`.
- Scan with a seed returns `n+1` items: `0+\[1 2 3]` is `[0 1 3 6]`.
- Joining ints with floats gives floats: `[1 2],[3 4.5]`.
- Lists need brackets (`[1 2 3]`, not `1 2 3`), and a name inside brackets is a symbol, not the variable: write `[;a;b]`.
- `a!b` and `a:%b` are C's `%` and `/`: `(-7):%2` is -3, `(-7)!2` is -1 and `7!-2` is 1. Float remainder is `fmod` (`7.5!2` is 1.5). `_` rounds toward minus infinity (`_(-3.5)` is -4).
- Evaluation order: the right operand of a dyad is evaluated before the left; call arguments and the elements of `[;a;b]` left to right. Compiled code keeps this order.
- Names contain letters, digits and `.` but no underscore. A Klong name maps to a C identifier by replacing `.` with `_`, so the extern `read.adc` is C `read_adc`.
- A function reads its parameters, its locals and the globals. It does not see the locals of its caller (KlongPy's scoping is dynamic; the profile's is not). A function literal used as a verb may read the locals of the function it is written in, but not assign them.

## Host functions and entry points

```
.extern(name; signature)             :" a function the target provides; module defaults to env "
.extern(name; signature; module)
.entry(name; signature)              :" a function the host calls "
```

The name, the signature and the module are string literals. A signature is
`params -> result`: `params` is a space-separated list of types (possibly
empty) and `result` is a type or `void`. Types: `int`, `float`, `int[N]`,
`float[N,M]`, and `str` (extern parameters only). Examples: `"int -> float"`,
`"float[64] -> float"`, `"int float -> void"`, `"-> int"` (called as `name()`).

- In C an extern is a prototype under the mapped name (`read.adc` is `read_adc`) and an entry is `prog_<name>`. There is no WebAssembly build yet; it is meant to make an extern an import from `module` and an entry an export.
- An extern name must not map to a C keyword or start with `kg.`; an entry cannot be called `init`, `t`, `size` or `storage.t` (`prog_init` and the others are generated).
- `.entry` needs the function to be defined before it and to take as many arguments as the signature has parameters. A function may have several entries with different shapes.
- A `str` parameter takes a string literal.
- The result of a `void` extern cannot be used as a value.

### Running a static program under KlongPy

The same file runs in KlongPy with Python functions standing in for the
externs:

```python
from klongpy import KlongInterpreter

klong = KlongInterpreter(strict=2)
klong.externs["read.adc"] = lambda channel: 0.3
klong.externs["write.pwm"] = lambda channel, duty: 0
klong('.l("examples/embedded/pid.kg")')
klong("step(0.31)")                      # 0.12005
```

or, in one call each:

```python
from klongpy.static import load_program, call_entry

klong = load_program("examples/embedded/pid.kg",
                     externs={"read.adc": lambda channel: 0.3, "write.pwm": lambda channel, duty: 0})
call_entry(klong, "step", 0.31)          # 0.12005, checked against the .entry signature
```

- `klong.externs` maps an extern name to a Python callable. It is looked up at each call, so a stand-in can be bound before or after the `.extern` line runs and replaced at any time. `klong.externs.declared` and `klong.externs.entries` hold what the program declared.
- A stand-in gets one positional argument per parameter: a Python `int` for `int`, a `float` for `float` (an int argument is converted), a `str` for `str`, a numpy `int64` or `float64` array of the declared shape for an array. It returns a value of the result type; for `void` the result is ignored and the call evaluates to `:undefined`.
- Arguments and result are checked against the signature on every call: a float where an int is declared is a type error, a wrong array shape a shape error, a wrong number of arguments an arity error, each naming the extern and positioned at the call. Calling an extern that has no stand-in is a system error naming it. An exception raised by a stand-in is a python error.
- `.entry` under KlongPy validates the declaration and records it; the function is called as usual. `call_entry` calls it the way compiled code is called, with arguments and result checked against the signature.

## Differences from KlongPy

A compiled program differs from the interpreter in a few specified ways: integer width, float width, division by zero, indexing out of range, and shapes that the checker rejects at compile time. They are in one table on the page [Compiling Klong to C](embedded.md#differences-from-the-interpreter).

## What is checked without running the program

`kgpy check` never evaluates anything. The type of a global is
known to it when the definition is a literal (`pos::0`,
`buf::[0.0 0.0 0.0 0.0]`); the value of any other compile-time definition
(`taps::raw%+/raw`) exists only once the host has evaluated it. Such a global
is treated as numeric of unknown type and shape, the rules that need a type
or shape are applied only where it is known, and the summary says how many
globals are in that state. Rules about forms (a rejected operator, a string,
a function used as a value, recursion, a system function, a count that is
not constant) never need types and are always decided. The compiler
evaluates the definitions and then applies the same rules with everything
known.

## Reference programs

`examples/embedded/` holds three programs that are inside the profile, each
with the results KlongPy gives (`.expect`):

- `mavg.kg`: a moving average over the last four samples; the window and its write position are state.
- `pid.kg`: one step of a PID controller that reads a sensor and writes an output through two externs.
- `fir.kg`: a 16-tap FIR filter over a block of 256 samples; the taps and the index table are built on the host.

## Extra rules for FPGA kernels

These rules apply only to programs built as FPGA kernels (`kgpy fpga source`, `kgpy fpga build`); C does not need them. [Klong functions as FPGA hardware](hardware.md) shows the flow. A kernel is one of two kinds. A *streaming kernel*, the default, takes one record per clock cycle when the program allows it. A *wrapped kernel* (`--wrapped`) runs the program's C unchanged once per record and is slower.

### Parameters in a kernel

A `.param` is a register of the kernel (an `s_axilite` argument). With `--lanes N` the kernel holds N copies of the program, the N values of every parameter are one table in card memory that the kernel reads when it starts, and `.fpga.set` fills the table. The host runtime cannot write an array to a register, which is why a table is used.

### A summary entry

An entry reads or changes the program's state, and every entry of a program works on the same state: under KlongPy the same globals, in C the same instance. A hardware kernel is one entry, so that sharing needs one more rule, for the case where a run feeds many records and reads a small result at the end (a strategy replayed over a tick history, where a result per tick would cost more than the computation).

A `void` entry with arguments (the driver) and one entry without arguments that returns a value (the reader) form a *summary pair*. The kernel of the driver makes its n calls, writes nothing for them, then calls the reader once and writes that one result record (with `--lanes N`, one per lane, lane first). n may be 0: the kernel then reads the state as it is. The reader is not a kernel of its own. Only a streaming kernel can have a summary pair; a wrapped kernel rejects it. A `void` entry without a reader cannot be a kernel, and a reader without a driver is an entry that takes no argument. Both are `[hls]` errors that say so.

A complete pair:

<!-- check-docs: link examples -->
```console
$ cat > sum_demo.kg <<'EOF'
> total::0.0
> count::0
> step::{total::total+x;count::count+1;0}
> report::{total,count}
> .entry("step";"float -> void")
> .entry("report";"-> float[2]")
> EOF
$ kgpy check sum_demo.kg
sum_demo.kg: inside the static profile (2 functions, 2 entries, 0 externs, 2 globals)
```

Every other target does the same by hand: `call_entry(klong, "step", x)` for each record, then `call_entry(klong, "report")`, and the C emitter has `prog_step(p, x)` and `prog_report(p, out)` on one instance. `examples/hardware/crossover.kg` is a worked pair.

### Checking that a kernel can stream

`kgpy check --streamable prog.kg` answers a second question about a program that is inside the profile: can a hardware kernel take one record per clock cycle? It reports, with the span of the construct and what to change, each of:

* a loop with a data-dependent bound (`stream-loop`);
* state that is read and written back through a multi-cycle operation, a float operation or a division by a value that is not a constant (`stream-state`);
* a data-dependent index into an array of more than 1024 items (`stream-table`);
* an entry that does more than 4096 element operations per call and is not a sliding window (`stream-block`).

A sliding window is `+/taps*x@idx` with the index table `k+!w`. The kernel keeps the last `w` samples in a shift register and makes one output per input word.
"####;

/// The whole page.
pub fn render_page() -> String {
    format!("{HEAD}{BEGIN}\n{}{END}\n{TAIL}", generated_block())
}

/// (text up to and including BEGIN, the block between the markers, the text from END on)
fn split(text: &str) -> Option<(&str, &str, &str)> {
    let a = text.find(BEGIN)?;
    let b = text.find(END)?;
    if b < a {
        return None;
    }
    Some((&text[..a + BEGIN.len()], &text[a + BEGIN.len()..b], &text[b..]))
}

/// True if the generated block of a page text is up to date.
pub fn check_text(text: &str) -> bool {
    match split(text) {
        Some((_, block, _)) => block.trim_matches('\n') == generated_block().trim_matches('\n'),
        None => false,
    }
}

/// `klong profile-doc [--check [PATH] | --write [PATH]]`, as `python -m klongpy.static.profile_doc`:
/// no mode prints the page; `--check` exits 0 if the block of PATH is up to date, 1 if not, 2 if PATH is missing;
/// `--write` refreshes the block of PATH (or creates the page). A bad command line exits 2.
pub fn main(args: &[String], out: &mut dyn Write, err: &mut dyn Write) -> i32 {
    let mut argv: Vec<&str> = args.iter().map(String::as_str).collect();
    let mut mode = None;
    if matches!(argv.first(), Some(&"--check") | Some(&"--write")) {
        mode = Some(argv.remove(0));
    }
    if (mode.is_none() && !argv.is_empty()) || argv.len() > 1 {
        let _ = writeln!(err, "{USAGE}");
        return 2;
    }
    let mode = match mode {
        None => {
            let _ = out.write_all(render_page().as_bytes());
            return 0;
        }
        Some(m) => m,
    };
    let path = argv.first().copied().unwrap_or(DEFAULT_PATH);
    let text = std::fs::read_to_string(path).ok();
    if mode == "--check" {
        return match text {
            None => {
                let _ = writeln!(err, "{path}: not found; create it with: klong profile-doc --write {path}");
                2
            }
            Some(t) if check_text(&t) => 0,
            Some(_) => {
                let _ = writeln!(err, "{path}: the generated block is out of date; run: klong profile-doc --write {path}");
                1
            }
        };
    }
    let new = match text.as_deref().and_then(split) {
        None => render_page(),
        Some((before, _, after)) => format!("{before}\n{}{after}", generated_block()),
    };
    match std::fs::write(path, new) {
        Ok(()) => 0,
        Err(e) => {
            let _ = writeln!(err, "{path}: {e}");
            2
        }
    }
}
