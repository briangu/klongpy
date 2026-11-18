# Compiling Klong to C

Status: works for the static profile, a subset of the language. It is tested with desktop C compilers and compile-checked for bare Arm and x86 targets. Three examples have run compiled on two Raspberry Pi Picos, where they agree with the interpreter ([on a Pico](#on-a-pico)); no other microcontroller has run it. For hardware status see the [native runtime table](native/index.md#status-what-has-and-has-not-been-proven).

A small part of Klong, the [static profile](static-profile.md), can be compiled ahead of time to C99 with no heap, no recursion and no C library. A program written in it runs unchanged under KlongPy, which is how it is developed and tested, and the generated C does the same work on a target with no interpreter. The target is a microcontroller or any other place where a small, fixed-size, dependency-free function is wanted: a filter, a controller, a state machine over a few sensors.

This page walks through the whole path with two programs from `examples/embedded/`: check the program, run it under the interpreter, emit C, build it for the host and for a bare target, and call it from a `main`. The commands need a C compiler.

The rules of the profile (which operators, adverbs and forms are allowed, the types, the shape rules, the error ids) are on the [static profile page](static-profile.md). This page covers the tools around it.

!!! note "Scope"
    This page is about the generated C only. The HLS C++ that `kgpy fpga source` writes from the same programs is experimental; see [Klong functions as FPGA hardware](hardware.md). The native (non-Python) runtime is a separate Rust implementation of the language, described under [Native runtime](native/index.md). What was and was not compiled is listed [below](#what-has-not-been-done).

## The two programs

`mavg.kg` is a moving average over the last four samples. The window and the write position are state that lives between calls. The last line declares the function the host calls and its type.

```
buf::[0.0 0.0 0.0 0.0]
pos::0

step::{buf::buf:=x,pos;pos::(pos+1)!#buf;(+/buf)%#buf}

.entry("step";"float -> float")
```

`pid.kg` is one step of a PID controller. The target provides two functions, declared with `.extern`: `read.adc(channel)` and `write.pwm(channel;duty)`.

```
kp::2.0
ki::0.5
kd::0.1
dt::0.01
lo::0.0
hi::1.0

integ::0.0
prev::0.0

.extern("read.adc";"int -> float")
.extern("write.pwm";"int float -> int")

clamp::{(x|lo)&hi}

step::{[e u];e::x-read.adc(0);integ::clamp(integ+ki*e*dt);u::clamp((kp*e)+integ+kd*(e-prev)%dt);prev::e;write.pwm(0;u);u}

.entry("step";"float -> float")
```

(Both files also carry a comment at the top. The listings omit it.) A global defined with `::` at the top level is either a constant or, if a function assigns it, state: `buf`, `pos`, `integ` and `prev` are state, and become fields of a C struct.

## Check a program

`kgpy check` checks files against the profile without running them. It exits 0 when every file is inside, 1 when one is outside and 2 when one cannot be read.

<!-- check-docs: link examples -->
```console
$ kgpy check examples/embedded/mavg.kg examples/embedded/pid.kg; echo $?
examples/embedded/mavg.kg: inside the static profile (1 function, 1 entry, 0 externs, 2 globals)
examples/embedded/pid.kg: inside the static profile (2 functions, 1 entry, 2 externs, 8 globals)
0
```

A program outside the profile gets every problem reported, each with its position, the rule it breaks (the word in brackets) and a hint:

<!-- check-docs: continue -->
```console
$ cat > bad.kg <<'EOF'
> f::{y#x}
> .entry("f";"float[4] int -> float[4]")
> EOF
$ kgpy check bad.kg; echo $?
bad.kg:1:5: static error: the count of n#a (Take) is not a compile-time constant: the result length would depend on run-time data [const-count]
  f::{y#x}
      ^^^
  hint: the count must be a literal, #a or ^a, or a global that no function assigns
1 static error in 1 file
1
```

`-v` lists what the checker found: the entries, the externs, the globals with their types, and each function with the argument types it was checked for. `--int64` checks integer constants against a 64-bit `kg_int`.

## Run it under the interpreter

The same file runs in KlongPy. `klongpy.static.load_program` loads it at strict level 2 and `call_entry` calls an entry the way compiled code is called: the arguments and the result are checked against the `.entry` signature. An extern is a Python function bound by name; the stand-in is looked up at each call.

<!-- check-docs: continue -->
```console
$ cat > run_mavg.py <<'EOF'
> from klongpy.static import load_program, call_entry
> klong = load_program("examples/embedded/mavg.kg")
> for x in (4.0, 8.0, 12.0, 16.0, 20.0):
>     print("%g" % call_entry(klong, "step", x))
> EOF
$ python run_mavg.py
1
3
6
10
14
$ cat > run_pid.py <<'EOF'
> from klongpy.static import load_program, call_entry
> klong = load_program("examples/embedded/pid.kg", externs={
>     "read.adc": lambda channel: 0.3,
>     "write.pwm": lambda channel, duty: 0,
> })
> for _ in range(3):
>     print("%g" % call_entry(klong, "step", 0.31))
> EOF
$ python run_pid.py
0.12005
0.0201
0.02015
```

A stand-in gets one positional argument per parameter (a Python `int`, a `float`, a `str`, or a numpy array of the declared shape) and returns a value of the declared result type. A wrong type, shape or number of arguments is an error that names the extern. Calling an extern that has no stand-in is a system error. The details are on the [static profile page](static-profile.md#running-a-static-program-under-klongpy).

These results are the ones the compiled code must reproduce. They are the reference side of the differential tests described [below](#differences-from-the-interpreter).

## Emit C

`kgpy compile c FILE... -o DIR` writes `DIR/<prefix>.c`, `DIR/<prefix>.h` and a copy of the runtime header `DIR/klong_rt.h`, and prints a size report. The prefix is the file's stem made into a C identifier; `--prefix NAME` chooses another.

<!-- check-docs: continue -->
```console
$ kgpy compile c examples/embedded/mavg.kg -o mavg-c
examples/embedded/mavg.kg: wrote mavg-c/mavg.c, mavg.h and klong_rt.h
size report for mavg (kg_int 32 bits, kg_float 32 bits)
  constants          16 bytes (static const data; flash on a microcontroller)
  sizeof(mavg_t)  20 bytes (20 state, 0 array locals and temporaries)
  deepest call chain 2 C function(s): mavg_step > kg_step__1
  stack estimate     68 bytes (scalars, loop counters and call frames along that chain; externs not counted)
  C source           43 lines
$ ls mavg-c
klong_rt.h
mavg.c
mavg.h
```

The generated header declares a visible instance struct, the entry prototypes and, for a program that uses externs, the prototypes of the functions the target must define:

<!-- check-docs: continue -->
```console
$ grep -n '^struct\|^    kg_\|^void\|^size_t\|^kg_' mavg-c/mavg.h
20:struct mavg {
21:    kg_float g_buf[4]; /* state buf: float[4] */
22:    kg_int g_pos; /* state pos: int */
29:void mavg_init(mavg_t *p);
31:size_t mavg_size(void);
34:kg_float mavg_step(mavg_t *p, kg_float x); /* entry step: float -> float */
```

All storage is in the instance. The caller decides where it lives (a static, a heap block, external RAM) and may keep several instances; `<P>_init` sets the state to its initial values, `<P>_size()` returns `sizeof` for a host that allocates it itself, and `<P>_SIZE` is the same as a macro. Array arguments are `const T *` and an array result is a trailing `T *out`. The names `init`, `t`, `size` and `storage.t` cannot be entry names. One instance must not be entered from two threads at once.

The size report says what the program costs. Constants are `static const` tables (flash on a microcontroller). The instance is the state plus the storage of array temporaries. The stack figure is an estimate along the deepest call chain and does not count the externs.

### The `pid_t` clash

The type for `pid.kg` would be `pid_t`, which POSIX already defines, so a hosted build that includes `<sys/types.h>` before the header fails. Choose another prefix:

<!-- check-docs: continue -->
```console
$ kgpy compile c examples/embedded/pid.kg -o pid-c --prefix pidc
examples/embedded/pid.kg: wrote pid-c/pidc.c, pidc.h and klong_rt.h
size report for pidc (kg_int 32 bits, kg_float 32 bits)
  constants          0 bytes (static const data; flash on a microcontroller)
  sizeof(pidc_t)  8 bytes (8 state, 0 array locals and temporaries)
  deepest call chain 3 C function(s): pidc_step > kg_step__1 > kg_clamp__0
  stack estimate     108 bytes (scalars, loop counters and call frames along that chain; externs not counted)
  C source           55 lines
$ grep -n 'pidc_t\|read_adc\|write_pwm' pid-c/pidc.h
24:typedef struct pidc pidc_t;
29:void pidc_init(pidc_t *p);
30:/* sizeof(pidc_t), for a host that allocates the instance itself. */
34:kg_float pidc_step(pidc_t *p, kg_float x); /* entry step: float -> float */
37:kg_float read_adc(kg_int a0); /* Klong name read.adc: int -> float (module env) */
38:kg_int write_pwm(kg_int a0, kg_float a1); /* Klong name write.pwm: int float -> int (module env) */
```

An extern appears in the header as a prototype under its mapped name: `read.adc` is `read_adc`, and the comment gives the Klong name, the signature and the module.

### Other options

`--int64` and `--float64` choose the widths of `kg_int` and `kg_float` (the defaults are `int32_t` and `float`). `--line-directives` adds `#line` directives so that a debugger or a compiler error points at the Klong source. `--dump-ir` prints the typed intermediate form of the program. The compiler stops with the checker's messages if the program is outside the profile.

## Build it

The runtime is one header, `klong_rt.h`. The generated `.c` includes it and defines `KG_RT_IMPLEMENTATION` before doing so, which is what makes it define the weak default `kg_trap` (and, freestanding, the weak `memcpy` and `memset`) in that one translation unit; another file that includes the header gets only declarations.

### On the host

A minimal `main` for `mavg`: allocate the instance statically, initialise it, call the entry.

<!-- check-docs: continue; requires-cmd clang -->
```console
$ cat > main_mavg.c <<'EOF'
> #include <stdio.h>
> #include "mavg.h"
> static mavg_t mavg;
> int main(void) {
>     const float in[] = {4.0f, 8.0f, 12.0f, 16.0f, 20.0f};
>     mavg_init(&mavg);
>     for (int i = 0; i < 5; i++)
>         printf("%g\n", (double)mavg_step(&mavg, in[i]));
>     printf("sizeof %zu, mavg_size() %zu\n", sizeof(mavg_t), mavg_size());
>     return 0;
> }
> EOF
$ clang -std=c99 -Wall -Wextra -Werror -pedantic -Imavg-c main_mavg.c mavg-c/mavg.c -o mavg_host
$ ./mavg_host
1
3
6
10
14
sizeof 20, mavg_size() 20
```

The five values are those of the interpreter above, and `sizeof` is the `sizeof(mavg_t)` in the size report. The generated code compiles without a warning under `-std=c99 -Wall -Wextra -Werror -pedantic`.

### Externs and the linker

An extern is an undefined symbol in the object file. The program that uses it must define it with the prototype in the header. Here the externs return constants, as the Python stand-ins did:

<!-- check-docs: continue; requires-cmd clang -->
```console
$ cat > main_pid.c <<'EOF'
> #include <stdio.h>
> #include "pidc.h"
> static pidc_t pid;
> kg_float read_adc(kg_int channel) { (void)channel; return 0.3f; }
> kg_int write_pwm(kg_int channel, kg_float duty) { (void)channel; (void)duty; return 0; }
> int main(void) {
>     pidc_init(&pid);
>     for (int i = 0; i < 3; i++)
>         printf("%g\n", (double)pidc_step(&pid, 0.31f));
>     return 0;
> }
> EOF
$ clang -std=c99 -Wall -Wextra -Werror -pedantic -Ipid-c main_pid.c pid-c/pidc.c -o pid_host
$ ./pid_host
0.12005
0.0201
0.02015
```

Without the two definitions the link fails with undefined `read_adc` and `write_pwm`. The module name of an extern (`env` by default; the third argument of `.extern`) is not used in C. It is the import module of the WebAssembly build that does not exist yet.

### Freestanding

With `-ffreestanding -nostdlib -DKG_FREESTANDING` the generated file needs no C library and includes no header beyond `<stdint.h>` and `<stddef.h>`. The only symbols it leaves undefined are the externs the program declared. `nm -u` shows them:

<!-- check-docs: continue; requires-cmd clang nm -->
```console
$ clang -std=c99 -Wall -Wextra -Werror -pedantic -ffreestanding -nostdlib -DKG_FREESTANDING -Imavg-c -c mavg-c/mavg.c -o mavg_free.o
$ nm -u mavg_free.o | wc -l | tr -d ' '
0
$ clang -std=c99 -Wall -Wextra -Werror -pedantic -ffreestanding -nostdlib -DKG_FREESTANDING -Ipid-c -c pid-c/pidc.c -o pidc_free.o
$ nm -u pidc_free.o | sed -e 's/^[ U]*//' -e 's/^_//'
read_adc
write_pwm
```

`mavg` has no undefined symbol and `pidc` has its two externs. Two things to know:

- `memcpy` and `memset` are defined in the generated file, weakly, because a compiler may emit calls to them even under `-nostdlib`. A real C library, or your own definitions, take precedence. On a target without a hardware instruction for an operation (64-bit division on a 32-bit core, soft-float arithmetic) the compiler may also call its runtime library (compiler-rt or libgcc); those helpers are the other possible undefined symbols.
- `kg_pow` is computed inside the header in this mode, to about 1e-6 relative error for `float` and 1e-13 for `double`; with a C library it is `powf` or `pow`. Only a program that uses float `^` needs `-lm` on a hosted build.

## Policies

The runtime has a few switches, all set with `-D` when the generated file is compiled.

| Switch | Default | Meaning |
|---|---|---|
| `KG_BOUNDS` | `1` | An index out of range calls `kg_trap(KG_TRAP_BOUNDS, line)`. `0` removes the check; an out-of-range index is then undefined behaviour. A constant index is checked at compile time either way. A negative index counts from the end, as in KlongPy |
| `KG_DIV0` | `ieee` | `ieee`: float `x%y` and `x!y` with a zero divisor give inf or nan. `trap`: they call `kg_trap(KG_TRAP_DIV0, line)`. Integer division by zero (`:%`, `!`) always traps |
| `KG_FREESTANDING` | off | No C library (see above) |
| `KG_NO_MEM_BUILTINS` | off | Do not define `memcpy` and `memset` even when freestanding |
| `KG_NO_DEFAULT_TRAP` | off | Leave out the default `kg_trap`, so that a missing definition is a link error |

Integer arithmetic wraps at the width of `kg_int`, computed unsigned so that there is no undefined behaviour. `--int64` and `--float64` on the command line choose widths; the header refuses to be included with the other widths defined.

`kg_trap(int code, int line)` is called with `KG_TRAP_BOUNDS` (1) or `KG_TRAP_DIV0` (2) and the line of the Klong source. The default is weak and loops forever. A program that defines its own, in any file other than the generated one, replaces it. If it returns, the operation that trapped gives 0 (an index out of range reads or writes item 0), so memory stays safe, and the values computed after a trap are unspecified.

A small program shows the policies. It reads a constant table, divides, and adds one to an int:

<!-- check-docs: continue; requires-cmd clang -->
```console
$ cat > policy.kg <<'EOF'
> tab::[1.0 2.0 3.0]
> at::{tab@x}
> ratio::{x%y}
> inc::{x+1}
> .entry("at";"int -> float")
> .entry("ratio";"float float -> float")
> .entry("inc";"int -> int")
> EOF
$ kgpy compile c policy.kg -o policy-c
policy.kg: wrote policy-c/policy.c, policy.h and klong_rt.h
size report for policy (kg_int 32 bits, kg_float 32 bits)
  constants          12 bytes (static const data; flash on a microcontroller)
  sizeof(policy_t)  4 bytes (0 state, 0 array locals and temporaries)
  deepest call chain 2 C function(s): policy_ratio > kg_ratio__1
  stack estimate     64 bytes (scalars, loop counters and call frames along that chain; externs not counted)
  C source           58 lines
$ cat > main_policy.c <<'EOF'
> #include <stdio.h>
> #include <stdlib.h>
> #include "policy.h"
> static policy_t p;
> void kg_trap(int code, int line) {
>     printf("trap %d at policy.kg line %d\n", code, line);
>     exit(1);
> }
> int main(void) {
>     policy_init(&p);
>     printf("at(-1)        %g\n", (double)policy_at(&p, -1));
>     printf("ratio(1;0)    %g\n", (double)policy_ratio(&p, 1.0f, 0.0f));
>     printf("inc(2^31-1)   %d\n", (int)policy_inc(&p, 2147483647));
>     printf("at(3)         ");
>     fflush(stdout);
>     printf("%g\n", (double)policy_at(&p, 3));
>     return 0;
> }
> EOF
$ clang -std=c99 -Wall -Wextra -Werror -pedantic -Ipolicy-c main_policy.c policy-c/policy.c -o policy_ieee
$ ./policy_ieee
at(-1)        3
ratio(1;0)    inf
inc(2^31-1)   -2147483648
at(3)         trap 1 at policy.kg line 2
$ clang -std=c99 -Wall -Wextra -Werror -pedantic -DKG_DIV0=trap -Ipolicy-c main_policy.c policy-c/policy.c -o policy_trap
$ ./policy_trap
at(-1)        3
trap 2 at policy.kg line 3
```

The same program under the interpreter:

<!-- check-docs: continue -->
```console
$ cat > run_policy.py <<'EOF'
> from klongpy.static import load_program, call_entry
> klong = load_program("policy.kg")
> print(call_entry(klong, "at", -1))
> print(klong("ratio(1.0;0.0)"))
> print(call_entry(klong, "inc", 2147483647))
> try:
>     call_entry(klong, "at", 3)
> except Exception as e:
>     print(type(e).__name__ + ":", e)
> EOF
$ python run_policy.py
3.0
:undefined
2147483648
KlongRangeError: index 3 out of range for list of 3
```

The index and the integer wrap are the two cases where compiled code and the interpreter differ in a way the policies control. `call_entry` rejects `ratio(1.0;0.0)` because `:undefined` is not a float, so the example reads the value through `klong(...)`.

### Widths

The default build is `int32_t` and `float`, the usual choice for a 32-bit microcontroller. `--int64 --float64` builds for 64-bit integers and doubles, in which a float result is bit for bit the interpreter's.

<!-- check-docs: continue; requires-cmd clang -->
```console
$ kgpy compile c examples/embedded/mavg.kg -o mavg-c64 --int64 --float64
examples/embedded/mavg.kg: wrote mavg-c64/mavg.c, mavg.h and klong_rt.h
size report for mavg (kg_int 64 bits, kg_float 64 bits)
  constants          32 bytes (static const data; flash on a microcontroller)
  sizeof(mavg_t)  40 bytes (40 state, 0 array locals and temporaries)
  deepest call chain 2 C function(s): mavg_step > kg_step__1
  stack estimate     104 bytes (scalars, loop counters and call frames along that chain; externs not counted)
  C source           43 lines
$ clang -std=c99 -Wall -Wextra -Werror -pedantic -Imavg-c64 main_mavg.c mavg-c64/mavg.c -o mavg_host64
$ ./mavg_host64
1
3
6
10
14
sizeof 40, mavg_size() 40
```

## Differences from the interpreter

A compiled program gives the interpreter's answers except in the ways below. Each is specified, and each is pinned by a test in `tests/static/test_differential.py`, which builds the generated C into a shared library and compares it with the interpreter bit for bit on random and edge inputs. The checker turns the language-level rows (the ones marked "compile error") into errors, so that no program inside the profile meets them. The [static profile](static-profile.md) has the rules.

| Case | KlongPy | Compiled |
|---|---|---|
| Integer overflow | 64-bit integers wrap, as numpy int64 does; `2147483647+1` is 2147483648 | wraps at the width of `kg_int`: `-2147483648` with `int32_t` |
| Float width | doubles | `kg_float`: `float` by default. The tests compare a float32 result bit for bit when every operation is exact, and to 1e-5 otherwise. `--float64` is bit for bit except `^` (1e-15) |
| `x%0` | `:undefined`, per element for arrays | IEEE (inf or nan) by default, `kg_trap` under `KG_DIV0=trap` |
| Integer `:%` or `!` by zero | `:undefined` | `kg_trap` always; a constant zero divisor is a compile error |
| Index out of range | a range error; a negative index counts from the end | `kg_trap(KG_TRAP_BOUNDS, line)`; negative indices count from the end as in KlongPy; undefined behaviour with `KG_BOUNDS=0`; a constant index out of range is a compile error |
| `_x` beyond the range of `kg_int`, or of nan | saturates to the int64 range | saturates to the range of `kg_int`; nan gives 0 |
| `x~y` of one nan with itself | 1 | 0 |
| `+/` of floats | numpy adds pairwise from eight items and answers `+0.0` for a sum of negative zeros | adds left to right from the first item. `{x+y}/a` and `+\a` agree in both |
| `n{f}:*x` with a negative `n` | an error | zero iterations |
| `kg_pow` freestanding | libm | about 1e-6 relative (float), 1e-13 (double) |
| Comparison `~` of two scalars | tolerant (relative 1e-5, absolute 1e-8); exact for two arrays | the same |
| `^` result type | int when the result is whole (`2.0^2` is the int 4) | int only for int operands and a non-negative constant exponent, else float |
| `:%` of floats | accepted (`7.0:%2` is 3) | compile error: floor the operands first |
| `a~b` with different shapes | 0 | compile error |
| `a f'b` with different lengths | stops at the shorter | compile error |
| `n:+a` on a matrix | rotates the flattened data | compile error (rank 1 only) |
| Over on one item (`f/[5]`) | the item itself, whatever `f` returns | the item converted to the accumulator's type |
| A conditional with an int and a float branch | the type of the branch taken | float (the value is the same) |
| A callee reading its caller's locals | works (dynamic scope) | compile error (`undefined-name`) |

Programs written so that no value comes near these cases (small integers, no zero divisors, indices in range) give the same answers everywhere. The tests also include random programs: a seeded generator writes functions over floats, ints, short arrays, state globals, constant tables, locals, conditionals, every adverb, amend, index, take, drop, rotate, reshape and join, with values chosen to be exact in single precision, and compares the compiled result with the interpreter's for thousands of calls.

Some of the rows, run in the interpreter:

```kgpy
?> 2147483647+1
2147483648
?> 1.0%0.0
:undefined
?> _1e30
9223372036854775807
?> n::-3
-3
?> n{x+1}:*0
error: :*: count must not be negative, got -3
?> 1.0~1.000001
1
?> [1.0]~[1.000001]
0
```

## Hardware (experimental)

`kgpy fpga source` writes the same programs as HLS C++ for an FPGA kernel, and `.fpga` calls the built kernel from Klong. This is experimental and has run on one Alveo U250 card. The flow, its settings and the measured results are on [Klong functions as FPGA hardware](hardware.md).

## On a Pico

The Pico firmware of the [native runtime](native/pico-bench.md) links the C compiled from `examples/embedded/mavg.kg`, `pid.kg` and `fir.kg` beside the interpreter, as the functions `.c.mavg`, `.c.pid` and `.c.fir`. `.c.check()` runs each example through the interpreter and through the compiled code on the same inputs, on the board, and gives one row per example: whether they agree, the number of values, the interpreter's time and the compiled code's time in microseconds, and the largest relative difference in parts per billion.

| Example | Values | Interpreter | Compiled C | Largest difference |
|---|---|---|---|---|
| `mavg.kg`, one step per value | 64 | 21.2 ms | 0.198 ms | 0 |
| `pid.kg`, one step per value | 32 | 34.7 ms | 0.344 ms | 101 parts per billion |
| `fir.kg`, 256 samples through 16 taps | 241 | does not fit | 6.4 ms | 216 parts per billion |

Conditions: two Raspberry Pi Picos (RP2040, Cortex-M0+ at 125 MHz, no FPU), 2026-11-18, one firmware image; both boards agree on every value and their times are within 2%. The compiled code is about 100 times faster than the interpreter on these two step functions. The differences are float32 in the compiled code against float64 in the interpreter. The interpreter cannot load `fir.kg` on this board: its 256-sample arrays do not fit the heap beside the firmware (`memory error: out of memory`), so the compiled FIR is compared with values computed by KlongPy on a desktop and stored in the image. The compiled FIR needs no heap.

## What has not been done

- The C that this tool generates has run on one kind of microcontroller, the RP2040 of a Raspberry Pi Pico (Cortex-M0+, no FPU), and only for the three examples above. It has not run on a Cortex-M4 or M7, RISC-V or Xtensa device, or in an emulator; the other tests run it as a shared library on the host. The HLS C++ of `kgpy fpga source` has run on one Alveo U250 ([Klong functions as FPGA hardware](hardware.md), experimental).
- Compile-only builds were made with clang (Conditions: Apple clang 21 on an Apple M-series desktop) for `thumbv6m-none-eabi` (no FPU, soft float), `thumbv7em-none-eabihf` (single-precision FPU), `armv7m`, big-endian `armebv7m`, `i386` and `aarch64`, all freestanding with `-Werror`, with `nm -u` limited to the declared externs and compiler-rt helpers. RISC-V and Xtensa were not compile-checked: that clang has no backend for either. `riscv32` is in the list of targets the test tries and is skipped where the compiler cannot build for it; Xtensa is not in the list. Cross builds with GCC toolchains were not made.
- There is no WebAssembly build. The emitted C is meant to build with `--target=wasm32 -ffreestanding -nostdlib` and its externs to become imports, but that is not tested.
- A constant index table is stored as `kg_int`. `fir.kg`'s table is 15.4 KB of its 15.5 KB of constants at 32 bits; storing it in the smallest type that holds it would make it about 3.9 KB. Not done.
- The tests need a C compiler and skip without one. Set `KG_TEST_CC` to choose it and `KG_TEST_SANITIZE=undefined` to run them under UBSan.

Reference programs, with the results the interpreter gives and the size of the generated code, are in `examples/embedded/` (`mavg`, `pid` and `fir`, with `.expect` files).

## See also

- [Static profile](static-profile.md) for the rules of the subset.
- [Expression Compiler Notes](compiler-roadmap.md) for how this compiler differs from the expression compiler inside KlongPy.
- [Native runtime](native/index.md) and [Porting to a board](native/porting.md) for running Klong itself on a microcontroller, without generating C.
- [Roadmap](roadmap.md) for what is planned and not available.
