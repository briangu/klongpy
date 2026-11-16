# System Functions and Variables

This page lists every system function and variable that KlongPy defines, with a one-line description and the full text of its docstring. It is generated from the `eval_sys_*` docstrings, and the examples were run when the page was generated; the output shown is what the interpreter printed. The REPL command `]h .name` shows the docstring.

An error is shown as `error: message`. The REPL adds its own prefix to errors.

## Evaluation and output

| Name | Title | Description |
|------|-------|-------------|
| `.E(x)` | Evaluate | Evaluate a string of Klong at global scope and return the result. |
| `.d(x)` | Display | Display a value without a newline (strings print without quotes). |
| `.p(x)` | Print | Display a value followed by a newline. |
| `.pf(x;y)` | Print-Format | Fill a template like `.fmt`, print the result and a newline, and return the string. |
| `.fmt(x;y)` | Format | Fill a template string with `{}` placeholders and return the string. |
| `.w(x)` | Write | Write a value in readable form (strings keep their quotes), without a newline. |
| `.bkd(x)` | Backend-Display | Display a value in the backend's own format (a torch tensor stays a tensor). |
| `.bkp(x)` | Backend-Print | Print a value in the backend's own format, followed by a newline. |
| `.x(x)` | Exit | Exit the interpreter with a success or failure status. |
| `.l(x)` | Load | Load and run a Klong source file, searching `KLONGPATH`. |
| `.module(x)` | Module | Mark the start or end of a module; `.module(:name)` opens one and `.module(0)` closes it. |
| `.pc()` | Process-Clock | Return the process time used so far, in seconds. |
| `.rn()` | Random-Number | Return a random real number in 0 <= x < 1. |
| `.sys(x)` | System | Run a shell command and return its exit code. |

### `.E`

`.E(x)` [Evaluate]

```text
Evaluate the Klong program contained in the string "x" and
return its result. This is a direct interface to the Klong
system, e.g. .E("a::123");a will yield 123.

Evaluation happens at GLOBAL scope, so top-level definitions
(`name::...`) persist as globals, even when .E is called from
inside a function (e.g. runtime code generation). Without this,
new vars would land in the (transient) calling frame and be lost.
```

```kgpy
?> .E("a::123");a
123
```

### `.d`

`.d(x)` [Display]

```text
Display the object "x". Backend arrays are converted to numpy for cleaner output.
Use .bkd() for raw backend-specific display.
```

### `.p`

`.p(x)` [Print]

```text
Pretty-print the object "x" (like Display) and then print a
newline sequence. Backend arrays are converted to numpy for cleaner output.
Use .bkp() for raw backend-specific print.
```

```kgpy
?> .p([1 2 3])
[1 2 3]
"[1 2 3]"
```

### `.pf`

`.pf(x;y)` [Print-Format]

```text
Like Format (.fmt), but also write the result and a newline
sequence to the current output channel. Return the string.

Example:   .pf("a={}";1)   writes  a=1
```

```kgpy
?> .pf("a={}";1)
a=1
"a=1"
```

### `.fmt`

`.fmt(x;y)` [Format]

```text
Return the template string "x" with its placeholders replaced by
the arguments in "y". If "y" is a list (not a string), its elements
are the arguments; otherwise "y" is the single argument. To pass one
list as the only argument, wrap it: .fmt("{}";,[1 2 3]).

Placeholders: "{}" takes the next argument, "{0}", "{1}" take by
index (the two styles may not be mixed), "{{" and "}}" are literal
braces. A format may follow a colon, as in "{:8}", "{0:-8}" and
"{:5.2}", and is applied like Format2 (n$value): a positive width
pads on the right, a negative width on the left, and a fractional
format such as 5.2 formats a number (an integer is converted to a
real first). Arguments are rendered as Display does.

Examples:   .fmt("Spot: {}";3.5)       -->  "Spot: 3.5"
            .fmt("{1} {0}";[;1;2])     -->  "2 1"
            .fmt("{:5.2}";3)           -->  "    3.00"
```

```kgpy
?> .fmt("Spot: {}";3.5)
"Spot: 3.5"
?> .fmt("{1} {0}";[;1;2])
"2 1"
?> .fmt("{:8}";"ab")
"ab      "
?> .fmt("{:-8}";"ab")
"      ab"
?> .fmt("{:5.2}";3.0)
"    3.00"
?> .fmt("{{}}";1)
"{}"
?> .fmt("{}{}";1)
error: .fmt: 2 placeholder(s) but 1 argument(s) in template '{}{}'
```

### `.w`

`.w(x)` [Write]

```text
.d and .w both write "x" to the currently selected output port.
However, .w writes a "readable" representation of the given
object and .d pretty-prints the object. The "readable" output
is suitable for reading by .r.

For most types of object there is no difference. Only strings
and characters are printed in a different way:

Object       | .d(Object) | .w(Object)
----------------------------------------
0cx          |  x         | 0cx
"test"       |  test      | "test"
"say ""hi"""|  say "hi"  | "say ""hi"""

NOTE: the " was added above to work in docstring

For some objects, there is no readable representation, including
functions, operators, the undefined object, and the "end of file"
object. A symbolic representation will be printed for those:
:nilad, :monad, :dyad, :triad, :undefined, :eof.

None of these functions terminates its output with a newline
sequence. Use .p (Print) to do so.
```

### `.bkd`

`.bkd(x)` [Backend-Display]

```text
Display the object "x" in raw backend format (tensors shown as-is).
```

### `.bkp`

`.bkp(x)` [Backend-Print]

```text
Pretty-print the object "x" in raw backend format (tensors shown as-is)
and then print a newline sequence.
```

### `.x`

`.x(x)` [Exit]

```text
Terminate the Klong interpreter, returning "success" to the
operating system, if "x" is false (0, [], "") and "failure",
if "x" is not false.
```

### `.l`

`.l(x)` [Load]

```text
Load the content of the file specified in the string "x" as if
typed in at the interpreter prompt.

Klong will try the names "x", and a,".kg", in all directories
specified in the KLONGPATH environment variable. Directory names
in KLONGPATH are separated by colons.

When KLONGPATH is undefined, it defaults to ".:lib".

A program can be loaded from an absolute or relative path
(without a prefix from KLONGPATH) by starting "x" with a "/"
or "." character.

.l will return the last expression evaluated, i.e. it can be
used to load the value of a single expression from a file.
```

### `.module`

`.module(x)` [Module]

```text
Delimit a module. See MODULES documentation.
```

### `.pc`

`.pc()` [Process-Clock]

```text
Return the process time consumed by the Klong interpreter so
far. The return value is in seconds with a fractional part
whose resolution depends on the operating environment and may
be anywhere between 50Hz and 1MHz.

The program {[t0];t0::.pc();x@[];.pc()-t0} measures the process
time consumed by the nilad passed to it.

This function is not avaliable in the Plan 9 port of Klong.
```

### `.rn`

`.rn()` [Random-Number]

```text
Return a random number x, such that 0 <= x < 1.
```

```kgpy
?> .rn()<1
1
```

### `.sys`

`.sys(x)` [System]

```text
Pass the command in the string "x" to the operating system for
execution and return the exit code of the command. On a Unix
system, the command would be executed as

sh -c "command"

and an exit code of zero would indicate success. The command's
stdout and stderr are inherited from the interpreter process. On
Windows the command is run through cmd.
```

The exit code is returned as an integer. Output of the command goes to the terminal of the interpreter process, not to the Klong output channel, so `.tc` does not capture it.

```kgpy
?> .sys("exit 3")
3
?> .sys("true")
0
?> .sys("false")
1
```

## Reading and files

| Name | Title | Description |
|------|-------|-------------|
| `.r()` | Read | Read one data object from the current input channel. |
| `.rl()` | Read-Line | Read one line from the current input channel. |
| `.rls()` | Read-Lines | Read all remaining lines from the current input channel as a list of strings. |
| `.rs(x)` | Read-String | Read a data object from a string (the inverse of `.w`). |
| `.mi(x)` | More-Input | Return 1 if the input channel has more input, otherwise 0. |
| `.ic(x)` | Input-Channel | Open a file for reading and return an input channel. |
| `.oc(x)` | Output-Channel | Open a file for writing (truncating it) and return an output channel. |
| `.ac(x)` | Append-Channel | Open a file for appending and return an output channel. |
| `.cc(x)` | Close-Channel | Close a channel. |
| `.fc(x)` | From-Channel | Select the input channel, or restore standard input with a false value. |
| `.tc(x)` | To-Channel | Select the output channel, or restore standard output with a false value. |
| `.fl()` | Flush | Flush the current output channel. |
| `.df(x)` | Delete-File | Delete a file; an error if it cannot be deleted. |

### `.r`

`.r()` [Read]

```text
Read a single data object from the currently selected input port
and return it. The object being read may be an atom or a list.
When it is a dictionary or list, the input may span multiple
lines.
```

### `.rl`

`.rl()` [Read-Line]

```text
Read a line from the From Channel and return it as a string.
If there is a line separator at the end of the line, it will
be stripped from the string.
```

### `.rls`

`.rls()` [Read-Lines]

```text
Read all lines from the From Channel and return it as an array.
If there is a line separator at the end of the line, it will
be stripped from the string.
```

### `.rs`

`.rs(x)` [Read-String]

```text
.rs is like .r, but reads its input from the string "x". It is
intended for the converting sequentialized compound data objects,
such as lists, arrays, and dictionaries, back to their internal
forms.
```

```kgpy
?> .rs("[1 2 3]")
[1 2 3]
```

### `.mi`

`.mi(x)` [More-Input]

```text
This function returns 1, if the From Channel is not exhausted
(i.e. no reading beyond the EOF has been attempted on that
channel). When no more input is available, it returns 0.

This is a "negative EOF" function.
```

### `.ic`

`.ic(x)` [Input-Channel]

```text
Open the file named in the string "x", link it to an input
channel, and return that channel. Opening a non-existent file
is an error.
```

### `.oc`

`.oc(x)` [Output-Channel]

```text
Both of these functions open a file named in the string "x",
link it to an output channel, and return that channel. The
difference between them is that .oc truncates any existing
file and .ac appends to it.
```

### `.ac`

`.ac(x)` [Append-Channel]

```text
See [Output-Channel].
```

### `.cc`

`.cc(x)` [Close-Channel]

```text
Close the input or output channel "x", returning []. Closing an
already closed channel has no effect. A channel will be closed
automatically when no variable refers to it and it is not the
current From or To Channel.
```

### `.fc`

`.fc(x)` [From-Channel]

```text
.fc selects input channel "x" for reading and .tc selects
output channel "x" for writing, i.e. these functions select a
new From and To Channel, respectively. All input/output will
be redirected to the given channel. Both function will return
the channel that was previously in effect.

When a false value (0,[],"") is passed to these functions, they
restore the default From or To Channel (.cin or .cout).
```

### `.tc`

`.tc(x)` [To-Channel]

```text
See [From-Channel]
```

### `.fl`

`.fl()` [Flush]

```text
Make sure that all output sent to the To Channel is actually
written to the associated file or device ("flush" the channel).
```

### `.df`

`.df(x)` [Delete-File]

```text
Delete the file specified in the string "x". When the file
cannot be deleted (non-existent, no permission, etc), signal
an error.
```

## Python and backend

| Name | Title | Description |
|------|-------|-------------|
| `.py(x)` | Python | Import a Python module and make its functions callable from Klong. |
| `.pyf(x, y)` | Python-From | Import named functions from a Python module. |
| `.pya(x, y)` | Python-Attribute | Return an attribute of a Python object, such as a function in an imported module. |
| `.pyc(x, y, z)` | Python-Call | Call a Python function or method with positional and keyword arguments. |
| `.bkf(x)` | Backend-Function | Import functions from the active backend's array module (NumPy or torch). |

### `.py`

`.py(x)` [Python]

```text
Import a python module in to the current context.  All methods exposed in the module
are loaded into the current context space and callable from within KlongPy.

If the string "x" refers to a directory, KlongPy will append __init__.py
to determine if the directory is a Python module and attempt to load the module.

If the string "x" ends with a .py, then KlongPy will attempt to load this file
as a Python module.

Example:

    Consider a python module called Greetings that exports the "hello" method such as:

        def hello():
            print("Hello, World!")

    The following will load the "greetings" module from the Python modules in the same
    way that 'import hello' would do.  Then we can call 'hello'

        .py("greetings")
        hello() --> "Hello, World!"

    Additionally, a full path to a module can be specified.  This can be useful for
    when creating custom modules that don't need to be installed into Python modules.

        .py("<path>/<to>/greetings")
```

### `.pyf`

`.pyf(x, y)` [Python-From]

```text
Import selected sub-modules from a python module in to the current context.

[See .py() for detials on importing python modules]

Example:

    To import a single sub-module:

    .pyf("numpy"; "sqrt")
    sqrt(4) --> 2.0

    Multiple sub-modules can be imported at once:

    .pyf("numpy"; ["sqrt";"sin"])
    sqrt(4) --> 2.0
    sin(1) --> 0.8414709848078965
```

```kgpy
?> .pyf("numpy";"sqrt");sqrt(4)
2.0
```

### `.pya`

`.pya(x, y)` [Python-Attribute]

```text
Return the attribute named "y" of the Python object "x".
```

### `.pyc`

`.pyc(x, y, z)` [Python-Call]

```text
Call a python function or class function with args and kwargs.

This is a utility function to allow for Klong programs to call Python classes,
or functions, with positional and keyword arguments.

x may be a list with the first element reference the object symbol and the second reference the function name to call.
x may also be a symbol that references the function name to call.

y is either a list of or a single positional arguments to pass to the function.
z is a dictionary of keyword arguments to pass to the function.

Example:

    Python objects may be directly called:

        .pyc(obj;[1 2 3];:{"a":1,"b":2,"c":3})

    or a method on the object may be called:

        .pyc(obj,"method";[1 2 3];:{"a":1,"b":2,"c":3})

    if the python object name is snake case, then the quote is not needed:

    Here, parse_date is a function in the iso8601 module:

        .pyf("iso8601";"parse_date")

    and it can be called via:

        .pyc("parse_date";,"2020-01-01";:{})
```

### `.bkf`

`.bkf(x)` [Backend-Function]

```text
Import functions from the current backend's array module.
This is similar to .pyf() but uses backend-aware functions that
work with both numpy and torch backends.

When using the torch backend, these functions preserve gradient
tracking for autograd.

Example:

    .bkf("exp")
    exp(1.0) --> 2.718...

    .bkf(["exp";"sin";"cos"])
    sin(1.0) --> 0.841...

Common functions available: exp, sin, cos, tan, tanh, sqrt, abs,
log, log10, floor, ceil, round
```

```kgpy
?> .bkf("exp");exp(0.0)
1.0
```

## Strict mode

| Name | Title | Description |
|------|-------|-------------|
| `.strict()` | Strict | Turn on strict mode (level 1): a function may not create a global by assignment. |
| `.unsafe()` | Unsafe | Turn strict mode off (level 0). |

### `.strict`

`.strict()` [Strict]

```text
Enable strict mode for variable assignment (level 1).
In strict mode, a function cannot create a global variable by
assignment; assigning to a name that is not a parameter, a
declared local or an existing global is an error. There is no
global: prefix. To create a global from a function, use .E.

Example:
    .strict()
    counter::0
    increment::{counter::counter+1}  :" OK - counter already exists"
    f::{newvar::42}
    f()  :" ERROR - newvar is not defined"
```

KlongPy has no `global:` prefix. A function that needs to create a global uses `.E`, which evaluates its string at global scope.

```kgpy
?> .strict()
1
?> f::{newvar::42}
:nilad
?> f()
error: undefined variable: newvar
?> g::{.E("newvar::42")}
:nilad
?> g()
42
?> newvar
42
```

### `.unsafe`

`.unsafe()` [Unsafe]

```text
Disable strict mode for variable assignment (level 0).
Strict mode is off. Assigning to a new name inside a function
still makes it local to the call; it does not create a global.
Use .E for that.

Example:
    .unsafe()
    f::{newvar::42}
    f()  :" newvar is local to the call; no global is created"
```

```kgpy
?> .unsafe()
0
?> f::{newvar::42}
:nilad
?> f()
42
?> newvar
:newvar
```

## System variables

| Name | Title | Description |
|------|-------|-------------|
| `.a` | Arguments | The command line arguments of a Klong program. |
| `.cin` | Input-Channel | Standard input channel. |
| `.cout` | Output-Channel | Standard output channel. |
| `.cerr` | Error-Channel | Standard error channel. |
| `.cols` | Columns | Screen width in columns. |
| `.e` | Epsilon | Epsilon: `1+.e` is 1.0 and `.e` is 1e-18 in KlongPy. |
| `.edit` | Editor | Line editor switch. |
| `.fastpow` | Fast-Power | Switch for compiling `x^y` to a faster power function. |
| `.h` | Host | Symbol naming the host system (`:unix` or `:plan9`). |

### `.a`

`.a` [Arguments]

**Not defined in KlongPy.** The name is documented in the Klong manual; in KlongPy it evaluates to the symbol `.a` and has no effect.

```text
This variable holds a list of strings containing the command
line arguments passed to a Klong program.
```

### `.cin`

`.cin` [Input-Channel]

```text
These variables are bound to the standard input (.cin), standard
output (.cout), and standard error (.cerr) channels of the Klong
process. They can be selected for input or output using the
From-Channel (.fc) and To-Channel (.tc) functions.

The standard I/O channels cannot be closed.
```

### `.cout`

`.cout` [Output-Channel]

```text
See [Input-Channel]
```

### `.cerr`

`.cerr` [Error-Channel]

```text
See [Input-Channel]
```

### `.cols`

`.cols` [Columns]

**Not defined in KlongPy.** The name is documented in the Klong manual; in KlongPy it evaluates to the symbol `.cols` and has no effect.

```text
This variable stores the number of columns of the screen on
which the Klong session is running. It defaults to 80. This
variable is used by the line editor, if compiled in and enabled
(see .edit).
```

### `.e`

`.e` [Epsilon]

```text
.e is the smallest value by which two real numbers (0.1<=x<1)
can differ. For numbers smaller than 0.1, there would be a
smaller difference and number x>=1 cannot differ by .e, because
1+.e --> 1. the actual value of .e is implementation-dependent.
In KlongPy it is 1e-18. Numbers are IEEE doubles, so the
spacing between neighbouring reals is about 1.4e-17 at 0.1 and
1.1e-16 just below 1; the statement above about the smallest
difference does not hold.

The logarithm to base 10 of %.e (ln(%.e)%ln(10)) equals the
number of digits in the mantissa of a real number (this is
exactly the exponent in the scientific notation of %.e).
```

KlongPy numbers are IEEE doubles, so the statement above about the smallest difference does not hold. `.e` is 1e-18, `1+.e` is 1.0, and the spacing between neighbouring doubles is about 1.4e-17 at 0.1 and 1.1e-16 just below 1.

```kgpy
?> 1+.e
1.0
?> .e
1e-18
```

### `.edit`

`.edit` [Editor]

**Not defined in KlongPy.** The name is documented in the Klong manual; in KlongPy it evaluates to the symbol `.edit` and has no effect.

```text
When this variable set to a true value AND the line editor is
compiled into the Klong executable, then line editing and
history will be enabled on the Klong prompt. See the section on
LINE EDITING for details.
```

### `.fastpow`

`.fastpow` [Fast-Power]

**Not defined in KlongPy.** The name is documented in the Klong manual; in KlongPy it evaluates to the symbol `.fastpow` and has no effect.

```text
When this variable set to a true value, then expressions of the
form x^y will compile to .pow(x;y), which makes computations
involving powers of real numbers much faster (about eight times
on the author's hardware).

Setting .fastpow::0 will disable this feature and restore the
previous behavior of Klong, just in case.
```

### `.h`

`.h` [Host]

**Not defined in KlongPy.** The name is documented in the Klong manual; in KlongPy it evaluates to the symbol `.h` and has no effect.

```text
This variable holds a unique symbol identifying the host system
running the Klong process. Currently, this is either :unix or
:plan9.
```

## Timers

Defined in the base interpreter. See [Timers](timer.md).

| Name | Title | Description |
|------|-------|-------------|
| `.timer(x, y, z)` | Create-timer | Call a function every `y` seconds (fractions allowed) until it returns 0 or the timer is cancelled. |
| `.timerc(x)` | Cancel-timer | Cancel a timer. |

### `.timer`

`.timer(x, y, z)` [Create-timer]

```text
Create a timer named by "x" that repeats every "y" seconds (fractional
seconds are allowed) and calls function "z".

An interval value of 0 indicates immediately callback (async).

The callback function returns 1 to continue, 0 to stop time timer.

If "z" is a named function, the timer re-resolves it on each tick so redefinitions take effect.

Example:

    cb::{.p("hello")}
    th::.timer("greeting";1;cb)

    To stop the timer, it can be closed via:

    .timerc(th)

    The following example will create a timer which counts to 20 and then 
    terminates the timer by return 0 from the callback.

    counter::0
    u::{counter::counter+1;.p(counter);1}
    c::{.p("stopping timer");0}
    cb::{:[counter<20;u();c()]}
    th::.timer("count";1;cb)
```

The interval `y` may be fractional. A negative, `NaN` or non-numeric `y`, or a `z` that is not a function, raises an error instead of returning a message string. If the callback raises, the error is logged and the timer is cancelled.

```kgpy
?> cb::{0}
:nilad
?> .timer("t";-1;cb)
error: y must be a non-negative number
?> .timer("t";"x";cb)
error: y must be a non-negative number
?> .timer("t";0.5;3)
error: z must be a function
```

### `.timerc`

`.timerc(x)` [Cancel-timer]

```text
Cancel a timer created by .timer()
```

## IPC

Defined in the base interpreter. See [IPC Capabilities](ipc_capabilities.md).

| Name | Title | Description |
|------|-------|-------------|
| `.srv(x)` | Start-IPC-server | Start the IPC server on a port or address; raises if the bind fails; `.srv(0)` stops it. |
| `.cli(x)` | Create-IPC-client | Connect to a KlongPy IPC server and return a function that evaluates code remotely. |
| `.clid(x)` | Create-IPC-dict-client | Connect to a KlongPy IPC server and return a dictionary backed by its variables. |
| `.clic(x)` | Close-IPC-client | Close a connection opened by `.cli` or `.clid`. |
| `.async(x,y)` | Async-function-wrapper | Wrap a function so that it runs asynchronously and calls a completion function. |

### `.srv`

`.srv(x)` [Start-IPC-server]

```text
Open a server port to accept IPC connections.

If "x" is an integer, then it is interpreted as a port in "<all>:<port>".
if "x" is a string, then it is interpreted as a bind address "<bind>:<port>"

if "x" is 0, then the server is closed and existing client connections are dropped.

Returns 1 when the server is listening and 0 if it was already started
(or, for 0, if there was no server). If the port cannot be bound, an error
is raised.

Each interpreter has its own server, so several interpreters may serve
different ports.

Security: a bare port listens on all interfaces (0.0.0.0), and a client can
run any Klong code in this interpreter. KlongPy assumes every peer is
trusted. To limit access, bind to localhost, as in .srv("127.0.0.1:8888"),
and firewall the port. Never expose it to an untrusted network or the
internet.
```

When the port cannot be bound (for example it is in use), `.srv` raises an error such as `cannot listen on 0.0.0.0:8888: ... address already in use`. A call over `.cli` or `.clid` that gets no reply within 30 seconds raises an error containing `timed out`, closes the connection, and fails any other call waiting on it.

### `.cli`

`.cli(x)` [Create-IPC-client]

```text
Return a function which evaluates commands on a remote KlongPy server.

If "x" is an integer, then it is interpreted as a port in "localhost:<port>".
if "x" is a string, then it is interpreted as a host address "<host>:<port>"

If "x" is a remote dictionary, the underlying network connection
is shared and a remote function is returned.

Connection examples:

           .cli(8888)            --> remote function to localhost:8888
           .cli("localhost:8888") --> remote function to localhost:8888

           d::.clid(8888)
           .cli(d)                --> remote function to same connection as d

Evaluation examples:

           f::.cli(8888)

    A string is passed it is evaluated remotely:

           f("hello")             --> "hello" is evaluated remotely
           f("avg::{(+/x)%#x}")   --> "avg" function is defined remotely
           f("avg(!100)")         --> 49.5 (computed remotely)

    Remote functions may be evaluated by passing an array with the first element
    being the symbol of the remote function to execute.  The remaining elements
    are supplied as parameters:

    Example: call :avg with a locally generated !100 range which is passed to the remote server.

           f(:avg,,!100)          --> 49.5

    Similary:

           b::!100
           f(:avg,,b)             --> 49.5

    When a symbol is applied, the remote value is returned.
    For functions, a remote function proxy is returned.

    Example: retrieve a function proxy to :avg and then call it as if it were a local function.

           q::f(:avg)
           q(!100)                --> 49.5

    Example: retrieve a copy of a remote array.

           f("b::!100")
           p::f(:b)               --> "p: now holds a copy of the remote array "b"
```

### `.clid`

`.clid(x)` [Create-IPC-dict-client]

```text
Return a dictionary which evaluates set/get operations on a remote KlongPy server.

If "x" is an integer, then it is interpreted as a port in "localhost:<port>".
if "x" is a string, then it is interpreted as a host address "<host>:<port>"

If "x" is a remote function, the underlying network connection
is shared and a remote function is returned.

Examples:  .cli(8888)             --> remote function to localhost:8888
           .cli("localhost:8888") --> remote function to localhost:8888
           .cli(d)                --> remote function to same connection as d

Connection examples:

           .cli(8888)            --> remote function to localhost:8888
           .cli("localhost:8888") --> remote function to localhost:8888

           f::.cli(8888)
           .cli(f)                --> remote function to same connection as f

Evaluation examples:

           d::.cli(8888)

    Set a remote key/value pair :foo -> 2 on the remote server.

           d,[:foo 2]             --> sets :foo to 2
           d,[:bar "hello"]       --> sets :bar to "hello"
           d,:fn,{x+1}            --> sets :fn to the monad {x+1)

    Get a remote value:

           d?:foo                 --> 2
           d?:bar                 --> hello
           d?:fn                 --> remote function proxy to :avg

    To use the remote function proxy:
           q::d?:fn
           q(2)               --> 3 (remotely executed after passing 2)
```

### `.clic`

`.clic(x)` [Close-IPC-client]

```text
Close a remote dictionary or function opened by .cli or .clid.

Returns 1 if closed, 0 if already closed.

When a connection is closed, all remote proxies / functions tied to this connection
will also close and will fail if called.
```

### `.async`

`.async(x,y)` [Async-function-wrapper]

```text
Returns an async functional wrapper for the function "x" and calls "y"
when completed. The wrapper has the same arity as the wrapped function.
```

## Autograd

Defined in the base interpreter; most need the PyTorch backend. See [PyTorch Backend](torch_backend.md).

| Name | Title | Description |
|------|-------|-------------|
| `.jacobian(x;y)` | Jacobian | Jacobian of a function at a point. |
| `.gradcheck(x;y)` | Gradcheck | Check autograd gradients against numeric ones (PyTorch backend). |
| `.compile(x;y)` | Compile | Compile a function with `torch.compile` (PyTorch backend). |
| `.compilex(x;y;z)` | Compile-Extended | Compile a function with a chosen mode and backend (PyTorch backend). |
| `.cmodes()` | Compile-Modes | List the available `torch.compile` modes and backends (PyTorch backend). |
| `.export(x;y;z)` | Export | Export a function's computation graph to a file (PyTorch backend). |

### `.jacobian`

`.jacobian(x;y)` [Jacobian]

```text
Compute Jacobian matrix of function x at point y.

For f: R^n -> R^m, returns m x n matrix where J[i,j] = df_i/dx_j.

Examples:
    f::{[x@0^2 x@1^2]}
    .jacobian(f;[1 2])   -->  [[2 0] [0 4]]
```

### `.gradcheck`

`.gradcheck(x;y)` [Gradcheck]

```text
Verify that autograd gradients match numeric gradients.
Uses torch.autograd.gradcheck for rigorous verification.
Requires PyTorch backend (USE_TORCH=1).

Arguments:
    x       - Function to check (should return a scalar)
    y       - Input value or list of inputs to check

Returns:
    1 if gradients are correct
    Raises error if gradients don't match

Examples:
    f::{x^2}
    .gradcheck(f;3.0)            :" Returns 1

    g::{(x@0^2)+(x@1^2)}
    .gradcheck(g;[1.0 2.0])      :" Returns 1

Notes:
    - Only supported with PyTorch backend
    - Uses double precision (float64) when available (CPU/CUDA)
    - Falls back to float32 with relaxed tolerances on MPS
    - Useful for verifying custom gradient implementations
```

### `.compile`

`.compile(x;y)` [Compile]

```text
Compile a function for optimized execution using torch.compile.
Requires PyTorch backend (USE_TORCH=1).

Arguments:
    x           - Function to compile
    y           - Example input for tracing the computation graph

Returns:
    Compiled function (faster execution)

Examples:
    f::{x^2}
    cf::.compile(f;3.0)          :" Returns compiled function
    cf(5.0)                       :" 25.0 (optimized)

Notes:
    - Only supported with PyTorch backend
    - Raises error on NumPy backend
    - See .export() for saving graphs to files
```

### `.compilex`

`.compilex(x;y;z)` [Compile-Extended]

```text
Compile a function with extended options for mode and backend.
Requires PyTorch backend (USE_TORCH=1).

Arguments:
    x           - Function to compile
    y           - Example input for tracing the computation graph
    z           - Options dictionary with compile settings

Options (z):
    "mode"      - Compilation mode:
                  "default"         - Balanced (default)
                  "reduce-overhead" - Faster compile, less optimization
                  "max-autotune"    - Slower compile, best runtime
    "backend"   - Compilation backend:
                  "inductor"   - Default with C++/Triton codegen
                  "eager"      - No compilation (debugging)
                  "cudagraphs" - CUDA graphs (GPU only)
    "fullgraph" - 1 to require full graph compilation
    "dynamic"   - 1 for dynamic shapes, 0 for static

Mode Comparison:
    | Mode            | Compile | Runtime | Use Case          |
    |-----------------|---------|---------|-------------------|
    | default         | Medium  | Good    | General use       |
    | reduce-overhead | Fast    | OK      | Development       |
    | max-autotune    | Slow    | Best    | Production        |

Returns:
    Compiled function

Examples:
    f::{x^2}

    :" Fast compilation for development
    cf::.compilex(f;3.0;:{["mode" "reduce-overhead"]})

    :" Maximum optimization for production
    cf::.compilex(f;3.0;:{["mode" "max-autotune"]})

    :" Debug mode (no compilation)
    cf::.compilex(f;3.0;:{["backend" "eager"]})

Notes:
    - Only supported with PyTorch backend
    - Requires C++ compiler for inductor backend
    - Use .cmodes() to see all available options
```

### `.cmodes`

`.cmodes()` [Compile-Modes]

```text
Get information about available torch.compile modes and backends.
Requires PyTorch backend (USE_TORCH=1).

Returns:
    Dictionary with:
        "modes"           - Available compilation modes
        "backends"        - Available compilation backends
        "recommendations" - Suggested settings for common use cases

Examples:
    info::.cmodes()
    .p(info@"modes")          :" Print available modes
    .p(info@"recommendations") :" Print recommended settings

Mode Comparison:
    | Mode            | Compile Time | Runtime Speed | Best For     |
    |-----------------|--------------|---------------|--------------|
    | default         | Medium       | Good          | General use  |
    | reduce-overhead | Fast         | Moderate      | Development  |
    | max-autotune    | Slow         | Best          | Production   |

Backend Comparison:
    | Backend    | Description                              |
    |------------|------------------------------------------|
    | inductor   | Default - C++/Triton code generation     |
    | eager      | No compilation - for debugging           |
    | cudagraphs | CUDA graphs - reduces GPU launch overhead|
```

### `.export`

`.export(x;y;z)` [Export]

```text
Export a function's computation graph to a file for inspection.
Requires PyTorch backend (USE_TORCH=1).

Arguments:
    x           - Function to export
    y           - Example input for tracing the computation graph
    z           - Path to save the graph (.pt2 file)

Returns:
    Dictionary with:
        "compiled_fn" - The compiled function
        "export_path" - Path where graph was saved
        "graph"       - String representation of computation graph

Examples:
    f::{x^2}
    info::.export(f;3.0;"model.pt2")
    .p(info@"graph")              :" Print computation graph

Notes:
    - Only supported with PyTorch backend
    - The exported graph can be loaded with torch.export.load()
    - Use .compile() for just compiling without export
```

## Database

Load with `.py("klongpy.db")`. See [Fast Columnar Database](fast_columnar_database.md).

| Name | Title | Description |
|------|-------|-------------|
| `.db(x)` | Create-db | Create a DuckDB database from a dictionary of tables. |
| `.table(x)` | Table-Create | Create a table from column definitions. |
| `.insert(x, y)` | Table-Insert | Insert rows into a table. |
| `.index(x)` | Create-Table-Index | Create an index on columns of a table. |
| `.rindex(x)` | Reset-Table-Index | Remove the indexes of a table. |
| `.schema(x)` | Table-Schema | Return the columns of a table, or the schemas of a database. |

### `.db`

`.db(x)` [Create-db]

```text
Create a database from a map of tables "x".
The keys are table names and values are the tables.
```

### `.table`

`.table(x)` [Table-Create]

```text
Creates a table created from a sequence of tuples where each tuple
describes a column and it's associated values.  Column order is
determined by the order in the tuple sequence.

The resulting table treatable as a dict and key/value updates are
column operations.

Examples:

    a::[1 2 3]
    b::[2 3 4]
    c::[3 4 5]
    d::[]
    d::d,,"a",,a
    d::d,,"b",,b
    t::.table(d)
    t,"c",,c
```

### `.insert`

`.insert(x, y)` [Table-Insert]

```text
Insert values "y" into a table "x".  The values provided by "y" must be in the
corrensponding column position as specified when the table was created.
If the table is indexed, then the appropriate columns will be used as keys when
inserting values.  If the table is unindexed, then the values are appended.

Examples:

    t::.table(d)
    .insert(t, [1 2 3])

    for batch inserts supply an array of arrays:

    .insert(t, [[1 2 3] [4 5 6]])
```

### `.index`

`.index(x)` [Create-Table-Index]

```text
Creates an index on a table as specified by the columns in array "x".
An index may be one or more columns.

t::.table(d)
.index(t, ["a"])
```

### `.rindex`

`.rindex(x)` [Reset-Table-Index]

```text
Removes indexes from a table.
```

### `.schema`

`.schema(x)` [Table-Schema]

```text
Returns the schema of either a table or dictionary "x".  If "x" is a table,
then the columns are returned.  If "x" is a database, then a dict of
table name to table is returned.
```

## Key-value and table storage

Load with `.py("klongpy.db")`. See [Table and Key-Value Stores](table_and_key_value_stores.md).

| Name | Title | Description |
|------|-------|-------------|
| `.kvs(x)` | KeyValueStorage-Create | Create a file-backed key-value store at a path. |
| `.tables(x)` | TableStorage-Create | Create file-backed table storage at a path. |

### `.kvs`

`.kvs(x)` [KeyValueStorage-Create]

```text
Creates file backed key-value storage at file path "x".

Values are stored as bytes on disk via pickle.

Examples:

    kvs::.kvs("/tmp/storage")
    v::"value"
    kvs,"key",v
    v::kvs?"key"
```

### `.tables`

`.tables(x)` [TableStorage-Create]

```text
Creates Table storage at file path "x".

Table storage is similar to key-value storage except that setting a table value
causes a merge with the existing table on disk.

Examples:

    ts::.tables("/tmp/tables")

    cols::["s" "c" "v"]
    colsFromNames::{{x,,[]}'x}
    prices:::.table(colsFromNames(cols))
    ts,"tables/prices",prices

    .p(ts?"tables/prices")

    updateDb::{[u d];u::x;d::{u?x}'cols;.p(d);.insert(prices;d)}
    updateDb(:{["s" 1] ["c" 2] ["v" 3]})

    .p(prices)

    ts,"tables/prices",prices

    .p(ts?"tables/prices")
```

## Web server

Load with `.py("klongpy.web")`. See [Web Server](web_server.md).

| Name | Title | Description |
|------|-------|-------------|
| `.web(x, y, z)` | Start Web server | Start an HTTP server with GET and POST routes and return its handle. |
| `.webc(x)` | Stop Web server | Stop a web server. |

### `.web`

`.web(x, y, z)` [Start Web server]

```text
Start a web server and return its handle.

The web server is started at the address specifed by "x":

If "x" is an integer, then it is interpreted as a port in "0.0.0.0:<port>".
if "x" is a string, then it is interpreted as a bind address "<bind>:<port>"

Security: a bare port listens on all interfaces (0.0.0.0), and routes
run the Klong functions you register with the request parameters.
KlongPy assumes every peer is trusted. To limit access, bind to
localhost, as in .web("127.0.0.1:8080";get;post), and firewall the
port. Never expose it to an untrusted network or the internet.

GET routes are specified in a dictionary provided by "y".
POST routes are specified in a dictionary provided by "z".

Handler callbacks are called with a dictionary of parameters:

GET handlers are provided a dictionary of query parameters.
POST handlers are provided a dictionary of post body parameters.

Example:

    .py("klongpy.web")
    get:::{}
    get,"/",{x;"hello, world!"}
    wh::.web(8080;get;:{})

    You should now be able to run:

    > curl http://localhost:8080/
    hello, world!
```

### `.webc`

`.webc(x)` [Stop Web server]

```text
Stop and close the web server referenced by "x".
```

## WebSockets

Load with `.py("klongpy.ws")`. See [WebSockets](websockets.md).

| Name | Title | Description |
|------|-------|-------------|
| `.ws(x)` | Create-Websocket-client | Connect to a WebSocket server. |
| `.wsc(x)` | Close-Websocket-client | Close a WebSocket connection. |

### `.ws`

`.ws(x)` [Create-Websocket-client]

```text
Connect to the WebSocket server at address "x" and return the
client. Close it with .wsc.
```

### `.wsc`

`.wsc(x)` [Close-Websocket-client]

```text
Close a remote dictionary or function opened by .ws.

Returns 1 if closed, 0 if already closed.

When a connection is closed, all remote proxies / functions tied to this connection
will also close and will fail if called.
```
