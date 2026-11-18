# System Functions and Variables

This page lists every system function and variable that KlongPy defines, with a one-line description and the full text of its docstring. It is generated from the `eval_sys_*` docstrings, and the output shown is what the interpreter prints. The REPL command `]h .name` shows the docstring.

An error is shown as `error: message`, without the position the REPL adds; see [Reading an error](errors.md#reading-an-error).

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

Object          | .d(Object) | .w(Object)
-------------------------------------------
0cx             |  x         | 0cx
"test"          |  test      | "test"
"say ""hi"""  |  say "hi"  | "say ""hi"""

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
intended for converting serialized compound data objects,
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
| `.pyf(x;y)` | Python-From | Import named functions from a Python module. |
| `.pya(x;y)` | Python-Attribute | Return an attribute of a Python object, such as a function in an imported module. |
| `.pyc(x;y;z)` | Python-Call | Call a Python function or method with positional and keyword arguments. |
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

    Consider a Python package called greetings that exports the "hello" function:

        def hello():
            print("Hello, World!")

    This loads the package, as 'import greetings' does, and then calls 'hello',
    which prints Hello, World!

        .py("greetings")
        hello()

    Additionally, a full path to a module can be specified.  This can be useful for
    when creating custom modules that don't need to be installed into Python modules.

        .py("<path>/<to>/greetings")
```

### `.pyf`

`.pyf(x;y)` [Python-From]

```text
Import the named functions of a Python module into the current context.

[See .py() for details on importing python modules]

Example:

    To import one function:

    .pyf("numpy"; "sqrt")
    sqrt(4) --> 2.0

    To import several:

    .pyf("numpy"; ["sqrt" "sin"])
    sin(1) --> 0.8414709848078965
```

```kgpy
?> .pyf("numpy";"sqrt");sqrt(4)
2.0
```

### `.pya`

`.pya(x;y)` [Python-Attribute]

```text
Return the attribute named "y" of the Python object "x".
```

### `.pyc`

`.pyc(x;y;z)` [Python-Call]

```text
Call a python function or class function with args and kwargs.

This is a utility function to allow for Klong programs to call Python classes,
or functions, with positional and keyword arguments.

x may be a list: the object symbol, then the name of the method to call.
x may also be a symbol that references the function name to call.

y is a list of positional arguments, or a single positional argument.
z is a dictionary of keyword arguments to pass to the function.

Example:

    Python objects may be directly called:

        .pyc(obj;[1 2 3];:{["a" 1] ["b" 2] ["c" 3]})

    or a method on the object may be called:

        .pyc(obj,"method";[1 2 3];:{["a" 1] ["b" 2] ["c" 3]})

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

    .bkf(["exp" "sin" "cos"])
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
.strict(x)

Set the strict level and return it. .strict() sets level 1;
"x" may be 0, 1 or 2. Level 0 is the default: see .unsafe().

Level 1: malformed input that level 0 only warns about (input
left after a statement, an unterminated string or list, a
missing operand) is a syntax error, and so is an assignment
whose left side is not a name. Inside a function, assigning to
a name that is not a declared local or an existing variable is
an error, and so is reading a name that is not defined.

Level 2: in addition, reading an undefined name is an error
everywhere, as is reading a local variable before it has been
assigned and passing more arguments than a function takes; a
quoted symbol (:a) is always the symbol, never the value of a
variable of that name.

The rule for :: is the same at every level: when a variable of
that name exists in any scope, it is assigned there. Otherwise,
at top level the variable is created; inside a function the
name becomes local to the call (level 0) or the assignment is
an error (level 1 and 2). To change a variable from a function,
define it at top level first. To create one from inside a
function, use .E.

Text is checked for syntax at the level in force when it is
read, and code runs under the level in force when it runs.

Example:
    .strict()
    counter::0
    increment::{counter::counter+1}  :" OK - counter exists"
    f::{newvar::42}
    f()  :" ERROR - undefined variable: newvar"
    g::{[t];t::42;t}
    g()  :" OK - t is declared local"
```

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
Set strict level 0, the default, and return 0. See .strict().

At level 0, assigning to a name that does not exist inside a
function makes the name local to that call. It does not create
a variable outside the function; use .E for that.

Example:
    .unsafe()
    f::{newvar::42}
    f()     :" 42; newvar is local to the call"
    newvar  :" still undefined"
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
| `.timer(x;y;z)` | Create-timer | Call a function every `y` seconds (fractions allowed) until it returns 0 or the timer is cancelled. |
| `.timerc(x)` | Cancel-timer | Cancel a timer. |

### `.timer`

`.timer(x;y;z)` [Create-timer]

```text
Create a timer named by "x" that calls the function "z" every "y"
seconds (fractions allowed) and returns its handle. An interval of 0
calls "z" at once, on the Klong loop. The result of the callback decides
what happens next: a false value (0, an empty list or string) stops the
timer, and any other value runs it again. A named function is looked up
again on each tick, so a redefinition takes effect. Stop a timer with .timerc.

Example:

    n::0
    cb::{n::n+1;.p(n);:[n<3;1;0]}
    th::.timer("count";0.2;cb)
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
| `.async(x;y)` | Async-function-wrapper | Wrap a function so that it runs asynchronously and calls a completion function. |

### `.srv`

`.srv(x)` [Start-IPC-server]

```text
Listen for IPC connections. "x" is a port (all interfaces, 0.0.0.0), a
string "<bind>:<port>" for one interface, or 0 to stop the server and
drop its clients. Returns 1 when the server is now listening and 0 when
it was already started (for 0, when there was no server). If the port
cannot be bound, an error is raised. Clients speak the kdb+ IPC protocol.
Hooks, set as global functions: .srv.o(h) when a client connects,
.srv.c(h) when it disconnects, .srv.e(h;text) on an error that nobody
is waiting for, and .srv.auth(user;password), which returns 1 to
accept a client and 0 to refuse it. The .system key "ipc_max_message"
limits the message size (256 MiB by default). A 0.7 client (pickle
protocol, removed in 2.0) is closed at once and the cause is logged.

Security: a client can run any Klong code in this interpreter, and
credentials are sent in clear text. Bind to localhost, as in
.srv("127.0.0.1:8888"), and never expose the port to an untrusted
network. See the IPC Capabilities page.

Example:   .srv(8888)       -->  1
```

When the port cannot be bound (for example it is in use), `.srv` raises an error such as `cannot listen on 0.0.0.0:8888: ... address already in use`. A call over `.cli` or `.clid` that gets no reply within 30 seconds raises an error containing `timed out`, closes the connection, and fails any other call waiting on it.

### `.cli`

`.cli(x)` [Create-IPC-client]

```text
Connect to a server and return a function that runs code there. "x" is a
port (on localhost), "<host>:<port>", or "<host>:<port>:<user>:<password>"
(credentials travel in clear text); a remote dictionary from .clid shares
its connection. A string argument is Klong source for the server to run.
A symbol argument is the name of a server function, followed by its
arguments: f(:avg,,!100). A symbol alone returns the remote value, and
a remote function proxy for a function. The connection speaks the kdb+
IPC protocol, the only one; the pickle protocol of 0.7 was removed, and a
2.0 client cannot talk to a 0.7 server (the handshake times out).
A call waits 30 seconds at most (see .clit). An error on the server is
raised here as the error of its kind. If nothing listens on the port,
.cli blocks for about 155 seconds, then returns the function, and the
first call through it is a system error. See the IPC Capabilities page.

Example:   f::.cli(8888)
           f("1+1")         -->  2
           f(:avg,,!100)    -->  49.5
```

### `.clid`

`.clid(x)` [Create-IPC-dict-client]

```text
Connect to a server and return a dictionary of its variables. "x" is
an address as for .cli, or a remote function from .cli, whose
connection is shared. Setting a key sets a global on the server, and a
get of a name the server does not have is a name error. A function set
on the server is read back as a remote function proxy. The protocol,
timeout and errors are those of .cli.

Example:   d::.clid(8888)
           d,[:foo 2]       -->  sets :foo to 2 on the server
           d?:foo           -->  2
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

`.async(x;y)` [Async-function-wrapper]

```text
Returns an async functional wrapper for the function "x" and calls "y"
when completed. The wrapper has the same arity as the wrapped function.
Both must be functions, else a type error is raised.

Calling the wrapper returns 1 at once. A local function runs as a task on the
Klong loop and "y" is called with its result. When "x" is a connection (from
.cli) or a remote function proxy, the request is sent without blocking
and "y" runs on the Klong loop when the answer arrives; an error answer or a
lost connection is logged and "y" is not called. With a connection the
wrapper takes the one request argument, as the connection does.
```

## Autograd

Defined in the base interpreter; most need the PyTorch backend. See [PyTorch Backend and Autograd](torch_backend.md).

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
Requires the torch backend (kgpy --backend torch, or KlongInterpreter(backend="torch")).

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
Requires the torch backend (kgpy --backend torch, or KlongInterpreter(backend="torch")).

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
Requires the torch backend (kgpy --backend torch, or KlongInterpreter(backend="torch")).

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
Requires the torch backend (kgpy --backend torch, or KlongInterpreter(backend="torch")).

Returns:
    Dictionary with:
        "modes"           - Available compilation modes
        "backends"        - Available compilation backends
        "recommendations" - Suggested settings for common use cases

Examples:
    info::.cmodes()
    .p(info@"modes")          :" Print available modes
    .p(info@"recommendations") :" Print recommended settings
```

### `.export`

`.export(x;y;z)` [Export]

```text
Export a function's computation graph to a file for inspection.
Requires the torch backend (kgpy --backend torch, or KlongInterpreter(backend="torch")).

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

## Embedded programs

Defined in the base interpreter. See [Compiling Klong to C](embedded.md).

| Name | Title | Description |
|------|-------|-------------|
| `.extern(x;y)` | Extern | Declare a function the host provides, with a typed signature, for a program that is compiled. |
| `.entry(x;y)` | Entry | Mark a function as an entry point of a compiled program and fix its signature. |
| `.param(x;y)` | Param | Declare a global of a compiled program as a parameter the host sets between runs (a register in hardware). |
| `.fixed(x;y)` | Fixed | Convert a number to a fixed-point type, `fixed<W,I>`, in a compiled program. |
| `.float(x)` | Float | Convert a fixed-point or integer value to a float in a compiled program. |

### `.extern`

`.extern(x;y)` [Extern]

```text
.extern(x;y;z)

Declare a function the host provides. "x" is its name, a
string; "y" is its signature, a string of the form
"params -> result"; "z" is the module it is imported from
when the program is compiled to wasm ("env" when omitted).

A type is int, float, an array of them with fixed dimensions
(int[8], float[16,241]) or, for a parameter, str. The result
may also be void. Parameters are separated by spaces and may
be absent: "-> int" is a function of no arguments.

Under KlongPy the name is bound to the Python function that
the host program registered as its stand-in,
klong.externs["name"]. Arguments and result are checked
against the signature on every call; a call without a
stand-in is an error. A compiled program calls the C function
of that name, with "." written as "_" (read.adc is read_adc).

Returns the signature in its canonical form.

Examples:   .extern("read.adc";"int -> float")
            .extern("write.pwm";"int float -> void")
            .extern("fill";"float[64] -> float[64]";"dsp")
            read.adc(0)
```

### `.entry`

`.entry(x;y)` [Entry]

```text
Mark the function named "x" (a string) as an entry point of
a compiled program and fix the types of its arguments and
result with the signature "y" (see .extern; str is not
allowed). The function must already be defined and must take
as many arguments as the signature has parameters. A function
may have several entries with different shapes.

Under KlongPy this only checks the declaration and records it
in klong.externs.entries; the function is called as usual.

Returns the signature in its canonical form.

Example:   step::{x*2}
           .entry("step";"float -> float")
```

### `.param`

`.param(x;y)` [Param]

```text
.param(x;y;z)

Declare the global named by the string "x" as a tunable
parameter of a compiled program: an input register that the host
writes between runs and that is constant during one. "y" is its
type, "int", "float" or "fixed<W,I>" (a scalar); "z" is the
value it starts with (0 when omitted).

The name is defined as an ordinary global. The compiled program
may read it and may not assign it. Array shapes and loop bounds
cannot depend on it: a parameter may bound a window only as "use
the first p of a fixed maximum". Set it between runs with
klongpy.static.set_param(klong;name;value) (the hardware binding
.hw.set does the same on the card).

Returns the type in its canonical form.

Examples:   .param("window";"int";4)
            .param("stop";"fixed<32,16>";0.5)
```

### `.fixed`

`.fixed(x;y)` [Fixed]

```text
Convert the number or array "y" to the fixed-point type named by
the string "x", for example "fixed<32,16>" (32 bits, 16 of them
integer bits: the value of the raw integer r is r*2^-16). An int
converts exactly, a float and a wider fixed-point value are
truncated toward minus infinity, and the result wraps to the
width of the type.

This is how a fixed-point constant is written. Under KlongPy it
returns a fixed-point value that the static profile understands:
programs that use fixed-point numbers run through
klongpy.static.call_entry, not as plain Klong.

Examples:   .fixed("fixed<32,16>";0.5)
            .fixed("fixed<64,32>";price)
            .fixed("fixed<8,4>";[1 2 3])
```

### `.float`

`.float(x)` [Float]

```text
Convert the fixed-point number or array "x" (or an int) to a
float: the value raw*2^-(W-I). The result is exact when the type
has at most 53 bits; compiled code rounds it to float32.

Examples:   .float(.fixed("fixed<32,16>";0.5))  -->  0.5
```

## Hardware (experimental)

Defined in the base interpreter. See [Klong functions as FPGA hardware](hardware.md).

| Name | Title | Description |
|------|-------|-------------|
| `.hw(x)` | Hardware | Bind the FPGA kernels of a built program (or the simulator's) and return a dictionary of functions. |
| `.hw.lanes(x;y)` | Hw-Lanes | Bind the kernels of a program built with several lanes: copies of the program side by side. |
| `.hw.set(x;y;z)` | Hw-Set | Set a parameter of a bound kernel between runs. |
| `.hw.put(x;y)` | Hw-Put | Upload the arguments of calls of a kernel to the card once and return a buffer. |
| `.hw.get(x)` | Hw-Get | Download a buffer made by `.hw.put`. |
| `.hw.free(x)` | Hw-Free | Free a buffer made by `.hw.put`. |
| `.hw.reset(x)` | Hw-Reset | Make the next call of a kernel start the program from its initial state. |
| `.hw.info()` | Hw-Info | A dictionary describing the hardware in use and the loaded image. |

### `.hw`

`.hw(x)` [Hardware]

```text
Bind the FPGA kernels of a built program. "x" is the path of
a Klong program of the static profile that has .entry
declarations, or a list of paths for an image that was
built from several programs (kgpy --hw-build a.kg b.kg).
The program is checked and compiled, the image built from
exactly this source is looked up on the card machine
($KLONG_HW_HOST; the build cache must be there) and loaded, and a dictionary
from entry name to function is returned. Each entry is also
bound as "program.entry", where "program" is the file name
without ".kg": fir::.hw("fir.kg") makes fir.run, which is
also fir@"run".

A call checks its arguments against the entry's signature
and runs on the card; an argument with one extra leading
dimension means that many calls in one run, and the result
has that dimension too. The program's state is kept between
calls and starts fresh at the first call after .hw.

With KLONG_HW=sim the kernels run on this machine's CPU: the
generated HLS code compiled with the C++ compiler. That is a
simulator, not hardware. There is no image to build then.

Without KLONG_HW=sim, KLONG_HW_HOST or pyxrt, .hw is an
error that says so. Loading an image replaces what the card
held. See docs/hardware.md.

Examples:   fir::.hw("fir.kg")
            fir.run(256#1.0)
```

### `.hw.lanes`

`.hw.lanes(x;y)` [Hw-Lanes]

```text
Bind the kernels of a program built with --lanes y
(kgpy --hw-build FILE --lanes y). Like .hw: "x" is a path
or a list of paths. The kernel then runs y copies of the
program on the same stream of calls: every .param is set
with .hw.set as an array of y values, and a result has a
leading dimension y.

Example:   s::.hw.lanes("crossover.kg";64)
```

### `.hw.set`

`.hw.set(x;y;z)` [Hw-Set]

```text
Set the parameter "y" (declared in the program with
.param) of the kernels of "x" to the value "z", between
runs. "x" is what .hw returned, or one of its kernels.
The value is converted as the interpreter converts it
(a fixed-point parameter takes a number, truncating toward
minus infinity); for a kernel bound with .hw.lanes it is
an array with one value per lane. The kernel keeps the
value, through resets, until it is set again. Returns the
number of kernels set.

Example:   .hw.set(s;"fast";[3 5 8])
```

### `.hw.put`

`.hw.put(x;y)` [Hw-Put]

```text
Upload the arguments of calls of the kernel "x" to the
card once and return a buffer. "y" is the argument of the
kernel (with the extra leading dimension for many calls)
or, for a kernel with several arguments, a list of them.
A buffer is accepted by the kernel in place of its
arguments: fir.run(b). It stays on the card until it is
freed with .hw.free or the image is unloaded or the session
ends.

Example:   b::.hw.put(fir.run;x)
           fir.run(b)
```

### `.hw.get`

`.hw.get(x)` [Hw-Get]

```text
Download a buffer made by .hw.put as an array of the
arguments it holds (for a kernel with several arguments,
the records: one row per call, the arguments flattened one
after the other).

Example:   .hw.get(b)
```

### `.hw.free`

`.hw.free(x)` [Hw-Free]

```text
Free a buffer made by .hw.put. Returns 1. Freeing a buffer
twice is an error.

Example:   .hw.free(b)
```

### `.hw.reset`

`.hw.reset(x)` [Hw-Reset]

```text
Make the next call of the kernel "x" start the program from
its initial state (the program's globals as initialised
when the program was loaded). Returns 1.

Example:   .hw.reset(mavg.step)
```

### `.hw.info`

`.hw.info()` [Hw-Info]

```text
A dictionary describing the hardware in use: "mode"
("card", "local" or "sim"), "device", "shell", the loaded
"image" (path), its "key", "kernels", the "build" record
of the image (timing, utilisation, build time), and the
resident "buffers" and "buffer-bytes". It does not load
anything. Without a hardware to use it is an error.

Example:   .hw.info()
```

## Database

Load with `.py("klongpy.db")`. See [Columnar tables and SQL](fast_columnar_database.md).

| Name | Title | Description |
|------|-------|-------------|
| `.db(x)` | Create-db | Create a DuckDB database from a dictionary of tables. |
| `.table(x)` | Table-Create | Create a table from column definitions. |
| `.insert(x;y)` | Table-Insert | Insert rows into a table. |
| `.index(x;y)` | Create-Table-Index | Create an index on columns of a table. |
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
Create a table from a sequence of tuples, each of which
describes a column and its values.  Column order is
determined by the order in the tuple sequence.

The table can be used like a dictionary, and key/value updates are
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

`.insert(x;y)` [Table-Insert]

```text
Insert values "y" into a table "x".  The values provided by "y" must be in the
corresponding column position as specified when the table was created.
If the table is indexed, then the appropriate columns will be used as keys when
inserting values.  If the table is unindexed, then the values are appended.

Examples:

    t::.table(d)
    .insert(t;[1 2 3])

    for batch inserts supply an array of arrays:

    .insert(t;[[1 2 3] [4 5 6]])
```

### `.index`

`.index(x;y)` [Create-Table-Index]

```text
Creates an index on the table "x" as specified by the columns in array "y".
An index may be one or more columns.

t::.table(d)
.index(t;["a"])
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
| `.web(x;y;z)` | Start Web server | Start an HTTP server with GET and POST routes and return its handle. |
| `.webc(x)` | Stop Web server | Stop a web server. |

### `.web`

`.web(x;y;z)` [Start Web server]

```text
Start a web server and return its handle.

The web server is started at the address specified by "x":

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
Close a WebSocket client opened by .ws.

Returns 1 if closed, 0 if already closed.
```

## Other

| Name | Title | Description |
|------|-------|-------------|
| `.clit(x;y)` | Set-IPC-call-timeout | Set the call timeout of a remote function or dictionary opened by .cli or .clid (or handed to a server hook) to "y" seconds. |
| `.grad(x;y)` | Grad | Compute the numeric gradient of function x at point y, using finite differences. |

### `.clit`

`.clit(x;y)` [Set-IPC-call-timeout]

```text
Set the call timeout of a remote function or dictionary opened by .cli or .clid
(or handed to a server hook) to "y" seconds. A call that gets no answer in time
closes the connection and fails, as does every other call waiting on it.
The default is 30 seconds; 0 removes the limit.

Returns 1 when set, 0 if "x" is not a connection handle.

Example:

           c::.cli(8888)
           .clit(c;2.5)           --> 1: calls on c now fail after 2.5 seconds
```

### `.grad`

`.grad(x;y)` [Grad]

```text
Compute the numeric gradient of function x at point y, using finite
differences. This is the ASCII form of the nabla operator: .grad(f;y)
equals y∇f. The function comes first, as in .jacobian(f;y).

The point y is taken by value: a number or a list of numbers. (The
operator y∇f also accepts a symbol and then perturbs that variable;
.grad does not.) For automatic differentiation use f:>y.

Examples:
    .grad({x^2};3)               -->  6.0 (approximately)
    .grad({+/x^2};[1 2 3])       -->  [2 4 6] (approximately)
```
