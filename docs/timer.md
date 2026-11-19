# Timer

`.timer` calls a Klong function at a fixed interval, for polling, heartbeats and periodic broadcasts. A timer fires on the interpreter's event loop, so it runs only while the process is alive.

Timers fire while the REPL is open and while a file run is up. Type the examples at the `kgpy` prompt, or load a file and stay in the REPL with `kgpy -l file.kg`. A file run as `kgpy file.kg` ends after its last line unless an IPC server (`.srv`) or web server (`.web`) is listening or a `.cli` or `.ws` connection is open (see [The kgpy command](repl.md#options)); a timer alone does not keep it up, and a timer that has not fired by then never does, as with `kgpy -e EXPR`. To keep a timer-only program running, load it with `kgpy -l file.kg` or hold a server in it.

## Creating and stopping a timer

`.timer(name;seconds;callback)` creates a timer and returns its handle. `.timerc(handle)` cancels it.

```
cb::{.p("hello")}
th::.timer("greeting";1;cb)
.timerc(th)
```

## What a callback is

A callback is a function that takes no arguments, passed by name. Define it first, then pass the name:

<!-- check-docs: render -->
```kgpy
?> cb::{.p("tick");1}
:nilad
?> th::.timer("t";1;cb)
timer:t:1.0
?> .timerc(th)
1
?> .timer("t";1;{x+1})
1:1: type error: z must be a function that takes no arguments, not one of 1
  .timer("t";1;{x+1})
  ^^^^^^^^^^^^^^^^^^^
  hint: name the function first, cb::{...}, and pass cb; an inline function that takes arguments is rejected
```

An inline function literal is not a safe argument. One that takes arguments is rejected, as above. A literal with no arguments, `.timer("t";1;{.p("tick");1})`, is a function as well: `.timer` receives it and runs it every interval, as it does a named function, so the call returns a timer handle. (Only a function literal written as the argument of a user function is run where it is written.)

## The callback's result

The callback decides whether the timer runs again by its return value. A false value (0, an empty list or an empty string) stops the timer. Any other value runs it again after the interval. The callback in the example above prints and returns the string printed, which is true, so it keeps firing until `.timerc`.

This timer ticks every 0.2 seconds and stops itself after the third tick, because the callback returns 0 then:

```
n::0
cb::{n::n+1;.p(n);:[n<3;1;0]}
th::.timer("count";0.2;cb)
```

It displays:

```
1
2
3
```

## Intervals and errors

The interval is in seconds and may be fractional. An interval of 0 runs the callback as soon as the event loop is free. `.timer` raises an error when the interval is negative, `NaN` or not a number, and when the callback is not a function:

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

If the callback is a named function, such as `cb` above, the timer looks the name up again on every tick, so redefining `cb` while the timer runs changes what the next tick calls.

If the callback raises an error, the timer is cancelled and stderr gets the line `timer NAME callback failed; cancelling timer`, where NAME is the name given to `.timer`, followed by the error as the REPL shows it (with the Python traceback under `-v`). This holds for a Klong error and for a Python error alike, and the line appears once.

A callback runs on the interpreter's event loop. A callback that waits for a synchronous call blocks that loop, so a call that this process would have to answer itself, such as a call to its own server, is a system error at once (see [Calling your own server](ipc_capabilities.md#calling-your-own-server)). Use `.async` there. A callback that merely takes long delays every other timer and request until it returns.

`.x` in a callback ends the process, with the status `.x` gives, after flushing what was printed.

## See also

- [IPC and timers on a device](native/ipc.md): the timers of the native runtime.
- [System functions](system-functions.md): the reference for `.timer` and `.timerc`.
- [Roadmap](roadmap.md)
