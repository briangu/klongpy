# Timer

`.timer` calls a Klong function at a fixed interval, for polling, heartbeats and periodic broadcasts. A timer fires on the interpreter's event loop, so it runs only while the process is alive.

Timers fire while the REPL is open. Type the examples at the `kgpy` prompt, or load a file and stay in the REPL with `kgpy -l file.kg`. A file run as `kgpy file.kg` or `kgpy -e` can create timers, because they start the same system context as the REPL. The process exits when the file or the expression ends, though, and a timer that has not fired by then never does. A script that is meant to tick has to keep the process up, for example with a server (`.srv`), or be loaded with `-l`.

## Creating and stopping a timer

`.timer(name;seconds;callback)` creates a timer and returns its handle. `.timerc(handle)` cancels it.

```
cb::{.p("hello")}
th::.timer("greeting";1;cb)
.timerc(th)
```

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

If the callback raises an error, the error is logged with its traceback and the line `timer NAME callback failed; cancelling timer`, where NAME is the name given to `.timer`, and the timer is cancelled.

## See also

- [IPC and timers on a device](native/ipc.md): the timers of the native runtime.
- [System functions](system-functions.md): the reference for `.timer` and `.timerc`.
- [Roadmap](roadmap.md)
