# Timer

KlongPy includes periodic timer capabilities:

```
cb::{.p("hello")}
th::.timer("greeting";1;cb)
```

To stop the timer, it can be closed via:

```
.timerc(th)
```

The following example will create a timer which counts to 5 and then 
terminates the timer by return 0 from the callback.

```
counter::0
u::{counter::counter+1;.p(counter);1}
c::{.p("stopping timer");0}
cb::{:[counter<5;u();c()]}
th::.timer("count";1;cb)
```

which displays:

```
1
2
3
4
5
stopping timer
```


## Intervals and errors

The interval is in seconds and may be fractional. The timer below ticks every 0.2 seconds and stops itself after the third tick, because the callback returns 0:

```
n::0
cb::{n::n+1;.p(n);:[n<3;1;0]}
th::.timer("t";0.2;cb)
```

which displays:

```
1
2
3
```

An interval of 0 runs the callback as soon as the event loop is free. `.timer` raises an error when the interval is negative, `NaN` or not a number, and when the callback is not a function. Older versions returned a message string in these cases.

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

If the callback raises an error while the timer is running, the error is logged and the timer is cancelled.
