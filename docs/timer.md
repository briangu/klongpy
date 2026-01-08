# Timer

KlongPy includes periodic timer capabilities:

```
cb::{.p("hello")}
th::.timer("greeting";1;cb)
```

Timers look up the callback function each time they execute. Updating the
function definition after creating the timer will change the behavior on the
next tick.

Intervals may be fractional to allow sub-second timers, e.g. `.timer("g";0.5;cb)`
fires every half-second.

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

