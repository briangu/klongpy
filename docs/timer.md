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

