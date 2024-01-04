# Klong Language Examples

Explore the power and simplicity of Klong with these engaging examples. Each snippet highlights a unique aspect of Klong, demonstrating its versatility in various programming scenarios.

## 1. Basic Arithmetic

```klong
?> 5 + 3 * 2
11
```

## 2. Manipulating lists

Squaring numbers in a list

```bash
?> {x*x}'[1 2 3 4 5]
[1 4 9 16 25]
```

Vectorized approach will do an element-wise multiplication in bulk:

```bash
?> a::[1 2 3 4 5];a*a
[1 4 9 16 25]
```

## 3. Data Analysis with Python Integration

Integrating Klong with Python's NumPy for data analysis

```python
from klongpy import KlongInterpreter
import numpy as np

data = np.array([1, 2, 3, 4, 5])
klong = KlongInterpreter()
klong['data'] = data
klong('avg::{(+/x)%#x}')
klong('avg(data)')
```

## 4. Database Functionality

Database operation: creating and querying a table.

```bash
?> .py("klongpy.db")
?> t::.table([["name" ["Alice" "Bob"]] ["age" [25 30]]])
name age
Alice 25
Bob 30
?> db::.db(:{},"T",t)
?> db("select * from T where age > 27")
name age
Bob 30
```

## 5. Asynchronous and Remote Function Calls

Demonstrating IPC and async remote function calls
Execute on remote server with input 0-99

```bash
?> avg::{(+/x)%#x}
:monad
?> .srv(8888)
1
```

```bash
?> f::.cli(8888)
remote[localhost:8888]:fn
?> f(:avg,,!100)
49.5
?> avg::f(:avg)
remote[localhost:8888]:fn:avg:monad
?> afn::.async(avg;{.d("Avg calculated: ");.p(x)})
async::monad
?> afn(!100)
Avg calculated: 49.5
1
```

## 6. Web Server Implementation

Implementing a basic web server

create a file called web.kg with the following code that adds one index handler:

```text
.py("klongpy.web")
data::!10
index::{x; "Hello, Klong World! ",data}
.web(8888;:{},"/",index;:{})
.p("ready at http://localhost:8888")
```

```bash
$ kgpy web.kg
ready at http://localhost:8888
```

In another terminal:

```bash
$ curl http://localhost:8888
['Hello, Klong World! ' 0 1 2 3 4 5 6 7 8 9]
```

## Notes

These examples are designed to illustrate the ease of use and diverse applications of Klong, making it a versatile choice for various programming needs.
