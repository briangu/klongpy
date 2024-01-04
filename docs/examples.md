# Klong Language Examples

Explore the power and simplicity of Klong with these engaging examples. Each snippet highlights a unique aspect of Klong, demonstrating its versatility in various programming scenarios.

## 1. Basic Arithmetic

```kgpy
?> 5 + 3 * 2
11
```

```kgpy
?> sum::{+/x}  :" sum + over / the array x
:monad
?> sum([1 2 3])
6
?> count::{#x}
:monad
?> count([1 2 3])
3
```

```kgpy
?> avg::{sum(x)%count(x)} :" average is the sum divided by the number of elements
:monad
?> avg([1 2 3])
2
```

## 2. Math on arrays

Squaring numbers in a list

```kgpy
?> {x*x}'[1 2 3 4 5] :" square each element as we iterate over the array
[1 4 9 16 25]
```

Vectorized approach will do an element-wise multiplication in bulk:

```kgpy
?> a::[1 2 3 4 5];a*a  :" a*a multiplies the arrays
[1 4 9 16 25]
```

If you really want to see the performance difference, let's crank up the size of the array and time it:

```kgpy
$> .l("time")
:monad
$> a::!1000;#a
1
$> fast::{{a*a}'!1000}
:nilad
$> slow::{{{x*x}'a}'!1000}
:nilad
$> time(fast)
0.015867948532104492
$> time(slow)
2.8987138271331787
```

Vectors win by 182x!  Why?  Because when you perform a bulk vector operation the CPU can perform the math with much less overhead and do many more operations at a time because it has the entire computation presented to it at once.

KlongPy aims to give you tools that let you conveniently exploit this vectorization property - and go FAST!

Less code, but faster.

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

How about we use the NumPy FFT?

```kgpy
?> .pyf("numpy";"fft");fft::.pya(fft;"fft")
:monad
?> signal::[0.0 1.0 0.0 -1.0] :" Example simple signal
[0.0 1.0 0.0 -1.0]
?> result::fft(signal)
[0j -2j 0j 2j]
```

Now you can use NumPy or other libraries to provide complex functions while KlongPy lets you quickly prepare and process the vectors.

## 4. Database Functionality

KlongPy leverages a high-performance columnar store called DuckDb and uses zero-copy NumPy array operations, allowing you to quickly create arrays in KlongPy and then perform SQL on the data for deeper insights.

It's easy to create a table and a db to query:

```kgpy
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

```kgpy
?> avg::{(+/x)%#x}
:monad
?> .srv(8888)
1
```

```kgpy
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
