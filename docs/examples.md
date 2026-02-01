# KlongPy Examples

Explore the power and simplicity of KlongPy with these engaging examples. Each snippet highlights a unique aspect of Klong, demonstrating its versatility in various programming scenarios.

Before we get started, you may be wondering: *Why is the syntax so terse?*

The answer is that it's based on the APL style array language programming and there's a good reason why its compact nature is actually helpful.

Array language style lets you describe WHAT you want the computer to do and it lets the computer figure out HOW to do it.  This frees you up from the details while letting the computer figure out how to go as fast as possible.

Less code to write and faster execution.

---

Just so the following examples make more sense when you see the REPL outputs, there are a few quick rules about Klong functions.  Functions only take up to 3 parameters and they are ALWAYS called x,y and z.

A function with

* no parameters is called a nilad
* one parameter is called a monad (x)
* two parameters: dyad (x and y)
* three parameters: a triad (x, y and z)

The reason that Klong functions only take up to 3 parameters AND name them for you is both convenience and compactness.

---

## 1. Basic Arithmetic

Let's get started with the basics and build up to some more interesting math.

```kgpy
?> 5+3*2 :" Expressions are evaluated from right to left: 3*2 and then + 5
11
```

KlongPy is more about arrays of things, so let's define sum and count functions over an array:

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

Now that we know the sum and number of elements we can compute the average:

```kgpy
?> avg::{sum(x)%count(x)} :" average is the sum divided by the number of elements
:monad
?> avg([1 2 3])
2
```

## 2. Math on arrays

Let's dig into more interesting operations over array elements.  There's really big performance differences in how you approach the problem and it's important to see the difference.

For the simple case of squaring numbers in a list, let's try a couple solutions:

```kgpy
?> {x*x}'[1 2 3 4 5] :" square each element as we iterate over the array
[1 4 9 16 25]
```

The vectorized approach will do an element-wise multiplication in bulk:

```kgpy
?> a::[1 2 3 4 5];a*a  :" a*a multiplies the arrays
[1 4 9 16 25]
```

The vectorized approach is going to be MUCH faster.  Let's crank up the size of the array and time it:

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

Less code to write AND faster to compute.

## 3. Data Analysis with Python Integration

KlongPy integrates seamlessly with Python so that the strengths of both can be combined. It's easy to use KlongPy from Python and vice versa.

For example, let's say we have some data in Python that we want to operate on in KlongPy.  We can just directly use the interpreter in Python and run functions on data we put into the KlongPy context:

```python
from klongpy import KlongInterpreter
import numpy as np

data = np.array([1, 2, 3, 4, 5])
klong = KlongInterpreter()
# make the data NumPy array available to KlongPy code by passing it into the interpreter
# we are creating a symbol in KlongPy called 'data' and assigning the external NumPy array value
klong['data'] = data
# define the average function in KlongPY
klong('avg::{(+/x)%#x}')
# call the average function with the external data and return the result.
r = klong('avg(data)')
print(r) # expected value: 3
```

It doesn't make sense to write code in Klong that already exists in other libraries. We can directly access them via the Python import functions (.py and .pyf).

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

There's a lot more we can do with interop but let's move on for now!

## 4. Database Functionality

KlongPy leverages a high-performance columnar store called DuckDB that uses zero-copy NumPy array operations behind the scenes. This database allows fast interop between KlongPy and DuckDB (the arrays are not copied) so that applications can manage arrays in KlongPy and then instantly perform SQL on the data for deeper insights.

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

## 5. IPC, Remote Function Calls and Asynchronous operations

Inter Process Communication (IPC) lets you build distributed and interconnected KlongPy programs and services.

KlongPy treats IPC connections to servers as functions. These functions let you call the server and ask for things it has in its memory - they can be other functions or values, etc. For example you can ask for a reference to a remote function and you will get a local function that when you call it runs on the server with your arguments. This general "remote proxy" approach allows you to write your client code in the same way as if all the code were running locally.

To see this in action, let's setup a simple scenario where the server has an "avg" function and the client wants to call it.

Start a server in one terminal:

```kgpy
?> avg::{(+/x)%#x}
:monad
?> .srv(8888)
1
```

Start the client and make the connection to the server as 'f'.  In order to pass parameters to a remote function we form an array of the function symbol followed by the parameters (e.g. :avg,,!100)

```kgpy
?> f::.cli(8888) :" connect to the server
remote[localhost:8888]:fn
?> f(:avg,,!100) :" call the remote function avg directly with the parameter !100
49.5
```

Let's get fancy and make a local proxy to the remote function:

```kgpy
?> myavg::f(:avg) :" reference the remote function by it's symbol :avg and assign to a local variable called myavg
remote[localhost:8888]:fn:avg:monad
?> myavg(!100) :" this runs on the server with !100 array passed to it as a parameter
49.5
```

Since remote functions may take a while we can wrap them with an async wrapper and have it call our callback when completed:

```kgpy
?> afn::.async(myavg;{.d("Avg calculated: ");.p(x)})
async::monad
?> afn(!100)
Avg calculated: 49.5
1
```

## 6. Web Server Implementation

In addition to IPC we can also expose data via a standard web server.  This capability lets you have other ways of serving content that can be either exposing interesting details about some computation or just a simple web server for other reasons.

Let's create a file called web.kg with the following code that adds one index handler:

```text
.py("klongpy.web")
data::!10
index::{x; "Hello, Klong World! ",data}
.web(8888;:{},"/",index;:{})
.p("ready at http://localhost:8888")
```

We can run this web server as follows:

```bash
$ kgpy web.kg
ready at http://localhost:8888
```

In another terminal:

```bash
$ curl http://localhost:8888
['Hello, Klong World! ' 0 1 2 3 4 5 6 7 8 9]
```

## 7. Automatic Differentiation (Autograd)

KlongPy supports automatic differentiation, enabling gradient-based optimization and machine learning workflows.

### Numeric Gradient with `∇` (always numeric, any backend)

The `∇` operator **always** computes gradients using numeric differentiation:

```kgpy
?> f::{x^2}        :" Define f(x) = x^2
:monad
?> 3∇f             :" Compute f'(3) ≈ 6.0
6.0
```

### PyTorch Autograd with `:>` (recommended with torch backend)

Enable the torch backend for exact gradients:

```bash
$ USE_TORCH=1 kgpy
```

Use the `:>` operator for PyTorch autograd. The syntax is `function:>point`:

```kgpy
?> f::{x^2}        :" Define f(x) = x^2
:monad
?> f:>3            :" Compute f'(3) = 2*3 = 6
6.0
```

For vector-valued inputs, the gradient is computed element-wise:

```kgpy
?> h::{+/x^2}      :" h(x) = sum of squares
:monad
?> h:>[1.0 2.0 3.0]   :" Gradient: [2*1, 2*2, 2*3]
[2.0 4.0 6.0]
```

Simple gradient descent to minimize x^2:

```kgpy
?> f::{x^2}
:monad
?> x::5.0; lr::0.1
0.1
?> x::x-(lr*f:>x); x  :" One gradient step
4.0
?> x::x-(lr*f:>x); x  :" Another step
3.2
```

For complete examples including linear regression and neural networks, see the [autograd examples](https://github.com/briangu/klongpy/tree/main/examples/autograd).

## Conclusion

These examples are designed to illustrate the ease of use and diverse applications of Klong, making it a versatile choice for various programming needs.

Check out the references for details and deep dives on specific functionality.
