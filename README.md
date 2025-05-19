
![Unit Tests](https://github.com/briangu/klongpy/workflows/Unit%20Tests/badge.svg)
[![Last Commit](https://img.shields.io/github/last-commit/briangu/klongpy)](https://img.shields.io/github/last-commit/briangu/klongpy)
[![Dependency Status](https://img.shields.io/librariesio/github/briangu/klongpy)](https://libraries.io/github/briangu/klongpy)
[![Open Issues](https://img.shields.io/github/issues-raw/briangu/klongpy)](https://github.com/briangu/klongpy/issues)
[![Repo Size](https://img.shields.io/github/repo-size/briangu/klongpy)](https://img.shields.io/github/repo-size/briangu/klongpy)
[![GitHub star chart](https://img.shields.io/github/stars/briangu/klongpy?style=social)](https://star-history.com/#briangu/klongpy)

[![Release Notes](https://img.shields.io/github/release/briangu/klongpy)](https://github.com/briangu/klongpy/releases)
[![Downloads](https://static.pepy.tech/badge/klongpy/month)](https://pepy.tech/project/klongpy)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# KlongPy: High-Performance Array Programming in Python

KlongPy is a Python adaptation of the [Klong](https://t3x.org/klong) [array language](https://en.wikipedia.org/wiki/Array_programming), known for its high-performance vectorized operations that leverage the power of NumPy. Embracing a "batteries included" philosophy, KlongPy combines built-in modules with Python's expansive ecosystem, facilitating rapid application development with Klong's succinct syntax.

## Core Features

- **Vectorized Operations with NumPy:** At its core, KlongPy uses [NumPy](https://numpy.org/), an [Iverson Ghost](https://analyzethedatanotthedrivel.org/2018/03/31/NumPy-another-iverson-ghost/) descendant from APL, for high-efficiency array manipulations.
- **CPU and GPU Backend Support:** Incorporating [CuPy](https://github.com/cupy/cupy), KlongPy extends its capabilities to operate on both CPU and GPU backends, ensuring versatile and powerful computing options.
- **Seamless Integration with Python Ecosystem:** The combination of KlongPy's built-in features with Python's wide-ranging libraries enables developers to build complex applications effortlessly.

## KlongPy's Foundation and Applications

- **Inspired by Nils M Holm:** KlongPy is grounded in the work of [Nils M Holm](https://t3x.org), the original creator of Klong, and is further enriched by his [Klong Book](https://t3x.org/klong/book.html).
- **Ideal for Diverse Fields:** Data scientists, quantitative analysts, researchers, and programming language enthusiasts will find KlongPy especially beneficial for its versatility and performance.

KlongPy thus stands as a robust tool, blending the simplicity of Klong with the extensive capabilities of Python, suitable for a wide range of computational tasks.

# Quick install

```bash
pip3 install "klongpy[full]"
```

# Feature Overview

KlongPy is both an Array Language runtime and a set of powerful tools for building high performance data analysis and distributed computing applications.  Some of the features include:

* [__Array Programming__](https://en.wikipedia.org/wiki/Array_programming): Based on [Klong](https://t3x.org/klong), a concise, expressive, and easy-to-understand array programming language. Its simple syntax and rich feature set make it an excellent tool for data scientists and engineers.
* [__Speed__](docs/performance.md): Designed for high-speed vectorized computing, enabling you to process large data sets quickly and efficiently on either CPU or GPU.
* [__Fast Columnar Database__](docs/fast_columnar_database.md): Includes integration with [DuckDb](http://duckdb.org), a super fast in-process columnar store that can operate directly on NumPy arrays with zero-copy.
* [__Inter-Process Communication (IPC)__](docs/ipc_capabilities.md): Includes built-in support for IPC, enabling easy communication between different processes and systems. Ticker plants and similar pipelines are easy to build.
* **kdb+ Integration via qpython:** Experimental support for connecting to kdb+ processes using qpython.
* [__Table and Key-Value Store__](docs/table_and_key_value_stores.md): Includes a simple file-backed key value store that can be used to store database tables or raw key/value pairs.
* [__Python Integration__](docs/python_integration.md): Seamlessly compatible with Python and modules, allowing you to leverage existing Python libraries and frameworks.
* [__Web server__](docs/web_server.md): Includes a web server, making it easy to build sites backed by KlongPy capabilities.
* [__Timers__](docs/timer.md): Includes periodic timer facility to periodically perform tasks.

# KlongPy Examples

Explore KlongPy with these examples. Each snippet highlights a unique aspect of Klong, demonstrating its versatility in various programming scenarios.

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

The reason that Klong functions only take up to 3 parameters AND name them for you is both convience and compactness.

---

## 0. Start the REPL

```Bash
$ rlwrap kgpy

Welcome to KlongPy REPL v0.6.0
Author: Brian Guarraci
Web: http://klongpy.org
]h for help; crtl-d or ]q to quit

?>
```

## 1. Basic Arithmetic

Let's get started with the basics and build up to some more interesting math.  Expressions are evaluated from right to left: 3*2 and then + 5

```kgpy
?> 5+3*2
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

KlongPy integrates seamlessly with Python so that the strenghts of both can be combined.  It's easy to use KlongPy from Python and vice versa.

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

It doesn't make sense to write code in Klong that already exists in other libraries.  We can directly access them via the python inport functions (.py and .pyf).

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

KlongPy leverages a high-performance columnar store called DuckDb that uses zero-copy NumPy array operations behind the scenes.   This database allows fast interop between KlongPy and DuckDb (the arrays are not copied) so that applications can manage arrays in KlongPy and then instantly perform SQL on the data for deeper insights.

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

KlongPy treats IPC connections to servers as functions.  These functions let you call the server and ask for things it has in it's memory - they can be other functions or values, etc.  For example you can ask for a reference to a remote function and you will get a local function that when you call it runs on teh server with your arguemnts.  This general "remote proxy" approach allows you to write your client code in the same way as if all the code were running locally.

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
?> f(:avg,,!100) : call the remote function "avg" directly with the paramter !100
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

## Conclusion

These examples are designed to illustrate the "batteries included" approach, ease of use and diverse applications of KlongPy, making it a versatile choice for various programming needs.

[Check out the examples folder for more](https://github.com/briangu/klongpy/tree/main/examples).

# Installation

### CPU

```bash
pip3 install klongpy
```

### GPU support

Choose your CuPy prebuilt binary or from source.  Note, the [ROCM](docs/ROCM.md) support for CuPy is experimental and likely will have issues.

'cupy' => build from source
'cuda12x' => "cupy-cuda12x"
'cuda11x' => "cupy-cuda11x"
'cuda111' => "cupy-cuda111"
'cuda110' => "cupy-cuda110"
'cuda102' => "cupy-cuda102"
'rocm-5-0' => "cupy-rocm-5-0"
'rocm-4-3' => "cupy-rocm-4-3"

```bash
pip3 install "klongpy[cupy]"
```

### All application tools (db, web, REPL, etc.)

```bash
pip3 install "klongpy[full]"
```

# Status

KlongPy is a superset of the Klong array language.  It currently passes all of the integration tests provided by klong as well as additional suites.

Since CuPy is [not 100% compatible with NumPy](https://docs.cupy.dev/en/stable/user_guide/difference.html), there are currently some gaps in KlongPy between the two backends.  Notably, strings are supported in CuPy arrays so KlongPy GPU support currently is limited to math.

Primary ongoing work includes:

* Additional tools to make KlongPy applications more capable.
* Additional syntax error help
* Actively switch between CuPy and NumPy when incompatibilities are present

# Differences from Klong

KlongPy is effectively a superset of the Klong language, but has some key differences:

* Infinite precision: The main difference in this implementation of Klong is the lack of infinite precision.  By using NumPy we are restricted to doubles.
* Python integration: Most notably, the ".py" command allows direct import of Python modules into the current Klong context.
* KlongPy aims to be more "batteries included" approach to modules and contains additional features such as IPC, Web service, Websockets, etc.
* For array operations, KlongPy matches the shape of array arguments differently. Compare the results of an expression like `[1 2]+[[3 4][5 6]]` which in Klong produces `[[4 5] [7 8]]` but in KlongPy produces `[[4 6] [6 8]]`.

# Related

* [Klupyter - KlongPy in Jupyter Notebooks](https://github.com/briangu/klupyter)
* [Visual Studio Code Syntax Highlighting](https://github.com/briangu/klongpy-vscode)
* [Advent Of Code in KlongPy](https://github.com/briangu/aoc)

## Develop

git clone https://github.com/briangu/klongpy.git
cd klongpy
python3 setup.py develop

### Running tests

```bash
python3 -m unittest
```

# Acknowledgement

HUGE thanks to Nils M Holm for his work on Klong and providing the foundations for this interesting project.
