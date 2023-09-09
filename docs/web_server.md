# Web Server


KlongPy includes a simple web server module.  It's optional so you need to install the dependencies:

```bash
$ pip3 install klongpy[web]
```

The web server allows you to implement KlongPy functions as GET/POST handlers for registered routes.


Here's a simple example that lets you fetch and update a data array:

```
:" Import the Klongpy web module.  Requires pip3 install klongpy[web] first"
.py("klongpy.web")

:" Array of data to display"
data::[]

:" Return the data for a GET method at /"
index::{x;data}

:" Create the GET route handlers"
get:::{}
get,"/",index

:" Append the query param q value to data"
update::{[p];p::x?"p";.p(p);data::data,p}

:" Create the POST route handlers"
post:::{}
post,"/p",update

:" Start the web server with the GET and POST handlers"
.web(8888;get;post)

.p("curl -X POST -d""p=100"" ""http://localhost:8888/p""")
.p("curl ""http://localhost:8888""")

data
```

Test it out:

```bash
$ curl "http://localhost:8888"
[]
$ curl -X POST -d"p=100" "http://localhost:8888/p"
[100]
$ curl "http://localhost:8888"
[100]
```
