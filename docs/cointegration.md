# Cointegration

KlongPy includes a small wrapper around the Johansen cointegration test from the Python `statsmodels` package.

Load the library and run the test on two price series:

```kg
.l("cointegration.kg")
prices1::[100 101 99 102 98]
prices2::[50 50.5 49 51 48]
result::johansen([prices1;prices2];0;1)
.p(result)
```

The function `johansen(data;det;k)` returns the object produced by `statsmodels.tsa.vector_ar.vecm.coint_johansen`.
