import time
from klongpy.backend import np
from klongpy import KlongInterpreter

# create a billion random ints
data = np.random.rand(10**9)
klong = KlongInterpreter()
# define average function in Klong
# Note the '+/' (sum over) uses np.add.reduce under the hood
klong('avg::{(+/x)%#x}')
# make Numpy generated data available in KlongPy as 'data' variable
klong['data'] = data
# run Klong average function
start = time.perf_counter_ns()
r = klong('avg(data)')
stop = time.perf_counter_ns()
seconds = (stop - start) / (10**9)
print(f"avg={np.round(r,6)} in {round(seconds,6)} seconds")
