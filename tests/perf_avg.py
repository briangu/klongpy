import time
from klongpy.backend import np
from klongpy import KlongInterpreter

klong = KlongInterpreter()

# define average function in Klong. Note, the '+/' (sum over) uses np.add.reduce under the hood
klong('avg::{+/x%#x}')

# create a billion random ints
data = np.random.rand(10**9)

# run Klong average function
start = time.perf_counter_ns()
r = klong['avg'](data)
stop = time.perf_counter_ns()

print(f"avg={np.round(r,6)} in {round((stop - start) / (10**9),6)} seconds")
