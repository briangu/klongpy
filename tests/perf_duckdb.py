import duckdb
import numpy as np
import pandas as pd
import time

N = 1000000

# Create a NumPy array
data = np.arange(N, dtype=int)
data2 = np.arange(N, dtype=int)

df = pd.DataFrame({'c': data, 'd': data2})

con = duckdb.connect(':memory:')

# Register the DataFrame as a DuckDB table
#start_t = time.time_ns()
#con.register('T', df)
#print("register: ", time.time_ns() - start_t)

def call_db(sql, T):
    start_t = time.time_ns()
    result = con.execute(f'SELECT * FROM T where c > {N-10} and d > {N-5}').fetchdf()
    return time.time_ns() - start_t

avgs = []
sql = f'SELECT * FROM df where c > {N-10} and d > {N-5}'
for i in range(1000):
    avgs.append(call_db(sql,df))
avg = sum(avgs)/len(avgs)
print(N, avg, avg / len(data))

