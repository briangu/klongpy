import numpy as np
import pickle
import dill
import timeit
import zlib

# Create a large numpy array
#large_array = np.random.rand(10000, 10000)
large_array = np.random.rand(100000)

size_in_bytes = large_array.nbytes
print(f"Size of the numpy array: {size_in_bytes} bytes")
print()

# Define number of iterations
N = 10

# Pickle serialization
pickle_serialize_time = timeit.timeit(lambda: pickle.dumps(large_array), number=N)
print(f"Average pickle serialization time: {pickle_serialize_time / N} seconds")

# Pickle deserialization
pickle_data = pickle.dumps(large_array)
print("len(pickle_data): ",len(pickle_data))
pickle_deserialize_time = timeit.timeit(lambda: pickle.loads(pickle_data), number=N)
print(f"Average pickle deserialization time: {pickle_deserialize_time / N} seconds")
print()

# Dill serialization
dill_serialize_time = timeit.timeit(lambda: dill.dumps(large_array), number=N)
print(f"Average dill serialization time: {dill_serialize_time / N} seconds")

# Dill deserialization
dill_data = dill.dumps(large_array)
print("len(dill_data): ",len(dill_data))
dill_deserialize_time = timeit.timeit(lambda: dill.loads(dill_data), number=N)
print(f"Average dill deserialization time: {dill_deserialize_time / N} seconds")
print()

# Dill serialization with deflate
dill_serialize_time = timeit.timeit(lambda: zlib.compress(dill.dumps(large_array), level=0), number=N)
print(f"Average dill serialization time with deflate: {dill_serialize_time / N} seconds")
print()

# Dill deserialization with deflate
dill_data = zlib.compress(dill.dumps(large_array),level=0)
print("len(compressed_dill_data): ",len(dill_data))
dill_deserialize_time = timeit.timeit(lambda: dill.loads(zlib.decompress(dill_data)), number=N)
print(f"Average dill deserialization time with deflate: {dill_deserialize_time / N} seconds")
print()


