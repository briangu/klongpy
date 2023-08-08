import timeit
import asyncio

# Non-async function with recursive calls
def normal_function(n):
    if n == 0:
        return 1
    return normal_function(n-1) + 1

# Async function with recursive calls
async def async_function(n):
    if n == 0:
        return 1
    return await async_function(n-1) + 1

# Wrapper function to run async function
def run_async_function():
    return asyncio.run(async_function(10))

# Measure the time taken by both functions
non_async_time = timeit.timeit("normal_function(10)", globals=globals(), number=100000)
async_time = timeit.timeit("run_async_function()", globals=globals(), number=100000)

print(f"Non-async time: {non_async_time:.6f} seconds")
print(f"Async time: {async_time:.6f} seconds")
