import multiprocessing as mp

def square(n):
    """Function to square a number."""
    return n * n

def runit(numbers):
    """Apply the square function in parallel to a list of numbers."""
    pool = mp.Pool()
    try:
        results = pool.map(square, numbers)
    finally:
        # Let the workers finish and exit, rather than terminating them.
        pool.close()
        pool.join()
    return results
