import multiprocessing as mp

def square(n):
    """Function to square a number."""
    return n * n

def runit(numbers):
    """Apply the square function in parallel to a list of numbers."""
    with mp.Pool() as pool:
        results = pool.map(square, numbers)
    return results
