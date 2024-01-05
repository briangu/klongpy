import multiprocessing

def square(n):
    """Function to square a number."""
    return n * n

def runit(numbers):
    """Apply the square function in parallel to a list of numbers."""
    with multiprocessing.Pool(1) as pool:
        results = pool.map(square, numbers)
    return results
