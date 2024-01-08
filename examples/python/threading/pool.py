from multiprocessing.pool import ThreadPool

def square(numbers):
    """Function to square a number."""
    return [n * n for n in numbers]


def runit(numbers):
    """Apply the square function in parallel to a list of numbers."""
    with ThreadPool() as pool:
        return pool.apply_async(square, (numbers,)).get()

