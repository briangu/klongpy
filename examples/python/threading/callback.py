from multiprocessing.pool import ThreadPool
from klongpy.core import KGFnWrapper
import time

def use_klongpy(numbers, fn):
    """
    This runs in the same process as the KlongInterpreter, so we can use the fn directly.
    """
    return fn(numbers)


def runit(klong, numbers, fn):
    """Apply the square function in parallel to a list of numbers."""
    fn = KGFnWrapper(klong, fn) # TODO: this should already come wrapped from the interpreter
    with ThreadPool() as pool:
        return pool.apply_async(use_klongpy, (numbers, fn,)).get()
