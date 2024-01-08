from multiprocessing.pool import ThreadPool
from klongpy.core import KGFnWrapper
import time

def use_klongpy(numbers, fn):
    """
    This runs in the same process as the KlongInterpreter, so we can use the fn directly.
    """
    time.sleep(1)
    print("done sleeping")
    return fn(numbers)

pool = ThreadPool()

def runit(klong, numbers, fn, cb):
    """Apply the square function in parallel to a list of numbers."""
    fn = KGFnWrapper(klong, fn) # TODO: this should already come wrapped from the interpreter
    cb = KGFnWrapper(klong, cb)
    # with ThreadPool() as pool:
    return pool.apply_async(use_klongpy, (numbers, fn,), callback=cb)

def wait():
    pool.close()
    pool.join()
