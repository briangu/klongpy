import multiprocessing
from klongpy import KlongInterpreter
from klongpy.core import KGFnWrapper

def use_klongpy(args):
    """
    This runs in a separate process so we need to instantiate a Klong interpreter.
    The incoming function was serialized from the main process and marshaled to the parallel process.
    We can run it in the new Klong context.
    """
    n, fn = args
    klong = KlongInterpreter()
    fn = KGFnWrapper(klong, fn)
    return fn(n)


def runit(numbers, fn):
    """Apply the square function in parallel to a list of numbers."""
    pool = multiprocessing.Pool(1)
    try:
        results = pool.map(use_klongpy, [(n,fn) for n in numbers])
    finally:
        # Let the worker finish and exit, rather than terminating it.
        pool.close()
        pool.join()
    return results
