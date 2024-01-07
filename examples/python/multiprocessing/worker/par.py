import multiprocessing
from klongpy import KlongInterpreter
from klongpy.core import KGFnWrapper

def use_klongpy(args):
    """
    """
    n, fname, fn_name = args
    klong = KlongInterpreter()
    klong(f'.l("{fname}")')
    return klong[fn_name](n)


def runit(numbers, fname, fn_name):
    """
    """
    with multiprocessing.Pool(1) as pool:
        results = pool.map(use_klongpy, [(n,fname,fn_name) for n in numbers])
    return results
