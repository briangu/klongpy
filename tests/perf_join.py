import sys
from os import path
sys.path.append(path.abspath(path.join(path.dirname(__file__), '..')))

from utils import *
from klongpy.core import read_string
import timeit


def perf_join(k,number):
    a = "a" * k
    b = [x for x in a]
    r = timeit.timeit(lambda b=b: "".join(b), number=number)
    kr = r/number
    print(k,kr)
    return kr


def perf_read_string(s, number=1000):
    i = s.index('"')+1
    r = timeit.timeit(lambda t=s,i=i: read_string(t,i), number=number)
    kr = r/number
    print(s, kr)
    return kr



if __name__ == "__main__":
    perf_join(100,10000)
    perf_join(1000,10000)
    perf_join(10000,10000)

    t = '"hello"'
    perf_read_string(t)
    t = 'this is "hello" test'
    perf_read_string(t)
    t = 'this is "hello""""world" test'
    perf_read_string(t)
    t = 'this is "hello""""""world" test'
    perf_read_string(t)
    t = '""'
    perf_read_string(t)
    t = '""""'
    perf_read_string(t)
    t = '""""""'
    perf_read_string(t)
    t = '":["""";1;2]"'
    perf_read_string(t)
    t = 'A::""hello, world!""'
    perf_read_string(t)
