import timeit

from klongpy import KlongInterpreter


def parse_suite_file(fname, number=10):
    klong = KlongInterpreter()
    with open(f"tests/{fname}", "r") as f:
        d = f.read()
        b = len(d) * number
        r = timeit.timeit(lambda: klong.prog(d), number=number)
        return b, r, int(b / r), r / number


if __name__ == "__main__":
    number = 20
    b,r,bps,avg = parse_suite_file("test_join_over.kg", number=number)
    print(f"bytes processed: {b} time: {round(r,5)} bytes-per-second: {bps} time-per-pass: {round(avg,5)}")
