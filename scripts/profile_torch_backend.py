#!/usr/bin/python3
"""Simple performance profiler for the torch backend string shims."""

import os
import timeit
from klongpy import backend


def benchmark(strict: bool) -> float:
    os.environ["KLONGPY_TORCH_STRICT"] = "1" if strict else "0"
    backend.set_backend("torch")
    b = backend.current()
    x = b.array(list(range(1000)), dtype=float)

    def _op():
        b.mul(b.add(x, 1), b.add(x, 2))

    return timeit.timeit(_op, number=1000)


def main() -> None:
    no_check = benchmark(strict=True)
    with_check = benchmark(strict=False)
    print(f"Strict (no string check): {no_check:.4f}s")
    print(f"Default (with string check): {with_check:.4f}s")


if __name__ == "__main__":
    main()
