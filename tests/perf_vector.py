import timeit

import numpy as np

from klongpy import KlongInterpreter


def numpy_vec(number=100):
    r = timeit.timeit(lambda: np.multiply(np.add(np.arange(10000000), 1), 2), number=number)
    return r/number


def klong_vec(number=100, backend=None, device=None):
    klong = KlongInterpreter(backend=backend, device=device)
    r = timeit.timeit(lambda: klong("2*1+!10000000"), number=number)
    return r/number, klong._backend


def python_vec(number=100):
    r = timeit.timeit(lambda: [2 * (1 + x) for x in range(10000000)], number=number)
    return r/number


def get_torch_devices():
    """Get list of available devices for torch backend."""
    devices = ["cpu"]
    try:
        import torch
        if torch.cuda.is_available():
            devices.append("cuda")
        if torch.backends.mps.is_available():
            devices.append("mps")
    except ImportError:
        return []
    return devices


def has_torch():
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    number = 1000

    print("Python: ", end='')
    pr = python_vec(number=number)
    print(f"{round(pr,6)}s")

    print("Numpy: ", end='')
    nr = numpy_vec(number=number)
    print(f"{round(nr,6)}s")

    # NumPy backend
    kr, klong_backend = klong_vec(number=number, backend='numpy')
    print(f"KlongPy (backend={klong_backend.name}): {round(kr,6)}s")
    print(f"  Python / KlongPy => {round(pr/kr,6)}")
    print(f"  Numpy / KlongPy => {round(nr/kr,6)}")

    # Torch backend (all available devices)
    if has_torch():
        for device in get_torch_devices():
            kr, klong_backend = klong_vec(number=number, backend='torch', device=device)
            print(f"KlongPy (backend={klong_backend.name}, device={device}): {round(kr,6)}s")
            print(f"  Python / KlongPy => {round(pr/kr,6)}")
            print(f"  Numpy / KlongPy => {round(nr/kr,6)}")
