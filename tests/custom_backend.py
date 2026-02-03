import numpy as np

from klongpy.backends.numpy_backend import NumpyBackendProvider


class TestArray(np.ndarray):
    __array_priority__ = 1000
    __test__ = False

    def __new__(cls, input_array, dtype=None):
        if isinstance(input_array, TestArray):
            obj = input_array.view(cls)
        else:
            obj = np.asarray(input_array, dtype=dtype).view(cls)
        return obj

    def __array_finalize__(self, obj):
        pass


def _as_test_array(value):
    if isinstance(value, TestArray):
        return value
    if isinstance(value, np.ndarray):
        return value.view(TestArray)
    return np.asarray(value).view(TestArray)


class TestUfunc:
    def __init__(self, ufunc, wrap_result):
        self._ufunc = ufunc
        self._wrap_result = wrap_result

    def __call__(self, *args, **kwargs):
        return self._wrap_result(self._ufunc(*args, **kwargs))

    def reduce(self, *args, **kwargs):
        return self._wrap_result(self._ufunc.reduce(*args, **kwargs))

    def accumulate(self, *args, **kwargs):
        return self._wrap_result(self._ufunc.accumulate(*args, **kwargs))

    def __getattr__(self, name):
        return getattr(self._ufunc, name)


class TestNumpyModule:
    def __init__(self, base_module):
        self._np = base_module
        self.isarray = lambda x: isinstance(x, np.ndarray) and getattr(x, "ndim", 0) > 0

    def _wrap_result(self, result):
        if isinstance(result, np.ndarray) and not isinstance(result, TestArray):
            return result.view(TestArray)
        if isinstance(result, tuple):
            return tuple(self._wrap_result(item) for item in result)
        return result

    def __getattr__(self, name):
        attr = getattr(self._np, name)
        if isinstance(attr, np.ufunc):
            return TestUfunc(attr, self._wrap_result)
        if callable(attr):
            def wrapper(*args, **kwargs):
                return self._wrap_result(attr(*args, **kwargs))
            return wrapper
        return attr


class TestBackendProvider(NumpyBackendProvider):
    def __init__(self, device=None):
        super().__init__(device=device)
        self._np = TestNumpyModule(np)

    @property
    def name(self) -> str:
        return 'test_backend'

    def is_backend_array(self, x) -> bool:
        return isinstance(x, TestArray)

    def to_numpy(self, x):
        if isinstance(x, TestArray):
            return x.view(np.ndarray)
        return x

    def array_equal(self, a, b) -> bool:
        a_np = self.to_numpy(a)
        b_np = self.to_numpy(b)
        if isinstance(a_np, np.ndarray) and a_np.ndim == 0:
            a_np = a_np.item()
        if isinstance(b_np, np.ndarray) and b_np.ndim == 0:
            b_np = b_np.item()
        return super().kg_equal(a_np, b_np)

    def _is_list(self, x):
        if isinstance(x, np.ndarray):
            return x.ndim > 0 and x.size > 0
        if isinstance(x, (list, tuple)):
            return len(x) > 0
        return False

    def str_to_char_array(self, s):
        arr = super().str_to_char_array(s)
        return arr.view(TestArray)

    def kg_asarray(self, a):
        arr = super().kg_asarray(a)
        return _as_test_array(arr) if isinstance(arr, np.ndarray) else arr

    def argsort(self, a, descending=False):
        indices = self._np.argsort(a)
        if descending:
            indices = indices[::-1].copy()
        return indices

    def to_int_array(self, a):
        result = super().to_int_array(a)
        return _as_test_array(result) if isinstance(result, np.ndarray) else result

    def floor_to_int(self, a):
        result = super().floor_to_int(a)
        return _as_test_array(result) if isinstance(result, np.ndarray) else result

    def power(self, a, b):
        a_np = self.to_numpy(a)
        b_np = self.to_numpy(b)
        if isinstance(a_np, np.ndarray) and a_np.dtype.kind in ('i', 'u'):
            if isinstance(b_np, np.ndarray):
                if (b_np < 0).any():
                    a_np = a_np.astype(float)
            elif isinstance(b_np, (int, np.integer)) and b_np < 0:
                a_np = a_np.astype(float)
        elif isinstance(a_np, (int, np.integer)) and isinstance(b_np, np.ndarray):
            if (b_np < 0).any():
                a_np = float(a_np)
        elif isinstance(a_np, (int, np.integer)) and isinstance(b_np, (int, np.integer)) and b_np < 0:
            a_np = float(a_np)
        result = np.power(a_np, b_np)
        return _as_test_array(result) if isinstance(result, np.ndarray) else result


__all__ = ['TestBackendProvider', 'TestArray']
