"""
KlongPy output formatting and writing functions.

This module contains all the functions for converting Klong values to strings:
- Symbol, number, character, string formatting
- List and dictionary formatting
- Array sorting utilities
"""
import sys

from .backend import np
from .types import (
    KGSym, KGChar, KGFn, KGChannel, KGChannelDir, KLONG_UNDEFINED,
    is_integer, is_float, is_list, is_iterable, is_empty,
    get_dtype_kind
)


def kg_write_symbol(x, display=False):
    return str(x) if display else f":{x}"


def kg_write_integer(x, display=False):
    return str(int(x))


def kg_write_float(x, display=False):
    return str(x)


def kg_write_char(c, display=False):
    return c if display else f"0c{c}"


def kg_write_string(s, display=False):
    if display:
        return s
    arr = ['"']
    for c in s:
        if c == '"':
            arr.append('"')
        arr.append(c)
    arr.append('"')
    return ''.join(arr)


def kg_write_dict(d, backend, display=False):
    # determine if the object d has overwritten the default __str__ and call it
    # if so, otherwise use the default dict str
    if d.__class__.__name__ != 'dict':
        return str(d)
    return ''.join([':{', ' '.join([kg_write(list(e), backend, display=display) for e in d.items()]), '}'])


def kg_write_list(x, backend, display=False):
    return ''.join(['[', ' '.join([kg_write(q, backend, display=display) for q in x]), ']'])


def kg_write_fn(x, display=False):
    return str(x)


def kg_write_channel(x, display=False):
    if x.channel_dir == KGChannelDir.INPUT:
        return ":inchan.0"
    return f":outchan.{2 if x.raw == sys.stderr else 1}"


def kg_write(a, backend, display=False):
    _backend = backend
    # Convert backend arrays (e.g., torch tensors) to display-friendly format
    a = _backend.to_display(a)
    if a is KLONG_UNDEFINED:
        return ":undefined"
    if isinstance(a, KGSym):
        return kg_write_symbol(a, display=display)
    elif is_integer(a, _backend):
        return kg_write_integer(a, display=display)
    elif is_float(a, _backend):
        return kg_write_float(a, display=display)
    elif isinstance(a, KGChar):
        return kg_write_char(a, display=display)
    elif isinstance(a, str):
        return kg_write_string(a, display=display)
    elif isinstance(a, dict):
        return kg_write_dict(a, _backend, display=display)
    elif is_list(a):
        return kg_write_list(a, _backend, display=display)
    elif isinstance(a, KGFn):
        return kg_write_fn(a, display=display)
    elif isinstance(a, KGChannel):
        return kg_write_channel(a, display=display)
    elif hasattr(a, '__str__'):
        return str(a)


def kg_argsort(a, backend, descending=False):
    """
    Return the indices of the sorted array (may be nested) or a string.
    Duplicate elements are disambiguated by their position in the array.

    argsort("foobar") => [4 3 0 1 2 5]
                                ^ ^
                            arbitrary ordering resolved by index position

    argsort("foobar",descending=True) => [5 2 1 0 3 4]
                                            ^ ^
                            arbitrary ordering resolved by index position
    """
    if not is_iterable(a) or len(a) == 0:
        return a

    # Fast path: for simple 1D numeric arrays, use native argsort
    if hasattr(a, 'ndim') and a.ndim == 1:
        dtype_kind = get_dtype_kind(a, backend)
        if dtype_kind in ('i', 'f', 'u'):
            return backend.argsort(a, descending=descending)

    # Slow path: nested arrays or strings need element-by-element comparison
    def _e(x):
        return (-np.inf, x) if is_empty(a[x]) else (np.max(a[x]), x) if is_list(a[x]) else (a[x], x)
    return np.asarray(sorted(range(len(a)), key=_e, reverse=descending))
