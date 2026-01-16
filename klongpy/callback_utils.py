"""
Utilities for dynamic callback resolution.

This module provides helper functions to prevent lambda capture issues
when Klong callbacks are stored in Python closures. Instead of capturing
the callback directly, we capture its symbol name and re-resolve it on
each invocation.

This solves the issue where redefining a callback function in Klong doesn't
take effect because the old function reference is captured in Python closures.
"""

from klongpy.core import KGCall, KGFn, KGFnWrapper, reserved_dot_f_symbol, reserved_fn_symbols


def find_callback_symbol(klong, fn):
    """
    Find the symbol name for a KGFn in the current context.

    Args:
        klong: KlongInterpreter instance
        fn: Function to find (should be a KGFn)

    Returns:
        Symbol name (KGSym) if found, None otherwise.
        Skips reserved function symbols (x, y, z, .f).
    """
    if not isinstance(fn, KGFn) or isinstance(fn, KGCall):
        return None

    for sym, value in klong._context:
        if sym in reserved_fn_symbols or sym == reserved_dot_f_symbol:
            continue
        if value is fn:
            return sym
    return None


def coerce_callback(klong, fn):
    """
    Convert a callback function to a callable form.

    Args:
        klong: KlongInterpreter instance
        fn: Function to coerce

    Returns:
        - KGCall: Cannot be wrapped, return None
        - KGFn: Wrap with KGFnWrapper for invocation
        - Callable: Return as-is
        - Other: Return None
    """
    if isinstance(fn, KGCall):
        return None
    if isinstance(fn, KGFn):
        return KGFnWrapper(klong, fn)
    return fn if callable(fn) else None


def resolve_callback(klong, sym, fallback_fn):
    """
    Resolve a callback by symbol name with fallback.

    This is the core function for dynamic resolution. It tries to look up
    the current value of the symbol in the klong context. If the symbol
    is not found or the value is not callable, it falls back to the original
    function.

    Args:
        klong: KlongInterpreter instance
        sym: Symbol name (can be None)
        fallback_fn: Original function to use if symbol not found

    Returns:
        Resolved callable or None if nothing is callable
    """
    if sym is None:
        return coerce_callback(klong, fallback_fn)

    try:
        current = klong._context[sym]
    except KeyError:
        return coerce_callback(klong, fallback_fn)

    current_callback = coerce_callback(klong, current)
    return current_callback if current_callback is not None else coerce_callback(klong, fallback_fn)


def create_dynamic_callback(klong, fn):
    """
    Create a dynamic callback wrapper that re-resolves on each invocation.

    This is the primary function to use when storing callbacks that should
    respect redefinitions. It returns a wrapper function that looks up the
    current value of the symbol each time it's called.

    Args:
        klong: KlongInterpreter instance
        fn: Function to wrap

    Returns:
        Wrapper function that re-resolves dynamically, or the original
        function if it doesn't need dynamic resolution
    """
    sym = find_callback_symbol(klong, fn)
    fallback = coerce_callback(klong, fn)

    if sym is None or fallback is None:
        return fallback

    def dynamic_callback(*args, **kwargs):
        current_callback = resolve_callback(klong, sym, fn)
        if current_callback is None:
            raise RuntimeError(f"Callback symbol {sym} is no longer callable")
        return current_callback(*args, **kwargs)

    return dynamic_callback


def create_async_dynamic_callback(klong, fn):
    """
    Async version of create_dynamic_callback for async handlers.

    Args:
        klong: KlongInterpreter instance
        fn: Function to wrap

    Returns:
        Async wrapper function that re-resolves dynamically, or the original
        function if it doesn't need dynamic resolution
    """
    sym = find_callback_symbol(klong, fn)
    fallback = coerce_callback(klong, fn)

    if sym is None or fallback is None:
        return fallback

    async def dynamic_callback(*args, **kwargs):
        current_callback = resolve_callback(klong, sym, fn)
        if current_callback is None:
            raise RuntimeError(f"Callback symbol {sym} is no longer callable")
        return await current_callback(*args, **kwargs)

    return dynamic_callback
