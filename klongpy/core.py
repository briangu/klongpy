"""
KlongPy core module.

This module re-exports public symbols from the split modules.
New code should import directly from the specific modules:
- klongpy.types: Type definitions and type checking
- klongpy.writer: Output formatting functions
"""

# Re-export the types and the writer. The reader of Klong text is the native engine's: nothing here parses.
from .types import *
from .writer import *

# Re-export backend numpy-like module for shared array helpers
from .backend import bknp

__all__ = [
    # Types
    'KlongException',
    'KlongError',
    'KlongSyntaxError',
    'KlongNameError',
    'KlongArityError',
    'KlongTypeError',
    'KlongShapeError',
    'KlongRangeError',
    'KlongSystemError',
    'KlongPythonError',
    'KlongInternalError',
    'KlongMemoryError',
    'KlongInterrupt',
    'KlongStaticError',
    'KlongRemoteError',
    'KGSym',
    'KGFn',
    'KGFnWrapper',
    'KGCall',
    'KGOp',
    'KGAdverb',
    'KGChar',
    'KGCond',
    'KGUndefined',
    'KLONG_UNDEFINED',
    'KGLambda',
    'KGChannel',
    'KGChannelDir',
    'RangeError',
    # Type constants
    'reserved_fn_args',
    'reserved_fn_symbols',
    'reserved_fn_symbol_map',
    'reserved_dot_f_symbol',
    'reserved_all_args_symbol',
    # Type helpers
    'get_fn_arity_str',
    'safe_inspect',
    # Type checking
    'is_list',
    'is_iterable',
    'is_empty',
    'is_dict',
    'to_list',
    'is_integer',
    'is_float',
    'is_number',
    'str_is_float',
    'is_symbolic',
    'is_char',
    'is_atom',
    'kg_truth',
    'str_to_chr_arr',
    'get_dtype_kind',
    # Utilities
    'safe_eq',
    'in_map',
    'has_none',
    'rec_flatten',
    # Adverb utilities
    'is_adverb',
    'get_adverb_arity',
    # Function utilities
    # Writer
    'kg_write_symbol',
    'kg_write_integer',
    'kg_write_float',
    'kg_write_char',
    'kg_write_string',
    'kg_write_dict',
    'kg_write_list',
    'kg_write_fn',
    'kg_write_channel',
    'kg_write',
    'kg_argsort',
    # Backend re-exports
    'bknp',
]
