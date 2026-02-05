"""
KlongPy core module.

This module re-exports public symbols from the split modules.
New code should import directly from the specific modules:
- klongpy.types: Type definitions and type checking
- klongpy.parser: Parsing and lexing functions
- klongpy.writer: Output formatting functions
"""

# Re-export everything from the split modules
from .types import *
from .parser import *
from .writer import *

# Re-export backend numpy-like module for shared array helpers
from .backend import bknp

__all__ = [
    # Types
    'KlongException',
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
    'merge_projections',
    'get_fn_arity',
    # Parser - character matching
    'cmatch',
    'cmatch2',
    'cpeek',
    'cpeek2',
    'cexpect',
    'cexpect2',
    'UnexpectedChar',
    'UnexpectedEOF',
    # Parser - skip
    'skip_space',
    'skip',
    # Parser - comment
    'read_shifted_comment',
    'read_sys_comment',
    # Parser - lexeme readers
    'read_num',
    'read_char',
    'read_sym',
    'read_op',
    'read_string',
    'read_list',
    'kg_read',
    'kg_read_array',
    'read_cond',
    'read_expr_array',
    'KGExprArray',
    'list_to_dict',
    'copy_lambda',
    'peek_adverb',
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
