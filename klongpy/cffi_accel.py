"""
CFFI/C acceleration for KlongPy.

Provides native C implementations for:
- Sorting: counting sort, bucket sort, parallel merge sort
- Cumulative ops: cumsum, cumprod, running max/min
- Filtering: fused comparison + where/filter/count
- Unique, rank, inverse permutation
- Prefix scan for affine recurrences (EWMA)
- JIT-compiled fused element-wise and reduce expressions
"""
import math
import numpy
import operator as _op

# Lazy torch detection — avoids importing torch at module level
_torch_mod = None
_torch_checked = False

def _get_torch():
    global _torch_mod, _torch_checked
    if not _torch_checked:
        try:
            import torch
            _torch_mod = torch
        except ImportError:
            _torch_mod = None
        _torch_checked = True
    return _torch_mod

def _is_torch_tensor(a):
    t = _get_torch()
    return t is not None and isinstance(a, t.Tensor)


# ---------------------------------------------------------------------------
# Persistent thread pool for parallel operations (lazy-initialized)
# ---------------------------------------------------------------------------
_argsort_pool = None

def _get_pool():
    global _argsort_pool
    if _argsort_pool is None:
        from concurrent.futures import ThreadPoolExecutor
        _argsort_pool = ThreadPoolExecutor(max_workers=16)
    return _argsort_pool


# ---------------------------------------------------------------------------
# cffi utilities: lazy-compiled C library with native sort/scan/filter ops
# ---------------------------------------------------------------------------
_cffi_utils = None
_cffi_utils_checked = False

def _get_cffi_utils():
    global _cffi_utils, _cffi_utils_checked
    if _cffi_utils_checked:
        return _cffi_utils
    _cffi_utils_checked = True
    cffi = _get_cffi()
    if cffi is None:
        return None
    ffi = cffi.FFI()
    ffi.cdef('''
void cffi_running_max(const double* a, double* out, int64_t n);
void cffi_running_min(const double* a, double* out, int64_t n);
void cffi_cumsum(const double* a, double* out, int64_t n);
void cffi_cumprod(const double* a, double* out, int64_t n);
int64_t cffi_count_lt(const double* a, double val, int64_t n);
int64_t cffi_count_gt(const double* a, double val, int64_t n);
int64_t cffi_count_eq(const double* a, double val, int64_t n);
void cffi_minmax_i64(const int64_t* a, int64_t n, int64_t* out);
void cffi_counting_argsort(const int32_t* a, int64_t* out, int64_t n, int32_t mn, int64_t range);
void cffi_counting_argsort_i64(const int64_t* a, int64_t* out, int64_t n, int64_t mn, int64_t range);
void cffi_counting_argsort_u16(const uint16_t* keys, int64_t n, const int64_t* orig_idx, int64_t* out);
void cffi_counting_sort_values(const int64_t* a, int64_t* out, int64_t n, int64_t mn, int64_t range);
void cffi_inverse_perm(const int64_t* perm, int64_t* out, int64_t n);
int64_t cffi_unique_int(const int64_t* a, int64_t* out, int64_t n, int64_t mn, int64_t range);
void cffi_counting_rank(const int64_t* a, int64_t* out, int64_t n, int64_t mn, int64_t range);
void cffi_linear_scan(const double* x, double* out, int64_t n, double cx, double cy);
int64_t cffi_filter_gt(const double* a, double* out, int64_t n, double val);
int64_t cffi_filter_lt(const double* a, double* out, int64_t n, double val);
int64_t cffi_filter_eq(const double* a, double* out, int64_t n, double val);
int64_t cffi_where_gt(const double* a, int64_t* out, int64_t n, double val);
int64_t cffi_where_lt(const double* a, int64_t* out, int64_t n, double val);
int64_t cffi_where_eq(const double* a, int64_t* out, int64_t n, double val);
void cffi_bucket_prep_i64(const int64_t* a, uint8_t* bucket_ids, uint16_t* low16,
                           int64_t n, int64_t mn, int32_t shift);
int64_t cffi_find_bucket(const uint8_t* bucket_ids, int64_t* out, int64_t n, uint8_t target);
int64_t cffi_bucket_argsort(const uint8_t* bucket_ids, const uint16_t* low16,
                             int64_t* out, int64_t n, uint8_t target);
void cffi_add_scalar(double* a, int64_t n, double val);
''')
    try:
        lib = ffi.verify('''
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
void cffi_running_max(const double* a, double* out, int64_t n) {
    double mx = a[0]; out[0] = mx;
    for (int64_t i = 1; i < n; i++) { if (a[i] > mx) mx = a[i]; out[i] = mx; }
}
void cffi_running_min(const double* a, double* out, int64_t n) {
    double mn = a[0]; out[0] = mn;
    for (int64_t i = 1; i < n; i++) { if (a[i] < mn) mn = a[i]; out[i] = mn; }
}
void cffi_cumsum(const double* a, double* out, int64_t n) {
    double s = 0.0;
    for (int64_t i = 0; i < n; i++) { s += a[i]; out[i] = s; }
}
void cffi_cumprod(const double* a, double* out, int64_t n) {
    double p = 1.0;
    int64_t i;
    for (i = 0; i < n; i++) {
        p *= a[i]; out[i] = p;
        if (p == 0.0) { i++; break; }
    }
    if (i < n) memset(out + i, 0, (n - i) * sizeof(double));
}
int64_t cffi_count_lt(const double* a, double val, int64_t n) {
    int64_t c = 0; for (int64_t i = 0; i < n; i++) c += (a[i] < val); return c;
}
int64_t cffi_count_gt(const double* a, double val, int64_t n) {
    int64_t c = 0; for (int64_t i = 0; i < n; i++) c += (a[i] > val); return c;
}
int64_t cffi_count_eq(const double* a, double val, int64_t n) {
    int64_t c = 0; for (int64_t i = 0; i < n; i++) c += (a[i] == val); return c;
}
void cffi_minmax_i64(const int64_t* a, int64_t n, int64_t* out) {
    int64_t mn = a[0], mx = a[0];
    for (int64_t i = 1; i < n; i++) {
        int64_t v = a[i];
        if (v < mn) mn = v;
        if (v > mx) mx = v;
    }
    out[0] = mn;
    out[1] = mx;
}
void cffi_counting_argsort(const int32_t* a, int64_t* out, int64_t n, int32_t mn, int64_t range) {
    int32_t* counts = (int32_t*)calloc(range, sizeof(int32_t));
    for (int64_t i = 0; i < n; i++) counts[a[i] - mn]++;
    int32_t total = 0;
    for (int64_t v = 0; v < range; v++) { int32_t c = counts[v]; counts[v] = total; total += c; }
    for (int64_t i = 0; i < n; i++) out[counts[a[i] - mn]++] = i;
    free(counts);
}
void cffi_counting_argsort_i64(const int64_t* a, int64_t* out, int64_t n, int64_t mn, int64_t range) {
    int32_t* counts = (int32_t*)calloc(range, sizeof(int32_t));
    for (int64_t i = 0; i < n; i++) counts[a[i] - mn]++;
    int32_t total = 0;
    for (int64_t v = 0; v < range; v++) { int32_t c = counts[v]; counts[v] = total; total += c; }
    for (int64_t i = 0; i < n; i++) out[counts[a[i] - mn]++] = i;
    free(counts);
}
void cffi_counting_argsort_u16(const uint16_t* keys, int64_t n, const int64_t* orig_idx, int64_t* out) {
    int64_t* temp_idx = (int64_t*)malloc(n * sizeof(int64_t));
    uint16_t* temp_keys = (uint16_t*)malloc(n * sizeof(uint16_t));
    int32_t c0[256];
    memset(c0, 0, sizeof(c0));
    for (int64_t i = 0; i < n; i++) c0[keys[i] & 0xFF]++;
    int32_t t0 = 0;
    for (int32_t v = 0; v < 256; v++) { int32_t c = c0[v]; c0[v] = t0; t0 += c; }
    for (int64_t i = 0; i < n; i++) {
        int32_t p = c0[keys[i] & 0xFF]++;
        temp_idx[p] = orig_idx[i];
        temp_keys[p] = keys[i];
    }
    int32_t c1[256];
    memset(c1, 0, sizeof(c1));
    for (int64_t i = 0; i < n; i++) c1[temp_keys[i] >> 8]++;
    int32_t t1 = 0;
    for (int32_t v = 0; v < 256; v++) { int32_t c = c1[v]; c1[v] = t1; t1 += c; }
    for (int64_t i = 0; i < n; i++) out[c1[temp_keys[i] >> 8]++] = temp_idx[i];
    free(temp_idx);
    free(temp_keys);
}
void cffi_counting_sort_values(const int64_t* a, int64_t* out, int64_t n, int64_t mn, int64_t range) {
    int32_t* buf = (int32_t*)calloc(range, sizeof(int32_t));
    for (int64_t i = 0; i < n; i++) buf[a[i] - mn]++;
    int32_t total = 0;
    for (int64_t v = 0; v < range; v++) { int32_t c = buf[v]; buf[v] = total; total += c; }
    for (int64_t i = 0; i < n; i++) { int64_t v = a[i] - mn; out[buf[v]++] = a[i]; }
    free(buf);
}
void cffi_inverse_perm(const int64_t* perm, int64_t* out, int64_t n) {
    for (int64_t i = 0; i < n; i++) out[perm[i]] = i;
}
int64_t cffi_unique_int(const int64_t* a, int64_t* out, int64_t n, int64_t mn, int64_t range) {
    uint8_t* seen = (uint8_t*)calloc(range, sizeof(uint8_t));
    int64_t k = 0;
    for (int64_t i = 0; i < n; i++) {
        int64_t v = a[i] - mn;
        if (!seen[v]) { seen[v] = 1; out[k++] = a[i]; }
    }
    free(seen);
    return k;
}
void cffi_counting_rank(const int64_t* a, int64_t* out, int64_t n, int64_t mn, int64_t range) {
    int32_t* counts = (int32_t*)calloc(range, sizeof(int32_t));
    for (int64_t i = 0; i < n; i++) counts[a[i] - mn]++;
    int32_t total = 0;
    for (int64_t v = 0; v < range; v++) { int32_t c = counts[v]; counts[v] = total; total += c; }
    for (int64_t i = 0; i < n; i++) out[i] = counts[a[i] - mn]++;
    free(counts);
}
void cffi_linear_scan(const double* x, double* out, int64_t n, double cx, double cy) {
    out[0] = x[0];
    for (int64_t i = 1; i < n; i++) out[i] = cx * out[i-1] + cy * x[i];
}
int64_t cffi_filter_gt(const double* a, double* out, int64_t n, double val) {
    int64_t k = 0;
    for (int64_t i = 0; i < n; i++) { out[k] = a[i]; k += (a[i] > val); }
    return k;
}
int64_t cffi_filter_lt(const double* a, double* out, int64_t n, double val) {
    int64_t k = 0;
    for (int64_t i = 0; i < n; i++) { out[k] = a[i]; k += (a[i] < val); }
    return k;
}
int64_t cffi_filter_eq(const double* a, double* out, int64_t n, double val) {
    int64_t k = 0;
    for (int64_t i = 0; i < n; i++) { out[k] = a[i]; k += (a[i] == val); }
    return k;
}
int64_t cffi_where_gt(const double* a, int64_t* out, int64_t n, double val) {
    int64_t k = 0;
    for (int64_t i = 0; i < n; i++) { out[k] = i; k += (a[i] > val); }
    return k;
}
int64_t cffi_where_lt(const double* a, int64_t* out, int64_t n, double val) {
    int64_t k = 0;
    for (int64_t i = 0; i < n; i++) { out[k] = i; k += (a[i] < val); }
    return k;
}
int64_t cffi_where_eq(const double* a, int64_t* out, int64_t n, double val) {
    int64_t k = 0;
    for (int64_t i = 0; i < n; i++) { out[k] = i; k += (a[i] == val); }
    return k;
}
void cffi_bucket_prep_i64(const int64_t* a, uint8_t* bucket_ids, uint16_t* low16,
                           int64_t n, int64_t mn, int32_t shift) {
    uint32_t mask = (shift == 16) ? 0xFFFF : (uint32_t)((1 << shift) - 1);
    for (int64_t i = 0; i < n; i++) {
        uint32_t shifted = (uint32_t)((int32_t)(a[i] - mn));
        bucket_ids[i] = (uint8_t)(shifted >> shift);
        low16[i] = (uint16_t)(shifted & mask);
    }
}
int64_t cffi_find_bucket(const uint8_t* bucket_ids, int64_t* out, int64_t n, uint8_t target) {
    int64_t k = 0;
    for (int64_t i = 0; i < n; i++) { out[k] = i; k += (bucket_ids[i] == target); }
    return k;
}
int64_t cffi_bucket_argsort(const uint8_t* bucket_ids, const uint16_t* low16,
                             int64_t* out, int64_t n, uint8_t target) {
    int64_t k = 0;
    for (int64_t i = 0; i < n; i++) { out[k] = i; k += (bucket_ids[i] == target); }
    if (k <= 1) return k;
    uint16_t* keys = (uint16_t*)malloc(k * sizeof(uint16_t));
    for (int64_t i = 0; i < k; i++) keys[i] = low16[out[i]];
    int64_t* temp_idx = (int64_t*)malloc(k * sizeof(int64_t));
    uint16_t* temp_keys = (uint16_t*)malloc(k * sizeof(uint16_t));
    int32_t c0[256];
    memset(c0, 0, sizeof(c0));
    for (int64_t i = 0; i < k; i++) c0[keys[i] & 0xFF]++;
    int32_t t0 = 0;
    for (int32_t v = 0; v < 256; v++) { int32_t c = c0[v]; c0[v] = t0; t0 += c; }
    for (int64_t i = 0; i < k; i++) {
        int32_t p = c0[keys[i] & 0xFF]++;
        temp_idx[p] = out[i];
        temp_keys[p] = keys[i];
    }
    int32_t c1[256];
    memset(c1, 0, sizeof(c1));
    for (int64_t i = 0; i < k; i++) c1[temp_keys[i] >> 8]++;
    int32_t t1 = 0;
    for (int32_t v = 0; v < 256; v++) { int32_t c = c1[v]; c1[v] = t1; t1 += c; }
    for (int64_t i = 0; i < k; i++) out[c1[temp_keys[i] >> 8]++] = temp_idx[i];
    free(keys);
    free(temp_idx);
    free(temp_keys);
    return k;
}
void cffi_add_scalar(double* a, int64_t n, double val) {
    for (int64_t i = 0; i < n; i++) a[i] += val;
}
''', extra_compile_args=['-O3'])
        _cffi_utils = (ffi, lib)
    except Exception:
        pass
    return _cffi_utils


_minmax_out = None

def _cffi_minmax_i64(a):
    """Combined min/max in single pass via cffi."""
    global _minmax_out
    utils = _get_cffi_utils()
    if utils is not None and a.dtype == numpy.int64:
        ffi, lib = utils
        if _minmax_out is None:
            _minmax_out = numpy.empty(2, dtype=numpy.int64)
        lib.cffi_minmax_i64(ffi.from_buffer('int64_t[]', a), len(a),
                            ffi.from_buffer('int64_t[]', _minmax_out))
        return int(_minmax_out[0]), int(_minmax_out[1])
    return int(a.min()), int(a.max())


# ---------------------------------------------------------------------------
# cffi JIT for fused element-wise evaluation
# ---------------------------------------------------------------------------
_cffi_cache = {}
_cffi_mod = None
_cffi_checked = False

def _get_cffi():
    global _cffi_mod, _cffi_checked
    if not _cffi_checked:
        _cffi_checked = True
        try:
            import cffi
            _cffi_mod = cffi
        except ImportError:
            pass
    return _cffi_mod

def _python_expr_to_c(src, nvars, scalars=None):
    """Convert Python arithmetic expression to C with array indexing."""
    import ast
    tree = ast.parse(src, mode='eval')
    class CGen(ast.NodeVisitor):
        def visit_Expression(self, node):
            return self.visit(node.body)
        def visit_BinOp(self, node):
            left = self.visit(node.left)
            right = self.visit(node.right)
            if isinstance(node.op, ast.Pow):
                return f'pow({left},{right})'
            op_map = {ast.Add: '+', ast.Sub: '-', ast.Mult: '*', ast.Div: '/'}
            op_str = op_map.get(type(node.op))
            if op_str is None:
                raise ValueError(type(node.op).__name__)
            return f'({left}{op_str}{right})'
        def visit_Compare(self, node):
            if len(node.ops) != 1 or len(node.comparators) != 1:
                raise ValueError('multi-compare')
            left = self.visit(node.left)
            right = self.visit(node.comparators[0])
            cmp_map = {ast.Lt: '<', ast.Gt: '>', ast.LtE: '<=', ast.GtE: '>=', ast.Eq: '==', ast.NotEq: '!='}
            op_str = cmp_map.get(type(node.ops[0]))
            if op_str is None:
                raise ValueError(type(node.ops[0]).__name__)
            return f'({left}{op_str}{right})'
        def visit_UnaryOp(self, node):
            operand = self.visit(node.operand)
            if isinstance(node.op, ast.USub):
                return f'(-{operand})'
            return operand
        def visit_Name(self, node):
            if scalars and node.id in scalars:
                return node.id
            if node.id.startswith('_v') and node.id[2:].isdigit():
                return f'{node.id}[_i]'
            raise ValueError(node.id)
        def visit_Constant(self, node):
            if isinstance(node.value, (int, float)):
                return f'{float(node.value)}'
            raise ValueError(node.value)
        def generic_visit(self, node):
            raise ValueError(type(node).__name__)
    return CGen().visit(tree)

def _compile_cffi_fused(src, nvars):
    """Compile elementwise expression to fused C function via cffi."""
    cached = _cffi_cache.get(src)
    if cached is not None:
        return cached
    cffi = _get_cffi()
    if cffi is None:
        _cffi_cache[src] = False
        return False
    try:
        c_expr = _python_expr_to_c(src, nvars)
    except (ValueError, SyntaxError):
        _cffi_cache[src] = False
        return False
    params_c = ', '.join([f'const double* _v{i}' for i in range(nvars)] + ['double* _out', 'int64_t _n'])
    params_cdef = ', '.join(['const double*'] * nvars + ['double*', 'int64_t'])
    c_src = f'#include <stdint.h>\n#include <math.h>\nvoid _fused({params_c}) {{ for (int64_t _i = 0; _i < _n; _i++) {{ _out[_i] = {c_expr}; }} }}'
    ffi = cffi.FFI()
    ffi.cdef(f'void _fused({params_cdef});')
    try:
        lib = ffi.verify(c_src, extra_compile_args=['-O3'])
    except Exception:
        _cffi_cache[src] = False
        return False
    _cffi_cache[src] = (ffi, lib)
    return (ffi, lib)

# Reduce ops: identity values and C accumulator expressions
_CFFI_REDUCE_OPS = {
    '+': ('0.0', '_acc += %s;'),
    '*': ('1.0', '_acc *= %s;'),
    '|': ('-1.0/0.0', '{ double _v = %s; if (_v > _acc) _acc = _v; }'),
    '&': ('1.0/0.0', '{ double _v = %s; if (_v < _acc) _acc = _v; }'),
}

def _compile_cffi_reduce(inner_src, reduce_op, nvars):
    """Compile fused reduce: reduce_op / inner_expr(x) in a single C pass."""
    cache_key = f'__reduce_{reduce_op}_{inner_src}'
    cached = _cffi_cache.get(cache_key)
    if cached is not None:
        return cached
    cffi = _get_cffi()
    if cffi is None:
        _cffi_cache[cache_key] = False
        return False
    op_info = _CFFI_REDUCE_OPS.get(reduce_op)
    if op_info is None:
        _cffi_cache[cache_key] = False
        return False
    try:
        c_expr = _python_expr_to_c(inner_src, nvars)
    except (ValueError, SyntaxError):
        _cffi_cache[cache_key] = False
        return False
    identity, acc_tmpl = op_info
    acc_stmt = acc_tmpl % c_expr
    params_c = ', '.join([f'const double* _v{i}' for i in range(nvars)] + ['int64_t _n'])
    params_cdef = ', '.join(['const double*'] * nvars + ['int64_t'])
    c_src = f'#include <stdint.h>\n#include <math.h>\ndouble _reduce({params_c}) {{ double _acc = {identity}; for (int64_t _i = 0; _i < _n; _i++) {{ {acc_stmt} }} return _acc; }}'
    ffi = cffi.FFI()
    ffi.cdef(f'double _reduce({params_cdef});')
    try:
        lib = ffi.verify(c_src, extra_compile_args=['-O3', '-ffast-math'])
    except Exception:
        _cffi_cache[cache_key] = False
        return False
    _cffi_cache[cache_key] = (ffi, lib)
    return (ffi, lib)

_REDUCE_NP_FALLBACK = {'+': numpy.sum, '*': numpy.prod, '|': numpy.max, '&': numpy.min}

# Fused running_max/min + expression: single-pass C loop
_RUNNING_PATTERNS = {
    '_running_max(_v0)': ('>', '_v0[0]'),
    '_running_min(_v0)': ('<', '_v0[0]'),
}

def _compile_cffi_fused_running(src, mode='expr'):
    """Compile fused running_max/min + expression into single-pass C loop."""
    cache_key = f'__fused_running_{mode}_{src}'
    cached = _cffi_cache.get(cache_key)
    if cached is not None:
        return cached
    cffi = _get_cffi()
    if cffi is None:
        _cffi_cache[cache_key] = False
        return False
    pattern_found = None
    for pat, (cmp_op, init) in _RUNNING_PATTERNS.items():
        if pat in src:
            pattern_found = (pat, cmp_op, init)
            break
    if pattern_found is None:
        _cffi_cache[cache_key] = False
        return False
    pat, cmp_op, init = pattern_found
    c_src_expr = src.replace(pat, '_rm')
    try:
        c_expr = _python_expr_to_c(c_src_expr, 1, scalars={'_rm'})
    except (ValueError, SyntaxError):
        _cffi_cache[cache_key] = False
        return False
    ffi = cffi.FFI()
    if mode == 'expr':
        c_code = f'''#include <stdint.h>
#include <math.h>
void _fused(const double* _v0, double* _out, int64_t _n) {{
    double _rm = _v0[0];
    _out[0] = {c_expr.replace('_v0[_i]', '_v0[0]')};
    for (int64_t _i = 1; _i < _n; _i++) {{
        if (_v0[_i] {cmp_op} _rm) _rm = _v0[_i];
        _out[_i] = {c_expr};
    }}
}}'''
        ffi.cdef('void _fused(const double*, double*, int64_t);')
    elif mode.startswith('reduce_'):
        reduce_op = mode[7:]
        op_info = _CFFI_REDUCE_OPS.get(reduce_op)
        if op_info is None:
            _cffi_cache[cache_key] = False
            return False
        identity, acc_tmpl = op_info
        acc_stmt_0 = acc_tmpl % c_expr.replace('_v0[_i]', '_v0[0]')
        acc_stmt = acc_tmpl % c_expr
        c_code = f'''#include <stdint.h>
#include <math.h>
double _fused(const double* _v0, int64_t _n) {{
    double _rm = _v0[0];
    double _acc = {identity};
    {acc_stmt_0}
    for (int64_t _i = 1; _i < _n; _i++) {{
        if (_v0[_i] {cmp_op} _rm) _rm = _v0[_i];
        {acc_stmt}
    }}
    return _acc;
}}'''
        ffi.cdef('double _fused(const double*, int64_t);')
    else:
        _cffi_cache[cache_key] = False
        return False
    try:
        lib = ffi.verify(c_code, extra_compile_args=['-O3'])
    except Exception:
        _cffi_cache[cache_key] = False
        return False
    _cffi_cache[cache_key] = (ffi, lib)
    return (ffi, lib)

_CFFI_COUNT_FNS = {'<': 'cffi_count_lt', '>': 'cffi_count_gt', '==': 'cffi_count_eq'}


# ---------------------------------------------------------------------------
# Rank, sort, argsort
# ---------------------------------------------------------------------------

def _rank(a):
    """Efficient rank: argsort + inverse permutation."""
    if type(a) is numpy.ndarray and a.ndim == 1:
        dk = a.dtype.kind
        if dk == 'i' or dk == 'u':
            n = len(a)
            if n >= 1000:
                mn_i, mx_i = _cffi_minmax_i64(a) if a.dtype == numpy.int64 else (int(a.min()), int(a.max()))
                val_range = mx_i - mn_i + 1
                if val_range <= 2 * n:
                    utils = _get_cffi_utils()
                    if utils is not None:
                        ffi, lib = utils
                        a64 = a if a.dtype == numpy.int64 else a.astype(numpy.int64)
                        out = numpy.empty(n, dtype=numpy.int64)
                        lib.cffi_counting_rank(
                            ffi.cast('const int64_t*', a64.ctypes.data),
                            ffi.cast('int64_t*', out.ctypes.data),
                            n, mn_i, val_range)
                        return out
    idx = _argsort(a)
    n = len(idx)
    utils = _get_cffi_utils()
    if utils is not None and idx.dtype == numpy.int64:
        ffi, lib = utils
        rank = numpy.empty(n, dtype=numpy.int64)
        lib.cffi_inverse_perm(ffi.cast('const int64_t*', idx.ctypes.data),
                              ffi.cast('int64_t*', rank.ctypes.data), n)
        return rank
    rank = numpy.empty(n, dtype=numpy.intp)
    rank[idx] = numpy.arange(n)
    return rank

def _dotsum(a, b):
    """BLAS-optimized dot-sum."""
    return numpy.dot(a, b) if a.ndim == 1 else numpy.sum(a * b)

def _merge_sorted_indices(a, idx1, idx2, pool):
    combined = numpy.concatenate([idx1, idx2])
    order = numpy.argsort(a[combined], kind='stable')
    return combined[order]

def _bucket_find_and_sort(a, bucket_ids, b, bucket_min, use_u16):
    indices = numpy.flatnonzero(bucket_ids == b)
    if len(indices) <= 1:
        return indices
    if use_u16:
        rel_vals = (a[indices] - bucket_min).astype(numpy.uint16)
        return indices[numpy.argsort(rel_vals, kind='stable')]
    return indices[numpy.argsort(a[indices])]

def _bucket_sort_precomputed(low16, bucket_ids, b, buf_size=0):
    nn = len(bucket_ids)
    if nn >= 100_000:
        utils = _get_cffi_utils()
        if utils is not None:
            ffi, lib = utils
            alloc_n = buf_size if buf_size > 0 else nn
            buf = numpy.empty(alloc_n, dtype=numpy.int64)
            k = lib.cffi_bucket_argsort(
                ffi.cast('const uint8_t*', bucket_ids.ctypes.data),
                ffi.cast('const uint16_t*', low16.ctypes.data),
                ffi.cast('int64_t*', buf.ctypes.data), nn, b)
            if k >= alloc_n:
                buf = numpy.empty(nn, dtype=numpy.int64)
                k = lib.cffi_bucket_argsort(
                    ffi.cast('const uint8_t*', bucket_ids.ctypes.data),
                    ffi.cast('const uint16_t*', low16.ctypes.data),
                    ffi.cast('int64_t*', buf.ctypes.data), nn, b)
            return buf[:k]
    indices = numpy.flatnonzero(bucket_ids == b)
    n = len(indices)
    if n <= 1:
        return indices
    keys = low16[indices]
    if n >= 5000:
        utils = _get_cffi_utils()
        if utils is not None:
            ffi, lib = utils
            idx64 = indices if indices.dtype == numpy.int64 else indices.astype(numpy.int64)
            out = numpy.empty(n, dtype=numpy.int64)
            lib.cffi_counting_argsort_u16(
                ffi.cast('const uint16_t*', keys.ctypes.data), n,
                ffi.cast('const int64_t*', idx64.ctypes.data),
                ffi.cast('int64_t*', out.ctypes.data))
            return out
    return indices[numpy.argsort(keys, kind='stable')]

def _bucket_sort_values(a, low16, bucket_ids, b):
    indices = numpy.flatnonzero(bucket_ids == b)
    if len(indices) <= 1:
        return a[indices]
    order = numpy.argsort(low16[indices], kind='stable')
    return a[indices[order]]

def _argsort(a):
    """Fast argsort: cffi counting sort for small int, bucket sort for large int, parallel merge for floats."""
    if type(a) is numpy.ndarray:
        n = len(a)
        dk = a.dtype.kind
        _pre_mn = _pre_mx = None
        if (dk == 'i' or dk == 'u') and n >= 1_000:
            utils = _get_cffi_utils()
            if utils is not None:
                mn, mx = _cffi_minmax_i64(a) if a.dtype == numpy.int64 else (int(a.min()), int(a.max()))
                val_range = mx - mn + 1
                if val_range <= 2 * n and val_range <= 32_000:
                    ffi, lib = utils
                    out = numpy.empty(n, dtype=numpy.int64)
                    if a.dtype == numpy.int64:
                        lib.cffi_counting_argsort_i64(
                            ffi.cast('const int64_t*', a.ctypes.data),
                            ffi.cast('int64_t*', out.ctypes.data),
                            n, mn, val_range)
                    else:
                        a32 = a.astype(numpy.int32) if a.dtype != numpy.int32 else a
                        lib.cffi_counting_argsort(
                            ffi.cast('const int32_t*', a32.ctypes.data),
                            ffi.cast('int64_t*', out.ctypes.data),
                            n, numpy.int32(mn), val_range)
                    return out
                _pre_mn, _pre_mx = mn, mx
        if n >= 10_000:
            pool = _get_pool()
            if dk == 'i' or dk == 'u':
                if _pre_mn is not None:
                    mn, mx = _pre_mn, _pre_mx
                elif n >= 250_000:
                    f_mn = pool.submit(a.min)
                    f_mx = pool.submit(a.max)
                    mn, mx = int(f_mn.result()), int(f_mx.result())
                else:
                    mn, mx = int(a.min()), int(a.max())
                val_range = mx - mn
                shift = 16
                nbuckets = (val_range >> shift) + 1
                while nbuckets < 4 and shift > 12:
                    shift -= 1
                    nbuckets = (val_range >> shift) + 1
                if nbuckets <= 32:
                    if a.dtype == numpy.int64:
                        utils = _get_cffi_utils()
                        if utils is not None:
                            ffi, lib = utils
                            bucket_ids = numpy.empty(n, dtype=numpy.uint8)
                            low16 = numpy.empty(n, dtype=numpy.uint16)
                            lib.cffi_bucket_prep_i64(
                                ffi.cast('const int64_t*', a.ctypes.data),
                                ffi.cast('uint8_t*', bucket_ids.ctypes.data),
                                ffi.cast('uint16_t*', low16.ctypes.data),
                                n, mn, shift)
                            est = 2 * n // nbuckets + 2
                            futures = [pool.submit(_bucket_sort_precomputed, low16, bucket_ids, b, est) for b in range(nbuckets)]
                            result = numpy.empty(n, dtype=numpy.int64)
                            pos = 0
                            for f in futures:
                                chunk = f.result()
                                k = len(chunk)
                                result[pos:pos + k] = chunk
                                pos += k
                            return result
                    if a.dtype == numpy.int64 and mn >= -2147483648 and mx <= 2147483647:
                        a = a.astype(numpy.int32)
                    shifted = a - mn
                    bucket_ids = (shifted >> shift).astype(numpy.uint8)
                    if shift == 16:
                        low16 = shifted.astype(numpy.uint16)
                    else:
                        low16 = (shifted & ((1 << shift) - 1)).astype(numpy.uint16)
                    est = 2 * n // nbuckets + 2 if n >= 100_000 else 0
                    futures = [pool.submit(_bucket_sort_precomputed, low16, bucket_ids, b, est) for b in range(nbuckets)]
                else:
                    if a.dtype == numpy.int64 and mn >= -2147483648 and mx <= 2147483647:
                        a = a.astype(numpy.int32)
                    nbuckets = 16 if n >= 250_000 else 8
                    bucket_size = val_range // nbuckets + 1
                    use_u16 = bucket_size <= 65536
                    bucket_ids = ((a - mn) // bucket_size).astype(numpy.uint8)
                    futures = [pool.submit(_bucket_find_and_sort, a, bucket_ids, b, mn + b * bucket_size, use_u16) for b in range(nbuckets)]
                result = numpy.empty(n, dtype=numpy.int64)
                pos = 0
                for f in futures:
                    chunk = f.result()
                    k = len(chunk)
                    result[pos:pos + k] = chunk
                    pos += k
                return result
            # Float/other: parallel merge sort
            nways = 8 if n >= 250_000 else 4
            chunk = n // nways
            slices = [(i * chunk, (i + 1) * chunk if i < nways - 1 else n) for i in range(nways)]
            futures = [pool.submit(numpy.argsort, a[s:e]) for s, e in slices]
            sorted_indices = [f.result() + s for f, (s, e) in zip(futures, slices)]
            while len(sorted_indices) > 1:
                next_level = []
                merge_futures = []
                for i in range(0, len(sorted_indices), 2):
                    if i + 1 < len(sorted_indices):
                        merge_futures.append((len(next_level), pool.submit(
                            _merge_sorted_indices, a, sorted_indices[i], sorted_indices[i + 1], pool)))
                        next_level.append(None)
                    else:
                        next_level.append(sorted_indices[i])
                for pos, f in merge_futures:
                    next_level[pos] = f.result()
                sorted_indices = next_level
            return sorted_indices[0]
    return numpy.argsort(a)

def _fast_sort(a):
    """Fast sort: counting sort for small ranges, argsort-based for large arrays."""
    if type(a) is numpy.ndarray and len(a) >= 1000:
        dk = a.dtype.kind
        if dk == 'i' or dk == 'u':
            n = len(a)
            mn_i, mx_i = _cffi_minmax_i64(a) if a.dtype == numpy.int64 else (int(a.min()), int(a.max()))
            val_range = mx_i - mn_i + 1
            if val_range <= 2 * n and n <= 500_000:
                utils = _get_cffi_utils()
                if utils is not None:
                    ffi, lib = utils
                    a64 = a if a.dtype == numpy.int64 else a.astype(numpy.int64)
                    out = numpy.empty(n, dtype=numpy.int64)
                    lib.cffi_counting_sort_values(
                        ffi.cast('const int64_t*', a64.ctypes.data),
                        ffi.cast('int64_t*', out.ctypes.data),
                        n, mn_i, val_range)
                    return out
            if mn_i >= 0:
                if val_range <= 10 * n:
                    if n >= 50_000 and val_range >= n // 2:
                        pool = _get_pool()
                        if a.dtype == numpy.int64 and mn_i >= -2147483648 and mx_i <= 2147483647:
                            a = a.astype(numpy.int32)
                        shifted = a - mn_i
                        shift = 16
                        nbuckets = (val_range >> shift) + 1
                        while nbuckets < 4 and shift > 12:
                            shift -= 1
                            nbuckets = (val_range >> shift) + 1
                        bucket_ids = (shifted >> shift).astype(numpy.uint8)
                        if shift == 16:
                            low16 = shifted.astype(numpy.uint16)
                        else:
                            low16 = (shifted & ((1 << shift) - 1)).astype(numpy.uint16)
                        futures = [pool.submit(_bucket_sort_values, a, low16, bucket_ids, b) for b in range(nbuckets)]
                        return numpy.concatenate([f.result() for f in futures])
                    counts = numpy.bincount(a.ravel())
                    return numpy.repeat(numpy.arange(len(counts), dtype=a.dtype), counts)
    return numpy.sort(a)


# ---------------------------------------------------------------------------
# Prefix scan for affine recurrences
# ---------------------------------------------------------------------------

def _add_prefix(ls, p):
    return ls + p

def _prefix_scan_linear(x, c_x, c_y):
    """Affine recurrence y[n] = c_x * y[n-1] + c_y * x[n]."""
    n = len(x)
    if type(x) is numpy.ndarray:
        utils = _get_cffi_utils()
        if utils is not None:
            ffi, lib = utils
            xf = x if x.dtype == numpy.float64 else x.astype(numpy.float64)
            out = numpy.empty(n, dtype=numpy.float64)
            lib.cffi_linear_scan(ffi.from_buffer('double[]', xf),
                                 ffi.from_buffer('double[]', out), n, c_x, c_y)
            return out
    a_arr = numpy.full(n, c_x)
    a_arr[0] = 0.0
    b_arr = numpy.empty(n)
    b_arr[0] = x[0]
    numpy.multiply(c_y, x[1:], out=b_arr[1:])
    _tmp_ab = numpy.empty(n)
    _tmp_aa = numpy.empty(n)
    abs_cx = abs(c_x)
    if 0 < abs_cx < 1.0:
        max_passes = int(math.ceil(math.log2(max(1, 52 / (-math.log2(abs_cx)))))) + 1
    else:
        max_passes = 64
    d = 1
    pass_num = 0
    while d < n and pass_num < max_passes:
        k = n - d
        numpy.multiply(a_arr[d:d+k], b_arr[:k], out=_tmp_ab[:k])
        numpy.multiply(a_arr[d:d+k], a_arr[:k], out=_tmp_aa[:k])
        b_arr[d:d+k] += _tmp_ab[:k]
        a_arr[d:d+k] = _tmp_aa[:k]
        d *= 2
        pass_num += 1
    return b_arr


# ---------------------------------------------------------------------------
# Cumulative operations
# ---------------------------------------------------------------------------

def _cffi_cumsum_chunk(a_slice, out_slice, ffi, lib):
    lib.cffi_cumsum(ffi.from_buffer('double[]', a_slice),
                    ffi.from_buffer('double[]', out_slice), len(a_slice))

def _cumsum(a):
    """Cumulative sum: hybrid cffi+parallel for large float64, cffi serial for small."""
    if type(a) is numpy.ndarray:
        if a.dtype == numpy.float64:
            utils = _get_cffi_utils()
            if utils is not None:
                ffi, lib = utils
                n = len(a)
                if n >= 250_000:
                    pool = _get_pool()
                    nchunks = 4
                    chunk = n // nchunks
                    slices = [(i * chunk, (i + 1) * chunk if i < nchunks - 1 else n) for i in range(nchunks)]
                    out = numpy.empty(n, dtype=numpy.float64)
                    futures = [pool.submit(_cffi_cumsum_chunk, a[s:e], out[s:e], ffi, lib) for s, e in slices]
                    for f in futures: f.result()
                    running = 0.0
                    adjs = []
                    for i in range(nchunks - 1):
                        running += out[slices[i][1] - 1]
                        adjs.append(running)
                    for i in range(nchunks - 1):
                        out[slices[i+1][0]:slices[i+1][1]] += adjs[i]
                    return out
                out = numpy.empty(n, dtype=numpy.float64)
                lib.cffi_cumsum(ffi.from_buffer('double[]', a),
                                ffi.from_buffer('double[]', out), n)
                return out
        if len(a) >= 100_000:
            n = len(a)
            nchunks = 8
            chunk = n // nchunks
            slices = [(i * chunk, (i + 1) * chunk if i < nchunks - 1 else n) for i in range(nchunks)]
            pool = _get_pool()
            futures = [pool.submit(numpy.cumsum, a[s:e]) for s, e in slices]
            local_sums = [f.result() for f in futures]
            out = numpy.empty(n, dtype=a.dtype)
            out[slices[0][0]:slices[0][1]] = local_sums[0]
            prefixes = numpy.cumsum([ls[-1] for ls in local_sums[:-1]])
            for i in range(len(prefixes)):
                numpy.add(local_sums[i + 1], prefixes[i], out=out[slices[i + 1][0]:slices[i + 1][1]])
            return out
    return numpy.cumsum(a)

def _cumprod(a):
    """Cumulative product: cffi for float64, parallel numpy fallback for large non-float64."""
    if type(a) is numpy.ndarray:
        if a.dtype == numpy.float64:
            utils = _get_cffi_utils()
            if utils is not None:
                ffi, lib = utils
                out = numpy.empty(len(a), dtype=numpy.float64)
                lib.cffi_cumprod(ffi.from_buffer('double[]', a),
                                 ffi.from_buffer('double[]', out), len(a))
                return out
        if len(a) >= 50_000:
            n = len(a)
            nchunks = 4
            chunk = n // nchunks
            slices = [(i * chunk, (i + 1) * chunk if i < nchunks - 1 else n) for i in range(nchunks)]
            pool = _get_pool()
            futures = [pool.submit(numpy.cumprod, a[s:e]) for s, e in slices]
            local = [f.result() for f in futures]
            out = numpy.empty(n, dtype=a.dtype)
            out[slices[0][0]:slices[0][1]] = local[0]
            prefixes = numpy.cumprod([ls[-1] for ls in local[:-1]])
            for i in range(1, nchunks):
                s, e = slices[i]
                numpy.multiply(local[i], prefixes[i - 1], out=out[s:e])
            return out
    return numpy.cumprod(a)

def _running_max(a):
    """Running max: cffi for float64, parallel numpy fallback."""
    if type(a) is numpy.ndarray:
        if a.dtype == numpy.float64:
            utils = _get_cffi_utils()
            if utils is not None:
                ffi, lib = utils
                out = numpy.empty(len(a), dtype=numpy.float64)
                lib.cffi_running_max(ffi.from_buffer('double[]', a),
                                     ffi.from_buffer('double[]', out), len(a))
                return out
        if len(a) >= 50_000:
            n = len(a)
            nchunks = 4
            chunk = n // nchunks
            slices = [(i * chunk, (i + 1) * chunk if i < nchunks - 1 else n) for i in range(nchunks)]
            pool = _get_pool()
            futures = [pool.submit(numpy.maximum.accumulate, a[s:e]) for s, e in slices]
            local = [f.result() for f in futures]
            out = numpy.empty(n, dtype=a.dtype)
            out[slices[0][0]:slices[0][1]] = local[0]
            prefix_maxes = numpy.maximum.accumulate([ls[-1] for ls in local[:-1]])
            for i in range(1, nchunks):
                s, e = slices[i]
                numpy.maximum(local[i], prefix_maxes[i - 1], out=out[s:e])
            return out
    if _is_torch_tensor(a):
        return _get_torch().cummax(a, dim=0).values
    return numpy.maximum.accumulate(a)

def _running_min(a):
    """Running min: cffi for float64, parallel numpy fallback."""
    if type(a) is numpy.ndarray:
        if a.dtype == numpy.float64:
            utils = _get_cffi_utils()
            if utils is not None:
                ffi, lib = utils
                out = numpy.empty(len(a), dtype=numpy.float64)
                lib.cffi_running_min(ffi.from_buffer('double[]', a),
                                     ffi.from_buffer('double[]', out), len(a))
                return out
        if len(a) >= 50_000:
            n = len(a)
            nchunks = 4
            chunk = n // nchunks
            slices = [(i * chunk, (i + 1) * chunk if i < nchunks - 1 else n) for i in range(nchunks)]
            pool = _get_pool()
            futures = [pool.submit(numpy.minimum.accumulate, a[s:e]) for s, e in slices]
            local = [f.result() for f in futures]
            out = numpy.empty(n, dtype=a.dtype)
            out[slices[0][0]:slices[0][1]] = local[0]
            prefix_mins = numpy.minimum.accumulate([ls[-1] for ls in local[:-1]])
            for i in range(1, nchunks):
                s, e = slices[i]
                numpy.minimum(local[i], prefix_mins[i - 1], out=out[s:e])
            return out
    if _is_torch_tensor(a):
        return _get_torch().cummin(a, dim=0).values
    return numpy.minimum.accumulate(a)


# ---------------------------------------------------------------------------
# Parallel flatnonzero, fused where/filter/count
# ---------------------------------------------------------------------------

def _flatnonzero_chunk(mask, s, e):
    idx = numpy.flatnonzero(mask[s:e])
    idx += s
    return idx

def _flatnonzero(a):
    """Parallel flatnonzero for large boolean/integer arrays."""
    if type(a) is numpy.ndarray and len(a) >= 100_000:
        n = len(a)
        nchunks = 4
        chunk = n // nchunks
        slices = [(i * chunk, (i + 1) * chunk if i < nchunks - 1 else n) for i in range(nchunks)]
        pool = _get_pool()
        futures = [pool.submit(_flatnonzero_chunk, a, s, e) for s, e in slices]
        return numpy.concatenate([f.result() for f in futures])
    if _is_torch_tensor(a):
        return _get_torch().nonzero(a, as_tuple=False).flatten()
    return numpy.flatnonzero(a)

_CMP_FNS = {'<': numpy.less, '>': numpy.greater, '==': numpy.equal}
_OP_CMP_FNS = {'<': _op.lt, '>': _op.gt, '==': _op.eq}
_CFFI_WHERE_FNS = {'>': 'cffi_where_gt', '<': 'cffi_where_lt', '==': 'cffi_where_eq'}
_CFFI_FILTER_FNS = {'>': 'cffi_filter_gt', '<': 'cffi_filter_lt', '==': 'cffi_filter_eq'}

def _fused_where_chunk(a, cmp_fn, val, s, e):
    idx = numpy.flatnonzero(cmp_fn(a[s:e], val))
    idx += s
    return idx

def _fused_where(a, cmp_op, val):
    """Fused comparison + flatnonzero."""
    if type(a) is numpy.ndarray and a.dtype == numpy.float64 and len(a) >= 100_000:
        utils = _get_cffi_utils()
        if utils is not None and cmp_op in _CFFI_WHERE_FNS:
            ffi, lib = utils
            n = len(a)
            out = numpy.empty(n, dtype=numpy.int64)
            cfn = getattr(lib, _CFFI_WHERE_FNS[cmp_op])
            k = cfn(ffi.from_buffer('double[]', a), ffi.cast('int64_t*', out.ctypes.data), n, float(val))
            return out[:k]
    cmp_fn = _CMP_FNS[cmp_op]
    if type(a) is numpy.ndarray and len(a) >= 100_000:
        n = len(a)
        nchunks = 4
        chunk = n // nchunks
        slices = [(i * chunk, (i + 1) * chunk if i < nchunks - 1 else n) for i in range(nchunks)]
        pool = _get_pool()
        futures = [pool.submit(_fused_where_chunk, a, cmp_fn, val, s, e) for s, e in slices]
        return numpy.concatenate([f.result() for f in futures])
    if _is_torch_tensor(a):
        t = _get_torch()
        op_cmp = _OP_CMP_FNS[cmp_op]
        return t.nonzero(op_cmp(a, val), as_tuple=False).flatten()
    return numpy.flatnonzero(cmp_fn(a, val))

def _fused_filter_chunk(a, cmp_fn, val, s, e):
    chunk = a[s:e]
    idx = numpy.flatnonzero(cmp_fn(chunk, val))
    return chunk[idx]

def _fused_filter(a, cmp_op, val):
    """Fused comparison + filter."""
    if type(a) is numpy.ndarray and a.dtype == numpy.float64 and len(a) >= 100_000:
        utils = _get_cffi_utils()
        if utils is not None and cmp_op in _CFFI_FILTER_FNS:
            ffi, lib = utils
            n = len(a)
            out = numpy.empty(n, dtype=numpy.float64)
            cfn = getattr(lib, _CFFI_FILTER_FNS[cmp_op])
            k = cfn(ffi.from_buffer('double[]', a), ffi.from_buffer('double[]', out), n, float(val))
            return out[:k]
    cmp_fn = _CMP_FNS[cmp_op]
    if type(a) is numpy.ndarray and len(a) >= 100_000:
        n = len(a)
        nchunks = 6 if n >= 750_000 else 4
        chunk = n // nchunks
        slices = [(i * chunk, (i + 1) * chunk if i < nchunks - 1 else n) for i in range(nchunks)]
        pool = _get_pool()
        futures = [pool.submit(_fused_filter_chunk, a, cmp_fn, val, s, e) for s, e in slices]
        return numpy.concatenate([f.result() for f in futures])
    if _is_torch_tensor(a):
        op_cmp = _OP_CMP_FNS[cmp_op]
        return a[op_cmp(a, val)]
    idx = numpy.flatnonzero(cmp_fn(a, val))
    return a[idx]

def _fused_count_chunk(a, cmp_fn, val, s, e):
    return numpy.count_nonzero(cmp_fn(a[s:e], val))

def _fused_count(a, cmp_op, val):
    """Fused comparison + count."""
    if type(a) is numpy.ndarray and a.dtype == numpy.float64:
        utils = _get_cffi_utils()
        if utils is not None:
            ffi, lib = utils
            fn_name = _CFFI_COUNT_FNS.get(cmp_op)
            if fn_name is not None:
                cfn = getattr(lib, fn_name)
                return cfn(ffi.from_buffer('double[]', a), float(val), len(a))
    cmp_fn = _CMP_FNS[cmp_op]
    if type(a) is numpy.ndarray and len(a) >= 100_000:
        n = len(a)
        nchunks = 4
        chunk = n // nchunks
        slices = [(i * chunk, (i + 1) * chunk if i < nchunks - 1 else n) for i in range(nchunks)]
        pool = _get_pool()
        futures = [pool.submit(_fused_count_chunk, a, cmp_fn, val, s, e) for s, e in slices]
        return sum(f.result() for f in futures)
    if _is_torch_tensor(a):
        op_cmp = _OP_CMP_FNS[cmp_op]
        return int(op_cmp(a, val).sum().item())
    return int(numpy.count_nonzero(cmp_fn(a, val)))


# ---------------------------------------------------------------------------
# Unique
# ---------------------------------------------------------------------------

def _unique(a):
    """Fast unique preserving first-occurrence order for integer arrays."""
    if _is_torch_tensor(a):
        t = _get_torch()
        return t.unique(a)
    if type(a) is numpy.ndarray and a.ndim == 1:
        dk = a.dtype.kind
        if dk == 'i' or dk == 'u':
            n = len(a)
            if n > 0:
                mn, mx = _cffi_minmax_i64(a) if a.dtype == numpy.int64 else (int(a.min()), int(a.max()))
                val_range = mx - mn + 1
                if val_range <= max(10 * n, 1_000_000):
                    utils = _get_cffi_utils()
                    if utils is not None:
                        ffi, lib = utils
                        a64 = a if a.dtype == numpy.int64 else a.astype(numpy.int64)
                        out = numpy.empty(val_range, dtype=numpy.int64)
                        k = lib.cffi_unique_int(
                            ffi.cast('const int64_t*', a64.ctypes.data),
                            ffi.cast('int64_t*', out.ctypes.data),
                            n, mn, val_range)
                        return out[:k].astype(a.dtype)
                    first_pos = numpy.full(val_range, n, dtype=numpy.intp)
                    numpy.minimum.at(first_pos, a - mn, numpy.arange(n))
                    appeared = first_pos < n
                    unique_offsets = numpy.flatnonzero(appeared)
                    order = numpy.argsort(first_pos[unique_offsets])
                    return (unique_offsets[order] + mn).astype(a.dtype)
        _, ids = numpy.unique(a, return_index=True)
        ids.sort()
        return a[ids]
    return a


# ---------------------------------------------------------------------------
# Runtime fused reduce/running helpers
# ---------------------------------------------------------------------------

def _cffi_reduce_1(reduce_op, inner_src, a):
    """Runtime fused reduce for 1-variable expressions."""
    if type(a) is numpy.ndarray and a.dtype == numpy.float64 and len(a) > 0:
        result = _compile_cffi_reduce(inner_src, reduce_op, 1)
        if result:
            ffi, lib = result
            return lib._reduce(ffi.from_buffer('double[]', a), len(a))
    fb_globals = dict(_EVAL_GLOBALS)
    fb_globals['_v0'] = a
    inner_val = eval(inner_src, fb_globals)
    return _REDUCE_NP_FALLBACK[reduce_op](inner_val)

def _cffi_reduce_2(reduce_op, inner_src, a, b):
    """Runtime fused reduce for 2-variable expressions."""
    if type(a) is numpy.ndarray and a.dtype == numpy.float64 and \
       type(b) is numpy.ndarray and b.dtype == numpy.float64 and len(a) > 0:
        result = _compile_cffi_reduce(inner_src, reduce_op, 2)
        if result:
            ffi, lib = result
            return lib._reduce(ffi.from_buffer('double[]', a), ffi.from_buffer('double[]', b), len(a))
    fb_globals = dict(_EVAL_GLOBALS)
    fb_globals['_v0'] = a
    fb_globals['_v1'] = b
    inner_val = eval(inner_src, fb_globals)
    return _REDUCE_NP_FALLBACK[reduce_op](inner_val)

def _fused_running_expr(src, a):
    """Runtime: fused running_max/min + expression, single-pass C loop."""
    if type(a) is numpy.ndarray and a.dtype == numpy.float64 and len(a) > 0:
        result = _compile_cffi_fused_running(src, mode='expr')
        if result:
            ffi, lib = result
            out = numpy.empty(len(a), dtype=numpy.float64)
            lib._fused(ffi.from_buffer('double[]', a), ffi.from_buffer('double[]', out), len(a))
            return out
    fb_globals = dict(_EVAL_GLOBALS)
    fb_globals['_v0'] = a
    return eval(src, fb_globals)

def _fused_running_reduce(reduce_op, src, a):
    """Runtime: fused reduce + running_max/min + expression, single-pass C loop."""
    if type(a) is numpy.ndarray and a.dtype == numpy.float64 and len(a) > 0:
        result = _compile_cffi_fused_running(src, mode=f'reduce_{reduce_op}')
        if result:
            ffi, lib = result
            return lib._fused(ffi.from_buffer('double[]', a), len(a))
    fb_globals = dict(_EVAL_GLOBALS)
    fb_globals['_v0'] = a
    inner_val = eval(src, fb_globals)
    return _REDUCE_NP_FALLBACK[reduce_op](inner_val)


# ---------------------------------------------------------------------------
# Parallel element-wise eval
# ---------------------------------------------------------------------------

def _parallel_eval_2(fn, v0, v1):
    """Fused C eval for element-wise 2-arg expressions, numpy parallel fallback."""
    n = len(v0)
    src = getattr(fn, '_source', None)
    if src is not None and v0.dtype == numpy.float64 and v1.dtype == numpy.float64:
        cffi_result = _compile_cffi_fused(src, 2)
        if cffi_result:
            ffi, lib = cffi_result
            result = numpy.empty(n, dtype=numpy.float64)
            lib._fused(
                ffi.from_buffer('double[]', v0),
                ffi.from_buffer('double[]', v1),
                ffi.from_buffer('double[]', result),
                n
            )
            return result
    pool = _get_pool()
    nchunks = 6
    chunk = n // nchunks
    result = numpy.empty(n, dtype=numpy.float64)
    def _chunk_fn(s, e, out=result, f=fn, a=v0, b=v1):
        out[s:e] = f(a[s:e], b[s:e])
    slices = [(i * chunk, (i + 1) * chunk if i < nchunks - 1 else n) for i in range(nchunks)]
    futures = [pool.submit(_chunk_fn, s, e) for s, e in slices]
    for f in futures:
        f.result()
    return result


# ---------------------------------------------------------------------------
# Elementwise detection for parallel chunking
# ---------------------------------------------------------------------------

import re
_NON_ELEMENTWISE_NAMES = frozenset([
    '_argsort', '_fast_sort', '_rank', '_cumsum', '_cumprod',
    '_running_max', '_running_min', '_flatnonzero',
    '_fused_where', '_fused_filter', '_fused_count', '_unique', '_dotsum',
])
_NON_ELEMENTWISE_RE = re.compile(r'_(?!v\d)')

def _is_elementwise_source(src):
    """Check if compiled expression source is purely element-wise."""
    return '[' not in src and not _NON_ELEMENTWISE_RE.search(src)


# ---------------------------------------------------------------------------
# Globals dict for eval of compiled source
# ---------------------------------------------------------------------------
_EVAL_GLOBALS = {
    '_np': numpy,
    '_rank': _rank, '_dotsum': _dotsum,
    '_argsort': _argsort, '_fast_sort': _fast_sort,
    '_cumsum': _cumsum, '_cumprod': _cumprod,
    '_running_max': _running_max, '_running_min': _running_min,
    '_flatnonzero': _flatnonzero,
    '_fused_where': _fused_where, '_fused_filter': _fused_filter, '_fused_count': _fused_count,
    '_unique': _unique,
    '_cffi_reduce_1': _cffi_reduce_1, '_cffi_reduce_2': _cffi_reduce_2,
    '_fused_running_expr': _fused_running_expr, '_fused_running_reduce': _fused_running_reduce,
}
