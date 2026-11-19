//! SPIKE: the native Klong interpreter as a Python extension (module `klong_native`).
//!
//! * `Klong(strict, types)`: an interpreter. `run(src)`, `eval(src)`, `call(name, *args)`, `klong[name]`, `klong[name] = value`.
//! * Native numeric vectors come back as NumPy arrays: copies by default, or views of the interpreter's memory (`zero_copy(True)`,
//!   read only, through `__array_interface__`).
//! * A 1-d numeric NumPy array of at least `host_threshold` elements handed to the interpreter is kept as an opaque HOST array
//!   (a `Value::Ext`): `+ - * % ^ & | < > = ...`, a few monads and the reductions `+/ */ &/ |/` are forwarded to NumPy; every other use
//!   copies it into a native vector first. Smaller arrays and other dtypes are copied into native vectors on the way in.
//! * Python callables become system functions; they may call back into the interpreter.
use std::cell::{Cell, RefCell, UnsafeCell};
use std::collections::HashMap;
use std::sync::Mutex;
use std::rc::Rc;

use klong_core::value::{Arr, Dict, Ext, Func, Int};
use klong_core::{Config, Error, Interp, Value};
use klong_host::{Console, HostPlatform};
use klong_rt::Runtime;
use pyo3::exceptions::{PyException, PyOverflowError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple};

pyo3::create_exception!(klong_native, NativeError, PyException);

// ------------------------------------------------------------------------------------------------
// per-thread host tables (NumPy callables) and the Python types of KlongPy
// ------------------------------------------------------------------------------------------------

struct Host {
    ndarray: Py<PyAny>,
    asarray: Py<PyAny>,
    array: Py<PyAny>,
    int64: Py<PyAny>,
    float64: Py<PyAny>,
    ascontig: Py<PyAny>,
    mk_obj: Py<PyAny>,
    kdict: Py<PyAny>,
    views: Py<PyAny>,
    /// The table used by an interpreter that was given none: NumPy only (see `DefaultTable` in HOST_PY).
    default_table: Py<PyAny>,
    kgsym: Py<PyAny>,
    kgchar: Py<PyAny>,
    undefined: Py<PyAny>,
    threshold: std::sync::atomic::AtomicUsize,
}

static HOST: std::sync::OnceLock<&'static Host> = std::sync::OnceLock::new();
static ZERO_COPY: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

thread_local! {
    /// A Python exception raised inside a callback, to be re-raised at the boundary with its type and traceback.
    static PENDING: RefCell<Option<PyErr>> = const { RefCell::new(None) };
    /// The interpreter that is running (for conversions that need its symbol table and have no `ip` at hand).
    static CURRENT: Cell<*mut Interp> = const { Cell::new(std::ptr::null_mut()) };
    static OWNER: RefCell<Option<Py<Klong>>> = const { RefCell::new(None) };
    /// While set, a Python value converts to a NATIVE value only (an array is copied, never kept as a host value).
    static NATIVE_ONLY: Cell<bool> = const { Cell::new(false) };
}

// The callable table of a backend (what the embedder supplies; see klongpy/rust_host.py). The extension knows nothing of the
// backend's array type: it holds the object and asks the table. Methods, each given the glyph or object and the operands as Python
// values (a native vector is a NumPy array, a number a Python number, a host object itself):
//   is_host(o) -> bool             is this Python object a host value (a tensor)?
//   monad(glyph, x), dyad(glyph, x, y) -> value or NotImplemented
//   adverb(glyph, f, op, x, y) -> value or NotImplemented  (f: the function the adverb modifies, op: its operator text or None)
//   to_native(obj, what) -> a NumPy array or Python value to be copied into native form (raises when that would be wrong)
//   converged(p, q) -> bool or None (the converging adverbs),
//   items(obj) -> an iterable of the elements, or None,
//   truthy(obj) -> bool or None, format(obj, display) -> str or None, collect(items) -> value or None
// The default table below is what an interpreter has when none was installed: NumPy arrays of 256 or more elements.
const HOST_PY: &str = r#"
import numpy as np
i64 = np.int64
import weakref
from collections.abc import KeysView, ValuesView, ItemsView
views = weakref.WeakValueDictionary()
def _same(a, b):
    r = (a == b)
    if isinstance(r, np.ndarray):
        return np.shape(a) == np.shape(b) and bool(r.all())
    return bool(r)
class KDict(dict):
    """A native dictionary as a Python mapping: live (reads and writes go to the interpreter's dictionary), insertion ordered.
    A dict subclass so that isinstance(x, dict) holds; every method is the native dictionary's."""
    def __init__(self, h):
        dict.__init__(self)
        self._h = h
    def __getitem__(self, k): return self._h.get(k)
    def __setitem__(self, k, v): self._h.set(k, v)
    def __delitem__(self, k): self._h.delete(k)
    def __contains__(self, k): return self._h.contains(k)
    def __iter__(self): return iter(self._h.keys())
    def __reversed__(self): return reversed(self._h.keys())
    def __len__(self): return self._h.len()
    def keys(self): return KeysView(self)
    def values(self): return ValuesView(self)
    def items(self): return ItemsView(self)
    def get(self, k, default=None):
        return self._h.get(k) if self._h.contains(k) else default
    def pop(self, k, *d):
        if self._h.contains(k):
            v = self._h.get(k); self._h.delete(k); return v
        if d: return d[0]
        raise KeyError(k)
    def popitem(self):
        ks = self._h.keys()
        if not ks: raise KeyError("popitem(): dictionary is empty")
        k = ks[-1]; v = self._h.get(k); self._h.delete(k); return (k, v)
    def setdefault(self, k, default=None):
        if not self._h.contains(k): self._h.set(k, default)
        return self._h.get(k)
    def update(self, *a, **kw):
        for k, v in dict(*a, **kw).items(): self._h.set(k, v)
    def clear(self): self._h.clear()
    def copy(self): return dict(self.items())
    __copy__ = copy
    def __deepcopy__(self, memo):
        import copy
        return copy.deepcopy(dict(self.items()), memo)
    def __reduce__(self): return (dict, (dict(self.items()),))
    def __eq__(self, o):
        if not isinstance(o, dict): return NotImplemented
        if len(self) != len(o): return False
        for k in self._h.keys():
            if k not in o or not _same(self._h.get(k), o[k]): return False
        return True
    def __ne__(self, o):
        r = self.__eq__(o)
        return r if r is NotImplemented else not r
    __hash__ = None
    def __repr__(self): return repr(dict(self.items()))
    __str__ = __repr__
    def __or__(self, o): d = dict(self.items()); d.update(o); return d
    def __ror__(self, o): d = dict(o); d.update(self.items()); return d
    def __ior__(self, o): self.update(o); return self
# KlongPy's writer tells a plain dict by its class NAME (writer.py: `d.__class__.__name__ != 'dict'` prints a dict whose __str__ was
# overridden with str()): the view is written as a dictionary
KDict.__name__ = KDict.__qualname__ = 'dict'
KDict.__doc__ = dict.__doc__   # the view is a dict to help() and to inspect
def mk_obj(items):
    a = np.empty(len(items), dtype=object)
    for i, x in enumerate(items):
        a[i] = x
    return a
def _cmp(f):
    return lambda a, b: f(a, b).astype(i64)
_dyad = {
    '+': np.add, '-': np.subtract, '*': np.multiply, '%': np.true_divide, '^': np.power,
    '&': np.minimum, '|': np.maximum, ':%': np.floor_divide, '!': np.mod,
    '<': _cmp(np.less), '>': _cmp(np.greater), '=': _cmp(np.equal),
    '@': lambda a, i: a[i],
    ',': lambda a, b: np.concatenate((np.atleast_1d(a), np.atleast_1d(b))),
    '#': lambda n, a: (a[:n] if n >= 0 else a[n:]) if abs(n) <= len(a) else np.resize(a, abs(n)),
    '_': lambda n, a: a[n:] if n >= 0 else a[:n],
}
_monad = {
    '-': np.negative,
    '<': lambda a: np.argsort(a, kind='stable'),
    '>': lambda a: np.argsort(-a, kind='stable'),
    '|': lambda a: a[::-1],
    '#': lambda a: len(a),
    '*': lambda a: a[0],
    '_': lambda a: np.floor(a).astype(i64),
    '%': lambda a: 1.0 / a,
}
_reduce = {'+': np.add.reduce, '*': np.multiply.reduce, '&': np.minimum.reduce, '|': np.maximum.reduce}
def _num(v):
    return (type(v) in (int, float)) or (type(v) is np.ndarray and v.ndim == 1 and v.dtype.kind in 'if')
class DefaultTable:
    def is_host(self, o):
        return False
    def monad(self, g, x):
        f = _monad.get(g)
        return NotImplemented if f is None else f(x)
    def dyad(self, g, x, y):
        f = _dyad.get(g)
        if f is None:
            return NotImplemented
        ok = (type(x) is np.ndarray and _num(y)) if g == '@' else (type(x) is int and type(y) is np.ndarray) if g in ('#', '_') else (_num(x) and _num(y))
        return f(x, y) if ok else NotImplemented
    def adverb(self, g, f, op, x, y):
        if g == '/' and y is None and op in _reduce and len(x):
            return _reduce[op](x)
        return NotImplemented
    def to_native(self, o, what):
        return o
    def truthy(self, o):
        return None
    def converged(self, p, q):
        return None
    def items(self, o):
        return None
    def format(self, o, display):
        return None
    def collect(self, items):
        return None
"#;

fn host(_py: Python<'_>) -> PyResult<&'static Host> {
    HOST.get().copied().ok_or_else(|| PyTypeError::new_err("klong_native: no interpreter has been created"))
}

fn init_host(py: Python<'_>, types: &Bound<'_, PyDict>) -> PyResult<&'static Host> {
    if let Some(h) = HOST.get() {
        return Ok(h);
    }
    let np = py.import("numpy")?;
    let m = PyModule::from_code(py, std::ffi::CString::new(HOST_PY).unwrap().as_c_str(), c"host.py", c"klong_host_tables")?;
    let ty = |k: &str| -> PyResult<Py<PyAny>> { Ok(types.get_item(k)?.ok_or_else(|| PyTypeError::new_err(format!("types[{k:?}] missing")))?.unbind()) };
    let h: &'static Host = Box::leak(Box::new(Host {
        ndarray: np.getattr("ndarray")?.unbind(),
        asarray: np.getattr("asarray")?.unbind(),
        array: np.getattr("array")?.unbind(),
        int64: np.getattr("int64")?.unbind(),
        float64: np.getattr("float64")?.unbind(),
        ascontig: np.getattr("ascontiguousarray")?.unbind(),
        mk_obj: m.getattr("mk_obj")?.unbind(),
        kdict: m.getattr("KDict")?.unbind(),
        views: m.getattr("views")?.unbind(),
        default_table: m.getattr("DefaultTable")?.call0()?.unbind(),
        kgsym: ty("KGSym")?,
        kgchar: ty("KGChar")?,
        undefined: ty("undefined")?,
        threshold: std::sync::atomic::AtomicUsize::new(256),
    }));
    Ok(HOST.get_or_init(|| h))
}

// ------------------------------------------------------------------------------------------------
// errors
// ------------------------------------------------------------------------------------------------

/// A span of byte offsets into `text` as offsets in characters, which is how Python indexes a str.
fn char_span(text: &str, a: usize, b: usize) -> (i64, i64) {
    if text.is_ascii() {
        return (a as i64, b as i64);
    }
    let co = klong_tools::CharOffsets::new(text);
    (co.at(a) as i64, co.at(b) as i64)
}

/// A native error as a Python exception: args = (kind, message, start, end, source name, source text, hint); start and end are offsets in
/// characters of the source text.
fn k2py(e: Error) -> PyErr {
    if let Some(p) = PENDING.with(|p| p.borrow_mut().take()) {
        // the span of the expression that called the Python function that failed: `_klong_span` = (start, end, source name,
        // source text), set once (the innermost call)
        if let Some((a, b)) = e.span {
            Python::with_gil(|py| {
                let v = p.value(py);
                if !v.hasattr("_klong_span").unwrap_or(true) {
                    let (name, text) = match &e.source {
                        Some(s) => (s.name.clone(), s.text.to_string()),
                        None => (String::new(), String::new()),
                    };
                    let (ca, cb) = char_span(&text, a as usize, b as usize);
                    let _ = v.setattr("_klong_span", (ca, cb, name, text));
                }
            });
        }
        return p;
    }
    let (name, text) = match &e.source {
        Some(s) => (s.name.clone(), s.text.to_string()),
        None => (String::new(), String::new()),
    };
    let (a, b) = e.span.map(|(a, b)| char_span(&text, a as usize, b as usize)).unwrap_or((-1, -1));
    NativeError::new_err((e.kind.name().to_string(), e.message.to_string(), a, b, name, text, e.hint.as_ref().map(|h| h.to_string())))
}

fn py2k(e: PyErr) -> Error {
    let msg = e.to_string();
    let k = Python::with_gil(|py| {
        if e.is_instance_of::<pyo3::exceptions::PyValueError>(py) {
            Error::shape(msg)
        } else if e.is_instance_of::<PyTypeError>(py) {
            Error::typ(msg)
        } else {
            Error::system(msg)
        }
    });
    PENDING.with(|p| *p.borrow_mut() = Some(e));
    k
}

// ------------------------------------------------------------------------------------------------
// objects handed to Python
// ------------------------------------------------------------------------------------------------

/// A native numeric vector, seen by NumPy through `__array_interface__` without a copy. Holds the `Rc`, so the
/// interpreter can never mutate the data in place while Python has it.
#[pyclass(name = "NativeVec")]
struct NativeVec {
    v: Value,
}

// SAFETY (these three): the interpreter and its values use `Rc`, which is not thread safe. Every access happens with the GIL held, so
// two threads never touch them at the same time; that is the discipline KlongPy's own interpreter has. Thread-local state
// (CURRENT, OWNER, PENDING) is per thread on purpose.
unsafe impl Send for NativeVec {}
unsafe impl Sync for NativeVec {}
unsafe impl Send for KFn {}
unsafe impl Sync for KFn {}

#[pymethods]
impl NativeVec {
    #[getter]
    fn __array_interface__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        match &self.v {
            Value::Ints(a) => {
                d.set_item("data", (a.data.as_ptr() as usize, true))?;
                d.set_item("typestr", "<i8")?;
                d.set_item("shape", PyTuple::new(py, shape_of(a))?)?;
            }
            Value::Flts(a) => {
                d.set_item("data", (a.data.as_ptr() as usize, true))?;
                d.set_item("typestr", "<f8")?;
                d.set_item("shape", PyTuple::new(py, shape_of(a))?)?;
            }
            _ => return Err(PyTypeError::new_err("not a numeric vector")),
        }
        d.set_item("version", 3)?;
        Ok(d)
    }
}

fn shape_of<T: Copy>(a: &Arr<T>) -> Vec<usize> {
    if a.rank() <= 1 {
        vec![a.data.len()]
    } else {
        (0..a.rank()).map(|i| a.dim(i)).collect()
    }
}

/// A native object that has no Python form (a channel, an IPC handle): opaque, and given back as the same value.
#[pyclass(name = "NativeObject")]
struct NativeObject {
    v: Value,
}
unsafe impl Send for NativeObject {}
unsafe impl Sync for NativeObject {}

#[pymethods]
impl NativeObject {
    fn __repr__(&self, py: Python<'_>) -> String {
        let _ = py;
        with_ip_of(|ip| Ok(ip.format(&self.v))).unwrap_or_else(|_| "<native object>".to_string())
    }
}

/// The Python callables that went into the engine as function values, by the address of the native function's closure: handed back
/// as the same object (`to_py`). An entry lives as long as the closure (`Reg` removes it).
static CALLABLES: Mutex<Option<HashMap<usize, Py<PyAny>>>> = Mutex::new(None);

struct Reg(Cell<usize>);
impl Drop for Reg {
    fn drop(&mut self) {
        let k = self.0.get();
        if k != 0 {
            if let Ok(mut g) = CALLABLES.lock() {
                if let Some(m) = g.as_mut() {
                    m.remove(&k);
                }
            }
        }
    }
}

fn closure_key(f: &klong_core::value::NativeFn) -> usize {
    Rc::as_ptr(f) as *const () as usize
}

fn fn_key(v: &Value) -> usize {
    match v {
        Value::Fn(f) => Rc::as_ptr(f) as *const () as usize,
        Value::Prim(s) => 1 + s.0 as usize,
        _ => 0,
    }
}

/// A native dictionary, as the handle of its Python view (`KDict` in HOST_PY).
#[pyclass(name = "NativeDict")]
struct NativeDict {
    d: Rc<RefCell<Dict>>,
    owner: Py<Klong>,
}
unsafe impl Send for NativeDict {}
unsafe impl Sync for NativeDict {}

impl NativeDict {
    fn key(&self, py: Python<'_>, k: &Bound<'_, PyAny>) -> PyResult<Value> {
        let o = self.owner.borrow(py);
        o.with_ip(py, |ip| from_py(py, ip, k))
    }
    fn out<'py>(&self, py: Python<'py>, v: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let o = self.owner.borrow(py);
        let h = o.hook_out.borrow().as_ref().map(|h| h.clone_ref(py));
        match h {
            Some(h) => h.bind(py).call1((v,)),
            None => Ok(v),
        }
    }
}

#[pymethods]
impl NativeDict {
    fn get<'py>(&self, py: Python<'py>, key: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let k = self.key(py, key)?;
        let v = self.d.borrow().get(&k).cloned();
        match v {
            Some(v) => {
                let o = self.owner.borrow(py);
                let r = o.with_ip(py, |ip| to_py(py, ip, &v, Some(&self.owner)))?;
                self.out(py, r)
            }
            None => Err(pyo3::exceptions::PyKeyError::new_err(key.clone().unbind())),
        }
    }
    fn set(&self, py: Python<'_>, key: &Bound<'_, PyAny>, value: Bound<'_, PyAny>) -> PyResult<()> {
        let o = self.owner.borrow(py);
        let hook = o.hook_in.borrow().as_ref().map(|h| h.clone_ref(py));
        let value = match hook {
            Some(h) => h.bind(py).call1((value,))?,
            None => value,
        };
        let (k, v) = o.with_ip(py, |ip| -> PyResult<(Value, Value)> { Ok((from_py(py, ip, key)?, from_py_host(py, ip, &value)?)) })?;
        self.d.borrow_mut().set(k, v).map_err(k2py)
    }
    fn delete(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<()> {
        let k = self.key(py, key)?;
        if self.d.borrow_mut().remove(&k).map_err(k2py)? {
            Ok(())
        } else {
            Err(pyo3::exceptions::PyKeyError::new_err(key.clone().unbind()))
        }
    }
    fn contains(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<bool> {
        let k = match self.key(py, key) {
            Ok(k) => k,
            Err(_) => return Ok(false),
        };
        Ok(self.d.borrow().find(&k).is_some())
    }
    fn len(&self) -> usize {
        self.d.borrow().len()
    }
    fn clear(&self) {
        let keys: Vec<Value> = self.d.borrow().keys().to_vec();
        for k in keys {
            let _ = self.d.borrow_mut().remove(&k);
        }
    }
    fn keys<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyAny>>> {
        let keys: Vec<Value> = self.d.borrow().keys().to_vec();
        let o = self.owner.borrow(py);
        o.with_ip(py, |ip| keys.iter().map(|k| key_to_py(py, ip, k, Some(&self.owner))).collect())
    }
}

/// A dictionary key as Python has it: a list as a tuple.
fn key_to_py<'py>(py: Python<'py>, ip: &Interp, k: &Value, owner: Option<&Py<Klong>>) -> PyResult<Bound<'py, PyAny>> {
    match k {
        Value::List(_) | Value::Ints(_) | Value::Flts(_) => {
            let l = to_py(py, ip, k, owner)?;
            py.import("builtins")?.getattr("tuple")?.call1((l.call_method0("tolist")?,))
        }
        _ => to_py(py, ip, k, owner),
    }
}

/// A Klong function value (or operator) as a Python callable.
#[pyclass(name = "KFn")]
struct KFn {
    f: Value,
    owner: Py<Klong>,
}

#[pymethods]
impl KFn {
    #[pyo3(signature = (*args))]
    fn __call__(&self, py: Python<'_>, args: &Bound<'_, PyTuple>) -> PyResult<PyObject> {
        let k = self.owner.borrow(py);
        k.apply(py, &self.f, args)
    }
    fn __str__(&self, py: Python<'_>) -> String {
        self.owner.borrow(py).with_ip(py, |ip| ip.format(&self.f))
    }
    fn __repr__(&self, py: Python<'_>) -> String {
        self.__str__(py)
    }
    /// Two `KFn` are equal when they are the same function value (the same native function, the same operator).
    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        match other.downcast::<KFn>() {
            Ok(o) => fn_key(&o.borrow().f) == fn_key(&self.f),
            Err(_) => false,
        }
    }
    fn __hash__(&self) -> isize {
        fn_key(&self.f) as isize
    }

    /// `function`, `projection`, `native` (a system function) or `operator`.
    #[getter]
    fn kind(&self, py: Python<'_>) -> String {
        self.info(py).map(|i| i.kind.to_string()).unwrap_or_default()
    }
    /// The number of arguments it takes; for a projection, the holes it still has.
    #[getter]
    fn arity(&self, py: Python<'_>) -> u8 {
        self.info(py).map(|i| i.arity).unwrap_or(0)
    }
    /// The named parameters of `f(a;b)::{...}` (a list of names), or None.
    #[getter]
    fn params(&self, py: Python<'_>) -> Option<Vec<String>> {
        self.info(py).and_then(|i| i.params)
    }
    /// The locals declared by a leading `[a b]` in the body (a list of names).
    #[getter]
    fn locals(&self, py: Python<'_>) -> Vec<String> {
        self.info(py).map(|i| i.locals).unwrap_or_default()
    }
    /// The text of the function literal as it was written, or None.
    #[getter]
    fn source(&self, py: Python<'_>) -> Option<String> {
        self.info(py).and_then(|i| i.source)
    }
    /// (start, end) offsets in characters of the literal in its source (as every other position), or None.
    #[getter]
    fn span(&self, py: Python<'_>) -> Option<(i64, i64)> {
        self.info(py).and_then(|i| i.span_chars).map(|(a, b)| (a as i64, b as i64))
    }
    #[getter]
    fn is_projection(&self, py: Python<'_>) -> bool {
        self.info(py).map(|i| i.projection).unwrap_or(false)
    }
    /// A projection's arguments in order: the value, or None for a hole (an argument written as an expression is given as its source text).
    fn bound_args(&self, py: Python<'_>) -> PyResult<Vec<Option<PyObject>>> {
        let owner = self.owner.clone_ref(py);
        let bound = self.info(py).map(|i| i.bound).unwrap_or_default();
        let k = self.owner.borrow(py);
        k.with_ip(py, |ip| -> PyResult<Vec<Option<PyObject>>> {
            bound.iter().map(|b| b.as_ref().map(|v| Ok(to_py(py, ip, v, Some(&owner))?.unbind())).transpose()).collect()
        })
    }
    /// What a projection calls (a function value), or None.
    fn callee(&self, py: Python<'_>) -> PyResult<Option<PyObject>> {
        let owner = self.owner.clone_ref(py);
        let c = self.info(py).and_then(|i| i.callee);
        let k = self.owner.borrow(py);
        k.with_ip(py, |ip| -> PyResult<Option<PyObject>> { c.as_ref().map(|v| Ok(to_py(py, ip, v, Some(&owner))?.unbind())).transpose() })
    }
}

impl KFn {
    fn info(&self, py: Python<'_>) -> Option<klong_core::interp::FnInfo> {
        self.owner.borrow(py).with_ip(py, |ip| ip.fn_info(&self.f))
    }
}

/// An opaque host value (a NumPy array, a tensor: any Python object the interpreter's table says is one): the primitives and adverbs
/// that have it as an operand are forwarded to the table. `table` is the table of the interpreter that made it.
struct HostArr {
    obj: Py<PyAny>,
    table: Py<PyAny>,
    /// Not an array: any other Python object (a table, a store, a FixedArray's owner...). The primitives are still offered to the
    /// table (KlongPy's implementations dispatch on the object's type); what it does not take is an error, not a copy.
    opaque: bool,
}

/// The table of the interpreter that is running.
fn current_table(py: Python<'_>) -> PyResult<Py<PyAny>> {
    let t = OWNER.with(|o| o.borrow().as_ref().and_then(|k| k.borrow(py).table.borrow().as_ref().map(|t| t.clone_ref(py))));
    match t {
        Some(t) => Ok(t),
        None => Ok(host(py)?.default_table.clone_ref(py)),
    }
}

fn unhost(v: &Value, what: &str) -> klong_core::error::Result<Value> {
    if let Value::Ext(e) = v {
        if let Some(r) = e.0.host_native(what) {
            return r;
        }
    }
    Ok(v.clone())
}

impl HostArr {
    /// Ask the table: `Ok(None)` when it answers NotImplemented.
    fn forward(&self, py: Python<'_>, ip: &mut Interp, method: &str, args: Vec<Bound<'_, PyAny>>) -> PyResult<Option<Value>> {
        let r = self.table.bind(py).call_method1(method, PyTuple::new(py, args)?)?;
        if r.is(&py.NotImplemented()) {
            return Ok(None);
        }
        from_py_host(py, ip, &r).map(Some)
    }
}

impl Ext for HostArr {
    fn type_name(&self) -> &str {
        "host array"
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn host_shape(&self) -> Option<Vec<u32>> {
        if self.opaque {
            return None;
        }
        Python::with_gil(|py| self.obj.bind(py).getattr("shape").ok()?.extract::<Vec<u32>>().ok())
    }
    fn key_identity(&self) -> Option<usize> {
        // a Python object that is not an array is its own key, by identity (as Python hashes a function or a class)
        if self.opaque { Some(self.obj.as_ptr() as usize) } else { None }
    }
    fn host_native(&self, what: &str) -> Option<klong_core::error::Result<Value>> {
        if self.opaque {
            return Some(Err(Error::typ(klong_core::msg!("{}: a python object is not a Klong value this can use", what))));
        }
        Some(
            Python::with_gil(|py| -> PyResult<Value> {
                let r = self.table.bind(py).call_method1("to_native", (self.obj.bind(py), what))?;
                let prev = NATIVE_ONLY.with(|n| n.replace(true));
                let v = with_ip_of(|ip| from_py(py, ip, &r));
                NATIVE_ONLY.with(|n| n.set(prev));
                v
            })
            .map_err(py2k),
        )
    }
    fn host_monad(&self, ip: &mut Interp, glyph: &str, x: &Value) -> Option<klong_core::error::Result<Value>> {
        let forwarded = Python::with_gil(|py| -> PyResult<Option<Value>> {
            let a = operand(py, ip, x)?;
            self.forward(py, ip, "monad", vec![PyString::new(py, glyph).into_any(), a])
        });
        match forwarded {
            Ok(Some(v)) => Some(Ok(v)),
            Ok(None) if self.opaque => None,
            Ok(None) => {
                let nx = match unhost(x, &format!("the monad {glyph}")) {
                    Ok(v) => v,
                    Err(e) => return Some(Err(e)),
                };
                let op = ip.syms.lookup(glyph)?;
                Some(ip.monad(op, nx))
            }
            Err(e) => Some(Err(py2k(e))),
        }
    }
    fn host_dyad(&self, ip: &mut Interp, glyph: &str, x: &Value, y: &Value) -> Option<klong_core::error::Result<Value>> {
        let forwarded = Python::with_gil(|py| -> PyResult<Option<Value>> {
            let a = operand(py, ip, x)?;
            let b = operand(py, ip, y)?;
            self.forward(py, ip, "dyad", vec![PyString::new(py, glyph).into_any(), a, b])
        });
        match forwarded {
            Ok(Some(v)) => return Some(Ok(v)),
            Ok(None) if self.opaque => return None,
            Ok(None) => {}
            Err(e) => return Some(Err(py2k(e))),
        }
        let what = format!("the dyad {glyph}");
        let nx = match unhost(x, &what) {
            Ok(v) => v,
            Err(e) => return Some(Err(e)),
        };
        let ny = match unhost(y, &what) {
            Ok(v) => v,
            Err(e) => return Some(Err(e)),
        };
        let op = ip.syms.lookup(glyph)?;
        Some(ip.dyad(op, nx, ny))
    }
    fn host_adverb(&self, ip: &mut Interp, glyph: &str, verb: &Value, op: Option<&str>, x: &Value, y: Option<&Value>) -> Option<klong_core::error::Result<Value>> {
        let forwarded = Python::with_gil(|py| -> PyResult<Option<Value>> {
            let f = to_py_owner(py, ip, verb)?;
            let a = operand(py, ip, x)?;
            let b = match y {
                Some(y) => operand(py, ip, y)?,
                None => py.None().into_bound(py),
            };
            let op = match op {
                Some(o) => PyString::new(py, o).into_any(),
                None => py.None().into_bound(py),
            };
            self.forward(py, ip, "adverb", vec![PyString::new(py, glyph).into_any(), f, op, a, b])
        });
        match forwarded {
            Ok(v) => v.map(Ok),
            Err(e) => Some(Err(py2k(e))),
        }
    }
    fn host_converged(&self, p: &Value, q: &Value) -> Option<bool> {
        Python::with_gil(|py| -> PyResult<Option<bool>> {
            with_ip_of(|ip| {
                let (a, b) = (to_py_owner(py, ip, p)?, to_py_owner(py, ip, q)?);
                let r = self.table.bind(py).call_method1("converged", (a, b))?;
                if r.is_none() {
                    Ok(None)
                } else {
                    r.extract::<bool>().map(Some)
                }
            })
        })
        .unwrap_or(None)
    }
    fn host_items(&self) -> Option<klong_core::error::Result<Value>> {
        if self.opaque {
            return None;
        }
        let r = Python::with_gil(|py| -> PyResult<Option<Value>> {
            with_ip_of(|ip| {
                let r = self.table.bind(py).call_method1("items", (self.obj.bind(py),))?;
                if r.is_none() {
                    return Ok(None);
                }
                let mut items = Vec::new();
                for q in r.try_iter()? {
                    items.push(from_py_host(py, ip, &q?)?);
                }
                Value::raw_list(items).map(Some).map_err(k2py)
            })
        });
        match r {
            Ok(v) => v.map(Ok),
            Err(e) => Some(Err(py2k(e))),
        }
    }
    fn host_truthy(&self) -> Option<bool> {
        if self.opaque {
            return None;
        }
        Python::with_gil(|py| -> PyResult<Option<bool>> {
            let r = self.table.bind(py).call_method1("truthy", (self.obj.bind(py),))?;
            if r.is_none() {
                Ok(None)
            } else {
                r.extract::<bool>().map(Some)
            }
        })
        .unwrap_or(None)
    }
    fn host_format(&self, display: bool) -> Option<String> {
        Python::with_gil(|py| -> PyResult<Option<String>> {
            let r = self.table.bind(py).call_method1("format", (self.obj.bind(py), display))?;
            if r.is_none() {
                Ok(None)
            } else {
                r.extract::<String>().map(Some)
            }
        })
        .unwrap_or(None)
    }
    fn host_collect(&self, items: &[Value]) -> Option<klong_core::error::Result<Value>> {
        let r = Python::with_gil(|py| -> PyResult<Option<Value>> {
            with_ip_of(|ip| {
                let l = PyList::empty(py);
                for q in items {
                    l.append(to_py_owner(py, ip, q)?)?;
                }
                let r = self.table.bind(py).call_method1("collect", (l,))?;
                if r.is_none() {
                    return Ok(None);
                }
                from_py_host(py, ip, &r).map(Some)
            })
        });
        match r {
            Ok(v) => v.map(Ok),
            Err(e) => Some(Err(py2k(e))),
        }
    }
}

fn host_value(py: Python<'_>, obj: Py<PyAny>) -> PyResult<Value> {
    let table = current_table(py)?;
    Ok(Value::Ext(Rc::new(klong_core::value::ExtBox(Box::new(HostArr { obj, table, opaque: false })))))
}

/// An operand of a forwarded primitive as a Python object: the host value itself, or a native value converted.
fn operand<'py>(py: Python<'py>, ip: &Interp, v: &Value) -> PyResult<Bound<'py, PyAny>> {
    if let Value::Ext(e) = v {
        if let Some(h) = e.0.as_any().downcast_ref::<HostArr>() {
            return Ok(h.obj.bind(py).clone());
        }
    }
    to_py_with(py, ip, v, None, true)
}

/// A Python value as a Klong value, where a NumPy array of at least `host_threshold` elements stays a host value (the result of a
/// forwarded primitive, a value assigned from Python). Anything else is as in `from_py`.
fn from_py_host(py: Python<'_>, ip: &mut Interp, o: &Bound<'_, PyAny>) -> PyResult<Value> {
    let h = host(py)?;
    if !NATIVE_ONLY.with(|n| n.get()) && o.is_instance(h.ndarray.bind(py))? && o.is_exact_instance(h.ndarray.bind(py)) {
        let ndim: usize = o.getattr("ndim")?.extract()?;
        let kind: String = o.getattr("dtype")?.getattr("kind")?.extract()?;
        if ndim >= 1 && matches!(kind.as_str(), "i" | "u" | "f" | "b") {
            // large arrays stay as they are; so does an array of a dtype the engine has no kind for (float32, int32, ...) of any size:
            // the dtype is the program's (KlongPy keeps it through arithmetic)
            let native_dtype = {
                let name: String = o.getattr("dtype")?.getattr("name")?.extract()?;
                name == "int64" || name == "float64"
            };
            let adopting = OWNER.with(|w| w.borrow().as_ref().map(|k| k.borrow(py).adopting.get()).unwrap_or(false));
            if adopting || !native_dtype || o.getattr("size")?.extract::<usize>()? >= h.threshold.load(std::sync::atomic::Ordering::Relaxed) {
                return host_value(py, o.clone().unbind());
            }
        }
    }
    from_py(py, ip, o)
}

// ------------------------------------------------------------------------------------------------
// conversion
// ------------------------------------------------------------------------------------------------

fn to_py<'py>(py: Python<'py>, ip: &Interp, v: &Value, owner: Option<&Py<Klong>>) -> PyResult<Bound<'py, PyAny>> {
    to_py_with(py, ip, v, owner, false)
}

fn to_py_with<'py>(py: Python<'py>, ip: &Interp, v: &Value, owner: Option<&Py<Klong>>, view: bool) -> PyResult<Bound<'py, PyAny>> {
    Ok(match v {
        Value::Int(i) => i.into_pyobject(py)?.into_any(),
        Value::Big(_) => {
            // through hexadecimal text: Python limits the digits of a decimal one
            let hex = klong_core::big::to_hex(v).unwrap_or_default();
            py.get_type::<PyInt>().call1((hex, 16))?
        }
        Value::Flt(f) => PyFloat::new(py, *f).into_any(),
        Value::Char(c) => host(py)?.kgchar.bind(py).call1((c.to_string(),))?,
        Value::Str(s) => PyString::new(py, s.as_str()).into_any(),
        Value::Sym(s) => host(py)?.kgsym.bind(py).call1((ip.syms.name(*s),))?,
        Value::Undefined => host(py)?.undefined.bind(py).clone(),
        Value::Ints(_) | Value::Flts(_) => {
            let h = host(py)?;
            let nv = Bound::new(py, NativeVec { v: v.clone() })?;
            if view || ZERO_COPY.load(std::sync::atomic::Ordering::Relaxed) {
                h.asarray.bind(py).call1((nv,))?
            } else {
                h.array.bind(py).call1((nv,))?
            }
        }
        Value::List(l) => {
            let out = PyList::empty(py);
            for q in l.iter() {
                out.append(to_py_with(py, ip, q, owner, view)?)?;
            }
            host(py)?.mk_obj.bind(py).call1((out,))?
        }
        Value::Dict(d) => {
            let owner = owner.map(|o| o.clone_ref(py)).or_else(|| OWNER.with(|o| o.borrow().as_ref().map(|k| k.clone_ref(py))));
            let h = host(py)?;
            let id = Rc::as_ptr(d) as usize;
            let got = h.views.bind(py).call_method1("get", (id,))?;
            if !got.is_none() {
                return Ok(got);
            }
            match owner {
                Some(owner) => {
                    // one live view per native dictionary
                    let handle = Bound::new(py, NativeDict { d: d.clone(), owner })?;
                    let view = h.kdict.bind(py).call1((handle,))?;
                    h.views.bind(py).set_item(id, &view)?;
                    view
                }
                None => {
                    let out = PyDict::new(py);
                    let d = d.borrow();
                    for (k, val) in d.keys().iter().zip(d.vals().iter()) {
                        out.set_item(key_to_py(py, ip, k, None)?, to_py_with(py, ip, val, None, view)?)?;
                    }
                    out.into_any()
                }
            }
        }
        Value::Ext(e) => match e.0.as_any().downcast_ref::<HostArr>() {
            Some(h) => h.obj.bind(py).clone(),
            None => Bound::new(py, NativeObject { v: v.clone() })?.into_any(),
        },
        Value::Fn(fv) if matches!(&**fv, Func::Native { .. }) && native_callable(py, fv).is_some() => native_callable(py, fv).unwrap().into_bound(py),
        Value::Fn(_) | Value::Prim(_) => {
            let o = owner.map(|o| o.clone_ref(py)).or_else(|| OWNER.with(|w| w.borrow().as_ref().map(|k| k.clone_ref(py))));
            match o {
                Some(o) => Bound::new(py, KFn { f: v.clone(), owner: o })?.into_any(),
                None => return Err(PyTypeError::new_err("a Klong function needs its interpreter to be called")),
            }
        }
    })
}

/// The Python callable a native function was made from, if it was.
fn native_callable(py: Python<'_>, f: &Rc<Func>) -> Option<Py<PyAny>> {
    if let Func::Native { f, .. } = &**f {
        let g = CALLABLES.lock().ok()?;
        return g.as_ref()?.get(&closure_key(f)).map(|o| o.clone_ref(py));
    }
    None
}

/// A Python value as a Klong value. NumPy arrays are COPIED into native vectors (dtype int64/float64; others converted).
fn from_py(py: Python<'_>, ip: &mut Interp, o: &Bound<'_, PyAny>) -> PyResult<Value> {
    if o.is_none() {
        // None is a Python object like any other here, and is given back as None (`:undefined` is its own value)
        return Ok(Value::Ext(Rc::new(klong_core::value::ExtBox(Box::new(HostArr { obj: o.clone().unbind(), table: current_table(py)?, opaque: true })))));
    }
    if o.is_exact_instance_of::<PyString>() {
        return Value::text(o.downcast::<PyString>()?.to_str()?).map_err(k2py);
    }
    if let Ok(b) = o.downcast::<PyBool>() {
        return Ok(Value::Int(b.is_true() as Int));
    }
    if o.is_instance_of::<PyInt>() {
        if let Ok(i) = o.extract::<Int>() {
            return Ok(Value::Int(i));
        }
        let hex: String = o.call_method1("__format__", ("x",))?.extract()?;
        return klong_core::big::from_hex(&hex).ok_or_else(|| PyOverflowError::new_err("integer does not convert"));
    }
    if o.is_instance_of::<PyFloat>() {
        return Ok(Value::Flt(o.extract::<f64>()?));
    }
    let h = host(py)?;
    if o.is(h.undefined.bind(py)) {
        return Ok(Value::Undefined);
    }
    if o.is_instance_of::<PyString>() {
        let s: String = o.extract()?;
        if o.is_instance(h.kgchar.bind(py))? {
            if let Some(c) = s.chars().next() {
                return Ok(Value::Char(c));
            }
        }
        if o.is_instance(h.kgsym.bind(py))? {
            let id = ip.syms.intern(&s);
            return Ok(Value::Sym(id));
        }
        return Value::text(&s).map_err(k2py);
    }
    if let Ok(nv) = o.downcast::<NativeVec>() {
        return Ok(nv.borrow().v.clone());
    }
    if o.is_instance(h.kdict.bind(py))? {
        // the view of a native dictionary: that dictionary
        if let Ok(handle) = o.getattr("_h")?.downcast::<NativeDict>() {
            return Ok(Value::Dict(handle.borrow().d.clone()));
        }
    }
    if let Ok(kf) = o.downcast::<KFn>() {
        return Ok(kf.borrow().f.clone());
    }
    if let Ok(no) = o.downcast::<NativeObject>() {
        return Ok(no.borrow().v.clone());
    }
    if o.is_instance(h.ndarray.bind(py))? {
        if !o.is_exact_instance(h.ndarray.bind(py)) {
            // a subclass of ndarray (KlongPy's FixedArray, a masked array) has meaning of its own: kept as the object it is
            return Ok(Value::Ext(Rc::new(klong_core::value::ExtBox(Box::new(HostArr { obj: o.clone().unbind(), table: current_table(py)?, opaque: true })))));
        }
        return ndarray_to_native(py, &h, o);
    }
    if let Ok(l) = o.downcast::<PyList>() {
        let items = l.iter().map(|q| from_py(py, ip, &q)).collect::<PyResult<Vec<_>>>()?;
        return Value::from_items(items).map_err(k2py);
    }
    if let Ok(t) = o.downcast::<PyTuple>() {
        let items = t.iter().map(|q| from_py(py, ip, &q)).collect::<PyResult<Vec<_>>>()?;
        return Value::from_items(items).map_err(k2py);
    }
    if o.is_instance_of::<PyDict>() && !o.is_exact_instance_of::<PyDict>() {
        // a subclass of dict (an OrderedDict, a defaultdict, a table) has meaning of its own: kept as the object it is, not copied
        return Ok(Value::Ext(Rc::new(klong_core::value::ExtBox(Box::new(HostArr { obj: o.clone().unbind(), table: current_table(py)?, opaque: true })))));
    }
    if let Ok(d) = o.downcast::<PyDict>() {
        let mut out = Dict::default();
        for (k, v) in d.iter() {
            out.set(from_py(py, ip, &k)?, from_py(py, ip, &v)?).map_err(k2py)?;
        }
        return Ok(Value::dict(out));
    }
    if !NATIVE_ONLY.with(|n| n.get()) {
        let t = current_table(py)?;
        if t.bind(py).call_method1("is_host", (o,))?.is_truthy()? {
            return Ok(Value::Ext(Rc::new(klong_core::value::ExtBox(Box::new(HostArr { obj: o.clone().unbind(), table: t, opaque: false })))));
        }
    }
    if o.hasattr("dtype")? && o.hasattr("item")? {
        // a NumPy scalar
        let kind: String = o.getattr("dtype")?.getattr("kind")?.extract()?;
        if matches!(kind.as_str(), "M" | "m") {
            // a date or a time span has no native kind: it stays the object it is
            return Ok(Value::Ext(Rc::new(klong_core::value::ExtBox(Box::new(HostArr { obj: o.clone().unbind(), table: current_table(py)?, opaque: true })))));
        }
        if kind == "f" && o.getattr("dtype")?.getattr("itemsize")?.extract::<usize>()? < 8 {
            // a float32 is its shortest text (68.76, not 68.76000213623047)
            if let Ok(f) = o.str()?.to_str()?.parse::<f64>() {
                return Ok(Value::Flt(f));
            }
        }
        return from_py(py, ip, &o.call_method0("item")?);
    }
    if o.hasattr("_provide_klong")? && o.hasattr("fn")? {
        // a KlongPy KGLambda: its Python function, with the arity it declares
        let wild: bool = o.getattr("_wildcard")?.extract()?;
        let n: usize = o.getattr("args")?.len().unwrap_or(0);
        return Ok(callable_value_of(ip, "python", o.getattr("fn")?.unbind(), if wild { 255 } else { n as u8 }, o.clone().unbind()));
    }
    if o.is_callable() {
        return Ok(callable_value(ip, "python", o.clone().unbind(), 255));
    }
    Ok(Value::Ext(Rc::new(klong_core::value::ExtBox(Box::new(HostArr { obj: o.clone().unbind(), table: current_table(py)?, opaque: true })))))
}

fn with_ip_of<R>(f: impl FnOnce(&mut Interp) -> PyResult<R>) -> PyResult<R> {
    let p = CURRENT.with(|c| c.get());
    if p.is_null() {
        return Err(PyTypeError::new_err("no interpreter is running"));
    }
    // SAFETY: set by `Klong::with_ip` for the duration of the outermost call; the outer `&mut Interp` is lent to this call chain.
    f(unsafe { &mut *p })
}

/// An ndarray copied into a native vector: the one copy of the NumPy-to-native direction.
fn ndarray_to_native(py: Python<'_>, h: &Host, o: &Bound<'_, PyAny>) -> PyResult<Value> {
    let kind: String = o.getattr("dtype")?.getattr("kind")?.extract()?;
    let shape: Vec<usize> = o.getattr("shape")?.extract()?;
    if shape.is_empty() {
        return with_ip_of(|ip| from_py(py, ip, &o.call_method0("item")?));
    }
    if shape.len() > 4 {
        return Err(PyTypeError::new_err("native arrays have at most 4 dimensions"));
    }
    let (dt, floats) = match kind.as_str() {
        "i" | "u" | "b" => (h.int64.bind(py), false),
        "f" => (h.float64.bind(py), true),
        _ => {
            let l = o.call_method0("tolist")?;
            return with_ip_of(|ip| from_py(py, ip, &l));
        }
    };
    let narrow = floats && o.getattr("dtype")?.getattr("itemsize")?.extract::<usize>()? < 8;
    let a = h.ascontig.bind(py).call1((o, dt))?;
    let ai = a.getattr("__array_interface__")?;
    let ptr: usize = ai.get_item("data")?.get_item(0)?.extract()?;
    let n: usize = shape.iter().product();
    let dims: Vec<u32> = shape.iter().map(|&d| d as u32).collect();
    // SAFETY: `a` is a live contiguous array of n elements of the dtype asked for; it is read before `a` is dropped.
    unsafe {
        if floats {
            let s = std::slice::from_raw_parts(ptr as *const f64, n);
            // a float32 member is its shortest text (68.76, not 68.76000213623047)
            let v = if narrow { s.iter().map(|&x| format!("{}", x as f32).parse::<f64>().unwrap_or(x)).collect() } else { s.to_vec() };
            Ok(Value::Flts(Rc::new(Arr::shaped(v, &dims))))
        } else {
            let s = std::slice::from_raw_parts(ptr as *const i64, n);
            Ok(Value::Ints(Rc::new(Arr::shaped(s.to_vec(), &dims))))
        }
    }
}

/// A Python callable as a Klong system function.
fn callable_value(ip: &mut Interp, name: &str, f: Py<PyAny>, arity: u8) -> Value {
    let orig = Python::with_gil(|py| f.clone_ref(py));
    callable_value_of(ip, name, f, arity, orig)
}

/// `orig` is what `to_py` of the function gives back (the callable that was handed in).
fn callable_value_of(ip: &mut Interp, name: &str, f: Py<PyAny>, arity: u8, orig: Py<PyAny>) -> Value {
    // the same Python callable is the same Klong function while one is alive (a dictionary key is compared by identity)
    let pid = orig.as_ptr() as usize;
    let known = PYFNS.with(|m| m.borrow().get(&pid).and_then(|w| w.upgrade()));
    if let Some(k) = known {
        if let Func::Native { arity: a, .. } = &*k {
            if *a == arity {
                return Value::Fn(k);
            }
        }
    }
    let sym = ip.syms.intern(name);
    let reg = Rc::new(Reg(Cell::new(0)));
    let guard = reg.clone();
    let nf: klong_core::value::NativeFn = Rc::new(move |ip: &mut Interp, args: &[Value]| -> klong_core::error::Result<Value> {
        let _keep = &guard;
        Python::with_gil(|py| -> PyResult<Value> {
            let mut pa = Vec::with_capacity(args.len());
            for a in args {
                pa.push(to_py_owner(py, ip, a)?);
            }
            let r = f.bind(py).call1(PyTuple::new(py, pa)?)?;
            from_py_host(py, ip, &r)
        })
        .map_err(py2k)
    });
    let key = closure_key(&nf);
    if let Ok(mut g) = CALLABLES.lock() {
        g.get_or_insert_with(HashMap::new).insert(key, orig);
        reg.0.set(key);
    }
    let fv = Rc::new(Func::Native { name: sym, arity, f: nf });
    PYFNS.with(|m| {
        let mut m = m.borrow_mut();
        if m.len() > 64 && m.len().is_power_of_two() {
            m.retain(|_, w| w.strong_count() > 0);
        }
        m.insert(pid, Rc::downgrade(&fv));
    });
    Value::Fn(fv)
}

thread_local! {
    /// Python callables already made into Klong functions, by the object's address (weak: the function owns the object).
    static PYFNS: RefCell<HashMap<usize, std::rc::Weak<Func>>> = RefCell::new(HashMap::new());
}

/// Convert for a callback: functions become `KFn` of the interpreter that is running.
fn to_py_owner<'py>(py: Python<'py>, ip: &Interp, v: &Value) -> PyResult<Bound<'py, PyAny>> {
    let o = OWNER.with(|o| o.borrow().as_ref().map(|k| k.clone_ref(py)));
    to_py(py, ip, v, o.as_ref())
}

// ------------------------------------------------------------------------------------------------
// the interpreter
// ------------------------------------------------------------------------------------------------

#[pyclass(name = "Klong")]
struct Klong {
    rt: UnsafeCell<Runtime<HostPlatform>>,
    /// Set while the interpreter is running: a callback that re-enters (`Klong -> Python -> Klong`) uses it instead of borrowing again.
    cur: Cell<*mut Interp>,
    /// The thread that holds `cur` (its stack is what the evaluator measures); a re-entry from another thread rebases the stack.
    cur_thread: Cell<Option<std::thread::ThreadId>>,
    /// Floats print as float32 does (a torch backend): see `set_float32_display`.
    f32_display: Cell<bool>,
    /// The KLONGPATH `.l` was last given.
    load_path: RefCell<String>,
    me: RefCell<Option<Py<Klong>>>,
    /// The callable table of the backend (see HOST_PY); None: the default NumPy table.
    table: RefCell<Option<Py<PyAny>>>,
    /// The embedder's conversions applied to values that cross through a dictionary view: `out(v)` to what a read returns,
    /// `into(v)` to what a write stores (KlongPy's function wrappers).
    hook_out: RefCell<Option<Py<PyAny>>>,
    adopting: Cell<bool>,
    hook_in: RefCell<Option<Py<PyAny>>>,
    /// Input the live console's `read` returned and the channel has not taken yet (a chunk is at most 256 bytes; a reply may be longer).
    console_spare: Rc<RefCell<Vec<u8>>>,
}

unsafe impl Send for Klong {}
unsafe impl Sync for Klong {}

extern "C" {
    fn pthread_self() -> usize;
}
#[cfg(target_os = "macos")]
extern "C" {
    fn pthread_get_stackaddr_np(t: usize) -> usize;
    fn pthread_get_stacksize_np(t: usize) -> usize;
}
#[cfg(target_os = "linux")]
extern "C" {
    fn pthread_getattr_np(t: usize, attr: *mut u64) -> i32;
    fn pthread_attr_getstack(attr: *const u64, addr: *mut usize, size: *mut usize) -> i32;
    fn pthread_attr_destroy(attr: *mut u64) -> i32;
}

thread_local! {
    /// The lowest address of this thread's stack (0: unknown).
    static STACK_LOW: Cell<usize> = const { Cell::new(usize::MAX) };
}

fn stack_low() -> usize {
    let c = STACK_LOW.with(|c| c.get());
    if c != usize::MAX {
        return c;
    }
    let lo = unsafe { query_stack_low() };
    STACK_LOW.with(|c| c.set(lo));
    lo
}

#[cfg(target_os = "macos")]
unsafe fn query_stack_low() -> usize {
    let t = pthread_self();
    pthread_get_stackaddr_np(t).saturating_sub(pthread_get_stacksize_np(t))
}

#[cfg(target_os = "linux")]
unsafe fn query_stack_low() -> usize {
    let mut attr = [0u64; 16];
    if pthread_getattr_np(pthread_self(), attr.as_mut_ptr()) != 0 {
        return 0;
    }
    let (mut addr, mut size) = (0usize, 0usize);
    let r = pthread_attr_getstack(attr.as_ptr(), &mut addr, &mut size);
    pthread_attr_destroy(attr.as_mut_ptr());
    if r == 0 {
        addr
    } else {
        0
    }
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
unsafe fn query_stack_low() -> usize {
    0
}

/// The native stack the evaluator may use, from here: what is left of THIS thread's real stack (its own size: 8 MiB on the main thread,
/// what `threading.stack_size()` gave a Python thread), less a reserve for the frames of Python and libc that run below the evaluator.
fn stack_budget() -> usize {
    let marker = 0u8;
    let here = std::hint::black_box(&marker) as *const u8 as usize;
    let lo = stack_low();
    if lo == 0 || lo >= here {
        return 256 * 1024;
    }
    let avail = here - lo;
    avail - (128 * 1024).min(avail / 4)
}

type WarningTuple = (String, String, Option<String>, i64, i64, String, String);

fn warnings_of(ip: &mut Interp) -> Vec<WarningTuple> {
    ip.take_warnings()
        .into_iter()
        .map(|w| {
            let msg = match &w.note {
                Some(n) => format!("{}; {}", w.message, n),
                None => w.message.clone(),
            };
            let (name, text) = match &w.source {
                Some(s) => (s.name.clone(), s.text.to_string()),
                None => (String::new(), String::new()),
            };
            let (a, b) = char_span(&text, w.span.0 as usize, w.span.1 as usize);
            (w.code.to_string(), msg, w.hint.clone(), a, b, name, text)
        })
        .collect()
}

impl Klong {
    fn with_ip<R>(&self, py: Python<'_>, f: impl FnOnce(&mut Interp) -> R) -> R {
        let cur = self.cur.get();
        if !cur.is_null() {
            // re-entered from a callback: the outer `&mut Interp` is lent to the callback and not used until it returns
            let me = std::thread::current().id();
            if self.cur_thread.get() == Some(me) {
                return f(unsafe { &mut *cur });
            }
            // another thread (a callback's, an IPC loop's): its own stack, its own limit, measured from here
            let ip = unsafe { &mut *cur };
            let prev = ip.rebase_stack(Some(stack_budget()));
            let outer = self.cur_thread.replace(Some(me));
            let r = f(unsafe { &mut *cur });
            self.cur_thread.set(outer);
            unsafe { &mut *cur }.restore_stack(prev);
            return r;
        }
        let ip: *mut Interp = unsafe { &mut (*self.rt.get()).interp };
        unsafe { (*ip).cfg.stack_limit = Some(stack_budget()) };
        self.cur.set(ip);
        self.cur_thread.set(Some(std::thread::current().id()));
        self.sync_load_path();
        klong_core::fmt::FLOAT32_DISPLAY.store(self.f32_display.get(), std::sync::atomic::Ordering::Relaxed);
        let prev = CURRENT.with(|c| c.replace(ip));
        let prev_owner = OWNER.with(|o| o.replace(self.me.borrow().as_ref().map(|m| m.clone_ref(py))));
        let r = f(unsafe { &mut *ip });
        OWNER.with(|o| *o.borrow_mut() = prev_owner);
        CURRENT.with(|c| c.set(prev));
        self.cur.set(std::ptr::null_mut());
        self.drain(py);
        r
    }

    /// `.l` looks in the directories of KLONGPATH as it is when the program starts (the environment may have changed since the
    /// interpreter was made).
    fn sync_load_path(&self) {
        let path = std::env::var("KLONGPATH").ok().filter(|p| !p.is_empty()).unwrap_or_else(|| String::from(".:lib"));
        let mut last = self.load_path.borrow_mut();
        if *last != path {
            let rt = unsafe { &mut *self.rt.get() };
            rt.set_load_path(path.split(':').filter(|d| !d.is_empty()).map(String::from).collect());
            *last = path;
        }
    }

    /// What the program wrote with `.p` and friends goes to Python's `sys.stdout`.
    fn drain(&self, py: Python<'_>) {
        let rt = unsafe { &*self.rt.get() };
        let text = {
            let mut p = rt.shared.platform.borrow_mut();
            match &mut p.console {
                Console::Buffer(b) if !b.is_empty() => {
                    let s = String::from_utf8_lossy(b).into_owned();
                    b.clear();
                    Some(s)
                }
                _ => None,
            }
        };
        if let Some(s) = text {
            if let Ok(out) = py.import("sys").and_then(|m| m.getattr("stdout")) {
                let _ = out.call_method1("write", (s,));
            }
        }
    }

    /// Run `f` as the program `name`: while it runs, a relative `.l` that nothing else finds is looked for in the directory of that file.
    fn in_source<R>(&self, name: &str, f: impl FnOnce(&mut Interp) -> R, ip: &mut Interp) -> R {
        let dir = if name.starts_with('<') || !name.contains('/') { None } else { name.rfind('/').map(|i| if i == 0 { "/".to_string() } else { name[..i].to_string() }) };
        let rt = unsafe { &*self.rt.get() };
        if let Some(d) = &dir {
            rt.push_home(d);
        }
        let r = f(ip);
        if dir.is_some() {
            rt.pop_home();
        }
        r
    }

    fn owner(&self, py: Python<'_>) -> Py<Klong> {
        self.me.borrow().as_ref().map(|m| m.clone_ref(py)).expect("owner set in new()")
    }

    fn apply(&self, py: Python<'_>, f: &Value, args: &Bound<'_, PyTuple>) -> PyResult<PyObject> {
        let owner = self.owner(py);
        self.with_ip(py, |ip| -> PyResult<PyObject> {
            let vals = args.iter().map(|a| from_py_host(py, ip, &a)).collect::<PyResult<Vec<_>>>()?;
            let v = ip.apply(f, vals, None).map_err(k2py)?;
            Ok(to_py(py, ip, &v, Some(&owner))?.unbind())
        })
    }
}

#[pymethods]
impl Klong {
    /// `types`: {"KGSym": cls, "KGChar": cls, "undefined": the :undefined object}.
    #[new]
    #[pyo3(signature = (strict=0, types=None))]
    fn new(py: Python<'_>, strict: u8, types: Option<Bound<'_, PyDict>>) -> PyResult<Py<Klong>> {
        let types = match types {
            Some(t) => t,
            None => {
                let d = PyDict::new(py);
                d.set_item("KGSym", py.get_type::<PyString>())?;
                d.set_item("KGChar", py.get_type::<PyString>())?;
                d.set_item("undefined", py.None())?;
                d
            }
        };
        init_host(py, &types)?;
        // the Python main thread has the OS default stack (8 MiB on macOS and Linux); keep the evaluator inside half of it
        let cfg = Config { strict, stack_limit: Some(stack_budget()), max_depth: 1_000_000, ..Config::default() };
        let mut rt = Runtime::new(HostPlatform::new(Console::Buffer(Vec::new())), cfg, None);
        let path = std::env::var("KLONGPATH").ok().filter(|p| !p.is_empty()).unwrap_or_else(|| String::from(".:lib"));
        rt.set_load_path(path.split(':').filter(|d| !d.is_empty()).map(String::from).collect());
        let me = Py::new(py, Klong { rt: UnsafeCell::new(rt), cur: Cell::new(std::ptr::null_mut()), cur_thread: Cell::new(None), f32_display: Cell::new(false), load_path: RefCell::new(path), me: RefCell::new(None), table: RefCell::new(None), hook_out: RefCell::new(None), adopting: Cell::new(false), hook_in: RefCell::new(None), console_spare: Rc::new(RefCell::new(Vec::new())) })?;
        *me.borrow(py).me.borrow_mut() = Some(me.clone_ref(py));
        Ok(me)
    }

    /// Evaluate and convert the result.
    #[pyo3(signature = (src, name=None))]
    fn eval(&self, py: Python<'_>, src: &str, name: Option<&str>) -> PyResult<PyObject> {
        let owner = self.owner(py);
        let name = name.unwrap_or("<string>");
        self.with_ip(py, |ip| -> PyResult<PyObject> {
            match self.in_source(name, |ip| ip.run(src, name), ip).map_err(k2py)? {
                Some(v) => Ok(to_py(py, ip, &v, Some(&owner))?.unbind()),
                None => Ok(py.None()),
            }
        })
    }

    /// KlongPy's `exec`: the value of every statement, as a list.
    #[pyo3(signature = (src, name=None))]
    fn exec(&self, py: Python<'_>, src: &str, name: Option<&str>) -> PyResult<PyObject> {
        let owner = self.owner(py);
        let name = name.unwrap_or("<string>");
        self.with_ip(py, |ip| -> PyResult<PyObject> {
            ip.collect = Some(Vec::new());
            let r = self.in_source(name, |ip| ip.run(src, name), ip);
            let vals = ip.collect.take().unwrap_or_default();
            r.map_err(k2py)?;
            let out = PyList::empty(py);
            for v in &vals {
                out.append(to_py(py, ip, v, Some(&owner))?)?;
            }
            Ok(out.into_any().unbind())
        })
    }

    /// Evaluate and drop the result (the cost of the crossing without a conversion).
    #[pyo3(signature = (src, name=None))]
    fn run(&self, py: Python<'_>, src: &str, name: Option<&str>) -> PyResult<()> {
        let name = name.unwrap_or("<string>");
        self.with_ip(py, |ip| self.in_source(name, |ip| ip.run(src, name), ip).map(|_| ())).map_err(k2py)
    }

    /// The strict level of the program (a program's own `.strict(n)` is visible here).
    #[getter]
    fn strict(&self, py: Python<'_>) -> u8 {
        self.with_ip(py, |ip| ip.cfg.strict)
    }

    /// The names of every global that is defined: the program's variables and functions and the system functions.
    fn names(&self, py: Python<'_>) -> Vec<String> {
        self.with_ip(py, |ip| ip.global_names())
    }

    /// Conversions the embedder applies to values that cross through a dictionary view: `out(v)` on a read, `into(v)` on a write.
    #[pyo3(signature = (out=None, into=None))]
    fn set_converters(&self, out: Option<Py<PyAny>>, into: Option<Py<PyAny>>) {
        *self.hook_out.borrow_mut() = out;
        *self.hook_in.borrow_mut() = into;
    }

    /// Evaluate and return the native printed form of the result (`None` when there is no value).
    fn show(&self, py: Python<'_>, src: &str) -> PyResult<Option<String>> {
        self.with_ip(py, |ip| -> PyResult<Option<String>> {
            match ip.run(src, "<string>").map_err(k2py)? {
                Some(v) => Ok(Some(klong_core::fmt::format_value(&ip.syms, &v, false).map_err(k2py)?)),
                None => Ok(None),
            }
        })
    }

    fn __getitem__(&self, py: Python<'_>, name: &str) -> PyResult<PyObject> {
        let owner = self.owner(py);
        self.with_ip(py, |ip| -> PyResult<PyObject> {
            match ip.global(name) {
                Some(v) => Ok(to_py(py, ip, &v, Some(&owner))?.unbind()),
                None => Err(pyo3::exceptions::PyKeyError::new_err(name.to_string())),
            }
        })
    }

    fn __contains__(&self, py: Python<'_>, name: &str) -> bool {
        self.with_ip(py, |ip| ip.global(name).is_some())
    }

    fn __delitem__(&self, py: Python<'_>, name: &str) -> PyResult<()> {
        if self.with_ip(py, |ip| ip.undefine_global(name)) {
            Ok(())
        } else {
            Err(pyo3::exceptions::PyKeyError::new_err(name.to_string()))
        }
    }

    /// Set a global. Callables become system functions (arity from `__code__.co_argcount` if there is one); a numeric 1-d NumPy
    /// array of at least `host_threshold` elements stays a host array.
    fn __setitem__(&self, py: Python<'_>, name: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.with_ip(py, |ip| -> PyResult<()> {
            let h = host(py)?;
            let v = if value.is_instance(h.ndarray.bind(py))? {
                from_py_host(py, ip, value)?
            } else if value.is_callable() && !value.is_instance_of::<KFn>() {
                let ar = value.getattr("__code__").and_then(|c| c.getattr("co_argcount")).and_then(|n| n.extract::<u8>()).unwrap_or(255);
                callable_value(ip, name, value.clone().unbind(), ar)
            } else {
                from_py(py, ip, value)?
            };
            ip.define_global(name, v).map_err(k2py)
        })
    }

    /// Register a Python callable as a system function with an explicit arity (255: any number of arguments).
    /// `orig` (optional) is the object that stands for the function on the Python side: what Python gets back for it, and the identity
    /// it has as a dictionary key (when `f` is a wrapper of the function the program handed in).
    #[pyo3(signature = (name, f, arity, orig=None))]
    fn register(&self, py: Python<'_>, name: &str, f: Py<PyAny>, arity: u8, orig: Option<Py<PyAny>>) -> PyResult<()> {
        self.with_ip(py, |ip| {
            let v = match orig {
                Some(o) => callable_value_of(ip, name, f, arity, o),
                None => callable_value(ip, name, f, arity),
            };
            ip.define_global(name, v)
        })
        .map_err(k2py)
    }

    /// Floats print as float32 does: the shortest text that reads back as the same float32 (`0.1+0.2` shows `0.3`, `1%3` shows
    /// `0.33333334`). For the torch backend, whose floats are float32. Applies to the printing of this interpreter's programs.
    fn set_float32_display(&self, on: bool) {
        self.f32_display.set(on);
        klong_core::fmt::FLOAT32_DISPLAY.store(on, std::sync::atomic::Ordering::Relaxed);
    }

    /// Install the backend's callable table (see HOST_PY for its methods).
    fn set_host_table(&self, table: Py<PyAny>) {
        *self.table.borrow_mut() = Some(table);
    }

    /// Install the implementation of the gradient operators (`∇`, `∂`, `:>`): `f(glyph, *operands)`, the operands as Python values.
    /// For `:>` the third operand is the right operand's name when it was written as one, else None.
    fn set_host_ops(&self, py: Python<'_>, f: Py<PyAny>) {
        let owner = self.owner(py);
        self.with_ip(py, |ip| {
            ip.host_op = Some(Rc::new(move |ip: &mut Interp, glyph: &str, args: &[Value]| -> klong_core::error::Result<Value> {
                Python::with_gil(|py| -> PyResult<Value> {
                    let mut pa: Vec<Bound<'_, PyAny>> = vec![PyString::new(py, glyph).into_any()];
                    for a in args {
                        pa.push(to_py(py, ip, a, Some(&owner))?);
                    }
                    let r = f.bind(py).call1(PyTuple::new(py, pa)?)?;
                    from_py_host(py, ip, &r)
                })
                .map_err(py2k)
            }));
        });
    }

    /// With a backend whose arrays are not NumPy's (torch): every native numeric vector of at least `min_len` elements that evaluation
    /// produces (a literal, the result of a primitive) is handed to the table's `adopt(ndarray)` and kept as the host value it
    /// answers (a tensor on the device); a NumPy array handed in is kept as it is (host value, its dtype). `None` switches it off.
    #[pyo3(signature = (min_len=None))]
    fn set_adopt(&self, py: Python<'_>, min_len: Option<usize>) -> PyResult<()> {
        self.adopting.set(min_len.is_some());
        self.with_ip(py, |ip| {
            ip.adopt = match min_len {
                None => None,
                Some(min) => {
                    let cache: RefCell<HashMap<usize, (Value, Value)>> = RefCell::new(HashMap::new());
                    Some(Rc::new(move |ip: &mut Interp, v: &Value| -> Option<Value> {
                        let (n, key, shared) = match v {
                            Value::Ints(a) => (a.data.len(), Rc::as_ptr(a) as usize, Rc::strong_count(a) > 1),
                            Value::Flts(a) => (a.data.len(), Rc::as_ptr(a) as usize, Rc::strong_count(a) > 1),
                            _ => return None,
                        };
                        if n < min || n == 0 {
                            return None;
                        }
                        if shared {
                            // a literal of a tree or the value of a variable: converted once
                            if let Some((_, a)) = cache.borrow().get(&key) {
                                return Some(a.clone());
                            }
                        }
                        let r = Python::with_gil(|py| -> PyResult<Option<Value>> {
                            let arr = to_py_with(py, ip, v, None, false)?;
                            let t = current_table(py)?;
                            let r = t.bind(py).call_method1("adopt", (arr,))?;
                            if r.is_none() {
                                return Ok(None);
                            }
                            Ok(Some(Value::Ext(Rc::new(klong_core::value::ExtBox(Box::new(HostArr { obj: r.unbind(), table: t, opaque: false }))))))
                        });
                        match r {
                            Ok(Some(a)) => {
                                if shared {
                                    let mut c = cache.borrow_mut();
                                    if c.len() >= 1024 {
                                        c.clear();
                                    }
                                    c.insert(key, (v.clone(), a.clone()));
                                }
                                Some(a)
                            }
                            _ => None,
                        }
                    }) as Rc<dyn Fn(&mut Interp, &Value) -> Option<Value>>)
                }
            };
        });
        Ok(())
    }

    /// Always keep this array as a host array, whatever its size.
    fn set_host(&self, py: Python<'_>, name: &str, arr: Bound<'_, PyAny>) -> PyResult<()> {
        let v = host_value(py, arr.unbind())?;
        self.with_ip(py, |ip| ip.define_global(name, v)).map_err(k2py)
    }

    #[setter]
    fn set_host_threshold(&self, py: Python<'_>, n: usize) -> PyResult<()> {
        host(py)?.threshold.store(n, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    /// Call a global function by name.
    #[pyo3(signature = (name, *args))]
    fn call(&self, py: Python<'_>, name: &str, args: &Bound<'_, PyTuple>) -> PyResult<PyObject> {
        let f = self.with_ip(py, |ip| ip.global(name)).ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(name.to_string()))?;
        self.apply(py, &f, args)
    }

    /// The console channels (`.p .d .w` out, `.cerr`, and `.rl .r .mi` in) on Python file-like objects.
    ///
    /// `live=False` (the default of a new interpreter): what a program writes is collected and given to the current `sys.stdout` when the
    /// call that ran it returns; the console channel reads nothing.
    /// `live=True`: each write goes at once to `write(text)` (default: the `sys.stdout.write` of that moment, so a test that replaces
    /// `sys.stdout` is followed), console errors to `error(text)` (default `sys.stderr.write`), and the console channel reads from
    /// `read(n)`: the next input, at most `n` characters, a `str` or `bytes`, blocking until there is some, empty at the end of input
    /// (default: `sys.stdin.readline()` of that moment, ignoring `n`). Reading never happens in the background: only a program
    /// that reads the channel (`.rl`, `.r`, `.mi`) calls it.
    #[pyo3(signature = (live, write=None, error=None, read=None))]
    fn set_console(&self, py: Python<'_>, live: bool, write: Option<Py<PyAny>>, error: Option<Py<PyAny>>, read: Option<Py<PyAny>>) {
        let rt = unsafe { &*self.rt.get() };
        let mut p = rt.shared.platform.borrow_mut();
        if !live {
            p.sink = None;
            p.source = None;
            return;
        }
        let _ = py;
        p.sink = Some(Box::new(move |bytes: &[u8], is_err: bool| {
            let text = String::from_utf8_lossy(bytes).into_owned();
            Python::with_gil(|py| {
                let f = if is_err { error.as_ref().or(write.as_ref()) } else { write.as_ref() };
                let r = match f {
                    Some(f) => f.bind(py).call1((text,)).map(|_| ()),
                    None => py.import("sys").and_then(|m| m.getattr(if is_err { "stderr" } else { "stdout" })).and_then(|o| o.call_method1("write", (text,))).map(|_| ()),
                };
                if let Err(e) = r {
                    e.print(py);
                }
            })
        }));
        let spare = self.console_spare.clone();
        spare.borrow_mut().clear();
        p.source = Some(Box::new(move |buf: &mut [u8]| {
            let mut spare = spare.borrow_mut();
            if spare.is_empty() {
                let got = Python::with_gil(|py| -> PyResult<Vec<u8>> {
                    let r = match &read {
                        Some(f) => f.bind(py).call1((buf.len(),))?,
                        None => py.import("sys")?.getattr("stdin")?.call_method0("readline")?,
                    };
                    if let Ok(s) = r.downcast::<PyString>() {
                        return Ok(s.to_str()?.as_bytes().to_vec());
                    }
                    r.extract::<Vec<u8>>()
                });
                match got {
                    Ok(b) => *spare = b,
                    Err(e) => {
                        Python::with_gil(|py| e.print(py));
                        return 0;
                    }
                }
            }
            let n = spare.len().min(buf.len());
            buf[..n].copy_from_slice(&spare[..n]);
            spare.drain(..n);
            n
        }));
    }

    /// Read one data object from the start of `text` as `.r` does, and say how much it used: `(value, used, complete)`, `used` in characters of `text`
    /// (a host that read `text` from a stream puts back the rest with `seek(-(len(text) - used), 1)`, or by the characters' encoding in bytes), `complete`
    /// False when the text ended inside the object (an open `[`) or where an atom could go on (`12`): a caller that can read more does, and asks again.
    /// None when the text has no object (blank). Nothing else is evaluated. A text that does not start an object is a `NativeError`.
    fn read_datum(&self, py: Python<'_>, text: &str) -> PyResult<Option<(PyObject, usize, bool)>> {
        let owner = self.owner(py);
        self.with_ip(py, |ip| -> PyResult<Option<(PyObject, usize, bool)>> {
            match ip.read_datum_ex(text).map_err(k2py)? {
                Some((used, v, complete)) => {
                    let chars = text.get(..used).map(|t| t.chars().count()).unwrap_or(used);
                    Ok(Some((to_py(py, ip, &v, Some(&owner))?.unbind(), chars, complete)))
                }
                None => Ok(None),
            }
        })
    }

    /// The text the console channel read from the live console's `read` and the program has not used (what `.r` or `.rl` left of the last chunk),
    /// taken out of the channel and of the reader. A host that reads the console from a stream of its own gives it back to the stream; the program
    /// then no longer has it. Empty when the console is not live.
    fn take_console_unread(&self) -> String {
        let rt = unsafe { &*self.rt.get() };
        let mut bytes = rt.take_console_unread();
        bytes.extend(self.console_spare.borrow_mut().drain(..));
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// The module the program is in (`.module(:name)`): its name, or None outside any.
    fn module(&self) -> Option<String> {
        let rt = unsafe { &*self.rt.get() };
        rt.interp.current_module().map(String::from)
    }

    /// What the program has declared with `.extern`: a list of `(name, signature, module)` in order of declaration, the signature in its
    /// canonical text.
    fn declared_externs(&self) -> Vec<(String, String, String)> {
        let rt = unsafe { &*self.rt.get() };
        rt.declarations.externs().into_iter().map(|(n, s, m)| (n, s.to_string(), m)).collect()
    }

    /// `.module(:name)` for a host: the programs run from now on are read in the module `name`, and the names they define at top level belong to
    /// it, until `stop_module`. `module()` is the name.
    fn start_module(&self, py: Python<'_>, name: &str) -> PyResult<()> {
        self.with_ip(py, |ip| ip.set_module(Some(name))).map_err(k2py)
    }

    /// `.module(0)`: leave the module.
    fn stop_module(&self, py: Python<'_>) -> PyResult<()> {
        self.with_ip(py, |ip| ip.set_module(None)).map_err(k2py)
    }

    /// What the program has declared with `.param`: a list of `(name, type, default, value)` in order of declaration (the registry the host reads
    /// between runs). `type` is the canonical text (`"float"`, `"fixed<32,16>"`); `default` and `value` are host values (an int or a float; a fixed-point
    /// value as `(elem, raw)`); `value` is None when the name is not defined now.
    fn params<'py>(&self, py: Python<'py>) -> PyResult<Vec<(String, String, Bound<'py, PyAny>, Option<Bound<'py, PyAny>>)>> {
        params_rows(py, unsafe { &*self.rt.get() })
    }

    /// Set the parameter `name` (declared with `.param`) between runs: the value is checked against the declared type and converted (`int` takes an
    /// int, `float` an int or a float, `fixed<W,I>` a number or a fixed-point value). Returns the value as converted, as `params` shows values.
    /// A name that is not a parameter is a `NativeError` of kind `name`.
    fn set_param<'py>(&self, py: Python<'py>, name: String, value: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let v = to_host(&value);
        let r = self.with_ip(py, |_| klong_static::host::set_param_rt(unsafe { &mut *self.rt.get() }, &name, &v)).map_err(host_err)?;
        from_host(py, &r)
    }

    /// What the program has declared with `.entry`: a list of `(name, signature)` in order of declaration.
    fn declared_entries(&self) -> Vec<(String, String)> {
        let rt = unsafe { &*self.rt.get() };
        rt.declarations.entries().into_iter().map(|(n, s)| (n, s.to_string())).collect()
    }

    /// Install `lookup(name)`: the Python stand-in of a declared extern (a callable, or None when there is none), asked at each call of
    /// the extern, so a stand-in can be set, replaced or removed after the declaration. The callable gets the checked arguments as Python
    /// values and its result is checked against the signature. `lookup(".fixed")` is asked when a program calls `.fixed`: its
    /// callable makes the fixed-point array (the result stays the Python object it is). `None` removes the lookup.
    fn set_stand_in_lookup(&self, lookup: Option<Py<PyAny>>) {
        let rt = unsafe { &*self.rt.get() };
        rt.declarations.set_resolver(lookup.map(|lookup| {
            let r: klong_rt::declare::Resolver = Rc::new(move |ip: &mut Interp, name: &str, args: &[Value]| {
                Python::with_gil(|py| -> Option<PyResult<Value>> {
                    let f = match lookup.bind(py).call1((name,)) {
                        Ok(f) => f,
                        Err(e) => return Some(Err(e)),
                    };
                    if f.is_none() {
                        return None;
                    }
                    Some((|| {
                        let mut pa = Vec::with_capacity(args.len());
                        for a in args {
                            pa.push(to_py_owner(py, ip, a)?);
                        }
                        let r = f.call1(PyTuple::new(py, pa)?)?;
                        from_py_host(py, ip, &r)
                    })())
                })
                .map(|r| r.map_err(py2k))
            });
            r
        }));
    }

    /// Install `f(warnings)`, called for every program run at the top level once it has parsed and before its first statement runs, with
    /// what `take_warnings` gives (nothing is left to take afterwards, and `f` is not called for a program without warnings). When `f`
    /// raises, the run ends with that exception and no statement has run (a warning filter set to `error`). `None` removes it.
    fn set_warning_hook(&self, py: Python<'_>, f: Option<Py<PyAny>>) {
        self.with_ip(py, |ip| {
            ip.on_parsed = f.map(|f| {
                let hook: Rc<dyn Fn(&mut Interp) -> klong_core::error::Result<()>> = Rc::new(move |ip: &mut Interp| {
                    let ws = warnings_of(ip);
                    if ws.is_empty() {
                        return Ok(());
                    }
                    Python::with_gil(|py| f.bind(py).call1((ws,)).map(|_| ())).map_err(py2k)
                });
                hook
            });
        });
    }

    /// The syntax warnings the parser recorded since the last call (strict level 0: input the legacy grammar accepts and KlongPy
    /// warns about), in source order: a list of (code, message, hint or None, start, end, source name, source text). `message` is
    /// KlongPy's text of the warning (the condition, then `; ` and what happens to the input); `start` and `end` are offsets in characters
    /// of the source text, the same as an error's span.
    fn take_warnings(&self, py: Python<'_>) -> Vec<WarningTuple> {
        self.with_ip(py, warnings_of)
    }

    /// Call the function a symbol's value is: `name` is a global; when its value is a symbol, that symbol's value is the function
    /// (as `:name` called with arguments is in Klong). A name that is not defined is a `KeyError`.
    #[pyo3(signature = (name, *args))]
    fn call_sym(&self, py: Python<'_>, name: &str, args: &Bound<'_, PyTuple>) -> PyResult<PyObject> {
        let mut f = self.with_ip(py, |ip| ip.global(name)).ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(name.to_string()))?;
        for _ in 0..16 {
            let next = match &f {
                Value::Sym(s) => {
                    let n = self.with_ip(py, |ip| ip.syms.name(*s).to_string());
                    self.with_ip(py, |ip| ip.global(&n))
                }
                _ => None,
            };
            match next {
                Some(v) => f = v,
                None => break,
            }
        }
        self.apply(py, &f, args)
    }

    /// Native stack used at most, bytes.
    fn max_stack(&self, py: Python<'_>) -> usize {
        self.with_ip(py, |ip| ip.max_stack_seen)
    }
}

#[pyfunction]
fn zero_copy(on: bool) {
    ZERO_COPY.store(on, std::sync::atomic::Ordering::Relaxed);
}


// ----- the tools: lint ------------------------------------------------------------------------------------------------------------

/// A set of names or codes from Python: None, a string (comma- or space-separated) or any iterable of strings.
fn names_arg(o: Option<&Bound<'_, PyAny>>) -> PyResult<Option<Vec<String>>> {
    let Some(o) = o else { return Ok(None) };
    if o.is_none() {
        return Ok(None);
    }
    if let Ok(s) = o.extract::<String>() {
        return Ok(Some(s.split(|c: char| c == ',' || c.is_whitespace()).filter(|s| !s.is_empty()).map(String::from).collect()));
    }
    let mut out = Vec::new();
    for item in o.try_iter()? {
        out.push(item?.extract::<String>()?);
    }
    Ok(Some(out))
}

/// Run `f` on a thread with a large stack (the parser recurses on the native stack, as the `klong` binary's does), with the GIL released.
fn on_big_stack<T: Send + 'static>(py: Python<'_>, f: impl FnOnce() -> T + Send + 'static) -> PyResult<T> {
    py.allow_threads(move || {
        std::thread::Builder::new()
            .stack_size(256 << 20)
            .spawn(f)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("cannot start the lint thread: {e}")))?
            .join()
            .map_err(|_| pyo3::exceptions::PyRuntimeError::new_err("internal error in the linter"))
    })
}

/// `kgpy lint`: lint files and print the findings. Returns `(status, stdout, stderr)`; status as `klong lint`'s: 1 when an error was
/// found, 2 when a file could not be read or an option was wrong, else 0. `select`, `ignore` and `define` are strings or iterables of
/// codes or names; `lib_dir` is the library directory searched after KLONGPATH (None: the library built in); `extra_builtins` the
/// system names the Python package binds beyond the runtime's own.
#[pyfunction]
#[pyo3(signature = (paths, select=None, ignore=None, define=None, verbose=false, lib_dir=None, extra_builtins=None))]
fn lint_files(
    py: Python<'_>,
    paths: Vec<String>,
    select: Option<Bound<'_, PyAny>>,
    ignore: Option<Bound<'_, PyAny>>,
    define: Option<Bound<'_, PyAny>>,
    verbose: bool,
    lib_dir: Option<String>,
    extra_builtins: Option<Bound<'_, PyAny>>,
) -> PyResult<(i32, String, String)> {
    let join = |v: Option<Vec<String>>| v.map(|v| v.join(","));
    let select = join(names_arg(select.as_ref())?);
    let ignore = join(names_arg(ignore.as_ref())?);
    let define = join(names_arg(define.as_ref())?);
    let extra: std::collections::BTreeSet<String> = names_arg(extra_builtins.as_ref())?.unwrap_or_default().into_iter().collect();
    let lib_dirs: Vec<String> = lib_dir.into_iter().collect();
    on_big_stack(py, move || {
        let (mut out, mut err) = (Vec::new(), Vec::new());
        let status =
            klong_lint::run_files("kgpy lint", &paths, select.as_deref(), ignore.as_deref(), define.as_deref(), verbose, lib_dirs, extra, &mut out, &mut err);
        (status, String::from_utf8_lossy(&out).into_owned(), String::from_utf8_lossy(&err).into_owned())
    })
}

/// A finding: `(code, severity, start, end, line, col, message, hint)`; start and end are offsets and col a column, all in characters
/// of the text as Python indexes a str; line and col count from 1; hint is None when there is none.
type Finding = (String, String, usize, usize, usize, usize, String, Option<String>);

/// Lint a program text (nothing in it is evaluated and no other file is run): the findings sorted by position. `name` is the name in the
/// findings and, when it is a file, where `.l("file")` is looked up (its directory; or `base_dir`); `resolve=False` never reads another
/// file (every `.l` then counts as one that defines unknown names). A code that does not exist raises ValueError.
#[pyfunction]
#[pyo3(signature = (name, text, select=None, ignore=None, define=None, base_dir=None, resolve=true, lib_dir=None, extra_builtins=None))]
#[allow(clippy::too_many_arguments)]
fn lint_source(
    py: Python<'_>,
    name: String,
    text: String,
    select: Option<Bound<'_, PyAny>>,
    ignore: Option<Bound<'_, PyAny>>,
    define: Option<Bound<'_, PyAny>>,
    base_dir: Option<String>,
    resolve: bool,
    lib_dir: Option<String>,
    extra_builtins: Option<Bound<'_, PyAny>>,
) -> PyResult<Vec<Finding>> {
    let codes = |v: Option<Vec<String>>| -> PyResult<Option<std::collections::BTreeSet<String>>> {
        v.map(|v| klong_lint::parse_codes(&v.join(",")).map_err(pyo3::exceptions::PyValueError::new_err)).transpose()
    };
    let opts = klong_lint::Options {
        select: codes(names_arg(select.as_ref())?)?,
        ignore: codes(names_arg(ignore.as_ref())?)?.unwrap_or_default(),
        define: names_arg(define.as_ref())?.unwrap_or_default().into_iter().collect(),
        extra_builtins: names_arg(extra_builtins.as_ref())?.unwrap_or_default().into_iter().collect(),
        base_dir,
        no_resolve: !resolve,
        lib_dirs: lib_dir.into_iter().collect(),
    };
    on_big_stack(py, move || {
        let found = klong_lint::lint_source(&name, &text, &opts);
        // byte offsets to character offsets
        let ascii = text.is_ascii();
        let starts: Vec<usize> = if ascii { Vec::new() } else { text.char_indices().map(|(b, _)| b).collect() };
        let ch = |b: usize| if ascii { b } else { starts.partition_point(|&s| s < b) };
        found
            .into_iter()
            .map(|d| (d.code.to_string(), d.severity.to_string(), ch(d.span.0), ch(d.span.1), d.line, d.col, d.message, d.hint))
            .collect()
    })
}

// ----- the tools that read Klong text: ast, REPL input, IPC function text ---------------------------------------------------------

/// A `klong_tools::ToolError` as the exception the engine raises (`NativeError`): args = (kind, message, start, end, source name, source
/// text, hint), the offsets in characters of the source text (start is -1 when the error has no span).
fn tool_err(e: klong_tools::ToolError) -> PyErr {
    let (a, b) = match e.span {
        Some((a, b)) => {
            let co = klong_tools::CharOffsets::new(&e.source_text);
            (co.at(a) as i64, co.at(b) as i64)
        }
        None => (-1, -1),
    };
    NativeError::new_err((e.kind.to_string(), e.message, a, b, e.source_name, e.source_text, e.hint))
}

/// `kgpy ast`: the canonical parse tree of a program (`klongpy/dump.py`: `dump(parse_program(...))`), one line per statement. Nothing runs.
/// A text that does not parse raises `NativeError`, as evaluating it would.
#[pyfunction]
#[pyo3(signature = (text, name=None))]
fn ast_text(py: Python<'_>, text: String, name: Option<String>) -> PyResult<String> {
    let name = name.unwrap_or_else(|| "<string>".to_string());
    py.allow_threads(move || klong_tools::ast::ast_text(&text, &name)).map_err(tool_err)
}

/// The REPL's completeness check of an entry (`klongpy/repl_input.py`, `check_text`): "complete", "incomplete" or "error".
/// `comments=False` leaves out the two things only the text shows (a `:"` comment or `.comment("X")` left open), as
/// `check_complete(tokens)` without a source does.
#[pyfunction]
#[pyo3(signature = (text, comments=true))]
fn repl_check(py: Python<'_>, text: String, comments: bool) -> PyResult<&'static str> {
    on_big_stack(py, move || klong_tools::repl::check_complete(&text, comments).name())
}

/// Tab completion (`Completer.matches`): `(start, words)` for the text before the cursor, `start` in characters. `names` are the defined
/// names, `commands` the `]` commands without the bracket; `klongpath` (None: the environment's KLONGPATH) lists the directories searched
/// for `.kg` files. Reads the file system, evaluates nothing.
#[pyfunction]
#[pyo3(signature = (line, names, commands, klongpath=None))]
fn repl_complete(line: &str, names: Vec<String>, commands: Vec<String>, klongpath: Option<String>) -> (usize, Vec<String>) {
    let kp = klongpath.unwrap_or_else(|| std::env::var("KLONGPATH").unwrap_or_default());
    klong_tools::repl::complete(line, &names, &commands, &kp)
}

/// `parse_function` of the IPC endpoint, the parse step: `(named, arity, params, locals)` when the text is exactly one function literal,
/// plain (`{x+y}`) or named (`fn(a;b)::{a+b}`), else None. Nothing runs and nothing is defined; the function itself is then built by
/// evaluating `fn::TEXT` (plain) or TEXT (named) and restoring `fn`.
#[pyfunction]
fn function_form(py: Python<'_>, text: String) -> Option<(bool, u8, Option<Vec<String>>, Vec<String>)> {
    py.allow_threads(move || klong_tools::func::function_form(&text)).map(|f| (f.named, f.arity, f.params, f.locals))
}

/// The text a function goes out as over IPC: its literal, with `fn(a;b)::` in front when it has named parameters. None when the text
/// is not a literal (`{...}`).
#[pyfunction]
#[pyo3(signature = (literal, params=None))]
fn function_wire_text(literal: &str, params: Option<Vec<String>>) -> Option<String> {
    klong_tools::func::wire_text(literal, params.as_deref())
}

/// The arity named by a display form (`:nilad`, `:monad`, `:dyad`, `:triad`, `:fn/5`), else None.
#[pyfunction]
fn function_display_arity(text: &str) -> Option<u32> {
    klong_tools::func::display_arity(text)
}

// ----- the tools: the static profile ------------------------------------------------------------------------------------------------

/// `kgpy check`: check files against the static profile and print what `kgpy check` prints. Returns `(status, stdout, stderr)`; status
/// 1 when a file is outside the profile, 2 when a file could not be read, else 0. `streamable` is `--streamable` (`--dump-ir` is `dump_ir_files`).
#[pyfunction]
#[pyo3(signature = (paths, int64=false, verbose=false, lib_dir=None, streamable=false))]
fn check_files(py: Python<'_>, paths: Vec<String>, int64: bool, verbose: bool, lib_dir: Option<String>, streamable: bool) -> PyResult<(i32, String, String)> {
    let lib_dirs: Vec<String> = lib_dir.into_iter().collect();
    on_big_stack(py, move || {
        let (mut out, mut err) = (Vec::new(), Vec::new());
        let status = klong_static::cli::run_check_ex("kgpy check", &paths, int64, verbose, streamable, lib_dirs, &mut out, &mut err);
        (status, String::from_utf8_lossy(&out).into_owned(), String::from_utf8_lossy(&err).into_owned())
    })
}

/// A violation of the profile: `(rule, start, end, line, col, message, hint, function)`; offsets and column in characters; `function` is the
/// top-level function the span lies in, or None.
type StaticDiag = (String, usize, usize, usize, usize, String, Option<String>, Option<String>);
/// What a check found beyond its diagnostics: `(entries, externs, globals, functions)`.
type StaticInfo = (Vec<(String, String)>, Vec<(String, String, String)>, Vec<(String, Option<String>, bool, bool, bool)>, Vec<(String, bool, bool, Vec<(Vec<Option<String>>, Option<String>)>)>);

/// The values of compile-time definitions a caller evaluated, `{name: value}`, as the checker is told them (`check_source(globals=)` of the
/// Python package): an int, a float, a `bool`, a `str`, a `dict`, a callable, `None` (`:undefined`), a list, tuple, array or tensor of numbers, a
/// fixed-point array (an object with `spec.elem` and `raw`), or `(elem, shape)`, a type already known (`("float", (3,))`). Each is validated as a
/// global: a string, a ragged list, an int beyond the range of kg_int are reported as the checker reports them.
fn globals_arg(g: Option<&Bound<'_, PyAny>>) -> PyResult<HashMap<String, klong_static::check::LitVal>> {
    use pyo3::types::PyDict;
    let mut out = HashMap::new();
    let Some(g) = g else { return Ok(out) };
    if g.is_none() {
        return Ok(out);
    }
    for (k, v) in g.downcast::<PyDict>()?.iter() {
        out.insert(k.extract::<String>()?, litval_of(&v)?);
    }
    Ok(out)
}

fn litval_of(obj: &Bound<'_, PyAny>) -> PyResult<klong_static::check::LitVal> {
    use klong_static::ast::{Bad, Lit};
    use klong_static::check::LitVal;
    use klong_static::sig::Ty;
    use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyString, PyTuple};
    if obj.is_none() {
        return Ok(LitVal::Undefined);
    }
    if obj.is_instance_of::<PyBool>() {
        return Ok(LitVal::Int(obj.extract::<bool>()? as i128));
    }
    if obj.is_instance_of::<PyInt>() {
        return Ok(LitVal::Int(obj.extract::<i128>().unwrap_or(if obj.lt(0)? { i128::MIN } else { i128::MAX })));
    }
    if obj.is_instance_of::<PyFloat>() {
        return Ok(LitVal::Flt(obj.extract::<f64>()?));
    }
    if obj.is_instance_of::<PyString>() {
        return Ok(LitVal::Str);
    }
    if obj.is_instance_of::<PyDict>() {
        return Ok(LitVal::Dict);
    }
    if let Ok(h) = obj.downcast::<PyTuple>() {
        // (elem, shape): a type
        if h.len() == 2 {
            if let (Ok(elem), Ok(shape)) = (h.get_item(0)?.extract::<String>(), h.get_item(1)?.extract::<Vec<usize>>()) {
                return Ok(LitVal::Fixed(Ty::new(&elem, shape)));
            }
        }
    }
    if obj.hasattr("spec")? && obj.hasattr("raw")? {
        if let HostValue::Fixed { elem, shape, .. } = to_host(obj) {
            return Ok(LitVal::Fixed(Ty::new(&elem, shape)));
        }
    }
    if obj.hasattr("ndim")? && obj.hasattr("item")? && obj.getattr("ndim")?.extract::<usize>().ok() == Some(0) {
        return litval_of(&obj.call_method0("item")?);
    }
    if obj.is_callable() && !obj.hasattr("tolist")? && !obj.hasattr("__len__")? {
        return Ok(LitVal::Func);
    }
    match to_host(obj) {
        HostValue::Int(i) => Ok(LitVal::Int(i as i128)),
        HostValue::Float(f) => Ok(LitVal::Flt(f)),
        HostValue::Ints(shape, data) => Ok(LitVal::Arr(Lit::Num { shape, float: false, ints: data.into_iter().map(|x| x as i128).collect(), flts: Vec::new() })),
        HostValue::Floats(shape, data) => Ok(LitVal::Arr(Lit::Num { shape, float: true, ints: Vec::new(), flts: data })),
        HostValue::Other(w) if w == "an empty list" => Ok(LitVal::Arr(Lit::Num { shape: vec![0], float: false, ints: Vec::new(), flts: Vec::new() })),
        HostValue::Other(w) if w == "a ragged list" => Ok(LitVal::Arr(Lit::Bad(Bad::Ragged, Vec::new()))),
        HostValue::Other(_) => Ok(LitVal::Arr(Lit::Bad(Bad::Str(String::new()), Vec::new()))),
        _ => Ok(LitVal::Undefined),
    }
}

fn static_options(int64: bool, base_dir: Option<String>, resolve: bool, lib_dir: Option<String>, all_functions: bool, globals: HashMap<String, klong_static::check::LitVal>) -> klong_static::Options {
    klong_static::Options { int_bits: if int64 { 64 } else { 32 }, resolve, all_functions, base_dir, lib_dirs: lib_dir.into_iter().collect(), globals }
}

/// Check a program text against the static profile (nothing in it is evaluated): its diagnostics in source order, the declarations and the
/// functions with the argument types each was checked with. `name` is the name in the diagnostics and where `.l("file")` is looked up.
/// `globals` is `{name: value}`, the values of compile-time definitions the caller evaluated (see `globals_arg`).
#[pyfunction]
#[pyo3(signature = (name, text, int64=false, base_dir=None, resolve=true, lib_dir=None, all_functions=false, globals=None))]
#[allow(clippy::too_many_arguments)]
fn check_source(py: Python<'_>, name: String, text: String, int64: bool, base_dir: Option<String>, resolve: bool, lib_dir: Option<String>, all_functions: bool, globals: Option<Bound<'_, PyAny>>) -> PyResult<(Vec<StaticDiag>, StaticInfo)> {
    let globals = globals_arg(globals.as_ref())?;
    on_big_stack(py, move || {
        let opts = static_options(int64, base_dir, resolve, lib_dir, all_functions, globals);
        let r = klong_static::check_source(&text, &name, &opts);
        let diags = static_diags(&r);
        let entries = r.entries.iter().map(|e| (e.name.clone(), e.signature.to_string())).collect();
        let externs = r.externs.values().map(|x| (x.name.clone(), x.signature.to_string(), x.module.clone())).collect();
        let globals = r.globals.values().map(|g| (g.name.clone(), g.ty().map(|t| t.to_string()), g.assigned, g.resolved, g.param)).collect();
        let functions = r.functions.values().map(|f| (f.name.clone(), f.reachable, f.in_profile, specs_of(f))).collect();
        (diags, (entries, externs, globals, functions))
    })
}

type Specs = Vec<(Vec<Option<String>>, Option<String>)>;

fn specs_of(f: &klong_static::check::FunctionInfo) -> Specs {
    f.specializations.iter().map(|(a, b)| (a.iter().map(|t| t.as_ref().map(|t| t.to_string())).collect(), b.as_ref().map(|t| t.to_string()))).collect()
}

/// `(start, end)` of a span of the source `src` of a report, in characters of that source's text.
fn report_chars(r: &klong_static::Report, src: usize, sp: (usize, usize)) -> (usize, usize) {
    let t = &r.sources[src].source.text;
    if t.is_ascii() {
        return sp;
    }
    let ch = |b: usize| t.get(..b.min(t.len())).map(|p| p.chars().count()).unwrap_or(b);
    (ch(sp.0), ch(sp.1))
}

/// The whole report of the checker as a dict (`StaticReport` of the Python package, minus the parse nodes). Spans are `(start, end)` in
/// characters of the text of the source they lie in; `src` indexes `sources`, which lists the main text first and then every file `.l` loaded.
///
///     diagnostics  [StaticDiag]                                         as `check_source`
///     sources      [(name, text)]
///     statements   [(kind, name, start, end, src)]                      in the order they run; kind "function", "global", "extern", "entry", "param" or "load"
///     functions    [(name, reachable, in_profile, specializations, start, end, src, params, calls)]
///     globals      [(name, type or None, assigned, resolved, param, start, end, src)]
///     externs      [(name, signature, module, start, end, src)]
///     entries      [(name, signature, start, end, src)]
///     params       [(name, type)]
#[pyfunction]
#[pyo3(signature = (name, text, int64=false, base_dir=None, resolve=true, lib_dir=None, all_functions=false, globals=None))]
#[allow(clippy::too_many_arguments)]
fn check_report(py: Python<'_>, name: String, text: String, int64: bool, base_dir: Option<String>, resolve: bool, lib_dir: Option<String>, all_functions: bool, globals: Option<Bound<'_, PyAny>>) -> PyResult<PyObject> {
    use pyo3::types::PyDict;
    let globals = globals_arg(globals.as_ref())?;
    let r = on_big_stack(py, move || {
        let opts = static_options(int64, base_dir, resolve, lib_dir, all_functions, globals);
        let r = klong_static::check_source(&text, &name, &opts);
        let c = |src: usize, sp: (usize, usize)| report_chars(&r, src, sp);
        let sources: Vec<(String, String)> = r.sources.iter().map(|s| (s.name.clone(), s.source.text.clone())).collect();
        let statements: Vec<(String, String, usize, usize, usize)> = r.statements.iter().map(|s| { let (a, b) = c(s.src, s.span); (s.kind.to_string(), s.name.clone(), a, b, s.src) }).collect();
        let functions: Vec<(String, bool, bool, Specs, usize, usize, usize, Vec<String>, Vec<String>)> = r
            .functions
            .values()
            .map(|f| { let (a, b) = c(f.src, f.span); (f.name.clone(), f.reachable, f.in_profile, specs_of(f), a, b, f.src, f.params.clone(), f.calls.clone()) })
            .collect();
        let globals: Vec<(String, Option<String>, bool, bool, bool, usize, usize, usize)> = r
            .globals
            .values()
            .map(|g| { let (a, b) = c(g.src, g.span); (g.name.clone(), g.ty().map(|t| t.to_string()), g.assigned, g.resolved, g.param, a, b, g.src) })
            .collect();
        let externs: Vec<(String, String, String, usize, usize, usize)> = r.externs.values().map(|x| { let (a, b) = c(x.src, x.span); (x.name.clone(), x.signature.to_string(), x.module.clone(), a, b, x.src) }).collect();
        let entries: Vec<(String, String, usize, usize, usize)> = r.entries.iter().map(|e| { let (a, b) = c(e.src, e.span); (e.name.clone(), e.signature.to_string(), a, b, e.src) }).collect();
        let params: Vec<(String, String)> = r.params.items.iter().map(|(k, t)| (k.clone(), t.to_string())).collect();
        (static_diags(&r), sources, statements, functions, globals, externs, entries, params)
    })?;
    let d = PyDict::new(py);
    d.set_item("diagnostics", r.0)?;
    d.set_item("sources", r.1)?;
    d.set_item("statements", r.2)?;
    d.set_item("functions", r.3)?;
    d.set_item("globals", r.4)?;
    d.set_item("externs", r.5)?;
    d.set_item("entries", r.6)?;
    d.set_item("params", r.7)?;
    Ok(d.into())
}


/// Diagnostics of a report as `StaticDiag` tuples (offsets in characters).
fn static_diags(r: &klong_static::Report) -> Vec<StaticDiag> {
    r.diagnostics.iter().map(|d| static_diag(d, r)).collect()
}

fn static_diag(d: &klong_static::check::Diag, r: &klong_static::Report) -> StaticDiag {
    let src = &r.sources[d.src];
    let t = &src.source.text;
    let ch = |b: usize| t.get(..b.min(t.len())).map(|p| p.chars().count()).unwrap_or(b);
    let (line, col) = src.source.line_col(d.span.0);
    (d.rule.to_string(), ch(d.span.0), ch(d.span.1), line, col, d.message.clone(), d.hint.clone(), d.function.clone())
}

/// What a compile (stage one to lowering) gives: `(diags, error, program)`. `diags` are the rejections of the profile in source order (a
/// lowering stops at the first); `error` is the text of an error of the interpreter while it evaluated the compile-time definitions.
type CompileOutcome = (Vec<StaticDiag>, Option<String>);

fn compile_failure(f: klong_static::compile::Failure) -> CompileOutcome {
    use klong_static::compile::Failure;
    match f {
        Failure::Check(r) => (static_diags(&r), None),
        Failure::Lower(fail, r) => {
            let d = klong_static::check::Diag { rule: fail.rule, span: fail.span, src: fail.src, message: fail.message, hint: fail.hint, function: None };
            (vec![static_diag(&d, &r)], None)
        }
        Failure::Run(e) => (Vec::new(), Some(e.render())),
        Failure::Read(m) => (Vec::new(), Some(m)),
    }
}

/// `kgpy check --dump-ir` of a program text: `(ok, text, diags, error)`; `text` is the typed IR. `prefix` names the program in the IR. `lib_dir` is the
/// library directory a `.l` of a library file is looked up in (after KLONGPATH); a file loaded from it is reported with its path.
#[pyfunction]
#[pyo3(signature = (name, text, int64=false, prefix=None, lib_dir=None))]
fn dump_ir_source(py: Python<'_>, name: String, text: String, int64: bool, prefix: Option<String>, lib_dir: Option<String>) -> PyResult<(bool, String, Vec<StaticDiag>, Option<String>)> {
    let libs: Vec<String> = lib_dir.into_iter().collect();
    on_big_stack(py, move || match klong_static::compile::compile_program_in(&name, Some(&text), if int64 { 64 } else { 32 }, prefix.as_deref(), HashMap::new(), &libs) {
        Ok(c) => (true, klong_static::ir::dump(&c.program), Vec::new(), None),
        Err(f) => {
            let (d, e) = compile_failure(f);
            (false, String::new(), d, e)
        }
    })
}

/// The typed IR of a program text as a structure: `(ok, ir, diags, error)` where `ir` is JSON text (`klong_static::ir_json`: program, globals, externs,
/// entries and functions with every node typed and spanned in characters). Like `dump_ir_source` otherwise.
#[pyfunction]
#[pyo3(signature = (name, text, int64=false, prefix=None, lib_dir=None))]
fn ir_json_source(py: Python<'_>, name: String, text: String, int64: bool, prefix: Option<String>, lib_dir: Option<String>) -> PyResult<(bool, String, Vec<StaticDiag>, Option<String>)> {
    let libs: Vec<String> = lib_dir.into_iter().collect();
    on_big_stack(py, move || match klong_static::compile::compile_program_in(&name, Some(&text), if int64 { 64 } else { 32 }, prefix.as_deref(), HashMap::new(), &libs) {
        Ok(c) => (true, klong_static::ir_json::program_json(&c.program, &c.report), Vec::new(), None),
        Err(f) => {
            let (d, e) = compile_failure(f);
            (false, String::new(), d, e)
        }
    })
}

/// The size report of an emitted program: `(numbers, call_chain, entries, externs, need_math, text)`; `numbers` is a list of `(name, value)` for
/// int_bits, float_bits, const_bytes, instance_bytes, state_bytes, slot_bytes, call_depth, stack_bytes and c_lines.
type SizeReportT = (Vec<(String, usize)>, Vec<String>, Vec<String>, Vec<String>, bool, String);

fn size_report(e: &klong_static::emit_c::Emitted) -> SizeReportT {
    let r = &e.report;
    let n = |k: &str, v: usize| (k.to_string(), v);
    (
        vec![n("int_bits", r.int_bits as usize), n("float_bits", r.float_bits as usize), n("const_bytes", r.const_bytes), n("instance_bytes", r.instance_bytes), n("state_bytes", r.state_bytes), n("slot_bytes", r.slot_bytes), n("call_depth", r.call_depth), n("stack_bytes", r.stack_bytes), n("c_lines", r.c_lines)],
        r.call_chain.clone(),
        r.entries.clone(),
        r.externs.clone(),
        r.need_math,
        e.report_text(),
    )
}

/// `kgpy compile c` of a program text: `(ok, prefix, c, h, report, diags, error)`. The files `<prefix>.c`, `<prefix>.h` and `klong_rt.h`
/// (see `runtime_header`) are the caller's to write. `error` is the text of a refusal that is not a diagnostic (a bad prefix: the ValueError
/// of the Python emitter) or of the interpreter.
#[pyfunction]
#[pyo3(signature = (name, text, int64=false, float64=false, line_directives=false, prefix=None, lib_dir=None))]
#[allow(clippy::too_many_arguments)]
fn compile_c_source(py: Python<'_>, name: String, text: String, int64: bool, float64: bool, line_directives: bool, prefix: Option<String>, lib_dir: Option<String>) -> PyResult<(bool, String, String, String, Option<SizeReportT>, Vec<StaticDiag>, Option<String>)> {
    use klong_static::compile::EmitFailure;
    let libs: Vec<String> = lib_dir.into_iter().collect();
    on_big_stack(py, move || {
        match klong_static::compile::emit_program_in(&name, None, Some(&text), if int64 { 64 } else { 32 }, if float64 { 64 } else { 32 }, line_directives, prefix.as_deref(), None, &libs) {
            Ok(r) => (true, r.emitted.prefix.clone(), r.emitted.c.clone(), r.emitted.h.clone(), Some(size_report(&r.emitted)), Vec::new(), None),
            Err(EmitFailure::Compile(f)) => {
                let (d, e) = compile_failure(f);
                (false, String::new(), String::new(), String::new(), None, d, e)
            }
            Err(EmitFailure::Emit(klong_static::emit_c::EmitError::Static(message, hint, span, src), r)) => {
                let d = klong_static::check::Diag { rule: "extern", span, src, message, hint: Some(hint), function: None };
                (false, String::new(), String::new(), String::new(), None, vec![static_diag(&d, &r)], None)
            }
            Err(f @ EmitFailure::Emit(..)) | Err(f @ EmitFailure::Io(..)) => {
                let (lines, _) = f.render(&name);
                (false, String::new(), String::new(), String::new(), None, Vec::new(), Some(lines.join("\n")))
            }
        }
    })
}

/// `klong_rt.h`: the runtime header every compiled program includes.
#[pyfunction]
fn runtime_header() -> &'static str {
    klong_static::compile::RUNTIME_HEADER
}

/// `kgpy compile c FILE... -o DIR` (and `--dump-ir`): what it prints. Returns `(status, stdout, stderr)`.
#[pyfunction]
#[pyo3(signature = (paths, output=None, int64=false, float64=false, line_directives=false, prefix=None, dump_ir=false, lib_dir=None))]
#[allow(clippy::too_many_arguments)]
fn compile_c_files(py: Python<'_>, paths: Vec<String>, output: Option<String>, int64: bool, float64: bool, line_directives: bool, prefix: Option<String>, dump_ir: bool, lib_dir: Option<String>) -> PyResult<(i32, String, String)> {
    let libs: Vec<String> = lib_dir.into_iter().collect();
    on_big_stack(py, move || {
        let (mut out, mut err) = (Vec::new(), Vec::new());
        let status = klong_static::compile::emit_main_in(&paths, output.as_deref(), if int64 { 64 } else { 32 }, if float64 { 64 } else { 32 }, line_directives, prefix.as_deref(), dump_ir, &libs, &mut out, &mut err);
        (status, String::from_utf8_lossy(&out).into_owned(), String::from_utf8_lossy(&err).into_owned())
    })
}

/// `kgpy check --dump-ir FILE...`: `(status, stdout, stderr)`.
#[pyfunction]
#[pyo3(signature = (paths, int64=false, lib_dir=None))]
fn dump_ir_files(py: Python<'_>, paths: Vec<String>, int64: bool, lib_dir: Option<String>) -> PyResult<(i32, String, String)> {
    let libs: Vec<String> = lib_dir.into_iter().collect();
    on_big_stack(py, move || {
        let (mut out, mut err) = (Vec::new(), Vec::new());
        let status = klong_static::compile::dump_ir_main_in(&paths, if int64 { 64 } else { 32 }, &libs, &mut out, &mut err);
        (status, String::from_utf8_lossy(&out).into_owned(), String::from_utf8_lossy(&err).into_owned())
    })
}

/// `kgpy check --streamable` of a program text: the findings of the streamable tier as `StaticDiag`, or the diagnostics/error of the compile.
/// Returns `(ok, findings, diags, error)`.
#[pyfunction]
#[pyo3(signature = (name, text, int64=false, lib_dir=None))]
fn streamable_source(py: Python<'_>, name: String, text: String, int64: bool, lib_dir: Option<String>) -> PyResult<(bool, Vec<StaticDiag>, Vec<StaticDiag>, Option<String>)> {
    let libs: Vec<String> = lib_dir.into_iter().collect();
    on_big_stack(py, move || match klong_static::compile::compile_program_in(&name, Some(&text), if int64 { 64 } else { 32 }, None, HashMap::new(), &libs) {
        Ok(c) => {
            let found = klong_static::streamable::check_streamable(&c.program, &c.report);
            (true, found.iter().map(|d| static_diag(d, &c.report)).collect(), Vec::new(), None)
        }
        Err(f) => {
            let (d, e) = compile_failure(f);
            (false, Vec::new(), d, e)
        }
    })
}

// ----- tables as data: lint rules, the static profile, signatures -------------------------------------------------------------------

/// The lint rules: `[(code, severity, summary)]` in code order.
#[pyfunction]
fn lint_rules() -> Vec<(String, String, String)> {
    klong_lint::RULES.iter().map(|r| (r.code.to_string(), r.severity.to_string(), r.title.to_string())).collect()
}

/// The tables of the static profile (`klong_static::tables`, the source of truth), as a dict:
///
///     rules          [(id, group, title, section, hint or None)]    every rule: the profile's, then the streamable tier's
///     stream_rules   [(id, group, title, section, hint or None)]    the streamable tier's alone
///     groups         [(id, title)]                                   the groups of the error table, in order
///     optable        [(id, glyph, valence, name, allowed, operands, type_rule, shape_rule, rule, note, examples)]
///     forms          [(id, form, note, examples)]
///     operand_kinds, type_rules, shape_rules    [(name, description)]
///
/// `examples` is `[(body, signature, inputs)]`: `f::{body}` entered with `signature` is inside the profile with the result type the signature names;
/// `inputs` is JSON text of the arguments a test calls it with, or None.
#[pyfunction]
fn static_tables(py: Python<'_>) -> PyResult<PyObject> {
    use klong_static::tables as t;
    let d = PyDict::new(py);
    let rule = |r: &t::Rule| (r.id, r.group, r.title, r.section, r.hint);
    let ex = |e: &[t::Example]| -> Vec<(&str, &str, Option<&str>)> { e.iter().map(|x| (x.body, x.signature, x.inputs)).collect() };
    d.set_item("rules", t::RULES.iter().map(rule).collect::<Vec<_>>())?;
    d.set_item("stream_rules", t::STREAM_RULES.iter().map(rule).collect::<Vec<_>>())?;
    d.set_item("groups", t::GROUPS.to_vec())?;
    d.set_item(
        "optable",
        t::OPTABLE.iter().map(|o| (o.id, o.glyph, o.valence, o.name, o.allowed, o.operands.to_vec(), o.type_rule, o.shape_rule, o.rule, o.note, ex(o.examples))).collect::<Vec<_>>(),
    )?;
    d.set_item("forms", t::FORMS.iter().map(|f| (f.id, f.form, f.note, ex(f.examples))).collect::<Vec<_>>())?;
    d.set_item("operand_kinds", t::OPERAND_KINDS.to_vec())?;
    d.set_item("type_rules", t::TYPE_RULES.to_vec())?;
    d.set_item("shape_rules", t::SHAPE_RULES.to_vec())?;
    Ok(d.into())
}

pyo3::create_exception!(klong_native, SignatureError, pyo3::exceptions::PyValueError);

/// A malformed signature as the exception the Python package raises: `args = (message, start, end)`, offsets into the signature text in characters.
fn sig_err(e: klong_static::sig::SigError) -> PyErr {
    SignatureError::new_err((e.message, e.span.0, e.span.1))
}

type TyT = (String, Vec<usize>);

fn ty_t(t: klong_static::sig::Ty) -> TyT {
    (t.elem, t.shape)
}

fn ty_of(t: TyT) -> klong_static::sig::Ty {
    klong_static::sig::Ty { elem: t.0, shape: t.1 }
}

/// One type such as `float[16,241]` as `(elem, shape)`. A malformed one raises `SignatureError`.
#[pyfunction]
fn parse_type(text: &str) -> PyResult<TyT> {
    klong_static::sig::parse_type(text).map(ty_t).map_err(sig_err)
}

/// `params -> result` as `([(elem, shape)], (elem, shape))`; `allow_str=False` is the rule for `.entry`. A malformed one raises `SignatureError`.
#[pyfunction]
#[pyo3(signature = (text, allow_str=true))]
fn parse_signature(text: &str, allow_str: bool) -> PyResult<(Vec<TyT>, TyT)> {
    klong_static::sig::parse_signature(text, allow_str).map(|s| (s.params.into_iter().map(ty_t).collect(), ty_t(s.result))).map_err(sig_err)
}

/// The text of a type: `format_type("float", [16, 241])` is `float[16,241]`.
#[pyfunction]
fn format_type(elem: String, shape: Vec<usize>) -> String {
    ty_of((elem, shape)).to_string()
}

/// The canonical text of a signature: `"int float[64] -> float"`.
#[pyfunction]
fn format_signature(params: Vec<TyT>, result: TyT) -> String {
    klong_static::sig::Signature { params: params.into_iter().map(ty_of).collect(), result: ty_of(result) }.to_string()
}

/// None if `name` can be declared with `.extern`, else the reason as one line.
#[pyfunction]
fn check_extern_name(name: &str) -> Option<String> {
    klong_static::sig::check_extern_name(name)
}

/// None if `name` can be declared with `.entry`, else the reason as one line.
#[pyfunction]
fn check_entry_name(name: &str) -> Option<String> {
    klong_static::sig::check_entry_name(name)
}

/// The C identifier of a Klong name: every `.` becomes `_`.
#[pyfunction]
fn c_identifier(name: &str) -> String {
    klong_static::sig::c_identifier(name)
}

/// `(W, I)` of an element type such as `fixed<32,16>`, else None.
#[pyfunction]
fn fixed_spec(elem: &str) -> Option<(u32, u32)> {
    klong_static::sig::fixed_spec(elem)
}

/// Is `value` acceptable as an initialised global (`validate_global` of the checker)? `((elem, shape), None, None)` when it is, else `(None, rule,
/// message)`. `value` as for `check_source(globals=)`.
#[pyfunction]
#[pyo3(signature = (value, int64=false))]
fn validate_global(value: &Bound<'_, PyAny>, int64: bool) -> PyResult<(Option<TyT>, Option<String>, Option<String>)> {
    let v = litval_of(value)?;
    Ok(match klong_static::check::validate_global(&v, if int64 { 64 } else { 32 }) {
        Ok(t) => (Some(ty_t(t)), None, None),
        Err((rule, message)) => (None, Some(rule.to_string()), Some(message)),
    })
}

/// The block of docs/static-profile.md that is generated from the rule tables, and the whole page.
#[pyfunction]
fn profile_doc_block() -> String {
    klong_static::profile_doc::generated_block()
}

#[pyfunction]
fn profile_doc_page() -> String {
    klong_static::profile_doc::render_page()
}


// ----- running a static program: call_entry, set_param ----------------------------------------------------------------------------------

use klong_static::host::{HostFn, HostValue, Loaded};

enum Leaf {
    I(i64),
    F(f64),
}

fn flatten(obj: &Bound<'_, PyAny>, depth: usize, shape: &mut Vec<usize>, out: &mut Vec<Leaf>) -> Result<(), String> {
    use pyo3::types::{PyBool, PyFloat, PyInt, PyList, PyTuple};
    if obj.is_instance_of::<PyBool>() {
        out.push(Leaf::I(obj.extract::<bool>().unwrap() as i64));
        return if depth == shape.len() { Ok(()) } else { Err("a ragged list".to_string()) };
    }
    if obj.is_instance_of::<PyInt>() {
        let v: i64 = obj.extract().map_err(|_| "an integer beyond 64 bits".to_string())?;
        out.push(Leaf::I(v));
        return if depth == shape.len() { Ok(()) } else { Err("a ragged list".to_string()) };
    }
    if obj.is_instance_of::<PyFloat>() {
        out.push(Leaf::F(obj.extract::<f64>().unwrap()));
        return if depth == shape.len() { Ok(()) } else { Err("a ragged list".to_string()) };
    }
    if obj.is_instance_of::<PyList>() || obj.is_instance_of::<PyTuple>() {
        let items: Vec<Bound<'_, PyAny>> = obj.try_iter().map_err(|e| e.to_string())?.map(|x| x.unwrap()).collect();
        if depth == shape.len() {
            if out.is_empty() {
                shape.push(items.len());
            } else {
                return Err("a ragged list".to_string());
            }
        } else if shape[depth] != items.len() {
            return Err("a ragged list".to_string());
        }
        for it in &items {
            flatten(it, depth + 1, shape, out)?;
        }
        return Ok(());
    }
    if obj.hasattr("tolist").unwrap_or(false) {
        let l = obj.call_method0("tolist").map_err(|e| e.to_string())?;
        return flatten(&l, depth, shape, out);
    }
    Err(format!("a {}", obj.get_type().name().map(|n| n.to_string()).unwrap_or_default()))
}

fn to_host(obj: &Bound<'_, PyAny>) -> HostValue {
    use pyo3::types::PyString;
    if obj.hasattr("spec").unwrap_or(false) && obj.hasattr("raw").unwrap_or(false) {
        if let (Ok(spec), Ok(raw)) = (obj.getattr("spec"), obj.getattr("raw")) {
            if let (Ok(elem), Ok(l)) = (spec.getattr("elem").and_then(|e| e.extract::<String>()), raw.call_method0("tolist")) {
                let (mut shape, mut leaves) = (Vec::new(), Vec::new());
                if flatten(&l, 0, &mut shape, &mut leaves).is_ok() {
                    let raw: Vec<i64> = leaves.iter().map(|x| if let Leaf::I(i) = x { *i } else { 0 }).collect();
                    return HostValue::Fixed { elem, shape, raw };
                }
            }
        }
    }
    if let Ok(s) = obj.downcast::<PyString>() {
        return HostValue::Str(s.to_string());
    }
    let (mut shape, mut leaves) = (Vec::new(), Vec::new());
    match flatten(obj, 0, &mut shape, &mut leaves) {
        Ok(()) => {
            let any_f = leaves.iter().any(|l| matches!(l, Leaf::F(_)));
            if leaves.is_empty() {
                return HostValue::Other("an empty list".to_string());
            }
            if shape.is_empty() {
                return match leaves[0] {
                    Leaf::I(i) => HostValue::Int(i),
                    Leaf::F(f) => HostValue::Float(f),
                };
            }
            if any_f {
                HostValue::Floats(shape, leaves.iter().map(|l| match l { Leaf::I(i) => *i as f64, Leaf::F(f) => *f }).collect())
            } else {
                HostValue::Ints(shape, leaves.iter().map(|l| if let Leaf::I(i) = l { *i } else { 0 }).collect())
            }
        }
        Err(what) => HostValue::Other(what),
    }
}

fn nested<'py>(py: Python<'py>, shape: &[usize], data: &mut dyn Iterator<Item = Bound<'py, PyAny>>) -> PyResult<Bound<'py, PyAny>> {
    use pyo3::types::PyList;
    if shape.is_empty() {
        return Ok(data.next().unwrap());
    }
    let items: Vec<Bound<'py, PyAny>> = (0..shape[0]).map(|_| nested(py, &shape[1..], data)).collect::<PyResult<_>>()?;
    Ok(PyList::new(py, items)?.into_any())
}

/// A host value as Python: an int, a float or a str; an array as nested lists; a fixed-point value as `(elem, nested raw list)`; None for void.
fn from_host<'py>(py: Python<'py>, v: &HostValue) -> PyResult<Bound<'py, PyAny>> {
    use pyo3::IntoPyObjectExt;
    Ok(match v {
        HostValue::Int(i) => i.into_bound_py_any(py)?,
        HostValue::Float(f) => f.into_bound_py_any(py)?,
        HostValue::Str(s) => s.into_bound_py_any(py)?,
        HostValue::Void => py.None().into_bound(py),
        HostValue::Other(s) => s.into_bound_py_any(py)?,
        HostValue::Ints(shape, data) => {
            let mut it = data.iter().map(|x| x.into_bound_py_any(py).unwrap());
            nested(py, shape, &mut it)?
        }
        HostValue::Floats(shape, data) => {
            let mut it = data.iter().map(|x| x.into_bound_py_any(py).unwrap());
            nested(py, shape, &mut it)?
        }
        HostValue::Fixed { elem, shape, raw } => {
            let mut it = raw.iter().map(|x| x.into_bound_py_any(py).unwrap());
            let l = nested(py, shape, &mut it)?;
            (elem.clone(), l).into_bound_py_any(py)?
        }
    })
}

/// A static program (strict level 2) loaded on the native interpreter: `StaticProgram(name, text, externs=None)`, then `call_entry`,
/// `set_param` and `get`.
///
/// Values cross as: an int or a float as itself; a `str` as itself; an array as nested lists of ints or floats (a NumPy array of dtype
/// int64 or float64 is accepted on the way in, and is what an extern is called with); `None` for a void result.
///
/// A fixed-point value (`fixed<W,I>`, scalar or array) is the tuple `(elem, raw)` on the way out of `get`: `elem` is the type text in its
/// canonical form (`"fixed<32,16>"`), `raw` the scaled integers as Python ints, a bare int for a scalar and one level of nested lists
/// for each dimension. Each integer is the W-bit two's complement value, so it fits an int64 (W is at most 64) and nothing is lost.
/// Python builds its `klongpy.static.fixed.FixedArray` from it as
/// `FixedArray(numpy.array(raw, dtype=numpy.int64), spec_of(elem))`, scalars included. On the way in, `set_param` and `call_entry`
/// take any object with a `spec` that has `elem` and a `raw` that has `tolist()` (a `FixedArray` is one) as a fixed-point value, and a
/// number or an array of numbers for a parameter or argument of fixed type (converted as `.fixed(type;x)` does). `call_entry(...,
/// raw=True)` answers a fixed-point result as its raw integers in nested lists, without `elem`; without `raw` it is the float value.
#[pyclass(unsendable)]
struct StaticProgram {
    inner: Loaded,
    /// the interpreter the program shares, when it was made on one (`StaticProgram(..., klong=k)`)
    klong: Option<Py<Klong>>,
}

impl StaticProgram {
    /// Run `f` with the shared interpreter set up as for any call into it (its host table, converters and console).
    fn within<R>(&mut self, py: Python<'_>, f: impl FnOnce(&mut Loaded) -> R) -> R {
        match self.klong.as_ref().map(|k| k.clone_ref(py)) {
            Some(k) => {
                let k = k.bind(py).borrow();
                let inner = &mut self.inner;
                k.with_ip(py, |_| f(inner))
            }
            None => f(&mut self.inner),
        }
    }
}

fn host_err(e: Error) -> PyErr {
    k2py(e)
}

fn params_rows<'py>(py: Python<'py>, rt: &Runtime<HostPlatform>) -> PyResult<Vec<(String, String, Bound<'py, PyAny>, Option<Bound<'py, PyAny>>)>> {
    klong_static::host::params_of(rt)
        .into_iter()
        .map(|p| Ok((p.name, p.ty.to_string(), from_host(py, &p.default)?, p.value.map(|v| from_host(py, &v)).transpose()?)))
        .collect()
}

#[pymethods]
impl StaticProgram {
    /// Run a static program (strict level 2) with Python callables bound as its externs; a program that uses fixed-point numbers runs its entries on the typed-IR evaluator.
    ///
    /// With `klong` (a `Klong`) the program is not run: it is the program that interpreter has already run (strict level 2, its `.entry`, `.extern`
    /// and `.param` declared), and the object works on that interpreter, so entries, parameters and globals are the session's. `externs` then
    /// only serves the typed-IR evaluator of a fixed-point program (the interpreter has its stand-ins already).
    #[new]
    #[pyo3(signature = (name, text, externs=None, klong=None))]
    fn new(py: Python<'_>, name: String, text: String, externs: Option<Bound<'_, PyAny>>, klong: Option<Py<Klong>>) -> PyResult<StaticProgram> {
        use pyo3::types::PyDict;
        let mut table: HashMap<String, HostFn> = HashMap::new();
        if let Some(d) = externs {
            let d = d.downcast::<PyDict>()?;
            for (k, f) in d.iter() {
                let name: String = k.extract()?;
                let f: Py<PyAny> = f.unbind();
                let cb: HostFn = Rc::new(move |_ip: &mut klong_core::Interp, a: &[klong_core::Value]| {
                    Python::with_gil(|py| {
                        let args: PyResult<Vec<Bound<'_, PyAny>>> = a
                            .iter()
                            .map(|v| {
                                let h = HostValue::from_value(v);
                                match &h {
                                    HostValue::Ints(..) => py.import("numpy")?.call_method1("array", (from_host(py, &h)?, "int64")),
                                    HostValue::Floats(..) => py.import("numpy")?.call_method1("array", (from_host(py, &h)?, "float64")),
                                    _ => from_host(py, &h),
                                }
                            })
                            .collect();
                        let r = f.bind(py).call1(pyo3::types::PyTuple::new(py, args.map_err(|e| Error::system(e.to_string()))?).map_err(|e| Error::system(e.to_string()))?);
                        match r {
                            Ok(v) => Ok(to_host(&v).to_value()),
                            Err(e) => Err(Error::system(e.to_string())),
                        }
                    })
                });
                table.insert(name, cb);
            }
        }
        if let Some(k) = klong {
            let ptr = k.bind(py).borrow().rt.get();
            // the interpreter lives as long as the Py<Klong> this object keeps
            let inner = unsafe { Loaded::attach(ptr, &name, &text, table) };
            return Ok(StaticProgram { inner, klong: Some(k) });
        }
        let inner = klong_static::host::load_program(&name, Some(&text), table);
        Ok(StaticProgram { inner: inner.map_err(host_err)?, klong: None })
    }

    #[pyo3(signature = (name, args, signature=None, raw=false))]
    fn call_entry<'py>(&mut self, py: Python<'py>, name: String, args: Vec<Bound<'py, PyAny>>, signature: Option<String>, raw: bool) -> PyResult<Bound<'py, PyAny>> {
        let hv: Vec<HostValue> = args.iter().map(to_host).collect();
        let r = self.within(py, |p| klong_static::host::call_entry(p, &name, &hv, signature.as_deref(), raw)).map_err(host_err)?;
        from_host(py, &r)
    }

    fn set_param<'py>(&mut self, py: Python<'py>, name: String, value: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let v = to_host(&value);
        let r = self.within(py, |p| klong_static::host::set_param(p, &name, &v)).map_err(host_err)?;
        from_host(py, &r)
    }

    /// The value of a global of the program, as `call_entry` returns values.
    fn get<'py>(&self, py: Python<'py>, name: String) -> PyResult<Option<Bound<'py, PyAny>>> {
        self.inner.get(&name).map(|v| from_host(py, &v)).transpose()
    }

    /// The parameters the program declared with `.param`, in order: `[(name, type, default, value)]`. `type` is the canonical text
    /// (`"float"`, `"fixed<32,16>"`); `default` and `value` are host values as `get` gives them (`value` is None when the name is not defined).
    fn params<'py>(&self, py: Python<'py>) -> PyResult<Vec<(String, String, Bound<'py, PyAny>, Option<Bound<'py, PyAny>>)>> {
        params_rows(py, &self.inner.rt)
    }
}


/// `kgpy fpga source` of a program text: `(ok, files, entries, diags, error)`. `files` is `[(name, text)]` in sorted order, with `manifest.json`, the C, the
/// HLS C++, the test benches, the tcl and `klong_rt.h`; `entries` is `[(name, top, signature, in_bytes, out_bytes)]`. A program outside the profile, an
/// entry that cannot be a kernel (rule `hls`) or a public C name an extern would take is `diags`, as the other functions give them; `error` is the text
/// of a refusal that is not a diagnostic (a bad option, or a kernel refusal that lies in no source).
#[pyfunction]
#[pyo3(signature = (name, text, int64=false, prefix=None, clock_mhz=300, wrapped=false, lanes=None, reassociate=false, lib_dir=None))]
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
fn fpga_source(py: Python<'_>, name: String, text: String, int64: bool, prefix: Option<String>, clock_mhz: u32, wrapped: bool, lanes: Option<usize>, reassociate: bool, lib_dir: Option<String>) -> PyResult<(bool, Vec<(String, String)>, Vec<(String, String, String, usize, usize)>, Vec<StaticDiag>, Option<String>)> {
    use klong_static::compile::EmitFailure;
    use klong_static::emit_hls::{HlsError, HlsFailure};
    let libs: Vec<String> = lib_dir.into_iter().collect();
    on_big_stack(py, move || match klong_static::emit_hls::emit_hls_program_in(&name, None, Some(&text), if int64 { 64 } else { 32 }, prefix.as_deref(), clock_mhz, wrapped, lanes, reassociate, &libs) {
        Ok(r) => (
            true,
            r.out.files.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
            r.out.entries.iter().map(|e| (e.name.clone(), e.top.clone(), e.signature.clone(), e.in_bytes, e.out_bytes)).collect(),
            Vec::new(),
            None,
        ),
        Err(f) => {
            let (diags, error) = match f {
                HlsFailure::Emit(EmitFailure::Compile(c)) => compile_failure(c),
                HlsFailure::Emit(EmitFailure::Emit(klong_static::emit_c::EmitError::Static(message, hint, span, src), r)) => {
                    let d = klong_static::check::Diag { rule: "extern", span, src, message, hint: Some(hint), function: None };
                    (vec![static_diag(&d, &r)], None)
                }
                HlsFailure::Hls(HlsError::Static { message, hint, span: Some(span), src }, r) => {
                    let d = klong_static::check::Diag { rule: "hls", span, src, message, hint, function: None };
                    (vec![static_diag(&d, &r)], None)
                }
                other => (Vec::new(), Some(other.render(&name).0.join("\n"))),
            };
            (false, Vec::new(), Vec::new(), diags, error)
        }
    })
}

#[pymodule]
fn klong_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Klong>()?;
    m.add_class::<KFn>()?;
    m.add_class::<NativeVec>()?;
    m.add_class::<NativeObject>()?;
    m.add_class::<NativeDict>()?;
    m.add_class::<StaticProgram>()?;
    m.add("NativeError", m.py().get_type::<NativeError>())?;
    m.add_function(wrap_pyfunction!(zero_copy, m)?)?;
    m.add_function(wrap_pyfunction!(ast_text, m)?)?;
    m.add_function(wrap_pyfunction!(repl_check, m)?)?;
    m.add_function(wrap_pyfunction!(repl_complete, m)?)?;
    m.add_function(wrap_pyfunction!(function_form, m)?)?;
    m.add_function(wrap_pyfunction!(function_wire_text, m)?)?;
    m.add_function(wrap_pyfunction!(function_display_arity, m)?)?;
    m.add_function(wrap_pyfunction!(lint_files, m)?)?;
    m.add_function(wrap_pyfunction!(lint_source, m)?)?;
    m.add_function(wrap_pyfunction!(check_files, m)?)?;
    m.add_function(wrap_pyfunction!(check_source, m)?)?;
    m.add_function(wrap_pyfunction!(check_report, m)?)?;
    m.add_function(wrap_pyfunction!(ir_json_source, m)?)?;
    m.add_function(wrap_pyfunction!(dump_ir_source, m)?)?;
    m.add_function(wrap_pyfunction!(dump_ir_files, m)?)?;
    m.add_function(wrap_pyfunction!(compile_c_source, m)?)?;
    m.add_function(wrap_pyfunction!(compile_c_files, m)?)?;
    m.add_function(wrap_pyfunction!(runtime_header, m)?)?;
    m.add_function(wrap_pyfunction!(streamable_source, m)?)?;
    m.add_function(wrap_pyfunction!(fpga_source, m)?)?;
    m.add("SignatureError", m.py().get_type::<SignatureError>())?;
    m.add_function(wrap_pyfunction!(lint_rules, m)?)?;
    m.add_function(wrap_pyfunction!(static_tables, m)?)?;
    m.add_function(wrap_pyfunction!(parse_type, m)?)?;
    m.add_function(wrap_pyfunction!(parse_signature, m)?)?;
    m.add_function(wrap_pyfunction!(format_type, m)?)?;
    m.add_function(wrap_pyfunction!(format_signature, m)?)?;
    m.add_function(wrap_pyfunction!(check_extern_name, m)?)?;
    m.add_function(wrap_pyfunction!(check_entry_name, m)?)?;
    m.add_function(wrap_pyfunction!(c_identifier, m)?)?;
    m.add_function(wrap_pyfunction!(fixed_spec, m)?)?;
    m.add_function(wrap_pyfunction!(validate_global, m)?)?;
    m.add_function(wrap_pyfunction!(profile_doc_block, m)?)?;
    m.add_function(wrap_pyfunction!(profile_doc_page, m)?)?;
    Ok(())
}
