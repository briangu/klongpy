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
    dyad: HashMap<String, Py<PyAny>>,
    monad: HashMap<String, Py<PyAny>>,
    reduce: HashMap<String, Py<PyAny>>,
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
}

const HOST_PY: &str = r#"
import numpy as np
i64 = np.int64
def mk_obj(items):
    a = np.empty(len(items), dtype=object)
    for i, x in enumerate(items):
        a[i] = x
    return a
def _cmp(f):
    return lambda a, b: f(a, b).astype(i64)
dyad = {
    '+': np.add, '-': np.subtract, '*': np.multiply, '%': np.true_divide, '^': np.power,
    '&': np.minimum, '|': np.maximum, ':%': np.floor_divide, '!': np.mod,
    '<': _cmp(np.less), '>': _cmp(np.greater), '=': _cmp(np.equal),
    '@': lambda a, i: a[i],
    ',': lambda a, b: np.concatenate((np.atleast_1d(a), np.atleast_1d(b))),
    '#': lambda n, a: (a[:n] if n >= 0 else a[n:]) if abs(n) <= len(a) else np.resize(a, abs(n)),
    '_': lambda n, a: a[n:] if n >= 0 else a[:n],
}
monad = {
    '-': np.negative,
    '<': lambda a: np.argsort(a, kind='stable'),
    '>': lambda a: np.argsort(-a, kind='stable'),
    '|': lambda a: a[::-1],
    '#': lambda a: len(a),
    '*': lambda a: a[0],
    '_': lambda a: np.floor(a).astype(i64),
    '%': lambda a: 1.0 / a,
}
reduce = {'+': np.add.reduce, '*': np.multiply.reduce, '&': np.minimum.reduce, '|': np.maximum.reduce}
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
    let tab = |name: &str| -> PyResult<HashMap<String, Py<PyAny>>> {
        let d = m.getattr(name)?;
        let d = d.downcast::<PyDict>()?;
        d.iter().map(|(k, v)| Ok((k.extract::<String>()?, v.unbind()))).collect()
    };
    let ty = |k: &str| -> PyResult<Py<PyAny>> { Ok(types.get_item(k)?.ok_or_else(|| PyTypeError::new_err(format!("types[{k:?}] missing")))?.unbind()) };
    let h: &'static Host = Box::leak(Box::new(Host {
        ndarray: np.getattr("ndarray")?.unbind(),
        asarray: np.getattr("asarray")?.unbind(),
        array: np.getattr("array")?.unbind(),
        int64: np.getattr("int64")?.unbind(),
        float64: np.getattr("float64")?.unbind(),
        ascontig: np.getattr("ascontiguousarray")?.unbind(),
        mk_obj: m.getattr("mk_obj")?.unbind(),
        dyad: tab("dyad")?,
        monad: tab("monad")?,
        reduce: tab("reduce")?,
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

/// A native error as a Python exception: args = (kind, message, start, end, source name, source text, hint).
fn k2py(e: Error) -> PyErr {
    if let Some(p) = PENDING.with(|p| p.borrow_mut().take()) {
        return p;
    }
    let (name, text) = match &e.source {
        Some(s) => (s.name.clone(), s.text.to_string()),
        None => (String::new(), String::new()),
    };
    let (a, b) = e.span.map(|(a, b)| (a as i64, b as i64)).unwrap_or((-1, -1));
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
}

/// An opaque host array: the primitives on it are forwarded to Python.
struct HostArr {
    obj: Py<PyAny>,
}

/// Any other Python object (a datetime, a DataFrame, an instance) as an opaque value that Klong can hold and pass back.
struct PyObj {
    obj: Py<PyAny>,
}

impl Ext for PyObj {
    fn type_name(&self) -> &str {
        "python object"
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

fn is_host(v: &Value) -> bool {
    matches!(v, Value::Ext(e) if e.0.as_any().is::<HostArr>())
}

/// A native value or a host array that NumPy can combine with a host array without changing what Klong means.
fn numeric_operand(v: &Value) -> bool {
    match v {
        Value::Int(_) | Value::Flt(_) => true,
        Value::Ints(a) => a.rank() == 1,
        Value::Flts(a) => a.rank() == 1,
        _ => is_host(v),
    }
}

fn unhost(v: &Value) -> klong_core::error::Result<Value> {
    if let Value::Ext(e) = v {
        if let Some(r) = e.0.host_native() {
            return r;
        }
    }
    Ok(v.clone())
}

impl Ext for HostArr {
    fn type_name(&self) -> &str {
        "host array"
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn host_native(&self) -> Option<klong_core::error::Result<Value>> {
        Some(
            Python::with_gil(|py| -> PyResult<Value> {
                let h = host(py)?;
                ndarray_to_native(py, &h, self.obj.bind(py))
            })
            .map_err(py2k),
        )
    }
    fn host_monad(&self, ip: &mut Interp, glyph: &str, x: &Value) -> Option<klong_core::error::Result<Value>> {
        let forwarded = Python::with_gil(|py| -> PyResult<Option<Value>> {
            let h = host(py)?;
            let Some(f) = h.monad.get(glyph) else { return Ok(None) };
            let a = operand(py, ip, x)?;
            let r = f.bind(py).call1((a,))?;
            from_host_result(py, ip, &h, &r).map(Some)
        });
        match forwarded {
            Ok(Some(v)) => Some(Ok(v)),
            Ok(None) => {
                let nx = match unhost(x) {
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
        // forward when the operator is in the table and the other operand is numeric (a scalar, a vector or another host array)
        let ok = match glyph {
            "@" => is_host(x) && numeric_operand(y),
            "#" => matches!(x, Value::Int(_)) && is_host(y),
            "_" => matches!(x, Value::Int(_)) && is_host(y),
            _ => numeric_operand(x) && numeric_operand(y),
        };
        if ok {
            let forwarded = Python::with_gil(|py| -> PyResult<Option<Value>> {
                let h = host(py)?;
                let Some(f) = h.dyad.get(glyph) else { return Ok(None) };
                let a = operand(py, ip, x)?;
                let b = operand(py, ip, y)?;
                let r = f.bind(py).call1((a, b))?;
                from_host_result(py, ip, &h, &r).map(Some)
            });
            match forwarded {
                Ok(Some(v)) => return Some(Ok(v)),
                Ok(None) => {}
                Err(e) => return Some(Err(py2k(e))),
            }
        }
        let nx = match unhost(x) {
            Ok(v) => v,
            Err(e) => return Some(Err(e)),
        };
        let ny = match unhost(y) {
            Ok(v) => v,
            Err(e) => return Some(Err(e)),
        };
        let op = ip.syms.lookup(glyph)?;
        Some(ip.dyad(op, nx, ny))
    }
    fn host_reduce(&self, ip: &mut Interp, glyph: &str, x: &Value) -> Option<klong_core::error::Result<Value>> {
        let r = Python::with_gil(|py| -> PyResult<Option<Value>> {
            let h = host(py)?;
            let Some(f) = h.reduce.get(glyph) else { return Ok(None) };
            let a = operand(py, ip, x)?;
            if a.len()? == 0 {
                return Ok(None);
            }
            let r = f.bind(py).call1((a,))?;
            from_host_result(py, ip, &h, &r).map(Some)
        });
        match r {
            Ok(Some(v)) => Some(Ok(v)),
            Ok(None) => None,
            Err(e) => Some(Err(py2k(e))),
        }
    }
}

fn host_value(obj: Py<PyAny>) -> Value {
    Value::Ext(Rc::new(klong_core::value::ExtBox(Box::new(HostArr { obj }))))
}

/// An operand of a forwarded primitive as a Python object: the host array itself, or a native value converted.
fn operand<'py>(py: Python<'py>, ip: &Interp, v: &Value) -> PyResult<Bound<'py, PyAny>> {
    if let Value::Ext(e) = v {
        if let Some(h) = e.0.as_any().downcast_ref::<HostArr>() {
            return Ok(h.obj.bind(py).clone());
        }
    }
    to_py_with(py, ip, v, None, true)
}

/// The result of a forwarded primitive: 1-d numeric arrays stay host arrays, scalars become native scalars.
fn from_host_result(py: Python<'_>, ip: &mut Interp, h: &Host, r: &Bound<'_, PyAny>) -> PyResult<Value> {
    if r.is_instance(h.ndarray.bind(py))? {
        let nd: usize = r.getattr("ndim")?.extract()?;
        if nd == 0 {
            return from_py(py, ip, &r.call_method0("item")?);
        }
        if nd == 1 && matches!(r.getattr("dtype")?.getattr("kind")?.extract::<String>()?.as_str(), "i" | "f") {
            return Ok(host_value(r.clone().unbind()));
        }
        return ndarray_to_native(py, h, r);
    }
    from_py(py, ip, r)
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
            let out = PyDict::new(py);
            let d = d.borrow();
            for (k, val) in d.keys().iter().zip(d.vals().iter()) {
                let key = match k {
                    Value::List(_) | Value::Ints(_) | Value::Flts(_) => {
                        let l = to_py(py, ip, k, owner)?;
                        py.import("builtins")?.getattr("tuple")?.call1((l.call_method0("tolist")?,))?
                    }
                    _ => to_py(py, ip, k, owner)?,
                };
                out.set_item(key, to_py_with(py, ip, val, owner, view)?)?;
            }
            out.into_any()
        }
        Value::Ext(e) => match (e.0.as_any().downcast_ref::<HostArr>(), e.0.as_any().downcast_ref::<PyObj>()) {
            (Some(h), _) => h.obj.bind(py).clone(),
            (_, Some(p)) => p.obj.bind(py).clone(),
            _ => Bound::new(py, NativeObject { v: v.clone() })?.into_any(),
        },
        Value::Fn(_) | Value::Prim(_) => match owner {
            Some(o) => Bound::new(py, KFn { f: v.clone(), owner: o.clone_ref(py) })?.into_any(),
            None => return Err(PyTypeError::new_err("a Klong function needs its interpreter to be called")),
        },
    })
}

/// A Python value as a Klong value. NumPy arrays are COPIED into native vectors (dtype int64/float64; others converted).
fn from_py(py: Python<'_>, ip: &mut Interp, o: &Bound<'_, PyAny>) -> PyResult<Value> {
    if o.is_none() {
        return Ok(Value::Undefined);
    }
    if o.is_exact_instance_of::<PyString>() {
        return Value::text(o.downcast::<PyString>()?.to_str()?).map_err(k2py);
    }
    if let Ok(b) = o.downcast::<PyBool>() {
        return Ok(Value::Int(b.is_true() as Int));
    }
    if o.is_instance_of::<PyInt>() {
        return o.extract::<Int>().map(Value::Int).map_err(|_| PyOverflowError::new_err("integer does not fit in 64 bits (native has no big integers yet)"));
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
    if let Ok(kf) = o.downcast::<KFn>() {
        return Ok(kf.borrow().f.clone());
    }
    if let Ok(no) = o.downcast::<NativeObject>() {
        return Ok(no.borrow().v.clone());
    }
    if o.is_instance(h.ndarray.bind(py))? {
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
    if let Ok(d) = o.downcast::<PyDict>() {
        let mut out = Dict::default();
        for (k, v) in d.iter() {
            out.set(from_py(py, ip, &k)?, from_py(py, ip, &v)?).map_err(k2py)?;
        }
        return Ok(Value::dict(out));
    }
    if o.hasattr("dtype")? && o.hasattr("item")? {
        // a NumPy scalar
        return from_py(py, ip, &o.call_method0("item")?);
    }
    if o.hasattr("_provide_klong")? && o.hasattr("fn")? {
        // a KlongPy KGLambda: its Python function, with the arity it declares
        let wild: bool = o.getattr("_wildcard")?.extract()?;
        let n: usize = o.getattr("args")?.len().unwrap_or(0);
        return Ok(callable_value(ip, "python", o.getattr("fn")?.unbind(), if wild { 255 } else { n as u8 }));
    }
    if o.is_callable() {
        return Ok(callable_value(ip, "python", o.clone().unbind(), 255));
    }
    Ok(Value::Ext(Rc::new(klong_core::value::ExtBox(Box::new(PyObj { obj: o.clone().unbind() })))))
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
    let a = h.ascontig.bind(py).call1((o, dt))?;
    let ai = a.getattr("__array_interface__")?;
    let ptr: usize = ai.get_item("data")?.get_item(0)?.extract()?;
    let n: usize = shape.iter().product();
    let dims: Vec<u32> = shape.iter().map(|&d| d as u32).collect();
    // SAFETY: `a` is a live contiguous array of n elements of the dtype asked for; it is read before `a` is dropped.
    unsafe {
        if floats {
            let s = std::slice::from_raw_parts(ptr as *const f64, n);
            Ok(Value::Flts(Rc::new(Arr::shaped(s.to_vec(), &dims))))
        } else {
            let s = std::slice::from_raw_parts(ptr as *const i64, n);
            Ok(Value::Ints(Rc::new(Arr::shaped(s.to_vec(), &dims))))
        }
    }
}

/// A Python callable as a Klong system function.
fn callable_value(ip: &mut Interp, name: &str, f: Py<PyAny>, arity: u8) -> Value {
    let sym = ip.syms.intern(name);
    Value::Fn(Rc::new(Func::Native {
        name: sym,
        arity,
        f: Rc::new(move |ip: &mut Interp, args: &[Value]| -> klong_core::error::Result<Value> {
            Python::with_gil(|py| -> PyResult<Value> {
                let mut pa = Vec::with_capacity(args.len());
                for a in args {
                    pa.push(to_py_owner(py, ip, a)?);
                }
                let r = f.bind(py).call1(PyTuple::new(py, pa)?)?;
                from_py(py, ip, &r)
            })
            .map_err(py2k)
        }),
    }))
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
    me: RefCell<Option<Py<Klong>>>,
}

unsafe impl Send for Klong {}
unsafe impl Sync for Klong {}

/// The native stack the evaluator may use on this thread: Python's main thread has the OS default (8 MiB); other threads
/// have what `threading.stack_size()` gave them (512 KiB on macOS by default), so they get a small budget.
fn stack_budget() -> usize {
    #[cfg(unix)]
    unsafe {
        extern "C" {
            fn pthread_main_np() -> i32;
        }
        #[cfg(target_os = "macos")]
        if pthread_main_np() == 0 {
            return 256 * 1024;
        }
    }
    4 * 1024 * 1024
}

impl Klong {
    fn with_ip<R>(&self, py: Python<'_>, f: impl FnOnce(&mut Interp) -> R) -> R {
        let cur = self.cur.get();
        if !cur.is_null() {
            // re-entered from a callback: the outer `&mut Interp` is lent to the callback and not used until it returns
            return f(unsafe { &mut *cur });
        }
        let ip: *mut Interp = unsafe { &mut (*self.rt.get()).interp };
        unsafe { (*ip).cfg.stack_limit = Some(stack_budget()) };
        self.cur.set(ip);
        let prev = CURRENT.with(|c| c.replace(ip));
        let prev_owner = OWNER.with(|o| o.replace(self.me.borrow().as_ref().map(|m| m.clone_ref(py))));
        let r = f(unsafe { &mut *ip });
        OWNER.with(|o| *o.borrow_mut() = prev_owner);
        CURRENT.with(|c| c.set(prev));
        self.cur.set(std::ptr::null_mut());
        self.drain(py);
        r
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

    fn owner(&self, py: Python<'_>) -> Py<Klong> {
        self.me.borrow().as_ref().map(|m| m.clone_ref(py)).expect("owner set in new()")
    }

    fn apply(&self, py: Python<'_>, f: &Value, args: &Bound<'_, PyTuple>) -> PyResult<PyObject> {
        let owner = self.owner(py);
        self.with_ip(py, |ip| -> PyResult<PyObject> {
            let vals = args.iter().map(|a| from_py(py, ip, &a)).collect::<PyResult<Vec<_>>>()?;
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
        let cfg = Config { strict, stack_limit: Some(4 * 1024 * 1024), ..Config::default() };
        let mut rt = Runtime::new(HostPlatform::new(Console::Buffer(Vec::new())), cfg, None);
        let path = std::env::var("KLONGPATH").ok().filter(|p| !p.is_empty()).unwrap_or_else(|| String::from(".:lib"));
        rt.set_load_path(path.split(':').filter(|d| !d.is_empty()).map(String::from).collect());
        let me = Py::new(py, Klong { rt: UnsafeCell::new(rt), cur: Cell::new(std::ptr::null_mut()), me: RefCell::new(None) })?;
        *me.borrow(py).me.borrow_mut() = Some(me.clone_ref(py));
        Ok(me)
    }

    /// Evaluate and convert the result.
    fn eval(&self, py: Python<'_>, src: &str) -> PyResult<PyObject> {
        let owner = self.owner(py);
        self.with_ip(py, |ip| -> PyResult<PyObject> {
            match ip.run(src, "<string>").map_err(k2py)? {
                Some(v) => Ok(to_py(py, ip, &v, Some(&owner))?.unbind()),
                None => Ok(py.None()),
            }
        })
    }

    /// KlongPy's `exec`: the value of every statement, as a list.
    fn exec(&self, py: Python<'_>, src: &str) -> PyResult<PyObject> {
        let owner = self.owner(py);
        self.with_ip(py, |ip| -> PyResult<PyObject> {
            ip.collect = Some(Vec::new());
            let r = ip.run(src, "<string>");
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
    fn run(&self, py: Python<'_>, src: &str) -> PyResult<()> {
        self.with_ip(py, |ip| ip.run(src, "<string>").map(|_| ())).map_err(k2py)
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
                let ndim: usize = value.getattr("ndim")?.extract()?;
                let kind: String = value.getattr("dtype")?.getattr("kind")?.extract()?;
                let n: usize = value.getattr("size")?.extract()?;
                if ndim == 1 && matches!(kind.as_str(), "i" | "f") && n >= h.threshold.load(std::sync::atomic::Ordering::Relaxed) {
                    host_value(value.clone().unbind())
                } else {
                    from_py(py, ip, value)?
                }
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
    fn register(&self, py: Python<'_>, name: &str, f: Py<PyAny>, arity: u8) -> PyResult<()> {
        self.with_ip(py, |ip| {
            let v = callable_value(ip, name, f, arity);
            ip.define_global(name, v)
        })
        .map_err(k2py)
    }

    /// Always keep this array as a host array, whatever its size.
    fn set_host(&self, py: Python<'_>, name: &str, arr: Bound<'_, PyAny>) -> PyResult<()> {
        self.with_ip(py, |ip| ip.define_global(name, host_value(arr.unbind()))).map_err(k2py)
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

    /// Native stack used at most, bytes.
    fn max_stack(&self, py: Python<'_>) -> usize {
        self.with_ip(py, |ip| ip.max_stack_seen)
    }
}

#[pyfunction]
fn zero_copy(on: bool) {
    ZERO_COPY.store(on, std::sync::atomic::Ordering::Relaxed);
}

#[pymodule]
fn klong_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Klong>()?;
    m.add_class::<KFn>()?;
    m.add_class::<NativeVec>()?;
    m.add_class::<NativeObject>()?;
    m.add("NativeError", m.py().get_type::<NativeError>())?;
    m.add_function(wrap_pyfunction!(zero_copy, m)?)?;
    Ok(())
}
