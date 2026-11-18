//! An evaluator of the typed IR (`klongpy/static/eval_ir.py`): the interpreter stand-in for fixed-point programs.
//!
//! The IR that lowering builds (`infer.rs`) is evaluated on n-dimensional arrays: `i64` or `f64` data with a shape; fixed-point values
//! are the raw scaled `i64` (`fixed.rs`) and every integer wraps at the width of `kg_int`, which is what compiled code does. The
//! semantics of each node are those of `klong_rt.h` (and of the NumPy code of the Python evaluator, which this mirrors); where
//! compiled code traps (an index out of range, an integer or fixed-point division by zero) evaluation stops with a [`Trap`].
//!
//! ```ignore
//! let mut ev = IrEvaluator::new(&program);
//! let r = ev.run(&program.entries[0], &[Arr::flt(0.5)])?;
//! ev.state            // {name: value} the state globals (and parameters)
//! ```
use crate::fixed as fx;
use crate::ir::{Entry, FnSpec, Node, Program, K, V};
use crate::sig::{is_fixed, Ty};
use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt;
use std::ops::Deref;

pub const TRAP_BOUNDS: i32 = 1;
pub const TRAP_DIV0: i32 = 2;
/// Not a trap of compiled code: the evaluator cannot do what the node asks (Python raises NotImplementedError, ValueError, ...).
pub const TRAP_UNSUPPORTED: i32 = 0;
/// An extern the host could not call (no callback, or the callback failed).
pub const TRAP_EXTERN: i32 = 3;

/// What compiled code reports with `kg_trap`: code 1 an index out of range, 2 a division by zero.
#[derive(Clone, Debug, PartialEq)]
pub struct Trap {
    pub code: i32,
    pub message: String,
}

impl Trap {
    pub fn new(code: i32, message: impl Into<String>) -> Trap {
        Trap { code, message: message.into() }
    }
}

impl fmt::Display for Trap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for Trap {}

type R<T> = Result<T, Trap>;

fn unsupported<T>(m: impl Into<String>) -> R<T> {
    Err(Trap::new(TRAP_UNSUPPORTED, m))
}

// ---------------------------------------------------------------------------
// Values
// ---------------------------------------------------------------------------

/// The data of an array: int64 (also the raw values of fixed point) or float64.
#[derive(Clone, Debug, PartialEq)]
pub enum Data {
    I(Vec<i64>),
    F(Vec<f64>),
}

/// An n-dimensional array in row-major order; a scalar has an empty shape and one element.
#[derive(Clone, Debug, PartialEq)]
pub struct Arr {
    pub shape: Vec<usize>,
    pub data: Data,
}

impl Arr {
    pub fn int(v: i64) -> Arr {
        Arr { shape: Vec::new(), data: Data::I(vec![v]) }
    }
    pub fn flt(v: f64) -> Arr {
        Arr { shape: Vec::new(), data: Data::F(vec![v]) }
    }
    pub fn ints(shape: Vec<usize>, d: Vec<i64>) -> Arr {
        Arr { shape, data: Data::I(d) }
    }
    pub fn flts(shape: Vec<usize>, d: Vec<f64>) -> Arr {
        Arr { shape, data: Data::F(d) }
    }
    /// A zero of the type (`zero_of`).
    pub fn zero(ty: &Ty) -> Arr {
        let n: usize = ty.shape.iter().product();
        if ty.elem == "float" {
            Arr::flts(ty.shape.clone(), vec![0.0; n])
        } else {
            Arr::ints(ty.shape.clone(), vec![0; n])
        }
    }
    pub fn is_float(&self) -> bool {
        matches!(self.data, Data::F(_))
    }
    pub fn size(&self) -> usize {
        match &self.data {
            Data::I(d) => d.len(),
            Data::F(d) => d.len(),
        }
    }
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }
    /// The length of the first axis.
    pub fn len(&self) -> usize {
        self.shape.first().copied().unwrap_or(0)
    }
    pub fn is_empty(&self) -> bool {
        self.size() == 0
    }
    /// The elements as int64 (floats truncate, as `astype(int64)`).
    pub fn i64s(&self) -> Cow<'_, [i64]> {
        match &self.data {
            Data::I(d) => Cow::Borrowed(d),
            Data::F(d) => Cow::Owned(d.iter().map(|x| *x as i64).collect()),
        }
    }
    /// The elements as float64.
    pub fn f64s(&self) -> Cow<'_, [f64]> {
        match &self.data {
            Data::F(d) => Cow::Borrowed(d),
            Data::I(d) => Cow::Owned(d.iter().map(|x| *x as f64).collect()),
        }
    }
    /// The first element as an int (`int(a)` of a one-element array).
    pub fn item_i(&self) -> R<i64> {
        if self.size() != 1 {
            return unsupported("a scalar was expected");
        }
        Ok(self.i64s()[0])
    }
    /// The array converted to the data kind of the element type: float64 for "float", int64 otherwise (`np.asarray(v, dtype=_dtype(elem))`).
    pub fn as_elem(self, elem: &str) -> Arr {
        if elem == "float" {
            if self.is_float() {
                self
            } else {
                let d = self.f64s().into_owned();
                Arr::flts(self.shape, d)
            }
        } else if self.is_float() {
            let d = self.i64s().into_owned();
            Arr::ints(self.shape, d)
        } else {
            self
        }
    }
    fn rest(&self) -> usize {
        self.shape.iter().skip(1).product()
    }
    /// `a[j]`: the j-th sub-array.
    pub fn row(&self, j: usize) -> Arr {
        let rs = self.rest();
        let shape = self.shape[1..].to_vec();
        match &self.data {
            Data::I(d) => Arr::ints(shape, d[j * rs..(j + 1) * rs].to_vec()),
            Data::F(d) => Arr::flts(shape, d[j * rs..(j + 1) * rs].to_vec()),
        }
    }
    /// The sub-arrays at `idx` (all in range), as one array.
    fn rows_at(&self, idx: &[usize]) -> Arr {
        let rs = self.rest();
        let mut shape = vec![idx.len()];
        shape.extend_from_slice(&self.shape[1..]);
        match &self.data {
            Data::I(d) => Arr::ints(shape, idx.iter().flat_map(|&j| d[j * rs..(j + 1) * rs].iter().copied()).collect()),
            Data::F(d) => Arr::flts(shape, idx.iter().flat_map(|&j| d[j * rs..(j + 1) * rs].iter().copied()).collect()),
        }
    }
    /// `np.stack(rows)`: a new first axis; ints promote to float when mixed.
    pub fn stack(rows: &[Arr]) -> R<Arr> {
        let Some(first) = rows.first() else { return unsupported("need at least one array to stack") };
        if rows.iter().any(|r| r.shape != first.shape) {
            return unsupported("all input arrays must have the same shape");
        }
        let mut shape = vec![rows.len()];
        shape.extend_from_slice(&first.shape);
        if rows.iter().any(|r| r.is_float()) {
            Ok(Arr::flts(shape, rows.iter().flat_map(|r| r.f64s().into_owned()).collect()))
        } else {
            Ok(Arr::ints(shape, rows.iter().flat_map(|r| r.i64s().into_owned()).collect()))
        }
    }
    fn truthy(&self) -> R<bool> {
        if self.size() != 1 {
            return unsupported("the truth value of an array with more than one element is ambiguous");
        }
        Ok(match &self.data {
            Data::I(d) => d[0] != 0,
            Data::F(d) => d[0] != 0.0,
        })
    }
}

fn truth(b: bool) -> i64 {
    b as i64
}

// ---------------------------------------------------------------------------
// Element-wise helpers
// ---------------------------------------------------------------------------

/// Klong's atomic extension: the shorter shape is a prefix of the longer, so it gains trailing axes (NumPy would add leading ones).
fn align_shapes(a: &[usize], b: &[usize]) -> (Vec<usize>, Vec<usize>) {
    let (mut a, mut b) = (a.to_vec(), b.to_vec());
    while a.len() < b.len() {
        a.push(1);
    }
    while b.len() < a.len() {
        b.push(1);
    }
    (a, b)
}

/// The NumPy broadcast of two equal-rank shapes.
fn broadcast(a: &[usize], b: &[usize]) -> R<Vec<usize>> {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| if x == y || y == 1 { Ok(x) } else if x == 1 { Ok(y) } else { unsupported(format!("operands could not be broadcast together with shapes {:?} {:?}", a, b)) })
        .collect()
}

/// For each element of the broadcast result, the position of its operand elements.
fn positions(shape: &[usize], out: &[usize]) -> Vec<usize> {
    let n: usize = out.iter().product();
    if shape == out {
        return (0..n).collect();
    }
    let nd = out.len();
    let mut stride = vec![0usize; nd];
    let mut s = 1;
    for k in (0..nd).rev() {
        stride[k] = if shape[k] == 1 { 0 } else { s };
        s *= shape[k];
    }
    let mut idx = vec![0usize; nd];
    let mut pos = Vec::with_capacity(n);
    for _ in 0..n {
        pos.push(idx.iter().zip(&stride).map(|(i, s)| i * s).sum());
        for k in (0..nd).rev() {
            idx[k] += 1;
            if idx[k] < out[k] {
                break;
            }
            idx[k] = 0;
        }
    }
    pos
}

fn zip2<T: Copy, U>(sa: &[usize], da: &[T], sb: &[usize], db: &[T], f: impl Fn(T, T) -> U) -> R<(Vec<usize>, Vec<U>)> {
    let (sa, sb) = align_shapes(sa, sb);
    let out = broadcast(&sa, &sb)?;
    if sa == sb {
        return Ok((out, da.iter().zip(db).map(|(x, y)| f(*x, *y)).collect()));
    }
    let pa = positions(&sa, &out);
    let pb = positions(&sb, &out);
    Ok((out, pa.iter().zip(&pb).map(|(i, j)| f(da[*i], db[*j])).collect()))
}

fn wrap_bits(v: i64, bits: u32) -> i64 {
    if bits >= 64 {
        v
    } else {
        let h = 1i64 << (bits - 1);
        let m = (1i64 << bits) - 1;
        (v.wrapping_add(h) & m).wrapping_sub(h)
    }
}

/// `pow(x, y, 2**bits)` for y >= 0, wrapped to kg_int; 0 for a negative exponent.
fn ipow(x: i64, y: i64, bits: u32) -> i64 {
    if y < 0 {
        return 0;
    }
    let mut r: u64 = 1;
    let mut b: u64 = x as u64;
    let mut e = y as u64;
    while e > 0 {
        if e & 1 == 1 {
            r = r.wrapping_mul(b);
        }
        b = b.wrapping_mul(b);
        e >>= 1;
    }
    wrap_bits(r as i64, bits)
}

fn fmax(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if a > b {
        a
    } else {
        b
    }
}

fn fmin(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if a < b {
        a
    } else {
        b
    }
}

fn isclose(a: f64, b: f64) -> bool {
    let (rtol, atol) = (1e-5, 1e-8);
    if a.is_finite() && b.is_finite() {
        (a - b).abs() <= atol + rtol * b.abs()
    } else {
        a == b
    }
}

/// `np.array_equal`: equal shapes and equal elements (nan is not equal to nan).
fn array_equal(a: &Arr, b: &Arr) -> bool {
    if a.shape != b.shape {
        return false;
    }
    match (&a.data, &b.data) {
        (Data::I(x), Data::I(y)) => x == y,
        _ => a.f64s().iter().zip(b.f64s().iter()).all(|(x, y)| x == y),
    }
}

// ---------------------------------------------------------------------------
// The evaluator
// ---------------------------------------------------------------------------

struct Env {
    params: Vec<Arr>,
    locals: Vec<Arr>,
}

/// The callback that runs an extern: its name and its arguments (a `str` argument as its code points, rank 1), its result.
pub type ExternCb = Box<dyn FnMut(&str, &[Arr]) -> Result<Arr, String>>;

/// `P` is the program, owned or borrowed: `IrEvaluator::new(&program)`, or `IrEvaluator::new(Rc<Program>)`.
pub struct IrEvaluator<P: Deref<Target = Program> = std::rc::Rc<Program>> {
    pub program: P,
    bits: u32,
    consts: HashMap<String, Arr>,
    initial: HashMap<String, Arr>,
    pub state: HashMap<String, Arr>,
    /// What an `ExternCall` calls (Python's evaluator raises NotImplementedError: there it is an error when absent).
    pub extern_cb: Option<ExternCb>,
}

fn value_arr(ty: &Ty, v: &V) -> R<Arr> {
    let shape = ty.shape.clone();
    let a = match v {
        V::I(i) => Arr::int(*i),
        V::F(f) => Arr::flt(*f),
        V::AI(d) => Arr::ints(shape, d.clone()),
        V::AF(d) => Arr::flts(shape, d.clone()),
        V::S(s) => Arr::ints(vec![s.chars().count()], s.chars().map(|c| c as i64).collect()),
    };
    Ok(a.as_elem(&ty.elem))
}

impl<P: Deref<Target = Program>> IrEvaluator<P> {
    pub fn new(program: P) -> IrEvaluator<P> {
        let bits = program.int_bits;
        let mut consts = HashMap::new();
        let mut initial = HashMap::new();
        for g in &program.globals {
            if let Ok(a) = value_arr(&g.ty, &g.value) {
                if g.state {
                    initial.insert(g.name.clone(), a);
                } else {
                    consts.insert(g.name.clone(), a);
                }
            }
        }
        let state = initial.clone();
        IrEvaluator { program, bits, consts, initial, state, extern_cb: None }
    }

    pub fn reset(&mut self) {
        self.state = self.initial.clone();
    }

    // -- integers ---------------------------------------------------------------
    fn wrap(&self, v: i64) -> i64 {
        wrap_bits(v, self.bits)
    }

    /// A value of element type `src` as `dst` (the conversions of the IR).
    pub fn cast(&self, v: Arr, src: &str, dst: &str) -> Arr {
        if src == dst {
            return v;
        }
        if dst == "float" {
            if is_fixed(src) {
                let s = fx::spec_of(src);
                let d: Vec<f64> = v.i64s().iter().map(|r| fx::to_float(*r, s)).collect();
                return Arr::flts(v.shape, d);
            }
            return v.as_elem("float");
        }
        if is_fixed(dst) {
            let d = fx::spec_of(dst);
            let out: Vec<i64> = if is_fixed(src) {
                let s = fx::spec_of(src);
                v.i64s().iter().map(|r| fx::convert(*r, s, d)).collect()
            } else if src == "float" {
                v.f64s().iter().map(|x| fx::from_float(*x, d)).collect()
            } else {
                v.i64s().iter().map(|x| fx::from_int(*x, d)).collect()
            };
            return Arr::ints(v.shape, out);
        }
        v
    }

    // -- running ------------------------------------------------------------------
    pub fn run(&mut self, entry: &Entry, args: &[Arr]) -> R<Arr> {
        let f = entry.f.clone();
        self.run_fn(&f, args)
    }

    pub fn run_fn(&mut self, f: &FnSpec, args: &[Arr]) -> R<Arr> {
        if args.len() != f.param_tys.len() {
            return unsupported(format!("{} takes {} arguments, got {}", f.name, f.param_tys.len(), args.len()));
        }
        let vals: Vec<Arr> = args.iter().zip(&f.param_tys).map(|(a, t)| a.clone().as_elem(&t.elem)).collect();
        self.call(f, vals)
    }

    fn call(&mut self, f: &FnSpec, args: Vec<Arr>) -> R<Arr> {
        let mut env = Env { params: args, locals: f.locals.iter().map(Arr::zero).collect() };
        let mut out = None;
        for s in f.stmts() {
            out = Some(self.ev(s, &mut env)?);
        }
        match out {
            Some(o) => Ok(o),
            None => unsupported(format!("{} has no statements", f.name)),
        }
    }

    fn index(&self, i: &Arr, size: usize) -> R<Vec<usize>> {
        let mut out = Vec::new();
        for &x in i.i64s().iter() {
            let x = if x < 0 { x.wrapping_add(size as i64) } else { x };
            if x < 0 || x >= size as i64 {
                return Err(Trap::new(TRAP_BOUNDS, "index out of range"));
            }
            out.push(x as usize);
        }
        Ok(out)
    }

    fn caps(&mut self, caps: &[crate::ir::N], env: &mut Env) -> R<Vec<Arr>> {
        caps.iter().map(|c| self.ev(c, env)).collect()
    }

    fn ev(&mut self, n: &Node, env: &mut Env) -> R<Arr> {
        match &n.k {
            K::Const(v) => value_arr(&n.ty, v),
            K::GlobalRef { name, state, .. } => {
                let m = if *state { &self.state } else { &self.consts };
                match m.get(name) {
                    Some(a) => Ok(a.clone()),
                    None => unsupported(format!("global {} has no value", name)),
                }
            }
            K::LocalRef { slot, .. } => Ok(env.locals[*slot].clone()),
            K::ParamRef { index, .. } => Ok(env.params[*index].clone()),
            K::Convert { a } => {
                let v = self.ev(a, env)?;
                Ok(self.cast(v, &a.ty.elem, &n.ty.elem))
            }
            K::Unary { op, a, .. } => {
                let v = self.ev(a, env)?;
                self.unary(op, &a.ty.elem, v)
            }
            K::Binary { op, a, b } => {
                let bv = self.ev(b, env)?;
                let av = self.ev(a, env)?;
                self.binary(op, &a.ty.elem, av, bv)
            }
            K::Match { a, b } => {
                let bv = self.ev(b, env)?;
                let av = self.ev(a, env)?;
                if a.ty.elem == "float" && a.ty.shape.is_empty() {
                    let (x, y) = (av.f64s(), bv.f64s());
                    if x.len() != 1 || y.len() != 1 {
                        return unsupported("a scalar match of arrays");
                    }
                    return Ok(Arr::int(truth(isclose(x[0], y[0]))));
                }
                Ok(Arr::int(truth(array_equal(&av, &bv))))
            }
            K::Index { a, i, .. } => {
                let iv = self.ev(i, env)?;
                let av = self.ev(a, env)?;
                if av.ndim() == 0 {
                    return unsupported("too many indices for array");
                }
                let ix = self.index(&iv, av.len())?;
                if ix.len() != 1 {
                    return unsupported("a scalar index was expected");
                }
                Ok(av.row(ix[0]))
            }
            K::Gather { a, idx, .. } => {
                let iv = self.ev(idx, env)?;
                let av = self.ev(a, env)?;
                if av.ndim() == 0 {
                    return unsupported("too many indices for array");
                }
                let ix = self.index(&iv, av.len())?;
                let mut out = av.rows_at(&ix);
                let mut shape = iv.shape.clone();
                shape.extend_from_slice(&av.shape[1..]);
                out.shape = shape;
                Ok(out)
            }
            K::Take { a, n: cnt } => {
                let av = self.ev(a, env)?;
                let size = av.len() as i64;
                let count = cnt.abs();
                if size == 0 && count > 0 {
                    return unsupported("cannot take from an empty array");
                }
                let start = if *cnt > 0 { 0 } else { (-count).rem_euclid(size.max(1)) };
                let idx: Vec<usize> = (0..count).map(|k| (start + k).rem_euclid(size) as usize).collect();
                Ok(av.rows_at(&idx))
            }
            K::Drop { a, n: cnt } => {
                let av = self.ev(a, env)?;
                let len = av.len() as i64;
                let (lo, hi) = if *cnt > 0 { ((*cnt).min(len), len) } else { (0, (len + *cnt).max(0).min(len)) };
                let idx: Vec<usize> = (lo..hi.max(lo)).map(|k| k as usize).collect();
                Ok(av.rows_at(&idx))
            }
            K::Rotate { a, n: k } => {
                let kv = self.ev(k, env)?.item_i()?;
                let av = self.ev(a, env)?;
                let size = av.len() as i64;
                let idx: Vec<usize> = (0..size).map(|j| (j - kv).rem_euclid(size) as usize).collect();
                Ok(av.rows_at(&idx))
            }
            K::Reverse { a } => {
                let av = self.ev(a, env)?;
                let idx: Vec<usize> = (0..av.len()).rev().collect();
                Ok(av.rows_at(&idx))
            }
            K::Transpose { a } => {
                let av = self.ev(a, env)?;
                Ok(transpose(&av))
            }
            K::Reshape { a } => {
                let av = self.ev(a, env)?;
                let total: usize = n.ty.shape.iter().product();
                let m = av.size();
                Ok(match &av.data {
                    Data::I(d) => Arr::ints(n.ty.shape.clone(), (0..total).map(|k| if m == 0 { 0 } else { d[k % m] }).collect()),
                    Data::F(d) => Arr::flts(n.ty.shape.clone(), (0..total).map(|k| if m == 0 { 0.0 } else { d[k % m] }).collect()),
                })
            }
            K::Join { a, b } => {
                let bv = self.ev(b, env)?;
                let av = self.ev(a, env)?;
                join(av, bv)
            }
            K::Iota { n: cnt } => Ok(Arr::ints(vec![(*cnt).max(0) as usize], (0..*cnt).collect())),
            K::MakeList { items } => {
                let rows = items.iter().map(|q| self.ev(q, env)).collect::<R<Vec<_>>>()?;
                Arr::stack(&rows)
            }
            K::Amend { a, value, idx, .. } => {
                let iv = self.ev(idx, env)?;
                let vv = self.ev(value, env)?;
                let mut av = self.ev(a, env)?;
                let ix = self.index(&iv, av.len())?;
                amend(&mut av, &iv.shape, &ix, &vv)?;
                Ok(av)
            }
            K::Cond { c, t, e } => {
                let cv = self.ev(c, env)?;
                if cv.truthy()? {
                    self.ev(t, env)
                } else {
                    self.ev(e, env)
                }
            }
            K::Call { f, args } => {
                let vals = args.iter().map(|q| self.ev(q, env)).collect::<R<Vec<_>>>()?;
                self.call(f, vals)
            }
            K::ExternCall { ext, args } => {
                let vals = args.iter().map(|q| self.ev(q, env)).collect::<R<Vec<_>>>()?;
                let Some(cb) = self.extern_cb.as_mut() else {
                    return Err(Trap::new(TRAP_EXTERN, "externs are not available to the fixed-point evaluator"));
                };
                let r = cb(&ext.name, &vals).map_err(|m| Trap::new(TRAP_EXTERN, m))?;
                Ok(r)
            }
            K::Map { f, a, caps } => {
                let av = self.ev(a, env)?;
                let cv = self.caps(caps, env)?;
                let mut rows = Vec::new();
                for j in 0..av.len() {
                    let mut args = vec![av.row(j)];
                    args.extend(cv.iter().cloned());
                    rows.push(self.call(f, args)?);
                }
                self.stack_or_empty(rows, &n.ty)
            }
            K::Map2 { f, a, b, caps } => {
                let av = self.ev(a, env)?;
                let bv = self.ev(b, env)?;
                let cv = self.caps(caps, env)?;
                let mut rows = Vec::new();
                for j in 0..av.len() {
                    let mut args = vec![av.row(j), bv.row(j)];
                    args.extend(cv.iter().cloned());
                    rows.push(self.call(f, args)?);
                }
                self.stack_or_empty(rows, &n.ty)
            }
            K::EachPair { f, a, caps } => {
                let av = self.ev(a, env)?;
                let cv = self.caps(caps, env)?;
                let mut rows = Vec::new();
                for j in 0..av.len().saturating_sub(1) {
                    let mut args = vec![av.row(j), av.row(j + 1)];
                    args.extend(cv.iter().cloned());
                    rows.push(self.call(f, args)?);
                }
                self.stack_or_empty(rows, &n.ty)
            }
            K::Reduce { f, a, seed, caps } => self.fold(f, a, seed.as_ref(), caps, env, &n.ty.elem, false),
            K::Scan { f, a, seed, caps, acc_ty } => self.fold(f, a, seed.as_ref(), caps, env, &acc_ty.elem, true),
            K::Loop { kind, f, init, count, pred, caps, pcaps } => {
                let cnt = if kind == "iterate" { Some(self.ev(count.as_ref().ok_or_else(|| Trap::new(0, "iterate without a count"))?, env)?.item_i()?) } else { None };
                let mut state = self.ev(init, env)?;
                let cv = self.caps(caps, env)?;
                if let Some(cnt) = cnt {
                    for _ in 0..cnt.max(0) {
                        let mut args = vec![state];
                        args.extend(cv.iter().cloned());
                        state = self.call(f, args)?;
                    }
                    return Ok(state);
                }
                let pcv = self.caps(pcaps, env)?;
                let Some(pred) = pred else { return unsupported("while without a predicate") };
                loop {
                    let mut pa = vec![state.clone()];
                    pa.extend(pcv.iter().cloned());
                    if !self.call(pred, pa)?.truthy()? {
                        break;
                    }
                    let mut args = vec![state];
                    args.extend(cv.iter().cloned());
                    state = self.call(f, args)?;
                }
                Ok(state)
            }
            K::Assign { target, value } => {
                let v = self.ev(value, env)?;
                match &target.k {
                    K::LocalRef { slot, .. } => env.locals[*slot] = v.clone(),
                    K::GlobalRef { name, .. } => {
                        self.state.insert(name.clone(), v.clone());
                    }
                    _ => return unsupported("an assignment to something that is not a variable"),
                }
                Ok(v)
            }
            K::Seq { stmts } => {
                let mut out = None;
                for s in stmts {
                    out = Some(self.ev(s, env)?);
                }
                match out {
                    Some(o) => Ok(o),
                    None => unsupported("an empty sequence"),
                }
            }
        }
    }

    /// `np.stack(rows)`; an empty list gives an empty array of the node's type (Python raises ValueError).
    fn stack_or_empty(&self, rows: Vec<Arr>, ty: &Ty) -> R<Arr> {
        if rows.is_empty() {
            let mut shape = ty.shape.clone();
            if shape.is_empty() {
                shape.push(0);
            } else {
                shape[0] = 0;
            }
            return Ok(Arr::zero(&Ty { elem: ty.elem.clone(), shape }));
        }
        Arr::stack(&rows)
    }

    #[allow(clippy::too_many_arguments)]
    fn fold(&mut self, f: &FnSpec, a: &Node, seed: Option<&crate::ir::N>, caps: &[crate::ir::N], env: &mut Env, acc_elem: &str, scan: bool) -> R<Arr> {
        let sv = match seed {
            Some(s) => Some(self.ev(s, env)?),
            None => None,
        };
        let av = self.ev(a, env)?;
        let cv = self.caps(caps, env)?;
        let (acc, first) = match sv {
            Some(s) => (s, 0),
            None => {
                if av.len() == 0 {
                    return unsupported("index 0 is out of bounds for axis 0 with size 0");
                }
                (av.row(0), 1)
            }
        };
        let src = seed.map(|s| s.ty.elem.as_str()).unwrap_or(a.ty.elem.as_str());
        let mut acc = self.cast(acc, src, acc_elem);
        let mut out = vec![acc.clone()];
        for j in first..av.len() {
            let mut args = vec![acc, av.row(j)];
            args.extend(cv.iter().cloned());
            let r = self.call(f, args)?;
            acc = self.cast(r, &f.result_ty.elem, acc_elem);
            out.push(acc.clone());
        }
        if scan {
            Arr::stack(&out)
        } else {
            Ok(acc)
        }
    }

    fn unary(&self, op: &str, elem: &str, a: Arr) -> R<Arr> {
        if is_fixed(elem) {
            let s = fx::spec_of(elem);
            match op {
                "neg" => return Ok(Arr::ints(a.shape.clone(), a.i64s().iter().map(|x| fx::neg(*x, s)).collect())),
                "floor" => return Ok(Arr::ints(a.shape.clone(), a.i64s().iter().map(|x| fx::floor_int(*x, s, self.bits)).collect())),
                "not" => return Ok(Arr::ints(a.shape.clone(), a.i64s().iter().map(|x| truth(*x == 0)).collect())),
                _ => {}
            }
        }
        match op {
            "neg" => Ok(match &a.data {
                Data::I(d) if elem == "int" => Arr::ints(a.shape.clone(), d.iter().map(|x| self.wrap(x.wrapping_neg())).collect()),
                Data::I(d) => Arr::ints(a.shape.clone(), d.iter().map(|x| x.wrapping_neg()).collect()),
                Data::F(d) => Arr::flts(a.shape.clone(), d.iter().map(|x| -x).collect()),
            }),
            "floor" => {
                let (hi, lo) = if self.bits == 64 { (i64::MAX, i64::MIN) } else { ((1i64 << (self.bits - 1)) - 1, -(1i64 << (self.bits - 1))) };
                let lim = (1u128 << (self.bits - 1)) as f64;
                let d: Vec<i64> = match &a.data {
                    Data::I(d) => d.iter().map(|x| (*x).clamp(lo, hi)).collect(),
                    Data::F(d) => d
                        .iter()
                        .map(|x| {
                            let x = if x.is_nan() {
                                0.0
                            } else if *x == f64::INFINITY {
                                2f64.powi(70)
                            } else if *x == f64::NEG_INFINITY {
                                -(2f64.powi(70))
                            } else {
                                *x
                            };
                            let f = x.floor();
                            if f >= lim {
                                hi
                            } else if f < -lim {
                                lo
                            } else {
                                f as i64
                            }
                        })
                        .collect(),
                };
                Ok(Arr::ints(a.shape.clone(), d))
            }
            "recip" => Ok(Arr::flts(a.shape.clone(), a.f64s().iter().map(|x| 1.0 / x).collect())),
            "not" => Ok(match &a.data {
                Data::I(d) => Arr::ints(a.shape.clone(), d.iter().map(|x| truth(*x == 0)).collect()),
                Data::F(d) => Arr::ints(a.shape.clone(), d.iter().map(|x| truth(*x == 0.0)).collect()),
            }),
            _ => unsupported(format!("unary operator {}", op)),
        }
    }

    fn binary(&self, op: &str, elem: &str, a: Arr, b: Arr) -> R<Arr> {
        if is_fixed(elem) {
            let s = fx::spec_of(elem);
            let (x, y) = (a.i64s(), b.i64s());
            let go = |f: &dyn Fn(i64, i64) -> i64| -> R<Arr> {
                let (shape, d) = zip2(&a.shape, &x, &b.shape, &y, |p, q| f(p, q))?;
                Ok(Arr::ints(shape, d))
            };
            return match op {
                "add" => go(&|p, q| fx::add(p, q, s)),
                "sub" => go(&|p, q| fx::sub(p, q, s)),
                "mul" => go(&|p, q| fx::mul(p, q, s)),
                "div" => {
                    if y.iter().any(|v| *v == 0) {
                        return Err(Trap::new(TRAP_DIV0, "fixed-point division by zero"));
                    }
                    go(&|p, q| fx::div(p, q, s))
                }
                "max" => go(&|p, q| p.max(q)),
                "min" => go(&|p, q| p.min(q)),
                "lt" => go(&|p, q| truth(p < q)),
                "gt" => go(&|p, q| truth(p > q)),
                "eq" => go(&|p, q| truth(p == q)),
                _ => unsupported(format!("fixed-point operator {}", op)),
            };
        }
        if elem == "int" {
            let (x, y) = (a.i64s(), b.i64s());
            let go = |f: &dyn Fn(i64, i64) -> i64| -> R<Arr> {
                let (shape, d) = zip2(&a.shape, &x, &b.shape, &y, |p, q| f(p, q))?;
                Ok(Arr::ints(shape, d))
            };
            let bits = self.bits;
            match op {
                "add" => return go(&|p, q| wrap_bits(p.wrapping_add(q), bits)),
                "sub" => return go(&|p, q| wrap_bits(p.wrapping_sub(q), bits)),
                "mul" => return go(&|p, q| wrap_bits(p.wrapping_mul(q), bits)),
                "idiv" | "rem" => {
                    if y.iter().any(|v| *v == 0) {
                        return Err(Trap::new(TRAP_DIV0, "integer division by zero"));
                    }
                    return if op == "idiv" {
                        go(&|p, q| wrap_bits(p.wrapping_div(q), bits))
                    } else {
                        go(&|p, q| wrap_bits(p.wrapping_sub(p.wrapping_div(q).wrapping_mul(q)), bits))
                    };
                }
                "pow" => return go(&|p, q| ipow(p, q, bits)),
                _ => {}
            }
        } else {
            let (x, y) = (a.f64s(), b.f64s());
            let go = |f: &dyn Fn(f64, f64) -> f64| -> R<Arr> {
                let (shape, d) = zip2(&a.shape, &x, &b.shape, &y, |p, q| f(p, q))?;
                Ok(Arr::flts(shape, d))
            };
            match op {
                "add" => return go(&|p, q| p + q),
                "sub" => return go(&|p, q| p - q),
                "mul" => return go(&|p, q| p * q),
                "div" => return go(&|p, q| p / q),
                "rem" => return go(&|p, q| p % q),
                "pow" => return go(&|p, q| p.powf(q)),
                _ => {}
            }
        }
        // what both fall through to
        if op == "div" {
            let (x, y) = (a.f64s(), b.f64s());
            let (shape, d) = zip2(&a.shape, &x, &b.shape, &y, |p, q| p / q)?;
            return Ok(Arr::flts(shape, d));
        }
        if a.is_float() || b.is_float() {
            let (x, y) = (a.f64s(), b.f64s());
            match op {
                "max" => {
                    let (shape, d) = zip2(&a.shape, &x, &b.shape, &y, fmax)?;
                    return Ok(Arr::flts(shape, d));
                }
                "min" => {
                    let (shape, d) = zip2(&a.shape, &x, &b.shape, &y, fmin)?;
                    return Ok(Arr::flts(shape, d));
                }
                "lt" | "gt" | "eq" => {
                    let (shape, d) = zip2(&a.shape, &x, &b.shape, &y, |p, q| truth(match op {
                        "lt" => p < q,
                        "gt" => p > q,
                        _ => p == q,
                    }))?;
                    return Ok(Arr::ints(shape, d));
                }
                _ => {}
            }
        } else {
            let (x, y) = (a.i64s(), b.i64s());
            let (shape, d) = match op {
                "max" => zip2(&a.shape, &x, &b.shape, &y, |p: i64, q: i64| p.max(q))?,
                "min" => zip2(&a.shape, &x, &b.shape, &y, |p: i64, q: i64| p.min(q))?,
                "lt" => zip2(&a.shape, &x, &b.shape, &y, |p, q| truth(p < q))?,
                "gt" => zip2(&a.shape, &x, &b.shape, &y, |p, q| truth(p > q))?,
                "eq" => zip2(&a.shape, &x, &b.shape, &y, |p, q| truth(p == q))?,
                _ => return unsupported(format!("operator {}", op)),
            };
            return Ok(Arr::ints(shape, d));
        }
        unsupported(format!("operator {}", op))
    }
}

/// `.T`: the axes reversed.
fn transpose(a: &Arr) -> Arr {
    let nd = a.ndim();
    if nd < 2 {
        return a.clone();
    }
    let out_shape: Vec<usize> = a.shape.iter().rev().copied().collect();
    let mut stride = vec![1usize; nd];
    for k in (0..nd - 1).rev() {
        stride[k] = stride[k + 1] * a.shape[k + 1];
    }
    // the output axis k is the input axis nd-1-k
    let n = a.size();
    let mut pos = Vec::with_capacity(n);
    let mut idx = vec![0usize; nd];
    for _ in 0..n {
        pos.push((0..nd).map(|k| idx[k] * stride[nd - 1 - k]).sum::<usize>());
        for k in (0..nd).rev() {
            idx[k] += 1;
            if idx[k] < out_shape[k] {
                break;
            }
            idx[k] = 0;
        }
    }
    match &a.data {
        Data::I(d) => Arr::ints(out_shape, pos.iter().map(|&p| d[p]).collect()),
        Data::F(d) => Arr::flts(out_shape, pos.iter().map(|&p| d[p]).collect()),
    }
}

fn join(a: Arr, b: Arr) -> R<Arr> {
    let lift = |x: Arr, like: &Arr| -> Arr {
        if x.ndim() == 0 || x.ndim() + 1 == like.ndim() {
            let mut shape = vec![1];
            shape.extend_from_slice(&x.shape);
            Arr { shape, data: x.data }
        } else {
            x
        }
    };
    let (a2, b2) = (lift(a.clone(), &b), lift(b.clone(), &a));
    if a2.shape[1..] != b2.shape[1..] {
        return unsupported(format!("all the input array dimensions except for the concatenation axis must match exactly: {:?} {:?}", a2.shape, b2.shape));
    }
    let mut shape = a2.shape.clone();
    shape[0] += b2.shape[0];
    if a2.is_float() || b2.is_float() {
        let mut d = a2.f64s().into_owned();
        d.extend(b2.f64s().iter());
        Ok(Arr::flts(shape, d))
    } else {
        let mut d = a2.i64s().into_owned();
        d.extend(b2.i64s().iter());
        Ok(Arr::ints(shape, d))
    }
}

/// `a[ix] = value`: NumPy's assignment with the value broadcast to the indexed shape (the shape of the index, then the rest of `a`).
fn amend(a: &mut Arr, idx_shape: &[usize], ix: &[usize], value: &Arr) -> R<()> {
    let rest_shape: Vec<usize> = a.shape[1..].to_vec();
    let rs: usize = rest_shape.iter().product();
    let mut target: Vec<usize> = idx_shape.to_vec();
    target.extend_from_slice(&rest_shape);
    // pad the value's shape with leading ones (NumPy's broadcasting) and find the source of each target element
    if value.ndim() > target.len() {
        return unsupported("could not broadcast the value into the indexed shape");
    }
    let mut vs = vec![1usize; target.len() - value.ndim()];
    vs.extend_from_slice(&value.shape);
    for (v, t) in vs.iter().zip(&target) {
        if v != t && *v != 1 {
            return unsupported(format!("could not broadcast input array from shape {:?} into shape {:?}", value.shape, target));
        }
    }
    let src = positions(&vs, &target);
    match &mut a.data {
        Data::I(d) => {
            let v = value.i64s();
            for (p, &row) in ix.iter().enumerate() {
                for k in 0..rs {
                    d[row * rs + k] = v[src[p * rs + k]];
                }
            }
        }
        Data::F(d) => {
            let v = value.f64s();
            for (p, &row) in ix.iter().enumerate() {
                for k in 0..rs {
                    d[row * rs + k] = v[src[p * rs + k]];
                }
            }
        }
    }
    Ok(())
}
