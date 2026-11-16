//! Values (native-runtime.md section 3).
//!
//! Every payload is a scalar of at most 8 bytes or one thin pointer, so a `Value` is 16 bytes on
//! 64-bit targets and at most 16 on 32-bit ones (asserted in lib.rs).
//!
//! Normal form, enforced by [`Value::from_items`]: a list whose elements are all integers is
//! `Ints`; all numbers with at least one float, `Flts`; a rectangular nest of numeric rows is one
//! array of rank above 1; anything else is a `List`. Operators never see a `List` of plain
//! integers.
use alloc::boxed::Box;
use alloc::rc::Rc;
use alloc::vec::Vec;
use core::cell::RefCell;

use crate::error::{Error, Result};
use crate::sym::SymId;
use crate::tree::{NodeId, Tree};

#[cfg(not(feature = "int32"))]
pub type Int = i64;
#[cfg(feature = "int32")]
pub type Int = i32;
#[cfg(not(feature = "float32"))]
pub type Flt = f64;
#[cfg(feature = "float32")]
pub type Flt = f32;

pub const MAX_RANK: usize = 4;

/// A numeric array: flat data in row-major order plus a shape. Rank 1 is the common case and
/// its length is `data.len()`; `dims` is only meaningful for rank above 1.
pub struct Arr<T> {
    pub data: Vec<T>,
    rank: u8,
    dims: [u32; MAX_RANK],
}

impl<T: Copy> Arr<T> {
    pub fn vec(data: Vec<T>) -> Arr<T> {
        Arr { data, rank: 1, dims: [0; MAX_RANK] }
    }
    /// `dims` must multiply to `data.len()` and hold 2 to MAX_RANK entries.
    pub fn shaped(data: Vec<T>, dims: &[u32]) -> Arr<T> {
        if dims.len() <= 1 || dims.len() > MAX_RANK {
            return Arr::vec(data);
        }
        let mut d = [0; MAX_RANK];
        d[..dims.len()].copy_from_slice(dims);
        Arr { data, rank: dims.len() as u8, dims: d }
    }
    pub fn rank(&self) -> usize {
        self.rank as usize
    }
    /// Length along the leading dimension.
    pub fn len(&self) -> usize {
        if self.rank <= 1 {
            self.data.len()
        } else {
            self.dims[0] as usize
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn dim(&self, i: usize) -> usize {
        if self.rank <= 1 {
            self.data.len()
        } else {
            self.dims[i] as usize
        }
    }
    /// Dimensions after the first (empty for rank 1).
    pub fn row_dims(&self) -> &[u32] {
        if self.rank <= 1 {
            &[]
        } else {
            &self.dims[1..self.rank as usize]
        }
    }
    /// Number of scalars in one element along the leading dimension.
    pub fn row_size(&self) -> usize {
        self.row_dims().iter().map(|&d| d as usize).product()
    }
    pub fn same_shape(&self, o_rank: usize, o_len: usize, o_row: &[u32]) -> bool {
        self.rank() == o_rank && self.len() == o_len && self.row_dims() == o_row
    }
}

pub struct Bytes(pub Vec<u8>);

/// Insertion-ordered dictionary. A placeholder until ticket 103: lookup is linear.
#[derive(Default)]
pub struct Dict {
    pub keys: Vec<Value>,
    pub vals: Vec<Value>,
}

impl Dict {
    pub fn find(&self, k: &Value) -> Option<usize> {
        self.keys.iter().position(|q| key_eq(q, k))
    }
    pub fn get(&self, k: &Value) -> Option<&Value> {
        self.find(k).map(|i| &self.vals[i])
    }
    pub fn set(&mut self, k: Value, v: Value) {
        match self.find(&k) {
            Some(i) => self.vals[i] = v,
            None => {
                self.keys.push(k);
                self.vals.push(v);
            }
        }
    }
}

fn key_eq(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Int(x), Value::Int(y)) => x == y,
        (Value::Int(x), Value::Flt(y)) | (Value::Flt(y), Value::Int(x)) => *x as Flt == *y,
        (Value::Flt(x), Value::Flt(y)) => x == y,
        (Value::Char(x), Value::Char(y)) => x == y,
        (Value::Sym(x), Value::Sym(y)) => x == y,
        (Value::Str(x), Value::Str(y)) => x.0 == y.0,
        (Value::Char(c), Value::Str(s)) | (Value::Str(s), Value::Char(c)) => s.0.len() == 1 && s.0[0] == *c,
        _ => false,
    }
}

/// A system function supplied by the embedder (`Interp::register`).
pub type NativeFn = Rc<dyn Fn(&mut crate::interp::Interp, &[Value]) -> Result<Value>>;

pub const WILDCARD: u8 = 255;

/// One argument slot of a projection: empty, a value, or an expression of a tree that is
/// evaluated each time the projection is completed (a projection written at top level).
#[derive(Clone)]
pub enum Slot {
    Hole,
    Val(Value),
    Code(Rc<Tree>, NodeId),
}

/// What a function value can be. KlongPy has no closure type: a function value is a piece of
/// the parse tree (a literal, a projection written with holes, a call that was not complete),
/// and the conformance suite depends on the consequences, so the native core keeps that model.
pub enum Func {
    /// A tree node used as a value.
    Code { tree: Rc<Tree>, node: NodeId },
    /// A function applied to fewer values than it takes (made at run time by an adverb or `@`).
    /// A projection made at run time (a call with fewer arguments than the function takes, or
    /// with an empty slot): the callee and one slot for each argument.
    Proj { f: Value, args: Vec<Slot> },
    /// A system function. `arity` is `WILDCARD` for one that takes any number of arguments.
    Native { name: SymId, arity: u8, f: NativeFn },
}

/// An object of the embedder: a channel, an IPC handle. Placeholder until tickets 103 and 106.
pub trait Ext {
    fn type_name(&self) -> &str;
}
pub struct ExtBox(pub Box<dyn Ext>);

#[derive(Clone)]
pub enum Value {
    Int(Int),
    Flt(Flt),
    Char(u8),
    Sym(SymId),
    Undefined,
    Ints(Rc<Arr<Int>>),
    Flts(Rc<Arr<Flt>>),
    /// A string: bytes (UTF-8 text). `#` counts bytes.
    Str(Rc<Bytes>),
    /// Mixed or ragged.
    List(Rc<Vec<Value>>),
    Dict(Rc<RefCell<Dict>>),
    Fn(Rc<Func>),
    /// An operator as a value; the id is the interned glyph.
    Prim(SymId),
    Ext(Rc<ExtBox>),
}

/// A vector with room for `n` elements, or a `memory` error. Every allocation whose size
/// depends on data goes through here or through `try_reserve`.
pub fn try_vec<T>(n: usize) -> Result<Vec<T>> {
    let mut v = Vec::new();
    v.try_reserve_exact(n).map_err(|_| Error::memory())?;
    Ok(v)
}

pub fn try_push<T>(v: &mut Vec<T>, x: T) -> Result<()> {
    if v.len() == v.capacity() {
        v.try_reserve(1).map_err(|_| Error::memory())?;
    }
    v.push(x);
    Ok(())
}

pub fn try_extend<T: Copy>(v: &mut Vec<T>, xs: &[T]) -> Result<()> {
    v.try_reserve(xs.len()).map_err(|_| Error::memory())?;
    v.extend_from_slice(xs);
    Ok(())
}

impl Value {
    pub fn ints(v: Vec<Int>) -> Value {
        Value::Ints(Rc::new(Arr::vec(v)))
    }
    pub fn flts(v: Vec<Flt>) -> Value {
        Value::Flts(Rc::new(Arr::vec(v)))
    }
    pub fn str(v: Vec<u8>) -> Value {
        Value::Str(Rc::new(Bytes(v)))
    }
    pub fn empty() -> Value {
        Value::ints(Vec::new())
    }
    pub fn bool(b: bool) -> Value {
        Value::Int(b as Int)
    }
    /// A list of exactly these elements, without normalising. Only for elements known not to
    /// be all numbers.
    pub fn raw_list(v: Vec<Value>) -> Value {
        Value::List(Rc::new(v))
    }

    /// The normalising constructor (KlongPy: `kg_asarray`).
    pub fn from_items(items: Vec<Value>) -> Result<Value> {
        if items.is_empty() {
            return Ok(Value::empty());
        }
        let mut ints = true;
        let mut nums = true;
        for q in &items {
            match q {
                Value::Int(_) => {}
                Value::Flt(_) => ints = false,
                _ => {
                    ints = false;
                    nums = false;
                    break;
                }
            }
        }
        if ints {
            let mut v = try_vec(items.len())?;
            for q in &items {
                if let Value::Int(i) = q {
                    v.push(*i);
                }
            }
            return Ok(Value::ints(v));
        }
        if nums {
            let mut v = try_vec(items.len())?;
            for q in &items {
                match q {
                    Value::Int(i) => v.push(*i as Flt),
                    Value::Flt(f) => v.push(*f),
                    _ => {}
                }
            }
            return Ok(Value::flts(v));
        }
        // A rectangular nest of numeric rows becomes one array of higher rank.
        let (rank, len, any_flt) = match &items[0] {
            Value::Ints(a) => (a.rank(), a.len(), false),
            Value::Flts(a) => (a.rank(), a.len(), true),
            _ => return Ok(Value::raw_list(items)),
        };
        if len == 0 || rank >= MAX_RANK {
            return Ok(Value::raw_list(items));
        }
        let mut row = [0u32; MAX_RANK];
        let row_rank = rank - 1;
        match &items[0] {
            Value::Ints(a) => row[..row_rank].copy_from_slice(a.row_dims()),
            Value::Flts(a) => row[..row_rank].copy_from_slice(a.row_dims()),
            _ => {}
        }
        let row = &row[..row_rank];
        let mut flt = any_flt;
        for q in &items[1..] {
            match q {
                Value::Ints(a) if a.same_shape(rank, len, row) => {}
                Value::Flts(a) if a.same_shape(rank, len, row) => flt = true,
                _ => return Ok(Value::raw_list(items)),
            }
        }
        let mut dims = [0u32; MAX_RANK];
        dims[0] = items.len() as u32;
        dims[1] = len as u32;
        dims[2..rank + 1].copy_from_slice(row);
        let dims = &dims[..rank + 1];
        let total: usize = dims.iter().map(|&d| d as usize).product();
        if flt {
            let mut v: Vec<Flt> = try_vec(total)?;
            for q in &items {
                match q {
                    Value::Ints(a) => v.extend(a.data.iter().map(|&i| i as Flt)),
                    Value::Flts(a) => v.extend_from_slice(&a.data),
                    _ => {}
                }
            }
            Ok(Value::Flts(Rc::new(Arr::shaped(v, dims))))
        } else {
            let mut v: Vec<Int> = try_vec(total)?;
            for q in &items {
                if let Value::Ints(a) = q {
                    v.extend_from_slice(&a.data);
                }
            }
            Ok(Value::Ints(Rc::new(Arr::shaped(v, dims))))
        }
    }

    /// Ints, Flts or List: what KlongPy calls a list (an array of rank above 0).
    pub fn is_list(&self) -> bool {
        matches!(self, Value::Ints(_) | Value::Flts(_) | Value::List(_))
    }
    pub fn is_str(&self) -> bool {
        matches!(self, Value::Str(_))
    }
    /// A list or a string.
    pub fn is_iterable(&self) -> bool {
        self.is_list() || self.is_str()
    }
    pub fn is_number(&self) -> bool {
        matches!(self, Value::Int(_) | Value::Flt(_))
    }
    /// Length along the leading dimension, for lists and strings.
    pub fn len(&self) -> Option<usize> {
        match self {
            Value::Ints(a) => Some(a.len()),
            Value::Flts(a) => Some(a.len()),
            Value::List(a) => Some(a.len()),
            Value::Str(s) => Some(s.0.len()),
            _ => None,
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == Some(0)
    }
    /// Everything except a non-empty list or string.
    pub fn is_atom(&self) -> bool {
        match self.len() {
            Some(n) => n == 0,
            None => true,
        }
    }
    /// Element `i` along the leading dimension. `i` must be in range.
    pub fn item(&self, i: usize) -> Value {
        match self {
            Value::Ints(a) => {
                if a.rank() <= 1 {
                    Value::Int(a.data[i])
                } else {
                    let n = a.row_size();
                    Value::Ints(Rc::new(Arr::shaped(a.data[i * n..(i + 1) * n].to_vec(), a.row_dims())))
                }
            }
            Value::Flts(a) => {
                if a.rank() <= 1 {
                    Value::Flt(a.data[i])
                } else {
                    let n = a.row_size();
                    Value::Flts(Rc::new(Arr::shaped(a.data[i * n..(i + 1) * n].to_vec(), a.row_dims())))
                }
            }
            Value::List(a) => a[i].clone(),
            Value::Str(s) => Value::Char(s.0[i]),
            _ => self.clone(),
        }
    }
    /// The elements along the leading dimension.
    pub fn items(&self) -> Result<Vec<Value>> {
        let n = self.len().unwrap_or(0);
        let mut v = try_vec(n)?;
        for i in 0..n {
            v.push(self.item(i));
        }
        Ok(v)
    }
    /// True unless the value is the number 0 or an empty list or string (the rule of `:[`).
    pub fn truthy(&self) -> bool {
        match self {
            Value::Int(i) => *i != 0,
            Value::Flt(f) => *f != 0.0,
            _ => !self.is_empty(),
        }
    }
    pub fn as_int(&self) -> Option<Int> {
        match self {
            Value::Int(i) => Some(*i),
            _ => None,
        }
    }
    pub fn as_flt(&self) -> Option<Flt> {
        match self {
            Value::Int(i) => Some(*i as Flt),
            Value::Flt(f) => Some(*f),
            _ => None,
        }
    }
    /// A word for error messages.
    pub fn describe(&self) -> &'static str {
        match self {
            Value::Int(_) => "integer",
            Value::Flt(_) => "float",
            Value::Char(_) => "char",
            Value::Sym(_) => "symbol",
            Value::Undefined => ":undefined",
            Value::Ints(_) | Value::Flts(_) | Value::List(_) => "list",
            Value::Str(_) => "string",
            Value::Dict(_) => "dict",
            Value::Fn(_) | Value::Prim(_) => "function",
            Value::Ext(_) => "object",
        }
    }
}

/// `=` on two atoms: 1 == 1.0, a symbol equals the same symbol, a string the same string.
pub fn atom_eq(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Undefined, Value::Undefined) => true,
        (Value::Prim(x), Value::Prim(y)) => x == y,
        (Value::Fn(x), Value::Fn(y)) => Rc::ptr_eq(x, y),
        (Value::Dict(x), Value::Dict(y)) => Rc::ptr_eq(x, y),
        _ => key_eq(a, b),
    }
}

fn is_close(a: Flt, b: Flt) -> bool {
    // numpy.isclose with its default tolerances
    if a == b {
        return true;
    }
    if !a.is_finite() || !b.is_finite() {
        return false;
    }
    let d = if a > b { a - b } else { b - a };
    let m = if b < 0.0 { -b } else { b };
    d <= 1e-8 + 1e-5 * m
}

/// `~` (Match). Numbers match when they are close (as numpy.isclose says); two numeric arrays
/// match when they are equal; lists match element by element.
pub fn matches(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Ints(x), Value::Ints(y)) => x.rank() == y.rank() && x.len() == y.len() && x.row_dims() == y.row_dims() && x.data == y.data,
        (Value::Flts(x), Value::Flts(y)) => x.rank() == y.rank() && x.len() == y.len() && x.row_dims() == y.row_dims() && x.data == y.data,
        (Value::Ints(x), Value::Flts(y)) | (Value::Flts(y), Value::Ints(x)) => {
            x.rank() == y.rank()
                && x.len() == y.len()
                && x.row_dims() == y.row_dims()
                && x.data.iter().zip(y.data.iter()).all(|(&i, &f)| i as Flt == f)
        }
        _ => {
            if a.is_list() || b.is_list() {
                if !(a.is_list() && b.is_list()) {
                    return false;
                }
                let n = a.len().unwrap_or(0);
                if b.len() != Some(n) {
                    return false;
                }
                return (0..n).all(|i| matches(&a.item(i), &b.item(i)));
            }
            match (a.as_flt(), b.as_flt()) {
                (Some(x), Some(y)) => match (a, b) {
                    (Value::Int(i), Value::Int(j)) => i == j,
                    _ => is_close(x, y),
                },
                _ => atom_eq(a, b),
            }
        }
    }
}
