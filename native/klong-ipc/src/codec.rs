//! kdb+ object codec: a Klong [`Value`] to and from the bytes of one object (no message
//! header). The Rust twin of `klongpy/ipc/codec.py`; mapping tables in `ipc-kdb.md` sections
//! 3 and 4, layouts in `ref-kdb-ipc.md` section 3. Both are tested against the same
//! `tests/ipc_vectors/vectors.json`.
//!
//! Encoding is always little-endian. An integer goes out as a long (an int from an `int32`
//! build), a float as a float (a real from a `float32` build).
//!
//! Decoding never executes anything and never panics. Every count is checked against the
//! bytes that are left before anything is allocated, nesting is limited to [`MAX_DEPTH`],
//! and allocations are fallible. Two kinds of failure ([`DecodeError`]): `Protocol` (the bytes
//! are not a kdb+ object: the connection is closed) and `Value` (the object is fine but this
//! build cannot hold it: a long that does not fit 32 bits, a symbol longer than 255 bytes, a
//! list as a dictionary key, no memory; only this message fails).
//!
//! Where the native value model differs from KlongPy's (documented in ticket 106):
//! a general list always takes Klong's normal form (`Value::from_items`), so a list mixing
//! integers and floats is a float array here and an object array in KlongPy; a dictionary key
//! must be a number, a char, a string or a symbol.
use alloc::boxed::Box;
use alloc::rc::Rc;
use alloc::string::String;
use alloc::vec::Vec;

use klong_core::error::{Error, Result};
use klong_core::sym::Interner;
use klong_core::tree::{Node, Tree};
use klong_core::value::{try_push, try_string, try_vec, Arr, Dict, Ext, ExtBox, Func};
use klong_core::{Flt, Int, Value, INT_BITS};

/// Nesting limit for general lists, dictionaries, tables and function objects.
pub const MAX_DEPTH: usize = 64;

/// The bytes of the generic null `(::)`.
pub const GENERIC_NULL: [u8; 2] = [0x65, 0x00];

const FLT_IS_32: bool = core::mem::size_of::<Flt>() == 4;
const INT_IS_32: bool = INT_BITS == 32;

// ---------------------------------------------------------------------------------------
// values that have no Klong form
// ---------------------------------------------------------------------------------------

/// A lambda (type 100): context name and source text, as received (so that it goes back
/// byte for byte). Context `kg` is a Klong function; anything else is the peer's.
pub struct KdbLambda {
    pub context: Vec<u8>,
    pub text: Vec<u8>,
}

impl KdbLambda {
    pub fn is_klong(&self) -> bool {
        self.context == b"kg"
    }
    pub fn text_str(&self) -> Option<&str> {
        core::str::from_utf8(&self.text).ok()
    }
}

/// A function object 101 to 111 other than `(::)`, kept as received: the complete object,
/// type byte first, little-endian. It can only be sent back.
pub struct KdbOpaque {
    pub raw: Vec<u8>,
}

/// The error object (type -128): the signal text. Decoded as a value; the endpoint raises.
pub struct KdbError {
    pub text: Vec<u8>,
}

impl Ext for KdbLambda {
    fn type_name(&self) -> &str {
        "kdb-lambda"
    }
    fn as_any(&self) -> &dyn core::any::Any {
        self
    }
}
impl Ext for KdbOpaque {
    fn type_name(&self) -> &str {
        "kdb-function"
    }
    fn as_any(&self) -> &dyn core::any::Any {
        self
    }
}
impl Ext for KdbError {
    fn type_name(&self) -> &str {
        "kdb-error"
    }
    fn as_any(&self) -> &dyn core::any::Any {
        self
    }
}

fn ext<T: Ext + 'static>(x: T) -> Value {
    Value::Ext(Rc::new(ExtBox(Box::new(x))))
}

/// The object of the embedder inside a value, if it is a `T`.
pub fn ext_ref<T: 'static>(v: &Value) -> Option<&T> {
    match v {
        Value::Ext(e) => e.0.as_any().downcast_ref::<T>(),
        _ => None,
    }
}

/// An error object as a value.
pub fn error_value(text: &[u8]) -> Result<Value> {
    let mut t = try_vec(text.len())?;
    t.extend_from_slice(text);
    Ok(ext(KdbError { text: t }))
}

/// A lambda as a value.
pub fn lambda_value(context: &[u8], text: &[u8]) -> Result<Value> {
    let mut c = try_vec(context.len())?;
    c.extend_from_slice(context);
    let mut t = try_vec(text.len())?;
    t.extend_from_slice(text);
    Ok(ext(KdbLambda { context: c, text: t }))
}

// ---------------------------------------------------------------------------------------
// encoder
// ---------------------------------------------------------------------------------------

/// The source text of a function literal (`{...}` as written), if the function is one.
/// A projection, an operator, a system function or a function defined with a parameter list
/// has none and goes out in its display form (`:monad`).
pub fn function_text(f: &Func) -> Option<&str> {
    if let Func::Lazy(l) = f {
        // (kept as its text: the text is the answer)
        if l.has_params {
            return None;
        }
        let t = l.source.text.get(l.start as usize..l.end as usize)?;
        return if t.starts_with('{') && t.ends_with('}') { Some(t) } else { None };
    }
    if let Func::Code { tree, node } = f {
        if let Node::Fn { a, has_args: false, call: false, .. } = tree.node(*node) {
            if Tree::is_real(*a) {
                if let Node::Body { has_params: true, .. } = tree.node(*a) {
                    return None;
                }
            }
            let (s, e) = tree.span(*node)?;
            let src = tree.source.as_ref()?;
            let t = src.text.get(s as usize..e as usize)?;
            if t.starts_with('{') && t.ends_with('}') {
                return Some(t);
            }
        }
    }
    None
}

/// `:nilad :monad :dyad :triad` or `:fn/N`.
fn display_form(arity: u8, out: &mut Vec<u8>) -> Result<()> {
    let s: &[u8] = match arity {
        0 => b":nilad",
        1 => b":monad",
        2 => b":dyad",
        3 => b":triad",
        n => {
            put(out, b":fn/")?;
            let mut buf = [0u8; 3];
            let mut i = 3;
            let mut n = n;
            while n > 0 {
                i -= 1;
                buf[i] = b'0' + n % 10;
                n /= 10;
            }
            return put(out, &buf[i..]);
        }
    };
    put(out, s)
}

fn put(out: &mut Vec<u8>, b: &[u8]) -> Result<()> {
    out.try_reserve(b.len()).map_err(|_| Error::memory())?;
    out.extend_from_slice(b);
    Ok(())
}

fn too_deep() -> Error {
    Error::range("nesting deeper than 64")
}

struct Encoder<'a> {
    syms: &'a Interner,
    out: &'a mut Vec<u8>,
}

impl Encoder<'_> {
    fn put(&mut self, b: &[u8]) -> Result<()> {
        put(self.out, b)
    }

    /// Type byte, attribute byte, count.
    fn head(&mut self, t: u8, n: usize) -> Result<()> {
        if n > i32::MAX as usize {
            return Err(Error::range("too many elements for one kdb+ vector"));
        }
        let c = (n as u32).to_le_bytes();
        self.put(&[t, 0, c[0], c[1], c[2], c[3]])
    }

    fn charvec(&mut self, b: &[u8]) -> Result<()> {
        self.head(10, b.len())?;
        self.put(b)
    }

    fn sym(&mut self, s: klong_core::SymId) -> Result<()> {
        let name = self.syms.name(s).as_bytes();
        if name.contains(&0) {
            return Err(Error::typ("cannot send a symbol containing NUL"));
        }
        self.put(name)?;
        self.put(&[0])
    }

    fn int_atom(&mut self, i: Int) -> Result<()> {
        self.put(&[if INT_IS_32 { 0xfa } else { 0xf9 }])?;
        self.put(&i.to_le_bytes())
    }

    fn flt_atom(&mut self, f: Flt) -> Result<()> {
        self.put(&[if FLT_IS_32 { 0xf8 } else { 0xf7 }])?;
        self.put(&f.to_le_bytes())
    }

    fn ints(&mut self, data: &[Int]) -> Result<()> {
        self.head(if INT_IS_32 { 6 } else { 7 }, data.len())?;
        self.out.try_reserve(core::mem::size_of_val(data)).map_err(|_| Error::memory())?;
        for i in data {
            self.out.extend_from_slice(&i.to_le_bytes());
        }
        Ok(())
    }

    fn flts(&mut self, data: &[Flt]) -> Result<()> {
        self.head(if FLT_IS_32 { 8 } else { 9 }, data.len())?;
        self.out.try_reserve(core::mem::size_of_val(data)).map_err(|_| Error::memory())?;
        for f in data {
            self.out.extend_from_slice(&f.to_le_bytes());
        }
        Ok(())
    }

    /// An array: a typed vector for rank 1, a general list of its rows above.
    fn array<T: Copy>(&mut self, data: &[T], dims: &[usize], depth: usize, leaf: &dyn Fn(&mut Self, &[T]) -> Result<()>) -> Result<()> {
        if dims.len() <= 1 {
            if data.is_empty() {
                // the empty list: an empty general list, as for Klong's []
                return self.head(0, 0);
            }
            return leaf(self, data);
        }
        if depth >= MAX_DEPTH {
            return Err(too_deep());
        }
        let rows = dims[0];
        let row: usize = dims[1..].iter().product();
        self.head(0, rows)?;
        for r in 0..rows {
            let part = data.get(r * row..(r + 1) * row).ok_or_else(|| Error::internal("array shape"))?;
            self.array(part, &dims[1..], depth + 1, leaf)?;
        }
        Ok(())
    }

    fn arr<T: Copy>(&mut self, a: &Arr<T>, depth: usize, leaf: &dyn Fn(&mut Self, &[T]) -> Result<()>) -> Result<()> {
        let mut dims = [0usize; klong_core::value::MAX_RANK];
        let rank = a.rank().clamp(1, klong_core::value::MAX_RANK);
        for (i, d) in dims.iter_mut().enumerate().take(rank) {
            *d = a.dim(i);
        }
        self.array(&a.data, &dims[..rank], depth, leaf)
    }

    fn general(&mut self, items: &[Value], depth: usize) -> Result<()> {
        if depth >= MAX_DEPTH {
            return Err(too_deep());
        }
        self.head(0, items.len())?;
        for q in items {
            self.obj(q, depth + 1)?;
        }
        Ok(())
    }

    /// The keys or the values of a dictionary, or the items of a list: a typed vector when
    /// they are all integers, all floats or all symbols, else a general list.
    fn seq(&mut self, items: &[Value], depth: usize) -> Result<()> {
        if !items.is_empty() {
            if items.iter().all(|q| matches!(q, Value::Sym(_))) {
                self.head(11, items.len())?;
                for q in items {
                    if let Value::Sym(s) = q {
                        self.sym(*s)?;
                    }
                }
                return Ok(());
            }
            if items.iter().all(|q| matches!(q, Value::Int(_))) {
                let mut v: Vec<Int> = try_vec(items.len())?;
                v.extend(items.iter().filter_map(|q| q.as_int()));
                return self.ints(&v);
            }
            if items.iter().all(|q| matches!(q, Value::Flt(_))) {
                let mut v: Vec<Flt> = try_vec(items.len())?;
                v.extend(items.iter().filter_map(|q| q.as_flt()));
                return self.flts(&v);
            }
        }
        self.general(items, depth)
    }

    fn lambda(&mut self, ctx: &[u8], text: &[u8]) -> Result<()> {
        if ctx.contains(&0) {
            return Err(Error::typ("lambda context contains NUL"));
        }
        self.put(&[0x64])?;
        self.put(ctx)?;
        self.put(&[0])?;
        self.charvec(text)
    }

    fn function(&mut self, text: Option<&str>, arity: u8) -> Result<()> {
        match text {
            Some(t) => self.lambda(b"kg", t.as_bytes()),
            None => {
                let mut d = Vec::new();
                display_form(arity, &mut d)?;
                self.lambda(b"kg", &d)
            }
        }
    }

    fn obj(&mut self, v: &Value, depth: usize) -> Result<()> {
        match v {
            Value::Int(i) => self.int_atom(*i),
            Value::Flt(f) => self.flt_atom(*f),
            Value::Char(c) => {
                if (*c as u32) < 0x80 {
                    self.put(&[0xf6, *c as u8])
                } else {
                    // any other char goes as a char vector holding its UTF-8 bytes
                    let mut b = [0u8; 4];
                    self.charvec(c.encode_utf8(&mut b).as_bytes())
                }
            }
            Value::Sym(s) => {
                self.put(&[0xf5])?;
                self.sym(*s)
            }
            Value::Undefined => self.put(&GENERIC_NULL),
            Value::Str(s) => self.charvec(s.as_str().as_bytes()),
            Value::Ints(a) => self.arr(a, depth, &|e, d| e.ints(d)),
            Value::Flts(a) => self.arr(a, depth, &|e, d| e.flts(d)),
            Value::List(l) => self.seq(l, depth),
            Value::Dict(d) => {
                if depth >= MAX_DEPTH {
                    return Err(too_deep());
                }
                let d = d.try_borrow().map_err(|_| Error::internal("dictionary in use"))?;
                self.put(&[0x63])?;
                self.seq(d.keys(), depth + 1)?;
                self.seq(d.vals(), depth + 1)
            }
            Value::Fn(f) => self.function(function_text(f), klong_core::fmt::func_arity(f)),
            Value::Prim(_) => self.function(None, 2),
            Value::Ext(e) => {
                let any = e.0.as_any();
                if let Some(l) = any.downcast_ref::<KdbLambda>() {
                    self.lambda(&l.context, &l.text)
                } else if let Some(o) = any.downcast_ref::<KdbOpaque>() {
                    self.put(&o.raw)
                } else if let Some(err) = any.downcast_ref::<KdbError>() {
                    if err.text.contains(&0) {
                        return Err(Error::typ("error text contains NUL"));
                    }
                    self.put(&[0x80])?;
                    self.put(&err.text)?;
                    self.put(&[0])
                } else {
                    Err(Error::typ(klong_core::msg!("cannot send a value of kind {}", e.0.type_name())))
                }
            }
        }
    }
}

/// Append the kdb+ object for `v` to `out`. A `type` error for a value with no kdb+ form, a
/// `range` error for one that nests deeper than [`MAX_DEPTH`], `memory` when `out` cannot grow
/// (`out` then holds a partial object: truncate it).
pub fn encode(syms: &Interner, v: &Value, out: &mut Vec<u8>) -> Result<()> {
    Encoder { syms, out }.obj(v, 0)
}

/// Append a call envelope: always a general list (type 0), even when every item is a symbol,
/// so that `h(:f,:x)` cannot be mistaken for data.
pub fn encode_call(syms: &Interner, items: &[Value], out: &mut Vec<u8>) -> Result<()> {
    Encoder { syms, out }.general(items, 0)
}

// ---------------------------------------------------------------------------------------
// decoder
// ---------------------------------------------------------------------------------------

pub enum DecodeError {
    /// Not a kdb+ object. Fatal for the connection (kind `system`).
    Protocol(String),
    /// A well-formed object this build cannot hold. Fails the message only.
    Value(Error),
}

impl From<Error> for DecodeError {
    fn from(e: Error) -> DecodeError {
        DecodeError::Value(e)
    }
}

impl core::fmt::Debug for DecodeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            DecodeError::Protocol(s) => write!(f, "protocol error: {}", s),
            DecodeError::Value(e) => write!(f, "{:?}", e),
        }
    }
}

type DResult<T> = core::result::Result<T, DecodeError>;

fn perr<T>(s: &'static str) -> DResult<T> {
    Err(DecodeError::Protocol(String::from(s)))
}

const TRUNC: &str = "message truncated";

/// Bytes of one element of a vector of type `t` (1 for a symbol: its terminator).
fn elem_size(t: u8) -> Option<usize> {
    Some(match t {
        1 | 4 | 10 | 11 => 1,
        2 => 16,
        5 => 2,
        6 | 8 | 13 | 14 | 17 | 18 | 19 => 4,
        7 | 9 | 12 | 15 | 16 => 8,
        _ => return None,
    })
}

/// Text from the wire: UTF-8, invalid sequences replaced by U+FFFD (as KlongPy does).
fn text(b: &[u8]) -> Result<String> {
    let mut out = try_string(b.len())?;
    let mut rest = b;
    loop {
        match core::str::from_utf8(rest) {
            Ok(s) => {
                out.push_str(s);
                return Ok(out);
            }
            Err(e) => {
                let (good, bad) = rest.split_at(e.valid_up_to());
                // (valid by construction; the fallback keeps this free of panics)
                out.push_str(core::str::from_utf8(good).unwrap_or(""));
                out.try_reserve(3 + bad.len()).map_err(|_| Error::memory())?;
                out.push('\u{fffd}');
                let skip = e.error_len().unwrap_or(bad.len()).max(1).min(bad.len());
                rest = &bad[skip..];
            }
        }
    }
}

struct Decoder<'a> {
    buf: &'a [u8],
    pos: usize,
    little: bool,
    syms: &'a mut Interner,
}

macro_rules! rd {
    ($name:ident, $t:ty, $n:expr) => {
        fn $name(&mut self) -> DResult<$t> {
            let b = self.take($n)?;
            let mut a = [0u8; $n];
            a.copy_from_slice(b);
            Ok(if self.little { <$t>::from_le_bytes(a) } else { <$t>::from_be_bytes(a) })
        }
    };
}

impl<'a> Decoder<'a> {
    fn left(&self) -> usize {
        self.buf.len() - self.pos
    }

    fn byte(&mut self) -> DResult<u8> {
        match self.buf.get(self.pos) {
            Some(&b) => {
                self.pos += 1;
                Ok(b)
            }
            None => perr(TRUNC),
        }
    }

    fn peek(&self) -> DResult<u8> {
        match self.buf.get(self.pos) {
            Some(&b) => Ok(b),
            None => perr(TRUNC),
        }
    }

    fn take(&mut self, k: usize) -> DResult<&'a [u8]> {
        if k > self.left() {
            return perr(TRUNC);
        }
        let chunk = &self.buf[self.pos..self.pos + k];
        self.pos += k;
        Ok(chunk)
    }

    rd!(i16, i16, 2);
    rd!(i32, i32, 4);
    rd!(i64, i64, 8);
    rd!(u32, u32, 4);
    rd!(u64, u64, 8);

    /// A count, checked against the bytes that are left: `count * min_elem` must fit.
    fn count(&mut self, min_elem: usize) -> DResult<usize> {
        let c = self.i32()?;
        if c < 0 {
            return perr("negative vector count");
        }
        let c = c as usize;
        match c.checked_mul(min_elem) {
            Some(n) if n <= self.left() => Ok(c),
            _ => perr("vector count exceeds the message"),
        }
    }

    /// Bytes up to the next NUL, which is consumed.
    fn cstring(&mut self) -> DResult<&'a [u8]> {
        let rest = &self.buf[self.pos..];
        match rest.iter().position(|&b| b == 0) {
            Some(i) => {
                self.pos += i + 1;
                Ok(&rest[..i])
            }
            None => perr("unterminated string"),
        }
    }

    fn symbol(&mut self) -> DResult<Value> {
        let raw = self.cstring()?;
        let s = match core::str::from_utf8(raw) {
            Ok(t) => self.syms.intern(t),
            Err(_) => {
                let t = text(raw)?;
                self.syms.intern(&t)
            }
        };
        if let Some(e) = self.syms.failure() {
            return Err(DecodeError::Value(e));
        }
        Ok(Value::Sym(s))
    }

    fn fit(v: i64) -> DResult<Int> {
        Int::try_from(v).map_err(|_| DecodeError::Value(Error::range(klong_core::msg!("integer {} does not fit 32 bits", v))))
    }

    fn flt32(&mut self) -> DResult<Flt> {
        Ok(f32::from_bits(self.u32()?) as Flt)
    }

    fn flt64(&mut self) -> DResult<Flt> {
        Ok(f64::from_bits(self.u64()?) as Flt)
    }

    fn obj(&mut self, depth: usize) -> DResult<Value> {
        let t = self.byte()? as i8;
        if t < 0 {
            return self.atom(t.unsigned_abs());
        }
        let t = t as u8;
        match t {
            0 => {
                let items = self.general_items(depth)?;
                Ok(Value::from_items(items)?)
            }
            1..=19 => self.vector(t),
            98 => self.table(depth),
            99 | 127 => self.dict(depth),
            100 => self.lambda(),
            101..=111 => {
                if t == 101 && self.buf.get(self.pos) == Some(&0) {
                    self.pos += 1;
                    return Ok(Value::Undefined);
                }
                self.pos -= 1;
                let mut raw = Vec::new();
                self.copy(&mut raw, depth)?;
                Ok(ext(KdbOpaque { raw }))
            }
            _ => perr("unsupported type byte"),
        }
    }

    fn atom(&mut self, t: u8) -> DResult<Value> {
        Ok(match t {
            128 => {
                let raw = self.cstring()?;
                error_value(raw)?
            }
            11 => self.symbol()?,
            10 => {
                let b = self.byte()?;
                Value::Char(if b < 0x80 { b as char } else { '\u{fffd}' })
            }
            2 => {
                let raw = self.take(16)?;
                let mut v: Vec<Int> = try_vec(16)?;
                v.extend(raw.iter().map(|&b| b as Int));
                Value::ints(v)
            }
            1 | 4 => Value::Int(self.byte()? as Int),
            5 => Value::Int(self.i16()? as Int),
            6 | 13 | 14 | 17 | 18 | 19 => Value::Int(self.i32()? as Int),
            7 | 12 | 16 => Value::Int(Self::fit(self.i64()?)?),
            8 => Value::Flt(self.flt32()?),
            9 | 15 => Value::Flt(self.flt64()?),
            _ => return perr("unsupported atom type"),
        })
    }

    fn vector(&mut self, t: u8) -> DResult<Value> {
        self.byte()?; // attribute: ignored
        let size = match elem_size(t) {
            Some(s) => s,
            None => return perr("unsupported vector type"),
        };
        let n = self.count(size)?;
        match t {
            11 => {
                let mut items = try_vec(n)?;
                for _ in 0..n {
                    items.push(self.symbol()?);
                }
                if items.is_empty() {
                    return Ok(Value::empty());
                }
                Ok(Value::raw_list(items)?)
            }
            10 => Ok(Value::str(text(self.take(n)?)?)),
            2 => {
                let raw = self.take(16 * n)?;
                let mut v: Vec<Int> = try_vec(16 * n)?;
                v.extend(raw.iter().map(|&b| b as Int));
                if n == 0 {
                    return Ok(Value::empty());
                }
                Ok(Value::Ints(Rc::new(Arr::shaped(v, &[n as u32, 16]))))
            }
            8 | 9 | 15 => {
                let raw = self.take(n * size)?;
                let mut v: Vec<Flt> = try_vec(n)?;
                let le = self.little;
                if size == 4 {
                    v.extend(raw.chunks_exact(4).map(|c| {
                        let a = [c[0], c[1], c[2], c[3]];
                        (if le { f32::from_le_bytes(a) } else { f32::from_be_bytes(a) }) as Flt
                    }));
                } else {
                    v.extend(raw.chunks_exact(8).map(|c| {
                        let a = [c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]];
                        (if le { f64::from_le_bytes(a) } else { f64::from_be_bytes(a) }) as Flt
                    }));
                }
                Ok(Value::flts(v))
            }
            _ => {
                let raw = self.take(n * size)?;
                let mut v: Vec<Int> = try_vec(n)?;
                let le = self.little;
                match size {
                    1 => v.extend(raw.iter().map(|&b| b as Int)),
                    2 => v.extend(raw.chunks_exact(2).map(|c| {
                        let a = [c[0], c[1]];
                        (if le { i16::from_le_bytes(a) } else { i16::from_be_bytes(a) }) as Int
                    })),
                    4 => v.extend(raw.chunks_exact(4).map(|c| {
                        let a = [c[0], c[1], c[2], c[3]];
                        (if le { i32::from_le_bytes(a) } else { i32::from_be_bytes(a) }) as Int
                    })),
                    _ => {
                        for c in raw.chunks_exact(8) {
                            let a = [c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]];
                            let x = if le { i64::from_le_bytes(a) } else { i64::from_be_bytes(a) };
                            if INT_IS_32 {
                                v.push(Int::try_from(x).map_err(|_| DecodeError::Value(Error::range("integer vector does not fit 32 bits")))?);
                            } else {
                                v.push(x as Int);
                            }
                        }
                    }
                }
                Ok(Value::ints(v))
            }
        }
    }

    /// The items of a general list whose type byte has been read.
    fn general_items(&mut self, depth: usize) -> DResult<Vec<Value>> {
        if depth >= MAX_DEPTH {
            return perr("nesting deeper than 64");
        }
        self.byte()?; // attribute
        let n = self.count(2)?; // every object is at least 2 bytes
        let mut items = try_vec(n)?;
        for _ in 0..n {
            items.push(self.obj(depth + 1)?);
        }
        Ok(items)
    }

    /// The next object, which must be a list (type 0 to 19), as its items.
    fn seq_items(&mut self, depth: usize) -> DResult<Vec<Value>> {
        let t = self.peek()?;
        if t == 0 {
            self.pos += 1;
            return self.general_items(depth);
        }
        if (1..=19).contains(&t) {
            self.pos += 1;
            let v = self.vector(t)?;
            return Ok(v.items()?);
        }
        perr("dictionary part is not a list")
    }

    fn dict(&mut self, depth: usize) -> DResult<Value> {
        if depth >= MAX_DEPTH {
            return perr("nesting deeper than 64");
        }
        if self.peek()? == 98 {
            // a keyed table: table!table, as the pair of them
            self.pos += 1;
            let keys = self.table(depth + 1)?;
            if self.byte()? != 98 {
                return perr("keyed table values are not a table");
            }
            let vals = self.table(depth + 1)?;
            let mut pair = try_vec(2)?;
            pair.push(keys);
            pair.push(vals);
            return Ok(Value::raw_list(pair)?);
        }
        let keys = self.seq_items(depth + 1)?;
        let vals = self.seq_items(depth + 1)?;
        if keys.len() != vals.len() {
            return perr("dictionary keys and values differ in length");
        }
        let mut d = Dict::default();
        for (k, v) in keys.into_iter().zip(vals) {
            if matches!(k, Value::Dict(_)) {
                return perr("dictionary used as a dictionary key");
            }
            d.set(k, v)?;
        }
        Ok(Value::dict(d))
    }

    /// A table whose type byte has been read: a dictionary from column name to column.
    fn table(&mut self, depth: usize) -> DResult<Value> {
        if depth >= MAX_DEPTH {
            return perr("nesting deeper than 64");
        }
        self.byte()?; // attribute
        if !matches!(self.byte()?, 99 | 127) {
            return perr("table without its dictionary");
        }
        if self.byte()? != 11 {
            return perr("table column names are not a symbol vector");
        }
        let names = self.vector(11)?.items()?;
        if self.byte()? != 0 {
            return perr("table columns are not a general list");
        }
        let cols = self.general_items(depth + 1)?;
        if cols.len() != names.len() {
            return perr("table names and columns differ in number");
        }
        let mut rows = None;
        for c in &cols {
            let n = match c.len() {
                Some(n) => n,
                None => return perr("table column is not a list"),
            };
            if *rows.get_or_insert(n) != n {
                return perr("table columns differ in length");
            }
        }
        let mut d = Dict::default();
        for (k, v) in names.into_iter().zip(cols) {
            d.set(k, v)?;
        }
        Ok(Value::dict(d))
    }

    fn lambda(&mut self) -> DResult<Value> {
        let ctx = self.cstring()?;
        if self.byte()? != 10 {
            return perr("lambda text is not a char vector");
        }
        self.byte()?; // attribute
        let n = self.count(1)?;
        let txt = self.take(n)?;
        Ok(lambda_value(ctx, txt)?)
    }

    // ----- structural copy (opaque function objects) -------------------------------------

    fn copy_raw(&mut self, out: &mut Vec<u8>, n: usize) -> DResult<()> {
        let b = self.take(n)?;
        Ok(put(out, b)?)
    }

    /// Copy `n` elements of `width` bytes, little-endian in `out`.
    fn copy_swapped(&mut self, out: &mut Vec<u8>, n: usize, width: usize) -> DResult<()> {
        let b = self.take(n * width)?;
        out.try_reserve(b.len()).map_err(|_| Error::memory())?;
        if self.little || width == 1 {
            out.extend_from_slice(b);
        } else {
            for c in b.chunks_exact(width) {
                out.extend(c.iter().rev());
            }
        }
        Ok(())
    }

    fn copy_cstring(&mut self, out: &mut Vec<u8>) -> DResult<()> {
        let s = self.cstring()?;
        put(out, s)?;
        Ok(put(out, &[0])?)
    }

    fn copy_count(&mut self, out: &mut Vec<u8>, min_elem: usize) -> DResult<usize> {
        let n = self.count(min_elem)?;
        put(out, &(n as u32).to_le_bytes())?;
        Ok(n)
    }

    /// Validate one object and append its little-endian bytes to `out`.
    fn copy(&mut self, out: &mut Vec<u8>, depth: usize) -> DResult<()> {
        let tb = self.byte()?;
        put(out, &[tb])?;
        let t = tb as i8;
        if t < 0 {
            let t = t.unsigned_abs();
            return match t {
                128 | 11 => self.copy_cstring(out),
                2 => self.copy_raw(out, 16),
                10 => self.copy_raw(out, 1),
                1 | 4..=9 | 12..=19 => {
                    let w = elem_size(t).unwrap_or(1);
                    self.copy_swapped(out, 1, w)
                }
                _ => perr("unsupported atom type"),
            };
        }
        let t = t as u8;
        if t <= 19 {
            if t == 0 && depth >= MAX_DEPTH {
                return perr("nesting deeper than 64");
            }
            self.copy_raw(out, 1)?; // attribute
            return match t {
                0 => {
                    let n = self.copy_count(out, 2)?;
                    for _ in 0..n {
                        self.copy(out, depth + 1)?;
                    }
                    Ok(())
                }
                11 => {
                    let n = self.copy_count(out, 1)?;
                    for _ in 0..n {
                        self.copy_cstring(out)?;
                    }
                    Ok(())
                }
                _ => match elem_size(t) {
                    Some(size) => {
                        let n = self.copy_count(out, size)?;
                        if matches!(t, 1 | 2 | 4 | 10) {
                            self.copy_raw(out, n * size)
                        } else {
                            self.copy_swapped(out, n, size)
                        }
                    }
                    None => perr("unsupported vector type"),
                },
            };
        }
        if depth >= MAX_DEPTH {
            return perr("nesting deeper than 64");
        }
        match t {
            98 => {
                self.copy_raw(out, 1)?;
                self.copy(out, depth + 1)
            }
            99 | 127 => {
                self.copy(out, depth + 1)?;
                self.copy(out, depth + 1)
            }
            100 => {
                self.copy_cstring(out)?;
                self.copy(out, depth + 1)
            }
            101..=103 => self.copy_raw(out, 1),
            104 | 105 => {
                let n = self.copy_count(out, 2)?;
                for _ in 0..n {
                    self.copy(out, depth + 1)?;
                }
                Ok(())
            }
            106..=111 => self.copy(out, depth + 1),
            _ => perr("unsupported type byte"),
        }
    }
}

/// Decode one object: the bytes after the 8-byte message header. `little` is header byte 0.
/// The whole of `data` must be consumed. Symbols are interned in `syms`.
pub fn decode(data: &[u8], little: bool, syms: &mut Interner) -> DResult<Value> {
    let mut d = Decoder { buf: data, pos: 0, little, syms };
    let v = d.obj(0)?;
    if d.pos != d.buf.len() {
        return perr("bytes after the object");
    }
    Ok(v)
}

/// `try_push` for callers of this crate that build argument lists.
pub fn push(v: &mut Vec<Value>, x: Value) -> Result<()> {
    try_push(v, x)
}
