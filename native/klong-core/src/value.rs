//! Values (docs/native/index.md).
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
use alloc::string::String;
use alloc::vec::Vec;
use core::cell::{Cell, RefCell};
use core::sync::atomic::{AtomicU32, AtomicU8, Ordering};

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

// ---------------------------------------------------------------------------------------
// Limits and deferred failures
// ---------------------------------------------------------------------------------------

/// How deep lists (and projections) may nest. Every walk over data (drop, print, match, shape,
/// the atomic pairing) recurses once per level on the native stack, so the constructors refuse
/// to build anything deeper: a `range` error instead of a stack overflow. Set from
/// `Config::max_nest` when an interpreter is created (one value for the process).
static MAX_NEST: AtomicU32 = AtomicU32::new(DEFAULT_MAX_NEST);
pub const DEFAULT_MAX_NEST: u32 = 128;

pub fn set_max_nest(n: u32) {
    MAX_NEST.store(n.clamp(4, 4096), Ordering::Relaxed);
}
pub fn max_nest() -> u32 {
    MAX_NEST.load(Ordering::Relaxed)
}

/// A failure met where no `Result` can be returned (an allocation inside `Value::item`, a
/// walk that met data nested too deeply). The place leaves a harmless value behind and records
/// the failure here; the evaluator raises it before the value of the expression is used
/// (`Interp::eval` asks after every node). Load and store only: no compare-and-swap.
static PENDING: AtomicU8 = AtomicU8::new(0);
const PENDING_MEMORY: u8 = 1;
const PENDING_DEPTH: u8 = 2;

pub fn defer_memory() {
    PENDING.store(PENDING_MEMORY, Ordering::Relaxed);
}
pub fn defer_depth() {
    if PENDING.load(Ordering::Relaxed) == 0 {
        PENDING.store(PENDING_DEPTH, Ordering::Relaxed);
    }
}
/// The deferred failure, if there is one; it is cleared.
#[inline]
pub fn take_deferred() -> Option<Error> {
    match PENDING.load(Ordering::Relaxed) {
        0 => None,
        k => {
            PENDING.store(0, Ordering::Relaxed);
            Some(if k == PENDING_MEMORY { Error::memory() } else { too_deep() })
        }
    }
}

pub fn too_deep() -> Error {
    Error::range("data nested too deeply")
}

// ---------------------------------------------------------------------------------------
// Strings
// ---------------------------------------------------------------------------------------

/// A string: UTF-8 text whose elements are code points, as in KlongPy (`#` counts code points,
/// `@` indexes them). One byte per character for ASCII text, which is indexed directly; other
/// text is indexed by walking from the last position asked for, so going through a string in
/// order (in either direction) costs one step per element.
pub struct Text {
    s: String,
    /// Number of code points.
    n: u32,
    /// A position already located: (code point index, byte offset).
    at: Cell<(u32, u32)>,
}

impl Text {
    pub fn new(s: String) -> Text {
        let n = if s.is_ascii() { s.len() } else { s.chars().count() };
        Text { s, n: n as u32, at: Cell::new((0, 0)) }
    }
    pub fn as_str(&self) -> &str {
        &self.s
    }
    /// Length in code points.
    pub fn len(&self) -> usize {
        self.n as usize
    }
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }
    pub fn is_ascii(&self) -> bool {
        self.n as usize == self.s.len()
    }
    /// Byte offset of code point `i` (the length in bytes for `i` at or past the end).
    pub fn offset(&self, i: usize) -> usize {
        if self.is_ascii() {
            return i.min(self.s.len());
        }
        if i >= self.n as usize {
            return self.s.len();
        }
        let b = self.s.as_bytes();
        let (ci, bi) = self.at.get();
        let (mut c, mut o) = (ci as usize, bi as usize);
        if i < c && i < c - i {
            c = 0;
            o = 0;
        }
        while c < i {
            o += 1;
            while o < b.len() && b[o] & 0xC0 == 0x80 {
                o += 1;
            }
            c += 1;
        }
        while c > i {
            o -= 1;
            while o > 0 && b[o] & 0xC0 == 0x80 {
                o -= 1;
            }
            c -= 1;
        }
        self.at.set((c as u32, o as u32));
        o
    }
    pub fn char_at(&self, i: usize) -> char {
        if self.is_ascii() {
            return self.s.as_bytes().get(i).map(|&b| b as char).unwrap_or('\0');
        }
        let o = self.offset(i);
        self.s[o..].chars().next().unwrap_or('\0')
    }
    /// Code points `lo..hi` (clamped).
    pub fn slice(&self, lo: usize, hi: usize) -> &str {
        let hi = hi.min(self.n as usize);
        let lo = lo.min(hi);
        let a = self.offset(lo);
        let b = self.offset(hi);
        &self.s[a..b]
    }
}

/// A `String` with room for `n` bytes, or a `memory` error.
pub fn try_string(n: usize) -> Result<String> {
    let mut s = String::new();
    let _tag = crate::htag::Scope::new(crate::htag::STRING);
    s.try_reserve_exact(n).map_err(|_| Error::memory())?;
    Ok(s)
}

/// A copy of `s`, or a `memory` error.
pub fn try_str(s: &str) -> Result<String> {
    let mut out = try_string(s.len())?;
    out.push_str(s);
    Ok(out)
}

pub fn try_push_str(out: &mut String, s: &str) -> Result<()> {
    let _tag = crate::htag::Scope::new(crate::htag::STRING);
    out.try_reserve(s.len()).map_err(|_| Error::memory())?;
    out.push_str(s);
    Ok(())
}

pub fn try_push_char(out: &mut String, c: char) -> Result<()> {
    out.try_reserve(4).map_err(|_| Error::memory())?;
    out.push(c);
    Ok(())
}

// ---------------------------------------------------------------------------------------
// Lists
// ---------------------------------------------------------------------------------------

/// A mixed or ragged list. `depth` is how deep it nests (1 for a list of atoms; dictionaries
/// count as atoms) and is what bounds every recursive walk; `has_dict` says a dictionary is
/// somewhere inside (those are taken apart without recursion when dropped, see `Dict`).
pub struct List {
    items: Vec<Value>,
    depth: u16,
    has_dict: bool,
}

impl core::ops::Deref for List {
    type Target = [Value];
    fn deref(&self) -> &[Value] {
        &self.items
    }
}

/// Depth and dictionary flag of a run of values that are about to share a container.
fn nest_of(items: &[Value]) -> (u16, bool) {
    let mut depth = 0u16;
    let mut has_dict = false;
    for q in items {
        match q {
            Value::List(l) => {
                depth = depth.max(l.depth);
                has_dict |= l.has_dict;
            }
            Value::Dict(_) => has_dict = true,
            Value::Fn(f) => {
                if let Func::Proj { depth: d, has_dict: h, .. } = &**f {
                    depth = depth.max(*d);
                    has_dict |= *h;
                }
            }
            _ => {}
        }
    }
    (depth, has_dict)
}

// ---------------------------------------------------------------------------------------
// Dictionaries
// ---------------------------------------------------------------------------------------

/// Insertion-ordered dictionary: the keys and values in the order they were added, plus a
/// hash index into them (open addressing, linear probing) once there are more than
/// `LINEAR` entries. Small dictionaries are searched directly and carry no index at all; the
/// index costs 2 bytes per slot up to 65534 entries and 4 above, at a load of one half at
/// most, so a large dictionary pays 4 to 8 bytes per entry on top of its two values.
/// Deleting a key moves the entries after it and rebuilds the index (order is kept).
#[derive(Default)]
pub struct Dict {
    keys: Vec<Value>,
    vals: Vec<Value>,
    index: Index,
    /// What `.import` returns is read-only from outside: setting or removing a key is a `type` error (design point 5). Shallow: a
    /// dictionary stored in it is an ordinary shared dictionary.
    ro: bool,
}

#[derive(Default)]
enum Index {
    #[default]
    None,
    Small(Vec<u16>),
    Large(Vec<u32>),
}

const LINEAR: usize = 8;

impl Index {
    fn slots(&self) -> usize {
        match self {
            Index::None => 0,
            Index::Small(v) => v.len(),
            Index::Large(v) => v.len(),
        }
    }
    /// Entry number stored in a slot, plus one; 0 for an empty slot.
    fn get(&self, slot: usize) -> usize {
        match self {
            Index::None => 0,
            Index::Small(v) => v[slot] as usize,
            Index::Large(v) => v[slot] as usize,
        }
    }
    fn put(&mut self, slot: usize, entry_plus_one: usize) {
        match self {
            Index::None => {}
            Index::Small(v) => v[slot] = entry_plus_one as u16,
            Index::Large(v) => v[slot] = entry_plus_one as u32,
        }
    }
}

fn mix(mut h: u32) -> u32 {
    h ^= h >> 16;
    h = h.wrapping_mul(0x85eb_ca6b);
    h ^= h >> 13;
    h = h.wrapping_mul(0xc2b2_ae35);
    h ^ (h >> 16)
}

fn hash_int(i: i64) -> u32 {
    mix(i as u32 ^ mix((i >> 32) as u32 ^ 0x9e37_79b9))
}

fn hash_bytes(b: &[u8]) -> u32 {
    // FNV-1a
    let mut h: u32 = 0x811c_9dc5;
    for &c in b {
        h = (h ^ c as u32).wrapping_mul(0x0100_0193);
    }
    mix(h)
}

/// The hash of a key, or `None` for a value that cannot be one (a list, a dictionary, a
/// function). Agrees with `key_eq`: 1 and 1.0 hash alike, and so do 0ca and "a".
pub fn key_hash(k: &Value) -> Option<u32> {
    Some(match k {
        Value::Int(i) => hash_int(*i as i64),
        Value::Flt(f) => {
            let f = *f as f64;
            if f == (f as i64) as f64 {
                hash_int(f as i64)
            } else {
                let b = f.to_bits();
                mix(b as u32 ^ mix((b >> 32) as u32))
            }
        }
        #[cfg(feature = "bigint")]
        Value::Big(b) => {
            let f = crate::big::to_f64(b);
            if crate::big::cmp_flt(b, f) == Some(core::cmp::Ordering::Equal) {
                let bits = f.to_bits();
                mix(bits as u32 ^ mix((bits >> 32) as u32))
            } else {
                hash_bytes(&b.to_signed_bytes_le())
            }
        }
        Value::Char(c) => {
            let mut b = [0u8; 4];
            hash_bytes(c.encode_utf8(&mut b).as_bytes())
        }
        Value::Str(s) => hash_bytes(s.as_str().as_bytes()),
        Value::Sym(s) => mix(s.0 ^ 0x5bd1_e995),
        Value::Undefined => 0x1234_5678,
        // a system function or a host callable is its own key, by identity
        Value::Fn(f) if matches!(&**f, Func::Native { .. }) => hash_bytes(&(Rc::as_ptr(f) as usize).to_le_bytes()),
        Value::Ext(e) => hash_bytes(&e.0.key_identity()?.to_le_bytes()),
        _ => return None,
    })
}

pub fn bad_key(k: &Value) -> Error {
    Error::typ(crate::msg!("a {} cannot be a dictionary key", k.describe())).with_hint("a key is a number, a string, a char or a symbol")
}

impl Dict {
    pub fn len(&self) -> usize {
        self.keys.len()
    }
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }
    pub fn keys(&self) -> &[Value] {
        &self.keys
    }
    pub fn is_readonly(&self) -> bool {
        self.ro
    }
    /// Freeze the dictionary (see `ro`).
    pub fn set_readonly(&mut self) {
        self.ro = true;
    }
    fn read_only_error(&self) -> Error {
        let e = Error::typ(crate::dmsg!("read-only dictionary", "the dictionary is read-only (it was returned by .import)"));
        #[cfg(feature = "diag")]
        let e = e.with_hint("copy it with  e:::{};e::e,m  or change the module's file");
        e
    }
    pub fn vals(&self) -> &[Value] {
        &self.vals
    }

    /// (entry of the key if present, slot where it is or would go).
    fn probe(&self, k: &Value, h: u32) -> (Option<usize>, usize) {
        let n = self.index.slots();
        if n == 0 {
            return (self.keys.iter().position(|q| key_eq(q, k)), 0);
        }
        let mask = n - 1;
        let mut slot = h as usize & mask;
        loop {
            match self.index.get(slot) {
                0 => return (None, slot),
                e => {
                    if key_eq(&self.keys[e - 1], k) {
                        return (Some(e - 1), slot);
                    }
                }
            }
            slot = (slot + 1) & mask;
        }
    }

    pub fn find(&self, k: &Value) -> Option<usize> {
        let h = key_hash(k)?;
        self.probe(k, h).0
    }
    pub fn get(&self, k: &Value) -> Option<&Value> {
        self.find(k).map(|i| &self.vals[i])
    }

    /// Build the index for `want` entries (none while the dictionary is small).
    fn rebuild(&mut self, want: usize) -> Result<()> {
        if want <= LINEAR {
            self.index = Index::None;
            return Ok(());
        }
        let slots = (want * 2).next_power_of_two();
        self.index = if want < u16::MAX as usize {
            let mut v: Vec<u16> = try_vec(slots)?;
            v.resize(slots, 0);
            Index::Small(v)
        } else {
            let mut v: Vec<u32> = try_vec(slots)?;
            v.resize(slots, 0);
            Index::Large(v)
        };
        let mask = slots - 1;
        for (e, k) in self.keys.iter().enumerate() {
            let mut slot = key_hash(k).unwrap_or(0) as usize & mask;
            while self.index.get(slot) != 0 {
                slot = (slot + 1) & mask;
            }
            self.index.put(slot, e + 1);
        }
        Ok(())
    }

    /// Set a key. A `type` error for a value that cannot be a key, `memory` when the
    /// dictionary cannot grow (it is unchanged then).
    pub fn set(&mut self, k: Value, v: Value) -> Result<()> {
        if self.ro {
            return Err(self.read_only_error());
        }
        let h = key_hash(&k).ok_or_else(|| bad_key(&k))?;
        self.set_hashed(k, v, h)
    }

    fn set_hashed(&mut self, k: Value, v: Value, h: u32) -> Result<()> {
        let (found, mut slot) = self.probe(&k, h);
        if let Some(i) = found {
            self.vals[i] = v;
            return Ok(());
        }
        let n = self.keys.len() + 1;
        self.keys.try_reserve(1).map_err(|_| Error::memory())?;
        self.vals.try_reserve(1).map_err(|_| Error::memory())?;
        let grow = n > LINEAR && (n * 2 > self.index.slots() || (n >= u16::MAX as usize && matches!(self.index, Index::Small(_))));
        if grow {
            // room for twice the entries before the next rebuild
            let old = core::mem::take(&mut self.index);
            if let Err(e) = self.rebuild((n * 2).max(16)) {
                self.index = old;
                return Err(e);
            }
            slot = self.probe(&k, h).1;
        }
        self.keys.push(k);
        self.vals.push(v);
        self.index.put(slot, n);
        Ok(())
    }

    /// Remove a key; the order of the others is kept.
    pub fn remove(&mut self, k: &Value) -> Result<bool> {
        if self.ro {
            return Err(self.read_only_error());
        }
        let i = match self.find(k) {
            Some(i) => i,
            None => return Ok(false),
        };
        self.keys.remove(i);
        self.vals.remove(i);
        let n = self.keys.len();
        if self.index.slots() != 0 {
            // rebuilding in place: the table that is there is large enough
            match &mut self.index {
                Index::Small(v) => v.iter_mut().for_each(|s| *s = 0),
                Index::Large(v) => v.iter_mut().for_each(|s| *s = 0),
                Index::None => {}
            }
            if n <= LINEAR {
                self.index = Index::None;
            } else {
                let mask = self.index.slots() - 1;
                for e in 0..n {
                    let mut slot = key_hash(&self.keys[e]).unwrap_or(0) as usize & mask;
                    while self.index.get(slot) != 0 {
                        slot = (slot + 1) & mask;
                    }
                    self.index.put(slot, e + 1);
                }
            }
        }
        Ok(true)
    }
}

/// True for a value that can hold a dictionary somewhere inside.
fn holds_dict(v: &Value) -> bool {
    match v {
        Value::Dict(_) => true,
        Value::List(l) => l.has_dict,
        Value::Fn(f) => matches!(&**f, Func::Proj { has_dict: true, .. }),
        _ => false,
    }
}

/// Dictionaries are the one container whose nesting cannot be bounded when it is built (they
/// are shared and mutable: a dictionary can be put inside a list that is put inside it, or a
/// chain of them linked one insertion at a time). So a dictionary is not dropped by recursion:
/// what it holds that can lead to another dictionary is taken apart with a work list.
impl Drop for Dict {
    fn drop(&mut self) {
        if !self.vals.iter().any(holds_dict) {
            return;
        }
        let mut work: Vec<Value> = Vec::new();
        let take = |work: &mut Vec<Value>, vals: &mut Vec<Value>| {
            for v in vals.drain(..) {
                // without room for the work list the value is dropped the plain way
                if holds_dict(&v) && work.try_reserve(1).is_ok() {
                    work.push(v);
                }
            }
        };
        take(&mut work, &mut self.vals);
        while let Some(v) = work.pop() {
            match v {
                Value::Dict(rc) => {
                    if let Some(cell) = Rc::into_inner(rc) {
                        let mut d = cell.into_inner();
                        take(&mut work, &mut d.vals);
                    }
                }
                Value::List(rc) => {
                    if let Some(mut l) = Rc::into_inner(rc) {
                        take(&mut work, &mut l.items);
                    }
                }
                Value::Fn(rc) => {
                    if let Some(Func::Proj { f, args, .. }) = Rc::into_inner(rc) {
                        let mut vals: Vec<Value> = Vec::new();
                        if vals.try_reserve(args.len() + 1).is_ok() {
                            vals.push(f);
                            for a in args {
                                if let Slot::Val(v) = a {
                                    vals.push(v);
                                }
                            }
                            take(&mut work, &mut vals);
                        }
                    }
                }
                _ => {}
            }
        }
    }
}

pub fn key_eq(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Int(x), Value::Int(y)) => x == y,
        (Value::Int(x), Value::Flt(y)) | (Value::Flt(y), Value::Int(x)) => *x as Flt == *y,
        (Value::Flt(x), Value::Flt(y)) => x == y,
        (Value::Char(x), Value::Char(y)) => x == y,
        (Value::Sym(x), Value::Sym(y)) => x == y,
        (Value::Str(x), Value::Str(y)) => x.as_str() == y.as_str(),
        (Value::Char(c), Value::Str(s)) | (Value::Str(s), Value::Char(c)) => s.len() == 1 && s.char_at(0) == *c,
        (Value::Undefined, Value::Undefined) => true,
        (Value::Fn(x), Value::Fn(y)) => Rc::ptr_eq(x, y) && matches!(&**x, Func::Native { .. }),
        (Value::Ext(x), Value::Ext(y)) => matches!((x.0.key_identity(), y.0.key_identity()), (Some(p), Some(q)) if p == q),
        #[cfg(feature = "bigint")]
        (Value::Big(x), Value::Big(y)) => x == y,
        #[cfg(feature = "bigint")]
        (Value::Big(x), Value::Flt(f)) | (Value::Flt(f), Value::Big(x)) => crate::big::cmp_flt(x, *f as f64) == Some(core::cmp::Ordering::Equal),
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
    /// A function literal defined by a load, kept as where its text is, not as a tree: the tree is parsed again, from that text, when the
    /// function is called, and let go at the end of the program (`Interp::materialize`). A library of 6 KB of source was 75 KB of trees.
    Lazy(Lazy),
    /// A function applied to fewer values than it takes (made at run time by an adverb or `@`).
    /// A projection made at run time (a call with fewer arguments than the function takes, or
    /// with an empty slot): the callee and one slot for each argument.
    /// `depth` and `has_dict` describe the values it holds, as for a `List`.
    Proj { f: Value, args: Vec<Slot>, depth: u16, has_dict: bool },
    /// A system function. `arity` is `WILDCARD` for one that takes any number of arguments.
    Native { name: SymId, arity: u8, f: NativeFn },
}

/// A function literal of a loaded program, by its text (see `Func::Lazy`): the text, where the literal lies in it, what parsing it needs
/// (`ctx`), and what is asked of a function without a body: its arity, whether it is a function of a body (`kgfn`) and whether it names
/// its parameters.
pub struct Lazy {
    pub source: Rc<crate::error::Source>,
    pub start: u32,
    pub end: u32,
    pub ctx: Rc<LazyCtx>,
    pub arity: u8,
    pub kgfn: bool,
    pub has_params: bool,
}

/// What a text needs to be parsed as it was: the module in force and the strict level.
pub struct LazyCtx {
    pub module: Option<String>,
    /// The module scope the text belongs to (0: the program's own; see `Interp::module_new`).
    pub mid: u16,
    pub strict: u8,
}

impl Func {
    /// A projection of `f` over `args`; a `range` error when it would nest too deeply (a
    /// projection that holds a projection that holds a projection, without end).
    pub fn proj(f: Value, args: Vec<Slot>) -> Result<Func> {
        let (mut depth, mut has_dict) = nest_of(core::slice::from_ref(&f));
        for a in &args {
            if let Slot::Val(v) = a {
                let (d, h) = nest_of(core::slice::from_ref(v));
                depth = depth.max(d);
                has_dict |= h;
            }
        }
        if depth as u32 >= max_nest() {
            return Err(too_deep());
        }
        Ok(Func::Proj { f, args, depth: depth + 1, has_dict })
    }
}

/// An object of the embedder: a channel, an IPC handle. `as_any` lets the embedder get its
/// own type back.
pub trait Ext {
    fn type_name(&self) -> &str;
    fn as_any(&self) -> &dyn core::any::Any;
    /// What makes this object a dictionary key: an identity (two values with the same one are the same key). `None`: not a key.
    fn key_identity(&self) -> Option<usize> {
        None
    }
    /// `x?y` with this object as `x` (a remote dictionary answers a lookup). `None`: the
    /// object is not something Find applies to.
    fn find(&self, _ip: &mut crate::interp::Interp, _y: &Value) -> Option<Result<Value>> {
        None
    }
    /// `x,y` with this object as `x` (a remote dictionary takes a `[key value]` pair).
    fn join(&self, _ip: &mut crate::interp::Interp, _y: &Value) -> Option<Result<Value>> {
        None
    }
    /// HOST VALUES (feature `host`): an object of the embedder that stands for an array the embedder computes with (a NumPy array, a
    /// torch tensor). Each method is the object's answer to what the evaluator asks of an operand; `None` means "not handled"
    /// and the evaluator then goes on as it does for any value.
    ///
    /// A primitive monad / dyad with this object among the operands (`glyph` is the operator's text).
    #[cfg(feature = "host")]
    fn host_monad(&self, _ip: &mut crate::interp::Interp, _glyph: &str, _x: &Value) -> Option<Result<Value>> {
        None
    }
    #[cfg(feature = "host")]
    fn host_dyad(&self, _ip: &mut crate::interp::Interp, _glyph: &str, _x: &Value, _y: &Value) -> Option<Result<Value>> {
        None
    }
    /// An adverb applied to this object: `glyph` the adverb, `verb` the function value it modifies, `op` the verb's operator text when
    /// the verb is a primitive, `x` the operand (this object) and `y` the second operand of a dyadic form.
    #[cfg(feature = "host")]
    fn host_adverb(&self, _ip: &mut crate::interp::Interp, _glyph: &str, _verb: &Value, _op: Option<&str>, _x: &Value, _y: Option<&Value>) -> Option<Result<Value>> {
        None
    }
    /// The same object as an ordinary native value (a copy), for what is not forwarded. `what` names the operation that asks (for the
    /// error the embedder raises when a copy would be wrong: a tensor that requires gradients).
    #[cfg(feature = "host")]
    fn host_native(&self, _what: &str) -> Option<Result<Value>> {
        None
    }
    /// The truth of this object as the condition of `:[`.
    #[cfg(feature = "host")]
    fn host_truthy(&self) -> Option<bool> {
        None
    }
    /// Whether `p` and `q` (one of them this object) are equal as the converging adverbs (`:~`, `\\~`) compare successive values.
    #[cfg(feature = "host")]
    fn host_converged(&self, _p: &Value, _q: &Value) -> Option<bool> {
        None
    }
    /// The elements of this object (along its first dimension) as a list value, whose elements may be host objects themselves: what
    /// destructuring `[a b]::x` takes apart.
    #[cfg(feature = "host")]
    fn host_items(&self) -> Option<Result<Value>> {
        None
    }
    /// The shape of this object when it is an array (a tensor): its dimensions, first to last; `None` for anything else.
    #[cfg(feature = "host")]
    fn host_shape(&self) -> Option<Vec<u32>> {
        None
    }
    /// The text of this object as `.p` and `$` show it.
    #[cfg(feature = "host")]
    fn host_format(&self, _display: bool) -> Option<alloc::string::String> {
        None
    }
    /// The list `items` (at least one of them a host object, this one) as one value: a stacked array when the embedder can.
    #[cfg(feature = "host")]
    fn host_collect(&self, _items: &[Value]) -> Option<Result<Value>> {
        None
    }
}
pub struct ExtBox(pub Box<dyn Ext>);

#[derive(Clone)]
pub enum Value {
    Int(Int),
    Flt(Flt),
    /// A character: one code point.
    Char(char),
    Sym(SymId),
    Undefined,
    Ints(Rc<Arr<Int>>),
    Flts(Rc<Arr<Flt>>),
    /// A string: UTF-8 text; its elements are code points.
    Str(Rc<Text>),
    /// Mixed or ragged.
    List(Rc<List>),
    Dict(Rc<RefCell<Dict>>),
    Fn(Rc<Func>),
    /// An operator as a value; the id is the interned glyph.
    Prim(SymId),
    Ext(Rc<ExtBox>),
    /// An integer beyond the build width (feature `bigint`); never one that fits `Int`.
    #[cfg(feature = "bigint")]
    Big(crate::big::Big),
}

fn try_copy<T: Copy>(xs: &[T]) -> Option<Vec<T>> {
    let mut v = Vec::new();
    if v.try_reserve_exact(xs.len()).is_err() {
        defer_memory();
        return None;
    }
    v.extend_from_slice(xs);
    Some(v)
}

/// A vector with room for `n` elements, or a `memory` error. Every allocation whose size
/// depends on data goes through here or through `try_reserve`.
pub fn try_vec<T>(n: usize) -> Result<Vec<T>> {
    let mut v = Vec::new();
    let _tag = crate::htag::Scope::new(crate::htag::VECTOR);
    v.try_reserve_exact(n).map_err(|_| Error::memory())?;
    Ok(v)
}

pub fn try_push<T>(v: &mut Vec<T>, x: T) -> Result<()> {
    if v.len() == v.capacity() {
        let _tag = crate::htag::Scope::new(crate::htag::VECTOR);
        v.try_reserve(1).map_err(|_| Error::memory())?;
    }
    v.push(x);
    Ok(())
}

pub fn try_extend<T: Copy>(v: &mut Vec<T>, xs: &[T]) -> Result<()> {
    let _tag = crate::htag::Scope::new(crate::htag::VECTOR);
    v.try_reserve(xs.len()).map_err(|_| Error::memory())?;
    v.extend_from_slice(xs);
    Ok(())
}

impl Value {
    /// A host object (a tensor, an array of the embedder) as an ordinary native value, a copy; any other value as it is. For a system
    /// function that works on native values: `what` names it for the embedder's refusal.
    pub fn unhost(&self, what: &str) -> Result<Value> {
        #[cfg(feature = "host")]
        if let Value::Ext(e) = self {
            if let Some(r) = e.0.host_native(what) {
                return r;
            }
        }
        let _ = what;
        Ok(self.clone())
    }

    /// The elements of a host array as a list value (its elements may be host objects); `None` for any other value.
    pub fn host_items(&self) -> Option<Result<Value>> {
        #[cfg(feature = "host")]
        if let Value::Ext(e) = self {
            return e.0.host_items();
        }
        None
    }

    pub fn ints(v: Vec<Int>) -> Value {
        Value::Ints(Rc::new(Arr::vec(v)))
    }
    pub fn flts(v: Vec<Flt>) -> Value {
        Value::Flts(Rc::new(Arr::vec(v)))
    }
    pub fn str(s: String) -> Value {
        Value::Str(Rc::new(Text::new(s)))
    }
    /// A string holding a copy of `s`.
    pub fn text(s: &str) -> Result<Value> {
        Ok(Value::str(try_str(s)?))
    }
    pub fn empty_str() -> Value {
        Value::str(String::new())
    }
    /// The one-character string of `c`.
    pub fn char_str(c: char) -> Value {
        let mut s = String::new();
        s.push(c);
        Value::str(s)
    }
    pub fn dict(d: Dict) -> Value {
        Value::Dict(Rc::new(RefCell::new(d)))
    }
    pub fn empty() -> Value {
        Value::ints(Vec::new())
    }
    pub fn bool(b: bool) -> Value {
        Value::Int(b as Int)
    }
    /// A list of exactly these elements, without normalising. Only for elements known not to
    /// be all numbers. A `range` error when the list would nest too deeply.
    pub fn raw_list(items: Vec<Value>) -> Result<Value> {
        let (depth, has_dict) = nest_of(&items);
        if depth as u32 >= max_nest() {
            return Err(too_deep());
        }
        Ok(Value::List(Rc::new(List { items, depth: depth + 1, has_dict })))
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
        #[cfg(feature = "host")]
        for q in &items {
            if let Value::Ext(e) = q {
                if let Some(r) = e.0.host_collect(&items) {
                    return r;
                }
                break;
            }
        }
        // A rectangular nest of numeric rows becomes one array of higher rank.
        let (rank, len, any_flt) = match &items[0] {
            Value::Ints(a) => (a.rank(), a.len(), false),
            Value::Flts(a) => (a.rank(), a.len(), true),
            _ => return Value::raw_list(items),
        };
        if len == 0 || rank >= MAX_RANK {
            return Value::raw_list(items);
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
                _ => return Value::raw_list(items),
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
            Value::Str(s) => Some(s.len()),
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
    /// Element `i` along the leading dimension. `i` must be in range. A row of an array of
    /// rank above 1 is a copy; when there is no memory for it the result is `:undefined` and
    /// the failure is deferred (`defer_memory`), which the evaluator raises.
    pub fn item(&self, i: usize) -> Value {
        match self {
            Value::Ints(a) => {
                if a.rank() <= 1 {
                    Value::Int(a.data[i])
                } else {
                    let n = a.row_size();
                    match try_copy(&a.data[i * n..(i + 1) * n]) {
                        Some(v) => Value::Ints(Rc::new(Arr::shaped(v, a.row_dims()))),
                        None => Value::Undefined,
                    }
                }
            }
            Value::Flts(a) => {
                if a.rank() <= 1 {
                    Value::Flt(a.data[i])
                } else {
                    let n = a.row_size();
                    match try_copy(&a.data[i * n..(i + 1) * n]) {
                        Some(v) => Value::Flts(Rc::new(Arr::shaped(v, a.row_dims()))),
                        None => Value::Undefined,
                    }
                }
            }
            Value::List(a) => a[i].clone(),
            Value::Str(s) => Value::Char(s.char_at(i)),
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
            #[cfg(feature = "host")]
            Value::Ext(e) => e.0.host_truthy().unwrap_or(true),
            _ => !self.is_empty(),
        }
    }
    pub fn as_int(&self) -> Option<Int> {
        match self {
            Value::Int(i) => Some(*i),
            _ => None,
        }
    }
    #[cfg(feature = "bigint")]
    pub fn is_big(&self) -> bool {
        matches!(self, Value::Big(_))
    }
    pub fn as_flt(&self) -> Option<Flt> {
        match self {
            Value::Int(i) => Some(*i as Flt),
            Value::Flt(f) => Some(*f),
            _ => None,
        }
    }
    /// KlongPy's description of a value in an error message: `integer`, `string`, `list of 3`, `2x3 matrix`, `empty list`, `:undefined`.
    pub fn describe_long(&self) -> alloc::string::String {
        use alloc::string::ToString;
        let dims = |rank: usize, len: usize, row: &[u32]| -> alloc::string::String {
            if rank <= 1 {
                if len == 0 {
                    "empty list".to_string()
                } else {
                    crate::msg!("list of {}", len)
                }
            } else {
                let mut s = len.to_string();
                for d in row {
                    s.push('x');
                    s.push_str(&d.to_string());
                }
                crate::msg!("{} {}", s, if rank == 2 { "matrix" } else { "array" })
            }
        };
        match self {
            Value::Flt(f) if !f.is_finite() => ":undefined".to_string(),
            Value::Ints(a) => dims(a.rank(), a.len(), a.row_dims()),
            Value::Flts(a) => dims(a.rank(), a.len(), a.row_dims()),
            // a list of empty lists is a matrix of no columns (NumPy's shape (n, 0))
            Value::List(l) if !l.is_empty() && (0..l.len()).all(|i| { let x = self.item(i); x.is_list() && x.is_empty() }) => dims(2, l.len(), &[0]),
            Value::List(l) => dims(1, l.len(), &[]),
            #[cfg(feature = "host")]
            Value::Ext(e) if e.0.host_shape().is_some_and(|d| !d.is_empty()) => {
                let d = e.0.host_shape().unwrap_or_default();
                dims(d.len(), d[0] as usize, &d[1..])
            }
            #[cfg(feature = "host")]
            Value::Ext(e) if e.0.host_format(false).is_some() => "host array".to_string(),
            Value::Ext(_) => "channel".to_string(),
            v => v.describe().to_string(),
        }
    }

    #[cfg(feature = "diag")]
    /// For a type error on a list: what it holds that is not a number (`:undefined`, `a string`), KlongPy `_odd_member`.
    pub fn odd_member(&self, depth: u32, syms: Option<&crate::sym::Interner>) -> Option<alloc::string::String> {
        let items: &[Value] = match self {
            Value::List(l) => l,
            _ => return None,
        };
        for el in items.iter().take(1000) {
            let d = el.describe_long();
            // `:undefined` written inside a list literal is the symbol of that name, which KlongPy calls the same
            if matches!(el, Value::Sym(s) if syms.is_some_and(|t| t.name(*s) == "undefined")) {
                return Some(alloc::string::String::from(":undefined"));
            }
            if d == ":undefined" || matches!(el, Value::Undefined) {
                return Some(d);
            }
            if d.starts_with("list of") || d.ends_with("matrix") || d.ends_with("array") {
                if depth < 3 {
                    if let Some(o) = el.odd_member(depth + 1, syms) {
                        return Some(o);
                    }
                }
            } else if !matches!(d.as_str(), "integer" | "float" | "empty list") {
                let art = if matches!(d.as_bytes()[0], b'a' | b'e' | b'i' | b'o' | b'u') { "an " } else { "a " };
                return Some(crate::msg!("{}{}", art, d));
            }
        }
        None
    }

    /// How Python would `repr` the value, for the "got ..." of a message: `1`, `1.5`, `'a'`, `[-2, 3]`.
    pub fn py_repr(&self, syms: Option<&crate::sym::Interner>) -> alloc::string::String {
        use alloc::string::ToString;
        match self {
            Value::Int(i) => i.to_string(),
            #[cfg(feature = "bigint")]
            Value::Big(b) => crate::big::text(b),
            Value::Flt(_) => crate::fmt::format_value(syms.unwrap_or(&crate::sym::Interner::new()), self, false).unwrap_or_default(),
            Value::Str(s) => crate::msg!("'{}'", s.as_str()),
            Value::Char(c) => crate::msg!("'{}'", c),
            Value::Sym(s) => match syms {
                Some(t) => crate::msg!(":{}", t.name(*s)),
                None => "symbol".to_string(),
            },
            Value::Ints(_) | Value::Flts(_) | Value::List(_) => {
                let mut out = alloc::string::String::from("[");
                for i in 0..self.len().unwrap_or(0) {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    out.push_str(&self.item(i).py_repr(syms));
                }
                out.push(']');
                out
            }
            v => v.describe().to_string(),
        }
    }

    /// A word for error messages.
    pub fn describe(&self) -> &'static str {
        match self {
            Value::Int(_) => "integer",
            #[cfg(feature = "bigint")]
            Value::Big(_) => "integer",
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
/// match when they are equal; lists match element by element; dictionaries match when they
/// hold the same keys with matching values, in any order.
pub fn matches(a: &Value, b: &Value) -> bool {
    matches_at(a, b, 0)
}

fn same_dims<T: Copy, U: Copy>(x: &Arr<T>, y: &Arr<U>) -> bool {
    x.rank() == y.rank() && x.len() == y.len() && x.row_dims() == y.row_dims()
}

fn matches_at(a: &Value, b: &Value, depth: u32) -> bool {
    match (a, b) {
        (Value::Ints(x), Value::Ints(y)) => same_dims(x, y) && x.data == y.data,
        (Value::Flts(x), Value::Flts(y)) => same_dims(x, y) && x.data == y.data,
        (Value::Ints(x), Value::Flts(y)) | (Value::Flts(y), Value::Ints(x)) => same_dims(x, y) && x.data.iter().zip(y.data.iter()).all(|(&i, &f)| i as Flt == f),
        (Value::Dict(x), Value::Dict(y)) => {
            if Rc::ptr_eq(x, y) {
                return true;
            }
            if depth >= max_nest() {
                // dictionaries can nest without bound (they can even hold themselves)
                defer_depth();
                return false;
            }
            let (x, y) = match (x.try_borrow(), y.try_borrow()) {
                (Ok(x), Ok(y)) => (x, y),
                _ => return false,
            };
            x.len() == y.len()
                && x.keys.iter().zip(x.vals.iter()).all(|(k, v)| match y.get(k) {
                    Some(w) => matches_at(v, w, depth + 1),
                    None => false,
                })
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
                if depth >= max_nest() {
                    defer_depth();
                    return false;
                }
                return (0..n).all(|i| matches_at(&a.item(i), &b.item(i), depth + 1));
            }
            #[cfg(feature = "bigint")]
            if (a.is_big() || b.is_big()) && matches!(a, Value::Int(_) | Value::Flt(_) | Value::Big(_)) && matches!(b, Value::Int(_) | Value::Flt(_) | Value::Big(_)) {
                return is_close(crate::big::to_flt(a) as Flt, crate::big::to_flt(b) as Flt);
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
