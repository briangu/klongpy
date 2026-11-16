//! Operators: one table (glyph, monad, dyad, names, atomic flag) and their implementations.
//!
//! All atomic dyads share one pairing routine, [`atomic`], which implements KlongPy's rule
//! (`vec_fn2` in `klongpy/backends/base.py`): scalar extension, recursive pairing of lists,
//! leading-dimension agreement for arrays. Loops are plain. Integer arithmetic wraps at the
//! build width.
use alloc::format;
use alloc::rc::Rc;
use alloc::vec::Vec;
use core::cell::RefCell;
use core::cmp::Ordering;

use crate::error::{Error, Result};
use crate::interp::Interp;
use crate::sym::{SymId, FIRST_OP};
use crate::value::{atom_eq, matches, try_extend, try_push, try_vec, Arr, Dict, Flt, Int, Value};

pub type MonadFn = fn(&mut Interp, Value) -> Result<Value>;
pub type DyadFn = fn(&mut Interp, Value, Value) -> Result<Value>;

pub struct OpDef {
    pub glyph: &'static str,
    pub monad_name: &'static str,
    pub dyad_name: &'static str,
    pub monad: Option<MonadFn>,
    pub dyad: Option<DyadFn>,
    /// The dyad is atomic: it goes through the shared pairing routine.
    pub atomic: Option<Atomic>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Atomic {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    IDiv,
    Pow,
    Min,
    Max,
    Less,
    More,
    Eq,
}

macro_rules! op {
    ($g:expr, $mn:expr, $m:expr, $dn:expr, $d:expr, $a:expr) => {
        OpDef { glyph: $g, monad_name: $mn, dyad_name: $dn, monad: $m, dyad: $d, atomic: $a }
    };
}

fn absent_m(_: &mut Interp, _: Value) -> Result<Value> {
    Err(Error::name("operator not available on this build").with_hint("automatic differentiation is desktop only"))
}
fn absent_d(_: &mut Interp, _: Value, _: Value) -> Result<Value> {
    Err(Error::name("operator not available on this build").with_hint("automatic differentiation is desktop only"))
}
fn later_d(_: &mut Interp, _: Value, _: Value) -> Result<Value> {
    Err(Error::typ("operator not implemented in the native core yet (strings tier, ticket 103)"))
}
fn define_d(_: &mut Interp, _: Value, y: Value) -> Result<Value> {
    // `::` is evaluated by the interpreter (its left operand is not evaluated); as a function
    // value applied to two values it binds nothing.
    Ok(y)
}

macro_rules! at {
    ($name:ident, $op:expr) => {
        fn $name(_: &mut Interp, a: Value, b: Value) -> Result<Value> {
            atomic($op, &a, &b)
        }
    };
}
at!(d_add, Atomic::Add);
at!(d_sub, Atomic::Sub);
at!(d_mul, Atomic::Mul);
at!(d_div, Atomic::Div);
at!(d_rem, Atomic::Rem);
at!(d_idiv, Atomic::IDiv);
at!(d_pow, Atomic::Pow);
at!(d_min, Atomic::Min);
at!(d_max, Atomic::Max);
at!(d_less, Atomic::Less);
at!(d_more, Atomic::More);
at!(d_eq, Atomic::Eq);

/// The operator table. Order is fixed: the glyphs are interned in this order after x, y, z
/// and .f, so an operator's symbol id minus `FIRST_OP` is its index here.
pub static OPS: [OpDef; 31] = [
    op!("+", "Transpose", Some(m_transpose), "Plus", Some(d_add), Some(Atomic::Add)),
    op!("-", "Negate", Some(m_negate), "Minus", Some(d_sub), Some(Atomic::Sub)),
    op!("*", "First", Some(m_first), "Times", Some(d_mul), Some(Atomic::Mul)),
    op!("%", "Reciprocal", Some(m_reciprocal), "Divide", Some(d_div), Some(Atomic::Div)),
    op!("!", "Enumerate", Some(m_enumerate), "Remainder", Some(d_rem), Some(Atomic::Rem)),
    op!(":%", "", None, "Integer-Divide", Some(d_idiv), Some(Atomic::IDiv)),
    op!("^", "Shape", Some(m_shape), "Power", Some(d_pow), Some(Atomic::Pow)),
    op!("&", "Expand/Where", Some(m_where), "Min/And", Some(d_min), Some(Atomic::Min)),
    op!("|", "Reverse", Some(m_reverse), "Max/Or", Some(d_max), Some(Atomic::Max)),
    op!("<", "Grade-Up", Some(m_grade_up), "Less", Some(d_less), Some(Atomic::Less)),
    op!(">", "Grade-Down", Some(m_grade_down), "More", Some(d_more), Some(Atomic::More)),
    op!("=", "Group", Some(m_group), "Equal", Some(d_eq), Some(Atomic::Eq)),
    op!("@", "Atom", Some(m_atom), "Index/Apply", Some(d_at), None),
    op!(",", "List", Some(m_list), "Join", Some(d_join), None),
    op!(":#", "Char", Some(m_char), "Split", Some(d_split), None),
    op!("_", "Floor", Some(m_floor), "Drop", Some(d_drop), None),
    op!("$", "Format", Some(m_format), "Format2", Some(later_d), None),
    op!("~", "Not", Some(m_not), "Match", Some(d_match), None),
    op!("?", "Range", Some(m_range), "Find", Some(d_find), None),
    op!("#", "Size", Some(m_size), "Take", Some(d_take), None),
    op!(":_", "Undefined", Some(m_undefined), "Cut", Some(d_cut), None),
    op!(":=", "", None, "Amend", Some(d_amend), None),
    op!(":-", "", None, "Amend-in-Depth", Some(d_amend_depth), None),
    op!(":@", "", None, "Index-in-Depth", Some(d_index_depth), None),
    op!(":$", "", None, "Form", Some(later_d), None),
    op!(":^", "", None, "Reshape", Some(d_reshape), None),
    op!(":+", "", None, "Rotate", Some(d_rotate), None),
    op!("::", "", None, "Define", Some(define_d), None),
    // Desktop only (autograd): known to the parser so that trees agree; absent at run time.
    op!("\u{2207}", "Grad", Some(absent_m), "Grad", Some(absent_d), None),
    op!("\u{2202}", "", None, "Jacobian", Some(absent_d), None),
    op!(":>", "", None, "Autograd", Some(absent_d), None),
];

/// `˙` (Track) is a monad in KlongPy; it is the identity.
pub const TRACK: &str = "\u{2d9}";

pub fn find(glyph: &str) -> Option<&'static OpDef> {
    OPS.iter().find(|o| o.glyph == glyph)
}
pub fn by_id(s: SymId) -> Option<&'static OpDef> {
    OPS.get(s.0.checked_sub(FIRST_OP)? as usize)
}
pub fn is_monad(glyph: &str) -> bool {
    glyph == TRACK || find(glyph).map(|o| o.monad.is_some()).unwrap_or(false)
}
pub fn is_dyad(glyph: &str) -> bool {
    find(glyph).map(|o| o.dyad.is_some()).unwrap_or(false)
}

// ---------------------------------------------------------------------------------------
// Numbers
// ---------------------------------------------------------------------------------------

fn fabs(f: Flt) -> Flt {
    if f < 0.0 {
        -f
    } else {
        f
    }
}

#[cfg(not(feature = "float32"))]
mod m {
    pub fn floor(f: f64) -> f64 {
        libm::floor(f)
    }
    pub fn trunc(f: f64) -> f64 {
        libm::trunc(f)
    }
    pub fn fmod(a: f64, b: f64) -> f64 {
        libm::fmod(a, b)
    }
    pub fn pow(a: f64, b: f64) -> f64 {
        libm::pow(a, b)
    }
}
#[cfg(feature = "float32")]
mod m {
    pub fn floor(f: f32) -> f32 {
        libm::floorf(f)
    }
    pub fn trunc(f: f32) -> f32 {
        libm::truncf(f)
    }
    pub fn fmod(a: f32, b: f32) -> f32 {
        libm::fmodf(a, b)
    }
    pub fn pow(a: f32, b: f32) -> f32 {
        libm::powf(a, b)
    }
}

/// A number as an operand of an atomic dyad.
#[derive(Clone, Copy)]
enum Num {
    I(Int),
    F(Flt),
}

impl Num {
    fn f(self) -> Flt {
        match self {
            Num::I(i) => i as Flt,
            Num::F(f) => f,
        }
    }
    fn is_zero(self) -> bool {
        match self {
            Num::I(i) => i == 0,
            Num::F(f) => f == 0.0,
        }
    }
    fn value(self) -> Value {
        match self {
            Num::I(i) => Value::Int(i),
            Num::F(f) => Value::Flt(f),
        }
    }
}

fn num(v: &Value) -> Option<Num> {
    match v {
        Value::Int(i) => Some(Num::I(*i)),
        Value::Flt(f) => Some(Num::F(*f)),
        _ => None,
    }
}

fn int_pow(mut base: Int, mut exp: Int) -> Int {
    let mut r: Int = 1;
    while exp > 0 {
        if exp & 1 == 1 {
            r = r.wrapping_mul(base);
        }
        base = base.wrapping_mul(base);
        exp >>= 1;
    }
    r
}

fn flt_to_int(f: Flt) -> Result<Int> {
    if f.is_nan() || f.is_infinite() {
        return Err(Error::range("cannot convert a non-finite number to an integer"));
    }
    Ok(f as Int)
}

/// One pair of numbers. `None` is `:undefined` (a zero divisor).
fn num_op(op: Atomic, a: Num, b: Num) -> Result<Option<Num>> {
    use Num::{F, I};
    Ok(Some(match op {
        Atomic::Add => match (a, b) {
            (I(x), I(y)) => I(x.wrapping_add(y)),
            _ => F(a.f() + b.f()),
        },
        Atomic::Sub => match (a, b) {
            (I(x), I(y)) => I(x.wrapping_sub(y)),
            _ => F(a.f() - b.f()),
        },
        Atomic::Mul => match (a, b) {
            (I(x), I(y)) => I(x.wrapping_mul(y)),
            _ => F(a.f() * b.f()),
        },
        Atomic::Min => match (a, b) {
            (I(x), I(y)) => I(x.min(y)),
            _ => F(if b.f() < a.f() { b.f() } else { a.f() }),
        },
        Atomic::Max => match (a, b) {
            (I(x), I(y)) => I(x.max(y)),
            _ => F(if b.f() > a.f() { b.f() } else { a.f() }),
        },
        Atomic::Less => match (a, b) {
            (I(x), I(y)) => I((x < y) as Int),
            _ => I((a.f() < b.f()) as Int),
        },
        Atomic::More => match (a, b) {
            (I(x), I(y)) => I((x > y) as Int),
            _ => I((a.f() > b.f()) as Int),
        },
        Atomic::Eq => match (a, b) {
            (I(x), I(y)) => I((x == y) as Int),
            _ => I((a.f() == b.f()) as Int),
        },
        Atomic::Div => {
            if b.is_zero() {
                return Ok(None);
            }
            F(a.f() / b.f())
        }
        Atomic::Rem => {
            if b.is_zero() {
                return Ok(None);
            }
            match (a, b) {
                (I(x), I(y)) => I(x.wrapping_rem(y)),
                _ => F(m::fmod(a.f(), b.f())),
            }
        }
        Atomic::IDiv => {
            if b.is_zero() {
                return Ok(None);
            }
            match (a, b) {
                (I(x), I(y)) => I(x.wrapping_div(y)),
                _ => I(flt_to_int(m::trunc(a.f() / b.f()))?),
            }
        }
        Atomic::Pow => F(m::pow(a.f(), b.f())),
    }))
}

fn type_error(op: Atomic, a: &Value, b: &Value) -> Error {
    let (g, n) = match op {
        Atomic::Add => ("+", "Plus"),
        Atomic::Sub => ("-", "Minus"),
        Atomic::Mul => ("*", "Times"),
        Atomic::Div => ("%", "Divide"),
        Atomic::Rem => ("!", "Remainder"),
        Atomic::IDiv => (":%", "Integer-Divide"),
        Atomic::Pow => ("^", "Power"),
        Atomic::Min => ("&", "Min/And"),
        Atomic::Max => ("|", "Max/Or"),
        Atomic::Less => ("<", "Less"),
        Atomic::More => (">", "More"),
        Atomic::Eq => ("=", "Equal"),
    };
    Error::typ(format!("cannot apply {} ({}) to {} and {}", g, n, a.describe(), b.describe()))
}

fn text_of(v: &Value) -> Option<Vec<u8>> {
    match v {
        Value::Str(s) => Some(s.0.clone()),
        Value::Char(c) => Some(alloc::vec![*c]),
        _ => None,
    }
}

/// One pair of atoms (anything that is not a list).
fn atom_op(op: Atomic, a: &Value, b: &Value) -> Result<Value> {
    if let (Some(x), Some(y)) = (num(a), num(b)) {
        if op == Atomic::Pow {
            // KlongPy: a power whose result is a whole number is an integer.
            let r = m::pow(x.f(), y.f());
            if r.is_finite() && m::trunc(r) == r {
                if let (Num::I(i), Num::I(j)) = (x, y) {
                    if j >= 0 && fabs(r) < 9.0e15 {
                        return Ok(Value::Int(int_pow(i, j)));
                    }
                }
                return Ok(Value::Int(flt_to_int(r)?));
            }
            if !r.is_finite() && !r.is_nan() {
                return Err(Error::range("power: the result does not fit an integer"));
            }
            return Ok(Value::Flt(r));
        }
        return Ok(match num_op(op, x, y)? {
            Some(n) => n.value(),
            None => Value::Undefined,
        });
    }
    match op {
        Atomic::Eq => Ok(Value::bool(atom_eq(a, b))),
        Atomic::Less | Atomic::More => {
            let ord = match (a, b) {
                (Value::Sym(_), Value::Sym(_)) => return Err(type_error(op, a, b)),
                _ => match (text_of(a), text_of(b)) {
                    (Some(x), Some(y)) => x.cmp(&y),
                    _ => return Err(type_error(op, a, b)),
                },
            };
            Ok(Value::bool(if op == Atomic::Less { ord == Ordering::Less } else { ord == Ordering::Greater }))
        }
        _ => Err(type_error(op, a, b)),
    }
}

fn shape_error(a: usize, b: usize) -> Error {
    Error::shape(format!("atomic dyad shape mismatch: length {} and length {}", a, b))
}

/// The operands of a numeric array loop: a slice or one repeated number.
enum Operand<'a> {
    Ints(&'a [Int]),
    Flts(&'a [Flt]),
    One(Num),
}

impl<'a> Operand<'a> {
    fn of(v: &'a Value) -> Option<(Operand<'a>, Option<&'a Value>)> {
        match v {
            Value::Ints(a) => Some((Operand::Ints(&a.data), Some(v))),
            Value::Flts(a) => Some((Operand::Flts(&a.data), Some(v))),
            Value::Int(i) => Some((Operand::One(Num::I(*i)), None)),
            Value::Flt(f) => Some((Operand::One(Num::F(*f)), None)),
            _ => None,
        }
    }
    fn is_int(&self) -> bool {
        matches!(self, Operand::Ints(_) | Operand::One(Num::I(_)))
    }
    #[inline]
    fn get(&self, i: usize) -> Num {
        match self {
            Operand::Ints(s) => Num::I(s[i]),
            Operand::Flts(s) => Num::F(s[i]),
            Operand::One(n) => *n,
        }
    }
}

fn dims_of(v: &Value) -> (usize, usize, &[u32], usize) {
    // (rank, leading length, row dims, total scalars)
    match v {
        Value::Ints(a) => (a.rank(), a.len(), a.row_dims(), a.data.len()),
        Value::Flts(a) => (a.rank(), a.len(), a.row_dims(), a.data.len()),
        _ => (0, 0, &[], 1),
    }
}

fn shaped_like(like: &Value, ints: Option<Vec<Int>>, flts: Option<Vec<Flt>>) -> Value {
    let (rank, len, row, _) = dims_of(like);
    let mut dims = [0u32; crate::value::MAX_RANK];
    dims[0] = len as u32;
    dims[1..rank.max(1)].copy_from_slice(row);
    let dims = &dims[..rank.max(1)];
    match (ints, flts) {
        (Some(v), _) => Value::Ints(Rc::new(Arr::shaped(v, dims))),
        (_, Some(v)) => Value::Flts(Rc::new(Arr::shaped(v, dims))),
        _ => Value::empty(),
    }
}

/// Numeric array against numeric array or number.
fn array_op(op: Atomic, a: &Value, b: &Value) -> Result<Value> {
    let ((x, xa), (y, ya)) = match (Operand::of(a), Operand::of(b)) {
        (Some(x), Some(y)) => (x, y),
        _ => return Err(type_error(op, a, b)),
    };
    // Leading dimensions must agree; the operand of lower rank is repeated along the rest.
    let (like, n, xdiv, ydiv) = match (xa, ya) {
        (Some(p), Some(q)) => {
            let (pr, pl, prow, pn) = dims_of(p);
            let (qr, ql, qrow, qn) = dims_of(q);
            let r = pr.min(qr);
            let agree = pl == ql && prow[..r - 1] == qrow[..r - 1];
            if !agree {
                return Err(Error::shape("atomic dyad shape mismatch"));
            }
            if pr >= qr {
                (p, pn, 1, if qn == 0 { 1 } else { pn / qn })
            } else {
                (q, qn, if pn == 0 { 1 } else { qn / pn }, 1)
            }
        }
        (Some(p), None) => (p, dims_of(p).3, 1, 1),
        (None, Some(q)) => (q, dims_of(q).3, 1, 1),
        (None, None) => return atom_op(op, a, b),
    };
    let both_int = x.is_int() && y.is_int();
    let int_result = match op {
        Atomic::Less | Atomic::More | Atomic::Eq | Atomic::IDiv => true,
        Atomic::Add | Atomic::Sub | Atomic::Mul | Atomic::Min | Atomic::Max | Atomic::Rem => both_int,
        Atomic::Pow => both_int && matches!(&y, Operand::Ints(_) | Operand::One(Num::I(_))) && (0..n).all(|i| matches!(y.get(i / ydiv), Num::I(e) if e >= 0)),
        Atomic::Div => false,
    };
    if op == Atomic::Pow && both_int && !int_result {
        if matches!(x, Operand::Ints(_)) {
            return Err(Error::range("integers to negative integer powers are not allowed"));
        }
    }
    // Zero divisors give :undefined for that element, so the result is a mixed list.
    if matches!(op, Atomic::Div | Atomic::Rem | Atomic::IDiv) && (0..n).any(|i| y.get(i / ydiv).is_zero()) {
        let mut out = try_vec(n)?;
        for i in 0..n {
            out.push(match num_op(op, x.get(i / xdiv), y.get(i / ydiv))? {
                Some(v) => v.value(),
                None => Value::Undefined,
            });
        }
        return reshape_flat(out, like);
    }
    if int_result {
        let mut out: Vec<Int> = try_vec(n)?;
        if op == Atomic::Pow {
            for i in 0..n {
                if let (Num::I(p), Num::I(q)) = (x.get(i / xdiv), y.get(i / ydiv)) {
                    out.push(int_pow(p, q));
                }
            }
        } else {
            match (&x, &y, op) {
                // the common loops, written so that they vectorise
                (Operand::Ints(p), Operand::Ints(q), Atomic::Add) if xdiv == 1 && ydiv == 1 => out.extend(p.iter().zip(q.iter()).map(|(a, b)| a.wrapping_add(*b))),
                (Operand::Ints(p), Operand::One(Num::I(q)), Atomic::Add) => out.extend(p.iter().map(|a| a.wrapping_add(*q))),
                (Operand::Ints(p), Operand::One(Num::I(q)), Atomic::Mul) => out.extend(p.iter().map(|a| a.wrapping_mul(*q))),
                _ => {
                    for i in 0..n {
                        if let Some(Num::I(v)) = num_op(op, x.get(i / xdiv), y.get(i / ydiv))? {
                            out.push(v);
                        }
                    }
                }
            }
        }
        return Ok(shaped_like(like, Some(out), None));
    }
    let mut out: Vec<Flt> = try_vec(n)?;
    for i in 0..n {
        if let Some(v) = num_op(op, x.get(i / xdiv), y.get(i / ydiv))? {
            out.push(v.f());
        }
    }
    if op == Atomic::Pow && out.iter().all(|f| f.is_finite() && m::trunc(*f) == *f) {
        // KlongPy: a power array of whole numbers is an integer array.
        let mut ints: Vec<Int> = try_vec(n)?;
        ints.extend(out.iter().map(|f| *f as Int));
        return Ok(shaped_like(like, Some(ints), None));
    }
    Ok(shaped_like(like, None, Some(out)))
}

/// A flat run of values given the shape of the numeric array `like`.
fn reshape_flat(flat: Vec<Value>, like: &Value) -> Result<Value> {
    let (rank, len, _, total) = dims_of(like);
    if rank <= 1 {
        return Value::from_items(flat);
    }
    let per = if len == 0 { 0 } else { total / len };
    let mut rows = try_vec(len)?;
    let like_row = like.item(0);
    for r in 0..len {
        let mut part = try_vec(per)?;
        part.extend_from_slice(&flat[r * per..(r + 1) * per]);
        rows.push(reshape_flat(part, &like_row)?);
    }
    Value::from_items(rows)
}

/// The pairing routine of every atomic dyad.
pub fn atomic(op: Atomic, a: &Value, b: &Value) -> Result<Value> {
    let al = a.is_list();
    let bl = b.is_list();
    if !al && !bl {
        return atom_op(op, a, b);
    }
    let a_obj = matches!(a, Value::List(_));
    let b_obj = matches!(b, Value::List(_));
    if !a_obj && !b_obj {
        if (al || num(a).is_some()) && (bl || num(b).is_some()) {
            return array_op(op, a, b);
        }
        // a numeric array against an atom that is not a number
        if al {
            let n = a.len().unwrap_or(0);
            let mut out = try_vec(n)?;
            for i in 0..n {
                out.push(atomic(op, &a.item(i), b)?);
            }
            return Value::from_items(out);
        }
        let n = b.len().unwrap_or(0);
        let mut out = try_vec(n)?;
        for i in 0..n {
            out.push(atomic(op, a, &b.item(i))?);
        }
        return Value::from_items(out);
    }
    if al && bl {
        let n = a.len().unwrap_or(0);
        let k = b.len().unwrap_or(0);
        if n != k {
            return Err(shape_error(n, k));
        }
        let mut out = try_vec(n)?;
        for i in 0..n {
            out.push(atomic(op, &a.item(i), &b.item(i))?);
        }
        return Value::from_items(out);
    }
    if al {
        let n = a.len().unwrap_or(0);
        let mut out = try_vec(n)?;
        for i in 0..n {
            out.push(atomic(op, &a.item(i), b)?);
        }
        return Value::from_items(out);
    }
    let n = b.len().unwrap_or(0);
    let mut out = try_vec(n)?;
    for i in 0..n {
        out.push(atomic(op, a, &b.item(i))?);
    }
    Value::from_items(out)
}

/// Apply `f` to every atom of `a` (KlongPy `vec_fn` / `rec_fn`): lists are mapped element by
/// element, recursively; anything else is given to `f`.
fn map_atoms(a: &Value, f: &dyn Fn(&Value) -> Result<Value>) -> Result<Value> {
    if a.is_list() && !a.is_empty() {
        let n = a.len().unwrap_or(0);
        let mut out = try_vec(n)?;
        for i in 0..n {
            out.push(map_atoms(&a.item(i), f)?);
        }
        return Value::from_items(out);
    }
    f(a)
}

// ---------------------------------------------------------------------------------------
// Monads
// ---------------------------------------------------------------------------------------

fn m_atom(_: &mut Interp, a: Value) -> Result<Value> {
    Ok(Value::bool(a.is_atom()))
}

fn m_first(_: &mut Interp, a: Value) -> Result<Value> {
    Ok(if a.is_atom() { a } else { a.item(0) })
}

/// A list or string with the same kind as `like` from its elements.
fn rebuild(like: &Value, items: Vec<Value>) -> Result<Value> {
    if like.is_str() {
        let mut s = try_vec(items.len())?;
        for q in &items {
            match q {
                Value::Char(c) => s.push(*c),
                Value::Str(t) => try_extend(&mut s, &t.0)?,
                _ => return Value::from_items(items),
            }
        }
        return Ok(Value::str(s));
    }
    Value::from_items(items)
}

fn m_reverse(_: &mut Interp, a: Value) -> Result<Value> {
    if a.is_atom() {
        return Ok(a);
    }
    if let Value::Ints(x) = &a {
        if x.rank() == 1 {
            let mut v = try_vec(x.data.len())?;
            v.extend(x.data.iter().rev().copied());
            return Ok(Value::ints(v));
        }
    }
    let mut items = a.items()?;
    items.reverse();
    rebuild(&a, items)
}

fn m_where(_: &mut Interp, a: Value) -> Result<Value> {
    let counts: Vec<Value> = if a.is_list() { a.items()? } else { alloc::vec![a.clone()] };
    let mut total: usize = 0;
    for c in &counts {
        match c {
            Value::Int(n) if *n >= 0 => total = total.checked_add(*n as usize).ok_or_else(Error::memory)?,
            Value::Int(_) => return Err(Error::range("&: counts must not be negative")),
            Value::Flt(_) => return Err(Error::typ("&: counts must be integers")),
            _ => return Err(Error::range("&: expand needs a list of counts")),
        }
    }
    let mut out: Vec<Int> = try_vec(total)?;
    for (i, c) in counts.iter().enumerate() {
        if let Value::Int(n) = c {
            for _ in 0..*n {
                out.push(i as Int);
            }
        }
    }
    Ok(Value::ints(out))
}

fn m_transpose(_: &mut Interp, a: Value) -> Result<Value> {
    if !a.is_list() || a.is_empty() {
        return Ok(a);
    }
    let rows = a.items()?;
    let width = match rows[0].len() {
        Some(w) if rows[0].is_list() => w,
        _ => return Ok(a),
    };
    if !rows.iter().all(|r| r.is_list() && r.len() == Some(width)) {
        return Ok(a);
    }
    let mut out = try_vec(width)?;
    for j in 0..width {
        let mut col = try_vec(rows.len())?;
        for r in &rows {
            col.push(r.item(j));
        }
        out.push(Value::from_items(col)?);
    }
    Value::from_items(out)
}

fn m_list(_: &mut Interp, a: Value) -> Result<Value> {
    if let Value::Char(c) = a {
        return Ok(Value::str(alloc::vec![c]));
    }
    Value::from_items(alloc::vec![a])
}

fn m_char(_: &mut Interp, a: Value) -> Result<Value> {
    map_atoms(&a, &|x| match x {
        Value::Int(i) if (0..256).contains(i) => Ok(Value::Char(*i as u8)),
        Value::Int(_) => Err(Error::range(":#: character code out of range")),
        _ => Err(Error::typ(":#: char needs an integer")),
    })
}

fn m_enumerate(_: &mut Interp, a: Value) -> Result<Value> {
    match a {
        Value::Int(n) => {
            let n = if n < 0 { 0 } else { n as usize };
            let mut v: Vec<Int> = try_vec(n)?;
            v.extend((0..n).map(|i| i as Int));
            Ok(Value::ints(v))
        }
        _ => Err(Error::typ(format!("!: enumerate needs an integer, not {}", a.describe()))),
    }
}

fn m_floor(_: &mut Interp, a: Value) -> Result<Value> {
    if let Value::Ints(_) = a {
        return Ok(a);
    }
    map_atoms(&a, &|x| match x {
        Value::Int(i) => Ok(Value::Int(*i)),
        Value::Flt(f) => Ok(Value::Int(flt_to_int(m::floor(*f))?)),
        _ => Err(Error::range(format!("_: cannot floor {}", x.describe()))),
    })
}

fn m_format(ip: &mut Interp, a: Value) -> Result<Value> {
    if a.is_list() {
        let n = a.len().unwrap_or(0);
        let mut out = try_vec(n)?;
        for i in 0..n {
            out.push(m_format(ip, a.item(i))?);
        }
        return Value::from_items(out);
    }
    match &a {
        Value::Str(_) => Ok(a),
        Value::Char(c) => Ok(Value::str(alloc::vec![*c])),
        _ => Ok(Value::str(crate::fmt::format_value(&ip.syms, &a, false)?.into_bytes())),
    }
}

fn cmp_atoms(a: &Value, b: &Value) -> Result<Ordering> {
    match (a.as_flt(), b.as_flt()) {
        (Some(x), Some(y)) => Ok(match (a, b) {
            (Value::Int(i), Value::Int(j)) => i.cmp(j),
            _ => x.partial_cmp(&y).unwrap_or(Ordering::Equal),
        }),
        _ => match (text_of(a), text_of(b)) {
            (Some(x), Some(y)) => Ok(x.cmp(&y)),
            _ => Err(Error::typ("cannot order values of different kinds")),
        },
    }
}

/// The sort key of an element (KlongPy `kg_argsort`): a list sorts by its largest element,
/// an empty one before everything.
fn grade_key(ip: &Interp, v: &Value) -> Result<Option<Value>> {
    if let Value::Sym(s) = v {
        // a symbol sorts as its name
        return Ok(Some(Value::str(ip.syms.name(*s).as_bytes().to_vec())));
    }
    if v.is_empty() {
        return Ok(None);
    }
    if v.is_list() {
        let mut best = grade_key(ip, &v.item(0))?;
        for i in 1..v.len().unwrap_or(0) {
            let k = grade_key(ip, &v.item(i))?;
            if let (Some(x), Some(y)) = (&best, &k) {
                if cmp_atoms(y, x)? == Ordering::Greater {
                    best = k;
                }
            }
        }
        return Ok(best);
    }
    Ok(Some(v.clone()))
}

fn grade(ip: &Interp, a: Value, down: bool) -> Result<Value> {
    if !a.is_iterable() || a.is_empty() {
        return Ok(a);
    }
    let n = a.len().unwrap_or(0);
    let mut idx: Vec<usize> = try_vec(n)?;
    idx.extend(0..n);
    let numeric = match &a {
        Value::Ints(x) if x.rank() == 1 => {
            idx.sort_by(|&i, &j| x.data[i].cmp(&x.data[j]));
            true
        }
        Value::Flts(x) if x.rank() == 1 => {
            // a total order (sort_by may panic without one): NaN last, as numpy sorts
            idx.sort_by(|&i, &j| x.data[i].partial_cmp(&x.data[j]).unwrap_or_else(|| x.data[i].is_nan().cmp(&x.data[j].is_nan())));
            true
        }
        _ => false,
    };
    if numeric {
        // numpy argsort; descending is the ascending order reversed
        if down {
            idx.reverse();
        }
    } else {
        let mut keys = try_vec(n)?;
        for i in 0..n {
            keys.push(grade_key(ip, &a.item(i))?);
        }
        // The keys must be mutually comparable before sorting (all numbers, or all text): the
        // comparison given to sort_by has to be a total order.
        let mut numbers = false;
        let mut text = false;
        for k in keys.iter().flatten() {
            match k {
                Value::Int(_) => numbers = true,
                Value::Flt(f) if !f.is_nan() => numbers = true,
                Value::Str(_) | Value::Char(_) => text = true,
                _ => return Err(Error::typ("cannot order values of different kinds")),
            }
        }
        if numbers && text {
            return Err(Error::typ("cannot order values of different kinds"));
        }
        let mut err = None;
        idx.sort_by(|&i, &j| {
            let o = match (&keys[i], &keys[j]) {
                (None, None) => Ordering::Equal,
                (None, Some(_)) => Ordering::Less,
                (Some(_), None) => Ordering::Greater,
                (Some(x), Some(y)) => match cmp_atoms(x, y) {
                    Ok(o) => o,
                    Err(e) => {
                        err = Some(e);
                        Ordering::Equal
                    }
                },
            };
            let o = o.then(i.cmp(&j));
            if down {
                o.reverse()
            } else {
                o
            }
        });
        if let Some(e) = err {
            return Err(e);
        }
    }
    let mut out: Vec<Int> = try_vec(n)?;
    out.extend(idx.iter().map(|&i| i as Int));
    Ok(Value::ints(out))
}

fn m_grade_up(ip: &mut Interp, a: Value) -> Result<Value> {
    grade(ip, a, false)
}
fn m_grade_down(ip: &mut Interp, a: Value) -> Result<Value> {
    grade(ip, a, true)
}

/// Same value for grouping and range: kinds must agree (1 and 1.0 differ in a mixed list).
fn same_key(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Int(x), Value::Int(y)) => x == y,
        (Value::Flt(x), Value::Flt(y)) => x == y || (x.is_nan() && y.is_nan()),
        (Value::Char(x), Value::Char(y)) => x == y,
        (Value::Str(x), Value::Str(y)) => x.0 == y.0,
        (Value::Sym(x), Value::Sym(y)) => x == y,
        (Value::Undefined, Value::Undefined) => true,
        _ => {
            if a.is_list() && b.is_list() {
                let n = a.len().unwrap_or(0);
                b.len() == Some(n) && (0..n).all(|i| same_key(&a.item(i), &b.item(i)))
            } else {
                false
            }
        }
    }
}

fn m_group(_: &mut Interp, a: Value) -> Result<Value> {
    if !a.is_iterable() {
        return Err(Error::typ("=: group needs a list"));
    }
    if a.is_empty() {
        return Ok(a);
    }
    let n = a.len().unwrap_or(0);
    let mut keys: Vec<Value> = Vec::new();
    let mut groups: Vec<Vec<Int>> = Vec::new();
    for i in 0..n {
        let x = a.item(i);
        match keys.iter().position(|k| same_key(k, &x)) {
            Some(g) => try_push(&mut groups[g], i as Int)?,
            None => {
                try_push(&mut keys, x)?;
                try_push(&mut groups, alloc::vec![i as Int])?;
            }
        }
    }
    let mut out = try_vec(groups.len())?;
    out.extend(groups.into_iter().map(Value::ints));
    Value::from_items(out)
}

fn m_negate(_: &mut Interp, a: Value) -> Result<Value> {
    match &a {
        Value::Ints(x) if x.rank() == 1 => {
            let mut v = try_vec(x.data.len())?;
            v.extend(x.data.iter().map(|i| i.wrapping_neg()));
            return Ok(Value::ints(v));
        }
        Value::Flts(x) if x.rank() == 1 => {
            let mut v = try_vec(x.data.len())?;
            v.extend(x.data.iter().map(|f| -*f));
            return Ok(Value::flts(v));
        }
        _ => {}
    }
    map_atoms(&a, &|x| match x {
        Value::Int(i) => Ok(Value::Int(i.wrapping_neg())),
        Value::Flt(f) => Ok(Value::Flt(-*f)),
        _ if x.is_empty() => Ok(x.clone()),
        _ => Err(Error::typ(format!("cannot apply - (Negate) to {}", x.describe()))),
    })
}

fn m_not(_: &mut Interp, a: Value) -> Result<Value> {
    if a.is_empty() {
        return Ok(Value::Int(1));
    }
    map_atoms(&a, &|x| {
        Ok(Value::bool(match x {
            Value::Int(i) => *i == 0,
            Value::Flt(f) => *f == 0.0,
            _ => x.is_empty(),
        }))
    })
}

fn m_range(_: &mut Interp, a: Value) -> Result<Value> {
    if !a.is_iterable() {
        return Ok(a);
    }
    let n = a.len().unwrap_or(0);
    let mut out: Vec<Value> = Vec::new();
    for i in 0..n {
        let x = a.item(i);
        if !out.iter().any(|k| same_key(k, &x)) {
            try_push(&mut out, x)?;
        }
    }
    rebuild(&a, out)
}

fn m_reciprocal(_: &mut Interp, a: Value) -> Result<Value> {
    if let Some(n) = num(&a) {
        if n.is_zero() {
            return Ok(Value::Undefined);
        }
    }
    map_atoms(&a, &|x| match num(x) {
        Some(n) => Ok(Value::Flt(1.0 / n.f())),
        None if x.is_empty() => Ok(x.clone()),
        None => Err(Error::range(format!("cannot apply % (Reciprocal) to {}", x.describe()))),
    })
}

fn m_size(ip: &mut Interp, a: Value) -> Result<Value> {
    Ok(match &a {
        Value::Int(i) => Value::Int(i.wrapping_abs()),
        Value::Flt(f) => Value::Flt(fabs(*f)),
        Value::Char(c) => Value::Int(*c as Int),
        Value::Dict(d) => Value::Int(d.borrow().keys.len() as Int),
        Value::Sym(s) => Value::Int(ip.syms.name(*s).len() as Int),
        _ => match a.len() {
            Some(n) => Value::Int(n as Int),
            None => return Err(Error::typ(format!("cannot apply # (Size) to {}", a.describe()))),
        },
    })
}

fn m_undefined(_: &mut Interp, a: Value) -> Result<Value> {
    Ok(Value::bool(matches!(a, Value::Undefined)))
}

fn shape_of(a: &Value) -> Vec<Int> {
    match a {
        Value::Str(s) => alloc::vec![s.0.len() as Int],
        Value::Ints(x) => (0..x.rank()).map(|i| x.dim(i) as Int).collect(),
        Value::Flts(x) => (0..x.rank()).map(|i| x.dim(i) as Int).collect(),
        Value::List(v) => {
            let n = v.len();
            let mut out = alloc::vec![n as Int];
            if n > 0 {
                // each element's shape is computed once (twice is exponential in the depth)
                let first = shape_of(&v[0]);
                if !first.is_empty() && v[1..].iter().all(|q| shape_of(q) == first) {
                    out.extend(first);
                }
            }
            out
        }
        _ => Vec::new(),
    }
}

fn m_shape(_: &mut Interp, a: Value) -> Result<Value> {
    if a.is_atom() {
        return Ok(Value::Int(0));
    }
    Ok(Value::ints(shape_of(&a)))
}

// ---------------------------------------------------------------------------------------
// Dyads
// ---------------------------------------------------------------------------------------

fn whole(v: &Value) -> Option<Int> {
    match v {
        Value::Int(i) => Some(*i),
        _ => None,
    }
}

/// An index into something of length `n`: negative counts from the end.
fn norm_index(i: Int, n: usize, what: &str) -> Result<usize> {
    let n = n as i128;
    let i = i as i128;
    if i < -n || i >= n {
        return Err(Error::range(format!("index {} out of range for {} of {}", i, what, n)));
    }
    Ok((if i < 0 { i + n } else { i }) as usize)
}

fn index(a: &Value, b: &Value) -> Result<Value> {
    let n = a.len().unwrap_or(0);
    match b {
        Value::Int(i) => Ok(a.item(norm_index(*i, n, if a.is_str() { "string" } else { "list" })?)),
        Value::Flt(_) => Err(Error::range("an index must be an integer")),
        _ if b.is_list() => {
            let k = b.len().unwrap_or(0);
            let mut out = try_vec(k)?;
            for j in 0..k {
                out.push(index(a, &b.item(j))?);
            }
            rebuild(a, out)
        }
        _ => Ok(a.clone()),
    }
}

fn d_at(ip: &mut Interp, a: Value, b: Value) -> Result<Value> {
    match &a {
        Value::Fn(_) | Value::Sym(_) => {
            let args = if b.is_list() { b.items()? } else { alloc::vec![b] };
            ip.apply(&a, args, Some(1))
        }
        Value::Dict(d) => {
            let d = d.borrow();
            if b.is_list() {
                let n = b.len().unwrap_or(0);
                let mut out = try_vec(n)?;
                for i in 0..n {
                    out.push(d.get(&b.item(i)).cloned().unwrap_or(Value::Undefined));
                }
                return Value::from_items(out);
            }
            Ok(d.get(&b).cloned().unwrap_or(Value::Undefined))
        }
        _ if a.is_iterable() => {
            if b.is_list() && b.is_empty() {
                return Ok(if a.is_str() { Value::str(Vec::new()) } else { Value::empty() });
            }
            index(&a, &b)
        }
        _ => {
            if b.is_list() || matches!(b, Value::Int(_)) {
                Err(Error::typ(format!("cannot apply @ (Index/Apply) to {} and {}", a.describe(), b.describe())))
            } else {
                Ok(a)
            }
        }
    }
}

fn amend_index(v: &Value, n: usize, allow_end: bool) -> Result<usize> {
    let i = match v {
        Value::Int(i) => *i as i128,
        Value::Flt(f) if m::trunc(*f) == *f && f.is_finite() => *f as i128,
        _ => return Err(Error::typ(":=: index must be an integer")),
    };
    let j = if i < 0 { i + n as i128 } else { i };
    if j < 0 || j > n as i128 || (j == n as i128 && !allow_end) {
        return Err(Error::range(format!(":=: index {} out of range for length {}", i, n)));
    }
    Ok(j as usize)
}

fn d_amend(_: &mut Interp, a: Value, b: Value) -> Result<Value> {
    if !a.is_iterable() {
        return Err(Error::typ(":=: a must be list or str"));
    }
    let bn = match b.len() {
        Some(n) if b.is_list() => n,
        _ => return Err(Error::typ(":=: the right operand must be a list of a value and indexes")),
    };
    if bn <= 1 {
        return Ok(a);
    }
    let v = b.item(0);
    if let Value::Str(s) = &a {
        let ins = match text_of(&v) {
            Some(t) => t,
            None => return Err(Error::typ(":=: value for a string must be a character or string")),
        };
        let mut r = s.0.clone();
        for k in 1..bn {
            let n = r.len();
            let i = amend_index(&b.item(k), s.0.len(), true)?;
            let end = (i + ins.len()).min(n);
            let mut next = try_vec(n + ins.len())?;
            next.extend_from_slice(&r[..i.min(n)]);
            next.extend_from_slice(&ins);
            next.extend_from_slice(&r[end..]);
            r = next;
        }
        return Ok(Value::str(r));
    }
    let mut items = a.items()?;
    let n = items.len();
    for k in 1..bn {
        let i = amend_index(&b.item(k), n, false)?;
        items[i] = v.clone();
    }
    Value::from_items(items)
}

fn amend_depth(p: &Value, path: &[Value], v: &Value) -> Result<Value> {
    if !p.is_list() {
        return Err(Error::typ(":-: cannot amend inside an atom"));
    }
    let i = match whole(&path[0]) {
        Some(i) => norm_index(i, p.len().unwrap_or(0), "list")?,
        None => return Err(Error::typ(":-: index must be an integer")),
    };
    let mut items = p.items()?;
    items[i] = if path.len() > 1 { amend_depth(&items[i], &path[1..], v)? } else { v.clone() };
    Value::from_items(items)
}

fn d_amend_depth(_: &mut Interp, a: Value, b: Value) -> Result<Value> {
    if !b.is_list() || b.len().unwrap_or(0) < 2 {
        return Err(Error::typ(":-: the right operand must be a list of a value and a path"));
    }
    let items = b.items()?;
    amend_depth(&a, &items[1..], &items[0])
}

/// Python `b[lo:hi]` on a list or string.
fn slice(b: &Value, lo: usize, hi: usize) -> Result<Value> {
    let n = b.len().unwrap_or(0);
    let hi = hi.min(n);
    let lo = lo.min(hi);
    match b {
        Value::Ints(x) if x.rank() == 1 => {
            let mut v = try_vec(hi - lo)?;
            v.extend_from_slice(&x.data[lo..hi]);
            Ok(Value::ints(v))
        }
        Value::Flts(x) if x.rank() == 1 => {
            let mut v = try_vec(hi - lo)?;
            v.extend_from_slice(&x.data[lo..hi]);
            Ok(Value::flts(v))
        }
        Value::Str(s) => {
            let mut v = try_vec(hi - lo)?;
            v.extend_from_slice(&s.0[lo..hi]);
            Ok(Value::str(v))
        }
        _ => {
            let mut v = try_vec(hi - lo)?;
            for i in lo..hi {
                v.push(b.item(i));
            }
            Value::from_items(v)
        }
    }
}

fn d_drop(_: &mut Interp, a: Value, b: Value) -> Result<Value> {
    if let Value::Dict(d) = &b {
        let mut dm = d.borrow_mut();
        if let Some(i) = dm.find(&a) {
            dm.keys.remove(i);
            dm.vals.remove(i);
        }
        drop(dm);
        return Ok(b);
    }
    let k = match a {
        Value::Int(i) => i,
        _ => return Err(Error::typ(format!("cannot apply _ (Drop) to {} and {}", a.describe(), b.describe()))),
    };
    let n = match b.len() {
        Some(n) => n,
        None => return Err(Error::typ(format!("cannot apply _ (Drop) to {} and {}", a.describe(), b.describe()))),
    };
    if k >= 0 {
        slice(&b, (k as u128).min(n as u128) as usize, n)
    } else {
        slice(&b, 0, n.saturating_sub(k.unsigned_abs() as usize))
    }
}

fn d_index_depth(ip: &mut Interp, a: Value, b: Value) -> Result<Value> {
    if b.is_empty() {
        return Ok(b);
    }
    let path = if b.is_list() { b.items()? } else { alloc::vec![b] };
    if path.iter().all(|p| p.is_list()) {
        // numpy indexing with one index list per axis: the lists are walked together
        let n = path[0].len().unwrap_or(0);
        if !path.iter().all(|p| p.len() == Some(n)) {
            return Err(Error::shape(":@: index lists must have the same length"));
        }
        let mut out = try_vec(n)?;
        for k in 0..n {
            let mut one = try_vec(path.len())?;
            for p in &path {
                one.push(p.item(k));
            }
            out.push(d_index_depth(ip, a.clone(), Value::from_items(one)?)?);
        }
        return Value::from_items(out);
    }
    let mut cur = a;
    for p in &path {
        let i = match whole(p) {
            Some(i) => i,
            None => return Err(Error::range(":@: index must be an integer")),
        };
        if !cur.is_iterable() {
            return Err(Error::range(":@: too many indexes"));
        }
        cur = cur.item(norm_index(i, cur.len().unwrap_or(0), "list")?);
    }
    Ok(cur)
}

/// Python slice bound for an index that may be negative or out of range.
fn clamp_bound(i: Int, n: usize) -> usize {
    if i < 0 {
        (n as i128 + i as i128).max(0) as usize
    } else {
        (i as u128).min(n as u128) as usize
    }
}

fn d_cut(_: &mut Interp, a: Value, b: Value) -> Result<Value> {
    if !b.is_iterable() {
        return Err(Error::typ(":_: cut needs a list or a string"));
    }
    // numpy.array_split at positions: one piece more than there are positions
    let points = if a.is_list() { a.items()? } else { alloc::vec![a] };
    let n = b.len().unwrap_or(0);
    let mut out = try_vec(points.len() + 1)?;
    let mut prev: Int = 0;
    for p in points.iter() {
        let i = match whole(p) {
            Some(i) => i,
            None => return Err(Error::typ(":_: cut positions must be integers")),
        };
        out.push(slice(&b, clamp_bound(prev, n), clamp_bound(i, n))?);
        prev = i;
    }
    out.push(slice(&b, clamp_bound(prev, n), n)?);
    if n == 0 && !points.is_empty() {
        out.remove(0);
    }
    if b.is_str() {
        return Ok(Value::raw_list(out));
    }
    Value::from_items(out)
}

fn d_find(_: &mut Interp, a: Value, b: Value) -> Result<Value> {
    match &a {
        Value::Str(s) => {
            let sub = match text_of(&b) {
                Some(t) => t,
                None => return Err(Error::typ("?: find in a string needs a string or a char")),
            };
            let mut out: Vec<Int> = Vec::new();
            if !sub.is_empty() && sub.len() <= s.0.len() {
                for i in 0..=(s.0.len() - sub.len()) {
                    if s.0[i..i + sub.len()] == sub[..] {
                        try_push(&mut out, i as Int)?;
                    }
                }
            }
            Ok(Value::ints(out))
        }
        Value::Dict(d) => Ok(d.borrow().get(&b).cloned().unwrap_or(Value::Undefined)),
        _ if a.is_list() => {
            let n = a.len().unwrap_or(0);
            let mut out: Vec<Int> = Vec::new();
            for i in 0..n {
                let x = a.item(i);
                let hit = if b.is_list() { matches(&x, &b) } else { !x.is_list() && atom_eq(&x, &b) };
                if hit {
                    try_push(&mut out, i as Int)?;
                }
            }
            Ok(Value::ints(out))
        }
        _ => Ok(Value::ints(if atom_eq(&a, &b) { alloc::vec![0] } else { Vec::new() })),
    }
}

fn d_join(_: &mut Interp, a: Value, b: Value) -> Result<Value> {
    if let (Some(x), Some(y)) = (text_of(&a), text_of(&b)) {
        let mut v = try_vec(x.len() + y.len())?;
        v.extend_from_slice(&x);
        v.extend_from_slice(&y);
        return Ok(Value::str(v));
    }
    if let Value::Dict(d) = &a {
        if let Value::Dict(e) = &b {
            if !Rc::ptr_eq(d, e) {
                let e = e.borrow();
                let mut dm = d.borrow_mut();
                for i in 0..e.keys.len() {
                    dm.set(e.keys[i].clone(), e.vals[i].clone());
                }
            }
            return Ok(a);
        }
        if !(b.is_list() && b.len() == Some(2)) {
            return Err(Error::typ(",: a dictionary can only be joined with a tuple (a list of two members) or a dictionary"));
        }
        d.borrow_mut().set(b.item(0), b.item(1));
        return Ok(a);
    }
    if let Value::Dict(d) = &b {
        if a.is_list() && a.len() == Some(2) {
            d.borrow_mut().set(a.item(0), a.item(1));
            return Ok(b);
        }
    }
    match (&a, &b) {
        (Value::Ints(x), Value::Ints(y)) if x.rank() == 1 && y.rank() == 1 => {
            let mut v = try_vec(x.data.len() + y.data.len())?;
            v.extend_from_slice(&x.data);
            v.extend_from_slice(&y.data);
            return Ok(Value::ints(v));
        }
        (Value::Ints(x), Value::Int(y)) if x.rank() == 1 => {
            let mut v = try_vec(x.data.len() + 1)?;
            v.extend_from_slice(&x.data);
            v.push(*y);
            return Ok(Value::ints(v));
        }
        (Value::Flts(x), Value::Flts(y)) if x.rank() == 1 && y.rank() == 1 => {
            let mut v = try_vec(x.data.len() + y.data.len())?;
            v.extend_from_slice(&x.data);
            v.extend_from_slice(&y.data);
            return Ok(Value::flts(v));
        }
        _ => {}
    }
    if a.is_list() && b.is_list() && a.is_empty() {
        return Ok(b);
    }
    let an = if a.is_list() { a.len().unwrap_or(0) } else { 1 };
    let bn = if b.is_list() { b.len().unwrap_or(0) } else { 1 };
    let mut out = try_vec(an + bn)?;
    if a.is_list() {
        for i in 0..an {
            out.push(a.item(i));
        }
    } else {
        out.push(a.clone());
    }
    if b.is_list() {
        for i in 0..bn {
            out.push(b.item(i));
        }
    } else {
        out.push(b.clone());
    }
    // Two arrays of rank above 1 with the same row shape concatenate; otherwise the rows and
    // atoms form a list, which from_items normalises as numpy.asarray does.
    Value::from_items(out)
}

fn d_match(_: &mut Interp, a: Value, b: Value) -> Result<Value> {
    Ok(Value::bool(matches(&a, &b)))
}

/// The scalars of a list in row-major order (one level for a mixed list).
fn flatten(b: &Value) -> Result<Vec<Value>> {
    match b {
        Value::Ints(x) => {
            let mut v = try_vec(x.data.len())?;
            v.extend(x.data.iter().map(|i| Value::Int(*i)));
            Ok(v)
        }
        Value::Flts(x) => {
            let mut v = try_vec(x.data.len())?;
            v.extend(x.data.iter().map(|f| Value::Flt(*f)));
            Ok(v)
        }
        _ => b.items(),
    }
}

/// `flat` (used cyclically) arranged in the shape `dims`.
fn build_shape(flat: &[Value], dims: &[usize], at: &mut usize, is_str: bool) -> Result<Value> {
    if dims.len() == 1 {
        let mut row = try_vec(dims[0])?;
        for _ in 0..dims[0] {
            row.push(flat[*at % flat.len()].clone());
            *at += 1;
        }
        if is_str {
            return rebuild(&Value::str(Vec::new()), row);
        }
        return Value::from_items(row);
    }
    let mut rows = try_vec(dims[0])?;
    for _ in 0..dims[0] {
        rows.push(build_shape(flat, &dims[1..], at, is_str)?);
    }
    if is_str {
        return Ok(Value::raw_list(rows));
    }
    Value::from_items(rows)
}

fn d_reshape(_: &mut Interp, a: Value, b: Value) -> Result<Value> {
    let is_str = b.is_str();
    if a.is_list() {
        let mut dims: Vec<usize> = Vec::new();
        let b_size = if b.is_iterable() { flatten(&b)?.len() } else { 1 };
        for q in a.items()? {
            match q {
                Value::Int(i) if i < -1 => return Err(Error::range(":^: only -1 may be negative in a shape")),
                Value::Int(-1) => {
                    if !b.is_iterable() {
                        return Err(Error::range(":^: -1 in a shape needs a list as the source, not an atom"));
                    }
                    dims.push(b_size / 2)
                }
                Value::Int(i) => dims.push(i as usize),
                _ => return Err(Error::typ(":^: a shape must hold integers")),
            }
        }
        if dims.is_empty() {
            return Ok(b);
        }
        let total = dims.iter().try_fold(1usize, |p, &d| p.checked_mul(d)).ok_or_else(Error::memory)?;
        let flat = if b.is_iterable() { flatten(&b)? } else { alloc::vec![b.clone()] };
        if flat.is_empty() && total > 0 {
            return Err(Error::range(":^: cannot fill a non-empty shape from an empty source"));
        }
        if total == 0 {
            return Ok(Value::empty());
        }
        // (a memory check before building: the shape is data)
        let _probe: Vec<Value> = try_vec(total)?;
        let mut at = 0;
        return build_shape(&flat, &dims, &mut at, is_str);
    }
    let n = match a {
        Value::Int(i) if i < 0 => return Err(Error::range(":^: size must not be negative")),
        Value::Int(i) => i as usize,
        _ => return Err(Error::typ(":^: a size must be an integer")),
    };
    if n == 0 {
        return Ok(b);
    }
    if !b.is_iterable() {
        let mut v = try_vec(n)?;
        for _ in 0..n {
            v.push(b.clone());
        }
        return Value::from_items(v);
    }
    let len = b.len().unwrap_or(0);
    if len == 0 {
        return Err(Error::range(":^: cannot fill a non-empty shape from an empty source"));
    }
    // fewer than the rows there are: the first n scalars; otherwise the rows, cyclically
    let src = if n < len { flatten(&b)? } else { b.items()? };
    let mut v = try_vec(n)?;
    for i in 0..n {
        v.push(src[i % src.len()].clone());
    }
    rebuild(&b, v)
}

fn d_rotate(_: &mut Interp, a: Value, b: Value) -> Result<Value> {
    let k = match a {
        Value::Int(i) => i,
        _ => return Err(Error::typ(":+: rotate needs an integer")),
    };
    if k == 0 || !b.is_iterable() {
        return Ok(b);
    }
    let n = b.len().unwrap_or(0);
    if n == 0 {
        return Ok(b);
    }
    let shift = (k as i128).rem_euclid(n as i128) as usize;
    let items = b.items()?;
    let mut out = try_vec(n)?;
    for i in 0..n {
        out.push(items[(i + n - shift) % n].clone());
    }
    rebuild(&b, out)
}

fn d_split(_: &mut Interp, a: Value, b: Value) -> Result<Value> {
    let n = match b.len() {
        Some(n) => n,
        None => return Err(Error::typ(":#: split needs a list or a string")),
    };
    if n == 0 {
        return Ok(Value::empty());
    }
    let raw = if a.is_list() { a.items()? } else { alloc::vec![a] };
    if raw.is_empty() {
        return Err(Error::range(":#: segment sizes must not be empty"));
    }
    let mut sizes: Vec<usize> = Vec::new();
    for q in &raw {
        let s = match q {
            Value::Int(i) => *i,
            Value::Flt(f) if f.is_finite() && m::trunc(*f) == *f => *f as Int,
            _ => return Err(Error::typ(":#: segment sizes must be integers")),
        };
        if s < 1 {
            return Err(Error::range(":#: segment sizes must be positive"));
        }
        sizes.push(s as usize);
    }
    let mut parts: Vec<Value> = Vec::new();
    if sizes.len() == 1 {
        let s = sizes[0];
        if s >= n {
            parts.push(b.clone());
        } else {
            // numpy.array_split into ceil(n/s) sections: the first n mod k sections get one
            // element more than the others.
            let k = n.div_ceil(s);
            let each = n / k;
            let extras = n % k;
            let mut at = 0;
            for i in 0..k {
                let len = each + (i < extras) as usize;
                parts.push(slice(&b, at, at + len)?);
                at += len;
            }
        }
    } else {
        let mut q = 0;
        let mut p = 0;
        while q < n {
            parts.push(slice(&b, q, q + sizes[p])?);
            q += sizes[p];
            p = (p + 1) % sizes.len();
        }
    }
    if b.is_str() {
        return Ok(Value::raw_list(parts));
    }
    Value::from_items(parts)
}

fn d_take(_: &mut Interp, a: Value, b: Value) -> Result<Value> {
    let k = match a {
        Value::Int(i) => i,
        _ => return Err(Error::typ(format!("cannot apply # (Take) to {} and {}", a.describe(), b.describe()))),
    };
    let b = if b.is_iterable() { b } else { Value::from_items(alloc::vec![b])? };
    let n = b.len().unwrap_or(0);
    if n == 0 {
        return Ok(b);
    }
    let count = k.unsigned_abs() as usize;
    if count <= n {
        return if k < 0 { slice(&b, n - count, n) } else { slice(&b, 0, count) };
    }
    let mut out = try_vec(count)?;
    // the list repeated: forwards from its start, or backwards from its end
    let start = if k >= 0 { 0 } else { n - count % n };
    let items = b.items()?;
    for i in 0..count {
        out.push(items[(start + i) % n].clone());
    }
    rebuild(&b, out)
}

/// A deep copy of a value that holds dictionaries (a dictionary literal is copied each time
/// it is evaluated).
pub fn deep_copy(v: &Value) -> Result<Value> {
    match v {
        Value::Dict(d) => {
            let d = d.borrow();
            let mut n = Dict::default();
            for i in 0..d.keys.len() {
                n.keys.push(d.keys[i].clone());
                n.vals.push(deep_copy(&d.vals[i])?);
            }
            Ok(Value::Dict(Rc::new(RefCell::new(n))))
        }
        Value::List(items) => {
            let mut out = try_vec(items.len())?;
            for q in items.iter() {
                out.push(deep_copy(q)?);
            }
            Ok(Value::raw_list(out))
        }
        _ => Ok(v.clone()),
    }
}
