//! Operators: one table (glyph, monad, dyad, names, atomic flag) and their implementations.
//!
//! All atomic dyads share one pairing routine, [`atomic`], which implements KlongPy's rule
//! (`vec_fn2` in `klongpy/backends/base.py`): scalar extension, recursive pairing of lists,
//! leading-dimension agreement for arrays. Loops are plain. Integer arithmetic wraps at the
//! build width.
use alloc::rc::Rc;
use alloc::vec::Vec;
use core::cmp::Ordering;

use crate::error::{Error, Kind, Result};
use crate::interp::Interp;
use crate::sym::{Interner, SymId, FIRST_OP};
use crate::value::{atom_eq, matches, try_push, try_push_char, try_push_str, try_str, try_string, try_vec, Arr, Dict, Flt, Int, Value};

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

fn absent_m(_ip: &mut Interp, _x: Value) -> Result<Value> {
    #[cfg(feature = "host")]
    if let Some(h) = _ip.host_op.clone() {
        return h(_ip, "\u{2207}", &[_x]);
    }
    Err(Error::name("operator not available on this build").with_hint("automatic differentiation is desktop only"))
}
macro_rules! absent_dyad {
    ($name:ident, $glyph:expr) => {
        fn $name(_ip: &mut Interp, _x: Value, _y: Value) -> Result<Value> {
            #[cfg(feature = "host")]
            if let Some(h) = _ip.host_op.clone() {
                return h(_ip, $glyph, &[_x, _y]);
            }
            Err(Error::name("operator not available on this build").with_hint("automatic differentiation is desktop only"))
        }
    };
}
absent_dyad!(absent_grad, "\u{2207}");
absent_dyad!(absent_jac, "\u{2202}");
absent_dyad!(absent_auto, ":>");
fn define_d(_: &mut Interp, _: Value, y: Value) -> Result<Value> {
    // `::` is evaluated by the interpreter (its left operand is not evaluated); as a function
    // value applied to two values it binds nothing.
    Ok(y)
}

macro_rules! at {
    ($name:ident, $op:expr) => {
        fn $name(ip: &mut Interp, a: Value, b: Value) -> Result<Value> {
            atomic(&ip.syms, $op, &a, &b)
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
    op!("$", "Format", Some(m_format), "Format2", Some(d_format2), None),
    op!("~", "Not", Some(m_not), "Match", Some(d_match), None),
    op!("?", "Range", Some(m_range), "Find", Some(d_find), None),
    op!("#", "Size", Some(m_size), "Take", Some(d_take), None),
    op!(":_", "Undefined", Some(m_undefined), "Cut", Some(d_cut), None),
    op!(":=", "", None, "Amend", Some(d_amend), None),
    op!(":-", "", None, "Amend-in-Depth", Some(d_amend_depth), None),
    op!(":@", "", None, "Index-in-Depth", Some(d_index_depth), None),
    op!(":$", "", None, "Form", Some(d_form), None),
    op!(":^", "", None, "Reshape", Some(d_reshape), None),
    op!(":+", "", None, "Rotate", Some(d_rotate), None),
    op!("::", "", None, "Define", Some(define_d), None),
    // Desktop only (autograd): known to the parser so that trees agree; absent at run time.
    op!("\u{2207}", "Grad", Some(absent_m), "Grad", Some(absent_grad), None),
    op!("\u{2202}", "", None, "Jacobian", Some(absent_jac), None),
    op!(":>", "", None, "Autograd", Some(absent_auto), None),
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
        // a host build (the Python extension) takes the operating system's `pow`, as Python and NumPy do: the portable one differs in
        // the last digit for some operands
        #[cfg(feature = "host")]
        return a.powf(b);
        #[cfg(not(feature = "host"))]
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

/// `base^exp` for `exp >= 0` when it fits the build width.
#[cfg(feature = "bigint")]
fn int_pow_checked(mut base: Int, mut exp: Int) -> Option<Int> {
    let mut r: Int = 1;
    while exp > 0 {
        if exp & 1 == 1 {
            r = r.checked_mul(base)?;
        }
        exp >>= 1;
        if exp > 0 {
            base = base.checked_mul(base)?;
        }
    }
    Some(r)
}

fn flt_to_int(f: Flt) -> Result<Int> {
    if f.is_nan() || f.is_infinite() {
        return Err(Error::range("cannot convert a non-finite number to an integer"));
    }
    Ok(f as Int)
}

/// One pair of numbers. `None` is `:undefined` (a zero divisor).
#[cfg_attr(feature = "ram-hot", allow(unsafe_code), link_section = ".time_critical.klong_num_op")]
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
    let (g, n) = op_glyph(op);
    Error::typ(crate::msg!("cannot apply {} ({}) to {} and {}", g, n, a.describe(), b.describe())).generic()
}

fn op_glyph(op: Atomic) -> (&'static str, &'static str) {
    match op {
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
    }
}

/// The text of a string or of a char, without copying a string.
pub(crate) struct Txt<'a> {
    s: Option<&'a str>,
    buf: [u8; 4],
    n: u8,
}

impl Txt<'_> {
    pub(crate) fn as_str(&self) -> &str {
        match self.s {
            Some(s) => s,
            None => core::str::from_utf8(&self.buf[..self.n as usize]).unwrap_or(""),
        }
    }
}

pub(crate) fn text_of(v: &Value) -> Option<Txt<'_>> {
    match v {
        Value::Str(s) => Some(Txt { s: Some(s.as_str()), buf: [0; 4], n: 0 }),
        Value::Char(c) => {
            let mut buf = [0u8; 4];
            let n = c.encode_utf8(&mut buf).len() as u8;
            Some(Txt { s: None, buf, n })
        }
        _ => None,
    }
}

/// Two numbers under `+ - * < > =` (and `%` of floats by a divisor that is not zero), the answer the pairing routine gives, without the
/// routine; `None` for any other pair, which it does.
#[cfg(feature = "speed")]
#[inline]
pub(crate) fn scalar_pair(op: Option<Atomic>, a: &Value, b: &Value) -> Option<Value> {
    let op = op?;
    match (a, b) {
        (Value::Int(x), Value::Int(y)) => Some(match op {
            Atomic::Add => Value::Int(x.wrapping_add(*y)),
            Atomic::Sub => Value::Int(x.wrapping_sub(*y)),
            Atomic::Mul => Value::Int(x.wrapping_mul(*y)),
            Atomic::Less => Value::Int((x < y) as Int),
            Atomic::More => Value::Int((x > y) as Int),
            Atomic::Eq => Value::Int((x == y) as Int),
            _ => return None,
        }),
        (Value::Flt(x), Value::Flt(y)) => flt_pair(op, *x, *y),
        (Value::Int(x), Value::Flt(y)) => flt_pair(op, *x as Flt, *y),
        (Value::Flt(x), Value::Int(y)) => flt_pair(op, *x, *y as Flt),
        _ => None,
    }
}

#[cfg(feature = "speed")]
#[inline]
fn flt_pair(op: Atomic, x: Flt, y: Flt) -> Option<Value> {
    Some(match op {
        Atomic::Add => Value::Flt(x + y),
        Atomic::Sub => Value::Flt(x - y),
        Atomic::Mul => Value::Flt(x * y),
        Atomic::Div if y != 0.0 => Value::Flt(x / y),
        Atomic::Less => Value::Int((x < y) as Int),
        Atomic::More => Value::Int((x > y) as Int),
        Atomic::Eq => Value::Int((x == y) as Int),
        _ => return None,
    })
}

/// One pair of atoms (anything that is not a list).
// (not inlined: `atomic` recurses once per level of nesting and must stay small on the stack)
#[inline(never)]
#[cfg_attr(feature = "ram-hot", allow(unsafe_code), link_section = ".time_critical.klong_atom_op")]
fn atom_op(syms: &Interner, op: Atomic, a: &Value, b: &Value) -> Result<Value> {
    if let (Some(x), Some(y)) = (num(a), num(b)) {
        if op == Atomic::Pow {
            // An integer to a non-negative integer power is an integer: exact when it fits,
            // wrapping at the build width when it does not, as `+ - *` do.
            if let (Num::I(i), Num::I(j)) = (x, y) {
                if j >= 0 {
                    #[cfg(feature = "bigint")]
                    {
                        // exact beyond the build width (KlongPy: scalar `^` of two integers is exact)
                        return match int_pow_checked(i, j) {
                            Some(r) => Ok(Value::Int(r)),
                            None => match crate::big::op(Atomic::Pow, a, b)? {
                                Some(v) => Ok(v),
                                None => Err(Error::range(crate::msg!("cannot apply ^ (Power) to integer and integer")).generic()),
                            },
                        };
                    }
                    #[cfg(not(feature = "bigint"))]
                    return Ok(Value::Int(int_pow(i, j)));
                }
            }
            // KlongPy: any other power whose result is a whole number is an integer.
            let r = m::pow(x.f(), y.f());
            if r.is_finite() && m::trunc(r) == r {
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
    #[cfg(feature = "bigint")]
    if (a.is_big() || b.is_big()) && matches!(a, Value::Int(_) | Value::Flt(_) | Value::Big(_)) && matches!(b, Value::Int(_) | Value::Flt(_) | Value::Big(_)) {
        return match crate::big::op(op, a, b)? {
            Some(v) => Ok(v),
            None => Err(Error::range(crate::msg!("cannot apply {} ({}) to {} and {}", op_glyph(op).0, op_glyph(op).1, a.describe(), b.describe())).generic()),
        };
    }
    match op {
        Atomic::Eq => Ok(Value::bool(atom_eq(a, b))),
        Atomic::Less | Atomic::More => {
            // text orders by code point; a symbol orders as its name (KlongPy's symbols are
            // strings), also against a string
            let name = |v: &Value| match v {
                Value::Sym(s) => Some(syms.name(*s)),
                _ => None,
            };
            let (ta, tb) = (text_of(a), text_of(b));
            let x = name(a).or(ta.as_ref().map(|t| t.as_str()));
            let y = name(b).or(tb.as_ref().map(|t| t.as_str()));
            let ord = match (x, y) {
                (Some(x), Some(y)) => x.cmp(y),
                _ => return Err(type_error(op, a, b)),
            };
            Ok(Value::bool(if op == Atomic::Less { ord == Ordering::Less } else { ord == Ordering::Greater }))
        }
        _ => Err(type_error(op, a, b)),
    }
}

fn shape_error(a: usize, b: usize) -> Error {
    Error::shape(crate::msg!("atomic dyad shape mismatch: length {} and length {}", a, b)).generic()
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

pub(crate) fn dims_of(v: &Value) -> (usize, usize, &[u32], usize) {
    // (rank, leading length, row dims, total scalars)
    match v {
        Value::Ints(a) => (a.rank(), a.len(), a.row_dims(), a.data.len()),
        Value::Flts(a) => (a.rank(), a.len(), a.row_dims(), a.data.len()),
        _ => (0, 0, &[], 1),
    }
}

pub(crate) fn shaped_like(like: &Value, ints: Option<Vec<Int>>, flts: Option<Vec<Flt>>) -> Value {
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
#[inline(never)]
fn array_op(syms: &Interner, op: Atomic, a: &Value, b: &Value) -> Result<Value> {
    #[cfg(feature = "speed")]
    if let Some(v) = crate::fast::array(op, a, b)? {
        return Ok(v);
    }
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
                return Err(Error::shape("atomic dyad shape mismatch").generic());
            }
            if pr >= qr {
                (p, pn, 1, if qn == 0 { 1 } else { pn / qn })
            } else {
                (q, qn, if pn == 0 { 1 } else { qn / pn }, 1)
            }
        }
        (Some(p), None) => (p, dims_of(p).3, 1, 1),
        (None, Some(q)) => (q, dims_of(q).3, 1, 1),
        (None, None) => return atom_op(syms, op, a, b),
    };
    let both_int = x.is_int() && y.is_int();
    let int_result = match op {
        Atomic::Less | Atomic::More | Atomic::Eq | Atomic::IDiv => true,
        Atomic::Add | Atomic::Sub | Atomic::Mul | Atomic::Min | Atomic::Max | Atomic::Rem => both_int,
        Atomic::Pow => both_int && matches!(&y, Operand::Ints(_) | Operand::One(Num::I(_))) && (0..n).all(|i| matches!(y.get(i / ydiv), Num::I(e) if e >= 0)),
        Atomic::Div => false,
    };
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
        if op == Atomic::Pow && matches!((&x, &y), (Operand::Ints(_), Operand::One(Num::I(2)))) {
            if let Operand::Ints(p) = &x {
                out.extend(p.iter().map(|a| a.wrapping_mul(*a)));
            }
        } else if op == Atomic::Pow {
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
                (Operand::One(Num::I(q)), Operand::Ints(p), Atomic::Add) => out.extend(p.iter().map(|a| q.wrapping_add(*a))),
                (Operand::One(Num::I(q)), Operand::Ints(p), Atomic::Mul) => out.extend(p.iter().map(|a| q.wrapping_mul(*a))),
                (Operand::Ints(p), Operand::Ints(q), Atomic::Sub) if xdiv == 1 && ydiv == 1 => out.extend(p.iter().zip(q.iter()).map(|(a, b)| a.wrapping_sub(*b))),
                (Operand::Ints(p), Operand::Ints(q), Atomic::Mul) if xdiv == 1 && ydiv == 1 => out.extend(p.iter().zip(q.iter()).map(|(a, b)| a.wrapping_mul(*b))),
                (Operand::Ints(p), Operand::One(Num::I(q)), Atomic::Sub) => out.extend(p.iter().map(|a| a.wrapping_sub(*q))),
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
    // A power with a list operand that is not integers to non-negative integers is floats,
    // whole-valued or not, as `*` is (KlongPy); only the scalar power narrows a whole result.
    let mut out: Vec<Flt> = try_vec(n)?;
    // the common float loops, written so that they vectorise (same-length operands or one scalar)
    macro_rules! fast {
        ($f:expr) => {{
            let f = $f;
            match (&x, &y) {
                (Operand::Flts(p), Operand::Flts(q)) if xdiv == 1 && ydiv == 1 => {
                    out.extend(p.iter().zip(q.iter()).map(|(a, b)| f(*a, *b)));
                    true
                }
                (Operand::Flts(p), Operand::One(q)) => {
                    let q = q.f();
                    out.extend(p.iter().map(|a| f(*a, q)));
                    true
                }
                (Operand::One(p), Operand::Flts(q)) => {
                    let p = p.f();
                    out.extend(q.iter().map(|b| f(p, *b)));
                    true
                }
                _ => false,
            }
        }};
    }
    let done = match op {
        Atomic::Add => fast!(|a: Flt, b: Flt| a + b),
        Atomic::Sub => fast!(|a: Flt, b: Flt| a - b),
        Atomic::Mul => fast!(|a: Flt, b: Flt| a * b),
        Atomic::Div => fast!(|a: Flt, b: Flt| a / b),
        Atomic::Pow if matches!(&y, Operand::One(Num::I(2))) => match &x {
            Operand::Flts(p) => {
                out.extend(p.iter().map(|a| a * a));
                true
            }
            _ => false,
        },
        _ => false,
    };
    if done {
        return Ok(shaped_like(like, None, Some(out)));
    }
    for i in 0..n {
        if let Some(v) = num_op(op, x.get(i / xdiv), y.get(i / ydiv))? {
            out.push(v.f());
        }
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
// One recursion per level of nesting: the body is kept small on the stack (one loop; the work
// on atoms and on numeric arrays is in functions that are not inlined).
#[cfg_attr(feature = "ram-hot", allow(unsafe_code), link_section = ".time_critical.klong_atomic")]
/// The pairing routine of every atomic dyad. A failure of the operator itself (not of something it called) names the whole operands,
/// as KlongPy's `operator_error` does: `cannot apply + (Plus) to list of 3 and list of 2`.
pub fn atomic(syms: &Interner, op: Atomic, a: &Value, b: &Value) -> Result<Value> {
    match atomic_in(syms, op, a, b) {
        Err(e) if e.generic => Err(whole_error(e, op_glyph(op), &[a, b], Some(syms))),
        r => r,
    }
}

/// `cannot apply G (Name) to A and B` for operands the operator does not take.
#[cold]
fn apply_err(kind: Kind, g: &str, n: &str, a: &Value, b: &Value) -> Error {
    #[cfg(feature = "diag")]
    return whole_error(Error::new(kind, ""), (g, n), &[a, b], None);
    #[cfg(not(feature = "diag"))]
    Error::new(kind, crate::msg!("cannot apply {} ({}) to {} and {}", g, n, a.describe(), b.describe()))
}

/// The error of an operator applied to whole operands (KlongPy `operator_error`).
#[cfg(not(feature = "diag"))]
pub(crate) fn whole_error(e: Error, _g: (&str, &str), _operands: &[&Value], _syms: Option<&Interner>) -> Error {
    e
}

#[cfg(feature = "diag")]
#[cold]
pub(crate) fn whole_error(e: Error, g: (&str, &str), operands: &[&Value], syms: Option<&Interner>) -> Error {
    let mut m = crate::msg!("cannot apply {} ({}) to ", g.0, g.1);
    for (i, v) in operands.iter().enumerate() {
        if i > 0 {
            m.push_str(" and ");
        }
        m.push_str(&v.describe_long());
    }
    if e.kind == Kind::Type {
        for v in operands {
            if let Some(odd) = v.odd_member(0, syms) {
                m.push_str(": the list holds ");
                m.push_str(&odd);
                break;
            }
        }
    }
    let mut out = Error::new(e.kind, m).generic();
    out.span = e.span;
    out.source = e.source.clone();
    out
}

fn atomic_in(syms: &Interner, op: Atomic, a: &Value, b: &Value) -> Result<Value> {
    let al = a.is_list();
    let bl = b.is_list();
    if !al && !bl {
        return atom_op(syms, op, a, b);
    }
    let mixed = matches!(a, Value::List(_)) || matches!(b, Value::List(_));
    if !mixed && (al || num(a).is_some()) && (bl || num(b).is_some()) {
        return array_op(syms, op, a, b);
    }
    // element by element: two lists pair up, an atom is repeated
    let n = if al { a.len().unwrap_or(0) } else { b.len().unwrap_or(0) };
    if al && bl && b.len() != Some(n) {
        return Err(shape_error(n, b.len().unwrap_or(0)));
    }
    let mut out = try_vec(n)?;
    for i in 0..n {
        let r = match (al, bl) {
            (true, true) => atomic_in(syms, op, &a.item(i), &b.item(i)),
            (true, false) => atomic_in(syms, op, &a.item(i), b),
            _ => atomic_in(syms, op, a, &b.item(i)),
        };
        out.push(r?);
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
    if a.is_atom() {
        return Ok(a);
    }
    // KlongPy: the first element of a string is the string of that character (Python's
    // s[0]), where "abc"@0 is the char. The suite pins it (strings.str-first-of-reverse).
    if let Value::Str(s) = &a {
        return Ok(Value::char_str(s.char_at(0)));
    }
    Ok(a.item(0))
}

/// A list or string with the same kind as `like` from its elements.
fn rebuild(like: &Value, items: Vec<Value>) -> Result<Value> {
    if like.is_str() {
        let mut s = try_string(items.len())?;
        for q in &items {
            match q {
                Value::Char(c) => try_push_char(&mut s, *c)?,
                Value::Str(t) => try_push_str(&mut s, t.as_str())?,
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
    #[cfg(feature = "speed")]
    if let Value::Flts(x) = &a {
        if x.rank() == 1 {
            let mut v = try_vec(x.data.len())?;
            v.extend(x.data.iter().rev().copied());
            return Ok(Value::flts(v));
        }
    }
    if let Value::Str(t) = &a {
        let mut s = try_string(t.as_str().len())?;
        s.extend(t.as_str().chars().rev());
        return Ok(Value::str(s));
    }
    let mut items = a.items()?;
    items.reverse();
    rebuild(&a, items)
}

fn m_where(_: &mut Interp, a: Value) -> Result<Value> {
    #[cfg(feature = "speed")]
    if let Value::Ints(x) = &a {
        if x.rank() == 1 {
            // a run of 0 and 1 (a comparison's answer): the indices of the ones, written without a branch on the data (an unpredictable
            // mask costs a misprediction per element)
            if x.data.iter().fold(0u64, |m, &c| m | (c as u64 >> 1)) == 0 {
                let total: usize = x.data.iter().map(|&c| c as usize).sum();
                let mut out: Vec<Int> = try_vec(total + 1)?;
                out.resize(total + 1, 0);
                let mut k = 0usize;
                for (i, &c) in x.data.iter().enumerate() {
                    out[k] = i as Int;
                    k += c as usize;
                }
                out.truncate(total);
                return Ok(Value::ints(out));
            }
            let mut total: usize = 0;
            for &c in x.data.iter() {
                if c < 0 {
                    return Err(Error::range("&: counts must not be negative"));
                }
                total = total.checked_add(c as usize).ok_or_else(Error::memory)?;
            }
            let mut out: Vec<Int> = try_vec(total)?;
            if x.data.iter().all(|&c| c <= 1) {
                // (a run of 0 and 1: the indices of the ones)
                out.extend(x.data.iter().enumerate().filter(|(_, &c)| c == 1).map(|(i, _)| i as Int));
            } else {
                for (i, &c) in x.data.iter().enumerate() {
                    for _ in 0..c {
                        out.push(i as Int);
                    }
                }
            }
            return Ok(Value::ints(out));
        }
    }
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
        return Ok(Value::char_str(c));
    }
    // (one number: the vector of one, without the list of values that `from_items` takes apart)
    #[cfg(feature = "speed")]
    match &a {
        Value::Int(i) => {
            let mut v = try_vec(1)?;
            v.push(*i);
            return Ok(Value::ints(v));
        }
        Value::Flt(f) => {
            let mut v = try_vec(1)?;
            v.push(*f);
            return Ok(Value::flts(v));
        }
        _ => {}
    }
    Value::from_items(alloc::vec![a])
}

fn m_char(_: &mut Interp, a: Value) -> Result<Value> {
    map_atoms(&a, &|x| match x {
        Value::Int(i) => match u32::try_from(*i).ok().and_then(char::from_u32) {
            Some(c) => Ok(Value::Char(c)),
            None => Err(Error::range(":#: not a character code (0 to 1114111, surrogates excluded)")),
        },
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
        _ => Err(Error::typ(crate::msg!("!: enumerate needs an integer, not {}", a.describe()))),
    }
}

fn m_floor(_: &mut Interp, a: Value) -> Result<Value> {
    if let Value::Ints(_) = a {
        return Ok(a);
    }
    #[cfg(feature = "speed")]
    if let Value::Flts(x) = &a {
        // (every element finite: otherwise the generic loop, which names the failure)
        if x.rank() == 1 && x.data.iter().all(|f| f.is_finite()) {
            let mut v = try_vec(x.data.len())?;
            v.extend(x.data.iter().map(|f| m::floor(*f) as Int));
            return Ok(Value::ints(v));
        }
    }
    map_atoms(&a, &|x| match x {
        Value::Int(i) => Ok(Value::Int(*i)),
        Value::Flt(f) => Ok(Value::Int(flt_to_int(m::floor(*f))?)),
        #[cfg(feature = "bigint")]
        Value::Big(b) => Ok(Value::Int(crate::big::saturate(b))),
        _ => Err(Error::range(crate::msg!("_: cannot floor {}", x.describe()))),
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
        Value::Char(c) => Ok(Value::char_str(*c)),
        _ => Ok(Value::str(crate::fmt::format_value(&ip.syms, &a, false)?)),
    }
}

fn cmp_atoms(a: &Value, b: &Value) -> Result<Ordering> {
    #[cfg(feature = "bigint")]
    if a.is_big() || b.is_big() {
        if matches!(a, Value::Int(_) | Value::Flt(_) | Value::Big(_)) && matches!(b, Value::Int(_) | Value::Flt(_) | Value::Big(_)) {
            return Ok(crate::big::cmp_num(a, b).unwrap_or(Ordering::Equal));
        }
    }
    match (a.as_flt(), b.as_flt()) {
        (Some(x), Some(y)) => Ok(match (a, b) {
            (Value::Int(i), Value::Int(j)) => i.cmp(j),
            _ => x.partial_cmp(&y).unwrap_or(Ordering::Equal),
        }),
        _ => match (text_of(a), text_of(b)) {
            (Some(x), Some(y)) => Ok(x.as_str().cmp(y.as_str())),
            _ => Err(Error::typ("cannot order values of different kinds")),
        },
    }
}

/// The sort key of an element (KlongPy `kg_argsort`): a list sorts by its largest element,
/// an empty one before everything.
fn grade_key(ip: &Interp, v: &Value) -> Result<Option<Value>> {
    if let Value::Sym(s) = v {
        // a symbol sorts as its name
        return Ok(Some(Value::text(ip.syms.name(*s))?));
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
        // (the grade of an empty string is an empty list)
        if a.is_str() && a.is_empty() {
            return Ok(Value::empty());
        }
        return Ok(a);
    }
    let n = a.len().unwrap_or(0);
    let mut idx: Vec<usize> = try_vec(n)?;
    idx.extend(0..n);
    let numeric = match &a {
        Value::Ints(x) if x.rank() == 1 => {
            // (sort_unstable_by does not allocate; the index breaks ties, so the order is the stable one)
            idx.sort_unstable_by(|&i, &j| x.data[i].cmp(&x.data[j]).then(i.cmp(&j)));
            true
        }
        #[cfg(feature = "speed")]
        Value::Flts(x) if x.rank() == 1 => {
            idx = crate::fast::grade_flts(&x.data)?;
            true
        }
        #[cfg(not(feature = "speed"))]
        Value::Flts(x) if x.rank() == 1 => {
            // a total order (sort_by may panic without one): NaN last, as numpy sorts
            idx.sort_unstable_by(|&i, &j| x.data[i].partial_cmp(&x.data[j]).unwrap_or_else(|| x.data[i].is_nan().cmp(&x.data[j].is_nan())).then(i.cmp(&j)));
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
                #[cfg(feature = "bigint")]
                Value::Big(_) => numbers = true,
                Value::Flt(f) if !f.is_nan() => numbers = true,
                Value::Str(_) | Value::Char(_) => text = true,
                _ => return Err(Error::typ("cannot order values of different kinds")),
            }
        }
        if numbers && text {
            return Err(Error::typ("cannot order values of different kinds"));
        }
        let mut err = None;
        idx.sort_unstable_by(|&i, &j| {
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
        #[cfg(feature = "bigint")]
        (Value::Big(x), Value::Big(y)) => x == y,
        (Value::Flt(x), Value::Flt(y)) => x == y || (x.is_nan() && y.is_nan()),
        (Value::Char(x), Value::Char(y)) => x == y,
        (Value::Str(x), Value::Str(y)) => x.as_str() == y.as_str(),
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
        return Ok(Value::empty());
    }
    #[cfg(feature = "speed")]
    if let Some(v) = crate::fast::group(&a)? {
        return Ok(v);
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
        #[cfg(feature = "bigint")]
        Value::Big(b) => Ok(crate::big::negate(b)),
        _ if x.is_empty() => Ok(x.clone()),
        _ => Err(Error::typ(crate::msg!("cannot apply - (Negate) to {}", x.describe()))),
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
            #[cfg(feature = "bigint")]
            Value::Big(_) => false,
            _ => x.is_empty(),
        }))
    })
}

fn m_range(_: &mut Interp, a: Value) -> Result<Value> {
    if !a.is_iterable() {
        return Ok(a);
    }
    #[cfg(feature = "speed")]
    if let Some(v) = crate::fast::unique(&a)? {
        return Ok(v);
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
        #[cfg(feature = "bigint")]
        None if x.is_big() => Ok(Value::Flt((1.0 / crate::big::to_flt(x)) as Flt)),
        None if x.is_empty() => Ok(x.clone()),
        None => Err(Error::range(crate::msg!("cannot apply % (Reciprocal) to {}", x.describe()))),
    })
}

fn m_size(ip: &mut Interp, a: Value) -> Result<Value> {
    Ok(match &a {
        Value::Int(i) => Value::Int(i.wrapping_abs()),
        Value::Flt(f) => Value::Flt(fabs(*f)),
        #[cfg(feature = "bigint")]
        Value::Big(b) => crate::big::norm(num_traits::Signed::abs(&**b)),
        Value::Char(c) => Value::Int(*c as u32 as Int),
        Value::Dict(d) => Value::Int(d.borrow().len() as Int),
        Value::Sym(s) => Value::Int(ip.syms.name(*s).chars().count() as Int),
        _ => match a.len() {
            Some(n) => Value::Int(n as Int),
            None => return Err(Error::typ(crate::msg!("cannot apply # (Size) to {}", a.describe()))),
        },
    })
}

fn m_undefined(_: &mut Interp, a: Value) -> Result<Value> {
    Ok(Value::bool(matches!(a, Value::Undefined)))
}

fn shape_of(a: &Value) -> Vec<Int> {
    match a {
        Value::Str(s) => alloc::vec![s.len() as Int],
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
        return Err(Error::range(crate::msg!("index {} out of range for {} of {}", i, what, n)));
    }
    Ok((if i < 0 { i + n } else { i }) as usize)
}

fn index(a: &Value, b: &Value) -> Result<Value> {
    #[cfg(feature = "speed")]
    if let Some(v) = crate::fast::gather(a, b)? {
        return Ok(v);
    }
    let n = a.len().unwrap_or(0);
    match b {
        Value::Int(i) => Ok(a.item(norm_index(*i, n, if a.is_str() { "string" } else { "list" })?)),
        // (KlongPy: an index that is not a whole number leaves the list as it is: `x@1.5` is `x`)
        Value::Flt(_) => Ok(a.clone()),
        #[cfg(feature = "bigint")]
        Value::Big(x) => Err(Error::range(crate::msg!("index {} out of range for {} of {}", crate::big::text(x), if a.is_str() { "string" } else { "list" }, n))),
        _ if b.is_list() => {
            let k = b.len().unwrap_or(0);
            let mut out = try_vec(k)?;
            let (mut atoms, mut lists) = (false, false);
            for j in 0..k {
                let q = b.item(j);
                if q.is_list() {
                    lists = true;
                } else {
                    atoms = true;
                    if !matches!(q, Value::Int(_)) {
                        #[cfg(feature = "bigint")]
                        if q.is_big() {
                            return Err(index(a, &q).err().unwrap_or_else(|| Error::range("index out of range")));
                        }
                        // an index vector holds whole numbers
                        return Err(apply_err(Kind::Range, "@", "At/Index", a, b));
                    }
                }
                out.push(index(a, &q)?);
            }
            if a.is_str() && atoms && lists {
                return Err(apply_err(Kind::Type, "@", "At/Index", a, b));
            }
            if lists && !atoms {
                return Value::from_items(out);
            }
            rebuild(a, out)
        }
        _ => Ok(a.clone()),
    }
}

/// How deeply lists are nested in `v`: 0 for an atom.
fn list_depth(v: &Value) -> usize {
    if !v.is_list() {
        return 0;
    }
    1 + (0..v.len().unwrap_or(0)).map(|j| list_depth(&v.item(j))).max().unwrap_or(0)
}

/// A string indexed by a nested list. KlongPy takes a matrix of indexes and gives a matrix of pieces; an index that does not fit
/// (not whole numbers, out of range, an empty row, more than two levels) is the error of the whole operation, naming the whole index.
fn index_string_nested(a: &Value, b: &Value) -> Result<Value> {
    let whole_err = |kind| apply_err(kind, "@", "At/Index", a, b);
    if list_depth(b) >= 3 {
        return Err(whole_err(Kind::Type));
    }
    if (0..b.len().unwrap_or(0)).any(|j| b.item(j).is_list() && b.item(j).is_empty()) {
        return Err(whole_err(Kind::Range));
    }
    index(a, b).map_err(|e| match e.kind {
        Kind::Type => whole_err(Kind::Type),
        Kind::Range => whole_err(Kind::Range),
        _ => e,
    })
}

fn d_at(ip: &mut Interp, a: Value, b: Value) -> Result<Value> {
    // a host value that answers as a dictionary (a remote dictionary) answers `@` as it answers `?`
    if let Value::Ext(e) = &a {
        if let Some(r) = e.0.find(ip, &b) {
            return r;
        }
    }
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
                return Ok(if a.is_str() { Value::empty_str() } else { Value::empty() });
            }
            if a.is_str() && b.is_list() && (0..b.len().unwrap_or(0)).any(|j| b.item(j).is_list()) {
                return index_string_nested(&a, &b);
            }
            index(&a, &b)
        }
        _ => {
            if b.is_list() || matches!(b, Value::Int(_)) {
                Err(apply_err(Kind::Type, "@", "At/Index", &a, &b))
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
        _ => return Err(Error::typ(crate::msg!(":=: index must be an integer, got {}", v.py_repr(None)))),
    };
    let j = if i < 0 { i + n as i128 } else { i };
    if j < 0 || j > n as i128 || (j == n as i128 && !allow_end) {
        return Err(Error::range(crate::msg!(":=: index {} out of range for length {}", i, n)));
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
    if let Value::Str(t) = &a {
        let ins = match text_of(&v) {
            Some(t) => t,
            None => return Err(Error::typ(":=: value for a string must be a character or string")),
        };
        let ins = ins.as_str();
        let ins_len = ins.chars().count();
        // every index refers to the string as it was given; each replacement is made in the
        // result of the one before
        let orig = t.len();
        let mut r = crate::value::Text::new(try_str(t.as_str())?);
        for k in 1..bn {
            let n = r.len();
            let i = amend_index(&b.item(k), orig, true)?.min(n);
            let end = (i + ins_len).min(n);
            let mut next = try_string(r.as_str().len() + ins.len())?;
            next.push_str(r.slice(0, i));
            next.push_str(ins);
            next.push_str(r.slice(end, n));
            r = crate::value::Text::new(next);
        }
        return Ok(Value::Str(Rc::new(r)));
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

fn d_amend_depth(ip: &mut Interp, a: Value, b: Value) -> Result<Value> {
    if let Value::Flts(f) = &b {
        // a float value joined to the indices made one float list (`5.5,[0 1]`): the indices are integers again, the value stays a float
        if f.rank() == 1 && f.data.len() >= 2 {
            let mut path: Vec<Value> = try_vec(f.data.len() - 1)?;
            for &q in &f.data[1..] {
                if m::trunc(q) != q {
                    return Err(Error::typ(crate::msg!(":-: index must be an integer, got {}", crate::fmt::format_value(&ip.syms, &Value::Flt(q), false)?)));
                }
                path.push(Value::Int(q as Int));
            }
            return amend_depth(&a, &path, &Value::Flt(f.data[0]));
        }
    }
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
        Value::Str(s) => Value::text(s.slice(lo, hi)),
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
        d.borrow_mut().remove(&a)?;
        return Ok(b);
    }
    let k = match a {
        Value::Int(i) => i,
        _ => return Err(apply_err(Kind::Type, "_", "Drop", &a, &b)),
    };
    let n = match b.len() {
        Some(n) => n,
        None => return Err(apply_err(Kind::Type, "_", "Drop", &a, &b)),
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
        return Value::raw_list(out);
    }
    Value::from_items(out)
}

fn d_find(ip: &mut Interp, a: Value, b: Value) -> Result<Value> {
    if let Value::Ext(e) = &a {
        if let Some(r) = e.0.find(ip, &b) {
            return r;
        }
    }
    match &a {
        Value::Str(s) => {
            // KlongPy searches for str(b): the text of a string or a char, the name of a
            // symbol, the digits of a number
            let other;
            let held;
            let sub: &str = match text_of(&b) {
                Some(t) => {
                    held = t;
                    held.as_str()
                }
                None => match &b {
                    Value::Sym(q) => ip.syms.name(*q),
                    Value::Int(_) | Value::Flt(_) => {
                        other = crate::fmt::format_value(&ip.syms, &b, true)?;
                        &other
                    }
                    _ => return Err(Error::typ("?: find in a string needs a string or a char")),
                },
            };
            let hay = s.as_str();
            let mut out: Vec<Int> = Vec::new();
            if sub.is_empty() {
                // the empty string is found between any two characters, and at both ends
                out = try_vec(s.len() + 1)?;
                out.extend((0..=s.len()).map(|i| i as Int));
            } else if sub.len() <= hay.len() {
                // positions are code points; candidates are every character start
                let (hb, sb) = (hay.as_bytes(), sub.as_bytes());
                let mut ci = 0;
                for i in 0..=(hb.len() - sb.len()) {
                    if hb[i] & 0xC0 == 0x80 {
                        continue;
                    }
                    if hb[i..i + sb.len()] == sb[..] {
                        try_push(&mut out, ci as Int)?;
                    }
                    ci += 1;
                }
            }
            Ok(Value::ints(out))
        }
        Value::Dict(d) => Ok(d.borrow().get(&b).cloned().unwrap_or(Value::Undefined)),
        #[cfg(feature = "speed")]
        Value::Ints(x) if x.rank() == 1 && matches!(b, Value::Int(_)) => {
            let Value::Int(k) = b else { return Ok(Value::empty()) };
            let mut out: Vec<Int> = Vec::new();
            for (i, v) in x.data.iter().enumerate() {
                if *v == k {
                    try_push(&mut out, i as Int)?;
                }
            }
            Ok(Value::ints(out))
        }
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

fn d_join(ip: &mut Interp, a: Value, b: Value) -> Result<Value> {
    if let Value::Ext(e) = &a {
        if let Some(r) = e.0.join(ip, &b) {
            return r;
        }
    }
    if let (Some(x), Some(y)) = (text_of(&a), text_of(&b)) {
        let (x, y) = (x.as_str(), y.as_str());
        let mut v = try_string(x.len() + y.len())?;
        v.push_str(x);
        v.push_str(y);
        #[cfg(feature = "speed")]
        {
            let count = |q: &Value| match q {
                Value::Str(t) => Some(t.len()),
                Value::Char(_) => Some(1),
                _ => None,
            };
            if let (Some(n), Some(m)) = (count(&a), count(&b)) {
                return Ok(Value::Str(Rc::new(crate::value::Text::with_count(v, n + m))));
            }
        }
        return Ok(Value::str(v));
    }
    if let Value::Dict(d) = &a {
        if let Value::Dict(e) = &b {
            if !Rc::ptr_eq(d, e) {
                let e = e.borrow();
                let mut dm = d.borrow_mut();
                for i in 0..e.len() {
                    dm.set(e.keys()[i].clone(), e.vals()[i].clone())?;
                }
            }
            return Ok(a);
        }
        if !(b.is_list() && b.len() == Some(2)) {
            return Err(Error::typ(",: a dictionary can only be joined with a tuple (a list of two members) or a dictionary"));
        }
        d.borrow_mut().set(b.item(0), b.item(1))?;
        return Ok(a);
    }
    if let Value::Dict(d) = &b {
        if a.is_list() && a.len() == Some(2) {
            d.borrow_mut().set(a.item(0), a.item(1))?;
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
        #[cfg(feature = "speed")]
        (Value::Int(x), Value::Int(y)) => {
            let mut v = try_vec(2)?;
            v.push(*x);
            v.push(*y);
            return Ok(Value::ints(v));
        }
        #[cfg(feature = "speed")]
        (Value::Int(x), Value::Ints(y)) if y.rank() == 1 => {
            let mut v = try_vec(y.data.len() + 1)?;
            v.push(*x);
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
            return rebuild(&Value::empty_str(), row);
        }
        return Value::from_items(row);
    }
    let mut rows = try_vec(dims[0])?;
    for _ in 0..dims[0] {
        rows.push(build_shape(flat, &dims[1..], at, is_str)?);
    }
    if is_str {
        return Value::raw_list(rows);
    }
    Value::from_items(rows)
}

/// `shape:^b` for a flat numeric `b` and a shape of 2 to MAX_RANK positive integers: the elements fill the shape row by row, cyclically.
/// `None` for anything else, which the general code does (a -1, a zero, a bigger rank, a source of rank above 1).
#[cfg(feature = "speed")]
fn reshape_flat_numbers(b: &Value, shape: &[Int]) -> Option<Value> {
    if shape.len() < 2 || shape.len() > crate::value::MAX_RANK || shape.iter().any(|&d| d < 1 || d as u64 > u32::MAX as u64) {
        return None;
    }
    let mut dims = [0u32; crate::value::MAX_RANK];
    let mut total: usize = 1;
    for (k, &d) in shape.iter().enumerate() {
        dims[k] = d as u32;
        total = total.checked_mul(d as usize)?;
    }
    let dims = &dims[..shape.len()];
    fn fill<T: Copy>(src: &[T], total: usize, dims: &[u32]) -> Option<Arr<T>> {
        if src.is_empty() {
            return None;
        }
        let mut v: Vec<T> = Vec::new();
        v.try_reserve_exact(total).ok()?;
        if total <= src.len() {
            v.extend_from_slice(&src[..total]);
        } else {
            v.extend(src.iter().cycle().take(total).copied());
        }
        Some(Arr::shaped(v, dims))
    }
    match b {
        Value::Ints(x) if x.rank() == 1 => fill(&x.data, total, dims).map(|a| Value::Ints(Rc::new(a))),
        Value::Flts(x) if x.rank() == 1 => fill(&x.data, total, dims).map(|a| Value::Flts(Rc::new(a))),
        _ => None,
    }
}

fn d_reshape(_: &mut Interp, a: Value, b: Value) -> Result<Value> {
    let is_str = b.is_str();
    // a flat numeric vector into an array of rank 2 to MAX_RANK: the numbers are copied, not boxed one by one
    #[cfg(feature = "speed")]
    if let (Value::Ints(_) | Value::Flts(_), Value::Ints(sh)) = (&b, &a) {
        if let Some(v) = reshape_flat_numbers(&b, &sh.data) {
            return Ok(v);
        }
    }
    if a.is_list() {
        let mut dims: Vec<usize> = Vec::new();
        let b_size = if b.is_iterable() { flatten(&b)?.len() } else { 1 };
        for q in a.items()? {
            match q {
                Value::Int(i) if i < -1 => return Err(Error::range(crate::msg!(":^: only -1 may be negative in a shape, got {}", a.py_repr(None)))),
                Value::Int(-1) => {
                    if !b.is_iterable() {
                        return Err(Error::range(":^: -1 in a shape needs a list as the source, not an atom"));
                    }
                    try_push(&mut dims, b_size / 2)?
                }
                Value::Int(i) => try_push(&mut dims, i as usize)?,
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
            return Err(Error::range(crate::msg!(":#: segment sizes must be positive, got {}", s)));
        }
        try_push(&mut sizes, s as usize)?;
    }
    let mut parts: Vec<Value> = Vec::new();
    if sizes.len() == 1 {
        let s = sizes[0];
        if s >= n {
            try_push(&mut parts, b.clone())?;
        } else {
            // numpy.array_split into ceil(n/s) sections: the first n mod k sections get one
            // element more than the others.
            let k = n.div_ceil(s);
            let each = n / k;
            let extras = n % k;
            let mut at = 0;
            for i in 0..k {
                let len = each + (i < extras) as usize;
                try_push(&mut parts, slice(&b, at, at + len)?)?;
                at += len;
            }
        }
    } else {
        let mut q = 0;
        let mut p = 0;
        while q < n {
            try_push(&mut parts, slice(&b, q, q + sizes[p])?)?;
            q += sizes[p];
            p = (p + 1) % sizes.len();
        }
    }
    if b.is_str() {
        return Value::raw_list(parts);
    }
    Value::from_items(parts)
}

fn d_take(_: &mut Interp, a: Value, b: Value) -> Result<Value> {
    let k = match a {
        Value::Int(i) => i,
        _ => return Err(apply_err(Kind::Type, "#", "Take", &a, &b)),
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
    #[cfg(feature = "speed")]
    if let Some(v) = crate::fast::take_cycle(k, &b)? {
        return Ok(v);
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

// ---------------------------------------------------------------------------------------
// Format2 and Form
// ---------------------------------------------------------------------------------------

/// The pairing of the atomic dyads for an operator on atoms that are not all numbers
/// (KlongPy `vec_fn2`): two lists pair element by element, a list and an atom extend the atom,
/// strings are atoms.
fn pair_atoms(ip: &mut Interp, a: &Value, b: &Value, f: fn(&mut Interp, &Value, &Value) -> Result<Value>) -> Result<Value> {
    let (al, bl) = (a.is_list(), b.is_list());
    if !al && !bl {
        return f(ip, a, b);
    }
    let n = if al { a.len().unwrap_or(0) } else { b.len().unwrap_or(0) };
    if al && bl && b.len() != Some(n) {
        return Err(shape_error(n, b.len().unwrap_or(0)));
    }
    let mut out = try_vec(n)?;
    for i in 0..n {
        let r = match (al, bl) {
            (true, true) => pair_atoms(ip, &a.item(i), &b.item(i), f),
            (true, false) => pair_atoms(ip, &a.item(i), b, f),
            _ => pair_atoms(ip, a, &b.item(i), f),
        };
        out.push(r?);
    }
    Value::from_items(out)
}

/// `text` padded with blanks to `width` characters: on the right, or on the left.
fn pad(text: &str, width: usize, left: bool) -> Result<alloc::string::String> {
    let n = text.chars().count();
    let fill = width.saturating_sub(n);
    let mut out = try_string(text.len().checked_add(fill).ok_or_else(Error::memory)?)?;
    if left {
        out.extend(core::iter::repeat_n(' ', fill));
    }
    out.push_str(text);
    if !left {
        out.extend(core::iter::repeat_n(' ', fill));
    }
    Ok(out)
}

/// `a$b` on atoms (KlongPy `__e_dyad_format2`). Public for `.fmt`.
pub fn format2(ip: &mut Interp, a: &Value, b: &Value) -> Result<Value> {
    let whole = match a {
        Value::Int(i) => *i as i128,
        Value::Flt(f) if f.is_finite() => m::trunc(*f) as i128,
        Value::Flt(_) => return Err(Error::range("$: the width must be a finite number")),
        _ => return Err(apply_err(Kind::Type, "$", "Format2", &a, &b)),
    };
    if let Value::Flt(w) = a {
        let x = match b {
            Value::Flt(x) => *x,
            _ => return Err(Error::typ(crate::msg!("$: a real width needs a real number to format, got {}", b.describe())).with_hint("convert an integer first: 5.2$3*1.0")),
        };
        if whole != 0 {
            // Python "{:<w>f}".format(x), w being the text of the width: n.m is n characters
            // at least and m digits after the point; then the part before the point is
            // padded to n characters again
            let mut spec = crate::fmt::Checked::default();
            crate::fmt::write_float(*w, &mut spec);
            let spec = spec.text.trim_start_matches('-');
            let (n, m) = spec.split_once('.').unwrap_or((spec, "0"));
            let (n, m) = match (n.parse::<usize>(), m.parse::<usize>()) {
                (Ok(n), Ok(m)) if m <= 512 && n <= i32::MAX as usize => (n, m),
                _ => return Err(Error::range("$: a real width must have the form n.m")),
            };
            let mut num = crate::fmt::Checked::default();
            crate::fmt::write_fixed(x, m, &mut num);
            if num.oom {
                return Err(Error::memory());
            }
            let body = pad(&num.text, n, true)?;
            if *w < 0.0 {
                return Ok(Value::str(body));
            }
            return Ok(Value::str(match body.split_once('.') {
                Some((int, frac)) => {
                    let mut out = pad(int, n, true)?;
                    try_push_str(&mut out, ".")?;
                    try_push_str(&mut out, frac)?;
                    out
                }
                None => body,
            }));
        }
    }
    // the text of b as Python's str gives it: no quotes, and a symbol with its colon only
    // when it is padded
    let text = match b {
        Value::Sym(s) if whole != 0 => {
            let name = ip.syms.name(*s);
            let mut t = try_string(name.len() + 1)?;
            t.push(':');
            t.push_str(name);
            t
        }
        _ => crate::fmt::format_value(&ip.syms, b, true)?,
    };
    if whole == 0 {
        return Ok(Value::str(text));
    }
    let width = usize::try_from(whole.unsigned_abs()).map_err(|_| Error::memory())?;
    Ok(Value::str(pad(&text, width, whole < 0)?))
}

fn d_format2(ip: &mut Interp, a: Value, b: Value) -> Result<Value> {
    pair_atoms(ip, &a, &b, format2)
}

/// Python `int(text)`: blanks around, a sign, digits (single underscores between digits).
fn parse_int(text: &str) -> Option<Int> {
    let t = text.trim();
    let (neg, digits) = match t.as_bytes().first()? {
        b'-' => (true, &t[1..]),
        b'+' => (false, &t[1..]),
        _ => (false, t),
    };
    if digits.is_empty() {
        return None;
    }
    let mut v: Int = 0;
    let mut prev_digit = false;
    for c in digits.bytes() {
        match c {
            b'0'..=b'9' => {
                let d = (c - b'0') as Int;
                // out of the build's integer width: not an integer here
                v = if neg { v.checked_mul(10)?.checked_sub(d)? } else { v.checked_mul(10)?.checked_add(d)? };
                prev_digit = true;
            }
            b'_' if prev_digit => prev_digit = false,
            _ => return None,
        }
    }
    if prev_digit {
        Some(v)
    } else {
        None
    }
}

/// `parse_int`, and beyond the build width an exact integer (feature `bigint`).
fn parse_int_value(text: &str) -> Option<Value> {
    if let Some(i) = parse_int(text) {
        return Some(Value::Int(i));
    }
    #[cfg(feature = "bigint")]
    {
        let t = text.trim();
        let t = t.strip_prefix('+').unwrap_or(t);
        let body = t.strip_prefix('-').unwrap_or(t);
        if !body.is_empty() && !body.starts_with('_') && !body.ends_with('_') && !body.contains("__") && body.bytes().all(|c| c.is_ascii_digit() || c == b'_') {
            let clean: alloc::string::String = t.chars().filter(|&c| c != '_').collect();
            return crate::big::from_str(&clean).map(crate::big::norm);
        }
    }
    None
}

/// Python `float(text)`.
fn parse_float(text: &str) -> Option<Flt> {
    let t = text.trim();
    // Rust reads the same forms (digits, point, exponent, inf, infinity, nan, any case)
    if t.is_empty() || t.contains('_') {
        return None;
    }
    t.parse::<Flt>().ok()
}

/// `a:$b` on atoms (KlongPy `__e_dyad_form`): `b` converted to the kind of thing `a` is, or
/// `:undefined` when it cannot be.
fn form(ip: &mut Interp, a: &Value, b: &Value) -> Result<Value> {
    Ok(match a {
        Value::Sym(_) => {
            if b.is_empty() {
                return Ok(Value::Undefined);
            }
            let held;
            let name: &str = match b {
                Value::Str(s) => s.as_str().strip_prefix(':').unwrap_or(s.as_str()),
                Value::Sym(_) => return Ok(b.clone()),
                _ => {
                    held = crate::fmt::format_value(&ip.syms, b, true)?;
                    &held
                }
            };
            let s = ip.syms.intern(name);
            if let Some(e) = ip.syms.failure() {
                return Err(e);
            }
            Value::Sym(s)
        }
        Value::Int(_) => match b {
            Value::Str(s) => parse_int_value(s.as_str()).unwrap_or(Value::Undefined),
            _ => Value::Undefined,
        },
        #[cfg(feature = "bigint")]
        Value::Big(_) => match b {
            Value::Str(s) => parse_int_value(s.as_str()).unwrap_or(Value::Undefined),
            _ => Value::Undefined,
        },
        Value::Flt(_) => match b {
            Value::Str(s) => parse_float(s.as_str()).map(Value::Flt).unwrap_or(Value::Undefined),
            Value::Char(c) => c.to_digit(10).map(|d| Value::Flt(d as Flt)).unwrap_or(Value::Undefined),
            Value::Int(i) => Value::Flt(*i as Flt),
            #[cfg(feature = "bigint")]
            Value::Big(x) => Value::Flt(crate::big::to_f64(x) as Flt),
            Value::Flt(_) => b.clone(),
            _ => Value::Undefined,
        },
        Value::Char(_) => match b {
            Value::Char(_) => b.clone(),
            Value::Str(s) if s.len() == 1 => Value::Char(s.char_at(0)),
            Value::Int(i) if (0..10).contains(i) => Value::Char((b'0' + *i as u8) as char),
            _ => Value::Undefined,
        },
        _ => b.clone(),
    })
}

fn d_form(ip: &mut Interp, a: Value, b: Value) -> Result<Value> {
    pair_atoms(ip, &a, &b, form)
}

/// A deep copy of a value that holds dictionaries (a dictionary literal is copied each time
/// it is evaluated).
pub fn deep_copy(v: &Value) -> Result<Value> {
    deep_copy_at(v, 0)
}

fn deep_copy_at(v: &Value, depth: u32) -> Result<Value> {
    if depth > crate::value::max_nest() {
        return Err(crate::value::too_deep());
    }
    match v {
        Value::Dict(d) => {
            let d = d.try_borrow().map_err(|_| Error::internal("dictionary in use"))?;
            let mut n = Dict::default();
            for i in 0..d.len() {
                n.set(d.keys()[i].clone(), deep_copy_at(&d.vals()[i], depth + 1)?)?;
            }
            Ok(Value::dict(n))
        }
        Value::List(items) => {
            let mut out = try_vec(items.len())?;
            for q in items.iter() {
                out.push(deep_copy_at(q, depth + 1)?);
            }
            Value::raw_list(out)
        }
        _ => Ok(v.clone()),
    }
}
