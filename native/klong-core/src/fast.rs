//! Kernels for a desktop (feature `speed`): the numeric array loops written so that the compiler vectorises them, for the operand kinds
//! the generic loop of `ops::array_op` goes through one element at a time (a comparison, a mixed integer and float pair, the minimum and
//! maximum of floats), and the shorter paths of the reductions and of a few primitives. Off for boards: it is code size they have no room
//! for, and the generic loops give the same answers (the conformance suite runs both).
//!
//! The results are those of the generic loops, bit for bit: integers wrap at the build width as they do there, a float result is the
//! one IEEE operation on the two numbers (an integer operand is converted first), a comparison is 0 or 1.
use alloc::vec::Vec;

use crate::error::Result;
use crate::ops::{dims_of, shaped_like, Atomic};
use crate::value::{try_vec, Flt, Int, Value};

/// One operand of a pairing: a flat run, or one number repeated.
enum S<'a> {
    I(&'a [Int]),
    F(&'a [Flt]),
    Ki(Int),
    Kf(Flt),
}

fn side(v: &Value) -> Option<S<'_>> {
    match v {
        Value::Ints(a) => Some(S::I(&a.data)),
        Value::Flts(a) => Some(S::F(&a.data)),
        Value::Int(i) => Some(S::Ki(*i)),
        Value::Flt(f) => Some(S::Kf(*f)),
        _ => None,
    }
}

#[inline(always)]
fn vv<A: Copy, B: Copy, R>(p: &[A], q: &[B], out: &mut Vec<R>, f: impl Fn(A, B) -> R) {
    out.extend(p.iter().zip(q.iter()).map(|(a, b)| f(*a, *b)));
}
#[inline(always)]
fn vk<A: Copy, B: Copy, R>(p: &[A], q: B, out: &mut Vec<R>, f: impl Fn(A, B) -> R) {
    out.extend(p.iter().map(|a| f(*a, q)));
}
#[inline(always)]
fn kv<A: Copy, B: Copy, R>(p: A, q: &[B], out: &mut Vec<R>, f: impl Fn(A, B) -> R) {
    out.extend(q.iter().map(|b| f(p, *b)));
}

/// `x op y` for numeric operands of the same shape (or one of them a number): `Ok(None)` when this module does not do the pair or the
/// operator, and the generic loop does.
pub(crate) fn array(op: Atomic, a: &Value, b: &Value) -> Result<Option<Value>> {
    let (x, y) = match (side(a), side(b)) {
        (Some(x), Some(y)) => (x, y),
        _ => return Ok(None),
    };
    // the shape: two arrays agree exactly (the generic loop broadcasts a lower rank), or one is a number
    let like = match (a, b) {
        (Value::Ints(_) | Value::Flts(_), Value::Ints(_) | Value::Flts(_)) => {
            let (p, q) = (dims_of(a), dims_of(b));
            if p.0 != q.0 || p.1 != q.1 || p.2 != q.2 || p.3 != q.3 {
                return Ok(None);
            }
            a
        }
        (Value::Ints(_) | Value::Flts(_), _) => a,
        (_, Value::Ints(_) | Value::Flts(_)) => b,
        _ => return Ok(None),
    };
    let n = dims_of(like).3;
    use Atomic::*;
    if !matches!(op, Add | Sub | Mul | Div | Min | Max | Less | More | Eq) {
        return Ok(None);
    }
    // integers against integers
    let ints = matches!((&x, &y), (S::I(_) | S::Ki(_), S::I(_) | S::Ki(_)));
    if ints {
        if op == Div {
            return Ok(None);
        }
        let mut out: Vec<Int> = try_vec(n)?;
        macro_rules! go {
            ($f:expr) => {
                match (&x, &y) {
                    (S::I(p), S::I(q)) => vv(p, q, &mut out, $f),
                    (S::I(p), S::Ki(q)) => vk(p, *q, &mut out, $f),
                    (S::Ki(p), S::I(q)) => kv(*p, q, &mut out, $f),
                    _ => return Ok(None),
                }
            };
        }
        match op {
            Add => go!(|a: Int, b: Int| a.wrapping_add(b)),
            Sub => go!(|a: Int, b: Int| a.wrapping_sub(b)),
            Mul => go!(|a: Int, b: Int| a.wrapping_mul(b)),
            Min => go!(|a: Int, b: Int| a.min(b)),
            Max => go!(|a: Int, b: Int| a.max(b)),
            Less => go!(|a: Int, b: Int| (a < b) as Int),
            More => go!(|a: Int, b: Int| (a > b) as Int),
            Eq => go!(|a: Int, b: Int| (a == b) as Int),
            _ => return Ok(None),
        }
        return Ok(Some(shaped_like(like, Some(out), None)));
    }
    // at least one float: the integer operand is converted, element by element
    macro_rules! gof {
        ($out:ty, $f:expr) => {{
            let mut out: Vec<$out> = try_vec(n)?;
            match (&x, &y) {
                (S::F(p), S::F(q)) => vv(p, q, &mut out, $f),
                (S::F(p), S::Kf(q)) => vk(p, *q, &mut out, $f),
                (S::F(p), S::Ki(q)) => vk(p, *q as Flt, &mut out, $f),
                (S::Kf(p), S::F(q)) => kv(*p, q, &mut out, $f),
                (S::Ki(p), S::F(q)) => kv(*p as Flt, q, &mut out, $f),
                (S::I(p), S::F(q)) => vv(p, q, &mut out, |a: Int, b: Flt| $f(a as Flt, b)),
                (S::F(p), S::I(q)) => vv(p, q, &mut out, |a: Flt, b: Int| $f(a, b as Flt)),
                (S::I(p), S::Kf(q)) => vk(p, *q, &mut out, |a: Int, b: Flt| $f(a as Flt, b)),
                (S::Kf(p), S::I(q)) => kv(*p, q, &mut out, |a: Flt, b: Int| $f(a, b as Flt)),
                _ => return Ok(None),
            }
            out
        }};
    }
    // (the generic loop's rules: Min and Max keep the first operand unless the second is strictly smaller or larger, which is
    // what a NaN needs; Div has no zero test here: a zero divisor is the generic loop's, which makes :undefined of it)
    if op == Div {
        let zero = match &y {
            S::F(q) => q.iter().any(|v| *v == 0.0),
            S::I(q) => q.iter().any(|v| *v == 0),
            S::Kf(q) => *q == 0.0,
            S::Ki(q) => *q == 0,
        };
        if zero {
            return Ok(None);
        }
    }
    let flts: Vec<Flt> = match op {
        Add => gof!(Flt, |a: Flt, b: Flt| a + b),
        Sub => gof!(Flt, |a: Flt, b: Flt| a - b),
        Mul => gof!(Flt, |a: Flt, b: Flt| a * b),
        Div => gof!(Flt, |a: Flt, b: Flt| a / b),
        Min => gof!(Flt, |a: Flt, b: Flt| if b < a { b } else { a }),
        Max => gof!(Flt, |a: Flt, b: Flt| if b > a { b } else { a }),
        _ => {
            let ints: Vec<Int> = match op {
                Less => gof!(Int, |a: Flt, b: Flt| (a < b) as Int),
                More => gof!(Int, |a: Flt, b: Flt| (a > b) as Int),
                _ => gof!(Int, |a: Flt, b: Flt| (a == b) as Int),
            };
            return Ok(Some(shaped_like(like, Some(ints), None)));
        }
    };
    Ok(Some(shaped_like(like, None, Some(flts))))
}

/// The minimum or maximum of a float vector as the generic fold gives it: the first element, replaced by a later one that is strictly
/// smaller (larger); so a NaN first stays, and a NaN later is passed over. Eight lanes, then the lanes. A result that is a zero is the
/// first zero of the vector (the fold keeps the first it met, and nothing before it is lower or higher: the sign of a zero is the one
/// place the lanes could differ).
pub(crate) fn flt_extreme(x: &[Flt], max: bool) -> Flt {
    let fold = |s: &[Flt]| s[1..].iter().fold(s[0], |a, &b| if (max && b > a) || (!max && b < a) { b } else { a });
    if x.len() < 64 || x[0].is_nan() {
        return fold(x);
    }
    let mut lane = [x[0]; 8];
    let body = (x.len() - 1) / 8 * 8;
    for c in x[1..1 + body].chunks_exact(8) {
        for j in 0..8 {
            let b = c[j];
            if (max && b > lane[j]) || (!max && b < lane[j]) {
                lane[j] = b;
            }
        }
    }
    let mut r = lane[0];
    for &b in &lane[1..] {
        if (max && b > r) || (!max && b < r) {
            r = b;
        }
    }
    for &b in &x[1 + body..] {
        if (max && b > r) || (!max && b < r) {
            r = b;
        }
    }
    if r == 0.0 {
        return x.iter().copied().find(|v| *v == 0.0).unwrap_or(r);
    }
    r
}

/// The grade of a float vector: the indices that put it in ascending order, equal elements in the order they have, NaN last (as numpy
/// sorts). Sorting (key, index) pairs, the key an unsigned integer that orders as the float does, compares two numbers and not two
/// indirections into the vector; the pair's order breaks ties by index, which makes the sort the stable one.
pub(crate) fn grade_flts(x: &[Flt]) -> Result<Vec<usize>> {
    let mut keys: Vec<(u64, usize)> = try_vec(x.len())?;
    keys.extend(x.iter().enumerate().map(|(i, &f)| {
        let k = if f.is_nan() {
            u64::MAX
        } else {
            let bits = if f == 0.0 { 0u64 } else { (f as f64).to_bits() };
            if bits >> 63 == 1 {
                !bits
            } else {
                bits | (1 << 63)
            }
        };
        (k, i)
    }));
    keys.sort_unstable();
    let mut idx: Vec<usize> = try_vec(x.len())?;
    idx.extend(keys.iter().map(|k| k.1));
    Ok(idx)
}

/// `x op y` that is made in the buffer of an operand nobody else holds (a temporary: `(a+b)*c` makes one for `a+b`), so that no new vector is
/// allocated and written: the same numbers the loops of `array` give. Integers with integers, floats with floats, a number repeated, equal
/// shapes; the operators whose result has the kind of the operands. The operands come back when this is not the case.
pub(crate) fn array_owned(op: Option<Atomic>, x: Value, y: Value) -> core::result::Result<Value, (Value, Value)> {
    use alloc::rc::Rc;
    use Atomic::*;
    let op = match op {
        Some(op @ (Add | Sub | Mul | Min | Max)) => op,
        Some(op @ (Less | More | Eq)) if matches!((&x, &y), (Value::Ints(_), Value::Ints(_) | Value::Int(_)) | (Value::Int(_), Value::Ints(_))) => op,
        _ => return Err((x, y)),
    };
    // which operand's buffer: the left one when it is an array that nobody else holds, else the right one
    let unique = |v: &Value| match v {
        Value::Ints(a) => Rc::strong_count(a) == 1 && Rc::weak_count(a) == 0,
        Value::Flts(a) => Rc::strong_count(a) == 1 && Rc::weak_count(a) == 0,
        _ => false,
    };
    let left = unique(&x);
    if !left && !unique(&y) {
        return Err((x, y));
    }
    let (arr, other) = if left { (&x, &y) } else { (&y, &x) };
    let ok = match (arr, other) {
        (Value::Ints(_), Value::Int(_)) | (Value::Flts(_), Value::Flt(_)) => true,
        (Value::Ints(p), Value::Ints(q)) => p.rank() == q.rank() && dims_of(arr) == dims_of(other),
        (Value::Flts(p), Value::Flts(q)) => p.rank() == q.rank() && dims_of(arr) == dims_of(other),
        _ => false,
    };
    if !ok {
        return Err((x, y));
    }
    let (mut a, b) = if left { (x, y) } else { (y, x) };
    // (`a` is the buffer, `b` the other operand; f(buffer element, other element), flipped when the buffer was the right operand)
    macro_rules! go {
        ($ty:ident, $sc:ident, $f:expr) => {{
            let f = $f;
            let Value::$ty(rc) = &mut a else { unreachable!() };
            let Some(arr) = Rc::get_mut(rc) else { return Err(if left { (a, b) } else { (b, a) }) };
            match &b {
                Value::$ty(o) => {
                    if left {
                        arr.data.iter_mut().zip(o.data.iter()).for_each(|(p, q)| *p = f(*p, *q));
                    } else {
                        arr.data.iter_mut().zip(o.data.iter()).for_each(|(p, q)| *p = f(*q, *p));
                    }
                }
                Value::$sc(k) => {
                    let k = *k;
                    if left {
                        arr.data.iter_mut().for_each(|p| *p = f(*p, k));
                    } else {
                        arr.data.iter_mut().for_each(|p| *p = f(k, *p));
                    }
                }
                _ => unreachable!(),
            }
        }};
    }
    match (&a, op) {
        (Value::Ints(_), Add) => go!(Ints, Int, |p: Int, q: Int| p.wrapping_add(q)),
        (Value::Ints(_), Sub) => go!(Ints, Int, |p: Int, q: Int| p.wrapping_sub(q)),
        (Value::Ints(_), Mul) => go!(Ints, Int, |p: Int, q: Int| p.wrapping_mul(q)),
        (Value::Ints(_), Min) => go!(Ints, Int, |p: Int, q: Int| p.min(q)),
        (Value::Ints(_), Max) => go!(Ints, Int, |p: Int, q: Int| p.max(q)),
        (Value::Ints(_), Less) => go!(Ints, Int, |p: Int, q: Int| (p < q) as Int),
        (Value::Ints(_), More) => go!(Ints, Int, |p: Int, q: Int| (p > q) as Int),
        (Value::Ints(_), Eq) => go!(Ints, Int, |p: Int, q: Int| (p == q) as Int),
        (Value::Flts(_), Add) => go!(Flts, Flt, |p: Flt, q: Flt| p + q),
        (Value::Flts(_), Sub) => go!(Flts, Flt, |p: Flt, q: Flt| p - q),
        (Value::Flts(_), Mul) => go!(Flts, Flt, |p: Flt, q: Flt| p * q),
        (Value::Flts(_), Min) => go!(Flts, Flt, |p: Flt, q: Flt| if q < p { q } else { p }),
        (Value::Flts(_), Max) => go!(Flts, Flt, |p: Flt, q: Flt| if q > p { q } else { p }),
        _ => return Err(if left { (a, b) } else { (b, a) }),
    }
    Ok(a)
}
