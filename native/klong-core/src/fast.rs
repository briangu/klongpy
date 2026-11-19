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
    if !matches!(op, Add | Sub | Mul | Div | Rem | IDiv | Min | Max | Less | More | Eq) {
        return Ok(None);
    }
    // integers against integers
    let ints = matches!((&x, &y), (S::I(_) | S::Ki(_), S::I(_) | S::Ki(_)));
    if matches!(op, Rem | IDiv) && !ints {
        return Ok(None);
    }
    // (a zero divisor is the generic loop's, which makes :undefined of it)
    if ints && matches!(op, Div | Rem | IDiv) {
        let zero = match &y {
            S::I(q) => q.iter().any(|v| *v == 0),
            S::Ki(q) => *q == 0,
            _ => false,
        };
        if zero {
            return Ok(None);
        }
    }
    // (integers under `%` are divided as floats, below)
    if ints && op != Div {
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
            Rem => go!(|a: Int, b: Int| a.wrapping_rem(b)),
            IDiv => go!(|a: Int, b: Int| a.wrapping_div(b)),
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
                (S::I(p), S::I(q)) => vv(p, q, &mut out, |a: Int, b: Int| $f(a as Flt, b as Flt)),
                (S::I(p), S::Ki(q)) => vk(p, *q, &mut out, |a: Int, b: Int| $f(a as Flt, b as Flt)),
                (S::Ki(p), S::I(q)) => kv(*p, q, &mut out, |a: Int, b: Int| $f(a as Flt, b as Flt)),
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

/// The order-preserving key of a float for sorting runs of equal values together: `-0.0` and `0.0` are one value, and so are all the NaNs
/// (as `same_key` has them).
#[inline]
fn flt_key(f: Flt) -> u64 {
    let f = f as f64;
    if f.is_nan() {
        return u64::MAX;
    }
    let b = if f == 0.0 { 0.0f64.to_bits() } else { f.to_bits() };
    if b >> 63 == 1 {
        !b
    } else {
        b | (1 << 63)
    }
}

/// The (key, position) pairs of a flat vector, sorted: the equal values are runs, each in order of position.
fn runs_of<K: Ord + Copy>(keys: impl Iterator<Item = K>, n: usize) -> Result<Option<Vec<(K, u32)>>> {
    if n >= u32::MAX as usize {
        return Ok(None);
    }
    let mut p: Vec<(K, u32)> = try_vec(n)?;
    p.extend(keys.enumerate().map(|(i, k)| (k, i as u32)));
    p.sort_unstable();
    Ok(Some(p))
}

/// `?x` of a flat vector of integers or floats: the values in order of first occurrence, without the repeats (sorting, not a search of what
/// is kept so far: the generic loop takes time as the square of the length when most values are different).
pub(crate) fn unique(a: &Value) -> Result<Option<Value>> {
    macro_rules! go {
        ($x:expr, $key:expr, $mk:path) => {{
            let n = $x.data.len();
            if n < 2 {
                return Ok(None);
            }
            let Some(p) = runs_of($x.data.iter().map($key), n)? else { return Ok(None) };
            let mut first: Vec<u32> = try_vec(n)?;
            first.extend(p.iter().enumerate().filter(|(k, q)| *k == 0 || p[*k - 1].0 != q.0).map(|(_, q)| q.1));
            first.sort_unstable();
            let mut out = try_vec(first.len())?;
            out.extend(first.iter().map(|&i| $x.data[i as usize]));
            Ok(Some($mk(out)))
        }};
    }
    match a {
        Value::Ints(x) if x.rank() == 1 => go!(x, |&v| v, Value::ints),
        Value::Flts(x) if x.rank() == 1 => go!(x, |&v| flt_key(v), Value::flts),
        _ => Ok(None),
    }
}

/// `=x` of a flat vector of integers: the positions of each value, the values in order of first occurrence.
pub(crate) fn group(a: &Value) -> Result<Option<Value>> {
    let Value::Ints(x) = a else { return Ok(None) };
    let n = x.data.len();
    if x.rank() != 1 || n < 2 {
        return Ok(None);
    }
    let Some(p) = runs_of(x.data.iter().copied(), n)? else { return Ok(None) };
    // (first position, start, end) of each run, in order of first position
    let mut runs: Vec<(u32, u32, u32)> = Vec::new();
    let mut s = 0;
    while s < n {
        let mut e = s + 1;
        while e < n && p[e].0 == p[s].0 {
            e += 1;
        }
        crate::value::try_push(&mut runs, (p[s].1, s as u32, e as u32))?;
        s = e;
    }
    runs.sort_unstable();
    let mut out = try_vec(runs.len())?;
    for &(_, s, e) in &runs {
        let mut g: Vec<Int> = try_vec((e - s) as usize)?;
        g.extend(p[s as usize..e as usize].iter().map(|q| q.1 as Int));
        out.push(Value::ints(g));
    }
    Value::from_items(out).map(Some)
}

/// `a@b` for a flat vector `a` of integers or floats and a flat vector `b` of integers that are all in range (counting back from the end
/// when negative): the elements picked, without making each one a value. Any other case, an index out of range included, is the generic
/// loop's.
pub(crate) fn gather(a: &Value, b: &Value) -> Result<Option<Value>> {
    let Value::Ints(ix) = b else { return Ok(None) };
    if ix.rank() != 1 || ix.data.is_empty() {
        return Ok(None);
    }
    fn pick<T: Copy>(x: &[T], ix: &[Int]) -> Result<Option<Vec<T>>> {
        let n = x.len() as i128;
        if n == 0 || ix.iter().any(|&i| (i as i128) < -n || (i as i128) >= n) {
            return Ok(None);
        }
        let mut out: Vec<T> = try_vec(ix.len())?;
        out.extend(ix.iter().map(|&i| x[if i < 0 { (i as i128 + n) as usize } else { i as usize }]));
        Ok(Some(out))
    }
    Ok(match a {
        Value::Ints(x) if x.rank() == 1 => pick(&x.data, &ix.data)?.map(Value::ints),
        Value::Flts(x) if x.rank() == 1 => pick(&x.data, &ix.data)?.map(Value::flts),
        _ => None,
    })
}

/// `k#b` for a flat vector `b` of integers or floats when `k` asks for more elements than there are: the vector repeated, from its start
/// (`k` positive) or ending at its end (`k` negative).
pub(crate) fn take_cycle(k: Int, b: &Value) -> Result<Option<Value>> {
    fn rep<T: Copy>(x: &[T], k: Int) -> Result<Option<Vec<T>>> {
        let n = x.len();
        let count = k.unsigned_abs() as usize;
        if n == 0 || count <= n {
            return Ok(None);
        }
        let start = if k >= 0 { 0 } else { n - count % n };
        let mut out: Vec<T> = try_vec(count)?;
        let mut j = start % n;
        for _ in 0..count {
            out.push(x[j]);
            j += 1;
            if j == n {
                j = 0;
            }
        }
        Ok(Some(out))
    }
    Ok(match b {
        Value::Ints(x) if x.rank() == 1 => rep(&x.data, k)?.map(Value::ints),
        Value::Flts(x) if x.rank() == 1 => rep(&x.data, k)?.map(Value::flts),
        _ => None,
    })
}
