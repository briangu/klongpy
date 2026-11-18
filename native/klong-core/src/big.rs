//! Integers beyond the build width (feature `bigint`).
//!
//! KlongPy's integers are Python's: an operand beyond 64 bits makes `+ - * < > = | &` and the scalar `^` of two integers exact. The
//! native runtime keeps ordinary integers as machine integers and adds one value kind, `Value::Big`, used only for a value that does not
//! fit `Int`; every result that fits goes back to `Value::Int` (`norm`).
use alloc::rc::Rc;
use alloc::string::String;
use core::cmp::Ordering;
use num_bigint::{BigInt, Sign};
use num_traits::{Signed, ToPrimitive, Zero};

use crate::error::Result;
use crate::value::{Flt, Int, Value};

pub type Big = Rc<BigInt>;

/// The value of an integer: `Int` when it fits, else `Big`.
pub fn norm(b: BigInt) -> Value {
    #[cfg(not(feature = "int32"))]
    let small = b.to_i64();
    #[cfg(feature = "int32")]
    let small = b.to_i32();
    match small {
        Some(i) => Value::Int(i as Int),
        None => Value::Big(Rc::new(b)),
    }
}

pub fn from_str(d: &str) -> Option<BigInt> {
    d.parse::<BigInt>().ok()
}

/// The integer value of `Int` and `Big`.
pub fn int_of(v: &Value) -> Option<BigInt> {
    match v {
        Value::Int(i) => Some(BigInt::from(*i)),
        Value::Big(b) => Some((**b).clone()),
        _ => None,
    }
}

pub fn text(b: &BigInt) -> String {
    use alloc::string::ToString;
    b.to_string()
}

pub fn to_f64(b: &BigInt) -> f64 {
    b.to_f64().unwrap_or(if b.is_negative() { f64::NEG_INFINITY } else { f64::INFINITY })
}

/// The integer of a float's whole part (`None` for NaN and infinity).
pub fn from_f64(f: f64) -> Option<BigInt> {
    num_traits::FromPrimitive::from_f64(f)
}

/// Python's `a / b` on two integers: the correctly rounded quotient, whatever the size of the operands (infinite when it does not fit a
/// double: Python raises `OverflowError` there, the caller raises a range error).
pub fn ratio(a: &BigInt, b: &BigInt) -> f64 {
    if a.is_zero() {
        return 0.0;
    }
    let neg = a.is_negative() != b.is_negative();
    let (a, b) = (a.magnitude(), b.magnitude());
    // a quotient of 128 or more bits with a sticky bit for the remainder rounds correctly when converted
    let sh = 128i64 - (a.bits() as i64 - b.bits() as i64);
    let (n, d) = if sh >= 0 { (a << (sh as u64), b.clone()) } else { (a.clone(), b << ((-sh) as u64)) };
    let mut q = &n / &d;
    if !(&n % &d).is_zero() {
        q |= num_bigint::BigUint::from(1u32);
    }
    let f = num_traits::ToPrimitive::to_f64(&q).unwrap_or(f64::INFINITY);
    let r = if f.is_infinite() { f } else { libm::ldexp(f, -(sh as i32)) };
    if neg {
        -r
    } else {
        r
    }
}

/// Exact order of an integer and a float (`None` for NaN).
pub fn cmp_flt(b: &BigInt, f: f64) -> Option<Ordering> {
    if f.is_nan() {
        return None;
    }
    if f.is_infinite() {
        return Some(if f > 0.0 { Ordering::Less } else { Ordering::Greater });
    }
    let t = libm::trunc(f);
    let ti = from_f64(t)?;
    Some(match b.cmp(&ti) {
        Ordering::Equal => {
            let frac = f - t;
            if frac > 0.0 {
                Ordering::Less
            } else if frac < 0.0 {
                Ordering::Greater
            } else {
                Ordering::Equal
            }
        }
        o => o,
    })
}

/// Order of two numbers of which at least one is `Big` (`None` when a NaN is involved).
pub fn cmp_num(a: &Value, b: &Value) -> Option<Ordering> {
    match (a, b) {
        (Value::Flt(x), y) => cmp_num(y, &Value::Flt(*x)).map(Ordering::reverse),
        (x, Value::Flt(f)) => cmp_flt(&int_of(x)?, *f as f64),
        (x, y) => Some(int_of(x)?.cmp(&int_of(y)?)),
    }
}

/// Bits an exact integer power may have before the float answers instead (as in KlongPy).
const EXACT_POWER_BITS: u64 = 1 << 20;

/// `a^b` for two integers, exact while the result stays within `EXACT_POWER_BITS` bits (`None`: use floats).
pub fn int_pow(a: &BigInt, b: &BigInt) -> Option<BigInt> {
    if b.is_negative() {
        return None;
    }
    if a.is_zero() {
        return Some(if b.is_zero() { BigInt::from(1) } else { BigInt::zero() });
    }
    let e = b.to_u64()?;
    if a.bits().checked_mul(e)? > EXACT_POWER_BITS {
        return None;
    }
    Some(num_traits::pow::Pow::pow(a, e))
}

/// `a%b` as a float, for numbers of which one is `Big`.
pub fn quotient(a: &Value, b: &Value) -> f64 {
    match (int_of(a), int_of(b)) {
        (Some(x), Some(y)) => ratio(&x, &y),
        _ => to_flt(a) / to_flt(b),
    }
}

pub fn to_flt(v: &Value) -> f64 {
    match v {
        Value::Int(i) => *i as f64,
        Value::Flt(f) => *f as f64,
        Value::Big(b) => to_f64(b),
        _ => 0.0,
    }
}

pub fn is_zero(v: &Value) -> bool {
    match v {
        Value::Int(i) => *i == 0,
        Value::Flt(f) => *f == 0.0,
        _ => false,
    }
}

/// The saturated machine integer of a `Big` (what `_` of a float beyond the range gives).
pub fn saturate(b: &BigInt) -> Int {
    if b.sign() == Sign::Minus {
        Int::MIN
    } else {
        Int::MAX
    }
}

pub fn negate(b: &BigInt) -> Value {
    norm(-b)
}

/// Whether `v` is an integer beyond the build width.
pub fn is_big(v: &Value) -> bool {
    matches!(v, Value::Big(_))
}

/// The binary operators with an operand beyond the build width.
pub fn op(op: crate::ops::Atomic, a: &Value, b: &Value) -> Result<Option<Value>> {
    use crate::ops::Atomic as A;
    let both_int = !matches!(a, Value::Flt(_)) && !matches!(b, Value::Flt(_));
    let r = match op {
        A::Add | A::Sub | A::Mul => {
            if both_int {
                let (x, y) = (int_of(a).unwrap_or_default(), int_of(b).unwrap_or_default());
                norm(match op {
                    A::Add => x + y,
                    A::Sub => x - y,
                    _ => x * y,
                })
            } else {
                let (x, y) = (to_flt(a), to_flt(b));
                let r = match op {
                    A::Add => x + y,
                    A::Sub => x - y,
                    _ => x * y,
                };
                if to_flt(a).is_infinite() || to_flt(b).is_infinite() {
                    return Ok(None);
                }
                Value::Flt(r as Flt)
            }
        }
        A::Div => {
            if is_zero(b) {
                return Ok(Some(Value::Undefined));
            }
            let q = quotient(a, b);
            if q.is_infinite() && (both_int || (to_flt(a).is_finite() && to_flt(b).is_finite())) {
                return Ok(None);
            }
            Value::Flt(q as Flt)
        }
        A::IDiv => {
            if is_zero(b) {
                return Ok(Some(Value::Undefined));
            }
            let q = libm::trunc(quotient(a, b));
            match from_f64(q) {
                Some(i) => norm(i),
                None => return Ok(None),
            }
        }
        A::Rem => return Ok(None),
        A::Min | A::Max => match cmp_num(a, b) {
            Some(o) => {
                let a_wins = if op == A::Max { o != Ordering::Less } else { o != Ordering::Greater };
                let pick = if a_wins { a } else { b };
                if both_int {
                    pick.clone()
                } else {
                    // as NumPy gives it: a float when an operand is one
                    if to_flt(pick).is_infinite() {
                        return Ok(None);
                    }
                    Value::Flt(to_flt(pick) as Flt)
                }
            }
            None => b.clone(),
        },
        A::Less => Value::bool(cmp_num(a, b) == Some(Ordering::Less)),
        A::More => Value::bool(cmp_num(a, b) == Some(Ordering::Greater)),
        A::Eq => Value::bool(cmp_num(a, b) == Some(Ordering::Equal)),
        A::Pow => {
            if both_int {
                if let (Some(x), Some(y)) = (int_of(a), int_of(b)) {
                    if let Some(r) = int_pow(&x, &y) {
                        return Ok(Some(norm(r)));
                    }
                }
            }
            let r = libm::pow(to_flt(a), to_flt(b));
            if r.is_finite() && libm::trunc(r) == r {
                return Ok(Some(match from_f64(r) {
                    Some(i) => norm(i),
                    None => return Ok(None),
                }));
            }
            if !r.is_finite() && !r.is_nan() {
                return Ok(None);
            }
            Value::Flt(r as Flt)
        }
    };
    Ok(Some(r))
}

/// The integer of Python's `format(n, "x")` (a sign and hex digits): the form integers cross to a host that has its own big integers
/// (the hexadecimal text has no digit limit there, where decimal text has).
pub fn from_hex(s: &str) -> Option<Value> {
    let (neg, digits) = match s.strip_prefix('-') {
        Some(d) => (true, d),
        None => (false, s),
    };
    let m = BigInt::parse_bytes(digits.as_bytes(), 16)?;
    Some(norm(if neg { -m } else { m }))
}

/// The text `from_hex` reads, for an integer value (`Int` or `Big`).
pub fn to_hex(v: &Value) -> Option<String> {
    let b = int_of(v)?;
    let m = b.magnitude().to_str_radix(16);
    Some(if b.is_negative() { alloc::format!("-{}", m) } else { m })
}
