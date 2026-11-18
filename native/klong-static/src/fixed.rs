//! Fixed-point numbers of the static profile (`klongpy/static/fixed.py`, docs/hardware.md): `fixed<W,I>` is a signed number of W bits
//! of which I are integer bits; the value of the W-bit integer `raw` is `raw * 2^-(W-I)`. Raw values are `i64` (a product of two
//! operands of more than 32 bits is done on `i128`).
use crate::sig::fixed_spec;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Spec {
    pub w: u32,
    pub i: u32,
}

impl Spec {
    pub fn f(&self) -> u32 {
        self.w - self.i
    }
    pub fn elem(&self) -> String {
        format!("fixed<{},{}>", self.w, self.i)
    }
}

pub fn spec_of(elem: &str) -> Spec {
    let (w, i) = fixed_spec(elem).expect("a fixed-point type");
    Spec { w, i }
}

/// Two's complement wrap of an integer to W bits.
pub fn wrap(v: i128, s: Spec) -> i64 {
    if s.w >= 64 {
        return v as i64;
    }
    let m = (1i128 << s.w) - 1;
    let h = 1i128 << (s.w - 1);
    (((v + h) & m) - h) as i64
}

pub fn add(a: i64, b: i64, s: Spec) -> i64 {
    wrap(a as i128 + b as i128, s)
}
pub fn sub(a: i64, b: i64, s: Spec) -> i64 {
    wrap(a as i128 - b as i128, s)
}
pub fn neg(a: i64, s: Spec) -> i64 {
    wrap(-(a as i128), s)
}
pub fn mul(a: i64, b: i64, s: Spec) -> i64 {
    wrap((a as i128 * b as i128) >> s.f(), s)
}
/// floor(a * 2^F / b), wrapped; `b` is not 0 and W <= 32.
pub fn div(a: i64, b: i64, s: Spec) -> i64 {
    let n = (a as i128) << s.f();
    let b = b as i128;
    let q = n.div_euclid(b);
    // floor division: div_euclid rounds toward -inf for positive b, toward +inf for negative b
    let q = if b < 0 && n.rem_euclid(b) != 0 { q - 1 } else { q };
    wrap(q, s)
}
/// `_x`: floor to an int, wrapped to kg_int.
pub fn floor_int(a: i64, s: Spec, int_bits: u32) -> i64 {
    let v = a >> s.f();
    if int_bits == 64 {
        v
    } else {
        wrap_bits(v as i128, int_bits)
    }
}
pub fn wrap_bits(v: i128, bits: u32) -> i64 {
    if bits >= 64 {
        return v as i64;
    }
    let m = (1i128 << bits) - 1;
    let h = 1i128 << (bits - 1);
    (((v + h) & m) - h) as i64
}
pub fn from_int(v: i64, s: Spec) -> i64 {
    wrap((v as i128) << s.f(), s)
}
/// floor(v * 2^F) wrapped; nan and anything beyond a 64-bit integer give 0.
pub fn from_float(v: f64, s: Spec) -> i64 {
    let scaled = (v * 2f64.powi(s.f() as i32)).floor();
    if scaled.is_finite() && scaled >= -9223372036854775808.0 && scaled < 9223372036854775808.0 {
        wrap(scaled as i64 as i128, s)
    } else {
        0
    }
}
pub fn to_float(raw: i64, s: Spec) -> f64 {
    raw as f64 * 2f64.powi(-(s.f() as i32))
}
pub fn convert(raw: i64, src: Spec, dst: Spec) -> i64 {
    let d = dst.f() as i32 - src.f() as i32;
    if d >= 0 {
        wrap((raw as i128) << d, dst)
    } else {
        wrap((raw >> (-d)) as i128, dst)
    }
}
