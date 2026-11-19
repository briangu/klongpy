//! A fixed-point value of a program that runs on the interpreter (stage one of the compiler, `StaticProgram`): the arithmetic on it
//! follows the reference semantics of `fixed.rs` operation by operation, as `FixedArray.__array_ufunc__` does in
//! `klongpy/static/fixed.py`.
//!
//! What works: the dyads `+ - * % | & < > =` between two fixed-point values of the same type, or one of them and an int (a number or an
//! array, converted to the type of the other: `v * 2^F`, wrapped); a scalar against an array; the monads `-` (negate) and `_` (floor to
//! an int); `+/ |/ &/` and `+\ |\ &\` along the first axis. A comparison gives 0 or 1. A float or a value of another type is a type
//! error ("cannot mix ..."), as in Python. Everything else is not handled here and is an error of the interpreter's.
//!
//! Where Python's `FixedArray` is wrong and this is not: `-a` and `_a` (they return the raw integers negated or unchanged, a NumPy
//! array that has lost its type; here negation is fixed-point and floor is `raw >> F`), and `a=a`, `a<a`, `a>a` (NumPy arrays; here the
//! same 0 and 1, as ints). Division by zero is a range error here (NumPy answers 0 or :undefined by case); division is defined for
//! W <= 32 only, as in Python.
use crate::compile::FixedBox;
use crate::fixed as fx;
use klong_core::value::{Arr, Ext, Int};
use klong_core::{Error, Interp, Value};

fn fixed_of(v: &Value) -> Option<&FixedBox> {
    if let Value::Ext(e) = v {
        e.0.as_any().downcast_ref::<FixedBox>()
    } else {
        None
    }
}

fn raw_value(shape: &[usize], raw: &[i64]) -> Value {
    let d: Vec<Int> = raw.iter().map(|x| *x as Int).collect();
    if shape.is_empty() {
        return Value::Int(d[0]);
    }
    if shape.len() == 1 {
        return Value::ints(d);
    }
    let dims: Vec<u32> = shape.iter().map(|x| *x as u32).collect();
    Value::Ints(std::rc::Rc::new(Arr::shaped(d, &dims)))
}

fn boxed(elem: &str, shape: Vec<usize>, raw: Vec<i64>) -> Value {
    Value::Ext(std::rc::Rc::new(klong_core::value::ExtBox(Box::new(FixedBox { elem: elem.to_string(), shape, raw }))))
}

/// An int operand (a number or an array) as (shape, raw values of the fixed-point type), or what is wrong with it.
fn int_operand(v: &Value, spec: fx::Spec, elem: &str) -> Option<Result<(Vec<usize>, Vec<i64>), Error>> {
    match v {
        Value::Int(i) => Some(Ok((Vec::new(), vec![fx::from_int(*i as i64, spec)]))),
        Value::Ints(a) => Some(Ok(((0..a.rank()).map(|i| a.dim(i)).collect(), a.data.iter().map(|x| fx::from_int(*x as i64, spec)).collect()))),
        Value::Flt(_) | Value::Flts(_) => Some(Err(Error::typ(format!("cannot mix {} with a float", elem)))),
        _ => None,
    }
}

fn operand(v: &Value, other: &FixedBox) -> Option<Result<(Vec<usize>, Vec<i64>, fx::Spec), Error>> {
    let spec = fx::spec_of(&other.elem);
    if let Some(f) = fixed_of(v) {
        let s = fx::spec_of(&f.elem);
        if s != spec {
            return Some(Err(Error::typ(format!("cannot mix {} and {}", other.elem, f.elem))));
        }
        return Some(Ok((f.shape.clone(), f.raw.clone(), spec)));
    }
    int_operand(v, spec, &other.elem).map(|r| r.map(|(sh, raw)| (sh, raw, spec)))
}

fn broadcast(a: &[usize], b: &[usize]) -> Result<Vec<usize>, Error> {
    if a == b || b.is_empty() {
        Ok(a.to_vec())
    } else if a.is_empty() {
        Ok(b.to_vec())
    } else {
        Err(Error::shape(format!("fixed-point operands of shapes {:?} and {:?} cannot be combined", a, b)))
    }
}

fn dyad(glyph: &str, x: &Value, y: &Value) -> Option<Result<Value, Error>> {
    if !matches!(glyph, "+" | "-" | "*" | "%" | "|" | "&" | "<" | ">" | "=") {
        return None;
    }
    let anchor = fixed_of(x).or_else(|| fixed_of(y))?;
    let (l, r) = (operand(x, anchor)?, operand(y, anchor)?);
    Some((|| {
        let ((ls, lr, spec), (rs, rr, _)) = (l?, r?);
        let shape = broadcast(&ls, &rs)?;
        let n: usize = shape.iter().product();
        let at = |v: &Vec<i64>, s: &Vec<usize>, k: usize| if s.is_empty() { v[0] } else { v[k] };
        if glyph == "%" && spec.w > 32 {
            return Err(Error::typ("fixed-point division is defined for W <= 32"));
        }
        let mut out = Vec::with_capacity(n);
        for k in 0..n {
            let (a, b) = (at(&lr, &ls, k), at(&rr, &rs, k));
            out.push(match glyph {
                "+" => fx::add(a, b, spec),
                "-" => fx::sub(a, b, spec),
                "*" => fx::mul(a, b, spec),
                "%" => {
                    if b == 0 {
                        return Err(Error::range("fixed-point division by zero"));
                    }
                    fx::div(a, b, spec)
                }
                "|" => a.max(b),
                "&" => a.min(b),
                "<" => (a < b) as i64,
                ">" => (a > b) as i64,
                _ => (a == b) as i64,
            });
        }
        Ok(if matches!(glyph, "<" | ">" | "=") { raw_value(&shape, &out) } else { boxed(&anchor.elem, shape, out) })
    })())
}

fn monad(glyph: &str, x: &Value) -> Option<Result<Value, Error>> {
    let f = fixed_of(x)?;
    let spec = fx::spec_of(&f.elem);
    match glyph {
        "-" => Some(Ok(boxed(&f.elem, f.shape.clone(), f.raw.iter().map(|a| fx::neg(*a, spec)).collect()))),
        "_" => Some(Ok(raw_value(&f.shape, &f.raw.iter().map(|a| fx::floor_int(*a, spec, 64)).collect::<Vec<_>>()))),
        _ => None,
    }
}

/// `+/` `|/` `&/` and the scans, along the first axis.
fn adverb(glyph: &str, op: Option<&str>, x: &Value, y: Option<&Value>) -> Option<Result<Value, Error>> {
    if y.is_some() || !matches!(glyph, "/" | "\\") {
        return None;
    }
    let f = fixed_of(x)?;
    let op = op?;
    if !matches!(op, "+" | "|" | "&") {
        return Some(Err(Error::typ(format!("cannot apply {}{} to a fixed-point value", op, glyph))));
    }
    let spec = fx::spec_of(&f.elem);
    if f.shape.is_empty() || f.shape[0] == 0 {
        return Some(Err(Error::range(format!("{}{}: there is nothing to fold in a scalar or an empty array", op, glyph))));
    }
    let rows = f.shape[0];
    let width: usize = f.shape[1..].iter().product();
    let step = |a: i64, b: i64| match op {
        "+" => fx::add(a, b, spec),
        "|" => a.max(b),
        _ => a.min(b),
    };
    let mut acc: Vec<i64> = f.raw[..width].to_vec();
    let mut all: Vec<i64> = acc.clone();
    for r in 1..rows {
        for c in 0..width {
            acc[c] = step(acc[c], f.raw[r * width + c]);
        }
        all.extend_from_slice(&acc);
    }
    Some(Ok(if glyph == "/" { boxed(&f.elem, f.shape[1..].to_vec(), acc) } else { boxed(&f.elem, f.shape.clone(), all) }))
}

/// The text `.p` and `$` show: the raw integers, as the NumPy array of Python shows them.
fn show(f: &FixedBox) -> String {
    fn nest(shape: &[usize], raw: &[i64]) -> String {
        match shape {
            [] => raw[0].to_string(),
            [n] => format!("[{}]", raw[..*n].iter().map(|x| x.to_string()).collect::<Vec<_>>().join(" ")),
            [n, rest @ ..] => {
                let w: usize = rest.iter().product();
                format!("[{}]", (0..*n).map(|i| nest(rest, &raw[i * w..(i + 1) * w])).collect::<Vec<_>>().join(" "))
            }
        }
    }
    nest(&f.shape, &f.raw)
}

impl Ext for FixedBox {
    fn type_name(&self) -> &str {
        "fixed"
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn host_monad(&self, _ip: &mut Interp, glyph: &str, x: &Value) -> Option<Result<Value, Error>> {
        monad(glyph, x)
    }
    fn host_dyad(&self, _ip: &mut Interp, glyph: &str, x: &Value, y: &Value) -> Option<Result<Value, Error>> {
        dyad(glyph, x, y)
    }
    fn host_adverb(&self, _ip: &mut Interp, glyph: &str, _verb: &Value, op: Option<&str>, x: &Value, y: Option<&Value>) -> Option<Result<Value, Error>> {
        adverb(glyph, op, x, y)
    }
    fn host_format(&self, _display: bool) -> Option<String> {
        Some(show(self))
    }
}
