//! `.split .join .words .trim .num`: the string functions (KlongPy `eval_sys_split` and the
//! functions after it in `sys_fn.py`). They need no file system, so every build has them.
//! White space is the four characters space, tab, newline and carriage return, as in KlongPy.
use alloc::rc::Rc;
use alloc::vec::Vec;

use klong_core::error::Result;
use klong_core::value::{try_push_char, try_push_str, try_str, try_string, try_vec};
use klong_core::{Error, Interp, Value};

fn is_ws(c: char) -> bool {
    matches!(c, ' ' | '\t' | '\n' | '\r')
}

fn args2<'a>(a: &'a [Value], name: &str) -> Result<(&'a Value, &'a Value)> {
    match a {
        [x, y] => Ok((x, y)),
        _ => Err(Error::arity(klong_core::msg!("{} takes 2 arguments, got {}", name, a.len()))),
    }
}

/// The text of a string (a char is not one).
fn string<'a>(v: &'a Value, name: &str, which: &str) -> Result<&'a str> {
    match v {
        Value::Str(s) => Ok(s.as_str()),
        v => Err(Error::typ(klong_core::msg!("{}: {} must be a string, got {}", name, which, v.describe()))),
    }
}

/// A separator: a non-empty string or a char, as text. The char is copied into `buf`.
fn separator<'a>(v: &'a Value, buf: &'a mut [u8; 4], name: &str, which: &str) -> Result<&'a str> {
    match v {
        Value::Str(s) if s.as_str().is_empty() => Err(Error::range(klong_core::msg!("{}: {} must not be empty", name, which))),
        Value::Str(s) => Ok(s.as_str()),
        Value::Char(c) => Ok(c.encode_utf8(buf)),
        v => Err(Error::typ(klong_core::msg!("{}: {} must be a string or a char, got {}", name, which, v.describe()))),
    }
}

fn push(out: &mut Vec<Value>, piece: &str) -> Result<()> {
    let v = Value::text(piece)?;
    out.try_reserve(1).map_err(|_| Error::memory())?;
    out.push(v);
    Ok(())
}

fn split(a: &[Value]) -> Result<Value> {
    let (x, y) = args2(a, ".split")?;
    let text = string(x, ".split", "x")?;
    let mut buf = [0u8; 4];
    let sep = separator(y, &mut buf, ".split", "y")?;
    let mut out = try_vec(0)?;
    for piece in text.split(sep) {
        push(&mut out, piece)?;
    }
    Value::from_items(out)
}

fn join(a: &[Value]) -> Result<Value> {
    let (x, y) = args2(a, ".join")?;
    if !x.is_list() {
        return Err(Error::typ(klong_core::msg!(".join: x must be a list of strings, got {}", x.describe())));
    }
    let mut buf = [0u8; 4];
    let sep = match y {
        Value::Str(s) => s.as_str(),
        Value::Char(c) => c.encode_utf8(&mut buf),
        v => return Err(Error::typ(klong_core::msg!(".join: y must be a string or a char, got {}", v.describe()))),
    };
    let n = x.len().unwrap_or(0);
    let mut out = try_string(0)?;
    for i in 0..n {
        if i > 0 {
            try_push_str(&mut out, sep)?;
        }
        match x.item(i) {
            Value::Str(s) => try_push_str(&mut out, s.as_str())?,
            Value::Char(c) => try_push_char(&mut out, c)?,
            v => return Err(Error::typ(klong_core::msg!(".join: x must be a list of strings, got a list with {}", v.describe()))),
        }
    }
    Ok(Value::str(out))
}

fn words(a: &[Value]) -> Result<Value> {
    let x = a.first().ok_or_else(|| Error::arity(".words takes 1 argument, got 0"))?;
    let text = string(x, ".words", "x")?;
    let mut out = try_vec(0)?;
    for w in text.split(is_ws).filter(|w| !w.is_empty()) {
        push(&mut out, w)?;
    }
    Value::from_items(out)
}

fn trim(a: &[Value]) -> Result<Value> {
    let x = a.first().ok_or_else(|| Error::arity(".trim takes 1 argument, got 0"))?;
    Value::text(string(x, ".trim", "x")?.trim_matches(is_ws))
}

fn num(ip: &mut Interp, a: &[Value]) -> Result<Value> {
    let x = a.first().ok_or_else(|| Error::arity(".num takes 1 argument, got 0"))?;
    let t = string(x, ".num", "x")?.trim_matches(is_ws);
    // the characters a number can be made of: anything else is not one (and the reader
    // would take other digits than 0-9 as well)
    if t.is_empty() || !t.bytes().all(|b| b.is_ascii_digit() || matches!(b, b'.' | b'e' | b'+' | b'-')) {
        return Ok(Value::Undefined);
    }
    let t = try_str(t)?;
    match ip.read_datum(&t) {
        Ok(Some((used, v))) if used == t.len() => match v {
            Value::Int(_) => Ok(v),
            Value::Flt(f) if f.is_finite() => Ok(v),
            _ => Ok(Value::Undefined),
        },
        // not a number, or not one the reader can take (a malformed or too large one)
        Ok(_) => Ok(Value::Undefined),
        Err(e) if e.kind == klong_core::Kind::Memory || e.kind == klong_core::Kind::Interrupt => Err(e),
        Err(_) => Ok(Value::Undefined),
    }
}

pub(crate) fn register(interp: &mut Interp) {
    interp.register(".split", 2, Rc::new(|_: &mut Interp, a: &[Value]| split(a)));
    interp.register(".join", 2, Rc::new(|_: &mut Interp, a: &[Value]| join(a)));
    interp.register(".words", 1, Rc::new(|_: &mut Interp, a: &[Value]| words(a)));
    interp.register(".trim", 1, Rc::new(|_: &mut Interp, a: &[Value]| trim(a)));
    interp.register(".num", 1, Rc::new(|ip: &mut Interp, a: &[Value]| num(ip, a)));
}

