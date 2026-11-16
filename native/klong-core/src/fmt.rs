//! Printing: byte for byte what `klongpy/writer.py` prints.
use alloc::string::String;
use core::fmt::Write;

use crate::sym::Interner;
use crate::tree::Node;
use crate::value::{Flt, Func, Value, WILDCARD};

/// A float as Python's `repr` prints it: the shortest digits that read back as the same
/// number, in exponent form when the decimal exponent is below -4 or at least 16.
/// Where formatted text goes: a `String`, or [`Checked`], which stops at the first failed
/// allocation instead of aborting (the text of a value is sized by data).
pub trait Sink: core::fmt::Write {
    fn push(&mut self, c: char);
    fn push_str(&mut self, s: &str);
}

impl Sink for String {
    fn push(&mut self, c: char) {
        String::push(self, c)
    }
    fn push_str(&mut self, s: &str) {
        String::push_str(self, s)
    }
}

#[derive(Default)]
pub struct Checked {
    pub text: String,
    pub oom: bool,
}

impl Sink for Checked {
    fn push(&mut self, c: char) {
        if !self.oom && self.text.try_reserve(4).is_ok() {
            self.text.push(c);
        } else {
            self.oom = true;
        }
    }
    fn push_str(&mut self, s: &str) {
        if !self.oom && self.text.try_reserve(s.len()).is_ok() {
            self.text.push_str(s);
        } else {
            self.oom = true;
        }
    }
}

impl core::fmt::Write for Checked {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        Sink::push_str(self, s);
        Ok(())
    }
}

/// The text of a value, or a `memory` error.
pub fn format_value(syms: &Interner, v: &Value, display: bool) -> crate::error::Result<String> {
    let mut out = Checked::default();
    write_value(syms, v, display, &mut out);
    if out.oom {
        return Err(crate::error::Error::memory());
    }
    Ok(out.text)
}

pub fn write_float<S: Sink>(f: Flt, out: &mut S) {
    if f.is_nan() {
        out.push_str("nan");
        return;
    }
    if f.is_infinite() {
        out.push_str(if f < 0.0 { "-inf" } else { "inf" });
        return;
    }
    if f.is_sign_negative() {
        out.push('-');
    }
    let a = if f.is_sign_negative() { -f } else { f };
    // `{:e}` gives the shortest round-trip digits: "d.ddde-7", "1e16", "0e0".
    let mut buf = Small::new();
    let _ = write!(buf, "{:e}", a);
    let s = buf.as_str();
    let (mant, exp) = match s.split_once('e') {
        Some((m, e)) => (m, e.parse::<i32>().unwrap_or(0)),
        None => (s, 0),
    };
    let mut digits = Small::new();
    for c in mant.chars() {
        if c != '.' {
            let _ = digits.write_char(c);
        }
    }
    let d = digits.as_str();
    if (-4..16).contains(&exp) {
        if exp >= 0 {
            let int_len = exp as usize + 1;
            if d.len() <= int_len {
                out.push_str(d);
                for _ in d.len()..int_len {
                    out.push('0');
                }
                out.push_str(".0");
            } else {
                out.push_str(&d[..int_len]);
                out.push('.');
                out.push_str(&d[int_len..]);
            }
        } else {
            out.push_str("0.");
            for _ in 0..(-exp - 1) {
                out.push('0');
            }
            out.push_str(d);
        }
    } else {
        out.push_str(&d[..1]);
        if d.len() > 1 {
            out.push('.');
            out.push_str(&d[1..]);
        }
        out.push('e');
        out.push(if exp < 0 { '-' } else { '+' });
        let e = exp.unsigned_abs();
        if e < 10 {
            out.push('0');
        }
        let _ = write!(out, "{}", e);
    }
}

/// A fixed buffer for formatting one number without the heap.
struct Small {
    buf: [u8; 40],
    len: usize,
}

impl Small {
    fn new() -> Small {
        Small { buf: [0; 40], len: 0 }
    }
    fn as_str(&self) -> &str {
        core::str::from_utf8(&self.buf[..self.len]).unwrap_or("")
    }
}

impl Write for Small {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        let b = s.as_bytes();
        if self.len + b.len() > self.buf.len() {
            return Err(core::fmt::Error);
        }
        self.buf[self.len..self.len + b.len()].copy_from_slice(b);
        self.len += b.len();
        Ok(())
    }
}

pub fn arity_name(arity: u8) -> &'static str {
    match arity {
        0 => ":nilad",
        1 => ":monad",
        2 => ":dyad",
        _ => ":triad",
    }
}

fn push_bytes<S: Sink>(out: &mut S, b: &[u8]) {
    match core::str::from_utf8(b) {
        Ok(s) => out.push_str(s),
        Err(_) => {
            for &c in b {
                out.push(if c < 0x80 { c as char } else { '\u{fffd}' });
            }
        }
    }
}

/// The printed form of a value. `display` is what `.d` and `.p` use for strings and chars
/// (no quotes).
pub fn write_value<S: Sink>(syms: &Interner, v: &Value, display: bool, out: &mut S) {
    match v {
        Value::Undefined => out.push_str(":undefined"),
        Value::Int(i) => {
            let _ = write!(out, "{}", i);
        }
        Value::Flt(f) => write_float(*f, out),
        Value::Sym(s) => {
            if !display {
                out.push(':');
            }
            out.push_str(syms.name(*s));
        }
        Value::Char(c) => {
            if !display {
                out.push_str("0c");
            }
            push_bytes(out, &[*c]);
        }
        Value::Str(s) => {
            if display {
                push_bytes(out, &s.0);
            } else {
                out.push('"');
                let mut start = 0;
                for (i, &c) in s.0.iter().enumerate() {
                    if c == b'"' {
                        push_bytes(out, &s.0[start..=i]);
                        start = i;
                    }
                }
                push_bytes(out, &s.0[start..]);
                out.push('"');
            }
        }
        Value::Ints(_) | Value::Flts(_) | Value::List(_) => {
            out.push('[');
            let n = v.len().unwrap_or(0);
            for i in 0..n {
                if i > 0 {
                    out.push(' ');
                }
                write_value(syms, &v.item(i), display, out);
            }
            out.push(']');
        }
        Value::Dict(d) => {
            out.push_str(":{");
            let d = d.borrow();
            for i in 0..d.keys.len() {
                if i > 0 {
                    out.push(' ');
                }
                out.push('[');
                write_value(syms, &d.keys[i], display, out);
                out.push(' ');
                write_value(syms, &d.vals[i], display, out);
                out.push(']');
            }
            out.push('}');
        }
        Value::Fn(f) => out.push_str(arity_name(func_arity(f))),
        Value::Prim(p) => out.push_str(syms.name(*p)),
        Value::Ext(e) => {
            out.push(':');
            out.push_str(e.0.type_name());
        }
    }
}

pub fn func_arity(f: &Func) -> u8 {
    match f {
        Func::Code { tree, node } => match tree.node(*node) {
            // a projection displays by the number of arguments it still takes
            Node::Fn { arity, args, has_args, .. } => {
                let holes = if *has_args { tree.ids(*args).iter().filter(|&&q| q == crate::tree::NONE.0).count() } else { 0 };
                if holes > 0 {
                    holes as u8
                } else {
                    *arity
                }
            }
            Node::Monad { .. } => 1,
            Node::Dyad { .. } => 2,
            Node::Adv { dyad, .. } => {
                if *dyad {
                    2
                } else {
                    1
                }
            }
            _ => 0,
        },
        Func::Proj { args, .. } => {
            let holes = args.iter().filter(|q| matches!(q, crate::value::Slot::Hole)).count();
            (if holes > 0 { holes } else { args.len() }).min(255) as u8
        }
        Func::Native { arity, .. } => {
            if *arity == WILDCARD {
                0
            } else {
                *arity
            }
        }
    }
}

pub fn to_string(syms: &Interner, v: &Value) -> String {
    let mut s = String::new();
    write_value(syms, v, false, &mut s);
    s
}
