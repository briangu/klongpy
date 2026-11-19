//! `.json` and `.unjson` (KlongPy `eval_sys_json` and `eval_sys_unjson` in `sys_fn.py`): a value
//! as compact JSON text and back, written by hand, in the text KlongPy gets from Python's `json`
//! module (`json.dumps(separators=(",", ":"), ensure_ascii=False)` and `json.loads`), messages
//! of bad text included. Differences, all in what a build cannot hold: an integer beyond the
//! build width and an escaped lone surrogate (`"\ud800"`) are a `range` error here.
use alloc::rc::Rc;
use alloc::vec::Vec;
use core::fmt;

use klong_core::error::Result;
use klong_core::fmt::{write_float, Sink};
use klong_core::value::{max_nest, too_deep, try_push, try_push_char, try_push_str, try_string, try_vec, Arr, Dict, MAX_RANK};
use klong_core::{Error, Flt, Int, Interp, Value};

/// Where a member that has no JSON form is, as KlongPy words it.
enum Seg {
    Index(usize),
    Key(Value),
}

struct Path<'a>(&'a [Seg]);

impl fmt::Display for Path<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for s in self.0 {
            match s {
                Seg::Index(i) => write!(f, " at index {}", i)?,
                Seg::Key(Value::Str(t)) => write!(f, " at key \"{}\"", t.as_str())?,
                Seg::Key(Value::Char(c)) => write!(f, " at key \"{}\"", c)?,
                Seg::Key(_) => {}
            }
        }
        Ok(())
    }
}

/// One number's text, formatted without the heap.
struct Num {
    buf: [u8; 40],
    len: usize,
}

impl Num {
    fn text(&self) -> &str {
        core::str::from_utf8(&self.buf[..self.len]).unwrap_or("")
    }
}

impl fmt::Write for Num {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        let b = s.as_bytes();
        let end = self.len + b.len();
        if end > self.buf.len() {
            return Err(fmt::Error);
        }
        self.buf[self.len..end].copy_from_slice(b);
        self.len = end;
        Ok(())
    }
}

impl Sink for Num {
    fn push(&mut self, c: char) {
        let mut b = [0u8; 4];
        let _ = fmt::Write::write_str(self, c.encode_utf8(&mut b));
    }
    fn push_str(&mut self, s: &str) {
        let _ = fmt::Write::write_str(self, s);
    }
}

struct Enc {
    out: alloc::string::String,
    path: Vec<Seg>,
}

impl Enc {
    fn put(&mut self, s: &str) -> Result<()> {
        try_push_str(&mut self.out, s)
    }

    fn quoted(&mut self, s: &str) -> Result<()> {
        self.put("\"")?;
        let b = s.as_bytes();
        let mut run = 0;
        for (i, &c) in b.iter().enumerate() {
            let esc = match c {
                b'"' => "\\\"",
                b'\\' => "\\\\",
                b'\n' => "\\n",
                b'\r' => "\\r",
                b'\t' => "\\t",
                8 => "\\b",
                12 => "\\f",
                0..=31 => "",
                _ => continue,
            };
            self.put(&s[run..i])?;
            run = i + 1;
            if esc.is_empty() {
                const HEX: &[u8; 16] = b"0123456789abcdef";
                let u = [b'\\', b'u', b'0', b'0', HEX[(c >> 4) as usize], HEX[(c & 15) as usize]];
                self.put(core::str::from_utf8(&u).unwrap_or(""))?;
            } else {
                self.put(esc)?;
            }
        }
        self.put(&s[run..])?;
        self.put("\"")
    }

    fn int(&mut self, i: Int) -> Result<()> {
        let mut n = Num { buf: [0; 40], len: 0 };
        let _ = fmt::Write::write_fmt(&mut n, format_args!("{}", i));
        self.put(n.text())
    }

    fn flt(&mut self, f: Flt) -> Result<()> {
        if !f.is_finite() {
            return self.put("null");
        }
        let mut n = Num { buf: [0; 40], len: 0 };
        write_float(f, &mut n);
        self.put(n.text())
    }

    /// A numeric array of any rank: `data` in row-major order with the sizes `dims`.
    fn nd<T: Copy>(&mut self, data: &[T], dims: &[usize], one: fn(&mut Enc, T) -> Result<()>) -> Result<()> {
        self.put("[")?;
        if dims.len() == 1 {
            for (i, &x) in data.iter().enumerate() {
                if i > 0 {
                    self.put(",")?;
                }
                one(self, x)?;
            }
        } else {
            let row: usize = dims[1..].iter().product();
            for i in 0..dims[0] {
                if i > 0 {
                    self.put(",")?;
                }
                self.nd(&data[i * row..(i + 1) * row], &dims[1..], one)?;
            }
        }
        self.put("]")
    }

    fn array<T: Copy>(&mut self, a: &Arr<T>, one: fn(&mut Enc, T) -> Result<()>) -> Result<()> {
        let mut dims = [0usize; MAX_RANK];
        let r = a.rank().clamp(1, MAX_RANK);
        for (i, d) in dims.iter_mut().enumerate().take(r) {
            *d = a.dim(i);
        }
        self.nd(&a.data, &dims[..r], one)
    }

    fn enter(&mut self, seg: Seg) -> Result<()> {
        try_push(&mut self.path, seg)
    }

    fn value(&mut self, v: &Value, depth: u32) -> Result<()> {
        match v {
            Value::Undefined => self.put("null"),
            Value::Int(i) => self.int(*i),
            Value::Flt(f) => self.flt(*f),
            #[cfg(feature = "bigint")]
            Value::Big(b) => self.put(&klong_core::big::text(b)),
            Value::Char(c) => {
                let mut b = [0u8; 4];
                self.quoted(c.encode_utf8(&mut b))
            }
            Value::Str(s) => self.quoted(s.as_str()),
            Value::Sym(_) => Err(Error::typ(klong_core::msg!(".json: a symbol has no JSON form{}", Path(&self.path)))
                .with_hint("convert it with $ first")),
            // a host array (a tensor) nested in a list or a dictionary is its numbers, as one given to .json itself is
            Value::Ext(_) => match v.unhost(".json")? {
                n if !matches!(n, Value::Ext(_)) => self.value(&n, depth),
                _ => Err(Error::typ(klong_core::msg!(".json: a function has no JSON form{}", Path(&self.path)))),
            },
            Value::Fn(_) | Value::Prim(_) => {
                Err(Error::typ(klong_core::msg!(".json: a function has no JSON form{}", Path(&self.path))))
            }
            Value::Ints(a) => self.array(a, |e, x| e.int(x)),
            Value::Flts(a) => self.array(a, |e, x| e.flt(x)),
            Value::List(l) => {
                if depth >= max_nest() {
                    return Err(too_deep());
                }
                self.put("[")?;
                for (i, x) in l.iter().enumerate() {
                    if i > 0 {
                        self.put(",")?;
                    }
                    self.enter(Seg::Index(i))?;
                    self.value(x, depth + 1)?;
                    self.path.pop();
                }
                self.put("]")
            }
            Value::Dict(d) => {
                if depth >= max_nest() {
                    return Err(too_deep());
                }
                let d = d.borrow();
                self.put("{")?;
                for (i, (k, x)) in d.keys().iter().zip(d.vals()).enumerate() {
                    if i > 0 {
                        self.put(",")?;
                    }
                    match k {
                        Value::Str(s) => self.quoted(s.as_str())?,
                        Value::Char(c) => {
                            let mut b = [0u8; 4];
                            self.quoted(c.encode_utf8(&mut b))?
                        }
                        k => {
                            return Err(Error::typ(klong_core::msg!(
                                ".json: a dictionary key must be a string, got {}{}",
                                k.describe(),
                                Path(&self.path)
                            ))
                            .with_hint("convert the key with $"))
                        }
                    }
                    self.put(":")?;
                    self.enter(Seg::Key(k.clone()))?;
                    self.value(x, depth + 1)?;
                    self.path.pop();
                }
                self.put("}")
            }
        }
    }
}

fn json(a: &[Value]) -> Result<Value> {
    let x = a.first().ok_or_else(|| Error::arity(".json takes 1 argument, got 0"))?;
    let x = &x.unhost(".json")?;
    let mut e = Enc { out: try_string(0)?, path: try_vec(0)? };
    e.value(x, 0)?;
    Ok(Value::str(e.out))
}

struct Parser<'a> {
    t: &'a str,
    s: &'a [u8],
    i: usize,
    /// Where the first lone surrogate escape was: the text is otherwise read to its end, so
    /// that bad text is reported as Python reports it, and then this is the error.
    lone: Option<usize>,
}

fn digit(b: Option<&u8>) -> bool {
    matches!(b, Some(b'0'..=b'9'))
}

fn hex4(s: &[u8]) -> Option<u32> {
    let mut n = 0u32;
    for &c in s.get(..4)? {
        n = n * 16 + (c as char).to_digit(16)?;
    }
    Some(n)
}

impl Parser<'_> {
    /// The range error for bad text at byte `pos`: the message, then where, counted in characters.
    fn fail<T>(&self, m: &str, pos: usize) -> Result<T> {
        let before = self.t.get(..pos).unwrap_or("");
        let line = before.bytes().filter(|&b| b == b'\n').count() + 1;
        let from = before.rfind('\n').map_or(0, |p| p + 1);
        let col = before[from..].chars().count() + 1;
        Err(Error::range(klong_core::msg!(
            ".unjson: {} at line {} column {} (character {})",
            m,
            line,
            col,
            before.chars().count()
        )))
    }

    fn ws(&mut self) {
        while matches!(self.s.get(self.i), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.i += 1;
        }
    }

    fn word(&self, w: &str) -> bool {
        self.s[self.i..].starts_with(w.as_bytes())
    }

    fn value(&mut self, depth: u32) -> Result<Value> {
        match self.s.get(self.i) {
            Some(b'"') => {
                let open = self.i;
                self.i += 1;
                Ok(Value::str(self.string(open)?))
            }
            Some(b'{') => self.object(depth),
            Some(b'[') => self.array(depth),
            Some(b'n') if self.word("null") => {
                self.i += 4;
                Ok(Value::Undefined)
            }
            Some(b't') if self.word("true") => {
                self.i += 4;
                Ok(Value::Int(1))
            }
            Some(b'f') if self.word("false") => {
                self.i += 5;
                Ok(Value::Int(0))
            }
            Some(b'N') if self.word("NaN") => Err(Error::range(".unjson: NaN is not JSON")),
            Some(b'I') if self.word("Infinity") => Err(Error::range(".unjson: Infinity is not JSON")),
            Some(b'-') if self.word("-Infinity") => Err(Error::range(".unjson: -Infinity is not JSON")),
            _ => self.number(),
        }
    }

    fn number(&mut self) -> Result<Value> {
        let s = self.s;
        let start = self.i;
        let mut j = start;
        if s.get(j) == Some(&b'-') {
            j += 1;
        }
        match s.get(j) {
            Some(b'1'..=b'9') => {
                while digit(s.get(j)) {
                    j += 1;
                }
            }
            Some(b'0') => j += 1,
            _ => return self.fail("Expecting value", start),
        }
        let mut real = false;
        if s.get(j) == Some(&b'.') && digit(s.get(j + 1)) {
            real = true;
            j += 1;
            while digit(s.get(j)) {
                j += 1;
            }
        }
        if matches!(s.get(j), Some(b'e' | b'E')) {
            let mut k = j + 1;
            if matches!(s.get(k), Some(b'+' | b'-')) {
                k += 1;
            }
            if digit(s.get(k)) {
                while digit(s.get(k)) {
                    k += 1;
                }
                real = true;
                j = k;
            }
        }
        let text = &self.t[start..j];
        self.i = j;
        if real {
            text.parse::<Flt>().map(Value::Flt).map_err(|_| Error::range(".unjson: bad number"))
        } else {
            match text.parse::<Int>() {
                Ok(i) => Ok(Value::Int(i)),
                #[cfg(feature = "bigint")]
                Err(_) => match klong_core::big::from_str(text) {
                    Some(b) => Ok(klong_core::big::norm(b)),
                    None => self.fail("Integer out of range for this build", start),
                },
                #[cfg(not(feature = "bigint"))]
                Err(_) => self.fail("Integer out of range for this build", start),
            }
        }
    }

    /// The string after its opening quote (at `open`).
    fn string(&mut self, open: usize) -> Result<alloc::string::String> {
        let s = self.s;
        let mut out = try_string(0)?;
        loop {
            let from = self.i;
            let mut j = from;
            while j < s.len() && s[j] != b'"' && s[j] != b'\\' && s[j] >= 0x20 {
                j += 1;
            }
            try_push_str(&mut out, &self.t[from..j])?;
            match s.get(j) {
                None => return self.fail("Unterminated string starting at", open),
                Some(b'"') => {
                    self.i = j + 1;
                    return Ok(out);
                }
                Some(b'\\') => {}
                Some(_) => return self.fail("Invalid control character at", j),
            }
            let c = match s.get(j + 1) {
                None => return self.fail("Unterminated string starting at", open),
                Some(&c) => c,
            };
            self.i = j + 2;
            let ch = match c {
                b'"' | b'\\' | b'/' => c as char,
                b'b' => '\u{8}',
                b'f' => '\u{c}',
                b'n' => '\n',
                b'r' => '\r',
                b't' => '\t',
                b'u' => {
                    let at = j + 1;
                    let end = at + 5;
                    // as Python: the four digits and at least one character after them
                    if end >= s.len() {
                        return self.fail("Invalid \\uXXXX escape", at);
                    }
                    let mut u = match hex4(&s[at + 1..]) {
                        Some(u) => u,
                        None => return self.fail("Invalid \\uXXXX escape", at),
                    };
                    self.i = end;
                    if (0xD800..0xDC00).contains(&u) && end + 6 < s.len() && s[end] == b'\\' && s[end + 1] == b'u' {
                        let lo = match hex4(&s[end + 2..]) {
                            Some(l) => l,
                            None => return self.fail("Invalid \\uXXXX escape", end + 1),
                        };
                        if (0xDC00..0xE000).contains(&lo) {
                            u = 0x10000 + ((u - 0xD800) << 10) + (lo - 0xDC00);
                            self.i = end + 6;
                        }
                    }
                    // Python keeps a lone surrogate in the string; a Rust string cannot
                    char::from_u32(u).unwrap_or_else(|| {
                        self.lone.get_or_insert(j);
                        '\u{fffd}'
                    })
                }
                _ => return self.fail("Invalid \\escape", j),
            };
            try_push_char(&mut out, ch)?;
        }
    }

    fn array(&mut self, depth: u32) -> Result<Value> {
        if depth >= max_nest() {
            return Err(too_deep());
        }
        self.i += 1;
        self.ws();
        if self.s.get(self.i) == Some(&b']') {
            self.i += 1;
            return Ok(Value::empty());
        }
        let mut items: Vec<Value> = try_vec(0)?;
        loop {
            let v = self.value(depth + 1)?;
            try_push(&mut items, v)?;
            self.ws();
            match self.s.get(self.i) {
                Some(b']') => {
                    self.i += 1;
                    return Value::from_items(items);
                }
                Some(b',') => {
                    self.i += 1;
                    self.ws();
                }
                _ => return self.fail("Expecting ',' delimiter", self.i),
            }
        }
    }

    fn object(&mut self, depth: u32) -> Result<Value> {
        if depth >= max_nest() {
            return Err(too_deep());
        }
        self.i += 1;
        self.ws();
        let mut d = Dict::default();
        if self.s.get(self.i) == Some(&b'}') {
            self.i += 1;
            return Ok(Value::dict(d));
        }
        loop {
            if self.s.get(self.i) != Some(&b'"') {
                return self.fail("Expecting property name enclosed in double quotes", self.i);
            }
            let open = self.i;
            self.i += 1;
            let k = self.string(open)?;
            self.ws();
            if self.s.get(self.i) != Some(&b':') {
                return self.fail("Expecting ':' delimiter", self.i);
            }
            self.i += 1;
            self.ws();
            let v = self.value(depth + 1)?;
            d.set(Value::str(k), v)?;
            self.ws();
            match self.s.get(self.i) {
                Some(b'}') => {
                    self.i += 1;
                    return Ok(Value::dict(d));
                }
                Some(b',') => {
                    self.i += 1;
                    self.ws();
                }
                _ => return self.fail("Expecting ',' delimiter", self.i),
            }
        }
    }
}

fn unjson(a: &[Value]) -> Result<Value> {
    let x = a.first().ok_or_else(|| Error::arity(".unjson takes 1 argument, got 0"))?;
    let t = match x {
        Value::Str(s) => s.as_str(),
        v => return Err(Error::typ(klong_core::msg!(".unjson: x must be a string, got {}", v.describe()))),
    };
    let mut p = Parser { t, s: t.as_bytes(), i: 0, lone: None };
    if t.starts_with('\u{feff}') {
        return p.fail("Unexpected UTF-8 BOM (decode using utf-8-sig)", 0);
    }
    p.ws();
    let v = p.value(0)?;
    p.ws();
    if p.i != t.len() {
        return p.fail("Extra data", p.i);
    }
    match p.lone {
        Some(at) => p.fail("Unpaired surrogate escape", at),
        None => Ok(v),
    }
}

pub(crate) fn register(interp: &mut Interp) {
    interp.register(".json", 1, Rc::new(|_: &mut Interp, a: &[Value]| json(a)));
    interp.register(".unjson", 1, Rc::new(|_: &mut Interp, a: &[Value]| unjson(a)));
}
