//! Shared by the test files: a small JSON reader (the vectors file is the only JSON these tests
//! read, so a dependency is not worth it), structural equality of values, a seeded generator.
#![allow(dead_code)]
use std::collections::BTreeMap;
use std::rc::Rc;

use klong_core::value::Arr;
use klong_core::{Flt, Int, Interp, Value};
use klong_ipc::codec::{ext_ref, KdbError, KdbLambda, KdbOpaque};

// ----- JSON ------------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum Json {
    Null,
    Bool(bool),
    /// The digits as written: an integer may need all 64 bits.
    Num(String),
    Str(String),
    Arr(Vec<Json>),
    Obj(BTreeMap<String, Json>),
}

impl Json {
    pub fn get(&self, k: &str) -> Option<&Json> {
        match self {
            Json::Obj(m) => m.get(k),
            _ => None,
        }
    }
    pub fn str(&self) -> &str {
        match self {
            Json::Str(s) => s,
            other => panic!("not a string: {:?}", other),
        }
    }
    pub fn arr(&self) -> &[Json] {
        match self {
            Json::Arr(a) => a,
            other => panic!("not an array: {:?}", other),
        }
    }
}

struct P<'a> {
    s: &'a [u8],
    i: usize,
}

impl P<'_> {
    fn ws(&mut self) {
        while self.i < self.s.len() && self.s[self.i].is_ascii_whitespace() {
            self.i += 1;
        }
    }
    fn eat(&mut self, c: u8) {
        self.ws();
        assert_eq!(self.s[self.i], c, "json: expected {:?} at {}", c as char, self.i);
        self.i += 1;
    }
    fn hex4(&mut self) -> u32 {
        let h = std::str::from_utf8(&self.s[self.i..self.i + 4]).unwrap();
        self.i += 4;
        u32::from_str_radix(h, 16).unwrap()
    }
    fn string(&mut self) -> String {
        self.eat(b'"');
        let mut out = Vec::new();
        loop {
            let c = self.s[self.i];
            self.i += 1;
            match c {
                b'"' => break,
                b'\\' => {
                    let e = self.s[self.i];
                    self.i += 1;
                    match e {
                        b'n' => out.push(b'\n'),
                        b't' => out.push(b'\t'),
                        b'r' => out.push(b'\r'),
                        b'b' => out.push(8),
                        b'f' => out.push(12),
                        b'u' => {
                            let mut cp = self.hex4();
                            if (0xd800..0xdc00).contains(&cp) {
                                assert_eq!(&self.s[self.i..self.i + 2], b"\\u");
                                self.i += 2;
                                let lo = self.hex4();
                                cp = 0x10000 + ((cp - 0xd800) << 10) + (lo - 0xdc00);
                            }
                            let mut b = [0u8; 4];
                            out.extend_from_slice(char::from_u32(cp).unwrap().encode_utf8(&mut b).as_bytes());
                        }
                        other => out.push(other),
                    }
                }
                other => out.push(other),
            }
        }
        String::from_utf8(out).unwrap()
    }
    fn value(&mut self) -> Json {
        self.ws();
        match self.s[self.i] {
            b'{' => {
                self.i += 1;
                let mut m = BTreeMap::new();
                self.ws();
                if self.s[self.i] == b'}' {
                    self.i += 1;
                    return Json::Obj(m);
                }
                loop {
                    self.ws();
                    let k = self.string();
                    self.eat(b':');
                    m.insert(k, self.value());
                    self.ws();
                    if self.s[self.i] == b',' {
                        self.i += 1;
                    } else {
                        self.eat(b'}');
                        return Json::Obj(m);
                    }
                }
            }
            b'[' => {
                self.i += 1;
                let mut a = Vec::new();
                self.ws();
                if self.s[self.i] == b']' {
                    self.i += 1;
                    return Json::Arr(a);
                }
                loop {
                    a.push(self.value());
                    self.ws();
                    if self.s[self.i] == b',' {
                        self.i += 1;
                    } else {
                        self.eat(b']');
                        return Json::Arr(a);
                    }
                }
            }
            b'"' => Json::Str(self.string()),
            b't' => {
                self.i += 4;
                Json::Bool(true)
            }
            b'f' => {
                self.i += 5;
                Json::Bool(false)
            }
            b'n' => {
                self.i += 4;
                Json::Null
            }
            _ => {
                let start = self.i;
                while self.i < self.s.len() && matches!(self.s[self.i], b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9') {
                    self.i += 1;
                }
                Json::Num(String::from_utf8(self.s[start..self.i].to_vec()).unwrap())
            }
        }
    }
}

pub fn parse_json(text: &str) -> Json {
    let mut p = P { s: text.as_bytes(), i: 0 };
    let v = p.value();
    p.ws();
    assert_eq!(p.i, p.s.len(), "json: trailing text");
    v
}

pub fn unhex(h: &str) -> Vec<u8> {
    (0..h.len() / 2).map(|i| u8::from_str_radix(&h[2 * i..2 * i + 2], 16).unwrap()).collect()
}

// ----- values ----------------------------------------------------------------------------

fn flt_same(a: Flt, b: Flt) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

fn shape<T: Copy>(a: &Arr<T>) -> Vec<usize> {
    (0..a.rank().max(1)).map(|i| a.dim(i)).collect()
}

/// Exactly the same value: kinds, shapes, bits (any NaN equals any NaN), dictionary order.
pub fn same(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Int(x), Value::Int(y)) => x == y,
        (Value::Flt(x), Value::Flt(y)) => flt_same(*x, *y),
        (Value::Char(x), Value::Char(y)) => x == y,
        (Value::Sym(x), Value::Sym(y)) => x == y,
        (Value::Undefined, Value::Undefined) => true,
        (Value::Str(x), Value::Str(y)) => x.as_str() == y.as_str(),
        (Value::Ints(x), Value::Ints(y)) => shape(x) == shape(y) && x.data == y.data,
        (Value::Flts(x), Value::Flts(y)) => shape(x) == shape(y) && x.data.len() == y.data.len() && x.data.iter().zip(&y.data).all(|(p, q)| flt_same(*p, *q)),
        (Value::List(x), Value::List(y)) => x.len() == y.len() && x.iter().zip(y.iter()).all(|(p, q)| same(p, q)),
        (Value::Dict(x), Value::Dict(y)) => {
            let (x, y) = (x.borrow(), y.borrow());
            x.len() == y.len() && x.keys().iter().zip(y.keys()).all(|(p, q)| same(p, q)) && x.vals().iter().zip(y.vals()).all(|(p, q)| same(p, q))
        }
        (Value::Ext(_), Value::Ext(_)) => {
            if let (Some(p), Some(q)) = (ext_ref::<KdbLambda>(a), ext_ref::<KdbLambda>(b)) {
                return p.context == q.context && p.text == q.text;
            }
            if let (Some(p), Some(q)) = (ext_ref::<KdbOpaque>(a), ext_ref::<KdbOpaque>(b)) {
                return p.raw == q.raw;
            }
            if let (Some(p), Some(q)) = (ext_ref::<KdbError>(a), ext_ref::<KdbError>(b)) {
                return p.text == q.text;
            }
            false
        }
        _ => false,
    }
}

pub fn show(ip: &Interp, v: &Value) -> String {
    if let Some(l) = ext_ref::<KdbLambda>(v) {
        return format!("lambda({:?},{:?})", String::from_utf8_lossy(&l.context), String::from_utf8_lossy(&l.text));
    }
    if let Some(e) = ext_ref::<KdbError>(v) {
        return format!("error({:?})", String::from_utf8_lossy(&e.text));
    }
    let kind = match v {
        Value::Ints(_) => "ints",
        Value::Flts(_) => "flts",
        Value::List(_) => "list",
        _ => "",
    };
    format!("{}{}", kind, ip.format(v))
}

pub fn interp() -> Interp {
    Interp::new(klong_core::Config::default(), Box::new(klong_core::NoHooks))
}

// ----- a seeded generator ----------------------------------------------------------------

/// xorshift64*: the tests must give the same bytes on every run.
pub struct Rng(pub u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1)
    }
    pub fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    /// 0..n
    pub fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (self.next() >> 11) as usize % n
        }
    }
    pub fn chance(&mut self, p: f64) -> bool {
        ((self.next() >> 11) as f64 / (1u64 << 53) as f64) < p
    }
    pub fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| self.next() as u8).collect()
    }
}

/// A random value in Klong's normal form that survives a round trip unchanged.
pub fn random_value(ip: &mut Interp, r: &mut Rng, depth: usize) -> Value {
    let pick = r.below(if depth >= 4 { 9 } else { 13 });
    match pick {
        0 => Value::Int(r.next() as Int >> r.below(60)),
        1 => Value::Flt(match r.below(6) {
            0 => Flt::NAN,
            1 => Flt::INFINITY,
            2 => -0.0,
            _ => (r.next() as i64 as f64 / 1e6) as Flt,
        }),
        2 => Value::Char((b' ' + r.below(95) as u8) as char),
        3 => {
            let names = ["a", "b", "abc", "x.y", "", "with space", "é"];
            Value::Sym(ip.syms.intern(names[r.below(names.len())]))
        }
        4 => Value::Undefined,
        5 => {
            let n = r.below(6);
            let s: String = (0..n).map(|_| ['a', 'b', ' ', '"', 'é', '\u{4e16}', '\0'][r.below(7)]).collect();
            Value::str(s)
        }
        6 => Value::ints((0..r.below(8)).map(|_| r.next() as Int >> r.below(60)).collect()),
        7 => Value::flts((0..1 + r.below(7)).map(|_| (r.next() as i32 as f64 / 16.0) as Flt).collect()),
        8 => {
            // a rank 2 or 3 array
            let (a, b) = (1 + r.below(3), 1 + r.below(3));
            if r.below(2) == 0 {
                Value::Ints(Rc::new(Arr::shaped((0..a * b).map(|i| i as Int).collect(), &[a as u32, b as u32])))
            } else {
                let c = 1 + r.below(2);
                Value::Flts(Rc::new(Arr::shaped((0..a * b * c).map(|i| i as Flt / 2.0).collect(), &[a as u32, b as u32, c as u32])))
            }
        }
        9 | 10 => {
            let n = r.below(5);
            let items: Vec<Value> = (0..n).map(|_| random_value(ip, r, depth + 1)).collect();
            Value::from_items(items).unwrap()
        }
        11 => {
            let n = 1 + r.below(4);
            let items: Vec<Value> = (0..n).map(|_| Value::Sym(ip.syms.intern(["p", "q", "r"][r.below(3)]))).collect();
            Value::raw_list(items).unwrap()
        }
        _ => {
            let mut d = klong_core::value::Dict::default();
            let style = r.below(3);
            for i in 0..r.below(5) {
                let k = match style {
                    0 => Value::Int(i as Int * 3),
                    1 => Value::Sym(ip.syms.intern(["k0", "k1", "k2", "k3", "k4"][i])),
                    _ => {
                        if i % 2 == 0 {
                            Value::str(format!("s{}", i))
                        } else {
                            Value::Int(i as Int)
                        }
                    }
                };
                let v = random_value(ip, r, depth + 1);
                d.set(k, v).unwrap();
            }
            Value::dict(d)
        }
    }
}

/// Damage a byte string the way a broken or hostile peer would: flips, truncation, inserted
/// bytes, counts and type bytes overwritten with extreme values.
pub fn mutate(r: &mut Rng, b: &mut Vec<u8>) {
    for _ in 0..1 + r.below(3) {
        match r.below(8) {
            0 if !b.is_empty() => {
                let i = r.below(b.len());
                b[i] ^= 1 << r.below(8);
            }
            1 if !b.is_empty() => {
                let i = r.below(b.len());
                b[i] = r.next() as u8;
            }
            2 if !b.is_empty() => b.truncate(r.below(b.len())),
            3 => {
                let i = r.below(b.len() + 1);
                b.insert(i, r.next() as u8);
            }
            4 if b.len() >= 4 => {
                // a count field of the worst kind
                let i = r.below(b.len() - 3);
                let v: u32 = [0x7fff_ffff, 0xffff_ffff, 0x8000_0000, 1_000_000_000, 0x0100_0000, 65536][r.below(6)];
                b[i..i + 4].copy_from_slice(&v.to_le_bytes());
            }
            5 if !b.is_empty() => {
                // an interesting type byte anywhere
                let i = r.below(b.len());
                b[i] = [0, 1, 2, 7, 10, 11, 98, 99, 100, 101, 104, 106, 127, 128, 0xf5, 0xf6, 0xfe, 3, 20, 112][r.below(20)];
            }
            6 if b.len() > 2 => {
                let (i, j) = (r.below(b.len()), r.below(b.len()));
                let (lo, hi) = (i.min(j), i.max(j));
                let piece: Vec<u8> = b[lo..hi].to_vec();
                let at = r.below(b.len());
                for (k, x) in piece.into_iter().enumerate() {
                    b.insert(at + k, x);
                }
            }
            _ => {
                let n = r.below(9);
                b.extend(r.bytes(n));
            }
        }
    }
}

/// The hex strings of the vectors file: valid and invalid messages to start mutations from.
pub fn vector_messages() -> Vec<Vec<u8>> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/ipc_vectors/vectors.json");
    let text = std::fs::read_to_string(path).expect("tests/ipc_vectors/vectors.json");
    parse_json(&text).arr().iter().map(|e| unhex(e.get("hex").unwrap().str())).collect()
}
