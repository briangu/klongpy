//! Adverbs (KlongPy `klongpy/adverbs.py`). An adverb receives the chain and the level of the
//! function it modifies; it calls that function with `Interp::chain_call`.
use alloc::vec::Vec;

use crate::error::{Error, Result};
use crate::interp::{Chain, Interp};
use crate::sym::SymId;
use crate::tree::Node;
use crate::value::{matches, try_push, try_vec, Flt, Int, Value};

fn is_close(a: Flt, b: Flt) -> bool {
    if a == b || (a.is_nan() && b.is_nan()) {
        return true;
    }
    if !a.is_finite() || !b.is_finite() {
        return false;
    }
    let d = if a > b { a - b } else { b - a };
    let m = if b < 0.0 { -b } else { b };
    d <= 1e-8 + 1e-5 * m
}

fn converged(p: &Value, q: &Value) -> bool {
    match (p.as_flt(), q.as_flt()) {
        (Some(a), Some(b)) => is_close(a, b),
        _ => core::mem::discriminant(p) == core::mem::discriminant(q) && matches(p, q) || (p.is_list() && q.is_list() && matches(p, q)),
    }
}

fn iteration_count(a: &Value, name: &str) -> Result<usize> {
    match a {
        Value::Int(i) if *i >= 0 => Ok(*i as usize),
        Value::Int(i) => Err(Error::range(crate::msg!("{}: count must not be negative, got {}", name, i))),
        Value::Flt(f) if f.is_finite() && *f == (*f as Int) as Flt => {
            if *f < 0.0 {
                Err(Error::range(crate::msg!("{}: count must not be negative, got {}", name, a.py_repr(None))))
            } else {
                Ok(*f as usize)
            }
        }
        Value::Flt(_) => Err(Error::range(crate::msg!("{}: count must be a non-negative integer, got {}", name, a.py_repr(None)))),
        _ => Err(Error::typ(crate::msg!("{}: count must be a non-negative integer, got {}", name, a.py_repr(None)))),
    }
}

/// A scalar for each-left and each-right: anything that is not a list or a string.
fn is_scalar_operand(b: &Value) -> bool {
    !b.is_iterable()
}

/// The elements joined back: a string when every result is a char or a string.
fn join_text(items: Vec<Value>) -> Result<Value> {
    if !items.is_empty() && items.iter().all(|q| matches!(q, Value::Char(_))) {
        let mut s = crate::value::try_string(items.len())?;
        for q in &items {
            if let Value::Char(c) = q {
                crate::value::try_push_char(&mut s, *c)?;
            }
        }
        return Ok(Value::str(s));
    }
    Value::from_items(items)
}

/// numpy's pairwise summation, so that `+/` over floats gives the digits KlongPy gives.
fn pairwise_sum(a: &[Flt]) -> Flt {
    let n = a.len();
    if n < 8 {
        let mut res: Flt = 0.0;
        for &v in a {
            res += v;
        }
        res
    } else if n <= 128 {
        let mut r = [a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]];
        let mut i = 8;
        while i < n - (n % 8) {
            for j in 0..8 {
                r[j] += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        pairwise_sum(&a[..n2]) + pairwise_sum(&a[n2..])
    }
}

impl Interp {
    fn f1(&mut self, ch: &Chain, level: usize, x: Value) -> Result<Value> {
        self.tick()?;
        self.chain_call(ch, level - 1, x, None)
    }

    fn f2(&mut self, ch: &Chain, level: usize, x: Value, y: Value) -> Result<Value> {
        self.tick()?;
        self.chain_call(ch, level - 1, x, Some(y))
    }

    /// The operator glyph of the verb, when this adverb modifies the verb directly.
    fn verb_op<'a>(&'a self, ch: &Chain, level: usize) -> Option<&'a str> {
        if level != 1 {
            return None;
        }
        match ch.tree.node(ch.verb) {
            Node::OpVal { op, .. } => Some(self.syms.name(*op)),
            _ => None,
        }
    }

    /// SPIKE: a host array handed to an adverb that is not forwarded becomes a native value.
    fn unhost(v: Value) -> Result<Value> {
        if let Value::Ext(e) = &v {
            if let Some(r) = e.0.host_native() {
                return r;
            }
        }
        Ok(v)
    }

    pub(crate) fn adverb1(&mut self, ch: &Chain, level: usize, glyph: SymId, a: Value) -> Result<Value> {
        let g = self.syms.name(glyph);
        let a = if g == "/" { a } else { Self::unhost(a)? };
        match g {
            "'" => self.each(ch, level, a),
            "/" => self.over(ch, level, a),
            "\\" => self.scan(ch, level, a),
            ":~" => {
                let mut x = self.f1(ch, level, a)?;
                let mut xx = self.f1(ch, level, x.clone())?;
                while !converged(&x, &xx) {
                    x = xx;
                    xx = self.f1(ch, level, x.clone())?;
                }
                Ok(x)
            }
            "\\~" => {
                let mut x = a.clone();
                let mut xx = self.f1(ch, level, a.clone())?;
                let mut r = alloc::vec![a, xx.clone()];
                while !converged(&x, &xx) {
                    x = xx;
                    xx = self.f1(ch, level, x.clone())?;
                    try_push(&mut r, xx.clone())?;
                }
                r.pop();
                Value::from_items(r)
            }
            ":'" => self.each_pair(ch, level, a),
            "@'" => self.each_index(ch, level, a),
            _ => Err(Error::typ(crate::msg!("adverb {} needs two operands", g))),
        }
    }

    pub(crate) fn adverb2(&mut self, ch: &Chain, level: usize, glyph: SymId, a: Value, b: Value) -> Result<Value> {
        let g = self.syms.name(glyph);
        let (a, b) = (Self::unhost(a)?, Self::unhost(b)?);
        match g {
            "'" => {
                if a.is_empty() || b.is_empty() {
                    return Ok(if a.is_list() || b.is_list() { Value::empty() } else { Value::empty_str() });
                }
                if a.is_atom() && b.is_atom() {
                    return self.f2(ch, level, a, b);
                }
                if !a.is_iterable() || !b.is_iterable() {
                    return Err(Error::typ("': each-2 needs two lists or two atoms"));
                }
                let n = a.len().unwrap_or(0).min(b.len().unwrap_or(0));
                let mut out = try_vec(n)?;
                for i in 0..n {
                    out.push(self.f2(ch, level, a.item(i), b.item(i))?);
                }
                join_text(out)
            }
            "/" => {
                if b.is_empty() {
                    return Ok(a);
                }
                if b.is_atom() {
                    return self.f2(ch, level, a, b);
                }
                let mut acc = a;
                for i in 0..b.len().unwrap_or(0) {
                    acc = self.f2(ch, level, acc, b.item(i))?;
                }
                Ok(acc)
            }
            "\\" => {
                if b.is_empty() {
                    return Ok(a);
                }
                let items = if b.is_atom() { alloc::vec![b] } else { b.items()? };
                let mut out = try_vec(items.len() + 1)?;
                out.push(a.clone());
                let mut acc = a;
                for q in items {
                    acc = self.f2(ch, level, acc, q)?;
                    out.push(acc.clone());
                }
                Value::from_items(out)
            }
            "\\~" => {
                let mut b = b;
                let mut r = alloc::vec![b.clone()];
                while self.apply(&a, alloc::vec![b.clone()], None)?.truthy() {
                    b = self.f1(ch, level, b)?;
                    try_push(&mut r, b.clone())?;
                }
                r.pop();
                Value::from_items(r)
            }
            "\\*" => {
                let n = iteration_count(&a, "\\*")?;
                if n == 0 {
                    return Ok(b);
                }
                let mut b = b;
                let mut r = alloc::vec![b.clone()];
                for _ in 0..n {
                    b = self.f1(ch, level, b)?;
                    try_push(&mut r, b.clone())?;
                }
                Value::from_items(r)
            }
            ":\\" => {
                if is_scalar_operand(&b) {
                    return self.f2(ch, level, a, b);
                }
                let n = b.len().unwrap_or(0);
                let mut out = try_vec(n)?;
                for i in 0..n {
                    out.push(self.f2(ch, level, a.clone(), b.item(i))?);
                }
                Value::from_items(out)
            }
            ":/" => {
                if is_scalar_operand(&b) {
                    return self.f2(ch, level, b, a);
                }
                let n = b.len().unwrap_or(0);
                let mut out = try_vec(n)?;
                for i in 0..n {
                    out.push(self.f2(ch, level, b.item(i), a.clone())?);
                }
                Value::from_items(out)
            }
            ":*" => {
                let n = iteration_count(&a, ":*")?;
                let mut b = b;
                for _ in 0..n {
                    b = self.f1(ch, level, b)?;
                }
                Ok(b)
            }
            ":~" => {
                let mut b = b;
                while self.apply(&a, alloc::vec![b.clone()], None)?.truthy() {
                    b = self.f1(ch, level, b)?;
                }
                Ok(b)
            }
            // each-pair and each-index take one operand; written between two, they are applied
            // to the left one (KlongPy passes the right one where the verb's glyph goes).
            ":'" => self.each_pair(ch, level, a),
            "@'" => self.each_index(ch, level, a),
            _ => Err(Error::typ(crate::msg!("unknown adverb {}", g))),
        }
    }

    fn each(&mut self, ch: &Chain, level: usize, a: Value) -> Result<Value> {
        if let Value::Dict(d) = &a {
            let pairs: Vec<(Value, Value)> = {
                let d = d.borrow();
                let mut pairs = try_vec(d.len())?;
                pairs.extend(d.keys().iter().cloned().zip(d.vals().iter().cloned()));
                pairs
            };
            let mut out = try_vec(pairs.len())?;
            for (k, v) in pairs {
                let pair = Value::from_items(alloc::vec![k, v])?;
                out.push(self.f1(ch, level, pair)?);
            }
            return Value::from_items(out);
        }
        if !a.is_iterable() {
            return self.f1(ch, level, a);
        }
        if a.is_empty() {
            return Ok(a);
        }
        let n = a.len().unwrap_or(0);
        let mut out = try_vec(n)?;
        for i in 0..n {
            out.push(self.f1(ch, level, a.item(i))?);
        }
        if a.is_str() && out.iter().any(|q| matches!(q, Value::Char(_) | Value::Str(_))) {
            let mut s = alloc::string::String::new();
            for q in &out {
                match q {
                    Value::Char(c) => crate::value::try_push_char(&mut s, *c)?,
                    Value::Str(t) => crate::value::try_push_str(&mut s, t.as_str())?,
                    _ => return Err(Error::typ("': each over a string gave a mix of text and other values")),
                }
            }
            return Ok(Value::str(s));
        }
        Value::from_items(out)
    }

    fn over(&mut self, ch: &Chain, level: usize, a: Value) -> Result<Value> {
        if let Value::Ext(e) = &a {
            if let (2, Some(op)) = (ch.verb_arity, self.verb_op(ch, level)) {
                let op = alloc::string::String::from(op);
                if let Some(r) = e.0.host_reduce(self, &op, &a) {
                    return r;
                }
            }
            let n = Self::unhost(a.clone())?;
            if !matches!(n, Value::Ext(_)) {
                return self.over(ch, level, n);
            }
        }
        if a.is_atom() {
            return Ok(a);
        }
        let n = a.len().unwrap_or(0);
        if n == 1 {
            return Ok(a.item(0));
        }
        let op = self.verb_op(ch, level);
        if ch.verb_arity == 2 {
            match (&a, op) {
                (Value::Ints(x), Some("+")) if x.rank() == 1 => return Ok(Value::Int(x.data.iter().fold(0 as Int, |s, v| s.wrapping_add(*v)))),
                (Value::Ints(x), Some("*")) if x.rank() == 1 => return Ok(Value::Int(x.data.iter().fold(1 as Int, |s, v| s.wrapping_mul(*v)))),
                (Value::Ints(x), Some("&")) if x.rank() == 1 => return Ok(Value::Int(x.data.iter().copied().min().unwrap_or(0))),
                (Value::Ints(x), Some("|")) if x.rank() == 1 => return Ok(Value::Int(x.data.iter().copied().max().unwrap_or(0))),
                (Value::Flts(x), Some("+")) if x.rank() == 1 => return Ok(Value::Flt(x.data[0] + pairwise_sum(&x.data[1..]))),
                (Value::Ints(x), Some("%")) if x.rank() == 1 => {
                    // numpy.divide.reduce: no :undefined, IEEE results
                    return Ok(Value::Flt(x.data[1..].iter().fold(x.data[0] as Flt, |s, v| s / *v as Flt)));
                }
                (Value::Flts(x), Some("%")) if x.rank() == 1 => return Ok(Value::Flt(x.data[1..].iter().fold(x.data[0], |s, v| s / *v))),
                (Value::Str(s), Some("&")) => return Ok(Value::Char(s.as_str().chars().min().unwrap_or('\0'))),
                (Value::Str(s), Some("|")) => return Ok(Value::Char(s.as_str().chars().max().unwrap_or('\0'))),
                // ,/ over strings: one pass (the fold copies what it has joined at every step)
                (Value::List(l), Some(",")) if l.iter().all(|v| matches!(v, Value::Str(_))) => {
                    let total = l.iter().map(|v| if let Value::Str(t) = v { t.as_str().len() } else { 0 }).sum();
                    let mut out = crate::value::try_string(total)?;
                    for v in l.iter() {
                        if let Value::Str(t) = v {
                            out.push_str(t.as_str());
                        }
                    }
                    return Ok(Value::str(out));
                }
                _ => {}
            }
        }
        let mut acc = a.item(0);
        for i in 1..n {
            acc = self.f2(ch, level, acc, a.item(i))?;
        }
        Ok(acc)
    }

    fn scan(&mut self, ch: &Chain, level: usize, a: Value) -> Result<Value> {
        if a.is_atom() {
            return Ok(a);
        }
        let n = a.len().unwrap_or(0);
        if ch.verb_arity == 2 {
            if let (Value::Ints(x), Some("+")) = (&a, self.verb_op(ch, level)) {
                if x.rank() == 1 {
                    let mut out: Vec<Int> = try_vec(n)?;
                    let mut s: Int = 0;
                    for v in x.data.iter() {
                        s = s.wrapping_add(*v);
                        out.push(s);
                    }
                    return Ok(Value::ints(out));
                }
            }
        }
        let mut out = try_vec(n)?;
        let mut acc = a.item(0);
        out.push(acc.clone());
        for i in 1..n {
            acc = self.f2(ch, level, acc, a.item(i))?;
            out.push(acc.clone());
        }
        Value::from_items(out)
    }

    fn each_pair(&mut self, ch: &Chain, level: usize, a: Value) -> Result<Value> {
        if a.is_atom() || a.len() == Some(1) {
            return Ok(a);
        }
        let n = a.len().unwrap_or(0);
        let mut out = try_vec(n - 1)?;
        for i in 0..n - 1 {
            out.push(self.f2(ch, level, a.item(i), a.item(i + 1))?);
        }
        Value::from_items(out)
    }

    fn each_index(&mut self, ch: &Chain, level: usize, a: Value) -> Result<Value> {
        if a.is_empty() {
            return Ok(a);
        }
        if !a.is_iterable() {
            let pair = Value::from_items(alloc::vec![Value::Int(0), a])?;
            return self.f1(ch, level, pair);
        }
        let n = a.len().unwrap_or(0);
        let mut out = try_vec(n)?;
        for i in 0..n {
            let pair = Value::from_items(alloc::vec![Value::Int(i as Int), a.item(i)])?;
            out.push(self.f1(ch, level, pair)?);
        }
        Value::from_items(out)
    }
}
