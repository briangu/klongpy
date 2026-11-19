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
    #[cfg(feature = "host")]
    for v in [p, q] {
        if let Value::Ext(e) = v {
            if let Some(b) = e.0.host_converged(p, q) {
                return b;
            }
            break;
        }
    }
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

/// The running sum of integers (wrapping), appended to `out`. Its own function, so that the loop is compiled the same whatever
/// surrounds the call.
#[inline(never)]
fn running_sum(x: &[Int], out: &mut Vec<Int>) {
    let mut s: Int = 0;
    out.extend(x.iter().map(|v| {
        s = s.wrapping_add(*v);
        s
    }));
}

impl Interp {
    /// The condition function `a` of a `\\~` or `:~` applied to `b`: its body is found once (`cond`, from `fast_body`).
    #[inline]
    fn apply_cond(&mut self, a: &Value, cond: Option<&Value>, b: Value) -> Result<Value> {
        #[cfg(feature = "speed")]
        if let Some(g) = cond {
            return self.run_body(g.clone(), b, None);
        }
        #[cfg(not(feature = "speed"))]
        let _ = cond;
        self.apply_fast(a, b, None)
    }

    /// `a+'b` and the like: when the verb of the adverb is `+ - * & | < > =` itself and both operands are flat vectors of numbers of one
    /// length, what the kernel of the operator gives for the two vectors, which is the same as the operator on each pair.
    #[cfg(feature = "speed")]
    fn prim_pairs(&mut self, ch: &Chain, level: usize, a: &Value, b: &Value) -> Result<Option<Value>> {
        use crate::ops::Atomic::*;
        if level != 1 || ch.verb_arity != 2 {
            return Ok(None);
        }
        let Node::OpVal { op, .. } = ch.tree.node(ch.verb) else { return Ok(None) };
        let Some(at) = crate::ops::by_id(*op).and_then(|d| d.atomic) else { return Ok(None) };
        if !matches!(at, Add | Sub | Mul | Min | Max | Less | More | Eq) {
            return Ok(None);
        }
        let flat = |v: &Value| match v {
            Value::Ints(x) => x.rank() == 1,
            Value::Flts(x) => x.rank() == 1,
            _ => false,
        };
        if !flat(a) || !flat(b) || a.len() != b.len() {
            return Ok(None);
        }
        crate::fast::array(at, a, b)
    }

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

    /// HOST: a host object handed to an adverb the embedder does not take becomes a native value (a copy; the embedder refuses when
    /// the copy would be wrong).
    #[cfg(feature = "host")]
    fn unhost(v: Value, what: &str) -> Result<Value> {
        if let Value::Ext(e) = &v {
            if let Some(r) = e.0.host_native(what) {
                return r;
            }
        }
        Ok(v)
    }
    #[cfg(not(feature = "host"))]
    #[inline(always)]
    fn unhost(v: Value, _what: &str) -> Result<Value> {
        Ok(v)
    }

    /// HOST: the adverb at `level` of the chain, applied to an operand that is a host object, is the embedder's to do: it is given the
    /// function the adverb modifies (`level - 1` of the chain, as a function value) and the operator text when that is a primitive.
    #[cfg(feature = "host")]
    fn host_adverb(&mut self, ch: &Chain, level: usize, glyph: &str, a: &Value, b: Option<&Value>) -> Option<Result<Value>> {
        let e = match (a, b) {
            (Value::Ext(e), _) => e.clone(),
            (_, Some(Value::Ext(e))) => e.clone(),
            _ => return None,
        };
        let verb_op = self.verb_op(ch, 1).map(alloc::string::String::from);
        let (tree, verb, verb_arity, verb_val, advs, first_arity) = (ch.tree.clone(), ch.verb, ch.verb_arity, ch.verb_val.clone(), ch.advs.to_vec(), ch.first_arity);
        let lv = level - 1;
        let name = self.syms.intern("adverb");
        let arity = if lv == 0 { verb_arity } else if lv == 1 { first_arity } else { 1 };
        let f = Value::Fn(alloc::rc::Rc::new(crate::value::Func::Native {
            name,
            arity,
            f: alloc::rc::Rc::new(move |ip: &mut Interp, args: &[Value]| -> Result<Value> {
                let ch = Chain { tree: &tree, verb, verb_arity, verb_val: verb_val.clone(), #[cfg(feature = "speed")] body: None, advs: &advs, first_arity };
                match args.len() {
                    1 => ip.chain_call(&ch, lv, args[0].clone(), None),
                    2 => ip.chain_call(&ch, lv, args[0].clone(), Some(args[1].clone())),
                    _ => Err(Error::typ("the function of an adverb was given the wrong number of operands")),
                }
            }),
        }));
        e.0.host_adverb(self, glyph, &f, verb_op.as_deref(), a, b)
    }

    pub(crate) fn adverb1(&mut self, ch: &Chain, level: usize, glyph: SymId, a: Value) -> Result<Value> {
        #[cfg(feature = "host")]
        if matches!(a, Value::Ext(_)) {
            let g = alloc::string::String::from(self.syms.name(glyph));
            if let Some(r) = self.host_adverb(ch, level, &g, &a, None) {
                return r;
            }
        }
        let g = self.syms.name(glyph);
        let a = Self::unhost(a, g)?;
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
        #[cfg(feature = "host")]
        if matches!(a, Value::Ext(_)) || matches!(b, Value::Ext(_)) {
            let g = alloc::string::String::from(self.syms.name(glyph));
            if let Some(r) = self.host_adverb(ch, level, &g, &a, Some(&b)) {
                return r;
            }
        }
        let g = self.syms.name(glyph);
        let (a, b) = (Self::unhost(a, g)?, Self::unhost(b, g)?);
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
                // an operator between two flat vectors of numbers is the operator on the vectors
                #[cfg(feature = "speed")]
                if let Some(v) = self.prim_pairs(ch, level, &a, &b)? {
                    return Ok(v);
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
                let cond = self.fast_body(&a, 1);
                while self.apply_cond(&a, cond.as_ref(), b.clone())?.truthy() {
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
                let cond = self.fast_body(&a, 1);
                while self.apply_cond(&a, cond.as_ref(), b.clone())?.truthy() {
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
                #[cfg(feature = "speed")]
                (Value::Flts(x), Some("&")) if x.rank() == 1 => return Ok(Value::Flt(crate::fast::flt_extreme(&x.data, false))),
                #[cfg(feature = "speed")]
                (Value::Flts(x), Some("|")) if x.rank() == 1 => return Ok(Value::Flt(crate::fast::flt_extreme(&x.data, true))),
                #[cfg(feature = "speed")]
                (Value::Flts(x), Some("*")) if x.rank() == 1 => return Ok(Value::Flt(x.data[1..].iter().fold(x.data[0], |s, v| s * *v))),
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
            acc = match self.f2(ch, level, acc, a.item(i)) {
                Ok(v) => v,
                #[cfg(feature = "diag")]
                Err(e) => return Err(self.chain_error(ch, &[&a], e)),
                #[cfg(not(feature = "diag"))]
                Err(e) => return Err(e),
            };
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
                    running_sum(&x.data, &mut out);
                    return Ok(Value::ints(out));
                }
            }
        }
        // the running product, minimum and maximum of integers (the product wraps as the pairs do)
        #[cfg(feature = "speed")]
        if let (Value::Ints(x), Some(g @ ("*" | "&" | "|")), 2) = (&a, self.verb_op(ch, level), ch.verb_arity) {
            if x.rank() == 1 && !x.data.is_empty() {
                let mut out: Vec<Int> = try_vec(n)?;
                let mut s = x.data[0];
                out.push(s);
                for &v in &x.data[1..] {
                    s = match g {
                        "*" => s.wrapping_mul(v),
                        "&" => s.min(v),
                        _ => s.max(v),
                    };
                    out.push(s);
                }
                return Ok(Value::ints(out));
            }
        }
        #[cfg(feature = "speed")]
        if let (Value::Flts(x), Some("+"), 2) = (&a, self.verb_op(ch, level), ch.verb_arity) {
            if x.rank() == 1 {
                let mut out: Vec<Flt> = try_vec(n)?;
                let mut s = x.data[0];
                out.push(s);
                for v in &x.data[1..] {
                    s += *v;
                    out.push(s);
                }
                return Ok(Value::flts(out));
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
