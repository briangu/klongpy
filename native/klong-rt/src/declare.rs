//! The declarations of the static profile, `.extern` and `.entry`, as KlongPy's interpreter treats them
//! (`klongpy/sys_fn_static.py`, `klongpy/static/signature.py`, docs/static-profile.md): a program written for the
//! compiler runs as it stands.
//!
//! * `.extern(NAME;SIGNATURE[;MODULE])` declares a function the host provides and defines NAME. Called, NAME checks its
//!   arguments and its result against the signature and calls the host function of that name: a function the host registered
//!   before the declaration (`Interp::register`, which is what `klong_cfun_register` does). With none, a call is a `system`
//!   error. It returns the signature in its canonical form.
//! * `.entry(NAME;SIGNATURE)` checks a declaration of a function the program defines (its name, its arity against the
//!   signature) and returns the signature in its canonical form.
//! * `.fixed` is an error: the interpreter has no fixed-point numbers (see [`register`]).
//!
//! Where KlongPy and this differ: the host function is looked up when `.extern` runs and not at each call, so a function
//! registered later replaces the declared name as it is, unchecked; a type of the form `fixed<W,I>` in a signature is
//! accepted, and a call that would pass or return such a value is an error.
use alloc::rc::Rc;
use alloc::string::String;
use alloc::vec::Vec;
use core::cell::RefCell;

use klong_core::error::Result;
use klong_core::value::{try_push, try_str, try_vec, Arr, Func, NativeFn, Value, WILDCARD};
use klong_core::{Error, Interp};

// ----- types and signatures ---------------------------------------------------------------------------------------------------

const MAX_RANK: usize = 4;

#[derive(Clone, PartialEq)]
enum Elem {
    Int,
    Float,
    Str,
    Void,
    Fixed(u32, u32),
}

#[derive(Clone, PartialEq)]
struct Ty {
    elem: Elem,
    shape: Vec<u32>,
}

#[derive(Clone, PartialEq)]
struct Sig {
    params: Vec<Ty>,
    result: Ty,
}

impl core::fmt::Display for Ty {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match &self.elem {
            Elem::Int => f.write_str("int")?,
            Elem::Float => f.write_str("float")?,
            Elem::Str => f.write_str("str")?,
            Elem::Void => f.write_str("void")?,
            Elem::Fixed(w, i) => write!(f, "fixed<{},{}>", w, i)?,
        }
        for (k, d) in self.shape.iter().enumerate() {
            f.write_str(if k == 0 { "[" } else { "," })?;
            write!(f, "{}", d)?;
        }
        if !self.shape.is_empty() {
            f.write_str("]")?;
        }
        Ok(())
    }
}

impl core::fmt::Display for Sig {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        for (k, p) in self.params.iter().enumerate() {
            if k > 0 {
                f.write_str(" ")?;
            }
            write!(f, "{}", p)?;
        }
        if !self.params.is_empty() {
            f.write_str(" ")?;
        }
        write!(f, "-> {}", self.result)
    }
}

/// A malformed signature: what is wrong, and where (a byte offset into the text).
struct SigError {
    message: String,
    at: usize,
}

fn sig_err(message: String, at: usize) -> SigError {
    SigError { message, at }
}

fn lossy(b: &[u8]) -> String {
    String::from_utf8_lossy(b).into_owned()
}

fn space(b: u8) -> bool {
    b.is_ascii_whitespace()
}

/// The end of the run of non-space bytes that starts at `pos`.
fn word_end(t: &[u8], pos: usize) -> usize {
    let mut e = pos;
    while e < t.len() && !space(t[e]) {
        e += 1;
    }
    e
}

/// Read one type from `t[pos..end]`; the type and the position after it.
fn type_at(t: &[u8], pos: usize, end: usize) -> core::result::Result<(Ty, usize), SigError> {
    let t = &t[..end];
    let mut p = pos;
    while p < end && t[p].is_ascii_alphabetic() {
        p += 1;
    }
    if p == pos {
        let w = word_end(t, pos);
        let stop = if w > pos { w } else { (pos + 1).min(end) };
        return Err(sig_err(klong_core::msg!("expected a type but found '{}'", lossy(&t[pos..stop])), pos));
    }
    let word = &t[pos..p];
    let word_at = pos;
    // <...>: no space and no angle bracket inside
    let mut angle = None;
    if p < end && t[p] == b'<' {
        let mut q = p + 1;
        while q < end && t[q] != b'<' && t[q] != b'>' && !space(t[q]) {
            q += 1;
        }
        if q < end && t[q] == b'>' {
            angle = Some((p, q + 1));
            p = q + 1;
        }
    }
    // [...]: up to the first ]
    let mut dims = None;
    if p < end && t[p] == b'[' {
        if let Some(q) = t[p..].iter().position(|&b| b == b']') {
            dims = Some((p, p + q + 1));
            p += q + 1;
        }
    }
    let elem = match (word, angle) {
        (b"fixed", None) => return Err(sig_err(String::from("fixed needs its widths: fixed<W,I> (W total bits, I integer bits)"), word_at)),
        (b"fixed", Some((a, b))) => {
            let inner = &t[a + 1..b - 1];
            let nums: Vec<&[u8]> = inner.split(|&c| c == b',').collect();
            let number = |s: &[u8]| -> Option<u32> {
                if s.is_empty() || !s.iter().all(|c| c.is_ascii_digit()) {
                    return None;
                }
                Some(core::str::from_utf8(s).ok()?.parse::<u32>().unwrap_or(u32::MAX))
            };
            let (w, i) = match (nums.len() == 2).then(|| (number(nums[0]), number(nums[1]))) {
                Some((Some(w), Some(i))) => (w, i),
                _ => {
                    return Err(sig_err(
                        klong_core::msg!("a fixed-point type is written fixed<W,I> with two positive integers, found 'fixed{}'", lossy(&t[a..b])),
                        a,
                    ))
                }
            };
            if !(2..=64).contains(&w) {
                return Err(sig_err(klong_core::msg!("fixed<{},{}>: the total width W must be 2 to 64", w, i), a));
            }
            if !(1..=w).contains(&i) {
                return Err(sig_err(klong_core::msg!("fixed<{},{}>: the integer bits I must be 1 to W ({})", w, i, w), a));
            }
            Elem::Fixed(w, i)
        }
        (w, a) => {
            if a.is_some() {
                return Err(sig_err(klong_core::msg!("{} takes no <...> (only fixed<W,I> does)", lossy(w)), a.map_or(0, |x| x.0)));
            }
            match w {
                b"int" => Elem::Int,
                b"float" => Elem::Float,
                b"str" => Elem::Str,
                b"void" => Elem::Void,
                _ => return Err(sig_err(klong_core::msg!("unknown type '{}' (the types are int, float, fixed<W,I>, str and void)", lossy(w)), word_at)),
            }
        }
    };
    let mut shape = Vec::new();
    if let Some((a, b)) = dims {
        if matches!(elem, Elem::Str | Elem::Void) {
            return Err(sig_err(klong_core::msg!("{} cannot have dimensions", lossy(word)), a));
        }
        let inner = &t[a + 1..b - 1];
        let digits = |d: &[u8]| -> Vec<u8> { d.iter().copied().filter(|&c| !space(c)).collect() };
        // each piece is digits with blanks around them, and no blank inside
        let ok = inner.split(|&c| c == b',').all(|d| {
            let trimmed = d.trim_ascii();
            !trimmed.is_empty() && trimmed.iter().all(|c| c.is_ascii_digit())
        });
        if !ok {
            return Err(sig_err(klong_core::msg!("dimensions must be positive integers separated by commas, found '[{}]'", lossy(inner)), a));
        }
        for d in inner.split(|&c| c == b',') {
            let n = core::str::from_utf8(&digits(d)).ok().and_then(|s| s.parse::<u32>().ok()).unwrap_or(0);
            if n == 0 {
                return Err(sig_err(String::from("every dimension must be a positive integer"), a));
            }
            try_push(&mut shape, n).map_err(|_| sig_err(String::from("out of memory"), 0))?;
        }
        if shape.len() > MAX_RANK {
            return Err(sig_err(klong_core::msg!("an array has at most {} dimensions, found {}", MAX_RANK, shape.len()), a));
        }
    }
    if p < end && !space(t[p]) {
        let w = word_end(t, pos);
        return Err(sig_err(klong_core::msg!("'{}' is not a type", lossy(&t[pos..w])), pos));
    }
    Ok((Ty { elem, shape }, p))
}

/// `params -> result`; `allow_str` is false for `.entry`, where `str` is not a parameter type.
fn parse_signature(text: &str, allow_str: bool) -> core::result::Result<Sig, SigError> {
    let t = text.as_bytes();
    let Some(arrow) = text.find("->") else {
        return Err(sig_err(String::from("a signature needs '->' between the parameter types and the result type"), 0));
    };
    if let Some(i) = text[arrow + 2..].find("->") {
        return Err(sig_err(String::from("a signature has exactly one '->'"), arrow + 2 + i));
    }
    let mut params = Vec::new();
    let mut pos = 0;
    loop {
        while pos < arrow && space(t[pos]) {
            pos += 1;
        }
        if pos >= arrow {
            break;
        }
        let start = pos;
        let (ty, next) = type_at(t, pos, arrow)?;
        pos = next;
        if ty.elem == Elem::Void {
            return Err(sig_err(String::from("void is a result type only; a function of no parameters is written \"-> int\""), start));
        }
        if ty.elem == Elem::Str && !allow_str {
            return Err(sig_err(String::from("str is a parameter type of externs only"), start));
        }
        try_push(&mut params, ty).map_err(|_| sig_err(String::from("out of memory"), 0))?;
    }
    let end = text.trim_end().len();
    pos = arrow + 2;
    while pos < end && space(t[pos]) {
        pos += 1;
    }
    if pos >= end {
        return Err(sig_err(String::from("a signature needs a result type after '->' (void for none)"), arrow));
    }
    let start = pos;
    let (result, next) = type_at(t, pos, end)?;
    if result.elem == Elem::Str {
        return Err(sig_err(String::from("str cannot be a result type"), start));
    }
    if next != end {
        let mut rest = next;
        while rest < end && space(t[rest]) {
            rest += 1;
        }
        return Err(sig_err(klong_core::msg!("unexpected '{}' after the result type", lossy(&t[rest..end])), rest));
    }
    Ok(Sig { params, result })
}

fn signature(which: &str, text: &str, allow_str: bool) -> Result<Sig> {
    parse_signature(text, allow_str).map_err(|e| {
        let at = if text.is_empty() { String::new() } else { klong_core::msg!(" (at column {} of \"{}\")", e.at + 1, text) };
        Error::typ(klong_core::msg!("{} signature: {}{}", which, e.message, at))
            .with_hint("a signature is \"params -> result\", for example \"int float[64] -> float\"")
    })
}

// ----- names ------------------------------------------------------------------------------------------------------------------

const C_KEYWORDS: &[&str] = &[
    "auto", "break", "case", "char", "const", "continue", "default", "do", "double", "else", "enum", "extern", "float", "for", "goto", "if", "inline", "int",
    "long", "register", "restrict", "return", "short", "signed", "sizeof", "static", "struct", "switch", "typedef", "union", "unsigned", "void", "volatile",
    "while", "_Bool", "_Complex", "_Imaginary", "main", "memcpy", "memset", "NULL",
];

fn check_name(name: &str, what: &str) -> Option<String> {
    let b = name.as_bytes();
    let ok = b.first().is_some_and(|c| c.is_ascii_alphabetic()) && b.iter().all(|c| c.is_ascii_alphanumeric() || *c == b'.');
    if !ok {
        return Some(klong_core::msg!("{} name must be a Klong name of ASCII letters, digits and '.', starting with a letter (found '{}')", what, name));
    }
    if matches!(name, "x" | "y" | "z") {
        return Some(klong_core::msg!("{} is a function argument and cannot be {} name", name, what));
    }
    None
}

fn check_extern_name(name: &str) -> Option<String> {
    if let Some(bad) = check_name(name, "an extern") {
        return Some(bad);
    }
    let c: String = name.chars().map(|c| if c == '.' { '_' } else { c }).collect();
    if C_KEYWORDS.contains(&c.as_str()) {
        return Some(klong_core::msg!("the C name of extern {} is '{}', which is reserved in C", name, c));
    }
    if c.starts_with("kg_") {
        return Some(klong_core::msg!("the C name of extern {} is '{}': names that start with kg_ belong to the runtime", name, c));
    }
    None
}

fn check_entry_name(name: &str) -> Option<String> {
    if let Some(bad) = check_name(name, "an entry") {
        return Some(bad);
    }
    if matches!(name, "init" | "t" | "storage.t" | "size") {
        let c: String = name.chars().map(|c| if c == '.' { '_' } else { c }).collect();
        return Some(klong_core::msg!("an entry cannot be called {}: prog_{} is a name the compiler generates", name, c));
    }
    None
}

// ----- values against types ---------------------------------------------------------------------------------------------------

enum Mismatch {
    Type(String),
    Shape(String),
}

fn dims_of<T: Copy>(a: &Arr<T>) -> Vec<u32> {
    let mut d = Vec::new();
    if d.try_reserve(a.rank()).is_ok() {
        d.push(a.len() as u32);
        d.extend_from_slice(a.row_dims());
    }
    d
}

fn describe(v: &Value) -> String {
    let arr = |elem: &str, d: Vec<u32>| {
        let mut s = String::from(elem);
        for (k, n) in d.iter().enumerate() {
            s.push(if k == 0 { '[' } else { ',' });
            s.push_str(&klong_core::msg!("{}", n));
        }
        s.push(']');
        s
    };
    match v {
        Value::Int(_) => String::from("an integer"),
        Value::Flt(_) => String::from("a float"),
        Value::Str(_) => String::from("a string"),
        Value::Sym(_) | Value::Char(_) => String::from("a symbol or a character"),
        Value::Ints(a) => arr("int", dims_of(a)),
        Value::Flts(a) => arr("float", dims_of(a)),
        Value::Undefined => String::from("nothing"),
        other => klong_core::msg!("a {}", other.describe()),
    }
}

/// The value as the host function takes it, checked against `ty`: an int for `int`, an int or a float (as a float) for `float`, a
/// string for `str`, an array of the dimensions of the type (an int array is accepted for a float one and converted).
fn conform(ty: &Ty, v: &Value, what: &str) -> core::result::Result<Value, Mismatch> {
    let wrong = |v: &Value| Mismatch::Type(klong_core::msg!("{} must be {}, got {}", what, ty, describe(v)));
    match (&ty.elem, ty.shape.is_empty()) {
        (Elem::Void, _) => Ok(Value::Undefined),
        (Elem::Str, _) => match v {
            Value::Str(_) => Ok(v.clone()),
            _ => Err(wrong(v)),
        },
        (Elem::Fixed(..), _) => Err(Mismatch::Type(klong_core::msg!(
            "{}: a fixed-point value is not supported by the native interpreter",
            what
        ))),
        (Elem::Int, true) => match v {
            Value::Int(_) => Ok(v.clone()),
            _ => Err(wrong(v)),
        },
        (Elem::Float, true) => match v {
            Value::Flt(_) => Ok(v.clone()),
            Value::Int(i) => Ok(Value::Flt(*i as klong_core::Flt)),
            _ => Err(wrong(v)),
        },
        (Elem::Int, false) => match v {
            Value::Ints(a) if dims_of(a) == ty.shape => Ok(v.clone()),
            Value::Ints(_) => Err(Mismatch::Shape(klong_core::msg!("{} must be {}, got {}", what, ty, describe(v)))),
            _ => Err(wrong(v)),
        },
        (Elem::Float, false) => match v {
            Value::Flts(a) if dims_of(a) == ty.shape => Ok(v.clone()),
            Value::Ints(a) if dims_of(a) == ty.shape => {
                let mut data = try_vec(a.data.len()).map_err(|_| Mismatch::Type(String::from("out of memory")))?;
                data.extend(a.data.iter().map(|&i| i as klong_core::Flt));
                Ok(Value::Flts(Rc::new(Arr::shaped(data, &ty.shape))))
            }
            Value::Ints(_) | Value::Flts(_) => Err(Mismatch::Shape(klong_core::msg!("{} must be {}, got {}", what, ty, describe(v)))),
            _ => Err(wrong(v)),
        },
    }
}

fn mismatch(m: Mismatch, sig: &Sig) -> Error {
    match m {
        Mismatch::Type(s) => Error::typ(klong_core::msg!("{} (declared \"{}\")", s, sig)),
        Mismatch::Shape(s) => Error::shape(klong_core::msg!("{} (declared \"{}\")", s, sig)),
    }
}

fn plural(n: usize, word: &str) -> String {
    if n == 1 {
        klong_core::msg!("{} {}", n, word)
    } else {
        klong_core::msg!("{} {}s", n, word)
    }
}

// ----- the functions ----------------------------------------------------------------------------------------------------------

/// What `.extern` has declared: name, signature, module and, for a function of more than three parameters, what the named-parameter
/// function that was defined for it calls (a native function takes at most x, y and z).
type Declared = Rc<RefCell<Vec<Extern>>>;

struct Extern {
    name: String,
    sig: Sig,
    module: String,
    call: Option<NativeFn>,
}

/// The arguments `.extern.arg` has collected for the next `.extern.call`.
type Stash = Rc<RefCell<Vec<Value>>>;

/// The value a declared name is bound to: a check of the arguments, a call of the host function, a check of the result.
fn bound(name: &str, sig: Sig, host: Option<NativeFn>) -> Result<NativeFn> {
    let name = try_str(name)?;
    Ok(Rc::new(move |ip: &mut Interp, a: &[Value]| {
        if a.len() != sig.params.len() {
            return Err(Error::arity(klong_core::msg!(
                "{} is declared \"{}\" and takes {}, got {}",
                name,
                sig,
                plural(sig.params.len(), "argument"),
                a.len()
            )));
        }
        let Some(host) = &host else {
            return Err(Error::system(klong_core::msg!("extern {} has no stand-in on the host", name))
                .with_hint("the host registers a function of that name before the program declares it (klong_cfun_register in C)"));
        };
        let mut args = try_vec(a.len())?;
        for (i, (ty, v)) in sig.params.iter().zip(a).enumerate() {
            try_push(&mut args, conform(ty, v, &klong_core::msg!("argument {} of {}", i + 1, name)).map_err(|m| mismatch(m, &sig))?)?;
        }
        let r = host(ip, &args)?;
        conform(&sig.result, &r, &klong_core::msg!("the result of {}", name)).map_err(|m| mismatch(m, &sig))
    }))
}

fn string<'a>(which: &str, v: &'a Value, what: &str) -> Result<&'a str> {
    match v {
        Value::Str(s) => Ok(s.as_str()),
        _ => Err(Error::typ(klong_core::msg!("{}: the {} must be a string", which, what))),
    }
}

fn extern_fn(declared: &Declared, ip: &mut Interp, a: &[Value]) -> Result<Value> {
    if !(2..=3).contains(&a.len()) {
        return Err(Error::arity(klong_core::msg!(".extern takes 2 or 3 arguments, got {}", a.len())));
    }
    let name = string(".extern", &a[0], "name")?;
    let text = string(".extern", &a[1], "signature")?;
    let module = match a.get(2) {
        Some(m) => string(".extern", m, "module")?,
        None => "env",
    };
    let sig = signature(".extern", text, true)?;
    if let Some(bad) = check_extern_name(name) {
        return Err(Error::typ(klong_core::msg!(".extern: {}", bad)));
    }
    let canonical = klong_core::msg!("{}", sig);
    if let Some(old) = declared.borrow().iter().find(|e| e.name == name) {
        if old.sig != sig || old.module != module {
            return Err(Error::typ(klong_core::msg!(".extern: {} is already declared \"{}\" (module {})", name, old.sig, old.module)));
        }
        return Value::text(&canonical);
    }
    // a function the host registered under that name is the one that is called; anything else defined is a conflict
    let host = match ip.global(name) {
        None => None,
        Some(Value::Fn(f)) => match &*f {
            Func::Native { f: native, .. } => Some(native.clone()),
            _ => return Err(conflict(name)),
        },
        Some(_) => return Err(conflict(name)),
    };
    let n = sig.params.len();
    let call = bound(name, sig.clone(), host)?;
    try_push(&mut declared.borrow_mut(), Extern { name: try_str(name)?, sig, module: try_str(module)?, call: (n > 3).then(|| call.clone()) })?;
    if n <= 3 {
        // x, y and z: a wildcard function is given whichever of them the call supplied
        let s = ip.syms.intern(name);
        if let Some(e) = ip.syms.failure() {
            return Err(e);
        }
        ip.define_global(name, Value::Fn(Rc::new(Func::Native { name: s, arity: WILDCARD, f: call })))?;
    } else if let Err(e) = define_wide(ip, declared, name, n) {
        declared.borrow_mut().pop();
        return Err(e);
    }
    Value::text(&canonical)
}

fn conflict(name: &str) -> Error {
    Error::name(klong_core::msg!(".extern: {} is already defined in the program; an extern cannot replace it", name))
}

/// A function of more than three parameters is one with named parameters whose body hands its parameters to the declaration, three
/// at a time (`.extern.arg`) and the last ones with the name (`.extern.call`).
fn define_wide(ip: &mut Interp, declared: &Declared, name: &str, n: usize) -> Result<()> {
    register_wide(ip, declared);
    let mut text = String::new();
    let mut put = |t: &str| -> Result<()> { klong_core::value::try_push_str(&mut text, t) };
    put(name)?;
    put("(")?;
    for i in 1..=n {
        put(if i > 1 { ";arg" } else { "arg" })?;
        put(&klong_core::msg!("{}", i))?;
    }
    put(")::{")?;
    let mut i = 1;
    // (a native function is given at most three arguments: the name and two more go with the call)
    while n + 1 - i > 2 {
        put(&klong_core::msg!(".extern.arg(arg{};arg{};arg{});", i, i + 1, i + 2))?;
        i += 3;
    }
    put(&klong_core::msg!(".extern.call(\"{}\"", name))?;
    while i <= n {
        put(&klong_core::msg!(";arg{}", i))?;
        i += 1;
    }
    put(")}")?;
    ip.run_global(&text, "<extern>")?;
    Ok(())
}

fn entry_fn(ip: &mut Interp, a: &[Value]) -> Result<Value> {
    if a.len() != 2 {
        return Err(Error::arity(klong_core::msg!(".entry takes 2 arguments, got {}", a.len())));
    }
    let name = string(".entry", &a[0], "name")?;
    let text = string(".entry", &a[1], "signature")?;
    let sig = signature(".entry", text, false)?;
    if let Some(bad) = check_entry_name(name) {
        return Err(Error::typ(klong_core::msg!(".entry: {}", bad)));
    }
    let Some(f) = ip.global(name) else {
        return Err(Error::name(klong_core::msg!(".entry: the function {} is not defined", name)).with_hint("define the function before the .entry line"));
    };
    // a function the program defines: a literal, with or without named parameters (not a projection, an operator or a system function)
    let arity = match &f {
        Value::Fn(func) => match &**func {
            Func::Lazy(l) => Some(l.arity as usize),
            Func::Code { tree, node } => match tree.node(*node) {
                klong_core::tree::Node::Fn { has_args: false, call: false, arity, .. } => Some(*arity as usize),
                _ => None,
            },
            _ => None,
        },
        _ => None,
    };
    let Some(arity) = arity else {
        return Err(Error::typ(klong_core::msg!(".entry: {} is not a function defined in the program", name))
            .with_hint("an entry is a function written name::{...} or name(a;b)::{...}"));
    };
    if arity != sig.params.len() {
        return Err(Error::arity(klong_core::msg!(
            ".entry: {} takes {} but the signature \"{}\" has {}",
            name,
            plural(arity, "argument"),
            sig,
            plural(sig.params.len(), "parameter")
        )));
    }
    Value::text(&klong_core::msg!("{}", sig))
}

/// `.extern.arg` and `.extern.call`, which the body of a function of more than three parameters calls: defined with the first such
/// declaration, so that a program with none (and a board that has no use for them) pays for no names, functions or tables at start.
fn register_wide(ip: &mut Interp, declared: &Declared) {
    if ip.global(".extern.call").is_some() {
        return;
    }
    let stash: Stash = Rc::new(RefCell::new(Vec::new()));
    let st = stash.clone();
    let declared = declared.clone();
    ip.register(
        ".extern.arg",
        WILDCARD,
        Rc::new(move |_: &mut Interp, a: &[Value]| {
            let mut s = st.try_borrow_mut().map_err(|_| Error::system("extern arguments are in use"))?;
            for v in a {
                try_push(&mut s, v.clone())?;
            }
            Ok(Value::Undefined)
        }),
    );
    ip.register(
        ".extern.call",
        WILDCARD,
        Rc::new(move |ip: &mut Interp, a: &[Value]| {
            let name = string(".extern.call", a.first().ok_or_else(|| Error::arity(".extern.call takes a name"))?, "name")?;
            let call = declared.borrow().iter().find(|e| e.name == name).and_then(|e| e.call.clone());
            let call = call.ok_or_else(|| Error::name(klong_core::msg!("{} is not a declared extern", name)))?;
            let mut args = core::mem::take(&mut *stash.try_borrow_mut().map_err(|_| Error::system("extern arguments are in use"))?);
            for v in &a[1..] {
                try_push(&mut args, v.clone())?;
            }
            call(ip, &args)
        }),
    );
}

/// Define `.extern`, `.entry` and `.fixed`.
///
/// `.fixed` is an error, not an emulation: KlongPy's interpreter gives `.fixed(type;x)` a fixed-point array that knows its type,
/// and arithmetic on it follows the reference semantics of `klongpy/static/fixed.py` operation by operation (an add, a multiply
/// and a divide that wrap to the width, floor on conversion, a reduction that wraps at each step, a refusal to mix with floats).
/// The native values have no such kind and every operator would need a case for it, so a fixed-point program is compiled or run
/// by KlongPy.
pub fn register(ip: &mut Interp) {
    let declared: Declared = Rc::new(RefCell::new(Vec::new()));
    ip.register(".extern", WILDCARD, Rc::new(move |ip: &mut Interp, a: &[Value]| extern_fn(&declared, ip, a)));
    ip.register(".entry", WILDCARD, Rc::new(|ip: &mut Interp, a: &[Value]| entry_fn(ip, a)));
    ip.register(
        ".fixed",
        WILDCARD,
        Rc::new(|_: &mut Interp, _: &[Value]| {
            Err::<Value, Error>(
                Error::system("fixed-point programs are not supported by the native interpreter")
                    .with_hint("run the program with KlongPy (klongpy.static.call_entry) or compile it with kgpy compile"),
            )
        }),
    );
}
