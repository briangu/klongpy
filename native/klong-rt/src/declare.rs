//! The declarations of the static profile, `.extern` and `.entry`, as KlongPy's interpreter treats them
//! (`klongpy/sys_fn_static.py`, `klongpy/static/signature.py`, docs/static-profile.md): a program written for the
//! compiler runs as it stands.
//!
//! * `.extern(C_SYMBOL;SIGNATURE)` declares a function the host provides and returns it as a value: `adc::.extern("read_adc";"int -> float")`.
//!   Called, it checks its arguments and its result against the signature and calls the host function registered under the C symbol
//!   before the declaration (`Declarations::provide`, which is what `klong_extern_register` does) or the stand-in the host's resolver gives.
//!   A host function is kept in the declarations, not among the globals, so the only way to it is through the checks.
//!   With none, a call is a `system` error. The same symbol declared again with the same signature is the same declaration
//!   (the same function); with another signature it is an error. A statement that discards the value is an error (`module::discarded_extern`).
//! * `.entry(NAME;SIGNATURE)` checks a declaration of a function the program defines (its name, its arity against the
//!   signature) and returns the signature in its canonical form.
//! * `.fixed` is an error: the interpreter has no fixed-point numbers (see [`register`]).
//!
//! Where KlongPy and this differ: a type of the form `fixed<W,I>` in a signature is
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

/// The element type of a declared type.
#[derive(Clone, PartialEq)]
pub enum Elem {
    Int,
    Float,
    Str,
    Void,
    Fixed(u32, u32),
}

/// A declared type: an element type and the dimensions (none for a scalar). `Display` writes the canonical form.
#[derive(Clone, PartialEq)]
pub struct Ty {
    pub elem: Elem,
    pub shape: Vec<u32>,
}

/// A declared signature, `params -> result`; `Display` writes the canonical form.
#[derive(Clone, PartialEq)]
pub struct Sig {
    pub params: Vec<Ty>,
    pub result: Ty,
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

/// The reason `sym` cannot be the C symbol of an extern: a C identifier (letters, digits and underscore, not starting with a digit), not a
/// C keyword, not one of the runtime's (`kg_`).
pub fn check_symbol(sym: &str) -> Option<String> {
    let b = sym.as_bytes();
    if sym.contains('.') {
        return Some(klong_core::msg!("'{}' is not a C symbol: a C identifier has no dot (the name a program uses is the entry's key in its module)", sym));
    }
    if !b.first().is_some_and(|c| c.is_ascii_alphabetic() || *c == b'_') || !b.iter().all(|c| c.is_ascii_alphanumeric() || *c == b'_') {
        return Some(klong_core::msg!("'{}' is not a C identifier: letters, digits and underscore, not starting with a digit", sym));
    }
    if C_KEYWORDS.contains(&sym) {
        return Some(klong_core::msg!("'{}' is reserved in C", sym));
    }
    if sym.starts_with("kg_") {
        return Some(klong_core::msg!("'{}': symbols that start with kg_ belong to the runtime", sym));
    }
    None
}

/// A `.param` name: a Klong name whose C spelling (dots as underscores) is not reserved.
fn check_param_name(name: &str) -> Option<String> {
    if let Some(bad) = check_name(name, "a parameter") {
        return Some(bad);
    }
    let c: String = name.chars().map(|c| if c == '.' { '_' } else { c }).collect();
    if C_KEYWORDS.contains(&c.as_str()) {
        return Some(klong_core::msg!("the C name of parameter {} is '{}', which is reserved in C", name, c));
    }
    if c.starts_with("kg_") {
        return Some(klong_core::msg!("the C name of parameter {} is '{}': names that start with kg_ belong to the runtime", name, c));
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

fn conform_extern(ty: &Ty, v: &Value, what: &str, sig: &Sig) -> Result<Value> {
    // The static signature uses native numeric values, even when the embedding backend owns an array.
    let native;
    let v = if matches!(ty.elem, Elem::Int | Elem::Float) && v.host_shape().is_some() {
        native = v.unhost(what)?;
        &native
    } else {
        v
    };
    conform(ty, v, what).map_err(|m| mismatch(m, sig))
}

fn plural(n: usize, word: &str) -> String {
    if n == 1 {
        klong_core::msg!("{} {}", n, word)
    } else {
        klong_core::msg!("{} {}s", n, word)
    }
}

// ----- the functions ----------------------------------------------------------------------------------------------------------

/// What `.extern` has declared (by C symbol), what `.entry` has declared and, with `params`, the parameters.
type Declared = Rc<RefCell<Tables>>;

/// The host's stand-in for a declared symbol, looked up at each call: `(interpreter, symbol, arguments)`, `None` when the host has none
/// (then the call is the `system` error of an extern with no stand-in).
#[cfg(feature = "diag")]
pub type Resolver = Rc<dyn Fn(&mut Interp, &str, &[Value]) -> Option<Result<Value>>>;

#[derive(Default)]
struct Tables {
    #[cfg(feature = "diag")]
    resolver: Option<Resolver>,
    externs: Vec<Rc<Extern>>,
    /// The host functions by C symbol (a slot each, shared with the extern declared under that symbol): provided before or after the declaration.
    provided: Vec<(String, HostSlot)>,
    entries: Vec<(String, Sig)>,
    #[cfg(feature = "params")]
    params: Vec<Param>,
}

/// An extern the program declared: the C symbol, the signature, the host function (taken from the registrations when it was declared, or
/// given later by `Declarations::bind`) and the function value programs call.
pub struct Extern {
    pub sym: String,
    pub sig: Sig,
    host: HostSlot,
    value: RefCell<Value>,
}

type HostSlot = Rc<RefCell<Option<NativeFn>>>;

#[cfg(feature = "params")]
/// A tunable input register the program declared with `.param`: its name, its scalar type and the value it started with. For
/// `fixed<W,I>` the value is the number as declared (the interpreter has no fixed-point values).
pub struct Param {
    pub name: String,
    pub ty: Ty,
    pub default: Value,
}

/// What the program has declared (`.extern`, `.entry`, `.param`), for the host that runs it: a handle on the tables of one
/// interpreter, from [`register`] (`Runtime::declarations`).
#[derive(Clone)]
pub struct Declarations(Declared);

impl Declarations {
    #[cfg(feature = "params")]
    pub(crate) fn visit_host_roots(&self, visit: &mut dyn FnMut(&Value)) {
        for param in &self.0.borrow().params {
            visit(&param.default);
        }
    }

    /// The declared externs, (C symbol, signature), in order of declaration.
    pub fn externs(&self) -> Vec<(String, Sig)> {
        self.0.borrow().externs.iter().map(|e| (e.sym.clone(), e.sig.clone())).collect()
    }
    /// The host's lookup of stand-ins (see [`Resolver`]): asked at every call of a declared extern, so a stand-in may be set, replaced or
    /// removed after the declaration. Also asked for `.fixed`, which the interpreter has no fixed-point arrays for.
    #[cfg(feature = "diag")]
    pub fn set_resolver(&self, r: Option<Resolver>) {
        self.0.borrow_mut().resolver = r;
    }
    /// The declared entries, (name, signature), in order of declaration.
    pub fn entries(&self) -> Vec<(String, Sig)> {
        self.0.borrow().entries.clone()
    }
    /// The declared parameters, in order of declaration.
    #[cfg(feature = "params")]
    pub fn params(&self) -> Vec<(String, Ty, Value)> {
        self.0.borrow().params.iter().map(|p| (p.name.clone(), p.ty.clone(), p.default.clone())).collect()
    }
    /// The host function for the C symbol `sym` (replacing the one it had): before the program declares the extern, it is the function the
    /// declaration takes; after, the function the program already holds calls it from now on. Its calls are checked against the signature.
    /// The program has no other way to it: it is not a global.
    pub fn provide(&self, sym: &str, host: NativeFn) -> Result<()> {
        let slot = slot_of(&mut self.0.borrow_mut(), sym)?;
        *slot.borrow_mut() = Some(host);
        Ok(())
    }
    /// `provide` for a symbol that is declared: a symbol that is not declared is a `name` error.
    pub fn bind(&self, sym: &str, host: NativeFn) -> Result<()> {
        if !self.0.borrow().externs.iter().any(|e| e.sym == sym) {
            return Err(Error::name(klong_core::msg!("{} is not a declared extern", sym)));
        }
        self.provide(sym, host)
    }
    /// Set the parameter `name` between runs (`klongpy.static.set_param`): the value is checked against the declared type and
    /// converted (`int` takes an int, `float` an int or a float, `fixed<W,I>` an int or a float, kept as the number), assigned to
    /// the global and returned.
    #[cfg(feature = "params")]
    pub fn set_param(&self, ip: &mut Interp, name: &str, value: &Value) -> Result<Value> {
        let ty = match self.0.borrow().params.iter().find(|p| p.name == name) {
            Some(p) => p.ty.clone(),
            None => {
                return Err(Error::name(klong_core::msg!("{} is not a parameter of the program", name))
                    .with_hint(klong_core::msg!("declare it: .param(\"{}\";\"int\")", name)))
            }
        };
        let what = klong_core::msg!("the value of parameter {}", name);
        let v = match (&ty.elem, value) {
            (Elem::Fixed(..), Value::Int(_) | Value::Flt(_)) => value.clone(),
            (Elem::Fixed(..), _) => return Err(Error::typ(klong_core::msg!("{} must be {}, got {}", what, ty, describe(value)))),
            _ => conform(&ty, value, &what).map_err(|m| match m {
                Mismatch::Type(s) => Error::typ(s),
                Mismatch::Shape(s) => Error::shape(s),
            })?,
        };
        ip.define_global(name, v.clone())?;
        Ok(v)
    }
}

/// The arguments `.extern.arg` has collected for the next `.extern.call`.
type Stash = Rc<RefCell<Vec<Value>>>;

/// The slot of the host function of `sym`, made when there is none.
fn slot_of(t: &mut Tables, sym: &str) -> Result<HostSlot> {
    if let Some((_, s)) = t.provided.iter().find(|(n, _)| n == sym) {
        return Ok(s.clone());
    }
    let s: HostSlot = Rc::new(RefCell::new(None));
    try_push(&mut t.provided, (try_str(sym)?, s.clone()))?;
    Ok(s)
}

fn no_stand_in(sym: &str) -> Error {
    Error::system(klong_core::msg!("extern {} has no stand-in on the host", sym))
        .with_hint("the host registers a function under that C symbol before the program declares it (klong_cfun_register in C)")
}

/// A call of a declared extern: a check of the arguments, a call of the host function, a check of the result.
fn invoke(declared: &Declared, e: &Rc<Extern>, ip: &mut Interp, a: &[Value]) -> Result<Value> {
    let sig = &e.sig;
    if a.len() != sig.params.len() {
        return Err(Error::arity(klong_core::msg!(
            "{} is declared \"{}\" and takes {}, got {}",
            e.sym,
            sig,
            plural(sig.params.len(), "argument"),
            a.len()
        )));
    }
    let host = e.host.borrow().clone();
    #[cfg(feature = "diag")]
    let resolver = declared.borrow().resolver.clone();
    #[cfg(not(feature = "diag"))]
    let _ = declared;
    #[cfg(feature = "diag")]
    if host.is_none() && resolver.is_none() {
        return Err(no_stand_in(&e.sym));
    }
    #[cfg(not(feature = "diag"))]
    if host.is_none() {
        return Err(no_stand_in(&e.sym));
    }
    let mut args = try_vec(a.len())?;
    for (i, (ty, v)) in sig.params.iter().zip(a).enumerate() {
        try_push(&mut args, conform_extern(ty, v, &klong_core::msg!("argument {} of {}", i + 1, e.sym), sig)?)?;
    }
    #[cfg(feature = "diag")]
    let stand_in = match resolver {
        Some(r) => r(ip, &e.sym, &args),
        None => None,
    };
    #[cfg(not(feature = "diag"))]
    let stand_in: Option<Result<Value>> = None;
    let r = match stand_in {
        Some(v) => v?,
        None => match &host {
            Some(h) => h(ip, &args)?,
            None => return Err(no_stand_in(&e.sym)),
        },
    };
    conform_extern(&sig.result, &r, &klong_core::msg!("the result of {}", e.sym), sig)
}

fn string<'a>(which: &str, v: &'a Value, what: &str) -> Result<&'a str> {
    match v {
        Value::Str(s) => Ok(s.as_str()),
        _ => Err(Error::typ(klong_core::msg!("{}: the {} must be a string", which, what))),
    }
}

fn extern_fn(declared: &Declared, ip: &mut Interp, a: &[Value]) -> Result<Value> {
    if a.len() != 2 {
        return Err(Error::arity(klong_core::msg!(".extern takes 2 arguments (the C symbol and the signature), got {}", a.len()))
            .with_hint("the module placeholder, a third argument, is gone; write adc::.extern(\"read_adc\";\"int -> float\")"));
    }
    let sym = string(".extern", &a[0], "C symbol")?;
    let text = string(".extern", &a[1], "signature")?;
    let sig = signature(".extern", text, true)?;
    if let Some(bad) = check_symbol(sym) {
        return Err(Error::typ(klong_core::msg!(".extern: {}", bad))
            .with_hint("the first argument is the C symbol (read_adc); the name a program uses is the key of the module's export"));
    }
    if let Some(old) = declared.borrow().externs.iter().find(|e| e.sym == sym) {
        if old.sig != sig {
            return Err(Error::typ(klong_core::msg!(".extern: {} is already declared \"{}\"; it cannot also be \"{}\"", sym, old.sig, sig)));
        }
        return Ok(old.value.borrow().clone());
    }
    // the function the host provided under that symbol is the one that is called, reached only through the checks
    let host = slot_of(&mut declared.borrow_mut(), sym)?;
    let n = sig.params.len();
    let e = Rc::new(Extern { sym: try_str(sym)?, sig, host, value: RefCell::new(Value::Undefined) });
    let value = if n <= 3 {
        // x, y and z: a wildcard function is given whichever of them the call supplied
        let s = ip.syms.intern(sym);
        if let Some(err) = ip.syms.failure() {
            return Err(err);
        }
        let (d, ex) = (declared.clone(), e.clone());
        Value::Fn(Rc::new(Func::native(s, WILDCARD, Rc::new(move |ip: &mut Interp, a: &[Value]| invoke(&d, &ex, ip, a)))))
    } else {
        wide(ip, declared, sym, n)?
    };
    *e.value.borrow_mut() = value.clone();
    try_push(&mut declared.borrow_mut().externs, e)?;
    Ok(value)
}

/// A function of more than three parameters is one with named parameters whose body hands its parameters to the declaration, three
/// at a time (`.extern.arg`) and the last ones with the symbol (`.extern.call`). It is a function value, and nothing is defined.
fn wide(ip: &mut Interp, declared: &Declared, sym: &str, n: usize) -> Result<Value> {
    register_wide(ip, declared);
    let mut text = String::new();
    let mut put = |t: &str| -> Result<()> { klong_core::value::try_push_str(&mut text, t) };
    put("f(")?;
    for i in 1..=n {
        put(if i > 1 { ";arg" } else { "arg" })?;
        put(&klong_core::msg!("{}", i))?;
    }
    put(")::{")?;
    let mut i = 1;
    // (a native function is given at most three arguments: the symbol and two more go with the call)
    while n + 1 - i > 2 {
        put(&klong_core::msg!(".extern.arg(arg{};arg{};arg{});", i, i + 1, i + 2))?;
        i += 3;
    }
    put(&klong_core::msg!(".extern.call(\"{}\"", sym))?;
    while i <= n {
        put(&klong_core::msg!(";arg{}", i))?;
        i += 1;
    }
    put(")}")?;
    ip.function_from_text(&text)
}

fn entry_fn(declared: &Declared, ip: &mut Interp, a: &[Value]) -> Result<Value> {
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
    let canonical = klong_core::msg!("{}", sig);
    let mut t = declared.borrow_mut();
    if !t.entries.iter().any(|(n, s)| n == name && *s == sig) {
        try_push(&mut t.entries, (try_str(name)?, sig))?;
    }
    drop(t);
    Value::text(&canonical)
}

#[cfg(feature = "params")]
fn float_fn(a: &[Value]) -> Result<Value> {
    // (an array of the host, a torch tensor, is read as the native array it copies to)
    let arg = match a.first() {
        Some(v) => Some(v.unhost(".float")?),
        None => None,
    };
    match arg.as_ref() {
        Some(Value::Int(i)) => Ok(Value::Flt(*i as klong_core::Flt)),
        Some(v @ (Value::Flt(_) | Value::Flts(_))) => Ok(v.clone()),
        Some(Value::Ints(a)) => {
            let mut data = try_vec(a.data.len())?;
            data.extend(a.data.iter().map(|&i| i as klong_core::Flt));
            Ok(Value::Flts(Rc::new(Arr::shaped(data, &dims_of(a)))))
        }
        _ => Err(Error::typ(".float: the argument must be a fixed-point number or an array of numbers")),
    }
}

#[cfg(feature = "params")]
fn param_fn(declared: &Declared, ip: &mut Interp, a: &[Value]) -> Result<Value> {
    if !(2..=3).contains(&a.len()) {
        return Err(Error::arity(klong_core::msg!(".param takes 2 or 3 arguments, got {}", a.len())));
    }
    let name = string(".param", &a[0], "name")?;
    let text = string(".param", &a[1], "type")?;
    let t = text.as_bytes();
    let (pos, end) = (t.len() - text.trim_start().len(), text.trim_end().len());
    let ty = if pos >= end {
        Err(sig_err(String::from("expected a type but found nothing"), 0))
    } else {
        type_at(t, pos, end).and_then(|(ty, stop)| {
            if stop == end {
                Ok(ty)
            } else {
                Err(sig_err(klong_core::msg!("unexpected '{}' after the type", text[stop..end].trim()), stop))
            }
        })
    };
    let ty = ty.map_err(|e| {
        Error::typ(klong_core::msg!(".param: {}", e.message)).with_hint("a type is int, float or fixed<W,I>, for example \"fixed<32,16>\"")
    })?;
    if !ty.shape.is_empty() || matches!(ty.elem, Elem::Str | Elem::Void) {
        return Err(Error::typ(klong_core::msg!(".param: {} must be a scalar int, float or fixed<W,I>, not {}", name, ty)));
    }
    if let Some(bad) = check_param_name(name) {
        return Err(Error::typ(klong_core::msg!(".param: {}", bad)));
    }
    let default = match (a.get(2), &ty.elem) {
        (None, Elem::Int) => Value::Int(0),
        (None, Elem::Float) => Value::Flt(0.0),
        (None, _) => Value::Int(0),
        (Some(v @ Value::Int(_)), Elem::Int | Elem::Fixed(..)) => v.clone(),
        (Some(Value::Int(i)), Elem::Float) => Value::Flt(*i as klong_core::Flt),
        (Some(v @ Value::Flt(_)), Elem::Float | Elem::Fixed(..)) => v.clone(),
        (Some(_), Elem::Int) => return Err(Error::typ(klong_core::msg!(".param: the default of {} must be an int", name))),
        (Some(v), Elem::Fixed(..)) => {
            return Err(Error::typ(klong_core::msg!(".param {}: the default must be a number or an array of numbers, got {}", name, describe(v))))
        }
        (Some(_), _) => return Err(Error::typ(klong_core::msg!(".param: the default of {} must be a number", name))),
    };
    let canonical = klong_core::msg!("{}", ty);
    if let Some(old) = declared.borrow().params.iter().find(|p| p.name == name) {
        if old.ty != ty {
            return Err(Error::typ(klong_core::msg!(".param: {} is already declared as {}", name, old.ty)));
        }
        return Value::text(&canonical);
    }
    if ip.global(name).is_some() {
        return Err(Error::name(klong_core::msg!(".param: {} is already defined in the program", name))
            .with_hint("a parameter is defined by .param itself: remove the other definition"));
    }
    ip.define_global(name, default.clone())?;
    try_push(&mut declared.borrow_mut().params, Param { name: try_str(name)?, ty, default })?;
    Value::text(&canonical)
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
            let sym = string(".extern.call", a.first().ok_or_else(|| Error::arity(".extern.call takes a symbol"))?, "symbol")?;
            let e = declared.borrow().externs.iter().find(|e| e.sym == sym).cloned();
            let e = e.ok_or_else(|| Error::name(klong_core::msg!("{} is not a declared extern", sym)))?;
            let mut args = core::mem::take(&mut *stash.try_borrow_mut().map_err(|_| Error::system("extern arguments are in use"))?);
            for v in &a[1..] {
                try_push(&mut args, v.clone())?;
            }
            invoke(&declared, &e, ip, &args)
        }),
    );
}

/// Define `.extern`, `.entry`, `.param`, `.float` and `.fixed`.
///
/// `.fixed` is an error, not an emulation: KlongPy's interpreter gives `.fixed(type;x)` a fixed-point array that knows its type,
/// and arithmetic on it follows the reference semantics of `klongpy/static/fixed.py` operation by operation (an add, a multiply
/// and a divide that wrap to the width, floor on conversion, a reduction that wraps at each step, a refusal to mix with floats).
/// The native values have no such kind and every operator would need a case for it, so a fixed-point program is compiled or run
/// by KlongPy.
///
/// With the `params` feature (on by default; the board images leave it off) there are `.param` and `.float`.
///
/// A fixed-point `.param` is declared and holds the number it was given: the value is not a fixed-point one, so a program that
/// depends on the wrapping of the width runs in KlongPy. `.float` takes ints and number arrays only.
pub fn register(ip: &mut Interp) -> Declarations {
    let declared: Declared = Rc::new(RefCell::new(Tables::default()));
    let d = declared.clone();
    ip.register(".extern", WILDCARD, Rc::new(move |ip: &mut Interp, a: &[Value]| extern_fn(&d, ip, a)));
    let d = declared.clone();
    ip.register(".entry", WILDCARD, Rc::new(move |ip: &mut Interp, a: &[Value]| entry_fn(&d, ip, a)));
    #[cfg(feature = "params")]
    let d = declared.clone();
    #[cfg(feature = "params")]
    ip.register(".param", WILDCARD, Rc::new(move |ip: &mut Interp, a: &[Value]| param_fn(&d, ip, a)));
    #[cfg(feature = "params")]
    ip.register(".float", 1, Rc::new(|_: &mut Interp, a: &[Value]| float_fn(a)));
    let d = declared.clone();
    ip.register(
        ".fixed",
        WILDCARD,
        Rc::new(move |ip: &mut Interp, a: &[Value]| {
            // the host may supply fixed-point arrays (see `Declarations::set_resolver`, asked for ".fixed")
            #[cfg(feature = "diag")]
            {
                let r = d.borrow().resolver.clone();
                if let Some(r) = r {
                    if let Some(v) = r(ip, ".fixed", a) {
                        return v;
                    }
                }
            }
            #[cfg(not(feature = "diag"))]
            let _ = (&d, ip, a);
            Err::<Value, Error>(
                Error::system("fixed-point programs are not supported by the native interpreter")
                    .with_hint("run the program with KlongPy (klongpy.static.call_entry) or compile it with kgpy compile"),
            )
        }),
    );
    Declarations(declared)
}
