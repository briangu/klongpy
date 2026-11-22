//! Signatures of host functions and entry points (`klongpy/static/signature.py`): `int float[64] -> float`.
use std::fmt;

pub const MAX_RANK: usize = 4;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Ty {
    pub elem: String,
    pub shape: Vec<usize>,
}

impl Ty {
    pub fn new(elem: &str, shape: Vec<usize>) -> Ty {
        Ty { elem: elem.to_string(), shape }
    }
}

impl fmt::Display for Ty {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.shape.is_empty() {
            write!(f, "{}", self.elem)
        } else {
            let d: Vec<String> = self.shape.iter().map(|d| d.to_string()).collect();
            write!(f, "{}[{}]", self.elem, d.join(","))
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Signature {
    pub params: Vec<Ty>,
    pub result: Ty,
}

impl Signature {
    pub fn arity(&self) -> usize {
        self.params.len()
    }
}

impl fmt::Display for Signature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let left: Vec<String> = self.params.iter().map(|p| p.to_string()).collect();
        let left = left.join(" ");
        if left.is_empty() {
            write!(f, "-> {}", self.result)
        } else {
            write!(f, "{} -> {}", left, self.result)
        }
    }
}

#[derive(Clone, Debug)]
pub struct ExternDecl {
    /// The name the compiler knows it by: `prefix.name` for an extern a module declares, `name` in the program itself.
    pub name: String,
    /// The C symbol (the first argument of `.extern`): what the generated C declares and calls, and what a host registers.
    pub c_symbol: String,
    pub signature: Signature,
    pub span: (usize, usize),
    pub src: usize,
}

#[derive(Clone, Debug)]
pub struct EntryDecl {
    pub name: String,
    pub signature: Signature,
    pub span: (usize, usize),
    pub src: usize,
}

/// A malformed signature: one line, and the span of the signature text it is about (in characters).
#[derive(Debug)]
pub struct SigError {
    pub message: String,
    pub span: (usize, usize),
}

fn err<T>(message: impl Into<String>, span: (usize, usize)) -> Result<T, SigError> {
    Err(SigError { message: message.into(), span })
}

pub fn is_fixed(elem: &str) -> bool {
    elem.starts_with("fixed<")
}

/// (W, I) of `fixed<W,I>`.
pub fn fixed_spec(elem: &str) -> Option<(u32, u32)> {
    let inner = elem.strip_prefix("fixed<")?.strip_suffix('>')?;
    let (w, i) = inner.split_once(',')?;
    if w.is_empty() || i.is_empty() || !w.bytes().all(|b| b.is_ascii_digit()) || !i.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    Some((w.parse().ok()?, i.parse().ok()?))
}

pub fn int_range(bits: u32) -> (i128, i128) {
    if bits == 64 {
        (-(1i128 << 63), (1i128 << 63) - 1)
    } else {
        (-(1i128 << 31), (1i128 << 31) - 1)
    }
}

// The texts are indexed by characters as Python does: `chars` is the text as a vector of them.

struct Text {
    c: Vec<char>,
}

impl Text {
    fn s(&self, a: usize, b: usize) -> String {
        self.c[a.min(self.c.len())..b.min(self.c.len())].iter().collect()
    }
}

fn is_ws(c: char) -> bool {
    c.is_whitespace()
}

/// `([A-Za-z]+)(<[^<>\s]*>)?(\[([^\]]*)\])?` at `pos` within `end`: group spans.
struct TypeMatch {
    word: (usize, usize),
    angle: Option<(usize, usize)>,
    dims: Option<(usize, usize)>,
    inner: (usize, usize),
    end: usize,
}

fn match_type(t: &Text, pos: usize, end: usize) -> Option<TypeMatch> {
    let mut i = pos;
    while i < end && t.c[i].is_ascii_alphabetic() {
        i += 1;
    }
    if i == pos {
        return None;
    }
    let word = (pos, i);
    let mut angle = None;
    if i < end && t.c[i] == '<' {
        let mut j = i + 1;
        while j < end && t.c[j] != '<' && t.c[j] != '>' && !is_ws(t.c[j]) {
            j += 1;
        }
        if j < end && t.c[j] == '>' {
            angle = Some((i, j + 1));
            i = j + 1;
        }
    }
    let mut dims = None;
    let mut inner = (0, 0);
    if i < end && t.c[i] == '[' {
        let mut j = i + 1;
        while j < end && t.c[j] != ']' {
            j += 1;
        }
        if j < end {
            dims = Some((i, j + 1));
            inner = (i + 1, j);
            i = j + 1;
        }
    }
    Some(TypeMatch { word, angle, dims, inner, end: i })
}

fn word_at(t: &Text, pos: usize, end: usize) -> (usize, usize) {
    let mut i = pos;
    while i < end && !is_ws(t.c[i]) {
        i += 1;
    }
    (pos, i)
}

fn type_at(t: &Text, pos: usize, end: usize) -> Result<(Ty, usize), SigError> {
    let Some(m) = match_type(t, pos, end) else {
        let w = word_at(t, pos, end);
        let stop = if w.1 > w.0 { w.1 } else { (pos + 1).min(end) };
        return err(format!("expected a type but found '{}'", t.s(pos, stop)), (pos, stop));
    };
    let mut word = t.s(m.word.0, m.word.1);
    if word == "fixed" {
        let Some(a) = m.angle else {
            return err("fixed needs its widths: fixed<W,I> (W total bits, I integer bits)", m.word);
        };
        let angle = t.s(a.0, a.1);
        let body = &angle[1..angle.len() - 1];
        let ok = body.split_once(',').filter(|(w, i)| !w.is_empty() && !i.is_empty() && w.bytes().all(|b| b.is_ascii_digit()) && i.bytes().all(|b| b.is_ascii_digit()));
        let Some((w, i)) = ok else {
            return err(format!("a fixed-point type is written fixed<W,I> with two positive integers, found 'fixed{}'", angle), a);
        };
        let (w, i): (i64, i64) = (w.parse().unwrap_or(i64::MAX), i.parse().unwrap_or(i64::MAX));
        if !(2..=64).contains(&w) {
            return err(format!("fixed<{},{}>: the total width W must be 2 to 64", w, i), a);
        }
        if !(1..=w).contains(&i) {
            return err(format!("fixed<{},{}>: the integer bits I must be 1 to W ({})", w, i, w), a);
        }
        word = format!("fixed<{},{}>", w, i);
    } else {
        if let Some(a) = m.angle {
            return err(format!("{} takes no <...> (only fixed<W,I> does)", word), a);
        }
        if !["int", "float", "str", "void"].contains(&word.as_str()) {
            return err(format!("unknown type '{}' (the types are int, float, fixed<W,I>, str and void)", word), m.word);
        }
    }
    let mut shape = Vec::new();
    if let Some(d) = m.dims {
        let inner = t.s(m.inner.0, m.inner.1);
        if word == "str" || word == "void" {
            return err(format!("{} cannot have dimensions", word), d);
        }
        // \s*\d+\s*(,\s*\d+\s*)*\Z
        let parts: Vec<&str> = inner.split(',').collect();
        let ok = parts.iter().all(|p| {
            let p = p.trim_matches(|c: char| is_ws(c));
            !p.is_empty() && p.chars().all(|c| c.is_ascii_digit())
        });
        if !ok {
            return err(format!("dimensions must be positive integers separated by commas, found '[{}]'", inner), d);
        }
        for p in parts {
            shape.push(p.trim().parse::<usize>().unwrap_or(usize::MAX));
        }
        if shape.iter().any(|&d| d == 0) {
            return err("every dimension must be a positive integer", d);
        }
        if shape.len() > MAX_RANK {
            return err(format!("an array has at most {} dimensions, found {}", MAX_RANK, shape.len()), d);
        }
    }
    let stop = m.end;
    if stop < end && !is_ws(t.c[stop]) {
        let w = word_at(t, pos, end);
        return err(format!("'{}' is not a type", t.s(w.0, w.1)), w);
    }
    Ok((Ty { elem: word, shape }, stop))
}

/// Parse one type such as `float[16,241]`.
pub fn parse_type(text: &str) -> Result<Ty, SigError> {
    let t = Text { c: text.chars().collect() };
    let n = t.c.len();
    let mut pos = 0;
    while pos < n && is_ws(t.c[pos]) {
        pos += 1;
    }
    let mut end = n;
    while end > 0 && is_ws(t.c[end - 1]) {
        end -= 1;
    }
    if pos >= end {
        return err("expected a type but found nothing", (0, n));
    }
    let (ty, stop) = type_at(&t, pos, end)?;
    if stop != end {
        return err(format!("unexpected '{}' after the type", t.s(stop, end).trim()), (stop, end));
    }
    Ok(ty)
}

fn find(t: &Text, pat: &str, from: usize) -> Option<usize> {
    let p: Vec<char> = pat.chars().collect();
    (from..t.c.len()).find(|&i| t.c[i..].starts_with(&p))
}

/// Parse `params -> result`.
pub fn parse_signature(text: &str, allow_str: bool) -> Result<Signature, SigError> {
    let t = Text { c: text.chars().collect() };
    let n = t.c.len();
    let Some(arrow) = find(&t, "->", 0) else {
        return err("a signature needs '->' between the parameter types and the result type", (0, n));
    };
    if let Some(second) = find(&t, "->", arrow + 2) {
        return err("a signature has exactly one '->'", (second, second + 2));
    }
    let mut params = Vec::new();
    let mut pos = 0;
    loop {
        while pos < arrow && is_ws(t.c[pos]) {
            pos += 1;
        }
        if pos >= arrow {
            break;
        }
        let start = pos;
        let (ty, p) = type_at(&t, pos, arrow)?;
        pos = p;
        if ty.elem == "void" {
            return err("void is a result type only; a function of no parameters is written \"-> int\"", (start, pos));
        }
        if ty.elem == "str" && !allow_str {
            return err("str is a parameter type of externs only", (start, pos));
        }
        params.push(ty);
    }
    pos = arrow + 2;
    let mut end = n;
    while end > 0 && is_ws(t.c[end - 1]) {
        end -= 1;
    }
    while pos < end && is_ws(t.c[pos]) {
        pos += 1;
    }
    if pos >= end {
        return err("a signature needs a result type after '->' (void for none)", (arrow, n));
    }
    let start = pos;
    let (result, p) = type_at(&t, pos, end)?;
    pos = p;
    if result.elem == "str" {
        return err("str cannot be a result type", (start, pos));
    }
    if pos != end {
        let mut rest = pos;
        while rest < end && is_ws(t.c[rest]) {
            rest += 1;
        }
        return err(format!("unexpected '{}' after the result type", t.s(rest, end)), (rest, end));
    }
    Ok(Signature { params, result })
}

fn valid_name(name: &str) -> bool {
    // [A-Za-z][A-Za-z0-9.]*\Z
    let mut it = name.chars();
    match it.next() {
        Some(c) if c.is_ascii_alphabetic() => it.all(|c| c.is_ascii_alphanumeric() || c == '.'),
        _ => false,
    }
}

/// Python's `repr` of a str.
fn py_repr(s: &str) -> String {
    crate::util::py_repr(s)
}

fn check_name(name: &str, what: &str) -> Option<String> {
    if !valid_name(name) {
        return Some(format!("{} name must be a Klong name of ASCII letters, digits and '.', starting with a letter (found {})", what, py_repr(name)));
    }
    if ["x", "y", "z"].contains(&name) {
        return Some(format!("{} is a function argument and cannot be {} name", name, what));
    }
    None
}

pub const C_KEYWORDS: &[&str] = &[
    "auto", "break", "case", "char", "const", "continue", "default", "do", "double", "else", "enum", "extern", "float", "for", "goto", "if", "inline", "int", "long",
    "register", "restrict", "return", "short", "signed", "sizeof", "static", "struct", "switch", "typedef", "union", "unsigned", "void", "volatile", "while", "_Bool",
    "_Complex", "_Imaginary", "main", "memcpy", "memset", "NULL",
];

pub fn c_identifier(name: &str) -> String {
    name.replace('.', "_")
}

pub fn check_extern_name(name: &str) -> Option<String> {
    if let Some(b) = check_name(name, "an extern") {
        return Some(b);
    }
    let c = c_identifier(name);
    if C_KEYWORDS.contains(&c.as_str()) {
        return Some(format!("the C name of extern {} is '{}', which is reserved in C", name, c));
    }
    if c.starts_with("kg_") {
        return Some(format!("the C name of extern {} is '{}': names that start with kg_ belong to the runtime", name, c));
    }
    None
}

/// The reason `sym` cannot be the C symbol of an extern, or None (the interpreter's rule: `klong_rt::declare::check_symbol`).
pub fn check_symbol(sym: &str) -> Option<String> {
    klong_rt::declare::check_symbol(sym)
}

pub fn check_entry_name(name: &str) -> Option<String> {
    if let Some(b) = check_name(name, "an entry") {
        return Some(b);
    }
    if ["init", "t", "storage.t", "size"].contains(&name) {
        return Some(format!("an entry cannot be called {}: prog_{} is a name the compiler generates", name, c_identifier(name)));
    }
    None
}
