//! Interned symbols. Names and operator glyphs share one table.
use alloc::collections::BTreeMap;
use alloc::rc::Rc;
#[cfg(feature = "diag")]
use alloc::string::String;
use alloc::vec::Vec;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct SymId(pub u32);

pub const X: SymId = SymId(0);
pub const Y: SymId = SymId(1);
pub const Z: SymId = SymId(2);
pub const DOT_F: SymId = SymId(3);
/// Symbols 0..RESERVED are x, y, z and .f; the operator glyphs of `ops::OPS` follow, in order.
pub const FIRST_OP: u32 = 4;

#[cfg(feature = "diag")]
/// What the parser accepted at strict level 0 that it should not have (KlongPy's `KlongSyntaxWarning`): the input still means what it
/// always meant. At level 1 and above the same condition is a syntax error with the same message and hint.
pub struct Warning {
    /// The condition: `trailing-input`, `stray-closer`, `missing-operand`, `unterminated-string`, `unterminated-list`,
    /// `unterminated-dict`, `unterminated-comment`, `not-a-destructuring` or `not-a-definition`.
    pub code: &'static str,
    pub message: String,
    /// What happens to the input (it is part of the warning text, after a `; `).
    pub note: Option<String>,
    pub hint: Option<String>,
    pub span: crate::error::Span,
    pub source: Option<Rc<crate::error::Source>>,
}

/// The most warnings kept from one program unless the embedder says otherwise (`Interner::max_warnings`).
#[cfg(feature = "diag")]
pub const MAX_WARNINGS: usize = 64;

pub struct Interner {
    /// The warnings of the parses since they were last taken (`Interp::take_warnings`).
    #[cfg(feature = "diag")]
    pub warnings: Vec<Warning>,
    /// How many warnings `warn` keeps before it drops the rest (`MAX_WARNINGS`: an interpreter must not grow without bound on a program
    /// that warns at every token; a tool that reports them all, such as the linter, raises it).
    #[cfg(feature = "diag")]
    pub max_warnings: usize,
    /// Keep, in each tree, where the operands of its nodes were written (`tree::Aux`): for the tools that point at them.
    #[cfg(feature = "diag")]
    pub record_spans: bool,
    names: Vec<Rc<str>>,
    /// The qualified names the parser has read and the symbol of the first segment of each (empty in a program that has none).
    quals: Vec<(u32, u32)>,
    map: BTreeMap<Rc<str>, u32>,
    /// A name could not be stored (no memory); `x` was returned in its place. The parser
    /// reports it as a `memory` error.
    pub oom: bool,
    /// A name longer than `MAX_NAME` was refused (`x` was returned in its place).
    pub long: bool,
}

/// The longest name, in bytes: a name is stored once and for good, and its text is copied
/// where no failure can be reported, so it must be small (a module name and a name together
/// stay under the size the allocator treats as "sized by data").
pub const MAX_NAME: usize = 255;

impl Interner {
    pub fn new() -> Interner {
        let mut t = Interner { #[cfg(feature = "diag")] warnings: Vec::new(), #[cfg(feature = "diag")] max_warnings: MAX_WARNINGS, #[cfg(feature = "diag")] record_spans: false, names: Vec::new(), quals: Vec::new(), map: BTreeMap::new(), oom: false, long: false };
        for n in ["x", "y", "z", ".f"] {
            t.intern(n);
        }
        for op in crate::ops::OPS.iter() {
            t.intern(op.glyph);
        }
        t
    }

    /// Record a warning (dropped when there is no room: a warning is not worth a failure).
    #[cfg(feature = "diag")]
    pub fn warn(&mut self, w: Warning) {
        if self.warnings.len() < self.max_warnings && self.warnings.try_reserve(1).is_ok() {
            self.warnings.push(w);
        }
    }

    pub fn intern(&mut self, s: &str) -> SymId {
        if let Some(&i) = self.map.get(s) {
            return SymId(i);
        }
        // (a module-qualified name is two names and a backquote)
        if s.len() > 2 * MAX_NAME + 1 || (s.len() > MAX_NAME && !s.contains('`')) {
            self.long = true;
            return X;
        }
        let _tag = crate::htag::Scope::new(crate::htag::SYMBOLS);
        if self.names.try_reserve(1).is_err() {
            self.oom = true;
            return X;
        }
        let rc: Rc<str> = Rc::from(s);
        let i = self.names.len() as u32;
        self.names.push(rc.clone());
        self.map.insert(rc, i);
        SymId(i)
    }

    /// Record that `s` is a qualified name whose first segment is the symbol `base` and that has `keys` keys after it.
    pub fn set_qual(&mut self, s: SymId, base: SymId, keys: usize) {
        let _tag = crate::htag::Scope::new(crate::htag::SYMBOLS);
        if let Err(i) = self.quals.binary_search_by_key(&s.0, |e| e.0) {
            if self.quals.try_reserve(1).is_ok() {
                // (the number of keys in the top byte of the base: a name has at most 127 segments)
                self.quals.insert(i, (s.0, base.0 | (keys.min(255) as u32) << 24));
            }
        }
    }

    pub fn has_quals(&self) -> bool {
        !self.quals.is_empty()
    }

    /// The symbol of the first segment of `s` when it is a qualified name.
    pub fn qual_base(&self, s: SymId) -> Option<SymId> {
        self.qual_entry(s).map(|e| SymId(e & 0x00ff_ffff))
    }

    /// How many keys the qualified name `s` has after its base (0 for a name that is not qualified).
    pub fn qual_keys(&self, s: SymId) -> usize {
        self.qual_entry(s).map_or(0, |e| (e >> 24) as usize)
    }

    fn qual_entry(&self, s: SymId) -> Option<u32> {
        if self.quals.is_empty() {
            return None;
        }
        self.quals.binary_search_by_key(&s.0, |e| e.0).ok().map(|i| self.quals[i].1)
    }

    /// The failure of the interning since the last call, if any.
    pub fn failure(&mut self) -> Option<crate::error::Error> {
        if core::mem::take(&mut self.long) {
            self.oom = false;
            return Some(crate::error::Error::range("a name is longer than 255 bytes"));
        }
        if core::mem::take(&mut self.oom) {
            return Some(crate::error::Error::memory());
        }
        None
    }

    pub fn lookup(&self, s: &str) -> Option<SymId> {
        self.map.get(s).map(|&i| SymId(i))
    }

    pub fn name(&self, s: SymId) -> &str {
        self.names.get(s.0 as usize).map(|r| &**r).unwrap_or("?")
    }

    pub fn len(&self) -> usize {
        self.names.len()
    }

    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }

    /// Give back the room the table of names has beyond its names.
    pub fn shrink(&mut self) {
        let _tag = crate::htag::Scope::new(crate::htag::SYMBOLS);
        if self.names.try_reserve_exact(1).is_ok() {
            self.names.shrink_to_fit();
        }
    }
}

impl Default for Interner {
    fn default() -> Self {
        Self::new()
    }
}

/// x, y or z.
pub fn is_arg(s: SymId) -> bool {
    s.0 < 3
}

/// The segments of a qualified name (`a.b.c`): `None` for a name that is not qualified (no dot, or one that starts with a dot: `.p`,
/// `.srv.o` are one whole name); an error text when a segment is not a name (it starts with a digit, or is empty).
pub fn qual_segments(v: &str) -> Option<Result<Vec<&str>, &'static str>> {
    if v.starts_with('.') || !v.contains('.') {
        return None;
    }
    let mut out = Vec::new();
    for seg in v.split('.') {
        let mut cs = seg.chars();
        match cs.next() {
            None => return Some(Err("a qualified name has an empty segment (a trailing or a doubled dot)")),
            Some(c) if c.is_alphabetic() => {}
            Some(_) => return Some(Err("a segment of a qualified name cannot start with a digit: write a@5 for the entry 5")),
        }
        out.push(seg);
    }
    Some(Ok(out))
}

