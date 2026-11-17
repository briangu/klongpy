//! Interned symbols. Names and operator glyphs share one table.
use alloc::collections::BTreeMap;
use alloc::rc::Rc;
use alloc::vec::Vec;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct SymId(pub u32);

pub const X: SymId = SymId(0);
pub const Y: SymId = SymId(1);
pub const Z: SymId = SymId(2);
pub const DOT_F: SymId = SymId(3);
/// Symbols 0..RESERVED are x, y, z and .f; the operator glyphs of `ops::OPS` follow, in order.
pub const FIRST_OP: u32 = 4;

pub struct Interner {
    names: Vec<Rc<str>>,
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
        let mut t = Interner { names: Vec::new(), map: BTreeMap::new(), oom: false, long: false };
        for n in ["x", "y", "z", ".f"] {
            t.intern(n);
        }
        for op in crate::ops::OPS.iter() {
            t.intern(op.glyph);
        }
        t
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
        self.names.shrink_to_fit();
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
