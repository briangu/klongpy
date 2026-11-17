//! The evaluator: a recursive tree walk, right to left.
//!
//! Calling follows KlongPy (`_eval_fn`, `_resolve_fn`, `apply`, `_run_fn` in
//! `klongpy/interpreter.py`), because the conformance suite pins its consequences:
//!
//! * scope is dynamic: a function sees the locals of its callers (`functions.local-in-nested-call`);
//! * a function value is a piece of the tree, and a projection keeps its argument expressions
//!   unevaluated until the call is complete;
//! * a call with fewer arguments than the function takes is itself the value.
//!
//! Depth, interrupts and memory: every nested evaluation counts against `Config::max_depth`
//! (a `range` error, never a native stack overflow); every call and every adverb iteration
//! ticks, and every 4096 ticks the embedder's [`Hooks::tick`] runs.
use alloc::boxed::Box;
use alloc::rc::Rc;
use alloc::string::String;
use alloc::vec::Vec;
use core::cell::Cell;
use core::sync::atomic::{AtomicBool, Ordering};

use crate::error::{Error, Kind, Result};
use crate::ops;
use crate::parse::{self, Parsed};
use crate::sym::{self, Interner, SymId};
use crate::tree::{Node, NodeId, Range, Tree, NONE};
use crate::value::{try_push, try_vec, Dict, Func, NativeFn, Slot, Value, WILDCARD};

/// What the embedder gives the core. `tick` is called every 4096 calls or adverb iterations;
/// returning true interrupts the evaluation (Ctrl-C on a console).
pub trait Hooks {
    fn tick(&mut self) -> bool;
}

pub struct NoHooks;
impl Hooks for NoHooks {
    fn tick(&mut self) -> bool {
        false
    }
}

pub struct Config {
    /// Limit of nested evaluations: one per nested expression, so one to three per Klong call
    /// (the call, a conditional around it, an operator around that). The embedder must give the
    /// evaluating thread enough native stack for it: about 1 KiB per level in a release build
    /// on a 64-bit host (measured, ticket 102), several times that in a debug build.
    pub max_depth: u32,
    /// Strict level 0 to 2 (`errors-strict-lint.md` section 4).
    pub strict: u8,
    /// Stop with an `interrupt` error after this many ticks (fuzzing, tests).
    pub step_limit: Option<u64>,
    /// The slow tick (the platform's `idle`, Ctrl-C, the lane) runs when `ticks & tick_mask == 0`: every
    /// 4096 calls and adverb iterations by default. A slow processor lowers it: a Cortex-M0+ takes about
    /// 90 us for a call of a Klong function, so 4096 of them are 0.4 s without a look at the lane.
    pub tick_mask: u32,
    /// Set from outside (an interrupt handler) to stop the evaluation. Must live in memory
    /// where atomics work (internal RAM on the chips of `ref-platforms.md`).
    pub interrupt: Option<&'static AtomicBool>,
    /// True when the allocator has passed its soft limit and is living on the reserve: the
    /// evaluation unwinds with a `memory` error at the next tick.
    pub mem_probe: Option<fn() -> bool>,
    /// Bytes of native stack the evaluator and the parser may use, measured from where the
    /// outermost `run` was entered. An evaluation or a parse that would go deeper ends with a
    /// `range` error, like `max_depth`, whatever a level costs on the target (the limit is
    /// checked against the real stack pointer). `None`: only `max_depth` bounds the recursion.
    /// See [`Config::for_stack`].
    pub stack_limit: Option<usize>,
    /// How deep data may nest (lists in lists, projections in projections): the constructors
    /// refuse to build anything deeper, because every walk over data recurses once per level.
    pub max_nest: u32,
}

/// Native stack one level of data nesting costs in the deepest walk (printing, matching, the
/// atomic pairing, drop): about 300 bytes on a 64-bit host in a release build (ticket 103),
/// with a margin.
pub const STACK_PER_NEST: usize = 384;

impl Default for Config {
    fn default() -> Config {
        Config { max_depth: 512, strict: 0, step_limit: None, tick_mask: 4095, interrupt: None, mem_probe: None, stack_limit: None, max_nest: crate::value::DEFAULT_MAX_NEST }
    }
}

impl Config {
    /// Limits for a task whose native stack is `bytes` large: a quarter of it (at least
    /// 2 KiB) is kept for what runs at the bottom of the recursion (an operator walking
    /// nested data, a system function, the platform), the rest is the evaluator's, and data
    /// nesting is capped so that a walk fits in half of what is kept.
    pub fn for_stack(bytes: usize) -> Config {
        Config::for_stack_keep(bytes, 0)
    }

    /// `for_stack` with the part kept for the bottom of the recursion given (0: a quarter). A board that knows
    /// what runs there (the lane's pump, a walk over nested data) can keep less and give the evaluator more.
    pub fn for_stack_keep(bytes: usize, keep: usize) -> Config {
        let keep = if keep == 0 { bytes / 4 } else { keep };
        let keep = keep.max(2048).min(bytes / 2);
        let nest = ((keep / 2) / STACK_PER_NEST).clamp(8, crate::value::DEFAULT_MAX_NEST as usize) as u32;
        Config { stack_limit: Some(bytes - keep), max_nest: nest, max_depth: u32::MAX, ..Config::default() }
    }
}

/// The address of something on the stack of the caller, as a number.
#[inline(always)]
pub(crate) fn stack_here() -> usize {
    let marker = 0u8;
    core::hint::black_box(&marker) as *const u8 as usize
}

/// One global scope (KlongPy's context list): the program's own names, or a module.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Scope {
    Plain,
    Module(SymId),
}

const JOURNALED: u16 = 0x8000;
const NO_LINK: u32 = u32::MAX;

type Frame = Vec<(SymId, Option<Value>)>;

/// One list of arguments on its way to a function: the arguments of a call or of a
/// projection (expressions of a tree, `NONE` for a hole), or values that are already
/// evaluated.
enum Layer {
    None,
    Tree(Rc<Tree>, Range),
    Vals(Vec<Value>),
    Slots(Vec<Slot>),
}

#[derive(Clone)]
enum Arg {
    Code(Rc<Tree>, NodeId),
    Val(Value),
}

/// A program to run: borrowed (copied once, for error messages) or handed over (kept as that copy).
enum Prog<'a> {
    Text(&'a str),
    Owned(String),
}

pub struct Interp {
    pub syms: Interner,
    pub cfg: Config,
    globals: Vec<Option<Value>>,
    /// Per global: the scope it was created in (and the `JOURNALED` bit).
    gscope: Vec<u16>,
    /// The global scopes, oldest first. `.module(:m)` adds a module scope and `.module(0)` a
    /// plain one, as KlongPy pushes a context for each.
    scopes: Vec<Scope>,
    /// Names a module scope holds: (plain name, scope, qualified name), in creation order.
    exports: Vec<(SymId, u16, SymId)>,
    /// Per symbol: (generation, the symbol whose global a read of it yields). Only used once
    /// there is a module.
    link: Vec<Cell<(u32, u32)>>,
    generation: u32,
    /// The global bindings a load has replaced, for undoing a load that fails.
    journal: Option<Journal>,
    /// Stack address where the outermost `run` was entered, and how many runs are nested.
    stack_base: usize,
    running: u32,
    /// A system function answered "no value" (KlongPy's None): see `no_value`.
    void: bool,
    /// High-water mark of native stack used below `stack_base`, in bytes.
    pub max_stack_seen: usize,
    frames: Vec<Frame>,
    /// Per symbol: how many bindings of it are on the frame stack.
    shadow: Vec<u32>,
    hooks: Box<dyn Hooks>,
    depth: u32,
    ticks: u64,
    /// Emptied vectors kept for the next call: a call of a Klong function would otherwise allocate and free
    /// its argument frame and its list of projection layers (a Cortex-M0+ takes a few hundred cycles for each).
    frame_pool: Vec<Frame>,
    /// The function values made for a few tree nodes lately (a function literal's body is made into one at every
    /// call: an allocation). Each entry keeps its tree alive, so a tree's address cannot be taken for another's.
    body_cache: core::cell::RefCell<[Option<(usize, NodeId, Value)>; 8]>,
    body_next: Cell<u8>,
    layer_pool: Vec<Vec<Layer>>,
    module: Option<String>,
    /// High-water mark of `depth` (for sizing a board's stack).
    pub max_depth_seen: u32,
}

struct Journal {
    old: Vec<(SymId, Option<Value>)>,
    scopes: usize,
    exports: usize,
    module: Option<String>,
}

pub(crate) struct Chain<'a> {
    pub tree: &'a Rc<Tree>,
    pub verb: NodeId,
    pub verb_arity: u8,
    pub verb_val: Value,
    pub advs: &'a [u32],
    pub first_arity: u8,
}

impl Interp {
    pub fn new(cfg: Config, hooks: Box<dyn Hooks>) -> Interp {
        crate::value::set_max_nest(cfg.max_nest);
        Interp {
            syms: Interner::new(),
            cfg,
            globals: Vec::new(),
            gscope: Vec::new(),
            scopes: alloc::vec![Scope::Plain],
            exports: Vec::new(),
            link: Vec::new(),
            generation: 1,
            journal: None,
            stack_base: 0,
            running: 0,
            void: false,
            max_stack_seen: 0,
            frames: Vec::new(),
            shadow: Vec::new(),
            hooks,
            depth: 0,
            ticks: 0,
            frame_pool: Vec::new(),
            body_cache: core::cell::RefCell::new([None, None, None, None, None, None, None, None]),
            body_next: Cell::new(0),
            layer_pool: Vec::new(),
            module: None,
            max_depth_seen: 0,
        }
    }

    /// Make a system function available under `name` (`.p`, `.d`, ...). `arity` is the number
    /// of arguments, or `WILDCARD`.
    pub fn register(&mut self, name: &str, arity: u8, f: NativeFn) {
        let s = self.syms.intern(name);
        self.set_global(s, Value::Fn(Rc::new(Func::Native { name: s, arity, f })));
    }

    /// Define a global from the embedder (a system function, a system variable). Aborts if
    /// there is no memory for the table: for start-up only.
    pub fn set_global(&mut self, s: SymId, v: Value) {
        let i = s.0 as usize;
        if self.globals.len() <= i {
            self.globals.resize(i + 1, None);
        }
        if self.gscope.len() <= i {
            self.gscope.resize(i + 1, 0);
        }
        self.globals[i] = Some(v);
        self.generation = self.generation.wrapping_add(1);
    }

    pub fn global(&self, name: &str) -> Option<Value> {
        let s = self.syms.lookup(name)?;
        self.global_ref(s).cloned()
    }

    /// Define or replace a global by name (for the embedder: `.a`, `fullpath`).
    pub fn define_global(&mut self, name: &str, v: Value) -> Result<()> {
        let s = self.syms.intern(name);
        if let Some(e) = self.syms.failure() {
            return Err(e);
        }
        self.try_set_global(s, v)
    }

    pub fn format(&self, v: &Value) -> String {
        crate::fmt::to_string(&self.syms, v)
    }

    fn stack(&self) -> Option<(usize, usize)> {
        self.cfg.stack_limit.map(|l| (if self.running == 0 { stack_here() } else { self.stack_base }, l))
    }

    pub fn parse(&mut self, text: &str, name: &str) -> Result<Parsed> {
        let stack = self.stack();
        parse::parse(text, name, &mut self.syms, &mut self.module, self.cfg.strict, self.cfg.max_depth, stack)
    }

    fn parse_prog(&mut self, prog: Prog, name: &str) -> Result<Parsed> {
        match prog {
            Prog::Text(text) => self.parse(text, name),
            Prog::Owned(text) => {
                let stack = self.stack();
                parse::parse_owned(text, name, &mut self.syms, &mut self.module, self.cfg.strict, self.cfg.max_depth, stack)
            }
        }
    }

    /// Parse and run a program: every statement in order; the value of the last one, or `None`
    /// for a program without statements. May be called from a system function (`.l`, `.E`):
    /// the program then runs where the call was made, and sees the locals of its callers.
    pub fn run(&mut self, text: &str, name: &str) -> Result<Option<Value>> {
        self.run_prog(Prog::Text(text), name)
    }

    fn run_prog(&mut self, prog: Prog, name: &str) -> Result<Option<Value>> {
        if self.running == 0 {
            // the nesting bound is one value for the process (see `value::set_max_nest`): the
            // interpreter that runs asserts its own, whichever was created last
            crate::value::set_max_nest(self.cfg.max_nest);
            self.stack_base = stack_here();
            // a failure deferred by something outside an evaluation is not this program's
            let _ = crate::value::take_deferred();
        }
        self.running += 1;
        let r = self.run_parsed(prog, name);
        self.running -= 1;
        if self.running == 0 {
            // between two programs the pools of vectors made for speed (and the recent function bodies, each of
            // which keeps its parse tree) give their memory back: a board has no memory to keep for them
            self.release_caches();
        }
        r
    }

    fn run_parsed(&mut self, prog: Prog, name: &str) -> Result<Option<Value>> {
        let p = self.parse_prog(prog, name)?;
        let frames = self.frames.len();
        let depth = self.depth;
        let mut last = None;
        for &s in &p.stmts {
            self.void = false;
            match self.eval(&p.tree, s, true) {
                Ok(v) => last = Some(v),
                Err(e) => {
                    // unwind whatever the failed statement left behind
                    while self.frames.len() > frames {
                        self.pop_frame();
                    }
                    self.depth = depth;
                    return Err(e);
                }
            }
        }
        if self.void && matches!(last, Some(Value::Undefined)) {
            last = None;
        }
        self.void = false;
        Ok(last)
    }

    /// What a system function returns when it has nothing to return (`.module`, `.cc`, a load
    /// of an empty file). It is `:undefined` inside an expression; a program whose last
    /// statement ends this way has no value, and nothing is printed for it.
    pub fn no_value(&mut self) -> Value {
        self.void = true;
        Value::Undefined
    }

    /// Read one data object (a number, a string, a char, a symbol, a list or a dictionary
    /// literal) from the start of `text`: the number of bytes it took and the value, or `None`
    /// when there is nothing but blanks. For `.r` and `.rs`.
    pub fn read_datum(&mut self, text: &str) -> Result<Option<(usize, Value)>> {
        let stack = self.stack();
        parse::read_datum(text, &mut self.syms, &mut self.module, self.cfg.max_depth, stack)
    }

    /// Run a program at global scope, whatever function is running (`.E`): the frames of the
    /// callers are set aside, so a definition lands in the globals.
    pub fn run_global(&mut self, text: &str, name: &str) -> Result<Option<Value>> {
        let frames = core::mem::take(&mut self.frames);
        let shadow = core::mem::take(&mut self.shadow);
        let r = self.run(text, name);
        self.frames = frames;
        self.shadow = shadow;
        r
    }

    /// Run a program as a load (`.l`): when it fails, the global definitions it made or
    /// replaced, and the modules it opened, are undone, so that a failed load leaves nothing
    /// half-defined. (What it did to a dictionary or wrote to a channel stays done.) Loads
    /// nest; the outermost one decides.
    pub fn load(&mut self, text: &str, name: &str) -> Result<Option<Value>> {
        self.load_prog(Prog::Text(text), name)
    }

    /// `load` of a text that is handed over: it is kept as the copy that error messages are cut from, not copied (a library is not held twice).
    pub fn load_owned(&mut self, text: String, name: &str) -> Result<Option<Value>> {
        self.load_prog(Prog::Owned(text), name)
    }

    fn load_prog(&mut self, prog: Prog, name: &str) -> Result<Option<Value>> {
        if self.journal.is_some() {
            return self.run_prog(prog, name);
        }
        for g in self.gscope.iter_mut() {
            *g &= !JOURNALED;
        }
        self.journal = Some(Journal { old: Vec::new(), scopes: self.scopes.len(), exports: self.exports.len(), module: self.module.clone() });
        let r = self.run_prog(prog, name);
        if let Some(j) = self.journal.take() {
            if r.is_err() {
                for (s, old) in j.old.into_iter().rev() {
                    if let Some(slot) = self.globals.get_mut(s.0 as usize) {
                        *slot = old;
                    }
                }
                self.scopes.truncate(j.scopes.max(1));
                self.exports.truncate(j.exports);
                self.module = j.module;
                self.generation = self.generation.wrapping_add(1);
            }
        }
        r
    }

    // ----- modules ---------------------------------------------------------------------------

    /// `.module(:name)` at run time: a module scope. The names assigned at top level while
    /// it is the newest scope belong to it (the parser has already qualified them).
    pub fn module_start(&mut self, name: &str) -> Result<()> {
        let s = self.syms.intern(name);
        if let Some(e) = self.syms.failure() {
            return Err(e);
        }
        self.push_scope(Scope::Module(s))
    }

    /// `.module(0)`: a new plain scope on top of the module.
    pub fn module_stop(&mut self) -> Result<()> {
        self.push_scope(Scope::Plain)
    }

    fn push_scope(&mut self, sc: Scope) -> Result<()> {
        if self.scopes.len() >= (JOURNALED - 1) as usize {
            return Err(Error::range("too many modules"));
        }
        self.scopes.try_reserve(1).map_err(|_| Error::memory())?;
        try_push(&mut self.scopes, sc)?;
        self.generation = self.generation.wrapping_add(1);
        Ok(())
    }

    /// The module the parser is in (`.module` switches it while it parses).
    pub fn parse_module(&self) -> Option<&str> {
        self.module.as_deref()
    }

    #[inline]
    fn has_global(&self, s: SymId) -> bool {
        matches!(self.globals.get(s.0 as usize), Some(Some(_)))
    }

    /// The global a read of `s` yields. Without modules that is its own; with them, KlongPy's
    /// walk through the scopes from the newest: a plain scope holds a name as written; a
    /// module scope answers a plain name with its own variable of that name (`unit` finds
    /// `unit` of module `geo`); and a module's name that the module does not define is looked
    /// up, as a plain name, in the scopes older than the module.
    #[inline]
    #[cfg_attr(feature = "ram-hot", allow(unsafe_code), link_section = ".time_critical.klong_global_ref")]
    fn global_ref(&self, s: SymId) -> Option<&Value> {
        if self.scopes.len() <= 1 {
            return self.globals.get(s.0 as usize)?.as_ref();
        }
        let t = self.resolve_global(s)?;
        self.globals.get(t.0 as usize)?.as_ref()
    }

    #[inline(never)]
    fn resolve_global(&self, s: SymId) -> Option<SymId> {
        let cell = self.link.get(s.0 as usize);
        if let Some(c) = cell {
            let (g, t) = c.get();
            if g == self.generation {
                return if t == NO_LINK { None } else { Some(SymId(t)) };
            }
        }
        let t = self.resolve_uncached(s);
        if let Some(c) = cell {
            c.set((self.generation, t.map(|t| t.0).unwrap_or(NO_LINK)));
        }
        t
    }

    fn resolve_uncached(&self, s: SymId) -> Option<SymId> {
        match self.syms.name(s).split_once('`') {
            None => self.resolve_plain(s, usize::MAX),
            Some((base, module)) => {
                if self.has_global(s) {
                    return Some(s);
                }
                let limit = self.scopes.iter().rposition(|sc| matches!(sc, Scope::Module(n) if self.syms.name(*n) == module))?;
                self.resolve_plain(self.syms.lookup(base)?, limit)
            }
        }
    }

    /// The newest scope older than `limit` that answers the plain name `b`.
    fn resolve_plain(&self, b: SymId, limit: usize) -> Option<SymId> {
        let mut best: Option<(usize, SymId)> = None;
        if self.has_global(b) {
            let sc = (self.gscope.get(b.0 as usize).copied().unwrap_or(0) & !JOURNALED) as usize;
            if sc < limit {
                best = Some((sc, b));
            }
        }
        for &(base, sc, q) in &self.exports {
            let sc = sc as usize;
            if base == b && sc < limit && best.map(|(bs, _)| sc > bs).unwrap_or(true) && self.has_global(q) {
                best = Some((sc, q));
            }
        }
        best.map(|(_, t)| t)
    }

    // ----- ticks ---------------------------------------------------------------------------

    #[inline]
    pub(crate) fn tick(&mut self) -> Result<()> {
        self.ticks += 1;
        if self.ticks & self.cfg.tick_mask as u64 == 0 {
            return self.slow_tick();
        }
        // Every call and every adverb iteration allocates a little through paths that cannot
        // fail (frames, reference counts); a long run of them would use up the reserve between
        // two slow ticks. Ask every time: it is two loads.
        if let Some(probe) = self.cfg.mem_probe {
            if probe() {
                return Err(Error::memory());
            }
        }
        Ok(())
    }

    #[cold]
    fn slow_tick(&mut self) -> Result<()> {
        if self.hooks.tick() {
            return Err(Error::interrupt());
        }
        if let Some(flag) = self.cfg.interrupt {
            // load and store, not swap: targets without compare-and-swap (thumbv6m, riscv32imc)
            // have no atomic read-modify-write
            if flag.load(Ordering::Relaxed) {
                flag.store(false, Ordering::Relaxed);
                return Err(Error::interrupt());
            }
        }
        if let Some(limit) = self.cfg.step_limit {
            if self.ticks > limit {
                return Err(Error::interrupt());
            }
        }
        if let Some(probe) = self.cfg.mem_probe {
            if probe() {
                return Err(Error::memory());
            }
        }
        Ok(())
    }

    // ----- variables: dynamic scope ----------------------------------------------------------

    fn grow_shadow(&mut self, i: usize) -> Result<()> {
        if self.shadow.len() <= i {
            let more = i + 1 - self.shadow.len();
            self.shadow.try_reserve(more).map_err(|_| Error::memory())?;
            self.shadow.resize(i + 1, 0);
        }
        Ok(())
    }

    #[cfg_attr(feature = "ram-hot", allow(unsafe_code), link_section = ".time_critical.klong_push_frame")]

    fn push_frame(&mut self, f: Frame) -> Result<()> {
        self.frames.try_reserve(1).map_err(|_| Error::memory())?;
        let top = f.iter().map(|(s, _)| s.0 as usize).max().unwrap_or(0);
        self.grow_shadow(top)?;
        for (s, _) in &f {
            self.shadow[s.0 as usize] += 1;
        }
        try_push(&mut self.frames, f)?;
        Ok(())
    }

    /// Create or replace the global `s` itself (no module rule: that is for reads).
    fn try_set_global(&mut self, s: SymId, v: Value) -> Result<()> {
        let _tag = crate::htag::Scope::new(crate::htag::GLOBALS);
        let i = s.0 as usize;
        let n = self.syms.len().max(i + 1);
        if self.globals.len() <= i {
            self.globals.try_reserve(n - self.globals.len()).map_err(|_| Error::memory())?;
            self.globals.resize(n, None);
        }
        if self.gscope.len() <= i {
            self.gscope.try_reserve(n - self.gscope.len()).map_err(|_| Error::memory())?;
            self.gscope.resize(n, 0);
        }
        if self.scopes.len() > 1 && self.link.len() <= i {
            self.link.try_reserve(n - self.link.len()).map_err(|_| Error::memory())?;
            self.link.resize(n, Cell::new((0, NO_LINK)));
        }
        let existed = self.globals[i].is_some();
        if let Some(j) = &mut self.journal {
            if self.gscope[i] & JOURNALED == 0 {
                j.old.try_reserve(1).map_err(|_| Error::memory())?;
                try_push(&mut j.old, (s, self.globals[i].take()))?;
                self.gscope[i] |= JOURNALED;
            }
        }
        if !existed {
            let top = self.scopes.len() - 1;
            self.gscope[i] = top as u16 | (self.gscope[i] & JOURNALED);
            if let Scope::Module(_) = self.scopes[top] {
                if let Some((base, _)) = self.syms.name(s).split_once('`') {
                    // (the name is copied: interning borrows the table)
                    let mut buf = String::new();
                    buf.try_reserve(base.len()).map_err(|_| Error::memory())?;
                    buf.push_str(base);
                    let b = self.syms.intern(&buf);
                    if let Some(e) = self.syms.failure() {
                        return Err(e);
                    }
                    if !self.exports.iter().any(|&(eb, sc, _)| eb == b && sc as usize == top) {
                        self.exports.try_reserve(1).map_err(|_| Error::memory())?;
                        try_push(&mut self.exports, (b, top as u16, s))?;
                    }
                }
            }
            self.generation = self.generation.wrapping_add(1);
        }
        self.globals[i] = Some(v);
        Ok(())
    }

    #[cfg_attr(feature = "ram-hot", allow(unsafe_code), link_section = ".time_critical.klong_unshadow")]

    fn unshadow(&mut self, f: &Frame) {
        for (s, _) in f {
            if let Some(c) = self.shadow.get_mut(s.0 as usize) {
                *c = c.saturating_sub(1);
            }
        }
    }

    #[cfg_attr(feature = "ram-hot", allow(unsafe_code), link_section = ".time_critical.klong_pop_frame")]

    fn pop_frame(&mut self) {
        if let Some(mut f) = self.frames.pop() {
            self.unshadow(&f);
            f.clear();
            if self.frame_pool.len() < 32 && f.capacity() <= 16 {
                self.frame_pool.push(f);
            }
        }
    }

    /// An empty frame with room for `n` entries, from the pool when there is one.
    #[inline]
    #[cfg_attr(feature = "ram-hot", allow(unsafe_code), link_section = ".time_critical.klong_new_frame")]
    fn new_frame(&mut self, n: usize) -> Result<Frame> {
        match self.frame_pool.pop() {
            Some(mut f) => {
                f.try_reserve(n).map_err(|_| Error::memory())?;
                Ok(f)
            }
            None => try_vec(n),
        }
    }

    /// The value of a name: the innermost frame that binds it, else the global.
    #[cfg_attr(feature = "ram-hot", allow(unsafe_code), link_section = ".time_critical.klong_lookup")]
    pub fn lookup(&self, s: SymId) -> Result<Option<Value>> {
        if self.shadow.get(s.0 as usize).copied().unwrap_or(0) > 0 {
            for f in self.frames.iter().rev() {
                for (k, v) in f.iter() {
                    if *k == s {
                        return match v {
                            Some(v) => Ok(Some(v.clone())),
                            None => Err(Error::name(crate::msg!("local variable {} is read before it is assigned", self.syms.name(s)))),
                        };
                    }
                }
            }
        }
        Ok(self.global_ref(s).cloned())
    }

    /// `name::value` (KlongPy `KlongContext.__setitem__`): assign where the name lives; a new
    /// name is created in the innermost frame.
    pub fn assign(&mut self, s: SymId, v: Value) -> Result<()> {
        if !sym::is_arg(s) {
            if self.shadow.get(s.0 as usize).copied().unwrap_or(0) > 0 {
                for f in self.frames.iter_mut().rev() {
                    for (k, slot) in f.iter_mut() {
                        if *k == s {
                            *slot = Some(v);
                            return Ok(());
                        }
                    }
                }
            }
            if self.has_global(s) {
                if self.journal.is_some() {
                    return self.try_set_global(s, v);
                }
                self.globals[s.0 as usize] = Some(v);
                return Ok(());
            }
        }
        // x, y and z are the arguments of the running function and are assignable like its
        // locals (ticket 033); they are always bound in its own frame
        if self.cfg.strict >= 1 && !self.frames.is_empty() && !sym::is_arg(s) {
            let n = self.syms.name(s);
            return Err(Error::name(crate::msg!("undefined variable: {}", n))
                .with_hint(crate::msg!("declare it as a local in the parameter list, {{[{}]; ...}}, or define the global before the function is called", n)));
        }
        if self.frames.is_empty() {
            return self.try_set_global(s, v);
        }
        let i = s.0 as usize;
        self.grow_shadow(i)?;
        if let Some(f) = self.frames.last_mut() {
            for (k, slot) in f.iter_mut() {
                if *k == s {
                    *slot = Some(v);
                    return Ok(());
                }
            }
            f.try_reserve(1).map_err(|_| Error::memory())?;
            try_push(f, (s, Some(v)))?;
            self.shadow[i] += 1;
        }
        Ok(())
    }

    fn is_defined(&self, s: SymId) -> bool {
        self.shadow.get(s.0 as usize).copied().unwrap_or(0) > 0 || self.global_ref(s).is_some()
    }

    /// The value of a name that is not defined.
    #[inline(never)]
    fn undefined(&mut self, s: SymId) -> Result<Value> {
        let level = self.cfg.strict;
        if level == 2 || (level == 1 && !self.frames.is_empty()) {
            return Err(Error::name(crate::msg!("undefined: {}", self.syms.name(s))));
        }
        Ok(Value::Sym(s))
    }

    // ----- evaluation --------------------------------------------------------------------------

    /// A tree node as a value in function position: a leaf is its value, a name is the symbol,
    /// anything else is the node itself.
    #[cfg_attr(feature = "ram-hot", allow(unsafe_code), link_section = ".time_critical.klong_node_value")]
    fn node_value(&self, tree: &Rc<Tree>, id: NodeId) -> Value {
        match tree.node(id) {
            Node::Lit(v) => v.clone(),
            Node::Sym(s) | Node::QSym(s) => Value::Sym(*s),
            Node::OpVal { op, .. } => Value::Prim(*op),
            _ => {
                let key = Rc::as_ptr(tree) as usize;
                if let Ok(mut c) = self.body_cache.try_borrow_mut() {
                    if let Some((_, _, v)) = c.iter().flatten().find(|(t, n, _)| *t == key && *n == id) {
                        return v.clone();
                    }
                    let v = Value::Fn(Rc::new(Func::Code { tree: tree.clone(), node: id }));
                    let k = self.body_next.get() as usize % 8;
                    self.body_next.set(self.body_next.get().wrapping_add(1));
                    c[k] = Some((key, id, v.clone()));
                    return v;
                }
                Value::Fn(Rc::new(Func::Code { tree: tree.clone(), node: id }))
            }
        }
    }

    /// KlongPy `isinstance(v, (KGFn, KGLambda))`.
    fn is_fn_value(v: &Value) -> bool {
        match v {
            Value::Fn(f) => match &**f {
                Func::Code { tree, node } => tree.is_kgfn(*node),
                _ => true,
            },
            _ => false,
        }
    }

    /// Let go of what the evaluator keeps for speed: the recent function bodies (each holds its whole parse tree
    /// alive, so the tree of a library that was loaded and dropped stays) and the pool of argument layers.
    pub fn release_caches(&mut self) {
        if let Ok(mut c) = self.body_cache.try_borrow_mut() {
            *c = [None, None, None, None, None, None, None, None];
        }
        self.layer_pool.clear();
        self.layer_pool.shrink_to_fit();
        self.frame_pool.clear();
        self.frame_pool.shrink_to_fit();
        // the tables indexed by symbol grow by doubling while a library loads: what a table has beyond its symbols is kept for nothing
        // (and the block it grew into is the last, the one that stays)
        self.globals.shrink_to_fit();
        self.gscope.shrink_to_fit();
        self.link.shrink_to_fit();
        self.syms.shrink();
        if self.frames.is_empty() {
            self.frames = Vec::new();
        }
    }

    /// KlongPy `_CALLABLE`: what a call may be made on.
    fn is_callable(v: &Value) -> bool {
        matches!(v, Value::Fn(_) | Value::Prim(_))
    }

    /// KlongPy `_names_function`: `v`, the value found for the name `name` being called, is a symbol that
    /// leads to a function or to nothing defined (a call through a variable holding a symbol: the call then
    /// fails as an undefined function, as it always has). False for any other value, and for a symbol
    /// that names itself (what an undefined name is at strict level 0).
    #[inline(never)]
    fn names_function(&self, v: &Value, name: SymId) -> Result<bool> {
        let mut seen = [name; 4];
        let mut n = 1;
        let mut v = v.clone();
        for _ in 0..3 {
            let s = match v {
                Value::Sym(s) if !sym::is_arg(s) && !seen[..n].contains(&s) => s,
                _ => return Ok(false),
            };
            seen[n] = s;
            n += 1;
            match self.lookup(s)? {
                // (an undefined name is its own symbol at strict level 0, which names nothing)
                None if self.cfg.strict == 0 => v = Value::Sym(s),
                None => return Ok(true),
                Some(w) => {
                    if Self::is_callable(&w) {
                        return Ok(true);
                    }
                    v = w;
                }
            }
        }
        Ok(false)
    }

    /// KlongPy `_not_a_function`: the error for a call of `name`, whose value is not a function.
    #[inline(never)]
    #[cold]
    fn not_a_function(&self, name: SymId, v: &Value) -> Error {
        let shown = self.syms.name(name).split('`').next().unwrap_or("");
        let what = match v {
            Value::Int(_) => crate::msg!("an integer"),
            Value::Flt(x) if !x.is_finite() => crate::msg!(":undefined"),
            Value::Flt(_) => crate::msg!("a float"),
            Value::Char(_) => crate::msg!("a char"),
            Value::Sym(_) => crate::msg!("a symbol"),
            Value::Undefined => crate::msg!(":undefined"),
            Value::Str(_) => crate::msg!("a string"),
            Value::Dict(_) => crate::msg!("a dict"),
            Value::Ext(_) => crate::msg!("a channel"),
            Value::Fn(_) | Value::Prim(_) => crate::msg!("a function"),
            Value::Ints(a) => Self::list_kind(a.rank(), a.len(), a.row_dims()),
            Value::Flts(a) => Self::list_kind(a.rank(), a.len(), a.row_dims()),
            Value::List(l) => Self::list_kind(1, l.len(), &[]),
        };
        let mut hint = String::from("a function is written {...}");
        if matches!(v, Value::Ints(_) | Value::Flts(_) | Value::List(_) | Value::Str(_) | Value::Dict(_)) {
            hint.push_str(&crate::msg!("; to index, write {}@i", shown));
        }
        Error::typ(crate::msg!("cannot call {}: it is {}, not a function", shown, what)).with_hint(hint)
    }

    fn list_kind(rank: usize, len: usize, row: &[u32]) -> String {
        if rank <= 1 {
            return if len == 0 { crate::msg!("an empty list") } else { crate::msg!("a list of {}", len) };
        }
        let mut dims = crate::msg!("{}", len);
        for d in row {
            dims.push_str(&crate::msg!("x{}", d));
        }
        if rank == 2 { crate::msg!("a {} matrix", dims) } else { crate::msg!("a {} array", dims) }
    }

    /// Evaluate a node. With `call` set, a function node is invoked (KlongPy `call`); without
    /// it a function literal or projection is the value (`eval`).
    #[inline(always)]
    pub fn eval(&mut self, tree: &Rc<Tree>, id: NodeId, call: bool) -> Result<Value> {
        if !Tree::is_real(id) {
            return Err(Error::syntax("missing operand"));
        }
        self.depth += 1;
        if self.depth > self.max_depth_seen {
            self.max_depth_seen = self.depth;
        }
        if self.depth > self.cfg.max_depth {
            self.depth -= 1;
            return Err(Error::range("expression nested too deeply"));
        }
        if let Err(e) = self.check_stack() {
            self.depth -= 1;
            return Err(e);
        }
        let mut r = self.eval_node(tree, id, call);
        self.depth -= 1;
        // a failure met where no error could be returned (value.rs, `defer_memory`)
        if let Some(e) = crate::value::take_deferred() {
            if r.is_ok() {
                r = Err(e);
            }
        }
        match r {
            Ok(v) => Ok(v),
            Err(mut e) => {
                if e.span.is_none() {
                    if let Some(sp) = tree.span(id) {
                        e.span = Some(sp);
                        e.source = tree.source.clone();
                    }
                }
                Err(e)
            }
        }
    }

    #[inline(never)]
    #[cfg_attr(feature = "ram-hot", allow(unsafe_code), link_section = ".time_critical.klong_eval_node")]
    fn eval_node(&mut self, tree: &Rc<Tree>, id: NodeId, call: bool) -> Result<Value> {
        match tree.node(id) {
            Node::Lit(v) => Ok(v.clone()),
            Node::Sym(s) => match self.lookup(*s)? {
                Some(v) => Ok(v),
                None => self.undefined(*s),
            },
            Node::QSym(s) => self.quoted(*s),
            Node::OpVal { op, .. } => Ok(Value::Prim(*op)),
            Node::Monad { op, arg, missing } => self.eval_monad(tree, *op, *arg, *missing),
            Node::Dyad { op, l, r, missing } => {
                let (op, l, r, missing) = (*op, *l, *r, *missing);
                let res = self.eval_dyad(tree, op, l, r);
                if missing {
                    return res.map_err(|_| Error::syntax(crate::msg!("missing right operand for {}", self.syms.name(op))));
                }
                res
            }
            Node::Fn { call: is_call, has_args, a, .. } => {
                // A call written with an empty argument, f(1;), is a projection, made now
                // (ticket 033). A callee that is not defined yet leaves the call as written:
                // it is looked up when the projection is completed.
                let now = call
                    || *is_call
                    || (*has_args
                        && match tree.node(*a) {
                            Node::Sym(s) => self.is_defined(*s),
                            _ => true,
                        });
                if now {
                    self.eval_call(tree, id)
                } else {
                    Ok(Value::Fn(Rc::new(Func::Code { tree: tree.clone(), node: id })))
                }
            }
            Node::Adv { .. } => self.eval_adv(tree, id),
            Node::Cond(c, t, e) => {
                let q = self.eval(tree, *c, true)?;
                self.eval(tree, if q.truthy() { *t } else { *e }, true)
            }
            Node::EList(r) => self.eval_elist(tree, *r),
            Node::EDict(r) => self.eval_edict(tree, *r),
            Node::DictLit(v) | Node::CopyList(v) => ops::deep_copy(v),
            Node::Seq(r) => self.eval_seq(tree, *r),
            Node::Body { stmts, .. } => self.eval_seq(tree, *stmts),
            Node::Destructure(_) => Err(Error::internal("a destructuring target was evaluated")),
        }
    }

    /// The arms of `eval_node` that are not on the way of a call, apart, so that its stack frame (one per
    /// level of Klong recursion, on a chip with a stack of tens of KB) is not made for them.
    #[inline(never)]
    fn eval_monad(&mut self, tree: &Rc<Tree>, op: SymId, arg: NodeId, missing: bool) -> Result<Value> {
        let def = self.op_def(op)?;
        let f = def.monad.ok_or_else(|| self.unknown_op(op))?;
        let r = match self.eval(tree, arg, false) {
            Ok(x) => f(self, x),
            Err(e) => Err(e),
        };
        if missing {
            return r.map_err(|_| Error::syntax(crate::msg!("missing operand for {}", self.syms.name(op))));
        }
        r
    }

    #[inline(never)]
    fn eval_elist(&mut self, tree: &Rc<Tree>, r: Range) -> Result<Value> {
        let mut out = try_vec(r.len as usize)?;
        for k in 0..r.len {
            let q = NodeId(tree.ids(r)[k as usize]);
            let pushed = self.eval(tree, q, true)?;
            try_push(&mut out, pushed)?;
        }
        Value::from_items(out)
    }

    #[inline(never)]
    fn eval_edict(&mut self, tree: &Rc<Tree>, r: Range) -> Result<Value> {
        let mut d = Dict::default();
        let ids = tree.ids(r);
        let mut k = 0;
        while k + 1 < ids.len() {
            let key = self.eval(tree, NodeId(ids[k]), true)?;
            let val = self.eval(tree, NodeId(ids[k + 1]), true)?;
            d.set(key, val)?;
            k += 2;
        }
        Ok(Value::dict(d))
    }

    #[inline(never)]
    #[cfg_attr(feature = "ram-hot", allow(unsafe_code), link_section = ".time_critical.klong_eval_seq")]
    fn eval_seq(&mut self, tree: &Rc<Tree>, r: Range) -> Result<Value> {
        let mut last = Value::empty();
        for k in 0..r.len {
            let q = NodeId(tree.ids(r)[k as usize]);
            last = self.eval(tree, q, true)?;
        }
        Ok(last)
    }

    fn unknown_op(&self, op: SymId) -> Error {
        Error::syntax(crate::msg!("unknown operator {}", self.syms.name(op)))
    }

    #[cfg_attr(feature = "ram-hot", allow(unsafe_code), link_section = ".time_critical.klong_op_def")]

    fn op_def(&self, op: SymId) -> Result<&'static ops::OpDef> {
        ops::by_id(op).ok_or_else(|| self.unknown_op(op))
    }

    pub fn monad(&mut self, op: SymId, x: Value) -> Result<Value> {
        if self.syms.name(op) == ops::TRACK {
            return Ok(x);
        }
        let f = self.op_def(op)?.monad.ok_or_else(|| self.unknown_op(op))?;
        f(self, x)
    }

    pub fn dyad(&mut self, op: SymId, x: Value, y: Value) -> Result<Value> {
        let f = self.op_def(op)?.dyad.ok_or_else(|| self.unknown_op(op))?;
        f(self, x, y)
    }

    #[inline(never)]
    #[cfg_attr(feature = "ram-hot", allow(unsafe_code), link_section = ".time_critical.klong_eval_dyad")]
    fn eval_dyad(&mut self, tree: &Rc<Tree>, op: SymId, l: NodeId, r: NodeId) -> Result<Value> {
        let def = self.op_def(op)?;
        let f = def.dyad.ok_or_else(|| self.unknown_op(op))?;
        // right to left
        let y = self.eval(tree, r, false)?;
        if def.glyph == "::" {
            return self.define(tree, l, y);
        }
        let x = self.eval(tree, l, false)?;
        match def.atomic {
            Some(a) => ops::atomic(&self.syms, a, &x, &y),
            None => f(self, x, y),
        }
    }

    /// `l::v`. The left side is not evaluated.
    #[inline(never)]
    fn define(&mut self, tree: &Rc<Tree>, l: NodeId, v: Value) -> Result<Value> {
        if !Tree::is_real(l) {
            return Ok(v);
        }
        match tree.node(l) {
            Node::Sym(s) | Node::QSym(s) => {
                self.assign(*s, v.clone())?;
                Ok(v)
            }
            Node::Destructure(names) => {
                let names = tree.ids(*names);
                let n = names.len();
                if !v.is_iterable() {
                    return Err(Error::typ(crate::msg!("cannot destructure {} into {} name{}", v.describe(), n, if n != 1 { "s" } else { "" }))
                        .with_hint("the right side of [a b c]::value must be a list or a string"));
                }
                let m = v.len().unwrap_or(0);
                if m != n {
                    return Err(Error::shape(crate::msg!(
                        "cannot destructure {} value{} into {} name{}",
                        m,
                        if m != 1 { "s" } else { "" },
                        n,
                        if n != 1 { "s" } else { "" }
                    ))
                    .with_hint("the list needs exactly one element for each name"));
                }
                for (i, &name) in names.iter().enumerate() {
                    self.assign(SymId(name), v.item(i))?;
                }
                Ok(v)
            }
            // "1::5": accepted at strict level 0, binds nothing (legacy)
            _ => Ok(v),
        }
    }

    /// A quoted symbol `:foo`. Strict levels 0 and 1 keep the legacy rule: the variable's
    /// value when one is defined, the symbol otherwise. Level 2: always the symbol.
    #[inline(never)]
    fn quoted(&mut self, s: SymId) -> Result<Value> {
        // :undefined is the undefined value at every level (ticket 033)
        if self.syms.name(s) == "undefined" {
            return Ok(Value::Undefined);
        }
        if self.cfg.strict == 2 {
            return Ok(Value::Sym(s));
        }
        Ok(match self.lookup(s) {
            Ok(Some(v)) => v,
            _ => Value::Sym(s),
        })
    }

    // ----- calls ---------------------------------------------------------------------------------

    /// One pass of KlongPy's `_resolve_fn`: a name becomes the function it names; a function
    /// node becomes its callee, and the arguments a projection holds join the layers.
    #[inline(never)]
    #[cfg_attr(feature = "ram-hot", allow(unsafe_code), link_section = ".time_critical.klong_resolve")]
    fn resolve(&mut self, f: &mut Value, layers: &mut Vec<Layer>, arity: &mut u8, node: &mut bool, calling: bool) -> Result<()> {
        if let Value::Sym(s) = *f {
            match self.lookup(s)? {
                Some(v) => {
                    // a call of a name whose value is not a function (KlongPy `_resolve_fn`)
                    if calling && !*node && s != sym::DOT_F && !Self::is_callable(&v) && !self.names_function(&v, s)? {
                        return Err(self.not_a_function(s, &v));
                    }
                    if Self::is_fn_value(&v) || !sym::is_arg(s) {
                        *f = v;
                        *node = false;
                    } else {
                        // x, y or z holds a value that is not a function. If it is a symbol
                        // naming a function, call the function it names.
                        let mut g = None;
                        if let Value::Sym(t) = v {
                            if !sym::is_arg(t) {
                                if let Ok(Some(w)) = self.lookup(t) {
                                    if Self::is_fn_value(&w) {
                                        g = Some(w);
                                    }
                                }
                            }
                        }
                        match g {
                            Some(w) => {
                                *f = w;
                                *node = false;
                            }
                            None => {
                                if calling && !*node && !Self::is_callable(&v) {
                                    return Err(self.not_a_function(s, &v));
                                }
                                return Ok(());
                            }
                        }
                    }
                }
                None => {
                    if !sym::is_arg(s) {
                        let name = self.syms.name(s);
                        let e = Error::name(crate::msg!("undefined function: {}", name));
                        // a system function this build does not have (native-runtime.md section 9)
                        return Err(if name.starts_with('.') { e.with_hint(crate::msg!("{} is not available on this build", name)) } else { e });
                    }
                }
            }
        }
        if *arity > 0 {
            let mut next = None;
            if let Value::Fn(func) = &*f {
                match &**func {
                    Func::Code { tree, node: id } => {
                        if let Node::Fn { a, args, has_args, arity: fa, .. } = tree.node(*id) {
                            if !*has_args {
                                next = Some((self.node_value(tree, *a), *fa));
                            } else if tree.ids(*args).contains(&NONE.0) {
                                try_push(layers, Layer::Tree(tree.clone(), *args))?;
                                next = Some((self.node_value(tree, *a), *fa));
                            }
                        }
                    }
                    Func::Proj { f: callee, args, .. } => {
                        if args.iter().any(|q| matches!(q, Slot::Hole)) {
                            let mut held = try_vec(args.len())?;
                            held.extend(args.iter().cloned());
                            try_push(layers, Layer::Slots(held))?;
                            next = Some((callee.clone(), args.len().min(255) as u8));
                        }
                    }
                    Func::Native { .. } => {}
                }
            }
            if let Some((g, fa)) = next {
                *f = g;
                *arity = fa;
                *node = true;
            }
        }
        Ok(())
    }

    /// (The lists are as long as a call has arguments, which the program decides.)
    fn layer_args(layer: Layer) -> Result<Vec<Option<Arg>>> {
        Ok(match layer {
            Layer::None => Vec::new(),
            Layer::Tree(tree, r) => {
                let ids = tree.ids(r);
                let mut out = try_vec(ids.len())?;
                out.extend(ids.iter().map(|&q| if q == NONE.0 { None } else { Some(Arg::Code(tree.clone(), NodeId(q))) }));
                out
            }
            Layer::Vals(v) => {
                let mut out = try_vec(v.len())?;
                out.extend(v.into_iter().map(|x| Some(Arg::Val(x))));
                out
            }
            Layer::Slots(v) => {
                let mut out = try_vec(v.len())?;
                out.extend(v.into_iter().map(|q| match q {
                    Slot::Hole => None,
                    Slot::Val(x) => Some(Arg::Val(x)),
                    Slot::Code(t, id) => Some(Arg::Code(t, id)),
                }));
                out
            }
        })
    }

    /// KlongPy `merge_projections`: the holes of the innermost projection are filled, in
    /// order, from the argument lists that came after it. `layers` is innermost first.
    fn merge(layers: Vec<Vec<Option<Arg>>>) -> Vec<Option<Arg>> {
        let mut it = layers.into_iter();
        let mut sparse = match it.next() {
            Some(s) => s,
            None => return Vec::new(),
        };
        if !sparse.iter().any(|q| q.is_none()) {
            return sparse;
        }
        // Each later layer is the argument list of a call of the projection the layers before
        // it make: its arguments go to the empty slots in order, starting at the first empty
        // slot, and an argument it leaves empty leaves that slot empty.
        for fa in it {
            let mut j = 0;
            for slot in sparse.iter_mut() {
                if j >= fa.len() {
                    break;
                }
                if slot.is_none() {
                    *slot = fa[j].clone();
                    j += 1;
                }
            }
        }
        sparse
    }

    /// Strict level 2 (and functions with named parameters at every level): a call of a
    /// projection must not supply more arguments than the projection has holes.
    fn check_projection(&self, callee: &Value, layers: &[Vec<Option<Arg>>]) -> Result<()> {
        let mut sparse: Vec<bool> = try_vec(layers.first().map(|l| l.len()).unwrap_or(0))?;
        if let Some(l) = layers.first() {
            sparse.extend(l.iter().map(|q| q.is_some()));
        }
        for fa in layers.iter().skip(1) {
            let mut holes = 0;
            let mut j = 0;
            for slot in sparse.iter_mut() {
                if !*slot {
                    holes += 1;
                    if j < fa.len() {
                        *slot = fa[j].is_some();
                        j += 1;
                    }
                }
            }
            if fa[j.min(fa.len())..].iter().any(|q| q.is_some()) {
                return Err(self.too_many(callee, holes, fa.iter().filter(|q| q.is_some()).count()));
            }
        }
        Ok(())
    }

    fn too_many(&self, callee: &Value, takes: usize, given: usize) -> Error {
        let name = match callee {
            Value::Sym(s) if !sym::is_arg(*s) && *s != sym::DOT_F => self.syms.name(*s).split('`').next().unwrap_or("").into(),
            _ => String::from("the function"),
        };
        let what = match takes {
            0 => String::from("no arguments"),
            1 => String::from("1 argument"),
            n => crate::msg!("{} arguments", n),
        };
        Error::arity(crate::msg!("{} takes {}, got {}", name, what, given))
    }

    /// The named parameters of a function body, if it has any (ticket 061).
    #[cfg_attr(feature = "ram-hot", allow(unsafe_code), link_section = ".time_critical.klong_params_of")]
    fn params_of(f: &Value) -> Option<(Rc<Tree>, Range)> {
        if let Value::Fn(func) = f {
            if let Func::Code { tree, node } = &**func {
                if let Node::Body { params, has_params: true, .. } = tree.node(*node) {
                    return Some((tree.clone(), *params));
                }
            }
        }
        None
    }

    fn is_native(f: &Value) -> bool {
        matches!(f, Value::Fn(func) if matches!(&**func, Func::Native { .. }))
    }

    fn eval_arg(&mut self, a: &Arg) -> Result<Value> {
        match a {
            Arg::Code(tree, id) => self.eval(tree, *id, true),
            Arg::Val(v) => Ok(v.clone()),
        }
    }

    /// KlongPy `_resolve_deeper`: more than three projections; the result stands only if it
    /// is a function with named parameters.
    #[inline(never)]
    fn resolve_deeper(&mut self, f: &mut Value, layers: &mut Vec<Layer>, arity: &mut u8, node: &mut bool) -> Result<()> {
        let mut g = f.clone();
        let mut g_arity = *arity;
        let mut g_node = *node;
        let before = layers.len();
        for _ in 0..256 {
            let n = layers.len();
            let was = g_arity;
            let prev = g.clone();
            self.resolve(&mut g, layers, &mut g_arity, &mut g_node, false)?;
            if layers.len() == n && was == g_arity && same_value(&prev, &g) {
                break;
            }
        }
        // A function with named parameters, or the body of a function of x, y and z under more
        // projections than three passes go through (a short call is one).
        let reached_body = !same_value(f, &g)
            && match &g {
                Value::Sym(_) => false,
                Value::Fn(func) => match &**func {
                    Func::Code { tree, node } => !tree.is_kgfn(*node),
                    Func::Proj { .. } => false,
                    Func::Native { .. } => true,
                },
                _ => true,
            };
        if Self::params_of(&g).is_some() || reached_body {
            *f = g;
            *arity = g_arity;
            *node = g_node;
        } else {
            layers.truncate(before);
        }
        Ok(())
    }

    /// Evaluate a call node (KlongPy `_eval_fn`). The arguments are expressions; they are
    /// evaluated here, once, after projections are merged.
    #[inline(never)]
    #[cfg_attr(feature = "ram-hot", allow(unsafe_code), link_section = ".time_critical.klong_eval_call")]
    fn eval_call(&mut self, tree: &Rc<Tree>, id: NodeId) -> Result<Value> {
        let mut layers = self.layer_pool.pop().unwrap_or_default();
        let r = self.eval_call_in(tree, id, &mut layers);
        layers.clear();
        if self.layer_pool.len() < 16 && layers.capacity() <= 8 {
            self.layer_pool.push(layers);
        }
        r
    }

    #[inline(never)]
    #[cfg_attr(feature = "ram-hot", allow(unsafe_code), link_section = ".time_critical.klong_eval_call_in")]
    fn eval_call_in(&mut self, tree: &Rc<Tree>, id: NodeId, layers: &mut Vec<Layer>) -> Result<Value> {
        let (a, args, has_args, arity0, calling) = match tree.node(id) {
            // (a written call, not the evaluation of a body that is a bare name, and not the call of a literal)
            Node::Fn { a, args, has_args, arity, literal, .. } => (*a, *args, *has_args, *arity, *has_args && !*literal),
            _ => return Err(Error::internal("not a call")),
        };
        if !has_args && self.cfg.strict == 2 {
            if let Node::QSym(s) = tree.node(a) {
                // {:foo}: the body is a quoted symbol, and this is its evaluation
                return Ok(Value::Sym(*s));
            }
        }
        let mut f = self.node_value(tree, a);
        let mut arity = arity0;
        let mut node = false;
        try_push(layers, if has_args { Layer::Tree(tree.clone(), args) } else { Layer::None })?;
        // three passes as there are at most three arguments: x, y and z
        self.resolve(&mut f, layers, &mut arity, &mut node, calling)?;
        self.resolve(&mut f, layers, &mut arity, &mut node, calling)?;
        self.resolve(&mut f, layers, &mut arity, &mut node, calling)?;

        if layers.len() == 1 {
            // the common case: no projection involved
            let ids = if has_args { tree.ids(args) } else { &[] };
            let n = ids.len();
            if let Some((ptree, params)) = Self::params_of(&f) {
                let m = params.len as usize;
                let given = ids.iter().filter(|&&q| q != NONE.0).count();
                if n != m {
                    if n > m || !node {
                        return Err(self.too_many(&self.node_value(tree, a), m, given));
                    }
                    return self.projection(tree, id, None, m);
                }
                if ids.contains(&NONE.0) {
                    return self.projection(tree, id, None, m);
                }
                let mut frame: Frame = self.new_frame(n + 2)?;
                for (k, &q) in ids.iter().enumerate() {
                    let v = self.eval(tree, NodeId(q), true)?;
                    try_push(&mut frame, (SymId(ptree.ids(params)[k]), Some(v)))?;
                }
                return self.run_fn(f, frame);
            }
            if n < arity as usize || ids.contains(&NONE.0) {
                // a short call (fewer arguments than the function takes, or an argument left
                // empty): a projection, at every strict level
                return self.projection(tree, id, None, arity as usize);
            }
            if n > arity as usize && self.cfg.strict == 2 && !Self::is_native(&f) {
                return Err(self.too_many(&self.node_value(tree, a), arity as usize, n));
            }
            let mut frame: Frame = self.new_frame(n.min(3) + 2)?;
            for (k, &q) in ids.iter().take(3).enumerate() {
                let v = self.eval(tree, NodeId(q), true)?;
                try_push(&mut frame, (SymId(k as u32), Some(v)))?;
            }
            return self.run_fn(f, frame);
        }

        let callee = self.node_value(tree, a);
        self.eval_call_slow(tree, id, layers, callee, f, arity, node)
    }

    /// A call that goes through projections: its layers merged. Apart from the common case, which is
    /// on the way of every level of recursion and so keeps its stack frame small.
    #[inline(never)]
    #[allow(clippy::too_many_arguments)]
    fn eval_call_slow(&mut self, tree: &Rc<Tree>, id: NodeId, layers: &mut Vec<Layer>, callee: Value, mut f: Value, mut arity: u8, mut node: bool) -> Result<Value> {
        if layers.len() == 4 {
            self.resolve_deeper(&mut f, layers, &mut arity, &mut node)?;
        }
        let mut lists: Vec<Vec<Option<Arg>>> = layers.drain(..).map(Self::layer_args).collect::<Result<_>>()?;
        lists.reverse();
        let named = Self::params_of(&f);
        if self.cfg.strict == 2 || named.is_some() {
            self.check_projection(&callee, &lists)?;
        }
        let merged = Self::merge(lists);
        let n = merged.len();
        let holes = merged.iter().filter(|q| q.is_none()).count();
        if let Some((ptree, params)) = named {
            let m = params.len as usize;
            if n != m {
                if n > m || !node {
                    return Err(self.too_many(&callee, m, n - holes));
                }
                return self.projection(tree, id, Some(holes), m);
            }
            if holes > 0 {
                return self.projection(tree, id, Some(holes), m);
            }
            let mut frame: Frame = self.new_frame(n + 2)?;
            for (k, q) in merged.iter().enumerate() {
                if let Some(a) = q {
                    let v = self.eval_arg(a)?;
                    try_push(&mut frame, (SymId(ptree.ids(params)[k]), Some(v)))?;
                }
            }
            return self.run_fn(f, frame);
        }
        if n < arity as usize || holes > 0 {
            return self.projection(tree, id, Some(holes), arity as usize);
        }
        if n > arity as usize && self.cfg.strict == 2 && !Self::is_native(&f) {
            return Err(self.too_many(&callee, arity as usize, n));
        }
        let mut frame: Frame = self.new_frame(n.min(3) + 2)?;
        for (k, q) in merged.iter().take(3).enumerate() {
            if let Some(a) = q {
                let v = self.eval_arg(a)?;
                try_push(&mut frame, (SymId(k as u32), Some(v)))?;
            }
        }
        self.run_fn(f, frame)
    }

    /// The callee to store in a projection: as written, except that x, y, z and .f are
    /// replaced by what they hold now (they mean something else when it is completed).
    fn projected_callee(&self, a: Value) -> Value {
        if let Value::Sym(s) = a {
            if sym::is_arg(s) || s == sym::DOT_F {
                if let Ok(Some(v)) = self.lookup(s) {
                    return v;
                }
            }
        }
        a
    }

    /// The value of the written call `id` when it does not supply every argument (KlongPy
    /// `_projection`, ticket 033): a projection with an empty slot for each argument still
    /// missing. An argument may be left empty, f(1;), or left off the end, f(1); the two are the
    /// same thing. `merged_holes` is the number of empty slots after the arguments were merged
    /// into the projection the call calls (None: it calls none); `need` the number of arguments
    /// the function takes.
    ///
    /// Inside a function the arguments are evaluated here, once, and the projection holds their
    /// values (they are the function's arguments and locals, gone when the projection is
    /// completed). At top level it keeps the argument expressions, evaluated each time it is
    /// completed (legacy: a::1;g::f(a;);a::2 completes with 2).
    #[inline(never)]
    fn projection(&mut self, tree: &Rc<Tree>, id: NodeId, merged_holes: Option<usize>, need: usize) -> Result<Value> {
        let as_written = Value::Fn(Rc::new(Func::Code { tree: tree.clone(), node: id }));
        let (a, ids) = match tree.node(id) {
            Node::Fn { a, args, has_args: true, .. } => (*a, tree.ids(*args)),
            // not a call: a function value
            _ => return Ok(as_written),
        };
        let own_holes = ids.iter().filter(|&&q| q == NONE.0).count();
        let pad = match merged_holes {
            Some(h) => h.saturating_sub(own_holes),
            None => need.saturating_sub(ids.len()),
        };
        let mut slots: Vec<Slot> = try_vec(ids.len() + pad)?;
        if !self.frames.is_empty() {
            for &q in ids {
                let pushed = if q == NONE.0 { Slot::Hole } else { Slot::Val(self.eval(tree, NodeId(q), true)?) };
                try_push(&mut slots, pushed)?;
            }
        } else {
            let callee_is_arg = matches!(tree.node(a), Node::Sym(s) if sym::is_arg(*s));
            if pad == 0 && !callee_is_arg {
                return Ok(as_written);
            }
            for &q in ids {
                try_push(&mut slots, if q == NONE.0 { Slot::Hole } else { Slot::Code(tree.clone(), NodeId(q)) })?;
            }
        }
        for _ in 0..pad {
            try_push(&mut slots, Slot::Hole)?;
        }
        let callee = self.projected_callee(self.node_value(tree, a));
        Ok(Value::Fn(Rc::new(Func::proj(callee, slots)?)))
    }

    /// The projection `apply` returns when the values do not complete `f`: the values, held as
    /// they are, and one empty slot for each argument still missing.
    fn value_projection(&mut self, f: &Value, values: Vec<Value>, holes: usize) -> Result<Value> {
        let mut slots: Vec<Slot> = try_vec(values.len() + holes)?;
        slots.extend(values.into_iter().map(Slot::Val));
        for _ in 0..holes {
            try_push(&mut slots, Slot::Hole)?;
        }
        Ok(Value::Fn(Rc::new(Func::proj(self.projected_callee(f.clone()), slots)?)))
    }

    /// Apply a function to argument values that are already evaluated (KlongPy `apply`): no
    /// value is evaluated again. Every call made at run time (adverbs, the apply form of `@`,
    /// callbacks of the embedder) goes through here. With fewer values than the function takes,
    /// the unapplied call is the result.
    pub fn apply(&mut self, f: &Value, values: Vec<Value>, arity: Option<u8>) -> Result<Value> {
        if let Value::Prim(op) = f {
            let mut it = values.into_iter();
            return match (it.next(), it.next()) {
                (Some(x), None) => self.monad(*op, x),
                (Some(x), Some(y)) => self.dyad(*op, x, y),
                _ => Err(Error::arity("an operator needs one or two operands")),
            };
        }
        let given = values.len();
        let mut g = f.clone();
        let mut f_arity = arity.unwrap_or(given.min(255) as u8);
        let mut node = false;
        let mut layers: Vec<Layer> = Vec::new();
        try_push(&mut layers, Layer::Vals(values))?;
        self.resolve(&mut g, &mut layers, &mut f_arity, &mut node, true)?;
        self.resolve(&mut g, &mut layers, &mut f_arity, &mut node, true)?;
        self.resolve(&mut g, &mut layers, &mut f_arity, &mut node, true)?;

        if layers.len() == 1 {
            let values = match layers.pop() {
                Some(Layer::Vals(v)) => v,
                _ => Vec::new(),
            };
            if let Some((ptree, params)) = Self::params_of(&g) {
                if given != params.len as usize {
                    return Err(self.too_many(f, params.len as usize, given));
                }
                let mut frame: Frame = try_vec(given + 2)?;
                for (k, v) in values.into_iter().enumerate() {
                    try_push(&mut frame, (SymId(ptree.ids(params)[k]), Some(v)))?;
                }
                return self.run_fn(g, frame);
            }
            if given < f_arity as usize {
                // a short call is a projection
                return self.value_projection(f, values, f_arity as usize - given);
            }
            if given > f_arity as usize && self.cfg.strict == 2 && !Self::is_native(&g) {
                return Err(self.too_many(f, f_arity as usize, given));
            }
            let mut frame: Frame = try_vec(given.min(3) + 2)?;
            for (k, v) in values.into_iter().take(3).enumerate() {
                try_push(&mut frame, (SymId(k as u32), Some(v)))?;
            }
            return self.run_fn(g, frame);
        }

        let keep: Vec<Value> = match &layers[0] {
            Layer::Vals(v) => v.clone(),
            _ => Vec::new(),
        };
        if layers.len() == 4 {
            self.resolve_deeper(&mut g, &mut layers, &mut f_arity, &mut node)?;
        }
        let named = Self::params_of(&g);
        let mut lists: Vec<Vec<Option<Arg>>> = layers.into_iter().map(Self::layer_args).collect::<Result<_>>()?;
        lists.reverse();
        if named.is_some() || self.cfg.strict == 2 {
            self.check_projection(f, &lists)?;
        }
        let merged = Self::merge(lists);
        let n = merged.len();
        let holes = merged.iter().filter(|q| q.is_none()).count();
        if let Some((ptree, params)) = named {
            let m = params.len as usize;
            if n != m {
                return Err(self.too_many(f, m, n));
            }
            if holes > 0 {
                return Err(self.too_many(f, given + holes, given));
            }
            let mut frame: Frame = try_vec(n + 2)?;
            for (k, q) in merged.iter().enumerate() {
                if let Some(a) = q {
                    let v = self.eval_arg(a)?;
                    try_push(&mut frame, (SymId(ptree.ids(params)[k]), Some(v)))?;
                }
            }
            return self.run_fn(g, frame);
        }
        if n < f_arity as usize || holes > 0 {
            // the values do not fill the projection: a projection over the slots that are left
            return self.value_projection(f, keep, holes);
        }
        let mut frame: Frame = try_vec(n.min(3) + 2)?;
        for (k, q) in merged.iter().take(3).enumerate() {
            if let Some(a) = q {
                let v = self.eval_arg(a)?;
                try_push(&mut frame, (SymId(k as u32), Some(v)))?;
            }
        }
        self.run_fn(g, frame)
    }

    /// Run a resolved function with its arguments bound in `frame` (KlongPy `_run_fn`): declare
    /// its locals, set `.f`, evaluate the body in a new frame.
    #[inline(never)]
    #[cfg_attr(feature = "ram-hot", allow(unsafe_code), link_section = ".time_critical.klong_run_fn")]
    fn run_fn(&mut self, f: Value, mut frame: Frame) -> Result<Value> {
        self.tick()?;

        let func = match &f {
            Value::Fn(func) => func.clone(),
            Value::Sym(s) => {
                let s = *s;
                return self.run_name(f, s, frame);
            }
            _ => return Ok(f),
        };
        match &*func {
            Func::Code { tree, node } => {
                try_push(&mut frame, (sym::DOT_F, Some(f.clone())))?;
                let r = match tree.node(*node) {
                    Node::Body { stmts, locals, .. } => {
                        for &q in tree.ids(*locals) {
                            let s = SymId(q);
                            if !frame.iter().any(|(k, _)| *k == s) {
                                // an unassigned local is its own symbol; at strict level 2
                                // reading it is an error
                                frame.try_reserve(1).map_err(|_| Error::memory())?;
                                try_push(&mut frame, (s, if self.cfg.strict == 2 { None } else { Some(Value::Sym(s)) }))?;
                            }
                        }
                        self.push_frame(frame)?;
                        self.eval_seq(tree, *stmts)
                    }
                    _ => {
                        self.push_frame(frame)?;
                        self.eval(tree, *node, true)
                    }
                };
                self.pop_frame();
                r
            }
            Func::Native { arity, f: native, .. } => {
                let (arity, native) = (*arity, native.clone());
                self.run_native(frame, arity, native)
            }
            // a projection that was given nothing more is the value it is
            Func::Proj { .. } => Ok(f),
        }
    }

    /// The arms of `run_fn` that are not on the way of a recursion of Klong code, apart (see `eval_monad`).
    /// A symbol in call position that names no function: its value as a name.
    #[inline(never)]
    fn run_name(&mut self, f: Value, s: SymId, mut frame: Frame) -> Result<Value> {
        try_push(&mut frame, (sym::DOT_F, Some(f)))?;
        self.push_frame(frame)?;
        let r = match self.lookup(s) {
            Ok(Some(v)) => Ok(v),
            Ok(None) => self.undefined(s),
            Err(e) => Err(e),
        };
        self.pop_frame();
        r
    }

    /// A native function called with the arguments bound in `frame`.
    #[inline(never)]
    fn run_native(&mut self, frame: Frame, arity: u8, native: NativeFn) -> Result<Value> {
        let mut args: Vec<Value> = Vec::new();
        for (k, v) in frame.iter() {
            if sym::is_arg(*k) {
                if let Some(v) = v {
                    try_push(&mut args, v.clone())?;
                }
            }
        }
        if self.cfg.strict == 2 && arity != WILDCARD && args.len() > arity as usize {
            return Err(self.too_many(&Value::Undefined, arity as usize, args.len()));
        }
        native(self, &args)
    }

    // ----- adverb chains --------------------------------------------------------------------------

    #[inline(never)]
    fn eval_adv(&mut self, tree: &Rc<Tree>, id: NodeId) -> Result<Value> {
        let (verb, verb_arity, advs, first_arity, dyad, l, r) = match tree.node(id) {
            Node::Adv { verb, verb_arity, advs, first_arity, dyad, l, r } => (*verb, *verb_arity, *advs, *first_arity, *dyad, *l, *r),
            _ => return Err(Error::internal("not an adverb chain")),
        };
        let advs = tree.ids(advs);
        let ch = Chain { tree, verb, verb_arity, verb_val: self.node_value(tree, verb), advs, first_arity };
        let last_arity = if advs.len() == 1 { first_arity } else { 1 };
        if last_arity == 1 {
            let x = if dyad {
                // legacy: the pair of operands is evaluated as a statement list
                self.eval(tree, l, true)?;
                self.eval(tree, r, true)?
            } else {
                self.eval(tree, r, false)?
            };
            self.chain_call(&ch, advs.len(), x, None)
        } else {
            let x = self.eval(tree, l, false)?;
            let y = self.eval(tree, r, false)?;
            self.chain_call(&ch, advs.len(), x, Some(y))
        }
    }

    /// Call level `level` of a chain: level 0 is the verb, level i the verb modified by the
    /// first i adverbs.
    /// The native stack against `Config::stack_limit`: called wherever the evaluator recurses
    /// (a nested evaluation, a level of an adverb chain).
    #[inline]
    pub(crate) fn check_stack(&mut self) -> Result<()> {
        let used = self.stack_base.abs_diff(stack_here());
        if used > self.max_stack_seen {
            self.max_stack_seen = used;
            if let Some(limit) = self.cfg.stack_limit {
                if used > limit {
                    return Err(Error::range("expression nested too deeply").with_hint("the native stack budget of this build is used up"));
                }
            }
        }
        Ok(())
    }

    pub(crate) fn chain_call(&mut self, ch: &Chain, level: usize, x: Value, y: Option<Value>) -> Result<Value> {
        // a chain of adverbs recurses once per adverb without a nested evaluation
        self.check_stack()?;
        if (level as u64) + (self.depth as u64) > self.cfg.max_depth as u64 {
            return Err(Error::range("expression nested too deeply"));
        }
        if level == 0 {
            return match ch.tree.node(ch.verb) {
                Node::OpVal { op, .. } if ops::by_id(*op).is_some() || self.syms.name(*op) == ops::TRACK => match (ch.verb_arity, y) {
                    (1, None) => self.monad(*op, x),
                    (2, Some(y)) => {
                        let def = self.op_def(*op)?;
                        match def.atomic {
                            Some(a) => ops::atomic(&self.syms, a, &x, &y),
                            None => self.dyad(*op, x, y),
                        }
                    }
                    _ => Err(Error::typ("the verb of an adverb was given the wrong number of operands")),
                },
                Node::OpVal { op, .. } => Err(self.unknown_op(*op)),
                _ => match (ch.verb_arity, y) {
                    (1, None) => self.apply(&ch.verb_val, alloc::vec![x], None),
                    (2, Some(y)) => self.apply(&ch.verb_val, alloc::vec![x, y], None),
                    _ => Err(Error::typ("the verb of an adverb was given the wrong number of operands")),
                },
            };
        }
        let glyph = SymId(ch.advs[level - 1]);
        let arity = if level == 1 { ch.first_arity } else { 1 };
        match (arity, y) {
            (1, None) => self.adverb1(ch, level, glyph, x),
            (2, Some(y)) => self.adverb2(ch, level, glyph, x, y),
            _ => Err(Error::typ("an adverb was given the wrong number of operands")),
        }
    }
}

fn same_value(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Fn(x), Value::Fn(y)) => Rc::ptr_eq(x, y),
        (Value::Sym(x), Value::Sym(y)) => x == y,
        (Value::Fn(_), _) | (_, Value::Fn(_)) => false,
        _ => true,
    }
}

impl Kind {
    /// The exit line of the conformance runner protocol.
    pub fn protocol_line(self) -> String {
        crate::msg!("error: {}", self.name())
    }
}
