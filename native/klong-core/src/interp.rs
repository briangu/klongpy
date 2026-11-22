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
    /// on a 64-bit host (measured), several times that in a debug build.
    pub max_depth: u32,
    /// Strict level 0 to 2 (`docs/errors.md`).
    pub strict: u8,
    /// Stop with an `interrupt` error after this many ticks (fuzzing, tests).
    pub step_limit: Option<u64>,
    /// The slow tick (the platform's `idle`, Ctrl-C, the lane) runs when `ticks & tick_mask == 0`: every
    /// 4096 calls and adverb iterations by default. A slow processor lowers it: a Cortex-M0+ takes about
    /// 90 us for a call of a Klong function, so 4096 of them are 0.4 s without a look at the lane.
    pub tick_mask: u32,
    /// Set from outside (an interrupt handler) to stop the evaluation. Must live in memory
    /// where atomics work (internal RAM on the chips of `docs/native/porting.md`).
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
/// atomic pairing, drop): about 300 bytes on a 64-bit host in a release build,
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

/// A module scope made by `.l` (design point 1): the globals of one file. A function of the file runs in it (`Tree::module`).
#[derive(Default)]
struct ModScope {
    /// Sorted by symbol.
    vars: Vec<(SymId, Value)>,
    /// The dictionary `.export` fixed.
    export: Option<Value>,
    /// A classic `.module` block set a qualified name here: a plain name that is not found is looked for among them.
    classic: bool,
    /// What the runtime tells of the file the module came from (cache key and directory, empty once the module is let go), whether it
    /// is a file of the image's library, and whether its body has finished.
    key: String,
    dir: String,
    lib: bool,
    done: bool,
}

/// One global scope (KlongPy's context list): the program's own names, or a module.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Scope {
    Plain,
    Module(SymId),
}

const JOURNALED: u16 = 0x8000;
const NO_LINK: u32 = u32::MAX;

/// The bindings of one running function (its arguments, `.f`, its locals, what it assigns): `len` entries of `Interp::slots` from `start`.
/// A frame is the last one registered while it runs, so it grows at the end of the array.
#[derive(Clone, Copy)]
struct FrameRef {
    start: u32,
    len: u32,
}

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

/// Give back the room a table has beyond its elements, through a resize that the allocator may turn into a move: a table that grew
/// while memory was full lies where the memory was free, and what was freed above it since is better for it and for whatever
/// needs the room below (the board's allocator moves a small block up when there is a place higher: `kh_grow`).
fn compact<T>(v: &mut Vec<T>) {
    if v.try_reserve_exact(1).is_ok() {
        v.shrink_to_fit();
    }
}

/// A program to run: borrowed (copied once, for error messages) or handed over (kept as that copy).
/// A function literal of at least this many bytes of text, in a loaded program, is kept as text.
const LAZY_MIN: u32 = 24;
/// How many parsed functions kept as text are kept, and about how many bytes of trees they may hold between programs.
const LAZY_ENTRIES: usize = 24;
const LAZY_BUDGET: usize = 12288;

enum Prog<'a> {
    Text(&'a str),
    Owned(String),
    /// A text that lives as long as the program does (a library in the image): read in place, never copied.
    Static(&'static str),
}

pub struct Interp {
    pub syms: Interner,
    pub cfg: Config,
    /// SPIKE: when set, the value of every statement of the program run at the top level is kept here (KlongPy's `exec`).
    pub collect: Option<Vec<Value>>,
    /// HOST (feature `host`): the embedder's implementation of the operators the core does not have (the gradient operators): called
    /// with the operator's glyph and its operands.
    #[cfg(feature = "host")]
    pub host_op: Option<Rc<HostOp>>,
    /// Optional numeric backend, invoked before evaluating dyads or the direct plus reduction.
    #[cfg(feature = "host")]
    pub host_numeric: Option<Rc<HostNumeric>>,
    /// HOST: called when the program run at the top level has been parsed and before its first statement runs (its warnings are
    /// then in `take_warnings`); an error from it ends the run with that error and nothing has run.
    #[cfg(feature = "host")]
    pub on_parsed: Option<Rc<dyn Fn(&mut Interp) -> Result<()>>>,
    /// HOST: the embedder takes over a numeric vector a primitive or a literal has produced (a torch backend wants its arrays on its
    /// device): called on every value `eval` returns that is a native vector; `None` leaves it as it is.
    #[cfg(feature = "host")]
    adopt: Option<Rc<dyn Fn(&mut Interp, &Value) -> Option<Value>>>,
    /// HOST: float atoms are computed and held as float32 (the host's arrays are float32, a torch backend): every float value `eval`
    /// returns is rounded to float32 (`set_float_width`). Off by default.
    #[cfg(feature = "host")]
    f32_floats: bool,
    /// Typed IPC calls may forward existing scalar values at full wire precision.
    #[cfg(feature = "host")]
    ipc_float_reads: bool,
    /// HOST: `adopt` is set or `f32_floats` is on: what `eval` has to look at in a value it returns (one test, as when there was only `adopt`).
    #[cfg(feature = "host")]
    post: bool,
    /// HOST: the arguments beyond the third of the call of a system function that is being made (set just before it runs).
    #[cfg(feature = "wide-args")]
    pub extra_args: Vec<Value>,
    /// KGPY-ONLY: says whether a system function is one the Python-hosted command has and this program does not (`klong-cli`: `kgpy_only.rs`).
    #[cfg(feature = "kgpy-only")]
    pub kgpy_only: Option<fn(&str) -> bool>,
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
    /// The bindings of every running function in one array (a call claims the entries after the last, and gives them back when it returns:
    /// no allocation per call). Past the end of the last frame there can be arguments on their way to a call, see `arg`.
    slots: Vec<(SymId, Option<Value>)>,
    frames: Vec<FrameRef>,
    /// The first frame a name is looked for in by the code that is running: when it is the body of a function with named parameters
    /// (`f(a;b)::{...}`), the frame of that function, which sees its own locals and then the globals of its module and never the locals
    /// of whatever called it (docs/design-principles.md, principle 4). 0 for any other code: a function without a parameter list
    /// (`{x+y}`) sees every frame, as in Klong, so a callback written in a function still reads that function's locals.
    lex_base: usize,
    /// The module scope each frame belongs to (parallel to `frames`): a function sees the frames of its own module only.
    fmods: Vec<u16>,
    /// The module scope that is running (0: the program's own).
    cur_mod: u16,
    /// The scopes made by `.l`, by id minus one.
    modules: Vec<ModScope>,
    /// Module ids start after this many (the ids of modules that were all let go are not used again).
    mod_base: u16,
    /// Per symbol: how many bindings of it are on the frame stack.
    shadow: Vec<u32>,
    hooks: Box<dyn Hooks>,
    depth: u32,
    ticks: u64,
    /// The function values made for a few tree nodes lately (a function literal's body is made into one at every
    /// call: an allocation). Each entry keeps its tree alive, so a tree's address cannot be taken for another's.
    lazy_cache: core::cell::RefCell<Vec<(usize, u32, Rc<Tree>, NodeId, usize)>>,
    /// How many functions kept as text were parsed again, how many bytes of trees (about) they were, and how many were let go to make room.
    lazy_stats: core::cell::Cell<(u64, u64, u64)>,
    body_cache: core::cell::RefCell<[Option<(usize, NodeId, Value)>; 8]>,
    body_next: Cell<u8>,
    layer_pool: Vec<Vec<Layer>>,
    module: Option<String>,
    /// High-water mark of `depth` (for sizing a board's stack).
    pub max_depth_seen: u32,
}

/// An operator the embedder implements (see `Interp::host_op`).
#[cfg(feature = "host")]
pub type HostOp = dyn Fn(&mut Interp, &str, &[Value]) -> Result<Value>;

#[cfg(feature = "host")]
pub type HostNumeric = dyn Fn(&str, &Value, Option<&Value>) -> Option<Result<Value>>;

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
    /// The body of the verb when it is a function literal that `apply_fast` runs (found once for the whole chain).
    #[cfg(feature = "speed")]
    pub body: Option<Value>,
    pub advs: &'a [u32],
    pub first_arity: u8,
}

impl Interp {
    pub fn new(cfg: Config, hooks: Box<dyn Hooks>) -> Interp {
        crate::value::set_max_nest(cfg.max_nest);
        Interp {
            syms: Interner::new(),
            cfg,
            collect: None,
            #[cfg(feature = "host")]
            host_op: None,
            #[cfg(feature = "host")]
            host_numeric: None,
            #[cfg(feature = "host")]
            on_parsed: None,
            #[cfg(feature = "wide-args")]
            extra_args: Vec::new(),
            #[cfg(feature = "host")]
            adopt: None,
            #[cfg(feature = "host")]
            f32_floats: false,
            #[cfg(feature = "host")]
            ipc_float_reads: false,
            #[cfg(feature = "host")]
            post: false,
            #[cfg(feature = "kgpy-only")]
            kgpy_only: None,
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
            slots: Vec::new(),
            frames: Vec::new(),
            lex_base: 0,
            fmods: Vec::new(),
            cur_mod: 0,
            modules: Vec::new(),
            mod_base: 0,
            shadow: Vec::new(),
            hooks,
            depth: 0,
            ticks: 0,
            lazy_cache: core::cell::RefCell::new(Vec::new()),
            lazy_stats: core::cell::Cell::new((0, 0, 0)),
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
        self.set_global(s, Value::Fn(Rc::new(Func::native(s, arity, f))));
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

    /// SPIKE: remove a global (`del klong['x']`). True if it was defined.
    pub fn undefine_global(&mut self, name: &str) -> bool {
        let Some(s) = self.syms.lookup(name) else { return false };
        match self.globals.get_mut(s.0 as usize) {
            Some(slot) if slot.is_some() => {
                *slot = None;
                self.generation = self.generation.wrapping_add(1);
                true
            }
            _ => false,
        }
    }

    /// The names of every global that is defined (the embedder's `names()`), in symbol order.
    pub fn global_names(&self) -> Vec<String> {
        let mut out = Vec::new();
        for (i, g) in self.globals.iter().enumerate() {
            if g.is_some() {
                out.push(String::from(self.syms.name(SymId(i as u32))));
            }
        }
        out
    }

    /// The persistent values owned by the evaluator. Call only when it is parked.
    #[cfg(feature = "host")]
    pub fn visit_host_roots(&self, visit: &mut dyn FnMut(&Value), visit_tree: &mut dyn FnMut(&Rc<Tree>)) {
        self.globals.iter().flatten().for_each(&mut *visit);
        for module in &self.modules {
            module.vars.iter().for_each(|(_, value)| visit(value));
            module.export.iter().for_each(&mut *visit);
        }
        self.collect.iter().flatten().for_each(&mut *visit);
        #[cfg(feature = "wide-args")]
        self.extra_args.iter().for_each(&mut *visit);
        self.slots.iter().filter_map(|(_, v)| v.as_ref()).for_each(&mut *visit);
        if let Some(journal) = &self.journal {
            journal.old.iter().filter_map(|(_, v)| v.as_ref()).for_each(&mut *visit);
        }
        self.body_cache.borrow().iter().flatten().for_each(|(_, _, v)| visit(v));
        self.lazy_cache.borrow().iter().for_each(|(_, _, tree, _, _)| visit_tree(tree));
        for layer in self.layer_pool.iter().flatten() {
            match layer {
                Layer::None => {}
                Layer::Tree(tree, _) => visit_tree(tree),
                Layer::Vals(values) => values.iter().for_each(&mut *visit),
                Layer::Slots(slots) => {
                    for arg in slots {
                        match arg {
                            Slot::Hole => {}
                            Slot::Val(value) => visit(value),
                            Slot::Code(tree, _) => visit_tree(tree),
                        }
                    }
                }
            }
        }
    }

    pub fn global(&self, name: &str) -> Option<Value> {
        let s = self.syms.lookup(name)?;
        self.var(s)
    }

    /// `global` for a name that may be qualified: `mgmt.call` is looked up as `mgmt.call` is written (a name that starts with a dot is one
    /// name). For the embedder that calls a function by the name a remote caller sent.
    pub fn global_path(&self, name: &str) -> Option<Value> {

        if let Some(Ok(segs)) = sym::qual_segments(name) {
            // a name made of segments is looked up as `a.b` is (the dots of a name that starts with a dot are part of the name)
            let mut cur = self.var(self.syms.lookup(segs[0])?)?;
            for k in &segs[1..] {
                let next = match &cur {
                    Value::Dict(d) => d.borrow().get(&Value::text(k).ok()?)?.clone(),
                    _ => return None,
                };
                cur = next;
            }
            return Some(cur);
        }
        let s = self.syms.lookup(name)?;
        self.var(s)
    }

    /// The global called exactly `name`, even when the name is made of segments (a host that registered `read.adc` registered that
    /// name, not the entry `adc` of `read`).
    pub fn global_plain(&self, name: &str) -> Option<Value> {
        let s = self.syms.lookup(name)?;
        self.var(s)
    }

    /// A global of the running scope by symbol.
    fn var(&self, s: SymId) -> Option<Value> {
        if self.cur_mod == 0 {
            self.global_ref(s).cloned()
        } else {
            self.module_var(s)
        }
    }

    /// Define or replace a global by name (for the embedder: `.a`, `fullpath`).
    pub fn define_global(&mut self, name: &str, v: Value) -> Result<()> {
        let s = self.syms.intern(name);
        if let Some(e) = self.syms.failure() {
            return Err(e);
        }
        self.new_global(s, v)
    }

    // ----- module scopes (design point 1) -----------------------------------------------------

    /// The module that came from the file with this cache key: (its id, whether its body has finished). A module that was let go is not found.
    pub fn module_find(&self, key: &str) -> Option<(u16, bool)> {
        let i = self.modules.iter().position(|m| m.key == key)?;
        Some((self.mod_base + i as u16 + 1, self.modules[i].done))
    }

    /// The directory of the file module `id` came from, and whether it is a file of the library.
    pub fn module_dir(&self, id: u16) -> Option<(&str, bool)> {
        self.mslot(id).and_then(|i| self.modules.get(i)).map(|m| (m.dir.as_str(), m.lib))
    }

    pub fn module_done(&mut self, id: u16) {
        if let Some(m) = self.mslot(id).and_then(|i| self.modules.get_mut(i)) {
            m.done = true;
        }
    }

    /// The cache keys of the modules whose bodies are running, outermost first.
    pub fn module_loading(&self) -> Vec<&str> {
        self.modules.iter().filter(|m| !m.done && !m.key.is_empty()).map(|m| m.key.as_str()).collect()
    }

    /// Where module `id` is in `modules`, when it still is.
    fn mslot(&self, id: u16) -> Option<usize> {
        (id.checked_sub(self.mod_base)? as usize).checked_sub(1)
    }

    /// A new module scope, empty; its id is what `module_run` and `Tree::module` take.
    pub fn module_new(&mut self, key: &str, dir: &str, lib: bool) -> Result<u16> {
        if self.mod_base as usize + self.modules.len() >= (u16::MAX - 1) as usize {
            return Err(Error::range("too many modules"));
        }
        self.modules.try_reserve(1).map_err(|_| Error::memory())?;
        if self.modules.is_empty() {
            // (frames are tagged with their module from now on: the ones that are on the stack belong to the program's own scope)
            self.fmods.clear();
            self.fmods.try_reserve(self.frames.len() + 1).map_err(|_| Error::memory())?;
            self.fmods.resize(self.frames.len(), 0);
        }
        let (key, dir) = (crate::value::try_str(key)?, crate::value::try_str(dir)?);
        self.modules.push(ModScope { key, dir, lib, ..ModScope::default() });
        Ok(self.mod_base + self.modules.len() as u16)
    }

    /// Let go of what a module scope holds (a module whose body failed); its id is not used again.
    pub fn module_drop(&mut self, id: u16) {
        if let Some(m) = self.mslot(id).and_then(|i| self.modules.get_mut(i)) {
            *m = ModScope::default();
        }
    }

    /// Let go of the modules nothing refers to any more (design point 3 follow-up, principle 4: nothing is held that no value refers to).
    /// A module is referenced while a value outside it holds its export dictionary or one of the functions it exports. Called between
    /// the statements of the program at the top level; a library that was dropped with `m::0` is freed then.
    #[inline(never)]
    fn sweep_modules(&mut self) {
        if self.modules.is_empty() || self.running != 1 || !self.frames.is_empty() {
            return;
        }
        // (the recent function bodies are cached as values: they would count as references)
        if let Ok(mut c) = self.body_cache.try_borrow_mut() {
            *c = [None, None, None, None, None, None, None, None];
        }
        let n = self.modules.len();
        // the modules that something outside every module scope refers to, then what those refer to
        let mut live: Vec<bool> = (0..n).map(|i| self.referenced_from_outside(i)).collect();
        loop {
            let mut grew = false;
            for i in 0..n {
                if !live[i] || self.modules[i].export.is_none() {
                    continue;
                }
                for j in 0..n {
                    if !live[j] && self.holds_module(i, j) {
                        live[j] = true;
                        grew = true;
                    }
                }
            }
            if !grew {
                break;
            }
        }
        for i in 0..n {
            if !live[i] && self.modules[i].export.is_some() {
                self.modules[i] = ModScope::default();
            }
        }
        // when every module has gone the table goes too (a board gives back what it had; ids start again: a function of a module that was
        // let go has nothing to run in)
        if self.modules.iter().all(|m| m.key.is_empty()) && self.fmods.is_empty() {
            self.mod_base += self.modules.len() as u16;
            self.modules = Vec::new();
            self.fmods = Vec::new();
        }
    }

    /// The values a module scope holds at its top level: its variables and the entries of its export dictionary.
    fn scope_values(&self, i: usize) -> Vec<Value> {
        let m = &self.modules[i];
        let mut out: Vec<Value> = m.vars.iter().map(|(_, v)| v.clone()).collect();
        if let Some(Value::Dict(e)) = &m.export {
            out.extend(e.borrow().vals().iter().cloned());
        }
        out
    }

    /// Does the scope of module `i` hold the dictionary or an exported function of module `j`?
    fn holds_module(&self, i: usize, j: usize) -> bool {
        let Some(Value::Dict(ej)) = &self.modules[j].export else { return false };
        let ej = ej.borrow();
        self.scope_values(i).iter().any(|v| match v {
            Value::Dict(d) => core::ptr::eq(d.as_ptr() as *const Dict, &*ej as *const Dict),
            Value::Fn(f) => ej.vals().iter().any(|x| matches!(x, Value::Fn(g) if Rc::ptr_eq(f, g))),
            _ => false,
        })
    }

    /// Is a value that no module scope holds referring to module `i`: its export dictionary, or a function it exports?
    fn referenced_from_outside(&self, i: usize) -> bool {
        let Some(Value::Dict(e)) = &self.modules[i].export else { return false };
        let mut inside: Vec<Value> = Vec::new();
        for k in 0..self.modules.len() {
            inside.extend(self.scope_values(k));
        }
        // (the scope's own reference to the dictionary is the one that is counted off: `inside` holds the others, by name or by entry)
        let held = inside.iter().filter(|v| matches!(v, Value::Dict(d) if Rc::ptr_eq(d, e))).count();
        // (`inside` holds a clone of each value it lists, so every holder inside counts twice)
        if Rc::strong_count(e) > 1 + 2 * held {
            return true;
        }
        let d = e.borrow();
        d.vals().iter().any(|v| match v {
            Value::Fn(f) => Rc::strong_count(f) > 2 * inside.iter().filter(|x| matches!(x, Value::Fn(g) if Rc::ptr_eq(f, g))).count(),
            _ => false,
        })
    }

    /// The module scope that is running (0: the program's own).
    pub fn current_module_id(&self) -> u16 {
        self.cur_mod
    }

    /// Fix the dictionary a module exports. False when it already has one (`.export` twice).
    pub fn module_set_export(&mut self, id: u16, d: Value) -> bool {
        match self.mslot(id).and_then(|i| self.modules.get_mut(i)) {
            Some(m) if m.export.is_none() => {
                m.export = Some(d);
                true
            }
            _ => false,
        }
    }

    /// The variable `name` of module `id` (`export`, once the body has run).
    pub fn module_get(&self, id: u16, name: &str) -> Option<Value> {
        let s = self.syms.lookup(name)?;
        let m = self.mslot(id).and_then(|i| self.modules.get(i))?;
        let i = m.vars.binary_search_by_key(&s, |e| e.0).ok()?;
        Some(m.vars[i].1.clone())
    }

    /// Take the variable `name` out of module `id` (the interface variable once it has been read: the scope keeps no second copy of
    /// what the interface holds, so a module nothing refers to can be let go).
    pub fn module_remove(&mut self, id: u16, name: &str) {
        let Some(s) = self.syms.lookup(name) else { return };
        if let Some(m) = self.mslot(id).and_then(|i| self.modules.get_mut(i)) {
            if let Ok(i) = m.vars.binary_search_by_key(&s, |e| e.0) {
                m.vars.remove(i);
            }
        }
    }

    /// A function value written as `name(a;b;c;d)::{body}` (named parameters, so any number of them): the function, with nothing defined.
    /// For a host that builds a function of more arguments than a native one takes.
    pub fn function_from_text(&mut self, text: &str) -> Result<Value> {
        let mut module: Option<String> = None;
        let p = parse::parse(text, "<function>", &mut self.syms, &mut module, 0, self.cfg.max_depth, None)?;
        p.tree.module.set(self.cur_mod);
        let [stmt] = p.stmts[..] else { return Err(Error::internal("function_from_text: one statement expected")) };
        let Node::Dyad { r, .. } = p.tree.node(stmt) else { return Err(Error::internal("function_from_text: an assignment expected")) };
        let r = *r;
        self.eval(&p.tree, r, false)
    }

    pub fn module_export(&self, id: u16) -> Option<Value> {
        self.mslot(id).and_then(|i| self.modules.get(i)).and_then(|m| m.export.clone())
    }

    /// Run `text` as the body of module `id`: its top-level names are the module's, its functions run in it. The loader's frames and
    /// classic `.module` state are not seen and are given back. Streamed, as a load is.
    pub fn module_run(&mut self, id: u16, text: String, name: &str) -> Result<Option<Value>> {
        let saved = (self.cur_mod, self.module.take());
        self.cur_mod = id;
        let r = self.run_prog_mode(Prog::Owned(text), name, true);
        self.cur_mod = saved.0;
        self.module = saved.1;
        r
    }

    /// `module_run` of a text that lives for ever.
    pub fn module_run_static(&mut self, id: u16, text: &'static str, name: &str) -> Result<Option<Value>> {
        let saved = (self.cur_mod, self.module.take());
        self.cur_mod = id;
        let r = self.run_prog_mode(Prog::Static(text), name, true);
        self.cur_mod = saved.0;
        self.module = saved.1;
        r
    }

    pub fn format(&self, v: &Value) -> String {
        crate::fmt::to_string(&self.syms, v)
    }

    /// Measure the native stack from here, against `limit`, for a thread that re-enters a running interpreter (a callback or a
    /// loop on another thread: its stack is not the one the outer run started on). Returns what `restore_stack` puts back.
    pub fn rebase_stack(&mut self, limit: Option<usize>) -> (usize, Option<usize>, usize) {
        let prev = (self.stack_base, self.cfg.stack_limit, self.max_stack_seen);
        self.stack_base = stack_here();
        self.cfg.stack_limit = limit;
        self.max_stack_seen = 0;
        prev
    }

    /// Put back what `rebase_stack` returned (the high-water mark keeps the larger of the two).
    pub fn restore_stack(&mut self, prev: (usize, Option<usize>, usize)) {
        self.stack_base = prev.0;
        self.cfg.stack_limit = prev.1;
        self.max_stack_seen = self.max_stack_seen.max(prev.2);
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
            Prog::Static(text) => self.parse(text, name),
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

    /// `run` of a program that is handed over, one statement at a time: each is parsed, evaluated and let go before the next is read.
    /// What differs from `run`, which parses the whole program first (as KlongPy does): a syntax error after some statements comes
    /// after they have run, as if they had been typed at the prompt.
    pub fn run_streamed_owned(&mut self, text: String, name: &str) -> Result<Option<Value>> {
        self.run_prog_mode(Prog::Owned(text), name, true)
    }

    fn run_prog(&mut self, prog: Prog, name: &str) -> Result<Option<Value>> {
        self.run_prog_mode(prog, name, false)
    }

    fn run_prog_mode(&mut self, prog: Prog, name: &str, streamed: bool) -> Result<Option<Value>> {
        if self.running == 0 {
            // the nesting bound is one value for the process (see `value::set_max_nest`): the
            // interpreter that runs asserts its own, whichever was created last
            crate::value::set_max_nest(self.cfg.max_nest);
            self.stack_base = stack_here();
            // a failure deferred by something outside an evaluation is not this program's
            let _ = crate::value::take_deferred();
        }
        self.running += 1;
        let r = if streamed { self.run_streamed(prog, name) } else { self.run_parsed(prog, name) };
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
        p.tree.module.set(self.cur_mod);
        #[cfg(feature = "host")]
        if self.running == 1 {
            if let Some(h) = self.on_parsed.clone() {
                h(self)?;
            }
        }
        let frames = self.frames.len();
        let depth = self.depth;
        let mut last = None;
        for &s in &p.stmts {
            self.void = false;
            self.check_statement(&p.tree, s)?;
            match self.eval(&p.tree, s, true) {
                Ok(v) => {
                    if self.running == 1 {
                        if let Some(c) = self.collect.as_mut() {
                            c.push(v.clone());
                        }
                    }
                    last = Some(v);
                    if !self.modules.is_empty() {
                        self.sweep_modules();
                    }
                }
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

    /// `run_parsed` one statement at a time: each is parsed (its tokens from a window of the text, its tree its own), evaluated and
    /// let go before the next is read, so that what a load holds at once is one statement and what it keeps (the function bodies).
    #[inline(never)]
    /// A top-level statement of a form the language has retired (`module::discarded_extern`).
    fn check_statement(&self, tree: &Tree, s: NodeId) -> Result<()> {
        if crate::module::discarded_extern(tree, &self.syms, s) {
            let e = Error::typ(crate::module::DISCARDED_EXTERN);
            #[cfg(feature = "diag")]
            let e = e.with_hint(crate::module::DISCARDED_EXTERN_HINT);
            return Err(match tree.span(s) {
                Some(sp) => {
                    let mut e = e.with_span(sp);
                    e.source = tree.source.clone();
                    e
                }
                None => e,
            });
        }
        Ok(())
    }

    fn run_streamed(&mut self, prog: Prog, name: &str) -> Result<Option<Value>> {
        let text: alloc::borrow::Cow<'static, str> = match prog {
            Prog::Text(t) => alloc::borrow::Cow::Owned(crate::value::try_str(t)?),
            Prog::Owned(t) => alloc::borrow::Cow::Owned(t),
            Prog::Static(t) => alloc::borrow::Cow::Borrowed(t),
        };
        let source = Rc::new(crate::error::Source { name: crate::value::try_str(name)?, text });
        let frames = self.frames.len();
        let depth = self.depth;
        let mut last = None;
        let mut pos = 0usize;
        let mut ctx: Option<Rc<crate::value::LazyCtx>> = None;
        loop {
            let stack = self.stack();
            let mut window = 256usize; // (a statement that needs more finds it: the parser doubles it)
            if ctx.as_ref().map(|c| (&c.module, c.strict, c.mid)) != Some((&self.module, self.cfg.strict, self.cur_mod)) {
                ctx = Some(Rc::new(crate::value::LazyCtx { module: self.module.clone(), mid: self.cur_mod, strict: self.cfg.strict }));
            }
            let next = parse::parse_next(&source, pos, source.text.len(), &mut window, ctx.clone(), &mut self.syms, &mut self.module, self.cfg.strict, self.cfg.max_depth, stack)?;
            if next.skipped {
                pos = next.end;
                continue;
            }
            let Some(p) = next.parsed else { break };
            p.tree.module.set(self.cur_mod);
            for &s in &p.stmts {
                self.void = false;
                self.check_statement(&p.tree, s)?;
                match self.eval(&p.tree, s, true) {
                    Ok(v) => {
                        last = Some(v);
                        if !self.modules.is_empty() {
                        self.sweep_modules();
                    }
                    }
                    Err(e) => {
                        while self.frames.len() > frames {
                            self.pop_frame();
                        }
                        self.depth = depth;
                        return Err(e);
                    }
                }
            }
            if next.last {
                break;
            }
            pos = next.end;
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
    /// The warnings the parser has recorded since they were last taken (strict level 0), in source order, once each, as KlongPy lists
    /// them: a list or dictionary left open inside another that is left open is one warning.
    #[cfg(feature = "diag")]
    pub fn take_warnings(&mut self) -> Vec<crate::sym::Warning> {
        let mut ws = core::mem::take(&mut self.syms.warnings);
        ws.sort_by_key(|w| w.span);
        let mut out: Vec<crate::sym::Warning> = Vec::new();
        let mut open_seen = false;
        for w in ws {
            if out.iter().any(|o| o.span == w.span && o.code == w.code) {
                continue;
            }
            if w.code == "unterminated-list" || w.code == "unterminated-dict" {
                if open_seen {
                    continue;
                }
                open_seen = true;
            }
            out.push(w);
        }
        out
    }

    /// What Python-side code asks of a function value: arity, parameter names, locals, source text, whether it is a projection and what it
    /// has bound. `None` for a value that is not a function.
    #[cfg(feature = "diag")]
    pub fn fn_info(&mut self, v: &Value) -> Option<FnInfo> {
        let mut info = FnInfo::default();
        let func = match v {
            Value::Fn(f) => f.clone(),
            Value::Prim(op) => {
                info.arity = match ops::by_id(*op) {
                    Some(d) if d.dyad.is_some() => 2,
                    _ => 1,
                };
                info.kind = "operator";
                return Some(info);
            }
            _ => return None,
        };
        info.arity = crate::fmt::func_arity(&func);
        let (tree, node) = match &*func {
            Func::Code { tree, node } => (tree.clone(), *node),
            Func::Lazy(l) => match self.materialize(l) {
                Ok(t) => t,
                Err(_) => {
                    info.kind = "function";
                    return Some(info);
                }
            },
            Func::Proj { f, args, .. } => {
                info.kind = "projection";
                info.projection = true;
                info.callee = Some(f.clone());
                for a in args {
                    info.bound.push(match a {
                        Slot::Val(v) => Some(v.clone()),
                        // a value written in the call (`r::f(10)`): a literal is itself, an expression its text
                        Slot::Code(t, n) => match (t.node(*n), t.source.as_ref(), t.span(*n)) {
                            (Node::Lit(v), _, _) => Some(v.clone()),
                            (_, Some(src), Some((s, e))) => Some(Value::Str(Rc::new(crate::value::Text::new(String::from(&src.text.as_ref()[s as usize..e as usize]))))),
                            _ => Some(Value::Undefined),
                        },
                        Slot::Hole => None,
                    });
                }
                // a short call (`f(10)` of a function of three arguments) leaves the arguments it did not reach as holes
                if let Some(a) = self.fn_info_arity_of(f) {
                    while info.bound.len() < a as usize {
                        info.bound.push(None);
                    }
                }
                return Some(info);
            }
            Func::Native { name, arity, .. } => {
                info.kind = "native";
                info.name = Some(String::from(self.syms.name(*name)));
                info.arity = if *arity == WILDCARD { 0 } else { *arity };
                return Some(info);
            }
        };
        let text: Option<&str> = tree.source.as_ref().map(|s| s.text.as_ref());
        let name_of = |s: u32| String::from(self.syms.name(SymId(s)).split('`').next().unwrap_or(""));
        let body = match tree.node(node) {
            Node::Fn { a, args, has_args, .. } => {
                if *has_args {
                    let callee = self.node_value(&tree, *a);
                    // a short call (`f()` of a function of one argument) is a projection too: the arguments it did not reach are holes
                    let short = self.fn_info_arity_of(&callee).is_some_and(|n| (tree.ids(*args).len() as u8) < n);
                    info.projection = short || tree.ids(*args).iter().any(|&q| q == NONE.0);
                    for &q in tree.ids(*args) {
                        info.bound.push(if q == NONE.0 {
                            None
                        } else {
                            match (tree.node(NodeId(q)), text, tree.span(NodeId(q))) {
                                (Node::Lit(v), _, _) => Some(v.clone()),
                                (_, Some(t), Some((s, e))) => Some(Value::Str(Rc::new(crate::value::Text::new(String::from(&t[s as usize..e as usize]))))),
                                _ => Some(Value::Undefined),
                            }
                        });
                    }
                    if let Some(n) = self.fn_info_arity_of(&callee) {
                        while info.bound.len() < n as usize {
                            info.bound.push(None);
                        }
                    }
                    info.callee = Some(callee);
                }
                *a
            }
            _ => node,
        };
        info.kind = if info.projection { "projection" } else { "function" };
        if let Node::Body { locals, params, has_params, .. } = tree.node(body) {
            info.locals = tree.ids(*locals).iter().map(|&s| name_of(s)).collect();
            if *has_params {
                info.params = Some(tree.ids(*params).iter().map(|&s| name_of(s)).collect());
            }
        }
        if let (Some(t), Some((s, e))) = (text, tree.span(node)) {
            info.source = Some(String::from(&t[s as usize..e as usize]));
            info.span = Some((s, e));
            info.span_chars = Some((t[..s as usize].chars().count(), t[..e as usize].chars().count()));
        }
        Some(info)
    }

    /// The number of arguments a function value takes (what a short call is short of).
    #[cfg(feature = "diag")]
    fn fn_info_arity_of(&self, f: &Value) -> Option<u8> {
        match f {
            Value::Fn(func) => match &**func {
                Func::Proj { .. } => None,
                other => Some(crate::fmt::func_arity(other)).filter(|&a| a != WILDCARD),
            },
            _ => None,
        }
    }

    pub fn read_datum(&mut self, text: &str) -> Result<Option<(usize, Value)>> {
        let stack = self.stack();
        parse::read_datum(text, &mut self.syms, &mut self.module, self.cfg.max_depth, stack)
    }

    /// `read_datum` with whether the object is complete (see `parse::read_datum_ex`); `used` is a byte offset into `text`.
    pub fn read_datum_ex(&mut self, text: &str) -> Result<Option<(usize, Value, bool)>> {
        let stack = self.stack();
        parse::read_datum_ex(text, &mut self.syms, &mut self.module, self.cfg.max_depth, stack)
    }

    /// Run a program at global scope, whatever function is running (`.E`): the frames of the
    /// callers are set aside, so a definition lands in the globals.
    pub fn run_global(&mut self, text: &str, name: &str) -> Result<Option<Value>> {
        let frames = core::mem::take(&mut self.frames);
        let fmods = core::mem::take(&mut self.fmods);
        let shadow = core::mem::take(&mut self.shadow);
        let base = core::mem::take(&mut self.lex_base);
        let r = self.run(text, name);
        self.lex_base = base;
        self.frames = frames;
        self.fmods = fmods;
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

    /// `load` of a text that lives for ever (a library in the image's flash): it is read where it is, with no copy in RAM.
    pub fn load_static(&mut self, text: &'static str, name: &str) -> Result<Option<Value>> {
        self.load_prog(Prog::Static(text), name)
    }

    fn load_prog(&mut self, prog: Prog, name: &str) -> Result<Option<Value>> {
        if self.journal.is_some() {
            return self.run_prog_mode(prog, name, true);
        }
        for g in self.gscope.iter_mut() {
            *g &= !JOURNALED;
        }
        self.journal = Some(Journal { old: Vec::new(), scopes: self.scopes.len(), exports: self.exports.len(), module: self.module.clone() });
        let r = self.run_prog_mode(prog, name, true);
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
    /// The name of the module the program is in, `None` outside any.
    pub fn current_module(&self) -> Option<&str> {
        self.module.as_deref()
    }

    pub fn module_start(&mut self, name: &str) -> Result<()> {
        if self.cur_mod != 0 {
            // (in a module file the names of a block belong to the file's scope)
            return Ok(());
        }
        let s = self.syms.intern(name);
        if let Some(e) = self.syms.failure() {
            return Err(e);
        }
        self.push_scope(Scope::Module(s))
    }

    /// What `.module(:name)` does, for a host that is not running a program: the parser reads the following programs in the module `name` and the
    /// names assigned at top level belong to it; `None` is `.module(0)`.
    pub fn set_module(&mut self, name: Option<&str>) -> Result<()> {
        match name {
            Some(n) => {
                self.module_start(n)?;
                self.module = Some(String::from(n));
            }
            None => {
                self.module_stop()?;
                self.module = None;
            }
        }
        Ok(())
    }

    /// `.module(0)`: a new plain scope on top of the module.
    pub fn module_stop(&mut self) -> Result<()> {
        if self.cur_mod != 0 {
            return Ok(());
        }
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

    #[inline(always)]
    fn grow_shadow(&mut self, i: usize) -> Result<()> {
        if self.shadow.len() <= i {
            return self.grow_shadow_to(i);
        }
        Ok(())
    }

    #[cold]
    #[inline(never)]
    fn grow_shadow_to(&mut self, i: usize) -> Result<()> {
        let more = i + 1 - self.shadow.len();
        self.shadow.try_reserve(more).map_err(|_| Error::memory())?;
        self.shadow.resize(i + 1, 0);
        Ok(())
    }

    /// Put the argument `(s, v)` of a call that is being made on the slots, after the `k` that are there already (arguments are evaluated one
    /// after another, and what they call opens and closes its own frames above them). On failure the `k` are given back.
    #[inline(always)]
    fn arg(&mut self, k: usize, s: SymId, v: Value) -> Result<()> {
        if let Err(e) = try_push(&mut self.slots, (s, Some(v))) {
            self.drop_args(k);
            return Err(e);
        }
        Ok(())
    }

    /// Give back the `k` arguments on top of the slots (a call that fails before it runs).
    #[cold]
    fn drop_args(&mut self, k: usize) {
        let n = self.slots.len().saturating_sub(k);
        self.slots.truncate(n);
    }

    /// `r`, with the `k` arguments of the call being made given back when it is an error.
    #[inline(always)]
    fn args_or<T>(&mut self, k: usize, r: Result<T>) -> Result<T> {
        if r.is_err() {
            self.drop_args(k);
        }
        r
    }

    /// The `n` arguments on top of the slots become a new frame, with room for `extra` more bindings. On failure they are given back.
    #[cfg_attr(feature = "ram-hot", allow(unsafe_code), link_section = ".time_critical.klong_push_frame")]
    #[inline]
    fn open_frame(&mut self, n: usize, extra: usize) -> Result<()> {
        let r = self.open_frame_in(n, extra);
        if r.is_err() {
            self.drop_args(n);
        }
        r
    }

    fn open_frame_in(&mut self, n: usize, extra: usize) -> Result<()> {
        if self.slots.capacity() - self.slots.len() < extra {
            self.slots.try_reserve(extra).map_err(|_| Error::memory())?;
        }
        self.frames.try_reserve(1).map_err(|_| Error::memory())?;
        if !self.modules.is_empty() {
            self.fmods.try_reserve(1).map_err(|_| Error::memory())?;
        }
        let start = self.slots.len() - n;
        let top = self.slots[start..].iter().map(|(s, _)| s.0 as usize).max().unwrap_or(0);
        self.grow_shadow(top)?;
        for (s, _) in &self.slots[start..] {
            self.shadow[s.0 as usize] += 1;
        }
        // (the room was made above)
        self.frames.push(FrameRef { start: start as u32, len: n as u32 });
        if !self.modules.is_empty() {
            self.fmods.push(self.cur_mod);
        }
        Ok(())
    }

    /// Bind `s` in the frame that is running (the last, with nothing past it): `open_frame` made room for it.
    #[inline(always)]
    fn frame_add(&mut self, s: SymId, v: Option<Value>) -> Result<()> {
        self.grow_shadow(s.0 as usize)?;
        try_push(&mut self.slots, (s, v))?;
        self.shadow[s.0 as usize] += 1;
        if let Some(f) = self.frames.last_mut() {
            f.len += 1;
        }
        Ok(())
    }

    /// The frame of a call of the function `f`: its `n` arguments, `.f`, and each of `locals` that is not an argument (an unassigned local is
    /// its own symbol; at strict level 2 reading it is an error). On failure, nothing is left.
    fn open_fn_frame(&mut self, n: usize, f: Value, locals: &[u32]) -> Result<()> {
        self.open_frame(n, locals.len() + 1)?;
        let r = self.frame_add(sym::DOT_F, Some(f)).and_then(|()| {
            for &q in locals {
                let s = SymId(q);
                let last = self.frames.len() - 1;
                if !self.slots[self.frame_range(last)].iter().any(|(k, _)| *k == s) {
                    self.frame_add(s, if self.cfg.strict == 2 { None } else { Some(Value::Sym(s)) })?;
                }
            }
            Ok(())
        });
        if r.is_err() {
            self.pop_frame();
        }
        r
    }

    /// The entries of frame `i`, as a range of the slots.
    #[inline]
    fn frame_range(&self, i: usize) -> core::ops::Range<usize> {
        let f = self.frames[i];
        f.start as usize..(f.start + f.len) as usize
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

    #[cfg_attr(feature = "ram-hot", allow(unsafe_code), link_section = ".time_critical.klong_pop_frame")]
    fn pop_frame(&mut self) {
        if let Some(f) = self.frames.pop() {
            if !self.modules.is_empty() {
                self.fmods.pop();
            }
            let start = f.start as usize;
            debug_assert_eq!(self.slots.len(), start + f.len as usize, "a frame is released from the top");
            for (s, _) in &self.slots[start..] {
                if let Some(c) = self.shadow.get_mut(s.0 as usize) {
                    *c = c.saturating_sub(1);
                }
            }
            self.slots.truncate(start);
        }
    }

    /// The value of a name: the innermost frame of the running module that binds it, else the module's global, else (a name that starts
    /// with a dot) the system's; `a.b` is `a@"b"` (design point 4).
    #[cfg_attr(feature = "ram-hot", allow(unsafe_code), link_section = ".time_critical.klong_lookup")]
    pub fn lookup(&mut self, s: SymId) -> Result<Option<Value>> {
        if self.shadow.get(s.0 as usize).copied().unwrap_or(0) > 0 {
            let tagged = !self.modules.is_empty();
            for i in (self.lex_base..self.frames.len()).rev() {
                if tagged && self.fmods[i] != self.cur_mod {
                    continue;
                }
                for (k, v) in self.slots[self.frame_range(i)].iter() {
                    if *k == s {
                        return match v {
                            Some(v) => Ok(Some(v.dup())),
                            None => Err(Error::name(crate::msg!("local '{}' read before assignment", self.syms.name(s)))),
                        };
                    }
                }
            }
        }
        if self.cur_mod == 0 {
            if let Some(v) = self.global_ref(s) {
                return Ok(Some(v.dup()));
            }
        } else if let Some(v) = self.module_var(s) {
            return Ok(Some(v));
        }
        if let Some(base) = self.syms.qual_base(s) {
            return self.qual_get(s, base).map(Some);
        }
        Ok(None)
    }

    /// The global `s` of the running module (no frames, no qualified names): the module's own, then, for a name that starts with a dot,
    /// the system's.
    fn module_var(&self, s: SymId) -> Option<Value> {
        let m = self.mslot(self.cur_mod).and_then(|i| self.modules.get(i))?;
        if let Ok(i) = m.vars.binary_search_by_key(&s, |e| e.0) {
            return Some(m.vars[i].1.clone());
        }
        // A classic `.module(:x)` block inside this module: a plain name finds the names the block defined (`gauss` finds `gauss`math`), and
        // a name the block wrote (`gauss`spline`) that the block does not define is the module's own plain name, as it is in the globals.
        if m.classic || self.syms.name(s).contains('`') {
            let name = self.syms.name(s);
            let base = name.split_once('`').map_or(name, |(b, _)| b);
            for (k, v) in &m.vars {
                let kn = self.syms.name(*k);
                if kn == base || kn.split_once('`').is_some_and(|(b, _)| b == base) {
                    return Some(v.clone());
                }
            }
        }
        if self.syms.name(s).starts_with('.') {
            return self.global_ref(s).cloned();
        }
        None
    }

    fn module_set(&mut self, id: u16, s: SymId, v: Value) -> Result<()> {
        // (the table of a module is kept as long as the module is, and given back with it: placed with what is thrown away, the parse, not
        // among the names and tables that stay for good, between which the doublings of a table that grows while a library loads would
        // leave holes: the largest vector of a node that loaded the management library and dropped it lost 6% to them)
        let _tag = crate::htag::Scope::new(crate::htag::TREE);
        let classic = self.syms.name(s).contains('`');
        let Some(m) = self.mslot(id).and_then(|i| self.modules.get_mut(i)) else {
            return Err(Error::system("the module this function belongs to was let go"));
        };
        match m.vars.binary_search_by_key(&s, |e| e.0) {
            Ok(i) => m.vars[i].1 = v,
            Err(i) => {
                m.vars.try_reserve(1).map_err(|_| Error::memory())?;
                m.vars.insert(i, (s, v));
            }
        }
        if classic {
            m.classic = true;
        }
        Ok(())
    }

    /// Is `s` a global of the running scope (the module's own for a module, any name of the system's for the program)?
    fn scope_has(&self, s: SymId) -> bool {
        if self.cur_mod == 0 {
            return self.has_global(s);
        }
        self.mslot(self.cur_mod).and_then(|i| self.modules.get(i)).is_some_and(|m| m.vars.binary_search_by_key(&s, |e| e.0).is_ok())
    }

    /// The running code is in a function of its own module (so a new name is that function's local), not at the top level of a file.
    fn in_fn(&self) -> bool {
        if self.modules.is_empty() {
            return !self.frames.is_empty();
        }
        self.fmods.last() == Some(&self.cur_mod)
    }

    /// Create the global `s` in the running scope.
    fn new_global(&mut self, s: SymId, v: Value) -> Result<()> {
        if self.module_private_global(s) {
            return self.module_set(self.cur_mod, s, v);
        }
        self.try_set_global(s, v)
    }

    fn module_private_global(&self, s: SymId) -> bool {
        self.cur_mod != 0 && !self.syms.name(s).starts_with('.')
    }

    /// The text of the segment `i` of the qualified name `s` (0 is the base, 1 the first key): the text of the name, cut at its dots.
    fn qseg(&self, s: SymId, i: usize) -> &str {
        let name = self.syms.name(s);
        let name = &name[..name.find('`').unwrap_or(name.len())];
        // (the last key is the common one: cut from the end)
        if i == self.syms.qual_keys(s) {
            return name.rsplit('.').next().unwrap_or("");
        }
        name.split('.').nth(i).unwrap_or("")
    }

    #[cold]
    #[cfg_attr(not(feature = "diag"), allow(unused_variables))]
    fn not_a_dict(&self, s: SymId, i: usize, v: &Value) -> Error {
        Error::typ(crate::dmsg!("not a dictionary", "{}: {} is not a dictionary (it is {})", self.plain(s), self.qual_path(s, i), v.describe()))
    }

    /// The text of `a.b` up to the segment `n` (the base is segment 0).
    #[cfg(feature = "diag")]
    fn qual_path(&self, s: SymId, n: usize) -> String {
        let mut p = String::from(self.qseg(s, 0));
        for i in 1..=n {
            p.push('.');
            p.push_str(self.qseg(s, i));
        }
        p
    }

    /// The dictionary, or the host value that answers as one (a remote dictionary), at the end of the walk of the qualified name `s` over
    /// `n` of its keys (0: the base itself); an error that names the place when a name is not defined, an entry is missing, or the value
    /// is not a dictionary. A host value is read as `x@key` is: through the host (the host table of an embedder). The keys are read from
    /// the text of the name, as it is (nothing is made for them on the way: a read of `d.a` allocates nothing).
    fn qual_container(&mut self, s: SymId, base: SymId, n: usize) -> Result<Value> {
        let mut cur = match self.lookup(base)? {
            // (a declared local that was never assigned holds its own name)
            Some(Value::Sym(t)) if t == base => None,
            v => v,
        }
        .ok_or_else(|| {
            #[cfg(feature = "diag")]
            let (b, full) = (self.qseg(s, 0), self.plain(s));
            Error::name(crate::dmsg!("undefined", "{} is not defined ({} means {}@\"{}\")", b, full, b, full.split('.').nth(1).unwrap_or("")))
        })?;
        for i in 0..=n {
            match &cur {
                // (an array of the host, a tensor, is not a dictionary: `l.a` on one is the error it is on a native list)
                #[cfg(feature = "host")]
                Value::Ext(e) if e.0.host_shape().is_some() => return Err(self.not_a_dict(s, i, &cur)),
                Value::Dict(_) | Value::Ext(_) if i == n => return Ok(cur),
                Value::Dict(d) => {
                    let next = d.borrow().get_str(self.qseg(s, i + 1)).cloned();
                    cur = next.ok_or_else(|| self.no_entry(s, i))?;
                }
                Value::Ext(_) => {
                    let at = self.syms.intern("@");
                    let key = Value::text(self.qseg(s, i + 1))?;
                    cur = self.dyad(at, cur, key)?;
                }
                other => return Err(self.not_a_dict(s, i, other)),
            }
        }
        Err(Error::internal("qualified name"))
    }

    #[cfg_attr(not(feature = "diag"), allow(unused_variables))]
    fn no_entry(&self, s: SymId, i: usize) -> Error {
        Error::name(crate::dmsg!("no such entry", "{} has no entry \"{}\"", self.qual_path(s, i), self.qseg(s, i + 1)))
    }

    /// The value of `a.b.c`.
    #[inline(never)]
    fn qual_get(&mut self, s: SymId, base: SymId) -> Result<Value> {
        let n = self.syms.qual_keys(s) - 1;
        match self.qual_container(s, base, n)? {
            Value::Dict(d) => {
                let v = d.borrow().get_str(self.qseg(s, n + 1)).cloned();
                v.ok_or_else(|| self.no_entry(s, n))
            }
            host => {
                let at = self.syms.intern("@");
                let key = Value::text(self.qseg(s, n + 1))?;
                self.dyad(at, host, key)
            }
        }
    }

    /// `a.b::v`: the entry `b` of the dictionary `a` is set (a new key is added); `a.b.c::v` needs `a.b` to be there. A host value takes
    /// the pair as `a,[key value]` does.
    #[inline(never)]
    fn qual_set(&mut self, s: SymId, base: SymId, v: Value) -> Result<()> {
        let n = self.syms.qual_keys(s) - 1;
        let key = Value::text(self.qseg(s, n + 1))?;
        let d = match self.qual_container(s, base, n)? {
            Value::Dict(d) => d,
            host => {
                let comma = self.syms.intern(",");
                let pair = Value::from_items(alloc::vec![key, v])?;
                self.dyad(comma, host, pair)?;
                return Ok(());
            }
        };
        let mut d = d.borrow_mut();
        if d.is_readonly() {
            let e = Error::typ(crate::dmsg!("read-only dictionary", "cannot assign {}: {} is read-only (it is a module)", self.plain(s), self.qual_path(s, n)));
            #[cfg(feature = "diag")]
            let e = e.with_hint("copy it with  e:::{};e::e,m  or change the module's file");
            return Err(e);
        }
        d.set(key, v)
    }

    /// `name::value` (KlongPy `KlongContext.__setitem__`): assign where the name lives; a new
    /// name is created in the innermost frame.
    pub fn assign(&mut self, s: SymId, v: Value) -> Result<()> {
        if let Some(base) = self.syms.qual_base(s) {
            return self.qual_set(s, base, v);
        }
        if !sym::is_arg(s) {
            if self.shadow.get(s.0 as usize).copied().unwrap_or(0) > 0 {
                let cm = self.cur_mod;
                let tagged = !self.modules.is_empty();
                let base = self.lex_base;
                for i in (base..self.frames.len()).rev() {
                    if tagged && self.fmods[i] != cm {
                        continue;
                    }
                    let r = self.frame_range(i);
                    for (k, slot) in self.slots[r].iter_mut() {
                        if *k == s {
                            *slot = Some(v);
                            return Ok(());
                        }
                    }
                }
            }
            if self.cur_mod == 0 {
                if self.has_global(s) {
                    if self.journal.is_some() {
                        return self.try_set_global(s, v);
                    }
                    self.globals[s.0 as usize] = Some(v);
                    return Ok(());
                }
            } else if self.scope_has(s) {
                return self.module_set(self.cur_mod, s, v);
            } else if self.syms.name(s).starts_with('.') && self.has_global(s) {
                return self.try_set_global(s, v);
            }
        }
        // x, y and z are the arguments of the running function and are assignable like its
        // locals; they are always bound in its own frame
        if self.cfg.strict >= 1 && self.in_fn() && !sym::is_arg(s) {
            let n = self.syms.name(s);
            return Err(Error::name(crate::msg!("undefined variable: {}", n))
                .with_hint(crate::msg!("declare it as a local in the parameter list, {{[{}]; ...}}, or define the global before the function is called", n)));
        }
        if !self.in_fn() {
            return self.new_global(s, v);
        }
        let i = s.0 as usize;
        self.grow_shadow(i)?;
        if let Some(n) = self.frames.len().checked_sub(1) {
            let r = self.frame_range(n);
            let end = r.end;
            for (k, slot) in self.slots[r].iter_mut() {
                if *k == s {
                    *slot = Some(v);
                    return Ok(());
                }
            }
            self.slots.try_reserve(1).map_err(|_| Error::memory())?;
            // (past the frame there can be arguments on their way to a call, `a::1` written among them: they move up by one, and are
            // counted from the top by whoever made them)
            self.slots.insert(end, (s, Some(v)));
            self.frames[n].len += 1;
            self.shadow[i] += 1;
        }
        Ok(())
    }

    /// A name as the program wrote it (without the module a classic `.module` block added).
    #[cfg_attr(not(feature = "diag"), allow(dead_code))]
    fn plain(&self, s: SymId) -> &str {
        let n = self.syms.name(s);
        match n.find('`') {
            Some(i) => &n[..i],
            None => n,
        }
    }

    fn is_defined(&self, s: SymId) -> bool {
        self.shadow.get(s.0 as usize).copied().unwrap_or(0) > 0
            || if self.cur_mod == 0 { self.global_ref(s).is_some() } else { self.module_var(s).is_some() }
            // (a qualified name is looked up when it is used, and says what is missing then)
            || self.syms.qual_base(s).is_some()
    }

    /// The value of a name that is not defined.
    #[inline(never)]
    fn undefined(&mut self, s: SymId) -> Result<Value> {
        #[cfg(feature = "kgpy-only")]
        if let Some(e) = self.needs_kgpy(s) {
            if self.cfg.strict == 2 || (self.cfg.strict == 1 && self.in_fn()) {
                return Err(e);
            }
        }
        let level = self.cfg.strict;
        if level == 2 || (level == 1 && self.in_fn()) {
            let e = Error::name(crate::msg!("undefined: {}", self.syms.name(s)));
            #[cfg(feature = "diag")]
            if level == 2 {
                if let Some(h) = self.did_you_mean(s) {
                    return Err(e.with_hint(h));
                }
            }
            return Err(e);
        }
        Ok(Value::Sym(s))
    }

    /// The error for a system function that only the Python-hosted command has (`Interp::kgpy_only`).
    #[cfg(feature = "kgpy-only")]
    #[cold]
    fn needs_kgpy(&self, s: SymId) -> Option<Error> {
        let name = self.syms.name(s);
        if self.kgpy_only.is_some_and(|f| f(name)) {
            return Some(Error::name(crate::msg!("{} needs kgpy (the Python-hosted command)", name)));
        }
        None
    }

    /// The hint for an undefined name: the defined name closest to it, when one is (KlongPy `_did_you_mean`: difflib's ratio, at least 0.7).
    #[cfg(feature = "diag")]
    #[cold]
    fn did_you_mean(&self, s: SymId) -> Option<String> {
        fn matches(a: &[char], b: &[char]) -> usize {
            // difflib's gestalt matching: the longest common block, then what lies before it and after it
            if a.is_empty() || b.is_empty() {
                return 0;
            }
            let (mut best, mut bi, mut bj) = (0, 0, 0);
            for i in 0..a.len() {
                for j in 0..b.len() {
                    let mut k = 0;
                    while i + k < a.len() && j + k < b.len() && a[i + k] == b[j + k] {
                        k += 1;
                    }
                    if k > best {
                        best = k;
                        bi = i;
                        bj = j;
                    }
                }
            }
            if best == 0 {
                return 0;
            }
            best + matches(&a[..bi], &b[..bj]) + matches(&a[bi + best..], &b[bj + best..])
        }
        let name = self.plain(s);
        let want: Vec<char> = name.chars().collect();
        let mut best: Option<(f64, String)> = None;
        // the defined names: the globals and the locals of the functions that are running
        let globals = self.globals.iter().enumerate().filter(|(_, slot)| slot.is_some()).map(|(i, _)| SymId(i as u32));
        let locals = (0..self.frames.len()).flat_map(|i| self.slots[self.frame_range(i)].iter().map(|(k, _)| *k));
        for id in globals.chain(locals) {
            let other = self.syms.name(id).split('`').next().unwrap_or("");
            if other == name || other.is_empty() {
                continue;
            }
            let cand: Vec<char> = other.chars().collect();
            let r = 2.0 * matches(&want, &cand) as f64 / (want.len() + cand.len()) as f64;
            if r >= 0.7 {
                // (the highest ratio; among equals the greater name, as `heapq.nlargest` on (ratio, name) gives)
                let better = match &best {
                    None => true,
                    Some((br, bn)) => r > *br || (r == *br && other > bn.as_str()),
                };
                if better {
                    best = Some((r, String::from(other)));
                }
            }
        }
        best.map(|(_, n)| crate::msg!("did you mean {}?", n))
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

    /// The value of a function literal: in a program that is being loaded, where its text is (`Func::Lazy`), when it is long enough for that to
    /// be less than its tree; otherwise the tree.
    #[inline]
    fn lazy_literal(tree: &Tree, id: NodeId) -> Option<(&Rc<crate::value::LazyCtx>, &Rc<crate::error::Source>, u32, u32, u8)> {
        if let (Some(ctx), Some(src), Node::Fn { a, arity, has_args: false, call: false, .. }) = (&tree.lazy, &tree.source, tree.node(id)) {
            if let Some((s, e)) = tree.span(id) {
                let has_params = matches!(tree.node(*a), Node::Body { has_params: true, .. });
                // (a function defined with named parameters, f(a;b)::{...}, has them from the definition, which is not in the text of the literal)
                if e - s >= LAZY_MIN && Tree::is_real(*a) && !has_params {
                    return Some((ctx, src, s, e, *arity));
                }
            }
        }
        None
    }

    #[inline(never)]
    fn literal_value(&self, tree: &Rc<Tree>, id: NodeId) -> Value {
        if let Some((ctx, src, start, end, arity)) = Self::lazy_literal(tree, id) {
            return Value::Fn(Rc::new(Func::Lazy(crate::value::Lazy { source: src.clone(), start, end, ctx: ctx.clone(), arity, kgfn: tree.is_kgfn(id), has_params: false })));
        }
        Value::Fn(Rc::new(Func::Code { tree: tree.clone(), node: id }))
    }

    /// What `resolve` makes of a function kept as text: its body, parsed now, and its arity (apart, so that the frame of `resolve`, which is
    /// on the way of every recursion, is the one it was).
    #[inline(never)]
    fn lazy_next(&mut self, l: &crate::value::Lazy) -> Result<Option<(Value, u8)>> {
        let (t, n) = self.materialize(l)?;
        Ok(match t.node(n) {
            Node::Fn { a, arity: fa, has_args: false, .. } => Some((self.node_value(&t, *a), *fa)),
            _ => None,
        })
    }

    /// `run_fn` of a function kept as text (apart for the same reason).
    #[inline(never)]
    fn run_lazy(&mut self, l: &crate::value::Lazy, f: &Value, n: usize) -> Result<Value> {
        let (tree, node) = match self.materialize(l) {
            Ok(t) => t,
            Err(e) => {
                self.drop_args(n);
                return Err(e);
            }
        };
        let saved = core::mem::replace(&mut self.cur_mod, tree.module.get());
        let base = core::mem::replace(&mut self.lex_base, 0);
        let r = match self.open_fn_frame(n, f.clone(), &[]) {
            Ok(()) => {
                let r = self.eval(&tree, node, true);
                self.pop_frame();
                r
            }
            Err(e) => Err(e),
        };
        self.lex_base = base;
        self.cur_mod = saved;
        r
    }

    /// The tree of a function kept as text: parsed from its text in the module and strict level it was, or the one parsed lately. It
    /// is let go at the end of the program (`release_caches`).
    #[inline(never)]
    fn materialize(&mut self, l: &crate::value::Lazy) -> Result<(Rc<Tree>, NodeId)> {
        let key = (Rc::as_ptr(&l.source) as usize, l.start);
        if let Ok(mut c) = self.lazy_cache.try_borrow_mut() {
            if let Some(i) = c.iter().position(|(p, s, _, _, _)| (*p, *s) == key) {
                // (the newest last: what is let go to make room is what was used longest ago)
                let e = c.remove(i);
                let r = (e.2.clone(), e.3);
                c.push(e);
                return Ok(r);
            }
        }
        let stack = self.stack();
        let saved = core::mem::replace(&mut self.module, l.ctx.module.clone());
        let mut window = (l.end - l.start) as usize + 1;
        let r = parse::parse_next(&l.source, l.start as usize, l.end as usize, &mut window, None, &mut self.syms, &mut self.module, l.ctx.strict, self.cfg.max_depth, stack);
        self.module = saved;
        let next = r?;
        let Some(p) = next.parsed else { return Err(Error::syntax("a function kept as text did not parse again")) };
        p.tree.module.set(l.ctx.mid);
        let node = p.stmts[0];
        if let Ok(mut c) = self.lazy_cache.try_borrow_mut() {
            // (about what the tree holds: a node and its span, the ids of its calls)
            let bytes = p.tree.nodes.len() * 40 + p.tree.extra.len() * 4 + 64;
            c.push((key.0, key.1, p.tree.clone(), node, bytes));
            let (mut n, mut b, mut ev) = self.lazy_stats.get();
            n += 1;
            b += bytes as u64;
            while c.len() > LAZY_ENTRIES || (c.len() > 1 && c.iter().map(|e| e.4).sum::<usize>() > LAZY_BUDGET) {
                c.remove(0);
                ev += 1;
            }
            self.lazy_stats.set((n, b, ev));
        }
        Ok((p.tree, node))
    }

    /// KlongPy `isinstance(v, (KGFn, KGLambda))`.
    fn is_fn_value(v: &Value) -> bool {
        match v {
            Value::Fn(f) => match &**f {
                Func::Code { tree, node } => tree.is_kgfn(*node),
                Func::Lazy(l) => l.kgfn,
                _ => true,
            },
            _ => false,
        }
    }

    /// Let go of what the evaluator keeps for speed: the recent function bodies (each holds its whole parse tree
    /// alive, so the tree of a library that was loaded and dropped stays) and the pool of argument layers.
    /// (parses, bytes of trees parsed, evictions) of the functions kept as text since the start.
    pub fn lazy_stats(&self) -> (u64, u64, u64) {
        self.lazy_stats.get()
    }

    /// Let go of the functions kept as text that were parsed lately (the library that was dropped holds nothing, then).
    pub fn release_lazy(&mut self) {
        if let Ok(mut c) = self.lazy_cache.try_borrow_mut() {
            *c = Vec::new();
        }
    }

    pub fn release_caches(&mut self) {
        if let Ok(mut c) = self.body_cache.try_borrow_mut() {
            *c = [None, None, None, None, None, None, None, None];
        }
        // the functions kept as text that were parsed lately stay for the next program, within a budget of bytes, unless memory is short
        // (a call of the management library a second would otherwise parse what it uses every time: 13 ms of a call of 23 on the boards)
        if self.cfg.mem_probe.is_some_and(|f| f()) {
            if let Ok(mut c) = self.lazy_cache.try_borrow_mut() {
                *c = Vec::new();
            }
        }
        self.layer_pool.clear();
        self.layer_pool.shrink_to_fit();
        // the tables indexed by symbol grow by doubling while a library loads: what a table has beyond its symbols is kept for nothing
        // (and the block it grew into is the last, the one that stays)
        compact(&mut self.globals);
        compact(&mut self.gscope);
        compact(&mut self.link);
        self.syms.shrink();
        if self.frames.is_empty() {
            self.frames = Vec::new();
            if self.slots.is_empty() {
                self.slots = Vec::new();
            }
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
    fn names_function(&mut self, v: &Value, name: SymId) -> Result<bool> {
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
            #[cfg(feature = "bigint")]
            Value::Big(_) => crate::msg!("an integer"),
            Value::Flt(x) if !x.is_finite() => crate::msg!(":undefined"),
            Value::Flt(_) => crate::msg!("a float"),
            Value::Char(_) => crate::msg!("a char"),
            Value::Sym(_) => crate::msg!("a symbol"),
            Value::Undefined => crate::msg!(":undefined"),
            Value::Str(_) => crate::msg!("a string"),
            Value::Dict(_) => crate::msg!("a dict"),
            #[cfg(feature = "host")]
            Value::Ext(e) if e.0.host_shape().is_some_and(|d| !d.is_empty()) => {
                let d = e.0.host_shape().unwrap_or_default();
                Self::list_kind(d.len(), d[0] as usize, &d[1..])
            }
            Value::Ext(_) => crate::msg!("a channel"),
            Value::Fn(_) | Value::Prim(_) => crate::msg!("a function"),
            Value::Ints(a) => Self::list_kind(a.rank(), a.len(), a.row_dims()),
            Value::Flts(a) => Self::list_kind(a.rank(), a.len(), a.row_dims()),
            Value::List(l) => Self::list_kind(1, l.len(), &[]),
        };
        let mut hint = String::from("a function is written {...}");
        #[cfg(feature = "host")]
        let hosted = matches!(v, Value::Ext(e) if e.0.host_shape().is_some_and(|d| !d.is_empty()));
        #[cfg(not(feature = "host"))]
        let hosted = false;
        if hosted || matches!(v, Value::Ints(_) | Value::Flts(_) | Value::List(_) | Value::Str(_) | Value::Dict(_)) {
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
        self.eval_value(tree, id, call, false)
    }

    #[inline(always)]
    fn eval_value(&mut self, tree: &Rc<Tree>, id: NodeId, call: bool, _forward_float: bool) -> Result<Value> {
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
            Ok(v) => {
                #[cfg(feature = "host")]
                if self.post {
                    if self.adopt.is_some() && matches!(v, Value::Ints(_) | Value::Flts(_)) {
                        return Ok(self.adopted(v));
                    }
                    // (a vector that is read from a variable or written in the program is not looked at again: what made it was rounded)
                    if self.f32_floats && ((!_forward_float && matches!(v, Value::Flt(_))) || (matches!(v, Value::Flts(_)) && !matches!(tree.node(id), Node::Sym(_) | Node::Lit(_)))) {
                        return Ok(Self::rounded_f32(v));
                    }
                }
                Ok(v)
            }
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

    /// `eval` for an operand or an argument: a literal or a name is read without the bookkeeping of `eval` (depth, stack, deferred failures and
    /// the span of an error), which a leaf needs none of once the deepest evaluation and the deepest stack so far are recorded; anything else,
    /// and every failure, goes through `eval`.
    #[cfg(feature = "speed")]
    #[inline(always)]
    fn eval_op(&mut self, tree: &Rc<Tree>, id: NodeId, call: bool) -> Result<Value> {
        #[cfg(feature = "host")]
        let plain = !self.post;
        #[cfg(not(feature = "host"))]
        let plain = true;
        // (only where `eval` would record nothing: no new deepest evaluation, no new deepest stack; the statistics and the place where a
        // limit is met are what they are without this)
        if plain
            && self.depth < self.cfg.max_depth
            && self.depth < self.max_depth_seen
            && self.stack_base.abs_diff(stack_here()) <= self.checked_stack_window()
            && Tree::is_real(id)
        {
            match tree.node(id) {
                Node::Lit(v) => return Ok(v.dup()),
                Node::Sym(s) => {
                    if let Ok(Some(v)) = self.lookup(*s) {
                        return Ok(v);
                    }
                }
                Node::Dyad { op, l, r, missing: false } => {
                    if let Some(v) = self.leaf_dyad(tree, *op, *l, *r) {
                        return Ok(v);
                    }
                }
                _ => {}
            }
        }
        self.eval(tree, id, call)
    }

    /// An arithmetic or comparison dyad whose operands are literals and names holding numbers, such as `x<2` and `n-1`: its value without the
    /// evaluation of the node (nothing it would record is new: the operands are read one level deeper than the node and far enough from the
    /// deepest stack so far for the frames of the node to be within it); `None`, with nothing done that matters, for anything else.
    #[cfg(feature = "speed")]
    #[inline(never)]
    fn leaf_dyad(&mut self, tree: &Rc<Tree>, op: SymId, l: NodeId, r: NodeId) -> Option<Value> {
        if self.depth + 1 >= self.max_depth_seen
            || self.depth + 1 >= self.cfg.max_depth
            || self.stack_base.abs_diff(stack_here()) + 1024 > self.checked_stack_window()
        {
            return None;
        }
        let def = ops::by_id(op)?;
        def.atomic?;
        let y = self.leaf_number(tree, r)?;
        let x = self.leaf_number(tree, l)?;
        let v = ops::scalar_pair(def.atomic, &x, &y);
        // (both are numbers: nothing to release)
        core::mem::forget(x);
        core::mem::forget(y);
        v
    }

    #[cfg(feature = "speed")]
    #[inline(always)]
    fn leaf_number(&mut self, tree: &Rc<Tree>, id: NodeId) -> Option<Value> {
        if !Tree::is_real(id) {
            return None;
        }
        match tree.node(id) {
            Node::Lit(v @ (Value::Int(_) | Value::Flt(_))) => Some(v.dup()),
            Node::Sym(s) => match self.lookup(*s) {
                Ok(Some(v @ (Value::Int(_) | Value::Flt(_)))) => Some(v),
                _ => None,
            },
            _ => None,
        }
    }

    #[cfg(not(feature = "speed"))]
    #[inline(always)]
    fn eval_op(&mut self, tree: &Rc<Tree>, id: NodeId, call: bool) -> Result<Value> {
        self.eval(tree, id, call)
    }

    /// HOST: `set_float_width(32)` computes floats as float32 does: a float atom is the float32 nearest to what the operation gave
    /// (the host's arrays are float32 and the same numbers must come out of both); 64 is the default and rounds nothing. A flat float
    /// vector that the host did not take over (`adopt`) is rounded too. Applies to what `eval` returns, so each operation of an
    /// expression is rounded, a literal as well.
    #[cfg(feature = "host")]
    pub fn set_float_width(&mut self, bits: u32) {
        self.f32_floats = bits == 32;
        self.post = self.f32_floats || self.adopt.is_some();
    }

    /// HOST: the embedder's takeover of numeric vectors (see `adopt`); `None` switches it off.
    #[cfg(feature = "host")]
    pub fn set_adopt(&mut self, h: Option<Rc<dyn Fn(&mut Interp, &Value) -> Option<Value>>>) {
        self.adopt = h;
        self.post = self.f32_floats || self.adopt.is_some();
    }

    #[cfg(feature = "host")]
    #[cold]
    #[inline(never)]
    fn rounded_f32(v: Value) -> Value {
        match v {
            Value::Flt(f) => Value::Flt(f as f32 as crate::value::Flt),
            // (a vector nobody else holds is rounded where it is)
            Value::Flts(mut a) => match Rc::get_mut(&mut a) {
                Some(arr) => {
                    arr.data.iter_mut().for_each(|x| *x = *x as f32 as crate::value::Flt);
                    Value::Flts(a)
                }
                None => match crate::value::rounded_f32(&a) {
                    Some(r) => Value::Flts(Rc::new(r)),
                    None => Value::Flts(a),
                },
            },
            v => v,
        }
    }

    #[cfg(feature = "host")]
    #[cold]
    #[inline(never)]
    fn adopted(&mut self, v: Value) -> Value {
        match self.adopt.clone() {
            Some(h) => h(self, &v).unwrap_or(v),
            None => v,
        }
    }

    #[inline(never)]
    #[cfg_attr(feature = "ram-hot", allow(unsafe_code), link_section = ".time_critical.klong_eval_node")]
    fn eval_node(&mut self, tree: &Rc<Tree>, id: NodeId, call: bool) -> Result<Value> {
        match tree.node(id) {
            Node::Lit(v) => Ok(v.dup()),
            Node::Sym(s) => match self.lookup(*s)? {
                Some(v) => Ok(v),
                None => self.undefined(*s),
            },
            Node::QSym(s) => self.quoted(*s),
            Node::OpVal { op, .. } => Ok(Value::Prim(*op)),
            Node::Monad { op, arg, missing } => self.eval_monad(tree, id, *op, *arg, *missing),
            Node::Dyad { op, l, r, missing } => {
                let (op, l, r, missing) = (*op, *l, *r, *missing);
                let res = self.eval_dyad(tree, id, op, l, r);
                if missing {
                    return res.map_err(|_| self.at_operator(tree, id, Error::syntax(crate::msg!("missing right operand for {}", self.syms.name(op)))));
                }
                res
            }
            Node::Fn { call: is_call, has_args, a, .. } => {
                // A call written with an empty argument, f(1;), is a projection, made now
                //. A callee that is not defined yet leaves the call as written:
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
                    Ok(self.literal_value(tree, id))
                }
            }
            Node::Adv { .. } => self.eval_adv(tree, id),
            Node::Cond(c, t, e) => {
                let q = self.eval_op(tree, *c, true)?;
                self.eval_op(tree, if q.truthy() { *t } else { *e }, true)
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
    fn eval_monad(&mut self, tree: &Rc<Tree>, id: NodeId, op: SymId, arg: NodeId, missing: bool) -> Result<Value> {
        let def = match ops::by_id(op) {
            Some(d) => d,
            // `˙` (Track) is the identity
            None if !missing && self.syms.name(op) == ops::TRACK => return self.eval(tree, arg, false),
            None => return Err(self.unknown_op(op)),
        };
        let f = def.monad.ok_or_else(|| self.unknown_op(op))?;
        let r = match self.eval_op(tree, arg, false) {
            Ok(x) => match &x {
                #[cfg(feature = "host")]
                Value::Ext(e) => match e.0.host_monad(self, def.glyph, &x) {
                    Some(r) => r,
                    None => f(self, x),
                },
                _ => f(self, x),
            }
            .map_err(|e| self.at_operator(tree, id, e)),
            Err(e) => Err(e),
        };
        if missing {
            return r.map_err(|_| self.at_operator(tree, id, Error::syntax(crate::msg!("missing operand for {}", self.syms.name(op)))));
        }
        r
    }

    /// HOST: `x\u{2207}f` takes its left operand as a name when it is one (the variable to perturb) and `f:>w` tells the host that
    /// the right operand was written as a name (`loss:>w` with a niladic `loss` differentiates with respect to `w`).
    #[cfg(feature = "host")]
    #[inline(never)]
    fn eval_gradient_dyad(&mut self, tree: &Rc<Tree>, id: NodeId, glyph: &str, l: NodeId, r: NodeId, y: Value) -> Result<Value> {
        let h = self.host_op.clone().ok_or_else(|| Error::internal("no host operator"))?;
        let res = if glyph == ":>" {
            let x = self.eval(tree, l, false)?;
            let named = match tree.node(r) {
                Node::Sym(s) => Value::Sym(*s),
                _ => Value::Undefined,
            };
            h(self, glyph, &[x, y, named])
        } else {
            let x = match tree.node(l) {
                Node::Sym(s) => Value::Sym(*s),
                _ => self.eval(tree, l, false)?,
            };
            h(self, glyph, &[x, y])
        };
        res.map_err(|e| self.at_operator(tree, id, e))
    }

    #[inline(never)]
    fn eval_elist(&mut self, tree: &Rc<Tree>, r: Range) -> Result<Value> {
        let mut out = try_vec(r.len as usize)?;
        for k in 0..r.len {
            let q = NodeId(tree.ids(r)[k as usize]);
            // (a function literal without parameters is a value here: [;{...}] is new, it has no classic meaning to keep)
            let pushed = self.eval(tree, q, !matches!(tree.node(q), Node::Fn { has_args: false, .. }))?;
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
            let vn = NodeId(ids[k + 1]);
            let val = self.eval(tree, vn, !matches!(tree.node(vn), Node::Fn { has_args: false, .. }))?;
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
            #[cfg(feature = "host")]
            if k + 1 == r.len && self.forward_ipc_float(tree, q) {
                return self.eval_value(tree, q, true, true);
            }
            last = self.eval_op(tree, q, true)?;
        }
        Ok(last)
    }

    #[cfg(feature = "host")]
    fn forward_ipc_float(&self, tree: &Tree, id: NodeId) -> bool {
        self.ipc_float_reads && self.f32_floats && Tree::is_real(id) && matches!(tree.node(id), Node::Sym(_))
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
        let def = self.op_def(op)?;
        let f = def.monad.ok_or_else(|| self.unknown_op(op))?;
        #[cfg(feature = "host")]
        if let Value::Ext(e) = &x {
            if let Some(r) = e.0.host_monad(self, def.glyph, &x) {
                return r;
            }
        }
        f(self, x)
    }

    pub fn dyad(&mut self, op: SymId, x: Value, y: Value) -> Result<Value> {
        let def = self.op_def(op)?;
        let f = def.dyad.ok_or_else(|| self.unknown_op(op))?;
        #[cfg(feature = "host")]
        if let Some(r) = self.host_numeric.as_ref().and_then(|h| h(def.glyph, &x, Some(&y))) {
            return r;
        }
        #[cfg(feature = "host")]
        if let Some(r) = self.ext_dyad(def.glyph, &x, &y) {
            return r;
        }
        f(self, x, y)
    }

    /// HOST: offer a dyad to the host object among its operands, if any.
    #[cfg(feature = "host")]
    #[inline]
    fn ext_dyad(&mut self, glyph: &str, x: &Value, y: &Value) -> Option<Result<Value>> {
        if let Value::Ext(e) = x {
            return e.0.host_dyad(self, glyph, x, y);
        }
        if let Value::Ext(e) = y {
            return e.0.host_dyad(self, glyph, x, y);
        }
        None
    }

    /// `l::r`: the right side is evaluated, then bound to the name or names of the left.
    #[cfg(any(feature = "speed", feature = "heap-tags"))]
    #[inline(never)]
    fn eval_define(&mut self, tree: &Rc<Tree>, id: NodeId, l: NodeId, r: NodeId) -> Result<Value> {
        let nothing = !Tree::is_real(r);
        #[cfg(feature = "speed")]
        if !nothing {
            if let Some(v) = self.append_assign(tree, id, l, r) {
                return v;
            }
        }
        let y = if nothing {
            Value::Undefined
        } else {
            // A source-backed global function outlives its transient parse. Place its retained
            // function object with the global table so it cannot pin the parse's free space.
            #[cfg(feature = "heap-tags")]
            let _tag = if Tree::is_real(l)
                && matches!(tree.node(l), Node::Sym(s) | Node::QSym(s)
                    if !self.in_fn() && self.syms.qual_base(*s).is_none() && !self.module_private_global(*s))
                && Self::lazy_literal(tree, r).is_some()
            {
                Some(crate::htag::Scope::new(crate::htag::GLOBALS))
            } else {
                None
            };
            self.eval_op(tree, r, false)?
        };
        let v = self.define(tree, l, y).map_err(|e| self.at_operator(tree, id, e));
        if nothing {
            self.void = true;
        }
        v
    }

    #[inline(never)]
    #[cfg_attr(feature = "ram-hot", allow(unsafe_code), link_section = ".time_critical.klong_eval_dyad")]
    fn eval_dyad(&mut self, tree: &Rc<Tree>, id: NodeId, op: SymId, l: NodeId, r: NodeId) -> Result<Value> {
        let def = self.op_def(op)?;
        let f = def.dyad.ok_or_else(|| self.unknown_op(op))?;
        #[cfg(any(feature = "speed", feature = "heap-tags"))]
        if def.glyph == "::" {
            return self.eval_define(tree, id, l, r);
        }
        // right to left
        // a right operand that is missing (strict level 0 warned of it): nothing, which is `:undefined` here, as KlongPy has None
        let nothing = !Tree::is_real(r);
        let y = if nothing { Value::Undefined } else { self.eval_op(tree, r, false)? };
        #[cfg(not(any(feature = "speed", feature = "heap-tags")))]
        if def.glyph == "::" {
            let v = self.define(tree, l, y).map_err(|e| self.at_operator(tree, id, e));
            if nothing {
                self.void = true;
            }
            return v;
        }
        #[cfg(feature = "host")]
        if self.host_op.is_some() && matches!(def.glyph, "\u{2207}" | ":>") {
            return self.eval_gradient_dyad(tree, id, def.glyph, l, r, y);
        }
        let x = self.eval_op(tree, l, false)?;
        #[cfg(feature = "host")]
        if let Some(r) = self.host_numeric.as_ref().and_then(|h| h(def.glyph, &x, Some(&y))) {
            return r.map_err(|e| self.at_operator(tree, id, e));
        }
        // two numbers (the call-heavy programs): the arithmetic and the comparisons without the pairing routine
        #[cfg(feature = "speed")]
        if let Some(v) = ops::scalar_pair(def.atomic, &x, &y) {
            // (both are numbers: nothing to release)
            core::mem::forget(x);
            core::mem::forget(y);
            return Ok(v);
        }
        // an operand that is a temporary array is the result's buffer
        #[cfg(feature = "speed")]
        let (x, y) = match crate::fast::array_owned(def.atomic, x, y) {
            Ok(v) => return Ok(v),
            Err(p) => p,
        };
        #[cfg(feature = "host")]
        if let Some(r) = self.ext_dyad(def.glyph, &x, &y) {
            return r.map_err(|e| self.at_operator(tree, id, e));
        }
        match def.atomic {
            Some(a) => ops::atomic(&self.syms, a, &x, &y),
            None => f(self, x, y),
        }
        .map_err(|e| self.at_operator(tree, id, e))
    }

    /// `v::v,y` where `v` is a name whose value is a string or a flat integer vector that nothing else holds, and `y` a string, a character
    /// or an integer (or a vector of them): the join is made in the value where it is, and not in a copy (a loop that builds a string or a
    /// vector by appending is otherwise quadratic). Nothing is evaluated when `None` is the answer. The depth, the order of evaluation
    /// (the right operand, then the name), the error and the value are as `eval` of the join and `define` give them.
    #[cfg(feature = "speed")]
    #[inline(never)]
    fn append_assign(&mut self, tree: &Rc<Tree>, id: NodeId, l: NodeId, r: NodeId) -> Option<Result<Value>> {
        let s = match tree.node(l) {
            Node::Sym(s) if Tree::is_real(l) => *s,
            _ => return None,
        };
        let (op, l2, r2) = match tree.node(r) {
            Node::Dyad { op, l: l2, r: r2, missing: false } if Tree::is_real(r) && Tree::is_real(*r2) => (*op, *l2, *r2),
            _ => return None,
        };
        if !matches!(tree.node(l2), Node::Sym(t) if *t == s) {
            return None;
        }
        let def = self.op_def(op).ok()?;
        if def.glyph != "," || sym::is_arg(s) || self.scopes.len() > 1 || self.cur_mod != 0 || self.journal.is_some() {
            return None;
        }
        #[cfg(feature = "host")]
        if self.post {
            return None;
        }
        // (the evaluation of the join is a level of nesting; where a limit would be met, `eval` is what reports it)
        if self.depth + 2 >= self.cfg.max_depth || self.check_stack().is_err() {
            return None;
        }
        self.depth += 1;
        if self.depth > self.max_depth_seen {
            self.max_depth_seen = self.depth;
        }
        let r = self.append_join(tree, id, l, r, l2, r2, s, def);
        self.depth -= 1;
        Some(r)
    }

    #[cfg(feature = "speed")]
    fn append_join(&mut self, tree: &Rc<Tree>, id: NodeId, l: NodeId, r: NodeId, l2: NodeId, r2: NodeId, s: SymId, def: &'static ops::OpDef) -> Result<Value> {
        let at_join = |e: Error| -> Error {
            let mut e = e;
            if e.span.is_none() {
                if let Some(sp) = tree.span(r) {
                    e.span = Some(sp);
                    e.source = tree.source.clone();
                }
            }
            e
        };
        let y = self.eval_op(tree, r2, false)?;
        if let Some(slot) = self.append_slot(s) {
            if Self::append_in_place(slot, &y) {
                return Ok(slot.dup());
            }
        }
        let x = self.eval_op(tree, l2, false)?;
        let f = def.dyad.ok_or_else(|| Error::internal("join has no dyad"))?;
        #[cfg(feature = "host")]
        if let Some(v) = self.ext_dyad(",", &x, &y) {
            return v.map_err(|e| self.at_operator(tree, r, e)).map_err(&at_join);
        }
        let v = f(self, x, y).map_err(|e| self.at_operator(tree, r, e)).map_err(&at_join)?;
        if let Some(e) = crate::value::take_deferred() {
            return Err(e);
        }
        self.define(tree, l, v).map_err(|e| self.at_operator(tree, id, e))
    }

    /// Where `assign` of the name `s` would store a value that is already bound: the slot in a frame or the global; `None` where it would do
    /// anything else, or the slot is unassigned.
    #[cfg(feature = "speed")]
    fn append_slot(&mut self, s: SymId) -> Option<&mut Value> {
        if self.syms.qual_base(s).is_some() {
            return None;
        }
        if self.shadow.get(s.0 as usize).copied().unwrap_or(0) > 0 {
            let cm = self.cur_mod;
            let tagged = !self.modules.is_empty();
            let base = self.lex_base;
            for i in (base..self.frames.len()).rev() {
                if tagged && self.fmods[i] != cm {
                    continue;
                }
                let r = self.frame_range(i);
                if let Some(p) = self.slots[r.clone()].iter().position(|(k, _)| *k == s) {
                    return self.slots[r.start + p].1.as_mut();
                }
            }
        }
        self.globals.get_mut(s.0 as usize)?.as_mut()
    }

    #[cfg(feature = "speed")]
    fn append_in_place(slot: &mut Value, y: &Value) -> bool {
        match (slot, y) {
            (Value::Str(a), Value::Str(b)) => Rc::get_mut(a).map_or(false, |t| t.append(b.as_str(), b.len())),
            (Value::Str(a), Value::Char(c)) => {
                let mut buf = [0u8; 4];
                Rc::get_mut(a).map_or(false, |t| t.append(c.encode_utf8(&mut buf), 1))
            }
            (Value::Ints(a), Value::Int(i)) if a.rank() == 1 => match Rc::get_mut(a) {
                Some(arr) => arr.data.try_reserve(1).is_ok() && {
                    arr.data.push(*i);
                    true
                },
                None => false,
            },
            (Value::Ints(a), Value::Ints(b)) if a.rank() == 1 && b.rank() == 1 && !Rc::ptr_eq(a, b) => match Rc::get_mut(a) {
                Some(arr) => arr.data.try_reserve(b.data.len()).is_ok() && {
                    arr.data.extend_from_slice(&b.data);
                    true
                },
                None => false,
            },
            _ => false,
        }
    }

    /// An error of an operator itself, as it comes without a position: the position of the operator (KlongPy `op_span`), not of the whole
    /// expression (an error from an operand, or from a function the operator called, has its own).
    #[cfg(not(feature = "diag"))]
    #[inline(always)]
    fn at_operator(&self, _tree: &Rc<Tree>, _id: NodeId, e: Error) -> Error {
        e
    }

    #[cfg(feature = "diag")]
    #[cold]
    #[inline(never)]
    fn at_operator(&self, tree: &Rc<Tree>, id: NodeId, mut e: Error) -> Error {
        if e.span.is_some() || matches!(e.kind, Kind::Interrupt | Kind::Memory) {
            return e;
        }
        if let Some(sp) = self.operator_span(tree, id) {
            e.span = Some(sp);
            e.source = tree.source.clone();
        }
        e
    }

    /// Where the operator of a monad, dyad or adverb-chain node is written.
    #[cfg(feature = "diag")]
    fn operator_span(&self, tree: &Tree, id: NodeId) -> Option<crate::error::Span> {
        let text = tree.source.as_ref()?.text.as_ref();
        let (st, _) = tree.span(id)?;
        match tree.node(id) {
            Node::Monad { op, .. } => {
                let g = self.syms.name(*op);
                text.get(st as usize..)?.starts_with(g).then(|| (st, st + g.len() as u32))
            }
            Node::Dyad { op, l, .. } => {
                let g = self.syms.name(*op);
                let (_, mut p) = tree.span(*l)?;
                let b = text.as_bytes();
                while (p as usize) < b.len() && (b[p as usize].is_ascii_whitespace() || b[p as usize] == b')') {
                    p += 1;
                }
                text.get(p as usize..)?.starts_with(g).then(|| (p, p + g.len() as u32))
            }
            Node::Adv { verb, advs, .. } => {
                let (vs, ve) = tree.span(*verb)?;
                let mut e = ve;
                for a in tree.ids(*advs) {
                    e += self.syms.name(SymId(*a)).len() as u32;
                }
                Some((vs, e))
            }
            _ => None,
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
                #[cfg(feature = "host")]
                let v = match &v {
                    Value::Ext(e) => match e.0.host_items() {
                        Some(r) => r?,
                        None => v,
                    },
                    _ => v,
                };
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
        // :undefined is the undefined value at every level
        if self.syms.name(s) == "undefined" {
            return Ok(Value::Undefined);
        }
        if self.cfg.strict == 2 || self.syms.qual_base(s).is_some() {
            // (a quoted `a.b` is a symbol: only an expression `a.b` reads an entry)
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
            // (`x.a` as the body of a function names an entry of its argument, which is bound only when the body runs: like `x`)
            let found = match self.lookup(s) {
                Err(e) if e.kind == Kind::Name && self.syms.qual_base(s).is_some_and(sym::is_arg) => None,
                r => r?,
            };
            match found {
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
                    if !sym::is_arg(s) && !self.syms.qual_base(s).is_some_and(sym::is_arg) {
                        let name = self.syms.name(s);
                        #[cfg(feature = "kgpy-only")]
                        if let Some(e) = self.needs_kgpy(s) {
                            return Err(e);
                        }
                        let e = Error::name(crate::msg!("undefined function: {}", name));
                        // a system function this build does not have (docs/native/index.md)
                        return Err(if name.starts_with('.') && cfg!(not(feature = "host")) { e.with_hint(crate::msg!("{} is not available on this build", name)) } else { e });
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
                                // ({:x} at level 2 is a function whose body is the symbol, not a function that is the symbol's value)
                                if !(self.cfg.strict == 2 && matches!(tree.node(*a), Node::QSym(_))) {
                                    next = Some((self.node_value(tree, *a), *fa));
                                }
                            } else if tree.ids(*args).contains(&NONE.0) {
                                try_push(layers, Layer::Tree(tree.clone(), *args))?;
                                next = Some((self.node_value(tree, *a), *fa));
                            }
                        }
                    }
                    Func::Lazy(l) => next = self.lazy_next(l)?,
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

    /// Whether a pass of `resolve` could change `f`: a name, a function literal or projection or a function kept as text. Native functions and
    /// code whose node is not a call node are what it leaves as they are.
    #[cfg(feature = "speed")]
    #[inline(always)]
    fn resolves_further(f: &Value) -> bool {
        match f {
            Value::Sym(_) => true,
            Value::Fn(func) => match &**func {
                Func::Code { tree, node } => matches!(tree.node(*node), Node::Fn { .. }),
                Func::Native { .. } => false,
                _ => true,
            },
            _ => false,
        }
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
            Value::Fn(f) => match &**f {
                Func::Native { name, .. } => String::from(self.syms.name(*name)),
                _ => String::from("the function"),
            },
            _ => String::from("the function"),
        };
        let what = match takes {
            0 => String::from("no arguments"),
            1 => String::from("1 argument"),
            n => crate::msg!("{} arguments", n),
        };
        Error::arity(crate::msg!("{} takes {}, got {}", name, what, given))
    }

    /// The named parameters of a function body, if it has any.
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

    /// The arity of a function written as a literal (`{x+y}`): how many arguments it takes; 0 for anything else.
    fn literal_arity(f: &Value) -> u8 {
        if let Value::Fn(func) = f {
            if let Func::Code { tree, node } = &**func {
                if let Node::Fn { has_args: false, arity, .. } = tree.node(*node) {
                    return *arity;
                }
            } else if let Func::Lazy(l) = &**func {
                return if l.has_params { 0 } else { l.arity };
            }
        }
        0
    }

    #[cfg(feature = "wide-args")]
    fn is_wild(f: &Value) -> bool {
        matches!(f, Value::Fn(func) if matches!(&**func, Func::Native { arity, .. } if *arity == WILDCARD))
    }

    /// Argument `k` (written as node `q`) of a call of `f` is a function literal with no parameters, and `f` is a system function that
    /// takes a function there (`.try`, `.async`, `.timer`): the literal is that function, not its value. Everywhere else a literal
    /// without parameters is called where it is written, as in classic Klong (`f({41})` hands `f` the number 41); a system function
    /// that wants a function could never have used that, so this takes a meaning that was an error (docs/language-additions.md).
    #[inline]
    fn is_function_slot(&self, tree: &Tree, f: &Value, k: usize, q: u32) -> bool {
        if !matches!(tree.node(NodeId(q)), Node::Fn { has_args: false, .. }) {
            return false;
        }
        let Value::Fn(func) = f else { return false };
        let Func::Native { name, .. } = &**func else { return false };
        match self.syms.name(*name) {
            ".try" => k == 0,
            ".async" => k < 2,
            ".timer" => k == 2,
            _ => false,
        }
    }

    fn is_native(f: &Value) -> bool {
        matches!(f, Value::Fn(func) if matches!(&**func, Func::Native { .. }))
    }

    fn eval_arg(&mut self, a: &Arg) -> Result<Value> {
        match a {
            Arg::Code(tree, id) => {
                // (an argument written in another module, of a projection completed here, is read where it was written)
                let m = tree.module.get();
                if m == self.cur_mod {
                    return self.eval(tree, *id, true);
                }
                let saved = core::mem::replace(&mut self.cur_mod, m);
                let r = self.eval(tree, *id, true);
                self.cur_mod = saved;
                r
            }
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
                    Func::Lazy(l) => !l.kgfn,
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

    /// The call `f(a;b)` of a name that holds a function literal, with as many arguments as the function takes: the same call as `eval_call_in`
    /// makes, without the layers and the passes of `resolve` (what they would do here is to find the body of the literal). `None`, with
    /// nothing evaluated, for every other call, and for a short or long one, which `eval_call_in` answers.
    #[cfg(feature = "speed")]
    #[inline(never)]
    fn call_named(&mut self, tree: &Rc<Tree>, id: NodeId) -> Option<Result<Value>> {
        let (a, args) = match tree.node(id) {
            Node::Fn { a, args, has_args: true, arity, literal: false, .. } if *arity > 0 => (*a, *args),
            _ => return None,
        };
        let s = match tree.node(a) {
            Node::Sym(s) if !sym::is_arg(*s) && self.syms.qual_base(*s).is_none() => *s,
            _ => return None,
        };
        let f = match self.lookup(s) {
            Ok(Some(f @ Value::Fn(_))) => f,
            _ => return None,
        };
        let Value::Fn(func) = &f else { return None };
        let Func::Code { tree: ft, node: fnode } = &**func else { return None };
        let (body, fa) = match ft.node(*fnode) {
            // (a body that is a name, a constant or a function literal is resolved further: not here)
            Node::Fn { a: body, has_args: false, arity: fa, .. }
                if Tree::is_real(*body) && !matches!(ft.node(*body), Node::Lit(_) | Node::Sym(_) | Node::QSym(_) | Node::OpVal { .. } | Node::Fn { .. }) =>
            {
                (*body, *fa)
            }
            _ => return None,
        };
        let ids = tree.ids(args);
        let n = ids.len();
        if ids.contains(&NONE.0) {
            return None;
        }
        let g = self.node_value(ft, body);
        if let Some((ptree, params)) = Self::params_of(&g) {
            if n != params.len as usize {
                return None;
            }
            for (k, &q) in ids.iter().enumerate() {
                let v = self.eval_op(tree, NodeId(q), true);
                let v = match self.args_or(k, v) {
                    Ok(v) => v,
                    Err(e) => return Some(Err(e)),
                };
                if let Err(e) = self.arg(k, SymId(ptree.ids(params)[k]), v) {
                    return Some(Err(e));
                }
            }
            return Some(self.run_fn(g, n));
        }
        #[cfg(feature = "wide-args")]
        if n > 3 {
            return None;
        }
        if n != fa as usize {
            return None;
            return None;
        }
        for (k, &q) in ids.iter().take(3).enumerate() {
            let v = self.eval_op(tree, NodeId(q), true);
            let v = match self.args_or(k, v) {
                Ok(v) => v,
                Err(e) => return Some(Err(e)),
            };
            if let Err(e) = self.arg(k, SymId(k as u32), v) {
                return Some(Err(e));
            }
        }
        Some(self.run_fn(g, n.min(3)))
    }

    /// Evaluate a call node (KlongPy `_eval_fn`). The arguments are expressions; they are
    /// evaluated here, once, after projections are merged.
    #[inline(never)]
    #[cfg_attr(feature = "ram-hot", allow(unsafe_code), link_section = ".time_critical.klong_eval_call")]
    fn eval_call(&mut self, tree: &Rc<Tree>, id: NodeId) -> Result<Value> {
        #[cfg(feature = "speed")]
        if let Some(r) = self.call_named(tree, id) {
            return r;
        }
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
        let (a, args, has_args, arity0, calling, literal) = match tree.node(id) {
            // (a written call, not the evaluation of a body that is a bare name, and not the call of a literal)
            Node::Fn { a, args, has_args, arity, literal, .. } => (*a, *args, *has_args, *arity, *has_args && !*literal, *literal),
            _ => return Err(Error::internal("not a call")),
        };
        if !has_args && self.cfg.strict == 2 {
            if let Node::QSym(s) = tree.node(a) {
                // {:foo}: the body is a quoted symbol, and this is its evaluation
                return Ok(Value::Sym(*s));
            }
        }
        if literal && self.cfg.strict == 2 {
            if let Node::QSym(s) = tree.node(a) {
                // {:x}(1): a function whose body is the quoted symbol; it gives the symbol (the arguments are not read)
                return Ok(Value::Sym(*s));
            }
        }
        let mut f = self.node_value(tree, a);
        let mut arity = arity0;
        let mut node = false;
        try_push(layers, if has_args { Layer::Tree(tree.clone(), args) } else { Layer::None })?;
        // three passes as there are at most three arguments: x, y and z
        self.resolve(&mut f, layers, &mut arity, &mut node, calling)?;
        // (a pass over a function that is its own callee changes nothing: a native function, or code that is not a call node)
        #[cfg(feature = "speed")]
        for _ in 0..2 {
            if !Self::resolves_further(&f) {
                break;
            }
            self.resolve(&mut f, layers, &mut arity, &mut node, calling)?;
        }
        #[cfg(not(feature = "speed"))]
        {
            self.resolve(&mut f, layers, &mut arity, &mut node, calling)?;
            self.resolve(&mut f, layers, &mut arity, &mut node, calling)?;
        }

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
                for (k, &q) in ids.iter().enumerate() {
                    let v = self.eval_op(tree, NodeId(q), true);
                    let v = self.args_or(k, v)?;
                    self.arg(k, SymId(ptree.ids(params)[k]), v)?;
                }
                return self.run_fn(f, n);
            }
            // `f()`: written with no argument, a function of one to three arguments is called short, which is a projection
            let arity = if n == 0 && calling { arity.max(Self::literal_arity(&f)) } else { arity };
            if n < arity as usize || ids.contains(&NONE.0) {
                // a short call (fewer arguments than the function takes, or an argument left
                // empty): a projection, at every strict level
                return self.projection(tree, id, None, arity as usize);
            }
            if n > arity as usize && self.cfg.strict == 2 && !Self::is_native(&f) {
                return Err(self.too_many(&self.node_value(tree, a), arity as usize, n));
            }
            for (k, &q) in ids.iter().take(3).enumerate() {
                let v = self.eval_op(tree, NodeId(q), !self.is_function_slot(tree, &f, k, q));
                let v = self.args_or(k, v)?;
                self.arg(k, SymId(k as u32), v)?;
            }
            #[cfg(feature = "wide-args")]
            if n > 3 && Self::is_wild(&f) {
                // HOST: a system function of any arity is given every argument written
                let mut extra = Vec::new();
                for &q in &ids[3..] {
                    let v = self.eval(tree, NodeId(q), true);
                    extra.push(self.args_or(3, v)?);
                }
                self.extra_args = extra;
            }
            return self.run_fn(f, n.min(3));
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
            let mut c = 0;
            for (k, q) in merged.iter().enumerate() {
                if let Some(a) = q {
                    let v = self.eval_arg(a);
                    let v = self.args_or(c, v)?;
                    self.arg(c, SymId(ptree.ids(params)[k]), v)?;
                    c += 1;
                }
            }
            return self.run_fn(f, c);
        }
        if n < arity as usize || holes > 0 {
            return self.projection(tree, id, Some(holes), arity as usize);
        }
        if n > arity as usize && self.cfg.strict == 2 && !Self::is_native(&f) {
            return Err(self.too_many(&callee, arity as usize, n));
        }
        let mut c = 0;
        for (k, q) in merged.iter().take(3).enumerate() {
            if let Some(a) = q {
                let v = self.eval_arg(a);
                let v = self.args_or(c, v)?;
                self.arg(c, SymId(k as u32), v)?;
                c += 1;
            }
        }
        #[cfg(feature = "wide-args")]
        if n > 3 && Self::is_wild(&f) {
            let mut extra = Vec::new();
            for a in merged.iter().skip(3).flatten() {
                let v = self.eval_arg(a);
                extra.push(self.args_or(c, v)?);
            }
            self.extra_args = extra;
        }
        self.run_fn(f, c)
    }

    /// The callee to store in a projection: as written, except that x, y, z and .f are
    /// replaced by what they hold now (they mean something else when it is completed).
    fn projected_callee(&mut self, a: Value) -> Value {
        if let Value::Sym(s) = a {
            // (a projection made in a module, or of a qualified name, holds the function: it is completed wherever it was passed to)
            if sym::is_arg(s) || s == sym::DOT_F || self.cur_mod != 0 || self.syms.qual_base(s).is_some() {
                if let Ok(Some(v)) = self.lookup(s) {
                    if sym::is_arg(s) || s == sym::DOT_F || Self::is_fn_value(&v) {
                        return v;
                    }
                }
            }
        }
        a
    }

    /// The value of the written call `id` when it does not supply every argument (KlongPy
    /// `_projection`): a projection with an empty slot for each argument still
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
        if !self.frames.is_empty() || self.cur_mod != 0 {
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
        #[cfg(feature = "host")]
        if self.running == 0 {
            // HOST: a call made from outside an evaluation (another thread, Python) is a program of its own: it sets the stack
            // base as `run` does
            return self.apply_top(f, values, arity);
        }
        self.apply_in(f, values, arity)
    }

    /// Apply a typed wire request, forwarding plain name results without
    /// narrowing scalar values to the host's arithmetic width.
    #[cfg(feature = "host")]
    pub fn apply_ipc(&mut self, f: &Value, values: Vec<Value>) -> Result<Value> {
        let saved = core::mem::replace(&mut self.ipc_float_reads, true);
        let result = self.apply(f, values, None);
        self.ipc_float_reads = saved;
        result
    }

    #[cfg(feature = "host")]
    #[cold]
    #[inline(never)]
    fn apply_top(&mut self, f: &Value, values: Vec<Value>, arity: Option<u8>) -> Result<Value> {
        crate::value::set_max_nest(self.cfg.max_nest);
        self.stack_base = stack_here();
        let _ = crate::value::take_deferred();
        self.running += 1;
        let r = self.apply_in(f, values, arity);
        self.running -= 1;
        if self.running == 0 {
            self.release_caches();
        }
        r
    }

    /// The body of the function literal `f` when it is called with `given` arguments the short way: a literal of that arity, whose body needs
    /// nothing of `resolve` (a body that is a call with no empty argument resolves to itself, any other function literal or a quoted symbol
    /// under strict 2 is its business), and which has no named parameters.
    #[cfg(feature = "speed")]
    pub(crate) fn fast_body(&mut self, f: &Value, given: usize) -> Option<Value> {
        if let Value::Fn(func) = f {
            if let Func::Code { tree, node: id } = &**func {
                if let Node::Fn { a, has_args: false, arity, .. } = tree.node(*id) {
                    let plain = match tree.node(*a) {
                        Node::Fn { has_args, args, .. } => *has_args && !tree.ids(*args).contains(&NONE.0),
                        Node::QSym(_) => self.cfg.strict != 2,
                        _ => true,
                    };
                    #[cfg(feature = "host")]
                    let plain = plain && self.running != 0;
                    if *arity as usize == given && plain {
                        let g = self.node_value(tree, *a);
                        if Self::params_of(&g).is_none() {
                            return Some(g);
                        }
                    }
                }
            }
        }
        None
    }

    #[cfg(not(feature = "speed"))]
    #[inline(always)]
    pub(crate) fn fast_body(&mut self, _f: &Value, _given: usize) -> Option<Value> {
        None
    }

    /// Run the body `g` that `fast_body` found with the argument or arguments bound to `x` and `y`.
    #[cfg(feature = "speed")]
    #[inline]
    pub(crate) fn run_body(&mut self, g: Value, x: Value, y: Option<Value>) -> Result<Value> {
        self.arg(0, SymId(0), x)?;
        let mut n = 1;
        if let Some(y) = y {
            self.arg(1, SymId(1), y)?;
            n = 2;
        }
        self.run_fn(g, n)
    }

    /// `apply` of one or two values, for the adverbs: a function literal called with as many arguments as it takes (the common call of
    /// `{x*x}'a`, `{x+y}/a`) skips the argument list, the layers and the three resolutions and runs on a frame from the pool. Anything
    /// else is `apply` as it was.
    pub(crate) fn apply_fast(&mut self, f: &Value, x: Value, y: Option<Value>) -> Result<Value> {
        #[cfg(feature = "speed")]
        if let Some(g) = self.fast_body(f, 1 + y.is_some() as usize) {
            return self.run_body(g, x, y);
        }
        let mut v = try_vec(2)?;
        v.push(x);
        if let Some(y) = y {
            v.push(y);
        }
        self.apply(f, v, None)
    }

    fn apply_in(&mut self, f: &Value, values: Vec<Value>, arity: Option<u8>) -> Result<Value> {
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
                for (k, v) in values.into_iter().enumerate() {
                    self.arg(k, SymId(ptree.ids(params)[k]), v)?;
                }
                return self.run_fn(g, given);
            }
            if given < f_arity as usize {
                // a short call is a projection
                return self.value_projection(f, values, f_arity as usize - given);
            }
            if given > f_arity as usize && self.cfg.strict == 2 && !Self::is_native(&g) {
                return Err(self.too_many(f, f_arity as usize, given));
            }
            #[cfg(feature = "wide-args")]
            let mut values = values;
            #[cfg(feature = "wide-args")]
            if given > 3 && Self::is_wild(&g) {
                self.extra_args = values.split_off(3);
            }
            for (k, v) in values.into_iter().take(3).enumerate() {
                self.arg(k, SymId(k as u32), v)?;
            }
            return self.run_fn(g, given.min(3));
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
            let mut c = 0;
            for (k, q) in merged.iter().enumerate() {
                if let Some(a) = q {
                    let v = self.eval_arg(a);
                    let v = self.args_or(c, v)?;
                    self.arg(c, SymId(ptree.ids(params)[k]), v)?;
                    c += 1;
                }
            }
            return self.run_fn(g, c);
        }
        if n < f_arity as usize || holes > 0 {
            // the values do not fill the projection: a projection over the slots that are left
            return self.value_projection(f, keep, holes);
        }
        let mut c = 0;
        for (k, q) in merged.iter().take(3).enumerate() {
            if let Some(a) = q {
                let v = self.eval_arg(a);
                let v = self.args_or(c, v)?;
                self.arg(c, SymId(k as u32), v)?;
                c += 1;
            }
        }
        #[cfg(feature = "wide-args")]
        if n > 3 && Self::is_wild(&g) {
            let mut extra = Vec::new();
            for a in merged.iter().skip(3).flatten() {
                let v = self.eval_arg(a);
                extra.push(self.args_or(c, v)?);
            }
            self.extra_args = extra;
        }
        self.run_fn(g, c)
    }

    /// Run a resolved function with its arguments bound in `frame` (KlongPy `_run_fn`): declare
    /// its locals, set `.f`, evaluate the body in a new frame.
    #[inline(never)]
    #[cfg_attr(feature = "ram-hot", allow(unsafe_code), link_section = ".time_critical.klong_run_fn")]
    fn run_fn(&mut self, f: Value, n: usize) -> Result<Value> {
        // (the `n` arguments are on top of the slots: every way out gives them back or makes them the frame, which `pop_frame` gives back)
        if let Err(e) = self.tick() {
            self.drop_args(n);
            return Err(e);
        }
        let func = match &f {
            Value::Fn(func) => func.clone(),
            Value::Sym(s) => {
                let s = *s;
                return self.run_name(f, s, n);
            }
            _ => {
                self.drop_args(n);
                return Ok(f);
            }
        };
        match &*func {
            Func::Code { tree, node } => match tree.node(*node) {
                Node::Body { stmts, locals, has_params, .. } => {
                    let locals = tree.ids(*locals);
                    let saved = core::mem::replace(&mut self.cur_mod, tree.module.get());
                    // (a function with named parameters sees its own frame and not the callers': nothing between here and the restore returns early)
                    if let Err(e) = self.open_fn_frame(n, f, locals) {
                        self.cur_mod = saved;
                        return Err(e);
                    }
                    let base = core::mem::replace(&mut self.lex_base, if *has_params { self.frames.len() - 1 } else { 0 });
                    let r = self.eval_seq(tree, *stmts);
                    self.lex_base = base;
                    self.pop_frame();
                    self.cur_mod = saved;
                    r
                }
                _ => {
                    let saved = core::mem::replace(&mut self.cur_mod, tree.module.get());
                    if let Err(e) = self.open_fn_frame(n, f, &[]) {
                        self.cur_mod = saved;
                        return Err(e);
                    }
                    let base = core::mem::replace(&mut self.lex_base, 0);
                    #[cfg(feature = "host")]
                    let r = if self.forward_ipc_float(tree, *node) {
                        self.eval_value(tree, *node, true, true)
                    } else {
                        self.eval(tree, *node, true)
                    };
                    #[cfg(not(feature = "host"))]
                    let r = self.eval(tree, *node, true);
                    self.lex_base = base;
                    self.pop_frame();
                    self.cur_mod = saved;
                    r
                }
            },
            Func::Lazy(l) => self.run_lazy(l, &f, n),
            Func::Native { arity, f: native, name, .. } => {
                let (arity, native, name) = (*arity, native.clone(), *name);
                self.run_native(n, arity, native, name)
            }
            // a projection that was given nothing more is the value it is
            Func::Proj { .. } => {
                self.drop_args(n);
                Ok(f)
            }
        }
    }

    /// The arms of `run_fn` that are not on the way of a recursion of Klong code, apart (see `eval_monad`).
    /// A symbol in call position that names no function: its value as a name.
    #[inline(never)]
    fn run_name(&mut self, f: Value, s: SymId, n: usize) -> Result<Value> {
        self.open_fn_frame(n, f, &[])?;
        let r = match self.lookup(s) {
            Ok(Some(v)) => Ok(v),
            Ok(None) => self.undefined(s),
            Err(e) => Err(e),
        };
        self.pop_frame();
        r
    }

    /// A native function called with the `n` arguments on top of the slots (which are given back when it returns).
    #[inline(never)]
    fn run_native(&mut self, n: usize, arity: u8, native: NativeFn, name: SymId) -> Result<Value> {
        let r = self.run_native_in(n, arity, native, name);
        self.drop_args(n);
        r
    }

    fn run_native_in(&mut self, n: usize, arity: u8, native: NativeFn, name: SymId) -> Result<Value> {
        let mut args: Vec<Value> = Vec::new();
        let start = self.slots.len() - n;
        for (k, v) in self.slots[start..].iter() {
            if sym::is_arg(*k) {
                if let Some(v) = v {
                    try_push(&mut args, v.clone())?;
                }
            }
        }
        #[cfg(feature = "wide-args")]
        if !self.extra_args.is_empty() {
            args.append(&mut self.extra_args);
        }
        if self.cfg.strict == 2 && arity != WILDCARD && args.len() > arity as usize {
            return Err(self.too_many(&Value::Sym(name), arity as usize, args.len()));
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
        let verb_val = self.node_value(tree, verb);
        #[cfg(feature = "speed")]
        let body = if matches!(tree.node(verb), Node::OpVal { .. }) { None } else { self.fast_body(&verb_val, verb_arity as usize) };
        let ch = Chain { tree, verb, verb_arity, verb_val, #[cfg(feature = "speed")] body, advs, first_arity };
        let last_arity = if advs.len() == 1 { first_arity } else { 1 };
        if last_arity == 1 {
            let x = if dyad {
                // legacy: the pair of operands is evaluated as a statement list
                self.eval(tree, l, true)?;
                self.eval(tree, r, true)?
            } else {
                self.eval(tree, r, false)?
            };
            self.chain_call(&ch, advs.len(), x, None).map_err(|e| self.at_operator(tree, id, e))
        } else {
            let x = self.eval(tree, l, false)?;
            let y = self.eval(tree, r, false)?;
            self.chain_call(&ch, advs.len(), x, Some(y)).map_err(|e| self.at_operator(tree, id, e))
        }
    }

    /// The error of an adverb chain (KlongPy `_chain_error`): an operator error that came without a position of its own (the verb
    /// applied by the adverb itself) names the whole chain and its operands: `cannot apply +/ (Plus-Over) to list of 2`.
    #[cfg(feature = "diag")]
    #[cold]
    #[inline(never)]
    pub(crate) fn chain_error(&self, ch: &Chain, operands: &[&Value], e: Error) -> Error {
        let mut e = e;
        let tree = ch.tree;
        if e.generic && e.span.is_none() {
            let (mut glyph, mut name) = match tree.node(ch.verb) {
                Node::OpVal { op, .. } => match ops::by_id(*op) {
                    Some(d) => (String::from(d.glyph), String::from(if ch.verb_arity == 1 { d.monad_name } else { d.dyad_name })),
                    None => (String::from(self.syms.name(*op)), String::new()),
                },
                _ => (String::from("f"), String::new()),
            };
            for (i, a) in ch.advs.iter().enumerate() {
                let g = self.syms.name(SymId(*a));
                glyph.push_str(g);
                let two = if i == 0 { ch.first_arity == 2 } else { false };
                let n = match (g, two) {
                    ("/", false) => "Over",
                    ("/", true) => "Over-Neutral",
                    ("\\", false) => "Scan-Over",
                    ("\\", true) => "Scan-Over-Neutral",
                    ("'", false) => "Each",
                    ("'", true) => "Each-2",
                    (":'", _) => "Each-Pair",
                    (":\\", _) => "Each-Left",
                    (":/", _) => "Each-Right",
                    (":*", _) => "Iterate",
                    ("\\~", false) => "Scan-Converging",
                    ("\\~", true) => "Scan-While",
                    _ => "",
                };
                if !n.is_empty() {
                    if !name.is_empty() {
                        name.push('-');
                    }
                    name.push_str(n);
                }
            }
            e = ops::whole_error(e, (&glyph, &name), operands, Some(&self.syms));
        }
        e
    }

    /// Leaf evaluation may skip bookkeeping only within both the observed peak and the current budget.
    #[cfg(feature = "speed")]
    #[inline(always)]
    fn checked_stack_window(&self) -> usize {
        self.max_stack_seen.min(self.cfg.stack_limit.unwrap_or(usize::MAX))
    }

    /// The native stack against `Config::stack_limit`: called wherever the evaluator recurses
    /// (a nested evaluation, a level of an adverb chain).
    #[cfg_attr(feature = "speed", inline(always))]
    #[cfg_attr(not(feature = "speed"), inline)]
    pub(crate) fn check_stack(&mut self) -> Result<()> {
        let used = self.stack_base.abs_diff(stack_here());
        self.max_stack_seen = self.max_stack_seen.max(used);
        if self.cfg.stack_limit.is_some_and(|limit| used > limit) {
            return Err(Error::range("expression nested too deeply").with_hint("the native stack budget of this build is used up"));
        }
        Ok(())
    }

    /// Call level `level` of a chain. Level zero is the verb and level i applies the first i adverbs.
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
                        #[cfg(feature = "host")]
                        if let Some(r) = self.ext_dyad(def.glyph, &x, &y) {
                            return r;
                        }
                        match def.atomic {
                            Some(a) => ops::atomic(&self.syms, a, &x, &y),
                            None => self.dyad(*op, x, y),
                        }
                    }
                    _ => Err(Error::typ("the verb of an adverb was given the wrong number of operands")),
                },
                Node::OpVal { op, .. } => Err(self.unknown_op(*op)),
                _ => match (ch.verb_arity, y) {
                    #[cfg(feature = "speed")]
                    (1, None) if ch.body.is_some() => self.run_body(ch.body.clone().unwrap_or(Value::Undefined), x, None),
                    #[cfg(feature = "speed")]
                    (2, Some(y)) if ch.body.is_some() => self.run_body(ch.body.clone().unwrap_or(Value::Undefined), x, Some(y)),
                    (1, None) => self.apply_fast(&ch.verb_val, x, None),
                    (2, Some(y)) => self.apply_fast(&ch.verb_val, x, Some(y)),
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

/// What is known of a function value (`Interp::fn_info`).
#[cfg(feature = "diag")]
#[derive(Default)]
pub struct FnInfo {
    /// `function`, `projection`, `native` or `operator`.
    pub kind: &'static str,
    /// The number of arguments it takes (for a projection, the holes it still has).
    pub arity: u8,
    /// The named parameters of `f(a;b)::{...}`.
    pub params: Option<Vec<String>>,
    /// The locals declared by a leading `[a b]` in the body.
    pub locals: Vec<String>,
    /// The text of the function literal as written, and where.
    pub source: Option<String>,
    pub span: Option<crate::error::Span>,
    /// The same span in characters of the source text (`span` is in bytes).
    pub span_chars: Option<(usize, usize)>,
    pub projection: bool,
    /// A projection's arguments in order: a value, or `None` for a hole (an argument written as an expression is its source text).
    pub bound: Vec<Option<Value>>,
    /// What a projection or a call node calls.
    pub callee: Option<Value>,
    /// A system function's name.
    pub name: Option<String>,
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

#[cfg(test)]
mod stack_tests {
    use super::*;

    #[test]
    fn the_current_stack_budget_is_independent_of_an_earlier_peak() {
        let mut ip = Interp::new(Config::default(), Box::new(NoHooks));
        ip.cfg.stack_limit = Some(256);
        // Synthetic accounting values. No recursion or physical stack allocation is needed.
        ip.stack_base = stack_here().saturating_sub(16 * 1024);
        ip.max_stack_seen = usize::MAX;
        assert_eq!(ip.check_stack().unwrap_err().kind, Kind::Range);
        #[cfg(feature = "speed")]
        assert_eq!(ip.checked_stack_window(), 256);
    }
}
