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
use alloc::format;
use alloc::rc::Rc;
use alloc::string::String;
use alloc::vec::Vec;
use core::cell::RefCell;
use core::sync::atomic::{AtomicBool, Ordering};

use crate::error::{Error, Kind, Result};
use crate::ops;
use crate::parse::{self, Parsed};
use crate::sym::{self, Interner, SymId};
use crate::tree::{Node, NodeId, Range, Tree, NONE};
use crate::value::{try_vec, Dict, Func, NativeFn, Slot, Value, WILDCARD};

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
    /// Set from outside (an interrupt handler) to stop the evaluation. Must live in memory
    /// where atomics work (internal RAM on the chips of `ref-platforms.md`).
    pub interrupt: Option<&'static AtomicBool>,
    /// True when the allocator has passed its soft limit and is living on the reserve: the
    /// evaluation unwinds with a `memory` error at the next tick.
    pub mem_probe: Option<fn() -> bool>,
}

impl Default for Config {
    fn default() -> Config {
        Config { max_depth: 512, strict: 0, step_limit: None, interrupt: None, mem_probe: None }
    }
}

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

pub struct Interp {
    pub syms: Interner,
    pub cfg: Config,
    globals: Vec<Option<Value>>,
    frames: Vec<Frame>,
    /// Per symbol: how many bindings of it are on the frame stack.
    shadow: Vec<u32>,
    hooks: Box<dyn Hooks>,
    depth: u32,
    ticks: u64,
    module: Option<String>,
    /// High-water mark of `depth` (for sizing a board's stack).
    pub max_depth_seen: u32,
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
        Interp {
            syms: Interner::new(),
            cfg,
            globals: Vec::new(),
            frames: Vec::new(),
            shadow: Vec::new(),
            hooks,
            depth: 0,
            ticks: 0,
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

    pub fn set_global(&mut self, s: SymId, v: Value) {
        let i = s.0 as usize;
        if self.globals.len() <= i {
            self.globals.resize(i + 1, None);
        }
        self.globals[i] = Some(v);
    }

    pub fn global(&self, name: &str) -> Option<Value> {
        let s = self.syms.lookup(name)?;
        self.globals.get(s.0 as usize).cloned().flatten()
    }

    pub fn format(&self, v: &Value) -> String {
        crate::fmt::to_string(&self.syms, v)
    }

    pub fn parse(&mut self, text: &str, name: &str) -> Result<Parsed> {
        parse::parse(text, name, &mut self.syms, &mut self.module, self.cfg.strict, self.cfg.max_depth)
    }

    /// Parse and run a program: every statement in order; the value of the last one, or `None`
    /// for a program without statements.
    pub fn run(&mut self, text: &str, name: &str) -> Result<Option<Value>> {
        let p = self.parse(text, name)?;
        let mut last = None;
        for &s in &p.stmts {
            last = Some(self.eval_top(&p.tree, s)?);
        }
        Ok(last)
    }

    fn eval_top(&mut self, tree: &Rc<Tree>, s: NodeId) -> Result<Value> {
        let r = self.eval(tree, s, true);
        if r.is_err() {
            // unwind whatever the failed statement left behind
            while let Some(f) = self.frames.pop() {
                self.unshadow(&f);
            }
            self.depth = 0;
        }
        r
    }

    // ----- ticks ---------------------------------------------------------------------------

    #[inline]
    pub(crate) fn tick(&mut self) -> Result<()> {
        self.ticks += 1;
        if self.ticks & 4095 == 0 {
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

    fn push_frame(&mut self, f: Frame) -> Result<()> {
        self.frames.try_reserve(1).map_err(|_| Error::memory())?;
        let top = f.iter().map(|(s, _)| s.0 as usize).max().unwrap_or(0);
        self.grow_shadow(top)?;
        for (s, _) in &f {
            self.shadow[s.0 as usize] += 1;
        }
        self.frames.push(f);
        Ok(())
    }

    fn try_set_global(&mut self, s: SymId, v: Value) -> Result<()> {
        let i = s.0 as usize;
        if self.globals.len() <= i {
            let more = i + 1 - self.globals.len();
            self.globals.try_reserve(more).map_err(|_| Error::memory())?;
        }
        self.set_global(s, v);
        Ok(())
    }

    fn unshadow(&mut self, f: &Frame) {
        for (s, _) in f {
            if let Some(c) = self.shadow.get_mut(s.0 as usize) {
                *c = c.saturating_sub(1);
            }
        }
    }

    fn pop_frame(&mut self) {
        if let Some(f) = self.frames.pop() {
            self.unshadow(&f);
        }
    }

    /// The value of a name: the innermost frame that binds it, else the global.
    pub fn lookup(&self, s: SymId) -> Result<Option<Value>> {
        if self.shadow.get(s.0 as usize).copied().unwrap_or(0) > 0 {
            for f in self.frames.iter().rev() {
                for (k, v) in f.iter() {
                    if *k == s {
                        return match v {
                            Some(v) => Ok(Some(v.clone())),
                            None => Err(Error::name(format!("local variable {} is read before it is assigned", self.syms.name(s)))),
                        };
                    }
                }
            }
        }
        Ok(self.globals.get(s.0 as usize).cloned().flatten())
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
            if matches!(self.globals.get(s.0 as usize), Some(Some(_))) {
                self.globals[s.0 as usize] = Some(v);
                return Ok(());
            }
        }
        // x, y and z are the arguments of the running function and are assignable like its
        // locals (ticket 033); they are always bound in its own frame
        if self.cfg.strict >= 1 && !self.frames.is_empty() && !sym::is_arg(s) {
            let n = self.syms.name(s);
            return Err(Error::name(format!("undefined variable: {}", n))
                .with_hint(format!("declare it as a local in the parameter list, {{[{}]; ...}}, or define the global before the function is called", n)));
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
            f.push((s, Some(v)));
            self.shadow[i] += 1;
        }
        Ok(())
    }

    fn is_defined(&self, s: SymId) -> bool {
        self.shadow.get(s.0 as usize).copied().unwrap_or(0) > 0 || matches!(self.globals.get(s.0 as usize), Some(Some(_)))
    }

    /// The value of a name that is not defined.
    #[inline(never)]
    fn undefined(&mut self, s: SymId) -> Result<Value> {
        let level = self.cfg.strict;
        if level == 2 || (level == 1 && !self.frames.is_empty()) {
            return Err(Error::name(format!("undefined: {}", self.syms.name(s))));
        }
        Ok(Value::Sym(s))
    }

    // ----- evaluation --------------------------------------------------------------------------

    /// A tree node as a value in function position: a leaf is its value, a name is the symbol,
    /// anything else is the node itself.
    fn node_value(&self, tree: &Rc<Tree>, id: NodeId) -> Value {
        match tree.node(id) {
            Node::Lit(v) => v.clone(),
            Node::Sym(s) | Node::QSym(s) => Value::Sym(*s),
            Node::OpVal { op, .. } => Value::Prim(*op),
            _ => Value::Fn(Rc::new(Func::Code { tree: tree.clone(), node: id })),
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

    /// Evaluate a node. With `call` set, a function node is invoked (KlongPy `call`); without
    /// it a function literal or projection is the value (`eval`).
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
        let r = self.eval_node(tree, id, call);
        self.depth -= 1;
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
    fn eval_node(&mut self, tree: &Rc<Tree>, id: NodeId, call: bool) -> Result<Value> {
        match tree.node(id) {
            Node::Lit(v) => Ok(v.clone()),
            Node::CharLit(c) => {
                let mut b = [0u8; 4];
                Ok(Value::str(c.encode_utf8(&mut b).as_bytes().to_vec()))
            }
            Node::Sym(s) => match self.lookup(*s)? {
                Some(v) => Ok(v),
                None => self.undefined(*s),
            },
            Node::QSym(s) => self.quoted(*s),
            Node::OpVal { op, .. } => Ok(Value::Prim(*op)),
            Node::Monad { op, arg, missing } => {
                let def = self.op_def(*op)?;
                let f = def.monad.ok_or_else(|| self.unknown_op(*op))?;
                let r = match self.eval(tree, *arg, false) {
                    Ok(x) => f(self, x),
                    Err(e) => Err(e),
                };
                if *missing {
                    return r.map_err(|_| Error::syntax(format!("missing operand for {}", self.syms.name(*op))));
                }
                r
            }
            Node::Dyad { op, l, r, missing } => {
                let (op, l, r, missing) = (*op, *l, *r, *missing);
                let res = self.eval_dyad(tree, op, l, r);
                if missing {
                    return res.map_err(|_| Error::syntax(format!("missing right operand for {}", self.syms.name(op))));
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
            Node::EList(r) => {
                let mut out = try_vec(r.len as usize)?;
                for k in 0..r.len {
                    let q = NodeId(tree.ids(*r)[k as usize]);
                    out.push(self.eval(tree, q, true)?);
                }
                Value::from_items(out)
            }
            Node::EDict(r) => {
                let mut d = Dict::default();
                let ids = tree.ids(*r);
                let mut k = 0;
                while k + 1 < ids.len() {
                    let key = self.eval(tree, NodeId(ids[k]), true)?;
                    let val = self.eval(tree, NodeId(ids[k + 1]), true)?;
                    if key.is_list() || matches!(key, Value::Dict(_) | Value::Fn(_)) {
                        return Err(Error::typ(format!("a {} cannot be a dictionary key", key.describe())).with_hint("a key is a number, a string, a char or a symbol"));
                    }
                    d.set(key, val);
                    k += 2;
                }
                Ok(Value::Dict(Rc::new(RefCell::new(d))))
            }
            Node::DictLit(v) | Node::CopyList(v) => ops::deep_copy(v),
            Node::Seq(r) => self.eval_seq(tree, *r),
            Node::Body { stmts, .. } => self.eval_seq(tree, *stmts),
            Node::Destructure(_) => Err(Error::internal("a destructuring target was evaluated")),
        }
    }

    #[inline(never)]
    fn eval_seq(&mut self, tree: &Rc<Tree>, r: Range) -> Result<Value> {
        let mut last = Value::empty();
        for k in 0..r.len {
            let q = NodeId(tree.ids(r)[k as usize]);
            last = self.eval(tree, q, true)?;
        }
        Ok(last)
    }

    fn unknown_op(&self, op: SymId) -> Error {
        Error::syntax(format!("unknown operator {}", self.syms.name(op)))
    }

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
            Some(a) => ops::atomic(a, &x, &y),
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
                    return Err(Error::typ(format!("cannot destructure {} into {} name{}", v.describe(), n, if n != 1 { "s" } else { "" }))
                        .with_hint("the right side of [a b c]::value must be a list or a string"));
                }
                let m = v.len().unwrap_or(0);
                if m != n {
                    return Err(Error::shape(format!(
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
    fn resolve(&mut self, f: &mut Value, layers: &mut Vec<Layer>, arity: &mut u8, node: &mut bool) -> Result<()> {
        if let Value::Sym(s) = *f {
            match self.lookup(s)? {
                Some(v) => {
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
                            None => return Ok(()),
                        }
                    }
                }
                None => {
                    if !sym::is_arg(s) {
                        return Err(Error::name(format!("undefined function: {}", self.syms.name(s))));
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
                                layers.push(Layer::Tree(tree.clone(), *args));
                                next = Some((self.node_value(tree, *a), *fa));
                            }
                        }
                    }
                    Func::Proj { f: callee, args } => {
                        if args.iter().any(|q| matches!(q, Slot::Hole)) {
                            layers.push(Layer::Slots(args.clone()));
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

    fn layer_args(layer: Layer) -> Vec<Option<Arg>> {
        match layer {
            Layer::None => Vec::new(),
            Layer::Tree(tree, r) => tree.ids(r).iter().map(|&q| if q == NONE.0 { None } else { Some(Arg::Code(tree.clone(), NodeId(q))) }).collect(),
            Layer::Vals(v) => v.into_iter().map(|x| Some(Arg::Val(x))).collect(),
            Layer::Slots(v) => v
                .into_iter()
                .map(|q| match q {
                    Slot::Hole => None,
                    Slot::Val(x) => Some(Arg::Val(x)),
                    Slot::Code(t, id) => Some(Arg::Code(t, id)),
                })
                .collect(),
        }
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
        let mut sparse: Vec<bool> = layers.first().map(|l| l.iter().map(|q| q.is_some()).collect()).unwrap_or_default();
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
            n => format!("{} arguments", n),
        };
        Error::arity(format!("{} takes {}, got {}", name, what, given))
    }

    /// The named parameters of a function body, if it has any (ticket 061).
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
            self.resolve(&mut g, layers, &mut g_arity, &mut g_node)?;
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
    fn eval_call(&mut self, tree: &Rc<Tree>, id: NodeId) -> Result<Value> {
        let (a, args, has_args, arity0) = match tree.node(id) {
            Node::Fn { a, args, has_args, arity, .. } => (*a, *args, *has_args, *arity),
            _ => return Err(Error::internal("not a call")),
        };
        if !has_args && self.cfg.strict == 2 {
            if let Node::QSym(s) = tree.node(a) {
                // {:foo}: the body is a quoted symbol, and this is its evaluation
                return Ok(Value::Sym(*s));
            }
        }
        let callee = self.node_value(tree, a);
        let mut f = callee.clone();
        let mut arity = arity0;
        let mut node = false;
        let mut layers: Vec<Layer> = Vec::new();
        layers.push(if has_args { Layer::Tree(tree.clone(), args) } else { Layer::None });
        // three passes as there are at most three arguments: x, y and z
        self.resolve(&mut f, &mut layers, &mut arity, &mut node)?;
        self.resolve(&mut f, &mut layers, &mut arity, &mut node)?;
        self.resolve(&mut f, &mut layers, &mut arity, &mut node)?;

        if layers.len() == 1 {
            // the common case: no projection involved
            let ids = if has_args { tree.ids(args) } else { &[] };
            let n = ids.len();
            if let Some((ptree, params)) = Self::params_of(&f) {
                let m = params.len as usize;
                let given = ids.iter().filter(|&&q| q != NONE.0).count();
                if n != m {
                    if n > m || !node {
                        return Err(self.too_many(&callee, m, given));
                    }
                    return self.projection(tree, id, None, m);
                }
                if ids.contains(&NONE.0) {
                    return self.projection(tree, id, None, m);
                }
                let mut frame: Frame = try_vec(n + 2)?;
                for (k, &q) in ids.iter().enumerate() {
                    let v = self.eval(tree, NodeId(q), true)?;
                    frame.push((SymId(ptree.ids(params)[k]), Some(v)));
                }
                return self.run_fn(f, frame);
            }
            if n < arity as usize || ids.contains(&NONE.0) {
                // a short call (fewer arguments than the function takes, or an argument left
                // empty): a projection, at every strict level
                return self.projection(tree, id, None, arity as usize);
            }
            if n > arity as usize && self.cfg.strict == 2 && !Self::is_native(&f) {
                return Err(self.too_many(&callee, arity as usize, n));
            }
            let mut frame: Frame = try_vec(n.min(3) + 2)?;
            for (k, &q) in ids.iter().take(3).enumerate() {
                let v = self.eval(tree, NodeId(q), true)?;
                frame.push((SymId(k as u32), Some(v)));
            }
            return self.run_fn(f, frame);
        }

        if layers.len() == 4 {
            self.resolve_deeper(&mut f, &mut layers, &mut arity, &mut node)?;
        }
        let mut lists: Vec<Vec<Option<Arg>>> = layers.into_iter().map(Self::layer_args).collect();
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
            let mut frame: Frame = try_vec(n + 2)?;
            for (k, q) in merged.iter().enumerate() {
                if let Some(a) = q {
                    let v = self.eval_arg(a)?;
                    frame.push((SymId(ptree.ids(params)[k]), Some(v)));
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
        let mut frame: Frame = try_vec(n.min(3) + 2)?;
        for (k, q) in merged.iter().take(3).enumerate() {
            if let Some(a) = q {
                let v = self.eval_arg(a)?;
                frame.push((SymId(k as u32), Some(v)));
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
                slots.push(if q == NONE.0 { Slot::Hole } else { Slot::Val(self.eval(tree, NodeId(q), true)?) });
            }
        } else {
            let callee_is_arg = matches!(tree.node(a), Node::Sym(s) if sym::is_arg(*s));
            if pad == 0 && !callee_is_arg {
                return Ok(as_written);
            }
            for &q in ids {
                slots.push(if q == NONE.0 { Slot::Hole } else { Slot::Code(tree.clone(), NodeId(q)) });
            }
        }
        for _ in 0..pad {
            slots.push(Slot::Hole);
        }
        let callee = self.projected_callee(self.node_value(tree, a));
        Ok(Value::Fn(Rc::new(Func::Proj { f: callee, args: slots })))
    }

    /// The projection `apply` returns when the values do not complete `f`: the values, held as
    /// they are, and one empty slot for each argument still missing.
    fn value_projection(&mut self, f: &Value, values: Vec<Value>, holes: usize) -> Result<Value> {
        let mut slots: Vec<Slot> = try_vec(values.len() + holes)?;
        slots.extend(values.into_iter().map(Slot::Val));
        for _ in 0..holes {
            slots.push(Slot::Hole);
        }
        Ok(Value::Fn(Rc::new(Func::Proj { f: self.projected_callee(f.clone()), args: slots })))
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
        layers.push(Layer::Vals(values));
        self.resolve(&mut g, &mut layers, &mut f_arity, &mut node)?;
        self.resolve(&mut g, &mut layers, &mut f_arity, &mut node)?;
        self.resolve(&mut g, &mut layers, &mut f_arity, &mut node)?;

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
                    frame.push((SymId(ptree.ids(params)[k]), Some(v)));
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
                frame.push((SymId(k as u32), Some(v)));
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
        let mut lists: Vec<Vec<Option<Arg>>> = layers.into_iter().map(Self::layer_args).collect();
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
                    frame.push((SymId(ptree.ids(params)[k]), Some(v)));
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
                frame.push((SymId(k as u32), Some(v)));
            }
        }
        self.run_fn(g, frame)
    }

    /// Run a resolved function with its arguments bound in `frame` (KlongPy `_run_fn`): declare
    /// its locals, set `.f`, evaluate the body in a new frame.
    #[inline(never)]
    fn run_fn(&mut self, f: Value, mut frame: Frame) -> Result<Value> {
        self.tick()?;

        let func = match &f {
            Value::Fn(func) => func.clone(),
            Value::Sym(s) => {
                // a symbol in call position that names no function: its value as a name
                let s = *s;
                frame.push((sym::DOT_F, Some(f.clone())));
                self.push_frame(frame)?;
                let r = match self.lookup(s) {
                    Ok(Some(v)) => Ok(v),
                    Ok(None) => self.undefined(s),
                    Err(e) => Err(e),
                };
                self.pop_frame();
                return r;
            }
            _ => return Ok(f),
        };
        match &*func {
            Func::Code { tree, node } => {
                frame.push((sym::DOT_F, Some(f.clone())));
                let r = match tree.node(*node) {
                    Node::Body { stmts, locals, .. } => {
                        for &q in tree.ids(*locals) {
                            let s = SymId(q);
                            if !frame.iter().any(|(k, _)| *k == s) {
                                // an unassigned local is its own symbol; at strict level 2
                                // reading it is an error
                                frame.push((s, if self.cfg.strict == 2 { None } else { Some(Value::Sym(s)) }));
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
                let mut args: Vec<Value> = Vec::new();
                for (k, v) in frame.iter() {
                    if sym::is_arg(*k) {
                        if let Some(v) = v {
                            args.push(v.clone());
                        }
                    }
                }
                if self.cfg.strict == 2 && *arity != WILDCARD && args.len() > *arity as usize {
                    return Err(self.too_many(&Value::Undefined, *arity as usize, args.len()));
                }
                let native = native.clone();
                native(self, &args)
            }
            // a projection that was given nothing more is the value it is
            Func::Proj { .. } => Ok(f),
        }
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
    pub(crate) fn chain_call(&mut self, ch: &Chain, level: usize, x: Value, y: Option<Value>) -> Result<Value> {
        if level == 0 {
            return match ch.tree.node(ch.verb) {
                Node::OpVal { op, .. } if ops::by_id(*op).is_some() || self.syms.name(*op) == ops::TRACK => match (ch.verb_arity, y) {
                    (1, None) => self.monad(*op, x),
                    (2, Some(y)) => {
                        let def = self.op_def(*op)?;
                        match def.atomic {
                            Some(a) => ops::atomic(a, &x, &y),
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
        format!("error: {}", self.name())
    }
}
