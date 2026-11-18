//! Lowering to the typed IR, with type and shape inference and specialisation (`klongpy/static/infer.py`, docs/static-profile.md).
//!
//! `lower(report, values, name, int_bits)` takes the `Report` of a program the checker accepted with every global resolved and the
//! stage-one values of its globals, walks the same trees the checker walked from each `.entry` with the argument types of its
//! signature, and builds one `FnSpec` per function and tuple of argument types. It never evaluates anything. The type and shape of
//! every operator come from the columns of the operator table (`tables::Op`).
use crate::ast::{Kind, Lit, Node as A, Sp};
use crate::check::{validate_global, GlobalInfo, LitVal, Report};
use crate::fixed as fx;
use crate::ir::*;
use crate::sig::{int_range, is_fixed, parse_type, ExternDecl, Ty, MAX_RANK};
use crate::tables::{self, lookup, Op};
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

const MAX_PASSES: usize = 6;

/// A stage-one value of a global: what the checker is told, and the numbers (None when the value is not a number).
#[derive(Clone)]
pub struct StageValue {
    pub lit: LitVal,
    pub value: Option<V>,
}

/// A rejection found by lowering: the rule, where, and why.
#[derive(Debug, Clone)]
pub struct Fail {
    pub rule: &'static str,
    pub span: Sp,
    pub src: usize,
    pub message: String,
    pub hint: Option<String>,
}

type R<T> = Result<T, Fail>;

fn int() -> Ty {
    Ty::new("int", Vec::new())
}
fn float() -> Ty {
    Ty::new("float", Vec::new())
}
fn void() -> Ty {
    Ty::new("void", Vec::new())
}
fn str_ty() -> Ty {
    Ty::new("str", Vec::new())
}

fn promote(a: &str, b: &str) -> String {
    if is_fixed(a) {
        return a.to_string();
    }
    if is_fixed(b) {
        return b.to_string();
    }
    if a == "float" || b == "float" {
        "float".to_string()
    } else {
        "int".to_string()
    }
}

struct Var {
    kind: u8, // 0 param, 1 local
    index: usize,
    ty: Option<Ty>,
    hint: Option<String>,
    name: String,
}

struct Frame {
    parent: Option<usize>,
    fname: String,
    vars: HashMap<String, usize>,
    store: Vec<Var>,
    locals: Vec<usize>,
    captures: Vec<(String, N)>,
    n_own: usize,
    n_literals: usize,
}

/// Names a function body assigns (not looking into the function literals written in it).
fn assigned(stmts: &[Rc<A>]) -> HashSet<String> {
    let mut out = HashSet::new();
    fn go(n: &A, out: &mut HashSet<String>) {
        match &n.k {
            Kind::Cond(v) | Kind::EList(v) | Kind::EDict(v) | Kind::Body(v) => {
                for q in v {
                    go(q, out);
                }
            }
            Kind::Chain { l, r, dyadic, .. } => {
                if *dyadic {
                    go(l, out);
                }
                go(r, out);
            }
            Kind::Monad(_, a) => go(a, out),
            Kind::Dyad(g, l, r) => {
                if g == "::" {
                    match &l.k {
                        Kind::Destructure(names) => {
                            for t in names {
                                out.insert(t.clone());
                            }
                        }
                        Kind::Sym(s) => {
                            out.insert(s.clone());
                        }
                        _ => {}
                    }
                    go(r, out);
                } else {
                    go(l, out);
                    go(r, out);
                }
            }
            Kind::Call { callee, args, .. } => {
                go(callee, out);
                for q in args {
                    go(q, out);
                }
            }
            _ => {}
        }
    }
    for s in stmts {
        go(s, &mut out);
    }
    out
}

struct Lower<'a> {
    rep: &'a Report,
    values: &'a HashMap<String, StageValue>,
    name: String,
    int_bits: u32,
    lo: i128,
    hi: i128,
    memo: HashMap<(String, Vec<Ty>), F>,
    gdefs: Vec<GlobalDef>,
    gindex: HashMap<String, usize>,
    used_externs: HashSet<String>,
    src: usize,
    frames: Vec<Frame>,
}

impl<'a> Lower<'a> {
    fn fail<T>(&self, rule: &'static str, sp: Option<Sp>, message: String) -> R<T> {
        let r = tables::rule(rule).expect("a rule id");
        Err(Fail { rule: r.id, span: sp.unwrap_or((0, 0)), src: self.src, message, hint: r.hint.map(String::from) })
    }

    // -- constants -----------------------------------------------------------------
    fn ci(&self, v: i64, sp: Option<Sp>) -> N {
        Node::new(int(), sp, K::Const(V::I(v)))
    }
    fn cf(&self, v: f64, sp: Option<Sp>) -> N {
        Node::new(float(), sp, K::Const(V::F(v)))
    }
    fn const_v(&self, v: V, shape: Vec<usize>, sp: Option<Sp>) -> N {
        match v {
            V::I(i) => self.ci(i, sp),
            V::F(f) => self.cf(f, sp),
            V::AI(a) => Node::new(Ty::new("int", shape), sp, K::Const(V::AI(a))),
            V::AF(a) => Node::new(Ty::new("float", shape), sp, K::Const(V::AF(a))),
            V::S(s) => Node::new(str_ty(), sp, K::Const(V::S(s))),
        }
    }
    fn in_range(&self, v: i128) -> bool {
        self.lo <= v && v <= self.hi
    }

    fn conv(&self, n: &N, elem: &str) -> N {
        if is_fixed(elem) && n.ty.elem == "int" {
            return self.to_fixed(n, elem);
        }
        if n.ty.elem == elem || elem != "float" || n.ty.elem != "int" {
            return n.clone();
        }
        if let K::Const(v) = &n.k {
            let c = if n.ty.shape.is_empty() { V::F(v.as_f64()[0]) } else { V::AF(v.as_f64()) };
            return self.const_v(c, n.ty.shape.clone(), n.span);
        }
        Node::new(Ty::new("float", n.ty.shape.clone()), n.span, K::Convert { a: n.clone() })
    }

    /// n (an int, a float or a fixed-point value) as fixed-point `elem`: a constant when n is one.
    fn to_fixed(&self, n: &N, elem: &str) -> N {
        let spec = fx::spec_of(elem);
        let src = n.ty.elem.as_str();
        if src == elem {
            return n.clone();
        }
        if let K::Const(v) = &n.k {
            if !matches!(v, V::S(_)) {
                let raw: Vec<i64> = if is_fixed(src) {
                    let s = fx::spec_of(src);
                    v.as_i64().iter().map(|r| fx::convert(*r, s, spec)).collect()
                } else if src == "float" {
                    v.as_f64().iter().map(|x| fx::from_float(*x, spec)).collect()
                } else {
                    v.as_i64().iter().map(|x| fx::from_int(*x, spec)).collect()
                };
                let ty = Ty::new(elem, n.ty.shape.clone());
                let val = if n.ty.shape.is_empty() { V::I(raw[0]) } else { V::AI(raw) };
                return Node::new(ty, n.span, K::Const(val));
            }
        }
        Node::new(Ty::new(elem, n.ty.shape.clone()), n.span, K::Convert { a: n.clone() })
    }

    fn mk(&self, elem: &str, shape: &[usize], sp: Option<Sp>) -> R<Ty> {
        if shape.iter().any(|d| *d == 0) {
            return self.fail("array-shape", sp, "the result would have a dimension of 0: every dimension is a positive integer".to_string());
        }
        if shape.len() > MAX_RANK {
            return self.fail("array-shape", sp, format!("the result would have rank {}; the rank is 1 to {}", shape.len(), MAX_RANK));
        }
        Ok(Ty::new(elem, shape.to_vec()))
    }

    // -- globals -----------------------------------------------------------------------
    fn gdef(&mut self, name: &str) -> R<usize> {
        if let Some(&i) = self.gindex.get(name) {
            return Ok(i);
        }
        let info: &GlobalInfo = self.rep.globals.items.iter().find(|(k, _)| k == name).map(|(_, v)| v).unwrap();
        let sv = &self.values[name];
        let ty = match validate_global(&sv.lit, self.int_bits) {
            Ok(t) => t,
            Err((rule, message)) => {
                let r = tables::rule(rule).unwrap();
                return Err(Fail { rule: r.id, span: info.span, src: info.src, message: format!("{}: {}", name, message), hint: r.hint.map(String::from) });
            }
        };
        let raw = sv.value.clone().unwrap_or(V::I(0));
        let v = if is_fixed(&ty.elem) {
            if ty.shape.is_empty() {
                V::I(raw.as_i64()[0])
            } else {
                V::AI(raw.as_i64())
            }
        } else if !ty.shape.is_empty() {
            if ty.elem == "float" {
                V::AF(raw.as_f64())
            } else {
                V::AI(raw.as_i64())
            }
        } else if ty.elem == "float" {
            V::F(raw.as_f64()[0])
        } else {
            V::I(raw.as_i64()[0])
        };
        self.gdefs.push(GlobalDef { name: name.to_string(), ty, value: v, state: info.assigned, param: info.param });
        self.gindex.insert(name.to_string(), self.gdefs.len() - 1);
        Ok(self.gdefs.len() - 1)
    }

    fn global_ref(&mut self, name: &str, sp: Option<Sp>) -> R<N> {
        let gi = self.gdef(name)?;
        let g = &self.gdefs[gi];
        if g.state {
            return Ok(Node::new(g.ty.clone(), sp, K::GlobalRef { name: name.to_string(), state: true, known: None }));
        }
        if g.ty.shape.is_empty() {
            if is_fixed(&g.ty.elem) {
                return Ok(Node::new(g.ty.clone(), sp, K::Const(V::I(g.value.as_i64()[0]))));
            }
            let (v, shape) = (g.value.clone(), Vec::new());
            return Ok(self.const_v(v, shape, sp));
        }
        let known = if g.value.len() <= 4096 { Some(g.value.clone()) } else { None };
        Ok(Node::new(g.ty.clone(), sp, K::GlobalRef { name: name.to_string(), state: false, known }))
    }

    // -- functions ---------------------------------------------------------------------
    fn spec(&mut self, name: &str, arg_tys: &[Ty]) -> R<F> {
        let key = (name.to_string(), arg_tys.to_vec());
        if let Some(f) = self.memo.get(&key) {
            return Ok(f.clone());
        }
        let info = self.rep.functions.items.iter().find(|(k, _)| k == name).map(|(_, v)| v).unwrap();
        let saved = self.src;
        self.src = info.src;
        let r = self.literal(&info.node, arg_tys, None, name, "fn", Some(info.span));
        self.src = saved;
        let (f, _) = r?;
        self.memo.insert(key, f.clone());
        Ok(f)
    }

    fn new_frame(&mut self, parent: Option<usize>, fname: &str, n_own: usize) -> usize {
        self.frames.push(Frame { parent, fname: fname.to_string(), vars: HashMap::new(), store: Vec::new(), locals: Vec::new(), captures: Vec::new(), n_own, n_literals: 0 });
        self.frames.len() - 1
    }

    fn new_local(&mut self, fr: usize, name: &str, ty: Option<Ty>, hint: Option<String>) -> usize {
        let f = &mut self.frames[fr];
        let slot = f.locals.len();
        f.store.push(Var { kind: 1, index: slot, ty, hint, name: name.to_string() });
        let vi = f.store.len() - 1;
        f.locals.push(vi);
        vi
    }

    /// Lower a function literal at these argument types. Returns the FnSpec and the capture nodes in the parent's frame.
    fn literal(&mut self, node: &Rc<A>, arg_tys: &[Ty], parent: Option<usize>, name: &str, kind: &'static str, span: Option<Sp>) -> R<(F, Vec<N>)> {
        let Kind::FnLit { stmts, spans, params, locals: names } = &node.k else {
            return self.fail("fn-value", span, "a function literal is expected".to_string());
        };
        if params.len() != arg_tys.len() {
            return self.fail("arity", span, format!("{} takes {} argument(s); here it is given {}", name, params.len(), arg_tys.len()));
        }
        let asg = assigned(stmts);
        let default = node.span.or(span);
        let mut carried: HashMap<String, String> = HashMap::new();
        let base = self.frames.len();
        let mut result: Option<(Vec<N>, Vec<Option<Sp>>, usize)> = None;
        for _pass in 0..MAX_PASSES {
            self.frames.truncate(base);
            let fr = self.new_frame(parent, name, params.len());
            let mut pre: Vec<N> = Vec::new();
            for (i, (p, t)) in params.iter().zip(arg_tys).enumerate() {
                if asg.contains(p) {
                    let lt = Ty::new(if carried.get(p).map(|s| s == "float").unwrap_or(false) { "float" } else { &t.elem }, t.shape.clone());
                    let vi = self.new_local(fr, p, Some(lt.clone()), None);
                    self.frames[fr].vars.insert(p.clone(), vi);
                    let slot = self.frames[fr].store[vi].index;
                    let pr = Node::new(t.clone(), default, K::ParamRef { index: i, name: p.clone() });
                    let tgt = Node::new(lt.clone(), default, K::LocalRef { slot, name: p.clone() });
                    pre.push(Node::new(lt.clone(), default, K::Assign { target: tgt, value: self.conv(&pr, &lt.elem) }));
                } else {
                    let f = &mut self.frames[fr];
                    f.store.push(Var { kind: 0, index: i, ty: Some(t.clone()), hint: None, name: p.clone() });
                    let vi = f.store.len() - 1;
                    f.vars.insert(p.clone(), vi);
                }
            }
            for q in names {
                if !self.frames[fr].vars.contains_key(q) {
                    let vi = self.new_local(fr, q, None, carried.get(q).cloned());
                    self.frames[fr].vars.insert(q.clone(), vi);
                }
            }
            let mut body: Vec<N> = pre;
            let mut sps: Vec<Option<Sp>> = vec![None; body.len()];
            for (s, sp) in stmts.iter().zip(spans.iter()) {
                let n = self.ev(s, sp.or(default), fr)?;
                body.push(n);
                sps.push(*sp);
            }
            let mut now: HashMap<String, String> = HashMap::new();
            for &vi in &self.frames[fr].locals {
                let v = &self.frames[fr].store[vi];
                if let Some(t) = &v.ty {
                    now.insert(v.name.clone(), t.elem.clone());
                }
            }
            let done = now == carried;
            carried = now;
            result = Some((body, sps, fr));
            if done {
                break;
            }
            if _pass == MAX_PASSES - 1 {
                return self.fail("shape-change", default, format!("the types of the locals of {} do not settle", name));
            }
        }
        let (body, sps, fr) = result.unwrap();
        let rty = body.last().map(|b| b.ty.clone()).unwrap_or_else(void);
        let f = &self.frames[fr];
        let cap_tys: Vec<Ty> = f.captures.iter().map(|(_, n)| n.ty.clone()).collect();
        let cap_names: Vec<String> = f.captures.iter().map(|(n, _)| n.clone()).collect();
        let caps: Vec<N> = f.captures.iter().map(|(_, n)| n.clone()).collect();
        let loc_tys: Vec<Ty> = f.locals.iter().map(|&vi| f.store[vi].ty.clone().unwrap_or_else(int)).collect();
        let loc_names: Vec<String> = f.locals.iter().map(|&vi| f.store[vi].name.clone()).collect();
        self.frames.truncate(base);
        let mut ptys = arg_tys.to_vec();
        ptys.extend(cap_tys);
        let mut pnames = params.clone();
        pnames.extend(cap_names);
        let seq = Node::new(rty.clone(), default, K::Seq { stmts: body });
        let spec = FnSpec::new(name.to_string(), kind, ptys, params.len(), pnames, loc_tys, loc_names, seq, rty, span, self.src, sps);
        Ok((Rc::new(spec), caps))
    }

    fn op_spec(&mut self, glyph: &str, arg_tys: &[Ty], sp: Option<Sp>) -> R<F> {
        let ps: Vec<N> = arg_tys.iter().enumerate().map(|(i, t)| Node::new(t.clone(), sp, K::ParamRef { index: i, name: ["x", "y", "z"][i].to_string() })).collect();
        let body = if ps.len() == 1 { self.monad(glyph, ps[0].clone(), sp)? } else { self.dyad(glyph, ps[0].clone(), ps[1].clone(), sp)? };
        let rty = body.ty.clone();
        let names: Vec<String> = ["x", "y", "z"][..ps.len()].iter().map(|s| s.to_string()).collect();
        let seq = Node::new(rty.clone(), sp, K::Seq { stmts: vec![body] });
        Ok(Rc::new(FnSpec::new(glyph.to_string(), "op", arg_tys.to_vec(), ps.len(), names, Vec::new(), Vec::new(), seq, rty, sp, self.src, Vec::new())))
    }

    fn extern_spec(&mut self, decl: &Rc<ExternDecl>, arg_tys: &[Ty], sp: Option<Sp>) -> R<F> {
        let ps: Vec<N> = arg_tys
            .iter()
            .enumerate()
            .map(|(i, t)| Node::new(t.clone(), sp, K::ParamRef { index: i, name: if i < 3 { ["x", "y", "z"][i].to_string() } else { format!("a{}", i) } }))
            .collect();
        let body = self.extern_call(decl, ps.clone(), sp)?;
        let rty = body.ty.clone();
        let names: Vec<String> = ps.iter().map(|p| if let K::ParamRef { name, .. } = &p.k { name.clone() } else { String::new() }).collect();
        let seq = Node::new(rty.clone(), sp, K::Seq { stmts: vec![body] });
        Ok(Rc::new(FnSpec::new(decl.name.clone(), "lambda", arg_tys.to_vec(), ps.len(), names, Vec::new(), Vec::new(), seq, rty, sp, self.src, Vec::new())))
    }

    /// The verb of an adverb at these argument types: (FnSpec, captures).
    fn verb(&mut self, v: &Rc<A>, arg_tys: &[Ty], sp: Option<Sp>, fr: usize) -> R<(F, Vec<N>)> {
        match &v.k {
            Kind::Op(g) => {
                if g == "::" || g == ":=" {
                    return self.fail("unsupported-op", sp, format!("{} cannot be the verb of an adverb", g));
                }
                Ok((self.op_spec(g, arg_tys, sp)?, Vec::new()))
            }
            Kind::Sym(name) => {
                if self.lookup_var(name, fr).is_some() {
                    return self.fail("fn-value", sp, format!("{} is a variable used as the verb of an adverb", name));
                }
                if let Some(fi) = self.rep.functions.items.iter().find(|(k, _)| k == name).map(|(_, v)| v) {
                    if fi.params.len() != arg_tys.len() {
                        return self.fail("arity", sp, format!("{} takes {} argument(s); here it is given {}", name, fi.params.len(), arg_tys.len()));
                    }
                    return Ok((self.spec(name, arg_tys)?, Vec::new()));
                }
                if let Some(d) = self.rep.externs.items.iter().find(|(k, _)| k == name).map(|(_, v)| v) {
                    let d = Rc::new(d.clone());
                    return Ok((self.extern_spec(&d, arg_tys, sp)?, Vec::new()));
                }
                self.fail("undefined-name", sp, format!("the function {} is not defined", name))
            }
            Kind::FnLit { .. } => {
                self.frames[fr].n_literals += 1;
                let mut root = fr;
                while let Some(p) = self.frames[root].parent {
                    root = p;
                }
                if fr != root {
                    self.frames[root].n_literals += 1;
                }
                let nm = format!("{}.fn{}", self.frames[root].fname, self.frames[root].n_literals);
                self.literal(v, arg_tys, Some(fr), &nm, "lambda", sp)
            }
            _ => self.fail("fn-value", sp, "the verb of the adverb must be an operator, a function name or a function literal".to_string()),
        }
    }

    // -- variables ----------------------------------------------------------------------
    fn lookup_var(&self, name: &str, fr: usize) -> Option<usize> {
        let mut f = Some(fr);
        while let Some(i) = f {
            if self.frames[i].vars.contains_key(name) {
                return Some(i);
            }
            f = self.frames[i].parent;
        }
        None
    }

    fn read(&mut self, name: &str, sp: Option<Sp>, fr: usize) -> R<N> {
        if let Some(&vi) = self.frames[fr].vars.get(name) {
            let v = &self.frames[fr].store[vi];
            let Some(ty) = v.ty.clone() else {
                return self.fail("unset-local", sp, format!("the local {} is read before it is assigned", name));
            };
            return Ok(if v.kind == 0 {
                Node::new(ty, sp, K::ParamRef { index: v.index, name: name.to_string() })
            } else {
                Node::new(ty, sp, K::LocalRef { slot: v.index, name: name.to_string() })
            });
        }
        if let Some(p) = self.frames[fr].parent {
            if self.lookup_var(name, p).is_some() {
                let outer = self.read(name, sp, p)?;
                let f = &mut self.frames[fr];
                let idx = f.n_own + f.captures.len();
                f.captures.push((name.to_string(), outer.clone()));
                f.store.push(Var { kind: 0, index: idx, ty: Some(outer.ty.clone()), hint: None, name: name.to_string() });
                let vi = f.store.len() - 1;
                f.vars.insert(name.to_string(), vi);
                return Ok(Node::new(outer.ty.clone(), sp, K::ParamRef { index: idx, name: name.to_string() }));
            }
        }
        if self.rep.globals.items.iter().any(|(k, _)| k == name) {
            return self.global_ref(name, sp);
        }
        if self.rep.functions.items.iter().any(|(k, _)| k == name) || self.rep.externs.items.iter().any(|(k, _)| k == name) {
            return self.fail("fn-value", sp, format!("the function {} is used as a value", name));
        }
        self.fail("undefined-name", sp, format!("{} is not defined", name))
    }

    fn store(&mut self, name: &str, value: N, sp: Option<Sp>, fr: usize) -> R<N> {
        if value.ty.elem == "void" {
            return self.fail("void-value", sp, "the result of a void extern is assigned".to_string());
        }
        let vi = self.frames[fr].vars.get(name).copied();
        if let Some(vi) = vi {
            if self.frames[fr].store[vi].kind == 1 {
                let v = &mut self.frames[fr].store[vi];
                match &v.ty {
                    None => {
                        v.ty = Some(Ty::new(if v.hint.as_deref() == Some("float") { "float" } else { &value.ty.elem }, value.ty.shape.clone()));
                    }
                    Some(t) => {
                        if t.shape != value.ty.shape {
                            let msg = format!("{} was {} and is assigned {}: the shape of a variable is fixed", name, t, value.ty);
                            return self.fail("shape-change", sp, msg);
                        }
                        if value.ty.elem == "float" && t.elem == "int" {
                            v.ty = Some(Ty::new("float", t.shape.clone()));
                        }
                    }
                }
                let ty = v.ty.clone().unwrap();
                let slot = v.index;
                let tgt = Node::new(ty.clone(), sp, K::LocalRef { slot, name: name.to_string() });
                let val = self.conv(&value, &ty.elem);
                return Ok(Node::new(ty, sp, K::Assign { target: tgt, value: val }));
            }
        }
        if vi.is_some() || self.lookup_var(name, fr).is_some() {
            return self.fail("closure-assign", sp, format!("the nested function assigns {}, a variable of the function it is written in", name));
        }
        if self.rep.globals.items.iter().any(|(k, _)| k == name) {
            let gi = self.gdef(name)?;
            self.gdefs[gi].state = true;
            let gty = self.gdefs[gi].ty.clone();
            if gty != value.ty {
                let msg = format!("the global {} is {} and is assigned {}: the type and shape of a global are fixed", name, gty, value.ty);
                return self.fail("shape-change", sp, msg);
            }
            let tgt = Node::new(gty.clone(), sp, K::GlobalRef { name: name.to_string(), state: true, known: None });
            return Ok(Node::new(gty, sp, K::Assign { target: tgt, value }));
        }
        self.fail("undefined-name", sp, format!("{} is assigned but is neither a local nor a global", name))
    }

    // -- expressions --------------------------------------------------------------------
    fn ev(&mut self, n: &Rc<A>, sp: Option<Sp>, fr: usize) -> R<N> {
        let sp = n.span.or(sp);
        match &n.k {
            Kind::QSym(name) => self.fail("symbol", sp, format!("the symbol :{} is not a run-time value", name)),
            Kind::Sym(name) => self.read(name, sp, fr),
            Kind::Char | Kind::Str(_) => self.fail("string", sp, "a string is not a run-time value".to_string()),
            Kind::Int(v) => self.int_literal(*v, sp),
            Kind::Flt(f) => Ok(self.cf(*f, sp)),
            Kind::Arr(l) => match l {
                Lit::Num { shape, float, ints, flts } if shape.iter().product::<usize>() > 0 => {
                    let v = if *float { V::AF(flts.clone()) } else { V::AI(ints.iter().map(|x| *x as i64).collect()) };
                    Ok(self.const_v(v, shape.clone(), sp))
                }
                _ => self.fail("list-literal", sp, "the list is not a rectangular list of numbers".to_string()),
            },
            Kind::Cond(_) => self.cond(n, sp, fr),
            Kind::EList(_) => self.elist(n, sp, fr),
            Kind::Chain { .. } => self.adverb(n, sp, fr),
            Kind::Monad(g, a) => {
                if g == "-" {
                    if let Kind::Int(v) = &a.k {
                        return self.int_literal(-*v, sp);
                    }
                }
                let v = self.ev(a, n.arg_span(0, sp.unwrap_or((0, 0))).into(), fr)?;
                self.monad(g, v, sp)
            }
            Kind::OpN(g, k) => self.fail("unsupported-op", sp, format!("the operator {} is applied to {} operands", g, k)),
            Kind::Dyad(g, l, r) => {
                if g == "::" {
                    return self.assign(n, l, r, sp, fr);
                }
                if g == ":=" {
                    return self.amend(n, l, r, sp, fr);
                }
                let d = sp.unwrap_or((0, 0));
                let right = self.ev(r, Some(n.arg_span(1, d)), fr)?;
                let left = self.ev(l, Some(n.arg_span(0, d)), fr)?;
                self.dyad(g, left, right, sp)
            }
            Kind::FnLit { .. } => self.fail("fn-value", sp, "a function literal is used as a value".to_string()),
            Kind::Call { .. } => self.call(n, sp, fr),
            Kind::Op(_) => self.fail("unsupported-op", sp, "a KGOp is not part of the profile".to_string()),
            Kind::Destructure(_) | Kind::Body(_) => self.fail("unsupported-op", sp, "a KGDestructure is not part of the profile".to_string()),
            Kind::EDict(_) | Kind::DictLit(_) => self.fail("unsupported-op", sp, "a KGDict is not part of the profile".to_string()),
            Kind::Missing => self.fail("unsupported-op", sp, "a NoneType is not part of the profile".to_string()),
        }
    }

    fn int_literal(&mut self, v: i128, sp: Option<Sp>) -> R<N> {
        if !self.in_range(v) {
            return self.fail("int-range", sp, format!("{} is outside the range of a {}-bit kg_int", v, self.int_bits));
        }
        Ok(self.ci(v as i64, sp))
    }

    fn cond(&mut self, n: &Rc<A>, sp: Option<Sp>, fr: usize) -> R<N> {
        let Kind::Cond(parts) = &n.k else { unreachable!() };
        let d = sp.unwrap_or((0, 0));
        if parts.len() != 3 {
            return self.fail("unsupported-op", sp, "a conditional has a condition and two branches".to_string());
        }
        let c = self.ev(&parts[0], Some(n.arg_span(0, d)), fr)?;
        if !c.ty.shape.is_empty() || c.ty.elem == "void" {
            return self.fail("cond-scalar", Some(n.arg_span(0, d)), format!("the condition is {}; it must be a scalar", c.ty));
        }
        let t = self.ev(&parts[1], Some(n.arg_span(1, d)), fr)?;
        let e = self.ev(&parts[2], Some(n.arg_span(2, d)), fr)?;
        if t.ty.elem == "void" && e.ty.elem == "void" {
            return Ok(Node::new(void(), sp, K::Cond { c, t, e }));
        }
        if t.ty.elem == "void" || e.ty.elem == "void" || t.ty.shape != e.ty.shape {
            return self.fail("cond-shape", sp, format!("the branches of the conditional are {} and {}; they must have the same shape", t.ty, e.ty));
        }
        let elem = promote(&t.ty.elem, &e.ty.elem);
        let (t2, e2) = (self.conv(&t, &elem), self.conv(&e, &elem));
        Ok(Node::new(Ty::new(&elem, t.ty.shape.clone()), sp, K::Cond { c, t: t2, e: e2 }))
    }

    fn elist(&mut self, n: &Rc<A>, sp: Option<Sp>, fr: usize) -> R<N> {
        let Kind::EList(parts) = &n.k else { unreachable!() };
        let d = sp.unwrap_or((0, 0));
        let mut items = Vec::new();
        for (i, q) in parts.iter().enumerate() {
            items.push(self.ev(q, Some(n.arg_span(i, d)), fr)?);
        }
        if items.is_empty() {
            return self.fail("array-shape", sp, "an empty list has a dimension of 0".to_string());
        }
        let mut elem = "int".to_string();
        for it in &items {
            if it.ty.elem == "void" {
                return self.fail("void-value", sp, "a list element is the result of a void extern".to_string());
            }
            if it.ty.shape != items[0].ty.shape {
                return self.fail("list-literal", sp, format!("the elements of the list are {} and {}; they must have the same shape", items[0].ty, it.ty));
            }
            elem = promote(&elem, &it.ty.elem);
        }
        let mut shape = vec![items.len()];
        shape.extend(items[0].ty.shape.iter());
        let ty = self.mk(&elem, &shape, sp)?;
        let items = items.iter().map(|it| self.conv(it, &elem)).collect();
        Ok(Node::new(ty, sp, K::MakeList { items }))
    }

    fn assign(&mut self, n: &Rc<A>, left: &Rc<A>, right: &Rc<A>, sp: Option<Sp>, fr: usize) -> R<N> {
        let d = sp.unwrap_or((0, 0));
        let lsp = Some(n.arg_span(0, d));
        if right.is_fn_literal() {
            return self.fail("fn-value", Some(n.arg_span(1, d)), "a function is assigned to a variable inside a function".to_string());
        }
        let v = self.ev(right, Some(n.arg_span(1, d)), fr)?;
        if let Kind::Destructure(names) = &left.k {
            if v.ty.shape.is_empty() || v.ty.shape[0] != names.len() {
                return self.fail("shape-mismatch", sp, format!("cannot destructure {} into {} name(s)", v.ty, names.len()));
            }
            let hname = format!("%d{}", self.frames[fr].locals.len());
            let hv = self.new_local(fr, &hname, Some(v.ty.clone()), None);
            let slot = self.frames[fr].store[hv].index;
            let r = Node::new(v.ty.clone(), sp, K::LocalRef { slot, name: hname.clone() });
            let mut stmts = vec![Node::new(v.ty.clone(), sp, K::Assign { target: r.clone(), value: v.clone() })];
            let item = Ty::new(&v.ty.elem, v.ty.shape[1..].to_vec());
            for (i, q) in names.iter().enumerate() {
                let idx = Node::new(item.clone(), sp, K::Index { a: r.clone(), i: self.ci(i as i64, sp), checked: true });
                stmts.push(self.store(q, idx, lsp, fr)?);
            }
            stmts.push(r);
            return Ok(Node::new(v.ty.clone(), sp, K::Seq { stmts }));
        }
        let Kind::Sym(name) = &left.k else {
            return self.fail("strict-syntax", lsp, "the left side of :: must be a name".to_string());
        };
        self.store(name, v, lsp, fr)
    }

    // -- calls ----------------------------------------------------------------------------
    fn call(&mut self, n: &Rc<A>, sp: Option<Sp>, fr: usize) -> R<N> {
        let Kind::Call { callee, args, .. } = &n.k else { unreachable!() };
        let d = sp.unwrap_or((0, 0));
        if args.iter().any(|q| q.is_missing()) {
            return self.fail("projection", sp, "the call leaves an argument out: that is a projection, a function value".to_string());
        }
        let Kind::Sym(name) = &callee.k else {
            return self.fail("fn-value", sp, "a function literal is called in place".to_string());
        };
        if self.lookup_var(name, fr).is_some() {
            return self.fail("fn-value", sp, format!("{} is a variable called as a function", name));
        }
        if let Some(arity) = self.rep.functions.items.iter().find(|(k, _)| k == name).map(|(_, v)| v.params.len()) {
            if args.len() != arity {
                let rule = if args.len() < arity { "projection" } else { "arity" };
                return self.fail(rule, sp, format!("{} takes {} argument(s), got {}", name, arity, args.len()));
            }
            let mut vals = Vec::new();
            for (i, q) in args.iter().enumerate() {
                vals.push(self.ev(q, Some(n.arg_span(i, d)), fr)?);
            }
            for v in &vals {
                if v.ty.elem == "void" {
                    return self.fail("void-value", sp, format!("an argument of {} is the result of a void extern", name));
                }
            }
            let tys: Vec<Ty> = vals.iter().map(|v| v.ty.clone()).collect();
            let g = self.spec(name, &tys)?;
            let rt = g.result_ty.clone();
            return Ok(Node::new(rt, sp, K::Call { f: g, args: vals }));
        }
        if let Some(decl) = self.rep.externs.items.iter().find(|(k, _)| k == name).map(|(_, v)| Rc::new(v.clone())) {
            let sig = &decl.signature;
            if args.len() != sig.arity() {
                return self.fail("extern-call", sp, format!("{} is declared \"{}\" and takes {} argument(s), got {}", name, sig, sig.arity(), args.len()));
            }
            let mut vals = Vec::new();
            for (i, (q, ty)) in args.iter().zip(&sig.params).enumerate() {
                if ty.elem == "str" {
                    match &q.k {
                        Kind::Str(s) => vals.push(Node::new(str_ty(), Some(n.arg_span(i, d)), K::Const(V::S(s.clone())))),
                        _ => return self.fail("extern-call", Some(n.arg_span(i, d)), format!("argument {} of {} is declared str: it must be a string literal", i + 1, name)),
                    }
                } else {
                    vals.push(self.ev(q, Some(n.arg_span(i, d)), fr)?);
                }
            }
            return self.extern_call(&decl, vals, sp);
        }
        if name == ".f" {
            return self.fail("recursion", sp, ".f is the running function: calling it is recursion".to_string());
        }
        if name == ".fixed" {
            let Kind::Str(text) = &args[0].k else { return self.fail("fixed", sp, "the type of .fixed must be a string literal".to_string()) };
            let ty = match parse_type(text) {
                Ok(t) => t,
                Err(e) => return self.fail("signature", sp, e.message),
            };
            let v = self.ev(&args[1], Some(n.arg_span(1, d)), fr)?;
            return Ok(self.to_fixed(&v, &ty.elem));
        }
        if name == ".float" {
            let v = self.ev(&args[0], Some(n.arg_span(0, d)), fr)?;
            if v.ty.elem == "float" {
                return Ok(v);
            }
            if let K::Const(c) = &v.k {
                if !is_fixed(&v.ty.elem) {
                    let c = if v.ty.shape.is_empty() { V::F(c.as_f64()[0]) } else { V::AF(c.as_f64()) };
                    return Ok(self.const_v(c, v.ty.shape.clone(), sp));
                }
            }
            return Ok(Node::new(Ty::new("float", v.ty.shape.clone()), sp, K::Convert { a: v }));
        }
        if name.starts_with('.') {
            return self.fail("sys-fn", sp, format!("{} is a system function; it does not exist in a compiled program", name));
        }
        self.fail("undefined-name", sp, format!("the function {} is not defined", name))
    }

    fn extern_call(&mut self, decl: &Rc<ExternDecl>, vals: Vec<N>, sp: Option<Sp>) -> R<N> {
        let sig = &decl.signature;
        let mut out = Vec::new();
        for (i, (v, ty)) in vals.iter().zip(&sig.params).enumerate() {
            if ty.elem == "str" {
                if v.ty.elem != "str" {
                    return self.fail("extern-call", sp, format!("argument {} of {} is declared str: it must be a string literal", i + 1, decl.name));
                }
                out.push(v.clone());
                continue;
            }
            if v.ty.shape != ty.shape || (v.ty.elem == "float" && ty.elem == "int") || v.ty.elem == "void" || v.ty.elem == "str" {
                return self.fail("extern-call", sp, format!("argument {} of {} is declared {}, got {}", i + 1, decl.name, ty, v.ty));
            }
            out.push(self.conv(v, &ty.elem));
        }
        self.used_externs.insert(decl.name.clone());
        Ok(Node::new(sig.result.clone(), sp, K::ExternCall { ext: decl.clone(), args: out }))
    }

    // -- adverbs ----------------------------------------------------------------------------
    fn adverb(&mut self, n: &Rc<A>, sp: Option<Sp>, fr: usize) -> R<N> {
        let Kind::Chain { verb, advs, dyadic, l, r } = &n.k else { unreachable!() };
        let dyadic = *dyadic;
        let d = sp.unwrap_or((0, 0));
        let vsp = n.op_span.or(sp);
        if advs.len() != 1 {
            return self.fail("unsupported-op", vsp, format!("the adverb chain {} is not in the profile", advs.join("")));
        }
        let glyph = advs[0].clone();
        let row: Option<&'static Op> = lookup(&glyph, if dyadic { "adverb2" } else { "adverb1" });
        let row = match row {
            Some(r0) if r0.allowed => r0,
            other => {
                let rule = other.and_then(|o| o.rule).unwrap_or("unsupported-op");
                return self.fail(rule_id(rule), vsp, format!("the adverb {} is not in the profile", glyph));
            }
        };
        let kind = row.id;
        if kind == "while" || kind == "iterate" {
            let init = self.ev(r, Some(n.arg_span(1, d)), fr)?;
            if init.ty.elem == "void" {
                return self.fail("void-value", sp, "the starting value is the result of a void extern".to_string());
            }
            let (count, pred, pcap);
            if kind == "iterate" {
                let c = self.ev(l, Some(n.arg_span(0, d)), fr)?;
                if c.ty != int() {
                    return self.fail("operand", Some(n.arg_span(0, d)), format!("the count of Iterate is {}; it must be an int scalar", c.ty));
                }
                count = Some(c);
                pred = None;
                pcap = Vec::new();
            } else {
                count = None;
                let (p, pc) = self.verb(l, &[init.ty.clone()], Some(n.arg_span(0, d)), fr)?;
                if !p.result_ty.shape.is_empty() || p.result_ty.elem == "void" {
                    return self.fail("cond-scalar", Some(n.arg_span(0, d)), format!("the predicate of While returns {}; it must be a scalar", p.result_ty));
                }
                pred = Some(p);
                pcap = pc;
            }
            let (f, cap) = self.verb(verb, &[init.ty.clone()], vsp, fr)?;
            if f.result_ty != init.ty {
                return self.fail("loop-state", sp, format!("the state starts as {} and the verb returns {}; its type and shape must not change", init.ty, f.result_ty));
            }
            let ty = init.ty.clone();
            return Ok(Node::new(ty, sp, K::Loop { kind: kind.to_string(), f, init, count, pred, caps: cap, pcaps: pcap }));
        }
        let (a, b);
        if dyadic {
            b = self.ev(r, Some(n.arg_span(1, d)), fr)?;
            a = Some(self.ev(l, Some(n.arg_span(0, d)), fr)?);
        } else {
            b = self.ev(r, Some(n.arg_span(0, d)), fr)?;
            a = None;
        }
        for v in [a.as_ref(), Some(&b)].into_iter().flatten() {
            if v.ty.elem == "void" {
                return self.fail("void-value", sp, format!("an operand of {} ({}) is the result of a void extern", glyph, row.name));
            }
        }
        if b.ty.shape.is_empty() {
            return self.fail("operand", sp, format!("{} ({}) needs an array; the operand is {}", glyph, row.name, b.ty));
        }
        let item = Ty::new(&b.ty.elem, b.ty.shape[1..].to_vec());
        let count = b.ty.shape[0];
        let stacked = |me: &Self, r: &Ty, items: usize| -> R<Ty> {
            if r.elem == "void" {
                return Ok(void());
            }
            let mut sh = vec![items];
            sh.extend(r.shape.iter());
            me.mk(&r.elem, &sh, sp)
        };
        if kind == "each" {
            let (f, cap) = self.verb(verb, &[item], vsp, fr)?;
            let ty = stacked(self, &f.result_ty, count)?;
            return Ok(Node::new(ty, sp, K::Map { f, a: b, caps: cap }));
        }
        if kind == "each2" {
            let a = a.unwrap();
            if a.ty.shape.is_empty() {
                return self.fail("operand", sp, format!("{} ({}) needs two arrays; the left operand is {}", glyph, row.name, a.ty));
            }
            if a.ty.shape[0] != count {
                return self.fail("shape-mismatch", sp, format!("{} ({}) pairs the items of {} and {}: the lengths differ", glyph, row.name, a.ty, b.ty));
            }
            let (f, cap) = self.verb(verb, &[Ty::new(&a.ty.elem, a.ty.shape[1..].to_vec()), item], vsp, fr)?;
            let ty = stacked(self, &f.result_ty, count)?;
            return Ok(Node::new(ty, sp, K::Map2 { f, a, b, caps: cap }));
        }
        if kind == "each-pair" {
            if count < 2 {
                return self.fail("array-shape", sp, format!("{} ({}) of {} has no pairs: the result would be empty", glyph, row.name, b.ty));
            }
            let (f, cap) = self.verb(verb, &[item.clone(), item], vsp, fr)?;
            let ty = stacked(self, &f.result_ty, count - 1)?;
            return Ok(Node::new(ty, sp, K::EachPair { f, a: b, caps: cap }));
        }
        // over and scan: the accumulator is the fixed point of acc = f(acc, item), with one promotion
        let mut acc = a.as_ref().map(|x| x.ty.clone()).unwrap_or_else(|| item.clone());
        if a.is_some() && acc.elem == "int" && is_fixed(&item.elem) {
            acc = Ty::new(&item.elem, acc.shape.clone());
        }
        let (mut f, mut cap) = self.verb(verb, &[acc.clone(), item.clone()], vsp, fr)?;
        for _ in 0..2 {
            let rr = f.result_ty.clone();
            if rr.elem == "void" {
                return self.fail("void-value", sp, format!("the verb of {} ({}) returns nothing", glyph, row.name));
            }
            if rr.shape != acc.shape {
                return self.fail("loop-state", sp, format!("{} ({}): the accumulator is {} and the verb returns {}; its shape must not change", glyph, row.name, acc, rr));
            }
            if rr.elem == "float" && acc.elem == "int" {
                acc = Ty::new("float", acc.shape.clone());
                let rv = self.verb(verb, &[acc.clone(), item.clone()], vsp, fr)?;
                f = rv.0;
                cap = rv.1;
                continue;
            }
            break;
        }
        acc = Ty::new(&promote(&acc.elem, &f.result_ty.elem), acc.shape.clone());
        let seed = a.as_ref().map(|x| self.conv(x, &acc.elem));
        if kind == "over" || kind == "over-seed" {
            return Ok(Node::new(acc, sp, K::Reduce { f, a: b, seed, caps: cap }));
        }
        let mut shape = vec![count + if a.is_some() { 1 } else { 0 }];
        shape.extend(acc.shape.iter());
        let ty = self.mk(&acc.elem, &shape, sp)?;
        Ok(Node::new(ty, sp, K::Scan { f, a: b, seed, caps: cap, acc_ty: acc }))
    }

    // -- operators ---------------------------------------------------------------------------
    fn row(&self, glyph: &str, valence: &str, sp: Option<Sp>) -> R<&'static Op> {
        let Some(row) = lookup(glyph, valence) else {
            return self.fail("unsupported-op", sp, format!("the operator {} is not in the profile", glyph));
        };
        if !row.allowed {
            return self.fail(rule_id(row.rule.unwrap_or("unsupported-op")), sp, format!("{} ({}): {}", glyph, row.name, row.note));
        }
        Ok(row)
    }

    /// The constant int count of take, drop or enumerate.
    fn count(&self, a: &N, sp: Option<Sp>, what: &str) -> R<i64> {
        if a.ty != int() {
            return self.fail("operand", sp, format!("the count of {} is {}; it must be an int scalar", what, a.ty));
        }
        match a.cval() {
            Some(V::I(c)) if !a.effects() => Ok(c),
            _ => self.fail("const-count", sp, format!("the count of {} is not a compile-time constant: the result length would depend on run-time data", what)),
        }
    }

    fn monad(&mut self, glyph: &str, a: N, sp: Option<Sp>) -> R<N> {
        let row = self.row(glyph, "monad", sp)?;
        if a.ty.elem == "void" {
            return self.fail("void-value", sp, format!("the operand of {} ({}) is the result of a void extern", glyph, row.name));
        }
        let shape = a.ty.shape.clone();
        match row.id {
            "neg" => {
                let c = a.cval();
                if is_fixed(&a.ty.elem) {
                    return Ok(Node::new(a.ty.clone(), sp, K::Unary { op: "neg".into(), a, known: None }));
                }
                if let (K::Const(v), true) = (&a.k, shape.is_empty()) {
                    match v {
                        V::F(f) => return Ok(self.cf(-*f, sp)),
                        V::I(i) => {
                            if self.in_range(-(*i as i128)) {
                                return Ok(self.ci(-*i, sp));
                            }
                        }
                        _ => {}
                    }
                }
                let known = c.and_then(|v| match v {
                    V::I(i) => i.checked_neg().map(V::I),
                    V::F(f) => Some(V::F(-f)),
                    V::AI(x) => x.iter().map(|y| y.checked_neg()).collect::<Option<Vec<_>>>().map(V::AI),
                    V::AF(x) => Some(V::AF(x.iter().map(|y| -y).collect())),
                    V::S(_) => None,
                });
                Ok(Node::new(a.ty.clone(), sp, K::Unary { op: "neg".into(), a, known }))
            }
            "floor" => {
                if a.ty.elem == "int" {
                    Ok(a)
                } else {
                    Ok(Node::new(Ty::new("int", shape), sp, K::Unary { op: "floor".into(), a, known: None }))
                }
            }
            "recip" => {
                let a = self.conv(&a, "float");
                Ok(Node::new(Ty::new("float", shape), sp, K::Unary { op: "recip".into(), a, known: None }))
            }
            "not" => Ok(Node::new(Ty::new("int", shape), sp, K::Unary { op: "not".into(), a, known: None })),
            "size" | "shape" => {
                if shape.is_empty() {
                    return self.fail("size-of-atom", sp, format!("{} ({}) of {}: the operand is an atom", glyph, row.name, a.ty));
                }
                let c = if row.id == "size" { self.ci(shape[0] as i64, sp) } else { Node::new(Ty::new("int", vec![shape.len()]), sp, K::Const(V::AI(shape.iter().map(|d| *d as i64).collect()))) };
                if a.effects() {
                    Ok(Node::new(c.ty.clone(), sp, K::Seq { stmts: vec![a, c] }))
                } else {
                    Ok(c)
                }
            }
            "first" => {
                if shape.is_empty() {
                    Ok(a)
                } else {
                    let ty = Ty::new(&a.ty.elem, shape[1..].to_vec());
                    let z = self.ci(0, sp);
                    Ok(Node::new(ty, sp, K::Index { a, i: z, checked: true }))
                }
            }
            "reverse" => {
                if shape.is_empty() {
                    Ok(a)
                } else {
                    Ok(Node::new(a.ty.clone(), sp, K::Reverse { a }))
                }
            }
            "transpose" => {
                if shape.len() != 2 {
                    return self.fail("operand", sp, format!("+ (Transpose) needs a rank-2 array; the operand is {}", a.ty));
                }
                Ok(Node::new(Ty::new(&a.ty.elem, vec![shape[1], shape[0]]), sp, K::Transpose { a }))
            }
            "enumerate" => {
                let c = self.count(&a, sp, "!n (Enumerate)")?;
                if c <= 0 {
                    return self.fail("array-shape", sp, format!("!{} would be empty: n must be positive", c));
                }
                Ok(Node::new(Ty::new("int", vec![c as usize]), sp, K::Iota { n: c }))
            }
            "list" => {
                let mut sh = vec![1usize];
                sh.extend(shape.iter());
                let ty = self.mk(&a.ty.elem, &sh, sp)?;
                if shape.is_empty() {
                    Ok(Node::new(ty, sp, K::MakeList { items: vec![a] }))
                } else {
                    Ok(Node::new(ty, sp, K::Reshape { a }))
                }
            }
            _ => self.fail("unsupported-op", sp, format!("{} ({}) is not emitted", glyph, row.name)),
        }
    }

    fn prefix(&self, a: &N, b: &N, sp: Option<Sp>, row: &Op) -> R<Vec<usize>> {
        let (sa, sb) = (&a.ty.shape, &b.ty.shape);
        if sa.is_empty() {
            return Ok(sb.clone());
        }
        if sb.is_empty() {
            return Ok(sa.clone());
        }
        let (short, long) = if sa.len() <= sb.len() { (sa, sb) } else { (sb, sa) };
        if long[..short.len()] != short[..] {
            return self.fail("shape-mismatch", sp, format!("cannot apply {} ({}) to {} and {}", row.glyph, row.name, a.ty, b.ty));
        }
        Ok(long.clone())
    }

    /// The value of an int operation on two int constants, or None when it is not folded.
    fn fold(&self, k: &str, x: i128, y: i128) -> Option<i128> {
        match k {
            "add" => Some(x + y),
            "sub" => Some(x - y),
            "mul" => x.checked_mul(y),
            "max" => Some(x.max(y)),
            "min" => Some(x.min(y)),
            "idiv" if y != 0 => Some(x / y),
            "rem" if y != 0 => Some(x - (x / y) * y),
            "pow" if (0..=64).contains(&y) => x.checked_pow(y as u32),
            "lt" => Some((x < y) as i128),
            "gt" => Some((x > y) as i128),
            "eq" => Some((x == y) as i128),
            _ => None,
        }
    }

    fn dyad(&mut self, glyph: &str, a: N, b: N, sp: Option<Sp>) -> R<N> {
        let row = self.row(glyph, "dyad", sp)?;
        if a.ty.elem == "void" || b.ty.elem == "void" {
            return self.fail("void-value", sp, format!("an operand of {} ({}) is the result of a void extern", glyph, row.name));
        }
        let k = row.id;
        let (ea, eb) = (a.ty.elem.clone(), b.ty.elem.clone());
        if row.shape_rule == Some("prefix") {
            let shape = self.prefix(&a, &b, sp, row)?;
            let (ca, cb) = (a.cval(), b.cval());
            if matches!(cb, Some(V::I(0))) && ea == "int" && eb == "int" && (k == "idiv" || k == "rem" || (k == "div" && ca.is_some())) {
                return self.fail("div-zero", sp, format!("{} ({}) divides an int by the constant 0", glyph, row.name));
            }
            let rule = row.type_rule.unwrap_or("");
            let fxe = if is_fixed(&ea) {
                Some(ea.clone())
            } else if is_fixed(&eb) {
                Some(eb.clone())
            } else {
                None
            };
            if let Some(fxe) = fxe {
                let elem = if rule == "int" { "int".to_string() } else { fxe.clone() };
                if shape.len() > MAX_RANK {
                    self.mk(&elem, &shape, sp)?;
                }
                let (x, y) = (self.conv(&a, &fxe), self.conv(&b, &fxe));
                return Ok(Node::new(Ty::new(&elem, shape), sp, K::Binary { op: k.to_string(), a: x, b: y }));
            }
            let (elem, work);
            match rule {
                "promote" => {
                    elem = promote(&ea, &eb);
                    work = elem.clone();
                }
                "float" => {
                    elem = "float".to_string();
                    work = elem.clone();
                }
                "int" => {
                    elem = "int".to_string();
                    work = promote(&ea, &eb);
                }
                "int-only" => {
                    if ea == "float" || eb == "float" {
                        return self.fail("operand", sp, format!("{} ({}) needs int operands; they are {} and {}", glyph, row.name, a.ty, b.ty));
                    }
                    elem = "int".to_string();
                    work = elem.clone();
                }
                _ => {
                    let nonneg = matches!(cb, Some(V::I(c)) if c >= 0);
                    elem = if ea == "int" && eb == "int" && nonneg { "int".to_string() } else { "float".to_string() };
                    work = elem.clone();
                }
            }
            if shape.len() > MAX_RANK {
                self.mk(&elem, &shape, sp)?;
            }
            if let (Some(V::I(x)), Some(V::I(y)), true, true, true) = (&ca, &cb, elem == "int", shape.is_empty(), a.is_const() && b.is_const()) {
                if let Some(v) = self.fold(k, *x as i128, *y as i128) {
                    if self.in_range(v) {
                        return Ok(self.ci(v as i64, sp));
                    }
                }
            }
            let (x, y) = (self.conv(&a, &work), self.conv(&b, &work));
            return Ok(Node::new(Ty::new(&elem, shape), sp, K::Binary { op: k.to_string(), a: x, b: y }));
        }
        match k {
            "match" => {
                if a.ty.shape != b.ty.shape {
                    return self.fail("shape-mismatch", sp, format!("~ (Match) compares {} with {}: the shapes differ", a.ty, b.ty));
                }
                let work = promote(&ea, &eb);
                let (x, y) = (self.conv(&a, &work), self.conv(&b, &work));
                Ok(Node::new(int(), sp, K::Match { a: x, b: y }))
            }
            "index" => {
                if a.ty.shape.is_empty() {
                    return self.fail("operand", sp, format!("cannot index {}: @ (Index) needs an array on the left", a.ty));
                }
                if eb != "int" {
                    return self.fail("operand", sp, format!("the index is {}; an index must be an int", b.ty));
                }
                let n = a.ty.shape[0];
                let (b, checked) = self.const_index(b, n, sp, &a.ty.to_string())?;
                if b.ty.shape.is_empty() {
                    return Ok(Node::new(Ty::new(&ea, a.ty.shape[1..].to_vec()), sp, K::Index { a, i: b, checked }));
                }
                let mut sh = b.ty.shape.clone();
                sh.extend(a.ty.shape[1..].iter());
                let ty = self.mk(&ea, &sh, sp)?;
                Ok(Node::new(ty, sp, K::Gather { a, idx: b, checked }))
            }
            "take" => {
                let c = self.count(&a, sp, "n#a (Take)")?;
                if b.ty.shape.len() != 1 {
                    return self.fail("take-rank", sp, format!("n#a (Take) from {}: take works on rank-1 arrays only", b.ty));
                }
                if c == 0 {
                    return self.fail("array-shape", sp, "0#a would be empty: every dimension is a positive integer".to_string());
                }
                Ok(Node::new(Ty::new(&eb, vec![c.unsigned_abs() as usize]), sp, K::Take { a: b, n: c }))
            }
            "drop" => {
                let c = self.count(&a, sp, "n_a (Drop)")?;
                if b.ty.shape.is_empty() {
                    return self.fail("operand", sp, format!("n_a (Drop) from {}: the right operand must be an array", b.ty));
                }
                let left = b.ty.shape[0] as i64 - c.abs();
                if left <= 0 {
                    return self.fail("array-shape", sp, format!("dropping {} from {} leaves nothing", c.abs(), b.ty));
                }
                if c == 0 {
                    return Ok(b);
                }
                let mut sh = vec![left as usize];
                sh.extend(b.ty.shape[1..].iter());
                Ok(Node::new(Ty::new(&eb, sh), sp, K::Drop { a: b, n: c }))
            }
            "join" => {
                let elem = promote(&ea, &eb);
                let (sa, sb) = (a.ty.shape.clone(), b.ty.shape.clone());
                let (x, y) = (self.conv(&a, &elem), self.conv(&b, &elem));
                if sa.is_empty() && sb.is_empty() {
                    return Ok(Node::new(Ty::new(&elem, vec![2]), sp, K::Join { a: x, b: y }));
                }
                if sa.is_empty() || sb.is_empty() {
                    let arr = if sa.is_empty() { &sb } else { &sa };
                    if arr.len() != 1 {
                        return self.fail("shape-mismatch", sp, format!("joining {} with {} would give a ragged list", a.ty, b.ty));
                    }
                    return Ok(Node::new(Ty::new(&elem, vec![arr[0] + 1]), sp, K::Join { a: x, b: y }));
                }
                if sa.len() != sb.len() || sa[1..] != sb[1..] {
                    return self.fail("shape-mismatch", sp, format!("joining {} with {} would give a ragged list", a.ty, b.ty));
                }
                let mut sh = vec![sa[0] + sb[0]];
                sh.extend(sa[1..].iter());
                Ok(Node::new(Ty::new(&elem, sh), sp, K::Join { a: x, b: y }))
            }
            "rotate" => {
                if a.ty != int() {
                    return self.fail("operand", sp, format!("the count of n:+a (Rotate) is {}; it must be an int scalar", a.ty));
                }
                if b.ty.shape.len() != 1 {
                    return self.fail("operand", sp, format!("n:+a (Rotate) of {}: the profile rotates rank-1 arrays only", b.ty));
                }
                Ok(Node::new(b.ty.clone(), sp, K::Rotate { a: b, n: a }))
            }
            "reshape" => {
                let c = a.cval();
                if ea != "int" {
                    return self.fail("operand", sp, format!("the shape of s:^a (Reshape) is {}; it must hold ints", a.ty));
                }
                let Some(c) = c.filter(|_| !a.effects() && a.ty.shape.len() <= 1) else {
                    return self.fail("const-count", sp, "the shape of s:^a (Reshape) is not a compile-time constant list of ints".to_string());
                };
                if a.ty.shape.is_empty() && b.ty.shape.len() > 1 {
                    return self.fail("operand", sp, format!("n:^a (Reshape) of {} with a scalar n: write the shape as a list, [n]:^a", b.ty));
                }
                let dims: Vec<i64> = c.as_i64();
                if dims.iter().any(|d| *d <= 0) {
                    return self.fail("array-shape", sp, "the result would have a dimension of 0: every dimension is a positive integer".to_string());
                }
                let dims: Vec<usize> = dims.iter().map(|d| *d as usize).collect();
                let ty = self.mk(&eb, &dims, sp)?;
                Ok(Node::new(ty, sp, K::Reshape { a: b }))
            }
            _ => self.fail("unsupported-op", sp, format!("{} ({}) cannot be used here", glyph, row.name)),
        }
    }

    /// Range-check a constant index against length n and make it non-negative. Returns (node, checked).
    fn const_index(&self, b: N, n: usize, sp: Option<Sp>, what: &str) -> R<(N, bool)> {
        let Some(c) = b.cval() else { return Ok((b, false)) };
        let idx = c.as_i64();
        let (mn, mx) = (idx.iter().min().copied(), idx.iter().max().copied());
        if let (Some(mn), Some(mx)) = (mn, mx) {
            if mn < -(n as i64) || mx >= n as i64 {
                let worst = if mx >= n as i64 { mx } else { mn };
                return self.fail("index-range", sp, format!("index {} is out of range for {}", worst, what));
            }
        }
        if b.is_const() {
            let m = |x: i64| x.rem_euclid(n as i64);
            if b.ty.shape.is_empty() {
                return Ok((self.ci(m(idx[0]), b.span), true));
            }
            return Ok((self.const_v(V::AI(idx.iter().map(|x| m(*x)).collect()), b.ty.shape.clone(), b.span), true));
        }
        let ok = mn.map(|m| m >= 0).unwrap_or(true);
        Ok((b, ok))
    }

    fn amend(&mut self, n: &Rc<A>, target: &Rc<A>, pair: &Rc<A>, sp: Option<Sp>, fr: usize) -> R<N> {
        let d = sp.unwrap_or((0, 0));
        let mut psp = n.arg_span(1, d);
        let Kind::Dyad(g, pv, pi) = &pair.k else {
            return self.fail("amend-form", Some(psp), "the right side of := (Amend) must be written v,i: one value, then the index or indices".to_string());
        };
        if g != "," {
            return self.fail("amend-form", Some(psp), "the right side of := (Amend) must be written v,i: one value, then the index or indices".to_string());
        }
        if let Some(s) = pair.span {
            psp = s;
        }
        let idx = self.ev(pi, Some(pair.arg_span(1, psp)), fr)?;
        let val = self.ev(pv, Some(pair.arg_span(0, psp)), fr)?;
        let arr = self.ev(target, Some(n.arg_span(0, d)), fr)?;
        if idx.ty.elem == "void" || val.ty.elem == "void" || arr.ty.elem == "void" {
            return self.fail("void-value", sp, "an operand of := (Amend) is the result of a void extern".to_string());
        }
        if arr.ty.shape.len() != 1 {
            return self.fail("amend-form", sp, format!(":= (Amend) of {}: the profile amends rank-1 arrays only", arr.ty));
        }
        if !val.ty.shape.is_empty() {
            return self.fail("amend-form", Some(psp), format!("the value of := (Amend) is {}; it must be one scalar", val.ty));
        }
        if idx.ty.elem != "int" {
            return self.fail("operand", Some(psp), format!("the index of := (Amend) is {}; an index must be an int", idx.ty));
        }
        if idx.ty.shape.len() > 1 {
            return self.fail("amend-form", Some(psp), format!("the index of := (Amend) is {}; it must be a scalar or a rank-1 list", idx.ty));
        }
        let (idx, checked) = self.const_index(idx, arr.ty.shape[0], Some(psp), &arr.ty.to_string())?;
        let elem = promote(&arr.ty.elem, &val.ty.elem);
        let ty = Ty::new(&elem, arr.ty.shape.clone());
        let (a2, v2) = (self.conv(&arr, &elem), self.conv(&val, &elem));
        Ok(Node::new(ty, sp, K::Amend { a: a2, value: v2, idx, checked }))
    }

    // -- the program ----------------------------------------------------------------------------
    fn program(&mut self) -> R<Program> {
        let rep = self.rep;
        let mut per_fn: HashMap<String, usize> = HashMap::new();
        let mut seen: HashSet<(String, String)> = HashSet::new();
        for e in &rep.entries {
            if seen.insert((e.name.clone(), e.signature.to_string())) {
                *per_fn.entry(e.name.clone()).or_insert(0) += 1;
            }
        }
        let mut entries: Vec<Entry> = Vec::new();
        let mut pending: HashSet<(String, String)> = seen.clone();
        for e in &rep.entries {
            let key = (e.name.clone(), e.signature.to_string());
            if !pending.remove(&key) {
                continue;
            }
            let sig = &e.signature;
            self.src = rep.functions.items.iter().find(|(k, _)| *k == e.name).unwrap().1.src;
            let f = self.spec(&e.name, &sig.params)?;
            let (r, want) = (&f.result_ty, &sig.result);
            if want.elem != "void"
                && (r.elem == "void" || r.shape != want.shape || (r.elem == "float" && want.elem == "int") || ((is_fixed(&r.elem) || is_fixed(&want.elem)) && r.elem != want.elem))
            {
                self.src = e.src;
                return self.fail("entry", Some(e.span), format!("entry {} is declared to return {}, but the function returns {}", e.name, want, r));
            }
            let mut c_name = e.name.replace('.', "_");
            if per_fn[&e.name] > 1 {
                c_name.push_str("__");
                c_name.push_str(&suffix(sig));
            }
            entries.push(Entry { decl: e.clone(), f, c_name });
        }
        for x in &entries {
            if entries.iter().filter(|y| y.c_name == x.c_name).count() > 1 {
                self.src = x.decl.src;
                return self.fail("entry", Some(x.decl.span), format!("two entries have the C name {}", x.c_name));
            }
        }
        // order the specialisations callees first and number them
        let mut order: Vec<F> = Vec::new();
        let mut done: HashSet<usize> = HashSet::new();
        fn visit(f: &F, order: &mut Vec<F>, done: &mut HashSet<usize>) {
            if !done.insert(fn_key(f)) {
                return;
            }
            for n in f.body.walked() {
                for g in n.fnspecs() {
                    visit(g, order, done);
                }
            }
            order.push(f.clone());
        }
        for x in &entries {
            visit(&x.f, &mut order, &mut done);
        }
        for (i, f) in order.iter().enumerate() {
            f.index.set(i);
        }
        let mut used: HashSet<String> = HashSet::new();
        let mut ext_used: HashSet<String> = HashSet::new();
        for f in &order {
            for n in f.body.walked() {
                match &n.k {
                    K::GlobalRef { name, .. } => {
                        used.insert(name.clone());
                    }
                    K::ExternCall { ext, .. } => {
                        ext_used.insert(ext.name.clone());
                    }
                    _ => {}
                }
            }
        }
        let mut gl: Vec<GlobalDef> = Vec::new();
        let mut gdefs: Vec<Option<GlobalDef>> = std::mem::take(&mut self.gdefs).into_iter().map(Some).collect();
        for (name, _) in &rep.globals.items {
            let Some(&gi) = self.gindex.get(name) else { continue };
            let g = gdefs[gi].take().unwrap();
            if !g.ty.shape.is_empty() && !used.contains(name) {
                continue;
            }
            gl.push(g);
        }
        let externs: Vec<Rc<ExternDecl>> = rep.externs.items.iter().filter(|(k, _)| ext_used.contains(k)).map(|(_, d)| Rc::new(d.clone())).collect();
        Ok(Program { name: self.name.clone(), globals: gl, fns: order, entries, externs, int_bits: self.int_bits, source: 0 })
    }
}

fn rule_id(rule: &str) -> &'static str {
    tables::rule(rule).map(|r| r.id).unwrap_or("unsupported-op")
}

/// The C name suffix that tells apart the entries of one function: the parameter types, `f256` or `i_f3x2`.
fn suffix(sig: &crate::sig::Signature) -> String {
    let parts: Vec<String> = sig
        .params
        .iter()
        .map(|p| {
            let dims: Vec<String> = p.shape.iter().map(|d| d.to_string()).collect();
            format!("{}{}", p.elem.chars().next().unwrap(), dims.join("x"))
        })
        .collect();
    if parts.is_empty() {
        "void".to_string()
    } else {
        parts.join("_")
    }
}

/// Lower a checked program to the typed IR. `report` must be the result of `check_source` with `Options::globals` set, with no
/// diagnostics and no unresolved global.
pub fn lower(report: &Report, values: &HashMap<String, StageValue>, name: &str, int_bits: u32) -> Result<Program, Fail> {
    if report.entries.is_empty() {
        return Err(Fail {
            rule: "entry",
            span: (0, 0),
            src: 0,
            message: "the program declares no .entry: there is nothing to compile".to_string(),
            hint: Some("mark the functions the host calls: .entry(\"step\";\"float -> float\")".to_string()),
        });
    }
    let (lo, hi) = int_range(int_bits);
    let mut l = Lower {
        rep: report,
        values,
        name: name.to_string(),
        int_bits,
        lo,
        hi,
        memo: HashMap::new(),
        gdefs: Vec::new(),
        gindex: HashMap::new(),
        used_externs: HashSet::new(),
        src: 0,
        frames: Vec::new(),
    };
    l.program()
}
