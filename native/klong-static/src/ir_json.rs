//! The typed IR as a structure a host can read (the same information as `ir::dump`, and the spans the text leaves out): JSON text.
//!
//! ```text
//! program: {"name", "int_bits", "source": name of the main source, "globals": [global], "externs": [extern], "entries": [entry], "fns": [fn]}
//! global:  {"name", "ty", "value", "state", "param"}        ty: {"elem", "shape": [..]}
//! value:   {"int": n} | {"float": x} | {"ints": [..]} | {"floats": [..]} | {"str": text}     (NaN and the infinities as JSON.parse reads them)
//! extern:  {"name", "signature": text, "module", "c_name"}
//! entry:   {"name", "signature": text, "c_name", "fn": label, "span": [start, end]}
//! fn:      {"label": "name#index", "name", "index", "kind", "params": [{"name","ty"}], "n_own", "locals": [{"name","ty"}], "result": ty,
//!           "body": node, "span", "source": name, "stmt_spans": [span|null], "effects": bool}
//! node:    {"k": class name, "ty", "span": [start, end] | null, ...fields of the class}; operands are nodes, a function is its label,
//!          `known` and `value` are values (or null)
//! ```
//! A span counts characters of the text of its source (the function's `source` for a node of a function).
use crate::check::Report;
use crate::emit_hls::json::{dumps, s, J};
use crate::ir::{Entry, FnSpec, GlobalDef, Node, Program, K, V};
use crate::sig::{ExternDecl, Ty};

struct Ctx<'a> {
    report: &'a Report,
}

fn obj(items: Vec<(&str, J)>) -> J {
    J::Obj(items.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
}

fn int(n: usize) -> J {
    J::Int(n as i128)
}

fn ty(t: &Ty) -> J {
    obj(vec![("elem", s(&t.elem)), ("shape", J::Arr(t.shape.iter().map(|d| int(*d)).collect()))])
}

fn value(v: &V) -> J {
    match v {
        V::I(i) => obj(vec![("int", J::Int(*i as i128))]),
        V::F(f) => obj(vec![("float", J::Flt(*f))]),
        V::AI(a) => obj(vec![("ints", J::Arr(a.iter().map(|x| J::Int(*x as i128)).collect()))]),
        V::AF(a) => obj(vec![("floats", J::Arr(a.iter().map(|x| J::Flt(*x)).collect()))]),
        V::S(t) => obj(vec![("str", s(t))]),
    }
}

fn opt_value(v: &Option<V>) -> J {
    v.as_ref().map(value).unwrap_or(J::Null)
}

impl Ctx<'_> {
    fn chars(&self, src: usize, sp: (usize, usize)) -> J {
        let t = self.report.sources.get(src).map(|x| x.source.text.as_str()).unwrap_or("");
        let ch = |b: usize| if t.is_ascii() { b } else { t.get(..b.min(t.len())).map(|p| p.chars().count()).unwrap_or(b) };
        J::Arr(vec![int(ch(sp.0)), int(ch(sp.1))])
    }
    fn span(&self, src: usize, sp: Option<(usize, usize)>) -> J {
        sp.map(|p| self.chars(src, p)).unwrap_or(J::Null)
    }
    fn src_name(&self, src: usize) -> J {
        s(self.report.sources.get(src).map(|x| x.name.as_str()).unwrap_or(""))
    }

    fn nodes(&self, v: &[crate::ir::N], src: usize) -> J {
        J::Arr(v.iter().map(|n| self.node(n, src)).collect())
    }

    fn node(&self, n: &Node, src: usize) -> J {
        let c = |x: &Node| self.node(x, src);
        let f = |x: &FnSpec| s(&x.label());
        let mut o: Vec<(&str, J)> = vec![("k", s(n.class())), ("ty", ty(&n.ty)), ("span", self.span(src, n.span))];
        match &n.k {
            K::Const(v) => o.push(("value", value(v))),
            K::GlobalRef { name, state, known } => {
                o.push(("name", s(name)));
                o.push(("state", J::Bool(*state)));
                o.push(("known", opt_value(known)));
            }
            K::LocalRef { slot, name } => {
                o.push(("slot", int(*slot)));
                o.push(("name", s(name)));
            }
            K::ParamRef { index, name } => {
                o.push(("index", int(*index)));
                o.push(("name", s(name)));
            }
            K::Unary { op, a, known } => {
                o.push(("op", s(op)));
                o.push(("a", c(a)));
                o.push(("known", opt_value(known)));
            }
            K::Binary { op, a, b } => {
                o.push(("op", s(op)));
                o.push(("a", c(a)));
                o.push(("b", c(b)));
            }
            K::Convert { a } | K::Reverse { a } | K::Transpose { a } | K::Reshape { a } => o.push(("a", c(a))),
            K::Match { a, b } | K::Join { a, b } => {
                o.push(("a", c(a)));
                o.push(("b", c(b)));
            }
            K::Index { a, i, checked } => {
                o.push(("a", c(a)));
                o.push(("i", c(i)));
                o.push(("checked", J::Bool(*checked)));
            }
            K::Gather { a, idx, checked } => {
                o.push(("a", c(a)));
                o.push(("idx", c(idx)));
                o.push(("checked", J::Bool(*checked)));
            }
            K::Take { a, n: k } | K::Drop { a, n: k } => {
                o.push(("a", c(a)));
                o.push(("n", J::Int(*k as i128)));
            }
            K::Rotate { a, n: k } => {
                o.push(("a", c(a)));
                o.push(("n", c(k)));
            }
            K::Iota { n: k } => o.push(("n", J::Int(*k as i128))),
            K::MakeList { items } => o.push(("items", self.nodes(items, src))),
            K::Amend { a, value: v, idx, checked } => {
                o.push(("a", c(a)));
                o.push(("value", c(v)));
                o.push(("idx", c(idx)));
                o.push(("checked", J::Bool(*checked)));
            }
            K::Cond { c: cc, t, e } => {
                o.push(("c", c(cc)));
                o.push(("t", c(t)));
                o.push(("e", c(e)));
            }
            K::Call { f: g, args } => {
                o.push(("fn", f(g)));
                o.push(("args", self.nodes(args, src)));
            }
            K::ExternCall { ext, args } => {
                o.push(("extern", s(&ext.name)));
                o.push(("args", self.nodes(args, src)));
            }
            K::Map { f: g, a, caps } | K::EachPair { f: g, a, caps } => {
                o.push(("fn", f(g)));
                o.push(("a", c(a)));
                o.push(("caps", self.nodes(caps, src)));
            }
            K::Map2 { f: g, a, b, caps } => {
                o.push(("fn", f(g)));
                o.push(("a", c(a)));
                o.push(("b", c(b)));
                o.push(("caps", self.nodes(caps, src)));
            }
            K::Reduce { f: g, a, seed, caps } => {
                o.push(("fn", f(g)));
                o.push(("a", c(a)));
                o.push(("seed", seed.as_ref().map(|x| c(x)).unwrap_or(J::Null)));
                o.push(("caps", self.nodes(caps, src)));
            }
            K::Scan { f: g, a, seed, caps, acc_ty } => {
                o.push(("fn", f(g)));
                o.push(("a", c(a)));
                o.push(("seed", seed.as_ref().map(|x| c(x)).unwrap_or(J::Null)));
                o.push(("caps", self.nodes(caps, src)));
                o.push(("acc_ty", ty(acc_ty)));
            }
            K::Loop { kind, f: g, init, count, pred, caps, pcaps } => {
                o.push(("kind", s(kind)));
                o.push(("fn", f(g)));
                o.push(("init", c(init)));
                o.push(("count", count.as_ref().map(|x| c(x)).unwrap_or(J::Null)));
                o.push(("pred", pred.as_ref().map(|x| f(x)).unwrap_or(J::Null)));
                o.push(("caps", self.nodes(caps, src)));
                o.push(("pcaps", self.nodes(pcaps, src)));
            }
            K::Assign { target, value: v } => {
                o.push(("target", c(target)));
                o.push(("value", c(v)));
            }
            K::Seq { stmts } => o.push(("stmts", self.nodes(stmts, src))),
        }
        obj(o)
    }

    fn function(&self, f: &FnSpec) -> J {
        let named = |names: &[String], tys: &[Ty]| J::Arr(names.iter().zip(tys).map(|(n, t)| obj(vec![("name", s(n)), ("ty", ty(t))])).collect());
        obj(vec![
            ("label", s(&f.label())),
            ("name", s(&f.name)),
            ("index", J::Int(f.index.get() as isize as i128)),
            ("kind", s(f.kind)),
            ("params", named(&f.param_names, &f.param_tys)),
            ("n_own", int(f.n_own)),
            ("locals", named(&f.local_names, &f.locals)),
            ("result", ty(&f.result_ty)),
            ("body", self.node(&f.body, f.source)),
            ("span", self.span(f.source, f.span)),
            ("source", self.src_name(f.source)),
            ("stmt_spans", J::Arr(f.stmt_spans.iter().map(|p| self.span(f.source, *p)).collect())),
            ("effects", J::Bool(f.effects())),
        ])
    }

    fn global(&self, g: &GlobalDef) -> J {
        obj(vec![("name", s(&g.name)), ("ty", ty(&g.ty)), ("value", value(&g.value)), ("state", J::Bool(g.state)), ("param", J::Bool(g.param))])
    }

    fn external(&self, x: &ExternDecl) -> J {
        obj(vec![("name", s(&x.name)), ("signature", s(&x.signature.to_string())), ("module", s(&x.module)), ("c_name", s(&crate::ir::c_name_of(x)))])
    }

    fn entry(&self, e: &Entry) -> J {
        obj(vec![
            ("name", s(&e.decl.name)),
            ("signature", s(&e.decl.signature.to_string())),
            ("c_name", s(&e.c_name)),
            ("fn", s(&e.f.label())),
            ("span", self.chars(e.decl.src, e.decl.span)),
        ])
    }
}

/// The program as JSON text (see the module comment). `report` is the report the program was lowered from: its sources turn spans into characters.
pub fn program_json(p: &Program, report: &Report) -> String {
    let cx = Ctx { report };
    let j = obj(vec![
        ("name", s(&p.name)),
        ("int_bits", J::Int(p.int_bits as i128)),
        ("source", cx.src_name(p.source)),
        ("globals", J::Arr(p.globals.iter().map(|g| cx.global(g)).collect())),
        ("externs", J::Arr(p.externs.iter().map(|x| cx.external(x)).collect())),
        ("entries", J::Arr(p.entries.iter().map(|e| cx.entry(e)).collect())),
        ("fns", J::Arr(p.fns.iter().map(|f| cx.function(f)).collect())),
    ]);
    dumps(&j)
}
