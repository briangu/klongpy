//! The canonical S-expression dump of a parse tree (`tickets/design/frontend.md` section 7;
//! the mapping is the one in the docstring of `klongpy/dump.py`). The native parser must print
//! the same bytes as KlongPy for every golden file under `tests/golden`.
use alloc::string::String;
use core::fmt::Write;

use crate::fmt::write_float;
use crate::sym::{self, Interner};
use crate::tree::{Node, NodeId, Range, Tree, NONE};
use crate::value::Value;

/// A JSON string literal, as Python's `json.dumps` writes it (non-ASCII as `\uXXXX`).
fn json_str(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 || (c as u32) >= 0x80 => {
                let mut buf = [0u16; 2];
                for u in c.encode_utf16(&mut buf) {
                    let _ = write!(out, "\\u{:04x}", u);
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

pub struct Dumper<'a> {
    pub tree: &'a Tree,
    pub syms: &'a Interner,
}

impl<'a> Dumper<'a> {
    /// Data mode: the elements of list and dictionary literals.
    pub fn data(&self, v: &Value, out: &mut String) {
        match v {
            Value::Prim(p) => {
                out.push_str("(op ");
                json_str(self.syms.name(*p), out);
                out.push(')');
            }
            Value::Sym(s) => {
                out.push_str("(sym ");
                out.push_str(self.syms.name(*s));
                out.push(')');
            }
            Value::Char(c) => {
                out.push_str("(char ");
                let mut b = [0u8; 4];
                json_str(c.encode_utf8(&mut b), out);
                out.push(')');
            }
            Value::Str(s) => {
                out.push_str("(str ");
                json_str(s.as_str(), out);
                out.push(')');
            }
            Value::Int(i) => {
                let _ = write!(out, "(num {})", i);
            }
            Value::Flt(f) => {
                out.push_str("(num ");
                write_float(*f, out);
                out.push(')');
            }
            Value::Dict(d) => {
                let d = d.borrow();
                out.push_str("(dict");
                for i in 0..d.len() {
                    out.push_str(" (");
                    self.data(&d.keys()[i], out);
                    out.push(' ');
                    self.data(&d.vals()[i], out);
                    out.push(')');
                }
                out.push(')');
            }
            Value::Ints(_) | Value::Flts(_) | Value::List(_) => {
                out.push_str("(list");
                for i in 0..v.len().unwrap_or(0) {
                    out.push(' ');
                    self.data(&v.item(i), out);
                }
                out.push(')');
            }
            _ => out.push_str("(unknown)"),
        }
    }

    fn each(&self, r: Range, out: &mut String) {
        for &q in self.tree.ids(r) {
            out.push(' ');
            self.expr(NodeId(q), out);
        }
    }

    fn names(&self, r: Range, out: &mut String) {
        for &q in self.tree.ids(r) {
            out.push(' ');
            out.push_str(self.syms.name(crate::sym::SymId(q)));
        }
    }

    /// `(fn ARITY (params a b) (locals a b) BODY...)`.
    fn func(&self, body: NodeId, arity: u8, out: &mut String) {
        let _ = write!(out, "(fn {}", arity);
        match self.tree.node(body) {
            Node::Body { stmts, locals, params, has_params } => {
                if *has_params {
                    out.push_str(" (params");
                    self.names(*params, out);
                    out.push(')');
                }
                if locals.len > 0 {
                    out.push_str(" (locals");
                    self.names(*locals, out);
                    out.push(')');
                }
                self.each(*stmts, out);
            }
            Node::Seq(r) => self.each(*r, out),
            _ => {
                out.push(' ');
                self.expr(body, out);
            }
        }
        out.push(')');
    }

    fn callee(&self, a: NodeId, arity: u8, out: &mut String) {
        if matches!(self.tree.node(a), Node::Sym(_) | Node::QSym(_)) || self.tree.is_kgfn(a) {
            self.expr(a, out);
        } else {
            self.func(a, arity, out);
        }
    }

    fn operand(&self, id: NodeId, out: &mut String) {
        if id == NONE {
            out.push_str("(missing)");
        } else {
            self.expr(id, out);
        }
    }

    /// Expression mode.
    pub fn expr(&self, id: NodeId, out: &mut String) {
        if !Tree::is_real(id) {
            out.push('_');
            return;
        }
        match self.tree.node(id) {
            Node::QSym(s) => {
                out.push_str("(qsym ");
                out.push_str(self.syms.name(*s));
                out.push(')');
            }
            Node::Sym(s) => {
                out.push_str(if sym::is_arg(*s) { "(arg " } else { "(sym " });
                out.push_str(self.syms.name(*s));
                out.push(')');
            }
            Node::Lit(v) | Node::DictLit(v) | Node::CopyList(v) => self.data(v, out),
            Node::OpVal { op, .. } => {
                out.push_str("(opval ");
                json_str(self.syms.name(*op), out);
                out.push(')');
            }
            Node::Cond(c, t, e) => {
                out.push_str("(cond ");
                self.expr(*c, out);
                out.push(' ');
                self.expr(*t, out);
                out.push(' ');
                self.expr(*e, out);
                out.push(')');
            }
            Node::EDict(r) => {
                out.push_str("(edict");
                self.each(*r, out);
                out.push(')');
            }
            Node::EList(r) => {
                out.push_str("(elist");
                self.each(*r, out);
                out.push(')');
            }
            Node::Adv { verb, advs, dyad, l, r, .. } => {
                out.push_str(if *dyad { "(adv2 (verb " } else { "(adv (verb " });
                match self.tree.node(*verb) {
                    Node::OpVal { op, .. } => json_str(self.syms.name(*op), out),
                    _ => self.expr(*verb, out),
                }
                out.push_str(") (adverbs");
                for (i, &q) in self.tree.ids(*advs).iter().enumerate() {
                    out.push(if i == 0 { ' ' } else { ' ' });
                    json_str(self.syms.name(crate::sym::SymId(q)), out);
                }
                out.push_str(") ");
                if *dyad {
                    self.expr(*l, out);
                    out.push(' ');
                }
                self.expr(*r, out);
                out.push(')');
            }
            Node::Monad { op, arg, .. } => {
                out.push_str("(m ");
                json_str(self.syms.name(*op), out);
                out.push(' ');
                self.operand(*arg, out);
                out.push(')');
            }
            Node::Dyad { op, l, r, .. } => {
                let name = self.syms.name(*op);
                if let (true, Node::Destructure(names)) = (Tree::is_real(*l), if Tree::is_real(*l) { self.tree.node(*l) } else { self.tree.node(id) }) {
                    out.push_str("(destructure (");
                    let mut first = true;
                    for &q in self.tree.ids(*names) {
                        if !first {
                            out.push(' ');
                        }
                        first = false;
                        out.push_str(self.syms.name(crate::sym::SymId(q)));
                    }
                    out.push_str(") ");
                    self.operand(*r, out);
                    out.push(')');
                    return;
                }
                if name == "::" {
                    out.push_str("(def ");
                } else {
                    out.push_str("(d ");
                    json_str(name, out);
                    out.push(' ');
                }
                self.operand(*l, out);
                out.push(' ');
                self.operand(*r, out);
                out.push(')');
            }
            Node::Fn { a, args, has_args, arity, call, .. } => {
                if !*has_args {
                    self.func(*a, *arity, out);
                    return;
                }
                let ids = self.tree.ids(*args);
                let hole = ids.contains(&NONE.0);
                let infix = !*call && !hole && ids.len() == 2 && (matches!(self.tree.node(*a), Node::Sym(_) | Node::QSym(_)) || self.tree.is_kgfn(*a));
                if infix {
                    out.push_str("(infix ");
                    self.expr(*a, out);
                } else {
                    out.push_str(if *call { "(call " } else { "(proj " });
                    self.callee(*a, *arity, out);
                }
                self.each(*args, out);
                out.push(')');
            }
            Node::Seq(r) => {
                out.push_str("(seq");
                self.each(*r, out);
                out.push(')');
            }
            Node::Body { stmts, .. } => {
                out.push_str("(seq");
                self.each(*stmts, out);
                out.push(')');
            }
            Node::Destructure(_) => out.push_str("(unknown KGDestructure)"),
        }
    }
}

/// The dump of a program: one line per statement.
pub fn dump(tree: &Tree, syms: &Interner, stmts: &[NodeId]) -> String {
    let d = Dumper { tree, syms };
    let mut out = String::new();
    for &s in stmts {
        d.expr(s, &mut out);
        out.push('\n');
    }
    out
}
