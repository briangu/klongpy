//! The text of a function that an IPC peer sends (`klongpy/ipc/endpoint.py`, `parse_function`): a function travels as its literal,
//! `{x+y}`, or, when it has named parameters, as the definition `fn(a;b)::{a+b}` (`fn` is a placeholder: the receiver keeps the
//! function and defines nothing).
use crate::ast::with_parse;
use klong_core::tree::{Node, Tree};
use klong_core::sym::Interner;

/// What a text is when it is exactly one function literal.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FunctionForm {
    /// The named form `fn(a;b)::{...}`; false for the plain literal `{...}`.
    pub named: bool,
    /// The arity of the literal: the highest of x, y, z that its body mentions, or the number of named parameters.
    pub arity: u8,
    /// The named parameters (`a`, `b`), for the named form.
    pub params: Option<Vec<String>>,
    /// The names declared local at the head of the body.
    pub locals: Vec<String>,
}

fn base(name: &str) -> &str {
    name.split('`').next().unwrap_or(name)
}

fn literal(tree: &Tree, syms: &Interner, id: klong_core::tree::NodeId, named: bool) -> Option<FunctionForm> {
    let Node::Fn { a, arity, has_args: false, call: false, .. } = tree.node(id) else { return None };
    let (mut params, mut locals) = (None, Vec::new());
    if let Node::Body { locals: l, params: p, has_params, .. } = tree.node(*a) {
        let names = |r| tree.ids(r).iter().map(|&q| syms.name(klong_core::sym::SymId(q)).to_string()).collect::<Vec<_>>();
        locals = names(*l);
        if *has_params {
            params = Some(names(*p));
        }
    }
    if named != params.is_some() {
        return None;
    }
    Some(FunctionForm { named, arity: *arity, params, locals })
}

/// `parse_function`'s parse step: Some(form) when the text parses to exactly one statement that is a function literal (plain, or
/// the named form whose name is `fn`), None otherwise (a text that does not parse is None too). Nothing runs.
pub fn function_form(text: &str) -> Option<FunctionForm> {
    with_parse(text, "<function>", |ip, p| {
        if p.stmts.len() != 1 {
            return None;
        }
        let t = p.stmts[0];
        if !Tree::is_real(t) {
            return None;
        }
        match p.tree.node(t) {
            Node::Dyad { op, l, r, missing: false } if ip.syms.name(*op) == "::" => {
                let Node::Sym(s) = p.tree.node(*l) else { return None };
                if base(ip.syms.name(*s)) != "fn" || !Tree::is_real(*r) {
                    return None;
                }
                literal(&p.tree, &ip.syms, *r, true)
            }
            Node::Fn { .. } => literal(&p.tree, &ip.syms, t, false),
            _ => None,
        }
    })
    .ok()
    .flatten()
}

/// The text a function goes out as: the literal's text, with the definition around it when it has named parameters.
pub fn wire_text(literal: &str, params: Option<&[String]>) -> Option<String> {
    if !(literal.starts_with('{') && literal.ends_with('}')) {
        return None;
    }
    Some(match params {
        Some(p) => {
            let names: Vec<&str> = p.iter().map(|n| base(n)).collect();
            format!("fn({})::{}", names.join(";"), literal)
        }
        None => literal.to_string(),
    })
}

/// The arity of a display form (`:nilad`, `:monad`, `:dyad`, `:triad`, `:fn/5`) that a peer sends for a function whose text it cannot
/// give, else None.
pub fn display_arity(text: &str) -> Option<u32> {
    let rest = text.strip_prefix(':')?;
    // the pattern ends in `$`, which also matches before one trailing newline
    let rest = rest.strip_suffix('\n').unwrap_or(rest);
    match rest {
        "nilad" => Some(0),
        "monad" => Some(1),
        "dyad" => Some(2),
        "triad" => Some(3),
        _ => {
            let n = rest.strip_prefix("fn/")?;
            // ASCII digits only (Python's `\d` also takes the digits of other scripts)
            if n.is_empty() || !n.chars().all(|c| c.is_ascii_digit()) {
                return None;
            }
            n.parse().ok()
        }
    }
}
