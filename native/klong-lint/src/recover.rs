//! What the linter and the static checker share: the token scan, the statements of a program parsed one at a time with a
//! recovery after each syntax error, and the lookup of `.l` files.
//!
//! The parser stops at the first problem; a tool that reports every problem resumes at the next top-level statement. That
//! recovery needs the tokens (`tokenize`), their nesting (`scan`, whose `ctx` says which tokens are top level) and the end of the
//! statement that failed (`next_boundary`). `parse_recovering` is the whole of it: the statements that parsed, the problems of the
//! ones that did not, and the names the failed statements were defining.
use crate::tok::{tokenize as tok_text, Source as Text};
use klong_core::error::Source as CoreSource;
use klong_core::lex::K;
use klong_core::parse::parse_next;
use klong_core::sym::Interner;
use klong_core::tree::{NodeId, Tree};
use std::collections::BTreeSet;
use std::rc::Rc;

pub use crate::analysis::{next_boundary, Problem};
pub use crate::load::{find_import, find_load, py_dirname, py_join, Session};
pub use crate::scan::{scan, Frame, Scan, FK};
pub use crate::tok::{tokenize, Source, Tok};

/// One statement that parsed: its tree, its node, and where it was written (from the start of its first factor to the end of its
/// last token).
pub struct Statement {
    pub tree: Rc<Tree>,
    pub node: NodeId,
    pub span: (usize, usize),
}

/// The result of `parse_recovering`.
pub struct Recovered {
    pub statements: Vec<Statement>,
    /// What the parser said about each statement it rejected, in source order.
    pub problems: Vec<Problem>,
    /// The names that the rejected statements were defining (`name::` or `name(a;b)::` at their start).
    pub broken: BTreeSet<String>,
    /// The names of the trees' symbols.
    pub syms: Interner,
}

/// `[ \t]*([A-Za-z.][A-Za-z0-9.]*)[ \t]*(\([^)\n]*\))?[ \t]*::` at `at`: the name it defines.
fn defines(text: &str, at: usize) -> Option<String> {
    let b = text.as_bytes();
    let mut i = at;
    let skip = |i: &mut usize| {
        while *i < b.len() && (b[*i] == b' ' || b[*i] == b'\t') {
            *i += 1;
        }
    };
    skip(&mut i);
    let s = i;
    if i >= b.len() || !(b[i].is_ascii_alphabetic() || b[i] == b'.') {
        return None;
    }
    while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'.') {
        i += 1;
    }
    let name = &text[s..i];
    skip(&mut i);
    if i < b.len() && b[i] == b'(' {
        let mut j = i + 1;
        while j < b.len() && b[j] != b')' && b[j] != b'\n' {
            j += 1;
        }
        if j < b.len() && b[j] == b')' {
            i = j + 1;
        }
    }
    skip(&mut i);
    if text[i..].starts_with("::") {
        Some(name.to_string())
    } else {
        None
    }
}

/// Parse a text at a strict level into statements, resuming after each syntax error at the next top-level statement, so that every
/// error is found. The trees keep where their operands were written (`tree::Aux`).
pub fn parse_recovering(name: &str, text: &str, strict: u8) -> Recovered {
    let n = text.len();
    let src = Text::new(name, text.to_string());
    let (toks, _) = tok_text(&src.text);
    let sc = scan(&toks, &src.text);
    let ctx = sc.ctx;
    let core: Rc<CoreSource> = Rc::new(CoreSource { name: name.to_string(), text: std::borrow::Cow::Owned(text.to_string()) });
    let mut syms = Interner::new();
    syms.record_spans = true;
    syms.max_warnings = usize::MAX;
    let mut module: Option<String> = None;
    let mut out = Recovered { statements: Vec::new(), problems: Vec::new(), broken: BTreeSet::new(), syms: Interner::new() };

    // the offset of the first token of the top-level statement that holds `offset`
    let statement_start = |offset: usize| -> usize {
        let mut i = toks.partition_point(|t| t.s <= offset).saturating_sub(1);
        while i > 0 && !(ctx[i - 1] == 0 && matches!(toks[i - 1].k, K::Semi | K::Newline)) {
            i -= 1;
        }
        toks.get(i).map(|t| t.s).unwrap_or(offset)
    };

    // parse text[pos..end]: the first problem, if any (the statements read are added to `out`)
    let attempt = |pos: usize, end: usize, syms: &mut Interner, module: &mut Option<String>, out: &mut Recovered| -> Option<Problem> {
        let mut cur = pos;
        loop {
            let mut window = 256usize;
            let r = parse_next(&core, cur, end, &mut window, None, syms, module, strict, 512, None);
            match r {
                Err(e) => {
                    let span = e.span.map(|s| (s.0 as usize, s.1 as usize)).unwrap_or((pos, (pos + 1).min(end)));
                    let message = if e.message.is_empty() { "KlongSyntaxError".to_string() } else { e.message.to_string() };
                    return Some(Problem { span, message, hint: e.hint.as_ref().map(|h| h.to_string()) });
                }
                Ok(next) => {
                    if next.skipped {
                        cur = next.end;
                        continue;
                    }
                    match next.parsed {
                        None => return None,
                        Some(p) => {
                            if let Some(&node) = p.stmts.first() {
                                out.statements.push(Statement { tree: p.tree.clone(), node, span: next.stmt_span });
                            }
                            if next.last {
                                return None;
                            }
                            cur = next.end;
                        }
                    }
                }
            }
        }
    };

    let mut pos = 0usize;
    for _ in 0..100000 {
        if pos >= n {
            break;
        }
        let mark = out.statements.len();
        let problem = attempt(pos, n, &mut syms, &mut module, &mut out);
        let Some(problem) = problem else { break };
        // the statements read before the bad one stay; the bad one starts at the statement that holds the problem
        let stop = problem.span.0.max(pos);
        let bad = statement_start(stop).max(pos);
        // statements that lie at or after `bad` were not part of the earlier text: drop what this attempt read beyond it
        while out.statements.len() > mark && out.statements.last().map(|s| s.span.0 >= bad).unwrap_or(false) {
            out.statements.pop();
        }
        if bad > pos {
            // (the text before the bad statement is read again on its own: a problem it has itself is reported too)
            out.statements.truncate(mark);
            if let Some(p) = attempt(pos, bad, &mut syms, &mut module, &mut out) {
                out.problems.push(p);
            }
        }
        out.problems.push(problem);
        if let Some(d) = defines(text, bad) {
            out.broken.insert(d);
        }
        match next_boundary(&toks, &ctx, stop) {
            Some(nxt) if nxt > pos => pos = nxt,
            _ => break,
        }
    }
    out.syms = syms;
    out
}
