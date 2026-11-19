//! Parser: tokens to a tree. A port, decision for decision, of the token grammar in
//! `klongpy/interpreter.py` (`_prog`, `_expr`, `_factor`, `_read_fn`, `_read_fn_args`,
//! `_apply_adverbs`) and the `tk_*` readers of `klongpy/parser.py`, because the canonical dump
//! of the result must equal KlongPy's for every golden file (`conformance/README.md`,
//! and the pinned quirks in tests/golden/README.md).
//!
//! At strict level 0 the input the legacy grammar accepted is accepted here (KlongPy warns;
//! warnings are not part of this spike). At level 1 and above each such place is a syntax
//! error.
use alloc::format;
use alloc::rc::Rc;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use crate::error::{Error, Result, Source, Span};
use crate::lex::{Lexer, Tok, K, TV};
use crate::ops;
use crate::sym::{self, Interner, SymId};
use crate::tree::{Node, NodeId, Range, Tree, NONE, SEMI};
use crate::value::{Dict, Value};

pub struct Parsed {
    pub tree: Rc<Tree>,
    pub stmts: Vec<NodeId>,
}

struct NamedDef {
    name: SymId,
    params: Vec<u32>,
}

enum Target {
    Node(NodeId),
    Named(NamedDef),
}

pub struct Parser<'a> {
    lx: Lexer<'a>,
    tree: Tree,
    syms: &'a mut Interner,
    module: &'a mut Option<String>,
    strict: u8,
    last_list: Option<(NodeId, usize)>,
    dicts_read: u32,
    depth: u32,
    max_depth: u32,
    /// (stack address the budget is measured from, bytes allowed): `Config::stack_limit`.
    stack: Option<(usize, usize)>,
    /// The span of the expression `expr` read last: from the start of its first factor to the end of its last token.
    last_sp: Span,
    /// Spans written to `tree.aux`.
    #[cfg(feature = "diag")]
    rec: bool,
    /// The spans of the arguments `read_fn_args` read last.
    #[cfg(feature = "diag")]
    args_sp: Vec<Span>,
    /// The spans of the statements `prog` read last.
    #[cfg(feature = "diag")]
    prog_sp: Vec<Span>,
}

const REST: &str = "the rest of the input is not evaluated";
const TO_END: &str = "it runs to the end of the input";

const ADVERBS: [&str; 10] = ["'", "/", "\\", "\\~", "\\*", ":'", ":/", ":\\", ":~", ":*"];

fn adverb_arity(s: &str, ctx: u8) -> u8 {
    match s {
        "'" => ctx,
        ":\\" | ":'" | ":/" | "/" | "\\" => 2,
        _ => 1,
    }
}

/// Parse a program. `module` is the module in force (`.module(:name)` switches it while
/// parsing, as in KlongPy). `strict` is the strict level the text is parsed at.
pub fn parse(
    text: &str,
    name: &str,
    syms: &mut Interner,
    module: &mut Option<String>,
    strict: u8,
    max_depth: u32,
    stack: Option<(usize, usize)>,
) -> Result<Parsed> {
    // the copy of the text for error messages is sized by data
    let mut copy = String::new();
    {
        let _tag = crate::htag::Scope::new(crate::htag::TREE);
        copy.try_reserve_exact(text.len()).map_err(|_| Error::memory())?;
    }
    copy.push_str(text);
    parse_owned(copy, name, syms, module, strict, max_depth, stack)
}

/// `parse` of a text the caller has no more use for: the text is the copy that error messages are cut from, so a program held in
/// memory is not held twice (a library of 12 KB is 12 KB less at the peak of its load).
pub fn parse_owned(
    text: String,
    name: &str,
    syms: &mut Interner,
    module: &mut Option<String>,
    strict: u8,
    max_depth: u32,
    stack: Option<(usize, usize)>,
) -> Result<Parsed> {
    let _tag = crate::htag::Scope::new(crate::htag::TREE);
    let source = Rc::new(Source { name: crate::value::try_str(name)?, text: alloc::borrow::Cow::Owned(text) });
    let anchor = source.clone();
    let text: &str = &anchor.text;
    let mut p = Parser {
        lx: Lexer::new(text, strict == 0),
        tree: Tree::new(Some(source.clone())),
        syms,
        module,
        strict,
        last_list: None,
        dicts_read: 0,
        depth: 0,
        max_depth,
        stack,
        last_sp: (0, 0),
        #[cfg(feature = "diag")]
        rec: false,
        #[cfg(feature = "diag")]
        args_sp: Vec::new(),
        #[cfg(feature = "diag")]
        prog_sp: Vec::new(),
    };
    if p.lx.toks.is_empty() || !p.tree.reserve() {
        return Err(Error::memory());
    }
    let r = p.prog(false).and_then(|stmts| p.finish().map(|_| stmts));
    // tokens, nodes or names that could not be stored: whatever the parser made of the rest
    // is meaningless
    if p.lx.oom || p.tree.oom {
        p.syms.failure();
        return Err(Error::memory());
    }
    if let Some(e) = p.syms.failure() {
        return Err(if e.kind == crate::error::Kind::Range { Error::syntax("a name is longer than 255 bytes") } else { e });
    }
    match r {
        Ok(stmts) => {
            let mut ids = crate::value::try_vec(stmts.len())?;
            ids.extend(stmts.into_iter().map(NodeId));
            p.tree.shrink();
            Ok(Parsed { tree: Rc::new(p.tree), stmts: ids })
        }
        Err(mut e) => {
            if e.source.is_none() && e.span.is_some() {
                e.source = Some(source);
            }
            Err(e)
        }
    }
}

/// The next top-level statement of a program, from byte `start` of `source`, parsed on its own: the tokens are those of a window of
/// the text (`window` bytes, doubled and the statement parsed again if the parser reached the end of the window), the tree is the
/// statement's alone, and what the caller gets is that tree and where the next statement starts. A program of a few hundred
/// statements then never has its tokens, or its whole tree, in memory at once. `parsed` is `None` at the end of the program and
/// `last` is true when nothing more is to be parsed (the legacy parser drops what follows a complete statement that is not
/// followed by a separator). Errors carry absolute offsets into `source`.
pub struct Next {
    /// `.comment("END")` was the statement: nothing to run, the next starts at `end` (the raw text it hides is not lexed).
    pub skipped: bool,
    pub parsed: Option<Parsed>,
    pub end: usize,
    pub last: bool,
    /// Where the statement's last token ends (the offset just after the last token read; `end` is the end of the text when `last`).
    /// `end` itself when `parsed` is `None`.
    pub stmt_end: usize,
    /// The span of the statement as an operand: from the start of its first factor to the end of its last token (`stmt_end`).
    pub stmt_span: (usize, usize),
}

pub fn parse_next(
    source: &Rc<Source>,
    start: usize,
    limit: usize,
    window: &mut usize,
    lazy: Option<Rc<crate::value::LazyCtx>>,
    syms: &mut Interner,
    module: &mut Option<String>,
    strict: u8,
    max_depth: u32,
    stack: Option<(usize, usize)>,
) -> Result<Next> {
    let _tag = crate::htag::Scope::new(crate::htag::TREE);
    let full: &str = &source.text[..limit];
    if let Some(end) = skip_comment(full, start) {
        return Ok(Next { skipped: true, parsed: None, end, last: false, stmt_end: end, stmt_span: (end, end) });
    }
    #[cfg(feature = "diag")]
    let warned = syms.warnings.len();
    loop {
        let mut wend = start.saturating_add(*window).min(full.len());
        while !full.is_char_boundary(wend) {
            wend -= 1;
        }
        let truncated = wend < full.len();
        let saved = module.clone();
        let mut p = Parser {
            lx: Lexer::new_window(&full[..wend], start, strict == 0, truncated),
            tree: Tree::new(Some(source.clone())),
            syms: &mut *syms,
            module: &mut *module,
            strict,
            last_list: None,
            dicts_read: 0,
            depth: 0,
            max_depth,
            stack,
            last_sp: (0, 0),
            #[cfg(feature = "diag")]
            rec: false,
            #[cfg(feature = "diag")]
            args_sp: Vec::new(),
            #[cfg(feature = "diag")]
            prog_sp: Vec::new(),
        };
        #[cfg(feature = "diag")]
        {
            p.rec = p.syms.record_spans;
            if p.rec {
                p.tree.aux = Some(alloc::boxed::Box::default());
            }
        }
        if p.lx.toks.is_empty() || !p.tree.reserve() {
            return Err(Error::memory());
        }
        let r = p.step().and_then(|s| match s {
            Some((_, true)) | None => p.finish().map(|_| s),
            _ => Ok(s),
        });
        if truncated && p.lx.touched.get() {
            // the parser reached the end of the window: what it made of it means nothing; undo the module switch and take a larger one
            *p.module = saved;
            drop(p);
            // (and what it warned about, which was about the window's end and not the text's)
            #[cfg(feature = "diag")]
            syms.warnings.truncate(warned);
            *window = window.saturating_mul(2);
            continue;
        }
        if p.lx.oom || p.tree.oom {
            p.syms.failure();
            return Err(Error::memory());
        }
        if let Some(e) = p.syms.failure() {
            return Err(if e.kind == crate::error::Kind::Range { Error::syntax("a name is longer than 255 bytes") } else { e });
        }
        return match r {
            Ok(None) => Ok(Next { skipped: false, parsed: None, end: full.len(), last: true, stmt_end: full.len(), stmt_span: (full.len(), full.len()) }),
            Ok(Some((q, last))) => {
                let end = if last { full.len() } else { p.lx.offset() as usize };
                let stmt_end = p.lx.end() as usize;
                let stmt_span = (p.last_sp.0 as usize, p.last_sp.1 as usize);
                let mut ids = crate::value::try_vec(1)?;
                ids.push(NodeId(q));
                p.tree.shrink();
                p.tree.lazy = lazy;
                Ok(Next { skipped: false, parsed: Some(Parsed { tree: Rc::new(p.tree), stmts: ids }), end, last, stmt_end, stmt_span })
            }
            Err(mut e) => {
                if e.source.is_none() && e.span.is_some() {
                    e.source = Some(source.clone());
                }
                Err(e)
            }
        };
    }
}

/// Where a statement that is `.comment("END")` ends, if one starts at `start` (after blanks): the text it hides is raw, so it is
/// found without being lexed (a block of prose is a thousand tokens, and a window large enough for it was most of a load's memory).
/// The same end the parser finds (`Parser::comment_end`); None for anything that is not that exact shape.
fn skip_comment(full: &str, start: usize) -> Option<usize> {
    let mut s = start;
    loop {
        let rest = &full[s..];
        let t = rest.trim_start_matches([' ', '\t', '\n', '\r']);
        s += rest.len() - t.len();
        // a `:"..."` comment before it (a quote inside is doubled)
        let Some(c) = full[s..].strip_prefix(":\"") else { break };
        let mut i = s + 2;
        loop {
            let q = full[i..].find('"')?;
            i += q + 1;
            if full[i..].starts_with('"') {
                i += 1;
            } else {
                break;
            }
        }
        let _ = c;
        s = i;
    }
    let body = full[s..].strip_prefix(".comment(\"")?;
    let q = body.find('"')?;
    let marker = &body[..q];
    if marker.is_empty() || !body[q..].starts_with("\")") {
        return None;
    }
    let i = s + ".comment(\"".len() + q + 2;
    let mut j = full[i..].find(marker)?;
    while full.is_char_boundary(i + j + 1) && full[i + j + 1..].starts_with(marker) {
        j += 1;
    }
    Some(i + j + marker.len())
}

/// One data object from the start of `text` (`Interp::read_datum`).
pub fn read_datum(text: &str, syms: &mut Interner, module: &mut Option<String>, max_depth: u32, stack: Option<(usize, usize)>) -> Result<Option<(usize, Value)>> {
    Ok(read_datum_ex(text, syms, module, max_depth, stack)?.map(|(used, v, _)| (used, v)))
}

/// `read_datum` that also says whether the object is complete: false when the text ended inside it (an open `[` or `:{`) or exactly where
/// an atom could go on (`12` may be the start of `123`). A caller that can get more text reads it and tries again.
pub fn read_datum_ex(text: &str, syms: &mut Interner, module: &mut Option<String>, max_depth: u32, stack: Option<(usize, usize)>) -> Result<Option<(usize, Value, bool)>> {
    let mut p = Parser { lx: Lexer::new(text, true), tree: Tree::new(None), syms, module, strict: 0, last_list: None, dicts_read: 0, depth: 0, max_depth, stack, last_sp: (0, 0), #[cfg(feature = "diag")] rec: false, #[cfg(feature = "diag")] args_sp: Vec::new(), #[cfg(feature = "diag")] prog_sp: Vec::new() };
    if p.lx.toks.is_empty() || !p.tree.reserve() {
        return Err(Error::memory());
    }
    let j = p.skip_nl(0);
    let tok = p.tok(j).clone();
    let r = match tok.k {
        K::Eof => Ok(None),
        K::LBrack | K::Dict => p.read_data(j, true).map(|v| {
            let end = p.tok(p.lx.pos.saturating_sub(1)).end;
            Some((end as usize, v, !p.lx.at_end))
        }),
        K::Num | K::Str | K::Char => {
            let v = match &tok.v {
                TV::Int(i) => Value::Int(*i),
                TV::Flt(f) => Value::Flt(*f),
                #[cfg(feature = "bigint")]
                TV::Big(b) => Value::Big(b.clone()),
                TV::Str(s) => Value::text(s)?,
                TV::Char(c) => Value::Char(*c),
                _ => Value::Undefined,
            };
            Ok(Some((tok.end as usize, v, tok.k != K::Num || (tok.end as usize) < text.len())))
        }
        K::Name | K::QName => {
            let s = p.sym_for(p.lx.text_of(&tok));
            Ok(Some((tok.end as usize, Value::Sym(s), (tok.end as usize) < text.len())))
        }
        K::Op if p.lx.text_of(&tok) == "-" => {
            let nt = p.tok(j + 1).clone();
            match (nt.k, &nt.v, nt.ws) {
                (K::Num, TV::Int(i), false) => Ok(Some((nt.end as usize, Value::Int(i.wrapping_neg()), (nt.end as usize) < text.len()))),
                (K::Num, TV::Flt(f), false) => Ok(Some((nt.end as usize, Value::Flt(-*f), (nt.end as usize) < text.len()))),
                #[cfg(feature = "bigint")]
                (K::Num, TV::Big(b), false) => Ok(Some((nt.end as usize, crate::big::negate(b), (nt.end as usize) < text.len()))),
                _ => Err(Error::syntax("not a data object").with_span((tok.start, tok.end))),
            }
        }
        K::Error => Err(p.raise_tok(&tok, false)),
        _ => Err(Error::syntax("not a data object").with_span((tok.start, tok.end))),
    };
    if p.lx.oom || p.tree.oom {
        p.syms.failure();
        return Err(Error::memory());
    }
    if let Some(e) = p.syms.failure() {
        return Err(if e.kind == crate::error::Kind::Range { Error::syntax("a name is longer than 255 bytes") } else { e });
    }
    r
}

impl<'a> Parser<'a> {
    #[inline]
    fn out_of_stack(&self) -> bool {
        match self.stack {
            Some((base, limit)) => base.abs_diff(crate::interp::stack_here()) > limit,
            None => false,
        }
    }

    /// Append to a list the parser builds whose length the program decides (arguments,
    /// statements, the elements of a literal). Without memory the element is dropped and the
    /// parse ends as a `memory` error (`parse`).
    fn keep<T>(&mut self, v: &mut Vec<T>, x: T) {
        if v.len() == v.capacity() && v.try_reserve(1).is_err() {
            self.tree.oom = true;
            return;
        }
        v.push(x);
    }

    fn tok(&self, j: usize) -> &Tok {
        self.lx.tok(j)
    }

    fn kind(&self, j: usize) -> K {
        self.lx.tok(j).k
    }

    /// Index of the first token at or after `j` that is not a plain newline.
    fn skip_nl(&self, mut j: usize) -> usize {
        loop {
            let t = self.tok(j);
            if t.k == K::Newline && !t.nlc {
                j += 1;
            } else {
                return j;
            }
        }
    }

    /// Remember where the operands of `id` were written (a no-op unless the interner asks for spans).
    #[cfg(feature = "diag")]
    fn rec_args(&mut self, id: NodeId, spans: Vec<Span>) {
        if self.rec {
            if let Some(a) = self.tree.aux.as_deref_mut() {
                a.args.insert(id.0, spans);
            }
        }
    }
    #[cfg(not(feature = "diag"))]
    fn rec_args(&mut self, _id: NodeId, _spans: Vec<Span>) {}
    #[cfg(feature = "diag")]
    fn rec_op(&mut self, id: NodeId, span: Span) {
        if self.rec {
            if let Some(a) = self.tree.aux.as_deref_mut() {
                a.ops.insert(id.0, span);
            }
        }
    }
    #[cfg(not(feature = "diag"))]
    fn rec_op(&mut self, _id: NodeId, _span: Span) {}

    fn add(&mut self, n: Node, span: Span) -> NodeId {
        self.tree.add(n, span)
    }

    fn lit_str(&mut self, s: &str) -> NodeId {
        self.add(Node::Lit(Value::str(String::from(s))), (0, 0))
    }

    /// A `;` read where an expression was expected is the string ";" (legacy).
    fn unsemi(&mut self, id: NodeId) -> NodeId {
        if id == SEMI {
            self.lit_str(";")
        } else {
            id
        }
    }

    fn sym_for(&mut self, v: &str) -> SymId {
        if let Some(Ok(segs)) = sym::qual_segments(v) {
            return self.qual_sym(v, &segs);
        }
        self.plain_sym(v)
    }

    /// The symbol of a name that is not qualified.
    fn plain_sym(&mut self, v: &str) -> SymId {
        match v {
            "x" => sym::X,
            "y" => sym::Y,
            "z" => sym::Z,
            _ => match &*self.module {
                Some(m) if !v.starts_with('.') => {
                    // (two names: under the size the allocator treats as sized by data)
                    if v.len() > crate::sym::MAX_NAME {
                        self.syms.long = true;
                        return sym::X;
                    }
                    let q = format!("{}`{}", v, m);
                    self.syms.intern(&q)
                }
                _ => self.syms.intern(v),
            },
        }
    }

    /// The symbol of `a.b.c` and, recorded with it, the symbol of `a` and the keys `"b"` and `"c"`: `a.b` is `a@"b"` (design point 4).
    fn qual_sym(&mut self, v: &str, segs: &[&str]) -> SymId {
        let base = self.plain_sym(segs[0]);
        let mut full = String::from(v);
        if let Some(m) = &*self.module {
            full.push('`');
            full.push_str(m);
        }
        let s = self.syms.intern(&full);
        if self.syms.qual(s).is_none() {
            let mut keys = Vec::new();
            for k in &segs[1..] {
                match Value::text(k) {
                    Ok(t) => keys.push(t),
                    Err(_) => {
                        self.syms.oom = true;
                        return s;
                    }
                }
            }
            self.syms.set_qual(s, base, keys);
        }
        s
    }

    /// A local or a parameter is a plain name: `d.a` is an entry of `d`, not a variable to declare.
    fn plain_names(&self, ids: &[u32], span: Span) -> Result<()> {
        for &q in ids {
            if self.syms.qual(SymId(q)).is_some() {
                let e = Error::syntax(crate::msg!("{} is a qualified name; a local or a parameter must be a plain name", self.syms.name(SymId(q)).split('`').next().unwrap_or(""))).with_span(span);
                #[cfg(feature = "diag")]
                let e = e.with_hint("declare the dictionary itself, {[d]; ...}, and write d.a in the body");
                return Err(e);
            }
        }
        Ok(())
    }

    /// A qualified name written as an expression: its segments must be names.
    fn check_qualified(&self, v: &str, span: Span) -> Result<()> {
        if let Some(Err(why)) = sym::qual_segments(v) {
            let e = Error::syntax(crate::msg!("{}: {}", v, why)).with_span(span);
            #[cfg(feature = "diag")]
            let e = e.with_hint("a.b means a@\"b\"; a name that starts with a dot (.p, .srv.o) is one name");
            return Err(e);
        }
        Ok(())
    }

    /// Something the legacy grammar accepts and should not: an error at strict level 1 and
    /// above, accepted at level 0.
    ///
    /// `code` names the condition (KlongPy's `KlongSyntaxWarning.code`), `hint` is one line of advice (given by the error at level 1 and by
    /// the warning at level 0) and `note` says what happens to the input at level 0 (part of the warning only). At level 0 the warning is
    /// recorded in the interner (`Interner::warnings`) for the embedder to report.
    fn issue(&mut self, code: &'static str, message: String, span: Span, hint: Option<String>, note: Option<&str>) -> Result<()> {
        if self.strict > 0 {
            let e = Error::syntax(message).with_span(span);
            #[cfg(feature = "diag")]
            if let Some(h) = hint {
                return Err(e.with_hint(h));
            }
            return Err(e);
        }
        #[cfg(feature = "diag")]
        {
            let w = crate::sym::Warning { code, message, note: note.map(String::from), hint, span, source: self.tree.source.clone() };
            self.syms.warn(w);
        }
        #[cfg(not(feature = "diag"))]
        let _ = (code, hint, note);
        Ok(())
    }

    /// The hint when a string scanned so far ends in a backslash (the writer meant `\"` for a quote).
    #[cfg(feature = "diag")]
    fn escaped_quote_hint(&self) -> Option<String> {
        let text = self.lx.text.as_bytes();
        let found = self.lx.toks.iter().any(|t| t.k == K::Str && !t.unterminated && t.end - t.start >= 2 && text[t.end as usize - 2] == b'\\');
        if found {
            Some("a quote inside a string is written twice: \"\"; a backslash is an ordinary character".to_string())
        } else {
            None
        }
    }

    /// A token as written, quoted and clipped, for a message (KlongPy `_lexeme`).
    #[cfg(not(feature = "diag"))]
    fn lexeme(&self, _tok: &Tok) -> String {
        String::new()
    }
    #[cfg(not(feature = "diag"))]
    fn index_hint(&self, _j: usize) -> Option<String> {
        None
    }
    #[cfg(not(feature = "diag"))]
    fn escaped_quote_hint(&self) -> Option<String> {
        None
    }

    #[cfg(feature = "diag")]
    fn lexeme(&self, tok: &Tok) -> String {
        let s = &self.lx.text[tok.start as usize..tok.end as usize];
        let clipped = s.chars().count() > 20 || s.contains('\n');
        if clipped {
            let first = s.split('\n').next().unwrap_or("");
            let mut out: String = first.chars().take(20).collect();
            out.push_str("...");
            crate::msg!("'{}'", out)
        } else {
            crate::msg!("'{}'", s)
        }
    }

    /// The hint for `name[...]` (token `j` is the `[`), or `None` (KlongPy `tk_index_hint`).
    #[cfg(feature = "diag")]
    fn index_hint(&self, j: usize) -> Option<String> {
        let tok = self.tok(j);
        if tok.k != K::LBrack || tok.ws || j == 0 {
            return None;
        }
        let prev = self.tok(j - 1);
        if !matches!(prev.k, K::Name | K::QName | K::RBrack | K::RParen) {
            return None;
        }
        let text = self.lx.text;
        let m = self.matching(j);
        if prev.k == K::Name {
            if let Some(m) = m {
                let name = &text[prev.start as usize..prev.end as usize];
                let inner = text[tok.end as usize..self.tok(m).start as usize].trim();
                if m == j + 2 && matches!(self.tok(j + 1).k, K::Num | K::Name | K::Str | K::Char) {
                    return Some(crate::msg!("use {}@{} to index", name, inner));
                }
                if inner.chars().count() <= 24 && !inner.contains('\n') {
                    return Some(crate::msg!("use {}@[{}] to index", name, inner));
                }
            }
        }
        Some("use @ to index: a@1 for one element, a@[1 2] for several".to_string())
    }

    /// `index`: this is where a `name[...]` that was meant as an index is noticed, and the hint says so (the end of a function body and of a
    /// parenthesised expression; an argument list or a `;` expected does not).
    fn unexpected(&self, at: u32, want: char, index: bool) -> Error {
        let _ = index;
        let text = self.lx.text;
        let n = text.len();
        let i = (at as usize).min(n);
        // KlongPy `_found_at`: whitespace is skipped to name the lexeme after it, unless that is the expected one (then the whitespace is what is wrong)
        let mut j = i;
        for c in text[i..].chars() {
            if c != '\n' && c.is_whitespace() {
                j += c.len_utf8();
            } else {
                break;
            }
        }
        let (found, span): (String, (u32, u32)) = if j >= n {
            ("end of input".to_string(), (n as u32, n as u32))
        } else if text.as_bytes()[j] == b'\n' {
            ("end of line".to_string(), (j as u32, j as u32 + 1))
        } else {
            let rest = &text[j..];
            let b = rest.as_bytes();
            let name_start = |c: u8| c.is_ascii_alphabetic() || c == b'.';
            let colon = (b[0] == b':') as usize;
            let len = if b.len() > colon && name_start(b[colon]) {
                let mut k = colon + 1;
                while k < b.len() && (b[k].is_ascii_alphanumeric() || b[k] == b'.') {
                    k += 1;
                }
                k
            } else if b[0].is_ascii_digit() {
                let mut k = 1;
                while k < b.len() && (b[k].is_ascii_digit() || b[k] == b'.') {
                    k += 1;
                }
                k
            } else if b[0] == b'"' {
                let mut k = 1;
                while k < b.len() && k <= 16 && b[k] != b'"' && b[k] != b'\n' {
                    k += 1;
                }
                if k < b.len() && b[k] == b'"' {
                    k += 1;
                }
                k
            } else {
                colon + rest[colon..].chars().next().map(|c| c.len_utf8()).unwrap_or(0)
            };
            let lexeme = &rest[..len.min(rest.len())];
            if j > i && lexeme.starts_with(want) {
                ("whitespace".to_string(), (i as u32, j as u32))
            } else {
                (crate::msg!("'{}'", lexeme), (j as u32, (j + lexeme.len()) as u32))
            }
        };
        let e = Error::syntax(crate::msg!("expected '{}' but found {}", want, found)).with_span(span);
        #[cfg(feature = "diag")]
        if let Some(j) = (0..self.lx.toks.len()).filter(|_| index).find(|&j| self.lx.toks[j].start == span.0) {
            if let Some(h) = self.index_hint(j) {
                return e.with_hint(h);
            }
        }
        #[cfg(feature = "diag")]
        if let Some(h) = self.escaped_quote_hint() {
            return e.with_hint(h);
        }
        e
    }

    /// What the legacy readers raised for an error token; `neg`: the token follows a `-` that the data-list reader took as its sign.
    fn raise_tok(&self, tok: &Tok, neg: bool) -> Error {
        let m = match tok.v {
            TV::CharEof => "expected a character after 0c but found end of input".to_string(),
            _ => {
                // (a `:` directly before the digits is part of the token's span, not of the number)
                let t = &self.lx.text[tok.start as usize..tok.end as usize];
                let t = t.strip_prefix(':').unwrap_or(t);
                if neg {
                    crate::msg!("malformed number '-{}'", t)
                } else {
                    crate::msg!("malformed number '{}'", t)
                }
            }
        };
        let start = if neg { tok.start - 1 } else { tok.start };
        Error::syntax(m).with_span((start, tok.end))
    }

    // ----- adverbs -----------------------------------------------------------------------

    /// If an adverb starts exactly at the read position, consume it and return its glyph.
    fn peek_adverb(&mut self) -> Option<&'static str> {
        let j = self.lx.pos;
        let tok = self.tok(j);
        if tok.ws {
            return None;
        }
        let v = self.lx.text_of(tok);
        match tok.k {
            K::Op => {
                if v == "@" {
                    let nt = self.tok(j + 1);
                    if nt.k == K::Op && self.lx.text_of(nt) == "'" && !nt.ws {
                        self.lx.pos = j + 2;
                        return Some("@'");
                    }
                    return None;
                }
                for a in ["'", "/", "\\", "\\~", "\\*"] {
                    if v == a {
                        self.lx.pos = j + 1;
                        return Some(a);
                    }
                }
                None
            }
            K::Cop => {
                for a in [":'", ":/", ":\\", ":~", ":*"] {
                    if v == a {
                        self.lx.pos = j + 1;
                        return Some(a);
                    }
                }
                None
            }
            _ => None,
        }
    }

    /// The read position is just after the first adverb `aa` of the verb `a`.
    #[allow(clippy::too_many_arguments)]
    fn apply_adverbs(&mut self, a: NodeId, aa: &'static str, arity: u8, st: u32, ost: u32, dyad: bool, l: NodeId, lsp: Span) -> Result<NodeId> {
        let verb_arity = adverb_arity(aa, arity);
        if let Node::OpVal { arity: oa, .. } = self.tree.node_mut(a) {
            *oa = verb_arity;
        }
        let mut advs = alloc::vec![self.syms.intern(aa).0];
        while let Some(more) = self.peek_adverb() {
            let id = self.syms.intern(more).0;
            self.keep(&mut advs, id);
        }
        let oe = self.lx.end();
        // legacy: the operand is read without newline skipping, wherever we are
        let mut r = self.expr(false)?;
        let rsp = if r == NONE || r == SEMI { (0, 0) } else { self.last_sp };
        if r == NONE || r == SEMI {
            let what = &self.lx.text[ost as usize..oe as usize];
            let what = what.to_string();
            self.issue("missing-operand", if dyad { crate::msg!("missing right operand for {}", what) } else { crate::msg!("missing operand for {}", what) }, (ost, oe), None, None)?;
            r = self.unsemi(r);
        }
        let advs = self.tree.range(&advs);
        let e = self.lx.end();
        let id = self.add(Node::Adv { verb: a, verb_arity, advs, first_arity: arity, dyad, l, r }, (st, e));
        #[cfg(feature = "diag")]
        self.rec_args(id, if dyad { alloc::vec![lsp, rsp] } else { alloc::vec![rsp] });
        let _ = (lsp, rsp);
        self.rec_op(id, (ost, oe));
        Ok(id)
    }

    // ----- argument lists ----------------------------------------------------------------

    /// The token at the read position is `(` or `:(`. Returns the arguments (`NONE` for an
    /// omitted one).
    fn read_fn_args(&mut self) -> Result<Vec<u32>> {
        let mut j = self.lx.pos + 1;
        self.lx.pos = j;
        let mut arr: Vec<u32> = Vec::new();
        let tok = self.tok(j);
        if tok.k == K::RParen && !tok.ws {
            self.lx.pos = j + 1;
            #[cfg(feature = "diag")]
            self.args_sp.clear();
            return Ok(arr);
        }
        // legacy: an argument is omitted only where a ';' or the ')' directly follows the
        // previous separator, with nothing in between.
        let mut sep = j;
        let mut fresh = true;
        #[cfg(feature = "diag")]
        let mut asp: Vec<Span> = Vec::new();
        loop {
            j = self.skip_nl(self.lx.pos);
            let tok = self.tok(j).clone();
            match tok.k {
                K::Semi | K::Newline => {
                    if j == sep && !tok.ws {
                        self.keep(&mut arr, NONE.0);
                        #[cfg(feature = "diag")]
                        asp.push((0, 0));
                    }
                    sep = j + 1;
                    self.lx.pos = sep;
                    fresh = true;
                    continue;
                }
                K::RParen => {
                    if j == sep && !tok.ws {
                        self.keep(&mut arr, NONE.0);
                        #[cfg(feature = "diag")]
                        asp.push((0, 0));
                    }
                    break;
                }
                _ => {}
            }
            if !fresh && tok.k != K::Eof {
                let hint = self.index_hint(j).unwrap_or_else(|| "arguments are separated by ;".to_string());
                let m = crate::msg!("unexpected {} after the end of the argument", self.lexeme(&tok));
                self.issue("trailing-input", m, (tok.start, tok.end), Some(hint), Some("it is passed as one more argument"))?;
            }
            if tok.k == K::LBrack {
                let nt = self.tok(j + 1);
                if nt.k == K::Semi && !nt.ws {
                    self.check(j)?;
                }
            }
            let a = self.expr(true)?;
            if a == NONE {
                break;
            }
            let a = self.unsemi(a);
            self.keep(&mut arr, a.0);
            #[cfg(feature = "diag")]
            asp.push(self.last_sp);
            fresh = false;
        }
        #[cfg(feature = "diag")]
        {
            self.args_sp = asp;
        }
        // legacy: the ')' must follow the last argument directly
        let tok = self.tok(self.lx.pos);
        if tok.k != K::RParen || tok.ws {
            return Err(self.unexpected(self.lx.offset(), ')', false));
        }
        self.lx.pos += 1;
        Ok(arr)
    }

    // ----- function literals -------------------------------------------------------------

    /// (highest of x=1 y=2 z=3 mentioned, whether .f is mentioned); KlongPy `_scan_args`.
    fn scan_args(&self, id: NodeId) -> (u8, bool) {
        if !Tree::is_real(id) {
            return (0, false);
        }
        let join = |a: (u8, bool), b: (u8, bool)| (a.0.max(b.0), a.1 || b.1);
        let each = |r: Range| {
            let mut m = (0, false);
            for &q in self.tree.ids(r) {
                m = join(m, self.scan_args(NodeId(q)));
            }
            m
        };
        match self.tree.node(id) {
            Node::Sym(s) | Node::QSym(s) => {
                if sym::is_arg(*s) {
                    (s.0 as u8 + 1, false)
                } else if let Some(q) = self.syms.qual(*s).filter(|q| sym::is_arg(q.base)) {
                    // x.len is x@"len": it uses x
                    (q.base.0 as u8 + 1, false)
                } else {
                    (0, *s == sym::DOT_F)
                }
            }
            Node::Fn { a, args, has_args, .. } => {
                let mut m = if *has_args && matches!(self.tree.node(*a), Node::Sym(_) | Node::QSym(_)) { self.scan_args(*a) } else { (0, false) };
                if *has_args {
                    m = join(m, each(*args));
                }
                m
            }
            Node::Monad { arg, .. } => self.scan_args(*arg),
            Node::Dyad { l, r, .. } => join(self.scan_args(*l), self.scan_args(*r)),
            Node::Adv { verb, l, r, .. } => join(self.scan_args(*verb), join(self.scan_args(*l), self.scan_args(*r))),
            Node::Cond(c, t, e) => join(self.scan_args(*c), join(self.scan_args(*t), self.scan_args(*e))),
            Node::EList(r) | Node::EDict(r) | Node::Seq(r) => each(*r),
            Node::Body { stmts, .. } => each(*stmts),
            _ => (0, false),
        }
    }

    /// The arity of a function body (KlongPy `get_fn_arity`): the highest argument
    /// it mentions; a body that is itself a function value takes that value's arity.
    fn fn_arity(&self, a: NodeId) -> u8 {
        let m = self.scan_args(a).0;
        if !Tree::is_real(a) {
            return m;
        }
        if let Node::Fn { args, has_args, arity, .. } = self.tree.node(a) {
            if !*has_args {
                return m.max(*arity);
            }
            let holes = self.tree.ids(*args).iter().filter(|&&q| q == NONE.0).count() as u8;
            return m.max(holes);
        }
        m
    }

    /// The names a statement declares as locals: a non-empty data list that, leaving out
    /// ";", holds only symbols.
    fn decl_names(&self, id: NodeId) -> Option<Vec<u32>> {
        let v = match self.tree.node(id) {
            Node::Lit(v) if v.is_list() && !v.is_empty() => v,
            _ => return None,
        };
        let mut names = Vec::new();
        for i in 0..v.len().unwrap_or(0) {
            match v.item(i) {
                Value::Sym(s) => names.push(s.0),
                Value::Str(s) if s.as_str() == ";" => {}
                _ => return None,
            }
        }
        if names.is_empty() {
            None
        } else {
            Some(names)
        }
    }

    /// Read a function literal and the argument list that may follow it. The read position is
    /// just after the `{` whose offset is `st`.
    fn read_fn(&mut self, st: u32) -> Result<NodeId> {
        let body = self.prog(true)?;
        #[cfg(feature = "diag")]
        let mut bsp = core::mem::take(&mut self.prog_sp);
        // A body of one statement is that statement, except an evaluated dictionary literal.
        let mut a = if body.len() == 1 && !matches!(self.tree.node(NodeId(body[0])), Node::EDict(_)) {
            NodeId(body[0])
        } else {
            let r = self.tree.range(&body);
            self.add(Node::Seq(r), (0, 0))
        };
        #[cfg(feature = "diag")]
        let mut locals_sp: Option<Span> = None;
        let j = self.skip_nl(self.lx.pos);
        let tok = self.tok(j);
        if tok.k != K::RBrace {
            return Err(self.unexpected(tok.start, '}', true));
        }
        let rbrace_end = tok.end;
        self.lx.pos = j + 1;
        let arity = self.fn_arity(a);
        // Function metadata: a locals declaration leaves the stored body.
        if let Node::Seq(r) = self.tree.node(a) {
            let r = *r;
            if r.len > 1 {
                if let Some(names) = self.decl_names(NodeId(self.tree.ids(r)[0])) {
                    self.plain_names(&names, (st, rbrace_end))?;
                    let locals = self.tree.range(&names);
                    let stmts = Range { start: r.start + 1, len: r.len - 1 };
                    a = self.add(Node::Body { stmts, locals, params: Range::default(), has_params: false }, (0, 0));
                    #[cfg(feature = "diag")]
                    if !bsp.is_empty() {
                        locals_sp = Some(bsp.remove(0));
                    }
                }
            }
        }
        // The targets of a destructuring that heads the body are locals.
        a = self.head_destructuring(a);
        let nt = self.tok(j + 1);
        let called = matches!(nt.k, K::LParen | K::CallP) && !nt.ws;
        let node = if called {
            let fa = self.read_fn_args()?;
            let call = !fa.contains(&NONE.0);
            let args = self.tree.range(&fa);
            Node::Fn { a, args, has_args: true, arity, call, literal: true }
        } else {
            Node::Fn { a, args: Range::default(), has_args: false, arity, call: false, literal: false }
        };
        let e = self.lx.end();
        let id = self.add(node, (st, e));
        #[cfg(feature = "diag")]
        if self.rec {
            let asp = if called { core::mem::take(&mut self.args_sp) } else { Vec::new() };
            if let Some(x) = self.tree.aux.as_deref_mut() {
                x.stmts.insert(id.0, bsp);
                if let Some(l) = locals_sp {
                    x.locals.insert(id.0, l);
                }
                if called {
                    x.args.insert(id.0, asp);
                    x.ops.insert(id.0, (st, rbrace_end));
                }
            }
        }
        let _ = rbrace_end;
        Ok(id)
    }

    fn head_destructuring(&mut self, a: NodeId) -> NodeId {
        let (first, stmts, mut names, params, has_params) = match self.tree.node(a) {
            Node::Body { stmts, locals, params, has_params } => {
                if stmts.len == 0 {
                    return a;
                }
                (NodeId(self.tree.ids(*stmts)[0]), Some(*stmts), self.tree.ids(*locals).to_vec(), *params, *has_params)
            }
            Node::Seq(r) => {
                if r.len == 0 {
                    return a;
                }
                (NodeId(self.tree.ids(*r)[0]), Some(*r), Vec::new(), Range::default(), false)
            }
            _ => (a, None, Vec::new(), Range::default(), false),
        };
        if !Tree::is_real(first) {
            return a;
        }
        let targets = match self.tree.node(first) {
            Node::Dyad { op, l, .. } if self.syms.name(*op) == "::" && Tree::is_real(*l) => match self.tree.node(*l) {
                Node::Destructure(r) => self.tree.ids(*r).to_vec(),
                _ => return a,
            },
            _ => return a,
        };
        for t in targets {
            if !names.contains(&t) && self.syms.qual(SymId(t)).is_none() {
                names.push(t);
            }
        }
        let stmts = match stmts {
            Some(r) => r,
            None => self.tree.range(&[a.0]),
        };
        let locals = self.tree.range(&names);
        self.add(Node::Body { stmts, locals, params, has_params }, (0, 0))
    }

    // ----- assignment targets ---------------------------------------

    fn matching(&self, j: usize) -> Option<usize> {
        let mut depth = 0i32;
        for i in j..self.lx.toks.len() {
            match self.kind(i) {
                K::LParen | K::CallP | K::LBrack | K::LBrace | K::Cond | K::Dict => depth += 1,
                K::RParen | K::RBrack | K::RBrace => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(i);
                    }
                }
                _ => {}
            }
        }
        None
    }

    /// The list literal whose `[` is token `j` is the left side of `::`. True if it is a list
    /// of plain names: a destructuring assignment.
    fn destructure_ok(&mut self, j: usize) -> Result<bool> {
        let mut skip_marker = false;
        let mut names: Vec<&str> = Vec::new();
        let mut why: Option<String> = None;
        let mut i = j + 1;
        loop {
            let tok = self.tok(i);
            match tok.k {
                K::RBrack | K::Eof => break,
                K::Newline if !tok.nlc => {
                    i += 1;
                    continue;
                }
                K::Name => {
                    let v = self.lx.text_of(tok);
                    if why.is_none() {
                        if matches!(v, "x" | "y" | "z") {
                            why = Some("x, y and z are the arguments of a function and cannot be assigned".to_string());
                        } else if v == ".f" {
                            why = Some(".f is the running function and cannot be assigned".to_string());
                        } else if names.contains(&v) {
                            why = Some(crate::msg!("{} is named twice", v));
                        }
                    }
                    names.push(v);
                }
                K::LBrack => {
                    why = Some("nested lists are not supported".to_string());
                    break;
                }
                _ => {
                    if why.is_none() {
                        why = Some("the list on the left of :: must hold only names".to_string());
                    }
                    if tok.k == K::Op && self.lx.text_of(tok) == "_" {
                        skip_marker = true;
                    }
                }
            }
            i += 1;
        }
        if why.is_none() && names.is_empty() {
            why = Some("the list on the left of :: is empty".to_string());
        }
        match why {
            Some(w) => {
                let end = self.matching(j).map(|m| self.tok(m).end).unwrap_or(self.tok(j).end);
                let start = self.tok(j).start;
                let hint = if w.starts_with("nested lists") {
                    "destructure in two steps: [a t]::value;[b c]::t"
                } else if skip_marker {
                    "there is no skip marker: name every element"
                } else {
                    "write [a b c]::value with plain names, one for each element of value"
                };
                self.issue("not-a-destructuring", crate::msg!("cannot destructure: {}", w), (start, end), Some(hint.to_string()), Some("nothing is assigned"))?;
                Ok(false)
            }
            None => Ok(true),
        }
    }

    /// `a`, the left side of `::`, is a call. `f(a;b)` with plain names defines `f` with those
    /// parameters.
    fn named_def(&mut self, a: NodeId, st: u32, lend: u32) -> Result<Option<NamedDef>> {
        let (callee, args) = match self.tree.node(a) {
            Node::Fn { a: callee, args, has_args: true, call: true, .. } => (*callee, *args),
            _ => return Ok(None),
        };
        let name = match self.tree.node(callee) {
            Node::Sym(s) if !sym::is_arg(*s) => *s,
            _ => return Ok(None),
        };
        let start = self.tree.span(a).map(|s| s.0).unwrap_or(st) as usize;
        if self.lx.text.as_bytes().get(start) == Some(&b'{') {
            return Ok(None);
        }
        let params = self.tree.ids(args).to_vec();
        for &q in &params {
            if q == NONE.0 || !matches!(self.tree.node(NodeId(q)), Node::Sym(_)) {
                return Ok(None);
            }
        }
        let ids: Vec<u32> = params
            .iter()
            .map(|&q| match self.tree.node(NodeId(q)) {
                Node::Sym(s) => s.0,
                _ => 0,
            })
            .collect();
        self.plain_names(&ids, (st, lend))?;
        let mut why = None;
        for (i, &s) in ids.iter().enumerate() {
            if s < 3 {
                why = Some("x, y and z cannot be parameter names".to_string());
                break;
            }
            if ids[..i].contains(&s) {
                why = Some(crate::msg!("{} is named twice", self.syms.name(SymId(s)).split('`').next().unwrap_or("")));
                break;
            }
        }
        if let Some(w) = why {
            let n = self.syms.name(name).split('`').next().unwrap_or("").to_string();
            self.issue(
                "not-a-definition",
                crate::msg!("cannot define {} with named parameters: {}", n, w),
                (st, lend),
                Some("name each parameter once, with names other than x, y and z: f(a;b)::{a+b}".to_string()),
                Some("nothing is defined"),
            )?;
            return Ok(None);
        }
        Ok(Some(NamedDef { name, params: ids }))
    }

    fn def_target(&mut self, a: NodeId, st: u32, lend: u32) -> Result<Target> {
        if let Some((list, j)) = self.last_list {
            if list == a {
                if self.destructure_ok(j)? {
                    let names: Vec<u32> = match self.tree.node(a) {
                        Node::Lit(v) => (0..v.len().unwrap_or(0))
                            .filter_map(|i| match v.item(i) {
                                Value::Sym(s) => Some(s.0),
                                _ => None,
                            })
                            .collect(),
                        _ => Vec::new(),
                    };
                    let r = self.tree.range(&names);
                    return Ok(Target::Node(self.add(Node::Destructure(r), (st, lend))));
                }
                return Ok(Target::Node(a));
            }
        }
        if let Some(nd) = self.named_def(a, st, lend)? {
            return Ok(Target::Named(nd));
        }
        if self.strict > 0 {
            return Err(self.not_assignable(a, st, lend));
        }
        Ok(Target::Node(a))
    }

    fn not_assignable(&self, a: NodeId, st: u32, lend: u32) -> Error {
        let e = Error::syntax("cannot assign: the left side of :: must be a name").with_span((st, lend));
        // KlongPy `tk_assign_hint`: what the writer probably meant
        #[cfg(feature = "diag")]
        if Tree::is_real(a) {
            match self.tree.node(a) {
                Node::Fn { a: callee, has_args: true, .. } if matches!(self.tree.node(*callee), Node::Sym(_) | Node::QSym(_)) => {
                    if let Node::Sym(s) | Node::QSym(s) = self.tree.node(*callee) {
                        let name = self.syms.name(*s);
                        return e.with_hint(crate::msg!("to define a function write {}::{{...}} and use x, y and z for its arguments, or name its parameters: f(a;b)::{{...}}, with plain names between the parentheses", name));
                    }
                }
                Node::Dyad { op, .. } if self.syms.name(*op) == "@" => return e.with_hint("to change an element use amend, a:=value,index"),
                Node::Lit(v) if v.is_list() => return e.with_hint("assign one name at a time: a::1;b::2"),
                Node::Cond(..) | Node::EList(_) | Node::EDict(_) | Node::CopyList(_) => return e.with_hint("assign one name at a time: a::1;b::2"),
                _ => {}
            }
        }
        let _ = a;
        e
    }

    /// The first token that is x, y or z used by the function literal whose `{` is token `i`
    /// (ending before token `end`) itself: a nested literal has its own, and in a data list a
    /// name is a symbol.
    fn direct_arg(&self, mut i: usize, end: usize) -> Option<Tok> {
        let mut stack: Vec<u8> = Vec::new();
        let mut data = 0;
        let mut fnd = 0;
        i += 1;
        while i < end {
            // (this only improves an error message: without memory, no message)
            if stack.try_reserve(1).is_err() {
                return None;
            }
            let tok = self.tok(i);
            let k = tok.k;
            if data > 0 {
                if k == K::LBrack {
                    stack.push(b'l');
                    data += 1;
                } else if k == K::Dict {
                    stack.push(b'd');
                    data += 1;
                } else if (k == K::RBrack && stack.last() == Some(&b'l')) || (k == K::RBrace && stack.last() == Some(&b'd')) {
                    stack.pop();
                    data -= 1;
                }
            } else if k == K::Name {
                if fnd == 0 && matches!(self.lx.text_of(tok).split('.').next(), Some("x" | "y" | "z")) && !self.lx.text_of(tok).starts_with('.') {
                    return Some(tok.clone());
                }
            } else if k == K::LBrace {
                stack.push(b'f');
                fnd += 1;
            } else if k == K::LBrack || k == K::Dict {
                let nt = self.tok(i + 1);
                if nt.k == K::Semi && !nt.ws {
                    stack.push(b'c');
                } else {
                    stack.push(if k == K::LBrack { b'l' } else { b'd' });
                    data += 1;
                }
            } else if matches!(k, K::LParen | K::CallP | K::Cond) {
                stack.push(b'c');
            } else if matches!(k, K::RParen | K::RBrack | K::RBrace) && stack.pop() == Some(b'f') {
                fnd -= 1;
            }
            i += 1;
        }
        None
    }

    /// The first `{` token between token indexes `j0` and `j1` that sits directly inside a `[ ]` list literal (which holds data, so the `{`
    /// starts no function), or `None`. `[;...]` lists and index brackets are not list literals.
    #[cfg(feature = "diag")]
    fn brace_in_list(&mut self, j0: usize, j1: usize) -> Option<usize> {
        let mut stack: Vec<(K, usize)> = Vec::new();
        let mut i = j0;
        while i <= j1 {
            let k = self.kind(i);
            if k == K::Eof {
                break;
            }
            if k == K::LBrace {
                if let Some(&(t, b)) = stack.last() {
                    if t == K::LBrack && self.kind(b + 1) != K::Semi {
                        let prev = if b > 0 { Some(self.kind(b - 1)) } else { None };
                        let index = matches!(prev, Some(K::Name | K::QName | K::RBrack | K::RParen | K::RBrace)) && !self.tok(b).ws;
                        let decl = prev == Some(K::LBrace) && matches!(self.kind(b + 1), K::Name | K::QName);
                        if !index && !decl {
                            return Some(i);
                        }
                    }
                }
                stack.push((k, i));
            } else if matches!(k, K::LParen | K::CallP | K::LBrack | K::Cond | K::Dict) {
                stack.push((k, i));
            } else if matches!(k, K::RParen | K::RBrack | K::RBrace) {
                stack.pop();
            }
            i += 1;
        }
        None
    }

    /// Finish `f(a;b)::{...}`: the right side `r` (read from token `rj` on) must be a function
    /// literal that does not use x, y or z itself; it becomes a function of the parameters.
    fn finish_named_def(&mut self, nd: NamedDef, r: NodeId, mut rj: usize, osp: Span) -> Result<NodeId> {
        let body = match (Tree::is_real(r), if Tree::is_real(r) { Some(self.tree.node(r)) } else { None }) {
            (true, Some(Node::Fn { a, has_args: false, .. })) => *a,
            _ => {
                #[cfg(feature = "diag")]
                if let Some(b) = self.brace_in_list(rj, self.lx.pos) {
                    let t = self.tok(b);
                    return Err(Error::syntax("a { inside a [ ] list is not a function: a list holds data")
                        .with_span((t.start, t.end))
                        .with_hint("use [;...] for a list of expressions (a function can be a member there), or write the function outside the list"));
                }
                // a node the reader gives a span (a call, an operation, a conditional, a literal expression list or dictionary); any
                // other right side (a number, a string, a name, a stray closer read as a string) is the text from the start of the
                // factor to the last token read
                let spanned = Tree::is_real(r)
                    && !matches!(self.tree.node(r), Node::Lit(_) | Node::Sym(_) | Node::QSym(_) | Node::OpVal { .. } | Node::CopyList(_) | Node::Destructure(_));
                let semi = Tree::is_real(r) && matches!(self.tree.node(r), Node::Lit(Value::Str(s)) if s.len() == 1 && s.char_at(0) == ';');
                let span = if !Tree::is_real(r) || semi {
                    osp
                } else if spanned {
                    self.tree.span(r).unwrap_or(osp)
                } else {
                    (self.lx.start, self.lx.end())
                };
                return Err(Error::syntax("named-parameter definition needs a function body in braces")
                    .with_span(span)
                    .with_hint("write f(a;b)::{a+b}: the right side is one function literal"));
            }
        };
        let rstart = self.tree.span(r).map(|s| s.0).unwrap_or(0);
        while self.tok(rj).start < rstart && self.kind(rj) != K::Eof {
            rj += 1;
        }
        if let Some(tok) = self.direct_arg(rj, self.lx.pos) {
            return Err(Error::syntax(crate::msg!("{} is not available in a function with named parameters", self.lx.text_of(&tok)))
                .with_span((tok.start, tok.end))
                .with_hint("use the parameter names; x, y and z exist only in a function without a parameter list (a nested {...} has its own)"));
        }
        let params = self.tree.range(&nd.params);
        let new_body = match self.tree.node(body) {
            Node::Body { stmts, locals, .. } => Node::Body { stmts: *stmts, locals: *locals, params, has_params: true },
            Node::Seq(s) => Node::Body { stmts: *s, locals: Range::default(), params, has_params: true },
            _ => {
                let stmts = self.tree.range(&[body.0]);
                Node::Body { stmts, locals: Range::default(), params, has_params: true }
            }
        };
        let nb = self.add(new_body, (0, 0));
        if let Node::Fn { a, arity, .. } = self.tree.node_mut(r) {
            *a = nb;
            *arity = nd.params.len() as u8;
        }
        Ok(self.add(Node::Sym(nd.name), (0, 0)))
    }

    // ----- data mode: list and dictionary literals -----------------------------------------

    /// Read the body of a list (`close` is `RBrack`) or of a dictionary literal (`RBrace`) in
    /// data mode, starting at token `j` (just after the opener). Returns the index after the
    /// closer and the elements. Nothing is consumed.
    fn read_list(&mut self, mut j: usize, close: K) -> Result<(usize, Vec<Value>)> {
        self.depth += 1;
        if self.depth > self.max_depth || self.out_of_stack() {
            return Err(Error::range("expression nested too deeply"));
        }
        let opener = self.tok(j - 1).clone();
        let mut arr: Vec<Value> = Vec::new();
        j = self.skip_nl(j);
        loop {
            let mut tok = self.tok(j).clone();
            if tok.k == close || tok.k == K::Eof {
                break;
            }
            if tok.k == K::Newline {
                // legacy: a newline after a comment is skipped and whatever comes next is
                // read as a datum, even the closer, and a second such newline as ";".
                j = self.skip_nl(j + 1);
                tok = self.tok(j).clone();
                if tok.k == K::Eof {
                    break;
                }
            }
            match tok.k {
                K::Num => {
                    let v = match tok.v {
                        TV::Int(i) => Value::Int(i),
                        TV::Flt(f) => Value::Flt(f),
                        #[cfg(feature = "bigint")]
                        TV::Big(b) => Value::Big(b),
                        _ => Value::Undefined,
                    };
                    self.keep(&mut arr, v);
                    j += 1;
                }
                K::Str => {
                    if let TV::Str(s) = &tok.v {
                        self.keep(&mut arr, Value::text(s)?);
                    }
                    j += 1;
                }
                K::Char => {
                    if let TV::Char(c) = tok.v {
                        self.keep(&mut arr, Value::Char(c));
                    }
                    j += 1;
                }
                K::Name | K::QName => {
                    let s = self.sym_for(self.lx.text_of(&tok));
                    self.keep(&mut arr, Value::Sym(s));
                    j += 1;
                }
                K::Op => {
                    let nt = self.tok(j + 1).clone();
                    let colon = self.lx.text.as_bytes().get(nt.start as usize) == Some(&b':');
                    if self.lx.text_of(&tok) == "-" && !nt.ws && matches!(nt.k, K::Num | K::Error | K::Char) && !colon {
                        // legacy: '-' and the digits after it are one number
                        match (&nt.k, &nt.v) {
                            (K::Num, TV::Int(i)) => self.keep(&mut arr, Value::Int(i.wrapping_neg())),
                            (K::Num, TV::Flt(f)) => self.keep(&mut arr, Value::Flt(-*f)),
                            #[cfg(feature = "bigint")]
                            (K::Num, TV::Big(b)) => self.keep(&mut arr, crate::big::negate(b)),
                            (K::Error, TV::BadNum) => return Err(self.raise_tok(&nt, true)),
                            // "-0cx" is the number -0 and the name cx in KlongPy (its number
                            // reader does not know the 0c prefix). Not reproduced.
                            _ => return Err(Error::syntax("a negative sign before a character literal").with_span((tok.start, nt.end))),
                        }
                        j += 2;
                    } else {
                        let v = Value::Prim(self.syms.intern(self.lx.text_of(&tok)));
                        self.keep(&mut arr, v);
                        j += 1;
                    }
                }
                K::LBrack => {
                    let (nj, q) = self.read_list(j + 1, K::RBrack)?;
                    // the entries of a dictionary literal stay as written: [1.5 2] is the key
                    // 1.5 and the value 2, not a list of two floats
                    self.keep(&mut arr, if close == K::RBrace { Value::raw_list(q)? } else { Value::from_items(q)? });
                    j = nj;
                }
                K::Dict => {
                    let (nj, q) = self.read_list(j + 1, K::RBrace)?;
                    let v = self.make_dict(q, (tok.start, self.tok(nj - 1).end))?;
                    self.keep(&mut arr, v);
                    self.dicts_read += 1;
                    j = nj;
                }
                K::Cop | K::CallP => {
                    let v = Value::Prim(self.syms.intern(self.lx.text_of(&tok)));
                    self.keep(&mut arr, v);
                    j += 1;
                }
                K::Error => return Err(self.raise_tok(&tok, false)),
                K::Newline => {
                    self.keep(&mut arr, Value::text(";")?);
                    j += 1;
                }
                _ => {
                    // ; ( ) { } ] :[ :| are the strings they are written as
                    let v = Value::text(self.lx.text_of(&tok))?;
                    self.keep(&mut arr, v);
                    j += 1;
                }
            }
            j = self.skip_nl(j);
        }
        self.depth -= 1;
        if self.kind(j) == close {
            j += 1;
        } else {
            self.lx.at_end = true;
            if close == K::RBrack {
                self.issue("unterminated-list", "unterminated list: the closing ] is missing".to_string(), (opener.start, opener.end), None, Some(TO_END))?;
            } else {
                self.issue("unterminated-dict", "unterminated dictionary: the closing } is missing".to_string(), (opener.start, opener.end), None, Some(TO_END))?;
            }
        }
        Ok((j, arr))
    }

    /// The dictionary a `:{...}` literal denotes: every entry must be a `[key value]` pair.
    fn make_dict(&self, entries: Vec<Value>, span: Span) -> Result<Value> {
        let mut d = Dict::default();
        for e in &entries {
            let ok = (e.is_list() || e.is_str()) && e.len().unwrap_or(0) == 2;
            if !ok {
                return Err(Error::syntax("each entry of a dictionary literal must be a [key value] pair")
                    .with_span(span)
                    .with_hint("write :{[key1 value1] [key2 value2]}, or :{;key1;value1;key2;value2} to have the keys and values evaluated; a key is a number, a string, a char or a symbol"));
            }
            let conv = |v: Value| match v {
                Value::Char(c) if e.is_str() => Value::char_str(c),
                v => v,
            };
            let k = conv(e.item(0));
            if k.is_list() || matches!(k, Value::Dict(_) | Value::Prim(_)) {
                let what = if matches!(k, Value::Prim(_)) { "an operator cannot be a dictionary key" } else { "a list cannot be a dictionary key" };
                return Err(Error::syntax(what).with_span(span).with_hint("a key is a number, a string, a char or a symbol"));
            }
            let v = conv(e.item(1));
            // a value that is itself a list was read as one and is normalised now
            let v = match v {
                Value::List(_) => Value::from_items(v.items()?)?,
                v => v,
            };
            d.set(k, v).map_err(|e| if e.kind == crate::error::Kind::Type { Error::syntax("this value cannot be a dictionary key").with_span(span) } else { e })?;
        }
        Ok(Value::dict(d))
    }

    /// Read the list or dictionary literal whose opener is token `j`. With `consume` the read
    /// position moves past it. Returns the value (for a dictionary, the dictionary).
    fn read_data(&mut self, j: usize, consume: bool) -> Result<Value> {
        let tok = self.tok(j).clone();
        let close = if tok.k == K::LBrack { K::RBrack } else { K::RBrace };
        let at_end = self.lx.at_end;
        let depth = self.depth;
        let r = self.read_list(j + 1, close);
        self.depth = depth;
        if !consume {
            self.lx.at_end = at_end;
        }
        let (e, q) = r?;
        if consume {
            self.lx.pos = e;
        }
        let end = self.tok(e - 1).end;
        if close == K::RBrace {
            self.make_dict(q, (tok.start, end))
        } else {
            Value::from_items(q)
        }
    }

    /// legacy: a lexeme that was read only to look at it still raised.
    fn check(&mut self, j: usize) -> Result<()> {
        let tok = self.tok(j).clone();
        if tok.k == K::Error {
            return Err(self.raise_tok(&tok, false));
        }
        let n = self.dicts_read;
        let r = self.read_data(j, false).map(|_| ());
        self.dicts_read = n;
        r
    }

    // ----- conditionals, evaluated lists and dictionaries ------------------------------------

    /// The read position is just after `:[` (or `:|`); `st` is the offset of that token.
    fn read_cond(&mut self, st: u32) -> Result<NodeId> {
        let n = self.expr(true)?;
        #[cfg(feature = "diag")]
        let mut csp: Vec<Span> = alloc::vec![if n == NONE { (0, 0) } else { self.last_sp }];
        let c = self.unsemi(n);
        // Whitespace and newlines may come between the condition and its ';'.
        let j = self.skip_nl(self.lx.pos);
        if self.kind(j) != K::Semi {
            return Err(self.unexpected(self.lx.offset(), ';', false));
        }
        self.lx.pos = j + 1;
        let n = self.expr(true)?;
        #[cfg(feature = "diag")]
        csp.push(if n == NONE { (0, 0) } else { self.last_sp });
        let t = self.unsemi(n);
        let j = self.skip_nl(self.lx.pos);
        let tok = self.tok(j).clone();
        let e;
        if tok.k == K::Elif {
            self.lx.pos = j + 1;
            e = self.read_cond(tok.start)?;
            #[cfg(feature = "diag")]
            csp.push(self.tree.span(e).unwrap_or((0, 0)));
        } else {
            if tok.k != K::Semi {
                return Err(self.unexpected(tok.start, ';', false));
            }
            self.lx.pos = j + 1;
            let n = self.expr(true)?;
            #[cfg(feature = "diag")]
            csp.push(if n == NONE { (0, 0) } else { self.last_sp });
            e = self.unsemi(n);
            let j = self.skip_nl(self.lx.pos);
            let tok = self.tok(j);
            if tok.k != K::RBrack {
                return Err(self.unexpected(tok.start, ']', false));
            }
            self.lx.pos = j + 1;
        }
        let end = self.lx.end();
        let id = self.add(Node::Cond(c, t, e), (st, end));
        #[cfg(feature = "diag")]
        self.rec_args(id, csp);
        Ok(id)
    }

    /// The read position is just after `[;`; `st` is the offset of the `[`.
    fn read_expr_array(&mut self, st: u32) -> Result<NodeId> {
        let mut r: Vec<u32> = Vec::new();
        #[cfg(feature = "diag")]
        let mut esp: Vec<Span> = Vec::new();
        let mut j = self.skip_nl(self.lx.pos);
        self.lx.pos = j;
        loop {
            let k = self.kind(j);
            if k == K::Eof {
                self.lx.at_end = true;
                self.issue("unterminated-list", "unterminated list: the closing ] is missing".to_string(), (st, st + 1), None, Some(TO_END))?;
                break;
            }
            if k == K::RBrack {
                break;
            }
            if k == K::Newline {
                self.lx.pos += 1;
            }
            let expr = self.expr(true)?;
            if expr != NONE {
                let expr = self.unsemi(expr);
                self.keep(&mut r, expr.0);
                #[cfg(feature = "diag")]
                esp.push(self.last_sp);
            }
            j = self.skip_nl(self.lx.pos);
            if self.kind(j) == K::Semi {
                j = self.skip_nl(j + 1);
            }
            self.lx.pos = j;
        }
        if self.kind(j) == K::RBrack {
            self.lx.pos += 1;
        }
        let range = self.tree.range(&r);
        let end = self.lx.end();
        let id = self.add(Node::EList(range), (st, end));
        #[cfg(feature = "diag")]
        self.rec_args(id, esp);
        Ok(id)
    }

    /// The read position is just after `:{;`; `st` is the offset of the `:{`.
    fn read_expr_dict(&mut self, st: u32) -> Result<NodeId> {
        let mut r: Vec<u32> = Vec::new();
        #[cfg(feature = "diag")]
        let mut dsp: Vec<Span> = Vec::new();
        let hint = "write :{;key1;value1;key2;value2}";
        let skip = |p: &mut Self| {
            let mut j = p.lx.pos;
            while p.kind(j) == K::Newline {
                j += 1;
            }
            p.lx.pos = j;
            p.tok(j).clone()
        };
        let tok = skip(self);
        if tok.k == K::RBrace {
            self.lx.pos += 1;
        } else {
            loop {
                let tok = skip(self);
                match tok.k {
                    K::Eof => {
                        self.lx.at_end = true;
                        self.issue("unterminated-dict", "unterminated dictionary: the closing } is missing".to_string(), (st, st + 2), None, Some(TO_END))?;
                        break;
                    }
                    K::Semi | K::RBrace => {
                        return Err(Error::syntax("missing expression in a dictionary literal").with_span((tok.start, tok.end)).with_hint(hint));
                    }
                    _ => {}
                }
                let expr = self.expr(true)?;
                #[cfg(feature = "diag")]
                dsp.push(self.last_sp);
                let expr = self.unsemi(expr);
                self.keep(&mut r, expr.0);
                let tok = skip(self);
                match tok.k {
                    K::Semi => self.lx.pos += 1,
                    K::RBrace => {
                        self.lx.pos += 1;
                        break;
                    }
                    K::Eof => {
                        self.lx.at_end = true;
                        self.issue("unterminated-dict", "unterminated dictionary: the closing } is missing".to_string(), (st, st + 2), None, Some(TO_END))?;
                        break;
                    }
                    _ => {
                        return Err(Error::syntax(crate::msg!("expected ';' or '}}' but found {}", self.lexeme(&tok))).with_span((tok.start, tok.end)).with_hint(hint));
                    }
                }
            }
        }
        let end = self.lx.end();
        if r.len() % 2 == 1 {
            return Err(Error::syntax("the last key of the dictionary literal has no value").with_span((st, end)).with_hint(crate::msg!("{}: keys and values alternate", hint)));
        }
        let range = self.tree.range(&r);
        let id = self.add(Node::EDict(range), (st, end));
        #[cfg(feature = "diag")]
        self.rec_args(id, dsp);
        Ok(id)
    }

    // ----- the grammar -----------------------------------------------------------------------

    /// `.comment("END")` hides the raw text up to the marker.
    fn comment_end(&self, st: u32, args: &[u32]) -> Result<usize> {
        let marker = match args.first() {
            Some(&q) if q != NONE.0 => match self.tree.node(NodeId(q)) {
                // (the literal itself is shared, not copied)
                Node::Lit(Value::Str(s)) if !s.is_empty() => s.clone(),
                _ => return Err(Error::syntax(".comment needs a string").with_span((st, self.lx.end()))),
            },
            _ => return Err(Error::syntax(".comment needs a string").with_span((st, self.lx.end()))),
        };
        let i = self.lx.offset() as usize;
        let t = self.lx.text;
        let mut j = match t[i..].find(marker.as_str()) {
            Some(j) => j,
            None => {
                self.lx.touched.set(self.lx.truncated);
                return Err(Error::syntax("end of comment not found").with_span((st, self.lx.end())));
            }
        };
        // legacy: extend while the marker repeats
        while t.is_char_boundary(i + j + 1) && t[i + j + 1..].starts_with(marker.as_str()) {
            j += 1;
        }
        if i + j + marker.len() >= t.len() {
            self.lx.touched.set(self.lx.truncated); // (the marker may go on past the window)
        }
        Ok(i + j + marker.as_str().len())
    }

    fn parse_module(&mut self, args: &[u32]) {
        let name = match args.first() {
            Some(&q) if q != NONE.0 => match self.tree.node(NodeId(q)) {
                Node::Sym(s) | Node::QSym(s) => Some(self.syms.name(*s).split('`').next().unwrap_or("").to_string()),
                Node::Lit(Value::Int(0)) => None,
                Node::Lit(Value::Int(i)) => Some(format!("{}", i)),
                Node::Lit(v) if v.is_empty() => None,
                Node::Lit(Value::Str(s)) if s.as_str().len() <= crate::sym::MAX_NAME => Some(String::from(s.as_str())),
                Node::Lit(Value::Str(_)) => {
                    self.syms.long = true;
                    None
                }
                _ => None,
            },
            _ => None,
        };
        *self.module = name;
    }

    fn factor(&mut self, ign: bool) -> Result<NodeId> {
        let mut j = self.lx.pos;
        if ign {
            j = self.skip_nl(j);
        }
        let tok = self.tok(j).clone();
        let st = tok.start;
        let mut a;
        match tok.k {
            K::Name | K::QName => {
                let v = self.lx.text_of(&tok);
                if tok.k == K::Name {
                    self.check_qualified(v, (tok.start, tok.end))?;
                }
                let s = self.sym_for(v);
                a = self.add(if tok.k == K::QName { Node::QSym(s) } else { Node::Sym(s) }, (tok.start, tok.end));
                self.lx.pos = j + 1;
                let nt = self.tok(j + 1);
                if matches!(nt.k, K::LParen | K::CallP) && !nt.ws {
                    let fa = self.read_fn_args()?;
                    if v == ".comment" {
                        let end = self.comment_end(st, &fa)?;
                        self.lx.seek(end);
                        return self.factor(ign);
                    } else if v == ".module" {
                        self.parse_module(&fa);
                    }
                    let call = !fa.contains(&NONE.0);
                    let arity = fa.len().min(255) as u8;
                    let args = self.tree.range(&fa);
                    let e = self.lx.end();
                    a = self.add(Node::Fn { a, args, has_args: true, arity, call, literal: false }, (st, e));
                    #[cfg(feature = "diag")]
                    {
                        let sp = core::mem::take(&mut self.args_sp);
                        self.rec_args(a, sp);
                    }
                    self.rec_op(a, (st, tok.end));
                }
                if let Some(aa) = self.peek_adverb() {
                    a = self.apply_adverbs(a, aa, 1, st, st, false, NONE, (0, 0))?;
                }
            }
            K::Num | K::Str | K::Char => {
                self.lx.pos = j + 1;
                let n = match &tok.v {
                    TV::Int(i) => Node::Lit(Value::Int(*i)),
                    TV::Flt(f) => Node::Lit(Value::Flt(*f)),
                    #[cfg(feature = "bigint")]
                    TV::Big(b) => Node::Lit(Value::Big(b.clone())),
                    TV::Str(s) => Node::Lit(Value::text(s)?),
                    TV::Char(c) => Node::Lit(Value::Char(*c)),
                    _ => Node::Lit(Value::Undefined),
                };
                a = self.add(n, (tok.start, tok.end));
            }
            K::Op | K::Cop | K::CallP => {
                let v = self.lx.text_of(&tok);
                let op = self.syms.intern(v);
                self.lx.pos = j + 1;
                if ops::is_monad(v) {
                    if let Some(aa) = self.peek_adverb() {
                        let verb = self.add(Node::OpVal { op, arity: 1 }, (tok.start, tok.end));
                        a = self.apply_adverbs(verb, aa, 1, st, st, false, NONE, (0, 0))?;
                    } else {
                        let mut r = self.expr(ign)?;
                        let missing = r == NONE || r == SEMI;
                        if missing {
                            let m = crate::msg!("missing operand for {}", v);
                            self.issue("missing-operand", m, (st, tok.end), None, None)?;
                            r = self.unsemi(r);
                        }
                        let e = self.lx.end();
                        #[cfg(feature = "diag")]
                        let rsp = if missing { (0, 0) } else { self.last_sp };
                        a = self.add(Node::Monad { op, arg: r, missing }, (st, e));
                        #[cfg(feature = "diag")]
                        self.rec_args(a, alloc::vec![rsp]);
                        self.rec_op(a, (st, tok.end));
                    }
                } else {
                    a = self.add(Node::OpVal { op, arity: 0 }, (tok.start, tok.end));
                }
            }
            K::LBrack => {
                let nt = self.tok(j + 1);
                if nt.k == K::Semi && !nt.ws {
                    self.lx.pos = j + 2;
                    a = self.read_expr_array(st)?;
                } else {
                    let n = self.dicts_read;
                    let v = self.read_data(j, true)?;
                    let e = self.lx.end();
                    a = if self.dicts_read != n { self.add(Node::CopyList(v), (st, e)) } else { self.add(Node::Lit(v), (st, e)) };
                    self.last_list = Some((a, j));
                }
            }
            K::LParen => {
                self.lx.pos = j + 1;
                // legacy: "(;)" gives the ';' itself, which the callers treat as a separator
                a = self.expr(ign)?;
                let tok = self.tok(self.lx.pos);
                if tok.k != K::RParen || tok.ws {
                    return Err(self.unexpected(self.lx.offset(), ')', true));
                }
                self.lx.pos += 1;
            }
            K::LBrace => {
                self.lx.pos = j + 1;
                a = self.read_fn(st)?;
                if let Some(aa) = self.peek_adverb() {
                    a = self.apply_adverbs(a, aa, 1, st, st, false, NONE, (0, 0))?;
                }
            }
            K::Semi | K::Newline => {
                self.lx.pos = j + 1;
                a = SEMI;
            }
            K::Eof => {
                self.lx.pos = j;
                self.lx.at_end = true;
                return Ok(NONE);
            }
            K::Cond => {
                self.lx.pos = j + 1;
                a = self.read_cond(st)?;
            }
            K::Dict => {
                let nt = self.tok(j + 1);
                if nt.k == K::Semi && !nt.ws {
                    self.lx.pos = j + 2;
                    a = self.read_expr_dict(st)?;
                } else {
                    let v = self.read_data(j, true)?;
                    let e = self.lx.end();
                    a = self.add(Node::DictLit(v), (st, e));
                }
            }
            K::Error => return Err(self.raise_tok(&tok, false)),
            _ => {
                // ) } ] :| where a factor was expected: read as the string it is written as
                let v = self.lx.text_of(&tok);
                let v = v.to_string();
                let m = crate::msg!("unexpected '{}'", v);
                let note = crate::msg!("it is read as the string \"{}\"", v);
                self.issue("stray-closer", m, (tok.start, tok.end), None, Some(note.as_str()))?;
                self.lx.pos = j + 1;
                a = self.lit_str(&v);
            }
        }
        self.lx.start = st;
        Ok(a)
    }

    fn expr(&mut self, ign: bool) -> Result<NodeId> {
        self.depth += 1;
        if self.depth > self.max_depth || self.out_of_stack() {
            return Err(Error::range("expression nested too deeply"));
        }
        let r = self.expr_inner(ign);
        self.depth -= 1;
        self.last_sp = (self.lx.start, self.lx.end());
        r
    }

    fn expr_inner(&mut self, ign: bool) -> Result<NodeId> {
        let mut a = self.factor(ign)?;
        if a == NONE || a == SEMI {
            return Ok(a);
        }
        let st = self.lx.start;
        loop {
            let mut j = self.lx.pos;
            let lend = self.lx.end();
            if ign {
                j = self.skip_nl(j);
            }
            let tok = self.tok(j).clone();
            let mut target = false;
            // the operator (a glyph) or the infix function (a node)
            let mut op: Option<SymId> = None;
            let mut aa = NONE;
            match tok.k {
                K::Op | K::Cop | K::CallP => {
                    let v = self.lx.text_of(&tok);
                    if ops::is_dyad(v) {
                        target = v == "::" && !matches!(self.tree.node(a), Node::Sym(_) | Node::QSym(_));
                    } else if ADVERBS.contains(&v) {
                        // an adverb glyph where a dyadic operator is expected
                        let m = if tok.ws { crate::msg!("adverb {} must directly follow its verb", v) } else { crate::msg!("adverb {} does not follow a verb", v) };
                        let e = Error::syntax(m).with_span((tok.start, tok.end));
                        #[cfg(feature = "diag")]
                        let e = e.with_hint(if tok.ws { crate::msg!("an adverb follows its verb with no space: remove the space before {}", v) } else { crate::msg!("an adverb directly follows an operator, a name or a function literal") });
                        return Err(e);
                    }
                    op = Some(self.syms.intern(v));
                    self.lx.pos = j + 1;
                }
                K::Name | K::QName => {
                    if tok.k == K::Name {
                        self.check_qualified(self.lx.text_of(&tok), (tok.start, tok.end))?;
                    }
                    let s = self.sym_for(self.lx.text_of(&tok));
                    aa = self.add(Node::Sym(s), (tok.start, tok.end));
                    self.lx.pos = j + 1;
                    let nt = self.tok(j + 1);
                    if matches!(nt.k, K::LParen | K::CallP) && !nt.ws {
                        let fa = self.read_fn_args()?;
                        let call = !fa.contains(&NONE.0);
                        let arity = fa.len().min(255) as u8;
                        let args = self.tree.range(&fa);
                        let e = self.lx.end();
                        aa = self.add(Node::Fn { a: aa, args, has_args: true, arity, call, literal: false }, (tok.start, e));
                        #[cfg(feature = "diag")]
                        {
                            let sp = core::mem::take(&mut self.args_sp);
                            self.rec_args(aa, sp);
                        }
                        self.rec_op(aa, (tok.start, tok.end));
                    }
                }
                K::LBrace => {
                    self.lx.pos = j + 1;
                    aa = self.read_fn(tok.start)?;
                }
                _ => {
                    if matches!(tok.k, K::LBrack | K::Dict | K::Error) {
                        // legacy: the lexeme was read to look at it
                        self.check(j)?;
                    }
                    break;
                }
            }
            let ost = tok.start;
            if let Some(adv) = self.peek_adverb() {
                if target && self.strict > 0 {
                    return Err(self.not_assignable(a, st, lend));
                }
                let verb = match op {
                    Some(o) => self.add(Node::OpVal { op: o, arity: 0 }, (tok.start, tok.end)),
                    None => aa,
                };
                a = self.apply_adverbs(verb, adv, 2, st, ost, true, a, (st, lend))?;
            } else {
                let osp = (ost, self.lx.end());
                let mut named = None;
                let mut rj = 0;
                if target {
                    match self.def_target(a, st, lend)? {
                        Target::Node(n) => a = n,
                        Target::Named(nd) => named = Some(nd),
                    }
                    rj = self.lx.pos;
                }
                let mut r = self.expr(ign)?;
                #[cfg(feature = "diag")]
                let rsp = self.last_sp;
                let missing = r == NONE || r == SEMI;
                if missing {
                    let what = if tok.k == K::LBrace { "the function" } else { self.lx.text_of(&tok) };
                    let m = crate::msg!("missing right operand for {}", what);
                    self.issue("missing-operand", m, osp, None, None)?;
                    r = self.unsemi(r);
                }
                if let Some(nd) = named {
                    a = self.finish_named_def(nd, r, rj, osp)?;
                }
                let e = self.lx.end();
                #[cfg(feature = "diag")]
                let rsp = if missing { (0, 0) } else { rsp };
                a = match op {
                    Some(o) => self.add(Node::Dyad { op: o, l: a, r, missing }, (st, e)),
                    None => {
                        let args = self.tree.range(&[a.0, r.0]);
                        self.add(Node::Fn { a: aa, args, has_args: true, arity: 2, call: false, literal: false }, (st, e))
                    }
                };
                #[cfg(feature = "diag")]
                self.rec_args(a, alloc::vec![(st, lend), rsp]);
                self.rec_op(a, osp);
            }
        }
        if ign && Tree::is_real(a) {
            let nl = match self.tree.node(a) {
                Node::Lit(Value::Char('\n')) => true,
                Node::Lit(Value::Str(s)) => s.as_str() == "\n",
                _ => false,
            };
            if nl {
                // legacy: an expression that is just the newline character was taken for a
                // newline lexeme, and the whitespace and newlines after it were skipped.
                let j = self.skip_nl(self.lx.pos);
                self.lx.pos = j;
                let at = self.tok(j).start as usize;
                self.lx.seek(at);
            }
        }
        self.lx.start = st;
        Ok(a)
    }

    /// A `;`-separated sequence of expressions.
    fn prog(&mut self, ign: bool) -> Result<Vec<u32>> {
        let mut arr = Vec::new();
        #[cfg(feature = "diag")]
        let mut psp: Vec<Span> = Vec::new();
        #[cfg(feature = "diag")]
        let r = self.prog_inner(ign, &mut arr, &mut psp);
        #[cfg(not(feature = "diag"))]
        let r = self.prog_inner(ign, &mut arr);
        #[cfg(feature = "diag")]
        {
            self.prog_sp = psp;
        }
        r.map(|_| arr)
    }

    fn prog_inner(&mut self, ign: bool, arr: &mut Vec<u32>, #[cfg(feature = "diag")] psp: &mut Vec<Span>) -> Result<()> {
        loop {
            let q = self.expr(ign)?;
            if q == NONE {
                break;
            }
            if q == SEMI {
                continue;
            }
            self.keep(arr, q.0);
            #[cfg(feature = "diag")]
            psp.push(self.last_sp);
            let mut j = self.lx.pos;
            if ign {
                j = self.skip_nl(j);
            }
            match self.kind(j) {
                K::Semi | K::Newline => self.lx.pos = j + 1,
                k => {
                    if matches!(k, K::LBrack | K::Dict | K::Error) {
                        self.check(j)?;
                    }
                    // legacy: whatever follows a complete statement is dropped
                    break;
                }
            }
        }
        Ok(())
    }

    /// One top-level statement: its node, and whether it was followed by something that is not a separator (nothing more is parsed then).
    fn step(&mut self) -> Result<Option<(u32, bool)>> {
        loop {
            let q = self.expr(false)?;
            if q == NONE {
                return Ok(None);
            }
            if q == SEMI {
                continue;
            }
            let j = self.lx.pos;
            return match self.kind(j) {
                K::Semi | K::Newline => {
                    self.lx.pos = j + 1;
                    Ok(Some((q.0, false)))
                }
                k => {
                    if matches!(k, K::LBrack | K::Dict | K::Error) {
                        self.check(j)?;
                    }
                    Ok(Some((q.0, true)))
                }
            };
        }
    }

    /// The top-level parse has stopped: whatever is left is input the legacy parser dropped,
    /// and a string or comment may run to the end of the input.
    fn finish(&mut self) -> Result<Vec<NodeId>> {
        let j = self.lx.pos;
        let tok = self.tok(j).clone();
        if tok.k != K::Eof {
            let span = (tok.start, tok.end);
            if matches!(tok.k, K::RParen | K::RBrack | K::RBrace) {
                let v = self.lx.text[tok.start as usize..tok.end as usize].to_string();
                let note = if self.tok(j + 1).k != K::Eof { REST } else { "it is ignored" };
                self.issue("stray-closer", crate::msg!("unexpected '{}': nothing is open that it could close", v), span, None, Some(note))?;
            } else {
                let prev = if j > 0 { Some(self.tok(j - 1).clone()) } else { None };
                let pk = prev.as_ref().map(|p| p.k);
                let mut hint: Option<String> = None;
                if matches!(tok.k, K::LParen | K::CallP) && matches!(pk, Some(K::Name | K::QName | K::RBrace)) && tok.ws {
                    let text = self.lx.text;
                    let args = self.matching(j).map(|m| &text[tok.start as usize..self.tok(m).end as usize]);
                    let args = match args {
                        Some(a) if a.chars().count() <= 24 && !a.contains('\n') => a.to_string(),
                        _ => "(...)".to_string(),
                    };
                    let callee = match &prev {
                        Some(p) if p.k != K::RBrace => text[p.start as usize..p.end as usize].to_string(),
                        _ => "{...}".to_string(),
                    };
                    hint = Some(crate::msg!("no space may separate a function from its argument list: write {}{}", callee, args));
                } else if matches!(tok.k, K::LParen | K::CallP) {
                    hint = Some("only a name or a function literal can be followed by an argument list; assign the value to a name first".to_string());
                } else if tok.k == K::LBrack && self.index_hint(j).is_some() {
                    hint = self.index_hint(j);
                } else if matches!(tok.k, K::Num | K::Str | K::Char | K::Name | K::QName) && matches!(pk, Some(K::Num | K::Str | K::Char | K::Name | K::QName)) {
                    hint = Some("a list is written in brackets, [1 2 3]; statements are separated by ;".to_string());
                }
                let m = crate::msg!("unexpected {} after the end of the expression", self.lexeme(&tok));
                self.issue("trailing-input", m, span, hint, Some(REST))?;
            }
        }
        let n = self.lx.toks.len();
        if n >= 2 {
            let t = self.lx.toks[n - 2].clone();
            if t.k == K::Str && t.unterminated {
                let hint = self.escaped_quote_hint();
                self.issue("unterminated-string", "unterminated string: the closing \" is missing".to_string(), (t.start, t.end), hint, Some(TO_END))?;
            }
        }
        let comments: Vec<(u32, u32)> = self.lx.trivia.iter().filter(|c| c.unterminated).map(|c| (c.start, c.end)).collect();
        for (a, b) in comments {
            self.issue("unterminated-comment", "unterminated comment: the closing \" is missing".to_string(), (a, b), None, Some(TO_END))?;
        }
        Ok(Vec::new())
    }
}

impl<'a> Parser<'a> {
    #[allow(dead_code)]
    fn _unused(&self) {}
}
