//! Parser: tokens to a tree. A port, decision for decision, of the token grammar in
//! `klongpy/interpreter.py` (`_prog`, `_expr`, `_factor`, `_read_fn`, `_read_fn_args`,
//! `_apply_adverbs`) and the `tk_*` readers of `klongpy/parser.py`, because the canonical dump
//! of the result must equal KlongPy's for every golden file (`frontend.md` sections 5 to 7,
//! and the pinned quirks in the notes of tickets 030 and 031).
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
}

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
}

pub fn parse_next(
    source: &Rc<Source>,
    start: usize,
    window: &mut usize,
    syms: &mut Interner,
    module: &mut Option<String>,
    strict: u8,
    max_depth: u32,
    stack: Option<(usize, usize)>,
) -> Result<Next> {
    let _tag = crate::htag::Scope::new(crate::htag::TREE);
    let full: &str = &source.text;
    if let Some(end) = skip_comment(full, start) {
        return Ok(Next { skipped: true, parsed: None, end, last: false });
    }
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
        };
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
            Ok(None) => Ok(Next { skipped: false, parsed: None, end: full.len(), last: true }),
            Ok(Some((q, last))) => {
                let end = if last { full.len() } else { p.lx.offset() as usize };
                let mut ids = crate::value::try_vec(1)?;
                ids.push(NodeId(q));
                p.tree.shrink();
                Ok(Next { skipped: false, parsed: Some(Parsed { tree: Rc::new(p.tree), stmts: ids }), end, last })
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
    let mut p = Parser { lx: Lexer::new(text, true), tree: Tree::new(None), syms, module, strict: 0, last_list: None, dicts_read: 0, depth: 0, max_depth, stack };
    if p.lx.toks.is_empty() || !p.tree.reserve() {
        return Err(Error::memory());
    }
    let j = p.skip_nl(0);
    let tok = p.tok(j).clone();
    let r = match tok.k {
        K::Eof => Ok(None),
        K::LBrack | K::Dict => p.read_data(j, true).map(|v| {
            let end = p.tok(p.lx.pos.saturating_sub(1)).end;
            Some((end as usize, v))
        }),
        K::Num | K::Str | K::Char => {
            let v = match &tok.v {
                TV::Int(i) => Value::Int(*i),
                TV::Flt(f) => Value::Flt(*f),
                TV::Str(s) => Value::text(s)?,
                TV::Char(c) => Value::Char(*c),
                _ => Value::Undefined,
            };
            Ok(Some((tok.end as usize, v)))
        }
        K::Name | K::QName => {
            let s = p.sym_for(p.lx.text_of(&tok));
            Ok(Some((tok.end as usize, Value::Sym(s))))
        }
        K::Op if p.lx.text_of(&tok) == "-" => {
            let nt = p.tok(j + 1).clone();
            match (nt.k, &nt.v, nt.ws) {
                (K::Num, TV::Int(i), false) => Ok(Some((nt.end as usize, Value::Int(i.wrapping_neg())))),
                (K::Num, TV::Flt(f), false) => Ok(Some((nt.end as usize, Value::Flt(-*f)))),
                _ => Err(Error::syntax("not a data object").with_span((tok.start, tok.end))),
            }
        }
        K::Error => Err(p.raise_tok(&tok)),
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

    /// Something the legacy grammar accepts and should not: an error at strict level 1 and
    /// above, accepted at level 0.
    fn issue(&self, message: String, span: Span) -> Result<()> {
        if self.strict > 0 {
            Err(Error::syntax(message).with_span(span))
        } else {
            Ok(())
        }
    }

    fn unexpected(&self, at: u32, want: char) -> Error {
        let found = match self.lx.text[(at as usize).min(self.lx.text.len())..].chars().next() {
            Some(c) => crate::msg!("expected '{}' but found '{}'", want, c),
            None => crate::msg!("expected '{}' but found end of input", want),
        };
        Error::syntax(found).with_span((at, at + 1))
    }

    fn raise_tok(&self, tok: &Tok) -> Error {
        let m = match tok.v {
            TV::CharEof => "character literal 0c at end of input".to_string(),
            _ => crate::msg!("malformed number {}", &self.lx.text[tok.start as usize..tok.end as usize]),
        };
        Error::syntax(m).with_span((tok.start, tok.end))
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
    fn apply_adverbs(&mut self, a: NodeId, aa: &'static str, arity: u8, st: u32, ost: u32, dyad: bool, l: NodeId) -> Result<NodeId> {
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
        if r == NONE || r == SEMI {
            let what = &self.lx.text[ost as usize..oe as usize];
            self.issue(
                if dyad { crate::msg!("missing right operand for {}", what) } else { crate::msg!("missing operand for {}", what) },
                (ost, oe),
            )?;
            r = self.unsemi(r);
        }
        let advs = self.tree.range(&advs);
        let e = self.lx.end();
        Ok(self.add(Node::Adv { verb: a, verb_arity, advs, first_arity: arity, dyad, l, r }, (st, e)))
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
            return Ok(arr);
        }
        // legacy: an argument is omitted only where a ';' or the ')' directly follows the
        // previous separator, with nothing in between.
        let mut sep = j;
        let mut fresh = true;
        loop {
            j = self.skip_nl(self.lx.pos);
            let tok = self.tok(j).clone();
            match tok.k {
                K::Semi | K::Newline => {
                    if j == sep && !tok.ws {
                        self.keep(&mut arr, NONE.0);
                    }
                    sep = j + 1;
                    self.lx.pos = sep;
                    fresh = true;
                    continue;
                }
                K::RParen => {
                    if j == sep && !tok.ws {
                        self.keep(&mut arr, NONE.0);
                    }
                    break;
                }
                _ => {}
            }
            if !fresh && tok.k != K::Eof {
                self.issue("unexpected input after the end of the argument".to_string(), (tok.start, tok.end))?;
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
            fresh = false;
        }
        // legacy: the ')' must follow the last argument directly
        let tok = self.tok(self.lx.pos);
        if tok.k != K::RParen || tok.ws {
            return Err(self.unexpected(self.lx.offset(), ')'));
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

    /// The arity of a function body (KlongPy `get_fn_arity`, ticket 026): the highest argument
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
        // A body of one statement is that statement, except an evaluated dictionary literal.
        let mut a = if body.len() == 1 && !matches!(self.tree.node(NodeId(body[0])), Node::EDict(_)) {
            NodeId(body[0])
        } else {
            let r = self.tree.range(&body);
            self.add(Node::Seq(r), (0, 0))
        };
        let j = self.skip_nl(self.lx.pos);
        let tok = self.tok(j);
        if tok.k != K::RBrace {
            return Err(self.unexpected(tok.start, '}'));
        }
        self.lx.pos = j + 1;
        let arity = self.fn_arity(a);
        // Function metadata (ticket 032): a locals declaration leaves the stored body.
        if let Node::Seq(r) = self.tree.node(a) {
            let r = *r;
            if r.len > 1 {
                if let Some(names) = self.decl_names(NodeId(self.tree.ids(r)[0])) {
                    let locals = self.tree.range(&names);
                    let stmts = Range { start: r.start + 1, len: r.len - 1 };
                    a = self.add(Node::Body { stmts, locals, params: Range::default(), has_params: false }, (0, 0));
                }
            }
        }
        // Ticket 060: the targets of a destructuring that heads the body are locals.
        a = self.head_destructuring(a);
        let nt = self.tok(j + 1);
        let node = if matches!(nt.k, K::LParen | K::CallP) && !nt.ws {
            let fa = self.read_fn_args()?;
            let call = !fa.contains(&NONE.0);
            let args = self.tree.range(&fa);
            Node::Fn { a, args, has_args: true, arity, call, literal: true }
        } else {
            Node::Fn { a, args: Range::default(), has_args: false, arity, call: false, literal: false }
        };
        let e = self.lx.end();
        Ok(self.add(node, (st, e)))
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
            if !names.contains(&t) {
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

    // ----- assignment targets (tickets 050, 060, 061) ---------------------------------------

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
    fn destructure_ok(&self, j: usize) -> Result<bool> {
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
                self.issue(crate::msg!("cannot destructure: {}", w), (self.tok(j).start, end))?;
                Ok(false)
            }
            None => Ok(true),
        }
    }

    /// `a`, the left side of `::`, is a call. `f(a;b)` with plain names defines `f` with those
    /// parameters.
    fn named_def(&self, a: NodeId, st: u32, lend: u32) -> Result<Option<NamedDef>> {
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
            self.issue(crate::msg!("cannot define {} with named parameters: {}", n, w), (st, lend))?;
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
            return Err(self.not_assignable(st, lend));
        }
        Ok(Target::Node(a))
    }

    fn not_assignable(&self, st: u32, lend: u32) -> Error {
        Error::syntax("cannot assign: the left side of :: must be a name").with_span((st, lend))
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
                if fnd == 0 && matches!(self.lx.text_of(tok), "x" | "y" | "z") {
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

    /// Finish `f(a;b)::{...}`: the right side `r` (read from token `rj` on) must be a function
    /// literal that does not use x, y or z itself; it becomes a function of the parameters.
    fn finish_named_def(&mut self, nd: NamedDef, r: NodeId, mut rj: usize, osp: Span) -> Result<NodeId> {
        let body = match (Tree::is_real(r), if Tree::is_real(r) { Some(self.tree.node(r)) } else { None }) {
            (true, Some(Node::Fn { a, has_args: false, .. })) => *a,
            _ => {
                let span = if Tree::is_real(r) { self.tree.span(r).unwrap_or(osp) } else { osp };
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
                .with_hint("use the parameter names; x, y and z exist only in a function without a parameter list"));
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
                            (K::Error, TV::BadNum) => return Err(self.raise_tok(&nt)),
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
                K::Error => return Err(self.raise_tok(&tok)),
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
            let what = if close == K::RBrack { "unterminated list: the closing ] is missing" } else { "unterminated dictionary: the closing } is missing" };
            self.issue(what.to_string(), (opener.start, opener.end))?;
        }
        Ok((j, arr))
    }

    /// The dictionary a `:{...}` literal denotes: every entry must be a `[key value]` pair.
    fn make_dict(&self, entries: Vec<Value>, span: Span) -> Result<Value> {
        let mut d = Dict::default();
        for e in &entries {
            let ok = (e.is_list() || e.is_str()) && e.len().unwrap_or(0) >= 2;
            if !ok {
                return Err(Error::syntax("each entry of a dictionary literal must be a [key value] pair")
                    .with_span(span)
                    .with_hint("write :{[key1 value1] [key2 value2]}, or :{;key1;value1;key2;value2} to have the keys and values evaluated"));
            }
            let conv = |v: Value| match v {
                Value::Char(c) if e.is_str() => Value::char_str(c),
                v => v,
            };
            let k = conv(e.item(0));
            if k.is_list() || matches!(k, Value::Dict(_)) {
                return Err(Error::syntax("a list cannot be a dictionary key").with_span(span));
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
            return Err(self.raise_tok(&tok));
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
        let c = self.unsemi(n);
        // Whitespace and newlines may come between the condition and its ';' (ticket 031).
        let j = self.skip_nl(self.lx.pos);
        if self.kind(j) != K::Semi {
            return Err(self.unexpected(self.lx.offset(), ';'));
        }
        self.lx.pos = j + 1;
        let n = self.expr(true)?;
        let t = self.unsemi(n);
        let j = self.skip_nl(self.lx.pos);
        let tok = self.tok(j).clone();
        let e;
        if tok.k == K::Elif {
            self.lx.pos = j + 1;
            e = self.read_cond(tok.start)?;
        } else {
            if tok.k != K::Semi {
                return Err(self.unexpected(tok.start, ';'));
            }
            self.lx.pos = j + 1;
            let n = self.expr(true)?;
            e = self.unsemi(n);
            let j = self.skip_nl(self.lx.pos);
            let tok = self.tok(j);
            if tok.k != K::RBrack {
                return Err(self.unexpected(tok.start, ']'));
            }
            self.lx.pos = j + 1;
        }
        let end = self.lx.end();
        Ok(self.add(Node::Cond(c, t, e), (st, end)))
    }

    /// The read position is just after `[;`; `st` is the offset of the `[`.
    fn read_expr_array(&mut self, st: u32) -> Result<NodeId> {
        let mut r: Vec<u32> = Vec::new();
        let mut j = self.skip_nl(self.lx.pos);
        self.lx.pos = j;
        loop {
            let k = self.kind(j);
            if k == K::Eof {
                self.lx.at_end = true;
                self.issue("unterminated list: the closing ] is missing".to_string(), (st, st + 1))?;
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
        Ok(self.add(Node::EList(range), (st, end)))
    }

    /// Ticket 063. The read position is just after `:{;`; `st` is the offset of the `:{`.
    fn read_expr_dict(&mut self, st: u32) -> Result<NodeId> {
        let mut r: Vec<u32> = Vec::new();
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
                        self.issue("unterminated dictionary: the closing } is missing".to_string(), (st, st + 2))?;
                        break;
                    }
                    K::Semi | K::RBrace => {
                        return Err(Error::syntax("missing expression in a dictionary literal").with_span((tok.start, tok.end)).with_hint(hint));
                    }
                    _ => {}
                }
                let expr = self.expr(true)?;
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
                        self.issue("unterminated dictionary: the closing } is missing".to_string(), (st, st + 2))?;
                        break;
                    }
                    _ => {
                        return Err(Error::syntax("expected ';' or '}' in a dictionary literal").with_span((tok.start, tok.end)).with_hint(hint));
                    }
                }
            }
        }
        let end = self.lx.end();
        if r.len() % 2 == 1 {
            return Err(Error::syntax("the last key of the dictionary literal has no value").with_span((st, end)).with_hint(hint));
        }
        let range = self.tree.range(&r);
        Ok(self.add(Node::EDict(range), (st, end)))
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
                }
                if let Some(aa) = self.peek_adverb() {
                    a = self.apply_adverbs(a, aa, 1, st, st, false, NONE)?;
                }
            }
            K::Num | K::Str | K::Char => {
                self.lx.pos = j + 1;
                let n = match &tok.v {
                    TV::Int(i) => Node::Lit(Value::Int(*i)),
                    TV::Flt(f) => Node::Lit(Value::Flt(*f)),
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
                        a = self.apply_adverbs(verb, aa, 1, st, st, false, NONE)?;
                    } else {
                        let mut r = self.expr(ign)?;
                        let missing = r == NONE || r == SEMI;
                        if missing {
                            self.issue(crate::msg!("missing operand for {}", v), (st, tok.end))?;
                            r = self.unsemi(r);
                        }
                        let e = self.lx.end();
                        a = self.add(Node::Monad { op, arg: r, missing }, (st, e));
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
                    return Err(self.unexpected(self.lx.offset(), ')'));
                }
                self.lx.pos += 1;
            }
            K::LBrace => {
                self.lx.pos = j + 1;
                a = self.read_fn(st)?;
                if let Some(aa) = self.peek_adverb() {
                    a = self.apply_adverbs(a, aa, 1, st, st, false, NONE)?;
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
            K::Error => return Err(self.raise_tok(&tok)),
            _ => {
                // ) } ] :| where a factor was expected: read as the string it is written as
                let v = self.lx.text_of(&tok);
                self.issue(crate::msg!("unexpected '{}'", v), (tok.start, tok.end))?;
                self.lx.pos = j + 1;
                a = self.lit_str(v);
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
                        // ticket 031: an adverb glyph where a dyadic operator is expected
                        let m = if tok.ws { crate::msg!("adverb {} must directly follow its verb", v) } else { crate::msg!("adverb {} does not follow a verb", v) };
                        return Err(Error::syntax(m).with_span((tok.start, tok.end)));
                    }
                    op = Some(self.syms.intern(v));
                    self.lx.pos = j + 1;
                }
                K::Name | K::QName => {
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
                    return Err(self.not_assignable(st, lend));
                }
                let verb = match op {
                    Some(o) => self.add(Node::OpVal { op: o, arity: 0 }, (tok.start, tok.end)),
                    None => aa,
                };
                a = self.apply_adverbs(verb, adv, 2, st, ost, true, a)?;
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
                let missing = r == NONE || r == SEMI;
                if missing {
                    let what = if tok.k == K::LBrace { "the function" } else { self.lx.text_of(&tok) };
                    self.issue(crate::msg!("missing right operand for {}", what), osp)?;
                    r = self.unsemi(r);
                }
                if let Some(nd) = named {
                    a = self.finish_named_def(nd, r, rj, osp)?;
                }
                let e = self.lx.end();
                a = match op {
                    Some(o) => self.add(Node::Dyad { op: o, l: a, r, missing }, (st, e)),
                    None => {
                        let args = self.tree.range(&[a.0, r.0]);
                        self.add(Node::Fn { a: aa, args, has_args: true, arity: 2, call: false, literal: false }, (st, e))
                    }
                };
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
        loop {
            let q = self.expr(ign)?;
            if q == NONE {
                break;
            }
            if q == SEMI {
                continue;
            }
            self.keep(&mut arr, q.0);
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
        Ok(arr)
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
    /// and a string or comment may run to the end of the input (ticket 031).
    fn finish(&mut self) -> Result<Vec<NodeId>> {
        let tok = self.tok(self.lx.pos).clone();
        if tok.k != K::Eof {
            let span = (tok.start, tok.end);
            let v = &self.lx.text[tok.start as usize..tok.end as usize];
            if matches!(tok.k, K::RParen | K::RBrack | K::RBrace) {
                self.issue(crate::msg!("unexpected '{}': nothing is open that it could close", v), span)?;
            } else {
                self.issue(crate::msg!("unexpected '{}' after the end of the expression", v.lines().next().unwrap_or("")), span)?;
            }
        }
        let n = self.lx.toks.len();
        if n >= 2 {
            let t = &self.lx.toks[n - 2];
            if t.k == K::Str && t.unterminated {
                self.issue("unterminated string: the closing \" is missing".to_string(), (t.start, t.end))?;
            }
        }
        for c in &self.lx.trivia {
            if c.unterminated {
                self.issue("unterminated comment: the closing \" is missing".to_string(), (c.start, c.end))?;
            }
        }
        Ok(Vec::new())
    }
}

impl<'a> Parser<'a> {
    #[allow(dead_code)]
    fn _unused(&self) {}
}
