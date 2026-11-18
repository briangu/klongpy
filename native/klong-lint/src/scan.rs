//! The bracket scan: every token is assigned to a frame (top level, function body, parenthesis, call, evaluated list,
//! conditional, data list or dictionary body). Data frames hold data, not code, which is what lets W012 and W017 tell a
//! symbol from a use.
use crate::tok::Tok;
use klong_core::lex::K;
use std::collections::HashMap;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FK {
    Top,
    Fn,
    Paren,
    Call,
    EList,
    EDict,
    Cond,
    Data,
    Dict,
}

impl FK {
    pub fn is_data(self) -> bool {
        matches!(self, FK::Data | FK::Dict)
    }
}

/// What the parse tree says about a function literal.
#[derive(Clone, Copy, Debug)]
pub struct Meta {
    pub arity: u32,
    /// Named parameters of the stored body.
    pub params: Option<usize>,
}

pub struct Frame {
    pub kind: FK,
    /// Index of the opening token (None for the top frame).
    pub open: Option<usize>,
    pub close: Option<usize>,
    pub parent: Option<usize>,
    /// The enclosing function frame (itself, for a function).
    pub fnf: Option<usize>,
    pub locals: Vec<String>,
    /// fn: byte offsets (start, end) of the locals declaration.
    pub decl: Option<(usize, usize)>,
    /// fn: named parameters of `name(a;b)::{...}`.
    pub params: Option<Vec<String>>,
    pub is_decl: bool,
    pub is_target: bool,
    pub is_header: bool,
    pub meta: Option<Meta>,
}

impl Frame {
    /// The opening token index; only for frames that have one.
    pub fn o(&self) -> usize {
        self.open.unwrap_or(0)
    }
}

pub struct Scan {
    pub frames: Vec<Frame>,
    pub ctx: Vec<usize>,
    pub open_frame: HashMap<usize, usize>,
}

fn tk(toks: &[Tok], i: usize) -> &Tok {
    &toks[i.min(toks.len() - 1)]
}

fn is_cop(toks: &[Tok], text: &str, i: usize, glyph: &str) -> bool {
    let t = tk(toks, i);
    i < toks.len() && t.k == K::Cop && &text[t.s..t.e] == glyph
}

pub fn scan(toks: &[Tok], text: &str) -> Scan {
    let n = toks.len();
    let new = |kind: FK, open: Option<usize>, parent: Option<usize>, fnf: Option<usize>| Frame {
        kind,
        open,
        close: None,
        parent,
        fnf,
        locals: Vec::new(),
        decl: None,
        params: None,
        is_decl: false,
        is_target: false,
        is_header: false,
        meta: None,
    };
    let mut frames = vec![new(FK::Top, None, None, None)];
    let mut ctx = vec![0usize; n];
    let mut open_frame: HashMap<usize, usize> = HashMap::new();
    let mut stack: Vec<usize> = vec![0];
    let mut decl_at: HashMap<usize, usize> = HashMap::new();
    let mut params_at: HashMap<usize, Vec<String>> = HashMap::new();

    for i in 0..n {
        let t = &toks[i];
        let k = t.k;
        let cur = *stack.last().unwrap();
        ctx[i] = cur;
        let ck = frames[cur].kind;

        macro_rules! push {
            ($kind:expr) => {{
                let kind = $kind;
                let fnf = if kind == FK::Fn { None } else { frames[cur].fnf };
                let id = frames.len();
                let mut f = new(kind, Some(i), Some(cur), fnf);
                if kind == FK::Fn {
                    f.fnf = Some(id);
                }
                frames.push(f);
                stack.push(id);
                open_frame.insert(i, id);
                id
            }};
        }
        macro_rules! close {
            ($kinds:expr) => {{
                let kinds: &[FK] = $kinds;
                let mut found = None;
                for p in (1..stack.len()).rev() {
                    if kinds.contains(&frames[stack[p]].kind) {
                        let f = stack[p];
                        stack.truncate(p);
                        frames[f].close = Some(i);
                        ctx[i] = f;
                        found = Some(f);
                        break;
                    }
                }
                found
            }};
        }

        if ck.is_data() {
            match k {
                K::LBrack => {
                    let f = push!(FK::Data);
                    if decl_at.contains_key(&i) {
                        frames[f].is_decl = true;
                    }
                }
                K::Dict => {
                    push!(FK::Dict);
                }
                K::RBrack if ck == FK::Data => {
                    let f = close!(&[FK::Data]);
                    if let Some(f) = f {
                        if i + 1 < n && is_cop(toks, text, i + 1, "::") {
                            frames[f].is_target = true;
                        }
                    }
                }
                K::RBrace if ck == FK::Dict => {
                    close!(&[FK::Dict]);
                }
                _ => {}
            }
            continue;
        }
        match k {
            K::LBrace => {
                let f = push!(FK::Fn);
                if let Some(p) = params_at.get(&i) {
                    frames[f].params = Some(p.clone());
                }
                find_decl(toks, text, i, f, &mut frames, &mut decl_at);
            }
            K::RBrace => {
                close!(&[FK::Fn, FK::EDict]);
            }
            K::LParen | K::CallP => {
                let prev = if i > 0 { Some(toks[i - 1].k) } else { None };
                let call = k == K::CallP || (!t.ws && matches!(prev, Some(K::Name | K::QName | K::RBrace)));
                push!(if call { FK::Call } else { FK::Paren });
            }
            K::RParen => {
                let f = close!(&[FK::Call, FK::Paren]);
                if let Some(f) = f {
                    let fo = frames[f].o();
                    if frames[f].kind == FK::Call
                        && i + 2 < n
                        && is_cop(toks, text, i + 1, "::")
                        && toks[i + 2].k == K::LBrace
                        && fo > 0
                        && toks[fo - 1].k == K::Name
                    {
                        let mut names = Vec::new();
                        let mut ok = true;
                        for j in fo + 1..i {
                            match toks[j].k {
                                K::Name => names.push(text[toks[j].s..toks[j].e].to_string()),
                                K::Semi | K::Newline => {}
                                _ => {
                                    ok = false;
                                    break;
                                }
                            }
                        }
                        if ok {
                            frames[f].is_header = true;
                            params_at.insert(i + 2, names);
                        }
                    }
                }
            }
            K::Cond => {
                push!(FK::Cond);
            }
            K::LBrack => {
                if i + 1 < n && toks[i + 1].k == K::Semi && !toks[i + 1].ws {
                    push!(FK::EList);
                } else {
                    let f = push!(FK::Data);
                    if decl_at.contains_key(&i) {
                        frames[f].is_decl = true;
                    }
                }
            }
            K::RBrack => {
                close!(&[FK::Cond, FK::EList]);
            }
            K::Dict => {
                // :{;k;v;...} is the evaluated dictionary, as [;a;b] is the evaluated list: its body is code
                if i + 1 < n && toks[i + 1].k == K::Semi && !toks[i + 1].ws {
                    push!(FK::EDict);
                } else {
                    push!(FK::Dict);
                }
            }
            _ => {}
        }
    }
    Scan { frames, ctx, open_frame }
}

/// If the function opened at token `i` starts with a locals declaration `[a;b];`, set its locals and declaration.
fn find_decl(toks: &[Tok], text: &str, i: usize, f: usize, frames: &mut [Frame], decl_at: &mut HashMap<usize, usize>) {
    let n = toks.len();
    let mut j = i + 1;
    while tk(toks, j).k == K::Newline {
        j += 1;
    }
    if tk(toks, j).k != K::LBrack {
        return;
    }
    let mut names: Vec<String> = Vec::new();
    let mut q = j + 1;
    while q < n && matches!(toks[q].k, K::Name | K::QName | K::Semi | K::Newline) {
        if !matches!(toks[q].k, K::Semi | K::Newline) {
            names.push(toks[q].val(text).to_string());
        }
        q += 1;
    }
    if q >= n || toks[q].k != K::RBrack || names.is_empty() {
        return;
    }
    let mut r = q + 1;
    if is_cop(toks, text, r, "::") {
        // head destructuring, {[a b]::x; ...}: the names are locals (when a statement follows)
        frames[f].locals = names;
        frames[f].decl = Some((toks[j].s, toks[q].e));
        decl_at.insert(j, f);
        return;
    }
    while tk(toks, r).k == K::Newline {
        r += 1;
    }
    if tk(toks, r).k != K::Semi {
        return;
    }
    r += 1;
    while tk(toks, r).k == K::Newline {
        r += 1;
    }
    if matches!(tk(toks, r).k, K::RBrace | K::Eof) {
        return;
    }
    frames[f].locals = names.clone();
    frames[f].decl = Some((toks[j].s, toks[q].e));
    decl_at.insert(j, f);
    // {[t];[a b]::x; ...}: a head destructuring after the locals list declares its names too
    if tk(toks, r).k == K::LBrack {
        let mut more: Vec<String> = Vec::new();
        let mut q2 = r + 1;
        while q2 < n && matches!(toks[q2].k, K::Name | K::Semi | K::Newline) {
            if toks[q2].k == K::Name {
                more.push(toks[q2].val(text).to_string());
            }
            q2 += 1;
        }
        if q2 + 1 < n && toks[q2].k == K::RBrack && is_cop(toks, text, q2 + 1, "::") && !more.is_empty() {
            let mut all = names;
            all.extend(more);
            frames[f].locals = all;
            frames[f].decl = Some((toks[j].s, toks[q2].e));
        }
    }
}
