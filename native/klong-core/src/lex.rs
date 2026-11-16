//! Lexer: source text to tokens with spans (`tickets/design/frontend.md` section 4, as
//! implemented by `klongpy/lexer.py`).
//!
//! Context free; it never fails. A malformed number or `0c` at end of input is an `Error`
//! token, a string or comment that runs to end of input has `unterminated` set. Everything that
//! depends on context is the parser's: `-` is always an operator, a newline is always a
//! `Newline` token, adjacency is the `ws` flag.
use alloc::string::String;
use alloc::vec::Vec;

use crate::value::{Flt, Int};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum K {
    Num,
    Char,
    Str,
    Name,
    QName,
    Op,
    Cop,
    LBrack,
    RBrack,
    LBrace,
    RBrace,
    LParen,
    RParen,
    Semi,
    Cond,
    Elif,
    Dict,
    CallP,
    Newline,
    Error,
    Eof,
}

#[derive(Clone, Debug, PartialEq)]
pub enum TV {
    None,
    Int(Int),
    Flt(Flt),
    Char(char),
    Str(String),
    /// Error token: a malformed number.
    BadNum,
    /// Error token: `0c` at end of input.
    CharEof,
}

#[derive(Clone, Debug)]
pub struct Tok {
    pub k: K,
    pub start: u32,
    pub end: u32,
    /// Whitespace or a comment between the previous token and this one.
    pub ws: bool,
    /// Newline only: a `:"..."` comment sits between the previous token and this newline.
    pub nlc: bool,
    /// Str: end of input came before the closing quote.
    pub unterminated: bool,
    pub v: TV,
}

/// A `:"..."` comment (trivia; never in the token stream).
#[derive(Clone, Copy, Debug)]
pub struct Comment {
    pub start: u32,
    pub end: u32,
    pub unterminated: bool,
}

fn is_space(c: char) -> bool {
    c != '\n' && c.is_whitespace()
}

fn is_symbolic(c: char) -> bool {
    c.is_alphabetic() || c.is_numeric() || c == '.'
}

fn name_start(c: char) -> bool {
    c.is_alphabetic() || c == '.'
}

struct Cur<'a> {
    text: &'a str,
    pos: usize,
}

impl<'a> Cur<'a> {
    fn peek(&self) -> Option<char> {
        self.text[self.pos..].chars().next()
    }
    fn peek_at(&self, at: usize) -> Option<char> {
        self.text.get(at..).and_then(|s| s.chars().next())
    }
    fn bump(&mut self) -> Option<char> {
        let c = self.peek()?;
        self.pos += c.len_utf8();
        Some(c)
    }
}

/// End of a `"..."` body starting just after the opening quote: (end offset, terminated).
fn quoted_end(text: &str, mut i: usize) -> (usize, bool) {
    let b = text.as_bytes();
    loop {
        match b[i..].iter().position(|&c| c == b'"') {
            None => return (b.len(), false),
            Some(p) => {
                i += p + 1;
                if b.get(i) == Some(&b'"') {
                    i += 1;
                } else {
                    return (i, true);
                }
            }
        }
    }
}

/// The legacy number loop: digits and '.' anywhere; `e`, and after `e` an optional sign and
/// then one more character, whatever it is.
fn num_end(text: &str, start: usize) -> usize {
    let mut c = Cur { text, pos: start };
    c.bump();
    while let Some(ch) = c.peek() {
        if ch == '.' || ch.is_ascii_digit() {
            c.bump();
        } else if ch == 'e' {
            c.bump();
            if matches!(c.peek(), Some('+') | Some('-')) {
                c.bump();
                c.bump();
            }
        } else {
            break;
        }
    }
    c.pos
}

fn num_value(d: &str) -> TV {
    if d.contains('.') || d.contains('e') {
        // What Python's float() accepts of the text the loop above can produce.
        let ok = d.bytes().all(|b| b.is_ascii_digit() || matches!(b, b'.' | b'e' | b'+' | b'-'));
        match d.parse::<Flt>() {
            Ok(f) if ok => TV::Flt(f),
            _ => TV::BadNum,
        }
    } else {
        // An integer literal that does not fit the build width is a syntax error
        // (native-runtime.md section 3).
        match d.parse::<Int>() {
            Ok(i) => TV::Int(i),
            Err(_) => TV::BadNum,
        }
    }
}

/// Append the tokens of `text[pos..]` to `out` (ending with `Eof`) and its comments to
/// `trivia`. With `one` set, stop after the first token. With `nlc` false no newline is
/// flagged `nlc` (strict level 1 and above).
///
/// Returns false when there was no memory for the tokens: `out` then ends early (with `Eof`
/// if it could hold one token at all) and the caller reports a `memory` error.
pub fn lex(text: &str, pos: usize, out: &mut Vec<Tok>, trivia: &mut Vec<Comment>, one: bool, nlc: bool) -> bool {
    fn put(out: &mut Vec<Tok>, tok: Tok) -> bool {
        if out.try_reserve(1).is_err() {
            if let Some(last) = out.last_mut() {
                *last = Tok { k: K::Eof, start: tok.start, end: tok.start, ws: false, nlc: false, unterminated: false, v: TV::None };
            }
            return false;
        }
        out.push(tok);
        true
    }
    let n = text.len();
    let mut cur = Cur { text, pos };
    let mut gap = pos;
    let mut com = false;
    loop {
        while let Some(c) = cur.peek() {
            if is_space(c) {
                cur.bump();
            } else {
                break;
            }
        }
        let s = cur.pos;
        let ws = s > gap;
        let mut tok = Tok { k: K::Eof, start: s as u32, end: s as u32, ws, nlc: false, unterminated: false, v: TV::None };
        let c = match cur.peek() {
            None => {
                tok.start = n as u32;
                tok.end = n as u32;
                tok.ws = n > gap;
                return put(out, tok);
            }
            Some(c) => c,
        };
        let c2 = cur.peek_at(s + c.len_utf8());
        if name_start(c) {
            cur.bump();
            while matches!(cur.peek(), Some(ch) if is_symbolic(ch)) {
                cur.bump();
            }
            tok.k = K::Name;
        } else if c == '\n' {
            cur.bump();
            tok.k = K::Newline;
            tok.nlc = com && nlc;
        } else if let Some(k) = punct(c) {
            cur.bump();
            tok.k = k;
        } else if c == '0' && c2 == Some('c') {
            lex_char(&mut cur, &mut tok, s);
        } else if c.is_ascii_digit() {
            cur.pos = num_end(text, s);
            tok.v = num_value(&text[s..cur.pos]);
            tok.k = if tok.v == TV::BadNum { K::Error } else { K::Num };
        } else if c == '"' {
            let (e, term) = quoted_end(text, s + 1);
            let body = if term { &text[s + 1..e - 1] } else { &text[s + 1..e] };
            tok.k = K::Str;
            tok.unterminated = !term;
            tok.v = TV::Str(body.replace("\"\"", "\""));
            cur.pos = e;
        } else if c == ':' {
            match c2 {
                Some('"') => {
                    let (e, term) = quoted_end(text, s + 2);
                    if trivia.try_reserve(1).is_err() {
                        return false;
                    }
                    trivia.push(Comment { start: s as u32, end: e as u32, unterminated: !term });
                    com = true;
                    cur.pos = e;
                    continue;
                }
                Some('0') if cur.peek_at(s + 2) == Some('c') => {
                    cur.pos = s + 1;
                    lex_char(&mut cur, &mut tok, s + 1);
                }
                Some(d) if d.is_ascii_digit() => {
                    cur.pos = num_end(text, s + 1);
                    tok.v = num_value(&text[s + 1..cur.pos]);
                    tok.k = if tok.v == TV::BadNum { K::Error } else { K::Num };
                }
                Some(d) if name_start(d) => {
                    cur.pos = s + 1;
                    while matches!(cur.peek(), Some(ch) if is_symbolic(ch)) {
                        cur.bump();
                    }
                    tok.k = K::QName;
                }
                Some('[') => {
                    cur.pos = s + 2;
                    tok.k = K::Cond;
                }
                Some('{') => {
                    cur.pos = s + 2;
                    tok.k = K::Dict;
                }
                Some('|') => {
                    cur.pos = s + 2;
                    tok.k = K::Elif;
                }
                Some('(') => {
                    cur.pos = s + 2;
                    tok.k = K::CallP;
                }
                Some(d) => {
                    // ':' takes any next character, a space or a newline included
                    cur.pos = s + 1 + d.len_utf8();
                    tok.k = K::Cop;
                }
                None => {
                    cur.pos = s + 1;
                    tok.k = K::Op;
                }
            }
        } else if c == '\\' && matches!(c2, Some('~') | Some('*')) {
            cur.pos = s + 2;
            tok.k = K::Op;
        } else {
            cur.bump();
            tok.k = K::Op;
        }
        tok.end = cur.pos as u32;
        if !put(out, tok) {
            return false;
        }
        if one {
            return true;
        }
        gap = cur.pos;
        com = false;
    }
}

fn punct(c: char) -> Option<K> {
    Some(match c {
        ';' => K::Semi,
        '(' => K::LParen,
        ')' => K::RParen,
        '{' => K::LBrace,
        '}' => K::RBrace,
        '[' => K::LBrack,
        ']' => K::RBrack,
        _ => return None,
    })
}

/// `0c` plus one character; `at` is the offset of the `0`.
fn lex_char(cur: &mut Cur, tok: &mut Tok, at: usize) {
    cur.pos = at + 2;
    match cur.bump() {
        Some(ch) => {
            tok.k = K::Char;
            tok.v = TV::Char(ch);
        }
        None => {
            tok.k = K::Error;
            tok.v = TV::CharEof;
        }
    }
}

/// Every token of a text, ending with `Eof`.
pub fn tokenize(text: &str) -> Vec<Tok> {
    let mut out = Vec::new();
    let mut trivia = Vec::new();
    lex(text, 0, &mut out, &mut trivia, false, true);
    out
}

/// The token stream the parser reads: every token from the current position to `Eof` (always
/// the last element), and `seek`, which exists because the parser, not the lexer, knows that
/// `.comment("END")` hides raw text.
pub struct Lexer<'a> {
    pub text: &'a str,
    pub toks: Vec<Tok>,
    pub pos: usize,
    pub trivia: Vec<Comment>,
    /// Start offset of the expression just parsed.
    pub start: u32,
    /// A read ran into end of input.
    pub at_end: bool,
    base: (usize, u32),
    nlc: bool,
    /// There was no memory for the tokens.
    pub oom: bool,
}

impl<'a> Lexer<'a> {
    pub fn new(text: &'a str, nlc: bool) -> Lexer<'a> {
        let mut lx = Lexer { text, toks: Vec::new(), pos: 0, trivia: Vec::new(), start: 0, at_end: false, base: (0, 0), nlc, oom: false };
        lx.oom = !lex(text, 0, &mut lx.toks, &mut lx.trivia, false, nlc);
        lx
    }
    pub fn tok(&self, j: usize) -> &Tok {
        let last = self.toks.len().saturating_sub(1);
        &self.toks[j.min(last)]
    }
    pub fn text_of(&self, t: &Tok) -> &'a str {
        let s = if t.k == K::QName { t.start + 1 } else { t.start };
        &self.text[s as usize..t.end as usize]
    }
    /// The offset of the read position: the end of the last token consumed, or the length of
    /// the text once a read has run into end of input.
    pub fn offset(&self) -> u32 {
        if self.at_end && self.pos == self.toks.len() - 1 {
            return self.text.len() as u32;
        }
        if self.pos == self.base.0 {
            return self.base.1;
        }
        self.toks[self.pos - 1].end
    }
    /// End offset of the last token consumed.
    pub fn end(&self) -> u32 {
        if self.pos > 0 {
            self.toks[self.pos - 1].end
        } else {
            self.base.1
        }
    }
    /// Drop the lookahead and continue lexing from `offset`.
    pub fn seek(&mut self, offset: usize) {
        let offset = offset.min(self.text.len());
        let lo = self.offset();
        self.toks.truncate(self.pos);
        self.trivia.retain(|c| c.start < lo);
        if !lex(self.text, offset, &mut self.toks, &mut self.trivia, false, self.nlc) {
            self.oom = true;
        }
        self.base = (self.pos, offset as u32);
        self.at_end = false;
    }
}
