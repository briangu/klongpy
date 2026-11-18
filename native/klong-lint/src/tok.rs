//! Tokens in the shape the rules read them: the lexer of `klong-core`, with the `.comment("X")` bodies skipped
//! as the parser does, and positions that can be turned into line and column in characters.
use klong_core::lex::{Lexer, K, TV};
use std::rc::Rc;

/// A program text, with its name and where its lines start.
pub struct Source {
    pub name: String,
    pub text: String,
    starts: Vec<usize>,
}

impl Source {
    pub fn new(name: &str, text: String) -> Source {
        let mut starts = vec![0];
        for (i, b) in text.bytes().enumerate() {
            if b == b'\n' {
                starts.push(i + 1);
            }
        }
        Source { name: name.to_string(), text, starts }
    }

    /// (line, column) of a byte offset, 1-based, the column counted in characters.
    pub fn line_col(&self, off: usize) -> (usize, usize) {
        let mut off = off.min(self.text.len());
        while !self.text.is_char_boundary(off) {
            off -= 1;
        }
        let k = match self.starts.binary_search(&off) {
            Ok(k) => k,
            Err(k) => k - 1,
        };
        (k + 1, self.text[self.starts[k]..off].chars().count() + 1)
    }
}

#[derive(Clone, Debug)]
pub struct Tok {
    pub k: K,
    pub s: usize,
    pub e: usize,
    pub ws: bool,
    pub nlc: bool,
    pub unt: bool,
    /// Str: the decoded body.
    pub sv: Option<Rc<String>>,
}

impl Tok {
    /// The token's value as the Python lexer gives it: the text, a name or symbol without its colon, a string's body.
    pub fn val<'a>(&'a self, text: &'a str) -> &'a str {
        match self.k {
            K::Str => self.sv.as_ref().map(|s| s.as_str()).unwrap_or(""),
            K::QName => &text[self.s + 1..self.e],
            _ => &text[self.s..self.e],
        }
    }
}

/// End offset of a `.comment` body starting at `i` (the legacy `read_sys_comment` rule).
fn comment_end(text: &str, i: usize, marker: &str) -> Option<usize> {
    if marker.is_empty() {
        return None;
    }
    let mut j = i + text[i..].find(marker)?;
    while text.is_char_boundary(j + 1) && text[j + 1..].starts_with(marker) {
        j += 1;
    }
    Some(j + marker.len())
}

/// The tokens (ending with `Eof`) and the comments (start, end) of a text, skipping `.comment("X")` bodies.
pub fn tokenize(text: &str) -> (Vec<Tok>, Vec<(usize, usize)>) {
    let mut lx = Lexer::new(text, true);
    let mut out: Vec<Tok> = Vec::new();
    loop {
        let t = lx.toks[lx.pos.min(lx.toks.len() - 1)].clone();
        if t.k != K::Eof {
            lx.pos += 1;
        }
        let sv = match &t.v {
            TV::Str(s) => Some(s.clone()),
            _ => None,
        };
        let tk = Tok { k: t.k, s: t.start as usize, e: t.end as usize, ws: t.ws, nlc: t.nlc, unt: t.unterminated, sv };
        out.push(tk);
        if t.k == K::Eof {
            break;
        }
        if t.k == K::RParen && !t.ws && out.len() >= 4 {
            let n = out.len();
            let (a, b, c) = (&out[n - 4], &out[n - 3], &out[n - 2]);
            if matches!(a.k, K::Name | K::QName)
                && &text[a.s + if a.k == K::QName { 1 } else { 0 }..a.e] == ".comment"
                && matches!(b.k, K::LParen | K::CallP)
                && !b.ws
                && c.k == K::Str
                && !c.unt
            {
                let marker = c.sv.as_ref().map(|s| s.as_str()).unwrap_or("");
                if let Some(end) = comment_end(text, t.end as usize, marker) {
                    lx.seek(end);
                }
            }
        }
    }
    let mut trivia: Vec<(usize, usize)> = lx.trivia.iter().map(|c| (c.start as usize, c.end as usize)).collect();
    trivia.sort();
    (out, trivia)
}
