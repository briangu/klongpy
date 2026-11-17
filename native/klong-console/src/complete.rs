//! Is a console entry a whole entry? A port of `check_complete` and `check_text` of
//! `klongpy/repl_input.py`, on the tokens of `klong-core`'s lexer. The cases of
//! `tests/test_repl_complete.py` are run against it (klong-console/tests and the Python test
//! of native/tests).
//!
//! * `Incomplete`: an opener `( [ { :[ :{ :(` is still open at the end, the last token is a
//!   string without its closing quote, the text ends in a `:"` comment without one, or a
//!   `.comment("X")` has no end marker `X` after it.
//! * `Error`: a closer does not match the innermost opener, or closes nothing.
//! * `Complete`: anything else; a trailing dyad (`1+`) is complete.
//!
//! The tokens are read one at a time, so the memory used is the bracket stack (one byte per
//! open bracket) and one token; when even that cannot be had the answer is `Complete` and the
//! evaluator reports the memory error.
use alloc::rc::Rc;
use alloc::string::String;
use alloc::vec::Vec;

use klong_core::lex::{lex, Comment, Tok, K, TV};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Verdict {
    Complete,
    Incomplete,
    Error,
}

impl Verdict {
    pub fn name(self) -> &'static str {
        match self {
            Verdict::Complete => "complete",
            Verdict::Incomplete => "incomplete",
            Verdict::Error => "error",
        }
    }
}

/// What the `.comment("X")` pattern needs to remember of a token.
struct Seen {
    k: K,
    ws: bool,
    unterminated: bool,
    comment_name: bool,
    text: Option<Rc<String>>,
}

/// End offset of a `.comment` body that starts at `i`, or `None` when the marker is not there
/// (the legacy rule: the marker may overlap itself and the last overlap wins).
fn comment_end(text: &str, i: usize, marker: &str) -> Option<usize> {
    if marker.is_empty() {
        return None;
    }
    let mut j = i + text.get(i..)?.find(marker)?;
    loop {
        let step = text[j..].chars().next().map(|c| c.len_utf8()).unwrap_or(1);
        if text[j + step..].starts_with(marker) {
            j += step;
        } else {
            return Some(j + marker.len());
        }
    }
}

fn one(text: &str, pos: usize, toks: &mut Vec<Tok>, trivia: &mut Vec<Comment>) -> Option<Tok> {
    toks.clear();
    trivia.clear();
    if !lex(text, pos, toks, trivia, true, true) {
        return None;
    }
    toks.pop()
}

pub fn check(text: &str) -> Verdict {
    let mut toks: Vec<Tok> = Vec::new();
    let mut trivia: Vec<Comment> = Vec::new();
    let mut stack: Vec<u8> = Vec::new();
    let mut pos = 0usize;
    let mut win: [Option<Seen>; 3] = [None, None, None];
    // end offset of the last token, and whether it is a string without its closing quote
    let mut last: (usize, bool) = (0, false);
    let mut open_marker = false;
    loop {
        let t = match one(text, pos, &mut toks, &mut trivia) {
            Some(t) => t,
            None => return Verdict::Complete,
        };
        if t.k == K::Eof {
            break;
        }
        pos = t.end as usize;
        last = (pos, t.k == K::Str && t.unterminated);
        let opener = match t.k {
            K::LParen | K::CallP => Some(b')'),
            K::LBrack | K::Cond => Some(b']'),
            K::LBrace | K::Dict => Some(b'}'),
            _ => None,
        };
        if let Some(c) = opener {
            if stack.try_reserve(1).is_err() {
                return Verdict::Complete;
            }
            stack.push(c);
        }
        let closer = match t.k {
            K::RParen => Some(b')'),
            K::RBrack => Some(b']'),
            K::RBrace => Some(b'}'),
            _ => None,
        };
        if let Some(c) = closer {
            if stack.pop() != Some(c) {
                return Verdict::Error;
            }
        }
        // .comment("X") hides the text up to and including X
        if t.k == K::RParen && !t.ws {
            if let [Some(a), Some(b), Some(c)] = &win {
                if a.comment_name && matches!(b.k, K::LParen | K::CallP) && !b.ws && c.k == K::Str && !c.unterminated {
                    if let Some(marker) = &c.text {
                        match comment_end(text, pos, marker) {
                            Some(end) => pos = end,
                            None => open_marker |= !marker.is_empty(),
                        }
                    }
                }
            }
        }
        let comment_name = match t.k {
            K::Name => text.get(t.start as usize..t.end as usize) == Some(".comment"),
            K::QName => text.get(t.start as usize + 1..t.end as usize) == Some(".comment"),
            _ => false,
        };
        let seen = Seen {
            k: t.k,
            ws: t.ws,
            unterminated: t.unterminated,
            comment_name,
            text: match t.v {
                TV::Str(s) => Some(s),
                _ => None,
            },
        };
        win.rotate_left(1);
        win[2] = Some(seen);
    }
    if !stack.is_empty() || last.1 {
        return Verdict::Incomplete;
    }
    // A :" comment without its closing quote between the last token and the end. KlongPy looks
    // at the raw text after the last token, hidden .comment text included, and so does this.
    let mut at = last.0;
    loop {
        let t = match one(text, at, &mut toks, &mut trivia) {
            Some(t) => t,
            None => return Verdict::Complete,
        };
        if trivia.iter().any(|c| c.unterminated) {
            return Verdict::Incomplete;
        }
        if t.k == K::Eof {
            break;
        }
        at = t.end as usize;
    }
    if open_marker {
        return Verdict::Incomplete;
    }
    Verdict::Complete
}
