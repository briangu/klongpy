//! The REPL's reading of Klong text (`klongpy/repl_input.py`): whether an entry is complete, and tab completion. Neither runs anything.
use klong_core::lex::{Lexer, K};
use klong_lint::tok::{tokenize, Tok};
use std::collections::BTreeSet;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Completeness {
    /// Evaluate it.
    Complete,
    /// An opener is still open, the last string is not closed, or a `:"` comment or a `.comment("X")` is not closed: read another line.
    Incomplete,
    /// A closer that does not match the innermost opener, or closes nothing.
    Error,
}

impl Completeness {
    pub fn name(self) -> &'static str {
        match self {
            Completeness::Complete => "complete",
            Completeness::Incomplete => "incomplete",
            Completeness::Error => "error",
        }
    }
}

fn opener(k: K) -> Option<char> {
    match k {
        K::LParen | K::CallP => Some(')'),
        K::LBrack | K::Cond => Some(']'),
        K::LBrace | K::Dict => Some('}'),
        _ => None,
    }
}

fn closer(k: K) -> Option<char> {
    match k {
        K::RParen => Some(')'),
        K::RBrack => Some(']'),
        K::RBrace => Some('}'),
        _ => None,
    }
}

/// True when the text after the last token holds a `:"` comment with no closing quote.
fn unterminated_comment(tokens: &[Tok], text: &str) -> bool {
    let start = tokens.iter().rev().find(|t| t.k != K::Eof).map(|t| t.e).unwrap_or(0);
    let lx = Lexer::new(&text[start..], true);
    lx.trivia.iter().any(|c| c.unterminated)
}

/// End of a `.comment` body that starts at `i` (the rule of `read_sys_comment`), or None.
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

/// True when a `.comment("X")` call has no end marker X after it.
fn open_comment_marker(tokens: &[Tok], text: &str) -> bool {
    if tokens.len() < 4 {
        return false;
    }
    for w in tokens.windows(4) {
        let (a, b, c, d) = (&w[0], &w[1], &w[2], &w[3]);
        let aval = if a.k == K::QName { &text[a.s + 1..a.e] } else { &text[a.s..a.e] };
        if matches!(a.k, K::Name | K::QName)
            && aval == ".comment"
            && matches!(b.k, K::LParen | K::CallP)
            && !b.ws
            && c.k == K::Str
            && !c.unt
            && d.k == K::RParen
            && !d.ws
        {
            let marker = c.sv.as_ref().map(|s| s.as_str()).unwrap_or("");
            if !marker.is_empty() && comment_end(text, d.e, marker).is_none() {
                return true;
            }
        }
    }
    false
}

/// Decide whether a REPL entry is a whole entry. `comments` also looks at the text for what the tokens do not carry (a `:"` comment
/// or a `.comment("X")` left open); without it those read as complete.
pub fn check_complete(text: &str, comments: bool) -> Completeness {
    let (tokens, _) = tokenize(text);
    let mut stack: Vec<char> = Vec::new();
    for t in &tokens {
        if let Some(c) = opener(t.k) {
            stack.push(c);
        } else if let Some(c) = closer(t.k) {
            if stack.pop() != Some(c) {
                return Completeness::Error;
            }
        }
    }
    if !stack.is_empty() {
        return Completeness::Incomplete;
    }
    let last = tokens.iter().rev().find(|t| t.k != K::Eof);
    if let Some(t) = last {
        if t.k == K::Str && t.unt {
            return Completeness::Incomplete;
        }
    }
    if comments && (unterminated_comment(&tokens, text) || open_comment_marker(&tokens, text)) {
        return Completeness::Incomplete;
    }
    Completeness::Complete
}

// ------------------------------------------------------------------------------------------------ completion

/// What Python's `str.isspace` accepts, for the characters that `char::is_whitespace` leaves out.
fn is_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

fn name_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '.'
}

fn is_dir(p: &str) -> bool {
    std::fs::metadata(if p.is_empty() { "." } else { p }).map(|m| m.is_dir()).unwrap_or(false)
}

/// The file names that complete `prefix`: the entries of the directory it names (hidden ones only when the last part starts with a dot),
/// directories with a `/` added, plus the `.kg` files on KLONGPATH when the prefix has no `/`. Sorted.
pub fn kg_files(prefix: &str, klongpath: &str) -> Vec<String> {
    let mut out: BTreeSet<String> = BTreeSet::new();
    let (head, base) = match prefix.rfind('/') {
        Some(i) => (&prefix[..i + 1], &prefix[i + 1..]),
        None => ("", prefix),
    };
    // os.path.split drops the slashes that end the head, unless it is all slashes
    let dir_part = if head.chars().all(|c| c == '/') { head.to_string() } else { head.trim_end_matches('/').to_string() };
    let dir = if dir_part.is_empty() { ".".to_string() } else { dir_part.clone() };
    if let Ok(rd) = std::fs::read_dir(&dir) {
        for e in rd.flatten() {
            let Ok(name) = e.file_name().into_string() else { continue };
            if !name.starts_with(base) || (name.starts_with('.') && !base.starts_with('.')) {
                continue;
            }
            let p = if dir_part.is_empty() {
                name
            } else if dir_part.ends_with('/') {
                format!("{}{}", dir_part, name)
            } else {
                format!("{}/{}", dir_part, name)
            };
            out.insert(if is_dir(&p) { format!("{}/", p) } else { p });
        }
    }
    if !prefix.contains('/') {
        for d in klongpath.split(':') {
            if d.is_empty() || !is_dir(d) {
                continue;
            }
            if let Ok(rd) = std::fs::read_dir(d) {
                for e in rd.flatten() {
                    if let Ok(f) = e.file_name().into_string() {
                        if f.starts_with(prefix) && f.ends_with(".kg") {
                            out.insert(f);
                        }
                    }
                }
            }
        }
    }
    out.into_iter().collect()
}

fn chars_before(s: &str, byte: usize) -> usize {
    s[..byte].chars().count()
}

/// The `]l` file argument of a line: where it starts (a byte offset), when the line is `]l` and white space and the rest has no newline.
fn path_arg(line: &str) -> Option<usize> {
    let s = line.trim_start_matches(is_space);
    let off = line.len() - s.len();
    let rest = s.strip_prefix("]l")?;
    let arg = rest.trim_start_matches(is_space);
    if arg.len() == rest.len() {
        return None;
    }
    let start = off + 2 + (rest.len() - arg.len());
    let body = arg.strip_suffix('\n').unwrap_or(arg);
    if body.contains('\n') {
        return None;
    }
    Some(start)
}

/// Tab completion of the text before the cursor: `(start, words)`, each word replacing `line[start..]` (`start` in characters).
/// `names` are the defined names (system functions included), `commands` the names of the `]` commands without the bracket;
/// nothing is evaluated.
pub fn complete(line: &str, names: &[String], commands: &[String], klongpath: &str) -> (usize, Vec<String>) {
    // .l("PATH
    let mut best: Option<usize> = None;
    let mut from = 0;
    while let Some(i) = line[from..].find(".l(\"") {
        let at = from + i + 4;
        if !line[at..].contains('"') {
            best = Some(at);
            break;
        }
        from += i + 1;
    }
    if let Some(at) = best {
        return (chars_before(line, at), kg_files(&line[at..], klongpath));
    }
    let s = line.trim_start_matches(is_space);
    if s.starts_with(']') {
        let off = chars_before(line, line.len() - s.len());
        if !s.chars().any(is_space) {
            let mut w: Vec<String> = commands.iter().map(|k| format!("]{}", k)).filter(|k| k.starts_with(s)).collect();
            w.sort();
            w.dedup();
            return (off, w);
        }
        if s[1..].starts_with('l') {
            if let Some(start) = path_arg(line) {
                let arg = &line[start..];
                let arg = arg.strip_suffix('\n').unwrap_or(arg);
                return (chars_before(line, start), kg_files(arg, klongpath));
            }
        }
    }
    let end = if line.ends_with('\n') { line.len() - 1 } else { line.len() };
    let mut start = end;
    for (i, c) in line[..end].char_indices().rev() {
        if name_char(c) {
            start = i;
        } else {
            break;
        }
    }
    let prefix = &line[start..];
    let at = chars_before(line, start);
    if prefix.is_empty() {
        return (at, Vec::new());
    }
    let set: BTreeSet<&String> = names.iter().filter(|n| n.starts_with(prefix)).collect();
    (at, set.into_iter().cloned().collect())
}
