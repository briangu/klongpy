//! klong-lint: the static check for the classic Klong traps, a port of `klongpy/lint.py` onto the tokens and tree of
//! `klong-core`. See `docs/lint.md` for the rules.
//!
//! ```text
//!     lint_source(name, text, &options) -> Vec<Diagnostic>
//!     lint_files(&paths, &options)      -> Vec<FileResult>
//!     lint_main(args, ...)              -> exit status (the body of `kg lint` and `kgpy lint`)
//! ```
//!
//! The linter reads a program; it never runs it. No user code is evaluated; `.l("file")` is resolved by reading and
//! parsing that file. Positions are byte offsets in `Diagnostic::span`; `line` and `col` are 1-based, the column in
//! characters.
mod analysis;
pub mod recover;
mod builtins;
pub use builtins::builtin_names;
mod cli;
mod load;
mod rules;
mod scan;
pub mod tok;

pub use cli::{lint_main, run_files};
pub use load::Session;

use analysis::Analysis;
use std::collections::{BTreeSet, HashMap};
use tok::Source;

pub struct Rule {
    pub code: &'static str,
    pub severity: &'static str,
    pub title: &'static str,
}

pub static RULES: &[Rule] = &[
    Rule { code: "W001", severity: "warning", title: "_ between name characters: it is the Drop operator" },
    Rule { code: "W002", severity: "warning", title: "name used before its definition, or never defined" },
    Rule { code: "W003", severity: "warning", title: "declared local never used" },
    Rule { code: "W010", severity: "warning", title: "leading - followed by a number and a dyad negates the whole expression" },
    Rule { code: "W011", severity: "warning", title: "function uses y without x, or z without y" },
    Rule { code: "W012", severity: "warning", title: "bare name in a list is the symbol, not the value" },
    Rule { code: "W013", severity: "warning", title: ":[ in the else position of :[ can be the :| chain" },
    Rule { code: "W014", severity: "warning", title: "whitespace between a name or } and ( or an adverb" },
    Rule { code: "W015", severity: "warning", title: "newline after a comment inside a body or conditional acts as ;" },
    Rule { code: "W016", severity: "warning", title: ": directly before a digit is ignored" },
    Rule { code: "W017", severity: "warning", title: "; a bracket or an operator inside a data list is data" },
    Rule { code: "W018", severity: "warning", title: "number immediately followed by a name" },
    Rule { code: "W019", severity: "warning", title: "assignment to x, y or z" },
    Rule { code: "W020", severity: "warning", title: "a two-element list joins into a dictionary with ," },
    Rule { code: "W021", severity: "warning", title: "a.b::v where a is never a dictionary (a variable named with a dot in Klong 1)" },
    Rule { code: "W022", severity: "warning", title: "module imported by .import and never used" },
    Rule { code: "E001", severity: "error", title: "the parser rejects this at strict level 1" },
    Rule { code: "E002", severity: "error", title: "call with more arguments than the callee takes" },
    Rule { code: "E003", severity: "error", title: "assignment to an entry of what .import returned (read-only)" },
    Rule { code: "E004", severity: "error", title: "qualified name that the imported module does not export" },
];

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Diagnostic {
    pub code: &'static str,
    pub severity: &'static str,
    /// byte offsets (start, end) into the text
    pub span: (usize, usize),
    pub line: usize,
    /// in characters
    pub col: usize,
    pub message: String,
    pub hint: Option<String>,
    /// the name of the source
    pub file: String,
}

impl Diagnostic {
    /// `file:line:col: CODE message` (and, with `hint`, an indented `hint:` line).
    pub fn format(&self, hint: bool) -> String {
        let mut s = format!("{}:{}:{}: {} {}", self.file, self.line, self.col, self.code, self.message);
        if hint {
            if let Some(h) = self.hint.as_ref().filter(|h| !h.is_empty()) {
                s.push_str("\n  hint: ");
                s.push_str(h);
            }
        }
        s
    }
}

impl std::fmt::Display for Diagnostic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.format(false))
    }
}

#[derive(Clone, Default)]
pub struct Options {
    /// only these codes (None: all)
    pub select: Option<BTreeSet<String>>,
    /// leave these codes out
    pub ignore: BTreeSet<String>,
    /// names the host defines before the program runs: W002 does not report them
    pub define: BTreeSet<String>,
    /// the system names of an embedding beyond the runtime's own (the Python package's `.py`, `.fpga`, ...), as the caller lists them:
    /// like `define`, W002 does not report them
    pub extra_builtins: BTreeSet<String>,
    /// directory for `.l` lookups, instead of the directory of the name
    pub base_dir: Option<String>,
    /// false: never read other files; every `.l` then counts as one that defines unknown names (W002 off)
    pub no_resolve: bool,
    /// library directories searched after KLONGPATH (klongpy/lib); empty: the library built into the binary
    pub lib_dirs: Vec<String>,
}

fn parse_list(value: &str) -> Vec<&str> {
    value.split(|c: char| c == ',' || c.is_whitespace()).filter(|s| !s.is_empty()).collect()
}

/// A comma- or space-separated string of codes as a set of upper-case codes; Err(message) on an unknown code.
pub fn parse_codes(value: &str) -> Result<BTreeSet<String>, String> {
    let mut out = BTreeSet::new();
    for v in parse_list(value) {
        let c = v.trim().to_uppercase();
        if !RULES.iter().any(|r| r.code == c) {
            let codes: Vec<&str> = RULES.iter().map(|r| r.code).collect();
            return Err(format!("unknown lint code {}; codes are {}", cli::py_repr(v), codes.join(", ")));
        }
        out.insert(c);
    }
    Ok(out)
}

/// A comma- or space-separated string of names as a set.
pub fn parse_names(value: &str) -> BTreeSet<String> {
    parse_list(value).into_iter().map(String::from).collect()
}

/// {line: codes} from the `lint:ignore` comments.
fn suppressed(an: &Analysis) -> HashMap<usize, BTreeSet<String>> {
    let mut out: HashMap<usize, BTreeSet<String>> = HashMap::new();
    let text = an.text();
    for &(s, e) in &an.trivia {
        let c = &text[s..e];
        if let Some(codes) = ignore_codes(c) {
            if !codes.is_empty() {
                out.entry(an.src.line_col(s).0).or_default().extend(codes);
            }
        }
    }
    out
}

fn is_word(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// The codes of the first `lint:ignore\b([\sA-Za-z0-9,]*)` in a comment, as `re.search` finds it.
fn ignore_codes(c: &str) -> Option<Vec<String>> {
    const KEY: &str = "lint:ignore";
    let mut from = 0;
    while let Some(p) = c[from..].find(KEY) {
        let at = from + p + KEY.len();
        let boundary = match c[at..].chars().next() {
            None => true,
            Some(ch) => !is_word(ch),
        };
        if boundary {
            let rest = &c[at..];
            let end = rest.find(|ch: char| !(ch.is_whitespace() || ch.is_ascii_alphanumeric() || ch == ',')).unwrap_or(rest.len());
            let g = rest[..end].to_ascii_uppercase();
            let b = g.as_bytes();
            let mut codes = Vec::new();
            let mut i = 0;
            while i + 4 <= b.len() {
                if (b[i] == b'W' || b[i] == b'E') && b[i + 1..i + 4].iter().all(|d| d.is_ascii_digit()) {
                    codes.push(g[i..i + 4].to_string());
                    i += 4;
                } else {
                    i += 1;
                }
            }
            return Some(codes);
        }
        from = at;
    }
    None
}

/// `os.path.abspath`.
fn abspath(p: &str) -> String {
    let joined = if p.starts_with('/') {
        p.to_string()
    } else {
        let cwd = std::env::current_dir().map(|c| c.to_string_lossy().into_owned()).unwrap_or_default();
        load::py_join(&cwd, p)
    };
    let lead = if joined.starts_with("//") && !joined.starts_with("///") { 2 } else if joined.starts_with('/') { 1 } else { 0 };
    let mut out: Vec<&str> = Vec::new();
    for seg in joined.split('/') {
        if seg.is_empty() || seg == "." {
            continue;
        }
        if seg == ".." {
            if !out.is_empty() && *out.last().unwrap() != ".." {
                out.pop();
            } else if lead == 0 {
                out.push("..");
            }
            continue;
        }
        out.push(seg);
    }
    format!("{}{}", "/".repeat(lead), out.join("/"))
}

/// Lint a text with the files that `.l` loads shared through `sess`.
pub fn lint_source_in(sess: &mut Session, name: &str, text: &str, opts: &Options) -> Vec<Diagnostic> {
    let base_dir = match &opts.base_dir {
        Some(b) => b.clone(),
        None => {
            if std::fs::metadata(name).map(|m| m.is_file()).unwrap_or(false) {
                load::py_dirname(&abspath(name))
            } else {
                std::env::current_dir().map(|c| c.to_string_lossy().into_owned()).unwrap_or_default()
            }
        }
    };
    let an = Analysis::new(Source::new(name, text.to_string()), &base_dir, !opts.no_resolve, &[], 0, true, sess);
    let known: BTreeSet<String> = opts.define.union(&opts.extra_builtins).cloned().collect();
    let mut lin = rules::Linter::new(&an, &known);
    lin.run();
    let sup = suppressed(&an);
    let mut out: Vec<Diagnostic> = Vec::new();
    let mut seen: BTreeSet<(&'static str, (usize, usize), String)> = BTreeSet::new();
    for f in lin.found {
        if opts.select.as_ref().map(|s| !s.contains(f.code)).unwrap_or(false) || opts.ignore.contains(f.code) {
            continue;
        }
        let (line, col) = an.src.line_col(f.span.0);
        if sup.get(&line).map(|s| s.contains(f.code)).unwrap_or(false) {
            continue;
        }
        if !seen.insert((f.code, f.span, f.message.clone())) {
            continue;
        }
        let severity = RULES.iter().find(|r| r.code == f.code).map(|r| r.severity).unwrap_or("warning");
        out.push(Diagnostic { code: f.code, severity, span: f.span, line, col, message: f.message, hint: f.hint, file: name.to_string() });
    }
    out.sort_by(|a, b| (a.span.0, a.code).cmp(&(b.span.0, b.code)));
    out
}

/// Lint a program text. `name` is its name in the findings, and where `.l("file")` is looked up (the directory of
/// `name`, when that is a file).
pub fn lint_source(name: &str, text: &str, opts: &Options) -> Vec<Diagnostic> {
    let mut sess = Session::new(opts.lib_dirs.clone());
    lint_source_in(&mut sess, name, text, opts)
}

/// What reading a file gave.
pub struct FileResult {
    pub path: String,
    pub result: Result<Vec<Diagnostic>, String>,
}

/// Read a file as Python's `open(path, 'r').read()` does: UTF-8, and `\r\n` and `\r` read as `\n`.
pub fn read_text(path: &str) -> Result<String, String> {
    let bytes = std::fs::read(path).map_err(io_text)?;
    decode(&bytes)
}

pub(crate) fn io_text(e: std::io::Error) -> String {
    let s = e.to_string();
    match s.rfind(" (os error") {
        Some(i) => s[..i].to_string(),
        None => s,
    }
}

/// UTF-8 decoding with universal newlines; the error text is Python's `UnicodeDecodeError`.
pub fn decode(bytes: &[u8]) -> Result<String, String> {
    match std::str::from_utf8(bytes) {
        Ok(s) => Ok(translate_newlines(s)),
        Err(e) => {
            let pos = e.valid_up_to();
            Err(match e.error_len() {
                None => {
                    let n = bytes.len() - pos;
                    if n == 1 {
                        format!("'utf-8' codec can't decode byte 0x{:02x} in position {}: unexpected end of data", bytes[pos], pos)
                    } else {
                        format!("'utf-8' codec can't decode bytes in position {}-{}: unexpected end of data", pos, pos + n - 1)
                    }
                }
                Some(1) => {
                    let b = bytes[pos];
                    let why = if (0xc2..=0xf4).contains(&b) { "invalid continuation byte" } else { "invalid start byte" };
                    format!("'utf-8' codec can't decode byte 0x{:02x} in position {}: {}", b, pos, why)
                }
                Some(n) => format!("'utf-8' codec can't decode bytes in position {}-{}: invalid continuation byte", pos, pos + n - 1),
            })
        }
    }
}

fn translate_newlines(s: &str) -> String {
    if !s.contains('\r') {
        return s.to_string();
    }
    s.replace("\r\n", "\n").replace('\r', "\n")
}

/// Lint files; a file that cannot be read gives an Err with the reason. A path `-` is read from stdin.
pub fn lint_files(paths: &[String], opts: &Options) -> Vec<FileResult> {
    let mut sess = Session::new(opts.lib_dirs.clone());
    paths
        .iter()
        .map(|p| {
            let result = if p == "-" {
                use std::io::Read;
                let mut buf = Vec::new();
                match std::io::stdin().read_to_end(&mut buf) {
                    Ok(_) => decode(&buf).map(|t| lint_source_in(&mut sess, "<stdin>", &t, opts)),
                    Err(e) => Err(io_text(e)),
                }
            } else {
                read_text(p).map(|t| lint_source_in(&mut sess, p, &t, opts))
            };
            FileResult { path: p.clone(), result }
        })
        .collect()
}
