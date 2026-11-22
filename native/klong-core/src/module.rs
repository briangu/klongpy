//! What a module is and where a file is looked for: the one place the runtime, the linter and the checker ask.
//!
//! * A file is a module exactly when it has a top-level assignment to the variable `export` (`is_export`, `interface`, `scan`). The
//!   interface is the dictionary the variable holds; when the file writes it as a literal, `interface` gives its keys and the nodes
//!   of its values, which is what a tool that does not run the file needs.
//! * `search` is the order a name given to `.l` is looked for in, parameterised over the environment (the importing file's directory,
//!   the KLONGPATH directories, a probe that says whether a file is there, and the bundled library) so that the runtime of a board and
//!   the tools that read files from a disk follow the same rules.
use alloc::rc::Rc;
use alloc::string::String;
use alloc::vec::Vec;

use crate::error::Source;
use crate::parse::parse_next;
use crate::sym::Interner;
use crate::tree::{Node, NodeId, Tree};
use crate::value::Value;

/// The name of the variable that holds a module's interface.
pub const EXPORT: &str = "export";

/// One entry of a literal interface: its value is a node of the file's tree, or (a dictionary of constants only is read by the parser
/// as a constant) a value.
pub enum Entry {
    Node(NodeId),
    Const(Value),
}

/// The `export` assignment of a file.
pub struct Interface {
    /// The statement `export::...`.
    pub stmt: NodeId,
    /// The right side.
    pub value: NodeId,
    /// The keys and values when the right side is a dictionary literal `:{;"k";v;...}` with string keys; `None` for anything else
    /// (a variable, a call, a key that is not a string literal).
    pub entries: Option<Vec<(String, Entry)>>,
}

/// The right side of `stmt` when it is `export::value` (a top-level statement).
fn export_rhs(tree: &Tree, syms: &Interner, stmt: NodeId) -> Option<NodeId> {
    if !Tree::is_real(stmt) {
        return None;
    }
    if let Node::Dyad { op, l, r, missing: false } = tree.node(stmt) {
        if syms.name(*op) == "::" && Tree::is_real(*l) && Tree::is_real(*r) {
            if let Node::Sym(s) = tree.node(*l) {
                if syms.name(*s) == EXPORT {
                    return Some(*r);
                }
            }
        }
    }
    None
}

/// Is the top-level statement `stmt` an assignment to `export`?
pub fn is_export(tree: &Tree, syms: &Interner, stmt: NodeId) -> bool {
    export_rhs(tree, syms, stmt).is_some()
}

/// The interface the top-level statement `stmt` declares, or `None` when it is not an assignment to `export`.
pub fn interface(tree: &Tree, syms: &Interner, stmt: NodeId) -> Option<Interface> {
    let value = export_rhs(tree, syms, stmt)?;
    let entries = match tree.node(value) {
        Node::EDict(r) => {
            let ids = tree.ids(*r);
            let mut out: Vec<(String, Entry)> = Vec::new();
            let mut ok = ids.len() % 2 == 0;
            for pair in ids.chunks(2) {
                if !ok {
                    break;
                }
                match tree.node(NodeId(pair[0])) {
                    Node::Lit(Value::Str(k)) => out.push((String::from(k.as_str()), Entry::Node(NodeId(pair[1])))),
                    _ => ok = false,
                }
            }
            ok.then_some(out)
        }
        Node::DictLit(Value::Dict(d)) => {
            let d = d.borrow();
            let mut out: Vec<(String, Entry)> = Vec::new();
            let mut ok = true;
            for (k, v) in d.keys().iter().zip(d.vals().iter()) {
                match k {
                    Value::Str(s) => out.push((String::from(s.as_str()), Entry::Const(v.clone()))),
                    _ => ok = false,
                }
            }
            ok.then_some(out)
        }
        _ => None,
    };
    Some(Interface { stmt, value, entries })
}

/// Is the text a module? The statements are parsed one at a time (nothing is run, nothing stays in memory) until one assigns `export`.
/// A text that does not parse to that point is not a module (running it reports the error).
pub fn scan(source: &Rc<Source>, max_depth: u32) -> bool {
    if !source.text.contains(EXPORT) {
        return false;
    }
    let mut syms = Interner::new();
    let mut module: Option<String> = None;
    let mut pos = 0usize;
    loop {
        let mut window = 256usize;
        let next = match parse_next(source, pos, source.text.len(), &mut window, None, &mut syms, &mut module, 0, max_depth, None) {
            Ok(n) => n,
            Err(_) => return false,
        };
        if next.skipped {
            pos = next.end;
            continue;
        }
        let Some(p) = next.parsed else { return false };
        if p.stmts.iter().any(|&s| is_export(&p.tree, &syms, s)) {
            return true;
        }
        if next.last {
            return false;
        }
        pos = next.end;
    }
}

/// A top-level statement that calls `.extern` and discards what it returns. `.extern(c_symbol;signature)` is a function value now, and a
/// statement of the old form (it defined a name) declares nothing a program can call.
pub fn discarded_extern(tree: &Tree, syms: &Interner, stmt: NodeId) -> bool {
    if !Tree::is_real(stmt) {
        return false;
    }
    if let Node::Fn { a, call: true, literal: false, .. } = tree.node(stmt) {
        if let Node::Sym(s) = tree.node(*a) {
            return syms.name(*s) == ".extern";
        }
    }
    false
}

/// What to say about a discarded `.extern`.
pub const DISCARDED_EXTERN: &str = ".extern returns the host function as a value, and this statement throws it away";
pub const DISCARDED_EXTERN_HINT: &str = "assign it, in a module file that exports it:  adc::.extern(\"read_adc\";\"int -> float\")  and  export:::{;\"adc\";adc}";

// ----- names ------------------------------------------------------------------------------------------------------------------

/// The part of a path after its last slash.
pub fn last_part(p: &str) -> &str {
    match p.rfind(|ch| ch == '/' || cfg!(windows) && ch == '\\') {
        Some(i) => &p[i + 1..],
        None => p,
    }
}

/// The name a file binds when it is loaded: its last path component without `.kg`.
pub fn stem(spec: &str) -> &str {
    let l = last_part(spec);
    l.strip_suffix(".kg").unwrap_or(l)
}

/// Can `s` be bound as a plain variable by `.l`? ASCII letters and digits, starting with a letter (a dot would make it a qualified name).
pub fn is_binding_name(s: &str) -> bool {
    let b = s.as_bytes();
    b.first().is_some_and(|c| c.is_ascii_alphabetic()) && b.iter().all(|c| c.is_ascii_alphanumeric()) && !matches!(s, "x" | "y" | "z" | "export")
}

/// `os.path.join(dir, name)`: `name` alone when it is absolute or `dir` is empty.
pub fn join(dir: &str, name: &str) -> String {
    if is_absolute(name) || dir.is_empty() {
        return String::from(name);
    }
    let mut out = String::from(dir.trim_end_matches('/'));
    out.push('/');
    out.push_str(name);
    out
}

/// Paths exchanged by the loader use slash separators. A Windows desktop also has drive and UNC roots.
fn split_root(path: &str) -> (&str, &str) {
    if cfg!(windows) {
        let unc = path.strip_prefix("//?/UNC/").or_else(|| path.strip_prefix("//").filter(|tail| !tail.starts_with("?/")));
        if let Some(tail) = unc {
            let mut parts = tail.splitn(3, '/');
            if matches!((parts.next(), parts.next()), (Some(server), Some(share)) if !server.is_empty() && !share.is_empty()) {
                let rest = parts.next().unwrap_or("");
                return path.split_at(path.len() - rest.len());
            }
        }
        let drive = path.strip_prefix("//?/").unwrap_or(path);
        if matches!(drive.as_bytes(), [letter, b':', b'/', ..] if letter.is_ascii_alphabetic()) {
            return path.split_at(path.len() - drive.len() + 3);
        }
    }
    if path.starts_with('/') { path.split_at(1) } else { ("", path) }
}

pub fn is_absolute(path: &str) -> bool {
    !split_root(path).0.is_empty()
}

pub fn is_library_name(path: &str) -> bool {
    !path.contains('/')
}

/// The path with `.` and `..` removed, lexically (a relative path that climbs above its start keeps its `..`).
pub fn normalize(path: &str) -> String {
    let (root, tail) = split_root(path);
    let mut out: Vec<&str> = Vec::new();
    for seg in tail.split('/') {
        match seg {
            "" | "." => {}
            ".." => {
                if matches!(out.last(), Some(l) if *l != "..") {
                    out.pop();
                } else if root.is_empty() {
                    out.push("..");
                }
            }
            s => out.push(s),
        }
    }
    let mut s = String::from(root);
    for (i, seg) in out.iter().enumerate() {
        if i > 0 {
            s.push('/');
        }
        s.push_str(seg);
    }
    s
}

/// The directory part of a path: `""` when there is none, `"/"` for a file in the root.
pub fn dir_of(path: &str) -> String {
    let (root, _) = split_root(path);
    match path.rfind('/') {
        Some(i) => String::from(&path[..i.max(root.len())]),
        None => String::new(),
    }
}

// ----- search -----------------------------------------------------------------------------------------------------------------

/// The environment of one search.
pub struct Env<'a> {
    /// The directory of the file that contains the `.l` (`""`: the working directory, as at the prompt).
    pub dir: &'a str,
    /// The importer is itself a file of the bundled library.
    pub from_lib: bool,
    /// The directories of KLONGPATH, in order.
    pub klongpath: &'a [String],
}

/// Where a name was found.
#[derive(Debug, PartialEq, Eq)]
pub enum Place {
    /// A file at this path: tried in the importing file's directory, a KLONGPATH directory or as an absolute path.
    File(String),
    /// The bundled library, by name.
    Library(String),
}

pub struct Found {
    pub place: Place,
    /// A file that was found in the importing file's directory or in KLONGPATH has the name of a bundled library file.
    pub shadows: bool,
}

/// The names `spec` can be in `dir`: the name as written and, unless it ends in `.kg`, with `.kg`.
fn candidates(dir: &str, spec: &str, out: &mut Vec<String>) {
    let joined = join(dir, spec);
    let with_kg = if joined.ends_with(".kg") { None } else { Some(crate::msg!("{}.kg", joined)) };
    out.push(joined);
    out.extend(with_kg);
}

/// Look for `spec`:
///
/// 1. `./x` and `../x`: relative to the importing file's directory (a file of the library reaches another of it by name);
///    an absolute path is itself.
/// 2. any other name: the importing file's directory, then each KLONGPATH directory, then the bundled library (a file of the
///    library itself looks in the library first).
///
/// `probe(path)` is asked for each candidate in order and answers true to stop there (a file that is there, or a failure the caller keeps);
/// `in_lib(stem)` says whether the bundled library has a file of that name. `tried` collects the candidates asked, for the message of a
/// search that found nothing. The result is `None` when nothing was found or a probe stopped the search at a failure (the caller knows).
pub fn search(env: &Env, spec: &str, probe: &mut dyn FnMut(&str) -> bool, in_lib: &mut dyn FnMut(&str) -> bool, tried: &mut Vec<String>) -> Option<Found> {
    let name = stem(spec);
    let relative = spec.starts_with("./") || spec.starts_with("../");
    let absolute = is_absolute(spec);
    let lib_name = is_library_name(spec);
    if env.from_lib && (relative || lib_name) && in_lib(name) {
        return Some(Found { place: Place::Library(String::from(name)), shadows: false });
    }
    // (the working directory is "" and "." alike: it is asked once)
    let mut dirs: Vec<String> = Vec::new();
    if absolute {
        dirs.push(String::new());
    } else if relative {
        dirs.push(normalize(env.dir));
    } else {
        dirs.push(normalize(env.dir));
        for d in env.klongpath {
            let d = normalize(d);
            if !dirs.contains(&d) {
                dirs.push(d);
            }
        }
    }
    let mut files: Vec<String> = Vec::new();
    for d in &dirs {
        candidates(d, spec, &mut files);
    }
    for c in &files {
        let c = if relative { normalize(c) } else { c.clone() };
        tried.push(c.clone());
        if probe(&c) {
            let shadows = lib_name && in_lib(name);
            return Some(Found { place: Place::File(c), shadows });
        }
    }
    if lib_name && in_lib(name) {
        return Some(Found { place: Place::Library(String::from(name)), shadows: false });
    }
    None
}
