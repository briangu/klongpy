//! Modules: `.import`, `.export`, and `.l` of a file that exports (docs/design-principles.md).
//!
//! A file is a module and a module is a value (principles 3 and 4): `.import("geo")` runs the file in a scope of its own and gives
//! back the dictionary the file passed to `.export`. The scope itself, the lookup rule and the read-only dictionary are in the core
//! (`Interp::module_*`, `Dict::set_readonly`); this file is what needs the platform: finding the file, the cache, the cycle check.
//!
//! The rules that were decisions, each in one place:
//! * `resolve` (path resolution, resolution 8): `./x` and `../x` are relative to the importing file; an absolute path is itself;
//!   any other name is looked for in the import path (KLONGPATH) and then in the library of the image, never in the working
//!   directory or the importing file's directory, because a name that depends on where the program was started is a hidden input.
//! * `key_of` (the cache key): the real path of the file, `.kg` added when the name has no extension.
//! * `bind_alias` (`.l` of a module, resolution 9): the module is bound to the file's name in the loader's scope, as `.import` would.
//! * `select` (the two-argument form, resolution 6): the values of the named keys, in that order.
use alloc::format;
use alloc::rc::Rc;
use alloc::string::String;
use alloc::vec::Vec;
use core::cell::RefCell;

use klong_core::error::Result;
use klong_core::value::{Dict, Value, WILDCARD};
use klong_core::{Error, Interp};

use crate::{Platform, Shared};

#[derive(Default)]
pub(crate) struct Mods {
    /// The directories a bare name is looked for in (KLONGPATH); none by default.
    pub import_path: Vec<String>,
}

pub(crate) type ModsRef = RefCell<Mods>;

/// What an import found: the cached module, or the text of a file that is to run.
enum Found {
    Cached(Value),
    New { key: String, label: String, dir: String, lib: bool, text: Text },
}

pub(crate) enum Text {
    Owned(String),
    Static(&'static str),
}

/// The part of a path after its last slash.
fn last_part(p: &str) -> &str {
    match p.rfind('/') {
        Some(i) => &p[i + 1..],
        None => p,
    }
}

fn busy() -> Error {
    Error::internal("runtime state in use")
}

fn join(dir: &str, name: &str) -> String {
    if name.starts_with('/') || dir.is_empty() {
        return String::from(name);
    }
    let mut out = String::from(dir.trim_end_matches('/'));
    out.push('/');
    out.push_str(name);
    out
}

/// `x` becomes `x.kg`; a name that has an extension stays (resolution: the cache key and the file are the same for `m` and `m.kg`).
fn with_ext(name: &str) -> String {
    let mut out = String::from(name);
    if !last_part(name).contains('.') {
        out.push_str(".kg");
    }
    out
}

/// The path with `.` and `..` removed, lexically (a relative path that climbs above its start keeps its `..`).
fn normalize(path: &str) -> String {
    let mut out = String::new();
    if path.starts_with('/') {
        out.push('/');
    }
    for seg in path.split('/') {
        match seg {
            "" | "." => {}
            ".." if out == "/" => {}
            ".." if !out.is_empty() && !out.ends_with("..") => out.truncate(out.rfind('/').map_or(0, |i| i.max((out.starts_with('/')) as usize))),
            s => {
                if !out.is_empty() && !out.ends_with('/') {
                    out.push('/');
                }
                out.push_str(s);
            }
        }
    }
    out
}

/// The key of the file at `path`: its real path when the platform can say, else the absolute path with the dots taken out.
fn key_of<P: Platform + 'static>(sh: &Rc<Shared<P>>, path: &str) -> String {
    if let Ok(p) = sh.platform.try_borrow() {
        if let Some(r) = p.real_path(path) {
            return r;
        }
        if !path.starts_with('/') {
            if let Some(c) = p.cwd() {
                return normalize(&join(&c, path));
            }
        }
    }
    normalize(path)
}

/// Where the code that is running was written: its directory, and whether it is a file of the image's library.
fn importer<P: Platform + 'static>(sh: &Rc<Shared<P>>, ip: &Interp) -> Result<(String, bool)> {
    let id = ip.current_module_id();
    if id == 0 {
        let io = sh.io.try_borrow().map_err(|_| busy())?;
        return Ok((io.home.last().cloned().unwrap_or_default(), false));
    }
    Ok(ip.module_dir(id).map(|(d, l)| (String::from(d), l)).unwrap_or_default())
}

/// A cached module, or an error when the key is one whose body is running (a cycle). A module the core has let go is not cached.
fn cached<P: Platform + 'static>(_sh: &Rc<Shared<P>>, ip: &Interp, key: &str) -> Result<Option<Value>> {
    match ip.module_find(key) {
        None => Ok(None),
        Some((id, true)) => Ok(ip.module_export(id)),
        Some(_) => {
            // (the modules are listed in the order they began: from this one on, the ones still running are the cycle)
            let mut chain: Vec<&str> = ip.module_loading().into_iter().skip_while(|k| *k != key).collect();
            chain.push(key);
            #[cfg(not(feature = "diag"))]
            let _ = chain;
            let e = Error::system(klong_core::dmsg!("import cycle", "import cycle: {}", chain.join(" -> ")));
            #[cfg(feature = "diag")]
            let e = e.with_hint("a module cannot import a module that is importing it; move what they share into a third file");
            Err(e)
        }
    }
}

/// Find what `spec` names (resolution 8).
fn resolve<P: Platform + 'static>(sh: &Rc<Shared<P>>, ip: &Interp, spec: &str) -> Result<Found> {
    let (dir, from_lib) = importer(sh, ip)?;
    let relative = spec.starts_with("./") || spec.starts_with("../");
    let mut files: Vec<String> = Vec::new();
    let mut libname: Option<&str> = None;
    if relative && from_lib {
        // a file of the library reaches another by its name
        libname = Some(last_part(spec));
    } else if relative {
        files.push(normalize(&join(&dir, &with_ext(spec))));
    } else if spec.starts_with('/') {
        files.push(normalize(&with_ext(spec)));
    } else {
        let m = sh.mods.try_borrow().map_err(|_| busy())?;
        for d in &m.import_path {
            files.push(normalize(&join(d, &with_ext(spec))));
        }
        libname = Some(spec);
    }
    #[cfg(feature = "files")]
    for c in &files {
        let key = key_of(sh, c);
        if let Some(d) = cached(sh, ip, &key)? {
            return Ok(Found::Cached(d));
        }
        match crate::chan::read_file(sh, c) {
            Ok(text) => {
                let dir = crate::chan_dir(&key);
                return Ok(Found::New { key, label: c.clone(), dir, lib: false, text: Text::Owned(text) });
            }
            Err(e) if e.kind == klong_core::Kind::Memory => return Err(e),
            Err(e) => {
                // a file that is there and is not text is the answer, not a reason to look on
                if e.message.starts_with("not UTF-8") {
                    return Err(e);
                }
            }
        }
    }
    #[cfg(feature = "lib")]
    if let Some(n) = libname {
        if let Some((label, text)) = crate::library::find(n) {
            let key = format!("lib:{}", label);
            if let Some(d) = cached(sh, ip, &key)? {
                return Ok(Found::Cached(d));
            }
            return Ok(Found::New { key, label: String::from(label), dir: String::new(), lib: true, text: Text::Static(text) });
        }
    }
    #[cfg(not(feature = "lib"))]
    let _ = libname;
    let e = Error::system(klong_core::dmsg!("file does not exist", "file does not exist: {}", spec));
    #[cfg(feature = "diag")]
    let e = if files.is_empty() {
        e.with_hint("a name without ./ or ../ is looked for in KLONGPATH and the library, not in the working directory: write ./name for a file of your own")
    } else {
        e.with_hint(klong_core::dmsg!("module error", "looked for {}", files.join(", ")))
    };
    Err(e)
}

/// Run the file as a module (the scope is made, the body runs, `.export` has fixed the dictionary) and cache it.
fn run_new<P: Platform + 'static>(sh: &Rc<Shared<P>>, ip: &mut Interp, key: String, label: &str, dir: String, lib: bool, text: Text) -> Result<Value> {
    let id = ip.module_new(&key, &dir, lib)?;
    if let Ok(mut io) = sh.io.try_borrow_mut() {
        io.home.push(dir);
    }
    let r = match text {
        Text::Owned(t) => ip.module_run(id, t, label),
        Text::Static(t) => ip.module_run_static(id, t, label),
    };
    if let Ok(mut io) = sh.io.try_borrow_mut() {
        io.home.pop();
    }
    let r = r.and_then(|_| {
        ip.module_export(id).ok_or_else(|| {
            let e = Error::system(klong_core::dmsg!("not a module (no .export)", "{} has no .export, so it is not a module", label));
            #[cfg(feature = "diag")]
            let e = e.with_hint("load a classic file with .l, or end the file with .export(:{;\"name\";value})");
            e
        })
    });
    match r {
        Ok(d) => {
            ip.module_done(id);
            Ok(d)
        }
        Err(e) => {
            // a module that failed is not cached: the next import runs the file again
            ip.module_drop(id);
            Err(e)
        }
    }
}

/// `.import(spec)`: the module's dictionary.
pub(crate) fn import<P: Platform + 'static>(sh: &Rc<Shared<P>>, ip: &mut Interp, spec: &str) -> Result<Value> {
    match resolve(sh, ip, spec)? {
        Found::Cached(d) => Ok(d),
        Found::New { key, label, dir, lib, text } => run_new(sh, ip, key, &label, dir, lib, text),
    }
}

/// The module of a file that `.l` found (`path` for a file, `label` for a library file): the same module an `.import` of it gives.
pub(crate) fn import_loaded<P: Platform + 'static>(sh: &Rc<Shared<P>>, ip: &mut Interp, path: &str, text: Text) -> Result<Value> {
    let (key, dir, lib) = match &text {
        Text::Owned(_) => {
            let key = key_of(sh, path);
            let dir = crate::chan_dir(&key);
            (key, dir, false)
        }
        Text::Static(_) => (format!("lib:{}", path), String::new(), true),
    };
    if let Some(d) = cached(sh, ip, &key)? {
        return Ok(d);
    }
    run_new(sh, ip, key, path, dir, lib, text)
}

/// `.l` of a module binds the module's dictionary to the file's name in the scope that called `.l`: exactly `name::.import("name")`
/// (resolution 9, revised), so a program that loaded a library with `.l` and wrote `csv.read(...)` keeps working, and no plain name is defined.
pub(crate) fn bind_alias(ip: &mut Interp, path: &str, dict: Value) -> Result<()> {
    let stem = last_part(path);
    let stem = stem.strip_suffix(".kg").unwrap_or(stem);
    let s = ip.syms.intern(stem);
    if let Some(e) = ip.syms.failure() {
        return Err(e);
    }
    ip.assign(s, dict)
}

/// Does the text call `.export`? (A word `.export` followed by `(`, outside strings and comments.) `.l` runs such a file as a module.
pub(crate) fn calls_export(text: &str) -> bool {
    let b = text.as_bytes();
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'"' => {
                // a string: "" inside it is a quote
                i += 1;
                while i < b.len() {
                    if b[i] == b'"' {
                        if b.get(i + 1) == Some(&b'"') {
                            i += 2;
                            continue;
                        }
                        break;
                    }
                    i += 1;
                }
                i += 1;
            }
            b'.' if text[i..].starts_with(".export") => {
                let before = i == 0 || !(b[i - 1].is_ascii_alphanumeric() || b[i - 1] == b'.');
                let mut j = i + ".export".len();
                while j < b.len() && (b[j] == b' ' || b[j] == b'\t') {
                    j += 1;
                }
                if before && b.get(j) == Some(&b'(') {
                    return true;
                }
                i += 1;
            }
            _ => i += 1,
        }
    }
    false
}

/// `.import(path;keys)`: the values of `keys`, in that order; a key the module does not have is a `name` error that says which.
fn select(dict: &Value, spec: &str, keys: &Value) -> Result<Value> {
    if !keys.is_list() {
        return Err(Error::typ(klong_core::dmsg!(".import: names must be a list", ".import: the names to take must be a list of strings, got {}", keys.describe_long())));
    }
    let Value::Dict(d) = dict else { return Err(Error::internal("a module is not a dictionary")) };
    let n = keys.len().unwrap_or(0);
    let mut out: Vec<Value> = klong_core::value::try_vec(n)?;
    for i in 0..n {
        let k = keys.item(i);
        if !matches!(k, Value::Str(_)) {
            return Err(Error::typ(klong_core::dmsg!(".import: names must be strings", ".import: the names to take must be strings, got {}", k.describe_long())));
        }
        match d.borrow().get(&k) {
            Some(v) => out.push(v.clone()),
            None => {
                let name = if let Value::Str(s) = &k { s.as_str() } else { "" };
                return Err(Error::name(klong_core::dmsg!("no such entry", "{} has no entry \"{}\"", spec, name)));
            }
        }
    }
    Value::from_items(out)
}

pub(crate) fn register<P: Platform + 'static>(interp: &mut Interp, shared: &Rc<Shared<P>>) {
    let sh = shared.clone();
    interp.register(
        ".import",
        WILDCARD,
        Rc::new(move |ip: &mut Interp, a: &[Value]| {
            if a.is_empty() || a.len() > 2 {
                return Err(Error::arity(klong_core::dmsg!(".import: 1 or 2 arguments", ".import takes 1 or 2 arguments, got {}", a.len())));
            }
            let spec = match &a[0] {
                Value::Str(s) => s.as_str(),
                v => return Err(Error::typ(klong_core::dmsg!(".import: x must be a string", ".import: x must be a string, got {}", v.describe_long()))),
            };
            let d = import(&sh, ip, spec)?;
            match a.get(1) {
                Some(keys) => select(&d, spec, keys),
                None => Ok(d),
            }
        }),
    );
    // `.export(dict)`: what the file offers, once. The dictionary is copied, so the exported values are the values now (not variables that
    // change later), and it is read-only for whoever imports it. In a file that is not being imported it does nothing (a program may be
    // run directly, as a script).
    interp.register(
        ".export",
        WILDCARD,
        Rc::new(|ip: &mut Interp, a: &[Value]| {
            if a.len() != 1 {
                return Err(Error::arity(klong_core::dmsg!(".export: 1 argument", ".export takes 1 argument (a dictionary), got {}", a.len())));
            }
            let arg = crate::one(a, ".export")?;
            let Value::Dict(d) = arg else {
                return Err(Error::typ(klong_core::dmsg!(".export: x must be a dictionary", ".export: x must be a dictionary of names to values, got {}", arg.describe_long())));
            };
            let mut copy = Dict::default();
            {
                let d = d.borrow();
                for (k, v) in d.keys().iter().zip(d.vals().iter()) {
                    if !matches!(k, Value::Str(_)) {
                        return Err(Error::typ(klong_core::dmsg!(".export: keys must be strings", ".export: every key must be a string, got {}", k.describe_long())));
                    }
                    copy.set(k.clone(), v.clone())?;
                }
            }
            let id = ip.current_module_id();
            if id == 0 {
                return Ok(arg.clone());
            }
            copy.set_readonly();
            let v = Value::dict(copy);
            if !ip.module_set_export(id, v.clone()) {
                return Err(Error::system(".export was already called by this file"));
            }
            Ok(v)
        }),
    );
}
