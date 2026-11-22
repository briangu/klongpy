//! `.l`, the one loader, and the modules it makes (docs/design-principles.md).
//!
//! A file is a module exactly when it assigns the variable `export` at its top level (`klong_core::module::scan`: the same decision the
//! linter and the checker make). `.l(path)` runs a module in a scope of its own, once, and binds it to the file's stem; `.l(path;:name)`
//! binds it to `name`; `.l(path;["a" "b"])` binds those entries as plain names. A file that is not a module is loaded as classic Klong
//! loads it: its names go into the globals. The scope itself, the lookup rule and the read-only dictionary are in the core
//! (`Interp::module_*`, `Dict::set_readonly`); the search order is `klong_core::module::search`; this file is what needs the platform:
//! reading the file, the cache, the cycle check.
#[cfg(any(feature = "files", feature = "lib"))]
use alloc::format;
use alloc::rc::Rc;
#[allow(unused_imports)]
use alloc::string::String;
#[allow(unused_imports)]
use alloc::vec::Vec;

#[allow(unused_imports)]
use klong_core::error::Result;
#[allow(unused_imports)]
use klong_core::module::{self, Env, Place};
#[allow(unused_imports)]
use klong_core::value::{Dict, Value, WILDCARD};
#[allow(unused_imports)]
use klong_core::{Error, Interp};

use crate::{Platform, Shared};

#[cfg(any(feature = "files", feature = "lib"))]
pub(crate) enum Text {
    Owned(String),
    Static(&'static str),
}

/// What a search found.
#[cfg(any(feature = "files", feature = "lib"))]
enum Found {
    /// The module of this file is already made.
    Cached(Value),
    File { path: String, key: String, text: Text, lib: bool },
}

#[cfg(any(feature = "files", feature = "lib"))]
fn busy() -> Error {
    Error::internal("runtime state in use")
}

/// The key of the file at `path`: its real path when the platform can say, else the absolute path with the dots taken out.
#[cfg(feature = "files")]
fn key_of<P: Platform + 'static>(sh: &Rc<Shared<P>>, path: &str) -> String {
    if let Ok(p) = sh.platform.try_borrow() {
        if let Some(r) = p.real_path(path) {
            return r;
        }
        if !module::is_absolute(path) {
            if let Some(c) = p.cwd() {
                return module::normalize(&module::join(&c, path));
            }
        }
    }
    module::normalize(path)
}

/// Where the code that is running was written: its directory, and whether it is a file of the image's library.
#[cfg(any(feature = "files", feature = "lib"))]
fn importer<P: Platform + 'static>(sh: &Rc<Shared<P>>, ip: &Interp) -> Result<(String, bool)> {
    let id = ip.current_module_id();
    if id == 0 {
        let io = sh.io.try_borrow().map_err(|_| busy())?;
        return Ok((io.home.last().cloned().unwrap_or_default(), false));
    }
    Ok(ip.module_dir(id).map(|(d, l)| (String::from(d), l)).unwrap_or_default())
}

/// A cached module, or an error when the key is one whose body is running (a cycle). A module the core has let go is not cached.
#[cfg(any(feature = "files", feature = "lib"))]
fn cached(ip: &Interp, key: &str) -> Result<Option<Value>> {
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
            let e = e.with_hint("a module cannot load a module that is loading it; move what they share into a third file");
            Err(e)
        }
    }
}

#[cfg(any(feature = "files", feature = "lib"))]
fn library(name: &str) -> Option<&'static str> {
    #[cfg(feature = "lib")]
    {
        crate::library::find(name).map(|(_, t)| t)
    }
    #[cfg(not(feature = "lib"))]
    {
        let _ = name;
        None
    }
}

/// Find what `spec` names, by the shared search, and read it (or take the module already made from it).
#[cfg(any(feature = "files", feature = "lib"))]
fn locate<P: Platform + 'static>(sh: &Rc<Shared<P>>, ip: &mut Interp, spec: &str) -> Result<Found> {
    let (dir, from_lib) = importer(sh, ip)?;
    let klongpath: Vec<String> = sh.io.try_borrow().map_err(|_| busy())?.path.clone();
    let env = Env { dir: &dir, from_lib, klongpath: &klongpath };
    // what the probe found: a cached module, the text of a file, or the failure that ends the search
    #[allow(unused_mut)]
    let mut got: Option<Result<(String, String, Text)>> = None;
    #[allow(unused_mut)]
    let mut hit: Option<Value> = None;
    let mut tried: Vec<String> = Vec::new();
    let found = {
        #[cfg(feature = "files")]
        let mut probe = |c: &str| -> bool {
            let key = key_of(sh, c);
            match cached(ip, &key) {
                Ok(Some(d)) => {
                    hit = Some(d);
                    return true;
                }
                Ok(None) => {}
                Err(e) => {
                    got = Some(Err(e));
                    return true;
                }
            }
            match crate::chan::read_file(sh, c) {
                Ok(text) => {
                    got = Some(Ok((String::from(c), key, Text::Owned(text))));
                    true
                }
                // a file that is there and is not text is the answer, not a reason to look on
                Err(e) if e.kind == klong_core::Kind::Memory || e.message.starts_with("not UTF-8") => {
                    got = Some(Err(e));
                    true
                }
                Err(_) => false,
            }
        };
        #[cfg(not(feature = "files"))]
        let mut probe = |_: &str| -> bool { false };
        let mut in_lib = |n: &str| library(n).is_some();
        module::search(&env, spec, &mut probe, &mut in_lib, &mut tried)
    };
    if let Some(d) = hit {
        return Ok(Found::Cached(d));
    }
    if let Some(r) = got {
        let (path, key, text) = r?;
        if found.as_ref().is_some_and(|f| f.shadows) {
            shadow_warning(ip, spec, &path, &text);
        }
        return Ok(Found::File { path, key, text, lib: false });
    }
    if let Some(f) = found {
        if let Place::Library(name) = f.place {
            if let Some(text) = library(&name) {
                let key = format!("lib:{}", name);
                if let Some(d) = cached(ip, &key)? {
                    return Ok(Found::Cached(d));
                }
                return Ok(Found::File { path: name, key, text: Text::Static(text), lib: true });
            }
        }
    }
    #[cfg(not(feature = "diag"))]
    let e = Error::system("file does not exist");
    #[cfg(feature = "diag")]
    let e = {
        let mut msg = klong_core::msg!("file does not exist: {} (looked for {}", spec, tried.join(", "));
        if !tried.is_empty() && module::is_library_name(spec) {
            msg.push_str(", then the bundled library");
        }
        msg.push(')');
        Error::system(msg)
    };
    #[cfg(not(feature = "diag"))]
    let _ = tried;
    #[cfg(feature = "diag")]
    let e = if !cfg!(feature = "files") { e.with_hint("this platform has no file system; .l loads the built-in library only") } else { e };
    Err(e)
}

/// A file found in the importing file's directory or in KLONGPATH has the name of a file of the bundled library and is not that file:
/// the program gets the file it has, and is told (the engine's warning list).
#[cfg(any(feature = "files", feature = "lib"))]
fn shadow_warning(ip: &mut Interp, spec: &str, path: &str, text: &Text) {
    #[cfg(feature = "diag")]
    {
        let name = module::stem(spec);
        let same = match (library(name), text) {
            (Some(l), Text::Owned(t)) => l == t.as_str(),
            (Some(l), Text::Static(t)) => l == *t,
            (None, _) => true,
        };
        if same {
            return;
        }
        ip.syms.warn(klong_core::sym::Warning {
            code: "shadows-library",
            message: klong_core::msg!("{} is found at {}, which shadows the bundled library file of the same name", spec, path),
            note: None,
            hint: Some(klong_core::msg!("rename the file, or write ./{} to say that the local file is meant", name)),
            span: (0, 0),
            source: None,
        });
    }
    #[cfg(not(feature = "diag"))]
    let _ = (ip, spec, path, text);
}

/// Run the file as a module (the scope is made, the body runs, the variable `export` is read and frozen) and cache it.
#[cfg(any(feature = "files", feature = "lib"))]
fn run_new<P: Platform + 'static>(sh: &Rc<Shared<P>>, ip: &mut Interp, key: String, label: &str, lib: bool, text: Text) -> Result<Value> {
    let dir = if lib { String::new() } else { module::dir_of(&key) };
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
    let r = r.and_then(|_| freeze(ip, id, label));
    match r {
        Ok(d) => {
            ip.module_done(id);
            Ok(d)
        }
        Err(e) => {
            // a module that failed is not cached: the next load runs the file again
            ip.module_drop(id);
            Err(e)
        }
    }
}

/// The interface of module `id`: the dictionary its variable `export` holds, copied (the values are the values now, not variables that
/// change later) and made read-only.
#[cfg(any(feature = "files", feature = "lib"))]
fn freeze(ip: &mut Interp, id: u16, label: &str) -> Result<Value> {
    #[cfg(not(feature = "diag"))]
    let _ = label;
    let Some(v) = ip.module_get(id, module::EXPORT) else {
        return Err(Error::system(klong_core::dmsg!("export was not assigned", "{} did not assign export", label)));
    };
    let Value::Dict(d) = &v else {
        return Err(Error::typ(klong_core::dmsg!("export must be a dictionary", "{}: export must be a dictionary of names to values, got {}", label, v.describe_long()))
            .with_hint("write  export:::{;\"name\";value}"));
    };
    let mut copy = Dict::default();
    {
        let d = d.borrow();
        for (k, val) in d.keys().iter().zip(d.vals().iter()) {
            if !matches!(k, Value::Str(_)) {
                return Err(Error::typ(klong_core::dmsg!("export keys must be strings", "{}: every key of export must be a string, got {}", label, k.describe_long())));
            }
            copy.set(k.clone(), val.clone())?;
        }
    }
    copy.set_readonly();
    let out = Value::dict(copy);
    drop(v);
    ip.module_remove(id, module::EXPORT);
    if !ip.module_set_export(id, out.clone()) {
        return Err(Error::internal("the interface was fixed twice"));
    }
    Ok(out)
}

/// A name `.l` is to bind: a string or a symbol that is a plain variable name.
#[cfg(any(feature = "files", feature = "lib"))]
fn bind_name(ip: &mut Interp, name: &str, v: Value, what: &str) -> Result<()> {
    let _ = what;
    if !module::is_binding_name(name) {
        return Err(Error::name(klong_core::dmsg!(".l: not a name", ".l: \"{}\" cannot be the name of a variable; {}", name, what)));
    }
    let s = ip.syms.intern(name);
    if let Some(e) = ip.syms.failure() {
        return Err(e);
    }
    ip.assign(s, v)
}

/// The second argument of `.l`, checked before anything is loaded.
#[cfg(any(feature = "files", feature = "lib"))]
enum Bind {
    Stem,
    Alias(String),
    Names(Vec<String>),
}

#[cfg(any(feature = "files", feature = "lib"))]
fn bind_of(ip: &Interp, a: &[Value]) -> Result<Bind> {
    match a.get(1) {
        None => Ok(Bind::Stem),
        Some(Value::Sym(s)) => Ok(Bind::Alias(klong_core::value::try_str(ip.syms.name(*s))?)),
        Some(v) if v.is_list() => {
            let n = v.len().unwrap_or(0);
            let mut out: Vec<String> = Vec::new();
            for i in 0..n {
                match v.item(i) {
                    Value::Str(s) => {
                        klong_core::value::try_push(&mut out, klong_core::value::try_str(s.as_str())?)?;
                    }
                    other => {
                        let _ = &other;
                        return Err(Error::typ(klong_core::dmsg!(".l: names must be strings", ".l: the entries to bind must be strings, got {}", other.describe_long()))
                            .with_hint("write .l(\"file\";[\"name1\" \"name2\"])"))
                    }
                }
            }
            Ok(Bind::Names(out))
        }
        Some(v) => Err(Error::typ({ let _ = v; klong_core::dmsg!(".l: second argument", ".l: the second argument must be a symbol (the name to bind the module to) or a list of strings (the entries to bind), got {}", v.describe_long()) })
            .with_hint("write .l(\"file\";:name) or .l(\"file\";[\"a\" \"b\"])")),
    }
}

/// The file's stem is a name to bind, when no name was given: said before the module runs.
#[cfg(any(feature = "files", feature = "lib"))]
fn check_stem(spec: &str, how: &Bind) -> Result<()> {
    let stem = module::stem(spec);
    if matches!(how, Bind::Stem) && !module::is_binding_name(stem) {
        return Err(Error::typ(klong_core::dmsg!(".l: file name is not a name", ".l: the module \"{}\" cannot be bound to the name \"{}\"", spec, stem))
            .with_hint(klong_core::msg!("choose the name: .l(\"{}\";:name)", spec)));
    }
    Ok(())
}

/// Bind the module `dict` that `spec` loaded as `bind` says; the module is the value of `.l`.
#[cfg(any(feature = "files", feature = "lib"))]
fn bind(ip: &mut Interp, spec: &str, dict: &Value, how: Bind) -> Result<()> {
    match how {
        Bind::Stem => {
            check_stem(spec, &Bind::Stem)?;
            bind_name(ip, module::stem(spec), dict.clone(), "")
        }
        Bind::Alias(name) => bind_name(ip, &name, dict.clone(), "a name is letters and digits, starting with a letter"),
        Bind::Names(names) => {
            let Value::Dict(d) = dict else { return Err(Error::internal("a module is not a dictionary")) };
            let mut vals: Vec<Value> = Vec::new();
            for n in &names {
                let key = Value::text(n)?;
                match d.borrow().get(&key) {
                    Some(v) => klong_core::value::try_push(&mut vals, v.clone())?,
                    None => {
                        let e = Error::name(klong_core::dmsg!("no such entry", "{} has no entry \"{}\"", spec, n));
                        #[cfg(feature = "diag")]
                        let e = {
                            let mut have: Vec<String> = Vec::new();
                            for k in d.borrow().keys() {
                                if let Value::Str(s) = k {
                                    have.push(String::from(s.as_str()));
                                }
                            }
                            e.with_hint(klong_core::msg!("the module has: {}", have.join(" ")))
                        };
                        return Err(e);
                    }
                }
            }
            for (n, v) in names.iter().zip(vals) {
                bind_name(ip, n, v, "an entry is bound under its own name, which must be letters and digits")?;
            }
            Ok(())
        }
    }
}

/// `.l(spec)`, `.l(spec;:name)`, `.l(spec;[names])`.
#[cfg(any(feature = "files", feature = "lib"))]
fn load<P: Platform + 'static>(sh: &Rc<Shared<P>>, ip: &mut Interp, a: &[Value]) -> Result<Value> {
    if a.is_empty() || a.len() > 2 {
        return Err(Error::arity(klong_core::dmsg!(".l: 1 or 2 arguments", ".l takes 1 or 2 arguments, got {}", a.len())));
    }
    let spec = crate::text_arg(&a[..1], ".l")?;
    let spec = sh.platform.try_borrow().map_err(|_| busy())?.module_path(spec);
    let spec = spec.as_ref();
    let how = bind_of(ip, a)?;
    let given = a.len() == 2;
    let (path, key, text, lib) = match locate(sh, ip, spec)? {
        Found::Cached(d) => {
            bind(ip, spec, &d, how)?;
            return Ok(d);
        }
        Found::File { path, key, text, lib } => (path, key, text, lib),
    };
    // is it a module? the statements are parsed, not run
    let (text, is_module) = {
        let (cow, owned) = match text {
            Text::Owned(t) => (alloc::borrow::Cow::Owned(t), true),
            Text::Static(t) => (alloc::borrow::Cow::Borrowed(t), false),
        };
        let src = Rc::new(klong_core::error::Source { name: String::new(), text: cow });
        let m = module::scan(&src, ip.cfg.max_depth);
        let text = match Rc::try_unwrap(src) {
            Ok(s) => match (s.text, owned) {
                (alloc::borrow::Cow::Owned(t), _) => Text::Owned(t),
                (alloc::borrow::Cow::Borrowed(t), _) => Text::Static(t),
            },
            Err(s) => Text::Owned(klong_core::value::try_str(&s.text)?),
        };
        (text, m)
    };
    if is_module {
        check_stem(spec, &how)?;
        let d = run_new(sh, ip, key, &path, lib, text)?;
        bind(ip, spec, &d, how)?;
        return Ok(d);
    }
    if given {
        let e = Error::typ(klong_core::dmsg!(".l: not a module", ".l: {} is not a module (it does not assign export), so there is nothing to bind", spec));
        #[cfg(feature = "diag")]
        let e = e.with_hint(klong_core::msg!("write .l(\"{}\") to load it as classic Klong, or make it a module with  export:::{{;\"name\";value}}", spec));
        return Err(e);
    }
    // a classic file: while it runs, the directory of this file is where its own .l calls look first
    let dir = if lib { String::new() } else { module::dir_of(&path) };
    if let Ok(mut io) = sh.io.try_borrow_mut() {
        io.home.push(dir);
    }
    let r = match text {
        Text::Owned(t) => ip.load_owned(t, &path),
        Text::Static(t) => ip.load_static(t, &path),
    };
    if let Ok(mut io) = sh.io.try_borrow_mut() {
        io.home.pop();
    }
    match r? {
        Some(v) => Ok(v),
        None => Ok(ip.no_value()),
    }
}

pub(crate) fn register<P: Platform + 'static>(interp: &mut Interp, shared: &Rc<Shared<P>>) {
    // (a build with neither files nor the library has nothing to load)
    #[cfg(any(feature = "files", feature = "lib"))]
    {
        let sh = shared.clone();
        interp.register(".l", WILDCARD, Rc::new(move |ip: &mut Interp, a: &[Value]| load(&sh, ip, a)));
    }
    #[cfg(not(any(feature = "files", feature = "lib")))]
    let _ = (interp, shared, WILDCARD);
}
