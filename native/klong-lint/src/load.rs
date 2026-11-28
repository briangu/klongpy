//! `.l("file")`: where the file is, and what it defines.
//!
//! A bare name is looked up beside the linted file, then on KLONGPATH, and finally in the library.
use crate::analysis::{Analysis, MAX_DEPTH};
use crate::tok::Source;
use std::collections::HashMap;
use std::rc::Rc;

/// The library of klongpy/lib, built into the binary.
pub static EMBEDDED_LIB: &[(&str, &str)] = &[
    ("checks.kg", include_str!("../../../klongpy/lib/checks.kg")),
    ("csv.kg", include_str!("../../../klongpy/lib/csv.kg")),
    ("edt.kg", include_str!("../../../klongpy/lib/edt.kg")),
    ("eigenv.kg", include_str!("../../../klongpy/lib/eigenv.kg")),
    ("help.kg", include_str!("../../../klongpy/lib/help.kg")),
    ("huffman.kg", include_str!("../../../klongpy/lib/huffman.kg")),
    ("math.kg", include_str!("../../../klongpy/lib/math.kg")),
    ("mgmt.kg", include_str!("../../../klongpy/lib/mgmt.kg")),
    ("nstat.kg", include_str!("../../../klongpy/lib/nstat.kg")),
    ("print.kg", include_str!("../../../klongpy/lib/print.kg")),
    ("set.kg", include_str!("../../../klongpy/lib/set.kg")),
    ("spline.kg", include_str!("../../../klongpy/lib/spline.kg")),
    ("time.kg", include_str!("../../../klongpy/lib/time.kg")),
    ("util.kg", include_str!("../../../klongpy/lib/util.kg")),
];

const EMBEDDED_ROOT: &str = "<klong-lib>";

/// What a run of the linter shares: the options and the files already read.
pub struct Session {
    /// Library directories searched after KLONGPATH. Empty: the library built into the binary.
    pub lib_dirs: Vec<String>,
    cache: HashMap<(String, u128, u64), Rc<Analysis>>,
}

impl Session {
    pub fn new(lib_dirs: Vec<String>) -> Session {
        let lib_dirs = lib_dirs.into_iter().map(|dir| klong_host::module_path(&dir).into_owned()).collect();
        Session { lib_dirs, cache: HashMap::new() }
    }

    /// The analysis of a file that is loaded by `.l` (names and arities only), or None.
    pub fn load_info(&mut self, path: &str, seen: &[String], depth: usize) -> Option<Rc<Analysis>> {
        let (real, mtime, size, text) = if let Some(rest) = path.strip_prefix(EMBEDDED_ROOT).and_then(|r| r.strip_prefix('/')) {
            let t = EMBEDDED_LIB.iter().find(|(n, _)| *n == rest)?.1;
            (path.to_string(), 0u128, t.len() as u64, Some(t.to_string()))
        } else {
            let real = std::fs::canonicalize(klong_host::os_path(path).as_ref()).ok()?.to_string_lossy().into_owned();
            let md = std::fs::metadata(&real).ok()?;
            let mt = md.modified().ok().and_then(|m| m.duration_since(std::time::UNIX_EPOCH).ok()).map(|d| d.as_nanos()).unwrap_or(0);
            (real, mt, md.len(), None)
        };
        if seen.iter().any(|s| *s == real) || depth > MAX_DEPTH {
            return None;
        }
        let key = (real.clone(), mtime, size);
        if let Some(hit) = self.cache.get(&key) {
            return Some(hit.clone());
        }
        let text = match text {
            Some(t) => t,
            None => crate::read_text(&real).ok()?,
        };
        let mut seen2: Vec<String> = seen.to_vec();
        seen2.push(real);
        let base = py_dirname(path);
        let an = Analysis::new(Source::new(path, text), &base, true, &seen2, depth, false, self);
        let an = Rc::new(an);
        if self.cache.len() > 64 {
            self.cache.clear();
        }
        self.cache.insert(key, an.clone());
        Some(an)
    }

    /// The text of a file `find_load` found: the library built into the binary, or a file read as Python reads one.
    pub fn read_loaded(&self, path: &str) -> Result<String, String> {
        if let Some(rest) = path.strip_prefix(EMBEDDED_ROOT).and_then(|r| r.strip_prefix('/')) {
            return EMBEDDED_LIB.iter().find(|(n, _)| *n == rest).map(|(_, t)| t.to_string()).ok_or_else(|| "No such file or directory".to_string());
        }
        crate::read_text(path)
    }

    fn is_file(&self, p: &str) -> bool {
        if let Some(rest) = p.strip_prefix(EMBEDDED_ROOT).and_then(|r| r.strip_prefix('/')) {
            return EMBEDDED_LIB.iter().any(|(n, _)| *n == rest);
        }
        std::fs::metadata(klong_host::os_path(p).as_ref()).map(|m| m.is_file()).unwrap_or(false)
    }
}

/// `os.path.join(a, b)`.
pub fn py_join(a: &str, b: &str) -> String {
    if klong_core::module::is_absolute(b) || a.is_empty() {
        return b.to_string();
    }
    if a.ends_with('/') {
        format!("{}{}", a, b)
    } else {
        format!("{}/{}", a, b)
    }
}

/// `os.path.dirname(p)`.
#[cfg(not(windows))]
pub fn py_dirname(p: &str) -> String {
    let i = match p.rfind('/') {
        Some(i) => i + 1,
        None => return String::new(),
    };
    let head = &p[..i];
    if !head.is_empty() && head.bytes().any(|b| b != b'/') {
        head.trim_end_matches('/').to_string()
    } else {
        head.to_string()
    }
}

#[cfg(windows)]
pub fn py_dirname(p: &str) -> String {
    klong_core::module::dir_of(&klong_host::module_path(p))
}

/// Where `.l(x)` finds a file: the path, and whether it shadows a file of the bundled library.
pub struct Located {
    pub path: String,
    pub shadows: bool,
}

/// The directories of KLONGPATH as `.l` reads them.
fn klongpath() -> Vec<String> {
    klong_host::parse_load_path(&klong_host::load_path_env()).collect()
}

impl Session {
    /// The path of the bundled library file called `name`, when there is one: in the library directories, or the copy built into the binary.
    fn library_path(&self, name: &str) -> Option<String> {
        let file = format!("{}.kg", name);
        if self.lib_dirs.is_empty() {
            let p = py_join(EMBEDDED_ROOT, &file);
            return self.is_file(&p).then_some(p);
        }
        self.lib_dirs.iter().map(|d| py_join(d, &file)).find(|p| self.is_file(p))
    }
}

/// The file `.l(x)` reads, by the order the interpreter and the checker share (`klong_core::module::search`): `./x` and `../x` from the
/// directory of the linted file, an absolute path as it is, any other name in that directory, then in KLONGPATH, then in the library.
/// `home` is the directory of the linted file (empty: the working directory).
pub fn find(sess: &Session, x: &str, home: &str) -> Option<Located> {
    if x.is_empty() {
        return None;
    }
    let x = klong_host::module_path(x);
    let home = klong_host::module_path(home);
    let from_lib = home == EMBEDDED_ROOT || sess.lib_dirs.iter().any(|d| d == home.as_ref());
    let kp = klongpath();
    let env = klong_core::module::Env { dir: &home, from_lib, klongpath: &kp };
    let mut probe = |p: &str| sess.is_file(p);
    let mut in_lib = |n: &str| sess.library_path(n).is_some();
    let mut tried = Vec::new();
    let found = klong_core::module::search(&env, &x, &mut probe, &mut in_lib, &mut tried)?;
    match found.place {
        klong_core::module::Place::File(path) => {
            let mut shadows = found.shadows;
            if shadows {
                // the same file reached through KLONGPATH is not a shadow
                let lib = sess.library_path(klong_core::module::stem(&x));
                let same = match lib {
                    Some(l) => matches!((sess.read_loaded(&l), sess.read_loaded(&path)), (Ok(a), Ok(b)) if a == b),
                    None => true,
                };
                if same {
                    shadows = false;
                }
            }
            Some(Located { path, shadows })
        }
        klong_core::module::Place::Library(name) => Some(Located { path: sess.library_path(&name)?, shadows: false }),
    }
}

/// The path `.l(x)` would read, or None.
pub fn find_load(sess: &Session, x: &str, home: &str) -> Option<String> {
    find(sess, x, home).map(|l| l.path)
}
