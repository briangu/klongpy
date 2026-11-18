//! `.l("file")`: where the file is, and what it defines.
//!
//! The file is looked up as `.l` does (`x` and `x.kg` in each directory of KLONGPATH, default `.:lib`, taken from the
//! current directory), then in the library directories (klongpy/lib; for the standalone binary the copy built into it),
//! and last, for a relative name, in the directory of the linted file. The paths are strings, handled as Python's
//! `os.path` does, because they end up in nothing but lookups.
use crate::analysis::{Analysis, MAX_DEPTH};
use crate::tok::Source;
use std::collections::HashMap;
use std::rc::Rc;

/// The library of klongpy/lib, built into the binary.
pub static EMBEDDED_LIB: &[(&str, &str)] = &[
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
        Session { lib_dirs, cache: HashMap::new() }
    }

    /// The analysis of a file that is loaded by `.l` (names and arities only), or None.
    pub fn load_info(&mut self, path: &str, seen: &[String], depth: usize) -> Option<Rc<Analysis>> {
        let (real, mtime, size, text) = if let Some(rest) = path.strip_prefix(EMBEDDED_ROOT).and_then(|r| r.strip_prefix('/')) {
            let t = EMBEDDED_LIB.iter().find(|(n, _)| *n == rest)?.1;
            (path.to_string(), 0u128, t.len() as u64, Some(t.to_string()))
        } else {
            let real = std::fs::canonicalize(path).ok()?.to_string_lossy().into_owned();
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

    fn is_file(&self, p: &str) -> bool {
        if let Some(rest) = p.strip_prefix(EMBEDDED_ROOT).and_then(|r| r.strip_prefix('/')) {
            return EMBEDDED_LIB.iter().any(|(n, _)| *n == rest);
        }
        std::fs::metadata(p).map(|m| m.is_file()).unwrap_or(false)
    }
}

/// `os.path.join(a, b)`.
pub fn py_join(a: &str, b: &str) -> String {
    if b.starts_with('/') || a.is_empty() {
        return b.to_string();
    }
    if a.ends_with('/') {
        format!("{}{}", a, b)
    } else {
        format!("{}/{}", a, b)
    }
}

/// `os.path.dirname(p)`.
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

/// The path `.l(x)` would read, or None.
pub fn find_load(sess: &Session, x: &str, home: &str) -> Option<String> {
    if x.is_empty() {
        return None;
    }
    let klongpath = std::env::var("KLONGPATH").ok().filter(|s| !s.is_empty()).unwrap_or_else(|| ".:lib".to_string());
    let mut dirs: Vec<String> = klongpath.split(':').filter(|d| !d.is_empty()).map(String::from).collect();
    let extra: Vec<String> = if sess.lib_dirs.is_empty() { vec![EMBEDDED_ROOT.to_string()] } else { sess.lib_dirs.clone() };
    for d in extra {
        if !dirs.contains(&d) {
            dirs.push(d);
        }
    }
    let look = |base: &str| -> Option<String> {
        let cand = py_join(base, x);
        if sess.is_file(&cand) {
            return Some(cand);
        }
        if !cand.ends_with(".kg") {
            let c2 = format!("{}.kg", cand);
            if sess.is_file(&c2) {
                return Some(c2);
            }
        }
        None
    };
    for d in &dirs {
        if let Some(r) = look(d) {
            return Some(r);
        }
    }
    if !home.is_empty() && !x.starts_with('/') {
        return look(home);
    }
    None
}
