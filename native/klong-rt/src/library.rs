//! The pure-Klong library of `klongpy/lib`, as text in the image. `.l("math")` finds a file of
//! that name first; the built-in copy is what a board without storage, or without the file,
//! gets. Each is parsed when it is loaded, not at start-up.
//!
//! Which files are here, and why the others are not, is recorded in ticket 103: a file is
//! included when it loads and works with what the native runtime has.

/// (name, text)
pub static FILES: &[(&str, &str)] = &[
    ("math", include_str!("../../../klongpy/lib/math.kg")),
    ("nstat", include_str!("../../../klongpy/lib/nstat.kg")),
    ("set", include_str!("../../../klongpy/lib/set.kg")),
    ("util", include_str!("../../../klongpy/lib/util.kg")),
    ("csv", include_str!("../../../klongpy/lib/csv.kg")),
    ("time", include_str!("../../../klongpy/lib/time.kg")),
    ("print", include_str!("../../../klongpy/lib/print.kg")),
    ("huffman", include_str!("../../../klongpy/lib/huffman.kg")),
    ("spline", include_str!("../../../klongpy/lib/spline.kg")),
    ("eigenv", include_str!("../../../klongpy/lib/eigenv.kg")),
    ("help", include_str!("../../../klongpy/lib/help.kg")),
    ("edt", include_str!("../../../klongpy/lib/edt.kg")),
];

/// The library file `.l(name)` means: the last path component, without `.kg`.
/// Returns the name to report in errors and the text.
pub fn find(name: &str) -> Option<(&'static str, &'static str)> {
    let base = name.rsplit('/').next().unwrap_or(name);
    let base = base.strip_suffix(".kg").unwrap_or(base);
    FILES.iter().find(|(n, _)| *n == base).map(|(n, t)| (*n, *t))
}
