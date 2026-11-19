//! `kg lint FILE...`: the linter of `klong-lint`, with the options of `kgpy lint`.
use std::io::Write;
use std::process::ExitCode;

/// The library directories to search after KLONGPATH (colon-separated, `KLONG_LIB_DIRS`); none: the library built in.
pub fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(2).collect();
    let lib_dirs: Vec<String> = std::env::var("KLONG_LIB_DIRS").map(|v| v.split(':').filter(|d| !d.is_empty()).map(String::from).collect()).unwrap_or_default();
    // the parser recurses on the native stack: run on a thread with a stack of known size, as `kg FILE` does
    let h = std::thread::Builder::new().stack_size(256 << 20).spawn(move || {
        let out = std::io::stdout();
        let err = std::io::stderr();
        let (mut o, mut e) = (std::io::BufWriter::new(out.lock()), err.lock());
        let status = klong_lint::lint_main("kg lint", &args, lib_dirs, &mut o, &mut e);
        let _ = o.flush();
        status
    });
    match h.map(|h| h.join()) {
        Ok(Ok(status)) => ExitCode::from(status as u8),
        _ => ExitCode::from(70),
    }
}
