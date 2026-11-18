//! `klong profile-doc [--check|--write] [PATH]`: the generator of docs/static-profile.md, of `klong-static`
//! (the Python is a module: `python -m klongpy.static.profile_doc`).
use std::io::Write;
use std::process::ExitCode;

pub fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(2).collect();
    let out = std::io::stdout();
    let err = std::io::stderr();
    let (mut o, mut e) = (std::io::BufWriter::new(out.lock()), err.lock());
    let status = klong_static::profile_doc::main(&args, &mut o, &mut e);
    let _ = o.flush();
    ExitCode::from(status as u8)
}
