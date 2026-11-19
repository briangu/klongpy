//! `kg check FILE...`: the static profile checker of `klong-static`, with the options of `kgpy check`.
use std::io::Write;
use std::process::ExitCode;

pub fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(2).collect();
    // the parser recurses on the native stack: run on a thread with a stack of known size, as `kg FILE` does
    let h = std::thread::Builder::new().stack_size(256 << 20).spawn(move || {
        let out = std::io::stdout();
        let err = std::io::stderr();
        let (mut o, mut e) = (std::io::BufWriter::new(out.lock()), err.lock());
        let status = klong_static::check_main("kg check", &args, &mut o, &mut e);
        let _ = o.flush();
        status
    });
    match h.map(|h| h.join()) {
        Ok(Ok(status)) => ExitCode::from(status as u8),
        _ => ExitCode::from(70),
    }
}
