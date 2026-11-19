//! `kg compile c FILE...`, `kg fpga source FILE...`: the static compiler of `klong-static`, with the options of `kgpy compile c`
//! and `kgpy fpga source`.
use std::io::Write;
use std::process::ExitCode;

pub fn main(which: &str) -> ExitCode {
    let n = if which == "compile" { 3 } else { 3 };
    let args: Vec<String> = std::env::args().skip(n).collect();
    let which = which.to_string();
    // the parser recurses on the native stack: run on a thread with a stack of known size, as `kg FILE` does
    let h = std::thread::Builder::new().stack_size(256 << 20).spawn(move || {
        let out = std::io::stdout();
        let err = std::io::stderr();
        let (mut o, mut e) = (std::io::BufWriter::new(out.lock()), err.lock());
        let status = match which.as_str() {
            "compile" => klong_static::compile::compile_c_cli("kg compile c", &args, &mut o, &mut e),
            "fpga" => klong_static::emit_hls::fpga_source_cli("kg fpga source", &args, &mut o, &mut e),
            _ => 2,
        };
        let _ = o.flush();
        status
    });
    match h.map(|h| h.join()) {
        Ok(Ok(status)) => ExitCode::from(status as u8),
        _ => ExitCode::from(70),
    }
}
