//! The old name of the binary, `klong`: the same program as `kg`, built from the same source (main.rs) until the programs that call it
//! by that name call `kg`.
#[path = "main.rs"]
mod real;

fn main() -> std::process::ExitCode {
    real::main()
}
