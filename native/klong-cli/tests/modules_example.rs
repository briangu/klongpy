//! `examples/modules`: the program prints what EXPECTED.txt holds, and each program of `errors/` stops with the error it is named for.
use std::path::PathBuf;
use std::process::Command;

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../examples/modules")
}

fn run(file: &str) -> (i32, String, String) {
    let o = Command::new(env!("CARGO_BIN_EXE_kg")).arg(file).current_dir(dir()).env_remove("KLONGPATH").output().expect("run kg");
    (o.status.code().unwrap_or(-1), String::from_utf8_lossy(&o.stdout).into_owned(), String::from_utf8_lossy(&o.stderr).into_owned())
}

#[test]
fn the_backtest_prints_the_expected_text() {
    let want = std::fs::read_to_string(dir().join("EXPECTED.txt")).expect("EXPECTED.txt");
    let (code, out, err) = run("main.kg");
    assert_eq!((code, err.as_str()), (0, ""), "{out}");
    assert_eq!(out, want);
}

#[test]
fn each_error_program_stops_with_its_error() {
    // (file, kind, text the message must contain)
    let cases = [
        ("readonly", "type error", "read-only"),
        ("missing", "name error", "no entry \"nosuch\""),
        ("private", "name error", "no entry \"sq\""),
        ("cycle", "system error", "import cycle"),
        ("noexport", "system error", "has no .export"),
        ("undefined", "name error", "nosuch is not defined"),
        ("notadict", "type error", "not a dictionary"),
    ];
    for (name, kind, text) in cases {
        let (code, out, err) = run(&format!("errors/{name}.kg"));
        assert_eq!(code, 1, "{name}: {err}");
        assert_eq!(out, "", "{name}");
        assert!(err.contains(kind) && err.contains(text), "{name}: {err}");
    }
}

#[test]
fn a_classic_load_of_a_module_binds_its_name() {
    let (code, out, err) = run("classic_load.kg");
    assert_eq!((code, err.as_str()), (0, ""));
    assert_eq!(out, "9.0\n");
}
