//! `--help`, `--version` of `klong-sim`, and that the help lists every option it accepts.
use std::process::Command;

fn run(args: &[&str]) -> (i32, String, String) {
    let o = Command::new(env!("CARGO_BIN_EXE_klong-sim")).args(args).output().expect("run klong-sim");
    (o.status.code().unwrap_or(-1), String::from_utf8_lossy(&o.stdout).into_owned(), String::from_utf8_lossy(&o.stderr).into_owned())
}

#[test]
fn help_version_and_unknown_option() {
    for flag in ["--help", "-h"] {
        let (code, out, err) = run(&[flag]);
        assert_eq!((code, err.as_str()), (0, ""), "{flag}");
        assert!(out.starts_with("usage: klong-sim "), "{out}");
    }
    let (code, out, _) = run(&["--version"]);
    assert_eq!((code, out), (0, format!("klong-sim {}\n", env!("CARGO_PKG_VERSION"))));
    let (code, out, err) = run(&["--bogus"]);
    assert_eq!((code, out.as_str()), (2, ""));
    assert!(err.contains("--timeout-ms") && err.contains("--help"), "{err}");
}

#[test]
fn the_help_lists_every_option_the_parser_accepts() {
    let (_, help, _) = run(&["--help"]);
    let mut found = 0;
    for line in include_str!("../src/main.rs").lines() {
        let t = line.trim_start();
        if !t.starts_with("\"-") || !t.contains("=>") {
            continue;
        }
        for token in t.split("=>").next().unwrap().split('|') {
            let name = token.trim().trim_matches('"');
            assert!(help.contains(name), "{name} is accepted but not in the help");
            found += 1;
        }
    }
    assert!(found >= 15, "{found}");
}
