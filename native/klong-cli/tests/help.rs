//! `--help`, `--version`, and an unknown option; and that the help lists every option the
//! binary's argument parser accepts.
use std::process::Command;

fn run(args: &[&str]) -> (i32, String, String) {
    let o = Command::new(env!("CARGO_BIN_EXE_kg")).args(args).output().expect("run kg");
    (o.status.code().unwrap_or(-1), String::from_utf8_lossy(&o.stdout).into_owned(), String::from_utf8_lossy(&o.stderr).into_owned())
}

#[test]
fn help_and_version() {
    for flag in ["--help", "-h"] {
        let (code, out, err) = run(&[flag]);
        assert_eq!((code, err.as_str()), (0, ""), "{flag}");
        assert!(out.starts_with("usage: kg "), "{out}");
    }
    let (code, out, _) = run(&["--version"]);
    assert_eq!(code, 0);
    assert_eq!(out, format!("kg {}\n", env!("CARGO_PKG_VERSION")));
}

#[test]
fn an_unknown_option_prints_the_full_usage_and_exits_2() {
    let (code, out, err) = run(&["--bogus"]);
    assert_eq!(code, 2);
    assert_eq!(out, "");
    assert!(err.starts_with("usage: kg ") && err.contains("--fail-every") && err.contains("--help"), "{err}");
    // the same text as `--help`, on standard error
    let (_, help, _) = run(&["--help"]);
    assert_eq!(err, help);
}

#[test]
fn the_help_lists_every_option_the_parser_accepts() {
    let (_, help, _) = run(&["--help"]);
    let source = include_str!("../src/main.rs");
    let mut found = 0;
    for line in source.lines() {
        let t = line.trim_start();
        if !t.starts_with("\"-") || !t.contains("=>") {
            continue;
        }
        for token in t.split("=>").next().unwrap().split('|') {
            let name = token.trim().split(" if ").next().unwrap().trim().trim_matches('"');
            assert!(help.contains(name), "{name} is accepted but not in the help");
            found += 1;
        }
    }
    assert!(found >= 15, "{found}");
}
