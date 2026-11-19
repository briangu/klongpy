//! `kg ast`: the golden parse trees of the repository (`tests/golden`), and the command line.
use klong_tools::ast::{ast_main, ast_text};
use std::path::{Path, PathBuf};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn kg_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let mut entries: Vec<_> = std::fs::read_dir(dir).unwrap().flatten().map(|e| e.path()).collect();
    entries.sort();
    for p in entries {
        if p.is_dir() {
            kg_files(&p, out);
        } else if p.extension().map(|e| e == "kg").unwrap_or(false) {
            out.push(p);
        }
    }
}

/// Every (group, name, source, golden) of tests/golden/regen.py.
fn cases() -> Vec<(String, PathBuf, PathBuf)> {
    let r = root();
    let groups = [("language", r.join("tests/kgtests/language")), ("lib", r.join("klongpy/lib")), ("fixtures", r.join("tests/golden/fixtures"))];
    let mut out = Vec::new();
    for (g, base) in groups {
        let mut files = Vec::new();
        kg_files(&base, &mut files);
        for p in files {
            let rel = p.strip_prefix(&base).unwrap().with_extension("");
            let name = rel.to_string_lossy().replace('/', "__");
            out.push((format!("{}/{}", g, name), p, r.join("tests/golden").join(g).join(format!("{}.sexp", name))));
        }
    }
    out
}

#[test]
fn golden_trees() {
    let cs = cases();
    assert!(cs.len() > 100, "found {} programs", cs.len());
    let mut bad = Vec::new();
    for (name, src, golden) in &cs {
        let text = std::fs::read_to_string(src).unwrap();
        let want = std::fs::read_to_string(golden).unwrap();
        let got = ast_text(&text, "x");
        let ok = match (&got, want.starts_with("(error ")) {
            (Ok(g), false) => *g == want,
            (Err(_), true) => true,
            _ => false,
        };
        if !ok {
            bad.push(name.clone());
        }
    }
    assert!(bad.is_empty(), "{} of {} differ: {:?}", bad.len(), cs.len(), bad);
}

fn run(args: &[&str], stdin: &str) -> (i32, String, String) {
    let args: Vec<String> = args.iter().map(|s| s.to_string()).collect();
    let (mut out, mut err) = (Vec::new(), Vec::new());
    let st = ast_main(&args, &mut stdin.as_bytes(), &mut out, &mut err);
    (st, String::from_utf8(out).unwrap(), String::from_utf8(err).unwrap())
}

#[test]
fn expression_file_and_stdin() {
    let (st, out, _) = run(&["-e", "1+2"], "");
    assert_eq!((st, out.as_str()), (0, "(d \"+\" (num 1) (num 2))\n"));
    let (st, out, _) = run(&["-"], "a::1\n[1 2]\n");
    assert_eq!(st, 0);
    assert_eq!(out, "(def (sym a) (num 1))\n(list (num 1) (num 2))\n");
    let (st, out, err) = run(&["-e", "1+(2"], "");
    assert_eq!((st, out.as_str()), (1, ""));
    assert!(err.contains("syntax error: expected ')'"), "{}", err);
}

#[test]
fn usage_errors() {
    assert_eq!(run(&[], "").0, 2);
    assert_eq!(run(&["--bogus"], "").0, 2);
    assert_eq!(run(&["a", "-e", "1"], "").0, 2);
    let (st, out, _) = run(&["-h"], "");
    assert_eq!(st, 0);
    assert!(out.starts_with("usage: kg ast"));
    let (st, _, err) = run(&["/nonexistent/x.kg"], "");
    assert_eq!(st, 1);
    assert!(err.contains("No such file"), "{}", err);
}

#[test]
fn files_read_as_python_text_mode_does() {
    let d = std::env::temp_dir().join(format!("klong-tools-ast-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    let f = d.join("crlf.kg");
    std::fs::write(&f, "a::1\r\nb::\"x\r\ny\"\rc::2").unwrap();
    let (st, out, _) = run(&[f.to_str().unwrap()], "");
    assert_eq!(st, 0);
    assert_eq!(out, "(def (sym a) (num 1))\n(def (sym b) (str \"x\\ny\"))\n(def (sym c) (num 2))\n");
    std::fs::write(&f, [b'a', 0xff]).unwrap();
    let (st, _, err) = run(&[f.to_str().unwrap()], "");
    assert_eq!(st, 1);
    assert!(err.contains("codec can't decode"), "{}", err);
    std::fs::remove_dir_all(&d).unwrap();
}

#[test]
fn big_integers_and_modules() {
    assert_eq!(ast_text("18446744073709551616", "x").unwrap(), "(num 18446744073709551616)\n");
    assert_eq!(ast_text("-9223372036854775808", "x").unwrap(), "(m \"-\" (num 9223372036854775808))\n");
    // `.module()` with no argument is a call, not a fault
    assert_eq!(ast_text(".module()", "x").unwrap(), "(call (sym .module))\n");
}
