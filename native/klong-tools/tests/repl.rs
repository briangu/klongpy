//! The REPL's reading of Klong text: completeness of an entry and tab completion (`klongpy/repl_input.py`).
use klong_tools::repl::{check_complete, complete, Completeness};
use Completeness::{Complete as C, Error as E, Incomplete as I};

/// The table of `tests/test_repl_complete.py`.
const CASES: &[(&str, Completeness)] = &[
    ("", C),
    ("   ", C),
    ("1+2", C),
    ("f(1;2)", C),
    ("\n", C),
    ("(", I),
    ("[", I),
    ("{", I),
    (":[", I),
    (":{", I),
    (":(", I),
    ("(1+2", I),
    ("arr::[1", I),
    ("f::{", I),
    ("f::{x+1", I),
    (":[1;2", I),
    ("{x}(", I),
    ("d:{[1 2]", I),
    ("f:(1", I),
    ("x:[1;2;", I),
    ("()", C),
    ("[]", C),
    ("{}", C),
    (":[1;2;3]", C),
    (":{[1 2]}", C),
    ("f::{x+1}", C),
    ("[1 2 3]", C),
    ("(1+2)", C),
    ("{x}(1)", C),
    (":[a;b;c]", C),
    ("f::{[a];a::(1+[2", I),
    ("f::{[a];a::(1+[2])}", C),
    ("{(}", E),
    ("{[}", E),
    ("f({x}(", I),
    (":[1;:[2;3]", I),
    (":[1;:[2;3]]", C),
    ("[[1 2][3", I),
    ("(1]", E),
    ("[1)", E),
    ("{1)", E),
    (")", E),
    ("]", E),
    ("}", E),
    ("1)", E),
    ("(1))", E),
    (":[1;2)", E),
    (":{1]", E),
    ("(1]\n", E),
    ("(}", E),
    ("[ ( ]", E),
    ("[ (", I),
    ("[1 (2) 3]", C),
    ("[1 [2 3] 4", I),
    ("\"abc", I),
    ("\"abc\"", C),
    ("\"a[b\"", C),
    ("\"a(b\"", C),
    ("\"a\"\"b\"", C),
    ("\"a\"\"", I),
    ("\"a{\"", C),
    ("f(\"(", I),
    ("f(\"(\")", C),
    ("\"", I),
    ("\"\"", C),
    ("\"a\nb", I),
    ("\"a\nb\"", C),
    ("0c[", C),
    ("0c(", C),
    ("0c{", C),
    ("0c]", C),
    ("[0c[ 0c]]", C),
    ("0c\"", C),
    ("x::0c[", C),
    (":\"a { b\"", C),
    (":\"a ( b\" 1", C),
    ("1 :\"x", I),
    ("1+2 :\"unfinished", I),
    (":\"", I),
    ("1 :\"x[\" ", C),
    ("(1 :\"ok ) bracket\"", I),
    ("(1 :\"ok ) bracket\")", C),
    ("f::{x :\"{\"", I),
    ("f::{x :\"{\"}", C),
    (".comment(\"X\")", I),
    (".comment(\"X\"); 1", I),
    (".comment(\"X\")\nbody (\nX", C),
    (".comment(\"X\")\nbody\nX\n1+2", C),
    (".comment(\"END\")\n[ ( {\n", I),
    (".comment(\"END\")\n[ ( {\nEND", C),
    (".comment(\"END\")\n[ ( {\nEND\n(1", I),
    ("1+.comment(\"X\")abc", I),
    (".comment(\"X\") abc X", C),
    ("1+", C),
    ("x@", C),
    ("f::", C),
    ("1+2*", C),
    ("-", C),
    ("#", C),
    ("a,", C),
    ("1 2 3+", C),
    ("1;", C),
    ("a::", C),
    ("+/", C),
    ("f::{x}\n", C),
    ("f::{\n", I),
    ("f::{\nx\n", I),
    ("f::{\nx\n}\n", C),
    ("f(\"abc", I),
    ("{\"abc", I),
];

#[test]
fn table() {
    let mut bad = Vec::new();
    for (text, want) in CASES {
        let got = check_complete(text, true);
        if got != *want {
            bad.push(format!("{:?}: {:?} not {:?}", text, got, want));
        }
    }
    assert!(bad.is_empty(), "{} of {}: {:#?}", bad.len(), CASES.len(), bad);
}

#[test]
fn tokens_only_cannot_see_comments() {
    assert_eq!(check_complete("1 :\"x", false), C);
    assert_eq!(check_complete("1 :\"x", true), I);
    for (t, w) in [("(1", I), ("(1]", E), ("\"abc", I), ("1+", C), ("", C)] {
        assert_eq!(check_complete(t, false), w, "{}", t);
    }
}

#[test]
fn joined_lines_become_complete() {
    assert_eq!(check_complete("f::{", true), I);
    assert_eq!(check_complete("f::{\n x+1", true), I);
    assert_eq!(check_complete("f::{\n x+1\n}", true), C);
}

#[test]
fn never_panics_on_noise() {
    // a small linear congruential generator: no dependency, the same inputs every run
    let pieces = ["(", ")", "[", "]", "{", "}", ":", "\"", "\\", " ", ";", "\n", "+", "-", "0c", ".", "a", "b", "1", ":[", ":{", ":(", ".comment(\"X\")"];
    let mut s: u64 = 7;
    for _ in 0..2000 {
        let mut text = String::new();
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let n = (s >> 33) % 15;
        for _ in 0..n {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            text.push_str(pieces[((s >> 33) as usize) % pieces.len()]);
        }
        let _ = check_complete(&text, true);
    }
}

fn names() -> Vec<String> {
    ["alpha", "alphabet", "beta", ".p", ".pc", ".py"].iter().map(|s| s.to_string()).collect()
}

fn cmds() -> Vec<String> {
    ["h", "q", "t", "l"].iter().map(|s| s.to_string()).collect()
}

fn words(line: &str) -> (usize, Vec<String>) {
    complete(line, &names(), &cmds(), "")
}

#[test]
fn names_and_commands() {
    let s = |v: &[&str]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>();
    assert_eq!(words("alp"), (0, s(&["alpha", "alphabet"])));
    assert_eq!(words("1+be"), (2, s(&["beta"])));
    assert_eq!(words("f(alph"), (2, s(&["alpha", "alphabet"])));
    assert_eq!(words("1+"), (2, vec![]));
    assert_eq!(words("zzz"), (0, vec![]));
    assert_eq!(words(".p").1, s(&[".p", ".pc", ".py"]));
    assert_eq!(words("]"), (0, s(&["]h", "]l", "]q", "]t"])));
    assert_eq!(words("]h"), (0, s(&["]h"])));
    assert_eq!(words("  ]t").0, 2);
    assert!(words("]h alp").1.contains(&"alpha".to_string()));
    // positions are in characters
    assert_eq!(words("日本 alp").0, 3);
}

#[test]
fn file_names() {
    let d = std::env::temp_dir().join(format!("klong-tools-repl-{}", std::process::id()));
    std::fs::create_dir_all(d.join("sub")).unwrap();
    for f in ["one.kg", "other.txt", ".hidden.kg"] {
        std::fs::write(d.join(f), "").unwrap();
    }
    let p = d.to_str().unwrap();
    let (start, w) = words(&format!("]l {}/o", p));
    assert_eq!(start, 3);
    assert_eq!(w, vec![format!("{}/one.kg", p), format!("{}/other.txt", p)]);
    assert_eq!(words(&format!(".l(\"{}/s", p)).1, vec![format!("{}/sub/", p)]);
    // hidden files only when the name written starts with a dot
    assert_eq!(words(&format!("]l {}/", p)).1.len(), 3);
    assert_eq!(words(&format!("]l {}/.", p)).1, vec![format!("{}/.hidden.kg", p)]);
    // a bare name also finds the .kg files of KLONGPATH
    let (_, w) = complete("]l on", &[], &[], p);
    assert_eq!(w, vec!["one.kg".to_string()]);
    std::fs::remove_dir_all(&d).unwrap();
}
