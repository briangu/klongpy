//! The rule table of `tests/test_lint.py` (data in data/cases.rs), and the other classes of that file that do not
//! need the Python interpreter: suppression, select and ignore, cross-file names, the shape of the output, robustness.
#[path = "data/cases.rs"]
mod cases;

use klong_lint::{lint_files, lint_main, lint_source, parse_codes, Diagnostic, Options, RULES};
use std::collections::BTreeSet;
use std::path::PathBuf;

fn lint(text: &str) -> Vec<Diagnostic> {
    lint_source("t.kg", text, &Options::default())
}

fn found(text: &str, code: &str) -> Vec<(usize, usize)> {
    lint(text).iter().filter(|d| d.code == code).map(|d| (d.line, d.col)).collect()
}

fn codes(text: &str) -> Vec<&'static str> {
    lint(text).iter().map(|d| d.code).collect()
}

fn set(v: &[&str]) -> BTreeSet<String> {
    v.iter().map(|s| s.to_string()).collect()
}

#[test]
fn every_code_has_a_positive_and_a_negative_case() {
    for r in RULES {
        assert!(cases::CASES.iter().any(|c| c.0 == r.code && !c.2.is_empty()), "{}: no positive case", r.code);
        assert!(cases::CASES.iter().any(|c| c.0 == r.code && c.2.is_empty()), "{}: no negative case", r.code);
    }
    assert!(cases::CASES.iter().all(|c| RULES.iter().any(|r| r.code == c.0)));
}

#[test]
fn the_rule_table() {
    let mut bad = Vec::new();
    for (code, text, want) in cases::CASES {
        let got = found(text, code);
        if got != *want {
            bad.push(format!("{} {:?}: want {:?}, got {:?}", code, text, want, got));
        }
    }
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}

#[test]
fn messages_and_hints_are_one_line_and_severities_match() {
    for (code, text, want) in cases::CASES {
        if want.is_empty() {
            continue;
        }
        let opts = Options { select: Some(set(&[code])), ..Options::default() };
        for d in lint_source("t.kg", text, &opts) {
            assert!(!d.message.contains('\n'), "{:?}", d);
            assert!(d.hint.as_deref().map(|h| !h.contains('\n')).unwrap_or(true));
            assert_eq!(d.severity, RULES.iter().find(|r| r.code == *code).unwrap().severity);
        }
    }
}

#[test]
fn the_observed_inputs_give_the_codes_they_name() {
    for (text, code) in [
        ("my_var::5", "W001"), ("-1+2", "W010"), ("f::{y};f(5)", "W011"), ("a::1;[a b]", "W012"), ("f (1)", "W014"),
        ("{x} '[1 2]", "W014"), ("{x :\"c\"\n+2}(1)", "W015"), (":123", "W016"), ("[1;2]", "W017"), ("[1 + 2]", "W017"),
        ("10c", "W018"), ("{x::5;x}(1)", "W019"),
    ] {
        assert!(codes(text).contains(&code), "{:?} should give {}", text, code);
    }
}

#[test]
fn a_clean_program_has_no_findings() {
    let text = "sq::{x*x}\nsum::{[a];a::0;{a::a+x}'x;a}\nr::sum([1 2 3])\ns::sq(r)\n:[r>1;s;0]\n";
    assert!(lint(text).is_empty());
}

#[test]
fn a_syntax_error_does_not_stop_the_token_rules() {
    let c = codes("my_var::5\n1+\nx::10c\n");
    for want in ["E001", "W001", "W018"] {
        assert!(c.contains(&want), "{:?}", c);
    }
}

#[test]
fn parsing_resumes_after_an_error() {
    assert_eq!(found("a::1\nf (1)\ng (2)\n1+\n", "E001"), vec![(2, 3), (3, 3), (4, 2)]);
}

#[test]
fn diagnostic_fields() {
    let opts = Options { select: Some(set(&["W010"])), ..Options::default() };
    let ds = lint_source("dir/x.kg", "a::1\nb::-1+2", &opts);
    assert_eq!(ds.len(), 1);
    let d = &ds[0];
    assert_eq!((d.code, d.severity, d.line, d.col), ("W010", "warning", 2, 4));
    assert_eq!(d.span, (8, 11));
    assert_eq!(d.file, "dir/x.kg");
    assert!(d.hint.is_some());
    assert!(d.to_string().starts_with("dir/x.kg:2:4: W010 "));
}

#[test]
fn columns_count_characters_not_bytes() {
    // é and → are 2 and 3 bytes
    let ds = lint_source("t.kg", "é::1;→::2;my_var::5\n\"é→\" 10c", &Options::default());
    let w001 = ds.iter().find(|d| d.code == "W001").unwrap();
    assert_eq!((w001.line, w001.col), (1, 11));
    let w018 = ds.iter().find(|d| d.code == "W018").unwrap();
    assert_eq!((w018.line, w018.col), (2, 6));
    // a span is in bytes
    assert_eq!(w001.span, (13, 19));
}

#[test]
fn sorted_by_position() {
    let ds = lint("[1;2]\nmy_var::5\n-1+2\n");
    let pos: Vec<usize> = ds.iter().map(|d| d.span.0).collect();
    let mut s = pos.clone();
    s.sort();
    assert_eq!(pos, s);
}

#[test]
fn empty_and_comment_only() {
    assert!(lint("").is_empty());
    assert!(lint(":\" nothing here \"\n").is_empty());
}

#[test]
fn comment_directive_text_is_skipped() {
    assert!(lint(".comment(\"END\")\nmy_var::5 [1;2]\nEND\na::1\n").is_empty());
}

// -- suppression ---------------------------------------------------------------------------------------------

#[test]
fn ignore_comment_removes_that_code_on_that_line_only() {
    assert_eq!(found("a::1;[a b] :\"lint:ignore W012\"\nb::1;[b a]\n", "W012"), vec![(2, 7), (2, 9)]);
}

#[test]
fn other_codes_on_the_line_stay() {
    assert_eq!(codes("a::1;[a;b] :\"lint:ignore W012\"\n"), vec!["W017"]);
    assert_eq!(codes("a::1;[a;b] :\"lint:ignore W017\"\n"), vec!["W012"]);
}

#[test]
fn several_codes() {
    assert!(codes("a::1;[a;b] :\"lint:ignore W012 W017\"\n").is_empty());
    assert!(codes("a::1;[a;b] :\"lint:ignore W012, W017\"\n").is_empty());
}

#[test]
fn the_comment_does_not_reach_the_next_line() {
    assert_eq!(codes(":\"lint:ignore W016\"\n:123\n"), vec!["W016"]);
}

#[test]
fn errors_can_be_suppressed() {
    assert!(codes("1+ :\"lint:ignore E001\"\n").is_empty());
}

#[test]
fn a_comment_that_merely_mentions_it_does_nothing() {
    assert_eq!(codes(":\"lint:ignore\"\n:123\n"), vec!["W016"]);
    assert_eq!(codes(":\"remember to lint:ignore W016 someday\"\n:123\n"), vec!["W016"]);
}

#[test]
fn inside_a_function_body_and_after_a_multiline_statement() {
    assert!(found("f::{[a];a::1; :123 :\"lint:ignore W016\"\na}\n", "W016").is_empty());
}

// -- select and ignore -----------------------------------------------------------------------------------------

const TEXT: &str = "my_var::5\n[1;2]\n-1+2\n";

fn with(select: Option<&[&str]>, ignore: &[&str]) -> Vec<&'static str> {
    let opts = Options { select: select.map(set), ignore: set(ignore), ..Options::default() };
    lint_source("t.kg", TEXT, &opts).iter().map(|d| d.code).collect()
}

#[test]
fn select_and_ignore() {
    assert_eq!(with(Some(&["W001"]), &[]), vec!["W001"]);
    let mut v = with(Some(&["W001", "W010"]), &[]);
    v.sort();
    assert_eq!(v, vec!["W001", "W010"]);
    assert_eq!(with(Some(&["W017"]), &[]), vec!["W017"]);
    assert!(!with(None, &["W017"]).contains(&"W017"));
    assert!(with(None, &["W017"]).contains(&"W001"));
    assert!(with(None, &["W001", "W002", "W010", "W017"]).is_empty());
    assert!(with(Some(&["W001"]), &["W001"]).is_empty());
    // an empty selection selects nothing
    assert!(with(Some(&[]), &[]).is_empty());
}

#[test]
fn parse_codes_rules() {
    assert!(parse_codes("W001,nope").is_err());
    assert!(parse_codes("W999").unwrap_err().starts_with("unknown lint code 'W999'; codes are W001, W002, W003, W010"));
    assert_eq!(parse_codes(" w001 , e002 ").unwrap(), set(&["W001", "E002"]));
}

// -- cross-file names ------------------------------------------------------------------------------------------

struct Dir(PathBuf);

impl Dir {
    fn new(tag: &str) -> Dir {
        let p = std::env::temp_dir().join(format!("klong-lint-test-{}-{}", std::process::id(), tag));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        Dir(p)
    }
    fn write(&self, name: &str, text: &str) {
        let p = self.0.join(name);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, text).unwrap();
    }
    fn lint(&self, name: &str) -> Vec<(&'static str, usize, usize)> {
        self.lint_with(name, &Options::default())
    }
    fn lint_with(&self, name: &str, opts: &Options) -> Vec<(&'static str, usize, usize)> {
        let p = self.0.join(name).to_string_lossy().into_owned();
        let r = lint_files(&[p], opts).pop().unwrap();
        r.result.unwrap().iter().map(|d| (d.code, d.line, d.col)).collect()
    }
}

impl Drop for Dir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn names_of_a_loaded_file_are_defined() {
    let d = Dir::new("a");
    d.write("lib.kg", "helper::{x+1}\nconst::5\n");
    d.write("main.kg", ".l(\"lib.kg\")\nhelper(const)\n");
    assert!(d.lint("main.kg").is_empty());
}

#[test]
fn the_extension_is_optional_and_subdirectories_work() {
    let d = Dir::new("b");
    d.write("sub/lib.kg", "helper::{x+1}\n");
    d.write("main.kg", ".l(\"sub/lib\")\nhelper(1)\n");
    assert!(d.lint("main.kg").is_empty());
}

#[test]
fn a_use_before_the_load_is_flagged() {
    let d = Dir::new("c");
    d.write("lib.kg", "helper::{x+1}\n");
    d.write("main.kg", "helper(1)\n.l(\"lib.kg\")\n");
    assert_eq!(d.lint("main.kg"), vec![("W002", 1, 1)]);
}

#[test]
fn a_name_the_loaded_file_does_not_define_is_still_flagged() {
    let d = Dir::new("d");
    d.write("lib.kg", "helper::{x+1}\n");
    d.write("main.kg", ".l(\"lib.kg\")\nhelper(other)\n");
    assert_eq!(d.lint("main.kg"), vec![("W002", 2, 8)]);
}

#[test]
fn loads_are_followed_and_cycles_terminate() {
    let d = Dir::new("e");
    d.write("a.kg", ".l(\"b.kg\")\nfa::{x}\n");
    d.write("b.kg", "fb::{x}\n");
    d.write("main.kg", ".l(\"a.kg\")\nfa(fb(1))\n");
    assert!(d.lint("main.kg").is_empty());
    d.write("c.kg", ".l(\"d.kg\")\nfc::{x}\n");
    d.write("d.kg", ".l(\"c.kg\")\nfd::{x}\n");
    assert!(d.lint("c.kg").is_empty());
}

#[test]
fn arity_of_a_function_from_a_loaded_file() {
    let d = Dir::new("f");
    d.write("lib.kg", "helper::{x+1}\n");
    d.write("main.kg", ".l(\"lib.kg\")\nhelper(1;2)\n");
    assert_eq!(d.lint("main.kg"), vec![("E002", 2, 1)]);
}

#[test]
fn unknown_loads_turn_w002_off() {
    let d = Dir::new("g");
    d.write("main.kg", "n::\"lib.kg\"\n.l(n)\nunknown(1)\n");
    assert!(d.lint("main.kg").is_empty());
    d.write("m2.kg", ".l(\"nowhere.kg\")\nunknown(1)\n");
    assert!(d.lint("m2.kg").is_empty());
    for imp in [".py(\"m\")", ".pyf(\"m\";\"f\")", ".pya(x;\"a\")"] {
        d.write("m3.kg", &format!("{}\nunknown(1)\n", imp));
        assert!(d.lint("m3.kg").is_empty(), "{}", imp);
    }
}

#[test]
fn no_resolve_never_reads_files() {
    let d = Dir::new("h");
    d.write("lib.kg", "helper::{x+1}\n");
    d.write("main.kg", ".l(\"lib.kg\")\nunknown(1)\n");
    assert!(d.lint_with("main.kg", &Options { no_resolve: true, ..Options::default() }).is_empty());
}

#[test]
fn the_library_is_found() {
    // klongpy/lib/csv.kg defines csv.make (a dotted name) and csv.tbl
    let d = Dir::new("i");
    d.write("main.kg", ".l(\"csv\")\ncsv.make(1)\ncsv.nosuch(2)\n");
    assert_eq!(d.lint("main.kg"), vec![("W002", 3, 1)]);
    // and from a directory, as the extension passes it
    let lib = format!("{}/../../klongpy/lib", env!("CARGO_MANIFEST_DIR"));
    assert_eq!(d.lint_with("main.kg", &Options { lib_dirs: vec![lib], ..Options::default() }), vec![("W002", 3, 1)]);
}

#[test]
fn a_syntax_error_in_the_loaded_file_is_not_reported_in_the_loader() {
    let d = Dir::new("j");
    d.write("lib.kg", "helper::{x+1}\n1+\n");
    d.write("main.kg", ".l(\"lib.kg\")\nhelper(1)\n");
    assert!(d.lint("main.kg").is_empty());
}

#[test]
fn crlf_and_cr_are_newlines() {
    let d = Dir::new("k");
    d.write("a.kg", "a::1\r\nmy_var::5\rb::2\n[1;2]\n");
    assert_eq!(d.lint("a.kg"), vec![("W001", 2, 1), ("W017", 4, 3)]);
}

// -- the command line -----------------------------------------------------------------------------------------

fn run(args: &[&str]) -> (i32, String, String) {
    let args: Vec<String> = args.iter().map(|s| s.to_string()).collect();
    let (mut o, mut e) = (Vec::new(), Vec::new());
    let rc = lint_main("kgpy lint", &args, Vec::new(), &mut o, &mut e);
    (rc, String::from_utf8(o).unwrap(), String::from_utf8(e).unwrap())
}

#[test]
fn format_and_exit_status() {
    let d = Dir::new("l");
    d.write("w.kg", "a::1\nmy_var::5\n[1;2]\n");
    let p = d.0.join("w.kg").to_string_lossy().into_owned();
    let (rc, out, err) = run(&[&p]);
    assert_eq!(rc, 0);
    assert_eq!(err, "");
    assert_eq!(out, format!("{p}:2:1: W001 'my_var' reads as my _ var: _ is the Drop operator, names cannot contain underscores\n{p}:3:3: W017 ';' inside a list is the string \";\"\n0 errors, 2 warnings in 1 file\n"));
    d.write("e.kg", "f::{x}\nf(1;2)\n");
    let p2 = d.0.join("e.kg").to_string_lossy().into_owned();
    let (rc, out, _) = run(&[&p2, "-v"]);
    assert_eq!(rc, 1);
    assert!(out.contains(&format!("{p2}:2:1: E002 f takes 1 argument but is called with 2")));
    assert!(out.contains("\n  hint: an arity error at strict level 2\n"));
    assert!(out.ends_with("1 error, 0 warnings in 1 file\n"));
    let (rc, out, err) = run(&[&p2, &p, "--ignore", "E002"]);
    assert_eq!((rc, err.as_str()), (0, ""));
    assert!(out.ends_with("0 errors, 2 warnings in 2 files\n"));
}

#[test]
fn unreadable_files_and_usage_errors() {
    let d = Dir::new("m");
    d.write("a.kg", "my_var::5\n");
    let good = d.0.join("a.kg").to_string_lossy().into_owned();
    let missing = d.0.join("missing.kg").to_string_lossy().into_owned();
    let (rc, out, err) = run(&[&missing, &good]);
    assert_eq!(rc, 2);
    assert_eq!(err, format!("{missing}: cannot read: No such file or directory\n"));
    assert!(out.ends_with("0 errors, 1 warning in 1 file\n"));
    let (rc, _, err) = run(&[&good, "--select", "W999"]);
    assert_eq!(rc, 2);
    assert!(err.starts_with("kgpy lint: unknown lint code 'W999'"));
    let (rc, _, err) = run(&[]);
    assert_eq!(rc, 2);
    assert!(err.ends_with("kgpy lint: error: the following arguments are required: FILE\n"));
    d.write("bin.kg", "a::1\n\u{0}");
    std::fs::write(d.0.join("bad.kg"), [b'a', 0xff, b'\n']).unwrap();
    let bad = d.0.join("bad.kg").to_string_lossy().into_owned();
    let (rc, _, err) = run(&[&bad]);
    assert_eq!(rc, 2);
    assert_eq!(err, format!("{bad}: cannot read: 'utf-8' codec can't decode byte 0xff in position 1: invalid start byte\n"));
}

#[test]
fn options_in_any_order_and_prefixes() {
    let d = Dir::new("n");
    d.write("s.kg", "my_var::5\n[1;2]\n-1+2\n");
    let p = d.0.join("s.kg").to_string_lossy().into_owned();
    for argv in [vec![p.as_str(), "--select=W010"], vec!["--select", "W010", &p], vec![&p, "--sel", "W010"]] {
        let (_, out, _) = run(&argv);
        assert_eq!(out.matches(": W").count(), 1, "{:?}", argv);
    }
    let (_, out, _) = run(&[&p, "--select", "W010", "-v"]);
    assert!(out.contains("hint:"));
}

#[test]
fn define_names_the_host_provides() {
    let d = Dir::new("o");
    d.write("t.kg", "t(\"x\";1;1)\n");
    let p = d.0.join("t.kg").to_string_lossy().into_owned();
    assert!(run(&[&p]).1.contains("W002"));
    assert!(!run(&[&p, "--define", "t"]).1.contains("W002"));
}

// -- robustness -------------------------------------------------------------------------------------------------

#[test]
fn strange_input_does_not_crash() {
    let texts = [
        "", "\n\n", ":", ":\"", "[", "]", "{", "}", ")", "(", ":[", ":{", ":(", ":|", "0c", "[;", "{[", "{[a", "{[a];", "{[a b]::",
        "f(a;b)::", "[a b]::", "\"", ":\"abc", "é::1", "a::²", "1e", "1.2.3", ".comment(\"X\")", ".l(", ".l(\"", ".bkf(", ".bkf([\"a\"",
        "x::{", ":[1;2;:[", ",:{", "[1 2],:{", "a\u{0}b", "\t\r\n", "}{", "][", "){", "f(\n", "{x}(1;2;3;4;5;6)", "∇", "[a b]∇f",
    ];
    let opts = Options { no_resolve: true, ..Options::default() };
    for t in texts {
        for d in lint_source("t.kg", t, &opts) {
            assert!(RULES.iter().any(|r| r.code == d.code));
        }
    }
}

#[test]
fn mutated_library_sources_do_not_crash() {
    let dir = format!("{}/../../klongpy/lib", env!("CARGO_MANIFEST_DIR"));
    let mut texts = Vec::new();
    for e in std::fs::read_dir(dir).unwrap() {
        let p = e.unwrap().path();
        if p.extension().map(|x| x == "kg").unwrap_or(false) {
            texts.push(std::fs::read_to_string(p).unwrap());
        }
    }
    texts.sort();
    let chars: Vec<&str> = "[ ] ( ) { } ; : \" ' / \\ \n . , _ - + * % @ # | 0 1 2 a b x y z :[ :{ :( :| 0c :: :\" \\~".split(' ').map(|s| if s.is_empty() { " " } else { s }).collect();
    let mut state: u64 = 52;
    let mut rnd = |n: usize| -> usize {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((state >> 33) as usize) % n.max(1)
    };
    let opts = Options { no_resolve: true, ..Options::default() };
    for _ in 0..250 {
        let t = &texts[rnd(texts.len())];
        let cs: Vec<char> = t.chars().collect();
        let s = rnd(cs.len().saturating_sub(2000).max(1));
        let len = 50 + rnd(1950);
        let mut piece: Vec<String> = cs[s..(s + len).min(cs.len())].iter().map(|c| c.to_string()).collect();
        for _ in 0..1 + rnd(5) {
            let i = rnd(piece.len() + 1);
            if rnd(10) < 4 && !piece.is_empty() {
                piece.remove(i.min(piece.len() - 1));
            } else {
                piece.insert(i, chars[rnd(chars.len())].to_string());
            }
        }
        lint_source("fuzz.kg", &piece.concat(), &opts);
    }
}

#[test]
fn shipped_library_has_no_errors() {
    let dir = format!("{}/../../klongpy/lib", env!("CARGO_MANIFEST_DIR"));
    let mut paths: Vec<String> = std::fs::read_dir(dir).unwrap().map(|e| e.unwrap().path().to_string_lossy().into_owned()).filter(|p| p.ends_with(".kg")).collect();
    paths.sort();
    for r in lint_files(&paths, &Options::default()) {
        let errs: Vec<_> = r.result.unwrap().into_iter().filter(|d| d.severity == "error").collect();
        assert!(errs.is_empty(), "{}: {:?}", r.path, errs);
    }
}

#[test]
fn help_and_usage_are_argparses() {
    // `kgpy lint --help`, as argparse prints it
    let want = "usage: kgpy lint [-h] [--select CODES] [--ignore CODES] [--define NAMES] [-v]\n                 FILE [FILE ...]\n\nCheck FILEs for the classic Klong traps without running them, then exit (1 if\nany error was found).\n\npositional arguments:\n  FILE            the programs to check (- is stdin)\n\noptions:\n  -h, --help      show this help message and exit\n  --select CODES  report only these codes (comma-separated)\n  --ignore CODES  leave these codes out (comma-separated)\n  --define NAMES  names the host defines before the program runs (comma-\n                  separated)\n  -v, --verbose   enable verbose output\n";
    let (rc, out, _) = run(&["--help"]);
    assert_eq!((rc, out.as_str()), (0, want));
}
