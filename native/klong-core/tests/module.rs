//! The shared module reader and search (`klong_core::module`): what the runtime, the linter and the checker all ask.
use klong_core::error::Source;
use klong_core::module::{self, Entry, Env, Place};
use klong_core::parse::parse;
use klong_core::sym::Interner;
use std::borrow::Cow;
use std::rc::Rc;

fn is_module(text: &str) -> bool {
    module::scan(&Rc::new(Source { name: "t".into(), text: Cow::Owned(text.into()) }), 512)
}

fn interface_of(text: &str) -> Option<Option<Vec<String>>> {
    let mut syms = Interner::new();
    let mut m = None;
    let p = parse(text, "t", &mut syms, &mut m, 0, 512, None).unwrap();
    for &s in &p.stmts {
        if let Some(i) = module::interface(&p.tree, &syms, s) {
            return Some(i.entries.map(|e| e.into_iter().map(|(k, _)| k).collect()));
        }
    }
    None
}

#[test]
fn a_file_is_a_module_when_it_assigns_export_at_the_top_level() {
    assert!(is_module("f::{x}\nexport:::{;\"f\";f}"));
    assert!(is_module("export::d"));
    assert!(!is_module("f::{x}"));
    assert!(!is_module("f::{export::1}"));
    assert!(!is_module(":\"export::1\"\nf::1"));
    assert!(!is_module("s::\"export::1\""));
    assert!(!is_module("exports::1"));
    assert!(!is_module(".p(export)"));
}

#[test]
fn the_interface_is_read_from_a_literal() {
    assert_eq!(interface_of("export:::{;\"a\";a;\"b\";b}"), Some(Some(vec!["a".into(), "b".into()])));
    // constants only: the parser reads a constant dictionary
    assert_eq!(interface_of("export:::{;\"v\";3}"), Some(Some(vec!["v".into()])));
    assert_eq!(interface_of("export:::{}"), Some(Some(vec![])));
    assert_eq!(interface_of("export::d"), Some(None));
    assert_eq!(interface_of("export:::{;a;1}").map(|e| e.is_none()), Some(true), "a key that is not a string literal is not a literal interface");
    assert_eq!(interface_of("f::1"), None);
}

#[test]
fn the_values_of_a_literal_interface_are_nodes_or_constants() {
    let mut syms = Interner::new();
    let mut m = None;
    let p = parse("export:::{;\"f\";f;\"v\";3}", "t", &mut syms, &mut m, 0, 512, None).unwrap();
    let i = module::interface(&p.tree, &syms, p.stmts[0]).unwrap();
    let e = i.entries.unwrap();
    assert!(matches!(e[0].1, Entry::Node(_)) || matches!(e[0].1, Entry::Const(_)));
    assert_eq!(e.len(), 2);
}

#[test]
fn a_statement_that_calls_extern_and_discards_it_is_found() {
    let mut syms = Interner::new();
    let mut m = None;
    let p = parse(".extern(\"a\";\"int -> int\")\nf::.extern(\"a\";\"int -> int\")", "t", &mut syms, &mut m, 0, 512, None).unwrap();
    assert!(module::discarded_extern(&p.tree, &syms, p.stmts[0]));
    assert!(!module::discarded_extern(&p.tree, &syms, p.stmts[1]));
}

fn names(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| s.to_string()).collect()
}

/// `files` are the paths that exist; `lib` the bundled library's names.
fn find(env: &Env, spec: &str, files: &[&str], lib: &[&str]) -> Option<(Place, bool)> {
    let mut tried = Vec::new();
    let mut probe = |p: &str| files.contains(&p);
    let mut in_lib = |n: &str| lib.contains(&n);
    module::search(env, spec, &mut probe, &mut in_lib, &mut tried).map(|f| (f.place, f.shadows))
}

#[test]
fn a_bare_name_is_looked_for_beside_the_importer_then_in_klongpath_then_in_the_library() {
    let kp = names(&["libs"]);
    let env = Env { dir: "proj", from_lib: false, klongpath: &kp };
    assert_eq!(find(&env, "m", &["proj/m.kg", "libs/m.kg"], &[]), Some((Place::File("proj/m.kg".into()), false)));
    assert_eq!(find(&env, "m", &["libs/m.kg"], &[]), Some((Place::File("libs/m.kg".into()), false)));
    assert_eq!(find(&env, "csv", &[], &["csv"]), Some((Place::Library("csv".into()), false)));
    assert_eq!(find(&env, "m", &[], &["csv"]), None);
    // the name as written is tried before name.kg
    assert_eq!(find(&env, "m.kg", &["proj/m.kg"], &[]), Some((Place::File("proj/m.kg".into()), false)));
}

#[test]
fn a_local_file_that_has_the_name_of_a_library_file_shadows_it() {
    let kp: Vec<String> = Vec::new();
    let env = Env { dir: "proj", from_lib: false, klongpath: &kp };
    assert_eq!(find(&env, "csv", &["proj/csv.kg"], &["csv"]), Some((Place::File("proj/csv.kg".into()), true)));
    // a path that says where it is does not shadow anything
    assert_eq!(find(&env, "./csv", &["proj/csv.kg"], &["csv"]), Some((Place::File("proj/csv.kg".into()), false)));
}

#[test]
fn the_working_directory_is_the_importer_at_the_prompt() {
    let kp = names(&["."]);
    let env = Env { dir: "", from_lib: false, klongpath: &kp };
    let mut tried = Vec::new();
    let mut probe = |_: &str| false;
    let mut in_lib = |_: &str| false;
    assert!(module::search(&env, "m", &mut probe, &mut in_lib, &mut tried).is_none());
    // "" and "." are the same directory: asked once
    assert_eq!(tried, names(&["m", "m.kg"]));
}

#[test]
fn relative_names_follow_the_importer_and_an_absolute_path_is_itself() {
    let kp = names(&["libs"]);
    let env = Env { dir: "/a/b", from_lib: false, klongpath: &kp };
    assert_eq!(find(&env, "./x", &["/a/b/x.kg"], &[]), Some((Place::File("/a/b/x.kg".into()), false)));
    assert_eq!(find(&env, "../x", &["/a/x.kg"], &[]), Some((Place::File("/a/x.kg".into()), false)));
    assert_eq!(find(&env, "../x", &["libs/x.kg"], &[]), None, "a relative name is not looked for in KLONGPATH");
    assert_eq!(find(&env, "/c/x", &["/c/x.kg"], &[]), Some((Place::File("/c/x.kg".into()), false)));
}

#[test]
fn a_file_of_the_library_reaches_another_by_name() {
    let kp: Vec<String> = Vec::new();
    let env = Env { dir: "", from_lib: true, klongpath: &kp };
    assert_eq!(find(&env, "math", &["math.kg"], &["math"]), Some((Place::Library("math".into()), false)));
    assert_eq!(find(&env, "./math", &[], &["math"]), Some((Place::Library("math".into()), false)));
}

#[test]
fn names_and_paths() {
    assert_eq!(module::stem("a/b/nstat.kg"), "nstat");
    assert_eq!(module::stem("nstat"), "nstat");
    assert!(module::is_binding_name("nstat2"));
    assert!(!module::is_binding_name("my-mod"));
    assert!(!module::is_binding_name("2x"));
    assert!(!module::is_binding_name("a.b"));
    assert!(!module::is_binding_name("x"));
    assert_eq!(module::normalize("a/./b/../c"), "a/c");
    assert_eq!(module::normalize("../a"), "../a");
    assert_eq!(module::join("", "x"), "x");
    assert_eq!(module::join("/", "x"), "/x");
    assert_eq!(module::dir_of("/a/b.kg"), "/a");
    assert_eq!(module::dir_of("b.kg"), "");
}
