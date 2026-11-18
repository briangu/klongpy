//! The fixtures of the static profile (tests/static/accept and reject): an accept file is inside the profile, a reject file has a
//! violation of the rule its name gives, on the line its first line expects. These tests need no Python.
use klong_static::{check_source, check::Options};
use std::path::PathBuf;

fn dir(sub: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/static").join(sub)
}

fn names(sub: &str) -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(dir(sub)).unwrap().filter_map(|e| e.ok()).map(|e| e.file_name().to_string_lossy().into_owned()).filter(|n| n.ends_with(".kg")).map(|n| n[..n.len() - 3].to_string()).collect();
    v.sort();
    v
}

#[test]
fn accept_files_are_inside_the_profile() {
    for rule in names("accept") {
        if rule.starts_with("stream-") {
            continue; // the streamable tier is a different check
        }
        let path = dir("accept").join(format!("{}.kg", rule));
        let text = std::fs::read_to_string(&path).unwrap();
        let r = check_source(&text, path.to_str().unwrap(), &Options::default());
        let found: Vec<String> = r.diagnostics.iter().map(|d| format!("{}: {}", d.rule, d.message)).collect();
        assert!(found.is_empty(), "accept/{}: {:?}", rule, found);
    }
}

#[test]
fn reject_files_have_their_violation_where_they_say() {
    for rule in names("reject") {
        if rule.starts_with("stream-") {
            continue;
        }
        let path = dir("reject").join(format!("{}.kg", rule));
        let text = std::fs::read_to_string(&path).unwrap();
        let first = text.lines().next().unwrap();
        // :" expect: static line N rule "
        let want: usize = first.split("static line ").nth(1).and_then(|s| s.split(' ').next()).and_then(|n| n.parse().ok()).unwrap();
        let r = check_source(&text, path.to_str().unwrap(), &Options::default());
        let hit = r.diagnostics.iter().any(|d| {
            let line = text[..d.span.0].matches('\n').count() + 1;
            d.rule == rule && line == want
        });
        assert!(hit, "reject/{}: expected {} on line {}, got {:?}", rule, rule, want, r.diagnostics.iter().map(|d| (d.rule, d.span)).collect::<Vec<_>>());
    }
}

#[test]
fn every_rule_has_its_fixtures() {
    let rules: Vec<&str> = klong_static::tables::RULES.iter().map(|r| r.id).collect();
    let mut a = names("accept");
    let mut b = names("reject");
    a.sort();
    b.sort();
    let mut want: Vec<String> = rules.iter().map(|s| s.to_string()).collect();
    want.sort();
    assert_eq!(a, want);
    assert_eq!(b, want);
}

#[test]
fn a_program_without_an_entry_is_checked_as_a_library() {
    let r = check_source("f::{x+1}\ng::{f(x)*2}\n", "t.kg", &Options::default());
    assert!(r.ok());
    assert_eq!(r.functions.len(), 2);
    assert!(r.entries.is_empty());
}

#[test]
fn the_report_names_what_it_found() {
    let r = check_source("f::{x+y}\n.entry(\"f\";\"int -> int\")\n", "t.kg", &Options::default());
    assert_eq!(r.diagnostics.len(), 1);
    assert_eq!(r.diagnostics[0].rule, "entry");
}
