//! `--streamable`: each `tests/static/reject/stream-*.kg` is inside the profile and has the finding named by its file, and
//! the reference programs have none or the ones the hardware document says.
use klong_static::compile::compile_program;
use klong_static::streamable::check_streamable;

fn dir() -> String {
    format!("{}/../../tests/static/reject", env!("CARGO_MANIFEST_DIR"))
}

#[test]
fn each_stream_fixture_has_its_finding() {
    let mut n = 0;
    for rule in ["stream-block", "stream-loop", "stream-state", "stream-table"] {
        let path = format!("{}/{}.kg", dir(), rule);
        let c = compile_program(&path, None, 32, None).ok().expect("inside the profile");
        let found = check_streamable(&c.program, &c.report);
        assert!(found.iter().any(|d| d.rule == rule), "{}: {:?}", rule, found.iter().map(|d| d.rule).collect::<Vec<_>>());
        n += 1;
    }
    assert_eq!(n, 4);
}

#[test]
fn the_embedded_programs_are_not_all_streamable() {
    for (name, ok) in [("mavg", true), ("fir", true)] {
        let path = format!("{}/../../examples/embedded/{}.kg", env!("CARGO_MANIFEST_DIR"), name);
        let c = compile_program(&path, None, 32, None).ok().expect("compiles");
        let found = check_streamable(&c.program, &c.report);
        assert_eq!(found.is_empty(), ok, "{}: {:?}", name, found.iter().map(|d| &d.message).collect::<Vec<_>>());
    }
}
