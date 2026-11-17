//! `Config::max_nest`. In a file of its own: the bound is kept per process.
use klong_core::{Config, Interp, Kind, NoHooks};

#[test]
fn the_bound_on_nesting_is_configurable() {
    let mut ip = Interp::new(Config { max_nest: 16, ..Config::default() }, Box::new(NoHooks));
    let mut run = |text: &str| ip.run(text, "<test>").map(|_| ()).map_err(|e| e.kind);
    // (list levels are counted; the numeric array at the bottom adds its rank, at most 4)
    assert_eq!(run("#12{,x}:*1"), Ok(()));
    assert_eq!(run("24{,x}:*1"), Err(Kind::Range));
    assert_eq!(run("#12{,x}:*\"a\""), Ok(()));
    assert_eq!(run("18{,x}:*\"a\""), Err(Kind::Range));
    let deep = format!("{}1{}", "[".repeat(24), "]".repeat(24));
    assert_eq!(run(&deep), Err(Kind::Range));
}
