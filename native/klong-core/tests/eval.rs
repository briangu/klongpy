//! Evaluator, operators, adverbs, printing and limits.
use klong_core::{Config, Interp, Kind, NoHooks, Value};

fn interp() -> Interp {
    Interp::new(Config::default(), Box::new(NoHooks))
}

fn run_with(ip: &mut Interp, text: &str) -> Result<String, Kind> {
    match ip.run(text, "<test>") {
        Ok(Some(v)) => Ok(ip.format(&v)),
        Ok(None) => Ok(String::new()),
        Err(e) => Err(e.kind),
    }
}

fn run(text: &str) -> Result<String, Kind> {
    run_with(&mut interp(), text)
}

/// Depth tests run on a thread with a stack of known size: a debug build uses several KiB of
/// native stack per nested evaluation, more than a test thread has for 512 levels.
fn big<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> T {
    std::thread::Builder::new().stack_size(256 << 20).spawn(f).expect("thread").join().expect("join")
}

fn ok(text: &str, want: &str) {
    assert_eq!(run(text).as_deref(), Ok(want), "{}", text);
}

fn err(text: &str, want: Kind) {
    assert_eq!(run(text), Err(want), "{}", text);
}

#[test]
fn value_is_small() {
    assert!(core::mem::size_of::<Value>() <= 16);
    #[cfg(target_pointer_width = "64")]
    assert_eq!(core::mem::size_of::<Value>(), 16);
}

#[test]
fn arithmetic_is_right_to_left() {
    ok("1+2*3", "7");
    ok("2*3+1", "8");
    ok("10-2-3", "11");
    ok("7%2", "3.5");
    ok("7:%2", "3");
    ok("7!2", "1");
    ok("2^10", "1024");
    ok("2^0.5", "1.4142135623730951");
    ok("1%0", ":undefined");
}

#[test]
fn atomic_dyads_pair_lists() {
    ok("1+[1 2 3]", "[2 3 4]");
    ok("[1 2 3]*[4 5 6]", "[4 10 18]");
    ok("[[1 2] [3 4]]+[10 20]", "[[11 12] [23 24]]");
    ok("[1 2 3]<2", "[1 0 0]");
    ok("[1 2]+[1.5 2]", "[2.5 4.0]");
    err("[1 2 3]+[1 2]", Kind::Shape);
    err("1+:a", Kind::Type);
}

#[test]
fn lists() {
    ok("!5", "[0 1 2 3 4]");
    ok("|[1 2 3]", "[3 2 1]");
    ok("#[1 2 3]", "3");
    ok("[1 2 3]@1", "2");
    ok("2#[1 2 3]", "[1 2]");
    ok("5#[1 2 3]", "[1 2 3 1 2]");
    ok("(-2)#[1 2 3]", "[2 3]");
    ok("1_[1 2 3]", "[2 3]");
    ok("[1 2],[3]", "[1 2 3]");
    ok("<[3 1 2]", "[1 2 0]");
    ok(">[3 1 2]", "[0 2 1]");
    ok("=[1 2 1]", "[[0 2] [1]]");
    ok("?[1 2 1 3]", "[1 2 3]");
    ok("&[0 2 1]", "[1 1 2]");
    ok("+[[1 2] [3 4]]", "[[1 3] [2 4]]");
    ok("^[[1 2 3] [4 5 6]]", "[2 3]");
    ok("[2 2]:^!4", "[[0 1] [2 3]]");
    ok("[1 2 3]~[1 2 3]", "1");
    ok("[1 2 3]?2", "[1]");
    err("[1 2 3]@5", Kind::Range);
}

#[test]
fn functions_arguments_locals_and_recursion() {
    ok("f::{x+y};f(1;2)", "3");
    ok("{x*2}(21)", "42");
    ok("f::{[a];a::x*2;a+1};f(3)", "7");
    ok("f::{:[x<2;x;f(x-1)+f(x-2)]};f(15)", "610");
    ok("f::{:[x<2;1;x*.f(x-1)]};f(5)", "120");
    ok("f::{x-y};g::f(;1);g(10)", "9");
    ok("{x}@5", "5");
}

#[test]
fn scope_is_dynamic() {
    // a function sees the locals of its caller (KlongPy; conformance functions.local-in-nested-call)
    ok("g::{a};f::{[a];a::7;g()};f()", "7");
    ok("a::1;f::{a::x};f(5);a", "5");
}

#[test]
fn conditionals() {
    ok(":[1;2;3]", "2");
    ok(":[0;2;3]", "3");
    ok(":[0;1:|1;2;3]", "2");
    ok(":[[];1;2]", "2");
}

#[test]
fn adverbs() {
    ok("+/!10", "45");
    ok("+\\[1 2 3]", "[1 3 6]");
    ok("{x*2}'[1 2 3]", "[2 4 6]");
    ok("1+/[1 2 3]", "7");
    ok("[1 2 3],'[4 5 6]", "[[1 4] [2 5] [3 6]]");
    ok("1+:\\[1 2 3]", "[2 3 4]");
    ok("-:'[1 3 6]", "[-2 -3]");
    ok("3{x*2}:*1", "8");
    ok("{x<100}{x*2}:~1", "128");
    ok("+/'[[1 2] [3 4]]", "[3 7]");
}

#[test]
fn printing_floats_as_python_repr() {
    ok("0.1+0.2", "0.30000000000000004");
    ok("1.0e15", "1000000000000000.0");
    ok("1.0e16", "1e+16");
    ok("1.0e-4", "0.0001");
    ok("1.0e-5", "1e-05");
    ok("1.5e300", "1.5e+300");
    ok("2.0", "2.0");
    ok("-0.5", "-0.5");
}

#[test]
fn printing_values() {
    ok("[1 [2 3] \"s\" :a 0cx]", "[1 [2 3] \"s\" :a 0cx]");
    ok("[]", "[]");
    ok("{x+1}", ":monad");
    ok("{x+y}", ":dyad");
}

#[test]
fn strict_levels() {
    let mut ip = interp();
    assert_eq!(run_with(&mut ip, "nosuch"), Ok(":nosuch".into()));
    ip.cfg.strict = 2;
    assert_eq!(run_with(&mut ip, "nosuch"), Err(Kind::Name));
    ip.cfg.strict = 1;
    assert_eq!(run_with(&mut ip, "f::{undeclared::x};f(1)"), Err(Kind::Name));
    assert_eq!(run_with(&mut ip, "g::{[v];v::x;v};g(1)"), Ok("1".into()));
}

#[test]
fn recursion_without_end_is_a_range_error() {
    big(|| {
        let mut ip = interp();
        assert_eq!(run_with(&mut ip, "f::{f(x+1)};f(0)"), Err(Kind::Range));
        // and the interpreter is usable afterwards
        assert_eq!(run_with(&mut ip, "1+1"), Ok("2".into()));
        assert_eq!(ip.max_depth_seen, ip.cfg.max_depth + 1);
    });
}

#[test]
fn a_loop_without_end_stops_at_the_step_limit() {
    let mut ip = Interp::new(Config { step_limit: Some(50_000), ..Config::default() }, Box::new(NoHooks));
    assert_eq!(run_with(&mut ip, "{1}{x+1}:~0"), Err(Kind::Interrupt));
    assert_eq!(run_with(&mut ip, "2*3"), Ok("6".into()));
}

#[test]
fn hooks_can_interrupt() {
    struct Stop;
    impl klong_core::Hooks for Stop {
        fn tick(&mut self) -> bool {
            true
        }
    }
    let mut ip = Interp::new(Config::default(), Box::new(Stop));
    assert_eq!(run_with(&mut ip, "{1}{x+1}:~0"), Err(Kind::Interrupt));
}

#[test]
fn huge_allocations_are_memory_errors() {
    // sized by data: refused by try_reserve, not by an abort
    #[cfg(target_pointer_width = "64")]
    err("!1000000000000000", Kind::Memory);
}

#[test]
fn errors_render_with_position() {
    let mut ip = interp();
    let e = ip.run("a::1\n[1 2 3]@9", "prog.kg").err().map(|e| e.render()).unwrap_or_default();
    assert!(e.contains("range error"), "{}", e);
    assert!(e.contains("2:"), "{}", e);
}

#[test]
fn native_functions_can_be_registered() {
    let mut ip = interp();
    ip.register(".twice", 1, std::rc::Rc::new(|_: &mut Interp, a: &[Value]| match a.first() {
        Some(Value::Int(i)) => Ok(Value::Int(i * 2)),
        _ => Err(klong_core::Error::typ(".twice needs an integer")),
    }));
    assert_eq!(run_with(&mut ip, ".twice(21)"), Ok("42".into()));
    assert_eq!(run_with(&mut ip, ".twice'[1 2]"), Ok("[2 4]".into()));
}

#[test]
fn short_calls_are_projections() {
    // ticket 033: a call with fewer arguments than the function takes
    ok("r::{x+y};s::r(1);s(10)", "11");
    ok("g::{x,y,z};h::g(1);i::h(2);i(3)", "[1 2 3]");
    ok("g::{x,y,z};h::g(;2);h(1;3)", "[1 2 3]");
    ok("f::{x+y};{x(10)}'{f(x)}'[1 2 3]", "[11 12 13]");
    ok("f(a;b)::{a+b};g::f(1);g(2)", "3");
    ok("g::{x+y+z};g(1)", ":dyad");
    ok("f::{x-y};p::f@10;p(3)", "7");
    // at top level a projection keeps its argument expressions
    ok("f::{x+y};a::1;g::f(a;);a::2;g(10)", "12");
}

#[test]
fn arguments_are_assignable_and_undefined_is_a_value() {
    let mut ip = interp();
    ip.cfg.strict = 2;
    assert_eq!(run_with(&mut ip, "h::{x::7;x+1};h(1)"), Ok("8".into()));
    ok("u::1%0;u~:undefined", "1");
    ok(":_:undefined", "1");
}

#[test]
fn found_by_fuzzing() {
    // a comparison that is not a total order made the sort panic
    err("<[1 \"a\" 2.5]", Kind::Type);
    ok("<[3.5 1.0 2.5]", "[1 2 0]");
    // the shape of a deep list was exponential in its depth
    ok(&format!("^{}1{}", "[".repeat(60), "]".repeat(60)), &format!("[{}]", vec!["1"; 60].join(" ")));
}
