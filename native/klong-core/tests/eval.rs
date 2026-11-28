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
    // a call with fewer arguments than the function takes
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

#[test]
fn integer_power_is_exact_when_it_fits_and_wraps_when_it_does_not() {
    ok("2^10", "1024");
    ok("(-2)^3", "-8");
    ok("3^0", "1");
    ok("0^0", "1");
    ok("[1 2 3]^2", "[1 4 9]");
    #[cfg(all(not(feature = "int32"), not(feature = "bigint")))]
    {
        ok("2^62", "4611686018427387904");
        // wraps at 64 bits like `*`; it does not saturate at the largest integer
        ok("2^63", "-9223372036854775808");
        ok("2^64", "0");
        ok("3^40", "-6289078614652622815");
        ok("(-2)^63", "-9223372036854775808");
        ok("10^19", "-8446744073709551616");
        ok("2^63", &run("2*2^62").unwrap());
        ok("[2 3]^[63 40]", "[-9223372036854775808 -6289078614652622815]");
    }
    // with big integers a scalar power is exact (KlongPy); two machine integers still wrap in `*`, and a list wraps
    #[cfg(all(not(feature = "int32"), feature = "bigint"))]
    {
        ok("2^62", "4611686018427387904");
        ok("2^63", "9223372036854775808");
        ok("2^64", "18446744073709551616");
        ok("3^40", "12157665459056928801");
        ok("(-2)^63", "-9223372036854775808");
        ok("10^19", "10000000000000000000");
        ok("2*2^62", "-9223372036854775808");
        ok("[2 3]^[63 40]", "[-9223372036854775808 -6289078614652622815]");
    }
    #[cfg(feature = "int32")]
    {
        ok("2^30", "1073741824");
        ok("2^31", "-2147483648");
        ok("2^32", "0");
        ok("3^21", "1870418611");
        ok("3^22", "1316288537");
        ok("2^31", &run("2*2^30").unwrap());
    }
    // a negative exponent or a fraction gives a float; a whole float result is an integer
    ok("2^-1", "0.5");
    ok("4^0.5", "2");
    ok("1^-1", "1");
}

/// `load_owned` is `load` of a text that is handed over: the same definitions, and an error cut from the text that was handed over.
#[test]
fn load_owned_is_load_without_a_second_copy() {
    let mut ip = Interp::new(Config::default(), Box::new(NoHooks));
    ip.load_owned(String::from("sq::{x*x}\nsq(7)"), "<owned>").unwrap();
    let v = ip.run("sq(6)", "<t>").unwrap().unwrap();
    assert_eq!(ip.format(&v), "36");
    let e = match ip.load_owned(String::from("a::1\nb::2+)\nc::3"), "<owned>") {
        Err(e) => e,
        Ok(_) => panic!("a syntax error is an error"),
    };
    let shown = e.render();
    assert!(shown.starts_with("2:") && shown.contains("b::2+)"), "{shown}");
}

/// `v::v,y` is made in the value where nothing else holds it; every other holder of the value keeps what it had.
#[test]
fn append_assign_keeps_other_holders() {
    ok("a::\"x\";b::a;a::a,\"y\";a,\"|\",b", "\"xy|x\"");
    ok("v::!3;w::v;v::v,9;(v,-1),w", "[0 1 2 9 -1 0 1 2]");
    ok("s::\"ab\";s::s,s;s", "\"abab\"");
    ok("v::1,2;v::v,v;v", "[1 2 1 2]");
    ok("v::1,2;u::3,4;v::v,u;v::v,5;(v,-1),u", "[1 2 3 4 5 -1 3 4]");
    // a literal of a function is the same value at every call
    ok("g::{s::\"ab\";s::s,\"c\";s};g(0);g(0)", "\"abc\"");
    // the results of a loop are each their own string
    ok("s::\"\";{s::s,\"ab\"}'!3", "[\"ab\" \"abab\" \"ababab\"]");
    ok("s::\"é\";s::s,\"ü\";s::s,\"z\";#s", "3");
    ok("f::{[s];s::\"\";s::s,\"a\";s::s,0c!;s};f(0)", "\"a!\"");
    // not strings or vectors: the join as it was
    ok("l::[1 2];l::l,3;l", "[1 2 3]");
    ok("v::!2;v::v,1.5;v", "[0.0 1.0 1.5]");
}

#[test]
fn collection_reassignment_respects_module_and_frame_bindings() {
    let mut ip = interp();
    ip.run("v::[1 2 3]", "<global>").unwrap();
    let m = ip.module_new("collections", "", false).unwrap();
    ip.module_run_static(m, "v::[4 5];tick()::{v::v:=[9 1]};local()::{[v];v::[3 4];v::v:=[7 0]}", "<module>").unwrap();
    let value = ip.module_run_static(m, "tick();[;local();v]", "<call>").unwrap().unwrap();
    assert_eq!(ip.format(&value), "[[7 4] [4 9]]");
    assert_eq!(run_with(&mut ip, "v").as_deref(), Ok("[1 2 3]"));
}

#[cfg(feature = "host")]
#[test]
fn collection_assignment_respects_host_numeric_hooks() {
    let mut ip = interp();
    ip.host_numeric = Some(std::rc::Rc::new(|op, _, _| match op {
        "," => Some(Ok(Value::Int(41))),
        ":=" => Some(Ok(Value::Int(42))),
        _ => None,
    }));
    assert_eq!(run_with(&mut ip, "v::!3;v::v,5").as_deref(), Ok("41"));
    assert_eq!(run_with(&mut ip, "v::!3;v::v:=[2 0]").as_deref(), Ok("42"));
}
