//! Calling a value that is not a function is a type error at every strict level (KlongPy's
//! `tests/test_call_non_function.py`, on the native evaluator): what was called and what it is are in the message,
//! a list, string or dict gets the hint about indexing, an undefined name stays an undefined-function error, a
//! symbol that leads to a function is still called through, and what can be called still can.
use klong_core::{Config, Interp, Kind, NoHooks};

fn run(level: u8, lines: &[&str]) -> Result<String, (Kind, String, Option<String>)> {
    let mut ip = Interp::new(Config { strict: level, ..Config::default() }, Box::new(NoHooks));
    let mut last = String::new();
    for l in lines {
        match ip.run(l, "<test>") {
            Ok(Some(v)) => last = ip.format(&v),
            Ok(None) => last = String::new(),
            Err(e) => return Err((e.kind, e.message.to_string(), e.hint.as_ref().map(|h| h.to_string()))),
        }
    }
    Ok(last)
}

fn err(level: u8, lines: &[&str]) -> (Kind, String, Option<String>) {
    run(level, lines).expect_err(&format!("{:?} at level {}", lines, level))
}

const KINDS: [(&str, &str); 9] = [
    ("a::3", "an integer"),
    ("a::1.0", "a float"),
    ("a::\"text\"", "a string"),
    ("a::0cx", "a char"),
    ("a::[1 2 3]", "a list of 3"),
    ("a::[]", "an empty list"),
    ("a::[[1 2] [3 4]]", "a 2x2 matrix"),
    ("a:::{[1 2]}", "a dict"),
    ("a::1%0", ":undefined"),
];

const CALLS: [&str; 11] = ["a(3)", "a([3 4])", "a(1;2)", "a(1;2;3)", "a()", "1+a(2)", "{a(x)}(1)", "{[b];b::a;b(x)}(1)", "{x(2)}(a)", "{y(x)}(1;a)", "{[b];b::a;b()}()"];

#[test]
fn every_kind_of_value_every_form_every_level() {
    for level in 0..3 {
        for (define, kind) in KINDS {
            for call in CALLS {
                let (k, m, _) = err(level, &[define, call]);
                assert_eq!(k, Kind::Type, "{} {} level {}", define, call, level);
                assert!(m.starts_with("cannot call ") && m.ends_with(&format!("it is {}, not a function", kind)), "{} {}: {}", define, call, m);
            }
            let (_, m, _) = err(level, &[define, "a(1)"]);
            assert_eq!(m, format!("cannot call a: it is {}, not a function", kind));
        }
    }
}

#[test]
fn the_hint() {
    let (_, m, h) = err(0, &["avg::1.0", "avg([3 3 3 4 5 7])"]);
    assert_eq!(m, "cannot call avg: it is a float, not a function");
    assert_eq!(h.as_deref(), Some("a function is written {...}"));
    for define in ["a::[1 2 3]", "a::\"abc\"", "a:::{[1 2]}"] {
        assert_eq!(err(0, &[define, "a(1)"]).2.as_deref(), Some("a function is written {...}; to index, write a@i"), "{}", define);
    }
    assert_eq!(err(0, &["a::5", "a(1)"]).2.as_deref(), Some("a function is written {...}"));
}

#[test]
fn symbols_that_lead_to_a_function_are_called_and_the_others_are_an_error() {
    for level in 0..3 {
        assert_eq!(run(level, &["f::{x+1}", "g:::f", "g(2)"]).as_deref(), Ok("3"));
        assert_eq!(run(level, &["f::{x+1}", "g:::f", "h:::g", "h(2)"]).as_deref(), Ok("3"));
        assert_eq!(err(level, &["g:::f", "f::7", "g(2)"]).1, "cannot call g: it is a symbol, not a function");
        assert_eq!(err(level, &["g:::h", "h:::g", "g(2)"]).1, "cannot call g: it is a symbol, not a function");
        // a name given as a symbol to a function that calls its argument
        assert_eq!(run(level, &["f::{x+1}", "{x(1)}'[:f]"]).as_deref(), Ok("[2]"));
        assert_eq!(err(level, &["{x(2)}(:zzz)"]).1, "cannot call x: it is a symbol, not a function");
    }
    // a symbol that names nothing: at level 0 an undefined name is its own symbol (a symbol, not a function); above,
    // the call fails as an undefined function
    assert_eq!(err(0, &["a:::sym", "a(1)"]), (Kind::Type, "cannot call a: it is a symbol, not a function".into(), Some("a function is written {...}".into())));
    for level in 1..3 {
        let (k, m, _) = err(level, &["a:::sym", "a(1)"]);
        assert_eq!((k, m.as_str()), (Kind::Name, "undefined function: sym"));
    }
}

#[test]
fn an_undefined_name_keeps_its_error() {
    for level in 0..3 {
        let (k, m, _) = err(level, &["nope(1)"]);
        assert_eq!((k, m.as_str()), (Kind::Name, "undefined function: nope"));
    }
}

#[test]
fn a_nilad_literal_passed_and_called_is_an_error_a_name_passes_the_function() {
    for level in 0..3 {
        assert_eq!(err(level, &["ap::{x()}", "ap({7})"]).1, "cannot call x: it is an integer, not a function");
        assert_eq!(run(level, &["ap::{x()}", "g::{7}", "ap(g)"]).as_deref(), Ok("7"));
        // a literal whose body is a bare name is not a call of that name
        assert_eq!(run(level, &["f::{{x}(1)}", "f(5)"]).as_deref(), Ok("1"));
        assert_eq!(run(level, &["f::{a}", "a::4", "f()"]).as_deref(), Ok("4"));
    }
}

#[test]
fn adverbs_and_results_and_locals() {
    for level in 0..3 {
        for src in ["a'[1 2]", "a/[1 2]", "a\\[1 2]", "[1 2]a'[3 4]", "a:~5"] {
            assert_eq!(err(level, &["a::1", src]).1, "cannot call a: it is an integer, not a function", "{}", src);
        }
        assert_eq!(err(level, &["f::{x+1}", "g::f(3)", "g(1)"]).1, "cannot call g: it is an integer, not a function");
        assert_eq!(err(level, &["f::{[a];a::3;a(1)}", "f()"]).1, "cannot call a: it is an integer, not a function");
        assert_eq!(err(level, &["f::{y(1)}", "f(5;6)"]).1, "cannot call y: it is an integer, not a function");
        assert_eq!(err(level, &["f::{x+1}", "f::5", "f(1)"]).0, Kind::Type);
        assert_eq!(run(level, &["f::{x+1}", "f'[1 2]"]).as_deref(), Ok("[2 3]"));
        assert_eq!(run(level, &["f::{x+y}", "f/[1 2 3]"]).as_deref(), Ok("6"));
    }
}

#[test]
fn what_can_be_called_still_can() {
    for level in 0..3 {
        for (lines, want) in [
            (&["f::{x+1}", "f(2)"][..], "3"),
            (&["f::{x+y}", "f(2;3)"], "5"),
            (&["f::{7}", "f()"], "7"),
            (&["{x*2}(4)"], "8"),
            (&["f::{x+y}", "p::f(1;)", "p(5)"], "6"),
            (&["p::{x+y}(;1)", "p(5)"], "6"),
            (&["ap::{x(y)}", "g::{x+1}", "ap(g;5)"], "6"),
            (&["ap::{x(y)}", "ap({x+1};5)"], "6"),
            (&["fact::{:[x<2;1;x*.f(x-1)]}", "fact(5)"], "120"),
            (&["+/[1 2 3]"], "6"),
            (&["a::3", "a+1"], "4"),
            (&["a::[1 2 3]", "a@1"], "2"),
            (&["d:::{[1 2]}", "d@1"], "2"),
        ] {
            assert_eq!(run(level, lines).as_deref(), Ok(want), "{:?}", lines);
        }
    }
    assert_eq!(run(1, &["f(a;b)::{a+b}", "f(2;3)"]).as_deref(), Ok("5"));
}

#[test]
fn the_error_points_at_the_call() {
    let mut ip = Interp::new(Config::default(), Box::new(NoHooks));
    ip.run("avg::1.0", "<t>").unwrap();
    let e = match ip.run("1+avg([3 3 3 4 5 7])", "<t>") {
        Err(e) => e,
        Ok(_) => panic!("no error"),
    };
    let (sp, src) = (e.span.expect("a span"), e.source.as_ref().expect("a source"));
    assert_eq!(&src.text[sp.0 as usize..sp.1 as usize], "avg([3 3 3 4 5 7])");
}
