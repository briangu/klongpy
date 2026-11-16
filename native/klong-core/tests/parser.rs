//! Parser: the canonical dump (frontend.md section 7) is deterministic and stable under
//! re-parsing, arity is the highest argument used, and syntax errors carry a span.
use klong_core::{dump::dump, Config, Interp, Kind, NoHooks};

fn interp() -> Interp {
    Interp::new(Config::default(), Box::new(NoHooks))
}

fn tree(text: &str) -> String {
    let mut ip = interp();
    let p = ip.parse(text, "<test>").unwrap_or_else(|e| panic!("{}: {}", text, e.render()));
    dump(&p.tree, &ip.syms, &p.stmts)
}

fn kind(text: &str) -> Kind {
    let mut ip = interp();
    match ip.parse(text, "<test>") {
        Ok(_) => panic!("{} parsed", text),
        Err(e) => e.kind,
    }
}

#[test]
fn dump_is_deterministic() {
    for text in ["1+2*3", "f::{[a b];a::x;b::y;a+b}", ":[x;1:|y;2;3]", "+/{x*2}'!10", "[1 [2 3] \"s\" :sym 0cx 1.5]", "f(1;;3)", ":{[1 2] [:a \"b\"]}", "{x,y,z}"] {
        assert_eq!(tree(text), tree(text), "{}", text);
        assert!(!tree(text).is_empty());
    }
}

#[test]
fn right_to_left_and_parentheses_differ() {
    assert_ne!(tree("1+2*3"), tree("(1+2)*3"));
    assert_eq!(tree("1+2*3"), tree("1+(2*3)"));
}

#[test]
fn statements_are_separated() {
    assert_eq!(tree("a::1;b::2"), tree("a::1\nb::2"));
}

#[test]
fn arity_is_the_highest_argument_used() {
    let (m, d, t) = (tree("{x}"), tree("{y}"), tree("{z}"));
    assert_ne!(m, d);
    assert_ne!(d, t);
    assert_eq!(tree("{x+y}"), tree("{x+y}"));
}

#[test]
fn syntax_errors() {
    assert_eq!(kind("{x"), Kind::Syntax);
    assert_eq!(kind(":[1;2"), Kind::Syntax);
    // KlongPy accepts a list or a string that the end of input closes (strict level 0)
    assert_eq!(tree("[1 2"), tree("[1 2]"));
    assert_eq!(tree("\"abc"), tree("\"abc\""));
}

#[test]
fn syntax_error_has_a_span_and_renders() {
    let mut ip = interp();
    let e = match ip.parse("1+{x", "prog.kg") {
        Ok(_) => panic!("parsed"),
        Err(e) => e,
    };
    assert!(e.span.is_some());
    let text = e.render();
    assert!(text.contains("syntax error"), "{}", text);
}

#[test]
fn deep_nesting_is_an_error_not_a_stack_overflow() {
    let t = std::thread::Builder::new().stack_size(256 << 20).spawn(|| {
        let mut ip = Interp::new(Config::default(), Box::new(NoHooks));
        let text = format!("{}1{}", "(".repeat(5000), ")".repeat(5000));
        assert_eq!(ip.parse(&text, "<deep>").err().map(|e| e.kind), Some(Kind::Range));
        let text = "-".repeat(5000) + "1";
        assert_eq!(ip.parse(&text, "<deep>").err().map(|e| e.kind), Some(Kind::Range));
        let text = "[".repeat(5000);
        assert!(ip.parse(&text, "<deep>").is_err());
    });
    t.expect("thread").join().expect("join");
}
