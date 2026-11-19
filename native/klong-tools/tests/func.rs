//! The text of a function sent over IPC (`klongpy/ipc/endpoint.py`).
use klong_tools::func::{display_arity, function_form, wire_text, FunctionForm};

fn form(text: &str) -> Option<(bool, u8, Option<Vec<&'static str>>)> {
    function_form(text).map(|f: FunctionForm| (f.named, f.arity, f.params.map(|p| p.iter().map(|s| &*Box::leak(s.clone().into_boxed_str())).collect())))
}

#[test]
fn plain_literals() {
    assert_eq!(form("{x+y}"), Some((false, 2, None)));
    assert_eq!(form("{1}"), Some((false, 0, None)));
    assert_eq!(form("{x}"), Some((false, 1, None)));
    assert_eq!(form("{x+y+z}"), Some((false, 3, None)));
    assert_eq!(form("  {x}\n"), Some((false, 1, None)));
    assert_eq!(form("({x})"), Some((false, 1, None)));
    assert_eq!(form("{[a b];a::x;a+b}").map(|f| f.1), Some(1));
    assert_eq!(function_form("{[a b];a::x;a+b}").unwrap().locals, vec!["a", "b"]);
}

#[test]
fn named_form() {
    assert_eq!(form("fn(a;b)::{a+b}"), Some((true, 2, Some(vec!["a", "b"]))));
    assert_eq!(form("fn(a)::{a}"), Some((true, 1, Some(vec!["a"]))));
    // a name other than the placeholder is not a function that travels
    assert_eq!(form("g(a;b)::{a+b}"), None);
    // without parameters it is an assignment, not the named form
    assert_eq!(form("fn::{x}"), None);
}

#[test]
fn what_is_not_one_function() {
    for t in ["", "1", "x", "+", "{x}(1)", "{x}'[1 2]", "{x};{y}", "f::{x}", "{x", "}", "f(1;)", "+/", "[1 2]", "{x}:(1)", ":\"c\""] {
        assert_eq!(form(t), None, "{:?}", t);
    }
}

#[test]
fn wire_texts() {
    assert_eq!(wire_text("{x+y}", None).as_deref(), Some("{x+y}"));
    assert_eq!(wire_text("{a+b}", Some(&["a".to_string(), "b`m".to_string()])).as_deref(), Some("fn(a;b)::{a+b}"));
    assert_eq!(wire_text("x+y", None), None);
    // what goes out comes back as the same function
    let t = wire_text("{a*b}", Some(&["a".to_string(), "b".to_string()])).unwrap();
    assert_eq!(form(&t), Some((true, 2, Some(vec!["a", "b"]))));
}

#[test]
fn display_forms() {
    assert_eq!(display_arity(":nilad"), Some(0));
    assert_eq!(display_arity(":monad"), Some(1));
    assert_eq!(display_arity(":dyad"), Some(2));
    assert_eq!(display_arity(":triad"), Some(3));
    assert_eq!(display_arity(":fn/5"), Some(5));
    assert_eq!(display_arity(":monad\n"), Some(1));
    for t in ["monad", ":fn/", ":fn/x", ":quad", ":fn/-1", ": monad"] {
        assert_eq!(display_arity(t), None, "{:?}", t);
    }
}
