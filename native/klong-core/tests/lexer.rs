//! Lexer: token kinds, values, spans and the whitespace flag.
use klong_core::lex::{tokenize, K, TV};

fn kinds(text: &str) -> Vec<K> {
    tokenize(text).iter().map(|t| t.k).collect()
}

#[test]
fn numbers() {
    let t = tokenize("12 3.5 1e3");
    assert!(matches!(t[0].v, TV::Int(12)));
    assert!(matches!(t[1].v, TV::Flt(f) if f == 3.5));
    assert!(matches!(t[2].v, TV::Flt(f) if f == 1000.0));
    assert_eq!((t[0].start, t[0].end), (0, 2));
    assert_eq!((t[1].start, t[1].end), (3, 6));
    assert!(!t[0].ws && t[1].ws);
}

#[test]
fn names_symbols_strings_chars() {
    let t = tokenize("foo :bar \"a\"\"b\" 0cx");
    assert_eq!(t[0].k, K::Name);
    assert_eq!(t[1].k, K::QName);
    assert_eq!(t[2].k, K::Str);
    assert!(matches!(&t[2].v, TV::Str(s) if s.as_str() == "a\"b"));
    assert_eq!(t[3].k, K::Char);
    assert!(matches!(t[3].v, TV::Char('x')));
}

#[test]
fn operators_and_brackets() {
    assert_eq!(kinds("{x+1}")[..5], [K::LBrace, K::Name, K::Op, K::Num, K::RBrace]);
    assert_eq!(kinds(":[1;2;3]")[0], K::Cond);
    assert_eq!(kinds(":{[1 2]}")[0], K::Dict);
    // a two-character operator is one token
    let t = tokenize("a:=b");
    assert_eq!(t[1].k, K::Cop);
    assert_eq!((t[1].start, t[1].end), (1, 3));
}

#[test]
fn whitespace_before_a_parenthesis_is_recorded() {
    // f(1) is a call, f (1) is not: the parser decides from this flag
    assert_eq!(kinds("f(1)")[1], K::LParen);
    assert!(!tokenize("f(1)")[1].ws);
    assert!(tokenize("f (1)")[1].ws);
}

#[test]
fn unterminated_string_is_flagged() {
    let t = tokenize("\"abc");
    assert_eq!(t[0].k, K::Str);
    assert!(t[0].unterminated);
}

#[test]
fn ends_with_eof_and_never_panics_on_odd_bytes() {
    for text in ["", " ", "0c", "1e", ":", "\u{2207}x", "\u{e9}", "a::\u{1F600}", "\"", "::::", "1.2.3"] {
        let t = tokenize(text);
        assert_eq!(t.last().map(|t| t.k), Some(K::Eof), "{:?}", text);
    }
}
