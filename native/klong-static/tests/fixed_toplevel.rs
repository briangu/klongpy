//! Fixed-point arithmetic evaluated by the interpreter at the top level of a static program (`b::a+a` with `a` fixed): the reference
//! semantics of `fixed.rs`, as `FixedArray` has them in Python (src/fixed_ext.rs). The values are those of `klongpy/static/fixed.py`.
use klong_static::host::{load_program, HostValue};
use std::collections::HashMap;

fn run(text: &str) -> Result<HostValue, String> {
    let p = load_program("t.kg", Some(text), HashMap::new()).map_err(|e| format!("{}: {}", e.kind.name(), e.message))?;
    p.get("r").ok_or_else(|| "no r".to_string())
}

fn fixed(elem: &str, shape: &[usize], raw: &[i64]) -> HostValue {
    HostValue::Fixed { elem: elem.to_string(), shape: shape.to_vec(), raw: raw.to_vec() }
}

fn ints(shape: &[usize], v: &[i64]) -> HostValue {
    HostValue::Ints(shape.to_vec(), v.to_vec())
}

const A: &str = "a::.fixed(\"fixed<32,16>\";[0.5 1.5 -2.25])\n";

#[test]
fn the_case_of_the_task() {
    let v = run("a::.fixed(\"fixed<32,16>\";0.5)\nb::a+a\nr::b").unwrap();
    assert_eq!(v, fixed("fixed<32,16>", &[], &[65536]));
}

#[test]
fn dyads_between_fixed_values() {
    let g = |op: &str| run(&format!("{}r::a{}a", A, op)).unwrap();
    assert_eq!(g("+"), fixed("fixed<32,16>", &[3], &[65536, 196608, -294912]));
    assert_eq!(g("-"), fixed("fixed<32,16>", &[3], &[0, 0, 0]));
    assert_eq!(g("*"), fixed("fixed<32,16>", &[3], &[16384, 147456, 331776]));
    assert_eq!(g("%"), fixed("fixed<32,16>", &[3], &[65536, 65536, 65536]));
    assert_eq!(g("|"), fixed("fixed<32,16>", &[3], &[32768, 98304, -147456]));
    assert_eq!(g("&"), fixed("fixed<32,16>", &[3], &[32768, 98304, -147456]));
    assert_eq!(g("<"), ints(&[3], &[0, 0, 0]));
    assert_eq!(g(">"), ints(&[3], &[0, 0, 0]));
    assert_eq!(g("="), ints(&[3], &[1, 1, 1]));
}

#[test]
fn ints_convert_to_the_fixed_type() {
    assert_eq!(run(&format!("{}r::a+1", A)).unwrap(), fixed("fixed<32,16>", &[3], &[98304, 163840, -81920]));
    assert_eq!(run(&format!("{}r::1+a", A)).unwrap(), fixed("fixed<32,16>", &[3], &[98304, 163840, -81920]));
    assert_eq!(run(&format!("{}r::a+[1 2 3]", A)).unwrap(), fixed("fixed<32,16>", &[3], &[98304, 229376, 49152]));
    assert_eq!(run(&format!("{}r::a*2", A)).unwrap(), fixed("fixed<32,16>", &[3], &[65536, 196608, -294912]));
    assert_eq!(run(&format!("{}r::a%2", A)).unwrap(), fixed("fixed<32,16>", &[3], &[16384, 49152, -73728]));
    // a scalar against an array
    assert_eq!(run(&format!("{}s::.fixed(\"fixed<32,16>\";0.5)\nr::s+a", A)).unwrap(), fixed("fixed<32,16>", &[3], &[65536, 131072, -114688]));
    assert_eq!(run("s::.fixed(\"fixed<32,16>\";0.5)\nr::s*s").unwrap(), fixed("fixed<32,16>", &[], &[16384]));
}

#[test]
fn results_wrap_to_the_width() {
    // 100 * 100 does not fit fixed<16,8> (127.99): the product wraps as two's complement
    let v = run("a::.fixed(\"fixed<16,8>\";100)\nr::a*a").unwrap();
    let raw = 100i64 * 256;
    let want = ((raw * raw) >> 8) as i16 as i64;
    assert_eq!(v, fixed("fixed<16,8>", &[], &[want]));
    assert_eq!(run("a::.fixed(\"fixed<8,4>\";7)\nr::a+a").unwrap(), fixed("fixed<8,4>", &[], &[-32]));
}

#[test]
fn folds_and_scans() {
    assert_eq!(run(&format!("{}r::+/a", A)).unwrap(), fixed("fixed<32,16>", &[], &[-16384]));
    assert_eq!(run(&format!("{}r::+\\a", A)).unwrap(), fixed("fixed<32,16>", &[3], &[32768, 131072, -16384]));
    assert_eq!(run(&format!("{}r::|/a", A)).unwrap(), fixed("fixed<32,16>", &[], &[98304]));
    assert_eq!(run(&format!("{}r::&/a", A)).unwrap(), fixed("fixed<32,16>", &[], &[-147456]));
    assert_eq!(run(&format!("{}r::+/a*a", A)).unwrap(), fixed("fixed<32,16>", &[], &[495616]));
}

#[test]
fn negation_and_floor() {
    assert_eq!(run(&format!("{}r::-a", A)).unwrap(), fixed("fixed<32,16>", &[3], &[-32768, -98304, 147456]));
    assert_eq!(run(&format!("{}r::_a", A)).unwrap(), ints(&[3], &[0, 1, -3]));
}

#[test]
fn mixing_is_an_error() {
    let e = run(&format!("{}r::a+0.5", A)).unwrap_err();
    assert!(e.starts_with("type") && e.contains("cannot mix fixed<32,16> with a float"), "{}", e);
    let e = run(&format!("{}b::.fixed(\"fixed<16,8>\";[1 2 3])\nr::a+b", A)).unwrap_err();
    assert!(e.starts_with("type") && e.contains("cannot mix fixed<32,16> and fixed<16,8>"), "{}", e);
    let e = run(&format!("{}r::*/a", A)).unwrap_err();
    assert!(e.starts_with("type"), "{}", e);
    let e = run(&format!("{}r::a+[1 2]", A)).unwrap_err();
    assert!(e.starts_with("shape"), "{}", e);
}

#[test]
fn division() {
    let e = run(&format!("{}r::a%(a-a)", A)).unwrap_err();
    assert!(e.starts_with("range") && e.contains("division by zero"), "{}", e);
    let e = run("a::.fixed(\"fixed<64,32>\";1)\nr::a%a").unwrap_err();
    assert!(e.contains("W <= 32"), "{}", e);
}

#[test]
fn the_value_is_what_a_compile_time_global_holds() {
    // the typed program sees b as a fixed-point constant
    let v = run("a::.fixed(\"fixed<32,16>\";2)\nb::a*a\nf::{x*b}\n.entry(\"f\";\"fixed<32,16> -> fixed<32,16>\")\nr::b").unwrap();
    assert_eq!(v, fixed("fixed<32,16>", &[], &[4 << 16]));
}

#[test]
fn the_compiler_sees_the_computed_constant() {
    use klong_static::compile::compile_program;
    let text = "a::.fixed(\"fixed<32,16>\";0.5)\nb::a+a\nf::{x*b}\n.entry(\"f\";\"fixed<32,16> -> fixed<32,16>\")\n";
    let c = match compile_program("t.kg", Some(text), 32, None) {
        Ok(c) => c,
        Err(f) => panic!("{:?}", f.render()),
    };
    let b = c.values.get("b").expect("b is a global");
    assert_eq!(format!("{:?}", b.value), "Some(I(65536))");
    let emitted = klong_static::compile::emit_program("t.kg", None, Some(text), 32, 32, false, None, None);
    assert!(emitted.is_ok());
}
