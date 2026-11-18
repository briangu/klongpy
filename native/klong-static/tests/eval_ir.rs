//! The typed-IR evaluator (src/eval_ir.rs). Three kinds of proof:
//!
//!  * `evaluator_agrees_with_the_interpreter`: programs of every node kind run on the native interpreter and on the evaluator (the
//!    independent reference: the interpreter has its own implementation of every operator), over fixed inputs and edge values;
//!  * pinned values and traps (`fixed_point_*`, `traps_*`, `integers_wrap_*`): what klong_rt.h / the Python evaluator specify;
//!  * `python_corpus`: when `KG_EVAL_CASES` names a file, every call of that file (programs of the
//!    operator table with the answers of klongpy/static/eval_ir.py, bit for bit) is replayed; skipped otherwise.
use klong_static::compile::compile_program;
use klong_static::eval_ir::{Arr, Data, IrEvaluator, Trap};
use klong_static::host::{call_entry, load_program, HostValue};
use std::collections::HashMap;

fn compile(text: &str, bits: u32) -> klong_static::ir::Program {
    match compile_program("t.kg", Some(text), bits, None) {
        Ok(c) => c.program,
        Err(f) => panic!("{}\n{}", f.render().join("\n"), text),
    }
}

fn run(text: &str, bits: u32, name: &str, args: &[Arr]) -> Result<Arr, Trap> {
    let p = compile(text, bits);
    let mut ev = IrEvaluator::new(&p);
    let e = p.entries.iter().find(|e| e.decl.name == name).unwrap();
    ev.run(e, args)
}

fn i(v: i64) -> Arr {
    Arr::int(v)
}
fn f(v: f64) -> Arr {
    Arr::flt(v)
}
fn iv(v: &[i64]) -> Arr {
    Arr::ints(vec![v.len()], v.to_vec())
}
fn fv(v: &[f64]) -> Arr {
    Arr::flts(vec![v.len()], v.to_vec())
}
fn ints(a: &Arr) -> Vec<i64> {
    a.i64s().into_owned()
}
fn flts(a: &Arr) -> Vec<f64> {
    a.f64s().into_owned()
}

fn prog(body: &str, sig: &str) -> String {
    format!("f::{{{}}}\n.entry(\"f\";\"{}\")\n", body, sig)
}

// ---------------------------------------------------------------------------
// the interpreter against the evaluator
// ---------------------------------------------------------------------------

fn to_host(a: &Arr) -> HostValue {
    match &a.data {
        Data::I(d) if a.shape.is_empty() => HostValue::Int(d[0]),
        Data::F(d) if a.shape.is_empty() => HostValue::Float(d[0]),
        Data::I(d) => HostValue::Ints(a.shape.clone(), d.clone()),
        Data::F(d) => HostValue::Floats(a.shape.clone(), d.clone()),
    }
}

fn same(a: &HostValue, b: &Arr, what: &str) {
    let want = to_host(b);
    let close = |x: f64, y: f64| (x.is_nan() && y.is_nan()) || x == y || (x - y).abs() <= 1e-12 * x.abs().max(y.abs());
    match (a, &want) {
        (HostValue::Int(x), HostValue::Int(y)) => assert_eq!(x, y, "{}", what),
        (HostValue::Float(x), HostValue::Float(y)) => assert!(close(*x, *y), "{}: {} vs {}", what, x, y),
        (HostValue::Ints(s, x), HostValue::Ints(t, y)) => assert_eq!((s, x), (t, y), "{}", what),
        (HostValue::Floats(s, x), HostValue::Floats(t, y)) => {
            assert_eq!(s, t, "{}", what);
            assert!(x.iter().zip(y).all(|(p, q)| close(*p, *q)), "{}: {:?} vs {:?}", what, x, y);
        }
        _ => panic!("{}: interpreter {:?}, evaluator {:?}", what, a, want),
    }
}

#[test]
fn evaluator_agrees_with_the_interpreter() {
    let fe = [0.0, -1.0, 0.5, 2.0, -2.5, 3.0, 123456.75, 1.5];
    let ie = [0i64, 1, -1, 2, -2, 7, -7, 1000];
    let f4 = fv(&[1.5, -2.0, 0.25, 4.0]);
    let i4 = iv(&[3, -1, 4, -9]);
    let m23 = Arr::flts(vec![2, 3], vec![1.0, 2.0, 3.0, -4.0, 0.5, 6.0]);
    let i23 = Arr::ints(vec![2, 3], vec![1, 2, 3, -4, 5, 6]);
    // (body, signature, argument sets)
    let mut cases: Vec<(&str, &str, Vec<Vec<Arr>>)> = Vec::new();
    for body in ["x+y", "x-y", "x*y", "x|y", "x&y", "x<y", "x>y", "x=y", "x~y", "x%y", "x!y"] {
        let mut sets = Vec::new();
        for a in fe {
            for b in fe {
                if (body.ends_with("!y") || body.ends_with("%y")) && b == 0.0 {
                    continue;
                }
                sets.push(vec![f(a), f(b)]);
            }
        }
        cases.push((body, "float float -> float", sets.clone()));
        if body.contains('<') || body.contains('>') || body.contains('=') || body.contains('~') {
            cases.pop();
            cases.push((body, "float float -> int", sets));
        }
    }
    for body in ["x+y", "x-y", "x*y", "x|y", "x&y", "x:%y", "x!y"] {
        let mut sets = Vec::new();
        for a in ie {
            for b in ie {
                if (body.contains(":%") || body.ends_with("!y")) && b == 0 {
                    continue;
                }
                sets.push(vec![i(a), i(b)]);
            }
        }
        cases.push((body, "int int -> int", sets));
    }
    cases.push(("x<y", "int int -> int", ie.iter().flat_map(|a| ie.iter().map(move |b| vec![i(*a), i(*b)])).collect()));
    cases.push(("x%y", "int int -> float", ie.iter().flat_map(|a| ie.iter().filter(|b| **b != 0).map(move |b| vec![i(*a), i(*b)])).collect()));
    cases.push(("-x", "float[4] -> float[4]", vec![vec![f4.clone()]]));
    cases.push(("_x", "float[4] -> int[4]", vec![vec![f4.clone()], vec![fv(&[-0.5, 2.5, 1e30, -1e30])]]));
    cases.push(("%x", "float[4] -> float[4]", vec![vec![f4.clone()]]));
    cases.push(("~x", "int[4] -> int[4]", vec![vec![i4.clone()], vec![iv(&[0, 0, 1, 0])]]));
    cases.push(("x+y", "float[4] float -> float[4]", vec![vec![f4.clone(), f(2.0)]]));
    cases.push(("x*y", "float[2,3] float[2] -> float[2,3]", vec![vec![m23.clone(), fv(&[2.0, -1.0])]]));
    cases.push(("x@y", "float[4] int -> float", (-4..4).map(|k| vec![f4.clone(), i(k)]).collect()));
    cases.push(("x@y", "float[4] int[3] -> float[3]", vec![vec![f4.clone(), iv(&[0, -1, 3])]]));
    cases.push(("x@y", "int[2,3] int -> int[3]", vec![vec![i23.clone(), i(-1)]]));
    cases.push(("3#x", "int[4] -> int[3]", vec![vec![i4.clone()]]));
    cases.push(("6#x", "int[4] -> int[6]", vec![vec![i4.clone()]]));
    cases.push(("(-3)#x", "int[4] -> int[3]", vec![vec![i4.clone()]]));
    cases.push(("(-6)#x", "int[4] -> int[6]", vec![vec![i4.clone()]]));
    cases.push(("1_x", "int[4] -> int[3]", vec![vec![i4.clone()]]));
    cases.push(("(-1)_x", "int[4] -> int[3]", vec![vec![i4.clone()]]));
    cases.push(("x:+y", "int int[4] -> int[4]", (-5..6).map(|k| vec![i(k), i4.clone()]).collect()));
    cases.push(("|x", "float[2,3] -> float[2,3]", vec![vec![m23.clone()]]));
    cases.push(("+x", "float[2,3] -> float[3,2]", vec![vec![m23.clone()]]));
    cases.push(("[2 3]:^x", "float[4] -> float[2,3]", vec![vec![f4.clone()]]));
    cases.push(("x,y", "float[4] float[4] -> float[8]", vec![vec![f4.clone(), f4.clone()]]));
    cases.push(("x,y", "float float -> float[2]", vec![vec![f(1.0), f(2.5)]]));
    cases.push(("x,y", "int[4] float -> float[5]", vec![vec![i4.clone(), f(2.5)]]));
    cases.push(("[;x;y]", "float float -> float[2]", vec![vec![f(1.0), f(2.0)]]));
    cases.push(("x:=y,1", "float[4] float -> float[4]", vec![vec![f4.clone(), f(9.0)]]));
    cases.push(("x:=y,z", "float[4] float int -> float[4]", (-4..4).map(|k| vec![f4.clone(), f(9.0), i(k)]).collect()));
    cases.push(("+/x", "float[4] -> float", vec![vec![f4.clone()]]));
    cases.push(("+\\x", "float[4] -> float[4]", vec![vec![f4.clone()]]));
    cases.push(("0+\\x", "float[4] -> float[5]", vec![vec![f4.clone()]]));
    cases.push(("|/x", "int[4] -> int", vec![vec![i4.clone()]]));
    cases.push(("x{x+y}'y", "int[4] int[4] -> int[4]", vec![vec![i4.clone(), i4.clone()]]));
    cases.push(("{x*2.0}'x", "float[4] -> float[4]", vec![vec![f4.clone()]]));
    cases.push(("{y-x}:'x", "float[4] -> float[3]", vec![vec![f4.clone()]]));
    cases.push(("3{x*2.0}:*x", "float -> float", vec![vec![f(1.5)]]));
    cases.push(("{x<100}{x*2}:~x", "int -> int", vec![vec![i(3)], vec![i(500)]]));
    cases.push(("x^y", "int int -> float", [(2, 10), (3, 4), (-2, 3), (5, 0)].iter().map(|(a, b)| vec![i(*a), i(*b)]).collect()));
    cases.push(("x^y", "float float -> float", [(2.0, 0.5), (2.0, -1.0), (9.0, 0.5)].iter().map(|(a, b)| vec![f(*a), f(*b)]).collect()));
    cases.push((":[x>0;x*2;0-x]", "int -> int", vec![vec![i(3)], vec![i(-4)]]));
    cases.push(("[a];a::x*2;a+1", "int -> int", vec![vec![i(4)]]));
    cases.push(("*x", "float[4] -> float", vec![vec![f4.clone()]]));
    cases.push(("!5", "-> int[5]", vec![vec![]]));
    for (body, sig, sets) in cases {
        let text = prog(body, sig);
        let mut interp = load_program("t.kg", Some(&text), HashMap::new()).unwrap_or_else(|e| panic!("{}: {}", body, e.message));
        let p = compile(&text, 64);
        let mut ev = IrEvaluator::new(&p);
        let entry = &p.entries[0];
        for args in sets {
            let hv: Vec<HostValue> = args.iter().map(to_host).collect();
            let what = format!("{{{}}} {} {:?}", body, sig, hv);
            let r = ev.run(entry, &args).unwrap_or_else(|t| panic!("{}: trap {}", what, t));
            let h = call_entry(&mut interp, "f", &hv, None, false).unwrap_or_else(|e| panic!("{}: {}", what, e.message));
            same(&h, &r, &what);
        }
    }
}

// ---------------------------------------------------------------------------
// pinned values
// ---------------------------------------------------------------------------

#[test]
fn integers_wrap_at_the_width_of_kg_int() {
    let t = prog("x+y", "int int -> int");
    assert_eq!(ints(&run(&t, 32, "f", &[i(2147483647), i(1)]).unwrap()), vec![-2147483648]);
    assert_eq!(ints(&run(&t, 64, "f", &[i(2147483647), i(1)]).unwrap()), vec![2147483648]);
    assert_eq!(ints(&run(&t, 64, "f", &[i(i64::MAX), i(1)]).unwrap()), vec![i64::MIN]);
    let t = prog("x*y", "int int -> int");
    assert_eq!(ints(&run(&t, 32, "f", &[i(65536), i(65536)]).unwrap()), vec![0]);
    assert_eq!(ints(&run(&t, 32, "f", &[i(46341), i(46341)]).unwrap()), vec![-2147479015]);
    let t = prog("-x", "int -> int");
    assert_eq!(ints(&run(&t, 32, "f", &[i(-2147483648)]).unwrap()), vec![-2147483648]);
}

#[test]
fn integer_division_truncates_and_rem_takes_the_sign_of_the_dividend() {
    let d = prog("x:%y", "int int -> int");
    let r = prog("x!y", "int int -> int");
    for (a, b, q, m) in [(7, 2, 3, 1), (-7, 2, -3, -1), (7, -2, -3, 1), (-7, -2, 3, -1), (0, 5, 0, 0), (1, 5, 0, 1)] {
        assert_eq!(ints(&run(&d, 32, "f", &[i(a), i(b)]).unwrap()), vec![q], "{} :% {}", a, b);
        assert_eq!(ints(&run(&r, 32, "f", &[i(a), i(b)]).unwrap()), vec![m], "{} ! {}", a, b);
    }
    // Python's evaluator answers 1 and -9223372036854775803 for 5 :% MIN and 5 ! MIN at 64 bits (abs(MIN) overflows); C and this give 0 and 5
    assert_eq!(ints(&run(&d, 64, "f", &[i(5), i(i64::MIN)]).unwrap()), vec![0]);
    assert_eq!(ints(&run(&r, 64, "f", &[i(5), i(i64::MIN)]).unwrap()), vec![5]);
    let fr = prog("x!y", "float float -> float");
    assert_eq!(flts(&run(&fr, 32, "f", &[f(-7.5), f(2.0)]).unwrap()), vec![-1.5]);
    assert_eq!(flts(&run(&fr, 32, "f", &[f(7.5), f(-2.0)]).unwrap()), vec![1.5]);
}

#[test]
fn floor_saturates_and_nan_is_zero() {
    let t = prog("_x", "float[6] -> int[6]");
    let x = fv(&[2.5, -2.5, f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1e30]);
    assert_eq!(ints(&run(&t, 32, "f", &[x.clone()]).unwrap()), vec![2, -3, 0, 2147483647, -2147483648, 2147483647]);
    assert_eq!(ints(&run(&t, 64, "f", &[x]).unwrap()), vec![2, -3, 0, i64::MAX, i64::MIN, 1000000000000000019884624838656i128.min(i64::MAX as i128) as i64]);
}

#[test]
fn match_is_isclose_for_float_scalars_and_exact_otherwise() {
    let t = prog("x~y", "float float -> int");
    assert_eq!(ints(&run(&t, 32, "f", &[f(1.0), f(1.000001)]).unwrap()), vec![1]);
    assert_eq!(ints(&run(&t, 32, "f", &[f(1.0), f(1.0001)]).unwrap()), vec![0]);
    assert_eq!(ints(&run(&t, 32, "f", &[f(f64::NAN), f(f64::NAN)]).unwrap()), vec![0]);
    assert_eq!(ints(&run(&t, 32, "f", &[f(f64::INFINITY), f(f64::INFINITY)]).unwrap()), vec![1]);
    let t = prog("x~y", "float[2] float[2] -> int");
    assert_eq!(ints(&run(&t, 32, "f", &[fv(&[1.0, 2.0]), fv(&[1.0, 2.000001])]).unwrap()), vec![0]);
    assert_eq!(ints(&run(&t, 32, "f", &[fv(&[1.0, 2.0]), fv(&[1.0, 2.0])]).unwrap()), vec![1]);
}

#[test]
fn traps_have_codes_one_and_two() {
    let t = prog("x@y", "int[4] int -> int");
    for k in [4, -5, 100] {
        let e = run(&t, 32, "f", &[iv(&[1, 2, 3, 4]), i(k)]).unwrap_err();
        assert_eq!((e.code, e.message.as_str()), (1, "index out of range"), "index {}", k);
    }
    assert_eq!(ints(&run(&t, 32, "f", &[iv(&[1, 2, 3, 4]), i(-4)]).unwrap()), vec![1]);
    let e = run(&prog("x:%y", "int int -> int"), 32, "f", &[i(1), i(0)]).unwrap_err();
    assert_eq!((e.code, e.message.as_str()), (2, "integer division by zero"));
    let e = run(&prog("x!y", "int int -> int"), 32, "f", &[i(1), i(0)]).unwrap_err();
    assert_eq!(e.code, 2);
    let e = run(&prog("x:=y,z", "int[4] int int -> int[4]"), 32, "f", &[iv(&[1, 2, 3, 4]), i(1), i(4)]).unwrap_err();
    assert_eq!(e.code, 1);
    // floats do not trap
    let q = run(&prog("x%y", "float float -> float"), 32, "f", &[f(1.0), f(0.0)]).unwrap();
    assert_eq!(flts(&q), vec![f64::INFINITY]);
}

fn fixed_text(body: &str, sig: &str) -> String {
    prog(body, sig)
}

#[test]
fn fixed_point_follows_the_reference_semantics() {
    let t = "fixed<8,4>";
    let two = |b: &str, res: &str| fixed_text(b, &format!("{t}[3] {t}[3] -> {res}[3]"));
    let a = iv(&[127, -128, 5]);
    let b = iv(&[1, 1, -3]);
    // add wraps to W bits, mul is floor(a*b/2^F), div is floor(a*2^F/b)
    assert_eq!(ints(&run(&two("x+y", t), 32, "f", &[a.clone(), b.clone()]).unwrap()), vec![-128, -127, 2]);
    assert_eq!(ints(&run(&two("x-y", t), 32, "f", &[a.clone(), b.clone()]).unwrap()), vec![126, 127, 8]);
    assert_eq!(ints(&run(&two("x*y", t), 32, "f", &[a.clone(), b.clone()]).unwrap()), vec![7, -8, -1]);
    assert_eq!(ints(&run(&two("x%y", t), 32, "f", &[a.clone(), b.clone()]).unwrap()), vec![wrap8(127 * 16), wrap8(-128 * 16), wrap8(div_floor(5 * 16, -3))]);
    assert_eq!(ints(&run(&two("x<y", "int"), 32, "f", &[a.clone(), b.clone()]).unwrap()), vec![0, 1, 0]);
    assert_eq!(ints(&run(&two("x|y", t), 32, "f", &[a.clone(), b.clone()]).unwrap()), vec![127, 1, 5]);
    let n = fixed_text("-x", &format!("{t}[3] -> {t}[3]"));
    assert_eq!(ints(&run(&n, 32, "f", &[a.clone()]).unwrap()), vec![-127, -128, -5]);
    let fl = fixed_text("_x", &format!("{t}[3] -> int[3]"));
    assert_eq!(ints(&run(&fl, 32, "f", &[iv(&[17, -1, 15])]).unwrap()), vec![1, -1, 0]);
    // division by zero traps with code 2
    let e = run(&fixed_text("x%y", &format!("{t} {t} -> {t}")), 32, "f", &[i(5), i(0)]).unwrap_err();
    assert_eq!((e.code, e.message.as_str()), (2, "fixed-point division by zero"));
}

fn wrap8(v: i64) -> i64 {
    (((v + 128) % 256) + 256) % 256 - 128
}
fn div_floor(a: i64, b: i64) -> i64 {
    let q = a / b;
    if (a % b != 0) && ((a < 0) != (b < 0)) {
        q - 1
    } else {
        q
    }
}

#[test]
fn conversions_between_float_int_and_fixed() {
    let text = "f::{.fixed(\"fixed<16,8>\";x)}\ng::{.float(x)}\nh::{.fixed(\"fixed<32,16>\";x)}\nk::{.fixed(\"fixed<16,8>\";x)}\n\
                .entry(\"f\";\"float[6] -> fixed<16,8>[6]\")\n.entry(\"g\";\"fixed<16,8>[6] -> float[6]\")\n\
                .entry(\"h\";\"fixed<16,8>[6] -> fixed<32,16>[6]\")\n.entry(\"k\";\"fixed<32,16>[6] -> fixed<16,8>[6]\")\n";
    let r = run(text, 32, "f", &[fv(&[1.5, -1.5, 0.00390625, 127.99609375, -128.0, 200.0])]).unwrap();
    assert_eq!(ints(&r), vec![384, -384, 1, 32767, -32768, -14336]);
    let raw = iv(&[-32768, -1, 0, 1, 255, 32767]);
    assert_eq!(flts(&run(text, 32, "g", &[raw.clone()]).unwrap()), vec![-128.0, -0.00390625, 0.0, 0.00390625, 0.99609375, 127.99609375]);
    assert_eq!(ints(&run(text, 32, "h", &[raw]).unwrap()), vec![-8388608, -256, 0, 256, 65280, 8388352]);
    assert_eq!(ints(&run(text, 32, "k", &[iv(&[-70000, -1, 0, 1, 255, 70000])]).unwrap()), vec![-274, -1, 0, 0, 0, 273]);
}

#[test]
fn state_persists_between_runs_and_resets() {
    let text = "acc::.fixed(\"fixed<32,16>\";0)\nstep::{acc::acc+x;acc}\n.entry(\"step\";\"fixed<32,16> -> fixed<32,16>\")\n";
    let p = compile(text, 32);
    let mut ev = IrEvaluator::new(&p);
    let e = &p.entries[0];
    let mut got = Vec::new();
    for raw in [65536, 100, -3, 2147483647] {
        got.push(ev.run(e, &[i(raw)]).unwrap().i64s()[0]);
    }
    assert_eq!(got, vec![65536, 65636, 65633, -2147418016]);
    assert_eq!(ints(&ev.state["acc"]), vec![-2147418016]);
    ev.reset();
    assert_eq!(ints(&ev.state["acc"]), vec![0]);
}

#[test]
fn an_extern_goes_through_the_callback() {
    let text = ".extern(\"read.adc\";\"int -> float\")\nf::{read.adc(x)*2.0}\n.entry(\"f\";\"int -> float\")\n";
    let p = compile(text, 32);
    let mut ev = IrEvaluator::new(&p);
    let e = &p.entries[0];
    let err = ev.run(e, &[i(3)]).unwrap_err();
    assert_eq!(err.code, 3);
    ev.extern_cb = Some(Box::new(|name, args| {
        assert_eq!(name, "read.adc");
        Ok(Arr::flt(args[0].i64s()[0] as f64 + 0.25))
    }));
    assert_eq!(flts(&ev.run(e, &[i(3)]).unwrap()), vec![6.5]);
}

// ---------------------------------------------------------------------------
// the corpus of the Python evaluator
// ---------------------------------------------------------------------------

fn dec(s: &str) -> Arr {
    let hex = |h: &str| if h == "nan" { f64::NAN } else { f64::from_bits(u64::from_str_radix(h, 16).unwrap()) };
    if let Some(v) = s.strip_prefix("i:") {
        return Arr::int(v.parse().unwrap());
    }
    if let Some(v) = s.strip_prefix("f:") {
        return Arr::flt(hex(v));
    }
    let (head, data) = s.split_once("]:").unwrap();
    let shape: Vec<usize> = head[2..].split(',').map(|d| d.parse().unwrap()).collect();
    let items: Vec<&str> = if data.is_empty() { vec![] } else { data.split(',').collect() };
    if head.starts_with('I') {
        Arr::ints(shape, items.iter().map(|x| x.parse().unwrap()).collect())
    } else {
        Arr::flts(shape, items.iter().map(|x| hex(x)).collect())
    }
}

fn enc(a: &Arr) -> String {
    let hexf = |x: f64| if x.is_nan() { "nan".to_string() } else { format!("{:016x}", x.to_bits()) };
    if a.shape.is_empty() {
        return match &a.data {
            Data::I(d) => format!("i:{}", d[0]),
            Data::F(d) => format!("f:{}", hexf(d[0])),
        };
    }
    let dims: Vec<String> = a.shape.iter().map(|d| d.to_string()).collect();
    match &a.data {
        Data::I(d) => format!("I[{}]:{}", dims.join(","), d.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",")),
        Data::F(d) => format!("F[{}]:{}", dims.join(","), d.iter().map(|x| hexf(*x)).collect::<Vec<_>>().join(",")),
    }
}

#[test]
fn python_corpus() {
    let Ok(path) = std::env::var("KG_EVAL_CASES") else {
        return;
    };
    let Ok(text) = std::fs::read_to_string(&path) else {
        eprintln!("python_corpus: {} not found, skipped", path);
        return;
    };
    let mut lines = text.lines();
    let (mut calls, mut bad) = (0, Vec::new());
    let mut cur: Option<(klong_static::ir::Program, std::collections::HashMap<String, usize>)> = None;
    let mut ev: Option<IrEvaluator<&klong_static::ir::Program>> = None;
    let mut leaked: Vec<&'static klong_static::ir::Program> = Vec::new();
    let _ = &mut cur;
    let _ = &mut leaked;
    while let Some(l) = lines.next() {
        if let Some(h) = l.strip_prefix("PROG ") {
            let bits: u32 = h.split(' ').nth(1).unwrap().parse().unwrap();
            let src = lines.next().unwrap().replace("\\n", "\n").replace("\\\\", "\\");
            let p: &'static klong_static::ir::Program = Box::leak(Box::new(compile(&src, bits)));
            ev = Some(IrEvaluator::new(p));
            leaked.push(p);
        } else if let Some(c) = l.strip_prefix("CALL ") {
            let (lhs, want) = c.split_once(" => ").unwrap();
            let mut parts = lhs.split(' ');
            let name = parts.next().unwrap();
            let args: Vec<Arr> = parts.map(dec).collect();
            let evr = ev.as_mut().unwrap();
            let p = *leaked.last().unwrap();
            let e = p.entries.iter().find(|e| e.decl.name == name).unwrap();
            let got = match evr.run(e, &args) {
                Ok(r) => enc(&r),
                Err(t) if t.code == 0 => "PYERR".to_string(),
                Err(t) => format!("TRAP {}", t.code),
            };
            calls += 1;
            if got != want {
                bad.push(format!("{} {} => python {} rust {}", name, lhs, want, got));
            }
        }
    }
    eprintln!("python_corpus: {} calls replayed, {} differ", calls, bad.len());
    for b in bad.iter().take(30) {
        eprintln!("DIFF {}", b);
    }
    assert!(bad.is_empty(), "{} differences", bad.len());
}
