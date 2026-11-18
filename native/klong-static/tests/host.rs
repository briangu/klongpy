//! The host half of the static profile (src/host.rs): `load_program`, `call_entry`, `set_param` on the native interpreter, and the
//! typed-IR evaluator behind fixed-point programs. Proof:
//!
//!  * the reference programs of examples/embedded with their .expect files, tests/static/hw/*.kg, the inline programs of
//!    tests/static/test_extern_entry.py and test_fixed.py, with the values and the error messages those tests pin (below, no Python);
//!  * `python_scenarios`: when `KG_HOST_SCENARIOS` names a file, every scenario of that file is
//!    replayed (answers recorded from klongpy.static.host on the Python engine, floats bit for bit, errors by class and message).
use klong_core::value::{NativeFn, Value};
use klong_core::Kind;
use klong_static::compile::stage_runtime;
use klong_static::host::{call_entry, conform, load_program, set_param, HostFn, HostValue, Loaded};
use klong_static::sig::parse_type;
use std::collections::HashMap;
use std::rc::Rc;

const ADC: &str = ".extern(\"read.adc\";\"int -> float\")\n";

fn konst(v: HostValue) -> HostFn {
    let f: NativeFn = Rc::new(move |_ip, _a| Ok(v.to_value()));
    f
}

fn externs(list: &[(&str, HostValue)]) -> HashMap<String, HostFn> {
    list.iter().map(|(k, v)| (k.to_string(), konst(v.clone()))).collect()
}

fn load(text: &str, ext: &[(&str, HostValue)]) -> Loaded {
    match load_program("prog.kg", Some(text), externs(ext)) {
        Ok(l) => l,
        Err(e) => panic!("load: {}", e.message),
    }
}

fn call(p: &mut Loaded, name: &str, args: &[HostValue]) -> HostValue {
    match call_entry(p, name, args, None, false) {
        Ok(v) => v,
        Err(e) => panic!("{}: {}", name, e.message),
    }
}

fn err(p: &mut Loaded, name: &str, args: &[HostValue]) -> (Kind, String) {
    match call_entry(p, name, args, None, false) {
        Ok(v) => panic!("{} should fail, gave {:?}", name, v),
        Err(e) => (e.kind, e.message.to_string()),
    }
}

fn fl(v: f64) -> HostValue {
    HostValue::Float(v)
}
fn fv(v: &[f64]) -> HostValue {
    HostValue::floats(&[v.len()], v.to_vec())
}
fn iv(v: &[i64]) -> HostValue {
    HostValue::ints(&[v.len()], v.to_vec())
}

// ---------------------------------------------------------------------------
// the reference programs and their .expect files
// ---------------------------------------------------------------------------

fn nums(v: &Value) -> Vec<f64> {
    match HostValue::from_value(v) {
        HostValue::Int(i) => vec![i as f64],
        HostValue::Float(f) => vec![f],
        HostValue::Ints(_, d) => d.iter().map(|x| *x as f64).collect(),
        HostValue::Floats(_, d) => d,
        other => panic!("not a number: {:?}", other),
    }
}

/// The steps of an .expect file: (externs, [(expression, expected text)]).
fn expect(name: &str) -> (Vec<(String, HostValue)>, Vec<(String, String)>) {
    let text = std::fs::read_to_string(format!("../../examples/embedded/{}.expect", name)).unwrap();
    let (mut ext, mut steps) = (Vec::new(), Vec::new());
    for line in text.lines() {
        let t = line.trim_start_matches('#').trim();
        if let Some(rest) = line.strip_prefix('#').and_then(|l| l.trim().strip_prefix("extern ")) {
            let (k, v) = rest.split_once('=').unwrap();
            let v = v.trim();
            let hv = if v.contains('.') || v.contains('e') { HostValue::Float(v.parse().unwrap()) } else { HostValue::Int(v.parse().unwrap()) };
            ext.push((k.trim().to_string(), hv));
        } else if !line.trim().is_empty() && !line.starts_with('#') {
            let (e, w) = line.split_once(" => ").unwrap();
            steps.push((e.trim().to_string(), w.trim().to_string()));
        }
        let _ = t;
    }
    (ext, steps)
}

#[test]
fn the_reference_programs_give_the_values_of_their_expect_files() {
    for name in ["mavg", "pid", "fir"] {
        let (ext, steps) = expect(name);
        let mut p = load_program(&format!("../../examples/embedded/{}.kg", name), None, externs(&ext.iter().map(|(k, v)| (k.as_str(), v.clone())).collect::<Vec<_>>())).unwrap_or_else(|e| panic!("{}: {}", name, e.message));
        let mut literal = stage_runtime();
        assert!(!steps.is_empty());
        for (expr, want) in steps {
            let got = p.rt.run(&expr, "expect").unwrap_or_else(|e| panic!("{}: {}: {}", name, expr, e.message)).unwrap();
            let want = literal.run(&want, "want").unwrap().unwrap();
            let (g, w) = (nums(&got), nums(&want));
            assert_eq!(g.len(), w.len(), "{}: {}", name, expr);
            assert!(g.iter().zip(&w).all(|(a, b)| (a - b).abs() <= 1e-9), "{}: {}: {:?} vs {:?}", name, expr, g, w);
        }
    }
}

#[test]
fn call_entry_on_the_reference_programs() {
    // pid: the values of examples/embedded/pid.expect through the entry
    let mut p = load_program("../../examples/embedded/pid.kg", None, externs(&[("read.adc", fl(0.3)), ("write.pwm", HostValue::Int(0))])).unwrap();
    let r = call(&mut p, "step", &[fl(0.31)]);
    assert!(matches!(r, HostValue::Float(v) if (v - 0.12005).abs() < 1e-12), "{:?}", r);
    let r = call(&mut p, "step", &[fl(0.31)]);
    assert!(matches!(r, HostValue::Float(v) if (v - 0.0201).abs() < 1e-12), "{:?}", r);
    assert!(matches!(call(&mut p, "step", &[fl(0.5)]), HostValue::Float(v) if v == 1.0));
    // an int argument is converted to the float parameter
    assert!(matches!(call(&mut p, "step", &[HostValue::Int(1)]), HostValue::Float(_)));
    // fir: 256 ones give 241 ones (taps sum to 1)
    let mut fir = load_program("../../examples/embedded/fir.kg", None, HashMap::new()).unwrap();
    match call(&mut fir, "run", &[fv(&[1.0; 256])]) {
        HostValue::Floats(s, d) => {
            assert_eq!(s, vec![241]);
            assert!(d.iter().all(|x| (x - 1.0).abs() < 1e-12));
        }
        o => panic!("{:?}", o),
    }
    // mavg keeps state between calls
    let mut m = load_program("../../examples/embedded/mavg.kg", None, HashMap::new()).unwrap();
    assert_eq!(call(&mut m, "step", &[fl(8.0)]), fl(2.0));
}

// ---------------------------------------------------------------------------
// tests/static/test_extern_entry.py: TestHostHelpers, TestSameMistakeHostAndChecker
// ---------------------------------------------------------------------------

const STEP: &str = ".extern(\"read.adc\";\"int -> float\")\nacc::0.0\nstep::{acc::acc+x*read.adc(0);acc}\n.entry(\"step\";\"float -> float\")\n";

#[test]
fn load_program_and_call_entry() {
    let mut p = load(STEP, &[("read.adc", fl(0.5))]);
    assert_eq!(call(&mut p, "step", &[HostValue::Int(2)]), fl(1.0)); // the int argument is converted
    assert_eq!(call(&mut p, "step", &[fl(2.0)]), fl(2.0)); // state is kept between calls
    assert_eq!(call(&mut p, "step", &[fl(0.0)]), fl(2.0));
    assert_eq!(p.get("acc"), Some(fl(2.0)));
}

#[test]
fn call_entry_checks_arguments_and_result() {
    let mut p = load(STEP, &[("read.adc", fl(0.5))]);
    assert_eq!(err(&mut p, "step", &[fv(&[1.0, 2.0])]), (Kind::Type, "argument 1 of entry step must be float, got float[2] (declared \"float -> float\")".into()));
    assert_eq!(err(&mut p, "step", &[]), (Kind::Type, "entry step is declared \"float -> float\" and takes 1 argument(s), got 0".into()));
    assert_eq!(err(&mut p, "step", &[fl(1.0), fl(2.0)]).1, "entry step is declared \"float -> float\" and takes 1 argument(s), got 2");
    assert_eq!(err(&mut p, "nosuch", &[fl(1.0)]), (Kind::Name, "nosuch is not an entry of the program".into()));
    let mut q = load("f::{x%2}\n.entry(\"f\";\"int -> int\")", &[]); // the function returns a float
    assert_eq!(err(&mut q, "f", &[HostValue::Int(3)]), (Kind::Type, "the result of entry f must be int, got a float (declared \"int -> int\")".into()));
    assert_eq!(err(&mut q, "f", &[fl(3.0)]), (Kind::Type, "argument 1 of entry f must be int, got a float (declared \"int -> int\")".into()));
}

#[test]
fn a_missing_stand_in_is_a_system_error_naming_the_extern() {
    let e = load_program("p.kg", Some(&format!("{}v::read.adc(0)\n", ADC)), HashMap::new()).err().expect("fails");
    assert_eq!(e.kind, Kind::System);
    assert_eq!(e.message, "extern read.adc has no stand-in on the host");
    assert!(e.hint.as_deref().unwrap_or("").contains("klong_cfun_register"));
    // the same file with the stand-in bound before it runs
    let mut p = load_program("p.kg", Some(&format!("{}v::read.adc(0)\n", ADC)), externs(&[("read.adc", fl(9.0))])).unwrap();
    assert_eq!(p.get("v"), Some(fl(9.0)));
    // a call of the entry without one
    let mut q = load(&format!("{}step::{{read.adc(x)}}\n.entry(\"step\";\"int -> float\")\n", ADC), &[]);
    assert_eq!(err(&mut q, "step", &[HostValue::Int(1)]).0, Kind::System);
}

#[test]
fn several_entries_with_different_shapes() {
    let mut p = load("f::{x*2}\n.entry(\"f\";\"float[64] -> float[64]\")\n.entry(\"f\";\"float[8] -> float[8]\")\n.entry(\"f\";\"float[8]->float[8]\")", &[]);
    assert_eq!(call(&mut p, "f", &[fv(&[1.0; 8])]), fv(&[2.0; 8]));
    assert_eq!(call(&mut p, "f", &[fv(&[1.0; 64])]), fv(&[2.0; 64]));
    // the error is the one of the last candidate tried
    assert_eq!(err(&mut p, "f", &[fv(&[1.0; 5])]), (Kind::Shape, "argument 1 of entry f must be float[8], got float[5] (declared \"float[8] -> float[8]\")".into()));
    let r = call_entry(&mut p, "f", &[fv(&[2.0; 8])], Some("float[8] -> float[8]"), false).unwrap();
    assert_eq!(r, fv(&[4.0; 8]));
    // a signature that is not an entry (Python raises a KeyError here)
    let e = call_entry(&mut p, "f", &[fv(&[2.0; 8])], Some("int -> int"), false).err().unwrap();
    assert_eq!(e.kind, Kind::Name);
}

const BK: &str = ".extern(\"probe\";\"float[3] -> float[3]\")\n.extern(\"tick\";\"int -> int\")\nw::[1.5 2.5 3.5]\nf::{[a];a::probe(x+w);(a*2.0)+tick(1)}\n.entry(\"f\";\"float[3] -> float[3]\")\ng::{+/x}\n.entry(\"g\";\"int[4] -> int\")\n";

#[test]
fn the_boundary_carries_numbers_and_arrays_of_the_declared_type() {
    let seen: Rc<std::cell::RefCell<Vec<HostValue>>> = Rc::new(std::cell::RefCell::new(Vec::new()));
    let s1 = seen.clone();
    let probe: NativeFn = Rc::new(move |_ip, a| {
        let h = HostValue::from_value(&a[0]);
        s1.borrow_mut().push(h.clone());
        match h {
            HostValue::Floats(s, d) => Ok(HostValue::Floats(s, d.iter().map(|x| x * 0.5).collect()).to_value()),
            o => panic!("{:?}", o),
        }
    });
    let s2 = seen.clone();
    let tick: NativeFn = Rc::new(move |_ip, a| {
        s2.borrow_mut().push(HostValue::from_value(&a[0]));
        Ok(a[0].clone())
    });
    let mut ext: HashMap<String, HostFn> = HashMap::new();
    ext.insert("probe".into(), probe);
    ext.insert("tick".into(), tick);
    let mut p = load_program("p.kg", Some(BK), ext).unwrap();
    assert_eq!(call(&mut p, "f", &[fv(&[1.0, 2.0, 3.0])]), fv(&[3.5, 5.5, 7.5]));
    assert_eq!(call(&mut p, "g", &[iv(&[1, 2, 3, 4])]), HostValue::Int(10));
    assert_eq!(*seen.borrow(), vec![fv(&[2.5, 4.5, 6.5]), HostValue::Int(1)]);
    assert_eq!(err(&mut p, "f", &[fv(&[1.0, 2.0])]).0, Kind::Shape);
    assert_eq!(err(&mut p, "g", &[fv(&[1.0, 2.0, 3.0, 4.0])]).0, Kind::Type);
}

// ---------------------------------------------------------------------------
// parameters
// ---------------------------------------------------------------------------

#[test]
fn parameters_are_set_between_runs() {
    let mut k = load(".param(\"p\";\"int\";2)\nf::{x*p}\n.entry(\"f\";\"int -> int\")\n", &[]);
    assert_eq!(call(&mut k, "f", &[HostValue::Int(3)]), HostValue::Int(6));
    assert_eq!(set_param(&mut k, "p", &HostValue::Int(5)).unwrap(), HostValue::Int(5));
    assert_eq!(call(&mut k, "f", &[HostValue::Int(3)]), HostValue::Int(15));
    let e = set_param(&mut k, "p", &fl(1.5)).err().unwrap(); // an int parameter takes an int
    assert_eq!((e.kind, e.message.as_ref()), (Kind::Type, "the value of parameter p must be int, got a float"));
    let e = set_param(&mut k, "nope", &HostValue::Int(1)).err().unwrap();
    assert_eq!((e.kind, e.message.as_ref()), (Kind::Name, "nope is not a parameter of the program"));
    assert_eq!(e.hint.as_deref(), Some("declare it: .param(\"nope\";\"int\")"));
    let mut k2 = load(".param(\"g\";\"float\";0.5)\nf::{x*g}\n.entry(\"f\";\"float -> float\")\n", &[]);
    assert_eq!(set_param(&mut k2, "g", &HostValue::Int(2)).unwrap(), fl(2.0)); // an int converts to a float parameter
    assert_eq!(call(&mut k2, "f", &[fl(1.5)]), fl(3.0));
}

// ---------------------------------------------------------------------------
// fixed point (tests/static/test_fixed.py, tests/static/hw)
// ---------------------------------------------------------------------------

fn fx(elem: &str, shape: &[usize], raw: &[i64]) -> HostValue {
    HostValue::fixed(elem, shape, raw.to_vec())
}

fn raw_call(p: &mut Loaded, name: &str, args: &[HostValue]) -> HostValue {
    call_entry(p, name, args, None, true).unwrap_or_else(|e| panic!("{}: {}", name, e.message))
}

#[test]
fn fixed_param_in_the_evaluator() {
    let text = ".param(\"gain\";\"fixed<32,16>\";2)\n.param(\"n\";\"int\";3)\nacc::0\nf::{acc::acc+n;x*gain}\n.entry(\"f\";\"fixed<32,16> -> fixed<32,16>\")\n";
    let mut p = load(text, &[]);
    assert_eq!(call(&mut p, "f", &[fl(1.5)]), fl(3.0));
    set_param(&mut p, "gain", &fl(0.5)).unwrap();
    assert_eq!(call(&mut p, "f", &[fl(1.5)]), fl(0.75));
    assert!(set_param(&mut p, "nope", &HostValue::Int(1)).is_err());
    assert_eq!(p.get("acc"), Some(HostValue::Int(6)));
    // the raw form
    assert_eq!(raw_call(&mut p, "f", &[fx("fixed<32,16>", &[], &[98304])]), HostValue::Int(49152));
}

#[test]
fn fixed_state_persists_and_division_by_zero_is_a_system_error() {
    let text = "acc::.fixed(\"fixed<32,16>\";0)\nstep::{acc::acc+x;acc}\nq::{x%y}\n.entry(\"step\";\"fixed<32,16> -> fixed<32,16>\")\n.entry(\"q\";\"fixed<32,16> fixed<32,16> -> fixed<32,16>\")\n";
    let mut p = load(text, &[]);
    let mut got = Vec::new();
    for raw in [65536, 100, -3, 2147483647] {
        got.push(raw_call(&mut p, "step", &[fx("fixed<32,16>", &[], &[raw])]));
    }
    assert_eq!(got, vec![HostValue::Int(65536), HostValue::Int(65636), HostValue::Int(65633), HostValue::Int(-2147418016)]);
    assert_eq!(err(&mut p, "q", &[fx("fixed<32,16>", &[], &[5]), fx("fixed<32,16>", &[], &[0])]), (Kind::System, "entry q: fixed-point division by zero".into()));
    assert_eq!(raw_call(&mut p, "q", &[fl(5.5), fl(2.0)]), HostValue::Int(180224));
    // the state survives a trap and lives in the interpreter
    assert_eq!(p.get("acc"), Some(fx("fixed<32,16>", &[], &[-2147418016])));
}

#[test]
fn traps_of_the_evaluator_are_range_and_system_errors() {
    let text = "g::{x@y}\nzz::.fixed(\"fixed<16,8>\";0)\n.entry(\"g\";\"int[4] int -> int\")\nh::{x:%y}\n.entry(\"h\";\"int int -> int\")\n";
    let mut p = load(text, &[]);
    assert_eq!(call(&mut p, "g", &[iv(&[1, 2, 3, 4]), HostValue::Int(-4)]), HostValue::Int(1));
    assert_eq!(err(&mut p, "g", &[iv(&[1, 2, 3, 4]), HostValue::Int(4)]), (Kind::Range, "entry g: index out of range".into()));
    assert_eq!(err(&mut p, "h", &[HostValue::Int(1), HostValue::Int(0)]), (Kind::System, "entry h: integer division by zero".into()));
    assert_eq!(call(&mut p, "h", &[HostValue::Int(-7), HostValue::Int(2)]), HostValue::Int(-3));
}

#[test]
fn the_fixed_point_fir_matches_the_float_one() {
    let mut fixed = load_program("../../tests/static/hw/fir_fixed.kg", None, HashMap::new()).unwrap();
    let mut float = load_program("../../examples/embedded/fir.kg", None, HashMap::new()).unwrap();
    let xs: Vec<f64> = (0..256).map(|i| (((i * 37) % 64) as f64 - 32.0) / 4.0).collect();
    let (HostValue::Floats(_, a), HostValue::Floats(_, b)) = (call(&mut fixed, "run", &[fv(&xs)]), call(&mut float, "run", &[fv(&xs)])) else { panic!() };
    assert_eq!(a.len(), 241);
    assert!(a.iter().zip(&b).all(|(p, q)| (p - q).abs() < 2e-3), "{:?} {:?}", &a[..4], &b[..4]);
    // raw values are the fixed<32,16> integers; a fixed argument is passed as it is
    let HostValue::Ints(_, r) = raw_call(&mut fixed, "run", &[fv(&xs)]) else { panic!() };
    let raw_x: Vec<i64> = xs.iter().map(|v| (v * 65536.0) as i64).collect();
    let HostValue::Ints(_, r2) = raw_call(&mut fixed, "run", &[fx("fixed<32,16>", &[256], &raw_x)]) else { panic!() };
    assert_eq!(r, r2);
    assert_eq!(err(&mut fixed, "run", &[fv(&xs[..255])]).0, Kind::Shape);
}

#[test]
fn registers_of_hw_regs() {
    let mut p = load_program("../../tests/static/hw/regs.kg", None, HashMap::new()).unwrap();
    assert_eq!(call(&mut p, "step", &[fl(1.5)]), HostValue::Void);
    assert_eq!(call(&mut p, "step", &[fl(0.25)]), HostValue::Void);
    assert_eq!(call(&mut p, "summary", &[]), fv(&[3.5, 1.0, 6.0]));
    assert_eq!(p.get("tot"), Some(fl(3.5)));
    assert_eq!(p.get("cnt"), Some(HostValue::Int(6)));
    assert_eq!(p.get("acc"), Some(fx("fixed<32,16>", &[], &[65536])));
    set_param(&mut p, "gain", &fl(0.5)).unwrap();
    set_param(&mut p, "inc", &HostValue::Int(7)).unwrap();
    set_param(&mut p, "off", &fl(1.25)).unwrap();
    call(&mut p, "step", &[fl(2.0)]);
    assert_eq!(call(&mut p, "summary", &[]), fv(&[4.5, 1.0 + 0.0, 13.0]).clone_with(|d| d[1] = 2.25));
}

trait With {
    fn clone_with(self, f: impl FnOnce(&mut Vec<f64>)) -> HostValue;
}
impl With for HostValue {
    fn clone_with(self, f: impl FnOnce(&mut Vec<f64>)) -> HostValue {
        match self {
            HostValue::Floats(s, mut d) => {
                f(&mut d);
                HostValue::Floats(s, d)
            }
            o => o,
        }
    }
}

// ---------------------------------------------------------------------------
// conform
// ---------------------------------------------------------------------------

#[test]
fn conform_words_every_mismatch_as_python_does() {
    let ty = |t: &str| parse_type(t).unwrap();
    let m = |t: &str, v: HostValue| conform(&ty(t), &v, "argument 1 of f").err().map(|m| (m.kind, m.message));
    assert_eq!(m("int", fl(0.5)), Some(("type", "argument 1 of f must be int, got a float".into())));
    assert_eq!(m("int", iv(&[1, 2])), Some(("type", "argument 1 of f must be int, got int[2]".into())));
    assert_eq!(m("float[3]", fv(&[1.0, 2.0])), Some(("shape", "argument 1 of f must be float[3], got float[2]".into())));
    assert_eq!(m("float[3]", fl(1.0)), Some(("type", "argument 1 of f must be float[3], got a float".into())));
    assert_eq!(m("int[2]", fv(&[1.0, 2.0])), Some(("type", "argument 1 of f must be int[2], got float[2]".into())));
    assert_eq!(m("int[2,2]", HostValue::ints(&[2, 3], vec![0; 6])), Some(("shape", "argument 1 of f must be int[2,2], got int[2,3]".into())));
    assert_eq!(m("str", HostValue::Int(1)), Some(("type", "argument 1 of f must be str, got an integer".into())));
    assert_eq!(m("fixed<8,4>", HostValue::Str("a".into())), Some(("type", "argument 1 of f must be fixed<8,4>, got a string".into())));
    assert_eq!(m("fixed<8,4>[2]", fx("fixed<16,8>", &[2], &[1, 2])), Some(("type", "argument 1 of f must be fixed<8,4>[2], got fixed<16,8>[2]".into())));
    assert_eq!(m("fixed<8,4>[2]", fv(&[1.0, 2.0, 3.0])), Some(("shape", "argument 1 of f must be fixed<8,4>[2], got float[3]".into())));
    assert_eq!(m("fixed<8,4>", fv(&[1.0, 2.0])), Some(("type", "argument 1 of f must be fixed<8,4>, got float[2]".into())));
    // accepted forms
    let ok = |t: &str, v: HostValue| conform(&ty(t), &v, "x").unwrap();
    assert_eq!(ok("float", HostValue::Int(2)), fl(2.0));
    assert_eq!(ok("float[2]", iv(&[1, 2])), fv(&[1.0, 2.0]));
    assert_eq!(ok("void", fl(1.0)), HostValue::Void);
    assert_eq!(ok("fixed<8,4>", fl(1.99)), fx("fixed<8,4>", &[], &[31]));
    assert_eq!(ok("fixed<8,4>", fl(-1.99)), fx("fixed<8,4>", &[], &[-32]));
    assert_eq!(ok("fixed<8,4>", HostValue::Int(9)), fx("fixed<8,4>", &[], &[-112]));
    assert_eq!(ok("fixed<8,4>", fl(f64::NAN)), fx("fixed<8,4>", &[], &[0]));
}

// ---------------------------------------------------------------------------
// the scenarios of the Python host
// ---------------------------------------------------------------------------

fn hexf(x: f64) -> String {
    if x.is_nan() {
        "nan".to_string()
    } else {
        format!("{:016x}", x.to_bits())
    }
}

fn unhex(h: &str) -> f64 {
    if h == "nan" {
        f64::NAN
    } else {
        f64::from_bits(u64::from_str_radix(h, 16).unwrap())
    }
}

fn dec(s: &str) -> HostValue {
    if let Some(v) = s.strip_prefix("i:") {
        return HostValue::Int(v.parse().unwrap());
    }
    if let Some(v) = s.strip_prefix("f:") {
        return HostValue::Float(unhex(v));
    }
    if s.starts_with('X') {
        let (head, data) = s.split_once(':').unwrap();
        let close = head.find('>').unwrap();
        let elem = format!("fixed{}", &head[1..=close]);
        let shape: Vec<usize> = if head.len() > close + 1 { head[close + 2..head.len() - 1].split(',').map(|d| d.parse().unwrap()).collect() } else { vec![] };
        return HostValue::fixed(&elem, &shape, data.split(',').map(|x| x.parse().unwrap()).collect());
    }
    let (head, data) = s.split_once("]:").unwrap();
    let shape: Vec<usize> = head[2..].split(',').map(|d| d.parse().unwrap()).collect();
    let items: Vec<&str> = data.split(',').filter(|t| !t.is_empty()).collect();
    if head.starts_with('I') {
        HostValue::Ints(shape, items.iter().map(|x| x.parse().unwrap()).collect())
    } else {
        HostValue::Floats(shape, items.iter().map(|x| unhex(x)).collect())
    }
}

fn enc(v: &HostValue) -> String {
    match v {
        HostValue::Void => "none".to_string(),
        HostValue::Int(i) => format!("i:{}", i),
        HostValue::Float(f) => format!("f:{}", hexf(*f)),
        HostValue::Fixed { shape, raw, .. } if shape.is_empty() => format!("i:{}", raw[0]),
        HostValue::Ints(s, d) | HostValue::Fixed { shape: s, raw: d, .. } => {
            let dims: Vec<String> = s.iter().map(|x| x.to_string()).collect();
            format!("I[{}]:{}", dims.join(","), d.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","))
        }
        HostValue::Floats(s, d) => {
            let dims: Vec<String> = s.iter().map(|x| x.to_string()).collect();
            format!("F[{}]:{}", dims.join(","), d.iter().map(|x| hexf(*x)).collect::<Vec<_>>().join(","))
        }
        o => format!("OTHER {:?}", o),
    }
}

fn class(k: Kind) -> &'static str {
    match k {
        Kind::Name => "KlongNameError",
        Kind::Arity => "KlongArityError",
        Kind::Type => "KlongTypeError",
        Kind::Shape => "KlongShapeError",
        Kind::Range => "KlongRangeError",
        Kind::System => "KlongSystemError",
        Kind::Syntax => "KlongSyntaxError",
        _ => "KlongError",
    }
}

fn enc_err(e: &klong_core::Error) -> String {
    format!("ERR {} {}", class(e.kind), e.message)
}

#[test]
fn python_scenarios() {
    let Ok(path) = std::env::var("KG_HOST_SCENARIOS") else {
        return;
    };
    let Ok(text) = std::fs::read_to_string(&path) else {
        eprintln!("python_scenarios: {} not found, skipped", path);
        return;
    };
    let (mut checked, mut bad, mut known) = (0, Vec::<String>::new(), Vec::<String>::new());
    let mut scen = String::new();
    let mut loaded: Option<Loaded> = None;
    let mut ext: HashMap<String, HostFn> = HashMap::new();
    let mut src: (Option<String>, Option<String>) = (None, None);
    for line in text.lines() {
        if let Some(n) = line.strip_prefix("SCEN ") {
            scen = n.to_string();
            loaded = None;
            ext.clear();
            src = (None, None);
        } else if let Some(p) = line.strip_prefix("PATH ") {
            src.0 = Some(format!("../../{}", p));
        } else if let Some(t) = line.strip_prefix("TEXT ") {
            src.1 = Some(t.replace("\\n", "\n").replace("\\\\", "\\"));
        } else if let Some(e) = line.strip_prefix("EXT ") {
            let (k, v) = e.split_once(' ').unwrap();
            ext.insert(k.to_string(), konst(dec(v)));
        } else if let Some(r) = line.strip_prefix("LOAD => ") {
            let res = match load_program(src.0.as_deref().unwrap_or("prog.kg"), src.1.as_deref(), ext.clone()) {
                Ok(l) => {
                    loaded = Some(l);
                    "ok".to_string()
                }
                Err(e) => enc_err(&e),
            };
            checked += 1;
            if res != r {
                bad.push(format!("{}: LOAD python {} rust {}", scen, r, res));
            }
        } else if let Some(c) = line.strip_prefix("CALL ") {
            let (lhs, want) = c.split_once(" => ").unwrap();
            let mut it = lhs.split(' ');
            let name = it.next().unwrap();
            let sig = it.next().unwrap().strip_prefix("sig:").unwrap();
            let raw = it.next().unwrap() == "raw";
            let args: Vec<HostValue> = it.map(dec).collect();
            let sig = if sig == "-" { None } else { Some(sig.replace('_', " ")) };
            let res = match call_entry(loaded.as_mut().unwrap(), name, &args, sig.as_deref(), raw) {
                Ok(v) => enc(&v),
                Err(e) => enc_err(&e),
            };
            checked += 1;
            if res != want {
                let what = format!("{}: {} python {} rust {}", scen, lhs.chars().take(60).collect::<String>(), want, res);
                if want.starts_with("PYERR") {
                    known.push(what);
                } else {
                    bad.push(what);
                }
            }
        } else if let Some(g) = line.strip_prefix("GET ") {
            let (name, want) = g.split_once(" => ").unwrap();
            let res = loaded.as_ref().unwrap().get(name).map(|v| enc(&v)).unwrap_or_else(|| "PYERR".to_string());
            checked += 1;
            if res != want {
                bad.push(format!("{}: GET {} python {} rust {}", scen, name, want, res));
            }
        } else if let Some(p) = line.strip_prefix("PARAM ") {
            let (lhs, want) = p.split_once(" => ").unwrap();
            let (name, arg) = lhs.split_once(' ').unwrap();
            let res = match set_param(loaded.as_mut().unwrap(), name, &dec(arg)) {
                Ok(v) => enc(&v),
                Err(e) => enc_err(&e),
            };
            checked += 1;
            if res != want {
                bad.push(format!("{}: PARAM {} python {} rust {}", scen, lhs, want, res));
            }
        }
    }
    eprintln!("python_scenarios: {} lines replayed, {} differ, {} where Python crashes", checked, bad.len(), known.len());
    for b in bad.iter().chain(known.iter()) {
        eprintln!("DIFF {}", b);
    }
    assert!(bad.is_empty(), "{} differences", bad.len());
}
