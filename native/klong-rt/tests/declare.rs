//! `.extern`, `.entry` and `.fixed` (KlongPy's static-profile declarations, `klongpy/sys_fn_static.py`) and the reference programs of
//! `examples/embedded`, run by this interpreter with the host functions they declare and the values of their `.expect` files.
use klong_core::{Config, Interp, Kind, Value};
use klong_rt::memfs::MemFs;
use klong_rt::{FileSystem, Platform, Runtime};
use std::cell::RefCell;
use std::rc::Rc;

#[derive(Default)]
struct Board {
    fs: MemFs,
}

impl Platform for Board {
    fn now_us(&self) -> u64 {
        0
    }
    fn wall_clock(&self) -> Option<f64> {
        None
    }
    fn idle(&mut self) {}
    fn console_read(&mut self, _: &mut [u8]) -> usize {
        0
    }
    fn console_write(&mut self, _: &[u8]) {}
    fn fs(&mut self) -> Option<&mut dyn FileSystem> {
        Some(&mut self.fs)
    }
}

/// (One interpreter at a time: see klong-core/tests/full.rs.)
static ONE: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn rt() -> Runtime<Board> {
    Runtime::new(Board::default(), Config::default(), None)
}

fn show(rt: &mut Runtime<Board>, text: &str) -> Result<String, Kind> {
    let _one = ONE.lock().unwrap_or_else(|e| e.into_inner());
    match rt.run(text, "<test>") {
        Ok(Some(v)) => klong_core::fmt::format_value(&rt.interp.syms, &v, false).map_err(|e| e.kind),
        Ok(None) => Ok(String::new()),
        Err(e) => Err(e.kind),
    }
}

#[test]
fn declarations_return_the_canonical_signature() {
    let mut rt = rt();
    assert_eq!(show(&mut rt, ".extern(\"read.adc\";\"int -> float\")"), Ok("\"int -> float\"".into()));
    assert_eq!(show(&mut rt, ".extern(\"f\";\"int   float[8]->  void\";\"dsp\")"), Ok("\"int float[8] -> void\"".into()));
    assert_eq!(show(&mut rt, ".extern(\"read.adc\";\"int -> float\")"), Ok("\"int -> float\"".into()), "the same declaration again");
    assert_eq!(show(&mut rt, "step::{x*2};.entry(\"step\";\"float -> float\")"), Ok("\"float -> float\"".into()));
    assert_eq!(show(&mut rt, "mul(a;b)::{a*b};.entry(\"mul\";\"int  int -> int\")"), Ok("\"int int -> int\"".into()));
}

#[test]
fn declarations_that_are_wrong() {
    let cases: &[(&str, Kind)] = &[
        (".extern(\"f\")", Kind::Arity),
        (".extern(1;\"int -> int\")", Kind::Type),
        (".extern(\"f\";\"int\")", Kind::Type),
        (".extern(\"f\";\"quux -> int\")", Kind::Type),
        (".extern(\"f\";\"str -> str\")", Kind::Type),
        (".extern(\"f\";\"void -> int\")", Kind::Type),
        (".extern(\"f\";\"int[0] -> int\")", Kind::Type),
        (".extern(\"x\";\"int -> int\")", Kind::Type),
        (".extern(\"int\";\"int -> int\")", Kind::Type),
        (".extern(\"1a\";\"int -> int\")", Kind::Type),
        (".extern(\"f\";\"int -> int\");.extern(\"f\";\"float -> int\")", Kind::Type),
        ("f::1;.extern(\"f\";\"int -> int\")", Kind::Name),
        (".entry(\"nope\";\"int -> int\")", Kind::Name),
        ("v::3;.entry(\"v\";\"int -> int\")", Kind::Type),
        ("h::{x+y};p::h(1;);.entry(\"p\";\"int -> int\")", Kind::Type),
        ("step::{x*2};.entry(\"step\";\"int int -> int\")", Kind::Arity),
        ("step::{x*2};.entry(\"step\";\"str -> int\")", Kind::Type),
        ("init::{x};.entry(\"init\";\"int -> int\")", Kind::Type),
        (".fixed(\"fixed<32,16>\";0.5)", Kind::System),
    ];
    for (src, kind) in cases {
        assert_eq!(show(&mut rt(), src), Err(*kind), "{src}");
    }
}

#[test]
fn an_extern_with_no_host_function_is_a_system_error() {
    let mut rt = rt();
    show(&mut rt, ".extern(\"read.adc\";\"int -> float\");.extern(\"g\";\"int int int int int -> int\")").unwrap();
    assert_eq!(show(&mut rt, "read.adc(0)"), Err(Kind::System));
    assert_eq!(show(&mut rt, "read.adc(0;1)"), Err(Kind::Arity));
    assert_eq!(show(&mut rt, "read.adc()"), Err(Kind::Arity));
    assert_eq!(show(&mut rt, "g(1;2;3;4;5)"), Err(Kind::System));
    assert_eq!(show(&mut rt, "g(1;2;3;4;5;6)"), Err(Kind::Arity));
}

/// Register a host function as the C interface does (`Interp::register`).
fn host(rt: &mut Runtime<Board>, name: &str, arity: u8, f: impl Fn(&mut Interp, &[Value]) -> klong_core::error::Result<Value> + 'static) {
    rt.interp.register(name, arity, Rc::new(f));
}

#[test]
fn a_registered_function_is_what_the_extern_calls_and_its_arguments_and_result_are_checked() {
    let mut rt = rt();
    let seen = Rc::new(RefCell::new(Vec::<String>::new()));
    let log = seen.clone();
    host(&mut rt, "h.sum", WILDCARD, move |ip, a| {
        log.borrow_mut().push(a.iter().map(|v| ip.format(v)).collect::<Vec<_>>().join(","));
        Ok(match a.first() {
            Some(Value::Int(i)) => Value::Flt(*i as f64 * 2.0),
            _ => Value::Undefined,
        })
    });
    host(&mut rt, "h.bad", 1, |_, _| Ok(Value::Flt(1.5)));
    host(&mut rt, "h.many", 8, |_, a| Ok(Value::Int(a.len() as i64)));
    show(&mut rt, ".extern(\"h.sum\";\"int float[3] -> float\");.extern(\"h.bad\";\"int -> int\");.extern(\"h.many\";\"int int int int int int -> int\")").unwrap();
    // an int is converted to a float where the signature says float, and the result is a float
    assert_eq!(show(&mut rt, "h.sum(4;[1 2 3])"), Ok("8.0".into()));
    assert_eq!(show(&mut rt, "h.sum(4;[1.5 2 3])"), Ok("8.0".into()));
    assert_eq!(seen.borrow().as_slice(), ["4,[1.0 2.0 3.0]", "4,[1.5 2.0 3.0]"]);
    assert_eq!(show(&mut rt, "h.sum(4.0;[1 2 3])"), Err(Kind::Type), "a float where an int is declared");
    assert_eq!(show(&mut rt, "h.sum(4;[1 2])"), Err(Kind::Shape), "the wrong length");
    assert_eq!(show(&mut rt, "h.sum(4;3)"), Err(Kind::Type), "a scalar for an array");
    assert_eq!(show(&mut rt, "h.sum(4;\"abc\")"), Err(Kind::Type));
    assert_eq!(show(&mut rt, "h.sum(4)"), Err(Kind::Arity));
    assert_eq!(show(&mut rt, "h.bad(1)"), Err(Kind::Type), "the result is a float, an int is declared");
    assert_eq!(show(&mut rt, "h.many(1;2;3;4;5;6)"), Ok("6".into()), "more than three arguments reach the function");
    // a program that defines the name itself is not allowed to
    assert_eq!(show(&mut rt, "q::1;.extern(\"q\";\"int -> int\")"), Err(Kind::Name));
}

const WILDCARD: u8 = klong_core::value::WILDCARD;

fn text(dir: &str, file: &str) -> String {
    std::fs::read_to_string(format!("{}/../../examples/embedded/{}", env!("CARGO_MANIFEST_DIR"), file)).unwrap_or_else(|e| panic!("{dir}{file}: {e}"))
}

/// The steps of a `.expect` file: `expression => result`, in order, and the `# extern NAME = VALUE` stand-ins.
fn expect(name: &str) -> (Vec<(String, f64)>, Vec<(String, String)>) {
    let mut externs = Vec::new();
    let mut steps = Vec::new();
    for line in text("", &format!("{name}.expect")).lines() {
        if let Some(rest) = line.strip_prefix("# extern ") {
            let (n, v) = rest.split_once('=').expect("extern line");
            externs.push((n.trim().to_string(), v.trim().parse().expect("number")));
        } else if !line.trim().is_empty() && !line.starts_with('#') {
            let (e, w) = line.split_once(" => ").expect("step");
            steps.push((e.trim().to_string(), w.trim().to_string()));
        }
    }
    (externs, steps)
}

/// The reference program loaded as it stands, the host functions of its `.expect` file registered, every step of the file giving its value.
fn reference(name: &str) -> Vec<(i64, f64)> {
    let mut rt = rt();
    let writes = Rc::new(RefCell::new(Vec::new()));
    let (externs, steps) = expect(name);
    for (n, v) in externs {
        let w = writes.clone();
        let is_pwm = n == "write.pwm";
        let whole = v.fract() == 0.0;
        host(&mut rt, &n.clone(), WILDCARD, move |_, a| {
            if is_pwm {
                if let (Some(Value::Int(c)), Some(Value::Flt(d))) = (a.first(), a.get(1)) {
                    w.borrow_mut().push((*c, *d));
                }
            }
            Ok(if whole { Value::Int(v as i64) } else { Value::Flt(v) })
        });
    }
    let _one = ONE.lock().unwrap_or_else(|e| e.into_inner());
    rt.interp.load(&text("", &format!("{name}.kg")), &format!("{name}.kg")).unwrap_or_else(|e| panic!("{name}.kg: {}", e.message));
    assert!(!steps.is_empty());
    for (expr, want) in steps {
        // (`~` compares floats as numpy.isclose does)
        let got = rt.run(&format!("({expr})~({want})"), "<expect>").unwrap_or_else(|e| panic!("{name}: {expr}: {}", e.message));
        assert!(matches!(got, Some(Value::Int(1))), "{name}: {expr} is not {want}: it is {}", {
            let v = rt.run(&expr, "<expect>").ok().flatten().unwrap_or(Value::Undefined);
            rt.interp.format(&v)
        });
    }
    let w = writes.borrow().clone();
    w
}

#[test]
fn mavg_runs_with_the_values_of_its_expect_file() {
    assert!(reference("mavg").is_empty());
}

#[test]
fn fir_runs_with_the_values_of_its_expect_file() {
    assert!(reference("fir").is_empty());
}

#[test]
fn pid_runs_with_the_values_of_its_expect_file_and_writes_its_duty() {
    let writes = reference("pid");
    // one write.pwm(0;u) for each step() of the file, u being the value step returned
    let duty: Vec<f64> = writes.iter().map(|w| w.1).collect();
    assert_eq!(writes.iter().map(|w| w.0).collect::<Vec<_>>(), [0, 0, 0]);
    for (got, want) in duty.iter().zip([0.12005, 0.0201, 1.0]) {
        assert!((got - want).abs() < 1e-9, "{duty:?}");
    }
}

#[cfg(feature = "params")]
#[test]
fn float_converts_ints_and_number_arrays() {
    let mut rt = rt();
    assert_eq!(show(&mut rt, ".float(3)"), Ok("3.0".into()));
    assert_eq!(show(&mut rt, ".float(2.5)"), Ok("2.5".into()));
    assert_eq!(show(&mut rt, ".float([1 2 3])"), Ok("[1.0 2.0 3.0]".into()));
    assert_eq!(show(&mut rt, ".float([[1 2] [3 4]])"), Ok("[[1.0 2.0] [3.0 4.0]]".into()));
    assert_eq!(show(&mut rt, ".float(\"a\")"), Err(Kind::Type));
}

#[cfg(feature = "params")]
#[test]
fn param_declares_a_global_with_its_default() {
    let mut rt = rt();
    assert_eq!(show(&mut rt, ".param(\"window\";\"int\";4)"), Ok("\"int\"".into()));
    assert_eq!(show(&mut rt, "window"), Ok("4".into()));
    assert_eq!(show(&mut rt, ".param(\"g\";\"float\")"), Ok("\"float\"".into()));
    assert_eq!(show(&mut rt, "g"), Ok("0.0".into()));
    assert_eq!(show(&mut rt, ".param(\"h\";\"float\";2);h"), Ok("2.0".into()));
    assert_eq!(show(&mut rt, ".param(\"stop\";\"fixed<32,16>\";0.5)"), Ok("\"fixed<32,16>\"".into()));
    // the same type again keeps the first default; another type, a defined name, a bad type, a bad default are errors
    assert_eq!(show(&mut rt, ".param(\"window\";\"int\";9);window"), Ok("4".into()));
    assert_eq!(show(&mut rt, ".param(\"window\";\"float\")"), Err(Kind::Type));
    assert_eq!(show(&mut rt, "k::1;.param(\"k\";\"int\")"), Err(Kind::Name));
    assert_eq!(show(&mut rt, ".param(\"a\";\"int[4]\")"), Err(Kind::Type));
    assert_eq!(show(&mut rt, ".param(\"a\";\"int\";0.5)"), Err(Kind::Type));
    assert_eq!(show(&mut rt, ".param(\"a\";\"float\";\"s\")"), Err(Kind::Type));
    assert_eq!(show(&mut rt, ".param(\"x\";\"int\")"), Err(Kind::Type));
    assert_eq!(show(&mut rt, ".param(\"a\")"), Err(Kind::Arity));
    assert_eq!(show(&mut rt, ".extern(\"window\";\"int -> int\")"), Err(Kind::Name));
}

#[cfg(feature = "params")]
#[test]
fn the_host_reads_the_declarations_and_sets_the_parameters() {
    let mut rt = rt();
    show(&mut rt, ".extern(\"read.adc\";\"int -> float\");.param(\"n\";\"int\";3);.param(\"g\";\"float\");.param(\"s\";\"fixed<16,8>\")").unwrap();
    show(&mut rt, "step::{x*2};.entry(\"step\";\"int -> int\")").unwrap();
    let d = rt.declarations.clone();
    let ex = d.externs();
    assert_eq!((ex[0].0.as_str(), ex[0].1.to_string().as_str(), ex[0].2.as_str()), ("read.adc", "int -> float", "env"));
    let en = d.entries();
    assert_eq!((en[0].0.as_str(), en[0].1.to_string().as_str()), ("step", "int -> int"));
    let ps: Vec<(String, String)> = d.params().iter().map(|p| (p.0.clone(), p.1.to_string())).collect();
    assert_eq!(ps, [("n".into(), "int".into()), ("g".into(), "float".into()), ("s".into(), "fixed<16,8>".into())]);
    let _one = ONE.lock().unwrap_or_else(|e| e.into_inner());
    assert!(matches!(d.set_param(&mut rt.interp, "n", &Value::Int(7)), Ok(Value::Int(7))));
    assert!(matches!(d.set_param(&mut rt.interp, "g", &Value::Int(2)), Ok(Value::Flt(f)) if f == 2.0));
    assert!(d.set_param(&mut rt.interp, "s", &Value::Flt(0.5)).is_ok());
    assert_eq!(d.set_param(&mut rt.interp, "n", &Value::Flt(1.5)).err().map(|e| e.kind), Some(Kind::Type));
    assert_eq!(d.set_param(&mut rt.interp, "nope", &Value::Int(1)).err().map(|e| e.kind), Some(Kind::Name));
    drop(_one);
    assert_eq!(show(&mut rt, "n"), Ok("7".into()));
    assert_eq!(show(&mut rt, "g"), Ok("2.0".into()));
    // a host function bound after the declaration is what the extern calls
    let _one = ONE.lock().unwrap_or_else(|e| e.into_inner());
    d.bind(&mut rt.interp, "read.adc", Rc::new(|_: &mut Interp, a: &[Value]| Ok(Value::Flt(match a[0] { Value::Int(i) => i as f64 * 0.5, _ => 0.0 })))).unwrap();
    assert!(d.bind(&mut rt.interp, "other", Rc::new(|_: &mut Interp, _: &[Value]| Ok(Value::Undefined))).is_err());
    drop(_one);
    assert_eq!(show(&mut rt, "read.adc(5)"), Ok("2.5".into()));
}
