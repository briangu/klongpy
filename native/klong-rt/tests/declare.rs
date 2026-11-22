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
    // an extern is a function value; calling it with the wrong arity names the canonical signature
    show(&mut rt, "adc::.extern(\"read_adc\";\"int -> float\")").unwrap();
    show(&mut rt, "f::.extern(\"dsp_f\";\"int   float[8]->  void\")").unwrap();
    assert_eq!(show(&mut rt, "f(1)"), Err(Kind::Arity));
    assert_eq!(show(&mut rt, "adc2::.extern(\"read_adc\";\"int -> float\");adc2=adc"), Ok("1".into()), "the same declaration again is the same function");
    assert_eq!(show(&mut rt, "step::{x*2};.entry(\"step\";\"float -> float\")"), Ok("\"float -> float\"".into()));
    assert_eq!(show(&mut rt, "mul(a;b)::{a*b};.entry(\"mul\";\"int  int -> int\")"), Ok("\"int int -> int\"".into()));
}

#[test]
fn declarations_that_are_wrong() {
    let cases: &[(&str, Kind)] = &[
        ("a::.extern(\"f\")", Kind::Arity),
        ("a::.extern(\"f\";\"int -> int\";\"env\")", Kind::Arity),
        ("a::.extern(1;\"int -> int\")", Kind::Type),
        ("a::.extern(\"f\";\"int\")", Kind::Type),
        ("a::.extern(\"f\";\"quux -> int\")", Kind::Type),
        ("a::.extern(\"f\";\"str -> str\")", Kind::Type),
        ("a::.extern(\"f\";\"void -> int\")", Kind::Type),
        ("a::.extern(\"f\";\"int[0] -> int\")", Kind::Type),
        ("a::.extern(\"int\";\"int -> int\")", Kind::Type),
        ("a::.extern(\"1a\";\"int -> int\")", Kind::Type),
        ("a::.extern(\"read.adc\";\"int -> int\")", Kind::Type),
        ("a::.extern(\"kg_x\";\"int -> int\")", Kind::Type),
        ("a::.extern(\"f\";\"int -> int\");b::.extern(\"f\";\"float -> int\")", Kind::Type),
        (".extern(\"f\";\"int -> int\")", Kind::Type),
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
    show(&mut rt, "adc::.extern(\"read_adc\";\"int -> float\");g::.extern(\"gg\";\"int int int int int -> int\")").unwrap();
    assert_eq!(show(&mut rt, "adc(0)"), Err(Kind::System));
    assert_eq!(show(&mut rt, "adc(0;1)"), Err(Kind::Arity));
    assert_eq!(show(&mut rt, "adc()"), Err(Kind::Arity));
    assert_eq!(show(&mut rt, "g(1;2;3;4;5)"), Err(Kind::System));
    assert_eq!(show(&mut rt, "g(1;2;3;4;5;6)"), Err(Kind::Arity));
}

/// Give the host function of a C symbol, as the C interface does.
fn host(rt: &mut Runtime<Board>, name: &str, _arity: u8, f: impl Fn(&mut Interp, &[Value]) -> klong_core::error::Result<Value> + 'static) {
    rt.declarations.provide(name, Rc::new(f)).unwrap();
}

#[test]
fn a_registered_function_is_what_the_extern_calls_and_its_arguments_and_result_are_checked() {
    let mut rt = rt();
    let seen = Rc::new(RefCell::new(Vec::<String>::new()));
    let log = seen.clone();
    host(&mut rt, "h_sum", WILDCARD, move |ip, a| {
        log.borrow_mut().push(a.iter().map(|v| ip.format(v)).collect::<Vec<_>>().join(","));
        Ok(match a.first() {
            Some(Value::Int(i)) => Value::Flt(*i as f64 * 2.0),
            _ => Value::Undefined,
        })
    });
    host(&mut rt, "h_bad", 1, |_, _| Ok(Value::Flt(1.5)));
    host(&mut rt, "h_many", 8, |_, a| Ok(Value::Int(a.len() as i64)));
    show(&mut rt, "hsum::.extern(\"h_sum\";\"int float[3] -> float\");hbad::.extern(\"h_bad\";\"int -> int\");hmany::.extern(\"h_many\";\"int int int int int int -> int\")").unwrap();
    // an int is converted to a float where the signature says float, and the result is a float
    assert_eq!(show(&mut rt, "hsum(4;[1 2 3])"), Ok("8.0".into()));
    assert_eq!(show(&mut rt, "hsum(4;[1.5 2 3])"), Ok("8.0".into()));
    assert_eq!(seen.borrow().as_slice(), ["4,[1.0 2.0 3.0]", "4,[1.5 2.0 3.0]"]);
    assert_eq!(show(&mut rt, "hsum(4.0;[1 2 3])"), Err(Kind::Type), "a float where an int is declared");
    assert_eq!(show(&mut rt, "hsum(4;[1 2])"), Err(Kind::Shape), "the wrong length");
    assert_eq!(show(&mut rt, "hsum(4;3)"), Err(Kind::Type), "a scalar for an array");
    assert_eq!(show(&mut rt, "hsum(4;\"abc\")"), Err(Kind::Type));
    assert_eq!(show(&mut rt, "hsum(4)"), Err(Kind::Arity));
    assert_eq!(show(&mut rt, "hbad(1)"), Err(Kind::Type), "the result is a float, an int is declared");
    assert_eq!(show(&mut rt, "hmany(1;2;3;4;5;6)"), Ok("6".into()), "more than three arguments reach the function");
    // a provided function is not a global: the program reaches it only through its declaration
    host(&mut rt, "blink", 1, |_, _| Ok(Value::Int(1)));
    assert_eq!(show(&mut rt, "blink(1)"), Err(Kind::Name));
    assert_eq!(show(&mut rt, "h_sum(1;2)"), Err(Kind::Name));
    show(&mut rt, "bl::.extern(\"blink\";\"int -> int\")").unwrap();
    assert_eq!(show(&mut rt, "bl(1)"), Ok("1".into()));
    assert_eq!(show(&mut rt, "blink(1)"), Err(Kind::Name));
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
        // (the host registers by C symbol: read.adc of an older expect file is read_adc)
        let n = n.replace('.', "_");
        let is_pwm = n == "write_pwm";
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
    // the target's functions are module files next to the program (`read.kg`, `write.kg`), found by `.l`
    for m in ["board"] {
        let path = format!("{}/../../examples/embedded/{m}.kg", env!("CARGO_MANIFEST_DIR"));
        if let Ok(t) = std::fs::read_to_string(&path) {
            rt.shared.platform.borrow_mut().fs.put(&format!("{m}.kg"), t.as_bytes()).unwrap();
        }
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
}

#[cfg(feature = "params")]
#[test]
fn the_host_reads_the_declarations_and_sets_the_parameters() {
    let mut rt = rt();
    show(&mut rt, "adc::.extern(\"read_adc\";\"int -> float\");.param(\"n\";\"int\";3);.param(\"g\";\"float\");.param(\"s\";\"fixed<16,8>\")").unwrap();
    show(&mut rt, "step::{x*2};.entry(\"step\";\"int -> int\")").unwrap();
    let d = rt.declarations.clone();
    let ex = d.externs();
    assert_eq!((ex[0].0.as_str(), ex[0].1.to_string().as_str()), ("read_adc", "int -> float"));
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
    d.bind("read_adc", Rc::new(|_: &mut Interp, a: &[Value]| Ok(Value::Flt(match a[0] { Value::Int(i) => i as f64 * 0.5, _ => 0.0 })))).unwrap();
    assert!(d.bind("other", Rc::new(|_: &mut Interp, _: &[Value]| Ok(Value::Undefined))).is_err());
    drop(_one);
    assert_eq!(show(&mut rt, "adc(5)"), Ok("2.5".into()));
}

#[cfg(feature = "diag")]
#[test]
fn the_stand_in_is_looked_up_at_each_call() {
    let mut rt = rt();
    let which: Rc<RefCell<Option<f64>>> = Rc::new(RefCell::new(None));
    let w = which.clone();
    rt.declarations.set_resolver(Some(Rc::new(move |_ip: &mut Interp, name: &str, _args: &[Value]| {
        assert_eq!(name, "read_adc");
        w.borrow().map(|x| Ok(Value::Flt(x)))
    })));
    assert_eq!(show(&mut rt, "adc::.extern(\"read_adc\";\"int -> float\");f::{adc(x)*2}"), Ok(":monad".into()));
    // no stand-in yet: a system error at the call
    assert_eq!(show(&mut rt, "f(0)"), Err(Kind::System));
    *which.borrow_mut() = Some(1.0);
    assert_eq!(show(&mut rt, "f(0)"), Ok("2.0".into()));
    *which.borrow_mut() = Some(4.0);
    assert_eq!(show(&mut rt, "f(0)"), Ok("8.0".into()));
    *which.borrow_mut() = None;
    assert_eq!(show(&mut rt, "f(0)"), Err(Kind::System));
    assert_eq!(rt.declarations.externs().len(), 1);
}

#[cfg(feature = "diag")]
#[test]
fn fixed_is_asked_of_the_host() {
    let mut rt = rt();
    assert_eq!(show(&mut rt, ".fixed(\"fixed<8,4>\";1)"), Err(Kind::System));
    rt.declarations.set_resolver(Some(Rc::new(|_ip: &mut Interp, name: &str, args: &[Value]| {
        (name == ".fixed").then(|| Ok(args[1].clone()))
    })));
    assert_eq!(show(&mut rt, ".fixed(\"fixed<8,4>\";7)"), Ok("7".into()));
}
