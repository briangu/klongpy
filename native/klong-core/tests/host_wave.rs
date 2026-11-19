//! What a host (the Python extension) relies on: a host object or callable as a dictionary key, the hook between parsing and running,
//! a missing right operand at strict level 0, float32 printing, a thread re-entering a running interpreter, and the text of an error.
#![cfg(all(feature = "diag", feature = "host"))]
use klong_core::value::{Dict, Ext, ExtBox, Func, Value};
use klong_core::{Config, Interp, NoHooks};
use std::rc::Rc;
use std::sync::Mutex;

static ONE: Mutex<()> = Mutex::new(());
fn lock() -> std::sync::MutexGuard<'static, ()> {
    ONE.lock().unwrap_or_else(|e| e.into_inner())
}
fn interp() -> Interp {
    Interp::new(Config::default(), Box::new(NoHooks))
}
fn show(ip: &mut Interp, src: &str) -> String {
    match ip.run(src, "<t>") {
        Ok(Some(v)) => ip.format(&v),
        Ok(None) => String::from("(no value)"),
        Err(e) => format!("!{}", e.message),
    }
}

struct Obj(usize);
impl Ext for Obj {
    fn type_name(&self) -> &str {
        "obj"
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn key_identity(&self) -> Option<usize> {
        Some(self.0)
    }
}
fn obj(id: usize) -> Value {
    Value::Ext(Rc::new(ExtBox(Box::new(Obj(id)))))
}

#[test]
fn a_host_object_is_a_key_by_identity() {
    let _g = lock();
    let mut d = Dict::default();
    d.set(obj(7), Value::Int(1)).unwrap();
    d.set(obj(8), Value::Int(2)).unwrap();
    d.set(obj(7), Value::Int(3)).unwrap();
    assert_eq!(d.len(), 2);
    assert!(matches!(d.get(&obj(7)), Some(Value::Int(3))));
    assert!(d.get(&obj(9)).is_none());
}

#[test]
fn a_native_function_is_a_key_by_identity() {
    let _g = lock();
    let mut ip = interp();
    let f = Rc::new(Func::Native { name: ip.syms.intern("f"), arity: 1, f: Rc::new(|_, a| Ok(a[0].clone())) });
    let mut d = Dict::default();
    d.set(Value::Fn(f.clone()), Value::Int(1)).unwrap();
    d.set(Value::Fn(f.clone()), Value::Int(2)).unwrap();
    assert_eq!(d.len(), 1);
    assert!(matches!(d.get(&Value::Fn(f)), Some(Value::Int(2))));
    // a function written in Klong is not a key
    ip.run("g::{x}", "<t>").unwrap();
    assert!(d.set(ip.global("g").unwrap(), Value::Int(0)).is_err());
}

#[test]
fn the_hook_runs_between_parsing_and_the_first_statement() {
    let _g = lock();
    let mut ip = interp();
    ip.on_parsed = Some(Rc::new(|ip: &mut Interp| {
        let n = ip.take_warnings().len();
        if n > 0 {
            Err(klong_core::Error::system("stop"))
        } else {
            Ok(())
        }
    }));
    assert!(show(&mut ip, "a::5 6").starts_with("!stop"));
    assert!(ip.global("a").is_none(), "the statement before the junk did not run");
    assert_eq!(show(&mut ip, "a::7"), "7");
}

#[test]
fn a_missing_right_operand_is_nothing_at_level_0() {
    let _g = lock();
    let mut ip = interp();
    assert_eq!(show(&mut ip, "[5 6]@"), "[5 6]");
    assert_eq!(show(&mut ip, "5@"), "5");
    assert_eq!(show(&mut ip, "f::"), "(no value)");
    assert!(show(&mut ip, "1+").starts_with("!missing right operand for +"));
    assert!(show(&mut ip, "{x+1}@").starts_with("!missing right operand for @"));
    assert_eq!(ip.take_warnings().iter().filter(|w| w.code == "missing-operand").count(), 3); // the three that ran: a failed run keeps none
}

#[test]
fn the_error_for_a_list_holding_undefined_names_it() {
    let _g = lock();
    let mut ip = interp();
    assert_eq!(show(&mut ip, "[1.0 :undefined]+1"), "!cannot apply + (Plus) to list of 2 and integer: the list holds :undefined");
    assert_eq!(show(&mut ip, "[1 :foo]+1"), "!cannot apply + (Plus) to list of 2 and integer: the list holds a symbol");
}

#[test]
fn float32_printing_is_the_shortest_float32_text() {
    let _g = lock();
    let mut ip = interp();
    assert_eq!(show(&mut ip, "0.1+0.2"), "0.30000000000000004");
    klong_core::fmt::FLOAT32_DISPLAY.store(true, std::sync::atomic::Ordering::Relaxed);
    let (a, b, c, d) = (show(&mut ip, "0.1+0.2"), show(&mut ip, "1%3"), show(&mut ip, "100000000.0*3"), show(&mut ip, "1e-7"));
    klong_core::fmt::FLOAT32_DISPLAY.store(false, std::sync::atomic::Ordering::Relaxed);
    assert_eq!((a.as_str(), b.as_str(), c.as_str(), d.as_str()), ("0.3", "0.33333334", "300000000.0", "1e-07"));
}

#[test]
fn another_thread_re_entering_measures_its_own_stack() {
    let _g = lock();
    let mut ip = interp();
    ip.cfg.stack_limit = Some(200_000);
    let main_prev = ip.rebase_stack(Some(200_000));
    let p: *mut Interp = &mut ip;
    let addr = p as usize;
    // a thread far from this one's stack, calling in while this one is "inside": its own base and limit, not ours
    let t = std::thread::Builder::new()
        .stack_size(4 << 20)
        .spawn(move || {
            let ip = unsafe { &mut *(addr as *mut Interp) };
            let prev = ip.rebase_stack(Some(3 << 20));
            let r = show(ip, "f::{:[x>0;1+f(x-1);0]};f(50)");
            ip.restore_stack(prev);
            r
        })
        .unwrap();
    let r = t.join().unwrap();
    ip.restore_stack(main_prev);
    assert_eq!(r, "50");
}

#[test]
fn a_short_call_reports_what_it_bound_and_what_is_left() {
    let _g = lock();
    let mut ip = interp();
    ip.run("f::{x+y+z};r::f(10);s::f(;10);t::f(1;;3)", "<t>").unwrap();
    let mut bound = |n: &str| {
        let v = ip.global(n).unwrap();
        let i = ip.fn_info(&v).unwrap();
        assert!(i.projection);
        i.bound.iter().map(|b| b.as_ref().map(|v| ip.format(v))).collect::<Vec<_>>()
    };
    let none = || None::<String>;
    assert_eq!(bound("r"), vec![Some("10".to_string()), none(), none()]);
    assert_eq!(bound("s"), vec![none(), Some("10".to_string()), none()]);
    assert_eq!(bound("t"), vec![Some("1".to_string()), none(), Some("3".to_string())]);
}

#[test]
fn f_called_with_nothing_is_a_projection() {
    let _g = lock();
    let mut ip = interp();
    ip.run("f::{x+1};g::{x+y}", "<t>").unwrap();
    for (src, holes) in [("f()", 1), ("g()", 2)] {
        let v = ip.run(src, "<t>").unwrap().unwrap();
        let i = ip.fn_info(&v).unwrap();
        assert!(i.projection, "{src}: {} {:?} kind {}", ip.format(&v), i.arity, i.kind);
        assert_eq!(i.bound.len(), holes, "{src}");
        assert!(i.bound.iter().all(|b| b.is_none()), "{src}");
    }
}

#[test]
fn a_body_that_is_a_quoted_symbol_gives_the_symbol_at_level_2() {
    let _g = lock();
    let mut ip = Interp::new(Config { strict: 2, ..Config::default() }, Box::new(NoHooks));
    assert_eq!(show(&mut ip, "{:x}(1)"), ":x");
    assert_eq!(show(&mut ip, "f::{:x};f(1)"), ":x");
    assert_eq!(show(&mut ip, "{:y}(1;2)"), ":y");
}

#[test]
fn a_function_literal_span_counts_characters() {
    let _g = lock();
    let mut ip = interp();
    ip.run("s::\"é→ü\";f::{x+1}", "<t>").unwrap();
    let v = ip.global("f").unwrap();
    let i = ip.fn_info(&v).unwrap();
    assert_eq!(i.span_chars, Some((12, 17)));
    assert!(i.span.unwrap().0 > 12, "bytes are more than characters here");
}

#[test]
fn a_name_close_to_a_local_is_offered_and_the_module_is_named() {
    let _g = lock();
    let mut ip = Interp::new(Config { strict: 2, ..Config::default() }, Box::new(NoHooks));
    let e = ip.run("f::{[total];total::x;totl}", "<t>").and_then(|_| ip.run("f(1)", "<t>")).err().unwrap();
    assert_eq!(e.hint.as_deref(), Some("did you mean total?"));
    assert_eq!(ip.current_module(), None);
}

#[test]
fn power_of_an_integer_by_a_float_is_the_system_pow() {
    let _g = lock();
    let mut ip = interp();
    for (src, a, b) in [("2^-0.5", 2.0f64, -0.5f64), ("100^0.37", 100.0, 0.37), ("3^0.1", 3.0, 0.1)] {
        let want = ip.format(&Value::Flt(a.powf(b)));
        assert_eq!(show(&mut ip, src), want, "{src}");
    }
    assert_eq!(show(&mut ip, "2^-0.5"), "0.7071067811865476");
}

#[test]
fn an_operator_or_a_function_is_not_a_dictionary_key() {
    let _g = lock();
    let mut ip = interp();
    for src in [":{[+ 1]}", ":{[:: 1] [:b 2]}", ":{[{x} 1]}", ":{[:a 1 2]}", ":{ab}", ":{[1]}"] {
        let e = ip.run(src, "<t>").err().unwrap_or_else(|| panic!("{src} should be a syntax error"));
        assert_eq!(e.kind, klong_core::Kind::Syntax, "{src}");
        assert!(e.hint.as_deref().is_some_and(|h| h.contains("a key is a number, a string, a char or a symbol") || h.contains("[key1 value1]")), "{src}: {:?}", e.hint);
    }
    assert_eq!(ip.run(":{[+ 1]}", "<t>").err().unwrap().message, "an operator cannot be a dictionary key");
    // what is a key still is one: a symbol, a string, a char, a number, and the pair as a two-character string
    assert_eq!(show(&mut ip, ":{[:a 1] [\"b\" 2] [1 3]}@:a"), "1");
    assert_eq!(show(&mut ip, ":{\"ab\"}@\"a\""), "\"b\"");
}

#[test]
fn the_message_names_a_list_of_empty_lists_as_a_matrix() {
    let _g = lock();
    let mut ip = interp();
    let e = ip.run("\"abc\"@[[]]", "<t>").err().unwrap();
    assert_eq!(e.message, "cannot apply @ (At/Index) to string and 1x0 matrix");
    let e = ip.run("\"abc\"@[[] []]", "<t>").err().unwrap();
    assert_eq!(e.message, "cannot apply @ (At/Index) to string and 2x0 matrix");
    // a list of one is still a list of one
    let e = ip.run("\"abc\"@[1.5]", "<t>").err().unwrap();
    assert!(e.message.contains("list of 1"), "{}", e.message);
}

#[test]
fn a_host_starts_and_stops_a_module() {
    let _g = lock();
    let mut ip = interp();
    ip.set_module(Some("geo")).unwrap();
    assert_eq!(ip.current_module(), Some("geo"));
    ip.run("unit::5", "<t>").unwrap();
    ip.set_module(None).unwrap();
    assert_eq!(ip.current_module(), None);
    // the name belongs to the module, and is found from outside it as the program's own `.module` leaves it
    assert!(ip.global_names().iter().any(|n| n == "unit`geo"));
    assert_eq!(show(&mut ip, "unit"), "5");
}

#[test]
fn read_datum_says_how_much_it_used_and_whether_it_is_whole() {
    let _g = lock();
    let mut ip = interp();
    let (used, v, whole) = ip.read_datum_ex("[1 2 3] rest").unwrap().unwrap();
    assert_eq!((used, whole, ip.format(&v).as_str()), (7, true, "[1 2 3]"));
    let (_, _, whole) = ip.read_datum_ex("[1 2").unwrap().unwrap();
    assert!(!whole, "an open list is not whole");
    let (used, _, whole) = ip.read_datum_ex("12").unwrap().unwrap();
    assert_eq!((used, whole), (2, false));
    let (used, _, whole) = ip.read_datum_ex("12 ").unwrap().unwrap();
    assert_eq!((used, whole), (2, true));
    assert!(ip.read_datum_ex("  \n").unwrap().is_none());
}

/// A host value that stands for a dictionary somewhere else (a remote dictionary): the host answers `x@key`, and records what is put in.
struct Remote {
    puts: std::cell::RefCell<Vec<String>>,
}
impl Ext for Remote {
    fn type_name(&self) -> &str {
        "remote"
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn host_dyad(&self, ip: &mut Interp, glyph: &str, _x: &Value, y: &Value) -> Option<klong_core::error::Result<Value>> {
        match glyph {
            "@" => {
                let key = match y {
                    Value::Str(s) => s.as_str().to_string(),
                    _ => return None,
                };
                Some(match key.as_str() {
                    "n" => Ok(Value::Int(42)),
                    "f" => Ok(Value::Fn(Rc::new(Func::Native { name: ip.syms.intern("remote-f"), arity: 1, f: Rc::new(|_, a| Ok(a[0].clone())) }))),
                    "inner" => Ok(Value::Dict(Rc::new(std::cell::RefCell::new({
                        let mut d = Dict::default();
                        d.set(Value::text("k").unwrap(), Value::Int(7)).unwrap();
                        d
                    })))),
                    _ => Err(klong_core::Error::name(format!("the remote has no entry {key}"))),
                })
            }
            "," => {
                self.puts.borrow_mut().push(ip.format(y));
                Some(Ok(Value::Int(0)))
            }
            _ => None,
        }
    }
}

#[test]
fn a_qualified_name_on_a_host_value_asks_the_host_as_at_does() {
    let _g = lock();
    let mut ip = interp();
    let h = Value::Ext(Rc::new(ExtBox(Box::new(Remote { puts: Default::default() }))));
    ip.define_global("h", h.clone()).unwrap();
    // h.n is h@"n": the host table's answer
    assert_eq!(show(&mut ip, "h.n"), "42");
    assert_eq!(show(&mut ip, "h@\"n\""), "42");
    // h.f(1) calls what the host gave back
    assert_eq!(show(&mut ip, "h.f(5)"), "5");
    // what the host gave back is walked on: a dictionary below a host value
    assert_eq!(show(&mut ip, "h.inner.k"), "7");
    // inside a function, through a parameter
    assert_eq!(show(&mut ip, "g::{[d];d::x;d.n+1};g(h)"), "43");
    // a write is the pair put to the host
    assert_eq!(show(&mut ip, "h.k::3"), "3");
    let Value::Ext(e) = &h else { unreachable!() };
    let puts = e.0.as_any().downcast_ref::<Remote>().unwrap().puts.borrow().clone();
    assert_eq!(puts, vec!["[\"k\" 3]".to_string()]);
    // the host's own error is the error
    assert_eq!(show(&mut ip, "h.nope"), "!the remote has no entry nope");
}

#[test]
fn float_width_32_computes_floats_as_float32_and_64_is_the_default() {
    let _g = lock();
    let mut ip = interp();
    // off by default: float64 results, bit for bit
    assert_eq!(show(&mut ip, "0.1+0.2"), "0.30000000000000004");
    assert_eq!(show(&mut ip, "(0.1+0.2)=0.3"), "0");
    ip.set_float_width(32);
    klong_core::fmt::FLOAT32_DISPLAY.store(true, std::sync::atomic::Ordering::Relaxed);
    let want = |src: &str| match ip_run(src) {
        Some(Value::Flt(f)) => f,
        v => panic!("{src}: {}", v.map(|_| "not a float").unwrap_or("nothing")),
    };
    fn ip_run(src: &str) -> Option<Value> {
        let mut ip = interp();
        ip.set_float_width(32);
        ip.run(src, "<t>").ok().flatten()
    }
    // each result is the float32 that torch gives for the same float32 operands
    assert_eq!(want("0.1+0.2"), (0.1f32 + 0.2f32) as f64);
    assert_eq!(want("1%3"), (1.0f32 / 3.0f32) as f64);
    assert_eq!(want("0.1*3"), (0.1f32 * 3.0f32) as f64);
    assert_eq!(want("16777217.0+0"), 16777216.0);
    assert_eq!(want("a::0.1;a+a+a"), ((0.1f32 + 0.1f32) + 0.1f32) as f64);
    assert_eq!(show(&mut ip, "0.1+0.2"), "0.3");
    assert_eq!(show(&mut ip, "(0.1+0.2)=0.3"), "1");
    assert_eq!(show(&mut ip, "1%3"), "0.33333334");
    // a flat float vector is rounded too; integers are untouched
    assert_eq!(show(&mut ip, "+/[0.1 0.2 0.3]"), "0.6");
    assert_eq!(show(&mut ip, "{x*x}'[1 2 3]"), "[1 4 9]");
    // back to 64
    ip.set_float_width(64);
    klong_core::fmt::FLOAT32_DISPLAY.store(false, std::sync::atomic::Ordering::Relaxed);
    assert_eq!(show(&mut ip, "0.1+0.2"), "0.30000000000000004");
}
