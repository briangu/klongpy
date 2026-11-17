//! Strings as code points, format2 and form, the dictionary, modules, the bounds
//! on nesting and on names, and what fuzzing found.
use klong_core::{Config, Interp, Kind, NoHooks};

fn interp() -> Interp {
    Interp::new(Config::default(), Box::new(NoHooks))
}

/// The bound on nesting and the failures deferred from where no error can be returned are
/// kept per process (the core has no thread-local storage): one interpreter runs at a time.
/// (An interpreter sets the bound when a program starts, so creating one while another test
/// runs is harmless; two programs running at once, in two threads, would still interfere:
/// that is the limitation, and every test that runs a program takes this lock.)
static ONE: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn lock() -> std::sync::MutexGuard<'static, ()> {
    ONE.lock().unwrap_or_else(|e| e.into_inner())
}

/// Run a program; the caller holds `lock()` for as long as the interpreter and its values live
/// (dropping deeply nested data records a deferred failure too, and that must not reach
/// another test's program).
fn run_with(ip: &mut Interp, text: &str) -> Result<String, Kind> {
    match ip.run(text, "<test>") {
        Ok(Some(v)) => klong_core::fmt::format_value(&ip.syms, &v, false).map_err(|e| e.kind),
        Ok(None) => Ok(String::new()),
        Err(e) => Err(e.kind),
    }
}

fn run(text: &str) -> Result<String, Kind> {
    let _one = lock();
    let mut ip = interp();
    run_with(&mut ip, text)
}

fn ok(text: &str, want: &str) {
    assert_eq!(run(text).as_deref(), Ok(want), "{}", text);
}

fn err(text: &str, want: Kind) {
    assert_eq!(run(text), Err(want), "{}", text);
}

/// On a thread with a stack of known size (a debug build uses several KiB per level).
fn big<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> T {
    std::thread::Builder::new().stack_size(256 << 20).spawn(f).expect("thread").join().expect("join")
}

#[test]
fn strings_are_code_points() {
    ok("#\"héllo\"", "5");
    ok("\"héllo\"@1", "0cé");
    ok("|\"héllo\"", "\"olléh\"");
    ok("\"héllo\"?\"l\"", "[2 3]");
    ok("\"héllo\":=0cX,1", "\"hXllo\"");
    ok("2_\"héllo\"", "\"llo\"");
    ok("(-2)#\"héllo\"", "\"lo\"");
    ok("#0cé", "233");
    ok(":#233", "0cé");
    ok("\"a\"\"b\"", "\"a\"\"b\"");
    ok("#\"a\"\"b\"", "3");
    ok("*\"abc\"", "\"a\"");
    ok("\"\"?\"\"", "[0]");
    ok("=\"\"", "[]");
    ok("<\"hélloaé\"", "[5 0 2 3 4 1 6]");
    // access by index from both ends of a long string that is not ASCII
    ok("s::10000#\"aé\";(s@9999),(s@0),(s@5000),s@4999", "\"éaaé\"");
    ok("s::10000#\"aé\";#{x}'s", "10000");
}

#[test]
fn format2_and_form() {
    ok("5$123", "\"123  \"");
    ok("(-5)$123", "\"  123\"");
    ok("5.3$123.45", "\"  123.450\"");
    ok("(-7.2)$1.5", "\"   1.50\"");
    ok("[5 -5]$[1 2]", "[\"1    \" \"    2\"]");
    ok("5$:foo", "\":foo \"");
    ok("0$:foo", "\"foo\"");
    ok("3$\"héllo\"", "\"héllo\"");
    ok("7$\"héllo\"", "\"héllo  \"");
    ok("1:$\"42\"", "42");
    ok("1.0:$\"2.5\"", "2.5");
    ok("\"\":$\"str\"", "\"str\"");
    ok("0c0:$\"x\"", "0cx");
    ok(":a:$\"name\"", ":name");
    err("5.2$3", Kind::Type);
}

#[test]
fn dictionaries() {
    ok(":{[1 2]}@1.0", "2");
    ok(":{[1.5 2]}@1.5", "2");
    ok("d:::{[1 2] [3 4]};e:::{[3 4] [1 2]};d~e", "1");
    ok(":{[1 2]}~:{[1 3]}", "0");
    ok("d:::{};d,[1 2];d,[\"a\" 3];d,[0cb 4];d,[:s 5];d", ":{[1 2] [\"a\" 3] [0cb 4] [:s 5]}");
    // as in KlongPy, a char and the string of that char are one key, and so are 1 and 1.0
    ok("d:::{};d,[\"a\" 3];d,[0ca 4];d,[1 5];d,[1.0 6];d", ":{[\"a\" 4] [1 6.0]}");
    ok("d:::{[1 2]};d,[1 3];d", ":{[1 3]}");
    ok("d:::{[\"a\" 1] [\"b\" 2] [\"c\" 3]};{d,x,,0}'[\"a\"];(\"b\"_d)", ":{[\"a\" 0] [\"c\" 3]}");
    err(":{[[1 2] 3]}", Kind::Syntax);
    err("d:::{};d,[[1 2] 3]", Kind::Type);
    // the three index forms: none (up to 8 entries), 16-bit, 32-bit; and deletion in each
    ok("d:::{};{d,x,,x*2}'!8;(d?7),#d", "[14 8]");
    ok("d:::{};{d,x,,x*2}'!1000;(d?999),(d?0),#d", "[1998 0 1000]");
    ok("d:::{};{d,x,,x*2}'!1000;{x_d}'2*!500;(d?999),(d?998),#d", "[1998 :undefined 500]");
    ok("d:::{};{d,x,,x}'!70000;(d?69999),(d?65535),#d", "[69999 65535 70000]");
    ok("d:::{};{d,($x),,x}'!300;(d?\"299\"),#d", "[299 300]");
    // insertion order survives growth and deletion
    ok("d:::{};{d,x,,x}'[5 3 9 1];3_d;d,[3 0];*'{x}'d", "[5 9 1 3]");
}

#[test]
fn a_dictionary_that_holds_itself() {
    // printing, matching and dropping stop: an error, not a stack overflow or a leak loop
    err("d:::{};d,(\"self\",,d);d", Kind::Range);
    ok("d:::{};d,(\"self\",,d);d~d", "1");
    err("d:::{};e:::{};d,(\"o\",,e);e,(\"o\",,d);d~e", Kind::Range);
    ok("d:::{};d,(\"self\",,d);#d", "1");
    // a long chain of dictionaries is dropped without recursion
    big(|| ok("d:::{};{[e];e:::{};e,(1,,d);d::e}'!100000;d::0;1", "1"));
    // and printing or matching one stops at the bound
    big(|| err("d:::{};{[e];e:::{};e,(1,,d);d::e}'!1000;d", Kind::Range));
    // (the same dictionary matches itself without a walk)
    big(|| ok("d:::{};{[e];e:::{};e,(1,,d);d::e}'!1000;c::d;d~c", "1"));
}

#[test]
fn data_nesting_is_bounded() {
    ok("#100{,x}:*1", "1");
    err("200{,x}:*1", Kind::Range);
    err("a::1;{a::,a}'!1000;1", Kind::Range);
    // projections chain too
    ok("f::{x;y};3{f(x;)}:*f(1;)", ":monad");
}

#[test]
fn a_stack_budget_bounds_recursion_whatever_a_level_costs() {
    big(|| {
        let _one = lock();
        let mut ip = Interp::new(Config::for_stack(64 * 1024), Box::new(NoHooks));
        assert_eq!(run_with(&mut ip, "f::{:[x;f(x-1);0]};f(3)"), Ok("0".into()));
        assert_eq!(run_with(&mut ip, "f::{:[x;f(x-1);0]};f(100000)"), Err(Kind::Range));
        assert!(ip.max_stack_seen <= 64 * 1024, "{}", ip.max_stack_seen);
        // the interpreter is usable afterwards
        assert_eq!(run_with(&mut ip, "1+1"), Ok("2".into()));
        // the parser honours the budget too
        let deep = format!("{}1{}", "(".repeat(20000), ")".repeat(20000));
        assert_eq!(run_with(&mut ip, &deep), Err(Kind::Range));
    });
}

#[test]
fn found_by_fuzzing_103() {
    // a chain of adverbs recursed once per adverb without a check: stack overflow on a small stack
    big(|| {
        err(&format!("1+{}2", "'".repeat(600)), Kind::Range);
        let _one = lock();
        let mut ip = Interp::new(Config::for_stack(48 * 1024), Box::new(NoHooks));
        assert_eq!(run_with(&mut ip, &format!("(-1){{x+1}}\\*{}0", "'".repeat(5000))), Err(Kind::Range));
    });
    // names are bounded (they are stored for good and copied where nothing can fail)
    err(&format!("{}::1", "a".repeat(300)), Kind::Syntax);
    err(&format!(":a:$\"{}\"", "a".repeat(300)), Kind::Range);
    ok(&format!("{}::1", "a".repeat(255)), "1");
    // a message quotes at most MSG_MAX bytes of the program
    let _one = lock();
    let e = interp().run(&format!(".f{}(1)", "a".repeat(250)), "<t>").err().map(|e| e.render()).unwrap_or_default();
    assert!(e.contains("undefined function"), "{}", e);
    let e = interp().run(&format!("1+\"{}", "é".repeat(4000)), "<t>").err().map(|e| e.message.len()).unwrap_or(0);
    assert!(e > 0 && e <= klong_core::error::MSG_MAX + 3, "{}", e);
}

#[test]
fn a_monad_takes_the_whole_conditional_or_evaluated_list() {
    // KlongPy once applied the monad to the condition (conformance: syntax.monad-on-conditional*)
    ok("-:[0;2;3]", "-3");
    ok("-:[1;2;3]", "-2");
    ok("n::{-:[x;2;3]};n(0)", "-3");
    ok("#[;1;2;3]", "3");
    ok("#:[1;[1 2 3];0]", "3");
}

#[test]
fn functions_display_their_arity() {
    ok("f::{1};f", ":nilad");
    ok("f::{x};f", ":monad");
    ok("f::{x+y};f", ":dyad");
    ok("f::{x+y+z};f", ":triad");
    ok("f(a;b;c;d)::{a+b+c+d};f", ":fn/4");
    ok("f(a;b;c;d;e)::{a+b+c+d+e};f", ":fn/5");
}
