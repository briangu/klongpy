//! The system functions over a `Platform` whose files are in memory.
use klong_core::{Config, Kind};
use klong_rt::memfs::MemFs;
use klong_rt::{FileSystem, FsError, Mode, Platform, Runtime};

#[derive(Default)]
struct Board {
    out: Vec<u8>,
    input: Vec<u8>,
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
    fn console_read(&mut self, buf: &mut [u8]) -> usize {
        let n = buf.len().min(self.input.len());
        buf[..n].copy_from_slice(&self.input[..n]);
        self.input.drain(..n);
        n
    }
    fn console_write(&mut self, bytes: &[u8]) {
        self.out.extend_from_slice(bytes);
    }
    fn fs(&mut self) -> Option<&mut dyn FileSystem> {
        Some(&mut self.fs)
    }
}

fn rt() -> Runtime<Board> {
    Runtime::new(Board::default(), Config::default(), None)
}

/// (One interpreter at a time: see klong-core/tests/full.rs.)
static ONE: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn run(rt: &mut Runtime<Board>, text: &str) -> Result<String, Kind> {
    let _one = ONE.lock().unwrap_or_else(|e| e.into_inner());
    match rt.run(text, "<test>") {
        Ok(Some(v)) => klong_core::fmt::format_value(&rt.interp.syms, &v, false).map_err(|e| e.kind),
        Ok(None) => Ok(String::new()),
        Err(e) => Err(e.kind),
    }
}

fn out(rt: &mut Runtime<Board>) -> String {
    String::from_utf8(std::mem::take(&mut rt.shared.platform.borrow_mut().out)).expect("utf-8")
}

#[test]
fn files_round_trip() {
    let mut rt = rt();
    let prog = ".tc(.oc(\"t\"));.w([1 2 \"x\"]);.p(\"\");.p(\"line2\");.cc(.tc(0));.fc(.ic(\"t\"));a::.r();b::.rl();c::.rl();m::.mi();d::.rl();.cc(.fc(0));a,b,c,m,d,.mi()";
    assert_eq!(run(&mut rt, prog), Ok("[1 2 \"x\" \"\" \"line2\" 1 \"\" 1]".into()));
    assert_eq!(rt.shared.platform.borrow().fs.get("t"), Some(&b"[1 2 \"x\"]\nline2\n"[..]));
    assert_eq!(run(&mut rt, ".tc(.ac(\"t\"));.d(\"é\");.cc(.tc(0));.fc(.ic(\"t\"));.rl();.rl();s::.rl();.cc(.fc(0));s"), Ok("\"é\"".into()));
    assert_eq!(run(&mut rt, ".df(\"t\");.ic(\"t\")"), Err(Kind::System));
    assert_eq!(run(&mut rt, ".df(\"t\")"), Err(Kind::System));
    assert_eq!(out(&mut rt), "");
}

#[test]
fn console_functions() {
    let mut rt = rt();
    rt.shared.platform.borrow_mut().input.extend_from_slice("first\n[1 2] rest\n".as_bytes());
    assert_eq!(run(&mut rt, ".p(\"a\");.d(1);.w(\"b\");.pf(\"{}-{:4}|\";[1 2]);a::.rl();b::.r();a,,b"), Ok("[\"first\" [1 2]]".into()));
    assert_eq!(out(&mut rt), "a\n1\"b\"1-2   |\n");
    assert_eq!(run(&mut rt, ".fmt(\"{1}{0:-4}\";[1 \"a\"])"), Ok("\"a   1\"".into()));
    assert_eq!(run(&mut rt, ".fmt(\"{} {0}\";[1 2])"), Err(Kind::Range));
    assert_eq!(run(&mut rt, ".rs(\"[1 :a \"\"s\"\"]\")"), Ok("[1 :a \"s\"]".into()));
}

#[test]
fn load_is_all_or_nothing() {
    let mut rt = rt();
    rt.shared.platform.borrow_mut().fs.put("good.kg", b"a::7;b::0\n.module(:m);q::1;.module(0)\n").expect("put");
    rt.shared.platform.borrow_mut().fs.put("bad.kg", b"a::1\nb::2\nnew::3\nc::1+\"x\"\n").expect("put");
    assert_eq!(run(&mut rt, ".l(\"good\");a,b,q"), Ok("[7 0 1]".into()));
    assert_eq!(run(&mut rt, ".l(\"bad.kg\")"), Err(Kind::Type));
    assert_eq!(run(&mut rt, "a,b"), Ok("[7 0]".into()));
    // (a name the failed file created is undefined again: a symbol in the default mode)
    assert_eq!(run(&mut rt, "new"), Ok(":new".into()));
    assert_eq!(run(&mut rt, ".l(\"nowhere\")"), Err(Kind::System));
}

#[cfg(feature = "lib")]
#[test]
fn the_library_is_in_the_image() {
    let mut rt = rt();
    assert_eq!(run(&mut rt, ".l(\"nstat\");nstat.mu([1 2 3 4])"), Ok("2.5".into()));
    assert_eq!(run(&mut rt, ".l(\"set\");union([1 2];[2 3])"), Ok("[1 2 3]".into()));
    // a file of the same name on the load path wins
    rt.shared.platform.borrow_mut().fs.put("set.kg", b"union::{0}\n").expect("put");
    assert_eq!(run(&mut rt, ".l(\"set\");union([1 2];[2 3])"), Ok("0".into()));
}

#[test]
fn memfs_seek_gap_reads_zero_and_capacity_is_enforced() {
    let mut fs = MemFs::new();
    let f = fs.open("x", Mode::Write).expect("open");
    fs.write(f, b"ab").expect("write");
    fs.seek(f, 5).expect("seek");
    fs.write(f, b"c").expect("write");
    fs.close(f);
    assert_eq!(fs.get("x"), Some(&b"ab\0\0\0c"[..]));
    assert_eq!(fs.open("y", Mode::Read).err(), Some(FsError::NotFound));
    fs.capacity = 8;
    let f = fs.open("z", Mode::Write).expect("open");
    assert_eq!(fs.write(f, b"0123456789").err(), Some(FsError::NoSpace));
    fs.close(f);
    fs.rename("x", "w").expect("rename");
    let mut names = Vec::new();
    fs.list("", &mut |n, _| names.push(n.to_string())).expect("list");
    names.sort();
    assert_eq!(names, ["w", "z"]);
    fs.remove("w").expect("remove");
    assert_eq!(fs.remove("w").err(), Some(FsError::NotFound));
}

#[test]
fn version_and_misc() {
    let mut rt = rt();
    assert_eq!(run(&mut rt, "(.ver())?\"version\""), Ok("\"2.0.0-rc.1\"".into()));
    assert_eq!(run(&mut rt, ".rn()<1"), Ok("1".into()));
    assert_eq!(run(&mut rt, ".E(\"zz::5\");zz"), Ok("5".into()));
    assert_eq!(run(&mut rt, ".py(\"math\")"), Err(Kind::Name));
}

/// `.seed(n)`: the sequence of `.rn()` after it is the same on every target (xorshift64* from splitmix64 of n); 64-bit floats below.
#[test]
fn seed_makes_the_random_numbers_reproducible() {
    let ok = |text: &str, want: &str| assert_eq!(run(&mut rt(), text).as_deref(), Ok(want), "{}", text);
    ok(".seed(7)", "7");
    ok(".seed(42);a::.rn();.seed(42);a=.rn()", "1");
    ok(".seed(42);a::.rn();.seed(43);a=.rn()", "0");
    ok(".seed(1);a::.rn();b::.rn();.seed(1);c::.rn();d::.rn();(a=c)&b=d", "1");
    ok(".seed(-1);a::.rn();.seed(-1);a=.rn()", "1");
    ok(".seed(0);a::.rn();(a>0)&a<1", "1");
    #[cfg(not(feature = "float32"))]
    {
        ok(".seed(42);.rn()", "0.1941059175341826");
        ok(".seed(42);.rn();.rn()", "0.5626318272656207");
        ok(".seed(42);.rn();.rn();.rn()", "0.4861061377100522");
    }
    assert_eq!(run(&mut rt(), ".seed(1.5)"), Err(Kind::Type));
    assert_eq!(run(&mut rt(), ".seed(\"a\")"), Err(Kind::Type));
    assert_eq!(run(&mut rt(), ".seed()"), Err(Kind::Arity));
}

/// `.now()` is the date and time as seconds since 1970, where the platform has a clock that knows them; a board that does not says so.
#[test]
fn now_needs_a_wall_clock() {
    assert_eq!(run(&mut rt(), ".now()"), Err(Kind::System));
}

#[test]
fn modules() {
    let ok = |text: &str, want: &str| assert_eq!(run(&mut rt(), text).as_deref(), Ok(want), "{}", text);
    ok(".module(:geo);unit::10;scale::{x*unit};.module(0);scale(3),unit", "[30 10]");
    ok(".module(:geo);unit::10;scale::{x*unit};.module(0);unit::2;scale(3),unit", "[30 2]");
    ok("v::1;.module(:m);v::5;.module(0);v", "5");
    ok(".module(:a);n::1;.module(0);.module(:b);n::2;.module(0);n", "2");
    ok(".module(:a);n::1;f::{n};.module(0);.module(:b);n::2;.module(0);f()", "1");
}


/// `.try(f; args)` gives an error as a value, `.err(kind; message)` raises one.
#[test]
fn try_gives_errors_as_values_and_err_raises_them() {
    let ok = |text: &str, want: &str| assert_eq!(run(&mut rt(), text).as_deref(), Ok(want), "{}", text);
    ok(".try({x+y};[1 2])", "[1 3]");
    ok(".try({x};,[1 2 3])", "[1 [1 2 3]]");
    ok("f::{42};.try(f;[])", "[1 42]");
    // a function literal with no parameters, written as the function, is that function: the error it raises is caught
    ok(".try({7};[])", "[1 7]");
    ok("((.try({1+\"a\"};[]))@1)@\"kind\"", "\"type\"");
    ok("(.try({1+\"a\"};[]))@0", "0");
    // where a function is not wanted, a literal without parameters is still called where it is written (classic Klong)
    ok("f::{x+1};f({41})", "42");
    // the arguments are a list: a number or a string is a type error
    assert_eq!(run(&mut rt(), ".try({x+1};1)"), Err(Kind::Type));
    assert_eq!(run(&mut rt(), ".try({7};\"\")"), Err(Kind::Type));
    // the caught error is a dictionary: kind, message, source, line and col
    ok("e::(.try({x+y};[1 \"a\"]))@1;(e@\"kind\"),\"/\",e@\"message\"", "\"type/cannot apply + (Plus) to integer and string\"");
    ok("e::(.try({[1 2 3]@x};,9))@1;(e@\"kind\"),\"/\",e@\"message\"", "\"range/index 9 out of range for list of 3\"");
    ok("((.try(.E;,\"1+\"))@1)@\"kind\"", "\"syntax\"");
    ok("((.try({nosuch(x)};,1))@1)@\"kind\"", "\"name\"");
    // where: the program text, and the line and the column of what failed (a build with diag points at the operator, one without at the
    // start of the expression: the column of the operator is 12 there, of its first operand 11)
    ok("e::(.try({x+y};[1 \"a\"]))@1;(e@\"source\"),\"/\",$e@\"line\"", "\"<test>/1\"");
    let col = run(&mut rt(), "e::(.try({x+y};[1 \"a\"]))@1;e@\"col\"").expect("a column");
    assert!(col == "11" || col == "12", "{col}");
    ok("(.try({x+y};[1 \"a\"]))@0", "0");
    // every kind a program may raise comes back as itself, with its message
    for kind in ["syntax", "name", "arity", "type", "shape", "range", "system", "memory", "internal", "remote"] {
        ok(&format!("e::(.try({{.err(x;\"n1: gone\")}};,\"{kind}\"))@1;(e@\"kind\"),\"/\",e@\"message\""), &format!("\"{kind}/n1: gone\""));
        ok(&format!("e::(.try({{.err(x;\"m\")}};,:{kind}))@1;(e@\"kind\"),\"/\",e@\"message\""), &format!("\"{kind}/m\""));
        // what `.try` caught is raised again by `.err` as what it was
        ok(&format!("e::(.try({{.err(x;\"m\")}};,:{kind}))@1;(.try({{.err(x)}};,e))@1;((.try({{.err(x)}};,e))@1)@\"kind\""), &format!("\"{kind}\""));
    }
    // the evaluation goes on after a trapped error, at any depth
    ok("g::{.try({.err(\"system\";x)};,x)};a::g(\"one\");b::.try({1+g(x)};,\"two\");((a@1)@\"message\"),\"/\",(b@1)@\"kind\"", "\"one/type\"");
    let err = |text: &str, want: Kind| assert_eq!(run(&mut rt(), text), Err(want), "{}", text);
    err(".err(\"type\";\"bad\")", Kind::Type);
    err(".err(\"system\";\"n1: link down\")", Kind::System);
    err(".err(\"nope\";\"x\")", Kind::Range);
    err(".err(\"interrupt\";\"x\")", Kind::Range);
    err(".err(\"type\";5)", Kind::Type);
    err(".err(5;\"x\")", Kind::Type);
    err(".err(5)", Kind::Arity);
    err(".err(:{;\"kind\";\"range\"})", Kind::Type);
    err(".err(:{;\"kind\";\"range\";\"message\";\"from a dictionary\"})", Kind::Range);
    err(".err()", Kind::Arity);
    err(".try(5;[])", Kind::Type);
    err(".try({x};\"ab\")", Kind::Type);
    // a stop is not for the program to swallow
    err(".try({.x(x)};,0);1", Kind::Interrupt);
    // and the message is the one given
    let mut r = rt();
    let _one = ONE.lock().unwrap_or_else(|e| e.into_inner());
    let e = r.run(".err(\"range\";\"n2: index 9\")", "<test>").err().expect("an error");
    assert_eq!((e.kind, &*e.message), (Kind::Range, "n2: index 9"));
}

#[test]
fn string_functions() {
    let mut rt = rt();
    let cases: &[(&str, Result<&str, Kind>)] = &[
        (".split(\"a,,b\";\",\")", Ok("[\"a\" \"\" \"b\"]")),
        (".split(\"\";\",\")", Ok("[\"\"]")),
        (".split(\"aXYbXY\";\"XY\")", Ok("[\"a\" \"b\" \"\"]")),
        (".split(\"hé,é\";0c,)", Ok("[\"hé\" \"é\"]")),
        (".split(\"a\";\"\")", Err(Kind::Range)),
        (".split(1;\",\")", Err(Kind::Type)),
        (".split(0ca;\",\")", Err(Kind::Type)),
        (".split(\"a\";1)", Err(Kind::Type)),
        (".join([\"a\" \"b\"];\", \")", Ok("\"a, b\"")),
        (".join([];\",\")", Ok("\"\"")),
        (".join([0ca \"b\"];0c-)", Ok("\"a-b\"")),
        (".join([\"a\" 1];\",\")", Err(Kind::Type)),
        (".join(\"ab\";\",\")", Err(Kind::Type)),
        (".words(\" a\tb\r\nc  \")", Ok("[\"a\" \"b\" \"c\"]")),
        (".words(\"  \")", Ok("[]")),
        (".trim(\" \t\r\nhé \")", Ok("\"hé\"")),
        (".num(\"12\")", Ok("12")),
        (".num(\" -3.5\r\n\")", Ok("-3.5")),
        (".num(\"7 8\")", Ok(":undefined")),
        (".num(\"x\")", Ok(":undefined")),
        (".num(\"\")", Ok(":undefined")),
        (".num(\"1e\")", Ok(":undefined")),
        (".num(\"٣\")", Ok(":undefined")),
        (".num(5)", Err(Kind::Type)),
    ];
    for (src, want) in cases {
        assert_eq!(run(&mut rt, src), want.map(String::from), "{src}");
    }
}

#[test]
fn json_functions() {
    let mut rt = rt();
    // what a build cannot hold is a range error (the conformance cases cover the rest: KlongPy
    // has big integers and keeps a lone surrogate)
    let cases: &[(&str, Result<&str, Kind>)] = &[
        #[cfg(not(feature = "bigint"))]
        (".unjson(\"12345678901234567890\")", Err(Kind::Range)),
        #[cfg(feature = "bigint")]
        (".unjson(\"12345678901234567890\")", Ok("12345678901234567890")),
        #[cfg(not(feature = "bigint"))]
        (".unjson(\"[1,9223372036854775808]\")", Err(Kind::Range)),
        #[cfg(not(feature = "bigint"))]
        (".unjson(\"-9223372036854775809\")", Err(Kind::Range)),
        #[cfg(feature = "bigint")]
        (".unjson(\"[1,9223372036854775808]\")", Ok("[1 9223372036854775808]")),
        #[cfg(feature = "bigint")]
        (".unjson(\"-9223372036854775809\")", Ok("-9223372036854775809")),
        (".unjson(\"-9223372036854775808\")", Ok("-9223372036854775808")),
        (".unjson(\"\"\"\\ud800\"\"\")", Err(Kind::Range)),
        (".unjson(\"\"\"\\ude00\"\"\")", Err(Kind::Range)),
        (".unjson(\"\"\"\\ud83d\\ude00\"\"\")", Ok("\"😀\"")),
        (".json(.unjson(\"{\"\"a\"\":[1,2.5,null,true]}\"))", Ok("\"{\"\"a\"\":[1,2.5,null,1]}\"")),
    ];
    for (src, want) in cases {
        assert_eq!(run(&mut rt, src), want.map(String::from), "{src}");
    }
    // the message of bad text, as Python's json module words it
    let _one = ONE.lock().unwrap_or_else(|e| e.into_inner());
    let e = rt.run(".unjson(\"[1,\")", "<test>").err().expect("an error");
    assert_eq!((e.kind, &*e.message), (Kind::Range, ".unjson: Expecting value at line 1 column 4 (character 3)"));
    let e = rt.run(".json(:{;\"a\";[;1;:b]})", "<test>").err().expect("an error");
    assert_eq!((e.kind, &*e.message), (Kind::Type, ".json: a symbol has no JSON form at key \"a\" at index 1"));
}

#[test]
fn load_finds_a_file_next_to_the_program() {
    let mut rt = rt();
    {
        let mut b = rt.shared.platform.borrow_mut();
        b.fs.put("proj/common.kg", b"inc::{x+1}\n").expect("put");
        b.fs.put("proj/a/outer.kg", b".l(\"inner\");outer::{inner(x)+1}\n").expect("put");
        b.fs.put("proj/a/inner.kg", b"inner::{x*10}\n").expect("put");
        b.fs.put("here.kg", b"inc::{x+1000}\n").expect("put");
    }
    // no program file: the name is not found
    assert_eq!(run(&mut rt, ".l(\"common\")"), Err(Kind::System));
    rt.set_program("proj/prog.kg");
    assert_eq!(run(&mut rt, ".l(\"common\");inc(1)"), Ok("2".into()));
    assert_eq!(run(&mut rt, ".l(\"common.kg\");inc(2)"), Ok("3".into()));
    // a nested load looks next to the file it is in, not next to the program
    assert_eq!(run(&mut rt, ".l(\"a/outer\");outer(4)"), Ok("41".into()));
    assert_eq!(run(&mut rt, ".l(\"inner\")"), Err(Kind::System));
    // the directory of the file that calls .l comes first, then the load path
    rt.shared.platform.borrow_mut().fs.put("common.kg", b"inc::{x+1000}\n").expect("put");
    assert_eq!(run(&mut rt, ".l(\"common\");inc(1)"), Ok("2".into()));
    rt.set_load_path(vec!["lp".into()]);
    rt.shared.platform.borrow_mut().fs.put("lp/only.kg", b"inc::{x+7}\n").expect("put");
    assert_eq!(run(&mut rt, ".l(\"only\");inc(1)"), Ok("8".into()));
}

#[cfg(feature = "diag")]
#[test]
fn load_names_the_places_it_looked() {
    let mut rt = rt();
    rt.set_load_path(vec!["other".into(), "sub/".into()]);
    rt.shared.platform.borrow_mut().fs.put("sub/s.kg", b"sv::5\n").expect("put");
    // a name with a directory and the extension, a path that ends in a slash
    assert_eq!(run(&mut rt, ".l(\"s.kg\");sv"), Ok("5".into()));
    let e = rt.run(".l(\"nothere\")", "<test>").err().expect("an error");
    assert_eq!(&*e.message, "file does not exist: nothere (looked for nothere, nothere.kg, other/nothere, other/nothere.kg, sub/nothere, sub/nothere.kg, then the bundled library)");
    let e = rt.run(".l(\"/nowhere/x\")", "<test>").err().expect("an error");
    assert_eq!(&*e.message, "file does not exist: /nowhere/x (looked for /nowhere/x, /nowhere/x.kg)");
    rt.set_program("proj/prog.kg");
    let e = rt.run(".l(\"nothere.kg\")", "<test>").err().expect("an error");
    assert_eq!(&*e.message, "file does not exist: nothere.kg (looked for proj/nothere.kg, other/nothere.kg, sub/nothere.kg, then the bundled library)");
}

#[test]
fn the_current_module_is_named() {
    let mut rt = rt();
    assert_eq!(rt.interp.current_module(), None);
    assert_eq!(run(&mut rt, ".module(:m);1"), Ok("1".into()));
    assert_eq!(rt.interp.current_module(), Some("m"));
    assert_eq!(run(&mut rt, ".module(0);1"), Ok("1".into()));
    assert_eq!(rt.interp.current_module(), None);
}

#[test]
fn read_takes_one_object_and_leaves_the_rest() {
    let mut rt = rt();
    // a file: the rest of the line, the next line, an object that spans lines, a number that ends with the file
    run(&mut rt, ".tc(.oc(\"t\"));.p(\"[1 2 3 4];\"\"hello\"\"\");.p(\"[5\");.p(\" 6] tail\");.p(\"77\");.cc(.tc(0))").unwrap();
    run(&mut rt, ".fc(.ic(\"t\"))").unwrap();
    assert_eq!(run(&mut rt, ".r()"), Ok("[1 2 3 4]".into()));
    // the rest of the stream is unread: the next character is the ;
    assert_eq!(run(&mut rt, ".rl()"), Ok("\";\"\"hello\"\"\"".into()));
    assert_eq!(run(&mut rt, ".r()"), Ok("[5 6]".into()));
    assert_eq!(run(&mut rt, ".rl()"), Ok("\" tail\"".into()));
    assert_eq!(run(&mut rt, ".r()"), Ok("77".into()));
    assert_eq!(run(&mut rt, ".r()"), Ok(":undefined".into()));
    assert_eq!(run(&mut rt, ".mi()"), Ok("0".into()));
}

#[test]
fn read_on_the_console_asks_for_no_more_than_the_object_needs() {
    let mut rt = rt();
    // typed lines: a closed list, then (on the same line) more; the second read must not wait for input that is not there
    rt.shared.platform.borrow_mut().input.extend_from_slice("[1 2] [3 4]\nline\n".as_bytes());
    assert_eq!(run(&mut rt, ".r()"), Ok("[1 2]".into()));
    assert_eq!(run(&mut rt, ".r()"), Ok("[3 4]".into()));
    // the rest of the line is what .rl then reads
    assert_eq!(run(&mut rt, ".rl()"), Ok("\"\"".into()));
    assert_eq!(run(&mut rt, ".rl()"), Ok("\"line\"".into()));
}
