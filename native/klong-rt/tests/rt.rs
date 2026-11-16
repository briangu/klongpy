//! Ticket 103: the system functions over a `Platform` whose files are in memory.
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
    assert_eq!(run(&mut rt, ".l(\"nstat\");mu([1 2 3 4])"), Ok("2.5".into()));
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
    assert_eq!(run(&mut rt, "(.ver())?\"version\""), Ok("\"0.0.1\"".into()));
    assert_eq!(run(&mut rt, ".rn()<1"), Ok("1".into()));
    assert_eq!(run(&mut rt, ".E(\"zz::5\");zz"), Ok("5".into()));
    assert_eq!(run(&mut rt, ".py(\"math\")"), Err(Kind::Name));
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

