//! Ticket 104, host part: the console driven byte by byte as a terminal or an upload tool
//! drives a board, over a serial line in memory.
use std::sync::{Mutex, MutexGuard};

use klong_console::crc::crc32;
use klong_console::{hex, Console, ConsoleConfig};
use klong_core::Config;
use klong_rt::memfs::MemFs;
use klong_rt::Runtime;
use klong_sim::{Sim, SimBoard, SimOptions, Wire};

/// One interpreter at a time (statics in klong-core: see klong-core/tests/full.rs).
static ONE: Mutex<()> = Mutex::new(());

const BANNER: &str = "Klong 0.0.1 (sim, int64 float64)\nklong> ";
const P: &str = "klong> ";

fn lock() -> MutexGuard<'static, ()> {
    ONE.lock().unwrap_or_else(|e| e.into_inner())
}

fn sim_with(o: SimOptions) -> Sim {
    let mut s = Sim::new(o);
    assert_eq!(String::from_utf8_lossy(&s.run()), BANNER);
    s
}

/// A board with a terminal on it: echo and editing.
fn terminal() -> Sim {
    sim_with(SimOptions::default())
}

/// A board with a script on it: no echo.
fn dumb() -> Sim {
    sim_with(SimOptions { console: ConsoleConfig { dumb: true, ..ConsoleConfig::default() }, ..SimOptions::default() })
}

fn put(path: &str, data: &[u8]) -> Vec<u8> {
    format!("\x01put {} {} {:08x}\n", path, data.len(), crc32(data)).into_bytes()
}

fn run_cmd(program: &[u8]) -> Vec<u8> {
    format!("\x01run {} {:08x}\n", program.len(), crc32(program)).into_bytes()
}

fn record(kind: u8, addr: u16, data: &[u8]) -> String {
    let mut line = [0u8; hex::MAX_LINE];
    let n = hex::write(kind, addr, data, &mut line);
    String::from_utf8(line[..n].to_vec()).unwrap()
}

/// The records of a file: 16 bytes each, an extended linear address record at the start of
/// every 64 KiB after the first. Without the end-of-file record.
fn records(data: &[u8]) -> Vec<String> {
    let mut out = Vec::new();
    for (i, chunk) in data.chunks(16).enumerate() {
        let at = i * 16;
        if at > 0 && at % 65536 == 0 {
            out.push(record(4, 0, &[(at >> 24) as u8, (at >> 16) as u8]));
        }
        out.push(record(0, at as u16, chunk));
    }
    out
}

const EOF: &str = ":00000001FF";

fn pattern(n: usize) -> Vec<u8> {
    (0..n).map(|i| (i * 31 + i / 251) as u8).collect()
}

// ----- the prompt --------------------------------------------------------------------------

#[test]
fn prompt_evaluates_and_prints() {
    let _one = lock();
    let mut s = terminal();
    assert_eq!(s.io(b"1+2\n"), "1+2\n3\nklong> ");
    assert_eq!(s.io(b"\"abc\"\n"), "\"abc\"\n\"abc\"\nklong> ");
    assert_eq!(s.io(b".p(\"hi\");[1 2]\n"), ".p(\"hi\");[1 2]\nhi\n[1 2]\nklong> ");
    // an error is rendered and the prompt comes back
    let out = s.io(b"[1 2 3]@5\n");
    assert!(out.contains("range error"), "{out}");
    assert!(out.ends_with(P));
    // an empty line is only a new prompt; a definition prints as in KlongPy
    assert_eq!(s.io(b"\n"), "\nklong> ");
    assert_eq!(s.io(b"   \n"), "   \nklong> ");
    assert_eq!(s.io(b"a::5\n"), "a::5\n5\nklong> ");
    assert_eq!(s.io(b"a*a\n"), "a*a\n25\nklong> ");
}

#[test]
fn line_ends_cr_lf_crlf() {
    let _one = lock();
    let mut s = dumb();
    assert_eq!(s.io(b"1\r2\n3\r\n4\n"), "1\nklong> 2\nklong> 3\nklong> 4\nklong> ");
    // the pair may be split between two reads
    assert_eq!(s.io(b"5\r"), "5\nklong> ");
    assert_eq!(s.io(b"\n"), "");
    assert_eq!(s.io(b"\n"), P);
}

#[test]
fn an_incomplete_entry_continues() {
    let _one = lock();
    let mut s = dumb();
    assert_eq!(s.io(b"f::{x*\n"), "     > ");
    assert_eq!(s.io(b"\n"), "     > ");
    assert_eq!(s.io(b"2}\n"), ":monad\nklong> ");
    assert_eq!(s.io(b"f(21)\n"), "42\nklong> ");
    // an open string takes the line end into the string, an open comment hides it
    assert_eq!(s.io(b"\"a\nb\"\n"), "     > \"a\nb\"\nklong> ");
    assert_eq!(s.io(b"1+ :\"not\nyet\" 2\n"), "     > 3\nklong> ");
    assert_eq!(s.io(b".comment(\"END\")\nanything ( [ {\nEND\n7\n"), "     >      > klong> 7\nklong> ");
    // a closer that matches nothing is not waited for: it is the parser's error
    let out = s.io(b"(1]\n");
    assert!(out.contains("syntax error") && out.ends_with(P), "{out}");
    // a trailing dyad is complete (and an error)
    let out = s.io(b"1+\n");
    assert!(out.contains("error") && out.ends_with(P), "{out}");
}

#[test]
fn ctrl_c_abandons_the_entry_and_ctrl_d_is_ignored() {
    let _one = lock();
    let mut s = terminal();
    assert_eq!(s.io(b"[1 2\n"), "[1 2\n     > ");
    assert_eq!(s.io(b"3 4"), "3 4");
    assert_eq!(s.io(b"\x03"), "^C\nklong> ");
    assert_eq!(s.io(b"1\x04+\x041\n"), "1+1\n2\nklong> ");
    assert_eq!(s.io(b"\x04"), "");
    assert_eq!(s.io(b"\x03"), "^C\nklong> ");
}

#[test]
fn ctrl_a_in_the_middle_of_a_line_is_input() {
    let _one = lock();
    let mut s = terminal();
    // after a character: echoed as ^A, part of the line, not a command
    let out = s.io(b"1\x01ver\n");
    assert!(out.starts_with("1^Aver\n"), "{out:?}");
    assert!(!out.contains("\x01ok") && out.contains("syntax error"), "{out:?}");
    assert!(out.ends_with(P));
    assert!(!s.console.busy());
    // in a continuation line the entry is not empty either
    assert_eq!(s.io(b"(\n"), "(\n     > ");
    let out = s.io(b"\x01ver\n");
    assert!(!out.contains("\x01ok"), "{out:?}");
    s.io(b"\x03");
    // at the start of an empty entry it is a command
    let out = s.io(b"\x01ver\n");
    assert!(out.starts_with("ver\n\x01ok ver=0.0.1 "), "{out:?}");
    // and after blanks it is not
    let out = s.io(b" \x01ver\n");
    assert!(!out.contains("\x01ok"), "{out:?}");
}

#[test]
fn editing_keys_reach_the_evaluator() {
    let _one = lock();
    let mut s = terminal();
    // 1+3, backspace, 2
    assert!(s.io(b"1+3\x7f2\n").ends_with("\n3\nklong> "));
    // 13, left, + in the middle
    assert!(s.io(b"13\x1b[D+\n").ends_with("\n4\nklong> "));
    // home, delete, end: 9+1 becomes +1 becomes +10
    assert!(s.io(b"9+1\x1b[H\x1b[3~\x1b[F0\n").ends_with("\n10\nklong> "));
    // history: the line before the last, again
    let out = s.io(b"\x1b[A\x1b[A\n");
    assert!(out.ends_with("\n4\nklong> "), "{out:?}");
    // in a script's console the same bytes are not keys
    let mut d = dumb();
    assert_eq!(d.io(b"1+3\x7f2\n"), "3\nklong> ");
}

#[test]
fn the_same_bytes_in_any_rhythm_give_the_same_output() {
    let _one = lock();
    let file = pattern(300);
    let mut script: Vec<u8> = Vec::new();
    script.extend(b"f::{x*\n2}\nf(21)\n1+\x1b[D\x7f2+\x1b[F1\n\x01ver\n");
    script.extend(put("a.bin", &file));
    script.extend(&file);
    script.extend(b"\x01hex b.bin\n");
    for r in records(&file) {
        script.extend(r.as_bytes());
        script.extend(b"\r\n");
    }
    script.extend(b":00000001FF\n");
    script.extend(run_cmd(b".p(\"x\");3"));
    script.extend(b".p(\"x\");3");
    script.extend(b"[1 2\n\x033*3\n");
    let mut outputs = Vec::new();
    for burst in [0usize, 1, 2, 7, 31, 33] {
        let mut s = terminal();
        s.wire.set_burst(burst);
        let out = s.io(&script);
        assert_eq!(s.file("a.bin").as_deref(), Some(&file[..]));
        assert_eq!(s.file("b.bin").as_deref(), Some(&file[..]));
        outputs.push(out);
    }
    // and one byte per poll, with the board coming to rest in between
    let mut s = terminal();
    let mut out = String::new();
    for &b in &script {
        out.push_str(&s.io(&[b]));
    }
    outputs.push(out);
    for o in &outputs[1..] {
        assert_eq!(o, &outputs[0]);
    }
    assert!(outputs[0].ends_with("3*3\n9\nklong> "), "{}", outputs[0]);
}

#[test]
fn ctrl_c_stops_a_runaway_loop() {
    let _one = lock();
    let mut s = terminal();
    s.wire.send(b"{1}{x+1}:~0\n");
    // the byte arrives while the evaluation runs: the tick hook finds it
    s.wire.send_after_idles(3, b"\x03");
    let out = String::from_utf8_lossy(&s.run()).into_owned();
    assert!(out.contains("interrupt error"), "{out}");
    assert!(out.ends_with(P));
    assert!(s.wire.idles() >= 3);
    assert_eq!(s.io(b"1+1\n"), "1+1\n2\nklong> ");
    // what was typed while it ran is kept, in order, until a Ctrl-C: that ends the
    // evaluation and discards the typed-ahead input received before it
    s.wire.send(b"{1}{x+1}:~0\n");
    s.wire.send_after_idles(2, b"2+");
    s.wire.send_after_idles(3, b"2\n\x03");
    let out = String::from_utf8_lossy(&s.run()).into_owned();
    assert!(out.contains("interrupt error"), "{out}");
    assert!(!out.contains("2+2"), "{out}");
    // input after the Ctrl-C is kept, in order
    s.wire.send(b"{1}{x+1}:~0\n");
    s.wire.send_after_idles(2, b"2+");
    s.wire.send_after_idles(3, b"2\n\x033*3\n");
    let out = String::from_utf8_lossy(&s.run()).into_owned();
    assert!(out.contains("interrupt error"), "{out}");
    assert!(out.ends_with("3*3\n9\nklong> "), "{out}");
    assert!(!out.contains("2+2"), "{out}");
    // a second Ctrl-C at the prompt only abandons the line
    assert_eq!(s.io(b"\x03"), "^C\nklong> ");
}

const LOOP: &[u8] = b"{1}{x+1}:~0\n";

/// Run the bytes sent in one burst; the text written.
fn burst(bytes: &[u8]) -> String {
    let _one = lock();
    let mut s = terminal();
    s.wire.send(bytes);
    let out = String::from_utf8_lossy(&s.run()).into_owned();
    assert!(!s.console.busy(), "{out}");
    out
}

#[test]
fn ctrl_c_in_the_same_burst_as_the_line_that_starts_the_loop() {
    // the byte is already read (ahead) when the evaluation starts
    for tail in [&b"\x03"[..], b"\x03\n", b"\x03\n1+1\n", b"\x03\x03\n1+1\n", b"\x03\r\n1+1\r\n"] {
        let mut input = LOOP.to_vec();
        input.extend(tail);
        let out = burst(&input);
        assert!(out.contains("interrupt error"), "{tail:?}: {out}");
        assert!(out.ends_with(P), "{tail:?}: {out}");
        if tail.ends_with(b"1+1\n") || tail.ends_with(b"1+1\r\n") {
            assert!(out.contains("1+1\n2\nklong> "), "{tail:?}: {out}");
        }
    }
    // typed before the Ctrl-C: discarded; after it: kept
    let out = burst(b"{1}{x+1}:~0\n2+\x03\n3*3\n");
    assert!(out.contains("interrupt error") && out.ends_with("3*3\n9\nklong> ") && !out.contains("2+"), "{out}");
    let out = burst(b"{1}{x+1}:~0\n7*7\n\x03\n3*3\n");
    assert!(out.contains("interrupt error") && out.ends_with("3*3\n9\nklong> ") && !out.contains("7*7"), "{out}");
    // the same one byte at a time, and read in bursts of every size
    for n in 1..8 {
        let _one = lock();
        let mut s = terminal();
        s.wire.set_burst(n);
        s.wire.send(b"{1}{x+1}:~0\n\x03\n1+1\n");
        let out = String::from_utf8_lossy(&s.run()).into_owned();
        assert!(out.contains("interrupt error") && out.ends_with("1+1\n2\nklong> "), "burst {n}: {out}");
    }
}

#[test]
fn a_ctrl_c_after_a_short_evaluation_is_read_at_the_prompt() {
    // read ahead with the line, and the evaluation ends before its first tick: nothing was
    // interrupted, the Ctrl-C is an ordinary one at the prompt
    let out = burst(b"1+1\n\x032+2\n");
    assert!(out.ends_with("2\nklong> ^C\nklong> 2+2\n4\nklong> "), "{out}");
}

#[test]
fn a_line_or_an_entry_that_is_too_long_is_an_error_message() {
    let _one = lock();
    let o = SimOptions { console: ConsoleConfig { max_line: 64, max_entry: 200, dumb: true, ..ConsoleConfig::default() }, ..SimOptions::default() };
    let mut s = sim_with(o);
    let long = vec![b'1'; 5000];
    assert_eq!(s.io(&long), "");
    assert!(s.console.held_bytes() < 1024, "{}", s.console.held_bytes());
    assert_eq!(s.io(b"\n"), "range error: line longer than 64 bytes\nklong> ");
    assert_eq!(s.io(b"1+1\n"), "2\nklong> ");
    // exactly the limit is fine
    let mut exact = vec![b' '; 63];
    exact.push(b'7');
    exact.push(b'\n');
    assert_eq!(s.io(&exact), "7\nklong> ");
    // an entry of many lines
    let mut out = String::new();
    for _ in 0..60 {
        out = s.io(b"(1+\n");
        if out.contains("error") {
            break;
        }
    }
    assert_eq!(out, "range error: entry longer than 200 bytes\nklong> ");
    assert_eq!(s.io(b"2+2\n"), "4\nklong> ");
    // bytes that are not UTF-8
    assert_eq!(s.io(b"\xff\xfe\n"), "syntax error: the input is not valid UTF-8\nklong> ");
}

#[test]
fn crlf_boards_end_every_line_with_cr_lf() {
    let _one = lock();
    let mut s = Sim::new(SimOptions { crlf: true, ..SimOptions::default() });
    assert_eq!(String::from_utf8_lossy(&s.run()), BANNER.replace('\n', "\r\n"));
    assert_eq!(s.io(b".p(\"a\");1\r"), ".p(\"a\");1\r\na\r\n1\r\nklong> ");
}

#[test]
fn boot_kg_runs_before_the_prompt_and_x_asks_for_a_reset() {
    let _one = lock();
    let mut fs = MemFs::new();
    fs.put("boot.kg", b".p(\"booting\");k::42\n").unwrap();
    let wire = Wire::new();
    let board = SimBoard::new(Box::new(wire.clone()), Some(fs));
    let reset = board.reset.clone();
    let mut console = Console::new(Runtime::new(board, Config::default(), None), ConsoleConfig { dumb: true, ..ConsoleConfig::default() });
    console.start();
    assert_eq!(String::from_utf8_lossy(&wire.take()), format!("booting\n{BANNER}"));
    wire.send(b"k\n.x(0)\n");
    while console.poll() {}
    let out = String::from_utf8_lossy(&wire.take()).into_owned();
    assert!(out.starts_with("42\nklong> "), "{out}");
    assert!(reset.get());
}

// ----- command mode ------------------------------------------------------------------------

#[test]
fn ver_reports_the_build() {
    let _one = lock();
    let mut s = dumb();
    let out = s.io(b"\x01ver\n");
    // (the features are whatever klong-rt was built with: files and lib, and those of ticket 106)
    assert!(out.starts_with("\x01ok ver=0.0.1 int=64 float=64 features=files,lib"), "{out:?}");
    assert!(out.contains(" platform=sim fs=1 heap_free=unknown max_line=1024 max_run=65536 max_file=16777216\nklong> "), "{out:?}");
    assert!(out.ends_with(P));
    // blanks after the control byte and a carriage return are fine
    assert!(s.io(b"\x01 ver \r\n").starts_with("\x01ok ver="));
}

#[test]
fn commands_that_are_wrong_are_refused_in_one_line() {
    let _one = lock();
    let mut s = dumb();
    for (cmd, want) in [
        (&b"\x01\n"[..], "\x01err syntax: empty command: the commands are put, hex, run, ver\nklong> "),
        (b"\x01frob 1 2\n", "\x01err syntax: unknown command frob: the commands are put, hex, run, ver\nklong> "),
        (b"\x01put a\n", "\x01err syntax: usage: put <path> <length> <crc32-hex>\nklong> "),
        (b"\x01put a 12 xyz\n", "\x01err syntax: usage: put <path> <length> <crc32-hex>\nklong> "),
        (b"\x01put a -1 0\n", "\x01err syntax: usage: put <path> <length> <crc32-hex>\nklong> "),
        (b"\x01put a 1 123456789\n", "\x01err syntax: usage: put <path> <length> <crc32-hex>\nklong> "),
        (b"\x01put a 1 0 extra\n", "\x01err syntax: usage: put <path> <length> <crc32-hex>\nklong> "),
        (b"\x01run 1\n", "\x01err syntax: usage: run <length> <crc32-hex>\nklong> "),
        (b"\x01run x 0\n", "\x01err syntax: usage: run <length> <crc32-hex>\nklong> "),
        (b"\x01hex\n", "\x01err syntax: usage: hex <path>\nklong> "),
        (b"\x01hex a b\n", "\x01err syntax: usage: hex <path>\nklong> "),
        (b"\x01ver 1\n", "\x01err syntax: usage: ver\nklong> "),
        (b"\x01\xff\n", "\x01err syntax: the command is not valid UTF-8\nklong> "),
    ] {
        assert_eq!(s.io(cmd), want, "{:?}", String::from_utf8_lossy(cmd));
        assert!(!s.console.busy());
    }
    // a command line without an end is bounded
    let mut long = vec![0x01u8];
    long.extend(vec![b'p'; 100_000]);
    assert_eq!(s.io(&long), "");
    assert!(s.console.held_bytes() < 2048);
    assert_eq!(s.io(b"\n"), "\x01err syntax: command longer than 255 bytes\nklong> ");
    // Ctrl-C leaves command mode; backspace edits the command
    let mut t = terminal();
    assert_eq!(t.io(b"\x01pu\x03"), "pu^C\nklong> ");
    assert!(t.io(b"\x01vex\x7f\x7fer\n").starts_with("vex\x08 \x08\x08 \x08er\n\x01ok ver="));
    assert!(s.files().is_empty());
}

#[test]
fn put_stores_a_file() {
    let _one = lock();
    let mut s = dumb();
    let text = b"sq::{x*x}\n:\"a comment with \x03 and \x01 and \r\n in it\"\n";
    assert_eq!(s.io(&put("lib/sq.kg", text)), "\x01go\n");
    assert!(s.console.busy());
    assert_eq!(s.io(&text[..10]), "");
    assert_eq!(s.io(&text[10..]), format!("\x01ok {} {:08x}\nklong> ", text.len(), crc32(text)));
    assert_eq!(s.file("lib/sq.kg").as_deref(), Some(&text[..]));
    assert!(s.file("lib/sq.kg.tmp").is_none());
    assert_eq!(s.io(b".l(\"lib/sq.kg\");sq(7)\n"), "49\nklong> ");
    // replacing it, with a file of no bytes
    assert_eq!(s.io(&put("lib/sq.kg", b"")), "\x01go\n\x01ok 0 00000000\nklong> ");
    assert_eq!(s.file("lib/sq.kg").as_deref(), Some(&b""[..]));
    // a binary file: every byte value, nothing is a key
    let all: Vec<u8> = (0..=255u8).cycle().take(1000).collect();
    let mut both = put("all.bin", &all);
    both.extend(&all);
    assert_eq!(s.io(&both), format!("\x01go\n\x01ok 1000 {:08x}\nklong> ", crc32(&all)));
    assert_eq!(s.file("all.bin"), Some(all));
}

#[test]
fn put_reports_a_syntax_error_and_keeps_the_file() {
    let _one = lock();
    let mut s = dumb();
    let mut both = put("bad.kg", b"f::{x+\n");
    both.extend(b"f::{x+\n");
    let out = s.io(&both);
    assert!(out.starts_with("\x01go\n\x01ok 7 "), "{out:?}");
    assert!(out.contains(" warn syntax: "), "{out:?}");
    assert!(out.ends_with("\nklong> "));
    assert_eq!(out.matches('\n').count(), 2, "{out:?}");
    assert_eq!(s.file("bad.kg").as_deref(), Some(&b"f::{x+\n"[..]));
    // parsing the file did not define or open anything in the running interpreter
    let module = b".module(:m)\ng::1\n";
    let mut both = put("m.kg", module);
    both.extend(module);
    assert_eq!(s.io(&both), format!("\x01go\n\x01ok {} {:08x}\nklong> ", module.len(), crc32(module)));
    assert_eq!(s.io(b"h::2;h\n"), "2\nklong> ");
    let mut fresh = dumb();
    assert_eq!(s.io(b"g\n"), fresh.io(b"g\n"));
    assert_eq!(s.io(b".module(0)\n"), fresh.io(b".module(0)\n"));
    // a .kg file larger than a program may be is stored and not parsed
    let o = SimOptions { console: ConsoleConfig { max_run: 16, dumb: true, ..ConsoleConfig::default() }, ..SimOptions::default() };
    let mut s = sim_with(o);
    let text = b"1+1;2+2;3+3;4+4;5+5\n";
    let mut both = put("big.kg", text);
    both.extend(text);
    assert_eq!(s.io(&both), format!("\x01go\n\x01ok {} {:08x} unchecked\nklong> ", text.len(), crc32(text)));
}

#[test]
fn put_failures_leave_nothing_behind() {
    let _one = lock();
    let data = pattern(300);
    // a wrong check sum
    let mut s = dumb();
    let mut both = format!("\x01put f.bin 300 {:08x}\n", crc32(&data) ^ 1).into_bytes();
    both.extend(&data);
    assert_eq!(s.io(&both), format!("\x01go\n\x01err system: check sum {:08x}, expected {:08x}\nklong> ", crc32(&data), crc32(&data) ^ 1));
    assert!(s.files().is_empty());
    // a file that is already there survives a failed replacement
    let mut both = put("f.bin", b"old");
    both.extend(b"old");
    s.io(&both);
    let mut both = format!("\x01put f.bin 300 {:08x}\n", 0).into_bytes();
    both.extend(&data);
    assert!(s.io(&both).contains("\x01err system: check sum"));
    assert_eq!(s.file("f.bin").as_deref(), Some(&b"old"[..]));
    assert_eq!(s.files(), ["f.bin"]);

    // more than a file may have: refused before any data
    let o = SimOptions { console: ConsoleConfig { max_file: 299, dumb: true, ..ConsoleConfig::default() }, ..SimOptions::default() };
    let mut s = sim_with(o);
    assert_eq!(s.io(&put("f.bin", &data)), "\x01err range: 300 bytes is more than a file may have here (299)\nklong> ");
    assert!(!s.console.busy());

    // no room in the file system: every byte is still taken, then one error
    let o = SimOptions { fs_bytes: Some(200), console: ConsoleConfig { dumb: true, ..ConsoleConfig::default() }, ..SimOptions::default() };
    let mut s = sim_with(o);
    assert_eq!(s.io(&put("f.bin", &data)), "\x01go\n");
    assert_eq!(s.io(&data[..299]), "");
    assert_eq!(s.io(&data[299..]), "\x01err system: no space left\nklong> ");
    assert!(s.files().is_empty());
    assert_eq!(s.io(b"1+1\n"), "2\nklong> ");

    // no file system at all: put and hex fail, run works
    let o = SimOptions { fs_bytes: None, console: ConsoleConfig { dumb: true, ..ConsoleConfig::default() }, ..SimOptions::default() };
    let mut s = sim_with(o);
    assert_eq!(s.io(&put("f.bin", &data)), "\x01err system: this board has no file system\nklong> ");
    assert_eq!(s.io(b"\x01hex f.bin\n"), "\x01err system: this board has no file system\nklong> ");
    let mut both = run_cmd(b"6*7");
    both.extend(b"6*7");
    assert_eq!(s.io(&both), "\x01go\n42\n\x01ok value 2\nklong> ");
    assert!(s.io(b"\x01ver\n").contains(" fs=0 "));
}

#[test]
fn a_transfer_without_input_times_out() {
    let _one = lock();
    let data = pattern(100);
    let mut s = dumb();
    assert_eq!(s.io(&put("f.bin", &data)), "\x01go\n");
    assert_eq!(s.io(&data[..40]), "");
    s.wire.advance(4_900_000);
    assert_eq!(s.io(&data[40..41]), "");
    // input restarts the clock
    s.wire.advance(4_900_000);
    assert_eq!(s.io(b""), "");
    assert!(s.console.busy());
    s.wire.advance(200_000);
    assert_eq!(s.io(b""), "\x01err system: transfer timed out\nklong> ");
    assert!(!s.console.busy());
    assert!(s.files().is_empty());
    assert_eq!(s.io(b"1+1\n"), "2\nklong> ");
    // run and hex alike
    assert_eq!(s.io(b"\x01run 10 0\n1+"), "\x01go\n");
    s.wire.advance(5_000_001);
    assert_eq!(s.io(b""), "\x01err system: transfer timed out\nklong> ");
    assert_eq!(s.io(b"\x01hex f.bin\n:0100"), "\x01go\n");
    s.wire.advance(5_000_001);
    assert_eq!(s.io(b""), "\x01err system: transfer timed out\nklong> ");
    assert!(s.files().is_empty());
    assert_eq!(s.io(b"1+1\n"), "2\nklong> ");
    // the prompt itself never times out
    s.wire.advance(3_600_000_000);
    assert_eq!(s.io(b""), "");
}

#[test]
fn run_evaluates_without_storing() {
    let _one = lock();
    let mut s = dumb();
    let go = |s: &mut Sim, program: &[u8]| {
        let mut both = run_cmd(program);
        both.extend(program);
        s.io(&both)
    };
    assert_eq!(go(&mut s, b"+/!100"), "\x01go\n4950\n\x01ok value 4\nklong> ");
    // what the program writes comes first, then the value on its own line
    assert_eq!(go(&mut s, b".d(\"ab\");.p(\"c\");7"), "\x01go\nabc\n7\n\x01ok value 1\nklong> ");
    assert_eq!(go(&mut s, b".d(\"ab\");\"x\ny\""), "\x01go\nab\"x\ny\"\n\x01ok value 5\nklong> ");
    // no statement, no value
    assert_eq!(go(&mut s, b""), "\x01go\n\x01ok\nklong> ");
    assert_eq!(go(&mut s, b":\"only a comment\""), "\x01go\n\x01ok\nklong> ");
    // several lines are one program; definitions stay
    assert_eq!(go(&mut s, b"sq::{x*x}\nt::{\n sq(x)+1}\nt(3)"), "\x01go\n10\n\x01ok value 2\nklong> ");
    assert_eq!(s.io(b"t(4)\n"), "17\nklong> ");
    // errors: the kind first, the place when there is one; output before the error is kept
    assert_eq!(go(&mut s, b".p(1);[1 2 3]@5").lines().nth(1), Some("1"));
    let out = go(&mut s, b"[1 2 3]@5");
    assert!(out.starts_with("\x01go\n\x01err range: "), "{out:?}");
    assert_eq!(out.matches('\n').count(), 2);
    let out = go(&mut s, b"1+1\n(2]");
    assert!(out.starts_with("\x01go\n\x01err syntax: ") && out.contains(" at 2:"), "{out:?}");
    assert!(go(&mut s, b"nosuch(1)").starts_with("\x01go\n\x01err "));
    // bytes that are not text, a wrong check sum, too many bytes
    assert_eq!(go(&mut s, b"\xff\xfe"), "\x01go\n\x01err syntax: the program is not valid UTF-8\nklong> ");
    let out = s.io(b"\x01run 3 00000000\n1+1");
    assert!(out.starts_with("\x01go\n\x01err system: check sum "), "{out:?}");
    assert_eq!(s.io(b"\x01run 65537 0\n"), "\x01err range: 65537 bytes is more than a program sent with run may have (65536)\nklong> ");
    assert!(s.files().is_empty());
    // the largest program allowed
    let mut big = vec![b' '; 65535];
    big.push(b'5');
    assert_eq!(go(&mut s, &big), "\x01go\n5\n\x01ok value 1\nklong> ");
    assert!(s.console.held_bytes() < 4096, "{}", s.console.held_bytes());
}

// ----- Intel HEX ---------------------------------------------------------------------------

fn hex_session(s: &mut Sim, path: &str, lines: &[&str]) -> Vec<String> {
    let mut all = format!("\x01hex {}\n", path).into_bytes();
    for l in lines {
        all.extend(l.as_bytes());
        all.push(b'\n');
    }
    s.io(&all).split('\n').map(String::from).collect()
}

#[test]
fn hex_stores_a_file() {
    let _one = lock();
    let mut s = dumb();
    let data = pattern(100);
    let recs = records(&data);
    let mut lines: Vec<&str> = recs.iter().map(|r| r.as_str()).collect();
    lines.push(EOF);
    let out = hex_session(&mut s, "p.bin", &lines);
    let mut want = vec![String::from("\x01go")];
    want.extend((1..=7).map(|n| format!("\x01ack {n}")));
    want.push(format!("\x01ok 100 {:08x}", crc32(&data)));
    want.push(P.into());
    assert_eq!(out, want);
    assert_eq!(s.file("p.bin"), Some(data));
    assert_eq!(s.files(), ["p.bin"]);
    // a file of no bytes; blank lines and CR LF are fine
    assert_eq!(s.io(b"\x01hex e.bin\n\r\n\n:00000001FF\r\n"), "\x01go\n\x01ok 0 00000000\nklong> ");
    assert_eq!(s.file("e.bin").as_deref(), Some(&b""[..]));
    // Klong source as HEX is parsed like a put
    let text = b"f::{x+\n";
    let recs = records(text);
    let out = hex_session(&mut s, "bad.kg", &[&recs[0], EOF]);
    assert!(out[2].starts_with("\x01ok 7 ") && out[2].contains(" warn syntax: "), "{out:?}");
}

#[test]
fn hex_a_corrupted_record_is_nakked_and_resent_before_the_end() {
    let _one = lock();
    let mut s = dumb();
    let data = pattern(64);
    let recs = records(&data);
    // the third record with one digit changed
    let mut bad = recs[2].clone().into_bytes();
    bad[12] = if bad[12] == b'0' { b'1' } else { b'0' };
    let bad = String::from_utf8(bad).unwrap();
    let out = hex_session(&mut s, "p.bin", &[&recs[0], &recs[1], &bad, &recs[3]]);
    assert_eq!(out, ["\x01go", "\x01ack 1", "\x01ack 2", "\x01nak 3 checksum", "\x01ack 4", ""]);
    assert!(s.console.busy());
    assert_eq!(s.io(format!("{}\n", recs[2]).as_bytes()), "\x01ack 5\n");
    assert_eq!(s.io(b":00000001FF\n"), format!("\x01ok 64 {:08x}\nklong> ", crc32(&data)));
    assert_eq!(s.file("p.bin"), Some(data.clone()));

    // a record that never arrives: the answer names what is stored, which is not the file
    let out = hex_session(&mut s, "q.bin", &[&recs[0], &recs[1], &recs[3], EOF]);
    let mut stored = data.clone();
    stored[32..48].fill(0);
    assert_eq!(out[4], format!("\x01ok 64 {:08x}", crc32(&stored)));
    assert_ne!(crc32(&stored), crc32(&data));
    // the last record missing: the length tells
    let out = hex_session(&mut s, "q.bin", &[&recs[0], &recs[1], &recs[2], EOF]);
    assert_eq!(out[4], format!("\x01ok 48 {:08x}", crc32(&data[..48])));
}

#[test]
fn hex_records_in_any_order_with_gaps() {
    let _one = lock();
    let mut s = dumb();
    // out of order, with a gap: bytes 16..32 are never written and read as zero
    let a = record(0, 32, &[0xAA; 16]);
    let b = record(0, 0, &[0xBB; 16]);
    let out = hex_session(&mut s, "gap.bin", &[&a, &b, EOF]);
    let mut want = vec![0xBB; 16];
    want.extend([0u8; 16]);
    want.extend([0xAA; 16]);
    assert_eq!(out[3], format!("\x01ok 48 {:08x}", crc32(&want)));
    assert_eq!(s.file("gap.bin"), Some(want));

    // extended linear address: beyond 64 KiB, and back
    let up = record(4, 0, &[0x00, 0x01]);
    let hi = record(0, 0x0010, &[1, 2, 3]);
    let down = record(4, 0, &[0x00, 0x00]);
    let lo = record(0, 0xFFFE, &[9, 9]);
    let out = hex_session(&mut s, "far.bin", &[&up, &hi, &down, &lo, EOF]);
    assert_eq!(&out[1..5], ["\x01ack 1", "\x01ack 2", "\x01ack 3", "\x01ack 4"]);
    let f = s.file("far.bin").unwrap();
    assert_eq!(f.len(), 0x10013);
    assert_eq!(&f[0xFFFE..0x10000], [9, 9]);
    assert_eq!(&f[0x10010..], [1, 2, 3]);
    assert!(f[..0xFFFE].iter().all(|&b| b == 0));

    // extended segment address: base = segment * 16
    let seg = record(2, 0, &[0x10, 0x00]);
    let d = record(0, 0x0004, &[7]);
    hex_session(&mut s, "seg.bin", &[&seg, &d, EOF]);
    assert_eq!(s.file("seg.bin").unwrap().len(), 0x10005);

    // start addresses are accepted and mean nothing; other types are refused
    let out = hex_session(&mut s, "t.bin", &[":0400000508000000EF", ":0400000300003800C1", ":00000006FA", &b, EOF]);
    assert_eq!(&out[1..5], ["\x01ack 1", "\x01ack 2", "\x01nak 3 unsupported record type", "\x01ack 4"]);
    assert_eq!(s.file("t.bin"), Some(vec![0xBB; 16]));
}

#[test]
fn hex_every_error_is_a_nak_and_the_transfer_goes_on() {
    let _one = lock();
    let o = SimOptions { fs_bytes: Some(3000), console: ConsoleConfig { max_file: 2048, dumb: true, ..ConsoleConfig::default() }, ..SimOptions::default() };
    let mut s = sim_with(o);
    let good = record(0, 0, b"good");
    let far = record(0, 2040, &[1; 9]);
    let edge = record(0, 2040, &[1; 8]);
    let long = format!(":{}", "0".repeat(700));
    let lines = [
        "garbage",
        ":xx",
        ":0",
        ":00",
        ":0400000001020304",
        ":0400000001020300F2",
        ":0100000400FB",
        ":000000AB55",
        long.as_str(),
        far.as_str(),
        edge.as_str(),
        good.as_str(),
    ];
    let out = hex_session(&mut s, "e.bin", &lines);
    assert_eq!(
        out,
        [
            "\x01go",
            "\x01nak 1 no colon",
            "\x01nak 2 not hex",
            "\x01nak 3 not hex",
            "\x01nak 4 short record",
            "\x01nak 5 length mismatch",
            "\x01nak 6 checksum",
            "\x01nak 7 bad field size",
            "\x01nak 8 unsupported record type",
            "\x01nak 9 line too long",
            "\x01nak 10 file too large",
            "\x01ack 11",
            "\x01ack 12",
            ""
        ]
    );
    assert!(s.console.busy());
    // the file system fills up: a nak with its reason
    let more = record(0, 2048 - 16, &[1; 16]);
    let out = s.io(format!("{more}\n").as_bytes());
    assert_eq!(out, "\x01ack 13\n");
    // (a second file would not fit: this one is alone and within the limit)
    assert!(s.io(b":00000001FF\n").starts_with("\x01ok 2048 "));
    let out = hex_session(&mut s, "f.bin", &[&edge, &more]);
    assert_eq!(&out[1..3], ["\x01nak 1 no space left", "\x01nak 2 no space left"]);
    // Ctrl-C abandons the transfer and removes the temporary file
    assert_eq!(s.io(b":0000\x03"), "\x01err interrupt: transfer abandoned\nklong> ");
    assert_eq!(s.files(), ["e.bin"]);
    assert_eq!(s.io(b"1+1\n"), "2\nklong> ");
}

#[test]
fn hex_a_quarter_megabyte_reads_back_identical() {
    let _one = lock();
    let mut s = dumb();
    let data = pattern(256 * 1024 + 5);
    let mut all = b"\x01hex big.bin\n".to_vec();
    for r in records(&data) {
        all.extend(r.as_bytes());
        all.push(b'\n');
    }
    all.extend(b":00000001FF\n");
    let out = s.io(&all);
    assert!(!out.contains("nak"));
    assert!(out.ends_with(&format!("\x01ok {} {:08x}\nklong> ", data.len(), crc32(&data))), "{}", &out[out.len() - 60..]);
    assert_eq!(s.file("big.bin"), Some(data));
}

#[test]
fn nothing_grows_without_a_bound() {
    let _one = lock();
    let mut s = terminal();
    let flood = vec![b'x'; 300_000];
    // a line without an end
    s.io(&flood);
    assert!(s.console.held_bytes() <= 1024 + 2048 + 8192, "{}", s.console.held_bytes());
    s.io(b"\n");
    // an escape sequence without an end, a command without an end, a record without an end
    for start in [&b"\x1b["[..], b"\x1b[1;", b"\x01", b"\x01hex f\n", b"\x01hex f\n:"] {
        s.io(b"\x03\n");
        s.io(start);
        s.io(&vec![b'1'; 300_000]);
        assert!(s.console.held_bytes() <= 1024 + 2048 + 8192, "{:?}: {}", start, s.console.held_bytes());
        s.io(b"\x03\n");
    }
    // history is a ring
    for i in 0..500 {
        s.io(format!("{i}+{i}\n").as_bytes());
    }
    assert!(s.console.held_bytes() <= 1024 + 2048 + 8192, "{}", s.console.held_bytes());
    assert_eq!(s.io(b"\x1b[A\n"), "\x1b[K499+499\n998\nklong> ");
}
