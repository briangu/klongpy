//! The simulated board as a program.
//!
//!     klong-sim                 the console on standard input and output (a pipe)
//!     klong-sim --pty           the console on a pseudo-terminal; the first line of standard
//!                               output is `pty: <device>`, to be opened like a serial port
//!     klong-sim --check-complete
//!                               one entry per line of standard input, as hex digits; answers
//!                               complete, incomplete or error per line (klong-console's
//!                               port of KlongPy's check_complete, for the shared test cases)
//!
//! Options: --dumb (no echo, no editing), --crlf (lines end CR LF, as on a real terminal
//! line), --no-fs, --fs-bytes N, --heap-limit N, --stack-budget N (Config::for_stack, as a
//! board with that task stack), --strict N, --max-line N, --max-entry N, --max-run N,
//! --max-file N, --timeout-ms N, --stack N (native stack of the evaluating thread). Sizes
//! take K and M. `.x()` resets the board: a new runtime, an empty file system, the banner
//! again. End of input (a pipe) ends the program.
use std::alloc::System;
use std::cell::RefCell;
use std::collections::VecDeque;
use std::io::{BufRead, Read, Write};
use std::process::ExitCode;
use std::rc::Rc;
use std::sync::mpsc::{Receiver, RecvTimeoutError, TryRecvError};
use std::time::Duration;

use klong_console::ConsoleConfig;
use klong_core::Config;
use klong_rt::heap::Counting;
use klong_sim::{build, SerialPort, SharedPort, SimOptions};

#[global_allocator]
static HEAP: Counting<System> = Counting::new(System);

const RESERVE: usize = 16 * 1024;
const STACK: usize = 64 * 1024 * 1024;

fn mem_stats() -> (usize, usize, usize) {
    (HEAP.used(), HEAP.peak(), HEAP.limit())
}

fn mem_pressed() -> bool {
    HEAP.pressed()
}

/// A serial port over two byte streams: a thread reads the input, as an interrupt handler
/// fills a UART's receive buffer.
struct Stream {
    rx: Receiver<Vec<u8>>,
    buf: VecDeque<u8>,
    tx: Box<dyn Write>,
    closed: bool,
}

impl Stream {
    fn new(mut input: Box<dyn Read + Send>, tx: Box<dyn Write>, ends: bool) -> Stream {
        let (send, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let mut buf = [0u8; 4096];
            loop {
                match input.read(&mut buf) {
                    // a pipe ends; a pseudo-terminal nobody has open reads 0 or fails for a while
                    Ok(0) | Err(_) if ends => return,
                    Ok(0) | Err(_) => std::thread::sleep(Duration::from_millis(5)),
                    Ok(n) => {
                        if send.send(buf[..n].to_vec()).is_err() {
                            return;
                        }
                    }
                }
            }
        });
        Stream { rx, buf: VecDeque::new(), tx, closed: false }
    }
    fn pull(&mut self) {
        loop {
            match self.rx.try_recv() {
                Ok(bytes) => self.buf.extend(bytes),
                Err(TryRecvError::Empty) => return,
                Err(TryRecvError::Disconnected) => {
                    self.closed = true;
                    return;
                }
            }
        }
    }
}

impl SerialPort for Stream {
    fn read(&mut self, buf: &mut [u8]) -> usize {
        if self.buf.is_empty() {
            self.pull();
        }
        let n = buf.len().min(self.buf.len());
        for slot in buf[..n].iter_mut() {
            *slot = self.buf.pop_front().unwrap_or(0);
        }
        n
    }
    fn write(&mut self, bytes: &[u8]) {
        let _ = self.tx.write_all(bytes);
        let _ = self.tx.flush();
    }
    fn wait(&mut self, at_most: Duration) {
        if !self.buf.is_empty() || self.closed {
            return;
        }
        match self.rx.recv_timeout(at_most) {
            Ok(bytes) => self.buf.extend(bytes),
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => self.closed = true,
        }
    }
    fn closed(&self) -> bool {
        self.closed && self.buf.is_empty()
    }
}

#[derive(Default)]
struct Opts {
    pty: bool,
    check: bool,
    dumb: bool,
    crlf: bool,
    no_fs: bool,
    fs_bytes: Option<usize>,
    heap_limit: usize,
    stack_budget: Option<usize>,
    stack: Option<usize>,
    strict: u8,
    max_line: Option<usize>,
    max_entry: Option<usize>,
    max_run: Option<usize>,
    max_file: Option<usize>,
    timeout_ms: Option<u64>,
}

fn size_arg(s: &str) -> Option<usize> {
    let (digits, mult) = match s.as_bytes().last()? {
        b'k' | b'K' => (&s[..s.len() - 1], 1024),
        b'm' | b'M' => (&s[..s.len() - 1], 1024 * 1024),
        _ => (s, 1),
    };
    digits.parse::<usize>().ok()?.checked_mul(mult)
}

fn parse_args() -> Option<Opts> {
    let mut o = Opts::default();
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--pty" => o.pty = true,
            "--check-complete" => o.check = true,
            "--dumb" => o.dumb = true,
            "--crlf" => o.crlf = true,
            "--no-fs" => o.no_fs = true,
            "--fs-bytes" => o.fs_bytes = Some(size_arg(&it.next()?)?),
            "--heap-limit" => o.heap_limit = size_arg(&it.next()?)?,
            "--stack-budget" => o.stack_budget = Some(size_arg(&it.next()?)?),
            "--stack" => o.stack = Some(size_arg(&it.next()?)?),
            "--strict" => o.strict = it.next()?.parse().ok().filter(|n| *n <= 2)?,
            "--max-line" => o.max_line = Some(size_arg(&it.next()?)?),
            "--max-entry" => o.max_entry = Some(size_arg(&it.next()?)?),
            "--max-run" => o.max_run = Some(size_arg(&it.next()?)?),
            "--max-file" => o.max_file = Some(size_arg(&it.next()?)?),
            "--timeout-ms" => o.timeout_ms = Some(it.next()?.parse().ok()?),
            _ => return None,
        }
    }
    Some(o)
}

fn options(o: &Opts) -> SimOptions {
    let base = match o.stack_budget {
        Some(b) => Config::for_stack(b),
        None => {
            let stack = o.stack.unwrap_or(STACK);
            Config { stack_limit: Some(stack - stack / 4), ..Config::default() }
        }
    };
    let config = Config { strict: o.strict, mem_probe: Some(mem_pressed), ..base };
    let d = ConsoleConfig::default();
    let console = ConsoleConfig {
        dumb: o.dumb,
        max_line: o.max_line.unwrap_or(d.max_line),
        max_entry: o.max_entry.unwrap_or(d.max_entry),
        max_run: o.max_run.unwrap_or(d.max_run),
        max_file: o.max_file.map(|n| n as u64).unwrap_or(d.max_file),
        timeout_us: o.timeout_ms.map(|ms| ms * 1000).unwrap_or(d.timeout_us),
        ..d
    };
    let fs_bytes = if o.no_fs { None } else { Some(o.fs_bytes.unwrap_or(8 * 1024 * 1024)) };
    SimOptions { console, config, fs_bytes, crlf: o.crlf, mem: Some(mem_stats) }
}

fn hex_text(line: &str) -> Option<String> {
    let b = line.trim().as_bytes();
    if b.len() % 2 != 0 {
        return None;
    }
    let mut out = Vec::with_capacity(b.len() / 2);
    for pair in b.chunks(2) {
        out.push(u8::from_str_radix(std::str::from_utf8(pair).ok()?, 16).ok()?);
    }
    String::from_utf8(out).ok()
}

fn check_complete() -> ExitCode {
    let stdin = std::io::stdin();
    let mut out = std::io::stdout().lock();
    for line in stdin.lock().lines() {
        let verdict = match line.ok().as_deref().and_then(hex_text) {
            Some(text) => klong_console::complete::check(&text).name(),
            None => "bad input",
        };
        let _ = writeln!(out, "{}", verdict);
    }
    ExitCode::SUCCESS
}

fn serve(o: Opts) -> ExitCode {
    let port: Rc<RefCell<Stream>> = if o.pty {
        let pty = match klong_sim::pty::open() {
            Ok(p) => p,
            Err(e) => {
                eprintln!("klong-sim: no pseudo-terminal: {}", e);
                return ExitCode::from(1);
            }
        };
        println!("pty: {}", pty.path);
        let _ = std::io::stdout().flush();
        let reader = match pty.master.try_clone() {
            Ok(f) => f,
            Err(e) => {
                eprintln!("klong-sim: {}", e);
                return ExitCode::from(1);
            }
        };
        // the slave end stays open for as long as the program runs
        std::mem::forget(pty.slave);
        Rc::new(RefCell::new(Stream::new(Box::new(reader), Box::new(pty.master), false)))
    } else {
        Rc::new(RefCell::new(Stream::new(Box::new(std::io::stdin()), Box::new(std::io::stdout()), true)))
    };
    loop {
        // one life of the board, from power-up to reset
        HEAP.set_limit(0, 0);
        let (mut console, reset) = build(Box::new(SharedPort(port.clone())), options(&o));
        HEAP.reset_peak();
        HEAP.set_limit(if o.heap_limit == 0 { 0 } else { HEAP.used() + o.heap_limit }, RESERVE);
        console.start();
        loop {
            if !console.poll() {
                if port.borrow().closed() {
                    return ExitCode::SUCCESS;
                }
                port.borrow_mut().wait(Duration::from_millis(20));
            }
            if reset.get() {
                break;
            }
        }
    }
}

fn main() -> ExitCode {
    let o = match parse_args() {
        Some(o) => o,
        None => {
            eprintln!("usage: klong-sim [--pty] [--dumb] [--crlf] [--no-fs] [--fs-bytes N] [--heap-limit N] [--stack-budget N] [--strict N] [--max-line N] [--max-entry N] [--max-run N] [--max-file N] [--timeout-ms N] | --check-complete");
            return ExitCode::from(2);
        }
    };
    if o.check {
        return check_complete();
    }
    // The evaluator recurses on the native stack; a thread with a stack of known size makes
    // its bound hold (as in klong-cli).
    let stack = o.stack.unwrap_or(STACK);
    match std::thread::Builder::new().stack_size(stack).spawn(move || serve(o)) {
        Ok(h) => h.join().unwrap_or(ExitCode::from(70)),
        Err(_) => ExitCode::from(70),
    }
}
