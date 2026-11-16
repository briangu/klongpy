//! klong-sim: a board that is not there. [`SimBoard`] is a `Platform` like a board's: a
//! serial port for the console, a clock, a file system in RAM, and nothing else. The serial
//! port is a [`SerialPort`]: [`Wire`] (two queues in memory, with a virtual clock and input
//! that can be held back, for tests that need exact timing), or the byte streams of
//! `src/main.rs` (standard input and output, or a pseudo-terminal that a terminal program or
//! the upload tool opens like a serial device).
//!
//! [`Sim`] puts the real console (`klong-console`) on a `SimBoard` over a `Wire`: what the
//! tests of this crate drive byte by byte.
use std::cell::{Cell, RefCell};
use std::collections::VecDeque;
use std::rc::Rc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use klong_console::{Console, ConsoleConfig};
use klong_core::Config;
use klong_rt::memfs::MemFs;
use klong_rt::{FileSystem, MemStats, Platform, Runtime};

pub mod pty;

/// The serial port of a simulated board: what a UART driver gives a real one.
pub trait SerialPort {
    /// Bytes that have arrived, at most `buf.len()`; 0 when there are none. Never waits.
    fn read(&mut self, buf: &mut [u8]) -> usize;
    fn write(&mut self, bytes: &[u8]);
    /// `Platform::idle` was called (an evaluation is running, or the main loop turns).
    fn idle(&mut self) {}
    /// Wait until input may have arrived, at most this long (the main loop, never the console).
    fn wait(&mut self, _at_most: Duration) {}
    /// The other end is gone for good.
    fn closed(&self) -> bool {
        false
    }
    /// Microseconds of a clock this port brings along (`Wire`'s virtual one).
    fn now_us(&self) -> Option<u64> {
        None
    }
}

#[derive(Default)]
struct WireState {
    rx: VecDeque<u8>,
    /// Input that arrives later: after this many calls of `idle`.
    held: Vec<(u64, Vec<u8>)>,
    tx: Vec<u8>,
    /// The most output kept (0 = all of it): when more is written the oldest goes.
    tx_keep: usize,
    /// The most bytes one `read` gives (0 = as many as asked for).
    burst: usize,
    idles: u64,
    now_us: u64,
    /// Microseconds the clock moves at every `idle`.
    us_per_idle: u64,
}

/// A serial line in memory. The handle is shared: the board reads and writes one end, the
/// test the other.
#[derive(Clone, Default)]
pub struct Wire(Rc<RefCell<WireState>>);

impl Wire {
    pub fn new() -> Wire {
        Wire::default()
    }
    /// Bytes to the board.
    pub fn send(&self, bytes: &[u8]) {
        self.0.borrow_mut().rx.extend(bytes);
    }
    /// Bytes to the board that arrive only after `idles` more calls of `Platform::idle`:
    /// input that comes while an evaluation runs.
    pub fn send_after_idles(&self, idles: u64, bytes: &[u8]) {
        let mut w = self.0.borrow_mut();
        let at = w.idles + idles;
        w.held.push((at, bytes.to_vec()));
    }
    /// What the board has written since the last call.
    pub fn take(&self) -> Vec<u8> {
        std::mem::take(&mut self.0.borrow_mut().tx)
    }
    pub fn written(&self) -> usize {
        self.0.borrow().tx.len()
    }
    /// Bytes the board has not read yet.
    pub fn unread(&self) -> usize {
        let w = self.0.borrow();
        w.rx.len() + w.held.iter().map(|(_, b)| b.len()).sum::<usize>()
    }
    /// Keep at most this much unread output, the newest (0 = all). The line's own memory is
    /// the test's, not the board's: a test that limits the heap limits this too.
    pub fn set_keep(&self, n: usize) {
        self.0.borrow_mut().tx_keep = n;
    }
    /// One `read` gives at most this many bytes (0 = no limit).
    pub fn set_burst(&self, n: usize) {
        self.0.borrow_mut().burst = n;
    }
    /// Move the virtual clock.
    pub fn advance(&self, us: u64) {
        self.0.borrow_mut().now_us += us;
    }
    /// The virtual clock moves by this much at every `idle`.
    pub fn set_us_per_idle(&self, us: u64) {
        self.0.borrow_mut().us_per_idle = us;
    }
    pub fn idles(&self) -> u64 {
        self.0.borrow().idles
    }
}

impl SerialPort for Wire {
    fn read(&mut self, buf: &mut [u8]) -> usize {
        let mut w = self.0.borrow_mut();
        let mut n = buf.len().min(w.rx.len());
        if w.burst != 0 {
            n = n.min(w.burst);
        }
        for slot in buf[..n].iter_mut() {
            *slot = w.rx.pop_front().unwrap_or(0);
        }
        n
    }
    fn write(&mut self, bytes: &[u8]) {
        let mut w = self.0.borrow_mut();
        let keep = w.tx_keep;
        let bytes = if keep != 0 && bytes.len() > keep { &bytes[bytes.len() - keep..] } else { bytes };
        if keep != 0 && w.tx.len() + bytes.len() > keep {
            let drop = w.tx.len() + bytes.len() - keep;
            w.tx.drain(..drop);
        }
        if w.tx.try_reserve(bytes.len()).is_ok() {
            w.tx.extend_from_slice(bytes);
        }
    }
    fn idle(&mut self) {
        let mut w = self.0.borrow_mut();
        w.idles += 1;
        w.now_us += w.us_per_idle;
        let now = w.idles;
        let mut i = 0;
        while i < w.held.len() {
            if w.held[i].0 <= now {
                let (_, bytes) = w.held.remove(i);
                w.rx.extend(bytes);
            } else {
                i += 1;
            }
        }
    }
    fn now_us(&self) -> Option<u64> {
        Some(self.0.borrow().now_us)
    }
}

/// A port that several boards use one after the other (a board that resets keeps its wires).
pub struct SharedPort<T: SerialPort>(pub Rc<RefCell<T>>);

impl<T: SerialPort> SerialPort for SharedPort<T> {
    fn read(&mut self, buf: &mut [u8]) -> usize {
        self.0.borrow_mut().read(buf)
    }
    fn write(&mut self, bytes: &[u8]) {
        self.0.borrow_mut().write(bytes)
    }
    fn idle(&mut self) {
        self.0.borrow_mut().idle()
    }
    fn wait(&mut self, at_most: Duration) {
        self.0.borrow_mut().wait(at_most)
    }
    fn closed(&self) -> bool {
        self.0.borrow().closed()
    }
    fn now_us(&self) -> Option<u64> {
        self.0.borrow().now_us()
    }
}

/// The simulated board.
pub struct SimBoard {
    pub port: Box<dyn SerialPort>,
    start: Instant,
    /// `None`: a board without storage.
    pub fs: Option<MemFs>,
    /// Write CR LF for LF, as a board does whose console is used from a terminal program.
    pub crlf: bool,
    /// Set by `.x`: a board resets; the main loop builds a new runtime.
    pub reset: Rc<Cell<bool>>,
}

impl SimBoard {
    pub fn new(port: Box<dyn SerialPort>, fs: Option<MemFs>) -> SimBoard {
        SimBoard { port, start: Instant::now(), fs, crlf: false, reset: Rc::new(Cell::new(false)) }
    }
}

impl Platform for SimBoard {
    fn now_us(&self) -> u64 {
        self.port.now_us().unwrap_or_else(|| self.start.elapsed().as_micros() as u64)
    }
    fn wall_clock(&self) -> Option<f64> {
        SystemTime::now().duration_since(UNIX_EPOCH).ok().map(|d| d.as_secs_f64())
    }
    fn idle(&mut self) {
        self.port.idle()
    }
    fn console_read(&mut self, buf: &mut [u8]) -> usize {
        self.port.read(buf)
    }
    fn console_write(&mut self, bytes: &[u8]) {
        if !self.crlf {
            return self.port.write(bytes);
        }
        for part in bytes.split_inclusive(|&b| b == b'\n') {
            match part.split_last() {
                Some((b'\n', line)) => {
                    self.port.write(line);
                    self.port.write(b"\r\n");
                }
                _ => self.port.write(part),
            }
        }
    }
    fn fs(&mut self) -> Option<&mut dyn FileSystem> {
        self.fs.as_mut().map(|f| f as &mut dyn FileSystem)
    }
    fn name(&self) -> &str {
        "sim"
    }
    fn exit(&mut self, _code: i32) {
        self.reset.set(true);
    }
}

/// What a simulated board is built from.
pub struct SimOptions {
    pub console: ConsoleConfig,
    pub config: Config,
    /// Bytes of the RAM file system; `None` for a board without one.
    pub fs_bytes: Option<usize>,
    pub crlf: bool,
    pub mem: Option<MemStats>,
}

impl Default for SimOptions {
    fn default() -> SimOptions {
        SimOptions { console: ConsoleConfig::default(), config: Config::default(), fs_bytes: Some(4 * 1024 * 1024), crlf: false, mem: None }
    }
}

/// A console on a board on a port. The reset flag is returned with it.
pub fn build(port: Box<dyn SerialPort>, o: SimOptions) -> (Console<SimBoard>, Rc<Cell<bool>>) {
    let fs = o.fs_bytes.map(|n| {
        let mut fs = MemFs::new();
        fs.capacity = n;
        fs
    });
    let mut board = SimBoard::new(port, fs);
    board.crlf = o.crlf;
    let reset = board.reset.clone();
    let rt = Runtime::new(board, o.config, o.mem);
    let mut console = o.console;
    console.mem = o.mem;
    (Console::new(rt, console), reset)
}

/// The real console on a simulated board whose serial port is a [`Wire`]: for tests.
pub struct Sim {
    pub console: Console<SimBoard>,
    pub wire: Wire,
    pub reset: Rc<Cell<bool>>,
}

impl Sim {
    pub fn new(o: SimOptions) -> Sim {
        let wire = Wire::new();
        let (mut console, reset) = build(Box::new(wire.clone()), o);
        console.start();
        Sim { console, wire, reset }
    }
    /// Let the board run until it has nothing left to do; what it wrote meanwhile.
    pub fn run(&mut self) -> Vec<u8> {
        let mut turns = 0u64;
        while self.console.poll() {
            turns += 1;
            assert!(turns < 10_000_000, "the console does not come to rest");
        }
        self.wire.take()
    }
    /// Send bytes, let the board run, return what it wrote as text.
    pub fn io(&mut self, bytes: &[u8]) -> String {
        self.wire.send(bytes);
        String::from_utf8_lossy(&self.run()).into_owned()
    }
    /// The file system of the board.
    pub fn file(&self, path: &str) -> Option<Vec<u8>> {
        let p = self.console.rt.shared.platform.borrow();
        p.fs.as_ref().and_then(|fs| fs.get(path).map(|d| d.to_vec()))
    }
    pub fn files(&self) -> Vec<String> {
        let mut names = Vec::new();
        let mut p = self.console.rt.shared.platform.borrow_mut();
        if let Some(fs) = p.fs.as_mut() {
            let _ = fs.list("", &mut |name, _| names.push(name.to_string()));
        }
        names
    }
}
