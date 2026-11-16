//! klong-board: the skeleton of a board port (ticket 104). It names no chip. A port supplies
//! four small things and gets the whole runtime with its console:
//!
//! * a [`Uart`]: bytes in without waiting, bytes out;
//! * a [`Clock`]: monotonic microseconds ([`TickClock`] makes one from a timer interrupt);
//! * a [`System`]: the board's name, its watchdog, how it sleeps, how it resets;
//! * storage: any `klong_rt::FileSystem` (the RAM one, `klong_rt::memfs::MemFs`, or a flash
//!   file system the port wraps), or [`NoFs`];
//!
//! plus two numbers in [`BoardConfig`]: the stack of the task that runs the interpreter and
//! the heap figures of its allocator ([`heap::RegionHeap`] is one for a plain region of RAM).
//! [`BoardPlatform`] turns these into the `Platform` of `klong-rt`; [`run`] is the main loop
//! and never returns. `src/bin/image.rs` is all of it put together for a bare-metal target.
//! `native/PORTING.md` is the guide.
#![no_std]
#![deny(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used)]

extern crate alloc;

pub mod heap;

use core::sync::atomic::{AtomicU32, Ordering::Relaxed};

use klong_console::{Console, ConsoleConfig};
use klong_core::Config;
use klong_rt::{FileId, FileSystem, FsError, MemStats, Mode, Platform, Runtime};

/// The serial port of the console.
pub trait Uart {
    /// Bytes that have arrived, at most `buf.len()`; 0 when there are none. Must not wait.
    /// The driver's receive buffer must hold what arrives while the interpreter is busy for
    /// one tick (a few milliseconds) or while flash is written; what does not fit is lost.
    fn read(&mut self, buf: &mut [u8]) -> usize;
    /// Send all the bytes. May wait for room in the transmitter.
    fn write(&mut self, bytes: &[u8]);
}

/// Time.
pub trait Clock {
    /// Microseconds since some start; never goes back.
    fn now_us(&self) -> u64;
    /// Seconds since 1970, when the board knows.
    fn wall_clock(&self) -> Option<f64> {
        None
    }
}

/// The rest of the board.
pub trait System {
    /// A short name for `.ver()` and the banner.
    fn name(&self) -> &str;
    /// Called often (every 4096 steps of an evaluation, every turn of the main loop): feed
    /// the watchdog, let other tasks run. Must be cheap.
    fn idle(&mut self) {}
    /// Sleep until input may be ready or the clock reaches `deadline_us`; returning early is
    /// always right. The default returns at once.
    fn wait(&mut self, _deadline_us: Option<u64>) {}
    /// Bits to seed `.rn()` with, from a hardware generator if there is one.
    fn entropy(&mut self) -> Option<u64> {
        None
    }
    /// `.x(code)`: restart the board. If this returns, the running program is only stopped.
    fn reset(&mut self, _code: i32) {}
}

/// A clock from a periodic interrupt: the handler calls [`TickClock::tick`], the main loop
/// reads the time. The counter is 32 bits wide and atomic with load and store only (targets
/// without compare-and-swap have nothing else); it must live in internal RAM. Wrap-around
/// is extended to 64 bits as long as `now_us` is called at least once per wrap (49 days at
/// 1 kHz).
pub struct TickClock {
    ticks: AtomicU32,
    seen: AtomicU32,
    high: AtomicU32,
    us_per_tick: u32,
}

impl TickClock {
    pub const fn new(us_per_tick: u32) -> TickClock {
        TickClock { ticks: AtomicU32::new(0), seen: AtomicU32::new(0), high: AtomicU32::new(0), us_per_tick }
    }
    /// From the timer interrupt (the only writer of the counter).
    pub fn tick(&self) {
        self.ticks.store(self.ticks.load(Relaxed).wrapping_add(1), Relaxed);
    }
    /// From the main loop only.
    pub fn now_us(&self) -> u64 {
        let t = self.ticks.load(Relaxed);
        if t < self.seen.load(Relaxed) {
            self.high.store(self.high.load(Relaxed).wrapping_add(1), Relaxed);
        }
        self.seen.store(t, Relaxed);
        (((self.high.load(Relaxed) as u64) << 32) | t as u64).wrapping_mul(self.us_per_tick as u64)
    }
}

impl Clock for &'static TickClock {
    fn now_us(&self) -> u64 {
        TickClock::now_us(self)
    }
}

/// The storage of a board without storage. `put` and `hex` then fail with a `system` error
/// and `run` still works.
pub enum NoFs {}

impl FileSystem for NoFs {
    fn open(&mut self, _: &str, _: Mode) -> Result<FileId, FsError> {
        match *self {}
    }
    fn read(&mut self, _: FileId, _: &mut [u8]) -> Result<usize, FsError> {
        match *self {}
    }
    fn write(&mut self, _: FileId, _: &[u8]) -> Result<usize, FsError> {
        match *self {}
    }
    fn seek(&mut self, _: FileId, _: u64) -> Result<(), FsError> {
        match *self {}
    }
    fn close(&mut self, _: FileId) {
        match *self {}
    }
    fn remove(&mut self, _: &str) -> Result<(), FsError> {
        match *self {}
    }
    fn rename(&mut self, _: &str, _: &str) -> Result<(), FsError> {
        match *self {}
    }
    fn list(&mut self, _: &str, _: &mut dyn FnMut(&str, u64)) -> Result<(), FsError> {
        match *self {}
    }
}

/// The `Platform` of a board, from its parts.
pub struct BoardPlatform<U: Uart, C: Clock, S: System, F: FileSystem> {
    pub uart: U,
    pub clock: C,
    pub system: S,
    pub fs: Option<F>,
    /// Send CR LF for LF: what a terminal program on the other end of a UART expects. The
    /// upload tool reads either.
    pub crlf: bool,
}

impl<U: Uart, C: Clock, S: System, F: FileSystem> BoardPlatform<U, C, S, F> {
    pub fn new(uart: U, clock: C, system: S, fs: Option<F>) -> Self {
        BoardPlatform { uart, clock, system, fs, crlf: true }
    }
}

impl<U: Uart, C: Clock, S: System, F: FileSystem> Platform for BoardPlatform<U, C, S, F> {
    fn now_us(&self) -> u64 {
        self.clock.now_us()
    }
    fn wall_clock(&self) -> Option<f64> {
        self.clock.wall_clock()
    }
    fn idle(&mut self) {
        self.system.idle()
    }
    fn console_read(&mut self, buf: &mut [u8]) -> usize {
        self.uart.read(buf).min(buf.len())
    }
    fn console_write(&mut self, bytes: &[u8]) {
        if !self.crlf {
            return self.uart.write(bytes);
        }
        for part in bytes.split_inclusive(|&b| b == b'\n') {
            match part.split_last() {
                Some((b'\n', line)) => {
                    self.uart.write(line);
                    self.uart.write(b"\r\n");
                }
                _ => self.uart.write(part),
            }
        }
    }
    fn fs(&mut self) -> Option<&mut dyn FileSystem> {
        self.fs.as_mut().map(|f| f as &mut dyn FileSystem)
    }
    fn name(&self) -> &str {
        self.system.name()
    }
    fn entropy(&mut self) -> u64 {
        match self.system.entropy() {
            Some(bits) => bits,
            None => self.clock.now_us() ^ 0x9e37_79b9_7f4a_7c15,
        }
    }
    fn exit(&mut self, code: i32) {
        self.system.reset(code)
    }
    fn wait(&mut self, deadline_us: Option<u64>) {
        self.system.idle();
        self.system.wait(deadline_us)
    }
}

/// The numbers of a port.
#[derive(Clone)]
pub struct BoardConfig {
    /// Bytes of native stack of the task (or of the whole program) that runs the interpreter.
    /// `Config::for_stack` derives the recursion and nesting limits from it: about 0.8 KiB
    /// per level of Klong recursion on a 32-bit target, so 16 KiB gives about 14 levels and
    /// 64 KiB about 60.
    pub stack_bytes: usize,
    /// Heap figures (used, peak, limit) for `.mem()` and `ver`: `Counting::used` and friends.
    pub mem: Option<MemStats>,
    /// True when the allocator lives on its reserve (`Counting::pressed`): the evaluation
    /// then ends with a `memory` error instead of the board with a reset.
    pub mem_pressed: Option<fn() -> bool>,
    /// Strict level at start (0 to 2).
    pub strict: u8,
    pub console: ConsoleConfig,
}

impl Default for BoardConfig {
    fn default() -> BoardConfig {
        BoardConfig { stack_bytes: 32 * 1024, mem: None, mem_pressed: None, strict: 0, console: ConsoleConfig::default() }
    }
}

/// Build the runtime and its console, run `boot.kg`, print the banner and the first prompt.
pub fn start<P: Platform + 'static>(platform: P, cfg: BoardConfig) -> Console<P> {
    let config = Config { strict: cfg.strict, mem_probe: cfg.mem_pressed, ..Config::for_stack(cfg.stack_bytes) };
    let rt = Runtime::new(platform, config, cfg.mem);
    let mut console = Console::new(rt, ConsoleConfig { mem: cfg.mem, ..cfg.console });
    console.start();
    console
}

/// One turn of the main loop: console input, then timers and connections (ticket 106). False
/// when there was nothing to do.
pub fn step<P: Platform + 'static>(console: &mut Console<P>) -> bool {
    if let Ok(mut p) = console.rt.shared.platform.try_borrow_mut() {
        // the watchdog is fed every turn, busy or not
        p.idle();
    }
    let typed = console.poll();
    let events = console.rt.poll_events();
    typed || events
}

/// The main loop of a board. Call it from the task whose stack `cfg.stack_bytes` describes.
pub fn run<P: Platform + 'static>(platform: P, cfg: BoardConfig) -> ! {
    let mut console = start(platform, cfg);
    loop {
        if !step(&mut console) {
            // sleeps in Platform::wait until input or the next timer
            console.rt.events_wait();
        }
    }
}
