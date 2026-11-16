//! klong-host: the `Platform` of desktop operating systems. Standard library only.
//!
//! Spike scope (ticket 102): time and the console. Files, TCP, pseudo-terminal links and the
//! lossy test link are tickets 103 to 106.
use std::io::Write;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use klong_rt::Platform;

/// Where what a program prints with `.p .d .w` goes.
pub enum Console {
    Stdout,
    Stderr,
    /// Collected in memory (tests).
    Buffer(Vec<u8>),
}

pub struct HostPlatform {
    start: Instant,
    pub console: Console,
}

impl HostPlatform {
    pub fn new(console: Console) -> HostPlatform {
        HostPlatform { start: Instant::now(), console }
    }
}

impl Platform for HostPlatform {
    fn now_us(&self) -> u64 {
        self.start.elapsed().as_micros() as u64
    }
    fn wall_clock(&self) -> Option<f64> {
        SystemTime::now().duration_since(UNIX_EPOCH).ok().map(|d| d.as_secs_f64())
    }
    fn idle(&mut self) {}
    fn console_read(&mut self, _buf: &mut [u8]) -> usize {
        // The host console is line-buffered standard input, read by the CLI between
        // evaluations; nothing is read while one runs (no Ctrl-C byte: the terminal sends a
        // signal instead).
        0
    }
    fn console_write(&mut self, bytes: &[u8]) {
        match &mut self.console {
            Console::Stdout => {
                let _ = std::io::stdout().write_all(bytes);
            }
            Console::Stderr => {
                let _ = std::io::stderr().write_all(bytes);
            }
            Console::Buffer(b) => b.extend_from_slice(bytes),
        }
    }
}
