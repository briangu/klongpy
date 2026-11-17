//! The platform interface (docs/native/index.md): everything outside pure computation.
//!
//! This is the core part of a platform: time, the console, storage. Transports, `wait`
//! and externs are added later; they get default implementations, so a
//! platform written against this trait keeps compiling.

/// Everything a runtime needs from where it runs.
pub trait Platform {
    /// Monotonic microseconds.
    fn now_us(&self) -> u64;
    /// Seconds since 1970, if known.
    fn wall_clock(&self) -> Option<f64>;
    /// Watchdog, yield. Called often (every 4096 calls or adverb iterations of an evaluation).
    fn idle(&mut self);
    /// The evaluator's slow tick (every `Config::tick_mask + 1` calls of Klong functions): `idle` by default. A
    /// simulation that charges time for evaluation does it here.
    fn slow_tick(&mut self) {
        self.idle()
    }
    /// An IPC message was encoded and sent, or one arrived whole and is being handled (the kdb+ encode or decode, the
    /// session, the evaluation of a request): nothing by default. A simulation that charges time for the processor
    /// does it here.
    fn message_cpu(&mut self) {}
    /// Non-blocking; 0 if nothing is waiting.
    fn console_read(&mut self, buf: &mut [u8]) -> usize;
    fn console_write(&mut self, bytes: &[u8]);
    /// The error channel (`.cerr`); the console unless the platform has a separate one.
    fn console_error(&mut self, bytes: &[u8]) {
        self.console_write(bytes)
    }
    /// Storage, if the platform has any. Every file function goes through it.
    fn fs(&mut self) -> Option<&mut dyn FileSystem> {
        None
    }
    /// A short name for `.ver()`: "host", the board's name.
    fn name(&self) -> &str {
        "unknown"
    }
    /// Bits to seed `.rn()` with. The default mixes the clocks; a board with a hardware
    /// random number generator returns its output.
    fn entropy(&mut self) -> u64 {
        let wall = self.wall_clock().map(|s| (s * 1e6) as u64).unwrap_or(0);
        self.now_us() ^ wall.rotate_left(32) ^ 0x9e37_79b9_7f4a_7c15
    }
    /// `.x(code)`: leave the program. A desktop ends the process; a board may reset or do
    /// nothing, in which case the running evaluation is interrupted.
    fn exit(&mut self, _code: i32) {}
    /// The networks of this platform: `.cli` and `.srv` find a transport by the
    /// name in the address. None by default.
    fn transports(&mut self) -> &mut [alloc::boxed::Box<dyn crate::link::Transport>] {
        &mut []
    }
    /// True while the console's input comes from a program, not a person (a lane's datagram console):
    /// the console then neither echoes nor prompts. Asked at every input byte, so input that arrives
    /// while a turn of the main loop is under way is treated as it should be.
    fn console_quiet(&self) -> bool {
        false
    }
    /// Sleep until `deadline_us` (of `now_us`) or until input may be ready, whichever is
    /// first; `None` means until input. Returning early is always allowed, and the default
    /// returns at once (a busy loop: correct, not frugal).
    fn wait(&mut self, _deadline_us: Option<u64>) {
        self.idle()
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct FileId(pub u32);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mode {
    /// An existing file, from its start.
    Read,
    /// A new or emptied file.
    Write,
    /// At the end of the file, created if missing.
    Append,
    /// An existing or new file, kept as it is: for writing at any position (`seek`), as the
    /// Intel HEX upload does.
    Update,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FsError {
    NotFound,
    Denied,
    NoSpace,
    /// A bad name, a directory where a file is needed, a closed or unknown `FileId`.
    Invalid,
    /// Too many open files.
    Busy,
    Io,
}

impl FsError {
    pub fn text(self) -> &'static str {
        match self {
            FsError::NotFound => "file does not exist",
            FsError::Denied => "permission denied",
            FsError::NoSpace => "no space left",
            FsError::Invalid => "invalid file or name",
            FsError::Busy => "too many open files",
            FsError::Io => "input/output error",
        }
    }
}

/// Storage. The only way the runtime touches files.
pub trait FileSystem {
    fn open(&mut self, path: &str, mode: Mode) -> Result<FileId, FsError>;
    /// Bytes read; 0 at end of file.
    fn read(&mut self, f: FileId, buf: &mut [u8]) -> Result<usize, FsError>;
    /// Bytes written (may be fewer than given).
    fn write(&mut self, f: FileId, buf: &[u8]) -> Result<usize, FsError>;
    /// Move the position. Beyond the end is allowed: a later write leaves a gap that reads
    /// as zero.
    fn seek(&mut self, f: FileId, pos: u64) -> Result<(), FsError>;
    fn close(&mut self, f: FileId);
    fn remove(&mut self, path: &str) -> Result<(), FsError>;
    fn rename(&mut self, from: &str, to: &str) -> Result<(), FsError>;
    /// Call `each` with the name and size of every file in `dir`.
    fn list(&mut self, dir: &str, each: &mut dyn FnMut(&str, u64)) -> Result<(), FsError>;
}
