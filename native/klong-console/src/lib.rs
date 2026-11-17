//! klong-console: the serial console of a Klong board (native-runtime.md section 8, ticket
//! 104). A read-eval-print loop and file upload over one byte stream; never IPC.
//!
//! [`Console`] owns the [`Runtime`] and is a state machine fed one byte at a time from
//! `Platform::console_read`. It never waits for input: [`Console::poll`] takes what has
//! arrived and returns. The only thing that takes time is an evaluation, and that one is
//! interrupted by Ctrl-C through the tick hook of `klong-rt`. Nothing it keeps grows without
//! a bound: every buffer has a limit in [`ConsoleConfig`], and going over one is an error
//! message.
//!
//! # The prompt
//!
//! `klong> ` reads an entry; `     > ` continues one that [`complete::check`] calls
//! incomplete (an open bracket, an open string or comment). The value of an entry prints as
//! KlongPy's REPL prints it, an error as `Error::render` gives it. Ctrl-C abandons the entry,
//! or stops the evaluation that is running; Ctrl-D is ignored. Editing keys: [`edit`].
//! A line ends with CR, LF or CR LF.
//!
//! # Command mode
//!
//! The byte 0x01 (Ctrl-A) as the first byte of an empty entry starts a command line, ended
//! like any line. Anywhere else 0x01 is ordinary input. Replies are single lines that start
//! with 0x01.
//!
//! ```text
//! ^A ver                              ^A ok ver=0.0.1 int=64 float=64 features=files,lib platform=... heap_free=...
//! ^A put <path> <length> <crc32-hex>  ^A go, then exactly <length> raw bytes, then ^A ok <length> <crc32> [warn ...]
//! ^A run <length> <crc32-hex>         ^A go, then the bytes; the program's output; then the value and ^A ok value <n>,
//!                                     or ^A ok (no value), or ^A err <kind>: <message>
//! ^A hex <path>                       ^A go, then Intel HEX records, one per line, each answered ^A ack <n> or
//!                                     ^A nak <n> <reason>; the end-of-file record is answered ^A ok <length> <crc32>
//! ```
//!
//! Every failure is `^A err <kind>: <message>`. The sender waits for `go` before it sends
//! data, so a refused command never has its data read as input. `put` and `hex` write
//! `<path>.tmp` and rename it once the check sum agrees; a `.kg` file is also parsed and a
//! syntax error is reported as `warn syntax: ...` in the `ok` line without refusing the file.
//! A transfer that sees no byte for `timeout_us` (5 s) ends with `err system` and the prompt
//! returns. In a `hex` transfer Ctrl-C abandons it. `<n>` of `ok value <n>` is the number of
//! bytes of the printed value, which stands on the line before the reply (so a tool can tell
//! it from what the program wrote).
#![no_std]
#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used)]

extern crate alloc;

pub mod complete;
pub mod crc;
pub mod edit;
pub mod hex;

use alloc::string::String;
use alloc::vec::Vec;
use core::fmt::Write;

use klong_core::fmt::format_value;
use klong_core::{Config, Error, Interp, NoHooks, Value};
use klong_rt::{FileId, FileSystem, FsError, MemStats, Mode, Platform, Runtime, Shared};

use complete::Verdict;
use crc::Crc32;
use edit::{Editor, Event, Out};

/// The limits and the looks of a console.
#[derive(Clone)]
pub struct ConsoleConfig {
    /// Bytes of one line.
    pub max_line: usize,
    /// Bytes of an entry of several lines.
    pub max_entry: usize,
    /// Bytes of a program sent with `run` (it is held in memory), and of a `.kg` file that is
    /// parsed after an upload.
    pub max_run: usize,
    /// Bytes of a file sent with `put` or `hex` (it goes to the file system as it arrives).
    pub max_file: u64,
    /// Lines and bytes of history.
    pub history_lines: usize,
    pub history_bytes: usize,
    /// A transfer that sees no input for this long is abandoned.
    pub timeout_us: u64,
    /// No echo, no editing keys, no history: for scripts.
    pub dumb: bool,
    pub prompt: &'static str,
    pub more: &'static str,
    /// Heap figures for `ver` (the same function the runtime got for `.mem`).
    pub mem: Option<MemStats>,
}

impl Default for ConsoleConfig {
    fn default() -> ConsoleConfig {
        ConsoleConfig {
            max_line: 1024,
            max_entry: 8192,
            max_run: 64 * 1024,
            max_file: 16 * 1024 * 1024,
            history_lines: 32,
            history_bytes: 2048,
            timeout_us: 5_000_000,
            dumb: false,
            prompt: "klong> ",
            more: "     > ",
            mem: None,
        }
    }
}

/// Bytes of a command line after the 0x01.
const COMMAND_MAX: usize = 255;
/// Bytes handled by one `poll`.
const POLL_BYTES: usize = 1024;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum State {
    /// Reading a line of an entry.
    Line,
    /// A line that was too long: swallow it to its end (`true`: it was a command line).
    Skip(bool),
    /// Reading a command line.
    Command,
    /// Counting the raw bytes of `put` or `run`.
    Raw,
    /// Reading Intel HEX records.
    Hex,
}

/// Writes to the console through the runtime's shared platform.
struct Port<'a, P: Platform>(&'a Shared<P>);

impl<P: Platform> Out for Port<'_, P> {
    fn put(&mut self, bytes: &[u8]) {
        if !bytes.is_empty() {
            if let Ok(mut p) = self.0.platform.try_borrow_mut() {
                p.console_write(bytes);
            }
        }
    }
}

/// The text of a reply line: control characters become blanks and the length is bounded, so
/// that a reply is always one line.
struct Reply<'a, P: Platform> {
    port: Port<'a, P>,
    left: usize,
}

impl<P: Platform> Write for Reply<'_, P> {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        let mut take = s.len().min(self.left);
        while !s.is_char_boundary(take) {
            take -= 1;
        }
        self.left -= take;
        let bytes = &s.as_bytes()[..take];
        let mut from = 0;
        for (i, &b) in bytes.iter().enumerate() {
            if b < 0x20 {
                self.port.put(&bytes[from..i]);
                self.port.put(b" ");
                from = i + 1;
            }
        }
        self.port.put(&bytes[from..]);
        Ok(())
    }
}

/// What a transfer in progress holds.
struct Transfer {
    /// The open `<path>.tmp`.
    file: Option<FileId>,
    /// Where the file goes; empty for `run`.
    path: String,
    /// `run`: the program.
    program: Option<Vec<u8>>,
    left: u64,
    total: u64,
    crc: Crc32,
    want: u32,
    /// The first error of the file system; the rest of the bytes are still counted.
    failed: Option<FsError>,
    chunk: [u8; 128],
    chunk_len: usize,
    // Intel HEX
    base: u32,
    end: u64,
    line: u32,
    skip: bool,
}

impl Transfer {
    fn new() -> Transfer {
        Transfer {
            file: None,
            path: String::new(),
            program: None,
            left: 0,
            total: 0,
            crc: Crc32::new(),
            want: 0,
            failed: None,
            chunk: [0; 128],
            chunk_len: 0,
            base: 0,
            end: 0,
            line: 0,
            skip: false,
        }
    }
}

pub struct Console<P: Platform + 'static> {
    pub rt: Runtime<P>,
    pub cfg: ConsoleConfig,
    ed: Editor,
    /// The earlier lines of an entry of several lines.
    entry: Vec<u8>,
    state: State,
    /// Bytes read from the platform and not handled yet.
    inbuf: [u8; 32],
    in_pos: usize,
    in_len: usize,
    /// Bytes the tick hook read while an evaluation ran, on their way here.
    typed: Vec<u8>,
    typed_pos: usize,
    after_cr: bool,
    /// No echo and no prompt: a datagram peer is typing (see `lane_begin`).
    quiet: bool,
    forced_quiet: bool,
    /// Errors printed, and of them interrupts (a console that is driven by a program counts them).
    errors: core::cell::Cell<u32>,
    interrupts: core::cell::Cell<u32>,
    now: u64,
    last_input: u64,
    xfer: Transfer,
}

fn tmp_name(path: &str) -> Option<String> {
    let mut s = String::new();
    s.try_reserve_exact(path.len() + 4).ok()?;
    s.push_str(path);
    s.push_str(".tmp");
    Some(s)
}

impl<P: Platform + 'static> Console<P> {
    pub fn new(rt: Runtime<P>, cfg: ConsoleConfig) -> Console<P> {
        let ed = Editor::new(cfg.max_line, cfg.history_lines, cfg.history_bytes, cfg.dumb);
        Console {
            rt,
            cfg,
            ed,
            entry: Vec::new(),
            state: State::Line,
            inbuf: [0; 32],
            in_pos: 0,
            in_len: 0,
            typed: Vec::new(),
            typed_pos: 0,
            after_cr: false,
            quiet: false,
            forced_quiet: false,
            errors: core::cell::Cell::new(0),
            interrupts: core::cell::Cell::new(0),
            now: 0,
            last_input: 0,
            xfer: Transfer::new(),
        }
    }

    /// Start-up: run `boot.kg` if the file system has one, print the banner and the prompt.
    pub fn start(&mut self) {
        #[cfg(feature = "files")]
        if let Err(e) = self.rt.boot() {
            self.print_error(&e);
        }
        {
            let name = self.platform_name();
            let mut r = Reply { port: self.port(), left: 120 };
            let _ = write!(r, "Klong {} ({}, int{} float{})", env!("CARGO_PKG_VERSION"), name.as_str(), klong_core::INT_BITS, klong_core::FLT_BITS);
        }
        self.out(b"\n");
        self.prompt();
    }

    /// True while a transfer or a command line is in progress (the prompt is not showing).
    pub fn busy(&self) -> bool {
        self.state != State::Line
    }

    /// No echo, no editing and no prompt while true: a program (not a person at a terminal) is
    /// typing. Results and errors are still printed. Set it between turns.
    pub fn set_quiet(&mut self, quiet: bool) {
        self.forced_quiet = quiet;
        self.sync_quiet();
    }

    /// Echo and prompt are off when the owner said so (`set_quiet`) or the platform says the input is
    /// a program's. Looked at for every input byte: input that arrives in the middle of a turn counts.
    fn sync_quiet(&mut self) {
        let q = self.forced_quiet || self.rt.shared.platform.try_borrow().map(|p| p.console_quiet()).unwrap_or(false);
        if q != self.quiet {
            self.quiet = q;
            self.ed.dumb = q || self.cfg.dumb;
        }
    }

    /// Input that the console has read and not handled yet (so that it is not between two inputs).
    pub fn input_pending(&self) -> bool {
        self.in_pos < self.in_len || self.typed_pos < self.typed.len() || self.rt.shared.pending.try_borrow().map(|q| !q.is_empty()).unwrap_or(false)
    }

    /// (errors printed, of which interrupts) since the start, wrapping.
    pub fn error_counts(&self) -> (u32, u32) {
        (self.errors.get(), self.interrupts.get())
    }

    /// Bytes this console holds on the heap right now (tests and the memory budget).
    pub fn held_bytes(&self) -> usize {
        self.ed.buf.capacity()
            + self.ed.history_bytes()
            + self.entry.capacity()
            + self.typed.capacity()
            + self.xfer.path.capacity()
            + self.xfer.program.as_ref().map(|p| p.capacity()).unwrap_or(0)
    }

    fn port(&self) -> Port<'_, P> {
        Port(&self.rt.shared)
    }

    fn out(&self, bytes: &[u8]) {
        self.port().put(bytes);
    }

    fn echo(&self, bytes: &[u8]) {
        if !self.cfg.dumb && !self.quiet {
            self.out(bytes);
        }
    }

    fn prompt(&self) {
        if self.quiet {
            return;
        }
        self.out(if self.entry.is_empty() { self.cfg.prompt } else { self.cfg.more }.as_bytes());
    }

    fn reply(&self, args: core::fmt::Arguments) {
        if alloc::format!("{}", args).starts_with("err") {
            self.errors.set(self.errors.get().wrapping_add(1));
        }
        self.out(b"\x01");
        let mut r = Reply { port: self.port(), left: 400 };
        let _ = r.write_fmt(args);
        self.out(b"\n");
    }

    fn print_error(&self, e: &Error) {
        self.errors.set(self.errors.get().wrapping_add(1));
        if e.kind == klong_core::Kind::Interrupt {
            self.interrupts.set(self.interrupts.get().wrapping_add(1));
        }
        self.out(e.render().as_bytes());
        self.out(b"\n");
    }

    fn platform_name(&self) -> String {
        let mut s = String::new();
        if let Ok(p) = self.rt.shared.platform.try_borrow() {
            let name = p.name();
            let n = name.len().min(32);
            if s.try_reserve(n).is_ok() {
                s.push_str(name.get(..n).unwrap_or(""));
            }
        }
        s
    }

    fn clock(&self) -> u64 {
        self.rt.shared.platform.try_borrow().map(|p| p.now_us()).unwrap_or(self.now)
    }

    fn with_fs<R>(&self, f: impl FnOnce(&mut dyn FileSystem) -> Result<R, FsError>) -> Result<R, FsError> {
        let mut p = self.rt.shared.platform.try_borrow_mut().map_err(|_| FsError::Busy)?;
        match p.fs() {
            Some(fs) => f(fs),
            None => Err(FsError::Invalid),
        }
    }

    fn has_fs(&self) -> bool {
        self.rt.shared.platform.try_borrow_mut().map(|mut p| p.fs().is_some()).unwrap_or(false)
    }

    // ----- input ------------------------------------------------------------------------------

    fn next_byte(&mut self) -> Option<u8> {
        if self.in_pos < self.in_len {
            self.in_pos += 1;
            return Some(self.inbuf[self.in_pos - 1]);
        }
        if self.typed_pos >= self.typed.len() {
            self.typed.clear();
            self.typed_pos = 0;
            if let Ok(mut q) = self.rt.shared.pending.try_borrow_mut() {
                if !q.is_empty() {
                    core::mem::swap(&mut *q, &mut self.typed);
                }
            }
        }
        if self.typed_pos < self.typed.len() {
            self.typed_pos += 1;
            return Some(self.typed[self.typed_pos - 1]);
        }
        let n = match self.rt.shared.platform.try_borrow_mut() {
            Ok(mut p) => p.console_read(&mut self.inbuf),
            Err(_) => 0,
        };
        self.in_len = n.min(self.inbuf.len());
        self.in_pos = 0;
        if self.in_len == 0 {
            return None;
        }
        self.in_pos = 1;
        Some(self.inbuf[0])
    }

    /// Input that was already read when an evaluation starts is invisible to the tick hook,
    /// which reads the port. If it holds a Ctrl-C, that Ctrl-C belongs to the evaluation about
    /// to run: arm the first tick to report it. (If the evaluation ends before a tick, the
    /// Ctrl-C is still in the input and is read at the prompt as usual.)
    fn arm_interrupt(&mut self) {
        self.rt.shared.ctrl_c_seen.set(false);
        self.rt.shared.armed_hit.set(false);
        let held = self.inbuf[self.in_pos..self.in_len].contains(&0x03)
            || self.typed[self.typed_pos.min(self.typed.len())..].contains(&0x03)
            || self.rt.shared.pending.try_borrow().map(|q| q.contains(&0x03)).unwrap_or(false);
        self.rt.shared.interrupt_armed.set(held);
    }

    /// After an evaluation. A Ctrl-C ends the evaluation and discards the typed-ahead input
    /// that came before it; later input is kept, in order, and so are later Ctrl-C bytes (one
    /// Ctrl-C ends one evaluation). If it was read from the port during the evaluation, all
    /// the input read ahead before the evaluation is older and goes; if it was in the input
    /// read ahead (armed), the input up to and including the first such Ctrl-C goes.
    fn end_eval(&mut self) {
        self.rt.shared.interrupt_armed.set(false);
        if self.rt.shared.ctrl_c_seen.replace(false) {
            self.in_pos = self.in_len;
            self.typed_pos = self.typed.len();
        } else if self.rt.shared.armed_hit.replace(false) {
            if let Some(i) = self.inbuf[self.in_pos..self.in_len].iter().position(|&b| b == 0x03) {
                self.in_pos += i + 1;
            } else if let Some(j) = self.typed[self.typed_pos.min(self.typed.len())..].iter().position(|&b| b == 0x03) {
                self.typed_pos += j + 1;
                self.in_pos = self.in_len;
            } else if let Ok(mut q) = self.rt.shared.pending.try_borrow_mut() {
                if let Some(k) = q.iter().position(|&b| b == 0x03) {
                    q.drain(..=k);
                    self.typed_pos = self.typed.len();
                    self.in_pos = self.in_len;
                }
            }
        }
    }

    /// Take the bytes that have arrived (at most a bounded number of them), run at most the
    /// work they complete, and check the transfer timeout. Returns false when there was
    /// nothing to do: the caller may then wait for input. Never waits itself.
    pub fn poll(&mut self) -> bool {
        self.now = self.clock();
        let mut worked = false;
        for _ in 0..POLL_BYTES {
            match self.next_byte() {
                Some(b) => {
                    self.feed(b);
                    worked = true;
                }
                None => break,
            }
        }
        self.now = self.clock();
        if matches!(self.state, State::Raw | State::Hex) && self.now.saturating_sub(self.last_input) > self.cfg.timeout_us {
            self.abandon("system", "transfer timed out");
            worked = true;
        }
        worked
    }

    /// One byte from the serial port.
    pub fn feed(&mut self, b: u8) {
        self.sync_quiet();
        if self.after_cr {
            self.after_cr = false;
            if b == b'\n' {
                return;
            }
        }
        if self.state == State::Raw {
            return self.raw_byte(b);
        }
        let b = if b == b'\r' {
            self.after_cr = true;
            b'\n'
        } else {
            b
        };
        match self.state {
            State::Line => self.line_byte(b),
            State::Skip(command) => {
                if b == b'\n' || b == 0x03 {
                    self.skip_end(command);
                }
            }
            State::Command => self.command_byte(b),
            State::Hex => self.hex_byte(b),
            State::Raw => {}
        }
    }

    /// The end of a line that was too long.
    fn skip_end(&mut self, command: bool) {
        self.echo(b"\n");
        self.state = State::Line;
        self.ed.clear();
        self.entry.clear();
        if command {
            self.reply(format_args!("err syntax: command longer than {} bytes", COMMAND_MAX));
        } else {
            let mut r = Reply { port: self.port(), left: 120 };
            let _ = write!(r, "range error: line longer than {} bytes", self.cfg.max_line);
            self.out(b"\n");
        }
        self.prompt();
    }

    fn line_byte(&mut self, b: u8) {
        // (a control character ends an escape sequence that was not finished: Ctrl-C and the
        // line end always mean what they mean)
        if self.ed.in_escape() && b >= 0x20 {
            let mut port = Port(&self.rt.shared);
            self.ed.key(b, &mut port);
            return;
        }
        match b {
            b'\n' => self.submit(),
            0x03 => {
                self.echo(b"^C\n");
                self.ed.clear();
                self.entry.clear();
                self.prompt();
            }
            0x04 => {}
            0x01 if self.ed.buf.is_empty() && self.entry.is_empty() => {
                self.ed.clear();
                self.state = State::Command;
            }
            _ => {
                let mut port = Port(&self.rt.shared);
                if self.ed.key(b, &mut port) == Event::Overflow {
                    self.state = State::Skip(false);
                }
            }
        }
    }

    /// The end of a line: evaluate the entry, or ask for more of it.
    fn submit(&mut self) {
        {
            let mut port = Port(&self.rt.shared);
            if self.ed.finish(&mut port) == Event::Overflow {
                return self.skip_end(false);
            }
        }
        self.echo(b"\n");
        self.ed.remember();
        let sep = usize::from(!self.entry.is_empty());
        let fits = self.entry.len() + sep + self.ed.buf.len() <= self.cfg.max_entry;
        if !fits || self.entry.try_reserve(sep + self.ed.buf.len()).is_err() {
            let mut r = Reply { port: self.port(), left: 120 };
            let _ = write!(r, "range error: entry longer than {} bytes", self.cfg.max_entry);
            self.out(b"\n");
            self.entry.clear();
            self.ed.clear();
            return self.prompt();
        }
        if sep == 1 {
            self.entry.push(b'\n');
        }
        self.entry.extend_from_slice(&self.ed.buf);
        self.ed.clear();
        if self.entry.iter().all(|b| b.is_ascii_whitespace()) {
            self.entry.clear();
            return self.prompt();
        }
        let entry = core::mem::take(&mut self.entry);
        match core::str::from_utf8(&entry) {
            Err(_) => self.out(b"syntax error: the input is not valid UTF-8\n"),
            Ok(text) => {
                if complete::check(text) == Verdict::Incomplete {
                    self.entry = entry;
                    return self.prompt();
                }
                self.eval_print(text);
            }
        }
        // keep the buffer of an ordinary entry, give back that of a large one
        if entry.capacity() <= self.cfg.max_line {
            self.entry = entry;
            self.entry.clear();
        }
        self.prompt();
    }

    fn eval_print(&mut self, text: &str) {
        self.arm_interrupt();
        let r = self.rt.run(text, "<repl>");
        self.end_eval();
        let r = r.and_then(|v| match v {
            Some(v) => format_value(&self.rt.interp.syms, &v, false).map(Some),
            None => Ok(None),
        });
        match r {
            Ok(Some(s)) => {
                self.out(s.as_bytes());
                self.out(b"\n");
            }
            Ok(None) => {}
            Err(e) => self.print_error(&e),
        }
    }

    // ----- command mode -----------------------------------------------------------------------

    fn command_byte(&mut self, b: u8) {
        match b {
            b'\n' => {
                self.echo(b"\n");
                self.command();
            }
            0x03 => {
                self.echo(b"^C\n");
                self.ed.clear();
                self.state = State::Line;
                self.prompt();
            }
            0x08 | 0x7f => {
                if self.ed.buf.pop().is_some() {
                    self.echo(b"\x08 \x08");
                }
            }
            _ if b >= 0x20 => {
                if self.ed.buf.len() >= COMMAND_MAX || self.ed.buf.try_reserve(1).is_err() {
                    self.state = State::Skip(true);
                } else {
                    self.ed.buf.push(b);
                    self.echo(&[b]);
                }
            }
            _ => {}
        }
    }

    /// Back to the prompt after a command that is over.
    fn done(&mut self) {
        self.xfer = Transfer::new();
        self.ed.clear();
        self.state = State::Line;
        self.prompt();
    }

    fn command(&mut self) {
        let line = core::mem::take(&mut self.ed.buf);
        self.run_command(&line);
        // (the buffer goes back to the editor; Hex mode uses it for its lines)
        self.ed.buf = line;
        self.ed.buf.clear();
        if !matches!(self.state, State::Raw | State::Hex) {
            self.done();
        } else if self.state == State::Raw && self.xfer.left == 0 {
            self.raw_done();
        }
    }

    fn run_command(&mut self, line: &[u8]) {
        let text = match core::str::from_utf8(line) {
            Ok(t) => t,
            Err(_) => return self.reply(format_args!("err syntax: the command is not valid UTF-8")),
        };
        let mut words = text.split_ascii_whitespace();
        let (cmd, a, b, c, more) = (words.next(), words.next(), words.next(), words.next(), words.next());
        let number = |s: &str| s.parse::<u64>().ok();
        let crc = |s: &str| if s.len() <= 8 { u32::from_str_radix(s, 16).ok() } else { None };
        match (cmd, a, b, c, more) {
            (Some("ver"), None, ..) => self.ver(),
            (Some("put"), Some(path), Some(len), Some(sum), None) => match (number(len), crc(sum)) {
                (Some(len), Some(sum)) => self.start_put(path, len, sum),
                _ => self.reply(format_args!("err syntax: usage: put <path> <length> <crc32-hex>")),
            },
            (Some("run"), Some(len), Some(sum), None, _) => match (number(len), crc(sum)) {
                (Some(len), Some(sum)) => self.start_run(len, sum),
                _ => self.reply(format_args!("err syntax: usage: run <length> <crc32-hex>")),
            },
            (Some("hex"), Some(path), None, ..) => self.start_hex(path),
            (Some("put"), ..) => self.reply(format_args!("err syntax: usage: put <path> <length> <crc32-hex>")),
            (Some("run"), ..) => self.reply(format_args!("err syntax: usage: run <length> <crc32-hex>")),
            (Some("hex"), ..) => self.reply(format_args!("err syntax: usage: hex <path>")),
            (Some("ver"), ..) => self.reply(format_args!("err syntax: usage: ver")),
            (Some(other), ..) => self.reply(format_args!("err syntax: unknown command {}: the commands are put, hex, run, ver", other)),
            (None, ..) => self.reply(format_args!("err syntax: empty command: the commands are put, hex, run, ver")),
        }
    }

    fn ver(&mut self) {
        // the features are the runtime's to name: ask .ver()
        let mut features = String::new();
        if let Ok(Some(Value::Dict(d))) = self.rt.run(".ver()", "<ver>") {
            if let (Ok(d), Ok(key)) = (d.try_borrow(), Value::text("features")) {
                if let Some(Ok(items)) = d.get(&key).map(|v| v.items()) {
                    for v in items {
                        if let Value::Str(s) = v {
                            if features.try_reserve(s.as_str().len() + 1).is_ok() {
                                if !features.is_empty() {
                                    features.push(',');
                                }
                                features.push_str(s.as_str());
                            }
                        }
                    }
                }
            }
        }
        let name = self.platform_name();
        let fs = self.has_fs();
        self.out(b"\x01");
        let mut r = Reply { port: self.port(), left: 400 };
        let _ = write!(
            r,
            "ok ver={} int={} float={} features={} platform={} fs={}",
            env!("CARGO_PKG_VERSION"),
            klong_core::INT_BITS,
            klong_core::FLT_BITS,
            if features.is_empty() { "-" } else { features.as_str() },
            if name.is_empty() { "-" } else { name.as_str() },
            u8::from(fs),
        );
        match self.cfg.mem.map(|f| f()) {
            Some((used, peak, limit)) if limit != 0 => {
                let _ = write!(r, " heap_free={} heap_used={} heap_peak={} heap_limit={}", limit.saturating_sub(used), used, peak, limit);
            }
            Some((used, peak, _)) => {
                let _ = write!(r, " heap_free=unknown heap_used={} heap_peak={} heap_limit=0", used, peak);
            }
            None => {
                let _ = write!(r, " heap_free=unknown");
            }
        }
        let _ = write!(r, " max_line={} max_run={} max_file={}", self.cfg.max_line, self.cfg.max_run, self.cfg.max_file);
        self.out(b"\n");
    }

    fn go(&mut self, state: State) {
        self.state = state;
        self.now = self.clock();
        self.last_input = self.now;
        self.reply(format_args!("go"));
    }

    /// Open `<path>.tmp` for a new file.
    fn open_tmp(&mut self, path: &str, update: bool) -> bool {
        if !self.has_fs() {
            self.reply(format_args!("err system: this board has no file system"));
            return false;
        }
        let (tmp, mut keep) = match (tmp_name(path), String::new()) {
            (Some(t), k) => (t, k),
            _ => {
                self.reply(format_args!("err memory: out of memory"));
                return false;
            }
        };
        if keep.try_reserve_exact(path.len()).is_err() {
            self.reply(format_args!("err memory: out of memory"));
            return false;
        }
        keep.push_str(path);
        let opened = self.with_fs(|fs| {
            let f = fs.open(&tmp, Mode::Write)?;
            if !update {
                return Ok(f);
            }
            // an emptied file, then open for writing at any position
            fs.close(f);
            fs.open(&tmp, Mode::Update)
        });
        match opened {
            Ok(f) => {
                self.xfer = Transfer::new();
                self.xfer.file = Some(f);
                self.xfer.path = keep;
                true
            }
            Err(e) => {
                self.reply(format_args!("err system: cannot write {}: {}", tmp.as_str(), e.text()));
                false
            }
        }
    }

    fn start_put(&mut self, path: &str, len: u64, sum: u32) {
        if len > self.cfg.max_file {
            return self.reply(format_args!("err range: {} bytes is more than a file may have here ({})", len, self.cfg.max_file));
        }
        if self.open_tmp(path, false) {
            self.xfer.left = len;
            self.xfer.total = len;
            self.xfer.want = sum;
            self.go(State::Raw);
        }
    }

    fn start_run(&mut self, len: u64, sum: u32) {
        if len > self.cfg.max_run as u64 {
            return self.reply(format_args!("err range: {} bytes is more than a program sent with run may have ({})", len, self.cfg.max_run));
        }
        let mut program = Vec::new();
        if program.try_reserve_exact(len as usize).is_err() {
            return self.reply(format_args!("err memory: out of memory"));
        }
        self.xfer = Transfer::new();
        self.xfer.program = Some(program);
        self.xfer.left = len;
        self.xfer.total = len;
        self.xfer.want = sum;
        self.go(State::Raw);
    }

    fn start_hex(&mut self, path: &str) {
        if self.open_tmp(path, true) {
            self.go(State::Hex);
        }
    }

    // ----- transfers --------------------------------------------------------------------------

    /// Close and remove what a transfer left behind.
    fn drop_transfer(&mut self) {
        let file = self.xfer.file.take();
        let tmp = if self.xfer.path.is_empty() { None } else { tmp_name(&self.xfer.path) };
        if file.is_some() || tmp.is_some() {
            let _ = self.with_fs(|fs| {
                if let Some(f) = file {
                    fs.close(f);
                }
                match &tmp {
                    Some(t) => fs.remove(t),
                    None => Ok(()),
                }
            });
        }
        self.xfer = Transfer::new();
    }

    /// End a transfer with an error and return to the prompt.
    fn abandon(&mut self, kind: &str, message: &str) {
        self.drop_transfer();
        self.reply(format_args!("err {}: {}", kind, message));
        self.done();
    }

    fn write_all(&mut self, pos: Option<u64>, len: usize, data: &[u8; 255]) -> Result<(), FsError> {
        let file = self.xfer.file.ok_or(FsError::Invalid)?;
        self.with_fs(|fs| {
            if let Some(pos) = pos {
                fs.seek(file, pos)?;
            }
            let mut at = 0;
            while at < len {
                match fs.write(file, &data[at..len])? {
                    0 => return Err(FsError::Io),
                    n => at += n,
                }
            }
            Ok(())
        })
    }

    fn flush_chunk(&mut self) {
        let n = self.xfer.chunk_len;
        self.xfer.chunk_len = 0;
        if n == 0 || self.xfer.failed.is_some() {
            return;
        }
        let mut data = [0u8; 255];
        data[..n].copy_from_slice(&self.xfer.chunk[..n]);
        if let Err(e) = self.write_all(None, n, &data) {
            self.xfer.failed = Some(e);
        }
    }

    fn raw_byte(&mut self, b: u8) {
        self.last_input = self.now;
        self.xfer.crc.update(&[b]);
        match &mut self.xfer.program {
            // (the room was reserved when the command was accepted)
            Some(p) => {
                if p.len() < p.capacity() {
                    p.push(b);
                }
            }
            None => {
                self.xfer.chunk[self.xfer.chunk_len] = b;
                self.xfer.chunk_len += 1;
                if self.xfer.chunk_len == self.xfer.chunk.len() {
                    self.flush_chunk();
                }
            }
        }
        self.xfer.left = self.xfer.left.saturating_sub(1);
        if self.xfer.left == 0 {
            self.raw_done();
        }
    }

    /// The last byte of `put` or `run` has arrived.
    fn raw_done(&mut self) {
        let got = self.xfer.crc.finish();
        let want = self.xfer.want;
        if let Some(program) = self.xfer.program.take() {
            self.xfer = Transfer::new();
            self.state = State::Line;
            if got != want {
                self.reply(format_args!("err system: check sum {:08x}, expected {:08x}", got, want));
            } else {
                match core::str::from_utf8(&program) {
                    Ok(text) => self.run_program(text),
                    Err(_) => self.reply(format_args!("err syntax: the program is not valid UTF-8")),
                }
            }
            return self.done();
        }
        self.flush_chunk();
        if let Some(e) = self.xfer.failed {
            let e = e.text();
            return self.abandon("system", e);
        }
        if got != want {
            self.drop_transfer();
            self.reply(format_args!("err system: check sum {:08x}, expected {:08x}", got, want));
            return self.done();
        }
        let total = self.xfer.total;
        self.finish_file(total, got);
    }

    /// Evaluate a program sent with `run` and answer.
    fn run_program(&mut self, text: &str) {
        self.arm_interrupt();
        let r = self.rt.run(text, "<run>");
        self.end_eval();
        let r = r.and_then(|v| match v {
            Some(v) => format_value(&self.rt.interp.syms, &v, false).map(Some),
            None => Ok(None),
        });
        match r {
            Ok(Some(s)) => {
                self.out(s.as_bytes());
                self.out(b"\n");
                self.reply(format_args!("ok value {}", s.len()));
            }
            Ok(None) => self.reply(format_args!("ok")),
            Err(e) => self.reply_error(&e),
        }
    }

    fn reply_error(&self, e: &Error) {
        match (&e.source, e.span) {
            (Some(src), Some(span)) => {
                let (line, col) = src.line_col(span.0);
                self.reply(format_args!("err {}: {} at {}:{}", e.kind.name(), e.message, line, col));
            }
            _ => self.reply(format_args!("err {}: {}", e.kind.name(), e.message)),
        }
    }

    /// The temporary file is whole: close it, put it in place, look at it if it is Klong.
    fn finish_file(&mut self, len: u64, sum: u32) {
        let file = self.xfer.file.take();
        let path = core::mem::take(&mut self.xfer.path);
        let tmp = match tmp_name(&path) {
            Some(t) => t,
            None => {
                self.xfer.file = file;
                self.xfer.path = path;
                return self.abandon("memory", "out of memory");
            }
        };
        let renamed = self.with_fs(|fs| {
            if let Some(f) = file {
                fs.close(f);
            }
            fs.rename(&tmp, &path)
        });
        if let Err(e) = renamed {
            let _ = self.with_fs(|fs| fs.remove(&tmp));
            self.reply(format_args!("err system: cannot put {} in place: {}", path.as_str(), e.text()));
            return self.done();
        }
        if !path.ends_with(".kg") {
            self.reply(format_args!("ok {} {:08x}", len, sum));
        } else if len > self.cfg.max_run as u64 {
            self.reply(format_args!("ok {} {:08x} unchecked", len, sum));
        } else {
            match self.syntax_of(&path, len as usize) {
                Ok(()) => self.reply(format_args!("ok {} {:08x}", len, sum)),
                Err(e) => match (&e.source, e.span) {
                    (Some(src), Some(span)) => {
                        let (line, col) = src.line_col(span.0);
                        self.reply(format_args!("ok {} {:08x} warn {}: {} at {}:{}", len, sum, e.kind.name(), e.message, line, col));
                    }
                    _ => self.reply(format_args!("ok {} {:08x} warn {}: {}", len, sum, e.kind.name(), e.message)),
                },
            }
        }
        self.done();
    }

    /// Parse a stored file without running it. A second, empty interpreter does the parsing,
    /// so that the names and the module state of the running one are untouched.
    fn syntax_of(&mut self, path: &str, len: usize) -> Result<(), Error> {
        let mut data: Vec<u8> = Vec::new();
        data.try_reserve_exact(len).map_err(|_| Error::memory())?;
        let read = self.with_fs(|fs| {
            let f = fs.open(path, Mode::Read)?;
            let mut chunk = [0u8; 128];
            let r = loop {
                match fs.read(f, &mut chunk) {
                    Ok(0) => break Ok(()),
                    Ok(n) => {
                        if data.len() + n > len {
                            break Err(FsError::Io);
                        }
                        data.extend_from_slice(&chunk[..n]);
                    }
                    Err(e) => break Err(e),
                }
            };
            fs.close(f);
            r
        });
        read.map_err(|e| Error::system(e.text()))?;
        let text = core::str::from_utf8(&data).map_err(|_| Error::syntax("the file is not valid UTF-8"))?;
        let live = &self.rt.interp.cfg;
        let cfg = Config { strict: live.strict, max_depth: live.max_depth, stack_limit: live.stack_limit, max_nest: live.max_nest, mem_probe: live.mem_probe, ..Config::default() };
        let mut scratch = Interp::new(cfg, alloc::boxed::Box::new(NoHooks));
        scratch.parse(text, path).map(|_| ())
    }

    // ----- Intel HEX --------------------------------------------------------------------------

    fn hex_byte(&mut self, b: u8) {
        self.last_input = self.now;
        match b {
            0x03 => self.abandon("interrupt", "transfer abandoned"),
            b'\n' => {
                if self.xfer.skip {
                    self.xfer.skip = false;
                    self.xfer.line += 1;
                    self.reply(format_args!("nak {} line too long", self.xfer.line));
                } else if self.ed.buf.iter().any(|b| !b.is_ascii_whitespace()) {
                    self.xfer.line += 1;
                    let line = core::mem::take(&mut self.ed.buf);
                    self.hex_line(&line);
                    if self.state == State::Hex {
                        self.ed.buf = line;
                    }
                }
                self.ed.buf.clear();
            }
            _ => {
                if !self.xfer.skip {
                    if self.ed.buf.len() >= hex::MAX_LINE + 8 || self.ed.buf.try_reserve(1).is_err() {
                        self.xfer.skip = true;
                    } else {
                        self.ed.buf.push(b);
                    }
                }
            }
        }
    }

    fn hex_line(&mut self, line: &[u8]) {
        let n = self.xfer.line;
        let mut data = [0u8; hex::MAX_DATA];
        let record = match hex::parse(line, &mut data) {
            Ok(r) => r,
            Err(e) => return self.reply(format_args!("nak {} {}", n, e.text())),
        };
        match record {
            hex::Record::Data { addr, data: d } => {
                let len = d.len();
                let pos = self.xfer.base as u64 + addr as u64;
                if pos + len as u64 > self.cfg.max_file {
                    return self.reply(format_args!("nak {} file too large", n));
                }
                if len > 0 {
                    if let Err(e) = self.write_all(Some(pos), len, &data) {
                        return self.reply(format_args!("nak {} {}", n, e.text()));
                    }
                    self.xfer.end = self.xfer.end.max(pos + len as u64);
                }
                self.reply(format_args!("ack {}", n));
            }
            hex::Record::ExtSegment(s) => {
                self.xfer.base = (s as u32) << 4;
                self.reply(format_args!("ack {}", n));
            }
            hex::Record::ExtLinear(u) => {
                self.xfer.base = (u as u32) << 16;
                self.reply(format_args!("ack {}", n));
            }
            hex::Record::Start => self.reply(format_args!("ack {}", n)),
            hex::Record::Eof => self.hex_done(),
        }
    }

    /// The end-of-file record: the length and check sum of what was stored are the answer, so
    /// that the sender can tell a file with a record missing from a whole one.
    fn hex_done(&mut self) {
        let file = self.xfer.file.take();
        let tmp = match tmp_name(&self.xfer.path) {
            Some(t) => t,
            None => {
                self.xfer.file = file;
                return self.abandon("memory", "out of memory");
            }
        };
        let summed = self.with_fs(|fs| {
            if let Some(f) = file {
                fs.close(f);
            }
            let f = fs.open(&tmp, Mode::Read)?;
            let mut crc = Crc32::new();
            let mut len = 0u64;
            let mut chunk = [0u8; 128];
            let r = loop {
                match fs.read(f, &mut chunk) {
                    Ok(0) => break Ok((len, crc.finish())),
                    Ok(n) => {
                        crc.update(&chunk[..n]);
                        len += n as u64;
                    }
                    Err(e) => break Err(e),
                }
            };
            fs.close(f);
            r
        });
        match summed {
            Ok((len, sum)) => self.finish_file(len, sum),
            Err(e) => self.abandon("system", e.text()),
        }
    }
}

#[cfg(feature = "ipc")]
impl<P: Platform + 'static> Console<P> {
    /// Before one turn of the main loop on a board whose console also takes input from a lane's
    /// datagram channel (`klong_rt::node::NodeNet`): no echo and no prompt while a datagram's lines
    /// are being evaluated.
    pub fn lane_begin(&mut self, net: &klong_rt::node::NodeNet) {
        net.pump();
        self.set_quiet(net.console_quiet());
    }

    /// After the turn: tells the lane whether the console has finished with the datagram it was
    /// given (and how: errors, interrupts), so that `done ID STATUS` goes out after its output.
    pub fn lane_end(&mut self, net: &klong_rt::node::NodeNet) {
        let (e, i) = self.error_counts();
        net.console_turn(self.input_pending(), e, i);
    }
}
