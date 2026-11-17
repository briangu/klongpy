//! The line editor (docs/native/index.md): one line of bytes, a cursor, ANSI escape
//! sequences in and out, and a ring of earlier lines.
//!
//! Keys: printable bytes and UTF-8 insert at the cursor; backspace (0x08, 0x7f) and delete
//! (`ESC [ 3 ~`); left and right; home and end (`ESC [ H`, `ESC [ F`, `ESC [ 1 ~`, `ESC [ 4 ~`,
//! `ESC O H`, `ESC O F`); up and down through the history; Ctrl-U empties the line; a tab is a
//! space. Line end, Ctrl-C, Ctrl-D and Ctrl-A at the start of a line are the console's, not the
//! editor's.
//!
//! Everything is bounded: the line by `max` bytes, the history by a number of lines and a
//! number of bytes, an escape sequence by 16 bytes. A code point is taken to be one column wide
//! and a control character two (`^A`); a line longer than the terminal is wide wraps, and
//! moving the cursor across the wrap is as good as the terminal makes it.
use alloc::vec::Vec;

/// Where the editor's echo goes.
pub trait Out {
    fn put(&mut self, bytes: &[u8]);
}

impl Out for Vec<u8> {
    fn put(&mut self, bytes: &[u8]) {
        self.extend_from_slice(bytes);
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Event {
    None,
    /// The byte did not fit in the line (or there was no memory for it).
    Overflow,
}

#[derive(Clone, Copy)]
enum Esc {
    None,
    /// After ESC.
    Start,
    /// After `ESC [`: the first parameter, whether it is finished, bytes seen.
    Csi(u16, bool, u8),
    /// After `ESC O`.
    Ss3,
}

pub struct Editor {
    pub buf: Vec<u8>,
    cur: usize,
    max: usize,
    esc: Esc,
    utf: [u8; 4],
    utf_len: u8,
    utf_need: u8,
    hist: Vec<Vec<u8>>,
    hist_lines: usize,
    hist_bytes: usize,
    hpos: usize,
    stash: Vec<u8>,
    /// No echo, no escape sequences, no history: scripted use.
    pub dumb: bool,
}

fn is_ctl(b: u8) -> bool {
    b < 0x20 || b == 0x7f
}

/// Columns the bytes take on the terminal.
fn cols(bytes: &[u8]) -> usize {
    bytes.iter().map(|&b| if is_ctl(b) { 2 } else if b & 0xC0 == 0x80 { 0 } else { 1 }).sum()
}

/// Write bytes as the terminal should show them: control characters as `^X`.
fn show(bytes: &[u8], out: &mut dyn Out) {
    let mut from = 0;
    for (i, &b) in bytes.iter().enumerate() {
        if is_ctl(b) {
            out.put(&bytes[from..i]);
            out.put(&[b'^', b ^ 0x40]);
            from = i + 1;
        }
    }
    out.put(&bytes[from..]);
}

/// `ESC [ n final`, nothing for n = 0.
fn csi(n: usize, last: u8, out: &mut dyn Out) {
    if n == 0 {
        return;
    }
    let mut digits = [0u8; 20];
    let mut at = digits.len();
    let mut n = n;
    while n > 0 {
        at -= 1;
        digits[at] = b'0' + (n % 10) as u8;
        n /= 10;
    }
    out.put(b"\x1b[");
    out.put(&digits[at..]);
    out.put(&[last]);
}

impl Editor {
    /// A line of at most `max` bytes; a history of at most `lines` lines and `bytes` bytes.
    pub fn new(max: usize, lines: usize, bytes: usize, dumb: bool) -> Editor {
        Editor {
            buf: Vec::new(),
            cur: 0,
            max,
            esc: Esc::None,
            utf: [0; 4],
            utf_len: 0,
            utf_need: 0,
            hist: Vec::new(),
            hist_lines: lines,
            hist_bytes: bytes,
            hpos: 0,
            stash: Vec::new(),
            dumb,
        }
    }

    /// In the middle of an escape sequence: the next byte is the editor's whatever it is.
    pub fn in_escape(&self) -> bool {
        !matches!(self.esc, Esc::None)
    }

    /// Forget the line (not the history).
    pub fn clear(&mut self) {
        self.buf.clear();
        self.cur = 0;
        self.esc = Esc::None;
        self.utf_len = 0;
        self.utf_need = 0;
        self.hpos = self.hist.len();
        self.stash.clear();
    }

    /// The line is about to be taken: bytes of an unfinished UTF-8 sequence join it as they
    /// are.
    pub fn finish(&mut self, out: &mut dyn Out) -> Event {
        self.esc = Esc::None;
        self.flush_utf(out)
    }

    /// Put the line in the history (not an empty one, not the one just before it again).
    pub fn remember(&mut self) {
        self.hpos = self.hist.len();
        if self.dumb || self.hist_lines == 0 || self.buf.is_empty() || self.buf.len() > self.hist_bytes {
            return;
        }
        if self.hist.last().map(|l| l[..] == self.buf[..]).unwrap_or(false) {
            return;
        }
        let mut line = Vec::new();
        if line.try_reserve_exact(self.buf.len()).is_err() || self.hist.try_reserve(1).is_err() {
            return;
        }
        line.extend_from_slice(&self.buf);
        self.hist.push(line);
        while self.hist.len() > self.hist_lines || self.hist.iter().map(|l| l.len()).sum::<usize>() > self.hist_bytes {
            self.hist.remove(0);
        }
        self.hpos = self.hist.len();
    }

    /// Bytes the history holds (tests, the memory budget).
    pub fn history_bytes(&self) -> usize {
        self.hist.iter().map(|l| l.len()).sum()
    }

    pub fn history_len(&self) -> usize {
        self.hist.len()
    }

    /// One byte from the terminal.
    pub fn key(&mut self, b: u8, out: &mut dyn Out) -> Event {
        if self.dumb {
            return self.dumb_key(b);
        }
        if b < 0x20 && b != 0x1b {
            // a control character ends an unfinished escape sequence and is itself
            self.esc = Esc::None;
        }
        match self.esc {
            Esc::None => {}
            Esc::Start => {
                self.esc = match b {
                    b'[' => Esc::Csi(0, false, 0),
                    b'O' => Esc::Ss3,
                    _ => Esc::None,
                };
                return Event::None;
            }
            Esc::Ss3 => {
                self.esc = Esc::None;
                self.function(0, b, out);
                return Event::None;
            }
            Esc::Csi(p, done, n) => {
                self.esc = Esc::None;
                if n >= 16 {
                    return Event::None;
                }
                match b {
                    b'0'..=b'9' if !done => self.esc = Esc::Csi(p.saturating_mul(10).saturating_add((b - b'0') as u16), false, n + 1),
                    0x20..=0x3f => self.esc = Esc::Csi(p, true, n + 1),
                    0x40..=0x7e => self.function(p, b, out),
                    _ => {}
                }
                return Event::None;
            }
        }
        match b {
            0x1b => {
                self.esc = Esc::Start;
                Event::None
            }
            0x08 | 0x7f => {
                if self.flush_utf(out) == Event::Overflow {
                    return Event::Overflow;
                }
                self.backspace(out);
                Event::None
            }
            0x15 => {
                self.utf_len = 0;
                self.utf_need = 0;
                self.replace(&[], out);
                Event::None
            }
            b'\t' => self.insert(b" ", out),
            _ => self.byte(b, out),
        }
    }

    fn dumb_key(&mut self, b: u8) -> Event {
        match b {
            0x08 | 0x7f => {
                self.buf.pop();
            }
            b'\t' => return self.dumb_key(b' '),
            _ if b >= 0x20 => {
                if self.buf.len() >= self.max || self.buf.try_reserve(1).is_err() {
                    return Event::Overflow;
                }
                self.buf.push(b);
            }
            _ => {}
        }
        self.cur = self.buf.len();
        Event::None
    }

    fn flush_utf(&mut self, out: &mut dyn Out) -> Event {
        if self.utf_need == 0 {
            return Event::None;
        }
        let (seq, n) = (self.utf, self.utf_len as usize);
        self.utf_len = 0;
        self.utf_need = 0;
        self.insert(&seq[..n], out)
    }

    /// A byte of text. The bytes of one UTF-8 sequence are inserted together, so that the
    /// terminal never sees half a character followed by something else.
    fn byte(&mut self, b: u8, out: &mut dyn Out) -> Event {
        if self.utf_need > 0 {
            if b & 0xC0 == 0x80 {
                self.utf[self.utf_len as usize] = b;
                self.utf_len += 1;
                if self.utf_len == self.utf_need {
                    return self.flush_utf(out);
                }
                return Event::None;
            }
            // a broken sequence: keep what came, as it came
            if self.flush_utf(out) == Event::Overflow {
                return Event::Overflow;
            }
        }
        let need = match b {
            0xC2..=0xDF => 2,
            0xE0..=0xEF => 3,
            0xF0..=0xF4 => 4,
            _ => 1,
        };
        if need > 1 {
            self.utf[0] = b;
            self.utf_len = 1;
            self.utf_need = need;
            return Event::None;
        }
        self.insert(&[b], out)
    }

    fn insert(&mut self, bytes: &[u8], out: &mut dyn Out) -> Event {
        if self.buf.len() + bytes.len() > self.max || self.buf.try_reserve(bytes.len()).is_err() {
            return Event::Overflow;
        }
        let at = self.cur.min(self.buf.len());
        if at == self.buf.len() {
            self.buf.extend_from_slice(bytes);
            self.cur = self.buf.len();
            show(bytes, out);
        } else {
            for (i, &b) in bytes.iter().enumerate() {
                self.buf.insert(at + i, b);
            }
            self.cur = at + bytes.len();
            show(&self.buf[at..], out);
            csi(cols(&self.buf[self.cur..]), b'D', out);
        }
        Event::None
    }

    /// Start of the character before `at`.
    fn before(&self, at: usize) -> usize {
        let mut p = at.saturating_sub(1);
        while p > 0 && self.buf[p] & 0xC0 == 0x80 {
            p -= 1;
        }
        p
    }

    /// End of the character at `at`.
    fn after(&self, at: usize) -> usize {
        let mut p = (at + 1).min(self.buf.len());
        while p < self.buf.len() && self.buf[p] & 0xC0 == 0x80 {
            p += 1;
        }
        p
    }

    fn backspace(&mut self, out: &mut dyn Out) {
        if self.cur == 0 {
            return;
        }
        let p = self.before(self.cur);
        let width = cols(&self.buf[p..self.cur]);
        self.buf.drain(p..self.cur);
        self.cur = p;
        if self.cur == self.buf.len() {
            // the form every terminal understands
            for _ in 0..width {
                out.put(b"\x08 \x08");
            }
        } else {
            csi(width, b'D', out);
            self.redraw_tail(out);
        }
    }

    fn delete(&mut self, out: &mut dyn Out) {
        if self.cur >= self.buf.len() {
            return;
        }
        let e = self.after(self.cur);
        self.buf.drain(self.cur..e);
        self.redraw_tail(out);
    }

    /// Rewrite what is right of the cursor and come back.
    fn redraw_tail(&mut self, out: &mut dyn Out) {
        show(&self.buf[self.cur..], out);
        out.put(b"\x1b[K");
        csi(cols(&self.buf[self.cur..]), b'D', out);
    }

    /// Replace the whole line and put the cursor at its end.
    fn replace(&mut self, line: &[u8], out: &mut dyn Out) {
        csi(cols(&self.buf[..self.cur]), b'D', out);
        out.put(b"\x1b[K");
        self.buf.clear();
        let n = line.len().min(self.max);
        if self.buf.try_reserve(n).is_ok() {
            self.buf.extend_from_slice(&line[..n]);
        }
        self.cur = self.buf.len();
        show(&self.buf, out);
    }

    fn function(&mut self, param: u16, last: u8, out: &mut dyn Out) {
        if self.flush_utf(out) == Event::Overflow {
            return;
        }
        match (last, param) {
            (b'D', _) => {
                if self.cur > 0 {
                    let p = self.before(self.cur);
                    csi(cols(&self.buf[p..self.cur]), b'D', out);
                    self.cur = p;
                }
            }
            (b'C', _) => {
                if self.cur < self.buf.len() {
                    let e = self.after(self.cur);
                    csi(cols(&self.buf[self.cur..e]), b'C', out);
                    self.cur = e;
                }
            }
            (b'H', _) | (b'~', 1) | (b'~', 7) => {
                csi(cols(&self.buf[..self.cur]), b'D', out);
                self.cur = 0;
            }
            (b'F', _) | (b'~', 4) | (b'~', 8) => {
                csi(cols(&self.buf[self.cur..]), b'C', out);
                self.cur = self.buf.len();
            }
            (b'~', 3) => self.delete(out),
            (b'A', _) => {
                if self.hpos == 0 || self.hist.is_empty() {
                    return;
                }
                if self.hpos >= self.hist.len() {
                    self.hpos = self.hist.len();
                    self.stash.clear();
                    if self.stash.try_reserve(self.buf.len()).is_ok() {
                        self.stash.extend_from_slice(&self.buf);
                    }
                }
                self.hpos -= 1;
                let line = core::mem::take(&mut self.hist[self.hpos]);
                self.replace(&line, out);
                self.hist[self.hpos] = line;
            }
            (b'B', _) => {
                if self.hpos >= self.hist.len() {
                    return;
                }
                self.hpos += 1;
                if self.hpos == self.hist.len() {
                    let line = core::mem::take(&mut self.stash);
                    self.replace(&line, out);
                } else {
                    let line = core::mem::take(&mut self.hist[self.hpos]);
                    self.replace(&line, out);
                    self.hist[self.hpos] = line;
                }
            }
            _ => {}
        }
    }
}
