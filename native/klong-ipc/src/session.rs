//! kdb+ IPC session: one end of one connection, as a state machine with no I/O. The Rust twin
//! of `klongpy/ipc/session.py` (design `docs/ipc_capabilities.md`).
//!
//! The owner of the link
//!
//! * gives received bytes to [`Session::feed`] and takes events with [`Session::next_event`],
//!   one at a time (take them all before feeding more: that is what bounds the buffer to one
//!   message plus one read);
//! * asks the session to frame what it wants to send and writes the returned bytes;
//! * tells it when it starts and ends serving an incoming request, so that responses can be
//!   matched without message ids.
//!
//! The session reads no clock: deadlines are microsecond values the caller compares through
//! [`Session::check_deadlines`].
//!
//! Handshake: a client sends `user:password`, a capability byte and a NUL ([`Session::hello`]);
//! the first byte it receives is the server's answer. A server reads the first NUL-terminated
//! string (at most 256 bytes) and reports [`Event::Handshake`]; the endpoint decides and calls
//! [`Session::accept`] (one reply byte) or [`Session::reject`]. We offer 3 and answer
//! `min(peer, 3)` to peers offering 1 to 6; anything else is refused.
//!
//! Framing: 8-byte header (byte order, message type, compression flag, reserved zero, int32
//! total length). The length is checked against `max_message` before anything is buffered, and
//! so is the declared size of a compressed message.
//!
//! Response matching: outstanding sync requests live in levels; level 0
//! holds those of ordinary code; [`Session::begin_service`] opens a deeper level when a
//! blocking call is outstanding, and requests issued while serving go there; a response
//! completes the oldest request of the deepest level that has any; [`Session::end_service`]
//! closes the level, and one that still holds requests is a protocol error.
use alloc::collections::VecDeque;
use alloc::string::String;
use alloc::vec::Vec;

use klong_core::error::{Error, Result};
use klong_core::sym::Interner;
use klong_core::Value;

use crate::codec::{self, DecodeError};
use crate::compress;

/// Capability byte we send and the highest we negotiate.
pub const DEFAULT_CAPABILITY: u8 = 3;
/// A server reads at most this many bytes looking for the NUL of the greeting.
pub const HANDSHAKE_LIMIT: usize = 256;
/// Default depth of nested service levels.
pub const MAX_NESTING: usize = 8;
/// Default bound of one message on a host (a device sets it from its heap).
pub const MAX_MESSAGE: usize = 256 * 1024 * 1024;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    Async = 0,
    Sync = 1,
    Response = 2,
}

pub type PendingId = u32;

/// One outstanding sync request.
#[derive(Clone, Copy, Debug)]
pub struct Pending {
    pub id: PendingId,
    /// Its caller is blocked waiting (as opposed to an `.async` request).
    pub blocking: bool,
    /// Microseconds on the caller's clock.
    pub deadline: Option<u64>,
    pub level: u32,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Reason {
    Eof,
    /// The stream ended inside a message.
    Truncated,
    Timeout,
}

pub enum Event {
    /// Server: a client's greeting. Answer with `accept()` or `reject()`.
    Handshake { user: String, password: String, capability: u8 },
    /// Client: the server's reply byte; the negotiated capability.
    Accepted { capability: u8 },
    /// Client: the connection ended before the server's reply byte.
    Rejected,
    /// One complete message. For a response, `pending` is the request it answers. `value` is
    /// an error when the framing was fine but this build cannot hold the value: a problem of
    /// this message only.
    Message { kind: Kind, value: Result<Value>, pending: Option<PendingId>, size: usize },
    /// Fatal protocol failure: the session is closed; close the link and fail `pending`.
    Failure { error: String, pending: Vec<Pending> },
    /// The connection ended; `pending` will never be answered.
    Closed { reason: Reason, pending: Vec<Pending> },
}

/// An incoming request being served.
#[derive(Debug)]
pub struct Service {
    pub kind: Kind,
    opened_level: bool,
    answered: bool,
    serial: u32,
}

/// The request would nest deeper than `max_nesting`: answer it with [`Session::refuse`].
#[derive(Debug)]
pub struct NestingLimit;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum State {
    Handshake,
    Auth,
    Ready,
    Closed,
}

#[derive(Clone, Copy)]
pub struct Config {
    pub capability: u8,
    pub max_message: usize,
    pub max_nesting: usize,
}

impl Default for Config {
    fn default() -> Config {
        Config { capability: DEFAULT_CAPABILITY, max_message: MAX_MESSAGE, max_nesting: MAX_NESTING }
    }
}

pub struct Session {
    client: bool,
    cfg: Config,
    state: State,
    /// Negotiated capability, once the handshake is done.
    pub capability: u8,
    peer_capability: u8,
    hello_sent: bool,
    buf: Vec<u8>,
    off: usize,
    levels: Vec<VecDeque<Pending>>,
    services: Vec<u32>,
    next_id: PendingId,
    next_service: u32,
}

fn state_error(what: &'static str) -> Error {
    Error::internal(what)
}

impl Session {
    fn new(client: bool, cfg: Config) -> Session {
        let mut levels = Vec::new();
        levels.push(VecDeque::new());
        let capability = cfg.capability.clamp(1, 6);
        Session {
            client,
            cfg: Config { capability, ..cfg },
            state: State::Handshake,
            capability: 0,
            peer_capability: 0,
            hello_sent: false,
            buf: Vec::new(),
            off: 0,
            levels,
            services: Vec::new(),
            next_id: 1,
            next_service: 1,
        }
    }

    pub fn client(cfg: Config) -> Session {
        Session::new(true, cfg)
    }

    pub fn server(cfg: Config) -> Session {
        Session::new(false, cfg)
    }

    pub fn is_closed(&self) -> bool {
        self.state == State::Closed
    }

    pub fn is_ready(&self) -> bool {
        self.state == State::Ready
    }

    /// Number of open service levels above level 0.
    pub fn nesting(&self) -> usize {
        self.levels.len() - 1
    }

    /// Bytes buffered and not yet consumed.
    pub fn buffered(&self) -> usize {
        self.buf.len() - self.off
    }

    /// Outstanding requests, deepest level first, oldest first within a level.
    pub fn pending(&self) -> Vec<Pending> {
        let mut out = Vec::new();
        for lvl in self.levels.iter().rev() {
            for p in lvl {
                if out.try_reserve(1).is_ok() {
                    out.push(*p);
                }
            }
        }
        out
    }

    pub fn has_pending(&self) -> bool {
        self.levels.iter().any(|l| !l.is_empty())
    }

    // ----- handshake ---------------------------------------------------------------------

    /// Client: the bytes to send first. Credentials must be ASCII without NUL.
    pub fn hello(&mut self, user: &str, password: &str) -> Result<Vec<u8>> {
        if !self.client || self.hello_sent || self.state != State::Handshake {
            return Err(state_error("hello() is the client's first and only send"));
        }
        for part in [user, password] {
            if !part.is_ascii() || part.as_bytes().contains(&0) {
                return Err(Error::typ("credentials must be ASCII without NUL"));
            }
        }
        let mut out = Vec::new();
        out.try_reserve(user.len() + password.len() + 3).map_err(|_| Error::memory())?;
        out.extend_from_slice(user.as_bytes());
        out.push(b':');
        out.extend_from_slice(password.as_bytes());
        out.push(self.cfg.capability);
        out.push(0);
        self.hello_sent = true;
        Ok(out)
    }

    /// Server: accept the greeting; the single reply byte to send.
    pub fn accept(&mut self) -> Result<[u8; 1]> {
        if self.state != State::Auth {
            return Err(state_error("accept() answers a Handshake event"));
        }
        self.capability = self.peer_capability.min(self.cfg.capability);
        self.state = State::Ready;
        Ok([self.capability])
    }

    /// Server: refuse the greeting. kdb+ just closes the connection.
    pub fn reject(&mut self) {
        self.close();
    }

    // ----- receiving ---------------------------------------------------------------------

    /// Give the session received bytes. A `memory` error when they cannot be buffered (the
    /// session is then closed).
    pub fn feed(&mut self, data: &[u8]) -> Result<()> {
        if self.state == State::Closed || data.is_empty() {
            return Ok(());
        }
        if self.off > 0 && self.off == self.buf.len() {
            self.buf.clear();
            self.off = 0;
        } else if self.off > 4096 && self.off * 2 > self.buf.len() {
            self.buf.drain(..self.off);
            self.off = 0;
        }
        if self.buf.try_reserve(data.len()).is_err() {
            self.close();
            return Err(Error::memory());
        }
        self.buf.extend_from_slice(data);
        Ok(())
    }

    /// The next event the buffered bytes complete, if any. Symbols of a decoded value are
    /// interned in `syms`.
    pub fn next_event(&mut self, syms: &mut Interner) -> Option<Event> {
        let r = match self.state {
            State::Handshake => self.step_handshake(),
            State::Ready => self.step_frame(syms),
            State::Auth | State::Closed => Ok(None),
        };
        match r {
            Ok(ev) => ev,
            Err(text) => Some(self.fail(text)),
        }
    }

    /// The transport saw end-of-stream.
    pub fn feed_eof(&mut self) -> Option<Event> {
        if self.state == State::Closed {
            return None;
        }
        let in_handshake = self.state == State::Handshake;
        let partial = self.buffered() > 0;
        let pending = self.drain_pending();
        self.shut();
        if in_handshake && self.client {
            return Some(Event::Rejected);
        }
        Some(Event::Closed { reason: if partial { Reason::Truncated } else { Reason::Eof }, pending })
    }

    fn shut(&mut self) {
        self.state = State::Closed;
        self.buf = Vec::new();
        self.off = 0;
    }

    fn fail(&mut self, text: String) -> Event {
        let pending = self.drain_pending();
        self.shut();
        Event::Failure { error: text, pending }
    }

    fn step_handshake(&mut self) -> core::result::Result<Option<Event>, String> {
        let avail = &self.buf[self.off..];
        if self.client {
            if !self.hello_sent {
                if avail.is_empty() {
                    return Ok(None);
                }
                return Err(String::from("data received before hello was sent"));
            }
            let reply = match avail.first() {
                Some(&b) => b,
                None => return Ok(None),
            };
            self.off += 1;
            if reply > 6 {
                return Err(String::from("bad capability byte in the server's reply"));
            }
            self.capability = reply.min(self.cfg.capability);
            self.state = State::Ready;
            return Ok(Some(Event::Accepted { capability: self.capability }));
        }
        // server: the greeting ends at the first NUL, within HANDSHAKE_LIMIT bytes
        let window = &avail[..avail.len().min(HANDSHAKE_LIMIT)];
        let end = match window.iter().position(|&b| b == 0) {
            Some(e) => e,
            None => {
                if avail.len() >= HANDSHAKE_LIMIT {
                    return Err(String::from("no end of greeting within 256 bytes"));
                }
                return Ok(None);
            }
        };
        let text = &avail[..end];
        let (cap, creds) = match text.split_last() {
            Some((c, rest)) => (*c, rest),
            None => return Err(String::from("empty greeting")),
        };
        if !(1..=6).contains(&cap) {
            // includes capability 0 in both of its forms
            return Err(String::from("unsupported capability byte"));
        }
        if !creds.is_ascii() {
            return Err(String::from("credentials are not ASCII"));
        }
        let creds = core::str::from_utf8(creds).unwrap_or("");
        let (user, password) = match creds.find(':') {
            Some(i) => (&creds[..i], &creds[i + 1..]),
            None => (creds, ""),
        };
        let (user, password) = (String::from(user), String::from(password));
        self.off += end + 1;
        self.peer_capability = cap;
        self.state = State::Auth;
        Ok(Some(Event::Handshake { user, password, capability: cap }))
    }

    fn step_frame(&mut self, syms: &mut Interner) -> core::result::Result<Option<Event>, String> {
        let avail = self.buf.len() - self.off;
        if avail < 8 {
            return Ok(None);
        }
        let o = self.off;
        let hdr = &self.buf[o..o + 8];
        if hdr[0] > 1 {
            return Err(String::from("bad byte-order flag"));
        }
        if hdr[1] > 2 {
            return Err(String::from("bad message type"));
        }
        if hdr[2] > 1 {
            return Err(String::from("bad compression flag"));
        }
        if hdr[3] != 0 {
            return Err(String::from("reserved header byte set (large messages are not negotiated)"));
        }
        let little = hdr[0] == 1;
        let compressed = hdr[2] == 1;
        let kind = match hdr[1] {
            0 => Kind::Async,
            1 => Kind::Sync,
            _ => Kind::Response,
        };
        let raw = [hdr[4], hdr[5], hdr[6], hdr[7]];
        let length = if little { i32::from_le_bytes(raw) } else { i32::from_be_bytes(raw) };
        if length < 8 || (compressed && length < 12) {
            return Err(String::from("message length shorter than its header"));
        }
        let length = length as usize;
        if length > self.cfg.max_message {
            return Err(klong_core::msg!("message length {} exceeds the limit of {}", length, self.cfg.max_message));
        }
        if avail < length {
            // room for the rest in one step, so that a large message is not copied as it grows
            let _ = self.buf.try_reserve(length - avail);
            return Ok(None);
        }
        let frame = &self.buf[o..o + length];
        let decoded = if compressed {
            match compress::decompress(frame, self.cfg.max_message) {
                Ok(plain) => codec::decode(&plain[8..], little, syms),
                Err(compress::Failure::Memory) => Err(DecodeError::Value(Error::memory())),
                Err(f) => return Err(compress::failure_text(&f)),
            }
        } else {
            codec::decode(&frame[8..], little, syms)
        };
        self.off = o + length;
        if self.off == self.buf.len() {
            // a large message is not kept after it is decoded
            if self.buf.capacity() > 65536 {
                self.buf = Vec::new();
            } else {
                self.buf.clear();
            }
            self.off = 0;
        }
        let value = match decoded {
            Ok(v) => Ok(v),
            Err(DecodeError::Value(e)) => Err(e),
            Err(DecodeError::Protocol(s)) => return Err(s),
        };
        let pending = if kind == Kind::Response {
            match self.match_response() {
                Some(p) => Some(p),
                None => return Err(String::from("response without an outstanding request")),
            }
        } else {
            None
        };
        Ok(Some(Event::Message { kind, value, pending, size: length }))
    }

    fn match_response(&mut self) -> Option<PendingId> {
        for lvl in self.levels.iter_mut().rev() {
            if let Some(p) = lvl.pop_front() {
                return Some(p.id);
            }
        }
        None
    }

    // ----- sending -----------------------------------------------------------------------

    fn require_ready(&self) -> Result<()> {
        if self.state != State::Ready {
            return Err(Error::system("the connection is not open"));
        }
        Ok(())
    }

    /// A message: header and the object `body` writes (always little-endian).
    fn frame(&self, kind: Kind, compress_body: bool, body: &mut dyn FnMut(&mut Vec<u8>) -> Result<()>) -> Result<Vec<u8>> {
        self.require_ready()?;
        let mut msg = Vec::new();
        msg.try_reserve(64).map_err(|_| Error::memory())?;
        msg.extend_from_slice(&[1, kind as u8, 0, 0, 0, 0, 0, 0]);
        body(&mut msg)?;
        if msg.len() > i32::MAX as usize {
            return Err(Error::range("message exceeds 2 GB"));
        }
        let n = (msg.len() as u32).to_le_bytes();
        msg[4..8].copy_from_slice(&n);
        #[cfg(feature = "compress")]
        if compress_body && self.capability >= 1 {
            if let Some(z) = compress::compress(&msg, compress::COMPRESS_THRESHOLD) {
                return Ok(z);
            }
        }
        let _ = compress_body;
        Ok(msg)
    }

    /// Frame an async (type 0) message; nothing comes back.
    pub fn send_async(&mut self, syms: &Interner, value: &Value, compress_body: bool) -> Result<Vec<u8>> {
        self.frame(Kind::Async, compress_body, &mut |out| codec::encode(syms, value, out))
    }

    fn register(&mut self, blocking: bool, deadline: Option<u64>) -> Result<PendingId> {
        let id = self.next_id;
        let level = self.levels.len() - 1;
        let lvl = self.levels.last_mut().ok_or_else(|| state_error("no level"))?;
        lvl.try_reserve(1).map_err(|_| Error::memory())?;
        lvl.push_back(Pending { id, blocking, deadline, level: level as u32 });
        self.next_id = self.next_id.wrapping_add(1).max(1);
        Ok(id)
    }

    /// Frame a sync (type 1) request holding `value` and register it as outstanding.
    pub fn send_sync(&mut self, syms: &Interner, value: &Value, blocking: bool, deadline: Option<u64>, compress_body: bool) -> Result<(Vec<u8>, PendingId)> {
        let data = self.frame(Kind::Sync, compress_body, &mut |out| codec::encode(syms, value, out))?;
        Ok((data, self.register(blocking, deadline)?))
    }

    /// The same for a call envelope: `items` always go out as a general list.
    pub fn send_call(&mut self, syms: &Interner, items: &[Value], blocking: bool, deadline: Option<u64>, compress_body: bool) -> Result<(Vec<u8>, PendingId)> {
        let data = self.frame(Kind::Sync, compress_body, &mut |out| codec::encode_call(syms, items, out))?;
        Ok((data, self.register(blocking, deadline)?))
    }

    /// Forget a request that could not be sent after all.
    pub fn forget(&mut self, id: PendingId) {
        for lvl in self.levels.iter_mut() {
            lvl.retain(|p| p.id != id);
        }
    }

    // ----- serving -----------------------------------------------------------------------

    /// The endpoint starts serving a sync or async request. Opens a deeper level when a
    /// blocking call is outstanding; [`NestingLimit`] when that would exceed `max_nesting`
    /// (answer with [`Session::refuse`] and do not serve).
    pub fn begin_service(&mut self, kind: Kind) -> core::result::Result<Service, NestingLimit> {
        let blocking = self.levels.iter().any(|l| l.iter().any(|p| p.blocking));
        if blocking {
            if self.levels.len() - 1 >= self.cfg.max_nesting || self.levels.try_reserve(1).is_err() {
                return Err(NestingLimit);
            }
            self.levels.push(VecDeque::new());
        }
        if self.services.try_reserve(1).is_err() {
            if blocking {
                self.levels.pop();
            }
            return Err(NestingLimit);
        }
        let serial = self.next_service;
        self.next_service = self.next_service.wrapping_add(1);
        self.services.push(serial);
        Ok(Service { kind, opened_level: blocking, answered: false, serial })
    }

    /// The endpoint has finished serving. A level that closes with requests unanswered is a
    /// protocol error: the returned `Failure` event carries them and the session is closed.
    pub fn end_service(&mut self, service: Service) -> Option<Event> {
        if self.state == State::Closed || self.services.last() != Some(&service.serial) {
            return None; // the connection went away while serving
        }
        self.services.pop();
        if service.opened_level && self.levels.len() > 1 {
            let open = self.levels.last().map(|l| l.len()).unwrap_or(0);
            if open > 0 {
                return Some(self.fail(String::from("requests unanswered when their level closed")));
            }
            self.levels.pop();
        }
        None
    }

    fn service_check(&self, service: &Service) -> Result<()> {
        self.require_ready()?;
        if self.services.last() != Some(&service.serial) || service.kind != Kind::Sync || service.answered {
            return Err(state_error("not the innermost unanswered sync request"));
        }
        Ok(())
    }

    /// Frame the response (type 2) to the sync request `service`.
    pub fn send_response(&mut self, service: &mut Service, syms: &Interner, value: &Value, compress_body: bool) -> Result<Vec<u8>> {
        self.service_check(service)?;
        let data = self.frame(Kind::Response, compress_body, &mut |out| codec::encode(syms, value, out))?;
        service.answered = true;
        Ok(data)
    }

    /// Frame an error response (`80`, text, `00`). NUL bytes in the text are dropped.
    pub fn send_error(&mut self, service: &mut Service, text: &str) -> Result<Vec<u8>> {
        self.service_check(service)?;
        let data = self.error_frame(text)?;
        service.answered = true;
        Ok(data)
    }

    fn error_frame(&self, text: &str) -> Result<Vec<u8>> {
        self.frame(Kind::Response, false, &mut |out| {
            out.try_reserve(text.len() + 2).map_err(|_| Error::memory())?;
            out.push(0x80);
            out.extend(text.bytes().filter(|&b| b != 0));
            out.push(0);
            Ok(())
        })
    }

    /// Answer a request that was not served because of the nesting limit: an error response
    /// (wire name `index`, a range error) for a sync request, nothing for an async one.
    pub fn refuse(&mut self, kind: Kind) -> Result<Vec<u8>> {
        if kind != Kind::Sync {
            return Ok(Vec::new());
        }
        self.error_frame(&klong_core::msg!("index: nested calls deeper than {}", self.cfg.max_nesting))
    }

    // ----- ending ------------------------------------------------------------------------

    fn drain_pending(&mut self) -> Vec<Pending> {
        let out = self.pending();
        self.levels.truncate(1);
        if let Some(l) = self.levels.first_mut() {
            l.clear();
        }
        self.services.clear();
        out
    }

    /// Close the session; the pending requests, which now fail.
    pub fn close(&mut self) -> Vec<Pending> {
        if self.state == State::Closed {
            return Vec::new();
        }
        let out = self.drain_pending();
        self.shut();
        out
    }

    /// Earliest deadline among the outstanding requests.
    pub fn next_deadline(&self) -> Option<u64> {
        self.levels.iter().flat_map(|l| l.iter()).filter_map(|p| p.deadline).min()
    }

    /// A request without an id cannot be skipped: when any deadline has passed the connection
    /// is closed and every pending request fails.
    pub fn check_deadlines(&mut self, now: u64) -> Option<Event> {
        if self.state == State::Closed {
            return None;
        }
        match self.next_deadline() {
            Some(d) if now >= d => Some(Event::Closed { reason: Reason::Timeout, pending: self.close() }),
            _ => None,
        }
    }
}
