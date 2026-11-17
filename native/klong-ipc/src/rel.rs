//! The packet reliability layer (`docs/native/index.md`): any link that carries
//! bounded packets to one peer, and may lose, duplicate, reorder or (after its own checksum)
//! drop them, becomes the reliable ordered byte stream the IPC session needs. One layer for
//! every non-stream network. No I/O and no clock: the owner passes packets and the time in.
//!
//! Packet: one header byte, then data.
//!
//! ```text
//! bits 7..6  type: 0 DATA, 1 ACK, 2 SYN, 3 RST
//! bits 5..0  sequence number (DATA), next expected sequence (ACK), protocol version (SYN), 0 (RST)
//! ```
//!
//! * SYN carries 3 bytes of a random session id. The client sends SYN, the server answers SYN
//!   with its own id, and both start DATA at sequence 0. A SYN with a different id on an
//!   established session means the peer restarted: the session fails (`PeerRestarted`) and the
//!   owner starts a new one.
//! * DATA carries up to `mtu - 1` stream bytes. Go-back-N, window 16, sequence modulo 64.
//! * ACK is cumulative: sent when a window of in-order packets has arrived, at once for a
//!   packet out of order, and otherwise `rto/4` after the first packet not yet acknowledged.
//! * A duplicate ACK (the peer saw a packet out of order) makes the sender go back at once,
//!   one time per position of the window, without waiting for the timeout.
//! * Retransmission: the timeout starts at `rto_ms`, doubles per retry; after `retries`
//!   retries (8) without progress the session fails (`Timeout`).
//! * RST closes, and answers anything but a SYN that arrives for no session.
//! * Keepalive: an empty DATA packet after `idle_ms` without traffic (0: never). It takes a
//!   sequence number and is acknowledged at once; three unanswered and the session fails.
//! * On a link that says it is reliable the layer still frames the stream and runs SYN and
//!   RST, but sends no ACK, never retransmits, and a gap is a fatal error.
//!
//! Limit of the 6-bit sequence number: a packet or an ACK that the network holds back until
//! 48 or more newer packets have gone by can be taken for a new one. A serial line cannot
//! reorder at all; a network that can delay a packet that far needs its own sequencing.
//!
//! Use: [`Rel::write`] stream bytes in, [`Rel::read`] delivered bytes out, [`Rel::on_packet`]
//! for every packet received, [`Rel::poll_packet`] until it returns `None` for the packets to
//! send (a packet the link is too busy to take must be offered to the link again, not
//! dropped: the layer has already counted it as sent), [`Rel::next_wake`] for when to call
//! again without input.
use alloc::collections::VecDeque;

pub const VERSION: u8 = 1;
pub const WINDOW: usize = 16;
const SEQ_MASK: u8 = 0x3f;
const T_DATA: u8 = 0;
const T_ACK: u8 = 1;
const T_SYN: u8 = 2;
const T_RST: u8 = 3;
/// A keepalive gives up sooner than data does.
const KEEPALIVE_TRIES: u8 = 3;

#[derive(Clone, Copy, Debug)]
pub struct Config {
    /// Bytes per packet including the header byte; at least 4.
    pub mtu: usize,
    /// The link never loses, duplicates or reorders.
    pub reliable: bool,
    /// First retransmission timeout, milliseconds (the link's `rto_hint_ms`).
    pub rto_ms: u32,
    /// Retransmissions without progress before the session fails; 0 is fail-fast.
    pub retries: u8,
    /// Send a keepalive after this long without traffic; 0 turns it off.
    pub idle_ms: u32,
    /// Stream bytes [`Rel::write`] accepts before it answers 0 (back-pressure).
    pub send_buffer: usize,
    /// Delivered bytes held for [`Rel::read`]; packets beyond it are dropped unacknowledged.
    pub recv_buffer: usize,
}

impl Config {
    pub fn new(mtu: usize, reliable: bool, rto_ms: u32) -> Config {
        let mtu = mtu.max(4);
        Config { mtu, reliable, rto_ms: rto_ms.max(1), retries: 8, idle_ms: 0, send_buffer: (mtu * WINDOW * 4).max(4096), recv_buffer: (mtu * WINDOW * 4).max(4096) }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Error {
    /// No acknowledgement within the retry budget.
    Timeout,
    /// The peer sent RST, or had no session for what we sent.
    Reset,
    /// A SYN with another session id arrived on an established session.
    PeerRestarted,
    /// A reliable link delivered a packet out of sequence, or the peer speaks another version.
    Protocol,
}

impl Error {
    pub fn text(self) -> &'static str {
        match self {
            Error::Timeout => "no answer from the peer within the retry budget",
            Error::Reset => "connection reset by the peer",
            Error::PeerRestarted => "the peer restarted",
            Error::Protocol => "packet out of sequence on a reliable link",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum State {
    /// Server waiting for a SYN.
    Listen,
    /// Client whose SYN is not answered yet.
    SynSent,
    Open,
    /// Closed by [`Rel::close`].
    Closed,
    Failed(Error),
}

pub struct Rel {
    cfg: Config,
    state: State,
    client: bool,
    id: [u8; 3],
    peer_id: [u8; 3],
    // ----- sending
    /// Stream bytes not yet acknowledged; the first `sent` packets of it are in flight.
    sndq: VecDeque<u8>,
    /// Sequence number of the oldest packet in flight (or of the next one to send).
    base_seq: u8,
    /// Length of each packet in flight, oldest first (a retransmission must repeat it: the
    /// receiver took the bytes of the first copy).
    sent_len: [u16; WINDOW],
    /// Packets in flight.
    sent: usize,
    /// The packets in flight from this index on have not been transmitted since the last
    /// timeout (go-back-N).
    next: usize,
    /// A keepalive (an empty packet) is among the packets in flight.
    keepalive_out: bool,
    /// A duplicate ACK already caused a go-back for the current oldest packet.
    fast: bool,
    rto_at: Option<u64>,
    rto_cur: u32,
    tries: u8,
    syn_due: bool,
    // ----- receiving
    rcvq: VecDeque<u8>,
    rcv_next: u8,
    /// In-order packets since the last ACK.
    unacked: usize,
    ack_now: bool,
    ack_at: Option<u64>,
    rst_due: bool,
    last_traffic: u64,
    /// Counters, for tests and measurements.
    pub retransmissions: u32,
    pub fast_retransmissions: u32,
    pub dropped: u32,
}

impl Rel {
    fn new(cfg: Config, client: bool, id: [u8; 3], now: u64) -> Rel {
        let rto = cfg.rto_ms;
        Rel {
            cfg,
            state: if client { State::SynSent } else { State::Listen },
            client,
            id,
            peer_id: [0; 3],
            sndq: VecDeque::new(),
            base_seq: 0,
            sent_len: [0; WINDOW],
            sent: 0,
            next: 0,
            keepalive_out: false,
            fast: false,
            rto_at: None,
            rto_cur: rto,
            tries: 0,
            syn_due: client,
            rcvq: VecDeque::new(),
            rcv_next: 0,
            unacked: 0,
            ack_now: false,
            ack_at: None,
            rst_due: false,
            last_traffic: now,
            retransmissions: 0,
            fast_retransmissions: 0,
            dropped: 0,
        }
    }

    /// The side that opens the session. `id` should be random (it tells a restart apart).
    pub fn client(cfg: Config, id: [u8; 3], now_ms: u64) -> Rel {
        Rel::new(cfg, true, id, now_ms)
    }

    /// The side that waits for a SYN.
    pub fn server(cfg: Config, id: [u8; 3], now_ms: u64) -> Rel {
        Rel::new(cfg, false, id, now_ms)
    }

    pub fn state(&self) -> State {
        self.state
    }

    pub fn is_open(&self) -> bool {
        self.state == State::Open
    }

    /// Closed or failed: nothing more will be delivered or sent (but a last RST).
    pub fn is_done(&self) -> bool {
        matches!(self.state, State::Closed | State::Failed(_))
    }

    pub fn error(&self) -> Option<Error> {
        match self.state {
            State::Failed(e) => Some(e),
            _ => None,
        }
    }

    fn payload(&self) -> usize {
        self.cfg.mtu - 1
    }

    fn fail(&mut self, e: Error) {
        if !self.is_done() {
            self.state = State::Failed(e);
            self.rto_at = None;
            self.ack_at = None;
            self.ack_now = false;
            self.syn_due = false;
        }
    }

    /// Queue stream bytes; the number accepted (0 when the send buffer is full or the
    /// session is over).
    pub fn write(&mut self, data: &[u8]) -> usize {
        if self.is_done() {
            return 0;
        }
        let room = self.cfg.send_buffer.saturating_sub(self.sndq.len());
        let n = room.min(data.len());
        if self.sndq.try_reserve(n).is_err() {
            return 0;
        }
        self.sndq.extend(&data[..n]);
        n
    }

    /// Stream bytes delivered in order.
    pub fn read(&mut self, buf: &mut [u8]) -> usize {
        let n = buf.len().min(self.rcvq.len());
        for (b, q) in buf.iter_mut().zip(self.rcvq.drain(..n)) {
            *b = q;
        }
        n
    }

    /// Bytes written and not yet acknowledged by the peer.
    pub fn unsent(&self) -> usize {
        self.sndq.len()
    }

    /// Close: an RST goes out with the next [`Rel::poll_packet`].
    pub fn close(&mut self) {
        if !self.is_done() {
            self.rst_due = self.state != State::Listen;
            self.state = State::Closed;
            self.rto_at = None;
            self.ack_at = None;
        }
    }

    /// Acknowledge what has arrived with the next packet instead of after the usual delay
    /// (the owner is about to be busy for a while).
    pub fn ack_soon(&mut self) {
        if self.state == State::Open && (self.unacked > 0 || self.ack_at.is_some()) {
            self.ack_now = true;
        }
    }

    fn sent_bytes(&self, upto: usize) -> usize {
        self.sent_len[..upto].iter().map(|&l| l as usize).sum()
    }

    /// A packet arrived.
    pub fn on_packet(&mut self, pkt: &[u8], now: u64) {
        let (head, data) = match pkt.split_first() {
            Some(x) => x,
            None => return,
        };
        let (kind, low) = (head >> 6, head & SEQ_MASK);
        if self.is_done() {
            return;
        }
        match kind {
            T_SYN => {
                if low != VERSION || data.len() < 3 {
                    if self.state != State::Open {
                        self.rst_due = true;
                    }
                    return;
                }
                let id = [data[0], data[1], data[2]];
                match self.state {
                    State::Listen => {
                        self.peer_id = id;
                        self.state = State::Open;
                        self.syn_due = true;
                        self.last_traffic = now;
                    }
                    State::SynSent => {
                        self.peer_id = id;
                        self.state = State::Open;
                        self.rto_at = None;
                        self.rto_cur = self.cfg.rto_ms;
                        self.tries = 0;
                        self.last_traffic = now;
                    }
                    State::Open => {
                        if id != self.peer_id {
                            self.fail(Error::PeerRestarted);
                        } else if !self.client {
                            // our answer was lost: say it again
                            self.syn_due = true;
                        }
                    }
                    _ => {}
                }
            }
            T_RST => {
                if self.state != State::Listen {
                    self.fail(Error::Reset);
                }
            }
            T_DATA => {
                if self.state != State::Open {
                    if self.state == State::Listen {
                        self.rst_due = true; // no session for this
                    }
                    return;
                }
                self.last_traffic = now;
                if low != self.rcv_next {
                    if self.cfg.reliable {
                        self.fail(Error::Protocol);
                    } else {
                        self.dropped += 1;
                        self.ack_now = true;
                    }
                    return;
                }
                if self.rcvq.len() + data.len() > self.cfg.recv_buffer.max(data.len()) || self.rcvq.try_reserve(data.len()).is_err() {
                    // no room: not acknowledged, so it comes again
                    self.dropped += 1;
                    return;
                }
                self.rcvq.extend(data);
                self.rcv_next = (self.rcv_next + 1) & SEQ_MASK;
                if self.cfg.reliable {
                    return;
                }
                self.unacked += 1;
                if self.unacked >= WINDOW || data.is_empty() {
                    self.ack_now = true;
                } else if self.ack_at.is_none() {
                    self.ack_at = Some(now + (self.cfg.rto_ms as u64 / 4).max(1));
                }
            }
            _ => {
                // ACK
                if self.state != State::Open {
                    if self.state == State::Listen {
                        self.rst_due = true;
                    }
                    return;
                }
                self.last_traffic = now;
                let advance = (low.wrapping_sub(self.base_seq) & SEQ_MASK) as usize;
                if advance == 0 && self.sent > 0 && !self.fast && !self.cfg.reliable {
                    // the peer saw a packet out of order: something before it was lost. Go
                    // back now instead of waiting for the timeout, once per position.
                    self.fast = true;
                    self.next = 0;
                    self.fast_retransmissions += 1;
                    return;
                }
                if advance == 0 || advance > self.sent {
                    return;
                }
                self.fast = false;
                let bytes = self.sent_bytes(advance);
                self.sndq.drain(..bytes.min(self.sndq.len()));
                self.sent_len.copy_within(advance..self.sent, 0);
                self.sent -= advance;
                self.next = self.next.saturating_sub(advance);
                self.base_seq = (self.base_seq + advance as u8) & SEQ_MASK;
                self.tries = 0;
                self.rto_cur = self.cfg.rto_ms;
                if self.sent == 0 {
                    self.keepalive_out = false;
                }
                self.rto_at = if self.sent > 0 { Some(now + self.rto_cur as u64) } else { None };
            }
        }
    }

    fn head(kind: u8, low: u8) -> u8 {
        (kind << 6) | (low & SEQ_MASK)
    }

    /// Run the timers: retransmission, keepalive.
    fn tick(&mut self, now: u64) {
        if let Some(at) = self.rto_at {
            if now >= at {
                let limit = if self.state == State::Open && self.keepalive_out && self.sent == 1 { KEEPALIVE_TRIES.min(self.cfg.retries) } else { self.cfg.retries };
                if self.tries >= limit {
                    self.fail(Error::Timeout);
                    return;
                }
                self.tries += 1;
                self.rto_cur = self.rto_cur.saturating_mul(2);
                self.rto_at = Some(now + self.rto_cur as u64);
                self.retransmissions += 1;
                if self.state == State::SynSent {
                    self.syn_due = true;
                } else {
                    self.next = 0; // go back
                    self.fast = false;
                }
            }
        }
        if self.state == State::Open && !self.cfg.reliable && self.cfg.idle_ms > 0 && self.sent == 0 && self.sndq.is_empty() && now.saturating_sub(self.last_traffic) >= self.cfg.idle_ms as u64 {
            // an empty packet in the sequence: the peer acknowledges it at once
            self.sent_len[0] = 0;
            self.sent = 1;
            self.next = 0;
            self.keepalive_out = true;
            self.last_traffic = now;
        }
    }

    /// The next packet to transmit, written to `out` (at least `mtu` bytes); its length.
    /// `None`: nothing to send now.
    pub fn poll_packet(&mut self, now: u64, out: &mut [u8]) -> Option<usize> {
        if out.len() < self.cfg.mtu {
            return None;
        }
        if self.rst_due {
            self.rst_due = false;
            out[0] = Self::head(T_RST, 0);
            return Some(1);
        }
        if self.is_done() {
            return None;
        }
        self.tick(now);
        if self.is_done() {
            return None;
        }
        if self.syn_due {
            self.syn_due = false;
            out[0] = Self::head(T_SYN, VERSION);
            out[1..4].copy_from_slice(&self.id);
            if self.state == State::SynSent && self.rto_at.is_none() {
                self.rto_at = Some(now + self.rto_cur as u64);
            }
            return Some(4);
        }
        if self.state != State::Open {
            return None;
        }
        if self.ack_now || self.ack_at.is_some_and(|at| now >= at) {
            self.ack_now = false;
            self.ack_at = None;
            self.unacked = 0;
            out[0] = Self::head(T_ACK, self.rcv_next);
            return Some(1);
        }
        // DATA: first what is in flight and due again, then new packets while the window has room
        if self.next >= self.sent {
            if self.sent >= WINDOW {
                return None;
            }
            let queued = self.sndq.len() - self.sent_bytes(self.sent);
            if queued == 0 {
                return None;
            }
            self.sent_len[self.sent] = queued.min(self.payload()) as u16;
            self.sent += 1;
        }
        let k = self.next;
        let start = self.sent_bytes(k);
        let len = self.sent_len[k] as usize;
        out[0] = Self::head(T_DATA, (self.base_seq + k as u8) & SEQ_MASK);
        for (o, b) in out[1..1 + len].iter_mut().zip(self.sndq.iter().skip(start)) {
            *o = *b;
        }
        self.next += 1;
        self.last_traffic = now;
        if self.cfg.reliable {
            // nothing is kept for a link that cannot lose it
            self.sndq.drain(..len);
            self.sent = 0;
            self.next = 0;
            self.base_seq = (self.base_seq + 1) & SEQ_MASK;
        } else if self.rto_at.is_none() {
            self.rto_at = Some(now + self.rto_cur as u64);
        }
        Some(1 + len)
    }

    /// When to call [`Rel::poll_packet`] again if nothing arrives, milliseconds.
    pub fn next_wake(&self) -> Option<u64> {
        if self.is_done() {
            return None;
        }
        let mut t = self.rto_at;
        for c in [self.ack_at, if self.state == State::Open && self.cfg.idle_ms > 0 && !self.cfg.reliable { Some(self.last_traffic + self.cfg.idle_ms as u64) } else { None }] {
            t = match (t, c) {
                (Some(a), Some(b)) => Some(a.min(b)),
                (a, b) => a.or(b),
            };
        }
        t
    }
}
