//! Messages of 1 to 1024 bytes over 16-byte frames: START, DATA 0.., END, ACK (APP_DATA frames
//! with sub-types 0xD0..0xD3), best effort. No allocation: the sender owns one message buffer,
//! the receiver `SLOTS` reassembly buffers handed out to sources as they start a message.
//!
//! RAM, for `Datagrams<MAX, SLOTS>`: `(SLOTS + 1) * MAX` bytes of buffers, 24 bytes per
//! possible source (17) and about 650 bytes of queues and state. With the full 1024 bytes
//! and 17 slots that is 19 KB; with `SLOTS = 4` (four nodes and the supervisor never all
//! sending at once) it is 6 KB.
//!
//! Time is a number the caller passes in (milliseconds, any origin, never going back).
use crate::crc::{crc32, Crc32};
use crate::frame::{Frame, FPGA, SUPERVISOR, T_APP_DATA};

pub const SUB_START: u8 = 0xd0;
pub const SUB_DATA: u8 = 0xd1;
pub const SUB_END: u8 = 0xd2;
pub const SUB_ACK: u8 = 0xd3;
/// Longest message the format allows.
pub const MAX_MSG: usize = 1024;
/// Possible sources: the addresses 0..=16 (nodes 0..15 and the supervisor).
pub const SOURCES: usize = 17;
pub const FLAG_ACK: u8 = 1;
pub const ACK_DELIVERED: u8 = 0;
pub const ACK_BROKEN: u8 = 1;
pub const ACK_REFUSED: u8 = 2;

const NO_SLOT: u8 = 255;
const NO_ID: u32 = 1 << 16;
const ACK_QUEUE: usize = 8;
const NO_FRAME: Frame = Frame { ftype: 0, src: 0, dst: 0, seq: 0, sub: 0, len: 0, payload: [0; 4] };

#[derive(Clone, Copy, Debug)]
pub struct Config {
    /// Wait for an ACK this long after the END frame left, plus `ack_per_frame_us` per frame of the message.
    pub ack_base_ms: u32,
    pub ack_per_frame_us: u32,
    /// Resends of a whole message (same id) after a timeout or an ACK of 1.
    pub resends: u8,
    /// The id of the first message; set it from a clock or a random source so that a node
    /// that restarts does not reuse the ids its peers remember.
    pub first_id: u16,
}

impl Default for Config {
    fn default() -> Config {
        Config { ack_base_ms: 300, ack_per_frame_us: 1500, resends: 3, first_id: 0 }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Rx {
    Idle,
    Receiving,
    Broken,
    Refused,
}

#[derive(Clone, Copy)]
struct Meta {
    state: Rx,
    id: u16,
    len: u16,
    flags: u8,
    chan: u8,
    /// The fragment index expected next.
    next: u16,
    got: u16,
    slot: u8,
    /// Id of the last message delivered from this source (`NO_ID`: none).
    last: u32,
}

const META0: Meta = Meta { state: Rx::Idle, id: 0, len: 0, flags: 0, chan: 0, next: 0, got: 0, slot: NO_SLOT, last: NO_ID };

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SendError {
    /// The previous message is still being sent or waits for its ACK.
    Busy,
    /// 0 bytes, or more than the buffer (1024).
    Size,
    /// Own node number not known yet.
    NoAddress,
    /// Not a node (0..15) or the supervisor (16), or ourselves.
    BadDest,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Outcome {
    /// No ACK was asked for; the last frame left.
    Sent,
    /// Delivered (ACK 0).
    Acked,
    /// The receiver said it cannot take the message (ACK 2).
    Refused,
    /// No ACK after every resend, or it said broken every time.
    Failed,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct SendResult {
    pub id: u16,
    pub dest: u8,
    pub outcome: Outcome,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Tx {
    Idle,
    Sending,
    WaitAck,
}

#[derive(Default, Clone, Copy, Debug)]
pub struct Stats {
    pub delivered: u32,
    pub duplicates: u32,
    pub broken: u32,
    pub refused: u32,
    pub abandoned: u32,
    pub stray: u32,
    pub resent: u32,
    pub frames_out: u32,
}

pub struct Datagrams<const MAX: usize, const SLOTS: usize> {
    cfg: Config,
    own: Option<u8>,
    meta: [Meta; SOURCES],
    bufs: [[u8; MAX]; SLOTS],
    used: [bool; SLOTS],
    acks: [Frame; ACK_QUEUE],
    ack_n: usize,
    // sender
    tx: Tx,
    txbuf: [u8; MAX],
    tx_dest: u8,
    tx_chan: u8,
    tx_flags: u8,
    tx_id: u16,
    tx_len: usize,
    tx_frames: usize,
    tx_cursor: usize,
    tx_tries: u8,
    tx_crc: u32,
    tx_deadline: u64,
    next_id: u16,
    result: Option<SendResult>,
    pub stats: Stats,
}

impl<const MAX: usize, const SLOTS: usize> Datagrams<MAX, SLOTS> {
    pub const fn new(cfg: Config) -> Self {
        assert!(MAX >= 1 && MAX <= MAX_MSG);
        Datagrams {
            cfg,
            own: None,
            meta: [META0; SOURCES],
            bufs: [[0; MAX]; SLOTS],
            used: [false; SLOTS],
            acks: [NO_FRAME; ACK_QUEUE],
            ack_n: 0,
            tx: Tx::Idle,
            txbuf: [0; MAX],
            tx_dest: 0,
            tx_chan: 0,
            tx_flags: 0,
            tx_id: 0,
            tx_len: 0,
            tx_frames: 0,
            tx_cursor: 0,
            tx_tries: 0,
            tx_crc: 0,
            tx_deadline: 0,
            next_id: cfg.first_id,
            result: None,
            stats: Stats { delivered: 0, duplicates: 0, broken: 0, refused: 0, abandoned: 0, stray: 0, resent: 0, frames_out: 0 },
        }
    }

    /// Own node number (the source of every frame sent).
    pub const fn set_node(&mut self, node: u8) {
        self.own = Some(node);
    }
    pub fn node(&self) -> Option<u8> {
        self.own
    }
    pub fn max_message(&self) -> usize {
        MAX
    }

    // ----- sending ------------------------------------------------------------------------

    /// Start sending `data` to `dest` on `channel`. Frames are produced by [`Datagrams::poll_tx`].
    /// The message is copied; `ack` asks the receiver to confirm. Returns the message id.
    pub fn send(&mut self, dest: u8, channel: u8, data: &[u8], ack: bool) -> Result<u16, SendError> {
        let own = self.own.ok_or(SendError::NoAddress)?;
        if self.tx != Tx::Idle {
            return Err(SendError::Busy);
        }
        if data.is_empty() || data.len() > MAX {
            return Err(SendError::Size);
        }
        if dest > SUPERVISOR || dest == own {
            return Err(SendError::BadDest);
        }
        self.txbuf[..data.len()].copy_from_slice(data);
        self.tx_len = data.len();
        self.tx_frames = data.len().div_ceil(4) + 2;
        self.tx_dest = dest;
        self.tx_chan = channel;
        self.tx_flags = if ack { FLAG_ACK } else { 0 };
        self.tx_id = self.next_id;
        self.next_id = self.next_id.wrapping_add(1);
        self.tx_crc = crc32(data);
        self.tx_cursor = 0;
        self.tx_tries = 0;
        self.tx = Tx::Sending;
        self.result = None;
        Ok(self.tx_id)
    }

    /// True while a message is being sent or waits for its ACK.
    pub fn tx_busy(&self) -> bool {
        self.tx != Tx::Idle
    }

    /// What became of the last message, once, when it is over.
    pub fn take_result(&mut self) -> Option<SendResult> {
        self.result.take()
    }

    fn finish(&mut self, outcome: Outcome) {
        self.tx = Tx::Idle;
        self.result = Some(SendResult { id: self.tx_id, dest: self.tx_dest, outcome });
    }

    fn tx_frame(&self, k: usize) -> Frame {
        let own = self.own.unwrap_or(0);
        if k == 0 {
            let l = self.tx_len as u16;
            Frame::new(T_APP_DATA, own, self.tx_dest, self.tx_id, SUB_START, &[l as u8, (l >> 8) as u8, self.tx_flags, self.tx_chan])
        } else if k + 1 == self.tx_frames {
            Frame::new(T_APP_DATA, own, self.tx_dest, self.tx_id, SUB_END, &self.tx_crc.to_le_bytes())
        } else {
            let i = k - 1;
            let a = i * 4;
            let b = (a + 4).min(self.tx_len);
            Frame::new(T_APP_DATA, own, self.tx_dest, i as u16, SUB_DATA, &self.txbuf[a..b])
        }
    }

    fn restart(&mut self) -> bool {
        if self.tx_tries >= self.cfg.resends {
            return false;
        }
        self.tx_tries += 1;
        self.stats.resent += 1;
        self.tx_cursor = 0;
        self.tx = Tx::Sending;
        true
    }

    /// The next frame to put on the wire, if any: pending ACKs first, then the message.
    pub fn poll_tx(&mut self, now_ms: u64) -> Option<Frame> {
        if self.ack_n > 0 {
            let f = self.acks[0];
            self.acks.copy_within(1..self.ack_n, 0);
            self.ack_n -= 1;
            self.stats.frames_out += 1;
            return Some(f);
        }
        if self.tx == Tx::WaitAck && now_ms >= self.tx_deadline && !self.restart() {
            self.finish(Outcome::Failed);
        }
        if self.tx != Tx::Sending {
            return None;
        }
        let f = self.tx_frame(self.tx_cursor);
        self.tx_cursor += 1;
        self.stats.frames_out += 1;
        if self.tx_cursor == self.tx_frames {
            if self.tx_flags & FLAG_ACK != 0 {
                self.tx = Tx::WaitAck;
                self.tx_deadline = now_ms + self.cfg.ack_base_ms as u64 + (self.tx_frames as u64 * self.cfg.ack_per_frame_us as u64).div_ceil(1000);
            } else {
                self.finish(Outcome::Sent);
            }
        }
        Some(f)
    }

    /// When [`Datagrams::poll_tx`] next has something without input: an ACK timeout.
    pub fn next_wake(&self) -> Option<u64> {
        if self.tx == Tx::WaitAck {
            Some(self.tx_deadline)
        } else {
            None
        }
    }

    // ----- receiving ----------------------------------------------------------------------

    fn queue_ack(&mut self, to: u8, id: u16, status: u8) {
        let Some(own) = self.own else { return };
        if self.ack_n < ACK_QUEUE {
            self.acks[self.ack_n] = Frame::new(T_APP_DATA, own, to, id, SUB_ACK, &[status]);
            self.ack_n += 1;
        } // else dropped: the sender times out and resends, and the duplicate rule re-ACKs
    }

    fn release(&mut self, s: usize) {
        let m = &mut self.meta[s];
        if m.slot != NO_SLOT {
            self.used[m.slot as usize] = false;
            m.slot = NO_SLOT;
        }
        m.state = Rx::Idle;
    }

    /// Process a frame addressed to this node. Frames that are not datagram frames are
    /// ignored. A completed message is handed to `deliver(source, channel, bytes)` once, and
    /// before this returns.
    pub fn on_frame(&mut self, f: &Frame, _now_ms: u64, deliver: &mut dyn FnMut(u8, u8, &[u8])) {
        if f.ftype != T_APP_DATA || f.src > SUPERVISOR || f.src == FPGA {
            return;
        }
        let s = f.src as usize;
        match f.sub {
            SUB_START if f.len == 4 => self.rx_start(s, f),
            SUB_DATA if (1..=4).contains(&f.len) => self.rx_data(s, f),
            SUB_END if f.len == 4 => self.rx_end(s, f, deliver),
            SUB_ACK if f.len == 1 => self.rx_ack(f),
            _ => {}
        }
    }

    fn rx_start(&mut self, s: usize, f: &Frame) {
        if self.meta[s].state == Rx::Receiving || self.meta[s].state == Rx::Broken {
            self.stats.abandoned += 1;
        }
        self.release(s);
        let len = f.payload[0] as u16 | (f.payload[1] as u16) << 8;
        let m = &mut self.meta[s];
        m.id = f.seq;
        m.len = len;
        m.flags = f.payload[2];
        m.chan = f.payload[3];
        m.next = 0;
        m.got = 0;
        if len == 0 || len as usize > MAX {
            m.state = Rx::Refused;
            return;
        }
        match self.used.iter().position(|u| !u) {
            Some(slot) => {
                self.used[slot] = true;
                let m = &mut self.meta[s];
                m.slot = slot as u8;
                m.state = Rx::Receiving;
            }
            None => self.meta[s].state = Rx::Refused,
        }
    }

    fn rx_data(&mut self, s: usize, f: &Frame) {
        let m = &mut self.meta[s];
        match m.state {
            Rx::Receiving => {}
            Rx::Idle => {
                self.stats.stray += 1;
                return;
            }
            _ => return,
        }
        let off = f.seq as usize * 4;
        let n = f.len as usize;
        if f.seq != m.next || off != m.got as usize || off + n > m.len as usize {
            m.state = Rx::Broken;
            let slot = m.slot;
            m.slot = NO_SLOT;
            if slot != NO_SLOT {
                self.used[slot as usize] = false;
            }
            return;
        }
        let slot = m.slot as usize;
        self.bufs[slot][off..off + n].copy_from_slice(f.data());
        m.got += n as u16;
        m.next += 1;
    }

    fn rx_end(&mut self, s: usize, f: &Frame, deliver: &mut dyn FnMut(u8, u8, &[u8])) {
        let m = self.meta[s];
        if m.state == Rx::Idle || m.id != f.seq {
            self.stats.stray += 1;
            return;
        }
        let want = u32::from_le_bytes(f.payload);
        let ack = m.flags & FLAG_ACK != 0;
        let status = match m.state {
            Rx::Refused => {
                self.stats.refused += 1;
                ACK_REFUSED
            }
            Rx::Broken => {
                self.stats.broken += 1;
                ACK_BROKEN
            }
            _ => {
                let len = m.len as usize;
                let buf = &self.bufs[m.slot as usize][..len];
                let mut c = Crc32::new();
                c.update(buf);
                if m.got != m.len || c.finish() != want {
                    self.stats.broken += 1;
                    ACK_BROKEN
                } else if m.last == m.id as u32 {
                    self.stats.duplicates += 1;
                    ACK_DELIVERED
                } else {
                    self.meta[s].last = m.id as u32;
                    self.stats.delivered += 1;
                    deliver(s as u8, m.chan, buf);
                    ACK_DELIVERED
                }
            }
        };
        self.release(s);
        if ack {
            self.queue_ack(s as u8, f.seq, status);
        }
    }

    fn rx_ack(&mut self, f: &Frame) {
        if self.tx != Tx::WaitAck || f.src != self.tx_dest || f.seq != self.tx_id {
            return;
        }
        match f.payload[0] {
            ACK_DELIVERED => self.finish(Outcome::Acked),
            ACK_REFUSED => self.finish(Outcome::Refused),
            _ => {
                if !self.restart() {
                    self.finish(Outcome::Failed);
                }
            }
        }
    }

    /// Bytes of a message of source `src` collected so far (0 when none is being received).
    pub fn receiving(&self, src: u8) -> usize {
        let m = &self.meta[(src as usize).min(SOURCES - 1)];
        if m.state == Rx::Receiving {
            m.got as usize
        } else {
            0
        }
    }
}
