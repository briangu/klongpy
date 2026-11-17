//! A node on its lane: bytes in, bytes out, nothing else. It finds frames in the byte stream,
//! learns its node number from the first unicast frame it hears, answers PING with PONG,
//! sends STATUS to the supervisor once a second, ignores every other valid frame it does not
//! understand, and runs the datagram layer for the frames addressed to it.
//!
//! The owner calls [`Lane::rx`] for bytes that arrived and [`Lane::poll_tx`] whenever the
//! transmitter can take 16 bytes; neither blocks.
use crate::dgram::{Config, Datagrams, SendError};
use crate::frame::{is_fanout, Frame, Parser, FRAME, SUPERVISOR, T_APP_DATA, T_PING, T_PONG, T_STATUS};

#[derive(Default, Clone, Copy, Debug)]
pub struct LaneStats {
    pub frames_rx: u32,
    pub frames_tx: u32,
    /// Valid frames addressed to somebody else, or not understood.
    pub ignored: u32,
    pub pings: u32,
    pub ctl_dropped: u32,
}

/// What a node needs to be a citizen of the bench.
#[derive(Clone, Copy, Debug)]
pub struct LaneConfig {
    pub dgram: Config,
    /// Answer PING with PONG and send STATUS every second: what the bench supervisor expects,
    /// or it marks the node offline.
    pub supervisor_duties: bool,
    /// A fixed node number instead of learning one.
    pub node: Option<u8>,
}

impl Default for LaneConfig {
    fn default() -> LaneConfig {
        LaneConfig { dgram: Config { ack_base_ms: 300, ack_per_frame_us: 1500, resends: 3, first_id: 0 }, supervisor_duties: true, node: None }
    }
}

const CTL: usize = 4;
const NO_FRAME: Frame = Frame { ftype: 0, src: 0, dst: 0, seq: 0, sub: 0, len: 0, payload: [0; 4] };

pub struct Lane<const MAX: usize, const SLOTS: usize> {
    cfg: LaneConfig,
    parser: Parser,
    pub dg: Datagrams<MAX, SLOTS>,
    ctl: [Frame; CTL],
    ctl_n: usize,
    status_seq: u16,
    last_status: u64,
    started: u64,
    pub stats: LaneStats,
}

impl<const MAX: usize, const SLOTS: usize> Lane<MAX, SLOTS> {
    pub const fn new(cfg: LaneConfig) -> Self {
        let mut dg = Datagrams::new(cfg.dgram);
        if let Some(n) = cfg.node {
            dg.set_node(n);
        }
        Lane { cfg, parser: Parser::new(), dg, ctl: [NO_FRAME; CTL], ctl_n: 0, status_seq: 0, last_status: 0, started: 0, stats: LaneStats { frames_rx: 0, frames_tx: 0, ignored: 0, pings: 0, ctl_dropped: 0 } }
    }

    pub fn node(&self) -> Option<u8> {
        self.dg.node()
    }

    pub fn parser(&self) -> &Parser {
        &self.parser
    }

    /// Bytes from the wire. Completed messages go to `deliver(source, channel, bytes)`.
    pub fn rx(&mut self, bytes: &[u8], now_ms: u64, deliver: &mut dyn FnMut(u8, u8, &[u8])) {
        for &b in bytes {
            if let Some(f) = self.parser.push(b) {
                self.on_frame(&f, now_ms, deliver);
            }
        }
    }

    fn push_ctl(&mut self, f: Frame) {
        if self.ctl_n < CTL {
            self.ctl[self.ctl_n] = f;
            self.ctl_n += 1;
        } else {
            self.stats.ctl_dropped += 1;
        }
    }

    pub fn on_frame(&mut self, f: &Frame, now_ms: u64, deliver: &mut dyn FnMut(u8, u8, &[u8])) {
        self.stats.frames_rx += 1;
        let node = match self.dg.node() {
            Some(n) => n,
            None => {
                // only a unicast node address names this port; fan-out frames reach every port
                if f.dst >= SUPERVISOR {
                    self.stats.ignored += 1;
                    return;
                }
                self.dg.set_node(f.dst);
                self.started = now_ms;
                self.last_status = now_ms;
                f.dst
            }
        };
        if f.dst != node && !is_fanout(f.dst) {
            self.stats.ignored += 1;
            return;
        }
        match f.ftype {
            T_PING if self.cfg.supervisor_duties => {
                self.stats.pings += 1;
                let mut r = *f;
                r.ftype = T_PONG;
                r.src = node;
                r.dst = f.src;
                r.sub = 16;
                self.push_ctl(r);
            }
            T_APP_DATA if f.dst == node => self.dg.on_frame(f, now_ms, deliver),
            _ => self.stats.ignored += 1,
        }
    }

    /// Queue a message (see [`Datagrams::send`]).
    pub fn send(&mut self, dest: u8, channel: u8, data: &[u8], ack: bool) -> Result<u16, SendError> {
        self.dg.send(dest, channel, data, ack)
    }

    /// The next 16 bytes for the wire: PONG and STATUS, then ACKs, then the message in progress.
    pub fn poll_tx(&mut self, now_ms: u64) -> Option<[u8; FRAME]> {
        let node = self.dg.node()?;
        if self.cfg.supervisor_duties && now_ms.saturating_sub(self.last_status) >= 1000 && self.ctl_n < CTL {
            self.last_status = now_ms;
            let up = (now_ms.saturating_sub(self.started) / 1000) as u16;
            let f = Frame::new(T_STATUS, node, SUPERVISOR, self.status_seq, 16, &[0, 1, up as u8, (up >> 8) as u8]);
            self.status_seq = self.status_seq.wrapping_add(1);
            self.push_ctl(f);
        }
        let f = if self.ctl_n > 0 {
            let f = self.ctl[0];
            self.ctl.copy_within(1..self.ctl_n, 0);
            self.ctl_n -= 1;
            f
        } else {
            self.dg.poll_tx(now_ms)?
        };
        self.stats.frames_tx += 1;
        f.encode()
    }

    /// When the lane next has something to send without input, caller's milliseconds.
    pub fn next_wake(&self, now_ms: u64) -> Option<u64> {
        let status = if self.cfg.supervisor_duties && self.node().is_some() { Some(self.last_status + 1000) } else { None };
        match (status, self.dg.next_wake()) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        }
        .map(|t| t.max(now_ms))
    }
}
