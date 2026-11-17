//! IPC over a message-oriented link: the nodes of a packet switch that carries 16-byte frames
//! (the Pico bench), through the datagram layer of `klong-dgram`.
//!
//! The adapter is a [`Transport`] named `node`. `.cli("node:2")` opens a connection to node 2,
//! `.srv("node:")` accepts connections from any node. Each connection is a [`PacketLink`] of
//! 255-byte packets (a datagram of 256 bytes is 66 frames, 92 ms on a 115200 baud lane) under
//! the same reliability layer (`klong_ipc::rel`) that runs over a serial line: it provides the
//! handshake, the retransmission, the reset and the fragmentation, because it is a byte stream
//! that cuts what it is given into packets. An IPC message of any size therefore crosses the
//! link, and the kdb+ bytes inside are unchanged.
//!
//! Two bytes in front of every packet, a little-endian u16: bit 0 says which end sent it (1:
//! the client) and the other 15 bits are the connection's epoch, chosen by the client when it
//! connects and repeated by the server. So two nodes can be client and server of each other at
//! once, a packet of a connection that is gone (a stale reset, a late retransmission) is not
//! taken for one of the new connection, and a client that restarts and connects again shows up
//! at the server as a new epoch, which replaces the old connection. A node has at most one
//! connection of each kind to a given peer. Channel 0 of the datagram layer carries IPC; channel 1 carries
//! the text console (see [`NodeNet::console_read`]); other channels are dropped.
//!
//! Nothing here blocks. The owner of the lane supplies a [`LaneIo`] (a UART, or a simulated
//! one) and a clock, and the adapter moves frames whenever one of its functions is called; a
//! platform whose main loop does not call into IPC should call [`NodeNet::pump`] from its
//! `idle`, so that PING and ACK frames are answered while Klong code runs.
use alloc::boxed::Box;
use alloc::collections::VecDeque;
use alloc::rc::Rc;
use alloc::string::String;
use alloc::vec::Vec;
use core::cell::RefCell;

pub use klong_dgram::LaneConfig;
use klong_dgram::{Lane, SendError};

use crate::link::{Link, LinkError, LinkTuning, ListenerId, PacketLink, Transport};

pub use klong_dgram::lane::LaneStats;
pub use klong_dgram::Config as DgramConfig;

/// Datagram channels.
pub const CH_IPC: u8 = 0;
pub const CH_CONSOLE: u8 = 1;
/// Console control, ASCII, one message per datagram: `done ID STATUS`, `int`, `hello` and its reply.
pub const CH_CONTROL: u8 = 2;
/// The boot self-test's report.
pub const CH_SELFTEST: u8 = 3;
/// Bytes in a packet of a link by default, including the reliability layer's header byte (the
/// adapter adds two of its own, so a datagram is 128 bytes: 34 frames, 47 ms on a 115200 baud lane).
pub const LINK_MTU: usize = 126;
const HEAD: usize = 2;
/// Bytes read from the lane per pump.
const RX_PER_PUMP: usize = 8192;
/// Reassembly buffers: sources that may be in the middle of a message at the same moment.
pub const RX_SLOTS: usize = 6;
const MSG_MAX: usize = klong_dgram::MAX_MSG;
const REL_SYN: u8 = 2;
const REL_RST: u8 = 3;

/// The lane's UART, from the adapter's side. Neither call waits.
pub trait LaneIo {
    /// Bytes that have arrived.
    fn read(&mut self, buf: &mut [u8]) -> usize;
    /// Bytes the transmitter took (may be fewer than given, 0 when it is full).
    fn write(&mut self, buf: &[u8]) -> usize;
    /// What the UART driver counted (it lives below this layer).
    fn counters(&self) -> UartCounters {
        UartCounters::default()
    }
}

/// Losses below the frame layer, counted by the driver.
#[derive(Default, Clone, Copy, Debug, PartialEq, Eq)]
pub struct UartCounters {
    /// Bytes thrown away because the receive ring was full (the main loop was away too long).
    pub rx_ring_overruns: u32,
    /// Bytes lost in the hardware FIFO (the interrupt was late).
    pub rx_fifo_overruns: u32,
    /// Framing, break and parity errors.
    pub line_errors: u32,
    /// Times a frame could not be queued for transmission because the transmit ring was full.
    pub tx_ring_full: u32,
}

pub struct NodeConfig {
    pub lane: LaneConfig,
    /// First retransmission timeout of a connection, milliseconds (the ACK delay is a quarter of it).
    pub rto_ms: u32,
    /// Handshake and retry budget of a connection (see [`LinkTuning`]).
    pub handshake_timeout_us: u64,
    pub retries: u8,
    /// Messages one connection may hold unread; more are dropped (the peer retransmits).
    pub inbox_messages: usize,
    /// Bytes per packet of a connection, the reliability layer's header included. A lost frame
    /// costs one packet and a retransmission timeout, so smaller packets suit a lossier lane.
    pub link_mtu: usize,
    /// Seeds the connection epochs; give it entropy (a restart must not reuse the old ones).
    pub epoch_seed: u16,
    /// Console output kept for a peer that reads slowly.
    pub console_out_bytes: usize,
    /// Console input (datagrams not yet evaluated) the node keeps, bytes; more is dropped and counted.
    pub console_in_bytes: usize,
    /// Whether this node has a console: channel 1 is input and channel 2 is control. A supervisor's
    /// NodeNet has none and only logs what arrives (`keep_log`).
    pub console: bool,
    /// Named in the reply to `hello`: `klong VERSION node=N TARGET int64 float64`.
    pub target: String,
}

impl NodeConfig {
    /// Packet size, timeouts and retry budget for a lane of `baud` bits a second.
    ///
    /// * 115200 baud (a frame takes 1.4 ms): 126-byte packets (34 frames, 47 ms), first
    ///   timeout 500 ms, 6 retries (31 s), handshake 10 s.
    /// * 4.5 Mbaud (36 us a frame): the lane is 39 times faster and the endpoints are the
    ///   limit, so what matters is how long a peer's main loop may be away (an evaluation
    ///   between two `idle` calls, tens of milliseconds): first timeout 100 ms, 7 retries
    ///   (about 25 s in all), handshake 3 s; the packet stays 126 bytes (34 frames, 1.2 ms):
    ///   larger ones were no faster without loss (48 KB in 440 to 470 ms for every size from
    ///   126 to 1022) and much slower with it (0.2% of frames lost: 7 s for 126, 16 s for 1022).
    ///
    /// Rates in between get the nearer set.
    pub fn for_baud(baud: u32) -> NodeConfig {
        let mut c = NodeConfig::default();
        if baud >= 1_000_000 {
            c.rto_ms = 100;
            c.retries = 7;
            c.handshake_timeout_us = 3_000_000;
        }
        c
    }
}

impl Default for NodeConfig {
    fn default() -> NodeConfig {
        NodeConfig { lane: LaneConfig::default(), rto_ms: 500, handshake_timeout_us: 10_000_000, retries: 6, inbox_messages: 24, link_mtu: LINK_MTU, epoch_seed: 0, console_out_bytes: 16 * 1024, console_in_bytes: 8 * 1024, console: true, target: String::from("native") }
    }
}

#[derive(Default, Clone, Copy, Debug)]
pub struct NodeCounters {
    pub messages_in: u32,
    pub messages_out: u32,
    pub inbox_overflow: u32,
    pub refused_connects: u32,
    pub restarts: u32,
    pub console_dropped: u32,
    /// Console output that had no peer to go to (a timer's print, say), bytes.
    pub console_no_peer: u32,
    /// Console input dropped for lack of room, datagrams.
    pub console_in_dropped: u32,
    /// Channel-1 datagrams without the ACK flag (another console's output, not input), dropped.
    pub console_unsolicited: u32,
}

/// One input datagram on its way through the console.
struct Rec {
    id: u16,
    src: u8,
    data: Vec<u8>,
    pos: usize,
    /// The console has read some of it.
    started: bool,
    interrupted: bool,
    errors: u32,
}

struct Slot {
    id: u32,
    peer: u8,
    client: bool,
    epoch: u16,
    inbox: VecDeque<Vec<u8>>,
    closed: bool,
}

struct Core {
    cfg: NodeConfig,
    io: Box<dyn LaneIo>,
    clock: Box<dyn Fn() -> u64>,
    lane: Box<Lane<MSG_MAX, RX_SLOTS>>,
    pend: [u8; 16],
    pend_off: usize,
    pend_len: usize,
    slots: Vec<Slot>,
    next_id: u32,
    epoch_next: u16,
    listening: bool,
    accept: VecDeque<u32>,
    /// Input datagrams not yet evaluated, in order; the first is the one being evaluated.
    recs: VecDeque<Rec>,
    in_bytes: usize,
    /// An interrupt (0x03) for the console to read before anything else.
    ctrl: bool,
    last_errors: u32,
    last_interrupts: u32,
    /// Control messages for peers: (destination, text), sent after the console output.
    ctl_out: VecDeque<(u8, Vec<u8>)>,
    cpeer: Option<u8>,
    /// Console output not sent yet, by the peer it is for (a new sender takes the output from then on,
    /// not what was written before).
    cout: VecDeque<(u8, Vec<u8>)>,
    cout_bytes: usize,
    log: Option<(usize, VecDeque<(u8, u8, Vec<u8>)>)>,
    counters: NodeCounters,
    /// Resets that found the sender busy: (peer, header); they go out first.
    rst_out: Vec<(u8, u16)>,
}

impl Core {
    fn now_ms(&self) -> u64 {
        (self.clock)() / 1000
    }

    fn slot_mut(&mut self, id: u32) -> Option<&mut Slot> {
        self.slots.iter_mut().find(|s| s.id == id)
    }

    fn find(&self, peer: u8, client: bool) -> Option<usize> {
        self.slots.iter().position(|s| s.peer == peer && s.client == client && !s.closed)
    }

    fn open(&mut self, peer: u8, client: bool, epoch: u16) -> u32 {
        let id = self.next_id;
        self.next_id = self.next_id.wrapping_add(1);
        self.slots.push(Slot { id, peer, client, epoch, inbox: VecDeque::new(), closed: false });
        id
    }

    fn new_epoch(&mut self) -> u16 {
        self.epoch_next = self.epoch_next.wrapping_mul(25173).wrapping_add(13849);
        self.epoch_next & 0x7fff
    }

    /// Best effort, no retry: a refusal or a reset for a session that is gone. Sent by the
    /// server end, in the epoch of the packet that is being answered.
    fn send_rst(&mut self, peer: u8, epoch: u16) {
        let h = epoch << 1;
        let _ = self.lane.send(peer, CH_IPC, &[h as u8, (h >> 8) as u8, REL_RST << 6], false);
    }

    fn dispatch(&mut self, src: u8, chan: u8, id: u16, flags: u8, msg: Vec<u8>) {
        if chan != CH_IPC {
            if let Some((cap, log)) = self.log.as_mut() {
                if log.len() >= *cap {
                    log.pop_front();
                }
                log.push_back((src, chan, msg.clone()));
            }
        }
        if chan == CH_IPC {
            return self.dispatch_ipc(src, msg);
        }
        if !self.cfg.console {
            return;
        }
        match chan {
            CH_CONSOLE => {
                if flags & klong_dgram::dgram::FLAG_ACK == 0 {
                    // input asks for an ACK; this is the output of another console, which is not typed into this one
                    self.counters.console_unsolicited += 1;
                    return;
                }
                if self.in_bytes + msg.len() > self.cfg.console_in_bytes {
                    self.counters.console_in_dropped += 1;
                    return;
                }
                self.cpeer = Some(src);
                self.in_bytes += msg.len();
                self.recs.push_back(Rec { id, src, data: msg, pos: 0, started: false, interrupted: false, errors: 0 });
            }
            CH_CONTROL => match &msg[..] {
                b"int" => self.interrupt(src),
                b"hello" => {
                    let text = self.hello_text();
                    self.ctl_out.push_back((src, text.into_bytes()));
                }
                _ => {}
            },
            _ => {}
        }
    }

    fn hello_text(&self) -> alloc::string::String {
        let node = self.lane.node().map_or(alloc::string::String::from("?"), |n| alloc::format!("{}", n));
        alloc::format!("klong {} node={} {} int{} float{}", env!("CARGO_PKG_VERSION"), node, self.cfg.target.as_str(), klong_core::INT_BITS, klong_core::FLT_BITS)
    }

    /// `int` from the console's peer: if an evaluation of its input is running, end it by the serial
    /// console's Ctrl-C rule (a 0x03 the console reads; the input received before it is discarded,
    /// here the rest of this datagram and every datagram behind it, each of which is told `interrupted`).
    fn interrupt(&mut self, src: u8) {
        if self.cpeer != Some(src) || !self.recs.front().is_some_and(|r| r.started) {
            return;
        }
        let mut keep = VecDeque::new();
        let mut rest: VecDeque<Rec> = core::mem::take(&mut self.recs);
        if let Some(mut f) = rest.pop_front() {
            // stays until the end of the turn, so that `done` follows its output
            f.interrupted = true;
            f.pos = f.data.len();
            keep.push_back(f);
        }
        for r in rest {
            self.in_bytes = self.in_bytes.saturating_sub(r.data.len());
            self.ctl_out.push_back((r.src, alloc::format!("done {} interrupted", r.id).into_bytes()));
        }
        self.recs = keep;
        self.ctrl = true;
    }

    fn dispatch_ipc(&mut self, src: u8, msg: Vec<u8>) {
        if msg.len() < HEAD + 1 {
            return;
        }
        self.counters.messages_in += 1;
        let h = msg[0] as u16 | (msg[1] as u16) << 8;
        let (from_client, epoch) = (h & 1 != 0, h >> 1);
        let head = msg[HEAD];
        // a client end gets what the server end sends it, and the other way round
        let mut idx = self.find(src, !from_client);
        if let Some(i) = idx {
            if self.slots[i].epoch != epoch {
                if from_client && head >> 6 == REL_SYN {
                    // the peer restarted and connects again: the old connection is over
                    self.slots[i].closed = true;
                    self.counters.restarts += 1;
                    idx = None;
                } else {
                    // a packet of another epoch that is not the start of a connection: a late packet of a
                    // connection that is gone (its goodbye can leave after the first packet of the next
                    // one), which must not touch this one
                    return;
                }
            }
        }
        if idx.is_none() {
            if !from_client || head >> 6 == REL_RST {
                return; // an answer to a connection we no longer have, or a reset for nothing
            }
            if !self.listening {
                self.counters.refused_connects += 1;
                self.send_rst(src, epoch);
                return;
            }
            let id = self.open(src, false, epoch);
            self.accept.push_back(id);
            idx = self.find(src, false);
        }
        let Some(i) = idx else { return };
        let limit = self.cfg.inbox_messages;
        let s = &mut self.slots[i];
        if s.inbox.len() >= limit {
            self.counters.inbox_overflow += 1;
            return;
        }
        s.inbox.push_back(msg);
    }

    fn pump(&mut self) {
        let now = self.now_ms();
        let mut buf = [0u8; 64];
        let mut got: Vec<(u8, u8, u16, u8, Vec<u8>)> = Vec::new();
        // at most RX_PER_PUMP bytes (512 frames) per call, so that sending and the callers' own work
        // are not starved by a ring that holds many frames; the rest waits for the next call
        let mut taken = 0;
        while taken < RX_PER_PUMP {
            let n = self.io.read(&mut buf).min(buf.len());
            if n == 0 {
                break;
            }
            taken += n;
            self.lane.rx_id(&buf[..n], now, &mut |s, c, id, f, b| got.push((s, c, id, f, b.to_vec())));
        }
        for (s, c, id, f, m) in got {
            self.dispatch(s, c, id, f, m);
        }
        if !self.rst_out.is_empty() && !self.lane.dg.tx_busy() {
            let (peer, h) = self.rst_out.remove(0);
            let _ = self.lane.send(peer, CH_IPC, &[h as u8, (h >> 8) as u8, REL_RST << 6], false);
        }
        // the console's output first, then the control messages that follow it, one datagram at a time
        if !self.lane.dg.tx_busy() {
            match self.cout.front_mut() {
                Some((peer, bytes)) if !bytes.is_empty() => {
                    // up to 1024 bytes, cut at a line end when there is one
                    let peer = *peer;
                    let mut n = bytes.len().min(MSG_MAX);
                    if n == MSG_MAX && bytes.len() > MSG_MAX {
                        if let Some(p) = bytes[..n].iter().rposition(|&b| b == b'\n') {
                            n = p + 1;
                        }
                    }
                    let chunk: Vec<u8> = bytes.drain(..n).collect();
                    self.cout_bytes = self.cout_bytes.saturating_sub(n);
                    if bytes.is_empty() {
                        self.cout.pop_front();
                    }
                    let _ = self.lane.send(peer, CH_CONSOLE, &chunk, false);
                }
                _ => {
                    if let Some((dest, text)) = self.ctl_out.pop_front() {
                        let _ = self.lane.send(dest, CH_CONTROL, &text, false);
                    }
                }
            }
        }
        // 100 while the console has input in hand, 0 when it is idle
        self.lane.set_load(if self.recs.is_empty() { 0 } else { 100 });
        if self.recs.is_empty() {
            self.lane.dg.release_gate();
        }
        loop {
            if self.pend_off < self.pend_len {
                let n = self.io.write(&self.pend[self.pend_off..self.pend_len]);
                self.pend_off += n.min(self.pend_len - self.pend_off);
                if self.pend_off < self.pend_len {
                    break;
                }
            }
            match self.lane.poll_tx(now) {
                Some(b) => {
                    self.pend = b;
                    self.pend_off = 0;
                    self.pend_len = 16;
                }
                None => break,
            }
        }
    }
}

/// A lane with datagrams, IPC connections and the console channel on it. Clones share it.
#[derive(Clone)]
pub struct NodeNet(Rc<RefCell<Core>>);

impl NodeNet {
    /// `clock` is monotonic microseconds.
    pub fn new(io: Box<dyn LaneIo>, clock: Box<dyn Fn() -> u64>, cfg: NodeConfig) -> NodeNet {
        let mut lane = Box::new(Lane::new(cfg.lane));
        if cfg.console {
            lane.dg.set_gate(CH_CONSOLE);
        }
        let cfg_seed = cfg.epoch_seed;
        NodeNet(Rc::new(RefCell::new(Core {
            cfg,
            io,
            clock,
            lane,
            pend: [0; 16],
            pend_off: 0,
            pend_len: 0,
            slots: Vec::new(),
            next_id: 1,
            epoch_next: cfg_seed,
            listening: false,
            accept: VecDeque::new(),
            recs: VecDeque::new(),
            in_bytes: 0,
            ctrl: false,
            last_errors: 0,
            last_interrupts: 0,
            ctl_out: VecDeque::new(),
            cpeer: None,
            cout: VecDeque::new(),
            cout_bytes: 0,
            log: None,
            counters: NodeCounters::default(),
            rst_out: Vec::new(),
        })))
    }

    /// The transport to put in `Platform::transports` (named `node`).
    pub fn transport(&self) -> Box<dyn Transport> {
        Box::new(NodeTransport(self.clone()))
    }

    /// Move frames now: read the lane, answer, send what is queued. Safe to call often; a
    /// call while the adapter is already busy (from inside one of its own callbacks) does nothing.
    pub fn pump(&self) {
        if let Ok(mut c) = self.0.try_borrow_mut() {
            c.pump();
        }
    }

    /// This node's number, once it knows it.
    pub fn node(&self) -> Option<u8> {
        self.0.try_borrow().ok().and_then(|c| c.lane.node())
    }

    /// How long a connection waits for the answer to its hello, microseconds (the connections made from now on).
    pub fn handshake_timeout_us(&self) -> u64 {
        self.0.try_borrow().map(|c| c.cfg.handshake_timeout_us).unwrap_or(0)
    }
    pub fn set_handshake_timeout_us(&self, us: u64) {
        if let Ok(mut c) = self.0.try_borrow_mut() {
            c.cfg.handshake_timeout_us = us;
        }
    }

    /// Keep the last `n` messages that arrive on channels other than IPC (what a supervisor
    /// reads: the self-test's report, the console's text), with their sources.
    pub fn keep_log(&self, n: usize) {
        if let Ok(mut c) = self.0.try_borrow_mut() {
            c.log = Some((n, VecDeque::new()));
        }
    }

    /// The messages kept by [`NodeNet::keep_log`] since the last call.
    pub fn take_log(&self) -> Vec<(u8, u8, Vec<u8>)> {
        match self.0.try_borrow_mut() {
            Ok(mut c) => c.log.as_mut().map(|(_, l)| l.drain(..).collect()).unwrap_or_default(),
            Err(_) => Vec::new(),
        }
    }

    /// The flags byte of the STATUS frames (see `klong_dgram::lane::STATUS_*`).
    pub fn set_status_flags(&self, flags: u8) {
        if let Ok(mut c) = self.0.try_borrow_mut() {
            c.lane.set_status_flags(flags);
        }
    }

    /// The UART driver's counters (receive overruns and the like).
    pub fn uart_counters(&self) -> UartCounters {
        self.0.try_borrow().map(|c| c.io.counters()).unwrap_or_default()
    }

    /// Connections that are not closed (client and server ends): for tests and diagnostics.
    pub fn open_links(&self) -> usize {
        self.0.try_borrow().map(|c| c.slots.iter().filter(|s| !s.closed).count()).unwrap_or(0)
    }

    pub fn lane_stats(&self) -> LaneStats {
        self.0.try_borrow().map(|c| c.lane.stats).unwrap_or_default()
    }

    /// Windows of the frame parser that had the sync bytes and failed the CRC, and bytes it skipped.
    pub fn parser_bad(&self) -> u32 {
        self.0.try_borrow().map(|c| c.lane.parser().bad).unwrap_or(0)
    }
    pub fn parser_skipped(&self) -> u32 {
        self.0.try_borrow().map(|c| c.lane.parser().skipped).unwrap_or(0)
    }

    pub fn datagram_stats(&self) -> klong_dgram::dgram::Stats {
        self.0.try_borrow().map(|c| c.lane.dg.stats).unwrap_or_default()
    }

    pub fn counters(&self) -> NodeCounters {
        self.0.try_borrow().map(|c| c.counters).unwrap_or_default()
    }

    /// The console's input: the bytes of the input datagrams, one datagram at a time (a datagram is
    /// not handed out before the one before it is finished: `done` is per datagram), after an
    /// interrupt's 0x03 when there is one. Non-blocking.
    pub fn console_read(&self, buf: &mut [u8]) -> usize {
        let Ok(mut c) = self.0.try_borrow_mut() else { return 0 };
        c.pump();
        if buf.is_empty() {
            return 0;
        }
        if c.ctrl {
            c.ctrl = false;
            buf[0] = 0x03;
            return 1;
        }
        let Some(r) = c.recs.front_mut() else { return 0 };
        let n = buf.len().min(r.data.len() - r.pos);
        buf[..n].copy_from_slice(&r.data[r.pos..r.pos + n]);
        r.pos += n;
        r.started = true;
        n
    }

    /// Console output, for the peer that last sent input. With no peer (nothing was typed from a
    /// lane yet, or somebody typed on the local console since) it is dropped and counted.
    pub fn console_write(&self, bytes: &[u8]) {
        let Ok(mut c) = self.0.try_borrow_mut() else { return };
        if !c.cfg.console {
            return;
        }
        if c.cpeer.is_none() {
            c.counters.console_no_peer = c.counters.console_no_peer.saturating_add(bytes.len() as u32);
            return;
        }
        // a datagram peer gets "\n", not the "\r\n" a terminal wants (the board's console writes CR LF)
        let mut text: Vec<u8> = Vec::with_capacity(bytes.len());
        for (i, &b) in bytes.iter().enumerate() {
            if b == b'\r' && bytes.get(i + 1) == Some(&b'\n') {
                continue;
            }
            text.push(b);
        }
        let bytes = &text[..];
        let room = c.cfg.console_out_bytes.saturating_sub(c.cout_bytes);
        if room < bytes.len() {
            c.counters.console_dropped += (bytes.len() - room) as u32;
        }
        let n = room.min(bytes.len());
        let peer = c.cpeer.unwrap_or(0);
        match c.cout.back_mut() {
            Some((p, v)) if *p == peer => v.extend_from_slice(&bytes[..n]),
            _ => c.cout.push_back((peer, bytes[..n].to_vec())),
        }
        c.cout_bytes += n;
        c.pump();
    }

    /// Somebody typed on the local console (USB, say): the lane's peer no longer owns the output.
    pub fn console_local_input(&self) {
        if let Ok(mut c) = self.0.try_borrow_mut() {
            c.cpeer = None;
        }
    }

    /// True while input datagrams are being evaluated: the console should not echo or prompt.
    pub fn console_quiet(&self) -> bool {
        self.0.try_borrow().map(|c| !c.recs.is_empty()).unwrap_or(false)
    }

    /// At the end of a turn of the main loop: the console has `input_pending` bytes it has read and
    /// not handled, and has printed `errors` errors in all, `interrupts` of them interrupts. A
    /// datagram that has been read whole and handled is finished: `done ID STATUS` is queued behind
    /// its output, and the gate opens for another sender when none is left.
    pub fn console_turn(&self, input_pending: bool, errors: u32, interrupts: u32) {
        let Ok(mut c) = self.0.try_borrow_mut() else { return };
        let (de, di) = (errors.wrapping_sub(c.last_errors), interrupts.wrapping_sub(c.last_interrupts));
        c.last_errors = errors;
        c.last_interrupts = interrupts;
        if let Some(r) = c.recs.front_mut() {
            if r.started {
                r.errors += de;
                if di > 0 {
                    r.interrupted = true;
                }
            }
        }
        if input_pending {
            return;
        }
        let finished = c.recs.front().is_some_and(|r| r.started && r.pos >= r.data.len());
        if finished && !c.ctrl {
            if let Some(r) = c.recs.pop_front() {
                c.in_bytes = c.in_bytes.saturating_sub(r.data.len());
                let status = if r.interrupted {
                    "interrupted"
                } else if r.errors > 0 {
                    "error"
                } else {
                    "ok"
                };
                c.ctl_out.push_back((r.src, alloc::format!("done {} {}", r.id, status).into_bytes()));
            }
        }
        c.pump();
    }

    /// The peer the console channel talks to.
    pub fn console_peer(&self) -> Option<u8> {
        self.0.try_borrow().ok().and_then(|c| c.cpeer)
    }

    /// Send a message on any channel, for the owner's own protocols; `false` when the sender is busy.
    /// Like [`NodeNet::send`], and says the id of the message that was queued.
    pub fn send_id(&self, dest: u8, channel: u8, data: &[u8], ack: bool) -> Result<Option<u16>, SendError> {
        let Ok(mut c) = self.0.try_borrow_mut() else { return Ok(None) };
        c.pump();
        match c.lane.send(dest, channel, data, ack) {
            Ok(id) => {
                c.pump();
                Ok(Some(id))
            }
            Err(SendError::Busy) => Ok(None),
            Err(e) => Err(e),
        }
    }

    /// [`NodeNet::send_id`] with the message id chosen by the caller; `false` when the sender is busy.
    pub fn send_as(&self, dest: u8, channel: u8, data: &[u8], ack: bool, id: u16) -> Result<bool, SendError> {
        let Ok(mut c) = self.0.try_borrow_mut() else { return Ok(false) };
        c.pump();
        match c.lane.send_as(dest, channel, data, ack, id) {
            Ok(_) => {
                c.pump();
                Ok(true)
            }
            Err(SendError::Busy) => Ok(false),
            Err(e) => Err(e),
        }
    }

    /// What became of the last message sent with ACK asked for.
    pub fn take_send_result(&self) -> Option<klong_dgram::SendResult> {
        self.0.try_borrow_mut().ok().and_then(|mut c| c.lane.dg.take_result())
    }

    /// Send a message on any channel; `false` when the sender is busy.
    pub fn send(&self, dest: u8, channel: u8, data: &[u8], ack: bool) -> Result<bool, SendError> {
        let Ok(mut c) = self.0.try_borrow_mut() else { return Ok(false) };
        c.pump();
        match c.lane.send(dest, channel, data, ack) {
            Ok(_) => {
                c.pump();
                Ok(true)
            }
            Err(SendError::Busy) => Ok(false),
            Err(e) => Err(e),
        }
    }
}

struct NodeTransport(NodeNet);

fn parse_node(addr: &str) -> Option<u8> {
    let a = addr.strip_prefix('n').unwrap_or(addr);
    if a.is_empty() || a.len() > 2 || !a.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let n: u8 = a.parse().ok()?;
    if n <= klong_dgram::frame::SUPERVISOR {
        Some(n)
    } else {
        None
    }
}

impl Transport for NodeTransport {
    fn name(&self) -> &str {
        "node"
    }
    fn connect(&mut self, addr: &str) -> Result<Link, LinkError> {
        let peer = parse_node(addr).ok_or(LinkError::BadAddress)?;
        let mut c = (self.0).0.try_borrow_mut().map_err(|_| LinkError::Io)?;
        c.pump();
        match c.lane.node() {
            None => return Err(LinkError::Unreachable),
            Some(me) if me == peer => return Err(LinkError::BadAddress),
            _ => {}
        }
        if c.find(peer, true).is_some() {
            return Err(LinkError::InUse);
        }
        let epoch = c.new_epoch();
        let id = c.open(peer, true, epoch);
        let mtu = c.cfg.link_mtu.clamp(8, MSG_MAX - HEAD);
        drop(c);
        Ok(Link::Packet(Box::new(NodeLink { net: self.0.clone(), id, mtu })))
    }
    fn listen(&mut self, _addr: &str) -> Result<ListenerId, LinkError> {
        let mut c = (self.0).0.try_borrow_mut().map_err(|_| LinkError::Io)?;
        if c.listening {
            return Err(LinkError::InUse);
        }
        c.listening = true;
        Ok(ListenerId(1))
    }
    fn accept(&mut self, _l: ListenerId) -> Option<(Link, String)> {
        let mut c = (self.0).0.try_borrow_mut().ok()?;
        c.pump();
        while let Some(id) = c.accept.pop_front() {
            if let Some(s) = c.slot_mut(id) {
                let name = alloc::format!("node:{}", s.peer);
                let mtu = c.cfg.link_mtu.clamp(8, MSG_MAX - HEAD);
                drop(c);
                return Some((Link::Packet(Box::new(NodeLink { net: self.0.clone(), id, mtu })), name));
            }
        }
        None
    }
    fn unlisten(&mut self, _l: ListenerId) {
        if let Ok(mut c) = (self.0).0.try_borrow_mut() {
            c.listening = false;
            c.accept.clear();
        }
    }
}

struct NodeLink {
    net: NodeNet,
    id: u32,
    mtu: usize,
}

impl PacketLink for NodeLink {
    fn mtu(&self) -> usize {
        self.mtu
    }
    fn reliable(&self) -> bool {
        false
    }
    fn send(&mut self, pkt: &[u8]) -> Result<bool, LinkError> {
        if pkt.len() > self.mtu {
            return Err(LinkError::Io);
        }
        let mut c = self.net.0.try_borrow_mut().map_err(|_| LinkError::Io)?;
        c.pump();
        let (peer, h, closed) = match c.slot_mut(self.id) {
            Some(s) => (s.peer, s.epoch << 1 | s.client as u16, s.closed),
            None => return Err(LinkError::Closed),
        };
        if closed {
            return Err(LinkError::Closed);
        }
        let mut msg = [0u8; MSG_MAX];
        msg[0] = h as u8;
        msg[1] = (h >> 8) as u8;
        msg[HEAD..HEAD + pkt.len()].copy_from_slice(pkt);
        match c.lane.send(peer, CH_IPC, &msg[..HEAD + pkt.len()], false) {
            Ok(_) => {
                c.counters.messages_out += 1;
                c.pump();
                Ok(true)
            }
            Err(SendError::Busy) => {
                if pkt.len() == 1 && pkt[0] >> 6 == REL_RST {
                    // the goodbye of a connection that is being closed: nobody will offer it again
                    if c.rst_out.len() < 8 {
                        c.rst_out.push((peer, h));
                    }
                    return Ok(true);
                }
                Ok(false)
            }
            Err(_) => Err(LinkError::Io),
        }
    }
    fn recv(&mut self, buf: &mut [u8]) -> Result<Option<usize>, LinkError> {
        let mut c = self.net.0.try_borrow_mut().map_err(|_| LinkError::Io)?;
        c.pump();
        let Some(s) = c.slot_mut(self.id) else { return Err(LinkError::Closed) };
        match s.inbox.pop_front() {
            Some(m) => {
                let n = (m.len() - HEAD).min(buf.len());
                buf[..n].copy_from_slice(&m[HEAD..HEAD + n]);
                Ok(Some(n))
            }
            None if s.closed => Err(LinkError::Closed),
            None => Ok(None),
        }
    }
    fn rto_hint_ms(&self) -> u32 {
        self.net.0.try_borrow().map(|c| c.cfg.rto_ms).unwrap_or(500)
    }
    fn tuning(&self) -> LinkTuning {
        match self.net.0.try_borrow() {
            Ok(c) => LinkTuning { handshake_timeout_us: Some(c.cfg.handshake_timeout_us), retries: Some(c.cfg.retries), call_timeout_us: None, idle_ms: None },
            Err(_) => LinkTuning::default(),
        }
    }
    fn close(&mut self) {
        if let Ok(mut c) = self.net.0.try_borrow_mut() {
            let id = self.id;
            c.slots.retain(|s| s.id != id);
            c.pump();
        }
    }
}
