//! A model of the Pico bench's packet switch for the simulator: ports 0..N-1 for nodes and
//! port 16 for the supervisor, each a 115200 baud 8N1 lane, a FIFO of 256 frames per output
//! port, store and forward, best effort. Time is virtual: whoever reads or writes a lane
//! passes the clock's reading, and the switch catches up to it.
//!
//! * A frame takes 16 byte times (1.389 ms) to cross a lane, in either direction.
//! * A frame is routed when its last byte has arrived, by its destination: a node port, the
//!   supervisor, or (counted, dropped) anything else.
//! * An output port holds 256 frames waiting for its lane; a frame that finds it full is
//!   dropped (and counted). Frames from one input to one output stay in order.
//! * The baud rate is a parameter ([`Switch::set_baud`]). An endpoint is not free either:
//!   [`Switch::set_cpu`] charges each lane a processing time per frame received and sent (what
//!   a Cortex-M0+ spends on the interrupt, the parser, the CRC and the datagram layer; an
//!   estimate, see `docs/native/pico-bench.md`), and the receive ring holds a finite number of
//!   bytes ([`Switch::set_ring`]): what arrives while it is full is lost and counted.
//! * The supervisor sends PING to every node once a second when asked to ([`Switch::set_pings`]).
use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;

use klong_dgram::frame::{Frame, Parser, FRAME, SUPERVISOR, T_PING, T_PONG, T_STATUS};
use klong_rt::node::{LaneIo, UartCounters};

/// Microseconds per byte at 115200 baud, 8N1 (10 bits).
pub const BYTE_US_115200: f64 = 10.0 * 1_000_000.0 / 115_200.0;
/// A UART transmit FIFO (RP2040: 32 bytes).
pub const TX_FIFO: usize = 32;

#[derive(Default, Clone, Debug)]
pub struct SwitchStats {
    pub frames_in: u64,
    pub frames_out: u64,
    /// Dropped because the output port's queue was full.
    pub overflow: u64,
    /// Dropped by the fault injector.
    pub injected_loss: u64,
    /// Sent to an address the switch does not route here (17, broadcast, groups, no such port).
    pub bad_destination: u64,
    pub max_queue: usize,
    /// PONG and STATUS frames that reached the supervisor, per source node.
    pub pongs: [u64; 17],
    pub statuses: [u64; 17],
    pub pings_sent: u64,
    /// The flags, load and uptime of the last STATUS from each node (flags 0 before the first).
    pub status_flags: [u8; 17],
    pub status_load: [u8; 17],
    pub status_uptime: [u16; 17],
    /// Bytes thrown away because a node's receive ring was full, per node port.
    pub rx_ring_overruns: [u64; 17],
    /// Bytes of damaged or torn frames the switch's parsers skipped.
    pub skipped_bytes: u64,
}

struct Port {
    /// Frames waiting for this port's lane, with the time each was complete in the switch.
    queue: VecDeque<(Frame, f64)>,
    line_free_at: f64,
    /// Bytes on their way to the node: (arrival time, byte).
    to_node: VecDeque<(f64, u8)>,
    /// Bytes from the node on their way into the switch.
    from_node: VecDeque<(f64, u8)>,
    parser: Parser,
    tx_free_at: f64,
    /// The endpoint's processor is busy until this time.
    cpu_at: f64,
    rx_overruns: u64,
    tx_full: u64,
}

struct Inner {
    ports: Vec<Port>,
    stats: SwitchStats,
    queue_cap: usize,
    byte_us: f64,
    now: f64,
    rng: u64,
    /// Microseconds of processor per 16-byte frame received, and sent.
    rx_cost_us: f64,
    tx_cost_us: f64,
    ring_bytes: usize,
    tx_ring_bytes: usize,
    loss_ppm: u32,
    drop_next: u32,
    pings: bool,
    next_ping: f64,
    ping_seq: u16,
    /// A log of every frame routed, when asked for: (time, frame, delivered).
    trace: Option<Vec<(f64, Frame, bool)>>,
}

#[derive(Clone)]
pub struct Switch(Rc<RefCell<Inner>>);

fn index(addr: u8, nodes: usize) -> Option<usize> {
    if (addr as usize) < nodes {
        Some(addr as usize)
    } else if addr == SUPERVISOR {
        Some(nodes)
    } else {
        None
    }
}

impl Inner {
    fn rand(&mut self) -> u64 {
        let mut x = self.rng;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng = x;
        x
    }

    fn nodes(&self) -> usize {
        self.ports.len() - 1
    }

    fn route(&mut self, f: Frame, at: f64) {
        self.stats.frames_in += 1;
        let nodes = self.nodes();
        let mut delivered = false;
        if self.drop_next > 0 {
            self.drop_next -= 1;
            self.stats.injected_loss += 1;
        } else if self.loss_ppm > 0 && self.rand() % 1_000_000 < self.loss_ppm as u64 {
            self.stats.injected_loss += 1;
        } else {
            match index(f.dst, nodes) {
                None => self.stats.bad_destination += 1,
                Some(i) => {
                    if self.ports[i].queue.len() >= self.queue_cap {
                        self.stats.overflow += 1;
                    } else {
                        self.ports[i].queue.push_back((f, at));
                        let n = self.ports[i].queue.len();
                        self.stats.max_queue = self.stats.max_queue.max(n);
                        delivered = true;
                        if i == nodes {
                            let s = f.src as usize;
                            if s < 17 {
                                match f.ftype {
                                    T_PONG => self.stats.pongs[s] += 1,
                                    T_STATUS => {
                                        self.stats.statuses[s] += 1;
                                        if f.len == 4 {
                                            self.stats.status_load[s] = f.payload[0];
                                            self.stats.status_flags[s] = f.payload[1];
                                            self.stats.status_uptime[s] = f.payload[2] as u16 | (f.payload[3] as u16) << 8;
                                        }
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }
                }
            }
        }
        if let Some(t) = self.trace.as_mut() {
            t.push((at, f, delivered));
        }
    }

    fn advance(&mut self, now: f64) {
        if now > self.now {
            self.now = now;
        }
        let now = self.now;
        let byte = self.byte_us;
        if self.pings {
            while self.next_ping <= now {
                let t = self.next_ping;
                for n in 0..self.nodes() {
                    let f = Frame::new(T_PING, SUPERVISOR, n as u8, self.ping_seq, 16, &[]);
                    self.ping_seq = self.ping_seq.wrapping_add(1);
                    self.stats.pings_sent += 1;
                    self.route(f, t);
                }
                self.next_ping += 1_000_000.0;
            }
        }
        // bytes from the nodes that have arrived: frames are routed in time order
        let mut arrived: Vec<(f64, Frame)> = Vec::new();
        for p in self.ports.iter_mut() {
            while let Some(&(t, b)) = p.from_node.front() {
                if t > now {
                    break;
                }
                p.from_node.pop_front();
                let before = p.parser.skipped;
                if let Some(f) = p.parser.push(b) {
                    arrived.push((t, f));
                }
                self.stats.skipped_bytes += (p.parser.skipped - before) as u64;
            }
        }
        arrived.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        for (t, f) in arrived {
            self.route(f, t);
        }
        // serialize the queues down the lanes
        for p in self.ports.iter_mut() {
            while let Some(&(f, ready)) = p.queue.front() {
                let start = p.line_free_at.max(ready);
                if start > now {
                    break;
                }
                p.queue.pop_front();
                let Some(bytes) = f.encode() else { continue };
                for (k, b) in bytes.iter().enumerate() {
                    p.to_node.push_back((start + (k as f64 + 1.0) * byte, *b));
                }
                p.line_free_at = start + FRAME as f64 * byte;
                self.stats.frames_out += 1;
            }
        }
    }

    /// The earliest virtual time after `now` at which something moves in the switch, as a
    /// reader of a lane would see it.
    fn next_event(&self) -> Option<f64> {
        let mut t: Option<f64> = None;
        let mut take = |v: f64| {
            if v > self.now {
                t = Some(t.map_or(v, |x: f64| x.min(v)));
            }
        };
        for p in &self.ports {
            if let Some(&(a, _)) = p.to_node.front() {
                take(a);
            }
            if let Some(&(a, _)) = p.from_node.front() {
                take(a);
            }
            if let Some(&(_, ready)) = p.queue.front() {
                take(p.line_free_at.max(ready));
            }
        }
        if self.pings {
            take(self.next_ping);
        }
        t
    }
}

impl Switch {
    /// `nodes` node ports (0..nodes) and the supervisor's port (16).
    pub fn new(nodes: usize, seed: u64) -> Switch {
        let ports = (0..=nodes).map(|_| Port { queue: VecDeque::new(), line_free_at: 0.0, to_node: VecDeque::new(), from_node: VecDeque::new(), parser: Parser::new(), tx_free_at: 0.0, cpu_at: 0.0, rx_overruns: 0, tx_full: 0 }).collect();
        Switch(Rc::new(RefCell::new(Inner {
            ports,
            stats: SwitchStats::default(),
            queue_cap: 256,
            byte_us: BYTE_US_115200,
            now: 0.0,
            rng: seed | 1,
            rx_cost_us: 0.0,
            tx_cost_us: 0.0,
            ring_bytes: 4096,
            tx_ring_bytes: 32,
            loss_ppm: 0,
            drop_next: 0,
            pings: false,
            next_ping: 0.0,
            ping_seq: 0,
            trace: None,
        })))
    }

    /// The lane of node `n` (or of the supervisor, `n == 16`), as a [`LaneIo`] on the clock `now_us`.
    pub fn port(&self, addr: u8, clock: Rc<std::cell::Cell<u64>>) -> SwitchPort {
        let nodes = self.0.borrow().nodes();
        let index = index(addr, nodes).expect("no such port");
        SwitchPort { sw: self.clone(), index, clock }
    }

    pub fn stats(&self) -> SwitchStats {
        self.0.borrow().stats.clone()
    }
    pub fn set_queue_cap(&self, n: usize) {
        self.0.borrow_mut().queue_cap = n;
    }
    /// Baud rate of every lane (default 115200).
    pub fn set_baud(&self, baud: u32) {
        self.0.borrow_mut().byte_us = 10.0 * 1_000_000.0 / baud as f64;
    }
    /// Processor time of an endpoint per frame received and per frame sent, microseconds.
    pub fn set_cpu(&self, rx_us: f64, tx_us: f64) {
        let mut s = self.0.borrow_mut();
        s.rx_cost_us = rx_us;
        s.tx_cost_us = tx_us;
    }
    /// Bytes of the receive ring and of the transmit ring of every node lane.
    pub fn set_ring(&self, rx_bytes: usize, tx_bytes: usize) {
        let mut s = self.0.borrow_mut();
        s.ring_bytes = rx_bytes;
        s.tx_ring_bytes = tx_bytes;
    }
    /// Drop this fraction of frames (parts per million) at random, seeded.
    pub fn set_loss_ppm(&self, ppm: u32) {
        self.0.borrow_mut().loss_ppm = ppm;
    }
    /// Drop the next `n` frames, whatever they are.
    pub fn drop_next(&self, n: u32) {
        self.0.borrow_mut().drop_next = n;
    }
    /// The supervisor's once-a-second PING to every node.
    pub fn set_pings(&self, on: bool) {
        let mut s = self.0.borrow_mut();
        s.pings = on;
        s.next_ping = s.now;
    }
    pub fn set_trace(&self, on: bool) {
        self.0.borrow_mut().trace = if on { Some(Vec::new()) } else { None };
    }
    pub fn take_trace(&self) -> Vec<(f64, Frame, bool)> {
        self.0.borrow_mut().trace.as_mut().map(std::mem::take).unwrap_or_default()
    }
    /// Frames waiting for the lane of port `addr`.
    pub fn queued(&self, addr: u8) -> usize {
        let s = self.0.borrow();
        index(addr, s.nodes()).map_or(0, |i| s.ports[i].queue.len())
    }
    /// A node that was reset: what was in its UART and parser is gone.
    pub fn clear_lane(&self, addr: u8) {
        let mut s = self.0.borrow_mut();
        let nodes = s.nodes();
        if let Some(i) = index(addr, nodes) {
            let p = &mut s.ports[i];
            p.to_node.clear();
            p.from_node.clear();
            p.parser.reset();
            p.tx_free_at = 0.0;
            p.cpu_at = 0.0;
        }
    }
    pub fn advance(&self, now_us: u64) {
        self.0.borrow_mut().advance(now_us as f64);
    }
    /// When something next happens in the switch, virtual microseconds (rounded up).
    pub fn next_event_us(&self) -> Option<u64> {
        self.0.borrow().next_event().map(|t| t.ceil() as u64)
    }
}

/// One lane of the switch from the node's end.
pub struct SwitchPort {
    sw: Switch,
    index: usize,
    clock: Rc<std::cell::Cell<u64>>,
}

impl LaneIo for SwitchPort {
    fn read(&mut self, buf: &mut [u8]) -> usize {
        let mut s = self.sw.0.borrow_mut();
        s.advance(self.clock.get() as f64);
        let now = s.now;
        let (ring, per_byte) = (s.ring_bytes, s.rx_cost_us / FRAME as f64);
        let index = self.index;
        let p = &mut s.ports[index];
        // what has arrived and was not read is in the ring; what does not fit is lost
        let arrived = p.to_node.iter().take_while(|(t, _)| *t <= now).count();
        if arrived > ring {
            // the bytes that came after the ring filled up
            let lost = arrived - ring;
            p.to_node.drain(ring..arrived);
            p.rx_overruns += lost as u64;
        }
        let mut n = 0;
        while n < buf.len() {
            match p.to_node.front() {
                Some(&(t, b)) if t <= now && p.cpu_at <= now => {
                    buf[n] = b;
                    n += 1;
                    p.to_node.pop_front();
                    p.cpu_at = p.cpu_at.max(t) + per_byte;
                }
                _ => break,
            }
        }
        let over = p.rx_overruns;
        let addr = port_addr(index, s.nodes());
        s.stats.rx_ring_overruns[addr] = over;
        n
    }

    fn write(&mut self, bytes: &[u8]) -> usize {
        let mut s = self.sw.0.borrow_mut();
        s.advance(self.clock.get() as f64);
        let now = s.now;
        let byte = s.byte_us;
        let (cap, per_byte) = (s.tx_ring_bytes, s.tx_cost_us / FRAME as f64);
        let p = &mut s.ports[self.index];
        // bytes still waiting for their turn on the wire (the FIFO and the transmit ring)
        let queued = if p.tx_free_at > now { ((p.tx_free_at - now) / byte).ceil() as usize } else { 0 };
        let room = cap.saturating_sub(queued);
        let n = room.min(bytes.len());
        if n < bytes.len() {
            p.tx_full += 1;
        }
        let mut t = p.tx_free_at.max(now);
        for &b in &bytes[..n] {
            t += byte;
            p.from_node.push_back((t, b));
        }
        p.tx_free_at = t;
        p.cpu_at = p.cpu_at.max(now) + n as f64 * per_byte;
        n
    }

    fn counters(&self) -> UartCounters {
        let s = self.sw.0.borrow();
        let p = &s.ports[self.index];
        UartCounters { rx_ring_overruns: p.rx_overruns as u32, rx_fifo_overruns: 0, line_errors: 0, tx_ring_full: p.tx_full as u32 }
    }
}

fn port_addr(index: usize, nodes: usize) -> usize {
    if index == nodes {
        16
    } else {
        index
    }
}
