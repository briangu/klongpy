//! Links in memory: two runtimes in one process joined by a pipe, with no operating system.
//! For tests and for a simulated board: a clean byte stream, a packet link that loses,
//! duplicates and reorders, and a byte pipe with line noise behind the serial framing.
//!
//! A [`Hub`] is the "network"; each runtime gets a [`MemTransport`] on it under some name
//! (`mem`, `lossy`, `uart`). `listen("a")` on one and `connect("a")` on another join them.
use alloc::boxed::Box;
use alloc::collections::VecDeque;
use alloc::rc::Rc;
use alloc::string::String;
use alloc::vec::Vec;
use core::cell::RefCell;

use crate::link::{Link, LinkError, ListenerId, PacketLink, SerialLink, StreamLink, Transport};

/// Faults, in parts per million. For packets: per packet. For a serial pipe: `flip` and
/// `loss` per byte.
#[derive(Clone, Copy, Default, Debug)]
pub struct Faults {
    pub loss: u32,
    pub dup: u32,
    pub reorder: u32,
    /// A byte is replaced by a random one (serial pipes only).
    pub flip: u32,
}

#[derive(Clone, Copy, Debug)]
pub enum MemKind {
    /// Reliable ordered bytes, as TCP.
    Stream,
    /// Packets of at most `mtu` bytes; `reliable` is what the link claims to be.
    Packet { mtu: usize, reliable: bool, rto_ms: u32 },
    /// A raw byte pipe behind COBS and CRC-16 framing (`SerialLink`).
    Serial { mtu: usize, rto_ms: u32 },
}

/// One direction of a pipe.
#[derive(Default)]
struct Lane {
    bytes: VecDeque<u8>,
    packets: VecDeque<Vec<u8>>,
    /// A packet held back to be overtaken by the next one.
    held: Option<Vec<u8>>,
}

struct Pipe {
    lanes: [Lane; 2],
    closed: bool,
    rng: u64,
    faults: Faults,
    /// Counters: packets (or bytes) sent, lost or damaged.
    sent: u64,
    lost: u64,
}

impl Pipe {
    fn chance(&mut self, ppm: u32) -> bool {
        if ppm == 0 {
            return false;
        }
        self.next() % 1_000_000 < ppm as u64
    }
    fn next(&mut self) -> u64 {
        let mut x = self.rng;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng = x;
        x
    }
}

struct Waiting {
    addr: String,
    id: ListenerId,
    queue: VecDeque<(Link, String)>,
}

struct HubInner {
    listeners: Vec<Waiting>,
    next: u32,
    seed: u64,
    /// Every pipe made, for the counters.
    pipes: Vec<Rc<RefCell<Pipe>>>,
}

#[derive(Clone)]
pub struct Hub(Rc<RefCell<HubInner>>);

impl Hub {
    pub fn new(seed: u64) -> Hub {
        Hub(Rc::new(RefCell::new(HubInner { listeners: Vec::new(), next: 1, seed: seed | 1, pipes: Vec::new() })))
    }

    /// (packets or bytes sent, of which lost or damaged) over every pipe so far.
    pub fn counters(&self) -> (u64, u64) {
        let h = self.0.borrow();
        h.pipes.iter().fold((0, 0), |(s, l), p| {
            let p = p.borrow();
            (s + p.sent, l + p.lost)
        })
    }

    /// Cut every pipe: both ends see the link closed (a cable pulled, a peer that died).
    pub fn cut(&self) {
        for p in self.0.borrow().pipes.iter() {
            p.borrow_mut().closed = true;
        }
    }

    /// Change the faults of every pipe, existing and future ones of `t`.
    pub fn set_faults(&self, f: Faults) {
        for p in self.0.borrow().pipes.iter() {
            p.borrow_mut().faults = f;
        }
    }
}

struct End {
    pipe: Rc<RefCell<Pipe>>,
    /// The lane this end writes to; it reads from the other.
    side: usize,
    kind: MemKind,
}

impl StreamLink for End {
    fn read(&mut self, buf: &mut [u8]) -> Result<usize, LinkError> {
        let mut p = self.pipe.borrow_mut();
        let closed = p.closed;
        let lane = &mut p.lanes[1 - self.side];
        let n = buf.len().min(lane.bytes.len());
        for (b, q) in buf.iter_mut().zip(lane.bytes.drain(..n)) {
            *b = q;
        }
        if n == 0 && closed {
            return Err(LinkError::Closed);
        }
        Ok(n)
    }
    fn write(&mut self, buf: &[u8]) -> Result<usize, LinkError> {
        let mut p = self.pipe.borrow_mut();
        if p.closed {
            return Err(LinkError::Closed);
        }
        let noisy = matches!(self.kind, MemKind::Serial { .. });
        for &b in buf {
            p.sent += 1;
            let mut b = b;
            if noisy {
                let f = p.faults;
                if p.chance(f.loss) {
                    p.lost += 1;
                    continue;
                }
                if p.chance(f.flip) {
                    p.lost += 1;
                    b = p.next() as u8;
                }
            }
            p.lanes[self.side].bytes.push_back(b);
        }
        Ok(buf.len())
    }
    fn close(&mut self) {
        self.pipe.borrow_mut().closed = true;
    }
}

impl PacketLink for End {
    fn mtu(&self) -> usize {
        match self.kind {
            MemKind::Packet { mtu, .. } => mtu,
            _ => 4,
        }
    }
    fn reliable(&self) -> bool {
        matches!(self.kind, MemKind::Packet { reliable: true, .. })
    }
    fn send(&mut self, pkt: &[u8]) -> Result<bool, LinkError> {
        let mut p = self.pipe.borrow_mut();
        if p.closed {
            return Err(LinkError::Closed);
        }
        let f = p.faults;
        p.sent += 1;
        if p.chance(f.loss) {
            p.lost += 1;
            return Ok(true);
        }
        let copies = if p.chance(f.dup) { 2 } else { 1 };
        let hold = p.chance(f.reorder);
        let lane = &mut p.lanes[self.side];
        if hold && lane.held.is_none() {
            lane.held = Some(pkt.to_vec());
            return Ok(true);
        }
        for _ in 0..copies {
            lane.packets.push_back(pkt.to_vec());
        }
        if let Some(h) = lane.held.take() {
            lane.packets.push_back(h);
        }
        Ok(true)
    }
    fn recv(&mut self, buf: &mut [u8]) -> Result<Option<usize>, LinkError> {
        let mut p = self.pipe.borrow_mut();
        let closed = p.closed;
        let lane = &mut p.lanes[1 - self.side];
        // (a packet held back with nothing behind it arrives late rather than never)
        let pkt = lane.packets.pop_front().or_else(|| lane.held.take());
        match pkt {
            Some(pkt) => {
                let n = pkt.len().min(buf.len());
                buf[..n].copy_from_slice(&pkt[..n]);
                Ok(Some(n))
            }
            None if closed => Err(LinkError::Closed),
            None => Ok(None),
        }
    }
    fn rto_hint_ms(&self) -> u32 {
        match self.kind {
            MemKind::Packet { rto_ms, .. } => rto_ms,
            _ => 50,
        }
    }
    fn close(&mut self) {
        self.pipe.borrow_mut().closed = true;
    }
}

/// `host:port` and `:port` name the same place here: there is one host.
fn port_of(addr: &str) -> &str {
    addr.rsplit(':').next().unwrap_or(addr)
}

pub struct MemTransport {
    hub: Hub,
    name: String,
    kind: MemKind,
    pub faults: Faults,
}

impl MemTransport {
    pub fn new(hub: &Hub, name: &str, kind: MemKind, faults: Faults) -> MemTransport {
        MemTransport { hub: hub.clone(), name: String::from(name), kind, faults }
    }

    fn end(&self, pipe: &Rc<RefCell<Pipe>>, side: usize) -> Link {
        let end = Box::new(End { pipe: pipe.clone(), side, kind: self.kind });
        match self.kind {
            MemKind::Stream => Link::Stream(end),
            MemKind::Packet { .. } => Link::Packet(end),
            MemKind::Serial { mtu, rto_ms } => Link::Packet(Box::new(SerialLink::new(end, mtu, rto_ms))),
        }
    }
}

impl Transport for MemTransport {
    fn name(&self) -> &str {
        &self.name
    }
    fn connect(&mut self, addr: &str) -> Result<Link, LinkError> {
        let addr = port_of(addr);
        let mut h = self.hub.0.borrow_mut();
        h.seed = h.seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
        let pipe = Rc::new(RefCell::new(Pipe { lanes: [Lane::default(), Lane::default()], closed: false, rng: h.seed, faults: self.faults, sent: 0, lost: 0 }));
        let (ours, theirs) = (self.end(&pipe, 0), self.end(&pipe, 1));
        let w = h.listeners.iter_mut().find(|w| w.addr == addr).ok_or(LinkError::Refused)?;
        w.queue.push_back((theirs, alloc::format!("{}:{}", self.name, addr)));
        h.pipes.push(pipe);
        Ok(ours)
    }
    fn listen(&mut self, addr: &str) -> Result<ListenerId, LinkError> {
        let addr = port_of(addr);
        let mut h = self.hub.0.borrow_mut();
        if h.listeners.iter().any(|w| w.addr == addr) {
            return Err(LinkError::InUse);
        }
        let id = ListenerId(h.next);
        h.next += 1;
        h.listeners.push(Waiting { addr: String::from(addr), id, queue: VecDeque::new() });
        Ok(id)
    }
    fn accept(&mut self, l: ListenerId) -> Option<(Link, String)> {
        let mut h = self.hub.0.borrow_mut();
        h.listeners.iter_mut().find(|w| w.id == l)?.queue.pop_front()
    }
    fn unlisten(&mut self, l: ListenerId) {
        self.hub.0.borrow_mut().listeners.retain(|w| w.id != l);
    }
}
