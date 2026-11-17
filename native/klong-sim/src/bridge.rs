//! A model of the supervisor's Klong console bridge (`klong_bridge.h` of the esp32_cube bench
//! firmware), in front of the simulator's switch, for testing the whole path without hardware.
//! `native/tools/bench_bridge_sim.py` puts the HTTP routes (and the control rules) on it.
//!
//! What it does, as the real one does it:
//!
//! * One counter numbers every message the supervisor sends (input, `int`, `hello`), from 1, shared
//!   by every client; `eval` and `interrupt` return the id.
//! * Per node a queue of two messages (an evaluation and an interrupt); each goes out with the ACK
//!   flag, one at a time; what became of the last one is `sent`: sending, delivered, failed (no ACK
//!   after the resends), refused (ACK 2: the node's console is busy for another sender).
//! * `eval` adds a newline if there is none, refuses more than 1024 bytes, and puts each line in the
//!   transcript as `klong> LINE` before the node's output.
//! * Output (channel 1) is kept per node, the newest 8 KB, and handed out at most 4 KB at a time,
//!   never cutting a UTF-8 character; a `from` older than the window starts at the oldest byte and
//!   says `lost`.
//! * `done ID STATUS` (channel 2): the last one is kept. The first reply to `hello` is `info`; the
//!   self-test line (channel 3) is kept without its newline.
//! * A node is online if its STATUS came within 3 s and a Klong node if the flag 0x10 is set; the
//!   load is its STATUS load, for a Klong node.
//!
//! The text protocol of [`Bridge::handle`] is what `klong-bench --bridge` speaks on a pipe.
use std::cell::RefCell;
use std::collections::VecDeque;

use klong_rt::node::{NodeNet, CH_CONSOLE, CH_CONTROL, CH_SELFTEST};

use crate::bench::Bench;
use crate::switch::Switch;

pub const KEEP: usize = 8 * 1024;
pub const PER_RESPONSE: usize = 4 * 1024;
pub const QUEUE: usize = 2;
pub const NODES: usize = 4;
const OFFLINE_US: u64 = 3_000_000;
const HELLO_EVERY_US: u64 = 10_000_000;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Sent {
    None,
    Sending,
    Delivered,
    Failed,
    Refused,
}

impl Sent {
    pub fn name(&self) -> &'static str {
        match self {
            Sent::None => "none",
            Sent::Sending => "sending",
            Sent::Delivered => "delivered",
            Sent::Failed => "failed",
            Sent::Refused => "refused",
        }
    }
}

struct Out {
    id: u16,
    chan: u8,
    data: Vec<u8>,
}

#[derive(Default)]
struct NodeState {
    buf: VecDeque<u8>,
    /// Counter of the first byte in `buf`.
    base: u64,
    /// The newest first, at most 8.
    dones: VecDeque<(u16, String)>,
    info: String,
    selftest: String,
    queue: VecDeque<Out>,
    /// The head of the queue has been handed to the lane.
    inflight: bool,
    last_id: u16,
    last_state: Option<Sent>,
    hello_at: Option<u64>,
    was_online: bool,
}

struct State {
    nodes: Vec<NodeState>,
    next_id: u16,
}

pub struct Bridge {
    sup: NodeNet,
    switch: Switch,
    clock: std::rc::Rc<std::cell::Cell<u64>>,
    nodes: usize,
    state: RefCell<State>,
}

#[derive(Debug, PartialEq, Eq)]
pub enum EvalError {
    BadNode,
    Offline,
    NotKlong,
    Empty,
    TooLong,
    /// The queue of the node holds two messages already.
    Full,
}

impl EvalError {
    pub fn code(&self) -> u16 {
        match self {
            EvalError::BadNode | EvalError::Empty => 400,
            EvalError::Offline | EvalError::NotKlong => 503,
            EvalError::TooLong => 413,
            EvalError::Full => 429,
        }
    }
}

pub struct Output {
    pub next: u64,
    pub data: Vec<u8>,
    pub lost: bool,
    pub load: u8,
    pub online: bool,
    pub klong: bool,
    pub done: Option<(u16, String)>,
    /// The newest 8, newest first (`done` is the first).
    pub dones: Vec<(u16, String)>,
    pub sent: (u16, Sent),
}

pub struct NodeInfo {
    pub online: bool,
    pub klong: bool,
    pub load: u8,
    pub info: String,
    pub selftest: String,
    /// The STATUS flags byte of the node's last STATUS (0 if none).
    pub flags: u8,
}

impl Bridge {
    pub fn new(b: &Bench) -> Bridge {
        Bridge {
            sup: b.sup.clone(),
            switch: b.switch.clone(),
            clock: b.clock.clone(),
            nodes: b.len(),
            state: RefCell::new(State { nodes: (0..NODES).map(|_| NodeState::default()).collect(), next_id: 1 }),
        }
    }

    fn flags(&self, node: usize) -> (u8, u8, bool) {
        if node >= self.nodes {
            return (0, 0, false);
        }
        let s = self.switch.stats();
        let at = s.status_at_us[node];
        let online = at != 0 && self.clock.get().saturating_sub(at) <= OFFLINE_US;
        (s.status_flags[node], s.status_load[node], online)
    }

    fn take_id(st: &mut State) -> u16 {
        let id = st.next_id;
        st.next_id = if id == u16::MAX { 1 } else { id + 1 };
        id
    }

    /// Queue a message for a node (the caller has checked the node); the id.
    fn queue(&self, st: &mut State, node: usize, chan: u8, data: &[u8]) -> Result<u16, EvalError> {
        if st.nodes[node].queue.len() >= QUEUE {
            return Err(EvalError::Full);
        }
        let id = Self::take_id(st);
        let n = &mut st.nodes[node];
        n.queue.push_back(Out { id, chan, data: data.to_vec() });
        n.last_id = id;
        n.last_state = Some(Sent::Sending);
        Ok(id)
    }

    /// Take in what the nodes sent, move the queues along, ask the nodes who they are.
    pub fn poll(&self) {
        // the supervisor's lane is served too (the board loop does that in the real one)
        self.sup.pump();
        let mut st = self.state.borrow_mut();
        let now = self.clock.get();
        for (src, chan, msg) in self.sup.take_log() {
            let Some(n) = st.nodes.get_mut(src as usize) else { continue };
            match chan {
                CH_CONSOLE => {
                    for b in msg {
                        if n.buf.len() >= KEEP {
                            n.buf.pop_front();
                            n.base += 1;
                        }
                        n.buf.push_back(b);
                    }
                }
                CH_CONTROL => {
                    let t = String::from_utf8_lossy(&msg).into_owned();
                    if let Some(rest) = t.strip_prefix("done ") {
                        let mut w = rest.split(' ');
                        if let (Some(id), Some(status)) = (w.next().and_then(|v| v.parse::<u16>().ok()), w.next()) {
                            n.dones.push_front((id, status.to_string()));
                            n.dones.truncate(8);
                        }
                    } else if t.starts_with("klong ") {
                        n.info = t.chars().take(160).collect();
                    }
                }
                CH_SELFTEST => n.selftest = String::from_utf8_lossy(&msg).trim_end_matches(['\n', '\r']).chars().take(160).collect(),
                _ => {}
            }
        }
        // the message in flight: how it ended
        while let Some(r) = self.sup.take_send_result() {
            for n in st.nodes.iter_mut() {
                if n.inflight && n.queue.front().is_some_and(|o| o.id == r.id) {
                    n.queue.pop_front();
                    n.inflight = false;
                    let s = match r.outcome {
                        klong_dgram::Outcome::Acked | klong_dgram::Outcome::Sent => Sent::Delivered,
                        klong_dgram::Outcome::Refused => Sent::Refused,
                        klong_dgram::Outcome::Failed => Sent::Failed,
                    };
                    if n.last_id == r.id {
                        n.last_state = Some(s);
                    }
                }
            }
        }
        for i in 0..NODES {
            let (flags, _, online) = self.flags(i);
            if st.nodes[i].was_online && !online {
                // the node went away: what was queued for it is dropped
                let n = &mut st.nodes[i];
                if !n.queue.is_empty() {
                    n.last_state = Some(Sent::Failed);
                }
                n.queue.clear();
                n.inflight = false;
            }
            st.nodes[i].was_online = online;
            // hello every 10 s until the node answers
            let n = &st.nodes[i];
            if online && flags & 0x10 != 0 && n.info.is_empty() && n.queue.len() < QUEUE && n.hello_at.is_none_or(|t| now.saturating_sub(t) > HELLO_EVERY_US) {
                let _ = self.queue(&mut st, i, CH_CONTROL, b"hello");
                st.nodes[i].hello_at = Some(now);
            }
        }
        // hand the head of a queue to the lane when it is free
        for i in 0..NODES.min(self.nodes) {
            let n = &mut st.nodes[i];
            if n.inflight {
                continue;
            }
            if let Some(o) = n.queue.front() {
                if matches!(self.sup.send_as(i as u8, o.chan, &o.data, true, o.id), Ok(true)) {
                    n.inflight = true;
                    break; // one at a time; the next node's turn comes at the next poll
                }
            }
        }
    }

    fn check_node(&self, node: usize, need_klong: bool) -> Result<(), EvalError> {
        if node >= NODES {
            return Err(EvalError::BadNode);
        }
        let (flags, _, online) = self.flags(node);
        if need_klong && !online {
            return Err(EvalError::Offline);
        }
        if need_klong && flags & 0x10 == 0 {
            return Err(EvalError::NotKlong);
        }
        Ok(())
    }

    /// Console input for a node: lines, a newline added if there is none. The id of the message.
    pub fn eval(&self, node: usize, body: &[u8]) -> Result<u16, EvalError> {
        self.eval_with(node, body, false)
    }

    /// `raw`: the body goes as it is: no newline is added and nothing is echoed (for upload commands).
    pub fn eval_with(&self, node: usize, body: &[u8], raw: bool) -> Result<u16, EvalError> {
        self.poll();
        self.check_node(node, true)?;
        if body.is_empty() {
            return Err(EvalError::Empty);
        }
        let mut text = body.to_vec();
        if !raw && text.last() != Some(&b'\n') {
            text.push(b'\n');
        }
        if text.len() > 1024 {
            return Err(EvalError::TooLong);
        }
        let mut st = self.state.borrow_mut();
        let id = self.queue(&mut st, node, CH_CONSOLE, &text)?;
        if raw {
            return Ok(id);
        }
        // the transcript shows each line typed, after a prompt, before its result
        let n = &mut st.nodes[node];
        let mut echo = Vec::new();
        for line in text.split_inclusive(|&b| b == b'\n') {
            echo.extend_from_slice(b"klong> ");
            echo.extend_from_slice(line);
        }
        for b in echo {
            if n.buf.len() >= KEEP {
                n.buf.pop_front();
                n.base += 1;
            }
            n.buf.push_back(b);
        }
        Ok(id)
    }

    pub fn interrupt(&self, node: usize) -> Result<u16, EvalError> {
        self.poll();
        self.check_node(node, true)?;
        let mut st = self.state.borrow_mut();
        self.queue(&mut st, node, CH_CONTROL, b"int")
    }

    /// `from`: the counter the client has reached; `None` starts at the oldest byte kept.
    pub fn output(&self, node: usize, from: Option<u64>) -> Result<Output, EvalError> {
        self.poll();
        self.check_node(node, false)?;
        let (flags, load, online) = self.flags(node);
        let st = self.state.borrow();
        let n = &st.nodes[node];
        let total = n.base + n.buf.len() as u64;
        let (mut start, lost) = match from {
            None => (n.base, false),
            Some(f) if f < n.base => (n.base, true),
            Some(f) => (f.min(total), false),
        };
        start = start.min(total);
        let mut end = total.min(start + PER_RESPONSE as u64);
        // not in the middle of a UTF-8 character: leave the unfinished one for the next request
        for back in 1..=3u64 {
            if end <= start || end - back < start {
                break;
            }
            let b = n.buf[(end - back - n.base) as usize];
            if b & 0xc0 == 0x80 {
                continue;
            }
            let need = if b >= 0xf0 { 4 } else if b >= 0xe0 { 3 } else if b >= 0xc0 { 2 } else { 1 };
            if need > back {
                end -= back;
            }
            break;
        }
        let data: Vec<u8> = n.buf.iter().skip((start - n.base) as usize).take((end - start) as usize).copied().collect();
        let klong = online && flags & 0x10 != 0;
        Ok(Output { next: end, data, lost, load: if klong { load } else { 0 }, online, klong, done: n.dones.front().cloned(), dones: n.dones.iter().cloned().collect(), sent: (n.last_id, n.last_state.unwrap_or(Sent::None)) })
    }

    pub fn nodes(&self) -> Vec<NodeInfo> {
        self.poll();
        let st = self.state.borrow();
        (0..NODES)
            .map(|i| {
                let (flags, load, online) = self.flags(i);
                NodeInfo { online, klong: online && flags & 0x10 != 0, load: if online { load } else { 0 }, flags: if online { flags } else { 0 }, info: st.nodes[i].info.clone(), selftest: st.nodes[i].selftest.clone() }
            })
            .collect()
    }

    /// The text protocol of `klong-bench --bridge`. `bench` is `None` when the board is busy
    /// evaluating (a request that arrives meanwhile): `advance` then does nothing.
    pub fn handle(&self, line: &str, bench: Option<&Bench>) -> String {
        let mut w = line.split_whitespace();
        let hex = |s: &str| -> Vec<u8> { (0..s.len() / 2).filter_map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).ok()).collect() };
        let tohex = |b: &[u8]| -> String { b.iter().map(|x| format!("{:02x}", x)).collect() };
        let err = |e: EvalError| format!("err {} {:?}", e.code(), e);
        match w.next() {
            Some("advance") => {
                if let (Some(b), Some(ms)) = (bench, w.next().and_then(|v| v.parse::<u64>().ok())) {
                    b.spin(ms * 1000);
                }
                self.poll();
                // (no answer: time moves while a board may be busy for a long time, and the answers to the
                // requests that come meanwhile must not wait behind it)
                String::new()
            }
            Some("eval") => {
                let node = w.next().and_then(|v| v.parse::<usize>().ok()).unwrap_or(99);
                let raw = w.next() == Some("raw");
                match self.eval_with(node, &hex(w.next().unwrap_or("")), raw) {
                    Ok(id) => format!("ok {}", id),
                    Err(e) => err(e),
                }
            }
            Some("interrupt") => match self.interrupt(w.next().and_then(|v| v.parse::<usize>().ok()).unwrap_or(99)) {
                Ok(id) => format!("ok {}", id),
                Err(e) => err(e),
            },
            Some("output") => {
                let node = w.next().and_then(|v| v.parse::<usize>().ok()).unwrap_or(99);
                let from = w.next().and_then(|v| v.parse::<u64>().ok());
                match self.output(node, from) {
                    Ok(o) => format!(
                        "ok node={} next={} load={} online={} klong={} lost={} done={} dones={} sent={}:{} data={}",
                        node,
                        o.next,
                        o.load,
                        o.online as u8,
                        o.klong as u8,
                        o.lost as u8,
                        o.done.map_or("-".to_string(), |d| format!("{}:{}", d.0, d.1)),
                        if o.dones.is_empty() { "-".to_string() } else { o.dones.iter().map(|d| format!("{}:{}", d.0, d.1)).collect::<Vec<_>>().join(",") },
                        o.sent.0,
                        o.sent.1.name(),
                        tohex(&o.data)
                    ),
                    Err(e) => err(e),
                }
            }
            Some("check") => match self.check_node(w.next().and_then(|v| v.parse::<usize>().ok()).unwrap_or(99), true) {
                Ok(()) => "ok".into(),
                Err(e) => err(e),
            },
            Some("nodes") => {
                let mut out = String::new();
                for (i, n) in self.nodes().iter().enumerate() {
                    out.push_str(&format!("node {} online={} klong={} load={} flags={} info={} selftest={}\n", i, n.online as u8, n.klong as u8, n.load, n.flags, tohex(n.info.as_bytes()), tohex(n.selftest.as_bytes())));
                }
                out.push_str("end");
                out
            }
            _ => "err 400 what?".into(),
        }
    }
}
