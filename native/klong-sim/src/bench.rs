//! N simulated boards behind the packet switch of [`crate::switch`], in one thread and one
//! virtual clock: the harness for IPC between nodes on a lane. While one board waits (inside a
//! synchronous call, say) the others are polled, as in `klong-rt/tests/mgmt.rs`.
//!
//! Time is virtual and only moves when somebody waits or idles: a board that evaluates costs
//! `idle_step_us` per 4096 evaluation steps (a stand-in for CPU speed, 200 microseconds by
//! default), a frame costs 1.389 ms on its lane, and a wait jumps to the next thing that
//! can happen (a frame arriving, a timer), at most 1 ms ahead.
use std::cell::{Cell, RefCell};
use std::rc::Rc;

use klong_core::{Config, Kind};
use klong_dgram::{LaneConfig, Outcome};
use klong_rt::link::Transport;
use klong_rt::node::{NodeConfig, NodeNet};
use klong_rt::selftest::SelfTest;
use klong_rt::{Platform, Runtime};

use crate::switch::Switch;

type Hook = Rc<RefCell<Option<Box<dyn FnMut()>>>>;

pub struct BenchPlatform {
    clock: Rc<Cell<u64>>,
    idle_step: u64,
    out: Rc<RefCell<Vec<u8>>>,
    transports: Vec<Box<dyn Transport>>,
    hook: Hook,
    net: NodeNet,
    switch: Switch,
    seed: u64,
}

impl BenchPlatform {
    fn peers(&self) {
        if let Ok(mut h) = self.hook.try_borrow_mut() {
            if let Some(f) = h.as_mut() {
                f();
            }
        }
    }
}

impl Platform for BenchPlatform {
    fn now_us(&self) -> u64 {
        self.clock.get()
    }
    fn wall_clock(&self) -> Option<f64> {
        None
    }
    fn idle(&mut self) {
        self.clock.set(self.clock.get() + self.idle_step);
        self.net.pump();
        self.peers();
    }
    fn console_read(&mut self, buf: &mut [u8]) -> usize {
        self.net.console_read(buf)
    }
    fn console_write(&mut self, bytes: &[u8]) {
        self.out.borrow_mut().extend_from_slice(bytes);
        self.net.console_write(bytes);
    }
    fn transports(&mut self) -> &mut [Box<dyn Transport>] {
        &mut self.transports
    }
    fn entropy(&mut self) -> u64 {
        self.seed = self.seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.seed ^ self.clock.get()
    }
    fn wait(&mut self, deadline_us: Option<u64>) {
        self.net.pump();
        self.peers();
        let now = self.clock.get();
        let mut to = now + 1000;
        if let Some(d) = deadline_us {
            to = to.min(d);
        }
        if let Some(e) = self.switch.next_event_us() {
            to = to.min(e.max(now + 1));
        }
        self.clock.set(to.max(now + 1));
    }
}

pub struct BenchNode {
    pub rt: Rc<RefCell<Runtime<BenchPlatform>>>,
    pub net: NodeNet,
    /// The boot self-test, when the bench was made with `selftest: true` (the firmware's own logic).
    pub selftest: RefCell<Option<SelfTest>>,
    /// A board that runs other firmware: it hears nothing and answers nothing on the lane.
    pub mute: Cell<bool>,
    out: Rc<RefCell<Vec<u8>>>,
}

impl BenchNode {
    /// Evaluate source text; the value formatted, or the error's kind and message.
    pub fn run(&self, text: &str) -> Result<String, (Kind, String)> {
        let mut rt = self.rt.borrow_mut();
        match rt.run(text, "<bench>") {
            Ok(Some(v)) => klong_core::fmt::format_value(&rt.interp.syms, &v, false).map_err(|e| (e.kind, e.message.to_string())),
            Ok(None) => Ok(String::new()),
            Err(e) => Err((e.kind, e.message.to_string())),
        }
    }
    pub fn ok(&self, text: &str) -> String {
        match self.run(text) {
            Ok(s) => s,
            Err(e) => panic!("{}: {:?}", text, e),
        }
    }
    /// What the board wrote to its console since the last call.
    pub fn out(&self) -> String {
        String::from_utf8_lossy(&std::mem::take(&mut *self.out.borrow_mut())).into_owned()
    }
}

#[derive(Clone, Copy)]
pub struct BenchOpts {
    pub idle_step_us: u64,
    pub seed: u64,
    pub node: Option<fn() -> NodeConfig>,
    /// Override the adapter's packet size and first retransmission timeout (for tuning runs).
    pub link_mtu: Option<usize>,
    pub rto_ms: Option<u32>,
    /// The supervisor pings every node once a second.
    pub pings: bool,
    /// Baud rate of every lane.
    pub baud: u32,
    /// Processor time of a node per 16-byte frame received and sent, microseconds. An estimate
    /// for a Cortex-M0+ at 125 MHz: the interrupt (about 250 cycles for 16 bytes), the parser and
    /// the table CRC-16 (about 600) and the datagram layer (about 300) are about 1200 cycles,
    /// 10 us; encoding and queueing a frame about 5 us.
    pub rx_cost_us: f64,
    pub tx_cost_us: f64,
    /// The firmware's receive ring, bytes, and how many bytes it lets wait for the transmitter
    /// (0: 12 ms of wire time, as the firmware does; see `tx_depth`).
    pub rx_ring: usize,
    pub tx_ring: usize,
    /// Run the boot self-test on every node.
    pub selftest: bool,
}

impl Default for BenchOpts {
    fn default() -> BenchOpts {
        BenchOpts { idle_step_us: 200, seed: 118, node: None, link_mtu: None, rto_ms: None, pings: true, baud: 115_200, rx_cost_us: 10.0, tx_cost_us: 5.0, rx_ring: 32 * 1024, tx_ring: 0, selftest: false }
    }
}

type Nodes = Rc<RefCell<Vec<Rc<BenchNode>>>>;
type Schedule = Rc<RefCell<Vec<(u64, Box<dyn FnOnce()>)>>>;

/// Run what has come due (a node that resets at a given moment, a cable that is cut).
fn run_due(schedule: &Schedule, now: u64) {
    loop {
        let job = {
            let Ok(mut s) = schedule.try_borrow_mut() else { return };
            match s.iter().position(|(t, _)| *t <= now) {
                Some(i) => s.remove(i).1,
                None => return,
            }
        };
        job();
    }
}

pub struct Bench {
    pub clock: Rc<Cell<u64>>,
    pub switch: Switch,
    nodes: Nodes,
    schedule: Schedule,
    /// The supervisor's lane, with the datagram layer on it (for the console channel and raw datagrams).
    pub sup: NodeNet,
    opts: BenchOpts,
    generation: Cell<u64>,
}

fn make_node(clock: &Rc<Cell<u64>>, switch: &Switch, opts: BenchOpts, index: usize, generation: u64, nodes: &Nodes, schedule: &Schedule) -> Rc<BenchNode> {
    let io = switch.port(index as u8, clock.clone());
    let c = clock.clone();
    let mut cfg = opts.node.map(|f| f()).unwrap_or_else(|| NodeConfig::for_baud(opts.baud));
    cfg.epoch_seed = (opts.seed ^ (index as u64 * 7919) ^ (generation * 104729)) as u16;
    if let Some(m) = opts.link_mtu {
        cfg.link_mtu = m;
    }
    if let Some(r) = opts.rto_ms {
        cfg.rto_ms = r;
    }
    // ids of the datagram layer differ per node and per boot, as a real node's would
    cfg.lane = LaneConfig { dgram: klong_dgram::Config { first_id: (opts.seed.wrapping_mul(31).wrapping_add(index as u64 * 977 + generation * 4001)) as u16, ..cfg.lane.dgram }, ..cfg.lane };
    let net = NodeNet::new(Box::new(io), Box::new(move || c.get()), cfg);
    let out = Rc::new(RefCell::new(Vec::new()));
    let hook: Hook = Rc::new(RefCell::new(None));
    let platform = BenchPlatform {
        clock: clock.clone(),
        idle_step: opts.idle_step_us,
        out: out.clone(),
        transports: vec![net.transport()],
        hook: hook.clone(),
        net: net.clone(),
        switch: switch.clone(),
        seed: opts.seed ^ (index as u64 + 1) * 0x9e37_79b9 ^ generation << 20,
    };
    let rt = Rc::new(RefCell::new(Runtime::new(platform, Config::default(), None)));
    let weak = Rc::downgrade(nodes);
    let schedule = schedule.clone();
    let clock = clock.clone();
    *hook.borrow_mut() = Some(Box::new(move || {
        run_due(&schedule, clock.get());
        let Some(nodes) = weak.upgrade() else { return };
        let list: Vec<Rc<BenchNode>> = match nodes.try_borrow() {
            Ok(l) => l.clone(),
            Err(_) => return,
        };
        for (j, o) in list.iter().enumerate() {
            if j == index {
                continue;
            }
            if o.mute.get() {
                continue;
            }
            // a board's main loop calls Platform::idle every turn, which pumps its lane
            o.net.pump();
            if let Ok(mut rt) = o.rt.try_borrow_mut() {
                rt.poll_events();
            }
        }
    }));
    Rc::new(BenchNode { rt, net, out, selftest: RefCell::new(if opts.selftest { Some(SelfTest::new()) } else { None }), mute: Cell::new(false) })
}

impl Bench {
    /// `n` nodes behind the switch, started and numbered by the supervisor's first PING.
    pub fn new(n: usize, opts: BenchOpts) -> Bench {
        let clock = Rc::new(Cell::new(0));
        let switch = Switch::new(n, opts.seed);
        switch.set_baud(opts.baud);
        switch.set_cpu(opts.rx_cost_us, opts.tx_cost_us);
        switch.set_ring(opts.rx_ring, if opts.tx_ring == 0 { klong_dgram::tx_depth(opts.baud) } else { opts.tx_ring });
        let nodes: Nodes = Rc::new(RefCell::new(Vec::new()));
        let schedule: Schedule = Rc::new(RefCell::new(Vec::new()));
        for i in 0..n {
            let node = make_node(&clock, &switch, opts, i, 0, &nodes, &schedule);
            nodes.borrow_mut().push(node);
        }
        let c = clock.clone();
        let sup = NodeNet::new(
            Box::new(switch.port(klong_dgram::frame::SUPERVISOR, clock.clone())),
            Box::new(move || c.get()),
            NodeConfig { lane: LaneConfig { node: Some(klong_dgram::frame::SUPERVISOR), supervisor_duties: false, ..LaneConfig::default() }, ..NodeConfig::default() },
        );
        sup.keep_log(256);
        let b = Bench { clock, switch, nodes, schedule, sup, opts, generation: Cell::new(0) };
        b.switch.set_pings(opts.pings);
        if opts.pings {
            b.spin_ms(60);
            for i in 0..n {
                assert_eq!(b.node(i).net.node(), Some(i as u8), "node {} was not numbered", i);
            }
        }
        b
    }

    pub fn node(&self, i: usize) -> Rc<BenchNode> {
        self.nodes.borrow()[i].clone()
    }

    pub fn len(&self) -> usize {
        self.nodes.borrow().len()
    }

    pub fn now_us(&self) -> u64 {
        self.clock.get()
    }

    /// A board that lost power and starts again: a new runtime, no state, no node number yet.
    pub fn reset_node(&self, i: usize) {
        self.generation.set(self.generation.get() + 1);
        self.switch.clear_lane(i as u8);
        let node = make_node(&self.clock, &self.switch, self.opts, i, self.generation.get(), &self.nodes, &self.schedule);
        self.nodes.borrow_mut()[i] = node;
    }

    /// Reset node `i` at virtual time `at_us`, wherever the others are then (inside a call that waits, say).
    pub fn reset_node_at(&self, i: usize, at_us: u64) {
        let (clock, switch, opts, nodes, schedule) = (self.clock.clone(), self.switch.clone(), self.opts, self.nodes.clone(), self.schedule.clone());
        let gen = self.generation.get() + 1;
        self.generation.set(gen);
        let job: Box<dyn FnOnce()> = Box::new(move || {
            switch.clear_lane(i as u8);
            let node = make_node(&clock, &switch, opts, i, gen, &nodes, &schedule);
            nodes.borrow_mut()[i] = node;
        });
        self.schedule.borrow_mut().push((at_us, job));
    }

    /// Run `f` at virtual time `at_us`.
    pub fn at(&self, at_us: u64, f: impl FnOnce() + 'static) {
        self.schedule.borrow_mut().push((at_us, Box::new(f)));
    }

    /// Run every board's event loop (and the supervisor's lane) for `us` of virtual time.
    pub fn spin(&self, us: u64) {
        let until = self.clock.get() + us;
        while self.clock.get() < until {
            run_due(&self.schedule, self.clock.get());
            let mut did = false;
            let list: Vec<Rc<BenchNode>> = self.nodes.borrow().clone();
            for n in &list {
                if n.mute.get() {
                    continue;
                }
                // what a board's main loop does every turn: Platform::idle, which pumps the lane
                n.net.pump();
                if let Ok(mut rt) = n.rt.try_borrow_mut() {
                    did |= rt.poll_events();
                    if let Ok(mut st) = n.selftest.try_borrow_mut() {
                        if let Some(st) = st.as_mut() {
                            did |= st.step(&mut rt, &n.net);
                        }
                    }
                }
            }
            self.sup.pump();
            let now = self.clock.get();
            let mut to = now + if did { self.opts.idle_step_us.max(1) } else { 1000 };
            if let Some(e) = self.switch.next_event_us() {
                to = to.min(e.max(now + 1));
            }
            self.clock.set(to.min(until).max(now + 1));
        }
    }

    pub fn spin_ms(&self, ms: u64) {
        self.spin(ms * 1000);
    }

    /// Spin until `done` says so, at most `limit_us`; the virtual microseconds it took, or `None`.
    pub fn spin_until(&self, limit_us: u64, mut done: impl FnMut() -> bool) -> Option<u64> {
        let t0 = self.clock.get();
        while self.clock.get() - t0 < limit_us {
            if done() {
                return Some(self.clock.get() - t0);
            }
            self.spin(500);
        }
        None
    }
}

pub fn outcome_name(o: Outcome) -> &'static str {
    match o {
        Outcome::Sent => "sent",
        Outcome::Acked => "acked",
        Outcome::Refused => "refused",
        Outcome::Failed => "failed",
    }
}
