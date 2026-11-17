//! The management service (`klongpy/lib/mgmt.kg`, part of the image) fronting nodes
//! over links that lose, repeat and reorder packets and over a serial line that drops and
//! damages bytes. Three runtimes live in one thread on a simulated clock (the harness of
//! `events.rs`): while one waits, the others are polled. The same service over TCP between
//! processes is `tests/test_native_mgmt.py`.
#![cfg(all(feature = "ipc", feature = "timers", feature = "lib"))]
use std::cell::{Cell, RefCell};
use std::rc::Rc;

use klong_core::{Config, Kind};
use klong_rt::link::Transport;
use klong_rt::memlink::{Faults, Hub, MemKind, MemTransport};
use klong_rt::{Platform, Runtime};

type Hook = Rc<RefCell<Option<Box<dyn FnMut()>>>>;

struct Sim {
    clock: Rc<Cell<u64>>,
    idle_step: u64,
    out: Rc<RefCell<Vec<u8>>>,
    transports: Vec<Box<dyn Transport>>,
    hook: Hook,
}

impl Sim {
    fn peers(&self) {
        if let Ok(mut h) = self.hook.try_borrow_mut() {
            if let Some(f) = h.as_mut() {
                f();
            }
        }
    }
}

impl Platform for Sim {
    fn now_us(&self) -> u64 {
        self.clock.get()
    }
    fn wall_clock(&self) -> Option<f64> {
        None
    }
    fn idle(&mut self) {
        self.clock.set(self.clock.get() + self.idle_step);
        self.peers();
    }
    fn console_read(&mut self, _: &mut [u8]) -> usize {
        0
    }
    fn console_write(&mut self, bytes: &[u8]) {
        self.out.borrow_mut().extend_from_slice(bytes);
    }
    fn transports(&mut self) -> &mut [Box<dyn Transport>] {
        &mut self.transports
    }
    fn wait(&mut self, deadline_us: Option<u64>) {
        self.peers();
        let now = self.clock.get();
        self.clock.set(deadline_us.map_or(now + 1000, |d| d.max(now + 1)));
    }
}

struct Node {
    rt: Rc<RefCell<Runtime<Sim>>>,
    out: Rc<RefCell<Vec<u8>>>,
    hook: Hook,
}

/// `faults` are the packet link's; the serial line drops 0.05% of its bytes and damages 0.1%.
fn node(clock: &Rc<Cell<u64>>, hub: &Hub, idle_step: u64, faults: Faults) -> Node {
    let serial = Faults { loss: 500, dup: 0, reorder: 0, flip: 1000 };
    let out = Rc::new(RefCell::new(Vec::new()));
    let hook: Hook = Rc::new(RefCell::new(None));
    let transports: Vec<Box<dyn Transport>> = vec![
        Box::new(MemTransport::new(hub, "tcp", MemKind::Stream, Faults::default())),
        Box::new(MemTransport::new(hub, "lossy", MemKind::Packet { mtu: 32, reliable: false, rto_ms: 20 }, faults)),
        Box::new(MemTransport::new(hub, "uart", MemKind::Serial { mtu: 240, rto_ms: 50 }, serial)),
    ];
    let sim = Sim { clock: clock.clone(), idle_step, out: out.clone(), transports, hook: hook.clone() };
    Node { rt: Rc::new(RefCell::new(Runtime::new(sim, Config::default(), None))), out, hook }
}

impl Node {
    fn run(&self, text: &str) -> Result<String, (Kind, String)> {
        let mut rt = self.rt.borrow_mut();
        match rt.run(text, "<test>") {
            Ok(Some(v)) => klong_core::fmt::format_value(&rt.interp.syms, &v, false).map_err(|e| (e.kind, e.message.to_string())),
            Ok(None) => Ok(String::new()),
            Err(e) => Err((e.kind, e.message.to_string())),
        }
    }
    fn ok(&self, text: &str) -> String {
        match self.run(text) {
            Ok(s) => s,
            Err(e) => panic!("{}: {:?}", text, e),
        }
    }
    fn out(&self) -> String {
        String::from_utf8_lossy(&std::mem::take(&mut *self.out.borrow_mut())).into_owned()
    }
    /// Run the event loop until the clock reaches `until_us`.
    fn spin(&self, clock: &Cell<u64>, until_us: u64) {
        while clock.get() < until_us {
            let mut rt = self.rt.borrow_mut();
            if !rt.poll_events() {
                let now = clock.get();
                if rt.events_active() {
                    rt.events_wait();
                }
                if clock.get() == now {
                    clock.set(now + 100);
                }
                if clock.get() > until_us {
                    clock.set(until_us);
                }
            }
        }
    }
}

const NODE: &str = "sq::{x*x};add::{x+y};idf::{x};total::{+/x};fail::{1+\"a\"};raise::{.err(x;\"boom\")}";

/// A management node and three nodes behind it: n1 on a lossy packet link, n2 on a serial
/// line, n3 on a stream. Returns (management, nodes, clock, hub).
fn cluster(faults: Faults) -> (Node, Vec<Node>, Rc<Cell<u64>>, Hub) {
    let clock = Rc::new(Cell::new(0));
    let hub = Hub::new(108);
    let m = node(&clock, &hub, 1, faults);
    let nodes: Vec<Node> = (0..3).map(|_| node(&clock, &hub, 10_000, faults)).collect();
    // whoever waits polls everybody else (a runtime that is busy itself is skipped)
    let all: Vec<Rc<RefCell<Runtime<Sim>>>> = std::iter::once(&m).chain(nodes.iter()).map(|n| n.rt.clone()).collect();
    for n in std::iter::once(&m).chain(nodes.iter()) {
        let others: Vec<_> = all.iter().filter(|r| !Rc::ptr_eq(r, &n.rt)).cloned().collect();
        *n.hook.borrow_mut() = Some(Box::new(move || {
            for o in &others {
                if let Ok(mut o) = o.try_borrow_mut() {
                    o.poll_events();
                }
            }
        }));
    }
    for (n, addr) in nodes.iter().zip(["\"lossy:n1\"", "\"uart:2\"", "7003"]) {
        n.ok(NODE);
        assert_eq!(n.ok(&format!(".srv({})", addr)), "1");
    }
    m.ok(".l(\"mgmt\")");
    assert_eq!(m.ok("mgmt.add(\"n1\";\"lossy:n1\");mgmt.add(\"n2\";\"uart:2\");mgmt.add(\"n3\";\"localhost:7003\");mgmt.names()"), "[\"n1\" \"n2\" \"n3\"]");
    (m, nodes, clock, hub)
}

static ONE: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn states(m: &Node) -> String {
    m.ok("{(x@1)?\"state\"}'mgmt.status()")
}

#[test]
fn relay_fan_out_and_deploy_over_lossy_and_serial_links() {
    let _one = ONE.lock().unwrap_or_else(|e| e.into_inner());
    // 5% of packets lost, repeated and reordered; 0.1% of serial bytes dropped, 0.1% damaged
    let (m, nodes, _clock, hub) = cluster(Faults { loss: 50_000, dup: 50_000, reorder: 50_000, flip: 0 });
    assert_eq!(states(&m), "[\"up\" \"up\" \"up\"]");
    for name in ["n1", "n2", "n3"] {
        assert_eq!(m.ok(&format!("mgmt.call(\"{name}\";[:sq]@0;[7])")), "49");
        assert_eq!(m.ok(&format!("mgmt.call(\"{name}\";\"add\";[3 4])")), "7");
        assert_eq!(m.ok(&format!("big::!5000;r::mgmt.call(\"{name}\";[:idf]@0;,big);(r~big),#r")), "[1 5000]");
        // an error on the node keeps its kind and gains the node's name
        let e = m.run(&format!("mgmt.call(\"{name}\";[:fail]@0;[1])"));
        assert_eq!(e, Err((Kind::Type, format!("{name}: cannot apply + (Plus) to integer and string"))));
        for kind in [Kind::Syntax, Kind::Name, Kind::Arity, Kind::Shape, Kind::Range, Kind::System, Kind::Memory, Kind::Internal, Kind::Remote] {
            let e = m.run(&format!("mgmt.call(\"{name}\";[:raise]@0;,\"{}\")", kind.name()));
            assert_eq!(e, Err((kind, format!("{name}: boom"))));
        }
    }
    assert_eq!(m.run("mgmt.call(\"n9\";[:sq]@0;[7])"), Err((Kind::Name, "mgmt: no such node: n9".into())));
    assert_eq!(m.ok("mgmt.all([:sq]@0;[7])"), ":{[\"n1\" 49] [\"n2\" 49] [\"n3\" 49]}");
    assert_eq!(m.ok("mgmt.all([:total]@0;,!2001)"), ":{[\"n1\" 2001000] [\"n2\" 2001000] [\"n3\" 2001000]}");
    nodes[1].ok("sq::{x+\"a\"}");
    assert_eq!(
        m.ok("mgmt.all([:sq]@0;[7])"),
        ":{[\"n1\" 49] [\"n2\" :{[\"error\" \"type\"] [\"node\" \"n2\"] [\"message\" \"n2: cannot apply + (Plus) to integer and string\"]}] [\"n3\" 49]}"
    );
    // a source text of 6000 characters arrives whole (these nodes have no storage: it is evaluated)
    assert_eq!(m.ok("src::\"deployed::{x+1000};#\"\"\",(6000#\"klong \"),\"\"\"\";#src"), "6022");
    assert_eq!(m.ok("mgmt.all(\".E\";,src)"), ":{[\"n1\" 6000] [\"n2\" 6000] [\"n3\" 6000]}");
    assert_eq!(m.ok("mgmt.all([:deployed]@0;[1])"), ":{[\"n1\" 1001] [\"n2\" 1001] [\"n3\" 1001]}");
    let (sent, lost) = hub.counters();
    println!("management over lossy (mtu 32) and serial links: {} packets and bytes, {} lost or damaged", sent, lost);
    assert!(lost > 0);
    assert_eq!(m.out(), "");
}

#[test]
fn liveness_over_packet_links_and_a_link_that_dies() {
    let _one = ONE.lock().unwrap_or_else(|e| e.into_inner());
    let (m, _nodes, clock, hub) = cluster(Faults { loss: 20_000, dup: 20_000, reorder: 20_000, flip: 0 });
    assert_eq!(m.ok("mgmt.start(0.5)"), "1");
    m.spin(&clock, clock.get() + 3_000_000);
    assert_eq!(states(&m), "[\"up\" \"up\" \"up\"]");
    assert_eq!(m.ok("{((x@1)?\"age\")<1}'mgmt.status()"), "[1 1 1]");
    // every link that is open now loses everything from here on (a link opened later is a new
    // one and works); the stream to n3 is not touched
    hub.set_faults(Faults { loss: 1_000_000, ..Faults::default() });
    m.spin(&clock, clock.get() + 1_500_000);
    assert_eq!(states(&m), "[\"down\" \"down\" \"up\"]", "down within the ping interval and one more");
    // a fan-out returns n3's result and a system error for each of the others; the two dead
    // links cost their timeout once, not twice
    m.ok("mgmt.timeout(\"n1\";2);mgmt.timeout(\"n2\";2)");
    let t0 = clock.get();
    assert_eq!(m.ok("r::mgmt.all([:sq]@0;[7]);(,(r?\"n1\")?\"error\"),(,(r?\"n2\")?\"error\"),,r?\"n3\""), "[\"system\" \"system\" 49]");
    let took = (clock.get() - t0) / 1000;
    println!("fan-out with two dead links, timeout 2 s each: {} ms", took);
    assert!((1900..2500).contains(&took), "{} ms", took);
    assert_eq!(m.ok("{(x@1)?\"h\"}'mgmt.nodes")[..5], *"[0 0 ");
    // the check connects them again, on new links
    m.spin(&clock, clock.get() + 2_000_000);
    assert_eq!(states(&m), "[\"up\" \"up\" \"up\"]");
    assert_eq!(m.ok("mgmt.all([:sq]@0;[7])"), ":{[\"n1\" 49] [\"n2\" 49] [\"n3\" 49]}");
    assert_eq!(m.ok("mgmt.stop()"), "1");
    // the runtime logs each request that was out on a link when it failed; the service adds nothing
    let log = m.out();
    assert!(!log.is_empty() && log.lines().all(|l| l.starts_with(".async: remote call failed: call timed out after 2s: ")), "{}", log);
}

/// A call of the management library uses seven functions kept as text (about 8 KB of trees on the host): they are parsed once, whatever the
/// number of calls in a program and of programs (measured on the boards: ten calls in one expression took 242 ms, 24 a call, when a cache of
/// 6 KB held less than one call needs and every call parsed all seven again; 96 ms with a cache that held them).
#[test]
fn a_call_of_the_library_parses_each_function_it_uses_once() {
    let _one = ONE.lock().unwrap_or_else(|e| e.into_inner());
    let (m, _nodes, _clock, _hub) = cluster(Faults::default());
    let st = |m: &Node| m.rt.borrow().interp.lazy_stats();
    let s0 = st(&m);
    m.ok("mgmt.call(\"n3\";[:sq]@0;[7])");
    let s1 = st(&m);
    assert!(s1.0 - s0.0 >= 5 && s1.0 - s0.0 <= 12, "a call parses {:?}", (s1.0 - s0.0, s1.1 - s0.1));
    // a second program, then ten calls in one expression, then another program with three: nothing is parsed again, nothing is let go
    m.ok("mgmt.call(\"n3\";[:sq]@0;[7])");
    assert_eq!(m.ok("{mgmt.call(\"n3\";[:sq]@0;[7])}'!10"), "[49 49 49 49 49 49 49 49 49 49]");
    m.ok("mgmt.call(\"n3\";[:sq]@0;[7]);mgmt.call(\"n3\";\"add\";[3 4]);mgmt.call(\"n3\";[:sq]@0;[7])");
    let s2 = st(&m);
    assert_eq!((s2.0 - s1.0, s2.2 - s1.2), (0, 0), "later calls parsed {:?} functions and let {:?} go", s2.0 - s1.0, s2.2 - s1.2);
}
