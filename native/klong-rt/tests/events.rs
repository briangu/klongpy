//! Ticket 106: timers and IPC with a simulated clock and links in memory. Two runtimes live
//! in one thread; while one waits (`Platform::wait`, `Platform::idle`) the other is polled.
#![cfg(all(feature = "ipc", feature = "timers"))]
use std::cell::{Cell, RefCell};
use std::rc::Rc;

use klong_core::{Config, Kind};
use klong_rt::link::Transport;
use klong_rt::memlink::{Faults, Hub, MemKind, MemTransport};
use klong_rt::{Platform, Runtime};

type Hook = Rc<RefCell<Option<Box<dyn FnMut()>>>>;

struct Sim {
    clock: Rc<Cell<u64>>,
    /// Microseconds an `idle` call takes (an evaluation that runs for a while).
    idle_step: u64,
    out: Rc<RefCell<Vec<u8>>>,
    transports: Vec<Box<dyn Transport>>,
    hook: Hook,
    hub: Hub,
    /// Ctrl-C arrives on the console at this time (0: never).
    ctrl_c_at: Rc<Cell<u64>>,
}

impl Sim {
    fn peer(&self) {
        // (the peer's own waits end up here too: `try_borrow_mut` keeps them out)
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
        self.peer();
    }
    fn console_read(&mut self, buf: &mut [u8]) -> usize {
        let at = self.ctrl_c_at.get();
        if at != 0 && self.clock.get() >= at && !buf.is_empty() {
            self.ctrl_c_at.set(0);
            buf[0] = 0x03;
            return 1;
        }
        0
    }
    fn console_write(&mut self, bytes: &[u8]) {
        self.out.borrow_mut().extend_from_slice(bytes);
    }
    fn transports(&mut self) -> &mut [Box<dyn Transport>] {
        &mut self.transports
    }
    fn wait(&mut self, deadline_us: Option<u64>) {
        self.peer();
        let now = self.clock.get();
        self.clock.set(deadline_us.map_or(now + 1000, |d| d.max(now + 1)));
    }
    fn exit(&mut self, _: i32) {
        self.hub.cut(); // the node is gone
    }
}

struct Node {
    ctrl_c_at: Rc<Cell<u64>>,
    rt: Rc<RefCell<Runtime<Sim>>>,
    out: Rc<RefCell<Vec<u8>>>,
    hook: Hook,
}

static ONE: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn node(clock: &Rc<Cell<u64>>, hub: &Hub, idle_step: u64, faults: Faults) -> Node {
    let out = Rc::new(RefCell::new(Vec::new()));
    let hook: Hook = Rc::new(RefCell::new(None));
    let transports: Vec<Box<dyn Transport>> = vec![
        Box::new(MemTransport::new(hub, "tcp", MemKind::Stream, Faults::default())),
        Box::new(MemTransport::new(hub, "lossy", MemKind::Packet { mtu: 32, reliable: false, rto_ms: 20 }, faults)),
        Box::new(MemTransport::new(hub, "tiny", MemKind::Packet { mtu: 4, reliable: false, rto_ms: 20 }, faults)),
        Box::new(MemTransport::new(hub, "fabric", MemKind::Packet { mtu: 64, reliable: true, rto_ms: 20 }, Faults::default())),
        Box::new(MemTransport::new(hub, "uart", MemKind::Serial { mtu: 240, rto_ms: 50 }, faults)),
    ];
    let ctrl_c_at = Rc::new(Cell::new(0));
    let sim = Sim { clock: clock.clone(), idle_step, out: out.clone(), transports, hook: hook.clone(), hub: hub.clone(), ctrl_c_at: ctrl_c_at.clone() };
    Node { ctrl_c_at, rt: Rc::new(RefCell::new(Runtime::new(sim, Config::default(), None))), out, hook }
}

/// Two nodes on one hub and one clock; each is polled while the other waits.
fn pair(faults: Faults) -> (Node, Node, Rc<Cell<u64>>, Hub) {
    let clock = Rc::new(Cell::new(0));
    let hub = Hub::new(7);
    let a = node(&clock, &hub, 1, faults);
    let b = node(&clock, &hub, 10_000, faults);
    for (x, y) in [(&a, &b), (&b, &a)] {
        let other = y.rt.clone();
        *x.hook.borrow_mut() = Some(Box::new(move || {
            if let Ok(mut o) = other.try_borrow_mut() {
                o.poll_events();
            }
        }));
    }
    (a, b, clock, hub)
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

fn lock() -> std::sync::MutexGuard<'static, ()> {
    ONE.lock().unwrap_or_else(|e| e.into_inner())
}

#[test]
fn a_timer_fires_at_fractional_intervals_until_its_callback_returns_0() {
    let _one = lock();
    let (a, _b, clock, _) = pair(Faults::default());
    a.ok("n::0;at::[];cb::{n::n+1;at::at,_(.pc()*1000);n<4};th::.timer(\"t\";0.25;cb)");
    assert_eq!(a.ok("n"), "0");
    a.spin(&clock, 240_000);
    assert_eq!(a.ok("n"), "0");
    a.spin(&clock, 2_000_000);
    // four calls, a quarter of a second apart (the simulated clock is exact to about 0.1 ms)
    assert_eq!(a.ok("n"), "4");
    assert_eq!(a.ok("at"), "[250 500 750 1000]");
    assert!(!a.rt.borrow().events_active());
    // the handle of a timer that has stopped cancels nothing
    assert_eq!(a.ok(".timerc(th)"), "0");
}

#[test]
fn a_timer_is_cancelled_by_timerc_and_by_a_failing_callback() {
    let _one = lock();
    let (a, _b, clock, _) = pair(Faults::default());
    a.ok("n::0;cb::{n::n+1;1};th::.timer(\"t\";0.1;cb)");
    a.spin(&clock, 350_000);
    assert_eq!(a.ok("n"), "3");
    assert_eq!(a.ok(".timerc(th)"), "1");
    assert_eq!(a.ok(".timerc(th)"), "0");
    assert_eq!(a.ok(".timerc(5)"), "0");
    a.spin(&clock, 900_000);
    assert_eq!(a.ok("n"), "3");
    // a callback that fails: reported on the error channel, the timer is gone, others go on
    a.out();
    a.ok("m::0;k::0;bad::{m::m+1;1+\"a\"};good::{k::k+1;1};.timer(\"bad one\";0.1;bad);g::.timer(\"good\";0.1;good)");
    a.spin(&clock, 1_350_000);
    assert_eq!(a.ok("m"), "1");
    assert_eq!(a.ok("k"), "4");
    let log = a.out();
    assert!(log.contains("timer bad one callback failed; cancelling timer") && log.contains("type error"), "{}", log);
    // a timer cancels itself and starts another from its callback
    a.ok(".timerc(g);j::0;once::{j::j+1;.timer(\"next\";0.05;{j::j+10;0});0};.timer(\"once\";0.05;once)");
    a.spin(&clock, 2_000_000);
    assert_eq!(a.ok("j"), "11");
    assert!(!a.rt.borrow().events_active());
    // arguments
    assert_eq!(a.run(".timer(\"t\";-1;cb)").map_err(|e| e.0), Err(Kind::Range));
    assert_eq!(a.run(".timer(\"t\";\"x\";cb)").map_err(|e| e.0), Err(Kind::Type));
    assert_eq!(a.run(".timer(\"t\";1;5)").map_err(|e| e.0), Err(Kind::Type));
}

#[test]
fn an_interval_of_0_fires_on_every_pass() {
    let _one = lock();
    let (a, _b, _clock, _) = pair(Faults::default());
    a.ok("n::0;cb::{n::n+1;n<5};.timer(\"t\";0;cb)");
    for _ in 0..10 {
        a.rt.borrow_mut().poll_events();
    }
    assert_eq!(a.ok("n"), "5");
}

const SERVER: &str = "sq::{x*x};add::{x+y};nil::{42};v::!10;d:::{[1 2] [3 4]};fail::{1+\"a\"};\
    wait::{[t];t::.pc()+x;{.pc()<t}{x}:~0;x};opened::0;closed::0;.srv.o::{opened::opened+1};.srv.c::{closed::closed+1}";

fn served(addr: &str, faults: Faults) -> (Node, Node, Rc<Cell<u64>>, Hub) {
    let (a, b, clock, hub) = pair(faults);
    b.ok(SERVER);
    assert_eq!(b.ok(&format!(".srv({})", addr)), "1");
    (a, b, clock, hub)
}

#[test]
fn a_blocked_call_holds_timers_back_and_they_fire_once_afterwards() {
    let _one = lock();
    let (a, b, clock, _) = served("7001", Faults::default());
    a.ok("c::.cli(7001);n::0;at::[];cb::{n::n+1;at::at,_(.pc()*100);1};th::.timer(\"t\";0.1;cb)");
    let t0 = clock.get();
    // the peer computes for half a second; five intervals pass, and nothing fires meanwhile
    assert_eq!(a.ok("r::c(:wait,0.5);n"), "0");
    assert_eq!(a.ok("r"), "0.5");
    assert!(clock.get() - t0 >= 500_000);
    a.rt.borrow_mut().poll_events();
    a.rt.borrow_mut().poll_events();
    assert_eq!(a.ok("n"), "1", "one late call, not five");
    a.spin(&clock, t0 + 850_000);
    // then back on the multiples of the interval counted from the start
    let at = a.ok("at");
    let ticks: Vec<i64> = at.trim_matches(|c| c == '[' || c == ']').split(' ').map(|s| s.parse().expect("number")).collect();
    assert_eq!(ticks.len(), 4, "{}", at);
    assert!(ticks[0] >= 50 && ticks[1] == 60 && ticks[2] == 70 && ticks[3] == 80, "{}", at);
    a.ok(".timerc(th)");
    drop(b);
}

#[test]
fn calls_lookups_remote_dictionaries_and_errors_over_a_stream() {
    let _one = lock();
    let (a, b, _clock, _) = served("7002", Faults::default());
    a.ok("c::.cli(7002)");
    assert_eq!(a.ok("c(\"1+2\")"), "3");
    assert_eq!(a.ok("c(:v)"), "[0 1 2 3 4 5 6 7 8 9]");
    assert_eq!(a.ok("c(:sq,7)"), "49");
    assert_eq!(a.ok("c(:add,3,4)"), "7");
    assert_eq!(a.ok("c([:nil])"), "42");
    assert_eq!(a.ok("f::c(:sq);f(9)"), "81");
    assert_eq!(a.ok("g::c(:add);g(1;2)"), "3");
    assert_eq!(a.ok("h::c(:nil);h()"), "42");
    assert_eq!(a.ok("f'[1 2 3]"), "[1 4 9]");
    // data goes and comes back as itself
    for v in ["[1 2 3]", "[1.5 2.5]", "\"text\"", ":sym", "[:a :b]", "[1 \"two\" [3.0 :four]]", "0cx", ":{[1 2] [\"k\" :v]}", "[]", "1.5", "-7"] {
        a.ok(&format!("t::{}", v));
        assert_eq!(a.ok("d::.clid(c);d,:tt,,t;(d?:tt)~t"), "1", "{}", v);
    }
    // a remote dictionary
    assert_eq!(a.ok("d::.clid(c);d,[:zz 42];d?:zz"), "42");
    assert_eq!(b.ok("zz"), "42");
    assert_eq!(a.ok("e::.clid(7002);e?:v"), "[0 1 2 3 4 5 6 7 8 9]");
    // errors of the peer are errors of the same kind here
    assert_eq!(a.run("c(:fail,1)").map_err(|e| e.0), Err(Kind::Type));
    assert_eq!(a.run("c(\"1+\")").map_err(|e| e.0), Err(Kind::Syntax));
    assert_eq!(a.run("c(:nosuch)"), Err((Kind::Name, "symbol not found: nosuch".into())));
    assert_eq!(a.run("c(\"[1 2 3]@9\")").map_err(|e| e.0), Err(Kind::Range));
    // and the connection is still good
    assert_eq!(a.ok("c(\"2+2\")"), "4");
    // hooks, close
    assert_eq!(b.ok("opened,closed"), "[2 0]");
    assert_eq!(a.ok(".clic(c)"), "1");
    assert_eq!(a.ok(".clic(c)"), "0");
    assert_eq!(a.run("c(\"1\")").map_err(|e| e.0), Err(Kind::System));
    b.rt.borrow_mut().poll_events();
    b.rt.borrow_mut().poll_events();
    assert_eq!(b.ok("opened,closed"), "[2 1]");
    // nobody listens there
    assert_eq!(a.run(".cli(7999)").map_err(|e| e.0), Err(Kind::System));
    assert_eq!(a.run(".cli(\"nowhere:notaport\")").map_err(|e| e.0), Err(Kind::Type));
    // .srv(0) stops the server and drops its clients
    assert_eq!(b.ok(".srv(0)"), "1");
    assert_eq!(b.ok(".srv(0)"), "0");
    assert_eq!(a.run("e?:v").map_err(|e| e.0), Err(Kind::System));
}

#[test]
fn async_calls_run_their_callback_from_the_event_loop() {
    let _one = lock();
    let (a, _b, clock, _) = served("7003", Faults::default());
    a.ok("c::.cli(7003);got::[];cb::{got::got,x};f::c(:sq)");
    assert_eq!(a.ok("af::.async(f;cb);af(5)"), "1");
    assert_eq!(a.ok("got"), "[]", "the callback has not run inside the program that asked");
    a.spin(&clock, clock.get() + 10_000);
    assert_eq!(a.ok("got"), "[25]");
    // the connection itself, and a local function
    a.ok("ac::.async(c;cb);ac(\"100+1\");al::.async({x*3};cb);al(7)");
    a.spin(&clock, clock.get() + 10_000);
    assert_eq!(a.ok("got@<got"), "[21 25 101]");
    // a sync call made while an async one is out still gets its own answer
    assert_eq!(a.ok("af(6);c(\"1+1\")"), "2");
    a.spin(&clock, clock.get() + 10_000);
    assert_eq!(a.ok("got@<got"), "[21 25 36 101]");
    assert_eq!(a.run(".async(1;cb)").map_err(|e| e.0), Err(Kind::Type));
}

#[test]
fn authentication_hook_accepts_and_refuses() {
    let _one = lock();
    let (a, b, _clock, _) = served("7004", Faults::default());
    b.ok(".srv.auth::{(x~\"ann\")&y~\"pw\"}");
    assert_eq!(a.run(".cli(\"localhost:7004:bob:nope\")").map_err(|e| e.0), Err(Kind::System));
    assert_eq!(a.run(".cli(7004)").map_err(|e| e.0), Err(Kind::System));
    assert_eq!(a.ok("c::.cli(\"localhost:7004:ann:pw\");c(\"1+1\")"), "2");
    assert_eq!(b.ok("opened"), "1");
}

#[test]
fn a_call_times_out_and_takes_the_connection_with_it() {
    let _one = lock();
    let (a, b, clock, _) = served("7005", Faults::default());
    a.ok("c::.cli(7005)");
    assert_eq!(a.ok(".clit(c;0.2)"), "1");
    // the peer does not answer (here: nobody runs it)
    let hook = a.hook.borrow_mut().take();
    let t0 = clock.get();
    let r = a.run("c(:wait,1)");
    *a.hook.borrow_mut() = hook;
    assert!(matches!(&r, Err((Kind::System, m)) if m.contains("timed out")), "{:?}", r);
    let took = clock.get() - t0;
    assert!((200_000..260_000).contains(&took), "{}", took);
    assert_eq!(a.run("c(\"1\")").map_err(|e| e.0), Err(Kind::System));
    // a new connection works, and 0 means no limit
    assert_eq!(a.ok("c::.cli(7005);.clit(c;0);c(:wait,0.4)"), "0.4");
    a.ok(".clic(c)");
    // A callback to a node that is itself waiting cannot be served in one thread (the real
    // thing is tested between processes). The server's own call times out, which closes the
    // connection (a late answer could not be told from the next one), and the client's
    // call fails with it.
    b.ok("who::0;.srv.o::{who::x};back::{.clit(who;0.1);who(\"1+1\")}");
    a.ok("c::.cli(7005)");
    let r = a.run("c(:back,0)");
    assert!(matches!(&r, Err((Kind::System, m)) if m.contains("closed")), "{:?}", r);
}

#[test]
fn ctrl_c_on_the_console_ends_a_blocked_call() {
    let _one = lock();
    let (a, _b, clock, _) = served("7007", Faults::default());
    a.ok("c::.cli(7007)");
    let hook = a.hook.borrow_mut().take(); // the peer is silent
    let t0 = clock.get();
    a.ctrl_c_at.set(t0 + 50_000);
    assert_eq!(a.run("c(:wait,1)").map_err(|e| e.0), Err(Kind::Interrupt));
    assert!(clock.get() - t0 < 100_000, "{}", clock.get() - t0);
    *a.hook.borrow_mut() = hook;
    // the answer may still come, and could not be told from the next one: the connection is closed
    assert_eq!(a.run("c(\"1\")").map_err(|e| e.0), Err(Kind::System));
    assert_eq!(a.ok("c::.cli(7007);c(\"1+1\")"), "2");
}

#[test]
fn a_peer_that_dies_fails_the_pending_call() {
    let _one = lock();
    let (a, b, _clock, _) = served("7006", Faults::default());
    a.ok("c::.cli(7006)");
    let r = a.run("c(\".x(0)\")");
    assert!(matches!(&r, Err((Kind::System, m)) if m.contains("closed") || m.contains("lost")), "{:?}", r);
    assert_eq!(a.run("c(\"1\")").map_err(|e| e.0), Err(Kind::System));
    drop(b);
}

fn big_round_trip(addr: &str, faults: Faults, n: usize) -> (u64, u64) {
    let (a, _b, _clock, hub) = served(&format!("\"{}\"", addr), faults);
    a.ok(&format!("c::.cli(\"{}\");.clit(c;0)", addr));
    assert_eq!(a.ok("c(\"1+2\")"), "3");
    assert_eq!(a.ok(&format!("big::!{};r::c(:add,0,,big);(r~big),#r", n)), format!("[1 {}]", n));
    assert_eq!(a.ok("c(:sq,12)"), "144");
    hub.counters()
}

#[test]
fn ipc_over_packet_links_with_faults() {
    let _one = lock();
    let f = Faults { loss: 50_000, dup: 50_000, reorder: 50_000, flip: 0 };
    let (sent, lost) = big_round_trip("lossy:a", f, 5000);
    println!("lossy link, mtu 32, 5% loss/dup/reorder: 5000-integer array intact both ways, {} packets, {} lost", sent, lost);
    assert!(lost > 0);
    let (sent, lost) = big_round_trip("tiny:a", f, 1000);
    println!("lossy link, mtu 4, 5% loss/dup/reorder: 1000-integer array intact both ways, {} packets, {} lost", sent, lost);
    let (sent, _) = big_round_trip("fabric:a", Faults::default(), 5000);
    println!("reliable packet link, mtu 64: 5000-integer array intact both ways, {} packets", sent);
    let f = Faults { loss: 500, dup: 0, reorder: 0, flip: 1000 };
    let (sent, lost) = big_round_trip("uart:0", f, 5000);
    println!("serial framing, mtu 240, 0.15% of bytes dropped or damaged: 5000-integer array intact both ways, {} bytes, {} damaged", sent, lost);
    assert!(lost > 0);
}

#[test]
fn a_dead_packet_link_fails_the_call_after_the_retry_budget() {
    let _one = lock();
    let (a, _b, clock, hub) = served("\"lossy:b\"", Faults::default());
    a.ok("c::.cli(\"lossy:b\");.clit(c;0)");
    assert_eq!(a.ok("c(\"1+2\")"), "3");
    hub.set_faults(Faults { loss: 1_000_000, ..Faults::default() });
    let t0 = clock.get();
    let r = a.run("c(\"1+2\")");
    assert!(matches!(&r, Err((Kind::System, m)) if m.contains("lost")), "{:?}", r);
    println!("dead link: the call failed after {} ms", (clock.get() - t0) / 1000);
}
