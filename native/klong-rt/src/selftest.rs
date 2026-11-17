//! The boot self-test of the Pico bench image: one program for every board, run from the main
//! loop, reported to the supervisor.
//!
//! Every node serves (`.srv("node:")`, with two functions that count the calls they answer).
//! A node that is not number 0 then plays client against the nodes below it, from the nearest
//! down: it tries `.cli("node:K")` for K = N-1, N-2, ..., 0, and the first that answers is its
//! peer. (A node that runs another firmware does not answer; that costs the handshake timeout.)
//! So with two boards flashed, the lower serves and the higher connects, without either
//! knowing the other's number beforehand, and the same file is flashed to both. Against the
//! peer the client runs four steps and times them with the board's clock:
//!
//! 1. `sync`: a synchronous call;
//! 2. `async`: an asynchronous call, then a synchronous one, then the asynchronous answer;
//! 3. `echo`: 500 integers (4 KB) there and back, compared;
//! 4. `mgmt`: the management library (`mgmt.call`) on the peer; skipped when the library is not built in.
//!
//! The verdict is reported to address 16 on datagram channel 3, as text, every 3 seconds, and
//! as the flags byte of STATUS (`klong_dgram::lane::STATUS_SELFTEST_*`):
//!
//! | State | STATUS flags | Report text |
//! |---|---|---|
//! | serving (a server, or a client that found no peer yet, or is running the steps) | `0x31` | `klong selftest serving node=N calls=C` or `... running ...` |
//! | passed | `0x51` | `klong selftest pass 212ms node=N peer=K sync=3ms echo=130ms calls=C async=Ams load=Lms add=Dms call=Mms mgmt=Gms` |
//! | failed | `0x91` | `klong selftest fail STEP REASON node=N peer=K` |
//!
//! The first number after `pass` is the total time of the four steps in milliseconds, measured
//! with the board clock. A failed test is run again after 15 seconds and a missing peer is looked for
//! again after 10; a pass stays. `calls` is how many calls this node answered as a server.
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use klong_dgram::lane::{STATUS_SELFTEST_FAIL, STATUS_SELFTEST_PASS, STATUS_SELFTEST_RUNNING};

use crate::node::{NodeNet, CH_SELFTEST};
use crate::{Platform, Runtime};

const REPORT_EVERY_MS: u64 = 3000;
const RETRY_FAIL_MS: u64 = 15_000;
const RETRY_NO_PEER_MS: u64 = 10_000;
const RETRY_SOON_MS: u64 = 1_500;
/// How long a probe of a node below waits for its handshake. A node that is probing serves nobody (a blocked `.cli`), so this is the time it
/// is deaf, for each mute node below it (the bench has two): 400 ms each deafened it for 800 ms and a call to it took 830 ms (measured on the
/// boards, 2026-11-17); a Klong node answers in 5 ms (27 ms at 115200 baud), so 80 ms is a margin of three.
const PROBE_HANDSHAKE_US: u64 = 80_000;
/// Let the lower nodes start listening before the first connection attempt.
const START_DELAY_MS: u64 = 1000;
/// A node starts 600 ms later than the one above it (up to node 4): a node that probes a lower node that is mute waits for it
/// and serves nobody meanwhile, so the higher nodes ask first, and are done (a pass is 125 ms) before the lower one probes.
const START_STAGGER_MS: u64 = 600;
const STEP_LIMIT_US: u64 = 20_000_000;

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Verdict {
    /// Not decided: serving, looking for a peer, or running the steps.
    Running,
    Pass,
    Fail,
}

enum State {
    /// Waiting for a node number.
    Boot,
    /// Server set up, waiting to look for a peer.
    Ready { since: u64 },
    NoPeer { retry_at: u64 },
    Passed,
    Failed { retry_at: u64 },
}

pub struct SelfTest {
    state: State,
    node: u8,
    peer: Option<u8>,
    result: String,
    steps: Vec<(&'static str, u64)>,
    total_ms: u64,
    /// When the last report went out; `None`: one is due now.
    last_report: Option<u64>,
    pub attempts: u32,
    /// Why the board restarted (a fault, the watchdog), kept in every report until the next restart.
    note: String,
}

impl Default for SelfTest {
    fn default() -> Self {
        SelfTest::new()
    }
}

fn first_line(e: &str) -> String {
    let l = e.lines().next().unwrap_or("");
    let mut s: String = l.chars().take(120).collect();
    if s.is_empty() {
        s.push_str("error");
    }
    s
}

impl SelfTest {
    pub fn new() -> SelfTest {
        SelfTest { state: State::Boot, node: 0, peer: None, result: String::new(), steps: Vec::new(), total_ms: 0, last_report: None, attempts: 0, note: String::new() }
    }

    /// The reason of the restart, for the reports (empty: a power-on or an ordinary reset).
    pub fn set_note(&mut self, note: String) {
        self.note = note;
    }

    pub fn verdict(&self) -> Verdict {
        match self.state {
            State::Passed => Verdict::Pass,
            State::Failed { .. } => Verdict::Fail,
            _ => Verdict::Running,
        }
    }

    /// The steps that were timed in the last run, with their milliseconds.
    pub fn steps(&self) -> &[(&'static str, u64)] {
        &self.steps
    }

    /// The text of the last verdict line (empty before the first).
    pub fn report_text(&self) -> &str {
        &self.result
    }

    fn eval<P: Platform + 'static>(rt: &mut Runtime<P>, text: &str) -> Result<String, String> {
        match rt.run(text, "<selftest>") {
            Ok(Some(v)) => klong_core::fmt::format_value(&rt.interp.syms, &v, false).map_err(|e| first_line(&e.message)),
            Ok(None) => Ok(String::new()),
            Err(e) => Err(first_line(&format!("{}: {}", e.kind.name(), e.message))),
        }
    }

    fn now_ms<P: Platform + 'static>(rt: &Runtime<P>) -> u64 {
        rt.shared.platform.try_borrow().map(|p| p.now_us() / 1000).unwrap_or(0)
    }

    fn calls<P: Platform + 'static>(rt: &mut Runtime<P>) -> u64 {
        Self::eval(rt, "hits").ok().and_then(|s| s.parse().ok()).unwrap_or(0)
    }

    fn check<P: Platform + 'static>(rt: &mut Runtime<P>, step: &'static str, code: &str, want: &str) -> Result<(), (&'static str, String)> {
        match Self::eval(rt, code) {
            Ok(v) if v == want => Ok(()),
            Ok(v) => Err((step, format!("expected {} got {}", want, first_line(&v)))),
            Err(e) => Err((step, e)),
        }
    }

    /// The four steps against `peer`; the time of each.
    fn run_steps<P: Platform + 'static>(&mut self, rt: &mut Runtime<P>, net: &NodeNet) -> Result<(), (&'static str, String)> {
        self.steps.clear();
        let t0 = Self::now_ms(rt);
        let mut mark = t0;
        let mut done = |this: &mut SelfTest, rt: &Runtime<P>, name: &'static str| {
            let t = Self::now_ms(rt);
            this.steps.push((name, t - mark));
            mark = t;
        };
        let k = self.peer.unwrap_or(0);
        Self::check(rt, "sync", &format!("h(:stping{},,7)", k), "7")?;
        done(self, rt, "sync");
        Self::eval(rt, &format!("got::0;af::.async(h(:stping{});{{got::x}});af(5)", k)).map_err(|e| ("async", e))?;
        Self::check(rt, "async", &format!("h(:stping{},,6)", k), "6")?;
        let limit = rt.shared.platform.try_borrow().map(|p| p.now_us()).unwrap_or(0) + STEP_LIMIT_US;
        loop {
            rt.poll_events();
            if Self::eval(rt, "got").as_deref() == Ok("5") {
                break;
            }
            let now = rt.shared.platform.try_borrow().map(|p| p.now_us()).unwrap_or(0);
            if now > limit {
                return Err(("async", String::from("the asynchronous answer never came")));
            }
            net.pump();
            rt.events_wait();
        }
        done(self, rt, "async");
        Self::check(rt, "echo", &format!("big::!500;(h(:stecho{},,big))~big", k), "1")?;
        done(self, rt, "echo");
        // the connection is closed first: a node has one connection of each kind to a peer, and the
        // management library opens its own
        let _ = Self::eval(rt, ".clic(h)");
        // the management library on the peer, when this build has it
        match Self::eval(rt, ".l(\"mgmt\")") {
            Err(_) => {}
            Ok(_) => {
                done(self, rt, "load");
                Self::eval(rt, &format!("mgmt.add(\"p\";\"node:{}\")", self.peer.unwrap_or(0))).map_err(|e| ("mgmt", e))?;
                done(self, rt, "add");
                let r = Self::check(rt, "mgmt", &format!("mgmt.call(\"p\";[:stping{}]@0;[9])", k), "9");
                done(self, rt, "call");
                // the library's connection to the peer is closed, so that `.cli` to it works afterwards
                let _ = Self::eval(rt, ".clic((mgmt.nodes?\"p\")?\"h\")");
                r?;
                done(self, rt, "mgmt");
            }
        }
        self.total_ms = Self::now_ms(rt) - t0;
        Ok(())
    }

    /// Leave nothing behind: the test's variables, and the management library, whose functions are some 18 KB of the
    /// board's heap that a user's program would rather have (a board of 264 KB: the client of the test had half again as
    /// much in use as the server, for as long as it ran). The library is dropped by naming each of its definitions.
    fn clean_up<P: Platform + 'static>(rt: &mut Runtime<P>) {
        let _ = Self::eval(rt, "big::0;got::0;af::0;h::0");
        if let Some((_, text)) = crate::library::find("mgmt") {
            let mut prog = String::new();
            for line in text.lines() {
                let end = line.find(|c: char| !(c.is_ascii_alphanumeric() || c == '.' || c == '_')).unwrap_or(line.len());
                if end > 0 && line.as_bytes()[0].is_ascii_alphabetic() && line[end..].starts_with("::") {
                    prog.push_str(&line[..end]);
                    prog.push_str("::0;");
                    if prog.len() > 600 {
                        let _ = Self::eval(rt, &prog);
                        prog.clear();
                    }
                }
            }
            if !prog.is_empty() {
                let _ = Self::eval(rt, &prog);
            }
        }
        rt.interp.release_lazy();
        rt.interp.release_caches();
    }

    fn text(&self, rt_calls: u64, running: bool) -> String {
        let mut t = self.text_main(rt_calls, running);
        if !self.note.is_empty() {
            t.push_str(" boot=");
            t.push_str(&self.note);
        }
        t
    }

    fn text_main(&self, rt_calls: u64, running: bool) -> String {
        let n = self.node;
        match &self.state {
            State::Passed => {
                let step = |name: &str| self.steps.iter().find(|s| s.0 == name).map(|s| s.1).unwrap_or(0);
                format!("klong selftest pass {}ms node={} peer={} sync={}ms echo={}ms calls={} async={}ms load={}ms add={}ms call={}ms mgmt={}ms", self.total_ms, n, self.peer.unwrap_or(0), step("sync"), step("echo"), rt_calls, step("async"), step("load"), step("add"), step("call"), step("mgmt"))
            }
            State::Failed { .. } => format!("{} node={} peer={}", self.result, n, self.peer.unwrap_or(0)),
            _ if running => format!("klong selftest running node={} calls={}", n, rt_calls),
            _ => format!("klong selftest serving node={} calls={}", n, rt_calls),
        }
    }

    /// One turn, from the board's main loop. True when it did something.
    pub fn step<P: Platform + 'static>(&mut self, rt: &mut Runtime<P>, net: &NodeNet) -> bool {
        let now = Self::now_ms(rt);
        let mut did = false;
        match self.state {
            State::Boot => {
                let Some(n) = net.node() else { return false };
                self.node = n;
                // the served functions carry the node's number in their names: a node's own definitions
        // must not shadow its peer's (`.async(h(:f);cb)` runs a function that is defined here itself)
        let up = Self::eval(rt, &format!("hits::0;stping{n}::{{hits::hits+1;x}};stecho{n}::{{hits::hits+1;x}};.srv(\"node:\")", n = n));
                if up.is_err() {
                    return false;
                }
                net.set_status_flags(STATUS_SELFTEST_RUNNING);
                self.last_report = None;
                self.state = State::Ready { since: now };
                did = true;
            }
            State::Ready { since } if now >= since + START_DELAY_MS + START_STAGGER_MS * (4 - self.node.min(4) as u64) => {
                did = true;
                self.attempts += 1;
                self.find_and_run(rt, net, now);
            }
            State::NoPeer { retry_at } | State::Failed { retry_at } if now >= retry_at => {
                did = true;
                self.attempts += 1;
                self.find_and_run(rt, net, now);
            }
            _ => {}
        }
        if self.last_report.map_or(true, |t| now >= t + REPORT_EVERY_MS) && net.node().is_some() {
            let running = matches!(self.state, State::Ready { .. });
            let calls = Self::calls(rt);
            let text = self.text(calls, running);
            // a datagram to the supervisor on channel 1; if the sender is busy, the next turn tries again
            if let Ok(true) = net.send(klong_dgram::frame::SUPERVISOR, CH_SELFTEST, text.as_bytes(), false) {
                self.last_report = Some(now);
            }
            did = true;
        }
        did
    }

    fn find_and_run<P: Platform + 'static>(&mut self, rt: &mut Runtime<P>, net: &NodeNet, now: u64) {
        net.set_status_flags(STATUS_SELFTEST_RUNNING);
        self.peer = None;
        // the nodes below this one, nearest first
        let mut found = false;
        // A node that does not answer (the bench's other firmware sits below the first Klong node) is waited for while
        // this node serves nobody: a blocked `.cli` answers no hello of another node's. A Klong node answers in a few
        // milliseconds (27 ms at 115200), so the probe waits a fraction of a second, not the 3 s of a real connection;
        // with 3 s each, the lowest Klong node was deaf for 6 s after every boot and the one above it, which asked in that
        // time, waited 10 s to ask again (measured on the boards, 2026-11-17).
        let hs = net.handshake_timeout_us();
        net.set_handshake_timeout_us(PROBE_HANDSHAKE_US.min(hs));
        for k in (0..self.node).rev() {
            if Self::eval(rt, &format!("h::.cli(\"node:{}\")", k)).is_ok() {
                self.peer = Some(k);
                found = true;
                break;
            }
        }
        net.set_handshake_timeout_us(hs);
        if !found {
            // asked again soon at first (the node above may have asked while this one was probing), then every 10 s
            // (each node waits a second longer than the one below it, so that two nodes that probe do not probe at the same time)
            let wait = if self.attempts < 4 { RETRY_SOON_MS + 1000 * self.node as u64 } else { RETRY_NO_PEER_MS };
            self.state = State::NoPeer { retry_at: Self::now_ms(rt) + wait };
            return;
        }
        match self.run_steps(rt, net) {
            Ok(()) => {
                self.state = State::Passed;
                net.set_status_flags(STATUS_SELFTEST_PASS);
            }
            Err((step, why)) => {
                self.result = format!("klong selftest fail {} {}", step, why);
                self.state = State::Failed { retry_at: Self::now_ms(rt).max(now) + RETRY_FAIL_MS };
                net.set_status_flags(STATUS_SELFTEST_FAIL);
            }
        }
        let _ = Self::eval(rt, ".clic(h)");
        Self::clean_up(rt);
        // the report goes out at the next turn
        self.last_report = None;
    }
}
