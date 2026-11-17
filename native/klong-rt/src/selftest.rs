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
//! The verdict is reported to address 16 on datagram channel 1, as text, every 3 seconds, and
//! as the flags byte of STATUS (`klong_dgram::lane::STATUS_SELFTEST_*`):
//!
//! | State | STATUS flags | Report text |
//! |---|---|---|
//! | serving (a server, or a client that found no peer yet, or is running the steps) | `0x21` | `klong selftest serving node=N calls=C` or `... running ...` |
//! | passed | `0x41` | `klong selftest pass 212ms node=N peer=K sync=3ms echo=130ms calls=C` |
//! | failed | `0x81` | `klong selftest fail STEP REASON node=N peer=K` |
//!
//! The first number after `pass` is the total time of the four steps in milliseconds, measured
//! with the board clock. A failed test is run again after 15 seconds and a missing peer is looked for
//! again after 10; a pass stays. `calls` is how many calls this node answered as a server.
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use klong_dgram::lane::{STATUS_SELFTEST_FAIL, STATUS_SELFTEST_PASS, STATUS_SELFTEST_RUNNING};

use crate::node::{NodeNet, CH_CONSOLE};
use crate::{Platform, Runtime};

const REPORT_EVERY_MS: u64 = 3000;
const RETRY_FAIL_MS: u64 = 15_000;
const RETRY_NO_PEER_MS: u64 = 10_000;
/// Let the lower nodes start listening before the first connection attempt.
const START_DELAY_MS: u64 = 1000;
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
        SelfTest { state: State::Boot, node: 0, peer: None, result: String::new(), steps: Vec::new(), total_ms: 0, last_report: None, attempts: 0 }
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
                Self::eval(rt, &format!("mgmt.add(\"p\";\"node:{}\")", self.peer.unwrap_or(0))).map_err(|e| ("mgmt", e))?;
                Self::check(rt, "mgmt", &format!("mgmt.call(\"p\";[:stping{}]@0;[9])", k), "9")?;
                done(self, rt, "mgmt");
            }
        }
        self.total_ms = Self::now_ms(rt) - t0;
        Ok(())
    }

    fn text(&self, rt_calls: u64, running: bool) -> String {
        let n = self.node;
        match &self.state {
            State::Passed => {
                let step = |name: &str| self.steps.iter().find(|s| s.0 == name).map(|s| s.1).unwrap_or(0);
                format!("klong selftest pass {}ms node={} peer={} sync={}ms echo={}ms calls={}", self.total_ms, n, self.peer.unwrap_or(0), step("sync"), step("echo"), rt_calls)
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
            State::Ready { since } if now >= since + START_DELAY_MS => {
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
            if let Ok(true) = net.send(klong_dgram::frame::SUPERVISOR, CH_CONSOLE, text.as_bytes(), false) {
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
        for k in (0..self.node).rev() {
            if Self::eval(rt, &format!("h::.cli(\"node:{}\")", k)).is_ok() {
                self.peer = Some(k);
                found = true;
                break;
            }
        }
        if !found {
            self.state = State::NoPeer { retry_at: Self::now_ms(rt) + RETRY_NO_PEER_MS };
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
        // the report goes out at the next turn
        self.last_report = None;
    }
}
