//! The event side of the runtime (docs/native/index.md): timers, queued work, and the
//! IPC connections (`ipc.rs`), all driven by [`Runtime::poll_events`]. No thread and no
//! interrupt handler touches Klong state: a board calls `poll_events` from its loop (after its
//! console work) and sleeps in [`Runtime::events_wait`].
//!
//! Timers (feature `timers`): `.timer(name; seconds; fn)` and `.timerc(handle)`, as KlongPy:
//! fractional seconds, the callback returns 0 to stop, a callback that fails is cancelled.
//! Time is `Platform::now_us`; nothing fires while an evaluation or a blocked IPC call runs,
//! and a timer that came due meanwhile fires once afterwards, then at its next multiple of the
//! interval counted from its start.
use alloc::collections::VecDeque;
use alloc::rc::Rc;
use alloc::string::String;
use alloc::vec::Vec;
use core::cell::{Cell, RefCell};

use klong_core::error::Result;
use klong_core::fmt::format_value;
use klong_core::value::{Ext, ExtBox, Value};
use klong_core::{Error, Interp};

use crate::{Platform, Runtime, Shared};

/// Something to run from `poll_events`, between other work: `f` applied to `args`, and then
/// `then` applied to the result.
pub(crate) struct Work {
    pub f: Value,
    pub args: Vec<Value>,
    pub then: Option<Value>,
    pub what: &'static str,
}

struct Timer {
    id: u32,
    name: String,
    interval_us: u64,
    start_us: u64,
    next_us: u64,
    f: Value,
}

/// What `.timer` returns; `.timerc` takes it.
pub struct TimerHandle {
    id: u32,
}

impl Ext for TimerHandle {
    fn type_name(&self) -> &str {
        "timer"
    }
    fn as_any(&self) -> &dyn core::any::Any {
        self
    }
}

pub struct Events<P: Platform + 'static> {
    pub shared: Rc<Shared<P>>,
    pub(crate) work: RefCell<VecDeque<Work>>,
    timers: RefCell<Vec<Timer>>,
    next_timer: Cell<u32>,
    /// Passes of a wait loop without anything to do (for the back-off of the waits).
    pub(crate) idle: Cell<u32>,
    #[cfg(feature = "ipc")]
    pub ipc: crate::ipc::Ipc,
}

impl<P: Platform + 'static> Events<P> {
    pub(crate) fn visit_host_roots(&self, visit: &mut dyn FnMut(&Value)) {
        for work in self.work.borrow().iter() {
            visit(&work.f);
            work.args.iter().for_each(&mut *visit);
            work.then.iter().for_each(&mut *visit);
        }
        for timer in self.timers.borrow().iter() {
            visit(&timer.f);
        }
        #[cfg(feature = "ipc")]
        self.ipc.visit_host_roots(visit);
    }

    pub(crate) fn new(shared: Rc<Shared<P>>) -> Rc<Events<P>> {
        Rc::new(Events {
            shared,
            work: RefCell::new(VecDeque::new()),
            timers: RefCell::new(Vec::new()),
            next_timer: Cell::new(1),
            idle: Cell::new(0),
            #[cfg(feature = "ipc")]
            ipc: crate::ipc::Ipc::new(),
        })
    }

    pub(crate) fn now(&self) -> u64 {
        self.shared.platform.try_borrow().map(|p| p.now_us()).unwrap_or(0)
    }

    /// A line on the error channel: what KlongPy writes to its log.
    pub(crate) fn log(&self, text: &str) {
        if let Ok(mut p) = self.shared.platform.try_borrow_mut() {
            p.console_error(text.as_bytes());
            p.console_error(b"\n");
        }
    }

    pub(crate) fn queue(&self, w: Work) -> Result<()> {
        let mut q = self.work.try_borrow_mut().map_err(|_| Error::internal("runtime state in use"))?;
        let _tag = klong_core::htag::Scope::new(klong_core::htag::IPC);
        q.try_reserve(1).map_err(|_| Error::memory())?;
        q.push_back(w);
        Ok(())
    }

    /// Nothing happened in a wait loop: spin a little (an answer on a fast link is there in
    /// microseconds), then let the platform sleep, a little longer each time, never past
    /// `deadline_us`.
    pub(crate) fn idle_wait(&self, deadline_us: Option<u64>) {
        let n = self.idle.get().saturating_add(1);
        self.idle.set(n);
        let Ok(mut p) = self.shared.platform.try_borrow_mut() else { return };
        if n < 64 {
            p.idle();
            return;
        }
        let nap = (50u64 << ((n - 64) / 32).min(5)).min(2000);
        let until = p.now_us() + nap;
        p.wait(Some(deadline_us.map_or(until, |d| d.min(until))));
    }

    // ----- timers ------------------------------------------------------------------------

    #[cfg(feature = "timers")]
    fn timer_new(&self, ip: &mut Interp, a: &[Value]) -> Result<Value> {
        let (x, y, z) = match a {
            [x, y, z] => (x, y, z),
            _ => return Err(Error::arity(klong_core::msg!(".timer takes 3 arguments, got {}", a.len()))),
        };
        let seconds = match y {
            Value::Int(i) => *i as f64,
            Value::Flt(f) => *f as f64,
            _ => return Err(Error::typ("y must be a non-negative number")),
        };
        if !(seconds >= 0.0) || seconds > 1e12 {
            return Err(Error::range("y must be a non-negative number"));
        }
        if !matches!(z, Value::Fn(_) | Value::Prim(_)) {
            return Err(Error::typ("z must be a function"));
        }
        let name = format_value(&ip.syms, x, true)?;
        let interval_us = (seconds * 1e6 + 0.5) as u64;
        let now = self.now();
        let id = self.next_timer.get();
        self.next_timer.set(id.wrapping_add(1).max(1));
        let mut timers = self.timers.try_borrow_mut().map_err(|_| Error::internal("runtime state in use"))?;
        timers.try_reserve(1).map_err(|_| Error::memory())?;
        timers.push(Timer { id, name, interval_us, start_us: now, next_us: now + interval_us, f: z.clone() });
        Ok(Value::Ext(Rc::new(ExtBox(alloc::boxed::Box::new(TimerHandle { id })))))
    }

    #[cfg(feature = "timers")]
    fn timer_cancel(&self, x: &Value) -> Value {
        let id = match x {
            Value::Ext(e) => match e.0.as_any().downcast_ref::<TimerHandle>() {
                Some(h) => h.id,
                None => return Value::Int(0),
            },
            _ => return Value::Int(0),
        };
        let mut timers = match self.timers.try_borrow_mut() {
            Ok(t) => t,
            Err(_) => return Value::Int(0),
        };
        let before = timers.len();
        timers.retain(|t| t.id != id);
        Value::bool(timers.len() != before)
    }

    /// Run the callback of the timer that has been due longest, if one is due.
    fn fire_due(&self, ip: &mut Interp) -> bool {
        let now = self.now();
        let (id, f) = {
            let timers = match self.timers.try_borrow() {
                Ok(t) => t,
                Err(_) => return false,
            };
            match timers.iter().filter(|t| t.next_us <= now).min_by_key(|t| t.next_us) {
                Some(t) => (t.id, t.f.clone()),
                None => return false,
            }
        };
        // (the borrow is released: the callback may start and cancel timers)
        let r = ip.apply(&f, Vec::new(), Some(0));
        let now = self.now();
        let mut failed = None;
        if let Ok(mut timers) = self.timers.try_borrow_mut() {
            if let Some(i) = timers.iter().position(|t| t.id == id) {
                match &r {
                    Ok(v) if v.truthy() => {
                        let t = &mut timers[i];
                        t.next_us = if t.interval_us == 0 { now } else { now + (t.interval_us - (now.saturating_sub(t.start_us) % t.interval_us)) };
                    }
                    Ok(_) => {
                        timers.remove(i);
                    }
                    Err(e) => {
                        failed = Some(klong_core::msg!("timer {} callback failed; cancelling timer: {}", timers[i].name, e.render()));
                        timers.remove(i);
                    }
                }
            }
        }
        if let Some(text) = failed {
            self.log(&text);
        }
        true
    }

    fn run_work(&self, ip: &mut Interp) -> bool {
        let w = match self.work.try_borrow_mut() {
            Ok(mut q) => q.pop_front(),
            Err(_) => None,
        };
        let Some(w) = w else { return false };
        let n = w.args.len().min(255) as u8;
        let r = ip.apply(&w.f, w.args, Some(n)).and_then(|v| match &w.then {
            Some(cb) => {
                let mut one = klong_core::value::try_vec(1)?;
                one.push(v);
                ip.apply(cb, one, Some(1))
            }
            None => Ok(v),
        });
        if let Err(e) = r {
            self.log(&klong_core::msg!("{}: call failed: {}", w.what, e.render()));
        }
        true
    }

    /// One pass: IPC (accept, serve one request per connection, move bytes), one due timer,
    /// one queued call. True when something was done.
    pub fn poll(self: &Rc<Self>, ip: &mut Interp) -> bool {
        let mut did = false;
        #[cfg(feature = "ipc")]
        {
            did |= self.ipc_poll(ip);
        }
        did |= self.fire_due(ip);
        did |= self.run_work(ip);
        if did {
            self.idle.set(0);
        }
        did
    }

    /// Something can still happen without new input from the program: a timer is running, a
    /// call is queued, a server listens, a connection is being served or has a request out.
    pub fn active(&self) -> bool {
        let timers = self.timers.try_borrow().map(|t| !t.is_empty()).unwrap_or(true);
        let work = self.work.try_borrow().map(|q| !q.is_empty()).unwrap_or(true);
        #[cfg(feature = "ipc")]
        if self.ipc.active() {
            return true;
        }
        timers || work
    }

    /// When the next thing is due (microseconds of `now_us`), if anything is scheduled.
    pub fn next_deadline(&self) -> Option<u64> {
        if self.work.try_borrow().map(|q| !q.is_empty()).unwrap_or(true) {
            return Some(0);
        }
        let t = self.timers.try_borrow().ok().and_then(|t| t.iter().map(|t| t.next_us).min());
        #[cfg(feature = "ipc")]
        {
            let i = self.ipc.next_deadline();
            return match (t, i) {
                (Some(a), Some(b)) => Some(a.min(b)),
                (a, b) => a.or(b),
            };
        }
        #[cfg(not(feature = "ipc"))]
        t
    }
}

pub(crate) fn register<P: Platform + 'static>(interp: &mut Interp, ev: &Rc<Events<P>>) {
    #[cfg(feature = "timers")]
    {
        let w = Rc::downgrade(ev);
        interp.register(
            ".timer",
            3,
            Rc::new(move |ip: &mut Interp, a: &[Value]| match w.upgrade() {
                Some(ev) => ev.timer_new(ip, a),
                None => Err(Error::internal("no runtime")),
            }),
        );
        let w = Rc::downgrade(ev);
        interp.register(
            ".timerc",
            1,
            Rc::new(move |_: &mut Interp, a: &[Value]| match w.upgrade() {
                Some(ev) => Ok(ev.timer_cancel(crate::one(a, ".timerc")?)),
                None => Ok(Value::Int(0)),
            }),
        );
    }
    #[cfg(feature = "ipc")]
    crate::ipc::register(interp, ev);
    let _ = (interp, ev);
}

impl<P: Platform + 'static> Runtime<P> {
    /// One pass of the event loop; true when something was done. Call it between console
    /// lines; a board calls it from its main loop.
    pub fn poll_events(&mut self) -> bool {
        let ev = self.events.clone();
        ev.poll(&mut self.interp)
    }

    /// Whether timers, queued calls, a server or connections in use still need the loop.
    pub fn events_active(&self) -> bool {
        self.events.active()
    }

    /// Sleep until something is due or input may be ready (`Platform::wait`). Call it when
    /// `poll_events` returned false.
    pub fn events_wait(&mut self) {
        let d = self.events.next_deadline();
        self.events.idle_wait(d);
    }

    /// Serve until nothing is left to wait for (`events_active` is false).
    pub fn run_events(&mut self) {
        while self.events_active() {
            if !self.poll_events() {
                self.events_wait();
            }
        }
    }
}
