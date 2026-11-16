//! klong-rt: what a Klong runtime needs around the core, still without naming a board or an
//! operating system: the [`Platform`] trait, the system functions that need it (`.p .d .w`),
//! the tick hook that carries Ctrl-C from the console into a running evaluation, and heap
//! accounting ([`heap::Counting`]).
//!
//! This is the stub of ticket 102. The full trait (files, transports, externs, `wait`), the
//! event loop, the line editor and the upload receiver are tickets 104 and 106.
#![no_std]
#![deny(clippy::unwrap_used, clippy::expect_used)]

extern crate alloc;

pub mod heap;

use alloc::boxed::Box;
use alloc::rc::Rc;
use alloc::vec::Vec;
use core::cell::RefCell;

use klong_core::error::Result;
use klong_core::fmt::format_value;
use klong_core::value::{Value, WILDCARD};
use klong_core::{Config, Error, Hooks, Interp};

/// Everything outside pure computation (native-runtime.md section 6). Only the part the
/// spike needs: time, idle and the console.
pub trait Platform {
    /// Monotonic microseconds.
    fn now_us(&self) -> u64;
    /// Seconds since 1970, if known.
    fn wall_clock(&self) -> Option<f64>;
    /// Watchdog, yield. Called often.
    fn idle(&mut self);
    /// Non-blocking; 0 if nothing is waiting.
    fn console_read(&mut self, buf: &mut [u8]) -> usize;
    fn console_write(&mut self, bytes: &[u8]);
}

/// Heap figures for `.mem()`: (bytes in use, peak, limit; 0 = no limit).
pub type MemStats = fn() -> (usize, usize, usize);

struct RtHooks<P: Platform> {
    platform: Rc<RefCell<P>>,
    pending: Rc<RefCell<Vec<u8>>>,
}

impl<P: Platform> Hooks for RtHooks<P> {
    /// Called by the core every 4096 calls or adverb iterations: let the platform breathe,
    /// keep what was typed, and report Ctrl-C (byte 0x03).
    fn tick(&mut self) -> bool {
        let mut p = match self.platform.try_borrow_mut() {
            Ok(p) => p,
            Err(_) => return false,
        };
        p.idle();
        let mut buf = [0u8; 32];
        let mut stop = false;
        loop {
            let n = p.console_read(&mut buf);
            if n == 0 {
                break;
            }
            for &b in &buf[..n] {
                if b == 0x03 {
                    stop = true;
                } else if let Ok(mut q) = self.pending.try_borrow_mut() {
                    q.push(b);
                }
            }
        }
        stop
    }
}

pub struct Runtime<P: Platform + 'static> {
    pub interp: Interp,
    pub platform: Rc<RefCell<P>>,
    /// Console bytes read while an evaluation was running (for the line editor of ticket 104).
    pub pending: Rc<RefCell<Vec<u8>>>,
}

impl<P: Platform + 'static> Runtime<P> {
    pub fn new(platform: P, cfg: Config, mem: Option<MemStats>) -> Runtime<P> {
        let platform = Rc::new(RefCell::new(platform));
        let pending = Rc::new(RefCell::new(Vec::new()));
        let hooks = RtHooks { platform: platform.clone(), pending: pending.clone() };
        let mut interp = Interp::new(cfg, Box::new(hooks));

        let write = |platform: &Rc<RefCell<P>>, s: &str| {
            if let Ok(mut p) = platform.try_borrow_mut() {
                p.console_write(s.as_bytes());
            }
        };
        let p = platform.clone();
        interp.register(
            ".p",
            1,
            Rc::new(move |ip: &mut Interp, a: &[Value]| {
                let x = one(a, ".p")?;
                let s = format_value(&ip.syms, x, true)?;
                write(&p, &s);
                write(&p, "\n");
                Ok(Value::str(s.into_bytes()))
            }),
        );
        let p = platform.clone();
        interp.register(
            ".d",
            1,
            Rc::new(move |ip: &mut Interp, a: &[Value]| {
                let x = one(a, ".d")?;
                let s = format_value(&ip.syms, x, true)?;
                write(&p, &s);
                Ok(Value::str(s.into_bytes()))
            }),
        );
        let p = platform.clone();
        interp.register(
            ".w",
            1,
            Rc::new(move |ip: &mut Interp, a: &[Value]| {
                let x = one(a, ".w")?;
                let s = format_value(&ip.syms, x, false)?;
                write(&p, &s);
                Ok(x.clone())
            }),
        );
        interp.register(
            ".strict",
            WILDCARD,
            Rc::new(|ip: &mut Interp, a: &[Value]| {
                let level = match a.first() {
                    None => 1,
                    Some(Value::Int(i)) if (0..=2).contains(i) => *i as u8,
                    Some(Value::Int(_)) => return Err(Error::range("the strict level must be 0, 1 or 2")),
                    Some(_) => return Err(Error::typ("the strict level must be 0, 1 or 2")),
                };
                ip.cfg.strict = level;
                Ok(Value::Int(level as klong_core::Int))
            }),
        );
        interp.register(
            ".unsafe",
            0,
            Rc::new(|ip: &mut Interp, _: &[Value]| {
                ip.cfg.strict = 0;
                Ok(Value::Int(0))
            }),
        );
        interp.register(
            ".mem",
            0,
            Rc::new(move |_: &mut Interp, _: &[Value]| {
                let (used, peak, limit) = mem.map(|f| f()).unwrap_or((0, 0, 0));
                Ok(Value::ints(alloc::vec![used as klong_core::Int, peak as klong_core::Int, limit as klong_core::Int]))
            }),
        );
        Runtime { interp, platform, pending }
    }

    /// Run a program; the value of its last statement.
    pub fn run(&mut self, text: &str, name: &str) -> Result<Option<Value>> {
        self.interp.run(text, name)
    }
}

fn one<'a>(a: &'a [Value], name: &str) -> Result<&'a Value> {
    a.first().ok_or_else(|| Error::arity(alloc::format!("{} takes 1 argument, got 0", name)))
}
