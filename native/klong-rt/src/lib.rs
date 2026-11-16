//! klong-rt: what a Klong runtime needs around the core, still without naming a board or an
//! operating system: the [`Platform`] and [`FileSystem`] traits, the system functions
//! (native-runtime.md section 9), the tick hook that carries Ctrl-C from the console into a
//! running evaluation, heap accounting ([`heap::Counting`]) and the Klong library as text.
//!
//! Features: `files` (channels, `.l`, the file functions; needs `Platform::fs`), `lib` (the
//! pure-Klong library of `klongpy/lib` in the image, loaded by `.l("math")`). The event loop,
//! the line editor and the upload receiver are ticket 104; IPC and timers ticket 106.
#![no_std]
#![deny(clippy::unwrap_used, clippy::expect_used)]

extern crate alloc;

pub mod events;
pub mod heap;
#[cfg(feature = "ipc")]
pub mod ipc;
pub mod link;
pub mod memfs;
#[cfg(feature = "ipc")]
pub mod memlink;
pub mod platform;

#[cfg(feature = "files")]
mod chan;
mod fmtfn;
#[cfg(feature = "lib")]
pub mod library;

use alloc::boxed::Box;
use alloc::rc::Rc;
use alloc::string::String;
use alloc::vec::Vec;
use core::cell::RefCell;

use klong_core::error::Result;
use klong_core::fmt::format_value;
use klong_core::value::{Dict, Value, WILDCARD};
use klong_core::{Config, Error, Flt, Hooks, Int, Interp};

pub use platform::{FileId, FileSystem, FsError, Mode, Platform};

/// Heap figures for `.mem()`: (bytes in use, peak, limit; 0 = no limit).
pub type MemStats = fn() -> (usize, usize, usize);

/// What the system functions share: the platform, the console bytes read while an evaluation
/// ran, and the input/output state.
pub struct Shared<P: Platform> {
    pub platform: RefCell<P>,
    /// Console bytes read while an evaluation was running (for the line editor of ticket 104;
    /// `.rl` on the console reads them too).
    pub pending: RefCell<Vec<u8>>,
    /// A Ctrl-C that the console found in input it had already read when an evaluation
    /// started: the next tick reports it (an evaluation that ends before its first tick
    /// leaves it to the console's ordinary input). The console clears it when the evaluation ends.
    pub interrupt_armed: core::cell::Cell<bool>,
    /// A Ctrl-C was read from the port during the evaluation (the console then drops the
    /// input it had read ahead before the evaluation: it was typed before the Ctrl-C).
    pub ctrl_c_seen: core::cell::Cell<bool>,
    /// The tick reported the armed Ctrl-C.
    pub armed_hit: core::cell::Cell<bool>,
    pub(crate) io: RefCell<Io>,
}

#[cfg_attr(not(feature = "files"), allow(dead_code))]
pub(crate) struct Io {
    /// The From and To channels (`.fc`, `.tc`); `Undefined` before the channels exist (a build
    /// without `files`: the console).
    pub from: Value,
    pub to: Value,
    pub cin: Value,
    pub cout: Value,
    pub rng: u64,
    pub start_us: u64,
    /// Directories `.l` looks in, in order.
    pub path: Vec<String>,
}

impl<P: Platform> Shared<P> {
    /// Read what has arrived on the console during an evaluation. The first Ctrl-C (0x03) is
    /// reported (true) and the typed-ahead input received before it is discarded. Everything
    /// after it, including further Ctrl-C bytes, is kept in order for the console: one
    /// Ctrl-C ends one evaluation. Typed-ahead input is bounded (4096 bytes).
    pub(crate) fn scan_console(&self, p: &mut P) -> bool {
        let mut buf = [0u8; 32];
        let armed = self.interrupt_armed.replace(false);
        if armed {
            self.armed_hit.set(true);
        }
        let mut stop = armed;
        loop {
            let n = p.console_read(&mut buf);
            if n == 0 {
                break;
            }
            for &b in &buf[..n] {
                if b == 0x03 && !stop {
                    stop = true;
                    self.ctrl_c_seen.set(true);
                    if let Ok(mut q) = self.pending.try_borrow_mut() {
                        q.clear();
                    }
                } else if let Ok(mut q) = self.pending.try_borrow_mut() {
                    // typed-ahead input is bounded: a board has no room for an unread flood
                    if q.len() < 4096 && q.try_reserve(1).is_ok() {
                        q.push(b);
                    }
                }
            }
            if stop {
                // (the rest stays at the port until the next tick: a later Ctrl-C is for a later evaluation)
                break;
            }
        }
        stop
    }

    /// Write to the console, whatever the To channel is.
    pub fn console(&self, s: &str) {
        if let Ok(mut p) = self.platform.try_borrow_mut() {
            p.console_write(s.as_bytes());
        }
    }
}

struct RtHooks<P: Platform> {
    shared: Rc<Shared<P>>,
}

impl<P: Platform> Hooks for RtHooks<P> {
    /// Called by the core every 4096 calls or adverb iterations: let the platform breathe,
    /// keep what was typed, and report Ctrl-C (byte 0x03).
    fn tick(&mut self) -> bool {
        let mut p = match self.shared.platform.try_borrow_mut() {
            Ok(p) => p,
            Err(_) => return false,
        };
        p.idle();
        self.shared.scan_console(&mut p)
    }
}

pub struct Runtime<P: Platform + 'static> {
    pub interp: Interp,
    pub shared: Rc<Shared<P>>,
    /// Timers, queued calls and IPC connections (`events.rs`).
    pub events: Rc<events::Events<P>>,
}

pub(crate) fn one<'a>(a: &'a [Value], name: &str) -> Result<&'a Value> {
    a.first().ok_or_else(|| Error::arity(klong_core::msg!("{} takes 1 argument, got 0", name)))
}

pub(crate) fn text_arg<'a>(a: &'a [Value], name: &str) -> Result<&'a str> {
    match one(a, name)? {
        Value::Str(s) => Ok(s.as_str()),
        v => Err(Error::typ(klong_core::msg!("{} needs a string, got {}", name, v.describe()))),
    }
}

/// False as `.fc`, `.tc`, `.module` and `.x` mean it: 0 or something empty.
pub(crate) fn is_false(v: &Value) -> bool {
    matches!(v, Value::Int(0)) || matches!(v, Value::Flt(f) if *f == 0.0) || v.is_empty()
}

impl<P: Platform + 'static> Runtime<P> {
    pub fn new(mut platform: P, cfg: Config, mem: Option<MemStats>) -> Runtime<P> {
        let rng = platform.entropy() | 1;
        let start_us = platform.now_us();
        let mut path = Vec::new();
        path.push(String::from("."));
        path.push(String::from("lib"));
        let shared = Rc::new(Shared {
            platform: RefCell::new(platform),
            pending: RefCell::new(Vec::new()),
            interrupt_armed: core::cell::Cell::new(false),
            ctrl_c_seen: core::cell::Cell::new(false),
            armed_hit: core::cell::Cell::new(false),
            io: RefCell::new(Io { from: Value::Undefined, to: Value::Undefined, cin: Value::Undefined, cout: Value::Undefined, rng, start_us, path }),
        });
        let mut interp = Interp::new(cfg, Box::new(RtHooks { shared: shared.clone() }));

        // ----- output ---------------------------------------------------------------------------
        let sh = shared.clone();
        interp.register(
            ".p",
            1,
            Rc::new(move |ip: &mut Interp, a: &[Value]| {
                let mut s = format_value(&ip.syms, one(a, ".p")?, true)?;
                s.try_reserve(1).map_err(|_| Error::memory())?;
                s.push('\n');
                write_out(&sh, s.as_bytes())?;
                s.pop();
                Ok(Value::str(s))
            }),
        );
        let sh = shared.clone();
        interp.register(
            ".d",
            1,
            Rc::new(move |ip: &mut Interp, a: &[Value]| {
                let s = format_value(&ip.syms, one(a, ".d")?, true)?;
                write_out(&sh, s.as_bytes())?;
                Ok(Value::str(s))
            }),
        );
        let sh = shared.clone();
        interp.register(
            ".w",
            1,
            Rc::new(move |ip: &mut Interp, a: &[Value]| {
                let x = one(a, ".w")?;
                let s = format_value(&ip.syms, x, false)?;
                write_out(&sh, s.as_bytes())?;
                Ok(x.clone())
            }),
        );
        interp.register(".fmt", 2, Rc::new(|ip: &mut Interp, a: &[Value]| Ok(Value::str(fmtfn::render(ip, a, ".fmt")?))));
        let sh = shared.clone();
        interp.register(
            ".pf",
            2,
            Rc::new(move |ip: &mut Interp, a: &[Value]| {
                let mut s = fmtfn::render(ip, a, ".pf")?;
                s.try_reserve(1).map_err(|_| Error::memory())?;
                s.push('\n');
                write_out(&sh, s.as_bytes())?;
                s.pop();
                Ok(Value::str(s))
            }),
        );

        // ----- the interpreter itself -----------------------------------------------------------
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
                Ok(Value::Int(level as Int))
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
                Ok(Value::ints(alloc::vec![used as Int, peak as Int, limit as Int]))
            }),
        );
        interp.register(
            ".E",
            1,
            Rc::new(|ip: &mut Interp, a: &[Value]| {
                // (the text is copied: the program may replace the string it came from)
                let text = klong_core::value::try_str(text_arg(a, ".E")?)?;
                match ip.run_global(&text, "<.E>")? {
                    Some(v) => Ok(v),
                    None => Ok(ip.no_value()),
                }
            }),
        );
        // .try(f; args) and .err(kind; message): an error as a value, and a value as an error.
        // A program that speaks for other nodes (the management service, ticket 108) has to
        // survive their failures and pass them on as what they were.
        interp.register(
            ".try",
            2,
            Rc::new(|ip: &mut Interp, a: &[Value]| {
                let (f, args) = match a {
                    [f, args] => (f, args),
                    _ => return Err(Error::arity(klong_core::msg!(".try takes 2 arguments, got {}", a.len()))),
                };
                if !matches!(f, Value::Fn(_) | Value::Prim(_)) {
                    return Err(Error::typ(klong_core::msg!(".try: x must be a function, got {}", f.describe())));
                }
                let args = match args {
                    Value::Str(_) | Value::Dict(_) => return Err(Error::typ(klong_core::msg!(".try: y must be the list of arguments, got {}", args.describe()))),
                    v if v.is_empty() => Vec::new(),
                    v => v.items().map_err(|_| Error::typ(klong_core::msg!(".try: y must be the list of arguments, got {}", v.describe())))?,
                };
                let n = args.len().min(255) as u8;
                let mut r = klong_core::value::try_vec(3)?;
                match ip.apply(f, args, Some(n)) {
                    Ok(v) => {
                        r.push(Value::Int(1));
                        r.push(v);
                    }
                    // Ctrl-C and .x are not for the program to swallow
                    Err(e) if e.kind == klong_core::Kind::Interrupt => return Err(e),
                    Err(e) => {
                        r.push(Value::Int(0));
                        r.push(Value::text(e.kind.name())?);
                        r.push(Value::text(&e.message)?);
                    }
                }
                Value::from_items(r)
            }),
        );
        interp.register(
            ".err",
            2,
            Rc::new(|ip: &mut Interp, a: &[Value]| {
                let (k, m) = match a {
                    [k, m] => (k, m),
                    _ => return Err(Error::arity(klong_core::msg!(".err takes 2 arguments, got {}", a.len()))),
                };
                let name = match k {
                    Value::Str(s) => s.as_str(),
                    Value::Sym(s) => ip.syms.name(*s),
                    v => return Err(Error::typ(klong_core::msg!(".err: x must be the name of an error kind, got {}", v.describe()))),
                };
                use klong_core::Kind as K;
                let kind = [K::Syntax, K::Name, K::Arity, K::Type, K::Shape, K::Range, K::System, K::Memory, K::Internal, K::Remote].into_iter().find(|c| c.name() == name);
                let Some(kind) = kind else {
                    return Err(Error::range(klong_core::msg!(".err: {} is not an error kind", name)));
                };
                match m {
                    Value::Str(s) => Err(Error::new(kind, klong_core::msg!("{}", s.as_str()))),
                    v => Err(Error::typ(klong_core::msg!(".err: y must be the message, a string, got {}", v.describe()))),
                }
            }),
        );
        interp.register(
            ".module",
            1,
            Rc::new(|ip: &mut Interp, a: &[Value]| {
                let x = one(a, ".module")?;
                if is_false(x) {
                    ip.module_stop()?;
                } else {
                    let name = match x {
                        Value::Sym(s) => klong_core::value::try_str(ip.syms.name(*s).split('`').next().unwrap_or(""))?,
                        Value::Str(s) => klong_core::value::try_str(s.as_str())?,
                        v => format_value(&ip.syms, v, true)?,
                    };
                    ip.module_start(&name)?;
                }
                Ok(ip.no_value())
            }),
        );
        let sh = shared.clone();
        interp.register(
            ".ver",
            0,
            Rc::new(move |_: &mut Interp, _: &[Value]| {
                let mut d = Dict::default();
                d.set(Value::text("version")?, Value::text(env!("CARGO_PKG_VERSION"))?)?;
                d.set(Value::text("int")?, Value::Int(klong_core::INT_BITS as Int))?;
                d.set(Value::text("float")?, Value::Int(klong_core::FLT_BITS as Int))?;
                let mut features = Vec::new();
                for (on, name) in [(cfg!(feature = "files"), "files"), (cfg!(feature = "lib"), "lib"), (cfg!(feature = "int32"), "int32"), (cfg!(feature = "float32"), "float32"), (cfg!(feature = "timers"), "timers"), (cfg!(feature = "ipc"), "ipc")] {
                    if on {
                        features.push(Value::text(name)?);
                    }
                }
                d.set(Value::text("features")?, Value::from_items(features)?)?;
                let name = match sh.platform.try_borrow() {
                    Ok(p) => Value::text(p.name())?,
                    Err(_) => Value::text("unknown")?,
                };
                d.set(Value::text("platform")?, name)?;
                Ok(Value::dict(d))
            }),
        );

        // ----- time, chance, leaving ------------------------------------------------------------
        let sh = shared.clone();
        interp.register(
            ".pc",
            0,
            Rc::new(move |_: &mut Interp, _: &[Value]| {
                let now = sh.platform.try_borrow().map(|p| p.now_us()).unwrap_or(0);
                let start = sh.io.try_borrow().map(|io| io.start_us).unwrap_or(0);
                Ok(Value::Flt(now.saturating_sub(start) as Flt / 1e6))
            }),
        );
        let sh = shared.clone();
        interp.register(
            ".rn",
            0,
            Rc::new(move |_: &mut Interp, _: &[Value]| {
                let mut io = sh.io.try_borrow_mut().map_err(|_| Error::internal("runtime state in use"))?;
                // xorshift64*: small, no state beyond one word, good enough for simulations
                let mut x = io.rng;
                x ^= x >> 12;
                x ^= x << 25;
                x ^= x >> 27;
                io.rng = x;
                let r = x.wrapping_mul(0x2545_f491_4f6c_dd1d);
                #[cfg(not(feature = "float32"))]
                let f = (r >> 11) as Flt / (1u64 << 53) as Flt;
                #[cfg(feature = "float32")]
                let f = (r >> 40) as Flt / (1u64 << 24) as Flt;
                Ok(Value::Flt(f))
            }),
        );
        let sh = shared.clone();
        interp.register(
            ".x",
            1,
            Rc::new(move |_: &mut Interp, a: &[Value]| {
                let code = if is_false(one(a, ".x")?) { 0 } else { 1 };
                if let Ok(mut p) = sh.platform.try_borrow_mut() {
                    p.exit(code);
                }
                // a platform that cannot leave stops the program
                Err(Error::interrupt())
            }),
        );
        // .e (Epsilon), as KlongPy has it
        let e = interp.syms.intern(".e");
        interp.set_global(e, Value::Flt(1e-18));

        #[cfg(feature = "files")]
        chan::register(&mut interp, &shared);
        #[cfg(not(feature = "files"))]
        {
            #[cfg(feature = "lib")]
            {
                // without files, .l still loads the library that is part of the image
                interp.register(
                    ".l",
                    1,
                    Rc::new(|ip: &mut Interp, a: &[Value]| {
                        let name = text_arg(a, ".l")?;
                        match library::find(name) {
                            Some((label, text)) => match ip.load(text, label)? {
                                Some(v) => Ok(v),
                                None => Ok(ip.no_value()),
                            },
                            None => Err(Error::system(klong_core::msg!("file does not exist: {}", name)).with_hint("this build has no file system; .l loads the built-in library only")),
                        }
                    }),
                );
            }
        }
        let events = events::Events::new(shared.clone());
        events::register(&mut interp, &events);
        Runtime { interp, shared, events }
    }

    /// Run a program; the value of its last statement.
    pub fn run(&mut self, text: &str, name: &str) -> Result<Option<Value>> {
        self.interp.run(text, name)
    }

    /// Replace the directories `.l` looks in (the host sets them from KLONGPATH).
    pub fn set_load_path(&mut self, dirs: Vec<String>) {
        if let Ok(mut io) = self.shared.io.try_borrow_mut() {
            io.path = dirs;
        }
    }

    /// Start-up: run `boot.kg` if the file system has one. Returns whether there was one.
    #[cfg(feature = "files")]
    pub fn boot(&mut self) -> Result<bool> {
        match chan::read_file(&self.shared, "boot.kg") {
            Ok(text) => self.interp.load(&text, "boot.kg").map(|_| true),
            Err(_) => Ok(false),
        }
    }
}

/// Write to the To channel: the console, or the file `.tc` selected.
pub(crate) fn write_out<P: Platform + 'static>(sh: &Rc<Shared<P>>, bytes: &[u8]) -> Result<()> {
    #[cfg(feature = "files")]
    {
        let to = sh.io.try_borrow().map(|io| io.to.clone()).unwrap_or(Value::Undefined);
        if !matches!(to, Value::Undefined) {
            return chan::write::<P>(sh, &to, bytes);
        }
    }
    if let Ok(mut p) = sh.platform.try_borrow_mut() {
        p.console_write(bytes);
    }
    Ok(())
}
