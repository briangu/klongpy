//! klong-rt: what a Klong runtime needs around the core, still without naming a board or an
//! operating system: the [`Platform`] and [`FileSystem`] traits, the system functions
//! (docs/native/index.md), the tick hook that carries Ctrl-C from the console into a
//! running evaluation, heap accounting ([`heap::Counting`]) and the Klong library as text.
//!
//! Features: `files` (channels, `.l`, the file functions; needs `Platform::fs`), `lib` (the
//! pure-Klong library of `klongpy/lib` in the image, loaded by `.l("math")`). The event loop,
//! the line editor and the upload receiver, IPC and timers are in the console and ipc modules.
#![no_std]
#![deny(clippy::unwrap_used, clippy::expect_used)]

extern crate alloc;

pub mod declare;
pub mod events;
pub mod heap;
#[cfg(feature = "ipc")]
pub mod ipc;
pub mod link;
pub mod memfs;
#[cfg(feature = "ipc")]
pub mod memlink;
#[cfg(feature = "ipc")]
pub mod node;
#[cfg(feature = "ipc")]
pub mod selftest;
pub mod platform;

#[cfg(feature = "files")]
mod chan;
mod fmtfn;
mod json;
mod modules;
mod strfn;
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
    /// Console bytes read while an evaluation was running (for the line editor;
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
    /// The modules `.import` has made (`modules.rs`).
    pub(crate) mods: modules::ModsRef,
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
    /// The directories of the files being run or loaded, the innermost last (empty text: a
    /// file in the current directory, or a library file). A relative name that `path` and
    /// the library do not have is looked for in the last one.
    pub home: Vec<String>,
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
        p.slow_tick();
        self.shared.scan_console(&mut p)
    }
}

pub struct Runtime<P: Platform + 'static> {
    pub interp: Interp,
    pub shared: Rc<Shared<P>>,
    /// Timers, queued calls and IPC connections (`events.rs`).
    pub events: Rc<events::Events<P>>,
    /// What the program has declared with `.extern`, `.entry` and `.param` (the static profile).
    pub declarations: declare::Declarations,
}

pub(crate) fn one<'a>(a: &'a [Value], name: &str) -> Result<&'a Value> {
    a.first().ok_or_else(|| Error::arity(klong_core::msg!("{} takes 1 argument, got 0", name)))
}

pub(crate) fn text_arg<'a>(a: &'a [Value], name: &str) -> Result<&'a str> {
    match one(a, name)? {
        Value::Str(s) => Ok(s.as_str()),
        v => Err(Error::typ(klong_core::msg!("{}: x must be a string, got {}", name, v.describe_long()))),
    }
}

/// The dictionary `.try` hands back for an error: "kind", "message", "source", "line" and "col" (see `.try`).
fn error_dict(e: &Error) -> Result<Value> {
    let (source, line, col) = match (&e.source, e.span) {
        (Some(src), Some(span)) => {
            let (l, c) = src.line_col(span.0);
            (src.name.as_str(), l as Int, c as Int)
        }
        _ => ("", 0, 0),
    };
    let mut d = Dict::default();
    d.set(Value::text("kind")?, Value::text(e.kind.name())?)?;
    d.set(Value::text("message")?, Value::text(&e.message)?)?;
    d.set(Value::text("source")?, Value::text(source)?)?;
    d.set(Value::text("line")?, Value::Int(line))?;
    d.set(Value::text("col")?, Value::Int(col))?;
    Ok(Value::dict(d))
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
        path.push(String::new()); // the current directory, as the platform names it
        path.push(String::from("lib"));
        let shared = Rc::new(Shared {
            platform: RefCell::new(platform),
            pending: RefCell::new(Vec::new()),
            interrupt_armed: core::cell::Cell::new(false),
            ctrl_c_seen: core::cell::Cell::new(false),
            armed_hit: core::cell::Cell::new(false),
            io: RefCell::new(Io { from: Value::Undefined, to: Value::Undefined, cin: Value::Undefined, cout: Value::Undefined, rng, start_us, path, home: Vec::new() }),
            mods: RefCell::new(modules::Mods::default()),
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
        let declarations = declare::register(&mut interp);
        interp.register(".fmt", klong_core::value::WILDCARD, Rc::new(|ip: &mut Interp, a: &[Value]| Ok(Value::str(fmtfn::render(ip, a, ".fmt")?))));
        let sh = shared.clone();
        interp.register(
            ".pf",
            klong_core::value::WILDCARD,
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
        // .try(f; args) and .err(kind; message) or .err(e): an error as a value, and a value as an error (docs/native/management-node.md).
        // `.try` gives [1 result] or [0 e], where e is a dictionary: "kind" (the name of the kind, a string), "message", "source" (the
        // name of the program text the error was found in, "" when it has none), "line" and "col" (where, 1-based; 0 when the error has
        // no position). `.err` takes the kind and the message, or such a dictionary: what `.try` caught is what `.err` raises again. A
        // program that speaks for other nodes (the management service) has to survive their failures and pass them on as what they were.
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
                let mut r = klong_core::value::try_vec(2)?;
                match ip.apply(f, args, Some(n)) {
                    Ok(v) => {
                        r.push(Value::Int(1));
                        r.push(v);
                    }
                    // Ctrl-C and .x are not for the program to swallow
                    Err(e) if e.kind == klong_core::Kind::Interrupt => return Err(e),
                    Err(e) => {
                        r.push(Value::Int(0));
                        r.push(error_dict(&e)?);
                    }
                }
                Value::from_items(r)
            }),
        );
        interp.register(
            ".err",
            WILDCARD,
            Rc::new(|ip: &mut Interp, a: &[Value]| {
                use klong_core::Kind as K;
                let (k, m) = match a {
                    [Value::Dict(d)] => {
                        let d = d.borrow();
                        match (d.get(&Value::text("kind")?), d.get(&Value::text("message")?)) {
                            (Some(k), Some(m)) => (k.clone(), m.clone()),
                            _ => return Err(Error::typ(".err: a dictionary needs the entries \"kind\" and \"message\"")),
                        }
                    }
                    [k, m] => (k.clone(), m.clone()),
                    _ => return Err(Error::arity(klong_core::msg!(".err takes the kind and the message, or an error dictionary, got {} arguments", a.len()))),
                };
                let name = match &k {
                    Value::Str(s) => s.as_str(),
                    Value::Sym(s) => ip.syms.name(*s),
                    v => return Err(Error::typ(klong_core::msg!(".err: x must be the name of an error kind, got {}", v.describe()))),
                };
                let kind = [K::Syntax, K::Name, K::Arity, K::Type, K::Shape, K::Range, K::System, K::Memory, K::Internal, K::Remote].into_iter().find(|c| c.name() == name);
                let Some(kind) = kind else {
                    return Err(Error::range(klong_core::msg!(".err: {} is not an error kind", name)));
                };
                match &m {
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
        // `.now()`: seconds since 1970-01-01 00:00:00 UTC, a float, where the platform has a clock that knows the date (a desktop does; a
        // board that was not told the time does not, and says so). `.pc()` is the clock for measuring: it starts at 0 with the program.
        let sh = shared.clone();
        interp.register(
            ".now",
            0,
            Rc::new(move |_: &mut Interp, _: &[Value]| {
                let wall = sh.platform.try_borrow().map_err(|_| Error::internal("runtime state in use"))?.wall_clock();
                match wall {
                    Some(t) => Ok(Value::Flt(t as Flt)),
                    None => Err(Error::system(klong_core::dmsg!("no wall clock", "this platform has no wall clock: .now() needs one that knows the date (.pc() measures time since the start)"))),
                }
            }),
        );
        // `.rn()`: xorshift64* (Vigna, 2014), one 64-bit word of state, the same on every target. The float is the top 53 bits of the
        // output over 2^53 (64-bit floats) or the top 24 over 2^24 (32-bit floats: the 64-bit value cut to 24 bits). `.seed(n)` sets the
        // state from the integer n through one step of splitmix64, so that the sequence after `.seed(n)` is the same on every target.
        let sh = shared.clone();
        interp.register(
            ".rn",
            0,
            Rc::new(move |_: &mut Interp, _: &[Value]| {
                let mut io = sh.io.try_borrow_mut().map_err(|_| Error::internal("runtime state in use"))?;
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
            ".seed",
            1,
            Rc::new(move |_: &mut Interp, a: &[Value]| {
                let n = match one(a, ".seed")? {
                    Value::Int(n) => *n,
                    v => return Err(Error::typ(klong_core::msg!(".seed: x must be an integer, got {}", v.describe()))),
                };
                let mut io = sh.io.try_borrow_mut().map_err(|_| Error::internal("runtime state in use"))?;
                // splitmix64 of the seed (a negative integer is its 64-bit two's complement on every build); never the all-zero state
                let mut z = (n as i64 as u64).wrapping_add(0x9e37_79b9_7f4a_7c15);
                z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                z ^= z >> 31;
                io.rng = if z == 0 { 0x9e37_79b9_7f4a_7c15 } else { z };
                Ok(Value::Int(n))
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
        modules::register(&mut interp, &shared);
        strfn::register(&mut interp);
        json::register(&mut interp);
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
                let sh = shared.clone();
                interp.register(
                    ".l",
                    1,
                    Rc::new(move |ip: &mut Interp, a: &[Value]| {
                        let name = text_arg(a, ".l")?;
                        match library::find(name) {
                            Some((label, text)) if modules::calls_export(text) => {
                                let d = modules::import_loaded(&sh, ip, label, modules::Text::Static(text))?;
                                modules::bind_alias(ip, label, d)?;
                                Ok(ip.no_value())
                            }
                            Some((label, text)) => match ip.load_static(text, label)? {
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
        Runtime { interp, shared, events, declarations }
    }

    /// Run a program that is handed over, statement by statement (see `Interp::run_streamed_owned`).
    pub fn run_streamed(&mut self, text: String, name: &str) -> Result<Option<Value>> {
        self.interp.run_streamed_owned(text, name)
    }

    /// Run a program; the value of its last statement.
    pub fn run(&mut self, text: &str, name: &str) -> Result<Option<Value>> {
        self.interp.run(text, name)
    }

    /// The console input the program has read from the platform and not used, taken out of the channel (see `chan::take_console_unread`).
    #[cfg(feature = "files")]
    pub fn take_console_unread(&self) -> Vec<u8> {
        chan::take_console_unread(&self.shared)
    }

    /// Replace the directories `.l` looks in (the host sets them from KLONGPATH).
    pub fn set_load_path(&mut self, dirs: Vec<String>) {
        if let Ok(mut io) = self.shared.io.try_borrow_mut() {
            io.path = dirs;
        }
    }

    /// The directories a bare name of `.import` is looked for in, before the library of the image (KLONGPATH when it is set; none
    /// otherwise: the working directory is not searched unless it is named).
    pub fn set_import_path(&mut self, dirs: Vec<String>) {
        if let Ok(mut m) = self.shared.mods.try_borrow_mut() {
            m.import_path = dirs;
        }
    }

    /// Tell `.l` which file the program being run comes from: a relative name that is found
    /// neither in the load path nor in the library is then looked for next to that file.
    pub fn set_program(&mut self, path: &str) {
        if let Ok(mut io) = self.shared.io.try_borrow_mut() {
            io.home.clear();
            let dir = chan_dir(path);
            io.home.push(dir);
        }
    }

    /// A program from the file in `dir` begins (`""` for the current directory): a relative `.l` that nothing else finds is looked
    /// for there, until `pop_home`.
    pub fn push_home(&self, dir: &str) {
        if let Ok(mut io) = self.shared.io.try_borrow_mut() {
            io.home.push(String::from(dir));
        }
    }

    pub fn pop_home(&self) {
        if let Ok(mut io) = self.shared.io.try_borrow_mut() {
            io.home.pop();
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

/// The directory part of a file name: "" when there is none.
pub(crate) fn chan_dir(path: &str) -> String {
    match path.rfind('/') {
        Some(0) => String::from("/"),
        Some(i) => String::from(&path[..i]),
        None => String::new(),
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
