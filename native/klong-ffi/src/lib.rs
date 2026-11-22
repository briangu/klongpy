//! klong-ffi: the board runtime behind a C ABI (`include/klong.h`), as a static library for
//! an application built with a vendor SDK. The C side supplies the
//! console bytes, the clock, the heap and the reset; this crate turns them into the `Uart`,
//! `Clock` and `System` of `klong-board` and drives `klong-console` through
//! `klong_board::{start, step}`.
//!
//! Only integers and pointers cross the interface (no `f32`/`f64`), so the same header is
//! right whether the C side is built for a soft-float or a hard-float ABI.
//!
//! Unsafe code is in three places, each commented: the two statics behind `Global` (one
//! task only: the contract of klong.h), the calls through the C function pointers, and the
//! reads through the pointers C hands in (`klong_init`, `klong_heap`).
#![no_std]
#![deny(unsafe_op_in_unsafe_fn)]
#![deny(clippy::unwrap_used, clippy::expect_used)]

extern crate alloc;
// On a host the library is linked into an ordinary C program: std supplies the panic
// machinery there. On a bare-metal target the handler at the end of this file does.
#[cfg(not(target_os = "none"))]
extern crate std;

use core::alloc::{GlobalAlloc, Layout};
use core::cell::UnsafeCell;
use core::ffi::{c_char, c_void};

use klong_board::{BoardConfig, BoardPlatform, Clock, System, Uart};
use klong_console::{Console, ConsoleConfig};
use klong_rt::heap::Counting;
use klong_rt::memfs::MemFs;

pub const KLONG_ABI_VERSION: u32 = 1;
pub const KLONG_FLAG_CRLF: u32 = 1;
pub const KLONG_FLAG_DUMB: u32 = 2;
pub const KLONG_FLAG_NO_FS: u32 = 4;
pub const KLONG_OK: i32 = 0;
pub const KLONG_ERR_ARG: i32 = -1;
pub const KLONG_ERR_STATE: i32 = -2;

type Ctx = *mut c_void;

/// `struct klong_config` of klong.h, field for field.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct KlongConfig {
    pub abi_version: u32,
    pub size: u32,
    pub ctx: Ctx,
    pub alloc: Option<unsafe extern "C" fn(Ctx, usize, usize) -> *mut c_void>,
    pub free: Option<unsafe extern "C" fn(Ctx, *mut c_void, usize, usize)>,
    pub heap_limit: usize,
    pub heap_reserve: usize,
    pub stack_bytes: usize,
    pub write: Option<unsafe extern "C" fn(Ctx, *const u8, usize)>,
    pub read: Option<unsafe extern "C" fn(Ctx, *mut u8, usize) -> usize>,
    pub now_ms: Option<unsafe extern "C" fn(Ctx) -> u64>,
    pub idle: Option<unsafe extern "C" fn(Ctx)>,
    pub wait: Option<unsafe extern "C" fn(Ctx, u32)>,
    pub reset: Option<unsafe extern "C" fn(Ctx, i32)>,
    pub entropy: Option<unsafe extern "C" fn(Ctx) -> u64>,
    pub fatal: Option<unsafe extern "C" fn(Ctx, *const c_char, usize)>,
    pub name: *const c_char,
    pub flags: u32,
    /// Of `stack_bytes`, what is kept for the bottom of the recursion (0: a quarter).
    pub stack_keep: usize,
    /// Bytes the allocator keeps for each block (its header): see `Counting::set_overhead` (0: none).
    pub heap_overhead: usize,
    /// Blocks are counted rounded up to a multiple of this (see `Counting::set_round`; 0: as asked).
    pub heap_round: usize,
    /// Optional: resize a block where it is (see klong.h). Returns 1 when done.
    pub grow: Option<unsafe extern "C" fn(Ctx, *mut c_void, usize, usize, usize) -> i32>,
}

/// A static that one task uses. klong.h requires every call and callback to be on the task
/// that called `klong_init`; the library starts no thread and installs no interrupt handler.
struct Global<T>(UnsafeCell<T>);
// SAFETY: single task by the contract above; the cell is never touched concurrently.
unsafe impl<T> Sync for Global<T> {}

/// The callbacks, copied from the config by `klong_init`. Written once, before anything can
/// allocate or run; afterwards only read, by value.
static HOOKS: Global<Option<KlongConfig>> = Global(UnsafeCell::new(None));

fn hooks() -> Option<KlongConfig> {
    // SAFETY: a read of a `Copy` value; the only write is in `klong_init`, on the same task,
    // and no reference into the cell outlives a statement.
    unsafe { *HOOKS.0.get() }
}

// ----- heap ----------------------------------------------------------------------------------

/// The C side's allocator. Before `klong_init` there is none and every request fails.
struct CHeap;

// SAFETY: `alloc` returns what the C `alloc` callback returns, whose contract in klong.h is
// `size` bytes aligned to `align` or NULL, exclusively the caller's until `free`; `dealloc`
// hands the same pointer, size and alignment back. `realloc` is GlobalAlloc's default
// (allocate, copy, free), which needs nothing more.
unsafe impl GlobalAlloc for CHeap {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        match hooks() {
            // SAFETY: a callback of klong.h, called with its context on the runtime's task.
            Some(KlongConfig { alloc: Some(f), ctx, .. }) => unsafe { f(ctx, layout.size(), layout.align()) as *mut u8 },
            _ => core::ptr::null_mut(),
        }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if let Some(KlongConfig { free: Some(f), ctx, .. }) = hooks() {
            // SAFETY: `ptr` came from the `alloc` callback with this size and alignment.
            unsafe { f(ctx, ptr as *mut c_void, layout.size(), layout.align()) }
        }
    }
}

/// Everything the runtime allocates is counted here: the limit, the reserve, `.mem()`.
/// The counters are statics, so they are in internal RAM whatever memory `alloc` returns.
#[global_allocator]
static HEAP: Counting<CHeap> = Counting::new(CHeap);

/// `Counting`'s way to the C side's `grow`.
unsafe fn cgrow(ptr: *mut u8, old: usize, new: usize, align: usize) -> bool {
    match hooks() {
        // SAFETY: a callback of klong.h, called with its context on the runtime's task, for a block of its `alloc`.
        Some(KlongConfig { grow: Some(f), ctx, .. }) => unsafe { f(ctx, ptr as *mut c_void, old, new, align) != 0 },
        _ => false,
    }
}

fn mem() -> (usize, usize, usize) {
    (HEAP.used(), HEAP.peak(), HEAP.limit())
}
fn pressed() -> bool {
    HEAP.pressed()
}

// ----- the board, made of callbacks ----------------------------------------------------------

struct CUart;
impl Uart for CUart {
    fn quiet(&self) -> bool {
        #[cfg(feature = "lane")]
        {
            lane::quiet()
        }
        #[cfg(not(feature = "lane"))]
        false
    }
    fn read(&mut self, buf: &mut [u8]) -> usize {
        let n = match hooks() {
            // SAFETY: klong.h: `read` writes at most `cap` bytes into `buf` and returns the
            // count; the count is clamped in case it does not keep the second half.
            Some(KlongConfig { read: Some(f), ctx, .. }) => unsafe { f(ctx, buf.as_mut_ptr(), buf.len()) }.min(buf.len()),
            _ => 0,
        };
        #[cfg(feature = "lane")]
        {
            if n > 0 {
                // somebody typed on the local console: the lane's peer no longer owns the output
                lane::local_input();
            } else {
                // nothing typed on the console port: the console channel of the lane
                return lane::console_read(buf);
            }
        }
        n
    }
    fn write(&mut self, bytes: &[u8]) {
        if let Some(KlongConfig { write: Some(f), ctx, .. }) = hooks() {
            // SAFETY: klong.h: `write` reads `len` bytes and keeps no pointer.
            unsafe { f(ctx, bytes.as_ptr(), bytes.len()) }
        }
        #[cfg(feature = "lane")]
        lane::console_write(bytes);
    }
}

struct CClock;
impl CClock {
    fn now_ms() -> u64 {
        match hooks() {
            // SAFETY: a callback of klong.h with its context.
            Some(KlongConfig { now_ms: Some(f), ctx, .. }) => unsafe { f(ctx) },
            _ => 0,
        }
    }
}
impl Clock for CClock {
    fn now_us(&self) -> u64 {
        CClock::now_ms().saturating_mul(1000)
    }
}

struct CSystem {
    name: [u8; 31],
    name_len: usize,
}
impl System for CSystem {
    fn name(&self) -> &str {
        core::str::from_utf8(&self.name[..self.name_len]).unwrap_or("board")
    }
    fn idle(&mut self) {
        if let Some(KlongConfig { idle: Some(f), ctx, .. }) = hooks() {
            // SAFETY: a callback of klong.h with its context.
            unsafe { f(ctx) }
        }
        // the lane is served at every turn of the main loop and every 4096 steps of an
        // evaluation, so PING and ACK frames are answered while Klong code runs
        #[cfg(feature = "lane")]
        lane::pump();
    }
    fn wait(&mut self, deadline_us: Option<u64>) {
        if let Some(KlongConfig { wait: Some(f), ctx, .. }) = hooks() {
            let max_ms = match deadline_us {
                None => u32::MAX,
                // rounded up, so a timer is not polled in a busy loop during its last millisecond
                Some(d) => d.saturating_sub(self::CClock.now_us()).div_ceil(1000).min(u32::MAX as u64 - 1) as u32,
            };
            // SAFETY: a callback of klong.h with its context.
            unsafe { f(ctx, max_ms) }
        }
    }
    fn entropy(&mut self) -> Option<u64> {
        match hooks() {
            // SAFETY: a callback of klong.h with its context.
            Some(KlongConfig { entropy: Some(f), ctx, .. }) => Some(unsafe { f(ctx) }),
            _ => None,
        }
    }
    fn reset(&mut self, code: i32) {
        if let Some(KlongConfig { reset: Some(f), ctx, .. }) = hooks() {
            // SAFETY: a callback of klong.h with its context. It normally does not return.
            unsafe { f(ctx, code) }
        }
    }
}

type Board = BoardPlatform<CUart, CClock, CSystem, MemFs>;

struct State {
    console: Console<Board>,
    /// True while `klong_step` runs: a callback that calls back in gets "nothing to do".
    stepping: bool,
    /// Since when `klong_step` has had nothing to do (ms), and whether the functions kept as text were let go for it.
    idle_since: Option<u64>,
    lazy_released: bool,
}

static STATE: Global<Option<State>> = Global(UnsafeCell::new(None));

// ----- the C interface -----------------------------------------------------------------------

/// See klong.h.
///
/// # Safety
/// `config` is null or points to a `klong_config` whose callbacks keep the contracts of
/// klong.h and whose `name` is null or a NUL-terminated string. Called once, from the task
/// that will make every later call.
#[no_mangle]
pub unsafe extern "C" fn klong_init(config: *const KlongConfig) -> i32 {
    if config.is_null() {
        return KLONG_ERR_ARG;
    }
    // SAFETY: the first two fields are read before the rest, so that a caller built against
    // another version of the header is refused without reading past its struct.
    let (version, size) = unsafe { ((*config).abi_version, (*config).size) };
    if version != KLONG_ABI_VERSION || size as usize != core::mem::size_of::<KlongConfig>() {
        return KLONG_ERR_ARG;
    }
    // SAFETY: non-null, and of this version and size: a whole `klong_config`.
    let cfg = unsafe { *config };
    if cfg.alloc.is_none() || cfg.free.is_none() || cfg.write.is_none() || cfg.read.is_none() || cfg.now_ms.is_none() || cfg.stack_bytes == 0 {
        return KLONG_ERR_ARG;
    }
    // SAFETY: this task is the only user of the statics (klong.h). No reference into either
    // cell is alive here: nothing of the runtime exists before the first successful init.
    unsafe {
        if (*HOOKS.0.get()).is_some() {
            return KLONG_ERR_STATE;
        }
        *HOOKS.0.get() = Some(cfg);
    }
    HEAP.set_overhead(cfg.heap_overhead);
    HEAP.set_round(cfg.heap_round);
    HEAP.set_limit(cfg.heap_limit, cfg.heap_reserve);
    if cfg.grow.is_some() {
        HEAP.set_grow(Some(cgrow));
    }

    let mut system = CSystem { name: [0; 31], name_len: 0 };
    let fallback = b"board";
    system.name[..fallback.len()].copy_from_slice(fallback);
    system.name_len = fallback.len();
    if !cfg.name.is_null() {
        let mut n = 0;
        // SAFETY: `name` is NUL-terminated (klong.h); bytes are read up to the terminator
        // and at most 31 of them.
        unsafe {
            while n < system.name.len() && *cfg.name.add(n) != 0 {
                system.name[n] = *cfg.name.add(n) as u8;
                n += 1;
            }
        }
        if n > 0 && core::str::from_utf8(&system.name[..n]).is_ok() {
            system.name_len = n;
        } else {
            system.name[..fallback.len()].copy_from_slice(fallback);
        }
    }

    let fs = if cfg.flags & KLONG_FLAG_NO_FS != 0 { None } else { Some(MemFs::new()) };
    let mut platform: Board = BoardPlatform::new(CUart, CClock, system, fs);
    platform.crlf = cfg.flags & KLONG_FLAG_CRLF != 0;
    let board = BoardConfig {
        stack_bytes: cfg.stack_bytes,
        stack_keep: cfg.stack_keep,
        mem: Some(mem),
        mem_pressed: Some(pressed),
        console: ConsoleConfig { dumb: cfg.flags & KLONG_FLAG_DUMB != 0, ..ConsoleConfig::default() },
        ..BoardConfig::default()
    };
    #[cfg_attr(not(feature = "heapdiag"), allow(unused_mut))]
    let mut console = klong_board::start(platform, board);
    #[cfg(feature = "heapdiag")]
    heapdiag::register(&mut console.rt.interp);
    // SAFETY: as above; `STATE` is empty until this write.
    unsafe { *STATE.0.get() = Some(State { console, stepping: false, idle_since: None, lazy_released: false }) };
    KLONG_OK
}

/// See klong.h.
#[no_mangle]
pub extern "C" fn klong_step() -> i32 {
    let state = STATE.0.get();
    // SAFETY: one task (klong.h), so nothing else touches the cell. A callback that calls
    // back in while the console is borrowed below is turned away here, by a look at the
    // flag through a shared reference that ends with the statement.
    match unsafe { &*state } {
        None => return -1,
        Some(s) if s.stepping => return 0,
        Some(_) => {}
    }
    // SAFETY: not stepping, so no other reference into the cell exists.
    match unsafe { &mut *state } {
        Some(s) => {
            s.stepping = true;
            #[cfg(feature = "lane")]
            lane::turn_begin(&mut s.console);
            let mut busy = klong_board::step(&mut s.console);
            #[cfg(feature = "lane")]
            {
                lane::turn_end(&mut s.console);
                busy |= lane::selftest_step(&mut s.console.rt);
            }
            s.stepping = false;
            // the functions kept as text that programs parsed are kept while the node is busy (a call of the library after a call parses
            // nothing) and let go 300 ms after it has nothing to do: a node at rest holds no trees
            if busy {
                s.idle_since = None;
                s.lazy_released = false;
            } else {
                let now = CClock::now_ms();
                match s.idle_since {
                    None => s.idle_since = Some(now),
                    Some(t) if !s.lazy_released && now.saturating_sub(t) >= 300 => {
                        s.console.rt.interp.release_lazy();
                        s.lazy_released = true;
                    }
                    _ => {}
                }
            }
            busy as i32
        }
        None => -1,
    }
}

/// See klong.h.
#[no_mangle]
pub extern "C" fn klong_run() {
    loop {
        match klong_step() {
            1 => {}
            0 => {
                // SAFETY: as in `klong_step`; not stepping here, so this is the only reference.
                if let Some(s) = unsafe { &mut *STATE.0.get() } {
                    // Platform::wait: the `wait` callback, until input or the next timer
                    s.console.rt.events_wait();
                }
            }
            _ => return,
        }
    }
}

/// See klong.h.
///
/// # Safety
/// Each pointer is null or valid for a write of a `size_t`.
#[no_mangle]
pub unsafe extern "C" fn klong_heap(used: *mut usize, peak: *mut usize, limit: *mut usize) {
    let (u, p, l) = mem();
    for (ptr, v) in [(used, u), (peak, p), (limit, l)] {
        if !ptr.is_null() {
            // SAFETY: non-null, so valid for a write by the contract above.
            unsafe { ptr.write(v) }
        }
    }
}

// ----- C functions as system functions ------------------------------------------------------

/// `klong_cfun_register`: a C function becomes a system function (feature `cfun`). The arguments cross as `klong_val` structures
/// (pointers into the interpreter's values, valid during the call), the result comes back in one, and an error is a non-zero return
/// with a message. While the C function runs it may call `klong_eval` and `klong_load_static`, which run Klong text in the
/// interpreter that called it. 64-bit integers and floats only (the Pico bench's build).
#[cfg(feature = "cfun")]
mod cfun {
    use super::Global;
    use alloc::rc::Rc;
    use alloc::string::String;
    use alloc::vec::Vec;
    use core::cell::UnsafeCell;
    use core::ffi::{c_char, c_void};
    use klong_core::error::Error;
    use klong_core::interp::Interp;
    use klong_core::value::{Int, Value, WILDCARD};

    #[cfg(any(feature = "int32", feature = "float32"))]
    compile_error!("the cfun feature needs 64-bit integers and floats (no int32, no float32)");

    const T_OTHER: i32 = 0;
    const T_INT: i32 = 1;
    const T_FLT: i32 = 2;
    const T_INTS: i32 = 3;
    const T_FLTS: i32 = 4;
    const T_STR: i32 = 5;
    /// A function takes at most this many arguments.
    const MAX_ARGS: usize = 8;

    /// `struct klong_val` of klong.h, field for field.
    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct Val {
        tag: i32,
        rank: i32,
        i: i64,
        f: f64,
        len: usize,
        data: *const c_void,
    }

    impl Val {
        const NONE: Val = Val { tag: T_OTHER, rank: 0, i: 0, f: 0.0, len: 0, data: core::ptr::null() };
    }

    /// `int32_t fn(void *ctx, const klong_val *args, size_t nargs, klong_val *out)`: 0, or an error code (1 type, 2 range, 3 system) with a
    /// NUL-terminated message in `out->data` (4 memory and 5 interrupt, which `klong_eval` returns for those errors, are passed on as such).
    type Cfun = unsafe extern "C" fn(*mut c_void, *const Val, usize, *mut Val) -> i32;

    /// The interpreter a C function was called from: set for the length of the call, null otherwise.
    static CUR: Global<*mut Interp> = Global(UnsafeCell::new(core::ptr::null_mut()));
    /// What the last `klong_eval` returned, kept so that the pointers in its `klong_val` stay valid until the next one.
    static KEEP: Global<Option<Value>> = Global(UnsafeCell::new(None));
    /// The text of the last error `klong_eval` reported.
    static MSG: Global<Option<String>> = Global(UnsafeCell::new(None));

    fn describe(v: &Value) -> Val {
        match v {
            Value::Int(i) => Val { tag: T_INT, i: *i as i64, ..Val::NONE },
            Value::Flt(f) => Val { tag: T_FLT, f: *f, ..Val::NONE },
            Value::Ints(a) => Val { tag: T_INTS, rank: a.rank() as i32, len: a.data.len(), data: a.data.as_ptr() as *const c_void, ..Val::NONE },
            Value::Flts(a) => Val { tag: T_FLTS, rank: a.rank() as i32, len: a.data.len(), data: a.data.as_ptr() as *const c_void, ..Val::NONE },
            Value::Str(s) => Val { tag: T_STR, len: s.as_str().len(), data: s.as_str().as_ptr() as *const c_void, ..Val::NONE },
            _ => Val::NONE,
        }
    }

    /// The `Value` of what a C function returned. The vector data is copied. A result with `rank` 2 has `i` columns.
    ///
    /// # Safety
    /// `out.data` is valid for `out.len` elements of the tag's type.
    unsafe fn from_val(out: &Val) -> Result<Value, Error> {
        use klong_core::value::Arr;
        let shape = |len: usize| -> Result<Option<[u32; 2]>, Error> {
            if out.rank != 2 {
                return Ok(None);
            }
            let cols = out.i as usize;
            if cols == 0 || len % cols != 0 || len / cols > u32::MAX as usize || cols > u32::MAX as usize {
                return Err(Error::system("a C function returned a shape that does not fit its data"));
            }
            Ok(Some([(len / cols) as u32, cols as u32]))
        };
        match out.tag {
            T_INT => Ok(Value::Int(out.i as Int)),
            T_FLT => Ok(Value::Flt(out.f)),
            T_INTS | T_FLTS | T_STR if out.len > 0 && out.data.is_null() => Err(Error::system("a C function returned a vector with no data")),
            T_INTS => {
                let dims = shape(out.len)?;
                let mut v: Vec<Int> = klong_core::value::try_vec(out.len)?;
                // SAFETY: by the contract above `data` has `len` aligned 64-bit integers.
                v.extend_from_slice(unsafe { core::slice::from_raw_parts(out.data as *const i64, out.len) });
                Ok(match dims {
                    Some(d) => Value::Ints(Rc::new(Arr::shaped(v, &d))),
                    None => Value::ints(v),
                })
            }
            T_FLTS => {
                let dims = shape(out.len)?;
                let mut v: Vec<f64> = klong_core::value::try_vec(out.len)?;
                // SAFETY: as above, for doubles.
                v.extend_from_slice(unsafe { core::slice::from_raw_parts(out.data as *const f64, out.len) });
                Ok(match dims {
                    Some(d) => Value::Flts(Rc::new(Arr::shaped(v, &d))),
                    None => Value::flts(v),
                })
            }
            T_STR => {
                // SAFETY: as above, `len` bytes.
                let b = unsafe { core::slice::from_raw_parts(out.data as *const u8, out.len) };
                match core::str::from_utf8(b) {
                    Ok(s) => Value::text(s),
                    Err(_) => Err(Error::system("a C function returned a string that is not UTF-8")),
                }
            }
            _ => Err(Error::system("a C function returned no value")),
        }
    }

    /// The message a C function gave for an error, as an `Error`.
    ///
    /// # Safety
    /// `msg` is null or NUL-terminated.
    unsafe fn error_of(code: i32, msg: *const c_void) -> Error {
        let text = if msg.is_null() {
            "C function failed"
        } else {
            // SAFETY: NUL-terminated by the contract above.
            let mut n = 0;
            while n < 200 && unsafe { *(msg as *const u8).add(n) } != 0 {
                n += 1;
            }
            // SAFETY: the `n` bytes before the NUL (or the 200 we looked at) are readable.
            core::str::from_utf8(unsafe { core::slice::from_raw_parts(msg as *const u8, n) }).unwrap_or("C function failed")
        };
        let m = match klong_core::value::try_str(text) {
            Ok(m) => m,
            Err(e) => return e,
        };
        match code {
            1 => Error::typ(m),
            2 => Error::range(m),
            4 => Error::memory(),
            5 => Error::interrupt(),
            _ => Error::system(m),
        }
    }

    /// The function a registered C function becomes: its arguments cross as `klong_val`, its result comes back in one.
    fn host_fn(f: Cfun, ctx: usize, out_cap: usize) -> klong_core::value::NativeFn {
Rc::new(move |ip: &mut Interp, a: &[Value]| {
            if a.len() > MAX_ARGS {
                return Err(Error::arity("too many arguments for a C function"));
            }
            let mut args = [Val::NONE; MAX_ARGS];
            for (slot, v) in args.iter_mut().zip(a) {
                *slot = describe(v);
            }
            // a function that returns a vector gets room for `out_cap` elements for the length of the call, so that it keeps no buffer
            // of its own (RAM is what a board has least of)
            let mut scratch: Vec<i64> = Vec::new();
            let mut out = Val::NONE;
            if out_cap > 0 {
                scratch = klong_core::value::try_vec(out_cap)?;
                scratch.resize(out_cap, 0);
                out.len = out_cap;
                out.data = scratch.as_ptr() as *const c_void;
            }
            // SAFETY (all three): one task; `CUR` is the interpreter calling, put back to what it was after the call (a C function
            // may run Klong that calls another); `args` and the values behind them live to the end of the call.
            let saved = unsafe { *CUR.0.get() };
            unsafe { *CUR.0.get() = ip as *mut Interp };
            let code = unsafe { f(ctx as *mut c_void, args.as_ptr(), a.len(), &mut out) };
            unsafe { *CUR.0.get() = saved };
            if code != 0 {
                // SAFETY: the C function's message, NUL-terminated by the contract.
                return Err(unsafe { error_of(code, out.data) });
            }
            if out_cap > 0 && out.data == scratch.as_ptr() as *const c_void && out.len > out_cap {
                return Err(Error::system("a C function returned more than the room it was given"));
            }
            // SAFETY: the C function's result, valid by the contract (and, in the room it was given, checked just above).
            unsafe { from_val(&out) }
        })
    }

    /// See klong.h.
    ///
    /// # Safety
    /// `name` is a NUL-terminated string; `f` keeps the contract of `Cfun`; called on the task of `klong_init`, not from a callback.
    #[no_mangle]
    pub unsafe extern "C" fn klong_cfun_register(name: *const c_char, arity: u32, f: Option<Cfun>, ctx: *mut c_void, out_cap: usize) -> i32 {
        let Some(f) = f else { return super::KLONG_ERR_ARG };
        if name.is_null() || arity as usize > MAX_ARGS && arity != WILDCARD as u32 {
            return super::KLONG_ERR_ARG;
        }
        // SAFETY: NUL-terminated by the contract.
        let name = match unsafe { core::ffi::CStr::from_ptr(name) }.to_str() {
            Ok(n) => n,
            Err(_) => return super::KLONG_ERR_ARG,
        };
        let state = super::STATE.0.get();
        // SAFETY: one task; the console is not borrowed outside `klong_step`, which this is not called from (see above).
        let Some(s) = (unsafe { &mut *state }).as_mut() else { return super::KLONG_ERR_STATE };
        if s.stepping {
            return super::KLONG_ERR_STATE;
        }
        let ctx = ctx as usize;
        s.console.rt.interp.register(name, arity as u8, host_fn(f, ctx, out_cap));
        super::KLONG_OK
    }

    /// See klong.h.
    ///
    /// # Safety
    /// `symbol` is a NUL-terminated string; `f` keeps the contract of `Cfun`; called on the task of `klong_init`, not from a callback.
    #[no_mangle]
    pub unsafe extern "C" fn klong_extern_register(symbol: *const c_char, f: Option<Cfun>, ctx: *mut c_void, out_cap: usize) -> i32 {
        let Some(f) = f else { return super::KLONG_ERR_ARG };
        if symbol.is_null() {
            return super::KLONG_ERR_ARG;
        }
        // SAFETY: NUL-terminated by the contract.
        let symbol = match unsafe { core::ffi::CStr::from_ptr(symbol) }.to_str() {
            Ok(n) => n,
            Err(_) => return super::KLONG_ERR_ARG,
        };
        let state = super::STATE.0.get();
        // SAFETY: one task; the console is not borrowed outside `klong_step`, which this is not called from (see above).
        let Some(s) = (unsafe { &mut *state }).as_mut() else { return super::KLONG_ERR_STATE };
        if s.stepping {
            return super::KLONG_ERR_STATE;
        }
        match s.console.rt.declarations.provide(symbol, host_fn(f, ctx as usize, out_cap)) {
            Ok(()) => super::KLONG_OK,
            Err(_) => super::KLONG_ERR_ARG,
        }
    }

    /// Run `f` on the interpreter that called the C function; `None` outside a call.
    fn with_interp<R>(f: impl FnOnce(&mut Interp) -> R) -> Option<R> {
        // SAFETY: one task; the pointer is the interpreter of the closure above, which is still on the stack below us.
        let p = unsafe { *CUR.0.get() };
        if p.is_null() {
            None
        } else {
            Some(f(unsafe { &mut *p }))
        }
    }

    /// Put an evaluation's outcome in `out`: the value (kept until the next call), or an error code and its message.
    ///
    /// # Safety
    /// `out` is valid for a write.
    unsafe fn finish(r: Result<Option<Value>, Error>, out: *mut Val) -> i32 {
        match r {
            Ok(v) => {
                let v = v.unwrap_or(Value::Undefined);
                // SAFETY: one task; the old value is let go here, after the C function has stopped using it (its contract).
                let slot = unsafe { &mut *KEEP.0.get() };
                *slot = Some(v);
                if !out.is_null() {
                    if let Some(v) = slot.as_ref() {
                        // SAFETY: `out` is valid for a write.
                        unsafe { out.write(describe(v)) }
                    }
                }
                0
            }
            Err(e) => {
                // (a failure to copy the message is a failure of the evaluation all the same: the code below is the kind)
                let mut m = String::new();
                if m.try_reserve(e.message.len() + 1).is_ok() {
                    m.push_str(&e.message);
                }
                m.push('\0');
                // SAFETY: one task; as above.
                let slot = unsafe { &mut *MSG.0.get() };
                let ptr = slot.insert(m).as_ptr();
                if !out.is_null() {
                    // SAFETY: `out` is valid for a write.
                    unsafe { out.write(Val { data: ptr as *const c_void, ..Val::NONE }) }
                }
                match e.kind {
                    klong_core::error::Kind::Type => 1,
                    klong_core::error::Kind::Range => 2,
                    klong_core::error::Kind::Memory => 4,
                    klong_core::error::Kind::Interrupt => 5,
                    _ => 3,
                }
            }
        }
    }

    /// See klong.h.
    ///
    /// # Safety
    /// `text` is valid for `len` bytes; `out` is null or valid for a write.
    #[no_mangle]
    pub unsafe extern "C" fn klong_eval(text: *const c_char, len: usize, out: *mut Val) -> i32 {
        // SAFETY: `len` readable bytes.
        let bytes = unsafe { core::slice::from_raw_parts(text as *const u8, len) };
        let Ok(src) = core::str::from_utf8(bytes) else { return 3 };
        match with_interp(|ip| ip.run_global(src, "<C>")) {
            // SAFETY: `out` as above.
            Some(r) => unsafe { finish(r, out) },
            None => 3,
        }
    }

    /// See klong.h.
    ///
    /// # Safety
    /// `text` is valid for `len` bytes for ever; `label` is NUL-terminated; `out` is null or valid for a write.
    #[no_mangle]
    pub unsafe extern "C" fn klong_load_static(text: *const c_char, len: usize, label: *const c_char, out: *mut Val) -> i32 {
        // SAFETY: valid for ever by the contract, so the `'static` is true.
        let bytes: &'static [u8] = unsafe { core::slice::from_raw_parts(text as *const u8, len) };
        let Ok(src) = core::str::from_utf8(bytes) else { return 3 };
        // SAFETY: NUL-terminated by the contract.
        let label = unsafe { core::ffi::CStr::from_ptr(label) }.to_str().unwrap_or("<C>");
        match with_interp(|ip| ip.load_static(src, label)) {
            // SAFETY: `out` as above.
            Some(r) => unsafe { finish(r, out) },
            None => 3,
        }
    }
}

// ----- the lane of a packet switch -----------------------------------------------------------

#[cfg(feature = "lane")]
mod lane {
    use super::*;
    use alloc::boxed::Box;
    use klong_rt::node::{DgramConfig, LaneConfig, LaneIo, NodeConfig, NodeNet};

    pub const KLONG_LANE_NO_DUTIES: u32 = 1;
    pub const KLONG_LANE_NO_CONSOLE: u32 = 2;
    pub const KLONG_LANE_SELFTEST: u32 = 4;

    /// `struct klong_lane_config` of klong.h.
    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct KlongLaneConfig {
        pub abi_version: u32,
        pub size: u32,
        pub ctx: Ctx,
        pub lane_read: Option<unsafe extern "C" fn(Ctx, *mut u8, usize) -> usize>,
        pub lane_write: Option<unsafe extern "C" fn(Ctx, *const u8, usize) -> usize>,
        /// This node's number, or -1 to learn it from the first frame the switch sends.
        pub node: i32,
        /// Random bits for message ids and connection epochs.
        pub seed: u32,
        pub flags: u32,
        /// The lane's baud rate (0: 115200); picks the packet size and the timeouts.
        pub baud: u32,
        /// Four u32 counters the C driver keeps: receive ring overruns, FIFO overruns, line
        /// errors, times the transmit ring was full. Null if it has none.
        pub counters: *const u32,
        /// Null, or a short text (NUL-terminated) on why the board restarted: the self-test's reports carry it.
        pub note: *const core::ffi::c_char,
    }

    /// `struct klong_lane_stats` of klong.h.
    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    pub struct KlongLaneStats {
        pub frames_rx: u32,
        pub frames_tx: u32,
        pub frames_ignored: u32,
        pub pings: u32,
        pub bad_frames: u32,
        pub skipped_bytes: u32,
        pub datagrams_delivered: u32,
        pub datagrams_duplicate: u32,
        pub datagrams_broken: u32,
        pub datagrams_refused: u32,
        pub datagrams_resent: u32,
        pub ipc_in: u32,
        pub ipc_out: u32,
        pub inbox_overflow: u32,
        pub console_dropped: u32,
        pub console_peer: i32,
        /// Console output with no peer to go to (bytes), input dropped for lack of room (datagrams), and
        /// channel-1 datagrams without the ACK flag, which are another console's output (datagrams).
        pub console_no_peer: u32,
        pub console_in_dropped: u32,
        pub console_unsolicited: u32,
        pub uart_rx_ring_overruns: u32,
        pub uart_rx_fifo_overruns: u32,
        pub uart_line_errors: u32,
        pub uart_tx_ring_full: u32,
        /// The self-test: 0 not running or undecided, 1 passed, 2 failed.
        pub selftest: i32,
    }

    /// The restart note of the config as text (at most 120 bytes, printable ASCII).
    fn note_text(p: *const core::ffi::c_char) -> alloc::string::String {
        let mut s = alloc::string::String::new();
        if p.is_null() {
            return s;
        }
        for i in 0..120 {
            // SAFETY: NUL-terminated by the contract of klong_lane_config.note; read up to the terminator.
            let c = unsafe { *p.add(i) } as u8;
            if c == 0 {
                break;
            }
            s.push(if (0x20..0x7f).contains(&c) { c as char } else { '?' });
        }
        s
    }

    struct CLane(KlongLaneConfig);

    impl LaneIo for CLane {
        fn counters(&self) -> klong_rt::node::UartCounters {
            if self.0.counters.is_null() {
                return Default::default();
            }
            // SAFETY: klong.h: `counters` points to four u32 that stay valid for the life of the program.
            let c = unsafe { core::slice::from_raw_parts(self.0.counters, 4) };
            klong_rt::node::UartCounters { rx_ring_overruns: c[0], rx_fifo_overruns: c[1], line_errors: c[2], tx_ring_full: c[3] }
        }
        fn read(&mut self, buf: &mut [u8]) -> usize {
            match self.0.lane_read {
                // SAFETY: klong.h: writes at most `cap` bytes into `buf` and returns the count.
                Some(f) => unsafe { f(self.0.ctx, buf.as_mut_ptr(), buf.len()) }.min(buf.len()),
                None => 0,
            }
        }
        fn write(&mut self, buf: &[u8]) -> usize {
            match self.0.lane_write {
                // SAFETY: klong.h: reads `len` bytes, keeps no pointer, returns how many it took.
                Some(f) => unsafe { f(self.0.ctx, buf.as_ptr(), buf.len()) }.min(buf.len()),
                None => 0,
            }
        }
    }

    struct Lane {
        net: NodeNet,
        console: bool,
        selftest: Option<klong_rt::selftest::SelfTest>,
    }
    static LANE: Global<Option<Lane>> = Global(UnsafeCell::new(None));

    /// One turn of the boot self-test, if it was asked for; from `klong_step`.
    pub fn selftest_step(rt: &mut klong_rt::Runtime<super::Board>) -> bool {
        // SAFETY: one task (klong.h); the only mutable access to the lane's self-test is here.
        let Some(l) = (unsafe { (*LANE.0.get()).as_mut() }) else { return false };
        match l.selftest.as_mut() {
            Some(st) => st.step(rt, &l.net),
            None => false,
        }
    }

    fn get() -> Option<&'static Lane> {
        // SAFETY: one task (klong.h); written once by `klong_lane_init`, only read afterwards.
        unsafe { (*LANE.0.get()).as_ref() }
    }
    pub fn pump() {
        if let Some(l) = get() {
            l.net.pump();
        }
    }
    pub fn quiet() -> bool {
        get().is_some_and(|l| l.console && l.net.console_quiet())
    }
    pub fn local_input() {
        if let Some(l) = get() {
            if l.console {
                l.net.console_local_input();
            }
        }
    }
    /// Before and after one turn of the main loop (`klong_step`): the console is quiet while the
    /// lane's input is being evaluated, and tells the lane when it has finished with a datagram.
    pub fn turn_begin(c: &mut klong_console::Console<super::Board>) {
        if let Some(l) = get() {
            if l.console {
                c.lane_begin(&l.net);
            }
        }
    }
    pub fn turn_end(c: &mut klong_console::Console<super::Board>) {
        if let Some(l) = get() {
            if l.console {
                c.lane_end(&l.net);
            }
        }
    }
    pub fn console_read(buf: &mut [u8]) -> usize {
        match get() {
            Some(l) if l.console => l.net.console_read(buf),
            _ => 0,
        }
    }
    pub fn console_write(bytes: &[u8]) {
        if let Some(l) = get() {
            if l.console {
                l.net.console_write(bytes);
            }
        }
    }

    /// See klong.h.
    ///
    /// # Safety
    /// `config` is null or points to a `klong_lane_config` whose callbacks keep the contracts of klong.h.
    #[no_mangle]
    pub unsafe extern "C" fn klong_lane_init(config: *const KlongLaneConfig) -> i32 {
        if config.is_null() {
            return KLONG_ERR_ARG;
        }
        // SAFETY: the first two fields are read before the rest (as in `klong_init`).
        let (version, size) = unsafe { ((*config).abi_version, (*config).size) };
        if version != KLONG_ABI_VERSION || size as usize != core::mem::size_of::<KlongLaneConfig>() {
            return KLONG_ERR_ARG;
        }
        // SAFETY: of this version and size: a whole `klong_lane_config`.
        let cfg = unsafe { *config };
        if cfg.lane_read.is_none() || cfg.lane_write.is_none() || cfg.node < -1 || cfg.node > 16 {
            return KLONG_ERR_ARG;
        }
        if get().is_some() {
            return KLONG_ERR_STATE;
        }
        // SAFETY: one task; `STATE` is only read here and no reference outlives the statement.
        let Some(state) = (unsafe { &mut *STATE.0.get() }).as_mut() else { return KLONG_ERR_STATE };
        let base = NodeConfig::for_baud(if cfg.baud == 0 { 115_200 } else { cfg.baud });
        let node = NodeConfig {
            lane: LaneConfig {
                dgram: DgramConfig { first_id: cfg.seed as u16, ..DgramConfig::default() },
                supervisor_duties: cfg.flags & KLONG_LANE_NO_DUTIES == 0,
                node: if cfg.node >= 0 { Some(cfg.node as u8) } else { None },
            },
            epoch_seed: (cfg.seed >> 16) as u16 ^ cfg.seed as u16,
            target: state.console.rt.shared.platform.try_borrow().map(|p| alloc::string::String::from(p.system.name())).unwrap_or_default(),
            ..base
        };
        // A Cortex-M0+ spends about 90 us on a call of a Klong function: with the default slow tick (every 4096 calls)
        // the lane would not be looked at for 0.4 s during an evaluation. Every 128 calls is about 12 ms. And the
        // evaluator keeps an eighth of the stack, not a quarter, for what runs at the bottom of the recursion
        // (this board's stack is small and recursion depth is what it is short of).
        {
            let ip = &mut state.console.rt.interp.cfg;
            ip.tick_mask = 127;
            if let (Some(h), Some(_)) = (hooks(), ip.stack_limit) {
                let bytes = h.stack_bytes;
                // what the board says it keeps (klong_config.stack_keep), else an eighth
                let keep = if h.stack_keep != 0 { h.stack_keep.max(2048).min(bytes / 2) } else { (bytes / 8).max(4096).min(bytes / 2) };
                ip.stack_limit = Some(bytes - keep);
                ip.max_nest = ((keep / 2) / 384).clamp(8, ip.max_nest.max(8) as usize) as u32;
            }
        }
        let net = NodeNet::new(Box::new(CLane(cfg)), Box::new(|| CClock::now_ms().saturating_mul(1000)), node);
        match state.console.rt.shared.platform.try_borrow_mut() {
            Ok(mut p) => p.transports.push(net.transport()),
            Err(_) => return KLONG_ERR_STATE,
        }
        // SAFETY: as above; `LANE` is empty until this write.
        unsafe { *LANE.0.get() = Some(Lane { net, console: cfg.flags & KLONG_LANE_NO_CONSOLE == 0, selftest: if cfg.flags & KLONG_LANE_SELFTEST != 0 { { let mut t = klong_rt::selftest::SelfTest::new(); t.set_note(note_text(cfg.note)); Some(t) } } else { None } }) };
        KLONG_OK
    }

    /// See klong.h.
    #[no_mangle]
    pub extern "C" fn klong_lane_pump() {
        pump();
    }

    /// See klong.h.
    #[no_mangle]
    pub extern "C" fn klong_lane_node() -> i32 {
        get().and_then(|l| l.net.node()).map_or(-1, |n| n as i32)
    }

    /// See klong.h.
    ///
    /// # Safety
    /// `out` is null or valid for a write of a `klong_lane_stats`.
    /// (The C name is `klong_lane_stats_get`: `klong_lane_stats` is the struct.)
    #[no_mangle]
    pub unsafe extern "C" fn klong_lane_stats_get(out: *mut KlongLaneStats) {
        if out.is_null() {
            return;
        }
        let mut s = KlongLaneStats { console_peer: -1, ..KlongLaneStats::default() };
        if let Some(l) = get() {
            let (ls, ds, c) = (l.net.lane_stats(), l.net.datagram_stats(), l.net.counters());
            s.frames_rx = ls.frames_rx;
            s.frames_tx = ls.frames_tx;
            s.frames_ignored = ls.ignored;
            s.pings = ls.pings;
            s.bad_frames = l.net.parser_bad();
            s.skipped_bytes = l.net.parser_skipped();
            s.datagrams_delivered = ds.delivered;
            s.datagrams_duplicate = ds.duplicates;
            s.datagrams_broken = ds.broken;
            s.datagrams_refused = ds.refused;
            s.datagrams_resent = ds.resent;
            s.ipc_in = c.messages_in;
            s.ipc_out = c.messages_out;
            s.inbox_overflow = c.inbox_overflow;
            s.console_dropped = c.console_dropped;
            s.console_no_peer = c.console_no_peer;
            s.console_in_dropped = c.console_in_dropped;
            s.console_unsolicited = c.console_unsolicited;
            s.console_peer = l.net.console_peer().map_or(-1, |p| p as i32);
            let u = l.net.uart_counters();
            s.uart_rx_ring_overruns = u.rx_ring_overruns;
            s.uart_rx_fifo_overruns = u.rx_fifo_overruns;
            s.uart_line_errors = u.line_errors;
            s.uart_tx_ring_full = u.tx_ring_full;
            s.selftest = match l.selftest.as_ref().map(|t| t.verdict()) {
                Some(klong_rt::selftest::Verdict::Pass) => 1,
                Some(klong_rt::selftest::Verdict::Fail) => 2,
                _ => 0,
            };
        }
        // SAFETY: non-null, valid for a write by the contract above.
        unsafe { out.write(s) }
    }
}

// ----- panics on a bare-metal target ---------------------------------------------------------

#[cfg(target_os = "none")]
mod bare {
    use core::fmt::Write;

    struct Text {
        buf: [u8; 160],
        len: usize,
    }
    impl Write for Text {
        fn write_str(&mut self, s: &str) -> core::fmt::Result {
            let n = s.len().min(self.buf.len() - self.len);
            self.buf[self.len..self.len + n].copy_from_slice(&s.as_bytes()[..n]);
            self.len += n;
            Ok(())
        }
    }

    /// A panic is a bug in the runtime (errors of Klong programs, running out of heap and
    /// deep recursion are ordinary errors at the prompt). Tell the C side, which resets.
    #[panic_handler]
    fn panic(info: &core::panic::PanicInfo) -> ! {
        let mut text = Text { buf: [0; 160], len: 0 };
        let _ = write!(text, "{}", info);
        if let Some(super::KlongConfig { fatal: Some(f), ctx, .. }) = super::hooks() {
            // SAFETY: a callback of klong.h; `msg` is `len` initialised bytes.
            unsafe { f(ctx, text.buf.as_ptr() as *const core::ffi::c_char, text.len) }
        }
        loop {
            core::hint::spin_loop();
        }
    }
}

/// `.heap(mode)`: the board's heap as integers, for a diagnostic (feature `heapdiag`). The numbers come from a function the board
/// gives with `klong_set_diag` (boards/pico-bench/main.c); this only carries them into a vector of the interpreter.
#[cfg(feature = "heapdiag")]
mod heapdiag {
    use alloc::rc::Rc;
    use alloc::vec::Vec;
    use klong_core::error::Error;
    use klong_core::interp::Interp;
    use klong_core::value::{Int, Value};

    /// `size_t diag(uint32_t mode, int64_t *out, size_t cap)`: fill `out` (room for `cap` integers) for `mode`, return how many.
    type Diag = unsafe extern "C" fn(u32, *mut i64, usize) -> usize;
    static DIAG: super::Global<Option<Diag>> = super::Global(core::cell::UnsafeCell::new(None));

    /// See klong.h.
    ///
    /// # Safety
    /// `f` is null or a function that keeps the contract above; called from the task that calls `klong_init`.
    #[no_mangle]
    pub unsafe extern "C" fn klong_set_diag(f: Option<Diag>) {
        // SAFETY: one task (klong.h): the cell is written here and read in `.heap`, never at the same time.
        unsafe { *DIAG.0.get() = f }
        super::HEAP.set_peak_hook(f.map(|_| peak_noted as fn(usize)));
    }

    /// The high-water mark was raised to `used`: the board keeps what its heap holds (mode 1000, with `used` in place of the capacity).
    fn peak_noted(used: usize) {
        // SAFETY: as in `klong_set_diag`; the board's function writes nothing when it is given no output.
        if let Some(diag) = unsafe { *DIAG.0.get() } {
            unsafe { diag(1000, core::ptr::null_mut(), used) };
        }
    }

    pub fn register(interp: &mut Interp) {
        interp.register(
            ".heap",
            1,
            Rc::new(|_: &mut Interp, a: &[Value]| {
                let mode = match a.first() {
                    Some(Value::Int(m)) if *m >= 0 => *m as u32,
                    _ => return Err(Error::typ(".heap takes a mode, a non-negative integer")),
                };
                if mode == 300 {
                    // the high-water mark starts again from what is in use (and the board forgets the heap it kept at the old one); the old one is the answer
                    let old = super::HEAP.peak();
                    super::HEAP.reset_peak();
                    // SAFETY: as below.
                    if let Some(diag) = unsafe { *DIAG.0.get() } {
                        unsafe { diag(1001, core::ptr::null_mut(), 0) };
                    }
                    return Ok(Value::ints(alloc::vec![old as Int]));
                }
                // SAFETY: as in `klong_set_diag`: a read of a `Copy` value on the one task.
                let Some(diag) = (unsafe { *DIAG.0.get() }) else { return Err(Error::typ(".heap: the board gave no diagnostic")) };
                let mut out = [0i64; 96];
                // SAFETY: `out` has room for the 96 integers passed as `cap`; the board writes at most that many and returns the count.
                let n = unsafe { diag(mode, out.as_mut_ptr(), out.len()) }.min(out.len());
                let mut v: Vec<Int> = klong_core::value::try_vec(n)?;
                v.extend(out[..n].iter().map(|&x| x as Int));
                Ok(Value::ints(v))
            }),
        );
    }
}
