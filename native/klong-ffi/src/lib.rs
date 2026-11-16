//! klong-ffi: the board runtime behind a C ABI (`include/klong.h`), as a static library for
//! an application built with a vendor SDK (ticket 104, step 2). The C side supplies the
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

fn mem() -> (usize, usize, usize) {
    (HEAP.used(), HEAP.peak(), HEAP.limit())
}
fn pressed() -> bool {
    HEAP.pressed()
}

// ----- the board, made of callbacks ----------------------------------------------------------

struct CUart;
impl Uart for CUart {
    fn read(&mut self, buf: &mut [u8]) -> usize {
        match hooks() {
            // SAFETY: klong.h: `read` writes at most `cap` bytes into `buf` and returns the
            // count; the count is clamped in case it does not keep the second half.
            Some(KlongConfig { read: Some(f), ctx, .. }) => unsafe { f(ctx, buf.as_mut_ptr(), buf.len()) }.min(buf.len()),
            _ => 0,
        }
    }
    fn write(&mut self, bytes: &[u8]) {
        if let Some(KlongConfig { write: Some(f), ctx, .. }) = hooks() {
            // SAFETY: klong.h: `write` reads `len` bytes and keeps no pointer.
            unsafe { f(ctx, bytes.as_ptr(), bytes.len()) }
        }
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
    HEAP.set_limit(cfg.heap_limit, cfg.heap_reserve);

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
        mem: Some(mem),
        mem_pressed: Some(pressed),
        console: ConsoleConfig { dumb: cfg.flags & KLONG_FLAG_DUMB != 0, ..ConsoleConfig::default() },
        ..BoardConfig::default()
    };
    let console = klong_board::start(platform, board);
    // SAFETY: as above; `STATE` is empty until this write.
    unsafe { *STATE.0.get() = Some(State { console, stepping: false }) };
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
            let busy = klong_board::step(&mut s.console);
            s.stepping = false;
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
