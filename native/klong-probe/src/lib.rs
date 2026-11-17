//! Size probe (see Cargo.toml): everything a board build links (lexer, parser, evaluator,
//! operators, printing, the runtime's system functions, the file system in memory and the
//! Klong library when those features are on) behind one function, with a bump allocator and a
//! console that goes to an imported function (wasm32) or to a memory-mapped register (the
//! bare-metal targets, where `src/main.rs` supplies `_start`).
#![no_std]

extern crate alloc;

use alloc::string::String;
use core::alloc::{GlobalAlloc, Layout};
use core::cell::UnsafeCell;

use klong_core::Config;
use klong_rt::{Platform, Runtime};

/// Bare metal: the heap is this much of `.bss` (subtract it when reading the section sizes).
pub const HEAP_BYTES: usize = 64 * 1024;

struct Bump {
    next: UnsafeCell<usize>,
}

// SAFETY: one thread only (wasm32 without threads; a board before any interrupt is enabled).
unsafe impl Sync for Bump {}

#[cfg(target_arch = "wasm32")]
// SAFETY: memory comes from memory.grow, is aligned as asked and never reused.
unsafe impl GlobalAlloc for Bump {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: single-threaded access to the cell.
        let next = unsafe { &mut *self.next.get() };
        if *next == 0 {
            *next = core::arch::wasm32::memory_size(0) * 65536;
        }
        let start = (*next + layout.align() - 1) & !(layout.align() - 1);
        let end = match start.checked_add(layout.size()) {
            Some(e) => e,
            None => return core::ptr::null_mut(),
        };
        let have = core::arch::wasm32::memory_size(0) * 65536;
        if end > have {
            let pages = (end - have).div_ceil(65536);
            if core::arch::wasm32::memory_grow(0, pages) == usize::MAX {
                return core::ptr::null_mut();
            }
        }
        *next = end;
        start as *mut u8
    }
    unsafe fn dealloc(&self, _: *mut u8, _: Layout) {}
}

#[cfg(not(target_arch = "wasm32"))]
struct Arena(UnsafeCell<[u8; HEAP_BYTES]>);

// SAFETY: one thread only.
#[cfg(not(target_arch = "wasm32"))]
unsafe impl Sync for Arena {}

#[cfg(not(target_arch = "wasm32"))]
static ARENA: Arena = Arena(UnsafeCell::new([0; HEAP_BYTES]));

#[cfg(not(target_arch = "wasm32"))]
// SAFETY: memory comes from ARENA, is aligned as asked and never reused; a request that does
// not fit gets null (the runtime reports a memory error).
unsafe impl GlobalAlloc for Bump {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: single-threaded access to the cell.
        let next = unsafe { &mut *self.next.get() };
        let base = ARENA.0.get() as usize;
        let start = (base + *next + layout.align() - 1) & !(layout.align() - 1);
        match start.checked_add(layout.size()) {
            Some(end) if end <= base + HEAP_BYTES => {
                *next = end - base;
                start as *mut u8
            }
            _ => core::ptr::null_mut(),
        }
    }
    unsafe fn dealloc(&self, _: *mut u8, _: Layout) {}
}

#[global_allocator]
static HEAP: Bump = Bump { next: UnsafeCell::new(0) };

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    #[cfg(target_arch = "wasm32")]
    core::arch::wasm32::unreachable();
    #[cfg(not(target_arch = "wasm32"))]
    loop {}
}

#[cfg(target_arch = "wasm32")]
extern "C" {
    fn host_write(ptr: *const u8, len: usize);
}

/// Bare metal: a transmit register (the address of the UART of QEMU's `virt` machines; a
/// board port replaces this).
#[cfg(not(target_arch = "wasm32"))]
unsafe fn host_write(ptr: *const u8, len: usize) {
    for i in 0..len {
        // SAFETY: the caller's bytes are readable; the register address is the board's.
        unsafe { core::ptr::write_volatile(0x1000_0000 as *mut u8, *ptr.add(i)) }
    }
}

#[derive(Default)]
struct Board {
    #[cfg(feature = "files")]
    fs: klong_rt::memfs::MemFs,
}

impl Platform for Board {
    fn now_us(&self) -> u64 {
        0
    }
    fn wall_clock(&self) -> Option<f64> {
        None
    }
    fn idle(&mut self) {}
    fn console_read(&mut self, _: &mut [u8]) -> usize {
        0
    }
    fn console_write(&mut self, bytes: &[u8]) {
        // SAFETY: the host reads `len` bytes at `ptr` during the call.
        unsafe { host_write(bytes.as_ptr(), bytes.len()) }
    }
    #[cfg(feature = "files")]
    fn fs(&mut self) -> Option<&mut dyn klong_rt::FileSystem> {
        Some(&mut self.fs)
    }
}

/// The native stack the evaluator may use (`Config::for_stack`); 0 for no limit.
static STACK: core::sync::atomic::AtomicUsize = core::sync::atomic::AtomicUsize::new(48 * 1024);

#[no_mangle]
pub extern "C" fn klong_set_stack(bytes: usize) {
    STACK.store(bytes, core::sync::atomic::Ordering::Relaxed);
}

/// Run the program at `ptr`/`len`, print its value or its error; 0 on success.
///
/// # Safety
/// `ptr` and `len` must describe readable memory.
#[no_mangle]
pub unsafe extern "C" fn klong_run(ptr: *const u8, len: usize) -> i32 {
    // SAFETY: the caller's contract.
    let bytes = unsafe { core::slice::from_raw_parts(ptr, len) };
    let text = match core::str::from_utf8(bytes) {
        Ok(t) => t,
        Err(_) => return 2,
    };
    let mut rt = Runtime::new(
        Board::default(),
        match STACK.load(core::sync::atomic::Ordering::Relaxed) {
            0 => Config::default(),
            n => Config::for_stack(n),
        },
        None,
    );
    let (out, code): (String, i32) = match rt.run(text, "<probe>") {
        Ok(Some(v)) => (rt.interp.format(&v), 0),
        Ok(None) => (String::new(), 0),
        Err(e) => (e.render(), 1),
    };
    rt.shared.platform.borrow_mut().console_write(out.as_bytes());
    code
}

/// Memory for the host to put a program in.
#[no_mangle]
pub extern "C" fn klong_alloc(len: usize) -> *mut u8 {
    let mut v = alloc::vec::Vec::<u8>::with_capacity(len);
    let p = v.as_mut_ptr();
    core::mem::forget(v);
    p
}
