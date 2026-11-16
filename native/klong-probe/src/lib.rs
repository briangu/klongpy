//! Size probe (see Cargo.toml): everything a board build links (lexer, parser, evaluator,
//! operators, printing, the runtime's system functions) behind one exported function, with a
//! bump allocator and a console that goes to an imported function.
#![no_std]

extern crate alloc;

use alloc::string::String;
use core::alloc::{GlobalAlloc, Layout};
use core::cell::UnsafeCell;

use klong_core::Config;
use klong_rt::{Platform, Runtime};

struct Bump {
    next: UnsafeCell<usize>,
}

// SAFETY: wasm32-unknown-unknown without threads: one thread only.
unsafe impl Sync for Bump {}

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

#[global_allocator]
static HEAP: Bump = Bump { next: UnsafeCell::new(0) };

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    core::arch::wasm32::unreachable()
}

extern "C" {
    fn host_write(ptr: *const u8, len: usize);
}

struct Board;

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
    let mut rt = Runtime::new(Board, Config::default(), None);
    let (out, code): (String, i32) = match rt.run(text, "<probe>") {
        Ok(Some(v)) => (rt.interp.format(&v), 0),
        Ok(None) => (String::new(), 0),
        Err(e) => (e.render(), 1),
    };
    rt.platform.borrow_mut().console_write(out.as_bytes());
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
