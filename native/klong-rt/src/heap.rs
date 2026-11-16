//! A counting allocator that enforces a heap limit (native-runtime.md section 5). This module
//! is the one place in `klong-rt` that uses `unsafe`.
//!
//! * `limit` is the hard limit. `reserve` bytes below it are held back: a request of
//!   `LARGE` bytes or more that would cross `limit - reserve` fails, which the core turns
//!   into a `memory` error (every allocation sized by data goes through `try_reserve`).
//!   Small requests (reference-count boxes, frames) may dip into the reserve, so that there is
//!   room to unwind and print.
//! * [`Counting::pressed`] says the heap is living on its reserve; the core asks at every tick
//!   (`Config::mem_probe`) and unwinds with a `memory` error.
//! * `fail_every(n)` makes every nth large request fail (failure injection for tests).
//!
//! The counters are atomics used with load and store only (no compare-and-swap), so the type
//! works on targets without atomic read-modify-write. The runtime is single-threaded; on a
//! multi-threaded host the figures are approximate.
#![allow(unsafe_code)]

use core::alloc::{GlobalAlloc, Layout};
use core::sync::atomic::{AtomicUsize, Ordering::Relaxed};

/// Requests of at least this many bytes are "sized by data".
pub const LARGE: usize = 512;

pub struct Counting<A> {
    inner: A,
    used: AtomicUsize,
    peak: AtomicUsize,
    limit: AtomicUsize,
    reserve: AtomicUsize,
    every: AtomicUsize,
    count: AtomicUsize,
    failed: AtomicUsize,
}

impl<A> Counting<A> {
    pub const fn new(inner: A) -> Counting<A> {
        Counting {
            inner,
            used: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
            limit: AtomicUsize::new(0),
            reserve: AtomicUsize::new(0),
            every: AtomicUsize::new(0),
            count: AtomicUsize::new(0),
            failed: AtomicUsize::new(0),
        }
    }
    /// Set the hard limit in bytes (0 = none) and the reserve held back below it.
    pub fn set_limit(&self, limit: usize, reserve: usize) {
        self.limit.store(limit, Relaxed);
        self.reserve.store(reserve.min(limit / 2), Relaxed);
    }
    pub fn used(&self) -> usize {
        self.used.load(Relaxed)
    }
    pub fn peak(&self) -> usize {
        self.peak.load(Relaxed)
    }
    pub fn limit(&self) -> usize {
        self.limit.load(Relaxed)
    }
    pub fn reset_peak(&self) {
        self.peak.store(self.used.load(Relaxed), Relaxed);
    }
    /// Number of requests refused so far.
    pub fn failed(&self) -> usize {
        self.failed.load(Relaxed)
    }
    /// True when more than `limit - reserve` bytes are in use.
    pub fn pressed(&self) -> bool {
        let limit = self.limit.load(Relaxed);
        limit != 0 && self.used.load(Relaxed) > limit - self.reserve.load(Relaxed)
    }
    /// Make every nth large request fail (0 = off).
    pub fn fail_every(&self, n: usize) {
        self.every.store(n, Relaxed);
        self.count.store(0, Relaxed);
    }

    fn admit(&self, size: usize) -> bool {
        if size >= LARGE {
            let every = self.every.load(Relaxed);
            if every != 0 {
                let c = self.count.load(Relaxed) + 1;
                self.count.store(c, Relaxed);
                if c % every == 0 {
                    return false;
                }
            }
        }
        let limit = self.limit.load(Relaxed);
        if limit == 0 {
            return true;
        }
        let after = self.used.load(Relaxed).saturating_add(size);
        if after > limit {
            return false;
        }
        !(size >= LARGE && after > limit - self.reserve.load(Relaxed))
    }

    fn account(&self, size: usize) {
        let u = self.used.load(Relaxed) + size;
        self.used.store(u, Relaxed);
        if u > self.peak.load(Relaxed) {
            self.peak.store(u, Relaxed);
        }
    }

    fn release(&self, size: usize) {
        self.used.store(self.used.load(Relaxed).saturating_sub(size), Relaxed);
    }
}

// SAFETY: every request is forwarded unchanged to the inner allocator, or refused by
// returning null, which GlobalAlloc permits. The bookkeeping never touches the memory.
unsafe impl<A: GlobalAlloc> GlobalAlloc for Counting<A> {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if !self.admit(layout.size()) {
            self.failed.store(self.failed.load(Relaxed) + 1, Relaxed);
            return core::ptr::null_mut();
        }
        // SAFETY: the caller's contract for `alloc` is passed on unchanged.
        let p = unsafe { self.inner.alloc(layout) };
        if !p.is_null() {
            self.account(layout.size());
        }
        p
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        self.release(layout.size());
        // SAFETY: `ptr` came from `self.inner` with this layout.
        unsafe { self.inner.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let old = layout.size();
        if new_size > old && !self.admit(new_size - old) {
            self.failed.store(self.failed.load(Relaxed) + 1, Relaxed);
            return core::ptr::null_mut();
        }
        // SAFETY: the caller's contract for `realloc` is passed on unchanged.
        let p = unsafe { self.inner.realloc(ptr, layout, new_size) };
        if !p.is_null() {
            self.release(old);
            self.account(new_size);
        }
        p
    }
}
