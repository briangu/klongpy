//! A counting allocator that enforces a heap limit (docs/native/index.md). This module
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
    overhead: AtomicUsize,
    round: AtomicUsize,
    every: AtomicUsize,
    count: AtomicUsize,
    failed: AtomicUsize,
    /// `Grow` as a number (0: none), see `set_grow`.
    grow: AtomicUsize,
    /// A function called with `used` each time the peak is passed by 1 KB more than the last time it was called (see `set_peak_hook`), and that mark.
    peak_hook: AtomicUsize,
    snap_mark: AtomicUsize,
}

/// Resize a block in place if the allocator beneath can: `(ptr, old size, new size, align)`, true when the block is now of the new
/// size where it was (for a larger size that needs the room after it to be free; a smaller size gives the end back).
pub type Grow = unsafe fn(*mut u8, usize, usize, usize) -> bool;

impl<A> Counting<A> {
    pub const fn new(inner: A) -> Counting<A> {
        Counting {
            inner,
            used: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
            limit: AtomicUsize::new(0),
            reserve: AtomicUsize::new(0),
            overhead: AtomicUsize::new(0),
            round: AtomicUsize::new(0),
            every: AtomicUsize::new(0),
            count: AtomicUsize::new(0),
            failed: AtomicUsize::new(0),
            grow: AtomicUsize::new(0),
            peak_hook: AtomicUsize::new(0),
            snap_mark: AtomicUsize::new(0),
        }
    }
    /// Call `f(used)` when the high-water mark is raised, at most once for each KB it rises: the board's diagnostic keeps what the heap
    /// holds at the moment of the peak (a peak that is over is otherwise only a number).
    pub fn set_peak_hook(&self, f: Option<fn(usize)>) {
        self.peak_hook.store(f.map_or(0, |f| f as usize), Relaxed);
    }
    /// Let `realloc` ask the allocator beneath to resize a block where it is. A vector that doubles then needs the room of
    /// its growth only, not the old and the new block at once (which is what the default `realloc` needs, and what the limit
    /// is kept for when this says no).
    pub fn set_grow(&self, f: Option<Grow>) {
        self.grow.store(f.map_or(0, |f| f as usize), Relaxed);
    }
    fn grow_fn(&self) -> Option<Grow> {
        match self.grow.load(Relaxed) {
            0 => None,
            // SAFETY: only `set_grow` stores here, and it stores a `Grow`.
            n => Some(unsafe { core::mem::transmute::<usize, Grow>(n) }),
        }
    }
    /// Set the hard limit in bytes (0 = none) and the reserve held back below it.
    pub fn set_limit(&self, limit: usize, reserve: usize) {
        self.limit.store(limit, Relaxed);
        self.reserve.store(reserve.min(limit / 2), Relaxed);
    }
    /// Bytes the allocator beneath keeps for each block (malloc's header), and (`set_round`) the grain it gives blocks out in:
    /// every block is counted as its size rounded up to the grain plus this, so that what is counted is what the arena holds
    /// and the limit can be the arena's size. Both 0 (the default): the size asked for, as before.
    pub fn set_overhead(&self, per_block: usize) {
        self.overhead.store(per_block, Relaxed);
    }
    /// Blocks are counted rounded up to a multiple of `grain` (0: as asked): what the allocator beneath gives out in.
    pub fn set_round(&self, grain: usize) {
        self.round.store(grain, Relaxed);
    }
    /// What a block of `size` bytes takes.
    fn cost(&self, size: usize) -> usize {
        let (o, g) = (self.overhead.load(Relaxed), self.round.load(Relaxed));
        if o == 0 && g == 0 {
            size
        } else {
            let g = g.max(1);
            (size.max(1).saturating_add(g - 1) / g * g).saturating_add(o)
        }
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
        self.snap_mark.store(self.used.load(Relaxed), Relaxed);
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
        self.admit_cost(self.cost(size), size)
    }

    /// `add` bytes more in use, for a request of `size` (large requests keep the reserve).
    fn admit_cost(&self, add: usize, size: usize) -> bool {
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
        let after = self.used.load(Relaxed).saturating_add(add);
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
            let h = self.peak_hook.load(Relaxed);
            if h != 0 && u >= self.snap_mark.load(Relaxed).saturating_add(1024) {
                self.snap_mark.store(u, Relaxed);
                // SAFETY: only `set_peak_hook` stores here, and it stores an `fn(usize)`.
                let f: fn(usize) = unsafe { core::mem::transmute::<usize, fn(usize)>(h) };
                f(u);
            }
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
            self.account(self.cost(layout.size()));
        }
        p
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        self.release(self.cost(layout.size()));
        // SAFETY: `ptr` came from `self.inner` with this layout.
        unsafe { self.inner.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let old = layout.size();
        if let Some(grow) = self.grow_fn() {
            let (oc, nc) = (self.cost(old), self.cost(new_size));
            if new_size > old {
                // in place the block takes only what it grows by
                if !self.admit_cost(nc.saturating_sub(oc), new_size - old) {
                    self.failed.store(self.failed.load(Relaxed) + 1, Relaxed);
                    return core::ptr::null_mut();
                }
            }
            // SAFETY: `ptr` came from the allocator beneath with `layout`; `grow` keeps its contract.
            if new_size != old && unsafe { grow(ptr, old, new_size, layout.align()) } {
                self.release(oc);
                self.account(nc);
                return ptr;
            }
        }
        // growing is an allocation, a copy and a free (the default `realloc` of the allocator beneath): both blocks are there
        // at once, so the whole of the new one is asked for while the old one is still counted
        if new_size > old && !self.admit_cost(self.cost(new_size), new_size - old) {
            self.failed.store(self.failed.load(Relaxed) + 1, Relaxed);
            return core::ptr::null_mut();
        }
        // SAFETY: the caller's contract for `realloc` is passed on unchanged.
        let p = unsafe { self.inner.realloc(ptr, layout, new_size) };
        if !p.is_null() {
            self.release(self.cost(old));
            self.account(self.cost(new_size));
        }
        p
    }
}
