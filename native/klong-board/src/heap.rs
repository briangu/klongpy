//! An allocator over one region of RAM: first fit, an address-ordered free list, neighbours
//! merged when freed. No header on allocated blocks (`dealloc` is told the layout), so the
//! overhead is the rounding of sizes to two machine words. It is the simplest thing that
//! frees memory; a port with megabytes of heap and a long life should measure it against a
//! TLSF allocator (`ref-platforms.md` section 5) and may use the vendor's `malloc` instead.
//! Wrap it in `klong_rt::heap::Counting` for the limit, the reserve and `.mem()`.
//!
//! Single-threaded: the runtime allocates from one task and never from an interrupt handler.
//! This module is the one place in the crate that uses `unsafe`.
#![allow(unsafe_code)]

use core::alloc::{GlobalAlloc, Layout};
use core::cell::UnsafeCell;
use core::ptr::null_mut;

#[repr(C)]
struct Free {
    size: usize,
    next: *mut Free,
}

/// Every block starts at a multiple of this and has a multiple of it as its size, so what is
/// left of a block that is split is nothing or enough for a `Free`.
const UNIT: usize = 2 * core::mem::size_of::<usize>();

pub struct RegionHeap {
    head: UnsafeCell<*mut Free>,
}

// SAFETY: used from one thread of control only (the contract of this module).
unsafe impl Sync for RegionHeap {}

fn up(n: usize, to: usize) -> Option<usize> {
    n.checked_add(to - 1).map(|v| v & !(to - 1))
}

impl RegionHeap {
    /// A heap without memory: every allocation fails until `init`.
    pub const fn empty() -> RegionHeap {
        RegionHeap { head: UnsafeCell::new(null_mut()) }
    }

    /// Give the heap its region. Returns the bytes it can use (the region cut to whole units).
    ///
    /// # Safety
    /// `start..start + len` must be memory nothing else uses for as long as the heap lives,
    /// and this must be called once, before the first allocation.
    pub unsafe fn init(&self, start: *mut u8, len: usize) -> usize {
        let base = start as usize;
        let first = match up(base, UNIT) {
            Some(f) => f,
            None => return 0,
        };
        let end = (base.saturating_add(len)) & !(UNIT - 1);
        if end <= first || end - first < UNIT {
            return 0;
        }
        let block = first as *mut Free;
        // SAFETY: the block is inside the caller's region, aligned, and large enough.
        unsafe {
            block.write(Free { size: end - first, next: null_mut() });
            *self.head.get() = block;
        }
        end - first
    }

    /// Bytes free, and the largest block (tests, diagnostics): a walk over the free list.
    pub fn free(&self) -> (usize, usize) {
        let (mut total, mut largest) = (0, 0);
        // SAFETY: the list holds valid blocks; single-threaded.
        unsafe {
            let mut b = *self.head.get();
            while !b.is_null() {
                total += (*b).size;
                largest = largest.max((*b).size);
                b = (*b).next;
            }
        }
        (total, largest)
    }

    /// Number of free blocks (1 when everything was freed and merged).
    pub fn blocks(&self) -> usize {
        let mut n = 0;
        // SAFETY: as in `free`.
        unsafe {
            let mut b = *self.head.get();
            while !b.is_null() {
                n += 1;
                b = (*b).next;
            }
        }
        n
    }
}

// SAFETY: blocks handed out are inside the region, aligned as asked, do not overlap (a block
// leaves the free list when it is handed out) and return to the list only through `dealloc`.
unsafe impl GlobalAlloc for RegionHeap {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let size = match up(layout.size().max(1), UNIT) {
            Some(s) => s,
            None => return null_mut(),
        };
        let align = layout.align().max(UNIT);
        // SAFETY: single-threaded access; every pointer followed is a block of the list.
        unsafe {
            let mut link: *mut *mut Free = self.head.get();
            while !(*link).is_null() {
                let block = *link;
                let addr = block as usize;
                let bsize = (*block).size;
                let fits = up(addr, align).and_then(|start| start.checked_add(size).map(|end| (start, end)));
                if let Some((start, end)) = fits {
                    if end <= addr + bsize {
                        let next = (*block).next;
                        let mut tail = next;
                        if end < addr + bsize {
                            let t = end as *mut Free;
                            t.write(Free { size: addr + bsize - end, next });
                            tail = t;
                        }
                        if start > addr {
                            // (a multiple of UNIT, so it holds a Free)
                            (*block).size = start - addr;
                            (*block).next = tail;
                        } else {
                            *link = tail;
                        }
                        return start as *mut u8;
                    }
                }
                link = &mut (*block).next;
            }
        }
        null_mut()
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        let size = match up(layout.size().max(1), UNIT) {
            Some(s) => s,
            None => return,
        };
        let addr = ptr as usize;
        // SAFETY: `ptr` came from `alloc` with this layout, so `addr..addr + size` is a block
        // of the region that is not in the list.
        unsafe {
            let mut prev: *mut Free = null_mut();
            let mut next = *self.head.get();
            while !next.is_null() && (next as usize) < addr {
                prev = next;
                next = (*next).next;
            }
            let block = ptr as *mut Free;
            block.write(Free { size, next });
            if !next.is_null() && addr + size == next as usize {
                (*block).size += (*next).size;
                (*block).next = (*next).next;
            }
            if prev.is_null() {
                *self.head.get() = block;
            } else if prev as usize + (*prev).size == addr {
                (*prev).size += (*block).size;
                (*prev).next = (*block).next;
            } else {
                (*prev).next = block;
            }
        }
    }
}

/// A `RegionHeap` in a static, as a global allocator can hold it:
/// `static HEAP: Counting<HeapRef> = Counting::new(HeapRef(&REGION));`
pub struct HeapRef(pub &'static RegionHeap);

// SAFETY: forwards to `RegionHeap`.
unsafe impl GlobalAlloc for HeapRef {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller's contract.
        unsafe { self.0.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: the caller's contract.
        unsafe { self.0.dealloc(ptr, layout) }
    }
}
