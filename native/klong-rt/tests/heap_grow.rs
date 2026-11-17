//! `Counting::set_grow`: a block that the allocator beneath can resize where it is is counted by what it grows by, not by the old
//! block and the new at once (what the default `realloc`, an allocation, a copy and a free, needs). The allocator here is a bump
//! arena that never frees, so only the counting is under test.
use klong_rt::heap::Counting;
use std::alloc::{GlobalAlloc, Layout};
use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

struct Bump(UnsafeCell<[u8; 1 << 16]>, AtomicUsize);
// SAFETY: the test is one thread.
unsafe impl Sync for Bump {}
unsafe impl GlobalAlloc for Bump {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let at = (self.1.load(Relaxed) + 7) & !7;
        if at + l.size() > 1 << 16 {
            return std::ptr::null_mut();
        }
        self.1.store(at + l.size(), Relaxed);
        unsafe { (self.0.get() as *mut u8).add(at) }
    }
    unsafe fn dealloc(&self, _: *mut u8, _: Layout) {}
}

static GROWS: AtomicUsize = AtomicUsize::new(0);
/// Growing works up to 12,000 bytes; shrinking always.
unsafe fn grow(_: *mut u8, _old: usize, new: usize, _align: usize) -> bool {
    GROWS.store(GROWS.load(Relaxed) + 1, Relaxed);
    new <= 12_000
}

#[test]
fn a_block_resized_in_place_costs_what_it_grows_by() {
    let l = |n| Layout::from_size_align(n, 8).unwrap();
    // without it: 6,000 and 8,000 at once is 14,000, over the 10,000 limit
    let plain = Counting::new(Bump(UnsafeCell::new([0; 1 << 16]), AtomicUsize::new(0)));
    plain.set_limit(10_000, 0);
    let p = unsafe { plain.alloc(l(6000)) };
    assert!(!p.is_null());
    assert!(unsafe { plain.realloc(p, l(6000), 8000) }.is_null());
    assert_eq!((plain.used(), plain.failed()), (6000, 1));

    // with it: 2,000 more
    let h = Counting::new(Bump(UnsafeCell::new([0; 1 << 16]), AtomicUsize::new(0)));
    h.set_limit(10_000, 0);
    h.set_grow(Some(grow));
    let p = unsafe { h.alloc(l(6000)) };
    let q = unsafe { h.realloc(p, l(6000), 8000) };
    assert_eq!(q, p, "grown where it was");
    assert_eq!((h.used(), h.peak(), h.failed()), (8000, 8000, 0));
    // the limit still holds: 8,000 to 11,000 is 3,000 more
    assert!(unsafe { h.realloc(q, l(8000), 11_000) }.is_null());
    assert_eq!((h.used(), h.failed()), (8000, 1));
    // shrinking gives the end back
    let q = unsafe { h.realloc(q, l(8000), 5000) };
    assert_eq!((q, h.used()), (p, 5000));
    // when it cannot be done in place (over 12,000 here) the old way applies, with the old way's check: 5,000 and 13,000 are 18,000
    h.set_limit(13_000, 0);
    assert!(unsafe { h.realloc(q, l(5000), 13_000) }.is_null());
    assert!(GROWS.load(Relaxed) >= 3);
}
