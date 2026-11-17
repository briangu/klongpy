//! How many allocations a call of a Klong function costs (a counting allocator; in a file of its own).
use klong_core::{Config, Interp, NoHooks};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

struct Counting;
static ALLOCS: AtomicU64 = AtomicU64::new(0);
static BYTES: AtomicU64 = AtomicU64::new(0);
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Relaxed);
        BYTES.fetch_add(l.size() as u64, Relaxed);
        System.alloc(l)
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        System.dealloc(p, l)
    }
}
#[global_allocator]
static A: Counting = Counting;

#[test]
fn allocations_per_call() {
    let mut ip = Interp::new(Config::default(), Box::new(NoHooks));
    ip.run("fib::{:[x<2;x;fib(x-1)+fib(x-2)]}", "<t>").unwrap();
    ip.run("fib(5)", "<t>").unwrap();
    let (a0, b0) = (ALLOCS.load(Relaxed), BYTES.load(Relaxed));
    ip.run("fib(18)", "<t>").unwrap();
    let (a, b) = (ALLOCS.load(Relaxed) - a0, BYTES.load(Relaxed) - b0);
    let calls = 8361.0;
    println!("ALLOCS fib(18): {} allocations, {} bytes: {:.2} allocations and {:.0} bytes per call", a, b, a as f64 / calls, b as f64 / calls);
}
