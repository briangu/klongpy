//! How many allocations a call of a Klong function costs (a counting allocator; in a file of its own).
use klong_core::{Config, Interp, NoHooks};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

struct Counting;
// Measure the interpreter thread independently of allocations made by libtest and its other workers.
thread_local! {
    static ALLOCS: Cell<u64> = const { Cell::new(0) };
    static BYTES: Cell<u64> = const { Cell::new(0) };
    static LIVE: Cell<i64> = const { Cell::new(0) };
}
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        ALLOCS.set(ALLOCS.get() + 1);
        BYTES.set(BYTES.get() + l.size() as u64);
        LIVE.set(LIVE.get() + l.size() as i64);
        System.alloc(l)
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        LIVE.set(LIVE.get() - l.size() as i64);
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
    let (a0, b0) = (ALLOCS.get(), BYTES.get());
    ip.run("fib(18)", "<t>").unwrap();
    let (a, b) = (ALLOCS.get() - a0, BYTES.get() - b0);
    let calls = 8361.0;
    println!("ALLOCS fib(18): {} allocations, {} bytes: {:.2} allocations and {:.0} bytes per call", a, b, a as f64 / calls, b as f64 / calls);
}

#[test]
fn a_call_in_the_steady_state_allocates_nothing() {
    let mut ip = Interp::new(Config::default(), Box::new(NoHooks));
    ip.run("fib::{:[x<2;x;fib(x-1)+fib(x-2)]}\nsq(a)::{a*a}", "<t>").unwrap();
    ip.run("fib(10);{sq(x)}'!100", "<t>").unwrap();
    let a0 = ALLOCS.get();
    ip.run("fib(18)", "<t>").unwrap();
    let a = ALLOCS.get() - a0;
    // (8361 calls: the allocations are the parse of the text and the result, not the calls)
    assert!(a < 100, "{} allocations for 8361 calls", a);
}

/// Programs that fail at depth, again and again (the error leaves `run`): the heap in use is the same after each round, the slots are given back.
#[test]
fn failures_do_not_grow_the_heap() {
    let t = std::thread::Builder::new().stack_size(256 << 20).spawn(|| {
        let mut ip = Interp::new(Config::default(), Box::new(NoHooks));
        ip.run("rec::{rec(x+1)}\nrecc::{[a];a::x;1+recc(x+1)}\nbad(a)::{a+undefined_name}\nsq(a)::{a*a}", "<t>").unwrap();
        let progs = ["rec(0)", "{bad(x)}'!4", "recc(0)", "{:[x>3;bad(x);x+y]}/!9", "sq(bad(1))", "sq(1)+bad(2)", "{sq(x)+bad(x)}'!3"];
        for p in progs {
            assert!(ip.run(p, "<t>").is_err(), "{}", p);
            assert!(ip.run("sq(7)", "<t>").is_ok());
        }
        let before = LIVE.get();
        for _ in 0..30 {
            for p in progs {
                assert!(ip.run(p, "<t>").is_err());
            }
        }
        assert_eq!(LIVE.get(), before);
    });
    t.unwrap().join().unwrap();
}
