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

#[cfg(feature = "speed")]
fn assert_linear_collection_bytes(body: &'static str) {
    let mut ip = Interp::new(Config::default(), Box::new(NoHooks));
    let module = ip.module_new("collections", "", false).unwrap();
    ip.module_run_static(module, body, "<module>").unwrap();
    let mut measure = |call| {
        let before = BYTES.get();
        let value = ip.module_run_static(module, call, "<call>").unwrap().unwrap();
        (value.as_int().unwrap(), BYTES.get() - before)
    };
    measure("build(8)");
    let (small_result, small) = measure("build(256)");
    let (large_result, large) = measure("build(512)");
    assert_eq!((small_result, large_result), (256, 512));
    println!("256 steps: {small} bytes; 512 steps: {large} bytes");
    assert!(large < 3 * small, "doubling the workload allocated {large} bytes versus {small}");
}

#[cfg(feature = "speed")]
#[test]
fn record_appends_allocate_linear_bytes_in_a_module() {
    assert_linear_collection_bytes(r#"build(n)::{[q];q::[];{q::q,,[;x;"pulse"];0}'!n;#q}"#);
}

#[cfg(feature = "speed")]
#[test]
fn packed_amends_allocate_linear_bytes_in_a_module() {
    assert_linear_collection_bytes("build(n)::{[v];v::n#0;{v::v:=1,x;0}'!n;+/v}");
}

#[cfg(feature = "speed")]
#[test]
fn shared_vector_amend_allocates_one_packed_copy() {
    use klong_core::value::{Flt, Int};

    let mut ip = Interp::new(Config::default(), Box::new(NoHooks));
    for (setup, change, width) in [
        ("v::8192#0", "v::v:=[9 -1]", size_of::<Int>()),
        ("v::8192#0.0", "v::v:=[9.5 -1.0]", size_of::<Flt>()),
    ] {
        ip.run(setup, "<setup>").unwrap();
        ip.run("snapshot::v", "<alias>").unwrap();
        let before = BYTES.get();
        ip.run(change, "<update>").unwrap();
        let allocated = BYTES.get() - before;
        println!("{setup}: {allocated} bytes for an aliased amend");
        // The primitive copies one packed buffer. The remaining allowance covers
        // parsing the short standalone expression, dispatch and the array header.
        assert!(allocated <= (8192 * width + 8192) as u64, "allocated {allocated} bytes");
        let unchanged = ip.run("0=snapshot@(-1)", "<check>").unwrap().unwrap();
        assert_eq!(unchanged.as_int(), Some(1));
    }
}
