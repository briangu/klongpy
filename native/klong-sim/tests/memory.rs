//! Ticket 104: exhausting the heap from the prompt is a `memory` error and the prompt stays
//! alive. This test binary has the counting allocator of a board as its global allocator, so
//! it holds one test only.
use std::alloc::System;

use klong_core::Config;
use klong_rt::heap::Counting;
use klong_sim::{Sim, SimOptions};

#[global_allocator]
static HEAP: Counting<System> = Counting::new(System);

fn stats() -> (usize, usize, usize) {
    (HEAP.used(), HEAP.peak(), HEAP.limit())
}

fn pressed() -> bool {
    HEAP.pressed()
}

#[test]
fn exhausting_the_heap_is_a_memory_error_and_the_prompt_stays() {
    let mut s = Sim::new(SimOptions { config: Config { mem_probe: Some(pressed), ..Config::default() }, mem: Some(stats), ..SimOptions::default() });
    s.run();
    let base = HEAP.used();
    HEAP.set_limit(base + 256 * 1024, 16 * 1024);
    // one allocation that cannot be had
    let out = s.io(b"a::!10000000\n");
    assert!(out.contains("memory error"), "{out}");
    assert!(out.ends_with("klong> "));
    // many small ones, until the reserve is reached
    let out = s.io(b"a::[];{1}{a::a,[[1 2 3]]}:~0\n");
    assert!(out.contains("memory error"), "{out}");
    assert!(out.ends_with("klong> "));
    // the console itself still has what it needs
    let out = s.io(b"a::0\n\x01ver\n");
    let free: usize = out.split(" heap_free=").nth(1).and_then(|t| t.split(' ').next()).and_then(|n| n.parse().ok()).expect("heap_free in ver");
    assert!(free > 128 * 1024 && free <= 256 * 1024, "{out}");
    assert!(out.contains(&format!(" heap_limit={}", base + 256 * 1024)));
    assert_eq!(s.io(b"+/!1000\n"), "+/!1000\n499500\nklong> ");
    // a program sent with run, and the room for one that cannot be had
    let p = b"!10000000";
    let out = s.io(format!("\x01run {} {:08x}\n!10000000", p.len(), klong_console::crc::crc32(p)).as_bytes());
    assert!(out.contains("\x01err memory: "), "{out}");
    HEAP.set_limit(HEAP.used() + 30 * 1024, 16 * 1024);
    let out = s.io(b"\x01run 60000 0\n");
    assert!(out.contains("\x01err memory: out of memory\n"), "{out}");
    assert!(!s.console.busy());
    assert_eq!(HEAP.failed() > 0, true);
    HEAP.set_limit(0, 0);
    assert_eq!(s.io(b"2+2\n"), "2+2\n4\nklong> ");
}
