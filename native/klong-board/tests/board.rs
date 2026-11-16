//! The skeleton on the host: the allocator over a region, and a board made of test parts
//! running the real console through `start` and `step`.
use std::alloc::{GlobalAlloc, Layout};
use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;

use klong_board::heap::RegionHeap;
use klong_board::{start, step, BoardConfig, BoardPlatform, Clock, NoFs, System, TickClock, Uart};
use klong_console::ConsoleConfig;
use klong_rt::memfs::MemFs;

struct Rng(u64);
impl Rng {
    fn below(&mut self, n: usize) -> usize {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 33) as usize % n
    }
}

#[test]
fn region_heap_allocates_frees_and_merges() {
    let mut region = vec![0u64; 8192];
    let bytes = region.len() * 8;
    let heap = RegionHeap::empty();
    // nothing before init
    assert!(unsafe { heap.alloc(Layout::from_size_align(8, 8).unwrap()) }.is_null());
    let usable = unsafe { heap.init(region.as_mut_ptr() as *mut u8, bytes) };
    assert_eq!(usable, bytes);
    assert_eq!(heap.free(), (bytes, bytes));
    let mut rng = Rng(104);
    let mut live: Vec<(*mut u8, Layout, u8)> = Vec::new();
    let mut refused = 0;
    for round in 0..200_000u32 {
        if live.len() < 200 && rng.below(100) < 55 {
            let size = match rng.below(10) {
                0 => 1 + rng.below(4000),
                1 => 1,
                _ => 1 + rng.below(120),
            };
            let layout = Layout::from_size_align(size, 1 << rng.below(8)).unwrap();
            let p = unsafe { heap.alloc(layout) };
            if p.is_null() {
                refused += 1;
                continue;
            }
            assert_eq!(p as usize % layout.align(), 0);
            let lo = region.as_ptr() as usize;
            assert!(p as usize >= lo && p as usize + size <= lo + bytes);
            let tag = round as u8;
            unsafe { p.write_bytes(tag, size) };
            live.push((p, layout, tag));
        } else if !live.is_empty() {
            let (p, layout, tag) = live.swap_remove(rng.below(live.len()));
            // nobody else wrote into it
            let s = unsafe { std::slice::from_raw_parts(p, layout.size()) };
            assert!(s.iter().all(|&b| b == tag));
            unsafe { heap.dealloc(p, layout) };
        }
    }
    assert!(refused > 0, "the region never filled up");
    for (p, layout, tag) in live.drain(..) {
        let s = unsafe { std::slice::from_raw_parts(p, layout.size()) };
        assert!(s.iter().all(|&b| b == tag));
        unsafe { heap.dealloc(p, layout) };
    }
    // everything came back and merged into the one block it started as
    assert_eq!(heap.free(), (bytes, bytes));
    assert_eq!(heap.blocks(), 1);
    // a request larger than the region, and one that overflows
    assert!(unsafe { heap.alloc(Layout::from_size_align(bytes + 1, 8).unwrap()) }.is_null());
    assert!(unsafe { heap.alloc(Layout::from_size_align(isize::MAX as usize - 4096, 4096).unwrap()) }.is_null());
    let all = unsafe { heap.alloc(Layout::from_size_align(bytes, 16).unwrap()) };
    assert!(!all.is_null());
    assert_eq!(heap.free(), (0, 0));
    unsafe { heap.dealloc(all, Layout::from_size_align(bytes, 16).unwrap()) };
    assert_eq!(heap.blocks(), 1);
    // an unaligned or tiny region
    let odd = RegionHeap::empty();
    let usable = unsafe { odd.init((region.as_mut_ptr() as *mut u8).add(3), 100) };
    assert!(usable >= 64 && usable <= 96 && usable % 8 == 0, "{usable}");
    assert_eq!(unsafe { RegionHeap::empty().init(region.as_mut_ptr() as *mut u8, 7) }, 0);
}

#[test]
fn tick_clock_counts_and_survives_a_wrap() {
    static CLOCK: TickClock = TickClock::new(1000);
    assert_eq!(CLOCK.now_us(), 0);
    for _ in 0..5 {
        CLOCK.tick();
    }
    assert_eq!(CLOCK.now_us(), 5000);
    assert_eq!((&CLOCK as &'static TickClock).now_us(), 5000);
}

#[derive(Clone, Default)]
struct Wire(Rc<RefCell<(VecDeque<u8>, Vec<u8>)>>);

impl Uart for Wire {
    fn read(&mut self, buf: &mut [u8]) -> usize {
        let mut w = self.0.borrow_mut();
        let n = buf.len().min(w.0.len()).min(5);
        for slot in buf[..n].iter_mut() {
            *slot = w.0.pop_front().unwrap();
        }
        n
    }
    fn write(&mut self, bytes: &[u8]) {
        self.0.borrow_mut().1.extend_from_slice(bytes);
    }
}

struct Still;
impl Clock for Still {
    fn now_us(&self) -> u64 {
        0
    }
}

#[derive(Clone, Default)]
struct Sys(Rc<RefCell<(u32, u32, Option<i32>)>>);
impl System for Sys {
    fn name(&self) -> &str {
        "testboard"
    }
    fn idle(&mut self) {
        self.0.borrow_mut().0 += 1;
    }
    fn wait(&mut self, _: Option<u64>) {
        self.0.borrow_mut().1 += 1;
    }
    fn reset(&mut self, code: i32) {
        self.0.borrow_mut().2 = Some(code);
    }
}

fn talk<P: klong_rt::Platform + 'static>(console: &mut klong_console::Console<P>, wire: &Wire, bytes: &[u8]) -> String {
    wire.0.borrow_mut().0.extend(bytes);
    while step(console) {}
    String::from_utf8(std::mem::take(&mut wire.0.borrow_mut().1)).unwrap()
}

/// One interpreter at a time (statics in klong-core).
static ONE: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[test]
fn a_board_from_parts_runs_the_console() {
    let _one = ONE.lock().unwrap_or_else(|e| e.into_inner());
    let wire = Wire::default();
    let sys = Sys::default();
    let platform = BoardPlatform::new(wire.clone(), Still, sys.clone(), Some(MemFs::new()));
    let cfg = BoardConfig { stack_bytes: 256 * 1024, console: ConsoleConfig { dumb: true, ..ConsoleConfig::default() }, ..BoardConfig::default() };
    let mut console = start(platform, cfg);
    assert!(talk(&mut console, &wire, b"").starts_with("Klong 0.0.1 (testboard, int"));
    // lines end CR LF on the wire
    assert_eq!(talk(&mut console, &wire, b".p(\"a\");+/!10\r"), "a\r\n45\r\nklong> ");
    let out = talk(&mut console, &wire, b"\x01ver\n");
    assert!(out.contains(" platform=testboard fs=1 "), "{out}");
    // an upload, and the file is the board's
    let data = b"k::9\n";
    let cmd = format!("\x01put k.kg {} {:08x}\n", data.len(), klong_console::crc::crc32(data));
    assert_eq!(talk(&mut console, &wire, cmd.as_bytes()), "\x01go\r\n");
    assert!(talk(&mut console, &wire, data).starts_with("\x01ok 5 "));
    assert_eq!(talk(&mut console, &wire, b".l(\"k.kg\");k\n"), "9\r\nklong> ");
    // recursion is bounded by the stack the port declared
    let out = talk(&mut console, &wire, b"f::{f(x+1)};f(0)\n");
    assert!(out.contains("range error"), "{out}");
    // .x asks the board to reset
    talk(&mut console, &wire, b".x(0)\n");
    assert_eq!(sys.0.borrow().2, Some(0));
    assert!(sys.0.borrow().0 > 0);
}

#[test]
fn a_board_without_storage() {
    let _one = ONE.lock().unwrap_or_else(|e| e.into_inner());
    let wire = Wire::default();
    let mut platform = BoardPlatform::new(wire.clone(), Still, Sys::default(), None::<NoFs>);
    platform.crlf = false;
    let cfg = BoardConfig { stack_bytes: 256 * 1024, console: ConsoleConfig { dumb: true, ..ConsoleConfig::default() }, ..BoardConfig::default() };
    let mut console = start(platform, cfg);
    talk(&mut console, &wire, b"");
    assert_eq!(talk(&mut console, &wire, b"\x01put f 1 0\n"), "\x01err system: this board has no file system\nklong> ");
    assert_eq!(talk(&mut console, &wire, b"\x01run 3 0e9db3e2\n"), "\x01go\n");
    assert!(talk(&mut console, &wire, b"7*6").contains("\x01"));
    assert!(talk(&mut console, &wire, b"\x01ver\n").contains(" fs=0 "));
}
