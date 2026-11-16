//! A whole board image from the skeleton, for a bare-metal target
//! (riscv32imac-unknown-none-elf, thumbv7em-none-eabihf):
//!
//!     cargo +stable build -p klong-board --profile min-size --target riscv32imac-unknown-none-elf
//!
//! It links, and its section sizes are those of a real console build (scripts/sections.py),
//! but it is not bootable: a port adds the vendor's start-up code and linker script (stack
//! pointer, `.data` and `.bss`, the vector table), sets the UART's pins and baud rate, and
//! connects a timer interrupt to `TICKS.tick()`. The UART here is a 16550 as on QEMU's `virt`
//! machines. On the host this file is an empty program.
//!
//! Without the feature `console` the image runs one program instead of the console: the
//! difference between the two builds is what the console costs.
#![cfg_attr(target_os = "none", no_std, no_main)]

#[cfg(not(target_os = "none"))]
fn main() {}

#[cfg(target_os = "none")]
mod image {
    use core::cell::UnsafeCell;

    use klong_board::heap::{HeapRef, RegionHeap};
    use klong_board::{BoardConfig, BoardPlatform, System, TickClock, Uart};
    use klong_rt::heap::Counting;
    use klong_rt::memfs::MemFs;

    /// The heap: this much of `.bss`. A port with external RAM gives that region instead.
    const HEAP_BYTES: usize = 64 * 1024;
    /// Held back so that a `memory` error can unwind and print.
    const RESERVE: usize = 8 * 1024;
    /// What the start-up code gives the stack.
    const STACK_BYTES: usize = 32 * 1024;

    #[repr(C, align(16))]
    struct Arena(UnsafeCell<[u8; HEAP_BYTES]>);
    // SAFETY: only the allocator touches it, from one thread of control.
    unsafe impl Sync for Arena {}
    static ARENA: Arena = Arena(UnsafeCell::new([0; HEAP_BYTES]));

    static REGION: RegionHeap = RegionHeap::empty();
    #[global_allocator]
    static HEAP: Counting<HeapRef> = Counting::new(HeapRef(&REGION));

    /// 1 kHz from a timer interrupt (the port's handler calls `TICKS.tick()`).
    static TICKS: TickClock = TickClock::new(1000);

    fn mem() -> (usize, usize, usize) {
        (HEAP.used(), HEAP.peak(), HEAP.limit())
    }
    fn pressed() -> bool {
        HEAP.pressed()
    }

    /// A 16550 UART at a fixed address, polled.
    struct Ns16550(*mut u8);

    impl Uart for Ns16550 {
        fn read(&mut self, buf: &mut [u8]) -> usize {
            let mut n = 0;
            // SAFETY: the registers of the UART at the board's address.
            unsafe {
                while n < buf.len() && self.0.add(5).read_volatile() & 0x01 != 0 {
                    buf[n] = self.0.read_volatile();
                    n += 1;
                }
            }
            n
        }
        fn write(&mut self, bytes: &[u8]) {
            for &b in bytes {
                // SAFETY: as above; waits for room in the transmitter.
                unsafe {
                    while self.0.add(5).read_volatile() & 0x20 == 0 {}
                    self.0.write_volatile(b);
                }
            }
        }
    }

    struct Bare;
    impl System for Bare {
        fn name(&self) -> &str {
            "skeleton"
        }
    }

    #[panic_handler]
    fn panic(_: &core::panic::PanicInfo) -> ! {
        loop {}
    }

    #[no_mangle]
    pub extern "C" fn _start() -> ! {
        // SAFETY: the arena is used by nothing else, and nothing has allocated yet.
        let bytes = unsafe { REGION.init(ARENA.0.get() as *mut u8, HEAP_BYTES) };
        HEAP.set_limit(bytes, RESERVE);
        let platform = BoardPlatform::new(Ns16550(0x1000_0000 as *mut u8), &TICKS, Bare, Some(MemFs::new()));
        let cfg = BoardConfig { stack_bytes: STACK_BYTES, mem: Some(mem), mem_pressed: Some(pressed), ..BoardConfig::default() };
        #[cfg(feature = "console")]
        klong_board::run(platform, cfg);
        #[cfg(not(feature = "console"))]
        {
            use klong_rt::Platform;
            let config = klong_core::Config { mem_probe: cfg.mem_pressed, ..klong_core::Config::for_stack(cfg.stack_bytes) };
            let mut rt = klong_rt::Runtime::new(platform, config, cfg.mem);
            let text = match rt.run("f::{:[x<2;x;f(x-1)+f(x-2)]};f(20)", "<image>") {
                Ok(Some(v)) => rt.interp.format(&v),
                Ok(None) => alloc::string::String::new(),
                Err(e) => e.render(),
            };
            rt.shared.platform.borrow_mut().console_write(text.as_bytes());
            loop {}
        }
    }

    extern crate alloc;
}
