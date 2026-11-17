//! The probe as a linked image for a bare-metal target (riscv32imac-unknown-none-elf,
//! thumbv7em-none-eabihf): `_start` runs one program. Not bootable: there is no linker script
//! and no start-up code; it exists so that the section sizes are those of
//! everything a board build links. On wasm32 this file is empty (the library is the probe).
#![no_std]
#![no_main]

use klong_probe as _;

#[cfg(not(target_arch = "wasm32"))]
#[no_mangle]
pub extern "C" fn _start() -> ! {
    let program = "f::{:[x<2;x;f(x-1)+f(x-2)]};.p(f(20))";
    // SAFETY: the program is readable memory.
    unsafe { klong_probe::klong_run(program.as_ptr(), program.len()) };
    loop {}
}
