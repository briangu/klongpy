# Klong on the ESP32-P4

The serial console of the native Klong runtime (`native/PORTING.md`) as an ESP-IDF
application: ESP-IDF supplies boot, the UART driver, the clock and the heap; the runtime is
the Rust static library `native/klong-ffi` behind `klong-ffi/include/klong.h`.

Built for: ESP32-P4 chip revision v3.1 or later (the board at hand reads v3.2), 32 MB flash,
in-package PSRAM, console on UART0 (TX GPIO37, RX GPIO38: the board's USB-serial bridge),
115200 8N1. A chip below v3.0 needs another build (`CONFIG_ESP32P4_SELECTS_REV_LESS_V3=y`
in `sdkconfig.defaults`); the bootloader refuses this image on such a chip.

## Build

    ./build.sh

needs Docker with the ESP-IDF image pinned in `build.sh` (ESP-IDF v5.5.4) and stable Rust
with the P4's target (`rustup +stable target add riscv32imafc-unknown-none-elf`). It

1. builds `libklong_ffi.a` for `riscv32imafc-unknown-none-elf` (release, LTO) into `native/target/esp32p4/`;
2. runs `idf.py -DIDF_TARGET=esp32p4 build` and `idf.py size` in the container, with
   `native/` mounted.

Output in `build/`: `bootloader/bootloader.bin`, `partition_table/partition-table.bin`,
`klong_esp32p4.bin`, `flash_args`. For a clean build remove `build/` and `sdkconfig`.

Why that Rust target: every ESP-IDF build for the P4 uses the `ilp32f` calling convention
(`-march=rv32imafc... -mabi=ilp32f`), and the linker refuses to mix it with soft-float
(`ilp32`) objects such as those of `riscv32imac-unknown-none-elf`. The target
`riscv32imafc-unknown-none-elf` has the same ABI. `klong.h` passes only integers and
pointers. Doubles are computed in software on both sides (the P4 has single-precision
hardware only); the helpers (`__adddf3`, `__muldi3`, ...) and `memcpy` and friends resolve
to ESP-IDF's ROM and libc, Rust's own copies being weak symbols.

## Flash

Check the chip first (the image is for revision v3.1+ and 32 MB of flash):

    esptool --chip esp32p4 -p PORT chip-id
    esptool --chip esp32p4 -p PORT flash-id

Then, from this directory (esptool 5):

    esptool --chip esp32p4 -p PORT -b 460800 --before default-reset --after hard-reset \
        write-flash --flash-mode dio --flash-freq 80m --flash-size 32MB \
        0x2000 build/bootloader/bootloader.bin \
        0x8000 build/partition_table/partition-table.bin \
        0x10000 build/klong_esp32p4.bin

This overwrites the bootloader, the partition table and the first application partition
of whatever was on the board; back the flash up first if that matters.

## Use

Open the port at 115200 8N1 (no flow control). After the boot loader's and ESP-IDF's boot
log:

    Klong 0.0.1 (esp32p4, int64 float64)
    klong>

`native/tools/kgupload.py PORT ver` talks to the `^A` command mode; opening the port may
reset the board (DTR/RTS), the tool waits for the banner.

## How it is put together

| | |
|---|---|
| Task | `klong`, pinned to CPU 1, priority 1, **128 KiB stack** (internal RAM); the runtime gets a budget of 120 KiB (`Config::for_stack`), so deep recursion is a `range` error |
| Heap | `heap_caps_malloc` / `heap_caps_aligned_alloc` in PSRAM (`MALLOC_CAP_SPIRAM`); limit three quarters of the free PSRAM (about 24 MiB), 16 KiB reserve; without PSRAM 256 KiB of internal RAM. Blocks of up to 128 bytes come from free lists by size class in 4 KiB slabs of internal RAM (96 KiB at most; when internal RAM is spoken for, from the heap, 8-aligned, and freed there); ESP-IDF's own C is built with `CONFIG_COMPILER_OPTIMIZATION_PERF` |
| Console | ESP-IDF UART driver on UART0, 32 KiB receive buffer, `uart_read_bytes` with no wait, `uart_write_bytes`; raw bytes (no VFS, so no echo, line buffering or CR/LF translation by ESP-IDF); the runtime sends CR LF |
| Logging | `esp_log_level_set("*", ESP_LOG_NONE)` at the start of `app_main`; only the boot log comes before the banner. A panic (a bug, or a watchdog) still prints and restarts the board |
| Clock | `esp_timer_get_time` |
| Idle | `esp_task_wdt_reset` (the task is subscribed to the task watchdog, 30 s, panic); nothing to do: `vTaskDelay(1)`, 1 ms |
| `.x()` | `esp_restart` |
| Files | RAM file system (`MemFs`) in the same heap: uploads are lost at reset. Flash storage is not done |
| Features | `files`, `lib`, `timers`; no IPC (the C interface has no transports yet) |

## Verified on the board

The firmware has been built and its C interface tested on the host
(`cargo test -p klong-ffi`); the same `main.c` was run for an ESP32-C3 under QEMU. On the
P4 itself (2026-11-16) the REPL, a runaway loop stopped by Ctrl-C (`{1}{x+1}:~0`), the embedded
library, uploads and the conformance suite over the serial port (1266 of 1266) worked: see
`docs/native/esp32-p4.md`, which has the measurements. IPC (the firmware has none)
and flash storage (not done) are not there to check.
