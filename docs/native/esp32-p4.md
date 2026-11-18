# ESP32-P4

This page shows how to build and flash the native Klong runtime on an ESP32-P4 development board, connect to its serial console, upload files and run the conformance suite over the serial port. It is for someone who has such a board and wants to try it. What has run on this and other hardware is in the [status table](index.md#status-what-has-and-has-not-been-proven).

!!! note "What was measured on hardware"
    Conditions: an ESP32-P4 development board, chip revision v3.2, 32 MB flash, 32 MB PSRAM; ESP-IDF v5.5.4; measured on 2026-11-16 with one image of 691,328 bytes. Every figure on this page that comes from a board is for that image. The image predates the current source: the source differs in the receive-buffer drain at boot and in the C optimisation level and small-block allocation described below, and those differences have not been run on a board.

    * The REPL works, Ctrl-C stops a runaway loop, the embedded library loads and uploads work.
    * The conformance suite passed 1266 of 1266 cases over the serial port in 1877 seconds.
    * `fib(24)` took 4.2 seconds, against about 35 ms on a desktop. The cause is not known.

    The firmware has no IPC. The `thumbv7em` target has not run on any hardware; the two Picos ran the `thumbv6m` target (see the [status table](index.md#status-what-has-and-has-not-been-proven)).

The commands on this page that need the board, Docker or the toolchain are also in `native/boards/esp32p4/README.md`. The transcripts in this section use the [simulated board](console.md).

## What the firmware is

The firmware is an ESP-IDF application. ESP-IDF supplies the boot code, the UART driver, the clock and the heap. The runtime is the Rust static library `native/klong-ffi`, which is linked through the C interface in `native/klong-ffi/include/klong.h` (see [Porting to a Board](porting.md)). The firmware gives you the [serial console](console.md) on the board's serial port and nothing else. It has no network and no IPC.

It is built for an ESP32-P4 of chip revision v3.1 or later (a v3.2 chip was tested), with 32 MB of flash and in-package PSRAM, with the console on UART0 (TX on GPIO37, RX on GPIO38), which is the board's USB-serial bridge, at 115200 baud, 8 data bits, no parity, one stop bit. A chip below revision v3.0 needs another build (`CONFIG_ESP32P4_SELECTS_REV_LESS_V3=y` in `sdkconfig.defaults`), and the bootloader refuses this image on such a chip.

| | |
|---|---|
| Task | `klong`, pinned to CPU 1, priority 1, 128 KiB stack in internal RAM. The runtime is given a budget of 120 KiB (`Config::for_stack`), so deep recursion is a `range` error and not a crash |
| Heap | in PSRAM (`MALLOC_CAP_SPIRAM`); the limit is three quarters of the free PSRAM (about 24 MiB) with a reserve of 16 KiB; without PSRAM, 256 KiB of internal RAM |
| Console | the ESP-IDF UART driver on UART0 with a 32 KiB receive buffer; raw bytes, no echo or line buffering by ESP-IDF; the runtime sends CR LF |
| Logging | switched off at the start of `app_main`; only the boot log comes before the banner. A panic (a bug, or a watchdog) still prints and restarts the board |
| Clock | `esp_timer_get_time` |
| Idle | feeds the task watchdog (30 s, panic) and sleeps 1 ms |
| `.x()` | `esp_restart`, which is how the conformance runner gets a fresh interpreter per case |
| Files | a RAM file system in the same heap; **uploads are lost at reset**. Flash storage is not done |
| Features | `files`, `lib`, `timers`; no IPC (this firmware does not use the transports that the C interface offers, a UART link called a lane; see [Porting to a Board](porting.md#the-c-interface)) |

## Building the firmware

You need Docker with the ESP-IDF image that `build.sh` pins (ESP-IDF v5.5.4), and stable Rust with the target of the P4:

```sh
rustup +stable target add riscv32imafc-unknown-none-elf
cd native/boards/esp32p4
./build.sh
```

`build.sh` does two things. It builds `libklong_ffi.a` for `riscv32imafc-unknown-none-elf` (release, with LTO) into `native/target/esp32p4/`, and then it runs `idf.py -DIDF_TARGET=esp32p4 build` and `idf.py size` in the container, with `native/` mounted. The output is in `native/boards/esp32p4/build/`: `bootloader/bootloader.bin`, `partition_table/partition-table.bin`, `klong_esp32p4.bin` and `flash_args`. For a clean build, remove `build/` and `sdkconfig`.

The Rust target is `riscv32imafc` and not the more common `riscv32imac` because every ESP-IDF build for the P4 uses the `ilp32f` calling convention, and the linker refuses to mix it with the soft-float objects of `riscv32imac-unknown-none-elf`. The C interface passes only integers and pointers. Doubles are computed in software on both sides (the P4 has single-precision hardware only).

The image `klong_esp32p4.bin` is 691,328 bytes in a partition of 1.5 MB, so 55 percent of the partition is free. The bootloader is 23,072 bytes and the partition table 3,072. The Rust library is about 481 KB of the image. It is built with the speed optimisation level (`opt-level` 3) and not for size, since flash is not scarce on this board.

## Flashing

Check the chip first. The image is for revision v3.1 or later and 32 MB of flash:

```sh
esptool --chip esp32p4 -p PORT chip-id
esptool --chip esp32p4 -p PORT flash-id
```

Then, from `native/boards/esp32p4` (esptool 5; `write-flash @flash_args` from the `build/` directory does the same, which is what `build.sh` suggests when it ends):

```sh
esptool --chip esp32p4 -p PORT -b 460800 --before default-reset --after hard-reset \
    write-flash --flash-mode dio --flash-freq 80m --flash-size 32MB \
    0x2000 build/bootloader/bootloader.bin \
    0x8000 build/partition_table/partition-table.bin \
    0x10000 build/klong_esp32p4.bin
```

`PORT` is the serial device of the board, for example `/dev/ttyUSB0` on Linux or `/dev/cu.usbserial-110` on macOS. This overwrites the bootloader, the partition table and the first application partition of whatever was on the board. Back the flash up first if that matters.

## Connecting

Open the port at 115200 8N1 with no flow control, with any terminal program that opens a serial device. Two things are worth knowing before you do.

* **Opening the port resets the board.** The USB-serial bridge toggles DTR and RTS, which reset the chip, so you see the boot log every time you open the port. A program that holds the board's state in RAM loses it. The upload tool waits for the banner after it opens the port.
* **After a reset the first line you send can fail** with `input is not valid UTF-8`. A stray byte arrives with or after the boot, and the console takes it into the line you type. Known limit: press Enter once, or Ctrl-C, and type the line again. The current `main.c` discards the receive buffer before it prints the banner (`drain_rx`).

## What you see

After the boot messages of the ROM, the bootloader and ESP-IDF, the board prints the banner and the prompt:

```
Klong 0.0.1 (esp32p4, int64 float64)
klong>
```

From here it is the console described on the [Serial Console and Upload](console.md) page, with the line editor, the history and the continuation lines. The board does what that page shows: expressions are evaluated, an error shows its position and the prompt returns, `.l("nstat")` loads the embedded library, and a runaway loop such as `{1}{x+1}:~0` is stopped by Ctrl-C with an `interrupt` error.

Integers and floats are 64 bits wide in this firmware (`int64 float64` in the banner); the P4 has single-precision floating point hardware only, so doubles are computed in software. `fib(24)` (`f::{:[x<2;x;f(x-1)+f(x-2)]};f(24)`) took 4.2 seconds on the board, and about 35 ms on a desktop (Apple M2 Ultra), a factor of about 120. The program uses integers only. Known limit: the cause of the difference has not been investigated, and the board's speed is not measured beyond that one figure.

A call of a Klong function costs about 28 us (4.2 s over the 150,049 calls of `fib(24)`), about 10,000 cycles at 360 MHz.

## Uploading a file

The tool is `native/tools/kgupload.py`, which uses only the Python standard library. On a board it works as on the simulated one:

```sh
python native/tools/kgupload.py PORT ver
python native/tools/kgupload.py PORT upload hello.kg
python native/tools/kgupload.py PORT run -e '.l("hello.kg")'
```

The first command prints the version line: widths, features, platform name, `fs=1`, the free heap and the console limits. The second stores `hello.kg` in the RAM file system of the board; for a UTF-8 text file it uses the `put` command and for any other file Intel HEX, with a check of the CRC-32 either way. The third runs a program on the board, here one that loads the file you stored. All subcommands and options are on the [console page](console.md#uploading-with-kguploadpy). A transfer that is cut off leaves the old file in place.

Because the file system is in RAM, **the file is gone after a reset**, which includes opening the serial port again and `.x(0)`. A file that the board should have after power-up has to be uploaded again each time. A `boot.kg` on the file system runs before the banner, but since the file system is empty at power-up, that is only useful once flash storage exists.

If the board is slow to boot, the first command waits for the banner; give it a longer `--timeout`.

## Running the conformance suite over the wire

`native/scripts/console_conformance.py` runs the cases of the conformance suite on the board. For each case it resets the board with `.x(0)`, waits for the banner, sets the strict level if the case has one, and sends the program with `^A run`; it compares the value, the console output and the kind of the error with the case:

```sh
python native/scripts/console_conformance.py --port PORT --baud 115200
```

This gave 1266 of 1266 on the ESP32-P4, in 1877 seconds (about 31 minutes). Most of the time goes into the reset and boot of the board before each case: that is about 1.5 seconds per case. With a firmware built with `int32` and `float32`, add `--narrow` so that the cases for that build are chosen. The `sysio` cases write files; on a board without storage you would leave them out with `--tags`, and here they run against the RAM file system.

The runner needs `.x` to really reset the board, a banner within `--timeout` and a `max_run` of at least the largest case, which holds for this firmware.

## What has not been done

* **IPC and the management node on this board**, `.cli`, `.srv`: this firmware has no transport for them. The C interface has transports (the `lane` feature, which the Pico firmware uses), but the ESP32-P4 firmware does not use them. IPC has run on two Picos, and on a desktop over TCP. See [IPC and Timers](ipc.md) and [Management Node](management-node.md).
* **Flash storage.** Files live in RAM and are lost at reset.
* **The stack and heap on the board.** The firmware's numbers (a budget of 120 KiB of the 128 KiB stack, a heap limit of three quarters of the free PSRAM) come from the porting rules and the build; how much a level of recursion costs on this chip, how the heap behaves in PSRAM under load, and whether the 30 second task watchdog is long enough for the longest single operation, have not been measured.
* **The receive path at speed.** Whether bytes are lost during long evaluations at 115200 baud with the 32 KiB buffer is not known beyond the conformance run, which passed.
* **Why `fib(24)` is slow**, as said above. The current source builds ESP-IDF's C with `-O2` and serves the small blocks that a call allocates from free lists in internal RAM; `fib(24)` with it has not been measured.
* **The `thumbv7em` target and the other boards.** The code links for `thumbv7em-none-eabihf` and `riscv32imac-unknown-none-elf`, but those images have no start-up code and were never booted. The same `main.c` ran for an ESP32-C3 under QEMU, which is not hardware: it booted and ran expressions, uploads, timers and `.x`. Known limit: a runaway loop there ends in an interrupt watchdog timeout, and the cause is not investigated.
