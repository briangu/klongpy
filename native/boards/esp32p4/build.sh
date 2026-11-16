#!/bin/sh
# Build the ESP32-P4 firmware: the Rust runtime as a static library on the host, then the
# ESP-IDF application around it inside Docker. Usage: ./build.sh [idf.py arguments]
# Output: build/ (bootloader/bootloader.bin, partition_table/partition-table.bin,
# klong_esp32p4.bin, flash_args). Nothing is flashed.
set -eu
cd "$(dirname "$0")"
here=$PWD
native=$(cd ../.. && pwd)
IDF_IMAGE=espressif/idf@sha256:b9f2d6ea1c19e0c9f7959bdb74a9e3c775642f9d0f3b841937c5fa3363db892b
TARGET=riscv32imafc-unknown-none-elf

# 1. libklong_ffi.a for rv32imafc with the ilp32f ABI (what ESP-IDF uses on the P4), with the
#    stable toolchain: rustup +stable target add riscv32imafc-unknown-none-elf
(cd "$native" &&
    CARGO_PROFILE_RELEASE_LTO=fat CARGO_PROFILE_RELEASE_CODEGEN_UNITS=1 CARGO_PROFILE_RELEASE_PANIC=abort \
    cargo +stable build --release -p klong-ffi --target $TARGET --target-dir target/esp32p4)
ls -l "$native/target/esp32p4/$TARGET/release/libklong_ffi.a"

# 2. The ESP-IDF application. native/ is mounted because the project refers to
#    ../../klong-ffi/include and ../../target/esp32p4.
idf() {
    docker run --rm -v "$native:/work" -w /work/boards/esp32p4 "$IDF_IMAGE" idf.py -DIDF_TARGET=esp32p4 "$@"
}
idf "$@" build
idf size
echo
echo "flash (from $here/build): esptool --chip esp32p4 -p PORT -b 460800 --before default-reset --after hard-reset write-flash @flash_args"
