#!/bin/sh
# Build the Pico bench firmware image (.uf2). Nothing is flashed, no device is opened.
#   1. libklong_ffi.a for thumbv6m-none-eabi (Cortex-M0+) with the stable toolchain:
#        rustup +stable target add thumbv6m-none-eabi
#   2. the Pico SDK application around it, with the Arm GNU toolchain, out of tree in build/.
# The SDK and the toolchain are only read. Usage: ./build.sh
set -eu
cd "$(dirname "$0")"
here=$PWD
native=$(cd ../.. && pwd)
TARGET=thumbv6m-none-eabi
export PICO_SDK_PATH=${PICO_SDK_PATH:-$HOME/scm/pico-sdk}
ARM_BIN=${ARM_BIN:-$HOME/.local/arm-gnu-toolchain-15.3.rel2-darwin-arm64-arm-none-eabi/bin}
export PATH="$ARM_BIN:$PATH"

# KLONG_OPT=z gives the smaller image (about 420 KB instead of 610 KB) at some speed.
(cd "$native" &&
    CARGO_PROFILE_RELEASE_OPT_LEVEL=${KLONG_OPT:-3} CARGO_PROFILE_RELEASE_LTO=fat CARGO_PROFILE_RELEASE_CODEGEN_UNITS=1 CARGO_PROFILE_RELEASE_PANIC=abort \
    cargo +stable build --release -p klong-ffi --features lane --target $TARGET --target-dir target/pico-bench)
ls -l "$native/target/pico-bench/$TARGET/release/libklong_ffi.a"

# KLONG_SELFTEST=0 leaves the boot self-test out; LANE_BAUD=115200 builds for a slow lane
cmake -S "$here" -B "$here/build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DKLONG_SELFTEST="${KLONG_SELFTEST:-1}" -DLANE_BAUD="${LANE_BAUD:-4500000}"
cmake --build "$here/build"
arm-none-eabi-size "$here/build/klong_pico_bench.elf"
ls -l "$here/build/klong_pico_bench.uf2" "$here/build/klong_pico_bench.bin"
