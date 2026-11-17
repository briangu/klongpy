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
    cargo +stable build --release -p klong-ffi --features lane,pico --target $TARGET --target-dir target/pico-bench)
ls -l "$native/target/pico-bench/$TARGET/release/libklong_ffi.a"

# KLONG_SELFTEST=0 leaves the boot self-test out; LANE_BAUD=115200 builds for a slow lane
# KLONG_STACK_KB=96 gives the interpreter a stack of 96 KB (default 137, in klong_memory.h) and the heap what is left
cmake -S "$here" -B "$here/build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DKLONG_SELFTEST="${KLONG_SELFTEST:-1}" -DLANE_BAUD="${LANE_BAUD:-4500000}" -DKLONG_STACK_KB="${KLONG_STACK_KB:-}"
cmake --build "$here/build"
arm-none-eabi-size "$here/build/klong_pico_bench.elf"
ls -l "$here/build/klong_pico_bench.uf2" "$here/build/klong_pico_bench.bin"

# The RAM map: .data and .bss from the ELF, the stack from the top of RAM, the heap in between
nm=arm-none-eabi-nm
end=$($nm "$here/build/klong_pico_bench.elf" | awk '$3=="end"{print $1}')
stack=${KLONG_STACK_KB:+$((KLONG_STACK_KB * 1024))}
stack=${stack:-$(sed -n 's/^#define KLONG_STACK_BYTES (\([0-9]*\)u \* 1024)/\1/p' "$here/klong_memory.h" | head -1 | awk '{print $1 * 1024}')}
top=$((0x20042000))
echo "RAM: static $((0x$end - 0x20000000)) bytes, heap $((top - stack - 0x$end)) bytes, stack $stack bytes, of $((top - 0x20000000))"
