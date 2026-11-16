#!/bin/sh
# Builds and size figures of ticket 102. Run from anywhere: sh native/scripts/measure.sh
# Needs the wasm32-unknown-unknown target on the stable toolchain for the 32-bit builds
# (the only target without an operating system that was installed; a board target such as
# riscv32imafc-unknown-none-elf or thumbv7em-none-eabihf replaces it once installed).
set -e
cd "$(dirname "$0")/.."

echo "== host builds"
cargo build --release
cargo build --profile min-size -p klong-cli
cargo build --profile min-size -p klong-cli --features int32,float32 --target-dir target/narrow
ls -l target/release/klong target/min-size/klong target/narrow/min-size/klong | awk '{print $5, $9}'
for f in target/release/klong target/min-size/klong target/narrow/min-size/klong; do
    echo "$f: $(size -m "$f" 2>/dev/null | grep 'Section __text' | tr -d '\t')"
done

echo "== no_std: the core without default features, on the host"
cargo build -p klong-core --no-default-features
cargo build -p klong-core --no-default-features --features int32,float32

echo "== 32-bit, no operating system (wasm32-unknown-unknown): compile-time size_of::<Value>() checks, probe size"
cargo +stable build --release -p klong-core -p klong-rt --target wasm32-unknown-unknown
(cd klong-probe && cargo +stable build --release --target wasm32-unknown-unknown --target-dir ../target/probe)
(cd klong-probe && cargo +stable build --release --target wasm32-unknown-unknown --target-dir ../target/probe-narrow --features int32,float32)
ls -l target/probe/wasm32-unknown-unknown/release/klong_probe.wasm target/probe-narrow/wasm32-unknown-unknown/release/klong_probe.wasm | awk '{print $5, $9}'
if command -v node >/dev/null; then
    node scripts/probe.mjs target/probe/wasm32-unknown-unknown/release/klong_probe.wasm 'f::{:[x<2;x;f(x-1)+f(x-2)]};f(20)'
    node scripts/probe.mjs target/probe-narrow/wasm32-unknown-unknown/release/klong_probe.wasm '2147483647+1'
fi
