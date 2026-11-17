#!/bin/sh
# Builds and size figures. Run from anywhere: sh native/scripts/measure.sh
# The 32-bit builds need these targets on the stable toolchain (rustup +stable target add ...):
#   wasm32-unknown-unknown         the probe as a module that scripts/probe.mjs can run
#   riscv32imac-unknown-none-elf   the probe as a linked image (not bootable)
#   thumbv7em-none-eabihf          the same
# A target that is not installed is skipped.
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

have() { rustup +stable target list --installed 2>/dev/null | grep -qx "$1"; }

# name:cargo options -- the feature sets of the probe
SETS="default: narrow:--features=int32,float32 bare:--no-default-features bare-narrow:--no-default-features+--features=int32,float32"

if have wasm32-unknown-unknown; then
    echo "== wasm32-unknown-unknown: compile-time size_of::<Value>() checks, probe size"
    cargo +stable build --release -p klong-core -p klong-rt --target wasm32-unknown-unknown
    for set in $SETS; do
        name=${set%%:*}
        opts=$(echo "${set#*:}" | tr '+' ' ')
        (cd klong-probe && cargo +stable build -q --release --target wasm32-unknown-unknown --target-dir ../target/probe-$name $opts)
        ls -l target/probe-$name/wasm32-unknown-unknown/release/klong_probe.wasm | awk -v n="$name" '{print "  " n ": " $5 " bytes"}'
    done
    if command -v node >/dev/null; then
        node scripts/probe.mjs target/probe-default/wasm32-unknown-unknown/release/klong_probe.wasm 'f::{:[x<2;x;f(x-1)+f(x-2)]};f(20)'
        node scripts/probe.mjs target/probe-default/wasm32-unknown-unknown/release/klong_probe.wasm '.l("nstat");mu([1 2 3 4])'
        node scripts/probe.mjs target/probe-narrow/wasm32-unknown-unknown/release/klong_probe.wasm '2147483647+1'
    fi
fi

for target in riscv32imac-unknown-none-elf thumbv7em-none-eabihf; do
    have $target || { echo "== $target: not installed, skipped"; continue; }
    echo "== $target: sections of the linked probe (the 64 KiB heap of the probe is taken off .bss)"
    for set in $SETS; do
        name=${set%%:*}
        opts=$(echo "${set#*:}" | tr '+' ' ')
        (cd klong-probe && cargo +stable build -q --release --target $target --target-dir ../target/probe-$name $opts 2>/dev/null)
        echo "  [$name]"
        python3 scripts/sections.py target/probe-$name/$target/release/klong-probe --heap 65536 | tail -1
    done
    # what the image is made of: the same build with its symbols
    (cd klong-probe && cargo +stable build -q --release --target $target --target-dir ../target/probe-sym --config 'profile.release.strip=false' 2>/dev/null)
    python3 scripts/sections.py target/probe-sym/$target/release/klong-probe --heap 65536 --top 12
done

# The board skeleton with the console, and without it (one fixed program), as
# linked images. The difference is what the console and the main loop cost.
BOARD="console: noconsole:--no-default-features+--features=files,lib console-bare:--no-default-features+--features=console noconsole-bare:--no-default-features console-narrow:--features=int32,float32"
for target in riscv32imac-unknown-none-elf thumbv7em-none-eabihf; do
    have $target || continue
    echo "== $target: sections of klong-board-image (its 64 KiB heap is taken off .bss)"
    for set in $BOARD; do
        name=${set%%:*}
        opts=$(echo "${set#*:}" | tr '+' ' ')
        cargo +stable build -q -p klong-board --profile min-size --target $target --target-dir target/board-$name $opts 2>/dev/null
        echo "  [$name]"
        python3 scripts/sections.py target/board-$name/$target/min-size/klong-board-image --heap 65536 | tail -1
    done
done
