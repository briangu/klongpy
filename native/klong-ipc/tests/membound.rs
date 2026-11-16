//! Bounded memory: decoding a hostile message never allocates more than a fixed multiple of
//! its length (plus a constant), and decompressing never more than the declared limit. Alone
//! in its file so that no other test's allocations are counted.
mod common;

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use common::*;
use klong_ipc::{codec, compress};

struct Counting;
static USED: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let now = USED.fetch_add(l.size(), Ordering::Relaxed) + l.size();
        PEAK.fetch_max(now, Ordering::Relaxed);
        System.alloc(l)
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        USED.fetch_sub(l.size(), Ordering::Relaxed);
        System.dealloc(p, l)
    }
}

#[global_allocator]
static A: Counting = Counting;

/// Peak heap growth while `f` runs.
fn growth(f: impl FnOnce()) -> usize {
    let base = USED.load(Ordering::Relaxed);
    PEAK.store(base, Ordering::Relaxed);
    f();
    PEAK.load(Ordering::Relaxed).saturating_sub(base)
}

/// Bytes of heap a decoder may use per byte of message. The worst honest cases: a symbol
/// vector of empty symbols (1 byte each, a 16-byte value each), a boolean vector (1 byte to
/// an 8-byte integer), a dictionary (its keys and values as items, then as the dictionary).
const PER_BYTE: usize = 72;
const FIXED: usize = 16 * 1024;

#[test]
fn decoding_is_bounded_by_the_message() {
    let seeds = vector_messages();
    let mut worst = 0.0f64;
    for i in 0..30000u64 {
        let mut r = Rng::new(0xb0b + i);
        let mut b = match r.below(5) {
            0 => {
                let n = r.below(64);
                r.bytes(n)
            }
            // the densest things a message can hold
            1 => {
                let n = r.below(3000);
                let mut b = vec![[1u8, 11, 0, 10][r.below(4)], 0];
                b.extend_from_slice(&(n as u32).to_le_bytes());
                let fill = if b[0] == 0 { 0x65 } else { 0 };
                b.extend(std::iter::repeat(fill).take(if b[0] == 0 { 2 * n } else { n }));
                b
            }
            2 => {
                let n = r.below(1000);
                let mut b = vec![99u8, 1, 0];
                b.extend_from_slice(&(n as u32).to_le_bytes());
                b.extend((0..n).map(|k| k as u8));
                b.extend_from_slice(&[11, 0]);
                b.extend_from_slice(&(n as u32).to_le_bytes());
                b.extend(std::iter::repeat(0).take(n));
                b
            }
            _ => {
                let m = &seeds[r.below(seeds.len())];
                m[m.len().min(8)..].to_vec()
            }
        };
        if r.below(3) != 0 {
            mutate(&mut r, &mut b);
        }
        let mut ip = interp();
        let used = growth(|| {
            let _ = codec::decode(&b, true, &mut ip.syms);
        });
        assert!(used <= PER_BYTE * b.len() + FIXED, "seed {}: {} bytes of heap for a message of {}", i, used, b.len());
        if b.len() > 256 {
            worst = worst.max(used as f64 / b.len() as f64);
        }
    }
    println!("decoder memory: at most {:.1} bytes of heap per message byte (messages over 256 bytes)", worst);

    // the decompressor: never more than the limit, whatever the stream declares
    for i in 0..20000u64 {
        let mut r = Rng::new(0xd00d + i);
        let mut b = unhex("0100010025000000be040000000a00b004000061625503ff6203ff6203ff6203ff620103a4");
        mutate(&mut r, &mut b);
        let n = b.len() as u32;
        if b.len() >= 12 {
            b[4..8].copy_from_slice(&n.to_le_bytes());
            if r.below(2) == 0 {
                b[8..12].copy_from_slice(&[0x7fff_ffffu32, 1 << 28, 1 << 20, 5000][r.below(4)].to_le_bytes());
            }
        }
        let limit = 4096;
        let used = growth(|| {
            let _ = compress::decompress(&b, limit);
        });
        assert!(used <= limit + 64, "seed {}: {} bytes for a limit of {}", i, used, limit);
    }
}
