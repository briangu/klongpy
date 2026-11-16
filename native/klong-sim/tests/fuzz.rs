//! Ticket 104: random and mutated byte streams into the console state machine and the Intel
//! HEX parser. No panic, no hang, no memory without a bound, and the console answers
//! afterwards.
//!
//!     KLONG_FUZZ_STREAMS=20000 KLONG_FUZZ_SEED=7 cargo test --release -p klong-sim --test fuzz
//!
//! A seeded generator, not coverage-guided. The interpreter runs under a step limit and a
//! heap limit (this binary's global allocator is the counting one), so a random program ends.
use std::alloc::System;
use std::sync::Mutex;

use klong_console::crc::crc32;
use klong_console::{hex, ConsoleConfig};
use klong_core::Config;
use klong_rt::heap::Counting;
use klong_sim::{Sim, SimOptions};

#[global_allocator]
static HEAP: Counting<System> = Counting::new(System);

static ONE: Mutex<()> = Mutex::new(());

fn pressed() -> bool {
    HEAP.pressed()
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n.max(1) as u64) as usize
    }
    fn chance(&mut self, percent: usize) -> bool {
        self.below(100) < percent
    }
}

fn env(name: &str, default: u64) -> u64 {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn record(kind: u8, addr: u16, data: &[u8]) -> Vec<u8> {
    let mut line = [0u8; hex::MAX_LINE];
    let n = hex::write(kind, addr, data, &mut line);
    line[..n].to_vec()
}

fn random_bytes(rng: &mut Rng, n: usize) -> Vec<u8> {
    (0..n).map(|_| rng.next() as u8).collect()
}

/// A stream made of the things a console understands, in any order.
fn soup(rng: &mut Rng) -> Vec<u8> {
    const WORDS: &[&[u8]] = &[
        b"\x01", b"\x01", b"put ", b"hex ", b"run ", b"ver", b"f.kg ", b"a.bin ", b"0 ", b"3 ", b"17 ", b"99999999999 ", b"00000000", b"ffffffff ",
        b"\n", b"\n", b"\r", b"\r\n", b" ", b"\x03", b"\x04", b"\x7f", b"\x08", b"\x15", b"\t", b"\x1b", b"\x1b[A", b"\x1b[B", b"\x1b[C", b"\x1b[D",
        b"\x1b[H", b"\x1b[F", b"\x1b[3~", b"\x1b[1;5", b"\x1bO", b"[", b"]", b"(", b")", b"{", b"}", b":[", b":{", b"\"", b":\"", b".comment(\"X\")",
        b"X", b"1+2", b"!10", b"a::", b"a", b"{x}", b"'", b"/", b".p(1)", b".l(\"f.kg\")", b".x(0)", b".E(\"1\")", b".rl()", b".r()", b"{1}{x}:~0",
        b"!100000000", b"\xc3\xa9", b"\xe4\xb8\x96", b"\xff", b"\xc3", b":00000001FF", b":", b":0", b"0c", b";", b",",
    ];
    let mut out = Vec::new();
    for _ in 0..rng.below(60) {
        match rng.below(12) {
            0 => {
                // a well-formed command with its data, sometimes cut short or too long
                let n = rng.below(40);
                let data = random_bytes(rng, n);
                let text: &[u8] = [&b"1+1"[..], b".p(2);3", b"f::{x", b"", b"[1 2 3]@9"][rng.below(5)];
                let body = if rng.chance(50) { &data[..] } else { text };
                let sum = if rng.chance(85) { crc32(body) } else { rng.next() as u32 };
                let len = if rng.chance(85) { body.len() } else { rng.below(80) };
                match rng.below(3) {
                    0 => out.extend(format!("\x01put f{}.kg {} {:08x}\n", rng.below(3), len, sum).bytes()),
                    1 => out.extend(format!("\x01put b{}.bin {} {:08x}\n", rng.below(3), len, sum).bytes()),
                    _ => out.extend(format!("\x01run {} {:08x}\n", len, sum).bytes()),
                }
                out.extend(body);
            }
            1 => {
                out.extend(format!("\x01hex h{}.bin\n", rng.below(3)).bytes());
                for _ in 0..rng.below(8) {
                    let kind = [0u8, 0, 0, 0, 1, 2, 3, 4, 5, 6, 0xFF][rng.below(11)];
                    let n = match kind {
                        1 => 0,
                        2 | 4 => 2,
                        3 | 5 => 4,
                        _ => rng.below(40),
                    };
                    let n = if rng.chance(90) { n } else { rng.below(300) };
                    let data = random_bytes(rng, n);
                    let mut line = record(kind, rng.next() as u16, &data);
                    if rng.chance(15) && !line.is_empty() {
                        let at = rng.below(line.len());
                        line[at] = rng.next() as u8;
                    }
                    out.extend(line);
                    out.extend(if rng.chance(50) { &b"\n"[..] } else { b"\r\n" });
                }
                if rng.chance(70) {
                    out.extend(b":00000001FF\n");
                }
            }
            2 => {
                let n = rng.below(20);
                out.extend(random_bytes(rng, n));
            }
            _ => out.extend(WORDS[rng.below(WORDS.len())]),
        }
    }
    out
}

/// A session that works, with a few bytes changed, dropped, doubled or added.
fn mutated(rng: &mut Rng) -> Vec<u8> {
    let file: Vec<u8> = (0..100u8).collect();
    let mut s: Vec<u8> = Vec::new();
    s.extend(b"f::{x*\n2}\nf(21)\n1+\x1b[D\x7f2+\x1b[F1\n\x01ver\n");
    s.extend(format!("\x01put a.bin {} {:08x}\n", file.len(), crc32(&file)).bytes());
    s.extend(&file);
    s.extend(b"\x01hex b.bin\n");
    for (i, chunk) in file.chunks(16).enumerate() {
        s.extend(record(0, (i * 16) as u16, chunk));
        s.extend(b"\n");
    }
    s.extend(b":00000001FF\n");
    s.extend(format!("\x01run 9 {:08x}\n.p(1);2+2", crc32(b".p(1);2+2")).bytes());
    s.extend(b"[1 2\n\x033*3\n");
    for _ in 0..1 + rng.below(6) {
        let at = rng.below(s.len());
        match rng.below(5) {
            0 => s[at] = rng.next() as u8,
            1 => {
                s.remove(at);
            }
            2 => {
                let b = s[at];
                s.insert(at, b);
            }
            3 => s.insert(at, rng.next() as u8),
            _ => s[at] ^= 1 << rng.below(8),
        }
    }
    s
}

const HELD_MAX: usize = 256 + 512 + 1024 + 4096 + 2048;

#[test]
fn console_survives_random_and_mutated_streams() {
    let _one = ONE.lock().unwrap_or_else(|e| e.into_inner());
    let streams = env("KLONG_FUZZ_STREAMS", 1500);
    let mut rng = Rng(env("KLONG_FUZZ_SEED", 104).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    let mut bytes = 0usize;
    let (mut replies, mut errors, mut resets) = (0usize, 0usize, 0usize);
    for n in 0..streams {
        let stream = match n % 4 {
            0 => {
                let len = rng.below(600);
                random_bytes(&mut rng, len)
            }
            1 | 2 => soup(&mut rng),
            _ => mutated(&mut rng),
        };
        bytes += stream.len();
        if let Ok(path) = std::env::var("KLONG_FUZZ_TRACE") {
            // (finding a stream that ends the process: the last one written is it)
            let _ = std::fs::write(&path, &stream);
        }
        let dumb = rng.chance(30);
        let console = ConsoleConfig { max_line: 256, max_entry: 1024, max_run: 4096, max_file: 16 * 1024, history_lines: 8, history_bytes: 512, dumb, ..ConsoleConfig::default() };
        let config = Config { step_limit: Some(3000), mem_probe: Some(pressed), ..Config::default() };
        HEAP.set_limit(0, 0);
        let mut s = Sim::new(SimOptions { console, config, fs_bytes: if rng.chance(90) { Some(64 * 1024) } else { None }, crlf: rng.chance(20), mem: None });
        HEAP.set_limit(HEAP.used() + 2 * 1024 * 1024, 16 * 1024);
        s.wire.set_keep(16 * 1024);
        s.wire.set_burst(rng.below(40));
        // (what the board writes is counted and dropped: this test's memory is under the
        // board's heap limit too)
        let mut tally = |bytes: Vec<u8>| {
            for w in bytes.windows(3) {
                match w {
                    b"\x01ok" | b"\x01ac" => replies += 1,
                    b"\x01er" | b"\x01na" => errors += 1,
                    _ => {}
                }
            }
        };
        let mut rest = &stream[..];
        while !rest.is_empty() {
            let most = if rng.chance(50) { 3 } else { 200 };
            let n = (1 + rng.below(most)).min(rest.len());
            if rng.chance(10) {
                // some of it arrives while the board is busy evaluating
                s.wire.send_after_idles(1 + rng.below(3) as u64, &rest[..n]);
            } else {
                s.wire.send(&rest[..n]);
            }
            rest = &rest[n..];
            tally(s.run());
            if rng.chance(2) {
                s.wire.advance(5_000_001);
            }
            assert!(s.console.held_bytes() <= HELD_MAX, "stream {n}: the console holds {} bytes", s.console.held_bytes());
        }
        // whatever state it is in: a pause longer than the timeout, Ctrl-C, and it answers
        s.wire.advance(6_000_000);
        tally(s.run());
        s.wire.advance(6_000_000);
        tally(s.run());
        if s.wire.unread() > 0 {
            // input held back for an evaluation that never ran: it is not coming
            s.console.rt.shared.platform.borrow_mut().port.idle();
            s.console.rt.shared.platform.borrow_mut().port.idle();
            s.console.rt.shared.platform.borrow_mut().port.idle();
            tally(s.run());
            s.wire.advance(6_000_000);
            tally(s.run());
        }
        s.io(b"\x03");
        assert!(!s.console.busy(), "stream {n}: still busy");
        s.console.rt.interp.cfg.step_limit = None;
        let answer = s.io(format!("\x01run 3 {:08x}\n7*6", crc32(b"7*6")).as_bytes());
        assert!(answer.replace('\r', "").ends_with("\x01go\n42\n\x01ok value 2\nklong> "), "stream {n}: {answer:?} after {:?}", String::from_utf8_lossy(&stream));
        assert!(s.console.held_bytes() <= HELD_MAX);
        resets += usize::from(s.reset.get());
        // every file that is in place is whole: no temporary file is left once the board rests
        for name in s.files() {
            assert!(!name.ends_with(".tmp"), "stream {n}: {name} left behind");
        }
    }
    HEAP.set_limit(0, 0);
    println!("{streams} streams, {bytes} bytes: {replies} ok/ack, {errors} err/nak, {resets} resets asked for");
    assert!(replies > 0 && errors > 0);
}

#[test]
fn hex_parser_survives_and_never_accepts_a_damaged_record() {
    let _one = ONE.lock().unwrap_or_else(|e| e.into_inner());
    let rounds = env("KLONG_FUZZ_STREAMS", 1500) * 40;
    let mut rng = Rng(env("KLONG_FUZZ_SEED", 104).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    let mut buf = [0u8; hex::MAX_DATA];
    let mut out = [0u8; hex::MAX_LINE];
    let (mut good, mut bad) = (0u64, 0u64);
    for _ in 0..rounds {
        // random bytes, random hex digits after a colon: no panic; and what is accepted
        // writes back as the same record
        let n = rng.below(80);
        let mut line = random_bytes(&mut rng, n);
        if rng.chance(40) {
            // the right shape and the right check sum around random content: any type, any size
            let count = if rng.chance(50) { rng.below(6) } else { rng.below(256) };
            let mut bytes = vec![count as u8, rng.next() as u8, rng.next() as u8, if rng.chance(80) { rng.below(7) as u8 } else { rng.next() as u8 }];
            bytes.extend(random_bytes(&mut rng, count));
            let sum = bytes.iter().fold(0u8, |a, b| a.wrapping_add(*b));
            bytes.push(sum.wrapping_neg());
            let lower = rng.chance(50);
            line = vec![b':'];
            for b in bytes {
                line.extend(if lower { format!("{b:02x}") } else { format!("{b:02X}") }.bytes());
            }
        } else if rng.chance(70) {
            for b in line.iter_mut() {
                *b = b"0123456789abcdefABCDEF"[*b as usize % 22];
            }
            line.insert(0, b':');
        }
        match hex::parse(&line, &mut buf) {
            Ok(hex::Record::Data { addr, data }) => {
                let n = hex::write(0, addr, data, &mut out);
                assert!(out[..n].eq_ignore_ascii_case(&line), "{:?}", String::from_utf8_lossy(&line));
                good += 1;
            }
            Ok(_) => good += 1,
            Err(_) => bad += 1,
        }
        // a record that is right, then one digit of it replaced by another digit: refused
        let kind = [0u8, 0, 0, 1, 2, 3, 4, 5][rng.below(8)];
        let len = match kind {
            0 => rng.below(256),
            1 => 0,
            2 | 4 => 2,
            _ => 4,
        };
        let data = random_bytes(&mut rng, len);
        let addr = rng.next() as u16;
        let n = hex::write(kind, addr, &data, &mut out);
        let line = &mut out[..n];
        match hex::parse(line, &mut buf) {
            Ok(hex::Record::Data { addr: a, data: d }) => assert!(kind == 0 && a == addr && d == &data[..]),
            Ok(_) => assert!(kind != 0),
            Err(e) => panic!("{e:?} for {:?}", String::from_utf8_lossy(line)),
        }
        let at = 1 + rng.below(n - 1);
        let old = line[at];
        let new = loop {
            let c = b"0123456789ABCDEF"[rng.below(16)];
            if c != old {
                break c;
            }
        };
        line[at] = new;
        assert!(hex::parse(line, &mut buf).is_err(), "{:?} with digit {at} changed", String::from_utf8_lossy(line));
        // a digit lost, a digit doubled
        line[at] = old;
        let mut v = line.to_vec();
        if rng.chance(50) {
            v.remove(at);
        } else {
            v.insert(at, old);
        }
        assert!(hex::parse(&v, &mut buf).is_err());
    }
    println!("{rounds} rounds: {good} random lines accepted, {bad} refused");
}

/// One stream from a file, as the fuzz test runs it but all at once:
/// `KLONG_FUZZ_REPLAY=stream.bin cargo test -p klong-sim --test fuzz replay -- --nocapture`.
/// With `RUSTFLAGS=-Zoom=panic` (nightly) an allocation that is not fallible shows its
/// backtrace instead of ending the process.
#[test]
fn replay() {
    let path = match std::env::var("KLONG_FUZZ_REPLAY") {
        Ok(p) => p,
        Err(_) => return,
    };
    let _one = ONE.lock().unwrap_or_else(|e| e.into_inner());
    let stream = std::fs::read(path).expect("the stream file");
    let console = ConsoleConfig { max_line: 256, max_entry: 1024, max_run: 4096, max_file: 16 * 1024, history_lines: 8, history_bytes: 512, ..ConsoleConfig::default() };
    let config = Config { step_limit: Some(3000), mem_probe: Some(pressed), ..Config::default() };
    let mut s = Sim::new(SimOptions { console, config, fs_bytes: Some(64 * 1024), crlf: false, mem: None });
    HEAP.set_limit(HEAP.used() + 2 * 1024 * 1024, 16 * 1024);
    let out = s.io(&stream);
    HEAP.set_limit(0, 0);
    println!("{out}");
}
