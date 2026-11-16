//! Property tests and fuzzing of the codec, the decompressor and the session with a seeded
//! generator (no proptest dependency: the workspace has none, and a seed reproduces a failure
//! exactly). Random bytes and mutations of valid messages: nothing may panic or hang. The
//! memory bound has its own file (`membound.rs`). `KLONG_FUZZ_ITERS` multiplies the work.
mod common;

use common::*;
use klong_core::Value;
use klong_ipc::codec::{self, DecodeError};
use klong_ipc::compress;
use klong_ipc::session::{Config, Event, Kind, Session};

fn scale() -> usize {
    std::env::var("KLONG_FUZZ_ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(1)
}

#[test]
fn decode_of_encode_is_the_value() {
    let mut ip = interp();
    for seed in 0..4000 * scale() as u64 {
        let mut r = Rng::new(seed + 1);
        let v = random_value(&mut ip, &mut r, 0);
        let mut bytes = Vec::new();
        codec::encode(&ip.syms, &v, &mut bytes).unwrap();
        match codec::decode(&bytes, true, &mut ip.syms) {
            Ok(back) => assert!(same(&v, &back), "seed {}: {} came back as {}", seed, show(&ip, &v), show(&ip, &back)),
            Err(e) => panic!("seed {}: {} does not decode: {:?}", seed, show(&ip, &v), e),
        }
        // and encoding what came back gives the same bytes
        let back = codec::decode(&bytes, true, &mut ip.syms).ok().unwrap();
        let mut again = Vec::new();
        codec::encode(&ip.syms, &back, &mut again).unwrap();
        assert_eq!(bytes, again, "seed {}", seed);
    }
}

#[test]
fn decoder_survives_random_and_mutated_bytes() {
    let seeds = vector_messages();
    let mut ip = interp();
    let (mut ok, mut protocol, mut value) = (0u32, 0u32, 0u32);
    for i in 0..60000 * scale() as u64 {
        let mut r = Rng::new(0x5eed + i);
        let mut b = match r.below(4) {
            0 => {
                let n = r.below(48);
                r.bytes(n)
            }
            1 => {
                let v = random_value(&mut ip, &mut r, 0);
                let mut out = Vec::new();
                codec::encode(&ip.syms, &v, &mut out).unwrap();
                out
            }
            _ => {
                let m = &seeds[r.below(seeds.len())];
                m[m.len().min(8)..].to_vec()
            }
        };
        if r.below(8) != 0 {
            mutate(&mut r, &mut b);
        }
        match codec::decode(&b, r.below(4) != 0, &mut ip.syms) {
            Ok(v) => {
                ok += 1;
                // what decodes must encode again (or be refused as a value, never panic)
                let mut out = Vec::new();
                let _ = codec::encode(&ip.syms, &v, &mut out);
            }
            Err(DecodeError::Protocol(_)) => protocol += 1,
            Err(DecodeError::Value(_)) => value += 1,
        }
        if i % 5000 == 0 {
            ip = interp(); // (the symbol table only grows)
        }
    }
    println!("decoder fuzz: {} decoded, {} protocol errors, {} value errors", ok, protocol, value);
    assert!(ok > 0 && protocol > 0);
}

#[test]
fn deep_nesting_is_refused_not_overflowed() {
    let mut ip = interp();
    for depth in [63usize, 64, 65, 1000, 100_000] {
        let mut b = Vec::new();
        for _ in 0..depth {
            b.extend_from_slice(&[0, 0, 1, 0, 0, 0]);
        }
        b.extend_from_slice(&[0x65, 0]);
        let r = codec::decode(&b, true, &mut ip.syms);
        assert_eq!(r.is_ok(), depth <= 64, "depth {}", depth);
        // the same through dictionaries, tables, projections and adverbs
        for tag in [&[99u8, 0, 0, 0, 0, 0, 0][..], &[106u8][..], &[104u8, 1, 0, 0, 0][..], &[98u8, 0][..]] {
            let mut b = Vec::new();
            for _ in 0..depth {
                b.extend_from_slice(tag);
            }
            assert!(depth <= 64 || codec::decode(&b, true, &mut ip.syms).is_err());
        }
    }
}

#[test]
fn decompressor_survives_random_and_mutated_streams() {
    let good = unhex("0100010025000000be040000000a00b004000061625503ff6203ff6203ff6203ff620103a4");
    let plain = compress::decompress(&good, 1 << 20).ok().unwrap();
    assert_eq!(plain.len(), 1214);
    assert_eq!(&plain[..14], &unhex("01000000be0400000a00b0040000")[..]);
    assert!(plain[14..].chunks(2).all(|c| c == b"ab"));
    let (mut fine, mut bad) = (0u32, 0u32);
    for i in 0..40000 * scale() as u64 {
        let mut r = Rng::new(0xc0de + i);
        let mut b = if r.below(3) == 0 {
            let n = 12 + r.below(40);
            let mut b = r.bytes(n);
            b[0] = r.below(2) as u8;
            b[2] = 1;
            let n = b.len() as u32;
            let le = b[0] == 1;
            b[4..8].copy_from_slice(&if le { n.to_le_bytes() } else { n.to_be_bytes() });
            b
        } else {
            good.clone()
        };
        mutate(&mut r, &mut b);
        if r.below(2) == 0 && b.len() >= 8 {
            // keep the length field right so that the stream itself is reached
            let n = b.len() as u32;
            let le = b[0] == 1;
            b[4..8].copy_from_slice(&if le { n.to_le_bytes() } else { n.to_be_bytes() });
        }
        let limit = 1 << 16;
        match compress::decompress(&b, limit) {
            Ok(out) => {
                fine += 1;
                assert!(out.len() <= limit);
            }
            Err(_) => bad += 1,
        }
    }
    println!("decompressor fuzz: {} decompressed, {} refused", fine, bad);
    assert!(fine > 0 && bad > 0);
}

#[cfg(feature = "compress")]
#[test]
fn decompress_of_compress_is_the_message() {
    // the worked example of the reference (ref-kdb-ipc.md 5.5), byte for byte
    let mut msg = unhex("01000000be0400000a00b0040000");
    for _ in 0..600 {
        msg.extend_from_slice(b"ab");
    }
    let z = compress::compress(&msg, 0).unwrap();
    assert_eq!(z, unhex("0100010025000000be040000000a00b004000061625503ff6203ff6203ff6203ff620103a4"));
    let mut compressed = 0;
    for i in 0..3000 * scale() as u64 {
        let mut r = Rng::new(0xabc + i);
        let n = 20 + r.below(6000);
        let wide = r.below(2) == 0;
        let alphabet = 1 + r.below(if wide { 256 } else { 4 });
        let mut m = vec![1u8, r.below(3) as u8, 0, 0, 0, 0, 0, 0];
        let run = 1 + r.below(40);
        while m.len() < n {
            let b = r.below(alphabet) as u8;
            for _ in 0..1 + r.below(run) {
                m.push(b);
            }
        }
        let len = m.len() as u32;
        m[4..8].copy_from_slice(&len.to_le_bytes());
        if let Some(z) = compress::compress(&m, if r.below(2) == 0 { 0 } else { 2000 }) {
            compressed += 1;
            assert!(z.len() * 2 <= m.len() + 1, "seed {}", i);
            let back = compress::decompress(&z, 1 << 20).ok().unwrap();
            assert_eq!(back, m, "seed {}", i);
        }
    }
    assert!(compressed > 100);
}

fn ready(server: bool, ip: &mut klong_core::Interp) -> Session {
    if server {
        let mut s = Session::server(Config::default());
        s.feed(b"u:p\x03\x00").unwrap();
        assert!(matches!(s.next_event(&mut ip.syms), Some(Event::Handshake { .. })));
        s.accept().unwrap();
        s
    } else {
        let mut s = Session::client(Config::default());
        s.hello("", "").unwrap();
        s.feed(&[3]).unwrap();
        assert!(matches!(s.next_event(&mut ip.syms), Some(Event::Accepted { .. })));
        s
    }
}

#[test]
fn session_survives_random_and_mutated_streams() {
    let seeds = vector_messages();
    let mut ip = interp();
    let (mut messages, mut failures, mut closed) = (0u32, 0u32, 0u32);
    for i in 0..20000 * scale() as u64 {
        let mut r = Rng::new(0xfeed + i);
        // a stream of several messages, some damaged, fed in pieces of random size
        let mut stream = Vec::new();
        for _ in 0..1 + r.below(4) {
            let mut m = seeds[r.below(seeds.len())].clone();
            if r.below(3) == 0 {
                mutate(&mut r, &mut m);
            }
            stream.extend(m);
        }
        let mut s = match r.below(8) {
            0 => Session::server(Config::default()), // the stream meets the handshake
            1 => {
                let mut c = Session::client(Config::default());
                if r.below(2) == 0 {
                    c.hello("user", "pw").unwrap();
                }
                c
            }
            n => ready(n % 2 == 0, &mut ip),
        };
        let small = Config { max_message: 300, ..Config::default() };
        if r.below(6) == 0 {
            s = Session::server(small);
        }
        let mut outstanding = 0;
        let mut events = 0usize;
        let mut pos = 0;
        while pos < stream.len() {
            let n = (1 + r.below(40)).min(stream.len() - pos);
            if s.is_ready() && r.below(5) == 0 {
                if s.send_sync(&ip.syms, &Value::Int(1), r.below(2) == 0, Some(5), false).is_ok() {
                    outstanding += 1;
                }
            }
            s.feed(&stream[pos..pos + n]).unwrap();
            pos += n;
            while let Some(ev) = s.next_event(&mut ip.syms) {
                events += 1;
                assert!(events <= stream.len() + 8, "seed {}: more events than bytes", i);
                match ev {
                    Event::Handshake { .. } => {
                        if r.below(4) == 0 {
                            s.reject();
                        } else {
                            s.accept().unwrap();
                        }
                    }
                    Event::Message { kind, pending, .. } => {
                        messages += 1;
                        assert_eq!(kind == Kind::Response, pending.is_some());
                        if kind != Kind::Response && r.below(2) == 0 {
                            // serve it, sometimes badly
                            if let Ok(mut svc) = s.begin_service(kind) {
                                if kind == Kind::Sync {
                                    let _ = s.send_error(&mut svc, "type: no");
                                    let _ = s.send_error(&mut svc, "twice");
                                }
                                let _ = s.end_service(svc);
                            } else {
                                let _ = s.refuse(kind);
                            }
                        }
                    }
                    Event::Failure { pending, .. } => {
                        failures += 1;
                        assert!(pending.len() <= outstanding);
                        assert!(s.is_closed());
                    }
                    _ => {}
                }
            }
            assert!(s.buffered() <= 300.max(stream.len()), "seed {}", i);
        }
        if r.below(2) == 0 {
            if s.feed_eof().is_some() {
                closed += 1;
            }
            assert!(s.is_closed());
        } else {
            let _ = s.check_deadlines(r.below(10) as u64);
        }
        assert!(s.next_event(&mut ip.syms).is_none() || !s.is_closed());
        if i % 4000 == 0 {
            ip = interp();
        }
    }
    println!("session fuzz: {} messages, {} failures, {} closed at end of stream", messages, failures, closed);
    assert!(messages > 1000 && failures > 1000);
}
