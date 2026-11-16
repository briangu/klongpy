//! The serial framing and the packet reliability layer over a simulated link with a virtual
//! clock: loss, duplication, reordering, corruption (which serial framing turns into loss),
//! a dead link, a restarted peer. Seeded: every run sees the same faults.
mod common;

use common::*;
use klong_core::Value;
use klong_ipc::rel::{self, Rel};
use klong_ipc::serial::{self, Deframer};
use klong_ipc::session::{Config as SessionConfig, Event, Kind, Session};

// ----- framing ---------------------------------------------------------------------------

#[test]
fn crc_and_cobs_check_values() {
    assert_eq!(serial::crc16(b"123456789"), 0x29b1);
    assert_eq!(serial::crc16(b""), 0xffff);
    let cases: &[(&[u8], &[u8])] = &[
        (&[0x00], &[0x01, 0x01]),
        (&[0x00, 0x00], &[0x01, 0x01, 0x01]),
        (&[0x11, 0x22, 0x00, 0x33], &[0x03, 0x11, 0x22, 0x02, 0x33]),
        (&[0x11, 0x22, 0x33, 0x44], &[0x05, 0x11, 0x22, 0x33, 0x44]),
        (&[0x11, 0x00, 0x00, 0x00], &[0x02, 0x11, 0x01, 0x01, 0x01]),
        (&[], &[0x01]),
    ];
    for (plain, coded) in cases {
        let mut out = Vec::new();
        assert!(serial::cobs_encode(plain, &mut out));
        assert_eq!(&out, coded, "encoding {:02x?}", plain);
        let mut back = Vec::new();
        assert!(serial::cobs_decode(coded, &mut back));
        assert_eq!(&back, plain);
    }
    // 254 non-zero bytes: FF and the bytes; a decoder accepts the trailing 01 of simple encoders too
    let block: Vec<u8> = (1..=254u8).collect();
    let mut out = Vec::new();
    serial::cobs_encode(&block, &mut out);
    assert_eq!(out.len(), 255);
    assert_eq!(out[0], 0xff);
    for with_01 in [false, true] {
        let mut coded = out.clone();
        if with_01 {
            coded.push(0x01);
        }
        let mut back = Vec::new();
        assert!(serial::cobs_decode(&coded, &mut back));
        assert_eq!(back, block);
    }
    // 255 bytes 01..FF -> FF 01..FE 02 FF
    let long: Vec<u8> = (1..=255u8).collect();
    let mut out = Vec::new();
    serial::cobs_encode(&long, &mut out);
    assert_eq!(out.len(), 257);
    assert_eq!(&out[255..], &[0x02, 0xff]);
    // the frame on the wire: 11 22 00 33, its CRC, the delimiter
    let (mut frame, mut scratch) = (Vec::new(), Vec::new());
    assert!(serial::encode_frame(&[0x11, 0x22, 0x00, 0x33], &mut frame, &mut scratch));
    assert_eq!(*frame.last().unwrap(), 0);
    assert!(!frame[..frame.len() - 1].contains(&0));
    // malformed: a zero inside, a block that runs past the end
    let mut sink = Vec::new();
    assert!(!serial::cobs_decode(&[0x03, 0x11], &mut sink));
    assert!(!serial::cobs_decode(&[0x02, 0x00], &mut sink));
    assert!(!serial::cobs_decode(&[0x00], &mut sink));
}

#[test]
fn cobs_round_trips_any_bytes() {
    for seed in 0..3000u64 {
        let mut r = Rng::new(seed);
        let n = r.below(700);
        let zeros = r.below(4);
        let data: Vec<u8> = (0..n).map(|_| if zeros > 0 && r.below(zeros * 8) == 0 { 0 } else { 1 + r.below(255) as u8 }).collect();
        let mut coded = Vec::new();
        assert!(serial::cobs_encode(&data, &mut coded));
        assert!(!coded.contains(&0));
        assert!(coded.len() <= serial::cobs_max(data.len()));
        let mut back = Vec::new();
        assert!(serial::cobs_decode(&coded, &mut back), "seed {}", seed);
        assert_eq!(back, data, "seed {}", seed);
    }
}

/// Damage a byte stream as a bad serial line does. Returns how many faults were injected.
fn noisy(r: &mut Rng, bytes: &[u8], flip: f64, drop: f64, insert: f64, out: &mut Vec<u8>) -> usize {
    let mut faults = 0;
    for &b in bytes {
        if r.chance(drop) {
            faults += 1;
            continue;
        }
        if r.chance(insert) {
            faults += 1;
            out.push(r.next() as u8);
        }
        if r.chance(flip) {
            faults += 1;
            out.push(b ^ (1 << r.below(8)));
        } else {
            out.push(b);
        }
    }
    faults
}

#[test]
fn deframer_delivers_only_what_was_sent() {
    let (mut delivered, mut sent_total, mut faults) = (0usize, 0usize, 0usize);
    for seed in 0..300u64 {
        let mut r = Rng::new(0x5e41a1 + seed);
        let mtu = [4, 16, 240][r.below(3)];
        let rate = [0.0, 0.0005, 0.005, 0.03][r.below(4)];
        let mut d = Deframer::new(mtu);
        let mut sent: Vec<Vec<u8>> = Vec::new();
        let mut wire = Vec::new();
        let mut scratch = Vec::new();
        for _ in 0..200 {
            let n = r.below(mtu + 1);
            let p = r.bytes(n);
            let mut frame = Vec::new();
            assert!(serial::encode_frame(&p, &mut frame, &mut scratch));
            assert!(frame.len() <= mtu + 5);
            faults += noisy(&mut r, &frame, rate, rate, rate / 4.0, &mut wire);
            sent.push(p);
        }
        sent_total += sent.len();
        // in pieces of any size
        let mut next = 0;
        let mut pos = 0;
        while pos < wire.len() {
            let end = (pos + 1 + r.below(50)).min(wire.len());
            let mut piece = &wire[pos..end];
            while !piece.is_empty() {
                let (used, pkt) = d.push(piece);
                if let Some(p) = pkt {
                    // in order, and exactly a packet that was sent
                    match sent[next..].iter().position(|s| s.as_slice() == p) {
                        Some(k) => next += k + 1,
                        None => panic!("seed {}: a packet that was never sent was delivered: {:02x?}", seed, p),
                    }
                    delivered += 1;
                }
                piece = &piece[used..];
            }
            pos = end;
        }
        if rate == 0.0 {
            assert_eq!(next, sent.len(), "seed {}: a clean line delivers everything", seed);
        }
    }
    println!("deframer: {} of {} frames delivered through {} injected byte faults, none altered", delivered, sent_total, faults);
}

// ----- a simulated packet link -----------------------------------------------------------

#[derive(Clone, Copy)]
struct Faults {
    loss: f64,
    dup: f64,
    reorder: f64,
    /// Per byte, applied to the serial frame (only with `serial`).
    flip: f64,
    serial: bool,
}

const CLEAN: Faults = Faults { loss: 0.0, dup: 0.0, reorder: 0.0, flip: 0.0, serial: false };

/// One direction of the link: packets in flight with their arrival time.
struct Wire {
    q: Vec<(u64, Vec<u8>)>,
    deframer: Deframer,
    sent: usize,
    lost: usize,
}

struct Sim {
    r: Rng,
    now: u64,
    a: Rel,
    b: Rel,
    ab: Wire,
    ba: Wire,
    f: Faults,
    mtu: usize,
}

impl Sim {
    fn new(seed: u64, cfg: rel::Config, f: Faults) -> Sim {
        let mut r = Rng::new(seed);
        let ida = [r.next() as u8, r.next() as u8, r.next() as u8];
        let idb = [r.next() as u8, r.next() as u8, r.next() as u8];
        let w = || Wire { q: Vec::new(), deframer: Deframer::new(cfg.mtu), sent: 0, lost: 0 };
        Sim { r, now: 0, a: Rel::client(cfg, ida, 0), b: Rel::server(cfg, idb, 0), ab: w(), ba: w(), f, mtu: cfg.mtu }
    }

    fn transmit(r: &mut Rng, f: &Faults, now: u64, w: &mut Wire, pkt: &[u8]) {
        w.sent += 1;
        if r.chance(f.loss) {
            w.lost += 1;
            return;
        }
        let copies = if r.chance(f.dup) { 2 } else { 1 };
        for _ in 0..copies {
            let delay = 1 + if r.chance(f.reorder) { 1 + r.below(8) as u64 } else { 0 };
            let bytes = if f.serial {
                // the bytes of the frame, damaged on the line
                let (mut frame, mut scratch, mut out) = (Vec::new(), Vec::new(), Vec::new());
                serial::encode_frame(pkt, &mut frame, &mut scratch);
                noisy(r, &frame, f.flip, f.flip / 2.0, f.flip / 8.0, &mut out);
                out
            } else {
                pkt.to_vec()
            };
            w.q.push((now + delay, bytes));
        }
    }

    fn deliver(now: u64, serial_framing: bool, w: &mut Wire, to: &mut Rel) {
        let mut i = 0;
        while i < w.q.len() {
            if w.q[i].0 <= now {
                let (_, bytes) = w.q.remove(i);
                if serial_framing {
                    let mut piece = &bytes[..];
                    while !piece.is_empty() {
                        let (used, pkt) = w.deframer.push(piece);
                        if let Some(p) = pkt {
                            to.on_packet(p, now);
                        }
                        piece = &piece[used..];
                    }
                } else {
                    to.on_packet(&bytes, now);
                }
            } else {
                i += 1;
            }
        }
    }

    /// One millisecond of the link.
    fn step(&mut self) {
        let mut buf = vec![0u8; self.mtu];
        // A link has a rate. With reordering it is 4 packets per millisecond each way, so that
        // a packet held back (up to 9 ms here) is overtaken by at most 36 others: the 6-bit
        // sequence number tells a late packet from a new one only within 48 (64 less a window).
        let rate = if self.f.reorder > 0.0 { 4 } else { 64 };
        for _ in 0..rate {
            match self.a.poll_packet(self.now, &mut buf) {
                Some(n) => Self::transmit(&mut self.r, &self.f, self.now, &mut self.ab, &buf[..n]),
                None => break,
            }
        }
        for _ in 0..rate {
            match self.b.poll_packet(self.now, &mut buf) {
                Some(n) => Self::transmit(&mut self.r, &self.f, self.now, &mut self.ba, &buf[..n]),
                None => break,
            }
        }
        self.now += 1;
        Self::deliver(self.now, self.f.serial, &mut self.ab, &mut self.b);
        Self::deliver(self.now, self.f.serial, &mut self.ba, &mut self.a);
    }

    /// Move `a_data` from a to b and `b_data` from b to a; what each side received. Stops
    /// when both streams are through, when a side is done, or after `limit_ms`.
    fn exchange(&mut self, a_data: &[u8], b_data: &[u8], limit_ms: u64) -> (Vec<u8>, Vec<u8>) {
        let (mut ia, mut ib) = (0, 0);
        let (mut got_a, mut got_b) = (Vec::new(), Vec::new());
        let mut chunk = [0u8; 512];
        let end = self.now + limit_ms;
        while self.now < end {
            ia += self.a.write(&a_data[ia..]);
            ib += self.b.write(&b_data[ib..]);
            self.step();
            loop {
                let n = self.a.read(&mut chunk);
                if n == 0 {
                    break;
                }
                got_a.extend_from_slice(&chunk[..n]);
            }
            loop {
                let n = self.b.read(&mut chunk);
                if n == 0 {
                    break;
                }
                got_b.extend_from_slice(&chunk[..n]);
            }
            if got_b.len() == a_data.len() && got_a.len() == b_data.len() && self.a.unsent() == 0 && self.b.unsent() == 0 {
                break;
            }
            if self.a.is_done() || self.b.is_done() {
                break;
            }
        }
        (got_a, got_b)
    }
}

fn cfg(mtu: usize) -> rel::Config {
    rel::Config::new(mtu, false, 50)
}

#[test]
fn clean_link_carries_both_directions_at_every_mtu() {
    for mtu in [4usize, 5, 16, 64, 240, 1400] {
        let mut s = Sim::new(mtu as u64, cfg(mtu), CLEAN);
        let mut r = Rng::new(7 + mtu as u64);
        let (x, y) = (r.bytes(20_000), r.bytes(13_337));
        let (ga, gb) = s.exchange(&x, &y, 600_000);
        assert!(gb == x && ga == y, "mtu {}", mtu);
        assert_eq!(s.a.retransmissions + s.b.retransmissions, 0, "mtu {}: nothing is sent twice on a clean link", mtu);
        assert!(s.a.is_open() && s.b.is_open());
    }
}

#[test]
fn lossy_link_delivers_the_stream_intact() {
    let mut table = Vec::new();
    for (k, &loss) in [0.01, 0.05, 0.10, 0.20, 0.30].iter().enumerate() {
        for (m, &mtu) in [4usize, 32, 240].iter().enumerate() {
            let seed = 1000 + (k * 10 + m) as u64;
            let f = Faults { loss, dup: loss, reorder: loss, flip: 0.0, serial: false };
            // (at 30% loss eight doublings of the timeout are not always enough for one packet
            // in thousands: the budget is configurable, and this is what it is for)
            let mut c = cfg(mtu);
            if loss > 0.25 {
                c.retries = 16;
            }
            let mut s = Sim::new(seed, c, f);
            let mut r = Rng::new(seed);
            let n = if mtu == 4 { 6_000 } else { 40_000 };
            let (x, y) = (r.bytes(n), r.bytes(n / 3));
            let (ga, gb) = s.exchange(&x, &y, 3_600_000);
            assert!(gb == x && ga == y, "loss {} mtu {}: a {:?} b {:?}, {} of {} and {} of {} bytes", loss, mtu, s.a.state(), s.b.state(), gb.len(), x.len(), ga.len(), y.len());
            table.push(format!("loss/dup/reorder {:>3.0}% mtu {:>3}: {:>6} bytes each way intact, {} packets, {} lost, {} retransmission timeouts, {} ms", loss * 100.0, mtu, n + n / 3, s.ab.sent + s.ba.sent, s.ab.lost + s.ba.lost, s.a.retransmissions + s.b.retransmissions, s.now));
        }
    }
    println!("{}", table.join("\n"));
}

#[test]
fn random_fault_mixes_never_corrupt_the_stream() {
    // property: whatever the mix, what arrives is a prefix of what was sent, and with any
    // loss below 25% the whole of it arrives
    for seed in 0..120u64 {
        let mut r = Rng::new(0xfa17 + seed);
        let mtu = 4 + r.below(60);
        let f = Faults { loss: r.below(25) as f64 / 100.0, dup: r.below(30) as f64 / 100.0, reorder: r.below(50) as f64 / 100.0, flip: 0.0, serial: false };
        let mut c = cfg(mtu);
        c.retries = 12;
        let mut s = Sim::new(seed, c, f);
        let nx = r.below(5000);
        let ny = r.below(5000);
        let (x, y) = (r.bytes(nx), r.bytes(ny));
        let (ga, gb) = s.exchange(&x, &y, 3_600_000);
        assert!(x.starts_with(&gb) && y.starts_with(&ga), "seed {}: bytes altered", seed);
        assert!(gb.len() == x.len() && ga.len() == y.len(), "seed {} (mtu {}, loss {}): {:?} {:?}", seed, mtu, f.loss, s.a.state(), s.b.state());
    }
}

#[test]
fn serial_line_with_flips_and_drops_delivers_intact() {
    let mut table = Vec::new();
    for (k, &flip) in [0.0002, 0.002, 0.01].iter().enumerate() {
        for &mtu in &[32usize, 240] {
            if flip * mtu as f64 > 1.0 {
                continue; // (one byte in a hundred damaged: no 245-byte frame survives)
            }
            let f = Faults { loss: 0.0, dup: 0.0, reorder: 0.0, flip, serial: true };
            let mut c = cfg(mtu);
            c.retries = 12;
            let mut s = Sim::new(50 + k as u64, c, f);
            let mut r = Rng::new(900 + k as u64 + mtu as u64);
            let (x, y) = (r.bytes(30_000), r.bytes(9_000));
            let (ga, gb) = s.exchange(&x, &y, 3_600_000);
            assert!(gb == x && ga == y, "flip {} mtu {}: {:?} {:?}", flip, mtu, s.a.state(), s.b.state());
            table.push(format!("byte faults {:.2}% mtu {:>3}: 39000 bytes intact, {} frames dropped by CRC or COBS, {} retransmission timeouts", flip * 175.0, mtu, s.ab.deframer.dropped + s.ba.deframer.dropped, s.a.retransmissions + s.b.retransmissions));
        }
    }
    println!("{}", table.join("\n"));
}

#[test]
fn dead_link_fails_within_the_retry_budget() {
    // never connected: the SYN is retried 8 times, 50 ms doubling: 50 * (2^9 - 1) ms
    let dead = Faults { loss: 1.0, ..CLEAN };
    let mut s = Sim::new(1, cfg(16), dead);
    s.exchange(b"hello", b"", 100_000);
    assert_eq!(s.a.state(), rel::State::Failed(rel::Error::Timeout));
    assert_eq!(s.a.retransmissions, 8);
    assert!(s.now >= 25_550 && s.now <= 25_560, "failed after {} ms", s.now);

    // dies in the middle of a transfer
    let mut s = Sim::new(2, cfg(16), CLEAN);
    let data = Rng::new(3).bytes(200_000);
    let (_, got) = s.exchange(&data, b"", 40);
    assert!(!got.is_empty() && got.len() < data.len());
    s.f = dead;
    let t0 = s.now;
    let (_, more) = s.exchange(&data[..0], b"", 100_000);
    assert_eq!(s.a.state(), rel::State::Failed(rel::Error::Timeout));
    assert!(s.now - t0 <= 25_600, "failed after {} ms", s.now - t0);
    assert!(data.starts_with(&[got, more].concat()));

    // fail-fast: retries = 0, one lost window fails the session at the first timeout
    let mut c = cfg(16);
    c.retries = 0;
    let mut s = Sim::new(4, c, CLEAN);
    s.exchange(b"x", b"", 20);
    assert!(s.a.is_open());
    s.f = dead;
    let t0 = s.now;
    s.exchange(&data[..1000], b"", 10_000);
    assert_eq!(s.a.state(), rel::State::Failed(rel::Error::Timeout));
    assert!(s.now - t0 <= 52);
    // ... and the link recovers for the next session
    let mut s2 = Sim::new(5, c, CLEAN);
    let (_, got) = s2.exchange(&data[..1000], b"", 10_000);
    assert_eq!(got, &data[..1000]);
}

#[test]
fn peer_restart_is_detected() {
    // the client restarts in the middle of a transfer: the server sees a SYN with another id
    let mut s = Sim::new(11, cfg(32), CLEAN);
    let data = Rng::new(12).bytes(100_000);
    let (_, got) = s.exchange(&data, b"", 30);
    assert!(!got.is_empty() && got.len() < data.len());
    s.a = Rel::client(cfg(32), [9, 9, 9], s.now);
    s.exchange(b"again", b"", 2_000);
    assert_eq!(s.b.state(), rel::State::Failed(rel::Error::PeerRestarted));
    // a new server session on the same link then serves the retried SYN
    s.b = Rel::server(cfg(32), [7, 7, 7], s.now);
    let (_, got) = s.exchange(b"", b"", 5_000);
    assert_eq!(got, b"again");

    // the server restarts: it has no session for the client's data and answers RST
    let mut s = Sim::new(13, cfg(32), CLEAN);
    let (_, got) = s.exchange(&data, b"", 30);
    assert!(!got.is_empty());
    s.b = Rel::server(cfg(32), [1, 2, 3], s.now);
    s.exchange(&data[..0], b"", 5_000);
    assert_eq!(s.a.state(), rel::State::Failed(rel::Error::Reset));

    // close() tells the peer
    let mut s = Sim::new(14, cfg(32), CLEAN);
    s.exchange(b"x", b"y", 100);
    s.a.close();
    s.exchange(b"", b"", 100);
    assert_eq!(s.b.state(), rel::State::Failed(rel::Error::Reset));
}

#[test]
fn reliable_link_needs_no_acks_and_a_gap_is_fatal() {
    let c = rel::Config::new(64, true, 50);
    let mut s = Sim::new(21, c, CLEAN);
    let data = Rng::new(22).bytes(50_000);
    let (ga, gb) = s.exchange(&data, &data[..777], 60_000);
    assert!(gb == data && ga == data[..777]);
    // SYN each way, then only DATA: 50000/63 and 777/63 packets
    assert_eq!(s.ab.sent, 1 + 50_000usize.div_ceil(63));
    assert_eq!(s.ba.sent, 1 + 777usize.div_ceil(63));
    // a gap
    let mut s = Sim::new(23, c, CLEAN);
    s.exchange(b"abc", b"", 50);
    s.b.on_packet(&[5, b'x'], s.now); // DATA with sequence 5 where 1 is expected
    assert_eq!(s.b.state(), rel::State::Failed(rel::Error::Protocol));
}

#[test]
fn keepalive_keeps_an_idle_link_and_notices_a_dead_one() {
    let mut c = cfg(16);
    c.idle_ms = 200;
    let mut s = Sim::new(31, c, CLEAN);
    s.exchange(b"x", b"y", 5_000);
    for _ in 0..5_000 {
        s.step();
    }
    assert!(s.a.is_open() && s.b.is_open());
    assert!(s.ab.sent > 10, "keepalives were sent");
    s.f = Faults { loss: 1.0, ..CLEAN };
    let t0 = s.now;
    while s.a.is_open() && s.now - t0 < 60_000 {
        s.step();
    }
    assert_eq!(s.a.state(), rel::State::Failed(rel::Error::Timeout));
    // idle 200 ms, then 3 retries: 50 + 100 + 200 + 400 ms
    assert!(s.now - t0 <= 200 + 750 + 10, "noticed after {} ms", s.now - t0);
}

#[test]
fn a_slow_reader_holds_the_sender_back_without_loss() {
    let mut s = Sim::new(41, cfg(32), CLEAN);
    let data = Rng::new(42).bytes(60_000);
    let mut i = 0;
    let mut got = Vec::new();
    let mut chunk = [0u8; 100];
    // b reads 100 bytes every 4 ms: far slower than the link
    while got.len() < data.len() && s.now < 4_000_000 {
        i += s.a.write(&data[i..]);
        s.step();
        if s.now % 4 == 0 {
            let n = s.b.read(&mut chunk);
            got.extend_from_slice(&chunk[..n]);
        }
        assert!(!s.a.is_done() && !s.b.is_done(), "at {} ms: {:?} {:?}", s.now, s.a.state(), s.b.state());
    }
    assert!(got == data);
}

// ----- the whole stack: session over the reliability layer over a bad link -----------------

#[test]
fn a_100000_element_array_round_trips_at_mtu_4_with_5_percent_faults() {
    let f = Faults { loss: 0.05, dup: 0.05, reorder: 0.05, flip: 0.0, serial: false };
    let mut s = Sim::new(77, cfg(4), f);
    let mut ipa = interp();
    let mut ipb = interp();
    let mut client = Session::client(SessionConfig::default());
    let mut server = Session::server(SessionConfig::default());
    let array = Value::ints((0..100_000).map(|i| (i * 7 - 3) as klong_core::Int).collect());
    let mut a_out: Vec<u8> = client.hello("", "").unwrap();
    let mut b_out: Vec<u8> = Vec::new();
    let (mut ia, mut ib) = (0usize, 0usize);
    let mut chunk = [0u8; 512];
    let mut answer = None;
    let mut sent_request = false;
    while answer.is_none() && s.now < 360_000_000 {
        ia += s.a.write(&a_out[ia..]);
        ib += s.b.write(&b_out[ib..]);
        s.step();
        assert!(!s.a.is_done() && !s.b.is_done(), "{:?} {:?}", s.a.state(), s.b.state());
        loop {
            let n = s.b.read(&mut chunk);
            if n == 0 {
                break;
            }
            server.feed(&chunk[..n]).unwrap();
            while let Some(ev) = server.next_event(&mut ipb.syms) {
                match ev {
                    Event::Handshake { .. } => b_out.extend(server.accept().unwrap()),
                    Event::Message { kind: Kind::Sync, value, .. } => {
                        // the echo service
                        let mut svc = server.begin_service(Kind::Sync).unwrap();
                        b_out.extend(server.send_response(&mut svc, &ipb.syms, &value.unwrap(), false).unwrap());
                        assert!(server.end_service(svc).is_none());
                    }
                    _ => panic!("server: unexpected event"),
                }
            }
        }
        loop {
            let n = s.a.read(&mut chunk);
            if n == 0 {
                break;
            }
            client.feed(&chunk[..n]).unwrap();
            while let Some(ev) = client.next_event(&mut ipa.syms) {
                match ev {
                    Event::Accepted { .. } => {}
                    Event::Message { kind: Kind::Response, value, .. } => answer = Some(value.unwrap()),
                    _ => panic!("client: unexpected event"),
                }
            }
        }
        if client.is_ready() && !sent_request {
            sent_request = true;
            a_out.extend(client.send_sync(&ipa.syms, &array, true, None, false).unwrap().0);
        }
    }
    assert!(same(&answer.expect("an answer"), &array));
    println!(
        "100000-element array (800022 bytes each way) at mtu 4 with 5% loss, duplication and reordering: intact after {} packets ({} lost), {} retransmission timeouts, {:.0} s of link time at 4 packets/ms",
        s.ab.sent + s.ba.sent,
        s.ab.lost + s.ba.lost,
        s.a.retransmissions + s.b.retransmissions,
        s.now as f64 / 1000.0
    );
}

// ----- fuzzing the layer -----------------------------------------------------------------

#[test]
fn reliability_layer_survives_random_packets() {
    for seed in 0..4000u64 {
        let mut r = Rng::new(0x9e1 + seed);
        let mtu = 4 + r.below(30);
        let mut c = rel::Config::new(mtu, r.below(4) == 0, 1 + r.below(50) as u32);
        c.idle_ms = r.below(3) as u32 * 20;
        c.retries = r.below(9) as u8;
        let mut x = if r.below(2) == 0 { Rel::client(c, [1, 2, 3], 0) } else { Rel::server(c, [1, 2, 3], 0) };
        let mut now = 0u64;
        let mut buf = vec![0u8; mtu];
        let mut polls = 0usize;
        for _ in 0..400 {
            match r.below(6) {
                0 => {
                    // a plausible packet: right type bits, any sequence, any length (even too long)
                    let n = r.below(mtu + 4);
                    let mut p = r.bytes(n);
                    if !p.is_empty() && r.below(3) != 0 {
                        p[0] = ((r.below(4) as u8) << 6) | [0u8, 1, 2, 63, 17][r.below(5)];
                    }
                    x.on_packet(&p, now);
                }
                1 => {
                    let n = r.below(100);
                    let w = x.write(&r.bytes(n));
                    assert!(w <= n);
                }
                2 => {
                    let mut out = [0u8; 64];
                    let n = x.read(&mut out);
                    assert!(n <= 64);
                }
                3 => now += r.below(200) as u64,
                4 => x.on_packet(&[(2 << 6) | 1, 4, 5, 6], now), // a valid SYN
                _ => {
                    // drain: the layer never has an unbounded number of packets to send at once
                    let mut k = 0;
                    while let Some(n) = x.poll_packet(now, &mut buf) {
                        assert!(n >= 1 && n <= mtu, "seed {}", seed);
                        k += 1;
                        assert!(k <= 4 * rel::WINDOW + 8 + c.send_buffer, "seed {}: poll_packet does not stop", seed);
                    }
                    polls += k;
                }
            }
            assert!(x.unsent() <= c.send_buffer);
            if let Some(t) = x.next_wake() {
                assert!(t + 1_000_000 > now, "seed {}", seed);
            }
        }
        let _ = polls;
    }
}
