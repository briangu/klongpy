//! The datagram spec, rule by rule: check values, frame layout, resync, every rule of
//! "Receiving", the sender's frames and timers, and the lane's duties.
use klong_dgram::crc::{crc16, crc32, Crc32};
use klong_dgram::dgram::*;
use klong_dgram::frame::*;
use klong_dgram::{Config, Datagrams, Lane, LaneConfig};

type Dg = Datagrams<1024, 17>;

fn dg(node: u8) -> Box<Dg> {
    let mut d = Box::new(Datagrams::new(Config::default()));
    d.set_node(node);
    d
}

/// The frames of a message as a sender with address `src` sends it.
fn frames(src: u8, dst: u8, id: u16, chan: u8, flags: u8, data: &[u8]) -> Vec<Frame> {
    let l = data.len() as u16;
    let mut v = vec![Frame::new(T_APP_DATA, src, dst, id, SUB_START, &[l as u8, (l >> 8) as u8, flags, chan])];
    for (i, c) in data.chunks(4).enumerate() {
        v.push(Frame::new(T_APP_DATA, src, dst, i as u16, SUB_DATA, c));
    }
    v.push(Frame::new(T_APP_DATA, src, dst, id, SUB_END, &crc32(data).to_le_bytes()));
    v
}

#[derive(Default)]
struct Got(Vec<(u8, u8, Vec<u8>)>);
impl Got {
    fn feed(&mut self, d: &mut Dg, fs: &[Frame]) {
        for f in fs {
            d.on_frame(f, 0, &mut |s, c, b| self.0.push((s, c, b.to_vec())));
        }
    }
}

fn acks(d: &mut Dg) -> Vec<Frame> {
    let mut v = vec![];
    while let Some(f) = d.poll_tx(0) {
        v.push(f);
    }
    v
}

#[test]
fn check_values() {
    assert_eq!(crc16(b"123456789"), 0x29b1);
    assert_eq!(crc32(b"123456789"), 0xcbf4_3926);
    assert_eq!(crc32(b""), 0);
    let mut c = Crc32::new();
    c.update(b"1234");
    c.update(b"56789");
    assert_eq!(c.finish(), 0xcbf4_3926);
}

#[test]
fn frame_layout_and_validity() {
    let f = Frame::new(T_APP_DATA, 1, 2, 0x1234, 0xd0, &[0x05, 0x00, 0x01, 0x00]);
    let b = f.encode().unwrap();
    assert_eq!(&b[..10], &[0xa5, 0x5a, 1, 9, 1, 2, 0x34, 0x12, 0xd0, 4]);
    assert_eq!(&b[10..14], &[5, 0, 1, 0]);
    let c = crc16(&b[2..14]);
    assert_eq!((b[14], b[15]), (c as u8, (c >> 8) as u8));
    assert_eq!(Frame::decode(&b), Some(f));
    // a damaged byte, unused payload not zero, bad length, unknown type, bad address
    let mut x = b;
    x[11] ^= 1;
    assert_eq!(Frame::decode(&x), None);
    let short = Frame::new(T_APP_DATA, 1, 2, 0, 0xd1, &[7]);
    let mut s = short.encode().unwrap();
    s[12] = 9; // unused byte
    let c = crc16(&s[2..14]);
    s[14] = c as u8;
    s[15] = (c >> 8) as u8;
    assert_eq!(Frame::decode(&s), None);
    assert!(Frame { len: 5, ..f }.encode().is_none());
    assert!(Frame { ftype: 10, ..f }.encode().is_none());
    assert!(Frame { ftype: 0, ..f }.encode().is_none());
    assert!(Frame { dst: 100, ..f }.encode().is_none());
    assert!(Frame { src: 18, ..f }.encode().is_none());
    assert!(Frame::new(T_APP_DATA, 1, BROADCAST, 0, 0, &[]).encode().is_some());
}

#[test]
fn parser_resyncs() {
    let f1 = Frame::new(T_APP_DATA, 1, 2, 1, 0xd1, &[1, 2, 3, 4]);
    let f2 = Frame::new(T_PING, 16, 0, 2, 16, &[]);
    let mut p = Parser::new();
    let mut out = vec![];
    let mut feed = |p: &mut Parser, bytes: &[u8]| {
        for &b in bytes {
            if let Some(f) = p.push(b) {
                out.push(f);
            }
        }
    };
    // noise, then a frame, then a torn frame (its first 7 bytes), then a whole one: the torn
    // one's bytes are inside the window when the next frame starts, and the next is found
    feed(&mut p, &[0, 0xa5, 0xa5, 0x5a, 0x77, 0xff]);
    feed(&mut p, &f1.encode().unwrap());
    feed(&mut p, &f1.encode().unwrap()[..7]);
    feed(&mut p, &f2.encode().unwrap());
    // a lost byte in the middle of a frame, then good frames
    let mut broken = f1.encode().unwrap().to_vec();
    broken.remove(9);
    feed(&mut p, &broken);
    feed(&mut p, &f2.encode().unwrap());
    feed(&mut p, &f1.encode().unwrap());
    assert_eq!(out, vec![f1, f2, f2, f1]);
    // a frame whose payload holds A5 5A 01
    let tricky = Frame::new(T_APP_DATA, 1, 2, 0xa55a, 0xd1, &[0xa5, 0x5a, 1, 0xa5]);
    let mut p = Parser::new();
    let mut n = 0;
    for &b in tricky.encode().unwrap().iter().chain(tricky.encode().unwrap().iter()) {
        if p.push(b) == Some(tricky) {
            n += 1;
        }
    }
    assert_eq!(n, 2);
}

#[test]
fn delivers_and_acks() {
    for len in [1usize, 3, 4, 5, 1023, 1024] {
        let data: Vec<u8> = (0..len).map(|i| (i * 7 + 1) as u8).collect();
        let mut d = dg(0);
        let mut g = Got::default();
        g.feed(&mut d, &frames(3, 0, 77, 0, FLAG_ACK, &data));
        assert_eq!(g.0, vec![(3, 0, data.clone())], "len {}", len);
        let a = acks(&mut d);
        assert_eq!(a, vec![Frame::new(T_APP_DATA, 0, 3, 77, SUB_ACK, &[0])]);
        // every frame validly encodes
        assert!(frames(3, 0, 77, 0, 0, &data).iter().all(|f| f.encode().is_some()));
    }
}

#[test]
fn no_ack_unless_asked_and_channel_kept() {
    let mut d = dg(0);
    let mut g = Got::default();
    g.feed(&mut d, &frames(1, 0, 5, 7, 0, b"hello"));
    assert_eq!(g.0, vec![(1, 7, b"hello".to_vec())]);
    assert!(acks(&mut d).is_empty());
}

#[test]
fn start_with_bad_length_is_refused() {
    let mut d = dg(0);
    let mut g = Got::default();
    // length 0
    let mut f = frames(1, 0, 9, 0, FLAG_ACK, b"x");
    f[0].payload[0] = 0;
    g.feed(&mut d, &[f[0], f[2]]);
    assert_eq!(acks(&mut d), vec![Frame::new(T_APP_DATA, 0, 1, 9, SUB_ACK, &[2])]);
    // length 1025 (over the format), with the fragments following
    let data = vec![1u8; 1025];
    g.feed(&mut d, &frames(1, 0, 10, 0, FLAG_ACK, &data));
    assert_eq!(acks(&mut d), vec![Frame::new(T_APP_DATA, 0, 1, 10, SUB_ACK, &[2])]);
    // a smaller buffer refuses what the format would allow
    let mut small: Box<Datagrams<16, 2>> = Box::new(Datagrams::new(Config::default()));
    small.set_node(0);
    let mut got = vec![];
    for f in frames(1, 0, 3, 0, FLAG_ACK, &[0u8; 17]) {
        small.on_frame(&f, 0, &mut |s, c, b| got.push((s, c, b.to_vec())));
    }
    assert!(got.is_empty());
    assert_eq!(small.poll_tx(0).unwrap().payload[0], 2);
    assert!(g.0.is_empty());
}

#[test]
fn data_out_of_order_or_past_length_is_broken() {
    let data: Vec<u8> = (0..12).collect();
    let mut d = dg(0);
    let mut g = Got::default();
    let mut f = frames(1, 0, 20, 0, FLAG_ACK, &data);
    f.remove(2); // fragment 1 lost
    g.feed(&mut d, &f);
    assert_eq!(acks(&mut d), vec![Frame::new(T_APP_DATA, 0, 1, 20, SUB_ACK, &[1])]);
    // bytes past the length: a fragment that does not fit
    let mut f = frames(1, 0, 21, 0, FLAG_ACK, &data[..6]);
    f[2] = Frame::new(T_APP_DATA, 1, 0, 1, SUB_DATA, &[1, 2, 3, 4]); // 4 + 4 > 6
    g.feed(&mut d, &f);
    assert_eq!(acks(&mut d), vec![Frame::new(T_APP_DATA, 0, 1, 21, SUB_ACK, &[1])]);
    // a duplicated fragment
    let mut f = frames(1, 0, 22, 0, FLAG_ACK, &data);
    f.insert(2, f[1]);
    g.feed(&mut d, &f);
    assert_eq!(acks(&mut d), vec![Frame::new(T_APP_DATA, 0, 1, 22, SUB_ACK, &[1])]);
    assert!(g.0.is_empty());
    assert_eq!(d.stats.broken, 3);
}

#[test]
fn end_rules() {
    let data = b"abcdefgh".to_vec();
    let mut d = dg(0);
    let mut g = Got::default();
    // END of another id: dropped, nothing delivered, no ACK; the real END still works
    let mut f = frames(1, 0, 30, 0, FLAG_ACK, &data);
    let end = f.pop().unwrap();
    g.feed(&mut d, &f);
    g.feed(&mut d, &[Frame { seq: 31, ..end }]);
    assert!(g.0.is_empty() && acks(&mut d).is_empty());
    g.feed(&mut d, &[end]);
    assert_eq!(g.0.len(), 1);
    assert_eq!(acks(&mut d).len(), 1);
    // bad CRC: dropped, ACK 1
    let mut f = frames(1, 0, 32, 0, FLAG_ACK, &data);
    f[1].payload[0] ^= 1;
    g.feed(&mut d, &f);
    assert_eq!(g.0.len(), 1);
    assert_eq!(acks(&mut d)[0].payload[0], 1);
    // fragments missing at the end (last DATA frame lost): the index check cannot see it
    let mut f = frames(1, 0, 33, 0, FLAG_ACK, &data);
    f.remove(f.len() - 2);
    g.feed(&mut d, &f);
    assert_eq!(acks(&mut d)[0].payload[0], 1);
    assert_eq!(g.0.len(), 1);
    // END with no START, DATA with no START: ignored, no ACK
    let f = frames(2, 0, 34, 0, FLAG_ACK, &data);
    g.feed(&mut d, &f[1..]);
    assert!(acks(&mut d).is_empty());
    assert_eq!(g.0.len(), 1);
}

#[test]
fn start_mid_message_abandons_the_previous() {
    let a: Vec<u8> = (0..40).collect();
    let b = b"second".to_vec();
    let mut d = dg(0);
    let mut g = Got::default();
    let fa = frames(1, 0, 40, 0, FLAG_ACK, &a);
    g.feed(&mut d, &fa[..5]); // START and 4 fragments of the first
    g.feed(&mut d, &frames(1, 0, 41, 0, FLAG_ACK, &b));
    assert_eq!(g.0, vec![(1, 0, b.clone())]);
    assert_eq!(acks(&mut d), vec![Frame::new(T_APP_DATA, 0, 1, 41, SUB_ACK, &[0])]);
    assert_eq!(d.stats.abandoned, 1);
    // the rest of the first message arrives late: DATA and END without a START, ignored
    g.feed(&mut d, &fa[5..]);
    assert_eq!(g.0.len(), 1);
    assert!(acks(&mut d).is_empty());
}

#[test]
fn duplicates_are_acked_not_delivered() {
    let mut d = dg(0);
    let mut g = Got::default();
    let f = frames(1, 0, 50, 0, FLAG_ACK, b"once");
    g.feed(&mut d, &f);
    g.feed(&mut d, &f);
    g.feed(&mut d, &f);
    assert_eq!(g.0.len(), 1);
    let a = acks(&mut d);
    assert_eq!(a.len(), 3);
    assert!(a.iter().all(|x| x.payload[0] == 0 && x.seq == 50));
    assert_eq!(d.stats.duplicates, 2);
    // the next id is new, and after it the old id is new again (only the last one is kept)
    g.feed(&mut d, &frames(1, 0, 51, 0, 0, b"next"));
    g.feed(&mut d, &frames(1, 0, 50, 0, 0, b"once"));
    assert_eq!(g.0.len(), 3);
    // per source: the same id from another source is not a duplicate
    g.feed(&mut d, &frames(2, 0, 50, 0, 0, b"once"));
    assert_eq!(g.0.len(), 4);
}

#[test]
fn interleaved_sources_reassemble_separately() {
    let a: Vec<u8> = (0..100).collect();
    let b: Vec<u8> = (100..200).collect();
    let (fa, fb) = (frames(1, 0, 1, 0, 0, &a), frames(16, 0, 1, 3, 0, &b));
    let mut d = dg(0);
    let mut g = Got::default();
    let mut order = vec![];
    for i in 0..fa.len().max(fb.len()) {
        if i < fa.len() {
            order.push(fa[i]);
        }
        if i < fb.len() {
            order.push(fb[i]);
        }
    }
    g.feed(&mut d, &order);
    assert_eq!(g.0.len(), 2);
    assert!(g.0.contains(&(1, 0, a)) && g.0.contains(&(16, 3, b)));
}

#[test]
fn without_a_buffer_the_message_is_refused() {
    let mut d: Box<Datagrams<64, 1>> = Box::new(Datagrams::new(Config::default()));
    d.set_node(0);
    let mut got = vec![];
    let fa = frames(1, 0, 1, 0, FLAG_ACK, &[1; 20]);
    let fb = frames(2, 0, 1, 0, FLAG_ACK, &[2; 20]);
    for i in 0..fa.len() {
        for f in [fa[i], fb[i]] {
            d.on_frame(&f, 0, &mut |s, _, _| got.push(s));
        }
    }
    assert_eq!(got, vec![1]);
    let mut st = vec![];
    while let Some(f) = d.poll_tx(0) {
        st.push((f.dst, f.payload[0]));
    }
    assert!(st.contains(&(2, 2)) && st.contains(&(1, 0)), "{:?}", st);
    // the buffer is free again afterwards
    let mut got = vec![];
    for f in frames(2, 0, 2, 0, 0, &[3; 20]) {
        d.on_frame(&f, 0, &mut |s, _, _| got.push(s));
    }
    assert_eq!(got, vec![2]);
}

#[test]
fn foreign_frames_are_ignored() {
    let mut d = dg(0);
    let mut g = Got::default();
    // not APP_DATA, source out of range, unknown sub-types, wrong lengths
    let mut fs = frames(1, 0, 1, 0, FLAG_ACK, b"abc");
    fs.insert(0, Frame::new(T_PING, 16, 0, 1, 0xd0, &[3, 0, 1, 0]));
    fs.insert(0, Frame::new(T_APP_DATA, 1, 0, 0, 0xa0, &[1, 2, 3, 4]));
    fs.insert(0, Frame::new(T_APP_DATA, 1, 0, 0, 0xe1, &[1, 2, 3, 4]));
    fs.insert(0, Frame::new(T_APP_DATA, 1, 0, 0, SUB_START, &[1, 2])); // wrong length
    g.feed(&mut d, &fs);
    assert_eq!(g.0.len(), 1);
    assert_eq!(acks(&mut d).len(), 1);
}

#[test]
fn sender_frames_and_ack() {
    let mut d = dg(2);
    let id = d.send(0, 0, b"hello", true).unwrap();
    assert!(d.tx_busy());
    assert_eq!(d.send(1, 0, b"x", false), Err(SendError::Busy));
    let mut v = vec![];
    while let Some(f) = d.poll_tx(1000) {
        v.push(f);
    }
    assert_eq!(v, frames(2, 0, id, 0, FLAG_ACK, b"hello"));
    assert_eq!(v.len(), 4);
    // waiting: nothing until the timeout (300 ms + 4 frames * 1.5 ms, rounded up)
    assert_eq!(d.next_wake(), Some(1000 + 300 + 6));
    assert!(d.poll_tx(1305).is_none() && d.tx_busy());
    // ACK of another id or from another source changes nothing
    d.on_frame(&Frame::new(T_APP_DATA, 0, 2, id.wrapping_add(1), SUB_ACK, &[0]), 0, &mut |_, _, _| {});
    d.on_frame(&Frame::new(T_APP_DATA, 1, 2, id, SUB_ACK, &[0]), 0, &mut |_, _, _| {});
    assert!(d.tx_busy());
    d.on_frame(&Frame::new(T_APP_DATA, 0, 2, id, SUB_ACK, &[0]), 0, &mut |_, _, _| {});
    assert!(!d.tx_busy());
    assert_eq!(d.take_result(), Some(SendResult { id, dest: 0, outcome: Outcome::Acked }));
}

#[test]
fn sender_resends_with_the_same_id_at_most_three_times() {
    let mut d = dg(2);
    let id = d.send(0, 0, &[9; 1024], true).unwrap();
    let mut t = 0u64;
    let mut starts = 0;
    let mut frames_seen = 0;
    while d.tx_busy() || d.take_result().is_none() {
        match d.poll_tx(t) {
            Some(f) => {
                frames_seen += 1;
                if f.sub == SUB_START {
                    starts += 1;
                    assert_eq!(f.seq, id);
                }
            }
            None => t += 10,
        }
        if t > 100_000 {
            panic!("never ends");
        }
        if !d.tx_busy() {
            break;
        }
    }
    assert_eq!(starts, 4, "the message and three resends");
    assert_eq!(frames_seen, 4 * 258);
    // 258 frames: the timeout is 300 ms + 258 * 1.5 ms
    let mut d = dg(2);
    d.send(0, 0, &[9; 1024], true).unwrap();
    while d.poll_tx(0).is_some() {}
    assert_eq!(d.next_wake(), Some(300 + 387));
}

#[test]
fn ack_codes_drive_the_sender() {
    // 1: broken, resend now; 2: refused, give up
    let mut d = dg(2);
    let id = d.send(0, 0, b"abc", true).unwrap();
    while d.poll_tx(0).is_some() {}
    d.on_frame(&Frame::new(T_APP_DATA, 0, 2, id, SUB_ACK, &[1]), 0, &mut |_, _, _| {});
    assert!(d.tx_busy());
    assert_eq!(d.poll_tx(0).unwrap().sub, SUB_START);
    while d.poll_tx(0).is_some() {}
    d.on_frame(&Frame::new(T_APP_DATA, 0, 2, id, SUB_ACK, &[2]), 0, &mut |_, _, _| {});
    assert_eq!(d.take_result().unwrap().outcome, Outcome::Refused);
    // broken four times: failure
    let id = d.send(0, 0, b"abc", true).unwrap();
    for _ in 0..4 {
        while d.poll_tx(0).is_some() {}
        d.on_frame(&Frame::new(T_APP_DATA, 0, 2, id, SUB_ACK, &[1]), 0, &mut |_, _, _| {});
    }
    assert_eq!(d.take_result().unwrap().outcome, Outcome::Failed);
    // fire and forget ends when the last frame leaves
    let id = d.send(1, 0, b"abc", false).unwrap();
    while d.poll_tx(0).is_some() {}
    assert_eq!(d.take_result(), Some(SendResult { id, dest: 1, outcome: Outcome::Sent }));
}

#[test]
fn send_limits_and_ids() {
    let mut d = dg(2);
    assert_eq!(d.send(0, 0, &[], false), Err(SendError::Size));
    assert_eq!(d.send(0, 0, &[0; 1025], false), Err(SendError::Size));
    for bad in [2u8, 17, 32, 254, 64] {
        assert_eq!(d.send(bad, 0, b"x", false), Err(SendError::BadDest), "{}", bad);
    }
    assert!(d.send(16, 0, b"x", false).is_ok());
    let mut n = Box::new(Datagrams::<8, 1>::new(Config::default()));
    assert_eq!(n.send(0, 0, b"x", false), Err(SendError::NoAddress));
    // ids count up per message and wrap
    let mut d: Box<Dg> = Box::new(Datagrams::new(Config { first_id: 0xfffe, ..Config::default() }));
    d.set_node(1);
    let mut ids = vec![];
    for _ in 0..4 {
        ids.push(d.send(0, 0, b"x", false).unwrap());
        while d.poll_tx(0).is_some() {}
    }
    assert_eq!(ids, vec![0xfffe, 0xffff, 0, 1]);
}

fn cfg() -> LaneConfig {
    LaneConfig::default()
}

fn feed(l: &mut Lane<64, 2>, f: Frame, t: u64) {
    l.rx(&f.encode().unwrap(), t, &mut |_, _, _| {});
}

#[test]
fn lane_learns_its_number_and_answers_ping() {
    let mut l: Lane<64, 2> = Lane::new(cfg());
    assert!(l.poll_tx(0).is_none());
    // a frame to the supervisor address or a broadcast does not name this port
    feed(&mut l, Frame::new(T_PING, 16, 254, 1, 16, &[]), 0);
    feed(&mut l, Frame::new(T_PING, 16, 16, 1, 16, &[]), 0);
    assert_eq!(l.node(), None);
    assert!(l.poll_tx(0).is_none());
    // a unicast one does, and is answered
    feed(&mut l, Frame::new(T_PING, 16, 2, 7, 16, &[]), 5);
    assert_eq!(l.node(), Some(2));
    let pong = Frame::decode(&l.poll_tx(5).unwrap()).unwrap();
    assert_eq!((pong.ftype, pong.src, pong.dst, pong.seq), (T_PONG, 2, 16, 7));
    assert!(l.poll_tx(5).is_none());
    // frames for other nodes are ignored
    feed(&mut l, Frame::new(T_PING, 16, 3, 8, 16, &[]), 6);
    assert!(l.poll_tx(6).is_none());
    // STATUS once a second to the supervisor
    assert!(l.poll_tx(900).is_none());
    let s = Frame::decode(&l.poll_tx(1005).unwrap()).unwrap();
    assert_eq!((s.ftype, s.src, s.dst), (T_STATUS, 2, 16));
    assert_eq!(s.payload[1], 0x11);
    assert!(l.poll_tx(1500).is_none());
    assert!(l.poll_tx(2005).is_some());
}

#[test]
fn lane_ignores_valid_unknown_frames() {
    let mut l: Lane<64, 2> = Lane::new(LaneConfig { node: Some(1), ..cfg() });
    for f in [
        Frame::new(T_PONG, 16, 1, 1, 16, &[]),
        Frame::new(T_STATUS, 3, 1, 1, 16, &[0, 1, 0, 0]),
        Frame::new(4, 16, 1, 1, 16, &[50]),
        Frame::new(T_APP_DATA, 0, 1, 1, 0xa0, &[1, 0, 0, 0]),
        Frame::new(T_APP_DATA, 0, 1, 1, 0xe1, &[1, 0, 0, 0]),
        Frame::new(T_APP_DATA, 0, 1, 1, 0xd1, &[1]),
    ] {
        feed(&mut l, f, 10);
    }
    assert!(l.poll_tx(10).is_none());
    assert_eq!(l.stats.pings, 0);
}

/// The RAM the module's documentation promises: `(SLOTS + 1) * MAX` of buffers plus a few hundred bytes.
#[test]
fn ram_is_bounded() {
    use core::mem::size_of;
    let full = size_of::<Lane<1024, 17>>();
    let pico = size_of::<Lane<1024, 6>>();
    let small = size_of::<Lane<256, 4>>();
    println!("Lane<1024,17> {} bytes, Lane<1024,6> {} bytes, Lane<256,4> {} bytes", full, pico, small);
    assert!(full <= 18 * 1024 + 700, "{}", full);
    assert!(pico <= 7 * 1024 + 700, "{}", pico);
    assert!(small <= 5 * 256 + 700, "{}", small);
}

/// The three frames that `tools/lane_frames.py send 0 '1+2\r'` prints (a supervisor typing on
/// node 0's console): the Python tool and this layer agree byte for byte.
#[test]
fn frames_from_the_supervisor_side_tool() {
    let hex = ["a55a010910000100d004040000016cd9", "a55a010910000000d104312b320dff9f", "a55a010910000100d20415c10ace90e9"];
    let mut l: Lane<64, 2> = Lane::new(LaneConfig { node: Some(0), ..cfg() });
    let mut got = vec![];
    for h in hex {
        let bytes: Vec<u8> = (0..16).map(|i| u8::from_str_radix(&h[2 * i..2 * i + 2], 16).unwrap()).collect();
        l.rx(&bytes, 0, &mut |s, c, b| got.push((s, c, b.to_vec())));
    }
    assert_eq!(got, vec![(16, 1, b"1+2\r".to_vec())]);
}

#[test]
fn liveness_frames_are_exact() {
    use klong_dgram::lane::{STATUS_ALIVE, STATUS_SELFTEST_FAIL, STATUS_SELFTEST_PASS, STATUS_SELFTEST_RUNNING};
    assert_eq!((STATUS_ALIVE, STATUS_SELFTEST_RUNNING, STATUS_SELFTEST_PASS, STATUS_SELFTEST_FAIL), (0x11, 0x31, 0x51, 0x91));
    let mut l: Lane<64, 2> = Lane::new(LaneConfig { node: Some(3), ..cfg() });
    // a PING with a payload and an odd opaque byte: the PONG has the same sequence and length 0
    feed(&mut l, Frame::new(T_PING, 16, 3, 0xbeef, 77, &[1, 2]), 10);
    let pong = Frame::decode(&l.poll_tx(10).unwrap()).unwrap();
    assert_eq!((pong.ftype, pong.src, pong.dst, pong.seq, pong.len), (T_PONG, 3, 16, 0xbeef, 0));
    // STATUS: length 4, load, flags, uptime seconds LE, once a second, monotonic
    l.set_load(250);
    l.set_status_flags(STATUS_SELFTEST_PASS);
    let mut ups = vec![];
    for t in [1010u64, 2020, 3030, 70_000_000] {
        let s = Frame::decode(&l.poll_tx(t).unwrap()).unwrap();
        assert_eq!((s.ftype, s.dst, s.len), (T_STATUS, 16, 4));
        assert_eq!((s.payload[0], s.payload[1]), (100, 0x51));
        ups.push(s.payload[2] as u16 | (s.payload[3] as u16) << 8);
        assert!(l.poll_tx(t + 500).is_none());
    }
    assert_eq!(ups, vec![1, 2, 3, 65535]);
}

#[test]
fn a_gated_channel_has_one_owner_at_a_time() {
    let mut d = dg(0);
    d.set_gate(1);
    let mut got = vec![];
    let mut feed = |d: &mut Dg, fs: &[Frame]| {
        for f in fs {
            d.on_frame_id(f, 0, &mut |s, c, id, _, b| got.push((s, c, id, b.to_vec())));
        }
    };
    feed(&mut d, &frames(16, 0, 7, 1, FLAG_ACK, b"1+1\n"));
    assert_eq!(d.gate_owner(), Some(16));
    // another source on the same channel is refused (ACK 2) and may try again later; the channel 0 is free
    feed(&mut d, &frames(2, 0, 5, 1, FLAG_ACK, b"2+2\n"));
    feed(&mut d, &frames(2, 0, 6, 0, FLAG_ACK, b"free"));
    // output of another console (no ACK asked for) is not input: the gate does not stop it, and it does not take the gate
    feed(&mut d, &frames(3, 0, 9, 1, 0, b"out\n"));
    let acks: Vec<_> = acks(&mut d).iter().map(|a| (a.dst, a.seq, a.payload[0])).collect();
    assert_eq!(acks, vec![(16, 7, 0), (2, 5, 2), (2, 6, 0)]);
    assert_eq!(d.gate_owner(), Some(16));
    // the owner may go on; the id is given with the message
    feed(&mut d, &frames(16, 0, 8, 1, FLAG_ACK, b"3+3\n"));
    d.release_gate();
    feed(&mut d, &frames(2, 0, 5, 1, FLAG_ACK, b"2+2\n"));
    assert_eq!(got.iter().map(|g| (g.0, g.1, g.2)).collect::<Vec<_>>(), vec![(16, 1, 7), (2, 0, 6), (3, 1, 9), (16, 1, 8), (2, 1, 5)]);
}
