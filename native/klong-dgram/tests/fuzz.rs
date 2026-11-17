//! Property tests with a seeded generator: noise between frames, dropped, duplicated and
//! injected frames, START mid-message, interleaved senders, the sizes at the edges, id wrap.
use klong_dgram::crc::crc32;
use klong_dgram::dgram::*;
use klong_dgram::frame::*;
use klong_dgram::{Config, Datagrams, Lane, LaneConfig};
use std::collections::VecDeque;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    fn chance(&mut self, ppm: u32) -> bool {
        self.next() % 1_000_000 < ppm as u64
    }
    fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| self.next() as u8).collect()
    }
    /// 0..n random bytes.
    fn upto(&mut self, n: u64) -> Vec<u8> {
        let k = self.below(n) as usize;
        self.bytes(k)
    }
}

type L = Lane<1024, 17>;

#[derive(Clone, Copy, Default)]
struct Faults {
    drop: u32,
    dup: u32,
    /// Noise bytes inserted before a frame (ppm of frames) and their count.
    noise: u32,
    flip: u32,
    start_mid: u32,
}

struct Net {
    nodes: Vec<Box<L>>,
    rng: Rng,
    faults: Faults,
    /// Frames queued for each node's port (capacity 256), as bytes with the time they arrive.
    queues: Vec<VecDeque<[u8; 16]>>,
    cap: usize,
    pub overflow: u32,
    now_ms: f64,
    pub got: Vec<Vec<(u8, u8, Vec<u8>)>>,
    pub results: Vec<Vec<SendResult>>,
}

impl Net {
    fn new(n: usize, seed: u64, faults: Faults, ids: u16) -> Net {
        let nodes = (0..n)
            .map(|i| {
                let c = LaneConfig { node: Some(i as u8), supervisor_duties: false, dgram: Config { first_id: ids, ..Config::default() } };
                Box::new(L::new(c))
            })
            .collect();
        Net { nodes, rng: Rng(seed | 1), faults, queues: vec![VecDeque::new(); n], cap: 256, overflow: 0, now_ms: 0.0, got: vec![vec![]; n], results: vec![vec![]; n] }
    }

    /// One frame time (1.389 ms at 115200 baud): every node may send one frame, every port
    /// delivers one frame.
    fn tick(&mut self) {
        self.now_ms += 1.389;
        let now = self.now_ms as u64;
        for i in 0..self.nodes.len() {
            if let Some(bytes) = self.nodes[i].poll_tx(now) {
                self.route(bytes);
            }
            if let Some(r) = self.nodes[i].dg.take_result() {
                self.results[i].push(r);
            }
        }
        for i in 0..self.nodes.len() {
            let mut bytes: Vec<u8> = vec![];
            if self.rng.chance(self.faults.noise) {
                let n = 1 + self.rng.below(40) as usize;
                bytes.extend(self.rng.bytes(n));
            }
            if let Some(f) = self.queues[i].pop_front() {
                let mut f = f;
                if self.rng.chance(self.faults.flip) {
                    let k = self.rng.below(16) as usize;
                    f[k] ^= 1 << self.rng.below(8);
                }
                bytes.extend(f);
                if self.rng.chance(self.faults.start_mid) {
                    // a START from the sender of this frame, in the middle of whatever it was sending
                    let src = f[4];
                    let s = Frame::new(T_APP_DATA, src, i as u8, self.rng.next() as u16, SUB_START, &[9, 0, 0, 0]);
                    bytes.extend(s.encode().unwrap());
                }
            }
            if !bytes.is_empty() {
                let got = &mut self.got[i];
                self.nodes[i].rx(&bytes, now, &mut |s, c, b| got.push((s, c, b.to_vec())));
            }
        }
    }

    fn route(&mut self, bytes: [u8; 16]) {
        if self.rng.chance(self.faults.drop) {
            return;
        }
        let copies = if self.rng.chance(self.faults.dup) { 2 } else { 1 };
        let dst = bytes[5] as usize;
        for _ in 0..copies {
            if dst >= self.queues.len() {
                continue;
            }
            if self.queues[dst].len() >= self.cap {
                self.overflow += 1;
            } else {
                self.queues[dst].push_back(bytes);
            }
        }
    }

    fn idle(&self) -> bool {
        self.queues.iter().all(|q| q.is_empty()) && self.nodes.iter().all(|n| !n.dg.tx_busy())
    }
}

fn payload(rng: &mut Rng, n: usize) -> Vec<u8> {
    rng.bytes(n)
}

#[test]
fn sizes_at_the_edges_clean_link() {
    for &len in &[1usize, 3, 4, 5, 1023, 1024] {
        let mut net = Net::new(2, 7, Faults::default(), 0);
        let mut rng = Rng(99);
        let data = payload(&mut rng, len);
        net.nodes[0].send(1, 0, &data, true).unwrap();
        for _ in 0..3000 {
            net.tick();
        }
        assert_eq!(net.got[1], vec![(0, 0, data)], "len {}", len);
        assert_eq!(net.results[0].len(), 1);
        assert_eq!(net.results[0][0].outcome, Outcome::Acked);
    }
    let mut net = Net::new(2, 7, Faults::default(), 0);
    assert_eq!(net.nodes[0].send(1, 0, &[0; 1025], true), Err(SendError::Size));
}

#[test]
fn wire_time_of_1024_bytes() {
    // 258 frames of 16 bytes at 115200 baud, 8N1: about 0.36 s
    let mut net = Net::new(2, 3, Faults::default(), 0);
    net.nodes[0].send(1, 0, &[5; 1024], false).unwrap();
    let mut t = 0;
    while net.got[1].is_empty() {
        net.tick();
        t += 1;
        assert!(t < 1000);
    }
    assert!((257..=262).contains(&t), "{} frame times", t);
}

#[test]
fn noise_between_frames() {
    let mut net = Net::new(2, 11, Faults { noise: 300_000, ..Faults::default() }, 0);
    let mut rng = Rng(5);
    let mut sent = vec![];
    for i in 0..30 {
        let data = payload(&mut rng, 1 + (i * 37) % 200);
        net.nodes[0].send(1, 0, &data, true).unwrap();
        sent.push(data);
        while !net.idle() {
            net.tick();
        }
        for _ in 0..5 {
            net.tick();
        }
    }
    // noise is only dangerous when it eats a frame: with ACK + resends everything arrives
    let got: Vec<_> = net.got[1].iter().map(|g| g.2.clone()).collect();
    assert_eq!(got, sent);
}

fn lossy_run(seed: u64, faults: Faults, count: usize, first_id: u16) -> (Net, Vec<(Vec<u8>, Outcome)>) {
    let mut net = Net::new(2, seed, faults, first_id);
    let mut rng = Rng(seed ^ 0xabcdef);
    let mut sent = vec![];
    for _ in 0..count {
        let n = [1usize, 3, 4, 5, 17, 100, 255][rng.below(7) as usize];
        let data = payload(&mut rng, n);
        net.nodes[0].send(1, 0, &data, true).unwrap();
        let mut guard = 0;
        while !net.idle() {
            net.tick();
            guard += 1;
            assert!(guard < 100_000);
        }
        for _ in 0..3 {
            net.tick();
        }
        let r = net.results[0].pop().expect("a result");
        net.results[0].clear();
        sent.push((data, r.outcome));
    }
    (net, sent)
}

#[test]
fn drops_and_duplicates_exactly_once() {
    for seed in 1..=8u64 {
        let faults = Faults { drop: 5_000, dup: 3_000, noise: 20_000, flip: 1_000, start_mid: 0 };
        let (net, sent) = lossy_run(seed, faults, 60, 0xffe0);
        let got = &net.got[1];
        // every message the sender saw acknowledged was delivered once; none delivered twice;
        // nothing was delivered that was not sent
        let mut cursor = 0;
        for (data, outcome) in &sent {
            let n = got.iter().filter(|g| &g.2 == data).count();
            assert!(n <= 1, "seed {} delivered {} times", seed, n);
            if *outcome == Outcome::Acked {
                assert_eq!(n, 1, "seed {}: acked but not delivered", seed);
            }
            cursor += n;
        }
        assert_eq!(cursor, got.len(), "seed {}: something unsent was delivered", seed);
        let acked = sent.iter().filter(|s| s.1 == Outcome::Acked).count();
        assert!(acked >= 50, "seed {}: only {} of 60 acknowledged", seed, acked);
    }
}

#[test]
fn start_mid_message() {
    let faults = Faults { start_mid: 20_000, drop: 10_000, ..Faults::default() };
    let (net, sent) = lossy_run(21, faults, 40, 0);
    for g in &net.got[1] {
        assert!(sent.iter().any(|s| s.0 == g.2), "delivered something never sent");
    }
    assert!(sent.iter().filter(|s| s.1 == Outcome::Acked).count() >= 30);
}

#[test]
fn id_wraps() {
    let (net, sent) = lossy_run(33, Faults::default(), 100, 0xffc0);
    assert!(sent.iter().all(|s| s.1 == Outcome::Acked));
    assert_eq!(net.got[1].len(), 100);
    // the id went through 0xffff -> 0 and the first message after the wrap was still new
    assert_eq!(net.nodes[0].dg.stats.resent, 0);
}

#[test]
fn interleaved_senders_to_one_node() {
    // three senders, 1 KB each, with ACKs, into node 3 at once: their frames interleave and
    // the sum of three lanes does not fit one: the 256-frame queue overflows and ACK + resend
    // brings the messages through
    let mut net = Net::new(4, 77, Faults::default(), 0);
    let mut rng = Rng(8);
    let msgs: Vec<Vec<u8>> = (0..3).map(|_| payload(&mut rng, 1024)).collect();
    let pending: Vec<usize> = vec![0, 1, 2];
    for (i, m) in msgs.iter().enumerate() {
        net.nodes[i].send(3, 0, m, true).unwrap();
    }
    let mut done = [false; 3];
    for _ in 0..40_000 {
        net.tick();
        for i in 0..3 {
            if let Some(r) = net.results[i].pop() {
                done[i] = r.outcome == Outcome::Acked;
                if !done[i] && pending.contains(&i) {
                    // gave up after 3 resends: try again as a new message
                    net.nodes[i].send(3, 0, &msgs[i], true).unwrap();
                }
            }
        }
        if done.iter().all(|d| *d) {
            break;
        }
    }
    assert!(done.iter().all(|d| *d), "{:?} overflow {}", done, net.overflow);
    assert!(net.overflow > 0, "the test should overflow the queue");
    for m in &msgs {
        assert_eq!(net.got[3].iter().filter(|g| &g.2 == m).count(), 1);
    }
    for g in &net.got[3] {
        assert!(msgs.contains(&g.2));
    }
}

/// Arbitrary mutations of legitimate traffic from several sources into one receiver: whatever
/// is delivered was sent, and never twice in a row from one source.
#[test]
fn mutated_streams_deliver_only_what_was_sent() {
    for seed in 1..=300u64 {
        let mut rng = Rng(seed.wrapping_mul(0x9e3779b97f4a7c15) | 1);
        let mut d: Box<Datagrams<64, 3>> = Box::new(Datagrams::new(Config::default()));
        d.set_node(0);
        let mut sent: Vec<(u8, u8, Vec<u8>)> = vec![];
        let mut streams: Vec<Vec<Frame>> = vec![];
        for src in [1u8, 2, 16] {
            let mut v = vec![];
            for k in 0..6u16 {
                let len = 1 + rng.below(64) as usize;
                let data = rng.bytes(len);
                let chan = rng.below(3) as u8;
                let flags = rng.below(2) as u8;
                let l = len as u16;
                v.push(Frame::new(T_APP_DATA, src, 0, k.wrapping_add(rng.below(3) as u16 * 0), SUB_START, &[l as u8, (l >> 8) as u8, flags, chan]));
                let id = k;
                v[0].seq = id;
                for (i, c) in data.chunks(4).enumerate() {
                    v.push(Frame::new(T_APP_DATA, src, 0, i as u16, SUB_DATA, c));
                }
                v.push(Frame::new(T_APP_DATA, src, 0, id, SUB_END, &crc32(&data).to_le_bytes()));
                sent.push((src, chan, data));
            }
            streams.push(v);
        }
        // merge streams in random interleaving with mutations
        let mut pos = [0usize; 3];
        let mut last: [Option<(u8, Vec<u8>)>; 17] = Default::default();
        loop {
            let live: Vec<usize> = (0..3).filter(|&i| pos[i] < streams[i].len()).collect();
            if live.is_empty() {
                break;
            }
            let i = live[rng.below(live.len() as u64) as usize];
            let f = streams[i][pos[i]];
            pos[i] += 1;
            let mut out = vec![f];
            match rng.below(12) {
                0 => out.clear(),
                1 => out.push(f),
                2 => out.push(Frame::new(T_APP_DATA, f.src, 0, rng.next() as u16, SUB_START + rng.below(4) as u8, &rng.upto(5))),
                3 => out[0].payload[0] ^= 1 << rng.below(8),
                _ => {}
            }
            for f in out {
                d.on_frame(&f, 0, &mut |s, c, b| {
                    assert!(sent.iter().any(|m| m.0 == s && m.1 == c && m.2 == b), "seed {}: delivered something not sent", seed);
                    let key = (c, b.to_vec());
                    assert_ne!(last[s as usize].as_ref(), Some(&key), "seed {}: delivered twice in a row", seed);
                    last[s as usize] = Some(key);
                });
            }
            while d.poll_tx(0).is_some() {}
        }
    }
}

/// Anything at all on the wire: random bytes, random frames. Nothing panics and the parser
/// stays in step with a real frame that follows.
#[test]
fn garbage_never_breaks_the_lane() {
    let mut rng = Rng(12345);
    let mut l: Box<L> = Box::new(L::new(LaneConfig::default()));
    let mut delivered = 0;
    for round in 0..2000 {
        let mut bytes = rng.upto(60);
        for _ in 0..rng.below(3) {
            let f = Frame::new(
                1 + rng.below(9) as u8,
                rng.below(18) as u8,
                [0u8, 1, 2, 3, 16, 254, 32][rng.below(7) as usize],
                rng.next() as u16,
                0xd0 + rng.below(5) as u8,
                &rng.upto(5),
            );
            if let Some(b) = f.encode() {
                bytes.extend(b);
            }
        }
        l.rx(&bytes, round, &mut |_, _, _| delivered += 1);
        while l.poll_tx(round).is_some() {}
        let _ = l.next_wake(round);
    }
    // after all that, a clean message still arrives (if the node learned its number)
    let node = l.node().expect("a unicast frame came by");
    let mut got = 0;
    let src = if node == 1 { 2 } else { 1 };
    let l_ = 3u16;
    for f in [
        Frame::new(T_APP_DATA, src, node, 9000, SUB_START, &[l_ as u8, 0, 0, 5]),
        Frame::new(T_APP_DATA, src, node, 0, SUB_DATA, b"abc"),
        Frame::new(T_APP_DATA, src, node, 9000, SUB_END, &crc32(b"abc").to_le_bytes()),
    ] {
        l.rx(&f.encode().unwrap(), 5000, &mut |_, c, b| {
            assert_eq!((c, b), (5, &b"abc"[..]));
            got += 1;
        });
    }
    assert_eq!(got, 1);
    let _ = delivered;
}
