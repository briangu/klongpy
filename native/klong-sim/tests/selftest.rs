//! The boot self-test of the Pico image (`klong_rt::selftest`, the firmware's own logic) on boards
//! behind the switch model, with a simulated supervisor that reads the channel-1 reports and the
//! STATUS flags, at both lane rates.
use klong_dgram::frame::{Frame, T_APP_DATA, T_STATUS};
use klong_dgram::lane::{STATUS_SELFTEST_FAIL, STATUS_SELFTEST_PASS, STATUS_SELFTEST_RUNNING};
use klong_rt::node::LaneIo;
use klong_sim::bench::*;

static ONE: std::sync::Mutex<()> = std::sync::Mutex::new(());
const BAUDS: [u32; 2] = [115_200, 4_500_000];

fn bench(n: usize, baud: u32) -> Bench {
    Bench::new(n, BenchOpts { baud, selftest: true, ..BenchOpts::default() })
}

/// What the supervisor has read on channel 1 from `node` so far.
struct Reports(Vec<(u8, String)>);
impl Reports {
    fn collect(&mut self, b: &Bench) {
        for (src, chan, m) in b.sup.take_log() {
            assert_eq!(chan, 3);
            self.0.push((src, String::from_utf8_lossy(&m).into_owned()));
        }
    }
    fn last(&self, node: u8) -> Option<&str> {
        self.0.iter().rev().find(|r| r.0 == node).map(|r| r.1.as_str())
    }
}

/// Run until `node` has reported something that starts with `prefix`; the virtual seconds it took.
fn until(b: &Bench, r: &mut Reports, node: u8, prefix: &str, limit_s: u64) -> f64 {
    let t0 = b.now_us();
    while b.now_us() - t0 < limit_s * 1_000_000 {
        b.spin(20_000);
        r.collect(b);
        if r.last(node).is_some_and(|t| t.starts_with(prefix)) {
            return (b.now_us() - t0) as f64 / 1e6;
        }
    }
    panic!("no `{}` from node {} within {} s; reports: {:?}; flags {:?}", prefix, node, limit_s, r.0.iter().rev().take(4).collect::<Vec<_>>(), &b.switch.stats().status_flags[..4]);
}

#[test]
fn two_boards_one_image_pass() {
    let _g = ONE.lock().unwrap_or_else(|e| e.into_inner());
    for baud in BAUDS {
        let b = bench(2, baud);
        let mut r = Reports(vec![]);
        let took = until(&b, &mut r, 1, "klong selftest pass ", 60);
        // the higher node connected to the lower, which serves
        let pass = r.last(1).unwrap().to_string();
        println!("METRIC [{} baud] node 1 reports after {:.1} s: {}", baud, took, pass);
        assert!(pass.contains("node=1 peer=0"), "{}", pass);
        let ms: u64 = pass["klong selftest pass ".len()..].split("ms").next().unwrap().parse().unwrap();
        assert!(ms > 0 && ms < 20_000, "{}", ms);
        b.spin_ms(4000);
        r.collect(&b);
        until(&b, &mut r, 0, "klong selftest serving node=0 calls=", 30);
        let serving = r.last(0).unwrap();
        let calls: u64 = serving.rsplit("calls=").next().unwrap().parse().unwrap();
        assert!(calls >= 5, "{}", serving);
        println!("METRIC [{} baud] node 0 reports: {}", baud, serving);
        // the STATUS flags the supervisor sees, and the load and uptime
        b.spin_ms(1500);
        let st = b.switch.stats();
        assert_eq!(st.status_flags[1], STATUS_SELFTEST_PASS);
        assert_eq!(st.status_flags[0], STATUS_SELFTEST_RUNNING);
        assert_eq!(st.status_load[1], 0);
        assert!(st.status_uptime[1] >= 1);
        // the report repeats every few seconds, and the verdict stays
        let n0 = r.0.iter().filter(|x| x.0 == 1).count();
        b.spin_ms(10_000);
        r.collect(&b);
        let n1 = r.0.iter().filter(|x| x.0 == 1).count();
        assert!(n1 >= n0 + 2, "{} -> {}", n0, n1);
        assert!(r.last(1).unwrap().starts_with("klong selftest pass "));
        assert_eq!(b.switch.stats().rx_ring_overruns, [0; 17]);
    }
}

#[test]
fn the_nearest_answering_node_below_is_the_peer() {
    let _g = ONE.lock().unwrap_or_else(|e| e.into_inner());
    for baud in BAUDS {
        // node 1 runs other firmware: node 2 tries it, waits out the handshake, and then uses node 0
        let b = bench(3, baud);
        b.node(1).mute.set(true);
        let mut r = Reports(vec![]);
        let took = until(&b, &mut r, 2, "klong selftest pass ", 120);
        println!("METRIC [{} baud] with a mute node in between: node 2 passes after {:.1} s: {}", baud, took, r.last(2).unwrap());
        assert!(r.last(2).unwrap().contains("node=2 peer=0"));
        b.spin_ms(1500);
        assert_eq!(b.switch.stats().status_flags[2], STATUS_SELFTEST_PASS);
    }
}

#[test]
fn a_lone_board_serves_and_never_fails() {
    let _g = ONE.lock().unwrap_or_else(|e| e.into_inner());
    for baud in BAUDS {
        let b = bench(2, baud);
        b.node(0).mute.set(true); // the only board with the image is node 1; below it nobody answers
        let mut r = Reports(vec![]);
        until(&b, &mut r, 1, "klong selftest serving node=1", 60);
        b.spin_ms(30_000);
        r.collect(&b);
        assert!(r.0.iter().all(|x| x.1.starts_with("klong selftest serving") || x.1.starts_with("klong selftest running")), "{:?}", r.0);
        assert_eq!(b.switch.stats().status_flags[1], STATUS_SELFTEST_RUNNING);
    }
}

#[test]
fn a_broken_peer_fails_with_the_step_and_recovers() {
    let _g = ONE.lock().unwrap_or_else(|e| e.into_inner());
    for baud in BAUDS {
        let b = bench(2, baud);
        let mut r = Reports(vec![]);
        // node 0 serves a broken echo before node 1 gets to it
        b.spin_ms(100);
        b.node(0).ok("stecho0::{x+\"a\"}");
        let took = until(&b, &mut r, 1, "klong selftest fail ", 120);
        let fail = r.last(1).unwrap().to_string();
        println!("METRIC [{} baud] after {:.1} s: {}", baud, took, fail);
        assert!(fail.starts_with("klong selftest fail echo "), "{}", fail);
        b.spin_ms(1500);
        assert_eq!(b.switch.stats().status_flags[1], STATUS_SELFTEST_FAIL);
        // fixed, it is tried again and passes
        b.node(0).ok("stecho0::{hits::hits+1;x}");
        until(&b, &mut r, 1, "klong selftest pass ", 120);
        b.spin_ms(1500);
        assert_eq!(b.switch.stats().status_flags[1], STATUS_SELFTEST_PASS);
    }
}

#[test]
fn a_flood_of_the_deployers_frames_does_not_disturb_it() {
    let _g = ONE.lock().unwrap_or_else(|e| e.into_inner());
    for baud in BAUDS {
        let b = bench(2, baud);
        let mut io = b.switch.port(klong_dgram::frame::SUPERVISOR, b.clock.clone());
        let mut r = Reports(vec![]);
        let mut k = 0u16;
        let t0 = b.now_us();
        // 1 ms steps: a burst of foreign APP_DATA frames to each node (sub-types 0xA0..0xB1, 0xE1/0xE2) and a STATUS
        while !r.last(1).is_some_and(|t| t.starts_with("klong selftest pass ")) {
            assert!(b.now_us() - t0 < 120_000_000, "no pass under the flood: {:?}", r.0.last());
            for _ in 0..3 {
                for dst in 0..2u8 {
                    let sub = match k % 20 {
                        18 => 0xe1,
                        19 => 0xe2,
                        v => 0xa0 + v as u8,
                    };
                    let f = Frame::new(T_APP_DATA, 16, dst, k, sub, &[1, 2, 3, 4]);
                    let bytes = f.encode().unwrap();
                    let mut off = 0;
                    while off < 16 {
                        let n = io.write(&bytes[off..]);
                        if n == 0 {
                            break;
                        }
                        off += n;
                    }
                    k = k.wrapping_add(1);
                }
            }
            let junk = Frame::new(T_STATUS, 16, 1, k, 16, &[0, 1, 0, 0]).encode().unwrap();
            let _ = io.write(&junk);
            b.spin(1000);
            r.collect(&b);
        }
        let flooded = b.node(1).net.lane_stats();
        println!("METRIC [{} baud] passed under a flood of {} foreign frames; node 1 saw {} frames", baud, k, flooded.frames_rx);
        assert!(r.last(1).unwrap().contains("peer=0"));
    }
}
