//! Native IPC between boards on the lane of a packet switch (16-byte frames, 115200 baud,
//! 256-frame queues), in virtual time: `.srv`, `.cli`, the management library, a peer that
//! resets, a queue that overflows, and the numbers.
use klong_core::Kind;
use klong_sim::bench::*;

/// One thing at a time: the interpreters share per-process state (nesting bound).
static ONE: std::sync::Mutex<()> = std::sync::Mutex::new(());

const FUNCS: &str = "sq::{x*x};add::{x+y};idf::{x};total::{+/x};nil::{42};fail::{1+\"a\"};raise::{.err(x;\"boom\")};count::{#x}";

fn serve(b: &Bench, i: usize) {
    let n = b.node(i);
    n.ok(FUNCS);
    assert_eq!(n.ok(".srv(\"node:\")"), "1");
}

/// The two rates the bench is tested at. `opts` is the bench's settings for one of them.
const BAUDS: [u32; 2] = [115_200, 4_500_000];

fn opts(baud: u32) -> BenchOpts {
    println!("METRIC -- {} baud --", baud);
    BenchOpts { baud, ..BenchOpts::default() }
}

macro_rules! both {
    ($($name:ident => $body:ident),* $(,)?) => {$(
        #[test]
        fn $name() {
            for baud in BAUDS {
                $body(baud);
            }
        }
    )*};
}

fn ms(us: u64) -> f64 {
    us as f64 / 1000.0
}

fn two_nodes_calls_values_errors_body(baud: u32) {
    let _g = ONE.lock().unwrap_or_else(|e| e.into_inner());
    let b = Bench::new(2, opts(baud));
    let (a, c) = (b.node(0), b.node(1));
    serve(&b, 1);
    assert_eq!(a.ok("h::.cli(\"node:1\");h(:sq,,7)"), "49");
    assert_eq!(a.ok("h(:add,3,4)"), "7");
    assert_eq!(a.ok("h(\"1+2\")"), "3");
    assert_eq!(a.ok("f::h(:sq);f(9)"), "81");
    assert_eq!(a.ok("g::h(:nil);g()"), "42");
    assert_eq!(a.ok("f'[1 2 3]"), "[1 4 9]");
    // values of every mapped kind come back as they went
    for lit in ["5", "-3", "1.5", "\"text\"", ":sym", "[1 2 3]", "[1.5 2.5]", "[\"a\" \"b\"]", "[1 \"a\" :b]", "[[1 2] [3 4]]", ",5", ":{[1 2] [\"a\" 3]}", "\"\""] {
        assert_eq!(a.ok(&format!("h(:idf,,{})", lit)), a.ok(lit), "{}", lit);
    }
    // state on the server
    assert_eq!(a.ok("d::.clid(h);d,:q,,11;d?:q"), "11");
    assert_eq!(c.ok("q"), "11");
    // errors keep their kind and say who reported them
    let e = a.run("h(:fail,,1)");
    assert_eq!(e.as_ref().unwrap_err().0, Kind::Type, "{:?}", e);
    assert!(e.unwrap_err().1.contains("cannot apply + (Plus)"));
    assert_eq!(a.run("h(:nosuch,,1)").unwrap_err().0, Kind::Name);
    for kind in ["syntax", "range", "system", "memory"] {
        let e = a.run(&format!("h(:raise,,\"{}\")", kind)).unwrap_err();
        assert_eq!(e.0.name(), kind);
    }
    assert_eq!(a.ok("h(:sq,,6)"), "36", "the connection survives errors");
    assert_eq!(a.ok(".clic(h)"), "1");
}

fn async_calls_and_hooks_body(baud: u32) {
    let _g = ONE.lock().unwrap_or_else(|e| e.into_inner());
    let b = Bench::new(2, opts(baud));
    let (a, c) = (b.node(0), b.node(1));
    serve(&b, 1);
    c.ok("opened::0;closed::0;.srv.o::{opened::opened+1};.srv.c::{closed::closed+1}");
    a.ok("h::.cli(\"node:1\")");
    // async: returns at once, the callback runs from the event loop
    a.ok("got::[];cb::{got::got,x};af::.async(h(:sq);cb);af(5);af(6)");
    assert_eq!(a.ok("got"), "[]", "did not wait");
    b.spin_until(5_000_000, || a.ok("#got") == "2").expect("both answers");
    assert_eq!(a.ok("got"), "[25 36]", "in order");
    assert_eq!(c.ok("opened,closed"), "[1 0]");
    a.ok(".clic(h)");
    b.spin_until(5_000_000, || c.ok("opened,closed") == "[1 1]").expect("the close reaches the server");
}

fn call_timeout_body(baud: u32) {
    let _g = ONE.lock().unwrap_or_else(|e| e.into_inner());
    let b = Bench::new(2, opts(baud));
    let (a, c) = (b.node(0), b.node(1));
    serve(&b, 1);
    a.ok("h::.cli(\"node:1\")");
    // a call that takes longer than its timeout fails and the connection is gone
    c.ok("wait::{[t];t::.pc()+x;{.pc()<t}{x}:~0;x}");
    a.ok(".clit(h;0.5)");
    let e = a.run("h(:wait,,2)").unwrap_err();
    assert_eq!(e.0, Kind::System);
    assert!(e.1.contains("timed out"), "{}", e.1);
}

fn four_nodes_both_directions_body(baud: u32) {
    let _g = ONE.lock().unwrap_or_else(|e| e.into_inner());
    let b = Bench::new(4, opts(baud));
    for i in 0..4 {
        serve(&b, i);
    }
    // every node calls every other one, and each pair is client and server of each other
    for i in 0..4 {
        let n = b.node(i);
        for j in 0..4 {
            if i != j {
                n.ok(&format!("c{}::.cli(\"node:{}\")", j, j));
            }
        }
    }
    for i in 0..4 {
        let n = b.node(i);
        for j in 0..4 {
            if i != j {
                assert_eq!(n.ok(&format!("c{}(:add,{},{})", j, i, j)), format!("{}", i + j), "{} -> {}", i, j);
            }
        }
    }
    // a relay: 0 asks 1 to ask 2 to ask 3
    b.node(1).ok("relay::{c2(:sq,,x)}");
    assert_eq!(b.node(0).ok("c1(:relay,,5)"), "25");
    // a sum over every node
    let mut total = 0;
    for j in 1..4 {
        total += b.node(0).ok(&format!("c{}(:total,,!{})", j, 10 * j)).parse::<i64>().unwrap();
    }
    assert_eq!(total, 45 + 190 + 435);
    println!("METRIC 4 nodes, 12 connections, all calls: virtual time {:.1} s, switch {:?}", b.now_us() as f64 / 1e6, (b.switch.stats().frames_in, b.switch.stats().overflow));
}

fn messages_larger_than_one_datagram_body(baud: u32) {
    let _g = ONE.lock().unwrap_or_else(|e| e.into_inner());
    let b = Bench::new(2, opts(baud));
    let a = b.node(0);
    serve(&b, 1);
    a.ok("h::.cli(\"node:1\")");
    // a vector of 600 integers is about 4.8 KB on the wire, 20 packets, far over 1024 bytes
    a.ok("big::!600");
    let t0 = b.now_us();
    assert_eq!(a.ok("r::h(:idf,,big);(r~big),#r"), "[1 600]");
    let dt = b.now_us() - t0;
    println!("METRIC echo of 600 integers (about 4.8 KB each way): {:.2} s virtual", dt as f64 / 1e6);
    // and a string over 1024 characters and a dictionary
    a.ok("s::2000#\"abcdefghij\"");
    assert_eq!(a.ok("(h(:idf,,s))~s"), "1");
    assert_eq!(a.ok("d::.clid(h);d,:mine,,s;(d?:mine)~s"), "1");
}

fn refused_connections_and_bad_addresses_body(baud: u32) {
    let _g = ONE.lock().unwrap_or_else(|e| e.into_inner());
    let b = Bench::new(3, opts(baud));
    let a = b.node(0);
    // nobody listens on node 1
    let e = a.run(".cli(\"node:1\")").unwrap_err();
    assert_eq!(e.0, Kind::System);
    println!("METRIC refused: {}", e.1);
    assert!(e.1.contains("reset by the peer") || e.1.contains("closed"), "{}", e.1);
    for bad in ["node:", "node:x", "node:99", "node:0", "node:1:u:p"] {
        let e = a.run(&format!(".cli(\"{}\")", bad)).unwrap_err();
        assert_eq!(e.0, Kind::System, "{}", bad);
        assert!(e.1.contains("bad address") || e.1.contains("cannot connect"), "{}: {}", bad, e.1);
    }
    // one connection of each kind per peer
    serve(&b, 1);
    a.ok("h::.cli(\"node:1\")");
    let e = a.run(".cli(\"node:1\")").unwrap_err();
    assert!(e.1.contains("in use"), "{}", e.1);
    assert_eq!(a.ok(".clic(h)"), "1");
    assert_eq!(a.ok("h2::.cli(\"node:1\");h2(:sq,,3)"), "9", "after closing, again");
    // two listeners are not allowed, as with tcp
    assert_eq!(b.node(1).ok(".srv(\"node:\")"), "0");
}

fn supervisor_pings_are_answered_and_everything_else_ignored_body(baud: u32) {
    let _g = ONE.lock().unwrap_or_else(|e| e.into_inner());
    let b = Bench::new(2, opts(baud));
    serve(&b, 1);
    let a = b.node(0);
    a.ok("h::.cli(\"node:1\")");
    // 5 seconds with a call in the middle of the pings; and frames the node does not know
    let mut io = b.switch.port(klong_dgram::frame::SUPERVISOR, b.clock.clone());
    use klong_dgram::frame::*;
    use klong_rt::node::LaneIo;
    for f in [Frame::new(T_STATUS, 16, 1, 1, 16, &[0, 1, 0, 0]), Frame::new(T_APP_DATA, 16, 1, 2, 0xa0, &[1, 2, 3, 4]), Frame::new(T_APP_DATA, 16, 1, 3, 0xe1, &[1, 0, 0, 0]), Frame::new(4, 16, 1, 4, 16, &[50])] {
        let bytes = f.encode().unwrap();
        let mut off = 0;
        while off < 16 {
            off += io.write(&bytes[off..]);
            b.spin(500);
        }
    }
    for _ in 0..5 {
        assert_eq!(a.ok("h(:sq,,3)"), "9");
        b.spin_ms(1000);
    }
    let st = b.switch.stats();
    println!("METRIC 5 s: {} pings, pongs {:?}, statuses {:?}", st.pings_sent, &st.pongs[..2], &st.statuses[..2]);
    assert!(st.pongs[0] >= 5 && st.pongs[1] >= 5, "{:?}", st.pongs);
    assert!(st.statuses[0] >= 4 && st.statuses[1] >= 4, "{:?}", st.statuses);
    assert_eq!(st.bad_destination, 0);
    assert_eq!(st.skipped_bytes, 0);
    assert_eq!(b.node(1).net.lane_stats().ctl_dropped, 0);
}

fn management_library_across_four_nodes_body(baud: u32) {
    let _g = ONE.lock().unwrap_or_else(|e| e.into_inner());
    let b = Bench::new(4, opts(baud));
    for i in 1..4 {
        serve(&b, i);
    }
    let m = b.node(0);
    m.ok(".l(\"mgmt\")");
    assert_eq!(m.ok("mgmt.add(\"n1\";\"node:1\");mgmt.add(\"n2\";\"node:2\");mgmt.add(\"n3\";\"node:3\");mgmt.names()"), "[\"n1\" \"n2\" \"n3\"]");
    assert_eq!(m.ok("{(x@1)?\"state\"}'mgmt.status()"), "[\"up\" \"up\" \"up\"]");
    for name in ["n1", "n2", "n3"] {
        assert_eq!(m.ok(&format!("mgmt.call(\"{name}\";[:sq]@0;[7])")), "49");
        let e = m.run(&format!("mgmt.call(\"{name}\";[:fail]@0;[1])")).unwrap_err();
        assert_eq!(e.0, Kind::Type);
        assert!(e.1.starts_with(&format!("{name}: cannot apply + (Plus)")), "{}", e.1);
    }
    let t0 = b.now_us();
    assert_eq!(m.ok("mgmt.all([:sq]@0;[7])"), ":{[\"n1\" 49] [\"n2\" 49] [\"n3\" 49]}");
    println!("METRIC mgmt.all over 3 nodes, small call: {:.0} ms virtual", ms(b.now_us() - t0));
    // liveness: the check runs every 2 seconds, with the supervisor's pings going by
    assert_eq!(m.ok("mgmt.start(2)"), "1");
    b.spin_ms(7000);
    assert_eq!(m.ok("{(x@1)?\"state\"}'mgmt.status()"), "[\"up\" \"up\" \"up\"]");
    assert_eq!(m.ok("mgmt.stop()"), "1");
    // a source text of 2000 characters to every node, then called there
    assert_eq!(m.ok("src::\"deployed::{x+1000};#\"\"\",(2000#\"klong \"),\"\"\"\";#src"), "2022");
    let t0 = b.now_us();
    assert_eq!(m.ok("mgmt.all(\".E\";,src)"), ":{[\"n1\" 2000] [\"n2\" 2000] [\"n3\" 2000]}");
    println!("METRIC mgmt.all: 2 KB deployed to each of 3 nodes: {:.2} s virtual", (b.now_us() - t0) as f64 / 1e6);
    assert_eq!(m.ok("mgmt.all([:deployed]@0;[1])"), ":{[\"n1\" 1001] [\"n2\" 1001] [\"n3\" 1001]}");
    // a node that is told off one at a time: its error is a value in the result
    b.node(2).ok("sq::{x+\"a\"}");
    let r = m.ok("mgmt.all([:sq]@0;[7])");
    assert!(r.contains("[\"n2\" :{[\"error\" \"type\"]"), "{}", r);
    assert!(r.contains("[\"n1\" 49]") && r.contains("[\"n3\" 49]"), "{}", r);
    assert_eq!(m.out(), "");
}

fn queue_overflow_and_retries_body(baud: u32) {
    // enough bytes that the three answers overlap for long: 9.6 KB at 115200, 48 KB at 4.5 Mbaud
    let n = if baud > 1_000_000 { 6000 } else { 1200 };
    let _g = ONE.lock().unwrap_or_else(|e| e.into_inner());
    let b = Bench::new(4, opts(baud));
    for i in 1..4 {
        serve(&b, i);
    }
    let m = b.node(0);
    m.ok(".l(\"mgmt\")");
    m.ok("mgmt.add(\"n1\";\"node:1\");mgmt.add(\"n2\";\"node:2\");mgmt.add(\"n3\";\"node:3\")");
    // the three nodes answer with 10 KB each at the same moment, into the one port of the
    // manager: three lanes into one, so its 256-frame queue overflows
    for i in 1..4 {
        b.node(i).ok(&format!("blob::!{}", n));
    }
    if baud > 1_000_000 {
        // with 126-byte packets and these timeouts three senders at 4.5 Mbaud no longer fill 256
        // frames (the run above measured a queue of 172); cut the port's queue so that it overflows
        b.switch.set_queue_cap(64);
    }
    let before = b.switch.stats();
    let t0 = b.now_us();
    let r = m.ok("r::mgmt.all([:total]@0;,!1);x::mgmt.all(\"blob\";[]);#x");
    assert_eq!(r, "3");
    let shown = m.ok("x");
    assert_eq!(m.ok(&format!("{{(x@1)~!{}}}'x", n)), "[1 1 1]", "{}", &shown[..shown.len().min(300)]);
    let st = b.switch.stats();
    println!("METRIC fan-in of 3 x {} bytes into one port (queue of {} frames): {:.2} s virtual, {} frames dropped by overflow, max queue {}", 8 * n, if baud > 1_000_000 { 64 } else { 256 }, (b.now_us() - t0) as f64 / 1e6, st.overflow - before.overflow, st.max_queue);
    assert!(st.overflow > before.overflow, "the queue should overflow");
    // with the random loss of a worse switch on top the results are still right
    b.switch.set_loss_ppm(3_000);
    m.ok("mgmt.timeout(\"n1\";120);mgmt.timeout(\"n2\";120);mgmt.timeout(\"n3\";120)");
    assert_eq!(m.ok(&format!("x::mgmt.all(\"blob\";[]);{{(x@1)~!{}}}'x", n)), "[1 1 1]");
    assert!(b.switch.stats().injected_loss > 0);
}

fn a_peer_that_resets_mid_message_body(baud: u32) {
    let _g = ONE.lock().unwrap_or_else(|e| e.into_inner());
    let b = Bench::new(2, opts(baud));
    let (a, c) = (b.node(0), b.node(1));
    serve(&b, 1);
    a.ok("h::.cli(\"node:1\");big::!800");
    assert_eq!(a.ok("h(:sq,,2)"), "4");
    // node 1 loses power part-way into a call that moves 6 KB each way
    let delay = if baud > 1_000_000 { 30_000 } else { 1_500_000 };
    b.reset_node_at(1, b.now_us() + delay);
    let t0 = b.now_us();
    let e = a.run("h(:idf,,big)").unwrap_err();
    println!("METRIC reset mid-message: the call fails after {:.2} s virtual with: {}", ms(b.now_us() - t0) / 1000.0, e.1);
    assert_eq!(e.0, Kind::System);
    assert!(e.1.contains("reset") || e.1.contains("closed") || e.1.contains("lost"), "{}", e.1);
    // the rebooted node has no state, numbers itself from the next ping, and can be served again
    let c2 = b.node(1);
    assert!(!std::rc::Rc::ptr_eq(&c, &c2));
    b.spin_until(3_000_000, || c2.net.node() == Some(1)).expect("numbered again");
    serve(&b, 1);
    let e = a.run("h(:sq,,3)").unwrap_err();
    assert_eq!(e.0, Kind::System, "the old connection is dead for good");
    assert_eq!(a.ok("h::.cli(\"node:1\");h(:sq,,3)"), "9");
    // a restart while idle: found by the next call, which fails once and then works again on a new connection
    b.reset_node(1);
    b.spin_until(3_000_000, || b.node(1).net.node() == Some(1)).expect("numbered again");
    serve(&b, 1);
    assert!(a.run("h(:sq,,4)").is_err());
    assert_eq!(a.ok("h::.cli(\"node:1\");h(:sq,,4)"), "16");
    // the client restarting: the server's old connection is replaced by the new one
    b.reset_node(0);
    b.spin_until(3_000_000, || b.node(0).net.node() == Some(0)).expect("numbered again");
    let a2 = b.node(0);
    assert_eq!(a2.ok("h::.cli(\"node:1\");h(:sq,,5)"), "25");
    assert!(b.node(1).net.counters().restarts >= 1);
}

fn dropped_frames_cost_time_not_correctness_body(baud: u32) {
    let _g = ONE.lock().unwrap_or_else(|e| e.into_inner());
    for ppm in [0u32, 2_000, 5_000] {
        let b = Bench::new(2, opts(baud));
        serve(&b, 1);
        let a = b.node(0);
        a.ok("h::.cli(\"node:1\");big::!300;.clit(h;120)");
        b.switch.set_loss_ppm(ppm);
        let t0 = b.now_us();
        for _ in 0..3 {
            assert_eq!(a.ok("(h(:idf,,big))~big"), "1");
        }
        println!("METRIC loss {} ppm per frame: 3 echoes of 2.4 KB: {:.2} s virtual, {} frames lost", ppm, (b.now_us() - t0) as f64 / 1e6, b.switch.stats().injected_loss);
    }
}

fn latency_and_throughput_body(baud: u32) {
    let _g = ONE.lock().unwrap_or_else(|e| e.into_inner());
    let b = Bench::new(2, opts(baud));
    serve(&b, 1);
    let a = b.node(0);
    let t0 = b.now_us();
    a.ok("h::.cli(\"node:1\")");
    println!("METRIC connect (SYN, SYN, hello, answer): {:.0} ms", ms(b.now_us() - t0));
    a.ok("h(:sq,,1)");
    let mut v = vec![];
    for k in 0..10 {
        let t0 = b.now_us();
        a.ok(&format!("h(:sq,,{})", k));
        v.push(b.now_us() - t0);
    }
    v.sort();
    println!("METRIC small call round trip: median {:.1} ms, max {:.1} ms (10 calls)", ms(v[5]), ms(v[9]));
    // bulk: one way (the answer is one integer), and echo
    for n in [100usize, 500, 1500] {
        a.ok(&format!("big::!{}", n));
        let t0 = b.now_us();
        a.ok("h(:count,,big)");
        let dt = b.now_us() - t0;
        let bytes = 8 * n + 30;
        println!("METRIC one-way call with {} integers (about {} bytes): {:.2} s virtual, {:.0} bytes/s", n, bytes, dt as f64 / 1e6, bytes as f64 / (dt as f64 / 1e6));
        let t0 = b.now_us();
        a.ok("h(:idf,,big)");
        let dt = b.now_us() - t0;
        println!("METRIC echo with {} integers: {:.2} s virtual, {:.0} bytes/s each way", n, dt as f64 / 1e6, bytes as f64 / (dt as f64 / 1e6));
    }
    let st = b.switch.stats();
    assert_eq!(st.overflow, 0);
    println!("METRIC switch: {} frames in, max queue {}", st.frames_in, st.max_queue);
}

fn console_channel_between_lanes_body(baud: u32) {
    let _g = ONE.lock().unwrap_or_else(|e| e.into_inner());
    // the supervisor types on channel 1 of node 1; the node's console answers on the same channel
    let b = Bench::new(2, opts(baud));
    let n = b.node(1);
    let mut buf = [0u8; 64];
    assert!(b.sup.send(1, 1, b"1+1\r", false).unwrap());
    let ok = b.spin_until(1_000_000, || n.net.console_peer().is_some());
    assert!(ok.is_some());
    let k = n.net.console_read(&mut buf);
    assert_eq!(&buf[..k], b"1+1\r");
    assert_eq!(n.net.console_peer(), Some(16));
    n.net.console_write(b"2\r\nklong> ");
    let mut got = vec![];
    b.spin_until(2_000_000, || {
        let k = b.sup.console_read(&mut buf);
        got.extend_from_slice(&buf[..k]);
        got.len() >= 9
    })
    .expect("the reply");
    assert_eq!(got, b"2\r\nklong> ");
    // another node's console traffic does not leak: node 0 has nobody attached
    assert_eq!(b.node(0).net.console_peer(), None);
    // detach with ^D: output stops going to the lane
    assert!(b.sup.send(1, 1, &[4], false).unwrap());
    b.spin_until(1_000_000, || n.net.console_peer().is_none()).expect("detached");
    n.net.console_write(b"nobody listens");
    b.spin_ms(300);
    assert_eq!(b.sup.console_read(&mut buf), 0);
}

both! {
    two_nodes_calls_values_errors => two_nodes_calls_values_errors_body,
    async_calls_and_hooks => async_calls_and_hooks_body,
    call_timeout => call_timeout_body,
    four_nodes_both_directions => four_nodes_both_directions_body,
    messages_larger_than_one_datagram => messages_larger_than_one_datagram_body,
    refused_connections_and_bad_addresses => refused_connections_and_bad_addresses_body,
    supervisor_pings_are_answered_and_everything_else_ignored => supervisor_pings_are_answered_and_everything_else_ignored_body,
    management_library_across_four_nodes => management_library_across_four_nodes_body,
    queue_overflow_and_retries => queue_overflow_and_retries_body,
    a_peer_that_resets_mid_message => a_peer_that_resets_mid_message_body,
    dropped_frames_cost_time_not_correctness => dropped_frames_cost_time_not_correctness_body,
    latency_and_throughput => latency_and_throughput_body,
    console_channel_between_lanes => console_channel_between_lanes_body,
}
