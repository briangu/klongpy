//! The datagram console end to end in the simulator: a model of the supervisor's bridge
//! (`klong_sim::bridge`) in front of boards whose consoles take their input from channel 1 of the
//! lane: a REPL session, multi-line programs, errors, a runaway loop that is interrupted, long
//! output, upload and run, two nodes driven alternately, the ACK statuses, the self-test report
//! staying out of the console, and IPC between the nodes while a console is in use. At both lane rates.
use std::cell::Cell;
use std::rc::Rc;

use klong_dgram::crc::crc32;
use klong_sim::bench::*;
use klong_sim::bridge::{Bridge, EvalError, Sent};

static ONE: std::sync::Mutex<()> = std::sync::Mutex::new(());
const BAUDS: [u32; 2] = [115_200, 4_500_000];

struct Rig {
    b: Bench,
    br: Rc<Bridge>,
}

fn rig(n: usize, baud: u32, selftest: bool) -> Rig {
    let b = Bench::new(n, BenchOpts { baud, selftest, ..BenchOpts::default() });
    let br = Rc::new(Bridge::new(&b));
    // the supervisor's main loop: the bridge moves its queues whenever anything runs
    let br2 = br.clone();
    b.on_idle(move || br2.poll());
    b.spin_ms(2500); // the nodes are numbered, say STATUS, and are asked who they are
    br.poll();
    Rig { b, br }
}

/// What the real bridge puts in the transcript before the node's output: each line after a prompt.
fn echo_of(body: &[u8]) -> Vec<u8> {
    let mut text = body.to_vec();
    if text.last() != Some(&b'\n') {
        text.push(b'\n');
    }
    let mut e = Vec::new();
    for line in text.split_inclusive(|&b| b == b'\n') {
        e.extend_from_slice(b"klong> ");
        e.extend_from_slice(line);
    }
    e
}

impl Rig {
    /// Send input and wait for its `done`: (output since, without the echo, status).
    fn eval(&self, node: usize, text: &str) -> (String, String) {
        self.eval_bytes(node, text.as_bytes())
    }

    fn eval_bytes(&self, node: usize, body: &[u8]) -> (String, String) {
        let from = self.total(node);
        let id = self.send(node, body);
        let mut status = None;
        for _ in 0..200_000 {
            self.b.spin(500);
            let o = self.br.output(node, Some(from)).unwrap();
            assert!(!matches!(o.sent.1, Sent::Failed | Sent::Refused) || o.sent.0 != id, "input not delivered: {:?}", o.sent);
            if o.done.as_ref().is_some_and(|d| d.0 == id) {
                status = o.done.map(|d| d.1);
                break;
            }
        }
        let status = status.unwrap_or_else(|| panic!("no done {} from node {}", id, node));
        let mut data = Vec::new();
        let mut at = from;
        loop {
            let o = self.br.output(node, Some(at)).unwrap();
            if o.data.is_empty() {
                break;
            }
            data.extend(o.data);
            at = o.next;
        }
        // the transcript has the echo of the input first; the output is what follows it
        let echo = echo_of(body);
        assert!(data.starts_with(&echo), "no echo in {:?}", String::from_utf8_lossy(&data[..data.len().min(60)]));
        (String::from_utf8_lossy(&data[echo.len()..]).into_owned(), status)
    }

    fn total(&self, node: usize) -> u64 {
        self.br.output(node, Some(u64::MAX)).unwrap().next
    }

    /// Queue input (waiting while the queue is full) and wait for the node to have it.
    fn send(&self, node: usize, body: &[u8]) -> u16 {
        for _ in 0..10_000 {
            match self.br.eval(node, body) {
                Ok(id) => {
                    for _ in 0..20_000 {
                        let o = self.br.output(node, Some(u64::MAX)).unwrap();
                        if o.sent.0 == id {
                            match o.sent.1 {
                                Sent::Sending => self.b.spin(200),
                                Sent::Delivered => return id,
                                a => panic!("input not delivered: {:?}", a),
                            }
                        } else {
                            return id; // a later message went after it: it was delivered
                        }
                    }
                    panic!("no ACK");
                }
                Err(EvalError::Full) => self.b.spin(200),
                Err(e) => panic!("eval: {:?}", e),
            }
        }
        panic!("busy")
    }
}

#[test]
fn a_repl_session() {
    let _g = ONE.lock().unwrap_or_else(|e| e.into_inner());
    for baud in BAUDS {
        let r = rig(2, baud, false);
        // results, with neither the echo of the input nor a prompt
        assert_eq!(r.eval(1, "1+2\n"), ("3\n".into(), "ok".into()), "{}", baud);
        assert_eq!(r.eval(1, "x::5\n"), ("5\n".into(), "ok".into()));
        assert_eq!(r.eval(1, "x*x\n"), ("25\n".into(), "ok".into()));
        // several lines in one datagram are evaluated in order
        let (out, st) = r.eval(1, "a::2\nb::a+3\nb*10\n.p(\"hi\")\n");
        assert_eq!(st, "ok");
        assert!(out.starts_with("2\n5\n50\nhi\n"), "{:?}", out);
        // a multi-line function, and an entry that goes on over datagrams
        assert_eq!(r.eval(1, "f::{[t];t::x+1;\n   t*2}\nf(3)\n").1, "ok");
        assert_eq!(r.eval(1, "f(3)\n").0, "8\n");
        assert_eq!(r.eval(1, "g::{x+\n").1, "ok");
        assert_eq!(r.eval(1, "100}\ng(1)\n"), (":monad\n101\n".into(), "ok".into()));
        // the input datagrams were the only thing that took the node's time: the load is 0 again
        assert_eq!(r.br.output(1, Some(0)).unwrap().load, 0);
        // hello: what is on the node, in the nodes list
        let nodes = r.br.nodes();
        assert_eq!((nodes.len(), nodes[1].online, nodes[1].klong, nodes[1].load), (4, true, true, 0));
        assert!(nodes[1].info.starts_with("klong 2.0.0-rc.1 node=1 native int64 float64"), "{}", nodes[1].info);
        assert!(!nodes[3].online);
    }
}

#[test]
fn errors_are_text_on_channel_1_and_status_error() {
    let _g = ONE.lock().unwrap_or_else(|e| e.into_inner());
    for baud in BAUDS {
        let r = rig(2, baud, false);
        let (out, st) = r.eval(1, "1+\"a\"\n");
        assert_eq!(st, "error");
        assert!(out.contains("type error: cannot apply + (Plus)"), "{}", out);
        // the second line of a datagram runs after the first one failed; the status is `error`
        let (out, st) = r.eval(1, "1+\"a\"\n7\n");
        assert_eq!((out.ends_with("7\n"), st.as_str()), (true, "error"));
        assert_eq!(r.eval(1, "2+2\n"), ("4\n".into(), "ok".into()));
    }
}

#[test]
fn a_runaway_loop_is_interrupted() {
    let _g = ONE.lock().unwrap_or_else(|e| e.into_inner());
    for baud in BAUDS {
        let r = rig(2, baud, false);
        let from = r.br.output(1, Some(0)).unwrap().next;
        // (the events are set up before the input is sent: the evaluation does not return)
        // while it runs: load 100 in STATUS (seen by the bridge), then `int` from the bridge, from
        // the middle of the evaluation (the bridge is used from inside the board's idle hook)
        let load = Rc::new(Cell::new(0u8));
        let (l2, br2, st2) = (load.clone(), r.br.clone(), r.b.switch.clone());
        r.b.at(r.b.now_us() + 1_800_000, move || {
            br2.poll();
            l2.set(st2.stats().status_load[1]);
        });
        let br3 = r.br.clone();
        r.b.at(r.b.now_us() + 2_000_000, move || {
            let _ = br3.interrupt(1);
        });
        let id = r.send(1, b"{1}{x+1}:~0\n3+4\n");
        r.b.spin_ms(4000);
        assert_eq!(load.get(), 100, "STATUS load while evaluating");
        let o = r.br.output(1, Some(from)).unwrap();
        assert_eq!(o.done, Some((id, "interrupted".to_string())));
        let text = String::from_utf8_lossy(&o.data).into_owned();
        assert!(text.contains("interrupt error: interrupted"), "{}", text);
        // "3+4" came after the loop in the same datagram: discarded by the Ctrl-C rule
        assert!(!text.contains('7'), "{}", text);
        assert_eq!(r.br.output(1, Some(0)).unwrap().load, 0);
        assert_eq!(r.eval(1, "3+4\n"), ("7\n".into(), "ok".into()));
        // `int` when idle is ignored
        r.br.interrupt(1).unwrap();
        r.b.spin_ms(1500);
        assert_eq!(r.eval(1, "5\n"), ("5\n".into(), "ok".into()));
    }
}

#[test]
fn long_output_comes_in_datagrams_of_at_most_1024_bytes() {
    let _g = ONE.lock().unwrap_or_else(|e| e.into_inner());
    for baud in BAUDS {
        let r = rig(2, baud, false);
        r.b.sup.take_log();
        let (out, st) = r.eval(1, "!1500\n");
        let want = format!("[{}]\n", (0..1500).map(|i| i.to_string()).collect::<Vec<_>>().join(" "));
        assert_eq!((out.as_str(), st.as_str()), (want.as_str(), "ok"));
        assert!(want.len() > 5000);
        // a client that comes late gets the oldest byte the bridge kept and a notice
        let id = r.send(1, b"!3000\n");
        for _ in 0..40_000 {
            r.b.spin(500);
            if r.br.output(1, Some(u64::MAX)).unwrap().done.is_some_and(|d| d.0 == id) {
                break;
            }
        }
        let o = r.br.output(1, Some(0)).unwrap();
        assert!(o.lost);
        assert_eq!(o.data.len(), klong_sim::bridge::PER_RESPONSE, "at most 4 KB at a time");
        // asking again from where it left off goes on without a gap
        let o2 = r.br.output(1, Some(o.next)).unwrap();
        assert!(!o2.lost && o2.data.len() == klong_sim::bridge::PER_RESPONSE);
        assert!(r.br.output(1, None).unwrap().data.len() == klong_sim::bridge::PER_RESPONSE);
        // and one that stays up to date does not
        assert!(!r.br.output(1, Some(o2.next)).unwrap().lost);
    }
}

#[test]
fn upload_and_run_a_file_over_the_console() {
    let _g = ONE.lock().unwrap_or_else(|e| e.into_inner());
    for baud in BAUDS {
        let r = rig(2, baud, false);
        // `run`: the header, the node says `go`, the program follows
        let prog = b"sq::{x*x}\n.p(\"hello from the file\")\nsq(12)\n";
        let header = format!("\x01run {} {:08x}\n", prog.len(), crc32(prog));
        let (out, st) = r.eval(1, &header);
        assert!(out.contains("\x01go"), "{:?}", out);
        assert_eq!(st, "ok");
        let (out, st) = r.eval_bytes(1, prog);
        assert_eq!(st, "ok");
        assert!(out.contains("hello from the file\n") && out.contains("144"), "{:?}", out);
        assert!(out.contains("\x01ok"), "{:?}", out);
        // the program's definition stays: a following line uses it
        assert_eq!(r.eval(1, "sq(5)\n").0, "25\n");
        // `put`: lines of text, in datagrams that end at a line end and hold at most 1024 bytes with it: the
        // bridge adds a newline to input that has none, so a raw chunk cut anywhere else would be changed on the
        // way (binary files cannot be put through it). The console answers with the CRC-32 of what it wrote.
        let data: Vec<u8> = (0..300).flat_map(|i| format!("line {} of the file with some text in it\n", i).into_bytes()).collect();
        let header = format!("\x01put blob.txt {} {:08x}\n", data.len(), crc32(&data));
        let (out, _) = r.eval(1, &header);
        assert!(out.contains("\x01go"), "{:?}", out);
        let mut last = String::new();
        let mut chunk: Vec<u8> = Vec::new();
        let mut parts = 0;
        for line in data.split_inclusive(|&b| b == b'\n') {
            if chunk.len() + line.len() > 1024 {
                last = r.eval_bytes(1, &chunk).0;
                chunk.clear();
                parts += 1;
            }
            chunk.extend_from_slice(line);
        }
        last = if chunk.is_empty() { last } else { parts += 1; r.eval_bytes(1, &chunk).0 };
        assert!(parts >= 10);
        assert!(last.contains(&format!("\x01ok {} {:08x}", data.len(), crc32(&data))), "{:?}", last);
    }
}

#[test]
fn two_nodes_driven_alternately_have_their_own_output() {
    let _g = ONE.lock().unwrap_or_else(|e| e.into_inner());
    for baud in BAUDS {
        let r = rig(3, baud, false);
        for k in 0..3 {
            assert_eq!(r.eval(1, &format!("n::{}\n", k)).1, "ok");
            assert_eq!(r.eval(2, &format!("n::{}\n", 10 + k)).1, "ok");
            assert_eq!(r.eval(1, "n\n").0, format!("{}\n", k));
            assert_eq!(r.eval(2, "n\n").0, format!("{}\n", 10 + k));
        }
        // the counters are per node
        let (a, b) = (r.br.output(1, Some(0)).unwrap(), r.br.output(2, Some(0)).unwrap());
        assert!(a.next > 0 && b.next > 0 && a.data != b.data);
        // ids come from the supervisor's one counter: consecutive across nodes
        let i1 = r.send(1, b"1\n");
        let i2 = r.send(2, b"2\n");
        assert_eq!(i2, i1.wrapping_add(1));
    }
}

#[test]
fn a_second_sender_is_refused_while_an_evaluation_runs() {
    let _g = ONE.lock().unwrap_or_else(|e| e.into_inner());
    for baud in BAUDS {
        let r = rig(3, baud, false);
        let from = r.br.output(1, Some(0)).unwrap().next;
        // node 2 types into node 1's console while it is busy for the supervisor: ACK status 2
        let n2 = r.b.node(2);
        let res: Rc<Cell<Option<&'static str>>> = Rc::new(Cell::new(None));
        let res2 = res.clone();
        let net2 = n2.net.clone();
        r.b.at(r.b.now_us() + 1_000_000, move || {
            assert!(net2.send(1, 1, b"2+2\n", true).unwrap());
        });
        let net3 = n2.net.clone();
        r.b.at(r.b.now_us() + 1_600_000, move || {
            // (the first outcomes may be those of replies it sent on its own; look for the one to node 1)
            while let Some(x) = net3.take_send_result() {
                if x.dest == 1 {
                    res2.set(Some(match x.outcome {
                        klong_dgram::Outcome::Refused => "refused",
                        klong_dgram::Outcome::Acked => "delivered",
                        _ => "timeout",
                    }));
                }
            }
        });
        let br = r.br.clone();
        r.b.at(r.b.now_us() + 2_500_000, move || {
            let _ = br.interrupt(1);
        });
        let id = r.send(1, b"{1}{x+1}:~0\n");
        r.b.spin_ms(5000);
        assert_eq!(res.get(), Some("refused"), "the second address is refused with ACK 2");
        let o = r.br.output(1, Some(from)).unwrap();
        assert_eq!(o.done, Some((id, "interrupted".to_string())));
        assert!(!String::from_utf8_lossy(&o.data).contains('4'), "its input never reached the console");
        // the output still goes to the supervisor, which owns the console
        assert_eq!(n2.net.console_peer(), None);
        // when it is idle another node may type, and takes the output away from the supervisor
        let total = r.br.output(1, Some(0)).unwrap().next;
        assert!(n2.net.send(1, 1, b"6*7\n", true).unwrap());
        r.b.spin_ms(1500);
        assert_eq!(r.b.node(1).net.console_peer(), Some(2));
        assert_eq!(r.br.output(1, Some(0)).unwrap().next, total, "the page sees nothing of it");
        // and the supervisor takes it back with its next input
        let got = r.eval(1, "8\n");
        assert_eq!(got, ("8\n".into(), "ok".into()));
    }
}

#[test]
fn the_bridge_codes() {
    let _g = ONE.lock().unwrap_or_else(|e| e.into_inner());
    let r = rig(3, 115_200, false);
    // 413 over 1024 bytes (with the newline that is added), 400 for nothing, 400 for a node that does not exist
    assert_eq!(r.br.eval(1, &vec![b'1'; 1024]).unwrap_err(), EvalError::TooLong);
    assert_eq!(r.br.eval(1, b"").unwrap_err(), EvalError::Empty);
    assert_eq!(r.br.eval(9, b"1\n").unwrap_err(), EvalError::BadNode);
    assert!(r.br.eval(1, &vec![b' '; 1023]).is_ok()); // plus its newline: 1024
    r.b.spin_ms(1500);
    // a newline is added, and the transcript has the echo
    let (out, st) = r.eval(1, "6*7");
    assert_eq!((out.as_str(), st.as_str()), ("42\n", "ok"));
    // 429: the queue holds two (an evaluation and an interrupt)
    assert!(r.br.eval(1, b"1\n").is_ok());
    assert!(r.br.interrupt(1).is_ok());
    assert_eq!(r.br.eval(1, b"2\n").unwrap_err(), EvalError::Full);
    r.b.spin_ms(1500);
    // 503: a node that is gone (no STATUS for 3 s) is offline; node 3 does not exist as a board; a node that does
    // not run Klong is another 503 text (the STATUS flag 0x10 is clear)
    r.b.node(2).mute.set(true);
    r.b.spin_ms(4000);
    assert_eq!(r.br.eval(2, b"1\n").unwrap_err(), EvalError::Offline);
    assert_eq!(r.br.interrupt(2).unwrap_err(), EvalError::Offline);
    assert_eq!(r.br.eval(3, b"1\n").unwrap_err(), EvalError::Offline);
    let o = r.br.output(2, Some(0)).unwrap();
    assert!(!o.online && !o.klong, "output works for an offline node, and says so");
    assert_eq!(r.br.nodes().iter().map(|n| (n.online, n.klong)).collect::<Vec<_>>(), vec![(true, true), (true, true), (false, false), (false, false)]);
    // an input that does not arrive: `failed` after the resends; no `done` ever comes
    r.b.spin_ms(500);
    r.b.switch.set_loss_ppm(1_000_000);
    let id = r.br.eval(1, b"1\n").unwrap();
    r.b.spin_ms(4000);
    let o = r.br.output(1, Some(0)).unwrap();
    assert_eq!(o.sent, (id, Sent::Failed));
    r.b.switch.set_loss_ppm(0);
}

#[test]
fn the_selftest_report_stays_out_of_the_console_and_ipc_goes_on() {
    let _g = ONE.lock().unwrap_or_else(|e| e.into_inner());
    for baud in BAUDS {
        let r = rig(2, baud, true);
        r.b.spin_ms(20_000);
        let nodes = r.br.nodes();
        assert!(nodes[1].selftest.starts_with("klong selftest pass "), "{}", nodes[1].selftest);
        assert!(nodes[1].klong);
        // not one byte of console output for it
        assert_eq!(r.br.output(1, Some(0)).unwrap().next, 0);
        assert_eq!(r.br.output(0, Some(0)).unwrap().next, 0);
        // IPC between the nodes while a console is in use: node 0 is typed at and calls node 1
        let (out, status) = r.eval(0, "h::.cli(\"node:1\");h(:stping1,,7)\n");
        assert_eq!((out.as_str(), status.as_str()), ("7\n", "ok"));
        // and the other way: node 1's console is used while node 0 calls it from its own console
        assert_eq!(r.eval(1, "hits\n").1, "ok");
        assert_eq!(r.eval(0, "h(:stecho1,,!100)~!100\n").0, "1\n");
        assert_eq!(r.eval(0, ".clic(h)\n").1, "ok");
    }
}

#[test]
fn output_with_nobody_to_send_it_to_is_dropped_and_counted() {
    let _g = ONE.lock().unwrap_or_else(|e| e.into_inner());
    let r = rig(2, 115_200, false);
    let n = r.b.node(1);
    // a timer prints on a node that nobody has typed at
    n.ok("cb::{.p(\"tick\");1}");
    n.ok(".timer(\"t\";0.2;cb)");
    r.b.spin_ms(1500);
    assert!(n.net.counters().console_no_peer >= 20, "{:?}", n.net.counters());
    assert_eq!(r.br.output(1, Some(0)).unwrap().next, 0);
    // after somebody types, the timer's output goes to him
    assert_eq!(r.eval(1, "1\n").1, "ok");
    r.b.spin_ms(1500);
    let o = r.br.output(1, Some(0)).unwrap();
    assert!(String::from_utf8_lossy(&o.data).contains("tick"), "{:?}", o.data.len());
    n.ok(".timerc(\"t\")");
}

/// Seen on the bench: after the boot self-test between two nodes, the first console evaluations were
/// echoed and followed by a prompt, on one node and then the other, because input that arrives while a
/// turn of the board's loop is under way (after the console looked at the lane, before it read) was
/// not treated as a program's. The real sequence: self-test, then input to each node at irregular times.
#[test]
fn after_the_selftest_no_echo_and_no_prompt_whenever_the_input_arrives() {
    let _g = ONE.lock().unwrap_or_else(|e| e.into_inner());
    for baud in BAUDS {
        let r = rig(3, baud, true);
        r.b.spin_ms(8000); // the self-test of node 1 and 2 runs and passes
        let mut k = 0u64;
        for round in 0..60u64 {
            for node in [1usize, 2] {
                // an irregular gap, from nothing to a few frame times, so that the datagram's frames land at
                // every point of a turn
                r.b.spin(37 * ((round * 7 + node as u64 * 3) % 29));
                k += 1;
                let (out, st) = r.eval(node, &format!("{}+{}\n", k, k));
                assert_eq!((out.as_str(), st.as_str()), (format!("{}\n", 2 * k).as_str(), "ok"), "{} baud, round {}, node {}", baud, round, node);
            }
        }
        // the transcript is plain text: no CR, no prompt anywhere
        for node in [1usize, 2] {
            let o = r.br.output(node, None).unwrap();
            let text = String::from_utf8_lossy(&o.data).into_owned();
            assert!(!text.contains('\r') && !text.contains("klong> klong>"), "{}", text);
        }
        // a board's console writes CR LF for a terminal; a datagram peer is sent LF
        r.b.node(1).net.console_write(b"x\r\ny\r");
        r.b.spin_ms(500);
        let o = r.br.output(1, None).unwrap();
        assert!(String::from_utf8_lossy(&o.data).ends_with("x\ny\r"), "{:?}", String::from_utf8_lossy(&o.data[o.data.len().saturating_sub(10)..]));
    }
}
