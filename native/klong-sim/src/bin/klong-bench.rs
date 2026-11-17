//! N simulated boards behind the packet switch of the Pico bench, driven by a script on
//! standard input, in virtual time (no real time passes):
//!
//!     klong-bench [N] [--baud BAUD] [--loss PPM] [--seed S] [--no-pings] [--help] [--version]
//!
//! One command per line:
//!
//!     K> CODE         evaluate CODE on node K (0-based) and print `K: value`, or `K: error: kind: message`
//!     spin SECONDS    let every node and the switch run for that long
//!     reset K         node K loses power and boots again
//!     loss PPM        the switch drops this many frames per million, at random
//!     time            print the virtual time
//!     stats           print the switch's counters
//!
//! `--bridge` runs the model of the supervisor's console bridge instead (`klong_sim::bridge`; the
//! protocol of `Bridge::handle`, one request and one answer line each, on standard input and output)
//! in front of the boards: `native/tools/bench_bridge_sim.py` puts the HTTP routes on it. `--selftest`
//! runs the boot self-test on every board, `--mute K` makes node K a board with other firmware.
//!
//! Blank lines and lines that start with `#` are ignored. The default is 2 nodes. Every board
//! is `klong-rt` with the node transport (`.cli("node:K")`, `.srv("node:")`), the supervisor
//! pings each node once a second, and the lanes run at 115200 baud unless `--baud` says otherwise: see `docs/native/pico-bench.md`.
use std::io::{BufRead, Write};
use std::rc::Rc;
use std::sync::mpsc::{Receiver, RecvTimeoutError};
use std::process::ExitCode;

use klong_sim::bench::{Bench, BenchOpts};

fn main() -> ExitCode {
    let mut n = 2usize;
    let mut opts = BenchOpts::default();
    let mut loss = 0u32;
    let (mut bridge_mode, mut mute) = (false, Vec::new());
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--help" | "-h" => {
                println!("usage: klong-bench [N] [--baud BAUD] [--loss PPM] [--seed S] [--no-pings]  (commands on standard input; see the docs)");
                return ExitCode::SUCCESS;
            }
            "--version" => {
                println!("klong-bench {}", env!("CARGO_PKG_VERSION"));
                return ExitCode::SUCCESS;
            }
            "--loss" => loss = args.next().and_then(|v| v.parse().ok()).unwrap_or(0),
            "--seed" => opts.seed = args.next().and_then(|v| v.parse().ok()).unwrap_or(opts.seed),
            "--no-pings" => opts.pings = false,
            "--bridge" => bridge_mode = true,
            "--selftest" => opts.selftest = true,
            "--mute" => mute.extend(args.next().and_then(|v| v.parse::<usize>().ok())),
            "--baud" => opts.baud = args.next().and_then(|v| v.parse().ok()).unwrap_or(opts.baud),
            v => match v.parse::<usize>() {
                Ok(k) if (1..=16).contains(&k) => n = k,
                _ => {
                    eprintln!("klong-bench: bad argument {v}; try --help");
                    return ExitCode::from(2);
                }
            },
        }
    }
    let bench = Bench::new(n, opts);
    bench.switch.set_loss_ppm(loss);
    for k in mute {
        if k < bench.len() {
            bench.node(k).mute.set(true);
        }
    }
    if bridge_mode {
        return run_bridge(bench);
    }
    let mut failed = false;
    for line in std::io::stdin().lock().lines() {
        let Ok(line) = line else { break };
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some((k, code)) = line.split_once("> ") {
            if let Ok(k) = k.trim().parse::<usize>() {
                if k >= bench.len() {
                    println!("no node {k}");
                    failed = true;
                    continue;
                }
                match bench.node(k).run(code) {
                    Ok(v) => println!("{k}: {v}"),
                    Err((kind, msg)) => println!("{k}: error: {}: {}", kind.name(), msg),
                }
                let out = bench.node(k).out();
                if !out.is_empty() {
                    for l in out.lines() {
                        println!("{k}| {l}");
                    }
                }
                continue;
            }
        }
        let mut w = line.split_whitespace();
        match (w.next(), w.next()) {
            (Some("spin"), Some(s)) => match s.parse::<f64>() {
                Ok(s) => bench.spin((s * 1e6) as u64),
                Err(_) => failed = true,
            },
            (Some("reset"), Some(k)) => match k.parse::<usize>() {
                Ok(k) if k < bench.len() => bench.reset_node(k),
                _ => failed = true,
            },
            (Some("loss"), Some(p)) => bench.switch.set_loss_ppm(p.parse().unwrap_or(0)),
            (Some("time"), None) => println!("t = {:.3} s", bench.now_us() as f64 / 1e6),
            (Some("stats"), None) => {
                let s = bench.switch.stats();
                println!("frames in {}, out {}, dropped by overflow {}, injected loss {}, max queue {}", s.frames_in, s.frames_out, s.overflow, s.injected_loss, s.max_queue);
            }
            _ => {
                println!("what? {line}");
                failed = true;
            }
        }
    }
    if failed {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}

/// The bridge model on a pipe. Virtual time moves when a request says `advance MS` (the program that
/// puts HTTP on it maps real time to it) and while a board evaluates; requests that arrive while a board
/// is inside an evaluation (an interrupt for a runaway loop, a poll of the output) are answered from the
/// boards' idle hook.
fn run_bridge(bench: Bench) -> ExitCode {
    let bridge = Rc::new(klong_sim::bridge::Bridge::new(&bench));
    let (tx, rx) = std::sync::mpsc::channel::<String>();
    let reader = std::thread::spawn(move || {
        for line in std::io::stdin().lock().lines() {
            let Ok(line) = line else { break };
            if tx.send(line).is_err() {
                break;
            }
        }
    });
    let rx: Rc<Receiver<String>> = Rc::new(rx);
    let answer = |text: String| {
        let mut o = std::io::stdout().lock();
        let _ = writeln!(o, "{}", text);
        let _ = o.flush();
    };
    {
        let (br, rx) = (bridge.clone(), rx.clone());
        bench.on_idle(move || {
            br.poll();
            loop {
                let line = match rx.try_recv() {
                    Ok(l) => l,
                    Err(std::sync::mpsc::TryRecvError::Empty) => break,
                    // whoever drives this program is gone: do not run on, wherever the boards are
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => std::process::exit(0),
                };
                let r = br.handle(&line, None);
                if !r.is_empty() {
                    let mut o = std::io::stdout().lock();
                    let _ = writeln!(o, "{}", r);
                    let _ = o.flush();
                }
            }
        });
    }
    println!("ready {}", bench.len());
    let _ = std::io::stdout().flush();
    loop {
        match rx.recv_timeout(std::time::Duration::from_millis(50)) {
            Ok(line) => {
                if line.trim() == "quit" {
                    answer("ok".into());
                    break;
                }
                let r = bridge.handle(&line, Some(&bench));
                if !r.is_empty() {
                    answer(r);
                }
            }
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => break,
        }
    }
    drop(reader);
    ExitCode::SUCCESS
}
