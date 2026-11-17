//! N simulated boards behind the packet switch of the Pico bench, driven by a script on
//! standard input, in virtual time (no real time passes):
//!
//!     klong-bench [N] [--loss PPM] [--seed S] [--no-pings] [--help] [--version]
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
//! Blank lines and lines that start with `#` are ignored. The default is 2 nodes. Every board
//! is `klong-rt` with the node transport (`.cli("node:K")`, `.srv("node:")`), the supervisor
//! pings each node once a second, and the lanes run at 115200 baud: see `docs/native/pico-bench.md`.
use std::io::BufRead;
use std::process::ExitCode;

use klong_sim::bench::{Bench, BenchOpts};

fn main() -> ExitCode {
    let mut n = 2usize;
    let mut opts = BenchOpts::default();
    let mut loss = 0u32;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--help" | "-h" => {
                println!("usage: klong-bench [N] [--loss PPM] [--seed S] [--no-pings]  (commands on standard input; see the docs)");
                return ExitCode::SUCCESS;
            }
            "--version" => {
                println!("klong-bench {}", env!("CARGO_PKG_VERSION"));
                return ExitCode::SUCCESS;
            }
            "--loss" => loss = args.next().and_then(|v| v.parse().ok()).unwrap_or(0),
            "--seed" => opts.seed = args.next().and_then(|v| v.parse().ok()).unwrap_or(opts.seed),
            "--no-pings" => opts.pings = false,
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
