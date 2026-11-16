//! The `klong` binary.
//!
//!     klong FILE                      run a file
//!     klong -e EXPR                   evaluate and print
//!     klong                           read-eval-print loop
//!     klong --dump-ast FILE           canonical parse tree (frontend.md section 7)
//!     klong --conformance [--strict N]
//!                                     the runner protocol of conformance/README.md: program
//!                                     on stdin; printed result on stdout and exit 0, or
//!                                     `error: <kind>` and exit 1; console output on stderr
//!
//! Options: --strict N, --heap-limit BYTES (K and M suffixes), --max-depth N, --steps N,
//! --fail-every N (allocation failure injection), --stack BYTES (native stack of the
//! evaluating thread, default 512M), --stats (heap and depth figures on stderr),
//! --bench N (with -e: best in-process time of N evaluations on stderr).
use std::alloc::System;
use std::io::{BufRead, Read, Write};
use std::process::ExitCode;

use klong_core::{Config, Error};
use klong_host::{Console, HostPlatform};
use klong_rt::heap::Counting;
use klong_rt::Runtime;

#[global_allocator]
static HEAP: Counting<System> = Counting::new(System);

/// Held back below the heap limit so that an error can unwind and print.
const RESERVE: usize = 16 * 1024;

fn mem_stats() -> (usize, usize, usize) {
    (HEAP.used(), HEAP.peak(), HEAP.limit())
}

fn mem_pressed() -> bool {
    HEAP.pressed()
}

#[derive(Default)]
struct Opts {
    conformance: bool,
    stats: bool,
    strict: u8,
    dump_ast: Option<String>,
    expr: Option<String>,
    file: Option<String>,
    heap_limit: usize,
    max_depth: Option<u32>,
    steps: Option<u64>,
    fail_every: usize,
    stack: Option<usize>,
    bench: u32,
}

fn size_arg(s: &str) -> Option<usize> {
    let (digits, mult) = match s.as_bytes().last()? {
        b'k' | b'K' => (&s[..s.len() - 1], 1024),
        b'm' | b'M' => (&s[..s.len() - 1], 1024 * 1024),
        _ => (s, 1),
    };
    digits.parse::<usize>().ok()?.checked_mul(mult)
}

fn usage() -> ExitCode {
    eprintln!("usage: klong [--strict N] [--heap-limit BYTES] [--max-depth N] [--steps N] [--stats] [FILE | -e EXPR | --conformance | --dump-ast FILE]");
    ExitCode::from(2)
}

fn parse_args() -> Option<Opts> {
    let mut o = Opts::default();
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--conformance" => o.conformance = true,
            "--stats" => o.stats = true,
            "--strict" => o.strict = it.next()?.parse().ok().filter(|n| *n <= 2)?,
            "--dump-ast" => o.dump_ast = Some(it.next()?),
            "-e" => o.expr = Some(it.next()?),
            "--heap-limit" => o.heap_limit = size_arg(&it.next()?)?,
            "--max-depth" => o.max_depth = Some(it.next()?.parse().ok()?),
            "--steps" => o.steps = Some(it.next()?.parse().ok()?),
            "--bench" => o.bench = it.next()?.parse().ok()?,
            "--stack" => o.stack = Some(size_arg(&it.next()?)?),
            "--fail-every" => o.fail_every = it.next()?.parse().ok()?,
            s if s.starts_with('-') && s.len() > 1 => return None,
            _ if o.file.is_none() => o.file = Some(a),
            _ => return None,
        }
    }
    Some(o)
}

fn runtime(o: &Opts, console: Console) -> Runtime<HostPlatform> {
    let mut cfg = Config { strict: o.strict, step_limit: o.steps, mem_probe: Some(mem_pressed), ..Config::default() };
    if let Some(d) = o.max_depth {
        cfg.max_depth = d;
    }
    Runtime::new(HostPlatform::new(console), cfg, Some(mem_stats))
}

fn report(e: &Error) {
    eprintln!("{}", e.render());
}

fn stats(rt: &Runtime<HostPlatform>) {
    eprintln!("heap: peak {} bytes, in use {} bytes, limit {}, refused {}", HEAP.peak(), HEAP.used(), HEAP.limit(), HEAP.failed());
    eprintln!("depth: {} nested evaluations at most", rt.interp.max_depth_seen);
}

fn run(o: Opts) -> ExitCode {
    if let Some(path) = &o.dump_ast {
        let text = match std::fs::read(path).map(String::from_utf8) {
            Ok(Ok(t)) => t,
            _ => {
                println!("error: system");
                return ExitCode::from(1);
            }
        };
        let mut rt = runtime(&o, Console::Stderr);
        return match rt.interp.parse(&text, path) {
            Ok(p) => {
                let s = klong_core::dump::dump(&p.tree, &rt.interp.syms, &p.stmts);
                let _ = std::io::stdout().write_all(s.as_bytes());
                ExitCode::SUCCESS
            }
            Err(e) => {
                println!("error: {}", e.kind.name());
                ExitCode::from(1)
            }
        };
    }

    if o.conformance {
        let mut bytes = Vec::new();
        let text = match std::io::stdin().read_to_end(&mut bytes).map(|_| String::from_utf8(bytes)) {
            Ok(Ok(t)) => t,
            _ => {
                println!("error: system");
                return ExitCode::from(1);
            }
        };
        let mut rt = runtime(&o, Console::Stderr);
        // the limits apply to the program, not to starting up
        HEAP.reset_peak();
        HEAP.set_limit(if o.heap_limit == 0 { 0 } else { HEAP.used() + o.heap_limit }, RESERVE);
        HEAP.fail_every(o.fail_every);
        let r = rt.run(&text, "<stdin>");
        HEAP.fail_every(0);
        HEAP.set_limit(0, 0);
        let code = match r {
            Ok(Some(v)) => {
                let s = rt.interp.format(&v);
                if !s.is_empty() {
                    println!("{}", s);
                }
                ExitCode::SUCCESS
            }
            Ok(None) => ExitCode::SUCCESS,
            Err(e) => {
                println!("error: {}", e.kind.name());
                ExitCode::from(1)
            }
        };
        if o.stats {
            stats(&rt);
        }
        return code;
    }

    let mut rt = runtime(&o, Console::Stdout);
    HEAP.reset_peak();
    HEAP.set_limit(if o.heap_limit == 0 { 0 } else { HEAP.used() + o.heap_limit }, RESERVE);
    HEAP.fail_every(o.fail_every);

    if let Some(expr) = &o.expr {
        if o.bench > 0 {
            // time of one evaluation (parse and run), in process: the best of N
            let mut best = u128::MAX;
            for _ in 0..o.bench {
                let t = std::time::Instant::now();
                let r = rt.run(expr, "<-e>");
                best = best.min(t.elapsed().as_nanos());
                drop(r);
            }
            eprintln!("bench: {} ns", best);
        }
        let r = rt.run(expr, "<-e>");
        HEAP.set_limit(0, 0);
        HEAP.fail_every(0);
        let code = match r {
            Ok(Some(v)) => {
                println!("{}", rt.interp.format(&v));
                ExitCode::SUCCESS
            }
            Ok(None) => ExitCode::SUCCESS,
            Err(e) => {
                report(&e);
                ExitCode::from(1)
            }
        };
        if o.stats {
            stats(&rt);
        }
        return code;
    }

    if let Some(path) = &o.file {
        let text = match std::fs::read_to_string(path) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("klong: {}: {}", path, e);
                return ExitCode::from(1);
            }
        };
        let r = rt.run(&text, path);
        HEAP.set_limit(0, 0);
        HEAP.fail_every(0);
        let code = match r {
            Ok(_) => ExitCode::SUCCESS,
            Err(e) => {
                report(&e);
                ExitCode::from(1)
            }
        };
        if o.stats {
            stats(&rt);
        }
        return code;
    }

    // read-eval-print loop: one line, one program
    let stdin = std::io::stdin();
    let mut line = String::new();
    loop {
        print!("  ");
        let _ = std::io::stdout().flush();
        line.clear();
        match stdin.lock().read_line(&mut line) {
            Ok(0) | Err(_) => break,
            Ok(_) => {}
        }
        let src = line.trim_end_matches(['\n', '\r']);
        if src == "]q" {
            break;
        }
        match rt.run(src, "<repl>") {
            Ok(Some(v)) => println!("{}", rt.interp.format(&v)),
            Ok(None) => {}
            Err(e) => report(&e),
        }
    }
    ExitCode::SUCCESS
}

fn main() -> ExitCode {
    let o = match parse_args() {
        Some(o) => o,
        None => return usage(),
    };
    // The evaluator recurses on the native stack and bounds it with Config::max_depth. The
    // main thread's stack is whatever the system gives; a thread with a stack of known size
    // makes the bound hold.
    let stack = o.stack.unwrap_or(512 * 1024 * 1024);
    match std::thread::Builder::new().stack_size(stack).spawn(move || run(o)) {
        Ok(h) => h.join().unwrap_or(ExitCode::from(70)),
        Err(_) => ExitCode::from(70),
    }
}
