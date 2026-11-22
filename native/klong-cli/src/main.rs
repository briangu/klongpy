//! The `kg` binary.
//!
//!     kg FILE                      run a file
//!     kg -e EXPR                   evaluate and print
//!     kg                              read-eval-print loop
//!     kg --dump-ast FILE           canonical parse tree (conformance/README.md)
//!     kg ast FILE|-e EXPR          the same, as `kgpy ast` prints it
//!     kg --conformance [--strict N]
//!                                     the runner protocol of conformance/README.md: program
//!                                     on stdin; printed result on stdout and exit 0, or
//!                                     `error: <kind>` and exit 1; console output on stderr
//!
//! Options: --strict N, --heap-limit BYTES (K and M suffixes), --max-depth N, --steps N,
//! --fail-every N (allocation failure injection), --stack BYTES (native stack of the
//! evaluating thread, default 64M), --stack-budget BYTES (configure the interpreter as a board
//! with a task stack of that size would: Config::for_stack), --max-nest N (data nesting),
//! --stats (heap, depth and stack figures on stderr), --bench N (with -e: best in-process
//! time of N evaluations on stderr), --memfs (the file functions work on an empty file system
//! in memory instead of the real one: for fuzzing). After FILE, further arguments are the program's (.a).
//!
//! `.l` looks in the directories of the environment variable KLONGPATH (default the working directory and "lib"),
//! then in the library built into the binary, and last in the directory of the file that
//! is being run or loaded.
use std::alloc::System;
use std::io::{BufRead, Read, Write};
use std::process::ExitCode;

mod check;
mod compile;
mod gpu;
mod kgpy_only;
mod profile_doc;
mod lint;

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
    capabilities: bool,
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
    stack_budget: Option<usize>,
    max_nest: Option<u32>,
    bench: u32,
    memfs: bool,
    device: Option<String>,
    args: Vec<String>,
}

fn size_arg(s: &str) -> Option<usize> {
    let (digits, mult) = match s.as_bytes().last()? {
        b'k' | b'K' => (&s[..s.len() - 1], 1024),
        b'm' | b'M' => (&s[..s.len() - 1], 1024 * 1024),
        _ => (s, 1),
    };
    digits.parse::<usize>().ok()?.checked_mul(mult)
}

#[cfg(all(target_os = "macos", feature = "metal"))]
macro_rules! device_help {
    () => { "  --device NAME         numeric device (cpu, or mps[:INDEX] on macOS)\n" };
}

#[cfg(not(all(target_os = "macos", feature = "metal")))]
macro_rules! device_help {
    () => { "  --device NAME         numeric device (cpu)\n" };
}

const HELP: &str = concat!("\
usage: kg [OPTIONS] [FILE [ARG ...] | -e EXPR | --conformance | --dump-ast FILE]

Without FILE or -e: a read-eval-print loop on standard input.
  FILE [ARG ...]        run a file; the arguments are in .a
  -e EXPR               evaluate EXPR and print its value
  --conformance         the runner protocol of conformance/README.md (program on stdin)
  --dump-ast FILE       print the canonical parse tree of FILE and exit

  --capabilities        print .ver() as one JSON object and exit

Tools: kg lint, kg ast, kg check, kg compile c, kg fpga source

Options (before FILE or -e; sizes accept the suffixes K and M):
  --strict N            strict level 0, 1 or 2
", device_help!(), "\
  --heap-limit BYTES    most the program may allocate (default: no limit)
  --max-depth N         most nested evaluations before a range error (default 512)
  --steps N             stop after about N calls and adverb iterations (interrupt)
  --stack BYTES         native stack of the evaluating thread (default 64M)
  --stack-budget BYTES  set the limits as a board with a task stack of this size would
  --max-nest N          most levels of nested lists (default 128)
  --stats               print peak heap, nesting and stack figures on standard error
  --bench N             with -e: best time of N runs on standard error, in nanoseconds
  --memfs               the file functions work on an empty file system in memory
  --fail-every N        make every Nth large allocation fail (out-of-memory testing)
  -h, --help            print this text and exit
  --version             print the version and exit
");

fn usage() -> ExitCode {
    eprint!("{}", HELP);
    ExitCode::from(2)
}

fn parse_args() -> Option<Opts> {
    let mut o = Opts::default();
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        if o.file.is_some() {
            o.args.push(a);
            continue;
        }
        match a.as_str() {
            "--capabilities" => o.capabilities = true,
            // (before FILE only: after it the arguments are the program's)
            "-h" | "--help" if o.file.is_none() => {
                print!("{}", HELP);
                std::process::exit(0);
            }
            "--version" if o.file.is_none() => {
                println!("kg {}", env!("CARGO_PKG_VERSION"));
                std::process::exit(0);
            }
            "--conformance" => o.conformance = true,
            "--stats" => o.stats = true,
            "--memfs" => o.memfs = true,
            "--device" if o.file.is_none() => o.device = Some(it.next()?),
            "--strict" => o.strict = it.next()?.parse().ok().filter(|n| *n <= 2)?,
            "--dump-ast" => o.dump_ast = Some(it.next()?),
            "-e" => o.expr = Some(it.next()?),
            "--heap-limit" => o.heap_limit = size_arg(&it.next()?)?,
            "--max-depth" => o.max_depth = Some(it.next()?.parse().ok()?),
            "--steps" => o.steps = Some(it.next()?.parse().ok()?),
            "--bench" => o.bench = it.next()?.parse().ok()?,
            "--stack" => o.stack = Some(size_arg(&it.next()?)?),
            "--stack-budget" => o.stack_budget = Some(size_arg(&it.next()?)?),
            "--max-nest" => o.max_nest = Some(it.next()?.parse().ok()?),
            "--fail-every" => o.fail_every = it.next()?.parse().ok()?,
            s if s.starts_with('-') && s.len() > 1 && o.file.is_none() => return None,
            _ => o.file = Some(a),
        }
    }
    Some(o)
}

/// The native stack of the evaluating thread unless --stack says otherwise.
const STACK: usize = 64 * 1024 * 1024;

fn runtime(o: &Opts, console: Console, gpu: &gpu::Session) -> Runtime<HostPlatform> {
    // The interpreter gets the limits a board would compute from its task stack
    // (--stack-budget), or the host defaults: max_depth 512, and a stack limit that leaves a
    // quarter of the thread's stack to whatever runs at the bottom of the recursion.
    let base = match o.stack_budget {
        Some(b) => Config::for_stack(b),
        None => {
            let stack = o.stack.unwrap_or(STACK);
            Config { stack_limit: Some(stack - stack / 4), ..Config::default() }
        }
    };
    let mut cfg = Config { strict: o.strict, step_limit: o.steps, mem_probe: Some(mem_pressed), ..base };
    if let Some(d) = o.max_depth {
        cfg.max_depth = d;
    }
    if let Some(n) = o.max_nest {
        cfg.max_nest = n;
    }
    let mut platform = HostPlatform::new(console);
    // Bounded or fault-injected sessions charge memory to foreground work.
    if o.heap_limit == 0 && o.fail_every == 0 && std::env::var("KLONG_SQL_IDLE").as_deref() != Ok("0") {
        platform.enable_sql_idle(std::env::var("KLONG_SQL_IDLE_TRACE").as_deref() == Ok("1"));
    }
    if cfg!(all(target_os = "macos", feature = "metal")) { platform.features.push("metal"); }
    if o.memfs {
        // an empty file system in memory, at most 256 KiB: the real files are out of reach
        let mut fs = klong_rt::memfs::MemFs::new();
        fs.capacity = 256 * 1024;
        platform.fs = klong_host::Files::Mem(fs);
    }
    let mut rt = Runtime::new(platform, cfg, Some(mem_stats));
    gpu.install(&mut rt.interp);
    rt.interp.kgpy_only = Some(kgpy_only::is);
    let path = klong_host::load_path_env();
    // (the file system in memory has no directory called `.`: its files are named as they are, so the current directory is the empty name)
    rt.set_load_path(klong_host::parse_load_path(&path).map(|d| if o.memfs && d == "." { String::new() } else { d }).collect());
    // .a: the arguments of the program
    let mut args = Vec::new();
    for a in &o.args {
        if let Ok(v) = klong_core::Value::text(a) {
            args.push(v);
        }
    }
    if let Ok(v) = klong_core::Value::from_items(args) {
        let _ = rt.interp.define_global(".a", v);
    }
    rt
}

/// The printed form of a result, or the error that printing it is (out of memory, a
/// dictionary that holds itself).
fn show(rt: &Runtime<HostPlatform>, v: &klong_core::Value) -> Result<String, Error> {
    klong_core::fmt::format_value(&rt.interp.syms, &v.unhost("result")?, false)
}

fn report(e: &Error) {
    eprintln!("{}", e.render());
}

fn stats(rt: &Runtime<HostPlatform>) {
    eprintln!("heap: peak {} bytes, in use {} bytes, limit {}, refused {}", HEAP.peak(), HEAP.used(), HEAP.limit(), HEAP.failed());
    eprintln!("depth: {} nested evaluations at most", rt.interp.max_depth_seen);
    eprintln!("stack: {} bytes of native stack at most", rt.interp.max_stack_seen);
}

fn run(o: Opts) -> ExitCode {
    let gpu = match gpu::Session::open(o.device.as_deref()) {
        Ok(gpu) => gpu,
        Err(e) => {
            eprintln!("kg: {}", e);
            return ExitCode::from(1);
        }
    };
    let code = run_with_device(&o, &gpu);
    if let Err(e) = gpu.write_trace() {
        eprintln!("kg: KLONG_GPU_TRACE: {}", e);
        return ExitCode::from(1);
    }
    code
}

fn run_with_device(o: &Opts, gpu: &gpu::Session) -> ExitCode {
    if let Some(path) = &o.dump_ast {
        let text = match std::fs::read(path).map(String::from_utf8) {
            Ok(Ok(t)) => t,
            _ => {
                println!("error: system");
                return ExitCode::from(1);
            }
        };
        let mut rt = runtime(o, Console::Stderr, gpu);
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
        let mut rt = runtime(o, Console::Stderr, gpu);
        // the limits apply to the program, not to starting up
        HEAP.reset_peak();
        HEAP.set_limit(if o.heap_limit == 0 { 0 } else { HEAP.used() + o.heap_limit }, RESERVE);
        HEAP.fail_every(o.fail_every);
        let r = rt.run(&text, "<stdin>");
        HEAP.fail_every(0);
        HEAP.set_limit(0, 0);
        let code = match r.and_then(|v| v.map(|v| show(&rt, &v)).transpose()) {
            Ok(Some(s)) => {
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

    let mut rt = runtime(o, Console::Stdout, gpu);
    if o.capabilities {
        return match rt.run(".p(.json(.ver()))", "<capabilities>") {
            Ok(_) => ExitCode::SUCCESS,
            Err(error) => {
                report(&error);
                ExitCode::from(1)
            }
        };
    }
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
        let code = match r.and_then(|v| v.map(|v| show(&rt, &v)).transpose()) {
            Ok(Some(s)) => {
                println!("{}", s);
                ExitCode::SUCCESS
            }
            Ok(None) => ExitCode::SUCCESS,
            Err(e) => {
                report(&e);
                ExitCode::from(1)
            }
        };
        // timers, queued calls, a server: keep going while there is something to wait for
        let _ = std::io::stdout().flush();
        run_desktop_events(&mut rt);
        if o.stats {
            stats(&rt);
        }
        return code;
    }

    if let Some(path) = &o.file {
        let text = match std::fs::read_to_string(path) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("kg: {}: {}", path, e);
                return ExitCode::from(1);
            }
        };
        rt.set_program(path);
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
        let _ = std::io::stdout().flush();
        run_desktop_events(&mut rt);
        if o.stats {
            stats(&rt);
        }
        return code;
    }

    // read-eval-print loop: one line, one program
    // (lines are read by a thread so that timers and IPC are served while nothing is typed)
    let (tx, lines) = std::sync::mpsc::channel::<String>();
    std::thread::spawn(move || {
        let stdin = std::io::stdin();
        loop {
            let mut line = String::new();
            match stdin.lock().read_line(&mut line) {
                Ok(0) | Err(_) => break,
                Ok(_) => {}
            }
            if tx.send(line).is_err() {
                break;
            }
        }
    });
    loop {
        print!("  ");
        let _ = std::io::stdout().flush();
        let line = loop {
            if rt.poll_events() {
                run_sql_idle(&rt, true);
                continue;
            }
            // A ready line should wait for real events, not for the rest of a
            // series of optional SQL publication slices.
            match lines.try_recv() {
                Ok(line) => break Some(line),
                Err(std::sync::mpsc::TryRecvError::Disconnected) => break None,
                Err(std::sync::mpsc::TryRecvError::Empty) => {}
            }
            if run_sql_idle(&rt, false) {
                continue;
            }
            let nap = if rt.events_active() { 1 } else { 50 };
            match lines.recv_timeout(std::time::Duration::from_millis(nap)) {
                Ok(l) => break Some(l),
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break None,
            }
        };
        let Some(line) = line else { break };
        let src = line.trim_end_matches(['\n', '\r']);
        if src == "]q" {
            break;
        }
        let evaluated = rt.run(src, "<repl>");
        run_sql_idle(&rt, true);
        match evaluated.and_then(|v| v.map(|v| show(&rt, &v)).transpose()) {
            Ok(Some(s)) => println!("{}", s),
            Ok(None) => {}
            Err(e) => report(&e),
        }
    }
    ExitCode::SUCCESS
}

fn poll_desktop_events(rt: &mut Runtime<HostPlatform>) -> bool {
    let event = rt.poll_events();
    let idle = run_sql_idle(rt, event);
    event || idle
}

fn run_sql_idle(rt: &Runtime<HostPlatform>, activity: bool) -> bool {
    rt.shared.platform.borrow_mut().run_sql_idle(&rt.interp, activity)
}

fn run_desktop_events(rt: &mut Runtime<HostPlatform>) {
    while rt.events_active() {
        if !poll_desktop_events(rt) {
            rt.events_wait();
        }
    }
}

pub fn main() -> ExitCode {
    if std::env::args().nth(1).as_deref() == Some("check") {
        return check::main();
    }
    if std::env::args().nth(1).as_deref() == Some("compile") && std::env::args().nth(2).as_deref() == Some("c") {
        return compile::main("compile");
    }
    if std::env::args().nth(1).as_deref() == Some("fpga") && std::env::args().nth(2).as_deref() == Some("source") {
        return compile::main("fpga");
    }
    if std::env::args().nth(1).as_deref() == Some("profile-doc") {
        return profile_doc::main();
    }
    if std::env::args().nth(1).as_deref() == Some("lint") {
        return lint::main();
    }
    if std::env::args().nth(1).as_deref() == Some("ast") {
        let args: Vec<String> = std::env::args().skip(2).collect();
        let (out, err) = (std::io::stdout(), std::io::stderr());
        let (mut o, mut e) = (std::io::BufWriter::new(out.lock()), err.lock());
        let status = klong_tools::ast::ast_main(&args, &mut std::io::stdin().lock(), &mut o, &mut e);
        let _ = o.flush();
        return ExitCode::from(status as u8);
    }
    let o = match parse_args() {
        Some(o) => o,
        None => return usage(),
    };
    // Finding an allocation that is not fallible: build with RUSTFLAGS=-Zoom=panic (nightly)
    // and set KLONG_OOM_TRACE; the hook lifts the limits so that the trace can be printed.
    if std::env::var_os("KLONG_OOM_TRACE").is_some() {
        std::panic::set_hook(Box::new(|info| {
            HEAP.fail_every(0);
            HEAP.set_limit(0, 0);
            eprintln!("{}\n{}", info, std::backtrace::Backtrace::force_capture());
        }));
    }
    // The evaluator recurses on the native stack and bounds it with Config::max_depth. The
    // main thread's stack is whatever the system gives; a thread with a stack of known size
    // makes the bound hold.
    let stack = o.stack.unwrap_or(STACK);
    match std::thread::Builder::new().stack_size(stack).spawn(move || run(o)) {
        Ok(h) => h.join().unwrap_or(ExitCode::from(70)),
        Err(_) => ExitCode::from(70),
    }
}
