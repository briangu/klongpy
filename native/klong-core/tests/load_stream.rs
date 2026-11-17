//! A load is statement by statement: what it holds at once is one statement, and what it keeps (the function bodies). The peak of
//! loading the management library, measured with a counting allocator (so in this binary there is one test that measures), and what a
//! file with an error in it does.
use klong_core::{Config, Interp, NoHooks};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

struct Counting;
static USED: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let u = USED.fetch_add(l.size(), Relaxed) + l.size();
        if u > PEAK.load(Relaxed) {
            PEAK.store(u, Relaxed);
        }
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        USED.fetch_sub(l.size(), Relaxed);
        unsafe { System.dealloc(p, l) }
    }
}
#[global_allocator]
static A: Counting = Counting;

/// The counters are the process's: the tests take turns.
static TURN: std::sync::Mutex<()> = std::sync::Mutex::new(());

const MGMT: &str = include_str!("../../../klongpy/lib/mgmt.kg");

fn val(ip: &mut Interp, s: &str) -> String {
    let v = ip.run(s, "<t>").unwrap().unwrap();
    ip.format(&v)
}

fn interp() -> Interp {
    Interp::new(Config::default(), Box::new(NoHooks))
}

/// (peak above the start, what stays) of `f`.
fn measure(f: impl FnOnce(&mut Interp)) -> (usize, usize) {
    let mut ip = interp();
    ip.run("1", "<t>").unwrap();
    ip.release_caches();
    let base = USED.load(Relaxed);
    PEAK.store(base, Relaxed);
    f(&mut ip);
    ip.release_caches();
    let r = (PEAK.load(Relaxed) - base, USED.load(Relaxed).saturating_sub(base));
    drop(ip);
    r
}

#[test]
fn loading_the_management_library_peaks_at_about_what_it_keeps() {
    let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
    let (whole_peak, whole_kept) = measure(|ip| {
        ip.run(MGMT, "mgmt").unwrap();
    });
    let (peak, kept) = measure(|ip| {
        ip.load_static(MGMT, "mgmt").unwrap();
    });
    println!("whole file: peak {whole_peak} kept {whole_kept}; statement by statement: peak {peak} kept {kept} (64-bit host)");
    // the old way held the tokens of the whole file, then its whole tree, then the text: three and more times what stays
    assert!(whole_peak > 2 * whole_kept, "{whole_peak} {whole_kept}");
    // the new way holds what stays and one statement
    assert!(peak < kept + kept / 4, "peak {peak} against {kept} kept");
    assert!(peak * 5 < whole_peak * 2, "peak {peak} against {whole_peak} for the whole file");
    // nothing of the text is kept in memory: a library of the image is read in place
    assert!(kept < whole_kept, "{kept} {whole_kept}");
}

#[test]
fn a_file_with_an_error_in_its_middle_reports_its_line_and_what_was_done_before_it() {
    let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
    // a function longer than the window the parser reads a statement from (256 bytes: it is doubled until the statement fits), then
    // the error, so that the line and column are those of the file whatever window the statement was parsed from
    let long = (0..40).map(|i| format!("  r{i}::x+{i}")).collect::<Vec<_>>().join(";\n");
    let text = format!("a::1\nf::{{[r0];\n{long};r0}}\nb::(2+3\nc::3\n");
    let line_of_error = text.lines().position(|l| l.starts_with("b::")).unwrap() + 1;
    // .l: the load undoes what it did when it fails (as KlongPy, which parses the whole file first, defines nothing)
    let mut ip = interp();
    let e = match ip.load(&text, "t.kg") {
        Err(e) => e,
        Ok(_) => panic!("a syntax error is an error"),
    };
    let shown = e.render();
    assert!(shown.starts_with(&format!("t.kg:{line_of_error}:")), "{shown}");
    assert!(shown.contains("b::(2+3"), "{shown}");
    assert_eq!(val(&mut ip, "a"), ":a", "a failed load leaves nothing defined");
    // run, whole file first: the same, nothing runs
    let mut ip = interp();
    assert!(ip.run(&text, "t.kg").is_err());
    assert_eq!(val(&mut ip, "a"), ":a");
    // run, statement by statement (the console's `run`): what came before the error has run, as typed at the prompt; the error's line is the same
    let mut ip = interp();
    let e = match ip.run_streamed_owned(text.clone(), "t.kg") {
        Err(e) => e,
        Ok(_) => panic!("an error"),
    };
    assert!(e.render().starts_with(&format!("t.kg:{line_of_error}:")), "{}", e.render());
    assert_eq!(val(&mut ip, "a"), "1");
    assert_eq!(val(&mut ip, "f(1)"), "1");
    // an error when a statement runs, after a long one: the same line
    let text = format!("a::1\nf::{{[r0];\n{long};r0}}\nb::1+\"x\"\nc::3\n");
    let mut ip = interp();
    let e = match ip.load(&text, "t.kg") {
        Err(e) => e,
        Ok(_) => panic!("an error"),
    };
    assert!(e.render().starts_with(&format!("t.kg:{line_of_error}:")), "{}", e.render());
}
