//! The C ABI, tested from C: compiles `tests/host.c` with the system C compiler against the
//! host build of the static library and runs it. The program checks a console transcript
//! (banner, `1+2`, an upload with `^A put`, `^A run`, `^A ver`, the stack and heap limits,
//! a timer, `.x`) through the functions and callbacks of `include/klong.h`.
use std::path::PathBuf;
use std::process::Command;

#[test]
fn c_program_drives_the_console() {
    let here = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    // The test program is target/<profile>/deps/c_abi-<hash>; cargo has put the static
    // library next to it as libklong_ffi-<hash>.a (the newest, if old builds linger).
    let exe = std::env::current_exe().expect("test path");
    let deps = exe.parent().expect("deps directory");
    let lib = std::fs::read_dir(deps)
        .expect("deps directory is readable")
        .filter_map(|e| e.ok())
        .filter(|e| {
            let name = e.file_name().to_string_lossy().into_owned();
            name.starts_with("libklong_ffi-") && name.ends_with(".a")
        })
        .max_by_key(|e| e.metadata().and_then(|m| m.modified()).ok())
        .map(|e| e.path())
        .expect("libklong_ffi-*.a beside the test: cargo builds it with this test");
    let out = deps.join("klong-ffi-host");
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".into());
    let status = Command::new(&cc)
        .args(["-std=c11", "-Wall", "-Wextra", "-Werror", "-O1", "-I"])
        .arg(here.join("include"))
        .arg(here.join("tests/host.c"))
        .arg(&lib)
        .arg("-o")
        .arg(&out)
        .status()
        .expect("the C compiler runs");
    assert!(status.success(), "compiling tests/host.c failed");
    let run = Command::new(&out).output().expect("the C program runs");
    let text = String::from_utf8_lossy(&run.stdout);
    println!("{text}");
    assert!(run.status.success(), "the C program failed:\n{text}\n{}", String::from_utf8_lossy(&run.stderr));
    assert!(text.contains("<< 3\n") && text.trim_end().ends_with("-- ok"), "{text}");
}

/// The lane part of the interface (feature `lane`): `tests/lane.c`. Run with
/// `cargo test -p klong-ffi --features lane`.
#[cfg(feature = "lane")]
#[test]
fn c_program_drives_the_lane() {
    let here = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let exe = std::env::current_exe().expect("test path");
    let deps = exe.parent().expect("deps directory");
    let lib = std::fs::read_dir(deps)
        .expect("deps directory is readable")
        .filter_map(|e| e.ok())
        .filter(|e| {
            let name = e.file_name().to_string_lossy().into_owned();
            name.starts_with("libklong_ffi-") && name.ends_with(".a")
        })
        // A build without the feature leaves its own, possibly newer, archive
        // in the same directory: take the newest one that has the lane symbols.
        .filter(|e| {
            std::fs::read(e.path()).map(|b| b.windows(15).any(|w| w == b"klong_lane_init")).unwrap_or(false)
        })
        .max_by_key(|e| e.metadata().and_then(|m| m.modified()).ok())
        .map(|e| e.path())
        .expect("libklong_ffi-*.a with the lane feature beside the test");
    let out = deps.join("klong-ffi-lane");
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".into());
    let status = Command::new(&cc)
        .args(["-std=c11", "-Wall", "-Wextra", "-Werror", "-O1", "-I"])
        .arg(here.join("include"))
        .arg(here.join("tests/lane.c"))
        .arg(&lib)
        .arg("-o")
        .arg(&out)
        .status()
        .expect("the C compiler runs");
    assert!(status.success(), "compiling tests/lane.c failed");
    let run = Command::new(&out).output().expect("the C program runs");
    let text = String::from_utf8_lossy(&run.stdout);
    println!("{text}");
    assert!(run.status.success(), "the C program failed:\n{text}\n{}", String::from_utf8_lossy(&run.stderr));
    assert!(text.trim_end().ends_with("-- ok"), "{text}");
    // the same program with the boot self-test on a node numbered 0: it serves and reports
    let run = Command::new(&out).arg("selftest").output().expect("the C program runs");
    let text = String::from_utf8_lossy(&run.stdout);
    println!("{text}");
    assert!(run.status.success(), "the self-test run failed:\n{text}\n{}", String::from_utf8_lossy(&run.stderr));
    assert!(text.trim_end().ends_with("-- ok"), "{text}");
}

/// The Pico bench firmware's heap (`boards/pico-bench/klong_heap.h`, the source `main.c` compiles) over a model of
/// newlib's malloc with an arena of the image's size: the largest `!n`, the error at the limit and the memory
/// coming back, the slabs of a deep recursion going to a vector. `tests/pico_heap.c`.
#[test]
fn pico_firmware_heap_on_the_host() {
    let here = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let exe = std::env::current_exe().expect("test path");
    let deps = exe.parent().expect("deps directory");
    let lib = std::fs::read_dir(deps)
        .expect("deps directory is readable")
        .filter_map(|e| e.ok())
        .filter(|e| {
            let name = e.file_name().to_string_lossy().into_owned();
            name.starts_with("libklong_ffi-") && name.ends_with(".a")
        })
        // other builds of the library (without the diagnostic, say) lie in the same directory, and the newest is not always this one:
        // take the newest that has what this program links against
        .filter(|e| std::fs::read(e.path()).map(|b| b.windows(14).any(|w| w == b"klong_set_diag") && b.windows(15).any(|w| w == b"KLONG_ALLOC_TAG")).unwrap_or(false))
        .max_by_key(|e| e.metadata().and_then(|m| m.modified()).ok())
        .map(|e| e.path())
        .expect("libklong_ffi-*.a with the heap diagnostic beside the test");
    let out = deps.join("klong-ffi-pico-heap");
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".into());
    let board = here.join("../boards/pico-bench");
    let status = Command::new(&cc)
        .args(["-std=c11", "-Wall", "-Wextra", "-Werror", "-O1", "-I"])
        .arg(here.join("include"))
        .arg("-I")
        .arg(&board)
        .arg("-I")
        .arg(here.join("tests"))
        .arg(here.join("tests/pico_heap.c"))
        .arg(&lib)
        .arg("-o")
        .arg(&out)
        .status()
        .expect("the C compiler runs");
    assert!(status.success(), "compiling tests/pico_heap.c failed");
    let run = Command::new(&out).output().expect("the C program runs");
    let text = String::from_utf8_lossy(&run.stdout);
    println!("{text}");
    assert!(run.status.success(), "the C program failed:\n{text}\n{}", String::from_utf8_lossy(&run.stderr));
    assert!(text.trim_end().ends_with("-- ok"), "{text}");
}
